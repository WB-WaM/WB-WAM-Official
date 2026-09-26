from collections import deque
import json
from math import ceil
import os
from pathlib import Path
import re
import time

from accelerate import Accelerator
from accelerate.utils import send_to_device
from omegaconf import DictConfig, OmegaConf
import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableDistributedSampler
from .utils.train_hooks import TrainHookManager
from .utils.training_dynamics import TrainingDynamicsMonitor

logger = get_logger(__name__)


def _configure_deepspeed_gradient_clipping(deepspeed_plugin, *, max_grad_norm: float) -> None:
    if deepspeed_plugin is None:
        return

    # Accelerator.clip_grad_norm_ only reports the global norm in DeepSpeed mode.
    # DeepSpeed performs the actual clipping during optimizer.step(), so keep its
    # runtime config synchronized with the task-level max_grad_norm setting.
    deepspeed_plugin.deepspeed_config["gradient_clipping"] = max_grad_norm


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        *,
        cfg: DictConfig,
        # Sharding metadata is produced by configure_dataset_sharding().
        # When sharding is disabled, sampler fields are None and Trainer falls
        # back to the normal global Accelerator topology.
        sharding_metadata: dict | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_final = bool(OmegaConf.select(cfg, "save_final", default=True))
        self.eval_every = int(cfg.eval_every)
        self.eval_at_start = bool(OmegaConf.select(cfg, "eval_at_start", default=False))
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)

        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        if deepspeed_plugin is not None and deepspeed_plugin.is_auto("train_micro_batch_size_per_gpu"):
            # The train dataloader intentionally stays outside accelerator.prepare().
            # DeepSpeed can no longer infer the micro batch size from a dataloader,
            # so set it explicitly from the task config batch_size.
            deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = self.batch_size
            logger.info(
                "Set DeepSpeed train_micro_batch_size_per_gpu=%d because dataloader "
                "is not passed to accelerator.prepare().",
                self.batch_size,
            )
        if deepspeed_plugin is not None:
            _configure_deepspeed_gradient_clipping(deepspeed_plugin, max_grad_norm=self.max_grad_norm)
            logger.info(
                "Set DeepSpeed gradient_clipping=%.4f from task max_grad_norm.",
                self.max_grad_norm,
            )
        zero_stage = (
            deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
            if deepspeed_plugin is not None
            else "none"
        )

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d "
            "process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s "
            "grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        self.sharding_metadata = dict(sharding_metadata or {})
        self.sharding_metadata.setdefault("enabled", False)
        self.sharding_metadata.setdefault("num_nodes", 1)
        self.sharding_metadata.setdefault("shard_group_size", 1)
        self.sharding_metadata.setdefault("num_bins", 1)
        # sampler_num_replicas/sampler_rank are the sampler-local topology.
        # With dataset sharding, this is the shard group topology, not global
        # Accelerator world size/rank.
        metadata_num_replicas = self.sharding_metadata.get("sampler_num_replicas")
        metadata_rank = self.sharding_metadata.get("sampler_rank")
        self.sampler_num_replicas = int(
            metadata_num_replicas if metadata_num_replicas is not None else self.accelerator.num_processes
        )
        self.sampler_rank = int(metadata_rank if metadata_rank is not None else self.accelerator.process_index)
        logger.info(
            "Sampler topology: num_replicas=%d rank=%d shard_datasets_by_node=%s drop_last=True",
            self.sampler_num_replicas,
            self.sampler_rank,
            bool(self.sharding_metadata["enabled"]),
        )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # ZeRO creates FP32 master parameters during prepare; load warm-start weights first.
        self._load_weight_checkpoint()
        param_groups = self._build_param_groups()
        if self.accelerator.is_main_process:
            total_trainable_params = sum(
                param.numel() for group in param_groups for param in group["params"] if param.requires_grad
            )
            logger.info("Total trainable params: %.4fB", total_trainable_params / 1e9)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
        )

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        self.train_hooks = TrainHookManager(log_every=self.log_every)
        dynamics_cfg = OmegaConf.select(self.cfg, "training_dynamics")
        self.training_dynamics = None
        if dynamics_cfg is not None and bool(OmegaConf.select(dynamics_cfg, "enabled", default=False)):
            self.training_dynamics = TrainingDynamicsMonitor(
                dynamics_cfg,
                default_interval=max(self.log_every, 1),
            )
            self.training_dynamics.install(self.accelerator.unwrap_model(self.model))
            self.train_hooks.append(self.training_dynamics)
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._resume_training_state()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _build_param_groups(self):
        if hasattr(self.model, "get_param_group"):
            return self.model.get_param_group(
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        if hasattr(self.model, "setup_dit_only_train_mode"):
            self.model.setup_dit_only_train_mode()
        elif hasattr(self.model, "set_train_mode_for_training"):
            self.model.set_train_mode_for_training()

        trainable_params = []
        dit = getattr(self.model, "dit", None)
        if dit is not None:
            trainable_params.extend(list(dit.parameters()))
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        if not trainable_params:
            trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError(f"{self.model.__class__.__name__} has no trainable parameters.")

        return [
            {
                "params": trainable_params,
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            }
        ]

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableDistributedSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_replicas=self.sampler_num_replicas,
            rank=self.sampler_rank,
            drop_last=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            prefetch_factor=(None if self.num_workers == 0 else 4),
            persistent_workers=self.num_workers > 0,
        )

    def _set_train_dataset_epoch(self, epoch: int) -> None:
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}")

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        micro_steps_per_epoch = max(len(self.train_loader), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _estimate_eta(self):
        start_step, start_time = self.speed_window[0]
        end_step, end_time = self.speed_window[-1]
        elapsed = max(end_time - start_time, 1e-6)
        done_steps = max(end_step - start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _load_weight_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint before optimizer initialization: %s", resume)
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        logger.info("Warm start loaded; optimizer/scheduler/step will start fresh.")

    def _resume_training_state(self):
        if self.resume and Path(str(self.resume)).is_dir():
            logger.info("Resuming full training state from directory: %s", self.resume)
            self.load_training_state(str(self.resume))

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(
                f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}"
            )

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")

        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(f"`sample['action']` must be a torch.Tensor, got {type(action)}")
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(
                    "`sample['action']` temporal dimension must be divisible by "
                    f"video frames-1={num_video_frames - 1}, got {action.shape[1]}"
                )
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    "`context/context_mask` must be [B,L,D]/[B,L], got "
                    f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        eval_sample = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "action_horizon": action_horizon,
        }
        for key, unbatched_ndim in (
            ("action_is_pad", 1),
            ("action_dim_is_pad", 2),
            ("action_semantic_dim_is_pad", 2),
            ("image_is_pad", 1),
            ("proprio_dim_is_pad", 2),
            ("fk_joint_target", 2),
            ("fk_joint_offset", 1),
        ):
            value = sample.get(key, None)
            if value is None:
                continue
            value = torch.as_tensor(value)
            if value.ndim == unbatched_ndim:
                value = value.unsqueeze(0)
            expected_ndim = unbatched_ndim + 1
            if value.ndim != expected_ndim:
                raise ValueError(
                    f"`sample[{key!r}]` must be {expected_ndim}D after batching, got shape {tuple(value.shape)}"
                )
            if value.shape[0] != video.shape[0]:
                raise ValueError(
                    f"`sample[{key!r}]` batch mismatch: {value.shape[0]} vs video batch {video.shape[0]}"
                )
            eval_sample[key] = value

        has_video = sample.get("has_video", None)
        if has_video is not None:
            has_video = torch.as_tensor(has_video, dtype=torch.bool)
            if has_video.ndim == 0:
                has_video = has_video.unsqueeze(0)
            if has_video.ndim != 1 or has_video.shape[0] != video.shape[0]:
                raise ValueError(
                    "`sample['has_video']` must be 1D and match the video batch, "
                    f"got shape {tuple(has_video.shape)} vs batch {video.shape[0]}"
                )
            eval_sample["has_video"] = has_video
        if context is not None:
            eval_sample["context"] = context
            eval_sample["context_mask"] = context_mask
        return eval_sample

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])
        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )

        with self.accelerator.autocast():
            local_result = model.evaluate(
                sample=sample,
                num_inference_steps=int(self.eval_num_inference_steps),
                eval_video_path=video_path,
                seed=42,
            )

        video_metric_keys = ("psnr_rg", "ssim_rg", "psnr_rd", "ssim_rd", "psnr_dg", "ssim_dg")
        required_keys = ("val_loss", *video_metric_keys)
        for key in required_keys:
            if key not in local_result:
                raise ValueError(f"Model evaluate result missing required key: `{key}`")

        local_base = torch.tensor(
            [
                float(local_result["val_loss"]),
                *[float(local_result[key]) for key in video_metric_keys],
                float(local_result.get("video_metric_valid", 1.0)),
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_base = self.accelerator.gather_for_metrics(local_base)
        mean_val_loss = gathered_base[:, 0].mean()
        video_metric_weights = gathered_base[:, -1].clamp(0.0, 1.0)
        valid_video_count = video_metric_weights.sum()
        if valid_video_count.item() > 0:
            mean_video_metrics = (gathered_base[:, 1:-1] * video_metric_weights.unsqueeze(1)).sum(
                dim=0
            ) / valid_video_count
        else:
            mean_video_metrics = torch.zeros_like(gathered_base[0, 1:-1])

        # All ranks participate, including samples without action targets.
        local_action = torch.tensor(
            [
                float(local_result.get("action_l2", 0.0)),
                float("action_l2" in local_result),
                float(local_result.get("action_error_sum_valid", 0.0)),
                int(local_result.get("action_valid_count", 0)),
            ],
            device=self.accelerator.device,
            dtype=torch.float64,
        ).unsqueeze(0)
        gathered_action = self.accelerator.gather_for_metrics(local_action)
        action_l2_count = gathered_action[:, 1].sum().item()
        action_l2_mean = gathered_action[:, 0].sum().item() / action_l2_count if action_l2_count else None
        action_valid_count = int(gathered_action[:, 3].sum().item())
        action_mse_valid = (
            gathered_action[:, 2].sum().item() / action_valid_count if action_valid_count else None
        )

        result = {
            "val_loss": float(mean_val_loss.item()),
            "psnr_rg": float(mean_video_metrics[0].item()),
            "ssim_rg": float(mean_video_metrics[1].item()),
            "psnr_rd": float(mean_video_metrics[2].item()),
            "ssim_rd": float(mean_video_metrics[3].item()),
            "psnr_dg": float(mean_video_metrics[4].item()),
            "ssim_dg": float(mean_video_metrics[5].item()),
            "video_metric_valid_count": float(valid_video_count.item()),
            "video_path": str(local_result.get("video_path", video_path)),
            "action_mse_valid": action_mse_valid,
            "action_valid_count": action_valid_count,
            "action_valid_sample_count": int((gathered_action[:, 3] > 0).sum().item()),
        }
        if action_l2_mean is not None:
            result["action_l2"] = action_l2_mean
        return result

    def _evaluate_and_log(self):
        metrics = self.evaluate()
        self.accelerator.wait_for_everyone()
        if metrics is None or not self.accelerator.is_main_process:
            return

        description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
            self.global_step,
            metrics["val_loss"],
            metrics["psnr_rd"],
            metrics["ssim_rd"],
        )
        if "action_l2" in metrics:
            description += " action_l2=%.4f" % metrics["action_l2"]
        if metrics["action_mse_valid"] is not None:
            description += " action_mse_valid=%.4f" % metrics["action_mse_valid"]
        else:
            description += " action_mse_valid=N/A"
        logger.info(description)
        eval_payload = {
            "eval/val_loss": float(metrics["val_loss"]),
            "eval/psnr_rg": float(metrics["psnr_rg"]),
            "eval/ssim_rg": float(metrics["ssim_rg"]),
            "eval/psnr_rd": float(metrics["psnr_rd"]),
            "eval/ssim_rd": float(metrics["ssim_rd"]),
            "eval/psnr_dg": float(metrics["psnr_dg"]),
            "eval/ssim_dg": float(metrics["ssim_dg"]),
        }
        if "action_l2" in metrics:
            eval_payload["eval/action_l2"] = float(metrics["action_l2"])
        eval_payload["eval/action_valid_count"] = metrics["action_valid_count"]
        eval_payload["eval/action_valid_sample_count"] = metrics["action_valid_sample_count"]
        if metrics["action_mse_valid"] is not None:
            eval_payload["eval/action_mse_valid"] = float(metrics["action_mse_valid"])
        self._wandb_log(eval_payload)

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        # `batch_in_epoch` is the number of local dataloader batches already consumed.
        # Checkpoints are saved after optimizer step, so resume must skip these batches.
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "seed": int(self.seed),
            "batch_size": int(self.batch_size),
            # num_replicas is sampler-local. With sharding enabled this is
            # shard_group_size * local_world_size, not accelerator.num_processes.
            "num_replicas": int(self.sampler_num_replicas),
            # sampler_rank is saved for debugging only; resume validation must
            # not compare it because rank0 writes this state for all ranks.
            "sampler_rank": int(self.sampler_rank),
            "dataset_len": int(len(self.train_dataset)),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            # These fields guard against resuming with a different dataset
            # sharding topology.
            "shard_datasets_by_node": int(bool(self.sharding_metadata["enabled"])),
            "num_nodes": int(self.sharding_metadata["num_nodes"]),
            "shard_group_size": int(self.sharding_metadata["shard_group_size"]),
            "num_bins": int(self.sharding_metadata["num_bins"]),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def _validate_trainer_state(self, payload: dict, state_file: Path):
        expected = {
            "seed": int(self.seed),
            "batch_size": int(self.batch_size),
            "num_replicas": int(self.sampler_num_replicas),
            "dataset_len": int(len(self.train_dataset)),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            # Keep resume on the same dataset sharding topology. `sampler_rank`
            # is intentionally not validated because each rank has a different
            # value while only the main process writes trainer_state.json.
            "shard_datasets_by_node": int(bool(self.sharding_metadata["enabled"])),
            "num_nodes": int(self.sharding_metadata["num_nodes"]),
            "shard_group_size": int(self.sharding_metadata["shard_group_size"]),
            "num_bins": int(self.sharding_metadata["num_bins"]),
        }
        for key, expected_value in expected.items():
            if key not in payload:
                raise KeyError(f"`{state_file}` missing required resume field `{key}`.")
            actual_value = int(payload[key])
            if actual_value != expected_value:
                raise ValueError(
                    f"Resume field `{key}` mismatch: checkpoint={actual_value}, current={expected_value}."
                )

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._validate_trainer_state(payload, state_file)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch(self.epoch)
                self._set_train_dataset_epoch(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d local_sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self._set_train_dataset_epoch(self.epoch)
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self._set_train_dataset_epoch(self.epoch)
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        self._set_train_dataset_epoch(self.epoch)
        # DataLoader workers are created by iter(), not by the DataLoader constructor.
        # Start them before wandb initializes background threads to avoid forking a
        # multithreaded rank-0 process. persistent_workers keeps later epochs fork-free.
        data_iter = iter(self.train_loader)
        self._init_wandb()
        if self.eval_at_start and self.val_dataset is not None:
            self._evaluate_and_log()
        self.speed_window = deque([(self.global_step, time.perf_counter())], maxlen=51)
        data_time_total = 0.0
        model_time_total = 0.0
        timing_steps = 0
        import gc

        # Disable GC for more stable performance; call gc.collect() manually at the end.
        # gc.disable()
        while self.global_step < self.max_steps:
            if self.global_step == 1:
                logger.info("First training step completed.")
            data_start = time.perf_counter()
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                self.train_sampler.set_epoch(self.epoch)
                self._set_train_dataset_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue
            data_time_total += time.perf_counter() - data_start

            sample = send_to_device(sample, self.accelerator.device)
            model_start = time.perf_counter()
            with self.accelerator.accumulate(self.model):
                train_model = (
                    self.model
                    if hasattr(self.model, "training_loss")
                    else self.accelerator.unwrap_model(self.model)
                )
                hook_step = self.train_hooks.begin_step(
                    step=self.global_step + 1,
                    sync_gradients=bool(self.accelerator.sync_gradients),
                )

                hook_step.forward_start()
                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                hook_step.forward_end()
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    hook_step.before_gradient_clip(
                        model=unwrapped_model,
                        max_grad_norm=self.max_grad_norm,
                    )
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(self.accelerator.gather(loss.detach().float().reshape(1)).mean().item())
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(
                            float(value), device=loss.device, dtype=torch.float32
                        ).reshape(1)
                        global_loss_metrics[key] = float(self.accelerator.gather(metric_tensor).mean().item())
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                    self.speed_window.append((self.global_step, time.perf_counter()))

                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    hook_step.optimizer_step_end(
                        model=unwrapped_model,
                        optimizer=self.optimizer,
                    )
                    hook_metrics = hook_step.gather_metrics(self.accelerator, device=loss.device)
                    model_time_total += time.perf_counter() - model_start
                    timing_steps += 1

                    if (
                        self.log_every > 0
                        and self.global_step % self.log_every == 0
                        and self.accelerator.is_main_process
                    ):
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += (
                            "lr=%.2e data=%.3fs model=%.3fs speed=%.2f step/s, %.2f samples/s eta=%s"
                            % (
                                current_lr,
                                data_time_total / max(timing_steps, 1),
                                model_time_total / max(timing_steps, 1),
                                steps_per_sec,
                                steps_per_sec * self.batch_size * self.accelerator.num_processes,
                                eta_str,
                            )
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes,
                            "performance/data_time_sec": data_time_total / max(timing_steps, 1),
                            "performance/model_time_sec": model_time_total / max(timing_steps, 1),
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        wandb_payload = {**wandb_payload, **hook_metrics}
                        self._wandb_log(wandb_payload)
                        data_time_total = 0.0
                        model_time_total = 0.0
                        timing_steps = 0

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        self._evaluate_and_log()

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        gc.collect()  # Manually trigger GC after checkpoint to free up memory.

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint() if self.save_final else None
                        if self.accelerator.is_main_process:
                            if ckpt_info is not None:
                                logger.info(
                                    "[done] max_steps reached step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )
                            else:
                                logger.info("[done] max_steps reached step=%d final_save=false", self.global_step)
                        return
                else:
                    model_time_total += time.perf_counter() - model_start

        ckpt_info = self.save_checkpoint() if self.save_final else None
        if self.accelerator.is_main_process:
            if ckpt_info is not None:
                logger.info(
                    "[done] training finished step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
            else:
                logger.info("[done] training finished step=%d final_save=false", self.global_step)
