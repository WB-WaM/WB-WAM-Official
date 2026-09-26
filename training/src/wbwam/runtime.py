import logging
import math
import os
from pathlib import Path

from einops import repeat
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import torch

from .trainer import Trainer
from .utils import misc
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import (
    ensure_distributed_initialized,
    resolve_distributed_topology,
    set_global_seed,
)
from .utils.video_io import save_mp4

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def create_wbwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    compile_training_denoise: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wbwam import WBWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for WBWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return WBWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        compile_training_denoise=bool(compile_training_denoise),
    )


def create_wbwam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    compile_training_denoise: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wbwam_joint import WBWAMJoint

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for WBWAMJoint.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return WBWAMJoint.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        compile_training_denoise=bool(compile_training_denoise),
    )


def _create_wbwam_idm_from_cls(
    model_cls,
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    compile_training_denoise: bool = False,
    action_idm_prob: float | None = None,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for WBWAMIDM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    model = model_cls.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        compile_training_denoise=bool(compile_training_denoise),
    )
    if action_idm_prob is not None:
        prob = float(action_idm_prob)
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"`action_idm_prob` must be in [0, 1], got {prob}.")
        model.action_idm_prob = prob
    return model


def create_wbwam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    compile_training_denoise: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wbwam_idm import WBWAMIDM

    return _create_wbwam_idm_from_cls(
        WBWAMIDM,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        video_dit_config=video_dit_config,
        tokenizer_max_len=tokenizer_max_len,
        load_text_encoder=load_text_encoder,
        proprio_dim=proprio_dim,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        loss=loss,
        mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        compile_training_denoise=compile_training_denoise,
        redirect_common_files=redirect_common_files,
        model_dtype=model_dtype,
        device=device,
    )


def create_wbwam_optional_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    action_idm_prob: float,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    compile_training_denoise: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wbwam_optional_idm import WBWAMOptionalIDM

    return _create_wbwam_idm_from_cls(
        WBWAMOptionalIDM,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        video_dit_config=video_dit_config,
        tokenizer_max_len=tokenizer_max_len,
        load_text_encoder=load_text_encoder,
        proprio_dim=proprio_dim,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        loss=loss,
        mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        compile_training_denoise=compile_training_denoise,
        action_idm_prob=action_idm_prob,
        redirect_common_files=redirect_common_files,
        model_dtype=model_dtype,
        device=device,
    )


def _build_legacy_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _build_flat_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg, is_training_set=True)

    raw_val_set_proportion = data_cfg.get("val_set_proportion")
    if raw_val_set_proportion is None:
        raw_val_set_proportion = getattr(train_ds, "val_set_proportion", None)
    val_set_proportion = 0.0 if raw_val_set_proportion is None else float(raw_val_set_proportion)
    if val_set_proportion <= 0.0:
        logger.info(
            "Validation disabled because val_set_proportion=%.6f; using all eligible data for training.",
            val_set_proportion,
        )
        return train_ds, None
    default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
    pretrained_norm_stats = data_cfg.get("pretrained_norm_stats") or default_stats_path
    logger.info(
        "Building val dataset with val_set_proportion=%.6f and pretrained_norm_stats=%s",
        val_set_proportion,
        pretrained_norm_stats,
    )
    val_ds = instantiate(
        data_cfg,
        is_training_set=False,
        pretrained_norm_stats=pretrained_norm_stats,
    )
    return train_ds, val_ds


def build_datasets(data_cfg: DictConfig):
    if data_cfg.get("_target_") is None and data_cfg.get("train") is not None:
        return _build_legacy_datasets(data_cfg)
    return _build_flat_datasets(data_cfg)


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def _prepare_fk_data_config(cfg: DictConfig):
    """Resolve the opt-in before saving config or building any data workers."""
    config = cfg.get("fk_link_loss")
    if config is None:
        if OmegaConf.select(cfg, "data.return_fk_targets", default=False):
            raise ValueError("return_fk_targets requires a fk_link_loss config")
        return None
    options = OmegaConf.to_container(config, resolve=True)
    weight = float(options.get("weight", 0.05))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("FK weight must be finite and non-negative")
    enabled = bool(options.pop("enabled", False)) and weight > 0
    if enabled and OmegaConf.select(cfg, "data._target_") != "wbwam.datasets.wb_dataset.WBMixtureDataset":
        raise ValueError("FK requires the flat WBMixtureDataset config")
    if enabled or OmegaConf.select(cfg, "data.return_fk_targets") is not None:
        OmegaConf.update(cfg, "data.return_fk_targets", enabled, force_add=True)
    return options if enabled else None


def _attach_fk_link_loss(model, train_dataset, val_dataset, options):
    if options is None:
        return
    from .models.wan22.wbwam import WBWAM
    from .models.wan22.g1_fk_loss import G1LinkPositionLoss

    if not isinstance(model, WBWAM):
        raise ValueError("FK link loss is only supported by WBWAM and its IDM variants")
    auxiliary = G1LinkPositionLoss.from_dataset(train_dataset, **options)
    if val_dataset is not None and val_dataset is not train_dataset:
        val_auxiliary = G1LinkPositionLoss.from_dataset(val_dataset, **options)
        for name in ("scale", "offset", "model_indices"):
            if not torch.equal(getattr(auxiliary, name), getattr(val_auxiliary, name)):
                raise ValueError(f"FK train/validation {name} mismatch")
    model.fk_link_loss = auxiliary.to(device=model.device)
    logger.info(
        "Enabled pelvis-local G1 FK: weight=%s groups=%s scale_m=%s orientation_weight=%s "
        "scale_rad=%s urdf=%s sha256=%s",
        auxiliary.weight,
        auxiliary.group_weights,
        auxiliary.position_scale,
        auxiliary.orientation_weight,
        auxiliary.orientation_scale,
        auxiliary.fk.urdf_path,
        auxiliary.fk.urdf_sha256,
    )


def run_training(cfg: DictConfig):
    try:
        # Resolve topology before Accelerator exists because dataset sharding
        # must happen before dataset construction.
        topology = resolve_distributed_topology()

        ensure_distributed_initialized(topology)
        setup_logging(
            log_level=logging.INFO,
            is_main_process=topology.rank == 0,
        )
        misc.register_work_dir(cfg.output_dir)
        fk_options = _prepare_fk_data_config(cfg)
        # Materialize ${oc.load:...} nodes before writing the resolved config.
        OmegaConf.resolve(cfg)
        config_payload = OmegaConf.to_container(cfg, resolve=True)
        with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
            OmegaConf.save(config_payload, f)

        model_device = _resolve_train_device()
        mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
        model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
        set_global_seed(int(cfg.seed))
        logger.info("Instantiating model: device=%s dtype=%s", model_device, model_dtype)
        model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
        logger.info("Model instantiated.")
        logger.info("Building datasets...")
        train_ds, val_ds = build_datasets(cfg.data)
        _attach_fk_link_loss(model, train_ds, val_ds, fk_options)
        sharding_metadata = dict(getattr(train_ds, "sharding_metadata", {"enabled": False}))
        sharding_metadata.setdefault("enabled", False)
        if bool(sharding_metadata["enabled"]):
            datasets = [("train", train_ds)]
            if val_ds is not None:
                datasets.append(("val", val_ds))
            for dataset_name, dataset in datasets:
                if not hasattr(dataset, "align_for_distributed_sharding"):
                    raise TypeError(
                        "Dataset sharding is enabled, but "
                        f"{dataset_name} dataset does not implement align_for_distributed_sharding()."
                    )
                target_len = dataset.align_for_distributed_sharding()
                logger.info(
                    "[dataset sharding] Aligned %s dataset length: %d",
                    dataset_name,
                    target_len,
                )
        if val_ds is None:
            logger.info("Dataset built: train=%d validation=disabled", len(train_ds))
        else:
            logger.info("Datasets built: train=%d val=%d", len(train_ds), len(val_ds))
        logger.info("Creating Trainer...")
        trainer = Trainer(
            cfg=cfg,
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            sharding_metadata=sharding_metadata,
        )
        logger.info("Trainer ready. Starting training loop.")
        trainer.train()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run_inference(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()

    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
