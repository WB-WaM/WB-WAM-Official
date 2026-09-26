from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from wbwam.utils.logging_config import get_logger
from wbwam.utils.video_io import save_mp4
from wbwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import WanVideoDiT

logger = get_logger(__name__)


class WBWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert: WanVideoDiT,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        compile_training_denoise: bool = False,
    ):
        """Create a world-action model around video/action experts, MoT, VAE, and optional T5.

        Args:
            video_expert: Wan video DiT expert that prepares/posts video latents.
            action_expert: Action DiT expert that prepares/posts action trajectories.
            mot: Mixed transformer that jointly processes video and action tokens.
            vae: Frozen video VAE used for latent encoding/decoding.
            text_encoder: Optional frozen text encoder for on-the-fly prompt embedding.
            tokenizer: Optional tokenizer paired with `text_encoder`.
            text_dim: Text/context feature dimension. Required when no text encoder is loaded.
            proprio_dim: Optional proprioceptive state dimension to append as a context token.
            device: Model device.
            torch_dtype: Main model compute dtype.
            *_shift / *_num_train_timesteps: Continuous flow scheduler settings.
            loss_lambda_video: Scalar multiplier for video loss.
            loss_lambda_action: Scalar multiplier for action loss.
            compile_training_denoise: Whether training denoise core is lazily `torch.compile`d.

        Output:
            Initializes modules, schedulers, dtype/device state, and training freeze policy.
        """
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.compile_training_denoise = bool(compile_training_denoise)
        self._inference_action_valid_dim: Optional[int] = None
        self._inference_action_valid_indices: Optional[tuple[int, ...]] = None
        self._inference_action_invalid_indices: tuple[int, ...] = ()

        self.to(self.device)
        self.setup_dit_only_train_mode()

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        compile_training_denoise: bool = False,
    ):
        """Build `WBWAM` from Wan2.2 components plus an ActionDiT expert.

        Args:
            device: Device used for component construction.
            torch_dtype: Main model dtype.
            model_id: Wan2.2 model identifier or local path.
            tokenizer_model_id: Tokenizer identifier or local path.
            tokenizer_max_len: Tokenizer max sequence length.
            load_text_encoder: Whether to load the frozen T5 text encoder/tokenizer.
            proprio_dim: Optional proprioceptive state dimension.
            redirect_common_files: Whether loader redirects shared Wan files.
            video_dit_config: Video expert config; must include `text_dim`.
            action_dit_config: Action expert config.
            action_dit_pretrained_path: Optional ActionDiT backbone checkpoint path.
            skip_dit_load_from_pretrain: Skip pretrained DiT loading for random init/checkpoint override.
            mot_checkpoint_mixed_attn: Legacy config flag kept in runtime/config surface.
            *_shift / *_num_train_timesteps: Scheduler settings.
            loss_lambda_video: Video loss multiplier.
            loss_lambda_action: Action loss multiplier.
            compile_training_denoise: Whether to compile the trainable denoise core.

        Returns:
            A fully wired `WBWAM` with model path metadata.
        """
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for WBWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for WBWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(mixtures={"video": video_expert, "action": action_expert})

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            compile_training_denoise=compile_training_denoise,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        """Move/cast model submodules and cached helper modules consistently.

        Args:
            *args/**kwargs: Same arguments accepted by `torch.nn.Module.to`.

        Returns:
            `self`.
        """
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def set_inference_action_valid_dim(self, valid_dim: Optional[int]) -> None:
        """Keep right-padded action dimensions on the training-time zero manifold."""

        if valid_dim is None:
            self.set_inference_action_valid_indices(None)
            return
        valid_dim = int(valid_dim)
        model_dim = int(self.action_expert.action_dim)
        if valid_dim <= 0 or valid_dim > model_dim:
            raise ValueError(f"valid action dim must be in [1, {model_dim}], got {valid_dim}")
        self.set_inference_action_valid_indices(range(valid_dim))

    def set_inference_action_valid_indices(self, indices: Optional[Sequence[int]]) -> None:
        """Keep only selected action dimensions on the training-time manifold."""

        if indices is None:
            self._inference_action_valid_dim = None
            self._inference_action_valid_indices = None
            self._inference_action_invalid_indices = ()
            return

        valid_indices = tuple(int(index) for index in indices)
        model_dim = int(self.action_expert.action_dim)
        if not valid_indices:
            raise ValueError("valid action indices must not be empty")
        if len(set(valid_indices)) != len(valid_indices):
            raise ValueError("valid action indices must be unique")
        if any(index < 0 or index >= model_dim for index in valid_indices):
            raise ValueError(f"valid action indices must be in [0, {model_dim - 1}]")

        valid_index_set = set(valid_indices)
        self._inference_action_valid_indices = valid_indices
        self._inference_action_invalid_indices = tuple(
            index for index in range(model_dim) if index not in valid_index_set
        )
        self._inference_action_valid_dim = (
            len(valid_indices) if valid_indices == tuple(range(len(valid_indices))) else None
        )

    def _zero_inference_action_padding(
        self,
        action: torch.Tensor,
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Zero persistent layout padding and optional per-call semantic action dimensions."""

        if action.ndim != 3:
            raise ValueError(f"action latent must be [B,T,D], got {tuple(action.shape)}")
        model_dim = int(self.action_expert.action_dim)
        if action.shape[-1] != model_dim:
            raise ValueError(f"action latent dim must be {model_dim}, got {action.shape[-1]}")

        if self._inference_action_invalid_indices:
            action[..., self._inference_action_invalid_indices] = 0.0
        if action_dim_is_pad is None:
            return action

        dim_mask = torch.as_tensor(action_dim_is_pad, dtype=torch.bool, device=action.device)
        batch_size, action_horizon, _ = action.shape
        if dim_mask.ndim == 1:
            if dim_mask.shape[0] != model_dim:
                raise ValueError(
                    f"1D action dimension mask must be [{model_dim}], got {tuple(dim_mask.shape)}"
                )
            dim_mask = dim_mask.view(1, 1, model_dim)
        elif dim_mask.ndim == 2:
            if tuple(dim_mask.shape) != (action_horizon, model_dim):
                raise ValueError(
                    "2D action dimension mask must match [T,D]: "
                    f"got {tuple(dim_mask.shape)} vs ({action_horizon}, {model_dim})"
                )
            dim_mask = dim_mask.unsqueeze(0)
        elif dim_mask.ndim == 3:
            expected_tail = (action_horizon, model_dim)
            if tuple(dim_mask.shape[1:]) != expected_tail or dim_mask.shape[0] not in (1, batch_size):
                raise ValueError(
                    "3D action dimension mask must be [1,T,D] or [B,T,D]: "
                    f"got {tuple(dim_mask.shape)} for action {tuple(action.shape)}"
                )
        else:
            raise ValueError(
                "action dimension mask must be [D], [T,D], or [B,T,D], "
                f"got {tuple(dim_mask.shape)}"
            )

        action.masked_fill_(dim_mask.expand_as(action), 0.0)
        return action

    def setup_dit_only_train_mode(self):
        """Freeze non-DiT modules and leave MoT/action/video experts trainable.

        Args:
            None.

        Output:
            Mutates module train/eval and `requires_grad` state; returns `None`.
        """
        self.eval()
        self.requires_grad_(False)
        self.dit.train()
        self.dit.requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train()
            self.proprio_encoder.requires_grad_(True)

    def set_train_mode_for_training(self):
        """Restore train mode for modules that participate in optimization.

        Args:
            None.

        Output:
            Sets MoT and optional proprio encoder to train mode; returns `None`.
        """
        self.dit.train()
        if self.proprio_encoder is not None:
            self.proprio_encoder.train()

    def trainable_params(self):
        """Return parameters optimized by the trainer.

        Args:
            None.

        Returns:
            List containing MoT parameters and optional proprio encoder parameters.
        """
        params = list(self.dit.parameters())
        if self.proprio_encoder is not None:
            params.extend(list(self.proprio_encoder.parameters()))
        return params

    def get_param_group(self, lr: float, weight_decay: float):
        """Build optimizer parameter groups with no weight decay for 1D parameters.

        Args:
            lr: Learning rate for all trainable groups.
            weight_decay: Weight decay for non-1D trainable parameters.

        Returns:
            List of optimizer param group dictionaries.
        """
        trainable = [param for param in self.trainable_params() if param.requires_grad]
        if len(trainable) == 0:
            raise ValueError("WBWAM has no trainable parameters.")

        decay_params = []
        no_decay_params = []
        for param in trainable:
            if param.ndim == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = []
        if len(decay_params) > 0:
            param_groups.append(
                {
                    "params": decay_params,
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                }
            )
        if len(no_decay_params) > 0:
            param_groups.append(
                {
                    "params": no_decay_params,
                    "lr": float(lr),
                    "weight_decay": 0.0,
                }
            )
        return param_groups

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        """Round image/video sizes to values supported by the VAE/video expert.

        Args:
            height: Input image height.
            width: Input image width.
            num_frames: Requested video frame count.

        Returns:
            `(height, width, num_frames)` rounded so H/W are multiples of 16 and `T % 4 == 1`.
        """
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        """Tokenize and encode text prompts into frozen T5 context tensors.

        Usage:
            Shared path for on-the-fly text encoding in training and inference when cached context is absent.

        Args:
            prompt: String or batch of strings.

        Returns:
            `(context, context_mask)` where `context` is `[B, L, text_dim]` on `self.device` and
            `context_mask` is `[B, L]`. Padding embeddings are zeroed and the returned mask is all true to
            match existing cross-attention behavior.
        """
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        prompt_emb = prompt_emb.masked_fill(~mask.unsqueeze(-1), 0)
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append proprioception as one encoded context token.

        Usage:
            Shared helper used by training and inference when `proprio_dim` is enabled.

        Args:
            context: Existing context tokens, shaped `[B, L, text_dim]`.
            context_mask: Existing boolean context mask, shaped `[B, L]`.
            proprio: Optional proprio tensor, shaped `[B, proprio_dim]`.

        Returns:
            `(context, context_mask)` with one extra token when proprio encoding is enabled and provided.
        """
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        proprio_token = self.proprio_encoder(proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)).to(
            dtype=context.dtype
        )  # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _validate_infer_inputs(
        self,
        *,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: Optional[int],
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        """Validate and normalize common inference inputs before any compiled denoise core.

        Usage:
            Shared inference entrypoint helper for image, prompt/context, proprio, and latent shape checks.

        Args:
            prompt: Optional text prompt. Mutually exclusive with `context/context_mask`.
            input_image: Conditioning image shaped `[3, H, W]` or `[1, 3, H, W]`.
            num_video_frames: Optional generated video frame count. When present, must satisfy `T % 4 == 1`.
            proprio: Optional proprio vector shaped `[D]` or `[1, D]`.
            context: Optional precomputed context shaped `[L, D]` or `[1, L, D]`.
            context_mask: Optional precomputed context mask shaped `[L]` or `[1, L]`.

        Returns:
            Dict with normalized `input_image`, `context`, `context_mask`, optional `proprio`, image size,
            and latent spatial/temporal sizes.
        """
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if num_video_frames is None:
            if height % 16 != 0 or width % 16 != 0:
                raise ValueError(
                    f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
                )
        else:
            checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
            if (checked_h, checked_w) != (height, width):
                raise ValueError(
                    f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
                )
            if checked_t != num_video_frames:
                raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
        if context.shape[0] != 1:
            raise ValueError(f"Inference expects batch size 1 context, got {context.shape[0]}.")
        if context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"`context_mask` shape must match `context` first two dims, got {tuple(context_mask.shape)} vs {tuple(context.shape[:2])}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        latent_t = None
        if num_video_frames is not None:
            latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        patch_h = int(self.video_expert.patch_size[1])
        patch_w = int(self.video_expert.patch_size[2])
        if latent_h % patch_h != 0 or latent_w % patch_w != 0:
            raise ValueError(
                "VAE latent spatial shape must be divisible by video DiT patch size, "
                f"got HxW=({latent_h}, {latent_w}), patch=({patch_h}, {patch_w})"
            )

        return {
            "input_image": input_image.to(device=self.device, dtype=self.torch_dtype),
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio,
            "height": height,
            "width": width,
            "latent_t": latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
        }

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        """Encode a batch of training videos into VAE latents using the compiled VAE encoder.

        Usage:
            Training/evaluation only; used by `build_inputs` and VAE reconstruction metrics.

        Args:
            video_tensor: Video tensor shaped `[B, 3, T, H, W]`, normally in `[-1, 1]`.
            tiled: Unsupported for the current batched encode path.
            tile_size: Kept for legacy signature compatibility.
            tile_stride: Kept for legacy signature compatibility.

        Returns:
            VAE latents shaped `[B, z_dim, latent_T, latent_H, latent_W]`.
        """
        if tiled:
            raise NotImplementedError("Batched VAE encoding does not support tiled encoding.")
        if not hasattr(self, "_vae_encode_compiled"):
            self._vae_encode_compiled = torch.compile(
                self.vae.model.encode,
                mode="reduce-overhead",
                fullgraph=False,
            )
        return self._vae_encode_compiled(video_tensor.to(self.device), self.vae.scale).clone()

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        """Encode a single conditioning image as a one-frame VAE latent video.

        Usage:
            Inference only; used by `infer_joint` and `infer_action` to condition on the first frame.

        Args:
            input_image: Image tensor shaped `[3, H, W]` or `[1, 3, H, W]`, normally in `[-1, 1]`.
            tiled: Unsupported for this direct VAE encode path.
            tile_size: Kept for legacy signature compatibility.
            tile_stride: Kept for legacy signature compatibility.

        Returns:
            First-frame latent tensor shaped `[1, z_dim, 1, latent_H, latent_W]`.
        """
        if tiled:
            raise NotImplementedError("Batched VAE image encoding does not support tiled encoding.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        return self.vae.model.encode(image.unsqueeze(0), self.vae.scale)

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        """Decode VAE video latents into PIL frames.

        Usage:
            Inference/evaluation only; used to materialize generated or reconstructed videos.

        Args:
            latents: VAE latent tensor shaped `[1, z_dim, latent_T, latent_H, latent_W]`.
            tiled: Whether to use VAE tiled decode.
            tile_size: VAE tile size.
            tile_stride: VAE tile stride.

        Returns:
            List of RGB `PIL.Image` frames.
        """
        video_tensor = self.vae.decode(
            latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        """Validate and convert a training sample into tensors consumed by `training_loss`.

        Usage:
            Training/evaluation loss path only.

        Args:
            sample: Training batch containing `video` `[B, 3, T, H, W]`, `action` `[B, A, action_dim]`,
                optional text `context/context_mask` or `prompt`, optional `proprio`, and optional padding masks.
            tiled: Whether to request VAE tiled encoding; unsupported by the current batched encoder.

        Returns:
            Dict containing context tensors, VAE input latents, optional first-frame latent, action tensors,
            padding masks, and the VAE-fusion flag.
        """
        video = sample["video"]
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        has_context = context is not None or context_mask is not None
        if has_context:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist when either is provided.")
        else:
            prompt = sample.get("prompt", None)
            if prompt is None:
                raise ValueError("WBWAM training requires either `context/context_mask` or `prompt`.")
            if self.text_encoder is None or self.tokenizer is None:
                raise ValueError("Prompt-only training requires `model.load_text_encoder=true`.")
            context, context_mask = self.encode_prompt(prompt)
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 16, got H={height}, W={width}")
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for WBWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        if action.shape[0] != batch_size:
            raise ValueError(
                f"`sample['action']` batch size must match video batch size {batch_size}, got {action.shape[0]}"
            )
        if action.shape[2] != self.action_expert.action_dim:
            raise ValueError(
                f"`sample['action']` last dim must be {self.action_expert.action_dim}, got {action.shape[2]}"
            )
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        if action_dim_is_pad is not None:
            if action_dim_is_pad.ndim != 3:
                raise ValueError(
                    "`sample['action_dim_is_pad']` must be 3D [B, T, a_dim], "
                    f"got shape {tuple(action_dim_is_pad.shape)}"
                )
            expected_action_mask_shape = (batch_size, action_horizon, self.action_expert.action_dim)
            if tuple(action_dim_is_pad.shape) != expected_action_mask_shape:
                raise ValueError(
                    "`sample['action_dim_is_pad']` shape mismatch: "
                    f"got {tuple(action_dim_is_pad.shape)} vs expected {expected_action_mask_shape}"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        has_video = sample.get("has_video", None)
        if has_video is not None:
            has_video = torch.as_tensor(has_video)
        if has_video is None:
            has_video = (
                torch.ones((batch_size,), dtype=torch.bool) if image_is_pad is None else ~image_is_pad.all(dim=1)
            )
        elif has_video.ndim != 1 or has_video.shape[0] != batch_size:
            raise ValueError(
                "`sample['has_video']` must be 1D [B], "
                f"got shape {tuple(has_video.shape)} vs expected ({batch_size},)"
            )

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != batch_size:
            raise ValueError(
                f"`context` batch size must match video batch size {batch_size}, got {context.shape[0]}"
            )
        if context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"`context_mask` shape must match `context` first two dims, got {tuple(context_mask.shape)} vs {tuple(context.shape[:2])}"
            )
        patch_h = int(self.video_expert.patch_size[1])
        patch_w = int(self.video_expert.patch_size[2])
        if input_latents.shape[3] % patch_h != 0 or input_latents.shape[4] % patch_w != 0:
            raise ValueError(
                "VAE latent spatial shape must be divisible by video DiT patch size, "
                f"got HxW=({input_latents.shape[3]}, {input_latents.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        proprio_dim_is_pad = sample.get("proprio_dim_is_pad", None)
        if proprio_dim_is_pad is not None:
            if proprio_dim_is_pad.ndim != 3:
                raise ValueError(
                    "`sample['proprio_dim_is_pad']` must be 3D [B, T, d], "
                    f"got shape {tuple(proprio_dim_is_pad.shape)}"
                )
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[0] != batch_size:
                raise ValueError(
                    f"`sample['proprio']` batch size must match video batch size {batch_size}, got {proprio.shape[0]}"
                )
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            if proprio_dim_is_pad is not None:
                expected_proprio_mask_shape = (batch_size, proprio.shape[1], self.proprio_dim)
                if tuple(proprio_dim_is_pad.shape) != expected_proprio_mask_shape:
                    raise ValueError(
                        "`sample['proprio_dim_is_pad']` shape mismatch: "
                        f"got {tuple(proprio_dim_is_pad.shape)} vs expected {expected_proprio_mask_shape}"
                    )
                proprio_dim_is_pad = proprio_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            if proprio_dim_is_pad is not None:
                proprio = proprio.masked_fill(proprio_dim_is_pad, 0.0)
            proprio = proprio[:, 0, :]  # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
            action = action.masked_fill(action_dim_is_pad, 0.0)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        has_video = has_video.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "image_is_pad": image_is_pad,
            "has_video": has_video,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the mixed video/action self-attention mask used by MoT.

        Usage:
            Shared helper used by training, joint inference, and action-only inference setup.

        Args:
            video_seq_len: Number of video tokens.
            action_seq_len: Number of action tokens.
            video_tokens_per_frame: Number of video tokens belonging to one latent frame.
            device: Device for the returned mask.

        Returns:
            Boolean mask shaped `[video_seq_len + action_seq_len, video_seq_len + action_seq_len]`.
            Video uses the video expert causal mask, action attends to all action tokens and first-frame video.
        """
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        has_video: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-sample latent video MSE, excluding only unavailable videos.

        Usage:
            Training/evaluation loss path only.

        Args:
            pred_video: Predicted video latent target, shaped `[B, C, T_latent, H, W]`.
            target_video: Ground-truth training target with the same shape as `pred_video`.
            has_video: Optional video availability mask shaped `[B]`.

        Returns:
            Per-sample video loss tensor shaped `[B]`.
        """
        video_loss_per_sample = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 2, 3, 4))
        if has_video is None:
            return video_loss_per_sample
        if has_video.ndim != 1 or has_video.shape[0] != video_loss_per_sample.shape[0]:
            raise ValueError(
                "Video availability mask shape mismatch: "
                f"mask={tuple(has_video.shape)} loss={tuple(video_loss_per_sample.shape)}."
            )
        return video_loss_per_sample.masked_fill(
            ~has_video.to(device=video_loss_per_sample.device, dtype=torch.bool), 0.0
        )

    def _compute_action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute full action MSE; padding masks are used only for shape validation."""
        action_loss_dim = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")
        if action_dim_is_pad is not None:
            if action_dim_is_pad.shape != action_loss_dim.shape:
                raise ValueError(
                    "Action-loss dim mask shape mismatch: "
                    f"mask={tuple(action_dim_is_pad.shape)} loss={tuple(action_loss_dim.shape)}."
                )
        if action_is_pad is not None:
            if action_is_pad.shape != action_loss_dim.shape[:2]:
                raise ValueError(
                    "Action-loss temporal mask shape mismatch: "
                    f"mask={tuple(action_is_pad.shape)} loss={tuple(action_loss_dim.shape[:2])}."
                )
        return action_loss_dim.mean(dim=(1, 2))

    def _training_denoise_core(
        self,
        latents: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trainable video/action denoise forward shared by eager and compiled training paths.

        Usage:
            Training only; called by `training_loss`, optionally through `torch.compile`.

        Args:
            latents: Noisy video latents, shaped `[B, z_dim, T_latent, H_latent, W_latent]`.
            noisy_action: Noisy action tokens, shaped `[B, T_action, action_dim]`.
            timestep_video: Video diffusion timestep tensor shaped `[B]`.
            timestep_action: Action diffusion timestep tensor shaped `[B]`.
            context: Text/proprio context tokens, shaped `[B, L, text_dim]`.
            context_mask: Boolean context mask, shaped `[B, L]`.
            action: Clean action tensor used for video expert conditioning, shaped `[B, T_action, action_dim]`.
            fuse_vae_embedding_in_latents: Whether the video expert expects first-frame VAE fusion.

        Returns:
            `(pred_video, pred_action)` where `pred_video` is `[B, z_dim, T_latent, H_latent, W_latent]`
            and `pred_action` is `[B, T_action, action_dim]`.
        """
        (
            video_tokens,
            t_video,
            t_mod_video,
            context_video,
            context_mask_video,
            freqs_video,
            f_video,
            h_video,
            w_video,
            tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        pred_video_tokens, pred_action_tokens = self.mot.forward_joint_core(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            video_freqs=freqs_video,
            action_freqs=freqs_action,
            video_t_mod=t_mod_video,
            action_t_mod=t_mod_action,
            video_context=context_video,
            video_context_mask=context_mask_video,
            action_context=context_action,
            action_context_mask=context_mask_action,
            attention_mask=attention_mask,
        )

        pred_video = self.video_expert.post(pred_video_tokens, t_video, f_video, h_video, w_video)
        pred_action = self.action_expert.post(pred_action_tokens)
        return pred_video, pred_action

    def _add_fk_training_loss(
        self,
        loss_total,
        loss_dict,
        *,
        sample,
        pred_action,
        noisy_action,
        timestep_action,
        action_weight,
        action_is_pad,
        action_dim_is_pad,
    ):
        """Add opt-in geometry supervision without changing the base loss path."""
        auxiliary = getattr(self, "fk_link_loss", None)
        if auxiliary is None or auxiliary.weight == 0:
            return loss_total, loss_dict
        sigma = (timestep_action / float(self.train_action_scheduler.num_train_timesteps)).to(noisy_action.dtype)
        loss_fk, metrics = auxiliary(
            pred_flow=pred_action,
            noisy_action=noisy_action,
            sigma=sigma,
            target=sample["fk_joint_target"],
            joint_zero=sample["fk_joint_offset"],
            action_is_pad=action_is_pad,
            action_dim_is_pad=action_dim_is_pad,
            timestep_weight=action_weight,
        )
        loss_dict.update(metrics)
        return loss_total + loss_fk, loss_dict

    def training_loss(self, sample, tiled: bool = False):
        """Compute the joint video/action flow-matching training loss for one batch.

        Usage:
            Training/evaluation loss path; `forward` delegates here.

        Args:
            sample: Raw training batch accepted by `build_inputs`.
            tiled: Whether to request tiled VAE encode; currently unsupported for batched VAE encode.

        Returns:
            `(loss_total, loss_dict)` where `loss_total` is a differentiable scalar tensor and
            `loss_dict` contains detached Python floats for logging.
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        action_dim_is_pad = inputs["action_dim_is_pad"]
        has_video = inputs["has_video"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
        if action_dim_is_pad is not None:
            noisy_action = noisy_action.masked_fill(action_dim_is_pad, 0.0)
            target_action = target_action.masked_fill(action_dim_is_pad, 0.0)

        if self.compile_training_denoise:
            if not hasattr(self, "_training_denoise_core_compiled"):
                self._training_denoise_core_compiled = torch.compile(
                    self._training_denoise_core,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            denoise_core = self._training_denoise_core_compiled
        else:
            denoise_core = self._training_denoise_core

        pred_video, pred_action = denoise_core(
            latents=latents,
            noisy_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            has_video=has_video,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            action_dim_is_pad=action_dim_is_pad,
        )

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return self._add_fk_training_loss(
            loss_total,
            loss_dict,
            sample=sample,
            pred_action=pred_action,
            noisy_action=noisy_action,
            timestep_action=timestep_action,
            action_weight=action_weight,
            action_is_pad=action_is_pad,
            action_dim_is_pad=action_dim_is_pad,
        )

    def _denoise_joint(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one joint inference denoise step for video and action latents.

        Usage:
            Inference only; called repeatedly by `infer_joint`, optionally through `torch.compile`.

        Args:
            latents_video: Current video latents, shaped `[1, z_dim, T_latent, H_latent, W_latent]`.
            latents_action: Current action latents, shaped `[1, T_action, action_dim]`.
            timestep_video: Video inference timestep, shaped `[1]`.
            timestep_action: Action inference timestep, shaped `[1]`.
            context: Conditioning context, shaped `[1, L, text_dim]`.
            context_mask: Boolean context mask, shaped `[1, L]`.
            attention_mask: Mixed MoT mask, shaped `[S_video + S_action, S_video + S_action]`.
            fuse_vae_embedding_in_latents: Whether the video expert expects first-frame VAE fusion.
            gt_action: Optional action conditioning for video generation, shaped `[1, T_action, action_dim]`.

        Returns:
            `(pred_video, pred_action)` denoise predictions matching the video/action latent shapes.
        """
        (
            video_tokens,
            t_video,
            t_mod_video,
            context_video,
            context_mask_video,
            freqs_video,
            f_video,
            h_video,
            w_video,
            _tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        pred_video_tokens, pred_action_tokens = self.mot.forward_joint_core(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            video_freqs=freqs_video,
            action_freqs=freqs_action,
            video_t_mod=t_mod_video,
            action_t_mod=t_mod_action,
            video_context=context_video,
            video_context_mask=context_mask_video,
            action_context=context_action,
            action_context_mask=context_mask_action,
            attention_mask=attention_mask,
        )

        pred_video = self.video_expert.post(pred_video_tokens, t_video, f_video, h_video, w_video)
        pred_action = self.action_expert.post(pred_action_tokens)
        return pred_video, pred_action

    def _denoise_action_with_video_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        action_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Run one action-only denoise step using cached first-frame video attention K/V.

        Usage:
            Inference only; called repeatedly by `infer_action`, optionally through `torch.compile`.

        Args:
            latents_action: Current action latents, shaped `[1, T_action, action_dim]`.
            timestep_action: Action inference timestep, shaped `[1]`.
            context: Conditioning context, shaped `[1, L, text_dim]`.
            context_mask: Boolean context mask, shaped `[1, L]`.
            video_cache_k: Per-layer video key cache from `MoT.prefill_video_cache`.
            video_cache_v: Per-layer video value cache from `MoT.prefill_video_cache`.
            action_attention_mask: Boolean action attention mask, shaped `[S_action, S_video + S_action]`.
            action_freqs: Action RoPE frequencies for the action sequence.

        Returns:
            Predicted action denoise tensor shaped `[1, T_action, action_dim]`.
        """
        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            _freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=t_mod_action,
            action_context=context_action,
            action_context_mask=context_mask_action,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask,
        )
        return self.action_expert.post(action_tokens)

    def _denoise_action_only(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one action denoise step without video tokens or K/V cache."""
        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            action_freqs,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_seq_len = action_tokens.shape[1]
        action_attention_mask = torch.ones(
            (action_seq_len, action_seq_len),
            dtype=torch.bool,
            device=action_tokens.device,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=t_mod_action,
            action_context=context_action,
            action_context_mask=context_mask_action,
            video_cache_k=None,
            video_cache_v=None,
            action_attention_mask=action_attention_mask,
        )
        return self.action_expert.post(action_tokens)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[
            torch.Tensor
        ] = None,  # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        compile_action_infer: bool = False,
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Generate video and action jointly from an input image and text/context condition.

        Usage:
            Inference/evaluation entrypoint for joint video-action rollout.

        Args:
            prompt: Optional text prompt. Mutually exclusive with `context/context_mask`.
            input_image: Conditioning image shaped `[3, H, W]` or `[1, 3, H, W]`.
            num_video_frames: Number of output video frames; must satisfy `T % 4 == 1`.
            action_horizon: Number of output action steps.
            action: Optional clean action condition for video generation, shaped `[T, D]` or `[1, T, D]`.
            proprio: Optional proprio vector shaped `[D]` or `[1, D]`.
            context: Optional precomputed context tokens, shaped `[L, text_dim]` or `[1, L, text_dim]`.
            context_mask: Optional precomputed context mask, shaped `[L]` or `[1, L]`.
            negative_prompt: Kept for API compatibility; not used by this path.
            text_cfg_scale: Kept for API compatibility; classifier-free guidance is not applied here.
            num_inference_steps: Number of scheduler denoise steps.
            sigma_shift: Optional scheduler shift override.
            seed: Optional random seed.
            rand_device: Device used for random latent sampling.
            tiled: Whether to use tiled VAE decode/encode where supported.
            test_action_with_infer_action: Also run action-only path and warn if it diverges.
            compile_action_infer: Whether to compile the joint denoise step.
            action_dim_is_pad: Semantic dimension mask shaped `[D]`, `[T,D]`, `[1,T,D]`, or
                `[B,T,D]`. Masked dimensions stay zero throughout denoising; do not use it for
                temporal tail padding.

        Returns:
            Dict with `"video"` as PIL frames and `"action"` as `[T_action, action_dim]` CPU float tensor.
        """
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                action_dim_is_pad=action_dim_is_pad,
            )["action"]

        infer_inputs = self._validate_infer_inputs(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
        )
        input_image = infer_inputs["input_image"]
        context = infer_inputs["context"]
        context_mask = infer_inputs["context_mask"]
        latent_t = infer_inputs["latent_t"]
        latent_h = infer_inputs["latent_h"]
        latent_w = infer_inputs["latent_w"]
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            if action.shape[2] != self.action_expert.action_dim:
                raise ValueError(
                    f"`action` last dim must be {self.action_expert.action_dim}, got {action.shape[2]}"
                )
            if latent_t <= 1:
                raise ValueError(f"`action` conditioning requires at least 2 latent video frames, got {latent_t}.")
            if action.shape[1] % (latent_t - 1) != 0:
                raise ValueError(
                    f"`action` temporal dimension must be divisible by latent video transitions ({latent_t - 1}), got {action.shape[1]}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if self.video_expert.action_conditioned and latent_t > 1 and action is None:
            raise ValueError("Action-conditioned video expert requires `action` for multi-frame `infer_joint`.")

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = self._zero_inference_action_padding(latents_action, action_dim_is_pad)

        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        patch_t = int(self.video_expert.patch_size[0])
        patch_h = int(self.video_expert.patch_size[1])
        patch_w = int(self.video_expert.patch_size[2])
        video_tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        video_seq_len = (latent_t // patch_t) * video_tokens_per_frame
        joint_attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=video_tokens_per_frame,
            device=self.device,
        )

        if compile_action_infer:
            if action is not None:
                raise ValueError(
                    "`compile_action_infer=True` does not support `action` conditioning in `infer_joint`."
                )
            if not hasattr(self, "_denoise_joint_compiled"):
                self._denoise_joint_compiled = torch.compile(
                    self._denoise_joint,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            denoise_joint = self._denoise_joint_compiled
        else:
            denoise_joint = self._denoise_joint

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = denoise_joint(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                attention_mask=joint_attention_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_action = self._zero_inference_action_padding(latents_action, action_dim_is_pad)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Generate only an action trajectory conditioned on the input image and text/context.

        Usage:
            Inference entrypoint for online action-policy callers.

        Args:
            prompt: Optional text prompt. Mutually exclusive with `context/context_mask`.
            input_image: Conditioning image shaped `[3, H, W]` or `[1, 3, H, W]`.
            action_horizon: Number of output action steps.
            proprio: Optional proprio vector shaped `[D]` or `[1, D]`.
            context: Optional precomputed context tokens, shaped `[L, text_dim]` or `[1, L, text_dim]`.
            context_mask: Optional precomputed context mask, shaped `[L]` or `[1, L]`.
            negative_prompt: Kept for API compatibility; not used by this path.
            text_cfg_scale: Kept for API compatibility; classifier-free guidance is not applied here.
            num_inference_steps: Number of action scheduler denoise steps.
            sigma_shift: Optional scheduler shift override.
            seed: Optional random seed.
            rand_device: Device used for random action latent sampling.
            tiled: Whether to use tiled VAE image encode; unsupported by the direct image encode path.
            compile_action_infer: Whether to compile video-cache prefill and action denoise.
            action_dim_is_pad: Semantic dimension mask shaped `[D]`, `[T,D]`, `[1,T,D]`, or
                `[B,T,D]`. Masked dimensions stay zero throughout denoising; do not use it for
                temporal tail padding.

        Returns:
            Dict with `"action"` as `[T_action, action_dim]` CPU float tensor.
        """
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("`infer_action` requires `video_attention_mask_mode='first_frame_causal'`.")

        infer_inputs = self._validate_infer_inputs(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=None,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
        )
        input_image = infer_inputs["input_image"]
        context = infer_inputs["context"]
        context_mask = infer_inputs["context_mask"]

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = self._zero_inference_action_padding(latents_action, action_dim_is_pad)

        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        (
            video_tokens,
            _t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _f_video,
            _h_video,
            _w_video,
            tokens_per_frame,
        ) = self.video_expert.prepare(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_tokens.shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        action_seq_len = latents_action.shape[1]
        total_seq_len = video_seq_len + action_seq_len
        if attention_mask.ndim != 2 or attention_mask.shape != (total_seq_len, total_seq_len):
            raise ValueError(
                "`attention_mask` must be [video_seq_len + action_seq_len, video_seq_len + action_seq_len], "
                f"got {tuple(attention_mask.shape)} vs ({total_seq_len}, {total_seq_len})."
            )
        if action_seq_len > self.action_expert.max_rope_len:
            raise ValueError(
                f"Action token length {action_seq_len} exceeds RoPE cache {self.action_expert.max_rope_len}."
            )
        video_attention_mask = attention_mask[:video_seq_len, :video_seq_len]
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        action_freqs = self.action_expert.get_freqs(seq_len=action_seq_len)
        if compile_action_infer:
            if not hasattr(self, "_prefill_video_cache_compiled"):
                self._prefill_video_cache_compiled = torch.compile(
                    self.mot.prefill_video_cache,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            if not hasattr(self, "_denoise_action_with_video_cache_compiled"):
                self._denoise_action_with_video_cache_compiled = torch.compile(
                    self._denoise_action_with_video_cache,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            prefill_video_cache = self._prefill_video_cache_compiled
            denoise_action_with_video_cache = self._denoise_action_with_video_cache_compiled
        else:
            prefill_video_cache = self.mot.prefill_video_cache
            denoise_action_with_video_cache = self._denoise_action_with_video_cache

        cache_k_list, cache_v_list = prefill_video_cache(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        if compile_action_infer:
            # Clone CUDAGraph outputs from compiled prefill before reusing them as long-lived cache.
            cache_k_list = [cache_k.clone() for cache_k in cache_k_list]
            cache_v_list = [cache_v.clone() for cache_v in cache_v_list]

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action = denoise_action_with_video_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_cache_k=cache_k_list,
                video_cache_v=cache_v_list,
                action_attention_mask=action_attention_mask,
                action_freqs=action_freqs,
            )

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_action = self._zero_inference_action_padding(latents_action, action_dim_is_pad)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ):
        """Compatibility wrapper around `infer_joint`.

        Args:
            prompt: Optional text prompt.
            input_image: Conditioning image shaped `[3, H, W]` or `[1, 3, H, W]`.
            num_frames: Number of output video frames.
            action: Optional clean action condition for video generation.
            action_horizon: Number of action steps to generate.
            proprio: Optional proprio vector.
            context: Optional precomputed context tokens.
            context_mask: Optional precomputed context mask.
            negative_prompt: Kept for API compatibility.
            text_cfg_scale: Kept for API compatibility.
            action_cfg_scale: Kept for API compatibility.
            num_inference_steps: Number of denoise steps.
            sigma_shift: Optional scheduler shift override.
            seed: Optional random seed.
            rand_device: Device used for random latent sampling.
            tiled: Whether to use tiled VAE decode/encode where supported.
            action_dim_is_pad: Semantic dimension mask shaped `[D]`, `[T,D]`, `[1,T,D]`, or
                `[B,T,D]`; it must not encode temporal tail padding.

        Returns:
            Dict returned by `infer_joint`.
        """
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            action_dim_is_pad=action_dim_is_pad,
        )

    @torch.no_grad()
    def evaluate(
        self,
        sample,
        *,
        num_inference_steps: int,
        eval_video_path: str,
        seed: int = 42,
    ):
        """Run validation loss, rollout inference, VAE reconstruction, and video metrics for one sample.

        Usage:
            Evaluation only; called by trainer/eval scripts.

        Args:
            sample: Evaluation batch with video/action/prompt or precomputed context fields.
            num_inference_steps: Number of rollout denoise steps.
            eval_video_path: Output path for stitched rollout/reconstruction/ground-truth MP4.
            seed: Inference random seed.

        Returns:
            Metrics dict containing validation loss, PSNR/SSIM values, output video path, and optional action L2.
        """
        was_training = self.dit.training
        self.eval()
        val_loss, _ = self.training_loss(sample)
        val_loss = float(val_loss.float().item())

        prompt = sample["prompt"][0]
        video0 = sample["video"][0]
        action = sample["action"][0] if sample.get("action", None) is not None else None
        proprio = sample["proprio"][0, 0] if sample.get("proprio", None) is not None else None
        _, num_frames, height, width = video0.shape

        raw_has_video = sample.get("has_video", None)
        if raw_has_video is None:
            image_is_pad = sample.get("image_is_pad", None)
            has_video = True if image_is_pad is None else not bool(torch.as_tensor(image_is_pad).all().item())
        else:
            raw_has_video = torch.as_tensor(raw_has_video, dtype=torch.bool)
            if raw_has_video.numel() != 1:
                raise ValueError(
                    "WBWAM.evaluate expects a single-sample `has_video` mask, "
                    f"got shape {tuple(raw_has_video.shape)}."
                )
            has_video = bool(raw_has_video.reshape(-1)[0].item())

        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        if context is not None:
            if context_mask is None:
                raise ValueError("`context_mask` must exist when `context` is provided.")
            text_infer_kwargs = {
                "prompt": None,
                "context": context[0],
                "context_mask": context_mask[0],
            }
        else:
            text_infer_kwargs = {"prompt": prompt}

        semantic_mask = sample.get("action_semantic_dim_is_pad", None)
        if semantic_mask is None and sample.get("action_dim_is_pad", None) is not None:
            action_is_pad = sample.get("action_is_pad", None)
            if action_is_pad is not None and torch.as_tensor(action_is_pad).any():
                raise ValueError(
                    "Evaluation needs action_semantic_dim_is_pad when action_is_pad contains padding; "
                    "the combined training mask must not constrain the rollout's unknown end time."
                )
            semantic_mask = sample["action_dim_is_pad"]
        if semantic_mask is not None:
            semantic_mask = torch.as_tensor(semantic_mask, dtype=torch.bool)
            expected = (1, int(sample["action_horizon"]), int(self.action_expert.action_dim))
            if tuple(semantic_mask.shape) != expected:
                raise ValueError(f"Eval semantic action mask must be {expected}, got {tuple(semantic_mask.shape)}")
            semantic_mask = semantic_mask[0]

        if has_video:
            infer_kwargs = {
                "input_image": video0[:, 0].unsqueeze(0),
                "num_frames": num_frames,
                "action": action,
                "action_horizon": sample["action_horizon"],
                "action_dim_is_pad": semantic_mask,
                "proprio": proprio,
                "text_cfg_scale": 1.0,
                "action_cfg_scale": 1.0,
                "num_inference_steps": int(num_inference_steps),
                "seed": int(seed),
                "tiled": False,
                **text_infer_kwargs,
            }
            pred = self.infer(**infer_kwargs)
            pred_video_tensor = pil_frames_to_video_tensor(pred["video"])
            pred_action = pred.get("action", None)
            gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
            if pred_video_tensor.shape != gt_video_tensor.shape:
                raise ValueError(
                    "Eval infer prediction/GT shape mismatch: "
                    f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
                )

            psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
            ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

            gt_video_batch = video0.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
            vae_latents = self._encode_video_latents(gt_video_batch, tiled=False)
            vae_recon_video = self._decode_latents(vae_latents, tiled=False)
            vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)
            if vae_video_tensor.shape != gt_video_tensor.shape:
                raise ValueError(
                    "Eval VAE reconstruction/GT shape mismatch: "
                    f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
                )

            psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
            ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)
            psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
            ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)
            video_metric_valid = 1.0
        else:
            infer_action_no_video = getattr(self, "infer_action_no_video", None)
            if infer_action_no_video is None:
                raise RuntimeError(
                    f"{type(self).__name__} cannot evaluate a no-video sample because "
                    "`infer_action_no_video` is unavailable."
                )
            pred = infer_action_no_video(
                action_horizon=sample["action_horizon"],
                action_dim_is_pad=semantic_mask,
                proprio=proprio,
                num_inference_steps=int(num_inference_steps),
                seed=int(seed),
                **text_infer_kwargs,
            )
            pred_action = pred.get("action", None)
            pred_video_tensor = torch.zeros(
                (3, num_frames, height, width),
                dtype=torch.float32,
                device="cpu",
            )
            gt_video_tensor = torch.zeros_like(pred_video_tensor)
            vae_video_tensor = torch.zeros_like(pred_video_tensor)
            psnr_rollout_vs_gt = 0.0
            ssim_rollout_vs_gt = 0.0
            psnr_decode_vs_gt = 0.0
            ssim_decode_vs_gt = 0.0
            psnr_rollout_vs_decode = 0.0
            ssim_rollout_vs_decode = 0.0
            video_metric_valid = 0.0

        action_l2 = None
        action_error_sum_valid = 0.0
        action_valid_count = 0
        action_mse_valid = None
        if pred_action is not None and action is not None:
            action_target = action.to(device=pred_action.device, dtype=pred_action.dtype)
            if pred_action.shape != action_target.shape:
                raise ValueError(
                    f"Eval action prediction/target shape mismatch: {pred_action.shape} vs {action_target.shape}"
                )
            action_l2 = (pred_action - action_target).pow(2).mean().item()
            valid = torch.ones_like(pred_action, dtype=torch.bool)
            if self._inference_action_invalid_indices:
                valid[:, list(self._inference_action_invalid_indices)] = False
            for key in ("action_dim_is_pad", "action_semantic_dim_is_pad"):
                mask = sample.get(key, None)
                if mask is not None:
                    mask = torch.as_tensor(mask, dtype=torch.bool, device=pred_action.device)
                    if tuple(mask.shape) != (1, *pred_action.shape):
                        raise ValueError(f"Eval {key} must be [1,T,D], got {tuple(mask.shape)}")
                    valid &= ~mask[0]
            time_mask = sample.get("action_is_pad", None)
            if time_mask is not None:
                time_mask = torch.as_tensor(time_mask, dtype=torch.bool, device=pred_action.device)
                if tuple(time_mask.shape) != (1, pred_action.shape[0]):
                    raise ValueError(f"Eval action_is_pad must be [1,T], got {tuple(time_mask.shape)}")
                valid &= ~time_mask[0, :, None]
            action_valid_count = int(valid.sum().item())
            squared_error = (pred_action.float() - action_target.float()).square()
            action_error_sum_valid = squared_error[valid].double().sum().item()
            if action_valid_count:
                action_mse_valid = action_error_sum_valid / action_valid_count

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor], dim=2
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))
        save_mp4(stitched_frames, eval_video_path, fps=8)

        if was_training:
            self.set_train_mode_for_training()
        result = {
            "val_loss": val_loss,
            "psnr_rg": float(psnr_rollout_vs_gt),
            "ssim_rg": float(ssim_rollout_vs_gt),
            "psnr_rd": float(psnr_rollout_vs_decode),
            "ssim_rd": float(ssim_rollout_vs_decode),
            "psnr_dg": float(psnr_decode_vs_gt),
            "ssim_dg": float(ssim_decode_vs_gt),
            "video_path": eval_video_path,
            "video_metric_valid": video_metric_valid,
            "action_error_sum_valid": action_error_sum_valid,
            "action_valid_count": action_valid_count,
            "action_mse_valid": action_mse_valid,
        }
        if action_l2 is not None:
            result["action_l2"] = action_l2
        return result

    def save_checkpoint(self, path, optimizer=None, step=None):
        """Save trainable model state and optional optimizer state.

        Args:
            path: Destination checkpoint path.
            optimizer: Optional optimizer whose state should be saved.
            step: Optional training step metadata.

        Output:
            Writes a checkpoint file containing MoT weights, optional proprio encoder weights, and metadata.
        """
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, strict: Optional[bool] = None):
        """Load trainable weights from a world-action checkpoint.

        Args:
            path: Checkpoint path produced by `save_checkpoint` or a legacy DiT checkpoint.
            optimizer: Optional optimizer to restore when the payload contains optimizer state.
            strict: Explicit MoT strictness override. Defaults to non-strict loading when omitted.
                Legacy DiT loading remains non-strict.

        Returns:
            Loaded checkpoint payload.
        """
        payload = torch.load(path, map_location=self.device)
        if "mot" in payload:
            effective_strict = False if strict is None else strict
            incompatible = self.mot.load_state_dict(payload["mot"], strict=effective_strict)
            logger.info(
                "Loaded MoT checkpoint: strict=%s tensors=%d missing=%d unexpected=%d",
                effective_strict,
                len(payload["mot"]),
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                logger.info(
                    "Loaded proprio checkpoint strictly: tensors=%d",
                    len(payload["proprio_encoder"]),
                )
            else:
                logger.warning(
                    "Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params."
                )
        elif "proprio_encoder" in payload:
            logger.warning(
                "Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        """Alias module forward to `training_loss`.

        Args:
            *args/**kwargs: Arguments forwarded to `training_loss`.

        Returns:
            The `(loss_total, loss_dict)` tuple from `training_loss`.
        """
        return self.training_loss(*args, **kwargs)
