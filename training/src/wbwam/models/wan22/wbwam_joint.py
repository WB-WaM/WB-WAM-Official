from typing import Any, Optional

import torch

from wbwam.utils.logging_config import get_logger

from .wbwam import WBWAM

logger = get_logger(__name__)


class WBWAMJoint(WBWAM):
    """WBWAM variant where action attends to all video latent tokens."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        """Build a full-attention world-action model from Wan2.2 components.

        Args:
            **kwargs: Same keyword arguments accepted by `WBWAM.from_wan22_pretrained`;
                `video_dit_config` must be a dict with `action_conditioned=false`.

        Returns:
            A `WBWAMJoint` instance.
        """
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for WBWAMJoint."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "WBWAMJoint requires `video_dit_config['action_conditioned']=false`."
            )
        return super().from_wan22_pretrained(**kwargs)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the full-attention MoT mask for video/action tokens.

        Usage:
            Shared by training and full-attention inference through the base denoise core.

        Args:
            video_seq_len: Number of prepared video tokens.
            action_seq_len: Number of prepared action tokens.
            video_tokens_per_frame: Number of video tokens per latent frame.
            device: Device for the returned mask.

        Returns:
            Boolean attention mask shaped `[video_seq_len + action_seq_len, video_seq_len + action_seq_len]`;
            action tokens attend to all video tokens and all action tokens.
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
        # action -> full video
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
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
        """Run joint inference while disabling the base action-only consistency check.

        Usage:
            Inference/evaluation entrypoint; delegates to `WBWAM.infer_joint`.

        Args:
            prompt: Optional text prompt. Mutually exclusive with `context/context_mask`.
            input_image: Conditioning image shaped `[3, H, W]` or `[1, 3, H, W]`.
            num_video_frames: Number of generated video frames.
            action_horizon: Number of generated action steps.
            action: Optional action conditioning forwarded to the base joint path.
            proprio: Optional proprio vector shaped `[D]` or `[1, D]`.
            context: Optional precomputed context tokens.
            context_mask: Optional precomputed context mask.
            negative_prompt: Kept for API compatibility.
            text_cfg_scale: Kept for API compatibility.
            num_inference_steps: Number of scheduler denoise steps.
            sigma_shift: Optional scheduler shift override.
            seed: Optional random seed.
            rand_device: Device used for random sampling.
            tiled: Whether to use tiled VAE encode/decode where supported.
            test_action_with_infer_action: Ignored by this variant.
            compile_action_infer: Whether to compile the joint denoise step.
            action_dim_is_pad: Semantic dimension mask; masked dimensions stay zero throughout
                denoising. It must not encode temporal tail padding.

        Returns:
            Dict returned by `WBWAM.infer_joint`, including `"video"` and `"action"`.
        """
        if test_action_with_infer_action:
            logger.warning(
                "`WBWAMJoint.infer_joint` ignores `test_action_with_infer_action=True` "
                "and always runs with `test_action_with_infer_action=False`."
            )
        return super().infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
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
            test_action_with_infer_action=False,
            compile_action_infer=compile_action_infer,
            action_dim_is_pad=action_dim_is_pad,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
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
        compile_action_infer: bool = True,
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Infer actions by jointly denoising full video and action latents.

        Usage:
            Full-attention inference path; action attends to all current video tokens, so video is denoised
            together with action at every scheduler step.

        Args:
            prompt: Optional text prompt. Mutually exclusive with `context/context_mask`.
            input_image: Conditioning image shaped `[3, H, W]` or `[1, 3, H, W]`.
            action_horizon: Number of generated action steps.
            num_video_frames: Number of generated video frames used by the joint denoise path.
            proprio: Optional proprio vector shaped `[D]` or `[1, D]`.
            context: Optional precomputed context tokens.
            context_mask: Optional precomputed context mask.
            negative_prompt: Kept for API compatibility.
            text_cfg_scale: Kept for API compatibility.
            num_inference_steps: Number of scheduler denoise steps.
            sigma_shift: Optional scheduler shift override.
            seed: Optional random seed.
            rand_device: Device used for random sampling.
            tiled: Whether to use tiled VAE image encode where supported.
            compile_action_infer: Whether to compile the joint denoise core.
            action_dim_is_pad: Semantic dimension mask; masked dimensions stay zero throughout
                denoising. It must not encode temporal tail padding.

        Returns:
            Dict with `"action"` as `[T_action, action_dim]` CPU float tensor.
        """
        self.eval()

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
                gt_action=None,
            )

            latents_video = self.infer_video_scheduler.step(pred_video_posi, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)
            latents_action = self._zero_inference_action_padding(latents_action, action_dim_is_pad)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
