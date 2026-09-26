from typing import Any, Optional

import torch

from .wbwam import WBWAM
from .wbwam_idm import WBWAMIDM


class WBWAMOptionalIDM(WBWAMIDM):
    """WBWAM variant where IDM-style full-video conditioning is optional."""

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: Optional[torch.Tensor],
        action_horizon: int,
        num_video_frames: Optional[int] = None,
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
        action_infer_mode: str = "idm",
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Infer action with either IDM two-stage video conditioning or first-frame-only conditioning.

        Usage:
            Inference entrypoint for this mixed-training variant. `action_infer_mode="idm"` first denoises
            video and lets action attend to all denoised video tokens. `action_infer_mode="first_frame"` skips
            video denoising and runs the base WBWAM action path where action sees only first-frame tokens.
            `action_infer_mode="no_video"` uses text/proprio context without any video tokens.

        Args:
            prompt: Optional text prompt. Mutually exclusive with `context/context_mask`.
            input_image: Optional conditioning image. Required by `idm` and `first_frame`; ignored by
                `no_video`.
            action_horizon: Number of generated action steps.
            num_video_frames: Number of video frames for IDM two-stage inference.
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
            compile_action_infer: Whether to compile inference cores.
            action_infer_mode: One of `"idm"`, `"first_frame"`, or `"no_video"`.
            action_dim_is_pad: Semantic dimension mask; masked dimensions stay zero throughout
                denoising. It must not encode temporal tail padding.

        Returns:
            Dict with `"action"`.
        """
        if action_infer_mode == "no_video":
            return self.infer_action_no_video(
                prompt=prompt,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                compile_action_infer=compile_action_infer,
                action_dim_is_pad=action_dim_is_pad,
            )

        if action_infer_mode in {"idm", "first_frame"} and input_image is None:
            raise ValueError(f"`input_image` is required for `action_infer_mode={action_infer_mode!r}`.")

        if action_infer_mode == "idm":
            if num_video_frames is None:
                raise ValueError("`num_video_frames` is required for `action_infer_mode='idm'`.")
            out = self.infer_joint(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=int(num_video_frames),
                action_horizon=action_horizon,
                action=None,
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
            return {"action": out["action"]}

        if action_infer_mode == "first_frame":
            return WBWAM.infer_action(
                self,
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
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
                compile_action_infer=compile_action_infer,
                action_dim_is_pad=action_dim_is_pad,
            )

        raise ValueError("`action_infer_mode` must be one of: idm, first_frame, no_video.")

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        batch_size: int,
        device: torch.device,
        has_video: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the teacher-forcing mask with stochastic action access to cond-video tokens.

        Usage:
            Training core helper. For each batch element, action tokens either attend to all cond-video tokens
            or only the first cond-video latent frame. They never attend to the noisy-video branch.

        Args:
            noisy_video_seq_len: Token length of the noisy-video branch.
            cond_video_seq_len: Token length of the conditioning-video branch.
            action_seq_len: Token length of the action branch.
            noisy_video_tokens_per_frame: Noisy-video tokens per latent frame.
            cond_video_tokens_per_frame: Conditioning-video tokens per latent frame.
            batch_size: Batch size for per-sample action mask sampling.
            device: Device for the returned mask.
            has_video: Optional per-sample video availability mask shaped `[B]`. No-video
                samples always use action-only self-attention.

        Returns:
            Boolean mixed attention mask shaped `[B, 1, S_total, S_total]`.
        """
        full_cond_mask = super()._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_seq_len,
            noisy_video_tokens_per_frame=noisy_video_tokens_per_frame,
            cond_video_tokens_per_frame=cond_video_tokens_per_frame,
            batch_size=batch_size,
            device=device,
            has_video=has_video,
        )

        prob = float(self.action_idm_prob)
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"`action_idm_prob` must be in [0, 1], got {prob}.")

        noisy_end = noisy_video_seq_len
        cond_end = noisy_video_seq_len + cond_video_seq_len
        first_frame_tokens = min(cond_video_tokens_per_frame, cond_video_seq_len)

        first_frame_cond_mask = full_cond_mask.clone()
        first_frame_cond_mask[..., cond_end:, noisy_end:cond_end] = False
        first_frame_cond_mask[..., cond_end:, noisy_end : noisy_end + first_frame_tokens] = True
        if has_video is not None:
            first_frame_cond_mask[..., cond_end:, noisy_end : noisy_end + first_frame_tokens] &= (
                has_video.view(batch_size, 1, 1, 1)
            )

        # One Bernoulli sample per batch element; broadcast over heads, queries, and keys.
        use_idm_mask = torch.rand((batch_size, 1, 1, 1), device=device) < prob
        return torch.where(use_idm_mask, full_cond_mask, first_frame_cond_mask)
