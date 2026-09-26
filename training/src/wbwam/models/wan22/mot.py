from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from wbwam.utils.logging_config import get_logger

from .wan_video_dit import flash_attention, modulate, rope_apply

logger = get_logger(__name__)


class MoT(nn.Module):
    """Mixture-of-Transformers wrapper that jointly runs video and action experts."""

    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
    ):
        """Validate and store video/action experts with matched transformer topology.

        Args:
            mixtures: Mapping containing `"video"` and `"action"` experts. Each expert must expose
                `blocks`, `num_heads`, and `attn_head_dim`, and all experts must have the same layer count
                and attention head layout.

        Output:
            Initializes `self.mixtures`; no tensor output.
        """
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self._training_dynamics_monitor = None

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        
        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        """Split a block timestep modulation tensor into attention/MLP affine and gate terms.

        Usage:
            Shared helper used by training and inference MoT paths.

        Args:
            block: Expert transformer block with a learned `modulation` table.
            t_mod: Timestep modulation, shaped either `[B, 6C]` for shared token modulation or
                `[B, S, 1, 6C]` for token-wise modulation.

        Returns:
            Tuple `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`, each broadcastable
            to the expert token tensor.
        """
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the remainder of one expert block after mixed self-attention.

        Usage:
            Shared helper used by training, joint inference, and cached action inference.

        Args:
            block: Expert block that owns self-attention output projection, cross-attention, FFN, and gates.
            residual_x: Input token residual before self-attention, shaped `[B, S, C]`.
            mixed_attn_out: Mixed self-attention output before the expert output projection, shaped `[B, S, C]`.
            gate_msa: Self-attention residual gate, broadcastable to `[B, S, C]`.
            shift_mlp: MLP normalization shift, broadcastable to `[B, S, C]`.
            scale_mlp: MLP normalization scale, broadcastable to `[B, S, C]`.
            gate_mlp: MLP residual gate, broadcastable to `[B, S, C]`.
            context: Cross-attention context tokens, shaped `[B, L, D]`.
            context_mask: Prepared boolean cross-attention mask, shaped `[B, S, L]`.

        Returns:
            Updated expert tokens, shaped `[B, S, C]`.
        """
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        # Cross-attention expects a mask broadcastable to `[B, num_heads, S, L]`.
        x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask.unsqueeze(1))
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build one expert's Q/K/V and post-attention modulation terms for a layer.

        Usage:
            Shared helper used by training, video-cache prefill, and action/joint inference.

        Args:
            expert: Video or action expert module. Present for call-site clarity; block owns the used layers.
            block: Current expert transformer block.
            x: Expert input tokens, shaped `[B, S, C]`.
            freqs: RoPE frequencies for the expert sequence. Shape depends on RoPE format.
            t_mod: Timestep modulation for the expert, shaped `[B, 6C]` or token-wise variant.

        Returns:
            Tuple `(q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)`.
            `q/k/v` are RoPE-applied attention tensors shaped `[B, S, C]`; the remaining tensors are used
            by `_apply_expert_post_block`.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        video_attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run the video expert once and collect per-layer K/V caches for action-only inference.

        Usage:
            Inference only; used by `WBWAM.infer_action`.

        Args:
            video_tokens: Prepared video tokens, shaped `[B, S_video, C_video]`.
            video_freqs: Video RoPE frequencies for `video_tokens`.
            video_t_mod: Video timestep modulation.
            video_context: Video cross-attention context, shaped `[B, L, D]`.
            video_context_mask: Prepared boolean video cross-attention mask, shaped `[B, S_video, L]`.
            video_attention_mask: Boolean video self-attention mask, shaped `[S_video, S_video]`.

        Returns:
            `(cache_k_list, cache_v_list)`, each a list of length `num_layers`; item `i` is the video
            key/value tensor for layer `i`, shaped `[B, S_video, C]`.
        """
        expert = self.mixtures["video"]
        x = video_tokens
        cache_k_list: list[torch.Tensor] = []
        cache_v_list: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            mixed = flash_attention(
                q=q,
                k=k,
                v=v,
                num_heads=self.num_heads,
                ctx_mask=video_attention_mask.to(device=q.device),
            )
            x = self._apply_expert_post_block(
                block=block,
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context=video_context,
                context_mask=video_context_mask,
            )
            cache_k_list.append(k)
            cache_v_list.append(v)
        return cache_k_list, cache_v_list

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        video_cache_k: Optional[list[torch.Tensor]],
        video_cache_v: Optional[list[torch.Tensor]],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise action tokens while attending to cached video K/V from each layer.

        Usage:
            Inference only; used by `WBWAM.infer_action` after video-cache prefill.

        Args:
            action_tokens: Prepared action tokens, shaped `[B, S_action, C_action]`.
            action_freqs: Action RoPE frequencies for `action_tokens`.
            action_t_mod: Action timestep modulation.
            action_context: Action cross-attention context, shaped `[B, L, D]`.
            action_context_mask: Prepared boolean action cross-attention mask, shaped `[B, S_action, L]`.
            video_cache_k: Per-layer cached video keys from `prefill_video_cache`.
            video_cache_v: Per-layer cached video values from `prefill_video_cache`.
            action_attention_mask: Boolean action-to-video/action attention mask, shaped
                `[S_action, S_video + S_action]`.
                Both cache lists may be `None` for action-only self-attention.

        Returns:
            Updated action tokens after all MoT layers, shaped `[B, S_action, C_action]`.
        """
        if (video_cache_k is None) != (video_cache_v is None):
            raise ValueError("`video_cache_k` and `video_cache_v` must both be set or both be None.")
        if video_cache_k is not None and (
            len(video_cache_k) != self.num_layers or len(video_cache_v) != self.num_layers
        ):
            raise ValueError("Video K/V cache lengths must match the number of MoT layers.")

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            if video_cache_k is None:
                k_cat = k_action
                v_cat = v_action
            else:
                k_cat = torch.cat([video_cache_k[layer_idx], k_action], dim=1)
                v_cat = torch.cat([video_cache_v[layer_idx], v_action], dim=1)
            mixed = flash_attention(
                q=q_action,
                k=k_cat,
                v=v_cat,
                num_heads=self.num_heads,
                ctx_mask=action_attention_mask.to(device=q_action.device),
            )
            x = self._apply_expert_post_block(
                block=block,
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context=action_context,
                context_mask=action_context_mask,
            )
        return x

    def forward_joint_core(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run video and action experts jointly with mixed self-attention at every layer.

        Usage:
            Shared core used by training and joint video/action inference.

        Args:
            video_tokens: Prepared video tokens, shaped `[B, S_video, C_video]`.
            action_tokens: Prepared action tokens, shaped `[B, S_action, C_action]`.
            video_freqs: Video RoPE frequencies for `video_tokens`.
            action_freqs: Action RoPE frequencies for `action_tokens`.
            video_t_mod: Video timestep modulation.
            action_t_mod: Action timestep modulation.
            video_context: Video cross-attention context, shaped `[B, L, D]`.
            video_context_mask: Prepared boolean video cross-attention mask, shaped `[B, S_video, L]`.
            action_context: Action cross-attention context, shaped `[B, L, D]`.
            action_context_mask: Prepared boolean action cross-attention mask, shaped `[B, S_action, L]`.
            attention_mask: Boolean mixed self-attention mask, shaped `[S_total, S_total]` for a shared
                batch mask or `[B, 1, S_total, S_total]` for per-sample masks.

        Returns:
            `(video_tokens, action_tokens)` after all MoT layers, shaped `[B, S_video, C_video]` and
            `[B, S_action, C_action]`.
        """
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        x_video = video_tokens
        x_action = action_tokens
        for layer_idx in range(self.num_layers):
            video_block = video_expert.blocks[layer_idx]
            action_block = action_expert.blocks[layer_idx]
            monitor = self._training_dynamics_monitor
            if monitor is not None:
                monitor.record_residual("video", layer_idx, x_video)
                monitor.record_residual("action", layer_idx, x_action)
            (
                q_video,
                k_video,
                v_video,
                residual_video,
                gate_msa_video,
                shift_mlp_video,
                scale_mlp_video,
                gate_mlp_video,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=video_block,
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            (
                q_action,
                k_action,
                v_action,
                residual_action,
                gate_msa_action,
                shift_mlp_action,
                scale_mlp_action,
                gate_mlp_action,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=action_block,
                x=x_action,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            q_cat = torch.cat([q_video, q_action], dim=1)
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = flash_attention(
                q=q_cat,
                k=k_cat,
                v=v_cat,
                num_heads=self.num_heads,
                ctx_mask=attention_mask.to(device=q_cat.device),
            )
            video_seq_len = x_video.shape[1]
            mixed_video = mixed[:, :video_seq_len, :]
            mixed_action = mixed[:, video_seq_len:, :]
            x_video = self._apply_expert_post_block(
                block=video_block,
                residual_x=residual_video,
                mixed_attn_out=mixed_video,
                gate_msa=gate_msa_video,
                shift_mlp=shift_mlp_video,
                scale_mlp=scale_mlp_video,
                gate_mlp=gate_mlp_video,
                context=video_context,
                context_mask=video_context_mask,
            )
            x_action = self._apply_expert_post_block(
                block=action_block,
                residual_x=residual_action,
                mixed_attn_out=mixed_action,
                gate_msa=gate_msa_action,
                shift_mlp=shift_mlp_action,
                scale_mlp=scale_mlp_action,
                gate_mlp=gate_mlp_action,
                context=action_context,
                context_mask=action_context_mask,
            )
        return x_video, x_action

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        """Dictionary-based wrapper for joint MoT forward.

        Args:
            embeds_all: Expert token dict with keys `"video"` and `"action"`.
            attention_mask: Boolean mixed attention mask, shaped `[S_total, S_total]`.
            freqs_all: Expert RoPE frequency dict.
            context_all: Expert context dict; each value contains `"context"` `[B, L, D]` and prepared
                `"mask"` `[B, S_expert, L]`.
            t_mod_all: Expert timestep modulation dict.

        Returns:
            This method always raises. Use `forward_joint_core`, `prefill_video_cache`, or
            `forward_action_with_video_cache` explicitly.
        """
        raise RuntimeError(
            "MoT.forward is intentionally disabled to avoid the unused dict wrapper path. "
            "Call forward_joint_core, prefill_video_cache, or forward_action_with_video_cache explicitly."
        )
