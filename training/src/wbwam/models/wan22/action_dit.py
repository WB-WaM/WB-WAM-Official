import os
import torch
import torch.nn as nn
from typing import Any, Optional

from wbwam.utils.logging_config import get_logger

from .wan_video_dit import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
    precompute_freqs_sincos,
)

logger = get_logger(__name__)


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
        rope_format: str = "complex",
    ):
        super().__init__()
        if rope_format not in ("complex", "sincos"):
            raise ValueError(f"`rope_format` must be 'complex' or 'sincos', got {rope_format!r}.")
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.rope_format = rope_format

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs_complex = precompute_freqs_cis(attn_head_dim, end=1024)
        self.freqs_sincos = precompute_freqs_sincos(attn_head_dim, end=1024)

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )
        
        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    @property
    def max_rope_len(self) -> int:
        return int(self.freqs_complex.shape[0])

    def _apply(self, fn):
        result = super()._apply(fn)
        device = next(self.parameters()).device
        self.freqs_complex = self.freqs_complex.to(device=device)
        self.freqs_sincos = self.freqs_sincos.to(device=device)
        return result

    def get_freqs(self, seq_len: int) -> torch.Tensor:
        if self.rope_format == "sincos":
            return self.freqs_sincos[:seq_len].view(seq_len, 1, -1, 2)
        return self.freqs_complex[:seq_len].view(seq_len, 1, -1)

    def prepare(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = action_tokens.shape[1]
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        tokens = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.get_freqs(seq_len=seq_len)
        return tokens, t, t_mod, context_emb, context_attn_mask, freqs

    def post(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, _t, t_mod, context_emb, context_attn_mask, freqs = self.prepare(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )

        for block in self.blocks:
            x = block(x, context_emb, t_mod, freqs, context_mask=context_attn_mask)

        return self.post(x)
