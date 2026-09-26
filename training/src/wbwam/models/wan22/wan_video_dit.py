import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .wan_video_camera_controller import SimpleAdapter

from wbwam.utils.logging_config import get_logger

logger = get_logger(__name__)

_VIDEO_ROPE_CACHE_NAMES = (
    "freqs_complex_f",
    "freqs_complex_h",
    "freqs_complex_w",
    "freqs_sincos_f",
    "freqs_sincos_h",
    "freqs_sincos_w",
)

# try:
#     import flash_attn_interface
#     FLASH_ATTN_3_AVAILABLE = True
# except ModuleNotFoundError:
#     FLASH_ATTN_3_AVAILABLE = False

# try:
#     import flash_attn
#     FLASH_ATTN_2_AVAILABLE = True
# except ModuleNotFoundError:
#     FLASH_ATTN_2_AVAILABLE = False

# try:
#     from sageattention import sageattn
#     SAGE_ATTN_AVAILABLE = True
# except ModuleNotFoundError:
#     SAGE_ATTN_AVAILABLE = False


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None, compatibility_mode=True):
    """Run scaled dot-product attention over flattened multi-head projections.

    Args:
        q: Query tensor shaped `[B, S_q, num_heads * head_dim]`.
        k: Key tensor shaped `[B, S_k, num_heads * head_dim]`.
        v: Value tensor shaped `[B, S_k, num_heads * head_dim]`.
        num_heads: Number of attention heads.
        ctx_mask: Optional attention mask broadcastable to `[B, num_heads, S_q, S_k]`.
        compatibility_mode: Must remain true; non-compatibility kernels are disabled.

    Returns:
        Attention output shaped `[B, S_q, num_heads * head_dim]`.
    """
    if compatibility_mode:
        B, S_q, _ = q.shape
        S_k = k.shape[1]
        head_dim = q.shape[2] // num_heads
        q = q.view(B, S_q, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S_k, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, S_k, num_heads, head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = x.transpose(1, 2).reshape(B, S_q, -1)
        return x
    else:
        raise NotImplementedError("Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True.")
    # elif FLASH_ATTN_3_AVAILABLE:
    #     q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
    #     k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
    #     v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
    #     x = flash_attn_interface.flash_attn_func(q, k, v)
    #     if isinstance(x,tuple):
    #         x = x[0]
    #     x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    # elif FLASH_ATTN_2_AVAILABLE:
    #     q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
    #     k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
    #     v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
    #     x = flash_attn.flash_attn_func(q, k, v)
    #     x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    # elif SAGE_ATTN_AVAILABLE:
    #     q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    #     k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    #     v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    #     x = sageattn(q, k, v)
    #     x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    # else:
    #     q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    #     k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    #     v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    #     x = F.scaled_dot_product_attention(q, k, v)
    #     x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    # return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """Apply adaptive affine modulation to token activations.

    Args:
        x: Activation tensor.
        shift: Additive modulation tensor broadcastable to `x`.
        scale: Multiplicative modulation tensor broadcastable to `x`.

    Returns:
        Modulated tensor with the same shape as `x`.
    """
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    """Create 1D sinusoidal timestep or position embeddings.

    Args:
        dim: Output embedding dimension.
        position: Position/timestep tensor shaped `[N]` or flattened positions.

    Returns:
        Tensor shaped `[position.numel(), dim]` in the input position dtype.
    """
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute complex RoPE frequencies for frame, height, and width axes.

    Args:
        dim: Per-head RoPE dimension.
        end: Maximum index per axis.
        theta: RoPE base.

    Returns:
        Tuple of complex tensors `(freq_f, freq_h, freq_w)`.
    """
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_sincos_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute sin/cos RoPE frequencies for frame, height, and width axes.

    Args:
        dim: Per-head RoPE dimension.
        end: Maximum index per axis.
        theta: RoPE base.

    Returns:
        Tuple of real tensors with trailing `[cos, sin]` pairs.
    """
    f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(dim, end=end, theta=theta)
    return freqs_cis_to_sincos(f_freqs), freqs_cis_to_sincos(h_freqs), freqs_cis_to_sincos(w_freqs)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute complex 1D RoPE frequencies.

    Args:
        dim: RoPE dimension.
        end: Maximum sequence index.
        theta: RoPE base.

    Returns:
        Complex tensor shaped `[end, dim // 2]`.
    """
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def freqs_cis_to_sincos(freqs: torch.Tensor) -> torch.Tensor:
    """Convert complex RoPE frequencies into explicit cos/sin pairs.

    Args:
        freqs: Complex RoPE tensor.

    Returns:
        Real tensor with one extra trailing dimension `[real, imag]`.
    """
    if not torch.is_complex(freqs):
        raise ValueError("`freqs` must be a complex RoPE tensor.")
    return torch.stack((freqs.real, freqs.imag), dim=-1).contiguous()


def precompute_freqs_sincos(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute 1D RoPE frequencies in explicit cos/sin format.

    Args:
        dim: RoPE dimension.
        end: Maximum sequence index.
        theta: RoPE base.

    Returns:
        Real tensor shaped `[end, dim // 2, 2]`.
    """
    return freqs_cis_to_sincos(precompute_freqs_cis(dim, end=end, theta=theta))


def rope_apply_complex(x, freqs, num_heads):
    """Apply complex RoPE to flattened query/key projections.

    Args:
        x: Projection tensor shaped `[B, S, num_heads * head_dim]`.
        freqs: Complex RoPE frequencies shaped `[S, 1, head_dim // 2]`.
        num_heads: Number of attention heads.

    Returns:
        RoPE-rotated tensor with the same shape and dtype as `x`.
    """
    B, S, _ = x.shape
    x = x.view(B, S, num_heads, -1)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(B, S, x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def rope_apply_sincos(x, freqs, num_heads):
    """Apply sin/cos RoPE to flattened query/key projections.

    Args:
        x: Projection tensor shaped `[B, S, num_heads * head_dim]`.
        freqs: Real RoPE frequencies shaped `[S, 1, head_dim // 2, 2]`.
        num_heads: Number of attention heads.

    Returns:
        RoPE-rotated tensor with the same shape and dtype as `x`.
    """
    B, S, _ = x.shape
    out_dtype = x.dtype
    x_pair = x.view(B, S, num_heads, -1, 2).to(torch.float32)
    freqs = freqs.to(device=x.device, dtype=torch.float32)
    cos, sin = freqs.unbind(dim=-1)
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    x0, x1 = x_pair.unbind(dim=-1)
    x_out = torch.empty_like(x_pair)
    x_out[..., 0] = x0 * cos - x1 * sin
    x_out[..., 1] = x1 * cos + x0 * sin
    return x_out.reshape(B, S, -1).to(out_dtype)


def rope_apply(x, freqs, num_heads):
    """Dispatch RoPE application based on complex vs sin/cos frequency format.

    Args:
        x: Projection tensor shaped `[B, S, num_heads * head_dim]`.
        freqs: Complex or sin/cos RoPE frequencies.
        num_heads: Number of attention heads.

    Returns:
        RoPE-rotated tensor with the same shape as `x`.
    """
    if torch.is_complex(freqs):
        return rope_apply_complex(x, freqs, num_heads)
    return rope_apply_sincos(x, freqs, num_heads)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned scale."""

    def __init__(self, dim, eps=1e-5):
        """Create RMSNorm.

        Args:
            dim: Feature dimension.
            eps: Numerical stability epsilon.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        """Normalize activations by their RMS value.

        Args:
            x: Input tensor with feature dimension last.

        Returns:
            RMS-normalized tensor with the same shape as `x`.
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Apply RMS normalization and learned scale.

        Args:
            x: Input tensor with feature dimension last.

        Returns:
            Normalized tensor with the same shape as `x`.
        """
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    """Thin module wrapper around `flash_attention`."""

    def __init__(self, num_heads):
        """Create an attention wrapper.

        Args:
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, ctx_mask=None):
        """Run attention over flattened multi-head projections.

        Args:
            q: Query tensor shaped `[B, S_q, C]`.
            k: Key tensor shaped `[B, S_k, C]`.
            v: Value tensor shaped `[B, S_k, C]`.
            ctx_mask: Optional attention mask.

        Returns:
            Attention output shaped `[B, S_q, C]`.
        """
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return x


class SelfAttention(nn.Module):
    """Self-attention block with RMS-normalized Q/K and RoPE."""

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        """Create video-token self-attention.

        Args:
            hidden_dim: Token hidden size.
            attn_head_dim: Per-head attention dimension.
            num_heads: Number of heads.
            eps: Normalization epsilon.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        """Apply RoPE self-attention to token embeddings.

        Args:
            x: Token tensor shaped `[B, S, hidden_dim]`.
            freqs: RoPE frequencies for the sequence.
            self_attn_mask: Optional self-attention mask.

        Returns:
            Tensor shaped `[B, S, hidden_dim]`.
        """
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    """Cross-attention from video tokens to text/action context tokens."""

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,):
        """Create cross-attention.

        Args:
            hidden_dim: Token hidden size.
            attn_head_dim: Per-head attention dimension.
            num_heads: Number of heads.
            eps: Normalization epsilon.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        """Attend query tokens to context tokens.

        Args:
            x: Query tokens shaped `[B, S, hidden_dim]`.
            ctx: Context tokens shaped `[B, L, hidden_dim]`.
            ctx_mask: Optional attention mask for context access.

        Returns:
            Tensor shaped `[B, S, hidden_dim]`.
        """
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    """Residual gate used by adaptive DiT blocks."""

    def __init__(self,):
        """Create a residual gate module."""
        super().__init__()

    def forward(self, x, gate, residual):
        """Apply gated residual update.

        Args:
            x: Current tensor.
            gate: Gate tensor broadcastable to `residual`.
            residual: Residual update tensor.

        Returns:
            `x + gate * residual`.
        """
        return x + gate * residual

class DiTBlock(nn.Module):
    """Wan video DiT transformer block with self-attention, cross-attention, and MLP."""

    def __init__(self,  hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        """Create one DiT block.

        Args:
            hidden_dim: Token hidden size.
            attn_head_dim: Per-head attention dimension.
            num_heads: Number of attention heads.
            ffn_dim: Feed-forward hidden dimension.
            eps: Normalization epsilon.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor] = None):
        """Run one DiT block over prepared video tokens.

        Args:
            x: Video tokens shaped `[B, S, hidden_dim]`.
            context: Context tokens shaped `[B, L, hidden_dim]`.
            t_mod: Timestep modulation, either batch-wise or token-wise.
            freqs: RoPE frequencies for `x`.
            context_mask: Optional cross-attention mask.
            self_attn_mask: Optional self-attention mask.

        Returns:
            Updated video tokens shaped `[B, S, hidden_dim]`.
        """
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1) # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    """Small projection MLP used for context/action embeddings."""

    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        """Create the projection MLP.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            has_pos_emb: Whether to add a learned absolute positional embedding.
        """
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        """Project input features.

        Args:
            x: Input tensor shaped `[B, L, in_dim]`.

        Returns:
            Tensor shaped `[B, L, out_dim]`.
        """
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    """Final DiT head that maps video tokens back to latent patches."""

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        """Create the unpatchifying prediction head.

        Args:
            dim: Token hidden size.
            out_dim: Output latent channels.
            patch_size: 3D patch size `(T, H, W)`.
            eps: Normalization epsilon.
        """
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        """Project modulated tokens to flattened latent patches.

        Args:
            x: Tokens shaped `[B, S, dim]`.
            t_mod: Timestep modulation from `WanVideoDiT.prepare`.

        Returns:
            Flattened patch predictions shaped `[B, S, out_dim * prod(patch_size)]`.
        """
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanVideoDiT(torch.nn.Module):
    """Wan video DiT expert that prepares latent video tokens and posts token predictions."""

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode = "causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
        rope_format: str = "complex",
    ):
        """Create a Wan video DiT expert.

        Args:
            hidden_dim: Transformer token hidden size.
            in_dim: Input latent channel count.
            ffn_dim: Feed-forward hidden dimension in each block.
            out_dim: Output latent channel count.
            text_dim: Input context/text feature dimension.
            freq_dim: Sinusoidal timestep embedding dimension.
            eps: Normalization epsilon.
            patch_size: 3D latent patch size `(T, H, W)`.
            num_heads: Attention head count.
            attn_head_dim: Per-head attention dimension.
            num_layers: Number of DiT blocks.
            has_image_input: Legacy flag; current path requires false.
            has_image_pos_emb: Whether image positional embedding is configured.
            has_ref_conv: Whether to create reference convolution.
            add_control_adapter: Whether to create camera/control adapter.
            in_dim_control_adapter: Control adapter input channel count.
            seperated_timestep: Whether to create token-wise timestep modulation.
            require_vae_embedding: Legacy flag; current fused-latent path requires false.
            require_clip_embedding: Legacy flag; current path requires false.
            fuse_vae_embedding_in_latents: Whether first-frame VAE information is fused in latents.
            action_conditioned: Whether video context includes action tokens.
            action_dim: Action feature dimension when action conditioning is enabled.
            action_group_causal_mask_mode: Action-to-video grouping mask mode.
            video_attention_mask_mode: Video self-attention mask mode.
            use_gradient_checkpointing: Legacy config flag; not used by the current explicit paths.
            rope_format: `"complex"` or `"sincos"` RoPE cache format.
        """
        super().__init__()
        if rope_format not in ("complex", "sincos"):
            raise ValueError(f"`rope_format` must be 'complex' or 'sincos', got {rope_format!r}.")
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.rope_format = rope_format

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )

        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        assert has_image_input == False
        assert require_clip_embedding == False
        assert require_vae_embedding == False and fuse_vae_embedding_in_latents == True, "Only support fusing vae embedding in latents"

        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        freqs_complex = precompute_freqs_cis_3d(attn_head_dim)
        freqs_sincos = precompute_freqs_sincos_3d(attn_head_dim)
        self.freqs_complex_f = freqs_complex[0]
        self.freqs_complex_h = freqs_complex[1]
        self.freqs_complex_w = freqs_complex[2]
        self.freqs_sincos_f = freqs_sincos[0]
        self.freqs_sincos_h = freqs_sincos[1]
        self.freqs_sincos_w = freqs_sincos[2]
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, hidden_dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        """Convert latent video tensors into patch-grid features.

        Usage:
            Shared tensor path used by `prepare`.

        Args:
            x: Latent video tensor shaped `[B, C, T, H, W]`.
            control_camera_latents_input: Optional camera/control latent tensor for the control adapter.

        Returns:
            Patch-grid tensor shaped `[B, hidden_dim, T_patch, H_patch, W_patch]`.
        """
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def _apply(self, fn):
        """Apply module device/dtype moves while keeping RoPE caches on the module device.

        Args:
            fn: Function supplied by `torch.nn.Module._apply`.

        Returns:
            Result of the superclass `_apply` call.
        """
        result = super()._apply(fn)
        device = next(self.parameters()).device
        for name in _VIDEO_ROPE_CACHE_NAMES:
            setattr(self, name, getattr(self, name).to(device=device))
        return result

    def get_freqs(self, f: int, h: int, w: int) -> torch.Tensor:
        """Assemble 3D RoPE frequencies for a patch grid.

        Usage:
            Shared tensor helper used by `prepare`.

        Args:
            f: Number of temporal patch positions.
            h: Number of height patch positions.
            w: Number of width patch positions.

        Returns:
            RoPE tensor shaped `[f * h * w, 1, D]` for complex format or `[f * h * w, 1, D, 2]`
            for sin/cos format.
        """
        if self.rope_format == "sincos":
            return torch.cat([
                self.freqs_sincos_f[:f].view(f, 1, 1, -1, 2).expand(f, h, w, -1, -1),
                self.freqs_sincos_h[:h].view(1, h, 1, -1, 2).expand(f, h, w, -1, -1),
                self.freqs_sincos_w[:w].view(1, 1, w, -1, 2).expand(f, h, w, -1, -1)
            ], dim=-2).reshape(f * h * w, 1, -1, 2)

        return torch.cat([
            self.freqs_complex_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs_complex_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs_complex_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        """Reconstruct latent video tensors from flattened patch predictions.

        Args:
            x: Flattened patch predictions shaped `[B, F*H*W, C * prod(patch_size)]`.
            grid_size: Patch grid size `(F, H, W)`.

        Returns:
            Latent video tensor shaped `[B, C, F*T_patch, H*H_patch, W*W_patch]`.
        """
        f, h, w = grid_size[0], grid_size[1], grid_size[2]
        px, py, pz = self.patch_size[0], self.patch_size[1], self.patch_size[2]
        B = x.shape[0]
        c = x.shape[-1] // (px * py * pz)
        x = x.view(B, f, h, w, px, py, pz, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, c, f * px, h * py, w * pz)
        return x

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate legacy public forward inputs.

        Args:
            x: Latent video tensor shaped `[B, C, T, H, W]`.
            timestep: Timestep tensor shaped `[B]` or `[1]`.
            context: Context tensor shaped `[B, L, text_dim]`.
            context_mask: Optional context mask shaped `[B, L]`.
            action: Optional action tensor shaped `[B, T_action, action_dim]`.

        Returns:
            Validated `(x, timestep, context_mask)`, possibly expanding inference batch dimensions.
        """
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            allow_text_only_single_frame = (num_latent_frames == 1 and action is None)
            if not allow_text_only_single_frame:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}")
                if action.shape[2] != self.action_dim:
                    raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
                if num_latent_frames <= 1:
                    raise ValueError(f"video length must be > 1 for action-conditioned model, got {num_latent_frames}")
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        f"action horizon must be divisible by (num_latent_frames - 1), got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}")

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, "During training, timestep length must match batch_size."
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the video self-attention mask for the configured mask mode.

        Usage:
            Shared by training/inference MoT mask builders and video-only denoise.

        Args:
            video_seq_len: Number of video tokens.
            video_tokens_per_frame: Number of tokens corresponding to one latent frame.
            device: Device for the returned mask.

        Returns:
            Boolean mask shaped `[video_seq_len, video_seq_len]`.
        """
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def prepare(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
        """Prepare video latents for DiT/MoT block execution.

        Usage:
            Tensor-only core used by training and inference compiled denoise paths. Callers must validate
            shapes and masks before entering this function.

        Args:
            x: Latent video tensor shaped `[B, C, T, H, W]`.
            timestep: Timestep tensor shaped `[B]`.
            context: Context tensor shaped `[B, L, text_dim]`.
            context_mask: Boolean context mask shaped `[B, L]`.
            action: Optional action tensor used for action-conditioned video context.
            fuse_vae_embedding_in_latents: Whether the first latent frame has fused VAE conditioning.
            control_camera_latents_input: Optional camera/control latent tensor.

        Returns:
            Tuple `(x_tokens, t, t_mod, context, context_mask, freqs, f, h, w, tokens_per_frame)` for
            explicit DiT/MoT execution.
        """
        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        token_timesteps = torch.ones(
            (batch_size, x.shape[2], tokens_per_frame),
            dtype=timestep.dtype,
            device=timestep.device,
        ) * timestep.view(batch_size, 1, 1)
        token_timesteps[:, 0, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]

        context = self.text_embedding(context) # (B, L, dim)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action) # (B, action_len, dim)
            action_pos_embed = sinusoidal_embedding_1d(self.hidden_dim,
                torch.arange(action_len, device=action_emb.device)) # (action_len, dim)
            action_emb = action_emb + action_pos_embed.unsqueeze(0) # (B, action_len, dim)
            context = torch.cat([context, action_emb], dim=1) # (B, context_len + action_len, dim)

            # new mask
            num_temporal_groups = f - 1 # first latent frame do not attend to actions
            # Each latent frame (from the 2nd one) attends to the corresponding group of action tokens
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device) # ((f-1)*tokens_per_frame, action_len)

            seq_len = f * h * w # query length
            final_context_mask = torch.zeros((batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device) # (B, seq_len, L + action_len)
            # all latent frames attend to text tokens
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(-1, seq_len, -1) # (B, seq_len, L)
            # latent frames from the 2nd one attend to action tokens
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B, seq_len, action_len)
            context_mask = final_context_mask
        elif self.action_conditioned and action is None:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)
        else:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)

        x_tokens = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], -1, x.shape[1]).contiguous()

        freqs = self.get_freqs(f=f, h=h, w=w)

        return x_tokens, t, t_mod, context, context_mask, freqs, f, h, w, tokens_per_frame

    def post(self, x_tokens: torch.Tensor, t: torch.Tensor, f: int, h: int, w: int) -> torch.Tensor:
        """Convert output tokens back to latent video prediction.

        Usage:
            Shared by training and inference denoise cores after block execution.

        Args:
            x_tokens: Output tokens shaped `[B, f*h*w, hidden_dim]`.
            t: Timestep embedding returned by `prepare`.
            f: Temporal patch-grid size.
            h: Height patch-grid size.
            w: Width patch-grid size.

        Returns:
            Latent prediction tensor shaped `[B, out_dim, T, H, W]`.
        """
        x = self.head(x_tokens, t)
        return self.unpatchify(x, (f, h, w))

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        """Fail fast for the disabled legacy public forward path.

        Args:
            x: Unused latent video tensor.
            timestep: Unused timestep tensor.
            context: Unused context tensor.
            context_mask: Unused context mask.
            action: Unused action tensor.
            fuse_vae_embedding_in_latents: Unused fuse flag.

        Raises:
            RuntimeError: Always. Use `prepare`/`post` plus explicit block or MoT execution instead.
        """
        raise RuntimeError(
            "WanVideoDiT.forward is intentionally disabled. "
            "Use prepare/post plus the explicit MoT or block-level path instead."
        )
