"""Normalization utilities for WB state and action tensors."""

from collections import defaultdict
import json
from typing import Annotated, Any, Dict, Literal, Tuple, Union

import numpy as np
import torch

from wbwam.utils.logging_config import get_logger
from wbwam.utils.pytorch_utils import dict_apply

logger = get_logger(__name__)

ConstConstStr = Annotated[
    str, "format: 'const_min/const_max', where const_min and const_max give the constant range"
]
NormMode = Union[
    Literal[
        "dummy",
        "raw",
        "min/max",
        "q01/q99",
        "q001/q999",
        "q0001/q9999",
        "q00001/q99999",
        "z-score",
        "z-score-tail",  # fused: tail_compress -> z-score
        "q01/q99-tail",  # fused: tail_compress -> q01/q99 linear
        "tanh",
    ],
    ConstConstStr,
]


class SingleFieldLinearNormalizer:
    """
    Single-field linear normalizer with forward/backward transforms.

    Forward: normalize data according to mode (z-score, min/max, quantile, etc.)
    Backward: denormalize data

    This transform is invertible (within the valid range).
    """

    invertible = True
    std_reg = 1e-8
    tanh_eps = 1e-7
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0

    def __init__(
        self,
        stats,
        mode: NormMode = "min/max",
        dummy_clip_range: Tuple[float, float] = (-5.0, 5.0),
        tail_scale: float = 0.075,
    ):
        self.stats = stats
        self.mode = mode
        self.dummy_clip_min, self.dummy_clip_max = dummy_clip_range
        self._horizon_mismatch_warned = False

        # Tail-compress buffers; non-None only for *-tail modes.
        self._tail_q01 = None
        self._tail_q99 = None
        self._tail_c_pos = None
        self._tail_c_neg = None

        if mode in ("dummy", "raw"):
            self.scale = None
            self.offset = None
            return

        is_tail = mode.endswith("-tail")
        base_mode = mode[:-5] if is_tail else mode

        if base_mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            # Detect near-constant dimensions (std too small) to avoid
            # amplifying noise by 1/std.
            ignore_dim = input_std < self.range_tol
            scale = 1.0 / (input_std + self.std_reg)
            offset = -input_mean / (input_std + self.std_reg)
            # For near-constant dims: scale=1, offset=-mean -> output ~= 0.
            scale[ignore_dim] = 1.0
            offset[ignore_dim] = -input_mean[ignore_dim]
        else:
            if base_mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            elif base_mode in ("q01/q99", "tanh"):
                input_min, input_max = stats["q01"], stats["q99"]
            elif base_mode == "q001/q999":
                input_min, input_max = stats["q001"], stats["q999"]
            elif base_mode == "q0001/q9999":
                input_min, input_max = stats["q0001"], stats["q9999"]
            elif base_mode == "q00001/q99999":
                input_min, input_max = stats["q00001"], stats["q99999"]
            else:
                input_min, input_max = map(float, base_mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)

            input_range = input_max - input_min
            ignore_dim = input_range < self.range_tol
            input_range[ignore_dim] = self.output_max - self.output_min
            scale = (self.output_max - self.output_min) / input_range
            offset = self.output_min - scale * input_min
            offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]

        self.scale = scale
        self.offset = offset

        if is_tail:
            q01 = stats["q01"]
            q99 = stats["q99"]
            mean = stats["mean"]
            # For degenerate dims (q01 >= q99 or mean outside [q01, q99]),
            # disable tail by using identity-safe parameters.
            degenerate = (q99 <= q01) | (mean <= q01) | (mean >= q99)
            c_pos = tail_scale * (q99 - mean)
            c_neg = tail_scale * (mean - q01)
            # Ensure c > 0 to avoid div-by-zero in log1p / expm1.
            c_pos = torch.where(degenerate, torch.ones_like(c_pos), c_pos)
            c_neg = torch.where(degenerate, torch.ones_like(c_neg), c_neg)
            self._tail_q01 = q01
            self._tail_q99 = q99
            self._tail_c_pos = c_pos
            self._tail_c_neg = c_neg

    def get_stats(self):
        return self.stats

    def _apply_tail_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise log1p tail compression; identity on [q01, q99]."""
        q01 = self._tail_q01.to(x.device)
        q99 = self._tail_q99.to(x.device)
        c_pos = self._tail_c_pos.to(x.device)
        c_neg = self._tail_c_neg.to(x.device)
        pos_tail = q99 + c_pos * torch.log1p(torch.clamp((x - q99) / c_pos, min=0.0))
        neg_tail = q01 - c_neg * torch.log1p(torch.clamp((q01 - x) / c_neg, min=0.0))
        return torch.where(x > q99, pos_tail, torch.where(x < q01, neg_tail, x))

    def _apply_tail_backward(self, y: torch.Tensor) -> torch.Tensor:
        """Exact inverse of _apply_tail_forward."""
        q01 = self._tail_q01.to(y.device)
        q99 = self._tail_q99.to(y.device)
        c_pos = self._tail_c_pos.to(y.device)
        c_neg = self._tail_c_neg.to(y.device)
        pos_tail = q99 + c_pos * torch.expm1(torch.clamp((y - q99) / c_pos, min=0.0))
        neg_tail = q01 - c_neg * torch.expm1(torch.clamp((q01 - y) / c_neg, min=0.0))
        return torch.where(y > q99, pos_tail, torch.where(y < q01, neg_tail, y))

    def _match_horizon(self, x: torch.Tensor):
        """Slice scale/offset by horizon when stats horizon > input horizon."""
        scale, offset = self.scale, self.offset
        if scale.ndim < 2:
            return scale, offset

        h_stats, h_data = scale.shape[-2], x.shape[-2]
        if h_data == h_stats:
            return scale, offset

        if h_data > h_stats:
            raise ValueError(
                f"Data horizon ({h_data}) exceeds stats horizon ({h_stats}). "
                "Cannot normalize without sufficient statistics."
            )

        if not self._horizon_mismatch_warned:
            logger.warning(
                "Normalizer horizon mismatch: stats have %s steps but data has %s. Using first %s steps of stats.",
                h_stats,
                h_data,
                h_data,
            )
            self._horizon_mismatch_warned = True

        return scale[..., :h_data, :], offset[..., :h_data, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "raw":
            return x
        if self.mode == "dummy":
            return x.clamp(self.dummy_clip_min, self.dummy_clip_max)

        scale, offset = self._match_horizon(x)
        feat_dim = scale.shape[-1]
        x_main, x_pad = x[..., :feat_dim], x[..., feat_dim:]

        if self._tail_c_pos is not None:
            x_main = self._apply_tail_forward(x_main)

        x_main = x_main * scale + offset
        if self.mode == "tanh":
            x_main = torch.tanh(x_main)
        else:
            # Clamp range-based modes to avoid extreme outliers.
            x_main = x_main.clamp(-5.0, 5.0)

        return torch.cat([x_main, x_pad], dim=-1) if x_pad.shape[-1] > 0 else x_main

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in ("dummy", "raw"):
            return x

        scale, offset = self._match_horizon(x)
        feat_dim = scale.shape[-1]
        x_main, x_pad = x[..., :feat_dim], x[..., feat_dim:]

        if self.mode == "tanh":
            x_main = x_main.clamp(-1.0 + self.tanh_eps, 1.0 - self.tanh_eps)
            x_main = torch.atanh(x_main)

        x_main = (x_main - offset) / scale

        if self._tail_c_pos is not None:
            x_main = self._apply_tail_backward(x_main)

        return torch.cat([x_main, x_pad], dim=-1) if x_pad.shape[-1] > 0 else x_main


def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):

    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (defaultdict, dict)):
            return {k: convert_tensor(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tensor(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serializable_stats = convert_tensor(dataset_stats)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)


def load_dataset_stats_from_json(file_path: str, try_convert_tensor: bool = True) -> Dict[str, Any]:

    def is_numeric_list(obj):
        if isinstance(obj, list):
            if not obj:
                return True
            first = obj[0]
            if isinstance(first, (int, float)):
                return all(isinstance(x, (int, float)) for x in obj)
            elif isinstance(first, list):
                return all(is_numeric_list(item) for item in obj)
            else:
                return False
        return False

    def convert_back_to_tensor(obj):
        if isinstance(obj, dict):
            return {k: convert_back_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if is_numeric_list(obj):
                try:
                    arr = np.array(obj)
                    return torch.from_numpy(arr)
                except Exception:
                    return [convert_back_to_tensor(item) for item in obj]
            else:
                return [convert_back_to_tensor(item) for item in obj]
        else:
            return obj

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if try_convert_tensor:
        data = convert_back_to_tensor(data)

    data = dict_apply(
        data,
        lambda x: x.to(torch.float32) if isinstance(x, torch.Tensor) else x,
    )

    return data
