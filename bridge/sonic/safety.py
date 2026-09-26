from __future__ import annotations

import numpy as np


def ensure_finite_vector(value, *, name: str, dim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if dim is not None and arr.size != dim:
        raise ValueError(f"{name} dim {arr.size}, expected {dim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr
