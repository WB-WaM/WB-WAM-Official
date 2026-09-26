from __future__ import annotations

from typing import Any

import numpy as np

STATE_LAYOUT_QUAT = "quat"
STATE_LAYOUT_GRAVITY = "gravity"
STATE_DIM_QUAT = 108
STATE_DIM_GRAVITY = 107


def normalize_state_layout(layout: str | None, *, state_dim: int | None = None) -> str:
    if layout:
        normalized = str(layout).strip().lower()
        if normalized in {STATE_LAYOUT_QUAT, "quaternion", "base_quat"}:
            return STATE_LAYOUT_QUAT
        if normalized in {STATE_LAYOUT_GRAVITY, "base_gravity"}:
            return STATE_LAYOUT_GRAVITY
        raise ValueError(f"unsupported state_layout={layout!r}")
    if state_dim == STATE_DIM_GRAVITY:
        return STATE_LAYOUT_GRAVITY
    return STATE_LAYOUT_QUAT


def state_dim_for_layout(layout: str | None) -> int:
    normalized = normalize_state_layout(layout)
    if normalized == STATE_LAYOUT_GRAVITY:
        return STATE_DIM_GRAVITY
    return STATE_DIM_QUAT


def base_quat_to_gravity(base_quat: Any) -> np.ndarray:
    """Convert a wxyz base quaternion to gravity direction in the base frame."""

    quat = np.asarray(base_quat, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"base_quat has dim {quat.size}, expected 4")
    if not np.all(np.isfinite(quat)):
        raise ValueError("base_quat contains NaN/Inf")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("base_quat has near-zero norm")
    qw, qx, qy, qz = quat / norm
    return np.asarray(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )
