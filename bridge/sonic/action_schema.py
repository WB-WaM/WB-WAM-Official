from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

TOKEN_DIM = 64
WUJI_QPOS_DIM = 20
ACTION_DIM = TOKEN_DIM + 2 * WUJI_QPOS_DIM
FSQ_GRID_STEP = 1.0 / 16.0
WUJI_QPOS_LOWER_LIMITS = np.array(
    [
        0.0475,
        -0.1387,
        -0.4642,
        -0.4699,
        -0.1585,
        -0.3700,
        -0.4777,
        -0.4683,
        -0.1644,
        -0.3700,
        -0.4739,
        -0.4684,
        -0.1554,
        -0.3700,
        -0.4765,
        -0.4777,
        -0.1626,
        -0.3700,
        -0.4768,
        -0.4683,
    ],
    dtype=np.float32,
)
WUJI_QPOS_UPPER_LIMITS = np.array(
    [
        1.6033,
        0.9324,
        1.5623,
        1.5568,
        1.5604,
        0.3700,
        1.5485,
        1.5753,
        1.5516,
        0.3700,
        1.5512,
        1.5745,
        1.5585,
        0.3700,
        1.5487,
        1.5634,
        1.5585,
        0.3700,
        1.5490,
        1.5735,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class SonicAction:
    token_state: np.ndarray
    left_wuji_qpos: np.ndarray
    right_wuji_qpos: np.ndarray


def _as_f32_vector(value: Sequence[float] | np.ndarray, *, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"{name} has dim {arr.size}, expected {dim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def normalize_action_chunk(actions: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return an action chunk with shape [horizon, 104]."""

    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"actions must be [104] or [H, 104], got shape {arr.shape}")
    if arr.shape[1] != ACTION_DIM:
        raise ValueError(f"action dim {arr.shape[1]}, expected {ACTION_DIM}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("actions contain NaN/Inf")
    return arr


def apply_wuji_qpos_limits(qpos: Sequence[float] | np.ndarray) -> np.ndarray:
    """Clip an absolute Wuji command to the per-joint URDF limits."""

    arr = _as_f32_vector(qpos, dim=WUJI_QPOS_DIM, name="wuji_qpos").copy()
    return np.clip(arr, WUJI_QPOS_LOWER_LIMITS, WUJI_QPOS_UPPER_LIMITS).astype(np.float32, copy=False)


def snap_token_to_fsq_grid(
    token: Sequence[float] | np.ndarray,
    *,
    step: float = FSQ_GRID_STEP,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    arr = _as_f32_vector(token, dim=TOKEN_DIM, name="token_state")
    snapped = np.round(arr / step) * step
    if minimum is not None or maximum is not None:
        lo = -np.inf if minimum is None else minimum
        hi = np.inf if maximum is None else maximum
        snapped = np.clip(snapped, lo, hi)
    return snapped.astype(np.float32, copy=False)


def split_action_step(
    action: Sequence[float] | np.ndarray,
    *,
    snap_token_grid: bool = False,
    token_min: float | None = None,
    token_max: float | None = None,
) -> SonicAction:
    arr = _as_f32_vector(action, dim=ACTION_DIM, name="action")
    token = arr[:TOKEN_DIM]
    if snap_token_grid:
        token = snap_token_to_fsq_grid(token, minimum=token_min, maximum=token_max)
    elif token_min is not None or token_max is not None:
        lo = -np.inf if token_min is None else token_min
        hi = np.inf if token_max is None else token_max
        token = np.clip(token, lo, hi).astype(np.float32, copy=False)

    left = apply_wuji_qpos_limits(arr[TOKEN_DIM : TOKEN_DIM + WUJI_QPOS_DIM])
    right = apply_wuji_qpos_limits(arr[TOKEN_DIM + WUJI_QPOS_DIM :])
    return SonicAction(
        token_state=np.asarray(token, dtype=np.float32),
        left_wuji_qpos=left,
        right_wuji_qpos=right,
    )


def split_action_chunk(
    actions: np.ndarray | Sequence[Sequence[float]],
    *,
    snap_token_grid: bool = False,
    token_min: float | None = None,
    token_max: float | None = None,
) -> list[SonicAction]:
    chunk = normalize_action_chunk(actions)
    return [
        split_action_step(
            step,
            snap_token_grid=snap_token_grid,
            token_min=token_min,
            token_max=token_max,
        )
        for step in chunk
    ]
