from __future__ import annotations

import numpy as np

from .motion_schema import G1MotionChunk

SONIC_ENCODER_INPUT_DIM = 1762
ENCODER_INDEX_DIM = 4
NUM_FUTURE_FRAMES = 10
G1_FUTURE_DT_S = 0.1
Q_SLICE = slice(4, 294)
DQ_SLICE = slice(294, 584)
ROOT_ORIENTATION_6D_SLICE = slice(601, 661)


def full_future_action_horizon(
    *,
    horizon: int,
    timestep_s: float,
    future_dt_s: float = G1_FUTURE_DT_S,
) -> int:
    """Number of action rows whose complete 10-sample future window exists."""

    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be > 0")
    if future_dt_s <= 0.0:
        raise ValueError("future_dt_s must be > 0")
    step = max(1, int(round(future_dt_s / timestep_s)))
    lookahead = (NUM_FUTURE_FRAMES - 1) * step
    return max(0, horizon - lookahead)


def _future_indices(*, horizon: int, timestep_s: float, future_dt_s: float) -> np.ndarray:
    indices, _ = _future_index_grid(
        horizon=horizon,
        timestep_s=timestep_s,
        future_dt_s=future_dt_s,
    )
    return indices


def _future_index_grid(
    *,
    horizon: int,
    timestep_s: float,
    future_dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be > 0")
    if future_dt_s <= 0.0:
        raise ValueError("future_dt_s must be > 0")
    step = max(1, int(round(future_dt_s / timestep_s)))
    starts = np.arange(horizon, dtype=np.int64)[:, None]
    offsets = np.arange(NUM_FUTURE_FRAMES, dtype=np.int64)[None, :] * step
    requested = starts + offsets
    return np.minimum(requested, horizon - 1), requested >= horizon


def future_padding_counts(
    *,
    horizon: int,
    timestep_s: float,
    future_dt_s: float = G1_FUTURE_DT_S,
) -> np.ndarray:
    """Return the number of tail-padded future samples for every action row."""

    _, padding = _future_index_grid(
        horizon=int(horizon),
        timestep_s=float(timestep_s),
        future_dt_s=float(future_dt_s),
    )
    return padding.sum(axis=1, dtype=np.int64)


def _build_encoder_input_row(
    motion: G1MotionChunk,
    *,
    index: int,
    future: np.ndarray,
    future_padding: np.ndarray,
    root_relative_6d: np.ndarray,
    stationary: bool = False,
) -> np.ndarray:
    future_dq = motion.dq[future].copy()
    future_dq[future_padding] = 0.0
    # A final or explicit stationary row represents an indefinite hold.
    if stationary or index == motion.horizon - 1:
        future_dq.fill(0.0)
    parts = [
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        motion.q[future].reshape(-1),
        future_dq.reshape(-1),
        np.zeros((10,), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        np.zeros((6,), dtype=np.float32),
        np.asarray(root_relative_6d, dtype=np.float32).reshape(-1),
        np.zeros((120,), dtype=np.float32),
        np.zeros((120,), dtype=np.float32),
        np.zeros((9,), dtype=np.float32),
        np.zeros((12,), dtype=np.float32),
        np.zeros((720,), dtype=np.float32),
        np.zeros((60,), dtype=np.float32),
        np.zeros((60,), dtype=np.float32),
    ]
    row = np.concatenate(parts).astype(np.float32, copy=False)
    if row.shape != (SONIC_ENCODER_INPUT_DIM,):
        raise RuntimeError(
            f"internal SONIC encoder layout produced {row.shape}, expected {(SONIC_ENCODER_INPUT_DIM,)}"
        )
    return row


def build_g1_encoder_input_row(
    motion: G1MotionChunk,
    *,
    action_index: int,
    actual_base_quat_wxyz: np.ndarray,
    heading_alignment_rotation: np.ndarray,
    future_dt_s: float = G1_FUTURE_DT_S,
    stationary: bool = False,
) -> np.ndarray:
    """Build one 1762D row using the latest measured base orientation.

    ``heading_alignment_rotation`` is computed once when the chunk is
    activated and remains fixed. ``stationary`` repeats the selected pose in
    the entire future window and zeros all joint velocities for a true hold.
    """

    index = int(action_index)
    if not 0 <= index < motion.horizon:
        raise IndexError(f"action_index {index} outside motion horizon {motion.horizon}")
    if stationary:
        future = np.full((NUM_FUTURE_FRAMES,), index, dtype=np.int64)
        future_padding = np.zeros((NUM_FUTURE_FRAMES,), dtype=bool)
    else:
        future_grid, padding_grid = _future_index_grid(
            horizon=motion.horizon,
            timestep_s=motion.timestep_s,
            future_dt_s=float(future_dt_s),
        )
        future = future_grid[index]
        future_padding = padding_grid[index]
    root_relative_6d = motion.actual_root_6d(
        actual_base_quat_wxyz,
        future,
        heading_alignment_rotation=heading_alignment_rotation,
    )
    return _build_encoder_input_row(
        motion,
        index=index,
        future=future,
        future_padding=future_padding,
        root_relative_6d=root_relative_6d,
        stationary=stationary,
    )


def build_g1_encoder_inputs(
    motion: G1MotionChunk,
    *,
    future_dt_s: float = G1_FUTURE_DT_S,
) -> np.ndarray:
    """Build SONIC release G1-mode encoder observations with shape [H,1762]."""

    future, future_padding = _future_index_grid(
        horizon=motion.horizon,
        timestep_s=motion.timestep_s,
        future_dt_s=float(future_dt_s),
    )
    base = np.broadcast_to(
        np.arange(motion.horizon, dtype=np.int64)[:, None],
        future.shape,
    )
    root_relative_6d = motion.relative_root_6d(base, future)
    rows = np.zeros((motion.horizon, SONIC_ENCODER_INPUT_DIM), dtype=np.float32)

    for index in range(motion.horizon):
        rows[index] = _build_encoder_input_row(
            motion,
            index=index,
            future=future[index],
            future_padding=future_padding[index],
            root_relative_6d=root_relative_6d[index],
        )
    return rows
