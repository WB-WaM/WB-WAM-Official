from __future__ import annotations

import numpy as np

from bridge.common.state_layouts import STATE_LAYOUT_GRAVITY, STATE_LAYOUT_QUAT, normalize_state_layout
from bridge.sonic.joint_order import DEFAULT_ANGLES_ISAACLAB
from bridge.sonic.motion_schema import G1MotionChunk

from .representation import ActionRepresentation
from .schema import CurrentG1State, PhysicalActionChunk, PhysicalActionLayout


def isaaclab_delta_to_qpos(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != 29:
        raise ValueError(f"joint final dim must be 29, got {array.shape}")
    return array + DEFAULT_ANGLES_ISAACLAB.reshape((1,) * (array.ndim - 1) + (29,))


def _quat_to_rpy(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("base quaternion must be finite wxyz[4]")
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-8:
        raise ValueError("base quaternion has near-zero norm")
    w, x, y, z = quat / norm
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def _gravity_to_roll_pitch(gravity: np.ndarray) -> tuple[float, float]:
    gx, gy, gz = np.asarray(gravity, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm([gx, gy, gz]))
    if not np.isfinite(norm) or norm <= 1.0e-8:
        raise ValueError("base gravity is invalid")
    gx, gy, gz = gx / norm, gy / norm, gz / norm
    return float(np.arctan2(-gy, -gz)), float(np.arcsin(np.clip(gx, -1.0, 1.0)))


def current_g1_state(states: np.ndarray, *, state_layout: str | None = None) -> CurrentG1State:
    """Decode live telemetry or a projected WB semantic state."""

    state = np.asarray(states, dtype=np.float32).reshape(-1)
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN/Inf")
    if state.size == 72:
        return CurrentG1State(
            body_q_delta=state[0:29],
            body_dq=np.zeros((29,), dtype=np.float32),
            roll=float(state[29]),
            pitch=float(state[30]),
            yaw_velocity=float(state[31]),
            left_hand_qpos=state[32:52],
            right_hand_qpos=state[52:72],
        )
    if state.size == 101:
        return CurrentG1State(
            body_q_delta=state[0:29],
            body_dq=state[29:58],
            left_hand_qpos=state[58:78],
            right_hand_qpos=state[78:98],
            roll=float(state[98]),
            pitch=float(state[99]),
            yaw_velocity=float(state[100]),
        )

    layout = normalize_state_layout(state_layout, state_dim=state.size)
    if layout == STATE_LAYOUT_GRAVITY:
        if state.size != 107:
            raise ValueError(f"gravity state must be 107D, got {state.size}")
        roll, pitch = _gravity_to_roll_pitch(state[0:3])
        offset, yaw_velocity = 9, float(state[5])
    elif layout == STATE_LAYOUT_QUAT:
        if state.size != 108:
            raise ValueError(f"quaternion state must be 108D, got {state.size}")
        roll, pitch, _ = _quat_to_rpy(state[0:4])
        offset, yaw_velocity = 10, float(state[6])
    else:  # pragma: no cover
        raise ValueError(f"unsupported state layout {layout!r}")
    return CurrentG1State(
        body_q_delta=state[offset : offset + 29],
        body_dq=state[offset + 29 : offset + 58],
        left_hand_qpos=state[offset + 58 : offset + 78],
        right_hand_qpos=state[offset + 78 : offset + 98],
        roll=roll,
        pitch=pitch,
        yaw_velocity=yaw_velocity,
    )


def project_live_state_for_physical_wam(states: np.ndarray, *, state_layout: str | None = None) -> np.ndarray:
    """Build the latest WB 72D semantic state from live telemetry.

    This matches ``wb.yaml`` state slices in order: body q29, root
    roll/pitch/yaw-velocity3, left hand20, right hand20. Model-side padding to
    ``proprio_dim`` happens only after normalization in ``WBWAMAdapter``.
    """

    current = current_g1_state(states, state_layout=state_layout)
    return np.concatenate(
        [
            current.body_q_delta,
            np.asarray([current.roll, current.pitch, current.yaw_velocity], dtype=np.float32),
            current.left_hand_qpos,
            current.right_hand_qpos,
        ]
    ).astype(np.float32, copy=False)


def current_wb_semantic_state(current: CurrentG1State) -> np.ndarray:
    """Return the fixed-base 72D reference used by WB relative targets."""

    return np.concatenate(
        [
            current.body_q_delta,
            np.asarray([current.roll, current.pitch, current.yaw_velocity], dtype=np.float32),
            current.left_hand_qpos,
            current.right_hand_qpos,
        ]
    ).astype(np.float32, copy=False)


def _rpy_matrices(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    out = np.empty((roll.shape[0], 3, 3), dtype=np.float32)
    out[:, 0, 0], out[:, 0, 1], out[:, 0, 2] = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
    out[:, 1, 0], out[:, 1, 1], out[:, 1, 2] = sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr
    out[:, 2, 0], out[:, 2, 1], out[:, 2, 2] = -sp, cp * sr, cp * cr
    return out


def integrate_yaw_velocity(yaw_velocity: np.ndarray, *, timestep_s: float) -> np.ndarray:
    """Integrate sampled gyro-z values into a chunk-local yaw reference.

    The first action row is the reference origin, matching SONIC's
    motion-activation heading alignment. Subsequent rows integrate over the
    intervals between samples. ``root3[..., 2]`` is the training dataset's
    body-frame gyro-z signal; treating it as Euler yaw rate is the same
    near-upright approximation used by the WB action contract.
    """

    velocity = np.asarray(yaw_velocity, dtype=np.float64).reshape(-1)
    if velocity.size == 0:
        raise ValueError("yaw_velocity must contain at least one sample")
    if not np.isfinite(velocity).all():
        raise ValueError("yaw_velocity contains NaN/Inf")
    dt = float(timestep_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("timestep_s must be finite and > 0")

    yaw = np.zeros_like(velocity)
    if velocity.size > 1:
        interval_velocity = 0.5 * (velocity[:-1] + velocity[1:])
        yaw[1:] = np.cumsum(interval_velocity * dt, dtype=np.float64)
    return yaw.astype(np.float32)


def restore_physical_actions(
    actions: np.ndarray,
    current: CurrentG1State,
    *,
    representation: ActionRepresentation | str,
    relative_hands: bool = True,
    relative_joint_ranges: tuple[tuple[int, int], ...] | list[list[int]] | None = None,
) -> np.ndarray:
    """Restore a WAM chunk to absolute 72D actions using one fixed base."""

    mode = (
        representation
        if isinstance(representation, ActionRepresentation)
        else ActionRepresentation.parse(representation)
    )
    if mode == ActionRepresentation.SONIC_TOKEN:
        raise ValueError("sonic_token actions are already encoded")
    absolute_actions = np.asarray(actions, dtype=np.float32).copy()
    if absolute_actions.ndim != 2 or absolute_actions.shape[0] == 0:
        raise ValueError(f"physical actions must have shape [H,D], got {absolute_actions.shape}")
    if not np.isfinite(absolute_actions).all():
        raise ValueError("physical actions contain NaN/Inf")
    ranges = (
        mode.relative_joint_ranges(relative_hands=relative_hands)
        if relative_joint_ranges is None
        else tuple((int(item[0]), int(item[1])) for item in relative_joint_ranges)
    )
    reference = current_wb_semantic_state(current)
    for start, end in ranges:
        if start < 0 or end <= start or end > reference.size:
            raise ValueError(f"invalid relative_joint_range [{start}, {end}] for {reference.size}D WB action")
        absolute_actions[:, start:end] += reference[None, start:end]
    return absolute_actions


def physical_actions_to_motion(
    actions: np.ndarray,
    current: CurrentG1State,
    *,
    representation: ActionRepresentation | str,
    timestep_s: float,
    layout: PhysicalActionLayout | None = None,
    relative_hands: bool = True,
    relative_joint_ranges: tuple[tuple[int, int], ...] | list[list[int]] | None = None,
    previous_body_q_delta: np.ndarray | None = None,
) -> G1MotionChunk:
    """Invert a WAM action representation into absolute SONIC encoder motion."""

    absolute_actions = restore_physical_actions(
        actions,
        current,
        representation=representation,
        relative_hands=relative_hands,
        relative_joint_ranges=relative_joint_ranges,
    )
    dt = float(timestep_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("timestep_s must be finite and > 0")

    chunk = PhysicalActionChunk.unpack(absolute_actions, layout or PhysicalActionLayout())
    left, right = chunk.left_hand_qpos.copy(), chunk.right_hand_qpos.copy()
    q_delta, root3 = chunk.body_q_delta.copy(), chunk.root3.copy()

    qpos = isaaclab_delta_to_qpos(q_delta)
    previous_delta = (
        current.body_q_delta
        if previous_body_q_delta is None
        else np.asarray(previous_body_q_delta, dtype=np.float32).reshape(-1)
    )
    if previous_delta.shape != (29,) or not np.isfinite(previous_delta).all():
        raise ValueError("previous_body_q_delta must be finite 29D")
    previous_qpos = isaaclab_delta_to_qpos(previous_delta)
    dq = np.empty_like(qpos)
    dq[0] = (qpos[0] - previous_qpos) / dt
    if qpos.shape[0] > 1:
        dq[1:] = np.diff(qpos, axis=0) / dt
    yaw = integrate_yaw_velocity(root3[:, 2], timestep_s=dt)
    return G1MotionChunk(
        q=qpos,
        dq=dq,
        root_rotation=_rpy_matrices(root3[:, 0], root3[:, 1], yaw),
        timestep_s=dt,
        left_hand_qpos=left,
        right_hand_qpos=right,
    )
