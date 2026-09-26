"""The released SONIC 110D observation / 136D next-frame action contract."""

from collections.abc import Mapping

import numpy as np

FPS = 20
CAMERA = "observation.images.primary"
VERSION = 1
STATE_LAYOUT = {
    "0:3": "body-frame gravity direction",
    "3:6": "body-frame angular velocity (rad/s)",
    "6:9": "body-frame acceleration (m/s^2)",
    "9:38": "G1 position, collector IsaacLab joint order (rad)",
    "38:67": "G1 velocity, collector IsaacLab joint order (rad/s)",
    "67:87": "measured left Wuji qpos (rad)",
    "87:107": "measured right Wuji qpos (rad)",
    "107:110": "roll (rad), pitch (rad), body z angular velocity (rad/s)",
}
ACTION_LAYOUT = {
    "0:64": "next obs.token_state",
    "64:84": "next human_derived left Wuji target (rad)",
    "84:104": "next human_derived right Wuji target (rad)",
    "104:133": "next measured G1 position, collector IsaacLab order (rad)",
    "133:136": "next measured roll, pitch, body z angular velocity",
}
ACTION_SEMANTICS = (
    "Current RGB/state predicts the next recorded frame (+0.05s): observed SONIC token, "
    "teleoperation hand targets, measured body positions and root3. Absolute positions, "
    "not increments; hand actions are not measured hand feedback."
)


def vector(frame: Mapping, field: str, size: int, dtype=np.float32) -> np.ndarray:
    value = frame
    try:
        for part in field.split("."):
            value = value[part]
        result = np.asarray(value, dtype=dtype)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {field}") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a finite [{size}] vector, got {result.shape}")
    return result


def require_true(value, field: str) -> None:
    if value is not True and not (type(value) is int and value == 1):
        raise ValueError(f"{field} must be true; refusing to drop or relabel invalid feedback")


def state(frame: Mapping) -> np.ndarray:
    quat = vector(frame, "obs.base_quat", 4, np.float64)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("obs.base_quat has zero or nonfinite norm")
    w, x, y, z = quat / norm
    gravity = frame.get("obs", {}).get("base_gravity")
    if gravity is None:
        gravity = np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])
    else:
        gravity = vector(frame, "obs.base_gravity", 3)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    velocity = vector(frame, "obs.base_ang_vel", 3)
    feedback = frame.get("hand_feedback", {})
    for side in ("left", "right"):
        require_true(feedback.get(f"{side}_actual_position_valid"), f"hand_feedback.{side}_actual_position_valid")
    return np.concatenate(
        [
            gravity,
            velocity,
            vector(frame, "obs.base_accel", 3),
            vector(frame, "obs.body_q", 29),
            vector(frame, "obs.body_dq", 29),
            vector(frame, "hand_feedback.left_wuji_qpos_actual", 20),
            vector(frame, "hand_feedback.right_wuji_qpos_actual", 20),
            [roll, pitch, velocity[2]],
        ]
    ).astype(np.float32)


def action(frame: Mapping, observed: np.ndarray) -> np.ndarray:
    token = vector(frame, "obs.token_state", 64)
    if np.max(np.abs(token * 16 - np.rint(token * 16))) > 1e-5:
        raise ValueError("obs.token_state is not on the SONIC 1/16 FSQ grid")
    derived = frame.get("human_derived", {})
    for side in ("left", "right"):
        key = f"{side}_wuji_qpos_valid"
        if key in derived:
            require_true(derived[key], f"human_derived.{key}")
    return np.concatenate(
        [
            token,
            vector(frame, "human_derived.left_wuji_qpos", 20),
            vector(frame, "human_derived.right_wuji_qpos", 20),
            observed[9:38],
            observed[107:110],
        ]
    ).astype(np.float32)


def episode_arrays(frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    states, actions = [], []
    for index, frame in enumerate(frames):
        try:
            observed = state(frame)
            states.append(observed)
            if index:
                actions.append(action(frame, observed))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"frame {index}: {exc}") from exc
    return np.asarray(states[:-1], np.float32), np.asarray(actions, np.float32)
