"""Timestamp-aware interpolation for the 20 Hz HGPT collector."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .schema import PairedPose, WireSample, int_scalar

INTERPOLATION_LOWER_NS_KEY = "_collector_interpolation_lower_ns"
INTERPOLATION_UPPER_NS_KEY = "_collector_interpolation_upper_ns"


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """SLERP quaternions with either wxyz or xyzw component ordering."""
    left = np.asarray(q0, dtype=np.float64)
    right = np.asarray(q1, dtype=np.float64)
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    dot = np.sum(left * right, axis=-1, keepdims=True)
    right = np.where(dot < 0.0, -right, right)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-7
    denom = np.where(near, 1.0, sin_theta)
    result = np.sin((1.0 - alpha) * theta) / denom * left + np.sin(alpha * theta) / denom * right
    result = np.where(near, (1.0 - alpha) * left + alpha * right, result)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _safe_slerp(
    q0: np.ndarray,
    q1: np.ndarray,
    alpha: float,
    *,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    left = np.asarray(q0)
    right = np.asarray(q1)
    if left.shape != shape or right.shape != shape:
        return None
    try:
        left_norm = np.linalg.norm(left, axis=-1)
        right_norm = np.linalg.norm(right, axis=-1)
    except (FloatingPointError, TypeError, ValueError):
        return None
    if (
        not np.all(np.isfinite(left_norm))
        or not np.all(np.isfinite(right_norm))
        or np.any(left_norm < 1e-9)
        or np.any(right_norm < 1e-9)
    ):
        return None
    try:
        result = _slerp(left, right, alpha)
    except (FloatingPointError, TypeError, ValueError):
        return None
    if result.shape != shape or not np.all(np.isfinite(result)):
        return None
    return result


def _lerp_payload(
    lower: dict[str, np.ndarray],
    upper: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    """Interpolate common floating fields and keep categorical fields causal."""
    result: dict[str, np.ndarray] = {}
    for key, lower_value in lower.items():
        left = np.asarray(lower_value)
        right_value = upper.get(key)
        if right_value is None:
            continue
        right = np.asarray(right_value)
        if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
            continue
        elif left.dtype.kind == "f":
            result[key] = ((1.0 - alpha) * left + alpha * right).astype(left.dtype, copy=False)
        elif left.dtype.kind == "b":
            result[key] = np.logical_and(left, right)
        else:
            result[key] = left.copy()
    return result


def _set_scalar(payload: dict[str, np.ndarray], key: str, value: int) -> None:
    if key in payload:
        payload[key] = np.asarray([value], dtype=np.asarray(payload[key]).dtype)


def _interpolate_sample(
    lower: WireSample,
    upper: WireSample,
    *,
    target_ns: int,
    lower_ns: int,
    upper_ns: int,
) -> WireSample:
    if upper_ns <= lower_ns:
        alpha = 0.0
    else:
        alpha = float(np.clip((target_ns - lower_ns) / (upper_ns - lower_ns), 0.0, 1.0))
    payload = _lerp_payload(lower.payload, upper.payload, alpha)

    if lower.topic == "pose" and "g1_qpos" in payload:
        left = np.asarray(lower.payload["g1_qpos"])
        right = np.asarray(upper.payload["g1_qpos"])
        if left.shape == (36,) and right.shape == (36,):
            quaternion = _safe_slerp(
                left[3:7],
                right[3:7],
                alpha,
                shape=(4,),
            )
            if quaternion is not None:
                payload["g1_qpos"][3:7] = quaternion
    elif lower.topic == "teleop_raw" and "body_poses" in payload:
        left = np.asarray(lower.payload["body_poses"])
        right = np.asarray(upper.payload["body_poses"])
        if left.ndim == 2 and right.shape == left.shape and left.shape[1] == 7:
            quaternions = _safe_slerp(
                left[:, 3:7],
                right[:, 3:7],
                alpha,
                shape=(left.shape[0], 4),
            )
            if quaternions is not None:
                payload["body_poses"][:, 3:7] = quaternions
    elif lower.topic == "robot_state_action":
        if "base_quat" in payload:
            quaternion = _safe_slerp(
                lower.payload["base_quat"],
                upper.payload["base_quat"],
                alpha,
                shape=(4,),
            )
            if quaternion is not None:
                payload["base_quat"] = quaternion.astype(np.asarray(lower.payload["base_quat"]).dtype)
        if "reference_qpos" in payload:
            left = np.asarray(lower.payload["reference_qpos"])
            right = np.asarray(upper.payload["reference_qpos"])
            if left.shape == (36,) and right.shape == (36,):
                quaternion = _safe_slerp(
                    left[3:7],
                    right[3:7],
                    alpha,
                    shape=(4,),
                )
                if quaternion is not None:
                    payload["reference_qpos"][3:7] = quaternion

    for key in ("publish_monotonic_ns", "timestamp_monotonic_ns"):
        _set_scalar(payload, key, target_ns)
    payload[INTERPOLATION_LOWER_NS_KEY] = np.asarray(
        [lower_ns],
        dtype=np.int64,
    )
    payload[INTERPOLATION_UPPER_NS_KEY] = np.asarray(
        [upper_ns],
        dtype=np.int64,
    )
    return WireSample(lower.topic, target_ns, payload)


def _pair_time(pair: PairedPose) -> int:
    return pair.pose.publish_monotonic_ns


def _state_time(sample: WireSample) -> int:
    return int_scalar(sample.payload, "timestamp_monotonic_ns")


def _hand_time(sample: WireSample) -> int:
    return int_scalar(
        sample.payload,
        "source_timestamp_monotonic_pc_ns",
        int_scalar(sample.payload, "timestamp_monotonic_ns"),
    )


def _bracket(
    samples: Sequence,
    target_ns: int,
    timestamp: Callable[[object], int],
):
    lower = None
    for sample in samples:
        sample_ns = timestamp(sample)
        if sample_ns < 0:
            continue
        if sample_ns == target_ns:
            return sample, sample, sample_ns, sample_ns
        if sample_ns < target_ns:
            lower = sample
            continue
        if lower is None:
            return None
        return lower, sample, timestamp(lower), sample_ns
    return None


def interpolate_pair(history: Sequence[PairedPose], target_ns: int) -> PairedPose | None:
    bracket = _bracket(history, target_ns, _pair_time)
    if bracket is None:
        return None
    lower, upper, lower_ns, upper_ns = bracket
    pose = _interpolate_sample(
        lower.pose,
        upper.pose,
        target_ns=target_ns,
        lower_ns=lower_ns,
        upper_ns=upper_ns,
    )
    raw = _interpolate_sample(
        lower.raw,
        upper.raw,
        target_ns=target_ns,
        lower_ns=lower_ns,
        upper_ns=upper_ns,
    )
    return PairedPose(lower.frame_index, pose, raw)


def interpolate_sample(
    history: Sequence[WireSample],
    target_ns: int,
    *,
    topic: str,
) -> WireSample | None:
    timestamp = _state_time if topic == "robot_state_action" else _hand_time
    bracket = _bracket(history, target_ns, timestamp)
    if bracket is None:
        return None
    lower, upper, lower_ns, upper_ns = bracket
    return _interpolate_sample(
        lower,
        upper,
        target_ns=target_ns,
        lower_ns=lower_ns,
        upper_ns=upper_ns,
    )


def latest_causal_sample(history: Sequence[WireSample], target_ns: int, *, topic: str) -> WireSample | None:
    timestamp = _state_time if topic == "robot_state_action" else _hand_time
    result = None
    for sample in history:
        if timestamp(sample) <= target_ns:
            result = sample
        else:
            break
    return result
