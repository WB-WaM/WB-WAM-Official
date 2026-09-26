"""Pure data-health checks and throttled issue reporting for the HGPT collector."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from collector.sonic.core.camera_manager import CausalFrameMatch

from .interpolation import (
    INTERPOLATION_LOWER_NS_KEY,
    INTERPOLATION_UPPER_NS_KEY,
)
from .schema import PairedPose, WireSample

DEFAULT_FRESHNESS_NS = 100_000_000
DEFAULT_GRACE_NS = 1_000_000_000
DEFAULT_REPEAT_NS = 5_000_000_000


@dataclass(frozen=True, slots=True)
class HealthIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class HealthLogEvent:
    level: int
    code: str
    message: str


def _issue(issues: list[HealthIssue], code: str, message: str) -> None:
    issues.append(HealthIssue(code=code, message=message))


def _check_vector(
    issues: list[HealthIssue],
    payload: Mapping[str, Any],
    *,
    key: str,
    size: int,
    code: str,
    source: str,
) -> None:
    if key not in payload:
        _issue(issues, f"{code}_missing", f"{source} has no {key}")
        return
    value = np.asarray(payload[key])
    if value.shape != (size,):
        _issue(issues, f"{code}_shape", f"{key} shape={value.shape}, expected ({size},)")
        return
    try:
        finite = bool(np.all(np.isfinite(value)))
    except TypeError:
        finite = False
    if not finite:
        _issue(issues, f"{code}_nonfinite", f"{key} contains NaN/Inf or non-numeric values")


def _read_scalar(payload: Mapping[str, Any], key: str) -> tuple[Any | None, str | None]:
    if key not in payload:
        return None, "missing"
    value = np.asarray(payload[key])
    if value.size != 1:
        return None, f"shape={value.shape}, expected scalar"
    return value.reshape(-1)[0].item(), None


def _check_true_flag(
    issues: list[HealthIssue],
    payload: Mapping[str, Any],
    *,
    key: str,
    code: str,
    false_message: str,
) -> None:
    value, error = _read_scalar(payload, key)
    if error is not None:
        _issue(issues, code, f"{key} {error}")
        return
    if isinstance(value, bool):
        enabled = value
    elif isinstance(value, (int, float)) and np.isfinite(value) and value in (0, 1):
        enabled = bool(value)
    else:
        _issue(issues, code, f"{key} has invalid boolean value {value!r}")
        return
    if not enabled:
        _issue(issues, code, false_message)


def _read_timestamp(payload: Mapping[str, Any], keys: Sequence[str]) -> tuple[int | None, str | None]:
    for key in keys:
        if key not in payload:
            continue
        value, error = _read_scalar(payload, key)
        if error is not None:
            return None, f"{key} {error}"
        try:
            timestamp = int(value)
        except (TypeError, ValueError, OverflowError):
            return None, f"{key} is not an integer timestamp"
        if timestamp <= 0:
            return None, f"{key}={timestamp}, expected positive timestamp"
        return timestamp, None
    return None, f"missing {' or '.join(keys)}"


def _read_interpolation_bounds(
    payload: Mapping[str, Any],
) -> tuple[tuple[int, int] | None, str | None]:
    lower, lower_error = _read_scalar(payload, INTERPOLATION_LOWER_NS_KEY)
    upper, upper_error = _read_scalar(payload, INTERPOLATION_UPPER_NS_KEY)
    if lower_error == "missing" and upper_error == "missing":
        return None, None
    if lower_error is not None or upper_error is not None:
        return None, f"lower={lower_error or lower}, upper={upper_error or upper}"
    try:
        return (int(lower), int(upper)), None
    except (TypeError, ValueError, OverflowError):
        return None, f"lower={lower!r}, upper={upper!r}"


_TIMESTAMP_KEYS: dict[str, tuple[str, ...]] = {
    "pose": ("publish_monotonic_ns",),
    "robot": ("timestamp_monotonic_ns",),
    "hand": ("source_timestamp_monotonic_pc_ns", "timestamp_monotonic_ns"),
}


def sample_timestamps(
    pair: PairedPose | None,
    state: WireSample | None,
    hand: WireSample | None,
) -> dict[str, int]:
    """Return valid PC-clock source timestamps for regression tracking."""

    samples = {
        "pose": pair.pose if pair is not None else None,
        "robot": state,
        "hand": hand,
    }
    result: dict[str, int] = {}
    for stream, sample in samples.items():
        if sample is None:
            continue
        timestamp, error = _read_timestamp(sample.payload, _TIMESTAMP_KEYS[stream])
        if error is None and timestamp is not None:
            bounds, bounds_error = _read_interpolation_bounds(sample.payload)
            result[stream] = bounds[1] if bounds is not None and bounds_error is None else timestamp
    return result


def _check_stream_freshness(
    issues: list[HealthIssue],
    *,
    stream: str,
    sample: WireSample | None,
    tick_ns: int,
    freshness_ns: int,
    previous_timestamp: int | None,
) -> None:
    if sample is None:
        _issue(issues, f"{stream}.status_missing", f"no {sample_topic(stream)} sample received")
        return
    timestamp, error = _read_timestamp(sample.payload, _TIMESTAMP_KEYS[stream])
    if error is not None or timestamp is None:
        _issue(issues, f"{stream}.timestamp_invalid", error or "invalid timestamp")
        return
    bounds, bounds_error = _read_interpolation_bounds(sample.payload)
    progression_timestamp = bounds[1] if bounds is not None else timestamp
    if previous_timestamp is not None and progression_timestamp < previous_timestamp:
        _issue(
            issues,
            f"{stream}.timestamp_regressed",
            f"timestamp regressed from {previous_timestamp} to {progression_timestamp}",
        )
    age_ns = tick_ns - timestamp
    if age_ns < 0:
        _issue(issues, f"{stream}.timestamp_future", f"timestamp is {-age_ns / 1e6:.1f}ms in the future")
    elif age_ns > freshness_ns:
        _issue(issues, f"{stream}.status_stale", f"age={age_ns / 1e6:.1f}ms")

    if bounds is None and bounds_error is None:
        return
    if bounds_error is not None or bounds is None:
        _issue(
            issues,
            f"{stream}.interpolation_bounds_invalid",
            bounds_error or "invalid interpolation bounds",
        )
        return
    lower_ns, upper_ns = bounds
    lower_age_ns = tick_ns - lower_ns
    upper_age_ns = upper_ns - tick_ns
    if lower_age_ns < 0 or upper_age_ns < 0 or lower_age_ns > freshness_ns or upper_age_ns > freshness_ns:
        _issue(
            issues,
            f"{stream}.interpolation_gap",
            "capture tick is not within "
            f"{freshness_ns / 1e6:g}ms of both source samples "
            f"(lower={lower_age_ns / 1e6:.1f}ms, "
            f"upper={upper_age_ns / 1e6:.1f}ms)",
        )


def sample_topic(stream: str) -> str:
    return {
        "pose": "pose",
        "robot": "robot_state_action",
        "hand": "hand_status",
    }[stream]


def _mode(snapshot: Mapping[str, Any]) -> int:
    try:
        return int(snapshot.get("mode", 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _check_quaternion(
    issues: list[HealthIssue],
    payload: Mapping[str, Any],
    *,
    key: str,
    vector_size: int,
    offset: int,
    code: str,
) -> None:
    if key not in payload:
        return
    value = np.asarray(payload[key])
    if value.shape != (vector_size,):
        return
    quaternion = value[offset : offset + 4]
    try:
        norm = float(np.linalg.norm(quaternion))
    except (TypeError, ValueError):
        return
    if not np.isfinite(norm) or norm < 1e-9:
        _issue(
            issues,
            f"{code}_quaternion_invalid",
            f"{key} quaternion has invalid norm {norm!r}",
        )


def _check_pose(
    issues: list[HealthIssue],
    pair: PairedPose | None,
    *,
    require_reference: bool,
    tick_ns: int,
    freshness_ns: int,
) -> None:
    if pair is None:
        return
    payload = pair.pose.payload
    _check_vector(issues, payload, key="g1_qpos", size=36, code="pose.g1_qpos", source="pose")
    _check_quaternion(
        issues,
        payload,
        key="g1_qpos",
        vector_size=36,
        offset=3,
        code="pose.g1_qpos",
    )
    for side in ("left", "right"):
        _check_vector(
            issues,
            payload,
            key=f"{side}_wuji_qpos",
            size=20,
            code=f"pose.{side}_target",
            source="pose",
        )
    if require_reference:
        _check_true_flag(
            issues,
            payload,
            key="body_valid",
            code="pose.body_invalid",
            false_message="body_valid=false",
        )
        for side in ("left", "right"):
            _check_true_flag(
                issues,
                payload,
                key=f"{side}_wuji_qpos_valid",
                code=f"pose.{side}_target_invalid",
                false_message=f"{side}_wuji_qpos_valid=false",
            )
        source_ns, error = _read_timestamp(
            payload,
            ("body_source_monotonic_ns",),
        )
        if error is not None or source_ns is None:
            _issue(
                issues,
                "pose.body_source_invalid",
                error or "invalid body source timestamp",
            )
        else:
            age_ns = tick_ns - source_ns
            if age_ns < 0 or age_ns > freshness_ns:
                _issue(
                    issues,
                    "pose.body_source_stale",
                    f"age={age_ns / 1e6:.1f}ms",
                )


def _check_robot(
    issues: list[HealthIssue],
    state: WireSample | None,
    *,
    require_reference: bool,
) -> None:
    if state is None:
        return
    payload = state.payload
    for key, size in (
        ("base_quat", 4),
        ("base_ang_vel", 3),
        ("base_accel", 3),
        ("body_q", 29),
        ("body_dq", 29),
        ("body_action", 29),
        ("kp", 29),
        ("kd", 29),
        ("reference_qpos", 36),
    ):
        _check_vector(issues, payload, key=key, size=size, code=f"robot.{key}", source="robot_state_action")
    _check_quaternion(
        issues,
        payload,
        key="base_quat",
        vector_size=4,
        offset=0,
        code="robot.base_quat",
    )
    _check_quaternion(
        issues,
        payload,
        key="reference_qpos",
        vector_size=36,
        offset=3,
        code="robot.reference_qpos",
    )

    state_index, error = _read_scalar(payload, "state_index")
    if error is not None:
        suffix = "missing" if error == "missing" else "invalid"
        _issue(issues, f"robot.state_index_{suffix}", f"state_index {error}")
    else:
        try:
            numeric_index = float(state_index)
            valid_index = np.isfinite(numeric_index) and numeric_index.is_integer() and numeric_index >= 0
        except (TypeError, ValueError, OverflowError):
            valid_index = False
        if not valid_index:
            _issue(issues, "robot.state_index_invalid", f"state_index={state_index!r}")

    _check_true_flag(
        issues,
        payload,
        key="lowcmd_published",
        code="robot.lowcmd_not_published",
        false_message="body action was computed but not published",
    )
    if require_reference:
        _check_true_flag(
            issues,
            payload,
            key="reference_valid",
            code="robot.reference_invalid",
            false_message="reference_valid=false",
        )


def _check_hand(
    issues: list[HealthIssue],
    hand: WireSample | None,
    *,
    require_command: bool,
) -> None:
    if hand is None:
        return
    payload = hand.payload
    for side in ("left", "right"):
        _check_vector(
            issues,
            payload,
            key=f"{side}_wuji_qpos_command",
            size=20,
            code=f"hand.{side}_command",
            source="hand_status",
        )
        _check_vector(
            issues,
            payload,
            key=f"{side}_wuji_qpos_actual",
            size=20,
            code=f"hand.{side}_actual",
            source="hand_status",
        )
        _check_true_flag(
            issues,
            payload,
            key=f"{side}_actual_position_valid",
            code=f"hand.{side}_actual_invalid",
            false_message=f"{side}_actual_position_valid=false",
        )
        if require_command:
            _check_true_flag(
                issues,
                payload,
                key=f"{side}_command_valid",
                code=f"hand.{side}_command_invalid",
                false_message=f"{side}_command_valid=false",
            )
            _check_true_flag(
                issues,
                payload,
                key=f"{side}_apply_success",
                code=f"hand.{side}_apply_failed",
                false_message=f"{side}_apply_success=false",
            )


def _has_color(frame: Any) -> bool:
    encoded = getattr(frame, "encoded_color_jpeg", None)
    if encoded:
        return True
    rgb = getattr(frame, "rgb", None)
    return rgb is not None and np.asarray(rgb).size > 0


def _check_cameras(
    issues: list[HealthIssue],
    camera_matches: Mapping[str, CausalFrameMatch],
    *,
    expected_camera_keys: Collection[str],
    record_depth: bool,
) -> None:
    for key in sorted(expected_camera_keys):
        match = camera_matches.get(key)
        if match is None:
            _issue(issues, f"camera.{key}.missing", f"no matched frame for camera {key}")
            continue
        if bool(match.stale):
            _issue(issues, f"camera.{key}.stale", f"camera frame age={int(match.age_ns) / 1e6:.1f}ms")
        frame = match.frame
        if not _has_color(frame):
            _issue(issues, f"camera.{key}.color_missing", f"camera {key} has no color data")
        if record_depth:
            depth = getattr(frame, "depth", None)
            if depth is None or np.asarray(depth).size == 0:
                _issue(issues, f"camera.{key}.depth_missing", f"camera {key} has no depth data")


def _deduplicate(issues: Sequence[HealthIssue]) -> tuple[HealthIssue, ...]:
    result: list[HealthIssue] = []
    seen: set[str] = set()
    for issue in issues:
        if issue.code not in seen:
            seen.add(issue.code)
            result.append(issue)
    return tuple(result)


def check_health(
    *,
    tick_ns: int,
    pair: PairedPose | None,
    state: WireSample | None,
    hand: WireSample | None,
    snapshot: Mapping[str, Any],
    camera_matches: Mapping[str, CausalFrameMatch],
    expected_camera_keys: Collection[str],
    record_depth: bool,
    previous_timestamps: Mapping[str, int] | None = None,
    freshness_ns: int = DEFAULT_FRESHNESS_NS,
    require_training_actions: bool = False,
) -> tuple[HealthIssue, ...]:
    """Validate raw streams and camera matches without trusting snapshot zero defaults."""

    if freshness_ns < 0:
        raise ValueError("freshness_ns must be non-negative")
    previous = previous_timestamps or {}
    pose_sample = pair.pose if pair is not None else None
    issues: list[HealthIssue] = []
    for stream, sample in (("pose", pose_sample), ("robot", state), ("hand", hand)):
        _check_stream_freshness(
            issues,
            stream=stream,
            sample=sample,
            tick_ns=tick_ns,
            freshness_ns=freshness_ns,
            previous_timestamp=previous.get(stream),
        )

    require_reference = require_training_actions or _mode(snapshot) >= 1
    _check_pose(
        issues,
        pair,
        require_reference=require_reference,
        tick_ns=tick_ns,
        freshness_ns=freshness_ns,
    )
    _check_robot(issues, state, require_reference=require_reference)
    _check_hand(issues, hand, require_command=require_reference)
    _check_cameras(
        issues,
        camera_matches,
        expected_camera_keys=expected_camera_keys,
        record_depth=record_depth,
    )
    return _deduplicate(issues)


def issue_codes(issues: Sequence[HealthIssue]) -> list[str]:
    return [issue.code for issue in _deduplicate(issues)]


def stream_valid(issues: Sequence[HealthIssue], stream: str) -> bool:
    prefix = f"{stream}."
    return not any(issue.code.startswith(prefix) for issue in issues)


class IssueThrottle:
    """Emit immediate, periodic, and recovered events after a startup grace period."""

    def __init__(
        self,
        start_ns: int,
        *,
        grace_ns: int = DEFAULT_GRACE_NS,
        repeat_ns: int = DEFAULT_REPEAT_NS,
    ) -> None:
        if grace_ns < 0 or repeat_ns <= 0:
            raise ValueError("grace_ns must be non-negative and repeat_ns must be positive")
        self._grace_until_ns = int(start_ns) + int(grace_ns)
        self._repeat_ns = int(repeat_ns)
        self._active: dict[str, HealthIssue] = {}
        self._last_warning_ns: dict[str, int] = {}
        self._warned_active: set[str] = set()

    def update(
        self,
        issues: Sequence[HealthIssue],
        *,
        now_ns: int,
    ) -> tuple[HealthLogEvent, ...]:
        current = {issue.code: issue for issue in _deduplicate(issues)}
        events: list[HealthLogEvent] = []

        for code in self._active:
            if code in current:
                continue
            if code in self._warned_active:
                events.append(HealthLogEvent(logging.INFO, code, f"recovered: {code}"))
            self._warned_active.discard(code)

        if now_ns >= self._grace_until_ns:
            for code, issue in current.items():
                last_warning_ns = self._last_warning_ns.get(code)
                if last_warning_ns is None or now_ns - last_warning_ns >= self._repeat_ns:
                    events.append(HealthLogEvent(logging.WARNING, code, f"{code}: {issue.message}"))
                    self._last_warning_ns[code] = int(now_ns)
                    self._warned_active.add(code)

        self._active = current
        return tuple(events)
