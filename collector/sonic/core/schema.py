from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


RAW_POSE_KEYS = (
    "left_trigger",
    "right_trigger",
)

RAW_POSE_SCALAR_KEYS = (
    "left_trigger",
    "right_trigger",
)

WINDOWED_DERIVED_KEYS = (
)

PASSTHROUGH_DERIVED_KEYS = (
    "left_wuji_qpos",
    "right_wuji_qpos",
    "left_wuji_qpos_valid",
    "right_wuji_qpos_valid",
    "left_hand_binary_closed",
    "right_hand_binary_closed",
)

PASSTHROUGH_DERIVED_SCALAR_KEYS = (
    "left_wuji_qpos_valid",
    "right_wuji_qpos_valid",
    "left_hand_binary_closed",
    "right_hand_binary_closed",
)

OBS_KEYS = (
    "base_quat",
    "base_gravity",
    "base_ang_vel",
    "base_accel",
    "body_q",
    "body_dq",
    "left_hand_q",
    "right_hand_q",
    "token_state",
    "encoder_mode",
)

OBS_SCALAR_KEYS = (
    "encoder_mode",
)

ACTION_KEYS = (
)

COLLECTOR_CONTROL_START = 1
COLLECTOR_CONTROL_SAVE = 2
COLLECTOR_CONTROL_DISCARD = 3
COLLECTOR_CONTROL_COMMAND_NAMES = {
    COLLECTOR_CONTROL_START: "start",
    COLLECTOR_CONTROL_SAVE: "save",
    COLLECTOR_CONTROL_DISCARD: "discard",
}

HAND_STATUS_SCALAR_KEYS = (
    "source_frame_index",
    "source_timestamp_monotonic_pc_ns",
    "left_command_valid",
    "right_command_valid",
    "left_actual_position_valid",
    "right_actual_position_valid",
    "left_apply_success",
    "right_apply_success",
    "left_hand_binary_closed",
    "right_hand_binary_closed",
    "robot_receive_time_monotonic_ns",
    "robot_apply_time_monotonic_ns",
    "robot_read_time_monotonic_ns",
    "command_latency_ns",
)


def to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    return value


def _last_frame_value(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.shape[0] == 0:
        return []
    return to_python(array[-1])


def _scalar_value(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.ndim == 0 or array.size == 1:
        return array.reshape(-1)[0].item()
    return to_python(array)


def base_quat_to_gravity(base_quat: Any) -> list[float]:
    """Convert a wxyz base quaternion to gravity direction in the base frame."""

    quat = np.asarray(base_quat, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"base_quat must have 4 values, got {quat.size}")
    if not np.all(np.isfinite(quat)):
        raise ValueError("base_quat contains NaN/Inf")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("base_quat has near-zero norm")
    qw, qx, qy, qz = quat / norm
    return [
        float(2.0 * (-qz * qx + qw * qy)),
        float(-2.0 * (qz * qy + qw * qx)),
        float(1.0 - 2.0 * (qw * qw + qz * qz)),
    ]


def payload_timestamp_ns(payload: dict[str, np.ndarray]) -> int:
    if "timestamp_monotonic_ns" in payload:
        return int(np.asarray(payload["timestamp_monotonic_ns"]).reshape(-1)[0])
    if "timestamp_monotonic" in payload:
        return int(float(np.asarray(payload["timestamp_monotonic"]).reshape(-1)[0]) * 1e9)
    raise KeyError("payload missing monotonic timestamp")


def payload_frame_index(payload: dict[str, np.ndarray], use_last: bool = False) -> int:
    if "frame_index" not in payload:
        raise KeyError("payload missing frame_index")
    frame_index = np.asarray(payload["frame_index"]).reshape(-1)
    if frame_index.size == 0:
        raise ValueError("frame_index payload is empty")
    return int(frame_index[-1] if use_last else frame_index[0])


def split_pose_raw_payload(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    human_raw: dict[str, Any] = {}
    for key in RAW_POSE_KEYS:
        if key in payload:
            if key in RAW_POSE_SCALAR_KEYS:
                human_raw[key] = _scalar_value(payload[key])
            else:
                human_raw[key] = to_python(payload[key])
    return human_raw


def split_pose_control_payload(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    human_derived: dict[str, Any] = {}
    for key in WINDOWED_DERIVED_KEYS:
        if key in payload:
            human_derived[key] = _last_frame_value(payload[key])
    for key in PASSTHROUGH_DERIVED_KEYS:
        if key in payload:
            if key in PASSTHROUGH_DERIVED_SCALAR_KEYS:
                human_derived[key] = _scalar_value(payload[key])
            else:
                human_derived[key] = to_python(payload[key])
    return human_derived


def split_state_action_payload(payload: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    obs: dict[str, Any] = {}
    action: dict[str, Any] = {}
    for key in OBS_KEYS:
        if key in payload:
            if key in OBS_SCALAR_KEYS:
                obs[key] = _scalar_value(payload[key])
            else:
                obs[key] = to_python(payload[key])
            if key == "base_quat" and "base_gravity" not in payload:
                try:
                    obs["base_gravity"] = base_quat_to_gravity(obs["base_quat"])
                except ValueError:
                    pass
    for key in ACTION_KEYS:
        if key in payload:
            action[key] = to_python(payload[key])
    return obs, action


@dataclass(frozen=True)
class CameraInfo:
    key: str
    model: str
    serial: str
    product_line: str = ""
    firmware_version: str = ""
    usb_type: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model": self.model,
            "serial": self.serial,
            "product_line": self.product_line,
            "firmware_version": self.firmware_version,
            "usb_type": self.usb_type,
        }


@dataclass
class CameraFrame:
    key: str
    model: str
    serial: str
    rgb: np.ndarray
    depth: np.ndarray | None
    timestamp_ns: int
    encoded_color_jpeg: bytes | None = None
    source_time_ns: int | None = None
    source_realtime_ns: int | None = None
    sequence: int | None = None


@dataclass
class PoseControlSample:
    timestamp_ns: int
    frame_index: int
    human_derived: dict[str, Any]
    payload: dict[str, Any]


@dataclass
class PoseRawSample:
    timestamp_ns: int
    frame_index: int
    human_raw: dict[str, Any]
    payload: dict[str, Any]


@dataclass
class PoseSample:
    timestamp_ns: int
    frame_index: int
    human_raw: dict[str, Any]
    human_derived: dict[str, Any]
    payload: dict[str, Any]


@dataclass
class RobotStateActionSample:
    timestamp_ns: int
    obs: dict[str, Any]
    action: dict[str, Any]
    payload: dict[str, Any]


@dataclass
class HandStatusSample:
    timestamp_ns: int
    source_frame_index: int
    feedback: dict[str, Any]
    payload: dict[str, Any]


@dataclass
class CollectorControlSample:
    timestamp_ns: int
    sequence: int
    command: int
    command_name: str
    payload: dict[str, Any]


@dataclass
class DrainedSamples:
    pose_controls: list[PoseControlSample]
    pose_raws: list[PoseRawSample]
    state_actions: list[RobotStateActionSample]
    hand_statuses: list[HandStatusSample] = field(default_factory=list)
    collector_controls: list[CollectorControlSample] = field(default_factory=list)


@dataclass
class EpisodeFrame:
    time_ns: int
    primary_camera: str
    cameras: dict[str, CameraFrame]
    human_raw: dict[str, Any]
    human_derived: dict[str, Any]
    obs: dict[str, Any]
    action: dict[str, Any]


def make_pose_control_sample(payload: dict[str, np.ndarray]) -> PoseControlSample:
    human_derived = split_pose_control_payload(payload)
    serializable_payload = {key: to_python(value) for key, value in payload.items()}
    return PoseControlSample(
        timestamp_ns=payload_timestamp_ns(payload),
        frame_index=payload_frame_index(payload, use_last=True),
        human_derived=human_derived,
        payload=serializable_payload,
    )


def make_pose_raw_sample(payload: dict[str, np.ndarray]) -> PoseRawSample:
    human_raw = split_pose_raw_payload(payload)
    serializable_payload = {key: to_python(value) for key, value in payload.items()}
    return PoseRawSample(
        timestamp_ns=payload_timestamp_ns(payload),
        frame_index=payload_frame_index(payload, use_last=False),
        human_raw=human_raw,
        payload=serializable_payload,
    )


def make_state_action_sample(payload: dict[str, np.ndarray]) -> RobotStateActionSample:
    obs, action = split_state_action_payload(payload)
    serializable_payload = {key: to_python(value) for key, value in payload.items()}
    return RobotStateActionSample(
        timestamp_ns=payload_timestamp_ns(payload),
        obs=obs,
        action=action,
        payload=serializable_payload,
    )


def make_hand_status_sample(payload: dict[str, np.ndarray]) -> HandStatusSample:
    serializable_payload = {key: to_python(value) for key, value in payload.items()}
    source_timestamp = serializable_payload.get("source_timestamp_monotonic_pc_ns", -1)
    if isinstance(source_timestamp, list):
        source_timestamp = source_timestamp[0] if source_timestamp else -1
    source_frame_index = serializable_payload.get("source_frame_index", -1)
    if isinstance(source_frame_index, list):
        source_frame_index = source_frame_index[0] if source_frame_index else -1

    feedback: dict[str, Any] = {}
    for key in (
        "source_frame_index",
        "source_timestamp_monotonic_pc_ns",
        "left_wuji_qpos_command",
        "right_wuji_qpos_command",
        "left_wuji_qpos_actual",
        "right_wuji_qpos_actual",
        "left_command_valid",
        "right_command_valid",
        "left_actual_position_valid",
        "right_actual_position_valid",
        "left_apply_success",
        "right_apply_success",
        "left_hand_binary_closed",
        "right_hand_binary_closed",
        "robot_receive_time_monotonic_ns",
        "robot_apply_time_monotonic_ns",
        "robot_read_time_monotonic_ns",
        "command_latency_ns",
    ):
        if key not in serializable_payload:
            continue
        value = serializable_payload[key]
        if key in HAND_STATUS_SCALAR_KEYS and isinstance(value, list) and len(value) == 1:
            value = value[0]
        feedback[key] = value
    return HandStatusSample(
        timestamp_ns=int(source_timestamp),
        source_frame_index=int(source_frame_index),
        feedback=feedback,
        payload=serializable_payload,
    )


def make_collector_control_sample(payload: dict[str, np.ndarray]) -> CollectorControlSample:
    serializable_payload = {key: to_python(value) for key, value in payload.items()}
    command = int(np.asarray(payload["command"]).reshape(-1)[0])
    command_name = COLLECTOR_CONTROL_COMMAND_NAMES.get(command)
    if command_name is None:
        raise ValueError(f"unsupported collector_control command: {command}")
    sequence = int(np.asarray(payload["sequence"]).reshape(-1)[0])
    return CollectorControlSample(
        timestamp_ns=payload_timestamp_ns(payload),
        sequence=sequence,
        command=command,
        command_name=command_name,
        payload=serializable_payload,
    )
