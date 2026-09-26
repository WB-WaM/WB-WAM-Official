"""SONIC-compatible ZMQ pose protocol used by the HGPT pose manager."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

HEADER_SIZE = 1280
POSE_TOPIC = b"pose"
COMMAND_TOPIC = b"command"
COLLECTOR_CONTROL_TOPIC = b"collector_control"

COLLECTOR_CONTROL_START = 1
COLLECTOR_CONTROL_SAVE = 2
COLLECTOR_CONTROL_DISCARD = 3
COLLECTOR_CONTROL_COMMAND_NAMES = {
    COLLECTOR_CONTROL_START: "start",
    COLLECTOR_CONTROL_SAVE: "save",
    COLLECTOR_CONTROL_DISCARD: "discard",
}

_DTYPES = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}


def _header(fields: list[dict[str, Any]], version: int = 4) -> bytes:
    raw = json.dumps(
        {"v": version, "endian": "le", "count": 1, "fields": fields},
        separators=(",", ":"),
    ).encode()
    if len(raw) > HEADER_SIZE:
        raise ValueError(f"pose header is {len(raw)} bytes; maximum is {HEADER_SIZE}")
    return raw.ljust(HEADER_SIZE, b"\0")


def pack_topic_message(
    payload: dict[str, np.ndarray], *, topic: str = "pose", version: int = 4
) -> bytes:
    """Pack the exact topic/header/binary layout consumed by SONIC."""
    fields: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    dtype_names = {value: key for key, value in _DTYPES.items()}
    for name, value in payload.items():
        array = np.asarray(value)
        native = np.dtype(array.dtype.str.replace(">", "<"))
        dtype_name = dtype_names.get(native)
        if dtype_name is None:
            raise TypeError(f"unsupported dtype for {name}: {array.dtype}")
        array = np.ascontiguousarray(array.astype(_DTYPES[dtype_name], copy=False))
        fields.append({"name": name, "dtype": dtype_name, "shape": list(array.shape)})
        chunks.append(array.tobytes())
    return topic.encode() + _header(fields, version) + b"".join(chunks)


def decode_topic_message(message: bytes) -> tuple[str, dict[str, np.ndarray]]:
    """Decode a SONIC topic message without importing collector code."""
    topic_end = None
    for topic in (
        POSE_TOPIC,
        COMMAND_TOPIC,
        COLLECTOR_CONTROL_TOPIC,
        b"teleop_raw",
        b"robot_state_action",
        b"hand_status",
    ):
        if message.startswith(topic):
            topic_end = len(topic)
            break
    if topic_end is None or len(message) < topic_end + HEADER_SIZE:
        raise ValueError("unknown or truncated ZMQ topic message")
    header_raw = message[topic_end : topic_end + HEADER_SIZE].rstrip(b"\0")
    header = json.loads(header_raw.decode())
    offset = topic_end + HEADER_SIZE
    result: dict[str, np.ndarray] = {}
    for field in header.get("fields", []):
        name = str(field["name"])
        dtype_name = str(field["dtype"])
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported wire dtype {dtype_name!r}")
        shape = tuple(int(v) for v in field["shape"])
        dtype = _DTYPES[dtype_name]
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if offset + size > len(message):
            raise ValueError(f"truncated payload field {name!r}")
        result[name] = (
            np.frombuffer(
                message, dtype=dtype, count=size // dtype.itemsize, offset=offset
            )
            .reshape(shape)
            .copy()
        )
        offset += size
    if offset != len(message):
        raise ValueError(f"pose payload has {len(message) - offset} trailing bytes")
    return message[:topic_end].decode(), result


def pack_stop_command() -> bytes:
    return pack_topic_message(
        {
            "start": np.array([0], dtype=np.uint8),
            "stop": np.array([1], dtype=np.uint8),
            "planner": np.array([0], dtype=np.uint8),
        },
        topic="command",
        version=1,
    )


@dataclass(slots=True)
class CollectorControlSnapshot:
    """One edge-triggered Pico command for the data collector."""

    sequence: int
    command: int
    timestamp_monotonic_ns: int = 0

    def payload(self) -> dict[str, np.ndarray]:
        if self.sequence <= 0:
            raise ValueError("collector control sequence must be positive")
        if self.command not in COLLECTOR_CONTROL_COMMAND_NAMES:
            raise ValueError(f"unsupported collector control command: {self.command}")
        timestamp_ns = self.timestamp_monotonic_ns or time.monotonic_ns()
        return {
            "timestamp_monotonic_ns": np.array([timestamp_ns], dtype=np.int64),
            "sequence": np.array([self.sequence], dtype=np.int64),
            "command": np.array([self.command], dtype=np.int32),
        }

    def pack(self) -> bytes:
        return pack_topic_message(self.payload(), topic="collector_control")


@dataclass(frozen=True, slots=True)
class CollectorControlDecision:
    command: int | None
    ambiguous: bool = False


class PicoLeftModeControls:
    """Convert Pico left-stick clicks into rising-edge mode 0/1 toggles."""

    def __init__(self) -> None:
        self._previous_left_axis_click = False

    def update(self, *, left_axis_click: bool, current_mode: int) -> int | None:
        left_axis_click = bool(left_axis_click)
        left_axis_edge = left_axis_click and not self._previous_left_axis_click
        self._previous_left_axis_click = left_axis_click

        if not left_axis_edge or current_mode not in (0, 1):
            return None
        return 1 - current_mode


class PicoRightCollectorControls:
    """Convert Pico right-controller button states into rising-edge commands."""

    def __init__(self) -> None:
        self._previous_right_axis_click = False
        self._previous_a = False
        self._previous_b = False

    def update(
        self, *, right_axis_click: bool, a_pressed: bool, b_pressed: bool
    ) -> CollectorControlDecision:
        right_axis_click = bool(right_axis_click)
        a_pressed = bool(a_pressed)
        b_pressed = bool(b_pressed)
        right_axis_edge = right_axis_click and not self._previous_right_axis_click
        a_edge = a_pressed and not self._previous_a
        b_edge = b_pressed and not self._previous_b

        self._previous_right_axis_click = right_axis_click
        self._previous_a = a_pressed
        self._previous_b = b_pressed

        if a_pressed and b_pressed and (a_edge or b_edge):
            return CollectorControlDecision(None, ambiguous=True)
        if right_axis_edge:
            return CollectorControlDecision(COLLECTOR_CONTROL_START)
        if a_edge:
            return CollectorControlDecision(COLLECTOR_CONTROL_SAVE)
        if b_edge:
            return CollectorControlDecision(COLLECTOR_CONTROL_DISCARD)
        return CollectorControlDecision(None)


def _array(
    value: Any, shape: tuple[int, ...], dtype: np.dtype, name: str
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if result.dtype.kind == "f" and not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return np.ascontiguousarray(result)


@dataclass(slots=True)
class PoseSnapshot:
    frame_index: int
    source_timestamp_ns: int
    body_source_monotonic_ns: int
    mode: int
    reset_generation: int
    cmd_vel: np.ndarray
    g1_qpos: np.ndarray
    body_valid: bool
    left_wuji_qpos: np.ndarray
    right_wuji_qpos: np.ndarray
    left_wuji_qpos_valid: bool = False
    right_wuji_qpos_valid: bool = False
    sequence_done: bool = False
    kill: bool = False
    publish_monotonic_ns: int = 0

    def payload(self) -> dict[str, np.ndarray]:
        qpos = _array(self.g1_qpos, (36,), np.float32, "g1_qpos")
        quat_norm = float(np.linalg.norm(qpos[3:7]))
        if quat_norm < 1e-6:
            raise ValueError("g1_qpos root quaternion has zero norm")
        qpos = qpos.copy()
        qpos[3:7] /= quat_norm
        publish_ns = self.publish_monotonic_ns or time.monotonic_ns()
        return {
            "frame_index": np.array([self.frame_index], dtype=np.int64),
            "hand_frame_index": np.array([self.frame_index], dtype=np.int64),
            "source_timestamp_ns": np.array([self.source_timestamp_ns], dtype=np.int64),
            "body_source_monotonic_ns": np.array(
                [self.body_source_monotonic_ns], dtype=np.int64
            ),
            "publish_monotonic_ns": np.array([publish_ns], dtype=np.int64),
            "timestamp_monotonic_ns": np.array(
                [self.body_source_monotonic_ns or publish_ns], dtype=np.int64
            ),
            "mode": np.array([self.mode], dtype=np.int32),
            "reset_generation": np.array([self.reset_generation], dtype=np.int64),
            "kill": np.array([self.kill], dtype=np.bool_),
            "cmd_vel": _array(self.cmd_vel, (3,), np.float32, "cmd_vel"),
            "g1_qpos": qpos,
            "body_valid": np.array([self.body_valid], dtype=np.bool_),
            "left_wuji_qpos": _array(
                self.left_wuji_qpos, (20,), np.float32, "left_wuji_qpos"
            ),
            "right_wuji_qpos": _array(
                self.right_wuji_qpos, (20,), np.float32, "right_wuji_qpos"
            ),
            "left_wuji_qpos_valid": np.array(
                [self.left_wuji_qpos_valid], dtype=np.bool_
            ),
            "right_wuji_qpos_valid": np.array(
                [self.right_wuji_qpos_valid], dtype=np.bool_
            ),
            "sequence_done": np.array([self.sequence_done], dtype=np.bool_),
        }

    def pack(self) -> bytes:
        return pack_topic_message(self.payload())


@dataclass(slots=True)
class TeleopRawSnapshot:
    """Source-normalized body and hand observations paired with one pose frame."""

    frame_index: int
    source_timestamp_ns: int
    publish_monotonic_ns: int
    body_poses: np.ndarray
    body_valid: bool
    left_hand_keypoints: np.ndarray
    right_hand_keypoints: np.ndarray
    left_hand_valid: bool
    right_hand_valid: bool
    hand_source: int

    def payload(self) -> dict[str, np.ndarray]:
        return {
            "frame_index": np.array([self.frame_index], dtype=np.int64),
            "source_timestamp_ns": np.array([self.source_timestamp_ns], dtype=np.int64),
            "publish_monotonic_ns": np.array(
                [self.publish_monotonic_ns], dtype=np.int64
            ),
            "timestamp_monotonic_ns": np.array(
                [self.publish_monotonic_ns], dtype=np.int64
            ),
            "body_poses": _array(self.body_poses, (24, 7), np.float32, "body_poses"),
            "body_valid": np.array([self.body_valid], dtype=np.bool_),
            "left_hand_keypoints": _array(
                self.left_hand_keypoints, (21, 3), np.float32, "left_hand_keypoints"
            ),
            "right_hand_keypoints": _array(
                self.right_hand_keypoints, (21, 3), np.float32, "right_hand_keypoints"
            ),
            "left_hand_valid": np.array([self.left_hand_valid], dtype=np.bool_),
            "right_hand_valid": np.array([self.right_hand_valid], dtype=np.bool_),
            "hand_source": np.array([self.hand_source], dtype=np.int32),
        }

    def pack(self) -> bytes:
        return pack_topic_message(self.payload(), topic="teleop_raw")


def scalar(payload: dict[str, np.ndarray], name: str, default: Any = None) -> Any:
    if name not in payload:
        return default
    value = np.asarray(payload[name]).reshape(-1)
    return value[0].item() if value.size else default


class ZmqPoseSource:
    """Nonblocking latest-only consumer used by both sim and real HGPT loops."""

    def __init__(self, endpoint: str, *, context=None):
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for --pose-endpoint") from exc
        self._zmq = zmq
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, POSE_TOPIC)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(endpoint)
        self._payload: dict[str, np.ndarray] | None = None
        self._last_frame = -1
        self._last_reset_generation = -1
        self._reset_pending = False

    def refresh(self) -> dict[str, np.ndarray] | None:
        latest = None
        while True:
            try:
                latest = self._socket.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
        if latest is None:
            return self._payload
        topic, payload = decode_topic_message(latest)
        if topic != "pose":
            return self._payload
        frame = int(scalar(payload, "frame_index", -1))
        if frame <= self._last_frame:
            return self._payload
        qpos = _array(payload.get("g1_qpos"), (36,), np.float32, "g1_qpos")
        if bool(scalar(payload, "body_valid", False)):
            norm = float(np.linalg.norm(qpos[3:7]))
            if norm < 1e-6:
                raise ValueError("received qpos has zero root quaternion")
        generation = int(scalar(payload, "reset_generation", 0))
        if (
            self._last_reset_generation >= 0
            and generation != self._last_reset_generation
        ):
            self._reset_pending = True
        self._last_reset_generation = generation
        self._last_frame = frame
        self._payload = payload
        return payload

    def read(self) -> tuple[np.ndarray, float]:
        payload = self.refresh()
        if payload is None or not bool(scalar(payload, "body_valid", False)):
            return np.zeros(36, dtype=np.float32), 0.0
        timestamp_ns = int(scalar(payload, "body_source_monotonic_ns", 0))
        return np.asarray(
            payload["g1_qpos"], dtype=np.float32
        ).copy(), timestamp_ns / 1e9

    def command(self):
        """Return manager-owned mode and velocity as a keyboard-compatible command."""
        from deploy.keyboard_cmd import HighCommand

        payload = self.refresh()
        if payload is None:
            return HighCommand()
        velocity = _array(
            payload.get("cmd_vel", np.zeros(3)), (3,), np.float32, "cmd_vel"
        )
        return HighCommand(
            vel_lin_x=float(velocity[0]),
            vel_lin_y=float(velocity[1]),
            vel_ang_yaw=float(velocity[2]),
            mode=max(0, int(scalar(payload, "mode", 0))),
            kill=bool(scalar(payload, "kill", False)),
        )

    def step_command(self):
        return self.command()

    def check_reset_request(self) -> bool:
        pending = self._reset_pending
        self._reset_pending = False
        return pending

    def close(self) -> None:
        self._socket.close(linger=0)
