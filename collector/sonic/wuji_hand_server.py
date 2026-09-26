from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import time

from core.subscriber_hub import decode_topic_message
import numpy as np

LOGGER = logging.getLogger(__name__)
HEADER_SIZE = 1280
HAND_STATUS_TOPIC = "hand_status"
WUJI_FINGER_COUNT = 5
WUJI_JOINTS_PER_FINGER = 4
WUJI_QPOS_SIZE = WUJI_FINGER_COUNT * WUJI_JOINTS_PER_FINGER
WUJI_FILTER_CUTOFF_HZ = 2.0
READ_WARNING_INTERVAL_SECONDS = 2.0
COMMAND_CLAMP_WARNING_INTERVAL_SECONDS = 2.0
REPLAY_POSE_WATCHDOG_MS = 250
SERVER_POLL_MAX_MS = 50

# Keep these limits synchronized with the Wuji hand URDFs and
# gear_sonic_deploy/.../include/wuji_hand_constants.hpp.
WUJI_REPLAY_COMMAND_LOWER_LIMITS = np.asarray(
    [
         0.0475, -0.1387, -0.4642, -0.4699,
        -0.1585, -0.3700, -0.4777, -0.4683,
        -0.1644, -0.3700, -0.4739, -0.4684,
        -0.1554, -0.3700, -0.4765, -0.4777,
        -0.1626, -0.3700, -0.4768, -0.4683,
    ],
    dtype=np.float32,
)
WUJI_REPLAY_COMMAND_UPPER_LIMITS = np.asarray(
    [
        1.6033, 0.9324, 1.5623, 1.5568,
        1.5604, 0.3700, 1.5485, 1.5753,
        1.5516, 0.3700, 1.5512, 1.5745,
        1.5585, 0.3700, 1.5487, 1.5634,
        1.5585, 0.3700, 1.5490, 1.5735,
    ],
    dtype=np.float32,
)
try:
    import zmq
except ImportError:
    zmq = None


def _build_header(fields: list[dict], *, version: int = 1, count: int = 1) -> bytes:
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def pack_topic_message(payload: dict[str, np.ndarray], *, topic: str, version: int = 1) -> bytes:
    fields: list[dict] = []
    buffers: list[bytes] = []

    for name, value in payload.items():
        array = np.asarray(value)
        if array.dtype == np.float32:
            dtype = "f32"
        elif array.dtype == np.float64:
            dtype = "f64"
        elif array.dtype == np.int32:
            dtype = "i32"
        elif array.dtype == np.int64:
            dtype = "i64"
        elif array.dtype == np.uint8:
            dtype = "u8"
        elif array.dtype == np.bool_:
            dtype = "bool"
        else:
            array = array.astype(np.float32)
            dtype = "f32"

        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        if array.dtype.byteorder == ">":
            array = array.astype(array.dtype.newbyteorder("<"))

        fields.append({"name": name, "dtype": dtype, "shape": list(array.shape)})
        buffers.append(array.tobytes())

    return topic.encode("utf-8") + _build_header(fields, version=version) + b"".join(buffers)


def _as_bool(payload: dict[str, np.ndarray], key: str, default: bool = False) -> bool:
    if key not in payload:
        return default
    value = np.asarray(payload[key]).reshape(-1)
    if value.size == 0:
        return default
    return bool(value[0])


def _as_int(payload: dict[str, np.ndarray], key: str, *, use_last: bool = False, default: int = -1) -> int:
    if key not in payload:
        return default
    value = np.asarray(payload[key]).reshape(-1)
    if value.size == 0:
        return default
    return int(value[-1] if use_last else value[0])


def _timestamp_pc_ns(payload: dict[str, np.ndarray]) -> int:
    if "timestamp_monotonic_ns" in payload:
        return int(np.asarray(payload["timestamp_monotonic_ns"]).reshape(-1)[0])
    if "timestamp_monotonic" in payload:
        return int(float(np.asarray(payload["timestamp_monotonic"]).reshape(-1)[0]) * 1e9)
    return -1


def _is_replay_pose(payload: dict[str, np.ndarray]) -> bool:
    """Return whether a pose uses the explicit replay wire protocol."""

    # Both replay routes always provide hand_frame_index. catch_up is unique
    # to v1 replay and token_state is unique to v4, so either also makes a
    # partially malformed packet fail closed instead of falling back to the
    # legacy pose path.
    return any(
        key in payload for key in ("hand_frame_index", "catch_up", "token_state")
    )


def _source_frame_index(payload: dict[str, np.ndarray]) -> int:
    """Return the frame that produced the current hand command.

    Replay v1 packets carry a vector-valued ``frame_index`` for the current
    row and its future reference window. The hand command belongs to the
    current row, never the future tail, so replay packets use the first value.
    Normal PICO packets carry a history window and publish the hand command for
    its final/current row. New replay packets provide an explicit scalar
    ``hand_frame_index`` which takes precedence.
    """

    if "frame_index" not in payload:
        raise ValueError("frame_index is required")
    frame_index = np.asarray(payload["frame_index"])
    if frame_index.size == 0 or frame_index.dtype.kind not in "iu":
        raise ValueError("frame_index must contain integer source rows")
    frame_values = frame_index.reshape(-1)
    current_frame = int(
        frame_values[0] if _is_replay_pose(payload) else frame_values[-1]
    )

    if (
        "hand_frame_index" in payload
        and np.asarray(payload["hand_frame_index"]).size > 0
    ):
        hand_frame = np.asarray(payload["hand_frame_index"])
        if hand_frame.shape != (1,) or hand_frame.dtype.kind not in "iu":
            raise ValueError("hand_frame_index must be an integer scalar [1]")
        hand_frame_index = int(hand_frame[0])
        if hand_frame_index != current_frame:
            raise ValueError(
                "hand_frame_index must equal frame_index current row: "
                f"{hand_frame_index} != {current_frame}"
            )
        return hand_frame_index
    return current_frame


def _qpos(
    payload: dict[str, np.ndarray],
    key: str,
    *,
    clamp_limits: bool = True,
) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"{key} is required")
    return _validate_qpos(
        payload[key],
        key=key,
        clamp_limits=clamp_limits,
    )


def _validate_qpos(
    value: np.ndarray,
    *,
    key: str,
    clamp_limits: bool = True,
) -> np.ndarray:
    qpos = np.asarray(value, dtype=np.float32)
    if qpos.shape != (WUJI_QPOS_SIZE,):
        raise ValueError(
            f"{key} expected shape {(WUJI_QPOS_SIZE,)}, got {qpos.shape}"
        )
    if not np.all(np.isfinite(qpos)):
        raise ValueError(f"{key} contains NaN/Inf")
    if clamp_limits:
        qpos = np.clip(
            qpos,
            WUJI_REPLAY_COMMAND_LOWER_LIMITS,
            WUJI_REPLAY_COMMAND_UPPER_LIMITS,
        )
    return np.ascontiguousarray(qpos)


@dataclass
class HandSide:
    serial: str
    label: str
    device: object | None = None
    controller: object | None = None
    initialized: bool = False
    last_read_warning_time: float = 0.0


class WujiHandController:
    def __init__(self, *, left_serial: str, right_serial: str, dry_run: bool = False):
        self.dry_run = dry_run
        self.left = HandSide(serial=left_serial, label="left")
        self.right = HandSide(serial=right_serial, label="right")
        self._wujihandpy = None

    def initialize(self) -> None:
        if self.dry_run:
            LOGGER.warning("wuji_hand_server running in dry-run mode; no USB commands will be sent")
            self.left.initialized = bool(self.left.serial)
            self.right.initialized = bool(self.right.serial)
            return

        try:
            import wujihandpy
        except ImportError as exc:
            raise RuntimeError("wujihandpy is required for wuji_hand_server.py") from exc

        self._wujihandpy = wujihandpy
        self._initialize_side(self.left)
        self._initialize_side(self.right)
        if not self.left.initialized and not self.right.initialized:
            raise RuntimeError("no Wuji hands initialized; check serials, USB permissions, and cabling")

    def _initialize_side(self, side: HandSide) -> None:
        if not side.serial:
            LOGGER.warning("missing %s Wuji serial; that hand will be disabled", side.label)
            return

        assert self._wujihandpy is not None
        try:
            side.device = self._wujihandpy.Hand(side.serial)
            side.device.disable_thread_safe_check()
            side.device.write_joint_enabled(True)
            side.controller = side.device.realtime_controller(
                True,
                self._wujihandpy.filter.LowPass(WUJI_FILTER_CUTOFF_HZ),
            )
            time.sleep(0.5)
            side.initialized = True
            LOGGER.info(
                "initialized %s Wuji hand with serial %s (realtime upstream enabled)",
                side.label,
                side.serial,
            )
        except Exception as exc:
            side.device = None
            side.controller = None
            side.initialized = False
            LOGGER.warning("failed to initialize %s Wuji hand (%s): %s", side.label, side.serial, exc)

    def apply(self, *, is_left: bool, qpos: np.ndarray, valid: bool) -> bool:
        if not valid:
            return False
        side = self.left if is_left else self.right
        qpos = _validate_qpos(qpos, key=f"{side.label}_wuji_qpos")
        if not side.initialized or side.controller is None:
            return False
        if self.dry_run:
            return True

        try:
            command = np.asarray(qpos, dtype=np.float64).reshape(WUJI_FINGER_COUNT, WUJI_JOINTS_PER_FINGER)
            side.controller.set_joint_target_position(command)
            return True
        except Exception as exc:
            LOGGER.warning("failed to apply %s Wuji command: %s", side.label, exc)
            return False

    def read_actual_positions(self) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        left_actual, left_valid = self._read_side_actual_position(self.left)
        right_actual, right_valid = self._read_side_actual_position(self.right)
        return left_actual, right_actual, left_valid, right_valid

    def _read_side_actual_position(self, side: HandSide) -> tuple[np.ndarray, bool]:
        if self.dry_run or not side.initialized or side.controller is None:
            return np.zeros((WUJI_QPOS_SIZE,), dtype=np.float32), False

        try:
            actual = np.asarray(side.controller.get_joint_actual_position(), dtype=np.float32).reshape(-1)
            if actual.size != WUJI_QPOS_SIZE:
                raise ValueError(f"expected {WUJI_QPOS_SIZE} values, got {actual.size}")
            return actual.astype(np.float32, copy=False), True
        except Exception as exc:
            now = time.monotonic()
            if now - side.last_read_warning_time >= READ_WARNING_INTERVAL_SECONDS:
                LOGGER.warning("failed to read %s Wuji actual position: %s", side.label, exc)
                side.last_read_warning_time = now
            return np.zeros((WUJI_QPOS_SIZE,), dtype=np.float32), False

    def shutdown(self) -> None:
        for side in (self.left, self.right):
            try:
                if side.controller is not None:
                    side.controller.close()
            except Exception:
                pass
            try:
                if side.device is not None and not self.dry_run:
                    side.device.write_joint_enabled(False)
            except Exception:
                pass
            side.controller = None
            side.device = None
            side.initialized = False


class WujiHandServer:
    def __init__(
        self,
        *,
        pose_endpoint: str,
        status_bind: str,
        left_serial: str,
        right_serial: str,
        dry_run: bool = False,
    ):
        if zmq is None:
            raise RuntimeError("pyzmq is required for wuji_hand_server.py")

        self.pose_endpoint = pose_endpoint
        self.status_bind = status_bind
        self.context = zmq.Context.instance()
        self.pose_socket = self.context.socket(zmq.SUB)
        self.pose_socket.setsockopt(zmq.SUBSCRIBE, b"pose")
        self.pose_socket.setsockopt(zmq.SUBSCRIBE, b"command")
        self.pose_socket.setsockopt(zmq.RCVHWM, 256)
        self.pose_socket.connect(pose_endpoint)
        self._poller = zmq.Poller()
        self._poller.register(self.pose_socket, zmq.POLLIN)
        self.status_socket = self.context.socket(zmq.PUB)
        self.status_socket.setsockopt(zmq.LINGER, 0)
        self.status_socket.bind(status_bind)
        self.controller = WujiHandController(
            left_serial=left_serial,
            right_serial=right_serial,
            dry_run=dry_run,
        )
        self._last_replay_pose_receive_ns: int | None = None
        self._replay_watchdog_warned = False
        self._last_qpos_clamp_warning_ns: dict[str, int] = {}
        self._stopped = False

    def start(self) -> None:
        self.controller.initialize()
        LOGGER.info("wuji_hand_server subscribed to %s", self.pose_endpoint)
        LOGGER.info("wuji_hand_server publishing %s on %s", HAND_STATUS_TOPIC, self.status_bind)

    def _poll_timeout_ms(self, now_ns: int) -> int:
        if self._last_replay_pose_receive_ns is None:
            return SERVER_POLL_MAX_MS
        elapsed_ns = now_ns - self._last_replay_pose_receive_ns
        timeout_ns = REPLAY_POSE_WATCHDOG_MS * 1_000_000
        if elapsed_ns >= timeout_ns:
            if not getattr(self, "_replay_watchdog_warned", False):
                LOGGER.warning(
                    "replay pose exceeded %d ms; holding the last hand pose and waiting for recovery",
                    REPLAY_POSE_WATCHDOG_MS,
                )
                self._replay_watchdog_warned = True
            # Keep the controller alive and leave its last target latched. A
            # future replay packet will re-arm the watchdog automatically.
            self._last_replay_pose_receive_ns = None
            return SERVER_POLL_MAX_MS
        remaining_ms = math.ceil((timeout_ns - elapsed_ns) / 1_000_000)
        return max(1, min(SERVER_POLL_MAX_MS, remaining_ms))

    def serve_forever(self, *, monotonic_ns_fn=time.monotonic_ns) -> None:
        try:
            while True:
                timeout_ms = self._poll_timeout_ms(monotonic_ns_fn())
                events = dict(self._poller.poll(timeout_ms))
                if self.pose_socket not in events:
                    continue

                message = self.pose_socket.recv()
                receive_ns = monotonic_ns_fn()
                if message.startswith(b"command"):
                    decoded = decode_topic_message(message, "command")
                    if _as_bool(decoded.payload, "stop", default=False):
                        LOGGER.info(
                            "received stop command; holding the last Wuji hand pose "
                            "and waiting for the next bridge session"
                        )
                        self._last_replay_pose_receive_ns = None
                        self._replay_watchdog_warned = False
                        continue
                    if _as_bool(decoded.payload, "planner", default=False):
                        self._last_replay_pose_receive_ns = None
                        self._replay_watchdog_warned = False
                    continue
                if not message.startswith(b"pose"):
                    raise ValueError("received unsupported Wuji server topic")

                try:
                    decoded = decode_topic_message(message, "pose")
                    replay_pose = _is_replay_pose(decoded.payload)
                    if (
                        not replay_pose
                        and "left_wuji_qpos" not in decoded.payload
                        and "right_wuji_qpos" not in decoded.payload
                    ):
                        # Planner poses share the same topic but intentionally
                        # contain no hand command. Keep the current hand pose.
                        self._last_replay_pose_receive_ns = None
                        self._replay_watchdog_warned = False
                        continue
                    status = self._apply_pose_payload(
                        decoded.payload, receive_ns=receive_ns
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"malformed pose hand command: {exc}"
                    ) from exc
                if replay_pose:
                    self._last_replay_pose_receive_ns = receive_ns
                    self._replay_watchdog_warned = False
                self.status_socket.send(
                    pack_topic_message(
                        status, topic=HAND_STATUS_TOPIC, version=1
                    )
                )
        finally:
            self.stop()

    def _warn_qpos_clamped(
        self,
        *,
        key: str,
        original: np.ndarray,
        clamped: np.ndarray,
        now_ns: int,
    ) -> None:
        indices = np.flatnonzero(original != clamped)
        if indices.size == 0:
            return

        warning_times = getattr(self, "_last_qpos_clamp_warning_ns", None)
        if warning_times is None:
            warning_times = {}
            self._last_qpos_clamp_warning_ns = warning_times
        last_warning_ns = warning_times.get(key)
        warning_interval_ns = int(
            COMMAND_CLAMP_WARNING_INTERVAL_SECONDS * 1e9
        )
        if (
            last_warning_ns is not None
            and 0 <= now_ns - last_warning_ns < warning_interval_ns
        ):
            return

        index = int(indices[0])
        LOGGER.warning(
            "%s contained %d finite out-of-range joint(s); clamped to URDF "
            "limits (first: [%d] %.6f -> %.6f, range=[%.6f, %.6f])",
            key,
            int(indices.size),
            index,
            float(original[index]),
            float(clamped[index]),
            float(WUJI_REPLAY_COMMAND_LOWER_LIMITS[index]),
            float(WUJI_REPLAY_COMMAND_UPPER_LIMITS[index]),
        )
        warning_times[key] = now_ns

    def _apply_pose_payload(
        self,
        payload: dict[str, np.ndarray],
        *,
        receive_ns: int,
    ) -> dict[str, np.ndarray]:
        source_frame_index = _source_frame_index(payload)
        source_timestamp_pc_ns = _timestamp_pc_ns(payload)
        replay_pose = _is_replay_pose(payload)
        # Official legacy pose packets historically treated a missing validity
        # flag as valid. Replay packets are fail-safe: absence never enables a
        # hand command.
        validity_default = not replay_pose
        left_valid = _as_bool(
            payload, "left_wuji_qpos_valid", default=validity_default
        )
        right_valid = _as_bool(
            payload, "right_wuji_qpos_valid", default=validity_default
        )
        left_qpos = _qpos(
            payload,
            "left_wuji_qpos",
            clamp_limits=left_valid,
        )
        right_qpos = _qpos(
            payload,
            "right_wuji_qpos",
            clamp_limits=right_valid,
        )
        if left_valid:
            self._warn_qpos_clamped(
                key="left_wuji_qpos",
                original=np.asarray(
                    payload["left_wuji_qpos"], dtype=np.float32
                ),
                clamped=left_qpos,
                now_ns=receive_ns,
            )
        if right_valid:
            self._warn_qpos_clamped(
                key="right_wuji_qpos",
                original=np.asarray(
                    payload["right_wuji_qpos"], dtype=np.float32
                ),
                clamped=right_qpos,
                now_ns=receive_ns,
            )
        left_binary_closed = _as_bool(payload, "left_hand_binary_closed", default=False)
        right_binary_closed = _as_bool(payload, "right_hand_binary_closed", default=False)

        left_success = self.controller.apply(is_left=True, qpos=left_qpos, valid=left_valid)
        right_success = self.controller.apply(is_left=False, qpos=right_qpos, valid=right_valid)
        apply_ns = time.monotonic_ns()
        (
            left_actual,
            right_actual,
            left_actual_valid,
            right_actual_valid,
        ) = self.controller.read_actual_positions()
        read_ns = time.monotonic_ns()

        return {
            "source_frame_index": np.array([source_frame_index], dtype=np.int64),
            "source_timestamp_monotonic_pc_ns": np.array([source_timestamp_pc_ns], dtype=np.int64),
            "left_wuji_qpos_command": left_qpos.astype(np.float32),
            "right_wuji_qpos_command": right_qpos.astype(np.float32),
            "left_wuji_qpos_actual": left_actual.astype(np.float32),
            "right_wuji_qpos_actual": right_actual.astype(np.float32),
            "left_command_valid": np.array([left_valid], dtype=bool),
            "right_command_valid": np.array([right_valid], dtype=bool),
            "left_actual_position_valid": np.array([left_actual_valid], dtype=bool),
            "right_actual_position_valid": np.array([right_actual_valid], dtype=bool),
            "left_apply_success": np.array([left_success], dtype=bool),
            "right_apply_success": np.array([right_success], dtype=bool),
            "left_hand_binary_closed": np.array([left_binary_closed], dtype=bool),
            "right_hand_binary_closed": np.array([right_binary_closed], dtype=bool),
            "robot_receive_time_monotonic_ns": np.array([receive_ns], dtype=np.int64),
            "robot_apply_time_monotonic_ns": np.array([apply_ns], dtype=np.int64),
            "robot_read_time_monotonic_ns": np.array([read_ns], dtype=np.int64),
            # This is robot-side receive-to-apply time. End-to-end PC-to-robot latency
            # cannot be computed from monotonic clocks without a clock-offset estimate.
            "command_latency_ns": np.array([apply_ns - receive_ns], dtype=np.int64),
        }

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.controller.shutdown()
        self.pose_socket.close(0)
        self.status_socket.close(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot-side WujiHand server for remote PC deploy.")
    parser.add_argument("--pose-endpoint", required=True, help="PC pico_manager pose endpoint, e.g. tcp://192.168.123.222:5556")
    parser.add_argument("--status-bind", default="tcp://0.0.0.0:5559")
    parser.add_argument("--left-wuji-serial", default="")
    parser.add_argument("--right-wuji-serial", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )
    server = WujiHandServer(
        pose_endpoint=args.pose_endpoint,
        status_bind=args.status_bind,
        left_serial=args.left_wuji_serial,
        right_serial=args.right_wuji_serial,
        dry_run=args.dry_run,
    )
    server.start()
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
