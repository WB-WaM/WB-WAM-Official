from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SONIC_ROOT = REPO_ROOT / "tracker" / "sonic"
if str(SONIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SONIC_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

from .action_schema import SonicAction  # noqa: E402


@dataclass(frozen=True)
class SonicV4Payload:
    frame_index: int
    timestamp_monotonic_ns: int
    action: SonicAction
    replay_target_real: bool = False
    left_wuji_qpos_valid: bool = True
    right_wuji_qpos_valid: bool = True
    hand_frame_index: int | None = None
    timestamp_monotonic: float | None = None


def _f32_vector(value: np.ndarray, *, dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (dim,):
        raise ValueError(f"{name} has shape {array.shape}, expected {(dim,)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")
    return np.ascontiguousarray(array)


def build_pose_v4_message(payload: SonicV4Payload) -> bytes:
    token = _f32_vector(payload.action.token_state, dim=64, name="token_state")
    left = _f32_vector(payload.action.left_wuji_qpos, dim=20, name="left_wuji_qpos")
    right = _f32_vector(payload.action.right_wuji_qpos, dim=20, name="right_wuji_qpos")
    timestamp_ns = int(payload.timestamp_monotonic_ns)
    timestamp_s = (
        float(payload.timestamp_monotonic) if payload.timestamp_monotonic is not None else timestamp_ns / 1e9
    )
    if not np.isfinite(timestamp_s):
        raise ValueError("timestamp_monotonic must be finite")
    hand_frame_index = payload.frame_index if payload.hand_frame_index is None else payload.hand_frame_index
    return pack_pose_message(
        {
            "token_state": token,
            "frame_index": np.array([payload.frame_index], dtype=np.int64),
            "left_wuji_qpos": left,
            "right_wuji_qpos": right,
            "left_wuji_qpos_valid": np.array([payload.left_wuji_qpos_valid], dtype=bool),
            "right_wuji_qpos_valid": np.array([payload.right_wuji_qpos_valid], dtype=bool),
            "hand_frame_index": np.array([hand_frame_index], dtype=np.int64),
            "replay_target_real": np.array([payload.replay_target_real], dtype=bool),
            "timestamp_monotonic": np.array([timestamp_s], dtype=np.float64),
            "timestamp_monotonic_ns": np.array([timestamp_ns], dtype=np.int64),
        },
        topic="pose",
        version=4,
    )


def build_planner_idle_message() -> bytes:
    return build_planner_message(
        mode=0,
        movement=[0.0, 0.0, 0.0],
        facing=[1.0, 0.0, 0.0],
        speed=-1.0,
        height=-1.0,
    )


def build_mode_command_message(*, planner: bool) -> bytes:
    """Select planner or streamed mode without starting/stopping control."""

    return build_command_message(start=False, stop=False, planner=planner)


class SonicV4Publisher:
    def __init__(self, *, port: int = 5556):
        import zmq

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._endpoint = f"tcp://*:{port}"
        self._socket.bind(self._endpoint)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def send_stream_mode(self, *, repeats: int = 3) -> None:
        message = build_mode_command_message(planner=False)
        for _ in range(repeats):
            self._socket.send(message)
            time.sleep(0.03)

    def send_planner_mode(self, *, repeats: int = 3) -> None:
        message = build_mode_command_message(planner=True)
        for _ in range(repeats):
            self._socket.send(message)
            time.sleep(0.03)

    def send_start(self, *, planner: bool = False, repeats: int = 3) -> None:
        message = build_command_message(start=True, stop=False, planner=planner)
        for _ in range(repeats):
            self._socket.send(message)
            time.sleep(0.03)

    def send_stop(self, *, repeats: int = 3) -> None:
        message = build_command_message(start=False, stop=True, planner=False)
        for _ in range(repeats):
            self._socket.send(message)
            time.sleep(0.03)

    def send_planner_idle(self, *, repeats: int = 1) -> None:
        message = build_planner_idle_message()
        for _ in range(repeats):
            self._socket.send(message)
            time.sleep(0.03)

    def publish(self, payload: SonicV4Payload) -> None:
        self._socket.send(build_pose_v4_message(payload))

    def close(self) -> None:
        self._socket.close(0)
