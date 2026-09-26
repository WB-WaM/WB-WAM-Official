"""Collector telemetry publisher for one coherent HGPT control cycle."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from deploy.pose_zmq import pack_topic_message

POLICY_HOLD = 0
POLICY_WALK = 1
POLICY_TRACK = 2


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"{name} must be ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return np.ascontiguousarray(result)


@dataclass(slots=True)
class RobotStateActionSnapshot:
    state_index: int
    sensor_monotonic_ns: int
    mode: int
    cmd_vel: np.ndarray
    base_quat: np.ndarray
    base_ang_vel: np.ndarray
    base_accel: np.ndarray
    body_q: np.ndarray
    body_dq: np.ndarray
    body_tau: np.ndarray
    reference_qpos: np.ndarray
    reference_valid: bool
    policy_kind: int
    policy_obs: np.ndarray
    policy_action: np.ndarray
    body_action: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    lowcmd_published: bool
    publish_monotonic_ns: int = 0

    def payload(self) -> dict[str, np.ndarray]:
        policy_obs = np.asarray(self.policy_obs, dtype=np.float32).reshape(-1)
        policy_action = np.asarray(self.policy_action, dtype=np.float32).reshape(-1)
        if not np.isfinite(policy_obs).all() or not np.isfinite(policy_action).all():
            raise ValueError("policy telemetry contains NaN/Inf")
        publish_ns = self.publish_monotonic_ns or time.monotonic_ns()
        return {
            "state_index": np.array([self.state_index], dtype=np.int64),
            "timestamp_monotonic_ns": np.array(
                [self.sensor_monotonic_ns], dtype=np.int64
            ),
            "publish_monotonic_ns": np.array([publish_ns], dtype=np.int64),
            "mode": np.array([self.mode], dtype=np.int32),
            "cmd_vel": _vector(self.cmd_vel, 3, "cmd_vel"),
            "base_quat": _vector(self.base_quat, 4, "base_quat"),
            "base_ang_vel": _vector(self.base_ang_vel, 3, "base_ang_vel"),
            "base_accel": _vector(self.base_accel, 3, "base_accel"),
            "body_q": _vector(self.body_q, 29, "body_q"),
            "body_dq": _vector(self.body_dq, 29, "body_dq"),
            "body_tau": _vector(self.body_tau, 29, "body_tau"),
            "reference_qpos": _vector(self.reference_qpos, 36, "reference_qpos"),
            "reference_valid": np.array([self.reference_valid], dtype=np.bool_),
            "policy_kind": np.array([self.policy_kind], dtype=np.int32),
            "policy_obs": np.ascontiguousarray(policy_obs),
            "policy_action": np.ascontiguousarray(policy_action),
            "body_action": _vector(self.body_action, 29, "body_action"),
            "kp": _vector(self.kp, 29, "kp"),
            "kd": _vector(self.kd, 29, "kd"),
            "lowcmd_published": np.array([self.lowcmd_published], dtype=np.bool_),
        }

    def pack(self) -> bytes:
        return pack_topic_message(self.payload(), topic="robot_state_action", version=1)


class RobotStateActionPublisher:
    def __init__(self, bind: str):
        import zmq

        self._zmq = zmq
        self._socket = zmq.Context.instance().socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 4)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(bind)
        self.bind = bind
        self.dropped = 0

    def publish(self, snapshot: RobotStateActionSnapshot) -> None:
        try:
            self._socket.send(snapshot.pack(), flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            self.dropped += 1

    def close(self) -> None:
        self._socket.close(linger=0)
