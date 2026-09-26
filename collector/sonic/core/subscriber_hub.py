from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

from .pose_pairing import IncrementalPosePairer
from .schema import (
    CollectorControlSample,
    DrainedSamples,
    HandStatusSample,
    PoseControlSample,
    PoseRawSample,
    PoseSample,
    RobotStateActionSample,
    make_collector_control_sample,
    make_hand_status_sample,
    make_pose_control_sample,
    make_pose_raw_sample,
    make_state_action_sample,
)


LOGGER = logging.getLogger(__name__)
HEADER_SIZE = 1280
DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "bool": np.bool_,
}


@dataclass
class DecodedMessage:
    topic: str
    version: int
    payload: dict[str, np.ndarray]


def decode_topic_message(message: bytes, topic: str) -> DecodedMessage:
    if not message.startswith(topic.encode("utf-8")):
        raise ValueError(f"message does not start with topic '{topic}'")

    offset = len(topic)
    header_bytes = message[offset : offset + HEADER_SIZE]
    header_text = header_bytes.rstrip(b"\x00").decode("utf-8")
    header = json.loads(header_text)
    payload_bytes = memoryview(message[offset + HEADER_SIZE :])

    payload: dict[str, np.ndarray] = {}
    cursor = 0
    for field in header["fields"]:
        dtype = DTYPE_MAP[field["dtype"]]
        shape = tuple(field["shape"])
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        nbytes = np.dtype(dtype).itemsize * count
        raw = payload_bytes[cursor : cursor + nbytes]
        cursor += nbytes
        payload[field["name"]] = np.frombuffer(raw, dtype=dtype, count=count).reshape(shape).copy()

    return DecodedMessage(topic=topic, version=int(header["v"]), payload=payload)


class SubscriberHub:
    def __init__(self, pose_endpoint: str, state_action_endpoint: str, hand_status_endpoint: str = ""):
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for the standalone collector") from exc

        self._zmq = zmq
        self.context = zmq.Context.instance()
        self.poller = zmq.Poller()
        self.pose_socket = self._make_subscriber(pose_endpoint, "pose")
        self.pose_raw_socket = self._make_subscriber(pose_endpoint, "teleop_raw")
        self.state_socket = self._make_subscriber(state_action_endpoint, "robot_state_action")
        self.collector_control_socket = self._make_subscriber(pose_endpoint, "collector_control")
        self.hand_status_socket = (
            self._make_subscriber(hand_status_endpoint, "hand_status") if hand_status_endpoint else None
        )
        self.poller.register(self.pose_socket, zmq.POLLIN)
        self.poller.register(self.pose_raw_socket, zmq.POLLIN)
        self.poller.register(self.state_socket, zmq.POLLIN)
        self.poller.register(self.collector_control_socket, zmq.POLLIN)
        if self.hand_status_socket is not None:
            self.poller.register(self.hand_status_socket, zmq.POLLIN)
        self._pose_pairer = IncrementalPosePairer()
        self.latest_pose_control: PoseControlSample | None = None
        self.latest_pose_raw: PoseRawSample | None = None
        self.latest_pose_sample: PoseSample | None = None
        self.latest_state_action: RobotStateActionSample | None = None
        self.latest_hand_status: HandStatusSample | None = None
        self.latest_collector_control: CollectorControlSample | None = None

    def _make_subscriber(self, endpoint: str, topic: str):
        socket = self.context.socket(self._zmq.SUB)
        socket.setsockopt(self._zmq.RCVHWM, 512)
        socket.setsockopt(self._zmq.SUBSCRIBE, topic.encode("utf-8"))
        socket.connect(endpoint)
        return socket

    def drain(self, timeout_ms: int = 0) -> DrainedSamples:
        drained = DrainedSamples(
            pose_controls=[],
            pose_raws=[],
            state_actions=[],
            hand_statuses=[],
            collector_controls=[],
        )
        first_poll = True
        while True:
            events = dict(self.poller.poll(timeout_ms if first_poll else 0))
            first_poll = False
            if not events:
                pairs = self._pose_pairer.push_samples(drained.pose_controls, drained.pose_raws)
                if pairs:
                    self.latest_pose_sample = pairs[-1].sample
                return drained

            if self.pose_socket in events:
                pose_control = self._recv_pose()
                if pose_control is not None:
                    self.latest_pose_control = pose_control
                    drained.pose_controls.append(pose_control)

            if self.pose_raw_socket in events:
                pose_raw = self._recv_pose_raw()
                if pose_raw is not None:
                    self.latest_pose_raw = pose_raw
                    drained.pose_raws.append(pose_raw)

            if self.state_socket in events:
                state_action = self._recv_state_action()
                if state_action is not None:
                    self.latest_state_action = state_action
                    drained.state_actions.append(state_action)

            if self.collector_control_socket in events:
                collector_control = self._recv_collector_control()
                if collector_control is not None:
                    self.latest_collector_control = collector_control
                    drained.collector_controls.append(collector_control)

            if self.hand_status_socket is not None and self.hand_status_socket in events:
                hand_status = self._recv_hand_status()
                if hand_status is not None:
                    self.latest_hand_status = hand_status
                    drained.hand_statuses.append(hand_status)

    def _recv_pose(self) -> PoseControlSample | None:
        try:
            message = self.pose_socket.recv(self._zmq.NOBLOCK)
            decoded = decode_topic_message(message, "pose")
            return make_pose_control_sample(decoded.payload)
        except self._zmq.Again:
            return self.latest_pose_control
        except Exception as exc:
            LOGGER.warning("failed to decode pose message: %s", exc)
            return self.latest_pose_control

    def _recv_pose_raw(self) -> PoseRawSample | None:
        try:
            message = self.pose_raw_socket.recv(self._zmq.NOBLOCK)
            decoded = decode_topic_message(message, "teleop_raw")
            return make_pose_raw_sample(decoded.payload)
        except self._zmq.Again:
            return self.latest_pose_raw
        except Exception as exc:
            LOGGER.warning("failed to decode teleop_raw message: %s", exc)
            return self.latest_pose_raw

    def _recv_state_action(self) -> RobotStateActionSample | None:
        try:
            message = self.state_socket.recv(self._zmq.NOBLOCK)
            decoded = decode_topic_message(message, "robot_state_action")
            return make_state_action_sample(decoded.payload)
        except self._zmq.Again:
            return self.latest_state_action
        except Exception as exc:
            LOGGER.warning("failed to decode robot_state_action message: %s", exc)
            return self.latest_state_action

    def _recv_collector_control(self) -> CollectorControlSample | None:
        try:
            message = self.collector_control_socket.recv(self._zmq.NOBLOCK)
            decoded = decode_topic_message(message, "collector_control")
            return make_collector_control_sample(decoded.payload)
        except self._zmq.Again:
            return None
        except Exception as exc:
            LOGGER.warning("failed to decode collector_control message: %s", exc)
            return None

    def _recv_hand_status(self) -> HandStatusSample | None:
        if self.hand_status_socket is None:
            return None
        try:
            message = self.hand_status_socket.recv(self._zmq.NOBLOCK)
            decoded = decode_topic_message(message, "hand_status")
            return make_hand_status_sample(decoded.payload)
        except self._zmq.Again:
            return self.latest_hand_status
        except Exception as exc:
            LOGGER.warning("failed to decode hand_status message: %s", exc)
            return self.latest_hand_status

    def close(self) -> None:
        self.pose_socket.close(0)
        self.pose_raw_socket.close(0)
        self.state_socket.close(0)
        self.collector_control_socket.close(0)
        if self.hand_status_socket is not None:
            self.hand_status_socket.close(0)
