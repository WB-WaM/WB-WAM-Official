"""Latest-only ZMQ subscribers with pose/raw frame pairing."""

from __future__ import annotations

from collections import OrderedDict, deque
import logging
import time

import numpy as np

from .codec import decode_topic_message
from .interpolation import (
    interpolate_pair,
    interpolate_sample,
    latest_causal_sample,
)
from .schema import PairedPose, WireSample

LOGGER = logging.getLogger(__name__)
DECODE_WARNING_GRACE_NS = 1_000_000_000
DECODE_WARNING_REPEAT_NS = 5_000_000_000

COLLECTOR_CONTROL_COMMAND_NAMES = {
    1: "start",
    2: "save",
    3: "discard",
}


class SubscriberHub:
    def __init__(
        self,
        pose_endpoint: str,
        state_action_endpoint: str,
        hand_status_endpoint: str = "",
    ) -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._poller = zmq.Poller()
        self._pose = self._subscriber(pose_endpoint, "pose")
        self._raw = self._subscriber(pose_endpoint, "teleop_raw")
        self._state = self._subscriber(state_action_endpoint, "robot_state_action")
        self._collector_control = self._subscriber(pose_endpoint, "collector_control")
        self._hand = self._subscriber(hand_status_endpoint, "hand_status") if hand_status_endpoint else None
        for socket in (self._pose, self._raw, self._state, self._collector_control):
            self._poller.register(socket, zmq.POLLIN)
        if self._hand is not None:
            self._poller.register(self._hand, zmq.POLLIN)

        self._poses: OrderedDict[int, WireSample] = OrderedDict()
        self._raws: OrderedDict[int, WireSample] = OrderedDict()
        self.latest_pair: PairedPose | None = None
        self.latest_state: WireSample | None = None
        self.latest_hand: WireSample | None = None
        self._pair_history: deque[PairedPose] = deque(maxlen=256)
        self._state_history: deque[WireSample] = deque(maxlen=256)
        self._hand_history: deque[WireSample] = deque(maxlen=256)
        self.decode_errors = 0
        self.duplicate_frames = 0
        self.frame_gaps = 0
        self._last_pair_frame = -1
        self._pending_control_commands: deque[str] = deque()
        self._last_control_event: tuple[int, int] | None = None
        self._decode_warning_grace_until_ns = time.monotonic_ns() + DECODE_WARNING_GRACE_NS
        self._decode_warning_last_ns: dict[str, int] = {}
        self._decode_warning_active: set[str] = set()

    def _subscriber(self, endpoint: str, topic: str):
        socket = self._context.socket(self._zmq.SUB)
        socket.setsockopt(self._zmq.RCVHWM, 512)
        socket.setsockopt(self._zmq.SUBSCRIBE, topic.encode())
        socket.connect(endpoint)
        return socket

    def _recv(self, socket, topic: str) -> WireSample | None:
        try:
            message = socket.recv(self._zmq.NOBLOCK)
            decoded = decode_topic_message(message, topic)
            self._recover_decode_warning(f"decode.{topic}")
            return WireSample(topic, time.monotonic_ns(), decoded.payload)
        except self._zmq.Again:
            return None
        except Exception as exc:
            self.decode_errors += 1
            self._warn_decode(
                f"decode.{topic}",
                "failed to decode %s: %s",
                topic,
                exc,
            )
            return None

    def _warn_decode(
        self,
        key: str,
        message: str,
        *args,
        now_ns: int | None = None,
    ) -> None:
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        if now_ns < self._decode_warning_grace_until_ns:
            return
        last_ns = self._decode_warning_last_ns.get(key)
        if last_ns is not None and now_ns - last_ns < DECODE_WARNING_REPEAT_NS:
            return
        LOGGER.warning(message, *args)
        self._decode_warning_last_ns[key] = now_ns
        self._decode_warning_active.add(key)

    def _recover_decode_warning(self, key: str) -> None:
        if key not in self._decode_warning_active:
            return
        self._decode_warning_active.remove(key)
        LOGGER.info("recovered: %s", key)

    def _remember(self, cache: OrderedDict[int, WireSample], sample: WireSample):
        frame = sample.frame_index
        if frame < 0:
            return
        if frame in cache:
            self.duplicate_frames += 1
        cache[frame] = sample
        cache.move_to_end(frame)
        while len(cache) > 64:
            cache.popitem(last=False)

    def _pair(self) -> None:
        common = set(self._poses).intersection(self._raws)
        if not common:
            return
        for frame in sorted(value for value in common if value > self._last_pair_frame):
            if self._last_pair_frame >= 0 and frame > self._last_pair_frame + 1:
                self.frame_gaps += frame - self._last_pair_frame - 1
            pair = PairedPose(frame, self._poses[frame], self._raws[frame])
            self.latest_pair = pair
            self._pair_history.append(pair)
            self._last_pair_frame = frame
        for cache in (self._poses, self._raws):
            for old in tuple(cache):
                if old <= self._last_pair_frame:
                    cache.pop(old, None)

    def drain(self, timeout_ms: int = 0) -> None:
        first = True
        while True:
            events = dict(self._poller.poll(timeout_ms if first else 0))
            first = False
            if not events:
                self._pair()
                return
            if self._pose in events:
                sample = self._recv(self._pose, "pose")
                if sample is not None:
                    self._remember(self._poses, sample)
            if self._raw in events:
                sample = self._recv(self._raw, "teleop_raw")
                if sample is not None:
                    self._remember(self._raws, sample)
            if self._state in events:
                sample = self._recv(self._state, "robot_state_action")
                if sample is not None:
                    self.latest_state = sample
                    self._state_history.append(sample)
            if self._collector_control in events:
                sample = self._recv(self._collector_control, "collector_control")
                if sample is not None:
                    self._remember_collector_control(sample)
            if self._hand is not None and self._hand in events:
                sample = self._recv(self._hand, "hand_status")
                if sample is not None:
                    self.latest_hand = sample
                    self._hand_history.append(sample)

    def interpolated_inputs_at(self, target_ns: int) -> tuple[PairedPose, WireSample, WireSample | None] | None:
        """Return pose/state interpolated at target_ns when bracketed."""
        pair = interpolate_pair(self._pair_history, target_ns)
        state = interpolate_sample(self._state_history, target_ns, topic="robot_state_action")
        if pair is None or state is None:
            return None
        hand = interpolate_sample(self._hand_history, target_ns, topic="hand_status")
        if hand is None:
            hand = latest_causal_sample(self._hand_history, target_ns, topic="hand_status")
        return pair, state, hand

    def _remember_collector_control(self, sample: WireSample) -> None:
        payload = sample.payload
        try:
            sequence_values = np.asarray(payload["sequence"]).reshape(-1)
            command_values = np.asarray(payload["command"]).reshape(-1)
            timestamp_values = np.asarray(payload["timestamp_monotonic_ns"]).reshape(-1)
            if not (sequence_values.size == command_values.size == timestamp_values.size == 1):
                raise ValueError("collector_control fields must be scalar")
            sequence = int(sequence_values[0])
            command = int(command_values[0])
            timestamp_ns = int(timestamp_values[0])
            if sequence <= 0 or timestamp_ns <= 0:
                raise ValueError("collector_control sequence/timestamp must be positive")
            command_name = COLLECTOR_CONTROL_COMMAND_NAMES.get(command)
            if command_name is None:
                raise ValueError(f"unsupported collector_control command: {command}")
        except Exception as exc:
            self.decode_errors += 1
            self._warn_decode(
                "decode.collector_control_payload",
                "failed to decode collector_control payload: %s",
                exc,
            )
            return

        self._recover_decode_warning("decode.collector_control_payload")

        event = (timestamp_ns, sequence)
        if event == self._last_control_event:
            return
        self._last_control_event = event
        self._pending_control_commands.append(command_name)

    def take_control_commands(self) -> list[str]:
        commands = list(self._pending_control_commands)
        self._pending_control_commands.clear()
        return commands

    def close(self) -> None:
        for socket in (
            self._pose,
            self._raw,
            self._state,
            self._collector_control,
            self._hand,
        ):
            if socket is not None:
                socket.close(linger=0)
