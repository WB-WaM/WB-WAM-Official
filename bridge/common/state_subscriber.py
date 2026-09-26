from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Any

import numpy as np

from collector.sonic.core.schema import make_hand_status_sample, make_state_action_sample
from collector.sonic.core.subscriber_hub import decode_topic_message

from .state_layouts import (
    STATE_LAYOUT_GRAVITY,
    STATE_LAYOUT_QUAT,
    base_quat_to_gravity,
    normalize_state_layout,
    state_dim_for_layout,
)


@dataclass
class LatestRobotInputs:
    obs: dict[str, Any]
    hand_feedback: dict[str, Any]
    state_receive_monotonic_ns: int = -1
    state_payload: dict[str, Any] = field(default_factory=dict)
    hand_receive_monotonic_ns: int = -1


@dataclass(frozen=True)
class BaseQuatSnapshot:
    """A validated robot orientation sample received on this host."""

    base_quat_wxyz: np.ndarray
    source_timestamp_ns: int
    receive_monotonic_ns: int


def _normalize_base_quat_wxyz(value: Any) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"base_quat has dim {quat.size}, expected 4")
    if not np.all(np.isfinite(quat)):
        raise ValueError("base_quat contains NaN/Inf")
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise ValueError("base_quat has near-zero norm")
    normalized = np.asarray(quat / norm, dtype=np.float32)
    normalized.setflags(write=False)
    return normalized


class BaseQuatSubscriber:
    """Non-blocking latest-orientation subscriber owned by one thread."""

    def __init__(self, *, state_action_endpoint: str):
        import zmq

        self._owner_thread_id = threading.get_ident()
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"robot_state_action")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(state_action_endpoint)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        self.latest_snapshot: BaseQuatSnapshot | None = None

    def _require_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("BaseQuatSubscriber socket must be used by its owning thread")

    def poll(self, timeout_ms: int = 0) -> BaseQuatSnapshot | None:
        """Return a newly received sample, or ``None`` when no packet arrived."""

        self._require_owner_thread()
        events = dict(self._poller.poll(timeout_ms))
        if self._socket not in events:
            return None
        message = self._socket.recv()
        receive_monotonic_ns = time.monotonic_ns()
        decoded = decode_topic_message(message, "robot_state_action")
        sample = make_state_action_sample(decoded.payload)
        if "base_quat" not in sample.obs:
            raise ValueError("robot_state_action missing base_quat")
        if (
            self.latest_snapshot is not None
            and int(sample.timestamp_ns) <= self.latest_snapshot.source_timestamp_ns
        ):
            # A duplicate/reordered packet must not refresh the local watchdog.
            return None
        snapshot = BaseQuatSnapshot(
            base_quat_wxyz=_normalize_base_quat_wxyz(sample.obs["base_quat"]),
            source_timestamp_ns=int(sample.timestamp_ns),
            receive_monotonic_ns=receive_monotonic_ns,
        )
        self.latest_snapshot = snapshot
        return snapshot

    def close(self) -> None:
        self._require_owner_thread()
        self._socket.close(0)


def hand_feedback_has_valid_actual(feedback: dict[str, Any]) -> bool:
    return bool(feedback.get("left_actual_position_valid", False)) and bool(
        feedback.get("right_actual_position_valid", False)
    )


class RobotStateSubscriber:
    def __init__(self, *, state_action_endpoint: str, hand_status_endpoint: str = ""):
        import zmq

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._poller = zmq.Poller()
        self._state_socket = self._make_sub(state_action_endpoint, "robot_state_action")
        self._hand_socket = self._make_sub(hand_status_endpoint, "hand_status") if hand_status_endpoint else None
        self._poller.register(self._state_socket, zmq.POLLIN)
        if self._hand_socket is not None:
            self._poller.register(self._hand_socket, zmq.POLLIN)
        self.latest_obs: dict[str, Any] | None = None
        self.latest_state_receive_monotonic_ns = -1
        self.latest_state_payload: dict[str, Any] = {}
        self.latest_hand_feedback: dict[str, Any] | None = None
        self.latest_hand_receive_monotonic_ns = -1

    def _make_sub(self, endpoint: str, topic: str):
        socket = self._context.socket(self._zmq.SUB)
        socket.setsockopt(self._zmq.SUBSCRIBE, topic.encode("utf-8"))
        socket.setsockopt(self._zmq.CONFLATE, 1)
        socket.connect(endpoint)
        return socket

    def poll(self, timeout_ms: int = 0) -> LatestRobotInputs | None:
        events = dict(self._poller.poll(timeout_ms))
        if self._state_socket in events:
            decoded = decode_topic_message(self._state_socket.recv(), "robot_state_action")
            sample = make_state_action_sample(decoded.payload)
            self.latest_obs = sample.obs
            self.latest_state_receive_monotonic_ns = time.monotonic_ns()
            self.latest_state_payload = sample.payload
        if self._hand_socket is not None and self._hand_socket in events:
            decoded = decode_topic_message(self._hand_socket.recv(), "hand_status")
            sample = make_hand_status_sample(decoded.payload)
            self.latest_hand_feedback = sample.feedback
            self.latest_hand_receive_monotonic_ns = time.monotonic_ns()
        if self.latest_obs is None:
            return None
        return LatestRobotInputs(
            obs=self.latest_obs,
            hand_feedback=self.latest_hand_feedback or {},
            state_receive_monotonic_ns=self.latest_state_receive_monotonic_ns,
            state_payload=self.latest_state_payload,
            hand_receive_monotonic_ns=self.latest_hand_receive_monotonic_ns,
        )

    def wait_for_first(self, *, timeout_s: float = 5.0, require_hand_actual: bool = False) -> LatestRobotInputs:
        deadline = time.monotonic() + timeout_s
        latest_seen: LatestRobotInputs | None = None
        while time.monotonic() < deadline:
            latest = self.poll(timeout_ms=100)
            if latest is None:
                continue
            latest_seen = latest
            if not require_hand_actual or hand_feedback_has_valid_actual(latest.hand_feedback):
                return latest
        if require_hand_actual:
            if latest_seen is None:
                raise TimeoutError("timed out waiting for robot_state_action with valid hand actual position")
            raise TimeoutError(
                "timed out waiting for valid hand actual position "
                f"(left_valid={bool(latest_seen.hand_feedback.get('left_actual_position_valid', False))} "
                f"right_valid={bool(latest_seen.hand_feedback.get('right_actual_position_valid', False))})"
            )
        raise TimeoutError("timed out waiting for robot_state_action")

    def close(self) -> None:
        self._state_socket.close(0)
        if self._hand_socket is not None:
            self._hand_socket.close(0)


@dataclass(frozen=True)
class _ReceivedState:
    receive_monotonic_ns: int
    obs: dict[str, Any]
    payload: dict[str, Any]


@dataclass(frozen=True)
class _ReceivedHands:
    receive_monotonic_ns: int
    feedback: dict[str, Any]


class AlignedRobotStateSubscriber:
    """Continuously receive robot inputs and join them on this PC's clock."""

    def __init__(
        self,
        *,
        state_action_endpoint: str,
        hand_status_endpoint: str = "",
        history_size: int = 128,
    ):
        if history_size <= 0:
            raise ValueError("history_size must be > 0")
        self._state_action_endpoint = state_action_endpoint
        self._hand_status_endpoint = hand_status_endpoint
        self._state_history: deque[_ReceivedState] = deque(maxlen=history_size)
        self._hand_history: deque[_ReceivedHands] = deque(maxlen=history_size)
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=5.0):
            self._stop_event.set()
            self._thread.join(timeout=1.0)
            raise TimeoutError("timed out starting aligned robot-state subscriber")
        self._raise_if_failed()

    @staticmethod
    def _make_sub(context: Any, zmq_module: Any, endpoint: str, topic: str) -> Any:
        socket = context.socket(zmq_module.SUB)
        socket.setsockopt(zmq_module.SUBSCRIBE, topic.encode("utf-8"))
        socket.setsockopt(zmq_module.CONFLATE, 1)
        socket.connect(endpoint)
        return socket

    def _receive_loop(self) -> None:
        state_socket = None
        hand_socket = None
        try:
            import zmq

            context = zmq.Context.instance()
            state_socket = self._make_sub(
                context,
                zmq,
                self._state_action_endpoint,
                "robot_state_action",
            )
            hand_socket = (
                self._make_sub(context, zmq, self._hand_status_endpoint, "hand_status")
                if self._hand_status_endpoint
                else None
            )
            poller = zmq.Poller()
            poller.register(state_socket, zmq.POLLIN)
            if hand_socket is not None:
                poller.register(hand_socket, zmq.POLLIN)
            self._ready_event.set()

            while not self._stop_event.is_set():
                events = dict(poller.poll(50))
                received_state = None
                received_hands = None
                if state_socket in events:
                    message = state_socket.recv()
                    receive_ns = time.monotonic_ns()
                    decoded = decode_topic_message(message, "robot_state_action")
                    sample = make_state_action_sample(decoded.payload)
                    received_state = _ReceivedState(receive_ns, sample.obs, sample.payload)
                if hand_socket is not None and hand_socket in events:
                    message = hand_socket.recv()
                    receive_ns = time.monotonic_ns()
                    decoded = decode_topic_message(message, "hand_status")
                    sample = make_hand_status_sample(decoded.payload)
                    received_hands = _ReceivedHands(receive_ns, sample.feedback)
                if received_state is not None or received_hands is not None:
                    with self._condition:
                        if received_state is not None:
                            self._state_history.append(received_state)
                        if received_hands is not None:
                            self._hand_history.append(received_hands)
                        self._condition.notify_all()
        except BaseException as exc:
            self._error = exc
            self._ready_event.set()
            with self._condition:
                self._condition.notify_all()
        finally:
            if state_socket is not None:
                state_socket.close(0)
            if hand_socket is not None:
                hand_socket.close(0)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("aligned robot-state subscriber failed") from self._error

    @staticmethod
    def _nearest(items: list[Any], target_ns: int) -> Any | None:
        if not items:
            return None
        return min(items, key=lambda item: abs(item.receive_monotonic_ns - target_ns))

    def _select_aligned_locked(
        self,
        *,
        target_monotonic_ns: int,
        max_skew_ns: int,
        require_hand_actual: bool,
    ) -> LatestRobotInputs | None:
        state = self._nearest(list(self._state_history), target_monotonic_ns)
        if state is None or abs(state.receive_monotonic_ns - target_monotonic_ns) > max_skew_ns:
            return None

        hands = None
        if self._hand_status_endpoint:
            candidates = list(self._hand_history)
            if require_hand_actual:
                candidates = [item for item in candidates if hand_feedback_has_valid_actual(item.feedback)]
            hands = self._nearest(candidates, target_monotonic_ns)
            if hands is None or abs(hands.receive_monotonic_ns - target_monotonic_ns) > max_skew_ns:
                return None

        return LatestRobotInputs(
            obs=state.obs,
            hand_feedback={} if hands is None else hands.feedback,
            state_receive_monotonic_ns=state.receive_monotonic_ns,
            state_payload=state.payload,
            hand_receive_monotonic_ns=-1 if hands is None else hands.receive_monotonic_ns,
        )

    def wait_for_aligned(
        self,
        *,
        target_monotonic_ns: int,
        max_skew_ms: float,
        timeout_s: float,
        require_hand_actual: bool = False,
    ) -> LatestRobotInputs:
        if max_skew_ms <= 0.0:
            raise ValueError("max_skew_ms must be > 0")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be > 0")
        max_skew_ns = int(max_skew_ms * 1.0e6)
        timeout_deadline_ns = time.monotonic_ns() + int(timeout_s * 1.0e9)
        alignment_deadline_ns = target_monotonic_ns + max_skew_ns

        with self._condition:
            while True:
                self._raise_if_failed()
                aligned = self._select_aligned_locked(
                    target_monotonic_ns=target_monotonic_ns,
                    max_skew_ns=max_skew_ns,
                    require_hand_actual=require_hand_actual,
                )
                latest_state_ns = self._state_history[-1].receive_monotonic_ns if self._state_history else -1
                latest_hand_ns = self._hand_history[-1].receive_monotonic_ns if self._hand_history else -1
                hand_ready = not self._hand_status_endpoint or latest_hand_ns >= target_monotonic_ns
                if aligned is not None and latest_state_ns >= target_monotonic_ns and hand_ready:
                    return aligned

                now_ns = time.monotonic_ns()
                deadline_ns = min(timeout_deadline_ns, alignment_deadline_ns)
                if now_ns >= deadline_ns:
                    if aligned is not None:
                        return aligned
                    state_skew_ms = (
                        float("inf") if latest_state_ns < 0 else abs(latest_state_ns - target_monotonic_ns) / 1.0e6
                    )
                    hand_skew_ms = (
                        0.0
                        if not self._hand_status_endpoint
                        else (
                            float("inf")
                            if latest_hand_ns < 0
                            else abs(latest_hand_ns - target_monotonic_ns) / 1.0e6
                        )
                    )
                    raise TimeoutError(
                        "no robot observation matched camera receive time "
                        f"within {max_skew_ms:g}ms "
                        f"(state_skew={state_skew_ms:.1f}ms hand_skew={hand_skew_ms:.1f}ms)"
                    )
                self._condition.wait((deadline_ns - now_ns) / 1.0e9)

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=1.0)


def build_state_from_latest(
    latest: LatestRobotInputs,
    *,
    fallback_left_hand: np.ndarray | None = None,
    fallback_right_hand: np.ndarray | None = None,
    state_layout: str = STATE_LAYOUT_QUAT,
) -> np.ndarray:
    obs = latest.obs
    layout = normalize_state_layout(state_layout)
    if layout == STATE_LAYOUT_GRAVITY:
        if obs.get("base_gravity") is None:
            base_orientation = base_quat_to_gravity(obs["base_quat"])
        else:
            base_orientation = np.asarray(obs["base_gravity"], dtype=np.float32).reshape(-1)
    else:
        base_orientation = np.asarray(obs["base_quat"], dtype=np.float32).reshape(-1)
    parts = [
        base_orientation,
        np.asarray(obs["base_ang_vel"], dtype=np.float32).reshape(-1),
        np.asarray(obs["base_accel"], dtype=np.float32).reshape(-1),
        np.asarray(obs["body_q"], dtype=np.float32).reshape(-1),
        np.asarray(obs["body_dq"], dtype=np.float32).reshape(-1),
    ]
    feedback = latest.hand_feedback
    left_valid = bool(feedback.get("left_actual_position_valid", False))
    right_valid = bool(feedback.get("right_actual_position_valid", False))
    left_state_action = obs.get("left_hand_q")
    right_state_action = obs.get("right_hand_q")
    left = feedback.get("left_wuji_qpos_actual") if left_valid else left_state_action
    right = feedback.get("right_wuji_qpos_actual") if right_valid else right_state_action
    left = left if left is not None else fallback_left_hand
    right = right if right is not None else fallback_right_hand
    if left is None:
        left = np.zeros((20,), dtype=np.float32)
    if right is None:
        right = np.zeros((20,), dtype=np.float32)
    parts.append(np.asarray(left, dtype=np.float32).reshape(-1))
    parts.append(np.asarray(right, dtype=np.float32).reshape(-1))
    state = np.concatenate(parts).astype(np.float32, copy=False)
    expected_dim = state_dim_for_layout(layout)
    if state.size != expected_dim:
        raise ValueError(f"live state dim {state.size}, expected {expected_dim}")
    if not np.all(np.isfinite(state)):
        raise ValueError("live state contains NaN/Inf")
    return state
