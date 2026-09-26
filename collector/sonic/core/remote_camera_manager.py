from __future__ import annotations

from collections import deque
import logging
import threading
import time
from typing import Mapping

import cv2
import numpy as np

from .camera_discovery import resolve_camera_selector
from .camera_manager import (
    DEFAULT_CAMERA_HISTORY_CAPACITY,
    DEFAULT_CAMERA_STALE_AFTER_NS,
    CameraHistoryBatch,
    CameraHistoryEntry,
    CausalFrameMatch,
    StreamProfile,
)
from .remote_camera_protocol import (
    COMMAND_GET_FRAME,
    COMMAND_LIST_CAMERAS,
    decode_frame_reply,
    decode_list_cameras_response,
    encode_request,
)
from .schema import CameraFrame, CameraInfo

LOGGER = logging.getLogger(__name__)

try:
    import zmq
except ImportError:
    zmq = None


def _require_zmq():
    if zmq is None:
        raise RuntimeError("pyzmq is required for remote camera capture")
    return zmq


def _request_list_cameras(endpoint: str, timeout_ms: int = 1000):
    zmq_mod = _require_zmq()
    context = zmq_mod.Context.instance()
    socket = context.socket(zmq_mod.REQ)
    try:
        socket.setsockopt(zmq_mod.LINGER, 0)
        socket.setsockopt(zmq_mod.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq_mod.SNDTIMEO, timeout_ms)
        socket.connect(endpoint)
        socket.send(encode_request(COMMAND_LIST_CAMERAS))
        return decode_list_cameras_response(socket.recv())
    finally:
        socket.close(0)


def discover_remote_cameras(endpoint: str, timeout_ms: int = 1000) -> list[CameraInfo]:
    return [profile.to_camera_info() for profile in _request_list_cameras(endpoint, timeout_ms=timeout_ms)]


class RemoteManagedCamera:
    def __init__(
        self,
        camera_info: CameraInfo,
        *,
        endpoint: str,
        width: int,
        height: int,
        fps: int,
        record_depth: bool,
        timeout_ms: int = 1000,
        history_capacity: int = DEFAULT_CAMERA_HISTORY_CAPACITY,
    ):
        if history_capacity <= 0:
            raise ValueError("history_capacity must be positive")
        self.camera_info = camera_info
        self.endpoint = endpoint
        self.record_depth = record_depth
        self.timeout_ms = timeout_ms
        self.requested_profile = StreamProfile(width=width, height=height, fps=fps)
        self.active_profile: StreamProfile | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.latest_frame: CameraFrame | None = None
        self.latest_sequence = 0
        self._frame_history: deque[CameraHistoryEntry] = deque(maxlen=history_capacity)
        self._next_history_index = 0
        self._last_warned_stale = False
        self._last_fresh_time = time.monotonic()

    def _publish_frame(self, sequence: int, frame: CameraFrame) -> bool:
        """Publish one unique source frame. Caller must hold the condition lock."""
        if self.latest_frame is not None and sequence == self.latest_sequence:
            return False
        if frame.sequence is None:
            frame.sequence = sequence
        self.latest_sequence = sequence
        self.latest_frame = frame
        self._frame_history.append(
            CameraHistoryEntry(
                history_index=self._next_history_index,
                sequence=sequence,
                frame=frame,
            )
        )
        self._next_history_index += 1
        self.condition.notify_all()
        return True

    def start(self) -> None:
        self.stop_event.clear()
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self._poll_loop,
            name=f"remote-rs-{self.camera_info.key}",
            daemon=True,
        )
        self.thread.start()

    def _open_socket(self):
        zmq_mod = _require_zmq()
        context = zmq_mod.Context.instance()
        socket = context.socket(zmq_mod.REQ)
        socket.setsockopt(zmq_mod.LINGER, 0)
        socket.setsockopt(zmq_mod.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq_mod.SNDTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        return socket

    def _poll_loop(self) -> None:
        interval_s = 1.0 / max(1, self.requested_profile.fps)
        socket = None
        try:
            while not self.stop_event.is_set():
                start_time = time.monotonic()
                try:
                    if socket is None:
                        socket = self._open_socket()
                    socket.send(
                        encode_request(
                            COMMAND_GET_FRAME,
                            camera_key=self.camera_info.key,
                            include_depth=self.record_depth,
                        )
                    )
                    parts = socket.recv_multipart()
                    receive_time_ns = time.monotonic_ns()
                    meta, rgb_jpeg, depth_payload = decode_frame_reply(parts)
                    with self.condition:
                        self.active_profile = StreamProfile(
                            width=meta.width,
                            height=meta.height,
                            fps=meta.fps,
                        )
                        duplicate = self.latest_frame is not None and meta.sequence == self.latest_sequence
                    if duplicate:
                        continue

                    rgb_bgr = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if rgb_bgr is None:
                        raise RuntimeError(f"camera {self.camera_info.key} returned invalid RGB JPEG")

                    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                    depth = None
                    if depth_payload is not None:
                        depth = (
                            np.frombuffer(depth_payload, dtype=np.uint16).reshape((meta.height, meta.width)).copy()
                        )

                    frame = CameraFrame(
                        key=self.camera_info.key,
                        model=meta.model,
                        serial=meta.serial,
                        rgb=rgb,
                        depth=depth,
                        timestamp_ns=receive_time_ns,
                        encoded_color_jpeg=bytes(rgb_jpeg),
                        source_time_ns=meta.capture_monotonic_ns,
                        source_realtime_ns=meta.capture_realtime_ns,
                        sequence=meta.sequence,
                    )

                    with self.condition:
                        self._last_fresh_time = time.monotonic()
                        self._last_warned_stale = False
                        self._publish_frame(meta.sequence, frame)
                except Exception as exc:
                    if socket is not None:
                        try:
                            socket.close(0)
                        except Exception:
                            pass
                        socket = None
                    if self.stop_event.is_set():
                        break
                    if time.monotonic() - self._last_fresh_time >= 0.5 and not self._last_warned_stale:
                        LOGGER.warning(
                            "remote camera %s stale for more than 500 ms: %s", self.camera_info.key, exc
                        )
                        self._last_warned_stale = True
                    time.sleep(0.1)
                finally:
                    elapsed = time.monotonic() - start_time
                    remaining = interval_s - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            if socket is not None:
                try:
                    socket.close(0)
                except Exception:
                    pass

    def wait_for_new_frame(
        self,
        *,
        after_sequence: int | None,
        timeout_ms: int,
    ) -> tuple[int, CameraFrame] | None:
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        with self.condition:
            while True:
                if self.latest_frame is not None and (
                    after_sequence is None or self.latest_sequence > after_sequence
                ):
                    return self.latest_sequence, self.latest_frame

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(timeout=remaining)

    def get_latest_frame(self) -> CameraFrame | None:
        with self.condition:
            return self.latest_frame

    def get_latest_snapshot(self) -> tuple[int, CameraFrame] | None:
        with self.condition:
            if self.latest_frame is None:
                return None
            return self.latest_sequence, self.latest_frame

    def get_frame_history_batch(
        self,
        after_history_index: int | None = None,
    ) -> CameraHistoryBatch:
        with self.condition:
            entries = tuple(self._frame_history)
            oldest = entries[0].history_index if entries else None
            latest = entries[-1].history_index if entries else None
            overflowed = (
                after_history_index is not None and oldest is not None and after_history_index < oldest - 1
            )
            if after_history_index is not None:
                entries = tuple(entry for entry in entries if entry.history_index > after_history_index)
            return CameraHistoryBatch(
                entries=entries,
                overflowed=overflowed,
                oldest_history_index=oldest,
                latest_history_index=latest,
            )

    def get_causal_frame_match(
        self,
        tick_time_ns: int,
        *,
        previous_sequence: int | None = None,
        stale_after_ns: int = DEFAULT_CAMERA_STALE_AFTER_NS,
    ) -> CausalFrameMatch | None:
        if stale_after_ns < 0:
            raise ValueError("stale_after_ns must be non-negative")
        with self.condition:
            for entry in reversed(self._frame_history):
                if entry.frame.timestamp_ns > tick_time_ns:
                    continue
                age_ns = tick_time_ns - entry.frame.timestamp_ns
                return CausalFrameMatch(
                    sequence=entry.sequence,
                    frame=entry.frame,
                    age_ns=age_ns,
                    reused=previous_sequence == entry.sequence,
                    stale=age_ns > stale_after_ns,
                    offset_ns=-age_ns,
                )
        return None

    def get_nearest_frame_match(
        self,
        tick_time_ns: int,
        *,
        previous_sequence: int | None = None,
        stale_after_ns: int = DEFAULT_CAMERA_STALE_AFTER_NS,
        max_future_ns: int = 0,
    ) -> CausalFrameMatch | None:
        """Select the history frame nearest to a tick within the look-ahead bound."""
        if stale_after_ns < 0:
            raise ValueError("stale_after_ns must be non-negative")
        if max_future_ns < 0:
            raise ValueError("max_future_ns must be non-negative")
        with self.condition:
            candidates = [
                entry for entry in self._frame_history if entry.frame.timestamp_ns <= tick_time_ns + max_future_ns
            ]
            if not candidates:
                return None
            entry = min(
                candidates,
                key=lambda item: (
                    abs(item.frame.timestamp_ns - tick_time_ns),
                    item.frame.timestamp_ns > tick_time_ns,
                    -item.sequence,
                ),
            )
            offset_ns = entry.frame.timestamp_ns - tick_time_ns
            error_ns = abs(offset_ns)
            return CausalFrameMatch(
                sequence=entry.sequence,
                frame=entry.frame,
                age_ns=error_ns,
                reused=previous_sequence == entry.sequence,
                stale=error_ns > stale_after_ns,
                offset_ns=offset_ns,
            )

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None


class RemoteMultiCameraManager:
    def __init__(
        self,
        cameras: list[CameraInfo],
        *,
        endpoint: str,
        width: int,
        height: int,
        fps: int,
        record_depth: bool,
    ):
        self.cameras = list(cameras)
        self.endpoint = endpoint
        self._managed = {
            camera.key: RemoteManagedCamera(
                resolve_camera_selector(self.cameras, camera.key),
                endpoint=endpoint,
                width=width,
                height=height,
                fps=fps,
                record_depth=record_depth,
            )
            for camera in self.cameras
        }

    @property
    def camera_keys(self) -> list[str]:
        return [camera.key for camera in self.cameras]

    def start(self) -> None:
        started: list[RemoteManagedCamera] = []
        try:
            for camera in self.cameras:
                managed = self._managed[camera.key]
                managed.start()
                started.append(managed)
        except Exception:
            for managed in started:
                managed.stop()
            raise

    def stop(self) -> None:
        for managed in self._managed.values():
            managed.stop()

    def get_latest_frames(self) -> dict[str, CameraFrame]:
        frames: dict[str, CameraFrame] = {}
        for key, managed in self._managed.items():
            frame = managed.get_latest_frame()
            if frame is not None:
                frames[key] = frame
        return frames

    def get_latest_frame_snapshots(self) -> dict[str, tuple[int, CameraFrame]]:
        snapshots: dict[str, tuple[int, CameraFrame]] = {}
        for key, managed in self._managed.items():
            snapshot = managed.get_latest_snapshot()
            if snapshot is not None:
                snapshots[key] = snapshot
        return snapshots

    def get_frame_history_batches(
        self,
        after_history_indices: Mapping[str, int] | None = None,
    ) -> dict[str, CameraHistoryBatch]:
        cursors = after_history_indices or {}
        return {key: managed.get_frame_history_batch(cursors.get(key)) for key, managed in self._managed.items()}

    def get_causal_frame_matches(
        self,
        tick_time_ns: int,
        previous_sequences: Mapping[str, int] | None = None,
        *,
        stale_after_ns: int = DEFAULT_CAMERA_STALE_AFTER_NS,
    ) -> dict[str, CausalFrameMatch]:
        previous = previous_sequences or {}
        matches: dict[str, CausalFrameMatch] = {}
        for key, managed in self._managed.items():
            match = managed.get_causal_frame_match(
                tick_time_ns,
                previous_sequence=previous.get(key),
                stale_after_ns=stale_after_ns,
            )
            if match is not None:
                matches[key] = match
        return matches

    def get_nearest_frame_matches(
        self,
        tick_time_ns: int,
        previous_sequences: Mapping[str, int] | None = None,
        *,
        stale_after_ns: int = DEFAULT_CAMERA_STALE_AFTER_NS,
        max_future_ns: int = 0,
    ) -> dict[str, CausalFrameMatch]:
        previous = previous_sequences or {}
        matches: dict[str, CausalFrameMatch] = {}
        for key, managed in self._managed.items():
            match = managed.get_nearest_frame_match(
                tick_time_ns,
                previous_sequence=previous.get(key),
                stale_after_ns=stale_after_ns,
                max_future_ns=max_future_ns,
            )
            if match is not None:
                matches[key] = match
        return matches

    def wait_for_camera_frame(
        self,
        camera_key: str,
        *,
        after_sequence: int | None,
        timeout_ms: int,
    ) -> tuple[int, CameraFrame] | None:
        managed = self._managed.get(camera_key)
        if managed is None:
            raise KeyError(f"unknown camera key: {camera_key}")
        return managed.wait_for_new_frame(after_sequence=after_sequence, timeout_ms=timeout_ms)

    def get_active_profile(self, camera_key: str) -> StreamProfile | None:
        managed = self._managed.get(camera_key)
        if managed is None:
            return None
        return managed.active_profile
