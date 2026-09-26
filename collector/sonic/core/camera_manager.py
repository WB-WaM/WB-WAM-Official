from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Mapping

import numpy as np

from .schema import CameraFrame, CameraInfo

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


LOGGER = logging.getLogger(__name__)
DEFAULT_CAMERA_HISTORY_CAPACITY = 64
DEFAULT_CAMERA_STALE_AFTER_NS = 25_000_000


@dataclass(frozen=True)
class StreamProfile:
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class CameraHistoryEntry:
    history_index: int
    sequence: int
    frame: CameraFrame


@dataclass(frozen=True)
class CameraHistoryBatch:
    entries: tuple[CameraHistoryEntry, ...]
    overflowed: bool
    oldest_history_index: int | None
    latest_history_index: int | None


@dataclass(frozen=True)
class CausalFrameMatch:
    sequence: int
    frame: CameraFrame
    age_ns: int
    reused: bool
    stale: bool
    offset_ns: int = 0


class ManagedCamera:
    def __init__(
        self,
        camera_info: CameraInfo,
        width: int,
        height: int,
        fps: int,
        record_depth: bool,
        history_capacity: int = DEFAULT_CAMERA_HISTORY_CAPACITY,
    ):
        if history_capacity <= 0:
            raise ValueError("history_capacity must be positive")
        self.camera_info = camera_info
        self.record_depth = record_depth
        self.requested_profile = StreamProfile(width=width, height=height, fps=fps)
        self.active_profile: StreamProfile | None = None
        self.pipeline = None
        self.align = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.latest_frame: CameraFrame | None = None
        self.latest_sequence = 0
        self._frame_history: deque[CameraHistoryEntry] = deque(maxlen=history_capacity)
        self._next_history_index = 0

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
        if rs is None:
            raise RuntimeError("pyrealsense2 is required for camera capture")

        profile = self.requested_profile
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.camera_info.serial)
        if self.record_depth:
            config.enable_stream(rs.stream.depth, profile.width, profile.height, rs.format.z16, profile.fps)
        config.enable_stream(rs.stream.color, profile.width, profile.height, rs.format.rgb8, profile.fps)
        align = rs.align(rs.stream.color) if self.record_depth else None
        try:
            pipeline.start(config)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise RuntimeError(
                "failed to start camera "
                f"{self.camera_info.key} ({self.camera_info.model}, {self.camera_info.serial}) "
                f"at {profile.width}x{profile.height}@{profile.fps}: {exc}"
            ) from exc

        self.pipeline = pipeline
        self.align = align
        self.active_profile = profile
        LOGGER.info(
            "camera %s started at %dx%d@%d",
            self.camera_info.key,
            profile.width,
            profile.height,
            profile.fps,
        )

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, name=f"rs-{self.camera_info.key}", daemon=True)
        self.thread.start()

    def _capture_loop(self) -> None:
        assert self.pipeline is not None

        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=250)
                if self.align is not None:
                    frames = self.align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if self.record_depth else None
                if not color_frame or (self.record_depth and not depth_frame):
                    continue

                depth = np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None
                rgb = np.asanyarray(color_frame.get_data()).copy()
                timestamp_ns = time.monotonic_ns()
                frame = CameraFrame(
                    key=self.camera_info.key,
                    model=self.camera_info.model,
                    serial=self.camera_info.serial,
                    rgb=rgb,
                    depth=depth,
                    timestamp_ns=timestamp_ns,
                )
                with self.condition:
                    self._publish_frame(self.latest_sequence + 1, frame)
            except RuntimeError:
                continue
            except Exception as exc:
                LOGGER.warning("camera %s capture loop error: %s", self.camera_info.key, exc)
                time.sleep(0.1)

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
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.align = None


class MultiCameraManager:
    def __init__(self, cameras: list[CameraInfo], width: int, height: int, fps: int, record_depth: bool):
        self.cameras = list(cameras)
        self._managed = {
            camera.key: ManagedCamera(
                camera,
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
        started: list[ManagedCamera] = []
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
