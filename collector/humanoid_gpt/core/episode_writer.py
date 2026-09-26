"""Episode writer for causal HGPT snapshots and RealSense frames."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

from collector.sonic.core.async_writers import AsyncImageWriter, DepthWriterProcess
from collector.sonic.core.camera_manager import CausalFrameMatch
from collector.sonic.core.episode_writer import (
    compress_deferred_episode_depth_outputs,
)
from collector.sonic.core.schema import CameraInfo

LOGGER = logging.getLogger(__name__)


class EpisodeWriter:
    def __init__(
        self,
        output_root: Path,
        task_name: str,
        *,
        capture_fps: int,
        record_depth: bool,
        defer_depth_compression: bool,
        downsample_method: str = "causal_latest",
        interpolation_delay_ms: float = 0.0,
    ) -> None:
        self.task_root = output_root / task_name
        self._preflight_task_root()
        self.capture_fps = int(capture_fps)
        self.record_depth = bool(record_depth)
        self.defer_depth_compression = bool(defer_depth_compression)
        self.downsample_method = str(downsample_method)
        self.interpolation_delay_ms = float(interpolation_delay_ms)
        self.episode_dir: Path | None = None
        self._frames: list[dict[str, Any]] = []
        self._images: AsyncImageWriter | None = None
        self._depth: DepthWriterProcess | None = None
        self._owner = uuid.uuid4().hex

    def _preflight_task_root(self) -> None:
        token = uuid.uuid4().hex
        pending = self.task_root / f".collector_preflight_{token}.tmp"
        committed = self.task_root / f".collector_preflight_{token}.ready"
        payload = f"hgpt-collector-preflight:{token}\n".encode("ascii")
        try:
            self.task_root.mkdir(parents=True, exist_ok=True)
            with pending.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            pending.replace(committed)
            if committed.read_bytes() != payload:
                raise OSError(f"read-back mismatch for {committed}")
            committed.unlink()
            if committed.exists():
                raise OSError(f"delete check failed for {committed}")
        except Exception as exc:
            LOGGER.warning("output preflight failed for %s: %s", self.task_root, exc)
            raise RuntimeError(f"output preflight failed for {self.task_root}: {exc}") from exc
        finally:
            for probe in (pending, committed):
                try:
                    probe.unlink(missing_ok=True)
                except OSError:
                    pass

    def write_metadata(
        self,
        *,
        cameras: list[CameraInfo],
        primary_camera: str,
        camera_fps: int,
        camera_backend: str,
        camera_endpoint: str,
        pose_endpoint: str,
        state_action_endpoint: str,
        hand_status_endpoint: str,
        notes: str,
        downsample_method: str = "causal_latest",
        interpolation_delay_ms: float = 0.0,
    ) -> None:
        interpolated = downsample_method == "interpolate"
        payload = {
            "schema": "humanoid_gpt.v1",
            "task_name": self.task_root.name,
            "robot_name": "unitree_g1",
            "capture_fps": self.capture_fps,
            "capture_mode": downsample_method,
            "downsample_method": downsample_method,
            "interpolation_delay_ms": float(interpolation_delay_ms),
            "clock_source": ("collector_strict_20hz" if interpolated else "collector_monotonic_timer"),
            "camera_alignment": (
                "nearest_pc_receive_to_tick" if interpolated else "latest_pc_receive_at_or_before_tick"
            ),
            "body_dq_source": ("finite_difference_20hz_body_q" if interpolated else "robot_telemetry"),
            "collector_ema": "none",
            "camera_fps": camera_fps,
            "camera_backend": camera_backend,
            "camera_endpoint": camera_endpoint,
            "primary_camera": primary_camera,
            "connected_cameras": [camera.to_record() for camera in cameras],
            "record_depth": self.record_depth,
            "defer_depth_compression": self.defer_depth_compression,
            "pose_endpoint": pose_endpoint,
            "state_action_endpoint": state_action_endpoint,
            "hand_status_endpoint": hand_status_endpoint,
            "notes": notes,
        }
        path = self.task_root / "metadata.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(path)

    def next_episode_dir(self) -> Path:
        indices = []
        for path in self.task_root.glob("episode_*"):
            try:
                indices.append(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return self.task_root / f"episode_{max(indices, default=-1) + 1:06d}"

    def start(self, cameras: list[CameraInfo]) -> Path:
        if self.episode_dir is not None:
            raise RuntimeError("episode already active")
        episode = self.next_episode_dir()
        episode.mkdir(parents=True, exist_ok=False)
        (episode / ".collector_owner").write_text(self._owner, encoding="ascii")
        for camera in cameras:
            (episode / f"color_{camera.key}").mkdir()
            if self.record_depth:
                (episode / f"depth_{camera.key}").mkdir()
        self.episode_dir = episode
        self._frames = []
        self._images = AsyncImageWriter()
        self._depth = DepthWriterProcess() if self.record_depth else None
        return episode

    def append(
        self,
        *,
        frame_index: int,
        tick_ns: int,
        primary_camera: str,
        camera_matches: dict[str, CausalFrameMatch],
        snapshot: dict[str, Any],
    ) -> None:
        if self.episode_dir is None or self._images is None:
            raise RuntimeError("episode not active")
        cameras: dict[str, dict[str, Any]] = {}
        for key, match in camera_matches.items():
            frame = match.frame
            color_rel = f"color_{key}/frame_{frame_index:06d}.jpg"
            self._images.write(
                self.episode_dir / color_rel,
                frame.rgb,
                encoded_jpeg=frame.encoded_color_jpeg,
            )
            record: dict[str, Any] = {
                "color": color_rel,
                "time_ns": int(frame.timestamp_ns),
                "model": frame.model,
                "serial": frame.serial,
                "sequence": int(match.sequence),
                "camera_age_ns": int(match.age_ns),
                "camera_alignment_error_ns": int(match.age_ns),
                "camera_offset_ns": int(match.offset_ns),
                "reused": bool(match.reused),
                "stale": bool(match.stale),
                "transport": (
                    "remote_realsense"
                    if frame.encoded_color_jpeg is not None or frame.source_time_ns is not None
                    else "local_realsense"
                ),
            }
            if frame.source_time_ns is not None:
                record["source_time_ns"] = int(frame.source_time_ns)
            if frame.source_realtime_ns is not None:
                record["source_realtime_ns"] = int(frame.source_realtime_ns)
            if self.record_depth:
                if frame.depth is None or self._depth is None:
                    raise RuntimeError(f"camera {key} has no depth frame")
                depth_rel = f"depth_{key}/frame_{frame_index:06d}.npy"
                self._depth.write(self.episode_dir / depth_rel, frame.depth)
                record["depth"] = depth_rel
            cameras[key] = record

        row = {
            "frame_index": frame_index,
            "time_ns": frame_index * int(round(1e9 / self.capture_fps)),
            "tick_time_ns": int(tick_ns),
            "primary_camera": primary_camera,
            "cameras": cameras,
            **snapshot,
        }
        self._frames.append(row)

    def _recompute_interpolated_body_dq(self) -> None:
        if not (self.capture_fps == 20 and self.downsample_method == "interpolate"):
            return
        body_q = []
        for index, frame in enumerate(self._frames):
            obs = frame.get("obs")
            if not isinstance(obs, dict):
                LOGGER.warning(
                    "skipping 20 Hz body_dq derivation: frame %d has invalid obs; keeping recorded body_dq",
                    index,
                )
                return
            try:
                values = np.asarray(obs.get("body_q"), dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                values = np.empty(0, dtype=np.float64)
            if values.shape != (29,) or not np.all(np.isfinite(values)):
                LOGGER.warning(
                    "skipping 20 Hz body_dq derivation: frame %d has invalid body_q; keeping recorded body_dq",
                    index,
                )
                return
            body_q.append(values)
        positions = np.asarray(body_q, dtype=np.float64)
        edge_order = 2 if len(positions) >= 3 else 1
        velocities = np.gradient(
            positions,
            1.0 / self.capture_fps,
            axis=0,
            edge_order=edge_order,
        )
        if not np.all(np.isfinite(velocities)):
            LOGGER.warning("skipping 20 Hz body_dq derivation: result is non-finite; keeping recorded body_dq")
            return
        for frame, velocity in zip(self._frames, velocities, strict=True):
            frame["obs"]["body_dq"] = velocity.astype(np.float32).tolist()

    @staticmethod
    def _fixed_motion_vector(value: Any, size: int) -> tuple[np.ndarray, bool]:
        try:
            vector = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return np.zeros(size, dtype=np.float32), False
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            return np.zeros(size, dtype=np.float32), False
        return vector, True

    def _write_motion(self) -> None:
        assert self.episode_dir is not None
        qpos_rows = []
        left_rows = []
        right_rows = []
        cmd_vel_rows = []
        body_valid = []
        left_valid = []
        right_valid = []
        modes = []
        placeholder_fields: dict[str, int] = {}
        for frame in self._frames:
            raw_derived = frame.get("human_derived")
            derived = raw_derived if isinstance(raw_derived, dict) else {}
            qpos, qpos_ok = self._fixed_motion_vector(derived.get("g1_qpos"), 36)
            left, left_ok = self._fixed_motion_vector(derived.get("left_wuji_qpos"), 20)
            right, right_ok = self._fixed_motion_vector(derived.get("right_wuji_qpos"), 20)
            cmd_vel, cmd_vel_ok = self._fixed_motion_vector(derived.get("cmd_vel"), 3)
            for name, valid in (
                ("g1_qpos", qpos_ok),
                ("left_wuji_qpos", left_ok),
                ("right_wuji_qpos", right_ok),
                ("cmd_vel", cmd_vel_ok),
            ):
                if not valid:
                    placeholder_fields[name] = placeholder_fields.get(name, 0) + 1
            try:
                mode = int(derived.get("mode", 0))
                if not np.iinfo(np.int32).min <= mode <= np.iinfo(np.int32).max:
                    raise OverflowError
            except (TypeError, ValueError, OverflowError):
                mode = 0
                placeholder_fields["mode"] = placeholder_fields.get("mode", 0) + 1
            qpos_rows.append(qpos)
            left_rows.append(left)
            right_rows.append(right)
            cmd_vel_rows.append(cmd_vel)
            body_valid.append(qpos_ok and derived.get("body_valid") is True)
            left_valid.append(left_ok and derived.get("left_wuji_qpos_valid") is True)
            right_valid.append(right_ok and derived.get("right_wuji_qpos_valid") is True)
            modes.append(mode)
        if placeholder_fields:
            LOGGER.warning("motion.npz used zero placeholders for invalid fields: %s", placeholder_fields)
        temp = self.episode_dir / "motion.tmp.npz"
        np.savez_compressed(
            temp,
            qpos=np.stack(qpos_rows),
            frequency=np.array(self.capture_fps, dtype=np.float32),
            downsample_method=np.array(self.downsample_method),
            interpolation_delay_ms=np.array(self.interpolation_delay_ms, dtype=np.float32),
            left_wuji_qpos=np.stack(left_rows),
            right_wuji_qpos=np.stack(right_rows),
            left_wuji_qpos_valid=np.asarray(left_valid, dtype=bool),
            right_wuji_qpos_valid=np.asarray(right_valid, dtype=bool),
            body_valid=np.asarray(body_valid, dtype=bool),
            mode=np.asarray(modes, dtype=np.int32),
            cmd_vel=np.stack(cmd_vel_rows),
        )
        temp.replace(self.episode_dir / "motion.npz")

    def _close_writers(self, *, cancel: bool) -> None:
        failures: list[tuple[str, BaseException]] = []
        for name, attribute in (("image", "_images"), ("depth", "_depth")):
            writer = getattr(self, attribute)
            if writer is None:
                continue
            setattr(self, attribute, None)
            try:
                writer.close(cancel=cancel)
            except BaseException as exc:
                failures.append((name, exc))
        if failures:
            details = "; ".join(f"{name}: {exc}" for name, exc in failures)
            raise RuntimeError(f"failed to close episode writers: {details}") from failures[0][1]

    def _reset(self) -> None:
        self._images = None
        self._depth = None
        self.episode_dir = None
        self._frames = []

    def commit(self) -> Path:
        if self.episode_dir is None or len(self._frames) < 2:
            raise RuntimeError("cannot commit an episode with fewer than two frames")
        if self._images is not None:
            self._images.flush()
        if self._depth is not None:
            self._depth.flush()
        self._recompute_interpolated_body_dq()
        data_path = self.episode_dir / "data.json"
        temp = self.episode_dir / "data.json.tmp"
        temp.write_text(json.dumps(self._frames, indent=2), encoding="utf-8")
        temp.replace(data_path)
        self._write_motion()
        if not self.defer_depth_compression and self.record_depth:
            compress_deferred_episode_depth_outputs(self.episode_dir)
        owner_path = self.episode_dir / ".collector_owner"
        result = data_path
        self._close_writers(cancel=False)
        owner_path.unlink(missing_ok=True)
        self._reset()
        return result

    def discard(self) -> None:
        episode = self.episode_dir
        marker = episode / ".collector_owner" if episode is not None else None
        try:
            owned = marker is not None and marker.exists() and marker.read_text(encoding="ascii") == self._owner
        except OSError as exc:
            LOGGER.warning("cannot verify episode owner marker %s: %s", marker, exc)
            owned = False
        failure: BaseException | None = None
        try:
            self.close(cancel=True)
        except BaseException as exc:
            failure = exc
        if owned and episode is not None:
            try:
                shutil.rmtree(episode)
            except BaseException as exc:
                failure = failure or exc
        if failure is not None:
            raise failure

    def close(self, *, cancel: bool = False) -> None:
        try:
            self._close_writers(cancel=cancel)
        finally:
            self._reset()
