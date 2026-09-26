from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import shutil
import uuid

from .async_writers import AsyncImageWriter, AsyncJSONLWriter, DepthWriterProcess
from .camera_manager import CausalFrameMatch
from .constants import DEFAULT_CAPTURE_FPS, normalize_capture_fps
from .depth_io import compress_raw_depth, load_depth, write_compressed_depth
from .schema import (
    CameraFrame,
    CameraInfo,
    DrainedSamples,
    HandStatusSample,
    PoseControlSample,
    PoseRawSample,
    RobotStateActionSample,
    to_python,
)
from .session_merger import SessionMerger

LOGGER = logging.getLogger(__name__)
OWNER_MARKER_NAME = ".collector_owner"
DEPTH_COMPRESSION_PROGRESS_INTERVAL = 100
MAX_DEPTH_COMPRESSION_WORKERS = 4


def _compressed_depth_path(raw_depth_path: Path) -> Path:
    return raw_depth_path.with_suffix(raw_depth_path.suffix + ".lzma")


def _compress_depth_file(raw_depth_path: Path) -> None:
    compress_raw_depth(raw_depth_path, _compressed_depth_path(raw_depth_path))


def _write_compressed_depth_file(raw_depth_path: Path) -> Path:
    compressed_depth_path = _compressed_depth_path(raw_depth_path)
    write_compressed_depth(raw_depth_path, compressed_depth_path)
    return compressed_depth_path


def _episode_raw_depth_paths(episode_dir: Path) -> list[Path]:
    return sorted(episode_dir.glob("depth_*/frame_*.npy"))


def _load_episode_data(data_path: Path) -> tuple[object, list[dict]]:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, payload
    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        return payload, payload["frames"]
    raise ValueError(f"{data_path} must contain a frame list or a dict with frames")


def _episode_depth_references(data_path: Path) -> set[str]:
    _, frames = _load_episode_data(data_path)
    references = set()
    for frame in frames:
        cameras = frame.get("cameras")
        if not isinstance(cameras, dict):
            continue
        for camera_record in cameras.values():
            if not isinstance(camera_record, dict):
                continue
            depth_rel = camera_record.get("depth")
            if isinstance(depth_rel, str):
                references.add(depth_rel)
    return references


def _valid_compressed_depth(compressed_depth_path: Path) -> bool:
    if not compressed_depth_path.is_file():
        return False
    try:
        load_depth(compressed_depth_path)
    except Exception as exc:
        LOGGER.warning(
            "keeping raw depth because compressed output is invalid: %s (%s)",
            compressed_depth_path,
            exc,
        )
        return False
    return True


def _update_episode_depth_paths(data_path: Path, rel_path_map: dict[str, str]) -> int:
    payload, frames = _load_episode_data(data_path)
    updated = 0
    for frame in frames:
        cameras = frame.get("cameras")
        if not isinstance(cameras, dict):
            continue
        for camera_record in cameras.values():
            if not isinstance(camera_record, dict):
                continue
            depth_rel = camera_record.get("depth")
            if depth_rel in rel_path_map:
                camera_record["depth"] = rel_path_map[depth_rel]
                updated += 1

    if updated == 0:
        return 0

    temp_path = data_path.with_name(data_path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    temp_path.replace(data_path)
    return updated


def _compress_depth_paths(
    raw_depth_paths: list[Path],
    *,
    delete_raw: bool,
) -> None:
    total = len(raw_depth_paths)
    if total == 0:
        return

    max_workers = min(total, os.cpu_count() or 1, MAX_DEPTH_COMPRESSION_WORKERS)
    LOGGER.info("compressing %d depth frame(s) with %d worker(s)", total, max_workers)

    worker = _compress_depth_file if delete_raw else _write_compressed_depth_file
    if max_workers <= 1:
        for index, raw_depth_path in enumerate(raw_depth_paths, start=1):
            worker(raw_depth_path)
            if index % DEPTH_COMPRESSION_PROGRESS_INTERVAL == 0 or index == total:
                LOGGER.info("compressed %d/%d depth frame(s)", index, total)
        return

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="depth-compress") as executor:
        for index, _ in enumerate(executor.map(worker, raw_depth_paths), start=1):
            if index % DEPTH_COMPRESSION_PROGRESS_INTERVAL == 0 or index == total:
                LOGGER.info("compressed %d/%d depth frame(s)", index, total)


def compress_deferred_episode_depth_outputs(episode_dir: Path) -> int:
    raw_depth_paths = _episode_raw_depth_paths(episode_dir)
    if not raw_depth_paths:
        return 0

    data_path = episode_dir / "data.json"
    if not data_path.exists():
        LOGGER.warning("skipping deferred depth compression without data.json: %s", episode_dir)
        return 0

    depth_references = _episode_depth_references(data_path)
    rel_path_map = {}
    raw_depth_paths_to_compress = []
    completed_raw_depth_paths = []
    orphan_raw_depth_paths = []
    for raw_path in raw_depth_paths:
        raw_rel = raw_path.relative_to(episode_dir).as_posix()
        compressed_path = _compressed_depth_path(raw_path)
        compressed_rel = compressed_path.relative_to(episode_dir).as_posix()
        if raw_rel in depth_references:
            rel_path_map[raw_rel] = compressed_rel
            raw_depth_paths_to_compress.append(raw_path)
        elif compressed_rel in depth_references and _valid_compressed_depth(compressed_path):
            completed_raw_depth_paths.append(raw_path)
        else:
            orphan_raw_depth_paths.append(raw_path)

    if orphan_raw_depth_paths:
        LOGGER.warning(
            "%d raw depth file(s) are not backed by a valid data.json reference; keeping them",
            len(orphan_raw_depth_paths),
        )

    converted_raw_depth_paths = []
    if raw_depth_paths_to_compress:
        _compress_depth_paths(raw_depth_paths_to_compress, delete_raw=False)
        updated = _update_episode_depth_paths(data_path, rel_path_map)
        if updated == 0:
            LOGGER.warning("no depth paths updated in %s; keeping raw depth files", data_path)
        else:
            converted_raw_depth_paths = raw_depth_paths_to_compress

    cleanup_raw_depth_paths = completed_raw_depth_paths + converted_raw_depth_paths
    for raw_path in cleanup_raw_depth_paths:
        raw_path.unlink(missing_ok=True)
    return len(cleanup_raw_depth_paths)


def _iter_tqdm(items: list[Path]):
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(items, unit="episode")


def compress_deferred_depth_outputs(task_root: Path, *, use_tqdm: bool = True) -> int:
    episode_dirs = [
        episode_dir
        for episode_dir in sorted(task_root.glob("episode_*"))
        if episode_dir.is_dir() and _episode_raw_depth_paths(episode_dir)
    ]
    if not episode_dirs:
        LOGGER.info("no deferred depth outputs to compress in %s", task_root)
        return 0

    LOGGER.info("compressing deferred depth outputs for %d episode(s)", len(episode_dirs))
    compressed_total = 0
    progress = _iter_tqdm(episode_dirs) if use_tqdm else None
    if progress is None:
        total = len(episode_dirs)
        for index, episode_dir in enumerate(episode_dirs, start=1):
            raw_count = len(_episode_raw_depth_paths(episode_dir))
            LOGGER.info("[%d/%d] %s compressing %d depth frame(s)", index, total, episode_dir.name, raw_count)
            compressed_total += compress_deferred_episode_depth_outputs(episode_dir)
            LOGGER.info("[%d/%d] %s done", index, total, episode_dir.name)
        LOGGER.info("deferred depth compression complete: %d depth frame(s)", compressed_total)
        return compressed_total

    with progress:
        for episode_dir in progress:
            raw_count = len(_episode_raw_depth_paths(episode_dir))
            progress.set_description(episode_dir.name)
            progress.set_postfix(depth_frames=raw_count)
            compressed_total += compress_deferred_episode_depth_outputs(episode_dir)
    LOGGER.info("deferred depth compression complete: %d depth frame(s)", compressed_total)
    return compressed_total


class EpisodeWriter:
    def __init__(
        self,
        keep_temp_logs: bool = False,
        record_depth: bool = True,
        defer_depth_compression: bool = False,
        capture_fps: int = DEFAULT_CAPTURE_FPS,
        sample_mode: str = "nearest_hold",
    ):
        self.capture_fps = normalize_capture_fps(capture_fps)
        self.sample_mode = sample_mode
        self.keep_temp_logs = keep_temp_logs
        self.record_depth = record_depth
        self.defer_depth_compression = defer_depth_compression
        self.episode_dir: Path | None = None
        self.temp_dir: Path | None = None
        self.camera_infos: dict[str, CameraInfo] = {}
        self.primary_camera = ""
        self._jsonl_writer: AsyncJSONLWriter | None = None
        self._image_writer: AsyncImageWriter | None = None
        self._depth_writer: DepthWriterProcess | None = None
        self._owner_token = uuid.uuid4().hex
        self._owned_episode_dir: Path | None = None

    def start_episode(
        self,
        episode_dir: Path,
        camera_infos: list[CameraInfo],
        *,
        primary_camera: str,
    ) -> None:
        self.episode_dir = episode_dir
        self.temp_dir = episode_dir / "_tmp"
        self.camera_infos = {camera.key: camera for camera in camera_infos}
        self.primary_camera = primary_camera
        episode_dir.mkdir(parents=True, exist_ok=False)
        self._owned_episode_dir = episode_dir.resolve()
        self._write_owner_marker()
        self.temp_dir.mkdir(parents=True, exist_ok=False)
        for camera in camera_infos:
            (episode_dir / f"color_{camera.key}").mkdir(parents=False, exist_ok=False)
            if self.record_depth:
                (episode_dir / f"depth_{camera.key}").mkdir(parents=False, exist_ok=False)
        self._jsonl_writer = AsyncJSONLWriter()
        self._image_writer = AsyncImageWriter()
        self._depth_writer = DepthWriterProcess() if self.record_depth else None

    def append_tick(
        self,
        *,
        frame_index: int,
        tick_time_ns: int,
        primary_camera: str,
        camera_snapshots: dict[str, tuple[int, CameraFrame] | CausalFrameMatch],
        previous_sequences: dict[str, int],
        latest_samples: dict | None = None,
    ) -> None:
        if self.episode_dir is None:
            raise RuntimeError("episode not started")

        camera_records: dict[str, dict] = {}
        for key, snapshot in camera_snapshots.items():
            if isinstance(snapshot, CausalFrameMatch):
                causal_match = snapshot
                sequence, camera_frame = snapshot.sequence, snapshot.frame
            else:
                causal_match = None
                sequence, camera_frame = snapshot
            color_rel = f"color_{key}/frame_{frame_index:06d}.jpg"
            color_path = self.episode_dir / color_rel

            assert self._image_writer is not None
            self._image_writer.write(
                color_path,
                camera_frame.rgb,
                encoded_jpeg=camera_frame.encoded_color_jpeg,
            )

            camera_records[key] = {
                "color": color_rel,
                "time_ns": camera_frame.timestamp_ns,
                "model": camera_frame.model,
                "serial": camera_frame.serial,
                "reused": (
                    causal_match.reused if causal_match is not None else previous_sequences.get(key) == sequence
                ),
                "sequence": sequence,
                "transport": (
                    "remote_realsense"
                    if camera_frame.encoded_color_jpeg is not None or camera_frame.source_time_ns is not None
                    else "local_realsense"
                ),
            }
            if causal_match is not None:
                camera_records[key]["camera_age_ns"] = causal_match.age_ns
                camera_records[key]["stale"] = causal_match.stale
            if camera_frame.source_time_ns is not None:
                camera_records[key]["source_time_ns"] = camera_frame.source_time_ns
            if camera_frame.source_realtime_ns is not None:
                camera_records[key]["source_realtime_ns"] = camera_frame.source_realtime_ns
            if self.record_depth:
                if camera_frame.depth is None:
                    raise RuntimeError(f"camera {key} depth frame missing while record_depth is enabled")
                depth_rel = (
                    f"depth_{key}/frame_{frame_index:06d}.npy"
                    if self.defer_depth_compression
                    else f"depth_{key}/frame_{frame_index:06d}.npy.lzma"
                )
                depth_path = self.episode_dir / depth_rel
                raw_depth_path = depth_path if self.defer_depth_compression else self._raw_depth_path(depth_path)

                assert self._depth_writer is not None
                self._depth_writer.write(raw_depth_path, camera_frame.depth)
                camera_records[key]["depth"] = depth_rel

        tick_record = {
            "frame_index": frame_index,
            "tick_time_ns": tick_time_ns,
            "primary_camera": primary_camera,
            "cameras": camera_records,
        }
        if latest_samples is not None:
            tick_record["latest_samples"] = latest_samples
        self._append_jsonl("camera_ticks.jsonl", tick_record)

    def append_subscriber_samples(self, drained: DrainedSamples) -> None:
        if self.temp_dir is None:
            return
        for sample in drained.pose_controls:
            self._append_jsonl("pose.jsonl", self._pose_control_record(sample))
        for sample in drained.pose_raws:
            self._append_jsonl("teleop_raw.jsonl", self._pose_raw_record(sample))
        for sample in drained.state_actions:
            self._append_jsonl("robot_state_action.jsonl", self._state_action_record(sample))
        for sample in drained.hand_statuses:
            self._append_jsonl("hand_status.jsonl", self._hand_status_record(sample))

    def _append_jsonl(self, name: str, payload: dict) -> None:
        if self.temp_dir is None or self._jsonl_writer is None:
            return
        self._jsonl_writer.write(self.temp_dir / name, payload)

    def commit(self) -> Path:
        if self.episode_dir is None or self.temp_dir is None:
            raise RuntimeError("episode not started")
        LOGGER.info("finalizing pending writes")
        self._flush_pending_writes()
        if self.defer_depth_compression:
            LOGGER.info("deferring depth compression until collector exit")
        else:
            LOGGER.info("compressing depth outputs")
            self._compress_depth_outputs()
        LOGGER.info("merging episode metadata and samples")
        data_path = SessionMerger(
            self.episode_dir,
            self.temp_dir,
            capture_fps=self.capture_fps,
            sample_mode=self.sample_mode,
        ).write(primary_camera=self.primary_camera)
        if not self.keep_temp_logs and self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        return data_path

    def discard(self) -> None:
        if self.episode_dir is None:
            return
        owned_episode = self._is_current_owned_episode()
        self._close_writers(cancel=True)
        if owned_episode:
            shutil.rmtree(self.episode_dir, ignore_errors=True)
        else:
            LOGGER.warning("refusing to discard unowned episode directory: %s", self.episode_dir)
        self._reset()

    def finalize(self) -> None:
        self._close_writers(cancel=False)
        self._remove_owner_marker()
        self._reset()

    def _reset(self) -> None:
        self.episode_dir = None
        self.temp_dir = None
        self.camera_infos = {}
        self.primary_camera = ""
        self._jsonl_writer = None
        self._image_writer = None
        self._depth_writer = None
        self._owned_episode_dir = None
        self._owner_token = uuid.uuid4().hex

    def _pose_control_record(self, sample: PoseControlSample) -> dict:
        return {
            "timestamp_ns": sample.timestamp_ns,
            "frame_index": sample.frame_index,
            "human_derived": sample.human_derived,
        }

    def _pose_raw_record(self, sample: PoseRawSample) -> dict:
        return {
            "timestamp_ns": sample.timestamp_ns,
            "frame_index": sample.frame_index,
            "human_raw": sample.human_raw,
        }

    def _state_action_record(self, sample: RobotStateActionSample) -> dict:
        return {
            "timestamp_ns": sample.timestamp_ns,
            "obs": sample.obs,
            "action": sample.action,
        }

    def _hand_status_record(self, sample: HandStatusSample) -> dict:
        return {
            "timestamp_ns": sample.timestamp_ns,
            "source_frame_index": sample.source_frame_index,
            "feedback": to_python(sample.feedback),
        }

    def _flush_pending_writes(self) -> None:
        if self._jsonl_writer is not None:
            self._jsonl_writer.flush()
        if self._image_writer is not None:
            self._image_writer.flush()
        if self._depth_writer is not None:
            self._depth_writer.flush()

    def _close_writers(self, *, cancel: bool) -> None:
        if self._jsonl_writer is not None:
            self._jsonl_writer.close(cancel=cancel)
        if self._image_writer is not None:
            self._image_writer.close(cancel=cancel)
        if self._depth_writer is not None:
            self._depth_writer.close(cancel=cancel)

    def _compress_depth_outputs(self) -> None:
        if self.episode_dir is None or not self.record_depth:
            return
        _compress_depth_paths(_episode_raw_depth_paths(self.episode_dir), delete_raw=True)

    @staticmethod
    def _raw_depth_path(depth_path: Path) -> Path:
        return depth_path.with_suffix("")

    def _owner_marker_path(self) -> Path | None:
        if self.episode_dir is None:
            return None
        return self.episode_dir / OWNER_MARKER_NAME

    def _write_owner_marker(self) -> None:
        marker_path = self._owner_marker_path()
        if marker_path is None:
            return
        marker_path.write_text(self._owner_token, encoding="utf-8")

    def _remove_owner_marker(self) -> None:
        marker_path = self._owner_marker_path()
        if marker_path is None:
            return
        marker_path.unlink(missing_ok=True)

    def _is_current_owned_episode(self) -> bool:
        if self.episode_dir is None or self._owned_episode_dir is None:
            return False
        if self.episode_dir.resolve() != self._owned_episode_dir:
            return False
        marker_path = self._owner_marker_path()
        if marker_path is None or not marker_path.exists():
            return False
        return marker_path.read_text(encoding="utf-8").strip() == self._owner_token
