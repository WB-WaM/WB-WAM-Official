"""Media and statistics helpers for the HGPT LeRobot exporter.

Extracted from the legacy SONIC preprocessing tools so collection exports do
not depend on the root datasets/ source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import lzma
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import imageio.v2 as iio
import numpy as np


@dataclass(frozen=True)
class DatasetStats:
    min: list[float]
    max: list[float]
    mean: list[float]
    std: list[float]
    count: int


def compute_dataset_stats(vectors: Sequence[Sequence[float]]) -> DatasetStats:
    if not vectors:
        raise ValueError("cannot compute stats for an empty vector list")
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("stats input must be a 2D vector list")
    return DatasetStats(
        min=np.min(arr, axis=0).astype(float).tolist(),
        max=np.max(arr, axis=0).astype(float).tolist(),
        mean=np.mean(arr, axis=0).astype(float).tolist(),
        std=np.std(arr, axis=0).astype(float).tolist(),
        count=int(arr.shape[0]),
    )


def resolve_frame_camera(frame: Mapping[str, Any], camera: str) -> str:
    if camera != "primary":
        return camera
    primary = frame.get("primary_camera")
    if not primary:
        raise ValueError("camera='primary' but frame is missing primary_camera")
    return str(primary)


def frame_camera_record(frame: Mapping[str, Any], camera: str) -> Mapping[str, Any]:
    camera_key = resolve_frame_camera(frame, camera)
    cameras = frame.get("cameras")
    if not isinstance(cameras, Mapping) or camera_key not in cameras:
        raise ValueError(f"frame is missing camera entry: {camera_key}")
    camera_info = cameras[camera_key]
    if not isinstance(camera_info, Mapping):
        raise ValueError(f"frame camera {camera_key} is not a mapping")
    return camera_info


def frame_image_path(episode_dir: Path, frame: Mapping[str, Any], camera: str) -> Path:
    camera_key = resolve_frame_camera(frame, camera)
    camera_info = frame_camera_record(frame, camera)
    if not camera_info.get("color"):
        raise ValueError(f"frame camera {camera_key} is missing color path")
    return episode_dir / str(camera_info["color"])


def frame_depth_path(episode_dir: Path, frame: Mapping[str, Any], camera: str) -> Path:
    camera_key = resolve_frame_camera(frame, camera)
    camera_info = frame_camera_record(frame, camera)
    if not camera_info.get("depth"):
        raise ValueError(f"frame camera {camera_key} is missing depth path")
    return episode_dir / str(camera_info["depth"])


def _load_depth(depth_path: Path) -> np.ndarray:
    if depth_path.suffix == ".lzma":
        with lzma.open(depth_path, "rb") as handle:
            return np.load(handle, allow_pickle=False)
    with depth_path.open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def encode_depth_to_gray_rgb(depth: np.ndarray, *, max_depth_mm: float) -> np.ndarray:
    if max_depth_mm <= 0:
        raise ValueError("max_depth_mm must be > 0")
    depth_arr = np.asarray(depth)
    if depth_arr.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth_arr.shape}")
    depth_mm = depth_arr.astype(np.float32, copy=False)
    finite = np.isfinite(depth_mm)
    if not finite.all():
        depth_mm = depth_mm.copy()
        depth_mm[~finite] = 0.0
    if np.issubdtype(depth_arr.dtype, np.floating) and finite.any() and float(depth_mm[finite].max()) <= 20.0:
        depth_mm = depth_mm * 1000.0
    gray = np.rint(np.clip(depth_mm, 0.0, max_depth_mm) / max_depth_mm * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _iter_encoded_depth_images(depth_paths: Iterable[Path], *, max_depth_mm: float) -> Iterable[np.ndarray]:
    for path in depth_paths:
        yield encode_depth_to_gray_rgb(_load_depth(path), max_depth_mm=max_depth_mm)


def _write_video(video_path: Path, image_paths: Sequence[Path], *, fps: int, codec: str) -> None:
    try:
        writer = iio.get_writer(str(video_path), format="FFMPEG", fps=fps, codec=codec)
        try:
            for path in image_paths:
                writer.append_data(iio.imread(path))
        finally:
            writer.close()
    except ImportError:
        _write_video_with_ffmpeg_cli(video_path, image_paths, fps=fps, codec=codec)


def _write_video_from_arrays(
    video_path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: int,
    codec: str,
) -> None:
    try:
        writer = iio.get_writer(str(video_path), format="FFMPEG", fps=fps, codec=codec)
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
    except ImportError:
        _write_video_arrays_with_ffmpeg_cli(video_path, frames, fps=fps, codec=codec)


def _write_video_with_ffmpeg_cli(
    video_path: Path,
    image_paths: Sequence[Path],
    *,
    fps: int,
    codec: str,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "MP4 writing requires either imageio[ffmpeg] or a system ffmpeg binary"
        )
    with tempfile.TemporaryDirectory(prefix="ffmpeg_frames_", dir=str(video_path.parent)) as tmp:
        frame_dir = Path(tmp)
        for idx, source in enumerate(image_paths):
            target = frame_dir / f"frame_{idx:06d}.jpg"
            try:
                os.symlink(source.resolve(), target)
            except OSError:
                shutil.copy2(source, target)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%06d.jpg"),
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with code {result.returncode}: {result.stderr.strip()}")


def _write_video_arrays_with_ffmpeg_cli(
    video_path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: int,
    codec: str,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "MP4 writing requires either imageio[ffmpeg] or a system ffmpeg binary"
        )
    with tempfile.TemporaryDirectory(prefix="ffmpeg_frames_", dir=str(video_path.parent)) as tmp:
        frame_dir = Path(tmp)
        frame_count = 0
        for idx, frame in enumerate(frames):
            iio.imwrite(frame_dir / f"frame_{idx:06d}.png", frame)
            frame_count += 1
        if frame_count == 0:
            raise ValueError(f"no frames to write for {video_path}")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%06d.png"),
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with code {result.returncode}: {result.stderr.strip()}")
