"""Native LeRobot v3 writer matching the released real_archive record layout.

One numerical Parquet per record; videos split only between episodes. Metadata
and statistics follow the same layout as the archive release packager.
"""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from .common import file_hash, image_paths, load_frames, write_json
from .schema import ACTION_LAYOUT, ACTION_SEMANTICS, CAMERA, FPS, STATE_LAYOUT, episode_arrays


def statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    result = {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }
    for q in (0.01, 0.10, 0.50, 0.90, 0.99):
        result[f"q{int(q * 100):02d}"] = np.quantile(values, q, axis=0).tolist()
    return result


def numeric_table(states, actions, episode_index: int, offset: int) -> pa.Table:
    size = len(states)
    data = {
        "observation.state": pa.array(states.tolist(), pa.list_(pa.float32())),
        "action": pa.array(actions.tolist(), pa.list_(pa.float32())),
        "state_mask_110": pa.array(np.zeros((size, 110), bool).tolist(), pa.list_(pa.bool_())),
        "action_mask_136": pa.array(np.zeros((size, 136), bool).tolist(), pa.list_(pa.bool_())),
        "timestamp": pa.array(np.arange(size, dtype=np.float32) / FPS, pa.float32()),
        "frame_index": pa.array(np.arange(size), pa.int64()),
        "episode_index": pa.array(np.full(size, episode_index), pa.int64()),
        "index": pa.array(np.arange(offset, offset + size), pa.int64()),
        "task_index": pa.array(np.zeros(size, dtype=np.int64)),
        "next.done": pa.array(np.arange(size) == size - 1),
    }
    return pa.table(data)


class VideoWriter:
    def __init__(self, root: Path, image_size: tuple[int, int], max_frames: int):
        self.root, self.size, self.max_frames = root, image_size, max_frames
        self.container = None
        self.stream = None
        self.index = -1
        self.frames = 0

    def close(self):
        if self.container is not None:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.container = None

    def start_episode(self, length: int) -> dict:
        if self.container is None or self.frames + length > self.max_frames:
            self.close()
            self.index += 1
            self.frames = 0
            path = self.root / f"videos/{CAMERA}/chunk-000/file-{self.index:03d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.container = av.open(str(path), "w")
            self.stream = self.container.add_stream("libx264", rate=FPS)
            self.stream.width, self.stream.height = self.size
            self.stream.pix_fmt = "yuv420p"
            self.stream.codec_context.thread_count = 1
            self.stream.options = {"crf": "18", "preset": "medium"}
        return {
            f"videos/{CAMERA}/chunk_index": 0,
            f"videos/{CAMERA}/file_index": self.index,
            f"videos/{CAMERA}/from_timestamp": self.frames / FPS,
            f"videos/{CAMERA}/to_timestamp": (self.frames + length) / FPS,
        }

    def append(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            width, height = self.size
            if image.width * height != image.height * width:
                raise ValueError(f"{path}: image aspect ratio differs from --image-size; refusing stretch")
            rgb = np.asarray(image.convert("RGB").resize(self.size, Image.Resampling.LANCZOS))
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts, frame.time_base = self.frames, Fraction(1, FPS)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1
        return rgb


def build_record(job: dict, output: Path, options: dict) -> None:
    width, height = options["image_size"]
    video = VideoWriter(output, (width, height), options["video_max_frames"])
    parquet = None
    all_states, all_actions, pixels, episodes, sources = [], [], [], [], []
    offset = 0
    try:
        for index, episode in enumerate(job["episodes"]):
            frames = load_frames(episode)
            try:
                states, actions = episode_arrays(frames)
            except ValueError as exc:
                raise ValueError(f"{episode}: {exc}") from exc
            images = image_paths(episode, frames, job["camera"])
            size = len(states)
            table = numeric_table(states, actions, index, offset)
            if parquet is None:
                path = output / "data/chunk-000/file-000.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                parquet = pq.ParquetWriter(path, table.schema, compression="zstd")
            parquet.write_table(table, row_group_size=8192)
            row = {
                "episode_index": index,
                "tasks": [job["task"]],
                "length": size,
                "dataset_from_index": offset,
                "dataset_to_index": offset + size,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                **video.start_episode(size),
            }
            for image_index, path in enumerate(images):
                rgb = video.append(path)
                if image_index in {0, size // 2, size - 1}:
                    pixels.append(rgb[::4, ::4].reshape(-1, 3))
            episodes.append(row)
            sources.append(
                {
                    "episode_index": index,
                    "source_episode": episode.name,
                    "data_sha256": file_hash(episode / "data.json"),
                    "source_frames": len(frames),
                    "output_frames": size,
                }
            )
            all_states.append(states)
            all_actions.append(actions)
            offset += size
            print(f"[OK] {job['relative_output']} {episode.name}: {size} samples", flush=True)
    finally:
        try:
            video.close()
        finally:
            if parquet is not None:
                parquet.close()

    features = {
        "observation.state": {"dtype": "float32", "shape": [110], "names": None},
        "action": {"dtype": "float32", "shape": [136], "names": None},
        "state_mask_110": {"dtype": "bool", "shape": [110], "names": None},
        "action_mask_136": {"dtype": "bool", "shape": [136], "names": None},
    }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index", "next.done"):
        dtype = "float32" if key == "timestamp" else "bool" if key == "next.done" else "int64"
        features[key] = {"dtype": dtype, "shape": [1], "names": None}
    features[CAMERA] = {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": FPS,
            "video.height": height,
            "video.width": width,
            "video.channels": 3,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "has_audio": False,
            "video.is_depth_map": False,
        },
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1_wuji",
        "fps": FPS,
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "state_layout": STATE_LAYOUT,
        "action_layout": ACTION_LAYOUT,
        "action_semantics": ACTION_SEMANTICS,
        "mask_semantics": "True=invalid/not supervised; all exported dimensions are valid",
        "image_stats_sampling": "first/middle/last source RGB per episode, resized, stride 4 pixels",
    }
    write_json(output / "meta/info.json", info)
    pd.DataFrame({"task_index": [0]}, index=[job["task"]]).to_parquet(output / "meta/tasks.parquet")
    episode_path = output / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episodes), episode_path, compression="zstd")
    stats = {
        "observation.state": statistics(np.concatenate(all_states)),
        "action": statistics(np.concatenate(all_actions)),
    }
    image_stats = statistics(np.concatenate(pixels).astype(np.float64) / 255)
    stats[CAMERA] = {
        key: np.asarray(value).reshape(3, 1, 1).tolist() if key != "count" else [len(pixels)]
        for key, value in image_stats.items()
    }
    write_json(output / "meta/stats.json", stats)
    write_json(
        output / "meta/conversion.json",
        {
            "schema_version": 1,
            "source_task": job["source"].name,
            "capture_mode": job["capture_mode"],
            "source_fps": FPS,
            "output_fps": FPS,
            "camera": job["camera"],
            "include_depth": False,
            "alignment": "preserve collector frame pairing; no interpolation or new image-state matching",
            "timestamp": "frame_index / 20; raw clock timestamps remain in the source dataset",
            "action_semantics": ACTION_SEMANTICS,
            "episodes": sources,
        },
    )
