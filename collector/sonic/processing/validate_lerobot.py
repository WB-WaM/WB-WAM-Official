#!/usr/bin/env python3
"""Validate a SONIC native LeRobot v3 record or a task/record archive, offline."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "processing"

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .common import contained_path, read_json, write_json
from .schema import CAMERA, FPS


def validate_record(root: Path, *, decode=True) -> dict:
    info = read_json(contained_path(root, "meta/info.json"))
    if info.get("codebase_version") != "v3.0" or info.get("fps") != FPS:
        raise ValueError(f"{root}: expected native v3.0 at {FPS} Hz")
    columns = {"observation.state": 110, "action": 136, "state_mask_110": 110, "action_mask_136": 136}
    features = info["features"]
    for key, dim in columns.items():
        if features[key]["shape"] != [dim]:
            raise ValueError(f"Invalid feature dimension: {key}")
    tasks = pq.read_table(contained_path(root, "meta/tasks.parquet")).to_pylist()
    task_map = {int(row["task_index"]): row["__index_level_0__"] for row in tasks}
    if len(task_map) != len(tasks) or len(tasks) != info["total_tasks"]:
        raise ValueError("Duplicate or inconsistent task indices")
    for task in task_map.values():
        if not isinstance(task, str) or not task.strip() or "_" in task or task != " ".join(task.split()):
            raise ValueError("Task text must be nonempty, normalized natural language")
    metadata_files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not metadata_files:
        raise ValueError("Missing native v3 episode Parquet metadata")
    episodes = [row for path in metadata_files for row in pq.read_table(path).to_pylist()]
    episodes.sort(key=lambda row: row["episode_index"])
    if len(episodes) != info["total_episodes"] or not episodes:
        raise ValueError("Episode count mismatch")
    data_paths = {(int(row["data/chunk_index"]), int(row["data/file_index"])) for row in episodes}
    if len(data_paths) != 1:
        raise ValueError("The WB record contract requires exactly one numerical Parquet")
    chunk, file = next(iter(data_paths))
    table = pq.read_table(contained_path(root, info["data_path"].format(chunk_index=chunk, file_index=file)))
    if len(table) != info["total_frames"]:
        raise ValueError("Numerical row count mismatch")
    arrays = {key: np.asarray(table[key].to_pylist()) for key in columns}
    for key, size in columns.items():
        if arrays[key].shape != (len(table), size) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid shape or nonfinite values: {key}")
        expected_dtype = "bool" if "mask" in key else "float32"
        if features[key]["dtype"] != expected_dtype or table[key].type != pa.list_(
            pa.type_for_alias(expected_dtype)
        ):
            raise ValueError(f"Incorrect dtype for {key}")
        if "mask" in key and arrays[key].any():
            raise ValueError("SONIC export must not silently mask missing dimensions")
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index", "next.done"):
        dtype = "float32" if key == "timestamp" else "bool" if key == "next.done" else "int64"
        if table[key].type != pa.type_for_alias(dtype) or table[key].null_count:
            raise ValueError(f"Incorrect dtype or null scalar: {key}")
    if not np.array_equal(np.asarray(table["index"]), np.arange(len(table))):
        raise ValueError("Global index is not contiguous")
    videos, offset = {}, 0
    for index, episode in enumerate(episodes):
        size = int(episode["length"])
        if size < 1 or episode["episode_index"] != index:
            raise ValueError("Invalid episode index or length")
        if episode["dataset_from_index"] != offset or episode["dataset_to_index"] != offset + size:
            raise ValueError("Episode offsets must be contiguous half-open intervals")
        part = table.slice(offset, size)
        if not np.all(np.asarray(part["episode_index"]) == index):
            raise ValueError("Episode index in numerical table is inconsistent")
        if not np.array_equal(np.asarray(part["frame_index"]), np.arange(size)):
            raise ValueError("Frame index is not contiguous")
        if not np.array_equal(np.asarray(part["timestamp"]), np.arange(size, dtype=np.float32) / FPS):
            raise ValueError("Frame timestamps are not 20 Hz")
        if not np.array_equal(np.asarray(part["next.done"]), np.arange(size) == size - 1):
            raise ValueError("Invalid next.done flags")
        task_indices = set(np.asarray(part["task_index"]).tolist())
        if len(task_indices) != 1 or not task_indices <= task_map.keys():
            raise ValueError("Invalid per-episode task_index")
        if episode["tasks"] != [task_map[next(iter(task_indices))]]:
            raise ValueError("Episode task text does not match tasks.parquet")
        state = arrays["observation.state"][offset : offset + size]
        action = arrays["action"][offset : offset + size]
        if not np.allclose(action[:-1, 104:133], state[1:, 9:38], atol=1e-5, rtol=0):
            raise ValueError("Next-state body action is shifted incorrectly")
        if not np.allclose(action[:-1, 133:136], state[1:, 107:110], atol=1e-5, rtol=0):
            raise ValueError("Next-state root3 action is shifted incorrectly")
        token = action[:, :64]
        if np.max(np.abs(token * 16 - np.rint(token * 16))) > 1e-5:
            raise ValueError("Tokens are not on the SONIC FSQ grid")
        video_key = (episode[f"videos/{CAMERA}/chunk_index"], episode[f"videos/{CAMERA}/file_index"])
        start = float(episode[f"videos/{CAMERA}/from_timestamp"]) * FPS
        end = float(episode[f"videos/{CAMERA}/to_timestamp"]) * FPS
        expected_start = videos.get(video_key, 0)
        if abs(start - expected_start) > 1e-4 or abs(end - start - size) > 1e-4:
            raise ValueError("Non-contiguous or incorrect episode video boundaries")
        videos[video_key] = expected_start + size
        offset += size
    if offset != len(table):
        raise ValueError("Episode lengths do not cover all numerical rows")
    height, width, channels = features[CAMERA]["shape"]
    if channels != 3:
        raise ValueError("Expected RGB video")
    for (chunk, file), expected_frames in videos.items():
        path = contained_path(
            root,
            info["video_path"].format(
                video_key=CAMERA,
                chunk_index=chunk,
                file_index=file,
            ),
        )
        if decode:
            count = 0
            with av.open(str(path)) as container:
                container.streams.video[0].thread_count = 1
                for frame in container.decode(video=0):
                    if (frame.width, frame.height) != (width, height):
                        raise ValueError(f"Video geometry mismatch: {path}")
                    if frame.pts is None or abs(float(frame.pts * frame.time_base) - count / FPS) > 1e-5:
                        raise ValueError(f"Video PTS mismatch: {path} frame {count}")
                    count += 1
            if count != expected_frames:
                raise ValueError(f"Video length mismatch: {path}: {count} != {expected_frames}")
    stats = read_json(contained_path(root, "meta/stats.json"))
    for key, shape in (("observation.state", (110,)), ("action", (136,)), (CAMERA, (3, 1, 1))):
        for field in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
            value = np.asarray(stats[key][field])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid statistics for {key}.{field}")
        if key != CAMERA:
            if stats[key]["count"] != [len(table)]:
                raise ValueError(f"Incorrect statistics count for {key}")
            for field in ("min", "max", "mean", "std"):
                expected = getattr(arrays[key], field)(axis=0)
                if not np.allclose(stats[key][field], expected, rtol=1e-6, atol=1e-8):
                    raise ValueError(f"Incorrect statistics for {key}.{field}")
    return {"ok": True, "episodes": len(episodes), "frames": offset, "videos": len(videos), "decoded": decode}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional JSON report outside the dataset")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    records = (
        [root]
        if (root / "meta/info.json").is_file()
        else sorted(path.parent.parent for path in root.glob("*/record_*/meta/info.json"))
    )
    try:
        if not records:
            raise ValueError("No records found at --root")
        if args.report and args.report.resolve().is_relative_to(root):
            raise ValueError("--report must be outside --root to keep validation read-only")
        results = {}
        for record in records:
            result = validate_record(record)
            results[str(record.relative_to(root))] = result
            print(f"[OK] {record}: {result}")
        if args.report:
            write_json(args.report, results)
    except (ValueError, KeyError, OSError, av.FFmpegError) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
