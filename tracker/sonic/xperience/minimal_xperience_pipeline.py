#!/usr/bin/env python3
"""Minimal Xperience-10M processing for G1 + WUJI + SONIC experiments.

This file is intentionally self-contained and writes only to paths passed on the
command line. It keeps the raw episode small by downloading only:

  - annotation.hdf5
  - stereo_left.mp4
  - stereo_right.mp4

The processed episode keeps Xperience depth/imu/video/metadata/caption, drops
calibration and slam, repairs missing hand mocap with carry-forward/zero fill,
creates a 40-D WUJI hand action, prepares 107-D states and 104-D actions,
and encodes SONIC action tokens through the same exported encoder contract used
by the deploy code when onnxruntime is available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import queue as queue_module
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SONIC_ROOT = Path(__file__).resolve().parents[1]
WB_ROOT = SONIC_ROOT.parents[1]

import h5py
import numpy as np
import pandas as pd


FULL_REPO_ID = "ropedia-ai/xperience-10m"
SAMPLE_REPO_ID = "ropedia-ai/xperience-10m-sample"
REPO_TYPE = "dataset"
KEEP_FILENAMES = {"annotation.hdf5", "stereo_left.mp4", "stereo_right.mp4"}
DROP_HDF5_GROUPS = {"calibration", "slam"}
PROCESSED_COMPLETE_MARKER = "processing_complete.json"
PROCESSED_REQUIRED_FILENAMES = (
    "state_action_107_104.parquet",
    "stereo_left.mp4",
    "stereo_right.mp4",
    "caption.json",
    "episode_metadata.json",
    "lerobot_episode_manifest.json",
    "processing_report.json",
)
INTERMEDIATE_OUTPUT_FILENAMES = (
    "wuji_hand_action_40d.npy",
    "wuji_retarget_input_hand_joints.npz",
    "wuji_retarget_output_qpos.npz",
    "g1_motion_gmr_raw.npz",
    "g1_motion_gmr.npz",
    "g1_motion_gmr_replay.pkl",
    "g1_motion_minimal.npz",
    "smpl_motion_minimal.npz",
    "gmr_input_smpl24_joints.npz",
    "gmr_input_smplx_params.npz",
    "sonic_encoder_obs_1762d.npy",
    "sonic_action_token_64d.npy",
)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
TOKEN_DIM = 64
WUJI_QPOS_DIM = 20
HAND_ACTION_DIM = 2 * WUJI_QPOS_DIM
STATE_DIM = 107
ACTION_DIM = TOKEN_DIM + HAND_ACTION_DIM
SONIC_ENCODER_OBS_DIM = 1762
DEFAULT_FPS = 20
SONIC_G1_FUTURE_DT = 0.1
SONIC_SMPL_FUTURE_DT = 0.02
FSQ_GRID_STEP = 1.0 / 16.0
HAND_MIN_MEAN_TIP_DISTANCE_M = 0.03
MANO_TIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)
GMR_POST_SMOOTH_ENABLED = True
GMR_POST_SMOOTH_WINDOW = 9
GMR_POST_SMOOTH_POLYORDER = 2
DEFAULT_WUJI_ROOT = Path(os.environ.get("WB_WAM_WUJI_ROOT", WB_ROOT / "third_party" / "wuji_retargeting"))
DEFAULT_WUJI_PYTHON = Path(os.environ.get("WB_WAM_WUJI_PYTHON", sys.executable))
DEFAULT_WUJI_LEFT_CONFIG = DEFAULT_WUJI_ROOT / "example/config/retarget_manus_left.yaml"
DEFAULT_WUJI_RIGHT_CONFIG = DEFAULT_WUJI_ROOT / "example/config/retarget_manus_right.yaml"

LOWER_BODY_ISAAC_INDEX = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int64)
G1_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
WRIST_ISAAC_INDEX = np.array([23, 24, 25, 26, 27, 28], dtype=np.int64)
DEFAULT_G1_QPOS = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

SMPL_24_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)
SMPL_24_PARENT_INDEX = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)
GMR_SMPLX_BODY_INDEX: dict[str, int] = {
    "pelvis": 0,
    "spine3": 9,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_foot": 10,
    "right_foot": 11,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

ENCODER_LAYOUT: tuple[tuple[str, int], ...] = (
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_dt0.1s", 290),
    ("motion_joint_velocities_10frame_dt0.1s", 290),
    ("motion_root_z_position_10frame_dt0.1s", 10),
    ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6),
    ("motion_anchor_orientation_10frame_dt0.1s", 60),
    ("motion_joint_positions_lowerbody_10frame_dt0.1s", 120),
    ("motion_joint_velocities_lowerbody_10frame_dt0.1s", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_dt0.02s_clamped_to_dataset_fps", 720),
    ("smpl_anchor_orientation_10frame_dt0.02s_clamped_to_dataset_fps", 60),
    ("motion_joint_positions_wrists_10frame_dt0.02s_clamped_to_dataset_fps", 60),
)


class DiskAlmostFull(RuntimeError):
    pass


@dataclass(frozen=True)
class HandValidityReport:
    side: str
    frames: int
    finite_frames: int
    valid_frames: int
    degenerate_frames: int
    invalid_frames: int
    leading_invalid_frames: int
    invalid_runs: list[list[int]]
    source_path: str | None


@dataclass(frozen=True)
class RetargetStatus:
    method: str
    ok: bool
    message: str
    artifact: str | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def existing_parent(path: Path) -> Path:
    path = path.expanduser().resolve()
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    return probe


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(existing_parent(path))
    return int(usage.free)


def ensure_space(path: Path, min_free_gb: float, incoming_bytes: int = 0) -> None:
    min_free = int(min_free_gb * (1024**3))
    free = free_bytes(path)
    if free - int(incoming_bytes) < min_free:
        raise DiskAlmostFull(
            f"free space would fall below {min_free_gb:.1f} GiB at {path}: "
            f"free={free / (1024**3):.2f} GiB incoming={incoming_bytes / (1024**3):.2f} GiB"
        )


def import_hf() -> Any:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("huggingface_hub is required for download/list-remote") from exc
    return HfApi, hf_hub_download


def hf_repo_id(sample: bool) -> str:
    return SAMPLE_REPO_ID if sample else FULL_REPO_ID


def h5_read(path: str, h5: h5py.File, default: Any = None) -> Any:
    if path in h5:
        return h5[path][()]
    return default


def first_h5_dataset(h5: h5py.File, paths: Sequence[str]) -> tuple[str | None, Any]:
    for path in paths:
        if path in h5 and isinstance(h5[path], h5py.Dataset):
            return path, h5[path][()]
    return None, None


def decode_h5_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return item
    if arr.dtype.kind in {"S", "O", "U"}:
        return [decode_h5_scalar(x) for x in arr.reshape(-1).tolist()]
    return arr.tolist()


def h5_attrs_to_dict(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {str(k): decode_h5_scalar(v) for k, v in obj.attrs.items()}


def read_h5_dataset_compact(h5: h5py.File, path: str, max_items: int = 2048) -> Any:
    if path not in h5 or not isinstance(h5[path], h5py.Dataset):
        return None
    value = h5[path][()]
    arr = np.asarray(value)
    if arr.shape == () or arr.size <= max_items:
        return decode_h5_scalar(value)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "summary": "omitted_large_array",
    }


def extract_caption_payload(h5: h5py.File) -> dict[str, Any]:
    if "caption" not in h5 or not isinstance(h5["caption"], h5py.Dataset):
        return {"ok": False, "message": "caption dataset missing"}
    raw = decode_h5_scalar(h5["caption"][()])
    text = raw if isinstance(raw, str) else json.dumps(raw, default=_json_default)
    payload: dict[str, Any] = {"ok": True, "raw_text": text}
    try:
        parsed = json.loads(text)
        payload["parsed"] = parsed
        config = parsed.get("config") if isinstance(parsed, Mapping) else None
        if isinstance(config, Mapping):
            payload["main_task"] = config.get("Main Task") or config.get("main_task")
            payload["total_frames"] = config.get("total_frames")
            payload["total_tokens"] = config.get("total_tokens")
        segments = parsed.get("segments") if isinstance(parsed, Mapping) else None
        if isinstance(segments, list):
            payload["segments_count"] = len(segments)
    except Exception as exc:
        payload["parse_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def extract_compact_metadata(h5: h5py.File, caption_payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hdf5_attrs": h5_attrs_to_dict(h5),
        "caption_main_task": caption_payload.get("main_task"),
        "caption_segments_count": caption_payload.get("segments_count"),
    }
    if "metadata" in h5 and isinstance(h5["metadata"], h5py.Group):
        group = h5["metadata"]
        metadata["metadata_attrs"] = h5_attrs_to_dict(group)
        datasets: dict[str, Any] = {}
        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets[name] = read_h5_dataset_compact(h5, f"metadata/{name}")
        group.visititems(visitor)
        metadata["metadata_datasets"] = datasets
    for path in ("video/frame_nums", "video/frame_number", "full_body_mocap/frame_nums"):
        value = read_h5_dataset_compact(h5, path)
        if value is not None:
            metadata[path.replace("/", "_")] = value
    return metadata


def write_lerobot_ready_files(
    *,
    output_dir: Path,
    caption_payload: Mapping[str, Any],
    compact_metadata: Mapping[str, Any],
    report: Mapping[str, Any],
    rows: int,
) -> dict[str, str]:
    caption_path = output_dir / "caption.json"
    metadata_path = output_dir / "episode_metadata.json"
    manifest_path = output_dir / "lerobot_episode_manifest.json"
    task = caption_payload.get("main_task")
    if not task:
        task = output_dir.name
    write_json(caption_path, caption_payload)
    write_json(metadata_path, compact_metadata)
    manifest = {
        "format": "xperience_lerobot_ready_v1",
        "episode_id": output_dir.name,
        "task": task,
        "fps": report.get("fps"),
        "frames": report.get("frames"),
        "rows": rows,
        "state_dim": report.get("state_dim"),
        "action_dim": report.get("action_dim"),
        "videos": {
            "stereo_left": "stereo_left.mp4",
            "stereo_right": "stereo_right.mp4",
        },
        "state_action": "state_action_107_104.parquet",
        "caption": "caption.json",
        "metadata": "episode_metadata.json",
        "notes": (
            "LeRobot conversion can build episode_index/index/task_index and copy or trim videos "
            "from this compact processed episode."
        ),
    }
    write_json(manifest_path, manifest)
    return {
        "caption": caption_path.name,
        "episode_metadata": metadata_path.name,
        "lerobot_episode_manifest": manifest_path.name,
    }


def cleanup_intermediate_outputs(output_dir: Path) -> list[str]:
    removed: list[str] = []
    for filename in INTERMEDIATE_OUTPUT_FILENAMES:
        path = output_dir / filename
        if path.exists():
            if path.is_file():
                path.unlink()
                removed.append(filename)
            elif path.is_dir():
                shutil.rmtree(path)
                removed.append(filename)
    return removed


def list_remote_files(args: argparse.Namespace) -> int:
    HfApi, _ = import_hf()
    api = HfApi(token=args.token)
    repo_id = hf_repo_id(args.sample)
    rows = []
    for item in api.list_repo_tree(repo_id=repo_id, repo_type=REPO_TYPE, revision=args.revision, recursive=True):
        item_type = getattr(item, "type", None)
        path = getattr(item, "path", "")
        size = getattr(item, "size", None)
        if item_type == "file" or size is not None:
            if args.only_kept and Path(path).name not in KEEP_FILENAMES:
                continue
            rows.append((path, int(size or 0)))
    total = sum(size for _, size in rows)
    for path, size in rows[: args.limit]:
        print(f"{size:12d}  {path}")
    if len(rows) > args.limit:
        print(f"... {len(rows) - args.limit} more")
    print(f"files={len(rows)} total={total / (1024**3):.3f} GiB repo={repo_id}")
    return 0


def group_episode_files(paths: Iterable[tuple[str, int]]) -> dict[str, dict[str, tuple[str, int]]]:
    episodes: dict[str, dict[str, tuple[str, int]]] = {}
    for hf_path, size in paths:
        filename = Path(hf_path).name
        if filename not in KEEP_FILENAMES:
            continue
        parent = str(Path(hf_path).parent)
        episode_id = "sample" if parent == "." else parent.replace("/", "__")
        episodes.setdefault(episode_id, {})[filename] = (hf_path, size)
    return episodes


def remote_kept_files(repo_id: str, token: str | None, revision: str) -> list[tuple[str, int]]:
    HfApi, _ = import_hf()
    api = HfApi(token=token)
    out: list[tuple[str, int]] = []
    for item in api.list_repo_tree(repo_id=repo_id, repo_type=REPO_TYPE, revision=revision, recursive=True):
        path = getattr(item, "path", "")
        size = int(getattr(item, "size", 0) or 0)
        if Path(path).name in KEEP_FILENAMES:
            out.append((path, size))
    return out


def download_one_file(
    *,
    repo_id: str,
    hf_path: str,
    size: int,
    raw_root: Path,
    token: str | None,
    revision: str,
    min_free_gb: float,
) -> Path:
    _, hf_hub_download = import_hf()
    target = raw_root / hf_path
    if target.exists() and (size <= 0 or target.stat().st_size == size):
        return target
    ensure_space(raw_root, min_free_gb, incoming_bytes=size)
    target.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        filename=hf_path,
        revision=revision,
        token=token,
        local_dir=str(raw_root),
    )
    return Path(path)


def episode_raw_dir(raw_root: Path, files: Mapping[str, tuple[str, int]]) -> Path:
    annotation = files.get("annotation.hdf5")
    if not annotation:
        return raw_root
    parent = Path(annotation[0]).parent
    return raw_root if str(parent) == "." else raw_root / parent


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def episode_complete_marker(raw_root: Path, files: Mapping[str, tuple[str, int]]) -> Path:
    return episode_raw_dir(raw_root, files) / ".xperience_download_complete.json"


def episode_files_complete(raw_root: Path, files: Mapping[str, tuple[str, int]]) -> bool:
    for filename in KEEP_FILENAMES:
        item = files.get(filename)
        if item is None:
            return False
        hf_path, expected_size = item
        local = raw_root / hf_path
        if not local.exists() or not local.is_file():
            return False
        if expected_size > 0 and local.stat().st_size != expected_size:
            return False
    return True


def write_episode_complete_marker(
    raw_root: Path,
    episode_id: str,
    files: Mapping[str, tuple[str, int]],
    repo_id: str,
    revision: str,
) -> Path:
    marker = episode_complete_marker(raw_root, files)
    payload = {
        "episode_id": episode_id,
        "repo_id": repo_id,
        "revision": revision,
        "completed_unix_time": time.time(),
        "kept_files": {
            filename: {
                "hf_path": files[filename][0],
                "expected_size": int(files[filename][1]),
                "local_path": str(raw_root / files[filename][0]),
            }
            for filename in sorted(KEEP_FILENAMES)
            if filename in files
        },
    }
    write_json(marker, payload)
    return marker


def delete_episode_raw_files(raw_root: Path, files: Mapping[str, tuple[str, int]]) -> list[str]:
    raw_root = raw_root.expanduser().resolve()
    deleted: list[str] = []
    cleanup_dirs: set[Path] = set()

    candidate_paths: list[Path] = []
    for filename in sorted(KEEP_FILENAMES):
        item = files.get(filename)
        if item is None:
            continue
        candidate_paths.append(raw_root / item[0])
    candidate_paths.append(episode_complete_marker(raw_root, files))

    for candidate in candidate_paths:
        local = candidate.expanduser().resolve()
        if not _path_is_under(local, raw_root):
            raise RuntimeError(f"refusing to delete path outside raw root: {local}")
        if local.exists():
            if not local.is_file():
                raise RuntimeError(f"refusing to delete non-file raw artifact: {local}")
            local.unlink()
            deleted.append(str(local))
        cleanup_dirs.add(local.parent)

    for directory in sorted(cleanup_dirs, key=lambda p: len(p.parts), reverse=True):
        current = directory
        while current != raw_root and _path_is_under(current, raw_root):
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent
    return deleted


def processed_complete_marker(output_dir: Path) -> Path:
    return output_dir.expanduser().resolve() / PROCESSED_COMPLETE_MARKER


def processed_output_complete(output_dir: Path) -> bool:
    output_dir = output_dir.expanduser().resolve()
    marker = processed_complete_marker(output_dir)
    if not marker.exists():
        return False
    for filename in PROCESSED_REQUIRED_FILENAMES:
        path = output_dir / filename
        if not path.exists() or not path.is_file():
            return False
    return True


def write_processed_complete_marker(output_dir: Path, report: Mapping[str, Any]) -> Path:
    output_dir = output_dir.expanduser().resolve()
    marker = processed_complete_marker(output_dir)
    payload = {
        "completed_unix_time": time.time(),
        "output_dir": str(output_dir),
        "source_episode_dir": report.get("source_episode_dir"),
        "frames": report.get("frames"),
        "fps": report.get("fps"),
        "state_dim": report.get("state_dim"),
        "action_dim": report.get("action_dim"),
        "required_files": list(PROCESSED_REQUIRED_FILENAMES),
    }
    write_json(marker, payload)
    return marker


def process_downloaded_episode(
    *,
    args: argparse.Namespace,
    raw_root: Path,
    out_root: Path,
    episode_id: str,
    episode_files: Mapping[str, tuple[str, int]],
) -> dict[str, Any]:
    out_dir = out_root / episode_id
    skip_existing_processed = bool(getattr(args, "skip_existing_processed", False))
    delete_raw_after_parse = bool(getattr(args, "delete_raw_after_parse", False))
    out_complete = processed_output_complete(out_dir)
    if skip_existing_processed and out_complete and not args.overwrite:
        print(f"[INFO] skip existing processed episode {episode_id}: {out_dir}")
        status = "skipped_existing_processed"
    else:
        ep_dir = episode_raw_dir(raw_root, episode_files)
        overwrite = bool(args.overwrite or (out_dir.exists() and not out_complete))
        convert_episode_dir(
            ep_dir,
            out_dir,
            resize_videos=args.resize_videos,
            require_gmr=args.require_gmr,
            gmr_root=args.gmr_root,
            gmr_python=args.gmr_python,
            smplx_folder=args.smplx_folder,
            gmr_robot=args.gmr_robot,
            wuji_root=args.wuji_root,
            wuji_python=args.wuji_python,
            wuji_left_config=args.wuji_left_config,
            wuji_right_config=args.wuji_right_config,
            require_wuji=args.require_wuji,
            encoder_model=args.encoder_model,
            min_free_gb=args.min_free_gb,
            overwrite=overwrite,
            keep_intermediate=args.keep_intermediate,
        )
        status = "converted"

    deleted: list[str] = []
    if delete_raw_after_parse:
        deleted = delete_episode_raw_files(raw_root, episode_files)
        print(f"[INFO] deleted raw files for {episode_id}: files={len(deleted)}")
    return {
        "episode_id": episode_id,
        "status": status,
        "output_dir": str(out_dir),
        "deleted_raw_files": deleted,
    }


def download(args: argparse.Namespace) -> int:
    repo_id = hf_repo_id(args.sample)
    raw_root = args.raw_root.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve() if args.output_root else None
    raw_root.mkdir(parents=True, exist_ok=True)
    files = remote_kept_files(repo_id, args.token, args.revision)
    episodes = group_episode_files(files)
    episode_items = sorted(episodes.items())
    if args.max_episodes is not None:
        episode_items = episode_items[: args.max_episodes]

    print(f"repo={repo_id} episodes={len(episode_items)} raw_root={raw_root}")
    done = 0
    for episode_id, episode_files in episode_items:
        missing = KEEP_FILENAMES - set(episode_files)
        if missing:
            print(f"[WARN] skip {episode_id}: missing {sorted(missing)}")
            continue
        print(f"[INFO] downloading {episode_id}")
        for filename in sorted(KEEP_FILENAMES):
            hf_path, size = episode_files[filename]
            local = download_one_file(
                repo_id=repo_id,
                hf_path=hf_path,
                size=size,
                raw_root=raw_root,
                token=args.token,
                revision=args.revision,
                min_free_gb=args.min_free_gb,
            )
            print(f"  {filename}: {local}")
        if not episode_files_complete(raw_root, episode_files):
            raise RuntimeError(f"episode did not pass completion check after download: {episode_id}")
        marker = write_episode_complete_marker(raw_root, episode_id, episode_files, repo_id, args.revision)
        print(f"  complete: {marker}")
        done += 1
        if args.parse_after_download:
            if out_root is None:
                raise ValueError("--parse-after-download requires --output-root")
            process_downloaded_episode(
                args=args,
                raw_root=raw_root,
                out_root=out_root,
                episode_id=episode_id,
                episode_files=episode_files,
            )
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)
    print(f"[INFO] downloaded episodes={done}")
    return 0


def stream_download_process(args: argparse.Namespace) -> int:
    repo_id = hf_repo_id(args.sample)
    raw_root = args.raw_root.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    files = remote_kept_files(repo_id, args.token, args.revision)
    episodes = group_episode_files(files)
    episode_items = sorted(episodes.items())
    if args.max_episodes is not None:
        episode_items = episode_items[: args.max_episodes]

    process_workers = max(1, int(args.process_workers))
    prefetch_episodes = max(0, int(args.prefetch_episodes))
    raw_capacity = max(1, process_workers + prefetch_episodes)
    work_queue: queue_module.Queue[Any] = queue_module.Queue()
    raw_slots = threading.Semaphore(raw_capacity)
    stop_event = threading.Event()
    stats_lock = threading.Lock()
    errors: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "downloaded": 0,
        "converted": 0,
        "skipped_existing_processed": 0,
        "deleted_raw_files": 0,
    }
    sentinel = object()

    print(
        f"repo={repo_id} episodes={len(episode_items)} raw_root={raw_root} "
        f"output_root={out_root} process_workers={process_workers} "
        f"prefetch_episodes={prefetch_episodes} raw_capacity={raw_capacity}"
    )

    def record_error(stage: str, episode_id: str | None, exc: BaseException) -> None:
        with stats_lock:
            errors.append(
                {
                    "stage": stage,
                    "episode_id": episode_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "is_disk_full_stop": isinstance(exc, DiskAlmostFull),
                }
            )
        stop_event.set()
        print(f"[ERROR] {stage} {episode_id or ''}: {type(exc).__name__}: {exc}", file=sys.stderr)

    def downloader() -> None:
        try:
            for episode_id, episode_files in episode_items:
                if stop_event.is_set():
                    break
                missing = KEEP_FILENAMES - set(episode_files)
                if missing:
                    print(f"[WARN] skip {episode_id}: missing {sorted(missing)}")
                    continue

                raw_slots.acquire()
                queued_for_processing = False
                try:
                    print(f"[INFO] downloading {episode_id}")
                    for filename in sorted(KEEP_FILENAMES):
                        if stop_event.is_set():
                            raise RuntimeError("stream stopped before episode download completed")
                        hf_path, size = episode_files[filename]
                        local = download_one_file(
                            repo_id=repo_id,
                            hf_path=hf_path,
                            size=size,
                            raw_root=raw_root,
                            token=args.token,
                            revision=args.revision,
                            min_free_gb=args.min_free_gb,
                        )
                        print(f"  {episode_id} {filename}: {local}")
                    if not episode_files_complete(raw_root, episode_files):
                        raise RuntimeError(f"episode did not pass completion check: {episode_id}")
                    marker = write_episode_complete_marker(raw_root, episode_id, episode_files, repo_id, args.revision)
                    print(f"  {episode_id} complete: {marker}")
                    if stop_event.is_set():
                        raise RuntimeError("stream stopped after episode download completed")
                    work_queue.put((episode_id, episode_files))
                    queued_for_processing = True
                    with stats_lock:
                        stats["downloaded"] += 1
                    if args.sleep_s > 0:
                        time.sleep(args.sleep_s)
                except BaseException:
                    if not queued_for_processing:
                        raw_slots.release()
                    raise
        except BaseException as exc:
            record_error("download", None, exc)
        finally:
            for _ in range(process_workers):
                work_queue.put(sentinel)

    def processor(worker_id: int) -> None:
        while True:
            item = work_queue.get()
            try:
                if item is sentinel:
                    return
                episode_id, episode_files = item
                try:
                    result = process_downloaded_episode(
                        args=args,
                        raw_root=raw_root,
                        out_root=out_root,
                        episode_id=episode_id,
                        episode_files=episode_files,
                    )
                    with stats_lock:
                        if result["status"] == "converted":
                            stats["converted"] += 1
                        elif result["status"] == "skipped_existing_processed":
                            stats["skipped_existing_processed"] += 1
                        stats["deleted_raw_files"] += len(result["deleted_raw_files"])
                    print(f"[INFO] worker={worker_id} done {episode_id}: {result['status']}")
                except BaseException as exc:
                    record_error("process", episode_id, exc)
                finally:
                    raw_slots.release()
            finally:
                work_queue.task_done()

    download_thread = threading.Thread(target=downloader, name="xperience-downloader", daemon=True)
    process_threads = [
        threading.Thread(target=processor, args=(i,), name=f"xperience-processor-{i}", daemon=True)
        for i in range(process_workers)
    ]
    for thread in process_threads:
        thread.start()
    download_thread.start()
    download_thread.join()
    work_queue.join()
    for thread in process_threads:
        thread.join()

    with stats_lock:
        final_stats = dict(stats)
        final_errors = list(errors)
    print(
        "[INFO] stream summary "
        f"downloaded={final_stats['downloaded']} converted={final_stats['converted']} "
        f"skipped_existing_processed={final_stats['skipped_existing_processed']} "
        f"deleted_raw_files={final_stats['deleted_raw_files']}"
    )
    if final_errors:
        err = final_errors[0]
        print(
            f"[STOP] first error stage={err['stage']} episode={err['episode_id']} "
            f"type={err['exception_type']} message={err['message']}",
            file=sys.stderr,
        )
        return 75 if any(e["is_disk_full_stop"] for e in final_errors) else 1
    return 0


def finite_frame_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.asarray([bool(np.isfinite(arr))])
    flat = arr.reshape(arr.shape[0], -1)
    return np.all(np.isfinite(flat), axis=1)


def invalid_runs(valid: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    start: int | None = None
    for i, ok in enumerate(valid.tolist()):
        if not ok and start is None:
            start = i
        if ok and start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(valid) - 1])
    return runs


def repair_framewise(
    arr: np.ndarray,
    *,
    fill_value: float = 0.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if valid_mask is None:
        valid = finite_frame_mask(arr)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != arr.shape[0]:
            raise ValueError(f"valid_mask length {valid.shape[0]} does not match frames {arr.shape[0]}")
    out = arr.copy()
    last = np.full(arr.shape[1:], fill_value, dtype=np.float32)
    seen_valid = False
    leading_invalid = 0
    for i, ok in enumerate(valid.tolist()):
        if ok:
            last = out[i].copy()
            seen_valid = True
        else:
            if not seen_valid:
                leading_invalid += 1
            out[i] = last
    return out, valid, leading_invalid


def hand_frame_valid_mask(joints: np.ndarray, mano_pose: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(joints, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.reshape(1, *arr.shape)
    if arr.ndim == 3 and arr.shape[1] * arr.shape[2] == 63 and arr.shape[1] != 21:
        arr = arr.reshape(arr.shape[0], 21, 3)
    if arr.ndim != 3 or arr.shape[1] < 21 or arr.shape[2] != 3:
        frames = arr.shape[0] if arr.ndim > 0 else 0
        valid = np.zeros((frames,), dtype=bool)
        return valid, ~valid

    hand = arr[:, :21, :]
    finite = finite_frame_mask(hand)
    nonzero = np.linalg.norm(hand.reshape(hand.shape[0], -1), axis=1) > 1e-8
    tip_distance = np.linalg.norm(hand[:, MANO_TIP_INDICES, :] - hand[:, 0:1, :], axis=-1)
    scale_ok = np.mean(tip_distance, axis=1) >= HAND_MIN_MEAN_TIP_DISTANCE_M
    valid = finite & nonzero & scale_ok
    if mano_pose is not None:
        pose = np.asarray(mano_pose, dtype=np.float32)
        if pose.shape[0] == hand.shape[0]:
            pose_ok = np.linalg.norm(pose.reshape(pose.shape[0], -1), axis=1) > 1e-8
            valid &= pose_ok
    degenerate = finite & ~valid
    return valid, degenerate


def hand_report(
    side: str,
    arr: np.ndarray | None,
    source_path: str | None,
    mano_pose: np.ndarray | None = None,
) -> HandValidityReport:
    if arr is None:
        return HandValidityReport(side, 0, 0, 0, 0, 0, 0, [], source_path)
    arr = np.asarray(arr, dtype=np.float32)
    finite = finite_frame_mask(arr)
    valid, degenerate = hand_frame_valid_mask(arr, mano_pose=mano_pose)
    _, _, leading = repair_framewise(arr, valid_mask=valid)
    return HandValidityReport(
        side=side,
        frames=int(valid.size),
        finite_frames=int(np.count_nonzero(finite)),
        valid_frames=int(np.count_nonzero(valid)),
        degenerate_frames=int(np.count_nonzero(degenerate)),
        invalid_frames=int(valid.size - np.count_nonzero(valid)),
        leading_invalid_frames=int(leading),
        invalid_runs=invalid_runs(valid),
        source_path=source_path,
    )


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = normalize(a - b)
    bc = normalize(c - b)
    cosang = np.sum(ba * bc, axis=-1)
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def joints_to_wuji20(joints: np.ndarray | None, mano_pose: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert MANO-style 21x3 joints to a deterministic 20-D hand proxy.

    This is not a calibrated WUJI retargeter. It is a finite geometric proxy
    that makes sample-pipeline validation possible until a measured WUJI map is
    supplied. Each finger contributes 3 flexion angles plus 1 spread value.
    """

    if joints is None:
        return np.zeros((0, WUJI_QPOS_DIM), dtype=np.float32), {
            "method": "missing_hand_joints_zero",
            "calibrated": False,
        }

    arr = np.asarray(joints, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.reshape(1, *arr.shape)
    if arr.ndim == 3 and arr.shape[1] * arr.shape[2] == 63 and arr.shape[1] != 21:
        arr = arr.reshape(arr.shape[0], 21, 3)
    if arr.ndim != 3 or arr.shape[1] < 21 or arr.shape[2] != 3:
        return np.zeros((arr.shape[0], WUJI_QPOS_DIM), dtype=np.float32), {
            "method": f"unsupported_hand_shape_{tuple(arr.shape)}_zero",
            "calibrated": False,
        }

    valid, degenerate = hand_frame_valid_mask(arr[:, :21, :], mano_pose=mano_pose)
    repaired, repaired_valid, leading_invalid = repair_framewise(arr[:, :21, :], valid_mask=valid)
    wrist = repaired[:, 0]
    middle_mcp = repaired[:, 9]
    index_mcp = repaired[:, 5]
    pinky_mcp = repaired[:, 17]
    palm_forward = normalize(middle_mcp - wrist)
    palm_side = normalize(index_mcp - pinky_mcp)

    fingers = (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    )
    qpos = np.zeros((repaired.shape[0], WUJI_QPOS_DIM), dtype=np.float32)
    for finger_idx, (mcp, pip, dip, tip) in enumerate(fingers):
        base = finger_idx * 4
        flex0 = math.pi - angle_between(wrist, repaired[:, mcp], repaired[:, pip])
        flex1 = math.pi - angle_between(repaired[:, mcp], repaired[:, pip], repaired[:, dip])
        flex2 = math.pi - angle_between(repaired[:, pip], repaired[:, dip], repaired[:, tip])
        ray = normalize(repaired[:, mcp] - wrist)
        spread = np.arctan2(np.sum(ray * palm_side, axis=-1), np.sum(ray * palm_forward, axis=-1))
        qpos[:, base + 0] = np.clip(flex0, 0.0, 2.2)
        qpos[:, base + 1] = np.clip(spread, -0.8, 0.8)
        qpos[:, base + 2] = np.clip(flex1, 0.0, 2.2)
        qpos[:, base + 3] = np.clip(flex2, 0.0, 2.2)
    qpos[~np.isfinite(qpos)] = 0.0
    if leading_invalid > 0:
        qpos[:leading_invalid] = 0.0
    return qpos, {
        "method": "mano21_joint_geometry_proxy_v1",
        "calibrated": False,
        "wuji_order": "finger-major [mcp_flexion, spread_signed, pip_flexion, dip_flexion]",
        "valid_policy": "finite and non-degenerate hand geometry; leading invalid frames zero, later invalid frames carry previous",
        "valid_frames": int(np.count_nonzero(repaired_valid)),
        "invalid_frames": int(repaired_valid.size - np.count_nonzero(repaired_valid)),
        "degenerate_frames": int(np.count_nonzero(degenerate)),
        "leading_invalid_frames": int(leading_invalid),
    }


def resolve_wuji_python(wuji_python: Path | None) -> Path:
    if wuji_python is not None:
        return wuji_python.expanduser().resolve()
    if DEFAULT_WUJI_PYTHON.exists():
        return DEFAULT_WUJI_PYTHON
    return Path(sys.executable).resolve()


def retarget_wuji_subprocess(
    *,
    left_joints: np.ndarray | None,
    right_joints: np.ndarray | None,
    left_pose: np.ndarray | None,
    right_pose: np.ndarray | None,
    output_dir: Path,
    wuji_root: Path | None,
    wuji_python: Path | None,
    wuji_left_config: Path | None,
    wuji_right_config: Path | None,
    require_wuji: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    root = (wuji_root or DEFAULT_WUJI_ROOT).expanduser().resolve()
    left_config = (wuji_left_config or DEFAULT_WUJI_LEFT_CONFIG).expanduser().resolve()
    right_config = (wuji_right_config or DEFAULT_WUJI_RIGHT_CONFIG).expanduser().resolve()
    python = resolve_wuji_python(wuji_python)

    left_arr = None if left_joints is None else np.asarray(left_joints, dtype=np.float32)
    right_arr = None if right_joints is None else np.asarray(right_joints, dtype=np.float32)
    left_valid = np.zeros((0,), dtype=bool) if left_arr is None else hand_frame_valid_mask(left_arr, left_pose)[0]
    right_valid = np.zeros((0,), dtype=bool) if right_arr is None else hand_frame_valid_mask(right_arr, right_pose)[0]

    def fallback(reason: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
        if require_wuji:
            raise RuntimeError(reason)
        left_wuji, left_method = joints_to_wuji20(left_arr, mano_pose=left_pose)
        right_wuji, right_method = joints_to_wuji20(right_arr, mano_pose=right_pose)
        left_method["fallback_from"] = "wuji_retargeting"
        left_method["fallback_reason"] = reason
        right_method["fallback_from"] = "wuji_retargeting"
        right_method["fallback_reason"] = reason
        return left_wuji, right_wuji, left_method, right_method

    missing = [
        str(path)
        for path in (root, left_config, right_config, python)
        if not path.exists()
    ]
    if missing:
        return fallback(f"missing Wuji retarget dependency path(s): {missing}")

    input_npz = output_dir / "wuji_retarget_input_hand_joints.npz"
    output_npz = output_dir / "wuji_retarget_output_qpos.npz"
    np.savez_compressed(
        input_npz,
        left_joints=np.zeros((0, 21, 3), dtype=np.float32) if left_arr is None else left_arr[:, :21, :],
        right_joints=np.zeros((0, 21, 3), dtype=np.float32) if right_arr is None else right_arr[:, :21, :],
        left_valid=left_valid,
        right_valid=right_valid,
    )

    script = r"""
import json
import sys
import time
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
left_config = Path(sys.argv[2])
right_config = Path(sys.argv[3])
input_npz = Path(sys.argv[4])
output_npz = Path(sys.argv[5])
sys.path.insert(0, str(root))

from wuji_retargeting import Retargeter

data = np.load(input_npz)

def solve_side(side, joints, valid, config_path):
    joints = np.asarray(joints, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    out = np.zeros((joints.shape[0], 20), dtype=np.float32)
    meta = {
        "method": "wuji_retargeting.Retargeter",
        "config": str(config_path),
        "hand_side": side,
        "valid_frames": int(np.count_nonzero(valid)),
        "invalid_frames": int(valid.size - np.count_nonzero(valid)),
        "failures": 0,
        "calibrated": True,
        "wuji_order": "finger-major URDF order finger{1..5}_joint{1..4}",
        "filter": "Retargeter low-pass filter from config",
    }
    if joints.shape[0] == 0:
        return out, meta

    retargeter = Retargeter.from_yaml(str(config_path), hand_side=side)
    if hasattr(retargeter.optimizer, "set_timing_enabled"):
        retargeter.optimizer.set_timing_enabled(False)
    meta["joint_names"] = list(retargeter.optimizer.robot.dof_joint_names)
    meta["joint_limits"] = np.asarray(retargeter.optimizer.robot.joint_limits, dtype=float).tolist()

    last = None
    t0 = time.perf_counter()
    for i in range(joints.shape[0]):
        if not valid[i]:
            if last is not None:
                out[i] = last
            continue
        try:
            qpos = np.asarray(retargeter.retarget(joints[i], apply_filter=True), dtype=np.float32).reshape(-1)
            if qpos.size != 20 or not np.all(np.isfinite(qpos)):
                raise ValueError(f"bad qpos shape={qpos.shape} finite={np.all(np.isfinite(qpos))}")
            out[i] = qpos
            last = qpos.copy()
        except Exception as exc:
            meta["failures"] += 1
            meta["last_error"] = repr(exc)
            if last is not None:
                out[i] = last
    meta["elapsed_s"] = time.perf_counter() - t0
    return out, meta

left_wuji, left_meta = solve_side("left", data["left_joints"], data["left_valid"], left_config)
right_wuji, right_meta = solve_side("right", data["right_joints"], data["right_valid"], right_config)
np.savez_compressed(
    output_npz,
    left_wuji=left_wuji,
    right_wuji=right_wuji,
    meta=np.array(json.dumps({"left": left_meta, "right": right_meta})),
)
"""
    result = subprocess.run(
        [str(python), "-c", script, str(root), str(left_config), str(right_config), str(input_npz), str(output_npz)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not output_npz.exists():
        reason = (
            f"Wuji retarget subprocess failed code={result.returncode}; "
            f"stdout={result.stdout.strip()!r}; stderr={result.stderr.strip()!r}"
        )
        return fallback(reason)

    out = np.load(output_npz, allow_pickle=False)
    meta = json.loads(str(out["meta"].item()))
    for side_meta in meta.values():
        side_meta["python"] = str(python)
        side_meta["root"] = str(root)
    return (
        np.asarray(out["left_wuji"], dtype=np.float32),
        np.asarray(out["right_wuji"], dtype=np.float32),
        meta["left"],
        meta["right"],
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    out = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )
    return quat_normalize(out)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    vq = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=np.float32), v.astype(np.float32)], axis=-1)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[..., 1:]


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def matrix_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float32)
    out = np.zeros(m.shape[:-2] + (4,), dtype=np.float32)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    positive = trace > 0.0
    s = np.sqrt(np.maximum(trace[positive] + 1.0, 1e-8)) * 2.0
    out[positive, 0] = 0.25 * s
    out[positive, 1] = (m[positive, 2, 1] - m[positive, 1, 2]) / s
    out[positive, 2] = (m[positive, 0, 2] - m[positive, 2, 0]) / s
    out[positive, 3] = (m[positive, 1, 0] - m[positive, 0, 1]) / s
    neg = ~positive
    if np.any(neg):
        out[neg] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat_normalize(out)


def heading_quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(np.asarray(q, dtype=np.float32))
    yaw = np.arctan2(2.0 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1.0 - 2.0 * (q[..., 2] ** 2 + q[..., 3] ** 2))
    half = -0.5 * yaw
    return np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=-1).astype(np.float32)


def root_anchor_6d(root_quat: np.ndarray, frame: int, base_frame: int | None = None) -> np.ndarray:
    target = min(max(frame, 0), root_quat.shape[0] - 1)
    base = target if base_frame is None else min(max(base_frame, 0), root_quat.shape[0] - 1)
    rel = quat_mul(quat_conj(root_quat[base]), root_quat[target])
    rot = quat_to_rotmat(rel)
    return rot[..., :2].reshape(-1).astype(np.float32)


def window_indices(t: int, total: int, num_frames: int, step: int) -> np.ndarray:
    idx = t + np.arange(num_frames, dtype=np.int64) * step
    return np.minimum(idx, total - 1)


def finite_diff(values: np.ndarray, fps: int = DEFAULT_FPS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] <= 1:
        return np.zeros_like(values)
    grad = np.zeros_like(values)
    grad[1:-1] = (values[2:] - values[:-2]) * (fps / 2.0)
    grad[0] = (values[1] - values[0]) * fps
    grad[-1] = (values[-1] - values[-2]) * fps
    return grad


def _h5_first_existing(h5: h5py.File, paths: Sequence[str]) -> tuple[str | None, np.ndarray | None]:
    path, value = first_h5_dataset(h5, paths)
    if value is None:
        return None, None
    return path, np.asarray(value)


def extract_smplx_params(h5: h5py.File, output_path: Path) -> dict[str, Any]:
    """Write SMPL/SMPL-X parameter npz when the HDF5 exposes compatible fields."""

    candidates = {
        "trans": (
            "full_body_mocap/trans",
            "full_body_mocap/transl",
            "full_body_mocap/root_trans",
            "full_body_mocap/smplx/trans",
            "full_body_mocap/smplx/transl",
        ),
        "root_orient": (
            "full_body_mocap/root_orient",
            "full_body_mocap/global_orient",
            "full_body_mocap/root_pose",
            "full_body_mocap/smplx/root_orient",
            "full_body_mocap/smplx/global_orient",
        ),
        "pose_body": (
            "full_body_mocap/pose_body",
            "full_body_mocap/body_pose",
            "full_body_mocap/smplx/pose_body",
            "full_body_mocap/smplx/body_pose",
        ),
        "betas": (
            "full_body_mocap/betas",
            "full_body_mocap/shape",
            "full_body_mocap/smplx/betas",
        ),
        "poses": (
            "full_body_mocap/poses",
            "full_body_mocap/smpl_poses",
            "full_body_mocap/smplx/poses",
        ),
    }

    found: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    for key, paths in candidates.items():
        source, value = _h5_first_existing(h5, paths)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        arr, _, _ = repair_framewise(arr)
        found[key] = arr
        sources[key] = str(source)

    if "poses" not in found:
        if "root_orient" in found and "pose_body" in found:
            root = found["root_orient"].reshape(found["root_orient"].shape[0], -1)
            body = found["pose_body"].reshape(found["pose_body"].shape[0], -1)
            found["poses"] = np.concatenate([root[:, :3], body], axis=1)
            sources["poses"] = "root_orient+pose_body"
        else:
            return {"ok": False, "message": "no SMPL/SMPL-X pose parameter fields found", "sources": sources}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"poses": found["poses"]}
    if "trans" in found:
        payload["trans"] = found["trans"].reshape(found["trans"].shape[0], -1)[:, :3]
        payload["transl"] = payload["trans"]
    if "betas" in found:
        betas = found["betas"]
        if betas.ndim > 1:
            betas = betas[0]
        payload["betas"] = betas.reshape(-1)[:16]
    payload["mocap_frame_rate"] = np.array(DEFAULT_FPS, dtype=np.float32)
    np.savez(output_path, **payload)
    return {"ok": True, "path": str(output_path), "sources": sources, "keys": sorted(payload)}


def extract_smpl_motion(h5: h5py.File) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    key_path, keypoints = first_h5_dataset(
        h5,
        (
            "full_body_mocap/keypoints",
            "full_body_mocap/joints",
            "full_body_mocap/smpl_joints",
        ),
    )
    if keypoints is None:
        smpl_joints = np.zeros((1, 24, 3), dtype=np.float32)
    else:
        kp = np.asarray(keypoints, dtype=np.float32)
        if kp.ndim == 2 and kp.shape[-1] % 3 == 0:
            kp = kp.reshape(kp.shape[0], kp.shape[-1] // 3, 3)
        if kp.ndim != 3 or kp.shape[-1] != 3:
            smpl_joints = np.zeros((max(1, kp.shape[0]), 24, 3), dtype=np.float32)
        else:
            kp, _, _ = repair_framewise(kp)
            if kp.shape[1] < 24:
                pad = np.zeros((kp.shape[0], 24 - kp.shape[1], 3), dtype=np.float32)
                smpl_joints = np.concatenate([kp, pad], axis=1)
            else:
                smpl_joints = kp[:, :24, :]

    t = int(smpl_joints.shape[0])
    ts = h5_read("full_body_mocap/Ts_world_root", h5)
    body_quats = h5_read("full_body_mocap/body_quats", h5)
    root_pos = smpl_joints[:, 0, :].copy()
    root_quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (t, 1))
    joint_quat = np.tile(np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32), (t, 24, 1))
    root_pose_source = "identity"
    joint_quat_source = "identity"
    if ts is not None:
        ts_arr = np.asarray(ts, dtype=np.float32)
        if ts_arr.ndim == 3 and ts_arr.shape[-2:] == (4, 4):
            root_pos = ts_arr[:t, :3, 3]
            root_quat = matrix_to_quat_wxyz(ts_arr[:t, :3, :3])
            root_pose_source = "full_body_mocap/Ts_world_root_matrix"
        elif ts_arr.ndim == 2 and ts_arr.shape[-1] == 7:
            n = min(t, ts_arr.shape[0])
            # Xperience stores root pose as [quat_wxyz(4), xyz(3)] in this sample.
            root_quat[:n] = quat_normalize(ts_arr[:n, :4])
            root_pos[:n] = ts_arr[:n, 4:7]
            root_pose_source = "full_body_mocap/Ts_world_root_quat_wxyz_xyz"

    if body_quats is not None:
        bq = np.asarray(body_quats, dtype=np.float32)
        if bq.ndim == 3 and bq.shape[-1] == 4:
            n = min(t, bq.shape[0])
            m = min(21, bq.shape[1], 23)
            repaired_quat, _, _ = repair_framewise(bq[:n, :m, :])
            repaired_quat = quat_normalize(repaired_quat)
            bad = ~np.isfinite(repaired_quat).all(axis=-1) | (
                np.linalg.norm(repaired_quat, axis=-1) < 1e-6
            )
            repaired_quat[bad] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            joint_quat[:n, 0, :] = root_quat[:n]
            for local_idx in range(m):
                joint_idx = local_idx + 1
                parent_idx = int(SMPL_24_PARENT_INDEX[joint_idx])
                parent_quat = joint_quat[:n, parent_idx, :]
                joint_quat[:n, joint_idx, :] = quat_mul(parent_quat, repaired_quat[:, local_idx, :])
            joint_quat_source = "full_body_mocap/body_quats_local_21_composed_to_global"
        elif bq.ndim == 2 and bq.shape[-1] == 4:
            root_quat = quat_normalize(bq[:t, :])
            bad_root = ~np.isfinite(root_quat).all(axis=-1) | (np.linalg.norm(root_quat, axis=-1) < 1e-6)
            root_quat[bad_root] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            joint_quat[:, 0, :] = root_quat
            root_pose_source = "full_body_mocap/body_quats_root"
            joint_quat_source = "root_only_from_body_quats"

    joint_quat[:, 0, :] = root_quat
    smpl_poses = np.zeros((t, 21, 3), dtype=np.float32)
    return {
        "smpl_joints": smpl_joints.astype(np.float32),
        "smpl_poses": smpl_poses,
        "root_pos": root_pos.astype(np.float32),
        "root_quat_wxyz": root_quat.astype(np.float32),
        "smpl_joint_quat_wxyz": joint_quat.astype(np.float32),
    }, {"keypoints_source": key_path, "root_pose_source": root_pose_source, "joint_quat_source": joint_quat_source}


def infer_xperience_ground_height(h5: h5py.File, smpl_joints: np.ndarray) -> tuple[float, str]:
    for path in ("ground_height", "floor_z", "metadata/ground_height", "metadata/floor_z"):
        value = h5_read(path, h5)
        if value is not None:
            ground = float(np.asarray(value).flat[0])
            if np.isfinite(ground):
                return ground, path
    for path in ("body_height", "metadata/body_height"):
        value = h5_read(path, h5)
        if value is not None:
            height = float(np.asarray(value).flat[0])
            if np.isfinite(height) and height > 0.0:
                return -height, path

    joints = np.asarray(smpl_joints, dtype=np.float32)
    body_z = joints[:, : min(22, joints.shape[1]), 2]
    finite = body_z[np.isfinite(body_z)]
    if finite.size:
        return float(np.percentile(finite, 1.0)), "smpl_body_z_percentile_1"
    return 0.0, "default_zero"


def normalize_smpl_ground_for_gmr(smpl: Mapping[str, np.ndarray], ground_height: float) -> dict[str, np.ndarray]:
    out = {k: np.asarray(v, dtype=np.float32).copy() for k, v in smpl.items()}
    out["smpl_joints"][..., 2] -= np.float32(ground_height)
    out["root_pos"][..., 2] -= np.float32(ground_height)
    return out


def _valid_savgol_window(num_frames: int, requested: int, polyorder: int) -> int:
    if num_frames <= polyorder + 1:
        return 0
    window = min(int(requested), int(num_frames))
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    if window > num_frames:
        window = num_frames if num_frames % 2 == 1 else num_frames - 1
    return window if window > polyorder else 0


def _numpy_savgol_axis0(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(values.shape[0], -1)
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    design = np.stack([x**order for order in range(polyorder + 1)], axis=1)
    coeff = np.linalg.pinv(design)[0].astype(np.float32)
    padded = np.pad(flat, ((half, half), (0, 0)), mode="edge")
    out = np.zeros_like(flat)
    for i, weight in enumerate(coeff):
        out += np.float32(weight) * padded[i : i + flat.shape[0]]
    return out.reshape(values.shape).astype(np.float32)


def smooth_axis0_light(values: np.ndarray, window: int, polyorder: int) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float32)
    actual_window = _valid_savgol_window(arr.shape[0], window, polyorder)
    if actual_window == 0:
        return arr.copy(), {"method": "none_too_few_frames", "window_frames": 0, "polyorder": polyorder}
    try:
        from scipy.signal import savgol_filter

        smoothed = savgol_filter(arr, actual_window, polyorder, axis=0, mode="interp")
        return smoothed.astype(np.float32), {
            "method": "scipy.signal.savgol_filter",
            "window_frames": actual_window,
            "polyorder": polyorder,
        }
    except Exception as exc:
        return _numpy_savgol_axis0(arr, actual_window, polyorder), {
            "method": "numpy_savgol_fallback",
            "window_frames": actual_window,
            "polyorder": polyorder,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }


def quat_sign_continuous(quat: np.ndarray) -> np.ndarray:
    q = quat_normalize(np.asarray(quat, dtype=np.float32)).reshape(quat.shape[0], -1, 4)
    out = q.copy()
    for t in range(1, out.shape[0]):
        dots = np.sum(out[t - 1] * out[t], axis=-1)
        out[t, dots < 0.0] *= -1.0
    return out.reshape(quat.shape)


def smooth_quat_axis0_light(quat: np.ndarray, window: int, polyorder: int) -> tuple[np.ndarray, dict[str, Any]]:
    continuous = quat_sign_continuous(quat)
    smoothed, meta = smooth_axis0_light(continuous, window, polyorder)
    return quat_normalize(smoothed).astype(np.float32), meta


def smooth_g1_after_gmr(
    g1: Mapping[str, np.ndarray],
    *,
    enabled: bool = GMR_POST_SMOOTH_ENABLED,
    window: int = GMR_POST_SMOOTH_WINDOW,
    polyorder: int = GMR_POST_SMOOTH_POLYORDER,
    fps: int = DEFAULT_FPS,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    out = {k: np.asarray(v, dtype=np.float32).copy() for k, v in g1.items()}
    meta: dict[str, Any] = {
        "enabled": bool(enabled),
        "requested_window_frames": int(window),
        "requested_window_sec": float(window / fps),
        "polyorder": int(polyorder),
        "fps": int(fps),
        "fields": [],
        "quaternion_policy": "sign-continuous component Savitzky-Golay, then normalize",
        "velocity_policy": "recompute dof_vel from smoothed dof_pos with centered finite differences",
    }
    if not enabled:
        meta["method"] = "disabled"
        return out, meta

    out["root_pos"], root_pos_meta = smooth_axis0_light(out["root_pos"], window, polyorder)
    out["root_quat_wxyz"], root_quat_meta = smooth_quat_axis0_light(
        out["root_quat_wxyz"], window, polyorder
    )
    out["dof_pos"], dof_pos_meta = smooth_axis0_light(out["dof_pos"], window, polyorder)
    out["dof_vel"] = finite_diff(out["dof_pos"], fps=fps)

    meta["fields"] = ["root_pos", "root_quat_wxyz", "dof_pos", "dof_vel"]
    meta["field_methods"] = {
        "root_pos": root_pos_meta,
        "root_quat_wxyz": root_quat_meta,
        "dof_pos": dof_pos_meta,
        "dof_vel": {"method": "finite_diff_from_smoothed_dof_pos"},
    }
    meta["actual_window_frames"] = int(dof_pos_meta.get("window_frames", 0))
    meta["actual_window_sec"] = float(meta["actual_window_frames"] / fps)
    meta["method"] = dof_pos_meta.get("method", "unknown")
    return out, meta


def estimate_human_height_from_joints(smpl_joints: np.ndarray) -> float:
    joints = np.asarray(smpl_joints, dtype=np.float32)
    if joints.size == 0:
        return 1.8
    z = joints[..., 2]
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return 1.8
    height = float(np.percentile(finite, 99.0) - np.percentile(finite, 1.0))
    return float(np.clip(height, 1.2, 2.2))


def gmr_frame_from_smpl_joints(smpl_joints: np.ndarray, joint_quat: np.ndarray, frame: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    joints = np.asarray(smpl_joints, dtype=np.float32)
    quats = np.asarray(joint_quat, dtype=np.float32)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body_name, idx in GMR_SMPLX_BODY_INDEX.items():
        pos = joints[frame, idx, :].astype(np.float32)
        quat = quat_normalize(quats[frame, idx, :].astype(np.float32))
        if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-6:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out[body_name] = (pos, quat)
    return out


def write_gmr_joint_motion_npz(smpl: Mapping[str, np.ndarray], output_path: Path) -> dict[str, Any]:
    smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)
    joint_quat = np.asarray(
        smpl.get(
            "smpl_joint_quat_wxyz",
            np.tile(np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32), (smpl_joints.shape[0], 24, 1)),
        ),
        dtype=np.float32,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        smpl_joints=smpl_joints,
        smpl_joint_quat_wxyz=joint_quat,
        smpl_24_names=np.asarray(SMPL_24_NAMES),
        gmr_body_index=np.asarray([f"{name}:{idx}" for name, idx in GMR_SMPLX_BODY_INDEX.items()]),
        estimated_human_height=np.array(estimate_human_height_from_joints(smpl_joints), dtype=np.float32),
        fps=np.array(DEFAULT_FPS, dtype=np.float32),
    )
    return {
        "ok": True,
        "path": str(output_path),
        "frames": int(smpl_joints.shape[0]),
        "joint_order": "SMPL_24",
        "orientation": "body_quats if present, otherwise identity quaternions",
    }


def _fallback_g1_motion(smpl: Mapping[str, np.ndarray], status: RetargetStatus) -> tuple[dict[str, np.ndarray], RetargetStatus]:
    t = int(smpl["smpl_joints"].shape[0])
    dof_pos = np.tile(DEFAULT_G1_QPOS.reshape(1, -1), (t, 1)).astype(np.float32)
    return {
        "root_pos": np.asarray(smpl["root_pos"], dtype=np.float32),
        "root_quat_wxyz": np.asarray(smpl["root_quat_wxyz"], dtype=np.float32),
        "dof_pos": dof_pos,
        "dof_vel": finite_diff(dof_pos),
    }, status


def write_g1_replay_pkl(g1: Mapping[str, np.ndarray], output_path: Path, fps: int = DEFAULT_FPS) -> dict[str, Any]:
    root_quat_wxyz = quat_normalize(np.asarray(g1["root_quat_wxyz"], dtype=np.float32))
    payload = {
        "fps": float(fps),
        "root_pos": np.asarray(g1["root_pos"], dtype=np.float32),
        "root_rot": root_quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32),
        "dof_pos": np.asarray(g1["dof_pos"], dtype=np.float32),
        "local_body_pos": None,
        "link_body_list": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    return {
        "path": str(output_path),
        "format": "GMR RobotMotionViewer pickle",
        "root_rot_order": "xyzw",
        "fps": float(fps),
    }


def _run_gmr_api(
    *,
    smplx_file: Path,
    gmr_root: Path | None,
    smplx_folder: Path | None,
    gmr_robot: str,
    save_path: Path,
) -> tuple[dict[str, np.ndarray], RetargetStatus]:
    added_paths: list[str] = []
    if gmr_root is not None:
        root = str(gmr_root.expanduser().resolve())
        for candidate in (root, str(Path(root) / "scripts")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
                added_paths.append(candidate)
    try:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR
        from general_motion_retargeting.utils.smpl import (
            get_smplx_data_offline_fast,
            load_smplx_file,
        )

        resolved_smplx_folder = smplx_folder.expanduser().resolve() if smplx_folder is not None else None
        if resolved_smplx_folder is None and gmr_root is not None:
            assets_root = gmr_root.expanduser().resolve() / "assets" / "body_models"
            resolved_smplx_folder = assets_root if assets_root.exists() else None
        if resolved_smplx_folder is None:
            raise FileNotFoundError(
                "GMR SMPL-X body models not found. Pass --gmr-root pointing to a GMR checkout "
                "with assets/body_models, or pass --smplx-folder."
            )

        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            str(smplx_file),
            resolved_smplx_folder,
        )
        smplx_frames, aligned_fps = get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=DEFAULT_FPS,
        )
        retarget = GMR(actual_human_height=actual_human_height, src_human="smplx", tgt_robot=gmr_robot)
        qpos_list = []
        for frame in smplx_frames:
            qpos = np.asarray(retarget.retarget(frame), dtype=np.float32).reshape(-1)
            qpos_list.append(qpos)
        if not qpos_list:
            raise RuntimeError("GMR produced no qpos frames")
        qpos_arr = np.stack(qpos_list, axis=0)
        if qpos_arr.shape[1] < 7 + 29:
            raise RuntimeError(f"GMR qpos dim {qpos_arr.shape[1]} is too small for G1")
        root_pos = qpos_arr[:, :3].astype(np.float32)
        root_quat = quat_normalize(qpos_arr[:, 3:7].astype(np.float32))
        dof_pos = qpos_arr[:, 7 : 7 + 29].astype(np.float32)
        payload = {
            "fps": np.array(aligned_fps, dtype=np.float32),
            "root_pos": root_pos,
            "root_quat_wxyz": root_quat,
            "dof_pos": dof_pos,
            "dof_vel": finite_diff(dof_pos, fps=int(aligned_fps) if aligned_fps else DEFAULT_FPS),
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **payload)
        return payload, RetargetStatus(
            method="gmr_api_smplx_to_unitree_g1",
            ok=True,
            message=f"GMR retargeted {len(qpos_list)} frames with robot={gmr_robot}",
            artifact=str(save_path),
        )
    finally:
        for candidate in added_paths:
            try:
                sys.path.remove(candidate)
            except ValueError:
                pass


def _run_gmr_subprocess(
    *,
    gmr_python: Path,
    smplx_file: Path,
    gmr_root: Path | None,
    smplx_folder: Path | None,
    gmr_robot: str,
    save_path: Path,
) -> tuple[dict[str, np.ndarray], RetargetStatus]:
    gmr_python = gmr_python.expanduser().resolve()
    if not gmr_python.exists():
        raise FileNotFoundError(f"GMR python not found: {gmr_python}")
    root_expr = repr(str(gmr_root.expanduser().resolve())) if gmr_root is not None else "None"
    smplx_expr = repr(str(smplx_folder.expanduser().resolve())) if smplx_folder is not None else "None"
    helper = f"""
from pathlib import Path
import sys
import numpy as np

gmr_root = {root_expr}
smplx_folder = {smplx_expr}
if gmr_root:
    sys.path.insert(0, gmr_root)
    sys.path.insert(0, str(Path(gmr_root) / "scripts"))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

if smplx_folder is None:
    if gmr_root is None:
        raise FileNotFoundError("no --gmr-root/--smplx-folder provided")
    candidate = Path(gmr_root) / "assets" / "body_models"
    if not candidate.exists():
        raise FileNotFoundError(f"SMPL-X body model folder not found: {{candidate}}")
    smplx_folder = str(candidate)

smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
    {str(smplx_file)!r},
    Path(smplx_folder),
)
frames, aligned_fps = get_smplx_data_offline_fast(
    smplx_data,
    body_model,
    smplx_output,
    tgt_fps={DEFAULT_FPS},
)
retarget = GMR(actual_human_height=actual_human_height, src_human="smplx", tgt_robot={gmr_robot!r})
qpos = []
for frame in frames:
    qpos.append(np.asarray(retarget.retarget(frame), dtype=np.float32).reshape(-1))
if not qpos:
    raise RuntimeError("GMR produced no qpos frames")
qpos = np.stack(qpos, axis=0)
if qpos.shape[1] < 36:
    raise RuntimeError(f"GMR qpos dim {{qpos.shape[1]}} is too small for G1")
root_pos = qpos[:, :3].astype(np.float32)
root_quat_wxyz = qpos[:, 3:7].astype(np.float32)
root_quat_wxyz = root_quat_wxyz / np.maximum(np.linalg.norm(root_quat_wxyz, axis=-1, keepdims=True), 1e-8)
dof_pos = qpos[:, 7:36].astype(np.float32)
if dof_pos.shape[0] <= 1:
    dof_vel = np.zeros_like(dof_pos)
else:
    fps = int(aligned_fps) if aligned_fps else {DEFAULT_FPS}
    dof_vel = np.zeros_like(dof_pos)
    dof_vel[1:-1] = (dof_pos[2:] - dof_pos[:-2]) * (fps / 2.0)
    dof_vel[0] = (dof_pos[1] - dof_pos[0]) * fps
    dof_vel[-1] = (dof_pos[-1] - dof_pos[-2]) * fps
Path({str(save_path)!r}).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    {str(save_path)!r},
    fps=np.array(aligned_fps, dtype=np.float32),
    root_pos=root_pos,
    root_quat_wxyz=root_quat_wxyz,
    dof_pos=dof_pos,
    dof_vel=dof_vel,
)
print(f"saved {{qpos.shape[0]}} frames to {str(save_path)!r}")
"""
    result = subprocess.run(
        [str(gmr_python), "-c", helper],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GMR subprocess failed ({result.returncode}): {result.stderr.strip()}")
    data = np.load(save_path)
    motion = {
        "root_pos": np.asarray(data["root_pos"], dtype=np.float32),
        "root_quat_wxyz": np.asarray(data["root_quat_wxyz"], dtype=np.float32),
        "dof_pos": np.asarray(data["dof_pos"], dtype=np.float32),
        "dof_vel": np.asarray(data["dof_vel"], dtype=np.float32),
    }
    return motion, RetargetStatus(
        method="gmr_subprocess_smplx_to_unitree_g1",
        ok=True,
        message=result.stdout.strip() or f"GMR retargeted {motion['dof_pos'].shape[0]} frames",
        artifact=str(save_path),
    )


def _motion_from_gmr_qpos(qpos_arr: np.ndarray, fps: int = DEFAULT_FPS) -> dict[str, np.ndarray]:
    qpos_arr = np.asarray(qpos_arr, dtype=np.float32)
    if qpos_arr.ndim != 2 or qpos_arr.shape[1] < 7 + 29:
        raise RuntimeError(f"GMR qpos shape {qpos_arr.shape} is too small for G1")
    root_pos = qpos_arr[:, :3].astype(np.float32)
    root_quat = quat_normalize(qpos_arr[:, 3:7].astype(np.float32))
    dof_pos = qpos_arr[:, 7 : 7 + 29].astype(np.float32)
    return {
        "root_pos": root_pos,
        "root_quat_wxyz": root_quat,
        "dof_pos": dof_pos,
        "dof_vel": finite_diff(dof_pos, fps=fps),
    }


def _run_gmr_joints_api(
    *,
    smpl: Mapping[str, np.ndarray],
    gmr_root: Path | None,
    gmr_robot: str,
    save_path: Path,
) -> tuple[dict[str, np.ndarray], RetargetStatus]:
    added_paths: list[str] = []
    if gmr_root is not None:
        root = str(gmr_root.expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
            added_paths.append(root)
    try:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR

        smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)
        joint_quat = np.asarray(smpl["smpl_joint_quat_wxyz"], dtype=np.float32)
        retarget = GMR(
            actual_human_height=estimate_human_height_from_joints(smpl_joints),
            src_human="smplx",
            tgt_robot=gmr_robot,
            verbose=False,
        )
        qpos = []
        for i in range(smpl_joints.shape[0]):
            frame = gmr_frame_from_smpl_joints(smpl_joints, joint_quat, i)
            qpos.append(np.asarray(retarget.retarget(frame), dtype=np.float32).reshape(-1))
        if not qpos:
            raise RuntimeError("GMR produced no qpos frames")
        motion = _motion_from_gmr_qpos(np.stack(qpos, axis=0), fps=DEFAULT_FPS)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, fps=np.array(DEFAULT_FPS, dtype=np.float32), **motion)
        return motion, RetargetStatus(
            method="gmr_api_smpl24_joints_to_unitree_g1",
            ok=True,
            message=f"GMR retargeted {motion['dof_pos'].shape[0]} joint frames with robot={gmr_robot}",
            artifact=str(save_path),
        )
    finally:
        for candidate in added_paths:
            try:
                sys.path.remove(candidate)
            except ValueError:
                pass


def _run_gmr_joints_subprocess(
    *,
    gmr_python: Path,
    joint_motion_file: Path,
    gmr_root: Path | None,
    gmr_robot: str,
    save_path: Path,
) -> tuple[dict[str, np.ndarray], RetargetStatus]:
    gmr_python = gmr_python.expanduser().resolve()
    if not gmr_python.exists():
        raise FileNotFoundError(f"GMR python not found: {gmr_python}")
    root_expr = repr(str(gmr_root.expanduser().resolve())) if gmr_root is not None else "None"
    body_index_expr = repr(GMR_SMPLX_BODY_INDEX)
    helper = f"""
from pathlib import Path
import sys
import numpy as np

gmr_root = {root_expr}
if gmr_root:
    sys.path.insert(0, gmr_root)

from general_motion_retargeting import GeneralMotionRetargeting as GMR

body_index = {body_index_expr}
data = np.load({str(joint_motion_file)!r}, allow_pickle=False)
smpl_joints = data["smpl_joints"].astype(np.float32)
if "smpl_joint_quat_wxyz" in data:
    joint_quat = data["smpl_joint_quat_wxyz"].astype(np.float32)
else:
    joint_quat = np.tile(np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32), (smpl_joints.shape[0], 24, 1))
height = float(data["estimated_human_height"]) if "estimated_human_height" in data else 1.8

def normalize_quat(q):
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if (not np.isfinite(q).all()) or norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm

retarget = GMR(actual_human_height=height, src_human="smplx", tgt_robot={gmr_robot!r}, verbose=False)
qpos = []
for frame_idx in range(smpl_joints.shape[0]):
    if frame_idx % 500 == 0:
        print(f"GMR joint retarget frame {{frame_idx}}/{{smpl_joints.shape[0]}}", flush=True)
    frame = {{}}
    for body_name, joint_idx in body_index.items():
        frame[body_name] = (
            smpl_joints[frame_idx, joint_idx, :].astype(np.float32),
            normalize_quat(joint_quat[frame_idx, joint_idx, :]),
        )
    qpos.append(np.asarray(retarget.retarget(frame), dtype=np.float32).reshape(-1))
if not qpos:
    raise RuntimeError("GMR produced no qpos frames")
qpos = np.stack(qpos, axis=0)
if qpos.shape[1] < 36:
    raise RuntimeError(f"GMR qpos dim {{qpos.shape[1]}} is too small for G1")
root_pos = qpos[:, :3].astype(np.float32)
root_quat_wxyz = qpos[:, 3:7].astype(np.float32)
root_quat_wxyz = root_quat_wxyz / np.maximum(np.linalg.norm(root_quat_wxyz, axis=-1, keepdims=True), 1e-8)
dof_pos = qpos[:, 7:36].astype(np.float32)
if dof_pos.shape[0] <= 1:
    dof_vel = np.zeros_like(dof_pos)
else:
    dof_vel = np.zeros_like(dof_pos)
    dof_vel[1:-1] = (dof_pos[2:] - dof_pos[:-2]) * ({DEFAULT_FPS} / 2.0)
    dof_vel[0] = (dof_pos[1] - dof_pos[0]) * {DEFAULT_FPS}
    dof_vel[-1] = (dof_pos[-1] - dof_pos[-2]) * {DEFAULT_FPS}
Path({str(save_path)!r}).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    {str(save_path)!r},
    fps=np.array({DEFAULT_FPS}, dtype=np.float32),
    root_pos=root_pos,
    root_quat_wxyz=root_quat_wxyz,
    dof_pos=dof_pos,
    dof_vel=dof_vel,
)
print(f"saved {{qpos.shape[0]}} joint frames to {str(save_path)!r}")
"""
    result = subprocess.run(
        [str(gmr_python), "-u", "-c", helper],
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GMR joint subprocess failed ({result.returncode}): {result.stderr.strip()}")
    data = np.load(save_path)
    motion = {
        "root_pos": np.asarray(data["root_pos"], dtype=np.float32),
        "root_quat_wxyz": np.asarray(data["root_quat_wxyz"], dtype=np.float32),
        "dof_pos": np.asarray(data["dof_pos"], dtype=np.float32),
        "dof_vel": np.asarray(data["dof_vel"], dtype=np.float32),
    }
    return motion, RetargetStatus(
        method="gmr_subprocess_smpl24_joints_to_unitree_g1",
        ok=True,
        message=f"GMR retargeted {motion['dof_pos'].shape[0]} joint frames",
        artifact=str(save_path),
    )


def retarget_g1_minimal(
    smpl: Mapping[str, np.ndarray],
    *,
    require_gmr: bool = False,
    smplx_file: Path | None = None,
    joint_motion_file: Path | None = None,
    gmr_root: Path | None = None,
    gmr_python: Path | None = None,
    smplx_folder: Path | None = None,
    gmr_robot: str = "unitree_g1",
    gmr_output: Path | None = None,
) -> tuple[dict[str, np.ndarray], RetargetStatus]:
    """Return G1 motion arrays, using GMR when SMPL-X params and GMR are available."""

    if smplx_file is not None and smplx_file.exists():
        try:
            if gmr_python is not None:
                return _run_gmr_subprocess(
                    gmr_python=gmr_python,
                    smplx_file=smplx_file,
                    gmr_root=gmr_root,
                    smplx_folder=smplx_folder,
                    gmr_robot=gmr_robot,
                    save_path=gmr_output or smplx_file.with_suffix(".g1_gmr.npz"),
                )
            return _run_gmr_api(
                smplx_file=smplx_file,
                gmr_root=gmr_root,
                smplx_folder=smplx_folder,
                gmr_robot=gmr_robot,
                save_path=gmr_output or smplx_file.with_suffix(".g1_gmr.npz"),
            )
        except Exception as exc:
            if require_gmr:
                raise
            return _fallback_g1_motion(
                smpl,
                RetargetStatus(
                    method="default_g1_pose_fallback_after_gmr_error",
                    ok=False,
                    message=f"GMR retarget failed: {exc}",
                    artifact=str(smplx_file),
                ),
            )

    if joint_motion_file is not None and joint_motion_file.exists():
        try:
            if gmr_python is not None:
                return _run_gmr_joints_subprocess(
                    gmr_python=gmr_python,
                    joint_motion_file=joint_motion_file,
                    gmr_root=gmr_root,
                    gmr_robot=gmr_robot,
                    save_path=gmr_output or joint_motion_file.with_suffix(".g1_gmr.npz"),
                )
            return _run_gmr_joints_api(
                smpl=smpl,
                gmr_root=gmr_root,
                gmr_robot=gmr_robot,
                save_path=gmr_output or joint_motion_file.with_suffix(".g1_gmr.npz"),
            )
        except Exception as exc:
            if require_gmr:
                raise
            return _fallback_g1_motion(
                smpl,
                RetargetStatus(
                    method="default_g1_pose_fallback_after_gmr_joint_error",
                    ok=False,
                    message=f"GMR joint retarget failed: {exc}",
                    artifact=str(joint_motion_file),
                ),
            )

    status = RetargetStatus(
        method="default_g1_pose_fallback_no_gmr_input",
        ok=False,
        message="No compatible SMPL/SMPL-X params or SMPL-24 joint motion was available, so GMR was not called.",
        artifact=None,
    )
    if require_gmr:
        raise RuntimeError(status.message)
    return _fallback_g1_motion(smpl, status)


def build_sonic_encoder_obs(
    g1: Mapping[str, np.ndarray],
    smpl: Mapping[str, np.ndarray],
    *,
    encoder_mode: int = 0,
) -> np.ndarray:
    dof_pos_mujoco = np.asarray(g1["dof_pos"], dtype=np.float32)
    dof_vel_mujoco = np.asarray(g1["dof_vel"], dtype=np.float32)
    dof_pos = dof_pos_mujoco[:, G1_MUJOCO_TO_ISAACLAB_DOF]
    dof_vel = dof_vel_mujoco[:, G1_MUJOCO_TO_ISAACLAB_DOF]
    root_pos = np.asarray(g1["root_pos"], dtype=np.float32)
    root_quat = np.asarray(g1["root_quat_wxyz"], dtype=np.float32)
    smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)
    total = int(dof_pos.shape[0])
    obs = np.zeros((total, SONIC_ENCODER_OBS_DIM), dtype=np.float32)
    g1_step = max(1, int(round(SONIC_G1_FUTURE_DT * DEFAULT_FPS)))
    smpl_step = max(1, int(round(SONIC_SMPL_FUTURE_DT * DEFAULT_FPS)))

    for t in range(total):
        parts: list[np.ndarray] = []
        parts.append(np.array([float(encoder_mode), 0.0, 0.0, 0.0], dtype=np.float32))

        idx = window_indices(t, total, 10, g1_step)
        parts.append(dof_pos[idx].reshape(-1))
        parts.append(dof_vel[idx].reshape(-1))
        if encoder_mode == 0:
            # SONIC deploy g1 mode only gathers joint pos/vel and multi-frame anchor orientation.
            # Other encoder observations are left zero, matching the mode-filtered deploy path.
            parts.append(np.zeros(10, dtype=np.float32))
            parts.append(np.zeros(1, dtype=np.float32))
            parts.append(np.zeros(6, dtype=np.float32))
        else:
            parts.append(root_pos[idx, 2].reshape(-1))
            parts.append(root_pos[t : t + 1, 2].reshape(-1) if t < total else root_pos[-1:, 2].reshape(-1))
            parts.append(root_anchor_6d(root_quat, t, t))
        parts.append(np.concatenate([root_anchor_6d(root_quat, int(i), t) for i in idx], axis=0))
        if encoder_mode == 1:
            parts.append(dof_pos[idx][:, LOWER_BODY_ISAAC_INDEX].reshape(-1))
            parts.append(dof_vel[idx][:, LOWER_BODY_ISAAC_INDEX].reshape(-1))
        else:
            parts.append(np.zeros(120, dtype=np.float32))
            parts.append(np.zeros(120, dtype=np.float32))
        parts.append(np.zeros(9, dtype=np.float32))
        parts.append(np.zeros(12, dtype=np.float32))

        idx_smpl = window_indices(t, smpl_joints.shape[0], 10, smpl_step)
        sj = smpl_joints[idx_smpl]
        if sj.shape[1] < 24:
            pad = np.zeros((sj.shape[0], 24 - sj.shape[1], 3), dtype=np.float32)
            sj = np.concatenate([sj, pad], axis=1)
        if encoder_mode == 2:
            parts.append(sj[:, :24, :].reshape(-1))
            idx_smpl_g1 = window_indices(t, total, 10, smpl_step)
            parts.append(np.concatenate([root_anchor_6d(root_quat, int(i), t) for i in idx_smpl_g1], axis=0))
            parts.append(dof_pos[idx_smpl_g1][:, WRIST_ISAAC_INDEX].reshape(-1))
        else:
            parts.append(np.zeros(720, dtype=np.float32))
            parts.append(np.zeros(60, dtype=np.float32))
            parts.append(np.zeros(60, dtype=np.float32))

        row = np.concatenate(parts, axis=0)
        if row.shape[0] != SONIC_ENCODER_OBS_DIM:
            raise ValueError(f"SONIC encoder obs dim {row.shape[0]}, expected {SONIC_ENCODER_OBS_DIM}")
        obs[t] = row
    return obs


def encode_sonic_tokens(encoder_obs: np.ndarray, encoder_model: Path | None) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode tokens with SONIC's exported deploy encoder.

    Parent SONIC code path:
      gear_sonic.trl.modules.universal_token_modules.UniversalTokenModule.encode()
        -> _encode_single()
        -> FSQ quantizer
      gear_sonic.utils.inference_helpers.export_universal_token_encoders_as_onnx()
        exports input obs_dict and output encoded_tokens
      gear_sonic_deploy EncoderEngine::Encode()
        runs the same ONNX/TensorRT contract.
    """

    encoder_obs = np.asarray(encoder_obs, dtype=np.float32)
    if encoder_model is None:
        return np.zeros((encoder_obs.shape[0], TOKEN_DIM), dtype=np.float32), {
            "ok": False,
            "method": "zeros_no_encoder_model",
            "sonic_contract": "obs_dict[1,1762] -> encoded_tokens[1,64]",
        }
    try:
        import onnxruntime as ort
    except Exception as exc:
        return np.zeros((encoder_obs.shape[0], TOKEN_DIM), dtype=np.float32), {
            "ok": False,
            "method": "zeros_onnxruntime_missing_for_sonic_encoder_onnx",
            "message": str(exc),
            "encoder_model": str(encoder_model),
            "sonic_contract": "obs_dict[1,1762] -> encoded_tokens[1,64]",
        }
    session = ort.InferenceSession(str(encoder_model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = list(session.get_inputs()[0].shape)
    output_shape = list(session.get_outputs()[0].shape)
    if encoder_obs.ndim != 2 or encoder_obs.shape[1] != SONIC_ENCODER_OBS_DIM:
        raise ValueError(f"encoder_obs shape {encoder_obs.shape}, expected [T,{SONIC_ENCODER_OBS_DIM}]")
    tokens: list[np.ndarray] = []
    fixed_batch_one = input_shape and input_shape[0] == 1
    if fixed_batch_one:
        for row in encoder_obs:
            tokens.append(session.run([output_name], {input_name: row.reshape(1, -1)})[0])
    else:
        for start in range(0, encoder_obs.shape[0], 512):
            batch = encoder_obs[start : start + 512].astype(np.float32, copy=False)
            tokens.append(session.run([output_name], {input_name: batch})[0])
    arr = np.concatenate(tokens, axis=0).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != TOKEN_DIM:
        raise ValueError(f"encoded token shape {arr.shape}, expected [T,{TOKEN_DIM}]")
    return arr, {
        "ok": True,
        "method": "onnxruntime_sonic_exported_encoder",
        "encoder_model": str(encoder_model),
        "input_name": input_name,
        "input_shape": input_shape,
        "output_name": output_name,
        "output_shape": output_shape,
        "sonic_contract": "obs_dict[1,1762] -> encoded_tokens[1,64]",
        "source_code_reference": [
            "gear_sonic/trl/modules/universal_token_modules.py:encode",
            "gear_sonic/utils/inference_helpers.py:export_universal_token_encoders_as_onnx",
            "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/encoder.hpp:EncoderEngine::Encode",
        ],
    }


def snap_token_grid(tokens: np.ndarray) -> np.ndarray:
    return (np.round(tokens / FSQ_GRID_STEP) * FSQ_GRID_STEP).astype(np.float32)


def build_state_action(
    *,
    g1: Mapping[str, np.ndarray],
    hand_action: np.ndarray,
    tokens: np.ndarray,
    imu: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    total = min(int(g1["dof_pos"].shape[0]), int(hand_action.shape[0]), int(tokens.shape[0]))
    if total < 2:
        raise ValueError("need at least 2 synchronized frames for state/action rows")

    root_quat = np.asarray(g1["root_quat_wxyz"], dtype=np.float32)[:total]
    gravity_world = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (total, 1))
    base_gravity = quat_rotate(quat_conj(root_quat), gravity_world)
    base_ang_vel = np.zeros((total, 3), dtype=np.float32)
    base_accel = np.zeros((total, 3), dtype=np.float32)
    gyro = imu.get("gyro_xyz")
    accel = imu.get("accel_xyz")
    if gyro is not None and np.asarray(gyro).ndim == 2 and np.asarray(gyro).shape[1] >= 3:
        n = min(total, np.asarray(gyro).shape[0])
        base_ang_vel[:n] = np.asarray(gyro, dtype=np.float32)[:n, :3]
    if accel is not None and np.asarray(accel).ndim == 2 and np.asarray(accel).shape[1] >= 3:
        n = min(total, np.asarray(accel).shape[0])
        base_accel[:n] = np.asarray(accel, dtype=np.float32)[:n, :3]

    states = np.concatenate(
        [
            base_gravity,
            base_ang_vel,
            base_accel,
            np.asarray(g1["dof_pos"], dtype=np.float32)[:total],
            np.asarray(g1["dof_vel"], dtype=np.float32)[:total],
            hand_action[:total],
        ],
        axis=1,
    )
    if states.shape[1] != STATE_DIM:
        raise ValueError(f"state dim {states.shape[1]}, expected {STATE_DIM}")

    actions = np.concatenate([tokens[:total], hand_action[:total]], axis=1)
    if actions.shape[1] != ACTION_DIM:
        raise ValueError(f"action dim {actions.shape[1]}, expected {ACTION_DIM}")

    rows = []
    for i in range(total - 1):
        rows.append(
            {
                "states": states[i].astype(np.float32).tolist(),
                "action": actions[i + 1].astype(np.float32).tolist(),
                "timestamp": i / float(DEFAULT_FPS),
                "frame_index": i,
                "next.done": i == total - 2,
            }
        )
    return pd.DataFrame(rows)


def read_imu(h5: h5py.File) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in ("accel_xyz", "gyro_xyz", "device_timestamp_ns", "keyframe_indices"):
        path = f"imu/{name}"
        if path in h5:
            out[name] = np.asarray(h5[path][()])
    return out


def copy_selected_hdf5(src_path: Path, dst_path: Path) -> None:
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key in src.keys():
            if key in DROP_HDF5_GROUPS:
                continue
            src.copy(key, dst, name=key)
        dst.attrs["xperience_minimal_drop_groups"] = ",".join(sorted(DROP_HDF5_GROUPS))


def materialize_video(src: Path, dst: Path, *, resize: bool) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if resize and shutil.which("ffmpeg"):
        if dst.exists():
            dst.unlink()
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-vf",
            f"scale={IMAGE_WIDTH}:{IMAGE_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={IMAGE_WIDTH}:{IMAGE_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
            "-an",
            "-vsync",
            "0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0:
            return {"path": str(dst), "resized": True, "height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
        print(f"[WARN] ffmpeg resize failed for {src}: {result.stderr.strip()}")
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return {"path": str(dst), "resized": False, "mode": "copy"}


def summarize_hdf5(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "datasets": {}, "groups": []}
    with h5py.File(path, "r") as h5:
        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                summary["datasets"][name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": h5_attrs_to_dict(obj),
                }
            else:
                summary["groups"].append(name)

        h5.visititems(visitor)
    return summary


def inspect_hdf5(args: argparse.Namespace) -> int:
    path = args.annotation.expanduser().resolve()
    summary = summarize_hdf5(path)
    with h5py.File(path, "r") as h5:
        left_path, left = first_h5_dataset(h5, ("hand_mocap/left_joints_3d", "hand_mocap/left_joints"))
        right_path, right = first_h5_dataset(h5, ("hand_mocap/right_joints_3d", "hand_mocap/right_joints"))
        _, left_pose = first_h5_dataset(h5, ("hand_mocap/left_mano_hand_pose",))
        _, right_pose = first_h5_dataset(h5, ("hand_mocap/right_mano_hand_pose",))
        reports = [
            asdict(
                hand_report(
                    "left",
                    None if left is None else np.asarray(left),
                    left_path,
                    mano_pose=None if left_pose is None else np.asarray(left_pose),
                )
            ),
            asdict(
                hand_report(
                    "right",
                    None if right is None else np.asarray(right),
                    right_path,
                    mano_pose=None if right_pose is None else np.asarray(right_pose),
                )
            ),
        ]
    payload = {"hdf5": summary, "hand_validity": reports}
    if args.output:
        write_json(args.output.expanduser().resolve(), payload)
    else:
        print(json.dumps(payload, indent=2, default=_json_default))
    return 0


def convert_episode_dir(
    episode_dir: Path,
    output_dir: Path,
    *,
    resize_videos: bool,
    require_gmr: bool,
    gmr_root: Path | None,
    gmr_python: Path | None,
    smplx_folder: Path | None,
    gmr_robot: str,
    wuji_root: Path | None,
    wuji_python: Path | None,
    wuji_left_config: Path | None,
    wuji_right_config: Path | None,
    require_wuji: bool,
    encoder_model: Path | None,
    min_free_gb: float,
    overwrite: bool,
    keep_intermediate: bool,
) -> dict[str, Any]:
    episode_dir = episode_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    annotation = episode_dir / "annotation.hdf5"
    if not annotation.exists():
        raise FileNotFoundError(f"missing annotation.hdf5 in {episode_dir}")
    for filename in ("stereo_left.mp4", "stereo_right.mp4"):
        if not (episode_dir / filename).exists():
            raise FileNotFoundError(f"missing {filename} in {episode_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_space(output_dir, min_free_gb)

    video_info = {
        "stereo_left": materialize_video(episode_dir / "stereo_left.mp4", output_dir / "stereo_left.mp4", resize=resize_videos),
        "stereo_right": materialize_video(episode_dir / "stereo_right.mp4", output_dir / "stereo_right.mp4", resize=resize_videos),
    }

    minimal_h5 = output_dir / "annotation_minimal.hdf5"
    copy_selected_hdf5(annotation, minimal_h5)

    with h5py.File(annotation, "r") as h5:
        caption_payload = extract_caption_payload(h5)
        compact_metadata = extract_compact_metadata(h5, caption_payload)
        left_path, left_joints = first_h5_dataset(h5, ("hand_mocap/left_joints_3d", "hand_mocap/left_joints"))
        right_path, right_joints = first_h5_dataset(h5, ("hand_mocap/right_joints_3d", "hand_mocap/right_joints"))
        _, left_pose = first_h5_dataset(h5, ("hand_mocap/left_mano_hand_pose",))
        _, right_pose = first_h5_dataset(h5, ("hand_mocap/right_mano_hand_pose",))
        left_joints_arr = None if left_joints is None else np.asarray(left_joints)
        right_joints_arr = None if right_joints is None else np.asarray(right_joints)
        left_pose_arr = None if left_pose is None else np.asarray(left_pose)
        right_pose_arr = None if right_pose is None else np.asarray(right_pose)
        left_report = hand_report("left", left_joints_arr, left_path, mano_pose=left_pose_arr)
        right_report = hand_report("right", right_joints_arr, right_path, mano_pose=right_pose_arr)
        left_wuji, right_wuji, left_method, right_method = retarget_wuji_subprocess(
            left_joints=left_joints_arr,
            right_joints=right_joints_arr,
            left_pose=left_pose_arr,
            right_pose=right_pose_arr,
            output_dir=output_dir,
            wuji_root=wuji_root,
            wuji_python=wuji_python,
            wuji_left_config=wuji_left_config,
            wuji_right_config=wuji_right_config,
            require_wuji=require_wuji,
        )
        smpl, smpl_meta = extract_smpl_motion(h5)
        ground_height, ground_source = infer_xperience_ground_height(h5, smpl["smpl_joints"])
        smpl_meta["xperience_ground_height"] = ground_height
        smpl_meta["xperience_ground_height_source"] = ground_source
        smpl_meta["gmr_coordinate_policy"] = (
            "Xperience body keypoints are already in a z-up world frame; for GMR only translate z by "
            "-ground_height so the floor is z=0."
        )
        smplx_param_info = extract_smplx_params(h5, output_dir / "gmr_input_smplx_params.npz")
        smpl_meta["smplx_params"] = smplx_param_info
        imu = read_imu(h5)

    total = max(
        int(smpl["smpl_joints"].shape[0]),
        int(left_wuji.shape[0]) if left_wuji.size else 0,
        int(right_wuji.shape[0]) if right_wuji.size else 0,
        1,
    )
    if left_wuji.shape[0] == 0:
        left_wuji = np.zeros((total, WUJI_QPOS_DIM), dtype=np.float32)
    if right_wuji.shape[0] == 0:
        right_wuji = np.zeros((total, WUJI_QPOS_DIM), dtype=np.float32)
    total = min(total, left_wuji.shape[0], right_wuji.shape[0], smpl["smpl_joints"].shape[0])
    left_wuji = left_wuji[:total]
    right_wuji = right_wuji[:total]
    hand_action = np.concatenate([left_wuji, right_wuji], axis=1).astype(np.float32)
    smpl = {k: np.asarray(v, dtype=np.float32)[:total] for k, v in smpl.items()}
    smpl_for_gmr = normalize_smpl_ground_for_gmr(smpl, float(smpl_meta["xperience_ground_height"]))
    joint_motion_file = output_dir / "gmr_input_smpl24_joints.npz"
    smpl_meta["gmr_joint_input"] = write_gmr_joint_motion_npz(smpl_for_gmr, joint_motion_file)
    smplx_file = output_dir / "gmr_input_smplx_params.npz"
    if not smplx_file.exists():
        smplx_file = None
    g1_raw, g1_status = retarget_g1_minimal(
        smpl_for_gmr,
        require_gmr=require_gmr,
        smplx_file=smplx_file,
        joint_motion_file=joint_motion_file,
        gmr_root=gmr_root,
        gmr_python=gmr_python,
        smplx_folder=smplx_folder,
        gmr_robot=gmr_robot,
        gmr_output=output_dir / "g1_motion_gmr_raw.npz",
    )
    g1_raw = {k: np.asarray(v, dtype=np.float32)[:total] for k, v in g1_raw.items()}
    g1, gmr_post_smoothing = smooth_g1_after_gmr(g1_raw, fps=DEFAULT_FPS)
    if keep_intermediate:
        np.savez_compressed(
            output_dir / "g1_motion_gmr.npz",
            fps=np.array(DEFAULT_FPS, dtype=np.float32),
            root_pos=g1["root_pos"],
            root_quat_wxyz=g1["root_quat_wxyz"],
            dof_pos=g1["dof_pos"],
            dof_vel=g1["dof_vel"],
        )

    sonic_encoder_mode = 0
    encoder_obs = build_sonic_encoder_obs(g1, smpl, encoder_mode=sonic_encoder_mode)
    tokens, token_status = encode_sonic_tokens(encoder_obs, encoder_model)
    tokens = snap_token_grid(tokens)
    frames = build_state_action(g1=g1, hand_action=hand_action, tokens=tokens, imu=imu)

    if keep_intermediate:
        np.save(output_dir / "wuji_hand_action_40d.npy", hand_action)
        np.save(output_dir / "sonic_encoder_obs_1762d.npy", encoder_obs)
        np.save(output_dir / "sonic_action_token_64d.npy", tokens)
        replay_info = write_g1_replay_pkl(g1, output_dir / "g1_motion_gmr_replay.pkl", fps=DEFAULT_FPS)
        np.savez_compressed(
            output_dir / "g1_motion_minimal.npz",
            fps=np.array(DEFAULT_FPS, dtype=np.float32),
            root_pos=g1["root_pos"],
            root_quat_wxyz=g1["root_quat_wxyz"],
            dof_pos=g1["dof_pos"],
            dof_vel=g1["dof_vel"],
        )
        np.savez_compressed(
            output_dir / "smpl_motion_minimal.npz",
            fps=np.array(DEFAULT_FPS, dtype=np.float32),
            smpl_joints=smpl["smpl_joints"],
            smpl_poses=smpl["smpl_poses"],
            root_pos=smpl["root_pos"],
            root_quat_wxyz=smpl["root_quat_wxyz"],
            smpl_joint_quat_wxyz=smpl["smpl_joint_quat_wxyz"],
        )
    else:
        replay_info = {"kept": False, "reason": "KEEP_INTERMEDIATE=false"}
    frames.to_parquet(output_dir / "state_action_107_104.parquet", index=False)

    report = {
        "source_episode_dir": str(episode_dir),
        "output_dir": str(output_dir),
        "annotation_minimal": str(minimal_h5),
        "kept_videos": video_info,
        "target_image_resolution": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "frames": int(total),
        "fps": DEFAULT_FPS,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "hand_action_dim": HAND_ACTION_DIM,
        "token_dim": TOKEN_DIM,
        "hand_validity": [asdict(left_report), asdict(right_report)],
        "hand_policy": (
            "frame invalid if non-finite or degenerate/zero hand geometry; "
            "fill from previous valid frame; leading invalid frames fill zeros"
        ),
        "hand_retarget": {"left": left_method, "right": right_method},
        "smpl": smpl_meta,
        "g1_retarget": asdict(g1_status),
        "gmr_post_smoothing": gmr_post_smoothing,
        "g1_motion_policy": (
            "g1_motion_gmr_raw.npz is direct GMR output. g1_motion_gmr.npz, replay, "
            "SONIC encoder obs, tokens, and parquet use post-GMR smoothed G1 motion."
        ),
        "sonic_encoder_mode": {"id": sonic_encoder_mode, "name": "g1"},
        "sonic_encoder_inputs": [
            "encoder_mode_4",
            "motion_joint_positions_10frame_dt0.1s_isaaclab_order",
            "motion_joint_velocities_10frame_dt0.1s_isaaclab_order",
            "motion_anchor_orientation_10frame_dt0.1s = inv(root_quat[t]) * root_quat[t + k]",
        ],
        "sonic_encoder_coordinate_policy": (
            "G1 encoder root orientation is the SONIC tracking-command relative root rotation. "
            "GMR/replay files stay in MuJoCo DOF order; encoder obs converts DOFs to IsaacLab order."
        ),
        "sonic_g1_future_dt_sec": SONIC_G1_FUTURE_DT,
        "sonic_g1_future_step_frames": max(1, int(round(SONIC_G1_FUTURE_DT * DEFAULT_FPS))),
        "sonic_smpl_future_dt_sec": SONIC_SMPL_FUTURE_DT,
        "sonic_smpl_future_step_frames": max(1, int(round(SONIC_SMPL_FUTURE_DT * DEFAULT_FPS))),
        "sonic_dof_order": "post-smoothed g1_motion_gmr.npz/replay use MuJoCo order; sonic_encoder_obs_1762d.npy uses G1_MUJOCO_TO_ISAACLAB_DOF.",
        "sonic_token": token_status,
        "dropped_hdf5_groups": sorted(DROP_HDF5_GROUPS),
        "keep_intermediate": bool(keep_intermediate),
        "lerobot_ready_required_files": list(PROCESSED_REQUIRED_FILENAMES),
        "outputs": {
            "state_action": "state_action_107_104.parquet",
            "stereo_left": "stereo_left.mp4",
            "stereo_right": "stereo_right.mp4",
            "processing_complete": PROCESSED_COMPLETE_MARKER,
        },
        "intermediate_outputs": {
            "hand_action": "wuji_hand_action_40d.npy",
            "g1_motion_gmr_raw": "g1_motion_gmr_raw.npz",
            "g1_motion_gmr": "g1_motion_gmr.npz",
            "g1_motion": "g1_motion_minimal.npz",
            "smpl_motion": "smpl_motion_minimal.npz",
            "gmr_joint_input": "gmr_input_smpl24_joints.npz",
            "g1_replay": "g1_motion_gmr_replay.pkl",
            "sonic_encoder_obs": "sonic_encoder_obs_1762d.npy",
            "sonic_action_token": "sonic_action_token_64d.npy",
        },
        "replay": replay_info,
    }
    lerobot_files = write_lerobot_ready_files(
        output_dir=output_dir,
        caption_payload=caption_payload,
        compact_metadata=compact_metadata,
        report=report,
        rows=len(frames),
    )
    report["outputs"].update(lerobot_files)
    removed_intermediate = [] if keep_intermediate else cleanup_intermediate_outputs(output_dir)
    report["removed_intermediate_files"] = removed_intermediate
    write_json(output_dir / "processing_report.json", report)
    marker = write_processed_complete_marker(output_dir, report)
    print(f"[INFO] wrote {output_dir}")
    print(f"[INFO] processed complete marker: {marker}")
    print(f"[INFO] frames={total} state_dim={STATE_DIM} action_dim={ACTION_DIM}")
    print(f"[INFO] G1 retarget ok={g1_status.ok} method={g1_status.method}")
    print(f"[INFO] SONIC token ok={token_status.get('ok')} method={token_status.get('method')}")
    return report


def convert(args: argparse.Namespace) -> int:
    convert_episode_dir(
        args.episode_dir,
        args.output_dir,
        resize_videos=args.resize_videos,
        require_gmr=args.require_gmr,
        gmr_root=args.gmr_root,
        gmr_python=args.gmr_python,
        smplx_folder=args.smplx_folder,
        gmr_robot=args.gmr_robot,
        wuji_root=args.wuji_root,
        wuji_python=args.wuji_python,
        wuji_left_config=args.wuji_left_config,
        wuji_right_config=args.wuji_right_config,
        require_wuji=args.require_wuji,
        encoder_model=args.encoder_model,
        min_free_gb=args.min_free_gb,
        overwrite=args.overwrite,
        keep_intermediate=args.keep_intermediate,
    )
    return 0


def self_test(args: argparse.Namespace) -> int:
    tmp = args.tmp_dir.expanduser().resolve()
    if tmp.exists():
        shutil.rmtree(tmp)
    raw = tmp / "raw"
    out = tmp / "processed"
    raw.mkdir(parents=True)

    t = 12
    left = np.zeros((t, 21, 3), dtype=np.float32)
    right = np.zeros((t, 21, 3), dtype=np.float32)
    for i in range(21):
        left[:, i, :] = np.array([i * 0.01, 0.02, 0.03], dtype=np.float32)
        right[:, i, :] = np.array([-i * 0.01, 0.02, 0.03], dtype=np.float32)
    left[:2] = np.nan
    right[5:7] = np.nan
    keypoints = np.zeros((t, 24, 3), dtype=np.float32)
    keypoints[:, :, 2] = 1.0
    ts = np.tile(np.eye(4, dtype=np.float32).reshape(1, 4, 4), (t, 1, 1))
    ts[:, 2, 3] = 0.8
    with h5py.File(raw / "annotation.hdf5", "w") as h5:
        h5.create_group("calibration").attrs["dropped"] = True
        h5.create_group("slam").attrs["dropped"] = True
        h5.create_dataset("depth/depth", data=np.zeros((t, 2, 2), dtype=np.float32))
        h5.create_dataset("hand_mocap/left_joints_3d", data=left)
        h5.create_dataset("hand_mocap/right_joints_3d", data=right)
        h5.create_dataset("full_body_mocap/keypoints", data=keypoints)
        h5.create_dataset("full_body_mocap/Ts_world_root", data=ts)
        h5.create_dataset("imu/accel_xyz", data=np.zeros((t, 3), dtype=np.float32))
        h5.create_dataset("imu/gyro_xyz", data=np.zeros((t, 3), dtype=np.float32))
        h5.create_dataset("video/frame_number", data=np.arange(t, dtype=np.int64))
        h5.create_dataset("caption", data=np.bytes_("synthetic test"))
    for name in ("stereo_left.mp4", "stereo_right.mp4"):
        (raw / name).write_bytes(b"not a real mp4; self-test disables resize")

    convert_episode_dir(
        raw,
        out,
        resize_videos=False,
        require_gmr=False,
        gmr_root=None,
        gmr_python=None,
        smplx_folder=None,
        gmr_robot="unitree_g1",
        wuji_root=None,
        wuji_python=None,
        wuji_left_config=None,
        wuji_right_config=None,
        require_wuji=False,
        encoder_model=None,
        min_free_gb=0.0,
        overwrite=True,
        keep_intermediate=True,
    )
    report = json.loads((out / "processing_report.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(out / "state_action_107_104.parquet")
    assert report["frames"] == t
    assert len(df) == t - 1
    assert len(df.iloc[0]["states"]) == STATE_DIM
    assert len(df.iloc[0]["action"]) == ACTION_DIM
    with h5py.File(out / "annotation_minimal.hdf5", "r") as h5:
        assert "calibration" not in h5
        assert "slam" not in h5
        assert "depth" in h5
    print(f"[INFO] self-test passed: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list-remote", help="List remote files that would be used")
    p.add_argument("--sample", action="store_true", help="Use xperience-10m-sample")
    p.add_argument("--token", default=None)
    p.add_argument("--revision", default="main")
    p.add_argument("--only-kept", action="store_true", help="Show only annotation/stereo files")
    p.add_argument("--limit", type=int, default=80)
    p.set_defaults(func=list_remote_files)

    p = sub.add_parser("download", help="Download minimal raw files and optionally parse after each episode")
    p.add_argument("--sample", action="store_true", help="Use xperience-10m-sample")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--revision", default="main")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--min-free-gb", type=float, default=50.0)
    p.add_argument("--parse-after-download", action="store_true")
    p.add_argument("--delete-raw-after-parse", action="store_true", help="Delete kept raw episode files after successful parsing")
    p.add_argument(
        "--skip-existing-processed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When parsing, skip conversion if the processed output directory already exists",
    )
    p.add_argument("--resize-videos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-gmr", action="store_true")
    p.add_argument("--gmr-root", type=Path, default=None, help="Optional checkout of https://github.com/YanjieZe/GMR")
    p.add_argument("--gmr-python", type=Path, default=None, help="Optional Python executable from a GMR environment")
    p.add_argument("--smplx-folder", type=Path, default=None, help="Optional SMPL-X body model folder for GMR")
    p.add_argument("--gmr-robot", type=str, default="unitree_g1", help="GMR robot target, e.g. unitree_g1")
    p.add_argument("--wuji-root", type=Path, default=None, help="Optional checkout of wuji-hand")
    p.add_argument("--wuji-python", type=Path, default=None, help="Python executable with wuji_retargeting dependencies")
    p.add_argument("--wuji-left-config", type=Path, default=None, help="Left-hand Wuji retarget YAML")
    p.add_argument("--wuji-right-config", type=Path, default=None, help="Right-hand Wuji retarget YAML")
    p.add_argument("--require-wuji", action="store_true", help="Fail instead of falling back if Wuji retargeting cannot run")
    p.add_argument("--encoder-model", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true", help="Keep GMR/WUJI/SONIC debug artifacts in processed output")
    p.add_argument("--sleep-s", type=float, default=0.0)
    p.set_defaults(func=download)

    p = sub.add_parser("stream", help="Stream episodes: download one complete episode, process in parallel, then optionally delete raw")
    p.add_argument("--sample", action="store_true", help="Use xperience-10m-sample")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--token", default=None)
    p.add_argument("--revision", default="main")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--min-free-gb", type=float, default=50.0)
    p.add_argument("--delete-raw-after-parse", action="store_true", help="Delete kept raw episode files after successful parsing")
    p.add_argument(
        "--skip-existing-processed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip conversion if the processed output directory already exists and --overwrite is not set",
    )
    p.add_argument("--process-workers", type=int, default=1, help="Number of parallel episode conversion workers")
    p.add_argument(
        "--prefetch-episodes",
        type=int,
        default=1,
        help="Extra raw episode slots beyond active processors; 1 overlaps one download with one conversion",
    )
    p.add_argument("--resize-videos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-gmr", action="store_true")
    p.add_argument("--gmr-root", type=Path, default=None, help="Optional checkout of https://github.com/YanjieZe/GMR")
    p.add_argument("--gmr-python", type=Path, default=None, help="Optional Python executable from a GMR environment")
    p.add_argument("--smplx-folder", type=Path, default=None, help="Optional SMPL-X body model folder for GMR")
    p.add_argument("--gmr-robot", type=str, default="unitree_g1", help="GMR robot target, e.g. unitree_g1")
    p.add_argument("--wuji-root", type=Path, default=None, help="Optional checkout of wuji-hand")
    p.add_argument("--wuji-python", type=Path, default=None, help="Python executable with wuji_retargeting dependencies")
    p.add_argument("--wuji-left-config", type=Path, default=None, help="Left-hand Wuji retarget YAML")
    p.add_argument("--wuji-right-config", type=Path, default=None, help="Right-hand Wuji retarget YAML")
    p.add_argument("--require-wuji", action="store_true", help="Fail instead of falling back if Wuji retargeting cannot run")
    p.add_argument("--encoder-model", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true", help="Keep GMR/WUJI/SONIC debug artifacts in processed output")
    p.add_argument("--sleep-s", type=float, default=0.0)
    p.set_defaults(func=stream_download_process)

    p = sub.add_parser("inspect-hdf5", help="Inspect HDF5 tree and hand non-finite runs")
    p.add_argument("annotation", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.set_defaults(func=inspect_hdf5)

    p = sub.add_parser("convert", help="Convert one already-downloaded episode directory")
    p.add_argument("--episode-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resize-videos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-gmr", action="store_true")
    p.add_argument("--gmr-root", type=Path, default=None, help="Optional checkout of https://github.com/YanjieZe/GMR")
    p.add_argument("--gmr-python", type=Path, default=None, help="Optional Python executable from a GMR environment")
    p.add_argument("--smplx-folder", type=Path, default=None, help="Optional SMPL-X body model folder for GMR")
    p.add_argument("--gmr-robot", type=str, default="unitree_g1", help="GMR robot target, e.g. unitree_g1")
    p.add_argument("--wuji-root", type=Path, default=None, help="Optional checkout of wuji-hand")
    p.add_argument("--wuji-python", type=Path, default=None, help="Python executable with wuji_retargeting dependencies")
    p.add_argument("--wuji-left-config", type=Path, default=None, help="Left-hand Wuji retarget YAML")
    p.add_argument("--wuji-right-config", type=Path, default=None, help="Right-hand Wuji retarget YAML")
    p.add_argument("--require-wuji", action="store_true", help="Fail instead of falling back if Wuji retargeting cannot run")
    p.add_argument("--encoder-model", type=Path, default=None)
    p.add_argument("--min-free-gb", type=float, default=10.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-intermediate", action="store_true", help="Keep GMR/WUJI/SONIC debug artifacts in processed output")
    p.set_defaults(func=convert)

    p = sub.add_parser("self-test", help="Run a tiny synthetic conversion test under /tmp")
    p.add_argument("--tmp-dir", type=Path, default=Path("/tmp/xperience_minimal_pipeline_test"))
    p.set_defaults(func=self_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "token") and args.token is None:
        args.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        return int(args.func(args))
    except DiskAlmostFull as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
