#!/usr/bin/env python3
"""Download minimal Xperience episodes from Hugging Face into a processing queue."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FULL_REPO_ID = "ropedia-ai/xperience-10m"
SAMPLE_REPO_ID = "ropedia-ai/xperience-10m-sample"
REPO_TYPE = "dataset"
KEEP_FILENAMES = {"annotation.hdf5", "stereo_left.mp4", "stereo_right.mp4"}
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


class DiskAlmostFull(RuntimeError):
    pass


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


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
        raise RuntimeError("huggingface_hub is required") from exc
    return HfApi, hf_hub_download


def hf_repo_id(sample: bool) -> str:
    return SAMPLE_REPO_ID if sample else FULL_REPO_ID


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
    marker = episode_raw_dir(raw_root, files) / ".xperience_download_complete.json"
    payload = build_manifest(
        raw_root=raw_root,
        episode_id=episode_id,
        files=files,
        repo_id=repo_id,
        revision=revision,
    )
    write_json_atomic(marker, payload)
    return marker


def build_manifest(
    *,
    raw_root: Path,
    episode_id: str,
    files: Mapping[str, tuple[str, int]],
    repo_id: str,
    revision: str,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "repo_id": repo_id,
        "revision": revision,
        "raw_root": str(raw_root),
        "episode_dir": str(episode_raw_dir(raw_root, files)),
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


def queue_backlog_count(queue_dir: Path) -> int:
    queue_dir.mkdir(parents=True, exist_ok=True)
    return sum(1 for _ in queue_dir.glob("*.json")) + sum(1 for _ in queue_dir.glob("*.json.processing"))


def wait_for_queue_slot(queue_dir: Path, max_queued_episodes: int, sleep_s: float, stop_file: Path | None) -> None:
    if max_queued_episodes <= 0:
        return
    while queue_backlog_count(queue_dir) >= max_queued_episodes:
        if stop_file is not None and stop_file.exists():
            raise RuntimeError(f"stop file exists: {stop_file}")
        print(
            f"[INFO] queue backlog {queue_backlog_count(queue_dir)} >= {max_queued_episodes}; waiting",
            flush=True,
        )
        time.sleep(max(0.5, sleep_s))


def write_queue_manifest(queue_dir: Path, manifest: Mapping[str, Any]) -> Path:
    queue_dir.mkdir(parents=True, exist_ok=True)
    episode_id = str(manifest["episode_id"])
    path = queue_dir / f"{episode_id}.json"
    processing_path = path.with_suffix(path.suffix + ".processing")
    if path.exists() or processing_path.exists():
        print(f"[INFO] queue manifest already exists: {path}")
        return path
    write_json_atomic(path, manifest)
    return path


def processed_complete(processed_root: Path | None, episode_id: str) -> bool:
    if processed_root is None:
        return False
    output_dir = processed_root / episode_id
    if not (output_dir / PROCESSED_COMPLETE_MARKER).exists():
        return False
    return all((output_dir / filename).is_file() for filename in PROCESSED_REQUIRED_FILENAMES)


def run_stream(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    repo_id = hf_repo_id(args.sample)
    raw_root = args.raw_root.expanduser().resolve()
    queue_dir = args.queue_dir.expanduser().resolve()
    processed_root = args.processed_root.expanduser().resolve() if args.processed_root else None
    stop_file = args.stop_file.expanduser().resolve() if args.stop_file else None
    raw_root.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)

    files = remote_kept_files(repo_id, token, args.revision)
    episodes = sorted(group_episode_files(files).items())
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    print(
        f"repo={repo_id} episodes={len(episodes)} raw_root={raw_root} queue_dir={queue_dir} "
        f"max_queued_episodes={args.max_queued_episodes}",
        flush=True,
    )
    queued = 0
    downloaded = 0
    skipped_processed = 0
    for episode_id, episode_files in episodes:
        if stop_file is not None and stop_file.exists():
            print(f"[STOP] stop file exists: {stop_file}", file=sys.stderr)
            return 1
        missing = KEEP_FILENAMES - set(episode_files)
        if missing:
            print(f"[WARN] skip {episode_id}: missing {sorted(missing)}")
            continue
        if processed_complete(processed_root, episode_id):
            skipped_processed += 1
            print(f"[INFO] skip already processed episode {episode_id}")
            continue

        wait_for_queue_slot(queue_dir, args.max_queued_episodes, args.queue_wait_s, stop_file)
        print(f"[INFO] downloading {episode_id}", flush=True)
        for filename in sorted(KEEP_FILENAMES):
            hf_path, size = episode_files[filename]
            local = download_one_file(
                repo_id=repo_id,
                hf_path=hf_path,
                size=size,
                raw_root=raw_root,
                token=token,
                revision=args.revision,
                min_free_gb=args.min_free_gb,
            )
            print(f"  {episode_id} {filename}: {local}", flush=True)
        if not episode_files_complete(raw_root, episode_files):
            raise RuntimeError(f"episode did not pass completion check: {episode_id}")
        write_episode_complete_marker(raw_root, episode_id, episode_files, repo_id, args.revision)
        manifest = build_manifest(
            raw_root=raw_root,
            episode_id=episode_id,
            files=episode_files,
            repo_id=repo_id,
            revision=args.revision,
        )
        queue_path = write_queue_manifest(queue_dir, manifest)
        print(f"[INFO] queued {episode_id}: {queue_path}", flush=True)
        downloaded += 1
        queued += 1
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    print(
        f"[INFO] download summary downloaded={downloaded} queued={queued} "
        f"skipped_processed={skipped_processed}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stream", help="Download kept episode files and enqueue completed episodes")
    p.add_argument("--sample", action="store_true", help="Use xperience-10m-sample")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--queue-dir", type=Path, required=True)
    p.add_argument("--processed-root", type=Path, default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--revision", default="main")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--min-free-gb", type=float, default=50.0)
    p.add_argument("--max-queued-episodes", type=int, default=2)
    p.add_argument("--queue-wait-s", type=float, default=5.0)
    p.add_argument("--sleep-s", type=float, default=0.0)
    p.add_argument("--stop-file", type=Path, default=None)
    p.set_defaults(func=run_stream)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DiskAlmostFull as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
