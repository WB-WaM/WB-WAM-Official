#!/usr/bin/env python3
"""Export 20 Hz SONIC collector tasks to a native LeRobot v3 task/record archive."""

import argparse
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "processing"

import av

from .common import (
    check_output_location,
    digest,
    discover,
    output_lock,
    read_json,
    source_signature,
    tree_hashes,
    write_json,
)
from .schema import VERSION
from .validate_lerobot import validate_record
from .writer import build_record


def convert(args) -> list[Path]:
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    check_output_location(source, output)
    jobs = discover(
        source,
        task=args.task,
        task_id=args.task_id,
        exclude=args.exclude_episode,
        limit=args.limit_episodes,
    )
    options = {"image_size": args.image_size, "video_max_frames": args.video_max_frames}
    for job in jobs:
        print(
            f"[SCAN] {job['source'].name} -> {job['relative_output']}: {len(job['episodes'])} episodes", flush=True
        )
        job["signature"] = source_signature(job, options)
    identity = {
        "version": VERSION,
        "options": options,
        "records": {job["relative_output"]: job["signature"] for job in jobs},
        "exclude_episode": sorted(args.exclude_episode),
    }
    manifest = {"identity": identity, "signature": digest(identity)}
    if args.dry_run:
        print(
            "[OK] source paths/fingerprints checked; no files written (numerical/video validation runs on export)"
        )
        return []
    with output_lock(output):
        manifest_path = output / ".conversion.json"
        if manifest_path.is_symlink():
            raise ValueError("Refusing a symlinked output manifest")
        if manifest_path.exists():
            if not args.resume:
                raise FileExistsError(
                    "Output already exists; use --resume with identical sources/options or a new path"
                )
            if read_json(manifest_path) != manifest:
                raise ValueError("Resume source/selection/options changed; use a new output directory")
        else:
            if any(path.name != ".conversion.lock" for path in output.iterdir()):
                raise ValueError(
                    "Output is nonempty and is not owned by this converter; nothing will be overwritten"
                )
            write_json(manifest_path, manifest)
        for job in jobs:
            final = output / job["relative_output"]
            if final.parent.is_symlink() or final.is_symlink():
                raise ValueError("Refusing symlinked output task/record directories")
            if final.exists():
                marker = final / "meta/processing_complete.json"
                if not marker.is_file() or marker.is_symlink():
                    raise ValueError(f"Unowned/incomplete record at {final}; use a new output path")
                completed = read_json(marker)
                hashes = tree_hashes(final)
                hashes.pop("meta/processing_complete.json")
                if completed.get("source_signature") != job["signature"] or completed.get("files") != hashes:
                    raise ValueError(f"Record changed since export: {final}; refusing to reuse it")
                print(f"[REUSE] {final}", flush=True)
                continue
            with tempfile.TemporaryDirectory(prefix=".sonic-record-", dir=output) as temporary:
                staging = Path(temporary) / "record"
                staging.mkdir()
                build_record(job, staging, options)
                result = validate_record(staging)
                if source_signature(job, options) != job["signature"]:
                    raise ValueError("Source changed during conversion; refusing to publish this record")
                write_json(
                    staging / "meta/processing_complete.json",
                    {
                        "source_signature": job["signature"],
                        "files": tree_hashes(staging),
                        "validation": result,
                    },
                )
                final.parent.mkdir(parents=True, exist_ok=True)
                staging.rename(final)
            print(f"[DONE] {final}: {result['frames']} samples", flush=True)
    return [output / job["relative_output"] for job in jobs]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="One collector task, or a parent containing tasks"
    )
    parser.add_argument("--output", type=Path, required=True, help="New archive root (separate from the input)")
    parser.add_argument("--task", help="Natural-language task override; single-task input only")
    parser.add_argument("--task-id", help="Output task folder, e.g. checkout; single-task input only")
    parser.add_argument("--image-size", nargs=2, type=int, default=[360, 270], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--video-max-frames", type=int, default=3600, help="Split videos between episodes only")
    parser.add_argument(
        "--exclude-episode", action="append", default=[], help="task_dir/episode_XXXXXX; repeatable"
    )
    parser.add_argument("--limit-episodes", type=int, help="Select the first N non-excluded episodes per record")
    parser.add_argument(
        "--resume", action="store_true", help="Verify/reuse completed records; retry incomplete records"
    )
    parser.add_argument("--dry-run", action="store_true", help="Check discovery and input paths without writing")
    args = parser.parse_args(argv)
    if any(value <= 0 or value % 2 for value in args.image_size):
        parser.error("--image-size dimensions must be positive even integers")
    if args.video_max_frames <= 0 or (args.limit_episodes is not None and args.limit_episodes <= 0):
        parser.error("--video-max-frames and --limit-episodes must be positive")
    return parser, args


def main(argv=None) -> int:
    parser, args = parse_args(argv)
    try:
        convert(args)
    except (ValueError, KeyError, OSError, av.FFmpegError) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
