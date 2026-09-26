"""Create deterministic episode batches and summarize benchmark results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def episode_identity(task: str, seed: int, repeat: int) -> dict[str, int]:
    digest = hashlib.sha256(f"{task}|{seed}|{repeat}".encode()).digest()
    return {
        "seed": seed,
        "repeat_idx": repeat,
        "episode_seed": int.from_bytes(digest[-4:], "big") & 0x7FFFFFFF,
        "episode_index": seed * 20 + repeat,
    }


def prepare(
    root: Path,
    seed: int,
    task: str,
    label: str,
    max_steps: int,
    repeats: int = 20,
    repeat_list: list[int] | None = None,
) -> None:
    repeats_to_run = repeat_list if repeat_list is not None else list(range(repeats))
    episodes = []
    for repeat in repeats_to_run:
        identity = episode_identity(task, seed, repeat)
        index = identity["episode_index"]
        episodes.append(
            {
                **identity,
                "result_json": str(root / f"episodes/trial_{index}.json"),
                "success_video_dir": str(root / "videos/success"),
                "failure_video_dir": str(root / "videos/failure"),
                "recording_save_dir": str(root / "recordings"),
                "model_label": label,
                "max_steps": max_steps,
                "video_fps": 30,
                "post_termination_record_steps": 50,
            }
        )
    for directory in ("episodes", "videos/success", "videos/failure", "recordings"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "episode_batch.json").write_text(json.dumps({"episodes": episodes}, indent=2))


def completed_repeats(root: Path, task: str, seed: int, repeats: int) -> set[int]:
    completed = set()
    for repeat in range(repeats):
        identity = episode_identity(task, seed, repeat)
        path = root / f"episodes/trial_{identity['episode_index']}.json"
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record.get("success"), bool) and record.get("failure_reason") in {
            None,
            "success",
            "fall",
            "timeout",
        }:
            completed.add(repeat)
    return completed


def summarize(root: Path, expected: int | None = None) -> dict:
    paths = sorted((root / "episodes").glob("trial_*.json"))
    counts = {"success": 0, "fall": 0, "timeout": 0, "other": 0}
    for path in paths:
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            counts["other"] += 1
            continue
        outcome = "success" if record.get("success") else record.get("failure_reason")
        counts[outcome if outcome in ("success", "fall", "timeout") else "other"] += 1
    counts["completed"] = len(paths)
    counts["missing"] = max(0, expected - len(paths)) if expected is not None else 0
    counts["videos"] = len(list((root / "videos").rglob("*.mp4")))
    log = (root / "sim.log").read_text(errors="replace") if (root / "sim.log").exists() else ""
    markers = (
        "get_action error:",
        "GEAR-SONIC inference error:",
        "Traceback (most recent call last)",
        "[sim_eval_vla] startup failed:",
    )
    counts["runtime_errors"] = [marker for marker in markers if marker in log]
    counts["success_rate"] = counts["success"] / len(paths) if paths else None
    (root / "summary.json").write_text(json.dumps(counts, indent=2))
    return counts


def _parse_repeat_list(value: str) -> list[int] | None:
    repeats = [int(item) for item in value.split(",") if item.strip()]
    if not repeats:
        return None
    if len(set(repeats)) != len(repeats) or not all(0 <= repeat < 20 for repeat in repeats):
        raise argparse.ArgumentTypeError("repeat indices must be distinct values within 0..19")
    return repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--task", default="task")
    parser.add_argument("--label", default="model")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--repeat-list", type=_parse_repeat_list, default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.max_steps is None or args.max_steps <= 0:
            parser.error("prepare requires --max-steps with a positive value")
        prepare(
            args.root,
            args.seed,
            args.task,
            args.label,
            args.max_steps,
            args.repeats,
            args.repeat_list,
        )
    else:
        result = summarize(args.root, expected=args.repeats)
        print(json.dumps(result))
        if args.require_complete and (result["missing"] or result["other"] or result["runtime_errors"]):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
