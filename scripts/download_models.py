#!/usr/bin/env python3
"""Download selected WB-WAM checkpoints into the training directory layout."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

BASE_REPO = "WB-WAM/WB-WAM-Pretrain-Midtrain"
ARENA_REPO = "WB-WAM/WB-WAM-HumanoidArena"
BASE_STEPS = {"pretrain": "step_037617.pt", "midtrain": "step_020560.pt"}
ARENA_STEPS = {
    "box_shelf": "step_013000.pt",
    "hammer": "step_019150.pt",
    "kick_football": "step_010600.pt",
    "obstacle_navigation": "step_020450.pt",
    "open_door": "step_015550.pt",
    "punch_markers": "step_010500.pt",
    "sit_sofa": "step_013300.pt",
}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "training/checkpoints/wbwam"


def checkpoint_paths(stage: str, task: str | None = None) -> list[str]:
    if stage in BASE_STEPS:
        directories = [(stage, BASE_STEPS[stage])]
    elif stage == "arena":
        tasks = [task] if task else ARENA_STEPS
        directories = [
            (f"humanoid_arena_native50/{name}", ARENA_STEPS[name]) for name in tasks
        ]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return [
        f"{directory}/{name}"
        for directory, step in directories
        for name in (step, "config.yaml", "dataset_stats.json")
    ]


def find_hf_cli() -> Path | None:
    """Find the Hugging Face CLI on PATH or beside the active Python."""
    for name in ("hf", "huggingface-cli"):
        path = shutil.which(name)
        if path:
            return Path(path)
    python_bin = Path(sys.executable).resolve().parent
    for name in ("hf", "huggingface-cli"):
        path = python_bin / name
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def run_hf(repo: str, patterns: list[str], output: Path, dry_run: bool) -> None:
    cli = find_hf_cli()
    cli_name = cli.name if cli else "hf"
    args = [str(cli or cli_name), "download", repo, "--local-dir", str(output)]
    if cli_name == "hf":
        for pattern in patterns:
            args.extend(["--include", pattern])
    else:
        args.extend(["--include", *patterns])
    print(shlex.join(args), flush=True)
    if dry_run:
        return
    if cli is None:
        raise RuntimeError("Install huggingface_hub with `hf` or `huggingface-cli` first")
    subprocess.run(args, check=True)


def download(stage: str, task: str | None, output: Path, dry_run: bool) -> None:
    if task and stage != "arena":
        raise ValueError("--task is only valid with the arena stage")
    if stage == "arena" and task is not None and task not in ARENA_STEPS:
        raise ValueError("Choose a valid Arena task with --task, or omit --task for all tasks")

    selected = ("pretrain", "midtrain", "arena") if stage == "all" else (stage,)
    jobs = []
    if "pretrain" in selected or "midtrain" in selected:
        base_stages = [name for name in ("pretrain", "midtrain") if name in selected]
        jobs.append(
            (
                BASE_REPO,
                [f"{name}/**" for name in base_stages],
                [path for name in base_stages for path in checkpoint_paths(name)],
            )
        )
    if "arena" in selected:
        pattern = f"humanoid_arena_native50/{task}/**" if task else "humanoid_arena_native50/**"
        jobs.append((ARENA_REPO, [pattern], checkpoint_paths("arena", task)))

    for repo, patterns, paths in jobs:
        run_hf(repo, patterns, output, dry_run)
        if not dry_run:
            missing = [
                path
                for path in paths
                if not (output / path).is_file() or (output / path).stat().st_size == 0
            ]
            if missing:
                raise RuntimeError(f"Incomplete checkpoint release in {repo}: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pretrain", "midtrain", "arena", "all"))
    parser.add_argument("--task", choices=tuple(ARENA_STEPS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    download(args.stage, args.task, args.output.expanduser().resolve(), args.dry_run)


if __name__ == "__main__":
    main()
