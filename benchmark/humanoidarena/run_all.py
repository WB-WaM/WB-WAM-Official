"""Run all HumanoidArena tasks on a user-selected pool of GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

try:
    from . import run_eval
except ImportError:
    import run_eval


HERE = Path(__file__).resolve().parent


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)) or len(gpus) > 7:
        raise argparse.ArgumentTypeError("GPUs must be a comma-separated list of 1-7 distinct IDs")
    return gpus


def gpu_config(base: dict, gpu: str, root: Path) -> Path:
    config = dict(base)
    config["inference_gpu"] = gpu
    config["simulation_gpu"] = gpu
    config["allow_shared_gpu"] = True
    safe_gpu = gpu.replace("/", "_").replace(":", "_")
    for key in ("isaac_home", "isaac_cache"):
        path = Path(str(base[key])).expanduser() / f"gpu_{safe_gpu}"
        path.mkdir(parents=True, exist_ok=True)
        config[key] = str(path.resolve())
    path = root / f"runtime_gpu_{safe_gpu}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run_task(args: argparse.Namespace, task: str, config: Path) -> None:
    command = [
        sys.executable,
        str(HERE / "run_eval.py"),
        "--config",
        str(config),
        "--task",
        task,
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--output",
        str(args.output),
        "--seeds",
        ",".join(map(str, args.seeds)),
        "--repeats",
        str(args.repeats),
    ]
    if args.weight_file:
        command.extend(("--weight-file", args.weight_file))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0"))
    parser.add_argument("--seeds", type=run_eval.parse_seeds, default=run_eval.parse_seeds("0,1,2"))
    parser.add_argument("--repeats", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--weight-file")
    args = parser.parse_args()

    base = run_eval.load_yaml(args.config.expanduser().resolve())
    tasks = list(run_eval.load_tasks())
    args.output = args.output.expanduser().resolve()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="wbwam-humanoidarena-") as temporary:
        temporary_root = Path(temporary)
        configs = {gpu: gpu_config(base, gpu, temporary_root) for gpu in args.gpus}
        failures: list[tuple[str, str]] = []

        def worker(slot: int) -> None:
            gpu = args.gpus[slot]
            for task in tasks[slot:: len(args.gpus)]:
                print(f"[gpu {gpu}] run {task}", flush=True)
                try:
                    run_task(args, task, configs[gpu])
                except subprocess.CalledProcessError as error:
                    failures.append((task, f"gpu={gpu} exit={error.returncode}"))
                    return

        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = [executor.submit(worker, slot) for slot in range(len(args.gpus))]
            for future in as_completed(futures):
                future.result()

    summary = run_eval.aggregate(args.output, tasks, args.seeds, args.repeats)
    run_eval.print_summary(summary)
    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
