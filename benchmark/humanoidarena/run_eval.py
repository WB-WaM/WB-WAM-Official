"""Run or summarize the WB-WAM HumanoidArena Native50 benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import yaml

try:
    from .artifacts import completed_repeats, summarize
except ImportError:
    from artifacts import completed_repeats, summarize

HERE = Path(__file__).resolve().parent
TASKS_FILE = HERE / "configs/tasks.yaml"
UPSTREAM_LOCK = HERE / "upstream.lock.json"
COMMON_RUNTIME_KEYS = {
    "runtime_root",
    "inference_python",
    "humanoidarena_root",
    "humanoidarena_assets",
    "dependencies_root",
    "wbwam_root",
    "base_models_root",
    "isaac_home",
    "isaac_cache",
    "sonic_release",
    "inference_gpu",
    "simulation_gpu",
    "allow_shared_gpu",
    "max_preexisting_gpu_mib",
}
CONTAINER_RUNTIME_KEYS = {
    "isaac_sim_image",
    "isaac_container_mode",
    "apptainer_binary",
    "isaac_site_packages",
}
NATIVE_RUNTIME_KEYS = {"simulation_python"}


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def load_tasks(path: Path = TASKS_FILE) -> dict[str, dict]:
    data = load_yaml(path)
    tasks = data.get("tasks")
    if data.get("version") != 1 or not isinstance(tasks, dict) or len(tasks) != 7:
        raise ValueError("tasks.yaml must contain the seven version-1 benchmark tasks")
    required = {
        "instruction_key",
        "expected_prompt",
        "gym_task",
        "env_config",
        "max_steps",
        "denoise_steps",
        "replan",
    }
    for name, task in tasks.items():
        if not isinstance(task, dict) or set(task) != required:
            raise ValueError(f"Invalid task configuration: {name}")
        if task["denoise_steps"] not in (20, 40) or task["replan"] not in (14, 20):
            raise ValueError(f"Unsupported benchmark D/R values for {name}")
        if not isinstance(task["max_steps"], int) or task["max_steps"] <= 0:
            raise ValueError(f"Invalid max_steps for {name}")
    return tasks


def load_runtime(path: Path, validate_paths: bool = True) -> dict:
    config = load_yaml(path)
    backend = config.get("simulation_backend", "container")
    if backend not in {"container", "native"}:
        raise ValueError("simulation_backend must be 'container' or 'native'")
    config["simulation_backend"] = backend
    required = COMMON_RUNTIME_KEYS | (CONTAINER_RUNTIME_KEYS if backend == "container" else NATIVE_RUNTIME_KEYS)
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing runtime keys: {', '.join(missing)}")
    if not isinstance(config["allow_shared_gpu"], bool):
        raise ValueError("allow_shared_gpu must be true or false")
    if config["inference_gpu"] == config["simulation_gpu"] and not config["allow_shared_gpu"]:
        raise ValueError("The same GPU is selected twice; set allow_shared_gpu: true to opt in")
    if backend == "container" and config["isaac_container_mode"] not in {"sif", "sandbox"}:
        raise ValueError("isaac_container_mode must be 'sif' or 'sandbox'")
    if not isinstance(config["max_preexisting_gpu_mib"], int) or config["max_preexisting_gpu_mib"] < 0:
        raise ValueError("max_preexisting_gpu_mib must be a non-negative integer")
    bind_roots = config.get("container_bind_roots", [])
    if not isinstance(bind_roots, list) or not all(isinstance(item, str) and item for item in bind_roots):
        raise ValueError("container_bind_roots must be a list of non-empty paths")
    if validate_paths:
        path_keys = required - {
            "inference_gpu",
            "simulation_gpu",
            "allow_shared_gpu",
            "isaac_container_mode",
            "max_preexisting_gpu_mib",
        }
        for key in sorted(path_keys):
            value = Path(str(config[key])).expanduser()
            if not value.exists():
                raise FileNotFoundError(f"{key} does not exist: {value}")
            config[key] = str(value.resolve())
        if not os.access(config["inference_python"], os.X_OK):
            raise PermissionError(f"inference_python is not executable: {config['inference_python']}")
        if backend == "container" and not os.access(config["apptainer_binary"], os.X_OK):
            raise PermissionError(f"apptainer_binary is not executable: {config['apptainer_binary']}")
        if backend == "native" and not os.access(config["simulation_python"], os.X_OK):
            raise PermissionError(f"simulation_python is not executable: {config['simulation_python']}")
        validate_upstream(config)
        assets = Path(config["humanoidarena_assets"])
        for directory in ("objects", "robots"):
            if not (assets / directory).is_dir():
                raise FileNotFoundError(f"humanoidarena_assets is missing the {directory}/ directory: {assets}")
        resolved_bind_roots = []
        for item in bind_roots:
            value = Path(item).expanduser()
            if not value.is_dir():
                raise FileNotFoundError(f"container_bind_roots entry is not a directory: {value}")
            resolved_bind_roots.append(str(value.resolve()))
        config["container_bind_roots"] = resolved_bind_roots
        paligemma = str(config.get("paligemma_root", "")).strip()
        if paligemma:
            value = Path(paligemma).expanduser()
            if not value.exists():
                raise FileNotFoundError(f"paligemma_root does not exist: {value}")
            config["paligemma_root"] = str(value.resolve())
    return config


def validate_upstream(config: dict) -> None:
    """Require pinned upstream source; allow only locally restored assets."""

    upstream = json.loads(UPSTREAM_LOCK.read_text())["humanoidarena"]
    arena = Path(config["humanoidarena_root"])
    head = subprocess.check_output(["git", "-C", str(arena), "rev-parse", "HEAD"], text=True).strip()
    if head != upstream["commit"]:
        raise ValueError(f"HumanoidArena commit mismatch: {head} != {upstream['commit']}")
    status = subprocess.check_output(["git", "-C", str(arena), "status", "--porcelain"], text=True)
    for line in status.splitlines():
        if line.startswith("?? isaaclab_twist2_g1/assets"):
            continue
        raise ValueError(f"third_party/HumanoidArena source must remain unmodified: {line}")


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct non-negative integers")
    return seeds


def checkpoint_paths(checkpoint: Path) -> tuple[Path, Path, Path]:
    """Resolve both the public flat bundle and the legacy nested layout."""

    flat = (checkpoint / "config.yaml", checkpoint / "dataset_stats.json", checkpoint)
    nested = (
        checkpoint / "metadata/config.yaml",
        checkpoint / "metadata/dataset_stats.json",
        checkpoint / "weights",
    )
    for paths in (flat, nested):
        if paths[0].is_file() and paths[1].is_file() and paths[2].is_dir():
            return paths
    raise FileNotFoundError(
        f"{checkpoint} is not a WB-WAM checkpoint directory; expected "
        "config.yaml, dataset_stats.json, and step_*.pt"
    )


def select_weight(checkpoint: Path, requested: str | None) -> str:
    _, _, weights = checkpoint_paths(checkpoint)
    if requested:
        path = weights / requested
        if not path.is_file():
            raise FileNotFoundError(path)
        return requested
    candidates = sorted(weights.glob("step_*.pt"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one step_*.pt in {checkpoint}; found {len(candidates)}. "
            "Use --weight-file to choose explicitly."
        )
    return candidates[0].name


def checkpoint_for(args: argparse.Namespace, task: str) -> Path:
    if args.checkpoint:
        if args.task == "all":
            raise ValueError("--checkpoint is only valid for a single task")
        checkpoint = args.checkpoint
    elif args.checkpoint_root:
        checkpoint = args.checkpoint_root / task
    else:
        raise ValueError("Provide --checkpoint for one task or --checkpoint-root for one/all tasks")
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_paths(checkpoint)
    return checkpoint


def validate_resume(root: Path, expected: dict) -> None:
    episodes = root / "episodes"
    config_path = root / "run_config.json"
    if not root.exists() or (not episodes.exists() and not config_path.exists()):
        return
    if not config_path.is_file():
        raise ValueError(f"Existing output has no run_config.json and cannot be resumed safely: {root}")
    actual = json.loads(config_path.read_text())
    mismatches = {key: (actual.get(key), value) for key, value in expected.items() if actual.get(key) != value}
    if mismatches:
        raise ValueError(f"Existing output configuration mismatch at {root}: {mismatches}")


def task_env(
    runtime: dict,
    task_name: str,
    task: dict,
    checkpoint: Path,
    weight: str,
    output: Path,
    seed: int,
    repeats: int,
    missing: list[int],
    denoise_steps: int,
    replan: int,
) -> dict[str, str]:
    env = dict(os.environ)
    # The simulator processes all missing repeats in one persistent batch.
    # Allow 2 steps/s plus one hour for startup and teardown.
    batch_timeout = 3600 + (len(missing) * task["max_steps"] + 1) // 2
    values = {
        "HA_BENCHMARK_ROOT": HERE,
        "HA_RUNTIME_ROOT": runtime["runtime_root"],
        "HA_INFERENCE_PYTHON": runtime["inference_python"],
        "HA_ARENA_ROOT": runtime["humanoidarena_root"],
        "HA_ARENA_ASSETS": runtime["humanoidarena_assets"],
        "HA_DEPENDENCIES_ROOT": runtime["dependencies_root"],
        "HA_WBWAM_ROOT": runtime["wbwam_root"],
        "HA_BASE_MODELS_ROOT": runtime["base_models_root"],
        "HA_SIMULATION_BACKEND": runtime["simulation_backend"],
        "HA_SIMULATION_PYTHON": runtime.get("simulation_python", ""),
        "HA_ISAAC_IMAGE": runtime.get("isaac_sim_image", ""),
        "HA_ISAAC_CONTAINER_MODE": runtime.get("isaac_container_mode", ""),
        "HA_APPTAINER": runtime.get("apptainer_binary", ""),
        "HA_ISAAC_SITE": runtime.get("isaac_site_packages", ""),
        "HA_ISAAC_HOME": runtime["isaac_home"],
        "HA_ISAAC_CACHE": runtime["isaac_cache"],
        "HA_SONIC_RELEASE": runtime["sonic_release"],
        "HA_PALIGEMMA_ROOT": runtime.get("paligemma_root", ""),
        "HA_INFERENCE_GPU": runtime["inference_gpu"],
        "HA_SIMULATION_GPU": runtime["simulation_gpu"],
        "HA_MAX_PREEXISTING_GPU_MIB": runtime["max_preexisting_gpu_mib"],
        "HA_CUDA_ROOT": runtime.get("cuda_root", ""),
        "HA_CUDNN_ROOT": runtime.get("cudnn_root", ""),
        "HA_CONTAINER_LD_LIBRARY_PATH": runtime.get("container_ld_library_path", ""),
        "HA_CONTAINER_BIND_ROOTS": os.pathsep.join(runtime.get("container_bind_roots", [])),
        "HA_TASK_NAME": task_name,
        "HA_GYM_TASK": task["gym_task"],
        "HA_ENV_CONFIG": Path(runtime["humanoidarena_root"]) / "isaaclab_twist2_g1" / task["env_config"],
        "HA_INSTRUCTION_KEY": task["instruction_key"],
        "HA_EXPECTED_PROMPT": task["expected_prompt"],
        "HA_CHECKPOINT": checkpoint,
        "HA_WEIGHT_FILE": weight,
        "HA_OUTPUT": output,
        "HA_SEED": seed,
        "HA_REPEATS": repeats,
        "HA_REPEAT_LIST": ",".join(map(str, missing)),
        "HA_BATCH_TIMEOUT_SECONDS": batch_timeout,
        "HA_MAX_STEPS": task["max_steps"],
        "HA_DENOISE_STEPS": denoise_steps,
        "HA_REPLAN": replan,
    }
    env.update({key: str(value) for key, value in values.items()})
    return env


def aggregate(output: Path, tasks: list[str], seeds: list[int], repeats: int) -> dict:
    task_rows = {}
    overall_rates = []
    for task in tasks:
        seed_rows = []
        for seed in seeds:
            root = output / task / f"seed_{seed}"
            root.mkdir(parents=True, exist_ok=True)
            counts = summarize(root, expected=repeats)
            rate = counts["success_rate"]
            seed_rows.append({"seed": seed, **counts})
            if rate is not None:
                overall_rates.append(rate)
        rates = [row["success_rate"] for row in seed_rows if row["success_rate"] is not None]
        task_rows[task] = {
            "seeds": seed_rows,
            "mean": statistics.fmean(rates) if rates else None,
            "std": statistics.pstdev(rates) if rates else None,
        }
    result = {
        "tasks": task_rows,
        "overall": {
            "count": len(overall_rates),
            "mean": statistics.fmean(overall_rates) if overall_rates else None,
            "std": statistics.pstdev(overall_rates) if overall_rates else None,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark_summary.json").write_text(json.dumps(result, indent=2))
    return result


def print_summary(result: dict) -> None:
    print("task\tseed successes\tmean ± std")
    for task, row in result["tasks"].items():
        seeds = ", ".join(f"{item['seed']}:{item['success']}/{item['completed']}" for item in row["seeds"])
        metric = "incomplete" if row["mean"] is None else f"{100 * row['mean']:.1f}±{100 * row['std']:.1f}%"
        print(f"{task}\t{seeds}\t{metric}")
    overall = result["overall"]
    if overall["mean"] is not None:
        print(f"overall ({overall['count']} task-seeds)\t{100 * overall['mean']:.1f}±{100 * overall['std']:.1f}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", required=True, help="one task name or 'all'")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--weight-file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2"))
    parser.add_argument("--repeats", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--denoise-steps", type=int, choices=range(1, 101))
    parser.add_argument("--replan", type=int, choices=range(1, 33))
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks_config = load_tasks()
    if args.task != "all" and args.task not in tasks_config:
        raise ValueError(f"Unknown task {args.task!r}; choose from {', '.join(tasks_config)} or all")
    selected = list(tasks_config) if args.task == "all" else [args.task]
    output = args.output.expanduser().resolve()
    if args.summarize_only:
        print_summary(aggregate(output, selected, args.seeds, args.repeats))
        return
    runtime = load_runtime(args.config.expanduser().resolve())
    for task_name in selected:
        task = tasks_config[task_name]
        checkpoint = checkpoint_for(args, task_name)
        weight = select_weight(checkpoint, args.weight_file)
        denoise_steps = args.denoise_steps or task["denoise_steps"]
        replan = args.replan or task["replan"]
        for seed in args.seeds:
            seed_root = output / task_name / f"seed_{seed}"
            validate_resume(
                seed_root,
                {
                    "task": task_name,
                    "seed": seed,
                    "persistent": True,
                    "checkpoint": str(checkpoint),
                    "max_steps": task["max_steps"],
                    "denoise_steps": denoise_steps,
                    "replan": replan,
                    "inference_mode": "idm",
                    "weight_file": weight,
                },
            )
            done = completed_repeats(seed_root, task["gym_task"], seed, args.repeats)
            missing = [repeat for repeat in range(args.repeats) if repeat not in done]
            if not missing:
                print(f"skip complete {task_name} seed={seed}")
                continue
            print(f"run {task_name} seed={seed} repeats={','.join(map(str, missing))}", flush=True)
            env = task_env(
                runtime,
                task_name,
                task,
                checkpoint,
                weight,
                seed_root,
                seed,
                args.repeats,
                missing,
                denoise_steps,
                replan,
            )
            subprocess.run(["bash", str(HERE / "run_seed.sh")], env=env, check=True)
    print_summary(aggregate(output, selected, args.seeds, args.repeats))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, PermissionError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
