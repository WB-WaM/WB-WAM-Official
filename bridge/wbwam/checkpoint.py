"""Resolve WB-WAM exported weights and their saved training contract."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
import io
import os
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "training"


def _resolve_path(value: str | Path | None, *, base: Path = REPO_ROOT) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _checkpoint_run_root(path: Path) -> Path:
    parent = path.parent
    if parent.name == "weights":
        if parent.parent.name == "checkpoints":
            return parent.parent.parent
        return parent.parent
    return parent


def _resolve_checkpoint_path(value: str | Path | None) -> Path:
    checkpoint_path = _resolve_path(value)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"WBWAM checkpoint not found: {checkpoint_path}")
    if checkpoint_path.is_dir():
        search_dirs = (
            checkpoint_path / "checkpoints" / "weights",
            checkpoint_path / "weights",
            checkpoint_path,
        )
        candidates: list[tuple[int, Path]] = []
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for candidate in directory.glob("step_*.pt"):
                match = re.fullmatch(r"step_(\d+)\.pt", candidate.name)
                if match and candidate.is_file() and candidate.stat().st_size > 0:
                    candidates.append((int(match.group(1)), candidate))
        if not candidates:
            raise FileNotFoundError(f"no non-empty WBWAM step_*.pt checkpoint found under {checkpoint_path}")
        checkpoint_path = max(candidates, key=lambda item: item[0])[1].resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"WBWAM checkpoint must be a file or run directory: {checkpoint_path}")
    if checkpoint_path.stat().st_size == 0:
        raise ValueError(
            f"WBWAM checkpoint is empty; replace the placeholder with the real weights: {checkpoint_path}"
        )
    return checkpoint_path


@lru_cache(maxsize=1)
def load_training_environment() -> None:
    """Apply optional local settings, then portable bridge asset defaults."""
    if (TRAINING_ROOT / ".env").is_file():
        from training.scripts.load_env import main

        # The loader updates os.environ and emits shell exports. Python callers
        # only need the environment; do not print machine-local configuration.
        with redirect_stdout(io.StringIO()):
            main()
    # Shared by both launchers and the isolated prompt encoder. Keep explicit
    # environment/.env settings, including offline mode, authoritative.
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(REPO_ROOT / "checkpoints" / "base_models"))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "false")


def add_training_path() -> None:
    """Use this checkout's model code, never a removed baseline package."""
    load_training_environment()
    src = str(TRAINING_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


@dataclass(frozen=True)
class CheckpointArtifacts:
    weights: Path
    config: Path
    stats: Path


def _find_metadata(weights: Path, name: str) -> Path:
    root = _checkpoint_run_root(weights)
    for path in (root / "metadata" / name, root / name):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No {name} beside {weights}; set its path explicitly in policy")


def load_training_config(policy: dict[str, Any]):
    """Load the saved resolved config, not a newly composed training preset."""
    from omegaconf import OmegaConf

    load_training_environment()
    path = _resolve_path(policy.get("config_path"))
    if path is None:
        path = _find_metadata(_resolve_checkpoint_path(policy.get("checkpoint_path")), "config.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"WB-WAM saved training config not found: {path}")
    cfg = OmegaConf.load(path)
    if not OmegaConf.is_dict(cfg) or "model" not in cfg or "data" not in cfg:
        raise ValueError(f"Expected a resolved training config with model and data: {path}")
    target = str(cfg.model.get("_target_", ""))
    allowed = {f"wbwam.runtime.create_wbwam{suffix}" for suffix in ("", "_idm", "_optional_idm")}
    if target not in allowed:
        raise ValueError(f"Unsupported WB-WAM model factory: {target!r}")
    return path, cfg


def resolve_artifacts(policy: dict[str, Any]):
    weights = _resolve_checkpoint_path(policy.get("checkpoint_path"))
    config_path, cfg = load_training_config(policy)
    stats = _resolve_path(policy.get("dataset_stats_path"))
    if stats is None:
        # Prefer colocated exported metadata; the training path can refer to
        # a different machine. A separately stored stats file may be explicit.
        stats = _find_metadata(weights, "dataset_stats.json")
    if not stats.is_file():
        raise FileNotFoundError(f"WB-WAM dataset_stats.json not found: {stats}")
    return CheckpointArtifacts(weights, config_path, stats), cfg


def validate_deployment_config(policy: dict[str, Any]):
    """CPU-only semantic/layout checks before allocating the model on GPU."""
    add_training_path()
    from wbwam.datasets.normalization import load_dataset_stats_from_json

    from bridge.wbwam.codec import WBFlatCodec

    artifacts, cfg = resolve_artifacts(policy)
    data = cfg.data
    if int(data.get("state_dim", 0)) != 72 or int(data.get("action_dim", 0)) != 72:
        raise ValueError("SONIC bridge requires WB-WAM semantic state_dim=action_dim=72")
    expected_slices = {
        "state_slices": ["states[9:38]", "states[107:110]", "states[67:87]", "states[87:107]"],
        "action_slices": ["action[104:133]", "action[133:136]", "action[64:84]", "action[84:104]"],
    }
    for name, expected in expected_slices.items():
        if list(data.get(name, [])) != expected:
            raise ValueError(f"data.{name} does not match the WB-WAM -> SONIC semantic layout")
    stats = load_dataset_stats_from_json(str(artifacts.stats))
    horizon = int(data.num_frames) - 1
    metadata = stats.get("metadata") or {}
    if "action_horizon" in metadata and int(metadata["action_horizon"]) != horizon:
        raise ValueError("dataset_stats action_horizon does not match training config")
    for category, key in (("state", "proprio"), ("action", "action")):
        for name, tensor in stats.get(category, {}).get(key, {}).items():
            if str(name).startswith("global_"):
                expected_shape = (72,)
            elif str(name).startswith("stepwise_"):
                # State stats describe one observation; action stats span the prediction horizon.
                expected_shape = (1 if category == "state" else horizon, 72)
            else:
                continue
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"dataset_stats {category}.{key}.{name} must have shape {expected_shape}")
            if not tensor.isfinite().all():
                raise ValueError(f"dataset_stats {category}.{key}.{name} contains NaN/Inf")
    codec = WBFlatCodec(data, stats)
    if int(cfg.model.get("proprio_dim", 0)) != codec.proprio_dim:
        raise ValueError("model.proprio_dim does not match data.proprio_dim")
    if int(cfg.model.action_dit_config.get("action_dim", 0)) != codec.action_target_dim:
        raise ValueError("model action_dim does not match data.action_target_dim")
    horizon = int(data.num_frames) - 1
    ratio = int(data.get("action_video_freq_ratio", 4))
    if horizon <= 0 or ratio <= 0 or horizon % ratio:
        raise ValueError("Invalid training action/video horizon")
    for name, expected in (("action_horizon", horizon), ("state_dim", 72), ("action_dim", 72)):
        if name in policy and int(policy[name]) != expected:
            raise ValueError(f"policy.{name}={policy[name]} does not match training value {expected}")
    return artifacts, cfg, codec


def format_deployment_prompt(cfg: Any, policy: dict[str, Any], task: str) -> str:
    """Share training's punctuation/whitespace rules and robot descriptions."""
    add_training_path()
    from wbwam.datasets.prompt_builder import (
        DEFAULT_WB_INSTRUCTION_TEMPLATE,
        build_configured_instruction,
    )

    instruction = {}
    archives = cfg.data.get("archives", [])
    for archive in archives:
        if str(archive.get("name", "")).lower() == "real":
            instruction = archive.get("instruction") or {}
            break
    template = str(
        policy.get("prompt_template") or cfg.data.get("instruction_template") or DEFAULT_WB_INSTRUCTION_TEMPLATE
    )
    values = {
        key: str(policy.get(key) or instruction.get(key) or "")
        for key in ("visual_description", "control_description")
    }
    if "{visual_description}" not in template and "{control_description}" not in template:
        # Older checkpoints may have task-only templates; don't add context.
        return template.format(task=task)
    if not all(values.values()):
        raise ValueError("Set policy.visual_description and policy.control_description to match training")
    return build_configured_instruction(task, template=template, **values)
