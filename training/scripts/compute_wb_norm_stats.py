#!/usr/bin/env python
"""Compute WB dataset normalization stats without constructing the model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from wbwam.datasets.normalization import save_dataset_stats_to_json
from wbwam.datasets.wb_archive import resolve_wb_dataset_configs
from wbwam.datasets.wb_dataset import (
    WB_STATS_VERSION,
    WBDataset,
    _global_wb_stats_from_arrays,
    _load_bad_episode_manifest,
    _slices_metadata,
)
from wbwam.utils import misc
from wbwam.utils.config_resolvers import register_default_resolvers
from wbwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value


def _dataset_kwargs(data_cfg: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": str(entry["root"]),
        "name": str(entry.get("name") or Path(str(entry["root"])).name),
        "num_frames": int(data_cfg.get("num_frames", 33)),
        "action_video_freq_ratio": int(data_cfg.get("action_video_freq_ratio", 8)),
        "video_size": data_cfg.get("video_size", [224, 224]),
        "camera_key": str(
            entry.get(
                "camera_key",
                data_cfg.get("camera_key", "observation.images.stereo_left"),
            )
        ),
        "state_dim": int(data_cfg.get("state_dim", 110)),
        "action_dim": int(data_cfg.get("action_dim", 136)),
        "raw_state_dim": data_cfg.get("raw_state_dim"),
        "raw_action_dim": data_cfg.get("raw_action_dim"),
        "dataset_format": entry.get("dataset_format", data_cfg.get("dataset_format", "sealed")),
        "state_key": entry.get("state_key", data_cfg.get("state_key", "observation.state")),
        "action_key": entry.get("action_key", data_cfg.get("action_key", "action")),
        "state_mask_key": entry.get("state_mask_key", data_cfg.get("state_mask_key", "state_mask_110")),
        "action_mask_key": entry.get("action_mask_key", data_cfg.get("action_mask_key", "action_mask_136")),
        "state_slices": data_cfg.get("state_slices"),
        "action_slices": data_cfg.get("action_slices"),
        "action_representation": data_cfg.get("action_representation", "absolute"),
        "relative_joint_ranges": data_cfg.get("relative_joint_ranges"),
        "relative_action_reference": data_cfg.get("relative_action_reference", "observation"),
        "val_set_proportion": float(data_cfg.get("val_set_proportion", 0.05)),
        "val_split_mode": entry.get("val_split_mode", data_cfg.get("val_split_mode", "sequential")),
        "exclude_episode_ranges": entry.get(
            "exclude_episode_ranges",
            data_cfg.get("exclude_episode_ranges"),
        ),
        "exclude_episode_indices": entry.get(
            "exclude_episode_indices",
            data_cfg.get("exclude_episode_indices"),
        ),
        "is_training_set": True,
        # Norm stats only need parquet state/action/masks. Avoid text/video work.
        "text_embedding_cache_dir": None,
        "text_embedding_cache_suffix": data_cfg.get("text_embedding_cache_suffix", "wan22ti2v5b"),
        "use_text_embed_cache": False,
        "context_len": int(data_cfg.get("context_len", 128)),
        "instruction_template": data_cfg.get("instruction_template"),
        "instruction_config": entry.get("instruction"),
        "include_robot_metadata_in_prompt": bool(data_cfg.get("include_robot_metadata_in_prompt", True)),
        "override_instruction": data_cfg.get("override_instruction"),
        "video_backend": None,
        "tolerance_s": data_cfg.get("tolerance_s"),
        # Filtering can split one parquet shard into many adjacent segments.
        # Cache only the current shard to avoid rereading it for every segment.
        "shard_cache_size": 1,
        "strict_video_decode": bool(data_cfg.get("strict_video_decode", False)),
        "stats_downsample_rate": int(
            entry.get("stats_downsample_rate", data_cfg.get("stats_downsample_rate")) or 1
        ),
    }


def _load_one(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    data_cfg, entry = payload
    ds = WBDataset(**_dataset_kwargs(data_cfg, entry))
    return {
        "name": ds.name,
        "root": str(ds.root),
        "weight": float(entry.get("weight", 1.0)),
        "length": len(ds),
        "state_slices": _slices_metadata(ds.state_slices),
        "action_slices": _slices_metadata(ds.action_slices),
        "val_split_mode": ds.val_split_mode,
        "exclude_episode_ranges": [list(item) for item in ds.exclude_episode_ranges],
        "exclude_episode_indices_count": len(ds.exclude_episode_indices),
        "stats_downsample_rate": ds.stats_downsample_rate,
        "arrays": ds.get_stats_arrays(),
    }


def _normalized_weights(
    lengths: list[int], raw_weights: list[float], use_weight_normalization: bool
) -> list[float]:
    if not use_weight_normalization:
        return list(raw_weights)
    total_len = float(sum(lengths))
    denom = sum(float(length) * float(weight) for length, weight in zip(lengths, raw_weights))
    if denom <= 0:
        raise ValueError("Invalid WB dataset weights: weighted length denominator <= 0.")
    scale = total_len / denom
    return [scale * float(weight) for weight in raw_weights]


def _metadata(data_cfg: dict[str, Any], results: list[dict[str, Any]], weights: list[float]) -> dict[str, Any]:
    return {
        "wb_norm_stats_version": WB_STATS_VERSION,
        "names": [item["name"] for item in results],
        "roots": [item["root"] for item in results],
        "raw_weights": [float(item["weight"]) for item in results],
        "normalized_weights": list(weights),
        "dataset_lengths": [int(item["length"]) for item in results],
        "norm_weights_applied": False,
        "sampling_weights_applied_to_norm": False,
        "state_dim": int(data_cfg.get("state_dim", 110)),
        "action_dim": int(data_cfg.get("action_dim", 136)),
        "action_representation": str(data_cfg.get("action_representation", "absolute")),
        "relative_joint_ranges": [list(item) for item in data_cfg.get("relative_joint_ranges") or []],
        "relative_action_reference": str(data_cfg.get("relative_action_reference", "observation")),
        "proprio_dim": int(data_cfg.get("proprio_dim", data_cfg.get("state_dim", 110))),
        "action_target_dim": int(data_cfg.get("action_target_dim", data_cfg.get("action_dim", 136))),
        "model_dim_indices": data_cfg.get("model_dim_indices"),
        "state_model_dim_indices": data_cfg.get("state_model_dim_indices"),
        "action_model_dim_indices": data_cfg.get("action_model_dim_indices"),
        "dataset_format": data_cfg.get("dataset_format", "sealed"),
        "state_key": data_cfg.get("state_key", "observation.state"),
        "action_key": data_cfg.get("action_key", "action"),
        "state_mask_key": data_cfg.get("state_mask_key", "state_mask_110"),
        "action_mask_key": data_cfg.get("action_mask_key", "action_mask_136"),
        "raw_state_dim": data_cfg.get("raw_state_dim"),
        "raw_action_dim": data_cfg.get("raw_action_dim"),
        "state_slices": [item["state_slices"] for item in results],
        "action_slices": [item["action_slices"] for item in results],
        "val_split_modes": [item["val_split_mode"] for item in results],
        "exclude_episode_ranges": [item["exclude_episode_ranges"] for item in results],
        "exclude_episode_indices_counts": [int(item["exclude_episode_indices_count"]) for item in results],
        "bad_episodes_path": data_cfg.get("bad_episodes_path"),
        "dataset_stats_downsample_rates": [int(item["stats_downsample_rate"]) for item in results],
        "num_frames": int(data_cfg.get("num_frames", 33)),
        "action_horizon": int(data_cfg.get("num_frames", 33)) - 1,
        "action_video_freq_ratio": int(data_cfg.get("action_video_freq_ratio", 8)),
        "mask_aware": True,
        "norm_default_mode": str(data_cfg.get("norm_default_mode", "q01/q99")),
        "use_stepwise_action_norm": bool(data_cfg.get("use_stepwise_action_norm", True)),
        "stats_downsample_rate": int(data_cfg.get("stats_downsample_rate") or 1),
        "stats_compute": {
            "script": "scripts/compute_wb_norm_stats.py",
            "parallel_by_dataset": True,
            "executor": "thread",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-file", default="dataset_stats.json")
    parser.add_argument("--stats-downsample-rate", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _dataset_payloads(
    data_cfg: dict[str, Any],
    datasets: list[dict[str, Any]] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if datasets is None:
        datasets = resolve_wb_dataset_configs(
            archives=data_cfg.get("archives"),
            datasets=data_cfg.get("datasets"),
        )
    bad_episode_indices = _load_bad_episode_manifest(data_cfg.get("bad_episodes_path"))
    payloads = []
    for raw_entry in datasets:
        entry = dict(raw_entry)
        name = str(entry.get("name") or Path(str(entry["root"])).name)
        entry.setdefault("exclude_episode_indices", bad_episode_indices.get(name, ()))
        payloads.append((data_cfg, entry))
    return payloads


def main() -> None:
    args = _parse_args()
    register_default_resolvers()
    setup_logging()

    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=args.config_name, overrides=list(args.overrides))

    if args.output_dir:
        OmegaConf.update(cfg, "output_dir", args.output_dir, merge=False)
    if args.stats_downsample_rate is not None:
        OmegaConf.update(cfg, "data.stats_downsample_rate", args.stats_downsample_rate, merge=False)
    OmegaConf.resolve(cfg)

    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    misc.register_work_dir(output_dir)
    output_path = output_dir / args.output_file

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    if not isinstance(data_cfg, dict):
        raise ValueError("Expected cfg.data to resolve to a mapping.")
    datasets = resolve_wb_dataset_configs(
        archives=data_cfg.get("archives"),
        datasets=data_cfg.get("datasets"),
    )
    workers = max(1, min(int(args.num_workers), len(datasets)))

    logger.info(
        "Computing WB norm stats: datasets=%d workers=%d output=%s",
        len(datasets),
        workers,
        output_path,
    )

    payloads = _dataset_payloads(data_cfg, datasets)
    results_by_index: dict[int, dict[str, Any]] = {}
    if workers == 1:
        for index, payload in enumerate(payloads):
            result = _load_one(payload)
            logger.info("Loaded stats arrays for %s frames=%d", result["name"], result["length"])
            results_by_index[index] = result
    else:
        # Large NumPy arrays stay in one address space; processes would serialize
        # and duplicate tens of gigabytes before exact-global aggregation.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_load_one, payload): index for index, payload in enumerate(payloads)}
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                logger.info("Loaded stats arrays for %s frames=%d", result["name"], result["length"])
                results_by_index[index] = result

    results = [results_by_index[index] for index in range(len(datasets))]
    lengths = [int(item["length"]) for item in results]
    raw_weights = [float(item["weight"]) for item in results]
    weights = _normalized_weights(
        lengths,
        raw_weights,
        bool(data_cfg.get("use_weight_normalization", True)),
    )
    stats = _global_wb_stats_from_arrays(
        [item["arrays"] for item in results],
        metadata=_metadata(data_cfg, results, weights),
        action_horizon=int(data_cfg.get("num_frames", 33)) - 1,
        relative_joint_ranges=(
            data_cfg.get("relative_joint_ranges")
            if data_cfg.get("action_representation", "absolute") == "relative_joint"
            else ()
        ),
        relative_action_reference=str(data_cfg.get("relative_action_reference", "observation")),
    )
    save_dataset_stats_to_json(stats, str(output_path))
    logger.info("Saved WB norm stats to %s", output_path)


if __name__ == "__main__":
    main()
