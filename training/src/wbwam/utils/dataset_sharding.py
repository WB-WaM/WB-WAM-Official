"""Dir-level balanced dataset sharding for multi-node pretraining.

Instead of assigning entire embodiments to nodes (which causes severe load
imbalance when embodiment sizes vary drastically), this module enumerates all
individual dataset directories, reads their total_frames from meta/info.json,
and uses greedy largest-first bin-packing to balance total frames across shard bins.

WBWAM uses this before dataset construction: the config is narrowed to the
dataset_dirs assigned to the current shard group, then MixtureLerobotDataset
can initialize only that subset.
"""

import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardTopology:
    """Topology for group-level dataset sharding.

    `num_bins` is the number of distinct dataset shards. `group_id` selects the
    shard owned by this node group. The sampler fields are local to that group,
    because only ranks that see the same dataset shard should stride-partition
    samples together.
    """

    num_bins: int
    group_id: int
    node_rank_in_group: int
    sampler_num_replicas: int
    sampler_rank: int


def compute_shard_topology(
    num_nodes: int,
    node_rank: int,
    local_world_size: int,
    local_rank: int,
    shard_group_size: int,
) -> ShardTopology:
    """Derive dataset-shard and sampler-local indices from node topology."""
    if shard_group_size < 1:
        raise ValueError(f"shard_group_size must be >= 1, got {shard_group_size}")
    if shard_group_size > num_nodes:
        raise ValueError(
            f"shard_group_size ({shard_group_size}) cannot exceed num_nodes ({num_nodes})"
        )
    if num_nodes % shard_group_size != 0:
        raise ValueError(
            f"num_nodes ({num_nodes}) must be divisible by shard_group_size ({shard_group_size})"
        )

    num_bins = num_nodes // shard_group_size
    group_id = node_rank // shard_group_size
    node_rank_in_group = node_rank % shard_group_size
    sampler_num_replicas = shard_group_size * local_world_size
    sampler_rank = node_rank_in_group * local_world_size + local_rank

    return ShardTopology(
        num_bins=num_bins,
        group_id=group_id,
        node_rank_in_group=node_rank_in_group,
        sampler_num_replicas=sampler_num_replicas,
        sampler_rank=sampler_rank,
    )


@dataclass
class DirInfo:
    """Metadata for a single dataset directory before bin-packing."""

    emb_name: str
    group_idx: int
    dir_path: str
    total_frames: int


@dataclass
class GroupAssignment:
    """Assigned directories for one original dataset_group in this shard."""

    group_idx: int
    dir_paths: List[str]


def _extract_feature_last_dims(info: Dict[str, Any]) -> Dict[str, int]:
    """Extract last-dimension size for each feature from info.json."""
    dims: Dict[str, int] = {}
    for key, feat in (info.get("features") or {}).items():
        shape = feat.get("shape") if isinstance(feat, dict) else None
        if isinstance(shape, list) and len(shape) > 0 and isinstance(shape[-1], int):
            dims[key] = int(shape[-1])
    return dims


def _read_info_json(dir_path: str) -> Tuple[int, Dict[str, int]]:
    """Read total_frames + feature dims from {dir_path}/meta/info.json."""
    info_path = Path(dir_path) / "meta" / "info.json"
    try:
        with open(info_path, "r") as f:
            info = json.load(f)
        return int(info.get("total_frames", 0)), _extract_feature_last_dims(info)
    except Exception as e:
        logger.warning(f"Failed to read {info_path}: {e}, using total_frames=0")
        return 0, {}


def _collect_shape_requirements(emb_cfg) -> List[Dict[str, Any]]:
    """Collect shape constraints from action/state shape_meta entries."""
    requirements: List[Dict[str, Any]] = []
    shape_meta = emb_cfg.get("shape_meta", {})
    for section in ("action", "state"):
        metas = shape_meta.get(section, [])
        for meta in metas:
            lerobot_key = meta.get("lerobot_key")
            start = meta.get("start_index")
            width = meta.get("raw_shape")
            key = meta.get("key", "<unknown>")
            if not isinstance(lerobot_key, str):
                continue
            if not isinstance(start, int):
                continue
            if not isinstance(width, int):
                continue
            requirements.append(
                {
                    "section": section,
                    "key": key,
                    "lerobot_key": lerobot_key,
                    "start_index": int(start),
                    "raw_shape": int(width),
                }
            )
    return requirements


def _validate_meta_shape_requirements(
    all_entries: List[Tuple[str, int, str]],
    info_results: List[Tuple[int, Dict[str, int]]],
    emb_requirements: Dict[str, List[Dict[str, Any]]],
    *,
    max_errors: int,
) -> None:
    """Validate shape_meta slicing constraints against dataset info.json features."""
    errors: List[str] = []

    for (emb_name, _group_idx, dir_path), (_total_frames, feature_dims) in zip(
        all_entries, info_results
    ):
        requirements = emb_requirements.get(emb_name, [])
        if not requirements:
            continue

        for req in requirements:
            lerobot_key = req["lerobot_key"]
            feature_dim: Optional[int] = feature_dims.get(lerobot_key)
            if feature_dim is None:
                errors.append(
                    f"[{emb_name}] {dir_path}: missing feature '{lerobot_key}' in info.json"
                )
            else:
                end = req["start_index"] + req["raw_shape"]
                if end > feature_dim:
                    errors.append(
                        f"[{emb_name}] {dir_path}: {req['section']}.{req['key']} expects "
                        f"slice [{req['start_index']}:{end}] from '{lerobot_key}' "
                        f"(dim={feature_dim})"
                    )

            if len(errors) >= max_errors:
                break
        if len(errors) >= max_errors:
            break

    if errors:
        msg = "\n  - " + "\n  - ".join(errors)
        raise ValueError(
            "[Dataset Precheck] shape_meta is incompatible with dataset info.json. "
            f"Showing up to {max_errors} errors:{msg}"
        )


def read_all_dir_sizes(
    embodiment_datasets_cfg,
    *,
    validate_meta_shapes: bool = False,
    max_validation_errors: int = 20,
) -> List[DirInfo]:
    """Enumerate all mixture dataset_dirs and read their sizes in parallel.

    Args:
        embodiment_datasets_cfg: OmegaConf dict of embodiment datasets config

    Returns:
        Sorted list of DirInfo (sorted by emb_name, group_idx, dir_path for determinism)
    """
    all_entries = []
    emb_dir_counts = {}  # emb_name -> num dirs (for logging)
    emb_requirements: Dict[str, List[Dict[str, Any]]] = {}
    for emb_name in sorted(embodiment_datasets_cfg.keys()):
        emb_cfg = embodiment_datasets_cfg[emb_name]
        emb_requirements[emb_name] = _collect_shape_requirements(emb_cfg)
        dataset_groups = emb_cfg.get("dataset_groups", [])
        count_before = len(all_entries)
        for group_idx, group in enumerate(dataset_groups):
            dataset_dirs = group.get("dataset_dirs", [])
            for dir_path in dataset_dirs:
                all_entries.append((emb_name, group_idx, str(dir_path)))
        emb_dir_counts[emb_name] = len(all_entries) - count_before

    if not all_entries:
        raise ValueError(
            "[Dataset Sharding] No dataset_dirs found in data.embodiment_datasets."
        )

    logger.info(
        f"[Dataset Sharding] Enumerated {len(emb_dir_counts)} embodiments, "
        f"{len(all_entries)} dataset dirs total"
    )
    for emb_name, count in emb_dir_counts.items():
        logger.info(f"[Dataset Sharding]   {emb_name}: {count} dirs")

    # Read total_frames in parallel via thread pool (I/O bound on NAS)
    logger.info(
        f"[Dataset Sharding] Reading meta/info.json for {len(all_entries)} dirs (32 threads)..."
    )
    t0 = time.time()
    dir_paths = [e[2] for e in all_entries]
    with ThreadPoolExecutor(max_workers=32) as executor:
        info_results = list(executor.map(_read_info_json, dir_paths))
    elapsed = time.time() - t0
    logger.info(f"[Dataset Sharding] Read {len(all_entries)} info.json files in {elapsed:.1f}s")

    if validate_meta_shapes:
        t1 = time.time()
        _validate_meta_shape_requirements(
            all_entries,
            info_results,
            emb_requirements,
            max_errors=max_validation_errors,
        )
        logger.info(f"[Dataset Precheck] shape_meta validation passed in {time.time() - t1:.1f}s")

    all_dirs = []
    failed_count = 0
    for (emb_name, group_idx, dir_path), (total_frames, _feature_dims) in zip(
        all_entries, info_results
    ):
        if total_frames == 0:
            failed_count += 1
        all_dirs.append(
            DirInfo(
                emb_name=emb_name,
                group_idx=group_idx,
                dir_path=dir_path,
                total_frames=total_frames,
            )
        )

    # Sort deterministically
    all_dirs.sort(key=lambda d: (d.emb_name, d.group_idx, d.dir_path))

    total_frames = sum(d.total_frames for d in all_dirs)
    logger.info(
        f"[Dataset Sharding] Total: {len(all_dirs)} dirs, {total_frames} frames"
        + (f" ({failed_count} dirs with 0 frames)" if failed_count else "")
    )

    # Per-embodiment frame summary
    emb_frames: Dict[str, int] = defaultdict(int)
    for d in all_dirs:
        emb_frames[d.emb_name] += d.total_frames
    for emb_name in sorted(emb_frames.keys()):
        logger.info(
            f"[Dataset Sharding]   {emb_name}: "
            f"{emb_dir_counts[emb_name]} dirs, {emb_frames[emb_name]} frames"
        )

    return all_dirs


def bin_pack_dirs(
    all_dirs: List[DirInfo],
    num_nodes: int,
    node_rank: int,
) -> Tuple[Dict[str, List[GroupAssignment]], Dict[int, Dict[str, Any]]]:
    """Stratified greedy bin-packing of dataset dirs across shard bins.

    Dirs are first grouped by (embodiment, dataset_group). Each dataset_group
    is then spread across bins independently, so every bin gets a piece of each
    group when the group has enough dirs. Ties are broken by current global raw
    frame load to keep memory/init load balanced.

    Args:
        all_dirs: List of DirInfo from read_all_dir_sizes
        num_nodes: Number of shard bins.
        node_rank: Current bin/group id.

    Returns:
        Tuple of:
        - assigned: Dict[emb_name, List[GroupAssignment]] for this bin
        - stats: Dict[bin_id, {num_dirs, total_frames, embodiments}] for all bins
    """
    if not all_dirs:
        raise ValueError("[Dataset Sharding] Cannot bin-pack an empty dataset_dirs list.")
    if num_nodes <= 0:
        raise ValueError(f"[Dataset Sharding] num_bins must be > 0, got {num_nodes}.")
    if node_rank < 0 or node_rank >= num_nodes:
        raise ValueError(
            f"[Dataset Sharding] bin id must be in [0, {num_nodes}), got {node_rank}."
        )

    # Track assignments per shard bin.
    node_assignments: Dict[int, List[DirInfo]] = {nid: [] for nid in range(num_nodes)}
    global_loads = [0 for _ in range(num_nodes)]

    dirs_by_group: Dict[Tuple[str, int], List[DirInfo]] = defaultdict(list)
    for dir_info in all_dirs:
        dirs_by_group[(dir_info.emb_name, dir_info.group_idx)].append(dir_info)

    group_items = sorted(
        dirs_by_group.items(),
        key=lambda item: (
            -sum(d.total_frames for d in item[1]),
            item[0][0],
            item[0][1],
        ),
    )
    undersized_groups = 0

    for (_emb_name, _group_idx), group_dirs in group_items:
        if len(group_dirs) < num_nodes:
            undersized_groups += 1
        sorted_group_dirs = sorted(
            group_dirs,
            key=lambda d: (-d.total_frames, d.emb_name, d.group_idx, d.dir_path),
        )
        group_counts = [0 for _ in range(num_nodes)]
        group_loads = [0 for _ in range(num_nodes)]

        for dir_info in sorted_group_dirs:
            # Prefer bins that have fewer dirs from this dataset_group. Once the
            # group is covered broadly, use global raw load as the main
            # tie-breaker so memory/init load stays balanced.
            bin_id = min(
                range(num_nodes),
                key=lambda nid: (
                    group_counts[nid],
                    global_loads[nid],
                    group_loads[nid],
                    nid,
                ),
            )
            node_assignments[bin_id].append(dir_info)
            group_counts[bin_id] += 1
            group_loads[bin_id] += dir_info.total_frames
            global_loads[bin_id] += dir_info.total_frames

    # Compute stats for all shard bins.
    stats: Dict[int, Dict[str, Any]] = {}
    for nid in range(num_nodes):
        dirs = node_assignments[nid]
        stats[nid] = {
            "num_dirs": len(dirs),
            "total_frames": sum(d.total_frames for d in dirs),
            "embodiments": sorted(set(d.emb_name for d in dirs)),
        }

    # Log balance quality
    all_frames = [stats[nid]["total_frames"] for nid in range(num_nodes)]
    min_frames, max_frames = min(all_frames), max(all_frames)
    imbalance = max_frames / min_frames if min_frames > 0 else float("inf")
    logger.info(
        f"[Dataset Sharding] Bin-packing done: {len(all_dirs)} dirs -> {num_nodes} bins, "
        f"imbalance ratio: {imbalance:.3f}x "
        f"(min={min_frames}, max={max_frames})"
    )
    if undersized_groups > 0:
        logger.warning(
            "[Dataset Sharding] %d/%d dataset groups have fewer dirs than bins; "
            "they cannot cover every shard bin.",
            undersized_groups,
            len(group_items),
        )

    # Build grouped assignments for this shard bin.
    group_map: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for d in node_assignments[node_rank]:
        group_map[(d.emb_name, d.group_idx)].append(d.dir_path)

    # Convert to Dict[emb_name, List[GroupAssignment]]
    assigned: Dict[str, List[GroupAssignment]] = defaultdict(list)
    for (emb_name, group_idx), dir_paths in sorted(group_map.items()):
        assigned[emb_name].append(
            GroupAssignment(
                group_idx=group_idx,
                dir_paths=sorted(dir_paths),
            )
        )

    # Log this shard bin's assignment summary.
    logger.info(
        f"[Dataset Sharding] Bin {node_rank} assignment: "
        f"{len(assigned)} embodiments, "
        f"{sum(len(ga) for ga in assigned.values())} dataset groups"
    )
    for emb_name in sorted(assigned.keys()):
        groups = assigned[emb_name]
        total_dirs = sum(len(g.dir_paths) for g in groups)
        logger.info(
            f"[Dataset Sharding]   Bin {node_rank} <- {emb_name}: "
            f"{total_dirs} dirs across {len(groups)} dataset groups"
        )

    return dict(assigned), stats


def apply_assigned_dataset_dirs(data_cfg, assigned: Dict[str, List[GroupAssignment]], group_id: int) -> None:
    """Apply a bin-packed assignment to wbwam mixture data config in-place.

    This preserves the original embodiment name and dataset_group boundaries,
    only replacing each group's dataset_dirs with the subset assigned to this
    shard. Empty embodiments/groups are removed before Hydra instantiates the
    dataset.
    """
    all_emb_names = sorted(data_cfg.embodiment_datasets.keys())
    local_emb_set = set(assigned.keys())
    if not local_emb_set:
        raise RuntimeError(
            f"[Dataset Sharding] Shard group {group_id} received no dataset dirs. "
            "Reduce shard bins or check dataset config."
        )

    removed_embs = set(all_emb_names) - local_emb_set
    if removed_embs:
        logger.info(
            "[Dataset Sharding] Shard group %s: removing %d embodiments with 0 dirs: %s",
            group_id,
            len(removed_embs),
            sorted(removed_embs),
        )
    logger.info(
        "[Dataset Sharding] Shard group %s: keeping %d embodiments: %s",
        group_id,
        len(local_emb_set),
        sorted(local_emb_set),
    )

    OmegaConf.set_struct(data_cfg, False)
    for emb_name in list(data_cfg.embodiment_datasets.keys()):
        if emb_name not in local_emb_set:
            del data_cfg.embodiment_datasets[emb_name]
            continue

        assigned_groups = {a.group_idx: a.dir_paths for a in assigned[emb_name]}
        emb_cfg = data_cfg.embodiment_datasets[emb_name]
        orig_num_groups = len(emb_cfg.dataset_groups)
        new_groups = []
        for group_idx, group in enumerate(emb_cfg.dataset_groups):
            if group_idx not in assigned_groups:
                continue
            orig_dirs = len(group.dataset_dirs)
            group.dataset_dirs = assigned_groups[group_idx]
            new_groups.append(group)
            logger.info(
                "[Dataset Sharding] Shard group %s: %s dataset_group[%d]: %d -> %d dirs",
                group_id,
                emb_name,
                group_idx,
                orig_dirs,
                len(assigned_groups[group_idx]),
            )
        removed_groups = orig_num_groups - len(new_groups)
        if removed_groups > 0:
            logger.info(
                "[Dataset Sharding] Shard group %s: %s: removed %d/%d empty dataset groups",
                group_id,
                emb_name,
                removed_groups,
                orig_num_groups,
            )
        emb_cfg.dataset_groups = new_groups
    OmegaConf.set_struct(data_cfg, True)

    if data_cfg.get("processors", None):
        OmegaConf.set_struct(data_cfg, False)
        local_type_set = {
            str(data_cfg.embodiment_datasets[emb_name].embodiment_type)
            for emb_name in data_cfg.embodiment_datasets.keys()
        }
        removed_procs = []
        for key in list(data_cfg.processors.keys()):
            if key not in local_type_set:
                del data_cfg.processors[key]
                removed_procs.append(key)
        OmegaConf.set_struct(data_cfg, True)
        if removed_procs:
            logger.info(
                "[Dataset Sharding] Shard group %s: removed %d processors: %s",
                group_id,
                len(removed_procs),
                removed_procs,
            )
        logger.info(
            "[Dataset Sharding] Shard group %s: keeping %d processors: %s",
            group_id,
            len(data_cfg.processors),
            sorted(data_cfg.processors.keys()),
        )


def _default_sharding_metadata(topology, shard_group_size: int) -> dict:
    return {
        "enabled": False,
        # Global process topology from the launcher.
        "world_size": int(topology.world_size),
        "rank": int(topology.rank),
        # Per-node topology.
        "local_world_size": int(topology.local_world_size),
        "local_rank": int(topology.local_rank),
        "num_nodes": int(topology.num_nodes),
        "node_rank": int(topology.node_rank),
        # Dataset sharding topology. num_bins is the number of distinct
        # dataset shards; each shard may be shared by shard_group_size nodes.
        "shard_group_size": int(shard_group_size),
        "num_bins": 1,
        "group_id": 0,
        "node_rank_in_group": 0,
        # Sampler-local topology. None means Trainer should use global
        # Accelerator topology because dataset sharding is disabled.
        "sampler_num_replicas": None,
        "sampler_rank": None,
    }


def configure_dataset_sharding(cfg, topology) -> dict:
    """Configure directory-level dataset sharding before dataset construction."""
    shard_datasets_by_node = bool(cfg.get("shard_datasets_by_node", False))
    shard_group_size = int(cfg.get("shard_group_size", 1))
    metadata = _default_sharding_metadata(topology, shard_group_size)

    if topology.local_rank == 0:
        logger.info(
            "[Node-level dataset sharding topology] "
            "local_world_size=%d local_rank=%d num_nodes=%d node_rank=%d",
            topology.local_world_size,
            topology.local_rank,
            topology.num_nodes,
            topology.node_rank,
        )

    if shard_datasets_by_node and topology.num_nodes <= 1:
        logger.info("shard_datasets_by_node is enabled but only 1 node detected, disabling.")
        return metadata
    if not shard_datasets_by_node:
        return metadata

    shard_topo = compute_shard_topology(
        num_nodes=topology.num_nodes,
        node_rank=topology.node_rank,
        local_world_size=topology.local_world_size,
        local_rank=topology.local_rank,
        shard_group_size=shard_group_size,
    )
    if shard_topo.num_bins <= 1:
        logger.info(
            "shard_datasets_by_node enabled but shard_group_size=%d makes num_bins=1 "
            "on %d nodes, disabling.",
            shard_group_size,
            topology.num_nodes,
        )
        return metadata

    if not cfg.data.get("embodiment_datasets", None):
        raise AssertionError("shard_datasets_by_node requires data.embodiment_datasets.")
    if not cfg.data.get("pretrained_norm_stats", None):
        raise AssertionError(
            "shard_datasets_by_node requires data.pretrained_norm_stats "
            "(pre-computed global stats)."
        )
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "configure_dataset_sharding requires torch.distributed to be initialized "
            "when shard_datasets_by_node is enabled."
        )

    all_emb_names = sorted(cfg.data.embodiment_datasets.keys())
    logger.info(
        "[Dataset Sharding] Starting dir-level bin-packing for %d bins "
        "(num_nodes=%d, shard_group_size=%d), %d embodiments",
        shard_topo.num_bins,
        topology.num_nodes,
        shard_group_size,
        len(all_emb_names),
    )

    if topology.rank == 0:
        logger.info("[Dataset Sharding] Rank 0: reading all meta/info.json...")
        all_dirs = read_all_dir_sizes(cfg.data.embodiment_datasets)
        logger.info("[Dataset Sharding] Rank 0: broadcasting %d DirInfo to all ranks", len(all_dirs))
    else:
        all_dirs = None
        logger.info("[Dataset Sharding] Rank %d: waiting for broadcast...", topology.rank)

    container = [all_dirs]
    torch.distributed.broadcast_object_list(container, src=0)
    all_dirs = container[0]
    logger.info(
        "[Dataset Sharding] Rank %d: received %d DirInfo via broadcast",
        topology.rank,
        len(all_dirs),
    )

    assigned, shard_stats = bin_pack_dirs(all_dirs, shard_topo.num_bins, shard_topo.group_id)
    if topology.local_rank == 0 and shard_topo.node_rank_in_group == 0:
        for bin_id in range(shard_topo.num_bins):
            ns = shard_stats[bin_id]
            logger.info(
                "[Dataset Sharding] Bin %d: %d dirs, %d frames, %d embodiments%s",
                bin_id,
                ns["num_dirs"],
                ns["total_frames"],
                len(ns["embodiments"]),
                " <-- THIS GROUP" if bin_id == shard_topo.group_id else "",
            )

    apply_assigned_dataset_dirs(cfg.data, assigned, shard_topo.group_id)

    metadata.update(
        {
            "enabled": True,
            "num_bins": int(shard_topo.num_bins),
            "group_id": int(shard_topo.group_id),
            "node_rank_in_group": int(shard_topo.node_rank_in_group),
            # These are consumed by Trainer's ResumableDistributedSampler. They
            # are local to the shard group, not the global world.
            "sampler_num_replicas": int(shard_topo.sampler_num_replicas),
            "sampler_rank": int(shard_topo.sampler_rank),
        }
    )
    return metadata


def postprocess_sharded_datasets(train_ds, val_ds, sharding_metadata: dict) -> None:
    """Synchronize mixture weights and cap lengths after sharded dataset construction.

    This runs after cfg.data has been sharded and datasets are built, but before
    Trainer creates the sampler. It keeps all ranks at a common effective length
    so Trainer's rank-consistency check and sampler math remain valid.
    """
    if not bool(sharding_metadata.get("enabled", False)):
        return

    seen = set()
    for name, dataset in (("train", train_ds), ("val", val_ds)):
        if id(dataset) in seen:
            continue
        seen.add(id(dataset))
        for method_name in ("sync_weights_for_sharding", "cap_length_for_sharding"):
            if not hasattr(dataset, method_name):
                raise TypeError(
                    f"Sharded {name} dataset must implement `{method_name}`; got {type(dataset)}."
                )
        logger.info("[Dataset Sharding] Postprocess %s dataset weights/length.", name)
        dataset.sync_weights_for_sharding()
        dataset.cap_length_for_sharding()
