"""WB-WAM LeRobot v3 loader for sealed archives and official native50 data."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Optional

from accelerate import PartialState
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as transforms_F

from wbwam.datasets.dataset_utils import (
    VideoAugmentor as _VideoAugmentor,
    apply_relative_joint_action as _apply_relative_joint_action,
    concat_feature_slices as _concat_slices,
    list_column_to_numpy as _list_column_to_numpy,
    normalize_relative_action_reference as _normalize_relative_action_reference,
    restore_absolute_joint_action as restore_absolute_joint_action,
    right_pad_tensor_and_mask as _right_pad_tensor_and_mask,
    scatter_tensor_and_mask as _scatter_tensor_and_mask,
)
from wbwam.datasets.normalization import (
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)
from wbwam.datasets.prompt_builder import (
    DEFAULT_WB_INSTRUCTION_TEMPLATE,
    build_configured_instruction,
    build_text_prompt,
)
from wbwam.datasets.wb_archive import resolve_wb_dataset_configs
from wbwam.datasets.wb_stats import (
    WB_QUANTILE_SPEC as WB_QUANTILE_SPEC,
    WB_STATS_VERSION as WB_STATS_VERSION,
    WBFlatNormalizer as WBFlatNormalizer,
    WBStatsArrays as WBStatsArrays,
    _action_stats_from_arrays as _action_stats_from_arrays,
    _fill_quantiles_2d as _fill_quantiles_2d,
    _global_wb_stats_from_arrays as _global_wb_stats_from_arrays,
    _masked_stats_2d as _masked_stats_2d,
    _quantile_indices as _quantile_indices,
    _state_stats_from_arrays as _state_stats_from_arrays,
)
from wbwam.utils import misc
from wbwam.utils.logging_config import get_logger

logger = get_logger(__name__)

WB_PROMPT = "A video of a humanoid robot doing: {task}. Robot: {robot_type}. Configuration: {config_summary}."


def _load_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "WBDataset requires pyarrow to read WB LeRobot v3 parquet shards. "
            "Install the WBWAM training environment before constructing the dataset."
        ) from exc
    return pq


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_bad_episode_manifest(path: str | os.PathLike[str] | None) -> dict[str, frozenset[int]]:
    """Load dataset-name keyed episode exclusions from a versioned JSON manifest."""

    if path is None:
        return {}
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"WB bad episode manifest does not exist: {manifest_path}")

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"WB bad episode manifest must contain a JSON object: {manifest_path}")
    if manifest.get("schema_version") != 1:
        raise ValueError(
            "Unsupported WB bad episode manifest schema_version "
            f"{manifest.get('schema_version')!r} in {manifest_path}; expected 1."
        )
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError(f"WB bad episode manifest `datasets` must be an object: {manifest_path}")

    exclusions: dict[str, frozenset[int]] = {}
    for dataset_name, episode_indices in datasets.items():
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError(f"WB bad episode manifest contains an invalid dataset name: {dataset_name!r}")
        if not isinstance(episode_indices, list):
            raise ValueError(f"WB bad episode entries for {dataset_name!r} must be a list: {manifest_path}")
        invalid = [
            value
            for value in episode_indices
            if isinstance(value, bool) or not isinstance(value, int) or value < 0
        ]
        if invalid:
            raise ValueError(
                f"WB bad episode entries for {dataset_name!r} must be non-negative integers; "
                f"found {invalid[:5]!r}."
            )
        exclusions[dataset_name] = frozenset(episode_indices)

    logger.info(
        "Loaded WB bad episode manifest path=%s datasets=%d episodes=%d",
        manifest_path,
        len(exclusions),
        sum(len(indices) for indices in exclusions.values()),
    )
    return exclusions


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _validate_canonical_task_text(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} must be a non-empty string, got {value!r}.")
    if "_" in value:
        raise ValueError(f"{source} must not contain underscores, got {value!r}.")
    if value != " ".join(value.split()):
        raise ValueError(f"{source} must use canonical single-space whitespace, got {value!r}.")
    if not value[0].isupper():
        raise ValueError(f"{source} must start with an uppercase character, got {value!r}.")
    if value[-1] not in ".!?":
        raise ValueError(f"{source} must end with punctuation, got {value!r}.")
    return value


def _plain_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "ListConfig":
        return [_plain_config(item) for item in value]
    return value


def _to_rel_path(path: Any) -> str | None:
    if path is None:
        return None
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    text = str(path).strip()
    return text or None


_SLICE_RE = re.compile(r"^(?:(?P<name>states|state|action)\s*)?\[(?P<start>\d+):(?P<end>\d+)\]$")
_RANGE_RE = re.compile(r"^(?P<start>\d+):(?P<end>\d+)$")


def _feature_dim(features: dict[str, Any], key: str, default: int) -> int:
    feature = features.get(key) or {}
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if isinstance(shape, (list, tuple)) and shape:
        return int(shape[0])
    return int(default)


def _slice_dim(slices: list[tuple[int, int, str]]) -> int:
    return sum(end - start for start, end, _label in slices)


def _slice_range_from_text(text: str, feature_name: str) -> tuple[int, int]:
    value = text.strip()
    match = _SLICE_RE.match(value) or _RANGE_RE.match(value)
    if match is None:
        raise ValueError(
            f"Invalid {feature_name} slice `{text}`. Expected `{feature_name}[start:end]` or `start:end`."
        )
    name = match.groupdict().get("name")
    if name is not None:
        valid_names = {feature_name}
        if feature_name == "states":
            valid_names.add("state")
        if name not in valid_names:
            raise ValueError(f"Slice `{text}` belongs to `{name}`, not `{feature_name}`.")
    return int(match.group("start")), int(match.group("end"))


def _parse_slice_spec(
    spec: Any,
    *,
    feature_name: str,
    layout: dict[str, Any],
    raw_dim: int,
) -> tuple[int, int, str]:
    if isinstance(spec, dict):
        if "key" in spec:
            spec = spec["key"]
        elif "range" in spec:
            spec = spec["range"]
        else:
            if "start" not in spec or "end" not in spec:
                raise ValueError(
                    f"{feature_name}_slices dict entries must contain `key`, `range`, or both `start` and `end`."
                )
            start = int(spec["start"])
            end = int(spec["end"])
            label = str(spec.get("label") or f"{feature_name}[{start}:{end}]")
            if start < 0 or end <= start or end > raw_dim:
                raise ValueError(f"Invalid {feature_name} slice {start}:{end} for raw dim {raw_dim}.")
            return start, end, label

    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        start = int(spec[0])
        end = int(spec[1])
        label = f"{feature_name}[{start}:{end}]"
    else:
        label = str(spec).strip()
        if label in layout:
            start, end = _slice_range_from_text(label, feature_name)
        else:
            start, end = _slice_range_from_text(label, feature_name)
            label = f"{feature_name}[{start}:{end}]"

    if start < 0 or end <= start or end > raw_dim:
        raise ValueError(f"Invalid {feature_name} slice {start}:{end} for raw dim {raw_dim}.")
    return start, end, label


def _normalize_slice_specs(
    specs: Any,
    *,
    feature_name: str,
    layout: dict[str, Any],
    raw_dim: int,
) -> list[tuple[int, int, str]]:
    if specs is None:
        return [(0, raw_dim, f"{feature_name}[0:{raw_dim}]")]
    if isinstance(specs, (str, dict)):
        specs = [specs]
    slices = [_parse_slice_spec(spec, feature_name=feature_name, layout=layout, raw_dim=raw_dim) for spec in specs]
    if not slices:
        raise ValueError(f"`{feature_name}_slices` must not be empty.")
    return slices


def _is_config_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) or value.__class__.__name__ == "ListConfig"


def _normalize_model_dim_indices(
    specs: Any,
    *,
    source_dim: int,
    target_dim: int,
    feature_name: str,
) -> tuple[int, ...] | None:
    if specs is None:
        return None
    if not _is_config_sequence(specs):
        raise TypeError(f"{feature_name} model_dim_indices must be a sequence, got {type(specs).__name__}.")
    indices = tuple(int(index) for index in specs)
    if len(indices) != source_dim:
        raise ValueError(f"{feature_name} model_dim_indices must have {source_dim} entries, got {len(indices)}.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{feature_name} model_dim_indices must be unique.")
    if any(index < 0 or index >= target_dim for index in indices):
        raise ValueError(f"{feature_name} model_dim_indices must be in [0, {target_dim}), got {indices}.")
    return indices


def _normalize_shared_model_dim_indices(
    specs: Any,
    *,
    state_dim: int,
    action_dim: int,
    proprio_dim: int,
    action_target_dim: int,
) -> tuple[int, ...] | None:
    proprio_indices = _normalize_model_dim_indices(
        specs,
        source_dim=state_dim,
        target_dim=proprio_dim,
        feature_name="proprio",
    )
    action_indices = _normalize_model_dim_indices(
        specs,
        source_dim=action_dim,
        target_dim=action_target_dim,
        feature_name="action",
    )
    if proprio_indices != action_indices:
        raise ValueError("Shared model_dim_indices must be valid for both proprio and action.")
    return proprio_indices


def _normalize_model_dim_index_maps(
    shared: Any,
    state: Any,
    action: Any,
    *,
    state_dim: int,
    action_dim: int,
    proprio_dim: int,
    action_target_dim: int,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    if shared is not None and (state is not None or action is not None):
        raise ValueError("Shared and separate model_dim_indices cannot be combined.")
    if shared is not None:
        indices = _normalize_shared_model_dim_indices(
            shared,
            state_dim=state_dim,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            action_target_dim=action_target_dim,
        )
        return indices, indices
    return (
        _normalize_model_dim_indices(state, source_dim=state_dim, target_dim=proprio_dim, feature_name="state"),
        _normalize_model_dim_indices(
            action, source_dim=action_dim, target_dim=action_target_dim, feature_name="action"
        ),
    )


def _range_from_mapping(spec: Any, *, option_name: str) -> Any:
    if not isinstance(spec, Mapping):
        return spec
    if "range" in spec:
        return spec["range"]
    if "start" in spec and "end" in spec:
        return spec["start"], spec["end"]
    raise ValueError(f"{option_name} dict entries must contain `range` or both `start` and `end`.")


def _validate_non_overlapping_ranges(
    ranges: list[tuple[int, int]],
    *,
    description: str,
) -> list[tuple[int, int]]:
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"Overlapping {description}: {previous} and {current}.")
    return ranges


def _normalize_relative_joint_range_entry(spec: Any, dim: int) -> tuple[int, int]:
    if isinstance(spec, torch.Tensor):
        if spec.numel() != 2:
            raise ValueError(
                "relative_joint_ranges tensor entries must contain exactly two values, "
                f"got shape {tuple(spec.shape)}."
            )
        spec = spec.detach().cpu().reshape(-1).tolist()
    elif isinstance(spec, np.ndarray):
        if spec.size != 2:
            raise ValueError(
                "relative_joint_ranges ndarray entries must contain exactly two values, "
                f"got shape {tuple(spec.shape)}."
            )
        spec = spec.reshape(-1).tolist()

    spec = _range_from_mapping(spec, option_name="relative_joint_ranges")
    if _is_config_sequence(spec) and len(spec) == 2:
        start, end = int(spec[0]), int(spec[1])
    else:
        start, end = _slice_range_from_text(str(spec), "action")
    if start < 0 or end <= start or end > dim:
        raise ValueError(f"Invalid relative joint range {start}:{end} for action dim {dim}.")
    return start, end


def _normalize_relative_joint_ranges(specs: Any, dim: int) -> list[tuple[int, int]]:
    if specs is None:
        return []
    if isinstance(specs, (str, Mapping)):
        specs = [specs]
    ranges = [_normalize_relative_joint_range_entry(spec, dim) for spec in specs]
    return _validate_non_overlapping_ranges(
        ranges,
        description="relative joint ranges",
    )


def _normalize_episode_range_entry(spec: Any) -> tuple[int, int]:
    spec = _range_from_mapping(spec, option_name="exclude_episode_ranges")
    if _is_config_sequence(spec) and len(spec) == 2:
        start, end = int(spec[0]), int(spec[1])
    else:
        start, end = _slice_range_from_text(str(spec), "episode")
    if start < 0 or end <= start:
        raise ValueError(f"Invalid end-exclusive episode range {start}:{end}.")
    return start, end


def _normalize_episode_ranges(specs: Any) -> list[tuple[int, int]]:
    """Normalize end-exclusive episode ranges used to exclude corrupt data."""

    if specs is None:
        return []
    if isinstance(specs, (str, Mapping)) or (
        _is_config_sequence(specs)
        and len(specs) == 2
        and not isinstance(specs[0], (list, tuple, Mapping))
        and specs[0].__class__.__name__ != "ListConfig"
    ):
        specs = [specs]
    ranges = [_normalize_episode_range_entry(spec) for spec in specs]
    return _validate_non_overlapping_ranges(
        ranges,
        description="episode exclusion ranges",
    )


def _normalize_episode_indices(specs: Any) -> frozenset[int]:
    """Normalize individual episode indices used to exclude corrupt data."""

    if specs is None:
        return frozenset()
    if isinstance(specs, (str, bytes, Mapping)) or not (
        _is_config_sequence(specs) or isinstance(specs, (set, frozenset))
    ):
        raise ValueError("exclude_episode_indices must be a sequence of non-negative integers.")
    indices = []
    for value in specs:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"Invalid excluded episode index: {value!r}.")
        index = int(value)
        if index < 0:
            raise ValueError(f"Invalid excluded episode index: {index}.")
        indices.append(index)
    return frozenset(indices)


def _slices_metadata(slices: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
    return [{"start": start, "end": end, "label": label} for start, end, label in slices]


def _validate_action_stats_metadata(
    stats: dict[str, Any],
    *,
    action_representation: str,
    relative_joint_ranges: list[tuple[int, int]],
    action_dim: int,
    relative_action_reference: str = "observation",
) -> None:
    metadata = stats.get("metadata") or {}
    stats_representation = str(metadata.get("action_representation", "absolute"))
    if stats_representation != action_representation:
        raise ValueError(
            f"WB norm stats action_representation={stats_representation!r} does not match "
            f"configured {action_representation!r}."
        )
    if action_representation != "relative_joint":
        return
    stats_ranges = _normalize_relative_joint_ranges(metadata.get("relative_joint_ranges"), action_dim)
    if stats_ranges != relative_joint_ranges:
        raise ValueError(
            f"WB norm stats relative_joint_ranges={stats_ranges} do not match configured {relative_joint_ranges}."
        )
    stats_reference = _normalize_relative_action_reference(
        metadata.get("relative_action_reference", "observation")
    )
    configured_reference = _normalize_relative_action_reference(relative_action_reference)
    if stats_reference != configured_reference:
        raise ValueError(
            f"WB norm stats relative_action_reference={stats_reference!r} does not match "
            f"configured {configured_reference!r}."
        )


class _ShardCache:
    def __init__(self, root: Path, cache_size: int = 2):
        self.root = Path(root)
        self.cache_size = max(int(cache_size), 0)
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def _resolve_path(self, relpath: str) -> Path:
        path = Path(relpath)
        if path.is_absolute():
            return path
        return self.root / path

    def read_rows(self, relpath: str, row_from: int, row_to: int, columns: list[str]):
        pq = _load_pyarrow()
        row_from = int(row_from)
        row_to = int(row_to)
        if row_to <= row_from:
            raise ValueError(f"Invalid shard row range {row_from}:{row_to} for {relpath}")

        if self.cache_size > 0 and relpath in self._cache:
            table = self._cache.pop(relpath)
            self._cache[relpath] = table
        else:
            path = self._resolve_path(relpath)
            table = pq.read_table(path, columns=columns).combine_chunks()
            if self.cache_size > 0:
                self._cache[relpath] = table
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return table.slice(row_from, row_to - row_from)


class WBDataset(Dataset):
    """Single WB-WAM LeRobot v3 root."""

    _EPISODE_REQUIRED_COLUMNS = (
        "episode_index",
        "tasks",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    )

    def __init__(
        self,
        root: str,
        *,
        name: Optional[str] = None,
        num_frames: int = 33,
        action_video_freq_ratio: int = 8,
        video_size: Iterable[int] = (224, 224),
        camera_key: str = "observation.images.stereo_left",
        image_augmentation: Optional[Mapping[str, Any]] = None,
        state_dim: int = 110,
        action_dim: int = 136,
        proprio_dim: Optional[int] = None,
        action_target_dim: Optional[int] = None,
        model_dim_indices: Any = None,
        state_model_dim_indices: Any = None,
        action_model_dim_indices: Any = None,
        raw_state_dim: Optional[int] = None,
        raw_action_dim: Optional[int] = None,
        dataset_format: str = "sealed",
        state_key: str = "observation.state",
        action_key: str = "action",
        state_mask_key: Optional[str] = "state_mask_110",
        action_mask_key: Optional[str] = "action_mask_136",
        state_slices: Any = None,
        action_slices: Any = None,
        action_representation: str = "absolute",
        relative_joint_ranges: Any = None,
        relative_action_reference: str = "observation",
        val_set_proportion: float = 0.05,
        val_split_mode: str = "sequential",
        exclude_episode_ranges: Any = None,
        exclude_episode_indices: Any = None,
        is_training_set: bool = True,
        text_embedding_cache_dir: Optional[str] = None,
        text_embedding_cache_suffix: Optional[str] = "wan22ti2v5b",
        use_text_embed_cache: bool = True,
        context_len: int = 128,
        include_robot_metadata_in_prompt: bool = True,
        instruction_template: Optional[str] = None,
        instruction_config: Optional[Mapping[str, Any]] = None,
        override_instruction: Optional[str] = None,
        video_backend: Optional[str] = None,
        tolerance_s: Optional[float] = None,
        shard_cache_size: int = 2,
        strict_video_decode: bool = False,
        stats_downsample_rate: int = 1,
        return_fk_targets: bool = False,
        fk_joint_space: Optional[str] = None,
    ):
        self.root = Path(root).expanduser()
        self.name = str(name or self.root.name)
        self.num_frames = int(num_frames)
        self.action_horizon = self.num_frames - 1
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.video_sample_offsets = list(range(0, self.num_frames, self.action_video_freq_ratio))
        self.video_size = [int(x) for x in video_size]
        self.camera_key = str(camera_key)
        self.video_augmentor = _VideoAugmentor(image_augmentation, self.video_size)
        requested_state_dim = int(state_dim)
        requested_action_dim = int(action_dim)
        self.dataset_format = str(dataset_format)
        if self.dataset_format not in {"sealed", "official_lerobot_v3_1"}:
            raise ValueError(f"Unsupported WB dataset_format: {self.dataset_format}")
        self.state_key = str(state_key)
        self.action_key = str(action_key)
        self.state_mask_key = state_mask_key
        self.action_mask_key = action_mask_key
        if not self.state_key or not self.action_key:
            raise ValueError("WB state/action keys must be non-empty.")
        if self.dataset_format == "sealed" and (not state_mask_key or not action_mask_key):
            raise ValueError("Sealed WB archives require state/action mask columns.")
        self.table_columns = list(dict.fromkeys(
            key for key in (self.state_key, self.action_key, state_mask_key, action_mask_key) if key is not None
        ))
        self.val_set_proportion = float(val_set_proportion)
        self.val_split_mode = str(val_split_mode).strip().lower()
        if self.val_split_mode not in {"sequential", "task_stratified"}:
            raise ValueError(
                f"Unsupported WB val_split_mode: {val_split_mode}. "
                "Expected one of: ['sequential', 'task_stratified']."
            )
        self.exclude_episode_ranges = _normalize_episode_ranges(exclude_episode_ranges)
        self.exclude_episode_indices = _normalize_episode_indices(exclude_episode_indices)
        self.is_training_set = bool(is_training_set)
        self.text_embedding_cache_dir = None if text_embedding_cache_dir is None else str(text_embedding_cache_dir)
        self.text_embedding_cache_suffix = text_embedding_cache_suffix
        self.use_text_embed_cache = bool(use_text_embed_cache)
        self.context_len = int(context_len)
        self.include_robot_metadata_in_prompt = bool(include_robot_metadata_in_prompt)
        self.instruction_template = str(instruction_template or DEFAULT_WB_INSTRUCTION_TEMPLATE)
        self.instruction_config = None if instruction_config is None else _plain_config(instruction_config)
        self.override_instruction = override_instruction
        self.video_backend = video_backend
        self.action_representation = str(action_representation)
        if self.action_representation not in {"absolute", "relative_joint", "native"}:
            raise ValueError(f"Unsupported WB action_representation: {self.action_representation}")
        self.relative_action_reference = _normalize_relative_action_reference(relative_action_reference)
        if self.action_representation != "relative_joint" and self.relative_action_reference != "observation":
            raise ValueError("relative_action_reference is only configurable for relative_joint actions.")
        self.strict_video_decode = bool(strict_video_decode)
        self.stats_downsample_rate = max(1, int(stats_downsample_rate))
        self.normalizer: Optional[WBFlatNormalizer] = None

        self.return_fk_targets = bool(return_fk_targets)
        self.fk_joint_offset = None
        if self.return_fk_targets:
            from wbwam.models.wan22.g1_fk_loss import joint_offset

            if self.action_representation != "absolute":
                raise ValueError("FK targets require absolute action_representation")
            self.fk_joint_offset = joint_offset(fk_joint_space)

        self._validate_temporal_config()
        self._load_dataset_metadata(
            requested_state_dim=requested_state_dim,
            requested_action_dim=requested_action_dim,
            raw_state_dim=raw_state_dim,
            raw_action_dim=raw_action_dim,
            state_slices=state_slices,
            action_slices=action_slices,
            relative_joint_ranges=relative_joint_ranges,
            proprio_dim=proprio_dim,
            action_target_dim=action_target_dim,
            model_dim_indices=model_dim_indices,
            state_model_dim_indices=state_model_dim_indices,
            action_model_dim_indices=action_model_dim_indices,
        )
        self._init_video_and_episode_index(
            tolerance_s=tolerance_s,
            shard_cache_size=shard_cache_size,
        )
        self._log_loaded_dataset()

    def _validate_temporal_config(self) -> None:
        if self.num_frames <= 1:
            raise ValueError(f"`num_frames` must be > 1, got {self.num_frames}")
        if self.action_video_freq_ratio <= 0:
            raise ValueError(f"`action_video_freq_ratio` must be positive, got {self.action_video_freq_ratio}")
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                "num_frames-1 must be divisible by action_video_freq_ratio, "
                f"got {self.num_frames - 1} and {self.action_video_freq_ratio}"
            )
        if len(self.video_size) != 2:
            raise ValueError(f"`video_size` must be [H, W], got {self.video_size}")

    def _load_dataset_metadata(
        self,
        *,
        requested_state_dim: int,
        requested_action_dim: int,
        raw_state_dim: Optional[int],
        raw_action_dim: Optional[int],
        state_slices: Any,
        action_slices: Any,
        relative_joint_ranges: Any,
        proprio_dim: Optional[int],
        action_target_dim: Optional[int],
        model_dim_indices: Any,
        state_model_dim_indices: Any,
        action_model_dim_indices: Any,
    ) -> None:
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing WB LeRobot metadata: {info_path}")
        self.info = _read_json(info_path)
        self.features = dict(self.info.get("features") or {})
        self.raw_state_dim = _feature_dim(
            self.features,
            self.state_key,
            requested_state_dim if raw_state_dim is None else raw_state_dim,
        )
        self.raw_action_dim = _feature_dim(
            self.features, self.action_key, requested_action_dim if raw_action_dim is None else raw_action_dim
        )
        self.state_slices = _normalize_slice_specs(
            state_slices,
            feature_name="states",
            layout=dict(self.info.get("state_layout") or {}),
            raw_dim=self.raw_state_dim,
        )
        self.action_slices = _normalize_slice_specs(
            action_slices,
            feature_name="action",
            layout=dict(self.info.get("action_layout") or {}),
            raw_dim=self.raw_action_dim,
        )
        self.state_dim = _slice_dim(self.state_slices)
        self.action_dim = _slice_dim(self.action_slices)
        self.relative_joint_ranges = _normalize_relative_joint_ranges(
            relative_joint_ranges,
            self.action_dim,
        )
        if self.action_representation == "native" and self.relative_joint_ranges:
            raise ValueError("Native actions cannot define relative_joint_ranges.")
        if self.action_representation == "relative_joint":
            if not self.relative_joint_ranges:
                raise ValueError("relative_joint action representation requires non-empty ranges.")
            if self.relative_action_reference == "observation" and self.state_dim != self.action_dim:
                raise ValueError("Observation-relative actions require matching selected state/action dims.")
        if requested_state_dim != self.state_dim:
            raise ValueError(
                f"Configured state_dim={requested_state_dim} does not match selected "
                f"state_slices dim={self.state_dim} for {self.name}."
            )
        if requested_action_dim != self.action_dim:
            raise ValueError(
                f"Configured action_dim={requested_action_dim} does not match selected "
                f"action_slices dim={self.action_dim} for {self.name}."
            )
        self.proprio_dim = self.state_dim if proprio_dim is None else int(proprio_dim)
        self.action_target_dim = self.action_dim if action_target_dim is None else int(action_target_dim)
        if self.proprio_dim < self.state_dim:
            raise ValueError(
                f"Configured proprio_dim={self.proprio_dim} is smaller than selected "
                f"state_dim={self.state_dim} for {self.name}."
            )
        if self.action_target_dim < self.action_dim:
            raise ValueError(
                f"Configured action_target_dim={self.action_target_dim} is smaller than selected "
                f"action_dim={self.action_dim} for {self.name}."
            )
        self.state_model_dim_indices, self.action_model_dim_indices = _normalize_model_dim_index_maps(
            model_dim_indices, state_model_dim_indices, action_model_dim_indices,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            action_target_dim=self.action_target_dim,
        )
        self.model_dim_indices = model_dim_indices if model_dim_indices is None else self.state_model_dim_indices
        self.robot_type = str(self.info.get("robot_type") or "unknown")
        self.fps = int(round(float(self.info.get("fps") or 20)))
        if self.return_fk_targets:
            raw_indices = [i for start, stop, _ in self.action_slices for i in range(start, stop)]
            if "g1" not in self.robot_type.lower() or raw_indices[:29] != list(range(104, 133)):
                raise ValueError("FK requires G1 WB body action[104:133] in semantic slots 0:29")

    def _init_video_and_episode_index(
        self,
        *,
        tolerance_s: Optional[float],
        shard_cache_size: int,
    ) -> None:
        self.tolerance_s = float(tolerance_s) if tolerance_s is not None else 0.5 / max(self.fps, 1)
        video_feature = self.features.get(self.camera_key)
        if not isinstance(video_feature, Mapping) or video_feature.get("dtype") != "video":
            raise ValueError(f"{self.name} does not contain video feature {self.camera_key!r}.")
        if not str(self.info.get("video_path") or "").strip():
            raise ValueError(f"{self.name} metadata is missing the LeRobot v3 video_path template.")
        self.has_video = True
        self.video_frames = len(self.video_sample_offsets)
        self._shards = _ShardCache(self.root, cache_size=shard_cache_size)
        self.tasks, self.task_indices_by_text = self._load_tasks()
        self.episodes = self._load_selected_episodes()
        self._episode_ends: list[int] = []
        total = 0
        for episode in self.episodes:
            total += int(episode["_length"])
            self._episode_ends.append(total)
        if not self.episodes:
            raise ValueError(f"No usable episodes found in WB dataset root: {self.root}")

    def _log_loaded_dataset(self) -> None:
        logger.info(
            "Loaded WBDataset name=%s root=%s split=%s episodes=%d frames=%d "
            "state_dim=%d->%d/%d action_dim=%d->%d/%d video=%s camera=%s",
            self.name,
            self.root,
            "train" if self.is_training_set else "val",
            len(self.episodes),
            len(self),
            self.state_dim,
            self.proprio_dim,
            self.raw_state_dim,
            self.action_dim,
            self.action_target_dim,
            self.raw_action_dim,
            self.has_video,
            self.camera_key,
        )

    def _load_tasks(self) -> tuple[dict[int, str], dict[str, int]]:
        tasks_path = self.root / "meta" / "tasks.parquet"
        if not tasks_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot v3 task metadata: {tasks_path}")

        pq = _load_pyarrow()
        table = pq.read_table(tasks_path).combine_chunks()
        text_column = "task" if self.dataset_format == "official_lerobot_v3_1" else "__index_level_0__"
        required_columns = {"task_index", text_column}
        missing = sorted(required_columns - set(table.column_names))
        if missing:
            raise ValueError(f"Task metadata {tasks_path} is missing required columns: {missing}")
        if len(table) == 0:
            raise ValueError(f"Task metadata contains no rows: {tasks_path}")

        tasks: dict[int, str] = {}
        task_indices_by_text: dict[str, int] = {}
        for row in table.to_pylist():
            try:
                task_index = int(row["task_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Task metadata {tasks_path} contains an invalid task_index.") from exc
            if task_index < 0:
                raise ValueError(f"Task metadata {tasks_path} contains invalid task_index={task_index}.")
            text = _validate_canonical_task_text(
                row[text_column],
                source=f"Task metadata {tasks_path} task_index={task_index}",
            )
            if task_index in tasks:
                raise ValueError(f"Task metadata {tasks_path} contains duplicate task_index={task_index}.")
            if text in task_indices_by_text:
                raise ValueError(f"Task metadata {tasks_path} contains duplicate task text {text!r}.")
            tasks[task_index] = text
            task_indices_by_text[text] = task_index
        return tasks, task_indices_by_text

    def _episode_task(self, episode: Mapping[str, Any]) -> tuple[int, str]:
        episode_index = episode.get("episode_index", "unknown")
        if self.dataset_format == "official_lerobot_v3_1":
            task_index = int(episode["task_index"])
            if task_index not in self.tasks:
                raise ValueError(f"{self.name} episode {episode_index} has unknown task_index={task_index}.")
            return task_index, self.tasks[task_index]
        task_entries = episode.get("tasks")
        if not isinstance(task_entries, (list, tuple)) or len(task_entries) != 1:
            raise ValueError(
                f"{self.name} episode {episode_index} must contain exactly one task text; "
                f"got {task_entries!r}."
            )
        text = _validate_canonical_task_text(
            task_entries[0],
            source=f"{self.name} episode {episode_index} task text",
        )
        if text not in self.task_indices_by_text:
            raise ValueError(
                f"{self.name} episode {episode_index} references unknown task text {text!r}."
            )
        task_index = self.task_indices_by_text[text]
        return task_index, self.tasks[task_index]

    def _load_episodes_raw(self) -> list[dict[str, Any]]:
        episodes_root = self.root / "meta" / "episodes"
        episode_files = sorted(episodes_root.glob("chunk-*/file-*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"Missing LeRobot v3 episode metadata under {episodes_root}")

        pq = _load_pyarrow()
        rows: list[dict[str, Any]] = []
        required_columns = set(self._EPISODE_REQUIRED_COLUMNS) | set(self._episode_video_columns())
        for path in episode_files:
            table = pq.read_table(path).combine_chunks()
            missing = sorted(required_columns - set(table.column_names))
            if missing:
                raise ValueError(f"Episode metadata {path} is missing required columns: {missing}")
            rows.extend(table.to_pylist())
        if not rows:
            raise ValueError(f"No episode rows found under {episodes_root}")
        return rows

    def _episode_video_columns(self) -> tuple[str, str, str, str]:
        prefix = f"videos/{self.camera_key}"
        return (
            f"{prefix}/chunk_index",
            f"{prefix}/file_index",
            f"{prefix}/from_timestamp",
            f"{prefix}/to_timestamp",
        )

    def _episode_row_range(self, episode: Mapping[str, Any]) -> tuple[int, int]:
        episode_index = episode.get("episode_index", "unknown")
        try:
            length = int(episode.get("_length", episode["length"]))
            official = self.dataset_format == "official_lerobot_v3_1"
            row_from_key = "shard_row_from" if official else "dataset_from_index"
            row_to_key = "shard_row_to" if official else "dataset_to_index"
            row_from = int(episode[row_from_key])
            row_to = int(episode[row_to_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.name} episode {episode_index} has invalid LeRobot v3 row metadata."
            ) from exc
        if length <= 0 or row_from < 0 or row_to <= row_from or row_to - row_from != length:
            raise ValueError(
                f"{self.name} episode {episode_index} row range {row_from}:{row_to} "
                f"does not match positive length {length}."
            )
        return row_from, row_to

    def _episode_video_range(self, episode: Mapping[str, Any]) -> tuple[int, int]:
        episode_index = episode.get("episode_index", "unknown")
        _, _, from_key, to_key = self._episode_video_columns()
        try:
            timestamp_from = float(episode[from_key])
            timestamp_to = float(episode[to_key])
            length = int(episode.get("_length", episode["length"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.name} episode {episode_index} has invalid video timestamps for {self.camera_key}."
            ) from exc
        frame_from_float = timestamp_from * self.fps
        frame_to_float = timestamp_to * self.fps
        frame_from = round(frame_from_float)
        frame_to = round(frame_to_float)
        on_frame_grid = np.isclose(
            [frame_from_float, frame_to_float],
            [frame_from, frame_to],
            rtol=0.0,
            atol=1.0e-3,
        ).all()
        if (
            not np.isfinite(timestamp_from)
            or not np.isfinite(timestamp_to)
            or timestamp_from < 0.0
            or timestamp_to <= timestamp_from
            or not on_frame_grid
            or frame_to - frame_from != length
        ):
            raise ValueError(
                f"{self.name} episode {episode_index} video range "
                f"{timestamp_from}:{timestamp_to}s ({frame_from}:{frame_to} at {self.fps}Hz) "
                f"does not match length {length}."
            )
        return frame_from, frame_to

    def _load_selected_episodes(self) -> list[dict[str, Any]]:
        rows = sorted(self._load_episodes_raw(), key=lambda row: int(row["episode_index"]))
        if self.dataset_format == "official_lerobot_v3_1":
            file_starts: dict[tuple[int, int], int] = {}
            for row in rows:
                data_file = (int(row["data/chunk_index"]), int(row["data/file_index"]))
                start = int(row["dataset_from_index"])
                file_starts[data_file] = min(start, file_starts.get(data_file, start))
        normalized: list[dict[str, Any]] = []
        episode_indices: set[int] = set()
        ranges_by_file: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        excluded_by_index = 0
        for row in rows:
            ep = dict(row)
            if self.dataset_format == "official_lerobot_v3_1":
                data_file = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
                ep["shard_row_from"] = int(ep["dataset_from_index"]) - file_starts[data_file]
                ep["shard_row_to"] = int(ep["dataset_to_index"]) - file_starts[data_file]
            length = int(ep["length"])
            ep["_length"] = length
            episode_index = int(ep["episode_index"])
            if episode_index in episode_indices:
                raise ValueError(f"{self.name} contains duplicate episode_index={episode_index}.")
            episode_indices.add(episode_index)
            row_from, row_to = self._episode_row_range(ep)
            self._episode_video_range(ep)
            if self.dataset_format == "sealed":
                self._episode_task(ep)
            data_file = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
            ranges_by_file.setdefault(data_file, []).append((row_from, row_to, episode_index))
            if any(start <= episode_index < end for start, end in self.exclude_episode_ranges):
                continue
            if episode_index in self.exclude_episode_indices:
                excluded_by_index += 1
                continue
            normalized.append(ep)

        if self.exclude_episode_indices:
            logger.info(
                "Filtered WB bad episodes name=%s matched=%d manifest_entries=%d",
                self.name,
                excluded_by_index,
                len(self.exclude_episode_indices),
            )

        if self.dataset_format == "sealed" and len(ranges_by_file) != 1:
            raise ValueError(
                f"{self.name} open-source archive must contain exactly one data parquet; "
                f"found episode references to {sorted(ranges_by_file)}."
            )
        for data_file, ranges in ranges_by_file.items():
            previous_to = -1
            previous_episode = None
            for row_from, row_to, episode_index in sorted(ranges):
                if row_from < previous_to:
                    raise ValueError(
                        f"{self.name} data file {data_file} has overlapping episode ranges: "
                        f"episode {previous_episode} ends at {previous_to}, "
                        f"episode {episode_index} starts at {row_from}."
                    )
                previous_to = row_to
                previous_episode = episode_index

        if self.dataset_format == "official_lerobot_v3_1":
            self._fill_official_task_indices(normalized)

        if not normalized:
            return []
        if self.val_set_proportion > 1.0e-6 and len(normalized) > 1:
            if self.val_split_mode == "task_stratified":
                return self._task_stratified_episode_split(normalized)
            split_ep = int(len(normalized) * (1.0 - self.val_set_proportion))
            split_ep = max(1, split_ep)
            val_start_ep = min(split_ep, len(normalized) - 1)
            return normalized[:split_ep] if self.is_training_set else normalized[val_start_ep:]
        return normalized

    def _fill_official_task_indices(self, episodes: list[dict[str, Any]]) -> None:
        by_shard: dict[str, list[dict[str, Any]]] = {}
        for episode in episodes:
            by_shard.setdefault(self._resolve_shard_path(episode), []).append(episode)
        pq = _load_pyarrow()
        for relpath, shard_episodes in by_shard.items():
            table = pq.read_table(self.root / relpath, columns=["episode_index", "task_index"])
            episode_column = table["episode_index"]
            task_column = table["task_index"]
            for episode in shard_episodes:
                row_from, row_to = self._episode_row_range(episode)
                if row_from < 0 or row_to > table.num_rows:
                    raise ValueError(f"{self.name} episode {episode['episode_index']} exceeds {relpath}.")
                endpoints = (row_from, row_to - 1)
                if any(int(episode_column[i].as_py()) != int(episode["episode_index"]) for i in endpoints):
                    raise ValueError(
                        f"{self.name} episode {episode['episode_index']} row alignment differs in {relpath}."
                    )
                task_indices = {int(task_column[i].as_py()) for i in endpoints}
                if len(task_indices) != 1:
                    raise ValueError(f"{self.name} episode {episode['episode_index']} changes task_index.")
                episode["task_index"] = task_indices.pop()
                self._episode_task(episode)

    @staticmethod
    def _episode_task_key(episode: Mapping[str, Any]) -> tuple[str, ...]:
        tasks = episode.get("tasks")
        return ("tasks", str(tasks[0]))

    def _task_stratified_episode_split(
        self,
        episodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        positions_by_task: dict[tuple[str, ...], list[int]] = {}
        for position, episode in enumerate(episodes):
            positions_by_task.setdefault(self._episode_task_key(episode), []).append(position)

        val_positions: set[int] = set()
        for positions in positions_by_task.values():
            if len(positions) <= 1:
                continue
            val_count = int(len(positions) * self.val_set_proportion + 0.5)
            val_count = min(max(val_count, 1), len(positions) - 1)
            val_positions.update(positions[-val_count:])

        if not val_positions:
            split_ep = max(1, int(len(episodes) * (1.0 - self.val_set_proportion)))
            val_positions.update(range(min(split_ep, len(episodes) - 1), len(episodes)))

        if self.is_training_set:
            return [episode for position, episode in enumerate(episodes) if position not in val_positions]
        return [episode for position, episode in enumerate(episodes) if position in val_positions]

    def __len__(self) -> int:
        return int(self._episode_ends[-1])

    def _resolve_index(self, idx: int) -> tuple[dict[str, Any], int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"WBDataset index {idx} out of range for length {len(self)}")
        ep_idx = bisect.bisect_right(self._episode_ends, idx)
        ep_start = 0 if ep_idx == 0 else self._episode_ends[ep_idx - 1]
        return self.episodes[ep_idx], int(idx - ep_start)

    def _resolve_shard_path(self, episode: dict[str, Any]) -> str:
        shard_path = _to_rel_path(episode.get("shard_path"))
        if shard_path is not None:
            return shard_path
        data_path = _to_rel_path(episode.get("data/path") or episode.get("data_path"))
        if data_path is not None:
            return data_path
        shard_index = int(episode.get("shard_index", 0))
        data_path = str(self.info.get("data_path") or "data/shard-{shard_index:05d}.parquet")
        format_values = {
            "shard_index": shard_index,
            "chunk_index": int(episode.get("chunk_index") or episode.get("data/chunk_index") or 0),
            "file_index": int(episode.get("file_index") or episode.get("data/file_index") or 0),
            "episode_chunk": int(episode.get("episode_chunk") or episode.get("chunk_index") or 0),
            "episode_index": int(episode.get("episode_index", 0)),
        }
        return data_path.format(**format_values)

    def _resolve_video_source(self, episode: Mapping[str, Any]) -> tuple[Path, int]:
        chunk_key, file_key, _, _ = self._episode_video_columns()
        frame_from, _ = self._episode_video_range(episode)
        try:
            relpath = str(self.info["video_path"]).format(
                video_key=self.camera_key,
                chunk_index=int(episode[chunk_key]),
                file_index=int(episode[file_key]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.name} episode {episode.get('episode_index', 'unknown')} "
                f"cannot resolve video path for {self.camera_key}."
            ) from exc
        path = Path(relpath)
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            raise FileNotFoundError(
                f"{self.name} episode {episode.get('episode_index', 'unknown')} video does not exist: {path}"
            )
        return path, frame_from

    def _resolve_video_path(self, episode: Mapping[str, Any]) -> Path:
        return self._resolve_video_source(episode)[0]

    def _read_episode_table(self, episode: dict[str, Any]):
        relpath = self._resolve_shard_path(episode)
        row_from, row_to = self._episode_row_range(episode)
        return self._shards.read_rows(relpath, row_from, row_to, self.table_columns)

    def _decode_and_slice_table(
        self,
        table,
        *,
        length: int,
        source_description: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_arrays = (
            _list_column_to_numpy(table[self.state_key], np.float32, self.raw_state_dim),
            _list_column_to_numpy(table[self.action_key], np.float32, self.raw_action_dim),
            np.zeros((length, self.raw_state_dim), dtype=bool) if self.state_mask_key is None
            else _list_column_to_numpy(table[self.state_mask_key], bool, self.raw_state_dim),
            np.zeros((length, self.raw_action_dim), dtype=bool) if self.action_mask_key is None
            else _list_column_to_numpy(table[self.action_mask_key], bool, self.raw_action_dim),
        )
        specifications = (
            ("states", self.raw_state_dim, self.state_slices),
            ("action", self.raw_action_dim, self.action_slices),
            ("state_mask", self.raw_state_dim, self.state_slices),
            ("action_mask", self.raw_action_dim, self.action_slices),
        )
        for values, (label, raw_dim, _slices) in zip(raw_arrays, specifications):
            expected_shape = (length, raw_dim)
            if values.shape != expected_shape:
                raise ValueError(
                    f"{self.name} {source_description} {label} shape {values.shape}, expected {expected_shape}"
                )
        return tuple(
            _concat_slices(values, slices)
            for values, (_label, _raw_dim, slices) in zip(raw_arrays, specifications)
        )

    def _read_sliced_episode_arrays(
        self,
        episode: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._decode_and_slice_table(
            self._read_episode_table(episode),
            length=int(episode["_length"]),
            source_description=f"episode {episode.get('episode_index')}",
        )

    def set_normalizer(self, normalizer: Optional[WBFlatNormalizer]) -> None:
        self.normalizer = normalizer

    def _stats_segments(self) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for episode in self.episodes:
            relpath = self._resolve_shard_path(episode)
            length = int(episode["_length"])
            row_from, row_to = self._episode_row_range(episode)
            if segments and segments[-1]["relpath"] == relpath and segments[-1]["row_to"] == row_from:
                segments[-1]["row_to"] = row_to
                segments[-1]["episode_lengths"].append(length)
            else:
                segments.append(
                    {
                        "relpath": relpath,
                        "row_from": row_from,
                        "row_to": row_to,
                        "episode_lengths": [length],
                    }
                )
        return segments

    def _read_sliced_dataset_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
        columns = self.table_columns
        state_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        state_mask_parts: list[np.ndarray] = []
        action_mask_parts: list[np.ndarray] = []
        episode_lengths: list[int] = []
        segments = self._stats_segments()
        logger.info(
            "Reading WB stats arrays for %s: episodes=%d segments=%d stats_downsample_rate=%d",
            self.name,
            len(self.episodes),
            len(segments),
            self.stats_downsample_rate,
        )

        for segment in segments:
            relpath = str(segment["relpath"])
            row_from = int(segment["row_from"])
            row_to = int(segment["row_to"])
            length = row_to - row_from
            table = self._shards.read_rows(relpath, row_from, row_to, columns)
            states, actions, state_mask, action_mask = self._decode_and_slice_table(
                table,
                length=length,
                source_description=f"shard segment {relpath}:{row_from}:{row_to}",
            )
            state_parts.append(states)
            action_parts.append(actions)
            state_mask_parts.append(state_mask)
            action_mask_parts.append(action_mask)
            episode_lengths.extend(int(x) for x in segment["episode_lengths"])

        if not state_parts:
            raise ValueError(f"No usable stats arrays for WB dataset {self.name}")
        if len(state_parts) == 1:
            return (
                state_parts[0],
                action_parts[0],
                state_mask_parts[0],
                action_mask_parts[0],
                episode_lengths,
            )
        return (
            np.concatenate(state_parts, axis=0),
            np.concatenate(action_parts, axis=0),
            np.concatenate(state_mask_parts, axis=0),
            np.concatenate(action_mask_parts, axis=0),
            episode_lengths,
        )

    def get_dataset_stats(self) -> dict[str, Any]:
        arrays = self.get_stats_arrays()
        logger.info(
            "Computing WB norm stats for %s: frames=%d state_dim=%d action_dim=%d stats_downsample_rate=%d",
            self.name,
            arrays.states.shape[0],
            self.state_dim,
            self.action_dim,
            self.stats_downsample_rate,
        )
        return {
            "state": {
                "proprio": _state_stats_from_arrays(
                    arrays.states,
                    arrays.state_dim_is_pad,
                    self.stats_downsample_rate,
                )
            },
            "action": {
                "action": _action_stats_from_arrays(
                    arrays.actions,
                    arrays.action_dim_is_pad,
                    arrays.states,
                    arrays.state_dim_is_pad,
                    list(arrays.episode_lengths),
                    action_horizon=self.action_horizon,
                    stats_downsample_rate=self.stats_downsample_rate,
                    relative_joint_ranges=(
                        self.relative_joint_ranges if self.action_representation == "relative_joint" else ()
                    ),
                    relative_action_reference=self.relative_action_reference,
                )
            },
        }

    def get_stats_arrays(self) -> WBStatsArrays:
        states, actions, state_mask, action_mask, episode_lengths = self._read_sliced_dataset_arrays()
        return WBStatsArrays(
            states=states,
            actions=actions,
            state_dim_is_pad=state_mask,
            action_dim_is_pad=action_mask,
            episode_lengths=tuple(episode_lengths),
            name=self.name,
            stats_downsample_rate=self.stats_downsample_rate,
        )

    def _task_text(self, episode: dict[str, Any]) -> str:
        if self.override_instruction is not None:
            return _validate_canonical_task_text(
                self.override_instruction,
                source=f"Dataset {self.name!r} override_instruction",
            )
        task_index, task_text = self._episode_task(episode)
        if self.instruction_config is not None:
            task_overrides = self.instruction_config.get("task_overrides", {})
            if not isinstance(task_overrides, Mapping):
                raise ValueError(f"Dataset {self.name!r} instruction task_overrides must be a mapping.")
            for key in (task_text, task_index, str(task_index)):
                if key in task_overrides:
                    return _validate_canonical_task_text(
                        task_overrides[key],
                        source=f"Dataset {self.name!r} task override {key!r}",
                    )
        return task_text

    def _config_summary(self, episode: dict[str, Any]) -> str:
        parts: list[str] = []
        for key, label in (
            ("robot_model", "robot model"),
            ("hand_model", "hand model"),
            ("mask_key", "mask"),
        ):
            value = episode.get(key) or self.info.get(key)
            if value:
                parts.append(f"{label} {value}")

        valid_signals = []
        for key, label in (
            ("sonic_valid", "SONIC action tokens"),
            ("body_valid", "body"),
            ("hand_state_valid", "hand state"),
            ("hand_action_valid", "hand action"),
        ):
            if _as_bool(episode.get(key), default=False):
                valid_signals.append(label)
        if valid_signals:
            parts.append("valid " + ", ".join(valid_signals))

        root3 = episode.get("root3_source") or self.info.get("root3_source")
        if root3:
            parts.append(f"root3 {root3}")
        if not parts:
            parts.append("WB-WAM 110D state and 136D action")
        return "; ".join(parts)

    def _instruction_descriptions(self, episode: dict[str, Any]) -> dict[str, str]:
        if self.instruction_config is None:
            raise ValueError(f"Dataset {self.name!r} has no config-driven instruction.")

        resolved = {
            key: value
            for key, value in self.instruction_config.items()
            if key not in {"task_overrides", "variants"}
        }
        variants = self.instruction_config.get("variants", [])
        if not isinstance(variants, list):
            raise ValueError(f"Dataset {self.name!r} instruction variants must be a list.")
        for variant_index, variant in enumerate(variants):
            if not isinstance(variant, Mapping):
                raise ValueError(f"Dataset {self.name!r} instruction variant {variant_index} must be a mapping.")
            match = variant.get("match", {})
            if not isinstance(match, Mapping) or not match:
                raise ValueError(
                    f"Dataset {self.name!r} instruction variant {variant_index} "
                    "must define a non-empty `match` mapping."
                )
            matches = all(
                str(episode.get(key, self.info.get(key))) == str(expected) for key, expected in match.items()
            )
            if matches:
                resolved.update({key: value for key, value in variant.items() if key != "match"})
                break

        required = ("visual_description", "control_description")
        missing = [key for key in required if not str(resolved.get(key, "")).strip()]
        if missing:
            raise ValueError(
                f"Dataset {self.name!r} instruction config is missing {missing} "
                f"for episode {episode.get('episode_index', 'unknown')}."
            )
        return {key: str(resolved[key]) for key in required}

    def build_prompt_for_episode(self, episode: dict[str, Any]) -> str:
        task = self._task_text(episode)
        if self.instruction_config is not None:
            descriptions = self._instruction_descriptions(episode)
            return build_configured_instruction(
                task,
                visual_description=descriptions["visual_description"],
                control_description=descriptions["control_description"],
                template=self.instruction_template,
            )
        if not self.include_robot_metadata_in_prompt:
            return build_text_prompt(task)
        return WB_PROMPT.format(
            task=task,
            robot_type=self.robot_type,
            config_summary=self._config_summary(episode),
        )

    def unique_prompts(self) -> list[str]:
        prompts = []
        seen = set()
        for episode in self.episodes:
            prompt = self.build_prompt_for_episode(episode)
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
        return prompts

    def _get_cached_text_context(self, prompt: str):
        if not self.use_text_embed_cache:
            return None, None
        if self.text_embedding_cache_dir is None:
            raise ValueError("`text_embedding_cache_dir` is required when use_text_embed_cache=true.")
        cache_dir = Path(self.text_embedding_cache_dir).expanduser()
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        suffix = self.text_embedding_cache_suffix
        cache_path = cache_dir / f"{hashed}.t5_len{self.context_len}.{suffix}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py with the matching WB data config first."
            )
        payload = torch.load(str(cache_path), map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2 or context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context must be [context_len,D] with context_len={self.context_len}, "
                f"got {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1 or context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask must be [context_len] with context_len={self.context_len}, "
                f"got {tuple(context_mask.shape)} in {cache_path}"
            )
        context = context.clone()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        return context, context_mask

    def _read_video(self, episode: dict[str, Any], frame_indices: np.ndarray) -> torch.Tensor:
        video_path, frame_offset = self._resolve_video_source(episode)
        from wbwam.datasets.video_decode import (
            VideoIntegrityError,
            decode_video_frames,
            decode_video_frames_by_indices_torchcodec,
            get_safe_default_codec,
        )

        source_indices = frame_indices.astype(np.int64) + int(frame_offset)
        _, frame_end = self._episode_video_range(episode)
        if np.any(source_indices < frame_offset) or np.any(source_indices >= frame_end):
            raise VideoIntegrityError(
                f"Requested frame crosses the episode video boundary: "
                f"dataset={self.name} episode={episode.get('episode_index')} "
                f"range={frame_offset}:{frame_end} requested={source_indices.tolist()}"
            )
        backend = self.video_backend or get_safe_default_codec()
        if backend == "torchcodec":
            # WB metadata maps each row directly to a sequential video frame.
            # Exact index decode avoids malformed container FPS headers changing that mapping.
            return decode_video_frames_by_indices_torchcodec(
                video_path=video_path,
                frame_indices=source_indices.tolist(),
                source_episode_id=f"{self.name}/episode_{episode.get('episode_index', 'unknown')}",
            )
        timestamps = [float(idx) / float(max(self.fps, 1)) for idx in source_indices.tolist()]
        return decode_video_frames(
            video_path=video_path,
            timestamps=timestamps,
            tolerance_s=self.tolerance_s,
            backend=backend,
        )

    def _video_or_black(
        self,
        episode: dict[str, Any],
        frame_indices: np.ndarray,
        image_is_pad: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames = None
        if self.has_video:
            try:
                frames = self._read_video(episode, frame_indices)
            except Exception as err:
                from wbwam.datasets.video_decode import (
                    VideoIntegrityError,
                )

                if isinstance(err, (VideoIntegrityError, FileNotFoundError, ValueError)):
                    raise
                if self.strict_video_decode:
                    raise
                logger.warning(
                    "Failed to decode video for dataset=%s episode=%s; using black video pad.",
                    self.name,
                    episode.get("episode_index"),
                    exc_info=True,
                )

        if frames is None:
            frames = torch.zeros(
                (self.video_frames, 3, self.video_size[0], self.video_size[1]),
                dtype=torch.float32,
            )
            image_is_pad = np.ones((self.video_frames,), dtype=bool)
        else:
            if frames.ndim != 4:
                raise ValueError(f"Decoded video must be [T,C,H,W], got {tuple(frames.shape)}")
            if frames.shape[1] != 3 and frames.shape[-1] == 3:
                frames = frames.permute(0, 3, 1, 2)
            if frames.shape[1] != 3:
                raise ValueError(f"Decoded video channel dimension must be 3, got {tuple(frames.shape)}")
            frames = transforms_F.resize(
                frames,
                size=self.video_size,
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            frames = frames.to(dtype=torch.float32).clamp_(0.0, 1.0)
            if self.is_training_set and self.video_augmentor.enabled:
                frames = self.video_augmentor(frames)

        video = frames.mul(2.0).sub(1.0).permute(1, 0, 2, 3).contiguous()
        return video, torch.as_tensor(image_is_pad, dtype=torch.bool)

    def _build_numeric_window(
        self,
        episode: dict[str, Any],
        local_start: int,
    ) -> dict[str, torch.Tensor]:
        length = int(episode["_length"])
        states, actions, state_mask, action_mask = self._read_sliced_episode_arrays(episode)

        action_offsets = np.arange(self.action_horizon, dtype=np.int64)
        action_unclamped = local_start + action_offsets
        action_temporal_pad = action_unclamped >= length
        action_indices = np.clip(action_unclamped, 0, length - 1)

        proprio = states[action_indices].copy()
        action = actions[action_indices].copy()
        proprio_dim_is_pad = state_mask[action_indices].copy()
        action_dim_is_pad = action_mask[action_indices].copy()
        action_semantic_dim_is_pad = action_dim_is_pad.copy()
        if action_temporal_pad.any():
            action_dim_is_pad[action_temporal_pad, :] = True
            proprio_dim_is_pad[action_temporal_pad, :] = True

        if self.action_representation == "relative_joint":
            if self.relative_action_reference == "first_action":
                reference = action[0].copy()
                reference_dim_is_pad = action_dim_is_pad[0].copy()
            else:
                reference = states[local_start]
                reference_dim_is_pad = state_mask[local_start]
            action, action_dim_is_pad = _apply_relative_joint_action(
                action,
                action_dim_is_pad,
                reference,
                reference_dim_is_pad,
                self.relative_joint_ranges,
            )
            for start, end in self.relative_joint_ranges:
                action_semantic_dim_is_pad[:, start:end] |= reference_dim_is_pad[None, start:end]

        proprio[proprio_dim_is_pad] = 0.0
        fk_target = None
        if getattr(self, "return_fk_targets", False):
            fk_target = torch.as_tensor(action[:, :29].copy(), dtype=torch.float32)
        action[action_dim_is_pad] = 0.0
        action_is_pad = action_temporal_pad | action_dim_is_pad.all(axis=1)
        proprio_is_pad = action_temporal_pad | proprio_dim_is_pad.all(axis=1)

        proprio_tensor = torch.as_tensor(proprio, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        proprio_dim_is_pad_tensor = torch.as_tensor(proprio_dim_is_pad, dtype=torch.bool)
        action_dim_is_pad_tensor = torch.as_tensor(action_dim_is_pad, dtype=torch.bool)
        if self.normalizer is not None:
            proprio_tensor = self.normalizer.normalize_proprio(proprio_tensor)
            action_tensor = self.normalizer.normalize_action(action_tensor)
            proprio_tensor = proprio_tensor.masked_fill(proprio_dim_is_pad_tensor, 0.0)
            action_tensor = action_tensor.masked_fill(action_dim_is_pad_tensor, 0.0)

        if self.state_model_dim_indices is None:
            proprio_tensor, proprio_dim_is_pad_tensor = _right_pad_tensor_and_mask(
                proprio_tensor,
                proprio_dim_is_pad_tensor,
                self.proprio_dim,
                feature_name="proprio",
            )
        else:
            proprio_tensor, proprio_dim_is_pad_tensor = _scatter_tensor_and_mask(
                proprio_tensor,
                proprio_dim_is_pad_tensor,
                self.proprio_dim,
                self.state_model_dim_indices,
                feature_name="proprio",
            )
        if self.action_model_dim_indices is None:
            action_tensor, action_dim_is_pad_tensor = _right_pad_tensor_and_mask(
                action_tensor,
                action_dim_is_pad_tensor,
                self.action_target_dim,
                feature_name="action",
            )
        else:
            action_tensor, action_dim_is_pad_tensor = _scatter_tensor_and_mask(
                action_tensor,
                action_dim_is_pad_tensor,
                self.action_target_dim,
                self.action_model_dim_indices,
                feature_name="action",
            )
        semantic_mask = torch.ones_like(action_dim_is_pad_tensor)
        semantic_source = torch.as_tensor(action_semantic_dim_is_pad, dtype=torch.bool)
        if self.action_model_dim_indices is None:
            semantic_mask[:, :semantic_source.shape[-1]] = semantic_source
        else:
            semantic_mask[:, list(self.action_model_dim_indices)] = semantic_source
        numeric = {
            "proprio": proprio_tensor,
            "action": action_tensor,
            "proprio_is_pad": torch.as_tensor(proprio_is_pad, dtype=torch.bool),
            "action_is_pad": torch.as_tensor(action_is_pad, dtype=torch.bool),
            "proprio_dim_is_pad": proprio_dim_is_pad_tensor,
            "action_dim_is_pad": action_dim_is_pad_tensor,
            "action_semantic_dim_is_pad": semantic_mask,
        }
        if fk_target is not None:
            numeric["fk_joint_target"] = fk_target
            numeric["fk_joint_offset"] = self.fk_joint_offset.clone()
        return numeric

    def _build_video_window(
        self,
        episode: dict[str, Any],
        local_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = int(episode["_length"])
        video_unclamped = local_start + np.asarray(self.video_sample_offsets, dtype=np.int64)
        image_is_pad = video_unclamped >= length
        video_indices = np.clip(video_unclamped, 0, length - 1)
        return self._video_or_black(episode, video_indices, image_is_pad)

    def _assemble_sample(
        self,
        *,
        episode: dict[str, Any],
        local_start: int,
        numeric: dict[str, torch.Tensor],
        video: torch.Tensor,
        image_is_pad: torch.Tensor,
        prompt: str,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "video": video,
            "action": numeric["action"],
            "proprio": numeric["proprio"],
            "prompt": prompt,
            "image_is_pad": image_is_pad,
            "has_video": ~image_is_pad.all(),
            "action_is_pad": numeric["action_is_pad"],
            "proprio_is_pad": numeric["proprio_is_pad"],
            "action_dim_is_pad": numeric["action_dim_is_pad"],
            "action_semantic_dim_is_pad": numeric["action_semantic_dim_is_pad"],
            "proprio_dim_is_pad": numeric["proprio_dim_is_pad"],
            "dataset_name": self.name,
            "dataset_root": str(self.root),
            "episode_index": int(episode.get("episode_index", -1)),
            "frame_index": int(local_start),
        }
        for key in ("fk_joint_target", "fk_joint_offset"):
            if key in numeric:
                sample[key] = numeric[key]
        if context is not None:
            sample["context"] = context
            sample["context_mask"] = context_mask
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode, local_start = self._resolve_index(int(idx))
        numeric = self._build_numeric_window(episode, local_start)
        video, image_is_pad = self._build_video_window(episode, local_start)
        prompt = self.build_prompt_for_episode(episode)
        context, context_mask = self._get_cached_text_context(prompt)
        return self._assemble_sample(
            episode=episode,
            local_start=local_start,
            numeric=numeric,
            video=video,
            image_is_pad=image_is_pad,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
        )


class WBMixtureDataset(Dataset):
    """Mixture of WB-WAM LeRobot v3 roots."""

    _VIDEO_RESAMPLE_ATTEMPTS = 3

    def __init__(
        self,
        datasets: Optional[list[dict[str, Any]]] = None,
        *,
        archives: Optional[list[dict[str, Any]]] = None,
        state_dim: int = 110,
        action_dim: int = 136,
        proprio_dim: Optional[int] = None,
        action_target_dim: Optional[int] = None,
        model_dim_indices: Any = None,
        state_model_dim_indices: Any = None,
        action_model_dim_indices: Any = None,
        raw_state_dim: Optional[int] = None,
        raw_action_dim: Optional[int] = None,
        dataset_format: str = "sealed",
        state_key: str = "observation.state",
        action_key: str = "action",
        state_mask_key: Optional[str] = "state_mask_110",
        action_mask_key: Optional[str] = "action_mask_136",
        state_slices: Any = None,
        action_slices: Any = None,
        action_representation: str = "absolute",
        relative_joint_ranges: Any = None,
        relative_action_reference: str = "observation",
        num_frames: int = 33,
        action_video_freq_ratio: int = 8,
        video_size: Iterable[int] = (224, 224),
        camera_key: str = "observation.images.stereo_left",
        image_augmentation: Optional[Mapping[str, Any]] = None,
        val_set_proportion: float = 0.05,
        is_training_set: bool = True,
        exclude_episode_ranges: Any = None,
        bad_episodes_path: Optional[str] = None,
        use_weight_normalization: bool = True,
        use_weight_for_sampling: bool = False,
        sampling_weights: Optional[Mapping[str, float]] = None,
        text_embedding_cache_dir: Optional[str] = None,
        text_embedding_cache_suffix: Optional[str] = "wan22ti2v5b",
        use_text_embed_cache: bool = True,
        context_len: int = 128,
        include_robot_metadata_in_prompt: bool = True,
        instruction_template: Optional[str] = None,
        video_backend: Optional[str] = None,
        tolerance_s: Optional[float] = None,
        shard_cache_size: int = 2,
        strict_video_decode: bool = False,
        stats_downsample_rate: int = 1,
        pretrained_norm_stats: Optional[str] = None,
        compute_norm_stats: bool = True,
        norm_default_mode: str = "q01/q99",
        norm_exception_mode: Optional[dict[str, dict[str, str]]] = None,
        use_stepwise_action_norm: bool = True,
        norm_missing_key_mode: str = "error",
        dummy_clip_default: Iterable[float] = (-5.0, 5.0),
        norm_tail_scale: float = 0.075,
        n_datasets: Optional[int] = None,
        return_fk_targets: bool = False,
        **_unused,
    ):
        datasets = resolve_wb_dataset_configs(archives=archives, datasets=datasets)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.proprio_dim = self.state_dim if proprio_dim is None else int(proprio_dim)
        self.action_target_dim = self.action_dim if action_target_dim is None else int(action_target_dim)
        self.model_dim_indices = model_dim_indices
        self.state_model_dim_indices = state_model_dim_indices
        self.action_model_dim_indices = action_model_dim_indices
        self.raw_state_dim = None if raw_state_dim is None else int(raw_state_dim)
        self.raw_action_dim = None if raw_action_dim is None else int(raw_action_dim)
        self.dataset_format = str(dataset_format)
        self.state_key = str(state_key)
        self.action_key = str(action_key)
        self.state_mask_key = state_mask_key
        self.action_mask_key = action_mask_key
        self.state_slices = state_slices
        self.action_slices = action_slices
        self.action_representation = str(action_representation)
        self.relative_action_reference = _normalize_relative_action_reference(relative_action_reference)
        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.video_size = [int(x) for x in video_size]
        self.camera_key = str(camera_key)
        self.val_set_proportion = float(val_set_proportion)
        self.image_augmentation = image_augmentation
        self.is_training_set = bool(is_training_set)
        self.exclude_episode_ranges = exclude_episode_ranges
        self.bad_episodes_path = bad_episodes_path
        self.use_weight_normalization = bool(use_weight_normalization)
        self.use_weight_for_sampling = bool(use_weight_for_sampling)
        self.sampling_weight_overrides = {
            str(name): float(weight) for name, weight in dict(sampling_weights or {}).items()
        }
        self.compute_norm_stats = bool(compute_norm_stats)
        self.stats_downsample_rate = max(1, int(stats_downsample_rate))
        self.norm_default_mode = str(norm_default_mode)
        self.norm_exception_mode = norm_exception_mode or {}
        self.use_stepwise_action_norm = bool(use_stepwise_action_norm)
        self.norm_missing_key_mode = str(norm_missing_key_mode)
        self.norm_tail_scale = float(norm_tail_scale)
        self.normalizer: Optional[WBFlatNormalizer] = None

        self.return_fk_targets = bool(return_fk_targets)

        self._validate_mixture_config(
            relative_joint_ranges=relative_joint_ranges,
            dummy_clip_default=dummy_clip_default,
        )
        self._sampling_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.sharding_metadata = {"enabled": False}

        dataset_cfgs = self._select_dataset_configs(datasets, n_datasets)
        bad_episode_indices = _load_bad_episode_manifest(self.bad_episodes_path)
        dataset_defaults = {
            "text_embedding_cache_dir": text_embedding_cache_dir,
            "text_embedding_cache_suffix": text_embedding_cache_suffix,
            "use_text_embed_cache": use_text_embed_cache,
            "context_len": context_len,
            "include_robot_metadata_in_prompt": include_robot_metadata_in_prompt,
            "instruction_template": instruction_template,
            "video_backend": video_backend,
            "tolerance_s": tolerance_s,
            "shard_cache_size": shard_cache_size,
            "strict_video_decode": strict_video_decode,
            "image_augmentation": image_augmentation,
        }
        self._build_child_datasets(dataset_cfgs, dataset_defaults, bad_episode_indices)
        self._init_dataset_weights()
        self._init_mixture_normalizer(pretrained_norm_stats)
        self.effective_lengths = self._effective_lengths()
        self._set_offsets(self.effective_lengths)
        self._log_loaded_mixture()

    def _validate_mixture_config(
        self,
        *,
        relative_joint_ranges: Any,
        dummy_clip_default: Iterable[float],
    ) -> None:
        if self.proprio_dim < self.state_dim:
            raise ValueError(
                f"WBMixtureDataset requires proprio_dim >= state_dim, "
                f"got proprio_dim={self.proprio_dim}, state_dim={self.state_dim}."
            )
        if self.action_target_dim < self.action_dim:
            raise ValueError(
                f"WBMixtureDataset requires action_target_dim >= action_dim, "
                f"got action_target_dim={self.action_target_dim}, action_dim={self.action_dim}."
            )
        self.state_model_dim_indices, self.action_model_dim_indices = _normalize_model_dim_index_maps(
            self.model_dim_indices, self.state_model_dim_indices, self.action_model_dim_indices,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            action_target_dim=self.action_target_dim,
        )
        self.model_dim_indices = self.state_model_dim_indices if self.model_dim_indices is not None else None
        if self.action_representation not in {"absolute", "relative_joint", "native"}:
            raise ValueError(f"Unsupported WB action_representation: {self.action_representation}")
        if self.action_representation != "relative_joint" and self.relative_action_reference != "observation":
            raise ValueError("relative_action_reference is only configurable for relative_joint actions.")
        self.relative_joint_ranges = _normalize_relative_joint_ranges(relative_joint_ranges, self.action_dim)
        if self.action_representation == "native" and self.relative_joint_ranges:
            raise ValueError("Native actions cannot define relative_joint_ranges.")
        if self.action_representation == "relative_joint":
            if not self.relative_joint_ranges:
                raise ValueError("relative_joint action representation requires non-empty ranges.")
            if self.relative_action_reference == "observation" and self.state_dim != self.action_dim:
                raise ValueError("Observation-relative actions require matching selected state/action dims.")
        if any(weight < 0 for weight in self.sampling_weight_overrides.values()):
            raise ValueError("WB sampling_weights must be non-negative.")
        if self.action_representation == "relative_joint" and not self.use_stepwise_action_norm:
            raise ValueError("relative_joint action representation requires stepwise action normalization.")
        dummy_clip = list(dummy_clip_default)
        if len(dummy_clip) != 2:
            raise ValueError(f"dummy_clip_default must contain two values, got {dummy_clip_default}")
        self.dummy_clip_default = (float(dummy_clip[0]), float(dummy_clip[1]))

    @staticmethod
    def _select_dataset_configs(
        datasets: list[dict[str, Any]],
        n_datasets: Optional[int],
    ) -> list[dict[str, Any]]:
        limit = None if n_datasets is None else int(n_datasets)
        if limit is not None and limit <= 0:
            raise ValueError(f"`n_datasets` must be positive or None, got {n_datasets}")
        return list(datasets if limit is None else datasets[:limit])

    def _build_child_datasets(
        self,
        dataset_cfgs: list[dict[str, Any]],
        defaults: dict[str, Any],
        bad_episode_indices: Mapping[str, frozenset[int]],
    ) -> None:
        self.datasets: list[WBDataset] = []
        self.names: list[str] = []
        self.raw_weights: list[float] = []
        self.raw_sampling_weights: list[float] = []
        for config in dataset_cfgs:
            cfg = dict(config)
            root = cfg.pop("root")
            name = cfg.pop("name", None)
            resolved_name = str(name or Path(root).expanduser().name)
            weight = float(cfg.pop("weight", 1.0))
            if weight < 0:
                raise ValueError(f"WB dataset weight must be non-negative, got {weight} for {name}.")
            override_instruction = cfg.pop("override_instruction", None)
            instruction_config = cfg.pop("instruction", None)
            dataset = WBDataset(
                root=root,
                name=name,
                num_frames=self.num_frames,
                action_video_freq_ratio=self.action_video_freq_ratio,
                video_size=self.video_size,
                camera_key=cfg.pop("camera_key", self.camera_key),
                image_augmentation=cfg.pop("image_augmentation", defaults["image_augmentation"]),
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                proprio_dim=self.proprio_dim,
                action_target_dim=self.action_target_dim,
                model_dim_indices=self.model_dim_indices,
                state_model_dim_indices=(
                    None if self.model_dim_indices is not None else self.state_model_dim_indices
                ),
                action_model_dim_indices=(
                    None if self.model_dim_indices is not None else self.action_model_dim_indices
                ),
                raw_state_dim=cfg.pop("raw_state_dim", self.raw_state_dim),
                raw_action_dim=cfg.pop("raw_action_dim", self.raw_action_dim),
                dataset_format=cfg.pop("dataset_format", self.dataset_format),
                state_key=cfg.pop("state_key", self.state_key),
                action_key=cfg.pop("action_key", self.action_key),
                state_mask_key=cfg.pop("state_mask_key", self.state_mask_key),
                action_mask_key=cfg.pop("action_mask_key", self.action_mask_key),
                state_slices=cfg.pop("state_slices", self.state_slices),
                action_slices=cfg.pop("action_slices", self.action_slices),
                action_representation=self.action_representation,
                relative_joint_ranges=self.relative_joint_ranges,
                relative_action_reference=self.relative_action_reference,
                val_set_proportion=self.val_set_proportion,
                val_split_mode=cfg.pop("val_split_mode", "sequential"),
                exclude_episode_ranges=cfg.pop("exclude_episode_ranges", self.exclude_episode_ranges),
                exclude_episode_indices=cfg.pop(
                    "exclude_episode_indices",
                    bad_episode_indices.get(resolved_name, ()),
                ),
                is_training_set=self.is_training_set,
                text_embedding_cache_dir=cfg.pop(
                    "text_embedding_cache_dir",
                    defaults["text_embedding_cache_dir"],
                ),
                text_embedding_cache_suffix=cfg.pop(
                    "text_embedding_cache_suffix",
                    defaults["text_embedding_cache_suffix"],
                ),
                use_text_embed_cache=bool(cfg.pop("use_text_embed_cache", defaults["use_text_embed_cache"])),
                context_len=int(cfg.pop("context_len", defaults["context_len"])),
                include_robot_metadata_in_prompt=bool(
                    cfg.pop(
                        "include_robot_metadata_in_prompt",
                        defaults["include_robot_metadata_in_prompt"],
                    )
                ),
                instruction_template=cfg.pop(
                    "instruction_template",
                    defaults["instruction_template"],
                ),
                instruction_config=instruction_config,
                override_instruction=override_instruction,
                video_backend=cfg.pop("video_backend", defaults["video_backend"]),
                tolerance_s=cfg.pop("tolerance_s", defaults["tolerance_s"]),
                shard_cache_size=int(cfg.pop("shard_cache_size", defaults["shard_cache_size"])),
                strict_video_decode=bool(cfg.pop("strict_video_decode", defaults["strict_video_decode"])),
                stats_downsample_rate=int(cfg.pop("stats_downsample_rate", self.stats_downsample_rate)),
                return_fk_targets=self.return_fk_targets,
                fk_joint_space=cfg.pop("fk_joint_space", None),
            )
            if cfg:
                logger.warning(
                    "Ignoring unused WB dataset config keys for %s: %s",
                    dataset.name,
                    sorted(cfg),
                )
            self.datasets.append(dataset)
            self.names.append(dataset.name)
            self.raw_weights.append(weight)
            self.raw_sampling_weights.append(self.sampling_weight_overrides.get(dataset.name, weight))

    def _init_dataset_weights(self) -> None:
        self.actual_lengths = [len(dataset) for dataset in self.datasets]
        unknown_sampling_names = sorted(set(self.sampling_weight_overrides) - set(self.names))
        if unknown_sampling_names:
            raise ValueError(f"Unknown WB sampling_weights datasets: {unknown_sampling_names}")
        if self.use_weight_normalization:
            self.weights = self._normalized_weights(self.raw_weights)
            self.sampling_weights = self._normalized_weights(self.raw_sampling_weights)
        else:
            self.weights = list(self.raw_weights)
            self.sampling_weights = list(self.raw_sampling_weights)

    def _init_mixture_normalizer(self, pretrained_norm_stats: Optional[str]) -> None:
        if not self.compute_norm_stats:
            return
        dataset_stats = self._resolve_or_compute_dataset_stats(pretrained_norm_stats)
        self.normalizer = WBFlatNormalizer(
            dataset_stats,
            norm_default_mode=self.norm_default_mode,
            norm_exception_mode=self.norm_exception_mode,
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            norm_missing_key_mode=self.norm_missing_key_mode,
            dummy_clip_default=self.dummy_clip_default,
            norm_tail_scale=self.norm_tail_scale,
        )
        for dataset in self.datasets:
            dataset.set_normalizer(self.normalizer)

    def _log_loaded_mixture(self) -> None:
        logger.info(
            "Loaded WBMixtureDataset split=%s roots=%d frames=%d weighted=%s norm=%s",
            "train" if self.is_training_set else "val",
            len(self.datasets),
            len(self),
            self.use_weight_for_sampling,
            self.compute_norm_stats,
        )

    def _stats_metadata(self) -> dict[str, Any]:
        return {
            "wb_norm_stats_version": WB_STATS_VERSION,
            "names": list(self.names),
            "roots": [str(ds.root) for ds in self.datasets],
            "raw_weights": list(self.raw_weights),
            "normalized_weights": list(self.weights),
            "raw_sampling_weights": list(self.raw_sampling_weights),
            "normalized_sampling_weights": list(self.sampling_weights),
            "dataset_lengths": list(self.actual_lengths),
            "norm_weights_applied": False,
            "sampling_weights_applied_to_norm": False,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_representation": self.action_representation,
            "relative_joint_ranges": [list(item) for item in self.relative_joint_ranges],
            "relative_action_reference": self.relative_action_reference,
            "proprio_dim": self.proprio_dim,
            "action_target_dim": self.action_target_dim,
            "model_dim_indices": (None if self.model_dim_indices is None else list(self.model_dim_indices)),
            "state_model_dim_indices": (
                None if self.model_dim_indices is not None or self.state_model_dim_indices is None
                else list(self.state_model_dim_indices)
            ),
            "action_model_dim_indices": (
                None if self.model_dim_indices is not None or self.action_model_dim_indices is None
                else list(self.action_model_dim_indices)
            ),
            "raw_state_dim": self.raw_state_dim,
            "raw_action_dim": self.raw_action_dim,
            "dataset_format": self.dataset_format,
            "state_key": self.state_key,
            "action_key": self.action_key,
            "state_mask_key": self.state_mask_key,
            "action_mask_key": self.action_mask_key,
            "dataset_fps": [ds.fps for ds in self.datasets],
            "state_slices": [_slices_metadata(ds.state_slices) for ds in self.datasets],
            "action_slices": [_slices_metadata(ds.action_slices) for ds in self.datasets],
            "val_split_modes": [ds.val_split_mode for ds in self.datasets],
            "exclude_episode_ranges": [[list(item) for item in ds.exclude_episode_ranges] for ds in self.datasets],
            "dataset_stats_downsample_rates": [ds.stats_downsample_rate for ds in self.datasets],
            "num_frames": self.num_frames,
            "action_horizon": self.num_frames - 1,
            "action_video_freq_ratio": self.action_video_freq_ratio,
            "mask_aware": True,
            "norm_default_mode": self.norm_default_mode,
            "use_stepwise_action_norm": self.use_stepwise_action_norm,
            "stats_downsample_rate": self.stats_downsample_rate,
        }

    def _resolve_or_compute_dataset_stats(self, pretrained_norm_stats: Optional[str]) -> dict[str, Any]:
        if pretrained_norm_stats:
            stats = load_dataset_stats_from_json(str(pretrained_norm_stats))
            logger.info("Using WB pretrained_norm_stats: %s", pretrained_norm_stats)
            _validate_action_stats_metadata(
                stats,
                action_representation=self.action_representation,
                relative_joint_ranges=self.relative_joint_ranges,
                action_dim=self.action_dim,
                relative_action_reference=self.relative_action_reference,
            )
            if PartialState().is_main_process:
                current_run_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
                if os.path.abspath(str(pretrained_norm_stats)) != os.path.abspath(current_run_stats_path):
                    save_dataset_stats_to_json(stats, current_run_stats_path)
            return stats

        if not self.is_training_set:
            raise ValueError("pretrained_norm_stats must be provided for WB validation/test datasets.")

        if PartialState().is_main_process:
            logger.info("Calculating WB mixture dataset stats for normalization...")
            stats = self.get_dataset_stats()
            save_dataset_stats_to_json(stats, os.path.join(misc.get_work_dir(), "dataset_stats.json"))
        else:
            stats = None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            obj_list = [stats]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            stats = obj_list[0]
        if stats is None:
            raise RuntimeError("WB norm stats were not computed or broadcast.")
        _validate_action_stats_metadata(
            stats,
            action_representation=self.action_representation,
            relative_joint_ranges=self.relative_joint_ranges,
            action_dim=self.action_dim,
            relative_action_reference=self.relative_action_reference,
        )
        return stats

    def get_dataset_stats(self) -> dict[str, Any]:
        return _global_wb_stats_from_arrays(
            [dataset.get_stats_arrays() for dataset in self.datasets],
            metadata=self._stats_metadata(),
            action_horizon=self.num_frames - 1,
            relative_joint_ranges=(
                self.relative_joint_ranges if self.action_representation == "relative_joint" else ()
            ),
            relative_action_reference=self.relative_action_reference,
        )

    def _normalized_weights(self, raw_weights: list[float]) -> list[float]:
        total_len = float(sum(self.actual_lengths))
        denom = sum(float(length) * float(weight) for length, weight in zip(self.actual_lengths, raw_weights))
        if denom <= 0:
            raise ValueError("Invalid WB dataset weights: weighted length denominator <= 0.")
        scale = total_len / denom
        return [scale * float(weight) for weight in raw_weights]

    def _effective_lengths(self) -> list[int]:
        if not (self.is_training_set and self.use_weight_for_sampling):
            return list(self.actual_lengths)
        return [
            0 if weight == 0 else max(1, int(length * weight))
            for length, weight in zip(self.actual_lengths, self.sampling_weights)
        ]

    def _set_offsets(self, lengths: list[int]) -> None:
        total = 0
        self._ends = []
        for length in lengths:
            total += int(length)
            self._ends.append(total)

    def __len__(self) -> int:
        return int(self._ends[-1])

    @property
    def sampling_epoch(self) -> int:
        return int(self._sampling_epoch.item())

    def set_epoch(self, epoch: int) -> None:
        self._sampling_epoch.fill_(int(epoch))

    def _map_local_index(self, dataset_idx: int, local_idx: int) -> int:
        actual_len = int(self.actual_lengths[dataset_idx])
        effective_len = int(self.effective_lengths[dataset_idx])
        if not (self.is_training_set and self.use_weight_for_sampling):
            return int(local_idx)
        if effective_len >= actual_len:
            return int(local_idx % actual_len)
        stride = max(1, int(actual_len * 0.61803398875))
        while math.gcd(stride, actual_len) != 1:
            stride += 1
            if stride >= actual_len:
                stride = 1
                break
        offset = (self.sampling_epoch * 1000003 + dataset_idx) % actual_len
        return int((local_idx * stride + offset) % actual_len)

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"WBMixtureDataset index {idx} out of range for length {len(self)}")
        dataset_idx = bisect.bisect_right(self._ends, idx)
        start = 0 if dataset_idx == 0 else self._ends[dataset_idx - 1]
        local_idx = int(idx - start)
        return dataset_idx, self._map_local_index(dataset_idx, local_idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from wbwam.datasets.video_decode import (
            VideoIntegrityError,
        )

        dataset_idx, local_idx = self._resolve_index(int(idx))
        dataset = self.datasets[dataset_idx]
        original_local_idx = local_idx
        for attempt in range(self._VIDEO_RESAMPLE_ATTEMPTS + 1):
            try:
                sample = dataset[local_idx]
            except VideoIntegrityError as err:
                sample = None
                all_video_pad = True
                failure = str(err)
            else:
                image_is_pad = sample.get("image_is_pad")
                all_video_pad = bool(
                    dataset.has_video
                    and image_is_pad is not None
                    and torch.as_tensor(image_is_pad, dtype=torch.bool).all().item()
                )
                failure = "fully padded video"
            if not all_video_pad:
                assert sample is not None
                sample["mixture_dataset_index"] = dataset_idx
                return sample
            if attempt < self._VIDEO_RESAMPLE_ATTEMPTS:
                logger.warning(
                    "Discarding invalid video sample dataset=%s local_idx=%d reason=%s; resampling.",
                    self.names[dataset_idx],
                    local_idx,
                    failure,
                )
                local_idx = (original_local_idx + (attempt + 1) * 1000003) % self.actual_lengths[dataset_idx]

        raise RuntimeError(
            f"Failed to load a valid video sample from dataset={self.names[dataset_idx]} "
            f"after {self._VIDEO_RESAMPLE_ATTEMPTS} replacements."
        )

    def unique_prompts(self) -> list[str]:
        prompts: list[str] = []
        seen = set()
        for dataset in self.datasets:
            for prompt in dataset.unique_prompts():
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)
        return prompts
