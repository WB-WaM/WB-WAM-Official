from __future__ import annotations

from typing import Any

import numpy as np

RELATIVE_ACTION_REFERENCES = ("observation", "first_action")
SUPPORTED_WB_STATS_VERSIONS = frozenset({2, 3})
LEGACY_WB_MODEL_DIM = 96


def _ranges(value: Any, *, dim: int) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    result: list[tuple[int, int]] = []
    for item in value:
        if hasattr(item, "detach"):
            item = item.detach().cpu().reshape(-1).tolist()
        elif isinstance(item, np.ndarray):
            item = item.reshape(-1).tolist()
        is_sequence = isinstance(item, (list, tuple)) or item.__class__.__name__ == "ListConfig"
        if not is_sequence or len(item) != 2:
            raise ValueError(f"relative_joint_ranges entries must be [start, end], got {item!r}")
        start, end = int(item[0]), int(item[1])
        if start < 0 or end <= start or end > dim:
            raise ValueError(f"invalid relative_joint_range [{start}, {end}] for {dim}D action")
        result.append((start, end))
    result.sort()
    for previous, current in zip(result, result[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"overlapping relative_joint_ranges: {previous} and {current}")
    return tuple(result)


def _model_dim_indices(
    value: Any,
    *,
    source_dim: int,
    target_dim: int,
    feature_name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    is_sequence = isinstance(value, (list, tuple)) or value.__class__.__name__ == "ListConfig"
    if not is_sequence:
        raise TypeError(f"{feature_name} model_dim_indices must be a sequence, got {type(value).__name__}")
    indices = tuple(int(index) for index in value)
    if len(indices) != source_dim:
        raise ValueError(f"{feature_name} model_dim_indices must have {source_dim} entries, got {len(indices)}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{feature_name} model_dim_indices must be unique")
    if any(index < 0 or index >= target_dim for index in indices):
        raise ValueError(f"{feature_name} model_dim_indices must be in [0, {target_dim}), got {indices}")
    return indices


def _shared_model_dim_indices(
    value: Any,
    *,
    state_dim: int,
    action_dim: int,
    proprio_dim: int,
    action_target_dim: int,
) -> tuple[int, ...] | None:
    proprio_indices = _model_dim_indices(
        value,
        source_dim=state_dim,
        target_dim=proprio_dim,
        feature_name="proprio",
    )
    action_indices = _model_dim_indices(
        value,
        source_dim=action_dim,
        target_dim=action_target_dim,
        feature_name="action",
    )
    if proprio_indices != action_indices:
        raise ValueError("shared model_dim_indices must be valid for both proprio and action")
    return proprio_indices


def _model_dim_index_maps(
    model_dim_indices: Any,
    state_model_dim_indices: Any,
    action_model_dim_indices: Any,
    *,
    state_dim: int,
    action_dim: int,
    proprio_dim: int,
    action_target_dim: int,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None, tuple[int, ...] | None]:
    if model_dim_indices is not None and (
        state_model_dim_indices is not None or action_model_dim_indices is not None
    ):
        raise ValueError(
            "model_dim_indices cannot be combined with state_model_dim_indices or action_model_dim_indices"
        )
    shared = _shared_model_dim_indices(
        model_dim_indices,
        state_dim=state_dim,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        action_target_dim=action_target_dim,
    )
    if shared is not None:
        return shared, shared, shared
    state_indices = _model_dim_indices(
        state_model_dim_indices,
        source_dim=state_dim,
        target_dim=proprio_dim,
        feature_name="state",
    )
    action_indices = _model_dim_indices(
        action_model_dim_indices,
        source_dim=action_dim,
        target_dim=action_target_dim,
        feature_name="action",
    )
    return None, state_indices, action_indices


def _validate_stats_last_dim(
    dataset_stats: dict[str, Any],
    *,
    category: str,
    key: str,
    expected_dim: int,
    semantic_name: str,
) -> None:
    category_stats = dataset_stats.get(category)
    if not hasattr(category_stats, "get"):
        return
    feature_stats = category_stats.get(key)
    if not hasattr(feature_stats, "items"):
        return
    for name, value in feature_stats.items():
        if not str(name).startswith(("global_", "stepwise_")):
            continue
        shape = tuple(value.shape) if hasattr(value, "shape") else np.asarray(value).shape
        actual_dim = shape[-1] if shape else None
        if actual_dim != expected_dim:
            raise ValueError(
                f"WB dataset_stats {category}.{key}.{name} last dim={actual_dim} "
                f"does not match semantic {semantic_name}={expected_dim}"
            )


class WBFlatCodec:
    """Latest WB flat-config normalization and semantic/model padding contract."""

    def __init__(self, data_config: Any, dataset_stats: dict[str, Any]) -> None:
        from wbwam.datasets.wb_stats import WB_STATS_VERSION, WBFlatNormalizer

        self.state_dim = int(data_config.get("state_dim"))
        self.action_dim = int(data_config.get("action_dim"))
        self.proprio_dim = int(data_config.get("proprio_dim", self.state_dim))
        self.action_target_dim = int(data_config.get("action_target_dim", self.action_dim))
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("WB semantic state/action dims must be positive")
        if self.proprio_dim < self.state_dim:
            raise ValueError(
                f"WB proprio_dim={self.proprio_dim} is smaller than semantic state_dim={self.state_dim}"
            )
        if self.action_target_dim < self.action_dim:
            raise ValueError(
                "WB action_target_dim="
                f"{self.action_target_dim} is smaller than semantic action_dim={self.action_dim}"
            )
        (
            self.model_dim_indices,
            self.state_model_dim_indices,
            self.action_model_dim_indices,
        ) = _model_dim_index_maps(
            data_config.get("model_dim_indices"),
            data_config.get("state_model_dim_indices"),
            data_config.get("action_model_dim_indices"),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            action_target_dim=self.action_target_dim,
        )

        self.action_representation = str(data_config.get("action_representation", "absolute"))
        self.relative_joint_ranges = _ranges(
            data_config.get("relative_joint_ranges"),
            dim=self.action_dim,
        )
        if self.action_representation != "relative_joint" and self.relative_joint_ranges:
            raise ValueError(f"{self.action_representation} WB data config cannot define relative_joint_ranges")
        if self.action_representation not in {"absolute", "relative_joint", "native"}:
            raise ValueError(f"unsupported WB action_representation={self.action_representation!r}")

        self.relative_action_reference = (
            str(data_config.get("relative_action_reference", "observation")).strip().lower()
        )
        if self.relative_action_reference not in RELATIVE_ACTION_REFERENCES:
            raise ValueError(f"unsupported WB relative_action_reference={self.relative_action_reference!r}")
        if self.action_representation != "relative_joint" and self.relative_action_reference != "observation":
            raise ValueError("relative_action_reference is only configurable for relative_joint actions")

        metadata = dataset_stats.get("metadata") or {}
        stats_version = int(metadata.get("wb_norm_stats_version", WB_STATS_VERSION))
        if stats_version not in SUPPORTED_WB_STATS_VERSIONS:
            raise ValueError(
                f"WB dataset_stats version={stats_version} is unsupported; "
                f"supported versions={sorted(SUPPORTED_WB_STATS_VERSIONS)}"
            )
        expected_metadata = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        if self.state_model_dim_indices is None:
            expected_metadata["proprio_dim"] = self.proprio_dim
        if self.action_model_dim_indices is None:
            expected_metadata["action_target_dim"] = self.action_target_dim
        for name, expected in expected_metadata.items():
            if name in metadata and int(metadata[name]) != expected:
                raise ValueError(f"WB dataset_stats {name}={int(metadata[name])} does not match config {expected}")
        mapped_model_widths = (
            ("proprio_dim", self.proprio_dim, self.state_model_dim_indices),
            ("action_target_dim", self.action_target_dim, self.action_model_dim_indices),
        )
        for name, expected, indices in mapped_model_widths:
            if indices is not None:
                if name not in metadata:
                    continue
                actual = int(metadata[name])
                if actual not in {expected, LEGACY_WB_MODEL_DIM}:
                    raise ValueError(
                        f"WB dataset_stats {name}={actual} does not match config {expected} "
                        f"or supported legacy width {LEGACY_WB_MODEL_DIM}"
                    )
        map_keys = {"model_dim_indices", "state_model_dim_indices", "action_model_dim_indices"}
        if map_keys.intersection(metadata):
            (
                stats_model_dim_indices,
                stats_state_model_dim_indices,
                stats_action_model_dim_indices,
            ) = _model_dim_index_maps(
                metadata.get("model_dim_indices"),
                metadata.get("state_model_dim_indices"),
                metadata.get("action_model_dim_indices"),
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                proprio_dim=self.proprio_dim,
                action_target_dim=self.action_target_dim,
            )
            if stats_model_dim_indices != self.model_dim_indices:
                raise ValueError(
                    "WB dataset_stats model_dim_indices="
                    f"{stats_model_dim_indices} do not match config {self.model_dim_indices}"
                )
            if (
                stats_state_model_dim_indices != self.state_model_dim_indices
                or stats_action_model_dim_indices != self.action_model_dim_indices
            ):
                raise ValueError("WB dataset_stats state/action model dimension maps do not match the data config")
        stats_representation = str(metadata.get("action_representation", self.action_representation))
        if stats_representation != self.action_representation:
            raise ValueError(
                "WB dataset_stats action_representation="
                f"{stats_representation!r} does not match config {self.action_representation!r}"
            )
        stats_reference = (
            str(metadata.get("relative_action_reference", self.relative_action_reference)).strip().lower()
        )
        if stats_reference not in RELATIVE_ACTION_REFERENCES:
            raise ValueError(f"unsupported WB dataset_stats relative_action_reference={stats_reference!r}")
        if stats_reference != self.relative_action_reference:
            raise ValueError(
                "WB dataset_stats relative_action_reference="
                f"{stats_reference!r} does not match config {self.relative_action_reference!r}"
            )
        if self.action_representation == "relative_joint" or self.action_model_dim_indices is not None:
            stats_ranges = _ranges(
                metadata.get("relative_joint_ranges", self.relative_joint_ranges),
                dim=self.action_dim,
            )
            if stats_ranges != self.relative_joint_ranges:
                raise ValueError(
                    "WB dataset_stats relative_joint_ranges="
                    f"{stats_ranges} do not match config {self.relative_joint_ranges}"
                )
        if self.state_model_dim_indices is not None:
            _validate_stats_last_dim(
                dataset_stats,
                category="state",
                key="proprio",
                expected_dim=self.state_dim,
                semantic_name="state_dim",
            )
        if self.action_model_dim_indices is not None:
            _validate_stats_last_dim(
                dataset_stats,
                category="action",
                key="action",
                expected_dim=self.action_dim,
                semantic_name="action_dim",
            )

        self._normalizer = WBFlatNormalizer(
            dataset_stats,
            norm_default_mode=str(data_config.get("norm_default_mode", "q01/q99")),
            norm_exception_mode=data_config.get("norm_exception_mode"),
            use_stepwise_action_norm=bool(data_config.get("use_stepwise_action_norm", True)),
            norm_missing_key_mode=str(data_config.get("norm_missing_key_mode", "error")),
            dummy_clip_default=tuple(data_config.get("dummy_clip_default", (-5.0, 5.0))),
            norm_tail_scale=float(data_config.get("norm_tail_scale", 0.075)),
        )

    @staticmethod
    def _right_pad(value: Any, target_dim: int, *, name: str):
        import torch.nn.functional as functional

        if value.shape[-1] > target_dim:
            raise ValueError(f"{name} dim {value.shape[-1]} is larger than model target_dim={target_dim}")
        pad = target_dim - int(value.shape[-1])
        return functional.pad(value, (0, pad)) if pad else value

    @staticmethod
    def _scatter(value: Any, target_dim: int, indices: tuple[int, ...], *, name: str):
        if value.shape[-1] != len(indices):
            raise ValueError(
                f"{name} dim {value.shape[-1]} does not match model_dim_indices length={len(indices)}"
            )
        result = value.new_zeros((*value.shape[:-1], target_dim))
        result[..., list(indices)] = value
        return result

    def normalize_state(self, state: np.ndarray):
        import torch

        array = np.asarray(state, dtype=np.float32).reshape(-1)
        if array.size != self.state_dim:
            raise ValueError(f"WB semantic state dim {array.size}, expected {self.state_dim}")
        value = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)
        normalized = self._normalizer.normalize_proprio(value)
        if self.state_model_dim_indices is not None:
            return self._scatter(
                normalized,
                self.proprio_dim,
                self.state_model_dim_indices,
                name="proprio",
            )
        return self._right_pad(normalized, self.proprio_dim, name="proprio")

    def denormalize_action(self, action: Any) -> np.ndarray:
        import torch

        value = action
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError(f"expected WB-WAM action [B,T,D], got {tuple(value.shape)}")
        if self.action_model_dim_indices is None:
            if value.shape[-1] < self.action_dim:
                raise ValueError(
                    f"model action dim {value.shape[-1]} is smaller than semantic action_dim={self.action_dim}"
                )
            semantic = value[..., : self.action_dim]
        elif value.shape[-1] < self.action_target_dim:
            raise ValueError(
                f"model action dim {value.shape[-1]} is smaller than required model dim={self.action_target_dim}"
            )
        else:
            semantic = value[..., list(self.action_model_dim_indices)]
        semantic = semantic.to(dtype=torch.float32, device="cpu")
        denormalize = getattr(self._normalizer, "denormalize_action", None)
        if callable(denormalize):
            result = denormalize(semantic)
        else:
            result = self._normalizer.action_normalizer.backward(semantic)
        return result.numpy()[0].astype(np.float32, copy=False)
