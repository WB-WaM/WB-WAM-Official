"""Mask-aware WB dataset normalization statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import torch

from wbwam.datasets.dataset_utils import (
    apply_relative_joint_action as _apply_relative_joint_action,
    normalize_relative_action_reference as _normalize_relative_action_reference,
)
from wbwam.datasets.normalization import SingleFieldLinearNormalizer
from wbwam.utils.logging_config import get_logger

logger = get_logger("wbwam.datasets.wb_dataset")
WB_STATS_VERSION = 3
WB_QUANTILE_SPEC: tuple[tuple[str, float], ...] = (
    ("q01", 0.01),
    ("q99", 0.99),
    ("q001", 0.001),
    ("q999", 0.999),
    ("q0001", 0.0001),
    ("q9999", 0.9999),
    ("q00001", 0.00001),
    ("q99999", 0.99999),
)


def _quantile_indices(count: int) -> list[int]:
    if count <= 0:
        return []
    return [max(0, min(int(q * (count - 1)), count - 1)) for _name, q in WB_QUANTILE_SPEC]


def _fill_quantiles_2d(
    out: dict[str, np.ndarray],
    columns: np.ndarray,
    values: np.ndarray,
) -> None:
    if columns.size == 0 or values.shape[0] == 0:
        return
    ks = _quantile_indices(int(values.shape[0]))
    partitioned = np.partition(values, ks, axis=0)
    for (name, _q), k in zip(WB_QUANTILE_SPEC, ks):
        out[name][columns] = partitioned[k].astype(np.float32, copy=False)


def _masked_stats_2d(values: np.ndarray, dim_is_pad: np.ndarray) -> dict[str, torch.Tensor]:
    if values.ndim != 2 or dim_is_pad.ndim != 2:
        raise ValueError(f"Masked stats expect 2D arrays, got values={values.shape}, mask={dim_is_pad.shape}")
    if values.shape != dim_is_pad.shape:
        raise ValueError(f"Masked stats shape mismatch: {values.shape} vs {dim_is_pad.shape}")

    values = np.asarray(values, dtype=np.float32)
    dim_is_pad = np.asarray(dim_is_pad, dtype=bool)
    rows, dim = int(values.shape[0]), int(values.shape[1])
    valid_counts = (~dim_is_pad).sum(axis=0).astype(np.int64)
    out: dict[str, np.ndarray] = {
        "min": np.zeros((dim,), dtype=np.float32),
        "max": np.zeros((dim,), dtype=np.float32),
        "mean": np.zeros((dim,), dtype=np.float32),
        "std": np.zeros((dim,), dtype=np.float32),
        "count": valid_counts.astype(np.float32),
    }
    for name, _ in WB_QUANTILE_SPEC:
        out[name] = np.zeros((dim,), dtype=np.float32)

    if rows == 0 or dim == 0:
        return {key: torch.from_numpy(value) for key, value in out.items()}

    full_columns = np.flatnonzero(valid_counts == rows)
    if full_columns.size:
        cur = values[:, full_columns]
        out["min"][full_columns] = cur.min(axis=0)
        out["max"][full_columns] = cur.max(axis=0)
        out["mean"][full_columns] = cur.mean(axis=0)
        out["std"][full_columns] = cur.std(axis=0, ddof=0)
        _fill_quantiles_2d(out, full_columns, cur)

    mixed_columns = np.flatnonzero((valid_counts > 0) & (valid_counts < rows))
    for dim_idx in mixed_columns:
        valid = ~dim_is_pad[:, dim_idx]
        cur = values[valid, dim_idx]
        out["min"][dim_idx] = float(cur.min())
        out["max"][dim_idx] = float(cur.max())
        out["mean"][dim_idx] = float(cur.mean())
        out["std"][dim_idx] = float(cur.std(ddof=0))
        ks = _quantile_indices(int(cur.shape[0]))
        partitioned = np.partition(cur, ks)
        for (name, _q), k in zip(WB_QUANTILE_SPEC, ks):
            out[name][dim_idx] = float(partitioned[k])

    return {key: torch.from_numpy(value) for key, value in out.items()}


def _state_stats_from_arrays(
    states: np.ndarray,
    state_dim_is_pad: np.ndarray,
    stats_downsample_rate: int = 1,
) -> dict[str, torch.Tensor]:
    rate = max(1, int(stats_downsample_rate))
    if rate > 1:
        states = states[::rate]
        state_dim_is_pad = state_dim_is_pad[::rate]
    base = _masked_stats_2d(states, state_dim_is_pad)
    stats: dict[str, torch.Tensor] = {}
    for key, value in base.items():
        stats[f"stepwise_{key}"] = value.unsqueeze(0)
        stats[f"global_{key}"] = value
    return stats


def _episode_offsets(
    episode_lengths: list[int],
    *,
    total_rows: int,
) -> tuple[list[int], int]:
    offsets: list[int] = []
    total = 0
    for length in episode_lengths:
        offsets.append(total)
        total += int(length)
    if total != total_rows:
        raise ValueError(f"Episode lengths sum to {total}, but actions has {total_rows} rows.")
    return offsets, total


def _downsampled_action_indices(
    total: int,
    episode_lengths: list[int],
    rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    base_indices = np.arange(0, total, rate, dtype=np.int64)
    episode_ends = np.cumsum(np.asarray(episode_lengths, dtype=np.int64))
    if base_indices.size:
        base_episode_indices = np.searchsorted(episode_ends, base_indices, side="right")
        base_episode_ends = episode_ends[base_episode_indices]
    else:
        base_episode_ends = np.asarray([], dtype=np.int64)
    return base_indices, base_episode_ends


def _empty_action_step(action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((0, action_dim), dtype=np.float32),
        np.ones((0, action_dim), dtype=bool),
    )


def _full_rate_action_step(
    *,
    actions: np.ndarray,
    action_dim_is_pad: np.ndarray,
    states: np.ndarray,
    state_dim_is_pad: np.ndarray,
    offsets: list[int],
    episode_lengths: list[int],
    step: int,
    include_base_state: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    value_parts = []
    mask_parts = []
    base_state_parts = []
    base_state_mask_parts = []
    for offset, length in zip(offsets, episode_lengths):
        length = int(length)
        if length <= step:
            continue
        step_slice = slice(int(offset) + step, int(offset) + length)
        value_parts.append(actions[step_slice])
        mask_parts.append(action_dim_is_pad[step_slice])
        if include_base_state:
            base_slice = slice(int(offset), int(offset) + length - step)
            base_state_parts.append(states[base_slice])
            base_state_mask_parts.append(state_dim_is_pad[base_slice])

    if not value_parts:
        step_values, step_mask = _empty_action_step(actions.shape[1])
        if include_base_state:
            return step_values, step_mask, np.zeros_like(step_values), np.ones_like(step_mask)
        return step_values, step_mask, None, None

    step_values = np.concatenate(value_parts, axis=0)
    step_mask = np.concatenate(mask_parts, axis=0)
    if not include_base_state:
        return step_values, step_mask, None, None
    return (
        step_values,
        step_mask,
        np.concatenate(base_state_parts, axis=0),
        np.concatenate(base_state_mask_parts, axis=0),
    )


def _downsampled_action_step(
    *,
    actions: np.ndarray,
    action_dim_is_pad: np.ndarray,
    states: np.ndarray,
    state_dim_is_pad: np.ndarray,
    base_indices: np.ndarray,
    base_episode_ends: np.ndarray,
    step: int,
    include_base_state: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    valid = base_indices + step < base_episode_ends
    if not np.any(valid):
        step_values, step_mask = _empty_action_step(actions.shape[1])
        if include_base_state:
            return step_values, step_mask, np.zeros_like(step_values), np.ones_like(step_mask)
        return step_values, step_mask, None, None

    current_indices = base_indices[valid]
    indices = current_indices + step
    current_states = states[current_indices] if include_base_state else None
    current_state_mask = state_dim_is_pad[current_indices] if include_base_state else None
    return actions[indices], action_dim_is_pad[indices], current_states, current_state_mask


def _stack_stepwise_action_stats(
    per_step: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    stats: dict[str, torch.Tensor] = {}
    step_keys = ["min", "max", "mean", "std", "count"] + [name for name, _ in WB_QUANTILE_SPEC]
    for key in step_keys:
        stats[f"stepwise_{key}"] = torch.stack([step_stats[key] for step_stats in per_step], dim=0)
    return stats


def _global_action_arrays(
    actions: np.ndarray,
    action_dim_is_pad: np.ndarray,
    base_indices: np.ndarray,
    rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    if rate == 1:
        return actions, action_dim_is_pad
    if base_indices.size:
        return actions[base_indices], action_dim_is_pad[base_indices]
    return _empty_action_step(actions.shape[1])


def _action_stats_from_arrays(
    actions: np.ndarray,
    action_dim_is_pad: np.ndarray,
    states: np.ndarray,
    state_dim_is_pad: np.ndarray,
    episode_lengths: list[int],
    action_horizon: int,
    stats_downsample_rate: int = 1,
    relative_joint_ranges: Iterable[tuple[int, int]] = (),
    relative_action_reference: str = "observation",
) -> dict[str, torch.Tensor]:
    if actions.shape != action_dim_is_pad.shape:
        raise ValueError(f"Action/mask shape mismatch: {actions.shape} vs {action_dim_is_pad.shape}")
    relative_joint_ranges = list(relative_joint_ranges)
    relative_action_reference = _normalize_relative_action_reference(relative_action_reference)
    if (
        relative_joint_ranges
        and relative_action_reference == "observation"
        and (states.shape != state_dim_is_pad.shape or states.shape != actions.shape)
    ):
        raise ValueError(
            f"Relative action stats require matching state/action/mask shapes, got "
            f"states={states.shape}, state_mask={state_dim_is_pad.shape}, actions={actions.shape}."
        )
    reference_values = actions if relative_action_reference == "first_action" else states
    reference_dim_is_pad = action_dim_is_pad if relative_action_reference == "first_action" else state_dim_is_pad

    offsets, total = _episode_offsets(
        episode_lengths,
        total_rows=int(actions.shape[0]),
    )
    rate = max(1, int(stats_downsample_rate))
    base_indices, base_episode_ends = _downsampled_action_indices(
        total,
        episode_lengths,
        rate,
    )

    per_step: list[dict[str, torch.Tensor]] = []
    include_base_state = bool(relative_joint_ranges)
    for step in range(action_horizon):
        if rate == 1:
            step_values, step_mask, current_states, current_state_mask = _full_rate_action_step(
                actions=actions,
                action_dim_is_pad=action_dim_is_pad,
                states=reference_values,
                state_dim_is_pad=reference_dim_is_pad,
                offsets=offsets,
                episode_lengths=episode_lengths,
                step=step,
                include_base_state=include_base_state,
            )
        else:
            step_values, step_mask, current_states, current_state_mask = _downsampled_action_step(
                actions=actions,
                action_dim_is_pad=action_dim_is_pad,
                states=reference_values,
                state_dim_is_pad=reference_dim_is_pad,
                base_indices=base_indices,
                base_episode_ends=base_episode_ends,
                step=step,
                include_base_state=include_base_state,
            )
        if relative_joint_ranges:
            assert current_states is not None
            assert current_state_mask is not None
            step_values, step_mask = _apply_relative_joint_action(
                step_values,
                step_mask,
                current_states,
                current_state_mask,
                relative_joint_ranges,
            )
        per_step.append(_masked_stats_2d(step_values, step_mask))

    stats = _stack_stepwise_action_stats(per_step)

    # A fixed-base relative target depends on both horizon step and its
    # reference. It has no unique per-frame global representation.
    if relative_joint_ranges:
        return stats

    global_values, global_mask = _global_action_arrays(
        actions,
        action_dim_is_pad,
        base_indices,
        rate,
    )
    global_base = _masked_stats_2d(global_values, global_mask)
    for key, value in global_base.items():
        stats[f"global_{key}"] = value
    return stats


@dataclass(frozen=True)
class WBStatsArrays:
    """Raw arrays needed to compute mask-aware statistics for one dataset."""

    states: np.ndarray
    actions: np.ndarray
    state_dim_is_pad: np.ndarray
    action_dim_is_pad: np.ndarray
    episode_lengths: tuple[int, ...]
    name: str = ""
    stats_downsample_rate: int = 1

    def __post_init__(self) -> None:
        if self.states.ndim != 2 or self.actions.ndim != 2:
            raise ValueError(
                f"WB stats arrays must be 2D, got states={self.states.shape}, actions={self.actions.shape}."
            )
        if self.states.shape != self.state_dim_is_pad.shape:
            raise ValueError(f"State/mask shape mismatch: {self.states.shape} vs {self.state_dim_is_pad.shape}")
        if self.actions.shape != self.action_dim_is_pad.shape:
            raise ValueError(f"Action/mask shape mismatch: {self.actions.shape} vs {self.action_dim_is_pad.shape}")
        if self.states.shape[0] != self.actions.shape[0]:
            raise ValueError(f"State/action row mismatch: {self.states.shape[0]} vs {self.actions.shape[0]}")
        _episode_offsets(
            list(self.episode_lengths),
            total_rows=int(self.actions.shape[0]),
        )


def _concatenate_2d(parts: list[np.ndarray], *, dim: int, dtype: np.dtype) -> np.ndarray:
    if not parts:
        return np.zeros((0, dim), dtype=dtype)
    if len(parts) == 1:
        return np.asarray(parts[0], dtype=dtype)
    return np.concatenate(parts, axis=0, dtype=dtype)


def _global_wb_stats_from_arrays(
    datasets: Iterable[WBStatsArrays],
    *,
    metadata: dict[str, Any],
    action_horizon: int,
    relative_joint_ranges: Iterable[tuple[int, int]] = (),
    relative_action_reference: str = "observation",
) -> dict[str, Any]:
    """Compute exact stats over the union of all valid, downsampled values."""

    datasets = list(datasets)
    if not datasets:
        raise ValueError("Exact-global WB stats require at least one dataset.")
    if action_horizon <= 0:
        raise ValueError(f"WB action_horizon must be positive, got {action_horizon}.")

    state_dim = int(datasets[0].states.shape[1])
    action_dim = int(datasets[0].actions.shape[1])
    for dataset in datasets:
        if dataset.states.shape[1] != state_dim or dataset.actions.shape[1] != action_dim:
            raise ValueError(
                "Exact-global WB stats require consistent feature dimensions, got "
                f"state/action={dataset.states.shape[1]}/{dataset.actions.shape[1]} "
                f"for {dataset.name or '<unnamed>'} and expected {state_dim}/{action_dim}."
            )

    relative_joint_ranges = list(relative_joint_ranges)
    relative_action_reference = _normalize_relative_action_reference(relative_action_reference)
    if relative_joint_ranges and relative_action_reference == "observation" and state_dim != action_dim:
        raise ValueError(
            f"Relative action stats require matching state/action dims, got {state_dim} and {action_dim}."
        )

    rates = [max(1, int(dataset.stats_downsample_rate)) for dataset in datasets]
    logger.info(
        "Computing exact-global WB stats: datasets=%d state_dim=%d action_dim=%d horizon=%d rates=%s",
        len(datasets),
        state_dim,
        action_dim,
        action_horizon,
        rates,
    )
    state_parts = []
    state_mask_parts = []
    for dataset, rate in zip(datasets, rates):
        state_parts.append(dataset.states[::rate])
        state_mask_parts.append(dataset.state_dim_is_pad[::rate])
    states = _concatenate_2d(state_parts, dim=state_dim, dtype=np.dtype(np.float32))
    state_mask = _concatenate_2d(state_mask_parts, dim=state_dim, dtype=np.dtype(bool))
    state_stats = _state_stats_from_arrays(states, state_mask)
    del states, state_mask, state_parts, state_mask_parts

    prepared = []
    for dataset in datasets:
        rate = max(1, int(dataset.stats_downsample_rate))
        offsets, total = _episode_offsets(
            list(dataset.episode_lengths),
            total_rows=int(dataset.actions.shape[0]),
        )
        base_indices, base_episode_ends = _downsampled_action_indices(
            total,
            list(dataset.episode_lengths),
            rate,
        )
        prepared.append((dataset, rate, offsets, base_indices, base_episode_ends))

    per_step = []
    include_base_state = bool(relative_joint_ranges)
    for step in range(action_horizon):
        value_parts = []
        mask_parts = []
        for dataset, rate, offsets, base_indices, base_episode_ends in prepared:
            if relative_action_reference == "first_action":
                reference_values = dataset.actions
                reference_dim_is_pad = dataset.action_dim_is_pad
            else:
                reference_values = dataset.states
                reference_dim_is_pad = dataset.state_dim_is_pad
            if rate == 1:
                values, mask, current_states, current_state_mask = _full_rate_action_step(
                    actions=dataset.actions,
                    action_dim_is_pad=dataset.action_dim_is_pad,
                    states=reference_values,
                    state_dim_is_pad=reference_dim_is_pad,
                    offsets=offsets,
                    episode_lengths=list(dataset.episode_lengths),
                    step=step,
                    include_base_state=include_base_state,
                )
            else:
                values, mask, current_states, current_state_mask = _downsampled_action_step(
                    actions=dataset.actions,
                    action_dim_is_pad=dataset.action_dim_is_pad,
                    states=reference_values,
                    state_dim_is_pad=reference_dim_is_pad,
                    base_indices=base_indices,
                    base_episode_ends=base_episode_ends,
                    step=step,
                    include_base_state=include_base_state,
                )
            if relative_joint_ranges:
                assert current_states is not None
                assert current_state_mask is not None
                values, mask = _apply_relative_joint_action(
                    values,
                    mask,
                    current_states,
                    current_state_mask,
                    relative_joint_ranges,
                )
            value_parts.append(values)
            mask_parts.append(mask)

        values = _concatenate_2d(value_parts, dim=action_dim, dtype=np.dtype(np.float32))
        mask = _concatenate_2d(mask_parts, dim=action_dim, dtype=np.dtype(bool))
        per_step.append(_masked_stats_2d(values, mask))
        logger.info(
            "Finished exact-global WB action stats step=%d/%d rows=%d",
            step + 1,
            action_horizon,
            values.shape[0],
        )
        del values, mask, value_parts, mask_parts

    action_stats = _stack_stepwise_action_stats(per_step)
    if not relative_joint_ranges:
        value_parts = []
        mask_parts = []
        for dataset, rate, _offsets, base_indices, _base_episode_ends in prepared:
            values, mask = _global_action_arrays(
                dataset.actions,
                dataset.action_dim_is_pad,
                base_indices,
                rate,
            )
            value_parts.append(values)
            mask_parts.append(mask)
        values = _concatenate_2d(value_parts, dim=action_dim, dtype=np.dtype(np.float32))
        mask = _concatenate_2d(mask_parts, dim=action_dim, dtype=np.dtype(bool))
        for key, value in _masked_stats_2d(values, mask).items():
            action_stats[f"global_{key}"] = value

    exact_metadata = dict(metadata)
    exact_metadata.update(
        {
            "wb_norm_stats_version": WB_STATS_VERSION,
            "aggregation": "exact_global",
            "quantile_source": (
                "raw_valid_values"
                if all(rate == 1 for rate in rates)
                else "deterministically_downsampled_raw_valid_values"
            ),
            "norm_distribution": "natural_valid_frame_distribution",
        }
    )
    return {
        "state": {"proprio": state_stats},
        "action": {"action": action_stats},
        "metadata": exact_metadata,
    }


class WBFlatNormalizer:
    def __init__(
        self,
        stats: dict[str, Any],
        *,
        norm_default_mode: str = "q01/q99",
        norm_exception_mode: Optional[dict[str, dict[str, str]]] = None,
        use_stepwise_action_norm: bool = True,
        norm_missing_key_mode: str = "error",
        dummy_clip_default: tuple[float, float] = (-5.0, 5.0),
        norm_tail_scale: float = 0.075,
    ):
        self.stats = stats
        self.norm_default_mode = str(norm_default_mode)
        self.norm_exception_mode = norm_exception_mode or {}
        self.use_stepwise_action_norm = bool(use_stepwise_action_norm)
        self.norm_missing_key_mode = str(norm_missing_key_mode)
        self.state_normalizer = self._build_normalizer(
            category="state",
            key="proprio",
            use_stepwise=False,
            dummy_clip_default=dummy_clip_default,
            norm_tail_scale=norm_tail_scale,
        )
        self.action_normalizer = self._build_normalizer(
            category="action",
            key="action",
            use_stepwise=self.use_stepwise_action_norm,
            dummy_clip_default=dummy_clip_default,
            norm_tail_scale=norm_tail_scale,
        )

    def _mode_for(self, category: str, key: str) -> str:
        category_exceptions = self.norm_exception_mode.get(category) or {}
        return str(category_exceptions.get(key, self.norm_default_mode))

    def _select_stats(self, category: str, key: str, *, use_stepwise: bool) -> dict[str, torch.Tensor]:
        if category not in self.stats or key not in self.stats[category]:
            if self.norm_missing_key_mode == "error":
                raise KeyError(f"Missing WB norm stats for {category}.{key}")
            logger.warning("Missing WB norm stats for %s.%s; using dummy normalizer.", category, key)
            return {}
        prefix = "stepwise_" if use_stepwise else "global_"
        selected = {
            name.removeprefix(prefix): value
            for name, value in self.stats[category][key].items()
            if name.startswith(prefix)
        }
        if not selected and self.norm_missing_key_mode == "error":
            raise KeyError(f"Missing `{prefix}*` WB norm stats for {category}.{key}")
        return selected

    def _build_normalizer(
        self,
        *,
        category: str,
        key: str,
        use_stepwise: bool,
        dummy_clip_default: tuple[float, float],
        norm_tail_scale: float,
    ) -> SingleFieldLinearNormalizer:
        mode = self._mode_for(category, key)
        selected = self._select_stats(category, key, use_stepwise=use_stepwise)
        if not selected and self.norm_missing_key_mode != "error":
            mode = "dummy"
        return SingleFieldLinearNormalizer(
            stats=selected,
            mode=mode,
            dummy_clip_range=dummy_clip_default,
            tail_scale=norm_tail_scale,
        )

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.state_normalizer.forward(proprio)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_normalizer.forward(action)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_normalizer.backward(action)
