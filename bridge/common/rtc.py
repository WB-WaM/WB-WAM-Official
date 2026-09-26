from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

RTC_MODE_AUTO = "auto"
RTC_MODE_OFF = "off"
RTC_MODE_TRAINING_PREFIX = "training_prefix"
RTC_MODE_INFERENCE_GUIDANCE = "inference_guidance"
RTC_MODES = {
    RTC_MODE_AUTO,
    RTC_MODE_OFF,
    RTC_MODE_TRAINING_PREFIX,
    RTC_MODE_INFERENCE_GUIDANCE,
}


@dataclass(frozen=True)
class RtcPrefix:
    actions: np.ndarray
    mask: np.ndarray
    source_indices: tuple[int | None, ...]


def normalize_rtc_mode(value: object) -> str:
    if value is None:
        return RTC_MODE_AUTO
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"", "none", "null"}:
        return RTC_MODE_AUTO
    if normalized in {"0", "false", "no", "disabled", "disable"}:
        return RTC_MODE_OFF
    if normalized in {"1", "true", "yes", "enabled", "enable", "rtc"}:
        return RTC_MODE_AUTO
    if normalized in {"training", "prefix", "training_prefix", "train_prefix"}:
        return RTC_MODE_TRAINING_PREFIX
    if normalized in {"inference", "guidance", "inference_guidance", "inference_time", "inference_time_guidance"}:
        return RTC_MODE_INFERENCE_GUIDANCE
    if normalized not in RTC_MODES:
        raise ValueError(
            f"runtime.rtc_mode must be one of auto, off, training_prefix, or inference_guidance; got {value!r}"
        )
    return normalized


def rtc_mode_uses_overlap(mode: str | None) -> bool:
    return normalize_rtc_mode(mode) in {RTC_MODE_TRAINING_PREFIX, RTC_MODE_INFERENCE_GUIDANCE}


def estimate_rtc_prefix_steps(
    runtime_config: dict,
    *,
    inference_duration_s: float | None,
    publish_rate_hz: float,
    max_delay_steps: int,
    action_horizon: int,
) -> int:
    override = runtime_config.get("rtc_prefix_steps")
    limit = max(0, min(int(max_delay_steps), int(action_horizon)))
    if override is not None:
        return max(0, min(int(override), limit))
    fixed_delay = runtime_config.get("rtc_fixed_delay")
    if fixed_delay is not None:
        return max(0, min(int(fixed_delay), limit))
    if inference_duration_s is None or inference_duration_s <= 0 or publish_rate_hz <= 0:
        return 0
    return max(0, min(int(math.ceil(inference_duration_s * publish_rate_hz)), limit))


def build_rtc_prefix_info(
    cached_raw_chunk: np.ndarray | None,
    *,
    chunk_index: int,
    action_horizon: int,
    action_dim: int,
    prefix_steps: int,
    last_raw_action: np.ndarray | None = None,
) -> RtcPrefix | None:
    prefix_steps = max(0, min(int(prefix_steps), int(action_horizon)))
    if prefix_steps <= 0:
        return None

    prefix = np.zeros((action_horizon, action_dim), dtype=np.float32)
    mask = np.zeros((action_horizon,), dtype=np.bool_)
    source_indices: list[int | None] = []
    chunk = None if cached_raw_chunk is None else np.asarray(cached_raw_chunk, dtype=np.float32)
    fallback = None if last_raw_action is None else np.asarray(last_raw_action, dtype=np.float32).reshape(-1)
    if fallback is not None and fallback.size != action_dim:
        raise ValueError(f"last_raw_action dim {fallback.size}, expected {action_dim}")

    for i in range(prefix_steps):
        action = None
        if chunk is not None:
            source_index = int(chunk_index) + i
            if 0 <= source_index < chunk.shape[0]:
                action = chunk[source_index]
                source_indices.append(source_index)
        if action is None:
            action = fallback
            source_indices.append(None)
        if action is None:
            break

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size != action_dim:
            raise ValueError(f"rtc prefix action dim {action_arr.size}, expected {action_dim}")
        if not np.all(np.isfinite(action_arr)):
            raise ValueError("rtc prefix action contains NaN/Inf")
        prefix[i] = action_arr
        mask[i] = True
        fallback = action_arr

    if not bool(np.any(mask)):
        return None
    return RtcPrefix(actions=prefix, mask=mask, source_indices=tuple(source_indices[: int(mask.sum())]))


def build_rtc_prefix(
    cached_raw_chunk: np.ndarray | None,
    *,
    chunk_index: int,
    action_horizon: int,
    action_dim: int,
    prefix_steps: int,
    last_raw_action: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    prefix = build_rtc_prefix_info(
        cached_raw_chunk,
        chunk_index=chunk_index,
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_steps=prefix_steps,
        last_raw_action=last_raw_action,
    )
    if prefix is None:
        return None, None
    return prefix.actions, prefix.mask


def build_rtc_guidance_target(
    cached_raw_chunk: np.ndarray | None,
    *,
    chunk_index: int,
    action_horizon: int,
    action_dim: int,
    prefix_steps: int,
    last_raw_action: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    prefix = build_rtc_prefix_info(
        cached_raw_chunk,
        chunk_index=chunk_index,
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_steps=prefix_steps,
        last_raw_action=last_raw_action,
    )
    if prefix is None:
        return None, None
    mask = np.zeros((action_horizon, action_dim), dtype=np.float32)
    mask[prefix.mask] = 1.0
    return prefix.actions, mask


def summarize_action_delta(
    lhs: np.ndarray | None,
    rhs: np.ndarray | None,
    *,
    token_dim: int = 64,
    hand_dim: int = 20,
) -> str:
    if lhs is None or rhs is None:
        return "n/a"
    a = np.asarray(lhs, dtype=np.float32).reshape(-1)
    b = np.asarray(rhs, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        return f"shape_mismatch {a.shape}!={b.shape}"
    delta = np.abs(b - a)
    left_start = token_dim
    right_start = token_dim + hand_dim
    slices = {
        "all": delta,
        "token": delta[:token_dim],
        "left": delta[left_start:right_start],
        "right": delta[right_start : right_start + hand_dim],
    }
    parts = []
    for name, values in slices.items():
        if values.size == 0:
            continue
        parts.append(f"{name}_max={float(values.max()):.4f} {name}_mean={float(values.mean()):.4f}")
    return " ".join(parts)
