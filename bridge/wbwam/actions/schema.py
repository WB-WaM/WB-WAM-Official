from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bridge.sonic.motion_schema import HAND_DIM, JOINT_DIM

ROOT3_DIM = 3
PHYSICAL_ACTION_DIM = HAND_DIM * 2 + JOINT_DIM + ROOT3_DIM


def _slice(value: Any, *, name: str, default: tuple[int, int], width: int) -> slice:
    if value is None:
        start, stop = default
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, stop = int(value[0]), int(value[1])
    else:
        raise ValueError(f"{name} must be [start, stop], got {value!r}")
    if start < 0 or stop <= start or stop - start != width:
        raise ValueError(f"{name} must select exactly {width} values, got [{start}, {stop}]")
    return slice(start, stop)


@dataclass(frozen=True)
class PhysicalActionLayout:
    """Latest WB 72D semantic layout after applying ``data.action_slices``."""

    body_q: slice = field(default_factory=lambda: slice(0, 29))
    root3: slice = field(default_factory=lambda: slice(29, 32))
    left_hand: slice = field(default_factory=lambda: slice(32, 52))
    right_hand: slice = field(default_factory=lambda: slice(52, 72))

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "PhysicalActionLayout":
        values = dict(config or {})
        return cls(
            body_q=_slice(values.get("body_q_slice"), name="body_q_slice", default=(0, 29), width=29),
            root3=_slice(values.get("root3_slice"), name="root3_slice", default=(29, 32), width=3),
            left_hand=_slice(values.get("left_hand_slice"), name="left_hand_slice", default=(32, 52), width=20),
            right_hand=_slice(values.get("right_hand_slice"), name="right_hand_slice", default=(52, 72), width=20),
        )

    @property
    def minimum_dim(self) -> int:
        return max(self.left_hand.stop, self.right_hand.stop, self.body_q.stop, self.root3.stop)


@dataclass(frozen=True)
class CurrentG1State:
    body_q_delta: np.ndarray
    body_dq: np.ndarray
    left_hand_qpos: np.ndarray
    right_hand_qpos: np.ndarray
    roll: float
    pitch: float
    yaw_velocity: float


@dataclass(frozen=True)
class PhysicalActionChunk:
    left_hand_qpos: np.ndarray
    right_hand_qpos: np.ndarray
    body_q_delta: np.ndarray
    root3: np.ndarray

    @classmethod
    def unpack(cls, actions: np.ndarray, layout: PhysicalActionLayout) -> "PhysicalActionChunk":
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(f"physical actions must have shape [H,D], got {array.shape}")
        if array.shape[1] < layout.minimum_dim:
            raise ValueError(f"physical action dim {array.shape[1]} is smaller than required {layout.minimum_dim}")
        if not np.isfinite(array).all():
            raise ValueError("physical actions contain NaN/Inf")
        return cls(
            left_hand_qpos=np.ascontiguousarray(array[:, layout.left_hand]),
            right_hand_qpos=np.ascontiguousarray(array[:, layout.right_hand]),
            body_q_delta=np.ascontiguousarray(array[:, layout.body_q]),
            root3=np.ascontiguousarray(array[:, layout.root3]),
        )
