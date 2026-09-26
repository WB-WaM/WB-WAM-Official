from __future__ import annotations

from dataclasses import dataclass

import numpy as np

JOINT_DIM = 29
HAND_DIM = 20
ROOT_ROTATION_SHAPE = (3, 3)


def _matrix(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [H,{width}], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} horizon must be positive")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return np.ascontiguousarray(array)


def rotation_from_quat_wxyz(value: np.ndarray) -> np.ndarray:
    """Return the world-from-body rotation for a normalized wxyz quaternion."""

    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"base_quat_wxyz must have shape (4,), got {quat.shape}")
    if not np.isfinite(quat).all():
        raise ValueError("base_quat_wxyz contains NaN/Inf")
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-8:
        raise ValueError("base_quat_wxyz has zero norm")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _heading_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != ROOT_ROTATION_SHAPE:
        raise ValueError(f"rotation must have shape {ROOT_ROTATION_SHAPE}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("rotation contains NaN/Inf")
    yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class G1MotionChunk:
    """Canonical future G1 motion consumed by the SONIC G1 encoder.

    Joint positions and velocities are physical absolute values in IsaacLab
    joint order. ``root_rotation`` is the absolute root rotation for each
    future step; encoder input construction converts it to root-relative 6D.
    """

    q: np.ndarray
    dq: np.ndarray
    root_rotation: np.ndarray
    timestep_s: float
    left_hand_qpos: np.ndarray | None = None
    right_hand_qpos: np.ndarray | None = None

    def __post_init__(self) -> None:
        q = _matrix(self.q, name="q", width=JOINT_DIM)
        dq = _matrix(self.dq, name="dq", width=JOINT_DIM)
        if dq.shape != q.shape:
            raise ValueError(f"dq shape {dq.shape} does not match q shape {q.shape}")
        rotation = np.asarray(self.root_rotation, dtype=np.float32)
        expected = (q.shape[0], *ROOT_ROTATION_SHAPE)
        if rotation.shape != expected:
            raise ValueError(f"root_rotation must have shape {expected}, got {rotation.shape}")
        if not np.isfinite(rotation).all():
            raise ValueError("root_rotation contains NaN/Inf")
        timestep_s = float(self.timestep_s)
        if not np.isfinite(timestep_s) or timestep_s <= 0.0:
            raise ValueError("timestep_s must be finite and > 0")
        left = self._hand(self.left_hand_qpos, name="left_hand_qpos", horizon=q.shape[0])
        right = self._hand(self.right_hand_qpos, name="right_hand_qpos", horizon=q.shape[0])
        if (left is None) != (right is None):
            raise ValueError("left and right hand qpos must either both be present or both be absent")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "dq", dq)
        object.__setattr__(self, "root_rotation", np.ascontiguousarray(rotation))
        object.__setattr__(self, "timestep_s", timestep_s)
        object.__setattr__(self, "left_hand_qpos", left)
        object.__setattr__(self, "right_hand_qpos", right)

    @staticmethod
    def _hand(value: np.ndarray | None, *, name: str, horizon: int) -> np.ndarray | None:
        if value is None:
            return None
        array = _matrix(value, name=name, width=HAND_DIM)
        if array.shape[0] != horizon:
            raise ValueError(f"{name} horizon {array.shape[0]} does not match body horizon {horizon}")
        return array

    @property
    def horizon(self) -> int:
        return int(self.q.shape[0])

    def relative_root_6d(self, base_indices: np.ndarray, future_indices: np.ndarray) -> np.ndarray:
        base = self.root_rotation[np.asarray(base_indices, dtype=np.int64)]
        future = self.root_rotation[np.asarray(future_indices, dtype=np.int64)]
        relative = np.swapaxes(base, -1, -2) @ future
        return relative[..., :, :2].reshape(relative.shape[:-2] + (6,)).astype(np.float32)

    def heading_alignment_rotation(
        self,
        actual_base_quat_wxyz: np.ndarray,
        *,
        reference_index: int = 0,
    ) -> np.ndarray:
        """Align the reference heading once when a motion chunk is activated."""

        index = int(reference_index)
        if not 0 <= index < self.horizon:
            raise IndexError(f"reference_index {index} outside motion horizon {self.horizon}")
        actual_heading = _heading_rotation(rotation_from_quat_wxyz(actual_base_quat_wxyz))
        reference_heading = _heading_rotation(self.root_rotation[index])
        return np.ascontiguousarray(actual_heading @ reference_heading.T, dtype=np.float32)

    def actual_root_6d(
        self,
        actual_base_quat_wxyz: np.ndarray,
        future_indices: np.ndarray,
        *,
        heading_alignment_rotation: np.ndarray,
    ) -> np.ndarray:
        """Encode aligned future rotations relative to the latest measured base."""

        alignment = np.asarray(heading_alignment_rotation, dtype=np.float32)
        if alignment.shape != ROOT_ROTATION_SHAPE:
            raise ValueError(
                f"heading_alignment_rotation must have shape {ROOT_ROTATION_SHAPE}, got {alignment.shape}"
            )
        if not np.isfinite(alignment).all():
            raise ValueError("heading_alignment_rotation contains NaN/Inf")
        actual = rotation_from_quat_wxyz(actual_base_quat_wxyz)
        future = self.root_rotation[np.asarray(future_indices, dtype=np.int64)]
        aligned_future = alignment @ future
        relative = actual.T @ aligned_future
        return relative[..., :, :2].reshape(relative.shape[:-2] + (6,)).astype(np.float32)
