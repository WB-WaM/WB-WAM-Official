"""Rotation and gravity transform utilities for data collection."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_to_rot6d(q):
    """Convert scalar-first quaternion(s) (wxyz) to 6D rotation representation.

    The 6D representation consists of the first two columns of the rotation
    matrix, flattened (Zhou et al., CVPR 2019).

    Accepted input shapes:
        * ``(4,)``   -- single quaternion  -> returns ``(6,)``
        * ``(N, 4)`` -- batch of quats     -> returns ``(N, 6)``
        * ``(N*4,)`` -- flat concatenated  -> returns ``(N*6,)``
    """
    q = np.asarray(q)
    if q.ndim == 1 and q.shape[0] > 4:
        assert q.shape[0] % 4 == 0, f"Flat quat length {q.shape[0]} is not divisible by 4"
        q = q.reshape(-1, 4)
        rot_6d = quat_to_rot6d(q)
        return rot_6d.ravel()

    single = q.ndim == 1
    q = np.atleast_2d(q)
    q_xyzw = q[:, [1, 2, 3, 0]]
    rot_mat = R.from_quat(q_xyzw).as_matrix()  # (N, 3, 3)
    rot_6d = rot_mat[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)  # (N, 6)
    if single:
        return rot_6d[0].astype(q.dtype)
    return rot_6d.astype(q.dtype)


def compute_projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """Compute projected gravity vector in robot's body frame from base quaternion.

    Projects the world gravity vector [0, 0, -1] into the robot's body frame by
    rotating it by the inverse of the base quaternion.

    Args:
        base_quat: Base quaternion [qw, qx, qy, qz] of shape (4,)

    Returns:
        Projected gravity vector [gx, gy, gz] of shape (3,) in robot's body frame
    """
    base_quat = np.asarray(base_quat, dtype=np.float64)
    if base_quat.shape != (4,):
        raise ValueError(f"base_quat must have shape (4,), got {base_quat.shape}")

    gravity_vec_world = np.array([0.0, 0.0, -1.0])
    base_rotation = R.from_quat(base_quat, scalar_first=True)
    projected_gravity = base_rotation.inv().apply(gravity_vec_world)

    return projected_gravity.astype(np.float32)
