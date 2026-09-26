import numpy as np
from scipy.spatial.transform import Rotation as R


def WarpPi(ang):
    return np.arctan2(np.sin(ang), np.cos(ang))


def quat2yaw(q_wxyz: np.ndarray) -> np.ndarray:
    """Extract heading yaw via arctan2(R[1,0], R[0,0]).

    Matches the training-time ``transforms_jax.quat2yaw`` exactly.

    Args:
        q_wxyz: (..., 4) quaternions in (w, x, y, z) order.
    Returns:
        yaw: (...,) heading angles in radians, wrapped to [-pi, pi].
    """
    w = q_wxyz[..., 0]
    x = q_wxyz[..., 1]
    y = q_wxyz[..., 2]
    z = q_wxyz[..., 3]
    R00 = 1.0 - 2.0 * (y * y + z * z)
    R10 = 2.0 * (x * y + w * z)
    return WarpPi(np.arctan2(R10, R00))

def wxyz2xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """(…,4) wxyz -> xyzw"""
    return np.roll(q_wxyz, -1, axis=-1)


def xyzw2wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """(…,4) xyzw -> wxyz"""
    return np.roll(q_xyzw, 1, axis=-1)


def quat2mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert batch of quaternions (w, x, y, z) to rotation matrices (3x3).
    Args:
        quat_wxyz: (N, 4) array of quaternions (w, x, y, z)
    Returns:
        rot_mats: (N, 3, 3) array of rotation matrices

    Uses a pure-numpy formula for small batches (N <= 4): scipy's
    ``Rotation.from_quat`` has ~100 us of per-call object/validation overhead
    which dominates for the (1, 4) case used every control step on the robot.
    Falls back to scipy for larger batches where its vectorized code wins.
    """
    q = np.asarray(quat_wxyz)
    if q.ndim == 2 and q.shape[0] <= 4:
        w = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]
        # Assume already normalized; if not, the rotation matrix is wrong by
        # a uniform scale factor.  Caller (IMU quaternion) guarantees unit norm.
        xx = x * x; yy = y * y; zz = z * z
        xy = x * y; xz = x * z; yz = y * z
        wx = w * x; wy = w * y; wz = w * z
        n = q.shape[0]
        out = np.empty((n, 3, 3), dtype=q.dtype if q.dtype.kind == "f" else np.float32)
        out[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
        out[:, 0, 1] = 2.0 * (xy - wz)
        out[:, 0, 2] = 2.0 * (xz + wy)
        out[:, 1, 0] = 2.0 * (xy + wz)
        out[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
        out[:, 1, 2] = 2.0 * (yz - wx)
        out[:, 2, 0] = 2.0 * (xz - wy)
        out[:, 2, 1] = 2.0 * (yz + wx)
        out[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
        return out
    r = R.from_quat(q, scalar_first=True)
    return r.as_matrix()


def mat2quat(rot_mats: np.ndarray) -> np.ndarray:
    """
    Convert batch of rotation matrices to quaternions (w, x, y, z).
    Args:
        rot_mats: (N, 3, 3) array of rotation matrices
    Returns:
        quats: (N, 4) array of quaternions (w, x, y, z)
    """
    r = R.from_matrix(rot_mats)
    return r.as_quat(scalar_first=True)


def quat2euler(
    quat_wxyz: np.ndarray, order: str = "xyz", degrees: bool = False
) -> np.ndarray:
    """
    Convert batch of quaternions (w, x, y, z) to Euler angles.
    Args:
        quat_wxyz: (N, 4) array of quaternions (w, x, y, z)
        order: Euler sequence, e.g. 'xyz', 'zyx'
        degrees: If True, return angles in degrees
    Returns:
        eulers: (N, 3) Euler angles
    """
    r = R.from_quat(quat_wxyz, scalar_first=True)
    return r.as_euler(order, degrees=degrees)


def base2navi(base2world: np.ndarray) -> np.ndarray:
    # Copy + project onto the xy-plane (matches the jax counterpart in
    # utils.transforms_jax.base2navi); avoids aliasing mutations on the
    # caller's rotation matrix via numpy views.
    x_proj = base2world[:, 0].astype(np.float64, copy=True)
    x_proj[2] = 0.0
    x_proj /= np.linalg.norm(x_proj)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_proj)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def batch_base2navi(base2world: np.ndarray) -> np.ndarray:
    x_proj = base2world[..., :3, 0]
    x_proj = x_proj / np.linalg.norm(x_proj, axis=-1, keepdims=True)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_proj)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=-1, keepdims=True)
    x_axis = np.cross(y_axis, z_axis)
    return np.stack((x_axis, y_axis, np.broadcast_to(z_axis, x_axis.shape)), axis=-1)


def vec7d_to_pose(vec):
    """converts a 7d vector (pos, quat) to a 4x4 transformation matrix."""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = vec[0:3]
    pose[:3, :3] = R.from_quat(np.roll(vec[3:7], -1)).as_matrix()
    return pose


def pose_to_vec7d(pose):
    """converts a 4x4 transformation matrix to a 7d vector (pos, quat)."""
    vec = np.zeros(7, dtype=np.float32)
    vec[0:3] = pose[:3, 3]
    vec[3:7] = np.roll(R.from_matrix(pose[:3, :3]).as_quat(), 1)
    return vec


def flip_pose_around_x_axis(pose):
    """flips a transformation matrix around the x-z plane."""
    flipped_pose = np.eye(4, dtype=np.float32)
    # flip translation
    flipped_pose[:3, 3] = pose[:3, 3] * np.array([1, -1, 1], dtype=np.float32)
    # flip rotation
    rotvec = R.from_matrix(pose[:3, :3]).as_rotvec()
    flipped_rotvec = rotvec * np.array([-1, 1, -1], dtype=np.float32)
    flipped_pose[:3, :3] = R.from_rotvec(flipped_rotvec).as_matrix()
    return flipped_pose


def qpos_flip_around_x(qpos, dof_pos_flip_mapping):
    """
    This function adapt from @YunLiu
    flips a motion sequence (qpos) from left to right.
    qpos: (n_frame, n_dof) motion data.
    dof_pos_flip_mapping: mapping for flipping joint dofs.
    returns: (n_frame, n_dof) flipped motion data.
    """
    flipped_qpos = np.zeros_like(qpos, dtype=np.float32)

    # flip joint dofs using the provided mapping
    indices = 7 + dof_pos_flip_mapping[:, 0].astype(int)
    signs = dof_pos_flip_mapping[:, 1].astype(np.float32)
    flipped_qpos[:, 7:] = qpos[:, indices] * signs

    # flip root pose sequentially
    n_frame = qpos.shape[0]
    if n_frame > 0:
        flipped_init_pose = flip_pose_around_x_axis(vec7d_to_pose(qpos[0, 0:7]))
        flipped_qpos[0, 0:7] = pose_to_vec7d(flipped_init_pose)

        for i in range(1, n_frame):
            ori_last_pose = vec7d_to_pose(qpos[i - 1, 0:7])
            ori_curr_pose = vec7d_to_pose(qpos[i, 0:7])
            flipped_last_pose = vec7d_to_pose(flipped_qpos[i - 1, 0:7])

            delta_pose = np.linalg.inv(ori_last_pose) @ ori_curr_pose
            flipped_delta_pose = flip_pose_around_x_axis(delta_pose)

            flipped_qpos[i, 0:7] = pose_to_vec7d(flipped_last_pose @ flipped_delta_pose)

    return flipped_qpos
