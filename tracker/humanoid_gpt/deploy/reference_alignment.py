"""Reference-frame alignment helpers for offline HGPT motions."""

from __future__ import annotations

import numpy as np


def quat_yaw_wxyz(quaternion: np.ndarray) -> float:
    """Return the Z yaw of one normalized ``wxyz`` quaternion."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be finite with shape (4,)")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("quaternion norm must be non-zero")
    w, x, y, z = q / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def align_qpos_first_frame(
    qpos: np.ndarray,
    *,
    target_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
    target_yaw: float = 0.0,
) -> np.ndarray:
    """Rigidly align a qpos trajectory's first root XY/yaw to a target frame.

    The same world-frame SE(2) transform is applied to every root pose. Root
    height, roll/pitch, joint angles, and all motion relative to frame zero are
    preserved. Input root quaternions use MuJoCo's ``wxyz`` convention.
    """
    source = np.asarray(qpos)
    if source.ndim != 2 or source.shape[1] < 7 or len(source) == 0:
        raise ValueError(f"qpos must have shape (T,D>=7) with T>=1, got {source.shape}")
    if not np.all(np.isfinite(source)):
        raise ValueError("qpos must contain only finite values")

    target_xy_array = np.asarray(target_xy, dtype=np.float64)
    if target_xy_array.shape != (2,) or not np.all(np.isfinite(target_xy_array)):
        raise ValueError("target_xy must be finite with shape (2,)")
    if not np.isfinite(target_yaw):
        raise ValueError("target_yaw must be finite")

    quaternion = np.asarray(source[:, 3:7], dtype=np.float64)
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("qpos root quaternions must have non-zero norm")
    quaternion /= norms[:, None]

    yaw_offset = float(target_yaw) - quat_yaw_wxyz(quaternion[0])
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    rotation_2d = np.array([[c, -s], [s, c]], dtype=np.float64)

    output = np.asarray(source, dtype=np.float64).copy()
    relative_xy = np.asarray(source[:, :2], dtype=np.float64) - source[0, :2]
    output[:, :2] = relative_xy @ rotation_2d.T + target_xy_array

    # Pre-multiply q_offset * q_source so the yaw correction is applied in the
    # world frame and therefore does not distort source roll/pitch.
    half = 0.5 * yaw_offset
    ow, oz = np.cos(half), np.sin(half)
    w, x, y, z = quaternion.T
    output[:, 3] = ow * w - oz * z
    output[:, 4] = ow * x - oz * y
    output[:, 5] = ow * y + oz * x
    output[:, 6] = ow * z + oz * w
    output[:, 3:7] /= np.linalg.norm(output[:, 3:7], axis=1, keepdims=True)

    return output.astype(source.dtype, copy=False)


def align_reference_trajectory_first_frame(
    reference: dict[str, np.ndarray],
    *,
    target_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
    target_yaw: float = 0.0,
) -> dict[str, np.ndarray]:
    """Align a converted HGPT reference trajectory to a runtime root frame.

    Gravity-view quantities are invariant under a rigid world-frame SE(2)
    transform. World-frame root qpos/qvel and ``gv2wrd_pose`` are transformed;
    all other arrays are copied unchanged so cached offline motions remain
    reusable across replay sessions.
    """
    if "qpos" not in reference:
        raise ValueError("reference trajectory is missing qpos")
    source_qpos = np.asarray(reference["qpos"])
    source_xy = np.asarray(source_qpos[0, :2], dtype=np.float64)
    source_yaw = quat_yaw_wxyz(source_qpos[0, 3:7])
    yaw_offset = float(target_yaw) - source_yaw
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    rotation_2d = np.array([[c, -s], [s, c]], dtype=np.float64)
    target_xy_array = np.asarray(target_xy, dtype=np.float64)

    output = {key: np.asarray(value).copy() for key, value in reference.items()}
    output["qpos"] = align_qpos_first_frame(
        source_qpos,
        target_xy=target_xy_array,
        target_yaw=target_yaw,
    )

    if "qvel" in output:
        qvel = output["qvel"]
        if qvel.ndim != 2 or qvel.shape[1] < 6 or len(qvel) != len(source_qpos):
            raise ValueError("reference qvel must have shape (T,D>=6)")
        qvel[:, :2] = qvel[:, :2] @ rotation_2d.T
        qvel[:, 3:5] = qvel[:, 3:5] @ rotation_2d.T

    if "gv2wrd_pose" in output:
        poses = output["gv2wrd_pose"]
        if poses.shape != (len(source_qpos), 4, 4):
            raise ValueError("reference gv2wrd_pose must have shape (T,4,4)")
        transform = np.eye(4, dtype=np.float64)
        transform[:2, :2] = rotation_2d
        transform[:2, 3] = target_xy_array - rotation_2d @ source_xy
        output["gv2wrd_pose"] = np.einsum(
            "ij,tjk->tik", transform, poses, optimize=True
        ).astype(poses.dtype, copy=False)

    return output
