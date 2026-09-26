import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from deploy.reference_alignment import (
    align_qpos_first_frame,
    align_reference_trajectory_first_frame,
    quat_yaw_wxyz,
)


def _wxyz(rotation: Rotation) -> np.ndarray:
    xyzw = rotation.as_quat()
    return xyzw[[3, 0, 1, 2]]


def test_align_qpos_first_frame_preserves_relative_motion() -> None:
    qpos = np.zeros((3, 36), dtype=np.float32)
    qpos[:, :3] = [[4.0, -2.0, 0.80], [4.0, -1.0, 0.82], [3.0, 0.0, 0.79]]
    rotations = Rotation.from_euler(
        "xyz", [[0.1, -0.2, np.pi / 2], [0.1, -0.2, 2.0], [0.0, 0.1, 2.2]]
    )
    qpos[:, 3:7] = np.stack([_wxyz(rotation) for rotation in rotations])
    qpos[:, 7:] = np.arange(29, dtype=np.float32)
    original = qpos.copy()

    aligned = align_qpos_first_frame(qpos)

    np.testing.assert_allclose(aligned[0, :2], [0.0, 0.0], atol=1e-6)
    assert quat_yaw_wxyz(aligned[0, 3:7]) == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(aligned[1, :2], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(aligned[2, :2], [2.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(aligned[:, 2], original[:, 2])
    np.testing.assert_allclose(aligned[:, 7:], original[:, 7:])
    np.testing.assert_allclose(np.linalg.norm(aligned[:, 3:7], axis=1), 1.0)
    np.testing.assert_array_equal(qpos, original)


def test_align_qpos_first_frame_supports_nonzero_target() -> None:
    qpos = np.zeros((2, 7), dtype=np.float64)
    qpos[:, :3] = [[1.0, 2.0, 0.8], [2.0, 2.0, 0.8]]
    qpos[:, 3:7] = _wxyz(Rotation.from_euler("z", -0.4))

    aligned = align_qpos_first_frame(qpos, target_xy=(3.0, 4.0), target_yaw=0.7)

    np.testing.assert_allclose(aligned[0, :2], [3.0, 4.0], atol=1e-7)
    np.testing.assert_allclose(
        aligned[1, :2] - aligned[0, :2],
        [np.cos(1.1), np.sin(1.1)],
        atol=1e-7,
    )
    assert quat_yaw_wxyz(aligned[0, 3:7]) == pytest.approx(0.7, abs=1e-7)


def test_align_reference_trajectory_to_runtime_yaw() -> None:
    qpos = np.zeros((2, 7), dtype=np.float32)
    qpos[:, :3] = [[0.0, 0.0, 0.8], [1.0, 0.0, 0.8]]
    qpos[:, 3] = 1.0
    qvel = np.zeros((2, 6), dtype=np.float32)
    qvel[:, 0] = 1.0
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    poses[1, 0, 3] = 1.0
    invariant = np.arange(6, dtype=np.float32).reshape(2, 3)
    reference = {
        "qpos": qpos,
        "qvel": qvel,
        "gv2wrd_pose": poses,
        "gv_vel": invariant,
    }

    aligned = align_reference_trajectory_first_frame(
        reference,
        target_xy=(0.0, 0.0),
        target_yaw=np.pi / 2,
    )

    np.testing.assert_allclose(aligned["qpos"][0, :2], 0.0, atol=1e-6)
    np.testing.assert_allclose(aligned["qpos"][1, :2], [0.0, 1.0], atol=1e-6)
    assert quat_yaw_wxyz(aligned["qpos"][0, 3:7]) == pytest.approx(np.pi / 2, abs=1e-6)
    np.testing.assert_allclose(aligned["qvel"][:, :2], [[0.0, 1.0]] * 2, atol=1e-6)
    np.testing.assert_allclose(aligned["gv2wrd_pose"][1, :2, 3], [0.0, 1.0], atol=1e-6)
    np.testing.assert_array_equal(aligned["gv_vel"], invariant)
    np.testing.assert_array_equal(reference["qpos"], qpos)


@pytest.mark.parametrize("bad_quaternion", [[0, 0, 0, 0], [np.nan, 0, 0, 1]])
def test_align_qpos_first_frame_rejects_invalid_quaternion(bad_quaternion) -> None:
    qpos = np.zeros((1, 7), dtype=np.float32)
    qpos[0, 3:7] = bad_quaternion
    with pytest.raises(ValueError):
        align_qpos_first_frame(qpos)
