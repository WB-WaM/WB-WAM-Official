from __future__ import annotations

import numpy as np
import pytest

from deploy.offline_pose import load_offline_pose
from deploy.wuji_hand import (
    PICO_TO_MEDIAPIPE,
    WUJI_LOWER_LIMITS,
    WUJI_UPPER_LIMITS,
    pico_hand_to_mediapipe,
    validate_wuji_qpos,
)
from tracking.constants import DEFAULT_QPOS


def _pico_hand() -> np.ndarray:
    state = np.zeros((26, 7), dtype=np.float32)
    for index in range(26):
        state[index, :3] = [index * 0.005, (index % 5) * 0.01, index * 0.002]
    return state


def test_pico_26_to_mediapipe_21_mapping() -> None:
    state = _pico_hand()
    result = pico_hand_to_mediapipe(state)
    assert result.shape == (21, 3)
    np.testing.assert_allclose(result, state[PICO_TO_MEDIAPIPE, :3])


def test_pico_hand_rejects_bad_shape_and_nan() -> None:
    with pytest.raises(ValueError):
        pico_hand_to_mediapipe(np.zeros((25, 7), dtype=np.float32))
    state = _pico_hand()
    state[3, 0] = np.nan
    with pytest.raises(ValueError):
        pico_hand_to_mediapipe(state)


def test_wuji_qpos_is_clipped_to_server_limits() -> None:
    raw = WUJI_UPPER_LIMITS + 1.0
    result, count = validate_wuji_qpos(raw, name="left")
    assert count == 20
    np.testing.assert_allclose(result, WUJI_UPPER_LIMITS)
    assert np.all(result >= WUJI_LOWER_LIMITS)


def test_offline_20_to_50_hz_and_invalid_gap(tmp_path) -> None:
    frames = 5
    qpos = np.tile(DEFAULT_QPOS, (frames, 1)).astype(np.float32)
    qpos[:, 0] = np.arange(frames) * 0.1
    # End at the same rotation with the opposite quaternion sign.
    qpos[-1, 3:7] *= -1
    hand = np.linspace(0.0, 0.5, frames, dtype=np.float32)[:, None] * np.ones(
        (1, 20), dtype=np.float32
    )
    valid = np.array([True, True, False, True, True])
    path = tmp_path / "motion.npz"
    np.savez(
        path,
        qpos=qpos,
        frequency=np.array(20.0),
        left_wuji_qpos=hand,
        right_wuji_qpos=hand,
        left_wuji_qpos_valid=valid,
        right_wuji_qpos_valid=valid,
    )
    motion = load_offline_pose(path, default_qpos=DEFAULT_QPOS, target_hz=50.0)
    # 0.2 s source -> 11 samples, plus 25 start, 25 end, 25 hold.
    assert len(motion) == 86
    np.testing.assert_allclose(
        np.linalg.norm(motion.qpos[:, 3:7], axis=1), 1.0, atol=1e-5
    )
    assert not motion.left_valid[-1]
    # No interpolation may bridge the invalid third source row.
    assert np.count_nonzero(~motion.left_valid[25:36]) > 0


def test_offline_rejects_hand_length_mismatch(tmp_path) -> None:
    qpos = np.tile(DEFAULT_QPOS, (3, 1)).astype(np.float32)
    path = tmp_path / "bad.npz"
    np.savez(path, qpos=qpos, left_wuji_qpos=np.zeros((2, 20), dtype=np.float32))
    with pytest.raises(ValueError, match="left_wuji_qpos"):
        load_offline_pose(path, default_qpos=DEFAULT_QPOS)
