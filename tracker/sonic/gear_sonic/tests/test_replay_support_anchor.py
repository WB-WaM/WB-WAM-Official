from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


SONIC_ROOT = Path(__file__).resolve().parents[2]
if str(SONIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SONIC_ROOT))

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv  # noqa: E402
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand  # noqa: E402


class _FakeMjData:
    def __init__(self):
        self.xpos = np.asarray([[0.0, 0.0, 0.7918637524]], dtype=np.float64)
        self.xfrc_applied = np.ones((1, 6), dtype=np.float64)
        self.qpos = np.asarray(
            [0.0, 0.0, 0.7918637524, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.time = 0.0


class ReplaySupportAnchorTest(unittest.TestCase):
    def test_managed_anchor_has_zero_initial_wrench_at_grounded_pose(self):
        band = ElasticBand(managed=True)
        grounded_position = np.asarray([0.0, 0.0, 0.7918637524])
        band.set_anchor_position(grounded_position)
        pose = np.concatenate(
            (
                grounded_position,
                np.asarray([1.0, 0.0, 0.0, 0.0]),
                np.zeros(6),
            )
        )

        np.testing.assert_allclose(band.Advance(pose), np.zeros(6), atol=1e-12)
        np.testing.assert_allclose(band.point, grounded_position)
        self.assertEqual(band.length, 0)

    def test_legacy_band_keeps_original_one_meter_lifting_target(self):
        band = ElasticBand()
        pose = np.asarray(
            [
                0.0,
                0.0,
                0.7918637524,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        wrench = band.Advance(pose)
        self.assertAlmostEqual(wrench[2], 10000.0 * (1.0 - pose[2]))
        np.testing.assert_array_equal(band.point, [0, 0, 1])

    def test_reset_hold_recaptures_current_link_position_and_clears_old_force(self):
        environment = DefaultEnv.__new__(DefaultEnv)
        environment.elastic_band = ElasticBand(managed=True)
        environment.band_attached_link = 0
        environment.mj_model = object()
        environment.mj_data = _FakeMjData()

        def reset():
            environment.mj_data.xpos[0] = [0.03, -0.02, 0.805]
            environment.mj_data.qpos[:3] = [0.03, -0.02, 0.805]
            environment.mj_data.xfrc_applied[0] = 99.0

        environment.reset = reset
        with patch("gear_sonic.utils.mujoco_sim.base_sim.mujoco.mj_forward"):
            environment.reset_and_hold_replay_support()

        np.testing.assert_allclose(
            environment.elastic_band.point,
            [0.03, -0.02, 0.805],
        )
        np.testing.assert_array_equal(
            environment.mj_data.xfrc_applied[0], np.zeros(6)
        )
        self.assertTrue(environment.elastic_band.enable)
        status = environment.replay_support_status()
        np.testing.assert_allclose(
            status["support_anchor_position"], [0.03, -0.02, 0.805]
        )
        self.assertAlmostEqual(status["support_position_error_norm"], 0.0)

    def test_anchor_rejects_invalid_positions(self):
        band = ElasticBand(managed=True)
        for value in ([0.0, 1.0], [0.0, 0.0, np.nan]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                band.set_anchor_position(value)


if __name__ == "__main__":
    unittest.main()
