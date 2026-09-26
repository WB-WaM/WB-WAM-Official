from __future__ import annotations

from pathlib import Path
import sys
import unittest


SONIC_ROOT = Path(__file__).resolve().parents[2]
if str(SONIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SONIC_ROOT))

from gear_sonic.utils.mujoco_sim.base_sim import ReplaySupportServer  # noqa: E402


class _FailingSimEnv:
    elastic_band = object()

    def replay_support_status(self):
        return {
            "support_available": True,
            "support_active": True,
            "support_force_norm": 0.0,
            "sim_time_s": 0.0,
            "base_height_m": 0.8,
            "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }

    def reset_and_hold_replay_support(self):
        raise RuntimeError("synthetic reset failure")

    def set_replay_support_enabled(self, enabled):
        raise RuntimeError("synthetic support failure")


class ReplaySupportFailureTest(unittest.TestCase):
    def test_mutation_failure_is_replied_cached_and_does_not_advance_generation(self):
        server = ReplaySupportServer.__new__(ReplaySupportServer)
        server._initialize_protocol_state(_FailingSimEnv())
        request = {
            "schema_version": 1,
            "operation": "reset_hold",
            "run_id": "run-a",
            "request_id": "reset-fails",
            "generation": 0,
        }

        response = server.handle_request(request)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error_code"], "operation_failed")
        self.assertIn("synthetic reset failure", response["error_message"])
        self.assertEqual(response["generation"], 0)
        self.assertIsNone(response["owner_run_id"])

        duplicate = server.handle_request(request)
        self.assertEqual(duplicate, response)
        status = server.handle_request(
            {
                "schema_version": 1,
                "operation": "status",
                "run_id": "run-a",
                "request_id": "status-after-failure",
            }
        )
        self.assertTrue(status["ok"])
        self.assertEqual(status["generation"], 0)


if __name__ == "__main__":
    unittest.main()
