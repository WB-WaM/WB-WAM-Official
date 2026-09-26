from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import fields
import io
from pathlib import Path
import socket
import sys
import unittest

import numpy as np
import zmq


SONIC_ROOT = Path(__file__).resolve().parents[2]
if str(SONIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SONIC_ROOT))

from gear_sonic.scripts.run_sim_loop import _configure_replay_support  # noqa: E402
from gear_sonic.utils.mujoco_sim.base_sim import (  # noqa: E402
    ReplaySupportServer,
    validate_replay_support_endpoint,
)
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig  # noqa: E402
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand  # noqa: E402


class _FakeBand:
    def __init__(self):
        self.enable = True


class _FakeSimEnv:
    def __init__(self):
        self.elastic_band = _FakeBand()
        self.reset_count = 0

    def set_replay_support_enabled(self, enabled: bool) -> None:
        self.elastic_band.enable = bool(enabled)

    def reset_and_hold_replay_support(self) -> None:
        self.reset_count += 1
        self.elastic_band.enable = True

    def replay_support_status(self) -> dict[str, object]:
        return {
            "support_available": True,
            "support_active": self.elastic_band.enable,
            "support_force_norm": 42.0 if self.elastic_band.enable else 0.0,
            "sim_time_s": 1.25,
            "base_height_m": 0.8,
            "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }


def _request(
    operation: str,
    request_id: str,
    *,
    run_id: str = "run-a",
    generation: int | None = None,
) -> dict[str, object]:
    request: dict[str, object] = {
        "schema_version": 1,
        "operation": operation,
        "run_id": run_id,
        "request_id": request_id,
    }
    if generation is not None:
        request["generation"] = generation
    return request


def _free_loopback_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


class ReplaySupportControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = _FakeSimEnv()
        self.server = ReplaySupportServer.__new__(ReplaySupportServer)
        self.server._initialize_protocol_state(self.environment)

    def test_mutations_are_owned_generation_checked_and_idempotent(self):
        status = self.server.handle_request(_request("status", "status-0"))
        self.assertTrue(status["ok"])
        self.assertTrue(status["support_active"])
        self.assertEqual(status["generation"], 0)

        reset_request = _request("reset_hold", "reset-0", generation=0)
        reset = self.server.handle_request(reset_request)
        self.assertTrue(reset["ok"])
        self.assertEqual(reset["generation"], 1)
        self.assertEqual(reset["owner_run_id"], "run-a")
        self.assertEqual(self.environment.reset_count, 1)

        duplicate = self.server.handle_request(reset_request)
        self.assertEqual(duplicate, reset)
        self.assertEqual(self.environment.reset_count, 1)

        wrong_owner = self.server.handle_request(
            _request("release", "release-wrong", run_id="run-b", generation=1)
        )
        self.assertFalse(wrong_owner["ok"])
        self.assertEqual(wrong_owner["error_code"], "owner_mismatch")
        self.assertTrue(self.environment.elastic_band.enable)
        self.assertEqual(wrong_owner["generation"], 1)

        release_request = _request("release", "release-0", generation=1)
        release = self.server.handle_request(release_request)
        self.assertTrue(release["ok"])
        self.assertFalse(release["support_active"])
        self.assertEqual(release["generation"], 2)

        duplicate_release = self.server.handle_request(release_request)
        self.assertEqual(duplicate_release, release)
        self.assertEqual(duplicate_release["generation"], 2)

        stale = self.server.handle_request(_request("hold", "hold-stale", generation=1))
        self.assertFalse(stale["ok"])
        self.assertEqual(stale["error_code"], "generation_mismatch")
        self.assertFalse(self.environment.elastic_band.enable)

        takeover = self.server.handle_request(
            _request("hold", "hold-b", run_id="run-b", generation=2)
        )
        self.assertTrue(takeover["ok"])
        self.assertEqual(takeover["owner_run_id"], "run-b")
        self.assertEqual(takeover["generation"], 3)
        self.assertTrue(self.environment.elastic_band.enable)

    def test_reusing_request_id_for_different_content_is_rejected(self):
        first = self.server.handle_request(_request("status", "same-id"))
        self.assertTrue(first["ok"])

        reused = self.server.handle_request(
            _request("hold", "same-id", generation=0)
        )
        self.assertFalse(reused["ok"])
        self.assertEqual(reused["error_code"], "request_id_reused")
        self.assertEqual(reused["generation"], 0)

    def test_real_rep_socket_is_polled_without_blocking(self):
        endpoint = _free_loopback_endpoint()
        context = zmq.Context()
        server = ReplaySupportServer(endpoint, _FakeSimEnv(), context=context)
        client = context.socket(zmq.REQ)
        client.setsockopt(zmq.LINGER, 0)
        client.connect(endpoint)
        try:
            self.assertFalse(server.poll_once())
            client.send_json(_request("status", "wire-status"))
            poller = zmq.Poller()
            poller.register(client, zmq.POLLIN)
            for _ in range(100):
                server.poll_once()
                if client in dict(poller.poll(1)):
                    break
            else:
                self.fail("REP server did not answer a local status request")
            response = client.recv_json()
            self.assertTrue(response["ok"])
            self.assertTrue(response["support_active"])
            self.assertEqual(response["request_id"], "wire-status")
        finally:
            client.close(0)
            server.close()
            context.term()

    def test_only_local_control_endpoints_are_accepted(self):
        self.assertEqual(
            validate_replay_support_endpoint("tcp://127.0.0.1:5561"),
            "tcp://127.0.0.1:5561",
        )
        self.assertEqual(
            validate_replay_support_endpoint("ipc:///tmp/sonic-replay-support"),
            "ipc:///tmp/sonic-replay-support",
        )
        for endpoint in (
            "",
            "tcp://*:5561",
            "tcp://0.0.0.0:5561",
            "tcp://192.168.1.2:5561",
            "udp://127.0.0.1:5561",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_replay_support_endpoint(endpoint)

    def test_default_sim_config_does_not_enable_replay_support(self):
        endpoint_field = next(
            field for field in fields(SimLoopConfig) if field.name == "replay_support_endpoint"
        )
        self.assertIsNone(endpoint_field.default)

        original = {"ENABLE_ELASTIC_BAND": False}
        unchanged = dict(original)
        _configure_replay_support(unchanged, None)
        self.assertEqual(unchanged, original)

        configured = dict(original)
        _configure_replay_support(configured, "tcp://127.0.0.1:5561")
        self.assertTrue(configured["ENABLE_ELASTIC_BAND"])
        self.assertEqual(
            configured["REPLAY_SUPPORT_ENDPOINT"], "tcp://127.0.0.1:5561"
        )

    def test_managed_band_ignores_manual_toggle_but_legacy_band_does_not(self):
        legacy = ElasticBand()
        legacy.handle_keyboard_button("9")
        self.assertFalse(legacy.enable)

        managed = ElasticBand(managed=True)
        with redirect_stdout(io.StringIO()):
            managed.handle_keyboard_button("9")
        self.assertTrue(managed.enable)


if __name__ == "__main__":
    unittest.main()
