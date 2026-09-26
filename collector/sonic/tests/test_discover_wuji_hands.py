"""Exercise USB discovery with fake sysfs trees; no hardware or SDK required."""

import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/discover_wuji_hands.py"))
main = script["main"]


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def device(self, port, vendor="0483", product="2000", serial="000123"):
        path = self.root / port
        path.mkdir()
        for name, value in (("idVendor", vendor), ("idProduct", product), ("serial", serial)):
            if value is not None:
                (path / name).write_text(value + "\n")

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(args), usb_root=self.root)
        return code, out.getvalue(), err.getvalue()

    def test_lists_current_and_legacy_hands_only(self):
        self.device("1-1")
        self.device("1-2", product="7530", serial="000456")
        self.device("1-3", product="5700")  # Same vendor, different device.
        self.device("1-4", vendor="1234")
        (self.root / "1-1:1.0").mkdir()  # An interface, not another hand.
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(len(out.splitlines()), 2)
        self.assertIn("usb_serial=000123", out)
        self.assertIn("usb_serial=000456", out)

    def test_resolves_both_confirmations_using_saved_order(self):
        self.device("1-1")
        self.device("1-2", serial="000456")
        result = self.root / "motion.json"
        receipt = {"version": 2, "status": "sides_confirmed", "serials_in_order": ["000456", "000123"],
                   "sides_in_order": ["left", "right"]}
        result.write_text(json.dumps(receipt))
        self.assertEqual(self.run_cli("--resolve", str(result))[:2],
                         (0, "LEFT_WUJI_SERIAL=000456\nRIGHT_WUJI_SERIAL=000123\n"))
        receipt["sides_in_order"] = ["right", "left"]
        result.write_text(json.dumps(receipt))
        self.assertEqual(self.run_cli("--resolve", str(result))[:2],
                         (0, "LEFT_WUJI_SERIAL=000123\nRIGHT_WUJI_SERIAL=000456\n"))
        (self.root / "1-2" / "serial").write_text("changed")
        code, out, err = self.run_cli("--resolve", str(result))
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("does not match", err)

    def test_old_incomplete_and_duplicate_side_results_are_rejected(self):
        self.device("1-1")
        self.device("1-2", serial="000456")
        result = self.root / "motion.json"
        for sides in (None, ["left"], ["left", "left"], ["left", {}]):
            result.write_text(json.dumps({"version": 2, "status": "sides_confirmed",
                                          "serials_in_order": ["000123", "000456"], "sides_in_order": sides}))
            self.assertEqual(self.run_cli("--resolve", str(result))[0], 1)
        result.write_text('{"version": 1, "status": "motion_complete", "serials_in_order": ["000123", "000456"]}')
        self.assertEqual(self.run_cli("--resolve", str(result))[0], 1)

    def confirmed_receipt(self):
        self.device("1-1")
        self.device("1-2", serial="000456")
        result = self.root / "motion.json"
        result.write_text(json.dumps({"version": 2, "status": "sides_confirmed",
                                      "serials_in_order": ["000123", "000456"],
                                      "sides_in_order": ["right", "left"]}))
        return result

    def test_save_verified_mapping_preserves_other_settings_and_loads_correctly(self):
        result = self.confirmed_receipt()
        config = self.root / "wuji_hand_server.env"
        config.write_text('# local settings\nPC_ZMQ_HOST=example-host\n: "${LEFT_WUJI_SERIAL:=old-left}"\n'
                          'export RIGHT_WUJI_SERIAL=old-right\nCOLLECTOR_PYTHON=/custom/python\n')
        code, out, _ = self.run_cli("--resolve", str(result), "--write-env", str(config))
        self.assertEqual(code, 0)
        self.assertNotIn("000123", out)
        self.assertEqual(config.read_text(), '# local settings\nPC_ZMQ_HOST=example-host\nLEFT_WUJI_SERIAL=000456\n'
                         'RIGHT_WUJI_SERIAL=000123\nCOLLECTOR_PYTHON=/custom/python\n')
        r = subprocess.run(['bash', '-c', 'source "$1"; printf "%s %s" "$LEFT_WUJI_SERIAL" "$RIGHT_WUJI_SERIAL"',
                            'verify', str(config)], capture_output=True, text=True, check=True,
                           env={**os.environ, 'LEFT_WUJI_SERIAL': 'stale-override'})
        self.assertEqual(r.stdout, '000456 000123')

    def test_save_creates_private_env_from_adjacent_template(self):
        result = self.confirmed_receipt()
        config = self.root / "wuji_hand_server.env"
        template = config.with_name(config.name + ".example")
        original = '# template\nPC_ZMQ_HOST=example-host\n'
        template.write_text(original)
        self.assertEqual(self.run_cli("--resolve", str(result), "--write-env", str(config))[0], 0)
        self.assertEqual(template.read_text(), original)
        self.assertEqual(config.stat().st_mode & 0o777, 0o600)
        self.assertIn('LEFT_WUJI_SERIAL=000456', config.read_text())

    def test_invalid_receipt_cannot_update_env(self):
        result = self.confirmed_receipt()
        config = self.root / "wuji_hand_server.env"
        config.write_text('LEFT_WUJI_SERIAL=old\nRIGHT_WUJI_SERIAL=existing\n')
        original = config.read_text()
        result.write_text('{"version":2,"status":"sides_confirmed","sides_in_order":["right"]}')
        self.assertEqual(self.run_cli("--resolve", str(result), "--write-env", str(config))[0], 1)
        self.assertEqual(config.read_text(), original)

    def test_invalid_or_ambiguous_env_is_not_replaced(self):
        result = self.confirmed_receipt()
        config = self.root / "wuji_hand_server.env"
        for original in ('LEFT_WUJI_SERIAL=a\nLEFT_WUJI_SERIAL=b\n',
                         'if true; then\n', 'some_command "$LEFT_WUJI_SERIAL"\n'):
            config.write_text(original)
            self.assertEqual(self.run_cli("--resolve", str(result), "--write-env", str(config))[0], 1)
            self.assertEqual(config.read_text(), original)
            self.assertEqual(list(self.root.glob('.wuji-serials-*')), [])

    def test_save_option_requires_completed_result_resolution(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli("--write-env", str(self.root / "wuji_hand_server.env"))
        self.assertEqual(error.exception.code, 2)

    def test_missing_serial_does_not_emit_assignment(self):
        self.device("1-1", serial=None)
        code, out, err = self.run_cli()
        self.assertEqual(code, 1)
        self.assertNotIn("LEFT_WUJI_SERIAL=", out)
        self.assertIn("no readable USB serial", err)

    def test_empty_or_unavailable_usb_tree(self):
        code, _, err = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("robot where the hands are plugged in", err)
        self.root = self.root / "absent"
        code, _, err = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("Cannot read Linux USB descriptors", err)

    def test_motion_requires_explicit_flag_and_two_distinct_hands(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli("--identify", "--result", str(self.root / "new.json"))
        self.assertEqual(error.exception.code, 2)
        self.device("1-1")
        for add_second in (False, True):
            if add_second:
                self.device("1-2")  # Duplicate serial is invalid.
            code, _, err = self.run_cli("--identify", "--motion-approved", "--result", str(self.root / "new.json"))
            self.assertEqual(code, 1)
            self.assertIn("exactly two", err)
            self.assertFalse((self.root / "new.json").exists())


class FakeSDK:
    def __init__(self):
        self.hands = []
        self.events = []
        self.fail_serial = None
        self.interrupt_serial = None
        self.error_serial = None
        self.disable_failure = None
        self.stuck = False

    def Hand(self, serial_number, **kwargs):
        sdk = self

        class Hand:
            def __init__(self):
                self.serial = serial_number
                self.actual = np.full((5, 4), 0.2)
                self.enabled = False

            def read_joint_error_code(self):
                return np.ones((5, 4)) if sdk.error_serial == self.serial else np.zeros((5, 4))

            def read_joint_actual_position(self):
                return self.actual.copy()

            def read_joint_lower_limit(self):
                return np.array(script["LOWER"]).reshape(5, 4) - 0.02

            def read_joint_upper_limit(self):
                return np.array(script["UPPER"]).reshape(5, 4) + 0.02

            def write_joint_enabled(self, value):
                if not value and sdk.disable_failure == self.serial:
                    raise RuntimeError("disable failed")
                self.enabled = value
                sdk.events.append((self.serial, "enable", value))

            def write_joint_target_position(self, value):
                if self.enabled and sdk.fail_serial == self.serial:
                    raise RuntimeError("USB disconnected")
                if self.enabled and sdk.interrupt_serial == self.serial:
                    raise KeyboardInterrupt()
                sdk.events.append((self.serial, "target", value.copy()))
                if not sdk.stuck:
                    self.actual = value.copy()

        hand = Hand()
        self.hands.append(hand)
        return hand


class MotionTest(unittest.TestCase):
    devices = [("1-1", "0483:2000", "first"), ("1-2", "0483:2000", "second")]

    def run_motion(self, sdk, commands=None, stage_timeout=300):
        self.output = io.StringIO()
        now = [0.0]
        started = {}
        sent = set()
        self.stage_times = {}

        def sleep(seconds):
            now[0] += seconds

        def poll():
            stage = 0
            for index in (1, 2):
                if f"HAND {index}/2 AWAITING CONFIRMATION" in self.output.getvalue():
                    stage = index
            if not stage:
                return []
            started.setdefault(stage, now[0])
            elapsed = now[0] - started[stage]
            if commands is not None:
                return commands(stage, elapsed)
            if elapsed >= 12.1 and stage not in sent:
                sent.add(stage)
                self.stage_times[stage] = now[0]
                return [f"confirm {stage} {'left' if stage == 1 else 'right'}"]
            return []

        with contextlib.redirect_stdout(self.output):
            return script["run_motion"](self.devices, sdk, np, sleep=sleep, poll=poll,
                                        clock=lambda: now[0], stage_timeout=stage_timeout)

    def test_reset_then_thumb_only_in_serial_order_and_disable(self):
        sdk = FakeSDK()
        self.assertEqual(self.run_motion(sdk), ["left", "right"])
        neutral = np.clip(np.zeros((5, 4)), np.array(script["LOWER"]).reshape(5, 4) + 0.02,
                          np.array(script["UPPER"]).reshape(5, 4) - 0.02)
        seen_neutral = set()
        thumb_order = []
        previous = {}
        for serial, kind, value in sdk.events:
            if kind != "target":
                continue
            if serial in previous:
                self.assertLessEqual(float(np.max(np.abs(value - previous[serial]))), 0.015 + 1e-9)
            previous[serial] = value
            if np.allclose(value, neutral):
                seen_neutral.add(serial)
            elif serial in seen_neutral:
                self.assertEqual(seen_neutral, {"first", "second"})
                delta = value - neutral
                self.assertEqual(list(zip(*np.nonzero(np.abs(delta) > 1e-9))), [(0, 1)])
                self.assertLessEqual(delta[0, 1], np.deg2rad(30) + 1e-9)
                if not thumb_order or thumb_order[-1] != serial:
                    thumb_order.append(serial)
        self.assertEqual(thumb_order, ["first", "second"])
        # Multiple full cycles happen while waiting; each leg lasts >= 1.8 seconds.
        for serial in ("first", "second"):
            flexed = neutral.copy()
            flexed[0, 1] += np.deg2rad(30)
            self.assertGreaterEqual(sum(kind == "target" and name == serial and np.allclose(value, flexed)
                                        for name, kind, value in sdk.events), 3)
        self.assertGreaterEqual(self.stage_times[2] - self.stage_times[1], 12.5)
        first_disable = next(
            i for i, (name, kind, value) in enumerate(sdk.events)
            if name == "first" and kind == "enable" and value is False)
        self.assertTrue(all(kind != "target" or name != "first"
                            for name, kind, value in sdk.events[first_disable + 1:]))
        for hand in sdk.hands:
            self.assertFalse(hand.enabled)
            np.testing.assert_allclose(hand.actual, neutral)

    def test_no_confirmation_repeats_first_hand_until_timeout(self):
        sdk = FakeSDK()
        with self.assertRaisesRegex(RuntimeError, "confirmation timed out"):
            self.run_motion(sdk, commands=lambda *_: [], stage_timeout=14)
        self.assertNotIn("HAND 2/2 AWAITING", self.output.getvalue())
        self.assertTrue(all(not hand.enabled for hand in sdk.hands))

    def test_second_hand_requires_new_opposite_confirmation(self):
        sdk = FakeSDK()
        sent = set()

        def commands(stage, elapsed):
            if stage not in sent:
                sent.add(stage)
                # Future-stage input cannot pre-confirm hand 2; same-side input is invalid.
                return ["confirm 1 left", "confirm 2 right"] if stage == 1 else ["confirm 2 left"]
            return []

        with self.assertRaisesRegex(RuntimeError, "Hand 2 confirmation timed out"):
            self.run_motion(sdk, commands=commands, stage_timeout=14)
        self.assertIn("both hands cannot have the same side", self.output.getvalue())
        self.assertNotIn("HAND 2/2 CONFIRMED", self.output.getvalue())
        self.assertTrue(all(not hand.enabled for hand in sdk.hands))

    def test_stop_or_control_disconnect_disables_both_hands(self):
        for disconnect in (False, True):
            sdk = FakeSDK()

            def commands(stage, elapsed):
                if stage == 1:
                    return ["confirm 1 right"]
                if disconnect:
                    raise RuntimeError("Control input closed")
                return ["stop"]

            with self.assertRaises(RuntimeError if disconnect else KeyboardInterrupt):
                self.run_motion(sdk, commands=commands)
            self.assertTrue(all(not hand.enabled for hand in sdk.hands))

    def test_failure_or_interrupt_disables_every_activated_hand(self):
        for attribute, exception in (("fail_serial", RuntimeError), ("interrupt_serial", KeyboardInterrupt)):
            sdk = FakeSDK()
            setattr(sdk, attribute, "second")
            with self.assertRaises(exception):
                self.run_motion(sdk)
            self.assertEqual(len(sdk.hands), 2)
            self.assertTrue(all(not hand.enabled for hand in sdk.hands))

    def test_stalled_feedback_aborts_before_thumb_motion(self):
        sdk = FakeSDK()
        sdk.stuck = True
        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            self.run_motion(sdk)
        self.assertTrue(all(not hand.enabled for hand in sdk.hands))

    def test_preflight_fault_on_second_hand_prevents_all_motion(self):
        sdk = FakeSDK()
        sdk.error_serial = "second"
        with self.assertRaisesRegex(RuntimeError, "joint errors"):
            self.run_motion(sdk)
        self.assertEqual(sdk.events, [])

    def test_disable_failure_still_disables_other_hand_and_reports_failure(self):
        sdk = FakeSDK()
        sdk.disable_failure = "first"
        with self.assertRaisesRegex(RuntimeError, "Could not disable every hand"):
            self.run_motion(sdk)
        self.assertFalse(sdk.hands[1].enabled)

    def test_result_written_only_after_success_and_never_overwritten(self):
        identify = script["identify"]
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict("sys.modules", {"wujihandpy": types.SimpleNamespace()}):
            result = Path(temp) / "motion.json"
            with mock.patch.dict(identify.__globals__, {"run_motion": mock.Mock(side_effect=RuntimeError("failed"))}):
                with self.assertRaises(RuntimeError):
                    identify(self.devices, result)
                self.assertFalse(result.exists())
            motion = mock.Mock(return_value=["right", "left"])
            with mock.patch.dict(identify.__globals__, {"run_motion": motion}), contextlib.redirect_stdout(io.StringIO()):
                identify(self.devices, result)
                self.assertEqual(result.stat().st_mode & 0o777, 0o600)
                self.assertEqual(json.loads(result.read_text())["sides_in_order"], ["right", "left"])
                with self.assertRaises(FileExistsError):
                    identify(self.devices, result)
                motion.assert_called_once()

    def test_missing_confirmation_cannot_produce_result(self):
        identify = script["identify"]
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict("sys.modules", {"wujihandpy": types.SimpleNamespace()}):
            result = Path(temp) / "motion.json"
            for outcome in (["left"], ["left", "left"], None):
                with mock.patch.dict(identify.__globals__, {"run_motion": mock.Mock(return_value=outcome)}):
                    with self.assertRaises(ValueError):
                        identify(self.devices, result)
                    self.assertFalse(result.exists())


class InputTest(unittest.TestCase):
    def test_partial_and_multiple_lines_and_eof(self):
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "r") as input_stream:
            try:
                with mock.patch.object(sys, "stdin", input_stream):
                    poll = script["stdin_commands"]()
                self.assertEqual(poll(), [])
                os.write(write_fd, b"confirm 1 le")
                self.assertEqual(poll(), [])
                os.write(write_fd, b"ft\nconfirm 2 right\nstop\n")
                self.assertEqual(poll(), ["confirm 1 left", "confirm 2 right", "stop"])
            finally:
                os.close(write_fd)
            with self.assertRaisesRegex(RuntimeError, "Control input closed"):
                poll()


class TuningTest(unittest.TestCase):
    def test_selected_root_joint_moves_thirty_degrees(self):
        sdk = FakeSDK()
        output = io.StringIO()
        now = [0.0]
        start_event = []

        def sleep(seconds):
            now[0] += seconds

        def poll():
            if 'BOTH THUMBS ACTIVE' in output.getvalue() and not start_event:
                start_event.append(len(sdk.events))
            return ['stop'] if 'CYCLE 1:' in output.getvalue() else []

        with contextlib.redirect_stdout(output), self.assertRaises(KeyboardInterrupt):
            script['run_motion'](MotionTest.devices, sdk, np, sleep=sleep, poll=poll,
                                 clock=lambda: now[0], simultaneous=True, thumb_joint=(0, 0),
                                 amplitude=np.deg2rad(30), period=4.2)
        neutral = np.clip(np.zeros((5, 4)), np.array(script['LOWER']).reshape(5, 4) + 0.02,
                          np.array(script['UPPER']).reshape(5, 4) - 0.02)
        for _, kind, value in sdk.events[start_event[0]:]:
            if kind == 'target':
                delta = value - neutral
                self.assertLessEqual(abs(delta[0, 0]), np.deg2rad(30) + 1e-9)
                delta[0, 0] = 0
                np.testing.assert_allclose(delta, 0)
        self.assertIn('hand1=30.0deg hand2=30.0deg; elapsed=4.20s', output.getvalue())
        self.assertTrue(all(not hand.enabled for hand in sdk.hands))

    def test_usb_latency_does_not_accumulate_into_cycle_period(self):
        sdk = FakeSDK()
        output = io.StringIO()
        now = [0.0]
        make_hand = sdk.Hand

        def delayed_hand(*args, **kwargs):
            hand = make_hand(*args, **kwargs)
            for name in ('read_joint_error_code', 'read_joint_actual_position', 'write_joint_target_position'):
                original = getattr(hand, name)

                def delayed(*a, _original=original):
                    now[0] += 0.015
                    return _original(*a)

                setattr(hand, name, delayed)
            return hand

        sdk.Hand = delayed_hand

        def sleep(seconds):
            now[0] += seconds

        def poll():
            return ['stop'] if 'CYCLE 2:' in output.getvalue() else []

        with contextlib.redirect_stdout(output), self.assertRaises(KeyboardInterrupt):
            script['run_motion'](MotionTest.devices, sdk, np, sleep=sleep, poll=poll,
                                 clock=lambda: now[0], simultaneous=True)
        cycles = [line for line in output.getvalue().splitlines() if line.startswith('CYCLE ')]
        self.assertEqual(len(cycles), 2)
        for line in cycles:
            duration = float(line.split('elapsed=')[1][:-1])
            self.assertGreaterEqual(duration, 4.19)
            self.assertLess(duration, 4.45)
        self.assertTrue(all(not hand.enabled for hand in sdk.hands))

    def test_both_thumb_targets_are_synchronized_and_stop_disables_motors(self):
        sdk = FakeSDK()
        output = io.StringIO()
        now = [0.0]
        start_event = []

        def sleep(seconds):
            now[0] += seconds

        def poll():
            if 'BOTH THUMBS ACTIVE' in output.getvalue() and not start_event:
                start_event.append(len(sdk.events))
            return ['stop'] if 'CYCLE 2:' in output.getvalue() else []

        with contextlib.redirect_stdout(output), self.assertRaises(KeyboardInterrupt):
            script['run_motion'](MotionTest.devices, sdk, np, sleep=sleep, poll=poll,
                                 clock=lambda: now[0], simultaneous=True)
        targets = [(serial, value) for serial, kind, value in sdk.events[start_event[0]:] if kind == 'target']
        self.assertGreater(len(targets), 100)
        for first, second in zip(targets[::2], targets[1::2]):
            self.assertEqual((first[0], second[0]), ('first', 'second'))
            np.testing.assert_allclose(first[1], second[1])
        self.assertIn('hand1=30.0deg hand2=30.0deg; elapsed=4.20s', output.getvalue())
        self.assertTrue(all(not hand.enabled for hand in sdk.hands))

    def test_excessive_speed_or_invalid_tuning_parameters_prevent_motion(self):
        for amplitude, period in [(np.deg2rad(40), 3), (float('nan'), 3), (-1, 3), (0.1, 0.5)]:
            sdk = FakeSDK()
            with self.assertRaises(ValueError):
                script['run_motion'](MotionTest.devices, sdk, np, poll=lambda: [],
                                     simultaneous=True, amplitude=amplitude, period=period)
            self.assertEqual(sdk.events, [])

    def test_joint_fault_during_tuning_disables_both_hands(self):
        sdk = FakeSDK()
        output = io.StringIO()
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        def poll():
            if 'BOTH THUMBS ACTIVE' in output.getvalue():
                sdk.error_serial = 'second'
            return []

        with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, 'joint errors'):
            script['run_motion'](MotionTest.devices, sdk, np, sleep=sleep,
                                 clock=lambda: now[0], poll=poll, simultaneous=True)
        self.assertTrue(all(not hand.enabled for hand in sdk.hands))


if __name__ == "__main__":
    unittest.main()
