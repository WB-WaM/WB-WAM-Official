from pathlib import Path
import runpy
import tempfile
import unittest


script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/probe_manus.py"))


class FakeSDK:
    def __init__(self, changing=True, same_id=False, init_error=False):
        self.changing = changing
        self.same_id = same_id
        self.init_error = init_error
        self.frame = 0
        self.closed = False

    def init(self, timeout):
        return 1 if self.init_error else 0

    def shutdown(self):
        self.closed = True

    def get_latest_state(self):
        self.frame += int(self.changing)
        return {"left_glove_id": [11], "right_glove_id": [11 if self.same_id else 22],
                "11_position": [self.frame * 0.001] * 75,
                "22_position": [self.frame * 0.002] * 75}


class ProbeTest(unittest.TestCase):
    def probe(self, sdk):
        now = [0.0]
        def sleep(duration):
            now[0] += duration
        return script["probe_stream"](sdk, 1.0, clock=lambda: now[0], sleep=sleep)

    def test_usb_filters_vendor_and_skips_interfaces(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for port, vid in (("1-1", "3325"), ("1-2", "0483")):
                device = root / port
                device.mkdir()
                (device / "idVendor").write_text(vid)
            (root / "1-1:1.0").mkdir()
            devices = script["discover_usb"](root)
            self.assertEqual(len(devices), 1)
            self.assertEqual(devices[0]["port"], "1-1")
            self.assertEqual(devices[0]["serial"], "unknown")

    def test_changing_pair_passes_and_closes(self):
        sdk = FakeSDK()
        self.assertEqual(self.probe(sdk), {"left": 11, "right": 22})
        self.assertTrue(sdk.closed)

    def test_cached_duplicate_and_init_failure_do_not_pass(self):
        for sdk in (FakeSDK(changing=False), FakeSDK(same_id=True), FakeSDK(init_error=True)):
            with self.assertRaises(RuntimeError):
                self.probe(sdk)
            self.assertTrue(sdk.closed)

    def test_invalid_and_missing_skeletons_are_rejected(self):
        for positions in ([], [0.0] * 4, [float("nan")] * 75):
            self.assertIsNone(script["read_side"]({"left_glove_id": [11], "11_position": positions}, "left"))
        self.assertIsNone(script["read_side"]({}, "right"))

    def test_one_sided_stream_does_not_pass(self):
        sdk = FakeSDK()
        original = sdk.get_latest_state
        def left_only():
            state = original()
            del state["right_glove_id"]
            return state
        sdk.get_latest_state = left_only
        with self.assertRaises(RuntimeError):
            self.probe(sdk)
        self.assertTrue(sdk.closed)


if __name__ == "__main__":
    unittest.main()
