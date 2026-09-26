from pathlib import Path
import runpy
import unittest


observe = runpy.run_path(str(Path(__file__).parents[1] / "scripts/probe_pico.py"))["observe"]


class FakeXRT:
    def __init__(self, stale=False, invalid=False):
        self.stale, self.invalid = stale, invalid
        self.stamp, self.closed = 1, False

    def init(self):
        pass

    def close(self):
        self.closed = True

    def get_headset_pose(self):
        return [0, 0, 0, 0, 0, 0, 0 if self.invalid else 1]

    get_left_controller_pose = get_headset_pose
    get_right_controller_pose = get_headset_pose

    def get_body_joints_pose(self):
        self.stamp += int(not self.stale)
        return [self.get_headset_pose()] * 24

    def get_time_stamp_ns(self):
        return self.stamp

    get_body_timestamp_ns = get_time_stamp_ns

    def is_body_data_available(self):
        return True


class PicoTest(unittest.TestCase):
    def check(self, xrt):
        now = [0.0]
        def sleep(duration):
            now[0] += duration
        observe(xrt, 1.0, clock=lambda: now[0], sleep=sleep)

    def test_updating_data_passes_and_closes(self):
        xrt = FakeXRT()
        self.check(xrt)
        self.assertTrue(xrt.closed)

    def test_stale_and_invalid_data_fail_and_close(self):
        for xrt in (FakeXRT(stale=True), FakeXRT(invalid=True)):
            with self.assertRaises(RuntimeError):
                self.check(xrt)
            self.assertTrue(xrt.closed)


if __name__ == "__main__":
    unittest.main()
