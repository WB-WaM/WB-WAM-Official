#!/usr/bin/env python3
"""Check PICO tracking through the PC XRoboToolkit service, without publishing commands."""

from __future__ import annotations

import argparse
import math
import sys
import time


def valid_pose(pose):
    return (len(pose) == 7 and all(math.isfinite(float(value)) for value in pose)
            and 0.5 < sum(float(value) ** 2 for value in pose[3:]) < 1.5)


def observe(xrt, duration, clock=time.monotonic, sleep=time.sleep):
    last = None
    updates = 0
    try:
        xrt.init()
        deadline = clock() + duration
        while clock() < deadline:
            body = xrt.get_body_joints_pose()
            poses = (xrt.get_headset_pose(), xrt.get_left_controller_pose(), xrt.get_right_controller_pose())
            stamps = (int(xrt.get_time_stamp_ns()), int(xrt.get_body_timestamp_ns()))
            if (xrt.is_body_data_available() and len(body) == 24
                    and all(valid_pose(pose) for pose in (*poses, *body)) and min(stamps) > 0):
                if last is not None and all(a > b for a, b in zip(stamps, last)):
                    updates += 1
                elif last is not None and any(a < b for a, b in zip(stamps, last)):
                    updates = 0
                last = stamps
                if updates >= 3:
                    return
            else:
                last, updates = None, 0
            sleep(0.05)
        raise RuntimeError("No valid updating head/controller/full-body stream. Check wired IPs, "
                           "PC Service WORKING, Head/Controller/Send, Full body and tracker calibration.")
    finally:
        xrt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=12.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_s) or not 1 <= args.duration_s <= 60:
        parser.error("--duration-s must be between 1 and 60")
    try:
        import xrobotoolkit_sdk as xrt
        observe(xrt, args.duration_s)
        print("PASS: valid head, controllers and full-body poses with advancing timestamps. No robot commands sent.")
        return 0
    except KeyboardInterrupt:
        print("PICO probe interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"PICO probe failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
