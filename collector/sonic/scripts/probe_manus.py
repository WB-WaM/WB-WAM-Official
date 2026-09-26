#!/usr/bin/env python3
"""Detect PC-local MANUS USB receivers and changing left/right glove data.

No robot connection, command publisher, vibration, or calibration writes.
Use --usb-only with system Python before installing the teleop environment.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
USB_ROOT = Path("/sys/bus/usb/devices")
# The repository's 70-manus-hid.rules identifies MANUS devices by this VID.
MANUS_VENDOR = "3325"


def discover_usb(root=USB_ROOT):
    devices = []
    for path in sorted(root.iterdir()):
        vendor = path / "idVendor"
        if not vendor.exists() or vendor.read_text().strip().lower() != MANUS_VENDOR:
            continue
        record = {"port": path.name}
        for key in ("idProduct", "product", "serial"):
            field = path / key
            record[key] = field.read_text().strip() if field.exists() else "unknown"
        devices.append(record)
    return devices


def read_side(state, side):
    values = state.get(f"{side}_glove_id", [])
    if len(values) != 1 or not math.isfinite(float(values[0])):
        return None
    glove_id = int(values[0])
    if glove_id <= 0 or glove_id != values[0]:
        return None
    positions = tuple(float(value) for value in state.get(f"{glove_id}_position", []))
    if len(positions) < 63 or len(positions) % 3 or not all(map(math.isfinite, positions)):
        return None
    return glove_id, positions


def probe_stream(sdk, duration, clock=time.monotonic, sleep=time.sleep):
    """Require stable, distinct IDs and recent changes on BOTH gloves.

    The current binding has no per-frame timestamp. Repeated reads of its
    cached dictionary are not evidence of live data; ask the user to move.
    """
    ids, previous, changes, last_change = {}, {}, {}, {}
    try:
        if int(sdk.init(10.0)) != 0:
            raise RuntimeError("MANUS initialization failed/timed out; check USB permissions, pairing and SDK license.")
        deadline = clock() + duration
        while clock() < deadline:
            now = clock()
            state = sdk.get_latest_state()
            present = set()
            for side in ("left", "right"):
                reading = read_side(state, side)
                if reading is None:
                    continue
                glove_id, positions = reading
                present.add(side)
                if side in ids and ids[side] != glove_id:
                    raise RuntimeError(f"{side} glove ID changed; pair only the intended gloves and retry.")
                ids[side] = glove_id
                old = previous.get(side)
                if old is not None and len(old) == len(positions) and any(abs(a - b) > 1e-6 for a, b in zip(old, positions)):
                    changes[side] = changes.get(side, 0) + 1
                    last_change[side] = now
                previous[side] = positions
            if len(ids) == 2 and ids["left"] == ids["right"]:
                raise RuntimeError("SDK reported the same glove ID for both sides.")
            other_ids = {key.removesuffix("_position") for key in state if key.endswith("_position")}
            if other_ids - {str(value) for value in ids.values()} and len(other_ids) > 2:
                raise RuntimeError("More than two glove streams detected; select one operator's pair and retry.")
            if present == {"left", "right"} and all(
                changes.get(side, 0) >= 2 and now - last_change[side] <= 0.5
                for side in ("left", "right")
            ):
                return ids
            sleep(0.05)
        raise RuntimeError("No changing data from both gloves within the time limit. "
                           "Power/pair both gloves and gently move BOTH hands during the probe; "
                           "cached or one-sided data does not pass.")
    finally:
        sdk.shutdown()


def load_sdk():
    built = REPO_ROOT / "tracker/sonic/.deps/manus_sdk"
    source = REPO_ROOT / "third_party/MANUS_SDK/ManusSDK_v3.1.1/SDKClient_Linux"
    sys.path.insert(0, str(built if built.is_dir() else source))
    return importlib.import_module("ManusServer")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usb-only", action="store_true", help="List MANUS USB candidates only; does not validate glove streams.")
    parser.add_argument("--duration-s", type=float, default=12.0, help="Stream observation limit after SDK init (1–60 seconds).")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_s) or not 1 <= args.duration_s <= 60:
        parser.error("--duration-s must be between 1 and 60")
    try:
        devices = discover_usb()
        if not devices:
            raise RuntimeError("No MANUS USB receiver/device (VID 3325) found on this PC. "
                               "Plug the wireless receiver into the PC, not the robot, and retry.")
        for device in devices:
            print(f"MANUS USB candidate: {device}", flush=True)
        if args.usb_only:
            print("USB detected; glove pairing, left/right data and calibration remain unverified.")
            return 0
        print("Stop other MANUS consumers. Gently move both gloves now; this probe sends no robot commands.", flush=True)
        sdk = load_sdk()
        ids = probe_stream(sdk, args.duration_s)
        for side, glove_id in ids.items():
            print(f"{side}: glove_id=0x{glove_id:08X}, changing skeleton data received")
        print("PASS: both glove streams detected. Use HAND_CONTROL_MODE=manus; verify per-user calibration separately.")
        return 0
    except KeyboardInterrupt:
        print("MANUS probe interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"MANUS probe failed: {exc}", file=sys.stderr)
        print("For missing ManusServer, run scripts/env/setup_envs.sh teleop and use .venv_teleop/bin/python.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
