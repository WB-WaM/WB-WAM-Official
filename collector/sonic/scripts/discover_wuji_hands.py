#!/usr/bin/env python3
"""Discover Wuji USB serials, or identify two connected hands by thumb motion.

Standalone: copy this file to the robot and run it with system Python 3.
Default mode only reads sysfs. --identify requires numpy, wujihandpy, and
--motion-approved; each thumb repeats slowly until its side is confirmed on stdin.
Enter 'confirm 1 left' (or right), then independently confirm hand 2.
--resolve prints the two confirmed serial assignments without moving.
Add --write-env PATH to save those assignments into a robot-local launcher env.
--tune-thumbs moves both thumbs together without changing the saved mapping.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


USB_ROOT = Path("/sys/bus/usb/devices")
# Current WujihandPy USB ID and the legacy ID in the bundled WujihandCpp README.
# https://docs.wuji.tech/docs/en/wujihandpy/latest/tutorial/
WUJI_USB_IDS = {("0483", "2000"), ("0483", "7530")}
THUMB_JOINT = (0, 1)  # Thumb joint 1: second joint counted from the palm.
THUMB_AMPLITUDE_DEGREES = 30.0  # Selected by the operator during real-hardware tuning.
THUMB_AMPLITUDE = math.radians(THUMB_AMPLITUDE_DEGREES)
MAX_SPEED = 0.3  # radians/second for commanded reset and thumb trajectories
THUMB_PERIOD = 4.2
THUMB_HOLD_SECONDS = 0.3  # 1.8s each way plus 0.3s at each end.
STAGE_TIMEOUT = 300.0  # Stop without a result if the conversation is abandoned.
# Conservative limits shared with wuji_hand_server.py (5 fingers x 4 joints).
LOWER = [0.0475, -0.1387, -0.4642, -0.4699,
         -0.1585, -0.3700, -0.4777, -0.4683,
         -0.1644, -0.3700, -0.4739, -0.4684,
         -0.1554, -0.3700, -0.4765, -0.4777,
         -0.1626, -0.3700, -0.4768, -0.4683]
UPPER = [1.6033, 0.9324, 1.5623, 1.5568,
         1.5604, 0.3700, 1.5485, 1.5753,
         1.5516, 0.3700, 1.5512, 1.5745,
         1.5585, 0.3700, 1.5487, 1.5634,
         1.5585, 0.3700, 1.5490, 1.5735]


def discover(usb_root: Path) -> list[tuple[str, str, str]]:
    """Return (USB port, VID:PID, USB serial) for known hand device IDs."""
    devices = []
    for device in sorted(usb_root.iterdir()):
        if not (device / "idVendor").exists():
            continue  # USB interfaces do not have device descriptors.
        vendor = (device / "idVendor").read_text().strip().lower()
        product = (device / "idProduct").read_text().strip().lower()
        if (vendor, product) not in WUJI_USB_IDS:
            continue
        serial_path = device / "serial"
        serial = serial_path.read_text().strip() if serial_path.exists() else ""
        devices.append((device.name, f"{vendor}:{product}", serial))
    return devices


def pair_serials(devices):
    serials = [device[2] for device in devices]
    if len(serials) != 2 or not all(serials) or len(set(serials)) != 2:
        raise ValueError("Identification requires exactly two hands with distinct, readable USB serials.")
    return serials


def stdin_commands():
    """Poll complete lines without blocking motion monitoring or buffering ahead."""
    fd = sys.stdin.fileno()
    buffer = b""

    def poll():
        nonlocal buffer
        if not select.select([fd], [], [], 0)[0]:
            return []
        chunk = os.read(fd, 4096)
        if not chunk:
            raise RuntimeError("Control input closed; stopping identification without a result.")
        buffer += chunk
        if len(buffer) > 4096:
            raise ValueError("Control input too long; stopping identification.")
        lines = buffer.split(b"\n")
        buffer = lines.pop()
        return [line.decode("utf-8").strip().lower() for line in lines]

    return poll


def run_motion(devices, sdk, np, sleep=time.sleep, *, poll=None,
               clock=time.monotonic, stage_timeout=STAGE_TIMEOUT,
               simultaneous=False, amplitude=THUMB_AMPLITUDE, period=THUMB_PERIOD,
               thumb_joint=THUMB_JOINT):
    """Repeat each thumb until independently confirmed; disable before returning.

    SDK position/limit APIs use radians. No realtime controller, calibration,
    error reset, or effort-limit changes are used.
    https://docs.wuji.tech/docs/en/wujihandpy/latest/api-reference/
    """
    pair_serials(devices)
    if thumb_joint not in ((0, 0), (0, 1), (0, 2), (0, 3)):
        raise ValueError("Select a thumb joint index from 0 (root) to 3.")
    if not math.isfinite(stage_timeout) or not 0 < stage_timeout <= 600:
        raise ValueError("Stage timeout must be positive and at most 600 seconds.")
    leg_seconds = period / 2 - THUMB_HOLD_SECONDS
    if (not math.isfinite(amplitude) or amplitude <= 0 or not math.isfinite(period)
            or leg_seconds <= 0 or amplitude / leg_seconds > MAX_SPEED + 1e-9):
        raise ValueError("Invalid amplitude/period or requested speed exceeds 0.3 rad/s; use a longer period.")
    if poll is None:
        poll = stdin_commands()
    activated = []
    confirmed = []
    stage = 0
    pending = None
    deadline = None

    def control():
        nonlocal pending
        if deadline is not None and clock() >= deadline:
            raise RuntimeError("Thumb tuning timed out; stopping." if simultaneous else
                               f"Hand {stage} confirmation timed out; no completed mapping produced.")
        for line in poll():
            if line in ("stop", "quit"):
                raise KeyboardInterrupt("Operator stopped identification")
            if not line:
                continue
            if simultaneous:
                print("TUNING: type 'stop' before restarting with different amplitude/period parameters.", flush=True)
                continue
            parts = line.split()
            if (stage not in (1, 2) or len(parts) != 3 or parts[0] != "confirm"
                    or parts[1] != str(stage) or parts[2] not in ("left", "right")):
                print(f"INPUT REJECTED: waiting for stage {stage}; use 'confirm {stage} left/right' "
                      "only after observing that hand, or 'stop'.", flush=True)
                continue
            if parts[2] in confirmed:
                print("INPUT REJECTED: both hands cannot have the same side. Observe again, "
                      "or type 'stop' to restart identification.", flush=True)
                continue
            if pending is None:
                pending = parts[2]
                print(f"HAND {stage}: confirmation received; finishing this slow cycle before stopping.", flush=True)

    def pause(seconds):
        until = clock() + seconds
        while clock() < until:
            control()
            for enabled_hand in activated:
                check_errors(enabled_hand)
            remaining = until - clock()
            if remaining > 0:
                sleep(min(0.05, remaining))

    def positions(value):
        result = np.asarray(value, dtype=float)
        if result.shape != (5, 4) or not np.all(np.isfinite(result)):
            raise ValueError("Expected finite 5x4 joint data from the hand SDK.")
        return result.copy()

    def check_errors(hand):
        errors = np.asarray(hand.read_joint_error_code())
        if errors.shape != (5, 4) or not np.all(errors == 0):
            raise RuntimeError("Hand reports joint errors; aborting identification.")

    def wait_for(hand, target):
        for _ in range(20):
            check_errors(hand)
            actual = positions(hand.read_joint_actual_position())
            if np.max(np.abs(actual - target)) <= 0.04:
                return
            pause(0.1)
        raise RuntimeError("Hand did not reach the commanded pose; aborting identification.")

    def ramp_group(trajectories, minimum_seconds):
        # SDK calls consume substantial wall time over USB. Interpolate by the
        # actual clock rather than adding a sleep after each nominal sample.
        duration = max(minimum_seconds, max(float(np.max(np.abs(target - start))) / MAX_SPEED
                                           for _, start, target in trajectories))
        started = clock()
        while True:
            control()
            for hand, _, _ in trajectories:
                check_errors(hand)
            elapsed = clock() - started
            fraction = 1.0 if elapsed >= duration - 1e-9 else elapsed / duration
            for hand, start, target in trajectories:
                hand.write_joint_target_position(start + (target - start) * fraction)
            if fraction >= 1.0:
                break
            pause(min(0.05, max(0.0, started + duration - clock())))
        for hand, _, target in trajectories:
            wait_for(hand, target)

    def ramp(hand, start, target, minimum_seconds=0.6):
        ramp_group([(hand, start, target)], minimum_seconds)

    try:
        control()  # Reject a closed control channel before enabling any motor.
        prepared = []
        # Validate BOTH devices before enabling either one.
        for _, usb_id, serial in devices:
            vendor, product = (int(part, 16) for part in usb_id.split(":"))
            hand = sdk.Hand(serial_number=serial, usb_vid=vendor, usb_pid=product)
            check_errors(hand)
            actual = positions(hand.read_joint_actual_position())
            firmware_low = positions(hand.read_joint_lower_limit())
            firmware_high = positions(hand.read_joint_upper_limit())
            low = np.maximum(firmware_low, np.array(LOWER).reshape(5, 4)) + 0.02
            high = np.minimum(firmware_high, np.array(UPPER).reshape(5, 4)) - 0.02
            if np.any(low >= high) or np.any(actual < firmware_low - 0.05) or np.any(actual > firmware_high + 0.05):
                raise ValueError("Invalid joint limits or current pose; aborting identification.")
            neutral = np.clip(np.zeros((5, 4)), low, high)
            flexed = neutral.copy()
            flexed[thumb_joint] += amplitude
            if flexed[thumb_joint] > high[thumb_joint]:
                raise ValueError("Insufficient thumb travel for the identification motion.")
            prepared.append((hand, neutral, flexed))

        print("RESET: both hands return to a neutral open pose; ignore movement during reset.", flush=True)
        for hand, neutral, _ in prepared:
            control()
            actual = positions(hand.read_joint_actual_position())
            hand.write_joint_target_position(actual)  # Avoid an old target on enable.
            activated.append(hand)  # Include even a partially failed enable in cleanup.
            hand.write_joint_enabled(True)
            ramp(hand, actual, neutral)
        print("RESET DONE. " + ("Both thumbs will move together; type 'stop' to finish." if simultaneous else
                                "Each thumb repeats until separately confirmed; type 'stop' to abort."), flush=True)
        pause(2)
        if simultaneous:
            deadline = clock() + stage_timeout
            print(f"BOTH THUMBS ACTIVE: joint={thumb_joint[1]} (0=root), {math.degrees(amplitude):g} degrees, "
                  f"approximately {period:g}s per cycle; timeout {stage_timeout:g}s. "
                  "Calibration mapping is unchanged.", flush=True)
            cycle = 0
            while True:
                started = clock()
                extended = []
                for outward in (True, False):
                    trajectories = [(hand, neutral, flexed) if outward else (hand, flexed, neutral)
                                    for hand, neutral, flexed in prepared]
                    ramp_group(trajectories, leg_seconds)
                    measured = [float(positions(hand.read_joint_actual_position())[thumb_joint])
                                for hand, _, _ in prepared]
                    if outward:
                        extended = measured
                    pause(max(0.0, started + (period / 2 if outward else period) - clock()))
                cycle += 1
                travel = [math.degrees(high - low) for high, low in zip(extended, measured)]
                print(f"CYCLE {cycle}: measured travel hand1={travel[0]:.1f}deg "
                      f"hand2={travel[1]:.1f}deg; elapsed={clock() - started:.2f}s", flush=True)
        for index, (hand, neutral, flexed) in enumerate(prepared, start=1):
            stage = index
            pending = None
            deadline = clock() + stage_timeout
            print(f"HAND {index}/2 AWAITING CONFIRMATION: thumb joint {thumb_joint[1]} repeating "
                  f"{math.degrees(amplitude):g} degrees every approximately {period:g}s. "
                  f"Observe the robot's own side, then enter 'confirm {index} left' or "
                  f"'confirm {index} right'. Timeout: {stage_timeout:g}s.", flush=True)
            # At least one complete, feedback-checked cycle for each observed hand.
            while True:
                started = clock()
                ramp(hand, neutral, flexed, leg_seconds)
                pause(max(0.0, started + period / 2 - clock()))
                ramp(hand, flexed, neutral, leg_seconds)
                pause(max(0.0, started + period - clock()))
                if pending is not None:
                    break
            hand.write_joint_enabled(False)
            confirmed.append(pending)
            print(f"HAND {index}/2 CONFIRMED {pending}; returned to neutral and disabled.", flush=True)
            stage = 0
            deadline = None
        return confirmed
    finally:
        failures = []
        for hand in activated:
            try:
                hand.write_joint_enabled(False)
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("Could not disable every hand; stop the hardware. " + "; ".join(failures))


def identify(devices, result_path, stage_timeout=STAGE_TIMEOUT):
    serials = pair_serials(devices)
    import numpy as np
    import wujihandpy

    # Reserve a NEW private result before motion, so failed runs cannot reuse old order.
    fd = os.open(result_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    complete = False
    previous = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Identification interrupted by signal {signum}")

    try:
        with os.fdopen(fd, "w") as output:
            for sig in (signal.SIGTERM, signal.SIGHUP):
                previous[sig] = signal.signal(sig, interrupted)
            sides = run_motion(devices, wujihandpy, np, stage_timeout=stage_timeout)
            if not isinstance(sides, list) or len(sides) != 2 or set(sides) != {"left", "right"}:
                raise ValueError("Both hands must be independently confirmed with different sides.")
            json.dump({"version": 2, "status": "sides_confirmed", "serials_in_order": serials,
                       "sides_in_order": sides}, output)
            output.write("\n")
        complete = True
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if not complete:
            result_path.unlink(missing_ok=True)
    print("Identification complete: both sides confirmed and both hands disabled. Use --resolve to read the mapping.", flush=True)


def write_serial_config(config_path, mapping):
    """Atomically update just the two serial fields, without running the config."""
    config_path = config_path.expanduser().resolve()
    source = config_path if config_path.exists() else config_path.with_name(config_path.name + ".example")
    if not source.is_file():
        raise ValueError("No local env file or adjacent .env.example template; copy the robot service templates first.")
    original = source.read_text()
    lines = original.splitlines(keepends=True)
    for side in ("left", "right"):
        key = f"{side.upper()}_WUJI_SERIAL"
        pattern = re.compile(r'^\s*(?:(?:export\s+)?' + key + r'=.*|:\s*"\$\{' + key + r':=.*\}")\s*$')
        matches = []
        for index, line in enumerate(lines):
            if line.lstrip().startswith("#"):
                continue
            if pattern.fullmatch(line.rstrip("\r\n")):
                matches.append(index)
            elif re.search(r'\b' + key + r'\b', line):
                raise ValueError(f"Unsupported configuration syntax for {key}; existing file was not changed.")
        if len(matches) > 1:
            raise ValueError(f"Multiple definitions for {key}; existing file was not changed.")
        assignment = key + "=" + shlex.quote(mapping[side]) + "\n"
        if matches:
            lines[matches[0]] = assignment
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(assignment)
    updated = "".join(lines)
    fd, temporary = tempfile.mkstemp(prefix=".wuji-serials-", dir=config_path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(updated)
            output.flush()
            os.fsync(output.fileno())
        syntax = subprocess.run(["bash", "-n", temporary], capture_output=True, text=True)
        if syntax.returncode:
            raise ValueError("Launcher env shell syntax check failed; existing file was not changed.")
        mode = (config_path.stat().st_mode & 0o777) if config_path.exists() else 0o600
        os.chmod(temporary, mode)
        if not config_path.exists() or updated != original:
            # Refuse to overwrite a concurrent edit made after the initial read.
            if source == config_path and config_path.read_text() != original:
                raise ValueError("Launcher env changed during identification; retry saving the result.")
            if source != config_path and config_path.exists():
                raise ValueError("Launcher env was created concurrently; retry saving the result.")
            os.replace(temporary, config_path)
        if config_path.read_text() != updated:
            raise ValueError("Launcher env readback differs; configuration verification failed.")
    finally:
        Path(temporary).unlink(missing_ok=True)
    print("Serial configuration saved and read back successfully; unrelated settings preserved.", flush=True)


def resolve(devices, result_path, config_path=None):
    serials = pair_serials(devices)
    receipt = json.loads(result_path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError("Invalid motion result; repeat identification.")
    order = receipt.get("serials_in_order")
    sides = receipt.get("sides_in_order")
    if (receipt.get("version") != 2 or receipt.get("status") != "sides_confirmed"
            or not isinstance(order, list) or len(order) != 2
            or not all(isinstance(item, str) for item in order) or set(order) != set(serials)
            or not isinstance(sides, list) or len(sides) != 2
            or not all(isinstance(item, str) for item in sides) or set(sides) != {"left", "right"}):
        raise ValueError("Result does not match two independently confirmed, connected hands; repeat identification.")
    mapping = dict(zip(sides, order))
    if config_path is not None:
        write_serial_config(config_path, mapping)
        return
    left, right = mapping["left"], mapping["right"]
    print(f"LEFT_WUJI_SERIAL={shlex.quote(left)}")
    print(f"RIGHT_WUJI_SERIAL={shlex.quote(right)}")


def tune_thumbs(devices, amplitude_degrees, period, timeout, thumb_joint=THUMB_JOINT[1]):
    import numpy as np
    import wujihandpy

    previous = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Tuning interrupted by signal {signum}")

    try:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, interrupted)
        run_motion(devices, wujihandpy, np, simultaneous=True,
                   amplitude=math.radians(amplitude_degrees), period=period, stage_timeout=timeout,
                   thumb_joint=(0, thumb_joint))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv: list[str] | None = None, *, usb_root: Path = USB_ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--identify", action="store_true", help="Reset hands; repeat each thumb until confirmed via interactive stdin.")
    mode.add_argument("--resolve", type=Path, metavar="RESULT", help="Resolve a completed motion result; no motor commands.")
    mode.add_argument("--tune-thumbs", action="store_true", help="Move both thumbs together; no calibration result or config changes.")
    parser.add_argument("--motion-approved", action="store_true", help="Operator has approved this reset/thumb motion and is watching.")
    parser.add_argument("--result", type=Path, help="New private JSON path for the motion order; never overwritten.")
    parser.add_argument("--write-env", type=Path, metavar="PATH",
                        help="With --resolve, save verified serials into local env; preserve other settings.")
    parser.add_argument("--stage-timeout", type=float, default=STAGE_TIMEOUT,
                        help="Seconds allowed per confirmation (default 300, maximum 600); expiry aborts, never advances.")
    parser.add_argument("--amplitude-deg", type=float, default=THUMB_AMPLITUDE_DEGREES,
                        help="Thumb tuning travel in degrees (default 30).")
    parser.add_argument("--period", type=float, default=THUMB_PERIOD,
                        help="Thumb tuning cycle in seconds (default 4.2).")
    parser.add_argument("--thumb-joint", type=int, choices=(0, 1, 2, 3), default=THUMB_JOINT[1],
                        help="Tuning joint: 0 is nearest the palm; default 1 is the second joint from the palm.")
    args = parser.parse_args(argv)
    if args.write_env and not args.resolve:
        parser.error("--write-env requires --resolve after both hands have been confirmed")
    if args.identify and (not args.motion_approved or args.result is None):
        parser.error("--identify requires --motion-approved and --result NEW_PATH")
    if args.tune_thumbs and not args.motion_approved:
        parser.error("--tune-thumbs requires --motion-approved")
    if args.result and not args.identify:
        parser.error("--result requires --identify")
    if not (args.identify or args.tune_thumbs) and (args.motion_approved or args.stage_timeout != STAGE_TIMEOUT):
        parser.error("Motion options require --identify or --tune-thumbs")
    if not args.tune_thumbs and (args.amplitude_deg != THUMB_AMPLITUDE_DEGREES
                               or args.period != THUMB_PERIOD or args.thumb_joint != THUMB_JOINT[1]):
        parser.error("--amplitude-deg, --period and --thumb-joint require --tune-thumbs")
    if not math.isfinite(args.stage_timeout) or not 0 < args.stage_timeout <= 600:
        parser.error("--stage-timeout must be positive and at most 600 seconds")
    try:
        devices = discover(usb_root)
    except OSError as exc:
        print(f"Cannot read Linux USB descriptors: {exc}. Run on the robot USB host; "
              "check sysfs access and retry after USB connections are stable.", file=sys.stderr)
        return 1
    if not devices:
        print("No Wuji USB hands found (0483:2000 / 0483:7530). Run this script on the "
              "robot where the hands are plugged in, not the Ethernet-connected PC. "
              "Check hand power, USB cables, and lsusb for other firmware IDs.", file=sys.stderr)
        return 1
    if args.identify or args.resolve or args.tune_thumbs:
        try:
            if args.identify:
                identify(devices, args.result, args.stage_timeout)
            elif args.resolve:
                resolve(devices, args.resolve, args.write_env)
            else:
                tune_thumbs(devices, args.amplitude_deg, args.period, args.stage_timeout, args.thumb_joint)
            return 0
        except KeyboardInterrupt:
            print("Thumb tuning stopped; motors disabled, calibration mapping unchanged." if args.tune_thumbs else
                  "Identification interrupted; no completed mapping produced.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"{'Thumb tuning' if args.tune_thumbs else 'Identification'} failed: {exc}", file=sys.stderr)
            return 1
    for port, usb_id, serial in devices:
        print(f"usb_port={port} usb_id={usb_id} usb_serial={shlex.quote(serial) if serial else '<missing>'}")
    if any(not serial for _, _, serial in devices):
        print("A matching device has no readable USB serial. Check USB descriptors and "
              "permissions; do not substitute a product serial or USB port number.", file=sys.stderr)
        return 1
    print("Keep both hands connected. Use --identify with operator approval to reset and move "
          "each thumb until independently confirmed, then --resolve the saved result. Listing alone does not "
          "identify left/right.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
