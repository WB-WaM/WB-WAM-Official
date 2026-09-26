#!/usr/bin/env python3
"""Exercise real C++/SDK serial packets against a PTY, with no robot hardware."""
import os
import pty
import select
import struct
import subprocess
import threading
import time
import unittest

BINARY = os.environ.get('WBWAM_HEAD_SERVO_TEST_BINARY', '')


def crc(data):
    value = 0
    for byte in data:
        value ^= byte << 8
        for _ in range(8):
            value = ((value << 1) ^ (0x8005 if value & 0x8000 else 0)) & 0xffff
    return value


class Servos:
    def __init__(self, modes=(3, 3), fail_enable=None, models=(1060, 1240), positions=(2000, 2100)):
        self.master, self.slave = pty.openpty()
        self.path = os.ttyname(self.slave)
        self.modes = modes
        self.fail_enable = fail_enable
        self.models = models
        self.writes = []
        self.torque = [0, 0]
        self.positions = list(positions)
        self.gains = [{80: 0, 82: 0, 84: 800} for _ in range(2)]
        self.running = True
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        data = b''
        while self.running:
            if not select.select([self.master], [], [], .05)[0]:
                continue
            try:
                data += os.read(self.master, 4096)
            except OSError:
                continue
            while len(data) >= 7:
                count = int.from_bytes(data[5:7], 'little') + 7
                if len(data) < count:
                    break
                msg, data = data[:count], data[count:]
                assert msg[:4] == b'\xff\xff\xfd\x00'
                assert crc(msg[:-2]) == int.from_bytes(msg[-2:], 'little')
                ident, instruction, params = msg[4], msg[7], msg[8:-2]
                error, answer = 0, b''
                if instruction == 1:
                    answer = struct.pack('<HB', self.models[ident], 1)
                elif instruction == 2:
                    addr, length = struct.unpack('<HH', params)
                    value = {11: self.modes[ident], 48: 4095, 52: 0, 64: self.torque[ident], 70: 0, 132: self.positions[ident], **self.gains[ident]}[addr]
                    answer = value.to_bytes(length, 'little')
                elif instruction == 3:
                    addr = int.from_bytes(params[:2], 'little')
                    value = int.from_bytes(params[2:], 'little')
                    self.writes.append((ident, addr, value))
                    if addr in (80, 82, 84):
                        assert len(params) == 4, 'Gain registers require 2-byte writes'
                        self.gains[ident][addr] = value
                    if addr == 116 and self.torque[ident]:
                        self.positions[ident] = value
                    if addr == 64:
                        self.torque[ident] = value
                        if ident == self.fail_enable and value == 1:
                            error = 4
                reply = b'\xff\xff\xfd\x00' + bytes([ident]) + struct.pack('<H', len(answer) + 4) + bytes([0x55, error]) + answer
                os.write(self.master, reply + struct.pack('<H', crc(reply)))

    def close(self):
        self.running = False
        self.thread.join(1)
        os.close(self.master); os.close(self.slave)


@unittest.skipUnless(BINARY, 'set WBWAM_HEAD_SERVO_TEST_BINARY to the compiled executable')
class HeadServoSerialTests(unittest.TestCase):
    def run_case(self, args, **kwargs):
        servos = Servos(**kwargs)
        try:
            result = subprocess.run([BINARY, '--device', servos.path, *args], text=True, capture_output=True, timeout=5)
            return result, list(servos.writes), list(servos.torque)
        finally:
            servos.close()

    def test_probe_never_writes_controls(self):
        result, writes, _ = self.run_case(['--probe', '--duration', '.1'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(writes, [])

    def test_hold_sets_current_goal_before_enable_and_disables_both(self):
        result, writes, torque = self.run_case(['--hold-current', '--motion-approved', '--duration', '.1'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(writes, [(0, 116, 2000), (0, 64, 1), (1, 116, 2100), (1, 64, 1), (0, 64, 0), (1, 64, 0)])
        self.assertEqual(torque, [0, 0])

    def test_second_axis_wrong_mode_prevents_all_motion(self):
        result, writes, _ = self.run_case(['--hold-current', '--motion-approved', '--duration', '.1'], modes=(3, 4))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(writes, [])

    def test_default_target_uses_calibration_and_joint_directions(self):
        result, writes, torque = self.run_case(['--hold-target', '--motion-approved', '--duration', '.1',
                                               '--servo0-calibration', '1990', '--servo1-calibration', '2320'])
        self.assertEqual(result.returncode, 0, result.stderr)
        for ident, initial, expected in ((0, 2000, 1990), (1, 2100, 1979)):
            goals = [value for axis, addr, value in writes if axis == ident and addr == 116]
            self.assertEqual(goals[0], initial)
            self.assertEqual(goals[-1], expected)
            self.assertTrue(all(abs(b-a) <= 3 for a, b in zip(goals, goals[1:])))
        self.assertIn('target_deg=-50', result.stdout)
        self.assertIn('target_deg=10', result.stdout)
        self.assertEqual(torque, [0, 0])

    def test_missing_calibration_prevents_all_writes(self):
        result, writes, _ = self.run_case(['--hold-target', '--motion-approved'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('calibration', result.stderr)
        self.assertEqual(writes, [])

    def test_raw_pose_needs_no_angle_calibration(self):
        result, writes, torque = self.run_case(['--hold-raw', '--motion-approved', '--duration', '.1',
                                               '--joint0-encoder', '1990', '--joint1-encoder', '2090'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([v for i,a,v in writes if i == 0 and a == 116][-1], 1990)
        self.assertEqual([v for i,a,v in writes if i == 1 and a == 116][-1], 2090)
        self.assertEqual(torque, [0, 0])

    def test_default_raw_pose_needs_no_configuration_or_calibration(self):
        result, writes, torque = self.run_case(
            ['--hold-raw', '--motion-approved', '--duration', '.1'], positions=(3000, 1870))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([v for i, a, v in writes if i == 0 and a == 116][-1], 3027)
        self.assertEqual([v for i, a, v in writes if i == 1 and a == 116][-1], 1849)
        self.assertIn('READY: both head servos holding pose', result.stdout)
        self.assertEqual(torque, [0, 0])

    def test_teach_locks_moved_pose_without_torque_off_and_restores_gains(self):
        servos = Servos()
        proc = subprocess.Popen([BINARY, '--device', servos.path, '--teach', '--motion-approved'],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 3
            while servos.torque != [1, 1] and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(servos.torque, [1, 1])
            self.assertEqual([g[84] for g in servos.gains], [0, 0])
            # Hand moves well outside the old 32-count hold tolerance.
            servos.positions[:] = [2200, 2350]
            proc.stdin.write('hold\n'); proc.stdin.flush()
            deadline = time.monotonic() + 3
            while any(g[84] == 0 for g in servos.gains) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual([g[84] for g in servos.gains], [800, 800])
            self.assertEqual(servos.torque, [1, 1])
            self.assertFalse(any(a == 64 and v == 0 for i,a,v in servos.writes))
            self.assertIn((0, 116, 2200), servos.writes)
            self.assertIn((1, 116, 2350), servos.writes)
            proc.terminate()
            out, err = proc.communicate(timeout=3)
            self.assertEqual(proc.returncode, 130, err)
            self.assertIn('CAPTURED: joint0_encoder=2200 joint1_encoder=2350', out)
            self.assertEqual(servos.torque, [0, 0])
            self.assertEqual([g[80] for g in servos.gains], [0, 0])
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()
            servos.close()

    def test_teach_input_close_disables_and_restores_gains(self):
        servos = Servos()
        try:
            result = subprocess.run([BINARY, '--device', servos.path, '--teach', '--motion-approved'],
                                    input='', capture_output=True, text=True, timeout=3)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(servos.torque, [0, 0])
            self.assertEqual([g[84] for g in servos.gains], [800, 800])
            self.assertEqual([g[80] for g in servos.gains], [0, 0])
        finally:
            servos.close()

    def test_joint_limit_violation_prevents_all_writes(self):
        result, writes, _ = self.run_case(['--hold-target', '--motion-approved', '--joint0-deg', '-51',
                                         '--servo0-calibration', '1990', '--servo1-calibration', '2320'])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(writes, [])

    def test_encoder_wraparound_prevents_all_writes(self):
        result, writes, _ = self.run_case(['--hold-target', '--motion-approved',
                                         '--servo0-calibration', '1990', '--servo1-calibration', '0'])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(writes, [])

    def test_partial_enable_fault_cleans_up_both_axes(self):
        result, writes, torque = self.run_case(['--hold-current', '--motion-approved', '--duration', '.1'], fail_enable=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(torque, [0, 0])
        self.assertEqual(writes[-2:], [(0, 64, 0), (1, 64, 0)])

    def test_sigterm_disables_both_axes(self):
        servos = Servos()
        try:
            proc = subprocess.Popen([BINARY, '--device', servos.path, '--hold-current', '--motion-approved'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 3
            while servos.torque != [1, 1] and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(servos.torque, [1, 1])
            proc.terminate()
            out, err = proc.communicate(timeout=3)
            self.assertEqual(proc.returncode, 130, err)
            self.assertEqual(servos.torque, [0, 0])
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()
            servos.close()


if __name__ == '__main__':
    unittest.main()
