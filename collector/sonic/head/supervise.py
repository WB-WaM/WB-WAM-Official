#!/usr/bin/env python3
"""Run the camera only while the head's bounded-feedback hold service is healthy."""
import argparse
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


class Stopped(Exception):
    pass


def stop_signal(signum, frame):
    raise Stopped()


def terminate(process, stop_signal=signal.SIGTERM):
    if process is None or process.poll() is not None:
        return
    process.send_signal(stop_signal)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print('FAIL: service did not stop; verify head torque on site', file=sys.stderr)
        process.kill()
        process.wait()


def supervise(head_command, camera_command, head_env, timeout=30):
    head = camera = selector = None
    try:
        # Read raw bytes so buffered log lines cannot hide the READY line from select().
        # Only the supervisor receives terminal signals; forward each stop once.
        head = subprocess.Popen(head_command, env=head_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(head.stdout, selectors.EVENT_READ)
        pending = b''
        deadline = time.monotonic() + timeout
        ready = False
        while not ready:
            if head.poll() is not None:
                raise RuntimeError('head service exited before ready')
            if time.monotonic() >= deadline:
                raise RuntimeError('head service readiness timeout')
            for key, _ in selector.select(.1):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    raise RuntimeError('head service output closed before ready')
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    print(line.decode(errors='replace'), flush=True)
                    if line == b'READY: both head servos holding pose':
                        ready = True
        camera = subprocess.Popen(camera_command, start_new_session=True)
        print('Head hold ready; camera started', flush=True)
        while True:
            if head.poll() is not None:
                raise RuntimeError('head service stopped; stopping camera')
            if camera.poll() is not None:
                raise RuntimeError(f'camera stopped (exit {camera.returncode}); stopping head hold')
            for key, _ in selector.select(.1):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if chunk:
                    sys.stdout.write(chunk.decode(errors='replace')); sys.stdout.flush()
    except Stopped:
        return 130
    except (RuntimeError, OSError) as e:
        print(f'FAIL: {e}', file=sys.stderr)
        return 1
    finally:
        # Ignore repeated stop signals while children perform torque-disable cleanup.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        # Camera's Python finally block closes its RealSense pipeline on SIGINT.
        terminate(camera, signal.SIGINT)
        terminate(head)
        if head is not None and head.stdout is not None:
            tail = head.stdout.read()
            if tail:
                sys.stdout.write(tail.decode(errors='replace')); sys.stdout.flush()
            head.stdout.close()
        if selector is not None:
            selector.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--head-launcher', required=True)
    ap.add_argument('--head-config', required=True)
    ap.add_argument('camera', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    camera = args.camera[1:] if args.camera and args.camera[0] == '--' else args.camera
    if not camera:
        ap.error('camera command is required after --')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop_signal)
    head_env = dict(os.environ, LAUNCHER_CONFIG_FILE=str(Path(args.head_config).resolve()))
    return supervise(['bash', args.head_launcher, '--motion-approved'], camera, head_env)


if __name__ == '__main__':
    raise SystemExit(main())
