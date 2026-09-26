import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HEAD = Path(__file__).resolve().parents[1] / 'head'


def load(name):
    spec = importlib.util.spec_from_file_location(name, HEAD / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HeadSetupTests(unittest.TestCase):
    def test_brltty_excludes_only_ch340_and_is_idempotent(self):
        module = load('configure_usb')
        original = 'ENV{PRODUCT}=="1a86/7523/*", ENV{BRLTTY_BRAILLE_DRIVER}="bm"\nENV{PRODUCT}=="403/de58/*", ENV{BRLTTY_BRAILLE_DRIVER}="hd"\n'
        result = module.exclude_ch340(original)
        self.assertTrue(result.startswith('# WB-WAM CH340'))
        self.assertIn('\nENV{PRODUCT}=="403/de58/*"', result)
        self.assertEqual(result, module.exclude_ch340(result))

    def test_discovery_uses_usb_identity_and_stable_path(self):
        module = load('discover_serial')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sysfs, dev = root / 'sys', root / 'dev'
            sysfs.mkdir(); dev.mkdir()
            for index, vendor in ((4, '1a86'), (0, '0403')):
                usb = root / f'usb{index}'
                usb.mkdir()
                (usb / 'idVendor').write_text(vendor)
                (usb / 'idProduct').write_text('7523')
                tty = usb / f'ttyUSB{index}'; tty.mkdir()
                (sysfs / tty.name).symlink_to(tty)
                (dev / tty.name).touch()
            aliases = dev / 'serial' / 'by-path'; aliases.mkdir(parents=True)
            stable = aliases / 'usb-head'
            stable.symlink_to(dev / 'ttyUSB4')
            self.assertEqual(module.discover(sysfs, dev), [str(stable)])


class HeadSupervisorTests(unittest.TestCase):
    def run_services(self, fail):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / 'service.py'
            fake.write_text('''import os, signal, sys, time
from pathlib import Path
root = Path(os.environ['HEAD_TEST_DIR'])
role = sys.argv[1]
failure = os.environ['HEAD_TEST_FAIL']
def stop(signum, frame):
    (root / (role + '.stopped')).touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
if role == 'head' and failure == 'startup':
    raise SystemExit(1)
(root / (role + '.started')).touch()
if role == 'head':
    print('READY: both head servos holding pose', flush=True)
    if failure == 'head':
        deadline = time.monotonic() + 3
        while not (root / 'camera.started').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        raise SystemExit(1)
elif failure == 'camera':
    raise SystemExit(1)
while True:
    time.sleep(.01)
''')
            launcher = root / 'head.sh'
            launcher.write_text('exec "$HEAD_TEST_PYTHON" "$HEAD_TEST_SCRIPT" head\n')
            env = dict(os.environ, HEAD_TEST_PYTHON=sys.executable, HEAD_TEST_SCRIPT=str(fake),
                       HEAD_TEST_DIR=temp, HEAD_TEST_FAIL=fail)
            proc = subprocess.Popen([sys.executable, str(HEAD / 'supervise.py'),
                                     '--head-launcher', str(launcher), '--head-config', str(root / 'head.env'),
                                     '--', sys.executable, str(fake), 'camera'],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                if fail == 'signal':
                    deadline = time.monotonic() + 3
                    while not (root / 'camera.started').exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue((root / 'camera.started').exists())
                    proc.terminate()
                out, _ = proc.communicate(timeout=8)
                markers = {p.name for p in root.glob('*.started')} | {p.name for p in root.glob('*.stopped')}
                return proc.returncode, out, markers
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.communicate(timeout=8)

    def test_head_startup_failure_prevents_camera_start(self):
        code, out, markers = self.run_services('startup')
        self.assertEqual(code, 1, out)
        self.assertNotIn('camera.started', markers)

    def test_head_failure_stops_camera(self):
        code, out, markers = self.run_services('head')
        self.assertEqual(code, 1, out)
        self.assertIn('camera.stopped', markers)

    def test_camera_failure_stops_head(self):
        code, out, markers = self.run_services('camera')
        self.assertEqual(code, 1, out)
        self.assertIn('head.stopped', markers)

    def test_sigterm_stops_both(self):
        code, out, markers = self.run_services('signal')
        self.assertEqual(code, 130, out)
        self.assertTrue({'head.stopped', 'camera.stopped'} <= markers)


if __name__ == '__main__':
    unittest.main()
