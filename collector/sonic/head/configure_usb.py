#!/usr/bin/env python3
"""Root-only installer: restrict the brltty workaround to the CH340 USB match."""
from pathlib import Path
import os
import shutil
import subprocess


def exclude_ch340(text):
    lines = text.splitlines(keepends=True)
    return ''.join('# WB-WAM CH340 head adapter: ' + line if
                   'ENV{PRODUCT}=="1a86/7523/*"' in line and not line.lstrip().startswith('#')
                   else line for line in lines)


def main():
    if os.geteuid() != 0:
        raise SystemExit('Run this installer as root')
    dest = Path('/etc/udev/rules.d/85-brltty.rules')
    candidates = [dest, Path('/usr/lib/udev/rules.d/85-brltty.rules'), Path('/lib/udev/rules.d/85-brltty.rules')]
    source = next((p for p in candidates if p.is_file()), None)
    if source:
        original = source.read_text()
        updated = exclude_ch340(original)
        if updated != original:
            if dest.exists() and not dest.with_suffix('.rules.wbwam-backup').exists():
                shutil.copy2(dest, dest.with_suffix('.rules.wbwam-backup'))
            dest.write_text(updated)
            dest.chmod(0o644)
            print('Disabled brltty auto-claim only for CH340 1a86:7523')
    rule = Path('/etc/udev/rules.d/99-wbwam-head-serial.rules')
    rule.write_text('SUBSYSTEM=="usb", ATTR{idVendor}=="1a86", ATTR{idProduct}=="7523", ENV{BRLTTY_BRAILLE_DRIVER}="", ENV{BRLTTY_BRAILLE_DEVICE}="", ENV{BRLTTY_PID_FILE}="", ENV{SYSTEMD_WANTS}=""\n'
                    'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", GROUP="dialout", MODE="0660"\n')
    rule.chmod(0o644)
    # An already running udev-launched brltty keeps its old device selection.
    # Stop that instance only when this CH340 is marked as its braille device.
    for device in Path('/sys/bus/usb/devices').glob('*'):
        try:
            if (device / 'idVendor').read_text().strip() != '1a86' or (device / 'idProduct').read_text().strip() != '7523':
                continue
        except FileNotFoundError:
            continue
        props = subprocess.check_output(['udevadm', 'info', '-q', 'property', '-p', str(device)], text=True)
        if 'BRLTTY_BRAILLE_DEVICE=usb:vendor=0X1a86+product=0X7523' in props:
            subprocess.run(['systemctl', 'stop', 'brltty-udev.service'], check=True)
            print('Stopped the stale brltty-udev instance claiming this CH340; no service disabled or uninstalled')


if __name__ == '__main__':
    main()
