#!/usr/bin/env python3
"""Find the CH340 head UART on its USB host; never guess ttyUSB0."""
import argparse
from pathlib import Path
import sys
import time


def discover(root=Path('/sys/class/tty'), dev=Path('/dev')):
    result = []
    for tty in sorted(root.glob('ttyUSB*')):
        node = tty.resolve()
        for parent in (node, *node.parents):
            try:
                if (parent / 'idVendor').read_text().strip() == '1a86' and (parent / 'idProduct').read_text().strip() == '7523':
                    actual = dev / tty.name
                    aliases = [p for kind in ('by-id', 'by-path') for p in sorted((dev / 'serial' / kind).glob('*')) if p.resolve() == actual.resolve()]
                    result.append(str(aliases[0] if aliases else actual))
                    break
            except (FileNotFoundError, PermissionError):
                continue
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wait', type=float, default=0)
    ap.add_argument('--device', help='Select one detected port if multiple CH340 adapters exist')
    args = ap.parse_args()
    until = time.monotonic() + max(0, args.wait)
    while True:
        found = discover()
        if found or time.monotonic() >= until:
            break
        time.sleep(.2)
    if args.device:
        found = [p for p in found if Path(p).resolve() == Path(args.device).resolve()]
    if len(found) != 1:
        print(f'FAIL: expected one selected CH340 head UART, found {len(found)}. Check ch341, USB power and brltty; set HEAD_SERIAL_DEVICE for multiple adapters.', file=sys.stderr)
        return 1
    print(found[0])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
