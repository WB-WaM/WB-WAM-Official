#!/usr/bin/env python3
"""Change a Wuji Glove network address through the official Wuji SDK."""

from __future__ import annotations

import argparse
import ipaddress
import sys
import time
from typing import Any


def _ip(value: str) -> str:
    try:
        return str(ipaddress.ip_address(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_manager() -> Any:
    try:
        import wuji_sdk  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import wuji_sdk. Install the official Wuji SDK, or place it under "
            "third_party/wuji_sdk."
        ) from exc

    for name in ("SdkManager", "Manager"):
        manager_cls = getattr(wuji_sdk, name, None)
        if manager_cls is None:
            continue
        if hasattr(manager_cls, "instance"):
            return manager_cls.instance()
        return manager_cls()

    raise RuntimeError("wuji_sdk does not expose SdkManager or Manager")


def _param_value(device: Any, name: str) -> Any:
    param = getattr(device, name)()
    return param.get()


def _set_param(device: Any, name: str, value: Any) -> None:
    param = getattr(device, name)()
    param.set(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set one Wuji Glove IP address. Connect only one glove with the current IP."
    )
    parser.add_argument("--current-ip", type=_ip, default="192.168.1.100")
    parser.add_argument("--current-port", type=int, default=50000)
    parser.add_argument("--target-ip", type=_ip, required=True)
    parser.add_argument("--target-port", type=int, default=None)
    parser.add_argument("--device-name", default="glove")
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    address = f"{args.current_ip}:{args.current_port}"

    print("[INFO] Loading Wuji SDK")
    manager = _load_manager()

    print(f"[INFO] Connecting to Wuji glove at {address}")
    glove = manager.connect(address=address, device_name=args.device_name)

    try:
        old_ip = _param_value(glove, "ip")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Connected, but failed to read glove.ip(): {exc}") from exc

    try:
        old_port = _param_value(glove, "port")
    except Exception as exc:  # noqa: BLE001
        old_port = f"<read failed: {exc}>"

    print(f"[INFO] Current glove IP:   {old_ip}")
    print(f"[INFO] Current data port: {old_port}")
    print(f"[INFO] Target glove IP:    {args.target_ip}")
    if args.target_port is not None:
        print(f"[INFO] Target data port:  {args.target_port}")

    if args.dry_run:
        print("[DRY-RUN] No device parameters changed")
        return 0

    if str(old_ip) != args.target_ip:
        print(f"[INFO] Setting glove IP to {args.target_ip}")
        _set_param(glove, "ip", args.target_ip)
    else:
        print("[INFO] IP already matches target")

    if args.target_port is not None:
        print(f"[INFO] Setting data port to {args.target_port}")
        _set_param(glove, "port", args.target_port)

    if args.settle_s > 0:
        time.sleep(args.settle_s)

    try:
        new_ip = _param_value(glove, "ip")
        print(f"[OK] Device reports IP: {new_ip}")
    except Exception as exc:  # noqa: BLE001
        print(f"[OK] Parameter write completed; reconnect may be required ({exc})")

    print("[NEXT] Power-cycle or reconnect the glove, then ping the target IP.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
