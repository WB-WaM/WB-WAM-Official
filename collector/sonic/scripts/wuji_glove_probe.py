#!/usr/bin/env python3
"""Probe Wuji glove skeleton streams and retargeting output."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for import_root in (
    REPO_ROOT / "tracker" / "sonic",
    REPO_ROOT / "third_party" / "wuji_sdk",
    REPO_ROOT / "third_party" / "wuji_retargeting",
):
    if import_root.is_dir():
        import_root_str = str(import_root)
        if import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)

from gear_sonic.scripts.pico_manager_thread_server import (  # noqa: E402
    DEFAULT_WUJI_GLOVE_LEFT_ADDRESS,
    DEFAULT_WUJI_GLOVE_RIGHT_ADDRESS,
    DEFAULT_WUJI_GLOVE_STALE_S,
    WUJI_GLOVE_LEFT_RETARGETING_CONFIG,
    WUJI_GLOVE_RIGHT_RETARGETING_CONFIG,
    WUJI_QPOS_SIZE,
    WujiGloveSkeletonProvider,
    apply_wuji_qpos_limits,
    compute_wuji_qpos_from_keypoints,
    init_wuji_retargeters,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Wuji Glove skeleton -> qpos path.")
    parser.add_argument("--left-address", default=DEFAULT_WUJI_GLOVE_LEFT_ADDRESS)
    parser.add_argument("--right-address", default=DEFAULT_WUJI_GLOVE_RIGHT_ADDRESS)
    parser.add_argument("--stale-s", type=float, default=DEFAULT_WUJI_GLOVE_STALE_S)
    parser.add_argument("--wait-s", type=float, default=5.0)
    parser.add_argument("--left-retargeting-config", default=str(WUJI_GLOVE_LEFT_RETARGETING_CONFIG))
    parser.add_argument("--right-retargeting-config", default=str(WUJI_GLOVE_RIGHT_RETARGETING_CONFIG))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    left_retargeter, right_retargeter = init_wuji_retargeters(
        args.left_retargeting_config,
        args.right_retargeting_config,
    )
    if left_retargeter is None or right_retargeter is None:
        print("[ERROR] Failed to initialize Wuji retargeters", file=sys.stderr)
        return 2

    provider = WujiGloveSkeletonProvider(
        left_address=args.left_address,
        right_address=args.right_address,
        stale_s=args.stale_s,
    )
    for side in ("left", "right"):
        info = provider.device_infos.get(side, {})
        print(
            f"{side}_address={info.get('address', '<unknown>')} "
            f"sn={info.get('serial_number', '<unknown>')} "
            f"hand_side={info.get('hand_side', '<unknown>')}"
        )
    try:
        deadline = time.monotonic() + args.wait_s
        left_keypoints = np.zeros((21, 3), dtype=np.float32)
        right_keypoints = np.zeros((21, 3), dtype=np.float32)
        left_valid = False
        right_valid = False
        while time.monotonic() < deadline:
            left_keypoints, left_valid, right_keypoints, right_valid = provider.latest_keypoints(
                left_keypoints,
                right_keypoints,
            )
            if left_valid and right_valid:
                break
            time.sleep(0.02)

        print(f"left_skeleton_shape={left_keypoints.shape} valid={left_valid}")
        print(f"right_skeleton_shape={right_keypoints.shape} valid={right_valid}")
        if not left_valid or not right_valid:
            print("[ERROR] Timed out waiting for fresh skeleton frames", file=sys.stderr)
            return 3

        left_qpos, left_qpos_valid, right_qpos, right_qpos_valid = compute_wuji_qpos_from_keypoints(
            left_keypoints,
            right_keypoints,
            left_retargeter,
            right_retargeter,
        )
        left_qpos = apply_wuji_qpos_limits(left_qpos)
        right_qpos = apply_wuji_qpos_limits(right_qpos)
        print(f"left_qpos_shape={left_qpos.shape} valid={left_qpos_valid}")
        print(f"right_qpos_shape={right_qpos.shape} valid={right_qpos_valid}")
        if left_qpos.shape != (WUJI_QPOS_SIZE,) or right_qpos.shape != (WUJI_QPOS_SIZE,):
            print("[ERROR] Retargeted qpos has unexpected shape", file=sys.stderr)
            return 4
        if not left_qpos_valid or not right_qpos_valid:
            print("[ERROR] Retargeted qpos is invalid", file=sys.stderr)
            return 5
        print("[OK] Wuji glove skeleton -> 20D qpos probe passed")
        return 0
    finally:
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
