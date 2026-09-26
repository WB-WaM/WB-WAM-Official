#!/usr/bin/env python3
"""Hand-only MANUS Integrated SDK -> Wuji Hand command publisher."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MANUS_BINDING_DIR = (
    REPO_ROOT
    / "third_party"
    / "MANUS_SDK"
    / "ManusSDK_v3.1.1"
    / "SDKClient_Linux"
)
for import_root in (
    MANUS_BINDING_DIR,
    REPO_ROOT / "tracker" / "sonic",
    REPO_ROOT / "third_party" / "wuji_retargeting",
):
    if import_root.is_dir():
        import_root_str = str(import_root)
        if import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)

try:
    import zmq
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("pyzmq is required for hand-only testing") from exc

from gear_sonic.scripts.pico_manager_thread_server import (  # noqa: E402
    WUJI_QPOS_SIZE,
    apply_wuji_qpos_limits,
    compute_wuji_qpos_from_keypoints,
    init_wuji_retargeters,
)
from gear_sonic.scripts.manus_provider import (  # noqa: E402
    DEFAULT_MANUS_CALIBRATION_FILE,
    DEFAULT_MANUS_INIT_TIMEOUT_S,
    DEFAULT_MANUS_LEFT_CALIBRATION_FILE,
    DEFAULT_MANUS_MCP_JOINT,
    DEFAULT_MANUS_RIGHT_CALIBRATION_FILE,
    DEFAULT_MANUS_STALE_S,
    ManusIntegratedProvider,
    VALID_MANUS_MCP_JOINTS,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402

DEFAULT_LEFT_RETARGETING_CONFIG = (
    REPO_ROOT / "third_party" / "wuji_retargeting" / "example" / "config" / "retarget_manus_left.yaml"
)
DEFAULT_RIGHT_RETARGETING_CONFIG = (
    REPO_ROOT / "third_party" / "wuji_retargeting" / "example" / "config" / "retarget_manus_right.yaml"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish Wuji hand qpos from MANUS Integrated SDK raw skeleton streams.",
    )
    parser.add_argument("--bind", default="tcp://*:5556", help="Pose PUB bind endpoint for robot hand server.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means run until Ctrl-C.")
    parser.add_argument("--stale-s", type=float, default=DEFAULT_MANUS_STALE_S)
    parser.add_argument("--init-timeout-s", type=float, default=DEFAULT_MANUS_INIT_TIMEOUT_S)
    parser.add_argument("--left-retargeting-config", default=str(DEFAULT_LEFT_RETARGETING_CONFIG))
    parser.add_argument("--right-retargeting-config", default=str(DEFAULT_RIGHT_RETARGETING_CONFIG))
    parser.add_argument(
        "--calibration-file",
        default=str(DEFAULT_MANUS_CALIBRATION_FILE),
        help="Fallback calibration file used when side-specific files are missing.",
    )
    parser.add_argument("--left-calibration-file", default=str(DEFAULT_MANUS_LEFT_CALIBRATION_FILE))
    parser.add_argument("--right-calibration-file", default=str(DEFAULT_MANUS_RIGHT_CALIBRATION_FILE))
    parser.add_argument("--no-load-calibration", action="store_true")
    parser.add_argument(
        "--manus-mcp-joint",
        choices=VALID_MANUS_MCP_JOINTS,
        default=DEFAULT_MANUS_MCP_JOINT,
        help="Which MANUS non-thumb joint should be used as the MediaPipe MCP landmark.",
    )
    parser.add_argument("--debug-node-info", action="store_true", help="Print one MANUS raw node table per glove.")
    parser.add_argument("--status-endpoint", default="", help="Optional robot hand_status SUB endpoint for feedback logs.")
    parser.add_argument("--yes", action="store_true", help="Skip live-command confirmation prompt.")
    return parser


def _confirm(args: argparse.Namespace) -> None:
    if args.yes:
        return
    print("MANUS hand-only test will command robot hands.")
    print(f"  publish_bind:     {args.bind}")
    print(f"  fps:              {args.fps:g}")
    print(f"  stale_s:          {args.stale_s:g}")
    if args.no_load_calibration:
        print("  calibration:      <disabled>")
    else:
        print(f"  left_calibration: {args.left_calibration_file}")
        print(f"  right_calibration:{args.right_calibration_file}")
    print("Make sure run_pico_manager.sh is NOT running on the same PC/port.")
    typed = input("Type HAND to continue: ").strip()
    if typed != "HAND":
        raise SystemExit("aborted")


def _make_status_sub(context: zmq.Context, endpoint: str):
    endpoint = str(endpoint or "").strip()
    if not endpoint:
        return None
    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"hand_status")
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 0)
    sock.connect(endpoint)
    return sock


def _decode_status(message: bytes) -> dict[str, np.ndarray]:
    collector_root = REPO_ROOT / "collector" / "sonic"
    collector_root_str = str(collector_root)
    if collector_root_str not in sys.path:
        sys.path.insert(0, collector_root_str)
    from core.subscriber_hub import decode_topic_message  # noqa: PLC0415

    return decode_topic_message(message, "hand_status").payload


def _drain_status(status_sock) -> dict[str, np.ndarray] | None:
    if status_sock is None:
        return None
    latest = None
    while True:
        try:
            latest = _decode_status(status_sock.recv(zmq.NOBLOCK))
        except zmq.Again:
            return latest


def _as_bool(payload: dict[str, np.ndarray], key: str) -> bool:
    value = np.asarray(payload.get(key, np.array([False], dtype=bool))).reshape(-1)
    return bool(value[0]) if value.size else False


def _payload(
    *,
    frame_index: int,
    left_qpos: np.ndarray,
    right_qpos: np.ndarray,
    left_valid: bool,
    right_valid: bool,
) -> dict[str, np.ndarray]:
    now = time.monotonic()
    now_ns = time.monotonic_ns()
    return {
        "frame_index": np.array([frame_index], dtype=np.int64),
        "timestamp_monotonic": np.array([now], dtype=np.float64),
        "timestamp_monotonic_ns": np.array([now_ns], dtype=np.int64),
        "left_wuji_qpos": np.asarray(left_qpos, dtype=np.float32).reshape(WUJI_QPOS_SIZE),
        "right_wuji_qpos": np.asarray(right_qpos, dtype=np.float32).reshape(WUJI_QPOS_SIZE),
        "left_wuji_qpos_valid": np.array([left_valid], dtype=bool),
        "right_wuji_qpos_valid": np.array([right_valid], dtype=bool),
        "left_hand_binary_closed": np.array([False], dtype=bool),
        "right_hand_binary_closed": np.array([False], dtype=bool),
    }


def main() -> int:
    args = _build_parser().parse_args()
    if args.fps <= 0.0:
        raise ValueError("--fps must be > 0")
    _confirm(args)

    left_retargeter, right_retargeter = init_wuji_retargeters(
        args.left_retargeting_config,
        args.right_retargeting_config,
    )
    if left_retargeter is None or right_retargeter is None:
        raise RuntimeError(
            "failed to initialize MANUS retargeters; check retarget_manus_{left,right}.yaml"
        )

    context = zmq.Context.instance()
    pub = context.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(args.bind)
    status_sock = _make_status_sub(context, args.status_endpoint)

    provider = ManusIntegratedProvider(
        stale_s=args.stale_s,
        init_timeout_s=args.init_timeout_s,
        calibration_file=args.calibration_file,
        left_calibration_file=args.left_calibration_file,
        right_calibration_file=args.right_calibration_file,
        load_calibration=not args.no_load_calibration,
        debug_node_info=args.debug_node_info,
        mcp_joint=args.manus_mcp_joint,
    )
    stop = False

    def _handle_stop(signum, frame):  # noqa: ARG001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    print("[OK] MANUS hand-only publisher started")
    print(f"[INFO] Robot-side run_wuji_hand_server.sh should subscribe to this PC at {args.bind}")
    if args.status_endpoint:
        print(f"[INFO] Listening for hand_status feedback at {args.status_endpoint}")

    period_s = 1.0 / args.fps
    deadline = time.monotonic() + args.duration_s if args.duration_s > 0.0 else None
    frame_index = 0
    last_report = 0.0
    last_status_at = 0.0
    warned_no_status = False
    last_left_keypoints = None
    last_right_keypoints = None
    left_qpos = np.zeros((WUJI_QPOS_SIZE,), dtype=np.float32)
    right_qpos = np.zeros((WUJI_QPOS_SIZE,), dtype=np.float32)

    try:
        while not stop:
            loop_start = time.monotonic()
            if deadline is not None and loop_start >= deadline:
                break

            left_keypoints, left_fresh, right_keypoints, right_fresh = provider.latest_keypoints(
                last_left_keypoints,
                last_right_keypoints,
            )
            if left_fresh:
                last_left_keypoints = left_keypoints.copy()
            if right_fresh:
                last_right_keypoints = right_keypoints.copy()

            left_qpos, left_valid, right_qpos, right_valid = compute_wuji_qpos_from_keypoints(
                left_keypoints,
                right_keypoints,
                left_retargeter,
                right_retargeter,
            )
            left_valid = bool(left_valid and left_fresh)
            right_valid = bool(right_valid and right_fresh)
            left_qpos = apply_wuji_qpos_limits(left_qpos)
            right_qpos = apply_wuji_qpos_limits(right_qpos)

            pub.send(
                pack_pose_message(
                    _payload(
                        frame_index=frame_index,
                        left_qpos=left_qpos,
                        right_qpos=right_qpos,
                        left_valid=left_valid,
                        right_valid=right_valid,
                    ),
                    topic="pose",
                )
            )

            status = _drain_status(status_sock)
            now = time.monotonic()
            if status is not None:
                last_status_at = now
                warned_no_status = False
            elif (
                status_sock is not None
                and frame_index > max(10, int(args.fps * 3.0))
                and last_status_at <= 0.0
                and not warned_no_status
            ):
                print(
                    "[WARN] No hand_status feedback received yet. "
                    "Robot-side run_wuji_hand_server.sh may not be running, may still use old serials, "
                    "or PC_ZMQ_HOST/status endpoint may be wrong."
                )
                warned_no_status = True
            if now - last_report >= 1.0:
                line = (
                    f"[HAND] frame={frame_index} valid L={int(left_valid)} R={int(right_valid)} "
                    f"qpos0 L={left_qpos[0]:.3f} R={right_qpos[0]:.3f}"
                )
                if status is not None:
                    line += (
                        " apply"
                        f" L={int(_as_bool(status, 'left_apply_success'))}"
                        f" R={int(_as_bool(status, 'right_apply_success'))}"
                        " actual"
                        f" L={int(_as_bool(status, 'left_actual_position_valid'))}"
                        f" R={int(_as_bool(status, 'right_actual_position_valid'))}"
                    )
                print(line)
                last_report = now

            frame_index += 1
            elapsed = time.monotonic() - loop_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
    finally:
        provider.close()
        pub.close(0)
        if status_sock is not None:
            status_sock.close(0)

    print("[OK] MANUS hand-only publisher stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
