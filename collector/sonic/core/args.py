from __future__ import annotations

import argparse
from pathlib import Path

from .constants import (
    DEFAULT_CAPTURE_FPS,
    SUPPORTED_CAMERA_FPS,
    SUPPORTED_CAPTURE_FPS,
    default_camera_fps_for_capture,
    normalize_camera_fps,
    normalize_capture_fps,
)

DEFAULT_LISTEN = "0.0.0.0:13579"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "sonic"
DEFAULT_CAMERA_ENDPOINT = "tcp://127.0.0.1:5560"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone multi-camera collector with integrated XR listen service.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--robot-name", type=str, default="g1")
    parser.add_argument("--pose-endpoint", type=str, default="tcp://127.0.0.1:5556")
    parser.add_argument("--state-action-endpoint", type=str, default="tcp://127.0.0.1:5558")
    parser.add_argument(
        "--camera-backend",
        type=str,
        default="local_realsense",
        choices=("local_realsense", "remote_realsense"),
    )
    parser.add_argument("--camera-endpoint", type=str, default=DEFAULT_CAMERA_ENDPOINT)
    parser.add_argument(
        "--hand-control-mode",
        type=str,
        default="gesture_wuji",
        choices=("gesture_wuji", "binary_trigger", "wuji_glove", "manus"),
        help="Task-level hand control mode used by the teleoperation stack.",
    )
    parser.add_argument(
        "--hand-backend",
        type=str,
        default="local_wuji",
        choices=("local_wuji", "remote_wuji_proxy"),
        help="Deploy hand hardware backend used for this task.",
    )
    parser.add_argument(
        "--hand-status-endpoint",
        type=str,
        default="",
        help="Optional robot-side hand_status PUB endpoint, e.g. tcp://192.168.123.164:5559.",
    )
    parser.add_argument("--listen", type=str, default=DEFAULT_LISTEN)
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=DEFAULT_CAPTURE_FPS,
        choices=SUPPORTED_CAPTURE_FPS,
        help=(
            "Dataset FPS (default: 20). collector_timer supports 20/30; "
            "50 requires --capture-mode official_latest."
        ),
    )
    parser.add_argument(
        "--capture-mode",
        type=str,
        default="collector_timer",
        choices=("collector_timer", "official_latest"),
        help=(
            "Capture sampling mode: collector_timer uses nearest/hold at 20/30Hz; "
            "official_latest latches the latest received samples at 50Hz."
        ),
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=None,
        choices=SUPPORTED_CAMERA_FPS,
        help="Physical/poll camera FPS. Defaults to 60 for 50Hz capture, otherwise capture-fps.",
    )
    parser.add_argument(
        "--xr-video-host",
        type=str,
        default="",
        help="Optional headset video return IP override. Empty uses the IP from OPEN_CAMERA.",
    )
    parser.add_argument(
        "--listen-backend",
        type=str,
        default="gst",
        choices=("gst", "psi0_image"),
        help="Preview backend for headset/browser listening. Use psi0_image to avoid gi/GStreamer.",
    )
    parser.add_argument(
        "--broadcast-camera",
        type=str,
        default="",
        help="Primary camera for XR broadcast. Accepts key, serial, or 1-based index.",
    )
    parser.add_argument("--disable-listen", action="store_true")
    parser.add_argument(
        "--record-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Record depth images alongside color images. "
            "Use --no-record-depth to disable depth capture and storage."
        ),
    )
    parser.add_argument(
        "--defer-depth-compression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Defer depth .npy.lzma compression until collector exit.",
    )
    parser.add_argument("--keep-temp-logs", action="store_true")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.broadcast_camera = args.broadcast_camera.strip()
    args.camera_endpoint = args.camera_endpoint.strip()
    args.hand_status_endpoint = args.hand_status_endpoint.strip()
    args.xr_video_host = args.xr_video_host.strip()
    args.capture_fps = normalize_capture_fps(args.capture_fps)
    args.camera_fps = normalize_camera_fps(
        args.camera_fps if args.camera_fps is not None else default_camera_fps_for_capture(args.capture_fps)
    )
    if args.capture_fps == 50:
        compatible_modes = {"official_latest"}
    else:
        compatible_modes = {"collector_timer"}
    if args.capture_mode not in compatible_modes:
        parser.error(f"--capture-mode {args.capture_mode!r} is incompatible with --capture-fps {args.capture_fps}")
    return args
