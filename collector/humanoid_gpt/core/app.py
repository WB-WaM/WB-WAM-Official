"""Timestamp-aligned 20/50 Hz HumanoidGPT real-robot collector."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

from collector.sonic.core.camera_discovery import (
    discover_cameras,
    resolve_camera_selector,
)
from collector.sonic.core.camera_manager import MultiCameraManager
from collector.sonic.core.episode_writer import compress_deferred_depth_outputs
from collector.sonic.core.key_input import KeyCommandReader
from collector.sonic.core.remote_camera_manager import (
    RemoteMultiCameraManager,
    discover_remote_cameras,
)
from collector.sonic.core.xr_listen_server import XRListenServer

from .episode_writer import EpisodeWriter
from .health import (
    HealthIssue,
    check_health,
    sample_timestamps,
)
from .schema import build_snapshot, snapshot_ready
from .subscriber import SubscriberHub

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]


def _interpolation_anchor_ns(pair, state_sample) -> int | None:
    timestamps = sample_timestamps(pair, state_sample, None)
    if "pose" not in timestamps or "robot" not in timestamps:
        return None
    return max(timestamps["pose"], timestamps["robot"])


def _initial_training_gate(
    *,
    already_passed: bool,
    tick_ns: int,
    pair,
    state_sample,
    hand_sample,
    snapshot,
) -> tuple[bool, tuple[HealthIssue, ...]]:
    """Check training streams once per collector process, before episode one."""

    if already_passed:
        return True, ()
    issues = check_health(
        tick_ns=tick_ns,
        pair=pair,
        state=state_sample,
        hand=hand_sample,
        snapshot=snapshot,
        camera_matches={},
        expected_camera_keys=(),
        record_depth=False,
        require_training_actions=True,
    )
    return not issues, issues


def _append_frame(writer, **kwargs) -> bool:
    try:
        writer.append(**kwargs)
    except Exception as exc:
        LOGGER.warning(
            "failed to append frame %s to active episode %s: %s; capture frozen, retry save or discard",
            kwargs.get("frame_index", "?"),
            writer.episode_dir,
            exc,
        )
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HumanoidGPT timestamp-aligned 20/50 Hz real-robot collector")
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "humanoid_gpt",
    )
    parser.add_argument("--capture-fps", type=int, choices=(20, 50), default=20)
    parser.add_argument("--camera-fps", type=int, choices=(20, 30, 60), default=None)
    parser.add_argument(
        "--downsample-method",
        choices=("auto", "causal_latest", "interpolate"),
        default="auto",
        help="Temporal sampling; auto selects interpolate at 20 Hz and causal_latest at 50 Hz",
    )
    parser.add_argument(
        "--interpolation-delay-ms",
        type=float,
        default=40.0,
        help="Collector-only look-ahead delay for interpolation mode",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("local_realsense", "remote_realsense"),
        default="remote_realsense",
    )
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--primary-camera", default="")
    parser.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--state-action-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--hand-status-endpoint", default="")
    parser.add_argument("--record-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--defer-depth-compression",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--disable-listen", action="store_true")
    parser.add_argument("--listen", default="0.0.0.0:13579")
    parser.add_argument("--xr-video-host", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser


class CollectorApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.args.output_root = self.args.output_root.expanduser().resolve()
        if self.args.downsample_method == "auto":
            self.args.downsample_method = "interpolate" if self.args.capture_fps == 20 else "causal_latest"
        if self.args.downsample_method == "interpolate" and self.args.capture_fps != 20:
            raise ValueError("--downsample-method interpolate requires --capture-fps 20")
        if self.args.interpolation_delay_ms < 0.0:
            raise ValueError("--interpolation-delay-ms must be non-negative")
        self.interpolation_delay_ns = int(round(self.args.interpolation_delay_ms * 1e6))
        self.camera_fps = self.args.camera_fps if self.args.camera_fps is not None else 60

    def _camera_matches(self, manager, tick_ns, previous_sequences):
        if self.args.downsample_method == "interpolate":
            return manager.get_nearest_frame_matches(
                tick_ns,
                previous_sequences,
                max_future_ns=self.interpolation_delay_ns,
            )
        return manager.get_causal_frame_matches(tick_ns, previous_sequences)

    def _discover(self):
        if self.args.camera_backend == "local_realsense":
            return discover_cameras()
        while True:
            try:
                cameras = discover_remote_cameras(self.args.camera_endpoint)
                if cameras:
                    return cameras
            except Exception as exc:
                LOGGER.warning("waiting for camera server: %s", exc)
            time.sleep(1.0)

    def _manager(self, cameras):
        common = dict(
            width=640,
            height=480,
            fps=self.camera_fps,
            record_depth=self.args.record_depth,
        )
        if self.args.camera_backend == "remote_realsense":
            return RemoteMultiCameraManager(cameras, endpoint=self.args.camera_endpoint, **common)
        return MultiCameraManager(cameras, **common)

    def run(self) -> int:
        logging.basicConfig(
            level=getattr(logging, self.args.log_level.upper(), logging.INFO),
            format="[%(levelname)s] %(message)s",
        )
        LOGGER.info(
            "endpoints: camera=%s pose=%s robot_state_action=%s hand_status=%s",
            self.args.camera_endpoint,
            self.args.pose_endpoint,
            self.args.state_action_endpoint,
            self.args.hand_status_endpoint or "<disabled>",
        )
        writer = EpisodeWriter(
            self.args.output_root,
            self.args.task_name,
            capture_fps=self.args.capture_fps,
            record_depth=self.args.record_depth,
            defer_depth_compression=self.args.defer_depth_compression,
            downsample_method=self.args.downsample_method,
            interpolation_delay_ms=(
                self.args.interpolation_delay_ms if self.args.downsample_method == "interpolate" else 0.0
            ),
        )
        cameras = self._discover()
        if not cameras:
            raise RuntimeError("no RealSense camera available")
        primary = (
            resolve_camera_selector(cameras, self.args.primary_camera) if self.args.primary_camera else cameras[0]
        )
        manager = self._manager(cameras)
        hub = SubscriberHub(
            self.args.pose_endpoint,
            self.args.state_action_endpoint,
            self.args.hand_status_endpoint,
        )
        writer.write_metadata(
            cameras=cameras,
            primary_camera=primary.key,
            camera_fps=self.camera_fps,
            camera_backend=self.args.camera_backend,
            camera_endpoint=self.args.camera_endpoint,
            pose_endpoint=self.args.pose_endpoint,
            state_action_endpoint=self.args.state_action_endpoint,
            hand_status_endpoint=self.args.hand_status_endpoint,
            notes=self.args.notes,
            downsample_method=self.args.downsample_method,
            interpolation_delay_ms=(
                self.args.interpolation_delay_ms if self.args.downsample_method == "interpolate" else 0.0
            ),
        )

        xr = None
        state = "idle"
        frame_index = 0
        next_tick = 0.0
        period = 1.0 / self.args.capture_fps
        period_ns = int(round(1e9 / self.args.capture_fps))
        next_target_ns = 0
        previous_sequences: dict[str, int] = {}
        initial_training_check_passed = False
        last_report = time.monotonic()
        LOGGER.info(
            "HGPT collector ready: fps=%d camera_fps=%d downsample=%s; "
            "Pico right stick/A/B=start/save/discard; "
            "keyboard s/q/d=start/save/discard, exit=quit",
            self.args.capture_fps,
            self.camera_fps,
            self.args.downsample_method,
        )
        try:
            manager.start()
            if not self.args.disable_listen:
                xr = XRListenServer(
                    self.args.listen,
                    manager,
                    primary.key,
                    video_host=self.args.xr_video_host,
                )
                xr.start()

            with KeyCommandReader() as keys:
                while True:
                    hub.drain(0)
                    tick_ns = time.monotonic_ns()
                    pair = hub.latest_pair
                    state_sample = hub.latest_state
                    hand_sample = hub.latest_hand
                    snapshot = build_snapshot(
                        tick_ns=tick_ns,
                        pair=pair,
                        state=state_sample,
                        hand=hand_sample,
                    )
                    commands = [
                        {"s": "start", "q": "save", "d": "discard"}.get(command, command)
                        for command in keys.poll()
                    ]
                    commands.extend(hub.take_control_commands())
                    for command in commands:
                        if command == "start" and state == "idle":
                            check_was_passed = initial_training_check_passed
                            initial_training_check_passed, issues = _initial_training_gate(
                                already_passed=initial_training_check_passed,
                                tick_ns=tick_ns,
                                pair=pair,
                                state_sample=state_sample,
                                hand_sample=hand_sample,
                                snapshot=snapshot,
                            )
                            if issues:
                                for issue in issues:
                                    LOGGER.warning(
                                        "initial training data check failed: %s: %s",
                                        issue.code,
                                        issue.message,
                                    )
                                LOGGER.warning(
                                    "first episode rejected; fix body/hand "
                                    "state-action streams and start again"
                                )
                                continue
                            if not check_was_passed:
                                LOGGER.info(
                                    "initial body/hand state-action check passed; "
                                    "disabled for the rest of this collector session"
                                )
                            next_target_ns = (
                                0 if self.args.downsample_method == "interpolate" else time.monotonic_ns()
                            )
                            state = "waiting"
                            LOGGER.info("waiting for fresh camera+pose+robot data")
                        elif command == "save" and state in {
                            "recording",
                            "save_failed",
                        }:
                            try:
                                path = writer.commit()
                            except Exception as exc:
                                state = "save_failed"
                                LOGGER.warning(
                                    "failed to save active episode %s: %s; capture frozen, retry save or discard",
                                    writer.episode_dir,
                                    exc,
                                )
                            else:
                                state = "idle"
                                LOGGER.info("saved %s", path)
                        elif command == "discard" and state in {
                            "waiting",
                            "recording",
                            "save_failed",
                        }:
                            if state in {"recording", "save_failed"}:
                                writer.discard()
                            state = "idle"
                            LOGGER.info("discarded current episode")
                        elif command == "exit":
                            if state in {"recording", "save_failed"}:
                                LOGGER.warning("save or discard the active episode first")
                            else:
                                return 0

                    if state == "idle":
                        time.sleep(0.005)
                        continue
                    if state == "save_failed":
                        time.sleep(0.005)
                        continue

                    if state == "waiting" and self.args.downsample_method == "interpolate":
                        now_ns = time.monotonic_ns()
                        if next_target_ns <= 0:
                            anchor_ns = _interpolation_anchor_ns(
                                hub.latest_pair,
                                hub.latest_state,
                            )
                            if anchor_ns is None:
                                time.sleep(0.001)
                                continue
                            next_target_ns = anchor_ns
                        if now_ns < next_target_ns + self.interpolation_delay_ns:
                            time.sleep(0.001)
                            continue
                        inputs = hub.interpolated_inputs_at(next_target_ns)
                        if inputs is None:
                            latest_anchor_ns = _interpolation_anchor_ns(
                                hub.latest_pair,
                                hub.latest_state,
                            )
                            if latest_anchor_ns is not None:
                                if latest_anchor_ns - next_target_ns > 100_000_000:
                                    next_target_ns = latest_anchor_ns
                            time.sleep(0.001)
                            continue
                        tick_ns = next_target_ns
                        pair, state_sample, hand_sample = inputs
                        snapshot = build_snapshot(
                            tick_ns=tick_ns,
                            pair=pair,
                            state=state_sample,
                            hand=hand_sample,
                        )
                    if state == "waiting":
                        matches = self._camera_matches(
                            manager,
                            tick_ns,
                            previous_sequences,
                        )
                        cameras_ready = len(matches) == len(manager.camera_keys)
                        if cameras_ready and snapshot_ready(snapshot):
                            episode = writer.start(cameras)
                            appended = _append_frame(
                                writer,
                                frame_index=0,
                                tick_ns=tick_ns,
                                primary_camera=primary.key,
                                camera_matches=matches,
                                snapshot=snapshot,
                            )
                            if not appended:
                                state = "save_failed"
                                continue
                            previous_sequences = {key: match.sequence for key, match in matches.items()}
                            frame_index = 1
                            if self.args.downsample_method == "interpolate":
                                next_target_ns = tick_ns + period_ns
                            else:
                                next_tick = time.monotonic() + period
                            state = "recording"
                            LOGGER.info("recording %s", episode.name)
                        else:
                            time.sleep(0.005)
                        continue

                    if self.args.downsample_method == "interpolate":
                        now_ns = time.monotonic_ns()
                        ready_ns = next_target_ns + self.interpolation_delay_ns
                        if now_ns < ready_ns:
                            time.sleep(min(0.005, (ready_ns - now_ns) / 1e9))
                            continue
                        inputs = hub.interpolated_inputs_at(next_target_ns)
                        if inputs is None:
                            latest_anchor_ns = _interpolation_anchor_ns(
                                hub.latest_pair,
                                hub.latest_state,
                            )
                            if latest_anchor_ns is not None and latest_anchor_ns - next_target_ns > 100_000_000:
                                next_target_ns = latest_anchor_ns
                            time.sleep(0.001)
                            continue
                        tick_ns = next_target_ns
                        pair, state_sample, hand_sample = inputs
                        snapshot = build_snapshot(
                            tick_ns=tick_ns,
                            pair=pair,
                            state=state_sample,
                            hand=hand_sample,
                        )
                        now = time.monotonic()
                    else:
                        now = time.monotonic()
                        if now < next_tick:
                            time.sleep(min(0.005, next_tick - now))
                            continue
                        tick_ns = time.monotonic_ns()
                        hub.drain(0)
                        pair = hub.latest_pair
                        state_sample = hub.latest_state
                        hand_sample = hub.latest_hand
                        snapshot = build_snapshot(
                            tick_ns=tick_ns,
                            pair=pair,
                            state=state_sample,
                            hand=hand_sample,
                        )
                    matches = self._camera_matches(manager, tick_ns, previous_sequences)
                    if len(matches) != len(manager.camera_keys):
                        time.sleep(0.001)
                        continue
                    appended = _append_frame(
                        writer,
                        frame_index=frame_index,
                        tick_ns=tick_ns,
                        primary_camera=primary.key,
                        camera_matches=matches,
                        snapshot=snapshot,
                    )
                    if not appended:
                        state = "save_failed"
                        continue
                    previous_sequences = {key: match.sequence for key, match in matches.items()}
                    frame_index += 1
                    if self.args.downsample_method == "interpolate":
                        next_target_ns += period_ns
                    else:
                        next_tick = now + period
                    if time.monotonic() - last_report >= 2.0:
                        LOGGER.info(
                            "frames=%d pose_age=%.1fms robot_age=%.1fms "
                            "hand_age=%.1fms gaps=%d decode_errors=%d",
                            frame_index,
                            snapshot["pose_age_ns"] / 1e6,
                            snapshot["robot_age_ns"] / 1e6,
                            snapshot["hand_age_ns"] / 1e6,
                            hub.frame_gaps,
                            hub.decode_errors,
                        )
                        last_report = time.monotonic()
        finally:
            if state == "recording":
                LOGGER.warning("discarding incomplete active episode")
                writer.discard()
            elif state == "save_failed":
                LOGGER.warning(
                    "preserving failed episode at %s for manual recovery",
                    writer.episode_dir,
                )
                writer.close(cancel=True)
            else:
                writer.close()
            if xr is not None:
                xr.stop()
            manager.stop()
            hub.close()
            if state != "save_failed" and self.args.record_depth and self.args.defer_depth_compression:
                compress_deferred_depth_outputs(writer.task_root)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return CollectorApp(args).run()
