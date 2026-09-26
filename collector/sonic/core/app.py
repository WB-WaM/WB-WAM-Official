from __future__ import annotations

import logging
import time
from typing import Any

from .args import parse_args
from .camera_discovery import format_camera_lines, resolve_camera_selector
from .camera_manager import MultiCameraManager
from .constants import CAMERA_HEIGHT, CAMERA_WIDTH
from .dataset_layout import DatasetLayout
from .episode_state_machine import EpisodeStateMachine
from .episode_writer import EpisodeWriter, compress_deferred_depth_outputs
from .key_input import KeyCommandReader
from .official_latest import build_official_latest_snapshot
from .startup_gate import StartupGate
from .subscriber_hub import SubscriberHub

LOGGER = logging.getLogger(__name__)
REMOTE_DISCOVERY_RETRY_DELAY_S = 1.0


def build_episode_writer(args: Any) -> EpisodeWriter:
    sample_modes = {"collector_timer": "nearest_hold", "official_latest": "tick_latest"}
    return EpisodeWriter(
        keep_temp_logs=args.keep_temp_logs,
        record_depth=args.record_depth,
        defer_depth_compression=args.defer_depth_compression,
        capture_fps=args.capture_fps,
        sample_mode=sample_modes[args.capture_mode],
    )


class CollectorApp:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def _log_idle_prompt() -> None:
        LOGGER.info("collector ready: s=start q=commit d=discard type 'exit' to quit")

    @staticmethod
    def _normalize_keyboard_command(command: str) -> str:
        return {"s": "start", "q": "save", "d": "discard"}.get(command, command)

    def _handle_runtime_command(
        self,
        command: str,
        *,
        state_machine: EpisodeStateMachine,
        writer: EpisodeWriter,
        hub: SubscriberHub,
        startup_gate: StartupGate,
        initial_capture_time: float | None,
    ) -> tuple[bool, float | None]:
        if command == "start":
            if state_machine.is_idle:
                startup_gate.reset()
                state_machine.start_waiting()
                LOGGER.info("waiting for first valid human+robot sample before starting episode")
            elif state_machine.is_waiting:
                LOGGER.warning("recording start already requested; still waiting for the first valid frame")
            elif state_machine.is_recording:
                LOGGER.warning("recording already in progress; use save or discard first")
            return False, initial_capture_time

        if command == "save":
            if state_machine.is_recording:
                writer.append_subscriber_samples(hub.drain(0))
                data_path = writer.commit()
                writer.finalize()
                state_machine.finish()
                LOGGER.info("saved episode: %s", data_path)
                LOGGER.info("episode saved successfully; ready for the next command")
                self._log_idle_prompt()
                return False, None
            if state_machine.is_waiting:
                LOGGER.warning("not recording yet; still waiting for the first valid human+robot sample")
            else:
                LOGGER.warning("no recording in progress; ignoring save command")
            return False, initial_capture_time

        if command == "discard":
            if state_machine.is_recording:
                writer.discard()
                state_machine.finish()
                LOGGER.info("discarded current episode")
                self._log_idle_prompt()
                return False, None
            if state_machine.is_waiting:
                startup_gate.reset()
                state_machine.finish()
                LOGGER.info("cancelled waiting for recording start")
                self._log_idle_prompt()
            else:
                LOGGER.warning("no recording in progress; ignoring discard command")
            return False, initial_capture_time

        if command == "exit":
            if state_machine.is_recording:
                LOGGER.warning("recording in progress; use q to save or d to discard first")
                return False, initial_capture_time
            return True, initial_capture_time

        LOGGER.warning("unknown collector command: %s", command)
        return False, initial_capture_time

    def _choose_primary_camera(self, cameras):
        if self.args.broadcast_camera:
            return resolve_camera_selector(cameras, self.args.broadcast_camera)
        if len(cameras) == 1:
            return cameras[0]

        print("Connected RealSense cameras:")
        for line in format_camera_lines(cameras):
            print(f"  {line}")

        while True:
            selection = input("Select primary XR broadcast camera by index, key, or serial: ").strip()
            try:
                return resolve_camera_selector(cameras, selection)
            except ValueError as exc:
                print(exc)

    def _wait_for_initial_frames(
        self,
        camera_manager: Any,
        timeout_s: float = 5.0,
    ):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshots = camera_manager.get_latest_frame_snapshots()
            if len(snapshots) == len(camera_manager.camera_keys):
                return snapshots
            time.sleep(0.01)
        return None

    def _validate_listen_backend(self) -> None:
        if self.args.disable_listen:
            return
        if self.args.listen_backend != "gst":
            return

        from .gst_streamer import require_gst_environment

        require_gst_environment()

    def _discover_cameras(self):
        if self.args.camera_backend == "remote_realsense":
            from .remote_camera_manager import discover_remote_cameras

            endpoint = self.args.camera_endpoint
            while True:
                try:
                    cameras = discover_remote_cameras(endpoint)
                except Exception as exc:
                    LOGGER.warning(
                        "waiting for remote camera server at %s: %s",
                        endpoint,
                        exc,
                    )
                    time.sleep(REMOTE_DISCOVERY_RETRY_DELAY_S)
                    continue

                if cameras:
                    return cameras

                LOGGER.warning(
                    "remote camera server at %s is reachable but no cameras are available yet; retrying",
                    endpoint,
                )
                time.sleep(REMOTE_DISCOVERY_RETRY_DELAY_S)

        from .camera_discovery import discover_cameras

        return discover_cameras()

    def _build_camera_manager(self, cameras):
        if self.args.camera_backend == "remote_realsense":
            from .remote_camera_manager import RemoteMultiCameraManager

            return RemoteMultiCameraManager(
                cameras,
                endpoint=self.args.camera_endpoint,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=self.args.camera_fps,
                record_depth=self.args.record_depth,
            )

        return MultiCameraManager(
            cameras,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=self.args.camera_fps,
            record_depth=self.args.record_depth,
        )

    def run(self) -> int:
        log_level = getattr(logging, str(self.args.log_level).upper(), logging.INFO)
        logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

        self._validate_listen_backend()

        cameras = self._discover_cameras()
        if not cameras:
            raise RuntimeError("no RealSense cameras detected")

        primary_camera = self._choose_primary_camera(cameras)
        LOGGER.info("primary XR camera: %s (%s)", primary_camera.key, primary_camera.serial)
        for line in format_camera_lines(cameras):
            LOGGER.info("camera: %s", line)
        LOGGER.info("capture fps: %d", self.args.capture_fps)
        LOGGER.info("camera fps: %d", self.args.camera_fps)
        LOGGER.info("capture mode: %s", self.args.capture_mode)
        LOGGER.info("depth recording: %s", "enabled" if self.args.record_depth else "disabled")
        LOGGER.info("hand control mode: %s", self.args.hand_control_mode)
        LOGGER.info("hand backend: %s", self.args.hand_backend)
        if self.args.hand_status_endpoint:
            LOGGER.info("hand status endpoint: %s", self.args.hand_status_endpoint)
        else:
            LOGGER.info("hand status endpoint: disabled")
        LOGGER.info(
            "depth compression: %s",
            "deferred until exit" if self.args.defer_depth_compression else "per episode",
        )

        is_official_capture = self.args.capture_mode == "official_latest"
        layout = DatasetLayout(self.args.output_root, self.args.task_name)
        layout.write_metadata(
            robot_name=self.args.robot_name,
            primary_camera=primary_camera.key,
            connected_cameras=cameras,
            capture_fps=self.args.capture_fps,
            camera_fps=self.args.camera_fps,
            capture_mode=self.args.capture_mode,
            clock_source="collector_monotonic_timer",
            camera_alignment=(
                "latest_pc_receive_at_or_before_collector_tick" if is_official_capture else "nearest_with_hold"
            ),
            record_depth=self.args.record_depth,
            notes=self.args.notes,
            camera_backend=self.args.camera_backend,
            camera_endpoint=self.args.camera_endpoint,
            capture_transport=self.args.camera_backend,
            hand_control_mode=self.args.hand_control_mode,
            hand_backend=self.args.hand_backend,
            hand_status_endpoint=self.args.hand_status_endpoint,
            hand_command_source="pose_topic",
            defer_depth_compression=self.args.defer_depth_compression,
        )

        state_machine = EpisodeStateMachine()
        writer = build_episode_writer(self.args)
        hub = SubscriberHub(
            self.args.pose_endpoint,
            self.args.state_action_endpoint,
            self.args.hand_status_endpoint,
        )
        camera_manager = self._build_camera_manager(cameras)
        xr_server = None
        frame_count = 0
        last_written_sequences: dict[str, int] = {}
        initial_capture_time: float | None = None
        last_hand_feedback_warning_time = 0.0
        startup_gate = StartupGate(capture_fps=self.args.capture_fps)
        capture_delay_s = 1.0 / self.args.capture_fps

        try:
            camera_manager.start()
            if not self.args.disable_listen:
                if self.args.listen_backend == "psi0_image":
                    from .psi0_image_listen_server import Psi0ImageListenServer

                    xr_server = Psi0ImageListenServer(self.args.listen, camera_manager, primary_camera.key)
                else:
                    from .xr_listen_server import XRListenServer

                    xr_server = XRListenServer(
                        self.args.listen,
                        camera_manager,
                        primary_camera.key,
                        video_host=self.args.xr_video_host,
                    )
                xr_server.start()

            self._log_idle_prompt()
            with KeyCommandReader() as key_reader:
                while True:
                    drained = hub.drain(0)
                    if state_machine.is_recording:
                        writer.append_subscriber_samples(drained)
                        if self.args.hand_status_endpoint:
                            now = time.monotonic()
                            latest_hand_status = hub.latest_hand_status
                            hand_status_stale = (
                                latest_hand_status is None
                                or latest_hand_status.timestamp_ns < 0
                                or time.monotonic_ns() - latest_hand_status.timestamp_ns > 1_000_000_000
                            )
                            if hand_status_stale and now - last_hand_feedback_warning_time > 5.0:
                                LOGGER.warning(
                                    "hand_status feedback missing or stale; continuing recording without blocking"
                                )
                                last_hand_feedback_warning_time = now

                    commands = [self._normalize_keyboard_command(command) for command in key_reader.poll()]
                    commands.extend(sample.command_name for sample in drained.collector_controls)
                    for command in commands:
                        should_exit, initial_capture_time = self._handle_runtime_command(
                            command,
                            state_machine=state_machine,
                            writer=writer,
                            hub=hub,
                            startup_gate=startup_gate,
                            initial_capture_time=initial_capture_time,
                        )
                        if should_exit:
                            return 0

                    if state_machine.is_waiting:
                        startup_gate.ingest(drained)
                        reference_time_ns = time.monotonic_ns()
                        if is_official_capture:
                            snapshots = camera_manager.get_causal_frame_matches(reference_time_ns)
                        else:
                            snapshots = camera_manager.get_latest_frame_snapshots()
                        if startup_gate.ready(
                            camera_snapshot_count=len(snapshots),
                            expected_camera_count=len(camera_manager.camera_keys),
                            reference_time_ns=reference_time_ns,
                        ):
                            episode_dir = layout.next_episode_dir()
                            writer.start_episode(
                                episode_dir,
                                cameras,
                                primary_camera=primary_camera.key,
                            )
                            writer.append_subscriber_samples(startup_gate.build_start_samples())
                            latest_samples = None
                            if is_official_capture:
                                latest_samples = build_official_latest_snapshot(
                                    tick_time_ns=reference_time_ns,
                                    pose_sample=hub.latest_pose_sample,
                                    state_action=hub.latest_state_action,
                                    hand_status=hub.latest_hand_status,
                                )
                            writer.append_tick(
                                frame_index=0,
                                tick_time_ns=reference_time_ns,
                                primary_camera=primary_camera.key,
                                camera_snapshots=snapshots,
                                previous_sequences={},
                                latest_samples=latest_samples,
                            )
                            state_machine.start_recording()
                            startup_gate.reset()
                            frame_count = 1
                            if is_official_capture:
                                last_written_sequences = {key: match.sequence for key, match in snapshots.items()}
                                initial_capture_time = time.monotonic() + capture_delay_s
                            else:
                                last_written_sequences = {
                                    key: sequence for key, (sequence, _) in snapshots.items()
                                }
                                initial_capture_time = time.monotonic()
                            LOGGER.info("recording episode: %s", episode_dir.name)
                        else:
                            time.sleep(0.01)
                        continue

                    if not state_machine.is_recording:
                        time.sleep(0.01)
                        continue

                    if is_official_capture:
                        assert initial_capture_time is not None
                        time_now = time.monotonic()
                        if initial_capture_time > time_now:
                            time.sleep(min(0.005, initial_capture_time - time_now))
                            continue

                        tick_time_ns = time.monotonic_ns()
                        snapshots = camera_manager.get_causal_frame_matches(
                            tick_time_ns,
                            last_written_sequences,
                        )
                        if len(snapshots) != len(camera_manager.camera_keys):
                            time.sleep(0.001)
                            continue

                        latest_samples = build_official_latest_snapshot(
                            tick_time_ns=tick_time_ns,
                            pose_sample=hub.latest_pose_sample,
                            state_action=hub.latest_state_action,
                            hand_status=hub.latest_hand_status,
                        )
                        writer.append_tick(
                            frame_index=frame_count,
                            tick_time_ns=tick_time_ns,
                            primary_camera=primary_camera.key,
                            camera_snapshots=snapshots,
                            previous_sequences=last_written_sequences,
                            latest_samples=latest_samples,
                        )
                        last_written_sequences = {key: match.sequence for key, match in snapshots.items()}
                        frame_count += 1
                        initial_capture_time = time_now + capture_delay_s
                        if frame_count % self.args.capture_fps == 0:
                            LOGGER.info("recorded %d frames", frame_count)
                        continue

                    snapshots = camera_manager.get_latest_frame_snapshots()
                    if len(snapshots) != len(camera_manager.camera_keys):
                        time.sleep(0.001)
                        continue

                    assert initial_capture_time is not None
                    next_capture_time = initial_capture_time + frame_count * capture_delay_s
                    time_now = time.monotonic()
                    if next_capture_time > time_now:
                        time.sleep(min(0.005, next_capture_time - time_now))
                        continue

                    if time_now - next_capture_time > capture_delay_s:
                        LOGGER.warning(
                            "collector tick lagging by %.2f ms",
                            (time_now - next_capture_time) * 1000.0,
                        )

                    writer.append_tick(
                        frame_index=frame_count,
                        tick_time_ns=time.monotonic_ns(),
                        primary_camera=primary_camera.key,
                        camera_snapshots=snapshots,
                        previous_sequences=last_written_sequences,
                    )
                    last_written_sequences = {key: sequence for key, (sequence, _) in snapshots.items()}
                    frame_count += 1
                    if frame_count % self.args.capture_fps == 0:
                        LOGGER.info("recorded %d frames", frame_count)
        finally:
            if state_machine.is_recording:
                LOGGER.warning("collector stopped while recording; discarding incomplete episode")
                writer.discard()
                state_machine.finish()
            if xr_server is not None:
                xr_server.stop()
            camera_manager.stop()
            hub.close()
            if self.args.record_depth and self.args.defer_depth_compression:
                compress_deferred_depth_outputs(layout.task_root, use_tqdm=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = CollectorApp(args)
    return app.run()
