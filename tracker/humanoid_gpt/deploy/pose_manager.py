"""Independent Pico/GMR + WujiHand publisher for HGPT deployment."""

from __future__ import annotations

import multiprocessing as mp
import signal
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from deploy.keyboard_cmd import DeployKeyboardCMD
from deploy.offline_pose import OfflinePose, load_offline_directory
from deploy.pico import PicoFrameCalibrator, PicoXrtSource, pico_frame_to_gmr_frame
from deploy.pose_zmq import (
    COLLECTOR_CONTROL_COMMAND_NAMES,
    CollectorControlSnapshot,
    PicoLeftModeControls,
    PicoRightCollectorControls,
    PoseSnapshot,
    TeleopRawSnapshot,
    pack_stop_command,
)
from deploy.retarget import ema_smooth_g1_qpos, validate_g1_qpos
from deploy.wuji_hand import (
    WUJI_OPEN_QPOS,
    HandPair,
    PicoHandSource,
    WujiRetargetPair,
    default_retarget_configs,
)
from tracking.constants import DEFAULT_QPOS

DEFAULT_WUJI_HAND_STATUS_ENDPOINT = "tcp://192.168.123.164:5559"


@dataclass(frozen=True, slots=True)
class _HandWorkerSample:
    raw_left: np.ndarray
    raw_right: np.ndarray
    raw_left_valid: bool
    raw_right_valid: bool
    left_qpos: np.ndarray
    right_qpos: np.ndarray
    left_valid: bool
    right_valid: bool
    age_ms: float


def _run_manus_hand_worker(
    source_kwargs,
    left_config,
    right_config,
    poll_hz,
    raw_buffer,
    qpos_buffer,
    state_buffer,
    updated_ns,
    counters,
    shared_lock,
    stop_event,
    ready_event,
    error_conn,
) -> None:
    """Own all MANUS SDK and Wuji IK calls outside the body publisher."""
    import traceback

    source = None
    try:
        from deploy.manus import ManusIntegratedProvider

        source = ManusIntegratedProvider(**source_kwargs)
        hands = WujiRetargetPair(left_config, right_config)
        ready_event.set()
        period_s = 1.0 / float(poll_hz)
        last_error_s = 0.0
        while not stop_event.is_set():
            started_s = time.monotonic()
            try:
                left, left_valid, right, right_valid = source.latest_keypoints(
                    None, None
                )
                pair = HandPair(left, left_valid, right, right_valid)
                left_qpos, left_qpos_valid, right_qpos, right_qpos_valid = (
                    hands.retarget(pair)
                )
                now_ns = time.monotonic_ns()
                with shared_lock:
                    raw = np.frombuffer(raw_buffer, dtype=np.float32)
                    raw[:63] = np.asarray(left, dtype=np.float32).reshape(-1)
                    raw[63:] = np.asarray(right, dtype=np.float32).reshape(-1)
                    qpos = np.frombuffer(qpos_buffer, dtype=np.float32)
                    qpos[:20] = left_qpos
                    qpos[20:] = right_qpos
                    state = np.frombuffer(state_buffer, dtype=np.int8)
                    state[:] = (
                        bool(left_valid),
                        bool(right_valid),
                        bool(left_qpos_valid),
                        bool(right_qpos_valid),
                    )
                    updated_ns.value = now_ns
                    count_values = np.frombuffer(counters, dtype=np.int64)
                    count_values[:] = (hands.clipped, hands.invalid)
            except Exception as exc:
                now_s = time.monotonic()
                if now_s - last_error_s >= 2.0:
                    print(f"[ManusWorker] hand update failed: {exc}")
                    last_error_s = now_s
                with shared_lock:
                    np.frombuffer(state_buffer, dtype=np.int8)[:] = 0
                    updated_ns.value = time.monotonic_ns()
            remaining_s = period_s - (time.monotonic() - started_s)
            if remaining_s > 0.0:
                stop_event.wait(remaining_s)
    except BaseException:
        with suppress(Exception):
            error_conn.send(traceback.format_exc())
        ready_event.set()
    finally:
        if source is not None:
            with suppress(Exception):
                source.close()
        error_conn.close()


class _ManusHandWorker:
    """Spawn-isolated latest-only MANUS/Wuji cache for the body loop."""

    def __init__(
        self,
        *,
        source_kwargs: dict,
        left_config: str | Path,
        right_config: str | Path,
        stale_s: float,
        poll_hz: float = 90.0,
    ) -> None:
        if stale_s <= 0.0 or poll_hz <= 0.0:
            raise ValueError("MANUS worker stale_s and poll_hz must be positive")
        ctx = mp.get_context("spawn")
        self._stale_ns = int(float(stale_s) * 1e9)
        self._raw = ctx.Array("f", 126, lock=False)
        self._qpos = ctx.Array("f", 40, lock=False)
        self._state = ctx.Array("b", 4, lock=False)
        self._updated_ns = ctx.Value("q", 0, lock=False)
        self._counters = ctx.Array("q", 2, lock=False)
        self._lock = ctx.Lock()
        self._stop = ctx.Event()
        self._ready = ctx.Event()
        self._error_recv, error_send = ctx.Pipe(duplex=False)
        self._error_send = error_send
        self._process = ctx.Process(
            target=_run_manus_hand_worker,
            args=(
                source_kwargs,
                str(left_config),
                str(right_config),
                float(poll_hz),
                self._raw,
                self._qpos,
                self._state,
                self._updated_ns,
                self._counters,
                self._lock,
                self._stop,
                self._ready,
                error_send,
            ),
            name="hgpt-manus-hand",
            daemon=True,
        )
        self._started = False
        self._closed = False

    def start(self, timeout_s: float) -> None:
        timeout = float(timeout_s)
        if timeout <= 0.0:
            raise ValueError("MANUS worker startup timeout must be positive")
        self._process.start()
        self._error_send.close()
        self._started = True
        deadline = time.monotonic() + timeout
        while not self._ready.wait(timeout=0.05):
            if not self._process.is_alive():
                break
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError(
                    f"MANUS hand worker did not initialize within {timeout:.1f} s"
                )
        if self._error_recv.poll():
            error = self._error_recv.recv()
            self.close()
            raise RuntimeError("MANUS hand worker failed to initialize:\n" + error)
        if not self._process.is_alive():
            self.close()
            raise RuntimeError("MANUS hand worker exited during initialization")
        print(f"[ManusWorker] ready pid={self._process.pid}")

    def sample(self, now_ns: int | None = None) -> _HandWorkerSample:
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self._lock:
            raw = np.frombuffer(self._raw, dtype=np.float32).copy()
            qpos = np.frombuffer(self._qpos, dtype=np.float32).copy()
            state = np.frombuffer(self._state, dtype=np.int8).copy()
            updated = int(self._updated_ns.value)
        age_ns = now - updated if updated > 0 else self._stale_ns + 1
        fresh = 0 <= age_ns <= self._stale_ns and self._process.is_alive()
        return _HandWorkerSample(
            raw_left=raw[:63].reshape(21, 3),
            raw_right=raw[63:].reshape(21, 3),
            raw_left_valid=fresh and bool(state[0]),
            raw_right_valid=fresh and bool(state[1]),
            left_qpos=qpos[:20],
            right_qpos=qpos[20:],
            left_valid=fresh and bool(state[2]),
            right_valid=fresh and bool(state[3]),
            age_ms=age_ns / 1e6,
        )

    def counters(self) -> tuple[int, int]:
        with self._lock:
            values = np.frombuffer(self._counters, dtype=np.int64).copy()
        return int(values[0]), int(values[1])

    def close(self, timeout_s: float = 1.0) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._started:
            self._error_recv.close()
            self._error_send.close()
            return
        self._stop.set()
        self._process.join(timeout=max(0.0, float(timeout_s)))
        if self._process.is_alive():
            print("[ManusWorker] blocked during shutdown; terminating isolated worker")
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._error_recv.close()
        self._started = False


class _HandStatusMonitor:
    def __init__(self, endpoint: str):
        import zmq

        self._zmq = zmq
        self._socket = zmq.Context.instance().socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"hand_status")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(endpoint)
        self.last_received_s = time.monotonic()
        self._warned = False

    def poll(self) -> None:
        received = False
        while True:
            try:
                self._socket.recv(self._zmq.NOBLOCK)
                received = True
            except self._zmq.Again:
                break
        if received:
            self.last_received_s = time.monotonic()
            self._warned = False
        elif (
            self.last_received_s
            and time.monotonic() - self.last_received_s > 1.0
            and not self._warned
        ):
            print("[PoseManager] Wuji hand status unavailable; body stream continues")
            self._warned = True

    def close(self) -> None:
        self._socket.close(linger=0)


@dataclass
class PoseManagerArgs:
    bind: str = "tcp://*:5556"
    hand_source: str = "pico"  # pico | manus
    publish_wuji_hand: bool = False
    hand_status_endpoint: str = ""
    track_dir: str = "storage/test"
    freq: int = 50
    human_height: float = 1.7
    pico_service_mode: str = "auto"
    pico_startup_timeout_s: float = 15.0
    pico_wait_forever: bool = True
    left_retarget_config: str = ""
    right_retarget_config: str = ""
    manus_stale_s: float = 0.2
    manus_init_timeout_s: float = 10.0
    manus_calibration_file: str = ""
    manus_left_calibration_file: str = ""
    manus_right_calibration_file: str = ""
    manus_load_calibration: bool = True
    manus_debug_node_info: bool = False
    manus_mcp_joint: str = "proximal"
    max_steps: int = 0


class HgptPoseManager:
    def __init__(self, args: PoseManagerArgs):
        if args.hand_source not in {"pico", "manus"}:
            raise ValueError("--hand-source must be pico or manus")
        if args.freq <= 0 or args.max_steps < 0:
            raise ValueError("freq must be positive and max_steps non-negative")
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError(
                "pyzmq is required; run scripts/setup_pico_env.sh"
            ) from exc
        from general_motion_retargeting import GeneralMotionRetargeting as GMR

        self.args = args
        self._zmq = zmq
        self._socket = zmq.Context.instance().socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.bind(args.bind)
        self._source = PicoXrtSource(
            service_mode=args.pico_service_mode,
            startup_timeout_s=args.pico_startup_timeout_s,
        )
        self._source.start(wait_forever=args.pico_wait_forever)
        self._calibrator = PicoFrameCalibrator()
        self._gmr = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=args.human_height,
        )
        defaults = default_retarget_configs(args.hand_source)
        left_config = (
            Path(args.left_retarget_config)
            if args.left_retarget_config
            else defaults[0]
        )
        right_config = (
            Path(args.right_retarget_config)
            if args.right_retarget_config
            else defaults[1]
        )
        self._hand_source = None
        self._hands = None
        self._manus_worker = None
        if args.hand_source == "pico":
            self._hand_source = PicoHandSource(self._source.sdk)
            self._hands = WujiRetargetPair(left_config, right_config)
        else:
            self._manus_worker = _ManusHandWorker(
                source_kwargs=self._manus_source_kwargs(),
                left_config=left_config,
                right_config=right_config,
                stale_s=args.manus_stale_s,
            )
        self._offline = load_offline_directory(
            args.track_dir, default_qpos=np.asarray(DEFAULT_QPOS), target_hz=args.freq
        )
        self._keyboard = DeployKeyboardCMD(num_track_ref=len(self._offline))
        hand_status_endpoint = args.hand_status_endpoint.strip()
        if args.publish_wuji_hand and not hand_status_endpoint:
            hand_status_endpoint = DEFAULT_WUJI_HAND_STATUS_ENDPOINT
        self._status = (
            _HandStatusMonitor(hand_status_endpoint) if hand_status_endpoint else None
        )
        self._running = True
        self._frame_index = 0
        self._reset_generation = 0
        self._last_mode = 0
        self._last_body_source_ns = 0
        self._last_raw_source_ns = 0
        self._qpos = validate_g1_qpos(np.asarray(DEFAULT_QPOS))
        self._qpos_last: np.ndarray | None = None
        self._body_valid = False
        self._left = WUJI_OPEN_QPOS.copy()
        self._right = WUJI_OPEN_QPOS.copy()
        self._left_valid = False
        self._right_valid = False
        self._raw_body = np.zeros((24, 7), dtype=np.float32)
        self._raw_left = np.zeros((21, 3), dtype=np.float32)
        self._raw_right = np.zeros((21, 3), dtype=np.float32)
        self._raw_body_valid = False
        self._raw_left_valid = False
        self._raw_right_valid = False
        self._last_hand_age_ms = float("inf")
        self._offline_motion: OfflinePose | None = None
        self._offline_step = 0
        self._sequence_done = False
        self._last_report_s = time.monotonic()
        self._last_report_frame = self._frame_index
        self._mode_controls = PicoLeftModeControls()
        self._collector_controls = PicoRightCollectorControls()
        self._collector_control_sequence = 0
        if self._manus_worker is not None:
            self._manus_worker.start(args.manus_init_timeout_s + 10.0)

    def _manus_source_kwargs(self) -> dict:
        from deploy.manus import (
            DEFAULT_MANUS_CALIBRATION_FILE,
            DEFAULT_MANUS_LEFT_CALIBRATION_FILE,
            DEFAULT_MANUS_RIGHT_CALIBRATION_FILE,
        )

        args = self.args
        return {
            "stale_s": args.manus_stale_s,
            "init_timeout_s": args.manus_init_timeout_s,
            "calibration_file": args.manus_calibration_file
            or str(DEFAULT_MANUS_CALIBRATION_FILE),
            "left_calibration_file": args.manus_left_calibration_file
            or str(DEFAULT_MANUS_LEFT_CALIBRATION_FILE),
            "right_calibration_file": args.manus_right_calibration_file
            or str(DEFAULT_MANUS_RIGHT_CALIBRATION_FILE),
            "load_calibration": args.manus_load_calibration,
            "debug_node_info": args.manus_debug_node_info,
            "mcp_joint": args.manus_mcp_joint,
        }

    def stop(self, *_args) -> None:
        self._running = False

    def _reset_mode(self, mode: int) -> None:
        self._reset_generation += 1
        self._sequence_done = False
        self._last_report_s = time.monotonic()
        self._last_report_frame = self._frame_index
        self._body_valid = False
        self._left_valid = self._right_valid = False
        self._raw_body_valid = False
        self._raw_left_valid = self._raw_right_valid = False
        if mode == 1:
            self._calibrator.reset()
            self._qpos_last = None
            print("[PoseManager] Mode 1: waiting for fresh Pico calibration frame")
        elif mode >= 2:
            index = mode - 2
            self._offline_motion = (
                self._offline[index] if index < len(self._offline) else None
            )
            self._offline_step = 0
            if self._offline_motion is None:
                print(f"[PoseManager] Invalid offline mode {mode}")
            else:
                print(f"[PoseManager] Mode {mode}: {self._offline_motion.filename}")
        else:
            self._offline_motion = None
            print("[PoseManager] Mode 0: walk")

    def _update_live(self) -> None:
        try:
            frame = self._source.read()
        except Exception as exc:
            self._body_valid = False
            print(f"[PoseManager] Pico source frame rejected: {exc}")
            frame = None
        if frame is not None:
            received_ns = time.monotonic_ns()
            try:
                gmr_frame = pico_frame_to_gmr_frame(frame, self._calibrator)
                qpos = validate_g1_qpos(self._gmr.retarget(gmr_frame))
                self._qpos = ema_smooth_g1_qpos(self._qpos_last, qpos, alpha=0.75)
                self._qpos_last = self._qpos.copy()
                self._last_body_source_ns = received_ns
                self._last_raw_source_ns = frame.timestamp_ns
                self._raw_body = np.asarray(frame.body_poses, dtype=np.float32).copy()
                self._raw_body_valid = True
                self._body_valid = True
            except Exception as exc:
                self._body_valid = False
                print(f"[PoseManager] Pico/GMR frame rejected: {exc}")
        self._update_hands()

    def _update_hands(self) -> None:
        if self._manus_worker is not None:
            sample = self._manus_worker.sample()
            self._raw_left = sample.raw_left
            self._raw_right = sample.raw_right
            self._raw_left_valid = sample.raw_left_valid
            self._raw_right_valid = sample.raw_right_valid
            self._left = sample.left_qpos
            self._right = sample.right_qpos
            self._left_valid = sample.left_valid
            self._right_valid = sample.right_valid
            self._last_hand_age_ms = sample.age_ms
            return

        assert self._hand_source is not None
        assert self._hands is not None
        pair = self._hand_source.read() if hasattr(self._hand_source, "read") else None
        if pair is None:
            left, lv, right, rv = self._hand_source.latest_keypoints(None, None)
            pair = HandPair(left, lv, right, rv)
        self._raw_left = np.asarray(pair.left, dtype=np.float32).copy()
        self._raw_right = np.asarray(pair.right, dtype=np.float32).copy()
        self._raw_left_valid = bool(pair.left_valid)
        self._raw_right_valid = bool(pair.right_valid)
        self._left, self._left_valid, self._right, self._right_valid = (
            self._hands.retarget(pair)
        )
        self._last_hand_age_ms = 0.0

    def _update_offline(self) -> None:
        motion = self._offline_motion
        if motion is None or self._sequence_done:
            self._body_valid = False
            self._left_valid = self._right_valid = False
            return
        index = min(self._offline_step, len(motion) - 1)
        self._qpos = motion.qpos[index].copy()
        self._left = motion.left[index].copy()
        self._right = motion.right[index].copy()
        self._left_valid = bool(motion.left_valid[index])
        self._right_valid = bool(motion.right_valid[index])
        now_ns = time.monotonic_ns()
        self._last_body_source_ns = now_ns
        self._last_raw_source_ns = now_ns
        self._body_valid = True
        if index == len(motion) - 1:
            self._sequence_done = True
            self._left_valid = self._right_valid = False
            print(f"[PoseManager] Offline sequence complete: {motion.filename}")
        else:
            self._offline_step += 1

    def _snapshot(self, cmd, *, kill: bool = False) -> PoseSnapshot:
        hand_authorized = (
            self.args.publish_wuji_hand and cmd.mode >= 1 and self._body_valid
        )
        return PoseSnapshot(
            frame_index=self._frame_index,
            source_timestamp_ns=self._last_raw_source_ns,
            body_source_monotonic_ns=self._last_body_source_ns,
            mode=cmd.mode,
            reset_generation=self._reset_generation,
            cmd_vel=np.array(
                [cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw], dtype=np.float32
            ),
            g1_qpos=self._qpos,
            body_valid=self._body_valid and cmd.mode >= 1,
            left_wuji_qpos=self._left,
            right_wuji_qpos=self._right,
            left_wuji_qpos_valid=hand_authorized and self._left_valid,
            right_wuji_qpos_valid=hand_authorized and self._right_valid,
            sequence_done=self._sequence_done,
            kill=kill or cmd.kill,
        )

    def _raw_snapshot(self, *, publish_ns: int) -> TeleopRawSnapshot:
        return TeleopRawSnapshot(
            frame_index=self._frame_index,
            source_timestamp_ns=self._last_raw_source_ns,
            publish_monotonic_ns=publish_ns,
            body_poses=self._raw_body,
            body_valid=self._raw_body_valid and self._last_mode == 1,
            left_hand_keypoints=self._raw_left,
            right_hand_keypoints=self._raw_right,
            left_hand_valid=self._raw_left_valid and self._last_mode == 1,
            right_hand_valid=self._raw_right_valid and self._last_mode == 1,
            hand_source=1 if self.args.hand_source == "pico" else 2,
        )

    def _report_status(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_report_s
        if elapsed < 2.0:
            return
        hz = (self._frame_index - self._last_report_frame) / elapsed
        age_ms = (
            (time.monotonic_ns() - self._last_body_source_ns) / 1e6
            if self._last_body_source_ns
            else float("inf")
        )
        if self._manus_worker is not None:
            hand_clipped, hand_invalid = self._manus_worker.counters()
        else:
            assert self._hands is not None
            hand_clipped, hand_invalid = self._hands.clipped, self._hands.invalid
        print(
            f"[PoseManager] pose_hz={hz:.1f} body_age_ms={age_ms:.1f} "
            f"hand_valid=L{int(self._left_valid)}/R{int(self._right_valid)} "
            f"hand_age_ms={self._last_hand_age_ms:.1f} "
            f"hand_clipped={hand_clipped} hand_invalid={hand_invalid}"
        )
        self._last_report_s = now
        self._last_report_frame = self._frame_index

    @staticmethod
    def _safe_pico_button(sdk, getter_name: str) -> bool:
        try:
            getter = getattr(sdk, getter_name)
            return bool(getter())
        except Exception:
            return False

    def _apply_pico_mode_toggle(self, cmd) -> None:
        next_mode = self._mode_controls.update(
            left_axis_click=self._safe_pico_button(
                self._source.sdk, "get_left_axis_click"
            ),
            current_mode=cmd.mode,
        )
        if next_mode is None:
            return
        previous_mode = cmd.mode
        self._keyboard.set_mode(next_mode)
        cmd.mode = next_mode
        print(
            f"[PoseManager] Pico left stick: "
            f"Mode {previous_mode} -> Mode {next_mode}"
        )

    def _publish_collector_control(self) -> None:
        sdk = self._source.sdk
        decision = self._collector_controls.update(
            right_axis_click=self._safe_pico_button(sdk, "get_right_axis_click"),
            a_pressed=self._safe_pico_button(sdk, "get_A_button"),
            b_pressed=self._safe_pico_button(sdk, "get_B_button"),
        )
        if decision.ambiguous:
            print(
                "[PoseManager] WARNING: A and B pressed together; "
                "ignoring ambiguous collector save/discard"
            )
            return
        if decision.command is None:
            return
        self._collector_control_sequence += 1
        self._socket.send(
            CollectorControlSnapshot(
                sequence=self._collector_control_sequence,
                command=decision.command,
            ).pack()
        )
        command_name = COLLECTOR_CONTROL_COMMAND_NAMES[decision.command]
        print(
            f"[PoseManager] Sent collector_control: {command_name} "
            f"sequence={self._collector_control_sequence}"
        )

    def run(self) -> None:
        period = 1.0 / self.args.freq
        print(f"[PoseManager] Publishing {self.args.bind} at {self.args.freq} Hz")
        print(
            f"[PoseManager] Hand source={self.args.hand_source}; real hand publish={self.args.publish_wuji_hand}"
        )
        print(
            "[PoseManager] Pico controllers: left stick-click=toggle Mode 0/1; "
            "right stick-click=start, A=save, B=discard collector"
        )
        step = 0
        try:
            while self._running and not (
                self.args.max_steps and step >= self.args.max_steps
            ):
                started = time.monotonic()
                cmd = self._keyboard.step_command()
                self._apply_pico_mode_toggle(cmd)
                self._publish_collector_control()
                if cmd.kill:
                    self._running = False
                if cmd.mode != self._last_mode:
                    self._reset_mode(cmd.mode)
                if self._keyboard.check_reset_request():
                    self._reset_mode(cmd.mode)
                if cmd.mode == 1:
                    self._update_live()
                elif cmd.mode >= 2:
                    self._update_offline()
                else:
                    try:
                        self._source.read()
                    except Exception as exc:
                        print(f"[PoseManager] Pico source frame rejected: {exc}")
                    self._body_valid = False
                    self._left_valid = self._right_valid = False
                    self._raw_body_valid = False
                    self._raw_left_valid = self._raw_right_valid = False
                if self._status is not None:
                    self._status.poll()
                publish_ns = time.monotonic_ns()
                self._last_mode = cmd.mode
                self._socket.send(self._snapshot(cmd).pack())
                self._socket.send(self._raw_snapshot(publish_ns=publish_ns).pack())
                self._frame_index += 1
                self._report_status()
                self._last_mode = cmd.mode
                step += 1
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self.close()

    def close(self) -> None:
        from deploy.keyboard_cmd import HighCommand

        cmd = HighCommand(mode=self._last_mode)
        for _ in range(5):
            self._socket.send(self._snapshot(cmd, kill=True).pack())
            self._frame_index += 1
            time.sleep(0.02)
        self._socket.send(pack_stop_command())
        self._keyboard.close()
        if self._manus_worker is not None:
            self._manus_worker.close()
        elif self._hand_source is not None:
            self._hand_source.close()
        self._source.close()
        if self._status is not None:
            self._status.close()
        self._socket.close(linger=100)


def main(args: PoseManagerArgs) -> None:
    manager = HgptPoseManager(args)
    signal.signal(signal.SIGINT, manager.stop)
    signal.signal(signal.SIGTERM, manager.stop)
    manager.run()


if __name__ == "__main__":
    main(tyro.cli(PoseManagerArgs))
