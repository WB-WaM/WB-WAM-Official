"""Online motion-capture retarget subprocess.

Launches a background process that reads from OptiTrack, PNLink, or
Xsens MVN, runs GMR (General Motion Retargeting), and writes the
latest qpos_full into shared memory for the main deploy loop to
consume.  Source selection is controlled by the ``mocap_type``
argument (see :class:`MocapType`).
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import threading
import time
from collections import deque
from enum import IntEnum
from multiprocessing.sharedctypes import SynchronizedArray

import numpy as np


class MocapType(IntEnum):
    OPTITRACK = 1
    PNLINK = 2
    XSENS = 3
    PICO = 4


# Hand detection joint names (for PNLink / OptiTrack)
_HAND_JOINTS = {
    "left": {"wrist": "LeftHand"},
    "right": {"wrist": "RightHand"},
}
_HAND_THRESHOLD = 0.05

# Keep IPC primitives/process handles alive for spawn context.
# If these objects are garbage-collected too early, spawned children may fail
# rebuilding SemLock with FileNotFoundError.
_RETARGET_SESSIONS: list[dict] = []


def validate_g1_qpos(qpos: np.ndarray) -> np.ndarray:
    """Return a normalized ``float32`` G1 qpos or raise ``ValueError``."""
    result = np.asarray(qpos, dtype=np.float32)
    if result.shape != (36,):
        raise ValueError(f"G1 qpos must have shape (36,), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("G1 qpos contains NaN or Inf")
    quat_norm = float(np.linalg.norm(result[3:7]))
    if quat_norm < 1e-6:
        raise ValueError("G1 root quaternion has near-zero norm")
    result = result.copy()
    result[3:7] /= quat_norm
    return result


def ema_smooth_g1_qpos(
    previous: np.ndarray | None,
    current: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    """EMA a G1 qpos while preserving quaternion sign and unit length."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"EMA alpha must be in [0, 1), got {alpha}")
    current = validate_g1_qpos(current)
    if previous is None:
        return current

    previous = validate_g1_qpos(previous)
    result = previous * alpha + current * (1.0 - alpha)
    previous_quat = previous[3:7]
    current_quat = current[3:7]
    if float(np.dot(previous_quat, current_quat)) < 0.0:
        current_quat = -current_quat
    quat = previous_quat * alpha + current_quat * (1.0 - alpha)
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm < 1e-6:
        raise ValueError("EMA produced a near-zero root quaternion")
    result[3:7] = quat / quat_norm
    return validate_g1_qpos(result)


def _read_calibration_generation(calibration_generation) -> int:
    """Read the current retarget calibration generation atomically."""
    with calibration_generation.get_lock():
        return int(calibration_generation.value)


def _invalidate_calibration_generation(calibration_generation, ts) -> int:
    """Advance calibration generation and invalidate its published timestamp.

    Holding the generation lock until the timestamp is zero prevents an
    in-flight publisher from committing an old-generation frame afterward.
    """
    with calibration_generation.get_lock():
        calibration_generation.value += 1
        generation = int(calibration_generation.value)
        with ts.get_lock():
            ts.value = 0.0
    return generation


def _publish_retarget_output_if_current(
    *,
    calibration_generation,
    frame_generation: int,
    buf,
    buf_hand,
    ts,
    qpos: np.ndarray,
    hand_data: np.ndarray,
    frame_received_at_s: float | None = None,
) -> bool:
    """Publish a current-generation result with its source receive time.

    The generation lock covers both the comparison and all shared-buffer
    writes. A parent calibration request therefore either happens before
    this commit (and rejects it) or after it (and immediately zeros the
    published timestamp). The timestamp uses the host monotonic clock and is
    sampled when the source frame is received, so GMR and queue delay count
    toward the consumer freshness age.
    """
    qpos = validate_g1_qpos(qpos)
    hand = np.asarray(hand_data, dtype=np.float32)
    if hand.shape != (4,) or not np.isfinite(hand).all():
        raise ValueError("Retarget hand data must be a finite shape-(4,) array")
    received_at_s = (
        time.monotonic()
        if frame_received_at_s is None
        else float(frame_received_at_s)
    )
    if not np.isfinite(received_at_s) or received_at_s <= 0.0:
        raise ValueError("frame receive monotonic timestamp must be finite and positive")

    with calibration_generation.get_lock():
        if int(calibration_generation.value) != frame_generation:
            return False
        with buf_hand.get_lock():
            np.frombuffer(buf_hand.get_obj(), dtype=np.float32)[:] = hand
        with buf.get_lock(), ts.get_lock():
            target = np.frombuffer(buf.get_obj(), dtype=np.float32)
            if target.shape != qpos.shape:
                raise ValueError(
                    f"Shared qpos buffer has shape {target.shape}, expected {qpos.shape}"
                )
            target[:] = qpos
            ts.value = received_at_s
    return True


def _detect_hand_open(frame, wrist: str, threshold: float = _HAND_THRESHOLD):
    """Detect hand open/close from finger-to-thumb distance."""
    try:
        if wrist == "RightHand":
            dist = np.linalg.norm(
                np.array(frame["RightHandIndex3"][0])
                - np.array(frame["RightHandThumb3"][0])
            )
        else:
            dist = np.linalg.norm(
                np.array(frame["LeftHandIndex3"][0])
                - np.array(frame["LeftHandThumb3"][0])
            )
        return dist > threshold, dist
    except KeyError:
        return False, 0.0


def _retarget_worker(
    buf,
    buf_hand,
    ts,
    ready_evt,
    stop_evt,
    calibration_generation,
    server_ip,
    client_ip,
    robot,
    use_multicast,
    actual_human_height,
    mocap_type,
    buffer_ms,
    rt_pin,
    xsens_host="0.0.0.0",
    xsens_port=9763,
    xsens_protocol="tcp",
    pico_service_mode="auto",
    pico_startup_timeout_s=15.0,
):
    """Worker process: mocap -> GMR retarget -> shared memory.

    ``rt_pin``: optional ``(cpu_id, fifo_priority)`` tuple.  When set, pin
    this process to ``cpu_id`` and run it under ``SCHED_FIFO`` at
    ``fifo_priority``.  Intended for resource-constrained on-board targets
    (e.g. Jetson via ``deploy.onboard_deploy.play_track_onboard``) where
    isolating GMR on a dedicated core measurably reduces mocap jitter.
    On general-purpose workstations leave this ``None`` — pinning a single
    core there would only contend with viewer / camera / IDE threads.
    """
    if rt_pin is not None:
        import os

        cpu_id, fifo_prio = rt_pin
        try:
            os.sched_setaffinity(0, {int(cpu_id)})
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(fifo_prio)))
        except (OSError, PermissionError):
            pass

    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    pico_calibrator = None
    client = None
    if mocap_type == MocapType.OPTITRACK:
        from general_motion_retargeting.optitrack_vendor.NatNetClient import (
            setup_optitrack,
        )

        client = setup_optitrack(
            server_address=server_ip,
            client_address=client_ip,
            use_multicast=use_multicast,
        )
        if not client:
            return
        threading.Thread(target=client.run, daemon=True).start()
        get_frame = client.get_frame_upgraded
        src_human = "fbx"
    elif mocap_type == MocapType.PNLINK:
        from noitom import NoitomClient

        client = NoitomClient()
        client.start_thread()

        def get_frame():
            return client.get_frame_data(timeout=True)

        src_human = "fbx_noitom"
    elif mocap_type == MocapType.XSENS:
        from deploy.xsens.client import XsensClient

        client = XsensClient(
            host=xsens_host,
            port=xsens_port,
            protocol=xsens_protocol,
        )
        client.start_thread()

        def get_frame():
            return client.get_frame_data(timeout=0.5)

        src_human = "fbx_xsens"
    elif mocap_type == MocapType.PICO:
        from deploy.pico import (
            PicoFrameCalibrator,
            PicoXrtSource,
            pico_frame_to_gmr_frame,
        )

        client = PicoXrtSource(
            service_mode=pico_service_mode,
            startup_timeout_s=pico_startup_timeout_s,
        )
        get_frame = client.read
        pico_calibrator = PicoFrameCalibrator()
        src_human = "xrobot"
    else:
        raise ValueError(f"Unknown mocap_type: {mocap_type}")

    retarget = GMR(
        src_human=src_human, tgt_robot=robot, actual_human_height=actual_human_height
    )
    if mocap_type == MocapType.PICO:
        client.start()

    qpos_last = None
    ema_alpha = 0.75
    active_generation = _read_calibration_generation(calibration_generation)

    # -- Jitter buffer: absorb Noitom delivery timing jitter --
    _use_jbuf = buffer_ms > 0
    if _use_jbuf:
        _nominal_hz = 90.0
        _target_depth = max(1, round(buffer_ms / 1000.0 * _nominal_hz))
        _jbuf: deque[tuple[int, np.ndarray, np.ndarray, float]] = deque()
        _jbuf_lock = threading.Lock()
        _jbuf_filled = threading.Event()

        def _jitter_output():
            dt_out = 1.0 / _nominal_hz
            _out_generation = active_generation
            _out_qpos = None
            _out_hand = np.zeros(4, dtype=np.float32)
            _out_received_at_s = 0.0
            while not stop_evt.is_set():
                if not _jbuf_filled.wait(timeout=0.05):
                    continue
                popped = False
                with _jbuf_lock:
                    if _jbuf:
                        (
                            _out_generation,
                            _out_qpos,
                            _out_hand,
                            _out_received_at_s,
                        ) = _jbuf.popleft()
                        depth = len(_jbuf)
                        popped = True
                    else:
                        _jbuf_filled.clear()
                        depth = 0
                if popped and _out_qpos is not None:
                    published = _publish_retarget_output_if_current(
                        calibration_generation=calibration_generation,
                        frame_generation=_out_generation,
                        buf=buf,
                        buf_hand=buf_hand,
                        ts=ts,
                        qpos=_out_qpos,
                        hand_data=_out_hand,
                        frame_received_at_s=_out_received_at_s,
                    )
                    if published and not ready_evt.is_set():
                        ready_evt.set()
                depth_err = depth - _target_depth
                dt_out = (1.0 / _nominal_hz) * (1.0 - 0.02 * depth_err)
                dt_out = max(0.005, min(0.030, dt_out))
                time.sleep(dt_out)

        threading.Thread(target=_jitter_output, daemon=True).start()
        print(
            f"[Retarget] Jitter buffer enabled: {buffer_ms:.0f} ms ({_target_depth} frames)"
        )

    try:
        while not stop_evt.is_set():
            observed_generation = _read_calibration_generation(calibration_generation)
            if observed_generation != active_generation:
                if mocap_type == MocapType.PICO:
                    assert pico_calibrator is not None
                    pico_calibrator.reset()
                qpos_last = None
                active_generation = observed_generation
                if _use_jbuf:
                    with _jbuf_lock:
                        _jbuf.clear()
                        _jbuf_filled.clear()

            frame_generation = active_generation

            try:
                frame = get_frame()
                frame_received_at_s = time.monotonic()
            except Exception as exc:
                print(f"[Retarget] source frame rejected: {exc}")
                if mocap_type == MocapType.PICO:
                    time.sleep(0.001)
                continue
            if frame is None:
                if mocap_type == MocapType.PICO:
                    time.sleep(0.001)
                continue

            # Hand detection
            if mocap_type == MocapType.PICO:
                hand_data = np.zeros(4, dtype=np.float32)
                try:
                    assert pico_calibrator is not None
                    frame = pico_frame_to_gmr_frame(frame, pico_calibrator)
                except Exception as exc:
                    print(f"[Retarget] Pico conversion rejected: {exc}")
                    continue
            else:
                l_open, l_dist = _detect_hand_open(frame, **_HAND_JOINTS["left"])
                r_open, r_dist = _detect_hand_open(frame, **_HAND_JOINTS["right"])
                hand_data = np.array(
                    [float(l_open), l_dist, float(r_open), r_dist],
                    dtype=np.float32,
                )

            # Retarget
            try:
                qpos = validate_g1_qpos(retarget.retarget(frame))
                qpos = ema_smooth_g1_qpos(qpos_last, qpos, alpha=ema_alpha)
            except Exception as e:
                import traceback

                print(f"[Retarget] error: {e}\n{traceback.format_exc()}")
                continue

            qpos_last = qpos.copy()

            if _use_jbuf:
                with _jbuf_lock:
                    _jbuf.append(
                        (
                            frame_generation,
                            qpos.copy(),
                            hand_data,
                            frame_received_at_s,
                        )
                    )
                    if not _jbuf_filled.is_set() and len(_jbuf) >= _target_depth:
                        _jbuf_filled.set()
                    while len(_jbuf) > _target_depth * 3:
                        _jbuf.popleft()
            else:
                published = _publish_retarget_output_if_current(
                    calibration_generation=calibration_generation,
                    frame_generation=frame_generation,
                    buf=buf,
                    buf_hand=buf_hand,
                    ts=ts,
                    qpos=qpos,
                    hand_data=hand_data,
                    frame_received_at_s=frame_received_at_s,
                )
                if published and not ready_evt.is_set():
                    ready_evt.set()
    finally:
        if mocap_type == MocapType.PICO and client is not None:
            client.close()
        elif mocap_type in (MocapType.PNLINK, MocapType.XSENS) and hasattr(
            client, "stop"
        ):
            client.stop()


def _visualize_worker(buf, stop_evt, robot="unitree_g1"):
    from general_motion_retargeting import RobotMotionViewer

    viewer = RobotMotionViewer(robot_type=robot, motion_fps=120.0)
    while not stop_evt.is_set():
        with buf.get_lock():
            qpos = np.frombuffer(buf.get_obj(), dtype=np.float32).copy()
        viewer.step(root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:])


def start_realtime_retarget(
    server_ip: str,
    client_ip: str,
    robot: str = "unitree_g1",
    use_multicast: bool = False,
    dof_full: int = 36,
    actual_human_height: float = 1.6,
    visualize_retarget: bool = False,
    mocap_type: MocapType = MocapType.PNLINK,
    buffer_ms: float = 0.0,
    rt_pin: tuple[int, int] | None = None,
    pico_service_mode: str = "auto",
    pico_startup_timeout_s: float = 15.0,
    xsens_host: str = "0.0.0.0",
    xsens_port: int = 9763,
    xsens_protocol: str = "tcp",
) -> tuple[SynchronizedArray, ...]:
    """Launch retarget worker and return shared buffers.

    Args:
        rt_pin: Optional ``(cpu_id, fifo_priority)`` for the GMR subprocess.
            When set, the worker pins itself to ``cpu_id`` and runs under
            ``SCHED_FIFO`` at ``fifo_priority``.  Use only on resource-
            constrained on-board targets (e.g. Jetson); leave ``None`` for
            workstation runs (``deploy/play_track.py``, ``collect_data``,
            etc.) to avoid contending with viewer / camera / IDE threads.

    Returns:
        (buf_qpos, ts, buf_hand)
        - buf_qpos: Array('f', dof_full) – latest retargeted full qpos
        - ts: Value('d') – host monotonic time when the source frame was received
        - buf_hand: Array('f', 4) – [left_open, left_dist, right_open, right_dist]
    """
    # Use "spawn" to avoid inheriting X11/GL state from parent process.
    # This prevents xcb thread-sequence crashes when multiple GUI viewers run.
    if mocap_type == MocapType.PICO and dof_full != 36:
        raise ValueError("Pico -> unitree_g1 requires dof_full=36")
    ctx = mp.get_context("spawn")
    buf = ctx.Array("f", dof_full, lock=True)
    with buf.get_lock():
        np.frombuffer(buf.get_obj(), dtype=np.float32)[3] = 1.0
    buf_hand = ctx.Array("f", 4, lock=True)
    ts = ctx.Value("d", 0.0)
    ready_evt = ctx.Event()
    stop_evt = ctx.Event()
    calibration_generation = ctx.Value("q", 0)

    p = ctx.Process(
        target=_retarget_worker,
        args=(
            buf,
            buf_hand,
            ts,
            ready_evt,
            stop_evt,
            calibration_generation,
            server_ip,
            client_ip,
            robot,
            use_multicast,
            actual_human_height,
            mocap_type,
            buffer_ms,
            rt_pin,
            xsens_host,
            xsens_port,
            xsens_protocol,
            pico_service_mode,
            pico_startup_timeout_s,
        ),
        daemon=True,
    )
    p.start()

    vis_p = None
    if visualize_retarget:
        vis_p = ctx.Process(target=_visualize_worker, args=(buf, stop_evt), daemon=True)
        vis_p.start()

    # Persist references so spawn children can always rebuild synchronization
    # primitives from valid OS handles.
    _RETARGET_SESSIONS.append(
        {
            "proc": p,
            "vis_proc": vis_p,
            "ready_evt": ready_evt,
            "stop_evt": stop_evt,
            "calibration_generation": calibration_generation,
            "mocap_type": mocap_type,
            "buf": buf,
            "buf_hand": buf_hand,
            "ts": ts,
        }
    )
    atexit.register(stop_realtime_retarget, buf)

    return buf, ts, buf_hand


def _session_for_buffer(buf):
    return next(
        (session for session in _RETARGET_SESSIONS if session["buf"] is buf),
        None,
    )


def wait_realtime_retarget_ready(
    buf,
    *,
    timeout_s: float,
    poll_s: float = 0.05,
) -> None:
    """Wait for the first published retarget frame or fail on worker exit.

    ``start_realtime_retarget`` is intentionally asynchronous. Real-robot
    startup must use this helper so a Pico source timeout cannot leave the
    parent process running with a permanently empty mocap buffer.
    """
    timeout = float(timeout_s)
    poll = float(poll_s)
    if not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("retarget ready timeout_s must be finite and positive")
    if not np.isfinite(poll) or poll <= 0.0:
        raise ValueError("retarget ready poll_s must be finite and positive")

    session = _session_for_buffer(buf)
    if session is None:
        raise RuntimeError("retarget session is not registered")
    process = session["proc"]
    ready_evt = session["ready_evt"]
    deadline = time.monotonic() + timeout

    while True:
        if ready_evt.is_set():
            return
        if not process.is_alive():
            raise RuntimeError(
                "retarget subprocess exited before its first valid frame "
                f"(exitcode={process.exitcode})"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                "Timed out waiting for the first valid retarget frame after "
                f"{timeout:.1f} s"
            )
        ready_evt.wait(timeout=min(poll, remaining))


def request_retarget_calibration(buf) -> bool:
    """Request fresh Pico XY/yaw/ground calibration and invalidate old qpos."""
    session = _session_for_buffer(buf)
    if session is None or session["mocap_type"] != MocapType.PICO:
        return False
    _invalidate_calibration_generation(
        session["calibration_generation"],
        session["ts"],
    )
    return True


def stop_realtime_retarget(buf, join_timeout_s: float = 2.0) -> bool:
    """Stop a retarget session, allowing owned Pico services to close."""
    session = _session_for_buffer(buf)
    if session is None:
        return False

    session["stop_evt"].set()
    proc = session["proc"]
    proc.join(timeout=join_timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=join_timeout_s)

    vis_proc = session["vis_proc"]
    if vis_proc is not None:
        vis_proc.join(timeout=0.2)
        if vis_proc.is_alive():
            vis_proc.terminate()
            vis_proc.join(timeout=join_timeout_s)

    _RETARGET_SESSIONS.remove(session)
    return True


def read_mocap_buffer(buf, ts) -> tuple[np.ndarray, float]:
    """Read the latest qpos_full and timestamp from shared memory."""
    with buf.get_lock(), ts.get_lock():
        qpos_full = np.frombuffer(buf.get_obj(), dtype=np.float32).copy()
        timestamp = ts.value
    if np.all(qpos_full == 0):
        qpos_full[3] = 1.0
    return qpos_full, timestamp


def read_hand_buffer(buf_hand) -> tuple[bool, float, bool, float] | None:
    """Read hand open/close state from shared memory."""
    if buf_hand is None:
        return None
    with buf_hand.get_lock():
        data = np.frombuffer(buf_hand.get_obj(), dtype=np.float32).copy()
    return bool(data[0]), data[1], bool(data[2]), data[3]
