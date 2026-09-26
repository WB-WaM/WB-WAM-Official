"""4090-host side of the cable-free onboard deploy.

This process runs on the workstation that sits next to the human operator:

    Noitom PNLink (WiFi)   OR   Xsens MVN (TCP/UDP, port 9763)
            │                            │
            ▼                            ▼
    NoitomClient                    XsensClient    (one or the other,
            │                            │         selected by --mocap-type)
            └─── reused via deploy.retarget subprocess ──────────┐
                                                                 │
            GeneralMotionRetargeting (GMR, IK) + EMA             │
                                 │                               │
                                 ▼                               │
            Shared-memory ring (qpos_full + hand)  <─────────────┘
                                 │
                                 ▼
            UDP send (this main loop) ── WiFi (same SSID as G1) ──►  Robot Jetson

We deliberately re-use ``deploy.retarget.start_realtime_retarget`` instead of
re-implementing the mocap+GMR pipeline, so the host stream stays byte-
compatible with the existing onboard mode-1 deployment regardless of which
mocap source is active.  The only addition is that the latest qpos is
broadcast over UDP to the robot.

Usage::

    # Noitom Axis Studio:
    python -m deploy.onboard_deploy_wo_GMR.host_sender \
        --robot-ip 192.168.1.42 \
        --server-ip 192.168.1.100

    # Xsens MVN (Network Streamer; default port 9763, TCP):
    python -m deploy.onboard_deploy_wo_GMR.host_sender \
        --robot-ip 192.168.1.42 \
        --mocap-type xsens \
        --xsens-protocol tcp --xsens-port 9763

Stop with Ctrl+C; the retarget subprocess is terminated on exit.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import tyro

import deploy.retarget as retarget_module
from deploy.retarget import (
    MocapType,
    read_hand_buffer,
    read_mocap_buffer,
    start_realtime_retarget,
)

from .protocol import (
    BRAINCO_QPOS_FLOATS,
    DEFAULT_PORT,
    G1_DOF_FULL,
    HAND_FLOATS,
    encode_frame,
    packet_size,
)


@dataclass
class HostSenderArgs:
    """Run on the 4090 workstation; ships retargeted frames to the G1 over WiFi."""

    # ---- UDP ---------------------------------------------------------------
    robot_ip: str = "192.168.1.42"
    """G1 Jetson WiFi address (the robot's wlan0 IP, same subnet as the host)."""
    robot_port: int = DEFAULT_PORT
    """UDP port the robot listens on (see protocol.DEFAULT_PORT)."""

    # ---- Mocap -------------------------------------------------------------
    mocap_type: str = "pnlink"
    """'pnlink' (Noitom Axis Studio), 'optitrack', or 'xsens' (Xsens MVN
    Network Streamer)."""
    server_ip: str = "192.168.1.100"
    """Noitom/OptiTrack server IP, reachable from the host.  Ignored when
    --mocap-type xsens (Xsens MVN connects TO us)."""
    client_ip: str = ""
    """Local IP for OptiTrack multicast; leave empty for PNLink/Xsens."""
    human_height: float = 1.7

    # ---- Xsens MVN (only used when --mocap-type xsens) ---------------------
    xsens_host: str = "0.0.0.0"
    """Local bind address for the Xsens MVN MXTP02 listener."""
    xsens_port: int = 9763
    """Local port the Xsens MVN Network Streamer connects to (its default
    is 9763).  Must match the port configured in MVN Studio."""
    xsens_protocol: str = "tcp"
    """'tcp' or 'udp' -- must match the MVN Studio Network Streamer setting.
    The Xsens branch always uses the Xsens-tuned GMR IK config
    (``fbx_xsens``).  Generate it via ``python -m deploy.xsens.make_ik_config``
    if it is missing."""

    # ---- Pipeline ----------------------------------------------------------
    buffer_ms: float = 0.0
    """Host-side jitter buffer for the GMR subprocess.  Keep at 0 here so we
    forward each retargeted frame as soon as it lands; the *robot* applies
    its own jitter buffer over the network."""
    no_hand: bool = False
    """If set, do not include the hand state in the wire packet (smaller
    packets, but the robot side cannot drive Dex3 hands from this stream).
    Has no effect on the BrainCo qpos field; that is controlled by
    ``--enable-brainco-hand``."""
    send_hz: float = 0.0
    """Send pacing.  ``0`` means ``send-on-update`` (one packet per new
    retargeted frame, ~90 Hz with Noitom).  Set to a positive number to
    rate-limit (e.g. 60.0) when the network is congested."""

    # ---- BrainCo dex-hand --------------------------------------------------
    enable_brainco_hand: bool = False
    """Switch to the BrainCo-aware GMR subprocess
    (``deploy.brainco.play_track_brainco.start_realtime_retarget_with_brainco_hands``)
    and append a 24-D BrainCo hand qpos to each packet.  Requires the
    ``deploy.brainco`` stack and a GMR build that supports ``--hand-target``.
    """
    hand_target: str = "brainco2"
    """GMR hand-retarget target.  One of ``brainco``, ``brainco2``, ``brainco3``."""
    visualize_retarget: bool = False
    """When True, also spawn a mujoco viewer subprocess for the retargeted
    body -- useful for local debugging on the workstation (verify the
    GMR output looks right before the robot moves).  Works for both the
    Dex3 (and Xsens) and BrainCo paths.  Requires a display server, so
    don't enable it over plain SSH or on headless setups."""

    # ---- Diagnostics -------------------------------------------------------
    log_every_sec: float = 2.0
    """Stats print interval."""
    startup_timeout_sec: float = 30.0
    """Maximum time to wait for the first retargeted frame from GMR."""


def _iter_retarget_session_lists():
    """Yield every ``_RETARGET_SESSIONS`` list that has been registered
    by an already-loaded retarget module.

    IMPORTANT: this function must NEVER trigger an ``import`` of a
    new module.  It is called from the SIGINT handler, and importing
    inside a signal handler will run arbitrary module-level code while
    the GIL is held in a signal-context state, which on this codebase
    pulls in ``mujoco.mjx``, ``warp``, ``pygame``, etc. (several
    seconds of work) -- the visible symptom of which is "Ctrl+C does
    nothing".  We only look at ``sys.modules`` for retarget modules
    that have already been loaded by the normal startup path.
    """
    lists = [retarget_module._RETARGET_SESSIONS]
    ptb = sys.modules.get("deploy.brainco.play_track_brainco")
    if ptb is not None:
        ptb_sessions = getattr(ptb, "_RETARGET_SESSIONS", None)
        if ptb_sessions is not None:
            lists.append(ptb_sessions)
    return lists


_shutdown_requested = threading.Event()
_ctrl_c_count = 0


def _install_signal_handlers() -> None:
    """Make Ctrl+C / SIGTERM responsive even when blocked inside an
    interprocess lock or socket call.

    The handler does ONLY pure-memory operations -- setting a
    ``threading.Event`` and a few ``mp.Event``s.  Any I/O, logging,
    or module import would risk re-entering the GIL in a signal
    context and could deadlock or take seconds to complete (which is
    exactly what "Ctrl+C does nothing" looks like to the user).

    * First signal: set a shutdown flag and tell already-spawned GMR
      workers to drain via their ``stop_evt``.  The main loop sees
      the flag on its next iteration and exits cleanly through its
      ``finally`` (where logging is safe again).
    * Second signal: hard-exit immediately with ``os._exit(1)`` so the
      user is never trapped if a child process actually dead-locks.
    """

    def _handler(signum, _frame):  # noqa: ARG001
        global _ctrl_c_count
        _ctrl_c_count += 1
        if _ctrl_c_count >= 2:
            os._exit(1)
        _shutdown_requested.set()
        # Poke every already-registered stop_evt so workers leave their
        # poll loops at the next 0.5 s tick instead of waiting for the
        # parent's join+terminate fallback.  ``_iter_retarget_session_lists``
        # is import-free by construction (see its docstring).
        for sessions in _iter_retarget_session_lists():
            for sess in sessions:
                stop_evt = sess.get("stop_evt")
                if stop_evt is not None:
                    stop_evt.set()

    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):
        # SIGTERM may not be installable in some sandboxes / threads.
        pass


def _resolve_mocap_type(name: str) -> MocapType:
    """Map the ``--mocap-type`` string to a :class:`MocapType` enum value.
    Mirrors ``deploy.play_track._resolve_mocap_type`` so the two entry
    points accept exactly the same flag values."""
    key = (name or "").lower()
    if key == "pnlink":
        return MocapType.PNLINK
    if key == "xsens":
        return MocapType.XSENS
    return MocapType.OPTITRACK


def _wait_first_frame(buf_mocap, ts_mocap, timeout: float) -> None:
    print(f"[host_sender] Waiting for first retargeted frame (timeout={timeout:.1f}s)...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if _shutdown_requested.is_set():
            raise KeyboardInterrupt
        _, ts = read_mocap_buffer(buf_mocap, ts_mocap)
        if ts > 0.0:
            print(f"[host_sender]   First frame after {time.perf_counter() - t0:.2f}s")
            return
        time.sleep(0.01)
    raise TimeoutError(
        "No retargeted frame arrived from the GMR subprocess. "
        "Check Noitom Axis/PNLink/Xsens connection and the worker stderr above."
    )


def _stop_retarget_sessions() -> None:
    """Stop GMR subprocesses launched from either retarget module.

    We register the same _RETARGET_SESSIONS list for both
    ``deploy.retarget`` (Dex3 path) and
    ``deploy.brainco.play_track_brainco`` (BrainCo path); each module owns
    its own list so we drain both.
    """
    for sessions in _iter_retarget_session_lists():
        for sess in sessions:
            stop_evt = sess.get("stop_evt")
            if stop_evt is not None:
                stop_evt.set()
        for sess in sessions:
            for key in ("proc", "vis_proc"):
                proc = sess.get(key)
                if proc is None:
                    continue
                # Shorter than the previous (2 s + 1 s).  Workers poll
                # stop_evt every 0.5 s, so 1.0 s join is plenty for the
                # graceful path; if they still hang, SIGTERM then 0.5 s.
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)


def _send_loop(
    args: HostSenderArgs,
    sock: socket.socket,
    dest: tuple[str, int],
    buf_mocap,
    ts_mocap,
    buf_hand,
    buf_brainco_qpos,
) -> None:
    has_hand = (not args.no_hand) and (buf_hand is not None)
    has_brainco = buf_brainco_qpos is not None
    pkt_size = packet_size(G1_DOF_FULL, has_hand=has_hand, has_brainco=has_brainco)
    print(
        f"[host_sender] Streaming to {dest[0]}:{dest[1]}  "
        f"(packet={pkt_size}B, has_hand={has_hand}, has_brainco={has_brainco})"
    )

    pace_dt = 1.0 / args.send_hz if args.send_hz > 0 else 0.0
    next_send_t = time.perf_counter()

    seq = 0
    sent = 0
    dup = 0          # poll found no new frame -> skipped
    last_ts = 0.0
    bytes_sent = 0
    err_send = 0

    win_t0 = time.perf_counter()
    win_sent0 = 0
    win_lag_acc = 0.0
    win_lag_n = 0
    win_lag_max = 0.0

    while not _shutdown_requested.is_set():
        # 1) Read latest retargeted frame from the GMR subprocess.
        qpos_full, mocap_ts = read_mocap_buffer(buf_mocap, ts_mocap)

        hand = None
        if has_hand:
            h = read_hand_buffer(buf_hand)
            if h is not None:
                hand = np.array(
                    [float(h[0]), float(h[1]), float(h[2]), float(h[3])],
                    dtype=np.float32,
                )
            else:
                hand = np.zeros(HAND_FLOATS, dtype=np.float32)

        brainco_qpos = None
        if has_brainco:
            with buf_brainco_qpos.get_lock():
                bq = np.frombuffer(
                    buf_brainco_qpos.get_obj(), dtype=np.float32,
                    count=BRAINCO_QPOS_FLOATS,
                ).copy()
            brainco_qpos = bq

        is_new = mocap_ts > 0.0 and mocap_ts != last_ts
        if is_new:
            last_ts = mocap_ts
            payload = encode_frame(
                seq, time.time(), qpos_full, hand=hand, brainco_qpos=brainco_qpos,
            )
            try:
                sock.sendto(payload, dest)
            except OSError as e:
                err_send += 1
                if err_send <= 5 or err_send % 100 == 0:
                    print(f"[host_sender] WARN sendto failed: {e}", file=sys.stderr)
            else:
                sent += 1
                bytes_sent += len(payload)
                seq = (seq + 1) & 0xFFFFFFFF
                # Source-receive-to-send lag (host monotonic seconds).
                lag = max(0.0, time.monotonic() - mocap_ts)
                win_lag_acc += lag
                win_lag_n += 1
                if lag > win_lag_max:
                    win_lag_max = lag
        else:
            dup += 1

        # 2) Pacing -- "send on update" by default, optional explicit Hz cap.
        if pace_dt > 0.0:
            next_send_t += pace_dt
            sleep_for = next_send_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_send_t = time.perf_counter()
        else:
            # No pacing: poll just slightly faster than the GMR producer
            # to keep duplicate-read overhead bounded.  At ~90 Hz producer
            # rate a 200 Hz poll yields ~50% dup rate which is still
            # negligible CPU on a 4090 workstation.
            time.sleep(0.001)

        # 3) Periodic stats.
        now = time.perf_counter()
        elapsed = now - win_t0
        if elapsed >= args.log_every_sec:
            win_sent = sent - win_sent0
            send_hz = win_sent / elapsed if elapsed > 0 else 0.0
            bw_kbps = bytes_sent / max(elapsed, 1e-6) / 1024.0 * 8.0
            lag_mean = (win_lag_acc / win_lag_n * 1e3) if win_lag_n else 0.0
            lag_max = win_lag_max * 1e3
            print(
                f"[host_sender] send={send_hz:5.1f}Hz  "
                f"sent={sent}  dup_polls={dup}  err={err_send}  "
                f"lag_ms mean={lag_mean:5.2f} max={lag_max:5.2f}  "
                f"bw={bw_kbps:6.1f} kbps"
            )
            win_t0 = now
            win_sent0 = sent
            bytes_sent = 0
            win_lag_acc = 0.0
            win_lag_n = 0
            win_lag_max = 0.0


def _start_dex3_subprocess(args: HostSenderArgs, mocap_type: MocapType):
    buf_mocap, ts_mocap, buf_hand = start_realtime_retarget(
        server_ip=args.server_ip,
        client_ip=args.client_ip,
        robot="unitree_g1",
        dof_full=G1_DOF_FULL,
        actual_human_height=args.human_height,
        visualize_retarget=args.visualize_retarget,
        mocap_type=mocap_type,
        buffer_ms=args.buffer_ms,
        # No SCHED_FIFO on a workstation: leave the GMR subprocess as a
        # normal-priority child so it does not contend with the IDE / viewer.
        rt_pin=None,
        # Xsens MVN options - silently ignored by the noitom / optitrack
        # branches inside ``start_realtime_retarget``.
        xsens_host=args.xsens_host,
        xsens_port=args.xsens_port,
        xsens_protocol=args.xsens_protocol,
    )
    return buf_mocap, ts_mocap, buf_hand, None


def _start_brainco_subprocess(args: HostSenderArgs, mocap_type: MocapType):
    # Import lazily so workstations without the brainco stack can still use
    # the Dex3 path.  The helper is a thin wrapper around deploy.retarget's
    # worker plus the per-frame BrainCo hand-qpos retarget.
    from deploy.brainco.play_track_brainco import (
        start_realtime_retarget_with_brainco_hands,
    )

    buf_mocap, ts_mocap, buf_hand, buf_brainco_qpos = (
        start_realtime_retarget_with_brainco_hands(
            server_ip=args.server_ip,
            client_ip=args.client_ip,
            robot="unitree_g1",
            dof_full=G1_DOF_FULL,
            actual_human_height=args.human_height,
            # IMPORTANT: must be explicit -- the upstream helper defaults to
            # True and would spawn a mujoco viewer child process here, which
            # both wastes a CPU core and crashes on headless / SSH-only
            # workstations.  Use the `--visualize-retarget` switch (below)
            # to opt back in when running on the local console.
            visualize_retarget=args.visualize_retarget,
            mocap_type=mocap_type,
            buffer_ms=args.buffer_ms,
            hand_target=args.hand_target,
            rt_pin=None,
        )
    )
    return buf_mocap, ts_mocap, buf_hand, buf_brainco_qpos


def main(args: HostSenderArgs) -> None:
    _install_signal_handlers()
    backend = "BrainCo" if args.enable_brainco_hand else "Dex3"
    mocap_label = args.mocap_type.lower()
    if mocap_label == "xsens":
        source_info = (
            f"mocap_type=xsens  listen={args.xsens_host}:{args.xsens_port}/"
            f"{args.xsens_protocol.upper()}  src_human=fbx_xsens"
        )
    else:
        source_info = f"mocap_type={mocap_label}  server_ip={args.server_ip}"
    print(f"[host_sender] Starting mocap+GMR ({backend}) subprocess on this workstation...")
    print(
        f"[host_sender]   {source_info}  "
        f"human_height={args.human_height:.2f}  host_buffer_ms={args.buffer_ms:.1f}"
        f"{('  hand_target=' + args.hand_target) if args.enable_brainco_hand else ''}"
    )

    mocap_type = _resolve_mocap_type(args.mocap_type)
    if args.enable_brainco_hand:
        if mocap_type == MocapType.XSENS:
            raise SystemExit(
                "[host_sender] --enable-brainco-hand requires per-finger mocap "
                "data, which the standard 23-segment Xsens MVN stream does not "
                "provide.  Use Noitom (--mocap-type pnlink) for BrainCo, or "
                "drop --enable-brainco-hand when running with Xsens."
            )
        buf_mocap, ts_mocap, buf_hand, buf_brainco_qpos = _start_brainco_subprocess(
            args, mocap_type
        )
    else:
        buf_mocap, ts_mocap, buf_hand, buf_brainco_qpos = _start_dex3_subprocess(
            args, mocap_type
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Small send buffer + non-blocking is fine -- packets are tiny.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 16)
    dest = (args.robot_ip, args.robot_port)

    try:
        _wait_first_frame(buf_mocap, ts_mocap, args.startup_timeout_sec)
        _send_loop(args, sock, dest, buf_mocap, ts_mocap, buf_hand, buf_brainco_qpos)
    except KeyboardInterrupt:
        print("\n[host_sender] Ctrl+C received, shutting down...")
    finally:
        try:
            sock.close()
        except OSError:
            pass
        _stop_retarget_sessions()
        print("[host_sender] Shutdown complete.")


if __name__ == "__main__":
    main(tyro.cli(HostSenderArgs))
