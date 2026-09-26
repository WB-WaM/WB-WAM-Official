"""Benchmark the cable-free onboard mocap receiver.

This is the receiver-side equivalent of
``deploy.onboard_deploy.bench_noitom_stream`` for the new workstation-assisted
deployment.  It exercises *only* the network path:

    UDP datagram on wlan0 -> NetMocapReceiver._rx_loop -> decode_frame -> latch

and measures throughput, latency, packet integrity, and basic content
sanity.  It does not start MuJoCo, ONNX, or DDS, so it is safe to run on the
G1 Jetson without touching motors.

Typical usage on the G1 (over SSH), while ``host_sender`` is already running
on the 4090 workstation::

    python -m deploy.onboard_deploy_wo_GMR.bench_net_recv

For a quick local sanity check that does not require the workstation, use
``--self-test`` -- a background thread will synthesize fake packets on
loopback and the receiver will consume them::

    python -m deploy.onboard_deploy_wo_GMR.bench_net_recv \
        --self-test --duration-sec 5 --listen-ip 127.0.0.1
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import numpy as np
import tyro

from deploy.onboard_deploy._bench_utils import aarch64_preload, print_environment_info

from .net_recv import NetMocapReceiver
from .protocol import (
    BRAINCO_QPOS_FLOATS,
    DEFAULT_PORT,
    G1_DOF_FULL,
    HAND_FLOATS,
    encode_frame,
    packet_size,
)

aarch64_preload()


@dataclass
class BenchNetRecvArgs:
    """Receiver-side benchmark for the cable-free onboard mocap stream."""

    duration_sec: float = 30.0
    """Measured benchmark duration after warmup."""

    warmup_sec: float = 3.0
    """Warmup duration before statistics are recorded."""

    poll_hz: float = 200.0
    """Main-loop polling rate over the receiver latch.

    Matches the rough rate at which the control thread reads the latest
    mocap frame in ``play_track_onboard_wo_GMR.py``.  200 Hz comfortably
    over-samples a ~90 Hz producer so we never miss a wire frame in stats.
    """

    listen_ip: str = "0.0.0.0"
    """Bind address.  Use 0.0.0.0 to accept from any interface (wlan0)."""

    listen_port: int = DEFAULT_PORT
    """UDP port; must match host_sender's --robot-port."""

    startup_timeout_sec: float = 30.0
    """Max wait for the first valid packet."""

    stale_warn_sec: float = 1.0
    """Print a WARN when no new packet arrives for this long."""

    print_every_sec: float = 1.0
    """Periodic progress print interval."""

    has_hand: bool = True
    """Expected packet layout (for size-budget reporting in the summary)."""

    has_brainco: bool = False
    """Set when the host_sender is run with --enable-brainco-hand so the
    summary reports the right expected packet size."""

    self_test: bool = False
    """Start a localhost sender thread that emits synthetic frames.

    Useful for sanity-checking the receiver without bringing up the
    workstation pipeline.  When set, ``listen_ip`` should typically be
    ``127.0.0.1`` to avoid leaking onto the WiFi.
    """

    self_test_hz: float = 90.0
    """Send rate for --self-test (matches Noitom's nominal producer Hz)."""

    self_test_brainco: bool = False
    """Make the synthetic --self-test sender emit BrainCo qpos too."""

    rt_pin: tuple[int, int] | None = (2, 40)
    """Pin the recv subprocess to (cpu_id, SCHED_FIFO priority).  Matches
    the default used by play_track_onboard_wo_GMR.  Set to None when
    running without CAP_SYS_NICE (e.g. on a laptop)."""


@dataclass
class StreamStats:
    polls: int
    elapsed_sec: float
    arrival_dt: list[float]      # producer-side dt from packet send_ts
    poll_arrival_dt: list[float] # local dt between distinct frame arrivals
    read_ms: list[float]         # latency of NetMocapReceiver.read()
    lag_ms: list[float]          # max(0, recv_wall - packet.send_ts)
    qpos_qnorm: list[float]      # |q| of root quaternion (should be ~1.0)
    qpos_height: list[float]     # qpos[2] -- root height for sanity
    brainco_min: list[float]     # min element of received brainco_qpos (if any)
    brainco_max: list[float]     # max element of received brainco_qpos (if any)
    brainco_nan_frames: int      # # of frames where brainco_qpos had NaN/Inf
    brainco_seen: bool           # did we ever observe a brainco frame?
    stale_events: int
    # Snapshot of receiver counters at start/end of the measurement window.
    recv_start: int
    recv_end: int
    dropped_start: int
    dropped_end: int
    ooo_start: int
    ooo_end: int
    missing_start: int
    missing_end: int


def _percentile_line(name: str, values: list[float], unit: str) -> str:
    if not values:
        return f"  {name:<18}: no samples"
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"  {name:<18}: mean={arr.mean():8.3f}{unit}  "
        f"p50={np.percentile(arr, 50):8.3f}{unit}  "
        f"p90={np.percentile(arr, 90):8.3f}{unit}  "
        f"p99={np.percentile(arr, 99):8.3f}{unit}  "
        f"max={arr.max():8.3f}{unit}"
    )


def _start_self_test_sender(
    target_ip: str,
    target_port: int,
    hz: float,
    stop_evt: threading.Event,
    with_brainco: bool = False,
) -> threading.Thread:
    """Localhost synthetic packet generator -- used by --self-test only."""

    def _run():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Deterministic but mocap-like values: identity quat + small angles.
        qpos = np.zeros(G1_DOF_FULL, dtype=np.float32)
        qpos[2] = 0.75            # plausible standing root height
        qpos[3] = 1.0             # identity quaternion (w, x, y, z)
        hand = np.array([0.0, 0.03, 1.0, 0.08], dtype=np.float32)
        brainco = (
            np.linspace(0.0, 0.5, BRAINCO_QPOS_FLOATS, dtype=np.float32)
            if with_brainco
            else None
        )
        seq = 0
        dt = 1.0 / max(hz, 1.0)
        next_t = time.perf_counter()
        while not stop_evt.is_set():
            payload = encode_frame(
                seq=seq, send_ts=time.time(), qpos=qpos, hand=hand,
                brainco_qpos=brainco,
            )
            try:
                sock.sendto(payload, (target_ip, target_port))
            except OSError:
                pass
            seq += 1
            next_t += dt
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.perf_counter()
        sock.close()

    t = threading.Thread(target=_run, name="bench-selftest-tx", daemon=True)
    t.start()
    return t


def _wait_first_frame(recv: NetMocapReceiver, timeout_sec: float) -> None:
    print(f"Waiting for first valid packet (timeout={timeout_sec:.1f}s)...")
    t0 = time.perf_counter()
    if recv.wait_first(timeout_sec):
        print(f"  First packet received after {time.perf_counter() - t0:.2f}s")
        return
    raise TimeoutError(
        "No valid packet arrived before timeout.  Check that:\n"
        "  - host_sender is running on the workstation\n"
        "  - --listen-port matches host_sender's --robot-port\n"
        "  - the G1 and the workstation are on the same WiFi subnet\n"
        "  - no firewall is dropping UDP on the G1"
    )


def _collect_stream_stats(
    recv: NetMocapReceiver,
    *,
    duration_sec: float,
    poll_hz: float,
    stale_warn_sec: float,
    print_every_sec: float,
) -> StreamStats:
    poll_dt = 1.0 / poll_hz if poll_hz > 0 else 0.0

    arrival_dt: list[float] = []
    poll_arrival_dt: list[float] = []
    read_ms: list[float] = []
    lag_ms: list[float] = []
    qpos_qnorm: list[float] = []
    qpos_height: list[float] = []
    brainco_min: list[float] = []
    brainco_max: list[float] = []
    brainco_nan_frames = 0
    brainco_seen = False

    polls = 0
    stale_events = 0
    last_send_ts = 0.0
    last_arrival_wall = 0.0
    last_progress_t = time.perf_counter()
    last_progress_polls = 0
    last_progress_recv = recv.stats()["recv"]
    last_stale_report_t = 0.0
    start_perf = time.perf_counter()
    next_poll_t = start_perf

    s0 = recv.stats()
    recv_start = s0["recv"]
    dropped_start = s0["dropped"]
    ooo_start = s0["ooo"]
    missing_start = s0["missing"]

    while True:
        now_perf = time.perf_counter()
        elapsed = now_perf - start_perf
        if elapsed >= duration_sec:
            break

        t0 = time.perf_counter()
        qpos, send_ts = recv.read()
        read_ms.append((time.perf_counter() - t0) * 1e3)
        polls += 1

        now_wall = time.time()
        if send_ts > 0.0 and send_ts != last_send_ts:
            if last_send_ts > 0.0:
                arrival_dt.append(send_ts - last_send_ts)
            if last_arrival_wall > 0.0:
                poll_arrival_dt.append(now_wall - last_arrival_wall)
            lag_ms.append(max(0.0, now_wall - send_ts) * 1e3)

            # Content sanity: root quaternion norm + root height.
            q = qpos[3:7]
            qpos_qnorm.append(float(np.linalg.norm(q)))
            qpos_height.append(float(qpos[2]))

            bq = recv.read_brainco_qpos()
            if bq is not None:
                brainco_seen = True
                if not np.isfinite(bq).all():
                    brainco_nan_frames += 1
                else:
                    brainco_min.append(float(bq.min()))
                    brainco_max.append(float(bq.max()))

            last_send_ts = send_ts
            last_arrival_wall = now_wall

        stale_for = now_wall - last_arrival_wall
        if (
            last_arrival_wall > 0.0
            and stale_for > stale_warn_sec
            and now_wall - last_stale_report_t > stale_warn_sec
        ):
            stale_events += 1
            last_stale_report_t = now_wall
            print(f"  WARN: no new packet for {stale_for:.2f}s")

        if now_perf - last_progress_t >= print_every_sec:
            win_elapsed = now_perf - last_progress_t
            s_now = recv.stats()
            win_recv = s_now["recv"] - last_progress_recv
            win_polls = polls - last_progress_polls
            print(
                f"  t={elapsed:6.1f}s  recv={win_recv / win_elapsed:6.1f}Hz  "
                f"poll={win_polls / win_elapsed:6.1f}Hz  "
                f"loss={s_now['missing']}  ooo={s_now['ooo']}  "
                f"bad={s_now['dropped']}  total_recv={s_now['recv']}"
            )
            last_progress_t = now_perf
            last_progress_polls = polls
            last_progress_recv = s_now["recv"]

        if poll_dt > 0.0:
            next_poll_t += poll_dt
            sleep_for = next_poll_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_poll_t = time.perf_counter()

    s1 = recv.stats()
    return StreamStats(
        polls=polls,
        elapsed_sec=time.perf_counter() - start_perf,
        arrival_dt=arrival_dt,
        poll_arrival_dt=poll_arrival_dt,
        read_ms=read_ms,
        lag_ms=lag_ms,
        qpos_qnorm=qpos_qnorm,
        qpos_height=qpos_height,
        brainco_min=brainco_min,
        brainco_max=brainco_max,
        brainco_nan_frames=brainco_nan_frames,
        brainco_seen=brainco_seen,
        stale_events=stale_events,
        recv_start=recv_start,
        recv_end=s1["recv"],
        dropped_start=dropped_start,
        dropped_end=s1["dropped"],
        ooo_start=ooo_start,
        ooo_end=s1["ooo"],
        missing_start=missing_start,
        missing_end=s1["missing"],
    )


def _print_summary(args: BenchNetRecvArgs, stats: StreamStats) -> None:
    n_recv = stats.recv_end - stats.recv_start
    n_bad = stats.dropped_end - stats.dropped_start
    n_ooo = stats.ooo_end - stats.ooo_start
    n_missing = stats.missing_end - stats.missing_start

    recv_hz = n_recv / stats.elapsed_sec if stats.elapsed_sec > 0 else 0.0
    poll_hz = stats.polls / stats.elapsed_sec if stats.elapsed_sec > 0 else 0.0
    duplicate_reads = max(0, stats.polls - n_recv)
    duplicate_pct = 100.0 * duplicate_reads / max(stats.polls, 1)
    # Loss% relative to (received + missing) -- the in-flight denominator.
    denom = max(n_recv + n_missing, 1)
    loss_pct = 100.0 * n_missing / denom

    pkt_bytes = packet_size(
        G1_DOF_FULL, has_hand=args.has_hand, has_brainco=args.has_brainco,
    )
    bw_kbps = recv_hz * pkt_bytes * 8.0 / 1024.0

    print(f"\n{'=' * 88}")
    print("  Network mocap receiver benchmark")
    print(f"{'=' * 88}")
    print(f"  duration         : {stats.elapsed_sec:.2f}s")
    print(f"  listen           : {args.listen_ip}:{args.listen_port}")
    print(
        f"  expected pkt     : {pkt_bytes} B  "
        f"(has_hand={args.has_hand}, has_brainco={args.has_brainco})"
    )
    print(f"  received         : {n_recv} ({recv_hz:.2f} Hz, {bw_kbps:.1f} kbps)")
    print(f"  polled           : {stats.polls} ({poll_hz:.2f} Hz)")
    print(f"  duplicate reads  : {duplicate_reads} ({duplicate_pct:.1f}% of polls)")
    print(f"  bad packets      : {n_bad} (wrong magic / length / dof)")
    print(f"  out-of-order     : {n_ooo} (older send_ts -> dropped)")
    print(f"  estimated loss   : {n_missing} ({loss_pct:.2f}% of in-flight)")
    print(f"  stale warnings   : {stats.stale_events}")

    if stats.arrival_dt:
        producer_hz = 1.0 / float(np.mean(stats.arrival_dt))
        print(f"  producer hz      : {producer_hz:.2f} Hz from packet send_ts")
    print(_percentile_line("arrival dt (snd)", [v * 1e3 for v in stats.arrival_dt], "ms"))
    print(_percentile_line("arrival dt (loc)", [v * 1e3 for v in stats.poll_arrival_dt], "ms"))
    print(_percentile_line("read latency",     stats.read_ms,                       "ms"))
    print(_percentile_line("net lag (snd->rd)", stats.lag_ms,                       "ms"))

    if stats.qpos_qnorm:
        qn = np.asarray(stats.qpos_qnorm)
        qn_err = float(np.max(np.abs(qn - 1.0)))
        bad_q = int(np.sum(np.abs(qn - 1.0) > 1e-2))
        ht = np.asarray(stats.qpos_height)
        print(
            f"  root |q| sanity  : mean={qn.mean():.6f}  "
            f"max|err|={qn_err:.4f}  off>1e-2:{bad_q}/{len(qn)}"
        )
        print(
            f"  root height (m)  : mean={ht.mean():.3f}  "
            f"min={ht.min():.3f}  max={ht.max():.3f}"
        )

    if args.has_brainco or stats.brainco_seen:
        if stats.brainco_seen and stats.brainco_min:
            bn = np.asarray(stats.brainco_min)
            bx = np.asarray(stats.brainco_max)
            print(
                f"  brainco_qpos     : seen={len(bn)} frames  "
                f"min(mean)={bn.mean():+.3f}  max(mean)={bx.mean():+.3f}  "
                f"nan_frames={stats.brainco_nan_frames}"
            )
        else:
            print(
                "  brainco_qpos     : NOT received -- did the host launch with "
                "--enable-brainco-hand?"
            )
    print(f"{'=' * 88}")

    if args.self_test:
        print(
            "Note: --self-test uses a synthetic loopback sender; lag/loss numbers\n"
            "      reflect localhost only, not real WiFi conditions."
        )
    else:
        print(
            "Note: net lag = max(0, recv_wall - packet.send_ts).  It includes WiFi\n"
            "      flight time AND clock skew between the two hosts.  Run NTP/chrony\n"
            "      on both ends if you want absolute latency to be meaningful."
        )


def main(args: BenchNetRecvArgs) -> None:
    print_environment_info(extra_packages=[])
    print("\nStarting NetMocapReceiver...")
    print(
        f"  listen={args.listen_ip}:{args.listen_port}  poll_hz={args.poll_hz:.1f}"
    )

    recv = NetMocapReceiver(
        host=args.listen_ip, port=args.listen_port, dof_full=G1_DOF_FULL,
        rt_pin=args.rt_pin,
    )
    recv.start()

    selftest_stop = threading.Event()
    selftest_thread: threading.Thread | None = None
    if args.self_test:
        target = "127.0.0.1" if args.listen_ip == "0.0.0.0" else args.listen_ip
        print(
            f"  self-test sender -> {target}:{args.listen_port} "
            f"@ {args.self_test_hz:.1f} Hz  brainco={args.self_test_brainco}"
        )
        selftest_thread = _start_self_test_sender(
            target, args.listen_port, args.self_test_hz, selftest_stop,
            with_brainco=args.self_test_brainco,
        )

    try:
        _wait_first_frame(recv, args.startup_timeout_sec)

        if args.warmup_sec > 0.0:
            print(f"Warmup for {args.warmup_sec:.1f}s...")
            _collect_stream_stats(
                recv,
                duration_sec=args.warmup_sec,
                poll_hz=args.poll_hz,
                stale_warn_sec=args.stale_warn_sec,
                print_every_sec=max(args.warmup_sec + 1.0, 1.0),
            )

        print(f"\nBenchmarking for {args.duration_sec:.1f}s...")
        stats = _collect_stream_stats(
            recv,
            duration_sec=args.duration_sec,
            poll_hz=args.poll_hz,
            stale_warn_sec=args.stale_warn_sec,
            print_every_sec=args.print_every_sec,
        )
        _print_summary(args, stats)
    finally:
        if selftest_thread is not None:
            selftest_stop.set()
            selftest_thread.join(timeout=1.0)
        recv.stop()


if __name__ == "__main__":
    main(tyro.cli(BenchNetRecvArgs))
