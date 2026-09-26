"""Robot-side UDP receiver for the cable-free onboard deploy.

This is the **subprocess** version of the receiver: a dedicated child
process owns the ``recvfrom + decode_frame`` loop and publishes the
latest frame via shared memory.  Compared with the previous same-process
thread implementation, this avoids GIL contention with the 50 Hz control
loop -- the worker holds its own GIL in a separate interpreter, so the
loco thread is never blocked waiting for the receiver to release
Python's interpreter lock (which was the dominant source of motor
"click" jitter in the threaded build).

The shared-memory layout intentionally mirrors
``deploy.retarget.start_realtime_retarget`` so this class remains a
drop-in replacement for the previous threaded ``NetMocapReceiver``.
Out-of-order packets are discarded by comparing the host-side
``send_ts``; gaps in the wire ``seq`` are counted and exposed via
:meth:`stats` for diagnostics.

This module is intentionally dependency-light (numpy + stdlib only) so
the robot's tracking loop can pull it in without dragging mujoco /
tracking imports through unit tests.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import socket
import time

import numpy as np

from .protocol import (
    BRAINCO_QPOS_FLOATS,
    G1_DOF_FULL,
    HAND_FLOATS,
    MAX_PACKET_SIZE,
    decode_frame,
)

_log = logging.getLogger("onboard_wo_gmr.net_recv")


# Slot indices into the uint64 stats array.
_STATS_RECV = 0
_STATS_DROPPED = 1
_STATS_OOO = 2
_STATS_MISSING = 3
_STATS_LEN = 4


def _rx_worker(
    host: str,
    port: int,
    dof_full: int,
    arr_qpos,
    arr_hand,
    arr_brainco,
    val_send_ts,
    val_recv_ts,
    val_has_hand,
    val_has_brainco,
    arr_stats,
    shared_lock,
    ready_evt,
    stop_evt,
    rt_pin,
) -> None:
    """Child-process entry point: read UDP, decode, publish to shm."""
    if rt_pin is not None:
        # macOS / Windows don't expose sched_setaffinity / SCHED_FIFO;
        # treat any failure here as non-fatal so the worker still runs
        # under the default scheduler.
        try:
            cpu_id, fifo_prio = rt_pin
            os.sched_setaffinity(0, {int(cpu_id)})
            os.sched_setscheduler(
                0, os.SCHED_FIFO, os.sched_param(int(fifo_prio))
            )
        except (OSError, PermissionError, AttributeError, ValueError) as e:
            print(
                f"[net_recv] RT pin failed ({e}); using default scheduling",
                flush=True,
            )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 18)
    try:
        sock.bind((host, port))
    except OSError as e:
        print(f"[net_recv] UDP bind failed on {host}:{port}: {e}", flush=True)
        return
    sock.settimeout(0.5)

    qpos_view = np.frombuffer(arr_qpos, dtype=np.float32)
    hand_view = np.frombuffer(arr_hand, dtype=np.float32)
    brainco_view = np.frombuffer(arr_brainco, dtype=np.float32)
    stats_view = np.frombuffer(arr_stats, dtype=np.uint64)

    last_seq: int | None = None

    try:
        while not stop_evt.is_set():
            try:
                buf, _addr = sock.recvfrom(MAX_PACKET_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                pkt = decode_frame(buf)
            except ValueError:
                with shared_lock:
                    stats_view[_STATS_DROPPED] += 1
                continue

            if pkt.qpos.size != dof_full:
                with shared_lock:
                    stats_view[_STATS_DROPPED] += 1
                continue

            recv_ts = time.time()

            with shared_lock:
                # Drop strictly older frames so the latch never moves
                # backwards.
                if pkt.send_ts < val_send_ts.value:
                    stats_view[_STATS_OOO] += 1
                    continue

                # Gap detection (sender's seq wraps at 2^32; treat
                # positive forward gap > 1 as in-flight loss).
                if last_seq is not None:
                    gap = (pkt.seq - last_seq) & 0xFFFFFFFF
                    if 0 < gap < (1 << 31) and gap > 1:
                        stats_view[_STATS_MISSING] += gap - 1
                last_seq = pkt.seq

                qpos_view[:] = pkt.qpos
                if pkt.hand is not None:
                    hand_view[:] = pkt.hand
                    val_has_hand.value = 1
                if pkt.brainco_qpos is not None:
                    brainco_view[:] = pkt.brainco_qpos
                    val_has_brainco.value = 1
                val_send_ts.value = pkt.send_ts
                val_recv_ts.value = recv_ts
                stats_view[_STATS_RECV] += 1

            if not ready_evt.is_set():
                ready_evt.set()
    finally:
        try:
            sock.close()
        except OSError:
            pass


_ACTIVE_RECEIVERS: list["NetMocapReceiver"] = []


def _stop_all_at_exit() -> None:
    for r in list(_ACTIVE_RECEIVERS):
        try:
            r.stop()
        except Exception:
            pass


atexit.register(_stop_all_at_exit)


class NetMocapReceiver:
    """UDP receiver running in a dedicated subprocess.

    The worker runs in a separate Python interpreter (spawned via
    ``mp.get_context("spawn")``) so its GIL never blocks the on-board
    50 Hz control loop.  Set ``rt_pin=(cpu_id, fifo_prio)`` to pin the
    worker to a specific core with SCHED_FIFO scheduling, mirroring how
    GMR is run by ``deploy.retarget.start_realtime_retarget``.

    The public API (``start`` / ``stop`` / ``wait_first`` / ``read`` /
    ``read_hand`` / ``read_brainco_qpos`` / ``stats``) is a strict
    superset of the previous threaded version, so existing callers do
    not need to be updated.
    """

    def __init__(
        self,
        host: str,
        port: int,
        dof_full: int = G1_DOF_FULL,
        rt_pin: tuple[int, int] | None = None,
    ):
        self._host = host
        self._port = port
        self._dof_full = dof_full
        self._rt_pin = rt_pin

        # spawn (not fork) keeps us safe if the parent has already
        # imported ORT / torch / CUDA; fork after those imports can
        # deadlock in their internal mutexes.
        ctx = mp.get_context("spawn")
        self._ctx = ctx

        # All shared state lives behind a single Lock so reads observe a
        # coherent snapshot.  We use lock=False on the Arrays/Values to
        # avoid per-field locking, which would otherwise allow torn reads
        # across the qpos / send_ts / hand fields.
        self._arr_qpos = ctx.Array("f", dof_full, lock=False)
        # identity quat so reads issued before the first packet stay sane
        np.frombuffer(self._arr_qpos, dtype=np.float32)[3] = 1.0
        self._arr_hand = ctx.Array("f", HAND_FLOATS, lock=False)
        self._arr_brainco = ctx.Array("f", BRAINCO_QPOS_FLOATS, lock=False)
        self._val_send_ts = ctx.Value("d", 0.0, lock=False)
        self._val_recv_ts = ctx.Value("d", 0.0, lock=False)
        self._val_has_hand = ctx.Value("i", 0, lock=False)
        self._val_has_brainco = ctx.Value("i", 0, lock=False)
        self._arr_stats = ctx.Array("Q", _STATS_LEN, lock=False)
        self._shared_lock = ctx.Lock()
        self._ready_evt = ctx.Event()
        self._stop_evt = ctx.Event()
        self._proc: mp.Process | None = None

        # numpy views in the parent process.  These point at the same
        # physical shared-memory pages as the worker's views, so reads
        # here observe the worker's writes (under the shared lock).
        self._qpos_view = np.frombuffer(self._arr_qpos, dtype=np.float32)
        self._hand_view = np.frombuffer(self._arr_hand, dtype=np.float32)
        self._brainco_view = np.frombuffer(self._arr_brainco, dtype=np.float32)
        self._stats_view = np.frombuffer(self._arr_stats, dtype=np.uint64)

    def start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = self._ctx.Process(
            target=_rx_worker,
            args=(
                self._host,
                self._port,
                self._dof_full,
                self._arr_qpos,
                self._arr_hand,
                self._arr_brainco,
                self._val_send_ts,
                self._val_recv_ts,
                self._val_has_hand,
                self._val_has_brainco,
                self._arr_stats,
                self._shared_lock,
                self._ready_evt,
                self._stop_evt,
                self._rt_pin,
            ),
            name="netmocap-rx",
            daemon=True,
        )
        self._proc.start()
        _ACTIVE_RECEIVERS.append(self)
        _log.info(
            "NetMocapReceiver subprocess started "
            "(pid=%s, listen=%s:%d, rt_pin=%s)",
            self._proc.pid, self._host, self._port, self._rt_pin,
        )

    def stop(self) -> None:
        self._stop_evt.set()
        proc = self._proc
        if proc is not None:
            proc.join(timeout=2.0)
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
                proc.join(timeout=1.0)
            self._proc = None
        try:
            _ACTIVE_RECEIVERS.remove(self)
        except ValueError:
            pass

    def wait_first(self, timeout_sec: float) -> bool:
        """Block until the first valid packet arrives (or timeout)."""
        return self._ready_evt.wait(timeout=timeout_sec)

    def read(self) -> tuple[np.ndarray, float]:
        """Return (qpos_full, send_ts).  Matches MocapBuffer.read() shape."""
        with self._shared_lock:
            qpos = self._qpos_view.copy()
            send_ts = self._val_send_ts.value
        return qpos, send_ts

    def read_hand(self) -> tuple[bool, float, bool, float] | None:
        with self._shared_lock:
            if not self._val_has_hand.value:
                return None
            data = self._hand_view.copy()
        return bool(data[0]), float(data[1]), bool(data[2]), float(data[3])

    def read_brainco_qpos(self) -> np.ndarray | None:
        """Return the latest 24-D BrainCo dex-hand qpos, or None if never seen.

        Layout matches :func:`deploy.brainco.play_track_brainco.brainco_qpos24_to_cmd12`
        (left12 then right12).  Caller owns the returned copy.
        """
        with self._shared_lock:
            if not self._val_has_brainco.value:
                return None
            return self._brainco_view.copy()

    def stats(self) -> dict:
        with self._shared_lock:
            return {
                "recv": int(self._stats_view[_STATS_RECV]),
                "dropped": int(self._stats_view[_STATS_DROPPED]),
                "ooo": int(self._stats_view[_STATS_OOO]),
                "missing": int(self._stats_view[_STATS_MISSING]),
                "last_send_ts": float(self._val_send_ts.value),
                "last_recv_ts": float(self._val_recv_ts.value),
            }
