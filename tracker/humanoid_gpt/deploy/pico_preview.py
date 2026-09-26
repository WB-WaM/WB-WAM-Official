"""Preview the live Pico-to-G1 retargeting stream.

This command intentionally stops before loading a HumanoidGPT policy.  It is
useful for checking the XRobo coordinate convention and GMR output before the
result is allowed to drive the MuJoCo tracking controller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from deploy.pico import (
    PicoFrameCalibrator,
    PicoXrtSource,
    pico_frame_to_gmr_frame,
)
from deploy.retarget import ema_smooth_g1_qpos, validate_g1_qpos

_ROOT_QUAT_SLICE = slice(3, 7)


@dataclass
class PicoPreviewArgs:
    """Command-line options for the Pico/GMR preview."""

    service_mode: Literal["auto", "external"] = "auto"
    """Start Robotics Service locally, or connect to an externally run one."""

    startup_timeout_s: float = 15.0
    """Maximum time to wait for the first valid XRobo body frame."""

    duration_s: float = 0.0
    """Stop after this many seconds; zero runs until Ctrl-C."""

    human_height: float = 1.7
    """Operator height in metres passed to GMR."""

    ema_alpha: float = 0.75
    """Previous-frame weight used for quaternion-safe EMA smoothing."""

    calibrate: bool = True
    """Rebase first-frame pelvis XY/yaw and foot ground to the GMR world."""

    stale_ms: float = 100.0
    """Count a stream-stale event after this long without a new valid frame."""

    stats_interval_s: float = 1.0
    """Interval between console statistics updates."""

    idle_sleep_ms: float = 1.0
    """Sleep after an empty source read to avoid a busy loop."""

    headless: bool = False
    """Disable the GMR MuJoCo viewer."""

    viewer_fps: float = 60.0
    """Nominal motion rate supplied to RobotMotionViewer."""

    record_path: Path | None = None
    """Optional NPZ path for raw Pico frames, timestamps, and G1 qpos."""


class _SourceClockAligner:
    """Estimate relative frame age for an arbitrary device clock.

    XRobo timestamps and ``time.monotonic_ns`` need not share an epoch.  The
    minimum observed offset is treated as the transport baseline, making the
    reported age a useful jitter/backlog measure without assuming a clock
    domain.
    """

    def __init__(self) -> None:
        self._minimum_offset_ns: int | None = None

    def frame_age_ms(self, source_timestamp_ns: int, host_timestamp_ns: int) -> float:
        offset_ns = host_timestamp_ns - source_timestamp_ns
        if self._minimum_offset_ns is None or offset_ns < self._minimum_offset_ns:
            self._minimum_offset_ns = offset_ns
        return max(0.0, (offset_ns - self._minimum_offset_ns) / 1e6)


def _save_recording(
    path: Path,
    raw_frames: list[np.ndarray],
    timestamps_ns: list[int],
    qpos_frames: list[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.stack(raw_frames).astype(np.float32, copy=False)
    qpos = np.stack(qpos_frames).astype(np.float32, copy=False)
    np.savez_compressed(
        path,
        body_poses=raw,
        timestamps_ns=np.asarray(timestamps_ns, dtype=np.int64),
        qpos=qpos,
    )
    print(f"[PicoPreview] saved {len(raw_frames)} frames to {path}")


def run_preview(args: PicoPreviewArgs) -> None:
    """Run Pico -> GMR -> optional viewer until timeout or Ctrl-C."""

    if args.startup_timeout_s <= 0.0:
        raise ValueError("startup_timeout_s must be positive")
    if args.duration_s < 0.0:
        raise ValueError("duration_s must be non-negative")
    if args.human_height <= 0.0:
        raise ValueError("human_height must be positive")
    if not 0.0 <= args.ema_alpha < 1.0:
        raise ValueError("ema_alpha must be in [0, 1)")
    if args.stale_ms <= 0.0:
        raise ValueError("stale_ms must be positive")
    if args.stats_interval_s <= 0.0:
        raise ValueError("stats_interval_s must be positive")
    if args.idle_sleep_ms < 0.0:
        raise ValueError("idle_sleep_ms must be non-negative")

    # Keep both optional heavyweight dependencies lazy so this module remains
    # importable in CI and on hosts used only for environment checks.
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    viewer = None
    source = PicoXrtSource(
        service_mode=args.service_mode,
        startup_timeout_s=args.startup_timeout_s,
    )
    retarget = GMR(
        src_human="xrobot",
        tgt_robot="unitree_g1",
        actual_human_height=args.human_height,
    )

    raw_frames: list[np.ndarray] = []
    timestamps_ns: list[int] = []
    qpos_frames: list[np.ndarray] = []
    clock_aligner = _SourceClockAligner()

    calibrator = PicoFrameCalibrator() if args.calibrate else None
    previous_qpos: np.ndarray | None = None
    invalid_count = 0
    stale_count = 0
    stream_is_stale = False
    input_count = 0
    gmr_count = 0
    interval_input_count = 0
    interval_gmr_count = 0
    interval_gmr_latency_ms = 0.0
    last_frame_age_ms = float("nan")
    start_time = time.monotonic()
    stats_time = start_time
    last_valid_time = start_time

    try:
        source.start()
        if not args.headless:
            from general_motion_retargeting import RobotMotionViewer

            viewer = RobotMotionViewer(
                robot_type="unitree_g1",
                motion_fps=args.viewer_fps,
            )
        print(
            "[PicoPreview] live; "
            f"viewer={'off' if args.headless else 'on'}, Ctrl-C to stop"
        )
        while args.duration_s == 0.0 or time.monotonic() - start_time < args.duration_s:
            now = time.monotonic()
            frame = None
            try:
                frame = source.read()
            except Exception as exc:  # Hardware errors should not corrupt a recording.
                invalid_count += 1
                print(f"[PicoPreview] source read rejected: {exc}")

            if frame is None:
                if (
                    now - last_valid_time
                ) * 1000.0 > args.stale_ms and not stream_is_stale:
                    stale_count += 1
                    stream_is_stale = True
                if args.idle_sleep_ms:
                    time.sleep(args.idle_sleep_ms / 1000.0)
            else:
                input_count += 1
                interval_input_count += 1
                host_timestamp_ns = time.monotonic_ns()
                try:
                    gmr_frame = pico_frame_to_gmr_frame(frame, calibrator)
                    retarget_start_ns = time.perf_counter_ns()
                    qpos = validate_g1_qpos(retarget.retarget(gmr_frame))
                    gmr_latency_ms = (time.perf_counter_ns() - retarget_start_ns) / 1e6
                    qpos = ema_smooth_g1_qpos(previous_qpos, qpos, args.ema_alpha)
                except Exception as exc:
                    invalid_count += 1
                    print(f"[PicoPreview] frame rejected: {exc}")
                    continue

                source_timestamp_ns = int(frame.timestamp_ns)
                last_frame_age_ms = clock_aligner.frame_age_ms(
                    source_timestamp_ns,
                    host_timestamp_ns,
                )
                if last_frame_age_ms > args.stale_ms:
                    stale_count += 1
                    stream_is_stale = True
                else:
                    stream_is_stale = False

                previous_qpos = qpos
                last_valid_time = time.monotonic()
                gmr_count += 1
                interval_gmr_count += 1
                interval_gmr_latency_ms += gmr_latency_ms

                if args.record_path is not None:
                    raw_frames.append(
                        np.asarray(frame.body_poses, dtype=np.float32).copy()
                    )
                    timestamps_ns.append(source_timestamp_ns)
                    qpos_frames.append(qpos.copy())

                if viewer is not None:
                    viewer.step(
                        root_pos=qpos[:3],
                        root_rot=qpos[_ROOT_QUAT_SLICE],
                        dof_pos=qpos[7:],
                    )

            stats_now = time.monotonic()
            interval_s = stats_now - stats_time
            if interval_s >= args.stats_interval_s:
                latency_ms = (
                    interval_gmr_latency_ms / interval_gmr_count
                    if interval_gmr_count
                    else float("nan")
                )
                print(
                    "[PicoPreview] "
                    f"input={interval_input_count / interval_s:5.1f} Hz "
                    f"gmr={interval_gmr_count / interval_s:5.1f} Hz "
                    f"latency={latency_ms:5.2f} ms "
                    f"age={last_frame_age_ms:5.2f} ms "
                    f"invalid={invalid_count} stale={stale_count}"
                )
                stats_time = stats_now
                interval_input_count = 0
                interval_gmr_count = 0
                interval_gmr_latency_ms = 0.0
    except KeyboardInterrupt:
        print("\n[PicoPreview] interrupted")
    finally:
        source.close()
        if viewer is not None and hasattr(viewer, "close"):
            viewer.close()
        if args.record_path is not None:
            if raw_frames:
                _save_recording(
                    args.record_path,
                    raw_frames,
                    timestamps_ns,
                    qpos_frames,
                )
            else:
                print("[PicoPreview] no valid frames; recording was not written")

    elapsed_s = max(time.monotonic() - start_time, 1e-9)
    print(
        "[PicoPreview] done: "
        f"input={input_count / elapsed_s:.1f} Hz, "
        f"gmr={gmr_count / elapsed_s:.1f} Hz, "
        f"invalid={invalid_count}, stale={stale_count}"
    )


def main() -> None:
    run_preview(tyro.cli(PicoPreviewArgs))


if __name__ == "__main__":
    main()
