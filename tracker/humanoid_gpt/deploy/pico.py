"""Pico/XRoboToolkit body-frame adapter for GMR.

The adapter keeps the XR SDK behind a lazy boundary so coordinate and
validation tests do not need Pico hardware or native libraries. Pico v1 is
body-only: hand poses are retained as body joints but no hand command is made.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

PICO_BODY_JOINT_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)

# Official GMR XRobo convention: Unity (x, y, z) -> GMR (x, -z, y).
UNITY_TO_GMR_BASIS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
UNITY_TO_GMR_QUAT_WXYZ = np.array(
    [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
    dtype=np.float32,
)

_BODY_SHAPE = (len(PICO_BODY_JOINT_NAMES), 7)
_MIN_BODY_EXTENT_M = 0.25
_MAX_BODY_EXTENT_M = 4.0


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Hamilton product for scalar-first quaternions."""
    w1, x1, y1, z1 = np.asarray(lhs, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(rhs, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _normalized_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("quaternion has near-zero or invalid norm")
    return quat / norm


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = _normalized_quat_wxyz(quat)
    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


@dataclass(frozen=True, slots=True)
class PicoBodyFrame:
    """A fresh global 24x7 xyz+xyzw XRobo body frame."""

    timestamp_ns: int
    body_poses: np.ndarray

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, (int, np.integer))
            or int(self.timestamp_ns) <= 0
        ):
            raise ValueError("timestamp_ns must be a positive integer")
        poses = np.asarray(self.body_poses, dtype=np.float32)
        if poses.shape != _BODY_SHAPE:
            raise ValueError(
                f"Pico body_poses must have shape {_BODY_SHAPE}, got {poses.shape}"
            )
        if not np.isfinite(poses).all():
            raise ValueError("Pico body_poses contains NaN or Inf")

        quat_norms = np.linalg.norm(poses[:, 3:7], axis=1)
        if np.any(quat_norms < 1e-6):
            raise ValueError("Pico body_poses contains a zero quaternion")

        extent = float(np.linalg.norm(np.ptp(poses[:, :3], axis=0)))
        if not _MIN_BODY_EXTENT_M <= extent <= _MAX_BODY_EXTENT_M:
            raise ValueError(
                "Pico body scale is implausible: "
                f"extent={extent:.3f} m, expected "
                f"[{_MIN_BODY_EXTENT_M}, {_MAX_BODY_EXTENT_M}]"
            )

        poses = np.array(poses, dtype=np.float32, copy=True, order="C")
        poses[:, 3:7] /= quat_norms[:, None]
        poses.setflags(write=False)
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "body_poses", poses)


class PicoFrameCalibrator:
    """Rebase first-frame pelvis XY/yaw and foot ground to the GMR world."""

    def __init__(self) -> None:
        self.reset()

    @property
    def is_calibrated(self) -> bool:
        return (
            self._origin_xy is not None
            and self._origin_yaw is not None
            and self._origin_ground_z is not None
        )

    def reset(self) -> None:
        self._origin_xy: np.ndarray | None = None
        self._origin_yaw: float | None = None
        self._origin_ground_z: float | None = None

    def apply(
        self,
        frame: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        required_joints = ("Pelvis", "Left_Foot", "Right_Foot")
        missing = [name for name in required_joints if name not in frame]
        if missing:
            raise ValueError(
                "GMR frame is missing calibration joints: " + ", ".join(missing)
            )

        pelvis_pos, pelvis_quat = frame["Pelvis"]
        if not self.is_calibrated:
            self._origin_xy = np.asarray(pelvis_pos, dtype=np.float32)[:2].copy()
            self._origin_yaw = _yaw_from_quat_wxyz(pelvis_quat)
            self._origin_ground_z = min(
                float(np.asarray(frame[name][0], dtype=np.float32)[2])
                for name in ("Left_Foot", "Right_Foot")
            )

        assert self._origin_xy is not None
        assert self._origin_yaw is not None
        assert self._origin_ground_z is not None
        yaw = -self._origin_yaw
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        yaw_quat = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
            dtype=np.float32,
        )

        calibrated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (position, quaternion) in frame.items():
            position = np.asarray(position, dtype=np.float32).copy()
            position[:2] -= self._origin_xy
            # XRobo body positions are relative to the headset tracking
            # origin, so a standing operator's feet are commonly below zero
            # after Unity-to-GMR conversion. Ground the first lower foot while
            # preserving subsequent relative vertical motion.
            position[2] -= self._origin_ground_z
            position = rotation @ position
            quaternion = _normalized_quat_wxyz(_quat_mul_wxyz(yaw_quat, quaternion))
            calibrated[name] = (position, quaternion)
        return calibrated


def pico_frame_to_gmr_frame(
    frame: PicoBodyFrame,
    calibrator: PicoFrameCalibrator | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert an XRobo frame to GMR's global xrobot dictionary."""
    if not isinstance(frame, PicoBodyFrame):
        raise TypeError(f"expected PicoBodyFrame, got {type(frame).__name__}")

    positions = frame.body_poses[:, :3] @ UNITY_TO_GMR_BASIS.T
    raw_xyzw = frame.body_poses[:, 3:7]
    converted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, name in enumerate(PICO_BODY_JOINT_NAMES):
        raw_wxyz = raw_xyzw[index, [3, 0, 1, 2]]
        quat_wxyz = _normalized_quat_wxyz(
            _quat_mul_wxyz(UNITY_TO_GMR_QUAT_WXYZ, raw_wxyz)
        )
        converted[name] = (
            positions[index].astype(np.float32, copy=True),
            quat_wxyz,
        )
    if calibrator is not None:
        return calibrator.apply(converted)
    return converted


class PicoXrtSource:
    """Lifecycle wrapper around the XRoboToolkit Python SDK."""

    def __init__(
        self,
        *,
        service_mode: Literal["auto", "external"] = "auto",
        startup_timeout_s: float = 15.0,
        service_path: str | Path = "/opt/apps/roboticsservice/runService.sh",
        poll_interval_s: float = 0.01,
        xrt_module: Any | None = None,
        process_factory: Any = subprocess.Popen,
        clock: Any = time.monotonic,
        sleep: Any = time.sleep,
    ) -> None:
        if service_mode not in {"auto", "external"}:
            raise ValueError("service_mode must be 'auto' or 'external'")
        if startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self.service_mode = service_mode
        self.startup_timeout_s = float(startup_timeout_s)
        self.service_path = Path(service_path)
        self.poll_interval_s = float(poll_interval_s)
        self._xrt = xrt_module
        self._process_factory = process_factory
        self._clock = clock
        self._sleep = sleep
        self._service_process: Any | None = None
        self._initialized = False
        self._started = False
        self._last_body_timestamp_ns = 0
        self._last_global_timestamp_ns = 0
        self._use_global_timestamp = False
        self._timestamp_source: str | None = None

    @property
    def owns_service(self) -> bool:
        return self._service_process is not None

    @property
    def sdk(self) -> Any:
        """Return initialized SDK for synchronized body/hand sampling."""
        if not self._started or self._xrt is None:
            raise RuntimeError("PicoXrtSource.start() must be called before sdk access")
        return self._xrt

    def start(
        self,
        *,
        wait_forever: bool = False,
        wait_log_interval_s: float = 1.0,
    ) -> None:
        if self._started:
            return
        if not np.isfinite(wait_log_interval_s) or wait_log_interval_s <= 0.0:
            raise ValueError("wait_log_interval_s must be finite and positive")
        try:
            if self.service_mode == "auto":
                if not self.service_path.is_file():
                    raise FileNotFoundError(
                        f"Robotics Service not found: {self.service_path}"
                    )
                self._service_process = self._process_factory(
                    ["bash", str(self.service_path)],
                    cwd=str(self.service_path.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

            if self._xrt is None:
                self._xrt = importlib.import_module("xrobotoolkit_sdk")

            # PXREARobotSDK creates a localhost gRPC channel. gRPC cannot parse
            # the socks5h proxies commonly inherited from developer shells.
            proxy_keys = (
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
            )
            saved_proxy_env = {
                key: os.environ.pop(key) for key in proxy_keys if key in os.environ
            }
            try:
                init_result = self._xrt.init()
            finally:
                os.environ.update(saved_proxy_env)

            if init_result is False:
                raise RuntimeError("xrobotoolkit_sdk.init() returned False")
            self._initialized = True

            deadline = (
                None if wait_forever else self._clock() + self.startup_timeout_s
            )
            next_wait_log_s = self._clock()
            while not self._xrt.is_body_data_available():
                now_s = self._clock()
                if wait_forever and now_s >= next_wait_log_s:
                    print("[Pico] Waiting for body data...")
                    next_wait_log_s = now_s + wait_log_interval_s
                if deadline is not None and now_s >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for the first Pico body frame "
                        f"after {self.startup_timeout_s:.1f} s"
                    )
                process = self._service_process
                if process is not None and hasattr(process, "poll"):
                    return_code = process.poll()
                    if return_code is not None:
                        raise RuntimeError(
                            "Robotics Service exited during startup "
                            f"with code {return_code}"
                        )
                self._sleep(self.poll_interval_s)
            if wait_forever:
                print("[Pico] Body data available.")
            self._started = True
        except (Exception, KeyboardInterrupt):
            with suppress(Exception):
                self.close()
            raise

    def read(self) -> PicoBodyFrame | None:
        if not self._started or self._xrt is None:
            raise RuntimeError("PicoXrtSource.start() must be called before read()")
        if not self._xrt.is_body_data_available():
            return None

        body_timestamp_ns = int(self._xrt.get_body_timestamp_ns())
        global_timestamp_getter = getattr(self._xrt, "get_time_stamp_ns", None)
        global_timestamp_ns = (
            int(global_timestamp_getter()) if callable(global_timestamp_getter) else 0
        )
        body_is_fresh = (
            body_timestamp_ns > 0 and body_timestamp_ns > self._last_body_timestamp_ns
        )
        global_is_fresh = (
            global_timestamp_ns > 0
            and global_timestamp_ns > self._last_global_timestamp_ns
        )

        # Consume both counters together so the same SDK callback cannot be
        # emitted once by each clock on consecutive reads.
        self._last_body_timestamp_ns = max(
            self._last_body_timestamp_ns, body_timestamp_ns
        )
        self._last_global_timestamp_ns = max(
            self._last_global_timestamp_ns, global_timestamp_ns
        )

        # Some XRobo Full Body payloads omit Body.timeStampNs while advancing
        # value.timeStampNs. Once fallback is needed, stay on SONIC's global
        # clock for this source lifecycle to avoid clock-domain bouncing.
        if self._use_global_timestamp:
            if not global_is_fresh:
                return None
            timestamp_ns = global_timestamp_ns
            timestamp_source = "global"
        elif body_is_fresh:
            timestamp_ns = body_timestamp_ns
            timestamp_source = "body"
        elif global_is_fresh:
            self._use_global_timestamp = True
            timestamp_ns = global_timestamp_ns
            timestamp_source = "global"
        else:
            return None

        frame = PicoBodyFrame(
            timestamp_ns=timestamp_ns,
            body_poses=self._xrt.get_body_joints_pose(),
        )
        if timestamp_source != self._timestamp_source:
            print(f"[PicoXrt] timestamp source: {timestamp_source}")
            self._timestamp_source = timestamp_source
        return frame

    def close(self) -> None:
        close_error: Exception | None = None
        try:
            if self._initialized and self._xrt is not None:
                close = getattr(self._xrt, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            close_error = exc
        finally:
            self._initialized = False
            self._started = False
            self._last_body_timestamp_ns = 0
            self._last_global_timestamp_ns = 0
            self._use_global_timestamp = False
            self._timestamp_source = None

            process = self._service_process
            self._service_process = None
            if process is not None and (
                not hasattr(process, "poll") or process.poll() is None
            ):
                process.terminate()
            if process is not None:
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)

        if close_error is not None:
            raise RuntimeError("xrobotoolkit_sdk.close() failed") from close_error

    def __enter__(self) -> PicoXrtSource:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
