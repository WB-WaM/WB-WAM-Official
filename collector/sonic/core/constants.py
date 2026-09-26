from __future__ import annotations

DEFAULT_CAPTURE_FPS = 20
SUPPORTED_CAPTURE_FPS = (20, 30, 50)
SUPPORTED_CAMERA_FPS = (20, 30, 60)


def normalize_capture_fps(value: int | str) -> int:
    fps = int(value)
    if fps not in SUPPORTED_CAPTURE_FPS:
        supported = ", ".join(str(item) for item in SUPPORTED_CAPTURE_FPS)
        raise ValueError(f"capture_fps must be one of {supported}, got {fps}")
    return fps


def normalize_camera_fps(value: int | str) -> int:
    fps = int(value)
    if fps not in SUPPORTED_CAMERA_FPS:
        supported = ", ".join(str(item) for item in SUPPORTED_CAMERA_FPS)
        raise ValueError(f"camera_fps must be one of {supported}, got {fps}")
    return fps


def default_camera_fps_for_capture(capture_fps: int | str) -> int:
    capture = normalize_capture_fps(capture_fps)
    return 60 if capture == 50 else capture


def frame_interval_ns_for_fps(fps: int) -> int:
    return int(round(1e9 / normalize_capture_fps(fps)))


def half_delay_ns_for_fps(fps: int) -> int:
    delay = 1.0 / normalize_capture_fps(fps)
    return int(round((delay / 2.0) * 1e9))


FREQ = DEFAULT_CAPTURE_FPS
DELAY = 1 / FREQ
FRAME_INTERVAL_NS = frame_interval_ns_for_fps(FREQ)
HALF_DELAY_NS = half_delay_ns_for_fps(FREQ)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = default_camera_fps_for_capture(FREQ)
