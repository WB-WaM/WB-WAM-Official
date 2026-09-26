from __future__ import annotations

import re

from .schema import CameraInfo


try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def normalize_camera_key(model: str) -> str:
    upper = model.upper()
    if "D455" in upper:
        return "d455"
    if "D435" in upper:
        return "d435"

    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return slug or "camera"


def _safe_get_info(device, key, default: str = "") -> str:
    try:
        return device.get_info(key)
    except Exception:
        return default


def discover_cameras() -> list[CameraInfo]:
    if rs is None:
        raise RuntimeError("pyrealsense2 is required for camera discovery")

    context = rs.context()
    devices = list(context.query_devices())
    cameras: list[CameraInfo] = []
    used_keys: dict[str, int] = {}

    for device in devices:
        model = _safe_get_info(device, rs.camera_info.name, "Unknown RealSense")
        serial = _safe_get_info(device, rs.camera_info.serial_number, "")
        base_key = normalize_camera_key(model)
        suffix_index = used_keys.get(base_key, 0)
        used_keys[base_key] = suffix_index + 1
        key = base_key if suffix_index == 0 else f"{base_key}_{suffix_index + 1}"
        cameras.append(
            CameraInfo(
                key=key,
                model=model,
                serial=serial,
                product_line=_safe_get_info(device, rs.camera_info.product_line, ""),
                firmware_version=_safe_get_info(device, rs.camera_info.firmware_version, ""),
                usb_type=_safe_get_info(device, rs.camera_info.usb_type_descriptor, ""),
            )
        )

    return cameras


def resolve_camera_selector(cameras: list[CameraInfo], selector: str) -> CameraInfo:
    selector = selector.strip()
    if not selector:
        raise ValueError("camera selector cannot be empty")

    if selector.isdigit():
        index = int(selector) - 1
        if 0 <= index < len(cameras):
            return cameras[index]

    for camera in cameras:
        if selector in {camera.key, camera.serial}:
            return camera

    raise ValueError(f"unknown camera selector: {selector}")


def format_camera_lines(cameras: list[CameraInfo]) -> list[str]:
    lines: list[str] = []
    for index, camera in enumerate(cameras, start=1):
        lines.append(
            f"{index}. key={camera.key} model={camera.model} serial={camera.serial or 'N/A'}"
        )
    return lines
