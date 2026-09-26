from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from collector.sonic.core.remote_camera_protocol import CameraServiceProfile

from .camera_client import RemoteCameraClient


def derive_camera_key_from_image_key(image_key: str | None) -> str | None:
    if not image_key:
        return None
    tail = str(image_key).strip().split(".")[-1].split("/")[-1]
    return tail or None


def camera_sort_key(camera: CameraServiceProfile) -> tuple[int, str]:
    text = f"{camera.key} {camera.model}".lower()
    if "d435" in text:
        rank = 0
    elif "d455" in text:
        rank = 1
    else:
        rank = 2
    return rank, str(camera.key)


def default_camera_index(cameras: Sequence[CameraServiceProfile], default_key: str) -> int:
    for index, camera in enumerate(cameras):
        if camera.key == default_key or camera.serial == default_key:
            return index
    return 0


def format_camera(camera: CameraServiceProfile, index: int) -> str:
    return (
        f"  {index}. key={camera.key} model={camera.model} serial={camera.serial} "
        f"profile={camera.width}x{camera.height}@{camera.fps}"
    )


def select_camera_from_list(
    cameras: Sequence[CameraServiceProfile],
    *,
    default_key: str,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], Any] = print,
) -> str:
    if not cameras:
        return default_key

    sorted_cameras = sorted(cameras, key=camera_sort_key)
    default_index = default_camera_index(sorted_cameras, default_key)

    print_fn("Remote RealSense cameras:")
    for index, camera in enumerate(sorted_cameras, start=1):
        print_fn(format_camera(camera, index))

    while True:
        selected_default = sorted_cameras[default_index]
        prompt = f"Select Policy camera [default: {selected_default.key}, #{default_index + 1}]: "
        choice = input_fn(prompt).strip()
        if not choice:
            selected = selected_default
            break

        selected = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_cameras):
                selected = sorted_cameras[idx]
        if selected is None:
            for camera in sorted_cameras:
                if choice in {camera.key, camera.serial}:
                    selected = camera
                    break
        if selected is not None:
            break
        print_fn("[WARNING] invalid camera selection")

    print_fn(f"[INFO] selected Policy camera: key={selected.key} model={selected.model} serial={selected.serial}")
    return selected.key


def resolve_camera_key(
    *,
    endpoint: str,
    default_key: str,
    timeout_ms: int,
    interactive: bool,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], Any] = print,
) -> str:
    if not interactive:
        print_fn(f"[INFO] using Policy camera key={default_key!r} (non-interactive)")
        return default_key

    print_fn(f"[INFO] listing remote cameras from {endpoint}")
    try:
        cameras = RemoteCameraClient.list_cameras(endpoint=endpoint, timeout_ms=timeout_ms)
    except Exception as exc:
        print_fn(f"[WARNING] failed to list remote cameras; using camera_key={default_key!r}: {exc}")
        return default_key
    if not cameras:
        print_fn(f"[WARNING] remote camera list is empty; using camera_key={default_key!r}")
        return default_key
    return select_camera_from_list(
        cameras,
        default_key=default_key,
        input_fn=input_fn,
        print_fn=print_fn,
    )
