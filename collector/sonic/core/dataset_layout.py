from __future__ import annotations

import json
from pathlib import Path

from .schema import CameraInfo


class DatasetLayout:
    def __init__(self, output_root: Path, task_name: str):
        self.output_root = output_root
        self.task_name = task_name
        self.task_root = output_root / task_name
        self.task_root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.task_root / "metadata.json"

    def write_metadata(
        self,
        *,
        robot_name: str,
        primary_camera: str,
        connected_cameras: list[CameraInfo],
        capture_fps: int,
        record_depth: bool,
        notes: str,
        camera_fps: int | None = None,
        capture_mode: str = "collector_timer",
        clock_source: str = "collector_monotonic_timer",
        camera_alignment: str = "nearest_with_hold",
        camera_backend: str = "local_realsense",
        camera_endpoint: str = "",
        capture_transport: str = "local_realsense",
        hand_control_mode: str = "gesture_wuji",
        hand_backend: str = "local_wuji",
        hand_status_endpoint: str = "",
        hand_command_source: str = "pose_topic",
        defer_depth_compression: bool = False,
    ) -> Path:
        payload = {
            "task_name": self.task_name,
            "robot_name": robot_name,
            "primary_camera": primary_camera,
            "connected_cameras": [camera.to_record() for camera in connected_cameras],
            "capture_fps": capture_fps,
            "camera_fps": capture_fps if camera_fps is None else camera_fps,
            "capture_mode": capture_mode,
            "clock_source": clock_source,
            "camera_alignment": camera_alignment,
            "record_depth": record_depth,
            "notes": notes,
            "camera_backend": camera_backend,
            "camera_endpoint": camera_endpoint,
            "capture_transport": capture_transport,
            "hand_control_mode": hand_control_mode,
            "hand_backend": hand_backend,
            "hand_status_endpoint": hand_status_endpoint,
            "hand_command_source": hand_command_source,
            "defer_depth_compression": defer_depth_compression,
        }
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        return self.metadata_path

    def next_episode_dir(self) -> Path:
        max_index = -1
        for child in self.task_root.glob("episode_*"):
            if not child.is_dir():
                continue
            try:
                max_index = max(max_index, int(child.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return self.task_root / f"episode_{max_index + 1:06d}"
