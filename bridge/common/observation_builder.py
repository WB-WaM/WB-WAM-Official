from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .state_layouts import (
    STATE_DIM_GRAVITY,
    STATE_DIM_QUAT,
    STATE_LAYOUT_GRAVITY,
    STATE_LAYOUT_QUAT,
    base_quat_to_gravity,
    normalize_state_layout,
    state_dim_for_layout,
)

STATE_DIM = STATE_DIM_QUAT
ACTION_DIM = 104
DEPTH_MAX_MM = 5000.0


BODY_STATE_FIELDS_QUAT = (
    ("obs", "base_quat", 4),
    ("obs", "base_ang_vel"),
    ("obs", "base_accel"),
    ("obs", "body_q", 29),
    ("obs", "body_dq", 29),
)

BODY_STATE_FIELDS_GRAVITY = (
    ("obs", "base_gravity", 3),
    ("obs", "base_ang_vel", 3),
    ("obs", "base_accel", 3),
    ("obs", "body_q", 29),
    ("obs", "body_dq", 29),
)


def _nested(frame: Mapping[str, Any], *keys: str) -> Any:
    value: Any = frame
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _as_vector(value: Any, *, field: str, expected_dim: int | None = None) -> np.ndarray:
    if value is None:
        raise ValueError(f"missing {field}")
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if expected_dim is not None and arr.size != expected_dim:
        raise ValueError(f"{field} dim {arr.size}, expected {expected_dim}")
    if arr.size == 0:
        raise ValueError(f"{field} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field} contains NaN/Inf")
    return arr


def _build_body_parts(frame: Mapping[str, Any], *, state_layout: str) -> list[np.ndarray]:
    layout = normalize_state_layout(state_layout)
    parts: list[np.ndarray] = []
    if layout == STATE_LAYOUT_GRAVITY:
        base_gravity = _nested(frame, "obs", "base_gravity")
        if base_gravity is None:
            parts.append(base_quat_to_gravity(_nested(frame, "obs", "base_quat")))
        else:
            parts.append(_as_vector(base_gravity, field="obs.base_gravity", expected_dim=3))
        fields = BODY_STATE_FIELDS_GRAVITY[1:]
    else:
        fields = BODY_STATE_FIELDS_QUAT

    for field in fields:
        parent, key = field[:2]
        expected_dim = field[2] if len(field) > 2 else None
        parts.append(_as_vector(_nested(frame, parent, key), field=f"{parent}.{key}", expected_dim=expected_dim))
    return parts


def build_state_from_frame(frame: Mapping[str, Any], *, state_layout: str = STATE_LAYOUT_QUAT) -> np.ndarray:
    layout = normalize_state_layout(state_layout)
    parts = _build_body_parts(frame, state_layout=layout)
    parts.append(
        _as_vector(
            _nested(frame, "hand_feedback", "left_wuji_qpos_actual"),
            field="hand_feedback.left_wuji_qpos_actual",
            expected_dim=20,
        )
    )
    parts.append(
        _as_vector(
            _nested(frame, "hand_feedback", "right_wuji_qpos_actual"),
            field="hand_feedback.right_wuji_qpos_actual",
            expected_dim=20,
        )
    )
    state = np.concatenate(parts).astype(np.float32, copy=False)
    expected_dim = state_dim_for_layout(layout)
    if state.size != expected_dim:
        raise ValueError(f"state dim {state.size}, expected {expected_dim}")
    return state


def build_fake_state(*, state_layout: str = STATE_LAYOUT_QUAT) -> np.ndarray:
    layout = normalize_state_layout(state_layout)
    state = np.zeros((state_dim_for_layout(layout),), dtype=np.float32)
    if layout == STATE_LAYOUT_GRAVITY:
        state[:3] = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        state[0] = 1.0  # base quaternion w
    return state


def append_last_action_to_state(
    state: np.ndarray,
    *,
    last_action: np.ndarray | None,
    action_dim: int = ACTION_DIM,
) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if last_action is None:
        action_arr = np.zeros((action_dim,), dtype=np.float32)
    else:
        action_arr = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if action_arr.size != action_dim:
            raise ValueError(f"last_action dim {action_arr.size}, expected {action_dim}")
        if not np.all(np.isfinite(action_arr)):
            raise ValueError("last_action contains NaN/Inf")
    return np.concatenate([state_arr, action_arr]).astype(np.float32, copy=False)


def _normalize_rgb_image(image: np.ndarray, *, field: str) -> np.ndarray:
    image_arr = np.asarray(image)
    if image_arr.ndim != 3 or image_arr.shape[2] != 3:
        raise ValueError(f"{field} must be HxWx3, got {image_arr.shape}")
    if image_arr.dtype != np.uint8:
        image_arr = np.clip(image_arr, 0, 255).astype(np.uint8)
    return image_arr


def _normalize_state(state: np.ndarray, *, expected_state_dim: int | None) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if expected_state_dim is not None and state_arr.size != expected_state_dim:
        raise ValueError(f"state dim {state_arr.size}, expected {expected_state_dim}")
    return state_arr


def encode_depth_to_gray_rgb(depth: np.ndarray, *, max_depth_mm: float = DEPTH_MAX_MM) -> np.ndarray:
    if max_depth_mm <= 0:
        raise ValueError("max_depth_mm must be > 0")
    depth_arr = np.asarray(depth)
    if depth_arr.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth_arr.shape}")
    depth_mm = depth_arr.astype(np.float32, copy=False)
    finite = np.isfinite(depth_mm)
    if not finite.all():
        depth_mm = depth_mm.copy()
        depth_mm[~finite] = 0.0
    # Some sources store depth in meters as float; RealSense remote frames use uint16 millimeters.
    if np.issubdtype(depth_arr.dtype, np.floating) and finite.any() and float(depth_mm[finite].max()) <= 20.0:
        depth_mm = depth_mm * 1000.0
    gray = np.rint(np.clip(depth_mm, 0.0, max_depth_mm) / max_depth_mm * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def build_observation(
    *,
    image: np.ndarray,
    state: np.ndarray,
    prompt: str,
    expected_state_dim: int | None = STATE_DIM,
) -> dict[str, Any]:
    image_arr = _normalize_rgb_image(image, field="image")
    state_arr = _normalize_state(state, expected_state_dim=expected_state_dim)
    return {
        "observation/image": image_arr,
        "states": state_arr,
        "prompt": str(prompt),
    }


def build_rgbd_observation(
    *,
    image: np.ndarray,
    depth: np.ndarray,
    state: np.ndarray,
    prompt: str,
    expected_state_dim: int | None = STATE_DIM_GRAVITY,
    depth_max_mm: float = DEPTH_MAX_MM,
) -> dict[str, Any]:
    image_arr = _normalize_rgb_image(image, field="image")
    depth_image = encode_depth_to_gray_rgb(depth, max_depth_mm=depth_max_mm)
    if depth_image.shape[:2] != image_arr.shape[:2]:
        raise ValueError(f"depth shape {depth_image.shape[:2]} does not match image shape {image_arr.shape[:2]}")
    state_arr = _normalize_state(state, expected_state_dim=expected_state_dim)
    return {
        "observation/image": image_arr,
        "observation/depth_image": depth_image,
        "states": state_arr,
        "prompt": str(prompt),
    }


def load_episode_frames(episode_dir: Path) -> list[dict[str, Any]]:
    data_path = episode_dir / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"data.json not found: {data_path}")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        return payload["frames"]
    raise ValueError("data.json must be a list or a dict with frames")


def _camera_record(frame: Mapping[str, Any], camera: str) -> Mapping[str, Any]:
    cameras = frame.get("cameras")
    if not isinstance(cameras, Mapping):
        raise ValueError("frame missing cameras")
    key = str(frame.get("primary_camera", camera)) if camera == "primary" else camera
    record = cameras.get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"frame missing camera {key}")
    return record


def load_episode_image(episode_dir: Path, frame: Mapping[str, Any], *, camera: str = "primary") -> np.ndarray:
    import cv2

    record = _camera_record(frame, camera)
    color_rel = record.get("color")
    if not isinstance(color_rel, str):
        raise ValueError("camera record missing color path")
    image_path = episode_dir / color_rel
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_episode_depth_image(
    episode_dir: Path, frame: Mapping[str, Any], *, camera: str = "primary"
) -> np.ndarray:
    from collector.sonic.core.depth_io import load_depth

    record = _camera_record(frame, camera)
    depth_rel = record.get("depth")
    if not isinstance(depth_rel, str):
        raise ValueError("camera record missing depth path")
    depth_path = episode_dir / depth_rel
    if not depth_path.exists():
        raise FileNotFoundError(f"failed to read depth: {depth_path}")
    return load_depth(depth_path)
