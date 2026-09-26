#!/usr/bin/env python3
"""Convert HGPT collector episodes to the WB-WAM/pi0.5 LeRobot v3 layout.

The converted dataset deliberately carries two compatible action views:

* HGPT reference action (76D semantic):
  ``action[0:36] + action[64:104]`` =
  ``qpos36(root xyz, root quat wxyz, grouped G1 joints29) + Wuji hands40``.
* Existing physical WB action (72D semantic):
  ``action[104:133] + action[133:136] + action[64:104]`` =
  ``IsaacLab-order body delta29 + root roll/pitch/yaw-rate3 + Wuji hands40``.

The source HGPT reference is ``human_derived.g1_qpos``, matching ``motion.npz``
and the input accepted by HGPT's online/offline reference path. Each episode is
rigidly aligned with HGPT's production first-frame rule: root XY and yaw become
zero, while absolute root height, roll/pitch, joint angles, and relative motion
are preserved.

Rows follow the existing WB training convention: image/state are from frame t,
and action is from frame t+1. A T-frame source episode therefore produces T-1
rows.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import imageio.v2 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collector.humanoid_gpt.processing import lerobot_helpers  # noqa: E402

from bridge.sonic.joint_order import (  # noqa: E402
    DEFAULT_ANGLES_MUJOCO,
    MUJOCO_TO_ISAACLAB,
)
from tracker.humanoid_gpt.deploy.reference_alignment import (  # noqa: E402
    align_qpos_first_frame,
)

FPS = 20
CHUNKS_SIZE = 1000
RAW_STATE_DIM = 146
RAW_ACTION_DIM = 136
HGPT_QPOS_DIM = 36
HAND_DIM = 20
HGPT_SEMANTIC_DIM = 76
PHYSICAL_SEMANTIC_DIM = 72
CODEBASE_VERSION = "hgpt_reference_lerobot_v3_20hz_v3"
CODEBASE_VERSION_PREFIX = "hgpt_reference_lerobot_v3_20hz_v"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets/humanoid_gpt_lerobot_v3_20hz"
PLACEHOLDER_TASK_NAMES = {
    "",
    "example",
    "example task",
    "hgpt test",
    "test",
    "test task",
}

STATE_LAYOUT = {
    "gravity": "states[0:3]",
    "base_angular_velocity": "states[3:6]",
    "base_acceleration": "states[6:9]",
    "body_q_delta_isaaclab": "states[9:38]",
    "body_dq_isaaclab": "states[38:67]",
    "left_wuji_actual": "states[67:87]",
    "right_wuji_actual": "states[87:107]",
    "actual_root_roll_pitch_yaw_rate": "states[107:110]",
    "current_hgpt_reference_qpos": "states[110:146]",
}

ACTION_LAYOUT = {
    "next_hgpt_reference_qpos": "action[0:36]",
    "padding": "action[36:64]",
    "next_left_wuji_target": "action[64:84]",
    "next_right_wuji_target": "action[84:104]",
    "next_body_q_delta_isaaclab": "action[104:133]",
    "next_reference_root_roll_pitch_yaw_rate": "action[133:136]",
}

HGPT_TRAINING_CONTRACT = {
    "raw_state_dim": RAW_STATE_DIM,
    "raw_action_dim": RAW_ACTION_DIM,
    "state_dim": HGPT_SEMANTIC_DIM,
    "action_dim": HGPT_SEMANTIC_DIM,
    "proprio_dim": 96,
    "action_target_dim": 96,
    "state_slices": ["states[110:146]", "states[67:107]"],
    "action_slices": ["action[0:36]", "action[64:104]"],
    "semantic_state": "current aligned qpos36 + measured Wuji hands40",
    "semantic_action": "next aligned qpos36 + commanded Wuji hands40",
}

PHYSICAL_TRAINING_CONTRACT = {
    "raw_state_dim": RAW_STATE_DIM,
    "raw_action_dim": RAW_ACTION_DIM,
    "state_dim": PHYSICAL_SEMANTIC_DIM,
    "action_dim": PHYSICAL_SEMANTIC_DIM,
    "proprio_dim": 96,
    "action_target_dim": 96,
    "state_slices": [
        "states[9:38]",
        "states[107:110]",
        "states[67:87]",
        "states[87:107]",
    ],
    "action_slices": [
        "action[104:133]",
        "action[133:136]",
        "action[64:84]",
        "action[84:104]",
    ],
    "semantic_state": "actual body29 + actual root3 + measured Wuji hands40",
    "semantic_action": "next reference body29 + reference root3 + commanded Wuji hands40",
}


@dataclass(frozen=True)
class SourceEpisode:
    path: str
    source_task_dir: str
    source_episode: str
    episode_index: int
    task_index: int
    task: str
    source_fps: float
    source_frames: int
    rows: int
    global_start_index: int


@dataclass
class PreparedEpisode:
    frames: list[dict[str, Any]]
    aligned_qpos: np.ndarray
    root_xyz_delta: np.ndarray
    reference_root3: np.ndarray
    left_target: np.ndarray
    right_target: np.ndarray
    left_target_imputed: np.ndarray
    right_target_imputed: np.ndarray
    left_actual: np.ndarray
    right_actual: np.ndarray
    left_actual_imputed: np.ndarray
    right_actual_imputed: np.ndarray
    camera_imputed: np.ndarray
    source_initial_root_xyz: np.ndarray
    source_initial_yaw: float
    camera_source_stale: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    camera_source_reused: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    left_target_source_invalid: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    right_target_source_invalid: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(dict(row), handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
    temporary.replace(path)


def _finite_vector(value: Any, dimension: int, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (dimension,):
        raise ValueError(f"{field} has shape {array.shape}, expected ({dimension},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} contains non-finite values")
    return array


def _strict_bool(value: Any, field: str) -> bool:
    """Parse collector booleans without Python's truthy-container ambiguity."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        sequence = np.asarray(value, dtype=object).reshape(-1)
        if sequence.shape == (1,):
            return _strict_bool(sequence[0], field)
    raise ValueError(f"{field} must be a bool or a single-element bool list")


def _normalized_wxyz(value: Any, field: str) -> np.ndarray:
    quaternion = _finite_vector(value, 4, field).astype(np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        raise ValueError(f"{field} has near-zero norm")
    return (quaternion / norm).astype(np.float32)


def _quat_roll_pitch_yaw_wxyz(quaternion: Any, field: str) -> tuple[float, float, float]:
    w, x, y, z = _normalized_wxyz(quaternion, field).astype(np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def _continuous_quaternions_wxyz(qpos: np.ndarray) -> np.ndarray:
    output = np.asarray(qpos, dtype=np.float32).copy()
    quaternions = output[:, 3:7].astype(np.float64)
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1.0e-8) or not np.all(np.isfinite(norms)):
        raise ValueError("reference trajectory contains an invalid root quaternion")
    quaternions /= norms[:, None]
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    output[:, 3:7] = quaternions.astype(np.float32)
    return output


def align_hgpt_qpos(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply HGPT's production first-frame SE(2) alignment.

    The returned qpos keeps absolute root Z. The companion ``xyz_delta`` is
    fully relative to frame zero and is exported as an auxiliary diagnostic,
    but it must not replace qpos root Z when driving HGPT.
    """

    source = np.asarray(qpos, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != HGPT_QPOS_DIM or len(source) < 2:
        raise ValueError(f"qpos must have shape (T,{HGPT_QPOS_DIM}) with T>=2")
    if not np.all(np.isfinite(source)):
        raise ValueError("qpos contains non-finite values")
    source[0, :3].copy()
    initial_yaw = _quat_roll_pitch_yaw_wxyz(source[0, 3:7], "qpos[0].root_quat")[2]
    aligned = align_qpos_first_frame(
        source,
        target_xy=(0.0, 0.0),
        target_yaw=0.0,
    )
    aligned = _continuous_quaternions_wxyz(aligned)
    xyz_delta = aligned[:, :3].copy()
    xyz_delta[:, 2] -= aligned[0, 2]
    return aligned, xyz_delta.astype(np.float32), initial_yaw


def _repair_camera_frames(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Validate camera flags while preserving every row's own media paths."""

    preserved = copy.deepcopy(list(frames))
    imputed = np.zeros((len(preserved),), dtype=bool)
    for frame_index, frame in enumerate(preserved):
        cameras = frame.get("cameras")
        if not isinstance(cameras, dict) or not cameras:
            raise ValueError(f"frame {frame_index} has no camera records")
        for camera_key, camera in cameras.items():
            if not isinstance(camera, dict):
                raise ValueError(f"frame {frame_index} camera {camera_key} is not a mapping")
            _strict_bool(
                camera.get("stale", False),
                f"frame {frame_index} camera {camera_key}.stale",
            )
            _strict_bool(
                camera.get("reused", False),
                f"frame {frame_index} camera {camera_key}.reused",
            )
    return preserved, imputed


def _camera_source_flags(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    stale = np.zeros((len(frames),), dtype=bool)
    reused = np.zeros((len(frames),), dtype=bool)
    for frame_index, frame in enumerate(frames):
        cameras = frame.get("cameras")
        if not isinstance(cameras, Mapping) or not cameras:
            raise ValueError(f"frame {frame_index} has no camera records")
        for camera_key, camera in cameras.items():
            if not isinstance(camera, Mapping):
                raise ValueError(f"frame {frame_index} camera {camera_key} is not a mapping")
            stale[frame_index] |= _strict_bool(
                camera.get("stale", False),
                f"frame {frame_index} camera {camera_key}.stale",
            )
            reused[frame_index] |= _strict_bool(
                camera.get("reused", False),
                f"frame {frame_index} camera {camera_key}.reused",
            )
    return stale, reused


def _load_hand_targets(
    values: Sequence[Any],
    valid: Sequence[Any],
    *,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) != len(valid):
        raise ValueError(f"{field}: values/valid lengths differ")
    targets = np.asarray(
        [_finite_vector(value, HAND_DIM, f"{field}[{index}]") for index, value in enumerate(values)],
        dtype=np.float32,
    )
    source_invalid = np.asarray(
        [not _strict_bool(flag, f"{field}_valid[{index}]") for index, flag in enumerate(valid)],
        dtype=bool,
    )
    return targets, source_invalid


def _load_hand_actual(
    values: Sequence[Any],
    valid: Sequence[Any],
    *,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) != len(valid):
        raise ValueError(f"{field}: values/valid lengths differ")
    actual = np.asarray(
        [_finite_vector(value, HAND_DIM, f"{field}[{index}]") for index, value in enumerate(values)],
        dtype=np.float32,
    )
    invalid = np.asarray(
        [not _strict_bool(flag, f"{field}_valid[{index}]") for index, flag in enumerate(valid)],
        dtype=bool,
    )
    if invalid.any():
        indices = np.flatnonzero(invalid).astype(int).tolist()
        raise ValueError(f"{field}: invalid actual hand frames: {indices[:20]}")
    return actual, invalid


def _projected_gravity_wxyz(quaternion: Any, field: str) -> np.ndarray:
    w, x, y, z = _normalized_wxyz(quaternion, field).astype(np.float64)
    return np.asarray(
        [
            2.0 * (w * y - x * z),
            -2.0 * (y * z + w * x),
            2.0 * (x * x + y * y) - 1.0,
        ],
        dtype=np.float32,
    )


def _validated_projected_gravity(obs: Mapping[str, Any], field: str) -> np.ndarray:
    projected = _projected_gravity_wxyz(obs.get("base_quat"), f"{field}.base_quat")
    recorded = _finite_vector(obs.get("base_gravity"), 3, f"{field}.base_gravity")
    error = float(np.max(np.abs(projected - recorded)))
    if error > 1.0e-5:
        raise ValueError(f"{field}: projected/recorded gravity max error {error:.3g} > 1e-5")
    return projected


def _load_frames(episode_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(episode_dir / "data.json")
    frames = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError(f"{episode_dir}/data.json must contain at least two frames")
    expected_indices = np.arange(len(frames), dtype=np.int64)
    actual_indices = np.asarray(
        [frame.get("frame_index", -1) for frame in frames],
        dtype=np.int64,
    )
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(f"{episode_dir}: frame_index is not contiguous from zero")
    times = np.asarray([frame.get("time_ns", -1) for frame in frames], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{episode_dir}: time_ns is not strictly increasing")
    expected_dt_ns = int(round(1.0e9 / FPS))
    if np.any(np.diff(times) != expected_dt_ns):
        raise ValueError(f"{episode_dir}: time_ns is not strict {FPS} Hz")
    return frames


def prepare_episode(episode_dir: Path) -> PreparedEpisode:
    raw_frames = _load_frames(episode_dir)
    frames, camera_imputed = _repair_camera_frames(raw_frames)
    camera_source_stale, camera_source_reused = _camera_source_flags(frames)

    stale_fields = ("pose_stale", "raw_stale", "robot_stale", "hand_stale")
    for index, frame in enumerate(frames):
        stale = [
            field_name
            for field_name in stale_fields
            if _strict_bool(
                frame.get(field_name, False),
                f"frame {index} {field_name}",
            )
        ]
        if stale:
            raise ValueError(f"{episode_dir}: frame {index} stale streams: {stale}")

    qpos = np.asarray(
        [
            _finite_vector(
                frame.get("human_derived", {}).get("g1_qpos"),
                HGPT_QPOS_DIM,
                f"frame {index} human_derived.g1_qpos",
            )
            for index, frame in enumerate(frames)
        ],
        dtype=np.float32,
    )
    body_valid = [
        _strict_bool(
            frame.get("human_derived", {}).get("body_valid", False),
            f"frame {index} human_derived.body_valid",
        )
        for index, frame in enumerate(frames)
    ]
    if not all(body_valid):
        invalid = [index for index, valid in enumerate(body_valid) if not valid]
        raise ValueError(f"{episode_dir}: invalid body reference frames: {invalid[:20]}")

    for index, frame in enumerate(frames):
        obs = frame.get("obs")
        if not isinstance(obs, Mapping):
            raise ValueError(f"frame {index} is missing obs")
        _validated_projected_gravity(obs, f"frame {index} obs")

    motion_path = episode_dir / "motion.npz"
    if motion_path.is_file():
        with np.load(motion_path, allow_pickle=False) as motion:
            motion_qpos = np.asarray(motion["qpos"], dtype=np.float32)
        if motion_qpos.shape != qpos.shape or not np.allclose(
            motion_qpos,
            qpos,
            atol=1.0e-6,
        ):
            raise ValueError(f"{episode_dir}: motion.npz qpos does not match data.json")

    aligned_qpos, root_xyz_delta, initial_yaw = align_hgpt_qpos(qpos)
    rpy = np.asarray(
        [
            _quat_roll_pitch_yaw_wxyz(quaternion, f"aligned_qpos[{index}].quat")
            for index, quaternion in enumerate(aligned_qpos[:, 3:7])
        ],
        dtype=np.float64,
    )
    unwrapped_yaw = np.unwrap(rpy[:, 2])
    edge_order = 2 if len(frames) >= 3 else 1
    yaw_rate = np.gradient(
        unwrapped_yaw,
        1.0 / FPS,
        edge_order=edge_order,
    )
    reference_root3 = np.column_stack([rpy[:, :2], yaw_rate]).astype(np.float32)

    derived = [frame.get("human_derived", {}) for frame in frames]
    left_target, left_target_source_invalid = _load_hand_targets(
        [item.get("left_wuji_qpos") for item in derived],
        [item.get("left_wuji_qpos_valid", False) for item in derived],
        field="human_derived.left_wuji_qpos",
    )
    right_target, right_target_source_invalid = _load_hand_targets(
        [item.get("right_wuji_qpos") for item in derived],
        [item.get("right_wuji_qpos_valid", False) for item in derived],
        field="human_derived.right_wuji_qpos",
    )
    left_target_imputed = np.zeros((len(frames),), dtype=bool)
    right_target_imputed = np.zeros((len(frames),), dtype=bool)

    feedback = [frame.get("hand_feedback", {}) for frame in frames]
    left_actual, left_actual_imputed = _load_hand_actual(
        [item.get("left_wuji_qpos_actual") for item in feedback],
        [item.get("left_actual_position_valid", False) for item in feedback],
        field="hand_feedback.left_wuji_qpos_actual",
    )
    right_actual, right_actual_imputed = _load_hand_actual(
        [item.get("right_wuji_qpos_actual") for item in feedback],
        [item.get("right_actual_position_valid", False) for item in feedback],
        field="hand_feedback.right_wuji_qpos_actual",
    )

    return PreparedEpisode(
        frames=frames,
        aligned_qpos=aligned_qpos,
        root_xyz_delta=root_xyz_delta,
        reference_root3=reference_root3,
        left_target=left_target,
        right_target=right_target,
        left_target_imputed=left_target_imputed,
        right_target_imputed=right_target_imputed,
        left_actual=left_actual,
        right_actual=right_actual,
        left_actual_imputed=left_actual_imputed,
        right_actual_imputed=right_actual_imputed,
        camera_imputed=camera_imputed,
        camera_source_stale=camera_source_stale,
        camera_source_reused=camera_source_reused,
        left_target_source_invalid=left_target_source_invalid,
        right_target_source_invalid=right_target_source_invalid,
        source_initial_root_xyz=qpos[0, :3].copy(),
        source_initial_yaw=initial_yaw,
    )


def _body_delta_isaaclab(body_q_grouped: Any, field: str) -> np.ndarray:
    body_q = _finite_vector(body_q_grouped, 29, field)
    return (body_q - DEFAULT_ANGLES_MUJOCO)[MUJOCO_TO_ISAACLAB].astype(np.float32)


def _body_dq_isaaclab(body_dq_grouped: Any, field: str) -> np.ndarray:
    body_dq = _finite_vector(body_dq_grouped, 29, field)
    return body_dq[MUJOCO_TO_ISAACLAB].astype(np.float32)


def build_state(prepared: PreparedEpisode, index: int) -> np.ndarray:
    frame = prepared.frames[index]
    obs = frame.get("obs")
    if not isinstance(obs, Mapping):
        raise ValueError(f"frame {index} is missing obs")
    gravity = _validated_projected_gravity(obs, f"frame {index} obs")
    angular_velocity = _finite_vector(
        obs.get("base_ang_vel"),
        3,
        f"frame {index} obs.base_ang_vel",
    )
    acceleration = _finite_vector(
        obs.get("base_accel"),
        3,
        f"frame {index} obs.base_accel",
    )
    body_q = _body_delta_isaaclab(
        obs.get("body_q"),
        f"frame {index} obs.body_q",
    )
    body_dq = _body_dq_isaaclab(
        obs.get("body_dq"),
        f"frame {index} obs.body_dq",
    )
    roll, pitch, _yaw = _quat_roll_pitch_yaw_wxyz(
        obs.get("base_quat"),
        f"frame {index} obs.base_quat",
    )
    actual_root3 = np.asarray(
        [roll, pitch, angular_velocity[2]],
        dtype=np.float32,
    )
    state = np.concatenate(
        [
            gravity,
            angular_velocity,
            acceleration,
            body_q,
            body_dq,
            prepared.left_actual[index],
            prepared.right_actual[index],
            actual_root3,
            prepared.aligned_qpos[index],
        ],
    ).astype(np.float32)
    if state.shape != (RAW_STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError(f"frame {index} produced invalid state shape {state.shape}")
    return state


def build_action(prepared: PreparedEpisode, target_index: int) -> np.ndarray:
    qpos = prepared.aligned_qpos[target_index]
    action = np.zeros((RAW_ACTION_DIM,), dtype=np.float32)
    action[0:36] = qpos
    action[64:84] = prepared.left_target[target_index]
    action[84:104] = prepared.right_target[target_index]
    action[104:133] = _body_delta_isaaclab(
        qpos[7:36],
        f"aligned_qpos[{target_index}].joints",
    )
    action[133:136] = prepared.reference_root3[target_index]
    if not np.all(np.isfinite(action)):
        raise ValueError(f"frame {target_index} produced a non-finite action")
    return action


def state_mask() -> np.ndarray:
    return np.zeros((RAW_STATE_DIM,), dtype=bool)


def action_mask() -> np.ndarray:
    mask = np.zeros((RAW_ACTION_DIM,), dtype=bool)
    mask[36:64] = True
    return mask


def _parquet_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("states", pa.list_(pa.float32(), RAW_STATE_DIM), nullable=False),
            pa.field("action", pa.list_(pa.float32(), RAW_ACTION_DIM), nullable=False),
            pa.field(
                "state_mask_110",
                pa.list_(pa.bool_(), RAW_STATE_DIM),
                nullable=False,
            ),
            pa.field(
                "action_mask_136",
                pa.list_(pa.bool_(), RAW_ACTION_DIM),
                nullable=False,
            ),
            pa.field(
                "hgpt.reference_qpos",
                pa.list_(pa.float32(), HGPT_QPOS_DIM),
                nullable=False,
            ),
            pa.field(
                "hgpt.root_xyz_delta",
                pa.list_(pa.float32(), 3),
                nullable=False,
            ),
            pa.field("quality.camera_imputed", pa.bool_(), nullable=False),
            pa.field("quality.camera_source_stale", pa.bool_(), nullable=False),
            pa.field("quality.camera_source_reused", pa.bool_(), nullable=False),
            pa.field("quality.left_hand_state_imputed", pa.bool_(), nullable=False),
            pa.field("quality.right_hand_state_imputed", pa.bool_(), nullable=False),
            pa.field("quality.left_hand_action_imputed", pa.bool_(), nullable=False),
            pa.field("quality.right_hand_action_imputed", pa.bool_(), nullable=False),
            pa.field(
                "quality.left_hand_action_source_invalid",
                pa.bool_(),
                nullable=False,
            ),
            pa.field(
                "quality.right_hand_action_source_invalid",
                pa.bool_(),
                nullable=False,
            ),
            pa.field("timestamp", pa.float32(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("index", pa.int64(), nullable=False),
            pa.field("task_index", pa.int64(), nullable=False),
            pa.field("next.done", pa.bool_(), nullable=False),
        ]
    )


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=_parquet_schema())
    pq.write_table(table, path, compression="snappy")


def _episode_paths(
    output_root: Path,
    episode_index: int,
    chunks_size: int,
) -> tuple[Path, Path, Path]:
    chunk = episode_index // chunks_size
    return (
        output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet",
        output_root / "videos" / f"chunk-{chunk:03d}" / "primary" / f"episode_{episode_index:06d}.mp4",
        output_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.d455_depth"
        / f"episode_{episode_index:06d}.mp4",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_paths(
    output_root: Path,
    episode_index: int,
    chunks_size: int,
    *,
    include_depth: bool,
) -> dict[str, Path]:
    parquet_path, rgb_path, depth_path = _episode_paths(
        output_root,
        episode_index,
        chunks_size,
    )
    paths = {"parquet": parquet_path, "rgb": rgb_path}
    if include_depth:
        paths["depth"] = depth_path
    return paths


def _output_fingerprints(
    output_root: Path,
    episode_index: int,
    chunks_size: int,
    *,
    include_depth: bool,
) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, path in _output_paths(
        output_root,
        episode_index,
        chunks_size,
        include_depth=include_depth,
    ).items():
        stat = path.stat()
        fingerprints[name] = {
            "path": path.relative_to(output_root).as_posix(),
            "size": stat.st_size,
            "sha256": _sha256(path),
        }
    return fingerprints


def _output_fingerprints_match(
    output_root: Path,
    expected_paths: Mapping[str, Path],
    recorded: Any,
    *,
    verify_sha256: bool,
) -> bool:
    if not isinstance(recorded, Mapping) or set(recorded) != set(expected_paths):
        return False
    for name, path in expected_paths.items():
        fingerprint = recorded.get(name)
        if not isinstance(fingerprint, Mapping):
            return False
        expected_relative = path.relative_to(output_root).as_posix()
        recorded_sha256 = str(fingerprint.get("sha256", ""))
        if (
            fingerprint.get("path") != expected_relative
            or int(fingerprint.get("size", -1)) != path.stat().st_size
            or not re.fullmatch(r"[0-9a-f]{64}", recorded_sha256)
        ):
            return False
        if verify_sha256 and _sha256(path) != recorded_sha256:
            return False
    return True


def _prepare_depth_output(path: Path, *, include_depth: bool) -> None:
    if include_depth:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.unlink(missing_ok=True)


def _record_path(output_root: Path, episode_index: int) -> Path:
    return output_root / "meta" / "progress" / "episodes" / f"episode_{episode_index:06d}.json"


def _completed_record(
    output_root: Path,
    episode: SourceEpisode,
    *,
    chunks_size: int,
    include_depth: bool,
    video_codec: str,
    depth_max_mm: float,
    verify_fingerprints: bool,
) -> dict[str, Any] | None:
    record_path = _record_path(output_root, episode.episode_index)
    expected_paths = _output_paths(
        output_root,
        episode.episode_index,
        chunks_size,
        include_depth=include_depth,
    )
    if not record_path.is_file() or any(
        not path.is_file() or path.stat().st_size == 0 for path in expected_paths.values()
    ):
        return None
    record = _read_json(record_path)
    episode_meta = record.get("episode_meta") or {}
    data_stat = (Path(episode.path) / "data.json").stat()
    recorded_stat = record.get("source_data_json") or {}
    if (
        record.get("status") != "completed"
        or record.get("version") != CODEBASE_VERSION
        or int(record.get("episode_index", -1)) != episode.episode_index
        or str(record.get("source_path", "")) != episode.path
        or int(record.get("task_index", -1)) != episode.task_index
        or str(record.get("task", "")) != episode.task
        or int(episode_meta.get("length", -1)) != episode.rows
        or int(episode_meta.get("source_frames", -1)) != episode.source_frames
        or int(episode_meta.get("dataset_from_index", -1)) != episode.global_start_index
        or bool(record.get("include_depth", False)) != include_depth
        or int(record.get("chunks_size", -1)) != chunks_size
        or str(record.get("video_codec", "")) != video_codec
        or float(record.get("depth_max_mm", -1.0)) != depth_max_mm
        or int(recorded_stat.get("size", -1)) != data_stat.st_size
        or int(recorded_stat.get("mtime_ns", -1)) != data_stat.st_mtime_ns
    ):
        return None
    if not _output_fingerprints_match(
        output_root,
        expected_paths,
        record.get("outputs"),
        verify_sha256=verify_fingerprints,
    ):
        return None
    return record


def _validate_source_media(
    episode_dir: Path,
    prepared: PreparedEpisode,
    *,
    include_depth: bool,
) -> None:
    for index, frame in enumerate(prepared.frames[:-1]):
        image_path = lerobot_helpers.frame_image_path(episode_dir, frame, "primary")
        if not image_path.is_file():
            raise FileNotFoundError(f"frame {index} missing RGB image: {image_path}")
        if include_depth:
            depth_path = lerobot_helpers.frame_depth_path(episode_dir, frame, "primary")
            if not depth_path.is_file():
                raise FileNotFoundError(f"frame {index} missing depth: {depth_path}")


def _convert_one_impl(job: dict[str, Any]) -> dict[str, Any]:
    source = SourceEpisode(**job["source"])
    episode_dir = Path(source.path)
    output_root = Path(job["output_root"])
    chunks_size = int(job["chunks_size"])
    include_depth = bool(job["include_depth"])
    video_codec = str(job["video_codec"])
    depth_max_mm = float(job["depth_max_mm"])

    prepared = prepare_episode(episode_dir)
    _validate_source_media(episode_dir, prepared, include_depth=include_depth)
    if len(prepared.frames) - 1 != source.rows:
        raise ValueError(f"{episode_dir}: discovered rows={source.rows}, prepared rows={len(prepared.frames) - 1}")

    rows: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    image_paths: list[Path] = []
    depth_paths: list[Path] = []
    state_pad_mask = state_mask().tolist()
    action_pad_mask = action_mask().tolist()
    for local_index in range(source.rows):
        target_index = local_index + 1
        state = build_state(prepared, local_index)
        action = build_action(prepared, target_index)
        rows.append(
            {
                "states": state.tolist(),
                "action": action.tolist(),
                # Keep the historical column names expected by both WB loaders.
                # Their actual dimensions are read from meta/info.json.
                "state_mask_110": state_pad_mask,
                "action_mask_136": action_pad_mask,
                "hgpt.reference_qpos": prepared.aligned_qpos[local_index].tolist(),
                "hgpt.root_xyz_delta": prepared.root_xyz_delta[local_index].tolist(),
                "quality.camera_imputed": bool(prepared.camera_imputed[local_index]),
                "quality.camera_source_stale": bool(prepared.camera_source_stale[local_index]),
                "quality.camera_source_reused": bool(prepared.camera_source_reused[local_index]),
                "quality.left_hand_state_imputed": bool(prepared.left_actual_imputed[local_index]),
                "quality.right_hand_state_imputed": bool(prepared.right_actual_imputed[local_index]),
                "quality.left_hand_action_imputed": bool(prepared.left_target_imputed[target_index]),
                "quality.right_hand_action_imputed": bool(prepared.right_target_imputed[target_index]),
                "quality.left_hand_action_source_invalid": bool(prepared.left_target_source_invalid[target_index]),
                "quality.right_hand_action_source_invalid": bool(
                    prepared.right_target_source_invalid[target_index]
                ),
                "timestamp": local_index / float(FPS),
                "frame_index": local_index,
                "episode_index": source.episode_index,
                "index": source.global_start_index + local_index,
                "task_index": source.task_index,
                "next.done": local_index == source.rows - 1,
            }
        )
        states.append(state)
        actions.append(action)
        frame = prepared.frames[local_index]
        image_paths.append(lerobot_helpers.frame_image_path(episode_dir, frame, "primary"))
        if include_depth:
            depth_paths.append(lerobot_helpers.frame_depth_path(episode_dir, frame, "primary"))

    parquet_path, video_path, depth_video_path = _episode_paths(
        output_root,
        source.episode_index,
        chunks_size,
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_depth_output(depth_video_path, include_depth=include_depth)
    with tempfile.TemporaryDirectory(
        prefix=f"{source.source_episode}_",
        dir=str(output_root),
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary_parquet = temporary / "episode.parquet"
        temporary_video = temporary / "episode.mp4"
        _write_parquet(temporary_parquet, rows)
        lerobot_helpers._write_video(  # noqa: SLF001
            temporary_video,
            image_paths,
            fps=FPS,
            codec=video_codec,
        )
        shutil.move(str(temporary_parquet), parquet_path)
        shutil.move(str(temporary_video), video_path)
        if include_depth:
            temporary_depth = temporary / "episode_depth.mp4"
            lerobot_helpers._write_video_from_arrays(  # noqa: SLF001
                temporary_depth,
                lerobot_helpers._iter_encoded_depth_images(  # noqa: SLF001
                    depth_paths,
                    max_depth_mm=depth_max_mm,
                ),
                fps=FPS,
                codec=video_codec,
            )
            shutil.move(str(temporary_depth), depth_video_path)

    state_array = np.asarray(states, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    episode_stats = {
        "episode_index": source.episode_index,
        "stats": {
            "states": asdict(lerobot_helpers.compute_dataset_stats(state_array.tolist())),
            "action": asdict(lerobot_helpers.compute_dataset_stats(action_array.tolist())),
        },
    }
    chunk = source.episode_index // chunks_size
    video_paths = {
        "observation.images.primary": (f"videos/chunk-{chunk:03d}/primary/episode_{source.episode_index:06d}.mp4")
    }
    if include_depth:
        video_paths["observation.images.d455_depth"] = (
            f"videos/chunk-{chunk:03d}/observation.images.d455_depth/episode_{source.episode_index:06d}.mp4"
        )
    episode_meta = {
        "episode_index": source.episode_index,
        "tasks": [source.task_index],
        "task_index": source.task_index,
        "length": source.rows,
        "dataset_from_index": source.global_start_index,
        "dataset_to_index": source.global_start_index + source.rows - 1,
        "episode_chunk": chunk,
        "source_task_dir": source.source_task_dir,
        "source_episode": source.source_episode,
        "source_frames": source.source_frames,
        "source_fps": source.source_fps,
        "target_fps": FPS,
        "video_paths": video_paths,
        "root_alignment": {
            "target_xy": [0.0, 0.0],
            "target_yaw": 0.0,
            "source_initial_xyz": prepared.source_initial_root_xyz.astype(float).tolist(),
            "source_initial_yaw": float(prepared.source_initial_yaw),
            "absolute_root_z_preserved": True,
        },
        "repairs": {
            "camera_stale_previous_frame": 0,
            "left_hand_target_imputed": int(prepared.left_target_imputed.sum()),
            "right_hand_target_imputed": int(prepared.right_target_imputed.sum()),
            "left_hand_actual_imputed": int(prepared.left_actual_imputed.sum()),
            "right_hand_actual_imputed": int(prepared.right_actual_imputed.sum()),
        },
        "source_quality": {
            "camera_stale": int(prepared.camera_source_stale[:-1].sum()),
            "camera_reused": int(prepared.camera_source_reused[:-1].sum()),
            "left_hand_target_invalid": int(prepared.left_target_source_invalid[1:].sum()),
            "right_hand_target_invalid": int(prepared.right_target_source_invalid[1:].sum()),
        },
    }
    return {
        "episode_meta": episode_meta,
        "episode_stats": episode_stats,
        "summary": {
            "episode_index": source.episode_index,
            "source": f"{source.source_task_dir}/{source.source_episode}",
            "rows": source.rows,
            "repairs": episode_meta["repairs"],
            "source_quality": episode_meta["source_quality"],
        },
    }


def _convert_one(job: dict[str, Any]) -> dict[str, Any]:
    source = SourceEpisode(**job["source"])
    output_root = Path(job["output_root"])
    source_data_stat = (Path(source.path) / "data.json").stat()
    started = time.time()
    try:
        result = _convert_one_impl(job)
        output_fingerprints = _output_fingerprints(
            output_root,
            source.episode_index,
            int(job["chunks_size"]),
            include_depth=bool(job["include_depth"]),
        )
        record = {
            "status": "completed",
            "version": CODEBASE_VERSION,
            "episode_index": source.episode_index,
            "source_path": source.path,
            "source_task_dir": source.source_task_dir,
            "source_episode": source.source_episode,
            "task_index": source.task_index,
            "task": source.task,
            "include_depth": bool(job["include_depth"]),
            "chunks_size": int(job["chunks_size"]),
            "video_codec": str(job["video_codec"]),
            "depth_max_mm": float(job["depth_max_mm"]),
            "source_data_json": {
                "size": source_data_stat.st_size,
                "mtime_ns": source_data_stat.st_mtime_ns,
            },
            "outputs": output_fingerprints,
            "length": source.rows,
            "elapsed_s": time.time() - started,
            **result,
        }
        _write_json(_record_path(output_root, source.episode_index), record)
        return record
    except Exception as exc:  # worker failures must not cancel independent episodes
        return {
            "status": "failed",
            "version": CODEBASE_VERSION,
            "episode_index": source.episode_index,
            "source_path": source.path,
            "source_task_dir": source.source_task_dir,
            "source_episode": source.source_episode,
            "task_index": source.task_index,
            "task": source.task,
            "include_depth": bool(job["include_depth"]),
            "chunks_size": int(job["chunks_size"]),
            "video_codec": str(job["video_codec"]),
            "depth_max_mm": float(job["depth_max_mm"]),
            "source_data_json": {
                "size": source_data_stat.st_size,
                "mtime_ns": source_data_stat.st_mtime_ns,
            },
            "elapsed_s": time.time() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _normalize_task_text(value: str) -> str:
    text = " ".join(str(value).replace("_", " ").split()).strip()
    if not text:
        raise ValueError("task text cannot be empty")
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _task_dirs(source_root: Path) -> list[Path]:
    if any(source_root.glob("episode_*")):
        return [source_root]
    directories = sorted(path for path in source_root.iterdir() if path.is_dir() and any(path.glob("episode_*")))
    if not directories:
        raise ValueError(f"no episode_* directories found under {source_root}")
    return directories


def _episode_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("episode_"))
    except ValueError as exc:
        raise ValueError(f"invalid episode directory name: {path}") from exc


def discover_source(
    source_root: Path,
    *,
    task_override: str | None,
    allow_placeholder_task: bool,
    limit_episodes: int | None,
) -> tuple[list[SourceEpisode], list[dict[str, Any]]]:
    task_directories = _task_dirs(source_root)
    if task_override is not None and len(task_directories) != 1:
        raise ValueError("--task requires a source root containing exactly one task directory")

    task_text_by_dir: dict[Path, str] = {}
    source_fps_by_dir: dict[Path, float] = {}
    for task_dir in task_directories:
        metadata_path = task_dir / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        source_fps = float(metadata.get("capture_fps", FPS))
        if abs(source_fps - FPS) > 1.0e-6:
            raise ValueError(f"{task_dir}: capture_fps={source_fps:g}, expected {FPS}")
        candidate = (
            task_override
            if task_override is not None
            else str(metadata.get("task_description") or metadata.get("task_name") or task_dir.name)
        )
        task_text = _normalize_task_text(candidate)
        if task_text.rstrip(".!?").lower() in PLACEHOLDER_TASK_NAMES and not allow_placeholder_task:
            raise ValueError(
                f"{task_dir}: placeholder task {task_text!r}; pass --task with the real "
                "natural-language instruction"
            )
        task_text_by_dir[task_dir] = task_text
        source_fps_by_dir[task_dir] = source_fps

    unique_tasks = sorted(set(task_text_by_dir.values()))
    task_indices = {task: index for index, task in enumerate(unique_tasks)}
    tasks = [{"task_index": index, "task": task, "description": task} for task, index in task_indices.items()]

    episodes: list[SourceEpisode] = []
    global_index = 0
    for task_dir in task_directories:
        task_text = task_text_by_dir[task_dir]
        for episode_dir in sorted(task_dir.glob("episode_*"), key=_episode_number):
            frame_payload = _read_json(episode_dir / "data.json")
            frames = frame_payload.get("frames") if isinstance(frame_payload, dict) else frame_payload
            if not isinstance(frames, list) or len(frames) < 2:
                raise ValueError(f"{episode_dir}: expected at least two frames")
            rows = len(frames) - 1
            episodes.append(
                SourceEpisode(
                    path=str(episode_dir.resolve()),
                    source_task_dir=task_dir.name,
                    source_episode=episode_dir.name,
                    episode_index=len(episodes),
                    task_index=task_indices[task_text],
                    task=task_text,
                    source_fps=source_fps_by_dir[task_dir],
                    source_frames=len(frames),
                    rows=rows,
                    global_start_index=global_index,
                )
            )
            global_index += rows
            if limit_episodes is not None and len(episodes) >= limit_episodes:
                return episodes, tasks
    return episodes, tasks


def _expected_count(payload: Mapping[str, Any], name: str) -> int | None:
    value = payload.get(f"expected_{name}")
    nested = payload.get("expected")
    if value is None and isinstance(nested, Mapping):
        value = nested.get(name)
    audit = payload.get("audit")
    if value is None and isinstance(audit, Mapping):
        value = audit.get(f"expected_{name}")
    if value is None:
        return None
    count = int(value)
    if count < 0:
        raise ValueError(f"expected {name} must be non-negative")
    return count


def _check_expected_counts(
    payload: Mapping[str, Any],
    *,
    episodes: int,
    source_frames: int,
    rows: int,
    context: str,
) -> None:
    actual = {
        "episodes": episodes,
        "source_frames": source_frames,
        "rows": rows,
    }
    for name, count in actual.items():
        expected = _expected_count(payload, name)
        if expected is not None and expected != count:
            raise ValueError(f"{context}: expected {name}={expected}, discovered {count}")


def discover_manifest(
    manifest_path: Path,
    *,
    limit_episodes: int | None,
) -> tuple[list[SourceEpisode], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    payload = _read_json(manifest_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{manifest_path}: manifest must be a JSON object")
    task_payloads = payload.get("tasks")
    if not isinstance(task_payloads, list) or not task_payloads:
        raise ValueError(f"{manifest_path}: tasks must be a non-empty list")

    scan_total = 0
    scan_task_dirs: list[Path] = []
    for task_payload in task_payloads:
        if not isinstance(task_payload, Mapping):
            raise ValueError(f"{manifest_path}: every tasks item must be an object")
        source_value = task_payload.get("source") or task_payload.get("source_dir") or task_payload.get("path")
        if not source_value:
            raise ValueError(f"{manifest_path}: a tasks item is missing source")
        task_dir = Path(str(source_value)).expanduser()
        if not task_dir.is_absolute():
            task_dir = manifest_path.parent / task_dir
        task_dir = task_dir.resolve()
        scan_task_dirs.append(task_dir)
        scan_total += sum(1 for _ in task_dir.glob("episode_*"))

    source_frame_counts: dict[Path, int] = {}
    with tqdm(
        total=scan_total,
        desc="Scan HGPT sources",
        unit="episode",
        dynamic_ncols=True,
    ) as scan_progress:
        for task_dir in scan_task_dirs:
            for episode_dir in sorted(task_dir.glob("episode_*"), key=_episode_number):
                frame_payload = _read_json(episode_dir / "data.json")
                frames = frame_payload.get("frames") if isinstance(frame_payload, dict) else frame_payload
                if not isinstance(frames, list) or len(frames) < 2:
                    raise ValueError(f"{episode_dir}: expected at least two frames")
                source_frame_counts[episode_dir.resolve()] = len(frames)
                scan_progress.update(1)

    episodes: list[SourceEpisode] = []
    tasks: list[dict[str, Any]] = []
    task_scans: list[dict[str, Any]] = []
    global_index = 0
    seen_tasks: set[str] = set()
    seen_sources: set[Path] = set()
    for task_index, task_payload in enumerate(task_payloads):
        if not isinstance(task_payload, Mapping):
            raise ValueError(f"{manifest_path}: tasks[{task_index}] must be an object")
        source_value = task_payload.get("source") or task_payload.get("source_dir") or task_payload.get("path")
        if not source_value:
            raise ValueError(f"{manifest_path}: tasks[{task_index}] is missing source")
        task_dir = Path(str(source_value)).expanduser()
        if not task_dir.is_absolute():
            task_dir = manifest_path.parent / task_dir
        task_dir = task_dir.resolve()
        if not task_dir.is_dir():
            raise FileNotFoundError(task_dir)
        if task_dir in seen_sources:
            raise ValueError(f"{manifest_path}: duplicate source directory {task_dir}")
        seen_sources.add(task_dir)

        task_id = str(task_payload.get("task", "")).strip()
        if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", task_id):
            raise ValueError(f"{manifest_path}: tasks[{task_index}].task must be snake_case")
        if task_id in seen_tasks:
            raise ValueError(f"{manifest_path}: duplicate task {task_id!r}")
        seen_tasks.add(task_id)
        description = _normalize_task_text(str(task_payload.get("description", "")))

        metadata_path = task_dir / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        source_fps = float(metadata.get("capture_fps", FPS))
        if abs(source_fps - FPS) > 1.0e-6:
            raise ValueError(f"{task_dir}: capture_fps={source_fps:g}, expected {FPS}")
        episode_dirs = sorted(task_dir.glob("episode_*"), key=_episode_number)
        if not episode_dirs:
            raise ValueError(f"{task_dir}: no episode_* directories found")

        task_source_frames = 0
        task_rows = 0
        for episode_dir in episode_dirs:
            source_frames = source_frame_counts[episode_dir.resolve()]
            rows = source_frames - 1
            episodes.append(
                SourceEpisode(
                    path=str(episode_dir.resolve()),
                    source_task_dir=task_dir.name,
                    source_episode=episode_dir.name,
                    episode_index=len(episodes),
                    task_index=task_index,
                    task=task_id,
                    source_fps=source_fps,
                    source_frames=source_frames,
                    rows=rows,
                    global_start_index=global_index,
                )
            )
            global_index += rows
            task_source_frames += source_frames
            task_rows += rows

        _check_expected_counts(
            task_payload,
            episodes=len(episode_dirs),
            source_frames=task_source_frames,
            rows=task_rows,
            context=f"{manifest_path}: task {task_id}",
        )
        tasks.append(
            {
                "task_index": task_index,
                "task": task_id,
                "description": description,
            }
        )
        task_scans.append(
            {
                "task_index": task_index,
                "task": task_id,
                "description": description,
                "source": str(task_dir),
                "episodes": len(episode_dirs),
                "source_frames": task_source_frames,
                "rows": task_rows,
            }
        )
    _check_expected_counts(
        payload,
        episodes=len(episodes),
        source_frames=sum(episode.source_frames for episode in episodes),
        rows=sum(episode.rows for episode in episodes),
        context=str(manifest_path),
    )
    source_scan = {
        "version": CODEBASE_VERSION,
        "manifest": str(manifest_path),
        "episodes": len(episodes),
        "source_frames": sum(episode.source_frames for episode in episodes),
        "rows": sum(episode.rows for episode in episodes),
        "tasks": task_scans,
    }
    if limit_episodes is not None:
        episodes = episodes[:limit_episodes]
    return episodes, tasks, source_scan


def _source_scan_payload(
    source_root: Path,
    episodes: Sequence[SourceEpisode],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "version": CODEBASE_VERSION,
        "source_root": str(source_root),
        "episodes": len(episodes),
        "source_frames": sum(episode.source_frames for episode in episodes),
        "rows": sum(episode.rows for episode in episodes),
        "tasks": [dict(task) for task in tasks],
    }


def _validate_overwrite_target(
    output_root: Path,
    episodes: Sequence[SourceEpisode],
) -> None:
    """Reject broad/source paths and unknown non-empty directories before deletion."""

    output_root = output_root.resolve()
    protected = {Path("/"), Path.home().resolve(), REPO_ROOT.resolve()}
    if output_root in protected:
        raise ValueError(f"refusing to overwrite protected directory: {output_root}")
    for episode in episodes:
        source_episode = Path(episode.path).resolve()
        if output_root == source_episode or output_root in source_episode.parents:
            raise ValueError(f"refusing to overwrite source episode or its ancestor: {output_root}")
    if not output_root.exists():
        return
    if not output_root.is_dir():
        raise ValueError(f"overwrite target is not a directory: {output_root}")
    if not any(output_root.iterdir()):
        return
    marker_path = output_root / "meta/source_scan.json"
    if not marker_path.is_file():
        raise ValueError(f"refusing to overwrite non-empty unrecognized directory: {output_root}")
    try:
        marker = _read_json(marker_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid converter marker: {marker_path}") from exc
    if not isinstance(marker, Mapping) or not str(marker.get("version", "")).startswith(CODEBASE_VERSION_PREFIX):
        raise ValueError(f"unrecognized converter marker: {marker_path}")


def _image_shape(first_episode: SourceEpisode) -> list[int]:
    prepared = prepare_episode(Path(first_episode.path))
    path = lerobot_helpers.frame_image_path(
        Path(first_episode.path),
        prepared.frames[0],
        "primary",
    )
    image = iio.imread(path)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB image at {path}, got {image.shape}")
    return [int(value) for value in image.shape]


def _write_metadata(
    output_root: Path,
    *,
    episodes: Sequence[SourceEpisode],
    tasks: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    chunks_size: int,
    include_depth: bool,
    depth_max_mm: float,
    image_shape: Sequence[int],
) -> None:
    total_frames = sum(episode.rows for episode in episodes)
    total_chunks = math.ceil(len(episodes) / chunks_size)
    video_features: dict[str, Any] = {
        "observation.images.primary": {
            "dtype": "video",
            "shape": list(image_shape),
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(FPS),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    }
    if include_depth:
        video_features["observation.images.d455_depth"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(FPS),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": True,
                "has_audio": False,
            },
        }
    scalar_features = {
        "states": {"dtype": "float32", "shape": [RAW_STATE_DIM]},
        "action": {"dtype": "float32", "shape": [RAW_ACTION_DIM]},
        # Names are retained for compatibility with current WB loaders.
        "state_mask_110": {"dtype": "bool", "shape": [RAW_STATE_DIM]},
        "action_mask_136": {"dtype": "bool", "shape": [RAW_ACTION_DIM]},
        "hgpt.reference_qpos": {"dtype": "float32", "shape": [HGPT_QPOS_DIM]},
        "hgpt.root_xyz_delta": {"dtype": "float32", "shape": [3]},
        "quality.camera_imputed": {"dtype": "bool", "shape": [1]},
        "quality.camera_source_stale": {"dtype": "bool", "shape": [1]},
        "quality.camera_source_reused": {"dtype": "bool", "shape": [1]},
        "quality.left_hand_state_imputed": {"dtype": "bool", "shape": [1]},
        "quality.right_hand_state_imputed": {"dtype": "bool", "shape": [1]},
        "quality.left_hand_action_imputed": {"dtype": "bool", "shape": [1]},
        "quality.right_hand_action_imputed": {"dtype": "bool", "shape": [1]},
        "quality.left_hand_action_source_invalid": {
            "dtype": "bool",
            "shape": [1],
        },
        "quality.right_hand_action_source_invalid": {
            "dtype": "bool",
            "shape": [1],
        },
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "unitree_g1",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(video_features),
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": FPS,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/primary/episode_{episode_index:06d}.mp4",
        "features": {**video_features, **scalar_features},
        "state_layout": STATE_LAYOUT,
        "action_layout": ACTION_LAYOUT,
    }
    episode_metadata = [
        dict(result["episode_meta"])
        for result in sorted(results, key=lambda item: int(item["episode_meta"]["episode_index"]))
    ]
    episode_stats = [
        dict(result["episode_stats"])
        for result in sorted(results, key=lambda item: int(item["episode_meta"]["episode_index"]))
    ]
    summaries = [
        dict(result["summary"])
        for result in sorted(results, key=lambda item: int(item["episode_meta"]["episode_index"]))
    ]
    meta = output_root / "meta"
    _write_json(meta / "info.json", info)
    _write_jsonl(meta / "tasks.jsonl", tasks)
    _write_jsonl(meta / "episodes.jsonl", episode_metadata)
    _write_jsonl(meta / "episodes_stats.jsonl", episode_stats)
    _write_jsonl(
        meta / "conversion_records.jsonl",
        sorted(results, key=lambda item: int(item["episode_index"])),
    )
    _write_json(
        meta / "conversion_config.json",
        {
            "version": CODEBASE_VERSION,
            "fps": FPS,
            "source_reference": "human_derived.g1_qpos (same source as motion.npz)",
            "row_alignment": "state/image at t, action at t+1",
            "root_alignment": (
                "HGPT SE(2): first XY/yaw -> 0; rotate the full XY/quaternion trajectory; preserve absolute root Z"
            ),
            "root_xyz_delta_auxiliary": (
                "hgpt.root_xyz_delta stores aligned xyz - aligned xyz[0]; "
                "do not use its zero-height Z as HGPT qpos"
            ),
            "quaternion_convention": "wxyz with per-episode sign continuity",
            "hgpt_joint_order": "grouped MuJoCo/HGPT order in action[7:36]",
            "physical_joint_order": "IsaacLab order, delta from G1 default pose",
            "hand_action_source": "human_derived left/right_wuji_qpos",
            "hand_state_source": "hand_feedback left/right_wuji_qpos_actual",
            "hand_invalid_policy": (
                "actual state invalid -> fail; finite human target is preserved and "
                "source valid=false is exported as a quality flag"
            ),
            "camera_stale_policy": (
                "preserve each source row's own RGB/depth paths; stale/reused are "
                "exported as quality flags; no camera media is imputed"
            ),
            "gravity_policy": (
                "states[0:3] is recomputed from obs.base_quat and must match "
                "obs.base_gravity within max absolute error 1e-5"
            ),
            "include_depth": include_depth,
            "depth_max_mm": depth_max_mm if include_depth else None,
            "state_layout": STATE_LAYOUT,
            "action_layout": ACTION_LAYOUT,
            "training_contracts": {
                "hgpt_reference_76d": HGPT_TRAINING_CONTRACT,
                "physical_wb_72d": PHYSICAL_TRAINING_CONTRACT,
            },
            "summaries": summaries,
        },
    )


def _write_progress(
    output_root: Path,
    *,
    episodes: Sequence[SourceEpisode],
    chunks_size: int,
    include_depth: bool,
    video_codec: str,
    depth_max_mm: float,
    failures: Sequence[Mapping[str, Any]],
    started: float,
    status: str = "running",
) -> dict[str, Any]:
    records = [
        record
        for episode in episodes
        if (
            record := _completed_record(
                output_root,
                episode,
                chunks_size=chunks_size,
                include_depth=include_depth,
                video_codec=video_codec,
                depth_max_mm=depth_max_mm,
                verify_fingerprints=False,
            )
        )
        is not None
    ]
    payload = {
        "status": status,
        "version": CODEBASE_VERSION,
        "total": len(episodes),
        "completed": len(records),
        "pending": len(episodes) - len(records),
        "failed": len(failures),
        "failures": [dict(failure) for failure in failures[-20:]],
        "completed_rows": sum(int(record.get("length", 0)) for record in records),
        "total_rows": sum(episode.rows for episode in episodes),
        "started_at_unix": started,
        "updated_at_unix": time.time(),
    }
    _write_json(output_root / "meta/progress/status.json", payload)
    return payload


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    if args.manifest is not None:
        episodes, tasks, source_scan = discover_manifest(
            args.manifest,
            limit_episodes=args.limit_episodes,
        )
        source_label = str(args.manifest.expanduser().resolve())
    else:
        source_root = args.source_root.expanduser().resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        episodes, tasks = discover_source(
            source_root,
            task_override=args.task,
            allow_placeholder_task=args.allow_placeholder_task,
            limit_episodes=args.limit_episodes,
        )
        source_scan = _source_scan_payload(source_root, episodes, tasks)
        source_label = str(source_root)
    if not episodes:
        raise ValueError(f"no usable episodes under {source_label}")

    print("HGPT -> WB-compatible LeRobot v3")
    print(f"  source:       {source_label}")
    print(f"  output:       {output_root}")
    print(f"  episodes:     {len(episodes)}")
    print(f"  rows:         {sum(episode.rows for episode in episodes)}")
    print(f"  tasks:        {[task['task'] for task in tasks]}")
    print(f"  fps:          {FPS}")
    print(f"  raw state:    {RAW_STATE_DIM}")
    print(f"  raw action:   {RAW_ACTION_DIM}")
    print(f"  HGPT target:  {HGPT_SEMANTIC_DIM}D (qpos36 + hands40)")
    print(f"  depth video:  {args.include_depth}")

    if args.dry_run:
        quality_totals = {
            "camera": 0,
            "camera_source_stale": 0,
            "camera_source_reused": 0,
            "left_target": 0,
            "right_target": 0,
            "left_target_source_invalid": 0,
            "right_target_source_invalid": 0,
            "left_actual": 0,
            "right_actual": 0,
        }
        with tqdm(
            total=len(episodes),
            desc="Validate HGPT",
            unit="episode",
            dynamic_ncols=True,
        ) as progress:
            validated_rows = 0
            for episode in episodes:
                prepared = prepare_episode(Path(episode.path))
                _validate_source_media(
                    Path(episode.path),
                    prepared,
                    include_depth=args.include_depth,
                )
                for index in range(episode.rows):
                    build_state(prepared, index)
                    build_action(prepared, index + 1)
                quality_totals["camera"] += int(prepared.camera_imputed.sum())
                quality_totals["camera_source_stale"] += int(prepared.camera_source_stale.sum())
                quality_totals["camera_source_reused"] += int(prepared.camera_source_reused.sum())
                quality_totals["left_target"] += int(prepared.left_target_imputed.sum())
                quality_totals["right_target"] += int(prepared.right_target_imputed.sum())
                quality_totals["left_target_source_invalid"] += int(prepared.left_target_source_invalid.sum())
                quality_totals["right_target_source_invalid"] += int(prepared.right_target_source_invalid.sum())
                quality_totals["left_actual"] += int(prepared.left_actual_imputed.sum())
                quality_totals["right_actual"] += int(prepared.right_actual_imputed.sum())
                validated_rows += episode.rows
                progress.update(1)
                progress.set_postfix(rows=validated_rows, failures=0)
        print(f"[OK] dry-run quality totals: {quality_totals}")
        print("[OK] dry run complete; no output files were written")
        return 0

    if args.overwrite:
        _validate_overwrite_target(output_root, episodes)
    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        elif not args.resume:
            raise FileExistsError(
                f"output root exists: {output_root}; pass --resume to continue or --overwrite to replace it"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "meta/source_scan.json", source_scan)

    image_shape = _image_shape(episodes[0])
    completed_records: list[dict[str, Any]] = []
    with tqdm(
        episodes,
        desc="Verify resume",
        unit="episode",
        dynamic_ncols=True,
    ) as progress:
        for episode in progress:
            record = _completed_record(
                output_root,
                episode,
                chunks_size=args.chunks_size,
                include_depth=args.include_depth,
                video_codec=args.video_codec,
                depth_max_mm=args.depth_max_mm,
                verify_fingerprints=True,
            )
            if record is not None:
                completed_records.append(record)
            progress.set_postfix(reused=len(completed_records))
    completed_indices = {int(record["episode_index"]) for record in completed_records}
    pending = [episode for episode in episodes if episode.episode_index not in completed_indices]
    jobs = [
        {
            "source": asdict(episode),
            "output_root": str(output_root),
            "chunks_size": args.chunks_size,
            "include_depth": args.include_depth,
            "video_codec": args.video_codec,
            "depth_max_mm": args.depth_max_mm,
        }
        for episode in pending
    ]
    failures: list[dict[str, Any]] = []
    started = time.time()
    initial_rows = sum(int(record["length"]) for record in completed_records)
    _write_progress(
        output_root,
        episodes=episodes,
        chunks_size=args.chunks_size,
        include_depth=args.include_depth,
        video_codec=args.video_codec,
        depth_max_mm=args.depth_max_mm,
        failures=failures,
        started=started,
    )
    with tqdm(
        total=len(episodes),
        initial=len(completed_records),
        desc="Convert HGPT",
        unit="episode",
        dynamic_ncols=True,
    ) as progress:
        completed_rows = initial_rows
        progress.set_postfix(rows=completed_rows, failures=0)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_episode = {
                executor.submit(_convert_one, job): episode for job, episode in zip(jobs, pending, strict=True)
            }
            for future in concurrent.futures.as_completed(future_to_episode):
                episode = future_to_episode[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "episode_index": episode.episode_index,
                        "source_task_dir": episode.source_task_dir,
                        "source_episode": episode.source_episode,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                if result["status"] == "failed":
                    failures.append(result)
                    print(
                        f"[ERROR] {episode.source_task_dir}/{episode.source_episode}: {result['error']}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    completed_rows += episode.rows
                progress.update(1)
                progress.set_postfix(
                    rows=completed_rows,
                    failures=len(failures),
                )
                _write_progress(
                    output_root,
                    episodes=episodes,
                    chunks_size=args.chunks_size,
                    include_depth=args.include_depth,
                    video_codec=args.video_codec,
                    depth_max_mm=args.depth_max_mm,
                    failures=failures,
                    started=started,
                )

    if failures:
        _write_progress(
            output_root,
            episodes=episodes,
            chunks_size=args.chunks_size,
            include_depth=args.include_depth,
            video_codec=args.video_codec,
            depth_max_mm=args.depth_max_mm,
            failures=failures,
            started=started,
            status="failed",
        )
        raise RuntimeError(f"{len(failures)} episode(s) failed; rerun the same command with --resume to retry")

    results: list[dict[str, Any]] = []
    for episode in episodes:
        record = _completed_record(
            output_root,
            episode,
            chunks_size=args.chunks_size,
            include_depth=args.include_depth,
            video_codec=args.video_codec,
            depth_max_mm=args.depth_max_mm,
            verify_fingerprints=False,
        )
        if record is None:
            raise RuntimeError(f"episode_{episode.episode_index:06d} is incomplete")
        results.append(record)

    _write_metadata(
        output_root,
        episodes=episodes,
        tasks=tasks,
        results=results,
        chunks_size=args.chunks_size,
        include_depth=args.include_depth,
        depth_max_mm=args.depth_max_mm,
        image_shape=image_shape,
    )
    _write_progress(
        output_root,
        episodes=episodes,
        chunks_size=args.chunks_size,
        include_depth=args.include_depth,
        video_codec=args.video_codec,
        depth_max_mm=args.depth_max_mm,
        failures=[],
        started=started,
        status="completed",
    )
    print(f"[OK] wrote LeRobot v3 dataset: {output_root}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-root",
        type=Path,
        help="One HGPT task directory, or a parent containing task directories.",
    )
    source_group.add_argument(
        "--manifest",
        type=Path,
        help=(
            "JSON task manifest. Relative source paths are resolved from the "
            "manifest directory and task list order is preserved."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--task",
        help=(
            "Explicit natural-language task. Allowed only when source-root resolves "
            "to one task; required for placeholders such as hgpt_test."
        ),
    )
    parser.add_argument("--allow-placeholder-task", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunks-size", type=int, default=CHUNKS_SIZE)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="Also encode compressed uint16 depth as an 8-bit grayscale RGB video.",
    )
    parser.add_argument("--depth-max-mm", type=float, default=5000.0)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report source quality without creating output files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-episode outputs and retry only incomplete episodes.",
    )
    args = parser.parse_args(argv)
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if args.chunks_size <= 0:
        parser.error("--chunks-size must be positive")
    if args.limit_episodes is not None and args.limit_episodes <= 0:
        parser.error("--limit-episodes must be positive")
    if args.depth_max_mm <= 0:
        parser.error("--depth-max-mm must be positive")
    if args.manifest is not None and args.task is not None:
        parser.error("--task cannot be combined with --manifest; define it in the manifest")
    if args.manifest is not None and args.allow_placeholder_task:
        parser.error("--allow-placeholder-task applies only to --source-root")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
