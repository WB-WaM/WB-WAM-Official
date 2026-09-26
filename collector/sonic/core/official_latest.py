from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .schema import HandStatusSample, PoseSample, RobotStateActionSample, to_python


OFFICIAL_SMPL_STALE_AFTER_NS = 100_000_000

_LEFT_WRIST_INDICES = (23, 25, 27)
_RIGHT_WRIST_INDICES = (24, 26, 28)
_MISSING = object()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _payload_section(payload: Any, *section_names: str) -> Mapping[str, Any]:
    direct = _mapping(payload)
    for name in section_names:
        nested = direct.get(name)
        if isinstance(nested, Mapping):
            return nested
    return direct


def _first_frame(value: Any, *, frame_ndim: int) -> Any:
    if value is None:
        return _MISSING
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return _MISSING
    if array.size == 0:
        return _MISSING
    if array.ndim > frame_ndim:
        array = array[0]
    return to_python(array)


def _first_scalar(value: Any, default: int = -1) -> Any:
    if value is None:
        return default
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return default
    if array.size == 0:
        return default
    return to_python(array.reshape(-1)[0])


def _zero_like(value: Any, default_shape: tuple[int, ...]) -> Any:
    if value is _MISSING:
        return np.zeros(default_shape, dtype=np.float32).tolist()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        array = np.zeros(default_shape, dtype=np.float32)
    return np.zeros(array.shape, dtype=np.float32).tolist()


def _identity_quat_like(value: Any) -> Any:
    if value is _MISSING:
        return [1.0, 0.0, 0.0, 0.0]
    try:
        source = np.asarray(value)
    except (TypeError, ValueError):
        source = np.zeros(4, dtype=np.float32)
    if source.size == 0 or source.shape[-1:] != (4,):
        source = np.zeros(4, dtype=np.float32)
    identity = np.zeros(source.shape, dtype=np.float32)
    identity.reshape(-1, 4)[:, 0] = 1.0
    return identity.tolist()


def _wrist_values(joint_pos: Any, indices: tuple[int, int, int]) -> list[Any]:
    if joint_pos is _MISSING:
        return [0.0, 0.0, 0.0]
    try:
        flat = np.asarray(joint_pos).reshape(-1)
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0]
    if flat.size <= max(indices):
        return [0.0, 0.0, 0.0]
    return [to_python(flat[index]) for index in indices]


def _pose_fields(
    pose_sample: PoseSample | None,
    *,
    tick_time_ns: int,
    smpl_stale_after_ns: int,
) -> tuple[int, int, int, bool, dict[str, Any], dict[str, Any]]:
    if pose_sample is None:
        human_derived = {
            "smpl_joints": _zero_like(_MISSING, (24, 3)),
            "smpl_pose": _zero_like(_MISSING, (21, 3)),
            "body_quat_w": _identity_quat_like(_MISSING),
            "joint_pos": _zero_like(_MISSING, (29,)),
            "left_wrist_joints": [0.0, 0.0, 0.0],
            "right_wrist_joints": [0.0, 0.0, 0.0],
        }
        return -1, -1, -1, False, {}, human_derived

    pose_timestamp_ns = int(pose_sample.timestamp_ns)
    smpl_age_ns = int(tick_time_ns) - pose_timestamp_ns
    pose_payload = _payload_section(pose_sample.payload, "pose")
    smpl_joints = _first_frame(pose_payload.get("smpl_joints"), frame_ndim=2)
    smpl_pose = _first_frame(pose_payload.get("smpl_pose"), frame_ndim=2)
    body_quat_w = _first_frame(pose_payload.get("body_quat_w"), frame_ndim=1)
    joint_pos = _first_frame(pose_payload.get("joint_pos"), frame_ndim=1)

    frame_index = _first_scalar(
        pose_payload.get("frame_index"),
        default=int(pose_sample.frame_index),
    )
    try:
        pose_frame_index = int(frame_index)
    except (TypeError, ValueError):
        pose_frame_index = int(pose_sample.frame_index)

    required_smpl_present = all(
        value is not _MISSING for value in (smpl_joints, smpl_pose, body_quat_w)
    )
    smpl_valid = (
        required_smpl_present
        and 0 <= smpl_age_ns <= int(smpl_stale_after_ns)
    )

    human_raw = to_python(dict(_mapping(pose_sample.human_raw)))
    human_derived = to_python(dict(_mapping(pose_sample.human_derived)))
    if smpl_valid:
        current_joints = smpl_joints
        current_pose = smpl_pose
        current_body_quat = body_quat_w
        current_joint_pos = (
            joint_pos if joint_pos is not _MISSING else _zero_like(_MISSING, (29,))
        )
        left_wrist = _wrist_values(joint_pos, _LEFT_WRIST_INDICES)
        right_wrist = _wrist_values(joint_pos, _RIGHT_WRIST_INDICES)
    else:
        current_joints = _zero_like(smpl_joints, (24, 3))
        current_pose = _zero_like(smpl_pose, (21, 3))
        current_body_quat = _identity_quat_like(body_quat_w)
        current_joint_pos = _zero_like(joint_pos, (29,))
        left_wrist = [0.0, 0.0, 0.0]
        right_wrist = [0.0, 0.0, 0.0]

    human_derived.update(
        {
            "smpl_joints": to_python(current_joints),
            "smpl_pose": to_python(current_pose),
            "body_quat_w": to_python(current_body_quat),
            "joint_pos": to_python(current_joint_pos),
            "left_wrist_joints": left_wrist,
            "right_wrist_joints": right_wrist,
        }
    )
    return (
        pose_frame_index,
        pose_timestamp_ns,
        smpl_age_ns,
        bool(smpl_valid),
        human_raw,
        human_derived,
    )


def _state_fields(
    state_action: RobotStateActionSample | None,
) -> tuple[int, int, dict[str, Any], dict[str, Any]]:
    if state_action is None:
        return -1, -1, {}, {}

    payload = _payload_section(
        state_action.payload,
        "state_action",
        "robot_state_action",
    )
    obs = to_python(dict(_mapping(state_action.obs)))
    action = to_python(dict(_mapping(state_action.action)))

    for key in (
        "base_quat",
        "base_gravity",
        "base_ang_vel",
        "base_accel",
        "body_q",
        "body_dq",
        "left_hand_q",
        "right_hand_q",
        "token_state",
    ):
        if key in payload:
            obs[key] = to_python(payload[key])

    if "token_state" not in payload and "token_state" not in obs:
        for alias in ("motion_token", "token", "tokens"):
            if alias in payload:
                obs["token_state"] = to_python(payload[alias])
                break

    if "encoder_mode" in payload:
        obs["encoder_mode"] = _first_scalar(payload["encoder_mode"])

    for key in ("body_action", "left_hand_action", "right_hand_action"):
        if key in payload:
            action[key] = to_python(payload[key])
    if "token_state" in obs and "motion_token" not in action:
        action["motion_token"] = to_python(obs["token_state"])

    state_index = _first_scalar(payload.get("state_index"), default=-1)
    try:
        state_index = int(state_index)
    except (TypeError, ValueError):
        state_index = -1
    return state_index, int(state_action.timestamp_ns), obs, action


def _hand_feedback(hand_status: HandStatusSample | None) -> dict[str, Any]:
    if hand_status is None:
        return {}
    feedback = _mapping(hand_status.feedback)
    if not feedback:
        payload = _mapping(hand_status.payload)
        feedback = _mapping(payload.get("feedback")) or payload
    return to_python(dict(feedback))


def build_official_latest_snapshot(
    *,
    tick_time_ns: int,
    pose_sample: PoseSample | None,
    state_action: RobotStateActionSample | None,
    hand_status: HandStatusSample | None = None,
    smpl_stale_after_ns: int = OFFICIAL_SMPL_STALE_AFTER_NS,
) -> dict[str, Any]:
    """Build one official-style 50 Hz latest-sample snapshot."""

    tick_time_ns = int(tick_time_ns)
    (
        pose_frame_index,
        pose_timestamp_ns,
        smpl_age_ns,
        smpl_valid,
        human_raw,
        human_derived,
    ) = _pose_fields(
        pose_sample,
        tick_time_ns=tick_time_ns,
        smpl_stale_after_ns=int(smpl_stale_after_ns),
    )
    state_index, robot_timestamp_ns, obs, action = _state_fields(state_action)

    return {
        "tick_time_ns": tick_time_ns,
        "state_index": state_index,
        "pose_frame_index": pose_frame_index,
        "pose_timestamp_ns": pose_timestamp_ns,
        "robot_timestamp_ns": robot_timestamp_ns,
        "smpl_age_ns": smpl_age_ns,
        "smpl_valid": smpl_valid,
        "human_raw": human_raw,
        "human_derived": human_derived,
        "obs": obs,
        "action": action,
        "hand_feedback": _hand_feedback(hand_status),
    }


__all__ = ["OFFICIAL_SMPL_STALE_AFTER_NS", "build_official_latest_snapshot"]
