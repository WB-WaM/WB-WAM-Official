"""HGPT collector samples and causal snapshot construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

POSE_STALE_NS = 100_000_000
STATE_STALE_NS = 100_000_000
HAND_STALE_NS = 100_000_000


def scalar(payload: dict[str, np.ndarray], key: str, default: Any = None) -> Any:
    value = payload.get(key)
    if value is None:
        return default
    array = np.asarray(value).reshape(-1)
    return array[0].item() if array.size else default


def int_scalar(
    payload: dict[str, np.ndarray],
    key: str,
    default: int = -1,
) -> int:
    try:
        return int(scalar(payload, key, default))
    except (TypeError, ValueError, OverflowError):
        return default


def to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_python(item) for key, item in value.items()}
    return value


def base_quat_to_gravity(quat: Any) -> list[float]:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([w, x, y, z])
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid base quaternion")
    w, x, y, z = np.array([w, x, y, z]) / norm
    return [
        float(-2.0 * (x * z - w * y)),
        float(-2.0 * (y * z + w * x)),
        float(-(1.0 - 2.0 * (x * x + y * y))),
    ]


@dataclass(frozen=True, slots=True)
class WireSample:
    topic: str
    receive_monotonic_ns: int
    payload: dict[str, np.ndarray]

    @property
    def frame_index(self) -> int:
        return int_scalar(self.payload, "frame_index")

    @property
    def publish_monotonic_ns(self) -> int:
        fallback = int_scalar(self.payload, "timestamp_monotonic_ns")
        return int_scalar(
            self.payload,
            "publish_monotonic_ns",
            fallback,
        )


@dataclass(frozen=True, slots=True)
class PairedPose:
    frame_index: int
    pose: WireSample
    raw: WireSample


def _age(tick_ns: int, sample_ns: int) -> int:
    return tick_ns - sample_ns if sample_ns >= 0 else -1


def build_snapshot(
    *,
    tick_ns: int,
    pair: PairedPose | None,
    state: WireSample | None,
    hand: WireSample | None,
) -> dict[str, Any]:
    pose_payload = pair.pose.payload if pair is not None else {}
    raw_payload = pair.raw.payload if pair is not None else {}
    state_payload = state.payload if state is not None else {}
    hand_payload = hand.payload if hand is not None else {}

    pose_publish_ns = pair.pose.publish_monotonic_ns if pair is not None else -1
    raw_publish_ns = pair.raw.publish_monotonic_ns if pair is not None else -1
    robot_timestamp_ns = int_scalar(state_payload, "timestamp_monotonic_ns")
    hand_timestamp_ns = int_scalar(
        hand_payload,
        "source_timestamp_monotonic_pc_ns",
        int_scalar(hand_payload, "timestamp_monotonic_ns"),
    )
    pose_age = _age(tick_ns, pose_publish_ns)
    raw_age = _age(tick_ns, raw_publish_ns)
    robot_age = _age(tick_ns, robot_timestamp_ns)
    hand_age = _age(tick_ns, hand_timestamp_ns)
    pose_fresh = 0 <= pose_age <= POSE_STALE_NS
    raw_fresh = 0 <= raw_age <= POSE_STALE_NS
    state_fresh = 0 <= robot_age <= STATE_STALE_NS
    hand_fresh = 0 <= hand_age <= HAND_STALE_NS

    mode = int_scalar(pose_payload, "mode", 0)
    source_ns = int_scalar(pose_payload, "body_source_monotonic_ns")
    source_age = _age(tick_ns, source_ns)
    source_fresh = 0 <= source_age <= POSE_STALE_NS
    body_valid = bool(scalar(pose_payload, "body_valid", False))
    reference_valid = pose_fresh and source_fresh and body_valid

    human_raw = {
        "frame_index": pair.frame_index if pair is not None else -1,
        "source_timestamp_ns": int_scalar(raw_payload, "source_timestamp_ns"),
        "publish_monotonic_ns": raw_publish_ns,
        "body_poses": to_python(raw_payload.get("body_poses", np.zeros((24, 7)))),
        "body_valid": bool(scalar(raw_payload, "body_valid", False)) and raw_fresh,
        "left_hand_keypoints": to_python(raw_payload.get("left_hand_keypoints", np.zeros((21, 3)))),
        "right_hand_keypoints": to_python(raw_payload.get("right_hand_keypoints", np.zeros((21, 3)))),
        "left_hand_valid": bool(scalar(raw_payload, "left_hand_valid", False)) and raw_fresh,
        "right_hand_valid": bool(scalar(raw_payload, "right_hand_valid", False)) and raw_fresh,
        "hand_source": int_scalar(raw_payload, "hand_source", 0),
    }
    human_derived = {
        "frame_index": pair.frame_index if pair is not None else -1,
        "source_timestamp_ns": int_scalar(pose_payload, "source_timestamp_ns"),
        "body_source_monotonic_ns": source_ns,
        "publish_monotonic_ns": pose_publish_ns,
        "g1_qpos": to_python(pose_payload.get("g1_qpos", np.zeros(36))),
        "body_valid": reference_valid,
        "left_wuji_qpos": to_python(pose_payload.get("left_wuji_qpos", np.zeros(20))),
        "right_wuji_qpos": to_python(pose_payload.get("right_wuji_qpos", np.zeros(20))),
        "left_wuji_qpos_valid": bool(scalar(pose_payload, "left_wuji_qpos_valid", False)) and pose_fresh,
        "right_wuji_qpos_valid": bool(scalar(pose_payload, "right_wuji_qpos_valid", False)) and pose_fresh,
        "mode": mode,
        "cmd_vel": to_python(pose_payload.get("cmd_vel", np.zeros(3))),
        "sequence_done": bool(scalar(pose_payload, "sequence_done", False)),
    }

    obs: dict[str, Any] = {}
    policy: dict[str, Any] = {}
    action: dict[str, Any] = {}
    if state_fresh:
        for key in (
            "base_quat",
            "base_ang_vel",
            "base_accel",
            "body_q",
            "body_dq",
            "body_tau",
        ):
            if key in state_payload:
                obs[key] = to_python(state_payload[key])
        if "base_quat" in obs:
            try:
                obs["base_gravity"] = base_quat_to_gravity(obs["base_quat"])
            except (TypeError, ValueError):
                pass
        policy = {
            "kind": int_scalar(state_payload, "policy_kind", 0),
            "obs": to_python(state_payload.get("policy_obs", np.empty(0))),
            "action": to_python(state_payload.get("policy_action", np.empty(0))),
            "reference_qpos": to_python(state_payload.get("reference_qpos", np.zeros(36))),
            "reference_valid": bool(scalar(state_payload, "reference_valid", False)),
        }
        action = {
            "body_action": to_python(state_payload.get("body_action", np.zeros(29))),
            "kp": to_python(state_payload.get("kp", np.zeros(29))),
            "kd": to_python(state_payload.get("kd", np.zeros(29))),
            "lowcmd_published": bool(scalar(state_payload, "lowcmd_published", False)),
        }

    return {
        "state_index": int_scalar(state_payload, "state_index"),
        "pose_frame_index": pair.frame_index if pair is not None else -1,
        "pose_publish_monotonic_ns": pose_publish_ns,
        "raw_publish_monotonic_ns": raw_publish_ns,
        "robot_timestamp_monotonic_ns": robot_timestamp_ns,
        "robot_publish_monotonic_ns": int_scalar(
            state_payload,
            "publish_monotonic_ns",
        ),
        "hand_timestamp_monotonic_ns": hand_timestamp_ns,
        "mode": mode,
        "pose_age_ns": pose_age,
        "raw_age_ns": raw_age,
        "body_source_age_ns": source_age,
        "robot_age_ns": robot_age,
        "hand_age_ns": hand_age,
        "pose_stale": not pose_fresh,
        "raw_stale": not raw_fresh,
        "robot_stale": not state_fresh,
        "hand_stale": not hand_fresh,
        "human_raw": human_raw,
        "human_derived": human_derived,
        "obs": obs,
        "policy": policy,
        "action": action,
        "hand_feedback": to_python(hand_payload) if hand_fresh else {},
    }


def snapshot_ready(snapshot: dict[str, Any]) -> bool:
    if snapshot["pose_stale"] or snapshot["robot_stale"]:
        return False
    return snapshot["mode"] != 1 or bool(snapshot["human_derived"]["body_valid"])
