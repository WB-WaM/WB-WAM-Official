"""Shared Unitree G1 joint-order constants for SONIC encoder inputs."""

from __future__ import annotations

import numpy as np

# Absolute motor / MuJoCo order. Shared by offline and live body-only viewers.
MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

MUJOCO_TO_ISAACLAB = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

DEFAULT_ANGLES_MUJOCO = np.asarray(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

DEFAULT_ANGLES_ISAACLAB = DEFAULT_ANGLES_MUJOCO[MUJOCO_TO_ISAACLAB]

# Preserved only to reproduce and diagnose the original WBWAM bridge.
LEGACY_BRIDGE_DEFAULT_ANGLES_ISAACLAB = np.asarray(
    [
        -0.312,
        -0.312,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.669,
        0.669,
        0.0,
        0.0,
        0.2,
        -0.363,
        -0.363,
        0.0,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.2,
        0.0,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

ENCODER_VARIANT_SONIC_CANONICAL = "sonic_canonical"
ENCODER_VARIANT_LEGACY_BRIDGE = "legacy_bridge"
ENCODER_VARIANTS = (
    ENCODER_VARIANT_SONIC_CANONICAL,
    ENCODER_VARIANT_LEGACY_BRIDGE,
)


def normalize_encoder_variant(value: str | None) -> str:
    variant = str(value or ENCODER_VARIANT_SONIC_CANONICAL).strip().lower().replace("-", "_")
    if variant not in ENCODER_VARIANTS:
        raise ValueError(f"unknown encoder variant {value!r}; expected one of {ENCODER_VARIANTS}")
    return variant


def default_angles_for_variant(value: str | None) -> np.ndarray:
    variant = normalize_encoder_variant(value)
    return (
        DEFAULT_ANGLES_ISAACLAB
        if variant == ENCODER_VARIANT_SONIC_CANONICAL
        else LEGACY_BRIDGE_DEFAULT_ANGLES_ISAACLAB
    )
