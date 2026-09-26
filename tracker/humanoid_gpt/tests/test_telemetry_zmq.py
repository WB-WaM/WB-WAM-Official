from __future__ import annotations

import numpy as np

from deploy.pose_zmq import TeleopRawSnapshot, decode_topic_message
from deploy.telemetry_zmq import (
    POLICY_TRACK,
    RobotStateActionSnapshot,
)


def test_teleop_raw_roundtrip():
    raw = TeleopRawSnapshot(
        frame_index=7,
        source_timestamp_ns=11,
        publish_monotonic_ns=13,
        body_poses=np.zeros((24, 7), dtype=np.float32),
        body_valid=True,
        left_hand_keypoints=np.zeros((21, 3), dtype=np.float32),
        right_hand_keypoints=np.zeros((21, 3), dtype=np.float32),
        left_hand_valid=True,
        right_hand_valid=False,
        hand_source=1,
    )
    topic, payload = decode_topic_message(raw.pack())
    assert topic == "teleop_raw"
    assert payload["frame_index"].item() == 7
    assert payload["body_poses"].shape == (24, 7)
    assert payload["left_hand_keypoints"].shape == (21, 3)


def test_zmq_pose_source_preserves_offline_mode():
    from deploy.pose_zmq import ZmqPoseSource

    source = object.__new__(ZmqPoseSource)
    source.refresh = lambda: {
        "mode": np.array([3], dtype=np.int32),
        "cmd_vel": np.zeros(3, dtype=np.float32),
        "kill": np.array([False]),
    }
    assert source.command().mode == 3


def test_robot_state_action_tracking_shapes():
    reference = np.zeros(36, dtype=np.float32)
    reference[3] = 1.0
    snapshot = RobotStateActionSnapshot(
        state_index=4,
        sensor_monotonic_ns=100,
        mode=1,
        cmd_vel=np.zeros(3),
        base_quat=np.array([1, 0, 0, 0]),
        base_ang_vel=np.zeros(3),
        base_accel=np.array([0, 0, 9.81]),
        body_q=np.zeros(29),
        body_dq=np.zeros(29),
        body_tau=np.zeros(29),
        reference_qpos=reference,
        reference_valid=True,
        policy_kind=POLICY_TRACK,
        policy_obs=np.zeros(136),
        policy_action=np.zeros(29),
        body_action=np.zeros(29),
        kp=np.ones(29),
        kd=np.ones(29),
        lowcmd_published=True,
    )
    topic, payload = decode_topic_message(snapshot.pack())
    assert topic == "robot_state_action"
    assert payload["policy_obs"].shape == (136,)
    assert payload["policy_action"].shape == (29,)
    assert payload["body_action"].shape == (29,)
