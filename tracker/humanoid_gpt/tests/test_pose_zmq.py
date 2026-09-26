from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from deploy.pose_zmq import (
    COLLECTOR_CONTROL_DISCARD,
    COLLECTOR_CONTROL_SAVE,
    COLLECTOR_CONTROL_START,
    CollectorControlSnapshot,
    PicoLeftModeControls,
    PicoRightCollectorControls,
    PoseSnapshot,
    decode_topic_message,
)


def _snapshot() -> PoseSnapshot:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[2] = 0.78
    qpos[3] = 1.0
    return PoseSnapshot(
        frame_index=42,
        source_timestamp_ns=123,
        body_source_monotonic_ns=456,
        publish_monotonic_ns=789,
        mode=1,
        reset_generation=2,
        cmd_vel=np.array([0.1, -0.2, 0.3], dtype=np.float32),
        g1_qpos=qpos,
        body_valid=True,
        left_wuji_qpos=np.zeros(20, dtype=np.float32),
        right_wuji_qpos=np.ones(20, dtype=np.float32) * 0.1,
        left_wuji_qpos_valid=True,
    )


def test_pose_round_trip_and_hand_frame_matches() -> None:
    topic, payload = decode_topic_message(_snapshot().pack())
    assert topic == "pose"
    assert int(payload["frame_index"][0]) == 42
    assert int(payload["hand_frame_index"][0]) == 42
    assert payload["g1_qpos"].shape == (36,)
    assert payload["left_wuji_qpos"].shape == (20,)
    assert bool(payload["left_wuji_qpos_valid"][0])
    assert not bool(payload["right_wuji_qpos_valid"][0])


def test_pose_is_decoded_by_original_sonic_collector() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "collector" / "sonic"))
    from core.subscriber_hub import decode_topic_message as sonic_decode

    decoded = sonic_decode(_snapshot().pack(), "pose")
    assert decoded.version == 4
    assert decoded.payload["frame_index"].tolist() == [42]
    assert decoded.payload["hand_frame_index"].tolist() == [42]
    assert decoded.payload["left_wuji_qpos"].dtype == np.float32


def test_collector_control_wire_format_is_sonic_compatible() -> None:
    message = CollectorControlSnapshot(
        sequence=7,
        command=COLLECTOR_CONTROL_START,
        timestamp_monotonic_ns=123456789,
    ).pack()
    topic, payload = decode_topic_message(message)
    assert topic == "collector_control"
    assert payload["sequence"].tolist() == [7]
    assert payload["command"].tolist() == [COLLECTOR_CONTROL_START]
    assert payload["timestamp_monotonic_ns"].tolist() == [123456789]

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "collector" / "sonic"))
    from core.subscriber_hub import decode_topic_message as sonic_decode

    decoded = sonic_decode(message, "collector_control")
    assert decoded.payload["command"].tolist() == [COLLECTOR_CONTROL_START]


def test_pico_right_collector_controls_are_rising_edge_triggered() -> None:
    controls = PicoRightCollectorControls()
    assert (
        controls.update(
            right_axis_click=False, a_pressed=False, b_pressed=False
        ).command
        is None
    )
    assert (
        controls.update(right_axis_click=True, a_pressed=False, b_pressed=False).command
        == COLLECTOR_CONTROL_START
    )
    assert (
        controls.update(right_axis_click=True, a_pressed=False, b_pressed=False).command
        is None
    )
    controls.update(right_axis_click=False, a_pressed=False, b_pressed=False)

    assert (
        controls.update(right_axis_click=False, a_pressed=True, b_pressed=False).command
        == COLLECTOR_CONTROL_SAVE
    )
    assert (
        controls.update(right_axis_click=False, a_pressed=True, b_pressed=False).command
        is None
    )
    controls.update(right_axis_click=False, a_pressed=False, b_pressed=False)

    assert (
        controls.update(right_axis_click=False, a_pressed=False, b_pressed=True).command
        == COLLECTOR_CONTROL_DISCARD
    )


def test_pico_right_collector_controls_reject_a_b_together() -> None:
    controls = PicoRightCollectorControls()
    decision = controls.update(right_axis_click=False, a_pressed=True, b_pressed=True)
    assert decision.command is None
    assert decision.ambiguous


def test_pico_left_stick_toggles_mode_zero_and_one_on_rising_edges() -> None:
    controls = PicoLeftModeControls()

    assert controls.update(left_axis_click=False, current_mode=1) is None
    assert controls.update(left_axis_click=True, current_mode=1) == 0
    assert controls.update(left_axis_click=True, current_mode=0) is None

    assert controls.update(left_axis_click=False, current_mode=0) is None
    assert controls.update(left_axis_click=True, current_mode=0) == 1


def test_pico_left_stick_ignores_offline_modes() -> None:
    controls = PicoLeftModeControls()

    assert controls.update(left_axis_click=True, current_mode=2) is None
    assert controls.update(left_axis_click=False, current_mode=2) is None
    assert controls.update(left_axis_click=True, current_mode=3) is None
