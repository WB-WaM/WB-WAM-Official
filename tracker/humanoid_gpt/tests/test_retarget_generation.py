from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from deploy.retarget import (
    _RETARGET_SESSIONS,
    MocapType,
    _publish_retarget_output_if_current,
    request_retarget_calibration,
)


def _qpos(joint_value: float) -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[3] = 1.0
    qpos[7:] = joint_value
    return qpos


def _shared_state():
    ctx = mp.get_context("spawn")
    buf = ctx.Array("f", 36, lock=True)
    buf_hand = ctx.Array("f", 4, lock=True)
    ts = ctx.Value("d", 0.0)
    generation = ctx.Value("q", 0)
    return buf, buf_hand, ts, generation


def _read_shared(buf, buf_hand, ts):
    with buf.get_lock():
        qpos = np.frombuffer(buf.get_obj(), dtype=np.float32).copy()
    with buf_hand.get_lock():
        hand = np.frombuffer(buf_hand.get_obj(), dtype=np.float32).copy()
    with ts.get_lock():
        timestamp = float(ts.value)
    return qpos, hand, timestamp


def test_calibration_request_rejects_inflight_old_generation_output() -> None:
    buf, buf_hand, ts, generation = _shared_state()
    initial_qpos = _qpos(-0.25)
    initial_hand = np.full(4, -1.0, dtype=np.float32)
    with buf.get_lock():
        np.frombuffer(buf.get_obj(), dtype=np.float32)[:] = initial_qpos
    with buf_hand.get_lock():
        np.frombuffer(buf_hand.get_obj(), dtype=np.float32)[:] = initial_hand
    with ts.get_lock():
        ts.value = 123.0

    session = {
        "buf": buf,
        "mocap_type": MocapType.PICO,
        "calibration_generation": generation,
        "ts": ts,
    }
    _RETARGET_SESSIONS.append(session)
    try:
        # The worker captured generation 0 before beginning conversion/GMR.
        frame_generation = 0
        assert request_retarget_calibration(buf)
        with generation.get_lock():
            assert generation.value == 1

        published = _publish_retarget_output_if_current(
            calibration_generation=generation,
            frame_generation=frame_generation,
            buf=buf,
            buf_hand=buf_hand,
            ts=ts,
            qpos=_qpos(0.75),
            hand_data=np.ones(4, dtype=np.float32),
            frame_received_at_s=456.0,
        )
        assert not published

        qpos, hand, timestamp = _read_shared(buf, buf_hand, ts)
        np.testing.assert_array_equal(qpos, initial_qpos)
        np.testing.assert_array_equal(hand, initial_hand)
        assert timestamp == 0.0
    finally:
        _RETARGET_SESSIONS.remove(session)


def test_current_generation_can_publish_after_calibration() -> None:
    buf, buf_hand, ts, generation = _shared_state()
    with generation.get_lock():
        generation.value = 2

    expected_qpos = _qpos(0.5)
    expected_hand = np.arange(4, dtype=np.float32)
    assert _publish_retarget_output_if_current(
        calibration_generation=generation,
        frame_generation=2,
        buf=buf,
        buf_hand=buf_hand,
        ts=ts,
        qpos=expected_qpos,
        hand_data=expected_hand,
        frame_received_at_s=789.0,
    )

    qpos, hand, timestamp = _read_shared(buf, buf_hand, ts)
    np.testing.assert_array_equal(qpos, expected_qpos)
    np.testing.assert_array_equal(hand, expected_hand)
    assert timestamp == pytest.approx(789.0)


def test_each_calibration_request_advances_generation() -> None:
    buf, _buf_hand, ts, generation = _shared_state()
    session = {
        "buf": buf,
        "mocap_type": MocapType.PICO,
        "calibration_generation": generation,
        "ts": ts,
    }
    _RETARGET_SESSIONS.append(session)
    try:
        assert request_retarget_calibration(buf)
        assert request_retarget_calibration(buf)
        with generation.get_lock():
            assert generation.value == 2
        with ts.get_lock():
            assert ts.value == 0.0
    finally:
        _RETARGET_SESSIONS.remove(session)


def test_default_publish_timestamp_uses_monotonic_clock(monkeypatch) -> None:
    import deploy.retarget as retarget_module

    buf, buf_hand, ts, generation = _shared_state()
    monkeypatch.setattr(retarget_module.time, "monotonic", lambda: 123.456)

    assert _publish_retarget_output_if_current(
        calibration_generation=generation,
        frame_generation=0,
        buf=buf,
        buf_hand=buf_hand,
        ts=ts,
        qpos=_qpos(0.25),
        hand_data=np.zeros(4, dtype=np.float32),
    )

    _qpos_value, _hand, timestamp = _read_shared(buf, buf_hand, ts)
    assert timestamp == pytest.approx(123.456)


@pytest.mark.parametrize("timestamp", [0.0, -1.0, np.nan, np.inf])
def test_publish_rejects_invalid_monotonic_timestamp_without_mutation(
    timestamp: float,
) -> None:
    buf, buf_hand, ts, generation = _shared_state()

    with pytest.raises(ValueError, match="monotonic timestamp"):
        _publish_retarget_output_if_current(
            calibration_generation=generation,
            frame_generation=0,
            buf=buf,
            buf_hand=buf_hand,
            ts=ts,
            qpos=_qpos(0.25),
            hand_data=np.ones(4, dtype=np.float32),
            frame_received_at_s=timestamp,
        )

    qpos, hand, published_timestamp = _read_shared(buf, buf_hand, ts)
    np.testing.assert_array_equal(qpos, np.zeros(36, dtype=np.float32))
    np.testing.assert_array_equal(hand, np.zeros(4, dtype=np.float32))
    assert published_timestamp == 0.0
