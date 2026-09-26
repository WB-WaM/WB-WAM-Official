from __future__ import annotations

import math

import numpy as np
import pytest

from deploy.pico import (
    PICO_BODY_JOINT_NAMES,
    UNITY_TO_GMR_QUAT_WXYZ,
    PicoBodyFrame,
    PicoFrameCalibrator,
    PicoXrtSource,
    pico_frame_to_gmr_frame,
)


def _body_poses() -> np.ndarray:
    poses = np.zeros((24, 7), dtype=np.float32)
    poses[:, 1] = np.linspace(0.0, 1.7, 24)
    poses[:, 6] = 1.0
    return poses


def test_official_joint_order() -> None:
    assert len(PICO_BODY_JOINT_NAMES) == 24
    assert PICO_BODY_JOINT_NAMES[:4] == (
        "Pelvis",
        "Left_Hip",
        "Right_Hip",
        "Spine1",
    )
    assert PICO_BODY_JOINT_NAMES[20:] == (
        "Left_Wrist",
        "Right_Wrist",
        "Left_Hand",
        "Right_Hand",
    )


def test_frame_validates_and_defensively_normalizes() -> None:
    poses = _body_poses()
    poses[:, 3:7] *= 2.0
    frame = PicoBodyFrame(1, poses)

    assert frame.body_poses.shape == (24, 7)
    assert frame.body_poses.dtype == np.float32
    assert not frame.body_poses.flags.writeable
    np.testing.assert_allclose(
        np.linalg.norm(frame.body_poses[:, 3:7], axis=1),
        1.0,
    )
    poses[0, 0] = 99.0
    assert frame.body_poses[0, 0] != 99.0


@pytest.mark.parametrize("timestamp", [0, -1, True, 1.5])
def test_frame_rejects_invalid_timestamp(timestamp) -> None:
    with pytest.raises(ValueError, match="timestamp_ns"):
        PicoBodyFrame(timestamp, _body_poses())


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda poses: poses[:23], "shape"),
        (lambda poses: _replace(poses, (0, 0), np.nan), "NaN or Inf"),
        (lambda poses: _replace(poses, (0, 3, 7), 0.0), "zero quaternion"),
        (lambda poses: _replace(poses, (slice(None), slice(0, 3)), 0.0), "scale"),
    ],
)
def test_frame_rejects_bad_data(mutator, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PicoBodyFrame(1, mutator(_body_poses()))


def _replace(array: np.ndarray, index, value) -> np.ndarray:
    result = array.copy()
    if len(index) == 3:
        row, start, stop = index
        result[row, start:stop] = value
    else:
        result[index] = value
    return result


def test_unity_position_and_identity_quaternion_conversion() -> None:
    poses = _body_poses()
    poses[0, :3] = [1.0, 2.0, 3.0]
    frame = PicoBodyFrame(1, poses)
    converted = pico_frame_to_gmr_frame(frame)

    np.testing.assert_allclose(converted["Pelvis"][0], [1.0, -3.0, 2.0])
    np.testing.assert_allclose(
        converted["Pelvis"][1],
        UNITY_TO_GMR_QUAT_WXYZ,
        atol=1e-6,
    )


def test_quaternion_basis_is_premultiplied() -> None:
    poses = _body_poses()
    half = math.sqrt(0.5)
    poses[0, 3:7] = [0.0, half, 0.0, half]
    converted = pico_frame_to_gmr_frame(PicoBodyFrame(1, poses))

    np.testing.assert_allclose(
        converted["Pelvis"][1],
        [0.5, 0.5, 0.5, 0.5],
        atol=1e-6,
    )
    assert np.linalg.norm(converted["Pelvis"][1]) == pytest.approx(1.0)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        dtype=np.float32,
    )


def test_xy_yaw_calibration_and_reset() -> None:
    calibrator = PicoFrameCalibrator()
    frame = {
        "Pelvis": (
            np.array([2.0, 3.0, 1.1], dtype=np.float32),
            _yaw_quat(math.pi / 2),
        ),
        "Left_Hand": (
            np.array([3.0, 3.0, 1.4], dtype=np.float32),
            _yaw_quat(math.pi / 2),
        ),
        "Left_Foot": (
            np.array([2.0, 3.1, 0.1], dtype=np.float32),
            _yaw_quat(math.pi / 2),
        ),
        "Right_Foot": (
            np.array([2.0, 2.9, 0.0], dtype=np.float32),
            _yaw_quat(math.pi / 2),
        ),
    }
    converted = calibrator.apply(frame)

    np.testing.assert_allclose(converted["Pelvis"][0], [0.0, 0.0, 1.1])
    np.testing.assert_allclose(converted["Left_Hand"][0], [0.0, -1.0, 1.4], atol=1e-6)
    np.testing.assert_allclose(
        converted["Pelvis"][1], [1.0, 0.0, 0.0, 0.0], atol=1e-6
    )

    calibrator.reset()
    moved = {
        name: (position + np.array([5.0, -2.0, 0.0]), quaternion)
        for name, (position, quaternion) in frame.items()
    }
    converted = calibrator.apply(moved)
    np.testing.assert_allclose(converted["Pelvis"][0], [0.0, 0.0, 1.1])


def test_calibration_grounds_feet_and_preserves_vertical_motion() -> None:
    calibrator = PicoFrameCalibrator()
    frame = {
        "Pelvis": (np.array([0.0, 0.0, -0.6], dtype=np.float32), _yaw_quat(0.0)),
        "Left_Foot": (
            np.array([0.0, 0.1, -1.5], dtype=np.float32),
            _yaw_quat(0.0),
        ),
        "Right_Foot": (
            np.array([0.0, -0.1, -1.6], dtype=np.float32),
            _yaw_quat(0.0),
        ),
    }

    grounded = calibrator.apply(frame)
    np.testing.assert_allclose(grounded["Pelvis"][0], [0.0, 0.0, 1.0])
    assert grounded["Left_Foot"][0][2] == pytest.approx(0.1)
    assert grounded["Right_Foot"][0][2] == pytest.approx(0.0)

    raised = {
        name: (position + np.array([0.0, 0.0, 0.2]), quaternion)
        for name, (position, quaternion) in frame.items()
    }
    raised_grounded = calibrator.apply(raised)
    assert raised_grounded["Pelvis"][0][2] == pytest.approx(1.2)
    assert raised_grounded["Right_Foot"][0][2] == pytest.approx(0.2)


def test_calibration_requires_both_feet() -> None:
    calibrator = PicoFrameCalibrator()
    frame = {
        "Pelvis": (np.zeros(3, dtype=np.float32), _yaw_quat(0.0)),
        "Left_Foot": (np.zeros(3, dtype=np.float32), _yaw_quat(0.0)),
    }

    with pytest.raises(ValueError, match="Right_Foot"):
        calibrator.apply(frame)


class _FakeXrt:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.timestamp_ns = 10
        self.global_timestamp_ns = 0
        self.poses = _body_poses()
        self.init_count = 0
        self.close_count = 0
        self.pose_calls = 0

    def init(self) -> bool:
        self.init_count += 1
        return True

    def close(self) -> None:
        self.close_count += 1

    def is_body_data_available(self) -> bool:
        return self.available

    def get_body_timestamp_ns(self) -> int:
        return self.timestamp_ns

    def get_time_stamp_ns(self) -> int:
        return self.global_timestamp_ns

    def get_body_joints_pose(self) -> np.ndarray:
        self.pose_calls += 1
        return self.poses


def test_source_uses_body_timestamp_and_drops_duplicates() -> None:
    xrt = _FakeXrt()
    source = PicoXrtSource(service_mode="external", xrt_module=xrt)
    source.start()

    first = source.read()
    duplicate = source.read()
    xrt.timestamp_ns = 20
    second = source.read()
    source.close()
    source.close()

    assert first is not None and first.timestamp_ns == 10
    assert duplicate is None
    assert second is not None and second.timestamp_ns == 20
    assert xrt.pose_calls == 2
    assert xrt.close_count == 1


def test_source_restart_accepts_a_reset_device_clock() -> None:
    xrt = _FakeXrt()
    source = PicoXrtSource(service_mode="external", xrt_module=xrt)
    source.start()
    assert source.read() is not None
    source.close()

    xrt.timestamp_ns = 5
    source.start()
    restarted = source.read()
    source.close()

    assert restarted is not None
    assert restarted.timestamp_ns == 5


def test_source_falls_back_to_global_timestamp() -> None:
    xrt = _FakeXrt()
    xrt.timestamp_ns = 0
    xrt.global_timestamp_ns = 100
    source = PicoXrtSource(service_mode="external", xrt_module=xrt)
    source.start()

    first = source.read()
    duplicate = source.read()
    xrt.global_timestamp_ns = 101
    second = source.read()
    source.close()

    assert first is not None and first.timestamp_ns == 100
    assert duplicate is None
    assert second is not None and second.timestamp_ns == 101
    assert xrt.pose_calls == 2


def test_source_locks_to_global_clock_after_body_clock_stalls() -> None:
    xrt = _FakeXrt()
    xrt.global_timestamp_ns = 100
    source = PicoXrtSource(service_mode="external", xrt_module=xrt)
    source.start()

    first = source.read()
    duplicate = source.read()
    xrt.global_timestamp_ns = 101
    fallback = source.read()
    xrt.timestamp_ns = 20
    late_body_only = source.read()
    xrt.global_timestamp_ns = 102
    next_global = source.read()
    source.close()

    assert first is not None and first.timestamp_ns == 10
    assert duplicate is None
    assert fallback is not None and fallback.timestamp_ns == 101
    assert late_body_only is None
    assert next_global is not None and next_global.timestamp_ns == 102
    assert xrt.pose_calls == 3


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


def test_source_startup_timeout_closes_sdk() -> None:
    xrt = _FakeXrt(available=False)
    clock = _FakeClock()
    source = PicoXrtSource(
        service_mode="external",
        startup_timeout_s=0.03,
        poll_interval_s=0.01,
        xrt_module=xrt,
        clock=clock,
        sleep=clock.sleep,
    )

    with pytest.raises(TimeoutError):
        source.start()
    assert xrt.init_count == 1
    assert xrt.close_count == 1


class _FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float) -> int:
        return 0


def test_source_only_stops_owned_service(tmp_path) -> None:
    service_path = tmp_path / "runService.sh"
    service_path.write_text("#!/bin/sh\n")
    process = _FakeProcess()
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return process

    source = PicoXrtSource(
        service_mode="auto",
        service_path=service_path,
        xrt_module=_FakeXrt(),
        process_factory=factory,
    )
    source.start()
    assert source.owns_service
    source.close()

    assert len(calls) == 1
    assert process.terminated
    assert not process.killed


def test_source_stops_owned_service_when_sdk_close_fails(tmp_path) -> None:
    class FailingCloseXrt(_FakeXrt):
        def close(self) -> None:
            raise RuntimeError("close failed")

    service_path = tmp_path / "runService.sh"
    service_path.write_text("#!/bin/sh\n")
    process = _FakeProcess()

    def factory(*args, **kwargs):
        del args, kwargs
        return process

    source = PicoXrtSource(
        service_mode="auto",
        service_path=service_path,
        xrt_module=FailingCloseXrt(),
        process_factory=factory,
    )
    source.start()

    with pytest.raises(RuntimeError, match="close"):
        source.close()
    assert process.terminated
    assert not source.owns_service
