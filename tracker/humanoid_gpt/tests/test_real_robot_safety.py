from __future__ import annotations

import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.constants import NUM_JOINT
from deploy.real_robot import (
    MAX_MOTOR_KD,
    MAX_MOTOR_KP,
    FastHGLowCmdPacker,
    LowLevelControlG1,
    RemoteController,
    _mode_machine_uint8,
    _UnitreeMotor,
    _validated_motor_gains,
    _validate_motor_target,
)


@pytest.mark.parametrize("container", [bytes, bytearray, list])
def test_remote_controller_parses_sdk_uint8_sequence_little_endian(container) -> None:
    payload = bytearray(40)
    struct.pack_into("<H", payload, 2, (1 << 2) | (1 << 3) | (1 << 8))
    struct.pack_into("<f", payload, 4, 0.25)
    struct.pack_into("<f", payload, 8, -0.5)
    struct.pack_into("<f", payload, 12, 0.75)
    struct.pack_into("<f", payload, 20, -1.0)
    remote = RemoteController()

    remote.set(container(payload))

    assert remote.button[2] == 1
    assert remote.button[3] == 1
    assert remote.button[8] == 1
    assert sum(remote.button) == 3
    assert remote.lx == pytest.approx(0.25)
    assert remote.rx == pytest.approx(-0.5)
    assert remote.ry == pytest.approx(0.75)
    assert remote.ly == pytest.approx(-1.0)

@pytest.mark.parametrize("length", [0, 24, 39, 41])
def test_remote_controller_rejects_wrong_payload_length(length: int) -> None:
    with pytest.raises(ValueError, match="exactly 40 bytes"):
        RemoteController().set([0] * length)

@pytest.mark.parametrize("value", [True, 5.0, "5", -1, 256])
def test_mode_machine_validator_rejects_non_uint8_values(value: object) -> None:
    with pytest.raises(RuntimeError, match="mode_machine"):
        _mode_machine_uint8(value, name="LowState mode_machine")

@pytest.mark.parametrize("value", [0, 5, 15, 255])
def test_mode_machine_validator_accepts_uint8_values(value: int) -> None:
    assert _mode_machine_uint8(value, name="LowState mode_machine") == value

def test_reviewed_motor_gain_bounds_accept_shipped_ranges() -> None:
    kps = _validated_motor_gains(
        np.full(NUM_JOINT, MAX_MOTOR_KP, dtype=np.float64),
        name="Kp",
        upper_bound=MAX_MOTOR_KP,
    )
    kds = _validated_motor_gains(
        np.full(NUM_JOINT, MAX_MOTOR_KD, dtype=np.float64),
        name="Kd",
        upper_bound=MAX_MOTOR_KD,
    )

    assert kps.shape == (NUM_JOINT,) and kps.dtype == np.float32
    assert kds.shape == (NUM_JOINT,) and kds.dtype == np.float32
    assert kps.flags.c_contiguous and kds.flags.c_contiguous

@pytest.mark.parametrize(
    ("value", "name", "upper_bound", "match"),
    [
        (np.zeros(NUM_JOINT - 1), "Kp", MAX_MOTOR_KP, "shape"),
        (np.full(NUM_JOINT, np.nan), "Kp", MAX_MOTOR_KP, "finite"),
        (np.full(NUM_JOINT, np.inf), "Kd", MAX_MOTOR_KD, "finite"),
        (np.full(NUM_JOINT, -0.1), "Kd", MAX_MOTOR_KD, "non-negative"),
        (
            np.full(NUM_JOINT, MAX_MOTOR_KP + 0.1),
            "Kp",
            MAX_MOTOR_KP,
            "reviewed limit",
        ),
        (
            np.full(NUM_JOINT, MAX_MOTOR_KD + 0.1),
            "Kd",
            MAX_MOTOR_KD,
            "reviewed limit",
        ),
    ],
)
def test_motor_gain_validation_rejects_unsafe_values(
    value: np.ndarray,
    name: str,
    upper_bound: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _validated_motor_gains(
            value,
            name=name,
            upper_bound=upper_bound,
        )

def test_low_level_wait_uses_configured_startup_timeout() -> None:
    waits: list[float] = []

    class ReadyEvent:
        def wait(self, timeout: float) -> bool:
            waits.append(timeout)
            return True

    controller = object.__new__(LowLevelControlG1)
    controller._low_state_ready = ReadyEvent()
    controller.low_state_startup_timeout_s = 1.25

    controller._wait_low_state()

    assert waits == [1.25]

def test_low_level_wait_times_out_without_first_low_state() -> None:
    class MissingEvent:

        def wait(_timeout: float) -> bool:
            return False

    controller = object.__new__(LowLevelControlG1)
    controller._low_state_ready = MissingEvent()
    controller.low_state_startup_timeout_s = 0.05

    with pytest.raises(TimeoutError, match="waiting for LowState"):
        controller._wait_low_state()

@pytest.mark.parametrize(
    ("target", "match"),
    [
        (np.zeros(NUM_JOINT - 1), "shape"),
        (np.full(NUM_JOINT, np.nan, dtype=np.float32), "NaN"),
        (np.full(NUM_JOINT, np.inf, dtype=np.float32), "NaN"),
    ],
)
def test_motor_target_validation_rejects_bad_shape_and_nonfinite(
    target: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_motor_target(target)

def test_low_level_fast_step_packs_unmodified_target_without_unitree_sdk() -> None:
    controller = object.__new__(LowLevelControlG1)
    controller._command_lock = threading.Lock()
    packed: list[np.ndarray] = []
    controller.fast_pack_motor_cmd = lambda target, _kps, _kds: packed.append(
        target.copy()
    )
    controller.fast_compute_crc = lambda: 0
    controller.fast_publish = lambda: None

    controller.fast_step(np.full(NUM_JOINT, 100.0, dtype=np.float32))

    assert len(packed) == 1
    np.testing.assert_allclose(
        packed[0], np.full(NUM_JOINT, 100.0, dtype=np.float32)
    )

@pytest.mark.parametrize(
    ("kps", "kds"),
    [
        (np.zeros(NUM_JOINT - 1), np.zeros(NUM_JOINT)),
        (np.full(NUM_JOINT, np.nan), np.zeros(NUM_JOINT)),
        (np.full(NUM_JOINT, -0.1), np.zeros(NUM_JOINT)),
        (np.full(NUM_JOINT, MAX_MOTOR_KP + 1.0), np.zeros(NUM_JOINT)),
        (np.zeros(NUM_JOINT), np.full(NUM_JOINT, np.inf)),
        (np.zeros(NUM_JOINT), np.full(NUM_JOINT, -0.1)),
        (np.zeros(NUM_JOINT), np.full(NUM_JOINT, MAX_MOTOR_KD + 1.0)),
    ],
)
def test_invalid_custom_gains_fail_before_pack_crc_or_publish(
    kps: np.ndarray,
    kds: np.ndarray,
) -> None:
    controller = object.__new__(LowLevelControlG1)
    controller._command_lock = threading.Lock()
    operations: list[str] = []
    controller.fast_pack_motor_cmd = lambda *_args: operations.append("pack")
    controller.fast_compute_crc = lambda: operations.append("crc")
    controller.fast_publish = lambda: operations.append("publish")

    with pytest.raises(ValueError):
        controller.fast_step(
            np.zeros(NUM_JOINT, dtype=np.float32),
            kps,
            kds,
        )

    assert operations == []

def test_fast_publish_raises_when_unitree_writer_returns_false() -> None:
    writes: list[object] = []

    class PublisherStub:
        @staticmethod
        def Write(message) -> bool:
            writes.append(message)
            return False

    controller = object.__new__(LowLevelControlG1)
    controller.debug = False
    controller._low_state_lock = threading.Lock()
    controller.mode_machine_ = 5
    controller.low_cmd = object()
    controller._pub = PublisherStub()
    controller._assert_publish_mode_machine_consistent = lambda **_kwargs: None

    with pytest.raises(RuntimeError, match="publish returned False"):
        controller.fast_publish()

    assert writes == [controller.low_cmd]

def test_fast_publish_debug_mode_never_calls_unitree_writer() -> None:
    class PublisherStub:
        @staticmethod
        def Write(_message) -> bool:
            raise AssertionError("debug mode must not call Unitree DDS Write")

    controller = object.__new__(LowLevelControlG1)
    controller.debug = True
    controller.low_cmd = object()
    controller._pub = PublisherStub()

    controller.fast_publish()

def test_fast_publish_echoes_observed_mode_machine_header() -> None:
    writes: list[object] = []

    class PublisherStub:
        @staticmethod
        def Write(message) -> bool:
            writes.append(message)
            return True

    controller = object.__new__(LowLevelControlG1)
    controller.debug = False
    controller._low_state_lock = threading.Lock()
    controller.mode_machine_ = 5
    controller.command_mode_machine_ = 5
    controller.low_cmd = SimpleNamespace(mode_machine=5)
    controller._fast_packer = SimpleNamespace(
        _buf=np.asarray([0, 5], dtype=np.uint8)
    )
    controller._pub = PublisherStub()

    controller.fast_publish()

    assert writes == [controller.low_cmd]
    assert writes[0].mode_machine == 5

@pytest.mark.parametrize(
    ("observed", "initialized", "command", "packed"),
    [
        (15, 5, 5, 5),
        (5, 5, 15, 5),
        (5, 5, 5, 15),
    ],
)
def test_fast_publish_rejects_runtime_mode_machine_changes_before_write(
    observed: int,
    initialized: int,
    command: int,
    packed: int,
) -> None:
    writes: list[object] = []

    class PublisherStub:
        @staticmethod
        def Write(message) -> bool:
            writes.append(message)
            return True

    controller = object.__new__(LowLevelControlG1)
    controller.debug = False
    controller._low_state_lock = threading.Lock()
    controller.mode_machine_ = observed
    controller.command_mode_machine_ = initialized
    controller.low_cmd = SimpleNamespace(mode_machine=command)
    controller._fast_packer = SimpleNamespace(
        _buf=np.asarray([0, packed], dtype=np.uint8)
    )
    controller._pub = PublisherStub()

    with pytest.raises(RuntimeError, match="no command was written"):
        controller.fast_publish()

    assert writes == []

def test_normal_then_damping_fast_crc_matches_unitree_sdk_crc() -> None:
    sdk_default = pytest.importorskip("unitree_sdk2py.idl.default")
    sdk_crc = pytest.importorskip("unitree_sdk2py.utils.crc")
    low_cmd = sdk_default.unitree_hg_msg_dds__LowCmd_()
    _UnitreeMotor.init_cmd_hg(
        low_cmd,
        mode_machine=15,
        mode_pr=_UnitreeMotor.MotorMode.PR,
    )
    crc = sdk_crc.CRC()
    controller = object.__new__(LowLevelControlG1)
    controller.low_cmd = low_cmd
    controller.active_j2m_ids = list(range(NUM_JOINT))
    controller._active_ids_np = np.arange(NUM_JOINT, dtype=np.intp)
    controller._kps_f32 = np.linspace(10.0, 100.0, NUM_JOINT, dtype=np.float32)
    controller._kds_f32 = np.linspace(1.0, 8.0, NUM_JOINT, dtype=np.float32)
    controller._fast_packer = FastHGLowCmdPacker(low_cmd, crc.crc_lib)

    target = np.linspace(-0.25, 0.25, NUM_JOINT, dtype=np.float32)
    controller.fast_pack_motor_cmd(target)
    assert controller.fast_compute_crc() == crc.Crc(low_cmd)

    # Make the actual SDK dq/tau fields non-zero. Damping must clear the real
    # fields as well as the byte mirror, otherwise the two CRCs diverge.
    for motor in low_cmd.motor_cmd:
        motor.dq = 1.25
        motor.tau = -2.5
    controller.fast_pack_damping(kd_value=8.0)

    assert all(motor.dq == 0.0 for motor in low_cmd.motor_cmd)
    assert all(motor.tau == 0.0 for motor in low_cmd.motor_cmd)
    assert controller.fast_compute_crc() == crc.Crc(low_cmd)

def test_command_lock_serializes_normal_and_damping_transactions() -> None:
    controller = object.__new__(LowLevelControlG1)
    controller._command_lock = threading.Lock()
    operations: list[str] = []
    normal_pack_entered = threading.Event()
    allow_normal_to_finish = threading.Event()
    damping_started = threading.Event()
    damping_pack_entered = threading.Event()

    def pack_normal(_target, _kps, _kds) -> None:
        operations.append("normal_pack")
        normal_pack_entered.set()
        assert allow_normal_to_finish.wait(timeout=1.0)

    def pack_damping(*, kd_value: float) -> None:
        assert kd_value == 8.0
        operations.append("damping_pack")
        damping_pack_entered.set()

    def compute_crc() -> int:
        operations.append(f"{threading.current_thread().name}_crc")
        return 0

    def publish() -> None:
        operations.append(f"{threading.current_thread().name}_publish")

    controller.fast_pack_motor_cmd = pack_normal
    controller.fast_pack_damping = pack_damping
    controller.fast_compute_crc = compute_crc
    controller.fast_publish = publish

    normal_thread = threading.Thread(
        name="normal",
        target=controller.fast_step,
        args=(np.zeros(NUM_JOINT, dtype=np.float32),),
    )

    def damping_worker() -> None:
        damping_started.set()
        controller.set_motor_damping()

    damping_thread = threading.Thread(name="damping", target=damping_worker)
    normal_thread.start()
    assert normal_pack_entered.wait(timeout=1.0)
    damping_thread.start()
    assert damping_started.wait(timeout=1.0)
    assert not damping_pack_entered.wait(timeout=0.05)

    allow_normal_to_finish.set()
    normal_thread.join(timeout=1.0)
    damping_thread.join(timeout=1.0)

    assert not normal_thread.is_alive()
    assert not damping_thread.is_alive()
    assert operations == [
        "normal_pack",
        "normal_crc",
        "normal_publish",
        "damping_pack",
        "damping_crc",
        "damping_publish",
    ]

def test_damping_command_lock_timeout_is_bounded() -> None:
    controller = object.__new__(LowLevelControlG1)
    controller._command_lock = threading.Lock()
    controller._command_lock.acquire()

    try:
        with pytest.raises(TimeoutError, match="LowCmd command lock"):
            controller.set_motor_damping(lock_timeout_s=0.001)
    finally:
        controller._command_lock.release()

def test_slow_send_holds_command_lock() -> None:
    class LockSpy:
        def __init__(self) -> None:
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *_args) -> None:
            self.exited = True

    class LowCmdStub:
        crc = 0

    class CrcStub:
        @staticmethod
        def Crc(_cmd) -> int:
            return 123

    controller = object.__new__(LowLevelControlG1)
    lock = LockSpy()
    controller._command_lock = lock
    controller.low_cmd = LowCmdStub()
    controller._crc = CrcStub()
    controller.debug = True

    controller._send()

    assert lock.entered and lock.exited
    assert controller.low_cmd.crc == 123
