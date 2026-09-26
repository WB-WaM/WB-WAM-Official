from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deploy.keyboard_cmd import DeployKeyboardCMD
from deploy.pico import PicoBodyFrame, pico_frame_to_gmr_frame
from deploy.play_track import (
    DeployArgs,
    LiveRefConverter,
    _advance_online_reference,
    _attempt_damping_sequence,
    _effective_buffer_ms,
    _release_active_motion_mode,
    _validate_motion_switcher_state,
)
from deploy.retarget import (
    _RETARGET_SESSIONS,
    MocapType,
    ema_smooth_g1_qpos,
    validate_g1_qpos,
    wait_realtime_retarget_ready,
)
from tracking import constants as consts

HUMANOID_GPT_ROOT = Path(__file__).resolve().parents[1]


def _qpos() -> np.ndarray:
    qpos = np.asarray(consts.DEFAULT_QPOS, dtype=np.float32).copy()
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    return qpos


@pytest.mark.parametrize("shape", [(35,), (37,), (1, 36)])
def test_qpos_validator_requires_exact_shape(shape) -> None:
    qpos = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        validate_g1_qpos(qpos)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_qpos_validator_rejects_nonfinite(value: float) -> None:
    qpos = _qpos()
    qpos[10] = value
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_g1_qpos(qpos)


def test_qpos_validator_normalizes_root_quaternion() -> None:
    qpos = _qpos()
    qpos[3] = 2.0
    result = validate_g1_qpos(qpos)
    assert result.dtype == np.float32
    assert np.linalg.norm(result[3:7]) == pytest.approx(1.0)

    qpos[3:7] = 0.0
    with pytest.raises(ValueError, match="quaternion"):
        validate_g1_qpos(qpos)


class _FakeRetargetProcess:
    def __init__(self, *, alive: bool, exitcode: int | None = None) -> None:
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self._alive


class _FakeReadyEvent:
    def __init__(self, ready: bool) -> None:
        self._ready = ready

    def is_set(self) -> bool:
        return self._ready

    def wait(self, timeout: float) -> bool:
        return self._ready


def test_wait_retarget_ready_accepts_ready_session_and_reports_worker_exit() -> None:
    ready_buf = object()
    dead_buf = object()
    _RETARGET_SESSIONS.extend(
        [
            {
                "buf": ready_buf,
                "proc": _FakeRetargetProcess(alive=True),
                "ready_evt": _FakeReadyEvent(True),
            },
            {
                "buf": dead_buf,
                "proc": _FakeRetargetProcess(alive=False, exitcode=1),
                "ready_evt": _FakeReadyEvent(False),
            },
        ]
    )
    try:
        wait_realtime_retarget_ready(ready_buf, timeout_s=1.0)
        with pytest.raises(RuntimeError, match="exitcode=1"):
            wait_realtime_retarget_ready(dead_buf, timeout_s=1.0)
    finally:
        _RETARGET_SESSIONS[:] = [
            session
            for session in _RETARGET_SESSIONS
            if session.get("buf") not in (ready_buf, dead_buf)
        ]


def test_ema_preserves_quaternion_sign_and_norm() -> None:
    previous = _qpos()
    current = _qpos()
    previous[7:] = 0.0
    current[3:7] = [-1.0, 0.0, 0.0, 0.0]
    current[7:] = 1.0

    smoothed = ema_smooth_g1_qpos(previous, current, alpha=0.75)
    np.testing.assert_allclose(smoothed[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(smoothed[7:], 0.25)
    assert np.linalg.norm(smoothed[3:7]) == pytest.approx(1.0)



def test_online_reference_pairing_matches_upstream_hgpt() -> None:
    first_qpos = _qpos()
    second_qpos = _qpos()
    second_qpos[7] += 0.25

    class Buffer:
        def __init__(self) -> None:
            self.frames = [first_qpos, second_qpos]

        def read(self):
            return self.frames.pop(0).copy(), 0.0

    class Converter:

        def convert(qpos):
            return {"qpos": qpos[None].copy()}

    buffer = Buffer()
    ref_curr, ref_next, previous = _advance_online_reference(
        buffer, Converter(), None
    )
    assert ref_curr is ref_next is previous
    np.testing.assert_allclose(ref_next["qpos"][0], first_qpos)

    ref_curr, ref_next, new_previous = _advance_online_reference(
        buffer, Converter(), previous
    )
    assert ref_curr is previous
    assert ref_next is new_previous
    np.testing.assert_allclose(ref_next["qpos"][0], second_qpos)


def test_main_locks_real_publish_without_explicit_opt_in(monkeypatch) -> None:
    import deploy.play_track as play_track

    calls: list[str] = []
    monkeypatch.setattr(play_track, "run_real", lambda _args: calls.append("real"))
    monkeypatch.setattr(play_track, "run_sim", lambda _args: calls.append("sim"))

    args = DeployArgs(real=True, debug=False, allow_real_publish=False)
    with pytest.raises(RuntimeError, match="publishing is locked"):
        play_track.main(args)

    assert calls == []


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (DeployArgs(real=False), "sim"),
        (DeployArgs(real=True, debug=True), "real"),
        (
            DeployArgs(real=True, debug=False, allow_real_publish=True),
            "real",
        ),
    ],
)
def test_main_routes_only_safe_or_explicit_real_modes(
    monkeypatch, args: DeployArgs, expected: str
) -> None:
    import deploy.play_track as play_track

    calls: list[str] = []
    monkeypatch.setattr(play_track, "run_real", lambda _args: calls.append("real"))
    monkeypatch.setattr(play_track, "run_sim", lambda _args: calls.append("sim"))

    play_track.main(args)

    assert calls == [expected]


def test_pico_has_zero_default_jitter_buffer() -> None:
    assert _effective_buffer_ms(MocapType.PICO, None) == 0.0
    assert _effective_buffer_ms(MocapType.PNLINK, None) == 30.0
    assert _effective_buffer_ms(MocapType.PICO, 12.0) == 12.0


def _synthetic_pico_frame() -> PicoBodyFrame:
    poses = np.zeros((24, 7), dtype=np.float32)
    poses[:, 1] = np.linspace(0.0, 1.7, 24)
    poses[:, 6] = 1.0
    return PicoBodyFrame(1, poses)


def test_synthetic_pico_to_qpos_to_live_reference() -> None:
    mujoco = pytest.importorskip("mujoco")
    del mujoco

    class FakeGmr:
        def retarget(self, frame):
            assert set(frame) == {
                "Pelvis",
                "Left_Hip",
                "Right_Hip",
                "Spine1",
                "Left_Knee",
                "Right_Knee",
                "Spine2",
                "Left_Ankle",
                "Right_Ankle",
                "Spine3",
                "Left_Foot",
                "Right_Foot",
                "Neck",
                "Left_Collar",
                "Right_Collar",
                "Head",
                "Left_Shoulder",
                "Right_Shoulder",
                "Left_Elbow",
                "Right_Elbow",
                "Left_Wrist",
                "Right_Wrist",
                "Left_Hand",
                "Right_Hand",
            }
            return _qpos()

    gmr_frame = pico_frame_to_gmr_frame(_synthetic_pico_frame())
    qpos = validate_g1_qpos(FakeGmr().retarget(gmr_frame))

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(HUMANOID_GPT_ROOT / consts.TRACK_XML))
    reference = LiveRefConverter(model, ctrl_dt=0.02).convert(qpos)
    assert reference["qpos"].shape == (1, 36)
    assert reference["qvel"].shape == (1, 35)
    assert reference["kpt2gv_pose"].shape[-2:] == (4, 4)
    assert reference["kpt_cvel_in_gv"].shape[-1] == 6
    assert all(np.isfinite(value).all() for value in reference.values())


def test_headless_onnx_policy_step_if_checkpoint_is_available() -> None:
    pytest.importorskip("onnxruntime")
    checkpoint = HUMANOID_GPT_ROOT / "storage/ckpts/pns_wo_priv216.onnx"
    if not checkpoint.is_file():
        pytest.skip("tracking checkpoint is not installed")

    from tracking.infer_utils import G1TrackInferFn, G1TrackMjSim, g1_infer_env_config
    from tracking.policy import Args as PolicyArgs, get_policy_onnx

    policy = get_policy_onnx(PolicyArgs(load_path=str(checkpoint), policy_type="mlp"))
    sim = G1TrackMjSim(
        init_qpos=_qpos(),
        headless=True,
        ctrl_dt=0.02,
        xml_path=str(HUMANOID_GPT_ROOT / consts.TRACK_XML),
    )
    state = sim.reset(sim.init_state())
    infer = G1TrackInferFn(
        g1_infer_env_config(ctrl_dt=0.02),
        sim.mj_model,
        policy,
        privileged=False,
    )
    reference = LiveRefConverter(sim.mj_model, ctrl_dt=0.02).convert(_qpos())
    action = infer.infer_onnx(
        state,
        {"ref_curr": reference, "ref_next": reference},
    )
    assert np.asarray(action).shape == (1, 29)
    assert np.isfinite(action).all()
    state = sim.step(state, action)
    assert np.isfinite(state.mj_data.qpos).all()


def test_keyboard_gui_exit_maps_to_kill() -> None:
    class DeadCommandPad:
        is_alive = False

    controller = object.__new__(DeployKeyboardCMD)
    controller._cmd_pad = DeadCommandPad()
    controller.mode = 1

    command = controller.step_command()

    assert command.kill
    assert command.mode == 1


def test_motion_switcher_state_validation_is_fail_closed() -> None:
    assert _validate_motion_switcher_state(0, {"name": ""}) == ""
    assert _validate_motion_switcher_state(0, {"name": "sport"}) == "sport"

    with pytest.raises(RuntimeError, match="CheckMode failed"):
        _validate_motion_switcher_state(1, None)
    with pytest.raises(RuntimeError, match="invalid result"):
        _validate_motion_switcher_state(0, None)
    with pytest.raises(RuntimeError, match="invalid mode name"):
        _validate_motion_switcher_state(0, {"name": None})


class _FakeMotionSwitcher:
    def __init__(
        self,
        modes: list[str],
        *,
        release_status: int = 0,
    ) -> None:
        self._modes = list(modes)
        self._check_index = 0
        self.release_status = release_status
        self.release_calls = 0

    def CheckMode(self):
        index = min(self._check_index, len(self._modes) - 1)
        self._check_index += 1
        return 0, {"name": self._modes[index]}

    def ReleaseMode(self):
        self.release_calls += 1
        return self.release_status, None


def test_motion_switcher_releases_ai_before_lowcmd() -> None:
    switcher = _FakeMotionSwitcher(["ai", "ai", ""])
    sleeps: list[float] = []

    _release_active_motion_mode(
        switcher,
        max_attempts=3,
        retry_s=0.25,
        sleep_fn=sleeps.append,
    )

    assert switcher.release_calls == 2
    assert sleeps == [0.25, 0.25]


def test_motion_switcher_release_failure_is_fail_closed() -> None:
    switcher = _FakeMotionSwitcher(["ai"], release_status=7)
    with pytest.raises(RuntimeError, match="ReleaseMode failed with status 7"):
        _release_active_motion_mode(switcher, sleep_fn=lambda _: None)

    stuck = _FakeMotionSwitcher(["ai"])
    with pytest.raises(RuntimeError, match="remained active after 2"):
        _release_active_motion_mode(
            stuck,
            max_attempts=2,
            retry_s=0.0,
            sleep_fn=lambda _: None,
        )


def test_main_rejects_debug_hand_before_real_runtime(monkeypatch) -> None:
    import deploy.play_track as play_track

    calls: list[str] = []
    monkeypatch.setattr(play_track, "run_real", lambda _args: calls.append("real"))

    with pytest.raises(RuntimeError, match="HandCmd has no debug publication gate"):
        play_track.main(
            DeployArgs(real=True, debug=True, enable_hand=True),
        )

    assert calls == []


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (
            DeployArgs(real=True, debug=False, allow_real_publish=False),
            "publishing is locked",
        ),
        (
            DeployArgs(real=True, debug=True, enable_hand=True),
            "HandCmd has no debug publication gate",
        ),
    ],
)
def test_run_real_enforces_access_gates_before_any_runtime_init(
    args: DeployArgs,
    match: str,
) -> None:
    import deploy.play_track as play_track

    with pytest.raises(RuntimeError, match=match):
        play_track.run_real(args)


def test_damping_sequence_can_be_retried_after_a_lock_timeout() -> None:
    class FlakyLowControl:
        def __init__(self) -> None:
            self.attempts = 0

        def set_motor_damping(self, *, lock_timeout_s: float) -> None:
            assert lock_timeout_s == 1.0
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("busy command lock")

    controller = FlakyLowControl()
    sleeps: list[float] = []

    first_error = _attempt_damping_sequence(
        controller,
        duration_s=0.02,
        ctrl_dt=0.02,
        lock_timeout_s=1.0,
        sleep_fn=sleeps.append,
    )
    assert isinstance(first_error, TimeoutError)

    second_error = _attempt_damping_sequence(
        controller,
        duration_s=0.02,
        ctrl_dt=0.02,
        lock_timeout_s=1.0,
        sleep_fn=sleeps.append,
    )
    assert second_error is None
    assert controller.attempts == 2
    assert sleeps == [0.02]
