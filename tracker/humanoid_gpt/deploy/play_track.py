"""Main entry point for tracking deployment (simulation and real robot).

Usage:
    # Simulation (offline tracking)
    python -m deploy.play_track --track_dir storage/data/exps

    # Simulation (online retarget, Noitom PNLink)
    python -m deploy.play_track --track_dir storage/data/exps --mocap_type pnlink

    # Simulation (online retarget, Xsens MVN over TCP on port 9763)
    python -m deploy.play_track --track_dir storage/data/exps \
        --mocap_type xsens --xsens_protocol tcp --xsens_port 9763

    # Real robot
    python -m deploy.play_track --real --net enx6c1ff7579fdf

Modes (keyboard number keys):
    0 = Walk policy
    1 = Online retarget (mocap)
    2+ = Offline tracking (reference trajectories from track_dir)
"""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro
from jax import tree_util as jtu
from loop_rate_limiters import RateLimiter

from deploy.constants import (
    DEFAULT_QPOS as DEFAULT_QPOS_JOINT,
    KDs_walking,
    KPs_walking,
)
from deploy.keyboard_cmd import DeployKeyboardCMD
from deploy.reference_alignment import (
    align_qpos_first_frame,
    align_reference_trajectory_first_frame,
    quat_yaw_wxyz,
)
from deploy.walk_policy import WalkPolicy
from tracking import constants as consts
from tracking.constants import KPT_NAMES
from tracking.convert_qpos2kpt import qpos2kpt
from tracking.infer_utils import (
    G1TrackInferFn,
    G1TrackMjSim,
    apply_ema_qpos,
    g1_infer_env_config,
)
from tracking.metrics import (
    calculate_joint_tracking_error,
    calculate_kpt_mae_error,
    calculate_max_errors,
    calculate_root_tracking_error,
    calculate_trajectory_length,
)
from tracking.policy import Args as PolicyArgs, get_policy_onnx
from utils.sim_mj import get_sensor_data as mj_sensor
from utils.transforms_np import base2navi, quat2mat

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402  – needed for DeployKeyboardCMD fork workaround


def _resolve_mocap_type(name: str):
    """Map the ``--mocap_type`` string to a :class:`MocapType` enum value.

    Kept here (instead of inside each run_* function) so both
    simulation and real-robot loops route mocap selection identically.
    """
    from deploy.retarget import MocapType

    key = (name or "").lower()
    if key == "pico":
        return MocapType.PICO
    if key == "pnlink":
        return MocapType.PNLINK
    if key == "xsens":
        return MocapType.XSENS
    if key == "optitrack":
        return MocapType.OPTITRACK
    raise ValueError(
        f"Unknown mocap type {name!r}; expected pico, pnlink, xsens, or optitrack"
    )


def _effective_buffer_ms(mocap_type, configured: float | None) -> float:
    if configured is not None:
        return float(configured)
    return 0.0 if mocap_type.name == "PICO" else 30.0


# ---------------------------------------------------------------------------
# Mocap buffer with caching and linear extrapolation
# ---------------------------------------------------------------------------


class MocapBuffer:
    """Read the latest mocap data from the shared buffer."""

    def __init__(self, buf, ts):
        self._buf = buf
        self._ts = ts

    def read(self) -> tuple[np.ndarray, float]:
        from deploy.retarget import read_mocap_buffer

        qpos_full, ts = read_mocap_buffer(self._buf, self._ts)
        return qpos_full, ts


def _reset_tracking_history(infer_fn, live_converter) -> None:
    infer_fn.info["last_action"][:] = 0
    infer_fn.info["step"] = 0
    live_converter.reset()


def _advance_online_reference(mocap_buffer, converter, previous_reference):
    """Read and convert one online frame using upstream HGPT reference pairing."""
    qpos_full, _ = mocap_buffer.read()
    reference = converter.convert(qpos_full)
    ref_curr = reference if previous_reference is None else previous_reference
    return ref_curr, reference, reference


# ---------------------------------------------------------------------------
# Live reference converter: qpos_full -> ref_state dict for G1TrackInferFn
# ---------------------------------------------------------------------------


def _batch_rot_log_so3(R: np.ndarray) -> np.ndarray:
    """Batch SO(3) log for K rotation matrices. R: (K,3,3) -> (K,3)."""
    tr = np.trace(R, axis1=1, axis2=2)  # (K,)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (K,)
    skew = np.stack(
        [
            R[:, 2, 1] - R[:, 1, 2],
            R[:, 0, 2] - R[:, 2, 0],
            R[:, 1, 0] - R[:, 0, 1],
        ],
        axis=1,
    )  # (K, 3)
    small = theta < 1e-6
    safe_theta = np.where(small, 1.0, theta)
    k = np.where(small, 0.5, theta / (2.0 * np.sin(safe_theta)))
    return k[:, None] * skew


def _batch_pose_delta_to_twist(
    T_prev: np.ndarray, T_curr: np.ndarray, dt: float
) -> np.ndarray:
    """Batch compute 6D twists (ang, lin) in world. T: (K,4,4) -> (K,6)."""
    R0, p0 = T_prev[:, :3, :3], T_prev[:, :3, 3]
    R1, p1 = T_curr[:, :3, :3], T_curr[:, :3, 3]
    v_w = (p1 - p0) / dt
    R_delta = np.einsum("kij,kjl->kil", R0.transpose(0, 2, 1), R1)
    rotvec_body = _batch_rot_log_so3(R_delta)
    w_w = np.einsum("kij,kj->ki", R0, rotvec_body / dt)
    return np.concatenate([w_w, v_w], axis=1).astype(np.float32)


def _quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (z-rotation) from a wxyz quaternion."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _wrap_to_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


class LiveRefConverter:
    """Convert a raw qpos_full (from mocap) to the ref_state dict expected
    by G1TrackInferFn, computing keypoint poses via MuJoCo FK.

    Keypoint velocities are computed via finite differences between
    consecutive frames, matching the pre-computed kpt_cvel_in_gv used
    in training reference data.

    For real-robot deployment, call ``set_robot_initial_pose()`` when
    entering tracking mode so that the reference root pose is rebiased
    to align with the robot's initial frame (IMU yaw + phantom [0,0,z]).
    """

    def __init__(self, mj_model: mujoco.MjModel, ctrl_dt: float):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)
        self.ctrl_dt = ctrl_dt
        self.kpt_body_ids = np.array([mj_model.body(n).id for n in KPT_NAMES])
        self._prev_kpt2wrd_pose = None
        self._prev_gv2wrd_pose = None
        self._prev_qpos = None
        # Initial-pose calibration (P3): set via set_robot_initial_pose()
        self._ref_init_xy = None
        self._ref_init_yaw = None
        self._robot_init_xy = None
        self._robot_init_yaw = None

    def reset(self):
        self._prev_kpt2wrd_pose = None
        self._prev_gv2wrd_pose = None
        self._prev_qpos = None
        self._ref_init_xy = None
        self._ref_init_yaw = None
        self._robot_init_xy = None
        self._robot_init_yaw = None

    def set_robot_initial_pose(self, robot_quat: np.ndarray, robot_xy: np.ndarray):
        """Record the robot's pose at the moment tracking begins.
        Called once per tracking session so that reference poses can be
        rebiased into the robot's coordinate frame."""
        self._robot_init_yaw = _quat_to_yaw(robot_quat)
        self._robot_init_xy = robot_xy[:2].copy()

    def _rebias_qpos(self, qpos_full: np.ndarray) -> np.ndarray:
        """Rebias reference root (xy, yaw) so that the first frame aligns
        with the robot's initial pose recorded by set_robot_initial_pose().
        This makes update_coord_cmd() compute correct relative differences
        on the real robot where the phantom state starts at [0,0,0.78]."""
        if self._robot_init_yaw is None:
            return qpos_full

        qpos = qpos_full.copy()

        # Capture reference initial pose on first call
        if self._ref_init_xy is None:
            self._ref_init_yaw = _quat_to_yaw(qpos[3:7])
            self._ref_init_xy = qpos[:2].copy()

        # Compute delta from reference initial
        ref_yaw = _quat_to_yaw(qpos[3:7])
        d_yaw = _wrap_to_pi(ref_yaw - self._ref_init_yaw)
        d_xy = qpos[:2] - self._ref_init_xy

        # Rotate d_xy by the yaw offset between robot and reference initial
        yaw_offset = _wrap_to_pi(self._robot_init_yaw - self._ref_init_yaw)
        c, s = np.cos(yaw_offset), np.sin(yaw_offset)
        rotated_dxy = np.array(
            [c * d_xy[0] - s * d_xy[1], s * d_xy[0] + c * d_xy[1]], dtype=np.float32
        )

        # Apply to robot initial pose
        qpos[:2] = self._robot_init_xy + rotated_dxy
        new_yaw = self._robot_init_yaw + d_yaw

        # Apply yaw correction in WORLD frame: q_new = q_dz * q_ref.
        # Right-multiplication applies a BODY-frame yaw and distorts roll/pitch
        # when the torso is not upright (e.g. bending/squatting).
        d_yaw_apply = _wrap_to_pi(new_yaw - ref_yaw)
        c2, s2 = np.cos(d_yaw_apply / 2), np.sin(d_yaw_apply / 2)
        q_dz = np.array([c2, 0.0, 0.0, s2], dtype=np.float32)
        q_new = _quat_mul_wxyz(q_dz, qpos[3:7])
        qpos[3:7] = q_new / np.clip(np.linalg.norm(q_new), 1e-8, None)

        return qpos

    def convert(self, qpos_full: np.ndarray) -> dict:
        """Convert a single qpos_full (36,) to ref_state dict (batch dim = 1)."""
        qpos_full = self._rebias_qpos(qpos_full)
        self.mj_data.qpos[:] = qpos_full
        self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Gravity-view frame
        base2wrd_rot = quat2mat(qpos_full[3:7])
        gvi2wrd_rot = base2navi(base2wrd_rot)
        gvi2wrd_pose = np.eye(4, dtype=np.float32)
        gvi2wrd_pose[:2, 3] = qpos_full[:2]
        gvi2wrd_pose[:3, :3] = gvi2wrd_rot

        # Keypoint poses in world (vectorised)
        num_kpt = len(self.kpt_body_ids)
        kpt2wrd_pose = np.tile(np.eye(4, dtype=np.float32), (num_kpt, 1, 1))
        kpt2wrd_pose[:, :3, 3] = self.mj_data.xpos[self.kpt_body_ids]
        kpt2wrd_pose[:, :3, :3] = self.mj_data.xmat[self.kpt_body_ids].reshape(-1, 3, 3)

        # Transform to gravity-view
        kpt2gv_pose = np.linalg.inv(gvi2wrd_pose) @ kpt2wrd_pose  # (K,4,4)

        # Keypoint velocities via finite differences (vectorised, no scipy)
        kpt_cvel_in_wrd = np.zeros((num_kpt, 6), dtype=np.float32)
        if self._prev_kpt2wrd_pose is not None:
            kpt_cvel_in_wrd = _batch_pose_delta_to_twist(
                self._prev_kpt2wrd_pose, kpt2wrd_pose, self.ctrl_dt
            )
        self._prev_kpt2wrd_pose = kpt2wrd_pose.copy()

        # Rotate world-frame velocities to gravity-view frame
        R_wrd2gv = gvi2wrd_pose[:3, :3].T
        kpt_cvel_in_gv = np.zeros_like(kpt_cvel_in_wrd)
        kpt_cvel_in_gv[:, :3] = kpt_cvel_in_wrd[:, :3] @ R_wrd2gv.T
        kpt_cvel_in_gv[:, 3:] = kpt_cvel_in_wrd[:, 3:] @ R_wrd2gv.T

        # Gravity-view velocity (root frame linear + yaw angular)
        gv_vel = np.zeros(3, dtype=np.float32)
        if self._prev_gv2wrd_pose is not None:
            curr2prev = np.linalg.inv(self._prev_gv2wrd_pose) @ gvi2wrd_pose
            gv_vel[:2] = curr2prev[:2, 3] / self.ctrl_dt
            gv_vel[2] = np.arctan2(curr2prev[1, 0], curr2prev[0, 0]) / self.ctrl_dt
        self._prev_gv2wrd_pose = gvi2wrd_pose.copy()

        # Joint velocity from finite differences
        qvel = np.zeros(35, dtype=np.float32)
        if self._prev_qpos is not None:
            qvel[6:] = (qpos_full[7:] - self._prev_qpos[7:]) / self.ctrl_dt
        self._prev_qpos = qpos_full.copy()

        # Build ref_state dict with batch dimension
        return {
            "qpos": qpos_full[None].astype(np.float32),
            "qvel": qvel[None].astype(np.float32),
            "kpt2gv_pose": kpt2gv_pose[None].astype(np.float32),
            "kpt_cvel_in_gv": kpt_cvel_in_gv[None].astype(np.float32),
            "gv2wrd_pose": gvi2wrd_pose[None].astype(np.float32),
            "gv_vel": gv_vel[None].astype(np.float32),
        }


# ---------------------------------------------------------------------------
# Reference motion loading
# ---------------------------------------------------------------------------


def load_offline_motions(
    track_dir: str, mj_model: mujoco.MjModel, freq: int = 50
) -> list[dict]:
    """Load and first-frame-align .npz trajectories before KPT conversion.

    Returns list of dicts. Each dict has numpy-array fields (safe for
    jtu.tree_map) plus a ``_filename`` key that is excluded before tree_map.
    """
    folder = Path(track_dir)
    files = [folder] if folder.is_file() else sorted(folder.glob("*.npz"))

    motions = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        if "qpos" not in data and {"root_pos", "root_rot", "dof_pos"} <= data.keys():
            data["qpos"] = np.concatenate(
                [data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1
            )
        if "qpos" not in data:
            print(f"[WARN] Skipping {f.name}: no qpos field")
            continue

        raw_qpos = np.asarray(data["qpos"], dtype=np.float32)
        source_xy = raw_qpos[0, :2].copy()
        source_yaw = quat_yaw_wxyz(raw_qpos[0, 3:7])
        target_yaw = quat_yaw_wxyz(consts.DEFAULT_QPOS[3:7])
        data["qpos"] = align_qpos_first_frame(
            raw_qpos,
            target_xy=consts.DEFAULT_QPOS[:2],
            target_yaw=target_yaw,
        )
        data["qpos"] = apply_ema_qpos(data["qpos"])
        freq_src = float(data.get("frequency", 50))
        kpt_data = qpos2kpt(
            mj_model,
            np.float32(data["qpos"]),
            freq_src=freq_src,
            freq_tgt=freq,
            interp_sec=0.5,
            end_default_sec=0.5,
            debug=False,
            foot_contact_est=False,
            height_clip_mode=None,
            video_path=None,
        )
        motions.append({"data": kpt_data, "filename": f.name})
        print(
            "    first-frame SE(2): "
            f"xy=({source_xy[0]:.3f},{source_xy[1]:.3f}), "
            f"yaw={np.degrees(source_yaw):.1f}deg -> "
            f"xy=({consts.DEFAULT_QPOS[0]:.3f},{consts.DEFAULT_QPOS[1]:.3f}), "
            f"yaw={np.degrees(target_yaw):.1f}deg"
        )
        print(f"  Mode {len(motions) + 1}: {f.name} ({len(kpt_data['qpos'])} frames)")

    return motions


# ---------------------------------------------------------------------------
# Offline tracking metrics (same format as inference.py)
# ---------------------------------------------------------------------------


def _print_offline_metrics(
    traj_metrics: dict, filename: str, ref_traj: dict, mj_model
) -> None:
    """Print tracking error metrics in the same format as scripts/inference.py."""
    actual_traj_len = len(ref_traj["qpos"])
    traj_length_ratio, termination_step = calculate_trajectory_length(
        traj_metrics["state_history"], ref_traj, mj_model
    )
    avg_kpt_pos = np.mean(traj_metrics["kpt_pos_errors"])
    avg_kpt_rot = np.mean(traj_metrics["kpt_rot_errors"])
    avg_joint_pos = np.mean(traj_metrics["joint_pos_errors"])
    avg_joint_vel = np.mean(traj_metrics["joint_vel_errors"])
    avg_root_pos = np.mean(traj_metrics["root_pos_errors"])
    avg_root_vel = np.mean(traj_metrics["root_vel_errors"])
    avg_root_yaw = np.mean(traj_metrics["root_yaw_errors"])
    max_errors = calculate_max_errors(traj_metrics)

    print(f"\n  [Offline Track] {filename} completed:")
    print(
        f"    Completion: {traj_length_ratio:.4f} ({termination_step}/{actual_traj_len} steps)"
    )
    print(
        f"    KPT Position MAE: {avg_kpt_pos:.6f} m (Max: {max_errors['max_kpt_pos_error']:.6f} m)"
    )
    print(
        f"    KPT Rotation MAE: {avg_kpt_rot:.6f} rad (Max: {max_errors['max_kpt_rot_error']:.6f} rad)"
    )
    print(
        f"    Joint Position MAE: {avg_joint_pos:.6f} rad (Max: {max_errors['max_joint_pos_error']:.6f} rad)"
    )
    print(
        f"    Joint Velocity MAE: {avg_joint_vel:.6f} rad/s (Max: {max_errors['max_joint_vel_error']:.6f} rad/s)"
    )
    print(
        f"    Root Pos Error: {avg_root_pos:.3f} mm (Max: {max_errors['max_root_pos_error']:.3f} mm)"
    )
    print(
        f"    Root Vel Error: {avg_root_vel:.3f} mm/s (Max: {max_errors['max_root_vel_error']:.3f} mm/s)"
    )
    print(
        f"    Root Yaw Error: {avg_root_yaw:.6f} rad (Max: {max_errors['max_root_yaw_error']:.6f} rad)\n"
    )


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


def run_sim(args: DeployArgs):
    if args.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)

    # Load ONNX tracking policy
    policy_args = PolicyArgs(
        load_path=args.onnx_track,
        policy_type=args.policy_type,
    )
    track_policy = get_policy_onnx(policy_args)

    # Load walk policy
    walk_policy = WalkPolicy(args.onnx_walk)

    # Load offline reference motions
    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    print("Loading offline reference motions...")
    print("  Mode 0: Walk")
    print("  Mode 1: Online retarget")
    ref_motions = load_offline_motions(args.track_dir, convert_model, freq)

    if args.pose_endpoint:
        from deploy.pose_zmq import ZmqPoseSource

        keyboard = ZmqPoseSource(args.pose_endpoint)
        print(f"[PoseZMQ] Connected to {args.pose_endpoint}")
    else:
        keyboard = DeployKeyboardCMD(num_track_ref=len(ref_motions))

    # MuJoCo sim (correct sim_dt=0.001 from Humanoid-GPT)
    init_qpos = consts.DEFAULT_QPOS.copy()
    mj_sim = G1TrackMjSim(init_qpos=init_qpos, headless=args.headless, ctrl_dt=ctrl_dt)
    infer_fn = G1TrackInferFn(env_cfg, mj_sim.mj_model, track_policy, privileged=False)
    state = mj_sim.init_state()
    state = mj_sim.reset(state)

    # Camera view: match inference.py
    if not mj_sim.headless and mj_sim.viewer is not None:
        viewer_cam = mj_sim.viewer.cam
        viewer_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer_cam.trackbodyid = 0
        viewer_cam.azimuth = 90.0
        viewer_cam.elevation = -20.0
        viewer_cam.distance = 2.0

    # Live retarget converter
    live_converter = LiveRefConverter(mj_sim.mj_model, ctrl_dt)

    # Online retarget (optional)
    mocap_buffer = None
    buf_mocap = None
    buf_hand = None
    if args.pose_endpoint:
        mocap_buffer = keyboard
        print("[PoseZMQ] Manager provides qpos36, mode, velocity, and reset")
    elif not args.no_mocap:
        try:
            from deploy.retarget import start_realtime_retarget

            mocap_type = _resolve_mocap_type(args.mocap_type)
            buffer_ms = _effective_buffer_ms(mocap_type, args.buffer_ms)
            buf_mocap, ts_mocap, buf_hand = start_realtime_retarget(
                server_ip=args.server_ip,
                client_ip=args.client_ip,
                robot="unitree_g1",
                dof_full=7 + 29,
                actual_human_height=args.human_height,
                visualize_retarget=args.visualize_retarget and not args.headless,
                mocap_type=mocap_type,
                buffer_ms=buffer_ms,
                pico_service_mode=args.pico_service_mode,
                pico_startup_timeout_s=args.pico_startup_timeout_s,
                xsens_host=args.xsens_host,
                xsens_port=args.xsens_port,
                xsens_protocol=args.xsens_protocol,
            )
            mocap_buffer = MocapBuffer(buf_mocap, ts_mocap)
            print("[Mocap] Retarget subprocess started.")
        except Exception as e:
            print(f"[Mocap] Failed to start retarget: {e}. Online mode disabled.")

    rate = RateLimiter(frequency=freq, warn=False)
    prev_online_ref = None
    last_mode = 0
    track_step = 0
    total_steps = 0
    ref_traj = None
    traj_metrics = None  # For offline tracking: kpt/joint/root errors, state_history
    traj_filename = None

    print("\n=== Simulation ready. Press number keys to switch modes. ===\n")

    try:
        while True:
            if keyboard.check_reset_request():
                state = mj_sim.reset(state)
                _reset_tracking_history(infer_fn, live_converter)
                if last_mode >= 2 and ref_traj is not None:
                    track_step = 0
                    traj_metrics = {
                        "kpt_pos_errors": [],
                        "kpt_rot_errors": [],
                        "joint_pos_errors": [],
                        "joint_vel_errors": [],
                        "root_pos_errors": [],
                        "root_vel_errors": [],
                        "root_yaw_errors": [],
                        "state_history": [],
                    }
                if last_mode == 1:
                    prev_online_ref = None
                    if buf_mocap is not None:
                        from deploy.retarget import request_retarget_calibration

                        request_retarget_calibration(buf_mocap)
                    live_converter.set_robot_initial_pose(
                        state.mj_data.qpos[3:7],
                        state.mj_data.qpos[:2],
                    )
                print("[Reset] Simulation reset.")

            cmd = keyboard.step_command()
            mode = cmd.mode

            # Mode transitions
            entering_track = mode >= 1 and mode != last_mode
            leaving_track = (last_mode >= 1) and (mode == 0)

            if entering_track:
                _reset_tracking_history(infer_fn, live_converter)
                if mode == 1:
                    if buf_mocap is not None:
                        from deploy.retarget import request_retarget_calibration

                        request_retarget_calibration(buf_mocap)
                    prev_online_ref = None
                    live_converter.set_robot_initial_pose(
                        state.mj_data.qpos[3:7],
                        state.mj_data.qpos[:2],
                    )
                    print("[Track] Online retarget reset")
                elif mode >= 2:
                    traj_idx = mode - 2
                    if traj_idx < len(ref_motions):
                        ref_traj = ref_motions[traj_idx]["data"]
                        track_step = 0
                        traj_filename = ref_motions[traj_idx]["filename"]
                        traj_metrics = {
                            "kpt_pos_errors": [],
                            "kpt_rot_errors": [],
                            "joint_pos_errors": [],
                            "joint_vel_errors": [],
                            "root_pos_errors": [],
                            "root_vel_errors": [],
                            "root_yaw_errors": [],
                            "state_history": [],
                        }
                        print(f"[Track] Start offline: {traj_filename}")
                    else:
                        print(f"[Track] Invalid trajectory index {traj_idx}")
                        mode = 0

            if leaving_track:
                if (
                    last_mode >= 2
                    and traj_metrics is not None
                    and len(traj_metrics["state_history"]) > 0
                ):
                    _print_offline_metrics(
                        traj_metrics, traj_filename, ref_traj, mj_sim.mj_model
                    )
                traj_metrics = None
                traj_filename = None
                live_converter.reset()
                prev_online_ref = None
                print("[Track] Back to walk mode")

            if mode == 0:
                # Walk policy
                cmd_vel = np.array(
                    [cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw], dtype=np.float32
                )
                gyro = mj_sensor(mj_sim.mj_model, state.mj_data, "gyro_pelvis")
                motor_targets = walk_policy.infer(
                    state.mj_data.qpos[3:7],
                    gyro,
                    state.mj_data.qpos[7:],
                    state.mj_data.qvel[6:],
                    cmd_vel,
                )
                # Walk uses different PD gains than tracking
                for _ in range(mj_sim.num_sim_substeps):
                    torques = KPs_walking * (
                        motor_targets - state.mj_data.qpos[7:]
                    ) + KDs_walking * (-state.mj_data.qvel[6:])
                    torques = np.clip(
                        torques, -consts.TORQUE_LIMIT, consts.TORQUE_LIMIT
                    )
                    state.mj_data.ctrl[:] = torques
                    mujoco.mj_step(mj_sim.mj_model, state.mj_data)

            elif mode == 1:
                if mocap_buffer is not None:
                    ref_curr, ref_next, prev_online_ref = _advance_online_reference(
                        mocap_buffer, live_converter, prev_online_ref
                    )
                    motor_targets = infer_fn.infer_onnx(
                        state,
                        {"ref_curr": ref_curr, "ref_next": ref_next},
                    )
                    state = mj_sim.step(state, motor_targets)

            else:
                # Offline tracking (mode >= 2)
                if ref_traj is not None:
                    traj_len = len(ref_traj["qpos"])
                    ref_curr = jtu.tree_map(
                        lambda x, step=track_step: x[step][None], ref_traj
                    )
                    next_step = min(track_step + 1, traj_len - 1)
                    ref_next = jtu.tree_map(
                        lambda x, step=next_step: x[step][None], ref_traj
                    )
                    motor_targets = infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_next}
                    )
                    state = mj_sim.step(state, motor_targets)

                    # Collect metrics (same as inference.py)
                    if traj_metrics is not None:
                        kpt_pos_mae, kpt_rot_mae = calculate_kpt_mae_error(
                            state, ref_curr, ref_next, mj_sim.mj_model
                        )
                        joint_pos_mae, joint_vel_mae = calculate_joint_tracking_error(
                            state, ref_curr
                        )
                        root_pos_err_mm, root_vel_err_mms, root_yaw_err = (
                            calculate_root_tracking_error(state, ref_curr)
                        )
                        traj_metrics["kpt_pos_errors"].append(kpt_pos_mae)
                        traj_metrics["kpt_rot_errors"].append(kpt_rot_mae)
                        traj_metrics["joint_pos_errors"].append(joint_pos_mae)
                        traj_metrics["joint_vel_errors"].append(joint_vel_mae)
                        traj_metrics["root_pos_errors"].append(root_pos_err_mm)
                        traj_metrics["root_vel_errors"].append(root_vel_err_mms)
                        traj_metrics["root_yaw_errors"].append(root_yaw_err)
                        traj_metrics["state_history"].append(
                            {
                                "qpos": state.mj_data.qpos.copy(),
                                "qvel": state.mj_data.qvel.copy(),
                                "xpos": state.mj_data.xpos.copy(),
                                "xmat": state.mj_data.xmat.copy(),
                            }
                        )

                    track_step = track_step + 1

                    # Print metrics when trajectory completes (after processing last frame)
                    if (
                        track_step >= traj_len
                        and traj_metrics is not None
                        and len(traj_metrics["state_history"]) > 0
                    ):
                        _print_offline_metrics(
                            traj_metrics, traj_filename, ref_traj, mj_sim.mj_model
                        )
                        traj_metrics = None
                        traj_filename = None
                        ref_traj = None  # Avoid index out of bounds on next iteration

            last_mode = mode
            if not args.headless:
                mj_sim.view(state)
            total_steps += 1
            if cmd.kill or (args.max_steps and total_steps >= args.max_steps):
                break
            rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        if buf_mocap is not None:
            try:
                from deploy.retarget import stop_realtime_retarget

                stop_realtime_retarget(buf_mocap)
            except Exception as exc:
                print(f"[Mocap] Retarget shutdown warning: {exc}")
        if not mj_sim.headless and getattr(mj_sim, "viewer", None) is not None:
            mj_sim.viewer.close()
        keyboard.close()
        print("[Sim] Exited.")


def _validate_motion_switcher_state(status: int, result: object) -> str:
    """Return the active high-level mode name or fail closed on bad RPC data."""
    if status != 0:
        raise RuntimeError(f"MotionSwitcher CheckMode failed with status {status}")
    if not isinstance(result, dict):
        raise RuntimeError(f"MotionSwitcher returned invalid result: {result!r}")
    name = result.get("name")
    if not isinstance(name, str):
        raise RuntimeError(f"MotionSwitcher returned invalid mode name: {name!r}")
    return name


def _release_active_motion_mode(
    motion_switcher: object,
    *,
    max_attempts: int = 10,
    retry_s: float = 0.5,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Release high-level ownership before publishing ``LowCmd``.

    Unitree's low-level examples explicitly call ``ReleaseMode`` until
    ``CheckMode`` reports an empty name. The remote L2+R2 indication is not a
    reliable substitute for this RPC state, especially when the ``ai``
    service remains resident or reacquires ownership.
    """
    attempts = int(max_attempts)
    delay = float(retry_s)
    if attempts <= 0:
        raise ValueError("MotionSwitcher max_attempts must be positive")
    if not np.isfinite(delay) or delay < 0.0:
        raise ValueError("MotionSwitcher retry_s must be finite and non-negative")

    status, result = motion_switcher.CheckMode()
    active_mode = _validate_motion_switcher_state(status, result)
    for attempt in range(1, attempts + 1):
        if not active_mode:
            print("[Real] MotionSwitcher high-level mode is released.")
            return

        print(
            f"[Real] Releasing high-level motion mode {active_mode!r} "
            f"({attempt}/{attempts})..."
        )
        release_status, _ = motion_switcher.ReleaseMode()
        if release_status != 0:
            raise RuntimeError(
                "MotionSwitcher ReleaseMode failed with status "
                f"{release_status} while releasing {active_mode!r}"
            )
        if delay:
            sleep_fn(delay)
        status, result = motion_switcher.CheckMode()
        active_mode = _validate_motion_switcher_state(status, result)

    raise RuntimeError(
        f"High-level motion mode {active_mode!r} remained active after "
        f"{attempts} ReleaseMode attempts; LowCmd publishing was not started"
    )


def _validate_real_runtime_access(
    *,
    debug: bool,
    allow_real_publish: bool,
    enable_hand: bool,
) -> None:
    """Enforce real DDS publication gates at every callable entry point."""
    if debug and enable_hand:
        raise RuntimeError(
            "--enable-hand is not allowed in real --debug mode because "
            "HandCmd has no debug publication gate"
        )
    if not debug and not allow_real_publish:
        raise RuntimeError(
            "Real LowCmd publishing is locked. Use the reviewed "
            "scripts/run_pico_real.sh --publish-lowcmd flow (suspended "
            "robot first), or pass --allow-real-publish explicitly."
        )


def _attempt_damping_sequence(
    low_ctrl,
    *,
    duration_s: float,
    ctrl_dt: float,
    lock_timeout_s: float,
    sleep_fn=time.sleep,
) -> Exception | None:
    """Attempt a bounded damping sequence and return its first error."""
    damping_steps = max(1, int(round(duration_s / ctrl_dt)))
    for _ in range(damping_steps):
        try:
            low_ctrl.set_motor_damping(lock_timeout_s=lock_timeout_s)
        except Exception as exc:
            return exc
        sleep_fn(ctrl_dt)
    return None


def _warmup_real_policy_instances(
    walk_policy: WalkPolicy,
    infer_fn: G1TrackInferFn,
    live_converter: LiveRefConverter,
    *,
    iterations: int = 20,
) -> None:
    """Warm the exact Walk and Online Pose instances used by the real loop."""
    if iterations <= 0:
        raise ValueError("warmup iterations must be positive")

    root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    root_gyro = np.zeros(3, dtype=np.float32)
    joint_qpos = np.asarray(DEFAULT_QPOS_JOINT, dtype=np.float32).copy()
    joint_qvel = np.zeros_like(joint_qpos)
    cmd_vel = np.zeros(3, dtype=np.float32)
    reference_qpos = np.asarray(consts.DEFAULT_QPOS, dtype=np.float32).copy()
    if joint_qpos.shape != (29,) or reference_qpos.shape != (36,):
        raise RuntimeError(
            "unexpected default pose shapes during real policy warmup: "
            f"joint={joint_qpos.shape}, reference={reference_qpos.shape}"
        )

    def _check_target(label: str, target: np.ndarray) -> None:
        value = np.asarray(target, dtype=np.float32).reshape(-1)
        if value.shape != (29,) or not np.all(np.isfinite(value)):
            raise RuntimeError(
                f"{label} warmup returned invalid motor target "
                f"shape={value.shape}, finite={np.all(np.isfinite(value))}"
            )

    walk_timings_ms = []
    online_timings_ms = []
    try:
        for _ in range(iterations):
            started_ns = time.perf_counter_ns()
            target = walk_policy.infer(
                root_quat, root_gyro, joint_qpos, joint_qvel, cmd_vel
            )
            walk_timings_ms.append((time.perf_counter_ns() - started_ns) / 1e6)
            _check_target("Walk", target)

        # Prime both MuJoCo reference conversion and the real-fast ONNX path.
        live_converter.reset()
        ref_curr = live_converter.convert(reference_qpos)
        ref_next = live_converter.convert(reference_qpos)
        reference = {"ref_curr": ref_curr, "ref_next": ref_next}
        for _ in range(iterations):
            started_ns = time.perf_counter_ns()
            target = infer_fn.infer_onnx_real_fast(
                root_quat, root_gyro, joint_qpos, joint_qvel, reference
            )
            online_timings_ms.append((time.perf_counter_ns() - started_ns) / 1e6)
            _check_target("Online Pose", target)
    finally:
        # Warmup must not leak gait phase or action/reference history to hardware.
        walk_policy.reset()
        _reset_tracking_history(infer_fn, live_converter)

    walk_p95 = float(np.percentile(walk_timings_ms, 95))
    online_p95 = float(np.percentile(online_timings_ms, 95))
    print(
        f"[Warmup] Walk ({walk_policy.provider}) {iterations} iterations: "
        f"p95={walk_p95:.3f} ms, max={max(walk_timings_ms):.3f} ms"
    )
    print(
        f"[Warmup] Online Pose ({infer_fn.nn_policy.__class__.__name__}) "
        f"{iterations} iterations: p95={online_p95:.3f} ms, "
        f"max={max(online_timings_ms):.3f} ms"
    )


# ---------------------------------------------------------------------------
# Real-robot loop
# ---------------------------------------------------------------------------


def run_real(args: DeployArgs):
    _validate_real_runtime_access(
        debug=args.debug,
        allow_real_publish=args.allow_real_publish,
        enable_hand=args.enable_hand,
    )
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    from deploy.control_thread import StoppablePeriodicThread
    from deploy.hand_control import Dex3Controller, update_hand_from_mocap
    from deploy.real_robot import KeyMap, LowLevelControlG1
    from deploy.retarget import (
        read_hand_buffer,
        request_retarget_calibration,
        start_realtime_retarget,
        stop_realtime_retarget,
        wait_realtime_retarget_ready,
    )
    from deploy.telemetry_zmq import (
        POLICY_HOLD,
        POLICY_TRACK,
        POLICY_WALK,
        RobotStateActionPublisher,
        RobotStateActionSnapshot,
    )

    class _OperatorStop(Exception):
        pass

    if args.freq <= 0:
        raise ValueError("--freq must be positive")
    positive_finite_args = [
        (
            "--real-low-state-startup-timeout-s",
            args.real_low_state_startup_timeout_s,
        ),
        ("--real-control-join-timeout-s", args.real_control_join_timeout_s),
        ("--real-damping-duration-s", args.real_damping_duration_s),
    ]
    for flag, value in positive_finite_args:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{flag} must be finite and positive")

    provider = args.real_policy_provider.strip().lower()
    if provider not in {"cpu", "tensorrt"}:
        raise ValueError("--real-policy-provider must be cpu or tensorrt")

    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    shutdown_requested = threading.Event()
    operator_damping_requested = threading.Event()
    worker = None
    worker_started = False
    worker_stopped = True
    low_ctrl = None
    keyboard = None
    buf_mocap = None
    buf_hand = None
    worker_error: BaseException | None = None
    automatic_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    damping_error: BaseException | None = None
    state_publisher = None

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _shutdown_signal(*_):
        operator_damping_requested.set()
        shutdown_requested.set()

    signal.signal(signal.SIGINT, _shutdown_signal)
    signal.signal(signal.SIGTERM, _shutdown_signal)

    try:
        policy_args = PolicyArgs(
            load_path=args.onnx_track,
            policy_type=args.policy_type,
            device="cpu" if provider == "cpu" else "cuda:0",
        )
        use_trt = provider == "tensorrt"
        track_policy = get_policy_onnx(
            policy_args,
            use_trt=use_trt,
            strict_trt=use_trt,
        )
        walk_policy = WalkPolicy(
            args.onnx_walk,
            provider="cpu" if provider == "cpu" else "cuda",
        )

        convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
        print("Loading offline reference motions...")
        print("  Mode 0: Walk")
        print("  Mode 1: Pico/online retarget")
        ref_motions = load_offline_motions(args.track_dir, convert_model, freq)

        # Warm the exact runtime instances before DDS ownership or publication.
        xml_path = str(consts.ROOT_PATH / "scene_mjx_track.xml")
        phantom_model = mujoco.MjModel.from_xml_path(xml_path)
        phantom_model.opt.timestep = 0.001
        infer_fn = G1TrackInferFn(
            env_cfg,
            phantom_model,
            track_policy,
            privileged=False,
        )
        live_converter = LiveRefConverter(phantom_model, ctrl_dt)
        _warmup_real_policy_instances(walk_policy, infer_fn, live_converter)

        if shutdown_requested.is_set():
            raise _OperatorStop

        pygame.init()
        pygame.display.quit()
        if args.pose_endpoint:
            from deploy.pose_zmq import ZmqPoseSource

            keyboard = ZmqPoseSource(args.pose_endpoint)
            print(f"[PoseZMQ] Connected to {args.pose_endpoint}")
        else:
            keyboard = DeployKeyboardCMD(num_track_ref=len(ref_motions))

        ChannelFactoryInitialize(0, args.net)
        if not args.debug:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )

            motion_switcher = MotionSwitcherClient()
            motion_switcher.SetTimeout(5.0)
            motion_switcher.Init()
            _release_active_motion_mode(motion_switcher)
        low_ctrl = LowLevelControlG1(
            ctrl_dt=ctrl_dt,
            debug=args.debug,
            low_state_startup_timeout_s=args.real_low_state_startup_timeout_s,
        )
        publish_mode = "DISABLED (debug)" if args.debug else "ENABLED"
        print(f"[Real] DDS LowCmd publishing: {publish_mode}")
        if args.state_action_bind:
            state_publisher = RobotStateActionPublisher(args.state_action_bind)
            print(
                f"[Collector] Publishing robot_state_action on {args.state_action_bind}"
            )

        mocap_buffer = None
        if args.pose_endpoint:
            mocap_buffer = keyboard
            print("[PoseZMQ] Manager provides qpos36, mode, velocity, and reset")
        elif not args.no_mocap:
            mocap_type = _resolve_mocap_type(args.mocap_type)
            if mocap_type.name == "PICO" and args.enable_hand:
                raise ValueError("Pico v1 is body-only; --enable-hand is not supported")
            buffer_ms = _effective_buffer_ms(mocap_type, args.buffer_ms)
            buf_mocap, ts_mocap, buf_hand = start_realtime_retarget(
                server_ip=args.server_ip,
                client_ip=args.client_ip,
                robot="unitree_g1",
                dof_full=7 + 29,
                actual_human_height=args.human_height,
                visualize_retarget=args.visualize_retarget,
                mocap_type=mocap_type,
                buffer_ms=buffer_ms,
                pico_service_mode=args.pico_service_mode,
                pico_startup_timeout_s=args.pico_startup_timeout_s,
                xsens_host=args.xsens_host,
                xsens_port=args.xsens_port,
                xsens_protocol=args.xsens_protocol,
            )
            mocap_buffer = MocapBuffer(buf_mocap, ts_mocap)
            print("[Mocap] Retarget subprocess started; waiting for first frame...")
            if mocap_type.name == "PICO":
                # Include a bounded allowance for GMR/MuJoCo initialization;
                # PicoXrtSource enforces its own source-frame timeout inside
                # the worker and a worker exit is surfaced immediately.
                wait_realtime_retarget_ready(
                    buf_mocap,
                    timeout_s=args.pico_startup_timeout_s + 30.0,
                )
                print("[Mocap] First Pico/GMR reference is ready.")

        hand_ctrl = None
        if args.enable_hand:
            try:
                hand_ctrl = Dex3Controller(net=args.net, re_init=False)
            except Exception as exc:
                print(f"[Hand] Failed to initialize: {exc}")

        last_mode = 0
        track_step = 0
        ref_traj = None
        last_left_hand = None
        last_right_hand = None
        prev_online_ref = None
        state_index = 0

        def _remote_pressed(button: int) -> bool:
            return low_ctrl.is_remote_button_pressed(button)

        def _send_target(target, kps, kds) -> None:
            if shutdown_requested.is_set():
                return
            target_array = np.asarray(target, dtype=np.float32).reshape(-1)
            low_ctrl.step(target_array, kps, kds)

        def _reset_online_reference(root_quat: np.ndarray) -> None:
            nonlocal prev_online_ref
            _reset_tracking_history(infer_fn, live_converter)
            prev_online_ref = None
            live_converter.set_robot_initial_pose(
                root_quat,
                np.zeros(2, dtype=np.float32),
            )
            if buf_mocap is not None:
                request_retarget_calibration(buf_mocap)
            print("[Track] Online retarget reset")

        def locomotion_step():
            nonlocal last_mode, track_step, ref_traj, state_index
            nonlocal last_left_hand, last_right_hand, prev_online_ref

            if shutdown_requested.is_set():
                return
            if _remote_pressed(KeyMap.select):
                print("[Real] Select pressed, stopping.")
                operator_damping_requested.set()
                shutdown_requested.set()
                return

            root_quat, root_gyro, jnt_qpos, jnt_qvel = low_ctrl.get_sensor_state()
            sensor_monotonic_ns = time.monotonic_ns()
            cmd = keyboard.step_command()
            if cmd.kill:
                print("[Real] Keyboard kill requested.")
                operator_damping_requested.set()
                shutdown_requested.set()
                return
            mode = cmd.mode
            cmd_vel = np.array(
                [cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw],
                dtype=np.float32,
            )
            motor_targets = DEFAULT_QPOS_JOINT.copy()
            command_kp = np.asarray(consts.KPs, dtype=np.float32)
            command_kd = np.asarray(consts.KDs, dtype=np.float32)
            policy_kind = POLICY_HOLD
            policy_obs = np.empty(0, dtype=np.float32)
            policy_action = np.empty(0, dtype=np.float32)
            reference_qpos = np.zeros(36, dtype=np.float32)
            reference_qpos[3] = 1.0
            reference_qpos[7:] = DEFAULT_QPOS_JOINT
            reference_valid = False

            entering_track = mode >= 1 and mode != last_mode
            leaving_track = last_mode >= 1 and mode == 0

            if entering_track:
                _reset_tracking_history(infer_fn, live_converter)
                if mode == 1:
                    _reset_online_reference(root_quat)
                else:
                    traj_idx = mode - 2
                    if traj_idx < len(ref_motions):
                        replay_yaw = quat_yaw_wxyz(root_quat)
                        ref_traj = align_reference_trajectory_first_frame(
                            ref_motions[traj_idx]["data"],
                            target_xy=(0.0, 0.0),
                            target_yaw=replay_yaw,
                        )
                        track_step = 0
                        print(
                            "[Track] Start offline: "
                            f"{ref_motions[traj_idx]['filename']} "
                            f"(first frame -> robot XY=[0,0], "
                            f"IMU yaw={np.degrees(replay_yaw):.1f}deg)"
                        )
                    else:
                        ref_traj = None
                        print(f"[Track] Invalid offline trajectory {traj_idx}")

            if leaving_track:
                live_converter.reset()
                prev_online_ref = None
                print("[Track] Back to walk mode")

            if keyboard.check_reset_request():
                if mode == 1:
                    _reset_online_reference(root_quat)
                elif mode >= 2 and ref_traj is not None:
                    _reset_tracking_history(infer_fn, live_converter)
                    track_step = 0
                    print("[Track] Offline trajectory reset")

            if mode == 0:
                motor_targets = walk_policy.infer(
                    root_quat,
                    root_gyro,
                    jnt_qpos,
                    jnt_qvel,
                    cmd_vel,
                )
                policy_kind = POLICY_WALK
                policy_obs = walk_policy.last_obs
                policy_action = walk_policy.last_action
                command_kp = np.asarray(KPs_walking, dtype=np.float32)
                command_kd = np.asarray(KDs_walking, dtype=np.float32)
                _send_target(
                    motor_targets,
                    KPs_walking,
                    KDs_walking,
                )

            elif mode == 1:
                if mocap_buffer is None:
                    last_mode = mode
                    return
                ref_curr, ref_next, prev_online_ref = _advance_online_reference(
                    mocap_buffer, live_converter, prev_online_ref
                )
                motor_targets = infer_fn.infer_onnx_real_fast(
                    root_quat,
                    root_gyro,
                    jnt_qpos,
                    jnt_qvel,
                    {"ref_curr": ref_curr, "ref_next": ref_next},
                )
                motor_targets = np.asarray(motor_targets, dtype=np.float32).reshape(-1)
                policy_kind = POLICY_TRACK
                policy_obs = np.asarray(infer_fn._obs_buf[0], dtype=np.float32).copy()
                policy_action = (
                    np.asarray(infer_fn.info["nn_action"], dtype=np.float32)
                    .reshape(-1)
                    .copy()
                )
                reference_qpos = (
                    np.asarray(ref_curr["qpos"], dtype=np.float32)
                    .reshape(-1)
                    .copy()
                )
                reference_valid = True
                _send_target(motor_targets, consts.KPs, consts.KDs)
                if (
                    hand_ctrl is not None
                    and buf_hand is not None
                    and not shutdown_requested.is_set()
                ):
                    hand_cmd = read_hand_buffer(buf_hand)
                    last_left_hand, last_right_hand = update_hand_from_mocap(
                        hand_ctrl,
                        hand_cmd,
                        last_left_hand,
                        last_right_hand,
                    )

            else:
                if ref_traj is None:
                    _send_target(
                        DEFAULT_QPOS_JOINT,
                        consts.KPs,
                        consts.KDs,
                    )
                else:
                    traj_len = len(ref_traj["qpos"])
                    ref_curr = jtu.tree_map(
                        lambda x: x[track_step][None],
                        ref_traj,
                    )
                    next_step = min(track_step + 1, traj_len - 1)
                    ref_next = jtu.tree_map(
                        lambda x: x[next_step][None],
                        ref_traj,
                    )
                    motor_targets = infer_fn.infer_onnx_real_fast(
                        root_quat,
                        root_gyro,
                        jnt_qpos,
                        jnt_qvel,
                        {"ref_curr": ref_curr, "ref_next": ref_next},
                    )
                    motor_targets = np.asarray(motor_targets, dtype=np.float32).reshape(
                        -1
                    )
                    policy_kind = POLICY_TRACK
                    policy_obs = np.asarray(
                        infer_fn._obs_buf[0], dtype=np.float32
                    ).copy()
                    policy_action = (
                        np.asarray(infer_fn.info["nn_action"], dtype=np.float32)
                        .reshape(-1)
                        .copy()
                    )
                    reference_qpos = (
                        np.asarray(ref_curr["qpos"], dtype=np.float32)
                        .reshape(-1)
                        .copy()
                    )
                    reference_valid = True
                    _send_target(
                        motor_targets,
                        consts.KPs,
                        consts.KDs,
                    )
                    track_step = next_step

            if state_publisher is not None:
                state_publisher.publish(
                    RobotStateActionSnapshot(
                        state_index=state_index,
                        sensor_monotonic_ns=sensor_monotonic_ns,
                        mode=mode,
                        cmd_vel=cmd_vel,
                        base_quat=np.asarray(root_quat).copy(),
                        base_ang_vel=np.asarray(root_gyro).copy(),
                        base_accel=low_ctrl.root_accel.copy(),
                        body_q=np.asarray(jnt_qpos).copy(),
                        body_dq=np.asarray(jnt_qvel).copy(),
                        body_tau=low_ctrl.joint_torque.copy(),
                        reference_qpos=reference_qpos,
                        reference_valid=reference_valid,
                        policy_kind=policy_kind,
                        policy_obs=policy_obs,
                        policy_action=policy_action,
                        body_action=np.asarray(motor_targets).reshape(-1),
                        kp=command_kp,
                        kd=command_kd,
                        lowcmd_published=not args.debug,
                    )
                )
                state_index += 1

            last_mode = mode

        def _check_startup_state() -> np.ndarray:
            if shutdown_requested.is_set():
                raise _OperatorStop
            if _remote_pressed(KeyMap.select):
                operator_damping_requested.set()
                raise _OperatorStop
            startup_cmd = keyboard.step_command()
            if startup_cmd.kill:
                print("[Real] Keyboard kill requested during startup.")
                operator_damping_requested.set()
                raise _OperatorStop
            _, _, joint_qpos, _ = low_ctrl.get_sensor_state()
            return joint_qpos.copy()

        print(
            "<Mode: Damping> Robot must be suspended. Waiting for <start> on remote..."
        )
        while not _remote_pressed(KeyMap.start):
            _check_startup_state()
            low_ctrl.set_motor_damping()
            if shutdown_requested.wait(ctrl_dt):
                raise _OperatorStop

        initial_qpos = _check_startup_state()
        move_steps = max(1, int(round(2.0 / ctrl_dt)))
        print("<Mode: Stand-up> Moving slowly to default pose...")
        for step_index in range(1, move_steps + 1):
            _check_startup_state()
            alpha = step_index / move_steps
            target = initial_qpos * (1.0 - alpha) + DEFAULT_QPOS_JOINT * alpha
            _send_target(target, consts.KPs * alpha, consts.KDs)
            if shutdown_requested.wait(ctrl_dt):
                raise _OperatorStop

        print("<Mode: Default> Waiting for <A> on remote...")
        while not _remote_pressed(KeyMap.A):
            _check_startup_state()
            _send_target(DEFAULT_QPOS_JOINT, consts.KPs, consts.KDs)
            if shutdown_requested.wait(ctrl_dt):
                raise _OperatorStop

        print(
            "<Mode: Locomotion> Control loop starting. "
            "Keyboard 0=walk, 1=Pico, remote Select=stop."
        )
        worker = StoppablePeriodicThread(
            interval=ctrl_dt,
            target=locomotion_step,
            name="hgpt-real-control",
            daemon=True,
        )
        worker.start()
        worker_started = True

        while not shutdown_requested.wait(0.02):
            if _remote_pressed(KeyMap.select):
                print("[Real] Select pressed, stopping.")
                operator_damping_requested.set()
                shutdown_requested.set()
            elif not worker.is_alive():
                worker_error = worker.exception or RuntimeError(
                    "control thread stopped unexpectedly"
                )
                shutdown_requested.set()
                break

    except _OperatorStop:
        print("[Real] Operator stop requested.")
    except Exception as exc:
        automatic_error = exc
        print(f"[Real] Control error: {exc}")
    finally:
        shutdown_requested.set()
        if worker is not None:
            worker.stop()
            if worker_started:
                worker_stopped = worker.join(
                    timeout=args.real_control_join_timeout_s,
                    raise_on_error=False,
                )
                worker_error = worker.exception
                if not worker_stopped:
                    print(
                        "[Real] Control thread did not stop before timeout."
                    )

        if low_ctrl is not None and operator_damping_requested.is_set():
            damping_error = _attempt_damping_sequence(
                low_ctrl,
                duration_s=args.real_damping_duration_s,
                ctrl_dt=ctrl_dt,
                lock_timeout_s=args.real_control_join_timeout_s,
            )
            if damping_error is not None:
                print(f"[Safety] Damping publish failed: {damping_error}")

        if worker is not None and worker_started and not worker_stopped:
            worker_stopped = worker.join(
                timeout=args.real_control_join_timeout_s,
                raise_on_error=False,
            )
            worker_error = worker.exception
            if not worker_stopped:
                cleanup_error = RuntimeError(
                    "control thread remained alive after bounded shutdown"
                )
                print(f"[Safety] {cleanup_error}")

        if (
            operator_damping_requested.is_set()
            and damping_error is not None
            and low_ctrl is not None
            and worker_stopped
        ):
            print("[Safety] Retrying damping after control thread shutdown.")
            damping_error = _attempt_damping_sequence(
                low_ctrl,
                duration_s=args.real_damping_duration_s,
                ctrl_dt=ctrl_dt,
                lock_timeout_s=args.real_control_join_timeout_s,
            )
            if damping_error is not None:
                print(f"[Safety] Damping retry failed: {damping_error}")

        if buf_mocap is not None:
            try:
                stop_realtime_retarget(buf_mocap)
            except Exception as exc:
                print(f"[Mocap] Retarget shutdown warning: {exc}")
        if keyboard is not None:
            keyboard.close()
        if state_publisher is not None:
            state_publisher.close()
        signal.signal(signal.SIGINT, previous_sigint_handler)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if args.debug:
            print("[Real] Debug run exited; no LowCmd was published.")
        elif low_ctrl is None:
            print("[Real] Exited before LowLevelControlG1 initialization.")
        elif operator_damping_requested.is_set() and damping_error is not None:
            print("[Safety] Exit completed without confirmed operator damping.")
        elif operator_damping_requested.is_set():
            print("[Real] Exited after operator-requested damping.")
        else:
            print("[Real] Exited without automatic damping.")

    if damping_error is not None:
        raise RuntimeError("final damping publication failed") from damping_error
    if automatic_error is not None:
        raise automatic_error
    if worker_error is not None:
        raise worker_error
    if cleanup_error is not None:
        raise cleanup_error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class DeployArgs:
    onnx_walk: str = "storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx"
    track_dir: str = "storage/test"
    onnx_track: str = "storage/ckpts/pns_wo_priv216.onnx"
    policy_type: str = "mlp"
    # Independent unified pose manager. Empty keeps the direct mocap path.
    pose_endpoint: str = ""

    convert_xml_path: str = str(consts.TRACK_XML)
    real: bool = False
    debug: bool = False
    freq: int = 50
    headless: bool = False
    max_steps: int = 0

    # Mocap
    no_mocap: bool = False
    mocap_type: str = "pnlink"  # pico | pnlink | optitrack | xsens
    server_ip: str = "169.254.117.205"
    client_ip: str = "169.254.117.206"
    human_height: float = 1.7
    visualize_retarget: bool = True
    buffer_ms: float | None = None

    # Pico/XRoboToolkit
    pico_service_mode: str = "auto"  # auto | external
    pico_startup_timeout_s: float = 15.0

    # Xsens MVN streamer (only used when --mocap_type xsens).
    xsens_host: str = "0.0.0.0"  # local bind address for the receiver
    xsens_port: int = 9763  # default MVN Network Streamer port
    xsens_protocol: str = "tcp"  # "tcp" or "udp" - match MVN Studio setting

    # Real robot
    net: str = "enx6c1ff76e8ef5"
    enable_hand: bool = False
    allow_real_publish: bool = False
    state_action_bind: str = "tcp://*:5558"
    real_policy_provider: str = "cpu"  # cpu | tensorrt
    real_low_state_startup_timeout_s: float = 5.0
    real_control_join_timeout_s: float = 1.0
    real_damping_duration_s: float = 0.5


def main(args: DeployArgs):
    if args.real:
        _validate_real_runtime_access(
            debug=args.debug,
            allow_real_publish=args.allow_real_publish,
            enable_hand=args.enable_hand,
        )
        run_real(args)
    else:
        run_sim(args)


if __name__ == "__main__":
    main(tyro.cli(DeployArgs))
