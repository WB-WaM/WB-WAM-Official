import tyro
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Literal
from dataclasses import dataclass
from collections.abc import Iterable

import mujoco
import mujoco.viewer
from mujoco import Renderer
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as R, Slerp

from utils.transforms_np import (
    base2navi,
    batch_base2navi,
    mat2quat,
    quat2mat,
    wxyz2xyzw,
    xyzw2wxyz
)
from tracking import constants as consts
from tracking.constants import KPT_NAMES
from utils.video_utils import images_to_video
from utils.transforms_np import qpos_flip_around_x
from utils.sim_mj import get_dof_ids, get_qpos_ids
from tracking.constants import (
    DEFAULT_QPOS,
    ACTION_JOINT_NAMES,
    ACTION_JOINT_NAMES_66177,
    ACTION_JOINT_NAMES_66155,
    ACTION_JOINT_NAMES_66144
)

FOOT_SITE_NAMES = ["left_foot", "right_foot"]
DOF_MIRROR_MAPPING = np.int32([
    [6, 1], [7, -1], [8, -1], [9, 1], [10, 1], [11, -1],
    [0, 1], [1, -1], [2, -1], [3, 1], [4, 1], [5, -1],
    [12, -1], [13, -1], [14, 1], [22, 1], [23, -1], [24, -1],
    [25, 1], [26, -1], [27, 1], [28, -1], [15, 1], [16, -1],
    [17, -1], [18, 1], [19, -1], [20, 1], [21, -1],
])


@dataclass
class Args:
    mocap_npz: str
    save_path: str | None = None
    start: int | None = None
    end: int | None = None
    debug: bool = False
    fix_freq_src: int | None = None
    freq_tgt: int = 50
    freq_cut: float | None = None  # low-pass cutoff (Hz); None disables filter
    xml_path: str = str(consts.DEBUG_TRACK_XML)
    interp_sec: float = 0.0  # smooth in/out duration (s)
    end_default_sec: float = 0.0  # maintain default qpos
    foot_contact_est: bool = True
    aug_flip: bool = False
    aug_freq_ratio: int = 1
    aug_freq_range: float = 0.1
    height_clip_mode: str | None = None
    video_path: str | None = None


class ArrowManager:
    """
    Manages multiple arrow visualizations in a MuJoCo viewer's user scene
    using a unique string ID to map to a geometry index.
    """

    def __init__(self, viewer, start_index=None):
        self.viewer = viewer
        self.next_geom_idx = (
            start_index if start_index is not None else viewer.user_scn.ngeom
        )
        self.id_to_idx = {}

    def update_arrow(
        self, arrow_id: str, start: np.ndarray, end: np.ndarray, color=None, radius=0.02
    ):
        if arrow_id not in self.id_to_idx:
            geom_idx = self.next_geom_idx

            if geom_idx >= self.viewer.user_scn.maxgeom:
                print(f"[WARN] Maximum number of geometries reached; cannot create arrow '{arrow_id}'.")
                return

            self.id_to_idx[arrow_id] = geom_idx
            self.viewer.user_scn.ngeom = geom_idx + 1
            self.next_geom_idx += 1

            default_color = [1.0, 0.0, 0.0, 1.0]
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[geom_idx],
                mujoco.mjtGeom.mjGEOM_CAPSULE,  # Base type
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                np.float32(color or default_color),
            )
            print(f"[INFO] Created new arrow '{arrow_id}' at geometry index {geom_idx}.")

        else:
            geom_idx = self.id_to_idx[arrow_id]
            if color is not None:
                self.viewer.user_scn.geoms[geom_idx].rgba[:] = np.float32(color)

        mujoco.mjv_connector(
            self.viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_ARROW,
            radius,
            np.float32(start),
            np.float32(end),
        )


def lp_filter(
    qpos: np.ndarray, freq_src: float, freq_cut: float, order: int = 4
) -> np.ndarray:
    """
    Low-pass filter a pose trajectory where columns are:
      [0:3]=position, [3:7]=quaternion (w,x,y,z), [7:]=other channels.
    Quaternions are linearized via rotation vectors (log-map), filtered, then re-exponentiated.

    Args:
      qpos: (T, D) array; expects quaternions in (w,x,y,z) at cols [3:7].
      freq_src: sample rate [Hz].
      freq_cut: low-pass cutoff [Hz].
      order: Butterworth order (2–4 is typical).

    Returns:
      qpos_filtered: same shape as qpos, low-passed over time.
    """
    length = len(qpos)
    if length < 2:
        return qpos.copy()

    rot = R.from_quat(np.roll(qpos[:, 3:7], -1, axis=-1)).as_matrix()

    delta_rv = np.zeros((length, 3), dtype=float)
    for i in range(1, length):
        curr2prev_rot = rot[i - 1].T @ rot[i]
        delta_rv[i] = R.from_matrix(curr2prev_rot).as_rotvec()

    q_dim = qpos.shape[1]
    qpos_lin = np.zeros((length, q_dim - 1), dtype=float)
    qpos_lin[:, 0:3] = qpos[:, 0:3]
    qpos_lin[:, 3:6] = delta_rv
    if q_dim > 7:
        qpos_lin[:, 6:] = qpos[:, 7:]

    sos = butter(order, freq_cut, fs=freq_src, btype="low", output="sos")
    qpos_lin_f = sosfiltfilt(sos, qpos_lin, axis=0)

    quat_f = np.empty((length, 4), dtype=float)
    quat_f[0] = qpos[0, 3:7]  # (w,x,y,z)

    R_prev = R.from_quat(np.roll(quat_f[0], -1)).as_matrix()
    for i in range(1, length):
        dR = R.from_rotvec(qpos_lin_f[i, 3:6]).as_matrix()
        R_curr = R_prev @ dR
        quat_f[i] = np.roll(R.from_matrix(R_curr).as_quat(), 1)
        R_prev = R_curr

    qpos_f = np.zeros_like(qpos, dtype=float)
    qpos_f[:, 0:3] = qpos_lin_f[:, 0:3]
    if q_dim > 7:
        qpos_f[:, 7:] = qpos_lin_f[:, 6:]
    qpos_f[:, 3:7] = quat_f
    return qpos_f


def contact_estimate(kpt_npose, gv_vel, lin_thresh=0.03, ang_thresh=0.1, freq=50):
    left_ankle_id = KPT_NAMES.index("left_ankle_roll_link")
    right_ankle_id = KPT_NAMES.index("right_ankle_roll_link")

    left_npose = kpt_npose[:, left_ankle_id]
    right_npose = kpt_npose[:, right_ankle_id]

    l_foot_vel_x = np.hstack([[0.0], np.diff(left_npose[:, 0, 3])]) * freq
    l_foot_moving = np.sign(l_foot_vel_x) * (np.abs(l_foot_vel_x) > lin_thresh)
    l_contact_x = np.sign(gv_vel[:, 0]) * l_foot_moving <= 0.0

    r_foot_vel_x = np.hstack([[0.0], np.diff(right_npose[:, 0, 3])]) * freq
    r_foot_moving = np.sign(r_foot_vel_x) * (np.abs(r_foot_vel_x) > lin_thresh)
    r_contact_x = np.sign(gv_vel[:, 0]) * r_foot_moving <= 0.0

    def yawvel(foot2gv2wrd_pose):
        vel_yaw_list = [0.0]
        for t in range(1, len(kpt_npose)):
            foot2gv_curr = foot2gv2wrd_pose[t]
            foot2gv_prev = foot2gv2wrd_pose[t - 1]
            curr2last = np.linalg.inv(foot2gv_prev) @ foot2gv_curr
            vel_yaw = R.from_matrix(curr2last[:3, :3]).as_euler("xyz")[2] * freq
            vel_yaw_list.append(vel_yaw)
        return np.float32(vel_yaw_list)

    l_foot_vel_yaw = yawvel(left_npose)
    l_foot_moving = np.sign(l_foot_vel_yaw) * (np.abs(l_foot_vel_yaw) > ang_thresh)
    l_contact_yaw = np.sign(gv_vel[:, 2]) * l_foot_moving <= 0.0

    r_foot_vel_yaw = yawvel(right_npose)
    r_foot_moving = np.sign(r_foot_vel_yaw) * (np.abs(r_foot_vel_yaw) > ang_thresh)
    r_contact_yaw = np.sign(gv_vel[:, 2]) * r_foot_moving <= 0.0

    contact = np.stack(
        [l_contact_x & l_contact_yaw, r_contact_x & r_contact_yaw], axis=1
    )
    return contact


def resample_state(
    qpos: np.ndarray,
    # qvel: np.ndarray,
    freq_src: float,
    freq_tgt: float,
) -> np.ndarray:
    """Resample (qpos, qvel) from freq_src -> freq_tgt with SLERP on quats (wxyz)."""
    n_src = qpos.shape[0]
    t_end = (n_src - 1) / float(freq_src)
    t_src = np.linspace(0.0, t_end, n_src, endpoint=True)
    n_tgt = int(np.floor(t_end * float(freq_tgt))) + 1
    t_tgt = np.linspace(0.0, t_end, n_tgt, endpoint=True)

    # xyz & joints: linear
    xyz = np.vstack([np.interp(t_tgt, t_src, qpos[:, i]) for i in range(3)]).T
    joints = np.vstack(
        [np.interp(t_tgt, t_src, qpos[:, 7 + i]) for i in range(qpos.shape[1] - 7)]
    ).T

    # quat (wxyz) -> SciPy (xyzw) -> SLERP -> back (wxyz)
    q_xyzw = wxyz2xyzw(qpos[:, 3:7])
    rot_src = R.from_quat(q_xyzw)
    slerp = Slerp(t_src, rot_src)
    q_xyzw_tgt = slerp(t_tgt).as_quat()
    q_wxyz_tgt = xyzw2wxyz(q_xyzw_tgt)

    # # velocities: linear
    # linv = np.vstack([np.interp(t_tgt, t_src, qvel[:, i]) for i in range(3)]).T
    # angv = np.vstack([np.interp(t_tgt, t_src, qvel[:, 3 + i]) for i in range(3)]).T
    # jvel = np.vstack(
    #     [np.interp(t_tgt, t_src, qvel[:, 6 + i]) for i in range(qvel.shape[1] - 6)]
    # ).T
    return np.hstack([xyz, q_wxyz_tgt, joints])  # , np.hstack([linv, angv, jvel])


def recompute_qvel(
    qpos_wxyz: np.ndarray,
    frequency: float | Iterable[float],
    frame: Literal["world", "body"] = "world",
) -> np.ndarray:
    """
    Recompute generalized velocities from positions.

    Args:
        qpos_wxyz: (num_steps, 7+J) array.
            Free joint [x y z qw qx qy qz ...] followed by J joint positions.
        frequency: sampling rate(s) in Hz.
            - scalar: uniform rate
            - array (num_steps,): per-sample rate (first T-1 entries used)
            - array (T-1,): per-interval rate
        frame: 'body' (default, angular velocity in body frame)
               or 'world' (angular velocity expressed in world frame)

    Returns:
        qvel: (num_steps, 6+J) with qvel[0] estimated via forward difference
    """
    qpos_wxyz = np.asarray(qpos_wxyz)
    if qpos_wxyz.ndim != 2 or qpos_wxyz.shape[1] < 7:
        raise ValueError("qpos must be (num_steps, 7+J) with free joint at front (wxyz).")
    num_steps, N = qpos_wxyz.shape

    qvel = np.zeros((num_steps, N - 1), dtype=qpos_wxyz.dtype)
    if num_steps < 2:
        return qvel

    # ---- frequency handling
    freq = np.asarray(frequency, dtype=float)
    if freq.ndim == 0:
        rate = np.full(num_steps - 1, float(freq), dtype=float)
    elif freq.shape == (num_steps - 1,):
        rate = freq
    elif freq.shape == (num_steps,):
        rate = freq[:-1]
    else:
        raise ValueError(f"`frequency` must be scalar, shape (T-1,), or shape (num_steps,). Got shape {freq.shape}.")
    if not np.all(np.isfinite(rate)) or np.any(rate <= 0.0):
        raise ValueError("`frequency` must be positive and finite.")

    # ---- linear velocity (world frame)
    qvel[1:, 0:3] = np.diff(qpos_wxyz[:, 0:3], axis=0) * rate[:, None]

    # ---- angular velocity
    q_xyzw = wxyz2xyzw(qpos_wxyz[:, 3:7])  # convert to [x y z w]
    # normalize
    q_xyzw /= np.linalg.norm(q_xyzw, axis=1, keepdims=True)

    r = R.from_quat(q_xyzw)
    rel = r[:-1].inv() * r[1:]
    omega_body = rel.as_rotvec() * rate[:, None]

    if frame == "body":
        qvel[1:, 3:6] = omega_body
    elif frame == "world":
        qvel[1:, 3:6] = r[:-1].apply(omega_body)
    else:
        raise ValueError("frame must be 'body' or 'world'.")

    # ---- joint velocities (continuous angles, no wrapping needed)
    if N > 7:
        qvel[1:, 6:] = np.diff(qpos_wxyz[:, 7:], axis=0) * rate[:, None]

    # Use forward difference for the first frame
    qvel[0] = qvel[1]

    return qvel


def smooth_start_end(
    qpos: np.ndarray,
    qvel: np.ndarray,
    qpos_jnt_ids: np.ndarray,
    qvel_jnt_ids: np.ndarray,
    interp_sec: float,
    end_default_sec: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend from DEFAULT_QPOS -> first frame and last frame -> DEFAULT_QPOS with SLERP on quats (wxyz)."""
    freq = int(1 / dt)
    steps = max(1, int(round(interp_sec / dt)))
    t = np.linspace(0.0, 1.0, steps)

    # build an initial pose in the *robot's indexing*
    init_qpos = DEFAULT_QPOS[np.int32(np.hstack([range(7), qpos_jnt_ids]))]

    def interp_block(a: np.ndarray, b: np.ndarray, pos_main: np.ndarray) -> np.ndarray:
        out = np.zeros((steps, qpos.shape[1]), dtype=qpos.dtype)
        out[:, :2] = pos_main[:2]
        out[:, 2] = np.linspace(a[2], b[2], steps)
        qa_xyzw = wxyz2xyzw(a[3:7][None, :])
        qb_xyzw = wxyz2xyzw(b[3:7][None, :])
        slerp = Slerp([0.0, 1.0], R.from_quat(np.vstack([qa_xyzw, qb_xyzw])))
        out[:, 3:7] = xyzw2wxyz(slerp(t).as_quat())
        out[:, 7:] = np.linspace(a[7:], b[7:], steps)
        return out

    start_init_qpos = init_qpos.copy()
    start_init_qpos[3:7] = R.from_matrix(base2navi(quat2mat(qpos[0, 3:7]))).as_quat()[
        [3, 0, 1, 2]
    ]
    end_init_qpos = init_qpos.copy()
    end_init_qpos[3:7] = R.from_matrix(base2navi(quat2mat(qpos[-1, 3:7]))).as_quat()[
        [3, 0, 1, 2]
    ]

    start_blk = interp_block(start_init_qpos, qpos[0], qpos[0])
    end_blk = interp_block(qpos[-1], end_init_qpos, qpos[-1])
    end_default_trj = np.full(
        (int(end_default_sec * freq), len(end_init_qpos)), end_blk[-1]
    )

    qpos_full = np.concatenate([start_blk, qpos, end_blk, end_default_trj], axis=0)
    qvel_full = recompute_qvel(qpos_full, 1.0 / dt)
    qvel_full[-1] = 0.0
    return qpos_full, qvel_full


def floor_foot_pos_clip_fn(
    mj_model: mujoco.MjModel,
    qpos_src: np.ndarray,
    height_clip_mode: str,
    site_ids_foot: list[int],
    body_ids_kpt: list[int] = None,
    debug: bool = False,
):
    mj_data = mujoco.MjData(mj_model)
    length = qpos_src.shape[0]
    qpos_rst = qpos_src.copy()

    if height_clip_mode == "first":
        mj_data.qpos[:] = qpos_src[0]
        mujoco.mj_forward(mj_model, mj_data)
        foot_pos_src = mj_data.site_xpos[site_ids_foot]
        offset_pos = np.zeros(3)
        offset_pos[:2] = -qpos_src[0, :2]
        offset_pos[2] = -np.min(foot_pos_src[:, 2])
    elif height_clip_mode == "last":
        mj_data.qpos[:] = qpos_src[-1]
        mujoco.mj_forward(mj_model, mj_data)
        foot_pos_src = mj_data.site_xpos[site_ids_foot]
        offset_pos = np.zeros(3)
        offset_pos[:2] = -qpos_src[-1, :2]
        offset_pos[2] = -np.min(foot_pos_src[:, 2])
    elif height_clip_mode == "all":
        offset_pos = np.zeros((length, 3))
        offset_pos[:, :2] = -qpos_src[0, :2]
        for i, q in enumerate(qpos_src):
            mj_data.qpos[:] = q
            mujoco.mj_forward(mj_model, mj_data)
            foot_pos_src = mj_data.site_xpos[site_ids_foot]
            offset_pos[i, 2] = -np.min(foot_pos_src[:, 2]).copy()
    elif height_clip_mode == "all_kpt":
        offset_pos = np.zeros((length, 3))
        offset_pos[:, :2] = -qpos_src[0, :2]
        for i, q in enumerate(qpos_src):
            mj_data.qpos[:] = q
            mujoco.mj_forward(mj_model, mj_data)
            foot_pos_src = mj_data.site_xpos[site_ids_foot]
            kpt_pos_src = mj_data.xpos[body_ids_kpt]
            body_pos = np.concat([kpt_pos_src, foot_pos_src])
            offset_pos[i, 2] = -np.min(body_pos[:, 2]).copy()
    else:
        raise NotImplementedError

    qpos_rst[:, :3] += offset_pos
    if debug:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        mj_data.qpos[:] = qpos_rst[0]
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()
    return qpos_rst


def mj_body_pose(mj_model, mj_data, name: str) -> np.ndarray:
    bid = mj_model.body(name).id
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = mj_data.xpos[bid]
    m[:3, :3] = mj_data.xmat[bid].reshape(3, 3)
    return m


def extract_kpt(
    mj_model: mujoco.MjModel,
    qpos_src: np.ndarray,
    qvel_src: np.ndarray,
    key_body_names: list[str],
    qpos_jnt_ids: np.ndarray = None,
    qvel_jnt_ids: np.ndarray = None,
    fps: float = 50,
    video_path: str | None = None,
    video_width: int = 640,
    video_height: int = 480,
) -> dict[str, np.ndarray]:
    """Roll the model with provided low-frequency states and export kpts/vels in a navigation frame."""
    dt = 1 / fps
    num_jnt = 29  # matches your data layout
    num_steps = int(qpos_src.shape[0])
    mj_data = mujoco.MjData(mj_model)

    kpt_body_ids = np.int32([mj_model.body(b).id for b in key_body_names])

    # fill robot qpos/qvel
    qpos_rb = np.zeros((num_steps, mj_model.nq), dtype=np.float32)
    qvel_rb = np.zeros((num_steps, mj_model.nv), dtype=np.float32)
    qpos_rb[:, :7] = qpos_src[:, :7]
    if qpos_jnt_ids is not None:
        qpos_rb[:, qpos_jnt_ids] = qpos_src[:, 7:]
    else:
        qpos_rb[:, 7:] = qpos_src[:, 7:]

    qvel_rb[:, :6] = qvel_src[:, :6]
    if qvel_jnt_ids is not None:
        qvel_rb[:, qvel_jnt_ids] = qvel_src[:, 6:]
    else:
        qvel_rb[:, 6:] = qvel_src[:, 6:]

    # navigation frame: yaw-aligned, origin on ground; rot from base quat
    base2wrd_rot = quat2mat(qpos_src[:, 3:7])
    gvi2wrd_rot = batch_base2navi(base2wrd_rot)
    gvi2wrd_pose = np.tile(np.eye(4, dtype=np.float32), (num_steps, 1, 1))
    gvi2wrd_pose[:, :2, 3] = qpos_src[:, :2]
    gvi2wrd_pose[:, :3, :3] = gvi2wrd_rot

    renderer = None
    buf_images = None
    if video_path is not None:
        renderer = Renderer(mj_model, height=video_height, width=video_width)
        buf_images = []
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 0

    out = {
        "qpos": np.zeros((num_steps, 7 + num_jnt), dtype=np.float32),
        "qvel": np.zeros((num_steps, 6 + num_jnt), dtype=np.float32),
        "kpt2gv_pose": np.zeros((num_steps, len(key_body_names), 4, 4), dtype=np.float32),
        "kpt_cvel_in_gv": np.zeros((num_steps, len(key_body_names), 6), dtype=np.float32),
        "gv_vel": np.zeros((num_steps, 3), dtype=np.float32),
        "gv2wrd_pose": gvi2wrd_pose.copy(),
        "foot_contact": np.zeros((num_steps, 2), dtype=np.float32),
    }
    kpt2wrd_pos = np.zeros((num_steps, len(key_body_names), 3), dtype=np.float32)
    kpt_cvel_in_wrd = np.zeros((num_steps, len(key_body_names), 6), dtype=np.float32)

    for t in range(num_steps):
        mj_data.qpos[:] = qpos_rb[t]
        mj_data.qvel[:] = qvel_rb[t]
        mujoco.mj_forward(mj_model, mj_data)

        kpt_world = np.float32([mj_body_pose(mj_model, mj_data, n) for n in key_body_names])  # (K,4,4)
        kpt_gv = np.linalg.inv(gvi2wrd_pose[t]) @ kpt_world  # (K,4,4)
        kpt2wrd_pos[t] = np.float32(kpt_world[:, :3, 3]).copy()
        kpt_cvel_in_wrd[t] = np.float32(mj_data.cvel[kpt_body_ids]).copy()  # (K,6)

        out["kpt2gv_pose"][t] = kpt_gv
        out["qpos"][t] = mj_data.qpos.copy()
        out["qvel"][t] = mj_data.qvel.copy()

        if renderer is not None:
            cam.azimuth = 90.0
            cam.elevation = -20.0
            cam.distance = 2.0
            renderer.update_scene(mj_data, cam)
            buf_images.append(renderer.render())

    for t in range(1, num_steps):
        nav2wrd_last = gvi2wrd_pose[t - 1]
        nav2wrd_curr = gvi2wrd_pose[t]
        curr2last = np.linalg.inv(nav2wrd_last) @ nav2wrd_curr
        vel_lin = curr2last[:2, 3] / dt
        vel_yaw = R.from_matrix(curr2last[:3, :3]).as_euler("xyz")[2] / dt
        out["gv_vel"][t] = np.hstack([vel_lin, [vel_yaw]])

    kpt_cvel_in_gv = np.zeros((num_steps, len(KPT_NAMES), 6))
    R_wrd2gv = np.swapaxes(gvi2wrd_pose[..., :3, :3], -1, -2)  # R_gv2wrd^T = R_wrd2gv
    kpt_cvel_in_gv[..., :3] = np.einsum("...ij,...kj->...ki", R_wrd2gv, kpt_cvel_in_wrd[..., :3])
    kpt_cvel_in_gv[..., 3:] = np.einsum("...ij,...kj->...ki", R_wrd2gv, kpt_cvel_in_wrd[..., 3:])
    out["kpt_cvel_in_gv"] = kpt_cvel_in_gv

    if renderer is not None and buf_images is not None and len(buf_images) > 0:
        try:
            images_to_video(buf_images, video_path, fps=int(round(fps)), color_format="RGB")
            print(f"Video saved to: {video_path}")
        except Exception as e:
            print(f"[WARN] Failed to save video to {video_path}: {e}")

    return out


def qpos2kpt(
    mj_model,
    qpos_src,
    freq_src,
    freq_tgt,
    interp_sec: float = 0.3,
    end_default_sec: float = 0.0,
    debug: bool = False,
    foot_contact_est: bool = True,
    height_clip_mode: str | None = None,
    video_path: str | None = None,
    video_width: int = 640,
    video_height: int = 480,
):
    # resample to target rate
    qpos_src = resample_state(qpos_src, freq_src, freq_tgt)
    qvel_src = recompute_qvel(qpos_src, freq_tgt)

    # model
    sim_dt = 1 / float(freq_tgt)
    mj_model.opt.timestep = sim_dt

    if qpos_src.shape[1] != 7 + len(ACTION_JOINT_NAMES):
        shape_map = {
            7 + len(ACTION_JOINT_NAMES_66177): {13, 14},
            7 + len(ACTION_JOINT_NAMES_66155): {13, 14, 20, 21, 27, 28},
            7 + len(ACTION_JOINT_NAMES_66144): {13, 14, 19, 20, 21, 26, 27, 28},
        }
        if excluded := shape_map.get(qpos_src.shape[1]):
            qpos_new = np.zeros((len(qpos_src), 7 + len(ACTION_JOINT_NAMES)))
            qvel_new = np.zeros((len(qpos_src), 6 + len(ACTION_JOINT_NAMES)))
            included = np.setdiff1d(np.arange(29), list(excluded))
            qpos_id = np.hstack([np.arange(7), 7 + included])
            qvel_id = np.hstack([np.arange(6), 6 + included])
            qpos_new[:, qpos_id] = qpos_src
            qvel_new[:, qvel_id] = qvel_src
            qpos_src = qpos_new
            qvel_src = qvel_new

    # physical clip
    if height_clip_mode is not None:
        qpos_src = floor_foot_pos_clip_fn(
            mj_model,
            qpos_src,
            height_clip_mode,
            site_ids_foot=[int(mj_model.site(n).id) for n in FOOT_SITE_NAMES],
            body_ids_kpt=[int(mj_model.body(n).id) for n in KPT_NAMES],
        )

    # smooth ramps + ids
    qpos_ids = get_qpos_ids(mj_model, ACTION_JOINT_NAMES)
    qvel_ids = get_dof_ids(mj_model, ACTION_JOINT_NAMES)
    if interp_sec > 0.0:
        qpos_src, qvel_src = smooth_start_end(
            qpos_src,
            qvel_src,
            qpos_ids,
            qvel_ids,
            interp_sec=interp_sec,
            end_default_sec=end_default_sec,
            dt=sim_dt,
        )

    # simulate & extract
    kpt_data = extract_kpt(
        mj_model,
        qpos_src=qpos_src,
        qvel_src=qvel_src,
        qpos_jnt_ids=qpos_ids,
        qvel_jnt_ids=qvel_ids,
        key_body_names=KPT_NAMES,
        fps=freq_tgt,
        video_path=video_path,
        video_width=video_width,
        video_height=video_height
    )

    if foot_contact_est:
        kpt_data["foot_contact"] = contact_estimate(kpt_data["kpt2gv_pose"], kpt_data["gv_vel"])

    if debug:
        import matplotlib
        from loop_rate_limiters import RateLimiter

        mj_data = mujoco.MjData(mj_model)
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        kpt_body_ids = np.int32([mj_model.body(b).id for b in KPT_NAMES])
        arrow_handler = ArrowManager(viewer)
        cmap = matplotlib.colormaps["viridis"]

        gv2wrd_pose = kpt_data["gv2wrd_pose"]
        wrd2gv_pose = np.linalg.inv(gv2wrd_pose)
        kpt_cvel_in_gv = kpt_data["kpt_cvel_in_gv"]
        kpt_cvel_in_wrd = np.zeros_like(kpt_cvel_in_gv)
        kpt_cvel_in_wrd[..., :3] = kpt_cvel_in_gv[..., :3] @ wrd2gv_pose[..., :3, :3]
        kpt_cvel_in_wrd[..., 3:] = kpt_cvel_in_gv[..., 3:] @ wrd2gv_pose[..., :3, :3]

        rate_ctrl = RateLimiter(frequency=freq_tgt)
        for t in range(len(qpos_src)):
            gvi2wrd_pose = kpt_data["gv2wrd_pose"][t]
            kpt_gv = kpt_data["kpt2gv_pose"][t]
            mj_data.qpos[:] = kpt_data["qpos"][t]
            mj_data.qvel[:] = kpt_data["qvel"][t]
            mujoco.mj_forward(mj_model, mj_data)

            kpt_world = np.float32(
                [mj_body_pose(mj_model, mj_data, n) for n in KPT_NAMES]
            )  # (K,4,4)

            for i in range(len(KPT_NAMES)):
                arrow_handler.update_arrow(
                    i,
                    start=kpt_world[i, :3, 3],
                    # end=kpt_world[i, :3, 3] + mj_data.cvel[kpt_body_ids][i, 3:],
                    end=kpt_world[i, :3, 3] + kpt_cvel_in_wrd[t, i, 3:],
                    radius=0.02,
                    color=cmap(i / len(KPT_NAMES)),
                )

            mj_data.mocap_pos[0] = gvi2wrd_pose[:3, 3]
            mj_data.mocap_quat[0] = mat2quat(gvi2wrd_pose[:3, :3])  # (wxyz)

            if foot_contact_est:
                body_id_foot_l = mj_model.body("left_ankle_roll_link").id
                body_id_foot_r = mj_model.body("right_ankle_roll_link").id
                foot_contact = kpt_data["foot_contact"][t]
                mj_data.mocap_pos[1] = (
                    mj_data.xpos[body_id_foot_l] * foot_contact[0]
                    + ~foot_contact[0] * -1
                )
                mj_data.mocap_pos[2] = (
                    mj_data.xpos[body_id_foot_r] * foot_contact[1]
                    + ~foot_contact[1] * -1
                )

            sph0 = 7
            for k in range(kpt_gv.shape[0]):
                mj_data.mocap_pos[sph0 + k] = kpt_gv[k, :3, 3]
                mj_data.mocap_quat[sph0 + k] = mat2quat(kpt_gv[k, :3, :3])
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            rate_ctrl.sleep()

        if viewer is not None:
            viewer.close()

    return kpt_data


def load_data(args):
    try:
        if args.mocap_npz.endswith(".npz") or args.mocap_npz.endswith(".npy"):
            data = dict(np.load(args.mocap_npz, allow_pickle=True))
        elif args.mocap_npz.endswith(".pkl"):
            data = pickle.load(open(args.mocap_npz, "rb"))
        else:
            raise ValueError("Only .npz/.npy and .pkl files are supported.")
    except Exception as e:
        print(e)
        return

    if "qpos" not in data:
        if {"root_pos", "root_rot", "dof_pos"} <= data.keys():
            data["qpos"] = np.concatenate(
                [data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1
            )
        elif {"joint_pos", "body_pos_w", "body_quat_w"} <= data.keys():
            data["qpos"] = np.concatenate(
                [data["body_pos_w"][:,0,:], data["body_quat_w"][:,0,:], data["joint_pos"]], axis=1
            )
    
    return data

# ---------- Pipeline ----------
def run_pipeline(args: Args):
    
    data = load_data(args)
    # Report the total length of the original mocap npz file.
    if data is None:
        print("Failed to load data.")
        return
    if 'qpos' not in data:
        print("No qpos in data.")
        return
    frames_src = len(np.asarray(data["qpos"], dtype=np.float32))
    freq_src = float(data["frequency"]) if "frequency" in data else 50
    freq_src = args.fix_freq_src if args.fix_freq_src is not None else freq_src
    print(f"Source mocap_npz total length: {frames_src} frames, {freq_src} Hz, {frames_src / freq_src:.2f} seconds.")

    q_slice = slice(args.start, args.end)
    # [x y z qw qx qy qz ...]
    qpos_src = np.asarray(data["qpos"][q_slice], dtype=np.float32)
    if len(qpos_src) < 100:
        print("The length of qpos is less than 100 frames.")
        return
    if args.freq_cut is not None:
        qpos_src = lp_filter(qpos_src, freq_src=freq_src, freq_cut=args.freq_cut)

    # Report the length of the sliced trajectory.
    sliced_frames = len(qpos_src)
    print(f"Sliced sequence length: {sliced_frames} frames (start={args.start}, end={args.end}).")
    print(f"Sliced sequence duration: {sliced_frames / freq_src:.2f} seconds.")

    mj_model = mujoco.MjModel.from_xml_path(args.xml_path)
    sim_dt = 1 / float(args.freq_tgt)
    mj_model.opt.timestep = sim_dt
    kpt_data = qpos2kpt(
        mj_model,
        qpos_src,
        freq_src,
        args.freq_tgt,
        interp_sec=args.interp_sec,
        end_default_sec=args.end_default_sec,
        debug=args.debug,
        foot_contact_est=args.foot_contact_est,
        height_clip_mode=args.height_clip_mode,
        video_path=args.video_path,
    )

    # Save the processed trajectory (skipped when `save_path` is empty or None).
    if args.save_path:
        traj_id_start = 0 if args.start is None else args.start
        traj_id_end = len(kpt_data["qpos"]) if args.end is None else args.end
        if args.start is not None and args.end is not None:
            save_path = args.save_path.replace(".npz", f"_{traj_id_start}_{traj_id_end}.npz")
        else:
            save_path = args.save_path
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **kpt_data)
        shapes = {k: tuple(v.shape) for k, v in kpt_data.items()}
        print(f"Saved keypoint data to {out_path} with shapes: {shapes}")
    else:
        print("[INFO] `save_path` is empty; skipping serialization of the keypoint npz file.")

    if args.aug_flip:
        qpos_full_flipped = qpos_flip_around_x(kpt_data["qpos"], DOF_MIRROR_MAPPING)
        qpos_src_flipped = qpos_full_flipped
        kpt_data_flipped = qpos2kpt(
            mj_model,
            qpos_src_flipped,
            args.freq_tgt,
            args.freq_tgt,
            interp_sec=0.0,
            end_default_sec=0.0,
            debug=args.debug,
            foot_contact_est=args.foot_contact_est
        )
        # Save the flipped trajectory (skipped when `save_path` is empty or None).
        if args.save_path:
            traj_id_start = 0 if args.start is None else args.start
            traj_id_end = len(kpt_data_flipped["qpos"]) if args.end is None else args.end
            if args.start is not None and args.end is not None:
                save_path = args.save_path.replace(".npz", f"_{traj_id_start}_{traj_id_end}-flipped.npz")
            else:
                save_path = args.save_path.replace(".npz", f"_flipped.npz")
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, **kpt_data_flipped)
            shapes = {k: tuple(v.shape) for k, v in kpt_data_flipped.items()}
            print(f"Saved flipped keypoint data to {out_path} with shapes: {shapes}")
        else:
            print("[INFO] `save_path` is empty; skipping serialization of the flipped keypoint npz file.")

    if args.aug_freq_ratio > 1:
        half_len = (args.aug_freq_ratio - 1) / 2
        for i in range(args.aug_freq_ratio):
            aug_freq_ratio = (1 + args.aug_freq_range * (i - half_len) / half_len)
            if aug_freq_ratio == 1:
                continue
            print("Save npz with aug freq ratio: ", aug_freq_ratio)
            freq_src_aug = freq_src * aug_freq_ratio

            kpt_data = qpos2kpt(
                mj_model,
                qpos_src,
                freq_src_aug,
                args.freq_tgt,
                interp_sec=0.0,
                end_default_sec=0.0,
                debug=args.debug,
                foot_contact_est=args.foot_contact_est
            )

            # Save the processed trajectory (skipped when `save_path` is empty or None).
            if args.save_path:
                traj_id_start = 0 if args.start is None else args.start
                traj_id_end = len(kpt_data["qpos"]) if args.end is None else args.end
                if args.start is not None and args.end is not None:
                    save_path = args.save_path.replace(".npz", f"_{traj_id_start}_{traj_id_end}X{aug_freq_ratio}.npz")
                else:
                    save_path = args.save_path.replace(".npz", f"*{aug_freq_ratio}.npz")
                out_path = Path(save_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_path, **kpt_data)

            if args.aug_flip:
                qpos_full_flipped = qpos_flip_around_x(kpt_data["qpos"], DOF_MIRROR_MAPPING)
                qpos_src_flipped = qpos_full_flipped
                kpt_data_flipped = qpos2kpt(
                    mj_model,
                    qpos_src_flipped,
                    args.freq_tgt,
                    args.freq_tgt,
                    interp_sec=0.0,
                    end_default_sec=0.0,
                    debug=args.debug,
                    foot_contact_est=args.foot_contact_est
                )
                # Save the flipped trajectory (skipped when `save_path` is empty or None).
                if args.save_path:
                    traj_id_start = 0 if args.start is None else args.start
                    traj_id_end = len(kpt_data_flipped["qpos"]) if args.end is None else args.end
                    if args.start is not None and args.end is not None:
                        save_path = args.save_path.replace(".npz", f"_{traj_id_start}_{traj_id_end}X{aug_freq_ratio}-flipped.npz")
                    else:
                        save_path = args.save_path.replace(".npz", f"*{aug_freq_ratio}_flipped.npz")
                    out_path = Path(save_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out_path, **kpt_data_flipped)

    logging.info("converting succeed.")
    return kpt_data


if __name__ == "__main__":
    run_pipeline(tyro.cli(Args))
