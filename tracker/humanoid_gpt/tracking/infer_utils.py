import jax
import numpy as np
import mujoco.viewer
import mujoco.mjx as mjx
from flax import struct
import onnxruntime as rt
from ml_collections import config_dict
from scipy.spatial.transform import Rotation as R

from tracking import constants as consts
from tracking.constants import ACTION_JOINT_NAMES, OBS_JOINT_NAMES, KPT_NAMES
from utils.transforms_np import (
    batch_base2navi,
    quat2mat,
    quat2yaw,
    WarpPi
)
from utils.sim_mj import (
    MJSim,
    geoms_colliding as mj_coll,
    get_sensor_data as mj_sensor
)
from utils.sim_mjx import (
    geoms_colliding as mjx_coll,
    get_sensor_data as mjx_sensor
)

EMA_ALPHA = 0.8


def g1_infer_env_config(ctrl_dt=0.02) -> config_dict.ConfigDict:
    """Lightweight env config for inference / deploy (no training-specific fields)."""
    return config_dict.create(
        ctrl_dt=ctrl_dt,
        action_scale=0.25,
        soft_joint_pos_limit_factor=0.90,
    )


def apply_ema_qpos(qpos, alpha=EMA_ALPHA):
    """Apply EMA (Exponential Moving Average) smoothing to reference qpos trajectory.

    smoothed[0] = qpos[0]
    smoothed[t] = smoothed[t-1] * alpha + qpos[t] * (1 - alpha)

    Handles quaternion (indices 3:7) sign continuity and renormalization.

    Args:
        qpos: np.ndarray of shape (T, D) where D >= 7.
        alpha: EMA coefficient. 0.8 means 80% previous + 20% current.

    Returns:
        Smoothed qpos array of the same shape.
    """
    T = len(qpos)
    if T <= 1:
        return qpos.copy()

    smoothed = np.empty_like(qpos)
    smoothed[0] = qpos[0]

    for t in range(1, T):
        q_prev = smoothed[t - 1, 3:7]
        q_curr = qpos[t, 3:7]
        if np.dot(q_prev, q_curr) < 0:
            q_curr = -q_curr

        smoothed[t, :3] = smoothed[t - 1, :3] * alpha + qpos[t, :3] * (1 - alpha)
        smoothed[t, 7:] = smoothed[t - 1, 7:] * alpha + qpos[t, 7:] * (1 - alpha)

        q_blended = q_prev * alpha + q_curr * (1 - alpha)
        q_norm = np.linalg.norm(q_blended)
        smoothed[t, 3:7] = q_blended / max(q_norm, 1e-8)

    return smoothed


NUM_ACTION = 29  # len(ACTION_JOINT_NAMES)
NUM_STATE = 136  # 3+3+29+29+29+29+1+3+6+2+2 (gyro+gvec+jnt_qpos+jnt_qvel+last_act+ref_jnt+height+gvec+cvel+yaw+xy)


@struct.dataclass
class State:
    mj_data: mujoco.MjData = None
    mjx_data: mjx.Data = None
    rng: jax.Array = None
    info: dict = None


class G1TrackMjSim(MJSim):
    def __init__(
        self,
        init_qpos,
        headless=False,
        ctrl_dt=0.02,
        sim_dt=0.001,
        xml_path=str(consts.ROOT_PATH / "scene_mjx_track.xml"),
    ):
        super().__init__(xml_path=xml_path, ctrl_dt=ctrl_dt, sim_dt=sim_dt, headless=headless)
        self.kps = np.float32(consts.KPs)
        self.kds = np.float32(consts.KDs)
        self.torque_limit = np.float32(consts.TORQUE_LIMIT)
        self.init_qpos = np.float32(init_qpos)


class G1TrackInferFn():
    def __init__(
        self,
        env_config,
        mj_model: mujoco.MjModel,
        nn_policy: rt.InferenceSession,
        num_envs: int = 1,
        use_mjx: bool = False,
        privileged: bool = False,
    ):
        self.env_config = env_config
        self.mj_model = mj_model
        self.nn_policy = nn_policy
        self.num_envs = num_envs
        self.use_mjx = use_mjx
        self.privileged = privileged

        # init
        self.ctrl_id_act = np.int32(
            [self.mj_model.actuator(n).id for n in ACTION_JOINT_NAMES]
        )
        self.ctrl_id_obs = np.int32(
            [self.mj_model.actuator(n).id for n in OBS_JOINT_NAMES]
        )
        self.dof_ids_g1 = np.int32(
            np.hstack([self.mj_model.joint(n).dofadr for n in consts.MotorName.FULL])
        )
        self.qpos_ids_g1 = np.int32(
            np.hstack([self.mj_model.joint(n).qposadr for n in consts.MotorName.FULL])
        )

        self._nom_jnt_qpos = np.full(
            (self.num_envs, consts.NUM_JOINT), consts.DEFAULT_QPOS[7:]
        )
        self.act_scale = np.array(consts.ACTION_SCALE)
        self.dt = env_config.ctrl_dt

        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self.env_config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self.env_config.soft_joint_pos_limit_factor

        self.site_id_chest = self.mj_model.site("chest").id
        self.site_ids_feet = np.array(
            [self.mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self.geom_id_floor = self.mj_model.geom("floor").id
        self.site_id_torso_imu = self.mj_model.site("imu_in_torso").id
        self.site_id_pelvis_imu = self.mj_model.site("imu_in_pelvis").id
        self.geom_ids_left_feet = np.array(
            [self.mj_model.geom(name).id for name in consts.LEFT_FEET_GEOMS]
        )
        self.geom_ids_right_feet = np.array(
            [self.mj_model.geom(name).id for name in consts.RIGHT_FEET_GEOMS]
        )
        self.body_ids_kpt_full = np.array([self.mj_model.body(n).id for n in KPT_NAMES])
        self.kpt_id_pelvis = KPT_NAMES.index("pelvis")
        self.kpt_id_torso = KPT_NAMES.index("torso_link")
        self.kpt_id_peltor = np.hstack([self.kpt_id_pelvis, self.kpt_id_torso])

        self.num_jnt = len(consts.DEFAULT_QPOS[7:])
        self.num_kpt = len(KPT_NAMES)

        self.info = {
            "step": 0,
            "nn_action": np.zeros((self.num_envs, NUM_ACTION), dtype=np.float32),
            "gyro_pelvis": np.zeros((self.num_envs, 3)),
            "gvec_pelvis": np.zeros((self.num_envs, 3)),
            "linvel_pelvis": np.zeros((self.num_envs, 3)),
            "qpos": np.zeros((self.num_envs, self.mj_model.nq)),
            "qvel": np.zeros((self.num_envs, self.mj_model.nv)),
            "last_action": np.zeros((self.num_envs, len(self.ctrl_id_act))),
            "motor_targets": self._nom_jnt_qpos.copy(),
            # hint state
            "navi_pelvis_rpy": np.zeros((self.num_envs, 3)),
            "navi_torso_rpy": np.zeros((self.num_envs, 3)),
            "feet_contact": np.zeros((self.num_envs, 2)),
            # kpt actual state
            "acu_kpt2gv_pose": np.full((self.num_envs, self.num_kpt, 4, 4), np.eye(4)),
            "acu_kpt_cvel_in_gv": np.zeros((self.num_envs, self.num_kpt, 6)),
            # kpt reference state
            # "ref_next_navi_vel": np.zeros((self.num_envs, 3)),
            # "next_jnt_qpos_res": np.zeros((self.num_envs, self.num_jnt)),
            # "next_jnt_qvel_res": np.zeros((self.num_envs, self.num_jnt)),
            # "ref_next_kpt_npose": np.full(
            #     (self.num_envs, self.num_kpt, 4, 4), np.eye(4)
            # ),
            # "ref_next_kpt_cvel": np.zeros((self.num_envs, self.num_kpt, 6)),
        }

    def infer_onnx(self, state: State, ref_state: np.ndarray | dict) -> np.ndarray:
        ref_next = ref_state.get("ref_next", ref_state["ref_curr"])
        self.update_state(state, ref_state)
        if self.privileged:
            obs = self.get_nn_priv_state(self.info, ref_next, self.info["last_action"])
        else:
            obs = self.get_nn_state(self.info, ref_next, self.info["last_action"])

        nn_action = self.nn_policy.infer(obs)

        motor_targets = self.nn2motor_action(nn_action)
        self.info["motor_targets"] = motor_targets.copy()

        self.info["step"] += 1
        self.info["nn_action"] = nn_action
        self.info["last_action"] = nn_action.copy()
        return motor_targets

    def infer_onnx_real(
        self,
        root_quat: np.ndarray,
        root_gyro: np.ndarray,
        jnt_qpos: np.ndarray,
        jnt_qvel: np.ndarray,
        ref_state: dict,
    ) -> np.ndarray:
        """Lightweight inference for real robot — no robot-side mj_forward().

        For non-privileged policy, the observation only needs direct sensor
        readings (gyro, gravity vector, joint qpos/qvel) plus reference FK
        data.  Robot-side FK (xpos, xmat, cvel) is NOT required, so we skip
        the expensive ``mj_forward()`` call on the robot state entirely.
        """
        ref_next = ref_state.get("ref_next", ref_state["ref_curr"])

        # Build full qpos / qvel arrays for update_coord_cmd (no FK needed)
        qpos = np.zeros((1, self.mj_model.nq), dtype=np.float32)
        qpos[0, :3] = [0.0, 0.0, 0.78]
        qpos[0, 3:7] = root_quat
        qpos[0, 7:] = jnt_qpos

        qvel = np.zeros((1, self.mj_model.nv), dtype=np.float32)
        qvel[0, 3:6] = root_gyro
        qvel[0, 6:] = jnt_qvel

        # Gravity vector from quaternion (pure math, no FK)
        pelvis2world_rot = quat2mat(root_quat[None])  # (1, 3, 3)
        gvec_pelvis = -pelvis2world_rot.transpose(0, 2, 1)[..., 2]  # (1, 3)

        # Populate info fields used by get_nn_state
        self.info["gyro_pelvis"][:] = root_gyro[None]
        self.info["gvec_pelvis"][:] = gvec_pelvis
        self.info["qpos"][:] = qpos
        self.info["qvel"][:] = qvel

        # Align yaw_d / xy_d computation with update_coord_cmd (same as infer_onnx)
        self.update_coord_cmd(ref_state)

        # Construct observation & run ONNX (ref_next for ref data, matching training)
        obs = self.get_nn_state(self.info, ref_next, self.info["last_action"])
        nn_action = self.nn_policy.infer(obs)

        motor_targets = self.nn2motor_action(nn_action)
        self.info["motor_targets"] = motor_targets.copy()

        self.info["step"] += 1
        self.info["nn_action"] = nn_action
        self.info["last_action"] = nn_action.copy()
        return motor_targets

    # ------------------------------------------------------------------
    # Fast deploy path: pre-allocated buffers + IOBinding (CPU)
    # ------------------------------------------------------------------

    def _ensure_fast_buffers(self) -> None:
        """Lazily allocate scratch buffers + IO binding for the fast path."""
        if getattr(self, "_fast_ready", False):
            return
        nq, nv = self.mj_model.nq, self.mj_model.nv
        ne = self.num_envs
        self._qpos_buf = np.zeros((ne, nq), dtype=np.float32)
        self._qpos_buf[:, :3] = (0.0, 0.0, 0.78)
        self._qvel_buf = np.zeros((ne, nv), dtype=np.float32)
        self._gvec_buf = np.empty((ne, 3), dtype=np.float32)

        # Composite indices: precompose ``qpos_ids_g1[ctrl_id_obs]`` once so
        # ``info["qpos"][:, idx]`` is a single fancy index per step instead of
        # two indexing ops + a temp.
        self._qpos_obs_ids = self.qpos_ids_g1[self.ctrl_id_obs]
        self._qvel_obs_ids = self.dof_ids_g1[self.ctrl_id_obs]
        self._nom_obs = np.ascontiguousarray(
            self._nom_jnt_qpos[:, self.ctrl_id_obs], dtype=np.float32
        )

        # Use persistent host IOBinding only when the provider marks it safe.
        if getattr(self.nn_policy, "supports_host_iobinding", False):
            self._obs_buf = self.nn_policy.bind_input_buffer(
                (ne, NUM_STATE)
            )
            self._use_iobinding = True
        else:
            self._obs_buf = np.zeros((ne, NUM_STATE), dtype=np.float32)
            self._use_iobinding = False

        # Pre-compute the obs slice layout (matches get_nn_state hstack order)
        # 3 + 3 + 29 + 29 + 29 + 29 + 1 + 3 + 6 + 2 + 2 = 136
        s = 0
        def _adv(n: int) -> slice:
            nonlocal s
            sl = slice(s, s + n); s += n; return sl
        self._sl_gyro = _adv(3)
        self._sl_gvec = _adv(3)
        self._sl_qposd = _adv(self.num_jnt)
        self._sl_qvel = _adv(self.num_jnt)
        self._sl_lastact = _adv(NUM_ACTION)
        self._sl_refqd = _adv(self.num_jnt)
        self._sl_height = _adv(1)
        self._sl_refgvec = _adv(3)
        self._sl_refcvel = _adv(6)
        self._sl_yaw = _adv(2)
        self._sl_xy = _adv(2)
        assert s == NUM_STATE, (s, NUM_STATE)
        self._fast_ready = True

    def _fill_obs_inplace(
        self,
        ref_next: dict,
        last_action: np.ndarray,
    ) -> None:
        """Write obs vector into ``self._obs_buf`` in place (no allocation).

        Mirrors :py:meth:`get_nn_state` exactly but writes each field at a
        precomputed slice instead of building intermediates + ``np.hstack``.
        """
        info = self.info
        buf = self._obs_buf
        nom_obs = self._nom_obs

        buf[:, self._sl_gyro] = info["gyro_pelvis"]
        buf[:, self._sl_gvec] = info["gvec_pelvis"]
        # Joint state — single fancy-index using the precomposed obs ids,
        # then in-place subtraction of the cached nominal qpos slice.
        np.subtract(
            info["qpos"][:, self._qpos_obs_ids], nom_obs,
            out=buf[:, self._sl_qposd],
        )
        buf[:, self._sl_qvel] = info["qvel"][:, self._qvel_obs_ids]
        buf[:, self._sl_lastact] = last_action

        np.subtract(
            ref_next["qpos"][:, 7:][:, self.ctrl_id_obs], nom_obs,
            out=buf[:, self._sl_refqd],
        )
        ref_root_pose = ref_next["kpt2gv_pose"][:, 0]
        buf[:, self._sl_height] = ref_root_pose[:, 2, 3:4]
        # -ref_root_pose[:, :3, :3].T[..., 2] == -ref_root_pose[:, 2, :3]
        np.negative(ref_root_pose[:, 2, :3], out=buf[:, self._sl_refgvec])
        buf[:, self._sl_refcvel] = ref_next["kpt_cvel_in_gv"][:, 0]

        yaw_d = info["yaw_d"]
        buf[:, self._sl_yaw.start] = np.cos(yaw_d)
        buf[:, self._sl_yaw.start + 1] = np.sin(yaw_d)
        buf[:, self._sl_xy] = info["xy_d"]

    def infer_onnx_real_fast(
        self,
        root_quat: np.ndarray,
        root_gyro: np.ndarray,
        jnt_qpos: np.ndarray,
        jnt_qvel: np.ndarray,
        ref_state: dict,
    ) -> np.ndarray:
        """Same as :py:meth:`infer_onnx_real` but with zero per-step alloc.

        Pre-allocated scratch buffers + IOBinding shave both the steady-state
        latency and the long-tail jitter of the ONNX call (see
        ``deploy/onboard_deploy/bench_online.py``).
        """
        self._ensure_fast_buffers()
        ref_next = ref_state.get("ref_next", ref_state["ref_curr"])

        # Update qpos / qvel scratch in place (root xy/z stays at default standing)
        self._qpos_buf[0, 3:7] = root_quat
        self._qpos_buf[0, 7:] = jnt_qpos
        self._qvel_buf[0, 3:6] = root_gyro
        self._qvel_buf[0, 6:] = jnt_qvel

        # Fast pure-numpy quat -> gravity vector (avoid scipy Rotation overhead)
        w, x, y, z = root_quat
        # gvec = -R(quat).T @ [0,0,1] = -[2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
        self._gvec_buf[0, 0] = -2.0 * (x * z - w * y)
        self._gvec_buf[0, 1] = -2.0 * (y * z + w * x)
        self._gvec_buf[0, 2] = -(1.0 - 2.0 * (x * x + y * y))

        info = self.info
        info["gyro_pelvis"][0] = root_gyro
        info["gvec_pelvis"][:] = self._gvec_buf
        info["qpos"][:] = self._qpos_buf
        info["qvel"][:] = self._qvel_buf

        self.update_coord_cmd(ref_state)

        # Build obs in place into the (possibly IOBinding-bound) buffer
        self._fill_obs_inplace(ref_next, info["last_action"])

        if self._use_iobinding:
            nn_action = self.nn_policy.infer_bound()
        else:
            nn_action = self.nn_policy.infer(self._obs_buf)

        motor_targets = self.nn2motor_action(nn_action)
        # In-place writes avoid two per-step allocations (last_action +
        # motor_targets) without changing semantics for downstream consumers
        # that read ``info["last_action"]`` and ``info["motor_targets"]``.
        info["last_action"][...] = nn_action
        info["motor_targets"][...] = motor_targets
        info["nn_action"] = nn_action
        info["step"] += 1
        return motor_targets

    def get_nn_state(self, info, ref_state: np.ndarray, last_action: np.ndarray):
        ref_qpos = ref_state["qpos"][:, 7:]
        ref_qvel = ref_state["qvel"][:, 6:]

        gyro_pelvis = info["gyro_pelvis"]
        gvec_pelvis = info["gvec_pelvis"]
        joint_qpos = info["qpos"][:, self.qpos_ids_g1]
        joint_qvel = info["qvel"][:, self.dof_ids_g1]

        ref_root_gv_pose = ref_state["kpt2gv_pose"][:, 0]
        ref_root_cvel_in_gv = ref_state["kpt_cvel_in_gv"][:, 0]

        # Encode yaw as [cos, sin] to match training logic
        yaw_cmd = np.stack([np.cos(info["yaw_d"]), np.sin(info["yaw_d"])], axis=-1)  # (num_envs, 2)
        # xy_d is already in navi frame (R(-yaw_curr) @ (ref-curr)); training uses same
        xy_cmd = info["xy_d"]

        state = np.hstack(
            [
                # pose state
                gyro_pelvis,  # 3
                gvec_pelvis,  # 3
                # joint state
                (joint_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],  # 29
                joint_qvel[:, self.ctrl_id_obs],  # 29  * scale_vel
                last_action,
                # commands
                (ref_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],
                # ref_qvel[:, self.ctrl_id_obs],
                ref_root_gv_pose[:, 2, 3][:, None],  # height (num_envs, 1)
                -ref_root_gv_pose[:, :3, :3].transpose((0, 2, 1))[..., 2],
                ref_root_cvel_in_gv,
                # global
                yaw_cmd,  # (num_envs, 2)
                xy_cmd    # (num_envs, 2)
            ]
        ).astype(np.float32)
        return state

    def get_nn_priv_state(self, info, ref_state: np.ndarray, last_action: np.ndarray):
        ref_qpos = ref_state["qpos"][:, 7:]
        ref_qvel = ref_state["qvel"][:, 6:]

        gyro_pelvis = info["gyro_pelvis"]
        gvec_pelvis = info["gvec_pelvis"]
        linvel_pelvis = info["linvel_pelvis"]
        joint_qpos = info["qpos"][:, self.qpos_ids_g1]
        joint_qvel = info["qvel"][:, self.dof_ids_g1]

        ref_root_gv_pose = ref_state["kpt2gv_pose"][:, 0]
        ref_root_cvel_in_gv = ref_state["kpt_cvel_in_gv"][:, 0]

        # Encode yaw as [cos, sin] to match training's priv_yaw_cmd
        priv_yaw_cmd = np.stack([np.cos(info["yaw_d"]), np.sin(info["yaw_d"])], axis=-1)  # (num_envs, 2)

        privileged_state = np.hstack(
            [
                # pose state
                gyro_pelvis,  # 3
                gvec_pelvis,  # 3
                # joint state
                (joint_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],  # 23
                joint_qvel[:, self.ctrl_id_obs],  # 23
                last_action,  # num_actions
                # reference motion
                (ref_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],
                # ref_qvel[:, self.ctrl_id_obs],
                # hint state
                linvel_pelvis,  # 3
                info["acu_root2gv_lin_vel"],
                info["acu_root2gv_ang_vel"],
                info["acu_kpt2gv_pose"][..., :3, :2].reshape(self.num_envs, -1),
                info["acu_kpt2gv_pose"][..., :3, 3].reshape(self.num_envs, -1),
                info["acu_kpt_cvel_in_gv"].reshape(self.num_envs, -1),
                # res
                info["next_ref2acu_gv_vel"],
                info["next_ref2acu_kpt_pose"][..., :3, :2].reshape(self.num_envs, -1),
                info["next_ref2acu_kpt_pose"][..., :3, 3].reshape(self.num_envs, -1),
                info["next_ref2acu_kpt_cvel"].reshape(self.num_envs, -1),
                # root command
                ref_root_gv_pose[:, 2, 3][:, None],  # height (num_envs, 1)
                -ref_root_gv_pose[:, :3, :3].transpose((0, 2, 1))[..., 2],
                ref_root_cvel_in_gv,
                # global
                priv_yaw_cmd,  # (num_envs, 2)
                info["xy_d"],  # (num_envs, 2)
            ]
        ).astype(np.float32)
        return privileged_state

    def update_state(self, state: State, ref_state):
        if self.use_mjx:
            mjx_data_np = jax.device_get(state.mjx_data)
            qpos = mjx_data_np.qpos.copy()
            qvel = mjx_data_np.qvel.copy()
            gyro_pelvis = mjx_sensor(self.mj_model, mjx_data_np, "gyro_pelvis")
            pelvis2world_rot = quat2mat(qpos[:, 3:7])
            pelvis2world_pos = qpos[:, :3]

            # privileged_state
            linvel_pelvis = mjx_sensor(
                self.mj_model, mjx_data_np, "local_linvel_pelvis"
            )
            torso2world_rot = mjx_data_np.site_xmat[:, self.site_id_torso_imu]
            feet_contact = self.get_mjx_feet_contact(mjx_data_np)
            kpt2wrd_rot = mjx_data_np.xmat[:, self.body_ids_kpt_full]
            kpt2wrd_pos = mjx_data_np.xpos[:, self.body_ids_kpt_full]
            acu_kpt_cvel_in_wrd = mjx_data_np.cvel[:, self.body_ids_kpt_full]

        else:
            mj_data = state.mj_data
            gyro_pelvis = mj_sensor(self.mj_model, mj_data, "gyro_pelvis")
            pelvis2world_rot = quat2mat(mj_data.qpos[3:7][None])
            pelvis2world_pos = mj_data.qpos[:3][None]
            qpos = mj_data.qpos.copy()[None]
            qvel = mj_data.qvel.copy()[None]

            # privileged state
            linvel_pelvis = mj_sensor(self.mj_model, mj_data, "local_linvel_pelvis")
            torso2world_rot = mj_data.site_xmat[self.site_id_torso_imu]
            torso2world_rot = torso2world_rot.reshape(1, 3, 3)
            feet_contact = self.get_mj_feet_contact(mj_data)[None]
            kpt2wrd_rot = mj_data.xmat[self.body_ids_kpt_full].reshape(-1, 3, 3)
            kpt2wrd_pos = mj_data.xpos[self.body_ids_kpt_full][None]
            acu_kpt_cvel_in_wrd = mj_data.cvel[self.body_ids_kpt_full][None]

        # state
        gvec_pelvis = -pelvis2world_rot.transpose(0, 2, 1)[..., 2]
        self.info["gyro_pelvis"][:] = gyro_pelvis.copy()
        self.info["linvel_pelvis"][:] = linvel_pelvis.copy()
        self.info["gvec_pelvis"][:] = gvec_pelvis.copy()
        self.info["qpos"][:] = qpos.copy()
        self.info["qvel"][:] = qvel.copy()

        # privileged state
        navi2world_rot = batch_base2navi(pelvis2world_rot)
        pelvis2navi_rot = navi2world_rot.transpose(0, 2, 1) @ pelvis2world_rot
        torso2navi_rot = navi2world_rot.transpose(0, 2, 1) @ torso2world_rot
        self.info["navi2world_rot"] = navi2world_rot.copy()
        self.info["navi_pelvis_rpy"][:] = (
            R.from_matrix(pelvis2navi_rot).as_euler("xyz").copy()
        )
        self.info["navi_torso_rpy"][:] = (
            R.from_matrix(torso2navi_rot).as_euler("xyz").copy()
        )
        self.info["feet_contact"][:] = feet_contact.copy()

        # gravity view frame
        acu_gv2wrd_pose = np.full((self.num_envs, 4, 4), np.eye(4))
        acu_gv2wrd_pose[:, :3, :3] = navi2world_rot
        acu_gv2wrd_pose[:, :2, 3] = pelvis2world_pos[:, :2]
        acu_kpt2wrd_pose = np.full((self.num_envs, self.num_kpt, 4, 4), np.eye(4))
        acu_kpt2wrd_pose[:, :, :3, :3] = kpt2wrd_rot
        acu_kpt2wrd_pose[:, :, :3, 3] = kpt2wrd_pos
        acu_kpt2gv_pose = np.linalg.inv(acu_gv2wrd_pose[:, None]) @ acu_kpt2wrd_pose
        self.info["acu_kpt2gv_pose"] = acu_kpt2gv_pose

        acu_kpt_cvel_in_gv = np.zeros_like(acu_kpt_cvel_in_wrd)
        R_wrd2gv = np.swapaxes(acu_gv2wrd_pose[..., :3, :3], -1, -2)  # R_gv2wrd^T = R_wrd2gv
        acu_kpt_cvel_in_gv[..., :3] = np.einsum("...ij,...kj->...ki", R_wrd2gv, acu_kpt_cvel_in_wrd[..., :3])
        acu_kpt_cvel_in_gv[..., 3:] = np.einsum("...ij,...kj->...ki", R_wrd2gv, acu_kpt_cvel_in_wrd[..., 3:])
        self.info["acu_kpt_cvel_in_gv"][:] = acu_kpt_cvel_in_gv

        self.update_coord_cmd(ref_state)

        if "ref_next" in ref_state:
            ref_next = ref_state["ref_next"]

            next_ref_gv2wrd_pose = ref_next["gv2wrd_pose"]
            self.info["next_ref2acu_gv_pose"] = (
                np.linalg.inv(acu_gv2wrd_pose) @ next_ref_gv2wrd_pose
            )
            next_ref2acu_kpt_pose = (
                np.linalg.inv(acu_kpt2gv_pose) @ ref_next["kpt2gv_pose"]
            )
            self.info["next_ref2acu_kpt_pose"] = next_ref2acu_kpt_pose

            ref_next_navi_vel = ref_next["gv_vel"]
            acu_root2gv_lin_vel = pelvis2navi_rot @ linvel_pelvis
            acu_root2gv_ang_vel = pelvis2navi_rot @ gyro_pelvis
            self.info["acu_root2gv_lin_vel"] = acu_root2gv_lin_vel
            self.info["acu_root2gv_ang_vel"] = acu_root2gv_ang_vel
            # self.info["acu_navi_vel"] = np.hstack(
            #     [navi_pelvis_lin_vel[:, :2], navi_pelvis_ang_vel[:, 2:3]]
            # )
            next_vel_lin_res = ref_next_navi_vel[:, :2] - acu_root2gv_lin_vel[:, :2]
            next_vel_ang_res = ref_next_navi_vel[:, 2] - acu_root2gv_ang_vel[:, 2]
            self.info["next_ref2acu_gv_vel"] = np.hstack(
                [next_vel_lin_res, next_vel_ang_res[:, None]]
            )
            # self.info["ref_next_navi_vel"][:] = ref_next["navi_vel"]
            # self.info["ref_next_kpt_npose"] = ref_next["kpt_cvel_in_gv"]
            # self.info["ref_next_kpt_cvel"] = ref_next["kpt_cvel"]
            # self.info["next_jnt_qpos_res"] = ref_next["qpos"][:, 7:] - qpos[:, 7:]
            # self.info["next_jnt_qvel_res"] = ref_next["qvel"][:, 6:] - qvel[:, 6:]
            self.info["next_ref2acu_kpt_cvel"] = (
                ref_next["kpt_cvel_in_gv"] - acu_kpt_cvel_in_gv
            )

    def update_coord_cmd(self, ref_state):
        curr_ref_state = ref_state["ref_curr"]
        q_ref = curr_ref_state["qpos"][:, 3:7]
        q_curr = self.info["qpos"][:, 3:7]
        yaw_ref = quat2yaw(q_ref)
        yaw_curr = quat2yaw(q_curr)
        yaw_cmd = curr_ref_state.get("yaw_cmd", np.zeros_like(yaw_curr))
        yaw_target = yaw_cmd + yaw_ref
        yaw_d = WarpPi(yaw_target - yaw_curr)

        p_ref = curr_ref_state["qpos"][:, :2]
        p_curr = self.info["qpos"][:, :2]

        c, s = np.cos(yaw_cmd), np.sin(yaw_cmd)
        R_m = np.stack([
            np.stack([c, -s], axis=-1),
            np.stack([s, c], axis=-1),
        ], axis=-2)
        xy_ref = p_ref
        xy_curr = p_curr
        xy_cmd = curr_ref_state.get("xy_cmd", np.zeros_like(xy_curr))
        xy_target = np.einsum("...ij,...j->...i", R_m, (xy_cmd + xy_ref))
        xy_d = xy_target - xy_curr

        c, s = np.cos(-yaw_curr), np.sin(-yaw_curr)
        R_m = np.stack([
            np.stack([c, -s], axis=-1),
            np.stack([s, c], axis=-1),
        ], axis=-2)
        xy_d = np.einsum("...ij,...j->...i", R_m, xy_d)
        # xy_d = np.clip(xy_d, self.coord_cfg["xy_move_range"][0], self.coord_cfg["xy_move_range"][1])

        self.info["yaw_d"] = yaw_d
        self.info["xy_d"] = xy_d

    def get_mj_feet_contact(self, mj_data: mujoco.MjData) -> np.ndarray:
        """
        Returns the contact state of the left and right feet.

        Contact state encoding:
                -1: fully in air (no contact)
                 1: partial contact (some or all foot geoms in contact)
        """
        left_contacts = np.array(
            [
                mj_coll(mj_data, geom_id, self.geom_id_floor)
                for geom_id in self.geom_ids_left_feet
            ]
        )
        right_contacts = np.array(
            [
                mj_coll(mj_data, geom_id, self.geom_id_floor)
                for geom_id in self.geom_ids_right_feet
            ]
        )

        left_state = np.where(left_contacts.any(), 1, -1)
        right_state = np.where(right_contacts.any(), 1, -1)

        return np.array([left_state, right_state])

    def get_mjx_feet_contact(self, mjx_data: mjx.Data):
        left_contacts = np.array(
            [
                mjx_coll(mjx_data, geom_id, self.geom_id_floor)
                for geom_id in self.geom_ids_left_feet
            ]
        )
        right_contacts = np.array(
            [
                mjx_coll(mjx_data, geom_id, self.geom_id_floor)
                for geom_id in self.geom_ids_right_feet
            ]
        )

        left_state = np.where(left_contacts.any(axis=0), 1, -1)[:, None]
        right_state = np.where(right_contacts.any(axis=0), 1, -1)[:, None]

        return np.concatenate([left_state, right_state], axis=1)

    def nn2motor_action(self, nn_action):
        motor_targets = self._nom_jnt_qpos.copy()
        motor_targets[:, self.ctrl_id_act] = (
            self._nom_jnt_qpos[:, self.ctrl_id_act]
            + nn_action
            * self.env_config.action_scale
            * self.act_scale[self.ctrl_id_act]
        )
        return motor_targets

