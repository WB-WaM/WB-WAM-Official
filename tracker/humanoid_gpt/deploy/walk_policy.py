"""Walk policy: ONNX locomotion controller with gait-phase clock."""

import numpy as np
import onnxruntime as rt

from deploy.constants import DEFAULT_QPOS
from tracking.policy import preload_onnx_cuda_dependencies


def _quat2gvec(quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity vector in body frame from quaternion (w,x,y,z)."""
    w, x, y, z = quat_wxyz
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    gx = -2 * (x * z - y * w)
    gy = -2 * (y * z + x * w)
    gz = -1 + 2 * (x * x + y * y)
    return np.array([gx, gy, gz], dtype=np.float32)


class WalkPolicy:
    """ONNX walk policy with 12-DoF lower-body action and gait phase clock."""

    def __init__(
        self,
        onnx_path: str,
        action_dim: int = 12,
        action_scale: float = 0.5,
        infer_dt: float = 0.02,
        gait_freq: float = 1.2,
        provider: str = "auto",
    ):
        self.infer_dt = infer_dt
        self.action_dim = action_dim
        self.action_scale = action_scale

        # fmt: off
        self._actuator_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self._obs_joint_ids = np.array([
            0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 11,
            12, 13, 14,
            15, 16, 17, 18,
            22, 23, 24, 25,
        ])
        # fmt: on
        self._default_qpos = DEFAULT_QPOS.copy()

        self.phase_dt = 2 * np.pi * infer_dt * gait_freq
        self._init_phase = np.array([0, np.pi])
        self._stance_phase = np.array([0.0, 0.0])

        self.reset()

        requested_provider = provider.strip().lower()
        if requested_provider not in {"auto", "cpu", "cuda"}:
            raise ValueError("provider must be auto, cpu, or cuda")
        if requested_provider != "cpu":
            preload_onnx_cuda_dependencies()
        available = rt.get_available_providers()
        if requested_provider == "cpu":
            providers = ["CPUExecutionProvider"]
            expected_provider = "CPUExecutionProvider"
        elif requested_provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider is unavailable for walk policy"
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            expected_provider = "CUDAExecutionProvider"
        else:
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            expected_provider = providers[0]
        self.infer_fn = rt.InferenceSession(onnx_path, providers=providers)
        actual_providers = self.infer_fn.get_providers()
        if requested_provider != "auto" and (
            not actual_providers or actual_providers[0] != expected_provider
        ):
            raise RuntimeError(
                "walk policy provider mismatch: "
                f"requested {expected_provider}, got {actual_providers}"
            )
        self.provider = actual_providers[0] if actual_providers else "unknown"

    def reset(self) -> None:
        """Reset gait phase and recurrent observation/action history."""
        self._step_ctr = 0
        self._timestamp_move2stop = 0
        self._phase = self._init_phase.copy()
        self._foot_height = 0.05
        self._loco_task_mask = 0
        self._last_has_vel = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._last_obs = np.zeros(73, dtype=np.float32)

    @property
    def default_qpos(self):
        return self._default_qpos.copy()

    @property
    def last_obs(self) -> np.ndarray:
        return self._last_obs.copy()

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def infer(self, root_quat, root_gyro, joint_qpos, joint_qvel, cmd_vel):
        """Run one walk-policy step.

        Args:
            root_quat: (4,) w,x,y,z
            root_gyro: (3,)
            joint_qpos: (29,)
            joint_qvel: (29,)
            cmd_vel: (3,)  [lin_x, lin_y, ang_yaw]

        Returns:
            motor_targets: (29,) full joint targets
        """
        nn_obs = self._get_obs(root_quat, root_gyro, joint_qpos, joint_qvel, cmd_vel)
        nn_action = self.infer_fn.run(["continuous_actions"], {"obs": nn_obs})[0]
        self._last_obs = nn_obs[0].copy()

        motor_targets = self._default_qpos.copy()
        motor_targets[self._actuator_ids] = (
            self._default_qpos[self._actuator_ids] + nn_action * self.action_scale
        )

        self._last_action = nn_action[0].copy()
        self._update_phase(cmd_vel)
        self._step_ctr += 1
        return motor_targets

    def _get_obs(self, root_quat, root_gyro, joint_qpos, joint_qvel, cmd_vel):
        gvec = _quat2gvec(root_quat)
        gait_phase = np.hstack([np.cos(self._phase), np.sin(self._phase)])
        qpos_obs = (joint_qpos - self._default_qpos)[self._obs_joint_ids]
        qvel_obs = joint_qvel[self._obs_joint_ids]
        obs_cmd = np.hstack([self._loco_task_mask, cmd_vel])

        obs = np.concatenate(
            [
                root_gyro,
                gvec,
                qpos_obs,
                qvel_obs,
                self._last_action,
                obs_cmd,
                [self._foot_height],
                gait_phase,
            ]
        )
        return obs.reshape(1, -1).astype(np.float32)

    def _update_phase(self, cmd_vel):
        has_vel = float(np.linalg.norm(cmd_vel) > 0.2)
        had_vel = self._last_has_vel

        move2stop = (had_vel == 1.0) & (has_vel == 0.0)
        stop2move = (had_vel == 0.0) & (has_vel == 1.0)

        self._timestamp_move2stop = np.where(
            move2stop, self._step_ctr + 50, self._timestamp_move2stop
        )
        after_delay = self._step_ctr > self._timestamp_move2stop
        moving = np.where((has_vel == 0.0) & after_delay, 0.0, 1.0)

        new_phase = (self._phase + self.phase_dt + np.pi) % (2 * np.pi) - np.pi
        phase = np.where(moving, new_phase, self._stance_phase)
        phase = np.where(stop2move, self._init_phase, phase)
        self._phase = phase

        self._loco_task_mask = moving
        self._last_has_vel = has_vel
