# Pico SMPL stream server for body tracking visualization

"""

# Recommended Command Line Arguments:
    # With VR3 PT visualization (by --vis_vr3pt) and optional SMPL body visualization (by --vis_smpl)
    # If you want to enable waist tracking in the VR3 PT visualization, please add --waist_tracking
    python pico_manager_thread_server.py --manager \
        --vis_vr3pt --vis_smpl \
        --waist_tracking

    # VR3 PT visualization only (without SMPL body) — lower latency
    python pico_manager_thread_server.py --manager --vis_vr3pt

# DEBUG VR3 PT VISUALIZATION:
    # A standalone test mode that captures one live frame and visualizes it.
    python pico_manager_thread_server.py --vr3pt_live
4
# TIMING COMPARISON:
    # The visualizer automatically reports timing every 5 seconds when running:
    #   [Vis Timing] vr3pt: X.XXms | smpl: X.XXms | render: X.XXms | vr3pt_only: X.XXms | both(vr3pt+smpl): X.XXms

"""

from collections import defaultdict, deque
from enum import Enum, IntEnum
import os
from pathlib import Path
import subprocess
import threading
import time
import sys
import termios
import tty
import select
import atexit

REPO_ROOT = Path(__file__).resolve().parents[4]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
for _import_root in (
    REPO_ROOT / "tracker" / "sonic",
    THIRD_PARTY_ROOT / "wuji_retargeting",
    THIRD_PARTY_ROOT / "xrobotoolkit_sdk",
):
    if _import_root.is_dir():
        _import_root_str = str(_import_root)
        if _import_root_str not in sys.path:
            sys.path.insert(0, _import_root_str)

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation as sRot
import torch
import zmq

from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)

try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        build_command_message,
        build_planner_message,
        pack_pose_message,
    )
except ImportError:

    def build_command_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_command_message unavailable")

    def build_planner_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_planner_message unavailable")

    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")


COMMAND_REPEAT_COUNT = 3
COMMAND_REPEAT_INTERVAL_S = 0.03
POSE_STATUS_INTERVAL_S = 1.0
COLLECTOR_CONTROL_TOPIC = "collector_control"
COLLECTOR_CONTROL_START = 1
COLLECTOR_CONTROL_SAVE = 2
COLLECTOR_CONTROL_DISCARD = 3
COLLECTOR_CONTROL_COMMAND_NAMES = {
    COLLECTOR_CONTROL_START: "start",
    COLLECTOR_CONTROL_SAVE: "save",
    COLLECTOR_CONTROL_DISCARD: "discard",
}


def send_command_packet(
    socket,
    *,
    start: bool,
    stop: bool,
    planner: bool,
    reason: str,
    repeats: int = COMMAND_REPEAT_COUNT,
    delay_s: float = COMMAND_REPEAT_INTERVAL_S,
) -> None:
    payload = build_command_message(start=start, stop=stop, planner=planner)
    for attempt in range(repeats):
        socket.send(payload)
        if attempt + 1 < repeats:
            time.sleep(delay_s)
    print(
        f"[Manager] Sent command ({reason}): "
        f"start={start} stop={stop} planner={planner} repeats={repeats}"
    )


def send_collector_control_packet(socket, *, command: int, sequence: int) -> None:
    command_name = COLLECTOR_CONTROL_COMMAND_NAMES.get(command)
    if command_name is None:
        raise ValueError(f"unsupported collector control command: {command}")
    payload = pack_pose_message(
        {
            "timestamp_monotonic_ns": np.array([time.monotonic_ns()], dtype=np.int64),
            "sequence": np.array([sequence], dtype=np.int64),
            "command": np.array([command], dtype=np.int32),
        },
        topic=COLLECTOR_CONTROL_TOPIC,
    )
    socket.send(payload)
    print(f"[Manager] Sent collector_control: {command_name} sequence={sequence}")


try:
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    print("Warning: gear_sonic.isaac_utils.rotations not available.")
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer
except ImportError:
    print("Warning: VR3PtPoseVisualizer not available (pyvista may not be installed).")
    VR3PtPoseVisualizer = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses
except ImportError:
    print("Warning: get_g1_key_frame_poses not available (pyvista may not be installed).")
    get_g1_key_frame_poses = None

def _resolve_wuji_retargeting_root() -> Path:
    env_root = os.environ.get("WUJI_RETARGETING_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    workspace_parent = REPO_ROOT.parent
    candidates = (
        THIRD_PARTY_ROOT / "wuji_retargeting",
        THIRD_PARTY_ROOT / "wuji-retargeting",
        workspace_parent / "wuji-retargeting",
        workspace_parent / "wuji-hand",
    )
    for candidate in candidates:
        if (candidate / "wuji_retargeting").is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_wuji_retargeting_config(root: Path) -> Path:
    env_config = os.environ.get("WUJI_RETARGETING_CONFIG")
    if env_config:
        return Path(env_config).expanduser().resolve()

    candidates = (
        root / "example" / "config" / "customer.yaml",
        root / "example" / "config" / "qc_retarget.yaml",
        root / "example" / "config" / "adaptive_analytical_avp.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_wuji_glove_retargeting_config(root: Path, hand_side: str) -> Path:
    env_key = f"WUJI_GLOVE_{hand_side.upper()}_RETARGETING_CONFIG"
    env_config = os.environ.get(env_key)
    if env_config:
        return Path(env_config).expanduser().resolve()

    return (
        root / "example" / "config" / f"adaptive_analytical_wuji_glove_{hand_side}.yaml"
    ).resolve()


def _resolve_manus_retargeting_config(root: Path, hand_side: str) -> Path:
    env_key = f"MANUS_{hand_side.upper()}_RETARGETING_CONFIG"
    env_config = os.environ.get(env_key)
    if env_config:
        return Path(env_config).expanduser().resolve()

    return (root / "example" / "config" / f"retarget_manus_{hand_side}.yaml").resolve()


WUJI_RETARGETING_ROOT = _resolve_wuji_retargeting_root()
if str(WUJI_RETARGETING_ROOT) not in sys.path:
    sys.path.insert(0, str(WUJI_RETARGETING_ROOT))

try:
    from wuji_retargeting import Retargeter
except ImportError as exc:
    print(f"Warning: Wuji Retargeter not available from {WUJI_RETARGETING_ROOT}: {exc}")
    Retargeter = None

try:
    from gear_sonic.scripts.manus_provider import (
        DEFAULT_MANUS_CALIBRATION_FILE,
        DEFAULT_MANUS_INIT_TIMEOUT_S,
        DEFAULT_MANUS_LEFT_CALIBRATION_FILE,
        DEFAULT_MANUS_MCP_JOINT,
        DEFAULT_MANUS_RIGHT_CALIBRATION_FILE,
        DEFAULT_MANUS_STALE_S,
        ManusIntegratedProvider,
        VALID_MANUS_MCP_JOINTS,
    )
    MANUS_PROVIDER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    ManusIntegratedProvider = None
    MANUS_PROVIDER_IMPORT_ERROR = exc
    DEFAULT_MANUS_CALIBRATION_DIR = THIRD_PARTY_ROOT / "MANUS_SDK" / "calibrations"
    DEFAULT_MANUS_CALIBRATION_FILE = DEFAULT_MANUS_CALIBRATION_DIR / "Calibration.mcal"
    DEFAULT_MANUS_LEFT_CALIBRATION_FILE = DEFAULT_MANUS_CALIBRATION_DIR / "Calibration_left.mcal"
    DEFAULT_MANUS_RIGHT_CALIBRATION_FILE = DEFAULT_MANUS_CALIBRATION_DIR / "Calibration_right.mcal"
    DEFAULT_MANUS_STALE_S = 0.2
    DEFAULT_MANUS_INIT_TIMEOUT_S = 10.0
    DEFAULT_MANUS_MCP_JOINT = "proximal"
    VALID_MANUS_MCP_JOINTS = ("proximal", "metacarpal")

WUJI_RETARGETING_CONFIG = _resolve_wuji_retargeting_config(WUJI_RETARGETING_ROOT)
WUJI_GLOVE_LEFT_RETARGETING_CONFIG = _resolve_wuji_glove_retargeting_config(
    WUJI_RETARGETING_ROOT, "left"
)
WUJI_GLOVE_RIGHT_RETARGETING_CONFIG = _resolve_wuji_glove_retargeting_config(
    WUJI_RETARGETING_ROOT, "right"
)
MANUS_LEFT_RETARGETING_CONFIG = _resolve_manus_retargeting_config(WUJI_RETARGETING_ROOT, "left")
MANUS_RIGHT_RETARGETING_CONFIG = _resolve_manus_retargeting_config(WUJI_RETARGETING_ROOT, "right")
WUJI_QPOS_SIZE = 20
WUJI_QPOS_NONNEGATIVE_JOINT_INDICES = np.array(
    [
        finger_index * 4 + joint_index
        for finger_index in range(WUJI_QPOS_SIZE // 4)
        for joint_index in (0, 2, 3)
    ],
    dtype=np.int64,
)
HAND_CONTROL_MODE_GESTURE_WUJI = "gesture_wuji"
HAND_CONTROL_MODE_BINARY_TRIGGER = "binary_trigger"
HAND_CONTROL_MODE_WUJI_GLOVE = "wuji_glove"
HAND_CONTROL_MODE_MANUS = "manus"
VALID_HAND_CONTROL_MODES = (
    HAND_CONTROL_MODE_GESTURE_WUJI,
    HAND_CONTROL_MODE_BINARY_TRIGGER,
    HAND_CONTROL_MODE_WUJI_GLOVE,
    HAND_CONTROL_MODE_MANUS,
)
DEFAULT_WUJI_GLOVE_LEFT_ADDRESS = "192.168.123.100:50001"
DEFAULT_WUJI_GLOVE_RIGHT_ADDRESS = "192.168.123.101:50001"
DEFAULT_WUJI_GLOVE_STALE_S = 0.2
PLANNER_CONTROL_SOURCE_PICO = "pico"
PLANNER_CONTROL_SOURCE_KEYBOARD = "keyboard"
PLANNER_CONTROL_SOURCE_HYBRID = "hybrid"
VALID_PLANNER_CONTROL_SOURCES = (
    PLANNER_CONTROL_SOURCE_PICO,
    PLANNER_CONTROL_SOURCE_KEYBOARD,
    PLANNER_CONTROL_SOURCE_HYBRID,
)
PLANNER_KEYBOARD_HOLD_S = 0.25
PLANNER_KEYBOARD_MODE_DEBOUNCE_S = 0.25
PLANNER_KEYBOARD_ACTIVITY_KEYS = ("w", "a", "s", "d", "j", "l", "[", "]", " ")


def apply_wuji_qpos_limits(qpos: np.ndarray) -> np.ndarray:
    """Clip Wuji finger joints 0/2/3 to non-negative values, leaving joint 1 free."""
    limited = np.asarray(qpos, dtype=np.float32).reshape(-1).copy()
    if limited.shape[0] != WUJI_QPOS_SIZE:
        raise ValueError(f"Expected {WUJI_QPOS_SIZE} Wuji qpos values, got {limited.shape}")
    limited[WUJI_QPOS_NONNEGATIVE_JOINT_INDICES] = np.maximum(
        limited[WUJI_QPOS_NONNEGATIVE_JOINT_INDICES],
        0.0,
    )
    return limited


def _build_binary_trigger_closed_wuji_qpos() -> np.ndarray:
    qpos = np.zeros((WUJI_QPOS_SIZE,), dtype=np.float32)
    for finger_index in range(1, 5):
        offset = finger_index * 4
        qpos[offset : offset + 4] = np.array([0.8, 0.0, 0.8, 0.8], dtype=np.float32)
    return qpos


WUJI_OPEN_QPOS = np.zeros((WUJI_QPOS_SIZE,), dtype=np.float32)
WUJI_BINARY_TRIGGER_CLOSED_QPOS = _build_binary_trigger_closed_wuji_qpos()
    
class TerminalKeyReader:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)   # 进入 cbreak 模式，按键无需回车
        atexit.register(self.restore)
        self.pressed_keys = set()  # 跟踪当前轮询周期内读到的键
        self.last_pressed_at = {}

    @staticmethod
    def _normalize_key(key):
        if isinstance(key, str) and len(key) == 1:
            return key.lower()
        return key

    def restore(self):
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        except Exception:
            pass

    def update_keys(self):
        """更新按键状态，读取所有可用字符"""
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if not rlist:
                break
            ch = sys.stdin.read(1)
            if ch:
                key = self._normalize_key(ch)
                self.pressed_keys.add(key)
                self.last_pressed_at[key] = time.monotonic()
            else:
                break

    def is_pressed(self, key):
        """检查当前轮询周期内是否读到了按键"""
        return self._normalize_key(key) in self.pressed_keys

    def is_recently_pressed(self, key, hold_s=PLANNER_KEYBOARD_HOLD_S):
        """Return True if key was pressed recently enough to be treated as held."""
        key = self._normalize_key(key)
        last_pressed = self.last_pressed_at.get(key)
        if last_pressed is None:
            return False
        return (time.monotonic() - last_pressed) <= hold_s

    def has_recent_press(self, keys, hold_s=PLANNER_KEYBOARD_HOLD_S):
        return any(self.is_recently_pressed(key, hold_s=hold_s) for key in keys)

    def consume_keys(self, *keys):
        """Remove one-shot keys from the current polling interval."""
        for key in keys:
            self.pressed_keys.discard(self._normalize_key(key))

    def clear_keys(self):
        """清除当前轮询周期按键；last_pressed_at 用于短暂 hold，不清除。"""
        self.pressed_keys.clear()


class LocomotionMode(IntEnum):
    """Locomotion mode enum for robot movement."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19


class StreamMode(Enum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


DEFAULT_PLANNER_LOCOMOTION_MODE = LocomotionMode.SLOW_WALK
DEFAULT_PLANNER_TRANSLATION_SPEED = 0.3
DEFAULT_PLANNER_TURN_YAW_GAIN = 0.8


### Parse 3 point pose from SMPL
#
# OFFSETS: Rotation corrections applied to each keypoint to align SMPL joint frames
# with the desired robot/visualization coordinate convention.
#
# Index mapping (based on [0, 22, 23, 12].index(joint_id)):
#   - OFFSETS[0]: Root/Pelvis (joint 0)
#   - OFFSETS[1]: Left Wrist (joint 22)
#   - OFFSETS[2]: Right Wrist (joint 23)
#   - OFFSETS[3]: Neck (joint 12) - more stable than Head (joint 15) for body tracking
#
# Scipy euler rotation convention:
#   - Lowercase "xyz" = EXTRINSIC rotations (about the FIXED/ORIGINAL frame's axes)
#   - Uppercase "XYZ" = INTRINSIC rotations (about the ROTATING body's axes)
#
# For EXTRINSIC "xyz" with angles [a, b, c]:
#   All rotations are about the ORIGINAL frame's axes (before any rotation):
#     R_total = R_z(c) @ R_y(b) @ R_x(a)   (matrix multiplication order)
#   Applied as: first rotate 'a' about original X, then 'b' about original Y, then 'c' about original Z
#
# For INTRINSIC "XYZ" with angles [a, b, c]:
#   Each rotation is about the CURRENT (rotated) frame's axis:
#     R_total = R_x(a) @ R_y(b) @ R_z(c)   (matrix multiplication order)
#   Applied as: first rotate 'a' about X, then 'b' about NEW Y, then 'c' about NEW Z
#
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root: yaw -90° about fixed Z
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist: roll +90° about fixed X
    sRot.from_euler(
        "xyz", [-90, 0, 180], degrees=True
    ),  # R-Wrist: roll -90° about fixed X, then yaw 180° about fixed Z
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck: yaw -90° about fixed Z
]

# Pico/OpenXR 26 joints (with palm) -> MediaPipe 21 landmarks.
# Joint layout follows the OpenXR hand tracking spec:
#   0: Palm, 1: Wrist
#   2-5: Thumb   (metacarpal, proximal, distal, tip)
#   6-10: Index  (metacarpal, proximal, intermediate, distal, tip)
#   11-15: Middle
#   16-20: Ring
#   21-25: Little
# MediaPipe does not include palm/finger metacarpals, so we skip those joints.
PICO_TO_MEDIAPIPE = (
    1,
    2, 3, 4, 5,
    7, 8, 9, 10,
    12, 13, 14, 15,
    17, 18, 19, 20,
    22, 23, 24, 25,
)


def convert_pico_to_wuji_keypoints(fingers_mat: np.ndarray) -> np.ndarray:
    """Convert Pico/OpenXR hand joints to Wuji retargeting keypoints (21, 3)."""
    fingers_mat = np.asarray(fingers_mat)
    if fingers_mat.shape[0] != 26:
        raise ValueError(f"Expected 26 hand joints, got shape {fingers_mat.shape}")

    mediapipe_pose = np.zeros((21, 3), dtype=np.float32)
    if fingers_mat.ndim == 3 and fingers_mat.shape[1:] == (4, 4):
        for mp_idx, pico_idx in enumerate(PICO_TO_MEDIAPIPE):
            mediapipe_pose[mp_idx] = fingers_mat[pico_idx][:3, 3].astype(np.float32)
    elif fingers_mat.ndim == 2 and fingers_mat.shape[1] >= 3:
        for mp_idx, pico_idx in enumerate(PICO_TO_MEDIAPIPE):
            mediapipe_pose[mp_idx] = fingers_mat[pico_idx][:3].astype(np.float32)
    else:
        raise ValueError(f"Unsupported fingers_mat shape: {fingers_mat.shape}")

    return mediapipe_pose


def _compute_rel_transform(pose, world_frame, scalar_first=True):
    """
    Transform a pose from Unity coordinate frame to robot coordinate frame.

    Args:
        pose: np.ndarray shape (7,) - [x, y, z, qx, qy, qz, qw] in Unity frame
        world_frame: np.ndarray shape (7,) - reference frame to compute relative transform
        scalar_first: bool - if True, quaternion is [qw, qx, qy, qz]; if False, [qx, qy, qz, qw]

    Returns:
        rel_pos: np.ndarray (3,) - position in robot frame
        rel_rot: np.ndarray (4,) - quaternion [qw, qx, qy, qz] in robot frame

    Coordinate transform matrix Q converts Unity (Y-up, left-handed) to Robot (Z-up, right-handed):
        Unity:  X-right, Y-up, Z-forward
        Robot:  X-forward, Y-left, Z-up
    """
    world_frame = world_frame.copy()

    # Q transforms Unity coordinates to Robot coordinates
    # Unity [x, y, z] -> Robot [-x, z, y]
    Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
    pose[:3] = Q @ pose[:3]
    world_frame[:3] = Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


def _process_3pt_pose(smpl_pose_np):
    """
    Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full SMPL body joint poses.

    NOTE: We use Neck (joint 12) instead of Head (joint 15) because:
      - Neck is more rigidly coupled to the torso
      - Head has high DoF (looking around) which doesn't reflect body pose
      - Neck provides more stable tracking for upper body orientation

    Args:
        smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints, each [x, y, z, qx, qy, qz, qw]
                      in Unity frame (scalar-last quaternion format)

    Returns:
        vr_3pt_pose: np.ndarray shape (3, 7) - 3 keypoints in robot frame
                     Each row is [x, y, z, qw, qx, qy, qz] (scalar-FIRST quaternion format)
                     Row 0: Left Wrist (SMPL joint 22)
                     Row 1: Right Wrist (SMPL joint 23)
                     Row 2: Neck (SMPL joint 12)

                     IMPORTANT: Positions and orientations are RELATIVE TO ROOT (pelvis).

    Processing Steps:
        1. Transform all 24 joints from Unity frame to robot frame
        2. Extract 4 keypoints: Root(0), L-Wrist(22), R-Wrist(23), Neck(12)
        3. Apply per-joint rotation OFFSETS to align joint frames
        4. Make L-Wrist, R-Wrist, Neck relative to Root (both position and orientation)
        5. Return only the 3 non-root keypoints

    Note: Position calibration (wrist offsets, neck kinematic chain) is done in
          ThreePointPose.apply_calibration() to ensure consistency with calibrated
          orientations.
    """

    # Defensive copy: _compute_rel_transform modifies pose[:3] in-place, which would
    # corrupt the caller's array (e.g. PicoReader._latest) and cause wrong results
    # if the same sample is processed more than once.
    smpl_pose_np = smpl_pose_np.copy()

    # =========================================================================
    # STEP 1: Transform all joints from Unity frame to robot frame
    # =========================================================================
    # Input: smpl_pose_np[i] = [x, y, z, qx, qy, qz, qw] in Unity frame (scalar-last)
    # Output: body_poses[i] = [x, y, z, qw, qx, qy, qz] in robot frame (scalar-first)
    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos  # Position in robot frame
        body_poses[i, 3:] = orn  # Quaternion [qw, qx, qy, qz] in robot frame

    # =========================================================================
    # STEP 2 & 3: Extract 4 keypoints and apply rotation OFFSETS
    # =========================================================================
    # We only care about these SMPL joint indices:
    #   - Joint 0:  Root/Pelvis (reference frame)
    #   - Joint 22: Left Wrist
    #   - Joint 23: Right Wrist
    #   - Joint 12: Neck (more stable than Head joint 15)
    #
    # kp_poses maps these to indices 0, 1, 2, 3 respectively
    positions = np.array([[p[0], p[1], p[2]] for p in body_poses])
    kp_poses = np.zeros((4, 7), dtype=np.float32)

    for i, pose in enumerate(body_poses):
        if i not in [0, 22, 23, 12]:
            continue  # Skip joints we don't care about

        pos = positions[i]

        # Map SMPL joint index to our keypoint index (0-3)
        # rel_i: 0=Root, 1=L-Wrist, 2=R-Wrist, 3=Neck
        rel_i = [0, 22, 23, 12].index(i)

        # Extract quaternion and apply rotation offset
        # pose[3:7] is [qw, qx, qy, qz] (scalar-first from _compute_rel_transform)
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])

        # Apply offset: new_rotation = original_rotation * OFFSET
        # This post-multiplies the offset (intrinsic rotation)
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )

        kp_poses[rel_i, 3:] = rot_quat  # Store as scalar-last temporarily for scipy compatibility
        kp_poses[rel_i, :3] = pos

    # =========================================================================
    # STEP 4: Make positions and orientations RELATIVE TO ROOT
    # =========================================================================
    # This transforms everything into the root's local coordinate frame.
    # After this step:
    #   - Root's position would be (0,0,0) and orientation identity (but we don't return root)
    #   - Other keypoints are expressed relative to root
    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()  # Still scalar-last for scipy

    for i in range(1, 4):
        # Position: subtract root position, then rotate by inverse of root orientation
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)

        # Orientation: compute relative rotation (root_inv * keypoint_rot)
        # Result stored as scalar-FIRST [qw, qx, qy, qz]
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    # =========================================================================
    # STEP 5: Return only L-Wrist, R-Wrist, Neck (skip Root)
    # =========================================================================
    # NOTE: Position and orientation calibration (including neck position via kinematic
    #       chain) is done in ThreePointPose.apply_calibration() to ensure consistency
    #       between calibrated orientation and computed neck position.
    # kp_poses[1:] = indices 1, 2, 3 = L-Wrist, R-Wrist, Neck
    # Each row: [x, y, z, qw, qx, qy, qz] relative to root, scalar-first quaternion
    return kp_poses[1:]


# =============================================================================
# VR 3-Point Pose Visualization Functions
# =============================================================================


def run_vr3pt_visualizer_test():
    """
    Standalone test for VR 3-point pose visualizer using PyVista.
    Run this to verify the reference frames are displayed correctly.
    """
    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Visualizer Test (PyVista)")
    print("=" * 60)
    print("\nExpected reference frames (all with RGB axes for XYZ):")
    print("  1. WHITE ball at origin (0, 0, 0) - World frame")
    print("  2. CYAN ball at (0, 0, 0.35) - Looking forward (identity)")
    print("  3. MAGENTA ball at (0, 0.4, 0.25) - Looking left (yaw +90°)")
    print("  4. YELLOW ball at (0.4, 0, 0.15) - Looking down (pitch +90°)")
    print("\nClose the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_static()


def run_vr3pt_live_visualizer():
    """
    Live visualizer for real VR 3-point pose data from Pico.
    Captures one frame from Pico and displays it alongside reference frames.
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use live visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Live Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Capturing VR 3-point pose...")

    # Capture body poses and compute vr_3pt_pose
    body_poses = xrt.get_body_joints_pose()
    body_poses_np = np.array(body_poses)

    # Process to get 3-point pose (L-Wrist, R-Wrist, Neck)
    vr_3pt_pose = _process_3pt_pose(body_poses_np)

    print(f"\nCaptured vr_3pt_pose shape: {vr_3pt_pose.shape}")
    print(f"  L-Wrist: pos={vr_3pt_pose[0, :3]}, quat_wxyz={vr_3pt_pose[0, 3:]}")
    print(f"  R-Wrist: pos={vr_3pt_pose[1, :3]}, quat_wxyz={vr_3pt_pose[1, 3:]}")
    print(f"  Neck:    pos={vr_3pt_pose[2, :3]}, quat_wxyz={vr_3pt_pose[2, 3:]}")

    print("\nDisplaying visualization...")
    print("Close the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_with_vr_pose(vr_3pt_pose)


def run_vr3pt_realtime_visualizer(update_hz: int = 10):
    """
    Real-time visualizer for VR 3-point pose data from Pico.
    Continuously updates the visualization with live data.

    Args:
        update_hz: Update rate in Hz (default 10)
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use realtime visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Real-time Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Starting real-time visualization...")
    print(f"Update rate: {update_hz} Hz")
    print("Close the window or press 'q' to exit.")
    print("=" * 60)

    # Use the VR3PtPoseVisualizer for real-time visualization with G1 robot
    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.create_realtime_plotter(interactive=True)

    try:
        while visualizer.is_open:
            # Get new data from Pico
            body_poses = xrt.get_body_joints_pose()
            body_poses_np = np.array(body_poses)
            vr_3pt_pose = _process_3pt_pose(body_poses_np)

            # Update visualization
            visualizer.update_vr_poses(vr_3pt_pose)
            visualizer.render()

            time.sleep(1.0 / update_hz)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()


def process_smpl_joints(body_pose, global_orient, transl):
    """Process SMPL parameters to compute local joints.

    Args:
        body_pose: Body pose tensor, shape (T, 69)
        global_orient: Global orientation tensor, shape (T, 3)
        transl: Translation tensor, shape (T, 3)

    Returns:
        Dictionary with processed joints and parameters
    """
    # Convert global_orient to quaternion and apply transformations (robust if utils missing)
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    if smpl_root_ytoz_up is not None:
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    # Compute joints and vertices using SMPL model (single forward pass)
    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
    )  # (*, 24, 3)

    # Apply base rotation removal and compute local joints
    if remove_smpl_base_rot is not None:
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)
    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)
    global_orient_6d = global_orient_mat[..., :2].reshape(1, 6)

    return {
        "smpl_pose": body_pose,
        "joints": joints,
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
        "global_orient_6d": global_orient_6d,
        "adjusted_transl": transl,
    }

# Joystick deadzone threshold
JOYSTICK_DEADZONE = 0.15


class YawAccumulator:
    """Accumulates yaw heading angle based on joystick input."""

    def __init__(self, yaw_gain: float = 1.5, deadzone: float = JOYSTICK_DEADZONE):
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()

    def reset(self):
        """Reset facing direction to default (1,0,0)."""
        self.heading = [1.0, 0.0, 0.0]
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0
        print("YawAccumulator: reset yaw angle to 0.0")

    def yaw_angle(self) -> float:
        """Get current yaw angle in radians."""
        return self.yaw_angle_rad

    def yaw_angle_change(self) -> float:
        """Get current yaw angle change in radians."""
        return self.dyaw

    def update(self, rx: float, dt: float) -> list[float]:
        """
        Update facing direction based on right stick x-axis input.

        Args:
            rx: Right stick x-axis value (-1 to 1)
            dt: Time delta in seconds

        Returns:
            Facing direction as [x, y, 0.0]
        """
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = [np.cos(self.yaw_angle_rad), np.sin(self.yaw_angle_rad), 0.0]
        return self.heading


def compute_from_body_poses(parent_indices: list, device, body_poses_np: np.ndarray):
    """
    Compute local joints and body orientation from provided body_poses_np.
    """
    positions = body_poses_np[:, :3]
    global_quats = body_poses_np[:, [6, 3, 4, 5]]

    # Convert to local rotations
    global_rots = sRot.from_quat(global_quats, scalar_first=True)
    global_rots = global_rots * sRot.from_euler("y", 180, degrees=True)

    local_rots = []
    for i in range(24):
        if parent_indices[i] == -1:
            local_rots.append(global_rots[i])
        else:
            local_rot = global_rots[parent_indices[i]].inv() * global_rots[i]
            local_rots.append(local_rot)

    pose_aa = np.array([rot.as_rotvec() for rot in local_rots])

    body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(device).unsqueeze(0)
    global_orient = torch.from_numpy(pose_aa[0]).float().to(device).unsqueeze(0)
    transl = torch.from_numpy(positions[0]).float().to(device).unsqueeze(0)

    return process_smpl_joints(body_pose, global_orient, transl)


# def compute_latest_frame(parent_indices: list, device) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Pull body data from XRoboToolkit, compute local SMPL joints and body orientation.
#     Returns (smpl_joints_local_np [24,3], global_orient_quat_np [4,])
#     """
#     body_poses = xrt.get_body_joints_pose()
#     body_poses_np = np.array(body_poses)
#     return compute_from_body_poses(parent_indices, device, body_poses_np)

def init_wuji_retargeters(
    left_config_path: str | Path | None = None,
    right_config_path: str | Path | None = None,
):
    """Initialize left/right Wuji retargeters if dependencies are available."""
    if Retargeter is None:
        return None, None

    left_config = (
        Path(left_config_path).expanduser().resolve()
        if left_config_path
        else WUJI_RETARGETING_CONFIG
    )
    right_config = (
        Path(right_config_path).expanduser().resolve()
        if right_config_path
        else left_config
    )
    if not left_config.exists():
        print(f"Warning: left Wuji retargeting config not found: {left_config}")
        return None, None
    if not right_config.exists():
        print(f"Warning: right Wuji retargeting config not found: {right_config}")
        return None, None

    try:
        left_retargeter = Retargeter.from_yaml(str(left_config), "left")
        right_retargeter = Retargeter.from_yaml(str(right_config), "right")
        if left_config == right_config:
            print(f"Wuji retargeters initialized from {left_config}")
        else:
            print(
                "Wuji retargeters initialized from "
                f"left={left_config}, right={right_config}"
            )
        return left_retargeter, right_retargeter
    except Exception as exc:
        print(f"Warning: failed to initialize Wuji retargeters: {exc}")
        return None, None


def get_controller_inputs():
    """Fetch controller button/trigger states from XRoboToolkit."""
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_menu_button = xrt.get_left_menu_button()
    return left_menu_button, left_trigger, right_trigger


def get_controller_axes():
    """Fetch joystick axes (lx, ly, rx, ry). Falls back to zeros if not available."""
    if xrt is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        left_axis = xrt.get_left_axis()  # expected [x, y]
        right_axis = xrt.get_right_axis()  # expected [x, y]
        lx = float(left_axis[0]) if len(left_axis) >= 1 else 0.0
        ly = float(left_axis[1]) if len(left_axis) >= 2 else 0.0
        rx = float(right_axis[0]) if len(right_axis) >= 1 else 0.0
        ry = float(right_axis[1]) if len(right_axis) >= 2 else 0.0
        return lx, ly, rx, ry
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def get_menu_buttons():
    """Fetch both menu buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_menu_button")
    right = _safe_btn("get_right_menu_button")
    return left, right


def get_axis_clicks():
    """Fetch both axis click buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_axis_click")
    right = _safe_btn("get_right_axis_click")
    return left, right


def get_face_buttons():
    """Fetch primary face buttons A and X. Returns (a_pressed, x_pressed)."""
    if xrt is None:
        return False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        x_pressed = bool(xrt.get_X_button())
        return a_pressed, x_pressed
    except Exception:
        return False, False


def get_abxy_buttons():
    """Fetch A,B,X,Y face buttons as booleans (a,b,x,y)."""
    if xrt is None:
        return False, False, False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        b_pressed = bool(xrt.get_B_button())
        x_pressed = bool(xrt.get_X_button())
        y_pressed = bool(xrt.get_Y_button())
        return a_pressed, b_pressed, x_pressed, y_pressed
    except Exception:
        return False, False, False, False

def compute_hand_joints_from_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Fetch the current tracked left/right hand states as numpy arrays."""
    left_hand_joints = xrt.get_left_hand_tracking_state()
    right_hand_joints = xrt.get_right_hand_tracking_state()
        
    # 确保返回numpy arrays，如果不是则转换
    if not isinstance(left_hand_joints, np.ndarray):
        left_hand_joints = np.array(left_hand_joints, dtype=np.float32)
    if not isinstance(right_hand_joints, np.ndarray):
        right_hand_joints = np.array(right_hand_joints, dtype=np.float32)

    return left_hand_joints, right_hand_joints


def _get_active_hand_tracking_state(is_left: bool):
    """Fetch the current active hand tracking state from Pico/XRT."""
    if xrt is None:
        return False, None

    active_getter = xrt.get_left_hand_is_active if is_left else xrt.get_right_hand_is_active
    state_getter = xrt.get_left_hand_tracking_state if is_left else xrt.get_right_hand_tracking_state

    try:
        is_active = bool(active_getter())
    except Exception:
        return False, None

    if not is_active:
        return False, None

    try:
        hand_state = state_getter()
    except Exception:
        return False, None

    if hand_state is None:
        return False, None

    hand_state = np.asarray(hand_state, dtype=np.float32)
    return True, hand_state


def compute_wuji_keypoints_from_inputs(
    last_left_wuji_keypoints=None,
    last_right_wuji_keypoints=None,
) -> tuple[np.ndarray, bool, np.ndarray, bool]:
    """Read Pico hand tracking and return Wuji retargeting keypoints with cache fallback."""
    empty = np.zeros((21, 3), dtype=np.float32)

    left_wuji_keypoints = (
        last_left_wuji_keypoints.copy()
        if last_left_wuji_keypoints is not None
        else empty.copy()
    )
    right_wuji_keypoints = (
        last_right_wuji_keypoints.copy()
        if last_right_wuji_keypoints is not None
        else empty.copy()
    )

    left_tracked = False
    right_tracked = False

    left_active, left_hand_state = _get_active_hand_tracking_state(is_left=True)
    right_active, right_hand_state = _get_active_hand_tracking_state(is_left=False)

    if left_active and left_hand_state is not None:
        try:
            left_wuji_keypoints = convert_pico_to_wuji_keypoints(left_hand_state)
            left_tracked = True
        except ValueError as exc:
            print(f"[WujiKeypoints] Failed to convert left hand state: {exc}")

    if right_active and right_hand_state is not None:
        try:
            right_wuji_keypoints = convert_pico_to_wuji_keypoints(right_hand_state)
            right_tracked = True
        except ValueError as exc:
            print(f"[WujiKeypoints] Failed to convert right hand state: {exc}")

    return left_wuji_keypoints, left_tracked, right_wuji_keypoints, right_tracked


def wuji_glove_skeleton_to_keypoints(skeleton) -> np.ndarray:
    """Convert a Wuji SDK HandSkeleton frame into 21x3 MediaPipe-style keypoints."""
    try:
        keypoints = np.array(
            [joint.pose.position for joint in skeleton.joints],
            dtype=np.float32,
        )
    except AttributeError as exc:
        raise ValueError("Wuji glove skeleton is missing joints/pose.position fields") from exc
    if keypoints.shape != (21, 3):
        raise ValueError(f"Expected Wuji glove skeleton shape (21, 3), got {keypoints.shape}")
    if not np.all(np.isfinite(keypoints)):
        raise ValueError("Wuji glove skeleton contains non-finite keypoints")
    return keypoints


class WujiGloveSkeletonProvider:
    """Non-blocking reader for two Wuji Glove hand_skeleton() streams."""

    def __init__(
        self,
        *,
        left_address: str,
        right_address: str,
        stale_s: float = DEFAULT_WUJI_GLOVE_STALE_S,
        connect_timeout_ms: int = 1000,
        connect_retry_count: int = 3,
    ):
        if stale_s <= 0.0:
            raise ValueError("wuji_glove_stale_s must be > 0")
        self.stale_s = float(stale_s)
        self._latest: dict[str, dict[str, object] | None] = {"left": None, "right": None}
        self._subscriptions = {}
        self._device_names = {}
        self._device_infos: dict[str, dict[str, str]] = {}
        self._last_error_log_at: dict[str, float] = {"left": 0.0, "right": 0.0}

        try:
            from wuji_sdk import ConnectOptions, SdkManager
        except ImportError as exc:
            raise RuntimeError(
                "wuji_glove mode requires wuji-sdk. Install it in the teleop env, "
                "for example: uv pip install --python .venv_teleop/bin/python wuji-sdk"
            ) from exc

        self._manager = SdkManager.instance()
        self._connect_options = ConnectOptions(
            timeout_ms=int(connect_timeout_ms),
            retry_count=int(connect_retry_count),
        )
        try:
            self._connect_side("left", left_address)
            self._connect_side("right", right_address)
        except Exception:
            self.close()
            raise

    @classmethod
    def from_subscriptions(
        cls,
        *,
        left_subscription,
        right_subscription,
        stale_s: float = DEFAULT_WUJI_GLOVE_STALE_S,
    ) -> "WujiGloveSkeletonProvider":
        provider = cls.__new__(cls)
        provider.stale_s = float(stale_s)
        provider._manager = None
        provider._latest = {"left": None, "right": None}
        provider._subscriptions = {
            "left": left_subscription,
            "right": right_subscription,
        }
        provider._device_names = {}
        provider._device_infos = {}
        provider._last_error_log_at = {"left": 0.0, "right": 0.0}
        return provider

    @property
    def device_infos(self) -> dict[str, dict[str, str]]:
        return {side: dict(info) for side, info in self._device_infos.items()}

    @staticmethod
    def _device_serial_number(device) -> str:
        for attr_name in ("serial_number", "sn"):
            try:
                value = getattr(device, attr_name)
            except Exception:
                continue
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            if value:
                return str(value)
        return "<unknown>"

    def _connect_side(self, side: str, address: str) -> None:
        address = str(address or "").strip()
        if not address:
            raise RuntimeError(f"Missing Wuji glove {side} address")
        device_name = f"{side}_wuji_glove"
        print(f"[WujiGlove] Connecting {side} glove at {address}")
        try:
            device = self._manager.connect(
                address=address,
                device_name=device_name,
                options=self._connect_options,
            )
            hand_side = "unknown"
            try:
                hand_side = device.hand_side().get()
            except Exception:
                pass
            serial_number = self._device_serial_number(device)
            print(
                f"[WujiGlove] Connected {side}: "
                f"sn={serial_number} hand_side={hand_side}"
            )
            self._subscriptions[side] = device.hand_skeleton().subscribe()
            self._device_names[side] = device_name
            self._device_infos[side] = {
                "address": address,
                "serial_number": serial_number,
                "hand_side": str(hand_side),
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to connect {side} Wuji glove at {address}: {exc}") from exc

    def close(self) -> None:
        for sub in list(self._subscriptions.values()):
            try:
                sub.close()
            except Exception:
                pass
        if self._manager is not None:
            for device_name in list(self._device_names.values()):
                try:
                    self._manager.disconnect(device_name)
                except Exception:
                    pass

    def latest_keypoints(
        self,
        last_left_keypoints: np.ndarray | None = None,
        last_right_keypoints: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, np.ndarray, bool]:
        self.poll()
        now = time.monotonic()
        left_keypoints, left_valid = self._side_keypoints("left", last_left_keypoints, now)
        right_keypoints, right_valid = self._side_keypoints("right", last_right_keypoints, now)
        return left_keypoints, left_valid, right_keypoints, right_valid

    def poll(self) -> None:
        for side, sub in self._subscriptions.items():
            self._poll_side(side, sub)

    def _poll_side(self, side: str, sub) -> None:
        try:
            skeleton = sub.recv()
            if skeleton is None:
                return
            while True:
                newer = sub.recv()
                if newer is None:
                    break
                skeleton = newer
            keypoints = wuji_glove_skeleton_to_keypoints(skeleton)
            self._latest[side] = {
                "keypoints": keypoints,
                "received_at": time.monotonic(),
            }
        except Exception as exc:
            self._log_error(side, exc)

    def _side_keypoints(
        self,
        side: str,
        fallback: np.ndarray | None,
        now: float,
    ) -> tuple[np.ndarray, bool]:
        record = self._latest.get(side)
        if record is not None:
            age_s = now - float(record["received_at"])
            if age_s <= self.stale_s:
                return np.asarray(record["keypoints"], dtype=np.float32).copy(), True

        if fallback is not None:
            return np.asarray(fallback, dtype=np.float32).copy(), False
        return np.zeros((21, 3), dtype=np.float32), False

    def _log_error(self, side: str, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_error_log_at.get(side, 0.0) >= 2.0:
            print(f"[WujiGlove] Failed to read {side} skeleton: {exc}")
            self._last_error_log_at[side] = now


def compute_wuji_qpos_from_keypoints(
    left_wuji_keypoints: np.ndarray,
    right_wuji_keypoints: np.ndarray,
    left_retargeter,
    right_retargeter,
) -> tuple[np.ndarray, bool, np.ndarray, bool]:
    """Retarget Wuji input keypoints into WujiHand joint targets."""
    left_wuji_qpos = np.zeros((WUJI_QPOS_SIZE,), dtype=np.float32)
    right_wuji_qpos = np.zeros((WUJI_QPOS_SIZE,), dtype=np.float32)
    left_wuji_qpos_valid = False
    right_wuji_qpos_valid = False

    if left_retargeter is not None and not np.allclose(left_wuji_keypoints, 0.0):
        try:
            left_wuji_qpos = np.asarray(
                left_retargeter.retarget(left_wuji_keypoints), dtype=np.float32
            ).reshape(-1)
            if left_wuji_qpos.shape[0] != WUJI_QPOS_SIZE:
                raise ValueError(
                    f"Expected {WUJI_QPOS_SIZE} left qpos values, got {left_wuji_qpos.shape}"
                )
            left_wuji_qpos_valid = True
        except Exception as exc:
            print(f"[WujiRetarget] Failed to retarget left hand: {exc}")

    if right_retargeter is not None and not np.allclose(right_wuji_keypoints, 0.0):
        try:
            right_wuji_qpos = np.asarray(
                right_retargeter.retarget(right_wuji_keypoints), dtype=np.float32
            ).reshape(-1)
            if right_wuji_qpos.shape[0] != WUJI_QPOS_SIZE:
                raise ValueError(
                    f"Expected {WUJI_QPOS_SIZE} right qpos values, got {right_wuji_qpos.shape}"
                )
            right_wuji_qpos_valid = True
        except Exception as exc:
            print(f"[WujiRetarget] Failed to retarget right hand: {exc}")

    return (
        left_wuji_qpos,
        left_wuji_qpos_valid,
        right_wuji_qpos,
        right_wuji_qpos_valid,
    )


def compute_binary_trigger_wuji_qpos(
    left_trigger: float,
    right_trigger: float,
    threshold: float = 0.5,
) -> tuple[np.ndarray, bool, bool, np.ndarray, bool, bool]:
    left_closed = bool(left_trigger > threshold)
    right_closed = bool(right_trigger > threshold)
    left_qpos = (
        WUJI_BINARY_TRIGGER_CLOSED_QPOS.copy() if left_closed else WUJI_OPEN_QPOS.copy()
    )
    right_qpos = (
        WUJI_BINARY_TRIGGER_CLOSED_QPOS.copy() if right_closed else WUJI_OPEN_QPOS.copy()
    )
    return left_qpos, True, left_closed, right_qpos, True, right_closed


def _quat_lerp_normalized(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linear interpolate two quaternions and renormalize. Input shape (4,), xyzw order.
    Ensures shortest path by flipping sign if dot < 0.
    """
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q


def _interp_pose_axis_angle(
    prev_pose: np.ndarray, curr_pose: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Interpolate axis-angle joint poses by converting to quats, lerp-normalize, then back.
    prev_pose, curr_pose: (21,3) axis-angle (rotvec)
    Returns (21,3) axis-angle.
    """
    prev_quats = sRot.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()  # (N,4) xyzw
    curr_quats = sRot.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for i in range(prev_quats.shape[0]):
        out_quats[i] = _quat_lerp_normalized(prev_quats[i], curr_quats[i], alpha)
    out_pose = sRot.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape)
    return out_pose


class PicoReader:
    """
    Background reader that pulls Pico/XRT data as fast as possible and computes dt/FPS.
    """

    def __init__(self, max_queue_size: int = 15):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_t = None
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if not xrt.is_body_data_available():
                time.sleep(0.001)
                continue
            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.000001)
                continue
            # Compute device-based dt/fps using timestamp deltas (ns -> s)
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns
            t_realtime = time.time()
            t_monotonic = time.monotonic()
            try:
                body_poses = xrt.get_body_joints_pose()

                sample = {
                    "body_poses_np": np.array(body_poses),
                    "timestamp_realtime": t_realtime,
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample
                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] dt_ts: {device_dt*1000.0:.2f} ms, fps: {self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as e:
                print(f"[PicoReader] read error: {e}")


def _pose_stream_common(
    socket,
    buffer_size: int,
    num_frames_to_send: int,
    target_fps: int,
    use_cuda: bool,
    record_dir: str,
    record_format: str,
    stop_event: threading.Event | None = None,
    log_prefix: str = "PoseLoop",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """Shared pose streaming loop used by run_pico."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run pose streaming."
        )

    # Create reader and start it
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()

    # Create 3-point pose processor with visualization settings
    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix=log_prefix,
    )

    streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix=log_prefix,
    )

    if stop_event is None:
        stop_event = threading.Event()

    try:
        while not stop_event.is_set():
            streamer.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup resources
        reader.stop()
        streamer.close()
        three_point.close()


class ThreePointPose:
    """
    Encapsulates everything around calculating 3-point pose from SMPL input.

    This includes:
    - Processing SMPL poses to extract 3-point VR pose (L-Wrist, R-Wrist, Neck)
    - Calibration logic to align VR poses with G1 robot
    - Optional visualization of 3-point poses

    Calibration is done in two steps:
    1. Neck orientation: Captures initial neck orientation to align subsequent poses as upright
    2. Wrist positions: Aligns wrist positions to match G1 robot key frame positions
    """

    # Kinematic chain constants for neck position (matches VR3PtPoseVisualizer)
    TORSO_LINK_OFFSET_Z = 0.05  # meters from root to torso_link
    NECK_LINK_LENGTH = 0.35  # meters from torso_link to neck along neck's local Z

    def __init__(
        self,
        enable_vis_vr3pt: bool = False,
        with_g1_robot: bool = True,
        enable_waist_tracking: bool = False,
        enable_smpl_vis: bool = False,
        log_prefix: str = "ThreePointPose",
        robot_model=None,
    ):
        """
        Initialize 3-point pose processor.

        Args:
            enable_vis_vr3pt: Whether to enable VR 3pt pose visualization (requires display)
            with_g1_robot: Whether to include G1 robot in visualization
            enable_waist_tracking: Whether to enable waist tracking in visualization
            enable_smpl_vis: Whether to render SMPL body joints in the VR3pt visualizer
            log_prefix: Prefix for log messages
            robot_model: Optional pre-instantiated RobotModel. If None, will create one.
                        Used for FK-based calibration (no display required).
        """
        self.log_prefix = log_prefix
        self.with_g1_robot = with_g1_robot
        self.enable_waist_tracking = enable_waist_tracking
        self.enable_smpl_vis = enable_smpl_vis

        # Robot model for FK-based calibration (headless, no display required)
        self._robot_model = robot_model
        if self._robot_model is None:
            from gear_sonic.data.robot_model.instantiation.g1 import (
                instantiate_g1_robot_model,
            )

            self._robot_model = instantiate_g1_robot_model()
            print(f"[{log_prefix}] Robot model loaded for FK calibration")

        # Optional visualization (requires display + PyVista)
        self.vr3pt_visualizer = None
        if enable_vis_vr3pt:
            if VR3PtPoseVisualizer is None:
                raise ImportError(
                    "VR3PtPoseVisualizer could not be imported but --vis_vr3pt was requested. "
                    "Ensure pyvista is installed: pip install pyvista"
                )
            self.vr3pt_visualizer = VR3PtPoseVisualizer(
                axis_length=0.08,
                ball_radius=0.015,
                with_g1_robot=with_g1_robot,
                robot_model=self._robot_model,
                enable_waist_tracking=enable_waist_tracking,
                enable_smpl_vis=enable_smpl_vis,
            )
            self.vr3pt_visualizer.create_realtime_plotter(interactive=True)
            g1_str = " with G1 robot" if with_g1_robot else ""
            waist_str = " + waist tracking" if enable_waist_tracking else ""
            smpl_str = " + SMPL body" if enable_smpl_vis else ""
            print(f"[{log_prefix}] VR 3pt pose visualization enabled{g1_str}{waist_str}{smpl_str}")

        # Calibration state — triggered explicitly by calibrate_now() or reset_with_measured_q()
        self._calibration_pending = False
        self._calibration_neck_quat_inv: np.ndarray | None = None  # inv(initial neck quat)
        self._calibration_lwrist_offset: np.ndarray | None = None  # position offset
        self._calibration_rwrist_offset: np.ndarray | None = None
        self._calibration_lwrist_rot_offset: sRot | None = None  # orientation offset
        self._calibration_rwrist_rot_offset: sRot | None = None
        # Override robot q for FK during recalibration (e.g. measured joints for VR 3PT)
        self._override_robot_q: np.ndarray | None = None

    @property
    def is_pending(self) -> bool:
        """Check if calibration is pending."""
        return self._calibration_pending

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration has been captured."""
        return self._calibration_neck_quat_inv is not None

    def process_smpl_pose(
        self,
        smpl_pose_np: np.ndarray,
        smpl_joints_local: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Process SMPL pose to extract and calibrate 3-point VR pose.

        Args:
            smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints
            smpl_joints_local: Optional np.ndarray shape (24, 3) - SMPL local joint
                               positions for body visualization. If provided and SMPL
                               visualization is enabled, the joint spheres are updated.

        Returns:
            vr_3pt_pose: np.ndarray shape (3, 7) - Calibrated 3-point pose
                         [L-Wrist, R-Wrist, Neck], each row [x, y, z, qw, qx, qy, qz]
        """
        # Extract raw 3-point pose from SMPL
        vr_3pt_pose_raw = _process_3pt_pose(smpl_pose_np)

        # Capture calibration on first valid frame (or after reset)
        if self._calibration_pending:
            self._capture_calibration(vr_3pt_pose_raw)

        # Apply calibration to get the final pose
        vr_3pt_pose = self._apply_calibration(vr_3pt_pose_raw)

        if self.vr3pt_visualizer is not None:
            self.vr3pt_visualizer.update_from_vr_pose(vr_3pt_pose, waist_scale=1.0)
            if smpl_joints_local is not None:
                self.vr3pt_visualizer.update_smpl_joints(smpl_joints_local)
            self.vr3pt_visualizer.render()

        return vr_3pt_pose

    def close(self) -> None:
        """Close and cleanup visualizer resources."""
        if self.vr3pt_visualizer is not None:
            try:
                self.vr3pt_visualizer.close()
            except Exception as e:
                print(f"[{self.log_prefix}] Warning: Error closing VR3pt visualizer: {e}")

    def calibrate_now(self, body_poses_np: np.ndarray) -> bool:
        """Calibrate using current SMPL frame against FK of all-zero body joints.
        Operator should be in zero-reference pose when calling this."""
        try:
            vr_3pt_pose_raw = _process_3pt_pose(body_poses_np)
            self._override_robot_q = np.zeros(29, dtype=np.float64)
            self._capture_calibration(vr_3pt_pose_raw)
            print(f"[{self.log_prefix}] Calibration completed (zero-pose reference)")
            return True
        except Exception as e:
            print(f"[{self.log_prefix}] Calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _capture_calibration(self, vr_3pt_pose: np.ndarray) -> None:
        """Capture calibration offsets from vr_3pt_pose against G1 FK reference.
        If neck calibration already exists (e.g. from calibrate_now), it is preserved
        to avoid jumps from SMPL noise during recalibration."""

        # Step 1: Neck orientation — only capture if not already set
        if self._calibration_neck_quat_inv is None:
            neck_quat_wxyz = vr_3pt_pose[2, 3:].copy()
            neck_rot = sRot.from_quat(neck_quat_wxyz, scalar_first=True)
            self._calibration_neck_quat_inv = neck_rot.inv().as_quat(scalar_first=True)
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Step 2: Rotate VR wrist positions/orientations by neck inverse
        lwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[0, :3].copy())
        rwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[1, :3].copy())
        lwrist_rot_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
        rwrist_rot_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)

        # Step 3: Get G1 FK reference poses
        if self._robot_model is None:
            raise RuntimeError(
                "Robot model is required for calibration but was not loaded. "
                "Ensure the G1 robot model and URDF are available."
            )
        if get_g1_key_frame_poses is None:
            raise RuntimeError(
                "get_g1_key_frame_poses could not be imported. "
                "Ensure gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer is available."
            )

        # Convert 29-DOF override to full model config if needed
        if self._override_robot_q is not None:
            robot_q = self._robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=self._override_robot_q[:29]
            )
        else:
            robot_q = None
        g1_poses = get_g1_key_frame_poses(self._robot_model, q=robot_q)

        g1_lwrist_pos = g1_poses["left_wrist"]["position"]
        g1_rwrist_pos = g1_poses["right_wrist"]["position"]
        g1_lwrist_rot = sRot.from_quat(
            g1_poses["left_wrist"]["orientation_wxyz"], scalar_first=True
        )
        g1_rwrist_rot = sRot.from_quat(
            g1_poses["right_wrist"]["orientation_wxyz"], scalar_first=True
        )

        # Compute position offsets: calibrated = neck_corrected - offset
        self._calibration_lwrist_offset = lwrist_pos_corrected - g1_lwrist_pos
        self._calibration_rwrist_offset = rwrist_pos_corrected - g1_rwrist_pos

        # Compute orientation offsets: calibrated = rot_offset * neck_corrected
        self._calibration_lwrist_rot_offset = g1_lwrist_rot * lwrist_rot_corrected.inv()
        self._calibration_rwrist_rot_offset = g1_rwrist_rot * rwrist_rot_corrected.inv()

        self._calibration_pending = False
        self._override_robot_q = None

        # Log summary
        source = "override q" if g1_lwrist_pos.any() else "default/zero"
        print(
            f"[{self.log_prefix}] Calibration captured (FK ref: {source}):\n"
            f"  L-Wrist pos offset: [{self._calibration_lwrist_offset[0]:.4f}, "
            f"{self._calibration_lwrist_offset[1]:.4f}, {self._calibration_lwrist_offset[2]:.4f}]\n"
            f"  R-Wrist pos offset: [{self._calibration_rwrist_offset[0]:.4f}, "
            f"{self._calibration_rwrist_offset[1]:.4f}, {self._calibration_rwrist_offset[2]:.4f}]"
        )

    def _apply_calibration(self, vr_3pt_pose: np.ndarray) -> np.ndarray:
        """Apply stored calibration offsets to raw VR 3-point pose."""
        if self._calibration_neck_quat_inv is None:
            return vr_3pt_pose

        calibrated = vr_3pt_pose.copy()
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Neck orientation: calibrated = inv(initial) * current
        neck_rot = sRot.from_quat(vr_3pt_pose[2, 3:], scalar_first=True)
        calibrated[2, 3:] = (calib_inv_rot * neck_rot).as_quat(scalar_first=True)

        # Wrist positions: rotate by neck inverse, then subtract offset
        if self._calibration_lwrist_offset is not None:
            calibrated[0, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[0, :3]) - self._calibration_lwrist_offset
            )
        if self._calibration_rwrist_offset is not None:
            calibrated[1, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[1, :3]) - self._calibration_rwrist_offset
            )

        # Wrist orientations: rot_offset * (neck_inv * current)
        if self._calibration_lwrist_rot_offset is not None:
            lw_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
            calibrated[0, 3:] = (self._calibration_lwrist_rot_offset * lw_corrected).as_quat(
                scalar_first=True
            )
        if self._calibration_rwrist_rot_offset is not None:
            rw_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)
            calibrated[1, 3:] = (self._calibration_rwrist_rot_offset * rw_corrected).as_quat(
                scalar_first=True
            )

        # Neck position via kinematic chain: root → torso_link (+Z) → neck (along calibrated Z)
        neck_z = sRot.from_quat(calibrated[2, 3:], scalar_first=True).apply([0, 0, 1])
        calibrated[2, :3] = (
            np.array([0, 0, self.TORSO_LINK_OFFSET_Z]) + self.NECK_LINK_LENGTH * neck_z
        ).astype(np.float32)

        return calibrated

    def _clear_calibration(self):
        """Clear all calibration state."""
        self._calibration_neck_quat_inv = None
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._override_robot_q = None

    def reset(self) -> None:
        """Reset calibration. Next process_smpl_pose() call will recalibrate."""
        self._clear_calibration()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Calibration reset, will re-calibrate on next frame")

    def reset_with_measured_q(self, body_q_measured: np.ndarray) -> None:
        """Recalibrate wrist offsets using measured robot joints (29 DOFs).
        Preserves neck calibration to avoid jumps from SMPL noise.
        Next process_smpl_pose() will recompute wrist offsets against FK of these joints."""
        # Preserve neck calibration — only clear wrist offsets
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._override_robot_q = body_q_measured.copy()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Wrist recalibration pending (neck preserved, measured q)")


class PoseStreamer:
    """Encapsulates the pose streaming loop state and logic."""

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        num_frames_to_send: int,
        target_fps: int,
        use_cuda: bool,
        record_dir: str,
        record_format: str,
        hand_control_mode: str = HAND_CONTROL_MODE_GESTURE_WUJI,
        wuji_glove_left_address: str = DEFAULT_WUJI_GLOVE_LEFT_ADDRESS,
        wuji_glove_right_address: str = DEFAULT_WUJI_GLOVE_RIGHT_ADDRESS,
        wuji_glove_stale_s: float = DEFAULT_WUJI_GLOVE_STALE_S,
        wuji_glove_left_retargeting_config: str | Path | None = None,
        wuji_glove_right_retargeting_config: str | Path | None = None,
        manus_stale_s: float = DEFAULT_MANUS_STALE_S,
        manus_init_timeout_s: float = DEFAULT_MANUS_INIT_TIMEOUT_S,
        manus_left_retargeting_config: str | Path | None = None,
        manus_right_retargeting_config: str | Path | None = None,
        manus_calibration_file: str | Path = DEFAULT_MANUS_CALIBRATION_FILE,
        manus_left_calibration_file: str | Path = DEFAULT_MANUS_LEFT_CALIBRATION_FILE,
        manus_right_calibration_file: str | Path = DEFAULT_MANUS_RIGHT_CALIBRATION_FILE,
        manus_load_calibration: bool = True,
        manus_debug_node_info: bool = False,
        manus_mcp_joint: str = DEFAULT_MANUS_MCP_JOINT,
        log_prefix: str = "PoseLoop",
    ):
        self.socket = socket
        self.reader = reader
        self.num_frames_to_send = num_frames_to_send
        self.target_fps = target_fps
        self.record_dir = record_dir
        self.log_prefix = log_prefix
        self.hand_control_mode = hand_control_mode

        # Injected dependencies
        self.reader = reader
        self.three_point = three_point

        self.device = (
            torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
        )

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.record_idx = 0

        self.left_wuji_retargeter = None
        self.right_wuji_retargeter = None
        self.wuji_glove_provider = None
        self.manus_provider = None
        if self.hand_control_mode == HAND_CONTROL_MODE_GESTURE_WUJI:
            self.left_wuji_retargeter, self.right_wuji_retargeter = init_wuji_retargeters()
            if self.left_wuji_retargeter is None or self.right_wuji_retargeter is None:
                raise RuntimeError(
                    "gesture_wuji mode requires Wuji retargeters. "
                    f"Check WUJI_RETARGETING_ROOT={WUJI_RETARGETING_ROOT}, "
                    f"WUJI_RETARGETING_CONFIG={WUJI_RETARGETING_CONFIG}, "
                    "and install the wuji-retargeting dependencies/assets."
                )
        elif self.hand_control_mode == HAND_CONTROL_MODE_WUJI_GLOVE:
            left_config = wuji_glove_left_retargeting_config or WUJI_GLOVE_LEFT_RETARGETING_CONFIG
            right_config = wuji_glove_right_retargeting_config or WUJI_GLOVE_RIGHT_RETARGETING_CONFIG
            self.left_wuji_retargeter, self.right_wuji_retargeter = init_wuji_retargeters(
                left_config,
                right_config,
            )
            if self.left_wuji_retargeter is None or self.right_wuji_retargeter is None:
                raise RuntimeError(
                    "wuji_glove mode requires Wuji glove retargeters. "
                    f"Check left={left_config}, right={right_config}, "
                    "and install the wuji-retargeting dependencies/assets."
                )
            self.wuji_glove_provider = WujiGloveSkeletonProvider(
                left_address=wuji_glove_left_address,
                right_address=wuji_glove_right_address,
                stale_s=wuji_glove_stale_s,
            )
        elif self.hand_control_mode == HAND_CONTROL_MODE_MANUS:
            if ManusIntegratedProvider is None:
                raise RuntimeError(
                    "manus mode requires the MANUS Integrated SDK pybind module. "
                    "Build third_party/MANUS_SDK/ManusSDK_v3.1.1/SDKClient_Linux."
                ) from MANUS_PROVIDER_IMPORT_ERROR
            left_config = manus_left_retargeting_config or MANUS_LEFT_RETARGETING_CONFIG
            right_config = manus_right_retargeting_config or MANUS_RIGHT_RETARGETING_CONFIG
            self.left_wuji_retargeter, self.right_wuji_retargeter = init_wuji_retargeters(
                left_config,
                right_config,
            )
            if self.left_wuji_retargeter is None or self.right_wuji_retargeter is None:
                raise RuntimeError(
                    "manus mode requires MANUS retargeters. "
                    f"Check left={left_config}, right={right_config}."
                )
            self.manus_provider = ManusIntegratedProvider(
                stale_s=manus_stale_s,
                init_timeout_s=manus_init_timeout_s,
                calibration_file=str(manus_calibration_file),
                left_calibration_file=str(manus_left_calibration_file),
                right_calibration_file=str(manus_right_calibration_file),
                load_calibration=manus_load_calibration,
                debug_node_info=manus_debug_node_info,
                mcp_joint=manus_mcp_joint,
            )
        self.parent_indices = [
            -1,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            9,
            9,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
        ][:24]

        self.step = 0
        self.last_fps_report = time.time()
        self.fps_counter = 0
        # NOTE: Sleep budget set to 95% of the ideal frame period so that the actual
        # FPS lands closer to target_fps despite per-frame processing overhead.
        self.frame_time = 0.95 / max(1, target_fps)
        self.frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))

        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.next_target_ns = None
        self.frame_start = time.time()

        self.buffer_cleared = (
            True  # Start with buffer cleared - wait for full buffer before first send
        )
        self.yaw_accumulator = YawAccumulator()
        self.last_left_wuji_keypoints = None
        self.last_right_wuji_keypoints = None
        self.pose_packets_sent = 0
        self.last_pose_status_time = 0.0
        self.last_left_hand_binary_closed = False
        self.last_right_hand_binary_closed = False
        self.last_left_wuji_qpos_valid = False
        self.last_right_wuji_qpos_valid = False

    def reset_yaw(self):
        """Called when entering pose mode. Resets yaw only.
        Calibration is triggered separately by the operator (A+B+X+Y → calibrate_now)."""
        self.yaw_accumulator.reset()

    def on_mode_exit(self):
        self.frame_buffer.clear()
        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.next_target_ns = None
        self.buffer_cleared = True
        self.step = 0
        self.last_left_wuji_keypoints = None
        self.last_right_wuji_keypoints = None
        self.pose_packets_sent = 0
        self.last_pose_status_time = 0.0
        self.last_left_hand_binary_closed = False
        self.last_right_hand_binary_closed = False
        self.last_left_wuji_qpos_valid = False
        self.last_right_wuji_qpos_valid = False
        if self.left_wuji_retargeter is not None:
            self.left_wuji_retargeter.reset()
        if self.right_wuji_retargeter is not None:
            self.right_wuji_retargeter.reset()

    def _report_pose_publish(self, frame_index: int, pico_fps: float) -> None:
        now = time.time()
        if self.pose_packets_sent > 1 and now - self.last_pose_status_time < POSE_STATUS_INTERVAL_S:
            return
        status = (
            f"[{self.log_prefix}] Publishing pose: "
            f"frame_index={frame_index} packets={self.pose_packets_sent} "
            f"buffer={len(self.frame_buffer['frame_index'])} pico_fps={pico_fps:.1f}"
        )
        if self.hand_control_mode == HAND_CONTROL_MODE_BINARY_TRIGGER:
            left_state = "CLOSED" if self.last_left_hand_binary_closed else "OPEN"
            right_state = "CLOSED" if self.last_right_hand_binary_closed else "OPEN"
            status += f" L={left_state} R={right_state}"
        elif self.hand_control_mode in (HAND_CONTROL_MODE_WUJI_GLOVE, HAND_CONTROL_MODE_MANUS):
            source = "manus" if self.hand_control_mode == HAND_CONTROL_MODE_MANUS else "glove"
            status += (
                f" {source}_valid L={int(self.last_left_wuji_qpos_valid)}"
                f" R={int(self.last_right_wuji_qpos_valid)}"
            )
        print(status)
        self.last_pose_status_time = now

    def close(self) -> None:
        if self.wuji_glove_provider is not None:
            self.wuji_glove_provider.close()
        if self.manus_provider is not None:
            self.manus_provider.close()

    def run_once(self):
        """Execute one iteration of the pose streaming loop."""
        sample = self.reader.get_latest()

        if sample is None:
            time.sleep(0.005)
            return

        latest_data = compute_from_body_poses(
            self.parent_indices, self.device, sample["body_poses_np"]
        )
        _, left_trigger, right_trigger = get_controller_inputs()
        lx, ly, rx, ry = get_controller_axes()

        left_hand_joints, right_hand_joints = compute_hand_joints_from_inputs()
        left_hand_tracking_valid, left_hand_tracking_state = _get_active_hand_tracking_state(
            is_left=True
        )
        right_hand_tracking_valid, right_hand_tracking_state = _get_active_hand_tracking_state(
            is_left=False
        )
        if left_hand_tracking_valid and left_hand_tracking_state is not None:
            left_hand_joints = left_hand_tracking_state
        else:
            left_hand_joints = np.zeros((26, 4, 4), dtype=np.float32)

        if right_hand_tracking_valid and right_hand_tracking_state is not None:
            right_hand_joints = right_hand_tracking_state
        else:
            right_hand_joints = np.zeros((26, 4, 4), dtype=np.float32)
        if self.hand_control_mode == HAND_CONTROL_MODE_WUJI_GLOVE:
            if self.wuji_glove_provider is None:
                raise RuntimeError("wuji_glove mode is missing WujiGloveSkeletonProvider")
            (
                left_wuji_keypoints,
                left_wuji_keypoints_tracked,
                right_wuji_keypoints,
                right_wuji_keypoints_tracked,
            ) = self.wuji_glove_provider.latest_keypoints(
                self.last_left_wuji_keypoints,
                self.last_right_wuji_keypoints,
            )
        elif self.hand_control_mode == HAND_CONTROL_MODE_MANUS:
            if self.manus_provider is None:
                raise RuntimeError("manus mode is missing ManusIntegratedProvider")
            (
                left_wuji_keypoints,
                left_wuji_keypoints_tracked,
                right_wuji_keypoints,
                right_wuji_keypoints_tracked,
            ) = self.manus_provider.latest_keypoints(
                self.last_left_wuji_keypoints,
                self.last_right_wuji_keypoints,
            )
        else:
            (
                left_wuji_keypoints,
                left_wuji_keypoints_tracked,
                right_wuji_keypoints,
                right_wuji_keypoints_tracked,
            ) = compute_wuji_keypoints_from_inputs(
                self.last_left_wuji_keypoints,
                self.last_right_wuji_keypoints,
            )
        if left_wuji_keypoints_tracked:
            self.last_left_wuji_keypoints = left_wuji_keypoints.copy()
        if right_wuji_keypoints_tracked:
            self.last_right_wuji_keypoints = right_wuji_keypoints.copy()
        left_hand_binary_closed = False
        right_hand_binary_closed = False
        if self.hand_control_mode == HAND_CONTROL_MODE_BINARY_TRIGGER:
            (
                left_wuji_qpos,
                left_wuji_qpos_valid,
                left_hand_binary_closed,
                right_wuji_qpos,
                right_wuji_qpos_valid,
                right_hand_binary_closed,
            ) = compute_binary_trigger_wuji_qpos(left_trigger, right_trigger)
        else:
            (
                left_wuji_qpos,
                left_wuji_qpos_valid,
                right_wuji_qpos,
                right_wuji_qpos_valid,
            ) = compute_wuji_qpos_from_keypoints(
                left_wuji_keypoints,
                right_wuji_keypoints,
                self.left_wuji_retargeter,
                self.right_wuji_retargeter,
            )
            if self.hand_control_mode in (HAND_CONTROL_MODE_WUJI_GLOVE, HAND_CONTROL_MODE_MANUS):
                left_wuji_qpos_valid = left_wuji_qpos_valid and left_wuji_keypoints_tracked
                right_wuji_qpos_valid = right_wuji_qpos_valid and right_wuji_keypoints_tracked
        left_wuji_qpos = apply_wuji_qpos_limits(left_wuji_qpos)
        right_wuji_qpos = apply_wuji_qpos_limits(right_wuji_qpos)
        self.last_left_hand_binary_closed = left_hand_binary_closed
        self.last_right_hand_binary_closed = right_hand_binary_closed
        self.last_left_wuji_qpos_valid = bool(left_wuji_qpos_valid)
        self.last_right_wuji_qpos_valid = bool(right_wuji_qpos_valid)

        smpl_pose_np = (
            latest_data["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)[0]
        ).astype(np.float32)
        smpl_joints_np = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0].astype(np.float32)
        )
        body_quat_np = (
            latest_data["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)
        )
        curr_stamp_ns = int(sample.get("timestamp_ns", 0))
        step_ns = int(1e9 / max(1, self.target_fps))
        if self.prev_stamp_ns is None:
            self.prev_stamp_ns = curr_stamp_ns
            self.prev_smpl_pose_np = smpl_pose_np
            self.prev_smpl_joints_np = smpl_joints_np
            self.prev_body_quat_np = body_quat_np
            self.next_target_ns = curr_stamp_ns
            return
        if curr_stamp_ns <= self.prev_stamp_ns:
            return
        if self.next_target_ns is None:
            self.next_target_ns = self.prev_stamp_ns + step_ns
        if self.next_target_ns < self.prev_stamp_ns:
            self.next_target_ns = self.prev_stamp_ns
        if self.next_target_ns > curr_stamp_ns:
            return
        denom = float(curr_stamp_ns - self.prev_stamp_ns)
        alpha = float(self.next_target_ns - self.prev_stamp_ns) / denom if denom > 0.0 else 1.0
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        use_joints = (1.0 - alpha) * self.prev_smpl_joints_np + alpha * smpl_joints_np
        use_pose = _interp_pose_axis_angle(self.prev_smpl_pose_np, smpl_pose_np, alpha).astype(
            np.float32
        )
        use_body_quat = _quat_lerp_normalized(self.prev_body_quat_np, body_quat_np, alpha).astype(
            np.float32
        )
        N = len(self.frame_buffer["frame_index"])

        ##### From @Jiefeng for directly setting the joint position ######
        joint_pos = np.zeros(29)
        body_pose = use_pose.reshape(-1, 21, 3)

        SMPL_L_ELBOW_IDX = 17
        SMPL_L_WRIST_IDX = 19
        SMPL_R_ELBOW_IDX = 18
        SMPL_R_WRIST_IDX = 20

        # G1_L_ELBOW_IDX = 0
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27

        # G1_R_ELBOW_IDX = 0
        G1_R_WRIST_ROLL_IDX = 24  # Done
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28
        smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
        smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
        smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
        smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

        g1_l_elbow_axis = np.array([0, 1, 0])
        g1_l_elbow_q_twist, g1_l_elbow_q_swing = decompose_rotation_aa(
            smpl_l_elbow_aa, g1_l_elbow_axis
        )

        g1_r_elbow_axis = np.array([0, 1, 0])
        g1_r_elbow_q_twist, g1_r_elbow_q_swing = decompose_rotation_aa(
            smpl_r_elbow_aa, g1_r_elbow_axis
        )

        # Move elbow roll/yaw into wrist while preserving wrist pitch from SMPL
        l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )
        r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )

        l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
        r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

        g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
        g1_l_wrist_pitch = -l_wrist_euler[:, 1]
        g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

        g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
        g1_r_wrist_pitch = -r_wrist_euler[:, 1]
        g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

        joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0]
        joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0]
        joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0]

        joint_pos[G1_R_WRIST_ROLL_IDX] = g1_r_wrist_roll[0]
        joint_pos[G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch[0]
        joint_pos[G1_R_WRIST_YAW_IDX] = g1_r_wrist_yaw[0]

        # Process SMPL pose to get calibrated 3-point VR pose and update visualization
        # Pass SMPL local joints for optional body visualization in the VR3Pt viewer
        smpl_joints_for_vis = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0]
            if self.three_point.enable_smpl_vis
            else None
        )
        vr_3pt_pose = self.three_point.process_smpl_pose(
            sample["body_poses_np"], smpl_joints_local=smpl_joints_for_vis
        )
        ##### From @Jiefeng for directly setting the joint position ######

        self.frame_buffer["smpl_pose"].append(use_pose)
        self.frame_buffer["smpl_joints"].append(use_joints)
        self.frame_buffer["body_quat_w"].append(use_body_quat)
        self.frame_buffer["frame_index"].append(int(self.step))
        self.frame_buffer["joint_pos"].append(joint_pos)
        pico_dt = float(sample.get("dt", 0.0))
        pico_fps = float(sample.get("fps", 0.0))
        N = len(self.frame_buffer["frame_index"])

        # Wait for buffer to be completely filled before sending first message after clearing
        buffer_is_full = len(self.frame_buffer["frame_index"]) >= self.num_frames_to_send
        if buffer_is_full and self.buffer_cleared:
            # Buffer is now full with fresh data, can start sending
            self.buffer_cleared = False
            print(
                f"[{self.log_prefix}] Pose stream primed: "
                f"buffer={self.num_frames_to_send} ready for publishing"
            )

        # Get joystick axes for yaw accumulation
        self.yaw_accumulator.update(rx, self.frame_time)

        # Only send if buffer is full and we're not waiting for fresh data
        if buffer_is_full and not self.buffer_cleared:
            # Keep the existing body/control payload, but only publish the final
            # WujiHand targets for hands instead of intermediate hand states.
            pose_data = {
                "smpl_pose": np.stack((self.frame_buffer["smpl_pose"]), axis=0),
                "smpl_joints": np.stack((self.frame_buffer["smpl_joints"]), axis=0),
                "body_quat_w": np.stack((self.frame_buffer["body_quat_w"]), axis=0),
                "joint_pos": np.stack((self.frame_buffer["joint_pos"]), axis=0),
                "joint_vel": np.zeros((N, 29), dtype=np.float32),
                "vr_position": vr_3pt_pose[:, :3].flatten(),
                "vr_orientation": vr_3pt_pose[:, 3:].flatten(),
                "frame_index": np.array((self.frame_buffer["frame_index"]), dtype=np.int64),
                "timestamp_monotonic": np.array(
                    [sample.get("timestamp_monotonic", 0.0)], dtype=np.float64
                ),
                "left_wuji_qpos": left_wuji_qpos.astype(np.float32),
                "right_wuji_qpos": right_wuji_qpos.astype(np.float32),
                "left_wuji_qpos_valid": np.array([left_wuji_qpos_valid], dtype=bool),
                "right_wuji_qpos_valid": np.array([right_wuji_qpos_valid], dtype=bool),
                "left_hand_binary_closed": np.array([left_hand_binary_closed], dtype=bool),
                "right_hand_binary_closed": np.array([right_hand_binary_closed], dtype=bool),
                "heading_increment": np.array(
                    [self.yaw_accumulator.yaw_angle_change()], dtype=np.float32
                ),
            }
            current_frame_index = int(self.frame_buffer["frame_index"][-1])
            pose_raw_data = {
                "body_poses_np": np.asarray(sample["body_poses_np"], dtype=np.float32),
                "pico_dt": np.array([pico_dt], dtype=np.float32),
                "pico_fps": np.array([pico_fps], dtype=np.float32),
                "timestamp_realtime": np.array(
                    [sample.get("timestamp_realtime", 0.0)], dtype=np.float64
                ),
                "timestamp_monotonic": np.array(
                    [sample.get("timestamp_monotonic", 0.0)], dtype=np.float64
                ),
                "timestamp_ns": np.array([curr_stamp_ns], dtype=np.int64),
                "frame_index": np.array([current_frame_index], dtype=np.int64),
                "left_trigger": np.array([left_trigger], dtype=np.float32),
                "right_trigger": np.array([right_trigger], dtype=np.float32),
                "left_wuji_keypoints": left_wuji_keypoints.astype(np.float32),
                "right_wuji_keypoints": right_wuji_keypoints.astype(np.float32),
                "left_wuji_keypoints_valid": np.array([left_wuji_keypoints_tracked], dtype=bool),
                "right_wuji_keypoints_valid": np.array([right_wuji_keypoints_tracked], dtype=bool),
                "left_hand_tracking_state": left_hand_joints.astype(np.float32),
                "right_hand_tracking_state": right_hand_joints.astype(np.float32),
                "left_hand_tracking_valid": np.array([left_hand_tracking_valid], dtype=bool),
                "right_hand_tracking_valid": np.array([right_hand_tracking_valid], dtype=bool),
            }

            self.socket.send(pack_pose_message(pose_data, topic="pose"))
            self.socket.send(pack_pose_message(pose_raw_data, topic="teleop_raw"))
            self.pose_packets_sent += 1
            self._report_pose_publish(current_frame_index, pico_fps)

            if self.record_dir:
                out_path = os.path.join(self.record_dir, f"pose_{self.record_idx:06d}.npz")
                record_payload = dict(pose_data)
                record_payload.update(
                    {f"pose_raw__{key}": value for key, value in pose_raw_data.items()}
                )
                np.savez_compressed(out_path, **record_payload)
                self.record_idx += 1

        self.step += 1
        self.next_target_ns += step_ns
        self.prev_stamp_ns = curr_stamp_ns
        self.prev_smpl_pose_np = smpl_pose_np
        self.prev_smpl_joints_np = smpl_joints_np
        self.prev_body_quat_np = body_quat_np
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_report >= 5.0:
            fps = self.fps_counter / (current_time - self.last_fps_report)
            print(f"[{self.log_prefix}] FPS: {fps:.2f}, Step: {self.step}")
            self.fps_counter = 0
            self.last_fps_report = current_time
        elapsed = time.time() - self.frame_start
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self.frame_start = time.time()


def run_pico(
    buffer_size: int = 15,
    port: int = 5556,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """Run Pico body tracking with real-time visualization and ZMQ streaming."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run Pico streaming."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"ZMQ socket bound to port {port}")
    if build_command_message is not None and build_planner_message is not None:
        try:
            socket.send(build_command_message(start=False, stop=False, planner=False))
            socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0))
        except Exception as e:
            print(f"Warning: failed to send initial command/planner messages: {e}")
    try:
        _pose_stream_common(
            socket=socket,
            buffer_size=buffer_size,
            num_frames_to_send=num_frames_to_send,
            target_fps=target_fps,
            use_cuda=use_cuda,
            record_dir=record_dir,
            record_format=record_format,
            stop_event=None,
            log_prefix="Main",
            enable_vis_vr3pt=enable_vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=enable_waist_tracking,
            enable_smpl_vis=enable_smpl_vis,
        )
    finally:
        socket.close()
        context.term()
        print("Threads stopped, ZMQ socket closed")


class FeedbackReader:
    """Reads feedback from robot via ZMQ and processes measured upper body position to use as frozen targets."""

    def __init__(self, zmq_feedback_host: str = "localhost", zmq_feedback_port: int = 5557):
        self.poller = ZMQPoller(host=zmq_feedback_host, port=zmq_feedback_port, topic="g1_debug")

        self.upper_body_joint_indices = self._get_upper_body_joint_indices()

        self.upper_body_position_target = None
        self.left_hand_position_target = None
        self.right_hand_position_target = None
        # Full body joint configuration (29 DOFs) as measured from robot,
        # used for FK when recalibrating VR 3PT tracking against actual robot pose
        self.full_body_q_measured: np.ndarray | None = None

    def _get_upper_body_joint_indices(self) -> list[int]:
        # TODO: get from robot model, not hardcoded
        # robot_model = instantiate_g1_robot_model()
        # return robot_model.get_joint_group_indices("upper_body")
        return [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

    def poll_feedback(self):
        """Poll for feedback once, and update internal state."""
        (
            self.upper_body_position_target,
            self.left_hand_position_target,
            self.right_hand_position_target,
            self.full_body_q_measured,
        ) = self._process_upper_body_position_targets()
        print("[PlannerLoop] Saved upper body position target:", self.upper_body_position_target)

    def _process_upper_body_position_targets(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        data = self.poller.get_data()

        if data is None:
            print("[PlannerLoop] No feedback data received")
            return None, None, None, None

        unpacked = msgpack.unpackb(data, raw=False)
        full_body_q = None
        if "body_q_measured" in unpacked:
            body_q_swizzled = unpacked["body_q_measured"]
            full_body_q = np.array(body_q_swizzled, dtype=np.float64)
            body_q = [body_q_swizzled[i] for i in self.upper_body_joint_indices]
        else:
            print("[PlannerLoop] body_q_measured not in feedback data")
            body_q = None

        if "left_hand_q_measured" in unpacked:
            left_hand_q = unpacked["left_hand_q_measured"]
        else:
            print("[PlannerLoop] left_hand_q_measured not in feedback data")
            left_hand_q = None

        if "right_hand_q_measured" in unpacked:
            right_hand_q = unpacked["right_hand_q_measured"]
        else:
            print("[PlannerLoop] right_hand_q_measured not in feedback data")
            right_hand_q = None

        return body_q, left_hand_q, right_hand_q, full_body_q


class PlannerStreamer:
    """Encapsulates the planner control loop state and logic."""

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        poll_hz: int = 20,
        zmq_feedback_host: str = "localhost",
        zmq_feedback_port: int = 5557,
        planner_control_source: str = PLANNER_CONTROL_SOURCE_PICO,
        keyboard_hold_s: float = PLANNER_KEYBOARD_HOLD_S,
        keyboard_mode_debounce_s: float = PLANNER_KEYBOARD_MODE_DEBOUNCE_S,
        planner_translation_speed: float = DEFAULT_PLANNER_TRANSLATION_SPEED,
        planner_turn_yaw_gain: float = DEFAULT_PLANNER_TURN_YAW_GAIN,
    ):
        if planner_control_source not in VALID_PLANNER_CONTROL_SOURCES:
            raise ValueError(
                f"unsupported planner_control_source={planner_control_source!r}; "
                f"expected one of {VALID_PLANNER_CONTROL_SOURCES}"
            )
        self.socket = socket
        self.reader = reader
        self.three_point = three_point
        self.feedback_reader = FeedbackReader(
            zmq_feedback_host=zmq_feedback_host, zmq_feedback_port=zmq_feedback_port
        )

        self.dt = 1.0 / max(1, poll_hz)
        self.planner_control_source = planner_control_source
        self.keyboard_hold_s = float(keyboard_hold_s)
        self.keyboard_mode_debounce_s = float(keyboard_mode_debounce_s)
        self.planner_translation_speed = float(planner_translation_speed)
        self.planner_turn_yaw_gain = float(planner_turn_yaw_gain)
        if self.planner_translation_speed < 0:
            raise ValueError("planner_translation_speed must be >= 0")
        if self.planner_turn_yaw_gain < 0:
            raise ValueError("planner_turn_yaw_gain must be >= 0")
        self.last_keyboard_mode_switch_s = 0.0
        # Current locomotion mode, default slow walk for data collection.
        self.mode = DEFAULT_PLANNER_LOCOMOTION_MODE
        self.prev_ab = False
        self.prev_xy = False
        # Persistent facing buffer (unit vector on XY plane)
        self.yaw_accumulator = YawAccumulator(yaw_gain=self.planner_turn_yaw_gain)
        self.last_send = time.time()
        self.last_xrt_timestamp = None

    def reset_yaw(self):
        """Called when entering planner mode. Resets state for fresh start."""
        self.yaw_accumulator.reset()

    def save_upper_body_position_target(self):
        """Poll feedback and save upper body position target."""
        self.feedback_reader.poll_feedback()

    def recalibrate_for_vr3pt(self):
        """
        Recalibrate VR 3-point pose tracking using the robot's current measured joints.

        Polls the g1_debug feedback to get the robot's actual joint state, then
        schedules recalibration so VR tracking aligns with the robot's current pose.
        This prevents sudden jumps when entering VR 3PT mode from PLANNER mode.
        """
        self.feedback_reader.poll_feedback()
        if self.feedback_reader.full_body_q_measured is not None:
            self.three_point.reset_with_measured_q(self.feedback_reader.full_body_q_measured)
            print("[PlannerLoop] VR 3PT recalibration scheduled with measured robot pose")
        else:
            # Fallback: use zeros if no feedback available
            print(
                "[PlannerLoop] WARNING: No feedback data for VR 3PT recalibration, "
                "using zero body_q as fallback"
            )
            self.three_point.reset_with_measured_q(np.zeros(29, dtype=np.float64))

    def _clamp_locomotion_mode(self, value: int) -> LocomotionMode:
        min_mode = LocomotionMode.IDLE.value
        max_mode = LocomotionMode.INJURED_WALK.value
        return LocomotionMode(min(max_mode, max(min_mode, int(value))))

    def _report_locomotion_mode_change(
        self,
        *,
        old_mode: LocomotionMode,
        new_mode: LocomotionMode,
        input_source: str,
    ) -> None:
        prev_mode = self._clamp_locomotion_mode(new_mode.value - 1)
        next_mode = self._clamp_locomotion_mode(new_mode.value + 1)
        source_text = f"Control source={self.planner_control_source}"
        if input_source != self.planner_control_source:
            source_text += f" input={input_source}"
        print(
            f"[PlannerLoop] {source_text} mode {old_mode.name} -> {new_mode.name}; "
            f"prev={prev_mode.name} next={next_mode.name}"
        )

    def _apply_locomotion_mode_delta(self, delta: int, *, input_source: str) -> None:
        if delta == 0:
            return
        old_mode = self.mode
        self.mode = self._clamp_locomotion_mode(self.mode.value + delta)
        self._report_locomotion_mode_change(
            old_mode=old_mode,
            new_mode=self.mode,
            input_source=input_source,
        )

    def _read_pico_planner_input(self) -> tuple[float, float, float, str]:
        lx, ly, rx, _ = get_controller_axes()
        return float(lx), float(ly), float(rx), PLANNER_CONTROL_SOURCE_PICO

    def _keyboard_key_active(self, key_reader: TerminalKeyReader | None, key: str) -> bool:
        if key_reader is None:
            return False
        return key_reader.is_recently_pressed(key, hold_s=self.keyboard_hold_s)

    def _read_keyboard_planner_input(
        self,
        key_reader: TerminalKeyReader | None,
    ) -> tuple[float, float, float, str, bool]:
        if key_reader is None:
            return 0.0, 0.0, 0.0, PLANNER_CONTROL_SOURCE_KEYBOARD, False

        active = key_reader.has_recent_press(
            PLANNER_KEYBOARD_ACTIVITY_KEYS,
            hold_s=self.keyboard_hold_s,
        )
        now = time.monotonic()
        mode_delta = 0
        if key_reader.is_pressed("[") and now - self.last_keyboard_mode_switch_s >= self.keyboard_mode_debounce_s:
            mode_delta = -1
            self.last_keyboard_mode_switch_s = now
            key_reader.consume_keys("[")
            active = True
        elif (
            key_reader.is_pressed("]")
            and now - self.last_keyboard_mode_switch_s >= self.keyboard_mode_debounce_s
        ):
            mode_delta = 1
            self.last_keyboard_mode_switch_s = now
            key_reader.consume_keys("]")
            active = True
        self._apply_locomotion_mode_delta(mode_delta, input_source=PLANNER_CONTROL_SOURCE_KEYBOARD)

        if self._keyboard_key_active(key_reader, " "):
            return 0.0, 0.0, 0.0, PLANNER_CONTROL_SOURCE_KEYBOARD, True

        lx = float(self._keyboard_key_active(key_reader, "d")) - float(
            self._keyboard_key_active(key_reader, "a")
        )
        ly = float(self._keyboard_key_active(key_reader, "w")) - float(
            self._keyboard_key_active(key_reader, "s")
        )
        rx = float(self._keyboard_key_active(key_reader, "l")) - float(
            self._keyboard_key_active(key_reader, "j")
        )

        move_norm = float(np.hypot(lx, ly))
        if move_norm > 1.0:
            lx /= move_norm
            ly /= move_norm
        return lx, ly, rx, PLANNER_CONTROL_SOURCE_KEYBOARD, active

    def _select_planner_input(
        self,
        key_reader: TerminalKeyReader | None,
    ) -> tuple[float, float, float, str]:
        if self.planner_control_source == PLANNER_CONTROL_SOURCE_KEYBOARD:
            lx, ly, rx, input_source, _ = self._read_keyboard_planner_input(key_reader)
            return lx, ly, rx, input_source

        if self.planner_control_source == PLANNER_CONTROL_SOURCE_HYBRID:
            lx, ly, rx, input_source, keyboard_active = self._read_keyboard_planner_input(key_reader)
            if keyboard_active:
                return lx, ly, rx, input_source

        return self._read_pico_planner_input()

    def _compute_planner_motion(
        self,
        *,
        lx: float,
        ly: float,
        rx: float,
    ) -> tuple[list[float], list[float], float, LocomotionMode]:
        # Facing from RIGHT stick/JL keys: continuous yaw based on rx.
        facing = self.yaw_accumulator.update(rx, self.dt)

        raw_mag = np.hypot(lx, ly)
        raw_mag = np.clip(raw_mag, 0.0, 1.0)
        if np.abs(raw_mag) < JOYSTICK_DEADZONE:
            mag = 0.0
            speed = -1.0
            mode_to_send = LocomotionMode.IDLE
        else:
            mag = (raw_mag - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
            if mag > 1.0:
                mag = 1.0
            mode_to_send = self.mode

            if self.mode == LocomotionMode.SLOW_WALK:
                speed = self.planner_translation_speed
            elif self.mode == LocomotionMode.WALK:
                speed = -1.0
            elif self.mode == LocomotionMode.RUN:
                speed = 1.5 + 3 * mag  # 1.5 .. 4.5
            else:
                speed = mag  # default 0 .. 1.0

        denom = raw_mag if raw_mag > 0.0 else 1.0
        scale = mag / denom
        movement_local = np.array([-lx, ly]) * scale
        perp_x, perp_y = -facing[1], facing[0]
        rotation_facing = np.array([[perp_x, perp_y], [facing[0], facing[1]]])
        movement_global = rotation_facing @ movement_local
        movement = [movement_global[0], movement_global[1], 0.0]
        return movement, facing, speed, mode_to_send

    def run_once(self, stream_mode: StreamMode, key_reader: TerminalKeyReader | None = None):
        """Execute one iteration of the planner control loop."""
        try:
            # Avoid sending old commands if XRT timestamp hasn't advanced, in case of headset disconnect
            xrt_timestamp = xrt.get_time_stamp_ns()
            if xrt_timestamp == self.last_xrt_timestamp:
                return
            self.last_xrt_timestamp = xrt_timestamp

            lx, ly, rx, _ = self._select_planner_input(key_reader)
            movement, facing, speed, mode_to_send = self._compute_planner_motion(
                lx=lx,
                ly=ly,
                rx=rx,
            )

            upper_body_position = None
            left_hand_position = None
            right_hand_position = None
            if stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                upper_body_position = self.feedback_reader.upper_body_position_target
                left_hand_position = self.feedback_reader.left_hand_position_target
                right_hand_position = self.feedback_reader.right_hand_position_target

            vr_3pt_position = None
            vr_3pt_orientation = None
            vr_3pt_compliance = None
            if stream_mode == StreamMode.PLANNER_VR_3PT:
                sample = self.reader.get_latest()
                if sample is not None:
                    print("[PlannerLoop] Sending VR 3-point pose as target")
                    vr_3pt_pose = self.three_point.process_smpl_pose(sample["body_poses_np"])
                    vr_3pt_position = (vr_3pt_pose[:, :3].flatten()).tolist()
                    vr_3pt_orientation = vr_3pt_pose[:, 3:].flatten().tolist()

                # Keep the existing VR 3PT hand path unchanged: stream the
                # currently tracked hand state directly as planner hand targets.
                lh_joints, rh_joints = compute_hand_joints_from_inputs()
                left_hand_position = lh_joints.reshape(-1).astype(np.float32).tolist()
                right_hand_position = rh_joints.reshape(-1).astype(np.float32).tolist()

            msg = build_planner_message(
                mode_to_send.value,
                movement,
                facing,
                speed=speed,
                height=-1.0,
                upper_body_position=upper_body_position,
                left_hand_position=left_hand_position,
                right_hand_position=right_hand_position,
                vr_3pt_position=vr_3pt_position,
                vr_3pt_orientation=vr_3pt_orientation,
                vr_3pt_compliance=vr_3pt_compliance,
            )
            self.socket.send(msg)
        except Exception as e:
            import traceback

            print(f"[PlannerLoop] error: {e}")
            traceback.print_exc()
            raise

        # pacing
        now = time.time()
        sleep_t = self.dt - (now - self.last_send)
        if sleep_t > 0:
            time.sleep(sleep_t)
        self.last_send = time.time()


def run_pico_manager(
    port: int = 5556,
    buffer_size: int = 15,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    zmq_feedback_host: str = "localhost",
    zmq_feedback_port: int = 5557,
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    hand_control_mode: str = HAND_CONTROL_MODE_GESTURE_WUJI,
    wuji_glove_left_address: str = DEFAULT_WUJI_GLOVE_LEFT_ADDRESS,
    wuji_glove_right_address: str = DEFAULT_WUJI_GLOVE_RIGHT_ADDRESS,
    wuji_glove_stale_s: float = DEFAULT_WUJI_GLOVE_STALE_S,
    wuji_glove_left_retargeting_config: str | Path | None = None,
    wuji_glove_right_retargeting_config: str | Path | None = None,
    manus_stale_s: float = DEFAULT_MANUS_STALE_S,
    manus_init_timeout_s: float = DEFAULT_MANUS_INIT_TIMEOUT_S,
    manus_left_retargeting_config: str | Path | None = None,
    manus_right_retargeting_config: str | Path | None = None,
    manus_calibration_file: str | Path = DEFAULT_MANUS_CALIBRATION_FILE,
    manus_left_calibration_file: str | Path = DEFAULT_MANUS_LEFT_CALIBRATION_FILE,
    manus_right_calibration_file: str | Path = DEFAULT_MANUS_RIGHT_CALIBRATION_FILE,
    manus_load_calibration: bool = True,
    manus_debug_node_info: bool = False,
    manus_mcp_joint: str = DEFAULT_MANUS_MCP_JOINT,
    planner_control_source: str = PLANNER_CONTROL_SOURCE_PICO,
    planner_translation_speed: float = DEFAULT_PLANNER_TRANSLATION_SPEED,
    planner_turn_yaw_gain: float = DEFAULT_PLANNER_TURN_YAW_GAIN,
):
    """
    Manager: creates shared PUB socket and runs pose/planner streamers based on current mode.
    Keyboard input in the manager terminal:
      1: enter POSE
      2: enter PLANNER
      3: enter FROZEN
      4: start/stop policy
      5: pause while in POSE
      W/A/S/D + J/L + [/] control planner when planner_control_source includes keyboard
    Controller input:
      X+Y: start/stop policy
      left stick click: toggle PLANNER/POSE
      right stick click/A/B: start/save/discard collector episode
    """
    if hand_control_mode not in VALID_HAND_CONTROL_MODES:
        raise ValueError(
            f"unsupported hand control mode: {hand_control_mode!r}; "
            f"expected one of {VALID_HAND_CONTROL_MODES}"
        )
    if planner_control_source not in VALID_PLANNER_CONTROL_SOURCES:
        raise ValueError(
            f"unsupported planner_control_source: {planner_control_source!r}; "
            f"expected one of {VALID_PLANNER_CONTROL_SOURCES}"
        )
    if planner_translation_speed < 0:
        raise ValueError("planner_translation_speed must be >= 0")
    if planner_turn_yaw_gain < 0:
        raise ValueError("planner_turn_yaw_gain must be >= 0")
    if wuji_glove_stale_s <= 0.0:
        raise ValueError("wuji_glove_stale_s must be > 0")
    if manus_stale_s <= 0.0:
        raise ValueError("manus_stale_s must be > 0")
    if manus_init_timeout_s <= 0.0:
        raise ValueError("manus_init_timeout_s must be > 0")
    if manus_mcp_joint not in VALID_MANUS_MCP_JOINTS:
        raise ValueError(
            f"unsupported manus_mcp_joint: {manus_mcp_joint!r}; "
            f"expected one of {VALID_MANUS_MCP_JOINTS}"
        )
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run the manager."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"[Manager] ZMQ socket bound to port {port}")
    print(f"[Manager] Hand control mode: {hand_control_mode}")
    if hand_control_mode == HAND_CONTROL_MODE_WUJI_GLOVE:
        print(f"[Manager] Wuji glove left: {wuji_glove_left_address}")
        print(f"[Manager] Wuji glove right: {wuji_glove_right_address}")
        print(f"[Manager] Wuji glove stale timeout: {wuji_glove_stale_s:g}s")
    if hand_control_mode == HAND_CONTROL_MODE_MANUS:
        print(f"[Manager] MANUS stale timeout: {manus_stale_s:g}s")
        print(f"[Manager] MANUS MCP joint: {manus_mcp_joint}")
        print(f"[Manager] MANUS left calibration: {manus_left_calibration_file}")
        print(f"[Manager] MANUS right calibration: {manus_right_calibration_file}")
        print(f"[Manager] MANUS left retargeting: {manus_left_retargeting_config or MANUS_LEFT_RETARGETING_CONFIG}")
        print(f"[Manager] MANUS right retargeting: {manus_right_retargeting_config or MANUS_RIGHT_RETARGETING_CONFIG}")
    print(f"[Manager] Planner control source: {planner_control_source}")
    print(f"[Manager] Planner translation speed: {planner_translation_speed:g} m/s")
    print(f"[Manager] Planner turn yaw gain: {planner_turn_yaw_gain:g} rad/s")

    # Print available locomotion modes
    try:
        print("[Manager] Available modes:")
        for mode in LocomotionMode:
            print(f"  {mode.value}: {mode.name}")
    except Exception:
        pass

    # Create shared reader and 3-point pose processor
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()

    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix="PoseLoop",
    )

    pose_streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        hand_control_mode=hand_control_mode,
        wuji_glove_left_address=wuji_glove_left_address,
        wuji_glove_right_address=wuji_glove_right_address,
        wuji_glove_stale_s=wuji_glove_stale_s,
        wuji_glove_left_retargeting_config=wuji_glove_left_retargeting_config,
        wuji_glove_right_retargeting_config=wuji_glove_right_retargeting_config,
        manus_stale_s=manus_stale_s,
        manus_init_timeout_s=manus_init_timeout_s,
        manus_left_retargeting_config=manus_left_retargeting_config,
        manus_right_retargeting_config=manus_right_retargeting_config,
        manus_calibration_file=manus_calibration_file,
        manus_left_calibration_file=manus_left_calibration_file,
        manus_right_calibration_file=manus_right_calibration_file,
        manus_load_calibration=manus_load_calibration,
        manus_debug_node_info=manus_debug_node_info,
        manus_mcp_joint=manus_mcp_joint,
        log_prefix="PoseLoop",
    )
    planner_streamer = PlannerStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        poll_hz=20,
        zmq_feedback_host=zmq_feedback_host,
        zmq_feedback_port=zmq_feedback_port,
        planner_control_source=planner_control_source,
        planner_translation_speed=planner_translation_speed,
        planner_turn_yaw_gain=planner_turn_yaw_gain,
    )

    # State machine diagram:
    #
    #   Keyboard controls:
    #     1 -> POSE, 2 -> PLANNER, 3 -> PLANNER_FROZEN_UPPER_BODY,
    #     4 -> start/stop policy, 5 -> POSE_PAUSE while in POSE.
    #
    #   Controller controls for collection:
    #     X+Y -> start/stop policy, left stick click toggles PLANNER <-> POSE,
    #     right stick click/A/B -> collector start/save/discard.
    #
    print("Manager keyboard: '1'=POSE, '2'=PLANNER, '3'=FROZEN, '4'=start/stop policy, '5'=pause (in POSE)")
    print(
        "Manager controller: X+Y=start/stop policy, left-stick toggles POSE, "
        "right-stick/A/B=start/save/discard collector"
    )
    print(
        "Planner keyboard controls: W/S forward/back, A/D strafe, J/L turn, "
        "'[' prev mode, ']' next mode, Space idle"
    )
    current_mode = StreamMode.OFF
    collector_control_sequence = 0
    try:
        prev_a_pressed = False
        prev_b_pressed = False
        prev_xy_pressed = False
        prev_left_axis_click = False
        prev_right_axis_click = False

        key_reader = TerminalKeyReader()

        while True:
            key_reader.update_keys()

            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()
            key1_pressed = key_reader.is_pressed('1')
            key2_pressed = key_reader.is_pressed('2')
            key3_pressed = key_reader.is_pressed('3')
            key4_pressed = key_reader.is_pressed('4')
            key5_pressed = key_reader.is_pressed('5')

            left_menu_button, _, _ = get_controller_inputs()
            left_axis_click, right_axis_click = get_axis_clicks()

            a_edge = bool(a_pressed) and not prev_a_pressed
            b_edge = bool(b_pressed) and not prev_b_pressed
            xy_pressed = bool(x_pressed) and bool(y_pressed)
            xy_edge = xy_pressed and not prev_xy_pressed
            left_axis_edge = bool(left_axis_click) and not prev_left_axis_click
            right_axis_edge = bool(right_axis_click) and not prev_right_axis_click

            collector_command = None
            if bool(a_pressed) and bool(b_pressed) and (a_edge or b_edge):
                print("[Manager] WARNING: A and B pressed together; ignoring ambiguous collector save/discard")
            elif right_axis_edge:
                collector_command = COLLECTOR_CONTROL_START
            elif a_edge:
                collector_command = COLLECTOR_CONTROL_SAVE
            elif b_edge:
                collector_command = COLLECTOR_CONTROL_DISCARD
            if collector_command is not None:
                collector_control_sequence += 1
                send_collector_control_packet(
                    socket,
                    command=collector_command,
                    sequence=collector_control_sequence,
                )

            mode_switch = None
            if key1_pressed:
                mode_switch = 'pose'
            elif key2_pressed:
                mode_switch = 'planner'
            elif key3_pressed:
                mode_switch = 'frozen'
            elif key4_pressed or xy_edge:
                mode_switch = 'start_stop'
            elif key5_pressed and current_mode == StreamMode.POSE:
                mode_switch = 'pause'
            elif left_axis_edge and current_mode == StreamMode.PLANNER:
                mode_switch = 'pose'
            elif left_axis_edge and current_mode == StreamMode.POSE:
                mode_switch = 'planner'

            # Consume only manager mode keys here; planner keys are read later by PlannerStreamer.
            key_reader.consume_keys('1', '2', '3', '4', '5')

            pose_pressed = mode_switch == 'pose'
            planner_pressed = mode_switch == 'planner'
            frozen_pressed = mode_switch == 'frozen'
            start_combo = mode_switch == 'start_stop'
            pause_pressed = mode_switch == 'pause'

            new_mode = current_mode
            if current_mode == StreamMode.OFF:
                if start_combo or planner_pressed:
                    print("--------Enter PLANNER mode--------")
                    new_mode = StreamMode.PLANNER
                    sample = reader.get_latest()
                    if sample is not None:
                        three_point.calibrate_now(sample["body_poses_np"])
                    else:
                        print("[Manager] WARNING: No SMPL data available for calibration")

            elif current_mode == StreamMode.PLANNER:
                if start_combo:
                    new_mode = StreamMode.OFF
                elif pose_pressed:
                    new_mode = StreamMode.POSE
                    print("--------Enter POSE mode--------")

            elif current_mode == StreamMode.POSE:
                if start_combo:
                    new_mode = StreamMode.OFF
                elif planner_pressed:
                    new_mode = StreamMode.PLANNER
                elif frozen_pressed:
                    new_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
                elif pause_pressed:
                    new_mode = StreamMode.POSE_PAUSE

            elif current_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                if start_combo:
                    new_mode = StreamMode.OFF
                elif pose_pressed:
                    new_mode = StreamMode.POSE
                elif planner_pressed:
                    new_mode = StreamMode.PLANNER

            elif current_mode == StreamMode.POSE_PAUSE:
                if start_combo:
                    new_mode = StreamMode.OFF
                elif not left_menu_button:
                    new_mode = StreamMode.POSE

            elif current_mode == StreamMode.PLANNER_VR_3PT:
                if start_combo:
                    new_mode = StreamMode.OFF
                elif pose_pressed:
                    new_mode = StreamMode.POSE
                elif planner_pressed:
                    new_mode = StreamMode.PLANNER
                elif frozen_pressed:
                    new_mode = StreamMode.POSE

            # Handle mode transitions before running loop
            if new_mode != current_mode:
                if current_mode == StreamMode.POSE:
                    pose_streamer.on_mode_exit()

                if new_mode == StreamMode.POSE:
                    pose_streamer.reset_yaw()
                    print(f"[Manager] Hand control mode: {hand_control_mode}")
                elif new_mode == StreamMode.PLANNER and current_mode != StreamMode.PLANNER_VR_3PT:
                    # Only reset yaw when freshly entering PLANNER from POSE,
                    # not when returning from VR_3PT sub-mode
                    planner_streamer.reset_yaw()
                elif new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                    if current_mode != StreamMode.PLANNER_VR_3PT:
                        # Freshly entering from POSE: reset yaw and grab initial targets
                        planner_streamer.reset_yaw()
                    # Always re-grab the latest robot state as frozen targets,
                    # whether entering from POSE or returning from VR_3PT
                    # (the old targets are stale after VR_3PT moved the arms)
                    planner_streamer.save_upper_body_position_target()
                elif new_mode == StreamMode.PLANNER_VR_3PT:
                    # Recalibrate VR tracking against the robot's actual current pose
                    # (read via g1_debug feedback + FK) to prevent sudden jumps
                    planner_streamer.recalibrate_for_vr3pt()

            # Run one iteration of the new mode
            if new_mode == StreamMode.POSE:
                pose_streamer.run_once()
            elif (
                new_mode == StreamMode.PLANNER
                or new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
                or new_mode == StreamMode.PLANNER_VR_3PT
            ):
                planner_streamer.run_once(new_mode, key_reader=key_reader)

            # Make sure to send command messages after loop iteration to ensure data arrives before mode switch
            if new_mode != current_mode:
                if new_mode == StreamMode.OFF:
                    send_command_packet(
                        socket,
                        start=False,
                        stop=True,
                        planner=True,
                        reason="stop policy",
                    )
                    exit()
                elif (
                    new_mode == StreamMode.PLANNER
                    or new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
                    or new_mode == StreamMode.PLANNER_VR_3PT
                ):
                    send_command_packet(
                        socket,
                        start=True,
                        stop=False,
                        planner=True,
                        reason=f"switch to {new_mode.name}",
                    )
                elif new_mode == StreamMode.POSE:
                    send_command_packet(
                        socket,
                        start=True,
                        stop=False,
                        planner=False,
                        reason="switch to POSE",
                    )

                print(f"[Manager] StreamMode switch: {current_mode.name} -> {new_mode.name}")
                current_mode = new_mode

            key_reader.clear_keys()

            prev_a_pressed = bool(a_pressed)
            prev_b_pressed = bool(b_pressed)
            prev_xy_pressed = xy_pressed
            prev_left_axis_click = bool(left_axis_click)
            prev_right_axis_click = bool(right_axis_click)

    except KeyboardInterrupt:
        print("\nStopping manager...")
    finally:
        # Cleanup resources
        reader.stop()
        pose_streamer.close()
        three_point.close()
        socket.close()
        context.term()
        print("[Manager] Shutdown complete")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=15, help="Sliding window buffer size")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port (default: 5556)")
    parser.add_argument(
        "--num_frames_to_send", type=int, default=5, help="Number of frames to send (default: 200)"
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Target loop FPS (default: 50)")
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA for tensors and model (default: CPU)"
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="Directory to save sent batches (default: disabled)",
    )
    parser.add_argument(
        "--record_format",
        type=str,
        default="npz",
        help="Recording format: 'npz' or 'bin' (default: npz)",
    )
    parser.add_argument(
        "--manager",
        action="store_true",
        help="Run manager with planner and pose threads (interactive)",
    )
    parser.add_argument(
        "--zmq_feedback_host",
        type=str,
        default="localhost",
        help="ZMQ feedback host (default: localhost)",
    )
    parser.add_argument(
        "--zmq_feedback_port",
        type=int,
        default=5557,
        help="ZMQ feedback port (default: 5557)",
    )
    parser.add_argument(
        "--vr3pt_test",
        action="store_true",
        help="Run VR 3-point pose visualizer test (reference frames only)",
    )
    parser.add_argument(
        "--vr3pt_live",
        action="store_true",
        help="Capture one frame of VR 3-point pose and visualize with reference frames",
    )
    parser.add_argument(
        "--vr3pt_realtime",
        action="store_true",
        help="Run standalone real-time VR 3-point pose visualizer",
    )
    parser.add_argument(
        "--vis_vr3pt",
        action="store_true",
        help="Enable inline VR 3-point pose visualization in pose streaming mode",
    )
    parser.add_argument(
        "--vr3pt_hz",
        type=int,
        default=10,
        help="Update rate for real-time VR visualization in Hz (default: 10)",
    )
    parser.add_argument(
        "--no_g1",
        action="store_true",
        help="Disable G1 robot visualization in VR 3pt pose view (G1 is shown by default)",
    )
    parser.add_argument(
        "--waist_tracking",
        action="store_true",
        help="Enable G1 robot waist to follow VR head orientation (disabled by default for performance)",
    )
    parser.add_argument(
        "--vis_smpl",
        action="store_true",
        help="Enable SMPL body joint visualization (24 joint spheres) in the VR3pt viewer",
    )
    parser.add_argument(
        "--hand-control-mode",
        type=str,
        default=HAND_CONTROL_MODE_GESTURE_WUJI,
        choices=VALID_HAND_CONTROL_MODES,
        help="Hand control mode for POSE teleoperation.",
    )
    parser.add_argument(
        "--wuji-glove-left-address",
        "--wuji_glove_left_address",
        dest="wuji_glove_left_address",
        type=str,
        default=DEFAULT_WUJI_GLOVE_LEFT_ADDRESS,
        help="Left Wuji glove address for wuji_glove mode, e.g. 192.168.123.100:50001.",
    )
    parser.add_argument(
        "--wuji-glove-right-address",
        "--wuji_glove_right_address",
        dest="wuji_glove_right_address",
        type=str,
        default=DEFAULT_WUJI_GLOVE_RIGHT_ADDRESS,
        help="Right Wuji glove address for wuji_glove mode, e.g. 192.168.123.101:50001.",
    )
    parser.add_argument(
        "--wuji-glove-stale-s",
        "--wuji_glove_stale_s",
        dest="wuji_glove_stale_s",
        type=float,
        default=DEFAULT_WUJI_GLOVE_STALE_S,
        help="Seconds before a glove skeleton frame is considered stale.",
    )
    parser.add_argument(
        "--wuji-glove-left-retargeting-config",
        "--wuji_glove_left_retargeting_config",
        dest="wuji_glove_left_retargeting_config",
        type=str,
        default="",
        help="Optional left glove retargeting YAML override.",
    )
    parser.add_argument(
        "--wuji-glove-right-retargeting-config",
        "--wuji_glove_right_retargeting_config",
        dest="wuji_glove_right_retargeting_config",
        type=str,
        default="",
        help="Optional right glove retargeting YAML override.",
    )
    parser.add_argument(
        "--manus-stale-s",
        "--manus_stale_s",
        dest="manus_stale_s",
        type=float,
        default=DEFAULT_MANUS_STALE_S,
        help="Seconds before a MANUS glove skeleton frame is considered stale.",
    )
    parser.add_argument(
        "--manus-init-timeout-s",
        "--manus_init_timeout_s",
        dest="manus_init_timeout_s",
        type=float,
        default=DEFAULT_MANUS_INIT_TIMEOUT_S,
        help="Seconds to wait while initializing the MANUS Integrated SDK.",
    )
    parser.add_argument(
        "--manus-left-retargeting-config",
        "--manus_left_retargeting_config",
        dest="manus_left_retargeting_config",
        type=str,
        default="",
        help="Optional left MANUS retargeting YAML override.",
    )
    parser.add_argument(
        "--manus-right-retargeting-config",
        "--manus_right_retargeting_config",
        dest="manus_right_retargeting_config",
        type=str,
        default="",
        help="Optional right MANUS retargeting YAML override.",
    )
    parser.add_argument(
        "--manus-calibration-file",
        "--manus_calibration_file",
        dest="manus_calibration_file",
        type=str,
        default=str(DEFAULT_MANUS_CALIBRATION_FILE),
        help="Fallback MANUS calibration file used when side-specific files are missing.",
    )
    parser.add_argument(
        "--manus-left-calibration-file",
        "--manus_left_calibration_file",
        dest="manus_left_calibration_file",
        type=str,
        default=str(DEFAULT_MANUS_LEFT_CALIBRATION_FILE),
        help="Left MANUS calibration file for manus mode.",
    )
    parser.add_argument(
        "--manus-right-calibration-file",
        "--manus_right_calibration_file",
        dest="manus_right_calibration_file",
        type=str,
        default=str(DEFAULT_MANUS_RIGHT_CALIBRATION_FILE),
        help="Right MANUS calibration file for manus mode.",
    )
    parser.add_argument(
        "--no-manus-load-calibration",
        dest="manus_load_calibration",
        action="store_false",
        default=True,
        help="Do not load project-local MANUS calibration files on startup.",
    )
    parser.add_argument(
        "--manus-debug-node-info",
        dest="manus_debug_node_info",
        action="store_true",
        help="Print one MANUS raw node table per glove.",
    )
    parser.add_argument(
        "--manus-mcp-joint",
        "--manus_mcp_joint",
        dest="manus_mcp_joint",
        type=str,
        choices=VALID_MANUS_MCP_JOINTS,
        default=DEFAULT_MANUS_MCP_JOINT,
        help="Which MANUS non-thumb joint should be used as the MediaPipe MCP landmark.",
    )
    parser.add_argument(
        "--planner-control-source",
        "--planner_control_source",
        dest="planner_control_source",
        type=str,
        default=PLANNER_CONTROL_SOURCE_PICO,
        choices=VALID_PLANNER_CONTROL_SOURCES,
        help="Planner locomotion input source: pico, keyboard, or hybrid.",
    )
    parser.add_argument(
        "--planner-translation-speed",
        "--planner_translation_speed",
        dest="planner_translation_speed",
        type=float,
        default=DEFAULT_PLANNER_TRANSLATION_SPEED,
        help="Planner SLOW_WALK target speed for forward/back/left/right movement in m/s.",
    )
    parser.add_argument(
        "--planner-turn-yaw-gain",
        "--planner_turn_yaw_gain",
        dest="planner_turn_yaw_gain",
        type=float,
        default=DEFAULT_PLANNER_TURN_YAW_GAIN,
        help="Planner turning yaw gain in rad/s at full right-stick or J/L keyboard input.",
    )
    args = parser.parse_args()

    # Standalone VR3Pt test modes (exit after finishing)
    if args.vr3pt_test:
        print("Running VR 3-point pose visualizer test...")
        run_vr3pt_visualizer_test()
        print("VR 3-point pose visualizer test completed")
        exit(0)

    if args.vr3pt_live:
        print("Running VR 3-point pose live capture...")
        run_vr3pt_live_visualizer()
        print("VR 3-point pose live visualizer completed")
        exit(0)

    if args.vr3pt_realtime:
        print("Running VR 3-point pose real-time visualizer...")
        run_vr3pt_realtime_visualizer(update_hz=args.vr3pt_hz)
        print("VR 3-point pose real-time visualizer completed")
        exit(0)

    # Main execution modes
    # G1 robot visualization is enabled by default when vis_vr3pt is used
    with_g1_robot = not args.no_g1

    if args.manager:
        run_pico_manager(
            port=args.port,
            buffer_size=args.buffer_size,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            zmq_feedback_host=args.zmq_feedback_host,
            zmq_feedback_port=args.zmq_feedback_port,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
            hand_control_mode=args.hand_control_mode,
            wuji_glove_left_address=args.wuji_glove_left_address,
            wuji_glove_right_address=args.wuji_glove_right_address,
            wuji_glove_stale_s=args.wuji_glove_stale_s,
            wuji_glove_left_retargeting_config=args.wuji_glove_left_retargeting_config or None,
            wuji_glove_right_retargeting_config=args.wuji_glove_right_retargeting_config or None,
            manus_stale_s=args.manus_stale_s,
            manus_init_timeout_s=args.manus_init_timeout_s,
            manus_left_retargeting_config=args.manus_left_retargeting_config or None,
            manus_right_retargeting_config=args.manus_right_retargeting_config or None,
            manus_calibration_file=args.manus_calibration_file,
            manus_left_calibration_file=args.manus_left_calibration_file,
            manus_right_calibration_file=args.manus_right_calibration_file,
            manus_load_calibration=args.manus_load_calibration,
            manus_debug_node_info=args.manus_debug_node_info,
            manus_mcp_joint=args.manus_mcp_joint,
            planner_control_source=args.planner_control_source,
            planner_translation_speed=args.planner_translation_speed,
            planner_turn_yaw_gain=args.planner_turn_yaw_gain,
        )
    else:
        # Run legacy single-thread pose streaming
        run_pico(
            buffer_size=args.buffer_size,
            port=args.port,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
        )
