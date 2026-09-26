#!/usr/bin/env python3
"""Visualize a G1 retarget replay pkl/npz in MuJoCo."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WB_ROOT = REPO_ROOT.parents[1]
DEFAULT_REPLAY = (
    WB_ROOT / "datasets" / "external" / "xperience" / "processed" / "xperience-10m-sample" / "g1_motion_gmr_replay.pkl"
)
DEFAULT_MODEL = REPO_ROOT / "gear_sonic_deploy/g1/scene_29dof.xml"
TORSO_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def _as_float_array(value: Any, name: str, ndim: int, width: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {arr.shape}")
    if width is not None and arr.shape[-1] != width:
        raise ValueError(f"{name} must have width {width}, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("root quaternion contains near-zero entries")
    return quat / norm


def _quat_to_wxyz(quat: np.ndarray, order: str) -> np.ndarray:
    if order == "wxyz":
        return quat
    if order == "xyzw":
        return quat[:, [3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion order: {order}")


def load_motion(path: Path, quat_order: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with path.open("rb") as f:
            data = pickle.load(f)
        fps = float(data.get("fps", 20.0))
        root_pos = _as_float_array(data["root_pos"], "root_pos", 2, 3)
        dof_pos = _as_float_array(data["dof_pos"], "dof_pos", 2, 29)
        if "root_quat_wxyz" in data:
            root_quat = _as_float_array(data["root_quat_wxyz"], "root_quat_wxyz", 2, 4)
            detected_order = "wxyz"
        else:
            root_quat = _as_float_array(data["root_rot"], "root_rot", 2, 4)
            detected_order = "xyzw"
    elif suffix == ".npz":
        data = np.load(path)
        fps = float(data["fps"]) if "fps" in data else 20.0
        root_pos = _as_float_array(data["root_pos"], "root_pos", 2, 3)
        dof_pos = _as_float_array(data["dof_pos"], "dof_pos", 2, 29)
        if "root_quat_wxyz" in data:
            root_quat = _as_float_array(data["root_quat_wxyz"], "root_quat_wxyz", 2, 4)
            detected_order = "wxyz"
        else:
            root_quat = _as_float_array(data["root_rot"], "root_rot", 2, 4)
            detected_order = "xyzw"
    else:
        raise ValueError(f"Only .pkl and .npz motion files are supported, got {path}")

    frames = min(root_pos.shape[0], root_quat.shape[0], dof_pos.shape[0])
    if frames <= 0:
        raise ValueError("motion has no frames")

    order = detected_order if quat_order == "auto" else quat_order
    root_quat_wxyz = _quat_to_wxyz(_normalize_quat(root_quat[:frames]), order)
    return {
        "path": path,
        "fps": fps,
        "root_pos": root_pos[:frames].copy(),
        "root_quat_wxyz": root_quat_wxyz,
        "dof_pos": dof_pos[:frames],
        "quat_order": order,
        "detected_quat_order": detected_order,
        "frames": frames,
    }


def get_single_dof_indices(mj_model: mujoco.MjModel, joint_names: tuple[str, ...]) -> np.ndarray:
    indices = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"joint not found in model: {joint_name}")
        dof_index = int(mj_model.jnt_qposadr[joint_id] - 7)
        if not 0 <= dof_index < 29:
            raise ValueError(f"joint {joint_name} maps outside 29-dof body slice: {dof_index}")
        indices.append(dof_index)
    return np.asarray(indices, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize G1 GMR replay motion in MuJoCo.")
    parser.add_argument(
        "replay_path",
        nargs="?",
        type=Path,
        default=DEFAULT_REPLAY,
        help=f"Replay .pkl/.npz path. Default: {DEFAULT_REPLAY}",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"G1 MuJoCo scene XML. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override playback fps.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--start-frame", type=int, default=0, help="Initial frame index.")
    parser.add_argument("--loop", action="store_true", help="Loop when reaching the final frame.")
    parser.add_argument(
        "--freeze-root",
        action="store_true",
        help="Keep root position and orientation fixed while replaying joint angles.",
    )
    parser.add_argument(
        "--root-frame",
        type=int,
        default=0,
        help="Root frame to hold when --freeze-root is set. Default: 0.",
    )
    parser.add_argument(
        "--freeze-torso",
        action="store_true",
        help="Keep waist yaw/roll/pitch fixed while replaying the other joints.",
    )
    parser.add_argument(
        "--torso-frame",
        type=int,
        default=0,
        help="Torso frame to hold when --freeze-torso is set. Default: 0.",
    )
    parser.add_argument(
        "--raw-height",
        action="store_true",
        help="Use root_pos z directly. Default lifts first root z to the model qpos0 height.",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Additional z offset after the default height adjustment.",
    )
    parser.add_argument(
        "--quat-order",
        choices=("auto", "xyzw", "wxyz"),
        default="auto",
        help="Quaternion order in the motion file. Replay pkl root_rot is normally xyzw.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load data/model and print stats only.")
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be positive")

    motion = load_motion(args.replay_path, args.quat_order)
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    if mj_model.nq < 36:
        raise ValueError(f"Expected G1 model nq >= 36, got nq={mj_model.nq}")

    mj_data = mujoco.MjData(mj_model)
    root_pos = motion["root_pos"]
    auto_z = 0.0
    if not args.raw_height:
        auto_z = float(mj_model.qpos0[2] - root_pos[0, 2])
        root_pos[:, 2] += auto_z
    if args.z_offset:
        root_pos[:, 2] += args.z_offset

    source_fps = float(motion["fps"])
    playback_fps = float(args.fps) if args.fps is not None else source_fps
    step_dt = 1.0 / (playback_fps * args.speed)
    mj_model.opt.timestep = step_dt

    start_frame = int(np.clip(args.start_frame, 0, motion["frames"] - 1))
    root_frame = int(np.clip(args.root_frame, 0, motion["frames"] - 1))
    torso_frame = int(np.clip(args.torso_frame, 0, motion["frames"] - 1))
    torso_dof_indices = get_single_dof_indices(mj_model, TORSO_JOINT_NAMES)
    frozen_torso = motion["dof_pos"][torso_frame, torso_dof_indices].copy()
    state = {"frame": start_frame, "paused": False}
    root_mode = f"frozen@frame{root_frame}" if args.freeze_root else "motion"
    torso_mode = f"frozen@frame{torso_frame}" if args.freeze_torso else "motion"

    print(
        f"Loaded {motion['path']} | frames={motion['frames']} | "
        f"source_fps={source_fps:g} | playback_fps={playback_fps:g} | "
        f"quat={motion['quat_order']}->wxyz | z_shift={auto_z + args.z_offset:.4f} | "
        f"root={root_mode} | torso={torso_mode}"
    )
    print("Keys: Space pause/resume, R reset, . next frame, , previous frame")

    if args.dry_run:
        print(f"Model: {model_path} | nq={mj_model.nq} nv={mj_model.nv} nu={mj_model.nu}")
        print(f"First qpos root: pos={root_pos[0].round(4)} quat_wxyz={motion['root_quat_wxyz'][0].round(4)}")
        if args.freeze_root:
            print(
                f"Frozen root: frame={root_frame} "
                f"pos={root_pos[root_frame].round(4)} "
                f"quat_wxyz={motion['root_quat_wxyz'][root_frame].round(4)}"
            )
        if args.freeze_torso:
            torso_values = ", ".join(
                f"{name}={value:.4f}" for name, value in zip(TORSO_JOINT_NAMES, frozen_torso)
            )
            print(
                f"Frozen torso: frame={torso_frame} "
                f"dof_indices={torso_dof_indices.tolist()} {torso_values}"
            )
        return

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode)
        except ValueError:
            key = ""
        if key == " ":
            state["paused"] = not state["paused"]
            print("Paused" if state["paused"] else "Playing")
        elif key in {"R", "r"}:
            state["frame"] = start_frame
            print("Reset")
        elif key == ".":
            state["frame"] = min(state["frame"] + 1, motion["frames"] - 1)
            print("frame", state["frame"])
        elif key == ",":
            state["frame"] = max(state["frame"] - 1, 0)
            print("frame", state["frame"])

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -20.0

        while viewer.is_running():
            step_start = time.time()
            frame = state["frame"] % motion["frames"] if args.loop else state["frame"]
            current_root_frame = root_frame if args.freeze_root else frame
            dof_pos = motion["dof_pos"][frame]
            if args.freeze_torso:
                dof_pos = dof_pos.copy()
                dof_pos[torso_dof_indices] = frozen_torso

            mj_data.qpos[:] = mj_model.qpos0
            mj_data.qpos[:3] = root_pos[current_root_frame]
            mj_data.qpos[3:7] = motion["root_quat_wxyz"][current_root_frame]
            mj_data.qpos[7:36] = dof_pos
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            if not state["paused"]:
                next_frame = state["frame"] + 1
                if next_frame >= motion["frames"]:
                    state["frame"] = 0 if args.loop else motion["frames"] - 1
                    state["paused"] = not args.loop
                else:
                    state["frame"] = next_frame

            sleep_time = step_dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
