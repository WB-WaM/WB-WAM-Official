"""Quantify 20 Hz downsample/reconstruction choices for HGPT replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from deploy.play_track import load_offline_motions
from tracking import constants as consts
from tracking.infer_utils import G1TrackInferFn, G1TrackMjSim, g1_infer_env_config
from tracking.metrics import calculate_kpt_mae_error
from tracking.policy import Args as PolicyArgs, get_policy_onnx

METHOD_FILES = {
    "causal_latest_20hz": "mode2_causal_latest_20hz.npz",
    "interpolate_20hz": "mode3_interpolate_20hz.npz",
    "master_50hz": "mode4_master_50hz.npz",
}
METHOD_LABELS = {
    "causal_latest_20hz": "20 Hz causal-latest",
    "interpolate_20hz": "20 Hz interpolation",
    "master_50hz": "Direct 50 Hz",
}
CORE_START = 25  # load_offline_motions adds a 0.5 s transition at 50 Hz.
COMMON_CORE_FRAMES = 1421  # 28.4 s common support across all three files.
WAIST_PITCH_QPOS_INDEX = 7 + 14


def _yaw(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion_xyzw = quaternion_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quaternion_xyzw).as_euler("xyz")[:, 2]


def _angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.asarray(values)[mask]))


def _load_reference(path: Path, model: mujoco.MjModel) -> dict[str, np.ndarray]:
    motions = load_offline_motions(str(path), model, freq=50)
    if len(motions) != 1:
        raise RuntimeError(
            f"expected exactly one motion from {path}, got {len(motions)}"
        )
    return motions[0]["data"]


def _simulate_reference(
    reference: dict[str, np.ndarray], policy
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    simulation = G1TrackMjSim(
        init_qpos=consts.DEFAULT_QPOS.copy(),
        headless=True,
        ctrl_dt=0.02,
    )
    infer = G1TrackInferFn(
        g1_infer_env_config(ctrl_dt=0.02),
        simulation.mj_model,
        policy,
        privileged=False,
    )
    state = simulation.reset(simulation.init_state())
    actual_qpos: list[np.ndarray] = []
    actual_qvel: list[np.ndarray] = []
    kpt_position_error: list[float] = []
    kpt_rotation_error: list[float] = []

    length = len(reference["qpos"])
    for index in range(length):
        next_index = min(index + 1, length - 1)
        current = {key: value[index : index + 1] for key, value in reference.items()}
        following = {
            key: value[next_index : next_index + 1] for key, value in reference.items()
        }
        target = infer.infer_onnx(
            state,
            {"ref_curr": current, "ref_next": following},
        )
        state = simulation.step(state, target)
        kpt_pos, kpt_rot = calculate_kpt_mae_error(
            state, current, following, simulation.mj_model
        )
        actual_qpos.append(state.mj_data.qpos.copy())
        actual_qvel.append(state.mj_data.qvel.copy())
        kpt_position_error.append(float(kpt_pos))
        kpt_rotation_error.append(float(kpt_rot))

    return (
        np.asarray(actual_qpos),
        np.asarray(actual_qvel),
        np.asarray(kpt_position_error),
        np.asarray(kpt_rotation_error),
    )


def _segment_masks(master_qpos: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    time_s = np.arange(len(master_qpos), dtype=np.float64) / 50.0
    root_xy = master_qpos[:, :2]
    speed = np.r_[0.0, np.linalg.norm(np.diff(root_xy, axis=0), axis=1) * 50.0]
    speed = np.convolve(speed, np.ones(11) / 11.0, mode="same")
    yaw = np.unwrap(_yaw(master_qpos[:, 3:7]))
    yaw_from_start_deg = np.degrees(
        np.arctan2(np.sin(yaw - yaw[0]), np.cos(yaw - yaw[0]))
    )
    waist_pitch_deg = np.degrees(master_qpos[:, WAIST_PITCH_QPOS_INDEX])

    masks = {
        "straight_walk": (time_s >= 1.2) & (time_s <= 7.8),
        "stationary_action": (time_s >= 8.8) & (time_s <= 19.8),
        "full_body_turn": (time_s >= 20.4) & (time_s <= 24.6),
        # The recording has no deep bend. This mask isolates its mild stationary
        # waist-flexion frames while excluding locomotion and the large yaw turn.
        "mild_waist_bend": (
            (time_s >= 8.8)
            & (time_s <= 19.8)
            & (speed < 0.08)
            & (np.abs(yaw_from_start_deg) < 10.0)
            & (waist_pitch_deg <= -3.0)
        ),
    }
    for name, mask in masks.items():
        if not np.any(mask):
            raise RuntimeError(f"segment {name} has no frames")

    definitions = {
        "straight_walk": "1.2-7.8 s; sustained forward translation",
        "stationary_action": "8.8-19.8 s; root nearly stationary while joints move",
        "full_body_turn": "20.4-24.6 s; approximately 90 degree yaw excursion",
        "mild_waist_bend": (
            "stationary-action frames with speed <0.08 m/s, |yaw-yaw0| <10 deg, "
            "and waist pitch <=-3 deg"
        ),
    }
    segment_context = {
        name: {
            "definition": definitions[name],
            "frames": int(np.count_nonzero(mask)),
            "duration_s": float(np.count_nonzero(mask) / 50.0),
        }
        for name, mask in masks.items()
    }
    segment_context["mild_waist_bend"]["peak_reference_flexion_deg"] = float(
        np.min(waist_pitch_deg[masks["mild_waist_bend"]])
    )
    return masks, segment_context


def _rotation_mae_deg(
    pose_a: np.ndarray, pose_b: np.ndarray, mask: np.ndarray
) -> float:
    rotation_a = pose_a[mask, :, :3, :3]
    rotation_b = pose_b[mask, :, :3, :3]
    relative = np.einsum("...ji,...jk->...ik", rotation_a, rotation_b, optimize=True)
    angles = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
    return float(np.degrees(np.mean(angles)))


def _straight_geometry(
    actual_xy: np.ndarray, master_xy: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    reference = master_xy[mask]
    actual = actual_xy[mask]
    reference_delta = reference[-1] - reference[0]
    reference_distance = float(np.linalg.norm(reference_delta))
    direction = reference_delta / max(reference_distance, 1e-9)
    normal = np.array([-direction[1], direction[0]])
    cross_track = (actual - reference[0]) @ normal
    actual_delta = actual[-1] - actual[0]
    actual_distance = float(np.linalg.norm(actual_delta))
    heading_error = np.arctan2(actual_delta[1], actual_delta[0]) - np.arctan2(
        reference_delta[1], reference_delta[0]
    )
    heading_error = abs(np.arctan2(np.sin(heading_error), np.cos(heading_error)))
    return {
        "cross_track_rms_mm": float(1000.0 * np.sqrt(np.mean(cross_track**2))),
        "cross_track_max_mm": float(1000.0 * np.max(np.abs(cross_track))),
        "distance_completion_ratio": actual_distance / max(reference_distance, 1e-9),
        "heading_error_deg": float(np.degrees(heading_error)),
        "reference_distance_m": reference_distance,
        "actual_distance_m": actual_distance,
    }


def analyze(input_dir: Path, output_dir: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(consts.TRACK_XML))
    references = {
        method: _load_reference(input_dir / filename, model)
        for method, filename in METHOD_FILES.items()
    }
    core_references = {
        method: {
            key: value[CORE_START : CORE_START + COMMON_CORE_FRAMES]
            for key, value in reference.items()
        }
        for method, reference in references.items()
    }
    master = core_references["master_50hz"]
    masks, segment_context = _segment_masks(master["qpos"])

    policy = get_policy_onnx(
        PolicyArgs(
            load_path="storage/ckpts/pns_wo_priv216.onnx",
            policy_type="mlp",
            device="cpu",
        )
    )
    simulations = {}
    for method, reference in references.items():
        print(f"[Analysis] Simulating {METHOD_LABELS[method]}")
        actual_qpos, actual_qvel, kpt_pos, kpt_rot = _simulate_reference(
            reference, policy
        )
        for name, values in {
            "qpos": actual_qpos,
            "qvel": actual_qvel,
            "kpt_pos": kpt_pos,
            "kpt_rot": kpt_rot,
        }.items():
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"{method} simulation produced non-finite {name}")
        simulations[method] = {
            "qpos": actual_qpos[CORE_START : CORE_START + COMMON_CORE_FRAMES],
            "qvel": actual_qvel[CORE_START : CORE_START + COMMON_CORE_FRAMES],
            "kpt_pos": kpt_pos[CORE_START : CORE_START + COMMON_CORE_FRAMES],
            "kpt_rot": kpt_rot[CORE_START : CORE_START + COMMON_CORE_FRAMES],
        }

    reconstruction_rows = []
    tracking_rows = []
    straight_rows = []
    special_rows = []
    master_yaw = _yaw(master["qpos"][:, 3:7])

    for method, reference in core_references.items():
        reference_yaw = _yaw(reference["qpos"][:, 3:7])
        reconstruction_root_xy = np.linalg.norm(
            reference["qpos"][:, :2] - master["qpos"][:, :2], axis=1
        )
        reconstruction_root_z = np.abs(reference["qpos"][:, 2] - master["qpos"][:, 2])
        reconstruction_yaw = _angle_error(reference_yaw, master_yaw)
        reconstruction_joint = np.mean(
            np.abs(reference["qpos"][:, 7:] - master["qpos"][:, 7:]), axis=1
        )
        reconstruction_kpt = np.mean(
            np.linalg.norm(
                reference["kpt2gv_pose"][:, :, :3, 3]
                - master["kpt2gv_pose"][:, :, :3, 3],
                axis=-1,
            ),
            axis=1,
        )

        actual = simulations[method]
        own_root_yaw = reference_yaw
        actual_yaw = _yaw(actual["qpos"][:, 3:7])
        end_to_end_root_xy = np.linalg.norm(
            actual["qpos"][:, :2] - master["qpos"][:, :2], axis=1
        )
        end_to_end_yaw = _angle_error(actual_yaw, master_yaw)
        end_to_end_joint = np.mean(
            np.abs(actual["qpos"][:, 7:] - master["qpos"][:, 7:]), axis=1
        )
        own_root_xy = np.linalg.norm(
            actual["qpos"][:, :2] - reference["qpos"][:, :2], axis=1
        )
        own_yaw = _angle_error(actual_yaw, own_root_yaw)
        own_joint = np.mean(
            np.abs(actual["qpos"][:, 7:] - reference["qpos"][:, 7:]), axis=1
        )

        for segment, mask in masks.items():
            reconstruction_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "segment": segment,
                    "frames": int(np.count_nonzero(mask)),
                    "root_xy_mae_mm": 1000.0 * _mean(reconstruction_root_xy, mask),
                    "root_z_mae_mm": 1000.0 * _mean(reconstruction_root_z, mask),
                    "root_yaw_mae_deg": np.degrees(_mean(reconstruction_yaw, mask)),
                    "joint_mae_deg": np.degrees(_mean(reconstruction_joint, mask)),
                    "kpt_position_mae_mm": 1000.0 * _mean(reconstruction_kpt, mask),
                    "kpt_rotation_mae_deg": _rotation_mae_deg(
                        reference["kpt2gv_pose"], master["kpt2gv_pose"], mask
                    ),
                }
            )
            tracking_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "segment": segment,
                    "frames": int(np.count_nonzero(mask)),
                    "e2e_root_xy_mae_mm": 1000.0 * _mean(end_to_end_root_xy, mask),
                    "e2e_root_yaw_mae_deg": np.degrees(_mean(end_to_end_yaw, mask)),
                    "e2e_joint_mae_deg": np.degrees(_mean(end_to_end_joint, mask)),
                    "own_root_xy_mae_mm": 1000.0 * _mean(own_root_xy, mask),
                    "own_root_yaw_mae_deg": np.degrees(_mean(own_yaw, mask)),
                    "own_joint_mae_deg": np.degrees(_mean(own_joint, mask)),
                    "own_kpt_position_mae_mm": 1000.0 * _mean(actual["kpt_pos"], mask),
                    "own_kpt_rotation_mae_deg": np.degrees(
                        _mean(actual["kpt_rot"], mask)
                    ),
                }
            )

        geometry = _straight_geometry(
            actual["qpos"][:, :2], master["qpos"][:, :2], masks["straight_walk"]
        )
        straight_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                **geometry,
            }
        )

        turn_mask = masks["full_body_turn"]
        bend_mask = masks["mild_waist_bend"]
        master_turn = np.unwrap(master_yaw[turn_mask])
        actual_turn = np.unwrap(actual_yaw[turn_mask])
        master_waist = np.degrees(master["qpos"][:, WAIST_PITCH_QPOS_INDEX])
        actual_waist = np.degrees(actual["qpos"][:, WAIST_PITCH_QPOS_INDEX])
        special_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "turn_reference_excursion_deg": float(
                    np.degrees(np.max(master_turn) - np.min(master_turn))
                ),
                "turn_actual_excursion_deg": float(
                    np.degrees(np.max(actual_turn) - np.min(actual_turn))
                ),
                "turn_yaw_mae_deg": np.degrees(_mean(end_to_end_yaw, turn_mask)),
                "bend_reference_peak_deg": float(np.min(master_waist[bend_mask])),
                "bend_actual_peak_deg": float(np.min(actual_waist[bend_mask])),
                "bend_waist_pitch_mae_deg": _mean(
                    np.abs(actual_waist - master_waist), bend_mask
                ),
            }
        )

    results = {
        "source": str(input_dir),
        "method_files": METHOD_FILES,
        "frequency_hz": 50,
        "common_frames": COMMON_CORE_FRAMES,
        "common_duration_s": (COMMON_CORE_FRAMES - 1) / 50.0,
        "pipeline": {
            "first_frame_alignment": "SE(2) XY/yaw alignment in load_offline_motions",
            "ema_alpha": 0.8,
            "ema_time_constant_ms_at_20hz": float(-1000.0 * 0.05 / np.log(0.8)),
            "ema_time_constant_ms_at_50hz": float(-1000.0 * 0.02 / np.log(0.8)),
            "ema_group_delay_ms_at_20hz": 200.0,
            "ema_group_delay_ms_at_50hz": 80.0,
            "policy": "pns_wo_priv216.onnx on CPUExecutionProvider",
            "simulation": "MuJoCo at 50 Hz, deterministic headless replay",
        },
        "segment_context": segment_context,
        "reconstruction": reconstruction_rows,
        "tracking": tracking_rows,
        "straight_geometry": straight_rows,
        "turn_and_bend": special_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for key in ("reconstruction", "tracking", "straight_geometry", "turn_and_bend"):
        rows = results[key]
        with (output_dir / f"{key}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/home/marslab1/zhust/WB-WAM/datasets/humanoid_gpt/"
            "hgpt_master_50hz/replay_20hz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/downsampling_replay_analysis"),
    )
    args = parser.parse_args()
    analyze(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
