#!/usr/bin/env python3
"""Validate the official SONIC-only HumanoidArena v3.1 release for WB-WAM training.

No copy, temporal resampling, action shift, or video transcoding is performed.
The WB-WAM dataset loader applies the model-slot mapping and 224px resize online.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from wbwam.datasets.wb_dataset import WBDataset

TRAINING_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = TRAINING_ROOT / "configs/data/wb_humanoid_arena_native_50hz.yaml"
TASK_CONFIGS = TRAINING_ROOT / "configs/wb_task/posttrain/humanoid_arena_native50"
RELEASE_SUBDIR = Path("HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1")
TASK_INDICES = {
    "hammer": 0,
    "kick_football": 1,
    "box_shelf": 3,
    "punch_markers": 4,
    "open_door": 5,
    "sit_sofa": 6,
    "obstacle_navigation": 7,
}


def _source_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not (root / "meta/info.json").is_file():
        root /= RELEASE_SUBDIR
    if not (root / "meta/info.json").is_file():
        raise FileNotFoundError(f"Official SONIC dataset not found below {path}")
    return root


def _check_release(root: Path) -> None:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "merge_manifest.json").read_text(encoding="utf-8"))
    protocol = info.get("vla_protocol") or {}
    features = info.get("features") or {}
    expected = {
        "fps": 50,
        "total_episodes": 800,
        "total_frames": 677159,
        "total_tasks": 8,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"Expected HumanoidArena {key}={value}, got {info.get(key)!r}")
    if features.get("observation.state", {}).get("shape") != [64]:
        raise ValueError("Official observation.state must have 64 dimensions")
    if features.get("action", {}).get("shape") != [40]:
        raise ValueError("Official action must have 40 dimensions")
    if features.get("observation.images.front", {}).get("dtype") != "video":
        raise ValueError("Official head-camera video is missing")
    if (
        protocol.get("schema") != "unitree_g1_gmt_refpose_v3_1"
        or protocol.get("backend_source") != "sonic"
        or protocol.get("action_semantics") != "reference_pose_not_robot_current_residual"
    ):
        raise ValueError(f"Unexpected SONIC reference-action protocol: {protocol}")
    sources = manifest.get("sources") or []
    if len(sources) != 8 or any(source.get("dataset_name") != "sonic_refpose_v3_1" for source in sources):
        raise ValueError("Expected the eight-task SONIC-only merge; TWIST2/all_16 is unsupported")


def _check_episode(
    ds: WBDataset, episode: dict, state_indices: list[int], action_indices: list[int]
) -> tuple[np.ndarray, int]:
    states, actions, state_mask, action_mask = ds._read_sliced_episode_arrays(episode)
    if states.shape != (episode["_length"], 64) or actions.shape != (episode["_length"], 40):
        raise ValueError(f"Episode {episode['episode_index']} has unexpected state/action shape")
    if state_mask.any() or action_mask.any() or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"Episode {episode['episode_index']} has invalid values or masks")
    if not np.isin(actions[:, 38:40], [0.0, 1.0]).all():
        raise ValueError(f"Episode {episode['episode_index']} has non-binary hand commands")
    numeric = ds._build_numeric_window(episode, 0)
    np.testing.assert_allclose(numeric["proprio"][0, state_indices].numpy(), states[0])
    np.testing.assert_allclose(numeric["action"][0, action_indices].numpy(), actions[0])
    missing = sorted(set(range(96)) - set(action_indices))
    if (
        numeric["action"][0, missing].count_nonzero()
        or not numeric["action_semantic_dim_is_pad"][0, missing].all()
    ):
        raise ValueError(f"Episode {episode['episode_index']} has invalid zero-filled model slots")
    return actions[:, 38:40].sum(axis=0), len(actions)


def validate(root: Path, *, full: bool = False, decode_video: bool = False) -> None:
    root = _source_root(root)
    _check_release(root)
    data_cfg = OmegaConf.load(DATA_CONFIG)
    state_indices = list(data_cfg.state_model_dim_indices)
    action_indices = list(data_cfg.action_model_dim_indices)
    if len(state_indices) != 64 or len(action_indices) != 40:
        raise ValueError("Native50 model-slot maps do not match 64D state / 40D action")

    print(f"Official HumanoidArena SONIC v3.1: {root}")
    print("Online adaptation: 50 Hz unchanged; 64D/40D -> 96D model slots; front RGB -> 224x224")
    for task, task_index in TASK_INDICES.items():
        task_cfg = OmegaConf.load(TASK_CONFIGS / f"{task}.yaml")
        ranges = OmegaConf.to_container(task_cfg.arena_exclude_ranges, resolve=True)
        ds = WBDataset(
            root=str(root),
            name=f"humanoid_arena_official_native_50hz_{task}",
            num_frames=int(data_cfg.num_frames),
            action_video_freq_ratio=int(data_cfg.action_video_freq_ratio),
            video_size=(224, 224),
            camera_key=str(data_cfg.camera_key),
            state_dim=int(data_cfg.state_dim),
            action_dim=int(data_cfg.action_dim),
            proprio_dim=int(data_cfg.proprio_dim),
            action_target_dim=int(data_cfg.action_target_dim),
            state_model_dim_indices=state_indices,
            action_model_dim_indices=action_indices,
            dataset_format=str(data_cfg.dataset_format),
            state_key=str(data_cfg.state_key),
            action_key=str(data_cfg.action_key),
            state_mask_key=None,
            action_mask_key=None,
            action_representation=str(data_cfg.action_representation),
            exclude_episode_ranges=ranges,
            val_set_proportion=0.0,
            use_text_embed_cache=False,
            strict_video_decode=decode_video,
        )
        expected = list(range(task_index * 100, (task_index + 1) * 100))
        actual = [int(episode["episode_index"]) for episode in ds.episodes]
        if actual != expected:
            raise ValueError(f"{task}: expected official episodes {expected[0]}..{expected[-1]}")
        if any(ds._episode_task(episode)[0] != task_index for episode in ds.episodes):
            raise ValueError(f"{task}: episode task_index does not match its preset")
        for episode in ds.episodes:
            ds._resolve_video_path(episode)
        selected = ds.episodes if full else (ds.episodes[0], ds.episodes[-1])
        hand_positives = np.zeros(2, dtype=np.float64)
        checked_rows = 0
        for episode in selected:
            positives, rows = _check_episode(ds, episode, state_indices, action_indices)
            hand_positives += positives
            checked_rows += rows
        if decode_video:
            sample = ds[0]
            video = sample["video"]
            if (
                tuple(video.shape) != (3, 9, 224, 224)
                or not video.isfinite().all()
                or sample["image_is_pad"].any()
                or video.std() < 1.0e-3
            ):
                raise ValueError(f"{task}: video decode/resize failed")
        print(f"  {task}: 100 episodes, {len(ds):,} frames, {len(selected)} numeric episodes checked")
        if full:
            left, right = hand_positives / checked_rows
            print(f"    binary hand positive rates: left={left:.4%}, right={right:.4%}")
    print("PASS: seven training tasks are compatible with the native50 presets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Official SONIC subdataset or downloaded repo root"
    )
    parser.add_argument("--full", action="store_true", help="Check every frame in all 700 selected episodes")
    parser.add_argument("--decode-video", action="store_true", help="Decode one 9-frame 224px clip per task")
    args = parser.parse_args()
    validate(args.root, full=args.full, decode_video=args.decode_video)


if __name__ == "__main__":
    main()
