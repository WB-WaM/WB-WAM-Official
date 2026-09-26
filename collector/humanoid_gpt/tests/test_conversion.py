"""Exercise offline export from a checkout without the legacy datasets tree."""

import json
import lzma
from pathlib import Path
import shutil
import subprocess
import sys

import imageio.v2 as iio
import numpy as np
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    files = [
        "collector/humanoid_gpt/__init__.py",
        "collector/humanoid_gpt/scripts/convert_hgpt_to_lerobot_v3.py",
        "collector/humanoid_gpt/processing/__init__.py",
        "collector/humanoid_gpt/processing/lerobot_helpers.py",
        "bridge/sonic/joint_order.py",
        "tracker/humanoid_gpt/deploy/reference_alignment.py",
    ]
    for name in files:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, destination)
    assert not (root / "datasets").exists()
    return root


def make_episode(root):
    episode = root / "episode_000000"
    episode.mkdir(parents=True)
    frames = []
    for index in range(4):
        iio.imwrite(episode / f"{index}.png", np.full((32, 32, 3), index * 60, dtype=np.uint8))
        with lzma.open(episode / f"{index}.npy.lzma", "wb") as handle:
            np.save(handle, np.full((32, 32), 1000 + index * 500, dtype=np.uint16))
        derived = {
            "g1_qpos": [2 + index * 0.1, 3, 0.8, 1, 0, 0, 0] + [0] * 29,
            "body_valid": True,
        }
        feedback = {}
        for side in ("left", "right"):
            derived[f"{side}_wuji_qpos"] = [index * 0.1] * 20
            derived[f"{side}_wuji_qpos_valid"] = True
            feedback[f"{side}_wuji_qpos_actual"] = [index * 0.05] * 20
            feedback[f"{side}_actual_position_valid"] = True
        frames.append({
            "frame_index": index,
            "time_ns": index * 50_000_000,
            "primary_camera": "head",
            "cameras": {"head": {"color": f"{index}.png", "depth": f"{index}.npy.lzma"}},
            "human_derived": derived,
            "hand_feedback": feedback,
            "obs": {
                "base_quat": [1, 0, 0, 0], "base_gravity": [0, 0, -1],
                "base_ang_vel": [0, 0, 0], "base_accel": [0, 0, 0],
                "body_q": [0] * 29, "body_dq": [0] * 29,
            },
        })
    (episode / "data.json").write_text(json.dumps(frames))
    return root


@pytest.mark.parametrize("depth", [False, True])
def test_export_without_datasets(checkout, tmp_path, depth):
    source = make_episode(tmp_path / "raw")
    output = tmp_path / "export"
    command = [
        sys.executable,
        str(checkout / "collector/humanoid_gpt/scripts/convert_hgpt_to_lerobot_v3.py"),
        "--source-root", str(source), "--output-root", str(output),
        "--task", "Move the box.", "--num-workers", "1",
    ]
    if depth:
        command.append("--include-depth")
    # Running outside the checkout also verifies direct-script import resolution.
    for extra in (["--dry-run"], [], ["--resume"]):
        result = subprocess.run(command + extra, cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        if extra == ["--dry-run"]:
            assert not output.exists()
    table = pq.read_table(next((output / "data").rglob("*.parquet")))
    assert table.num_rows == 3
    states = np.asarray(table["states"].to_pylist())
    actions = np.asarray(table["action"].to_pylist())
    assert states.shape == (3, 146)
    assert actions.shape == (3, 136)
    # State is frame t; targets are frame t+1, with initial root XY aligned to zero.
    np.testing.assert_allclose(states[:, 110], [0, 0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(actions[:, 0], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(actions[:, 64], [0.1, 0.2, 0.3], atol=1e-6)
    stats = json.loads((output / "meta/episodes_stats.jsonl").read_text())["stats"]
    for name, values in (("states", states), ("action", actions)):
        assert stats[name]["count"] == 3
        for key, reduction in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)):
            np.testing.assert_allclose(stats[name][key], reduction(values, axis=0), atol=1e-8)
    videos = list((output / "videos").rglob("*.mp4"))
    assert len(videos) == (2 if depth else 1)
    for video in videos:
        reader = iio.get_reader(video)
        try:
            assert reader.count_frames() == 3
            assert reader.get_meta_data()["fps"] == 20
            if "d455_depth" in str(video):
                np.testing.assert_allclose(reader.get_data(0), 51, atol=3)
        finally:
            reader.close()
    assert not (checkout / "datasets").exists()
