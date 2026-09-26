"""
Parallel headless video recording from npz motion files.

Usage:
    python -m utils.record_video_parallel --src_dir /path/to/npz_folder --save_dir /path/to/video_output --num_workers 8
"""

import os
import sys

# Set environment variables BEFORE importing mujoco (for main process)
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

import tyro
import numpy as np
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass

from tracking import constants as consts
from utils.video_utils import images_to_video


@dataclass
class Cfgs:
    src_dir: str
    save_dir: str
    num_workers: int = 8
    output_fps: int = 30  # Output video framerate (subsample from source)
    video_width: int = 640
    video_height: int = 480
    xml_path: str = str(consts.DEBUG_TRACK_XML)


def record_single_video(
    npz_path: str,
    video_path: str,
    xml_path: str,
    output_fps: int = 30,
    video_width: int = 640,
    video_height: int = 480,
) -> bool:
    """
    Record a single npz file to video (headless).

    Args:
        npz_path: Path to the source npz file
        video_path: Path to save the output video
        xml_path: Path to the MuJoCo XML model
        output_fps: Output video framerate (will subsample if source fps is higher)
        video_width: Width of the video
        video_height: Height of the video

    Returns:
        True if successful, False otherwise
    """
    # Lazy import mujoco AFTER environment variables are set
    import mujoco
    from mujoco import Renderer

    try:
        # Load npz data
        data = np.load(npz_path, allow_pickle=True)

        # Check for qpos data
        if "qpos" not in data:
            print(f"[WARN] No 'qpos' found in {npz_path}, skipping.")
            return False

        qpos = np.asarray(data["qpos"], dtype=np.float32)
        num_steps = len(qpos)

        # Read frequency from data, default to 50Hz if not present
        src_fps = int(data["frequency"]) if "frequency" in data else 50

        if num_steps < 10:
            print(f"[WARN] Too few frames ({num_steps}) in {npz_path}, skipping.")
            return False

        # Calculate frame step for subsampling (e.g., 50Hz -> 30Hz means step ~1.67, round to 2)
        # Use output_fps if it's lower than source, otherwise use source fps
        actual_output_fps = min(output_fps, src_fps)
        frame_step = max(1, src_fps // actual_output_fps)
        actual_output_fps = src_fps // frame_step  # Recalculate actual output fps

        # Initialize MuJoCo model and data
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_model.opt.timestep = 1.0 / src_fps
        mj_data = mujoco.MjData(mj_model)

        # Initialize headless renderer
        renderer = Renderer(mj_model, height=video_height, width=video_width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 0
        cam.azimuth = 90.0
        cam.elevation = -20.0
        cam.distance = 2.5

        # Render frames (with subsampling)
        frames = []
        for t in range(0, num_steps, frame_step):
            # Set qpos (handle different array shapes)
            q = qpos[t]
            mj_data.qpos[: len(q)] = q
            mujoco.mj_forward(mj_model, mj_data)

            # Render frame
            renderer.update_scene(mj_data, cam)
            frame = renderer.render()
            frames.append(frame)

        # Close renderer to release resources
        renderer.close()

        # Save video
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        images_to_video(frames, video_path, fps=actual_output_fps, color_format="RGB")
        print(f"[OK] Saved video: {video_path} ({len(frames)} frames @ {actual_output_fps}Hz, src: {num_steps} frames @ {src_fps}Hz)")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to process {npz_path}: {e}")
        return False


def _worker(
    npz_path_str: str,
    src_dir_str: str,
    save_dir_str: str,
    xml_path: str,
    output_fps: int,
    video_width: int,
    video_height: int,
):
    """Worker function for processing a single file in a subprocess."""
    # Set environment variables in subprocess BEFORE importing mujoco
    # This is critical for spawn mode where subprocesses start fresh
    import os
    import sys
    if sys.platform.startswith("linux"):
        os.environ["MUJOCO_GL"] = "egl"

    npz_path = Path(npz_path_str)
    src_dir = Path(src_dir_str)
    save_dir = Path(save_dir_str)

    # Compute relative path and generate output video path
    rel_path = npz_path.relative_to(src_dir)
    # Replace directory separators with underscores and change extension
    video_name = str(rel_path).replace("/", "_").replace("\\", "_")
    video_name = video_name.rsplit(".", 1)[0] + ".mp4"
    video_path = save_dir / video_name

    record_single_video(
        npz_path=str(npz_path),
        video_path=str(video_path),
        xml_path=xml_path,
        output_fps=output_fps,
        video_width=video_width,
        video_height=video_height,
    )


def main(args: Cfgs):
    print(f"Configuration: {args}")

    src_dir = Path(args.src_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find all npz files
    all_files = sorted(src_dir.rglob("*.npz"))
    if not all_files:
        print(f"No npz files found in {src_dir}")
        return

    print(f"Found {len(all_files)} npz files to process.")

    # Prepare job arguments
    jobs = [
        (
            str(file_path),
            str(src_dir),
            str(save_dir),
            args.xml_path,
            args.output_fps,
            args.video_width,
            args.video_height,
        )
        for file_path in all_files
    ]

    # Multi-process parallel execution
    ctx = mp.get_context("spawn")  # More robust for MuJoCo / GPU contexts
    with ctx.Pool(processes=args.num_workers) as pool:
        pool.starmap(_worker, jobs)

    print(f"Done! Videos saved to: {save_dir}")


if __name__ == "__main__":
    main(tyro.cli(Cfgs))
