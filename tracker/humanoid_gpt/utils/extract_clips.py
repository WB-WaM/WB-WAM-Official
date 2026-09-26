import tyro
import pickle
import numpy as np
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass


@dataclass
class Args:
    src_dir: str
    save_dir: str
    clip_length: float = 20.0  # clip duration in seconds
    num_workers: int = 32


def load_data(file_path: str):
    """Load data from npz or pkl file."""
    try:
        if file_path.endswith(".npz"):
            data = np.load(file_path, allow_pickle=True)
        elif file_path.endswith(".pkl"):
            data = pickle.load(open(file_path, "rb"))
        else:
            raise ValueError("Only .npz and .pkl files are supported.")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    return data


def _worker(file_path_str: str, src_dir: str, save_dir: str, clip_length: float):
    """Worker function for processing a single file, runs in a subprocess."""
    file_path = Path(file_path_str)
    src_dir = Path(src_dir)
    save_dir = Path(save_dir)

    data = load_data(file_path_str)
    if data is None:
        return 0

    if "qpos" not in data:
        print(f"No qpos in {file_path}, skipping.")
        return 0

    qpos = np.asarray(data["qpos"], dtype=np.float32)
    frequency = float(data["frequency"]) if "frequency" in data else 50.0

    total_frames = len(qpos)
    clip_frames = int(clip_length * frequency)

    print(f"Processing {file_path}: {total_frames} frames, {frequency} Hz, "
          f"{total_frames / frequency:.2f}s total")

    # Generate save name: relative path with / replaced by _
    rel_path = file_path.relative_to(src_dir)
    base_name = str(rel_path).replace("/", "_")

    # If trajectory is shorter than clip_length, export original trajectory directly
    if total_frames < clip_frames:
        save_path = save_dir / base_name
        save_data = {key: np.asarray(data[key]) for key in data.keys()}
        np.savez_compressed(save_path, **save_data)
        print(f"  -> Trajectory shorter than clip_length, saved original: {save_path}")
        return 1

    # Calculate number of complete clips
    num_clips = total_frames // clip_frames
    base_name_no_ext = base_name.replace(".npz", "")

    for i in range(num_clips):
        start_frame = i * clip_frames
        end_frame = start_frame + clip_frames

        # Extract clip data - slice all temporal arrays
        clip_data = {}
        for key in data.keys():
            arr = np.asarray(data[key])
            # Check if array has temporal dimension matching qpos length
            if arr.ndim >= 1 and arr.shape[0] == total_frames:
                clip_data[key] = arr[start_frame:end_frame]
            else:
                # Keep non-temporal data (e.g., frequency) as is
                clip_data[key] = arr

        # Save clip
        clip_name = f"{base_name_no_ext}_clip{i:04d}.npz"
        save_path = save_dir / clip_name
        np.savez_compressed(save_path, **clip_data)

    # Save remaining frames if any
    remaining_start = num_clips * clip_frames
    if remaining_start < total_frames:
        clip_data = {}
        for key in data.keys():
            arr = np.asarray(data[key])
            if arr.ndim >= 1 and arr.shape[0] == total_frames:
                clip_data[key] = arr[remaining_start:]
            else:
                clip_data[key] = arr

        clip_name = f"{base_name_no_ext}_clip{num_clips:04d}.npz"
        save_path = save_dir / clip_name
        np.savez_compressed(save_path, **clip_data)
        num_clips += 1

    print(f"  -> Extracted {num_clips} clips")
    return num_clips


def extract_clips(args: Args):
    """Extract clips from all npz files in src_dir and save to save_dir."""
    print(args)

    src_dir = Path(args.src_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(src_dir.rglob("*.npz"))
    if not all_files:
        print(f"No npz files found in {src_dir}")
        return

    # Pass only simple types to subprocess to avoid pickle issues
    jobs = [
        (str(file_path), str(src_dir), str(save_dir), args.clip_length)
        for file_path in all_files
    ]

    # Multi-process parallel execution
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.num_workers) as pool:
        results = pool.starmap(_worker, jobs)

    total_clips = sum(results)
    print(f"\nDone! Total clips extracted: {total_clips}")


if __name__ == "__main__":
    extract_clips(tyro.cli(Args))

