import tyro
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass
from tqdm import tqdm
from tracking.convert_qpos2kpt import Args as CvtArgs, run_pipeline


@dataclass
class Cfgs:
    src_dir: str
    save_dir: str
    fix_freq_src: int | None = None
    freq_tgt: int = 50
    smooth_in: float = 0.0
    num_workers: int = 32
    non_flip: bool = False
    aug_freq_ratio: int = 1
    aug_freq_range: float = 0.1


def _worker(job):
    """Worker function for processing a single file, runs in a subprocess."""
    args, raw_qpos_path_str, src_dir, save_dir, freq_tgt = job
    raw_qpos_path = Path(raw_qpos_path_str)
    rel_path = raw_qpos_path.relative_to(Path(src_dir))
    save_path = Path(save_dir) / (str(rel_path).replace("/", "_"))

    run_pipeline(
        CvtArgs(
            mocap_npz=str(raw_qpos_path),
            save_path=str(save_path),
            fix_freq_src=args.fix_freq_src,
            freq_tgt=freq_tgt,
            aug_flip=not args.non_flip,
            aug_freq_ratio=args.aug_freq_ratio,
            aug_freq_range=args.aug_freq_range,
            interp_sec=args.smooth_in,
        )
    )


def main(args):
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
        (args, str(file_path), str(src_dir), str(save_dir), args.freq_tgt)
        for file_path in all_files
    ]

    # Multi-process parallel execution
    ctx = mp.get_context("spawn")  # More robust, especially when involving JAX / PyTorch libraries
    with ctx.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_worker, jobs),
            total=len(jobs),
            desc="Converting",
            unit="file",
            dynamic_ncols=True,
        ):
            pass


if __name__ == "__main__":
    main(tyro.cli(Cfgs))
