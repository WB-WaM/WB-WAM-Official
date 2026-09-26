#!/usr/bin/env python3
"""Process one completed Xperience raw episode manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from minimal_xperience_pipeline import (  # noqa: E402
    KEEP_FILENAMES,
    convert_episode_dir,
    delete_episode_raw_files,
    processed_output_complete,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def files_from_manifest(manifest: Mapping[str, Any]) -> dict[str, tuple[str, int]]:
    kept = manifest.get("kept_files")
    if not isinstance(kept, Mapping):
        raise ValueError("manifest is missing kept_files")
    out: dict[str, tuple[str, int]] = {}
    for filename in sorted(KEEP_FILENAMES):
        item = kept.get(filename)
        if not isinstance(item, Mapping):
            raise ValueError(f"manifest is missing kept file entry: {filename}")
        hf_path = item.get("hf_path")
        expected_size = item.get("expected_size", 0)
        if not isinstance(hf_path, str) or not hf_path:
            raise ValueError(f"manifest kept_files.{filename}.hf_path is invalid")
        out[filename] = (hf_path, int(expected_size or 0))
    return out


def run_process(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = read_json(manifest_path)
    episode_id = str(manifest.get("episode_id") or "")
    if not episode_id:
        raise ValueError(f"manifest is missing episode_id: {manifest_path}")
    raw_root_value = manifest.get("raw_root") or (str(args.raw_root) if args.raw_root else None)
    if not raw_root_value:
        raise ValueError("raw root must be provided by manifest or --raw-root")
    raw_root = Path(str(raw_root_value)).expanduser().resolve()
    episode_dir_value = manifest.get("episode_dir")
    if not episode_dir_value:
        raise ValueError("manifest is missing episode_dir")
    episode_dir = Path(str(episode_dir_value)).expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / episode_id
    episode_files = files_from_manifest(manifest)

    complete = processed_output_complete(output_dir)
    if complete and not args.overwrite:
        print(f"[INFO] skip complete processed episode {episode_id}: {output_dir}")
    else:
        overwrite = bool(args.overwrite or (output_dir.exists() and not complete))
        if output_dir.exists() and not complete and not args.overwrite:
            print(f"[INFO] reprocessing incomplete output: {output_dir}")
        convert_episode_dir(
            episode_dir,
            output_dir,
            resize_videos=args.resize_videos,
            require_gmr=args.require_gmr,
            gmr_root=args.gmr_root,
            gmr_python=args.gmr_python,
            smplx_folder=args.smplx_folder,
            gmr_robot=args.gmr_robot,
            wuji_root=args.wuji_root,
            wuji_python=args.wuji_python,
            wuji_left_config=args.wuji_left_config,
            wuji_right_config=args.wuji_right_config,
            require_wuji=args.require_wuji,
            encoder_model=args.encoder_model,
            min_free_gb=args.min_free_gb,
            overwrite=overwrite,
            keep_intermediate=args.keep_intermediate,
        )
        if not processed_output_complete(output_dir):
            raise RuntimeError(f"processed output did not pass completion check: {output_dir}")

    if args.delete_raw_after_parse:
        deleted = delete_episode_raw_files(raw_root, episode_files)
        print(f"[INFO] deleted raw files for {episode_id}: files={len(deleted)}")
    print(f"[INFO] process complete {episode_id}: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--delete-raw-after-parse", action="store_true")
    parser.add_argument("--resize-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-gmr", action="store_true")
    parser.add_argument("--gmr-root", type=Path, default=None)
    parser.add_argument("--gmr-python", type=Path, default=None)
    parser.add_argument("--smplx-folder", type=Path, default=None)
    parser.add_argument("--gmr-robot", type=str, default="unitree_g1")
    parser.add_argument("--wuji-root", type=Path, default=None)
    parser.add_argument("--wuji-python", type=Path, default=None)
    parser.add_argument("--wuji-left-config", type=Path, default=None)
    parser.add_argument("--wuji-right-config", type=Path, default=None)
    parser.add_argument("--require-wuji", action="store_true")
    parser.add_argument("--encoder-model", type=Path, default=None)
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(run_process(args))


if __name__ == "__main__":
    raise SystemExit(main())
