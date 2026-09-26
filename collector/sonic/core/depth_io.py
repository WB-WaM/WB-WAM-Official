from __future__ import annotations

import lzma
import shutil
from pathlib import Path

import cv2
import numpy as np


def save_depth(depth_path: Path, depth: np.ndarray) -> None:
    with lzma.open(depth_path, "wb") as handle:
        np.save(handle, depth, allow_pickle=False)


def save_raw_depth(depth_path: Path, depth: np.ndarray) -> None:
    with depth_path.open("wb") as handle:
        np.save(handle, depth, allow_pickle=False)


def write_compressed_depth(raw_depth_path: Path, compressed_depth_path: Path) -> None:
    compressed_depth_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = compressed_depth_path.with_name(compressed_depth_path.name + ".tmp")
    try:
        with raw_depth_path.open("rb") as source, lzma.open(temp_path, "wb") as target:
            shutil.copyfileobj(source, target)
        temp_path.replace(compressed_depth_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def compress_raw_depth(raw_depth_path: Path, compressed_depth_path: Path) -> None:
    write_compressed_depth(raw_depth_path, compressed_depth_path)
    raw_depth_path.unlink(missing_ok=True)


def load_depth(depth_path: Path) -> np.ndarray:
    if depth_path.suffix != ".lzma":
        with depth_path.open("rb") as handle:
            return np.load(handle, allow_pickle=False)
    with lzma.open(depth_path, "rb") as handle:
        return np.load(handle, allow_pickle=False)


def depth_to_bgr(depth: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    clipped = depth.astype(np.float32)
    min_val = float(clipped[finite_mask].min())
    max_val = float(clipped[finite_mask].max())
    if max_val <= min_val:
        max_val = min_val + 1.0

    normalized = ((clipped - min_val) / (max_val - min_val) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
