#!/usr/bin/env python3
"""Play SONIC collector RGB/depth episodes side by side."""

from __future__ import annotations

import argparse
import json
import lzma
import re
import sys
import time
from pathlib import Path

import numpy as np

try:  # Preferred for smooth playback.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - depends on local environment
    Image = None

FRAME_RE = re.compile(r"frame_(\d+)")
RGB_EXTS = (".jpg", ".jpeg", ".png")


def _frame_index(path: Path) -> int:
    match = FRAME_RE.search(path.stem)
    return int(match.group(1)) if match else -1


def _episode_index(path: Path) -> int:
    match = re.search(r"episode_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _normalize_camera(camera: str) -> str:
    camera = camera.strip()
    for prefix in ("color_", "depth_"):
        if camera.startswith(prefix):
            camera = camera[len(prefix) :]
    return camera


def _load_metadata(root: Path) -> dict:
    metadata_path = root / "metadata.json"
    if not metadata_path.exists() and root.name.startswith("episode_"):
        metadata_path = root.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except Exception as exc:
        print(f"[WARN] failed to read metadata {metadata_path}: {exc}", file=sys.stderr)
        return {}


def _resolve_camera(root: Path, camera: str) -> str:
    if camera != "primary":
        return _normalize_camera(camera)
    metadata = _load_metadata(root)
    return _normalize_camera(str(metadata.get("primary_camera") or "d455"))


def _resolve_fps(root: Path, fps: float | None) -> float:
    if fps is not None:
        return fps
    metadata = _load_metadata(root)
    value = metadata.get("capture_fps", 20)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 20.0


def _parse_episode_filter(spec: str) -> set[str] | None:
    spec = spec.strip()
    if not spec:
        return None
    selected: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and not part.startswith("episode_"):
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            for index in range(start, end + 1):
                selected.add(f"episode_{index:06d}")
            continue
        if part.startswith("episode_"):
            selected.add(part)
        else:
            selected.add(f"episode_{int(part):06d}")
    return selected


def _discover_episodes(root: Path, episode_filter: set[str] | None) -> list[Path]:
    root = root.expanduser().resolve()
    if root.name.startswith("episode_"):
        episodes = [root]
    else:
        episodes = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("episode_")]
        episodes.sort(key=_episode_index)
    if episode_filter is not None:
        episodes = [episode for episode in episodes if episode.name in episode_filter]
    return episodes


def _rgb_files(color_dir: Path) -> list[Path]:
    files = [path for path in color_dir.iterdir() if path.is_file() and path.suffix.lower() in RGB_EXTS]
    files.sort(key=_frame_index)
    return files


def _depth_path_for(depth_dir: Path, frame_index: int) -> Path | None:
    stem = f"frame_{frame_index:06d}"
    candidates = (
        depth_dir / f"{stem}.npy.lzma",
        depth_dir / f"{stem}.npy",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_episode_path(episode: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else episode / path


def _data_json_frames(
    episode: Path, camera: str, with_depth: bool
) -> list[tuple[Path, Path | None]] | None:
    """Return per-tick camera references when data.json provides them.

    Exact 50 Hz episodes intentionally reuse some 60 Hz source images. Reading the
    row references preserves that alignment; enumerating the directory would play
    each unique source image once at 60 Hz instead.
    """

    data_path = episode / "data.json"
    if not data_path.exists():
        return None
    try:
        rows = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rows, list):
        return None

    frames: list[tuple[Path, Path | None]] = []
    found_camera_reference = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        cameras = row.get("cameras")
        if not isinstance(cameras, dict):
            continue
        camera_row = cameras.get(camera)
        if not isinstance(camera_row, dict):
            continue
        rgb_path = _resolve_episode_path(episode, camera_row.get("color"))
        if rgb_path is None:
            continue
        found_camera_reference = True
        if not rgb_path.is_file():
            raise FileNotFoundError(
                f"missing RGB frame referenced by data.json: {rgb_path}"
            )
        depth_path = (
            _resolve_episode_path(episode, camera_row.get("depth"))
            if with_depth
            else None
        )
        if depth_path is not None and not depth_path.is_file():
            depth_path = None
        frames.append((rgb_path, depth_path))
    return frames if found_camera_reference else None


def _read_rgb(path: Path) -> np.ndarray:
    if Image is not None:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    if cv2 is None:
        raise RuntimeError("reading RGB frames requires Pillow or OpenCV")
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to read RGB frame: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _read_depth(path: Path) -> np.ndarray:
    if path.suffix == ".lzma":
        with lzma.open(path, "rb") as handle:
            return np.load(handle, allow_pickle=False)
    with path.open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def _depth_to_rgb(
    depth: np.ndarray,
    *,
    depth_min_mm: float,
    depth_max_mm: float,
    auto_depth: bool,
    invert_depth: bool,
) -> np.ndarray:
    depth_f = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_f) & (depth_f > 0)
    if not np.any(valid):
        return np.zeros((*depth_f.shape[:2], 3), dtype=np.uint8)

    if auto_depth:
        lo, hi = np.percentile(depth_f[valid], [1.0, 99.0])
        if hi <= lo:
            lo, hi = float(depth_min_mm), float(depth_max_mm)
    else:
        lo, hi = float(depth_min_mm), float(depth_max_mm)
    if hi <= lo:
        hi = lo + 1.0

    scaled = (np.clip(depth_f, lo, hi) - lo) / (hi - lo)
    if invert_depth:
        scaled = 1.0 - scaled
    gray = (scaled * 255.0).astype(np.uint8)
    gray[~valid] = 0

    if cv2 is not None:
        color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return np.repeat(gray[..., None], 3, axis=2)


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height == target_height:
        return image
    target_width = max(1, int(round(width * target_height / height)))
    if cv2 is not None:
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    if Image is None:
        raise RuntimeError("resizing without OpenCV requires Pillow")
    pil = Image.fromarray(image)
    return np.asarray(pil.resize((target_width, target_height), Image.BILINEAR))


def _compose_panel(rgb: np.ndarray, depth_rgb: np.ndarray | None, label: str) -> np.ndarray:
    if depth_rgb is None:
        panel = rgb
    else:
        depth_rgb = _resize_to_height(depth_rgb, rgb.shape[0])
        panel = np.concatenate([rgb, depth_rgb], axis=1)
    if cv2 is not None:
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.putText(
            panel_bgr,
            label,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel_bgr,
            "q/esc quit | n next episode | space pause",
            (14, panel_bgr.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    return panel


def _episode_frames(episode: Path, camera: str, with_depth: bool) -> tuple[list[tuple[Path, Path | None]], Path, Path]:
    color_dir = episode / f"color_{camera}"
    depth_dir = episode / f"depth_{camera}"
    if not color_dir.is_dir():
        raise FileNotFoundError(f"missing RGB directory: {color_dir}")
    referenced_frames = _data_json_frames(episode, camera, with_depth)
    if referenced_frames is not None:
        if not referenced_frames:
            raise FileNotFoundError(
                f"no RGB frames referenced for camera {camera} in data.json"
            )
        return referenced_frames, color_dir, depth_dir
    rgb_paths = _rgb_files(color_dir)
    if not rgb_paths:
        raise FileNotFoundError(f"no RGB frames found in {color_dir}")
    frames: list[tuple[Path, Path | None]] = []
    for rgb_path in rgb_paths:
        frame_index = _frame_index(rgb_path)
        depth_path = _depth_path_for(depth_dir, frame_index) if with_depth and depth_dir.is_dir() else None
        frames.append((rgb_path, depth_path))
    return frames, color_dir, depth_dir


def _print_dry_run(episodes: list[Path], camera: str, with_depth: bool) -> None:
    for episode in episodes:
        try:
            frames, color_dir, depth_dir = _episode_frames(episode, camera, with_depth)
        except Exception as exc:
            print(f"{episode.name}: ERROR {exc}")
            continue
        depth_count = sum(1 for _, depth_path in frames if depth_path is not None)
        print(
            f"{episode.name}: rgb_frames={len(frames)} depth_frames={depth_count} "
            f"color_dir={color_dir.name} depth_dir={depth_dir.name}"
        )


def _play_with_cv2(
    episodes: list[Path],
    *,
    camera: str,
    fps: float,
    with_depth: bool,
    depth_min_mm: float,
    depth_max_mm: float,
    auto_depth: bool,
    invert_depth: bool,
    max_width: int,
) -> None:
    assert cv2 is not None
    delay_ms = max(1, int(round(1000.0 / fps)))
    window = "SONIC dataset viewer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    quit_all = False
    for episode in episodes:
        if quit_all:
            break
        try:
            frames, _, _ = _episode_frames(episode, camera, with_depth)
        except Exception as exc:
            print(f"[WARN] skipping {episode.name}: {exc}", file=sys.stderr)
            continue
        print(f"[INFO] playing {episode.name}: {len(frames)} frame(s), camera={camera}")
        paused = False
        frame_i = 0
        while frame_i < len(frames):
            rgb_path, depth_path = frames[frame_i]
            rgb = _read_rgb(rgb_path)
            depth_rgb = None
            if depth_path is not None:
                depth_rgb = _depth_to_rgb(
                    _read_depth(depth_path),
                    depth_min_mm=depth_min_mm,
                    depth_max_mm=depth_max_mm,
                    auto_depth=auto_depth,
                    invert_depth=invert_depth,
                )
            label = f"{episode.name} {rgb_path.stem} camera={camera}"
            panel = _compose_panel(rgb, depth_rgb, label)
            if max_width > 0 and panel.shape[1] > max_width:
                panel = _resize_to_height(panel, int(round(panel.shape[0] * max_width / panel.shape[1])))
            cv2.imshow(window, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key in (ord("q"), 27):
                quit_all = True
                break
            if key == ord("n"):
                break
            if key == ord(" "):
                paused = not paused
                continue
            frame_i += 1
    cv2.destroyWindow(window)


def _play_with_matplotlib(
    episodes: list[Path],
    *,
    camera: str,
    fps: float,
    with_depth: bool,
    depth_min_mm: float,
    depth_max_mm: float,
    auto_depth: bool,
    invert_depth: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("playback requires OpenCV or matplotlib") from exc

    delay_s = 1.0 / fps
    fig, ax = plt.subplots(num="SONIC dataset viewer")
    image_artist = None
    for episode in episodes:
        try:
            frames, _, _ = _episode_frames(episode, camera, with_depth)
        except Exception as exc:
            print(f"[WARN] skipping {episode.name}: {exc}", file=sys.stderr)
            continue
        print(f"[INFO] playing {episode.name}: {len(frames)} frame(s), camera={camera}")
        for rgb_path, depth_path in frames:
            if not plt.fignum_exists(fig.number):
                return
            rgb = _read_rgb(rgb_path)
            depth_rgb = None
            if depth_path is not None:
                depth_rgb = _depth_to_rgb(
                    _read_depth(depth_path),
                    depth_min_mm=depth_min_mm,
                    depth_max_mm=depth_max_mm,
                    auto_depth=auto_depth,
                    invert_depth=invert_depth,
                )
            panel = _compose_panel(rgb, depth_rgb, f"{episode.name} {rgb_path.stem} camera={camera}")
            if image_artist is None:
                image_artist = ax.imshow(panel)
                ax.axis("off")
            else:
                image_artist.set_data(panel)
            ax.set_title(f"{episode.name} {rgb_path.stem} camera={camera}")
            fig.canvas.draw_idle()
            plt.pause(delay_s)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-play SONIC collector RGB/depth episodes.")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Task directory containing episode_* folders, or a single episode directory.",
    )
    parser.add_argument(
        "--camera",
        default="primary",
        help="Camera key to play: primary, d455, d435, color_d455, etc. Default: primary from metadata.",
    )
    parser.add_argument(
        "--episodes",
        default="",
        help="Comma/range filter, e.g. 0,3,5-8,episode_000015. Default: all episodes.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS. Default: metadata capture_fps or 20.")
    parser.add_argument("--rgb-only", action="store_true", help="Only show RGB frames, no depth panel.")
    parser.add_argument("--depth-min-mm", type=float, default=200.0, help="Depth visualization lower clip.")
    parser.add_argument("--depth-max-mm", type=float, default=3000.0, help="Depth visualization upper clip.")
    parser.add_argument("--depth-auto", action="store_true", help="Use per-frame 1/99 percentile depth scaling.")
    parser.add_argument("--no-invert-depth", action="store_true", help="Do not make near depth brighter than far depth.")
    parser.add_argument("--max-width", type=int, default=1600, help="Resize display panel if wider than this; 0 disables.")
    parser.add_argument("--dry-run", action="store_true", help="List episodes/frame counts without opening a viewer.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = args.dataset.expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    camera = _resolve_camera(dataset, args.camera)
    fps = _resolve_fps(dataset, args.fps)
    if fps <= 0.0:
        raise ValueError("--fps must be > 0")
    episodes = _discover_episodes(dataset, _parse_episode_filter(args.episodes))
    if not episodes:
        raise RuntimeError(f"no episodes found under {dataset}")

    with_depth = not args.rgb_only
    print(f"[INFO] dataset={dataset}")
    print(f"[INFO] episodes={len(episodes)} camera={camera} fps={fps:g} depth={with_depth}")
    if args.dry_run:
        _print_dry_run(episodes, camera, with_depth)
        return 0

    if cv2 is not None:
        _play_with_cv2(
            episodes,
            camera=camera,
            fps=fps,
            with_depth=with_depth,
            depth_min_mm=args.depth_min_mm,
            depth_max_mm=args.depth_max_mm,
            auto_depth=args.depth_auto,
            invert_depth=not args.no_invert_depth,
            max_width=args.max_width,
        )
    else:
        _play_with_matplotlib(
            episodes,
            camera=camera,
            fps=fps,
            with_depth=with_depth,
            depth_min_mm=args.depth_min_mm,
            depth_max_mm=args.depth_max_mm,
            auto_depth=args.depth_auto,
            invert_depth=not args.no_invert_depth,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
