#!/usr/bin/env python3
"""Render one real sample and its config-built instruction per WB dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import textwrap
from typing import Any

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
import torch

WBWAM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WBWAM_ROOT / "configs/data/wb.yaml"
PANEL_WIDTH = 1600
PANEL_HEIGHT = 520
PAGE_SIZE = 7


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    font_dir = os.environ.get("WBWAM_FONT_DIR")
    if font_dir:
        path = Path(font_dir) / filename
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_dataset(config_path: Path):
    data_cfg = OmegaConf.load(config_path)
    wrapped = OmegaConf.create({"data": data_cfg})
    OmegaConf.resolve(wrapped)
    wrapped.data.compute_norm_stats = False
    wrapped.data.use_text_embed_cache = False
    wrapped.data.val_set_proportion = 0.0
    wrapped.data.is_training_set = True
    wrapped.data.shard_cache_size = 0
    wrapped.data.strict_video_decode = True
    return instantiate(wrapped.data)


def _to_uint8_frame(video: torch.Tensor, frame_index: int) -> Image.Image:
    array = video[:, frame_index].detach().cpu().float().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _video_strip(sample: dict[str, Any], width: int, height: int) -> Image.Image:
    video = sample["video"]
    image_is_pad = sample["image_is_pad"].detach().cpu().bool().numpy()
    if bool(image_is_pad.all()):
        image = Image.new("RGB", (width, height), (235, 238, 242))
        draw = ImageDraw.Draw(image)
        title_font = _font(38, bold=True)
        body_font = _font(24)
        title = "NO VIDEO"
        body = "This is a motion-only sample."
        title_box = draw.textbbox((0, 0), title, font=title_font)
        body_box = draw.textbbox((0, 0), body, font=body_font)
        draw.text(
            ((width - (title_box[2] - title_box[0])) / 2, height / 2 - 52),
            title,
            fill=(65, 72, 82),
            font=title_font,
        )
        draw.text(
            ((width - (body_box[2] - body_box[0])) / 2, height / 2 + 6),
            body,
            fill=(90, 98, 110),
            font=body_font,
        )
        return image

    frame_count = int(video.shape[1])
    indices = sorted({0, frame_count // 2, frame_count - 1})
    frame_width = width // len(indices)
    strip = Image.new("RGB", (width, height), "white")
    for slot, frame_index in enumerate(indices):
        frame = _to_uint8_frame(video, frame_index)
        frame.thumbnail((frame_width - 8, height), Image.Resampling.LANCZOS)
        x = slot * frame_width + (frame_width - frame.width) // 2
        y = (height - frame.height) // 2
        strip.paste(frame, (x, y))
    return strip


def _action_heatmap(sample: dict[str, Any], width: int, height: int) -> Image.Image:
    action = sample["action"].detach().cpu().float().numpy()
    invalid = sample.get("action_dim_is_pad")
    if invalid is None:
        invalid_mask = np.zeros_like(action, dtype=bool)
    else:
        invalid_mask = invalid.detach().cpu().bool().numpy()

    valid_values = action[~invalid_mask]
    if valid_values.size:
        low, high = np.percentile(valid_values, [2.0, 98.0])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(valid_values.min()), float(valid_values.max())
        if high <= low:
            high = low + 1.0
    else:
        low, high = 0.0, 1.0

    normalized = np.clip((action - low) / (high - low), 0.0, 1.0)
    rgb = np.empty((*action.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255.0 * normalized).astype(np.uint8)
    rgb[..., 1] = (150.0 * (1.0 - np.abs(normalized - 0.5) * 2.0)).astype(np.uint8)
    rgb[..., 2] = (255.0 * (1.0 - normalized)).astype(np.uint8)
    rgb[invalid_mask] = np.array([210, 210, 210], dtype=np.uint8)
    heatmap = Image.fromarray(np.transpose(rgb, (1, 0, 2)), mode="RGB")
    return heatmap.resize((width, height), Image.Resampling.NEAREST)


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    xy: tuple[int, int],
    width_chars: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    spacing: int = 5,
) -> int:
    lines: list[str] = []
    for source_line in text.splitlines():
        lines.extend(textwrap.wrap(source_line, width=width_chars) or [""])
    rendered = "\n".join(lines)
    draw.multiline_text(xy, rendered, font=font, fill=fill, spacing=spacing)
    bbox = draw.multiline_textbbox(xy, rendered, font=font, spacing=spacing)
    return bbox[3]


def _render_panel(dataset, sample: dict[str, Any], index: int) -> Image.Image:
    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "white")
    draw = ImageDraw.Draw(panel)
    title_font = _font(28, bold=True)
    text_font = _font(19)
    small_font = _font(16)

    draw.rectangle((0, 0, PANEL_WIDTH, 48), fill=(34, 44, 64))
    title = f"{index + 1:02d}. {dataset.name}"
    draw.text((20, 8), title, fill="white", font=title_font)

    instruction = str(sample["prompt"])
    text_bottom = _draw_wrapped_text(
        draw,
        instruction,
        xy=(20, 62),
        width_chars=142,
        font=text_font,
        fill=(25, 30, 38),
        spacing=4,
    )
    content_y = max(184, text_bottom + 14)
    content_height = 250
    visual_width = 760
    heatmap_width = 760
    visual = _video_strip(sample, visual_width, content_height)
    heatmap = _action_heatmap(sample, heatmap_width, content_height)
    panel.paste(visual, (20, content_y))
    panel.paste(heatmap, (820, content_y))

    draw.rectangle((20, content_y, 20 + visual_width, content_y + content_height), outline=(90, 98, 110), width=2)
    draw.rectangle(
        (820, content_y, 820 + heatmap_width, content_y + content_height), outline=(90, 98, 110), width=2
    )
    draw.text((20, content_y - 22), "Visual observation", fill=(65, 72, 82), font=small_font)
    draw.text(
        (820, content_y - 22),
        "Action trajectory: time left-to-right, dimensions top-to-bottom",
        fill=(65, 72, 82),
        font=small_font,
    )

    image_is_pad = sample["image_is_pad"].detach().cpu().bool()
    action_invalid = sample["action_dim_is_pad"].detach().cpu().bool()
    valid_action_dims = int((~action_invalid[0]).sum().item())
    status = (
        f"episode={sample['episode_index']}  frame={sample['frame_index']}  "
        f"video_valid={not bool(image_is_pad.all())}  "
        f"valid_action_dims={valid_action_dims}/{action_invalid.shape[-1]}"
    )
    draw.text((20, PANEL_HEIGHT - 28), status, fill=(65, 72, 82), font=small_font)
    return panel


def _save_pages(panels: list[Image.Image], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for page_index in range(0, len(panels), PAGE_SIZE):
        page_panels = panels[page_index : page_index + PAGE_SIZE]
        page = Image.new(
            "RGB",
            (PANEL_WIDTH, PANEL_HEIGHT * len(page_panels)),
            (225, 228, 234),
        )
        for row, panel in enumerate(page_panels):
            page.paste(panel, (0, row * PANEL_HEIGHT))
        path = output_dir / f"wb_samples_page_{page_index // PAGE_SIZE + 1}.jpg"
        page.save(path, quality=88, optimize=True)
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mixture = _load_dataset(args.config.expanduser().resolve())

    panels: list[Image.Image] = []
    records: list[dict[str, Any]] = []
    for index, dataset in enumerate(mixture.datasets):
        sample = dataset[0]
        panel = _render_panel(dataset, sample, index)
        panel_path = output_dir / f"{index + 1:02d}_{dataset.name}.jpg"
        panel.save(panel_path, quality=90, optimize=True)
        panels.append(panel)
        record = {
            "dataset": dataset.name,
            "dataset_root": str(dataset.root),
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "video_valid": not bool(sample["image_is_pad"].all()),
            "valid_action_dims": int((~sample["action_dim_is_pad"][0]).sum().item()),
            "instruction": str(sample["prompt"]),
            "visualization": str(panel_path),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    page_paths = _save_pages(panels, output_dir)
    report_path = output_dir / "instruction_examples.json"
    report_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pages": [str(path) for path in page_paths]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
