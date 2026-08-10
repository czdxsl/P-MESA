"""Consistent qualitative-result rendering for all three tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, *, alpha: float = 0.45) -> Image.Image:
    """Overlay a fixed red-yellow attribution map without per-method styling."""
    heatmap = np.asarray(heatmap, dtype=float)
    lo, hi = float(np.nanmin(heatmap)), float(np.nanmax(heatmap))
    normalized = np.zeros_like(heatmap) if hi - lo <= 1e-12 else (heatmap - lo) / (hi - lo)
    rgb = np.stack([
        np.clip(2.2 * normalized, 0, 1),
        np.clip(2.2 * normalized - 0.7, 0, 1),
        np.clip(2.0 * normalized - 1.5, 0, 1),
    ], axis=-1)
    layer = Image.fromarray(np.uint8(rgb * 255)).resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), layer, alpha)


def draw_selected_regions(image: Image.Image, masks: Sequence[np.ndarray], labels: Sequence[str]) -> Image.Image:
    if len(masks) != len(labels):
        raise ValueError("masks and labels must have equal length")
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i, (mask, label) in enumerate(zip(masks, labels)):
        ys, xs = np.where(np.asarray(mask, bool))
        if xs.size == 0:
            continue
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        color = colors[i % len(colors)]
        draw.rectangle(box, outline=color, width=max(2, image.width // 200))
        draw.text((box[0] + 2, box[1] + 2), label, fill=color)
    return output


def save_method_grid(
    rows: Sequence[tuple[str, Sequence[Image.Image]]],
    column_labels: Sequence[str],
    output: str | Path,
    *,
    padding: int = 12,
) -> None:
    """Save a raster grid with identical image sizes and visible method labels."""
    if not rows or not column_labels:
        raise ValueError("grid needs rows and columns")
    width, height = rows[0][1][0].size
    label_height, row_label_width = 30, 150
    canvas = Image.new("RGB", (row_label_width + len(column_labels) * (width + padding), label_height + len(rows) * (height + padding)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(column_labels):
        draw.text((row_label_width + col * (width + padding), 6), label, fill="black")
    for row_idx, (row_label, images) in enumerate(rows):
        if len(images) != len(column_labels):
            raise ValueError("each grid row must have one image per column")
        y = label_height + row_idx * (height + padding)
        draw.text((4, y + height // 2), row_label, fill="black")
        for col, image in enumerate(images):
            canvas.paste(image.convert("RGB").resize((width, height)), (row_label_width + col * (width + padding), y))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
