from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


PANEL_SIZE = (960, 600)


def fit_panel(image: Image.Image, size: tuple[int, int] = PANEL_SIZE) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "white")
    panel.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return panel


def colorize(image: Image.Image, heat: np.ndarray, scale: float) -> Image.Image:
    value = np.clip(heat / max(scale, 1e-12), 0, 1)
    rgb = np.stack([
        np.clip(2.2 * value, 0, 1),
        np.clip(2.2 * value - 0.7, 0, 1),
        np.clip(2 * value - 1.5, 0, 1),
    ], -1)
    layer = Image.fromarray(np.uint8(rgb * 255)).resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), layer, 0.45)


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fit_panel(image).save(path, format="JPEG", quality=95, subsampling=0)
