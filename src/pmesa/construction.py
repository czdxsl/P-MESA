"""Visual and textual evidence construction utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from .evidence import EvidenceKind, EvidenceUnit


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = np.logical_or(first, second).sum()
    return 0.0 if union == 0 else float(np.logical_and(first, second).sum() / union)


def filter_sam_masks(
    masks: Sequence[dict[str, Any]],
    image_shape: tuple[int, int],
    *,
    min_area_fraction: float = 0.01,
    nms_iou: float = 0.8,
) -> list[np.ndarray]:
    """Apply the Appendix B.1 area filter and mask NMS."""
    if not 0 <= min_area_fraction < 1 or not 0 < nms_iou <= 1:
        raise ValueError("invalid SAM filtering thresholds")
    image_area = image_shape[0] * image_shape[1]
    candidates = []
    for item in masks:
        mask = np.asarray(item["segmentation"], dtype=bool)
        if mask.shape != image_shape:
            raise ValueError("SAM mask shape does not match the image")
        area = int(mask.sum())
        if area >= min_area_fraction * image_area:
            candidates.append((float(item.get("predicted_iou", 0.0)), area, mask))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    retained: list[np.ndarray] = []
    for _, _, mask in candidates:
        if all(mask_iou(mask, previous) < nms_iou for previous in retained):
            retained.append(mask)
    return retained


def patch_masks(image_shape: tuple[int, int], grid: tuple[int, int]) -> list[np.ndarray]:
    """Return ViT-style fallback patches when SAM produces no valid region."""
    height, width = image_shape
    rows, columns = grid
    if min(height, width, rows, columns) < 1:
        raise ValueError("image and grid dimensions must be positive")
    y_edges = np.linspace(0, height, rows + 1).round().astype(int)
    x_edges = np.linspace(0, width, columns + 1).round().astype(int)
    output = []
    for row in range(rows):
        for column in range(columns):
            mask = np.zeros((height, width), dtype=bool)
            mask[y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1]] = True
            output.append(mask)
    return output


def visual_evidence_from_sam(
    image: Image.Image | np.ndarray,
    mask_generator: Any,
    *,
    fallback_grid: tuple[int, int] = (24, 24),
) -> list[EvidenceUnit]:
    """Generate frozen-SAM regions with ViT-patch fallback."""
    array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    masks = filter_sam_masks(mask_generator.generate(array), array.shape[:2])
    source = "sam"
    if not masks:
        masks = patch_masks(array.shape[:2], fallback_grid)
        source = "patch"
    return [
        EvidenceUnit(
            id=f"v:{index}",
            kind=EvidenceKind.VISUAL,
            label=f"visual {source} unit {index}",
            semantic_factors={"object": 1.0},
            payload={"mask": mask, "source": source},
        )
        for index, mask in enumerate(masks)
    ]


def textual_evidence_from_groups(
    labels: Sequence[str], token_groups: Sequence[Sequence[int]]
) -> list[EvidenceUnit]:
    """Create phrase-level textual units from parser-provided token groups."""
    if len(labels) != len(token_groups):
        raise ValueError("phrase labels and token groups must have equal length")
    units = []
    for index, (label, group) in enumerate(zip(labels, token_groups)):
        indices = tuple(int(value) for value in group)
        if not label.strip() or not indices:
            raise ValueError("textual phrases require a label and at least one token")
        units.append(EvidenceUnit(
            id=f"t:{index}",
            kind=EvidenceKind.TEXTUAL,
            label=label.strip(),
            semantic_factors={"phrase": 1.0},
            payload={"token_indices": indices},
        ))
    return units
