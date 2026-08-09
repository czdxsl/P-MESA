"""Faithfulness, localization, and stability metrics."""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np


def auc(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=float)
    if values.size < 2:
        raise ValueError("AUC requires at least two points")
    return float(np.trapz(values, dx=1.0 / (values.size - 1)))


def insertion_deletion_curves(
    score_subset: Callable[[set[int]], float], ordering: Sequence[int], universe_size: int
) -> tuple[list[float], list[float]]:
    inserted: set[int] = set()
    retained = set(range(universe_size))
    insertion = [float(score_subset(inserted))]
    deletion = [float(score_subset(retained))]
    for index in ordering:
        inserted.add(index)
        retained.discard(index)
        insertion.append(float(score_subset(inserted)))
        deletion.append(float(score_subset(retained)))
    return insertion, deletion


def sufficiency_comprehensiveness(
    score_subset: Callable[[set[int]], float], selected: Iterable[int], universe_size: int
) -> tuple[float, float]:
    selected = set(selected)
    full = set(range(universe_size))
    full_score = float(score_subset(full))
    return full_score - float(score_subset(selected)), full_score - float(score_subset(full - selected))


def jaccard(a: Iterable[object], b: Iterable[object]) -> float:
    a, b = set(a), set(b)
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def mean_pairwise_stability(subsets: Sequence[Iterable[object]]) -> float:
    pairs = list(combinations(subsets, 2))
    return 1.0 if not pairs else float(np.mean([jaccard(a, b) for a, b in pairs]))


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    union = np.logical_or(prediction, target).sum()
    return 1.0 if union == 0 else float(np.logical_and(prediction, target).sum() / union)


def pointing_game(heatmap: np.ndarray, target_mask: np.ndarray) -> float:
    """Return one when the maximum-attribution pixel lies in the target mask."""
    heatmap = np.asarray(heatmap, dtype=float)
    target_mask = np.asarray(target_mask, dtype=bool)
    if heatmap.shape != target_mask.shape or heatmap.size == 0:
        raise ValueError("heatmap and target mask must have the same non-empty shape")
    if not np.isfinite(heatmap).all():
        raise ValueError("heatmap must contain only finite values")
    return float(target_mask.flat[int(np.argmax(heatmap))])


def span_f1(predicted: Iterable[int], target: Iterable[int]) -> tuple[float, float, float]:
    predicted, target = set(predicted), set(target)
    overlap = len(predicted & target)
    precision = overlap / len(predicted) if predicted else float(not target)
    recall = overlap / len(target) if target else float(not predicted)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
