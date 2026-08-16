"""One-to-one evidence matching for robustness evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def match_visual_masks(
    first: Sequence[np.ndarray], second: Sequence[np.ndarray], *, threshold: float = 0.5
) -> dict[int, int]:
    candidates = []
    for i, left in enumerate(first):
        for j, right in enumerate(second):
            left, right = np.asarray(left, bool), np.asarray(right, bool)
            if left.shape != right.shape:
                raise ValueError("visual masks must use the same coordinate system")
            union = np.logical_or(left, right).sum()
            score = 0.0 if union == 0 else float(np.logical_and(left, right).sum() / union)
            if score >= threshold:
                candidates.append((score, i, j))
    matches: dict[int, int] = {}
    used_second: set[int] = set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in matches and j not in used_second:
            matches[i] = j
            used_second.add(j)
    return matches


def match_text_embeddings(
    first: np.ndarray, second: np.ndarray, *, threshold: float = 0.8
) -> dict[int, int]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("text embeddings must be matrices with a shared dimension")
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    similarity = first @ second.T
    candidates = [
        (float(similarity[i, j]), i, j)
        for i in range(first.shape[0])
        for j in range(second.shape[0])
        if similarity[i, j] >= threshold
    ]
    matches: dict[int, int] = {}
    used_second: set[int] = set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in matches and j not in used_second:
            matches[i] = j
            used_second.add(j)
    return matches
