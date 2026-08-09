"""High-level P-MESA orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from .attribution import PathAttribution, ScalarScore, path_integrated_gradients
from .evidence import EvidenceUnit
from .paths import generate_paths
from .subset import ObjectiveWeights, SubsetObjective, greedy_select


@dataclass(frozen=True)
class Explanation:
    units: Sequence[EvidenceUnit]
    selected_indices: Sequence[int]
    saliency: np.ndarray
    attribution: PathAttribution

    @property
    def selected_units(self) -> list[EvidenceUnit]:
        return [self.units[i] for i in self.selected_indices]


class PMESAExplainer:
    """Run path attribution and compact subset selection.

    The adapter supplies a differentiable scalar ``score(state)`` where state
    contains one restoration gate per evidence unit. This narrow contract makes
    the core usable with ALBEF, BLIP-family models, or controlled test models.
    """

    def __init__(
        self,
        *,
        steps: int = 50,
        path_count: int = 6,
        budget: int = 5,
        weights: ObjectiveWeights = ObjectiveWeights(),
        seed: int = 0,
    ) -> None:
        self.steps = steps
        self.path_count = path_count
        self.budget = budget
        self.weights = weights
        self.seed = seed

    def explain(
        self,
        units: Sequence[EvidenceUnit],
        score: ScalarScore,
        saliency: Sequence[float] | Callable[[], Sequence[float]],
    ) -> Explanation:
        if not units:
            raise ValueError("cannot explain an empty evidence set")
        raw_saliency = saliency() if callable(saliency) else saliency
        saliency_array = np.asarray(raw_saliency, dtype=float)
        if saliency_array.shape != (len(units),):
            raise ValueError("saliency must contain one value per evidence unit")
        paths = generate_paths(units, steps=self.steps, count=self.path_count, seed=self.seed)
        attribution = path_integrated_gradients(score, paths)
        objective = SubsetObjective(
            units,
            attribution.contribution.cpu().numpy(),
            saliency_array,
            attribution.stability.cpu().numpy(),
            weights=self.weights,
        )
        selected = greedy_select(objective, self.budget)
        return Explanation(units, selected, saliency_array, attribution)

