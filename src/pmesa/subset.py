"""Monotone submodular compact-subset objective and greedy solver."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Mapping, Sequence

import numpy as np

from .evidence import EvidenceUnit


@dataclass(frozen=True)
class ObjectiveWeights:
    contribution: float = 1.0
    saliency: float = 1.0
    coverage: float = 1.0
    stability: float = 1.0

    def __post_init__(self) -> None:
        if min(self.contribution, self.saliency, self.coverage, self.stability) < 0:
            raise ValueError("objective weights must be non-negative")


class SubsetObjective:
    def __init__(
        self,
        units: Sequence[EvidenceUnit],
        contribution: Sequence[float],
        saliency: Sequence[float],
        stability: Sequence[float],
        *,
        weights: ObjectiveWeights = ObjectiveWeights(),
        relation_caps: Mapping[str, float] | None = None,
        normalize: bool = True,
        budget: int | None = None,
    ) -> None:
        n = len(units)
        if not (len(contribution) == len(saliency) == len(stability) == n):
            raise ValueError("all score arrays must match the evidence-unit count")
        self.units = list(units)
        self.contribution = np.maximum(np.asarray(contribution, dtype=float), 0.0)
        self.saliency = np.maximum(np.asarray(saliency, dtype=float), 0.0)
        self.stability = np.maximum(np.asarray(stability, dtype=float), 0.0)
        self.weights = weights
        self.relation_caps = dict(relation_caps or {})
        self.normalize = normalize
        self.budget = n if budget is None else min(max(int(budget), 0), n)
        self._normalizers = self._component_normalizers()

    def raw_components(self, selected: Sequence[int]) -> tuple[float, float, float, float]:
        selected = list(selected)
        con = log1p(float(self.contribution[selected].sum())) if selected else 0.0
        sal = log1p(float(self.saliency[selected].sum())) if selected else 0.0
        factors: dict[str, float] = {}
        relations: dict[str, float] = {}
        for i in selected:
            for key, value in self.units[i].semantic_factors.items():
                factors[key] = factors.get(key, 0.0) + float(value)
            for key, value in self.units[i].relation_types.items():
                relations[key] = relations.get(key, 0.0) + float(value)
        cov = sum(log1p(value) for value in factors.values())
        cov += sum(min(self.relation_caps.get(key, 1.0), value) for key, value in relations.items())
        stab = float(self.stability[selected].sum()) if selected else 0.0
        return con, sal, cov, stab

    def _component_normalizers(self) -> np.ndarray:
        if not self.normalize:
            return np.ones(4, dtype=float)
        if self.budget == 0:
            return np.ones(4, dtype=float)
        con = log1p(float(np.sort(self.contribution)[-self.budget:].sum()))
        sal = log1p(float(np.sort(self.saliency)[-self.budget:].sum()))
        stab = float(np.sort(self.stability)[-self.budget:].sum())
        cov = self.raw_components(range(len(self.units)))[2]
        values = np.asarray((con, sal, cov, stab), dtype=float)
        return np.where(values > 1e-12, values, 1.0)

    def components(self, selected: Sequence[int]) -> tuple[float, float, float, float]:
        values = np.asarray(self.raw_components(selected), dtype=float) / self._normalizers
        return tuple(float(value) for value in values)

    def __call__(self, selected: Sequence[int]) -> float:
        c = self.components(selected)
        w = self.weights
        return w.contribution * c[0] + w.saliency * c[1] + w.coverage * c[2] + w.stability * c[3]


def greedy_select(objective: SubsetObjective, budget: int) -> list[int]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    selected: list[int] = []
    remaining = set(range(len(objective.units)))
    for _ in range(min(budget, len(remaining))):
        current = objective(selected)
        best = max(remaining, key=lambda i: (objective(selected + [i]) - current, -i))
        selected.append(best)
        remaining.remove(best)
    return selected
