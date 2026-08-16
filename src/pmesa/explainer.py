"""High-level P-MESA orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from .attribution import PathAttribution, ScalarScore, path_integrated_gradients, relation_path_attribution
from .evidence import EvidenceKind, EvidenceUnit
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
    contains one restoration gate per primitive visual or textual unit.
    Cross-modal relations are derived from endpoint interactions.
    """

    def __init__(
        self,
        *,
        steps: int = 50,
        path_count: int = 6,
        budget: int = 5,
        weights: ObjectiveWeights = ObjectiveWeights(),
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        self.steps = steps
        self.path_count = path_count
        self.budget = budget
        self.weights = weights
        self.seed = seed
        self.device = torch.device(device) if device is not None else None

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
        ids = [unit.id for unit in units]
        if len(set(ids)) != len(ids):
            raise ValueError("evidence ids must be unique")
        primitive_positions = [i for i, unit in enumerate(units) if unit.is_primitive]
        relation_positions = [i for i, unit in enumerate(units) if unit.kind is EvidenceKind.RELATION]
        primitive_units = [units[i] for i in primitive_positions]
        if not primitive_units:
            raise ValueError("at least one primitive evidence unit is required")
        primitive_index = {unit.id: i for i, unit in enumerate(primitive_units)}
        primitive_by_id = {unit.id: unit for unit in primitive_units}
        relation_endpoints: list[tuple[int, int]] = []
        for position in relation_positions:
            endpoints = units[position].endpoints
            assert endpoints is not None
            try:
                if primitive_by_id[endpoints[0]].kind is not EvidenceKind.TEXTUAL:
                    raise ValueError("the first relation endpoint must be textual")
                if primitive_by_id[endpoints[1]].kind is not EvidenceKind.VISUAL:
                    raise ValueError("the second relation endpoint must be visual")
                relation_endpoints.append((primitive_index[endpoints[0]], primitive_index[endpoints[1]]))
            except KeyError as error:
                raise ValueError(f"unknown relation endpoint: {error.args[0]}") from error

        paths = generate_paths(primitive_units, steps=self.steps, count=self.path_count, seed=self.seed)
        primitive_attribution = path_integrated_gradients(score, paths, device=self.device)
        relation_per_path = relation_path_attribution(
            score, paths, relation_endpoints, device=self.device
        )
        per_path = torch.zeros(
            (len(paths), len(units)),
            device=primitive_attribution.per_path.device,
            dtype=primitive_attribution.per_path.dtype,
        )
        per_path[:, primitive_positions] = primitive_attribution.per_path
        if relation_positions:
            per_path[:, relation_positions] = relation_per_path
        variance = per_path.var(dim=0, unbiased=False) if len(paths) > 1 else torch.zeros_like(per_path[0])
        attribution = PathAttribution(
            per_path=per_path,
            contribution=per_path.mean(dim=0),
            stability=1.0 / (1.0 + variance),
            completeness_error=primitive_attribution.completeness_error,
            primitive_count=len(primitive_units),
        )
        objective = SubsetObjective(
            units,
            attribution.contribution.cpu().numpy(),
            saliency_array,
            attribution.stability.cpu().numpy(),
            weights=self.weights,
            budget=self.budget,
        )
        selected = greedy_select(objective, self.budget)
        return Explanation(units, selected, saliency_array, attribution)
