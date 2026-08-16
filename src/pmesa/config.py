"""Validated manuscript-default P-MESA configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .subset import ObjectiveWeights


@dataclass(frozen=True)
class PMESAConfig:
    integration_points: int = 50
    restoration_paths: int = 6
    evidence_budget: int = 5
    relation_top_k: int = 3
    seeds: tuple[int, ...] = (17, 23, 42)
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    def __post_init__(self) -> None:
        if self.integration_points < 1:
            raise ValueError("integration_points must be positive")
        if self.restoration_paths < 1 or self.evidence_budget < 1 or self.relation_top_k < 1:
            raise ValueError("path, budget, and relation counts must be positive")
        if not self.seeds:
            raise ValueError("at least one random seed is required")


def load_config(path: str | Path) -> tuple[dict[str, Any], PMESAConfig]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    method = payload.get("pmesa", {})
    weight_values = method.get("objective_weights", {})
    config = PMESAConfig(
        integration_points=int(method.get("integration_points", 50)),
        restoration_paths=int(method.get("restoration_paths", 6)),
        evidence_budget=int(method.get("evidence_budget", 5)),
        relation_top_k=int(method.get("relation_top_k", 3)),
        seeds=tuple(int(value) for value in method.get("seeds", (17, 23, 42))),
        objective_weights=ObjectiveWeights(
            contribution=float(weight_values.get("contribution", 1.0)),
            saliency=float(weight_values.get("saliency", 1.0)),
            coverage=float(weight_values.get("coverage", 1.0)),
            stability=float(weight_values.get("stability", 1.0)),
        ),
    )
    return payload, config
