"""Small deterministic end-to-end example used for smoke testing."""

from __future__ import annotations

import torch

from .evidence import EvidenceKind, EvidenceUnit
from .explainer import PMESAExplainer


def run_demo(seed: int = 7):
    torch.manual_seed(seed)
    units = [
        EvidenceUnit("v-snow", EvidenceKind.VISUAL, "sunny beach", {"scene": 1.0}),
        EvidenceUnit("v-person", EvidenceKind.VISUAL, "person", {"object": 1.0}),
        EvidenceUnit("t-heavy-snow", EvidenceKind.TEXTUAL, "heavy snow", {"attribute": 1.0}),
        EvidenceUnit("t-city", EvidenceKind.TEXTUAL, "city"),
        EvidenceUnit(
            "r-conflict",
            EvidenceKind.RELATION,
            "snow contradicts beach",
            relation_types={"contradiction": 1.0},
            endpoints=("t-heavy-snow", "v-snow"),
        ),
    ]
    weights = torch.tensor([0.8, 0.1, 0.9, 0.2])

    def score(state: torch.Tensor) -> torch.Tensor:
        return (
            torch.dot(weights, state)
            + 0.7 * state[0] * state[2]
            + 0.5 * state[0] * state[2]
            - 1.5 * state[1] * state[3]
        )

    saliency = [0.82, 0.08, 0.91, 0.15, 0.95]
    return PMESAExplainer(budget=3, seed=seed).explain(units, score, saliency)
