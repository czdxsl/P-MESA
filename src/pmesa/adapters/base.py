"""Contract between P-MESA and task-specific target models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from ..attribution import ScalarScore
from ..evidence import EvidenceUnit


@dataclass(frozen=True)
class AdapterOutput:
    units: Sequence[EvidenceUnit]
    score: ScalarScore
    saliency: Sequence[float]


class PMESAAdapter(ABC):
    """Prepare one dataset example for the model-agnostic explainer.

    Implementations must keep target-model parameters frozen. ``score`` must
    remain differentiable with respect to every primitive visual or textual
    restoration coordinate. Relation units must reference primitive endpoints;
    the core derives their contributions from four model evaluations and never
    passes a synthetic relation gate to the target model.
    """

    @abstractmethod
    def prepare(self, example: object) -> AdapterOutput:
        raise NotImplementedError

    @staticmethod
    def freeze(model: torch.nn.Module) -> None:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
