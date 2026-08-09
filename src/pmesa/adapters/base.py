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
    remain differentiable with respect to every restoration coordinate. If a
    model cannot expose an independent relation gate, relation units should be
    scored with a documented cooperative-interaction estimator rather than
    silently treated as native model inputs.
    """

    @abstractmethod
    def prepare(self, example: object) -> AdapterOutput:
        raise NotImplementedError

    @staticmethod
    def freeze(model: torch.nn.Module) -> None:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

