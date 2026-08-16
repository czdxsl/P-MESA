"""Reusable primitive-gated multimodal score construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..baselines import restore_regions, weaken_text_embeddings


@dataclass
class GatedMultimodalScore:
    """Map primitive restoration gates to a target-model scalar score."""

    original_image: torch.Tensor
    baseline_image: torch.Tensor
    visual_masks: torch.Tensor
    original_text_embeddings: torch.Tensor
    weakened_text_embeddings: torch.Tensor
    token_groups: list[list[int]]
    target_score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    @property
    def primitive_count(self) -> int:
        return self.visual_masks.shape[0] + len(self.token_groups)

    def __call__(self, gates: torch.Tensor) -> torch.Tensor:
        if gates.ndim != 1 or gates.numel() != self.primitive_count:
            raise ValueError("gate vector does not match primitive evidence count")
        visual_count = self.visual_masks.shape[0]
        image = restore_regions(
            self.baseline_image,
            self.original_image,
            self.visual_masks,
            gates[:visual_count],
        )
        text = weaken_text_embeddings(
            self.original_text_embeddings,
            self.weakened_text_embeddings,
            self.token_groups,
            gates[visual_count:],
        )
        value = self.target_score(image, text)
        if value.ndim != 0:
            raise ValueError("target_score must return a scalar tensor")
        return value
