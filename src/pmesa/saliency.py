"""Dense and evidence-level integrated-gradient saliency."""

from __future__ import annotations

from typing import Callable

import torch


def integrated_gradients(
    score: Callable[[torch.Tensor], torch.Tensor],
    original: torch.Tensor,
    baseline: torch.Tensor,
    *,
    steps: int = 50,
) -> torch.Tensor:
    """Integrated gradients with midpoint quadrature."""
    if original.shape != baseline.shape:
        raise ValueError("original and baseline shapes must match")
    if steps < 1:
        raise ValueError("steps must be positive")
    delta = original - baseline
    total = torch.zeros_like(original)
    with torch.enable_grad():
        for alpha in (torch.arange(steps, device=original.device, dtype=original.dtype) + 0.5) / steps:
            point = (baseline + alpha * delta).detach().requires_grad_(True)
            output = score(point)
            if output.ndim != 0:
                raise ValueError("score function must return a scalar")
            total += torch.autograd.grad(output, point)[0].detach()
    return delta * total / steps


def aggregate_visual_saliency(saliency: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Average absolute saliency inside each HxW mask."""
    magnitude = saliency.abs()
    while magnitude.ndim > 2:
        magnitude = magnitude.sum(dim=0)
    masks = masks.to(device=magnitude.device, dtype=magnitude.dtype)
    flat = masks.flatten(1)
    denominator = flat.sum(dim=1).clamp_min(1.0)
    return (flat * magnitude.flatten()[None, :]).sum(dim=1) / denominator


def aggregate_text_saliency(saliency: torch.Tensor, token_groups: list[list[int]]) -> torch.Tensor:
    """Average absolute embedding attribution for each phrase."""
    token_magnitude = saliency.abs().sum(dim=-1)
    if token_magnitude.ndim > 1:
        token_magnitude = token_magnitude.sum(dim=tuple(range(token_magnitude.ndim - 1)))
    values = []
    for group in token_groups:
        if not group:
            raise ValueError("textual evidence groups cannot be empty")
        values.append(token_magnitude[group].mean())
    return torch.stack(values)


def interaction_saliency(text: torch.Tensor, visual: torch.Tensor, compatibility: torch.Tensor) -> torch.Tensor:
    """Eq. 4 for a list of relation endpoint indices.

    ``compatibility`` rows contain ``(text_index, visual_index, C)``.
    """
    if compatibility.ndim != 2 or compatibility.shape[1] != 3:
        raise ValueError("compatibility must have columns [text_index, visual_index, value]")
    return torch.stack([
        text[int(row[0].item())] * visual[int(row[1].item())] * row[2]
        for row in compatibility
    ]).abs()
