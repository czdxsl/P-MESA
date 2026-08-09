"""Numerically integrated path attribution in restoration space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .paths import RestorationPath

ScalarScore = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PathAttribution:
    per_path: torch.Tensor  # [paths, units]
    contribution: torch.Tensor  # [units]
    stability: torch.Tensor  # [units]
    completeness_error: torch.Tensor  # [paths]


def path_integrated_gradients(
    score: ScalarScore,
    paths: Sequence[RestorationPath],
    *,
    device: str | torch.device | None = None,
) -> PathAttribution:
    """Approximate Eq. 7 with midpoint line integration.

    Each segment uses the gradient at its midpoint and multiplies it by the
    segment displacement. This supports arbitrary monotone, piecewise-linear
    paths and preserves completeness up to quadrature error.
    """
    if not paths:
        raise ValueError("paths cannot be empty")
    values: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []
    with torch.enable_grad():
        target_device = torch.device(device) if device is not None else paths[0].states.device
        baseline_score = score(paths[0].states[0].to(device=target_device, dtype=torch.get_default_dtype()))
        input_score = score(paths[0].states[-1].to(device=target_device, dtype=torch.get_default_dtype()))
        score_delta = (input_score - baseline_score).detach()
        for path in paths:
            states = path.states.to(device=target_device, dtype=torch.get_default_dtype())
            contribution = torch.zeros(states.shape[1], dtype=states.dtype, device=states.device)
            for left, right in zip(states[:-1], states[1:]):
                delta = right - left
                if not bool(torch.any(delta)):
                    continue
                midpoint = ((left + right) * 0.5).detach().requires_grad_(True)
                output = score(midpoint)
                if output.ndim != 0:
                    raise ValueError("score function must return a scalar tensor")
                gradient = torch.autograd.grad(output, midpoint, create_graph=False)[0]
                contribution = contribution + gradient.detach() * delta
            values.append(contribution)
            errors.append(contribution.sum() - score_delta)
    per_path = torch.stack(values)
    variance = per_path.var(dim=0, unbiased=False) if len(paths) > 1 else torch.zeros_like(per_path[0])
    return PathAttribution(
        per_path=per_path,
        contribution=per_path.mean(dim=0),
        stability=1.0 / (1.0 + variance),
        completeness_error=torch.stack(errors),
    )
