"""Numerically integrated path attribution in restoration space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .paths import RestorationPath

ScalarScore = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PathAttribution:
    per_path: torch.Tensor
    contribution: torch.Tensor
    stability: torch.Tensor
    completeness_error: torch.Tensor
    primitive_count: int


def path_integrated_gradients(
    score: ScalarScore,
    paths: Sequence[RestorationPath],
    *,
    device: str | torch.device | None = None,
) -> PathAttribution:
    """Approximate Eq. (4) with midpoint line integration.

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
            residual = contribution.sum() - score_delta
            errors.append(residual.abs() / (score_delta.abs() + 1e-8))
    per_path = torch.stack(values)
    variance = per_path.var(dim=0, unbiased=False) if len(paths) > 1 else torch.zeros_like(per_path[0])
    return PathAttribution(
        per_path=per_path,
        contribution=per_path.mean(dim=0),
        stability=1.0 / (1.0 + variance),
        completeness_error=torch.stack(errors),
        primitive_count=per_path.shape[1],
    )


def relation_path_attribution(
    score: ScalarScore,
    paths: Sequence[RestorationPath],
    endpoints: Sequence[tuple[int, int]],
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Approximate Eq. (5) without introducing relation coordinates.

    For every path segment, the four endpoint states are evaluated at the
    midpoint and the interaction residual is integrated with respect to the
    change in joint endpoint availability.
    """
    if not paths:
        raise ValueError("paths cannot be empty")
    n = paths[0].states.shape[1]
    for first, second in endpoints:
        if first == second or min(first, second) < 0 or max(first, second) >= n:
            raise ValueError("relation endpoints must be distinct primitive indices")
    target_device = torch.device(device) if device is not None else paths[0].states.device
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for path in paths:
            states = path.states.to(device=target_device, dtype=torch.get_default_dtype())
            values = torch.zeros(len(endpoints), device=target_device, dtype=states.dtype)
            for left, right in zip(states[:-1], states[1:]):
                midpoint = (left + right) * 0.5
                for relation_index, (first, second) in enumerate(endpoints):
                    delta_availability = right[first] * right[second] - left[first] * left[second]
                    if not bool(delta_availability):
                        continue
                    neither = midpoint.clone()
                    neither[first] = 0
                    neither[second] = 0
                    first_only = neither.clone()
                    first_only[first] = midpoint[first]
                    second_only = neither.clone()
                    second_only[second] = midpoint[second]
                    both = midpoint.clone()
                    residual = score(both) - score(first_only) - score(second_only) + score(neither)
                    values[relation_index] += residual * delta_availability
            rows.append(values)
    if not rows:
        return torch.empty((len(paths), 0), device=target_device)
    return torch.stack(rows)
