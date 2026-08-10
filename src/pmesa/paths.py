"""Monotone restoration path construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .evidence import EvidenceKind, EvidenceUnit


@dataclass(frozen=True)
class RestorationPath:
    name: str
    states: torch.Tensor  # [steps + 1, units]

    def __post_init__(self) -> None:
        states = self.states
        if states.ndim != 2 or states.shape[0] < 2:
            raise ValueError("path states must have shape [steps+1, units]")
        if not torch.allclose(states[0], torch.zeros_like(states[0])):
            raise ValueError("a restoration path must start at zero")
        if not torch.allclose(states[-1], torch.ones_like(states[-1])):
            raise ValueError("a restoration path must end at one")
        if bool(torch.any(states[1:] < states[:-1] - 1e-7)):
            raise ValueError("restoration paths must be coordinate-wise monotone")


def _ordered_path(order: Sequence[int], n_units: int, steps: int, name: str) -> RestorationPath:
    """Piecewise-linear path that restores units in the supplied order."""
    if steps < n_units:
        raise ValueError("steps must be at least the number of units for ordered paths")
    t = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    rank = torch.empty(n_units, dtype=torch.float64)
    for position, index in enumerate(order):
        rank[index] = position
    states = torch.clamp(t[:, None] * n_units - rank[None, :], 0.0, 1.0)
    return RestorationPath(name, states)


def generate_paths(
    units: Sequence[EvidenceUnit],
    *,
    steps: int = 50,
    count: int = 6,
    seed: int = 0,
) -> list[RestorationPath]:
    """Generate text-first, vision-first, and interleaved monotone paths.

    Paths beyond the first three use reproducible within-family permutations,
    matching the manuscript's two paths per family when ``count=6``.
    """
    n = len(units)
    if n == 0 or count < 1:
        raise ValueError("at least one unit and one path are required")
    steps = max(steps, n)
    groups = {
        EvidenceKind.VISUAL: [i for i, u in enumerate(units) if u.kind is EvidenceKind.VISUAL],
        EvidenceKind.TEXTUAL: [i for i, u in enumerate(units) if u.kind is EvidenceKind.TEXTUAL],
        EvidenceKind.RELATION: [i for i, u in enumerate(units) if u.kind is EvidenceKind.RELATION],
    }
    generator = torch.Generator().manual_seed(seed)

    def shuffled(xs: list[int]) -> list[int]:
        if len(xs) < 2:
            return list(xs)
        return [xs[i] for i in torch.randperm(len(xs), generator=generator).tolist()]

    paths: list[RestorationPath] = []
    for i in range(count):
        variant = i % 3
        v, q, r = (shuffled(groups[k]) for k in (EvidenceKind.VISUAL, EvidenceKind.TEXTUAL, EvidenceKind.RELATION))
        if variant == 0:
            order, name = q + r + v, f"text-first-{i // 3 + 1}"
        elif variant == 1:
            order, name = v + r + q, f"vision-first-{i // 3 + 1}"
        else:
            order = []
            columns = [v, q, r]
            while any(columns):
                for column in columns:
                    if column:
                        order.append(column.pop(0))
            name = f"interleaved-{i // 3 + 1}"
        paths.append(_ordered_path(order, n, steps, name))
    return paths
