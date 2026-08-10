"""Qualitative-example selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CandidateExample:
    id: str
    metrics: Mapping[str, float]
    category: str = "default"


def select_representative_examples(
    examples: Iterable[CandidateExample],
    *,
    count: int,
    metric: str = "faithfulness",
    min_quality: float | None = None,
) -> list[CandidateExample]:
    """Select strong examples with category diversity and deterministic ties."""
    candidates = [e for e in examples if metric in e.metrics]
    if min_quality is not None:
        candidates = [e for e in candidates if e.metrics[metric] >= min_quality]
    ranked = sorted(candidates, key=lambda e: (-e.metrics[metric], e.id))
    selected: list[CandidateExample] = []
    seen: set[str] = set()
    for example in ranked:
        if example.category not in seen:
            selected.append(example)
            seen.add(example.category)
            if len(selected) == count:
                return selected
    for example in ranked:
        if example not in selected:
            selected.append(example)
            if len(selected) == count:
                break
    return selected
