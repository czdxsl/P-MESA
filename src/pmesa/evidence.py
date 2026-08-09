"""Semantic evidence representations used by P-MESA."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EvidenceKind(str, Enum):
    VISUAL = "visual"
    TEXTUAL = "textual"
    RELATION = "relation"


@dataclass(frozen=True)
class EvidenceUnit:
    """One restorable semantic unit.

    ``payload`` is adapter-owned metadata (mask, token indices, or relation
    endpoints). It is intentionally excluded from equality so unit identity is
    stable across serialization and perturbation runs.
    """

    id: str
    kind: EvidenceKind
    label: str
    semantic_factors: Mapping[str, float] = field(default_factory=dict)
    relation_types: Mapping[str, float] = field(default_factory=dict)
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("evidence id must be non-empty")
        for values in (self.semantic_factors, self.relation_types):
            if any(float(v) < 0 for v in values.values()):
                raise ValueError("coverage weights must be non-negative")

