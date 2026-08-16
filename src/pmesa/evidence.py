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
    """One semantic evidence unit.

    Visual and textual units are primitive restoration coordinates. Relation
    units are derived interactions and must name their primitive endpoints;
    they are never assigned an independent restoration gate.
    """

    id: str
    kind: EvidenceKind
    label: str
    semantic_factors: Mapping[str, float] = field(default_factory=dict)
    relation_types: Mapping[str, float] = field(default_factory=dict)
    endpoints: tuple[str, str] | None = None
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("evidence id must be non-empty")
        if self.kind is EvidenceKind.RELATION:
            if self.endpoints is None or len(self.endpoints) != 2:
                raise ValueError("relation evidence requires textual and visual endpoint ids")
            if self.endpoints[0] == self.endpoints[1]:
                raise ValueError("relation endpoints must be distinct")
        elif self.endpoints is not None:
            raise ValueError("only relation evidence may define endpoints")
        for values in (self.semantic_factors, self.relation_types):
            if any(float(v) < 0 for v in values.values()):
                raise ValueError("coverage weights must be non-negative")

    @property
    def is_primitive(self) -> bool:
        return self.kind is not EvidenceKind.RELATION
