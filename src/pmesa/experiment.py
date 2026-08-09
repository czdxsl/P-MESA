"""Dataset-agnostic experiment runner and JSONL result records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from .adapters import PMESAAdapter
from .explainer import PMESAExplainer


@dataclass(frozen=True)
class ResultRecord:
    example_id: str
    category: str
    selected_ids: list[str]
    contribution: list[float]
    saliency: list[float]
    stability: list[float]
    completeness_error: list[float]
    metrics: dict[str, float]


def run_dataset(
    examples: Iterable[object],
    *,
    id_of: Callable[[object], str],
    adapter: PMESAAdapter,
    explainer: PMESAExplainer,
    evaluate: Callable[[object, object], dict[str, float]],
    category_of: Callable[[object], str] = lambda _: "default",
    output: str | Path,
) -> list[ResultRecord]:
    """Explain/evaluate examples and atomically retain no fabricated fields."""
    records: list[ResultRecord] = []
    for example in examples:
        prepared = adapter.prepare(example)
        explanation = explainer.explain(prepared.units, prepared.score, prepared.saliency)
        records.append(ResultRecord(
            example_id=id_of(example),
            category=category_of(example),
            selected_ids=[unit.id for unit in explanation.selected_units],
            contribution=explanation.attribution.contribution.tolist(),
            saliency=explanation.saliency.tolist(),
            stability=explanation.attribution.stability.tolist(),
            completeness_error=explanation.attribution.completeness_error.tolist(),
            metrics=evaluate(example, explanation),
        ))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(r), sort_keys=True) for r in records) + "\n", encoding="utf-8")
    return records
