"""Dataset-agnostic experiment runner and JSONL result records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from .adapters import PMESAAdapter
from .explainer import PMESAExplainer


class ExperimentIntegration(Protocol):
    """Target-model and dataset operations required by the generic runner."""

    adapter: PMESAAdapter

    def examples(self) -> Iterable[object]: ...

    def example_id(self, example: object) -> str: ...

    def category(self, example: object) -> str: ...

    def evaluate(self, example: object, explanation: object) -> dict[str, float]: ...


@dataclass(frozen=True)
class ResultRecord:
    example_id: str
    category: str
    selected_ids: list[str]
    contribution: list[float]
    saliency: list[float]
    stability: list[float]
    completeness_error: list[float]
    primitive_count: int
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
    """Run P-MESA and save one record per example."""
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
            primitive_count=explanation.attribution.primitive_count,
            metrics=evaluate(example, explanation),
        ))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(json.dumps(asdict(r), sort_keys=True) for r in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    temporary.replace(path)
    return records
