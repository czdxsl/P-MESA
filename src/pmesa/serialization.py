"""JSON-safe explanation export."""

from __future__ import annotations

import json
from pathlib import Path

from .explainer import Explanation


def save_explanation(explanation: Explanation, path: str | Path) -> None:
    payload = {
        "units": [
            {"id": u.id, "kind": u.kind.value, "label": u.label, "endpoints": u.endpoints,
             "semantic_factors": dict(u.semantic_factors), "relation_types": dict(u.relation_types)}
            for u in explanation.units
        ],
        "selected_indices": list(explanation.selected_indices),
        "selected_ids": [u.id for u in explanation.selected_units],
        "saliency": explanation.saliency.tolist(),
        "contribution": explanation.attribution.contribution.tolist(),
        "per_path": explanation.attribution.per_path.tolist(),
        "stability": explanation.attribution.stability.tolist(),
        "completeness_error": explanation.attribution.completeness_error.tolist(),
        "primitive_count": explanation.attribution.primitive_count,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
