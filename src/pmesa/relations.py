"""Sparse cross-modal relation construction from primitive evidence."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from .attribution import ScalarScore
from .evidence import EvidenceKind, EvidenceUnit


def _relation_type(task: str, response_effect: float) -> str:
    if task == "vqa":
        return "grounding" if response_effect > 0 else "support"
    if task == "inconsistency":
        return "contradiction" if response_effect > 0 else "support"
    if task == "hallucination":
        return "support" if response_effect < 0 else "contradiction"
    raise ValueError(f"unsupported task semantics: {task}")


def construct_sparse_relations(
    primitive_units: Sequence[EvidenceUnit],
    text_embeddings: torch.Tensor,
    visual_embeddings: torch.Tensor,
    score: ScalarScore,
    *,
    task: str,
    top_k: int = 3,
) -> list[EvidenceUnit]:
    """Construct Eq. (2) candidates and assign task-conditioned types.

    Embedding rows must follow the textual and visual order in
    ``primitive_units``. The score accepts one gate per primitive unit.
    """
    if any(not unit.is_primitive for unit in primitive_units):
        raise ValueError("relation construction accepts primitive units only")
    textual = [(i, unit) for i, unit in enumerate(primitive_units) if unit.kind is EvidenceKind.TEXTUAL]
    visual = [(i, unit) for i, unit in enumerate(primitive_units) if unit.kind is EvidenceKind.VISUAL]
    if text_embeddings.shape[0] != len(textual) or visual_embeddings.shape[0] != len(visual):
        raise ValueError("embedding rows must match textual and visual evidence counts")
    if text_embeddings.ndim != 2 or visual_embeddings.ndim != 2:
        raise ValueError("evidence embeddings must be matrices")
    if text_embeddings.shape[1] != visual_embeddings.shape[1]:
        raise ValueError("textual and visual embeddings must share an aligned dimension")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    text_features = F.normalize(text_embeddings.float(), dim=-1)
    visual_features = F.normalize(visual_embeddings.float(), dim=-1)
    compatibility = (text_features @ visual_features.T).clamp_min(0)
    device = compatibility.device
    baseline = torch.zeros(len(primitive_units), device=device)
    with torch.no_grad():
        baseline_score = score(baseline)
    relations: list[EvidenceUnit] = []
    for text_row, (text_index, text_unit) in enumerate(textual):
        retained = torch.topk(compatibility[text_row], min(top_k, len(visual))).indices.tolist()
        candidates: list[tuple[EvidenceUnit, float]] = []
        for visual_row in retained:
            visual_index, visual_unit = visual[visual_row]
            text_state = baseline.clone()
            visual_state = baseline.clone()
            joint_state = baseline.clone()
            text_state[text_index] = 1
            visual_state[visual_index] = 1
            joint_state[text_index] = 1
            joint_state[visual_index] = 1
            with torch.no_grad():
                response = score(joint_state) - score(text_state) - score(visual_state) + baseline_score
            effect = float(response.item())
            relation_type = _relation_type(task, effect)
            value = float(compatibility[text_row, visual_row].item())
            candidates.append((EvidenceUnit(
                id=f"r:{text_unit.id}:{visual_unit.id}",
                kind=EvidenceKind.RELATION,
                label=f"{text_unit.label} <-> {visual_unit.label}",
                semantic_factors={"relation": value},
                relation_types={relation_type: value},
                endpoints=(text_unit.id, visual_unit.id),
                payload={"compatibility": value, "response_effect": effect},
            ), effect))
        if task == "hallucination" and candidates and not any(
            "support" in unit.relation_types for unit, _ in candidates
        ):
            unit, effect = max(candidates, key=lambda item: item[0].payload["compatibility"])
            replacement = EvidenceUnit(
                id=unit.id,
                kind=unit.kind,
                label=unit.label,
                semantic_factors=unit.semantic_factors,
                relation_types={"absence-of-support": unit.payload["compatibility"]},
                endpoints=unit.endpoints,
                payload=unit.payload,
            )
            candidates[candidates.index((unit, effect))] = (replacement, effect)
        relations.extend(unit for unit, _ in candidates)
    return relations


def relation_saliency(
    relations: Sequence[EvidenceUnit],
    units: Sequence[EvidenceUnit],
    primitive_saliency: Sequence[float],
) -> torch.Tensor:
    """Compute Eq. (3) from endpoint saliency and compatibility."""
    primitive = [unit for unit in units if unit.is_primitive]
    if len(primitive_saliency) != len(primitive):
        raise ValueError("primitive saliency count mismatch")
    scores = {unit.id: float(value) for unit, value in zip(primitive, primitive_saliency)}
    values = []
    for relation in relations:
        if relation.kind is not EvidenceKind.RELATION or relation.endpoints is None:
            raise ValueError("relation saliency requires relation units with endpoints")
        compatibility = float(relation.payload["compatibility"])
        values.append(abs(scores[relation.endpoints[0]] * scores[relation.endpoints[1]] * compatibility))
    return torch.tensor(values, dtype=torch.float32)
