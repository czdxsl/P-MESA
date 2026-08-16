import torch

from pmesa.config import load_config
from pmesa.evidence import EvidenceKind, EvidenceUnit
from pmesa.relations import construct_sparse_relations, relation_saliency


def test_manuscript_defaults_are_loaded():
    _, config = load_config("configs/vqa_x.yaml")
    assert config.integration_points == 50
    assert config.restoration_paths == 6
    assert config.evidence_budget == 5
    assert config.relation_top_k == 3


def test_sparse_relation_construction_uses_primitive_states():
    units = [
        EvidenceUnit("v", EvidenceKind.VISUAL, "dog"),
        EvidenceUnit("t", EvidenceKind.TEXTUAL, "cat"),
    ]
    relations = construct_sparse_relations(
        units,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        lambda z: z[0] + z[1] + z[0] * z[1],
        task="inconsistency",
    )
    assert len(relations) == 1
    assert relations[0].endpoints == ("t", "v")
    assert "contradiction" in relations[0].relation_types
    values = relation_saliency(relations, units, [.5, .8])
    assert torch.allclose(values, torch.tensor([.4]))
