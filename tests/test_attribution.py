import torch

from pmesa.attribution import path_integrated_gradients
from pmesa.evidence import EvidenceKind, EvidenceUnit
from pmesa.paths import generate_paths


def units(n=4):
    kinds = [EvidenceKind.VISUAL, EvidenceKind.TEXTUAL, EvidenceKind.RELATION, EvidenceKind.VISUAL]
    return [EvidenceUnit(str(i), kinds[i], str(i)) for i in range(n)]


def test_paths_are_monotone_and_named_by_family():
    paths = generate_paths(units(), steps=12, count=6, seed=2)
    assert len(paths) == 6
    assert {p.name.split("-")[0] for p in paths} == {"text", "vision", "interleaved"}
    for path in paths:
        assert torch.all(path.states[1:] >= path.states[:-1])
        assert torch.all(path.states[0] == 0)
        assert torch.all(path.states[-1] == 1)


def test_linear_score_has_exact_complete_path_attribution():
    weights = torch.tensor([0.2, 0.7, 1.3, 0.4])
    paths = generate_paths(units(), steps=20, count=6, seed=1)
    result = path_integrated_gradients(lambda z: torch.dot(weights, z), paths)
    assert torch.allclose(result.per_path, weights.expand_as(result.per_path), atol=1e-6)
    assert torch.allclose(result.completeness_error, torch.zeros(6), atol=1e-6)


def test_nonlinear_score_is_complete_to_midpoint_accuracy():
    paths = generate_paths(units(), steps=100, count=6, seed=1)
    result = path_integrated_gradients(lambda z: (z.square()).sum() + z[0] * z[1], paths)
    assert result.completeness_error.abs().max() < 2e-4

