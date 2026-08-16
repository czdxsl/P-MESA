import torch

from pmesa.attribution import path_integrated_gradients, relation_path_attribution
from pmesa.evidence import EvidenceKind, EvidenceUnit
from pmesa.paths import generate_paths
from pmesa.saliency import multimodal_saliency


def units(n=4):
    kinds = [EvidenceKind.VISUAL, EvidenceKind.TEXTUAL, EvidenceKind.VISUAL, EvidenceKind.TEXTUAL]
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


def test_explicit_cpu_device_is_supported():
    weights = torch.tensor([0.2, 0.7, 1.3, 0.4])
    paths = generate_paths(units(), steps=20, count=2, seed=1)
    result = path_integrated_gradients(lambda z: torch.dot(weights, z), paths, device="cpu")
    assert result.per_path.device.type == "cpu"


def test_relations_are_derived_from_primitive_endpoints():
    primitive = units(2)
    paths = generate_paths(primitive, steps=100, count=6, seed=1)
    values = relation_path_attribution(
        lambda z: z[0] + z[1] + 2 * z[0] * z[1], paths, [(0, 1)]
    )
    assert values.shape == (6, 1)
    assert torch.all(values > 0)
    assert torch.allclose(values.mean(), torch.tensor(1.0), atol=2e-3)


def test_relation_units_cannot_be_restoration_coordinates():
    relation = EvidenceUnit(
        "r", EvidenceKind.RELATION, "relation", endpoints=("a", "b")
    )
    try:
        generate_paths([relation])
    except ValueError as error:
        assert "primitive" in str(error)
    else:
        raise AssertionError("relation coordinate was accepted")


def test_multimodal_saliency_conditions_on_the_original_other_modality():
    image = torch.tensor([2.0])
    text = torch.tensor([3.0])
    visual, textual = multimodal_saliency(
        lambda current_image, current_text: (current_image * current_text).sum(),
        image,
        torch.zeros_like(image),
        text,
        torch.zeros_like(text),
        steps=20,
    )
    assert torch.allclose(visual, torch.tensor([6.0]))
    assert torch.allclose(textual, torch.tensor([6.0]))
