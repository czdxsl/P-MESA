import torch

from pmesa.adapters import GatedMultimodalScore


def test_gated_score_restores_both_modalities():
    original_image = torch.ones(1, 2, 2)
    baseline_image = torch.zeros_like(original_image)
    original_text = torch.ones(1, 2, 3)
    baseline_text = torch.zeros_like(original_text)
    score = GatedMultimodalScore(
        original_image,
        baseline_image,
        torch.ones(1, 2, 2),
        original_text,
        baseline_text,
        [[0, 1]],
        lambda image, text: image.sum() + text.sum(),
    )
    assert score.primitive_count == 2
    assert score(torch.zeros(2)) == 0
    assert score(torch.ones(2)) == original_image.sum() + original_text.sum()
