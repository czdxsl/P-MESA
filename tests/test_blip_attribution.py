import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from blip_attribution import integrated_attribution, integrated_gradients, smoothgrad_ig


def score(value):
    return value.sum()


def test_integrated_attribution_is_complete_for_linear_score():
    baseline = torch.zeros(1, 3, 8, 8)
    value = torch.ones_like(baseline)
    attribution = integrated_attribution(score, value, baseline, steps=8)
    assert torch.allclose(attribution, value)


def test_ig_and_smoothgrad_ig_return_finite_maps():
    baseline = torch.zeros(1, 3, 8, 8)
    value = torch.ones_like(baseline)
