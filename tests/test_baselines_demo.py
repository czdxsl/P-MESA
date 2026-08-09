import torch

from pmesa.baselines import low_evidence_image, restore_regions
from pmesa.demo import run_demo


def test_baseline_and_restoration_shapes_and_endpoints():
    image = torch.rand(3, 32, 32)
    baseline = low_evidence_image(image, downsample_factor=8, blur_kernel=5)
    masks = torch.ones(1, 32, 32)
    assert baseline.shape == image.shape
    assert torch.allclose(restore_regions(baseline, image, masks, torch.zeros(1)), baseline)
    assert torch.allclose(restore_regions(baseline, image, masks, torch.ones(1)), image)


def test_demo_is_deterministic_and_complete():
    first, second = run_demo(), run_demo()
    assert first.selected_indices == second.selected_indices
    assert first.attribution.completeness_error.abs().max() < 1e-4
    assert len(first.selected_indices) == 3
    assert {u.id for u in first.selected_units} == {"v-snow", "t-heavy-snow", "r-conflict"}
