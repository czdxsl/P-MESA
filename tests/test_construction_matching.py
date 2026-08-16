import numpy as np

from pmesa.construction import filter_sam_masks, patch_masks, textual_evidence_from_groups
from pmesa.matching import match_text_embeddings, match_visual_masks


def test_sam_filtering_and_patch_fallback_units():
    large = np.zeros((10, 10), bool)
    large[:5, :5] = True
    duplicate = large.copy()
    tiny = np.zeros((10, 10), bool)
    retained = filter_sam_masks([
        {"segmentation": large, "predicted_iou": .9},
        {"segmentation": duplicate, "predicted_iou": .8},
        {"segmentation": tiny, "predicted_iou": 1.0},
    ], (10, 10))
    assert len(retained) == 1
    assert len(patch_masks((8, 8), (2, 2))) == 4
    assert len(textual_evidence_from_groups(["red car"], [[1, 2]])) == 1


def test_one_to_one_visual_and_text_matching():
    first = [np.array([[1, 0], [0, 0]], bool)]
    second = [np.array([[1, 0], [0, 0]], bool)]
    assert match_visual_masks(first, second) == {0: 0}
    assert match_text_embeddings(np.array([[1.0, 0.0]]), np.array([[.9, .1]])) == {0: 0}
