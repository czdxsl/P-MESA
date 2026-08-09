import numpy as np

from pmesa.metrics import jaccard, mask_iou, mean_pairwise_stability, pointing_game, span_f1


def test_set_and_localization_metrics():
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    assert mean_pairwise_stability([{1}, {1}, {1}]) == 1
    assert mask_iou(np.array([[1, 0]]), np.array([[1, 1]])) == .5
    assert pointing_game(np.array([[.1, .9]]), np.array([[0, 1]])) == 1
    assert span_f1({1, 2}, {2, 3}) == (.5, .5, .5)
