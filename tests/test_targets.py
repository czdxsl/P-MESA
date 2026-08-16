import torch

from pmesa.targets import (
    inconsistency_score,
    length_normalized_log_likelihood,
    predicted_answer_logit,
    span_hallucination_probability,
)


def test_manuscript_target_scores():
    logits = torch.tensor([.2, .8])
    assert predicted_answer_logit(logits, 1) == .8
    assert inconsistency_score(logits) == .8
    assert torch.allclose(span_hallucination_probability(torch.tensor(0.0)), torch.tensor(.5))
    assert length_normalized_log_likelihood(torch.tensor([-1.0, -3.0])) == -2.0
