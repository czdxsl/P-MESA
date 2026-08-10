import torch

from pmesa.models import SpanHead


def test_span_head_returns_one_logit_per_span():
    head = SpanHead(hidden_size=16, dropout=0)
    logits = head(torch.randn(5, 16))
    assert logits.shape == (5,)
    assert torch.isfinite(logits).all()
