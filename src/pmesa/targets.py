"""Target scores used in the manuscript experiments."""

from __future__ import annotations

import torch


def predicted_answer_logit(logits: torch.Tensor, answer_index: int) -> torch.Tensor:
    """Pre-softmax ALBEF-VQA score for the fixed predicted answer."""
    return logits.reshape(-1)[answer_index]


def inconsistency_score(logits: torch.Tensor, inconsistent_index: int = 1) -> torch.Tensor:
    """ALBEF-ITM inconsistency score before normalization."""
    return logits.reshape(-1)[inconsistent_index]


def span_hallucination_probability(logit: torch.Tensor) -> torch.Tensor:
    """Probability assigned to one fixed response span."""
    if logit.numel() != 1:
        raise ValueError("a span target must contain one logit")
    return logit.reshape(()).sigmoid()


def length_normalized_log_likelihood(token_log_probabilities: torch.Tensor) -> torch.Tensor:
    """Eq. (23) for a fixed generated target sequence."""
    if token_log_probabilities.ndim != 1 or token_log_probabilities.numel() == 0:
        raise ValueError("token log probabilities must be a non-empty vector")
    return token_log_probabilities.mean()
