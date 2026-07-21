"""Fast shape tests for the PyTorch CNN."""

from __future__ import annotations

import torch

from src.cnn import DiabetesCNN


def test_cnn_returns_one_logit_per_record() -> None:
    model = DiabetesCNN(n_features=21)
    features = torch.randn(16, 21)
    logits = model(features)
    assert logits.shape == (16,)
    assert torch.isfinite(logits).all()
