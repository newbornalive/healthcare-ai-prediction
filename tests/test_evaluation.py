"""Tests for calibration and threshold selection."""

from __future__ import annotations

import numpy as np

from src.evaluation import (
    PlattCalibrator,
    evaluate_predictions,
    select_threshold,
)


def test_platt_calibrator_returns_valid_probabilities() -> None:
    probabilities = np.linspace(0.01, 0.99, 200)
    targets = (probabilities > 0.65).astype(int)
    calibrator = PlattCalibrator().fit(probabilities, targets)
    calibrated = calibrator.predict_proba(probabilities)
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert np.all(np.diff(calibrated) >= 0)


def test_threshold_selection_targets_sensitivity() -> None:
    targets = np.array([0] * 80 + [1] * 20)
    probabilities = np.concatenate(
        [np.linspace(0.01, 0.65, 80), np.linspace(0.35, 0.99, 20)]
    )
    selection = select_threshold(targets, probabilities, target_sensitivity=0.80)
    metrics = evaluate_predictions(
        "test", targets, probabilities, selection.threshold
    )
    assert metrics["sensitivity"] >= 0.75
    assert 0.01 <= selection.threshold <= 0.99
