"""Tests for schema validation and stratified splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET
from src.data import create_splits, validate_dataset


def make_frame(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.integers(0, 2, size=(n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    ).astype(float)
    frame["BMI"] = rng.integers(15, 50, size=n)
    frame["GenHlth"] = rng.integers(1, 6, size=n)
    frame["MentHlth"] = rng.integers(0, 31, size=n)
    frame["PhysHlth"] = rng.integers(0, 31, size=n)
    frame["Age"] = rng.integers(1, 14, size=n)
    frame["Education"] = rng.integers(1, 7, size=n)
    frame["Income"] = rng.integers(1, 9, size=n)
    frame[TARGET] = rng.binomial(1, 0.15, size=n)
    return frame[[TARGET, *FEATURE_COLUMNS]]


def test_validate_dataset_accepts_expected_schema() -> None:
    frame = make_frame()
    validate_dataset(frame)


def test_create_splits_has_expected_sizes_and_prevalence() -> None:
    frame = make_frame(2000)
    splits = create_splits(frame)
    assert len(splits.X_train) == 1200
    assert len(splits.X_validation) == 200
    assert len(splits.X_calibration) == 300
    assert len(splits.X_test) == 300
    overall_prevalence = frame[TARGET].mean()
    for target in [
        splits.y_train,
        splits.y_validation,
        splits.y_calibration,
        splits.y_test,
    ]:
        assert abs(target.mean() - overall_prevalence) < 0.03
