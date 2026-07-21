"""Data loading, validation, and leakage-safe splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, RANDOM_SEED, TARGET


@dataclass(frozen=True)
class DataSplits:
    """Container for mutually exclusive train, validation, calibration, and test sets."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_calibration: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_calibration: pd.Series
    y_test: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the CDC diabetes indicator data and enforce expected types."""
    frame = pd.read_csv(path)
    validate_dataset(frame)
    frame = frame.copy()
    frame[TARGET] = frame[TARGET].astype(int)
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Validate schema, target values, missingness, and obvious range errors."""
    expected = {TARGET, *FEATURE_COLUMNS}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame[list(expected)].isna().any().any():
        raise ValueError("Unexpected missing values were detected.")
    target_values = set(pd.Series(frame[TARGET]).astype(int).unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Target must be binary; observed {sorted(target_values)}")
    if (frame["BMI"] <= 0).any():
        raise ValueError("BMI must be positive.")
    if not frame["Sex"].isin([0, 1]).all():
        raise ValueError("Sex must be coded as 0 or 1.")


def create_splits(
    frame: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> DataSplits:
    """Create 60/10/15/15 stratified splits without shared observations."""
    X = frame[FEATURE_COLUMNS].copy()
    y = frame[TARGET].astype(int).copy()

    X_trainvalcal, X_test, y_trainvalcal, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=random_seed,
    )

    # The remaining 85% is divided into 60% train and 25% temporary data.
    X_train, X_valcal, y_train, y_valcal = train_test_split(
        X_trainvalcal,
        y_trainvalcal,
        test_size=25 / 85,
        stratify=y_trainvalcal,
        random_state=random_seed + 1,
    )

    # The 25% temporary data is divided into 10% validation and 15% calibration.
    X_validation, X_calibration, y_validation, y_calibration = train_test_split(
        X_valcal,
        y_valcal,
        test_size=0.60,
        stratify=y_valcal,
        random_state=random_seed + 2,
    )

    return DataSplits(
        X_train=X_train.reset_index(drop=True),
        X_validation=X_validation.reset_index(drop=True),
        X_calibration=X_calibration.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_validation=y_validation.reset_index(drop=True),
        y_calibration=y_calibration.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def make_data_summary(frame: pd.DataFrame, splits: DataSplits) -> dict[str, object]:
    """Create a serialisable summary of the modelling population."""
    return {
        "records": int(len(frame)),
        "features": int(len(FEATURE_COLUMNS)),
        "outcome_prevalence": float(frame[TARGET].mean()),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "split_sizes": {
            "train": int(len(splits.X_train)),
            "validation": int(len(splits.X_validation)),
            "calibration": int(len(splits.X_calibration)),
            "test": int(len(splits.X_test)),
        },
    }
