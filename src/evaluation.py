"""Calibration, threshold selection, and model evaluation functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)


class PlattCalibrator:
    """One-dimensional logistic calibration over model log-odds."""

    def __init__(self) -> None:
        self.model = LogisticRegression(solver="lbfgs")

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model.fit(self._logit(probabilities), np.asarray(y))
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(probabilities))[:, 1]


@dataclass(frozen=True)
class ThresholdSelection:
    """Selected threshold and complete sensitivity-specificity trade-off table."""

    threshold: float
    table: pd.DataFrame


def threshold_table(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Evaluate classification performance across a fixed probability grid."""
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        precision = tp / (tp + fp) if tp + fp else np.nan
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "youden_j": float(sensitivity + specificity - 1),
                "predicted_positive_rate": float(predictions.mean()),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float = 0.80,
) -> ThresholdSelection:
    """Select the most specific threshold that reaches target sensitivity."""
    table = threshold_table(y_true, probabilities)
    feasible = table.loc[table["sensitivity"] >= target_sensitivity]
    if feasible.empty:
        row = table.loc[table["youden_j"].idxmax()]
    else:
        row = feasible.sort_values(
            ["specificity", "threshold"], ascending=[False, False]
        ).iloc[0]
    return ThresholdSelection(threshold=float(row["threshold"]), table=table)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Calculate weighted absolute calibration error across equal-width bins."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    edges = np.linspace(0, 1, bins + 1)
    score = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (
            probabilities < right if right < 1 else probabilities <= right
        )
        if not np.any(mask):
            continue
        score += mask.mean() * abs(
            probabilities[mask].mean() - y_true[mask].mean()
        )
    return float(score)


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | str]:
    """Evaluate discrimination, calibration, and threshold-dependent performance."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "model": model_name,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities
        ),
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "negative_predictive_value": float(npv),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def bootstrap_intervals(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    iterations: int = 100,
    random_seed: int = 42,
) -> list[dict[str, float | str]]:
    """Estimate percentile confidence intervals for key probability metrics."""
    rng = np.random.default_rng(random_seed)
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    estimates: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "brier_score": [],
    }
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sampled_y = y_true[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        sampled_p = probabilities[indices]
        estimates["roc_auc"].append(roc_auc_score(sampled_y, sampled_p))
        estimates["pr_auc"].append(
            average_precision_score(sampled_y, sampled_p)
        )
        estimates["brier_score"].append(
            brier_score_loss(sampled_y, sampled_p)
        )

    rows: list[dict[str, float | str]] = []
    for metric, values in estimates.items():
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "estimate": float(
                    {
                        "roc_auc": roc_auc_score(y_true, probabilities),
                        "pr_auc": average_precision_score(y_true, probabilities),
                        "brier_score": brier_score_loss(y_true, probabilities),
                    }[metric]
                ),
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
                "bootstrap_iterations": int(len(values)),
            }
        )
    return rows
