"""Publication-style figures for discrimination, calibration, and diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


def plot_roc_curves(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot ROC curves for all calibrated models."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for model_name, model_probabilities in probabilities.items():
        RocCurveDisplay.from_predictions(
            y_true, model_probabilities, name=model_name, ax=ax
        )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("Calibrated ROC Curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_precision_recall_curves(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot precision-recall curves with the outcome prevalence baseline."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for model_name, model_probabilities in probabilities.items():
        PrecisionRecallDisplay.from_predictions(
            y_true, model_probabilities, name=model_name, ax=ax
        )
    ax.axhline(float(np.mean(y_true)), linestyle="--", linewidth=1)
    ax.set_title("Calibrated Precision-Recall Curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_calibration_curves(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot reliability curves for calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for model_name, model_probabilities in probabilities.items():
        observed, predicted = calibration_curve(
            y_true, model_probabilities, n_bins=10, strategy="quantile"
        )
        ax.plot(predicted, observed, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed outcome rate")
    ax.set_title("Probability Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot one confusion matrix at the selected sensitivity threshold."""
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ConfusionMatrixDisplay(matrix, display_labels=["No diabetes", "Elevated risk"]).plot(
        ax=ax, values_format=",", colorbar=False
    )
    ax.set_title(f"{model_name}: Threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_threshold_tradeoff(
    threshold_table: pd.DataFrame,
    model_name: str,
    selected_threshold: float,
    output_path: Path,
) -> None:
    """Plot sensitivity and specificity across decision thresholds."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(threshold_table["threshold"], threshold_table["sensitivity"], label="Sensitivity")
    ax.plot(threshold_table["threshold"], threshold_table["specificity"], label="Specificity")
    ax.axvline(selected_threshold, linestyle="--", linewidth=1, label=f"Selected: {selected_threshold:.2f}")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{model_name}: Threshold Sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_cnn_history(history: pd.DataFrame, output_path: Path) -> None:
    """Plot CNN training and validation losses by epoch."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Train loss")
    ax.plot(history["epoch"], history["validation_loss"], marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted binary cross-entropy")
    ax.set_title("CNN Training History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_feature_importance(
    importance: pd.DataFrame,
    title: str,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Plot the highest-ranked features for a model explanation method."""
    plotted = importance.sort_values("importance", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.barh(plotted["feature"], plotted["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_subgroup_auc(subgroups: pd.DataFrame, output_path: Path) -> None:
    """Plot subgroup ROC-AUC values for the selected model."""
    ordered = subgroups.sort_values(["subgroup_variable", "roc_auc"])
    labels = ordered["subgroup_variable"].str.replace("_group", "") + ": " + ordered["subgroup"]
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.barh(labels, ordered["roc_auc"])
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.set_xlim(0.5, 0.95)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Selected Model: Subgroup Discrimination")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
