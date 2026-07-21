"""Subgroup performance evaluation for the selected model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def add_subgroup_labels(X: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable labels from the encoded demographic variables."""
    frame = X.copy()
    frame["Sex_group"] = frame["Sex"].map({0: "Female", 1: "Male"})
    frame["Age_group"] = pd.cut(
        frame["Age"],
        bins=[0, 4, 7, 10, 13],
        labels=["18-34", "35-49", "50-64", "65+"],
        include_lowest=True,
    ).astype(str)
    frame["Income_group"] = pd.cut(
        frame["Income"],
        bins=[0, 2, 4, 6, 8],
        labels=["Low", "Lower-middle", "Upper-middle", "High"],
        include_lowest=True,
    ).astype(str)
    frame["Education_group"] = pd.cut(
        frame["Education"],
        bins=[0, 3, 4, 5, 6],
        labels=["Up to high school", "High school", "Some college", "College graduate"],
        include_lowest=True,
    ).astype(str)
    return frame


def evaluate_subgroups(
    X: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    """Calculate subgroup discrimination, prevalence, and operating metrics."""
    labelled = add_subgroup_labels(X)
    labelled = labelled.reset_index(drop=True)
    labelled["target"] = np.asarray(y_true)
    labelled["probability"] = np.asarray(probabilities)
    labelled["prediction"] = (
        labelled["probability"] >= threshold
    ).astype(int)

    rows: list[dict[str, object]] = []
    for subgroup_variable in [
        "Sex_group",
        "Age_group",
        "Income_group",
        "Education_group",
    ]:
        for subgroup_value, group in labelled.groupby(
            subgroup_variable, observed=True
        ):
            if len(group) < 100 or group["target"].nunique() < 2:
                continue
            tn, fp, fn, tp = confusion_matrix(
                group["target"], group["prediction"], labels=[0, 1]
            ).ravel()
            rows.append(
                {
                    "model": model_name,
                    "subgroup_variable": subgroup_variable,
                    "subgroup": str(subgroup_value),
                    "n": int(len(group)),
                    "prevalence": float(group["target"].mean()),
                    "mean_predicted_risk": float(
                        group["probability"].mean()
                    ),
                    "calibration_gap": float(
                        group["probability"].mean()
                        - group["target"].mean()
                    ),
                    "roc_auc": float(
                        roc_auc_score(group["target"], group["probability"])
                    ),
                    "pr_auc": float(
                        average_precision_score(
                            group["target"], group["probability"]
                        )
                    ),
                    "sensitivity": float(tp / (tp + fn)),
                    "specificity": float(tn / (tn + fp)),
                }
            )
    return pd.DataFrame(rows)
