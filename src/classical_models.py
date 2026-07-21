"""Classical and tree-based model definitions."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_logistic_regression(random_seed: int = 42) -> LogisticRegression:
    """Create a class-weighted logistic regression baseline."""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_seed,
    )


def build_random_forest(random_seed: int = 42) -> RandomForestClassifier:
    """Create a regularised random-forest benchmark."""
    return RandomForestClassifier(
        n_estimators=140,
        max_depth=12,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_seed,
    )


def build_xgboost(y_train: np.ndarray, random_seed: int = 42) -> XGBClassifier:
    """Create an imbalance-aware gradient-boosting model."""
    positives = max(float(np.sum(y_train)), 1.0)
    negatives = float(len(y_train) - np.sum(y_train))
    return XGBClassifier(
        n_estimators=320,
        max_depth=4,
        learning_rate=0.045,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=negatives / positives,
        early_stopping_rounds=25,
        n_jobs=6,
        random_state=random_seed,
        tree_method="hist",
    )
