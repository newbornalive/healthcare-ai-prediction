"""Train, calibrate, evaluate, explain, and save all project models."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical_models import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)
from src.cnn import integrated_gradients, predict_cnn, train_cnn
from src.config import (
    DATA_FILE,
    FEATURE_COLUMNS,
    IMAGES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_SENSITIVITY,
)
from src.data import create_splits, load_dataset, make_data_summary
from src.evaluation import (
    PlattCalibrator,
    bootstrap_intervals,
    evaluate_predictions,
    select_threshold,
)
from src.plots import (
    plot_calibration_curves,
    plot_cnn_history,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_subgroup_auc,
    plot_threshold_tradeoff,
)
from src.subgroups import evaluate_subgroups


def _ensure_directories() -> None:
    for directory in [MODELS_DIR, RESULTS_DIR, IMAGES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _predict_positive(model: object, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(X)
    return np.asarray(probabilities)[:, 1]


def _safe_model_name(model_name: str) -> str:
    if model_name == "1D CNN":
        return "cnn"
    return model_name.lower().replace(" ", "_")


def main() -> None:
    """Execute the complete reproducible modelling workflow."""
    _ensure_directories()
    started = time.perf_counter()

    frame = load_dataset(DATA_FILE)
    splits = create_splits(frame)
    data_summary = make_data_summary(frame, splits)
    (RESULTS_DIR / "data_summary.json").write_text(
        json.dumps(data_summary, indent=2), encoding="utf-8"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(splits.X_train)
    X_validation_scaled = scaler.transform(splits.X_validation)
    X_calibration_scaled = scaler.transform(splits.X_calibration)
    X_test_scaled = scaler.transform(splits.X_test)
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")

    raw_models: dict[str, object] = {}
    raw_calibration_probabilities: dict[str, np.ndarray] = {}
    raw_test_probabilities: dict[str, np.ndarray] = {}

    logistic = build_logistic_regression(RANDOM_SEED)
    logistic.fit(X_train_scaled, splits.y_train)
    raw_models["Logistic Regression"] = logistic
    raw_calibration_probabilities["Logistic Regression"] = _predict_positive(
        logistic, X_calibration_scaled
    )
    raw_test_probabilities["Logistic Regression"] = _predict_positive(
        logistic, X_test_scaled
    )
    joblib.dump(logistic, MODELS_DIR / "logistic_regression.joblib")

    random_forest = build_random_forest(RANDOM_SEED)
    random_forest.fit(splits.X_train, splits.y_train)
    raw_models["Random Forest"] = random_forest
    raw_calibration_probabilities["Random Forest"] = _predict_positive(
        random_forest, splits.X_calibration
    )
    raw_test_probabilities["Random Forest"] = _predict_positive(
        random_forest, splits.X_test
    )
    joblib.dump(random_forest, MODELS_DIR / "random_forest.joblib")

    xgboost = build_xgboost(splits.y_train.to_numpy(), RANDOM_SEED)
    xgboost.fit(
        splits.X_train,
        splits.y_train,
        eval_set=[(splits.X_validation, splits.y_validation)],
        verbose=False,
    )
    raw_models["XGBoost"] = xgboost
    raw_calibration_probabilities["XGBoost"] = _predict_positive(
        xgboost, splits.X_calibration
    )
    raw_test_probabilities["XGBoost"] = _predict_positive(
        xgboost, splits.X_test
    )
    xgboost.save_model(MODELS_DIR / "xgboost_model.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # A fixed stratified subset keeps the CNN reproducible and CPU-runnable.
    from sklearn.model_selection import train_test_split

    cnn_train_X, _, cnn_train_y, _ = train_test_split(
        X_train_scaled,
        splits.y_train.to_numpy(),
        train_size=min(40000, len(X_train_scaled)),
        stratify=splits.y_train.to_numpy(),
        random_state=RANDOM_SEED,
    )
    cnn_validation_X, _, cnn_validation_y, _ = train_test_split(
        X_validation_scaled,
        splits.y_validation.to_numpy(),
        train_size=min(10000, len(X_validation_scaled)),
        stratify=splits.y_validation.to_numpy(),
        random_state=RANDOM_SEED + 1,
    )
    cnn_result = train_cnn(
        cnn_train_X.astype(np.float32),
        cnn_train_y.astype(np.float32),
        cnn_validation_X.astype(np.float32),
        cnn_validation_y.astype(np.float32),
        random_seed=RANDOM_SEED,
        device=device,
    )
    raw_calibration_probabilities["1D CNN"] = predict_cnn(
        cnn_result.model, X_calibration_scaled.astype(np.float32)
    )
    raw_test_probabilities["1D CNN"] = predict_cnn(
        cnn_result.model, X_test_scaled.astype(np.float32)
    )
    torch.save(
        {
            "state_dict": cnn_result.model.state_dict(),
            "n_features": len(FEATURE_COLUMNS),
            "feature_order": FEATURE_COLUMNS,
            "best_epoch": cnn_result.best_epoch,
        },
        MODELS_DIR / "diabetes_cnn.pt",
    )
    history = pd.DataFrame(cnn_result.history)
    history.to_csv(RESULTS_DIR / "cnn_training_history.csv", index=False)
    plot_cnn_history(history, IMAGES_DIR / "cnn_training_history.png")

    calibrated_test_probabilities: dict[str, np.ndarray] = {}
    selected_thresholds: dict[str, float] = {}
    model_metrics: list[dict[str, float | str]] = []
    threshold_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, float | str]] = []

    for offset, model_name in enumerate(raw_test_probabilities):
        calibrator = PlattCalibrator().fit(
            raw_calibration_probabilities[model_name],
            splits.y_calibration.to_numpy(),
        )
        joblib.dump(
            calibrator,
            MODELS_DIR
            / f"{_safe_model_name(model_name)}_calibrator.joblib",
        )
        calibration_probabilities = calibrator.predict_proba(
            raw_calibration_probabilities[model_name]
        )
        test_probabilities = calibrator.predict_proba(
            raw_test_probabilities[model_name]
        )
        calibrated_test_probabilities[model_name] = test_probabilities

        threshold_selection = select_threshold(
            splits.y_calibration.to_numpy(),
            calibration_probabilities,
            target_sensitivity=TARGET_SENSITIVITY,
        )
        selected_thresholds[model_name] = threshold_selection.threshold
        threshold_frame = threshold_selection.table.copy()
        threshold_frame.insert(0, "model", model_name)
        threshold_frames.append(threshold_frame)

        model_metrics.append(
            evaluate_predictions(
                model_name,
                splits.y_test.to_numpy(),
                test_probabilities,
                threshold_selection.threshold,
            )
        )
        bootstrap_rows.extend(
            bootstrap_intervals(
                model_name,
                splits.y_test.to_numpy(),
                test_probabilities,
                iterations=40,
                random_seed=RANDOM_SEED + offset,
            )
        )
        plot_confusion_matrix(
            splits.y_test.to_numpy(),
            test_probabilities,
            threshold_selection.threshold,
            model_name,
            IMAGES_DIR
            / f"confusion_matrix_{_safe_model_name(model_name)}.png",
        )
        plot_threshold_tradeoff(
            threshold_selection.table,
            model_name,
            threshold_selection.threshold,
            IMAGES_DIR
            / f"threshold_analysis_{_safe_model_name(model_name)}.png",
        )

    metrics_frame = pd.DataFrame(model_metrics).sort_values(
        "pr_auc", ascending=False
    )
    metrics_frame.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    pd.concat(threshold_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "threshold_analysis.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        RESULTS_DIR / "bootstrap_confidence_intervals.csv", index=False
    )

    plot_roc_curves(
        splits.y_test.to_numpy(),
        calibrated_test_probabilities,
        IMAGES_DIR / "roc_curves.png",
    )
    plot_precision_recall_curves(
        splits.y_test.to_numpy(),
        calibrated_test_probabilities,
        IMAGES_DIR / "precision_recall_curves.png",
    )
    plot_calibration_curves(
        splits.y_test.to_numpy(),
        calibrated_test_probabilities,
        IMAGES_DIR / "calibration_curves.png",
    )

    best_model_name = str(metrics_frame.iloc[0]["model"])
    best_probabilities = calibrated_test_probabilities[best_model_name]
    best_threshold = selected_thresholds[best_model_name]
    subgroup_metrics = evaluate_subgroups(
        splits.X_test,
        splits.y_test.to_numpy(),
        best_probabilities,
        best_threshold,
        best_model_name,
    )
    subgroup_metrics.to_csv(RESULTS_DIR / "subgroup_metrics.csv", index=False)
    plot_subgroup_auc(
        subgroup_metrics, IMAGES_DIR / "subgroup_roc_auc.png"
    )

    # XGBoost explanation on a fixed test sample.
    explanation_sample = splits.X_test.sample(
        n=min(750, len(splits.X_test)), random_state=RANDOM_SEED
    )
    explainer = shap.TreeExplainer(xgboost)
    shap_values = explainer.shap_values(explanation_sample)
    shap_array = np.asarray(shap_values)
    xgb_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": np.mean(np.abs(shap_array), axis=0),
        }
    ).sort_values("importance", ascending=False)
    xgb_importance.to_csv(
        RESULTS_DIR / "xgboost_shap_importance.csv", index=False
    )
    plot_feature_importance(
        xgb_importance,
        "XGBoost Mean Absolute SHAP Importance",
        IMAGES_DIR / "xgboost_shap_importance.png",
    )

    # CNN explanation using integrated gradients over standardised inputs.
    rng = np.random.default_rng(RANDOM_SEED)
    explanation_indices = rng.choice(
        len(X_test_scaled), size=min(500, len(X_test_scaled)), replace=False
    )
    cnn_importance_values = integrated_gradients(
        cnn_result.model,
        X_test_scaled[explanation_indices].astype(np.float32),
        steps=12,
    )
    cnn_importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": cnn_importance_values}
    ).sort_values("importance", ascending=False)
    cnn_importance.to_csv(
        RESULTS_DIR / "cnn_integrated_gradients.csv", index=False
    )
    plot_feature_importance(
        cnn_importance,
        "1D CNN Mean Absolute Integrated Gradients",
        IMAGES_DIR / "cnn_integrated_gradients.png",
    )

    prediction_sample = splits.X_test.head(250).copy()
    prediction_sample["observed_outcome"] = splits.y_test.head(250).to_numpy()
    for model_name, probabilities in calibrated_test_probabilities.items():
        safe_name = _safe_model_name(model_name)
        prediction_sample[f"{safe_name}_probability"] = probabilities[:250]
    prediction_sample.to_csv(
        RESULTS_DIR / "prediction_sample.csv", index=False
    )

    metadata = {
        "best_model_by_pr_auc": best_model_name,
        "selected_thresholds": selected_thresholds,
        "target_sensitivity": TARGET_SENSITIVITY,
        "feature_order": FEATURE_COLUMNS,
        "cnn_best_epoch": cnn_result.best_epoch,
        "cnn_training_records": int(len(cnn_train_X)),
        "cnn_validation_records": int(len(cnn_validation_X)),
        "device_used_for_cnn_training": device,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(metrics_frame.to_string(index=False))
    print(f"Best model by test PR-AUC: {best_model_name}")
    print(f"Pipeline runtime: {metadata['runtime_seconds']:.1f} seconds")


if __name__ == "__main__":
    main()
