"""Score tabular records using a saved calibrated project model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn import DiabetesCNN, predict_cnn
from src.config import FEATURE_COLUMNS, MODELS_DIR

MODEL_NAMES = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "cnn": "1D CNN",
}
CALIBRATOR_FILES = {
    "logistic": "logistic_regression_calibrator.joblib",
    "random_forest": "random_forest_calibrator.joblib",
    "xgboost": "xgboost_calibrator.joblib",
    "cnn": "cnn_calibrator.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate calibrated diabetes-risk predictions."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_NAMES),
        default="xgboost",
    )
    return parser.parse_args()


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    return frame[FEATURE_COLUMNS].copy()


def load_probabilities(model_key: str, X: pd.DataFrame) -> np.ndarray:
    scaler = joblib.load(MODELS_DIR / "feature_scaler.joblib")
    if model_key == "logistic":
        model = joblib.load(MODELS_DIR / "logistic_regression.joblib")
        return model.predict_proba(scaler.transform(X))[:, 1]
    if model_key == "random_forest":
        model = joblib.load(MODELS_DIR / "random_forest.joblib")
        return model.predict_proba(X)[:, 1]
    if model_key == "xgboost":
        model = XGBClassifier()
        model.load_model(MODELS_DIR / "xgboost_model.json")
        return model.predict_proba(X)[:, 1]
    checkpoint = torch.load(
        MODELS_DIR / "diabetes_cnn.pt", map_location="cpu", weights_only=False
    )
    model = DiabetesCNN(n_features=int(checkpoint["n_features"]))
    model.load_state_dict(checkpoint["state_dict"])
    return predict_cnn(model, scaler.transform(X).astype(np.float32))


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    X = validate_input(frame)
    raw_probabilities = load_probabilities(args.model, X)
    calibrator = joblib.load(MODELS_DIR / CALIBRATOR_FILES[args.model])
    probabilities = calibrator.predict_proba(raw_probabilities)

    metadata = json.loads(
        (MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8")
    )
    model_name = MODEL_NAMES[args.model]
    threshold = float(metadata["selected_thresholds"][model_name])

    output = frame.copy()
    output["calibrated_risk_probability"] = probabilities
    output["research_risk_flag"] = (probabilities >= threshold).astype(int)
    output["model"] = model_name
    output["threshold"] = threshold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Scored {len(output):,} rows and wrote {args.output}")


if __name__ == "__main__":
    main()
