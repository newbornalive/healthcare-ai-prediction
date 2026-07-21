"""Project configuration and reproducibility constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
IMAGES_DIR = ROOT / "images"

DATA_FILE = RAW_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
TARGET = "Diabetes_binary"
RANDOM_SEED = 42
TARGET_SENSITIVITY = 0.80

FEATURE_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]
