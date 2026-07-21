"""Download the CDC Diabetes Health Indicators dataset from UCI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"


def main() -> None:
    """Fetch UCI dataset 891 and write the binary analytical table."""
    dataset = fetch_ucirepo(id=891)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if "Diabetes_binary" in targets.columns:
        target = targets["Diabetes_binary"]
    else:
        binary_candidates = [
            column for column in targets.columns if "binary" in column.lower()
        ]
        if not binary_candidates:
            raise ValueError(
                "UCI response did not contain the expected Diabetes_binary target."
            )
        target = targets[binary_candidates[0]]

    frame = pd.concat(
        [target.rename("Diabetes_binary"), features], axis=1
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(frame):,} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
