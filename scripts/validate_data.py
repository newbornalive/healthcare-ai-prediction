"""Validate the raw CDC diabetes indicator dataset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_FILE
from src.data import load_dataset


def main() -> None:
    frame = load_dataset(DATA_FILE)
    print(
        f"Validated {len(frame):,} records, {frame.shape[1] - 1} features, "
        f"and outcome prevalence {frame['Diabetes_binary'].mean():.2%}."
    )


if __name__ == "__main__":
    main()
