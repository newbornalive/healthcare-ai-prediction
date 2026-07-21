"""Convenience entry point for data validation and full model training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    commands = [
        [sys.executable, str(ROOT / "scripts" / "validate_data.py")],
        [sys.executable, str(ROOT / "scripts" / "train_models.py")],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
