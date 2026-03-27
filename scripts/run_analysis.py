#!/usr/bin/env python3
"""Example analysis script — run from repo root: ``python3 scripts/run_analysis.py``."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package (repo checkout)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from actuator_analysis.config_loader import data_path, load_yaml


def main() -> None:
    cfg = load_yaml("defaults")


if __name__ == "__main__":
    main()
