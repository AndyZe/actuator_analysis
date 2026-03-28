#!/usr/bin/env python3
"""Dump stream info — run from repo root: ``python3 scripts/extract_all_data.py``."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package (repo checkout)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from actuator_analysis.config_loader import key_time_points
from actuator_analysis.load_data import available_streams, extract_stream, load_recording

PITCH_CURRENT = "/motors/position/pitch/current"


def main() -> None:
    recording = load_recording()
    kp = key_time_points()
    try:
        streams = available_streams(recording)
        print("available_streams:", streams)
        pitch_except_firing = extract_stream(
            recording,
            PITCH_CURRENT,
            exclude_time_range=(kp.trigger_start, kp.trigger_effects_done),
        )
        print(
            f"{PITCH_CURRENT}: timeline={pitch_except_firing.timeline!r} "
            f"component={pitch_except_firing.component!r} "
            f"samples={len(pitch_except_firing.timestamps)}"
        )
    finally:
        recording.close()


if __name__ == "__main__":
    main()
