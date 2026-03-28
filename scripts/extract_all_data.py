#!/usr/bin/env python3
"""Dump stream info — run from repo root: ``python3 scripts/extract_all_data.py``."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package (repo checkout)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from actuator_analysis.config_loader import key_time_points, results_path
from actuator_analysis.load_data import (
    PitchData,
    YawData,
    available_streams,
    load_motor_axis_data,
    load_recording,
)
from actuator_analysis.plot_streams import plot_stream


def _print_axis_summary(name: str, axis: PitchData | YawData) -> None:
    for field in (
        "current_except_firing",
        "target_except_firing",
        "current_firing",
        "target_firing",
    ):
        stream = getattr(axis, field)
        print(
            f"{name}.{field}: timeline={stream.timeline!r} "
            f"component={stream.component!r} samples={len(stream.timestamps)}"
        )


def main() -> None:
    recording = load_recording()
    kp = key_time_points()
    try:
        streams = available_streams(recording)
        print("available_streams:", streams)
        bundle = load_motor_axis_data(recording, kp)
        _print_axis_summary("pitch", bundle.pitch)
        _print_axis_summary("yaw", bundle.yaw)
        out = results_path() / "pitch_target_firing.png"
        plot_stream(
            bundle.pitch.target_firing,
            title="pitch_target_firing",
            save_path=out,
            show=False,
        )
        print(f"wrote plot: {out}")
    finally:
        recording.close()


if __name__ == "__main__":
    main()
