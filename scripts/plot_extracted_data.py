#!/usr/bin/env python3
"""Plot extracted pitch data — run from repo root: ``python3 scripts/plot_extracted_data.py``."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package (repo checkout)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from actuator_analysis.config_loader import results_path
from actuator_analysis.extract_all_data import load_all_streams
from actuator_analysis.load_data import PitchData, YawData
from actuator_analysis.plot_streams import plot_stream, plot_two_streams


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
    result = load_all_streams()
    print("available_streams:", result.available_streams)
    bundle = result.bundle
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

    out = results_path() / "pitch_current_and_target_except_firing_first_100s.png"
    plot_two_streams(
        bundle.pitch.current_except_firing,
        bundle.pitch.target_except_firing,
        labels=("pitch_current_except_firing", "pitch_target_except_firing"),
        colors=("blue", "red"),
        title="pitch (first 100 s)",
        save_path=out,
        show=False,
        first_seconds=100.0,
    )
    print(f"wrote plot: {out}")

    out = results_path() / "yaw_current_and_target_except_firing_first_100s.png"
    plot_two_streams(
        bundle.yaw.current_except_firing,
        bundle.yaw.target_except_firing,
        labels=("yaw_current_except_firing", "yaw_target_except_firing"),
        colors=("blue", "red"),
        title="yaw (first 100 s)",
        save_path=out,
        show=False,
        first_seconds=100.0,
    )
    print(f"wrote plot: {out}")


if __name__ == "__main__":
    main()
