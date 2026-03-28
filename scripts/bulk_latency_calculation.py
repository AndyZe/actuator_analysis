#!/usr/bin/env python3
"""Cross-correlation latency (target vs current, except firing) — run from repo root."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package (repo checkout)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from actuator_analysis.extract_all_data import load_all_streams
from actuator_analysis.latency import aligned_uniform_series, latency_from_correlate
from actuator_analysis.load_data import PitchData, YawData


def _report_axis(name: str, axis: PitchData | YawData) -> None:
    target = axis.target_except_firing
    current = axis.current_except_firing
    aligned = aligned_uniform_series(target, current)
    if aligned is None:
        print(f"{name}: insufficient overlap or empty streams; skip")
        return
    lag_samples, latency_s = latency_from_correlate(
        aligned.y_target,
        aligned.y_current,
        aligned.dt_s,
    )
    n = aligned.y_target.size
    print(
        f"{name}: n={n} dt_s={aligned.dt_s:.6g} "
        f"lag_samples={lag_samples} latency_s={latency_s:.6g}"
    )


def main() -> None:
    result = load_all_streams()
    print("available_streams:", result.available_streams)
    _report_axis("pitch", result.bundle.pitch)
    _report_axis("yaw", result.bundle.yaw)


if __name__ == "__main__":
    main()
