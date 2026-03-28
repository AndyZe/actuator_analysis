#!/usr/bin/env python3
"""Cross-correlation latency (target vs current, except firing) — run from repo root."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package (repo checkout)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from actuator_analysis.extract_all_data import load_all_streams
from actuator_analysis.load_data import PitchData, StreamData, YawData
from actuator_analysis.plot_streams import _values_1d_float


def _timestamps_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Match ``load_data._timestamps_as_unix_seconds`` (ns → s when |t| is large)."""
    t = np.asarray(timestamps, dtype=np.float64)
    if t.size == 0:
        return t
    max_abs = float(np.nanmax(np.abs(t)))
    if max_abs > 1e12:
        return t * 1e-9
    return t


def _aligned_uniform_series(
    target: StreamData,
    current: StreamData,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Interpolate target and current onto a common uniform grid over time overlap.

    Returns ``(y_target, y_current, dt)`` or ``None`` if overlap is empty or degenerate.
    """
    t_t = _timestamps_to_seconds(target.timestamps)
    t_c = _timestamps_to_seconds(current.timestamps)
    if t_t.size == 0 or t_c.size == 0:
        return None

    y_t = _values_1d_float(target.values)
    y_c = _values_1d_float(current.values)
    if y_t.size != t_t.size or y_c.size != t_c.size:
        n = min(t_t.size, y_t.size, t_c.size, y_c.size)
        if n == 0:
            return None
        t_t, y_t = t_t[:n], y_t[:n]
        t_c, y_c = t_c[:n], y_c[:n]

    o_t = np.argsort(t_t)
    t_t, y_t = t_t[o_t], y_t[o_t]
    o_c = np.argsort(t_c)
    t_c, y_c = t_c[o_c], y_c[o_c]

    t_lo = max(float(np.min(t_t)), float(np.min(t_c)))
    t_hi = min(float(np.max(t_t)), float(np.max(t_c)))
    if t_hi <= t_lo:
        return None

    n = max(int(t_t.size), int(t_c.size))
    if n < 2:
        return None

    t_grid = np.linspace(t_lo, t_hi, n)
    y_t_i = np.interp(t_grid, t_t, y_t)
    y_c_i = np.interp(t_grid, t_c, y_c)
    dt = (t_hi - t_lo) / (n - 1)
    return y_t_i, y_c_i, dt


def _latency_from_correlate(
    y_target: np.ndarray,
    y_current: np.ndarray,
    dt: float,
) -> tuple[int, float]:
    """``numpy.correlate`` full mode; positive lag ⇒ current lags target."""
    a = y_target - np.mean(y_target)
    b = y_current - np.mean(y_current)
    n = a.size
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-(n - 1), n)
    k = int(np.argmax(corr))
    lag_samples = int(lags[k])
    latency_s = lag_samples * dt
    return lag_samples, latency_s


def _report_axis(name: str, axis: PitchData | YawData) -> None:
    target = axis.target_except_firing
    current = axis.current_except_firing
    aligned = _aligned_uniform_series(target, current)
    if aligned is None:
        print(f"{name}: insufficient overlap or empty streams; skip")
        return
    y_t, y_c, dt = aligned
    lag_samples, latency_s = _latency_from_correlate(y_t, y_c, dt)
    n = y_t.size
    print(f"{name}: n={n} dt_s={dt:.6g} lag_samples={lag_samples} latency_s={latency_s:.6g}")


def main() -> None:
    result = load_all_streams()
    print("available_streams:", result.available_streams)
    _report_axis("pitch", result.bundle.pitch)
    _report_axis("yaw", result.bundle.yaw)


if __name__ == "__main__":
    main()
