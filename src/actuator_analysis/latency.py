"""Shared latency-analysis helpers for actuator streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from actuator_analysis.load_data import StreamData
from actuator_analysis.plot_streams import _values_1d_float


@dataclass(frozen=True)
class AlignedSeries:
    """Target/current samples interpolated onto one uniform overlap grid."""

    t_grid: np.ndarray
    y_target: np.ndarray
    y_current: np.ndarray
    dt_s: float


def timestamps_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Convert timestamps to float seconds.

    Supports ``datetime64`` values, float-second timelines, and large integer
    nanosecond timelines.
    """
    t = np.asarray(timestamps)
    if t.size == 0:
        return np.asarray([], dtype=np.float64)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").astype(np.int64).astype(np.float64) * 1e-9

    t_s = np.asarray(t, dtype=np.float64)
    max_abs = float(np.nanmax(np.abs(t_s)))
    if max_abs > 1e12:
        return t_s * 1e-9
    return t_s


def sorted_stream_arrays(stream: StreamData) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted ``(timestamps_s, values)`` arrays trimmed to common length."""
    t_s = timestamps_to_seconds(stream.timestamps)
    y = _values_1d_float(stream.values)
    n = min(t_s.size, y.size)
    if n == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    t_s = t_s[:n]
    y = y[:n]
    order = np.argsort(t_s)
    return t_s[order], y[order]


def aligned_uniform_series(target: StreamData, current: StreamData) -> AlignedSeries | None:
    """Interpolate target and current onto a shared uniform grid over overlap."""
    t_target, y_target = sorted_stream_arrays(target)
    t_current, y_current = sorted_stream_arrays(current)
    if t_target.size < 2 or t_current.size < 2:
        return None

    t_lo = max(float(np.min(t_target)), float(np.min(t_current)))
    t_hi = min(float(np.max(t_target)), float(np.max(t_current)))
    if t_hi <= t_lo:
        return None

    n = max(int(t_target.size), int(t_current.size))
    if n < 2:
        return None

    t_grid = np.linspace(t_lo, t_hi, n)
    y_target_i = np.interp(t_grid, t_target, y_target)
    y_current_i = np.interp(t_grid, t_current, y_current)
    dt_s = (t_hi - t_lo) / (n - 1)
    return AlignedSeries(t_grid=t_grid, y_target=y_target_i, y_current=y_current_i, dt_s=dt_s)


def latency_from_correlate(
    y_target: np.ndarray,
    y_current: np.ndarray,
    dt_s: float,
) -> tuple[int, float]:
    """Return positive latency when ``current`` lags ``target``.

    For ``np.correlate(a, b, mode="full")``, a negative maximizing lag means
    ``b`` must be shifted earlier to line up with ``a``. With ``a=target`` and
    ``b=current``, that corresponds to ``current`` occurring later than
    ``target``.
    """
    a = np.asarray(y_target, dtype=np.float64) - np.mean(y_target)
    b = np.asarray(y_current, dtype=np.float64) - np.mean(y_current)
    n = a.size
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-(n - 1), n)
    k = int(np.argmax(corr))
    lag_samples = int(lags[k])
    current_lag_samples = -lag_samples
    latency_s = current_lag_samples * dt_s
    return current_lag_samples, latency_s


__all__ = [
    "AlignedSeries",
    "aligned_uniform_series",
    "latency_from_correlate",
    "sorted_stream_arrays",
    "timestamps_to_seconds",
]
