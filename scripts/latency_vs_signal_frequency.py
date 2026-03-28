#!/usr/bin/env python3
"""Chunked latency versus signal-frequency analysis.

For each axis, this script:
- loads the ``*_except_firing`` target/current motor streams
- splits both streams into aligned 60-second windows
- computes a spectral centroid for each target chunk
- computes best-fit latency for each target/current chunk pair with
  ``numpy.correlate``

Spectral centroid is computed from the target chunk after demeaning and after
dropping the DC bin so steady position offsets do not dominate the result.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running without installing the package (repo checkout)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from actuator_analysis.config_loader import results_path
from actuator_analysis.extract_all_data import load_all_streams
from actuator_analysis.load_data import PitchData, StreamData, YawData
from actuator_analysis.plot_streams import _values_1d_float

CHUNK_DURATION_S = 60.0


@dataclass(frozen=True)
class ChunkedStream:
    """One time-windowed subset of a stream."""

    index: int
    window_start_s: float
    window_end_s: float
    stream: StreamData


@dataclass(frozen=True)
class ChunkAnalysis:
    """Chunk-level metrics ready for later plotting."""

    axis: str
    chunk_index: int
    window_start_s: float
    window_end_s: float
    target_samples: int
    current_samples: int
    aligned_samples: int
    dt_s: float
    spectral_centroid_hz: float
    lag_samples: int
    latency_s: float


@dataclass(frozen=True)
class AxisAnalysis:
    """All chunk-level metrics for one axis."""

    axis: str
    chunks: tuple[ChunkAnalysis, ...]


@dataclass(frozen=True)
class LatencyFrequencyAnalysis:
    """Chunk-level latency/frequency analysis for pitch and yaw."""

    pitch: AxisAnalysis
    yaw: AxisAnalysis


def _timestamps_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Convert timestamps to float seconds.

    Supports Rerun's common int64 nanosecond timelines, float-second timelines,
    and ``datetime64`` values.
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


def _sorted_stream_arrays(stream: StreamData) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted ``(timestamps_s, values)`` arrays trimmed to a common length."""
    t_s = _timestamps_to_seconds(stream.timestamps)
    y = _values_1d_float(stream.values)
    n = min(t_s.size, y.size)
    if n == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    t_s = t_s[:n]
    y = y[:n]
    order = np.argsort(t_s)
    return t_s[order], y[order]


def _chunk_stream(
    stream: StreamData,
    *,
    chunk_duration_s: float,
    reference_start_s: float,
) -> tuple[ChunkedStream, ...]:
    """Split a stream into fixed windows relative to a shared reference start."""
    t_s, y = _sorted_stream_arrays(stream)
    if t_s.size == 0:
        return ()

    chunk_ids = np.floor((t_s - reference_start_s) / chunk_duration_s).astype(int)
    unique_ids = np.unique(chunk_ids)
    chunks: list[ChunkedStream] = []
    for chunk_index in unique_ids:
        if chunk_index < 0:
            continue
        mask = chunk_ids == chunk_index
        if not np.any(mask):
            continue

        window_start_s = chunk_index * chunk_duration_s
        window_end_s = window_start_s + chunk_duration_s
        chunk_stream = StreamData(
            timestamps=t_s[mask],
            values=y[mask],
            timeline=stream.timeline,
            component=stream.component,
            column_name=stream.column_name,
        )
        chunks.append(
            ChunkedStream(
                index=int(chunk_index),
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                stream=chunk_stream,
            )
        )
    return tuple(chunks)


def _aligned_uniform_series(
    target: StreamData,
    current: StreamData,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Interpolate target and current onto a common uniform grid over overlap."""
    t_t, y_t = _sorted_stream_arrays(target)
    t_c, y_c = _sorted_stream_arrays(current)
    if t_t.size < 2 or t_c.size < 2:
        return None

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


def _uniform_resample(stream: StreamData) -> tuple[np.ndarray, float] | None:
    """Resample one stream onto a uniform grid across its time span."""
    t_s, y = _sorted_stream_arrays(stream)
    if t_s.size < 2:
        return None

    t_lo = float(t_s[0])
    t_hi = float(t_s[-1])
    if t_hi <= t_lo:
        return None

    t_grid = np.linspace(t_lo, t_hi, t_s.size)
    y_i = np.interp(t_grid, t_s, y)
    dt = (t_hi - t_lo) / (t_s.size - 1)
    return y_i, dt


def _latency_from_correlate(
    y_target: np.ndarray,
    y_current: np.ndarray,
    dt: float,
) -> tuple[int, float]:
    """Return positive latency when ``current`` lags ``target``."""
    a = y_target - np.mean(y_target)
    b = y_current - np.mean(y_current)
    n = a.size
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-(n - 1), n)
    lag_samples = int(lags[int(np.argmax(corr))])
    current_lag_samples = -lag_samples
    latency_s = current_lag_samples * dt
    return current_lag_samples, latency_s


def _spectral_centroid_hz(stream: StreamData) -> float | None:
    """Compute a magnitude-weighted spectral centroid for one chunk."""
    resampled = _uniform_resample(stream)
    if resampled is None:
        return None

    y, dt = resampled
    y = y - np.mean(y)
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=dt)
    if spectrum.size <= 1 or freqs.size <= 1:
        return 0.0

    weights = spectrum[1:]
    freq_bins = freqs[1:]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return 0.0
    return float(np.sum(freq_bins * weights) / weight_sum)


def _analyze_axis(name: str, axis: PitchData | YawData) -> AxisAnalysis:
    """Compute chunk-level spectral centroid and latency for one axis."""
    target = axis.target_except_firing
    current = axis.current_except_firing

    target_times_s = _timestamps_to_seconds(target.timestamps)
    current_times_s = _timestamps_to_seconds(current.timestamps)
    if target_times_s.size == 0 or current_times_s.size == 0:
        return AxisAnalysis(axis=name, chunks=())

    reference_start_s = min(float(np.min(target_times_s)), float(np.min(current_times_s)))
    target_chunks = {
        chunk.index: chunk
        for chunk in _chunk_stream(
            target,
            chunk_duration_s=CHUNK_DURATION_S,
            reference_start_s=reference_start_s,
        )
    }
    current_chunks = {
        chunk.index: chunk
        for chunk in _chunk_stream(
            current,
            chunk_duration_s=CHUNK_DURATION_S,
            reference_start_s=reference_start_s,
        )
    }

    analyses: list[ChunkAnalysis] = []
    for chunk_index in sorted(set(target_chunks) & set(current_chunks)):
        target_chunk = target_chunks[chunk_index]
        current_chunk = current_chunks[chunk_index]

        centroid_hz = _spectral_centroid_hz(target_chunk.stream)
        aligned = _aligned_uniform_series(target_chunk.stream, current_chunk.stream)
        if centroid_hz is None or aligned is None:
            continue

        y_t, y_c, dt = aligned
        lag_samples, latency_s = _latency_from_correlate(y_t, y_c, dt)
        analyses.append(
            ChunkAnalysis(
                axis=name,
                chunk_index=chunk_index,
                window_start_s=target_chunk.window_start_s,
                window_end_s=target_chunk.window_end_s,
                target_samples=int(np.asarray(target_chunk.stream.values).size),
                current_samples=int(np.asarray(current_chunk.stream.values).size),
                aligned_samples=int(y_t.size),
                dt_s=float(dt),
                spectral_centroid_hz=float(centroid_hz),
                lag_samples=lag_samples,
                latency_s=float(latency_s),
            )
        )

    return AxisAnalysis(axis=name, chunks=tuple(analyses))


def analyze_latency_vs_signal_frequency() -> LatencyFrequencyAnalysis:
    """Run chunk-level analysis for both pitch and yaw."""
    result = load_all_streams()
    print("available_streams:", result.available_streams)
    return LatencyFrequencyAnalysis(
        pitch=_analyze_axis("pitch", result.bundle.pitch),
        yaw=_analyze_axis("yaw", result.bundle.yaw),
    )


def _print_axis_report(axis_analysis: AxisAnalysis) -> None:
    print(f"{axis_analysis.axis}: {len(axis_analysis.chunks)} analyzed chunk(s)")
    for chunk in axis_analysis.chunks:
        print(
            "  "
            f"chunk={chunk.chunk_index:02d} "
            f"window_s=[{chunk.window_start_s:.0f}, {chunk.window_end_s:.0f}) "
            f"target_n={chunk.target_samples} "
            f"current_n={chunk.current_samples} "
            f"aligned_n={chunk.aligned_samples} "
            f"centroid_hz={chunk.spectral_centroid_hz:.6g} "
            f"lag_samples={chunk.lag_samples} "
            f"latency_s={chunk.latency_s:.6g}"
        )


def _plot_latency_vs_centroid(analysis: LatencyFrequencyAnalysis) -> Path:
    """Plot chunk latency versus target spectral centroid for both axes."""
    out = results_path() / "latency_vs_signal_frequency.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    for axis_analysis, color, label in (
        (analysis.pitch, "tab:blue", "pitch"),
        (analysis.yaw, "tab:orange", "yaw"),
    ):
        if not axis_analysis.chunks:
            continue
        centroid_hz = [chunk.spectral_centroid_hz for chunk in axis_analysis.chunks]
        latency_s = [chunk.latency_s for chunk in axis_analysis.chunks]
        ax.scatter(centroid_hz, latency_s, color=color, label=label, alpha=0.8)

    ax.set_xlabel("spectral centroid (Hz)")
    ax.set_ylabel("latency of best fit (s)")
    ax.set_title("Latency vs signal centroidfrequency")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    analysis = analyze_latency_vs_signal_frequency()
    _print_axis_report(analysis.pitch)
    _print_axis_report(analysis.yaw)
    out = _plot_latency_vs_centroid(analysis)
    print(f"wrote plot: {out}")


if __name__ == "__main__":
    main()
