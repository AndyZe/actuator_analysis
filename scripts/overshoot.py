#!/usr/bin/env python3
"""Chunked overshoot analysis for high-frequency motion bins.

For each axis, this script:
- loads the ``*_except_firing`` target/current motor streams
- splits both streams into aligned 60-second windows
- keeps only chunks whose target spectral centroid is at least 1 Hz
- estimates chunk latency with ``numpy.correlate`` and shifts current backward
- detects target reversal events and measures aligned-current overshoot

The script keeps one in-memory event record per reversal so timestamps and
overshoot values are available for later plotting.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running without installing the package (repo checkout)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from actuator_analysis.config_loader import format_recording_offset_timestamp
from actuator_analysis.extract_all_data import load_all_streams
from actuator_analysis.load_data import PitchData, StreamData, YawData
from actuator_analysis.plot_streams import _values_1d_float

CHUNK_DURATION_S = 60.0
MIN_CENTROID_HZ = 1.0
SMOOTHING_WINDOW_S = 0.10
MIN_REVERSAL_SPAN_S = 0.20
MIN_REVERSAL_AMPLITUDE_FRACTION = 0.02


@dataclass(frozen=True)
class ChunkedStream:
    """One time-windowed subset of a stream."""

    index: int
    window_start_s: float
    window_end_s: float
    stream: StreamData


@dataclass(frozen=True)
class OvershootEvent:
    """Overshoot measured around one target reversal."""

    axis: str
    chunk_index: int
    window_start_s: float
    window_end_s: float
    spectral_centroid_hz: float
    reversal_kind: str
    commanded_direction: str
    reversal_index: int
    reversal_timestamp_s: float
    reversal_time_since_recording_start_s: float
    reversal_time_since_chunk_start_s: float
    reversal_value: float
    current_turn_timestamp_s: float
    current_turn_value: float
    overshoot_value: float


@dataclass(frozen=True)
class ChunkAnalysis:
    """Chunk-level overshoot metrics and stored reversal events."""

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
    reversal_count: int
    events: tuple[OvershootEvent, ...]


@dataclass(frozen=True)
class AxisAnalysis:
    """All analyzed chunks and overshoot events for one axis."""

    axis: str
    total_chunk_pairs: int
    selected_chunk_pairs: int
    chunks: tuple[ChunkAnalysis, ...]


@dataclass(frozen=True)
class OvershootAnalysis:
    """Chunked overshoot analysis for pitch and yaw."""

    pitch: AxisAnalysis
    yaw: AxisAnalysis

    @property
    def events(self) -> tuple[OvershootEvent, ...]:
        """Flatten all overshoot events for later plotting."""
        return tuple(_iter_events(self.pitch)) + tuple(_iter_events(self.yaw))


def _timestamps_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Convert timestamps to float seconds."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
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
    return t_grid, y_t_i, y_c_i, dt


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


def _odd_window_samples(dt: float, duration_s: float, *, minimum: int = 1) -> int:
    samples = max(minimum, int(round(duration_s / dt)))
    if samples % 2 == 0:
        samples += 1
    return samples


def _moving_average(y: np.ndarray, window_samples: int) -> np.ndarray:
    """Return a centered moving average with the same length as ``y``."""
    if window_samples <= 1:
        return np.asarray(y, dtype=np.float64)
    kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    return np.convolve(np.asarray(y, dtype=np.float64), kernel, mode="same")


def _fill_zero_signs(signs: np.ndarray) -> np.ndarray:
    """Fill zero entries from neighboring non-zero signs."""
    out = np.asarray(signs, dtype=np.int8).copy()
    for i in range(1, out.size):
        if out[i] == 0:
            out[i] = out[i - 1]
    for i in range(out.size - 2, -1, -1):
        if out[i] == 0:
            out[i] = out[i + 1]
    return out


def _shift_current_backward(
    t_grid: np.ndarray,
    y_current: np.ndarray,
    latency_s: float,
) -> np.ndarray:
    """Shift ``current`` backward in time so it better aligns with ``target``."""
    return np.interp(
        t_grid + latency_s,
        t_grid,
        y_current,
        left=np.nan,
        right=np.nan,
    )


def _find_reversal_candidates(
    y_target: np.ndarray,
    dt: float,
) -> tuple[tuple[int, str, int], ...]:
    """Return robust target turning points as ``(index, kind, new_direction)``."""
    if y_target.size < 3:
        return ()

    smooth_window = _odd_window_samples(dt, SMOOTHING_WINDOW_S, minimum=3)
    min_span_samples = _odd_window_samples(dt, MIN_REVERSAL_SPAN_S, minimum=3)
    y_smooth = _moving_average(y_target, smooth_window)
    dy = np.diff(y_smooth)
    if dy.size < 2:
        return ()

    signs = _fill_zero_signs(np.sign(dy))
    min_move = max(
        float(np.ptp(y_smooth)) * MIN_REVERSAL_AMPLITUDE_FRACTION,
        float(np.median(np.abs(dy))) * 5.0,
        1e-6,
    )
    candidates: list[tuple[int, str, int]] = []
    last_kept_index = -min_span_samples
    for i in range(1, signs.size):
        prev_sign = int(signs[i - 1])
        next_sign = int(signs[i])
        if prev_sign == 0 or next_sign == 0 or prev_sign == next_sign:
            continue

        idx = i
        if idx < min_span_samples or idx >= y_target.size - min_span_samples:
            continue
        if idx - last_kept_index < min_span_samples:
            continue

        pre_idx = idx - min_span_samples
        post_idx = idx + min_span_samples
        pre_move = abs(float(y_smooth[idx] - y_smooth[pre_idx]))
        post_move = abs(float(y_smooth[post_idx] - y_smooth[idx]))
        if pre_move < min_move or post_move < min_move:
            continue

        if prev_sign > 0 and next_sign < 0:
            candidates.append((idx, "peak", -1))
        elif prev_sign < 0 and next_sign > 0:
            candidates.append((idx, "valley", 1))
        else:
            continue
        last_kept_index = idx
    return tuple(candidates)


def _find_current_turn_index(
    y_current: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
    reversal_kind: str,
    dt: float,
) -> int:
    """Return the first aligned-current turning point after a target reversal."""
    if end_idx <= start_idx:
        return start_idx

    smooth_window = _odd_window_samples(dt, SMOOTHING_WINDOW_S, minimum=3)
    y_smooth = _moving_average(y_current[start_idx : end_idx + 1], smooth_window)
    dy = np.diff(y_smooth)
    if dy.size == 0:
        return start_idx

    signs = _fill_zero_signs(np.sign(dy))
    if reversal_kind == "peak":
        for rel_idx, sign in enumerate(signs):
            if sign <= 0:
                return start_idx + rel_idx
    else:
        for rel_idx, sign in enumerate(signs):
            if sign >= 0:
                return start_idx + rel_idx
    return end_idx


def _measure_overshoot_events(
    *,
    axis: str,
    chunk_index: int,
    window_start_s: float,
    window_end_s: float,
    reference_start_s: float,
    spectral_centroid_hz: float,
    t_grid: np.ndarray,
    y_target: np.ndarray,
    y_current_shifted: np.ndarray,
    dt: float,
) -> tuple[OvershootEvent, ...]:
    """Detect target reversals and measure aligned-current overshoot per event."""
    reversal_candidates = _find_reversal_candidates(y_target, dt)
    if not reversal_candidates:
        return ()

    events: list[OvershootEvent] = []
    for event_num, (reversal_idx, reversal_kind, new_direction) in enumerate(reversal_candidates):
        next_reversal_idx = (
            reversal_candidates[event_num + 1][0]
            if event_num + 1 < len(reversal_candidates)
            else y_target.size - 1
        )
        turn_idx = _find_current_turn_index(
            y_current_shifted,
            start_idx=reversal_idx,
            end_idx=next_reversal_idx,
            reversal_kind=reversal_kind,
            dt=dt,
        )

        reversal_value = float(y_target[reversal_idx])
        current_segment = y_current_shifted[reversal_idx : turn_idx + 1]
        if current_segment.size == 0:
            continue

        if reversal_kind == "peak":
            current_turn_value = float(np.max(current_segment))
            overshoot_value = max(0.0, current_turn_value - reversal_value)
            commanded_direction = "decreasing"
        else:
            current_turn_value = float(np.min(current_segment))
            overshoot_value = max(0.0, reversal_value - current_turn_value)
            commanded_direction = "increasing"

        reversal_time_s = float(t_grid[reversal_idx])
        events.append(
            OvershootEvent(
                axis=axis,
                chunk_index=chunk_index,
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                spectral_centroid_hz=float(spectral_centroid_hz),
                reversal_kind=reversal_kind,
                commanded_direction=commanded_direction,
                reversal_index=int(reversal_idx),
                reversal_timestamp_s=reversal_time_s,
                reversal_time_since_recording_start_s=float(reversal_time_s - reference_start_s),
                reversal_time_since_chunk_start_s=float(reversal_time_s - (reference_start_s + window_start_s)),
                reversal_value=reversal_value,
                current_turn_timestamp_s=float(t_grid[turn_idx]),
                current_turn_value=current_turn_value,
                overshoot_value=float(overshoot_value),
            )
        )
    return tuple(events)


def _analyze_axis(
    name: str,
    axis: PitchData | YawData,
) -> AxisAnalysis:
    """Compute chunked overshoot metrics for one axis."""
    target = axis.target_except_firing
    current = axis.current_except_firing

    target_times_s = _timestamps_to_seconds(target.timestamps)
    current_times_s = _timestamps_to_seconds(current.timestamps)
    if target_times_s.size == 0 or current_times_s.size == 0:
        return AxisAnalysis(
            axis=name,
            total_chunk_pairs=0,
            selected_chunk_pairs=0,
            chunks=(),
        )

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

    paired_indices = sorted(set(target_chunks) & set(current_chunks))
    analyses: list[ChunkAnalysis] = []
    for chunk_index in paired_indices:
        target_chunk = target_chunks[chunk_index]
        current_chunk = current_chunks[chunk_index]
        centroid_hz = _spectral_centroid_hz(target_chunk.stream)
        if centroid_hz is None or centroid_hz < MIN_CENTROID_HZ:
            continue

        aligned = _aligned_uniform_series(target_chunk.stream, current_chunk.stream)
        if aligned is None:
            continue

        t_grid, y_target, y_current, dt = aligned
        lag_samples, latency_s = _latency_from_correlate(y_target, y_current, dt)
        y_current_shifted = _shift_current_backward(t_grid, y_current, latency_s)

        valid = np.isfinite(y_current_shifted)
        if np.count_nonzero(valid) < 3:
            continue
        first_valid = int(np.argmax(valid))
        last_valid = int(valid.size - 1 - np.argmax(valid[::-1]))
        if last_valid - first_valid + 1 < 3:
            continue

        t_valid = t_grid[first_valid : last_valid + 1]
        y_target_valid = y_target[first_valid : last_valid + 1]
        y_current_valid = y_current_shifted[first_valid : last_valid + 1]
        events = _measure_overshoot_events(
            axis=name,
            chunk_index=chunk_index,
            window_start_s=target_chunk.window_start_s,
            window_end_s=target_chunk.window_end_s,
            reference_start_s=reference_start_s,
            spectral_centroid_hz=float(centroid_hz),
            t_grid=t_valid,
            y_target=y_target_valid,
            y_current_shifted=y_current_valid,
            dt=dt,
        )
        analyses.append(
            ChunkAnalysis(
                axis=name,
                chunk_index=chunk_index,
                window_start_s=target_chunk.window_start_s,
                window_end_s=target_chunk.window_end_s,
                target_samples=int(np.asarray(target_chunk.stream.values).size),
                current_samples=int(np.asarray(current_chunk.stream.values).size),
                aligned_samples=int(t_valid.size),
                dt_s=float(dt),
                spectral_centroid_hz=float(centroid_hz),
                lag_samples=lag_samples,
                latency_s=float(latency_s),
                reversal_count=len(events),
                events=events,
            )
        )

    return AxisAnalysis(
        axis=name,
        total_chunk_pairs=len(paired_indices),
        selected_chunk_pairs=len(analyses),
        chunks=tuple(analyses),
    )


def analyze_overshoot() -> OvershootAnalysis:
    """Run chunked overshoot analysis for both pitch and yaw."""
    result = load_all_streams()
    print("available_streams:", result.available_streams)
    return OvershootAnalysis(
        pitch=_analyze_axis("pitch", result.bundle.pitch),
        yaw=_analyze_axis("yaw", result.bundle.yaw),
    )


def _iter_events(axis_analysis: AxisAnalysis):
    for chunk in axis_analysis.chunks:
        yield from chunk.events


def _print_axis_report(axis_analysis: AxisAnalysis) -> None:
    event_count = sum(chunk.reversal_count for chunk in axis_analysis.chunks)
    print(
        f"{axis_analysis.axis}: "
        f"chunk_pairs={axis_analysis.total_chunk_pairs} "
        f"selected_chunks={axis_analysis.selected_chunk_pairs} "
        f"reversal_events={event_count}"
    )
    for chunk in axis_analysis.chunks:
        window_start = format_recording_offset_timestamp(chunk.window_start_s)
        window_end = format_recording_offset_timestamp(chunk.window_end_s)
        print(
            "  "
            f"chunk={chunk.chunk_index:02d} "
            f"window=[{window_start}, {window_end}) "
            f"centroid_hz={chunk.spectral_centroid_hz:.6g} "
            f"latency_s={chunk.latency_s:.6g} "
            f"aligned_n={chunk.aligned_samples} "
            f"reversals={chunk.reversal_count}"
        )


def main() -> None:
    analysis = analyze_overshoot()
    _print_axis_report(analysis.pitch)
    _print_axis_report(analysis.yaw)
    print(f"total_events={len(analysis.events)}")


if __name__ == "__main__":
    main()
