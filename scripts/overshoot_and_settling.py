#!/usr/bin/env python3
"""Whole-dataset overshoot analysis for except-firing motor signals.

For each axis, this script:
- loads the ``*_except_firing`` target/current motor streams
- computes one global latency across the full dataset
- shifts ``current`` backward by that latency to align it with ``target``
- runs ``_find_reversal_candidates()`` on overlapping sliding windows
- deduplicates reversal timestamps and measures overshoot per event

Each detected event keeps both UTC timestamps for inspection and numeric
timestamp/overshoot values for later plotting.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

# Allow running without installing the package (repo checkout)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from actuator_analysis.config_loader import format_recording_offset_timestamp, key_time_points
from actuator_analysis.extract_all_data import load_all_streams
from actuator_analysis.latency import aligned_uniform_series, latency_from_correlate
from actuator_analysis.load_data import PitchData, YawData
from actuator_analysis.plot_streams import _values_1d_float

SLIDING_WINDOW_DURATION_S = 60.0
SLIDING_WINDOW_STEP_S = 30.0
SMOOTHING_WINDOW_S = 0.10
MIN_REVERSAL_SPAN_S = 0.20
MIN_REVERSAL_AMPLITUDE_FRACTION = 0.02
REVERSAL_CONFIRMATION_S = 0.30
MIN_OVERSHOOT_REFERENCE_FRACTION = 0.01
SETTLING_THRESHOLD_FRACTION = 0.05
LARGE_OVERSHOOT_PERCENT_THRESHOLD = 10.0


@dataclass(frozen=True)
class OvershootEvent:
    """Overshoot measured around one target reversal."""

    axis: str
    reversal_kind: str
    commanded_direction: str
    reversal_index: int
    reversal_timestamp_s: float
    reversal_timestamp_utc: str
    reversal_time_since_recording_start_s: float
    reversal_value: float
    current_turn_index: int
    current_turn_timestamp_s: float
    current_turn_timestamp_utc: str
    current_turn_value: float
    overshoot_percent: float
    settling_time_s: float | None


@dataclass(frozen=True)
class AxisAnalysis:
    """Overshoot results for one full-dataset axis analysis."""

    axis: str
    target_samples: int
    current_samples: int
    aligned_samples: int
    dt_s: float
    lag_samples: int
    latency_s: float
    reversal_count: int
    events: tuple[OvershootEvent, ...]

    @property
    def average_overshoot_percent(self) -> float | None:
        """Return the mean overshoot percent across detected events."""
        if not self.events:
            return None
        return float(np.mean([event.overshoot_percent for event in self.events]))

    @property
    def large_overshoot_count(self) -> int:
        """Return how many events exceed the large-overshoot threshold."""
        return sum(
            1 for event in self.events if event.overshoot_percent > LARGE_OVERSHOOT_PERCENT_THRESHOLD
        )

    @property
    def average_settling_time_s(self) -> float | None:
        """Return the mean settling time across positive overshoot events."""
        settling_times = [
            event.settling_time_s
            for event in self.events
            if event.overshoot_percent > 0.0 and event.settling_time_s is not None
        ]
        if not settling_times:
            return None
        return float(np.mean(settling_times))


@dataclass(frozen=True)
class OvershootAnalysis:
    """Whole-dataset overshoot analysis for pitch and yaw."""

    pitch: AxisAnalysis
    yaw: AxisAnalysis

    @property
    def events(self) -> tuple[OvershootEvent, ...]:
        """Flatten all overshoot events for later plotting."""
        return self.pitch.events + self.yaw.events

    @property
    def average_settling_time_s(self) -> float | None:
        """Return the mean settling time across all positive overshoot events."""
        settling_times = [
            event.settling_time_s
            for event in self.events
            if event.overshoot_percent > 0.0 and event.settling_time_s is not None
        ]
        if not settling_times:
            return None
        return float(np.mean(settling_times))


def _odd_window_samples(dt: float, duration_s: float, *, minimum: int = 1) -> int:
    samples = max(minimum, int(round(duration_s / dt)))
    if samples % 2 == 0:
        samples += 1
    return samples


def _window_samples(dt: float, duration_s: float, *, minimum: int = 1) -> int:
    return max(minimum, int(round(duration_s / dt)))


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
    confirmation_samples = _window_samples(dt, REVERSAL_CONFIRMATION_S, minimum=2)
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
        if idx < confirmation_samples or idx >= y_target.size - confirmation_samples:
            continue
        if idx - last_kept_index < min_span_samples:
            continue

        pre_signs = signs[idx - confirmation_samples : idx]
        post_signs = signs[idx : idx + confirmation_samples]
        if pre_signs.size < confirmation_samples or post_signs.size < confirmation_samples:
            continue
        if not np.all(pre_signs == prev_sign) or not np.all(post_signs == next_sign):
            continue

        pre_idx = idx - confirmation_samples
        post_idx = idx + confirmation_samples
        neighborhood = y_smooth[pre_idx : post_idx + 1]
        if neighborhood.size < 3:
            continue

        if prev_sign > 0 and next_sign < 0:
            local_offset = int(np.argmax(neighborhood))
            reversal_kind = "peak"
            new_direction = -1
        elif prev_sign < 0 and next_sign > 0:
            local_offset = int(np.argmin(neighborhood))
            reversal_kind = "valley"
            new_direction = 1
        else:
            continue

        extreme_idx = pre_idx + local_offset
        if extreme_idx < min_span_samples or extreme_idx >= y_target.size - min_span_samples:
            continue
        if extreme_idx - last_kept_index < min_span_samples:
            continue

        left_shoulder = y_smooth[extreme_idx - confirmation_samples : extreme_idx]
        right_shoulder = y_smooth[extreme_idx + 1 : extreme_idx + 1 + confirmation_samples]
        if left_shoulder.size < confirmation_samples or right_shoulder.size < confirmation_samples:
            continue

        extreme_value = float(y_smooth[extreme_idx])
        left_level = float(np.median(left_shoulder))
        right_level = float(np.median(right_shoulder))
        pre_move = abs(extreme_value - left_level)
        post_move = abs(right_level - extreme_value)
        if pre_move < min_move or post_move < min_move:
            continue

        if reversal_kind == "peak":
            prominence = extreme_value - max(left_level, right_level)
        else:
            prominence = min(left_level, right_level) - extreme_value
        if prominence < min_move:
            continue

        candidates.append((extreme_idx, reversal_kind, new_direction))
        last_kept_index = extreme_idx
    return tuple(candidates)


def _sliding_window_bounds(
    sample_count: int,
    dt: float,
) -> tuple[tuple[int, int], ...]:
    """Return ``(start, end)`` bounds for overlapping reversal-detection windows."""
    if sample_count <= 0:
        return ()

    window_samples = _window_samples(dt, SLIDING_WINDOW_DURATION_S, minimum=3)
    step_samples = _window_samples(dt, SLIDING_WINDOW_STEP_S, minimum=1)
    if sample_count <= window_samples:
        return ((0, sample_count),)

    last_start = sample_count - window_samples
    starts = list(range(0, last_start + 1, step_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    return tuple((start, min(sample_count, start + window_samples)) for start in starts)


def _dedupe_reversal_candidates(
    candidates: list[tuple[int, str, int]],
    dt: float,
) -> tuple[tuple[int, str, int], ...]:
    """Collapse overlapping window detections into one global event per reversal."""
    if not candidates:
        return ()

    dedupe_samples = _window_samples(dt, MIN_REVERSAL_SPAN_S, minimum=1)
    ordered = sorted(candidates, key=lambda item: (item[0], item[1]))
    clusters: list[list[tuple[int, str, int]]] = [[ordered[0]]]
    for candidate in ordered[1:]:
        cluster = clusters[-1]
        last_index, last_kind, _ = cluster[-1]
        if candidate[1] == last_kind and candidate[0] - last_index <= dedupe_samples:
            cluster.append(candidate)
            continue
        clusters.append([candidate])

    merged: list[tuple[int, str, int]] = []
    for cluster in clusters:
        indices = [item[0] for item in cluster]
        kind = cluster[len(cluster) // 2][1]
        new_direction = -1 if kind == "peak" else 1
        merged.append((int(round(float(np.median(indices)))), kind, new_direction))
    return tuple(merged)


def _find_sliding_window_reversal_candidates(
    y_target: np.ndarray,
    dt: float,
) -> tuple[tuple[int, str, int], ...]:
    """Detect target reversals over overlapping windows and map them to global indices."""
    raw_candidates: list[tuple[int, str, int]] = []
    for start_idx, end_idx in _sliding_window_bounds(y_target.size, dt):
        window = y_target[start_idx:end_idx]
        if window.size < 3:
            continue
        for local_index, reversal_kind, new_direction in _find_reversal_candidates(window, dt):
            raw_candidates.append((start_idx + local_index, reversal_kind, new_direction))
    return _dedupe_reversal_candidates(raw_candidates, dt)


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


def _recording_start_s() -> float:
    """Return the configured recording start as Unix seconds in UTC."""
    recording_start = key_time_points().recording_start.replace(tzinfo=timezone.utc)
    return float(recording_start.timestamp())


def _format_utc_timestamp(timestamp_s: float, recording_start_s: float) -> str:
    """Format one absolute Unix timestamp using the repository's UTC helper."""
    offset_s = float(timestamp_s - recording_start_s)
    return format_recording_offset_timestamp(offset_s)


def _measure_settling_time_s(
    *,
    t_grid: np.ndarray,
    y_target: np.ndarray,
    y_current_shifted: np.ndarray,
    reversal_idx: int,
    end_idx: int,
    reference_value: float,
) -> float | None:
    """Return the first time current enters and stays within the 5% target band."""
    if end_idx < reversal_idx:
        return None

    tolerance = max(abs(reference_value) * SETTLING_THRESHOLD_FRACTION, 1e-6)
    target_segment = y_target[reversal_idx : end_idx + 1]
    current_segment = y_current_shifted[reversal_idx : end_idx + 1]
    if target_segment.size == 0 or current_segment.size != target_segment.size:
        return None

    in_band = np.abs(current_segment - target_segment) <= tolerance
    stays_settled = np.logical_and.accumulate(in_band[::-1])[::-1]
    if not np.any(stays_settled):
        return None

    settling_rel_idx = int(np.argmax(stays_settled))
    settling_idx = reversal_idx + settling_rel_idx
    return float(t_grid[settling_idx] - t_grid[reversal_idx])


def _measure_overshoot_events(
    *,
    axis: str,
    recording_start_s: float,
    t_grid: np.ndarray,
    y_target: np.ndarray,
    y_current_shifted: np.ndarray,
    dt: float,
) -> tuple[OvershootEvent, ...]:
    """Detect target reversals and measure aligned-current overshoot per event."""
    reversal_candidates = _find_sliding_window_reversal_candidates(y_target, dt)
    if not reversal_candidates:
        return ()

    # Percent overshoot becomes unstable when the target at the reversal is too close to zero.
    min_reference_value = max(float(np.ptp(y_target)) * MIN_OVERSHOOT_REFERENCE_FRACTION, 1e-6)
    events: list[OvershootEvent] = []
    for event_num, (reversal_idx, reversal_kind, _new_direction) in enumerate(reversal_candidates):
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
            commanded_direction = "decreasing"
            turn_idx = reversal_idx + int(np.argmax(current_segment))
        else:
            current_turn_value = float(np.min(current_segment))
            commanded_direction = "increasing"
            turn_idx = reversal_idx + int(np.argmin(current_segment))

        if abs(reversal_value) < min_reference_value:
            continue
        overshoot_percent = 100.0 * ((current_turn_value - reversal_value) / reversal_value)
        settling_time_s = None
        if overshoot_percent > 0.0:
            settling_time_s = _measure_settling_time_s(
                t_grid=t_grid,
                y_target=y_target,
                y_current_shifted=y_current_shifted,
                reversal_idx=int(reversal_idx),
                end_idx=int(next_reversal_idx),
                reference_value=reversal_value,
            )

        reversal_time_s = float(t_grid[reversal_idx])
        current_turn_time_s = float(t_grid[turn_idx])
        reversal_offset_s = float(reversal_time_s - recording_start_s)
        events.append(
            OvershootEvent(
                axis=axis,
                reversal_kind=reversal_kind,
                commanded_direction=commanded_direction,
                reversal_index=int(reversal_idx),
                reversal_timestamp_s=reversal_time_s,
                reversal_timestamp_utc=_format_utc_timestamp(reversal_time_s, recording_start_s),
                reversal_time_since_recording_start_s=reversal_offset_s,
                reversal_value=reversal_value,
                current_turn_index=int(turn_idx),
                current_turn_timestamp_s=current_turn_time_s,
                current_turn_timestamp_utc=_format_utc_timestamp(current_turn_time_s, recording_start_s),
                current_turn_value=current_turn_value,
                overshoot_percent=float(overshoot_percent),
                settling_time_s=settling_time_s,
            )
        )
    return tuple(events)


def _empty_axis_analysis(
    axis: str,
    *,
    target_samples: int = 0,
    current_samples: int = 0,
) -> AxisAnalysis:
    return AxisAnalysis(
        axis=axis,
        target_samples=target_samples,
        current_samples=current_samples,
        aligned_samples=0,
        dt_s=0.0,
        lag_samples=0,
        latency_s=0.0,
        reversal_count=0,
        events=(),
    )


def _analyze_axis(
    name: str,
    axis: PitchData | YawData,
) -> AxisAnalysis:
    """Compute whole-dataset overshoot metrics for one axis."""
    target = axis.target_except_firing
    current = axis.current_except_firing
    target_samples = int(_values_1d_float(target.values).size)
    current_samples = int(_values_1d_float(current.values).size)

    aligned = aligned_uniform_series(target, current)
    if aligned is None:
        return _empty_axis_analysis(
            name,
            target_samples=target_samples,
            current_samples=current_samples,
        )

    lag_samples, latency_s = latency_from_correlate(
        aligned.y_target,
        aligned.y_current,
        aligned.dt_s,
    )
    y_current_shifted = _shift_current_backward(aligned.t_grid, aligned.y_current, latency_s)

    valid = np.isfinite(y_current_shifted)
    if np.count_nonzero(valid) < 3:
        return _empty_axis_analysis(
            name,
            target_samples=target_samples,
            current_samples=current_samples,
        )

    # The latency shift only creates invalid edges, so trim to the largest valid span once.
    first_valid = int(np.argmax(valid))
    last_valid = int(valid.size - 1 - np.argmax(valid[::-1]))
    if last_valid - first_valid + 1 < 3:
        return _empty_axis_analysis(
            name,
            target_samples=target_samples,
            current_samples=current_samples,
        )

    t_valid = aligned.t_grid[first_valid : last_valid + 1]
    y_target_valid = aligned.y_target[first_valid : last_valid + 1]
    y_current_valid = y_current_shifted[first_valid : last_valid + 1]
    events = _measure_overshoot_events(
        axis=name,
        recording_start_s=_recording_start_s(),
        t_grid=t_valid,
        y_target=y_target_valid,
        y_current_shifted=y_current_valid,
        dt=aligned.dt_s,
    )
    return AxisAnalysis(
        axis=name,
        target_samples=target_samples,
        current_samples=current_samples,
        aligned_samples=int(t_valid.size),
        dt_s=float(aligned.dt_s),
        lag_samples=lag_samples,
        latency_s=float(latency_s),
        reversal_count=len(events),
        events=events,
    )


def analyze_overshoot() -> OvershootAnalysis:
    """Run whole-dataset overshoot analysis for both pitch and yaw."""
    result = load_all_streams()
    print("available_streams:", result.available_streams)
    return OvershootAnalysis(
        pitch=_analyze_axis("pitch", result.bundle.pitch),
        yaw=_analyze_axis("yaw", result.bundle.yaw),
    )


def _print_axis_report(axis_analysis: AxisAnalysis) -> None:
    average_overshoot_percent = axis_analysis.average_overshoot_percent
    average_overshoot_text = (
        f"{average_overshoot_percent:.6g}%"
        if average_overshoot_percent is not None
        else "n/a"
    )
    average_settling_time_s = axis_analysis.average_settling_time_s
    average_settling_text = (
        f"{average_settling_time_s:.6g}s"
        if average_settling_time_s is not None
        else "n/a"
    )
    print(
        f"{axis_analysis.axis}: "
        f"target_n={axis_analysis.target_samples} "
        f"current_n={axis_analysis.current_samples} "
        f"aligned_n={axis_analysis.aligned_samples} "
        f"latency_s={axis_analysis.latency_s:.6g} "
        f"reversal_events={axis_analysis.reversal_count} "
        f"average_overshoot_percent={average_overshoot_text} "
        f"average_settling_time_s={average_settling_text} "
        f"overshoots_gt_{LARGE_OVERSHOOT_PERCENT_THRESHOLD:.0f}pct="
        f"{axis_analysis.large_overshoot_count}"
    )
    for event in axis_analysis.events:
        settling_text = f"{event.settling_time_s:.6g}s" if event.settling_time_s is not None else "n/a"
        print(
            "  "
            f"t={event.reversal_timestamp_utc} "
            f"kind={event.reversal_kind} "
            f"direction={event.commanded_direction} "
            f"overshoot_percent={event.overshoot_percent:.6g}% "
            f"settling_time_s={settling_text}"
        )


def main() -> None:
    analysis = analyze_overshoot()
    _print_axis_report(analysis.pitch)
    _print_axis_report(analysis.yaw)
    print(f"total_events={len(analysis.events)}")
    average_settling_time_s = analysis.average_settling_time_s
    average_settling_text = (
        f"{average_settling_time_s:.6g}s"
        if average_settling_time_s is not None
        else "n/a"
    )
    print(f"average_settling_time_s={average_settling_text}")


if __name__ == "__main__":
    main()
