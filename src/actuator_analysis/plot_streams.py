"""Plot motor streams (value vs time)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from actuator_analysis.load_data import StreamData


def _values_1d_float(values: np.ndarray) -> np.ndarray:
    """Rerun scalars sometimes arrive as ``object`` dtype of length-1 arrays."""
    v = np.asarray(values)
    if v.dtype == object:
        return np.array([np.asarray(x, dtype=np.float64).ravel()[0] for x in v])
    return np.asarray(v, dtype=np.float64).ravel()


def _seconds_since_start(timestamps: np.ndarray) -> np.ndarray:
    """Align timestamps to float seconds from the first sample.

    Rerun ``log_time`` timelines use ``datetime64``; motor streams may use float seconds.
    """
    t = np.asarray(timestamps)
    if np.issubdtype(t.dtype, np.datetime64):
        t0 = np.min(t)
        return (t - t0) / np.timedelta64(1, "s")
    return np.asarray(t, dtype=np.float64) - float(np.min(t))


def _earliest_timestamp(*timestamps: np.ndarray) -> np.datetime64 | np.floating | float:
    return min(np.min(np.asarray(t)) for t in timestamps)


def _seconds_since_reference(timestamps: np.ndarray, t_ref: np.datetime64 | np.floating | float) -> np.ndarray:
    """Seconds since ``t_ref`` (must match the dtype family of ``timestamps``)."""
    t = np.asarray(timestamps)
    if np.issubdtype(t.dtype, np.datetime64):
        return (t - t_ref) / np.timedelta64(1, "s")
    return np.asarray(t, dtype=np.float64) - float(t_ref)


def plot_stream(
    stream: StreamData,
    *,
    title: str | None = None,
    show: bool = True,
    save_path: Path | str | None = None,
    first_seconds: float | None = None,
) -> None:
    """Plot ``stream.values`` vs ``stream.timestamps`` (same length, Rerun timeline).

    If ``first_seconds`` is set, only points from the start of the series
    (``min(timestamps)`` through ``min(timestamps) + first_seconds``) are plotted.

    If ``save_path`` is set, writes a PNG there (creates parent directories as needed).
    """
    t = np.asarray(stream.timestamps)
    y = _values_1d_float(stream.values)
    if first_seconds is not None:
        t_sec = _seconds_since_start(t)
        mask = t_sec <= first_seconds
        t = t_sec[mask]
        y = y[mask]
        xlabel = f"time since first sample (s) — {stream.timeline}"
    else:
        xlabel = f"time ({stream.timeline})"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(stream.column_name or "value")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{stream.component}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    if show and not isinstance(fig.canvas, FigureCanvasAgg):
        plt.show()
    plt.close(fig)


def plot_two_streams(
    stream_a: StreamData,
    stream_b: StreamData,
    *,
    labels: tuple[str, str],
    colors: tuple[str, str] = ("blue", "red"),
    title: str | None = None,
    show: bool = True,
    save_path: Path | str | None = None,
    first_seconds: float | None = None,
) -> None:
    """Plot two streams on one axes with a shared time axis."""
    t_a = np.asarray(stream_a.timestamps)
    t_b = np.asarray(stream_b.timestamps)
    y_a = _values_1d_float(stream_a.values)
    y_b = _values_1d_float(stream_b.values)

    if first_seconds is not None:
        t_ref = _earliest_timestamp(t_a, t_b)
        t_sec_a = _seconds_since_reference(t_a, t_ref)
        t_sec_b = _seconds_since_reference(t_b, t_ref)
        mask_a = t_sec_a <= first_seconds
        mask_b = t_sec_b <= first_seconds
        t_a = t_sec_a[mask_a]
        y_a = y_a[mask_a]
        t_b = t_sec_b[mask_b]
        y_b = y_b[mask_b]
        xlabel = f"time since earliest sample (s) — {stream_a.timeline}"
    else:
        xlabel = f"time ({stream_a.timeline})"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_a, y_a, linewidth=0.8, color=colors[0], label=labels[0])
    ax.plot(t_b, y_b, linewidth=0.8, color=colors[1], label=labels[1])
    ax.set_xlabel(xlabel)
    if stream_a.column_name == stream_b.column_name:
        ax.set_ylabel(stream_a.column_name or "value")
    else:
        ax.set_ylabel(f"{stream_a.column_name} / {stream_b.column_name}")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{stream_a.component} + {stream_b.component}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    if show and not isinstance(fig.canvas, FigureCanvasAgg):
        plt.show()
    plt.close(fig)
