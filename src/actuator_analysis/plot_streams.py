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


def plot_stream(
    stream: StreamData,
    *,
    title: str | None = None,
    show: bool = True,
    save_path: Path | str | None = None,
) -> None:
    """Plot ``stream.values`` vs ``stream.timestamps`` (same length, Rerun timeline).

    If ``save_path`` is set, writes a PNG there (creates parent directories as needed).
    """
    t = np.asarray(stream.timestamps)
    y = _values_1d_float(stream.values)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, linewidth=0.8)
    ax.set_xlabel(f"time ({stream.timeline})")
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
