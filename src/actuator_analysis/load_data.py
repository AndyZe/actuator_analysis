"""Load configured Rerun data and extract streams to NumPy arrays."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from actuator_analysis.config_loader import data_path


class LoadedRecording:
    """Holds a local Rerun catalog server and dataset for querying (``rerun-sdk`` 0.29+)."""

    __slots__ = ("_dataset", "_server")

    def __init__(self, server: Any, dataset: Any) -> None:
        self._server = server
        self._dataset = dataset

    @property
    def dataset(self) -> Any:
        return self._dataset

    def schema(self) -> Any:
        return self._dataset.schema()

    def close(self) -> None:
        self._server.shutdown()


@dataclass(frozen=True)
class StreamData:
    """Time-aligned values extracted from a single Rerun stream."""

    timestamps: np.ndarray
    values: np.ndarray
    timeline: str
    component: str
    column_name: str


def resolve_recording_path(*, config_name: str = "defaults"):
    """Resolve the recording path from the named config."""
    return data_path(config_name)


def load_recording(*, config_name: str = "defaults") -> LoadedRecording:
    """Load the configured Rerun recording via a local catalog server (query API).

    Call :meth:`LoadedRecording.close` when finished to shut down the server.
    """
    rr = _import_rerun()
    recording_path = resolve_recording_path(config_name=config_name)
    path_str = str(recording_path)
    if not Path(path_str).is_file():
        raise FileNotFoundError(f"Recording file not found: {path_str}")

    server = rr.server.Server(datasets={"_actuator": [path_str]})
    try:
        client = server.client()
    except Exception as exc:
        server.shutdown()
        raise ModuleNotFoundError(
            "Loading recordings for analysis requires the DataFusion extra. "
            "Install with: pip install 'rerun-sdk[datafusion]'"
        ) from exc

    dataset = client.get_dataset("_actuator")
    return LoadedRecording(server=server, dataset=dataset)


def available_timelines(recording: Any) -> tuple[str, ...]:
    """Return timeline names available in a recording schema."""
    return tuple(column.name for column in recording.schema().index_columns())


def available_streams(recording: Any) -> tuple[str, ...]:
    """Return unique entity paths available for extraction."""
    entity_paths = {
        column.entity_path
        for column in recording.schema().component_columns()
        if not getattr(column, "is_static", False)
    }
    return tuple(sorted(entity_paths))


def extract_stream(
    recording: Any,
    stream_path: str,
    *,
    timeline: str | None = None,
    component: str | None = None,
    fill_latest_at: bool = False,
    exclude_time_range: tuple[datetime, datetime] | None = None,
) -> StreamData:
    """Extract one stream from a recording into NumPy arrays.

    If ``exclude_time_range`` is set ``(start, end)``, samples whose timeline value falls
    in the closed interval ``[start, end]`` (compared as Unix time in seconds, with naive
    datetimes interpreted as UTC) are dropped. Typical use: omit data during a trigger window.
    """
    if isinstance(recording, LoadedRecording):
        return _extract_stream_from_dataset(
            recording.dataset,
            stream_path,
            timeline=timeline,
            component=component,
            fill_latest_at=fill_latest_at,
            exclude_time_range=exclude_time_range,
        )

    timeline_name = _resolve_timeline(recording, timeline)
    component_column = _resolve_component_column(recording.schema(), stream_path, component)

    view = recording.view(index=timeline_name, contents=stream_path)
    if fill_latest_at:
        view = view.fill_latest_at()
    else:
        view = view.filter_is_not_null(component_column.name)

    table = view.select(timeline_name, component_column.name).read_all()
    timestamps = _column_to_numpy(table.column(timeline_name))
    values = _column_to_numpy(table.column(component_column.name))

    if exclude_time_range is not None:
        lo, hi = exclude_time_range
        mask = _mask_keep_outside_time_range(timestamps, lo, hi)
        timestamps = np.asarray(timestamps)[mask]
        values = np.asarray(values)[mask]

    return StreamData(
        timestamps=timestamps,
        values=values,
        timeline=timeline_name,
        component=component_column.component,
        column_name=component_column.name,
    )


def _extract_stream_from_dataset(
    dataset: Any,
    stream_path: str,
    *,
    timeline: str | None,
    component: str | None,
    fill_latest_at: bool,
    exclude_time_range: tuple[datetime, datetime] | None,
) -> StreamData:
    from datafusion import col

    timeline_name = _resolve_timeline(dataset, timeline)
    component_column = _resolve_component_column(dataset.schema(), stream_path, component)

    view = dataset.filter_contents([stream_path])
    df = view.reader(index=timeline_name, fill_latest_at=fill_latest_at)
    if not fill_latest_at:
        df = df.filter(col(component_column.name).is_not_null())
    df = df.select(timeline_name, component_column.name)
    table = df.to_arrow_table()
    timestamps = _column_to_numpy(table.column(timeline_name))
    values = _column_to_numpy(table.column(component_column.name))

    if exclude_time_range is not None:
        lo, hi = exclude_time_range
        mask = _mask_keep_outside_time_range(timestamps, lo, hi)
        timestamps = np.asarray(timestamps)[mask]
        values = np.asarray(values)[mask]

    return StreamData(
        timestamps=timestamps,
        values=values,
        timeline=timeline_name,
        component=component_column.component,
        column_name=component_column.name,
    )


def load_streams(
    stream_paths: str | list[str] | tuple[str, ...],
    *,
    config_name: str = "defaults",
    timeline: str | None = None,
    component_overrides: dict[str, str] | None = None,
    fill_latest_at: bool = False,
    recording: Any | None = None,
) -> dict[str, StreamData]:
    """Load a recording once, then extract the requested streams."""
    loaded_recording = recording if recording is not None else load_recording(config_name=config_name)
    requested_paths = [stream_paths] if isinstance(stream_paths, str) else list(stream_paths)
    overrides = component_overrides or {}

    return {
        stream_path: extract_stream(
            loaded_recording,
            stream_path,
            timeline=timeline,
            component=overrides.get(stream_path),
            fill_latest_at=fill_latest_at,
        )
        for stream_path in requested_paths
    }


def _datetime_to_unix_seconds(dt: datetime) -> float:
    """Convert to Unix seconds; naive datetimes are treated as UTC."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).timestamp()
    return float(calendar.timegm(dt.utctimetuple())) + dt.microsecond * 1e-6


def _timestamps_as_unix_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Rerun timelines are often int64 nanoseconds since epoch; otherwise seconds as float."""
    t = np.asarray(timestamps, dtype=np.float64)
    if t.size == 0:
        return t
    max_abs = float(np.nanmax(np.abs(t)))
    if max_abs > 1e12:
        return t * 1e-9
    return t


def _mask_keep_outside_time_range(
    timestamps: np.ndarray,
    lo: datetime,
    hi: datetime,
) -> np.ndarray:
    if lo > hi:
        raise ValueError("exclude_time_range must satisfy start <= end")
    lo_s = _datetime_to_unix_seconds(lo)
    hi_s = _datetime_to_unix_seconds(hi)
    t_s = _timestamps_as_unix_seconds(timestamps)
    in_excluded = (t_s >= lo_s) & (t_s <= hi_s)
    return ~in_excluded


def _import_rerun() -> Any:
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rerun is required to load .rrd recordings. Install the package with rerun-sdk available."
        ) from exc
    return rr


def _resolve_timeline(recording: Any, timeline: str | None) -> str:
    available = available_timelines(recording)
    if not available:
        raise ValueError("Recording does not expose any timelines.")

    if timeline is not None:
        if timeline not in available:
            raise KeyError(f"Unknown timeline {timeline!r}. Available timelines: {available}")
        return timeline

    if len(available) == 1:
        return available[0]

    for preferred in ("timestamp", "time", "log_time", "frame_nr", "frame"):
        if preferred in available:
            return preferred

    raise ValueError(f"Recording has multiple timelines; choose one explicitly from {available}.")


def _resolve_component_column(schema: Any, stream_path: str, component: str | None) -> Any:
    candidates = [
        column
        for column in schema.component_columns()
        if column.entity_path == stream_path and not getattr(column, "is_static", False)
    ]
    if not candidates:
        raise KeyError(f"No stream found for {stream_path!r}.")

    matches = candidates if component is None else [column for column in candidates if _component_matches(column, component)]
    if not matches:
        available = tuple(column.component for column in candidates)
        raise KeyError(
            f"Component {component!r} was not found for {stream_path!r}. Available components: {available}"
        )

    if len(matches) == 1:
        return matches[0]

    scalar_matches = [column for column in matches if getattr(column, "component_type", None) == "Scalar"]
    if len(scalar_matches) == 1:
        return scalar_matches[0]

    available = tuple(column.component for column in matches)
    raise ValueError(
        f"Multiple components match {stream_path!r}; choose one explicitly from {available}."
    )


def _component_matches(column: Any, requested: str) -> bool:
    return requested in {column.component, column.name, f"{column.entity_path}:{column.component}"}


def _column_to_numpy(column: Any) -> np.ndarray:
    if hasattr(column, "combine_chunks"):
        column = column.combine_chunks()

    to_numpy = getattr(column, "to_numpy", None)
    if callable(to_numpy):
        try:
            return np.asarray(to_numpy(zero_copy_only=False))
        except TypeError:
            return np.asarray(to_numpy())
        except (NotImplementedError, ValueError):
            pass

    to_pylist = getattr(column, "to_pylist", None)
    if callable(to_pylist):
        return np.asarray(to_pylist())

    tolist = getattr(column, "tolist", None)
    if callable(tolist):
        return np.asarray(tolist())

    return np.asarray(column)
