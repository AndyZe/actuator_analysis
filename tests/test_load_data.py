"""Tests for load_data helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from actuator_analysis.config_loader import KeyTimePoints
from actuator_analysis.load_data import (
    DEFAULT_MOTOR_STREAM_PATHS,
    LoadedRecording,
    extract_stream,
    load_motor_axis_data,
    load_streams,
    resolve_recording_path,
)


class FakeIndexColumn:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeComponentColumn:
    def __init__(
        self,
        *,
        entity_path: str,
        component: str,
        name: str,
        component_type: str | None = None,
        is_static: bool = False,
    ) -> None:
        self.entity_path = entity_path
        self.component = component
        self.name = name
        self.component_type = component_type
        self.is_static = is_static


class FakeSchema:
    def __init__(self, index_names: list[str], component_columns: list[FakeComponentColumn]) -> None:
        self._index_columns = [FakeIndexColumn(name) for name in index_names]
        self._component_columns = component_columns

    def index_columns(self) -> list[FakeIndexColumn]:
        return self._index_columns

    def component_columns(self) -> list[FakeComponentColumn]:
        return self._component_columns


class FakeColumn:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def combine_chunks(self) -> "FakeColumn":
        return self

    def to_numpy(self, zero_copy_only: bool = False) -> np.ndarray:
        del zero_copy_only
        return np.asarray(self._values)


class FakeTable:
    def __init__(self, columns: dict[str, list[object]]) -> None:
        self._columns = {name: FakeColumn(values) for name, values in columns.items()}

    def column(self, name: str) -> FakeColumn:
        return self._columns[name]


class FakeDataFrame:
    """Mimics DataFusion DataFrame used by ``_extract_stream_from_dataset``."""

    def __init__(self, table: FakeTable, index: str, fill_latest_at: bool) -> None:
        self._table = table
        self.index = index
        self.fill_latest_at = fill_latest_at
        self.filter_called = False
        self.selected_columns: tuple[str, ...] | None = None

    def filter(self, _expr: Any) -> FakeDataFrame:
        self.filter_called = True
        return self

    def select(self, *cols: str) -> FakeDataFrame:
        self.selected_columns = cols
        return self

    def to_arrow_table(self) -> FakeTable:
        return self._table


class FakeDatasetContentsView:
    def __init__(self, table: FakeTable, dataset: FakeDataset) -> None:
        self._table = table
        self._dataset = dataset

    def reader(self, index: str, fill_latest_at: bool = False) -> FakeDataFrame:
        df = FakeDataFrame(self._table, index, fill_latest_at)
        self._dataset.last_df = df
        return df


class FakeDataset:
    """Mimics Rerun ``Dataset`` query API (``filter_contents`` / ``reader`` chain)."""

    def __init__(self, schema: FakeSchema, tables_by_entity: dict[str, FakeTable]) -> None:
        self._schema = schema
        self._tables = tables_by_entity
        self.filter_contents_calls = 0
        self.last_df: FakeDataFrame | None = None

    def schema(self) -> FakeSchema:
        return self._schema

    def filter_contents(self, paths: list[str]) -> FakeDatasetContentsView:
        self.filter_contents_calls += 1
        assert len(paths) == 1
        return FakeDatasetContentsView(self._tables[paths[0]], self)


class _FakeServer:
    def shutdown(self) -> None:
        pass


def make_fake_loaded_recording(schema: FakeSchema, tables_by_entity: dict[str, FakeTable]) -> LoadedRecording:
    return LoadedRecording(server=_FakeServer(), dataset=FakeDataset(schema, tables_by_entity))


def test_resolve_recording_path_uses_config(monkeypatch) -> None:
    expected = Path("/tmp/example.rrd")
    monkeypatch.setattr("actuator_analysis.load_data.data_path", lambda config_name: expected)

    assert resolve_recording_path() == expected
    assert resolve_recording_path(config_name="custom") == expected


def test_extract_stream_prefers_scalar_component() -> None:
    scalar_column = FakeComponentColumn(
        entity_path="/pitch/current",
        component="Scalars:scalars",
        name="/pitch/current:Scalars:scalars",
        component_type="Scalar",
    )
    label_column = FakeComponentColumn(
        entity_path="/pitch/current",
        component="TextDocument:markdown",
        name="/pitch/current:TextDocument:markdown",
        component_type="Text",
    )
    dataset = FakeDataset(
        FakeSchema(["log_time"], [scalar_column, label_column]),
        {
            "/pitch/current": FakeTable(
                {
                    "log_time": [1, 2, 3],
                    "/pitch/current:Scalars:scalars": [0.1, 0.2, 0.3],
                }
            )
        },
    )
    recording = LoadedRecording(server=_FakeServer(), dataset=dataset)

    stream = extract_stream(recording, "/pitch/current")

    np.testing.assert_array_equal(stream.timestamps, np.asarray([1, 2, 3]))
    np.testing.assert_array_equal(stream.values, np.asarray([0.1, 0.2, 0.3]))
    assert stream.timeline == "log_time"
    assert stream.component == "Scalars:scalars"
    assert dataset.last_df is not None
    assert dataset.last_df.filter_called is True
    assert dataset.last_df.selected_columns == ("log_time", "/pitch/current:Scalars:scalars")


def test_extract_stream_exclude_time_range_drops_samples_in_closed_interval() -> None:
    scalar_column = FakeComponentColumn(
        entity_path="/pitch/current",
        component="Scalars:scalars",
        name="/pitch/current:Scalars:scalars",
        component_type="Scalar",
    )
    recording = make_fake_loaded_recording(
        FakeSchema(["log_time"], [scalar_column]),
        {
            "/pitch/current": FakeTable(
                {
                    "log_time": [500.0, 1500.0, 1800.0, 2500.0],
                    "/pitch/current:Scalars:scalars": [0.5, 1.5, 1.8, 2.5],
                }
            )
        },
    )
    lo = datetime(1970, 1, 1, 0, 16, 40)
    hi = datetime(1970, 1, 1, 0, 33, 20)

    stream = extract_stream(recording, "/pitch/current", exclude_time_range=(lo, hi))

    np.testing.assert_array_equal(stream.timestamps, np.asarray([500.0, 2500.0]))
    np.testing.assert_array_equal(stream.values, np.asarray([0.5, 2.5]))


def test_extract_stream_include_time_range_keeps_samples_in_closed_interval() -> None:
    scalar_column = FakeComponentColumn(
        entity_path="/pitch/current",
        component="Scalars:scalars",
        name="/pitch/current:Scalars:scalars",
        component_type="Scalar",
    )
    recording = make_fake_loaded_recording(
        FakeSchema(["log_time"], [scalar_column]),
        {
            "/pitch/current": FakeTable(
                {
                    "log_time": [500.0, 1500.0, 1800.0, 2500.0],
                    "/pitch/current:Scalars:scalars": [0.5, 1.5, 1.8, 2.5],
                }
            )
        },
    )
    lo = datetime(1970, 1, 1, 0, 16, 40)
    hi = datetime(1970, 1, 1, 0, 33, 20)

    stream = extract_stream(recording, "/pitch/current", include_time_range=(lo, hi))

    np.testing.assert_array_equal(stream.timestamps, np.asarray([1500.0, 1800.0]))
    np.testing.assert_array_equal(stream.values, np.asarray([1.5, 1.8]))


def test_extract_stream_exclude_and_include_mutually_exclusive() -> None:
    scalar_column = FakeComponentColumn(
        entity_path="/pitch/current",
        component="Scalars:scalars",
        name="/pitch/current:Scalars:scalars",
        component_type="Scalar",
    )
    recording = make_fake_loaded_recording(
        FakeSchema(["log_time"], [scalar_column]),
        {
            "/pitch/current": FakeTable(
                {
                    "log_time": [1.0],
                    "/pitch/current:Scalars:scalars": [1.0],
                }
            )
        },
    )
    lo = datetime(1970, 1, 1)
    hi = datetime(1970, 1, 2)

    with pytest.raises(ValueError, match="at most one"):
        extract_stream(
            recording,
            "/pitch/current",
            exclude_time_range=(lo, hi),
            include_time_range=(lo, hi),
        )


def test_load_motor_axis_data_builds_bundle() -> None:
    paths = DEFAULT_MOTOR_STREAM_PATHS
    columns = [
        FakeComponentColumn(
            entity_path=path,
            component="Scalars:scalars",
            name=f"{path}:Scalars:scalars",
            component_type="Scalar",
        )
        for path in (
            paths.pitch_current,
            paths.pitch_target,
            paths.yaw_current,
            paths.yaw_target,
        )
    ]
    table = {
        "log_time": [500.0, 1500.0, 1800.0, 2500.0],
    }
    tables = {}
    for i, col in enumerate(columns):
        p = col.entity_path
        tables[p] = FakeTable(
            {
                **table,
                col.name: [0.1 + i, 0.2 + i, 0.3 + i, 0.4 + i],
            }
        )

    dataset = FakeDataset(FakeSchema(["log_time"], columns), tables)
    recording = LoadedRecording(server=_FakeServer(), dataset=dataset)
    lo = datetime(1970, 1, 1, 0, 16, 40)
    hi = datetime(1970, 1, 1, 0, 33, 20)
    kp = KeyTimePoints(
        recording_start=lo,
        recording_end=hi,
        trigger_start=lo,
        trigger_effects_done=hi,
    )

    bundle = load_motor_axis_data(recording, kp)

    np.testing.assert_array_equal(bundle.pitch.current_except_firing.timestamps, np.asarray([500.0, 2500.0]))
    np.testing.assert_array_equal(bundle.pitch.current_firing.timestamps, np.asarray([1500.0, 1800.0]))
    np.testing.assert_array_equal(bundle.yaw.target_except_firing.values, np.asarray([3.1, 3.4]))
    assert dataset.filter_contents_calls == 8


def test_extract_stream_fill_latest_at_skips_not_null_filter() -> None:
    scalar_column = FakeComponentColumn(
        entity_path="/pitch/current",
        component="Scalars:scalars",
        name="/pitch/current:Scalars:scalars",
        component_type="Scalar",
    )
    dataset = FakeDataset(
        FakeSchema(["log_time"], [scalar_column]),
        {
            "/pitch/current": FakeTable(
                {
                    "log_time": [10, 11],
                    "/pitch/current:Scalars:scalars": [1.5, 1.5],
                }
            )
        },
    )
    recording = LoadedRecording(server=_FakeServer(), dataset=dataset)

    extract_stream(recording, "/pitch/current", fill_latest_at=True)

    assert dataset.last_df is not None
    assert dataset.last_df.fill_latest_at is True
    assert dataset.last_df.filter_called is False


def test_load_streams_extracts_multiple_paths_from_same_recording() -> None:
    current_column = FakeComponentColumn(
        entity_path="/pitch/current",
        component="Scalars:scalars",
        name="/pitch/current:Scalars:scalars",
        component_type="Scalar",
    )
    voltage_column = FakeComponentColumn(
        entity_path="/pitch/voltage",
        component="Scalars:scalars",
        name="/pitch/voltage:Scalars:scalars",
        component_type="Scalar",
    )
    recording = make_fake_loaded_recording(
        FakeSchema(["timestamp"], [current_column, voltage_column]),
        {
            "/pitch/current": FakeTable(
                {
                    "timestamp": [100, 200],
                    "/pitch/current:Scalars:scalars": [2.0, 2.5],
                }
            ),
            "/pitch/voltage": FakeTable(
                {
                    "timestamp": [100, 200],
                    "/pitch/voltage:Scalars:scalars": [12.0, 12.2],
                }
            ),
        },
    )

    streams = load_streams(["/pitch/current", "/pitch/voltage"], recording=recording)

    assert set(streams) == {"/pitch/current", "/pitch/voltage"}
    np.testing.assert_array_equal(streams["/pitch/current"].values, np.asarray([2.0, 2.5]))
    np.testing.assert_array_equal(streams["/pitch/voltage"].values, np.asarray([12.0, 12.2]))
