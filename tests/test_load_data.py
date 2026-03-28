"""Tests for load_data helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from actuator_analysis.load_data import extract_stream, load_streams, resolve_recording_path


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


class FakeReader:
    def __init__(self, table: FakeTable) -> None:
        self._table = table

    def read_all(self) -> FakeTable:
        return self._table


class FakeView:
    def __init__(self, table: FakeTable) -> None:
        self._table = table
        self.filtered_column: str | None = None
        self.fill_latest_at_called = False
        self.selected_columns: tuple[str, ...] | None = None

    def filter_is_not_null(self, column: str) -> "FakeView":
        self.filtered_column = column
        return self

    def fill_latest_at(self) -> "FakeView":
        self.fill_latest_at_called = True
        return self

    def select(self, *columns: str) -> FakeReader:
        self.selected_columns = columns
        return FakeReader(self._table)


class FakeRecording:
    def __init__(self, schema: FakeSchema, tables_by_entity: dict[str, FakeTable]) -> None:
        self._schema = schema
        self._tables_by_entity = tables_by_entity
        self.views: list[FakeView] = []

    def schema(self) -> FakeSchema:
        return self._schema

    def view(self, *, index: str, contents: str) -> FakeView:
        del index
        view = FakeView(self._tables_by_entity[contents])
        self.views.append(view)
        return view


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
    recording = FakeRecording(
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

    stream = extract_stream(recording, "/pitch/current")

    np.testing.assert_array_equal(stream.timestamps, np.asarray([1, 2, 3]))
    np.testing.assert_array_equal(stream.values, np.asarray([0.1, 0.2, 0.3]))
    assert stream.timeline == "log_time"
    assert stream.component == "Scalars:scalars"
    assert recording.views[0].filtered_column == "/pitch/current:Scalars:scalars"
    assert recording.views[0].selected_columns == ("log_time", "/pitch/current:Scalars:scalars")


def test_extract_stream_fill_latest_at_skips_not_null_filter() -> None:
    scalar_column = FakeComponentColumn(
        entity_path="/pitch/current",
        component="Scalars:scalars",
        name="/pitch/current:Scalars:scalars",
        component_type="Scalar",
    )
    recording = FakeRecording(
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

    extract_stream(recording, "/pitch/current", fill_latest_at=True)

    assert recording.views[0].fill_latest_at_called is True
    assert recording.views[0].filtered_column is None


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
    recording = FakeRecording(
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
