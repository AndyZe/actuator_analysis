"""Tests for config loading."""

import os
from datetime import datetime
from pathlib import Path

import pytest

from actuator_analysis.config_loader import (
    KeyTimePoints,
    config_dir,
    data_path,
    expand_path,
    key_time_points,
    load_yaml,
    parse_instant,
    project_root,
)


def test_project_root_is_repo() -> None:
    root = project_root()
    assert (root / "pyproject.toml").is_file()


def test_config_dir_points_under_repo() -> None:
    d = config_dir()
    assert d == project_root() / "config"
    assert (d / "defaults.yaml").is_file()


def test_load_defaults() -> None:
    cfg = load_yaml("defaults")
    assert "data_path" in cfg
    assert isinstance(cfg["data_path"], str)
    assert "motor.rrd" in cfg["data_path"]
    for k in (
        "recording_start_time",
        "recording_end_time",
        "trigger_start_time",
        "trigger_effects_done",
    ):
        assert k in cfg and isinstance(cfg[k], str)


def test_expand_path_substitutes_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "testuser")
    p = expand_path("/home/$USER/Documents/motor.rrd")
    assert p == Path("/home/testuser/Documents/motor.rrd").resolve()


def test_data_path_matches_defaults_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", os.environ.get("USER", "testuser"))
    p = data_path()
    assert p.name == "motor.rrd"
    assert p.parent.name == "Documents"
    assert str(p).startswith("/home/")


def test_parse_instant_integer_and_fractional_seconds() -> None:
    assert parse_instant("2025-11-05 20:51:40") == datetime(2025, 11, 5, 20, 51, 40)
    assert parse_instant("2025-11-05 18:00:48.0") == datetime(2025, 11, 5, 18, 0, 48)
    assert parse_instant("2025-11-05 18:08:15.65") == datetime(
        2025, 11, 5, 18, 8, 15, 650_000
    )
    assert parse_instant(" 2025-11-05 18:08:28.71 ") == datetime(
        2025, 11, 5, 18, 8, 28, 710_000
    )


def test_parse_instant_rejects_empty_and_invalid() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_instant("  ")
    with pytest.raises(ValueError, match="not a valid"):
        parse_instant("not-a-date")


def test_key_time_points_matches_defaults_yaml() -> None:
    expected = KeyTimePoints(
        recording_start=datetime(2025, 11, 5, 18, 0, 48),
        recording_end=datetime(2025, 11, 5, 20, 51, 40),
        trigger_start=datetime(2025, 11, 5, 18, 8, 15, 650_000),
        trigger_effects_done=datetime(2025, 11, 5, 18, 8, 28, 710_000),
    )
    assert key_time_points("defaults") == expected


def test_key_time_points_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "actuator_analysis.config_loader.load_yaml",
        lambda _name: {"data_path": "/x.rrd"},
    )
    with pytest.raises(KeyError, match="recording_start_time"):
        key_time_points("defaults")
