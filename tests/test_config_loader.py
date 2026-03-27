"""Tests for config loading."""

import os
from pathlib import Path

import pytest

from actuator_analysis.config_loader import (
    config_dir,
    data_path,
    expand_path,
    load_yaml,
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
