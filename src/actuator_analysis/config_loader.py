"""Load YAML configuration from the project config directory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repository root when developing from ``src/actuator_analysis`` layout."""
    return Path(__file__).resolve().parent.parent.parent


def config_dir() -> Path:
    """Directory containing YAML config files (override with ``ACTUATOR_CONFIG_DIR``)."""
    override = os.environ.get("ACTUATOR_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "config"


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML file from the config directory by basename (e.g. ``defaults`` or ``defaults.yaml``)."""
    base = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    path = config_dir() / base
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Root of {path} must be a mapping, got {type(data).__name__}")
    return data


def expand_path(path: str) -> Path:
    """Expand ``~`` and environment variables (e.g. ``$USER``) in *path* and normalize to an absolute path."""
    expanded = os.path.expandvars(os.path.expanduser(path))
    return Path(expanded).resolve()


def data_path(config_name: str = "defaults") -> Path:
    """Resolved ``data_path`` from the given config file (default: ``defaults``)."""
    cfg = load_yaml(config_name)
    try:
        raw = cfg["data_path"]
    except KeyError as e:
        raise KeyError(f"'data_path' missing in config {config_name!r}") from e
    if not isinstance(raw, str):
        raise TypeError(f"'data_path' must be a string, got {type(raw).__name__}")
    return expand_path(raw)
