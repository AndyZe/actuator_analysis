"""Load YAML configuration from the project config directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


def results_path(config_name: str = "defaults") -> Path:
    """Resolved ``results_path`` from the given config file (default: ``defaults``)."""
    cfg = load_yaml(config_name)
    try:
        raw = cfg["results_path"]
    except KeyError as e:
        raise KeyError(f"'results_path' missing in config {config_name!r}") from e
    if not isinstance(raw, str):
        raise TypeError(f"'results_path' must be a string, got {type(raw).__name__}")
    return expand_path(raw)


def parse_instant(s: str) -> datetime:
    """Parse ``YYYY-MM-DD HH:MM:SS`` with optional fractional seconds on the last segment."""
    text = s.strip()
    if not text:
        raise ValueError("instant string is empty")
    # First space between date and time -> ``T`` so ``fromisoformat`` works on Python 3.10.
    norm = text.replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(norm)
    except ValueError as e:
        raise ValueError(f"not a valid YYYY-MM-DD HH:MM:SS instant: {s!r}") from e


@dataclass(frozen=True)
class KeyTimePoints:
    """Key recording and trigger instants from config (all timezone-naive)."""

    recording_start: datetime
    recording_end: datetime
    trigger_start: datetime
    trigger_effects_done: datetime


_TIME_CONFIG_KEYS = (
    "recording_start_time",
    "recording_end_time",
    "trigger_start_time",
    "trigger_effects_done",
)


def key_time_points(config_name: str = "defaults") -> KeyTimePoints:
    """Load ``recording_*`` and ``trigger_*`` instants from the given config file."""
    cfg = load_yaml(config_name)
    values: list[datetime] = []
    for k in _TIME_CONFIG_KEYS:
        try:
            raw = cfg[k]
        except KeyError as e:
            raise KeyError(f"{k!r} missing in config {config_name!r}") from e
        if not isinstance(raw, str):
            raise TypeError(f"{k!r} must be a string, got {type(raw).__name__}")
        values.append(parse_instant(raw))
    return KeyTimePoints(
        recording_start=values[0],
        recording_end=values[1],
        trigger_start=values[2],
        trigger_effects_done=values[3],
    )


def format_recording_offset_timestamp(
    offset_s: float,
    *,
    config_name: str = "defaults",
) -> str:
    """Format seconds from ``recording_start`` as UTC ISO text."""
    recording_start = key_time_points(config_name=config_name).recording_start
    timestamp = (recording_start + timedelta(seconds=offset_s)).replace(tzinfo=timezone.utc)
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
