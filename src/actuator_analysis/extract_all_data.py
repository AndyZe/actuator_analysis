"""Load motor axis streams (pitch/yaw, firing splits) from the configured recording."""

from __future__ import annotations

from dataclasses import dataclass

from actuator_analysis.config_loader import key_time_points
from actuator_analysis.load_data import (
    MotorStreamBundle,
    PitchData,
    YawData,
    available_streams,
    load_motor_axis_data,
    load_recording,
)

__all__ = [
    "MotorStreamsLoadResult",
    "MotorStreamBundle",
    "PitchData",
    "YawData",
    "load_all_streams",
    "load_motor_axis_data",
    "load_recording",
]


@dataclass(frozen=True)
class MotorStreamsLoadResult:
    """Pitch/yaw motor bundle plus schema stream paths from one recording open."""

    bundle: MotorStreamBundle
    available_streams: tuple[str, ...]


def load_all_streams(*, config_name: str = "defaults") -> MotorStreamsLoadResult:
    """Load the recording, list entity paths, extract motor axis data, then close it."""
    recording = load_recording(config_name=config_name)
    kp = key_time_points(config_name=config_name)
    try:
        streams = available_streams(recording)
        bundle = load_motor_axis_data(recording, kp)
        return MotorStreamsLoadResult(bundle=bundle, available_streams=streams)
    finally:
        recording.close()
