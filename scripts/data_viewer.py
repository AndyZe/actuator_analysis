#!/usr/bin/env python3
"""Load Rerun recording from ``data_path`` in ``config/defaults.yaml`` and view it in the browser."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import rerun as rr

from actuator_analysis.config_loader import data_path


def main() -> None:
    path = data_path()
    recording = rr.recording.load_recording(str(path))

    rr.init(recording.application_id())
    uri = rr.serve_grpc()
    rr.serve_web_viewer(connect_to=uri, open_browser=True)
    print(f"Serving {path} — web viewer should open (gRPC: {uri}).")

    rr.send_recording(recording)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
