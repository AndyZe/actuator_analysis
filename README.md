# actuator_analysis

## Dependency Installation

Tested on Ubuntu 24

`sudo apt install python3.12-venv`

## Run

From the repository root:

```bash
python3 scripts/run_analysis.py
```

The script adds `src/` to `sys.path`, so you do not need to install the package first.

## Test

Install the package in editable mode with dev dependencies (includes pytest), then run the test suite:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
