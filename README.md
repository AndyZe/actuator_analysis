# actuator_analysis

## Table of contents

- [Setup](#setup)
- [Run](#run)
- [Test](#test)

## Setup

Tested on Ubuntu 24

`sudo apt install python3.12-venv`

## Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python3 scripts/WHATEVER_SCRIPT.py
```

## Test

Install the package in editable mode with dev dependencies (includes pytest), then run the test suite:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
