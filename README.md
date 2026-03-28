# actuator_analysis

## Table of contents

- [Strategy](#strategy)
- [Dependency Installation](#dependency-installation)
- [Run](#run)
- [Test](#test)

## Strategy

I used times series and Fourier analysis to simplify the problem as much as possible.

Python was chosen because it's great for quick data analysis. I tried to follow good Python practices such as `venv` although I usually work with C++.

There is a good collection of unit tests and CI, as well.

## Setup

Tested on Ubuntu 24

`sudo apt install python3.12-venv`

## Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python3 scripts/run_analysis.py
```

## Test

Install the package in editable mode with dev dependencies (includes pytest), then run the test suite:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
