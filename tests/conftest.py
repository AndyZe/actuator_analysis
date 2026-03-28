"""Pytest configuration: minimal ``datafusion`` stub so unit tests run without the optional dependency."""

from __future__ import annotations

import sys
import types


class _FakeCol:
    def __call__(self, name: str) -> object:
        return _FakeExpr(name)


class _FakeExpr:
    def __init__(self, name: str) -> None:
        self.name = name

    def is_not_null(self) -> _FakeExpr:
        return self


# ``rerun-sdk[datafusion]`` provides ``datafusion`` at runtime; CI or minimal envs may lack it.
# Tests use ``FakeDataFrame.filter`` which ignores the expression; only ``col`` must exist.
if "datafusion" not in sys.modules:
    _mod = types.ModuleType("datafusion")
    _mod.col = _FakeCol()
    sys.modules["datafusion"] = _mod
