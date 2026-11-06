"""Shared pytest configuration for the continuation-recipe round."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

# Modules that build models or load the frozen PCBA cache/checkpoints.
SLOW_MODULES = {"test_cc_restore.py"}


def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.slow
    for item in items:
        if item.path.name in SLOW_MODULES:
            item.add_marker(marker)
