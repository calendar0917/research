"""Test bootstrap: put this track's ``code/`` directory on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TRACK = Path(__file__).resolve().parents[1]
CODE = TRACK / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import tsb_features as F  # noqa: E402


CACHE_AVAILABLE = F.MANIFEST_PATH.is_file()
