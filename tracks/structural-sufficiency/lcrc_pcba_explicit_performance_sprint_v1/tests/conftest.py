"""Test import path for the explicit-performance sprint."""

from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
