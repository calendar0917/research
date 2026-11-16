"""Test configuration: put this track's ``code/`` directory on ``sys.path``.

The tests are deliberately small (synthetic task, at most a few thousand rows)
so the local 27 GB machine never holds the real pair matrices; the real
215,040-record C matrix is only built on the GPU host.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
