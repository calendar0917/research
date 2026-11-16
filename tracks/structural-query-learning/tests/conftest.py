"""Test configuration: put the track's ``code/`` directory on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# Local tests must never eat the whole machine: cap the root-truth mask LRU.
import os

os.environ.setdefault("SQ_MASK_CACHE_MB", "96")
