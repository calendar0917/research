"""Forward the ``structure_context_support_v1`` unit tests into the track pytest

configuration.  The canonical test files live next to the round deliverable
(result-directory-local tests, code and artifacts); this module loads them by
path and re-exports their ``test_*`` functions so the repository pytest
configuration (``tracks/structure-contribution/tests``) collects them as well.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_RESULTS = Path(__file__).resolve().parents[1] / "results" / "structure_context_support_v1"
_CODE = _RESULTS / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

_TESTS = _RESULTS / "tests" / "test_attributed_canon.py"

_spec = importlib.util.spec_from_file_location("_scsv_test_attributed_canon", _TESTS)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

for _name, _value in vars(_mod).items():
    if _name.startswith("test_") and callable(_value):
        globals()[_name] = _value
