"""Forward the ``structure_concept_instance_gate_v1`` unit tests into the

track pytest configuration.  The canonical test file lives next to the round
deliverable (result-directory-local tests, code and artifacts); this module
loads it by path and re-exports its ``test_*`` functions so the repository
pytest configuration (``tracks/structure-contribution/tests``) collects them
as well.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TESTS = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "structure_concept_instance_gate_v1"
    / "tests"
    / "test_gate_v1.py"
)

_spec = importlib.util.spec_from_file_location("_gate_v1_canonical_tests", _TESTS)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

for _name, _value in vars(_mod).items():
    if _name.startswith("test_") and callable(_value):
        globals()[_name] = _value
