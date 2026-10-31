"""The 12 required CPU gates must all pass."""

from __future__ import annotations

import x1_common as C
import x1_gates


def test_all_required_gates_pass():
    payload = x1_gates.run_gates()
    assert payload["ok"], [c for c in payload["checks"] if not c["passed"]]
    assert payload["n_gates"] == 12
    assert payload["n_passed"] == 12
    assert payload["supplementary"]["passed"]
    ids = [c["id"] for c in payload["checks"]]
    assert ids == [spec[0] for spec in x1_gates.GATE_SPECS]
