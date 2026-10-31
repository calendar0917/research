from __future__ import annotations

import pytest

import c2_common as C


@pytest.mark.skipif(not C.PROTOCOL_FREEZE.exists(), reason="protocol has not been frozen yet")
def test_protocol_freeze_is_locked_and_matches_code_and_config():
    frozen = C.verify_frozen(require_clean=False)
    assert frozen["status"] == "FROZEN_BEFORE_TRAINING"
    assert frozen["test_access"] == "forbidden"
    assert frozen["width_selected"] == 45
    assert frozen["width_tightly_size_matched"] is True
    assert frozen["cpu_gates_status"] == "PASS"
