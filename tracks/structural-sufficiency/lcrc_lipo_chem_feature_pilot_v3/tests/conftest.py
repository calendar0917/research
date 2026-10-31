"""Shared fixtures for the v3 tests.

The session guard re-verifies v1 and v2 before and after the suite.  Tests do
not write outside ``tmp_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import v3_common as C  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _frozen_guard():
    v1_before = C.assert_v1_unchanged()
    v2_before = C.assert_v2_unchanged()
    assert v1_before["ok"], v1_before
    assert v2_before["ok"], v2_before
    yield
    v1_after = C.assert_v1_unchanged()
    v2_after = C.assert_v2_unchanged()
    assert v1_after["ok"], v1_after
    assert v2_after["ok"], v2_after
    assert v1_before["checks"] == v1_after["checks"]
    assert v2_before["checks"] == v2_after["checks"]


@pytest.fixture(autouse=True)
def _torch_threads():
    torch.set_num_threads(1)
    yield


@pytest.fixture(scope="session")
def store():
    import v3_features as F

    return F.build_store()


@pytest.fixture(scope="session")
def fold_extras_12():
    import v3_features as F

    return F.build_fold_extras(C.FOLD, "chem12")


@pytest.fixture(scope="session")
def fold_extras_13():
    import v3_features as F

    return F.build_fold_extras(C.FOLD, "chem13")
