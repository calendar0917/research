"""Shared fixtures for the v2 CPU gate suite.

The session-scoped guard hashes every v1 frozen artifact, the v1 manifest and
every file referenced by that manifest before the suite starts and after it
finishes.  A test that writes into the v1 round therefore fails the suite even
if its own assertions pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import v2_common as C  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _v1_immutability_guard():
    before = C.assert_v1_unchanged()
    assert before["ok"], before
    yield
    after = C.assert_v1_unchanged()
    assert after["ok"], after
    assert before["checks"] == after["checks"], "a v2 test changed a v1 frozen artifact"


@pytest.fixture(autouse=True)
def _restore_guard():
    yield
    try:
        C.v1_access_guard.disarm()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _torch_threads():
    torch.set_num_threads(1)
    yield


@pytest.fixture(scope="session")
def spec():
    return C.load_frozen_spec()


@pytest.fixture(scope="session")
def norm():
    return C.load_frozen_norm()


@pytest.fixture(scope="session")
def fit_cache():
    return C.load_fit_cache()["cache"]
