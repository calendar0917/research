from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import common as C  # noqa: E402
import data as D  # noqa: E402
import model as M  # noqa: E402


@pytest.fixture(scope="session")
def spec():
    return C.load_frozen_spec()


@pytest.fixture(scope="session")
def norm():
    return C.load_frozen_norm()


@pytest.fixture(scope="session")
def caches(spec):
    return D.load_caches(spec)


@pytest.fixture(scope="session")
def batch(caches):
    return D.collate(caches["fit"]["cache"], list(range(8)))


@pytest.fixture(scope="session")
def model_r(spec, norm):
    m, info = M.build_model("R", spec, norm, 0, "cpu")
    m.eval()
    return m, info


@pytest.fixture(scope="session")
def model_n(spec, norm):
    m, info = M.build_model("N", spec, norm, 0, "cpu")
    m.eval()
    return m, info


@pytest.fixture(autouse=True)
def _restore_guard():
    import access_guard
    yield
    access_guard.disarm()


@pytest.fixture(autouse=True)
def _torch_threads():
    torch.set_num_threads(1)
    yield
