"""Shared pytest fixtures for the LCRC Lipophilicity screen CPU gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import screen_common as C  # noqa: E402
import screen_data as D  # noqa: E402
import screen_models as M  # noqa: E402


@pytest.fixture(scope="session")
def spec():
    return C.load_frozen_spec()


@pytest.fixture(scope="session")
def norm():
    return C.load_frozen_norm()


@pytest.fixture(scope="session")
def split():
    return C.load_frozen_split()


@pytest.fixture(scope="session")
def caches(spec):
    return D.load_caches(spec)


@pytest.fixture(scope="session")
def fit_batch(caches):
    rows = list(range(8))
    return D.collate(caches["fit"]["cache"], rows), rows


@pytest.fixture(scope="session")
def model_a(spec, norm):
    model, info = M.build_model("A", spec, norm, 0, "cpu")
    model.eval()
    return model, info


@pytest.fixture(scope="session")
def model_b(spec, norm):
    model, info = M.build_model("B", spec, norm, 0, "cpu")
    model.eval()
    return model, info


@pytest.fixture(scope="session")
def model_c(spec, norm):
    model, info = M.build_model("C", spec, norm, 0, "cpu")
    model.eval()
    return model, info


@pytest.fixture(autouse=True)
def _restore_guard():
    import access_guard

    yield
    access_guard.disarm()


@pytest.fixture(autouse=True)
def _torch_threads():
    torch.set_num_threads(1)
    yield
