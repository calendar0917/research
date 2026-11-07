import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ssl_data as D  # noqa: E402
import ssl_model as M  # noqa: E402


@pytest.fixture(scope="session")
def structure():
    return D.load_structure()


@pytest.fixture(scope="session")
def norm(structure):
    return D.load_train_norm(structure)


@pytest.fixture(scope="session")
def scratch(structure, norm):
    return M.seed0_scratch_state(structure.spec, norm)


@pytest.fixture(scope="session")
def pretext(structure, norm):
    model, audit = M.build_pretext(structure.spec, norm)
    return model, audit


@pytest.fixture(scope="session")
def priors(structure):
    return D.class_priors(structure)


@pytest.fixture()
def batch(structure):
    rows = structure.fit_rows[:8]
    return D.build_pretext_batch(structure, rows, visit_index=0)
