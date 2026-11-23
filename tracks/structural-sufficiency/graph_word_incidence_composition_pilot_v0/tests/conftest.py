from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import gwic_data as D  # noqa: E402
import gwic_model as M  # noqa: E402
import gwic_structure as S  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    return D.load_bundle()


@pytest.fixture(scope="session")
def dims():
    return D.atom_bond_dims()


@pytest.fixture(scope="session")
def norm(bundle):
    return {
        "atom_scalar_mean": bundle.atom_scalar_mean,
        "atom_scalar_scale": bundle.atom_scalar_scale,
        "graph_mean": bundle.graph_mean,
        "graph_scale": bundle.graph_scale,
    }


@pytest.fixture(scope="session")
def rows(bundle):
    return np.union1d(bundle.fit_rows, bundle.eval_rows)[:12]


@pytest.fixture(scope="session")
def structs(bundle, rows):
    return S.build_all(bundle.raw, rows)


@pytest.fixture(scope="session")
def model_word(dims, norm):
    cfg = {"dims": {"atom": 48, "bond": 24, "sentence": 96, "word": 64, "q": 64,
                    "path_rank": 24, "path_end": 16, "path_close": 8, "nbr": 16,
                    "backbone_hidden": 192, "word_hidden": 160, "score_hidden": 96,
                    "q_hidden": 128, "readout_hidden": 192, "readout_mid": 96,
                    "dropout": 0.1}}
    model, info = M.build_model(dims[0], dims[1], 0, norm, True, cfg)
    return model, info
