"""Shared pytest fixtures for the contextual-contribution round."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXP_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = EXP_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import x1_common as C  # noqa: E402
import x1_data as XD  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return C.load_config()


@pytest.fixture(scope="session")
def frozen_data():
    split, labels = XD.load_frozen_split_and_labels()
    views = XD.build_lcrc_views(split, labels)
    return split, labels, views


@pytest.fixture(scope="session")
def train_cache(frozen_data):
    return frozen_data[2]["train_cache"]


@pytest.fixture(scope="session")
def valid_cache(frozen_data):
    return frozen_data[2]["valid_cache"]


@pytest.fixture()
def models(frozen_data):
    import x1_models as M

    _split, _labels, views = frozen_data
    spec = views["spec"]
    base, base_info = M.build_base(spec, views["train_norm"], C.TRAIN_SEED)
    context, base_ref, context_info = M.build_context(spec, views["train_norm"], C.TRAIN_SEED)
    return {
        "spec": spec,
        "norm": views["train_norm"],
        "base": base,
        "base_ref": base_ref,
        "context": context,
        "base_info": base_info,
        "context_info": context_info,
    }
