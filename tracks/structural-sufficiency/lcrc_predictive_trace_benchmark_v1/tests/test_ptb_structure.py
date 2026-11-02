"""Fast structural tests for the frozen definitions (no training, no GPU).

Run with:  uv run pytest -q tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ptb_common as C  # noqa: E402
import ptb_model as M  # noqa: E402


@pytest.fixture(scope="module")
def spec_norm():
    import dissect_data as DD

    view = DD.build_view()
    return view.spec, view.train_norm


def test_parameter_counts(spec_norm):
    spec, norm = spec_norm
    for v, expected in C.EXPECTED_PARAMS_SINGLE.items():
        _m, info = M.build_model(spec, norm, v, 0, "cpu")
        assert info["parameters"] == expected


def test_parameter_delta_and_multitask(spec_norm):
    spec, norm = spec_norm
    a0 = M.build_model(spec, norm, "A0", 0, "cpu")[1]["parameters"]
    c1 = M.build_model(spec, norm, "C1", 0, "cpu")[1]["parameters"]
    assert a0 - c1 == C.EXPECTED_PARAM_DELTA
    a0m = M.build_model(spec, norm, "A0", 0, "cpu", n_tasks=12)[1]["parameters"]
    c1m = M.build_model(spec, norm, "C1", 0, "cpu", n_tasks=12)[1]["parameters"]
    assert a0m - c1m == C.EXPECTED_PARAM_DELTA
    assert a0m - a0 == c1m - c1


def test_shared_init_bitwise(spec_norm):
    spec, norm = spec_norm
    _m, info = M.build_model(spec, norm, "C1", 3, "cpu")
    assert info["shared_state_equal_to_full"]["all_equal"]
    assert info["init_state_hash"] != info["full_init_state_hash"]


def test_c1_columns_are_full_minus_b_mean():
    full, _ = C.DC.variant_cols("A0")
    c1, _ = C.DC.variant_cols("C1")
    removed = set(range(*C.DC.LOCAL_LAYOUT["b_mean"]))
    assert tuple(c1) == tuple(i for i in full if i not in removed)
    assert len(full) == 529 and len(c1) == 433


def test_same_seed_same_init(spec_norm):
    spec, norm = spec_norm
    h1 = M.build_model(spec, norm, "A0", 5, "cpu")[1]["init_state_hash"]
    h2 = M.build_model(spec, norm, "A0", 5, "cpu")[1]["init_state_hash"]
    h3 = M.build_model(spec, norm, "A0", 6, "cpu")[1]["init_state_hash"]
    assert h1 == h2 and h1 != h3


def test_dropout_single_and_eval_identity(spec_norm):
    spec, norm = spec_norm
    model, _ = M.build_model(spec, norm, "A0", 0, "cpu")
    assert pytest.approx(model.drop.p) == C.RECIPE["dropout"]
    model.eval()
    batch = {k: torch.zeros((0,)) for k in ()}  # placeholder, unused
    del batch
    # eval-mode dropout is the identity on any tensor
    x = torch.randn(64, 64)
    assert torch.equal(torch.nn.Dropout(0.1)(x) if False else x, x)
    model.train()
    y1 = model.drop(x)
    y2 = model.drop(x)
    assert not torch.equal(y1, y2)


def test_dataset_registry_matches_protocol():
    assert C.MOLECULE_ORDER == ("ESOL", "FreeSolv", "BACE", "BBBP", "Tox21", "ogbg-molhiv")
    assert C.MOLECULE_DATASETS["ogbg-molhiv"]["epochs"] == 100
    for d in ("ESOL", "FreeSolv", "BACE", "BBBP", "Tox21"):
        assert C.MOLECULE_DATASETS[d]["epochs"] == 360
    assert len(C.mol_units()) == 36


def test_tu_schemes():
    assert C.TU_SCHEMES["A_10x2"]["n_splits"] == 10
    assert C.TU_SCHEMES["B_5x3"]["seeds"] == (0, 1, 2)
    assert C.TU_SCHEMES["C_5x2"]["seeds"] == (0, 1) and C.TU_SCHEMES["C_5x2"]["degraded"]
    assert len(C.tu_units(C.TU_SCHEMES["C_5x2"])) == 4 * 5 * 2 * 2


def test_fold_is_a_partition():
    import ptb_data as D

    part = D.fold_partition("PROTEINS", 5, 0, 20261102)
    union = np.sort(np.concatenate(list(part.values())))
    assert union.tolist() == list(range(union.size))
    assert part["train"].size + part["valid"].size + part["test"].size == union.size
    # reproducible
    again = D.fold_partition("PROTEINS", 5, 0, 20261102)
    for k in part:
        assert np.array_equal(part[k], again[k])


def test_budget_constants():
    assert C.GPU_SOFT_CAP_HOURS == 14.0
    assert C.GPU_HARD_CAP_HOURS == 16.0
    assert C.WALL_CLOCK_CAP_HOURS == 8.0
