"""Focused CPU smoke tests for the second-hop composition pilot.

These need the frozen LCRC-v1 cache on disk (``data/`` + track cache); they are
skipped, not failed, when the cache is absent.  They are not part of the
fresh-clone CI allowlist.
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

import sh_common as S  # noqa: E402
import sh_data as D  # noqa: E402
import sh_models as M  # noqa: E402

HAVE_DATA = (S.D.cache_path("valid")).exists()


@pytest.fixture(scope="module")
def valid():
    if not HAVE_DATA:
        pytest.skip("frozen LCRC valid cache not present")
    cache = S.D.load_cache("valid")
    sup = D.load_supplement("valid")
    return cache, sup


def test_path_enumeration_matches_frozen_cache():
    if not HAVE_DATA:
        pytest.skip("frozen LCRC valid cache not present")
    res = D.build_supplement("valid")
    xc = res["crosscheck"]
    assert xc["passed"], xc
    assert xc["n_paths"] == 69040
    assert xc["counts"]["per_atom_path_count_mismatch"] == 0
    assert xc["counts"]["e2_endpoint_is_path_end"] == 0
    assert all(c["equal"] for c in xc["codes"].values())


def test_supplement_sha_is_the_frozen_one():
    if not HAVE_DATA:
        pytest.skip("frozen LCRC valid cache not present")
    frozen = S.read_json_or_none(S.ANALYSIS / "supplement.json")
    if not frozen:
        pytest.skip("analysis/supplement.json not generated yet")
    sup = D.load_supplement("valid")
    assert S.supplement_content_hash(sup) == frozen["valid"]["content_sha256"]


def test_test_split_is_refused():
    with pytest.raises(ValueError):
        D.build_supplement("test", write=False)


def test_real_null_identical_init_and_zero_new_columns():
    norm = S.D.load_norm()
    real, info_r = M.build_model("Real", norm, 0, "cpu", dict(M.DEFAULT_CONFIG))
    null, info_n = M.build_model("Null", norm, 0, "cpu", dict(M.DEFAULT_CONFIG))
    assert info_r["initial_state_sha256"] == info_n["initial_state_sha256"]
    assert info_r["parameters"] == S.PARAMS_NEW
    W = real.local.mlp.lins[0].weight
    assert torch.count_nonzero(W[:, S.BASE_X_DIM:]).item() == 0
    assert torch.equal(W, null.local.mlp.lins[0].weight)
    assert not info_r["projection_trainable"]


def test_step0_equals_historical_A(valid):
    import lc_models as OLD

    cache, sup = valid
    norm = S.D.load_norm()
    old, old_info = OLD.build_model("A", norm, 0, "cpu", dict(OLD.DEFAULT_CONFIG))
    real, _ = M.build_model("Real", norm, 0, "cpu", dict(M.DEFAULT_CONFIG))
    null, _ = M.build_model("Null", norm, 0, "cpu", dict(M.DEFAULT_CONFIG))
    batch = D.collate(cache, sup, np.arange(8))
    with torch.no_grad():
        y_old = old(batch)
        assert torch.equal(y_old, real(batch))
        assert torch.equal(y_old, null(batch))
    assert old_info["initial_state_sha256"] == S.HISTORICAL[0]["initial_state_sha256"]


def test_pairing_identity_and_counterexample():
    mod = M.SecondHopContent(width=4, proj_dim=2, seed=S.PROJ_SEED)
    U = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b2 = U @ torch.linalg.pinv(mod.P_b)
    a_u = V @ torch.linalg.pinv(mod.P_a)
    a = torch.zeros(3, 4)
    a[1], a[2] = a_u[0], a_u[1]
    b = torch.zeros(2, 4)
    b[0], b[1] = b2[0], b2[1]
    common = {"path_atom": torch.tensor([0, 0]), "path_count": torch.tensor([2, 0, 0])}
    batch_a = dict(common, path_end=torch.tensor([1, 2]), path_e2_pos=torch.tensor([0, 1]))
    batch_b = dict(common, path_end=torch.tensor([1, 2]), path_e2_pos=torch.tensor([1, 0]))
    with torch.no_grad():
        Ba, Cra, Cna = mod(batch_a, a, b)
        Bb, Crb, Cnb = mod(batch_b, a, b)
    # identical marginals and path count
    assert torch.allclose(Ba, Bb, atol=1e-6)
    assert torch.allclose(Cna, Cnb, atol=1e-6)
    # only the true pairing differs
    assert (Cra - Crb).abs().max() > 1e-3
    # n_v <= 1 identity
    one = dict(path_atom=torch.tensor([0]), path_count=torch.tensor([1, 0, 0]),
               path_end=torch.tensor([1]), path_e2_pos=torch.tensor([0]))
    with torch.no_grad():
        B1, Cr1, Cn1 = mod(one, a, b)
    assert torch.equal(Cr1, Cn1)
    assert torch.equal(B1[0], b[0])


def test_batch_invariance_with_nonzero_new_weights(valid):
    cache, sup = valid
    norm = S.D.load_norm()
    model, _ = M.build_model("Real", norm, 0, "cpu", dict(M.DEFAULT_CONFIG))
    with torch.no_grad():
        W = model.local.mlp.lins[0].weight
        W[:, S.BASE_X_DIM:] = torch.randn_like(W[:, S.BASE_X_DIM:]) * 0.02
    model.eval()
    rows = np.arange(8)
    with torch.no_grad():
        b = D.collate(cache, sup, rows)
        y_full = model(b, pair_mode="null")
        y_single = torch.cat([model(D.collate(cache, sup, [int(r)]), pair_mode="null")
                              for r in rows])
    assert (y_full - y_single).abs().max().item() <= S.INVARIANCE_TOL
