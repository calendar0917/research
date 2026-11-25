"""Fast structural tests for the joint-support round (no GPU, no formal run)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import sj_common as C  # noqa: E402
import sj_gates as G  # noqa: E402
import sj_model as M  # noqa: E402
import sj_support as S  # noqa: E402


@pytest.fixture(scope="module")
def lipo():
    view, train_cache, valid_cache, test_cache, spec = S.task_caches("lipo")
    support = S.build(
        "lipo", train_cache, valid_cache=valid_cache, test_cache=test_cache, spec=spec
    )
    return view, support, spec


def test_support_identities(lipo):
    _view, support, _spec = lipo
    assert support.rank == support.n_columns - support.n_components
    assert np.array_equal(support.P, support.P.T)
    assert np.max(np.abs(support.P @ support.P - support.P)) < 1e-10
    assert np.max(np.abs(support.U.T @ support.U - np.eye(support.rank))) < 1e-10
    assert np.max(np.abs(support.X @ support.pinvX @ support.X - support.X)) < 1e-10
    assert np.all(support.omega_count >= support.omega_docfreq)
    assert np.all(np.diff(support.omega_code) > 0)
    assert support.joint_dim > 0


def test_pair_mode_table(lipo):
    _view, support, _spec = lipo
    seen = support.index_table >= 0
    assert np.all(support.pair_mode[seen] == S.MODE_OMEGA)
    assert np.all(support.pair_mode[~seen] >= S.MODE_MARGINAL_LS)
    n_bond = support.n_bond_types
    codes = np.arange(support.n_path_codes * support.n_nbr_codes)
    legal = (codes // support.n_nbr_codes) % n_bond == (codes % support.n_nbr_codes) % n_bond
    assert not np.any(support.pair_mode[~legal] == S.MODE_MARGINAL_LS)


def test_rectangles_count(lipo):
    _view, support, _spec = lipo
    rect = G.rectangles(support)
    assert rect.shape[0] == support.n_rectangles
    assert np.all(rect >= 0) and np.all(rect < support.n_omega)
    # every quadruple is four distinct Omega rows forming a real rectangle
    for row in rect[:50]:
        assert len(set(row.tolist())) == 4


def test_initialisation_equivalence(lipo):
    view, support, _spec = lipo
    device = torch.device("cpu")
    models = {arm: M.make_arm("lipo", view, support, arm, C.SEED, device) for arm in C.ARMS}
    a, b = models[C.ARM_A][0], models[C.ARM_B][0]
    sa, sb = a.state_dict(), b.state_dict()
    common = sorted((set(sa) & set(sb)) - {"local.path_u.weight", "local.path_v.weight",
                                           "local.joint_proj.weight", "local.joint_proj.bias"})
    for key in common:
        assert torch.equal(sa[key], sb[key]), key
    cache = G.subset_cache(view.train_cache, list(range(24)), view.spec)
    batch = C.PREP_DATA.collate(cache, list(range(24)))
    codes = batch["path_cat"].to(torch.long) * support.n_nbr_codes + batch["path_end63"].to(torch.long)
    with torch.no_grad():
        a.eval(); b.eval()
        ja = a.local.joint_proj(a.local.path_u(batch["path_cat"]) * a.local.path_v(batch["path_end63"]))
        jb = b.local.code_values(codes)
        ya = a(batch)
        yb = b(batch)
    assert float((ja - jb).abs().max()) < 1e-5
    assert float((ya - yb).abs().max()) < 1e-5


def test_additive_parameterisation(lipo):
    view, support, _spec = lipo
    device = torch.device("cpu")
    model_c, _ = M.make_arm("lipo", view, support, C.ARM_C, C.SEED, device)
    table = M.extract_table(model_c, support)
    resid = table - support.P @ table
    assert float(np.max(np.abs(resid))) < 1e-5
    theta = model_c.local.Theta.detach().double().numpy()
    assert np.max(np.abs(support.U @ theta - table)) < 1e-5


def test_config_is_frozen():
    import sj_config

    assert sj_config.main([]) == 0
