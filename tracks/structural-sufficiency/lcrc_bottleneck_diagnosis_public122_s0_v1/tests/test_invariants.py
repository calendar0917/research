"""Fast, CPU-only invariants for the bottleneck-diagnosis round.

These are plumbing/semantics checks, not science results.  They deliberately stay
small (no multi-thousand-update training) so they run in a few seconds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import lb_common as C  # noqa: E402
import lb_data as D  # noqa: E402
import lb_selection as S  # noqa: E402


# ---------------------------------------------------------------------------
# update-budget arithmetic
# ---------------------------------------------------------------------------


def test_u0_matches_frozen_constants():
    assert C.u0("C1_R1", "Lipophilicity") == 9720
    assert C.u0("DMPNN_D2F", "Lipophilicity") == 5300
    assert C.u0("C1_R1", "BACE") == 3600
    assert C.u0("DMPNN_D2F", "BACE") == 1900
    assert C.dmpnn_warmup_steps("Lipophilicity") == 106
    assert C.dmpnn_warmup_steps("BACE") == 38


def test_grids_have_50_points_per_phase():
    for model, ds in (("C1_R1", "Lipophilicity"), ("DMPNN_D2F", "BACE")):
        g = C.grid_updates(model, ds)
        assert len(g["E1"]) == 50
        assert g["E2"] == [] if ds == "BACE" else True
    g = C.grid_updates("C1_R1", "Lipophilicity")
    assert len(g["E2"]) == 50
    assert g["E1"][-1] == C.u0("C1_R1", "Lipophilicity")
    assert all(u > C.u0("C1_R1", "Lipophilicity") for u in g["E2"])


# ---------------------------------------------------------------------------
# nested subsets: label-blind, nested, full == official Train order
# ---------------------------------------------------------------------------


def test_nested_subsets_are_nested_and_hash_ordered():
    for ds, fracs in (("Lipophilicity", ("25", "50", "100")), ("BACE", ("50", "100"))):
        c = D.check_nested_consistency(ds, fracs)
        assert c["nested"] is True
        rows = D.split_rows(ds)["train"]
        order = sorted(range(len(rows)), key=lambda i: (C.subset_key(ds, rows[i]), rows[i]))
        pos = D.nested_subset_positions(ds, "100")["positions"]
        assert pos == sorted(order)
        assert D.nested_subset_positions(ds, "100")["global_rows"] == rows


def test_subset_is_label_blind():
    """Perturbing Train labels outside the demo subset can never change subset membership."""
    ds = "Lipophilicity"
    before = D.nested_subset_positions(ds, "25")["positions"]
    # subset membership is a pure function of (dataset, global row id)
    keys = [C.subset_key(ds, r) for r in D.split_rows(ds)["train"]]
    order = sorted(range(len(keys)), key=lambda i: (keys[i], i))
    assert before == sorted(order[:840])


# ---------------------------------------------------------------------------
# shared norms: label isolation
# ---------------------------------------------------------------------------


def test_target_norm_uses_only_smallest_fit_subset():
    ds = "Lipophilicity"
    tn = D.shared_target_norm(ds)["target_norm"]
    assert tn["n"] == C.FIT_N[ds]["25"]
    tgt = D.targets(ds, include_test=False)
    pos = D.nested_subset_positions(ds, "25")["positions"]
    y = np.asarray(tgt["y"]["train"], np.float64)[np.asarray(pos, np.int64)]
    assert math.isclose(tn["mean"], float(np.mean(y)), rel_tol=1e-6)
    assert math.isclose(tn["scale"], float(np.std(y, ddof=0)), rel_tol=1e-6)


def test_classification_has_no_target_norm():
    assert D.shared_target_norm("BACE")["target_norm"] is None


# ---------------------------------------------------------------------------
# selection rules
# ---------------------------------------------------------------------------


def _rec(u, rmse, mae, test_rmse=None):
    return {"update": u, "valid": {"rmse": rmse, "mae": mae},
            "train_fit": {"rmse": rmse}, "test": {"rmse": test_rmse if test_rmse is not None else rmse}}


def test_rsel_picks_min_valid_and_ties_keep_earlier():
    recs = [_rec(10, 0.5, 0.4), _rec(20, 0.5, 0.3), _rec(30, 0.6, 0.2)]
    assert S.select_state("regression", C.R_SEL, recs)["update"] == 10
    assert S.select_state("regression", C.A_SEL, recs)["update"] == 30


def test_wsel_window_end_and_46_windows():
    recs = [_rec(u, 1.0 - 0.01 * u, 0.5) for u in range(1, 51)]
    sel = S.select_state("regression", C.W_SEL, recs)
    assert sel["window"] == [46, 47, 48, 49, 50]
    assert sel["update"] == 50


def test_oracle_gap_sign_and_definition():
    recs = [_rec(1, 0.40, 0.3, test_rmse=0.60), _rec(2, 0.45, 0.3, test_rmse=0.55)]
    g = S.observed_oracle_gap("regression", recs)
    assert math.isclose(g["observed_oracle_gap"], 0.60 - 0.55)
    assert g["oracle_update"] == 2


def test_classification_selection_is_max_auc():
    recs = [{"update": 1, "valid": {"roc_auc": 0.70, "bce": 0.5}, "test": {"roc_auc": 0.7},
             "train_fit": {"roc_auc": 0.9}},
            {"update": 2, "valid": {"roc_auc": 0.80, "bce": 0.5}, "test": {"roc_auc": 0.6},
             "train_fit": {"roc_auc": 0.9}}]
    assert S.select_state("binary", C.R_SEL, recs)["update"] == 2


# ---------------------------------------------------------------------------
# probe invariants
# ---------------------------------------------------------------------------


def test_probe_new_param_count():
    import lb_probe as P
    import lb_train as T
    spec = D.frozen_spec()
    snorm = D.structure_norm("Lipophilicity")
    import lcpr_model as PM
    m = PM.build_model(spec, snorm, "c1", variant="C1")
    f_state = {k: v for k, v in m.state_dict().items() if k.startswith("f.")}
    rho_state = {k: v for k, v in m.state_dict().items() if k.startswith("rho.")}
    post = P.ProbeHead("P_POST", f_state, rho_state)
    assert post.n_new_params() == C.PROBE_NEW_PARAMS
    pre = P.ProbeHead("P_PRE", f_state, rho_state)
    assert pre.n_new_params() == C.PROBE_NEW_PARAMS
    # private seed -> identical psi/delta initial weights in POST and PRE
    for (k, a), (_, b) in zip(post.psi.state_dict().items(), pre.psi.state_dict().items()):
        assert np.array_equal(a.detach().numpy(), b.detach().numpy())


def test_probe_step0_equivalence():
    import torch
    import lb_probe as P
    import lcpr_model as PM
    spec = D.frozen_spec()
    snorm = D.structure_norm("Lipophilicity")
    m = PM.build_model(spec, snorm, "c1", variant="C1")
    f_state = {k: v for k, v in m.state_dict().items() if k.startswith("f.")}
    rho_state = {k: v for k, v in m.state_dict().items() if k.startswith("rho.")}
    h = torch.randn(37, 64)
    ag = torch.from_numpy(np.repeat(np.arange(5), [8, 7, 8, 7, 7]).astype(np.int64))
    zp = torch.randn(5, 171)
    for rule in ("P_POST", "P_PRE"):
        head = P.ProbeHead(rule, f_state, rho_state)
        d = (head.forward_frozen(h, ag, zp, 5) - head.warm_only(h, ag, zp, 5)).abs().max()
        assert float(d) < 1e-6, float(d)


def test_batch_plan_prefix_is_recoverable():
    import lb_train as T
    a = T.BatchPlan(37, 8)
    prefix = [a.next_batch() for _ in range(23)]
    st = a.state()
    b = T.BatchPlan(37, 8)
    b.load_state(st)
    suffix_a = [a.next_batch() for _ in range(9)]
    suffix_b = [b.next_batch() for _ in range(9)]
    assert suffix_a == suffix_b
    # batch sizes match the frozen trailing-partial rule
    assert [len(x) for x in prefix][:8] == [8, 8, 8, 8, 5, 8, 8, 8]


# ---------------------------------------------------------------------------
# RNG-safe evaluation
# ---------------------------------------------------------------------------


def test_eval_does_not_advance_rng():
    import torch
    import lb_train as T
    spec = D.frozen_spec()
    snorm = D.structure_norm("Lipophilicity")
    import lcpr_model as PM
    m = PM.build_model(spec, snorm, "c1", variant="C1")
    tgt = D.targets("Lipophilicity", include_test=False)
    cache = D.load_cache("Lipophilicity", "valid", tgt["rows"]["valid"])
    runner = T.C1Runner(m, torch.device("cpu"), {"valid": cache})
    m.eval()
    before = torch.get_rng_state().clone()
    p1 = runner.predict("valid", list(range(16)))
    mid = torch.get_rng_state()
    p2 = runner.predict("valid", list(range(16)))
    after = torch.get_rng_state()
    assert torch.equal(before, mid) and torch.equal(mid, after)
    assert np.allclose(p1, p2)
    assert p1.shape == (16,)
