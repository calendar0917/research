"""Regression tests for the ERRATA corrections.

Every fixture here is deliberately tiny and hand-computable; the point is to pin
the *definition* (which rows, which parameters, which lr), not just the output
shape.

Run: uv run pytest tracks/structural-sufficiency/lcrc_performance_route_8h_public122_s0_v1/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import lcpr_analysis as A  # noqa: E402
import lcpr_common as C  # noqa: E402
import lcpr_data as D  # noqa: E402


# --------------------------------------------------------------------------- #
# E1: the drop set is the k LARGEST errors, ties by ascending global row id
# --------------------------------------------------------------------------- #


def test_tail_rank_is_descending_error_with_row_id_tiebreak():
    y = np.array([0.0, 1.0, 10.0])
    pred = np.array([0.0, 3.0, 4.0])
    ae = np.abs(pred - y)                       # [0, 2, 6]
    assert A.tail_rank(y, pred).tolist() == [2, 1, 0]

    #: identical absolute errors: the tie must be broken by the global row id,
    #: never by array position
    y2 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 1.0, 1.0])
    rows2 = np.array([50, 7, 12])
    assert A.tail_rank(y2, p2, rows2).tolist() == [1, 2, 0]


def test_tail_sets_drop_the_largest_errors_and_lower_base_rmse():
    y = np.array([0.0, 1.0, 10.0])
    pred = np.array([0.0, 3.0, 4.0])
    rows = np.array([11, 4, 9])
    sets = A.tail_sets(y, pred, rows, fractions=(1.0 / 3.0,))
    block = sets[f"exclude_top_{1 / 3:g}"]
    assert block["k"] == 1
    assert block["positions"] == [2]
    assert block["global_row_ids"] == [9]
    assert block["abs_errors"] == [6.0]
    #: 6^2 / (36 + 4 + 0) = 0.9 of the base SSE sits in one row
    assert block["sse_share_of_base"] == pytest.approx(0.9)

    keep = np.array([0, 1])
    base_before = float(np.sqrt(np.mean((pred - y) ** 2)))
    base_after = float(np.sqrt(np.mean((pred[keep] - y[keep]) ** 2)))
    assert base_after <= base_before + 1e-12
    assert base_after == pytest.approx(np.sqrt((0.0 + 4.0) / 2))


def test_tail_view_positions_and_global_ids_stay_separate():
    pred = {"test_y": np.array([0.0, 1.0, 10.0]),
            "test_pred": np.array([0.0, 3.0, 4.0]),
            "test_rows": np.array([212, 2087, 3])}
    view = A.tail_view(pred, "test", fractions=(1.0 / 3.0,))
    key = f"exclude_top_{1 / 3:g}"
    #: the top row is *position* 2, whose published row id is 3 - these are not
    #: interchangeable (the old report conflated them)
    assert view["top_row"] == 2
    assert view["top_row_global_id"] == 3
    assert view[key] == [2]
    assert view[f"{key}_ids"] == [3]


def test_tail_comparison_uses_one_control_row_set_for_every_model():
    import lcpr_correction as CORR

    y = np.array([0.0, 1.0, 10.0, 2.0, -1.0])
    rows = np.array([5, 6, 7, 8, 9])
    control = {"test_y": y, "test_pred": y + np.array([0.0, 2.0, 6.0, 0.5, 0.1]),
               "test_rows": rows}
    arm = {"test_y": y, "test_pred": y + np.array([0.0, 1.0, 0.5, 0.5, 0.1]),
           "test_rows": rows}
    rec = CORR.tail_comparison(control, arm, "test", fractions=(0.2, 0.4))
    dropped = {f: set(rec["by_fraction"][f]["drop_row_positions"]) for f in rec["by_fraction"]}
    assert dropped["exclude_top_0.2"] == {2}
    assert dropped["exclude_top_0.4"] == {2, 1}
    #: the kept set is defined by the control only; the arm is scored on it
    keep = np.array([0, 3, 4])
    assert rec["by_fraction"]["exclude_top_0.4"]["control_rmse_same_rows"] == pytest.approx(
        float(np.sqrt(np.mean((control["test_pred"][keep] - y[keep]) ** 2))))
    assert rec["by_fraction"]["exclude_top_0.4"]["arm_rmse_same_rows"] == pytest.approx(
        float(np.sqrt(np.mean((arm["test_pred"][keep] - y[keep]) ** 2))))
    assert rec["full"]["n_rows_improved"] == 2
    assert len(rec["cumulative_delta_sse"]) == 5


# --------------------------------------------------------------------------- #
# E2: molecule-level support
# --------------------------------------------------------------------------- #


def test_mol_key_counts_dedupes_roots_and_counts_molecules():
    #: molecule 0: atoms 0,1 both carry key (3,0) and atom 2 carries (4,1)
    #: molecule 1: atom 3 also carries key (3,0)
    atom_ptr = np.array([0, 3, 4])            # 2 molecules, 4 atoms
    path_ptr = np.array([0, 2, 3, 4, 5])      # per-atom path offsets
    cat = np.array([3, 9, 3, 4, 3])
    end = np.array([0, 0, 0, 1, 0])
    counts = D._mol_key_counts(atom_ptr, path_ptr, cat, end, n_mol=2)
    #: (3,0) appears on two roots of molecule 0 -> counted once there, once in
    #: molecule 1 -> 2 distinct molecules
    assert counts[(3, 0)] == 2
    #: (9,0) is unique to one root of molecule 0
    assert counts[(9, 0)] == 1
    assert counts[(4, 1)] == 1
    #: duplicated roots never inflate the count past the number of molecules
    assert max(counts.values()) <= 2


def test_mol_key_counts_never_exceed_molecules():
    atom_ptr = np.array([0, 2, 4])
    path_ptr = np.array([0, 1, 2, 3, 4])
    cat = np.array([1, 1, 1, 1])
    end = np.array([0, 0, 0, 0])
    counts = D._mol_key_counts(atom_ptr, path_ptr, cat, end, n_mol=2)
    assert counts[(1, 0)] == 2


@pytest.mark.slow
def test_support_v2_matches_definition_on_lipophilicity():
    """The real table must satisfy 1 <= c_mol <= n_Train and c_mol <= c_root."""
    summary = D.build_support_v2("Lipophilicity")
    n_train = int(summary["n_train_molecules"])
    assert 1.0 <= summary["c_min"] <= summary["c_max"] <= n_train
    z_new = np.load(D.support_v2_path("Lipophilicity"), allow_pickle=False)
    z_old = np.load(D.support_path("Lipophilicity"), allow_pickle=False)
    new_c = np.asarray(z_new["counts"], dtype=np.float64)
    old_c = np.asarray(z_old["counts"], dtype=np.float64)
    n_codes = int(C.N_PATH_CODES) * int(C.N_NBR_CODES)
    new_k = np.asarray(z_new["keys"]).reshape(-1, 2)
    old_k = np.asarray(z_old["keys"]).reshape(-1, 2)
    old_flat = old_k[:, 0] * n_codes + old_k[:, 1]
    new_flat = new_k[:, 0] * n_codes + new_k[:, 1]
    pos = np.clip(np.searchsorted(old_flat, new_flat), 0, max(old_c.size - 1, 0))
    ok = old_flat[pos] == new_flat
    assert ok.any()
    assert bool((new_c[ok] <= old_c[pos[ok]] + 1e-6).all())
    #: the legacy root support contradicts its own definition
    assert float(old_c.max()) > n_train
    assert 0.0 < float(summary["kappa"]) <= n_train


# --------------------------------------------------------------------------- #
# E3: shared trunk / private heads
# --------------------------------------------------------------------------- #


def test_shared_and_private_parameter_split():
    import lcpr_correction as CORR
    import lcpr_model as M

    spec = D.frozen_spec()
    import lcpr_train as T

    norm = T._train_structure_norm(D.load_cache("Lipophilicity", "train"), spec)
    model = M.build_model(spec, norm, "a_logp")
    shared, private = CORR._split_params(model)
    shared_names = {n for n, _ in shared}
    private_names = {n for n, _ in private}
    assert shared_names and private_names
    assert shared_names | private_names == {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(n.startswith(("q_aux.", "rho_aux.", "f.", "rho.")) for n in shared_names)
    assert {"q_aux.lins.0.weight", "rho_aux.lins.0.weight"} <= private_names
    assert any(n.startswith("local.") for n in shared_names)


# --------------------------------------------------------------------------- #
# E4: the D-MPNN lr schedule realises 1e-4 -> peak -> 1e-4 only with base=init
# --------------------------------------------------------------------------- #


def test_noam_lambda_endpoints():
    import lcpr_dmpnn as DMN

    warmup, total, init, peak, final = 106, 5300, 1e-4, 3e-4, 1e-4
    f = lambda s: DMN._noam_lambda(s, warmup, total - warmup, init, peak, final)  # noqa: E731
    assert f(0) == pytest.approx(1.0)
    assert init * f(0) == pytest.approx(init)
    #: the protocol peak is hit on the first post-warmup step, not on the last
    #: warm-up step
    assert init * f(warmup) == pytest.approx(peak)
    assert init * f(warmup - 1) == pytest.approx(init + (peak - init) * (warmup - 1) / warmup)
    assert init * f(total) == pytest.approx(final)
    #: the historical construction tripled / x10 the lr at every step
    assert peak * f(warmup) == pytest.approx(peak * peak / init)


def test_schedule_trace_endpoints():
    import lcpr_correction as CORR

    legacy = CORR.toy_schedule_trace(base_lr=3e-4, peak_lr=3e-4, init_lr=1e-4, min_lr=1e-4,
                                     steps_per_epoch=53, warmup_epochs=2, epochs=100)
    fixed = CORR.toy_schedule_trace(base_lr=1e-4, peak_lr=3e-4, init_lr=1e-4, min_lr=1e-4,
                                    steps_per_epoch=53, warmup_epochs=2, epochs=100)
    assert legacy["total_steps"] == 5300
    assert legacy["warmup_steps"] == 106
    assert legacy["endpoints_as_traced"]["init"] == pytest.approx(3e-4)
    assert legacy["endpoints_as_traced"]["peak"] == pytest.approx(9e-4)
    assert legacy["endpoints_as_traced"]["final"] == pytest.approx(3e-4)
    assert fixed["endpoints_as_traced"]["init"] == pytest.approx(1e-4)
    assert fixed["endpoints_as_traced"]["peak"] == pytest.approx(3e-4)
    assert fixed["endpoints_as_traced"]["final"] == pytest.approx(1e-4)
    #: and the corrected recipe entry is what the trainer now uses as base lr
    assert C.DMPNN_RECIPES["D1F"]["optimizer_base_lr"] == pytest.approx(1e-4)
    assert C.DMPNN_RECIPES["D2F"]["optimizer_base_lr"] == pytest.approx(1e-4)
    assert C.DMPNN_RECIPES["D2F"]["peak_lr"] == pytest.approx(1e-3)
    assert C.DMPNN_RECIPES["D1"]["optimizer_base_lr"] == pytest.approx(3e-4)
    assert C.DMPNN_RECIPES["D2"]["optimizer_base_lr"] == pytest.approx(1e-3)
