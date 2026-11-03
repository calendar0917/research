"""Metrics, paired bootstrap and the pre-registered decision gates."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import nb_common as C
import nb_cv as CV


# ---------------------------------------------------------------------------
# synthetic fold results
# ---------------------------------------------------------------------------


def _branch(scores: np.ndarray, y: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": 0.5,
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_quantiles": {q: float(np.quantile(scores, float(q))) for q in ("0.05", "0.5", "0.95")},
        "score_mean_positive": float(scores[y == 1].mean()),
        "score_mean_negative": float(scores[y == 0].mean()),
        "n_predicted_positive_at_0.5": int((scores >= 0.5).sum()),
    }


def _fold_results(n_folds: int = 3, n: int = 300, seed: int = 0, signal: float = 1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    rows = np.arange(n)
    results = []
    for fold in range(n_folds):
        latent = y * signal + rng.normal(scale=1.0, size=n)
        # REAL ranks best, NULL is a noisier version, BASE noisier still
        real = 1.0 / (1.0 + np.exp(-(latent + 0.25)))
        null = 1.0 / (1.0 + np.exp(-(0.4 * latent + rng.normal(scale=1.0, size=n))))
        base = 1.0 / (1.0 + np.exp(-(0.2 * latent + rng.normal(scale=1.5, size=n))))
        results.append(
            {
                "outer_fold": fold,
                "selected_C": 0.1,
                "n_train": n,
                "n_valid": n,
                "n_positive_train": int(y.sum()),
                "n_positive_valid": int(y.sum()),
                "valid_rows": rows.tolist(),
                "train_rows": rows.tolist(),
                "branches": {
                    "BASE": _branch(base, y),
                    "BASE_NULL": _branch(null, y),
                    "BASE_REAL": _branch(real, y),
                },
                "scores": {"BASE": base, "BASE_NULL": null, "BASE_REAL": real},
            }
        )
    codes = np.arange(n) // 5  # groups of five rows
    return results, y, codes


def test_aggregate_structure():
    results, y, _ = _fold_results()
    agg = CV.aggregate(results, y)
    assert set(agg["per_fold"]["roc_auc"]) == set(C.BRANCHES)
    for branch in C.BRANCHES:
        assert len(agg["per_fold"]["roc_auc"][branch]) == 3
        assert agg["mean_roc_auc"][branch] == pytest.approx(np.mean(agg["per_fold"]["roc_auc"][branch]))
    assert len(agg["per_fold_roc_auc_diff"]["REAL_minus_NULL"]) == 3
    assert agg["mean_roc_auc_diff"]["REAL_minus_NULL"] == pytest.approx(
        np.mean(agg["per_fold_roc_auc_diff"]["REAL_minus_NULL"])
    )
    assert agg["mean_roc_auc_diff"]["NULL_minus_BASE"] == pytest.approx(
        agg["mean_roc_auc"]["BASE_NULL"] - agg["mean_roc_auc"]["BASE"]
    )


def test_paired_bootstrap_structure_and_direction():
    results, y, codes = _fold_results()
    boot = CV.paired_scaffold_bootstrap(results, y, codes, replicates=50, seed=7)
    assert boot["status"] == "OK"
    assert boot["n_valid_replicates"] == 50
    assert boot["REAL_minus_NULL"]["ci95"][0] <= boot["REAL_minus_NULL"]["ci95"][1]
    assert boot["REAL_minus_NULL"]["fraction_positive"] > 0.9
    assert boot["scope"].startswith("sampling uncertainty at fixed fitted models")


def test_paired_bootstrap_is_zero_when_the_branches_are_identical():
    results, y, codes = _fold_results()
    for rec in results:
        rec["scores"]["BASE_REAL"] = rec["scores"]["BASE_NULL"].copy()
        rec["branches"]["BASE_REAL"]["roc_auc"] = rec["branches"]["BASE_NULL"]["roc_auc"]
    boot = CV.paired_scaffold_bootstrap(results, y, codes, replicates=25, seed=3)
    assert boot["REAL_minus_NULL"]["mean"] == 0.0
    assert boot["REAL_minus_NULL"]["ci95"] == [0.0, 0.0]
    assert boot["REAL_minus_NULL"]["fraction_positive"] == 0.0


def test_paired_bootstrap_redraws_the_same_rows_for_all_branches():
    """Replicate 0 must equal a manual reproduction of the documented algorithm."""
    results, y, codes = _fold_results()
    seed = 4242
    boot = CV.paired_scaffold_bootstrap(results, y, codes, replicates=1, seed=seed)
    rng = np.random.default_rng(seed)
    per_fold = {b: [] for b in C.BRANCHES}
    for rec in sorted(results, key=lambda r: r["outer_fold"]):
        rows = np.asarray(rec["valid_rows"], dtype=np.int64)
        fold_codes = codes[rows]
        uniq = np.unique(fold_codes)
        picks = rng.integers(0, uniq.size, size=uniq.size)
        chosen = uniq[picks]
        idx = np.concatenate([np.flatnonzero(fold_codes == c) for c in chosen])
        labels = y[rows][idx]
        for b in C.BRANCHES:
            per_fold[b].append(roc_auc_score(labels, rec["scores"][b][idx]))
    expected_null = float(np.mean(per_fold["BASE_REAL"]) - np.mean(per_fold["BASE_NULL"]))
    expected_base = float(np.mean(per_fold["BASE_REAL"]) - np.mean(per_fold["BASE"]))
    assert boot["REAL_minus_NULL"]["mean"] == pytest.approx(expected_null)
    assert boot["REAL_minus_BASE"]["mean"] == pytest.approx(expected_base)


def test_bootstrap_incomplete_when_every_draw_is_single_class():
    results, y, codes = _fold_results()
    y_all_negative = np.zeros_like(y)
    boot = CV.paired_scaffold_bootstrap(results, y_all_negative, codes, replicates=10, max_attempts=3, seed=1)
    assert boot["status"] == "BOOTSTRAP_INCOMPLETE"
    assert boot["n_valid_replicates"] == 0
    assert boot["attempts_total"] <= 3


# ---------------------------------------------------------------------------
# C selection
# ---------------------------------------------------------------------------


class _StubModel:
    def __init__(self, reference: np.ndarray):
        self.reference = reference

    def predict_proba(self, X):
        p = self.reference[: X.shape[0]]
        out = np.zeros((p.size, 2), dtype=np.float64)
        out[:, 1] = p
        out[:, 0] = 1.0 - p
        return out


def _stub_fold(monkeypatch):
    rng = np.random.default_rng(0)
    n = 120
    y = rng.integers(0, 2, size=n)
    # the stub's predictions do not depend on C, so every C ties
    reference = rng.random(n)
    monkeypatch.setattr(CV, "fit_logreg", lambda X, y_, c: _StubModel(reference))
    inner = [
        {"inner_fold": 0, "train_rows": list(range(0, 60)), "valid_rows": list(range(60, 120))},
    ]
    ecfp = np.zeros((n, 4), dtype=np.uint8)
    g = np.zeros((n, C.GLOBAL_DIM), dtype=np.float64)
    real = np.zeros((n, C.N_PAIR_FEATURES), dtype=np.float64)
    null = np.zeros((n, C.N_PAIR_FEATURES), dtype=np.float64)
    features = {"ecfp4": ecfp, "g": g, "real": real, "null": null}
    fold = {"outer_fold": 0, "inner": inner, "train_rows": list(range(n)), "valid_rows": []}
    return fold, y, features


def test_c_selection_ties_resolve_to_the_smaller_value(monkeypatch):
    fold, y, features = _stub_fold(monkeypatch)
    selection = CV.select_c(fold, y, features)
    means = [selection["table"][f"{c:g}"]["mean_inner_auc"] for c in C.C_GRID]
    assert len(set(np.round(means, 12).tolist())) == 1  # a genuine tie
    assert selection["selected_C"] == min(C.C_GRID) == 0.01
    assert selection["selection_uses"] == "BASE only"


def test_c_selection_uses_the_higher_mean_when_it_is_strictly_better(monkeypatch):
    fold, y, features = _stub_fold(monkeypatch)
    calls = {"n": 0}

    def fake_fit(X, y_, c):
        # make C=1.0 strictly better by flipping which rows get high scores
        reference = np.linspace(0.0, 1.0, X.shape[0]) if c == 1.0 else np.linspace(1.0, 0.0, X.shape[0])
        calls["n"] += 1
        return _StubModel(reference)

    monkeypatch.setattr(CV, "fit_logreg", fake_fit)
    selection = CV.select_c(fold, y, features)
    grid_means = {c: selection["table"][f"{c:g}"]["mean_inner_auc"] for c in C.C_GRID}
    best = max(grid_means.values())
    assert selection["selected_mean_inner_auc"] == pytest.approx(best)
    assert grid_means[selection["selected_C"]] == pytest.approx(best)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def _metrics(real_minus_null: float, real_minus_base: float, fold_diffs, ap_diff: float):
    return {
        "mean_roc_auc_diff": {
            "REAL_minus_NULL": real_minus_null,
            "REAL_minus_BASE": real_minus_base,
            "NULL_minus_BASE": 0.0,
        },
        "mean_ap_diff": {"REAL_minus_NULL": ap_diff, "REAL_minus_BASE": 0.0, "NULL_minus_BASE": 0.0},
        "per_fold_roc_auc_diff": {
            "REAL_minus_NULL": fold_diffs,
            "REAL_minus_BASE": [0.0, 0.0, 0.0],
        },
    }


def _bootstrap(lower: float, status: str = "OK"):
    return {"status": status, "REAL_minus_NULL": {"ci95_lower": lower}}


def _correctness(ok: bool = True):
    return {"all_passed": ok}


def test_gates_promote_only_when_all_six_hold():
    gates = CV.evaluate_gates(
        _metrics(0.02, 0.012, [0.01, 0.03, 0.02], 0.001),
        _bootstrap(0.004),
        _correctness(True),
    )
    assert gates["all_passed"] is True
    assert gates["decision"] == "PROMOTE_INFORMATION_SIGNAL"
    assert gates["n_passed"] == gates["n_checks"] == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"real_minus_null": 0.016, "real_minus_base": 0.012, "fold_diffs": [0.01, 0.02, -0.001], "ap_diff": 0.001},
        {"real_minus_null": 0.016, "real_minus_base": 0.008, "fold_diffs": [0.01, 0.02, 0.01], "ap_diff": 0.001},
        {"real_minus_null": 0.014, "real_minus_base": 0.012, "fold_diffs": [0.01, 0.02, 0.01], "ap_diff": 0.001},
        {"real_minus_null": 0.016, "real_minus_base": 0.012, "fold_diffs": [0.01, 0.02, 0.01], "ap_diff": -0.001},
    ],
)
def test_gates_weak_when_a_single_condition_fails(kwargs):
    gates = CV.evaluate_gates(
        _metrics(
            kwargs["real_minus_null"], kwargs["real_minus_base"], kwargs["fold_diffs"], kwargs["ap_diff"]
        ),
        _bootstrap(0.004),
        _correctness(True),
    )
    assert gates["decision"] == "WEAK_OR_INCONSISTENT_SIGNAL"
    assert gates["all_passed"] is False
    assert gates["n_passed"] == 5


def test_gates_bootstrap_failure_blocks_promotion():
    gates = CV.evaluate_gates(
        _metrics(0.05, 0.05, [0.05, 0.05, 0.05], 0.05), _bootstrap(-0.001), _correctness(True)
    )
    assert gates["decision"] == "WEAK_OR_INCONSISTENT_SIGNAL"
    gates_incomplete = CV.evaluate_gates(
        _metrics(0.05, 0.05, [0.05, 0.05, 0.05], 0.05),
        _bootstrap(0.01, status="BOOTSTRAP_INCOMPLETE"),
        _correctness(True),
    )
    assert gates_incomplete["decision"] == "WEAK_OR_INCONSISTENT_SIGNAL"


def test_gates_no_observed_increment_when_mean_auc_diff_is_nonpositive():
    gates = CV.evaluate_gates(
        _metrics(0.0, 0.012, [0.01, -0.01, 0.0], 0.001), _bootstrap(0.004), _correctness(True)
    )
    assert gates["decision"] == "NO_OBSERVED_INCREMENT"
    gates_neg = CV.evaluate_gates(
        _metrics(-0.02, 0.012, [-0.01, -0.03, -0.02], 0.001), _bootstrap(0.004), _correctness(True)
    )
    assert gates_neg["decision"] == "NO_OBSERVED_INCREMENT"


def test_gates_infeasible_on_correctness_failure():
    gates = CV.evaluate_gates(
        _metrics(0.02, 0.012, [0.01, 0.03, 0.02], 0.001), _bootstrap(0.004), _correctness(False)
    )
    assert gates["decision"] == "INFEASIBLE"
    assert gates["all_passed"] is False


def test_gate_thresholds_are_the_registered_numbers():
    gates = CV.evaluate_gates(
        _metrics(0.0, 0.0, [0.0, 0.0, 0.0], 0.0), _bootstrap(0.0), _correctness(True)
    )
    frozen = gates["thresholds_frozen"]
    assert frozen["real_minus_null_auc_mean"] == 0.015
    assert frozen["real_minus_base_auc_mean"] == 0.010
    assert frozen["real_minus_null_ap_mean"] == 0.0
    assert frozen["all_fold_diffs_positive"] is True
    assert C.GATE_ALL_FOLD_DIFFS_POSITIVE is True


def test_gate_boundary_is_inclusive():
    gates = CV.evaluate_gates(
        _metrics(0.015, 0.010, [0.001, 0.001, 0.001], 0.0), _bootstrap(1e-9), _correctness(True)
    )
    assert gates["checks"]["g1_mean_auc_real_minus_null_ge_0.015"]["passed"] is True
    assert gates["checks"]["g3_mean_auc_real_minus_base_ge_0.010"]["passed"] is True
    assert gates["checks"]["g4_mean_ap_real_minus_null_ge_0"]["passed"] is True
    assert gates["decision"] == "PROMOTE_INFORMATION_SIGNAL"
