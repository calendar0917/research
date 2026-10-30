from __future__ import annotations

import numpy as np

import v2_common as C
import v2_evaluate as E


def test_metrics_and_bootstrap_are_finite():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    a = y + rng.normal(scale=0.2, size=200)
    b = y + rng.normal(scale=0.4, size=200)
    m = E.metrics(y, a)
    assert m["n"] == 200
    assert m["rmse"] > 0
    assert abs(E.rmse(y, y)) < 1e-12
    boot = E.bootstrap_rmse_diff(y, b, a, n_boot=200, seed=0)
    assert boot["point"] > 0
    assert boot["ci95_low"] <= boot["mean"] <= boot["ci95_high"]
    assert boot["n_boot"] == 200
    same = E.bootstrap_rmse_diff(y, a, a, n_boot=50, seed=0)
    assert abs(same["point"]) < 1e-12
    assert same["ci95_low"] <= 0 <= same["ci95_high"]


def test_decision_verdicts_from_synthetic_cv(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "ANALYSIS", tmp_path)
    cv = {
        "H1": {"promising": True, "mean_delta": 0.04, "all_folds_favor_A": True},
        "H2": {
            "any_promising": False,
            "arms": {
                "A": {"selected_recipe": "R0", "mean_delta": 0.0, "promising": False},
                "C": {"selected_recipe": "R1", "mean_delta": 0.005, "promising": False},
            },
        },
        "H3": {
            "promising": False,
            "mean_delta": -0.01,
            "consistent_direction": False,
            "pretext_gate": {"gate_pass": False, "status": "PRETEXT_NOT_LEARNED"},
        },
    }
    monitor = {
        "H1": {"delta_E_minus_A": 0.03},
        "H3": {"delta_scratch_minus_ssl": -0.01},
        "H4": {
            "CPU_rmse": 0.72,
            "A_rmse": 0.85,
            "A_minus_CPU": 0.13,
            "bootstrap_A_minus_CPU": {"point": 0.13, "ci95_low": 0.10, "ci95_high": 0.16},
        },
    }
    dec = E.decision(cv, monitor)
    assert dec["verdicts"]["H1"]["verdict"] == "PROMISING"
    assert dec["verdicts"]["H2"]["verdict"] == "NOT_PROMISING"
    assert dec["verdicts"]["H3"]["verdict"] == "PRETEXT_NOT_LEARNED"
    assert dec["verdicts"]["H4"]["verdict"] == "CPU_BASELINE_BETTER"
    assert (tmp_path / "decision_v2.json").exists()


def test_monitor_analysis_with_synthetic_predictions(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    rows = list(range(n))
    y = rng.normal(size=n)
    preds = {
        "full_A_scratch_s0": y + 0.2,
        "full_A_scratch_s1": y + 0.21,
        "full_E_R0_s0": y + 0.25,
        "full_C_tuned_s0": y + 0.23,
        "full_A_SSL_sel_s0": y + 0.19,
        "CPU_xgb_P3": y + 0.10,
    }
    manifest = {
        "predictions": {
            "full_A_scratch_s0": {"arm": "A", "recipe": "R0", "seed": 0, "variant": "scratch"},
            "full_A_scratch_s1": {"arm": "A", "recipe": "R0", "seed": 1, "variant": "scratch"},
            "full_E_R0_s0": {"arm": "E", "recipe": "R0", "seed": 0, "variant": "scratch"},
            "full_C_tuned_s0": {"arm": "C", "recipe": "R1", "seed": 0, "variant": "scratch"},
            "full_A_SSL_sel_s0": {"arm": "A", "recipe": "R0", "seed": 0, "variant": "ssl"},
            "CPU_xgb_P3": {"arm": "CPU", "recipe": "CPU", "seed": None, "variant": "cpu"},
        }
    }
    monkeypatch.setattr(E, "_load_manifest_predictions", lambda: (manifest, rows, y, preds))
    monkeypatch.setattr(C, "ANALYSIS", tmp_path)
    monkeypatch.setattr(C, "PREDICTIONS", tmp_path)
    (tmp_path / "MANIFEST.json").write_text("{}", encoding="utf-8")
    (tmp_path / "MANIFEST.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    out = E.monitor_analysis()
    assert out["n_molecules"] == n
    assert out["H1"]["delta_E_minus_A"] > 0
    assert out["H3"]["delta_scratch_minus_ssl"] > 0
    assert out["H4"]["A_minus_CPU"] > 0
    assert out["summaries"]["A:scratch"]["n_predictions"] == 2
