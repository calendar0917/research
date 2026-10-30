from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import v2_common as C
import v2_evaluate as E
import v2_folds as F


def _make_run(root: Path, key: str, rows, y, rmse_target: float, recipe: str = "R0"):
    d = root / key
    d.mkdir(parents=True, exist_ok=True)
    # A constant signed offset gives the requested RMSE exactly.
    pred = y + rmse_target
    np.savez(d / "predictions.npz", rows=np.asarray(rows, dtype=np.int64), pred=pred)
    payload = {
        "key": key,
        "recipe": recipe,
        "metrics": {
            "dev_rmse_raw": rmse_target,
            "dev_mae_raw": rmse_target,
            "train_rmse_raw": rmse_target,
            "generalisation_gap": 0.0,
        },
        "monitor_labels_read": False,
    }
    (d / "train.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def test_cv_analysis_paths_and_verdicts(monkeypatch, tmp_path):
    folds = F.load_folds()
    positions = F.fold_positions(folds, 0)
    fit = C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)[np.asarray(positions, dtype=np.int64)]
    runs = tmp_path / "runs"
    values = {
        "F0_A_R0_s0": 0.90,
        "F0_A_R1_s0": 0.85,
        "F0_C_R0_s0": 0.92,
        "F0_C_R1_s0": 0.90,
        "F0_E_R0_s0": 0.93,
        "F0_A_SSL_sel_s0": 0.82,
    }
    for key, v in values.items():
        _make_run(runs, key, positions, y, v, recipe="R1" if key.endswith("R1_s0") else "R0")
    monkeypatch.setattr(C, "RUNS", runs)
    monkeypatch.setattr(C, "ANALYSIS", tmp_path)
    monkeypatch.setattr(C, "BOOTSTRAP_N", 100)
    lock = tmp_path / "recipe_lock.json"
    lock.write_text(json.dumps({"selection": {"A": "R1", "C": "R1"}}), encoding="utf-8")
    gate = tmp_path / "ssl_gate.json"
    gate.write_text(json.dumps({"gate_pass": True, "status": "SSL_PRETEXT_LEARNED"}), encoding="utf-8")
    monkeypatch.setattr(C, "RECIPE_LOCK", lock)
    monkeypatch.setattr(C, "SSL_GATE", gate)
    cv = E.cv_analysis()
    assert cv["H1"]["folds"][0]["seeds"] == [0]
    assert abs(cv["H1"]["mean_delta"] - 0.03) < 1e-9
    assert cv["H1"]["promising"] is True
    assert cv["H2"]["arms"]["A"]["selected_recipe"] == "R1"
    assert abs(cv["H2"]["arms"]["A"]["mean_delta"] - 0.05) < 1e-9
    assert cv["H2"]["arms"]["A"]["promising"] is True
    assert cv["H2"]["arms"]["C"]["promising"] is True
    assert abs(cv["H3"]["mean_delta"] - 0.03) < 1e-9
    assert cv["H3"]["promising"] is True
    assert (tmp_path / "cv_results_v2.json").exists()
