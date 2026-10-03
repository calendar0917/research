"""Frozen analysis + gate for zinc-topology-crossfit-diagnostic-v1 (local CPU).

Consumes the two base result JSGs, the four readout predictions, the frozen
sub-fold index and the raw penalty labels.  Produces the main comparison, the
group-contribution table, the per-fold direction check, the coverage table and
the frozen gate.  No new model fits, no dev-tuned choices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tracks.ksvd.experiments.luyin16 import zinc_topology_crossfit_diagnostic_v1 as d
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

OUT = Path(__file__).resolve().parent
FOLDS = ("A", "B")
N_BOOT = 1000
BOOT_SEED = 20261003


def mae(p, y):
    return float(np.abs(np.asarray(p) - np.asarray(y)).mean())


def strata(y, pen):
    return {"penalty_0": pen == 0, "penalty_-1": pen == -1, "penalty_le_-2": pen <= -2}


def group_table(pred, y, pen):
    out = {}
    for name, sel in strata(y, pen).items():
        sel = np.asarray(sel)
        err = np.abs(np.asarray(pred)[sel] - np.asarray(y)[sel])
        out[name] = {"n": int(sel.sum()), "mae": float(err.mean()) if sel.any() else None,
                     "contrib": float(err.sum() / len(y))}
    out["overall"] = {"n": int(len(y)), "mae": mae(pred, y), "contrib": mae(pred, y)}
    return out


def paired_boot(p_a, p_b, y):
    rng = np.random.default_rng(BOOT_SEED)
    ea, eb = np.abs(np.asarray(p_a) - y), np.abs(np.asarray(p_b) - y)
    n = len(y)
    obs = float(ea.mean() - eb.mean())
    diffs = np.array([float(ea[rng.integers(0, n, n)].mean() - eb[rng.integers(0, n, n)].mean())
                      for _ in range(N_BOOT)])
    return {"point": obs, "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5)),
            "n_boot": N_BOOT, "seed": BOOT_SEED}


def main() -> int:
    pen, _ = zjd._load_penalties()
    idx = d.load_subfold_index()
    dev_idx = idx["dev_idx"]
    dev_pen = pen[dev_idx]
    bases = {f: json.loads((d.RESULTS_DIR / f"F_{f}.json").read_text()) for f in FOLDS}
    ro = {f: json.loads((d.RESULTS_DIR / "readouts.json").read_text())[f] for f in FOLDS}
    y = np.asarray(bases["A"]["dev_targets"], np.float64)
    assert np.array_equal(y, np.asarray(bases["B"]["dev_targets"]))
    assert np.array_equal(y, np.asarray(ro["A"]["dev_targets"]))

    pbase = {f: np.asarray(bases[f]["dev_predictions_cal"], np.float64) for f in FOLDS}
    pP = {f: np.asarray(ro[f]["dev_predictions_P"], np.float64) for f in FOLDS}
    pTP = {f: np.asarray(ro[f]["dev_predictions_TP"], np.float64) for f in FOLDS}
    pconst = {f: np.asarray(ro[f]["dev_predictions_const"], np.float64) for f in FOLDS}

    base = (pbase["A"] + pbase["B"]) / 2.0
    P = (pP["A"] + pP["B"]) / 2.0
    TP = (pTP["A"] + pTP["B"]) / 2.0
    CONST = (pconst["A"] + pconst["B"]) / 2.0

    main_table = {"base": mae(base, y), "P": mae(P, y), "TP": mae(TP, y), "const": mae(CONST, y)}
    gain_base = main_table["base"] - main_table["TP"]
    gain_topo = main_table["P"] - main_table["TP"]
    gain_const = main_table["base"] - main_table["const"]

    hist = json.loads((OUT / "corrected_historical_metrics.json").read_text())
    hist_F = hist["corrected_arms"]["F"]["correct_single_calibrated_mae"]

    groups = {name: group_table(pred, y, dev_pen) for name, pred in
              (("base", base), ("P", P), ("TP", TP))}
    per_fold = {}
    for f in FOLDS:
        per_fold[f] = {
            "base": mae(pbase[f], y), "P": mae(pP[f], y), "TP": mae(pTP[f], y),
            "TP_minus_base_gain": mae(pbase[f], y) - mae(pTP[f], y),
            "TP_minus_P_gain": mae(pP[f], y) - mae(pTP[f], y),
            "meta_mae_base": ro[f]["meta_mae_base"], "r_meta_median": ro[f]["r_meta_median"],
        }

    # delete the frozen "base max error" row
    drop = int(np.argmax(np.abs(base - y)))
    keep = np.ones(len(y), bool); keep[drop] = False
    del_gain = {"dropped_dev_index": int(dev_idx[drop]), "penalty": int(dev_pen[drop]),
                "gain_base": mae(base[keep], y[keep]) - mae(TP[keep], y[keep]),
                "gain_topo": mae(P[keep], y[keep]) - mae(TP[keep], y[keep])}

    g0 = dev_pen == 0
    base_g0 = mae(base[g0], y[g0]); P_g0 = mae(P[g0], y[g0]); TP_g0 = mae(TP[g0], y[g0])
    gate = {
        "gain_base": gain_base, "gain_topo": gain_topo, "gain_const": gain_const,
        "gain_base_ge_0.003": bool(gain_base >= 0.003),
        "gain_topo_ge_0.003": bool(gain_topo >= 0.003),
        "TP_vs_base_G0_worsening": float(TP_g0 - base_g0),
        "TP_vs_base_G0_worsening_le_0.001": bool((TP_g0 - base_g0) <= 0.001),
        "per_fold_gain_nonneg": {f: bool(per_fold[f]["TP_minus_base_gain"] >= 0) for f in FOLDS},
        "contribution_identity_ok": bool(all(
            abs(sum(groups[nm][k]["contrib"] for k in ("penalty_0", "penalty_-1", "penalty_le_-2"))
                - groups[nm]["overall"]["mae"]) < 1e-9 for nm in ("base", "P", "TP"))),
    }
    gate["passed"] = bool(gate["gain_base_ge_0.003"] and gate["gain_topo_ge_0.003"]
                          and gate["TP_vs_base_G0_worsening_le_0.001"]
                          and all(gate["per_fold_gain_nonneg"].values())
                          and gate["contribution_identity_ok"])

    boot = {"gain_base": paired_boot(base, TP, y), "gain_topo": paired_boot(P, TP, y),
            "gain_const": paired_boot(base, CONST, y)}

    # per-row listing for all ring and severe dev rows
    rows = []
    for j, i in enumerate(dev_idx.tolist()):
        if dev_pen[j] < 0:
            rows.append({"dev_row": j, "train_index": int(i), "penalty": int(dev_pen[j]),
                         "y": float(y[j]), "base": float(base[j]), "P": float(P[j]), "TP": float(TP[j]),
                         "base_err": float(abs(base[j] - y[j])), "P_err": float(abs(P[j] - y[j])),
                         "TP_err": float(abs(TP[j] - y[j])),
                         "PA": float(pbase["A"][j]), "PB": float(pbase["B"][j])})

    payload = {
        "historical_reference": {"F_8000_corrected_dev_mae": hist_F, "note": "reference only"},
        "main_table": main_table, "gain_base": gain_base, "gain_topo": gain_topo, "gain_const": gain_const,
        "group_table": groups, "per_fold": per_fold, "gate": gate, "bootstrap": boot,
        "drop_base_max_error_row": del_gain, "ring_and_severe_rows": rows,
        "n_dev": int(len(y)), "dev_strata": {"0": int(g0.sum()), "-1": int((dev_pen == -1).sum()),
                                              "le_-2": int((dev_pen <= -2).sum())},
        "official_test_loaded": False,
    }
    (OUT / "DECISION.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"main_table": main_table, "gain_base": gain_base, "gain_topo": gain_topo,
                      "gain_const": gain_const, "gate_passed": gate["passed"],
                      "per_fold": {f: per_fold[f]["TP_minus_base_gain"] for f in FOLDS},
                      "G0_worsening": gate["TP_vs_base_G0_worsening"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())