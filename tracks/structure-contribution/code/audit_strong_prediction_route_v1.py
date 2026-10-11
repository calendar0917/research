"""CSCL strong-prediction route audit v1 — read-only analysis (CPU, <=8 threads).

No training, no new seeds, no official valid/test load, no test prediction.
Everything here re-reads *already saved* artifacts to make the audit's tables
traceable.  Run:  uv run python <this file>
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tracks/structure-contribution/results/strong_prediction_route_audit_v1"

CSCL = ROOT / "tracks/structure-contribution"
KSVD = ROOT / "tracks/ksvd/results"

DEV_PREDS = CSCL / "results/feasibility_v2_gpu/full_y_seed0_dev_preds.npz"
FT_TARGETS = KSVD / "zinc_component_supervision_fulltrain_confirmation_seed0_v1/full_train_targets.npz"
FRESH_TARGETS = KSVD / "zinc_chemistry_component_supervision_seed0_v1/component_targets.npz"
LABEL_AUDIT = KSVD / "zinc_long_cycle_audit"


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def load_json(p):
    return json.loads(Path(p).read_text())


def k_strata():
    dev = np.load(DEV_PREDS)
    tg = np.load(FT_TARGETS, allow_pickle=False)
    row = np.asarray(dev["row"], np.int64)
    pred = np.asarray(dev["pred"], np.float64)
    pred_cal = np.asarray(dev["pred_cal"], np.float64)
    y = np.asarray(dev["y"], np.float64)
    k = np.asarray(tg["k"], np.int64)[row]
    y_full = np.asarray(tg["y"], np.float64)[row]
    assert np.max(np.abs(y - y_full)) <= 1e-6, np.max(np.abs(y - y_full))

    groups = {"k=0": k == 0, "k=-1": k == -1, "k<=-2": k <= -2}
    rows = []
    err_all = np.abs(pred_cal - y)
    tot = float(err_all.sum())
    for name, m in groups.items():
        n = int(m.sum())
        e = err_all[m]
        rows.append(
            dict(
                stratum=name,
                n=n,
                share_n=n / len(k),
                mae_raw=mae(pred[m], y[m]),
                mae_cal=mae(pred_cal[m], y[m]),
                err_mass_share=float(e.sum() / tot),
                median_abs_err=float(np.median(e)),
                p90_abs_err=float(np.percentile(e, 90)),
            )
        )
    allrow = dict(
        stratum="all",
        n=len(k),
        share_n=1.0,
        mae_raw=mae(pred, y),
        mae_cal=mae(pred_cal, y),
        err_mass_share=1.0,
        median_abs_err=float(np.median(err_all)),
        p90_abs_err=float(np.percentile(err_all, 90)),
    )
    with (OUT / "full_y_dev_k_strata.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows + [allrow])
    return rows, allrow


def component_target_stats():
    out = {}
    tg = np.load(FT_TARGETS, allow_pickle=False)
    keys = sorted(tg.files)
    out["full_train_targets_keys"] = keys
    ell = np.asarray(tg["ell"], np.float64)
    s = np.asarray(tg["s"], np.float64)
    g = np.asarray(tg["g"], np.float64)
    y = np.asarray(tg["y"], np.float64)
    c = np.asarray(tg["c"], np.float64)
    k = np.asarray(tg["k"], np.int64)
    out["identity_g_minus_ell_plus_s_max"] = float(np.max(np.abs(g - (ell + s))))
    out["identity_y_minus_g_plus_c_max"] = float(np.max(np.abs(y - (g + c))))
    if "s_SA" in keys and "epsilon" in keys:
        ssa = np.asarray(tg["s_SA"], np.float64)
        eps = np.asarray(tg["epsilon"], np.float64)
        out["epsilon_mae"] = mae(eps, np.zeros_like(eps))
        out["epsilon_max"] = float(np.max(np.abs(eps)))
        out["epsilon_mae_k0"] = mae(eps[k == 0], np.zeros(int((k == 0).sum())))
        out["s_plus_ell_minus_ssa_mae"] = mae(s + ell, ssa)
    out["k_counts"] = {int(v): int(n) for v, n in zip(*np.unique(k, return_counts=True))}
    out["k_negative_n"] = int((k < 0).sum())
    out["c_std"] = float(np.std(c))
    out["c_abs_median"] = float(np.median(np.abs(c)))
    out["c_nonzero_share"] = float(np.mean(np.abs(c) > 1e-9))
    out["var"] = {"y": float(np.var(y)), "g": float(np.var(g)), "c": float(np.var(c)), "ell": float(np.var(ell)), "s": float(np.var(s))}
    out["cov"] = {"g_c": float(np.cov(g, c)[0, 1]), "ell_s": float(np.cov(ell, s)[0, 1])}
    out["corr_ell_s"] = float(np.corrcoef(ell, s)[0, 1])
    out["c_by_k_group"] = {
        "k=0": {"n": int((k == 0).sum()), "mean_c": float(c[k == 0].mean())},
        "k=-1": {"n": int((k == -1).sum()), "mean_c": float(c[k == -1].mean())},
        "k<=-2": {"n": int((k <= -2).sum()), "mean_c": float(c[k <= -2].mean())},
    }
    out["var_share_ell_s_c"] = {
        "ell": float(np.var(ell) / (np.var(ell) + np.var(s) + np.var(c))),
        "s": float(np.var(s) / (np.var(ell) + np.var(s) + np.var(c))),
        "c": float(np.var(c) / (np.var(ell) + np.var(s) + np.var(c))),
    }
    out["constants"] = {
        str(name): float(val)
        for name, val in zip(
            np.asarray(tg["constant_names"]).tolist(),
            np.asarray(tg["constants"], np.float64).tolist(),
        )
    }
    out["mu_logP"] = float(np.asarray(tg["mu_logP"]).ravel()[0])
    out["logP_SA_raw_shapes"] = [list(tg[n].shape) for n in ("logP", "SA")]
    return out


def label_audit_stats():
    ref = load_json(LABEL_AUDIT / "stage_refine.json")
    perm = np.genfromtxt(LABEL_AUDIT / "permutation_sensitivity.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    uniq = np.asarray(perm["num_unique_basis_lengths"], np.float64)
    stable = float(np.mean(uniq <= 1.0))
    return {
        "n_total": ref["n_total"],
        "n_agree": ref["n_agree"],
        "n_mismatch": ref["n_mismatch"],
        "mismatch_ids": ref.get("mismatch_ids"),
        "snap_error_p95": ref["snap_error"]["p95"],
        "snap_error_max": ref["snap_error"]["max"],
        "constants_fitted": ref["constants_fitted"],
        "perm_rows": int(len(perm)),
        "perm_frac_stable_max_basis_len": stable,
        "perm_examples": [
            {k: (float(r[k]) if k != "molecule_id" and k != "split" else str(r[k])) for k in ("molecule_id", "split", "y", "original_max_basis_length", "perm_min", "perm_max", "num_unique_basis_lengths")}
            for r in perm[:5]
        ],
    }


def ksvd_headline():
    def j(p):
        return load_json(KSVD / p)

    fresh_gate = j("zinc_chemistry_component_supervision_seed0_v1/gate.json")
    fresh_gains = j("zinc_chemistry_component_supervision_seed0_v1/gains.json")
    ft_gate = j("zinc_component_supervision_fulltrain_confirmation_seed0_v1/gate.json")
    ft_gains = j("zinc_component_supervision_fulltrain_confirmation_seed0_v1/gains.json")
    vtc = j("zinc_full_decomposition_valid_test_confirmation_v1/analysis_summary.json")
    term = j("zinc_cycle_level_transfer_terminal_test_seed0_v1/analysis.json")
    proto = j("zinc_cycle_prototype_transfer_cpu_v1/gate.json")
    return dict(
        fresh_gate=fresh_gate,
        fresh_gains=fresh_gains,
        fulltrain_gate=ft_gate,
        fulltrain_gains=ft_gains,
        valid_test_confirmation_splits=vtc["splits"],
        valid_test_confirmation_gate_valid=vtc.get("gate_valid"),
        terminal_gate=term.get("gate"),
        terminal_valid=term.get("valid_evaluation"),
        terminal_test=term.get("test_evaluation"),
        prototype_gate=proto,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    strata, allrow = k_strata()
    index = {
        "round": "strong_prediction_route_audit_v1",
        "discipline": {
            "gpu": 0,
            "new_training": False,
            "new_seeds": False,
            "official_valid_loaded_here": False,
            "official_test_loaded_here": False,
            "test_predictions_recomputed": False,
            "note": "reads saved artifacts only; official valid/test numbers quoted from existing documents/artifacts",
        },
        "full_y_dev_k_strata": strata,
        "full_y_dev_all": allrow,
        "component_target_stats": component_target_stats(),
        "label_audit": label_audit_stats(),
        "ksvd": ksvd_headline(),
    }
    (OUT / "evidence_index.json").write_text(json.dumps(index, indent=1, sort_keys=False, default=str))
    print(json.dumps({k: v for k, v in index.items() if k not in ("ksvd",)}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
