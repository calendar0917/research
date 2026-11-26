#!/usr/bin/env python
"""Phase A analysis and the frozen route-gate decision (branch_decision.json).

Reads only the four A arms' own outputs.  No D0 edit result is opened before
``analysis/branch_decision.json`` is written.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import sr_common as C

ARMS = {"A_IS": ("IND", "STAR"), "A_IJ": ("IND", "FULL"),
        "A_FS": ("FULL", "STAR"), "A_FJ": ("FULL", "FULL")}
CONTRASTS = [("g_eta_I", "A_IS", "A_IJ"), ("g_eta_F", "A_FS", "A_FJ"),
             ("g_H_S", "A_IS", "A_FS"), ("g_H_J", "A_IJ", "A_FJ")]


def _load(arm: str) -> dict:
    d = C.RESULTS_DIR / "A" / arm
    summary = C.read_json(d / "summary.json")
    curves = C.read_json(d / "curves.json")["curves"]
    preds = np.load(d / "predictions.npz")
    return {"summary": summary, "curves": curves, "preds": preds, "dir": d}


def _paired(a: np.ndarray, b: np.ndarray, y: np.ndarray, n_boot: int = 2000,
            seed: int = C.BOOT_SEED) -> dict:
    se_a = (a - y) ** 2
    se_b = (b - y) ** 2
    rng = np.random.default_rng(int(seed))
    n = y.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    d_rmse = np.sqrt(se_b[idx].mean(axis=1)) - np.sqrt(se_a[idx].mean(axis=1))
    d_mae = np.abs(b[idx] - y[idx]).mean(axis=1) - np.abs(a[idx] - y[idx]).mean(axis=1)
    k = int(math.ceil(0.05 * n))
    key = 0.5 * (se_a + se_b)
    keep = np.argsort(key)[:-k]
    return {
        "n": int(n),
        "delta_rmse": float(np.sqrt(se_b.mean()) - np.sqrt(se_a.mean())),
        "delta_rmse_ci_low": float(np.quantile(d_rmse, 0.025)),
        "delta_rmse_ci_high": float(np.quantile(d_rmse, 0.975)),
        "delta_mae": float(np.abs(b - y).mean() - np.abs(a - y).mean()),
        "delta_mae_ci_low": float(np.quantile(d_mae, 0.025)),
        "delta_mae_ci_high": float(np.quantile(d_mae, 0.975)),
        "frac_molecules_b_better": float((se_b < se_a).mean()),
        "trim5pct_delta_rmse": float(np.sqrt(se_b[keep].mean()) - np.sqrt(se_a[keep].mean())),
        "trim5pct_n_dropped": int(k),
    }


def _sync_epoch(one: dict, two: dict) -> int:
    """Epoch minimising the mean of the two arms' Valid RMSE (frozen rule)."""
    best_epoch, best_value = None, float("inf")
    for r1, r2 in zip(one["curves"], two["curves"]):
        value = 0.5 * (float(r1["valid_rmse"]) + float(r2["valid_rmse"]))
        if value < best_value:
            best_value, best_epoch = value, int(r1["epoch"])
    return int(best_epoch)


def main() -> int:
    cfg = C.load_config()
    sigma_a = float(cfg["sigma"]["sigma_A_official_lipo_train"])
    gate_signal = float(cfg["sigma"]["gate_eta_signal"])
    gate_mae = float(cfg["sigma"]["gate_mae_not_worse"])

    data = {arm: _load(arm) for arm in ARMS}
    y = np.asarray(data["A_IS"]["preds"]["valid_y_true"], dtype=np.float64)
    rows = np.asarray(data["A_IS"]["preds"]["valid_row_ids"], dtype=np.int64)
    for arm in ARMS:
        if not np.array_equal(np.asarray(data[arm]["preds"]["valid_row_ids"]), rows):
            raise RuntimeError(f"{arm} valid row order differs")

    contrasts = {}
    rows_out = []
    for name, a_arm, b_arm in CONTRASTS:
        a = data[a_arm]
        b = data[b_arm]
        best = _paired(a["preds"]["best_valid_y_pred"], b["preds"]["best_valid_y_pred"], y)
        final = _paired(a["preds"]["final_valid_y_pred"], b["preds"]["final_valid_y_pred"], y)
        entry = {
            "contrast": name, "a": a_arm, "b": b_arm,
            "definition": f"RMSE({a_arm}) - RMSE({b_arm}); positive means {b_arm} better",
            "best": best,
            "best_rmse_a": float(a["summary"]["best_valid_rmse"]),
            "best_rmse_b": float(b["summary"]["best_valid_rmse"]),
            "delta_best": float(a["summary"]["best_valid_rmse"] - b["summary"]["best_valid_rmse"]),
            "delta_final": float(final["delta_rmse"]),
            "final_ci_low": final["delta_rmse_ci_low"],
            "final_ci_high": final["delta_rmse_ci_high"],
            "delta_last20": float(a["summary"]["last20_valid_rmse"]
                                  - b["summary"]["last20_valid_rmse"]),
            "best_epoch_a": int(a["summary"]["best_epoch"]),
            "best_epoch_b": int(b["summary"]["best_epoch"]),
        }
        # synchronized epoch: same epoch for both arms
        ep = _sync_epoch(a, b)
        ra = a["curves"][ep - 1]
        rb = b["curves"][ep - 1]
        entry["sync_epoch"] = ep
        entry["delta_sync"] = float(ra["valid_rmse"] - rb["valid_rmse"])
        entry["sync_rmse_a"] = float(ra["valid_rmse"])
        entry["sync_rmse_b"] = float(rb["valid_rmse"])
        entry["mae_a_best"] = float(a["summary"]["best_valid_mae"])
        entry["mae_b_best"] = float(b["summary"]["best_valid_mae"])
        entry["delta_mae_best"] = float(a["summary"]["best_valid_mae"]
                                        - b["summary"]["best_valid_mae"])
        contrasts[name] = entry
        rows_out.append({
            "contrast": name, "a": a_arm, "b": b_arm,
            "rmse_a_best": entry["best_rmse_a"], "rmse_b_best": entry["best_rmse_b"],
            "delta_best": entry["delta_best"],
            "delta_final": entry["delta_final"],
            "delta_final_ci_low": entry["final_ci_low"],
            "delta_final_ci_high": entry["final_ci_high"],
            "delta_sync": entry["delta_sync"], "sync_epoch": ep,
            "delta_last20": entry["delta_last20"],
            "delta_mae_best": entry["delta_mae_best"],
            "delta_rmse_boot_mean": best["delta_rmse"],
            "delta_rmse_boot_ci_low": best["delta_rmse_ci_low"],
            "delta_rmse_boot_ci_high": best["delta_rmse_ci_high"],
            "frac_molecules_b_better": best["frac_molecules_b_better"],
            "trim5pct_delta_rmse": best["trim5pct_delta_rmse"],
        })

    # Difference-in-differences of the eta gain in MSE units (descriptive only).
    msq = {}
    for arm in ARMS:
        se = (np.asarray(data[arm]["preds"]["best_valid_y_pred"], dtype=np.float64) - y) ** 2
        msq[arm] = float(se.mean())
    eta_gain_i = msq["A_IS"] - msq["A_IJ"]
    eta_gain_f = msq["A_FS"] - msq["A_FJ"]
    interaction = {
        "eta_mse_gain_H_IND": eta_gain_i,
        "eta_mse_gain_H_FULL": eta_gain_f,
        "difference_in_differences": eta_gain_f - eta_gain_i,
        "note": "descriptive interaction of one training run per cell; not a universal "
                "causal interaction",
    }

    # ---------------- frozen route gate -------------------------------
    refs = C.read_json(C.ANALYSIS_DIR / "references.json")
    b_ref = refs.get("B_ref")
    root_ratio = float(cfg["sigma"]["root_competitiveness_ratio"])
    full_pref = float(cfg["sigma"]["root_h_full_preference_sigma"])
    gate = {}
    for h in ("IND", "FULL"):
        eta_key = "g_eta_I" if h == "IND" else "g_eta_F"
        star_arm = "A_IS" if h == "IND" else "A_FS"
        full_arm = "A_IJ" if h == "IND" else "A_FJ"
        e = contrasts[eta_key]
        cond1 = e["delta_best"] >= gate_signal
        cond2 = (e["delta_final"] >= gate_signal) or (e["delta_last20"] >= gate_signal)
        cond3 = e["delta_mae_best"] <= gate_mae
        rmse_full = float(data[full_arm]["summary"]["best_valid_rmse"])
        cond4 = b_ref is not None and rmse_full <= root_ratio * float(b_ref)
        gate[h] = {
            "eta_contrast": eta_key,
            "g_eta_best": e["delta_best"], "gate_signal": gate_signal,
            "g_eta_final": e["delta_final"], "g_eta_last20": e["delta_last20"],
            "delta_mae_best": e["delta_mae_best"], "gate_mae": gate_mae,
            "full_valid_rmse": rmse_full,
            "root_limit": (None if b_ref is None else root_ratio * float(b_ref)),
            "cond_signal_best": bool(cond1), "cond_signal_direction": bool(cond2),
            "cond_mae": bool(cond3), "cond_competitive": bool(cond4),
            "signal_ok": bool(cond1 and cond2 and cond3),
            "root_eligible": bool(cond1 and cond2 and cond3 and cond4),
        }
    eligible = [h for h in ("IND", "FULL") if gate[h]["root_eligible"]]
    if not eligible:
        branch = "RESPONSE"
        chosen_h = None
        reason = "no fixed H satisfies both the eta signal gate and the competitiveness gate"
    elif len(eligible) == 1:
        branch, chosen_h = "ROOT", eligible[0]
        reason = f"only {eligible[0]} passes every ROOT gate"
    else:
        rmse_ind = gate["IND"]["full_valid_rmse"]
        rmse_full = gate["FULL"]["full_valid_rmse"]
        if rmse_ind - rmse_full >= full_pref:
            chosen_h = "FULL"
            reason = (f"both H pass; H_FULL lowers the FULL-arm Valid RMSE by "
                      f"{rmse_ind - rmse_full:.6f} >= {full_pref:.6f}")
        else:
            chosen_h = "IND"
            reason = (f"both H pass; H_FULL's advantage {rmse_ind - rmse_full:.6f} is below "
                      f"{full_pref:.6f}, so the default H_IND is kept")
        branch = "ROOT"

    complete = all((C.RESULTS_DIR / "A" / a / "summary.json").is_file() for a in ARMS)
    if not complete or b_ref is None:
        branch = "RESPONSE"
        gate_status = "A_UNRESOLVED"
        chosen_h = None
        reason = ("A incomplete or the reference is REFERENCE_BLOCKED; the preset engineering "
                  "fallback moves to RESPONSE and must not be read as evidence that FGCR lacks "
                  "benefit or competitiveness")
    else:
        gate_status = "A_RESOLVED"

    d0_rows = ({"1": "ROOT_STAR", "2": "ROOT_FULL", "G": "GINE_G0"} if branch == "ROOT"
               else {"1": "RESP_PRODUCT", "2": "RESP_FUSION_MLP", "G": "GINE_G0"})

    decision = {
        "round_id": C.ROUND_ID, "written_at": C.utc_stamp(),
        "phase_a_complete": bool(complete), "gate_status": gate_status,
        "branch": branch, "chosen_H": chosen_h, "d0_rows": d0_rows,
        "reason": reason,
        "B_ref": b_ref, "B_ref_status": refs.get("B_ref_status"),
        "B_ref_sources": refs.get("B_ref_sources"),
        "sigma_A": sigma_a, "sigma_Fit": float(cfg["sigma"]["sigma_fit_d0"]),
        "gate_per_H": gate, "contrasts": contrasts,
        "eta_interaction_mse": interaction,
        "arm_summaries": {a: {k: data[a]["summary"][k] for k in
                              ("best_epoch", "best_valid_rmse", "best_valid_mae",
                               "final_valid_rmse", "final_valid_mae",
                               "last20_valid_rmse", "last20_valid_mae",
                               "n_parameters", "seconds")} for a in ARMS},
        "notes": [
            "The eta signal gate is a resource-promotion rule for this round, not a "
            "significance or equivalence theorem.",
            "A CI spanning zero with an effect above the gate may be labelled "
            "PROMISING_UNCERTAIN; failing the gate is 'this round did not obtain an "
            "investment signal', not 'the joint response is useless'.",
            "GINE_G0 has no committed checkpoint or per-molecule predictions, so it stays a "
            "REPORTED_REFERENCE and is excluded from B_ref.",
        ],
    }
    C.write_json(C.ANALYSIS_DIR / "branch_decision.json", decision)
    C.write_csv(C.ANALYSIS_DIR / "phase_a_contrasts.csv", rows_out)
    print(json.dumps({"branch": branch, "chosen_H": chosen_h, "reason": reason,
                      "B_ref": b_ref, "gate": gate}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
