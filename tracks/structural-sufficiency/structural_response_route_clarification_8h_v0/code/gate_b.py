#!/usr/bin/env python
"""Frozen phase-B promotion gate (protocol section 9) on the D0 tables.

Reads ``edit_response_table.csv``, ``main_molecule_table.csv`` and
``paired_contrasts.csv``; writes ``analysis/phase_b_gate.json``.  Every threshold is
read from ``configs/experiment.json`` and never adjusted after seeing the tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sr_common as C

TARGETS = ("BASE", "ENDPOINT", "DELTA")


def _table(path: Path) -> list[dict]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(value) -> float:
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "phase_b_gate.json"))
    args = parser.parse_args()
    cfg = C.load_config()
    sigma_fit = float(cfg["sigma"]["sigma_fit_d0"])
    rel, _, rmse_ratio, mae_rel = (float(v) for v in cfg["phase_P"]["gates"])
    abs_gain = 0.015 * sigma_fit
    mae_slack = mae_rel * sigma_fit

    edits = _table(C.ROUND_DIR / "edit_response_table.csv")
    mols = _table(C.ROUND_DIR / "main_molecule_table.csv")
    contrasts = _table(C.ROUND_DIR / "paired_contrasts.csv")
    decision = C.read_json(C.ANALYSIS_DIR / "branch_decision.json")
    rows = sorted({r["row"] for r in mols})

    def ce(row: str, target: str, ckpt: str = "best", subset: str = "dev_dev_all") -> float:
        for r in edits:
            if (r["row"] == row and r["target"] == target and r["checkpoint"] == ckpt
                    and r["subset"] == subset):
                return _f(r["ce_edit_mae"])
        raise KeyError((row, target, ckpt, subset))

    def pair_mae(row: str, target: str, ckpt: str = "best",
                 subset: str = "dev_dev_all") -> float:
        for r in edits:
            if (r["row"] == row and r["target"] == target and r["checkpoint"] == ckpt
                    and r["subset"] == subset):
                return _f(r["pair_edit_mae"])
        raise KeyError((row, target, ckpt, subset))

    def dev(row: str, target: str, ckpt: str = "best", key: str = "rmse") -> float:
        for r in mols:
            if (r["row"] == row and r["target"] == target and r["checkpoint"] == ckpt
                    and r["split"] == "dev504"):
                return _f(r[key])
        raise KeyError((row, target, ckpt, key))

    def contrast(name: str, subset: str = "dev_dev_all", ckpt: str = "best"):
        for r in contrasts:
            if (r["contrast"] == name and r["subset"] == subset and r["checkpoint"] == ckpt):
                return r
        return None

    gate: dict[str, dict] = {}
    for row in rows:
        ce_b, ce_e, ce_d = (ce(row, "BASE"), ce(row, "ENDPOINT"), ce(row, "DELTA"))
        gain_e_d = ce_e - ce_d
        gain_b_d = ce_b - ce_d
        need_e = max(abs_gain, rel * ce_e)
        need_b = max(abs_gain, rel * ce_b)
        c1 = gain_e_d >= need_e
        c2 = gain_b_d >= need_b
        c3 = (pair_mae(row, "DELTA") <= pair_mae(row, "ENDPOINT")
              and pair_mae(row, "DELTA") <= pair_mae(row, "BASE"))
        base_rmse, d_rmse = dev(row, "BASE"), dev(row, "DELTA")
        c4 = (d_rmse <= rmse_ratio * base_rmse
              and dev(row, "DELTA", key="mae") <= dev(row, "BASE", key="mae") + mae_slack)
        c5 = (ce(row, "DELTA", subset="dev_dev_supported119")
              <= ce(row, "ENDPOINT", subset="dev_dev_supported119")
              and ce(row, "ENDPOINT", "final") - ce(row, "DELTA", "final") >= 0.0)
        con_e = contrast(f"{row}_D_vs_E") if row.startswith("RESP") else contrast("GINE_D_vs_E")
        fragile = False
        drop = None
        if con_e is not None and con_e.get("drop_worst_component"):
            drop = json.loads(con_e["drop_worst_component"])
            kept = float(drop.get("ce_delta_kept", 0.0))
            fragile = bool(kept <= 0.0)
        eligible = bool(c1 and c2 and c3 and c4 and c5)
        gate[row] = {
            "ce_base": ce_b, "ce_endpoint": ce_e, "ce_delta": ce_d,
            "gain_E_D": gain_e_d, "gain_B_D": gain_b_d,
            "need_gain_E": need_e, "need_gain_B": need_b,
            "cond1_gain_vs_endpoint": bool(c1), "cond2_gain_vs_base": bool(c2),
            "pair_equal": {"delta": pair_mae(row, "DELTA"),
                           "endpoint": pair_mae(row, "ENDPOINT"),
                           "base": pair_mae(row, "BASE")},
            "cond3_pair_equal_not_worse": bool(c3),
            "dev_rmse": {"base": base_rmse, "delta": d_rmse,
                         "limit": rmse_ratio * base_rmse},
            "dev_mae": {"base": dev(row, "BASE", key="mae"),
                        "delta": dev(row, "DELTA", key="mae"),
                        "limit": dev(row, "BASE", key="mae") + mae_slack},
            "cond4_molecule_regression_ok": bool(c4),
            "supported119": {"delta": ce(row, "DELTA", subset="dev_dev_supported119"),
                             "endpoint": ce(row, "ENDPOINT", subset="dev_dev_supported119")},
            "final_main_ce": {"endpoint": ce(row, "ENDPOINT", "final"),
                              "delta": ce(row, "DELTA", "final")},
            "cond5_supported_and_final": bool(c5),
            "eligible": eligible,
            "fragile_if_one_component_dropped": fragile,
            "component_dropped": (drop or {}).get("dropped_component"),
            "ce_delta_after_dropping_worst": (drop or {}).get("ce_delta_kept"),
            "drop_worst_component": drop,
            "relative_gain": (gain_e_d / ce_e) if ce_e else None,
        }

    qualified = [r for r in rows if gate[r]["eligible"]]
    chosen = None
    if qualified:
        qualified.sort(key=lambda r: (-gate[r]["relative_gain"], gate[r]["dev_rmse"]["delta"]))
        chosen = qualified[0]
    payload = {
        "round_id": C.ROUND_ID, "written_at": C.utc_stamp(),
        "branch": decision["branch"], "sigma_Fit": sigma_fit,
        "thresholds": {"abs_gain": abs_gain, "relative_gain": rel,
                       "rmse_ratio": rmse_ratio, "mae_slack": mae_slack},
        "rows": rows, "gate": gate,
        "qualified_rows": qualified, "chosen_row": chosen,
        "p_triggered": bool(qualified),
        "note": ("Promotion is a resource decision on the frozen strict-edit set; the "
                 "confidence intervals in paired_contrasts.csv are the uncertainty, and a "
                 "component-only reversal is flagged FRAGILE rather than silently accepted."),
    }
    C.write_json(args.out, payload)
    print(json.dumps({"qualified_rows": qualified, "chosen_row": chosen,
                      "per_row": {r: {k: gate[r][k] for k in
                                      ("cond1_gain_vs_endpoint", "cond2_gain_vs_base",
                                       "cond3_pair_equal_not_worse",
                                       "cond4_molecule_regression_ok",
                                       "cond5_supported_and_final", "eligible")}
                                  for r in rows}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
