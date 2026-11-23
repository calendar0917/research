"""Pre-registered trigger arithmetic (no post-hoc choice of thresholds).

Reads the machine tables and writes ``analysis/gates.json`` plus append-only
``analysis/branch_log.jsonl`` rows.  A gate is a resource-allocation decision,
not a significance test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import sr_common as C


def table(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    out = {}
    for r in rows:
        try:
            out[(r["arm"], r["split"], int(r["seed"]))] = r
        except Exception:
            continue
    return out


def val(ix: dict, arm: str, split: str, seed: int, key: str) -> float | None:
    row = ix.get((arm, split, seed))
    if not row or row.get(key) in (None, "", "nan"):
        return None
    try:
        return float(row[key])
    except ValueError:
        return None


def mean_val(ix: dict, arm: str, split: str, seeds: list[int],
             key: str = "dev_rmse") -> float | None:
    vals = [val(ix, arm, split, s, key) for s in seeds]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="stage1",
                        choices=["stage1", "baseline", "final"])
    args = parser.parse_args()
    cfg = C.load_config()
    thr = cfg["thresholds"]
    delta = float(thr["delta"])
    eps = float(thr["epsilon"])
    half = float(thr["delta_half"])
    rows = table(C.ANALYSIS_DIR / "main_table.csv")
    ix = index(rows)
    r2 = val(ix, "R2_REF", "D0", 0, "dev_rmse")
    e_mob = val(ix, "E_MOB", "D0", 0, "dev_rmse")
    f_mob = val(ix, "F_MOB", "D0", 0, "dev_rmse")
    f_raw = val(ix, "F_RAW", "D0", 0, "dev_rmse")
    e_raw = val(ix, "E_RAW", "D0", 0, "dev_rmse")
    ego2 = val(ix, "G_EGO2_5", "D0", 0, "dev_rmse")
    full = val(ix, "G_FULL5", "D0", 0, "dev_rmse")
    gates: dict[str, Any] = {"thresholds": {"delta": delta, "epsilon": eps,
                                            "delta_half": half},
                             "observed": {"R2_D0_S0": r2, "E_RAW_D0_S0": e_raw,
                                          "E_MOB_D0_S0": e_mob, "F_RAW_D0_S0": f_raw,
                                          "F_MOB_D0_S0": f_mob,
                                          "G_EGO2_D0_S0": ego2,
                                          "G_FULL5_D0_S0": full}}
    # ---- BIG -------------------------------------------------------------
    if None not in (r2, e_mob, f_mob, f_raw):
        gap = min(e_mob, f_mob, f_raw) - r2
        both_dev_better = (e_mob <= r2 and f_mob <= r2)
        r2_fit = val(ix, "R2_REF", "D0", 0, "fit_rmse_at_best")
        e_fit = val(ix, "E_MOB", "D0", 0, "fit_rmse_at_best")
        f_fit = val(ix, "F_MOB", "D0", 0, "fit_rmse_at_best")
        both_fit_better = (None not in (r2_fit, e_fit, f_fit)
                           and e_fit <= r2_fit and f_fit <= r2_fit)
        trigger = (gap > eps) and (not both_dev_better) and (not both_fit_better)
        gates["BIG"] = {
            "formula": "min(RMSE E_MOB,F_MOB,F_RAW) > R2+epsilon and not both MOB dev-better and not both MOB fit-better",
            "min_new": min(e_mob, f_mob, f_raw), "r2_plus_eps": r2 + eps,
            "gap": gap, "both_mob_dev_better": both_dev_better,
            "both_mob_fit_better": both_fit_better,
            "trigger": bool(trigger),
            "action": "run E_MOB_BIG + F_MOB_BIG D0/s0" if trigger else "SKIP",
        }
    # ---- radius 3 --------------------------------------------------------
    if None not in (r2, ego2, full):
        gap = ego2 - full
        trigger = (gap >= delta) and (full <= r2 + eps)
        gates["EGO3"] = {
            "formula": "RMSE(EGO2)-RMSE(FULL) >= delta and FULL <= R2+epsilon",
            "gap_ego2_minus_full": gap, "full_minus_r2": full - r2,
            "trigger": bool(trigger),
            "action": "run G_EGO3_5 D0/s0" if trigger else "SKIP",
        }
    # ---- basis -----------------------------------------------------------
    if None not in (r2, f_raw, f_mob):
        gap = abs(f_raw - f_mob)
        trigger = (gap >= delta) or (min(f_raw, f_mob) <= r2 + eps)
        gates["BASIS"] = {
            "formula": "|RMSE(F_RAW)-RMSE(F_MOB)| >= delta or min(F_RAW,F_MOB) <= R2+epsilon",
            "gap": gap, "min_f": min(f_raw, f_mob), "r2_plus_eps": r2 + eps,
            "trigger": bool(trigger),
            "action": "include F_RAW/F_MOB D0/s1" if trigger else "SKIP",
        }
    # ---- head ------------------------------------------------------------
    small = {k: val(ix, k, "D0", 0, "dev_rmse")
             for k in ("E_MOB", "F_MOB", "E_MOB_BIG", "F_MOB_BIG")}
    if all(v is not None for v in small.values()):
        gains = {"E": small["E_MOB"] - small["E_MOB_BIG"],
                 "F": small["F_MOB"] - small["F_MOB_BIG"]}
        fam = "F" if gains["F"] >= gains["E"] else "E"
        best_big = small[f"{fam}_MOB_BIG"]
        trigger = max(gains.values()) >= delta or best_big <= (r2 + eps if r2 else 1e9)
        gates["HEAD"] = {
            "formula": "family gain SMALL-BIG >= delta or best BIG <= R2+epsilon; larger gain, tie F",
            "gains": gains, "selected_family": fam,
            "best_big": best_big, "trigger": bool(trigger),
            "action": f"include {fam}_MOB SMALL/BIG D0/s1" if trigger else "SKIP",
        }
    # ---- D0 seed package status (only when s1 exists) --------------------
    def seed_pair(arm: str, ref: str) -> dict[str, Any] | None:
        g0 = None
        a0, b0 = val(ix, ref, "D0", 0, "dev_rmse"), val(ix, arm, "D0", 0, "dev_rmse")
        a1, b1 = val(ix, ref, "D0", 1, "dev_rmse"), val(ix, arm, "D0", 1, "dev_rmse")
        if None in (a0, b0, a1, b1):
            return None
        g0, g1 = a0 - b0, a1 - b1
        consistent = (g0 >= 0 and g1 >= 0) or (g0 <= 0 and g1 <= 0)
        mean_gain = (g0 + g1) / 2
        return {"gain_s0": g0, "gain_s1": g1, "consistent": consistent,
                "mean_gain": mean_gain,
                "pass": bool(consistent and abs(mean_gain) >= half)}

    pair_gates = {}
    for name, arm, ref in (("BASE_E_MOB", "E_MOB", "R2_REF"),
                           ("BASE_F_MOB", "F_MOB", "R2_REF"),
                           ("P_BASIS_F_RAW", "F_RAW", "F_MOB"),
                           ("P_SCOPE_FULL_EGO2", "G_FULL5", "G_EGO2_5"),
                           ("P_SCOPE_EGO2_R2", "G_EGO2_5", "R2_REF"),
                           ("P_SCOPE_FULL_R2", "G_FULL5", "R2_REF"),
                           ("P_HEAD_E_BIG", "E_MOB_BIG", "E_MOB"),
                           ("P_HEAD_F_BIG", "F_MOB_BIG", "F_MOB")):
        got = seed_pair(arm, ref)
        if got:
            pair_gates[name] = got
    if pair_gates:
        gates["D0_seed_packages"] = pair_gates
    C.write_json(C.ANALYSIS_DIR / "gates.json", gates)
    C.append_jsonl(C.ANALYSIS_DIR / "branch_log.jsonl",
                   {"utc": C.now_utc(), "event": "GATE_EVALUATION",
                    "stage": args.stage, "gates": gates})
    print(json.dumps(C.jsonable(gates), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())