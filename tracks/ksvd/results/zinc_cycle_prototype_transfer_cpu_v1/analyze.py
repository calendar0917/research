"""zinc_cycle_prototype_transfer_cpu_v1 — consolidated analysis.

Reads only the frozen round artifacts produced by freeze_prototype.py and
evaluate_valid.py; produces analysis.json and the routing/coverage/cancellation
summary tables referenced by REPORT.md. No new fitting, no model probe.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).resolve().parent


def read_json(name: str) -> Any:
    return json.loads((OUT / name).read_text())


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(x: str) -> float:
    return float(x)


def main() -> int:
    train_table = read_json("train_table.json")
    train_metrics = read_json("train_metrics.json")
    train_routing = read_json("train_routing.json")
    loo = read_json("loo_coverage.json")
    main_table = read_json("main_table.json")
    gate = read_json("gate.json")
    bootstrap = read_json("bootstrap.json")
    coverage = read_json("coverage.json")
    cancel = read_json("cancellation.json")
    sens = read_json("sensitivity.json")
    tail = read_json("tail_summary.json")
    benchmark = read_json("benchmark_marker.json")
    checks = read_json("checks.json")
    routing_rows = read_csv("routing_table.csv")
    group_rows = read_csv("group_table.csv")
    per_row = read_csv("per_row_valid.csv")

    # --- A. train table / unidentifiable part ---
    by_group_support: dict[str, Any] = {}
    k_train = [c for c in train_table["class_table"]]
    for name in ("k=0", "k=-1", "k=-2", "k<=-3"):
        classes = [c for c in k_train if c["k_group"] == name]
        by_group_support[name] = {
            "n_classes": len(classes),
            "n_rows": int(sum(c["n_support"] for c in classes)),
            "n_conflict_classes": int(sum(1 for c in classes if not c["is_consistent"])),
            "n_singleton_classes": int(sum(1 for c in classes if c["n_support"] == 1)),
        }

    # --- B. valid coverage by route x group ---
    route_group: dict[str, Any] = {}
    for row in routing_rows:
        route_group.setdefault(row["route"], {})[row["group"]] = {
            "n": int(row["n"]) if row["n"] else 0,
            "q_B_mae_vs_c": f(row["q_B_mae_vs_c"]) if row["q_B_mae_vs_c"] else None,
            "q_H_mae_vs_c": f(row["q_H_mae_vs_c"]) if row["q_H_mae_vs_c"] else None,
            "y_gain_cal": f(row["gain_cal"]) if row["gain_cal"] else None,
            "gain_contrib": f(row["gain_contrib"]) if row["gain_contrib"] else None,
        }

    # --- C. non-0172 consistent-hit cancellation (the bulk story) ---
    excl = [r for r in per_row if int(r["valid_row"]) != 172 and r["route"] == "CONSISTENT_HIT"]
    e_g = np.array([f(r["h_err_vs_g"]) for r in excl])
    e_c_B = np.array([f(r["q_B_err_vs_c"]) for r in excl])
    e_c_H = np.array([f(r["q_H_err_vs_c"]) for r in excl])
    bulk = {
        "n": len(excl),
        "e_g_mae": float(np.mean(np.abs(e_g))),
        "e_c_B_mae": float(np.mean(np.abs(e_c_B))),
        "e_c_H_mae": float(np.mean(np.abs(e_c_H))),
        "opposite_sign_rate_B": float(np.mean(np.sign(e_g) != np.sign(e_c_B))),
        "opposite_sign_rate_H": float(np.mean(np.sign(e_g) != np.sign(e_c_H))),
        "sum_abs_components_B": float(np.mean(np.abs(e_g) + np.abs(e_c_B))),
        "abs_of_sum_B": float(np.mean(np.abs(e_g + e_c_B))),
        "triangle_gap_B": float(np.mean(np.abs(e_g + e_c_B) - (np.abs(e_g) + np.abs(e_c_B)))),
        "sum_abs_components_H": float(np.mean(np.abs(e_g) + np.abs(e_c_H))),
        "abs_of_sum_H": float(np.mean(np.abs(e_g + e_c_H))),
        "triangle_gap_H": float(np.mean(np.abs(e_g + e_c_H) - (np.abs(e_g) + np.abs(e_c_H)))),
        "y_B_cal_mae": float(np.mean(np.abs(e_g + e_c_B + f(excl[0]["b_y_COMP"])))),
        "y_H_cal_mae": float(np.mean(np.abs(e_g + e_c_H + f(excl[0]["b_H"])))),
        "interpretation": (
            "on train-consistent exact-hit valid rows the body error e_g=h-g dominates (~0.087); "
            "Q's own cycle error e_c_B is already tiny (~0.0011) and partially cancels e_g; "
            "forcing q_H=c removes that small favourable cancellation, so y stays flat/slightly worse"
        ),
    }

    analysis = {
        "protocol_version": "zinc-cycle-prototype-transfer-cpu-v1",
        "A_train_table": {
            "n_train_rows": train_table["n_train_rows"],
            "n_exact_classes": train_table["n_exact_classes"],
            "n_consistent_classes": train_table["n_consistent_classes"],
            "n_conflict_classes": train_table["n_conflict_classes"],
            "n_singleton_classes": train_table["n_singleton_classes"],
            "n_multi_classes": train_table["n_multi_classes"],
            "collision_free": train_table["collision_free"],
            "c_rule_max_abs": train_table["c_rule_max_abs"],
            "max_span_consistent": train_table["max_span_consistent"],
            "span_ok": train_table["span_ok"],
            "train_class_constant_l1_floor": train_table["train_class_constant_l1_floor"],
            "by_k_group": by_group_support,
            "conflict_class_ids": train_table["conflict_class_ids"],
            "train_routing": train_routing,
            "loo_coverage": loo,
        },
        "B_valid_coverage": {
            "routes_overall": {r: int(sum(f(x["n"]) for g, x in v.items() if g != "overall")) for r, v in route_group.items()},
            "route_by_group": route_group,
            "consistent_hit_k_match": coverage["CONSISTENT_HIT"],
            "new_conflict_rows": coverage["new_conflict_rows"],
        },
        "C_cancellation": {
            "bulk_consistent_hit_excl_0172": bulk,
            "overall": cancel,
            "group_table": group_rows,
        },
        "D_tail_sensitivity": {
            "tail_summary": tail,
            "sensitivity": sens,
            "gate_markers": gate["markers"],
        },
        "main_table": main_table,
        "gate": gate,
        "bootstrap": bootstrap,
        "benchmark_marker": benchmark,
        "checks": checks,
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print("[analysis] written analysis.json")
    print(f"[analysis] A: classes={train_table['n_exact_classes']} consistent={train_table['n_consistent_classes']} conflict={train_table['n_conflict_classes']}")
    print(f"[analysis] B: routes overall = {analysis['B_valid_coverage']['routes_overall']}")
    print(f"[analysis] C bulk: e_g={bulk['e_g_mae']:.6f} e_c_B={bulk['e_c_B_mae']:.6f} -> y_B={bulk['y_B_cal_mae']:.6f}")
    print(f"[analysis] C bulk: e_c_H={bulk['e_c_H_mae']:.2e} -> y_H={bulk['y_H_cal_mae']:.6f}")
    print(f"[analysis] D: gain excl 0172 = {sens['exclude_valid_0172']['gain_cal']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
