"""Build the six-candidate main table and a machine-readable round decision.

Reads only local run logs; never constructs test data or test labels.
"""

from __future__ import annotations

import argparse
import json


import ad_common as C

CANDIDATES = ("R0", "R1", "T", "P0", "P1", "B")
P3_S0 = {
    "model": "P3",
    "config": "R3",
    "seed": 0,
    "parameters": 413505,
    "best_valid_mae": 0.07910549778419873,
    "best_epoch": 272,
    "train_mae_eval_mode_at_best": 0.011261,
    "steps": 28440,
    "gpu_hours": 0.201,
    "status": "completed (historical reference)",
    "ssl": "frozen train-only masked SSL (seed 0)",
}


def _load(candidate: str, seed: int = 0) -> dict | None:
    path = C.RUNS / f"{candidate}_s{seed}" / "train.json"
    if not path.exists():
        return None
    d = C.read_json(path)
    return {
        "model": candidate,
        "config": "seed0",
        "seed": int(d["seed"]),
        "parameters": int(d["parameter_count"]),
        "best_valid_mae": float(d["best_valid"]["valid_mae"]),
        "best_epoch": int(d["best_valid"]["epoch"]),
        "train_mae_eval_mode_at_best": float(d["fixed_objects"]["best"]["train_mae_eval_mode"]),
        "final_valid_mae": float(d["fixed_objects"]["final"]["valid_mae"]),
        "final_train_mae_eval_mode": float(d["fixed_objects"]["final"]["train_mae_eval_mode"]),
        "steps": int(d["recipe"]["actual_global_steps"]),
        "epochs": int(d["recipe"]["epochs"]),
        "gpu_hours": float(d["gpu_hours"]) if d.get("gpu_hours") is not None else None,
        "status": "completed" if not d.get("smoke_override") else "smoke/truncated",
        "ssl": (d.get("model_info", {}).get("ssl") or {}).get("source") or "none",
        "parameter_audit": d.get("parameter_audit"),
        "gradient_audit": d.get("gradient_audit"),
        "path": str(path),
    }


def build() -> dict:
    rows = []
    for cand in CANDIDATES:
        row = _load(cand)
        if row is not None:
            row["delta_vs_p3_s0"] = row["best_valid_mae"] - P3_S0["best_valid_mae"]
            rows.append(row)
    p3 = dict(P3_S0)
    p3["delta_vs_p3_s0"] = 0.0
    table = {"p3_reference": p3, "candidates": rows, "protocol": C.ROUND_ID}
    C.write_json(C.ANALYSIS / "main_table.json", table)
    lines = [
        "| model | best-valid | best epoch | train@best | params | steps | GPU-h | status | SSL | Δ vs P3 s0 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
        f"| P3 s0 | {p3['best_valid_mae']:.6f} | {p3['best_epoch']} | {p3['train_mae_eval_mode_at_best']:.6f} | {p3['parameters']:,} | {p3['steps']:,} | {p3['gpu_hours']:.3f} | {p3['status']} | yes | — |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} s0 | {r['best_valid_mae']:.6f} | {r['best_epoch']} | "
            f"{r['train_mae_eval_mode_at_best']:.6f} | {r['parameters']:,} | {r['steps']:,} | "
            f"{(r['gpu_hours'] or 0):.3f} | {r['status']} | {r['ssl']} | "
            f"{r['delta_vs_p3_s0']:+.6f} |"
        )
    (C.ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def decision(table: dict) -> dict:
    rows = table["candidates"]
    best = min(rows, key=lambda r: r["best_valid_mae"]) if rows else None
    p3 = table["p3_reference"]
    out = {
        "protocol": C.ROUND_ID,
        "observed": {
            "p3_s0_best_valid": p3["best_valid_mae"],
            "candidate_best": None if best is None else {
                "model": best["model"],
                "best_valid_mae": best["best_valid_mae"],
                "best_epoch": best["best_epoch"],
                "delta_vs_p3_s0": best["delta_vs_p3_s0"],
            },
            "candidates_below_p3": [r["model"] for r in rows if r["best_valid_mae"] < p3["best_valid_mae"]],
            "candidates_below_0_075": [r["model"] for r in rows if r["best_valid_mae"] <= 0.075],
        },
        "hypotheses": [],
        "next_choice": None,
        "test": "not read; no test prediction or MAE computed",
    }
    if best is None:
        out["hypotheses"] = ["No candidate run is present locally; nothing to compare."]
        out["next_choice"] = "Finish/pull the formal runs before deciding."
    elif best["best_valid_mae"] < p3["best_valid_mae"] - 0.001:
        out["hypotheses"].append(
            "The richest representation/readout may be worth a seed-1 repeat; this is a single-seed signal, not a stable improvement."
        )
        out["next_choice"] = f"Repeat {best['model']} at seed 1 if budget allows; do not add modules yet."
    else:
        out["hypotheses"].append(
            "At this 600-epoch / 47,400-update opportunity, none of the six representations beats the P3 seed-0 best-valid reference. This is a round-level no-strong-model result, not a claim that the broad representation families are invalid."
        )
        out["next_choice"] = "Report the negative round; the next purchase should target the most promising representation with a specific missing ingredient, not a seventh family or a full GRIT reproduction."
    C.write_json(C.ANALYSIS / "round_decision.json", out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--print", action="store_true")
    args = p.parse_args(argv)
    table = build()
    dec = decision(table)
    if args.print:
        print((C.ANALYSIS / "main_table.md").read_text())
        print(json.dumps(dec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
