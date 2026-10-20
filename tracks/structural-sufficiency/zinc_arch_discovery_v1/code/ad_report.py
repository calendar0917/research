"""Build the six-candidate main table and a machine-readable round decision.

Reads only local run logs; never constructs test data or test labels.
"""

from __future__ import annotations

import argparse
import json

import ad_common as C

MAIN_RUNS = (
    ("R0", "R0_s0", "R0"),
    ("R1", "R1_s0", "R1"),
    ("T", "T_s0", "T"),
    ("P0", "P0_s0", "P0"),
    ("P1", "P1_s0", "P1"),
    ("B", "B_s0", "B"),
)
FOLLOW_RUNS = (
    ("P0 corrected", "corr_P0_s0", "P0"),
    ("P1 corrected", "corr_P1_s0", "P1"),
    ("T wd=1e-4", "wd_T_s0", "T"),
)
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
    "status": "completed (historical reference; not rerun)",
    "ssl": "frozen train-only masked SSL",
}


def _ssl_label(info: dict) -> str:
    ssl = (info or {}).get("ssl") or {}
    if not ssl.get("source"):
        return "none"
    return "yes (P3 seed0, fine-tuned)"


def _load(run_dir: str, label: str, candidate: str) -> dict | None:
    path = C.RUNS / run_dir / "train.json"
    if not path.exists():
        return None
    d = C.read_json(path)
    return {
        "model": label,
        "candidate": candidate,
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
        "ssl": _ssl_label(d.get("model_info")),
        "weight_decay": float(d["recipe"].get("weight_decay", 0.0)),
        "parameter_audit": d.get("parameter_audit"),
        "gradient_audit": d.get("gradient_audit"),
        "path": str(path),
        "run_dir": run_dir,
    }


def _row_md(r: dict, p3: dict) -> str:
    delta = r["best_valid_mae"] - p3["best_valid_mae"]
    return (
        f"| {r['model']} | {r['best_valid_mae']:.6f} | {r['best_epoch']} | "
        f"{r['train_mae_eval_mode_at_best']:.6f} | {r['parameters']:,} | {r['steps']:,} | "
        f"{(r['gpu_hours'] or 0):.3f} | {r['status']} | {r['ssl']} | {delta:+.6f} |"
    )


def build() -> dict:
    p3 = dict(P3_S0)
    p3["delta_vs_p3_s0"] = 0.0
    main = []
    for label, run_dir, candidate in MAIN_RUNS:
        row = _load(run_dir, label, candidate)
        if row is not None:
            row["delta_vs_p3_s0"] = row["best_valid_mae"] - p3["best_valid_mae"]
            main.append(row)
    follow = []
    for label, run_dir, candidate in FOLLOW_RUNS:
        row = _load(run_dir, label, candidate)
        if row is not None:
            row["delta_vs_p3_s0"] = row["best_valid_mae"] - p3["best_valid_mae"]
            follow.append(row)
    training_gpu_h = sum((r["gpu_hours"] or 0.0) for r in main + follow)
    timing_gpu_h = 0.03  # measured 2-epoch timing/smoke attempts, rounded up
    table = {
        "protocol": C.ROUND_ID,
        "p3_reference": p3,
        "candidates": main,
        "followups": follow,
        "ledger": {
            "training_gpu_hours": training_gpu_h,
            "timing_and_failed_smoke_gpu_hours": timing_gpu_h,
            "total_this_round_gpu_hours": training_gpu_h + timing_gpu_h,
            "cap_gpu_hours": 10.0,
            "p3_reference_gpu_hours_not_rerun": p3["gpu_hours"],
        },
    }
    C.write_json(C.ANALYSIS / "main_table.json", table)
    header = (
        "| model | best-valid | best epoch | train@best | params | steps | GPU-h | status | SSL | Δ vs P3 s0 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|\n"
    )
    lines = [
        header.rstrip("\n"),
        f"| P3 s0 | {p3['best_valid_mae']:.6f} | {p3['best_epoch']} | {p3['train_mae_eval_mode_at_best']:.6f} | "
        f"{p3['parameters']:,} | {p3['steps']:,} | {p3['gpu_hours']:.3f} | {p3['status']} | {p3['ssl']} | — |",
    ]
    lines += [_row_md(r, p3) for r in main]
    lines += ["", "### Pre-declared follow-ups", "", header.rstrip("\n")]
    lines += [_row_md(r, p3) for r in follow]
    lines += ["", f"Training GPU-hours this round: {training_gpu_h:.3f}; "
                  f"timing/smoke ~{timing_gpu_h:.2f}; total ~{training_gpu_h + timing_gpu_h:.2f} "
                  f"of a 10.0 A100-h cap. P3 reference was not rerun."]
    (C.ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def decision(table: dict) -> dict:
    rows = table["candidates"] + table["followups"]
    best = min(rows, key=lambda r: r["best_valid_mae"]) if rows else None
    p3 = table["p3_reference"]
    below_p3 = [r["model"] for r in rows if r["best_valid_mae"] < p3["best_valid_mae"]]
    below_target = [r["model"] for r in rows if r["best_valid_mae"] <= 0.075]
    out = {
        "protocol": C.ROUND_ID,
        "observed": {
            "p3_s0_best_valid": p3["best_valid_mae"],
            "candidate_best": None if best is None else {
                "model": best["model"],
                "candidate": best["candidate"],
                "best_valid_mae": best["best_valid_mae"],
                "best_epoch": best["best_epoch"],
                "delta_vs_p3_s0": best["best_valid_mae"] - p3["best_valid_mae"],
                "train_mae_eval_mode_at_best": best["train_mae_eval_mode_at_best"],
            },
            "candidates_below_p3": below_p3,
            "candidates_below_0_075": below_target,
            "followups": [
                {"model": r["model"], "best_valid_mae": r["best_valid_mae"],
                 "best_epoch": r["best_epoch"], "delta_vs_p3_s0": r["delta_vs_p3_s0"]}
                for r in table["followups"]
            ],
            "ledger": table["ledger"],
        },
        "hypotheses": [],
        "next_choice": None,
        "test": "not read; no test prediction or MAE computed",
    }
    if best is None:
        out["hypotheses"] = ["No candidate run is present locally; nothing to compare."]
        out["next_choice"] = "Finish/pull the formal runs before deciding."
    else:
        out["hypotheses"] = [
            "The round did not find a representation+readout that beats the P3 seed-0 best-valid reference. This is a round-level negative result, not evidence that the broad global-attention, path, block, or association families are invalid.",
            "T is the best new candidate, but its train@best is lower than P3's while its best-valid is worse, so its gap is not explained by underfitting. The single pre-declared 10x weight-decay correction did not improve it.",
            "The P0/P1 raw sampled-walk sum/count leak was real: correcting it moved the same-seed best-valid from 0.625814/0.729133 to 0.332778/0.321900. The corrected path models still underfit (train@best 0.179/0.096) and remain far from P3, so the leak was not the only limitation.",
        ]
        out["next_choice"] = (
            "Stop this round and report no strong model. The most worth keeping new model is T "
            "(0.091375, 462,081 params, 0.356 GPU-h). The next purchase, if any, should be a targeted "
            "diagnosis of why T's rank-16 association readout loses to P3's Null-composer readout "
            "(information loss vs. optimization), or a stronger path pattern vocabulary/readout; "
            "do not start a seventh family or a full GRIT reproduction on this evidence."
        )
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
