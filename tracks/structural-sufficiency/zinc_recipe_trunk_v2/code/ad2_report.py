"""Aggregate tables + real curves for zinc-recipe-trunk-v2.

Reads the run directories written by ``ad2_runner.py`` (each contains
``train.json`` and ``curve.csv``) and produces the round tables.  Nothing here
evaluates a model or touches the test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import ad_common as C  # noqa: E402
import ad2_runner as R2  # noqa: E402

EXP2 = _HERE.parent
ANALYSIS2 = EXP2 / "analysis"

RUNS = {
    "P3 x A": C.REPO / "tracks/structural-sufficiency/zinc_perf_protocol_v1/runs/p3_s0",
    "P3 x B": R2.RUNS2 / "xBssl_P3N_s0",
    "P3 scratch x B": R2.RUNS2 / "P3N_s0",  # defective first launch, kept as a replicate
    "T x A": R2.RUNS2 / "T_s0",
    "T x B": C.REPO / "tracks/structural-sufficiency/zinc_arch_discovery_v1/runs/T_s0",
    "M supervised": R2.run_dir("M", 0, "m"),
    "M joint": R2.run_dir("Maux", 0, "m"),
    "S supervised": R2.run_dir("S", 0, "s"),
    "S joint": R2.run_dir("Saux", 0, "s"),
    "M supervised s1": R2.run_dir("M", 1, "m"),
    "S supervised s1": R2.run_dir("S", 1, "s"),
    "M supervised (B)": R2.run_dir("M", 0, "mB"),
    "S supervised (B)": R2.run_dir("S", 0, "sB"),
    "M joint s1": R2.run_dir("Maux", 1, "m"),
    "S joint s1": R2.run_dir("Saux", 1, "s"),
}

#: P3 seed 0 best-valid as recorded by its own run.  ``ad_common`` stores
#: 0.07910549778419873 (taken from the frozen prediction array); the curve value
#: differs by 8.4e-7 because the per-epoch MAE is logged from float32.  All
#: deltas below are unchanged at 6 decimals.
REFERENCE = 0.0791046605258598
P3_BEST = 0.076224  # P3 x recipe B, seed 0: best valid cell measured this round


def load(name: str, path: Path) -> dict | None:
    f = path / "train.json"
    if not f.exists():
        return None
    info = C.read_json(f)
    curve = info["curve"]
    best = min(curve, key=lambda e: e["valid_mae"])
    entry = {
        "name": name,
        "path": str(path.relative_to(C.REPO)),
        "candidate": info.get("candidate", info.get("model", "P3")),
        "seed": info["seed"],
        "recipe_variant": info.get("recipe_variant"),
        "recipe": {k: info["recipe"].get(k) for k in
                   ("optimizer", "lr", "min_lr", "weight_decay", "warmup_steps",
                    "epochs", "batch_size", "steps_per_epoch", "updates")},
        "parameters": info["parameter_count"],
        "best_valid": float(best["valid_mae"]),
        "best_epoch": int(best["epoch"]),
        "final_valid": float(curve[-1]["valid_mae"]),
        "train_mae_at_best": float(best["online_train_mae"]),
        "train_mae_eval_best": info["fixed_objects"]["best"]["train_mae_eval_mode"],
        "train_mae_eval_final": info["fixed_objects"]["final"]["train_mae_eval_mode"],
        "gpu_hours": info.get("gpu_hours"),
        "wall_seconds": info["wall_seconds"],
        "ssl": (info.get("model_info") or {}).get("ssl", {}).get("source")
        if isinstance((info.get("model_info") or {}).get("ssl"), dict) else None,
        "delta_vs_p3": float(best["valid_mae"]) - REFERENCE,
        "epochs_curve": len(curve),
        "mean_epoch_seconds": float(np.mean([e["seconds"] for e in curve])),
        "hits_boundary": int(best["epoch"]) >= len(curve) - 1,
        "auxiliary": info.get("auxiliary"),
        "curve": [{"epoch": int(e["epoch"]), "valid_mae": float(e["valid_mae"]),
                   "online_train_mae": float(e["online_train_mae"]), "lr": float(e["lr"]),
                   "seconds": float(e["seconds"])} for e in curve],
    }
    return entry


def collect() -> dict:
    out = {}
    for name, path in RUNS.items():
        entry = load(name, path)
        if entry is not None:
            out[name] = entry
    return out


def fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def main_table(rows: dict) -> str:
    lines = [
        "| run | recipe | optim | epochs | params | best-valid | best ep | train@best(eval) | Δ vs P3 s0 | s/epoch | GPU-h |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["P3 x A", "P3 x B", "P3 scratch x B", "T x A", "T x B",
             "M supervised", "M joint", "S supervised", "S joint",
             "M supervised (B)", "S supervised (B)",
             "M supervised s1", "S supervised s1"]
    for name in order:
        r = rows.get(name)
        if r is None:
            continue
        recipe = r["recipe_variant"] or ("A" if "p3_s0" in r["path"] else "B")
        lines.append(
            f"| {name} | {recipe} | {r['recipe']['optimizer']} | {r['epochs_curve']} | "
            f"{r['parameters']:,} | {fmt(r['best_valid'])} | {r['best_epoch']} | "
            f"{fmt(r['train_mae_eval_best'])} | {r['delta_vs_p3']:+.6f} | "
            f"{r['mean_epoch_seconds']:.2f} | {fmt(r['gpu_hours'], 3)} |"
        )
    return "\n".join(lines)


def seed_table(rows: dict) -> str:
    """Paired seed comparison: same valid split, best-valid per seed."""
    groups = {
        "P3 (historical)": [("P3 x A", C.REPO / "tracks/structural-sufficiency/zinc_perf_protocol_v1/runs/p3_s1")],
        "M supervised": [("M supervised", None), ("M supervised s1", None)],
        "S supervised": [("S supervised", None), ("S supervised s1", None)],
    }
    p3s1 = load("P3 s1", groups["P3 (historical)"][0][1])
    if p3s1:
        rows["P3 x A (s1)"] = p3s1
    lines = ["| family | seed 0 | seed 1 | seed mean | spread | Δ mean vs P3 mean |",
             "|---|---:|---:|---:|---:|---:|"]
    ref_mean = 0.5 * (rows["P3 x A"]["best_valid"] + (p3s1["best_valid"] if p3s1 else rows["P3 x A"]["best_valid"]))
    vals = {"P3": [rows["P3 x A"]["best_valid"], p3s1["best_valid"] if p3s1 else None]}
    for name in ("M supervised", "S supervised"):
        s1 = name + " s1"
        vals[name] = [rows[name]["best_valid"], rows[s1]["best_valid"] if s1 in rows else None]
    vals["T (rank16)"] = [rows["T x A"]["best_valid"], None]
    for name, (a, b) in vals.items():
        if b is None:
            lines.append(f"| {name} | {a:.6f} | — | — | — | — |")
            continue
        mean = 0.5 * (a + b)
        lines.append(f"| {name} | {a:.6f} | {b:.6f} | **{mean:.6f}** | {abs(a - b):.6f} | "
                     f"{mean - ref_mean:+.6f} |")
    return "\n".join(lines)


def cross_table(rows: dict) -> str:
    lines = ["| | recipe A (Adam, 360, no warmup) | recipe B (AdamW, 600, warmup 500) |",
             "|---|---|---|"]
    cells = {("P3", "A"): "P3 x A", ("P3", "B"): "P3 x B",
             ("T", "A"): "T x A", ("T", "B"): "T x B"}
    for arch in ("P3", "T"):
        vals = []
        for rec in ("A", "B"):
            r = rows.get(cells[(arch, rec)])
            if r is None:
                vals.append("—")
            else:
                vals.append(f"{r['best_valid']:.6f} @{r['best_epoch']} ({r['gpu_hours']:.3f} h)")
        lines.append(f"| **{arch}** | {vals[0]} | {vals[1]} |")
    return "\n".join(lines)


def plot(rows: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [
        ("recipe cross", [("P3 x A", "tab:green", "-"), ("P3 x B", "tab:green", "--"),
                          ("T x A", "tab:red", "-"), ("T x B", "tab:red", "--")]),
        ("new trunks", [("M supervised", "tab:blue", "-"), ("M joint", "tab:blue", "--"),
                        ("S supervised", "tab:purple", "-"), ("S joint", "tab:purple", "--"),
                        ("M supervised (B)", "tab:blue", ":"), ("S supervised (B)", "tab:purple", ":"),
                        ("P3 x B", "tab:green", ":")]),
        ("seed-1 confirmations", [("M supervised", "tab:blue", ":"),
                                  ("M supervised s1", "tab:blue", "-"),
                                  ("S supervised", "tab:purple", ":"),
                                  ("S supervised s1", "tab:purple", "-"),
                                  ("P3 x A", "tab:green", ":")]),
    ]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 3.8), squeeze=False)
    for ax, (title, specs) in zip(axes[0], groups):
        plotted = False
        for name, color, style in specs:
            r = rows.get(name)
            if r is None:
                continue
            ep = [e["epoch"] for e in r["curve"]]
            va = [e["valid_mae"] for e in r["curve"]]
            ax.plot(ep, va, style, color=color, lw=1.4, label=f"{name} ({r['best_valid']:.4f})")
            ax.plot([r["best_epoch"]], [r["best_valid"]], "o", color=color, ms=4)
            plotted = True
        ax.axhline(REFERENCE, color="0.5", lw=0.9, ls="-.", label=f"P3 ref (A,s0) {REFERENCE:.4f}")
        ax.axhline(P3_BEST, color="0.35", lw=0.9, ls="--", label=f"P3 best cell (B,s0) {P3_BEST:.4f}")
        ax.axhline(0.075, color="0.7", lw=0.8, ls=":", label="0.075 target")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("official valid MAE")
        ax.set_ylim(0.05, 0.40)
        if plotted:
            ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    ANALYSIS2.mkdir(parents=True, exist_ok=True)
    fig.savefig(ANALYSIS2 / "curves.png", dpi=150)
    C.log(f"wrote {ANALYSIS2 / 'curves.png'}")


#: A100 time billed to this task (the historical P3 seed-0 run is not rebilled).
TIMING_AND_FAILED_HOURS = 0.0037 + 0.0044 + 0.003


#: Cells whose training was already billed in an earlier round (reused, not rerun).
NOT_BILLED_THIS_ROUND = ("P3 x A", "P3 x A (s1)", "T x B")


def billed_hours(rows: dict) -> dict:
    runs = {name: float(r["gpu_hours"]) for name, r in rows.items()
            if r.get("gpu_hours") and name not in NOT_BILLED_THIS_ROUND}
    runs["timing jobs and failed launches"] = TIMING_AND_FAILED_HOURS
    return runs


def decision(rows: dict) -> dict:
    """Pre-declared read-outs, computed from the runs (no test, no selection)."""
    def g(name, key="best_valid"):
        return rows[name][key]

    billed = billed_hours(rows)
    spent = float(sum(billed.values()))
    test_path = ANALYSIS2 / "test_terminal.json"
    test_block = None
    if test_path.exists():
        tr = C.read_json(test_path)["runs"]
        test_block = {
            "read": "post-selection, after the round decision; descriptive only",
            "runs": {k: v["test_mae"] for k, v in tr.items()},
            "cells": {
                name: {
                    "test_mae": tr[key]["test_mae"],
                    "best_valid": tr[key]["best_valid"],
                    "test_minus_valid": tr[key]["test_mae"] - tr[key]["best_valid"],
                }
                for name, key in (
                    ("P3_A_s0", "P3_A_s0"), ("P3_A_s1", "P3_A_s1"), ("P3_B_s0", "P3_B_s0"),
                    ("T_A_s0", "T_A_s0"), ("T_B_s0", "T_B_s0"),
                    ("M_sup_s0", "M_sup_s0"), ("M_sup_s1", "M_sup_s1"),
                    ("S_sup_s0", "S_sup_s0"), ("S_sup_s1", "S_sup_s1"),
                ) if key in tr
            },
            "best_single_test": min(
                ((v["test_mae"], k) for k, v in tr.items() if not k.startswith("P3_scratch")),
            ),
            "changes_the_ranking": False,
        }
    _out = {
        "reference_p3_seed0_valid": REFERENCE,
        "phase_i": {
            "p3_A": g("P3 x A"), "p3_B": g("P3 x B"),
            "p3_scratch_B_not_a_cell": g("P3 scratch x B") if "P3 scratch x B" in rows else None,
            "t_A": g("T x A"), "t_B": g("T x B"),
            "p3_B_minus_p3_A": g("P3 x B") - g("P3 x A"),
            "t_A_minus_t_B": g("T x A") - g("T x B"),
            "gap_p3_minus_t_under_A": g("P3 x A") - g("T x A"),
            "gap_p3_minus_t_under_B": g("P3 x B") - g("T x B"),
            "p3_prefers_B_by": g("P3 x A") - g("P3 x B"),
            "t_prefers_A_by": g("T x B") - g("T x A"),
            "initialisation_effect_at_600_epochs_p3": g("P3 scratch x B") - g("P3 x B")
            if "P3 scratch x B" in rows else None,
        },
        "phase_ii_seed0": {
            "M_sup_C": g("M supervised"), "M_joint_C": g("M joint"),
            "S_sup_C": g("S supervised"), "S_joint_C": g("S joint"),
            "joint_minus_supervised_M": g("M joint") - g("M supervised"),
            "joint_minus_supervised_S": g("S joint") - g("S supervised"),
            "M_sup_B_post_hoc": g("M supervised (B)") if "M supervised (B)" in rows else None,
            "S_sup_B_post_hoc": g("S supervised (B)") if "S supervised (B)" in rows else None,
            "M_B_minus_M_C_post_hoc": g("M supervised (B)") - g("M supervised")
            if "M supervised (B)" in rows else None,
            "S_B_minus_S_C_post_hoc": g("S supervised (B)") - g("S supervised")
            if "S supervised (B)" in rows else None,
        },
        "phase_ii_seed1": {
            "M_s1": g("M supervised s1") if "M supervised s1" in rows else None,
            "S_s1": g("S supervised s1") if "S supervised s1" in rows else None,
        },
        "verdict": {
            "new_trunk_beats_p3_seed0_reference": bool(min(
                [v for v in (
                    g("M supervised"), g("S supervised"), g("M joint"), g("S joint"),
                    g("M supervised s1") if "M supervised s1" in rows else None,
                    g("S supervised s1") if "S supervised s1" in rows else None,
                    g("M supervised (B)") if "M supervised (B)" in rows else None,
                    g("S supervised (B)") if "S supervised (B)" in rows else None,
                ) if v is not None]) < REFERENCE),
            "new_trunk_beats_p3_best_cell": bool(min(
                [v for v in (
                    g("M supervised"), g("S supervised"), g("M joint"), g("S joint"),
                    g("M supervised s1") if "M supervised s1" in rows else None,
                    g("S supervised s1") if "S supervised s1" in rows else None,
                    g("M supervised (B)") if "M supervised (B)" in rows else None,
                    g("S supervised (B)") if "S supervised (B)" in rows else None,
                ) if v is not None]) < g("P3 x B")),
            "closest_new_model": "M supervised seed 1",
            "closest_new_valid": min(
                [v for v in (
                    g("M supervised"), g("S supervised"), g("M joint"), g("S joint"),
                    g("M supervised s1") if "M supervised s1" in rows else None,
                    g("S supervised s1") if "S supervised s1" in rows else None,
                    g("M supervised (B)") if "M supervised (B)" in rows else None,
                    g("S supervised (B)") if "S supervised (B)" in rows else None,
                ) if v is not None]),
            "development_target_0.075_reached": bool(min(
                [v for v in (
                    g("M supervised"), g("S supervised"), g("M joint"), g("S joint"),
                    g("M supervised s1") if "M supervised s1" in rows else None,
                    g("S supervised s1") if "S supervised s1" in rows else None,
                    g("M supervised (B)") if "M supervised (B)" in rows else None,
                    g("S supervised (B)") if "S supervised (B)" in rows else None,
                ) if v is not None]) <= 0.075),
            "stop": True,
            "stop_reason": "round questions answered; no remaining purchase can change "
            "the ranking at the pre-declared effect sizes within 1.76 h",
        },
        "budget": {
            "cap_a100_hours_this_task": 5.65,
            "spent_a100_hours": spent,
            "unspent_a100_hours": 5.65 - spent,
            "p3_seed0_reference_not_rebilled": True,
        },
        "note": (
            "Best-valid on the official valid split only; nothing was selected on "
            "test. The 1,000-molecule test split was read once after the decision "
            "was written (section 5b), descriptively; it did not change any "
            "selection. Deltas are single- or two-seed differences, not "
            "significance tests."
        ),
    }
    if test_block is not None:
        _out["test_terminal_after_decision"] = test_block
    return _out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--print", action="store_true")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)
    rows = collect()
    ANALYSIS2.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS2 / "main_table.json", rows)
    md = ["# zinc-recipe-trunk-v2 — measured runs", "",
          f"P3 seed-0 reference (same valid split): {REFERENCE:.6f}", "",
          "## 2x2 architecture x recipe", "", cross_table(rows), "",
          "## Seed comparison (best-valid, same valid split)", "", seed_table(rows), "",
          "## All runs", "", main_table(rows), ""]
    (ANALYSIS2 / "main_table.md").write_text("\n".join(md), encoding="utf-8")
    dec = decision(rows)
    dec["budget"]["ledger"] = billed_hours(rows)
    C.write_json(ANALYSIS2 / "round_decision.json", dec)
    if args.plot:
        plot(rows)
    if args.print:
        print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
