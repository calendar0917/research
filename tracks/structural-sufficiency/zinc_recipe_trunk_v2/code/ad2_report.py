"""Aggregate tables + real curves for zinc-recipe-trunk-v2.

Reads the run directories written by ``ad2_runner.py`` (each contains
``train.json`` and ``curve.csv``) and produces the round tables.  Nothing here
evaluates a model or touches the test split.
"""

from __future__ import annotations

import argparse
import json
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
    "P3 x B": R2.run_dir("P3N", 0, "xB"),
    "T x A": R2.run_dir("T", 0, "xA"),
    "T x B": C.REPO / "tracks/structural-sufficiency/zinc_arch_discovery_v1/runs/T_s0",
    "M supervised": R2.run_dir("M", 0, "m"),
    "M joint": R2.run_dir("Maux", 0, "m"),
    "S supervised": R2.run_dir("S", 0, "s"),
    "S joint": R2.run_dir("Saux", 0, "s"),
    "M supervised s1": R2.run_dir("M", 1, "m"),
    "S supervised s1": R2.run_dir("S", 1, "s"),
    "M joint s1": R2.run_dir("Maux", 1, "m"),
    "S joint s1": R2.run_dir("Saux", 1, "s"),
}

REFERENCE = 0.07910549778419873  # P3 seed 0, same valid split


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
        "candidate": info["candidate"],
        "seed": info["seed"],
        "recipe_variant": info.get("recipe_variant"),
        "recipe": {k: info["recipe"][k] for k in
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
        "| run | recipe | optim | epochs | params | best-valid | best ep | train@best(eval) | Δ vs P3 s0 | GPU-h |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["P3 x A", "P3 x B", "T x A", "T x B",
             "M supervised", "M joint", "S supervised", "S joint",
             "M supervised s1", "S supervised s1", "M joint s1", "S joint s1"]
    for name in order:
        r = rows.get(name)
        if r is None:
            continue
        recipe = r["recipe_variant"] or ("A" if "p3_s0" in r["path"] else "B")
        lines.append(
            f"| {name} | {recipe} | {r['recipe']['optimizer']} | {r['epochs_curve']} | "
            f"{r['parameters']:,} | {fmt(r['best_valid'])} | {r['best_epoch']} | "
            f"{fmt(r['train_mae_eval_best'])} | {r['delta_vs_p3']:+.6f} | "
            f"{fmt(r['gpu_hours'], 3)} |"
        )
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
                        ("P3 x A", "tab:green", ":")]),
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
        ax.axhline(REFERENCE, color="0.5", lw=0.8, ls="-.")
        ax.axhline(0.075, color="0.7", lw=0.8, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("official valid MAE")
        ax.set_ylim(0.06, 0.30)
        if plotted:
            ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    ANALYSIS2.mkdir(parents=True, exist_ok=True)
    fig.savefig(ANALYSIS2 / "curves.png", dpi=150)
    C.log(f"wrote {ANALYSIS2 / 'curves.png'}")


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
          "## All runs", "", main_table(rows), ""]
    (ANALYSIS2 / "main_table.md").write_text("\n".join(md), encoding="utf-8")
    if args.plot:
        plot(rows)
    if args.print:
        print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
