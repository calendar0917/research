"""Generate the three decision figures from frozen result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sr_common as C

OUT = C.ANALYSIS_DIR / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def route_a() -> None:
    rows = read_csv(C.ANALYSIS_DIR / "main_table.csv")
    index = {(r["arm"], r["split"], int(r["seed"])): float(r["dev_rmse"]) for r in rows}
    arms = ["R2_REF", "E_RAW", "E_MOB", "F_RAW", "F_MOB", "E_MOB_BIG", "F_MOB_BIG"]
    labels = ["R2", "E_RAW", "E_MOB", "F_RAW", "F_MOB", "E_BIG", "F_BIG"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    for ax, split in zip(axes, ["D0", "D0", "D1"]):
        pass
    panels = [("D0 seed 0", "D0", 0), ("D0 seed 1", "D0", 1), ("D1 seed 0", "D1", 0)]
    for ax, (title, split, seed) in zip(axes, panels):
        vals = [index.get((a, split, seed)) for a in arms]
        xs = np.arange(len(arms))
        plot_vals = [v if v is not None else np.nan for v in vals]
        colors = ["#333333"] + ["#4c78a8"] * 2 + ["#e45756"] * 2 + ["#72b7b2"] * 2
        ax.bar(xs, plot_vals, color=colors, width=0.72)
        for x, v in zip(xs, vals):
            if v is not None:
                ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs, labels, rotation=55, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0.45, 0.80)
    axes[0].set_ylabel("Dev RMSE (lower is better)")
    fig.suptitle("Route A: factor arms versus R2 (D0 two seeds; D1 locked base)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "route_a_rmse.png", dpi=180)
    plt.close(fig)


def route_b() -> None:
    rows = read_csv(C.ANALYSIS_DIR / "main_table.csv")
    index = {(r["arm"], r["split"], int(r["seed"])): float(r["dev_rmse"]) for r in rows}
    arms = ["G_FULL5", "G_EGO2_5", "R2_REF"]
    labels = ["G_FULL5 (whole molecule)", "G_EGO2_5 (radius-2 ego)", "R2 sentence"]
    groups = [("D0/s0", "D0", 0), ("D0/s1", "D0", 1), ("D1/s0", "D1", 0)]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.24
    xs = np.arange(len(groups))
    colors = ["#4c78a8", "#f58518", "#333333"]
    for j, (arm, label, color) in enumerate(zip(arms, labels, colors)):
        vals = [index.get((arm, s, seed)) for _, s, seed in groups]
        ax.bar(xs + (j - 1) * width, [v if v is not None else np.nan for v in vals],
               width=width, label=label, color=color)
        for x, v in zip(xs + (j - 1) * width, vals):
            if v is not None:
                ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xs, [g[0] for g in groups])
    ax.set_ylabel("Dev RMSE")
    ax.set_ylim(0.45, 0.60)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Route B: message range FULL vs radius-2 EGO2", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "route_b_scope.png", dpi=180)
    plt.close(fig)


def route_c() -> None:
    rows = read_csv(C.ANALYSIS_DIR / "response_metrics.csv")
    index = {(r["split"], r["scope"], r["model"]): float(r["component_equal_mae"])
             for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
    arms = ["ZERO", "EXACT_MEAN", "K_TRANSFORM", "K_CONDITIONAL_R2", "K_CONTEXT_ID"]
    labels = ["ZERO", "EXACT lookup", "K_TRANSFORM", "K_COND_R2", "K_CONTEXT_ID"]
    for ax, split in zip(axes, ["D0", "D1"]):
        vals = [index.get((split, "Dev-Dev", arm), np.nan) for arm in arms]
        xs = np.arange(len(arms))
        ax.bar(xs, vals, color=["#999999", "#999999", "#4c78a8", "#e45756", "#72b7b2"])
        for x, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(xs, labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(split)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("component-equal MAE (pairs)")
    fig.suptitle("Route C: frozen edit-response kernels, Dev-Dev", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "route_c_pairs.png", dpi=180)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    route_a()
    route_b()
    route_c()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())