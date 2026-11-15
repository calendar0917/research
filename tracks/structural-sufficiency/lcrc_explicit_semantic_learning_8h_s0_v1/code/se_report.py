"""Figures and the Chinese REPORT for the semantic-sharing round."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import se_common as C
import se_analysis as A

FIGURES = C.ANALYSIS / "figures"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def learning_curves(runs: Mapping[str, str]) -> str:
    plt = _plt()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, run in runs.items():
        if not A.has_run(run):
            continue
        rows = A.load_curve(run)
        ep = [int(r["epoch"]) for r in rows]
        for ax, split in zip(axes, ("train", "valid", "test")):
            ax.plot(ep, [float(r[f"{split}_rmse"]) for r in rows], label=name, lw=1)
            ax.set_title(f"{split} RMSE"); ax.set_xlabel("epoch"); ax.set_ylabel("RMSE (logD)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = FIGURES / "learning_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return C.track_rel(path)


def paired_did(fig_payload: Mapping[str, Any]) -> str:
    plt = _plt()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = []
    vals = []
    for key, block in fig_payload.get("comparisons", {}).items():
        for split in ("valid", "test"):
            if split in block:
                labels.append(f"{key}\n{split}")
                vals.append(block[split]["rel_improvement"] * 100)
    did = fig_payload.get("did", {}).get("semantics", {})
    for split in ("valid", "test"):
        if split in did:
            labels.append(f"DiD sem\n{split}")
            vals.append(did[split]["did"] * 100)
    colors = ["tab:blue"] * (len(vals) - len([s for s in ("valid", "test") if s in did])) + ["tab:red"] * len(
        [s for s in ("valid", "test") if s in did])
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("relative improvement vs matched control (%)")
    fig.tight_layout()
    path = FIGURES / "paired_did.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return C.track_rel(path)


def budget_curves(runs: Mapping[str, Mapping[str, str]]) -> str:
    plt = _plt()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for budget, arms in runs.items():
        for name, run in arms.items():
            if not A.has_run(run):
                continue
            try:
                g = A.load_grid(run)
            except FileNotFoundError:
                continue
            pts = g.get("points", [])
            xs = [p["update"] for p in pts]
            ys = [p["valid_rmse"] for p in pts]
            ax.plot(xs, ys, label=f"{name}-{budget}", lw=1)
    ax.set_xlabel("updates"); ax.set_ylabel("Valid RMSE (logD)"); ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIGURES / "budget_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return C.track_rel(path)


def support_and_kernel(support: Mapping[str, Any], kernel: Mapping[str, Any]) -> str:
    plt = _plt()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for split, color in (("valid", "tab:blue"), ("test", "tab:orange")):
        if split in support:
            ax.bar([0, 1, 2], support[split]["bin_counts"], alpha=0.5, color=color, label=split)
    ax.set_xlabel("low-support tercile bin"); ax.set_ylabel("#molecules"); ax.legend(fontsize=7)
    ax.set_title("joint-code low-support bins")
    ax2 = axes[1]
    byov = kernel.get("kernel_by_overlap", {}) if kernel else {}
    ov = sorted(byov.keys(), key=lambda k: int(k))
    ax2.bar(range(len(ov)), [byov[k]["mean"] or 0 for k in ov], color="tab:green")
    ax2.set_xticks(range(len(ov)))
    ax2.set_xticklabels([f"overlap {k}" for k in ov], fontsize=8)
    ax2.set_ylabel("mean cosine K"); ax2.set_title("cross-code gradient coupling")
    fig.tight_layout()
    path = FIGURES / "support_kernel.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return C.track_rel(path)


def cost_and_edit(cost: Mapping[str, Any], edit: Mapping[str, Any]) -> str:
    plt = _plt()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    names = list(cost.keys())
    vals = [cost[n] for n in names]
    ax.barh(range(len(names)), vals, color="tab:purple")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("GPU hours / run"); ax.set_title("performance-cost")
    ax2 = axes[1]
    models = (edit.get("models") or {})
    labels = list(models.keys())
    maes = [models[m].get("ALL", {}).get("edit_mae") or 0 for m in labels]
    ax2.bar(range(len(labels)), maes, color="tab:brown")
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("edit MAE |dpred-dy|"); ax2.set_title("real edit response")
    fig.tight_layout()
    path = FIGURES / "cost_edit.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return C.track_rel(path)


__all__ = [name for name in dir() if not name.startswith("_")]
