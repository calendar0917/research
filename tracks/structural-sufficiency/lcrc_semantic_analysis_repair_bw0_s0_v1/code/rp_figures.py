"""Two figures that actually change how the results read (no filler).

``fig1_kernel_grading.png``
    the corrected gradient-kernel cosine by number of shared original
    ``mid/b1/b2`` components, for the real codec (FS) and the permuted control
    (FP), at initialisation and at the own best-Valid checkpoint.  This is the
    one picture the numbers alone make hard to see: the grading exists *at init*
    and training *lowers* all three levels without destroying the order.

``fig2_performance_panel.png``
    the two selection rules side by side (old 50-point grid vs every-epoch
    best-Valid), Val and Test, for the arms that carry the round's decision.
    It makes visible that the same arm wins under both rules and that the
    grid/best Test values disagree for FS, which is why both tables are printed.

Both read only ``analysis/*.json``; they never recompute or select anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import rp_common as RP

FIGDIR = RP.ANALYSIS / "figures"

PANEL_ARMS = ("B_R", "B_W0", "B_Rwd", "LS_R", "LS_W0", "LS_Rwd", "FS_R", "FS_W0", "FS_Rwd", "FP_R")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_kernel() -> dict[str, Any]:
    plt = _plt()
    payload = RP.read_json(RP.ANALYSIS / "kernel_init_best_corrected.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    styles = {("FS", "init"): dict(color="#7f7f7f", ls="--", marker="o"),
              ("FS", "best"): dict(color="#1f77b4", ls="-", marker="o"),
              ("FP", "init"): dict(color="#c7c7c7", ls="--", marker="s"),
              ("FP", "best"): dict(color="#d62728", ls="-", marker="s")}
    xs = np.arange(3)
    for ax, arm, label in ((axes[0], "FS_R", "real codec (FS)"), (axes[1], "FP_R", "permuted control (FP)")):
        fam = "FS" if arm.startswith("FS") else "FP"
        for tag in ("init", "best"):
            blk = payload[arm][tag]["kernel_by_overlap"]
            ys = [blk[str(k)]["mean"] for k in range(3)]
            ax.plot(xs, ys, label=f"{tag}" + (f" (ep {payload[arm][tag]['best_epoch']})"
                                              if tag == "best" and "best_epoch" in payload[arm][tag] else ""),
                    **styles[(fam, tag)])
        ax.set_xticks(xs)
        ax.set_xticklabels(["0 shared", "1 shared", "2 shared"])
        ax.set_xlabel("shared original composition components")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("mean cosine of per-code gradient directions")
    fig.suptitle("Corrected gradient kernel: grading is present at init, training lowers it", fontsize=10.5)
    fig.tight_layout()
    out = FIGDIR / "fig1_kernel_grading.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return {"path": RP.track_rel(out)}


def fig_performance() -> dict[str, Any]:
    plt = _plt()
    payload = RP.read_json(RP.ANALYSIS / "performance_tables.json")
    tables = payload["tables"]
    rows = []
    for arm in PANEL_ARMS:
        rec = tables.get(arm)
        if not rec:
            continue
        a = rec.get("tableA_grid") or {}
        b = rec.get("tableB_best") or {}
        rows.append({
            "arm": arm,
            "a_val": a.get("valid_rmse"), "a_test": a.get("test_rmse"),
            "b_val": (b.get("best") or {}).get("valid", {}).get("rmse"),
            "b_test": (b.get("best") or {}).get("test", {}).get("rmse"),
        })
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), sharex=True, sharey=True)
    colors = {"B": "#333333", "LS": "#1f77b4", "FS": "#2ca02c", "FP": "#d62728"}
    for ax, tag, title in ((axes[0], "a", "A: old 50-point update grid"),
                           (axes[1], "b", "B: every-epoch best-Valid")):
        for r in rows:
            x, y = r[f"{tag}_val"], r[f"{tag}_test"]
            if x is None or y is None:
                continue
            fam = r["arm"].split("_")[0]
            ax.scatter(x, y, s=46, color=colors.get(fam, "#888888"),
                       marker={"B": "X", "LS": "o", "FS": "^", "FP": "v"}.get(fam, "o"))
            ax.annotate(r["arm"], (x, y), fontsize=6.8, xytext=(4, 3), textcoords="offset points")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Valid RMSE (selection split)")
    axes[0].set_ylabel("Test RMSE")
    axes[0].text(0.02, 0.02, "lower-left is better on both splits", transform=axes[0].transAxes, fontsize=7)
    fig.suptitle("Both selection rules printed in full; no cherry-picking across tables", fontsize=10.5)
    fig.tight_layout()
    out = FIGDIR / "fig2_performance_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return {"path": RP.track_rel(out), "n_arms": len(rows)}


def run() -> dict[str, Any]:
    return {"figures": [fig_kernel(), fig_performance()]}


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=1))


__all__ = ["run", "fig_kernel", "fig_performance"]
