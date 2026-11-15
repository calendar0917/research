"""Generate the five required figures from pulled artifacts (CSV/JSON only).

Kept deliberately light (no model tensors) so it never risks local RAM.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import se_common as C

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = C.ANALYSIS / "results"
FIG = C.ANALYSIS / "figures"


def _curve(arm):
    p = RESULTS / arm / "curve.csv"
    if not p.is_file():
        return None
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _grid_pts(arm):
    p = RESULTS / arm / "grid.json"
    if not p.is_file():
        return None
    return {q["j"]: q for q in json.loads(p.read_text())["points"]}


def fig1_curves():
    arms = {"LS": "LS_N100_lipo_R", "FS": "FS_N100_lipo_R", "LP": "LP_N100_lipo_R", "FP": "FP_N100_lipo_R"}
    colors = {"LS": "tab:blue", "FS": "tab:red", "LP": "tab:cyan", "FP": "tab:orange"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, arm in arms.items():
        cur = _curve(arm)
        if not cur:
            continue
        ep = [int(r["epoch"]) for r in cur]
        for ax, sp in zip(axes, ("train", "valid", "test")):
            ax.plot(ep, [float(r[f"{sp}_rmse"]) for r in cur], label=name, lw=0.9, color=colors[name])
            ax.set_title(f"{sp} RMSE"); ax.set_xlabel("epoch"); ax.set_ylabel("RMSE (logD)")
            ax.set_ylim(0.15, 1.6)
    # mark best epochs on test
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig1  LS/FS/LP/FP learning curves (lipo N100, R)")
    fig.tight_layout()
    out = FIG / "fig1_curves.png"; fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig2_paired_did():
    # grid-selected DiD across recipes
    recipes = {
        "lipo R": ("LS_N100_lipo_R", "FS_N100_lipo_R", "LP_N100_lipo_R", "FP_N100_lipo_R"),
        "lipo Rwd": ("LS_N100_lipo_Rwd", "FS_N100_lipo_Rwd", "LP_N100_lipo_Rwd", "FP_N100_lipo_Rwd"),
        "esol R": ("LS_N100_esol_R", "FS_N100_esol_R", "LP_N100_esol_R", "FP_N100_esol_R"),
    }
    labels, dval, dtest = [], [], []
    for name, (LS, FS, LP, FP) in recipes.items():
        g = {k: _grid_pts(a) for k, a in zip("LSFP", (LS, FS, LP, FP))}  # type: ignore
        if any(v is None for v in g.values()):
            continue
        for split, key in (("valid", "valid_rmse"), ("test", "test_rmse")):
            dsem = g["L"][g["L"]["__dummy__"] if False else 0] if False else None  # placeholder
    # simpler: read selection from json
    def sel(arm):
        p = RESULTS / arm / "grid.json"
        s = json.loads(p.read_text())["selection"]
        return s["valid_rmse"], s["test_rmse"]
    for name, (LS, FS, LP, FP) in recipes.items():
        lv, lt = sel(LS); fv, ft = sel(FS); pv, pt = sel(LP); qv, qt = sel(FP)
        dval.append(((lv - fv) - (pv - qv)) * 100)
        dtest.append(((lt - ft) - (pt - qt)) * 100)
        labels.append(name)
    x = range(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - w / 2 for i in x], dval, w, label="Valid DiD", color="tab:blue")
    ax.bar([i + w / 2 for i in x], dtest, w, label="Test DiD", color="tab:red")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("DiD_semantics  (LS-FS)-(LP-FP)  [RMSE ×100]")
    ax.set_title("Fig2  Semantic-specificity DiD (grid-selected, Val-only)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig2_paired_did.png"; fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig3_budget():
    groups = {"N25": ("LS_N25_lipo_R", "FS_N25_lipo_R"),
              "N50": ("LS_N50_lipo_R", "FS_N50_lipo_R"),
              "N100": ("LS_N100_lipo_R", "FS_N100_lipo_R")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, (ls, fs) in groups.items():
        for arm, lbl, style in ((ls, f"{name} LS", "-"), (fs, f"{name} FS", "--")):
            g = _grid_pts(arm)
            if not g:
                continue
            js = sorted(g)
            xs = [g[j]["update"] for j in js]
            for ax, key in zip(axes, ("valid_rmse", "test_rmse")):
                ax.plot(xs, [g[j][key] for j in js], style, lw=1, label=lbl)
    for ax, key in zip(axes, ("valid", "test")):
        ax.set_title(f"{key} RMSE vs updates"); ax.set_xlabel("updates"); ax.set_ylabel("RMSE")
        ax.legend(fontsize=6)
    fig.suptitle("Fig3  Fixed-update R50 grid: N25 / N50 / N100 (dashed = FS)")
    fig.tight_layout()
    out = FIG / "fig3_budget.png"; fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig4_support_kernel():
    kf = json.loads((C.ANALYSIS / "mech_kernel_fs.json").read_text()) if (C.ANALYSIS / "mech_kernel_fs.json").is_file() else None
    kp = json.loads((C.ANALYSIS / "mech_kernel_fp.json").read_text()) if (C.ANALYSIS / "mech_kernel_fp.json").is_file() else None
    sup = json.loads((C.ANALYSIS / "mech_support_fs.json").read_text()) if (C.ANALYSIS / "mech_support_fs.json").is_file() else None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    if sup:
        for sp, col in (("valid", "tab:blue"), ("test", "tab:orange")):
            cc = sup[sp]["bin_counts"]
            ax.bar([i + (0.2 if sp == "valid" else -0.2) for i in range(3)], cc, width=0.35, color=col, label=sp)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["low", "mid", "high"])
        ax.set_title("joint-code support bins (Train-LOO terciles)")
        ax.set_ylabel("#molecules"); ax.legend(fontsize=8)
    ax2 = axes[1]
    if kf and kp:
        ov = sorted(kf["kernel_by_overlap"], key=int)
        fs = [kf["kernel_by_overlap"][o]["mean"] or 0 for o in ov]
        fp = [kp["kernel_by_overlap"][o]["mean"] or 0 for o in ov]
        x = range(len(ov)); w = 0.35
        ax2.bar([i - w / 2 for i in x], fs, w, label="FS", color="tab:green")
        ax2.bar([i + w / 2 for i in x], fp, w, label="FP (permuted)", color="tab:gray")
        ax2.set_xticks(list(x)); ax2.set_xticklabels([f"{o} shared" for o in ov])
        ax2.set_ylabel("mean cosine K"); ax2.set_title("gradient coupling by composition overlap")
        ax2.legend(fontsize=8)
    fig.suptitle("Fig4  real code support and cross-code gradient kernel")
    fig.tight_layout()
    out = FIG / "fig4_support_kernel.png"; fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig5_cost_edit():
    cost = {}
    for arm in ("B_N100_lipo_Rwd", "LS_N100_lipo_R", "FS_N100_lipo_R", "FS_N100_lipo_Rwd",
                "FS_N25_lipo_R", "FS_N100_esol_R"):
        p = RESULTS / arm / "metrics.json"
        if p.is_file():
            m = json.loads(p.read_text())
            cost[arm] = m["timing"].get("gpu_hours_charged") or 0.0
    edit = json.loads((C.ANALYSIS / "mech_edit.json").read_text()) if (C.ANALYSIS / "mech_edit.json").is_file() else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    names = list(cost); vals = [cost[n] for n in names]
    ax.barh(range(len(names)), vals, color="tab:purple")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("GPU hours / run"); ax.set_title("training cost")
    ax2 = axes[1]
    if edit:
        mods = edit["models"]
        labels = list(mods)
        maes = [mods[m]["ALL"]["edit_mae"] for m in labels]
        ax2.bar(range(len(labels)), maes, color="tab:brown")
        ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels)
        ax2.set_ylabel("edit MAE |Δpred-Δy|"); ax2.set_title("real edit response (823 pairs)")
        for i, v in enumerate(maes):
            ax2.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)
    fig.suptitle("Fig5  performance-cost and real edit response")
    fig.tight_layout()
    out = FIG / "fig5_cost_edit.png"; fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    outs = {}
    for fn in (fig1_curves, fig2_paired_did, fig3_budget, fig4_support_kernel, fig5_cost_edit):
        try:
            outs[fn.__name__] = C.track_rel(fn())
        except Exception as exc:  # noqa: BLE001
            outs[fn.__name__] = f"ERROR: {type(exc).__name__}: {exc}"
    print(json.dumps(outs, indent=1))


if __name__ == "__main__":
    main()
