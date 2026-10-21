"""Three descriptive figures for the audit report (no model training)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import audit_common as C  # noqa: E402

ANALYSIS = C.ANALYSIS
FIG = ANALYSIS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"


def load_rows():
    return list(csv.DictReader(open(ANALYSIS / "per_molecule.csv", encoding="utf-8")))


def arr(rows, key):
    return np.asarray([float(r[key]) for r in rows])


def fig_error_shape(rows):
    ae = arr(rows, "abs_error_P3B_s0")
    s = np.sort(ae)[::-1]
    cum = np.cumsum(s) / s.sum()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    ax.plot(np.arange(1, len(s) + 1), s, color=BLUE, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("molecules sorted by |error| (descending)")
    ax.set_ylabel("|error| (log scale)")
    ax.set_title("P3B seed 0: heavy-tailed valid errors")
    ax.axhline(0.0762, color=GREY, ls="--", lw=1)
    ax.text(600, 0.080, "MAE 0.0762", color=GREY, fontsize=8)

    ax = axes[1]
    frac = np.arange(1, len(s) + 1) / len(s)
    ax.plot(frac * 100, cum * 100, color=ORANGE, lw=1.6)
    ax.plot([0, 100], [0, 100], color=GREY, ls=":", lw=1)
    for k, label in [(5, "top 5%"), (10, "top 10%"), (25, "top 25%")]:
        idx = int(round(len(s) * k / 100)) - 1
        ax.scatter([frac[idx] * 100], [cum[idx] * 100], color=RED, s=18, zorder=3)
        ax.annotate(f"{label}: {cum[idx]*100:.0f}%", (frac[idx] * 100, cum[idx] * 100),
                    xytext=(5, -12), textcoords="offset points", fontsize=8, color=RED)
    ax.set_xlabel("share of molecules (%)")
    ax.set_ylabel("share of total absolute error (%)")
    ax.set_title("Error concentration (Lorenz curve)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_error_shape.png", dpi=180)
    plt.close(fig)


def fig_models():
    m = json.load(open(ANALYSIS / "model_metrics.json"))
    names = list(m)
    maes = [m[n]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    colors = [BLUE if n.startswith("P3") else ORANGE if n.startswith("J") else GREEN if n.startswith("Q") else GREY for n in names]
    bars = ax.bar(names, maes, color=colors, width=0.65)
    ax.axhline(maes[0], color=BLUE, ls="--", lw=1)
    ax.axhline(0.070913, color=RED, ls=":", lw=1.2)
    ax.text(len(names) - 0.4, 0.07099, "7-model avg 0.0709", color=RED, ha="right", fontsize=8)
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0007, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_ylim(0.06, 0.092)
    ax.set_ylabel("valid MAE (raw y)")
    ax.set_title("Fixed best checkpoints on official valid-1000 (already exposed)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_models.png", dpi=180)
    plt.close(fig)


def fig_subgroups(rows):
    ae = arr(rows, "abs_error_P3B_s0")
    j = arr(rows, "j_unseen_mass")
    r2 = arr(rows, "r2_unseen_atom_frac")
    rings = arr(rows, "n_rings")
    y = arr(rows, "y_raw")
    groups = [
        ("all valid", np.ones(len(ae), dtype=bool)),
        ("radius-2 unseen\n(any)", r2 > 0),
        ("path unseen\n(any)", j > 0),
        ("rings >= 4", rings >= 4),
        ("y <= -3", y <= -3),
    ]
    labels = [g[0] for g in groups]
    masks = [g[1] for g in groups]
    maes = [ae[m].mean() for m in masks]
    shares = [ae[m].sum() / ae.sum() * 100 for m in masks]
    ns = [int(m.sum()) for m in masks]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    colors = [GREY] + [RED if "radius" in l else ORANGE if "path" in l else GREEN if "rings" in l else BLUE for l in labels[1:]]
    bars = ax.bar(labels, maes, color=colors, width=0.65)
    for b, v, n in zip(bars, maes, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}\nn={n}", ha="center", fontsize=7.5)
    ax.set_ylabel("P3B MAE")
    ax.set_title("Hard subgroups have 2-3x higher MAE")
    ax.tick_params(axis="x", labelsize=7.5)
    ax = axes[1]
    bars = ax.bar(labels, shares, color=colors, width=0.65)
    for b, v, n in zip(bars, shares, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("share of total absolute error (%)")
    ax.set_title("The same subgroups carry the error mass")
    ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_subgroups.png", dpi=180)
    plt.close(fig)


def fig_error_corr(rows):
    names = ["P3B_s0", "J_s0", "Q_s0", "P3A_s0", "P3A_s1", "M_s0", "M_s1"]
    err = np.stack([arr(rows, f"signed_error_{n}") for n in names], axis=1)
    corr = np.corrcoef(err, rowvar=False)
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i,j]) > 0.75 else "black")
    ax.set_title("Signed-error correlation across 7 fixed checkpoints")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_error_corr.png", dpi=180)
    plt.close(fig)


def main():
    rows = load_rows()
    fig_error_shape(rows)
    fig_models()
    fig_subgroups(rows)
    fig_error_corr(rows)
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
