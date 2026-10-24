"""Plot the corrected batch-BCE training curves (CPU, analysis only)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parent.parent
J = EXP / "analysis" / "t3_synth_corrected.json"
OUT = EXP / "analysis" / "t3_bce_curves.png"
ARMS = ("C", "P", "S", "Srand", "SP")
COLORS = {"C": "#777777", "P": "#1f77b4", "S": "#d62728",
          "Srand": "#ff7f0e", "SP": "#9467bd"}


def smooth(x, w=10):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def main() -> int:
    d = json.loads(J.read_text())
    runs = d["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, task in zip(axes, ("reach", "junction")):
        for arm in ARMS:
            per = []
            for seed in (0, 1):
                r = [x for x in runs if x["task"] == task and x["arm"] == arm
                     and x["seed"] == seed][0]
                y = np.array([c["train_bce_batch"] for c in r["curve"]])
                per.append(smooth(y, 10))
                ax.plot(np.arange(len(y)), y, color=COLORS[arm], lw=0.4, alpha=0.15)
            m = np.mean(per, axis=0)
            ax.plot(np.arange(len(m)) + 4, m, color=COLORS[arm], lw=1.9,
                    label=arm if task == "reach" else None)
        ax.axhline(np.log(2), color="black", lw=0.7, ls=":", alpha=0.8)
        ax.set_title(f"{task} — batch BCE (rolling mean 10, mean of seeds)")
        ax.set_xlabel("optimization step")
        ax.set_ylim(0.0, 1.25)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("train BCE (batch 64)")
    axes[0].legend(ncol=5, fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("SFC-v0 corrigendum replay: train BCE under consistent [H,T] features",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
