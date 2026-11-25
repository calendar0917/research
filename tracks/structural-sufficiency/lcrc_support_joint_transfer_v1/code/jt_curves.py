"""Small SVG figures from the frozen CSV curves (no smoothing, no selection)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROUND = Path(__file__).resolve().parents[1]
ANALYSIS = ROUND / "analysis"


def rows(name: str) -> list[dict[str, str]]:
    with (ANALYSIS / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def lipo() -> None:
    data = defaultdict(lambda: defaultdict(list))
    for r in rows("curves_lipo.csv"):
        data[r["arm"]]["epoch"].append(int(r["epoch"]))
        data[r["arm"]]["valid"].append(float(r["valid_rmse"]))
        data[r["arm"]]["test"].append(float(r["test_rmse"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for arm in ("C1", "C1_JOINT", "C1_ADD"):
        axes[0].plot(data[arm]["epoch"], data[arm]["valid"], label=arm, lw=1.4)
        axes[1].plot(data[arm]["epoch"], data[arm]["test"], label=arm, lw=1.4)
    for ax, title in zip(axes, ("Valid RMSE (raw)", "Test RMSE (raw)")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS / "curves_lipo.svg")
    plt.close(fig)


def pcba() -> None:
    data = defaultdict(lambda: defaultdict(list))
    for r in rows("curves_pcba.csv"):
        if not r["valid_ap"]:
            continue
        data[r["arm"]]["epoch"].append(int(r["epoch"]))
        data[r["arm"]]["valid"].append(float(r["valid_ap"]))
        if r["test_ap"]:
            data[r["arm"]]["test_epoch"].append(int(r["epoch"]))
            data[r["arm"]]["test"].append(float(r["test_ap"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for arm in ("W2_ORIGINAL", "W2_JOINT", "FIXED1_ORIGINAL", "FIXED1_JOINT"):
        if data[arm]["epoch"]:
            axes[0].plot(data[arm]["epoch"], data[arm]["valid"], label=arm, lw=1.4)
    for arm in ("W2_JOINT", "FIXED1_JOINT"):
        axes[1].plot(data[arm]["test_epoch"], data[arm]["test"], label=arm, lw=1.4)
    axes[1].axhline(0.253616, color="C0", ls="--", lw=0.9, label="W2 original best")
    axes[1].axhline(0.257001, color="C2", ls="--", lw=0.9, label="FIXED1 original best")
    for ax, title in zip(axes, ("Valid macro AP", "Test macro AP (new arms + original best)")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("AP")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(ANALYSIS / "curves_pcba.svg")
    plt.close(fig)


if __name__ == "__main__":
    lipo()
    pcba()
    print("wrote curves_lipo.svg curves_pcba.svg")
