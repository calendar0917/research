"""Plot the real train/valid curves from the frozen run logs.

Reads only `runs/*/train.json` (no test data, no labels) and writes
`analysis/curves.png`.  The online train MAE is the within-epoch pre-update
running mean; the snapshot is the eval-mode MAE on the fixed first 2,000
train rows; valid is the official 1,000-row valid MAE.  They are deliberately
drawn as separate quantities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import zpp_common as C  # noqa: E402


def _curve(run: str) -> dict:
    info = C.read_json(C.RUNS / run / "train.json")
    ep = [int(e["epoch"]) for e in info["curve"]]
    return {
        "epoch": ep,
        "valid": [float(e["valid_mae"]) for e in info["curve"]],
        "online": [float(e["online_train_mae"]) for e in info["curve"]],
        "snapshot": [
            (int(e["epoch"]), float(e["snapshot_train_mae"]))
            for e in info["curve"]
            if e.get("snapshot_train_mae") is not None
        ],
        "best": info["best_valid"],
    }


def _decompose(ax, run: str, title: str, color: str) -> None:
    c = _curve(run)
    ax.plot(c["epoch"], c["valid"], color="black", lw=1.4, label="valid (official 1k)")
    ax.plot(c["epoch"], c["online"], color=color, lw=1.4, label="train online (pre-update mean)")
    if c["snapshot"]:
        sx, sy = zip(*c["snapshot"])
        ax.plot(sx, sy, color=color, lw=1.2, ls="--", marker=".", ms=3,
                label="train snapshot (eval-mode, 2k)")
    ax.scatter([c["best"]["epoch"]], [c["best"]["valid_mae"]], marker="*", s=90,
               color="black", zorder=5, label=f"best valid @{c['best']['epoch']}")
    ax.set_yscale("log")
    ax.set_xlim(0, 361)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (raw y, log)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=6, loc="upper right")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(C.ANALYSIS / "curves.png"))
    args = p.parse_args(argv)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    ax = axes[0, 0]
    for run, label, color in (
        ("p3_s0", "P3 s0", "black"),
        ("p3_s1", "P3 s1", "0.45"),
        ("zsrt_A_main_s0", "ZSRT A_main s0", "tab:blue"),
        ("zsrt_B_lr_low_s0", "ZSRT B_lr_low s0", "tab:green"),
        ("zsrt_C_wd_high_s0", "ZSRT C_wd_high s0", "tab:red"),
        ("zsrt_D_no_rrwp_s0", "ZSRT D_no_rrwp s0", "tab:purple"),
    ):
        c = _curve(run)
        ax.plot(c["epoch"], c["valid"], lw=1.2, color=color, label=label)
        ax.scatter([c["best"]["epoch"]], [c["best"]["valid_mae"]], s=28, color=color, zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(0, 361)
    ax.set_xlabel("epoch")
    ax.set_ylabel("valid MAE (raw y, log)")
    ax.set_title("official-valid curves (star = best-valid checkpoint)", fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=6, loc="upper right")

    _decompose(axes[0, 1], "p3_s0", "P3 seed 0: valid vs train", "tab:blue")
    _decompose(axes[1, 0], "p3_s1", "P3 seed 1: valid vs train", "tab:orange")
    _decompose(axes[1, 1], "zsrt_C_wd_high_s0", "ZSRT C_wd_high seed 0: valid vs train", "tab:red")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    C.log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
