"""Plot the frozen training curves used by the pilot report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cpo_common as C

SERIES = {
    "A parent (cut-wise)": (C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0", "#1f77b4", "-"),
    "F_GRAPH (graph-level)": (C.run_dir("F_GRAPH_LIPO_R0"), "#d62728", "-"),
    "B_CSOI_ADD (cut-wise)": (C.CSOI_ROUND / "results" / "CSOI_ADD_LIPO_R0", "#2ca02c", "--"),
    "B_GRAPH (graph-level)": (C.run_dir("B_GRAPH_LIPO_R0"), "#98df8a", "-"),
    "C_CSOI_GLOBAL (cut-wise)": (C.CSOI_ROUND / "results" / "CSOI_GLOBAL_LIPO_R0", "#9467bd", "--"),
    "C_GRAPH (graph-level)": (C.run_dir("C_GRAPH_LIPO_R0"), "#c5b0d5", "-"),
}


def _best_epoch(run_dir: Path) -> int | None:
    man = run_dir / "manifest.json"
    if not man.is_file():
        return None
    return int(C.read_json(man).get("best", {}).get("epoch"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else C.results_root() / "figures" / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, (run_dir, color, ls) in SERIES.items():
        rows = C.read_csv(run_dir / "curves.csv")
        if not rows:
            continue
        epochs = [int(float(r["epoch"])) for r in rows]
        axes[0].plot(epochs, [float(r["valid_rmse"]) for r in rows], color=color, ls=ls,
                     lw=1.4, label=label)
        if "valid_cut_var_std" in rows[0]:
            axes[1].plot(epochs, [float(r["valid_cut_var_std"]) for r in rows], color=color,
                         ls=ls, lw=1.4, label=label)
        if "valid_graph_mse_std" in rows[0]:
            axes[2].plot(epochs, [float(r["valid_graph_mse_std"]) for r in rows], color=color,
                         ls=ls, lw=1.4, label=label)
        be = _best_epoch(run_dir)
        if be is not None:
            hit = min(rows, key=lambda r: abs(int(float(r["epoch"])) - be))
            axes[0].scatter([int(float(hit["epoch"]))], [float(hit["valid_rmse"])],
                            color=color, s=18, zorder=5)
    axes[0].set_title("Valid molecule RMSE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("RMSE (label units)")
    axes[1].set_title("Valid cut variance (standardised)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mean Var_cut")
    axes[2].set_title("Valid graph-level MSE (standardised)")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("mean (f_bar - y)^2")
    axes[0].legend(fontsize=7, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())