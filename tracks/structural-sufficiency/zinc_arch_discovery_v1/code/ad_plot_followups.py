"""Plot the pre-declared follow-up curves against the relevant originals."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ad_common as C


def load(run_dir, label: str) -> dict:
    d = C.read_json(Path(run_dir) / "train.json")
    return {
        "label": label,
        "epoch": [int(e["epoch"]) for e in d["curve"]],
        "valid": [float(e["valid_mae"]) for e in d["curve"]],
        "best": d["best_valid"],
        "gpu_h": d["gpu_hours"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(C.ANALYSIS / "curves_followups.png"))
    args = p.parse_args(argv)
    p3_path = C.REPO / "tracks/structural-sufficiency/zinc_perf_protocol_v1/runs/p3_s0/train.json"
    runs = []
    if p3_path.exists():
        runs.append(load(C.REPO / "tracks/structural-sufficiency/zinc_perf_protocol_v1/runs/p3_s0", "P3 s0 (reference)"))
    runs += [
        load(C.RUNS / "T_s0", "T s0 (wd 1e-5)"),
        load(C.RUNS / "wd_T_s0", "T s0 (wd 1e-4 correction)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    colors = ["black", "tab:orange", "tab:red"]
    for r, c in zip(runs, colors):
        axes[0].plot(r["epoch"], r["valid"], color=c, lw=1.3, label=f"{r['label']} best {r['best']['valid_mae']:.4f}@{r['best']['epoch']}")
        axes[0].scatter([r["best"]["epoch"]], [r["best"]["valid_mae"]], marker="*", s=90,
                        color=c, edgecolor="black", zorder=5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("valid MAE (raw y, log)")
    axes[0].set_title("T weight-decay correction")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=7)

    path_runs = [
        load(C.RUNS / "P0_s0", "P0 original (count leak)"),
        load(C.RUNS / "corr_P0_s0", "P0 corrected"),
        load(C.RUNS / "P1_s0", "P1 original (count leak)"),
        load(C.RUNS / "corr_P1_s0", "P1 corrected"),
    ]
    for r, c, ls in zip(path_runs, ["tab:green", "tab:green", "tab:olive", "tab:olive"],
                        ["--", "-", "--", "-"]):
        axes[1].plot(r["epoch"], r["valid"], color=c, ls=ls, lw=1.2,
                     label=f"{r['label']} best {r['best']['valid_mae']:.4f}@{r['best']['epoch']}")
        axes[1].scatter([r["best"]["epoch"]], [r["best"]["valid_mae"]], marker="*", s=80,
                        color=c, edgecolor="black", zorder=5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("valid MAE (raw y, log)")
    axes[1].set_title("P0/P1 sampled-count correction")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=7)
    fig.suptitle("Pre-declared follow-ups: true official-valid curves (test never read)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
