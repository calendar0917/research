"""Plot the real valid curves from the six local run logs (no test data)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ad_common as C

P3_RUN = C.REPO / "tracks/structural-sufficiency/zinc_perf_protocol_v1/runs/p3_s0/train.json"


def load_curve(path: Path):
    d = C.read_json(path)
    ep = [int(e["epoch"]) for e in d["curve"]]
    valid = [float(e["valid_mae"]) for e in d["curve"]]
    online = [float(e["online_train_mae"]) for e in d["curve"]]
    return {
        "epoch": ep,
        "valid": valid,
        "online": online,
        "best": d["best_valid"],
        "params": d["parameter_count"],
        "wall": d.get("wall_seconds"),
        "gpu_h": d.get("gpu_hours"),
        "name": d.get("candidate", d.get("model", "P3")),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(C.ANALYSIS / "curves.png"))
    args = p.parse_args(argv)
    cands = []
    for name in ("R0", "R1", "T", "P0", "P1", "B"):
        path = C.RUNS / f"{name}_s0/train.json"
        if path.exists():
            cands.append(load_curve(path))
    p3 = load_curve(P3_RUN) if P3_RUN.exists() else None

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=False)
    colors = dict(R0="tab:blue", R1="tab:cyan", T="tab:orange", P0="tab:green", P1="tab:olive", B="tab:purple")
    for ax, c in zip(axes.ravel(), cands):
        if p3 is not None:
            ax.plot(p3["epoch"], p3["valid"], color="black", lw=1.0, alpha=0.55, label="P3 s0 valid")
        ax.plot(c["epoch"], c["valid"], color=colors.get(c["name"], "tab:blue"), lw=1.4, label=f"{c['name']} valid")
        ax.plot(c["epoch"], c["online"], color=colors.get(c["name"], "tab:blue"), lw=0.8, alpha=0.35,
                label="online train (pre-update)")
        ax.scatter([c["best"]["epoch"]], [c["best"]["valid_mae"]], marker="*", s=90,
                   color=colors.get(c["name"], "tab:blue"), edgecolor="black", zorder=6,
                   label=f"best {c['best']['valid_mae']:.4f}@{c['best']['epoch']}")
        ax.set_title(f"{c['name']} s0  ({c['params']:,} params, {c['gpu_h']:.2f} GPU-h)", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MAE (raw y, log)")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=6, loc="upper right")
    for ax in axes.ravel()[len(cands):]:
        ax.axis("off")
    fig.suptitle("ZINC official-valid curves, six seed-0 candidates (test never read)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
