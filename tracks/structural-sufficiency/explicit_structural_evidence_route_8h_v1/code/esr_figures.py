"""Round figures (PNG, gitignored): curves, small-data bars, contribution profiles.

    python esr_figures.py --runs runs --analysis analysis --out figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import esr_common as C  # noqa: E402

HIST = {
    "W2": {
        "path": C.REPO / "tracks/structural-sufficiency/lcrc_support_joint_transfer_v1/runs/W2_JOINT/train_curve.csv",
        "label": "W2 (historical, 40 ep)",
    },
    "FIXED1": {
        "path": C.REPO / "tracks/structural-sufficiency/lcrc_support_joint_transfer_v1/runs/FIXED1_JOINT/train_curve.csv",
        "label": "FIXED1 (historical, 25 ep)",
    },
}


def _num(value: str) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _read_curve(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: _num(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(fh)]


def pcba_curves(runs: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, meta in HIST.items():
        if not meta["path"].exists():
            continue
        rows = _read_curve(meta["path"])
        ax.plot([r["epoch"] for r in rows], [r["valid_ap"] for r in rows], lw=1.2, ls="--", color="0.45", label=meta["label"])
    for run_json in sorted(runs.glob("*/run.json")):
        payload = json.loads(run_json.read_text())
        if payload.get("dataset") != C.PCBA or payload.get("smoke"):
            continue
        curve = _read_curve(run_json.parent / "curve.csv")
        if not curve:
            continue
        ax.plot([r["epoch"] for r in curve], [r["valid_ap"] for r in curve], lw=1.6, label=payload["arm"])
    ax.axvline(25, color="0.7", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Validation macro AP")
    ax.set_title("PCBA: explicit structural evidence arms vs historical backbones")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pcba_valid_ap_curves.png", dpi=160)
    plt.close(fig)


def small_bars(table: Path, out: Path) -> None:
    rows = list(csv.DictReader(open(table, newline="", encoding="utf-8")))
    datasets = [d for d in ("LIPO", "TOX21", "ESOL") if any(r["dataset"] == d for r in rows)]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.4))
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, dataset in zip(axes, datasets):
        sub = [r for r in rows if r["dataset"] == dataset and r["source"] == "this_round"]
        sub.sort(key=lambda r: float(r["primary_best_valid"]))
        vals = [float(r["primary_best_valid"]) for r in sub]
        ax.barh([r["arm"] for r in sub], vals, color="#4c78a8")
        ax.set_title(f"{dataset} best Validation")
        ax.set_xlabel("RMSE" if dataset in ("LIPO", "ESOL") else "macro ROC-AUC")
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "small_dataset_arms.png", dpi=160)
    plt.close(fig)


def contribution_profiles(trace: Path, out: Path) -> None:
    if not trace.exists():
        return
    payload = json.loads(trace.read_text())
    entries = [t for t in payload.get("traces", []) if t.get("rows")]
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels, shares = [], []
    for entry in entries:
        for row in entry["rows"]:
            if "share_local" not in row:
                continue
            labels.append(f"{entry['dataset']}/{entry['arm']}")
            shares.append(row["share_local"])
    if not shares:
        return
    ax.hist(shares, bins=20, color="#72b7b2")
    ax.set_xlabel("share of |local token contribution| (per traced molecule)")
    ax.set_ylabel("count")
    ax.set_title("Local vs relation contribution share")
    fig.tight_layout()
    fig.savefig(out / "contribution_share.png", dpi=160)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="round figures")
    ap.add_argument("--runs", default=str(C.RUNS))
    ap.add_argument("--analysis", default=str(C.ANALYSIS))
    ap.add_argument("--out", default=str(C.ROUND / "figures"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pcba_curves(Path(args.runs), out)
    small_bars(Path(args.analysis) / "main_table.csv", out)
    contribution_profiles(Path(args.analysis) / "decomposition_trace.json", out)
    C.log(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
