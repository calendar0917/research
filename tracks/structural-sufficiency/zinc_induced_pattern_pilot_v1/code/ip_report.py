"""Build the pilot's machine-readable tables, paired bootstrap and curves.

Reads only official-valid predictions/labels.  The official test is never
constructed or read.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
_V2 = _HERE.parents[1] / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
V2 = EXP.parent / "zinc_recipe_trunk_v2"
V1PERF = EXP.parent / "zinc_perf_protocol_v1"

MODELS = {
    "P3 x A s0": V1PERF / "runs" / "p3_s0",
    "P3 x A s1": V1PERF / "runs" / "p3_s1",
    "P3 x B s0": V2 / "runs" / "xBssl_P3N_s0",
    "J s0": RUNS / "J_s0",
    "Q s0": RUNS / "Q_s0",
}


def load_run(path: Path) -> dict:
    info = C.read_json(path / "train.json")
    curve = []
    with open(path / "curve.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            curve.append(
                {
                    "epoch": int(row["epoch"]),
                    "online_train_mae": float(row["online_train_mae"]),
                    "valid_mae": float(row["valid_mae"]),
                    "lr": float(row["lr"]),
                    "seconds": float(row["seconds"]),
                    "steps": int(row.get("steps", 79)),
                }
            )
    pred_path = path / "predictions" / "valid_best.npz"
    npz = np.load(pred_path)
    pred = np.asarray(npz["pred"], dtype=np.float64)
    row_idx = np.asarray(npz["row"], dtype=np.int64)
    sha = {name: C.sha256_file(path / name) for name in ("best.pt", "final.pt")}
    return {"path": path, "info": info, "curve": curve, "pred": pred, "row": row_idx, "sha": sha}


def main() -> int:
    valid = D.load_cache("valid")
    y = np.asarray(valid["labels"], dtype=np.float64)
    runs = {name: load_run(path) for name, path in MODELS.items()}
    # Validate the prediction row order and MAE against the stored best.
    for name, r in runs.items():
        if not np.array_equal(r["row"], np.arange(len(y), dtype=np.int64)):
            raise RuntimeError(f"{name}: valid prediction rows are not 0..999")
        mae = float(np.abs(r["pred"] - y).mean())
        r["pred_mae"] = mae
        stored = float(r["info"]["fixed_objects"]["best"]["valid_mae"])
        if abs(mae - stored) > 1e-5:
            raise RuntimeError(f"{name}: pred MAE {mae} != stored {stored}")
        r["recomputed_mae"] = mae

    # Paired bootstrap, descriptive only.
    rng = np.random.default_rng(20261021)
    n = len(y)
    boots = rng.integers(0, n, size=(10000, n))
    def pair(a: str, b: str) -> dict:
        da = np.abs(runs[a]["pred"] - y)
        db = np.abs(runs[b]["pred"] - y)
        diff = da - db
        means = diff[boots].mean(axis=1)
        return {
            "contrast": f"{a} - {b}",
            "mean_delta_MAE": float(diff.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "note": "negative means the first model has lower valid MAE",
        }
    contrasts = [
        pair("J s0", "P3 x B s0"),
        pair("Q s0", "P3 x B s0"),
        pair("J s0", "Q s0"),
        pair("J s0", "P3 x A s0"),
        pair("Q s0", "P3 x A s0"),
        pair("J s0", "P3 x A s1"),
        pair("Q s0", "P3 x A s1"),
    ]

    table = []
    for name, r in runs.items():
        info = r["info"]
        best = info["fixed_objects"]["best"]
        final = info["fixed_objects"]["final"]
        table.append(
            {
                "model": name,
                "parameters": int(info["parameter_count"]),
                "best_valid": float(info["best_valid"]["valid_mae"]),
                "best_epoch": int(info["best_valid"]["epoch"]),
                "train_at_best_eval": float(best["train_mae_eval_mode"]),
                "final_valid": float(final["valid_mae"]),
                "final_train_eval": float(final["train_mae_eval_mode"]),
                "gpu_hours_run_level": info.get("gpu_hours"),
                "wall_seconds": info.get("wall_seconds"),
                "best_sha256": r["sha"]["best.pt"],
                "final_sha256": r["sha"]["final.pt"],
                "recipe_variant": info.get("recipe_variant"),
            }
        )

    out = {
        "protocol_id": "zinc-induced-pattern-pilot-v1",
        "decision_split": "official valid 1000, labels reused from the established split",
        "test_loaded": False,
        "models": table,
        "contrasts": contrasts,
        "notes": [
            "P3 x A s0/s1 are historical background; the primary reference is P3 x B s0 0.076224.",
            "All deltas are paired molecule-wise on the same 1000 valid rows.",
            "The valid split has been read in earlier rounds; this is not a fresh blind evaluation.",
        ],
    }
    C.write_json(ANALYSIS / "main_table.json", out)
    C.write_json(ANALYSIS / "paired_bootstrap.json", {"contrasts": contrasts})
    lines = [
        "# Induced-pattern pilot — main table",
        "",
        "| model | params | best valid | epoch | train@best (eval) | final valid | final train (eval) | GPU-h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['model']} | {row['parameters']:,} | {row['best_valid']:.6f} | {row['best_epoch']} | "
            f"{row['train_at_best_eval']:.6f} | {row['final_valid']:.6f} | {row['final_train_eval']:.6f} | "
            f"{row['gpu_hours_run_level']:.4f} |"
        )
    lines += ["", "Paired bootstrap (10,000 resamples; negative = first model better):", "", "| contrast | mean Δ | 95% CI |", "|---|---:|---:|"]
    for c in contrasts:
        lines.append(f"| {c['contrast']} | {c['mean_delta_MAE']:+.6f} | [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] |")
    (ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Raw curves: no smoothing or reconstruction.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for name, color in (("P3 x A s0", "#888888"), ("P3 x B s0", "#111111"), ("J s0", "#1f77b4"), ("Q s0", "#d62728")):
        r = runs[name]
        ep = np.asarray([e["epoch"] for e in r["curve"]])
        va = np.asarray([e["valid_mae"] for e in r["curve"]])
        axes[0].plot(ep, va, label=name, color=color, lw=1.2)
        axes[1].plot(ep, va, label=name, color=color, lw=1.2)
        b = r["info"]["best_valid"]
        axes[1].scatter([b["epoch"]], [b["valid_mae"]], color=color, s=18, zorder=5)
    axes[0].set_title("full 600-epoch valid curves")
    axes[1].set_title("zoom: 0.06–0.12, best-valid markers")
    axes[1].set_ylim(0.06, 0.12)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("valid MAE (raw-y L1)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(ANALYSIS / "curves.png", dpi=140)
    plt.close(fig)

    decision = {
        "protocol_id": "zinc-induced-pattern-pilot-v1",
        "primary_reference": "P3 x B s0 = 0.07622375113883754",
        "observed": {
            "J_s0_best_valid": 0.07977334395417711,
            "J_s0_best_epoch": 335,
            "Q_s0_best_valid": 0.08846381501847646,
            "Q_s0_best_epoch": 293,
            "J_minus_P3B": contrasts[0]["mean_delta_MAE"],
            "Q_minus_P3B": contrasts[1]["mean_delta_MAE"],
            "J_minus_Q": contrasts[2]["mean_delta_MAE"],
            "J_minus_P3A_s0": contrasts[3]["mean_delta_MAE"],
            "Q_minus_P3A_s0": contrasts[4]["mean_delta_MAE"],
            "test_loaded": False,
        },
        "interpretation": [
            "J does not beat the pre-declared P3 x B seed-0 reference; the gap is +0.00355 valid MAE.",
            "Q is clearly worse than J and P3 x B on valid.",
            "The J-Q gap (+0.00869 in favour of J) is a performance signal that the full induced topology helps relative to the typed sequence control, but both models are behind P3.",
            "This is one seed per arm and the valid split is historically reused; no significance or SOTA claim.",
        ],
        "next_choice": [
            "Stop this pilot. J/Q show no meaningful gain over P3 x B; do not buy seed 1.",
            "Do not open rank/depth/path-length/pooling/auxiliary/optimizer grids or rerun P0/P1.",
            "If revisited, the only justified question is a different interface/object class, with a new pre-registration; the negative verdict is limited to this representation, interface and recipe.",
        ],
    }
    C.write_json(ANALYSIS / "round_decision.json", decision)
    C.log(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
