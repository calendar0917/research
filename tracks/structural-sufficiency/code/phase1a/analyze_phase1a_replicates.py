"""Compare Phase 1a R1 and R2 (same frozen design, clean-worktree replication).

R1: initial 18 runs on the shared checkout (unrelated untracked lock files made
    the environment report git_dirty=true; SSRC code diff was empty).
R2: exact same 18-run design, run from a clean worktree at the same frozen
    commit, with git_dirty=false.

The two replicates are reported together; neither is discarded.  This script
does not read Reserved or official valid/test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_phase1a import MODELS, N_LABEL, SEEDS, _audit, _load_runs  # noqa: E402
from phase1a_data import load_frozen_split, scaffold_groups_from_split  # noqa: E402

SPLIT = REPO / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"


def _summ(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {
        "values": [float(x) for x in a],
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n_positive": int((a > 0).sum()),
        "n_negative": int((a < 0).sum()),
        "n": int(a.size),
    }


def _metrics(payload: dict, split_name: str) -> float:
    return float(payload["best_metrics"][split_name]["mae"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", default=str(REPO / "tracks/structural-sufficiency/runs/phase1a"))
    ap.add_argument("--r2", default=str(REPO / "tracks/structural-sufficiency/runs/phase1a_r2"))
    ap.add_argument(
        "--out",
        default=str(REPO / "tracks/structural-sufficiency/results/phase1a/analysis_replicates.json"),
    )
    args = ap.parse_args()

    split = load_frozen_split(SPLIT)
    groups = scaffold_groups_from_split(split)
    r1 = _load_runs(Path(args.r1))
    r2 = _load_runs(Path(args.r2))
    audit1, audit2 = _audit(r1, split, groups), _audit(r2, split, groups)
    if audit1["git_commit"] != audit2["git_commit"]:
        raise AssertionError("replicates use different commits")
    if audit1["config_sha256"] != audit2["config_sha256"]:
        raise AssertionError("replicates use different configs")

    draws = {"R1": r1, "R2": r2}
    per_run: dict[str, dict] = {"R1": {}, "R2": {}}
    for rep, runs in draws.items():
        for n_label in N_LABEL:
            per_run[rep][n_label] = {}
            for seed in SEEDS:
                row = {}
                for model in MODELS:
                    p = runs[(model, n_label, seed)]
                    row[model] = {
                        "monitor_mae_best": _metrics(p, "monitor"),
                        "monitor_mae_last": float(p["last_metrics"]["monitor"]["mae"]),
                        "train_mae_best": _metrics(p, "train"),
                        "best_epoch": p["best_epoch"],
                        "gpu_seconds": float(p["gpu_seconds"]),
                        "flops_forward": p["flops"]["flops_forward"],
                        "params": p["parameters"],
                        "git_dirty": p["environment"]["git_dirty"],
                    }
                row["adv_set"] = row["S"]["monitor_mae_best"] - row["X"]["monitor_mae_best"]
                row["adv_set_last"] = row["S"]["monitor_mae_last"] - row["X"]["monitor_mae_last"]
                row["s_minus_g"] = row["S"]["monitor_mae_best"] - row["G"]["monitor_mae_best"]
                row["x_minus_g"] = row["X"]["monitor_mae_best"] - row["G"]["monitor_mae_best"]
                per_run[rep][n_label][str(seed)] = row

    pooled: dict[str, dict] = {}
    for n_label in N_LABEL:
        pooled[n_label] = {}
        for model in MODELS:
            vals = [
                per_run[rep][n_label][str(seed)][model]["monitor_mae_best"]
                for rep in ("R1", "R2")
                for seed in SEEDS
            ]
            pooled[n_label][model] = _summ(vals)
        for metric in ("adv_set", "adv_set_last", "s_minus_g", "x_minus_g"):
            vals = [
                per_run[rep][n_label][str(seed)][metric]
                for rep in ("R1", "R2")
                for seed in SEEDS
            ]
            pooled[n_label][metric] = _summ(vals)

    interactions = {
        rep: [
            per_run[rep]["small"][str(seed)]["adv_set"]
            - per_run[rep]["2352"][str(seed)]["adv_set"]
            for seed in SEEDS
        ]
        for rep in ("R1", "R2")
    }
    interaction_pooled = _summ(interactions["R1"] + interactions["R2"])

    # Same-seed, same-n R2-R1 differences quantify run-to-run nondeterminism.
    repro: dict[str, dict] = {}
    for n_label in N_LABEL:
        repro[n_label] = {}
        for model in MODELS:
            diffs = [
                per_run["R2"][n_label][str(seed)][model]["monitor_mae_best"]
                - per_run["R1"][n_label][str(seed)][model]["monitor_mae_best"]
                for seed in SEEDS
            ]
            repro[n_label][model] = {
                "r2_minus_r1": diffs,
                "mean_abs": float(np.mean(np.abs(diffs))),
                "max_abs": float(np.max(np.abs(diffs))),
            }

    cost = {
        "R1_gpu_hours_train_loop": sum(p["gpu_seconds"] for p in r1.values()) / 3600,
        "R2_gpu_hours_train_loop": sum(p["gpu_seconds"] for p in r2.values()) / 3600,
        "total_gpu_hours_train_loop": (
            sum(p["gpu_seconds"] for p in r1.values())
            + sum(p["gpu_seconds"] for p in r2.values())
        )
        / 3600,
        "R1_git_dirty_values": sorted({p["environment"]["git_dirty"] for p in r1.values()}),
        "R2_git_dirty_values": sorted({p["environment"]["git_dirty"] for p in r2.values()}),
    }

    payload = {
        "protocol_id": "ssrc-phase1a-lipo-v1",
        "status": "TWO_REPLICATES_SAME_FROZEN_DESIGN",
        "git_commit": audit1["git_commit"],
        "config_sha256": audit1["config_sha256"],
        "audit": {"R1": audit1, "R2": audit2},
        "per_run": per_run,
        "pooled_over_6_draws": pooled,
        "interaction": {
            "R1": _summ(interactions["R1"]),
            "R2": _summ(interactions["R2"]),
            "pooled": interaction_pooled,
        },
        "run_to_run_reproducibility": repro,
        "cost": cost,
        "notes": [
            "R1 and R2 use the same frozen commit/config/split/seeds; R2 was run from a clean worktree.",
            "R1 environment reports git_dirty=true because of unrelated untracked lock files in another track; git diff was empty.",
            "R2 environment reports git_dirty=false. R1 numbers are retained, not replaced.",
            "Pooled ranges include run-to-run nondeterminism; they are not independent-seed confidence intervals.",
            "No Reserved or official valid/test labels were read.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    for n_label in N_LABEL:
        p = pooled[n_label]
        print(
            n_label,
            "S/X/G:",
            round(p["S"]["mean"], 4),
            round(p["X"]["mean"], 4),
            round(p["G"]["mean"], 4),
            "adv:",
            round(p["adv_set"]["mean"], 4),
            f"({p['adv_set']['n_positive']}/{p['adv_set']['n']} X-better draws)",
        )
    print("interaction pooled:", interaction_pooled)
    print("repro max abs:", {n: {m: round(repro[n][m]["max_abs"], 4) for m in MODELS} for n in N_LABEL})
    print("gpu hours total:", round(cost["total_gpu_hours_train_loop"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
