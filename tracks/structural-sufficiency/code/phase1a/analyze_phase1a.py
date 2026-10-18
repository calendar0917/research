"""Analyze the 18 frozen Phase 1a trajectories.

Reads only Fit/Monitor artifacts produced by run_phase1a.py.  Never reads the
internal Reserved labels or the official valid/test labels.

Usage:
    uv run python tracks/structural-sufficiency/code/phase1a/analyze_phase1a.py \
        --runs tracks/structural-sufficiency/runs/phase1a \
        --out tracks/structural-sufficiency/results/phase1a/analysis_summary.json
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

from phase1a_data import load_frozen_split, scaffold_groups_from_split  # noqa: E402

SPLIT = REPO / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"
MODELS = ("S", "X", "G")
SEEDS = (0, 1, 2)
N_LABEL = {"small": 253, "2352": 2352}
BOOT = 5000
RNG_SEED = 20261019


def _load_runs(run_dir: Path) -> dict[tuple[str, str, int], dict]:
    runs: dict[tuple[str, str, int], dict] = {}
    for model in MODELS:
        for n_label in N_LABEL:
            for seed in SEEDS:
                path = run_dir / f"{model}_n{n_label}_seed{seed}.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                runs[(model, n_label, seed)] = json.loads(path.read_text())
    if len(runs) != 18:
        raise RuntimeError(f"expected 18 runs, found {len(runs)}")
    return runs


def _audit(runs: dict[tuple[str, str, int], dict], split, groups) -> dict:
    commits = set()
    config_hashes = set()
    protocol_ids = set()
    monitor_ref = None
    checks = []
    for (model, n_label, seed), payload in runs.items():
        commits.add(payload["environment"].get("git_commit"))
        config_hashes.add(payload["config_sha256"])
        protocol_ids.add(payload["protocol_id"])
        scope = payload["scope"]
        checks.append(
            {
                "model": model,
                "n_label": n_label,
                "seed": seed,
                "n_train": payload["n_train"],
                "n_monitor": payload["n_monitor"],
                "official_valid_test_labels_read": scope["official_valid_test_labels_read"],
                "internal_reserved_labels_read": scope["internal_reserved_labels_read"],
                "reserved_rows_in_train_or_monitor": scope["reserved_rows_in_train_or_monitor"],
                "monitor_row_sha256": payload["monitor_row_sha256"],
                "train_row_sha256": payload["train_row_sha256"],
                "parameters": payload["parameters"],
                "config_sha256": payload["config_sha256"],
            }
        )
        if payload["n_train"] != N_LABEL[n_label]:
            raise AssertionError(f"n_train mismatch: {model} {n_label} {seed}")
        if scope["official_valid_test_labels_read"] is not False:
            raise AssertionError("official labels read")
        if scope["internal_reserved_labels_read"] is not False:
            raise AssertionError("Reserved labels read")
        if scope["reserved_rows_in_train_or_monitor"] != 0:
            raise AssertionError("Reserved rows in train/monitor")
        if not set(payload["monitor_mol_index"]) <= set(split.monitor):
            raise AssertionError("non-Monitor row in monitor predictions")
        if set(payload["monitor_mol_index"]) & set(split.reserved):
            raise AssertionError("Reserved row in monitor predictions")
        if monitor_ref is None:
            monitor_ref = payload["monitor_mol_index"]
        elif payload["monitor_mol_index"] != monitor_ref:
            raise AssertionError("Monitor row order differs across runs")
    if len(commits) != 1 or len(config_hashes) != 1 or len(protocol_ids) != 1:
        raise AssertionError(f"freeze mismatch: commits={commits} configs={config_hashes}")
    expected_groups = {groups[r] for r in monitor_ref}
    return {
        "n_runs": len(runs),
        "git_commit": next(iter(commits)),
        "config_sha256": next(iter(config_hashes)),
        "protocol_id": next(iter(protocol_ids)),
        "monitor_rows": len(monitor_ref),
        "monitor_scaffold_groups": len(expected_groups),
        "checks": checks,
    }


def _mae(payload: dict, which: str, split_name: str) -> float:
    return float(payload[f"{which}_metrics"][split_name]["mae"])


def _bootstrap(
    runs: dict[tuple[str, str, int], dict],
    monitor_index: list[int],
    groups: dict[int, str],
    n_boot: int,
) -> dict:
    # Absolute errors on the same Monitor rows, in the same order.
    errors: dict[tuple[str, str, int], np.ndarray] = {}
    for key, payload in runs.items():
        model, n_label, seed = key
        pred = np.asarray(payload["monitor_pred_best"], dtype=np.float64)
        y = np.asarray(payload["monitor_y"], dtype=np.float64)
        errors[key] = np.abs(pred - y)

    monitor_groups = np.asarray([groups[r] for r in monitor_index], dtype=object)
    unique_groups = np.asarray(sorted(set(monitor_groups)), dtype=object)
    group_rows: dict[str, np.ndarray] = {
        g: np.flatnonzero(monitor_groups == g) for g in unique_groups
    }
    rng = np.random.default_rng(RNG_SEED)

    # adv[b, n, seed], aux_sg[b, n], aux_xg[b, n], inter[b, seed]
    adv = np.empty((n_boot, len(N_LABEL), len(SEEDS)))
    aux_sg = np.empty((n_boot, len(N_LABEL)))
    aux_xg = np.empty((n_boot, len(N_LABEL)))
    for b in range(n_boot):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_rows[g] for g in picked])
        for ni, n_label in enumerate(N_LABEL):
            per_seed = []
            for si, seed in enumerate(SEEDS):
                mae_s = float(errors[("S", n_label, seed)][idx].mean())
                mae_x = float(errors[("X", n_label, seed)][idx].mean())
                mae_g = float(errors[("G", n_label, seed)][idx].mean())
                adv[b, ni, si] = mae_s - mae_x
                per_seed.append((mae_s, mae_x, mae_g))
            aux_sg[b, ni] = np.mean([p[0] - p[2] for p in per_seed])
            aux_xg[b, ni] = np.mean([p[1] - p[2] for p in per_seed])
    inter = adv[:, 0, :] - adv[:, 1, :]

    def ci(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
            "q2_5": float(np.percentile(arr, 2.5)),
            "q97_5": float(np.percentile(arr, 97.5)),
        }

    return {
        "bootstrap_reps": n_boot,
        "bootstrap_seed": RNG_SEED,
        "resampling_unit": "Monitor scaffold group (421 groups)",
        "adv_set_mean_over_seeds": {
            n_label: ci(adv[:, ni, :].mean(axis=1)) for ni, n_label in enumerate(N_LABEL)
        },
        "adv_set_per_seed": {
            n_label: {
                str(seed): ci(adv[:, ni, si]) for si, seed in enumerate(SEEDS)
            }
            for ni, n_label in enumerate(N_LABEL)
        },
        "interaction_mean_over_seeds": ci(inter.mean(axis=1)),
        "interaction_per_seed": {
            str(seed): ci(inter[:, si]) for si, seed in enumerate(SEEDS)
        },
        "aux_s_minus_g_mean_over_seeds": {
            n_label: ci(aux_sg[:, ni]) for ni, n_label in enumerate(N_LABEL)
        },
        "aux_x_minus_g_mean_over_seeds": {
            n_label: ci(aux_xg[:, ni]) for ni, n_label in enumerate(N_LABEL)
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "tracks/structural-sufficiency/runs/phase1a"))
    ap.add_argument("--out", default=str(REPO / "tracks/structural-sufficiency/results/phase1a/analysis_summary.json"))
    ap.add_argument("--bootstrap", type=int, default=BOOT)
    args = ap.parse_args()

    run_dir = Path(args.runs)
    split = load_frozen_split(SPLIT)
    groups = scaffold_groups_from_split(split)
    runs = _load_runs(run_dir)
    audit = _audit(runs, split, groups)
    monitor_index = runs[("S", "small", 0)]["monitor_mol_index"]

    # Sanity: Monitor labels are identical across runs (same rows, same raw labels).
    y_ref = np.asarray(runs[("S", "small", 0)]["monitor_y"], dtype=np.float64)
    for payload in runs.values():
        if not np.allclose(y_ref, np.asarray(payload["monitor_y"], dtype=np.float64)):
            raise AssertionError("Monitor labels differ across runs")

    table: dict[str, dict] = {}
    for n_label in N_LABEL:
        table[n_label] = {}
        for seed in SEEDS:
            row = {}
            for model in MODELS:
                p = runs[(model, n_label, seed)]
                row[model] = {
                    "params": p["parameters"],
                    "best_epoch": p["best_epoch"],
                    "monitor_mae_best": _mae(p, "best", "monitor"),
                    "monitor_rmse_best": float(p["best_metrics"]["monitor"]["rmse"]),
                    "monitor_r2_best": float(p["best_metrics"]["monitor"]["r2"]),
                    "monitor_mae_last": _mae(p, "last", "monitor"),
                    "train_mae_best": _mae(p, "best", "train"),
                    "train_mae_last": _mae(p, "last", "train"),
                    "gpu_seconds": float(p["gpu_seconds"]),
                    "train_seconds": float(p["train_seconds"]),
                    "peak_gpu_mb": float(p["peak_gpu_memory_mb_allocated"]),
                    "step_time_ms_median": float(p["step_time_ms_median"]),
                    "flops_forward": p["flops"]["flops_forward"],
                }
            const_mae = float(
                np.mean(np.abs(y_ref - float(runs[("S", n_label, seed)]["label_standardization"]["mean"])))
            )
            row["constant_train_mean_monitor_mae"] = const_mae
            row["adv_set"] = row["S"]["monitor_mae_best"] - row["X"]["monitor_mae_best"]
            row["adv_set_last"] = row["S"]["monitor_mae_last"] - row["X"]["monitor_mae_last"]
            row["s_minus_g"] = row["S"]["monitor_mae_best"] - row["G"]["monitor_mae_best"]
            row["x_minus_g"] = row["X"]["monitor_mae_best"] - row["G"]["monitor_mae_best"]
            row["s_vs_constant"] = row["S"]["monitor_mae_best"] - const_mae
            row["x_vs_constant"] = row["X"]["monitor_mae_best"] - const_mae
            row["g_vs_constant"] = row["G"]["monitor_mae_best"] - const_mae
            table[n_label][str(seed)] = row

    # Seed summaries.
    seed_summary: dict[str, dict] = {}
    for n_label in N_LABEL:
        seed_summary[n_label] = {}
        for metric in ("adv_set", "adv_set_last", "s_minus_g", "x_minus_g"):
            vals = [table[n_label][str(s)][metric] for s in SEEDS]
            seed_summary[n_label][metric] = {
                "values": vals,
                "mean": float(np.mean(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_positive": int(sum(v > 0 for v in vals)),
                "n_negative": int(sum(v < 0 for v in vals)),
            }
    interactions = [
        table["small"][str(s)]["adv_set"] - table["2352"][str(s)]["adv_set"] for s in SEEDS
    ]
    seed_summary["interaction"] = {
        "values": interactions,
        "mean": float(np.mean(interactions)),
        "min": float(np.min(interactions)),
        "max": float(np.max(interactions)),
        "n_positive": int(sum(v > 0 for v in interactions)),
        "n_negative": int(sum(v < 0 for v in interactions)),
    }

    # Cost.
    gpu_seconds = sum(float(p["gpu_seconds"]) for p in runs.values())
    train_seconds = sum(float(p["train_seconds"]) for p in runs.values())
    preprocess_seconds = sum(float(p["preprocess_seconds"]) for p in runs.values())
    cost = {
        "n_runs": len(runs),
        "sum_gpu_seconds_train_loop": gpu_seconds,
        "sum_gpu_hours_train_loop": gpu_seconds / 3600,
        "sum_train_seconds": train_seconds,
        "sum_preprocess_seconds": preprocess_seconds,
        "sum_peak_gpu_mb_max": max(float(p["peak_gpu_memory_mb_allocated"]) for p in runs.values()),
        "per_model_n_mean_gpu_seconds": {
            f"{model}_{n_label}": float(
                np.mean([runs[(model, n_label, seed)]["gpu_seconds"] for seed in SEEDS])
            )
            for model in MODELS
            for n_label in N_LABEL
        },
    }

    boot = _bootstrap(runs, monitor_index, groups, args.bootstrap)

    payload = {
        "protocol_id": "ssrc-phase1a-lipo-v1",
        "analysis_status": "EXPLORATORY_SINGLE_SPLIT_THREE_SEEDS",
        "audit": audit,
        "per_run": table,
        "seed_summary": seed_summary,
        "bootstrap": boot,
        "cost": cost,
        "disclosures": [
            "Single frozen internal split; 3 seeds; Monitor used for early stopping.",
            "Scaffold-cluster bootstrap covers Monitor sampling only, not train/split uncertainty.",
            "No equivalence test; 'CI crosses zero' means unresolved, not equivalent.",
            "Phase 0 LOO used all 3360 official-train labels; internal Reserved is not historically fully blind, but Phase 1a did not read/score it.",
            "X uses ~1.5-1.6x S forward FLOPs on actual run batches (1.41x on the CPU-gate fixed batch); extra compute is an alternative explanation for an X advantage.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(
        "adv_set mean (S-X; positive=X better):",
        {n: round(seed_summary[n]["adv_set"]["mean"], 4) for n in N_LABEL},
    )
    print("interaction mean:", round(seed_summary["interaction"]["mean"], 4))
    print("gpu_hours_train_loop:", round(gpu_seconds / 3600, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
