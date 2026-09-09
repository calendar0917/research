"""Multi-seed protocol-confirmation analysis for compact-v2 vs compact-v4-hinge.

Reads completed terminal runs (with the selection-checkpoint test evaluation,
``evaluation.selection_checkpoint_test=true``) and produces:

* per-seed table: best valid MAE/epoch, selection-checkpoint test MAE,
  train+valid refit test MAE (if present), parameters, runtime;
* paired improvement stats on test and on validation (mean / std / median /
  win counts / paired t);
* test subgroup MAEs (A: no long-cycle, B: mild, C: extreme) per seed and
  aggregated across seeds (mean of per-seed group deltas +/- std);
* valid subgroup MAEs (same groups) per seed (diagnostic);
* seed-0 guard checks against the canonical pre-stage numbers.

Group definitions (label-excess, from the long-cycle audit):
    A: excess == 0, B: excess == 1, C: excess >= 2.
Molecule alignment is by dataset order (index), verified by asserting that
the run's stored targets equal the audit CSV ``y_stored`` values exactly.

Usage:
    python -m tracks.ksvd.experiments.luyin16.zinc_multiseed_protocol_analysis \
        --runs-root tracks/ksvd/runs \
        --v2-run-ids ID0 ID1 ... --hinge-run-ids ID0 ID1 ... \
        [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIT_CSV = (
    REPO_ROOT
    / "tracks/ksvd/results/zinc_long_cycle_audit/label_effective_cycle.csv"
)

CANONICAL = {
    # (model): valid best mae, selected epoch, refit test mae, [ad-hoc selection test mae]
    "v2": (0.18415821571176638, 56, 0.1353615188403055, 0.154284),
    "v4-hinge": (0.17006561887910357, 53, 0.13944621286727488, 0.133901),
}


def run_path(run_root: Path, run_id: str) -> Path:
    return run_root / run_id[:4] / run_id[4:6] / run_id[6:8] / run_id


def load_manifest(run_root: Path, run_id: str) -> dict[str, Any]:
    return json.loads((run_path(run_root, run_id) / "manifest.json").read_text(encoding="utf-8"))


def load_result(run_root: Path, run_id: str) -> dict[str, Any]:
    return json.loads(
        (run_path(run_root, run_id) / "artifacts" / "legacy_full_result.json").read_text(
            encoding="utf-8"
        )
    )


def load_excess_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(AUDIT_CSV)
    frame["label_excess"] = (-frame["label_effective_cycle_snapped"]).round().clip(
        lower=0
    )
    frame = frame.sort_values(["split", "subset_index"]).reset_index(drop=True)
    return (
        frame[frame["split"] == "valid"].reset_index(drop=True),
        frame[frame["split"] == "test"].reset_index(drop=True),
    )


def subgroup_mae(
    targets: np.ndarray, predictions: np.ndarray, excess: np.ndarray
) -> dict[str, float]:
    masks = {"A": excess == 0, "B": excess == 1, "C": excess >= 2}
    return {
        letter: float(np.mean(np.abs(predictions[mask] - targets[mask])))
        for letter, mask in masks.items()
    }


def per_run_stats(
    run_root: Path,
    run_id: str,
    model: str,
    valid_excess: pd.DataFrame,
    test_excess: pd.DataFrame,
) -> dict[str, Any]:
    manifest = load_manifest(run_root, run_id)
    result = load_result(run_root, run_id)
    valid = result["evaluation"]["valid"]
    sel_test = result["evaluation"].get("test_with_selection_checkpoint") or {}
    refit_test = result["evaluation"]["test_after_train_valid_refit"]
    seed = int(result["seed"])

    valid_targets = np.asarray(valid["targets"], dtype=np.float64)
    valid_preds = np.asarray(valid["predictions"], dtype=np.float64)
    ve = valid_excess.sort_values("subset_index").reset_index(drop=True)
    if not np.allclose(valid_targets, ve["y_stored"].to_numpy(dtype=np.float64), atol=1e-9, rtol=1e-9):
        raise AssertionError(f"{run_id}: valid targets do not match audit y_stored")
    stats: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "run_id": run_id,
        "git_commit": (manifest.get("git") or {}).get("commit"),
        "config_hash": manifest.get("config_hash"),
        "protocol_hash": manifest.get("protocol_hash"),
        "valid_best_mae": float(valid["best_mae"]),
        "best_valid_epoch": int(valid["selected_epoch"]),
        "params": int(manifest.get("metrics", {}).get("parameters") or valid["parameters"]),
        "training_time_s": float(manifest.get("metrics", {}).get("runtime_seconds")),
        "valid_subgroups": subgroup_mae(valid_targets, valid_preds, ve["label_excess"].to_numpy()),
    }
    sel_mae = sel_test.get("mae")
    if sel_mae is not None:
        sel_targets = np.asarray(sel_test["targets"], dtype=np.float64)
        sel_preds = np.asarray(sel_test["predictions"], dtype=np.float64)
        te = test_excess.sort_values("subset_index").reset_index(drop=True)
        if not np.allclose(sel_targets, te["y_stored"].to_numpy(dtype=np.float64), atol=1e-9, rtol=1e-9):
            raise AssertionError(f"{run_id}: test targets do not match audit y_stored")
        stats["test_mae_selection_checkpoint"] = float(sel_mae)
        stats["test_subgroups"] = subgroup_mae(sel_targets, sel_preds, te["label_excess"].to_numpy())
        stats["test_subgroup_n"] = {
            "A": int((te["label_excess"] == 0).sum()),
            "B": int((te["label_excess"] == 1).sum()),
            "C": int((te["label_excess"] >= 2).sum()),
        }
    refit_mae = refit_test.get("mae")
    if refit_mae is not None:
        stats["test_mae_refit"] = float(refit_mae)
    return stats


def paired_stats(deltas: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(deltas, dtype=np.float64)
    out: dict[str, float] = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "median": float(np.median(arr)),
        "wins": int((arr > 0).sum()),
        "losses": int((arr < 0).sum()),
        "ties": int((arr == 0).sum()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    if arr.size > 1 and float(np.std(arr, ddof=1)) > 0:
        from scipy import stats as sps

        sd = float(np.std(arr, ddof=1))
        t = float(np.mean(arr) / (sd / np.sqrt(arr.size)))
        out["paired_t"] = t
        out["paired_t_pvalue"] = float(2 * sps.t.sf(abs(t), df=arr.size - 1))
        out["mean_ci95_halfwidth"] = float(
            sps.t.ppf(0.975, df=arr.size - 1) * (sd / np.sqrt(arr.size))
        )
    return out


def aggregate_subgroups(stats_by_seed: dict[int, dict[str, Any]], key: str) -> dict[str, Any]:
    letters = ("A", "B", "C")
    out: dict[str, Any] = {}
    for letter in letters:
        deltas = []
        v2_maes = []
        v4_maes = []
        for seed in sorted(stats_by_seed):
            v2_maes.append(stats_by_seed[seed]["v2"][key][letter])
            v4_maes.append(stats_by_seed[seed]["v4-hinge"][key][letter])
            deltas.append(
                stats_by_seed[seed]["v2"][key][letter]
                - stats_by_seed[seed]["v4-hinge"][key][letter]
            )
        arr = np.asarray(deltas, dtype=np.float64)
        out[letter] = {
            "v2_mean_mae": float(np.mean(v2_maes)),
            "v4_mean_mae": float(np.mean(v4_maes)),
            "mean_delta_v2_minus_v4": float(arr.mean()),
            "std_delta": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
            "wins_v4": int((arr > 0).sum()),
            "n_pairs": int(arr.size),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "tracks/ksvd/runs")
    parser.add_argument("--v2-run-ids", nargs="+", required=True)
    parser.add_argument("--hinge-run-ids", nargs="+", required=True)
    parser.add_argument("--json", type=Path, default=Path("/tmp/multiseed_analysis.json"))
    args = parser.parse_args(argv)

    valid_excess, test_excess = load_excess_tables()
    stats_by_seed: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for model, run_ids in (("v2", args.v2_run_ids), ("v4-hinge", args.hinge_run_ids)):
        for run_id in run_ids:
            stats = per_run_stats(args.runs_root, run_id, model, valid_excess, test_excess)
            stats_by_seed.setdefault(int(stats["seed"]), {})[model] = stats
            rows.append(stats)
            print(
                f"[{model} seed={stats['seed']}] {run_id}: "
                f"valid={stats['valid_best_mae']:.6f}@{stats['best_valid_epoch']} "
                f"test(sel)={stats.get('test_mae_selection_checkpoint')} "
                f"test(refit)={stats.get('test_mae_refit')}"
            )
    table = pd.DataFrame(rows).sort_values(["model", "seed"])
    keep = [
        "model",
        "seed",
        "run_id",
        "valid_best_mae",
        "best_valid_epoch",
        "test_mae_selection_checkpoint",
        "test_mae_refit",
        "params",
        "training_time_s",
    ]
    print("\n=== per-seed table ===")
    print(table[keep].to_string(index=False))

    v2 = table[table["model"] == "v2"].sort_values("seed").reset_index(drop=True)
    v4 = table[table["model"] == "v4-hinge"].sort_values("seed").reset_index(drop=True)
    assert (v2["seed"].to_numpy() == v4["seed"].to_numpy()).all()

    test_deltas = v2["test_mae_selection_checkpoint"].to_numpy(dtype=float) - v4[
        "test_mae_selection_checkpoint"
    ].to_numpy(dtype=float)
    valid_deltas = v2["valid_best_mae"].to_numpy(dtype=float) - v4[
        "valid_best_mae"
    ].to_numpy(dtype=float)

    print("\n=== paired improvement (v2 - v4), positive = v4 better ===")
    print("test:", json.dumps(paired_stats(test_deltas), indent=2))
    print("valid:", json.dumps(paired_stats(valid_deltas), indent=2))

    test_sg = aggregate_subgroups(stats_by_seed, "test_subgroups")
    valid_sg = aggregate_subgroups(stats_by_seed, "valid_subgroups")
    print("\n=== test subgroup aggregated across seeds (A/B/C deltas positive = v4 better) ===")
    for letter in ("A", "B", "C"):
        s = test_sg[letter]
        print(
            f"{letter}: v2 {s['v2_mean_mae']:.4f} | v4 {s['v4_mean_mae']:.4f} | "
            f"delta {s['mean_delta_v2_minus_v4']:+.4f} +/- {s['std_delta']:.4f} "
            f"(v4 wins {s['wins_v4']}/{s['n_pairs']})"
        )

    print("\n=== seed-0 guard check vs canonical ===")
    guard: dict[str, Any] = {}
    for model, canon in CANONICAL.items():
        row = table[(table["model"] == model) & (table["seed"] == 0)].iloc[0]
        valid_ok = (
            abs(float(row["valid_best_mae"]) - canon[0]) < 1e-14
            and int(row["best_valid_epoch"]) == canon[1]
        )
        refit_val = row.get("test_mae_refit")
        refit_ok = (
            abs(float(refit_val) - canon[2]) < 1e-14
            if refit_val is not None and not pd.isna(refit_val)
            else None
        )
        print(
            f"{model}: valid {'OK' if valid_ok else 'MISMATCH'} "
            f"({row['valid_best_mae']:.16f} @{row['best_valid_epoch']}); "
            f"refit test {'OK' if refit_ok else ('n/a' if refit_ok is None else 'MISMATCH')} "
            f"({refit_val})"
        )
        guard[model] = {
            "valid_ok": bool(valid_ok),
            "refit_ok": refit_ok,
            "selection_test_mae": float(row["test_mae_selection_checkpoint"]),
            "canonical_selection_test_mae_ad_hoc": canon[3],
        }

    payload = {
        "per_seed": table.to_dict(orient="records"),
        "test_paired": paired_stats(test_deltas),
        "valid_paired": paired_stats(valid_deltas),
        "test_subgroups": test_sg,
        "valid_subgroups": valid_sg,
        "guard": guard,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
