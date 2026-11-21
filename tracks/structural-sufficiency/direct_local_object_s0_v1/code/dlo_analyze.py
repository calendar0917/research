"""Post-processing: main table, fixed deltas, paired bootstrap, manifest.

No retraining, no new seeds, no alternative checkpoints.  Reference numbers are
the frozen values from the previous round; their cached predictions are reopened
only to verify row identity and recompute the same RMSE.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import dlo_common as C

OLD_RUNS = {
    "WORD_PREFIX": ("lipo_WORD_PREFIX_s0", 160),
    "BAG_CONT": ("lipo_BAG_CONT_s0", 120),
    "GINE_RAW_REF": ("lipo_GINE_RAW_REF", 240),
}
OLD_PARAMS = {
    "WORD_PREFIX": 613033,
    "BAG_CONT": 808810,
    "GINE_RAW_REF": 387974,
}
OLD_PARAMS_TOTAL_ALLOCATED = {
    "WORD_PREFIX": 808810,
    "BAG_CONT": 808810,
    "GINE_RAW_REF": 387974,
}


def load_run(arm_dir: Path) -> dict[str, Any]:
    run = C.read_json(arm_dir / "run.json")
    run["pred_best"] = dict(np.load(arm_dir / "predictions_best.npz"))
    run["pred_final"] = dict(np.load(arm_dir / "predictions_final.npz"))
    ipath = arm_dir / "intervention.json"
    run["intervention"] = C.read_json(ipath) if ipath.is_file() else None
    return run


def reference_check(cfg: Mapping[str, Any], valid_rows: np.ndarray, test_rows: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, rel in cfg["old_results"].items():
        path = C.REPO_ROOT / rel
        if not path.is_file():
            out[name] = {"readable": False, "used_disclosed": cfg["reference_values"][name]}
            continue
        z = dict(np.load(path))
        record: dict[str, Any] = {"readable": True, "path": str(path)}
        ok = True
        for split, rows, key in (("valid", valid_rows, "valid"), ("test", test_rows, "test")):
            ids = z[f"{split}_row_id"].astype(np.int64)
            ok = ok and np.array_equal(ids, rows)
            pred = z[f"{split}_pred"].astype(np.float64)
            label = z[f"{split}_label"].astype(np.float64)
            record[f"{split}_rmse_recomputed"] = C.rmse(pred, label)
            record[f"{split}_mae_recomputed"] = C.mae(pred, label)
            record[f"{split}_rows_match"] = bool(np.array_equal(ids, rows))
        disclosed = cfg["reference_values"][name]
        record["valid_matches_disclosed"] = bool(abs(record["valid_rmse_recomputed"] - disclosed["valid_rmse"]) < 1e-9)
        record["test_matches_disclosed"] = bool(abs(record["test_rmse_recomputed"] - disclosed["test_rmse"]) < 1e-9)
        record["rows_match"] = bool(ok and record["valid_rows_match"] and record["test_rows_match"])
        out[name] = record
    return out


def paired_bootstrap(pred_a: np.ndarray, pred_b: np.ndarray, labels: np.ndarray,
                     n_boot: int = 1000, seed: int = 20261122, shared_idx: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    n = labels.size
    diff = np.empty(n_boot, np.float64)
    for b in range(n_boot):
        idx = shared_idx[b] if shared_idx is not None else rng.integers(0, n, n)
        diff[b] = C.rmse(pred_a[idx], labels[idx]) - C.rmse(pred_b[idx], labels[idx])
    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "median": float(np.median(diff)),
        "p2_5": float(np.quantile(diff, 0.025)),
        "p97_5": float(np.quantile(diff, 0.975)),
        "fraction_a_worse": float(np.mean(diff > 0)),
        "mean": float(np.mean(diff)),
    }


def decision(e_L: float, e_S: float, e_P: float, e_G: float) -> dict[str, Any]:
    if e_L <= 0.67 and (e_S - e_L) >= 0.015:
        code = "LARGER_OBJECT_PROMISING"
    elif min(e_L, e_S) <= 0.67:
        code = "DIRECT_LOCAL_PREDICTION_PROMISING_SCOPE_UNRESOLVED"
    elif min(e_L, e_S) > 0.67 and (e_S - e_L) >= 0.015 and e_L < e_P - 0.01:
        code = "CONTEXT_SIGNAL_WITH_PREDICTION_GAP"
    elif min(e_L, e_S) >= e_P - 0.01:
        code = "STOP_THIS_DIRECT_FAMILY"
    else:
        code = "MODEST_LOCAL_SIGNAL_STOP"
    return {
        "code": code,
        "inputs": {
            "e_L_valid": float(e_L),
            "e_S_valid": float(e_S),
            "e_P_valid": float(e_P),
            "e_G_valid": float(e_G),
            "min_valid": float(min(e_L, e_S)),
            "e_S_minus_e_L": float(e_S - e_L),
            "e_P_minus_e_L": float(e_P - e_L),
            "e_G_minus_e_L": float(e_G - e_L),
        },
        "note": "pre-registered investment gate, not a significance or equivalence test",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(C.CONFIG_PATH))
    ap.add_argument("--cache-stats", default=str(C.data_dir() / "dlo_cache_stats.json"))
    ap.add_argument("--checks", default=str(C.analysis_dir() / "checks.json"))
    ap.add_argument("--results", default=str(C.results_root()))
    ap.add_argument("--out", default=str(C.analysis_dir()))
    args = ap.parse_args()
    cfg = C.read_json(args.config)
    cache_stats = C.read_json(args.cache_stats)
    checks = C.read_json(args.checks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = Path(args.results)
    runs = {
        C.ARM_S56: load_run(results / "lipo_DIRECT_S56_s0"),
        C.ARM_W34: load_run(results / "lipo_MATCHED_W34_s0"),
    }
    valid_rows = runs[C.ARM_S56]["pred_best"]["valid_row_id"].astype(np.int64)
    test_rows = runs[C.ARM_S56]["pred_best"]["test_row_id"].astype(np.int64)
    for arm in C.ARMS:
        if not np.array_equal(runs[arm]["pred_best"]["valid_row_id"].astype(np.int64), valid_rows):
            raise ValueError("Valid row ids differ between the two arms")
        if not np.array_equal(runs[arm]["pred_best"]["test_row_id"].astype(np.int64), test_rows):
            raise ValueError("Test row ids differ between the two arms")

    refs = reference_check(cfg, valid_rows, test_rows)
    refs_values = cfg["reference_values"]

    # ------------------------------------------------------------------ table
    table: list[dict[str, Any]] = []
    for arm in C.ARMS:
        run = runs[arm]
        best = run["best"]
        rec = best.get("recomputed", {})
        table.append({
            "arm": arm,
            "role": "new_direct_model",
            "parameters": int(run["parameters"]["total"]),
            "parameters_total_allocated": int(run["parameters"]["total"]),
            "best_epoch": int(run["best_epoch"]),
            "horizon": int(run["horizon"]),
            "completed": bool(run.get("completed", False)),
            "status": run.get("status"),
            "train_rmse_best_valid": float(best["train_rmse"]),
            "valid_rmse_best_valid": float(best["valid_rmse"]),
            "test_rmse_best_valid": float(best["test_rmse"]),
            "train_mae_best_valid": float(best["train_mae"]),
            "valid_mae_best_valid": float(best["valid_mae"]),
            "test_mae_best_valid": float(best["test_mae"]),
            "train_rmse_final": float(run["final"]["train_rmse"]),
            "valid_rmse_final": float(run["final"]["valid_rmse"]),
            "test_rmse_final": float(run["final"]["test_rmse"]),
            "wall_seconds": float(run["wall_seconds_total"]),
            "gpu_hours": float(run["gpu_hours"]),
            "max_vram_mb": float(run["max_vram_mb"]),
            "cache_sha256": run.get("cache_sha256"),
            "recomputed_best_valid_rmse": float(rec.get("valid", {}).get("rmse", float("nan")))
            if rec else None,
        })
    for name in ("WORD_PREFIX", "BAG_CONT", "GINE_RAW_REF"):
        run_dir = C.REPO_ROOT / "tracks" / "structural-sufficiency" / "structural_word_functions_s0_v1" / "results" / OLD_RUNS[name][0]
        old = C.read_json(run_dir / "run.json")
        best = old.get("best", {})
        final: dict[str, Any] = {}
        curve_path = run_dir / "curves.csv"
        if curve_path.is_file():
            with curve_path.open(encoding="utf-8") as handle:
                last = list(csv.DictReader(handle))[-1]
            final = {
                "epoch": int(last.get("stage_epoch", last.get("epoch", 0))),
                "train_rmse": float(last["train_rmse"]),
                "valid_rmse": float(last["valid_rmse"]),
                "test_rmse": float(last["test_rmse"]),
                "train_mae": float(last["train_mae"]),
                "valid_mae": float(last["valid_mae"]),
                "test_mae": float(last["test_mae"]),
            }
        table.append({
            "arm": name,
            "role": "read_only_reference",
            "parameters": OLD_PARAMS[name],
            "parameters_total_allocated": OLD_PARAMS_TOTAL_ALLOCATED[name],
            "best_epoch": old.get("best_epoch"),
            "horizon": OLD_RUNS[name][1],
            "completed": True,
            "status": "read_only",
            "train_rmse_best_valid": best.get("train_rmse"),
            "valid_rmse_best_valid": refs_values[name]["valid_rmse"],
            "test_rmse_best_valid": refs_values[name]["test_rmse"],
            "train_mae_best_valid": best.get("train_mae"),
            "valid_mae_best_valid": best.get("valid_mae"),
            "test_mae_best_valid": best.get("test_mae"),
            "train_rmse_final": final.get("train_rmse"),
            "valid_rmse_final": final.get("valid_rmse"),
            "test_rmse_final": final.get("test_rmse"),
            "train_mae_final": final.get("train_mae"),
            "valid_mae_final": final.get("valid_mae"),
            "test_mae_final": final.get("test_mae"),
            "wall_seconds": old.get("wall_seconds"),
            "gpu_hours": old.get("gpu_hours"),
            "max_vram_mb": old.get("max_vram_mb"),
            "cache_sha256": None,
            "recomputed_best_valid_rmse": refs.get(name, {}).get("valid_rmse_recomputed"),
        })
    C.write_csv(out_dir / "main_table.csv", table)

    # ---------------------------------------------------------------- deltas
    e_L = float(runs[C.ARM_S56]["best"]["valid_rmse"])
    e_S = float(runs[C.ARM_W34]["best"]["valid_rmse"])
    t_L = float(runs[C.ARM_S56]["best"]["test_rmse"])
    t_S = float(runs[C.ARM_W34]["best"]["test_rmse"])
    e_P = float(refs_values["WORD_PREFIX"]["valid_rmse"])
    t_P = float(refs_values["WORD_PREFIX"]["test_rmse"])
    e_G = float(refs_values["GINE_RAW_REF"]["valid_rmse"])
    t_G = float(refs_values["GINE_RAW_REF"]["test_rmse"])
    deltas = {
        "valid": {
            "e_L_minus_e_S": e_L - e_S,
            "e_L_minus_e_P": e_L - e_P,
            "e_L_minus_e_G": e_L - e_G,
            "e_S_minus_e_P": e_S - e_P,
            "e_S_minus_e_G": e_S - e_G,
        },
        "test": {
            "t_L_minus_t_S": t_L - t_S,
            "t_L_minus_t_P": t_L - t_P,
            "t_L_minus_t_G": t_L - t_G,
            "t_S_minus_t_P": t_S - t_P,
            "t_S_minus_t_G": t_S - t_G,
        },
    }

    # -------------------------------------------------------------- bootstrap
    shared = np.random.default_rng(20261122).integers(0, valid_rows.size, size=(1000, valid_rows.size))
    boot_valid = paired_bootstrap(
        runs[C.ARM_S56]["pred_best"]["valid_pred"], runs[C.ARM_W34]["pred_best"]["valid_pred"],
        runs[C.ARM_S56]["pred_best"]["valid_label"], shared_idx=shared,
    )
    shared_test = np.random.default_rng(20261122).integers(0, test_rows.size, size=(1000, test_rows.size))
    boot_test = paired_bootstrap(
        runs[C.ARM_S56]["pred_best"]["test_pred"], runs[C.ARM_W34]["pred_best"]["test_pred"],
        runs[C.ARM_S56]["pred_best"]["test_label"], shared_idx=shared_test,
    )

    # ---------------------------------------------------------------- solvency
    both_complete = all(bool(runs[a].get("completed", False)) and runs[a].get("status") == "completed" for a in C.ARMS)
    engineering = not (both_complete and bool(checks.get("all_pass", False)))
    if engineering:
        decision_payload = {
            "code": "ENGINEERING_OR_BUDGET_INCOMPLETE",
            "inputs": {"e_L_valid": e_L, "e_S_valid": e_S},
            "note": "performance gate not applied; see run status and checks",
            "both_complete": both_complete,
            "checks_pass": bool(checks.get("all_pass", False)),
        }
    else:
        decision_payload = decision(e_L, e_S, e_P, e_G)

    split_conflict: list[dict[str, Any]] = []
    for arm in C.ARMS:
        run = runs[arm]
        v = float(run["best"]["valid_rmse"])
        t = float(run["best"]["test_rmse"])
        for ref_name, rv, rt in (
            ("WORD_PREFIX", e_P, t_P),
            ("other_new_arm", e_S if arm == C.ARM_S56 else e_L, t_S if arm == C.ARM_S56 else t_L),
        ):
            if v < rv - 0.01 and t > rt + 0.03:
                split_conflict.append({
                    "arm": arm,
                    "comparison": ref_name,
                    "valid_gain": v - rv,
                    "test_loss": t - rt,
                })

    summary = {
        "protocol_id": cfg["protocol_id"],
        "decision": decision_payload,
        "metrics": {
            "e_L_valid": e_L, "e_S_valid": e_S, "e_P_valid": e_P, "e_G_valid": e_G,
            "t_L_test": t_L, "t_S_test": t_S, "t_P_test": t_P, "t_G_test": t_G,
        },
        "deltas": deltas,
        "bootstrap": {
            "valid": boot_valid,
            "test": boot_test,
            "comparison": "DIRECT_S56 - MATCHED_W34, molecule-level RMSE difference",
            "note": "sample-composition sensitivity at the selected weights; not cross-seed uncertainty; same resample indices used for both models",
        },
        "split_conflict": split_conflict,
        "references": {
            "values": refs_values,
            "replay": refs,
        },
        "interventions": {arm: runs[arm]["intervention"] for arm in C.ARMS},
        "runs": {
            arm: {
                "run_id": runs[arm]["run_id"],
                "best_epoch": runs[arm]["best_epoch"],
                "horizon": runs[arm]["horizon"],
                "completed": runs[arm].get("completed"),
                "status": runs[arm].get("status"),
                "wall_seconds_total": runs[arm]["wall_seconds_total"],
                "gpu_hours": runs[arm]["gpu_hours"],
                "parameters": runs[arm]["parameters"],
                "init_state_hash": runs[arm].get("init_state_hash"),
                "cache_sha256": runs[arm].get("cache_sha256"),
            }
            for arm in C.ARMS
        },
    }
    C.write_json(out_dir / "summary.json", summary)

    # ------------------------------------------------------------- structure
    structure = {
        "raw_cache_sha256": cache_stats["raw_cache_sha256"],
        "old_sent_cache_sha256": cache_stats["old_sent_cache_sha256"],
        "old_base_cache_sha256": cache_stats["old_base_cache_sha256"],
        "new_cache_sha256": cache_stats["cache_sha256"],
        "new_cache_bytes": cache_stats["cache_bytes"],
        "n_molecules": cache_stats["n_molecules"],
        "splits": cache_stats["split_counts"],
        "luts": cache_stats["luts"],
        "arms": cache_stats["arms"],
        "target_scaler": cache_stats["target_scaler"],
        "notes": {
            "core": "5/6-atom unique (root, atom_set) sentences for DIRECT_S56; 3/4-atom words for MATCHED_W34",
            "padding": "complete connected components of size 1-4 (S56) / 1-2 (W34), every component atom as root",
            "frames": "all root-fixed permutations that realise the minimal adjacency bit; weights 1/n_frames per object",
            "dedup": "global exact dedup of the complete raw frame input; object rows are never deduplicated",
            "objects_per_split": "object rows whose molecule belongs to the split",
        },
    }
    C.write_json(out_dir / "structure_stats.json", structure)

    # -------------------------------------------------------------- manifest
    manifest = {
        "protocol_id": cfg["protocol_id"],
        "config": str(Path(args.config)),
        "config_sha256": C.sha256_file(args.config),
        "cache": args.cache_stats,
        "cache_sha256": cache_stats["cache_sha256"],
        "checks": args.checks,
        "checks_sha256": C.sha256_file(args.checks),
        "t0_local": cfg["budget"]["t0_local"],
        "hard_wall_minutes": cfg["budget"]["hard_wall_minutes"],
        "stop_launching_after_minutes": cfg["budget"]["stop_launching_after_minutes"],
        "stop_training_after_minutes": cfg["budget"]["stop_training_after_minutes"],
        "reserve_minutes": cfg["budget"]["reserve_minutes"],
        "actual_horizon_common": runs[C.ARM_S56]["horizon"] if runs[C.ARM_S56]["horizon"] == runs[C.ARM_W34]["horizon"] else None,
        "runs": {
            arm: {
                "run_id": runs[arm]["run_id"],
                "arm": arm,
                "run_json": str(results / ("lipo_DIRECT_S56_s0" if arm == C.ARM_S56 else "lipo_MATCHED_W34_s0") / "run.json"),
                "regime": runs[arm].get("regime"),
                "command": runs[arm].get("command"),
                "init_state_hash": runs[arm].get("init_state_hash"),
                "parameters": runs[arm].get("parameters"),
                "wall_seconds_total": runs[arm]["wall_seconds_total"],
                "gpu_hours": runs[arm]["gpu_hours"],
                "max_vram_mb": runs[arm]["max_vram_mb"],
                "stage_a_cost": runs[arm].get("stage_a_cost"),
                "status": runs[arm].get("status"),
                "completed": runs[arm].get("completed"),
            }
            for arm in C.ARMS
        },
        "git_commit": C.git_commit(),
        "tf32": runs[C.ARM_S56].get("tf32"),
        "init_parameter_hash_equal": bool(
            runs[C.ARM_S56].get("init_parameter_hash") is not None
            and runs[C.ARM_S56].get("init_parameter_hash") == runs[C.ARM_W34].get("init_parameter_hash")
        ),
        "init_parameter_hash": runs[C.ARM_S56].get("init_parameter_hash"),
        "init_buffer_hash_note": "run.json init_state_hash also covers the arm-specific mu_bucket buffer; the trainable parameter hash is identical",
    }
    provenance = out_dir / "rr_provenance.json"
    if provenance.is_file():
        manifest["rr_provenance"] = C.read_json(provenance)
    C.write_json(out_dir / "run_manifest.json", manifest)

    print(json.dumps({
        "table": str(out_dir / "main_table.csv"),
        "decision": decision_payload["code"],
        "e_L": e_L, "e_S": e_S,
        "deltas": deltas,
        "bootstrap_valid": boot_valid,
        "bootstrap_test": boot_test,
        "split_conflict": split_conflict,
    }, indent=2))


if __name__ == "__main__":
    main()
