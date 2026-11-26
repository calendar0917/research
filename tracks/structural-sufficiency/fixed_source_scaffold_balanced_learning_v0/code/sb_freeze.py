"""Freeze the round: scaffolds, weights, weight gate and the machine-readable config.

Runs *before* any training.  Reads only official Train/Valid structures (plus the
Train label column for nothing: the weights never see a label), writes

    analysis/scaffold_distribution.json
    analysis/weight_summary.json
    analysis/scaffold_weights.csv
    analysis/freeze_log.json
    configs/experiment.json

and exits with code 4 (without training) if the frozen weight-contrast gate says
``INSUFFICIENT_WEIGHT_CONTRAST``.
"""

from __future__ import annotations

import time

import numpy as np

import sb_common as C
import sb_data as D
import sb_scaffold as S


def main() -> int:
    t0 = time.perf_counter()
    official = S.load_official_rows()
    train_rows = np.asarray(official["train_rows"], dtype=np.int64)
    valid_rows = np.asarray(official["valid_rows"], dtype=np.int64)

    t_scaf = time.perf_counter()
    scaffold_by_row = S.scaffold_of_rows(official["smiles"], np.concatenate([train_rows, valid_rows]))
    scaffold_seconds = time.perf_counter() - t_scaf

    k_train = len({scaffold_by_row[int(r)] for r in train_rows})
    dist = S.scaffold_distribution(scaffold_by_row, train_rows, valid_rows)
    weights_by_row = S.balanced_weights(
        scaffold_by_row, train_rows, n_train=int(train_rows.size), k_train=int(k_train)
    )
    audit = S.weight_audit(scaffold_by_row, train_rows, weights_by_row)
    gate = S.weight_gate(dist, audit)

    # ---- data identity (bundle side, no Test metrics)
    bundle, identity = D.load_bundle()
    canon = D.canonical_check(bundle, official)
    if not canon["ok"]:
        raise RuntimeError(f"cache/official SMILES mismatch: {canon}")
    wpack = D.train_weights(bundle, official, scaffold_by_row, k_train=int(k_train))
    if wpack["sha256"] != audit["weight_sha256"]:
        raise RuntimeError("cache-aligned weight vector differs from the row-id weight vector")
    plan = D.batch_order_plan(C.EPOCHS)
    if plan["plan_sha256"] != C.HIST["train_plan_sha256"]:
        raise RuntimeError(f"batch-order plan {plan['plan_sha256']} != historical {C.HIST['train_plan_sha256']}")

    rows_out = []
    for idx in range(wpack["weights"].size):
        rows_out.append(
            {
                "cache_index": int(idx),
                "row_id": int(wpack["row_ids"][idx]),
                "split": "train",
                "scaffold": str(wpack["scaffold_ids"][idx]),
                "scaffold_size": int(wpack["scaffold_sizes"][idx]),
                "weight": float(wpack["weights"][idx]),
            }
        )
    C.write_csv(
        C.ANALYSIS_DIR / "scaffold_weights.csv",
        rows_out,
        ["cache_index", "row_id", "split", "scaffold", "scaffold_size", "weight"],
    )

    valid_scaffolds = [
        {"row_id": int(r), "scaffold": scaffold_by_row[int(r)]} for r in valid_rows
    ]
    C.write_json(C.ANALYSIS_DIR / "valid_scaffolds.json", {"valid": valid_scaffolds, "n": len(valid_scaffolds)})

    C.write_json(
        C.ANALYSIS_DIR / "scaffold_distribution.json",
        {
            "created_utc": C.utcnow(),
            "round_id": C.ROUND_ID,
            "data_identity": {
                "mol_csv_sha256": official["mol_csv_sha256"],
                "split_sha256": official["split_sha256"],
                "test_split_file_opened": False,
                "sfl_scaffold_rule_sha256": official["sfl_scaffold_sha256"],
                "cache_vs_official_smiles": canon,
            },
            "distribution": dist,
            "weights": audit,
            "gate": gate,
        },
    )
    C.write_json(
        C.ANALYSIS_DIR / "weight_summary.json",
        {
            "created_utc": C.utcnow(),
            "rule": C.WEIGHT_RULE,
            "n_train": int(train_rows.size),
            "k_train": int(k_train),
            "weights": audit,
            "gate": gate,
            "changed_relative_to_uniform": {
                "n_molecules_weight_above_1": int((wpack["weights"] > 1.0).sum()),
                "n_molecules_weight_below_1": int((wpack["weights"] < 1.0).sum()),
                "n_molecules_weight_equal_1": int((wpack["weights"] == 1.0).sum()),
                "molecules_in_singleton_scaffolds": int((wpack["scaffold_sizes"] == 1).sum()),
                "weight_of_singleton_scaffold": float(wpack["weights"][wpack["scaffold_sizes"] == 1][0])
                if (wpack["scaffold_sizes"] == 1).any()
                else None,
                "min_weight_scaffold": int(wpack["scaffold_sizes"].max()),
            },
        },
    )

    config = {
        "created_utc": C.utcnow(),
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "seed": C.SEED,
        "arm": "SCAFFOLD_BALANCED",
        "splits_used": list(C.ALLOWED_SPLITS),
        "test_policy": "NOT_READ_NOT_SCORED",
        "reused_readonly": {
            "model": "tracks/structural-sufficiency/context_conditioned_structural_computation_v0/code/cs_model.py (arm A, FIXED1)",
            "recipe": "context_conditioned_structural_computation_v0/code/cs_common.py::RECIPE",
            "data": "context_conditioned_structural_computation_v0/code/cs_data.py + lcrc_readout_context_pilot_small_v1 + tracks/structural-query-learning",
            "scaffold_rule": "lcrc_structural_failure_localization_v0/code/sfl_common.py::scaffold_key",
            "historical_predictions": C.HIST["run_dir"],
        },
        "recipe": C.recipe_of_record(k_train=int(k_train), weight_sha256=wpack["sha256"]),
        "weights": {
            "rule": C.WEIGHT_RULE,
            "n_train": int(train_rows.size),
            "k_train": int(k_train),
            "sha256": wpack["sha256"],
            "audit": audit,
            "gate": gate,
        },
        "batch_plan": {
            "batch_size": int(C.RECIPE["batch_size"]),
            "epochs": int(C.EPOCHS),
            "n_train": int(C.N_TRAIN),
            "plan_sha256": plan["plan_sha256"],
            "matches_historical_fixed1_plan": bool(plan["plan_sha256"] == C.HIST["train_plan_sha256"]),
        },
        "historical_baseline": C.HIST,
        "descriptive_references": {"C1": C.C1_REF, "GINE": C.GINE_REF},
        "verdict_rules": C.VERDICT,
        "analysis_constants": {
            "bootstrap_replicates": C.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": C.BOOTSTRAP_SEED,
            "morgan_radius": C.MORGAN_RADIUS,
            "morgan_bits": C.MORGAN_BITS,
            "trim_fraction": C.TRIM_FRACTION,
            "support_threshold": "Train leave-one-out max-Tanimoto median (fixed, not searched)",
        },
        "budget": {
            "gpu_hours_hard": 0.25,
            "gpu_hours_used": 0.0,
            "wall_clock_hard_minutes": 40,
            "trainer_hard_seconds": 2100.0,
            "device": "cpu (no local CUDA device)",
        },
        "freeze_seconds": float(time.perf_counter() - t0),
        "scaffold_seconds": float(scaffold_seconds),
    }
    C.write_json(C.CONFIG_DIR / "experiment.json", config)
    C.write_json(
        C.ANALYSIS_DIR / "freeze_log.json",
        {
            "created_utc": C.utcnow(),
            "seconds": float(time.perf_counter() - t0),
            "scaffold_seconds": float(scaffold_seconds),
            "k_train": int(k_train),
            "n_train": int(train_rows.size),
            "gate": gate,
            "uniform_gate_pass": bool(not gate["insufficient_weight_contrast"]),
            "batch_plan_sha256": plan["plan_sha256"],
            "weight_sha256": wpack["sha256"],
        },
    )
    C.log(
        f"frozen K={k_train} N={train_rows.size} max/min weight="
        f"{gate['max_over_min_weight']:.2f} ESS/N={gate['ess_fraction']:.3f} "
        f"insufficient={gate['insufficient_weight_contrast']}"
    )
    return 4 if gate["insufficient_weight_contrast"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
