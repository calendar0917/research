"""Freeze the protocol: one JSON with every constant, hash and split identity.

Written before the first formal run and never edited afterwards (the dose block
is added once, also before the first formal run).  ``--verify`` re-checks that the
frozen file still matches the code and the split payloads.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np

import lr_common as C
import lr_data as D
import lr_pairs as P

FREEZE = C.CONFIGS / "frozen_protocol.json"


def build() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "lcrc-lipo-route-frozen-protocol-v1",
        "round": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "version": C.VERSION,
        "created_utc": C.utcnow(),
        "design_anchor_commit": C.DESIGN_ANCHOR_COMMIT,
        "git_commit": C.git_commit(),
        "data": {
            "dataset": "ChemBench Lipophilicity (logD pH7.4)",
            "csv_sha256": C.LIPOP_CSV_SHA256,
            "split_file_sha256": C.SPLIT_FILE_SHA256,
            "train_rows": C.TRAIN_ROWS,
            "train_rows_sha256": C.TRAIN_ROWS_SHA256,
            "raw_artifact_sha256": C.RAW_ARTIFACT_SHA256,
            "pairs_csv_sha256": C.PAIRS_CSV_SHA256,
            "n_pairs": len(P.table()),
            "feature_spec_hash": C.FEATURE_SPEC_HASH,
            "audit_scope": C.AUDIT_SCOPE,
            "published_test_rows_used": False,
        },
        "splits": {
            s: {
                "status": D.split_payload(s)["status"],
                "sha256": C.sha256_file(D.SPLIT_FILES[s]),
                "partitions": {
                    p: {k: v for k, v in D.split_payload(s)["partitions"][p].items()
                        if k in ("n_rows", "n_pairs", "n_pair_molecules", "n_pair_components",
                                 "n_classes", "n_components", "row_ids_sha256")}
                    for p in C.PARTITIONS
                },
                "row_sha256": {p: C.row_sha256(D.rows_by_partition(s)[p]) for p in C.PARTITIONS},
            }
            for s in C.SPLITS + ("ft",)
        },
        "models": {
            "parameters": dict(C.EXPECTED_PARAMS),
            "names": dict(C.MODEL_NAMES),
            "deploy_eligible": list(C.DEPLOY_ELIGIBLE),
            "blocks_per_model": 2,
            "block_parameters": 25024,
            "neighbour_operator": "row-normalised mean over true undirected bonds, no self-loop, "
                                  "zero for isolated atoms, never across molecules",
        },
        "recipe": dict(C.RECIPE, adam_betas=list(C.RECIPE["adam_betas"])),
        "updates": {"formal": C.FORMAL_UPDATES, "sg": C.SG_UPDATES, "learning_curve": C.LC_UPDATES,
                    "full_train": C.FT_UPDATES, "lc_labels": dict(C.LC_LABELS),
                    "lc_steps": dict(C.LC_STEPS), "lc_epochs": dict(C.LC_EPOCHS)},
        "pair_stream": {
            "batch": C.PAIR_BATCH,
            "sampling": "uniform paired component, then uniform pair inside that component, with replacement",
            "null_min_non_mmp_fraction": P.NON_MMP_MIN_FRACTION,
            "null_max_repartner_attempts": P.MAX_REPARTNER_ATTEMPTS,
            "null_max_redraws": P.MAX_REDRAWS,
        },
        "dose": {"target_fraction": C.DOSE_TARGET_FRACTION, "clip": list(C.DOSE_CLIP),
                 "batches": C.DOSE_BATCHES, "probe_steps": list(C.DOSE_PROBE_STEPS),
                 "formula": "lambda = clip(0.2*median(gA)/max(median(gX),1e-12), 0.001, 10)"},
        "thresholds": {
            "purchase_min_gm": C.PURCHASE_MIN_GM, "purchase_edit_floor": C.PURCHASE_EDIT_FLOOR,
            "purchase_edit_fraction": C.PURCHASE_EDIT_FRACTION,
            "purchase_min_seeds": C.PURCHASE_MIN_SEEDS,
            "material_mol": C.MATERIAL_MOL, "material_edit_floor": C.MATERIAL_EDIT_FLOOR,
            "material_edit_fraction": C.MATERIAL_EDIT_FRACTION,
            "control_increment_mol": C.CONTROL_INCREMENT_MOL,
            "control_increment_edit": C.CONTROL_INCREMENT_EDIT,
            "pair_noninferior": C.PAIR_NONINFERIOR,
            "replication_rule": C.REPLICATION_RULE,
            "deploy_min_rmse_gain": C.DEPLOY_MIN_RMSE_GAIN,
            "deploy_min_positive_units": C.DEPLOY_MIN_POSITIVE_UNITS,
            "deploy_edit_slack": C.DEPLOY_EDIT_SLACK,
            "deploy_tie_tol": C.DEPLOY_TIE_TOL,
            "context_min_comparisons": C.CONTEXT_MIN_COMPARISONS,
            "min_edit_pairs": C.MIN_EDIT_PAIRS, "min_edit_components": C.MIN_EDIT_COMPONENTS,
            "min_fit_components": C.MIN_FIT_COMPONENTS,
        },
        "statistics": {"bootstrap_seed": C.BOOTSTRAP_SEED,
                       "bootstrap_replicates": C.BOOTSTRAP_REPLICATES},
        "budget": {"wall_target_hours": C.WALL_TARGET_HOURS, "wall_hard_hours": C.WALL_HARD_HOURS,
                   "gpu_soft_cap_hours": C.GPU_SOFT_CAP_HOURS,
                   "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS},
        "code_hashes": C.code_hashes(),
        "environment": C.env_info(),
        "rdkit": C.rdkit_version(),
    }
    payload["split_hash_combined"] = C.sha256_text(C.stable_json(
        {s: payload["splits"][s]["sha256"] for s in payload["splits"]}))
    plan_hashes = {}
    for split in C.SPLITS + ("ft",):
        steps = C.FT_UPDATES if split == "ft" else C.FORMAL_UPDATES
        for seed in C.SEEDS:
            plan_hashes[f"{split}_s{seed}"] = P.get_plan(split, seed, steps)["hash"]
    payload["pair_plan_hashes"] = plan_hashes
    payload["pair_plan_hash_combined"] = C.sha256_text(C.stable_json(plan_hashes))
    payload["matrix"] = {"path": C.track_rel(C.CONFIGS / "run_matrix.csv"),
                         "sha256": C.sha256_file(C.CONFIGS / "run_matrix.csv")
                         if (C.CONFIGS / "run_matrix.csv").exists() else None}
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="freeze the protocol constants")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    payload = build()
    if FREEZE.exists() and not args.force:
        old = C.read_json(FREEZE)
        diffs = [k for k in ("split_hash_combined", "pair_plan_hash_combined", "models", "thresholds",
                             "recipe", "updates", "data") if old.get(k) != payload.get(k)]
        if diffs:
            raise SystemExit(f"frozen protocol mismatch on {diffs}; refusing to overwrite")
        C.log("[freeze] frozen protocol still consistent")
        return 0
    C.write_json(FREEZE, payload)
    C.log(f"[freeze] wrote {C.track_rel(FREEZE)} combined_split={payload['split_hash_combined'][:12]} "
          f"plans={payload['pair_plan_hash_combined'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
