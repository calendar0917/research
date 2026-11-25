"""Write the frozen machine-readable configuration of CSC v0.

The configuration is derived from :mod:`cs_common` plus the dataset identity
audit, so it can never drift from the code that defines the protocol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cs_common as C

C.setup_paths()

import cs_data as D  # noqa: E402


def build() -> dict:
    identity = {}
    plan = {}
    for dataset in C.DATASETS:
        bundle = D.load_bundle(dataset)
        audit = bundle.extra["identity_audit"]
        identity[dataset] = {
            "split_sizes": audit["split_sizes"],
            "split_partition_sha256": audit["split_partition_sha256"],
            "split_row_id_sha256": audit["split_row_id_sha256"],
            "cache_files": {k: {"expected_sha256": v.get("expected_sha256"), "actual_sha256": v.get("actual_sha256"), "match": v.get("match")} for k, v in audit["cache_files"].items()},
            "feature_spec_hash": audit["feature_spec_hash"],
            "norm_fingerprint": audit["norm_fingerprint"],
            "target_scaler": audit["target_scaler"],
            "relation_mapping": audit["relation_mapping"],
        }
        p = D.batch_order_plan(dataset)
        plan[dataset] = {"plan_sha256": p["plan_sha256"], "n_train": p["n_train"], "epochs": p["epochs"], "batch_size": p["batch_size"]}
    return {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in C.RECIPE.items()},
        "architecture": {
            "base": "lcrc_pcba_backbone_scaling_v1.code.bs_model.BSModel, config_for_scale(1)",
            "width": C.WIDTH,
            "block_hidden": C.BLOCK_HIDDEN,
            "gate_hidden": C.GATE_HIDDEN,
            "delta_scale": C.DELTA_SCALE,
            "n_relations": C.N_RELATIONS,
            "relation_names": list(C.RELATION_NAMES),
            "bond_type_to_relation": {str(k): int(v) for k, v in C.BOND_TYPE_TO_RELATION.items()},
            "n_out": C.N_OUT,
            "expected_parameters": {"A": 294135, "B": 294135, "C": 307095, "D": 307095},
            "expected_new_parameters": {"A": 34304, "B": 34304, "C": 47264, "D": 47264},
            "expected_base_parameters": 259831,
        },
        "arms": C.ARMS,
        "queue": {k: list(v) for k, v in C.QUEUE.items()},
        "queue_order_all_units": [f"{ds}:{arm}" for ds in C.DATASETS for arm in C.QUEUE[ds]],
        "analysis_constants": {
            "quadrature_nodes": C.QUADRATURE_NODES,
            "quadrature_nodes_fallback": C.QUADRATURE_NODES_FALLBACK,
            "trace_abs_tol": C.TRACE_ABS_TOL,
            "trace_rel_tol": C.TRACE_REL_TOL,
            "tree_max_depth": C.TREE_MAX_DEPTH,
            "tree_min_leaf": C.TREE_MIN_LEAF,
            "tree_random_state": C.TREE_RANDOM_STATE,
            "tree_top_codes": C.TREE_TOP_CODES,
            "bootstrap_replicates": C.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": C.BOOTSTRAP_SEED,
        },
        "budget": {"gpu_target_hours": C.GPU_TARGET_HOURS, "gpu_hard_hours": C.GPU_HARD_HOURS},
        "data_identity": identity,
        "batch_plans": plan,
        "reused_readonly": [
            "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/code/bs_model.py",
            "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/code/sg_model.py",
            "tracks/structural-query-learning/code/{sq_data,sq_lcrc}.py",
            "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/{feature_spec,lcrc_data,adapters}.py",
            "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/code/{cr_common,cr_data}.py",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.CONFIG_DIR / "experiment.json"))
    args = parser.parse_args(argv)
    payload = build()
    C.write_json(Path(args.out), payload)
    C.log(f"frozen config written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
