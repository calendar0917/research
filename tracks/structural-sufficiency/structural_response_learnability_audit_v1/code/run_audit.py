"""Run the structural-response learnability audit and write the three analysis files.

Usage (from the audit directory)::

    uv run --project /home/calendar/code/research python code/run_audit.py

The script is deterministic given the frozen inputs and settings.  It never
trains a neural model and never reads Valid/Test labels.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import audit_common as A  # noqa: E402
import audit_features as AF  # noqa: E402
import audit_response as AR  # noqa: E402
import audit_support as AS  # noqa: E402
import audit_transfer as AT  # noqa: E402


def main() -> int:
    t0 = time.time()
    inp = A.load_inputs()
    pairs = A.build_pair_table(inp)
    features = AF.EnvironmentFeatures(inp, pairs).matrix()

    support = AS.run(inp, pairs)
    response = AR.run(inp, pairs)
    transfer = AT.run(inp, pairs, features)

    A.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "support": A.ANALYSIS_DIR / "support.json",
        "response": A.ANALYSIS_DIR / "response_variation.json",
        "transfer": A.ANALYSIS_DIR / "transfer_check.json",
    }
    shas = {
        "support": A.write_json(paths["support"], support),
        "response": A.write_json(paths["response"], response),
        "transfer": A.write_json(paths["transfer"], transfer),
    }

    # ------------------------------------------------------------------
    # consistency assertions across modules
    # ------------------------------------------------------------------
    exact_response_n = sum(int(r["n_pairs"]) for r in response["types"]["exact"])
    coarse_response_n = sum(int(r["n_pairs"]) for r in response["types"]["coarse"])
    exact_ctx2 = support["thresholds_train_support"]["exact"]["context_full"]["2"]
    coarse_ctx2 = support["thresholds_train_support"]["coarse"]["context_full"]["2"]
    assert exact_response_n == exact_ctx2["n_pairs_train"] == 1288
    assert coarse_response_n == coarse_ctx2["n_pairs_train"] == 2664
    assert len(response["types"]["exact"]) == exact_ctx2["n_types"] == 378
    assert len(response["types"]["coarse"]) == coarse_ctx2["n_types"] == 351
    assert response["variance_decomposition"]["n_pairs"] == 1288
    assert response["variance_decomposition"]["n_types"] == 378
    assert response["noise_floor"]["max_pairs_per_exact_type_full_context"] == 1
    assert support["evaluation_coverage"]["cross"]["type_seen_ctx2"] == \
        transfer["subsets"]["cross"]["seen_type_ctx2"]["n_pairs"]
    assert support["evaluation_coverage"]["dev_dev"]["type_seen_ctx2"] == \
        transfer["subsets"]["dev_dev"]["seen_type_ctx2"]["n_pairs"]
    assert transfer["scopes"]["cross"]["n_pairs"] == 474
    assert transfer["scopes"]["dev_dev"]["n_pairs"] == 611
    assert transfer["scopes"]["combined"]["n_pairs"] == 1085
    assert transfer["bootstrap"]["reps"] == A.BOOTSTRAP_REPS
    assert pairs.checks["transform_consistent"] is True
    assert pairs.checks["environment_key_mismatch"] == 0
    assert features.checks["context_feature_mismatch"] == 0
    assert features.checks["features_finite"] is True
    assert not np.isfinite(inp.labels[~inp.official_train]).any()
    assert np.isfinite(inp.labels[inp.official_train]).all()

    # ------------------------------------------------------------------
    # engineering checks and manifest
    # ------------------------------------------------------------------
    checks = {
        "protocol_id": "structural-response-learnability-audit-v1",
        "generated_utc": A.now_utc(),
        "git_commit": A.git_commit(),
        "elapsed_seconds": float(time.time() - t0),
        "peak_rss_mb": A.peak_rss_mb(),
        "settings": {
            "bootstrap_reps": A.BOOTSTRAP_REPS,
            "bootstrap_seed": A.BOOTSTRAP_SEED,
            "shrink_k": A.SHRINK_K,
            "ridge_lambda": A.RIDGE_LAMBDA,
            "fingerprint_bits": A.FINGERPRINT_BITS,
            "fingerprint_iters": A.FINGERPRINT_ITERS,
        },
        "inputs": inp.checks,
        "pair_table": pairs.checks,
        "features": features.checks,
        "label_guard": {
            "official_train_n": int(inp.official_train.sum()),
            "n_finite_labels_outside_official_train": int(np.isfinite(inp.labels[~inp.official_train]).sum()),
            "test_labels_used": False,
            "valid_labels_used": False,
            "valid_used_for_model_selection": False,
            "label_policy": "non-Train label entries masked to NaN immediately after loading; never used",
        },
        "support": {
            "totals": support["totals"],
            "exact_ctx2": exact_ctx2,
            "exact_ctx3": support["thresholds_train_support"]["exact"]["context_full"]["3"],
            "exact_ctx5": support["thresholds_train_support"]["exact"]["context_full"]["5"],
            "coarse_ctx2": coarse_ctx2,
            "unique_units": support["unique_units_train_support"],
            "strict_independence": support["strict_independence_train_support"],
            "duplicates": support["independence"]["duplicates"],
            "cross_split": support["independence"]["cross_split"],
            "evaluation_coverage": support["evaluation_coverage"],
        },
        "response": {
            "n_exact_types": len(response["types"]["exact"]),
            "n_coarse_types": len(response["types"]["coarse"]),
            "n_exact_pairs": exact_response_n,
            "n_coarse_pairs": coarse_response_n,
            "variance_decomposition": response["variance_decomposition"]["models"],
            "within_type_context_explanation": response["within_type_context_explanation"],
            "noise_floor": response["noise_floor"],
            "naive_split_half_exact": response["diagnostics"]["naive_split_half_exact"],
        },
        "transfer": {
            "scopes": {
                name: {
                    "n_pairs": block["n_pairs"],
                    "n_clusters": block["n_clusters"],
                    "cluster_unit": block["cluster_unit"],
                    "mae": {k: block["point"][k]["mae"] for k in block["point"]},
                }
                for name, block in transfer["scopes"].items()
            },
            "models": transfer["models"],
            "evaluation_coverage": transfer["evaluation_coverage"],
            "paired_B_minus_A": transfer["paired_B_minus_A"],
            "paired_A_minus_global": transfer["paired_A_minus_global"],
            "bootstrap": transfer["bootstrap"],
        },
        "consistency": {
            "exact_response_pairs": exact_response_n,
            "exact_ctx2_pairs": exact_ctx2["n_pairs_train"],
            "coarse_response_pairs": coarse_response_n,
            "coarse_ctx2_pairs": coarse_ctx2["n_pairs_train"],
            "cross_seen_ctx2_transfer": transfer["subsets"]["cross"]["seen_type_ctx2"]["n_pairs"],
            "dev_dev_seen_ctx2_transfer": transfer["subsets"]["dev_dev"]["seen_type_ctx2"]["n_pairs"],
            "all_assertions_passed": True,
        },
        "outputs": {
            name: {
                "path": A.relative(path),
                "sha256": shas[name],
                "bytes": int(path.stat().st_size),
            }
            for name, path in paths.items()
        },
    }
    A.write_json(A.ANALYSIS_DIR / "checks.json", checks)

    print(json.dumps({
        "elapsed_seconds": checks["elapsed_seconds"],
        "peak_rss_mb": checks["peak_rss_mb"],
        "support_pairs": support["totals"]["n_pairs_analyzed"],
        "exact_ctx2": exact_ctx2,
        "cross_A_mae": transfer["scopes"]["cross"]["point"]["A_lomo"]["mae"],
        "cross_B_mae": transfer["scopes"]["cross"]["point"]["B"]["mae"],
        "dev_dev_A_mae": transfer["scopes"]["dev_dev"]["point"]["A_lomo"]["mae"],
        "dev_dev_B_mae": transfer["scopes"]["dev_dev"]["point"]["B"]["mae"],
        "outputs": checks["outputs"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())