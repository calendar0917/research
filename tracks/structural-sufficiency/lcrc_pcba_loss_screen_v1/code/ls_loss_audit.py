"""CPU-side loss-definition audit and frozen task-count vector.

This writes ``analysis/loss_definition_audit.json``.  It uses the complete
official Train split to obtain ``n_t`` and never reads Validation/Test labels
for weights.  The numerical checks are independent of the GPU preflight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ls_common as C  # noqa: E402
import ls_losses as L  # noqa: E402
import lr_data  # noqa: E402


def run() -> dict:
    task = lr_data.load_task()
    train_rows = np.asarray(task.train_rows, dtype=np.int64)
    train_labels = task.labels[train_rows]
    n_task = L.task_counts(train_labels)
    train_observed = int(np.isfinite(train_labels).sum())
    train_positive = int((train_labels[np.isfinite(train_labels)] == 1.0).sum())
    checks = {
        "n_task_all_positive": bool(np.all(n_task > 0)),
        "n_task_sum_equals_train_observed": int(n_task.sum()) == train_observed,
        "n_task_sum_equals_data_audit": train_observed == int(task.data_audit["train_observed_entries"]),
        "train_rows_count": int(train_rows.size) == C.N_TRAIN,
        "train_rows_unique": int(np.unique(train_rows).size) == C.N_TRAIN,
        "n_tasks_128": int(n_task.size) == C.N_TASKS,
        "train_positive_matches_data_audit": train_positive
        == int(task.data_audit["train_positive_entries"]),
        "label_artifact_sha_matches": task.label_artifact_sha256
        == C.SOURCE["label_artifact_sha256"],
        "split_partition_sha_matches": task.split_partition_sha256
        == C.SOURCE["split_partition_sha256"],
        "feature_spec_hash_matches": task.spec.hash() == C.SOURCE["feature_spec_hash"],
        "norm_fingerprint_matches": task.norm["norm_fingerprint"] == C.SOURCE["norm_fingerprint"],
        "test_rows_all_nan": bool(not np.isfinite(task.labels[task.test_rows]).any()),
    }
    cpu = L.cpu_validation_suite()
    out = {
        "schema": "lcrc-pcba-loss-screen-loss-definition-audit-v1",
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "phase": "B_loss_definition",
        "formula": "L_T = N / (B*T) * sum_{i,t}(m_it * ell_it / n_t)",
        "formula_M": "L_M = mean over all observed entries of BCEWithLogits (historical pc_metrics.masked_bce_loss)",
        "constants": {
            "N_train_molecules": C.N_TRAIN,
            "T_tasks": C.N_TASKS,
            "B_batch": C.BATCH_SIZE,
            "N_frozen": C.N_TRAIN,
        },
        "n_task_source": "complete official Train split only; no Validation/Test labels",
        "n_task": n_task.tolist(),
        "task_weight_one_over_n_t": (1.0 / n_task.astype(np.float64)).tolist(),
        "n_task_sum": int(n_task.sum()),
        "n_task_min": int(n_task.min()),
        "n_task_max": int(n_task.max()),
        "n_task_mean": float(n_task.mean()),
        "n_task_sha256": C.sha256_array(n_task),
        "train_observed_entries": train_observed,
        "train_positive_entries": train_positive,
        "train_positive_rate": float(train_positive / train_observed),
        "data_identity": {
            "label_artifact_sha256": task.label_artifact_sha256,
            "split_partition_sha256": task.split_partition_sha256,
            "feature_spec_hash": task.spec.hash(),
            "norm_fingerprint": task.norm["norm_fingerprint"],
            "train_rows_sha256": C.sha256_int64(train_rows),
            "train_rows_sorted_sha256": C.sha256_int64(np.sort(train_rows)),
            "cache_identity": task.cache_identity,
        },
        "checks": {k: bool(v) for k, v in checks.items()},
        "cpu_validation": cpu,
        "all_passed": bool(all(checks.values()) and cpu["all_passed"]),
        "notes": [
            "The T loss adjusts only the weight of each task; it does not add "
            "positive-class weights, focal loss, resampling, or within-task balancing.",
            "n_t is frozen before training and is never recomputed from a batch.",
            "Invalid labels are replaced safely before the elementwise BCE and masked afterwards.",
        ],
    }
    C.write_json(C.ANALYSIS / "loss_definition_audit.json", out)
    C.log(
        f"loss audit all_passed={out['all_passed']} n_task_sum={out['n_task_sum']} "
        f"n_task_min={out['n_task_min']} n_task_max={out['n_task_max']}"
    )
    return out


if __name__ == "__main__":
    result = run()
    raise SystemExit(0 if result["all_passed"] else 1)
