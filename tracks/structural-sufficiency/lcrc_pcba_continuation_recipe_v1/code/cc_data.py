"""Read-only PCBA access and the new epoch41-80 order plan.

Training data still comes from the frozen ``sg_data`` path: same cache, same
official split, same Train-only normalization, same label artifact with Test
rows NaN, same official OGB AP.  This round adds only a newly generated,
L/R-shared order plan for epochs 41..80 and an explicitly gated reader for the
official Test labels used after the prediction freeze.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import cc_common as C
import sg_data as SD
import lr_data as LD
import pc_metrics as MET

OGB_ROOT = C.REPO / "data/ogb/ogbg_molpcba"
OGB_GRAPH_LABEL = OGB_ROOT / "raw/graph-label.csv.gz"

load_task = SD.load_task
task_counts = SD.task_counts
evaluate_validation = SD.evaluate_validation
evaluate_train = SD.evaluate_train
predict_rows = SD.predict_rows
PinnedStager = SD.PinnedStager


def load_and_gate() -> Any:
    """Load the frozen task and re-check every inherited identity."""
    task = SD.load_task()
    if task.labels.shape != (C.N_MOLECULES, C.N_TASKS):
        raise RuntimeError(f"label artifact shape {task.labels.shape}")
    if np.isfinite(task.labels[task.test_rows]).any():
        raise RuntimeError("official Test labels are exposed in the training artifact")
    checks = {
        "feature_spec_hash": task.spec.hash() == C.FEATURE_SPEC_HASH,
        "norm_fingerprint": task.norm["norm_fingerprint"] == C.NORM_FINGERPRINT,
        "split_partition_sha256": task.split_partition_sha256 == C.SPLIT_PARTITION_SHA256,
        "label_artifact_sha256": task.label_artifact_sha256 == C.LABEL_ARTIFACT_SHA256,
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"frozen data identity drifted: {failed}")
    return task


# ---------------------------------------------------------------------------
# epoch 41..80 order plan
# ---------------------------------------------------------------------------


def _rehash_epoch(record: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    rows = np.asarray(record["rows"], dtype=np.int64)
    batches = [np.asarray(b, dtype=np.int64) for b in record["batches"]]
    out = dict(record)
    out.update(
        {
            "epoch": int(epoch),
            "rows": rows,
            "batches": batches,
            "row_order_sha256": C.sha256_int64(rows),
            "batch_plan_sha256": C.sha256_batches(batches),
            "n_batches": int(len(batches)),
            "first_row": int(rows[0]),
            "last_row": int(rows[-1]),
            "first_batch_rows": [int(x) for x in batches[0]],
            "last_batch_size": int(len(batches[-1])),
        }
    )
    # The per-epoch generator state is inherited from the 80-epoch generation.
    out["generator_state_sha256"] = record["generator_state_sha256"]
    return out


def make_plan_41_80(task: Any, *, seed: int = C.SEED) -> list[dict[str, Any]]:
    """Generate the shared epoch-41..80 plan as a pure function of train rows.

    The plan is generated with a dedicated CPU ``torch.Generator`` seeded with
    ``seed`` and 80 ``randperm`` calls; epochs 1..40 are not used by this round
    and epochs 41..80 are renumbered.  The global dropout RNG is not touched.
    """
    before = torch.get_rng_state().clone()
    full = LD.make_order_plan(
        task.train_rows, batch_size=C.BATCH_SIZE, seed=int(seed), epochs=C.EPOCH80
    )
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("order-plan generation changed the global torch RNG state")
    if len(full) != C.EPOCH80:
        raise RuntimeError(f"plan has {len(full)} epochs, expected {C.EPOCH80}")
    plan = [_rehash_epoch(full[e - 1], e) for e in range(C.EPOCH40 + 1, C.EPOCH80 + 1)]
    if len(plan) != C.EPOCH80 - C.EPOCH40:
        raise RuntimeError("unexpected epoch-41..80 plan length")
    return plan


def plan_global_sha256(plan: Sequence[Mapping[str, Any]]) -> str:
    rows = [str(p["row_order_sha256"]) for p in plan]
    return C.sha256_bytes("|".join(rows).encode())


def plan_prefix_sha256(plan: Sequence[Mapping[str, Any]], consumed_epochs_41_80: int) -> str:
    n = int(consumed_epochs_41_80)
    rows = [str(p["row_order_sha256"]) for p in plan[:n]]
    return C.sha256_bytes("|".join(rows).encode())


def plan_summary(plan: Sequence[Mapping[str, Any]], task: Any, *, seed: int = C.SEED) -> dict[str, Any]:
    isolation = LD.assert_order_isolation(
        plan, task.train_rows, task.valid_rows, task.test_rows
    )
    if not isolation["passed"]:
        raise RuntimeError(f"epoch41-80 order isolation gate failed: {isolation}")
    detail = []
    for p in plan:
        detail.append(
            {
                "epoch": int(p["epoch"]),
                "n_batches": int(p["n_batches"]),
                "row_order_sha256": p["row_order_sha256"],
                "batch_plan_sha256": p["batch_plan_sha256"],
                "first_row": int(p["first_row"]),
                "last_row": int(p["last_row"]),
                "first_batch_rows": list(p["first_batch_rows"]),
                "last_batch_size": int(p["last_batch_size"]),
            }
        )
    return {
        "schema": C.SCHEMA_ORDER_PLAN,
        "round_id": C.ROUND_ID,
        "plan_kind": "new_epoch41_80_v1",
        "derivation": (
            "torch.Generator().manual_seed(seed) + torch.randperm(len(train_rows)) per "
            "epoch; 80 epochs generated, epochs 1-40 unused, epochs 41-80 used and "
            "renumbered.  Independent of the global torch/dropout RNG."
        ),
        "plan_seed": int(seed),
        "batch_size": int(C.BATCH_SIZE),
        "n_train_rows": int(task.train_rows.size),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "first_epoch": int(C.EPOCH40 + 1),
        "last_epoch": int(C.EPOCH80),
        "epochs": int(len(plan)),
        "global_plan_sha256": plan_global_sha256(plan),
        "epochs_detail": detail,
        "isolation": isolation,
        "global_torch_rng_unchanged": True,
        "test_rows_used": 0,
        "saved_utc": C.utcnow(),
        "git_commit": C.git_commit(),
    }


def write_plan(path: Path | str, summary: Mapping[str, Any]) -> str:
    C.write_json(path, dict(summary))
    return C.sha256_file(path)


def plan_for_checkpoint(plan: Sequence[Mapping[str, Any]], epoch: int) -> dict[str, Any]:
    consumed = int(epoch) - C.EPOCH40
    if consumed < 0 or consumed > len(plan):
        raise ValueError(f"epoch {epoch} outside the epoch41-80 plan")
    return {
        "plan_kind": "new_epoch41_80_v1",
        "plan_seed": C.SEED,
        "global_plan_sha256": plan_global_sha256(plan),
        "plan_sha256": plan_global_sha256(plan),
        "consumed_epochs": int(consumed),
        "consumed_through_epoch": int(epoch),
        "prefix_sha256": plan_prefix_sha256(plan, consumed),
        "first_epoch": int(C.EPOCH40 + 1),
        "last_epoch": int(C.EPOCH80),
        "batch_size": int(C.BATCH_SIZE),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
    }


# ---------------------------------------------------------------------------
# official Test label reader (only used after the prediction freeze)
# ---------------------------------------------------------------------------


def read_official_test_labels(task: Any, *, freeze_path: Path | str) -> dict[str, Any]:
    """Read ``raw/graph-label.csv.gz`` after verifying a completed freeze.

    This is the only code path in the round that opens the official Test
    targets.  It first verifies that the freeze manifest exists, declares
    ``test_labels_read: false`` and seals all prediction hashes.  The raw label
    file is then checked against the frozen Train/Valid artifact before any
    Test row is scored.
    """
    freeze_file = Path(freeze_path)
    if not freeze_file.is_file():
        raise RuntimeError("refusing to read official Test labels: prediction freeze is missing")
    freeze = C.read_json(freeze_file)
    if freeze.get("schema") != C.SCHEMA_TEST_FREEZE:
        raise RuntimeError("prediction freeze has the wrong schema")
    if freeze.get("test_labels_read") is not False:
        raise RuntimeError("prediction freeze does not declare test_labels_read=false")
    if freeze.get("seal_sha256") != _freeze_seal(freeze):
        raise RuntimeError("prediction freeze seal does not verify")
    if not OGB_GRAPH_LABEL.is_file():
        raise RuntimeError(f"official OGB label file missing: {OGB_GRAPH_LABEL}")

    rows: list[list[float]] = []
    with gzip.open(OGB_GRAPH_LABEL, "rt", newline="") as fh:
        for row in csv.reader(fh):
            rows.append([float("nan") if v == "" else float(v) for v in row])
    labels = np.asarray(rows, dtype=np.float32)
    if labels.shape != (C.N_MOLECULES, C.N_TASKS):
        raise RuntimeError(f"official label matrix {labels.shape} != {(C.N_MOLECULES, C.N_TASKS)}")

    # Alignment proof: the raw Train/Valid blocks must equal the frozen artifact.
    train = np.asarray(task.train_rows, dtype=np.int64)
    valid = np.asarray(task.valid_rows, dtype=np.int64)

    def _same(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(
            np.array_equal(np.nan_to_num(a, nan=-9.0), np.nan_to_num(b, nan=-9.0))
        )

    if not _same(labels[train], task.labels[train]):
        raise RuntimeError("raw OGB Train labels do not match the frozen label artifact")
    if not _same(labels[valid], task.labels[valid]):
        raise RuntimeError("raw OGB Valid labels do not match the frozen label artifact")
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    y_test = labels[test_rows]
    finite = np.isfinite(y_test)
    if not np.isin(y_test[finite], (0.0, 1.0)).all():
        raise RuntimeError("official Test labels are not binary 0/1")
    return {
        "labels": y_test,
        "path": C.track_rel(OGB_GRAPH_LABEL),
        "sha256": C.sha256_file(OGB_GRAPH_LABEL),
        "shape": list(y_test.shape),
        "observed_entries": int(finite.sum()),
        "positive_entries": int((y_test[finite] == 1.0).sum()),
        "negative_entries": int((y_test[finite] == 0.0).sum()),
        "test_rows_sha256": C.sha256_int64(test_rows),
        "train_valid_alignment_verified": True,
        "freeze_path": C.track_rel(freeze_file),
        "freeze_seal_sha256": freeze["seal_sha256"],
        "read_after_freeze": True,
        "read_utc": C.utcnow(),
    }


def _freeze_seal(freeze: Mapping[str, Any]) -> str:
    payload = {
        "logical_models": freeze.get("logical_models"),
        "predictions": freeze.get("predictions"),
        "test_rows_sha256": freeze.get("test_rows_sha256"),
        "selection_manifest_sha256": freeze.get("selection_manifest_sha256"),
    }
    return C.sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=C._json_default).encode()
    )


def freeze_seal(freeze: Mapping[str, Any]) -> str:
    return _freeze_seal(freeze)


__all__ = [
    "OGB_GRAPH_LABEL",
    "OGB_ROOT",
    "PinnedStager",
    "evaluate_train",
    "evaluate_validation",
    "freeze_seal",
    "load_and_gate",
    "load_task",
    "make_plan_41_80",
    "plan_for_checkpoint",
    "plan_global_sha256",
    "plan_prefix_sha256",
    "plan_summary",
    "predict_rows",
    "read_official_test_labels",
    "task_counts",
    "write_plan",
]
