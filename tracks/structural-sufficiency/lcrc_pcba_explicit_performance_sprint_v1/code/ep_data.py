"""Unified PCBA data entry for every arm.

The frozen order plan, split, labels, normalization and metric are inherited
through ``sg_data`` / ``lr_data`` / ``pc_metrics``.  What is *not* inherited is
the per-batch feature assembly: this module owns ``augmented_collate`` and the
prediction / scoring path, so the A1/A2/A3 inputs are present identically in
training and Validation and can never be dropped by an old collate.

Only Train and Valid rows are ever collated.  Test rows have NaN labels, zero
A3 tokens and NaN A2 descriptors, and every evaluation entry point asserts the
row set before touching the cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ep_common as C
import ep_features as F
import pc_metrics as MET
import pc_model as BM
import sg_data as SD

PinnedStager = SD.PinnedStager


@dataclass
class Features:
    modules: tuple[str, ...]
    physchem: dict[str, Any] | None = None
    branch: dict[str, Any] | None = None
    hashes: dict[str, Any] = field(default_factory=dict)

    def as_audit(self) -> dict[str, Any]:
        out: dict[str, Any] = {"modules": list(self.modules), "hashes": dict(self.hashes)}
        if self.physchem is not None:
            meta = self.physchem["meta"]
            out["physchem"] = {
                "cache_sha256": meta["files"]["cache_sha256"],
                "algorithm": meta["algorithm"],
                "rdkit_version": meta["rdkit_version"],
                "built_rows": meta["built_rows"],
                "failed_rows": meta["failed_rows"],
                "coverage_train": meta["coverage_train"],
                "coverage_valid": meta["coverage_valid"],
                "norm_fingerprint": self.physchem["norm_fingerprint"],
            }
        if self.branch is not None:
            meta = self.branch["meta"]
            out["branch"] = {
                "cache_sha256": meta["files"]["cache_sha256"],
                "algorithm": meta["algorithm"],
                "n_buckets": meta["n_buckets"],
                "n_tokens": meta["n_tokens"],
                "collision_stats": meta["collision_stats"],
            }
        return out


def load_task() -> Any:
    """Frozen PCBA task with every inherited gate enforced."""
    return SD.load_task()


def load_features(task: Any, modules: Sequence[str]) -> Features:
    mods = tuple(modules)
    out = Features(modules=mods)
    if "A2" in mods:
        out.physchem = F.load_physchem(task)
        out.hashes["physchem_cache_sha256"] = out.physchem["feature_hash"]
        out.hashes["physchem_norm_fingerprint"] = out.physchem["norm_fingerprint"]
    if "A3" in mods:
        out.branch = F.load_branch(task)
        out.hashes["branch_cache_sha256"] = out.branch["feature_hash"]
    out.hashes["module_spec_sha256"] = C.digest_module_spec(mods)
    return out


def make_plan40(task: Any, *, seed: int = C.SEED) -> list[dict[str, Any]]:
    return SD.make_plan40(task, seed=int(seed))


def plan_summary(plan: Sequence[Mapping[str, Any]], task: Any) -> dict[str, Any]:
    return SD.plan_summary(plan, task)


def plan_prefix(plan: Sequence[Mapping[str, Any]], prefix_len: int) -> dict[str, Any]:
    return SD.plan_prefix_check(plan, int(prefix_len))


def task_counts(task: Any) -> dict[str, Any]:
    return SD.task_counts(task)


def _assert_rows(rows: np.ndarray, task: Any, name: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not rows.size:
        raise ValueError(f"{name}: empty/ill-shaped rows")
    if np.intersect1d(rows, task.test_rows).size:
        raise PermissionError(f"{name}: official Test row requested")
    return rows


def branch_tensors(feature: Mapping[str, Any], atom_ptr: np.ndarray, rows: Sequence[int],
                   atom_counts_row: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-batch aligned branch-token pointer and bucket ids (numpy)."""
    row_ptr = feature["row_token_ptr"]
    atom_counts = feature["atom_token_count"]
    tokens = feature["token_bucket"]
    rows_arr = np.asarray(rows, dtype=np.int64)
    counts = np.concatenate(
        [np.asarray(atom_counts[int(atom_ptr[r]) : int(atom_ptr[r + 1])], dtype=np.int64) for r in rows_arr]
    ) if rows_arr.size else np.zeros(0, dtype=np.int64)
    ptr = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    if int(counts.size) != int(atom_counts_row.sum()):
        raise RuntimeError(f"branch atom rows {int(counts.size)} != batch atoms {int(atom_counts_row.sum())}")
    parts = [tokens[int(row_ptr[r]) : int(row_ptr[r + 1])] for r in rows_arr]
    bucket = np.concatenate(parts).astype(np.int16) if parts else np.zeros(0, dtype=np.int16)
    if int(ptr[-1]) != int(bucket.size):
        raise RuntimeError(f"branch token pointer {int(ptr[-1])} != tokens {int(bucket.size)}")
    return torch.from_numpy(ptr), torch.from_numpy(bucket)


def augmented_collate(task: Any, rows: Sequence[int], features: Features | None = None) -> dict[str, Any]:
    """Frozen collate plus exactly the features this arm's modules read."""
    rows_arr = _assert_rows(np.asarray(rows, dtype=np.int64), task, "collate")
    batch = BM.collate(task.cache, rows_arr.tolist())
    if features is None or not features.modules:
        return batch
    if features.physchem is not None:
        matrix = features.physchem["matrix"]
        values = np.asarray(matrix[rows_arr], dtype=np.float32)
        if not np.isfinite(values).all():
            raise RuntimeError("A2 descriptor input contains a non-finite value for a Train/Valid row")
        batch["physchem"] = torch.from_numpy(np.ascontiguousarray(values))
    if features.branch is not None:
        ptr, bucket = branch_tensors(
            features.branch,
            np.asarray(task.cache["atom_ptr"], dtype=np.int64),
            rows_arr,
            np.asarray(batch["atom_counts"], dtype=np.int64),
        )
        batch["bt_ptr"] = ptr
        batch["bt_bucket"] = bucket
    return batch


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_logits(
    model: torch.nn.Module,
    task: Any,
    features: Features,
    rows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    rows = _assert_rows(rows, task, "predict_logits")
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = _to_device(augmented_collate(task, chunk, features), device)
            logits = model(batch)
            if tuple(logits.shape) != (len(chunk), C.N_TASKS):
                raise RuntimeError(f"model returned {tuple(logits.shape)} for {len(chunk)} rows")
            out.append(logits.detach().cpu().numpy().astype(np.float32))
    logits = np.concatenate(out, axis=0) if out else np.zeros((0, C.N_TASKS), dtype=np.float32)
    if logits.shape[0] != len(rows):
        raise RuntimeError(f"prediction rows {logits.shape[0]} != {len(rows)}")
    return logits


def _sig(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))).astype(np.float32)


def _array_sha(arr: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def evaluate_validation(
    model: torch.nn.Module, task: Any, features: Features, device: torch.device, *, reason: str = ""
) -> dict[str, Any]:
    logits = predict_logits(model, task, features, task.valid_rows, device, batch_size=256)
    y = np.asarray(task.labels[task.valid_rows])
    probs = _sig(logits)
    report = MET.validation_report(y, probs)
    report.update(
        {
            "reason": reason,
            "valid_bce": float(MET.masked_bce_numpy(logits, y)),
            "valid_logits_sha256": _array_sha(logits),
            "valid_probs_sha256": _array_sha(probs),
            "valid_logits_shape": list(logits.shape),
            "evaluated_rows": int(len(task.valid_rows)),
            "bce_and_ap_are_nan_masked": True,
            "feature_hashes": dict(features.hashes),
        }
    )
    return report


def evaluate_train(model: torch.nn.Module, task: Any, features: Features, device: torch.device) -> dict[str, Any]:
    t0 = time.perf_counter()
    logits = predict_logits(model, task, features, task.train_rows, device, batch_size=512)
    y = np.asarray(task.labels[task.train_rows])
    probs = _sig(logits)
    report = MET.validation_report(y, probs)
    report.update(
        {
            "reason": "fixed_final_epoch_train_eval",
            "train_bce": float(MET.masked_bce_numpy(logits, y)),
            "train_logits_sha256": _array_sha(logits),
            "evaluated_rows": int(len(task.train_rows)),
            "seconds": float(time.perf_counter() - t0),
            "feature_hashes": dict(features.hashes),
        }
    )
    return report


__all__ = [
    "Features",
    "PinnedStager",
    "augmented_collate",
    "branch_tensors",
    "evaluate_train",
    "evaluate_validation",
    "load_features",
    "load_task",
    "make_plan40",
    "plan_prefix",
    "plan_summary",
    "predict_logits",
    "task_counts",
]
