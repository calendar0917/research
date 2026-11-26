"""Data layer: the frozen CSC v0 bundle, restricted to Train/Valid, plus the weights.

Everything about graphs, the official scaffold split, the ``lipo_v1`` FeatureSpec,
the Train-only normalisation and the collate layer is read-only reuse of the
frozen CSC v0 data layer (which itself reuses ``lcrc_readout_context_pilot_small_v1``,
``tracks/structural-query-learning`` and ``lcrc_cross_task_preparation_v1``).

This round adds only:

1. the official-row -> cache-row identity check between the two SMILES sources,
2. the per-molecule weight vector (Train structure only), aligned to cache rows,
3. a hard guard that refuses anything but ``train``/``valid`` and drops the Test
   arrays right after the inherited identity audit,
4. the shared batch-order plan, re-used verbatim from CSC v0.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import sb_common as C
import sb_scaffold as S

import cs_data as CD  # noqa: E402  (frozen CSC v0 data layer, read-only)


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


def _scrub_test(bundle: Any) -> dict[str, Any]:
    """Remove every Test-side array so no later code can touch it by accident."""
    removed: dict[str, Any] = {}
    for name in ("caches", "targets_raw", "y_std", "smiles", "n_atoms", "global_row_ids"):
        container = getattr(bundle, name, None)
        if isinstance(container, dict) and "test" in container:
            value = container.pop("test")
            if name == "caches" and isinstance(value, Mapping):
                removed[name] = {"n_molecules": int(value.get("n_molecules", -1))}
            elif name in ("targets_raw", "y_std", "smiles", "n_atoms", "global_row_ids"):
                removed[name] = {"n_rows": int(np.asarray(value).shape[0]) if name != "smiles" else len(value)}
            else:
                removed[name] = True
    for split in ("train", "valid"):
        assert split in bundle.caches, split
    return removed


def load_bundle() -> tuple[Any, dict[str, Any]]:
    """Frozen bundle with Train/Valid only + the inherited identity audit."""
    bundle = CD.load_bundle(C.DATASET, verify=True)
    audit = dict(bundle.extra["identity_audit"])
    removed = _scrub_test(bundle)
    # the identity audit is split-level; keep only the Train/Valid entries here
    audit["split_sizes"] = {k: v for k, v in audit["split_sizes"].items() if k in C.ALLOWED_SPLITS}
    audit["split_row_id_sha256"] = {k: v for k, v in audit["split_row_id_sha256"].items() if k in C.ALLOWED_SPLITS}
    audit["test_arrays_dropped_after_audit"] = removed
    audit["test_used_for_any_metric"] = False
    audit["test_batch_or_eval_calls"] = 0
    audit["note"] = (
        "the frozen cache builder (inherited, unchanged) reconstructs every split's graph tensors; "
        "this round runs no forward pass, no metric and no prediction on Test"
    )
    for split, size in (("train", C.N_TRAIN), ("valid", C.N_VALID)):
        if int(bundle.caches[split]["n_molecules"]) != size:
            raise RuntimeError(f"{split}: {bundle.caches[split]['n_molecules']} != {size}")
    return bundle, audit


def assert_split(split: str) -> str:
    if split not in C.ALLOWED_SPLITS:
        raise PermissionError(f"split {split!r} is not allowed in this round (Test is never read or scored)")
    return split


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def canonical_check(bundle: Any, official: Mapping[str, Any]) -> dict[str, Any]:
    """Cache SMILES and ``mapping/mol.csv.gz`` SMILES must denote the same molecules."""
    import sfl_common as SFL

    mismatched: list[int] = []
    for split in C.ALLOWED_SPLITS:
        cache_smiles = [str(x) for x in bundle.caches[split]["smiles"]]
        row_ids = np.asarray(bundle.global_row_ids[split], dtype=np.int64)
        if len(cache_smiles) != row_ids.size:
            raise RuntimeError(f"{split}: cache smiles {len(cache_smiles)} != row ids {row_ids.size}")
        for cache_row, row in enumerate(row_ids):
            a = SFL.canonical_smiles(cache_smiles[cache_row])
            b = SFL.canonical_smiles(str(official["smiles"][int(row)]))
            if a != b:
                mismatched.append(int(row))
    return {
        "rows_checked": int(C.N_TRAIN + C.N_VALID),
        "canonical_mismatches": mismatched[:20],
        "n_canonical_mismatches": int(len(mismatched)),
        "ok": bool(not mismatched),
    }


def train_weights(
    bundle: Any,
    official: Mapping[str, Any],
    scaffold_by_row: Mapping[int, str],
    *,
    k_train: int,
) -> dict[str, Any]:
    """Weight per cache-local Train row, aligned with the batch-order plan index space."""
    row_ids = np.asarray(bundle.global_row_ids[C.TRAIN_SPLIT], dtype=np.int64)
    w_global = S.balanced_weights(
        scaffold_by_row,
        np.asarray(official["train_rows"], dtype=np.int64),
        n_train=C.N_TRAIN,
        k_train=int(k_train),
    )
    w = np.asarray([float(w_global[int(r)]) for r in row_ids], dtype=np.float64)
    if w.size != C.N_TRAIN:
        raise RuntimeError(f"weight vector size {w.size} != {C.N_TRAIN}")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise RuntimeError("weights must be finite and strictly positive")
    counts = S.scaffold_counts(scaffold_by_row, official["train_rows"])
    counts_local = np.asarray([counts[scaffold_by_row[int(r)]] for r in row_ids], dtype=np.int64)
    return {
        "weights": w,
        "scaffold_ids": np.asarray([scaffold_by_row[int(r)] for r in row_ids], dtype=object),
        "scaffold_sizes": counts_local,
        "row_ids": row_ids,
        "weights_by_row_id": w_global,
        "sha256": C.sha256_array(w),
        "cache_local_index_note": "index i of the weight vector == index i of the batch-order plan (cache-local Train row)",
    }


# ---------------------------------------------------------------------------
# order plan / batches / evaluation
# ---------------------------------------------------------------------------


def batch_order_plan(epochs: int = C.EPOCHS) -> dict[str, Any]:
    plan = CD.batch_order_plan(C.DATASET, int(epochs))
    expected = C.HIST["train_plan_sha256"] if int(epochs) == C.EPOCHS else None
    plan["matches_historical_fixed1_plan"] = None if expected is None else bool(plan["plan_sha256"] == expected)
    return plan


def make_batch(bundle: Any, split: str, rows: np.ndarray, device=None) -> dict[str, Any]:
    assert_split(split)
    return CD.make_batch(bundle, split, np.asarray(rows, dtype=np.int64), device)


def split_order(bundle: Any, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert_split(split)
    return CD.split_order(bundle, split)


def evaluate(bundle: Any, model, split: str, device, *, batch_size: int | None = None) -> dict[str, Any]:
    """Full-split eval-mode metrics in raw label units; unweighted by construction."""
    assert_split(split)
    return CD.evaluate(bundle, model, split, device, batch_size=batch_size)


__all__ = [
    "CD",
    "assert_split",
    "batch_order_plan",
    "canonical_check",
    "evaluate",
    "load_bundle",
    "make_batch",
    "split_order",
    "train_weights",
]
