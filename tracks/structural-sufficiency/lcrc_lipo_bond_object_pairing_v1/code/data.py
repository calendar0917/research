"""Structural data access, Fit batching and label-blind prediction.

The caches are label-free.  ``collate`` returns no target field.  The three
index spaces documented in the historical screen are preserved verbatim:

* ``global_row_id``  original dataset row id (provenance / label key only);
* ``fit_cache_pos``  position in the frozen Fit cache and Fit label vector;
* ``shuffle_pos``    position in the training subset returned by the shuffle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import common as C


def load_caches(spec=None, *, allow_rebuild: bool = True) -> dict[str, dict[str, Any]]:
    spec = spec or C.load_frozen_spec()
    out = {}
    for split in ("fit", "monitor"):
        try:
            out[split] = C.load_frozen_cache(split, spec)
        except FileNotFoundError:
            if not allow_rebuild:
                raise
            out[split] = C.load_frozen_cache(split, spec)
    return out


def load_target_norm(*, verify: bool = True) -> dict[str, Any]:
    path = C.HIST_ANALYSIS / "norm_target.json"
    if not path.exists():
        raise FileNotFoundError(C.track_rel(path))
    norm = C.read_json(path)
    if norm.get("source") != "fit_labels_only" or norm.get("monitor_or_reserved_used") is not False:
        raise PermissionError("target normalisation is not certified Fit-only")
    if int(norm.get("n", -1)) != C.SPLIT_SIZES["fit"]:
        raise PermissionError("target normalisation does not cover exactly Fit2352")
    if verify:
        import fit_labels

        fit = fit_labels.load_fit_labels(arm_guard=False)
        stats = fit_labels.fit_label_stats(fit["y"])
        for key in ("mean", "scale", "fit_label_vector_sha256"):
            if norm[key] != stats[key]:
                raise PermissionError(f"target normalisation {key} does not match the Fit label vector")
    return norm


# ---------------------------------------------------------------------------
# Batch order
# ---------------------------------------------------------------------------


def shuffle_batches(n: int, batch_size: int, generator: torch.Generator) -> list[list[int]]:
    perm = torch.randperm(int(n), generator=generator).tolist()
    return [perm[i : i + int(batch_size)] for i in range(0, len(perm), int(batch_size))]


def batch_order_hash(seed: int, n: int = C.SPLIT_SIZES["fit"], epochs: int = C.EPOCHS) -> str:
    generator = torch.Generator().manual_seed(int(seed) + C.TRAIN_SHUFFLE_OFFSET)
    h = hashlib.sha256()
    for _ in range(int(epochs)):
        for idx in shuffle_batches(n, C.BATCH_SIZE, generator):
            h.update(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class FitBatchPlan:
    shuffle_positions: tuple[int, ...]
    cache_positions: tuple[int, ...]
    global_row_ids: tuple[int, ...]
    n_subset: int


def fit_training_positions(n_subset: int) -> list[int]:
    n = int(n_subset)
    if not 1 <= n <= C.SPLIT_SIZES["fit"]:
        raise ValueError(f"n_subset must be in [1, {C.SPLIT_SIZES['fit']}], got {n_subset!r}")
    return list(range(n))


def plan_fit_batch(
    shuffle_positions: Sequence[int],
    *,
    train_positions: Sequence[int],
    cache: Mapping[str, Any],
    fit_global_rows: Sequence[int],
) -> FitBatchPlan:
    if isinstance(shuffle_positions, (str, bytes)) or torch.is_tensor(shuffle_positions):
        raise TypeError("shuffle_positions must be a sequence of integer positions")
    sp: list[int] = []
    for value in shuffle_positions:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"shuffle positions must be integers, got {type(value).__name__}")
        sp.append(int(value))
    tp = [int(p) for p in train_positions]
    n_molecules = int(cache["n_molecules"])
    if len(set(tp)) != len(tp) or any(p < 0 or p >= n_molecules for p in tp):
        raise ValueError("train_positions must be distinct cache positions in [0, n_molecules)")
    if any(i < 0 or i >= len(tp) for i in sp):
        raise IndexError(
            "shuffle position outside the training subset; global dataset row ids "
            "are not shuffle positions"
        )
    if len(set(sp)) != len(sp):
        raise ValueError("duplicate shuffle position in batch")
    cache_positions = tuple(tp[i] for i in sp)
    if len(fit_global_rows) != n_molecules:
        raise ValueError("fit_global_rows must have one global row id per cache position")
    global_row_ids = tuple(int(cache["molecule_ids"][p]) for p in cache_positions)
    expected = tuple(int(fit_global_rows[p]) for p in cache_positions)
    if global_row_ids != expected:
        raise ValueError("cache molecule order does not match the frozen Fit split")
    return FitBatchPlan(
        shuffle_positions=tuple(sp),
        cache_positions=cache_positions,
        global_row_ids=global_row_ids,
        n_subset=len(tp),
    )


def assemble_fit_batch(cache: Mapping[str, Any], y_fit: Sequence[float] | np.ndarray, plan: FitBatchPlan) -> dict[str, Any]:
    if not isinstance(plan, FitBatchPlan):
        raise TypeError("assemble_fit_batch requires a FitBatchPlan")
    n_molecules = int(cache["n_molecules"])
    if any(p < 0 or p >= n_molecules for p in plan.cache_positions):
        raise ValueError("cache position outside [0, n_molecules)")
    y = np.asarray(y_fit, dtype=np.float64)
    if y.shape != (n_molecules,):
        raise ValueError(f"y_fit must have one value per cache position, got {y.shape}")
    targets = y[list(plan.cache_positions)]
    batch = collate(cache, plan.cache_positions)
    if int(batch["n_graphs"]) != len(plan.cache_positions) or targets.shape[0] != len(plan.cache_positions):
        raise ValueError("batch/target length mismatch")
    return {
        "batch": batch,
        "y": targets,
        "cache_positions": plan.cache_positions,
        "global_row_ids": plan.global_row_ids,
    }


def fit_batch_fingerprints(plans: Sequence[FitBatchPlan], y_fit: Sequence[float] | np.ndarray) -> dict[str, Any]:
    positions = [p for plan in plans for p in plan.cache_positions]
    global_ids = [g for plan in plans for g in plan.global_row_ids]
    n_subset = int(plans[0].n_subset) if plans else 0
    targets = np.asarray(y_fit, dtype=np.float64)[positions] if positions else np.zeros(0, dtype=np.float64)
    return {
        "n_batches": len(plans),
        "n_molecules": len(positions),
        "n_subset": n_subset,
        "cache_positions_sha256": hashlib.sha256(np.ascontiguousarray(np.asarray(positions, dtype="<i8")).tobytes()).hexdigest(),
        "global_row_ids_sha256": hashlib.sha256(np.ascontiguousarray(np.asarray(global_ids, dtype="<i8")).tobytes()).hexdigest(),
        "targets_sha256": hashlib.sha256(np.ascontiguousarray(np.asarray(targets, dtype="<f8")).tobytes()).hexdigest(),
        "all_positions_in_range": bool(all(0 <= p < C.SPLIT_SIZES["fit"] for p in positions)),
        "all_positions_unique": bool(len(set(positions)) == len(positions)),
        "covers_all_subset_positions": bool(sorted(positions) == list(range(n_subset))),
    }


# ---------------------------------------------------------------------------
# targets / prediction
# ---------------------------------------------------------------------------


def denormalise(pred: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(pred, dtype=np.float64) * float(norm["scale"]) + float(norm["mean"])


def raw_contribution_terms(contrib: np.ndarray, rho: np.ndarray, norm: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    scale = float(norm["scale"])
    mean = float(norm["mean"])
    return (
        np.asarray(contrib, dtype=np.float64) * scale,
        np.asarray(rho, dtype=np.float64) * scale + mean,
    )


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return C.prep_data.collate(cache, rows)


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def monitor_rows(caches: Mapping[str, Mapping[str, Any]]) -> list[int]:
    return [int(r) for r in caches["monitor"]["rows"]]


def monitor_rows_sha256(caches: Mapping[str, Mapping[str, Any]]) -> str:
    rows = monitor_rows(caches)
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


@torch.no_grad()
def predict_cache(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    device,
    *,
    batch_size: int = 256,
    return_aux: bool = False,
) -> dict[str, Any]:
    """Label-blind prediction over a structural cache, in cache order."""
    was_training = model.training
    model.eval()
    n = int(cache["n_molecules"])
    collected: dict[str, list[np.ndarray]] = {"pred": [], "pred_base": [], "delta": []}
    for start in range(0, n, int(batch_size)):
        rows = list(range(start, min(start + int(batch_size), n)))
        batch = to_device(collate(cache, rows), device)
        if return_aux:
            y, aux = model(batch, return_aux=True)
            collected["pred_base"].append(aux["y_base"].detach().cpu().numpy().reshape(-1))
            collected["delta"].append(aux["delta_bond"].detach().cpu().numpy().reshape(-1))
        else:
            y = model(batch)
        collected["pred"].append(y.detach().cpu().numpy().reshape(-1))
    out = {"pred": np.concatenate(collected["pred"]).astype(np.float64), "rows": list(range(n))}
    if return_aux:
        for key in ("pred_base", "delta"):
            out[key] = np.concatenate(collected[key]).astype(np.float64)
    if was_training:
        model.train()
    return out


def atom_counts(cache: Mapping[str, Any], rows: Iterable[int] | None = None) -> np.ndarray:
    atom_ptr = np.asarray(cache["atom_ptr"])
    if rows is None:
        return np.diff(atom_ptr).astype(np.int64)
    rows = np.asarray(list(rows), dtype=np.int64)
    return (atom_ptr[rows + 1] - atom_ptr[rows]).astype(np.int64)


def bond_diagnostics(cache: Mapping[str, Any], rows: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Label-free per-molecule bond counts and per-type counts."""
    if rows is None:
        rows = np.arange(int(cache["n_molecules"]), dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    ptr = np.asarray(cache["bond_ptr"])
    bt = np.asarray(cache["bond_type"])
    n_types = int(C.load_frozen_spec().n_bond_types)
    counts = np.zeros((len(rows), n_types), dtype=np.int64)
    for i, r in enumerate(rows):
        chunk = bt[ptr[r] : ptr[r + 1]]
        for t in chunk:
            counts[i, int(t) - 1] += 1
    return {
        "n_bonds": counts.sum(axis=1),
        "n_bonds_by_type": counts,
    }


__all__ = [
    "load_caches",
    "load_target_norm",
    "shuffle_batches",
    "batch_order_hash",
    "FitBatchPlan",
    "fit_training_positions",
    "plan_fit_batch",
    "assemble_fit_batch",
    "fit_batch_fingerprints",
    "denormalise",
    "raw_contribution_terms",
    "collate",
    "to_device",
    "monitor_rows",
    "monitor_rows_sha256",
    "predict_cache",
    "atom_counts",
    "bond_diagnostics",
]
