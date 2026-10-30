"""Frozen data access, batching and label-blind prediction.

Everything here is structural: the caches are label-free, the collated batch
has no label/target field, and prediction never receives a target.  Target
standardisation is Fit-only and is written once by the protocol freeze step.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import screen_common as C
from screen_common import prep_data


# ---------------------------------------------------------------------------
# frozen caches
# ---------------------------------------------------------------------------


def load_caches(spec=None) -> dict[str, dict[str, Any]]:
    """Load the Fit and Monitor structural caches (never Reserved)."""
    spec = spec or C.load_frozen_spec()
    out = {}
    for split in ("fit", "monitor"):
        out[split] = C.load_frozen_cache(split, spec)
    return out


def load_target_norm(*, verify: bool = True) -> dict[str, Any]:
    if not C.NORM_TARGET.exists():
        raise FileNotFoundError(
            f"missing {C.track_rel(C.NORM_TARGET)}; run `run_screen.py freeze-protocol` first"
        )
    norm = C.read_json(C.NORM_TARGET)
    if norm.get("source") != "fit_labels_only" or norm.get("monitor_or_reserved_used") is not False:
        raise PermissionError("target normalisation is not certified Fit-only")
    if int(norm.get("n", -1)) != C.SPLIT_SIZES["fit"]:
        raise PermissionError("target normalisation does not cover exactly Fit2352")
    if verify:
        import fit_labels

        fit = fit_labels.load_fit_labels()
        stats = fit_labels.fit_label_stats(fit["y"])
        for key in ("mean", "scale", "fit_label_vector_sha256"):
            if norm[key] != stats[key]:
                raise PermissionError(f"target normalisation {key} does not match the Fit label vector")
    return norm


def save_target_norm() -> dict[str, Any]:
    import fit_labels

    fit = fit_labels.load_fit_labels()
    stats = fit_labels.fit_label_stats(fit["y"])
    stats["label_artifact_sha256"] = fit["labels_sha256"]
    stats["label_audit"] = fit["audit"]
    C.write_json(C.NORM_TARGET, stats)
    return stats


# ---------------------------------------------------------------------------
# batch order
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


def epoch_batches(seed: int, epoch: int) -> list[list[int]]:
    """Deterministic batch rows for one epoch without replaying earlier epochs."""
    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    generator = torch.Generator().manual_seed(int(seed) + C.TRAIN_SHUFFLE_OFFSET)
    batches: list[list[int]] = []
    for _ in range(int(epoch)):
        batches = shuffle_batches(C.SPLIT_SIZES["fit"], C.BATCH_SIZE, generator)
    return batches


# ---------------------------------------------------------------------------
# Fit batch assembly: three distinct index spaces
# ---------------------------------------------------------------------------
#
# ``global_row_id``   original Lipophilicity dataset row id (0..4199).  Used only
#                     for identity checks, provenance and the label artifact key.
# ``fit_cache_pos``   position inside the frozen Fit cache (0..2351).  The cache
#                     stores molecules in frozen split order, so this is also the
#                     row of the frozen Fit label vector ``y_fit``.
# ``shuffle_pos``     position inside the current training subset (0..n_subset-1),
#                     i.e. what ``shuffle_batches`` returns.  The formal run uses
#                     the whole Fit set (n_subset == 2352); the smoke subset is a
#                     prefix of the cache positions.
#
# ``collate`` accepts **cache positions only**.  The helpers below are the only
# supported way to build a Fit batch, so a global row id cannot silently reach the
# collator and cannot be paired with a position-derived label.


@dataclass(frozen=True)
class FitBatchPlan:
    """Validated mapping from shuffle positions to Fit cache positions."""

    shuffle_positions: tuple[int, ...]
    cache_positions: tuple[int, ...]
    global_row_ids: tuple[int, ...]
    n_subset: int


def fit_training_positions(n_subset: int) -> list[int]:
    """Cache positions of the training subset (formal: 0..2351)."""
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
    """Translate shuffle positions into Fit cache positions and global ids.

    ``shuffle_positions`` are indices into ``train_positions``, never dataset row
    ids.  Out-of-subset values (the typical global-row-id failure mode: row ids
    reach 4199 while the Fit subset has 2352 slots) raise ``IndexError``.
    """
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


def assemble_fit_batch(
    cache: Mapping[str, Any], y_fit: Sequence[float] | np.ndarray, plan: FitBatchPlan
) -> dict[str, Any]:
    """Build one Fit batch (inputs in cache order, targets in the same order)."""
    if not isinstance(plan, FitBatchPlan):
        raise TypeError(
            "assemble_fit_batch requires a FitBatchPlan built by plan_fit_batch(); "
            "raw index lists (e.g. global row ids) are rejected"
        )
    n_molecules = int(cache["n_molecules"])
    if any(p < 0 or p >= n_molecules for p in plan.cache_positions):
        raise ValueError("cache position outside [0, n_molecules); global row ids must not be used here")
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


def fit_epoch_plans(
    seed: int,
    *,
    epoch: int = 1,
    n_subset: int = C.SPLIT_SIZES["fit"],
    cache: Mapping[str, Any],
    fit_global_rows: Sequence[int],
) -> list[FitBatchPlan]:
    """Plans for one epoch of the *same* shuffle stream the trainer consumes."""
    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    train_positions = fit_training_positions(n_subset)
    generator = torch.Generator().manual_seed(int(seed) + C.TRAIN_SHUFFLE_OFFSET)
    for _ in range(int(epoch) - 1):
        shuffle_batches(n_subset, C.BATCH_SIZE, generator)
    return [
        plan_fit_batch(
            batch,
            train_positions=train_positions,
            cache=cache,
            fit_global_rows=fit_global_rows,
        )
        for batch in shuffle_batches(n_subset, C.BATCH_SIZE, generator)
    ]


def fit_batch_fingerprints(plans: Sequence[FitBatchPlan], y_fit: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Order-preserving fingerprints of one epoch's (positions, ids, targets)."""
    positions = [p for plan in plans for p in plan.cache_positions]
    global_ids = [g for plan in plans for g in plan.global_row_ids]
    n_subset = int(plans[0].n_subset) if plans else 0
    targets = np.asarray(y_fit, dtype=np.float64)[positions] if positions else np.zeros(0, dtype=np.float64)
    return {
        "n_batches": len(plans),
        "n_molecules": len(positions),
        "n_subset": n_subset,
        "cache_positions_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(positions, dtype="<i8")).tobytes()
        ).hexdigest(),
        "global_row_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(global_ids, dtype="<i8")).tobytes()
        ).hexdigest(),
        "targets_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(targets, dtype="<f8")).tobytes()
        ).hexdigest(),
        "all_positions_in_range": bool(all(0 <= p < C.SPLIT_SIZES["fit"] for p in positions)),
        "all_positions_unique": bool(len(set(positions)) == len(positions)),
        "covers_all_subset_positions": bool(sorted(positions) == list(range(n_subset))),
    }


# ---------------------------------------------------------------------------
# target assembly
# ---------------------------------------------------------------------------


def normalised_target(y_fit: np.ndarray, rows: Sequence[int], norm: Mapping[str, Any]) -> torch.Tensor:
    y = np.asarray(y_fit, dtype=np.float64)[np.asarray(rows, dtype=np.int64)]
    z = (y - float(norm["mean"])) / float(norm["scale"])
    return torch.as_tensor(z, dtype=torch.float32)


def denormalise(pred: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(pred, dtype=np.float64) * float(norm["scale"]) + float(norm["mean"])


def raw_contribution_terms(
    contrib: np.ndarray, rho: np.ndarray, norm: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Map the standardised-space A decomposition to raw units.

    The constant Fit mean is absorbed into ``rho`` so that
    ``contrib_raw + rho_raw == pred_raw`` exactly (up to float rounding).
    """
    scale = float(norm["scale"])
    mean = float(norm["mean"])
    return (
        np.asarray(contrib, dtype=np.float64) * scale,
        np.asarray(rho, dtype=np.float64) * scale + mean,
    )


# ---------------------------------------------------------------------------
# collation and prediction
# ---------------------------------------------------------------------------


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return prep_data.collate(cache, rows)


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
    preds: list[np.ndarray] = []
    contribs: list[np.ndarray] = []
    rhos: list[np.ndarray] = []
    for start in range(0, n, int(batch_size)):
        rows = list(range(start, min(start + int(batch_size), n)))
        batch = to_device(collate(cache, rows), device)
        if return_aux:
            y, aux = model(batch, return_aux=True)
            contribs.append(aux["contrib"].detach().cpu().numpy())
            rhos.append(aux["rho"].detach().cpu().numpy())
        else:
            y = model(batch)
        preds.append(y.detach().cpu().numpy().reshape(-1))
    out = {"pred": np.concatenate(preds).astype(np.float64), "rows": list(range(n))}
    if return_aux:
        out["contrib"] = np.concatenate(contribs).astype(np.float64)
        out["rho"] = np.concatenate(rhos).astype(np.float64)
    if was_training:
        model.train()
    return out


def atom_counts(cache: Mapping[str, Any], rows: Iterable[int] | None = None) -> np.ndarray:
    atom_ptr = np.asarray(cache["atom_ptr"])
    if rows is None:
        return np.diff(atom_ptr).astype(np.int64)
    rows = np.asarray(list(rows), dtype=np.int64)
    return (atom_ptr[rows + 1] - atom_ptr[rows]).astype(np.int64)


__all__ = [
    "load_caches",
    "load_target_norm",
    "save_target_norm",
    "shuffle_batches",
    "batch_order_hash",
    "epoch_batches",
    "FitBatchPlan",
    "fit_training_positions",
    "plan_fit_batch",
    "assemble_fit_batch",
    "fit_epoch_plans",
    "fit_batch_fingerprints",
    "normalised_target",
    "denormalise",
    "raw_contribution_terms",
    "collate",
    "to_device",
    "monitor_rows",
    "monitor_rows_sha256",
    "predict_cache",
    "atom_counts",
]
