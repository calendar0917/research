"""Fold-specific data views and normalisation for the v2 round.

Every supervised comparison uses only Fit2352.  For internal CV, the training
fold's label mean/std, global42 mean/std and atom-scalar mean/std are recomputed
from that fold's training molecules.  The development fold is used only for
evaluation; it never contributes to standardisation, checkpoint selection or
early stopping.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import v2_common as C
import v2_folds as F


# ---------------------------------------------------------------------------
# cache subsetting
# ---------------------------------------------------------------------------


def select_cache(cache: Mapping[str, Any], positions: Sequence[int], *, split_label: str = "fit_subset") -> dict[str, Any]:
    """Return a molecule-indexed copy of a frozen structural cache."""
    pos = [int(p) for p in positions]
    if len(pos) != len(set(pos)):
        raise ValueError("duplicate cache position")
    n = int(cache["n_molecules"])
    if any(p < 0 or p >= n for p in pos):
        raise IndexError("cache position outside the frozen cache")

    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)

    atom_ranges = [(int(atom_ptr[p]), int(atom_ptr[p + 1])) for p in pos]
    bond_ranges = [(int(bond_ptr[p]), int(bond_ptr[p + 1])) for p in pos]
    nbr_ranges = [(int(nbr_ptr[a0]), int(nbr_ptr[a1])) for a0, a1 in atom_ranges]
    path_ranges = [(int(path_ptr[a0]), int(path_ptr[a1])) for a0, a1 in atom_ranges]

    out: dict[str, Any] = {}
    out["atom_feat"] = np.concatenate([np.asarray(cache["atom_feat"])[a0:a1] for a0, a1 in atom_ranges], axis=0)
    out["atom_type"] = np.concatenate([np.asarray(cache["atom_type"])[a0:a1] for a0, a1 in atom_ranges], axis=0)
    out["atom_scalar"] = np.concatenate([np.asarray(cache["atom_scalar"])[a0:a1] for a0, a1 in atom_ranges], axis=0)
    out["bond_feat"] = np.concatenate([np.asarray(cache["bond_feat"])[b0:b1] for b0, b1 in bond_ranges], axis=0)
    out["bond_type"] = np.concatenate([np.asarray(cache["bond_type"])[b0:b1] for b0, b1 in bond_ranges], axis=0)
    out["bond_index"] = np.concatenate([np.asarray(cache["bond_index"])[b0:b1] for b0, b1 in bond_ranges], axis=0)
    out["nbr_code"] = np.concatenate([np.asarray(cache["nbr_code"])[n0:n1] for n0, n1 in nbr_ranges], axis=0)
    out["path_cat"] = np.concatenate([np.asarray(cache["path_cat"])[p0:p1] for p0, p1 in path_ranges], axis=0)
    out["path_end63"] = np.concatenate([np.asarray(cache["path_end63"])[p0:p1] for p0, p1 in path_ranges], axis=0)
    out["path_end"] = np.concatenate([np.asarray(cache["path_end"])[p0:p1] for p0, p1 in path_ranges], axis=0)
    out["path_closure"] = np.concatenate([np.asarray(cache["path_closure"])[p0:p1] for p0, p1 in path_ranges], axis=0)
    out["global"] = np.asarray(cache["global"])[np.asarray(pos, dtype=np.int64)].copy()
    out["n_components"] = np.asarray(cache["n_components"])[np.asarray(pos, dtype=np.int64)].copy()
    out["n_junctions"] = np.asarray(cache["n_junctions"])[np.asarray(pos, dtype=np.int64)].copy()
    out["n_ring_systems"] = np.asarray(cache["n_ring_systems"])[np.asarray(pos, dtype=np.int64)].copy()
    out["molecule_ids"] = [str(cache["molecule_ids"][p]) for p in pos]
    out["smiles"] = [str(cache["smiles"][p]) for p in pos]

    # Pointers.
    atom_counts = np.asarray([a1 - a0 for a0, a1 in atom_ranges], dtype=np.int64)
    bond_counts = np.asarray([b1 - b0 for b0, b1 in bond_ranges], dtype=np.int64)
    out["atom_ptr"] = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int64)
    out["bond_ptr"] = np.concatenate([[0], np.cumsum(bond_counts)]).astype(np.int64)
    nbr_counts = np.concatenate(
        [np.asarray(cache["nbr_ptr"])[a0 + 1 : a1 + 1] - np.asarray(cache["nbr_ptr"])[a0:a1] for a0, a1 in atom_ranges]
    ) if atom_ranges else np.zeros(0, dtype=np.int64)
    path_counts = np.concatenate(
        [np.asarray(cache["path_ptr"])[a0 + 1 : a1 + 1] - np.asarray(cache["path_ptr"])[a0:a1] for a0, a1 in atom_ranges]
    ) if atom_ranges else np.zeros(0, dtype=np.int64)
    out["nbr_ptr"] = np.concatenate([[0], np.cumsum(nbr_counts)]).astype(np.int64)
    out["path_ptr"] = np.concatenate([[0], np.cumsum(path_counts)]).astype(np.int64)

    out["n_molecules"] = len(pos)
    meta = dict(cache.get("meta", {}))
    meta["n_molecules"] = len(pos)
    meta["subset_of"] = str(cache.get("split", ""))
    meta["subset_positions_sha256"] = hashlib.sha256(
        np.ascontiguousarray(np.asarray(pos, dtype=np.int64)).tobytes()
    ).hexdigest()
    out["meta"] = meta
    out["split"] = split_label
    return out


def cache_row_ids(cache: Mapping[str, Any]) -> list[str]:
    return [str(x) for x in cache["molecule_ids"]]


# ---------------------------------------------------------------------------
# fold views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataView:
    name: str
    train_cache: Mapping[str, Any]
    dev_cache: Mapping[str, Any]
    y_train: np.ndarray
    y_dev: np.ndarray
    train_norm: Mapping[str, Any]
    target_norm: Mapping[str, Any]
    train_positions: tuple[int, ...]
    dev_positions: tuple[int, ...]
    fold: int | None = None

    def audit(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fold": self.fold,
            "n_train": int(len(self.y_train)),
            "n_dev": int(len(self.y_dev)),
            "train_positions_sha256": hashlib.sha256(
                np.ascontiguousarray(np.asarray(self.train_positions, dtype=np.int64)).tobytes()
            ).hexdigest(),
            "dev_positions_sha256": hashlib.sha256(
                np.ascontiguousarray(np.asarray(self.dev_positions, dtype=np.int64)).tobytes()
            ).hexdigest(),
            "target_norm_mean": float(self.target_norm["mean"]),
            "target_norm_scale": float(self.target_norm["scale"]),
            "target_norm_n": int(self.target_norm["n"]),
            "train_norm_fingerprint": C.prep_data.norm_fingerprint(self.train_norm),
            "train_norm_source": self.train_norm.get("source"),
        }


def _fold_target_norm(y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError("invalid fold training labels")
    return {
        "source": "fold_train_labels_only",
        "n": int(y.shape[0]),
        "mean": float(y.mean()),
        "scale": float(y.std()),
        "scale_definition": "population_std_float64",
        "monitor_or_reserved_used": False,
        "label_vector_sha256": hashlib.sha256(np.ascontiguousarray(y, dtype="<f8").tobytes()).hexdigest(),
    }


def build_fold_view(
    fold: int,
    *,
    cache: Mapping[str, Any],
    y_fit: np.ndarray,
    folds: Mapping[str, Any],
    spec=None,
) -> DataView:
    spec = spec or C.load_frozen_spec()
    dev_pos = F.fold_positions(folds, int(fold))
    train_pos = F.train_positions(folds, int(fold))
    if set(dev_pos) & set(train_pos):
        raise ValueError("fold train/dev overlap")
    train_cache = select_cache(cache, train_pos, split_label=f"fold{fold}_train")
    dev_cache = select_cache(cache, dev_pos, split_label=f"fold{fold}_dev")
    y = np.asarray(y_fit, dtype=np.float64)
    y_train = y[np.asarray(train_pos, dtype=np.int64)]
    y_dev = y[np.asarray(dev_pos, dtype=np.int64)]
    train_norm = C.prep_data.fit_norm(train_cache, spec, source=f"fold{fold}_train_structures")
    target_norm = _fold_target_norm(y_train)
    return DataView(
        name=f"fold{fold}",
        train_cache=train_cache,
        dev_cache=dev_cache,
        y_train=y_train,
        y_dev=y_dev,
        train_norm=train_norm,
        target_norm=target_norm,
        train_positions=tuple(train_pos),
        dev_positions=tuple(dev_pos),
        fold=int(fold),
    )


def build_full_view(*, cache: Mapping[str, Any], y_fit: np.ndarray, spec=None) -> DataView:
    spec = spec or C.load_frozen_spec()
    frozen_norm = C.load_frozen_norm()
    # Recompute from Fit2352 structures and certify that it equals the frozen
    # v1 Fit-only norm (same data, same policy).
    recomputed = C.prep_data.fit_norm(cache, spec, source="fit2352_structures")
    # ``norm_fingerprint`` includes the provenance string, so compare the
    # actual normalisation arrays and constant-index sets.  The frozen v1 norm
    # was fitted on the same Fit2352 structures; only its ``source`` label
    # differs from this recomputation.
    for key in ("global_mean", "global_scale", "scalar_mean", "scalar_scale"):
        if not np.array_equal(np.asarray(recomputed[key], dtype=np.float32), np.asarray(frozen_norm[key], dtype=np.float32)):
            raise PermissionError(f"recomputed Fit2352 {key} differs from the frozen Fit-only norm")
    for key in ("global_constant_indices", "scalar_constant_indices"):
        if list(recomputed[key]) != list(frozen_norm[key]):
            raise PermissionError(f"recomputed Fit2352 {key} differs from the frozen Fit-only norm")
    if frozen_norm.get("fit_split") != "fit" or frozen_norm.get("valid_or_test_used") is not False:
        raise PermissionError("frozen norm is not certified Fit-only")
    target_norm = _fold_target_norm(np.asarray(y_fit, dtype=np.float64))
    frozen_target = C.v1d.load_target_norm(verify=True)
    for key in ("mean", "scale"):
        if abs(float(target_norm[key]) - float(frozen_target[key])) > 1e-12:
            raise PermissionError(f"recomputed Fit2352 target {key} differs from the frozen Fit-only value")
    return DataView(
        name="full_fit2352",
        train_cache=cache,
        dev_cache=cache,
        y_train=np.asarray(y_fit, dtype=np.float64),
        y_dev=np.asarray(y_fit, dtype=np.float64),
        train_norm=frozen_norm,
        target_norm=target_norm,
        train_positions=tuple(range(int(cache["n_molecules"]))),
        dev_positions=tuple(range(int(cache["n_molecules"]))),
        fold=None,
    )


def load_all_fit_data() -> dict[str, Any]:
    spec = C.load_frozen_spec()
    cache_info = C.load_fit_cache()
    cache = cache_info["cache"]
    fit_labels = C.load_fit_labels()
    y_fit = np.asarray(fit_labels["y"], dtype=np.float64)
    if y_fit.shape != (C.SPLIT_SIZES["fit"],):
        raise ValueError("Fit label vector has the wrong length")
    folds = F.load_folds()
    return {
        "spec": spec,
        "cache_info": cache_info,
        "cache": cache,
        "y_fit": y_fit,
        "folds": folds,
        "labels_audit": fit_labels["audit"],
    }


def build_all_views() -> dict[str, Any]:
    data = load_all_fit_data()
    views = {
        f"fold{f}": build_fold_view(
            f, cache=data["cache"], y_fit=data["y_fit"], folds=data["folds"], spec=data["spec"]
        )
        for f in C.FOLDS
    }
    views["full"] = build_full_view(cache=data["cache"], y_fit=data["y_fit"], spec=data["spec"])
    return {**data, "views": views}


# ---------------------------------------------------------------------------
# batching / prediction
# ---------------------------------------------------------------------------


def shuffle_batches(n: int, batch_size: int, seed: int) -> list[list[int]]:
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(int(n), generator=generator).tolist()
    return [perm[i : i + int(batch_size)] for i in range(0, len(perm), int(batch_size))]


def iter_epoch_rows(n: int, batch_size: int, seed: int, shuffle_offset: int, epochs: int):
    """Yield (epoch, batch_positions) for the fixed shuffle stream."""
    generator = torch.Generator().manual_seed(int(seed) + int(shuffle_offset))
    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(int(n), generator=generator).tolist()
        for start in range(0, len(perm), int(batch_size)):
            yield epoch, perm[start : start + int(batch_size)]


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return C.prep_data.collate(cache, rows)


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict_view(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    *,
    device,
    batch_size: int = 256,
    global_extra: np.ndarray | None = None,
) -> np.ndarray:
    """Label-blind prediction over a cache, in cache order."""
    was_training = model.training
    model.eval()
    n = int(cache["n_molecules"])
    preds: list[np.ndarray] = []
    for start in range(0, n, int(batch_size)):
        rows = list(range(start, min(start + int(batch_size), n)))
        batch = to_device(collate(cache, rows), device)
        if global_extra is not None:
            extra = torch.as_tensor(global_extra[rows], dtype=torch.float32, device=device)
            batch["global"] = torch.cat([batch["global"], extra], dim=1)
        preds.append(model(batch).detach().cpu().numpy().reshape(-1))
    if was_training:
        model.train()
    return np.concatenate(preds).astype(np.float64)


def denormalise(pred: np.ndarray, target_norm: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(pred, dtype=np.float64) * float(target_norm["scale"]) + float(target_norm["mean"])


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2)))


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))))


__all__ = [
    "select_cache",
    "cache_row_ids",
    "DataView",
    "build_fold_view",
    "build_full_view",
    "load_all_fit_data",
    "build_all_views",
    "shuffle_batches",
    "iter_epoch_rows",
    "collate",
    "to_device",
    "predict_view",
    "denormalise",
    "rmse",
    "mae",
]
