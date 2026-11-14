"""Data layer: frozen structure caches, nested subsets, shared norms, batches.

Everything here is deliberately a thin, *read-only* wrapper around the frozen
upstream modules (``lcpr_data`` / ``lcpr_common`` / ``lcrc_data``).  The only
new quantities are:

* the **nested supervision subsets** (label-blind prefixes of the official
  Train split, one per fit fraction) and their two coordinate systems;
* the **shared structure norm** (fitted once on all official Train structures)
  and the **shared target norm** (fitted once on the smallest fit subset);
* a typed label accessor that refuses to read Val/Test labels for fitting.

No cache is written into an upstream round: this track's ``cache/`` receives
independently named artefacts.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lb_common as C

import lcpr_data as PD
import lcpr_common as PC
import lcpr_train as PT


# ---------------------------------------------------------------------------
# dataset / spec / cache
# ---------------------------------------------------------------------------


def dataset_meta(dataset: str) -> dict[str, Any]:
    if dataset not in C.DATASETS:
        raise KeyError(dataset)
    return dict(C.DATASETS[dataset])


def frozen_spec():
    return PD.frozen_spec()


def load_source(dataset: str, labels: str = "train_valid") -> dict[str, Any]:
    return PD.load_source(dataset, labels=labels)


def load_cache(dataset: str, split: str, rows: Sequence[int] | None = None) -> dict[str, Any]:
    return PD.load_cache(dataset, split, rows)


def targets(dataset: str, *, include_test: bool) -> dict[str, Any]:
    return PD.targets(dataset, include_test=include_test)


# ---------------------------------------------------------------------------
# nested subsets
# ---------------------------------------------------------------------------


def split_rows(dataset: str) -> dict[str, list[int]]:
    src = load_source(dataset, labels="none")
    return {s: [int(x) for x in src["rows"][s]] for s in C.SPLITS}


def nested_subset_positions(dataset: str, frac: str) -> dict[str, Any]:
    """Return the frozen nested Train positions for ``frac`` in both coordinate systems."""
    rows = split_rows(dataset)["train"]
    positions = C.nested_subset(dataset, frac, rows)
    fit_rows = [rows[p] for p in positions]
    return {
        "dataset": dataset,
        "fit_fraction": frac,
        "n": len(positions),
        "positions": positions,                       # cache-local positions in official Train order
        "global_rows": fit_rows,                      # published global row ids
        "positions_sha256": C.row_sha256(positions),
        "global_rows_sha256": C.row_sha256(fit_rows),
        "key_definition": "ascending sha256('lcrc-bottleneck-v1|<dataset>|<global_row_id>'), tie by row id",
    }


def check_nested_consistency(dataset: str, fracs: Sequence[str]) -> dict[str, Any]:
    """A larger fit subset must be a strict superset of a smaller one."""
    sets: dict[str, set[int]] = {}
    for f in fracs:
        sets[f] = set(nested_subset_positions(dataset, f)["positions"])
    ordered = sorted(fracs, key=lambda f: C.FIT_N[dataset][f])
    out: dict[str, Any] = {"ordered_by_n": ordered, "nested": True, "pairs": {}}
    for a, b in zip(ordered, ordered[1:]):
        ok = sets[a] < sets[b]
        out["pairs"][f"{a}->{b}"] = bool(ok)
        out["nested"] = out["nested"] and ok
    out["sizes"] = {f: len(sets[f]) for f in ordered}
    return out


# ---------------------------------------------------------------------------
# shared norms
# ---------------------------------------------------------------------------


def structure_norm_path(dataset: str) -> Path:
    return C.CACHE / f"struct_norm_{dataset}.npz"


def structure_norm(dataset: str) -> dict[str, np.ndarray]:
    """Label-free structure norm fitted once on *all* official Train structures.

    Stored under a round-local name (never overwrites upstream ``train_norm_*``).
    """
    path = structure_norm_path(dataset)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    train = load_cache(dataset, "train")
    spec = frozen_spec()
    norm = PC.DD.PREP_DATA.fit_norm(train, spec, source=f"{dataset}_all_official_train_structures")
    if norm.get("valid_or_test_used") is not False:
        raise ValueError("structure norm must be Train-only")
    C.CACHE.mkdir(parents=True, exist_ok=True)
    keys = ("global_mean", "global_scale", "scalar_mean", "scalar_scale")
    np.savez(path, **{k: np.asarray(norm[k], dtype=np.float32) for k in keys})
    return {k: np.asarray(norm[k], dtype=np.float32) for k in keys}


def target_norm_path(dataset: str) -> Path:
    return C.CACHE / f"target_norm_{dataset}.npz"


def shared_target_norm(dataset: str) -> dict[str, Any]:
    """Target norm fitted once on the smallest fit subset's labels.

    Regression: Lipo-25 (840 rows).  Classification (BACE): raw 0/1, no norm.
    """
    meta = dataset_meta(dataset)
    if meta["task"] == "binary":
        return {"kind": "binary", "loss": "BCEWithLogits", "pos_weight": 1.0, "target_norm": None}
    path = target_norm_path(dataset)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {"kind": "regression", "target_norm": {
            "mean": float(z["mean"]), "scale": float(z["scale"]), "n": int(z["n"]),
            "source": str(z["source"]),
            "formula": "z = (y-mean)/scale; y_raw = mean + scale*z",
        }}
    frac = C.TARGET_NORM_SOURCE[dataset]
    tgt = targets(dataset, include_test=False)
    rows = split_rows(dataset)["train"]
    positions = nested_subset_positions(dataset, frac)["positions"]
    y = np.asarray(tgt["y"]["train"], np.float64).reshape(-1)[np.asarray(positions, np.int64)]
    mean, scale = float(np.mean(y)), float(np.std(y, ddof=0))
    if not (np.isfinite(mean) and np.isfinite(scale) and scale > 0):
        raise ValueError(f"{dataset}: degenerate shared target norm (scale={scale})")
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=np.float64(mean), scale=np.float64(scale), n=np.int64(y.size),
             source=f"smallest_fit_subset:{frac}:{y.size}")
    return {"kind": "regression", "target_norm": {
        "mean": mean, "scale": scale, "n": int(y.size),
        "source": f"smallest_fit_subset:{frac}:{y.size}",
        "formula": "z = (y-mean)/scale; y_raw = mean + scale*z",
    }}


def shared_norms_fingerprint(dataset: str) -> dict[str, Any]:
    sn = structure_norm(dataset)
    tn = shared_target_norm(dataset)
    return {
        "dataset": dataset,
        "structure_norm_fingerprint": C.sha256_text(C.stable_json(
            {k: np.asarray(v).tolist() for k, v in sn.items()})),
        "target_norm": tn["target_norm"],
        "structure_norm_source": "all_official_train_structures",
        "target_norm_source": tn["target_norm"]["source"] if tn["target_norm"] else "raw_0_1",
    }


# ---------------------------------------------------------------------------
# label-safe access
# ---------------------------------------------------------------------------


def fit_labels(dataset: str, positions: Sequence[int], tgt: Mapping[str, Any]) -> np.ndarray:
    """Labels of the *current* fit positions only (never from excluded Train / Val / Test)."""
    y = np.asarray(tgt["y"]["train"], np.float64).reshape(-1)
    return y[np.asarray(list(positions), np.int64)]


# ---------------------------------------------------------------------------
# batching (extra atom columns for K / chem8)
# ---------------------------------------------------------------------------


def chem8_atoms(dataset: str, split: str, cache: Mapping[str, Any]) -> np.ndarray:
    return PD.chem8_atoms(dataset, split, cache)


def concat_chem8(cache: Mapping[str, Any], extra: np.ndarray) -> np.ndarray:
    return PT._concat_chem8(cache, extra)


def collate(cache: Mapping[str, Any], rows: Sequence[int], *, path_count: np.ndarray | None = None):
    return PD.collate(cache, rows, path_count=path_count)


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return PD.to_device(batch, device)


# ---------------------------------------------------------------------------
# minimal batch plan long enough for any (model, N, phase) combination
# ---------------------------------------------------------------------------


def permutations_for(dataset: str, max_updates: int, batch_size: int) -> np.ndarray:
    """Frozen shuffle permutations covering ``max_updates`` batches.

    Rows are indices into the *official Train* split (position space).  A running
    batch cursor walks the concatenated permutation table; the small-N path may
    therefore need many epochs, which is expected and recorded.
    """
    total_batches = int(math.ceil(max_updates))
    n_train = C.official_train_n(dataset)
    per_epoch = math.ceil(n_train / batch_size)
    n_epochs_needed = math.ceil(total_batches / per_epoch)
    perms = PD.load_perms(dataset, max(n_epochs_needed, 1))
    if perms.shape[0] < n_epochs_needed:
        perms = PD.build_perms(dataset, n_epochs_needed)
    return perms[:n_epochs_needed]


__all__ = [n for n in dir() if not n.startswith("_")]
