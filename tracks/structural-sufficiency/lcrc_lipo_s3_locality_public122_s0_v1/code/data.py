"""Structure caches, Train-only normalisation and batched prediction helpers.

The computational graph is the frozen ``lipo_v1`` / LCRC-C1 one.  The structure
pipeline is called through the dissection round's own builder; only the cache
destination is new.  No target byte is stored in a structure cache.

Every cache is content-fingerprinted with the frozen upstream routine
(``lcrc_common.content_fingerprint``).  When a cache from the baseline round is
present the fingerprint must match **exactly**; a mismatch raises instead of
silently mixing two structure pipelines.

Target normalisation (``mean`` / ``std(ddof=0)``) and structure normalisation
are fitted on the published Train3360 only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import common as C

CACHE_VERSION = "lcrc-lipo-s3-locality-v1"


# ---------------------------------------------------------------------------
# caches
# ---------------------------------------------------------------------------


def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def frozen_spec():
    spec = C.DD.frozen_spec()
    if spec.hash() != C.FEATURE_SPEC_HASH:
        raise ValueError(f"lipo_v1 spec hash {spec.hash()} != frozen {C.FEATURE_SPEC_HASH}")
    if (spec.atom_feat_dim, spec.bond_feat_dim, spec.n_scalars, spec.global_dim) != (
        C.ATOM_FEAT_DIM,
        C.BOND_FEAT_DIM,
        C.ATOM_SCALAR_DIM,
        C.GLOBAL_DIM,
    ):
        raise ValueError("frozen lipo_v1 input widths changed")
    return spec


def build_split_cache(split: str, rows: Sequence[int], smiles: Sequence[str], spec) -> dict[str, Any]:
    cache = C.DD.build_split_cache(C.DATA_REPEAT, split, list(rows), list(smiles), spec, save=False)
    if cache["meta"].get("labels_present"):
        raise ValueError("structure cache must be label-free")
    upstream_fp = cache["meta"]["content_fingerprint"]
    cache["round_id"] = C.ROUND_ID
    cache["upstream_cache_version"] = cache.get("cache_version")
    cache["cache_version"] = CACHE_VERSION
    cache["meta"]["round_id"] = C.ROUND_ID
    cache["meta"]["upstream_content_fingerprint"] = upstream_fp
    _atomic_save(cache, C.cache_path(split))
    return cache


def load_cache(split: str, *, require: bool = True, rebuild: bool = False) -> dict[str, Any]:
    spec = frozen_spec()
    src = C.source_with_labels("none")
    rows = [int(r) for r in src["rows"][split]]
    path = C.cache_path(split)
    if path.exists() and not rebuild:
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"{path}: unexpected cache version {cache.get('cache_version')!r}")
        if [int(x) for x in np.asarray(cache["rows"]).tolist()] != rows:
            raise ValueError(f"{path}: cached row ids differ from the published {split} split")
        if cache["meta"].get("feature_spec_hash") != spec.hash():
            raise ValueError(f"{path}: cached feature spec hash differs")
        if cache["meta"].get("labels_present"):
            raise ValueError(f"{path}: structure cache must be label-free")
        if int(cache["n_molecules"]) != len(rows):
            raise ValueError(f"{path}: cache size differs")
        return cache
    if require and not rebuild:
        raise FileNotFoundError(f"{path} is missing; run code/prepare.py --cache-only on this host first")
    return build_split_cache(split, rows, src["smiles"], spec)


def build_all_caches(*, rebuild: bool = False) -> dict[str, Any]:
    spec = frozen_spec()
    src = C.source_with_labels("none")
    entries: dict[str, Any] = {}
    for split in C.SPLITS:
        rows = src["rows"][split]
        cache = build_split_cache(split, rows, src["smiles"], spec) if rebuild else load_cache(
            split, require=False
        )
        fp = cache["meta"]["upstream_content_fingerprint"]
        old = C.BASELINE_ROUND / "cache" / f"Lipophilicity_r{C.DATA_REPEAT}_{split}.pt"
        same_as_baseline = None
        if old.exists():
            old_cache = torch.load(old, map_location="cpu", weights_only=False)
            same_as_baseline = bool(old_cache["meta"]["content_fingerprint"] == fp)
            if not same_as_baseline:
                raise C.DataMismatch(
                    f"{split}: content fingerprint {fp} != baseline {old_cache['meta']['content_fingerprint']}"
                )
        entries[split] = {
            "n_molecules": int(cache["n_molecules"]),
            "rows_sha256": C.row_sha256(np.asarray(cache["rows"]).tolist()),
            "content_fingerprint": fp,
            "cache_path": C.track_rel(C.cache_path(split)),
            "cache_sha256": C.sha256_file(C.cache_path(split)),
            "labels_present": bool(cache["meta"]["labels_present"]),
            "same_as_baseline_reuse_checked": same_as_baseline,
        }
        C.log(f"[cache] {split}: n={cache['n_molecules']} fp={fp[:16]} baseline_match={same_as_baseline}")
    payload = {
        "schema": "lcrc-lipo-s3-locality-cache-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "feature_spec_hash": spec.hash(),
        "label_free": True,
        "splits": entries,
        "train_norm": train_norm(),
        "target_norm": target_norm(),
    }
    C.write_json(C.CACHE / "manifest.json", payload)
    return payload


def train_norm() -> dict[str, Any]:
    cache = load_cache("train")
    spec = frozen_spec()
    norm = C.PREP_DATA.fit_norm(cache, spec, source=f"Lipophilicity_r{C.DATA_REPEAT}_train_structures_only")
    if norm.get("valid_or_test_used") is not False:
        raise ValueError("structure normalisation must be Train-only")
    return norm


def target_norm() -> dict[str, Any]:
    src = C.source_with_labels("train_valid")
    return C.DD.target_normaliser(np.asarray(src["y"]["train"], dtype=np.float64))


def load_dataset(*, include_test: bool = False) -> dict[str, Any]:
    spec = frozen_spec()
    src = C.source_with_labels("all" if include_test else "train_valid")
    train_cache = load_cache("train")
    valid_cache = load_cache("valid")
    y_train = np.asarray(src["y"]["train"], dtype=np.float64)
    y_valid = np.asarray(src["y"]["valid"], dtype=np.float64)
    payload: dict[str, Any] = {
        "repeat": C.DATA_REPEAT,
        "spec": spec,
        "rows": {s: [int(x) for x in src["rows"][s]] for s in C.SPLITS},
        "smiles": [str(x) for x in src["smiles"]],
        "train_cache": train_cache,
        "valid_cache": valid_cache,
        "y_train": y_train,
        "y_valid": y_valid,
        "train_norm": train_norm(),
        "target_norm": C.DD.target_normaliser(y_train),
        "source": {
            "source_commit": src["source_commit"],
            "csv_sha256": src["csv_sha256"],
            "split_file": src["split_file"],
            "split_file_sha256": src["split_file_sha256"],
            "split_rows_sha256": src["split_rows_sha256"],
            "split_sizes": src["sizes"],
            "label_column": src["label_column"],
        },
    }
    if include_test:
        payload["test_cache"] = load_cache("test")
        payload["y_test"] = np.asarray(src["y"]["test"], dtype=np.float64)
    return payload


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def atom_indices(cache: Mapping[str, Any], rows: Sequence[int]) -> np.ndarray:
    """Flat per-atom indices (cache order) for the given molecules."""
    ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    return np.concatenate([np.arange(ptr[r], ptr[r + 1], dtype=np.int64) for r in rows])


def collate(cache: Mapping[str, Any], rows: Sequence[int], feats: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Frozen collate plus the fixed structural vectors of this round."""
    batch = C.PREP_DATA.collate(cache, list(rows))
    if feats is not None:
        if [int(x) for x in np.asarray(feats["rows"]).tolist()] != [int(x) for x in np.asarray(cache["rows"]).tolist()]:
            raise ValueError("feature rows differ from cache rows")
        if not np.array_equal(np.asarray(feats["atom_ptr"]), np.asarray(cache["atom_ptr"])):
            raise ValueError("feature atom_ptr differs from cache atom_ptr")
        idx = atom_indices(cache, rows)
        for key in ("s3", "s2"):
            if key in feats:
                batch[key] = torch.from_numpy(np.asarray(feats[key], dtype=np.float32)[idx])
        for key in ("u3", "u2"):
            if key in feats:
                batch[key] = torch.from_numpy(np.asarray(feats[key], dtype=np.float32)[idx])
    return batch


def predict_scores(
    model,
    cache: Mapping[str, Any],
    device,
    *,
    batch_size: int = C.EVAL_BATCH_SIZE,
    feats: Mapping[str, Any] | None = None,
) -> np.ndarray:
    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = to_device(collate(cache, rows, feats), device)
            pred = model(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def predict_rows(
    model,
    cache: Mapping[str, Any],
    cache_indices: Sequence[int],
    device,
    *,
    feats: Mapping[str, Any] | None = None,
) -> np.ndarray:
    batch = to_device(collate(cache, list(cache_indices), feats), device)
    with torch.no_grad():
        return model(batch).detach().float().cpu().numpy().reshape(-1).astype(np.float64)


__all__ = [
    "CACHE_VERSION",
    "frozen_spec",
    "build_split_cache",
    "load_cache",
    "build_all_caches",
    "train_norm",
    "target_norm",
    "load_dataset",
    "to_device",
    "atom_indices",
    "collate",
    "predict_scores",
    "predict_rows",
]
