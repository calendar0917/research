"""Label-free structure caches, Train-only normalisation and prediction helpers.

The computational graph is the frozen ``lipo_v1`` / LCRC-A one.  The structure
pipeline is called through the dissection round's own builder; only the cache
destination is new.  No target byte is stored in a structure cache.

Target normalisation (``mean`` / ``std(ddof=0)``) and structure normalisation
are fitted on the published Train3360 only and shared by every arm.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import common as C
import lcrc_common as PREP_COMMON
import lcrc_data as PREP_DATA

CACHE_VERSION = "lcrc-lipo-edit-supervision-v1"


def cache_path(split: str) -> Path:
    return C.CACHE / f"Lipophilicity_r{C.DATA_REPEAT}_{split}.pt"


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
    """Build one cache with the frozen dissection builder, then store it here."""
    cache = C.DD.build_split_cache(
        C.DATA_REPEAT, split, list(rows), list(smiles), spec, save=False
    )
    if cache["meta"].get("labels_present"):
        raise ValueError("structure cache must be label-free")
    cache["round_id"] = C.ROUND_ID
    cache["upstream_cache_version"] = cache.get("cache_version")
    cache["cache_version"] = CACHE_VERSION
    cache["meta"]["round_id"] = C.ROUND_ID
    cache["meta"]["content_fingerprint"] = PREP_COMMON.content_fingerprint(cache)
    _atomic_save(cache, cache_path(split))
    return cache


def load_cache(split: str, *, require: bool = True, rebuild: bool = False) -> dict[str, Any]:
    spec = frozen_spec()
    src = C.source_with_labels("none")
    rows = [int(r) for r in src["rows"][split]]
    path = cache_path(split)
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
    if require:
        raise FileNotFoundError(f"{path} is missing; run code/prepare.py --cache-only on this host first")
    return build_split_cache(split, rows, src["smiles"], spec)


def build_all_caches(*, rebuild: bool = False) -> dict[str, Any]:
    spec = frozen_spec()
    src = C.source_with_labels("none")
    entries: dict[str, Any] = {}
    for split in C.SPLITS:
        rows = src["rows"][split]
        cache = load_cache(split, require=False, rebuild=rebuild) if not rebuild else build_split_cache(
            split, rows, src["smiles"], spec
        )
        if not rebuild:
            # load_cache above either loaded a complete cache or built+saved it.
            cache = load_cache(split, require=True)
        entries[split] = {
            "n_molecules": int(cache["n_molecules"]),
            "rows_sha256": C.row_sha256(np.asarray(cache["rows"]).tolist()),
            "content_fingerprint": cache["meta"]["content_fingerprint"],
            "cache_path": C.track_rel(cache_path(split)),
            "cache_sha256": C.sha256_file(cache_path(split)),
            "labels_present": bool(cache["meta"]["labels_present"]),
        }
        C.log(
            f"[cache] {split}: n={cache['n_molecules']} "
            f"fp={cache['meta']['content_fingerprint'][:16]}"
        )
    payload = {
        "schema": "lcrc-lipo-edit-supervision-cache-manifest-v1",
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
    norm = PREP_DATA.fit_norm(cache, spec, source=f"Lipophilicity_r{C.DATA_REPEAT}_train_structures_only")
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


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return PREP_DATA.collate(cache, list(rows))


def predict_scores(model, cache: Mapping[str, Any], device, *, batch_size: int = C.EVAL_BATCH_SIZE) -> np.ndarray:
    return C.DD.predict_scores(model, cache, device, batch_size=batch_size)


def predict_rows(model, cache: Mapping[str, Any], cache_indices: Sequence[int], device) -> np.ndarray:
    batch = to_device(collate(cache, list(cache_indices)), device)
    with torch.no_grad():
        return model(batch).detach().float().cpu().numpy().reshape(-1).astype(np.float64)


__all__ = [
    "CACHE_VERSION",
    "cache_path",
    "frozen_spec",
    "build_split_cache",
    "load_cache",
    "build_all_caches",
    "train_norm",
    "target_norm",
    "load_dataset",
    "to_device",
    "collate",
    "predict_scores",
    "predict_rows",
]
