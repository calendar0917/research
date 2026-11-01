"""Label-free structural cache, split views and Fit-only normalisation for ESOL.

The ESOL cache is built with the frozen ``lipo_v1`` pipeline, not with a new
feature encoder:

    raw graph row --adapters.lipo_row_to_graph--> MolecularGraph
                  --lcrc_data.build_cache----> S1/S2/atom/bond/global tensors

Cache construction needs no RDKit, so the GPU hosts (which do not install it)
can rebuild it deterministically from the committed raw-graph artifact.

Scope discipline
----------------
* the split manifest contains row ids and scaffolds only;
* ``build_views(include_test=False)`` — the only view used by training — slices
  Train and Validation, fits the structural normaliser on Train, and fits the
  target scaler on Train labels.  The Test cache is neither built nor selected;
  the Test label slice is never materialised, summarised or used for selection
  before the checkpoint manifest is frozen;
* ``build_views(include_test=True)`` is called exactly once, by the one-shot
  Test evaluation, after that freeze.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import esol_common as C
import esol_raw as R
import esol_split as S

import adapters as PREP_ADAPTERS  # noqa: E402
import lcrc_data as PREP_DATA  # noqa: E402

CACHE_FILE = C.CACHE / "lcrc_esol_all1128.pt"


# ---------------------------------------------------------------------------
# structural cache
# ---------------------------------------------------------------------------


def build_label_free_cache(*, force: bool = False) -> dict[str, Any]:
    spec = C.spec()
    if CACHE_FILE.exists() and not force:
        cache = PREP_DATA.load_cache(CACHE_FILE)
        if int(cache["n_molecules"]) != C.N_MOLECULES:
            raise ValueError("cached ESOL structural cache has the wrong molecule count")
        if cache["meta"].get("labels_present") is not False:
            raise PermissionError("cached ESOL structural cache is not certified label-free")
        return {
            "cache": cache,
            "path": CACHE_FILE,
            "file_sha256": C.sha256_file(CACHE_FILE),
            "content_fingerprint": C.PREP_COMMON.content_fingerprint(cache),
            "rebuilt": False,
        }
    raw = R.load_raw_graphs()
    graphs = [PREP_ADAPTERS.lipo_row_to_graph(raw["rows"][i], i, spec) for i in range(C.N_MOLECULES)]
    cache = PREP_DATA.build_cache(graphs, spec, split="esol_all1128")
    if cache["meta"].get("labels_present") is not False:
        raise PermissionError("ESOL structural cache is not certified label-free")
    if [str(x) for x in cache["smiles"]] != [str(row["smiles"]) for row in raw["rows"]]:
        raise ValueError("ESOL cache SMILES identity drifted from the raw artifact")
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    digest = PREP_DATA.save_cache(cache, CACHE_FILE)
    return {
        "cache": cache,
        "path": CACHE_FILE,
        "file_sha256": digest,
        "content_fingerprint": C.PREP_COMMON.content_fingerprint(cache),
        "rebuilt": True,
    }


# ---------------------------------------------------------------------------
# cache subsetting
# ---------------------------------------------------------------------------


def select_cache(cache: Mapping[str, Any], positions: Sequence[int], *, split_label: str) -> dict[str, Any]:
    pos = [int(p) for p in positions]
    if len(pos) != len(set(pos)):
        raise ValueError("duplicate cache position")
    if any(p < 0 or p >= int(cache["n_molecules"]) for p in pos):
        raise IndexError("cache position outside the ESOL cache")

    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_ranges = [(int(atom_ptr[p]), int(atom_ptr[p + 1])) for p in pos]
    bond_ranges = [(int(bond_ptr[p]), int(bond_ptr[p + 1])) for p in pos]
    nbr_ranges = [(int(nbr_ptr[a0]), int(nbr_ptr[a1])) for a0, a1 in atom_ranges]
    path_ranges = [(int(path_ptr[a0]), int(path_ptr[a1])) for a0, a1 in atom_ranges]

    def cat_rows(key: str, ranges, dtype=None) -> np.ndarray:
        parts = [np.asarray(cache[key])[lo:hi] for lo, hi in ranges]
        arr = np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.asarray(cache[key]).dtype)
        return arr if dtype is None else arr.astype(dtype)

    def cat_flat(key: str, ranges) -> np.ndarray:
        parts = [np.asarray(cache[key])[lo:hi] for lo, hi in ranges]
        base = np.asarray(cache[key])
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=base.dtype)

    atom_counts = np.asarray([hi - lo for lo, hi in atom_ranges], dtype=np.int64)
    bond_counts = np.asarray([hi - lo for lo, hi in bond_ranges], dtype=np.int64)

    out: dict[str, Any] = {
        "atom_feat": cat_rows("atom_feat", atom_ranges),
        "atom_type": cat_rows("atom_type", atom_ranges),
        "atom_scalar": cat_rows("atom_scalar", atom_ranges),
        "bond_feat": cat_rows("bond_feat", bond_ranges),
        "bond_type": cat_rows("bond_type", bond_ranges),
        "bond_index": cat_rows("bond_index", bond_ranges),
        "nbr_code": cat_flat("nbr_code", nbr_ranges),
        "path_cat": cat_flat("path_cat", path_ranges),
        "path_end63": cat_flat("path_end63", path_ranges),
        "path_end": cat_flat("path_end", path_ranges),
        "path_closure": cat_flat("path_closure", path_ranges),
        "global": np.asarray(cache["global"])[np.asarray(pos, dtype=np.int64)].copy(),
        "n_components": np.asarray(cache["n_components"])[np.asarray(pos, dtype=np.int64)].copy(),
        "n_junctions": np.asarray(cache["n_junctions"])[np.asarray(pos, dtype=np.int64)].copy(),
        "n_ring_systems": np.asarray(cache["n_ring_systems"])[np.asarray(pos, dtype=np.int64)].copy(),
        "molecule_ids": [str(cache["molecule_ids"][p]) for p in pos],
        "smiles": [str(cache["smiles"][p]) for p in pos],
        "atom_ptr": np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int64),
        "bond_ptr": np.concatenate([[0], np.cumsum(bond_counts)]).astype(np.int64),
        "nbr_ptr": np.concatenate(
            [[0], np.cumsum(np.concatenate([np.asarray(cache["nbr_ptr"])[a0 + 1 : a1 + 1] - np.asarray(cache["nbr_ptr"])[a0:a1] for a0, a1 in atom_ranges]))]
        ).astype(np.int64)
        if atom_ranges
        else np.zeros(1, dtype=np.int64),
        "path_ptr": np.concatenate(
            [[0], np.cumsum(np.concatenate([np.asarray(cache["path_ptr"])[a0 + 1 : a1 + 1] - np.asarray(cache["path_ptr"])[a0:a1] for a0, a1 in atom_ranges]))]
        ).astype(np.int64)
        if atom_ranges
        else np.zeros(1, dtype=np.int64),
        "n_molecules": len(pos),
        "split": str(split_label),
    }
    meta = dict(cache.get("meta", {}))
    meta["n_molecules"] = len(pos)
    meta["subset_of"] = str(cache.get("split", ""))
    meta["subset_positions_sha256"] = C.rows_sha256(pos)
    meta["labels_present"] = False
    out["meta"] = meta
    return out


# ---------------------------------------------------------------------------
# target / physchem normalisation (Train only)
# ---------------------------------------------------------------------------


def fit_target_scaler(y_train: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    scale = float(y.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Train target standard deviation is not positive")
    return {
        "mean": float(y.mean()),
        "scale": scale,
        "source": "esol_train_labels_only",
        "fit_on": "ESOL Train split only",
        "n": int(y.size),
        "valid_or_test_used": False,
        "units": "log(mol/L)",
    }


def fit_physchem_norm(matrix: np.ndarray, train_rows: Sequence[int]) -> dict[str, Any]:
    rows = np.asarray(train_rows, dtype=np.int64)
    train = np.asarray(matrix, dtype=np.float64)[rows]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    const = [int(i) for i in np.flatnonzero(std <= 1e-6)]
    scale = np.where(std > 1e-6, std, 1.0)
    return {
        "names": list(C.PHYSCHEM_NAMES),
        "mean": [float(x) for x in mean],
        "scale": [float(x) for x in scale],
        "constant_indices": const,
        "fit_on": "ESOL Train split only",
        "n": int(rows.size),
        "valid_or_test_used": False,
        "source": "recomputed_from_smiles_rdkit",
    }


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


def build_views(*, include_test: bool) -> dict[str, Any]:
    split = S.load_split()
    csv_payload = R.read_csv()
    spec = C.spec()
    built = build_label_free_cache()
    cache_all = built["cache"]
    if [str(x) for x in cache_all["molecule_ids"]] != [str(i) for i in range(C.N_MOLECULES)]:
        raise PermissionError("ESOL structural cache is not in CSV row order")
    if [str(x) for x in cache_all["smiles"]] != [str(s) for s in csv_payload["smiles"]]:
        raise PermissionError("ESOL cache SMILES order differs from the CSV")

    rows = {name: [int(x) for x in split["rows"][name]] for name in ("train", "valid", "test")}
    names = ["train", "valid"] + (["test"] if include_test else [])
    caches: dict[str, Any] = {}
    for name in names:
        caches[name] = select_cache(cache_all, rows[name], split_label=f"esol_{name}")
        expected = [csv_payload["smiles"][row] for row in rows[name]]
        if [str(x) for x in caches[name]["smiles"]] != expected:
            raise ValueError(f"ESOL {name}: cache SMILES identity drifted")

    train_norm = PREP_DATA.fit_norm(caches["train"], spec, source="esol_train_structures")
    y_all = np.asarray(csv_payload["y"], dtype=np.float64)
    y_train = y_all[np.asarray(rows["train"], dtype=np.int64)]
    y_valid = y_all[np.asarray(rows["valid"], dtype=np.int64)]
    target_norm = fit_target_scaler(y_train)

    physchem = R.load_physchem()
    physchem_norm = fit_physchem_norm(physchem["matrix"], rows["train"])

    labels: dict[str, Any] = {
        "y_train": y_train,
        "y_valid": y_valid,
        "audit": {
            "source": "hash-pinned DeepChem delaney-processed.csv + frozen split manifest",
            "y_train_sha256": C.vector_sha256_float64(y_train),
            "y_valid_sha256": C.vector_sha256_float64(y_valid),
            "y_all_rows_present": int(y_all.size) == C.N_MOLECULES,
        },
    }
    if include_test:
        labels["y_test"] = y_all[np.asarray(rows["test"], dtype=np.int64)]
        labels["audit"]["y_test_sha256"] = C.vector_sha256_float64(labels["y_test"])

    data_override = {
        "split_file": C.track_rel(C.SPLIT_MANIFEST),
        "split_file_sha256": C.sha256_file(C.SPLIT_MANIFEST),
        "split_audit_sha256": C.sha256_file(C.SPLIT_AUDIT),
        "labels_file": C.track_rel(C.CSV_PATH),
        "labels_file_sha256": C.CSV_SHA256,
        "raw_graph_file": C.track_rel(C.RAW_GRAPH),
        "raw_graph_file_sha256": C.sha256_file(C.RAW_GRAPH),
        "train_rows_sha256": C.rows_sha256(rows["train"]),
        "valid_rows_sha256": C.rows_sha256(rows["valid"]),
        "test_rows_sha256": C.rows_sha256(rows["test"]),
        "n_train": len(rows["train"]),
        "n_valid": len(rows["valid"]),
        "n_test": len(rows["test"]),
        "split_scope": "esol_bemis_murcko_scaffold_seed_20261103",
        "lcrc_cache_path": C.track_rel(built["path"]),
        "lcrc_cache_file_sha256": built["file_sha256"],
        "lcrc_cache_content_fingerprint": built["content_fingerprint"],
        "lcrc_cache_labels_present": False,
        "feature_spec_hash": spec.hash(),
        "train_norm_fingerprint": PREP_DATA.norm_fingerprint(train_norm),
        "target_norm": dict(target_norm),
        "physchem_norm": dict(physchem_norm),
        "physchem_file": C.track_rel(C.PHYSCHEM_FILE),
        "physchem_file_sha256": C.sha256_file(C.PHYSCHEM_FILE),
        "test_row_values_used": False,
        "label_scope": (
            "train_and_validation_only_during_training"
            if not include_test
            else "test_labels_loaded_after_manifest_freeze"
        ),
    }
    return {
        "split": {"rows": rows, "smiles": csv_payload["smiles"], "name": "esol_scaffold_seed_20261103"},
        "labels": labels,
        "spec": spec,
        "physchem": physchem["matrix"],
        "physchem_norm": physchem_norm,
        "train_cache": caches["train"],
        "valid_cache": caches["valid"],
        "test_cache": caches.get("test"),
        "train_norm": train_norm,
        "train_norm_fingerprint": PREP_DATA.norm_fingerprint(train_norm),
        "target_norm": target_norm,
        "cache_built": {k: v for k, v in built.items() if k != "cache"},
        "split_seed": C.SPLIT_SEED,
        "split_scope": "esol_bemis_murcko_scaffold_seed_20261103",
        "data_override": data_override,
        "protocol_override": {
            "esol_protocol_id": C.PROTOCOL_ID,
            "esol_round_id": C.ROUND_ID,
            "esol_split_seed": C.SPLIT_SEED,
            "esol_split_manifest_sha256": C.sha256_file(C.SPLIT_MANIFEST),
        },
        "y_test": labels.get("y_test"),
    }


__all__ = [
    "CACHE_FILE",
    "build_label_free_cache",
    "select_cache",
    "fit_target_scaler",
    "fit_physchem_norm",
    "build_views",
]
