"""TU structural cache assembly, the single frozen 8:1:1 split and Fit-only norm.

The cache layout is identical to the frozen LCRC-A data layer
(``lcrc_cross_task_preparation_v1/code/lcrc_data.py``), so the *same*
``collate`` / ``fit_norm`` / ``norm_fingerprint`` code is reused unchanged:
S1 joint codes, typed S2 path records, per-node scalars, global statistics and
explicit pointer structures.  Only the per-dataset feature/spec construction is
new (``tu_adapter.py``).

No label is stored in a structural cache.  Labels live in separate
``analysis/<dataset>_y_<split>.npy`` files, so a training run can load Train and
Valid labels without the Test labels being part of its inputs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import tu_common as C  # noqa: E402  (inserts the frozen prep code path)

import lcrc_data as PREP_DATA  # noqa: E402
import molecular_graph as MG  # noqa: E402
import tu_adapter as A  # noqa: E402
from feature_spec import FeatureSpec, load_spec, save_spec  # noqa: E402


# ---------------------------------------------------------------------------
# per-molecule arrays
# ---------------------------------------------------------------------------


def build_tu_molecule(
    graph: MG.MolecularGraph,
    analysis: MG.GraphAnalysis,
    spec: FeatureSpec,
) -> dict[str, Any]:
    """One graph's label-free TU arrays in the frozen LCRC-A cache layout."""
    s1_records = MG.enumerate_s1(graph.atom_type, analysis.adj, spec.n_bond_types)
    s2_records = MG.enumerate_s2(graph.atom_type, analysis.adj, spec.n_bond_types, analysis.dist)

    nbr_counts = np.asarray([len(r) for r in s1_records], dtype=np.int64)
    nbr_ptr = np.concatenate([[0], np.cumsum(nbr_counts)]).astype(np.int64)
    nbr_code = (
        np.asarray([code for row in s1_records for code in row], dtype=np.int32)
        if any(s1_records)
        else np.zeros(0, dtype=np.int32)
    )

    path_counts = np.asarray([len(r) for r in s2_records], dtype=np.int64)
    path_ptr = np.concatenate([[0], np.cumsum(path_counts)]).astype(np.int64)
    if int(path_ptr[-1]):
        flat = [rec for row in s2_records for rec in row]
        path_cat = np.asarray([rec["path_cat"] for rec in flat], dtype=np.int32)
        path_end63 = np.asarray([rec["end63"] for rec in flat], dtype=np.int32)
        path_end = np.asarray([rec["end"] for rec in flat], dtype=np.int64)
        path_closure = np.asarray([rec["closure"] for rec in flat], dtype=np.int8)
    else:
        path_cat = np.zeros(0, dtype=np.int32)
        path_end63 = np.zeros(0, dtype=np.int32)
        path_end = np.zeros(0, dtype=np.int64)
        path_closure = np.zeros(0, dtype=np.int8)

    mol: dict[str, Any] = {
        "n_atoms": np.int64(analysis.n_atoms),
        "n_bonds": np.int64(analysis.n_bonds),
        "atom_feat": np.asarray(graph.atom_feat, dtype=np.float32),
        "atom_type": np.asarray(graph.atom_type, dtype=np.int64),
        "atom_scalar": A.build_atom_scalars(analysis, s2_records),
        "bond_feat": np.asarray(graph.bond_feat, dtype=np.float32),
        "bond_type": np.asarray(graph.bond_type, dtype=np.int64),
        "bond_index": np.stack([analysis.bond_u, analysis.bond_v], axis=1).astype(np.int64)
        if analysis.n_bonds
        else np.zeros((0, 2), dtype=np.int64),
        "nbr_ptr": nbr_ptr,
        "nbr_code": nbr_code,
        "path_ptr": path_ptr,
        "path_cat": path_cat,
        "path_end63": path_end63,
        "path_end": path_end,
        "path_closure": path_closure,
        "global": A.build_global_features(graph, analysis),
        "n_components": np.int64(analysis.n_components),
        "n_junctions": np.int64(analysis.n_junctions),
        "n_ring_systems": np.int64(analysis.n_ring_systems),
        "molecule_id": str(graph.molecule_id),
        "smiles": "",
    }
    PREP_DATA.validate_molecule(mol, spec)
    return mol


# ---------------------------------------------------------------------------
# cache assembly
# ---------------------------------------------------------------------------


def build_split_cache(
    dataset: str,
    records: Sequence[Mapping[str, Any]],
    rows: np.ndarray,
    spec: FeatureSpec,
    node_types: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    mols = []
    for i, gid in enumerate(np.asarray(rows, dtype=np.int64).tolist()):
        rec = records[int(gid)]
        graph, analysis = A.build_tu_graph(
            rec, spec, node_types, molecule_id=f"{dataset}#{int(gid)}"
        )
        mols.append(build_tu_molecule(graph, analysis, spec))
        if (i + 1) % 1000 == 0:
            C.log(f"[cache {dataset}/{split}] {i + 1}/{len(rows)} graphs {time.perf_counter() - t0:.1f}s")
    cache = PREP_DATA.concat_molecules(mols, spec, split=split)
    cache["gids"] = np.asarray(rows, dtype=np.int64).copy()
    cache["meta"]["build_seconds"] = float(time.perf_counter() - t0)
    cache["meta"]["dataset"] = dataset
    return cache


def cache_summary(cache: Mapping[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    checked = PREP_DATA.validate_cache(cache, spec)
    m = cache["meta"]
    return {
        "split": str(cache["split"]),
        "n_graphs": int(cache["n_molecules"]),
        "n_atoms": int(m["n_atoms_total"]),
        "n_bonds": int(m["n_bonds_total"]),
        "n_s1_codes": int(m["n_s1_codes"]),
        "n_s2_paths": int(m["n_s2_paths"]),
        "validation": checked,
        "gids_sha256": C.index_hash(cache["gids"]),
    }


# ---------------------------------------------------------------------------
# full dataset preparation
# ---------------------------------------------------------------------------


def prepare_dataset(dataset: str, *, force: bool = False) -> dict[str, Any]:
    """Load, split, build the three structural caches and the Train-only norm."""
    meta = C.DATASETS[dataset]
    ds = A.load_dataset(dataset)
    labels = ds["labels"]
    split = A.build_split(labels, meta["n_train"], meta["n_valid"], meta["n_test"])
    split_payload = A.split_record(dataset, split, labels)

    if meta["node_type"] == "label":
        train_labels: list[int] = []
        for gid in split["train"].tolist():
            train_labels.extend(int(x) for x in np.asarray(ds["records"][gid]["node_labels"]).tolist())
    else:
        train_labels = None
    node_types = A.node_type_table(meta["node_type"], train_labels)
    spec = A.build_tu_spec(dataset, node_types)

    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    spec_path = C.ANALYSIS / f"{dataset}_spec.json"
    if spec_path.exists() and not force:
        old = load_spec(spec_path)
        if old.hash() != spec.hash():
            raise RuntimeError(f"{dataset}: regenerated spec hash differs from the stored spec")
    save_spec(spec, spec_path)

    caches: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    local_cache_files: dict[str, Any] = {}
    for name in ("train", "valid", "test"):
        path = C.CACHE / f"{dataset}_{name}.pt"
        if path.exists() and not force:
            cache = PREP_DATA.load_cache(path)
            if str(cache["meta"].get("feature_spec_hash")) != spec.hash():
                raise RuntimeError(f"{dataset}/{name}: cached features use a different spec hash")
            if not np.array_equal(np.asarray(cache["gids"]), split[name]):
                raise RuntimeError(f"{dataset}/{name}: cached gids differ from the frozen split")
        else:
            cache = build_split_cache(dataset, ds["records"], split[name], spec, node_types, name)
            PREP_DATA.save_cache(cache, path)
        if not force:
            cache["meta"].setdefault("build_seconds", 0.0)
        caches[name] = cache
        summaries[name] = cache_summary(cache, spec)
        local_cache_files[name] = {
            "path": C.track_rel(path),
            "bytes": int(path.stat().st_size),
            "sha256": C.sha256_file(path),
            "build_seconds": float(cache["meta"]["build_seconds"]),
        }
        np.save(C.ANALYSIS / f"{dataset}_y_{name}.npy", labels[split[name]])

    norm = PREP_DATA.fit_norm(caches["train"], spec, source=f"train_split_only:{dataset}")
    norm_path = C.ANALYSIS / f"{dataset}_norm.json"
    norm_payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()}
    norm_payload["norm_fingerprint"] = PREP_DATA.norm_fingerprint(norm)
    norm_payload["spec_hash"] = spec.hash()
    C.write_json(norm_path, norm_payload)

    # Fit-only normalisation audit: the Valid/Test means must not be used.
    train_scalar_mean = np.asarray(caches["train"]["atom_scalar"], dtype=np.float64).mean(0)
    train_global_mean = np.asarray(caches["train"]["global"], dtype=np.float64).mean(0)
    norm_audit = {
        "scalar_mean_matches_train": bool(
            np.allclose(train_scalar_mean.astype(np.float32), np.asarray(norm["scalar_mean"], dtype=np.float32), atol=0.0, rtol=0.0)
        ),
        "global_mean_matches_train": bool(
            np.allclose(train_global_mean.astype(np.float32), np.asarray(norm["global_mean"], dtype=np.float32), atol=0.0, rtol=0.0)
        ),
        "fit_split": str(norm["fit_split"]),
        "valid_or_test_used": bool(norm["valid_or_test_used"]),
        "zero_variance_policy": str(norm["zero_variance_policy"]),
        "valid_global_mean": np.asarray(caches["valid"]["global"], dtype=np.float64).mean(0).tolist(),
        "test_global_mean": np.asarray(caches["test"]["global"], dtype=np.float64).mean(0).tolist(),
        "scalar_constant_indices": list(norm["scalar_constant_indices"]),
        "global_constant_indices": list(norm["global_constant_indices"]),
        "spec_hash": spec.hash(),
    }
    if not (norm_audit["scalar_mean_matches_train"] and norm_audit["global_mean_matches_train"]):
        raise RuntimeError(f"{dataset}: normalisation means do not equal the Train means")
    if norm_audit["valid_or_test_used"]:
        raise RuntimeError(f"{dataset}: normalisation used Valid/Test")

    payload = {
        "dataset": dataset,
        "n_classes": int(meta["n_classes"]),
        "node_type_mode": str(meta["node_type"]),
        "node_type_names": list(node_types["names"]),
        "node_type_unk_index": int(node_types["unk_index"]),
        "train_vocab": list(node_types["train_vocab"]),
        "n_node_types": len(node_types["names"]),
        "atom_feat_dim": spec.atom_feat_dim,
        "bond_feat_dim": spec.bond_feat_dim,
        "n_scalars": spec.n_scalars,
        "global_dim": spec.global_dim,
        "n_nbr_codes": spec.n_nbr_codes,
        "n_path_codes": spec.n_path_codes,
        "feature_spec_hash": spec.hash(),
        "spec_path": C.track_rel(spec_path),
        "norm_path": C.track_rel(norm_path),
        "norm_fingerprint": norm_payload["norm_fingerprint"],
        "split": split_payload,
        "caches": summaries,
        "norm_audit": norm_audit,
        "dataset_stats": ds["stats"],
        "raw_manifest": A.raw_manifest(dataset),
        "local_cache_files": local_cache_files,
        "measurement": {"prepared_utc": C.utcnow()},
    }
    return payload


def load_split_cache(dataset: str, split: str) -> tuple[dict[str, Any], np.ndarray]:
    """Load one split's structural cache and labels (same spec-hash guard)."""
    spec = load_spec(C.ANALYSIS / f"{dataset}_spec.json")
    cache = PREP_DATA.load_cache(C.CACHE / f"{dataset}_{split}.pt")
    labels = np.load(C.ANALYSIS / f"{dataset}_y_{split}.npy")
    if str(cache["meta"]["feature_spec_hash"]) != spec.hash():
        raise RuntimeError(f"{dataset}/{split}: cache spec hash mismatch")
    if labels.size != int(cache["n_molecules"]):
        raise RuntimeError(f"{dataset}/{split}: label count mismatch")
    return cache, labels


def load_prepared(dataset: str, splits: Sequence[str] = ("train", "valid", "test")) -> dict[str, Any]:
    """Load the frozen spec, split, norm and the requested split caches."""
    meta = C.DATASETS[dataset]
    spec = load_spec(C.ANALYSIS / f"{dataset}_spec.json")
    norm = C.read_json(C.ANALYSIS / f"{dataset}_norm.json")
    norm_np = {
        k: np.asarray(norm[k], dtype=np.float32)
        for k in ("scalar_mean", "scalar_scale", "global_mean", "global_scale")
    }
    split = C.read_json(C.SPLIT_FILE)[dataset]
    caches = {name: PREP_DATA.load_cache(C.CACHE / f"{dataset}_{name}.pt") for name in splits}
    labels = {name: np.load(C.ANALYSIS / f"{dataset}_y_{name}.npy") for name in splits}
    for name, cache in caches.items():
        if str(cache["meta"]["feature_spec_hash"]) != spec.hash():
            raise RuntimeError(f"{dataset}/{name}: cache spec hash mismatch")
        if labels[name].size != int(cache["n_molecules"]):
            raise RuntimeError(f"{dataset}/{name}: label count mismatch")
    return {
        "dataset": dataset,
        "n_classes": int(meta["n_classes"]),
        "spec": spec,
        "norm": norm_np,
        "norm_record": norm,
        "split": split,
        "caches": caches,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# analysis-file writers
# ---------------------------------------------------------------------------


def write_round_manifest(prepared: Mapping[str, Mapping[str, Any]]) -> Path:
    payload = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "tu_root": C.track_rel(C.TU_ROOT),
        "loader": {
            "interface": "torch_geometric.datasets.TUDataset",
            "cleaned": False,
            "use_node_attr": False,
            "use_edge_attr": False,
            "docs": [C.TU_PYG_DOCS, C.TU_DATASETS_DOCS],
            "source_archive": C.TU_ARCHIVE_ROOT,
        },
        "datasets": {
            name: {
                "raw_manifest": p["raw_manifest"],
                "dataset_stats": p["dataset_stats"],
                "node_type_mode": p["node_type_mode"],
                "train_vocab": p["train_vocab"],
                "feature_spec_hash": p["feature_spec_hash"],
                "atom_feat_dim": p["atom_feat_dim"],
                "bond_feat_dim": p["bond_feat_dim"],
                "global_dim": p["global_dim"],
                "n_nbr_codes": p["n_nbr_codes"],
                "n_path_codes": p["n_path_codes"],
                "split": p["split"],
                "caches": p["caches"],
                "norm_audit": p["norm_audit"],
                "norm_fingerprint": p["norm_fingerprint"],
            }
            for name, p in prepared.items()
        },
        "recipe": C.RECIPE,
        "model_config": C.MODEL_CONFIG,
        "gin_config": C.GIN_CONFIG,
        "model_widths": {
            "atom": "128 -> 96",
            "bond": "160 -> 96",
            "stat_rank": 32,
            "local_hidden": 192,
            "h": 64,
            "rho_hidden": 64,
        },
        "run_matrix": [{"dataset": d, "model": m} for d, m in C.RUN_MATRIX],
    }
    return C.write_json(C.DATA_MANIFEST, payload)


def write_local_prep_report(prepared: Mapping[str, Mapping[str, Any]], host: str) -> Path:
    """Host-local prep evidence (git-ignored): cache file hashes and build times."""
    payload = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "host": str(host),
        "environment": C.environment(),
        "code_state": C.code_state(),
        "datasets": {
            name: {
                "cache_files": p.get("local_cache_files", {}),
                "spec_path": p["spec_path"],
                "norm_path": p["norm_path"],
                "prepared_utc": p["measurement"]["prepared_utc"],
            }
            for name, p in prepared.items()
        },
    }
    return C.write_json(C.RUNS / "_prep" / f"prep_local_{host}.json", payload)


def write_split_file(prepared: Mapping[str, Mapping[str, Any]]) -> Path:
    payload = {name: p["split"] for name, p in prepared.items()}
    payload["_protocol"] = {
        "round_id": C.ROUND_ID,
        "split_seed": C.SPLIT_SEED,
        "training_seed": C.TRAINING_SEED,
        "procedure": "two stratified train_test_split calls, random_state=0; never re-drawn",
        "result_label": C.RESULT_LABEL,
    }
    return C.write_json(C.SPLIT_FILE, payload)


__all__ = [
    "build_tu_molecule",
    "build_split_cache",
    "cache_summary",
    "prepare_dataset",
    "load_prepared",
    "load_split_cache",
    "write_round_manifest",
    "write_local_prep_report",
    "write_split_file",
]
