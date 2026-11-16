"""Data layer: official OGB rows, the frozen caches, the ``lipo_v1`` split caches.

Everything here is *read-only reuse* of the previous rounds:

* ``sq_data.load_cache`` / ``split_indices`` / ``load_lipo_raw`` — the committed,
  SHA-256-verified official OGB graph cache and the frozen ``lipo_v1`` raw
  attribute cache;
* ``sq_lcrc.build_caches`` — the ``lipo_v1`` FeatureSpec, the molecular-graph
  adapter, the per-split cache + collate layer and the **Train-only** feature
  normalisation.

This round adds three things on top and nothing else:

1. a real byte-level re-verification of the four frozen caches;
2. the split-local -> official global ``row_id`` mapping (never 0..N_split-1);
3. the Train-only *active global* mask ``J`` (raw span > 1e-8) plus the two
   frozen context fields, resolved by field name.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cr_common as C

_PATHS_READY = False


def prepare_imports() -> None:
    """Put the read-only sibling tracks on ``sys.path`` (once)."""
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (str(C.SQL_CODE), str(C.PREP_CODE), str(C.DISSECT_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_READY = True


def modules() -> dict[str, Any]:
    prepare_imports()
    import adapters  # noqa: F401  (lipo_v1 raw row -> MolecularGraph)
    import dissect_common
    import dissect_model
    import feature_spec
    import lcrc_data
    import sq_common
    import sq_data
    import sq_lcrc

    return {
        "adapters": adapters,
        "dissect_common": dissect_common,
        "dissect_model": dissect_model,
        "feature_spec": feature_spec,
        "lcrc_data": lcrc_data,
        "sq_common": sq_common,
        "sq_data": sq_data,
        "sq_lcrc": sq_lcrc,
    }


# ---------------------------------------------------------------------------
# frozen active-global mask and context fields
# ---------------------------------------------------------------------------

#: A raw global column is "active" when its Train span exceeds this threshold.
ACTIVE_SPAN_EPS = C.ACTIVE_SPAN_EPS
#: The two frozen, named context fields of arm C (fixed in advance).
CONTEXT_FIELDS: tuple[str, ...] = C.CONTEXT_FIELDS
CONTEXT_QUANTILES: tuple[float, float] = C.CONTEXT_QUANTILES


@dataclass
class Bundle:
    """Everything one task needs, built once per process."""

    dataset: str
    payload: Mapping[str, Any]
    lipo_raw: Mapping[str, Any]
    spec: Any
    norm: Mapping[str, Any]
    caches: dict[str, dict[str, Any]]
    global_row_ids: dict[str, np.ndarray]
    targets_raw: dict[str, np.ndarray]
    scaler: dict[str, float]
    y_std: dict[str, np.ndarray]
    n_atoms: dict[str, np.ndarray]
    smiles: dict[str, list[str]]
    active_global: dict[str, Any]
    contexts: dict[str, Any]
    collate: Any = None
    build_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def split_sizes(self) -> dict[str, int]:
        return {split: int(self.caches[split]["n_molecules"]) for split in C.SPLITS}

    def row_ids_sorted(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """(order, row_ids) putting a split cache in ascending official row_id order."""
        ids = self.global_row_ids[split]
        order = np.argsort(ids, kind="stable")
        return order, ids[order]

    def targets_sorted(self, split: str) -> np.ndarray:
        order, _ = self.row_ids_sorted(split)
        return self.targets_raw[split][order]


# ---------------------------------------------------------------------------
# cache loading / verification
# ---------------------------------------------------------------------------


def cache_filename(dataset: str) -> str:
    return f"{dataset.replace('-', '_')}_sq-pilot-cache-v1.npz"


def lipo_raw_filename(dataset: str) -> str:
    return f"{dataset.replace('-', '_')}_sq-pilot-lipo-raw-v1.npz"


def verify_frozen_files(dataset: str) -> dict[str, Any]:
    """Compare the *real file bytes* against the frozen SHA-256 table."""
    mods = modules()
    graph_path = Path(mods["sq_data"].cache_path(dataset))
    raw_path = Path(mods["sq_data"].lipo_raw_path(dataset))
    out: dict[str, Any] = {"dataset": dataset, "files": {}}
    ok = True
    for path in (graph_path, raw_path):
        expected = C.EXPECTED_CACHE_SHA256[path.name]
        actual = C.sha256_file(path) if path.is_file() else None
        entry = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "bytes": int(path.stat().st_size) if path.is_file() else None,
            "match": bool(actual == expected),
        }
        ok = ok and entry["match"]
        out["files"][path.name] = entry
    out["all_match"] = ok
    if not ok:
        raise RuntimeError(
            f"frozen cache fingerprint mismatch for {dataset}: "
            + ", ".join(
                f"{name}={entry['actual_sha256']}"
                for name, entry in out["files"].items()
                if not entry["match"]
            )
        )
    return out


def load_payload(dataset: str) -> dict[str, Any]:
    """Verified official OGB graph cache (contains targets; see test_policy)."""
    mods = modules()
    payload = mods["sq_data"].load_cache(dataset)
    payload["labels"] = np.asarray(payload["labels"], dtype=np.float64)
    payload["row_ids"] = np.asarray(payload["row_ids"], dtype=np.int64)
    return payload


# ---------------------------------------------------------------------------
# active global mask
# ---------------------------------------------------------------------------


def _global_field_table(spec) -> list[dict[str, Any]]:
    rows = []
    offset = 0
    for f in spec.global_feature_fields:
        rows.append({"index": len(rows), "name": str(f.name), "dim": int(f.dim), "offset": int(offset)})
        offset += int(f.dim)
    if offset != spec.global_dim:
        raise RuntimeError(f"global field table covers {offset} != spec.global_dim {spec.global_dim}")
    return rows


def field_index(spec, name: str) -> int:
    """Resolve a global field name to its column index (asserts a 1-d field)."""
    matches = [row for row in _global_field_table(spec) if row["name"] == name]
    if len(matches) != 1:
        raise KeyError(f"global field {name!r} not found exactly once in the spec")
    row = matches[0]
    if row["dim"] != 1 or row["offset"] != row["index"]:
        raise ValueError(f"global field {name!r} is not a 1-d column at its own offset")
    return int(row["index"])


def build_active_global(spec, norm: Mapping[str, Any], train_global: np.ndarray) -> dict[str, Any]:
    """Train-only mask of non-constant raw global columns plus the frozen contexts."""
    raw = np.asarray(train_global, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != spec.global_dim:
        raise ValueError("train global matrix does not match the spec width")
    lo = raw.min(axis=0)
    hi = raw.max(axis=0)
    span = hi - lo
    active = np.flatnonzero(span > ACTIVE_SPAN_EPS).astype(np.int64)
    inactive = np.flatnonzero(span <= ACTIVE_SPAN_EPS).astype(np.int64)
    table = _global_field_table(spec)
    norm_const = set(int(i) for i in norm.get("global_constant_indices", []))
    inactive_set = set(int(i) for i in inactive.tolist())
    return {
        "span_epsilon": ACTIVE_SPAN_EPS,
        "n_columns": int(spec.global_dim),
        "n_active": int(active.size),
        "active_indices": active.tolist(),
        "active_names": [table[i]["name"] for i in active.tolist()],
        "inactive_indices": inactive.tolist(),
        "inactive_names": [table[i]["name"] for i in inactive.tolist()],
        "train_min": lo.tolist(),
        "train_max": hi.tolist(),
        "train_span": span.tolist(),
        "field_table": table,
        "norm_constant_indices": sorted(norm_const),
        "mask_matches_norm_constants": bool(inactive_set == norm_const),
        "norm_mismatch_explanation": (
            None
            if inactive_set == norm_const
            else "span mask and fit_norm constant set differ: see cr_data.build_active_global"
        ),
    }


def build_contexts(spec, norm: Mapping[str, Any], train_global: np.ndarray) -> dict[str, Any]:
    """Resolve the two frozen context columns by name and pre-compute their Train Q25/Q75."""
    raw = np.asarray(train_global, dtype=np.float64)
    gmean = np.asarray(norm["global_mean"], dtype=np.float64)
    gscale = np.asarray(norm["global_scale"], dtype=np.float64)
    lo_q, hi_q = CONTEXT_QUANTILES
    out: dict[str, Any] = {"fields": list(CONTEXT_FIELDS), "quantiles": [lo_q, hi_q], "per_field": {}}
    estimable = True
    reasons: list[str] = []
    for name in CONTEXT_FIELDS:
        idx = field_index(spec, name)
        col = raw[:, idx]
        span = float(col.max() - col.min())
        q_lo, q_hi = np.percentile(col, [lo_q, hi_q])
        iqr = float(q_hi - q_lo)
        active = bool(span > ACTIVE_SPAN_EPS)
        spread = bool(iqr > ACTIVE_SPAN_EPS)
        ok = active and spread
        estimable = estimable and ok
        if not active:
            reasons.append(f"{name}: Train span {span:.3e} <= {ACTIVE_SPAN_EPS:.0e}")
        if not spread:
            reasons.append(f"{name}: Train Q75-Q25 {iqr:.3e} <= {ACTIVE_SPAN_EPS:.0e}")
        out["per_field"][name] = {
            "index": int(idx),
            "train_span": span,
            "train_q25_raw": float(q_lo),
            "train_q75_raw": float(q_hi),
            "train_iqr_raw": iqr,
            "raw_mean": float(gmean[idx]),
            "raw_scale": float(gscale[idx]),
            "z_q25": float((q_lo - gmean[idx]) / gscale[idx]),
            "z_q75": float((q_hi - gmean[idx]) / gscale[idx]),
            "active": bool(idx in set(np.flatnonzero(
                (raw.max(axis=0) - raw.min(axis=0)) > ACTIVE_SPAN_EPS
            ).tolist())),
            "estimable": bool(ok),
        }
    out["estimable"] = bool(estimable)
    out["status"] = "CONTEXT_ESTIMABLE" if estimable else "CONTEXT_NOT_ESTIMABLE"
    out["reasons"] = reasons
    return out


# ---------------------------------------------------------------------------
# bundle construction
# ---------------------------------------------------------------------------


def build_bundle(dataset: str, *, verify: bool = True) -> Bundle:
    """Build the split caches, Train-only norm/scaler and the frozen masks."""
    mods = modules()
    if verify:
        verify_frozen_files(dataset)
    payload = load_payload(dataset)
    lipo_raw = mods["sq_data"].load_lipo_raw(dataset)
    if [str(s) for s in lipo_raw["smiles"]] != [str(s) for s in payload["smiles"]]:
        raise RuntimeError(f"{dataset}: lipo raw smiles do not match the graph cache smiles")
    built = mods["sq_lcrc"].build_caches(payload, dataset=dataset)
    spec, caches, norm = built["spec"], built["caches"], built["norm"]

    row_ids = {split: np.asarray(payload[C.SPLIT_FILE[split]], dtype=np.int64) for split in C.SPLITS}
    for split in C.SPLITS:
        n = int(caches[split]["n_molecules"])
        if row_ids[split].size != n:
            raise RuntimeError(f"{dataset}/{split}: cache {n} rows != official index {row_ids[split].size}")
        if row_ids[split].min() < 0 or row_ids[split].max() >= payload["labels"].size:
            raise RuntimeError(f"{dataset}/{split}: global row id outside the official table")
    scaler = C.target_scaler(payload["labels"][row_ids["train"]])
    targets = {split: np.asarray(payload["labels"][row_ids[split]], dtype=np.float64) for split in C.SPLITS}
    y_std = {split: (targets[split] - scaler["mean"]) / scaler["std"] for split in C.SPLITS}
    n_atoms = {
        split: (np.asarray(caches[split]["atom_ptr"])[1:] - np.asarray(caches[split]["atom_ptr"])[:-1]).astype(np.int64)
        for split in C.SPLITS
    }
    smiles = {split: [str(s) for s in caches[split]["smiles"]] for split in C.SPLITS}
    active = build_active_global(spec, norm, caches["train"]["global"])
    contexts = build_contexts(spec, norm, caches["train"]["global"])
    return Bundle(
        dataset=dataset,
        payload=payload,
        lipo_raw=lipo_raw,
        spec=spec,
        norm=norm,
        caches=caches,
        global_row_ids=row_ids,
        targets_raw=targets,
        scaler=scaler,
        y_std=y_std,
        n_atoms=n_atoms,
        smiles=smiles,
        active_global=active,
        contexts=contexts,
        collate=mods["lcrc_data"].collate,
        build_seconds=float(built["build_seconds"]),
        extra={
            "n_graphs_built": int(built["n_graphs"]),
            "norm_fingerprint": mods["lcrc_data"].norm_fingerprint(norm),
            "spec_hash": mods["sq_lcrc"].spec_hash(),
            "spec_name": str(spec.name),
        },
    )


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def _occurrence_audit(cache: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Every occurrence counted and in range; nothing truncated to a prefix."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    n_mol = int(cache["n_molecules"])
    n_atoms = int(atom_ptr[-1])
    if nbr_ptr.shape != (n_atoms + 1,) or path_ptr.shape != (n_atoms + 1,):
        raise RuntimeError(f"{split}: per-atom pointer shape {nbr_ptr.shape}/{path_ptr.shape} != {n_atoms + 1}")
    nbr_counts = np.diff(nbr_ptr)
    path_counts = np.diff(path_ptr)
    if int(nbr_counts.sum()) != int(nbr_ptr[-1]):
        raise RuntimeError(f"{split}: S1 occurrence counts do not sum to the pointer end")
    if int(path_counts.sum()) != int(path_ptr[-1]):
        raise RuntimeError(f"{split}: S2 occurrence counts do not sum to the pointer end")
    if int(cache["nbr_code"].size) != int(nbr_ptr[-1]) or int(cache["path_cat"].size) != int(path_ptr[-1]):
        raise RuntimeError(f"{split}: occurrence arrays are not the full pointer range")
    per_molecule_nbr = np.add.reduceat(nbr_counts, atom_ptr[:-1]) if n_atoms else np.zeros(0)
    per_molecule_path = np.add.reduceat(path_counts, atom_ptr[:-1]) if n_atoms else np.zeros(0)
    if int(per_molecule_nbr.sum()) != int(nbr_ptr[-1]) or int(per_molecule_path.sum()) != int(path_ptr[-1]):
        raise RuntimeError(f"{split}: per-molecule occurrence sums lose occurrences")
    n_bonds = int(cache["bond_feat"].shape[0])
    if int(nbr_ptr[-1]) != 2 * n_bonds:
        raise RuntimeError(f"{split}: S1 occurrences {int(nbr_ptr[-1])} != 2 * bonds {2 * n_bonds}")
    if int(bond_ptr[-1]) != n_bonds:
        raise RuntimeError(f"{split}: bond pointer end {int(bond_ptr[-1])} != bonds {n_bonds}")
    return {
        "split": split,
        "n_molecules": n_mol,
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "n_s1_occurrences": int(nbr_ptr[-1]),
        "n_s2_occurrences": int(path_ptr[-1]),
        "mean_s2_per_atom": float(path_counts.mean()) if n_atoms else 0.0,
        "molecules_without_s2": int((per_molecule_path == 0).sum()),
        "s2_sum_matches_pointer": True,
        "s1_equals_two_bonds": True,
        "occurrence_arrays_full_range": True,
    }


def audit_dataset(dataset: str, *, verify: bool = True) -> dict[str, Any]:
    """Full data audit for one task (one entry of ``analysis/data_audit.json``)."""
    mods = modules()
    files = verify_frozen_files(dataset) if verify else {"skipped": True}
    bundle = build_bundle(dataset, verify=False)
    payload = bundle.payload
    sizes = {split: int(bundle.global_row_ids[split].size) for split in C.SPLITS}
    expected_sizes = C.EXPECTED_SPLIT_SIZES[dataset]
    if sizes != expected_sizes:
        raise RuntimeError(f"{dataset}: split sizes {sizes} != frozen {expected_sizes}")
    union = np.concatenate([bundle.global_row_ids[split] for split in C.SPLITS])
    n_rows = int(payload["labels"].size)
    if n_rows != C.EXPECTED_TOTAL_ROWS[dataset]:
        raise RuntimeError(f"{dataset}: {n_rows} official rows != frozen {C.EXPECTED_TOTAL_ROWS[dataset]}")
    disjoint = int(union.size) == n_rows and int(np.unique(union).size) == n_rows
    covered = bool(np.array_equal(np.sort(union), np.arange(n_rows)))
    if not disjoint or not covered:
        raise RuntimeError(f"{dataset}: official splits are not a disjoint cover of the table")
    occurrence = {}
    validate = {}
    for split in C.SPLITS:
        cache = bundle.caches[split]
        occurrence[split] = _occurrence_audit(cache, split)
        validate[split] = mods["lcrc_data"].validate_cache(cache, bundle.spec)
    # Train constant columns seen off-Train (fixed-to-zero columns).
    raw_by_split = {split: np.asarray(bundle.caches[split]["global"], dtype=np.float64) for split in C.SPLITS}
    inactive = bundle.active_global["inactive_indices"]
    off_train = {}
    for idx in inactive:
        vals = {split: [float(raw_by_split[split][:, idx].min()), float(raw_by_split[split][:, idx].max())]
                for split in C.SPLITS}
        train_value = float(raw_by_split["train"][0, idx])
        deviating = bool(
            np.any(np.abs(raw_by_split["valid"][:, idx] - train_value) > C.ACTIVE_SPAN_EPS)
            or np.any(np.abs(raw_by_split["test"][:, idx] - train_value) > C.ACTIVE_SPAN_EPS)
        )
        off_train[str(idx)] = {
            "name": bundle.active_global["field_table"][idx]["name"],
            "train_value": train_value,
            "per_split_minmax": vals,
            "val_or_test_deviates_from_train_constant": deviating,
        }
    roundtrip = (
        (bundle.targets_raw["train"] - bundle.scaler["mean"]) / bundle.scaler["std"] * bundle.scaler["std"]
        + bundle.scaler["mean"]
    )
    return {
        "dataset": dataset,
        "files": files,
        "n_rows_official": n_rows,
        "split_sizes": sizes,
        "split_disjoint": bool(disjoint),
        "split_covers_all_rows": bool(covered),
        "row_ids_are_global": True,
        "row_id_sha256": {split: C.sha256_array(bundle.global_row_ids[split]) for split in C.SPLITS},
        "row_id_minmax": {
            split: [int(bundle.global_row_ids[split].min()), int(bundle.global_row_ids[split].max())]
            for split in C.SPLITS
        },
        "row_ids_sorted_ascending": {
            split: bool(np.all(np.diff(bundle.global_row_ids[split]) > 0)) for split in C.SPLITS
        },
        "occurrence": occurrence,
        "cache_validation": validate,
        "feature_spec": {
            "name": bundle.extra["spec_name"],
            "hash": bundle.extra["spec_hash"],
            "global_dim": int(bundle.spec.global_dim),
            "n_global_fields": len(bundle.spec.global_feature_fields),
            "atom_feat_dim": int(bundle.spec.atom_feat_dim),
            "bond_feat_dim": int(bundle.spec.bond_feat_dim),
            "n_scalars": int(bundle.spec.n_scalars),
        },
        "norm": {
            "source": str(bundle.norm.get("source")),
            "fit_split": str(bundle.norm.get("fit_split")),
            "valid_or_test_used": bool(bundle.norm.get("valid_or_test_used")),
            "fingerprint": bundle.extra["norm_fingerprint"],
            "constant_indices": [int(i) for i in bundle.norm.get("global_constant_indices", [])],
            "zero_variance_policy": str(bundle.norm.get("zero_variance_policy")),
        },
        "target_scaler": {**bundle.scaler, "target_roundtrip_max_abs_error": float(np.abs(roundtrip - bundle.targets_raw["train"]).max())},
        "active_global": {
            k: v
            for k, v in bundle.active_global.items()
            if k not in ("train_min", "train_max", "train_span", "field_table")
        },
        "inactive_columns_off_train": off_train,
        "contexts": bundle.contexts,
        "cache_build_seconds": bundle.build_seconds,
        # every structural invariant above raises instead of reporting a number,
        # so reaching this point with the file gate closed is a pass
        "frozen_files_match": bool(files.get("all_match", False)),
        "cache_validation_pass": bool(
            all(
                v.get("finite", False) and v.get("index_ranges_valid", False)
                for v in validate.values()
            )
        ),
        "pass": bool(
            files.get("all_match", False)
            and disjoint
            and covered
            and all(
                v.get("finite", False) and v.get("index_ranges_valid", False)
                for v in validate.values()
            )
        ),
        "test_policy_note": (
            "the frozen caches contain official Test targets; this round runs under "
            "test_policy=always, so Test labels are read every epoch and are never used "
            "for checkpoint or configuration selection"
        ),
    }
