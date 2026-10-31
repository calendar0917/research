"""Chem12 / Chem13 feature construction, alignment and fold-train normalisation.

Design (frozen before GPU)
--------------------------
* Features come *only* from unlabeled SMILES.  There is no label, Monitor,
  Reserved or official-test input anywhere in this module.
* ``Chem12`` is the fixed 12-name list in :data:`v3_common.CHEM12_NAMES`.
  ``Chem13`` is exactly ``Chem12`` followed by ``MolLogP``.
* Row identity is the canonical-SMILES SHA-256 recorded by the frozen v2
  descriptor cache and the frozen row manifest.  The loader re-verifies every
  row and aborts on a mismatch.
* Non-finite values are replaced by the *training-fold* median of that column.
  Mean/std standardisation is fitted on the training fold only.  The dev fold
  never contributes.
* The augmented :class:`FeatureSpec` appends exactly ``n_extra`` one-dimensional
  global fields.  atom51, bond13, S1, S2 and scalar9 are copied byte-for-byte
  from the frozen ``lipo_v1`` spec.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

import v3_common as C

import feature_spec as FS  # provided by the frozen prep sys.path


FEATURE_DESCRIPTIONS = {
    "TPSA": "topological polar surface area (RDKit TPSA)",
    "MolWt": "average molecular weight (RDKit MolWt)",
    "NumHAcceptors": "number of hydrogen-bond acceptors (RDKit)",
    "NumHDonors": "number of hydrogen-bond donors (RDKit)",
    "NumRotatableBonds": "number of rotatable bonds (RDKit)",
    "NumAromaticRings": "number of aromatic rings (RDKit)",
    "NumAliphaticRings": "number of aliphatic rings (RDKit)",
    "NumSaturatedRings": "number of saturated rings (RDKit)",
    "RingCount": "total ring count (RDKit)",
    "FractionCSP3": "fraction of sp3 carbons (RDKit)",
    "HeavyAtomCount": "heavy atom count (RDKit)",
    "MolMR": "molar refractivity (RDKit MolMR)",
    "MolLogP": "Crippen octanol-water logP (RDKit MolLogP)",
}


# ---------------------------------------------------------------------------
# feature store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChemStore:
    rows: list[int]
    physchem13: np.ndarray
    chem12: np.ndarray
    chem13: np.ndarray
    formal_charge: np.ndarray
    canonical_hashes: list[str]
    audit: dict[str, Any]


def _require_v2_cache() -> None:
    """The v2 cache is the frozen descriptor source; never rewrite it."""
    for rel in ("cache/features_v2.npz", "cache/features_v2.json"):
        path = C.V2 / rel
        if not path.exists():
            raise FileNotFoundError(f"missing frozen v2 descriptor cache {C.track_rel(path)}; v3 must not rebuild it in place")
        expected = C.V2_HISTORICAL_FILES[rel]
        C.verify_sha256(path, expected, rel)


def _select_columns(physchem13: np.ndarray, names: Sequence[str]) -> np.ndarray:
    source = list(C.DESCRIPTOR_13)
    if list(names)[:12] != list(C.CHEM12_NAMES):
        raise ValueError("Chem12 name order is not the frozen order")
    missing = [n for n in names if n not in source]
    if missing:
        raise ValueError(f"RDKit descriptor source lacks {missing}")
    idx = [source.index(n) for n in names]
    return np.asarray(physchem13[:, idx], dtype=np.float64)


def _formal_charges(rows: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    from rdkit import Chem

    raw, manifest = C.B._raw_and_manifest()
    values = []
    hashes = []
    for row in rows:
        mol, _canonical, digest = C.B._mol_and_canonical(raw, manifest, int(row))
        values.append(int(Chem.GetFormalCharge(mol)))
        hashes.append(digest)
    return np.asarray(values, dtype=np.int64), hashes


def _independent_descriptor_check(store_rows: Sequence[int], physchem13: np.ndarray, sample: Sequence[int]) -> dict[str, Any]:
    """Recompute CalcMolDescriptors for a deterministic sample and compare."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    raw, manifest = C.B._raw_and_manifest()
    rows = [int(r) for r in store_rows]
    position = {row: i for i, row in enumerate(rows)}
    checks = []
    for row in sample:
        pos = position[int(row)]
        record = raw["rows"][int(row)]
        if "exp" in record:
            raise PermissionError(f"raw row {row} unexpectedly contains a label")
        mol = Chem.MolFromSmiles(str(record["smiles"]))
        canonical = Chem.MolToSmiles(mol, canonical=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if digest != manifest["canonical_smiles_sha256"][int(row)]:
            raise ValueError(f"row {row}: canonical SMILES hash mismatch")
        desc = Descriptors.CalcMolDescriptors(mol)
        recomputed = np.asarray([float(desc[n]) for n in C.DESCRIPTOR_13], dtype=np.float64)
        cached = np.asarray(physchem13[pos], dtype=np.float64)
        same = bool(np.allclose(recomputed, cached, rtol=0.0, atol=1e-12, equal_nan=True))
        checks.append(
            {
                "row": int(row),
                "canonical_smiles_sha256": digest,
                "max_abs_diff_all_13": float(np.nanmax(np.abs(recomputed - cached))),
                "all_13_equal": same,
            }
        )
    return {
        "n_sampled": len(checks),
        "all_equal": all(x["all_13_equal"] for x in checks),
        "samples": checks,
        "selection": "deterministic: min/max of every Chem12 column, MolLogP min/max, first/last Fit row",
    }


@lru_cache(maxsize=1)
def build_store() -> ChemStore:
    _require_v2_cache()
    import rdkit

    store = C.B.load_or_compute_fit_features()
    split = C.load_frozen_split()
    fit_rows = [int(r) for r in split["split"]["fit"]]
    if store.rows != fit_rows:
        raise ValueError("v2 descriptor cache row order does not match the frozen Fit2352 order")
    if store.audit.get("labels_used") is not False:
        raise PermissionError("v2 descriptor audit is not certified label-free")
    if str(store.audit.get("rdkit_version")) != str(rdkit.__version__):
        raise PermissionError(
            f"RDKit version changed: cache {store.audit.get('rdkit_version')!r} != runtime {rdkit.__version__!r}"
        )
    phys = np.asarray(store.physchem13, dtype=np.float64)
    if phys.shape != (C.SPLIT_SIZES["fit"], 13):
        raise ValueError(f"physchem13 shape {phys.shape} != {(C.SPLIT_SIZES['fit'], 13)}")
    digest = hashlib.sha256(np.ascontiguousarray(phys).tobytes()).hexdigest()
    if digest != store.audit.get("physchem13_sha256"):
        raise PermissionError("physchem13 bytes do not match the v2 audit hash")
    chem12 = _select_columns(phys, C.CHEM12_NAMES)
    chem13 = _select_columns(phys, C.CHEM13_NAMES)
    if not np.array_equal(chem13[:, :12], chem12):
        raise ValueError("Chem13 must be exactly Chem12 followed by MolLogP")
    if not np.array_equal(chem13[:, 12], phys[:, list(C.DESCRIPTOR_13).index("MolLogP")]):
        raise ValueError("Chem13 last column is not MolLogP")
    formal, hashes = _formal_charges(fit_rows)
    if hashes != list(store.canonical_hashes):
        raise ValueError("formal-charge canonical SMILES hashes disagree with the descriptor cache")

    # Independent spot-check of the descriptor values (all 13 columns).
    rng = np.random.default_rng(20261101)
    sample = set()
    for j in range(12):
        sample.add(int(fit_rows[int(np.argmin(chem12[:, j]))]))
        sample.add(int(fit_rows[int(np.argmax(chem12[:, j]))]))
    sample.add(int(fit_rows[int(np.argmin(chem13[:, 12]))]))
    sample.add(int(fit_rows[int(np.argmax(chem13[:, 12]))]))
    sample.update(int(fit_rows[int(x)]) for x in rng.choice(C.SPLIT_SIZES["fit"], size=10, replace=False))
    independent = _independent_descriptor_check(fit_rows, phys, sorted(sample))
    if not independent["all_equal"]:
        raise RuntimeError(f"independent RDKit recomputation disagrees with the frozen cache: {independent}")

    audit = {
        "protocol_id": C.PROTOCOL_ID,
        "rows": len(fit_rows),
        "rows_sha256_int64": C.row_sha256_bytes(fit_rows),
        "rows_sha256_text": C.row_sha256_text(fit_rows),
        "rdkit_version": str(rdkit.__version__),
        "descriptor_source": "rdkit.Chem.Descriptors.CalcMolDescriptors",
        "canonical_smiles": "RDKit MolToSmiles(canonical=True); sha256",
        "canonical_smiles_sha256_list": hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest(),
        "rows_aligned_to_frozen_manifest": True,
        "chem12": {
            "names": list(C.CHEM12_NAMES),
            "order_fixed": True,
            "shape": list(chem12.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(chem12).tobytes()).hexdigest(),
        },
        "chem13": {
            "names": list(C.CHEM13_NAMES),
            "order_fixed": True,
            "definition": "Chem12 + MolLogP",
            "shape": list(chem13.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(chem13).tobytes()).hexdigest(),
        },
        "physchem13_sha256": digest,
        "nonfinite_counts": {
            "chem12_total": int((~np.isfinite(chem12)).sum()),
            "chem12_per_column": [int((~np.isfinite(chem12[:, j])).sum()) for j in range(12)],
            "chem13_total": int((~np.isfinite(chem13)).sum()),
        },
        "formal_charge": {
            "definition": "RDKit Chem.GetFormalCharge on the canonical molecule",
            "min": int(formal.min()),
            "max": int(formal.max()),
            "counts": {str(int(v)): int((formal == v).sum()) for v in np.unique(formal)},
        },
        "independent_descriptor_check": independent,
        "labels_used": False,
        "monitor_structures_used": False,
        "reserved_structures_used": False,
        "official_valid_test_used": False,
    }
    return ChemStore(fit_rows, phys, chem12, chem13, formal, hashes, audit)


def load_store() -> ChemStore:
    cache_npz = C.EXP_DIR / "cache" / "chem_features_v3.npz"
    cache_json = C.EXP_DIR / "cache" / "chem_features_v3.json"
    store = build_store()
    C.ensure_dir(cache_npz.parent)
    np.savez_compressed(
        cache_npz,
        rows=np.asarray(store.rows, dtype=np.int64),
        chem12=store.chem12,
        chem13=store.chem13,
        formal_charge=store.formal_charge,
        physchem13=store.physchem13,
    )
    C.write_json(cache_json, store.audit)
    return store


# ---------------------------------------------------------------------------
# fold-train normalisation
# ---------------------------------------------------------------------------


def fit_extra_norm(raw: np.ndarray, train_positions: Sequence[int]) -> dict[str, Any]:
    """Median-impute and standardise using training-fold rows only."""
    X = np.asarray(raw, dtype=np.float64)
    train_pos = np.asarray([int(p) for p in train_positions], dtype=np.int64)
    if train_pos.size == 0 or train_pos.max() >= X.shape[0]:
        raise ValueError("invalid training positions")
    train = X[train_pos]
    n, d = X.shape
    medians = np.zeros(d, dtype=np.float64)
    for j in range(d):
        finite = train[:, j][np.isfinite(train[:, j])]
        medians[j] = float(np.median(finite)) if finite.size else 0.0
    imputed = X.copy()
    bad = ~np.isfinite(imputed)
    if bad.any():
        imputed[bad] = np.take(medians, np.where(bad)[1])
    train_imputed = imputed[train_pos]
    mean = train_imputed.mean(axis=0)
    std = train_imputed.std(axis=0)
    constant = np.flatnonzero(std <= 1e-12)
    scale = np.where(std > 1e-12, std, 1.0)
    audit = {
        "fit_scope": "fold1_train_only",
        "fit_positions_sha256": hashlib.sha256(np.ascontiguousarray(train_pos, dtype=np.int64).tobytes()).hexdigest(),
        "n_train_rows": int(train_pos.size),
        "n_features": int(d),
        "nonfinite_train_total": int((~np.isfinite(train)).sum()),
        "nonfinite_all_total": int((~np.isfinite(X)).sum()),
        "median": [float(x) for x in medians],
        "mean": [float(x) for x in mean],
        "scale": [float(x) for x in scale],
        "zero_variance_indices": [int(x) for x in constant],
        "zero_variance_policy": "impute train median; use scale 1.0 for constant training columns",
        "dev_used_for_fit": False,
        "labels_used": False,
    }
    return {
        "median": medians,
        "mean": mean,
        "scale": scale,
        "constant_indices": constant,
        "raw_imputed": imputed,
        "audit": audit,
    }


def augment_norm(base_norm: Mapping[str, Any], extra_norm: Mapping[str, Any]) -> dict[str, Any]:
    base_m = np.asarray(base_norm["global_mean"], dtype=np.float32)
    base_s = np.asarray(base_norm["global_scale"], dtype=np.float32)
    if base_m.shape != (C.GLOBAL_DIM_BASE,) or base_s.shape != (C.GLOBAL_DIM_BASE,):
        raise ValueError("base global normalisation is not 42-dimensional")
    extra_m = np.asarray(extra_norm["mean"], dtype=np.float32)
    extra_s = np.asarray(extra_norm["scale"], dtype=np.float32)
    norm = {
        "source": f"{base_norm.get('source', 'fold1_train')} + chem train-fold extras",
        "fit_split": str(base_norm.get("fit_split", "fold1_train")),
        "valid_or_test_used": False,
        "zero_variance_policy": "constant_fit_features_use_scale_1.0",
        "scalar_constant_indices": list(base_norm.get("scalar_constant_indices", [])),
        "global_constant_indices": sorted(
            list(base_norm.get("global_constant_indices", []))
            + [C.GLOBAL_DIM_BASE + int(i) for i in extra_norm["constant_indices"]]
        ),
        "scalar_mean": np.asarray(base_norm["scalar_mean"], dtype=np.float32),
        "scalar_scale": np.asarray(base_norm["scalar_scale"], dtype=np.float32),
        "global_mean": np.concatenate([base_m, extra_m]),
        "global_scale": np.concatenate([base_s, extra_s]),
    }
    if not np.array_equal(norm["global_mean"][: C.GLOBAL_DIM_BASE], base_m):
        raise ValueError("augmented norm changed the original 42 global means")
    if not np.array_equal(norm["global_scale"][: C.GLOBAL_DIM_BASE], base_s):
        raise ValueError("augmented norm changed the original 42 global scales")
    return norm


def slice_base_norm(norm: Mapping[str, Any], base_dim: int = C.GLOBAL_DIM_BASE) -> dict[str, Any]:
    """Recover the original-width normalisation for the same-seed reference model."""
    if np.asarray(norm["global_mean"]).shape != (base_dim + len(norm["global_mean"]) - base_dim,):
        pass
    out = {
        "global_mean": np.asarray(norm["global_mean"], dtype=np.float32)[:base_dim],
        "global_scale": np.asarray(norm["global_scale"], dtype=np.float32)[:base_dim],
        "scalar_mean": np.asarray(norm["scalar_mean"], dtype=np.float32),
        "scalar_scale": np.asarray(norm["scalar_scale"], dtype=np.float32),
        "global_constant_indices": [int(i) for i in norm.get("global_constant_indices", []) if int(i) < base_dim],
        "scalar_constant_indices": list(norm.get("scalar_constant_indices", [])),
        "source": f"{norm.get('source', '')} [base {base_dim} slice]",
        "fit_split": str(norm.get("fit_split", "fold1_train")),
        "valid_or_test_used": False,
        "zero_variance_policy": "constant_fit_features_use_scale_1.0",
    }
    if out["global_mean"].shape != (base_dim,) or out["global_scale"].shape != (base_dim,):
        raise ValueError("could not slice the base global normalisation")
    return out


# ---------------------------------------------------------------------------
# augmented spec
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def augmented_spec(variant: str) -> Any:
    if variant not in ("chem12", "chem13"):
        raise ValueError(f"unknown chemistry variant {variant!r}")
    names = C.CHEM12_NAMES if variant == "chem12" else C.CHEM13_NAMES
    base = C.load_frozen_spec()
    if base.name != "lipo_v1" or base.global_dim != C.GLOBAL_DIM_BASE:
        raise ValueError("frozen base spec is not lipo_v1 with 42 global fields")
    d = base.to_dict()
    d.pop("dimensions", None)
    d.pop("spec_hash", None)
    d["name"] = f"lipo_v1_{variant}"
    d["description"] = (
        base.description
        + f" | v3 pilot: append {len(names)} label-free RDKit global physchem descriptors "
        + f"({variant}); atom/bond/S1/S2/scalar semantics unchanged."
    )
    d["global_feature_fields"] = list(d["global_feature_fields"]) + [
        {
            "name": name,
            "dim": 1,
            "dtype": "float32",
            "description": FEATURE_DESCRIPTIONS[name],
            "source": "RDKit CalcMolDescriptors from label-free canonical SMILES; v3 pilot append",
        }
        for name in names
    ]
    d["global_feature_source"] = (
        str(d.get("global_feature_source", ""))
        + f" | appended {variant}: RDKit Descriptors.CalcMolDescriptors (no labels)"
    )
    d["source_provenance"] = {
        **dict(d.get("source_provenance", {})),
        "v3_augmentation": {
            "variant": variant,
            "names": list(names),
            "base_spec_hash": base.hash(),
            "rule": "append one-dimensional global fields only",
            "atom_bond_scalar_fields_copied_unchanged": True,
        },
    }
    d["output_tensors"] = [
        t.to_dict()
        for t in FS._output_tensors(
            sum(f.dim for f in base.atom_feature_fields),
            sum(f.dim for f in base.bond_feature_fields),
            sum(f.dim for f in base.atom_scalar_fields),
            base.global_dim + len(names),
        )
    ]
    spec = FS.FeatureSpec.from_dict(d)
    if spec.global_dim != base.global_dim + len(names):
        raise ValueError("augmented spec width is wrong")
    for attr in ("atom_type_names", "bond_type_names", "atom_feature_fields", "bond_feature_fields", "atom_scalar_fields"):
        if getattr(spec, attr) != getattr(base, attr):
            raise ValueError(f"augmented spec changed frozen {attr}")
    return spec


# ---------------------------------------------------------------------------
# fold views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldExtras:
    fold: int
    variant: str
    spec: Any
    norm: Mapping[str, Any]
    base_norm: Mapping[str, Any]
    train_raw: np.ndarray
    dev_raw: np.ndarray
    train_positions: tuple[int, ...]
    dev_positions: tuple[int, ...]
    train_rows: tuple[int, ...]
    dev_rows: tuple[int, ...]
    audit: dict[str, Any]


@lru_cache(maxsize=4)
def build_fold_extras(fold: int = C.FOLD, variant: str = "chem12") -> FoldExtras:
    if fold != C.FOLD:
        raise ValueError(f"v3 only uses Fold {C.FOLD}")
    store = build_store()
    folds = C.F.load_folds()
    train_pos = tuple(int(p) for p in C.F.train_positions(folds, fold))
    dev_pos = tuple(int(p) for p in C.F.fold_positions(folds, fold))
    if set(train_pos) & set(dev_pos):
        raise ValueError("fold train/dev overlap")
    if sorted(train_pos + dev_pos) != list(range(C.SPLIT_SIZES["fit"])):
        raise ValueError("fold train+dev does not cover Fit2352")

    cache_info = C.load_fit_cache()
    view = C.D.build_fold_view(
        fold,
        cache=cache_info["cache"],
        y_fit=np.asarray(C.load_fit_labels()["y"], dtype=np.float64),
        folds=folds,
    )
    if tuple(view.train_positions) != train_pos or tuple(view.dev_positions) != dev_pos:
        raise ValueError("v2 fold view positions disagree with the frozen fold file")

    raw_all = store.chem12 if variant == "chem12" else store.chem13
    norm = fit_extra_norm(raw_all, train_pos)
    imputed = norm["raw_imputed"]
    train_raw = np.asarray(imputed[np.asarray(train_pos, dtype=np.int64)], dtype=np.float64)
    dev_raw = np.asarray(imputed[np.asarray(dev_pos, dtype=np.int64)], dtype=np.float64)
    aug_norm = augment_norm(view.train_norm, norm)

    # Row-identity alignment: local train/dev rows must be the same molecules
    # as the frozen fold file and the descriptor cache.
    train_rows = tuple(int(store.rows[p]) for p in train_pos)
    dev_rows = tuple(int(store.rows[p]) for p in dev_pos)
    expected_train_rows = tuple(
        int(folds["rows"][str(store.rows[p])]["global_row_id"]) for p in train_pos
    )
    expected_dev_rows = tuple(int(folds["rows"][str(store.rows[p])]["global_row_id"]) for p in dev_pos)
    if train_rows != expected_train_rows or dev_rows != expected_dev_rows:
        raise ValueError("fold row ids do not align with the frozen fold file")
    for pos in train_pos + dev_pos:
        info = folds["rows"][str(store.rows[pos])]
        if info["canonical_smiles_sha256"] != store.canonical_hashes[pos]:
            raise ValueError(f"canonical SMILES hash mismatch at cache position {pos}")

    audit = {
        "protocol_id": C.PROTOCOL_ID,
        "fold": int(fold),
        "variant": variant,
        "n_train": int(len(train_pos)),
        "n_dev": int(len(dev_pos)),
        "train_positions_sha256": C.row_sha256_bytes(train_pos),
        "dev_positions_sha256": C.row_sha256_bytes(dev_pos),
        "train_rows_sha256": C.row_sha256_bytes(train_rows),
        "dev_rows_sha256": C.row_sha256_bytes(dev_rows),
        "train_rows_first_last": [int(train_rows[0]), int(train_rows[-1])],
        "dev_rows_first_last": [int(dev_rows[0]), int(dev_rows[-1])],
        "train_dev_disjoint": True,
        "train_dev_cover_fit2352": True,
        "row_alignment": "global Fit row -> cache position -> canonical SMILES sha256 -> frozen fold file",
        "canonical_alignment_verified_rows": int(len(train_pos) + len(dev_pos)),
        "extra_norm": norm["audit"],
        "augmented_norm": {
            "global_mean_sha256": hashlib.sha256(np.ascontiguousarray(aug_norm["global_mean"], dtype=np.float32).tobytes()).hexdigest(),
            "global_scale_sha256": hashlib.sha256(np.ascontiguousarray(aug_norm["global_scale"], dtype=np.float32).tobytes()).hexdigest(),
            "base_42_matches_fold_train_norm": bool(
                np.array_equal(
                    np.asarray(aug_norm["global_mean"][: C.GLOBAL_DIM_BASE], dtype=np.float32),
                    np.asarray(view.train_norm["global_mean"], dtype=np.float32),
                )
                and np.array_equal(
                    np.asarray(aug_norm["global_scale"][: C.GLOBAL_DIM_BASE], dtype=np.float32),
                    np.asarray(view.train_norm["global_scale"], dtype=np.float32),
                )
            ),
        },
        "base_norm_fingerprint": C.prep_data.norm_fingerprint(view.train_norm),
        "spec_hash": augmented_spec(variant).hash(),
        "base_spec_hash": C.load_frozen_spec().hash(),
        "train_raw_sha256": hashlib.sha256(np.ascontiguousarray(train_raw, dtype=np.float64).tobytes()).hexdigest(),
        "dev_raw_sha256": hashlib.sha256(np.ascontiguousarray(dev_raw, dtype=np.float64).tobytes()).hexdigest(),
        "train_raw_standardised_sha256": hashlib.sha256(
            np.ascontiguousarray((train_raw - norm["mean"]) / norm["scale"], dtype=np.float64).tobytes()
        ).hexdigest(),
        "dev_raw_standardised_sha256": hashlib.sha256(
            np.ascontiguousarray((dev_raw - norm["mean"]) / norm["scale"], dtype=np.float64).tobytes()
        ).hexdigest(),
        "labels_used_for_features": False,
        "dev_labels_used_for_norm": False,
        "monitor_structures_used": False,
        "reserved_structures_used": False,
        "official_valid_test_used": False,
    }
    return FoldExtras(
        fold=int(fold),
        variant=variant,
        spec=augmented_spec(variant),
        norm=aug_norm,
        base_norm=view.train_norm,
        train_raw=train_raw,
        dev_raw=dev_raw,
        train_positions=train_pos,
        dev_positions=dev_pos,
        train_rows=train_rows,
        dev_rows=dev_rows,
        audit=audit,
    )


FROZEN_EXTRAS_NPZ = C.ANALYSIS / "frozen_extras_v3.npz"
FROZEN_EXTRAS_JSON = C.ANALYSIS / "frozen_extras_v3.json"


def export_frozen_extras() -> dict[str, Any]:
    """Write the exact Fold-1 matrices/spec/norm consumed by the GPU runs.

    The GPU hosts do not need RDKit: they read this committed, hash-pinned
    artifact and reconstruct only the widened FeatureSpec (which is pure
    metadata) and the 54-wide normalisation.
    """
    arrays: dict[str, np.ndarray] = {}
    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "fold": int(C.FOLD),
        "seed": int(C.SEED),
        "variants": {},
        "labels_used": False,
        "monitor_used": False,
        "reserved_used": False,
        "official_valid_test_used": False,
    }
    for variant in ("chem12", "chem13"):
        fx = build_fold_extras(C.FOLD, variant)
        arrays[f"train_{variant}"] = np.asarray(fx.train_raw, dtype=np.float64)
        arrays[f"dev_{variant}"] = np.asarray(fx.dev_raw, dtype=np.float64)
        arrays[f"global_mean_{variant}"] = np.asarray(fx.norm["global_mean"], dtype=np.float32)
        arrays[f"global_scale_{variant}"] = np.asarray(fx.norm["global_scale"], dtype=np.float32)
        arrays[f"scalar_mean_{variant}"] = np.asarray(fx.norm["scalar_mean"], dtype=np.float32)
        arrays[f"scalar_scale_{variant}"] = np.asarray(fx.norm["scalar_scale"], dtype=np.float32)
        arrays[f"train_positions_{variant}"] = np.asarray(fx.train_positions, dtype=np.int64)
        arrays[f"dev_positions_{variant}"] = np.asarray(fx.dev_positions, dtype=np.int64)
        arrays[f"train_rows_{variant}"] = np.asarray(fx.train_rows, dtype=np.int64)
        arrays[f"dev_rows_{variant}"] = np.asarray(fx.dev_rows, dtype=np.int64)
        arrays[f"global_constant_indices_{variant}"] = np.asarray(fx.norm["global_constant_indices"], dtype=np.int64)
        arrays[f"scalar_constant_indices_{variant}"] = np.asarray(fx.norm["scalar_constant_indices"], dtype=np.int64)
        payload["variants"][variant] = {
            "spec_hash": fx.spec.hash(),
            "base_spec_hash": C.load_frozen_spec().hash(),
            "names": list(C.CHEM12_NAMES if variant == "chem12" else C.CHEM13_NAMES),
            "n_extra": int(fx.train_raw.shape[1]),
            "train_shape": list(fx.train_raw.shape),
            "dev_shape": list(fx.dev_raw.shape),
            "train_raw_sha256": fx.audit["train_raw_sha256"],
            "dev_raw_sha256": fx.audit["dev_raw_sha256"],
            "norm_sha256": {
                "global_mean": hashlib.sha256(np.ascontiguousarray(fx.norm["global_mean"], dtype=np.float32).tobytes()).hexdigest(),
                "global_scale": hashlib.sha256(np.ascontiguousarray(fx.norm["global_scale"], dtype=np.float32).tobytes()).hexdigest(),
                "scalar_mean": hashlib.sha256(np.ascontiguousarray(fx.norm["scalar_mean"], dtype=np.float32).tobytes()).hexdigest(),
                "scalar_scale": hashlib.sha256(np.ascontiguousarray(fx.norm["scalar_scale"], dtype=np.float32).tobytes()).hexdigest(),
            },
            "train_positions_sha256": C.row_sha256_bytes(fx.train_positions),
            "dev_positions_sha256": C.row_sha256_bytes(fx.dev_positions),
            "train_rows_sha256": C.row_sha256_bytes(fx.train_rows),
            "dev_rows_sha256": C.row_sha256_bytes(fx.dev_rows),
            "audit": fx.audit,
        }
    C.ensure_dir(C.ANALYSIS)
    np.savez_compressed(FROZEN_EXTRAS_NPZ, **arrays)
    payload["npz_sha256"] = C.sha256_file(FROZEN_EXTRAS_NPZ)
    payload["npz_path"] = C.track_rel(FROZEN_EXTRAS_NPZ)
    C.write_json(FROZEN_EXTRAS_JSON, payload)
    return payload


@lru_cache(maxsize=2)
def load_frozen_extras(variant: str = "chem12"):
    if not FROZEN_EXTRAS_NPZ.exists() or not FROZEN_EXTRAS_JSON.exists():
        raise FileNotFoundError("missing frozen v3 extras; run `run_v3.py freeze-extras` on CPU")
    payload = C.read_json(FROZEN_EXTRAS_JSON)
    if payload.get("protocol_id") != C.PROTOCOL_ID:
        raise ValueError("frozen extras protocol id mismatch")
    if C.sha256_file(FROZEN_EXTRAS_NPZ) != payload.get("npz_sha256"):
        raise PermissionError("frozen extras npz hash mismatch")
    meta = payload["variants"][variant]
    data = np.load(FROZEN_EXTRAS_NPZ)
    spec = augmented_spec(variant)
    if spec.hash() != meta["spec_hash"]:
        raise PermissionError("frozen extras spec hash mismatch")
    norm = {
        "global_mean": data[f"global_mean_{variant}"],
        "global_scale": data[f"global_scale_{variant}"],
        "scalar_mean": data[f"scalar_mean_{variant}"],
        "scalar_scale": data[f"scalar_scale_{variant}"],
        "global_constant_indices": [int(x) for x in data[f"global_constant_indices_{variant}"].tolist()],
        "scalar_constant_indices": [int(x) for x in data[f"scalar_constant_indices_{variant}"].tolist()],
        "source": "frozen_v3_fold1_train_extras",
        "fit_split": "fold1_train",
        "valid_or_test_used": False,
        "zero_variance_policy": "constant_fit_features_use_scale_1.0",
    }
    arrays = {
        "train": np.asarray(data[f"train_{variant}"], dtype=np.float64),
        "dev": np.asarray(data[f"dev_{variant}"], dtype=np.float64),
        "train_positions": tuple(int(x) for x in data[f"train_positions_{variant}"].tolist()),
        "dev_positions": tuple(int(x) for x in data[f"dev_positions_{variant}"].tolist()),
        "train_rows": tuple(int(x) for x in data[f"train_rows_{variant}"].tolist()),
        "dev_rows": tuple(int(x) for x in data[f"dev_rows_{variant}"].tolist()),
    }
    for key, arr in (("train", arrays["train"]), ("dev", arrays["dev"])):
        digest = hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()
        if digest != meta[f"{key}_raw_sha256"]:
            raise PermissionError(f"frozen extras {key} array hash mismatch")
    return {"spec": spec, "norm": norm, "audit": meta, **arrays}


def extras_for_training(fold: int = C.FOLD, variant: str = "chem12") -> dict[str, Any]:
    """The exact objects accepted by ``v2_training.TrainJob``."""
    fx = build_fold_extras(fold, variant)
    return {
        "spec_override": fx.spec,
        "norm_override": fx.norm,
        "global_extra_train": fx.train_raw,
        "global_extra_dev": fx.dev_raw,
        "global_extra_audit": {
            "variant": fx.variant,
            "names": list(C.CHEM12_NAMES if fx.variant == "chem12" else C.CHEM13_NAMES),
            "column_order": list(C.CHEM12_NAMES if fx.variant == "chem12" else C.CHEM13_NAMES),
            "n_extra": int(fx.train_raw.shape[1]),
            "train_shape": list(fx.train_raw.shape),
            "dev_shape": list(fx.dev_raw.shape),
            "train_positions_sha256": C.row_sha256_bytes(fx.train_positions),
            "dev_positions_sha256": C.row_sha256_bytes(fx.dev_positions),
            "train_rows_sha256": C.row_sha256_bytes(fx.train_rows),
            "dev_rows_sha256": C.row_sha256_bytes(fx.dev_rows),
            "extra_norm": fx.audit["extra_norm"],
            "labels_used": False,
            "monitor_used": False,
        },
        "fold_extras": fx,
    }


__all__ = [
    "ChemStore",
    "FoldExtras",
    "build_store",
    "load_store",
    "fit_extra_norm",
    "augment_norm",
    "slice_base_norm",
    "augmented_spec",
    "build_fold_extras",
    "export_frozen_extras",
    "load_frozen_extras",
    "extras_for_training",
]
