"""Strong CPU descriptor baselines for Workstream B.

Feature groups
--------------
P1  Morgan/ECFP4, radius 2, 2048 bits, no feature invariants.
P2  every RDKit 2D descriptor returned by
    ``rdkit.Chem.Descriptors.CalcMolDescriptors``; names are frozen sorted.
P3  P1 || P2.

Preprocessing (fitted on the training fold only)
------------------------------------------------
* non-finite values are replaced by the training-fold median of the finite
  values in that column;
* a column with no finite training value or zero training variance is dropped;
* Ridge additionally standardises the kept columns with training-fold mean/std;
  XGBoost consumes the imputed values without standardisation.

The 13-dimensional physicochemical set and ``Descriptors.MolLogP`` are reported
separately.  They are not fitted from labels, but they encode chemical prior
knowledge and are therefore never described as "pure structural statistics".
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import v2_common as C

import adapters as prep_adapters  # noqa: E402


# ---------------------------------------------------------------------------
# feature computation
# ---------------------------------------------------------------------------


def _raw_and_manifest():
    C.verify_sha256(C.v1c.LIPO_RAW, C.v1c.EXPECTED_SHA256["lipo_raw"], "lipo_raw")
    C.verify_sha256(C.v1c.LIPO_ROW_MANIFEST, C.v1c.EXPECTED_SHA256["row_manifest"], "row_manifest")
    raw = prep_adapters.load_lipo_raw_artifact(C.v1c.LIPO_RAW)
    manifest = prep_adapters.load_lipo_row_manifest(C.v1c.LIPO_ROW_MANIFEST)
    return raw, manifest


def _mol_and_canonical(raw, manifest, row: int):
    from rdkit import Chem

    record = raw["rows"][int(row)]
    if "exp" in record:
        raise PermissionError(f"raw row {row} unexpectedly contains a label")
    mol = Chem.MolFromSmiles(str(record["smiles"]))
    if mol is None:
        raise ValueError(f"row {row}: RDKit failed to parse the frozen SMILES")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected = manifest["canonical_smiles_sha256"][int(row)]
    if digest != expected:
        raise ValueError(f"row {row}: canonical SMILES hash mismatch against the frozen row manifest")
    return mol, canonical, digest


def morgan_matrix(raw, manifest, rows: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X = np.zeros((len(rows), 2048), dtype=np.uint8)
    hashes = []
    for i, row in enumerate(rows):
        mol, canonical, digest = _mol_and_canonical(raw, manifest, int(row))
        hashes.append(digest)
        X[i] = np.frombuffer(generator.GetFingerprint(mol).ToBitString().encode(), dtype=np.uint8) - ord("0")
    return X, hashes


def rdkit_descriptor_matrix(raw, manifest, rows: Sequence[int]) -> tuple[np.ndarray, list[str], list[str]]:
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    names = None
    values = []
    hashes = []
    for row in rows:
        mol, canonical, digest = _mol_and_canonical(raw, manifest, int(row))
        hashes.append(digest)
        desc = Descriptors.CalcMolDescriptors(mol)
        keys = sorted(desc)
        if names is None:
            names = keys
        elif keys != names:
            raise ValueError("RDKit descriptor key set changed between molecules")
        values.append([float(desc[k]) for k in names])
    if names is None:
        raise ValueError("no rows requested")
    return np.asarray(values, dtype=np.float64), names, hashes


def physchem_matrix(raw, manifest, rows: Sequence[int]) -> tuple[np.ndarray, list[str], list[str]]:
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    values = []
    hashes = []
    for row in rows:
        mol, canonical, digest = _mol_and_canonical(raw, manifest, int(row))
        hashes.append(digest)
        desc = Descriptors.CalcMolDescriptors(mol)
        missing = [name for name in C.DESCRIPTOR_13 if name not in desc]
        if missing:
            raise ValueError(f"RDKit descriptor set lacks {missing}")
        values.append([float(desc[name]) for name in C.DESCRIPTOR_13])
    return np.asarray(values, dtype=np.float64), list(C.DESCRIPTOR_13), hashes


@dataclass
class FeatureStore:
    rows: list[int]
    P1: np.ndarray
    P2: np.ndarray
    P3: np.ndarray
    physchem13: np.ndarray
    p2_names: list[str]
    canonical_hashes: list[str]
    audit: dict[str, Any]

    def group(self, name: str) -> np.ndarray:
        if name == "P1":
            return self.P1
        if name == "P2":
            return self.P2
        if name == "P3":
            return self.P3
        raise KeyError(name)


def compute_features(rows: Sequence[int]) -> FeatureStore:
    import rdkit

    rows = [int(r) for r in rows]
    raw, manifest = _raw_and_manifest()
    P1, h1 = morgan_matrix(raw, manifest, rows)
    P2, names, h2 = rdkit_descriptor_matrix(raw, manifest, rows)
    if h1 != h2:
        raise ValueError("Morgan and descriptor canonical SMILES hashes disagree")
    P3 = np.concatenate([P1.astype(np.float64), P2], axis=1)
    phys, phys_names, h3 = physchem_matrix(raw, manifest, rows)
    if h1 != h3:
        raise ValueError("physchem canonical SMILES hashes disagree")
    audit = {
        "rdkit_version": rdkit.__version__,
        "rows": len(rows),
        "rows_sha256_int64": C.row_sha256_bytes(rows),
        "canonical_smiles_sha256": hashlib.sha256("\n".join(h1).encode()).hexdigest(),
        "P1": {"n_features": int(P1.shape[1]), "radius": 2, "bits": 2048, "use_features": False, "dtype": "uint8"},
        "P2": {
            "n_features": int(P2.shape[1]),
            "names": names,
            "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "source": "rdkit.Chem.Descriptors.CalcMolDescriptors",
        },
        "P3": {"n_features": int(P3.shape[1]), "composition": "P1_uint8_then_P2_float64"},
        "physchem13": {"names": phys_names, "source": "fixed subset of CalcMolDescriptors"},
        "P1_sha256": hashlib.sha256(np.ascontiguousarray(P1).tobytes()).hexdigest(),
        "P2_sha256": hashlib.sha256(np.ascontiguousarray(P2).tobytes()).hexdigest(),
        "P3_sha256": hashlib.sha256(np.ascontiguousarray(P3).tobytes()).hexdigest(),
        "physchem13_sha256": hashlib.sha256(np.ascontiguousarray(phys).tobytes()).hexdigest(),
        "labels_used": False,
    }
    return FeatureStore(rows, P1, P2, P3, phys, names, h1, audit)


_FEATURE_CACHE = C.CACHE / "features_v2.npz"
_FEATURE_CACHE_JSON = C.CACHE / "features_v2.json"


def load_or_compute_fit_features() -> FeatureStore:
    import v2_folds as F

    split = C.load_frozen_split()
    fit_rows = [int(r) for r in split["split"]["fit"]]
    if _FEATURE_CACHE.exists() and _FEATURE_CACHE_JSON.exists():
        meta = C.read_json(_FEATURE_CACHE_JSON)
        if meta.get("rows") == fit_rows:
            data = np.load(_FEATURE_CACHE)
            store = FeatureStore(
                rows=fit_rows,
                P1=data["P1"],
                P2=data["P2"],
                P3=data["P3"],
                physchem13=data["physchem13"],
                p2_names=meta["p2_names"],
                canonical_hashes=meta["canonical_hashes"],
                audit=meta["audit"],
            )
            if hashlib.sha256(np.ascontiguousarray(store.P3).tobytes()).hexdigest() != meta["audit"]["P3_sha256"]:
                raise PermissionError("cached descriptor matrix hash mismatch")
            return store
    store = compute_features(fit_rows)
    C.ensure_dir(C.CACHE)
    np.savez_compressed(
        _FEATURE_CACHE,
        P1=store.P1,
        P2=store.P2,
        P3=store.P3,
        physchem13=store.physchem13,
    )
    C.write_json(
        _FEATURE_CACHE_JSON,
        {
            "rows": store.rows,
            "p2_names": store.p2_names,
            "canonical_hashes": store.canonical_hashes,
            "audit": store.audit,
        },
    )
    return store


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


@dataclass
class Preprocessor:
    kind: str
    kept: np.ndarray
    median: np.ndarray
    mean: np.ndarray | None
    scale: np.ndarray | None
    audit: dict[str, Any]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)[:, self.kept].copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.median, np.where(bad)[1])
        if self.kind == "ridge":
            X = (X - self.mean) / self.scale
        if not np.isfinite(X).all():
            raise RuntimeError("preprocessed feature matrix is not finite")
        return X


def fit_preprocessor(X_train: np.ndarray, *, kind: str) -> Preprocessor:
    X = np.asarray(X_train, dtype=np.float64)
    n, d = X.shape
    medians = np.zeros(d, dtype=np.float64)
    finite_cols = np.zeros(d, dtype=bool)
    for j in range(d):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            medians[j] = float(np.median(finite))
            finite_cols[j] = True
        else:
            medians[j] = 0.0
    filled = X.copy()
    bad = ~np.isfinite(filled)
    if bad.any():
        filled[bad] = np.take(medians, np.where(bad)[1])
    std = filled.std(axis=0)
    kept = np.flatnonzero(finite_cols & (std > 1e-12))
    if kept.size == 0:
        raise ValueError("no usable descriptor columns remain on the training fold")
    filled = filled[:, kept]
    if kind == "ridge":
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
    else:
        mean = None
        scale = None
    audit = {
        "kind": kind,
        "n_input_features": int(d),
        "n_kept_features": int(kept.size),
        "n_dropped_constant_or_all_missing": int(d - kept.size),
        "kept_indices_sha256": hashlib.sha256(np.ascontiguousarray(kept, dtype=np.int64).tobytes()).hexdigest(),
        "median_nonzero_columns": int(np.sum(medians[kept] != 0.0)),
        "standardised": bool(kind == "ridge"),
        "fit_scope": "training_fold_only",
    }
    return Preprocessor(kind, kept, medians, mean, scale, audit)


def transform(prep: Preprocessor, X: np.ndarray) -> np.ndarray:
    return prep.transform(X)


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------


def fit_predict_ridge(X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=float(alpha), **C.RIDGE_FIXED)
    model.fit(X_train, y_train)
    pred = np.asarray(model.predict(X_dev), dtype=np.float64)
    if pred.shape != (X_dev.shape[0],) or not np.isfinite(pred).all():
        raise RuntimeError("Ridge produced an invalid prediction")
    return pred, {"alpha": float(alpha), "n_features": int(X_train.shape[1]), "coef_norm": float(np.linalg.norm(model.coef_))}


def fit_predict_xgb(X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    import xgboost as xgb

    kwargs = dict(C.XGB_FIXED)
    kwargs.update({k: v for k, v in params.items() if k in ("n_estimators", "max_depth", "min_child_weight")})
    model = xgb.XGBRegressor(**kwargs)
    model.fit(X_train, y_train)
    pred = np.asarray(model.predict(X_dev), dtype=np.float64)
    if pred.shape != (X_dev.shape[0],) or not np.isfinite(pred).all():
        raise RuntimeError("XGBoost produced an invalid prediction")
    return pred, {
        "n_estimators": int(kwargs["n_estimators"]),
        "max_depth": int(kwargs["max_depth"]),
        "min_child_weight": int(kwargs["min_child_weight"]),
        "n_features": int(X_train.shape[1]),
        "n_boosted_rounds": int(model.get_booster().num_boosted_rounds()),
    }


# ---------------------------------------------------------------------------
# candidate registry
# ---------------------------------------------------------------------------


def candidate_registry() -> list[dict[str, Any]]:
    out = []
    for alpha in C.RIDGE_ALPHAS:
        for group in ("P2", "P3"):
            out.append(
                {
                    "candidate_id": f"ridge_{group}_a{alpha:g}",
                    "family": "ridge",
                    "feature_group": group,
                    "params": {"alpha": float(alpha)},
                }
            )
    for group in ("P1", "P2", "P3"):
        for i, row in enumerate(C.XGB_GRID):
            out.append(
                {
                    "candidate_id": f"xgb_{group}_n{row['n_estimators']}_d{row['max_depth']}_w{row['min_child_weight']}",
                    "family": "xgboost",
                    "feature_group": group,
                    "params": dict(row),
                    "grid_index": i,
                }
            )
    return out


def _complexity_key(cand: Mapping[str, Any]) -> tuple:
    params = cand["params"]
    if cand["family"] == "ridge":
        config_rank = float(params["alpha"])
    else:
        config_rank = (int(params["n_estimators"]), int(params["max_depth"]), int(params["min_child_weight"]))
    return (
        C.FAMILY_COMPLEXITY[cand["family"]],
        C.FEATURE_COMPLEXITY[cand["feature_group"]],
        config_rank,
    )


def select_best(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("no CPU baseline results")
    means = {}
    for row in results:
        means.setdefault(row["candidate_id"], []).append(float(row["dev_rmse"]))
    summary = []
    for cid, vals in means.items():
        cand = next(r for r in results if r["candidate_id"] == cid)
        summary.append(
            {
                "candidate_id": cid,
                "family": cand["family"],
                "feature_group": cand["feature_group"],
                "params": cand["params"],
                "mean_rmse": float(np.mean(vals)),
                "mean_mae": float(np.mean([r["dev_mae"] for r in results if r["candidate_id"] == cid])),
                "fold_rmse": vals,
                "complexity_key": _complexity_key(cand),
            }
        )
    summary.sort(key=lambda x: (x["mean_rmse"], x["complexity_key"]))
    best = summary[0]
    # <0.01 tie: prefer the lower-complexity candidate.
    tied = [x for x in summary if x["mean_rmse"] - best["mean_rmse"] < 0.01]
    tied.sort(key=lambda x: x["complexity_key"])
    chosen = tied[0]
    return {
        "best_by_mean_rmse": best["candidate_id"],
        "chosen": chosen["candidate_id"],
        "tie_rule": "within 0.01 mean RMSE prefer fewer parameters / simpler features, then fixed candidate order",
        "chosen_row": chosen,
        "top10": summary[:10],
        "n_candidates": len(summary),
    }


# ---------------------------------------------------------------------------
# CV execution
# ---------------------------------------------------------------------------


def run_cv_baselines() -> dict[str, Any]:
    import v2_folds as F

    split = C.load_frozen_split()
    fit_rows = [int(r) for r in split["split"]["fit"]]
    store = load_or_compute_fit_features()
    if store.rows != fit_rows:
        raise ValueError("feature store row order does not match Fit2352")
    folds = F.load_folds()
    y = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
    candidates = candidate_registry()
    results = []
    start_all = time.perf_counter()
    for fold in C.FOLDS:
        train_pos = F.train_positions(folds, fold)
        dev_pos = F.fold_positions(folds, fold)
        y_train = y[np.asarray(train_pos, dtype=np.int64)]
        y_dev = y[np.asarray(dev_pos, dtype=np.int64)]
        # MolLogP reference (no fitting, no labels used to derive it).
        logp = store.physchem13[:, 0]
        results.append(
            {
                "candidate_id": "rdkit_mollogp",
                "family": "reference",
                "feature_group": "MolLogP",
                "params": {},
                "fold": int(fold),
                "dev_rmse": float(np.sqrt(np.mean((logp[dev_pos] - y_dev) ** 2))),
                "dev_mae": float(np.mean(np.abs(logp[dev_pos] - y_dev))),
                "dev_n": int(len(dev_pos)),
                "train_n": int(len(train_pos)),
            }
        )
        for cand in candidates:
            X = store.group(cand["feature_group"])
            prep = fit_preprocessor(X[train_pos], kind="ridge" if cand["family"] == "ridge" else "xgboost")
            Xtr = prep.transform(X[train_pos])
            Xdv = prep.transform(X[dev_pos])
            t0 = time.perf_counter()
            if cand["family"] == "ridge":
                pred, info = fit_predict_ridge(Xtr, y_train, Xdv, cand["params"]["alpha"])
            else:
                pred, info = fit_predict_xgb(Xtr, y_train, Xdv, cand["params"])
            rmse = float(np.sqrt(np.mean((pred - y_dev) ** 2)))
            mae = float(np.mean(np.abs(pred - y_dev)))
            results.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "family": cand["family"],
                    "feature_group": cand["feature_group"],
                    "params": cand["params"],
                    "fold": int(fold),
                    "dev_rmse": rmse,
                    "dev_mae": mae,
                    "dev_n": int(len(dev_pos)),
                    "train_n": int(len(train_pos)),
                    "fit_seconds": time.perf_counter() - t0,
                    "preprocessor": prep.audit,
                    "estimator": info,
                }
            )
            C.log(f"[cpu-baseline] fold={fold} {cand['candidate_id']} rmse={rmse:.5f} mae={mae:.5f}")
    selection = select_best([r for r in results if r["family"] != "reference"])
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "CV_COMPLETE",
        "feature_store": store.audit,
        "n_fits": len([r for r in results if r["family"] != "reference"]),
        "xgb_fits": len([r for r in results if r["family"] == "xgboost"]),
        "ridge_fits": len([r for r in results if r["family"] == "ridge"]),
        "expected_xgb_fits": 72,
        "expected_ridge_fits": 18,
        "elapsed_seconds": time.perf_counter() - start_all,
        "results": results,
        "selection": selection,
        "mollogp_reference": {
            "per_fold": [r for r in results if r["candidate_id"] == "rdkit_mollogp"],
            "mean_rmse": float(np.mean([r["dev_rmse"] for r in results if r["candidate_id"] == "rdkit_mollogp"])),
            "mean_mae": float(np.mean([r["dev_mae"] for r in results if r["candidate_id"] == "rdkit_mollogp"])),
        },
        "monitor_used": False,
        "labels_used_for_features": False,
    }
    C.write_json(C.ANALYSIS / "cpu_baselines_v2.json", payload)
    C.write_json(
        C.ANALYSIS / "cpu_baselines_v2_table.json",
        {
            "candidates": candidate_registry(),
            "selection": selection,
            "mollogp_reference": payload["mollogp_reference"],
        },
    )
    return payload


def refit_best_and_predict_monitor() -> dict[str, Any]:
    """Refit the chosen CPU candidate on Fit2352 and predict Monitor504."""
    split = C.load_frozen_split()
    fit_rows = [int(r) for r in split["split"]["fit"]]
    monitor_rows = [int(r) for r in split["split"]["monitor"]]
    store = load_or_compute_fit_features()
    y = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
    selection = C.read_json(C.ANALYSIS / "cpu_baselines_v2.json")["selection"]
    cid = selection["chosen"]
    cand = next(c for c in candidate_registry() if c["candidate_id"] == cid)
    monitor_store = compute_features(monitor_rows)
    X = store.group(cand["feature_group"])
    Xm = monitor_store.group(cand["feature_group"])
    prep = fit_preprocessor(X, kind="ridge" if cand["family"] == "ridge" else "xgboost")
    Xtr = prep.transform(X)
    Xmon = prep.transform(Xm)
    if cand["family"] == "ridge":
        pred, info = fit_predict_ridge(Xtr, y, Xmon, cand["params"]["alpha"])
    else:
        pred, info = fit_predict_xgb(Xtr, y, Xmon, cand["params"])
    key = f"CPU_{cid}"
    npz_path = C.PREDICTIONS / f"{key}.npz"
    C.ensure_dir(npz_path.parent)
    np.savez(npz_path, rows=np.asarray(monitor_rows, dtype=np.int64), pred=pred)
    meta = {
        "protocol_id": C.PROTOCOL_ID,
        "key": key,
        "arm": "CPU",
        "candidate_id": cid,
        "family": cand["family"],
        "feature_group": cand["feature_group"],
        "params": cand["params"],
        "file": C.track_rel(npz_path),
        "sha256": C.sha256_file(npz_path),
        "rows": monitor_rows,
        "rows_sha256": C.row_sha256_bytes(monitor_rows),
        "fit_rows": fit_rows,
        "fit_rows_sha256": C.row_sha256_bytes(fit_rows),
        "y_fit_sha256": hashlib.sha256(np.ascontiguousarray(y, dtype="<f8").tobytes()).hexdigest(),
        "preprocessor": prep.audit,
        "estimator": info,
        "feature_store": {"fit": store.audit, "monitor": monitor_store.audit},
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
        "n_predictions": int(pred.shape[0]),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
    }
    C.write_json(C.PREDICTIONS / f"{key}.json", meta)
    return meta


__all__ = [
    "FeatureStore",
    "compute_features",
    "load_or_compute_fit_features",
    "fit_preprocessor",
    "transform",
    "fit_predict_ridge",
    "fit_predict_xgb",
    "candidate_registry",
    "select_best",
    "run_cv_baselines",
    "refit_best_and_predict_monitor",
]
