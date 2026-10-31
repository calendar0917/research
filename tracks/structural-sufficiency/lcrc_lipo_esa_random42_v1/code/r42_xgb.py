"""CPU chemistry baseline: ECFP4 (2048, radius 2) + 217 RDKit descriptors + XGBoost.

The feature definition and the estimator hyper-parameters are the frozen
historical v2 P3 definition (``xgb_P3_n800_d5_w1``), migrated unchanged.  The
217 descriptor names and their order are read once from the v2 audit artifact,
verified against the current RDKit version, and frozen in
``data/xgb_descriptor_names_v1.json``.

Preprocessing is fitted on Train3360 only: non-finite values use the training
median per column and zero-variance columns are dropped.  XGBoost consumes the
imputed values without standardisation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import r42_common as C
import r42_data as D
import r42_labels as L
import r42_metrics as MET

DESCRIPTOR_NAMES_FILE = C.DATA_DIR / "xgb_descriptor_names_v1.json"
FEATURE_CACHE = C.CACHE_DIR / "xgb_features_v1.npz"
HIST_V2_BASELINES = C.HIST_V2 / "analysis/cpu_baselines_v2.json"


def freeze_descriptor_names(path: Path | str = DESCRIPTOR_NAMES_FILE) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    payload = C.read_json(HIST_V2_BASELINES)
    names = list(payload["feature_store"]["P2"]["names"])
    if len(names) != 217:
        raise ValueError(f"historical P2 descriptor list has {len(names)} names, expected 217")
    mol = Chem.MolFromSmiles("CCO")
    current = sorted(Descriptors.CalcMolDescriptors(mol))
    if current != names:
        missing = sorted(set(names) - set(current))
        extra = sorted(set(current) - set(names))
        raise PermissionError(f"RDKit descriptor set changed; missing={missing[:5]} extra={extra[:5]}")
    out = {
        "protocol_id": C.PROTOCOL_ID,
        "n_names": len(names),
        "names": names,
        "order": "ascending lexicographic (frozen v2 P3 definition)",
        "names_sha256": C.sha256_bytes("\n".join(names).encode("utf-8")),
        "source": C.track_rel(HIST_V2_BASELINES),
        "source_feature_group": "P2",
        "rdkit_version_at_freeze": __import__("rdkit").__version__,
    }
    C.write_json(path, out)
    return out


def load_descriptor_names() -> list[str]:
    payload = C.read_json(DESCRIPTOR_NAMES_FILE)
    names = list(payload["names"])
    if len(names) != 217 or C.sha256_bytes("\n".join(names).encode("utf-8")) != payload["names_sha256"]:
        raise PermissionError("frozen descriptor name list drifted")
    return names


def _mol_and_canonical(raw: Mapping[str, Any], manifest: Mapping[str, Any], row: int):
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
        raise ValueError(f"row {row}: canonical SMILES hash mismatch")
    return mol


def compute_features(rows: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (Morgan uint8 [n,2048], descriptors float64 [n,217])."""
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    raw = D.load_raw_graph_artifact()
    manifest = C.read_json(C.ROW_MANIFEST)
    names = load_descriptor_names()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    morgan = np.zeros((len(rows), 2048), dtype=np.uint8)
    desc = np.zeros((len(rows), len(names)), dtype=np.float64)
    for i, row in enumerate(rows):
        mol = _mol_and_canonical(raw, manifest, int(row))
        morgan[i] = np.frombuffer(generator.GetFingerprint(mol).ToBitString().encode(), dtype=np.uint8) - ord("0")
        values = Descriptors.CalcMolDescriptors(mol)
        if sorted(values) != names:
            raise ValueError("RDKit descriptor key set changed between molecules")
        desc[i] = [float(values[name]) for name in names]
    return morgan, desc


def load_or_build_features() -> dict[str, Any]:
    if FEATURE_CACHE.exists():
        meta_path = FEATURE_CACHE.with_suffix(".json")
        if meta_path.exists():
            meta = C.read_json(meta_path)
            if C.sha256_file(FEATURE_CACHE) == meta["file_sha256"]:
                with np.load(FEATURE_CACHE) as npz:
                    return {"P1": npz["P1"], "P2": npz["P2"], "meta": meta, "rebuilt": False}
    rows = list(range(C.N_MOLECULES))
    morgan, desc = compute_features(rows)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, P1=morgan, P2=desc)
    meta = {
        "protocol_id": C.PROTOCOL_ID,
        "n_rows": len(rows),
        "P1": {"radius": 2, "bits": 2048, "dtype": "uint8", "sha256": C.sha256_bytes(morgan.tobytes())},
        "P2": {"n_features": int(desc.shape[1]), "dtype": "float64", "sha256": C.sha256_bytes(desc.tobytes())},
        "descriptor_names_sha256": C.read_json(DESCRIPTOR_NAMES_FILE)["names_sha256"],
        "file_sha256": C.sha256_file(FEATURE_CACHE),
    }
    C.write_json(FEATURE_CACHE.with_suffix(".json"), meta)
    return {"P1": morgan, "P2": desc, "meta": meta, "rebuilt": True}


def fit_preprocessor(X: np.ndarray) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    medians = np.zeros(d, dtype=np.float64)
    finite_cols = np.zeros(d, dtype=bool)
    for j in range(d):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            medians[j] = float(np.median(finite))
            finite_cols[j] = True
    filled = X.copy()
    bad = ~np.isfinite(filled)
    if bad.any():
        filled[bad] = np.take(medians, np.where(bad)[1])
    std = filled.std(axis=0)
    kept = np.flatnonzero(finite_cols & (std > 1e-12))
    if kept.size == 0:
        raise ValueError("no usable descriptor columns remain on Train3360")
    return {
        "kept": kept,
        "medians": medians,
        "n_input_features": int(d),
        "n_kept_features": int(kept.size),
        "n_dropped_constant_or_all_missing": int(d - kept.size),
        "kept_indices_sha256": C.sha256_bytes(np.ascontiguousarray(kept, dtype=np.int64).tobytes()),
        "fit_scope": "train3360_only",
    }


def transform(prep: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)[:, prep["kept"]].copy()
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(prep["medians"][prep["kept"]], np.where(bad)[1])
    if not np.isfinite(X).all():
        raise RuntimeError("preprocessed feature matrix is not finite")
    return X


def train_xgb(cfg: Mapping[str, Any], split: Mapping[str, Any], labels: L.TrainValidLabels, run_dir: Path) -> dict[str, Any]:
    import xgboost as xgb

    labels = L.assert_train_valid_only(labels)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    store = load_or_build_features()
    rows_train = [int(r) for r in split["rows"]["train"]]
    rows_valid = [int(r) for r in split["rows"]["valid"]]
    X_train = np.concatenate([store["P1"][rows_train], store["P2"][rows_train]], axis=1)
    X_valid = np.concatenate([store["P1"][rows_valid], store["P2"][rows_valid]], axis=1)
    prep = fit_preprocessor(X_train)
    Xtr = transform(prep, X_train)
    Xva = transform(prep, X_valid)
    params = {
        "n_estimators": int(cfg["n_estimators"]),
        "max_depth": int(cfg["max_depth"]),
        "min_child_weight": int(cfg["min_child_weight"]),
        "learning_rate": float(cfg["learning_rate"]),
        "subsample": float(cfg["subsample"]),
        "colsample_bytree": float(cfg["colsample_bytree"]),
        "reg_lambda": float(cfg["reg_lambda"]),
        "tree_method": str(cfg["tree_method"]),
        "random_state": int(cfg["random_state"]),
        "n_jobs": int(cfg["n_jobs"]),
    }
    model = xgb.XGBRegressor(**params)
    t_fit = time.perf_counter()
    model.fit(Xtr, labels.y_train)
    fit_seconds = time.perf_counter() - t_fit
    t_pred = time.perf_counter()
    pred_train = np.asarray(model.predict(Xtr), dtype=np.float64)
    pred_valid = np.asarray(model.predict(Xva), dtype=np.float64)
    predict_seconds = time.perf_counter() - t_pred
    if not (np.isfinite(pred_train).all() and np.isfinite(pred_valid).all()):
        raise RuntimeError("XGBoost produced non-finite predictions")
    train_metrics = MET.all_metrics(pred_train, labels.y_train)
    valid_metrics = MET.all_metrics(pred_valid, labels.y_valid)
    head_path = run_dir / "xgb_p3.json"
    model.save_model(head_path)
    np.savez(run_dir / "valid_predictions.npz", rows=np.asarray(rows_valid, dtype=np.int64), pred=pred_valid)
    np.savez(run_dir / "train_predictions.npz", rows=np.asarray(rows_train, dtype=np.int64), pred=pred_train)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "model": "XGB-P3",
        "seed": int(cfg["random_state"]),
        "config": dict(cfg),
        "config_hash": C.config_hash(dict(cfg)),
        "env": {**C.env_info(), "device": "cpu", "rdkit": __import__("rdkit").__version__, "xgboost": xgb.__version__},
        "features": {
            **store["meta"],
            "rebuilt": store["rebuilt"],
            "preprocessor": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in prep.items()},
            "n_features_used": int(prep["n_kept_features"]),
            "source": "historical v2 P3 definition (ECFP4 2048 + 217 RDKit descriptors)",
        },
        "data": {
            "split_file_sha256": C.sha256_file(C.SPLIT_FILE),
            "train_rows_sha256": C.rows_sha256_int64(rows_train),
            "valid_rows_sha256": C.rows_sha256_int64(rows_valid),
            "n_train": len(rows_train),
            "n_valid": len(rows_valid),
            "label_audit": dict(labels.audit),
            "test_labels_read": False,
        },
        "training": {
            "checkpoint_rule": "single_fit_no_selection",
            "n_boosted_rounds": int(model.get_booster().num_boosted_rounds()),
            "fit_seconds": float(fit_seconds),
            "predict_seconds_train_plus_valid": float(predict_seconds),
            "total_seconds": float(time.perf_counter() - t0),
        },
        "metrics": {
            "train": {**train_metrics, "units": "raw_logD"},
            "valid": {**valid_metrics, "units": "raw_logD"},
        },
        "model_path": C.track_rel(head_path),
        "model_sha256": C.sha256_file(head_path),
        "predictions": {
            "valid_path": C.track_rel(run_dir / "valid_predictions.npz"),
            "valid_sha256": C.sha256_file(run_dir / "valid_predictions.npz"),
            "train_path": C.track_rel(run_dir / "train_predictions.npz"),
            "train_sha256": C.sha256_file(run_dir / "train_predictions.npz"),
        },
        "label_scope": "train3360_and_valid420_only",
        "test_labels_read": False,
        "gpu_seconds": 0.0,
    }
    C.write_json(run_dir / "train.json", payload)
    C.log(f"[XGB-P3] train={train_metrics['rmse']:.5f} valid={valid_metrics['rmse']:.5f} fit={fit_seconds:.1f}s")
    return payload


def predict_rows(model_path: Path | str, rows: Sequence[int]) -> np.ndarray:
    import xgboost as xgb

    store = load_or_build_features()
    X = np.concatenate([store["P1"][list(rows)], store["P2"][list(rows)]], axis=1)
    # The preprocessor is recovered from the frozen train manifest.
    run = C.read_json(C.RUNS / "xgb_p3_s0" / "train.json")
    prep = run["features"]["preprocessor"]
    prep = {**prep, "kept": np.asarray(prep["kept"], dtype=np.int64), "medians": np.asarray(prep["medians"], dtype=np.float64)}
    X = transform(prep, X)
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    return np.asarray(model.predict(X), dtype=np.float64)


__all__ = [
    "freeze_descriptor_names",
    "load_descriptor_names",
    "compute_features",
    "load_or_build_features",
    "fit_preprocessor",
    "transform",
    "train_xgb",
    "predict_rows",
]
