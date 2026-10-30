"""Baseline D: ECFP4 (Morgan r=2, 2048 bits) + XGBoost, refit on Fit2352.

The fingerprint and XGBoost hyper-parameters are the frozen BJOC
Lipophilicity-benchmark values (``baselines.json``).  Everything is re-fit on
this round's Fit 2352 rows and scored on this round's Monitor 504 rows; the
previous BJOC train3360/valid420 numbers are never reused as a comparison.

The baseline is CPU-only and therefore runs locally (RDKit is an optional local
dependency and is not installed in the remote GPU environment).  It reads only
Fit labels; ``predict`` never touches the label artifact at all.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

import access_guard
import fit_labels
import screen_common as C


def _load_raw():
    import adapters as prep_adapters

    C.verify_sha256(C.LIPO_RAW, C.EXPECTED_SHA256["lipo_raw"], "lipo_raw")
    C.verify_sha256(C.LIPO_ROW_MANIFEST, C.EXPECTED_SHA256["row_manifest"], "row_manifest")
    raw = prep_adapters.load_lipo_raw_artifact(C.LIPO_RAW)
    manifest = prep_adapters.load_lipo_row_manifest(C.LIPO_ROW_MANIFEST)
    return raw, manifest, prep_adapters


def fingerprint_rows(rows: list[int], cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build ECFP4 bits for explicit rows, verifying SMILES row alignment."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    cfg = cfg or C.load_config()
    raw, manifest, prep_adapters = _load_raw()
    radius = int(cfg["baseline"]["fp_radius"])
    bits = int(cfg["baseline"]["fp_bits"])
    use_features = bool(cfg["baseline"]["fp_use_features"])
    generator_kwargs: dict[str, Any] = {"radius": int(radius), "fpSize": int(bits)}
    if use_features:
        # RDKit >= 2025 exposes feature invariants separately from the generator.
        generator_kwargs["atomInvariantsGenerator"] = (
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        )
    generator = rdFingerprintGenerator.GetMorganGenerator(**generator_kwargs)
    fp = np.zeros((len(rows), bits), dtype=np.uint8)
    hashes = []
    for i, row in enumerate(rows):
        record = raw["rows"][int(row)]
        if "exp" in record:
            raise PermissionError(f"raw row {row} unexpectedly contains a label")
        smiles = str(record["smiles"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"row {row}: RDKit failed to parse the frozen SMILES")
        canonical = Chem.MolToSmiles(mol, canonical=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        expected = manifest["canonical_smiles_sha256"][int(row)]
        if digest != expected:
            raise ValueError(f"row {row}: canonical SMILES hash mismatch against the frozen row manifest")
        hashes.append(digest)
        bitvect = generator.GetFingerprint(mol)
        fp[i] = np.frombuffer(bitvect.ToBitString().encode(), dtype=np.uint8) - ord("0")
    return {
        "X": fp,
        "rows": list(rows),
        "canonical_smiles_sha256": hashes,
        "feature_sha256": hashlib.sha256(np.ascontiguousarray(fp).tobytes()).hexdigest(),
        "spec": {
            "kind": "Morgan",
            "radius": radius,
            "n_bits": bits,
            "use_features": use_features,
            "dtype": "uint8",
        },
    }


def _xgb_params(cfg: Mapping[str, Any], random_state: int) -> dict[str, Any]:
    xgb = dict(cfg["baseline"]["xgboost"])
    xgb["random_state"] = int(random_state)
    return xgb


def fit_and_predict(replicates: tuple[int, ...] | None = None) -> dict[str, Any]:
    import xgboost as xgb

    cfg = C.load_config()
    replicates = tuple(cfg["baseline"]["random_states"]) if replicates is None else tuple(replicates)
    access_guard.arm()
    split = C.load_frozen_split()
    fit_rows = [int(r) for r in split["split"]["fit"]]
    monitor_rows = [int(r) for r in split["split"]["monitor"]]
    fit_labels_payload = fit_labels.load_fit_labels(arm_guard=False)
    y_fit = fit_labels_payload["y"]

    fit_fp = fingerprint_rows(fit_rows, cfg)
    monitor_fp = fingerprint_rows(monitor_rows, cfg)
    C.ensure_dir(C.PREDICTIONS)
    rows_sha = hashlib.sha256(np.ascontiguousarray(np.asarray(monitor_rows, dtype=np.int64)).tobytes()).hexdigest()
    out: dict[str, Any] = {"fits": {}, "config_hash": C.config_hash(cfg)}
    for r in replicates:
        params = _xgb_params(cfg, r)
        model = xgb.XGBRegressor(**params)
        model.fit(fit_fp["X"].astype(np.float32), np.asarray(y_fit, dtype=np.float64))
        pred = np.asarray(model.predict(monitor_fp["X"].astype(np.float32)), dtype=np.float64)
        if pred.shape != (len(monitor_rows),) or not np.isfinite(pred).all():
            raise RuntimeError("XGBoost produced an invalid prediction vector")
        key = f"D_r{r}"
        npz_path = C.PREDICTIONS / f"{key}.npz"
        np.savez(npz_path, rows=np.asarray(monitor_rows, dtype=np.int64), pred=pred)
        sha = C.sha256_file(npz_path)
        meta = {
            "protocol_id": C.PROTOCOL_ID,
            "key": key,
            "arm": "D",
            "seed": int(r),
            "random_state": int(r),
            "file": C.track_rel(npz_path),
            "sha256": sha,
            "rows": monitor_rows,
            "rows_sha256": rows_sha,
            "model": {
                "model": "ecfp4_xgb",
                "parameters": int(model.get_booster().num_boosted_rounds()),
                "trees": int(model.get_booster().num_boosted_rounds()),
            },
            "features": {
                "fit": fit_fp["spec"],
                "fit_feature_sha256": fit_fp["feature_sha256"],
                "monitor_feature_sha256": monitor_fp["feature_sha256"],
                "canonical_smiles_alignment": "verified_against_frozen_row_manifest",
            },
            "xgboost": params,
            "fit_rows": len(fit_rows),
            "fit_label_audit": fit_labels_payload["audit"],
            "config_hash": C.config_hash(cfg),
            "monitor_labels_read": False,
            "reserved_labels_read": False,
            "official_valid_test_read": False,
            "n_predictions": int(pred.shape[0]),
            "prediction_mean": float(pred.mean()),
            "prediction_std": float(pred.std()),
        }
        C.write_json(C.PREDICTIONS / f"{key}.json", meta)
        out["fits"][key] = {"sha256": sha, "path": C.track_rel(npz_path), "params": params}
        C.log(f"[baseline D] random_state={r} -> {C.track_rel(npz_path)} sha={sha[:12]}")
    access_guard.disarm()
    return out


def determinism_check(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Refit the frozen baseline twice with the same seed and compare exactly."""
    import xgboost as xgb

    cfg = cfg or C.load_config()
    access_guard.arm()
    split = C.load_frozen_split()
    fit_rows = [int(r) for r in split["split"]["fit"]]
    monitor_rows = [int(r) for r in split["split"]["monitor"]]
    y_fit = fit_labels.load_fit_labels(arm_guard=False)["y"]
    fit_fp = fingerprint_rows(fit_rows, cfg)["X"]
    monitor_fp = fingerprint_rows(monitor_rows, cfg)["X"]
    preds = []
    for _ in range(2):
        model = xgb.XGBRegressor(**_xgb_params(cfg, C.PRIMARY_BASELINE_REPLICATE))
        model.fit(fit_fp.astype(np.float32), np.asarray(y_fit, dtype=np.float64))
        preds.append(np.asarray(model.predict(monitor_fp.astype(np.float32)), dtype=np.float64))
    access_guard.disarm()
    diff = float(np.abs(preds[0] - preds[1]).max())
    return {
        "max_abs_diff": diff,
        "bitwise_identical": bool(diff == 0.0),
        "n_jobs": _xgb_params(cfg, C.PRIMARY_BASELINE_REPLICATE)["n_jobs"],
        "random_state": C.PRIMARY_BASELINE_REPLICATE,
        "prediction_sha256": hashlib.sha256(preds[0].tobytes()).hexdigest(),
    }


__all__ = ["fingerprint_rows", "fit_and_predict", "determinism_check"]
