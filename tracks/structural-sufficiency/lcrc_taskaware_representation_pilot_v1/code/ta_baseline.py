"""P1 -- the frozen ECFP4 + RDKit-descriptor + XGBoost baseline on the two new
Lipophilicity repeats.

The recipe is Sprint 02's, unchanged: the same 2048-bit Morgan radius-2 block,
the same 217-name RDKit descriptor list, the same Train-only median imputation
and constant-column drop, the same XGBoost hyperparameters, no tuning, no
Validation-driven choice.  What changes is only the row list: the model is
fitted on ``Lipop.random123`` / ``Lipop.random124`` Train rows and validated on
their Validation rows.

The feature matrix itself is *label-free* (it is built from all 4200 SMILES), so
the same cached matrix is reused; only the split slices differ.

Stages
------
``--stage train``   fit on the repeat's Train rows, predict Validation, write the model
``--stage freeze``  write ``analysis/baseline_repeat_manifest.json`` (pre-Test gate)
``--stage test``    one-shot Test scoring for both repeats, after the freeze
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import ta_common as C  # noqa: I001

import mb2_baseline as B
import mb2_metrics as MET
import ta_data as TD

OUT_DIR = C.ANALYSIS / "baseline_repeat"
MODEL_DIR = OUT_DIR / "models"
MANIFEST = C.ANALYSIS / "baseline_repeat_manifest.json"


def _model_path(suffix: int):
    return MODEL_DIR / f"Lipophilicity_random{suffix}.ubj"


def train_repeat(suffix: int, *, overwrite: bool = False) -> dict[str, Any]:
    repeat = TD.lipo_repeat_split(suffix)
    rows = repeat["rows"]
    X, feat_info = B.load_features("Lipophilicity")
    labels = TD.chembench_targets("Lipophilicity", rows, include_test=False)
    y_train = np.asarray(labels["y"]["train"], dtype=np.float64).reshape(-1)
    y_valid = np.asarray(labels["y"]["valid"], dtype=np.float64).reshape(-1)
    out_path = OUT_DIR / f"Lipophilicity_random{suffix}.json"
    if out_path.exists() and not overwrite:
        return C.read_json(out_path)

    idx_train = np.asarray(rows["train"], dtype=np.int64)
    idx_valid = np.asarray(rows["valid"], dtype=np.int64)
    X_train, X_valid = X[idx_train], X[idx_valid]
    stats = B._preprocess_train_only(X_train)
    Xtr, Xva = B._apply(stats, X_train), B._apply(stats, X_valid)
    model = B._xgb("regression")
    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    fit_seconds = time.perf_counter() - t0
    valid_pred = np.asarray(model.predict(Xva), dtype=np.float64).reshape(-1, 1)
    train_pred = np.asarray(model.predict(Xtr), dtype=np.float64).reshape(-1, 1)
    train_metrics = MET.regression_metrics(train_pred[:, 0], y_train)
    valid_metrics = MET.regression_metrics(valid_pred[:, 0], y_valid)
    C.ensure_dir(MODEL_DIR)
    model.save_model(str(_model_path(suffix)))
    np.savez(OUT_DIR / f"Lipophilicity_random{suffix}_valid_predictions.npz", rows=idx_valid, pred=valid_pred[:, 0], y_true=y_valid)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "dataset": "Lipophilicity",
        "repeat_suffix": int(suffix),
        "split_file": repeat["file"],
        "split_file_sha256": repeat["sha256"],
        "split_rows_sha256": repeat["rows_sha256"],
        "rows": {k: len(v) for k, v in rows.items()},
        "features": {
            "config": dict(C.V2C.BASELINE_CONFIG),
            "feature_matrix": feat_info,
            "n_features_after_drop": int(len(stats["keep"])),
            "n_dropped_constant": int(stats["n_dropped_columns"]),
            "median_sha256": stats["median_sha256"],
            "keep_sha256": stats["keep_sha256"],
            "preprocessing": dict(B.PREPROCESSING),
        },
        "fit_seconds": float(fit_seconds),
        "train": train_metrics,
        "valid": valid_metrics,
        "model_path": C.track_rel(_model_path(suffix)),
        "model_sha256": C.sha256_file(_model_path(suffix)),
        "valid_predictions": C.track_rel(OUT_DIR / f"Lipophilicity_random{suffix}_valid_predictions.npz"),
        "test_scored": False,
        "test_used_for_any_choice": False,
        "environment": {"xgboost": __import__("xgboost").__version__, "git_commit": C.git_commit()},
    }
    C.write_json(out_path, payload)
    C.log(
        f"[baseline lipo random{suffix}] train RMSE {train_metrics['rmse']:.5f} "
        f"valid RMSE {valid_metrics['rmse']:.5f} ({fit_seconds:.1f}s)"
    )
    return payload


def freeze() -> dict[str, Any]:
    entries = {}
    for suffix in C.LIPO_EXTRA_REPEATS:
        path = OUT_DIR / f"Lipophilicity_random{suffix}.json"
        if not path.exists():
            raise SystemExit(f"missing baseline report {C.track_rel(path)}; run --stage train")
        entries[str(suffix)] = {
            "report": C.track_rel(path),
            "report_sha256": C.sha256_file(path),
            "model": C.track_rel(_model_path(suffix)),
            "model_sha256": C.sha256_file(_model_path(suffix)),
            "valid_rmse": C.read_json(path)["valid"]["rmse"],
            "split_rows_sha256": C.read_json(path)["split_rows_sha256"],
        }
    payload = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "workstream": "P1 Lipophilicity repeat robustness -- matched classical baseline",
        "recipe": dict(C.V2C.BASELINE_CONFIG),
        "entries": entries,
        "test_policy": "Test is scored once, after this freeze, per repeat; never used for any choice",
    }
    C.write_json(MANIFEST, payload)
    C.log(f"wrote {C.track_rel(MANIFEST)}")
    return payload


def score_test(suffix: int) -> dict[str, Any]:
    if not MANIFEST.exists():
        raise SystemExit("baseline is not frozen; run --stage freeze before any Test access")
    manifest = C.read_json(MANIFEST)
    entry = manifest["entries"][str(suffix)]
    report_path = C.ROOT / entry["report"] if hasattr(C, "ROOT") else C.REPO / entry["report"]
    report = C.read_json(report_path)
    if C.sha256_file(_model_path(suffix)) != entry["model_sha256"]:
        raise SystemExit("baseline model changed after the freeze")
    repeat = TD.lipo_repeat_split(suffix)
    X, _ = B.load_features("Lipophilicity")
    labels = TD.chembench_targets("Lipophilicity", repeat["rows"], include_test=True)
    y_test = np.asarray(labels["y"]["test"], dtype=np.float64).reshape(-1)
    stats = B._preprocess_train_only(X[np.asarray(repeat["rows"]["train"], dtype=np.int64)])
    import xgboost as xgb

    model = xgb.XGBRegressor(**dict(C.V2C.BASELINE_CONFIG["xgboost"]))
    model.load_model(str(_model_path(suffix)))
    X_test = B._apply(stats, X[np.asarray(repeat["rows"]["test"], dtype=np.int64)])
    pred = np.asarray(model.predict(X_test), dtype=np.float64).reshape(-1)
    metrics = MET.regression_metrics(pred, y_test)
    np.savez(
        OUT_DIR / f"Lipophilicity_random{suffix}_test_predictions.npz",
        rows=np.asarray(repeat["rows"]["test"], dtype=np.int64),
        pred=pred,
        y_true=y_test,
    )
    payload = {
        "dataset": "Lipophilicity",
        "repeat_suffix": int(suffix),
        "test": metrics,
        "n_test": int(y_test.size),
        "split_rows_sha256": repeat["rows_sha256"]["test"],
        "test_used_for_any_selection": False,
        "model_sha256": entry["model_sha256"],
        "prediction_sha256": C.sha256_file(OUT_DIR / f"Lipophilicity_random{suffix}_test_predictions.npz"),
        "scored_utc": C.utcnow(),
        "model_train_report": report["model_path"],
        "exposure": "PUBLIC_SPLIT_ROBUSTNESS (Lipophilicity Test labels were already read in Sprint 01)",
    }
    C.write_json(OUT_DIR / f"Lipophilicity_random{suffix}_test.json", payload)
    C.log(f"[baseline lipo random{suffix}] Test RMSE {metrics['rmse']:.5f}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 matched baseline on the two new Lipophilicity repeats")
    parser.add_argument("--stage", required=True, choices=["train", "freeze", "test", "all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.stage in ("train", "all"):
        for suffix in C.LIPO_EXTRA_REPEATS:
            train_repeat(suffix, overwrite=args.overwrite)
    if args.stage in ("freeze", "all"):
        freeze()
    if args.stage == "test":
        for suffix in C.LIPO_EXTRA_REPEATS:
            score_test(suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
