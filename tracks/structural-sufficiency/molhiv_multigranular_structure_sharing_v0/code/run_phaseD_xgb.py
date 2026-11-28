"""Phase D: XGB_STRUCT = historical_XGB_features + C + F.

The historical baseline (``lcrc_molnet_benchmark_pilot_v2``) is **never retrained
for reporting**.  This script

1. rebuilds the historical feature matrix + Train-only preprocessing from the
   frozen cache and proves it reproduces the frozen manifest hashes
   (``median_sha256``/``keep_sha256``/``n_used_columns``);
2. re-fits the *frozen* historical recipe once as a **verification** and compares
   its per-molecule predictions with the stored historical predictions
   (bit-level identity is what makes the Phase D comparison EXACT_MATCH);
3. fits exactly one new model, ``XGB_STRUCT``, with the same recipe and the same
   base features plus the frozen C and F blocks;
4. reports Train/Valid/Test metrics for the new model and does paired
   per-molecule comparisons against the historical predictions.

Why dense: with a sparse DMatrix and ``missing=np.nan``, *un-stored* zeros are
treated as missing, not as the value 0 (verified with ``DMatrix.num_nonmissing()``
and by a dense-vs-sparse prediction test).  The historical recipe is dense
float32 anyway, so the combined design keeps every explicit zero an explicit
zero.  The only sparse block is documented in ``analysis/phaseD_xgb.json``.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

import numpy as np

import msds_common as C
import msds_data as D
import msds_stats as ST
import run_phaseB_linear as LIN

HIST_MANIFEST_KEY = "ogbg-molhiv"


def _column_median_std(X: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = X.shape[1]
    median = np.zeros(width, dtype=np.float64)
    std = np.zeros(width, dtype=np.float64)
    for j in range(width):
        col = X[rows, j]
        median[j] = np.nanmedian(col)
    median = np.where(np.isfinite(median), median, 0.0)
    np.copyto(X, median[None, :], where=np.isnan(X))
    for j in range(width):
        std[j] = X[rows, j].std()
    return median, std


def historical_block() -> tuple[np.ndarray, dict[str, Any]]:
    """Historical dense feature matrix with Train-only preprocessing applied in place."""
    from pathlib import Path

    if not C.HIST_BASELINE_FEATURES.exists():
        raise SystemExit(f"missing historical feature cache: {C.rel(C.HIST_BASELINE_FEATURES)}")
    manifest = C.read_json(C.HIST_BASELINE_MANIFEST)
    entry = manifest["datasets"][HIST_MANIFEST_KEY]
    rows = {k: np.asarray(v, dtype=np.int64) for k, v in entry["rows"].items() if k in ("train", "valid", "test")}
    with np.load(C.HIST_BASELINE_FEATURES, allow_pickle=False) as z:
        X = np.array(z["features"], dtype=np.float32, copy=True)
        descriptor_names = [str(s) for s in z["descriptor_names"]]
    n_imputed_before = int(np.isnan(X).sum())
    median, std = _column_median_std(X, rows["train"])
    keep = np.flatnonzero(std > 0)
    checks = {
        "features_path": C.rel(C.HIST_BASELINE_FEATURES),
        "features_sha256": C.sha256_file(C.HIST_BASELINE_FEATURES),
        "features_shape": list(X.shape),
        "n_imputed_cells_train": n_imputed_before,
        "median_sha256": C.sha256_bytes(np.ascontiguousarray(median, dtype=np.float64).tobytes()),
        "keep_sha256": C.sha256_bytes(np.ascontiguousarray(keep, dtype=np.int64).tobytes()),
        "n_used_columns": int(keep.size),
        "n_dropped_columns": int(X.shape[1] - keep.size),
        "manifest_n_used_columns": int(entry["preprocessing"]["n_used_columns"]),
        "manifest_n_dropped_columns": int(entry["preprocessing"]["n_dropped_columns"]),
        "manifest_median_sha256": entry["preprocessing"]["median_sha256"],
        "manifest_keep_sha256": entry["preprocessing"]["keep_sha256"],
        "manifest_features_sha256": entry["features_sha256"],
        "descriptor_names_match": len(descriptor_names) == 217,
    }
    checks["preprocessing_reproduced"] = bool(
        checks["median_sha256"] == checks["manifest_median_sha256"]
        and checks["keep_sha256"] == checks["manifest_keep_sha256"]
        and checks["n_used_columns"] == checks["manifest_n_used_columns"]
        and checks["features_sha256"] == checks["manifest_features_sha256"]
    )
    Xk = np.ascontiguousarray(X[:, keep])
    del X
    return Xk, checks


def _dense_block(blocks: list[Any], rows: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((rows.size, width), dtype=np.float32)
    col = 0
    for mat in blocks:
        sub = mat[rows]
        out[:, col: col + sub.shape[1]] = np.asarray(sub.todense(), dtype=np.float32)
        col += sub.shape[1]
    return out


def fit_xgb(X: np.ndarray, y: np.ndarray, params: dict[str, Any]):
    import xgboost as xgb

    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    model.fit(X, y.astype(int))
    return model


def main() -> None:
    C.ensure_dirs()
    t0 = time.perf_counter()
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = C.read_json(C.HIST_BASELINE_MANIFEST)
    params = dict(manifest["frozen_config"]["xgboost"])
    molecules = D.load_molecules()
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}
    y = molecules.label.astype(np.float64)

    Xk, hist_checks = historical_block()
    print(f"[phaseD] historical block {Xk.shape}, preprocessing reproduced="
          f"{hist_checks['preprocessing_reproduced']}", flush=True)
    if not hist_checks["preprocessing_reproduced"]:
        raise SystemExit("historical preprocessing was not reproduced; refusing to report Phase D")

    # ---- verification refit of the frozen historical recipe (not a reported baseline)
    t_fit = time.perf_counter()
    model_hist = fit_xgb(Xk[splits["train"]], y[splits["train"]], params)
    hist_refit_seconds = time.perf_counter() - t_fit
    refit = {}
    for split_name in ("valid", "test"):
        rows = splits[split_name]
        logit = model_hist.predict(Xk[rows], output_margin=True)
        refit[split_name] = ST.auc_ap(y[rows], logit)
        stored = np.load(C.HIST_BASELINE_DIR / f"ogbg-molhiv_{split_name}_predictions.npz")
        same = np.array_equal(np.asarray(stored["rows"]), rows)
        diff = float(np.abs(np.asarray(stored["pred"]).reshape(-1) - logit).max()) if same else None
        refit[split_name].update({
            "stored_rows_match": bool(same),
            "max_abs_diff_vs_stored_predictions": diff,
            "stored_roc_auc": manifest_rows_auc(split_name),
        })
    # metrics of the frozen historical model as recorded in its own artefacts
    for split_name in ("valid", "test"):
        if split_name == "valid":
            rec = manifest["datasets"][HIST_MANIFEST_KEY]["metrics_valid"]["roc_auc"]
        else:
            rec = C.read_json(C.HIST_BASELINE_DIR / "ogbg-molhiv_test.json")["metrics"]["test"]["roc_auc"]
        refit[split_name]["recorded_roc_auc"] = float(rec)
    hist_identity = {
        "recipe": params,
        "refit_seconds": hist_refit_seconds,
        "splits": refit,
        "exact_match": bool(all(
            refit[s]["max_abs_diff_vs_stored_predictions"] is not None
            and refit[s]["max_abs_diff_vs_stored_predictions"] <= 1e-9 for s in refit
        )),
        "note": "the historical XGBoost model is NOT retrained for reporting; this refit only proves EXACT_MATCH identity",
    }
    print(f"[phaseD] historical refit: valid={refit['valid']['roc_auc']:.6f} test={refit['test']['roc_auc']:.6f} "
          f"exact_match={hist_identity['exact_match']}", flush=True)

    # ---- the one new model: historical features + C + F
    design = LIN.load_design()
    blocks = [design["C"], design["F"]]
    width = Xk.shape[1] + design["C"].shape[1] + design["F"].shape[1]
    n_train = splits["train"].size
    Xtr = np.zeros((n_train, width), dtype=np.float32)
    Xtr[:, : Xk.shape[1]] = Xk[splits["train"]]
    Xtr[:, Xk.shape[1]:] = _dense_block(blocks, splits["train"], width - Xk.shape[1])
    ytr = y[splits["train"]]
    print(f"[phaseD] XGB_STRUCT design {Xtr.shape}", flush=True)

    t_fit = time.perf_counter()
    model = fit_xgb(Xtr, ytr, params)
    del Xtr
    fit_seconds = time.perf_counter() - t_fit

    per_split: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for split_name in ("train", "valid", "test"):
        rows = splits[split_name]
        Xs = np.zeros((rows.size, width), dtype=np.float32)
        Xs[:, : Xk.shape[1]] = Xk[rows]
        Xs[:, Xk.shape[1]:] = _dense_block(blocks, rows, width - Xk.shape[1])
        logit = model.predict(Xs, output_margin=True)
        del Xs
        per_split[split_name] = ST.auc_ap(y[rows], logit)
        per_split[split_name]["ogb_check"] = LIN.ogb_check(logit, y[rows])
        if split_name == "test":
            per_split[split_name]["evidence_label"] = "EXPOSED_DEVELOPMENT_TEST"
        preds[split_name] = logit

    # ---- paired comparisons against the historical per-molecule predictions
    scaffolds, _meta = D.load_scaffolds(molecules)
    uniq = {s: i for i, s in enumerate(sorted(set(scaffolds)))}
    group = np.asarray([uniq[s] for s in scaffolds], dtype=np.int64)
    paired: dict[str, Any] = {}
    for split_name in ("valid", "test"):
        rows = splits[split_name]
        stored = np.load(C.HIST_BASELINE_DIR / f"ogbg-molhiv_{split_name}_predictions.npz")
        if not np.array_equal(np.asarray(stored["rows"]), rows):
            raise SystemExit(f"historical {split_name} prediction rows do not match the official split")
        hist_logit = np.asarray(stored["pred"]).reshape(-1).astype(np.float64)
        boot = ST.paired_scaffold_bootstrap(
            y[rows], preds[split_name], hist_logit, group[rows],
            label_a="XGB_STRUCT", label_b="historical_XGB",
        )
        paired[split_name] = {
            "historical": ST.auc_ap(y[rows], hist_logit),
            "xgb_struct": per_split[split_name],
            "bootstrap": boot,
            "evidence_label": "EXPOSED_DEVELOPMENT_TEST" if split_name == "test" else "DEVELOPMENT_VALID",
        }

    # ---- which structure columns the trees actually used
    import xgboost as xgb

    booster = model.get_booster()
    gain = booster.get_score(importance_type="total_gain")
    gains = {k: float(v) for k, v in gain.items()}
    total_gain = sum(gains.values())
    hist_gain = sum(v for k, v in gains.items() if int(k[1:]) < Xk.shape[1])
    struct_gain = total_gain - hist_gain
    n_struct_cols = width - Xk.shape[1]
    used = {
        "total_gain": total_gain,
        "historical_block_gain": hist_gain,
        "structural_block_gain": struct_gain,
        "structural_block_gain_share": struct_gain / total_gain if total_gain else None,
        "n_columns_historical": int(Xk.shape[1]),
        "n_columns_structural": int(n_struct_cols),
        "n_used_features_total": len(gains),
        "n_used_features_structural": sum(1 for k in gains if int(k[1:]) >= Xk.shape[1]),
    }
    top_struct = sorted(
        ((float(v), int(k[1:]) - Xk.shape[1]) for k, v in gains.items() if int(k[1:]) >= Xk.shape[1]),
        reverse=True,
    )[:40]
    used["top_structural_features"] = [
        {"column": int(col), "block": "C" if col < design["C"].shape[1] else "F",
         "block_column": int(col if col < design["C"].shape[1] else col - design["C"].shape[1]),
         "total_gain": float(v)} for v, col in top_struct
    ]

    payload = {
        "round_id": C.ROUND_ID,
        "phase": "D: strong traditional baseline increment",
        "created_utc": created,
        "historical_identity": {
            "source_round": "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2",
            "manifest": C.rel(C.HIST_BASELINE_MANIFEST),
            "manifest_sha256": C.sha256_file(C.HIST_BASELINE_MANIFEST),
            "evidence_type": "EXACT_MATCH" if hist_identity["exact_match"] else "REFERENCE_UNVERIFIED",
            **hist_checks,
        },
        "historical_refit_verification": hist_identity,
        "xgb_struct": {
            "features": "historical 2258-column ECFP4+RDKit block (unchanged) + C + F",
            "n_columns": int(width),
            "recipe": params,
            "dense": True,
            "sparse_missing_policy": (
                "no sparse input: dense float32 with missing=NaN; after Train-only median "
                "imputation no cell is NaN, so every explicit zero stays the value 0. "
                "Verified that xgboost's sparse DMatrix would treat un-stored zeros as missing"
            ),
            "fit_seconds": float(fit_seconds),
            "metrics": per_split,
            "feature_usage": used,
        },
        "paired": paired,
        "seconds": float(time.perf_counter() - t0),
        "git": C.git_state(),
        "environment": C.environment(),
    }
    C.write_json(C.ANALYSIS / "phaseD_xgb.json", payload)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        C.CACHE / "phaseD_xgb_struct_predictions.npz",
        valid=preds["valid"], test=preds["test"], train=preds["train"],
        valid_rows=splits["valid"], test_rows=splits["test"], train_rows=splits["train"], y=y,
    )
    print(f"[phaseD] XGB_STRUCT valid={per_split['valid']['roc_auc']:.4f} test={per_split['test']['roc_auc']:.4f} "
          f"(historical valid={refit['valid']['roc_auc']:.4f} test={refit['test']['roc_auc']:.4f})", flush=True)
    print(f"[phaseD] paired valid Δ={paired['valid']['bootstrap']['point']['delta_roc_auc']:+.4f} "
          f"[{paired['valid']['bootstrap']['delta_roc_auc']['p2.5']:+.4f},"
          f"{paired['valid']['bootstrap']['delta_roc_auc']['p97.5']:+.4f}] | "
          f"test Δ={paired['test']['bootstrap']['point']['delta_roc_auc']:+.4f} "
          f"[{paired['test']['bootstrap']['delta_roc_auc']['p2.5']:+.4f},"
          f"{paired['test']['bootstrap']['delta_roc_auc']['p97.5']:+.4f}]", flush=True)
    print(f"[phaseD] done in {time.perf_counter() - t0:.1f}s", flush=True)


def manifest_rows_auc(split_name: str) -> float:
    from sklearn.metrics import roc_auc_score

    stored = np.load(C.HIST_BASELINE_DIR / f"ogbg-molhiv_{split_name}_predictions.npz")
    return float(roc_auc_score(np.asarray(stored["y_true"]).reshape(-1), np.asarray(stored["pred"]).reshape(-1)))


if __name__ == "__main__":
    main()
