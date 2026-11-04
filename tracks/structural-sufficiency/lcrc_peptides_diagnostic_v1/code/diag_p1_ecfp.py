"""P1 — fixed Count-ECFP4 + LightGBM baseline (Train -> Validation only).

Runs in an isolated CPU environment (``scikit-fingerprints`` 2.1.0 + ``lightgbm``
4.7.0) that does not touch the repository's pinned GPU environment:

    /home/calendar/.cache/research-venvs/peptides-ecfp/bin/python \
        tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/code/diag_p1_ecfp.py

Frozen choices (no search of any kind): Morgan radius 2, 2048 bits, count
variant, RDKit default atom invariants; per-label LGBMClassifier(n_estimators=500,
is_unbalance=True, importance_type="gain", random_state=0, n_jobs=8, verbose=-1).
Test labels are neither read nor scored.
"""

from __future__ import annotations

import csv
import time
from typing import Any

import numpy as np

import diag_common as C

FINGERPRINT_KWARGS = {"count": True, "radius": 2, "fp_size": 2048}
LGBM_KWARGS = {
    "n_estimators": 500,
    "is_unbalance": True,
    "importance_type": "gain",
    "random_state": 0,
    "n_jobs": 8,
    "verbose": -1,
}


def versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for name in (
        "scikit-fingerprints",
        "lightgbm",
        "rdkit",
        "scikit-learn",
        "numpy",
        "scipy",
        "pandas",
    ):
        try:
            out[name] = md.version(name)
        except Exception:  # noqa: BLE001
            out[name] = "UNKNOWN"
    return out


def write_csv(path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    gate_path = C.ANALYSIS / "row_identity.json"
    if not gate_path.exists():
        raise SystemExit("ROW_ALIGNMENT_BLOCKED: analysis/row_identity.json missing")
    gate = C.read_json(gate_path)
    if gate.get("status") != "PASS":
        raise SystemExit(f"ROW_ALIGNMENT_BLOCKED: gate status {gate.get('status')}")

    from rdkit import Chem, RDLogger
    from skfp.datasets.lrgb import load_lrgb_mol_splits, load_peptides_func
    from skfp.fingerprints import ECFPFingerprint

    RDLogger.DisableLog("rdApp.*")

    t_start = time.perf_counter()
    smiles, label_matrix = load_peptides_func(data_dir=str(C.SKFP_DATA), mol_type="SMILES")
    train_idx, valid_idx, _test_idx = load_lrgb_mol_splits(
        "Peptides-func", data_dir=str(C.SKFP_DATA)
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    label_matrix = np.asarray(label_matrix, dtype=np.int64)

    checks: dict[str, Any] = {
        "n_smiles": len(smiles),
        "n_label_matrix": int(label_matrix.shape[0]),
        "n_train": int(train_idx.size),
        "n_valid": int(valid_idx.size),
        "sizes_ok": (len(smiles), train_idx.size, valid_idx.size) == (15535, 10873, 2331),
        "labels_10_columns": label_matrix.shape[1] == C.N_LABELS,
        "labels_binary": bool(np.all((label_matrix == 0) | (label_matrix == 1))),
        "train_valid_disjoint": bool(np.intersect1d(train_idx, valid_idx).size == 0),
    }
    if not all(checks[k] for k in ("sizes_ok", "labels_10_columns", "labels_binary", "train_valid_disjoint")):
        raise SystemExit(f"ROW_ALIGNMENT_BLOCKED: {checks}")

    # The split position must reproduce the official label rows exactly (already
    # proven atom-by-atom in diag_row_identity.py; this is the label-only echo).
    y_train = label_matrix[train_idx]
    y_valid = label_matrix[valid_idx]
    checks["train_labels_equal_fast_v1"] = bool(
        np.array_equal(y_train, np.load(C.PREV_ANALYSIS / "peptides_func_y_train.npy").astype(np.int64))
    )
    checks["valid_labels_equal_fast_v1"] = bool(
        np.array_equal(y_valid, np.load(C.PREV_ANALYSIS / "peptides_func_y_valid.npy").astype(np.int64))
    )
    if not (checks["train_labels_equal_fast_v1"] and checks["valid_labels_equal_fast_v1"]):
        raise SystemExit("ROW_ALIGNMENT_BLOCKED: label rows do not match the Fast v1 arrays")

    mols_train = [Chem.MolFromSmiles(smiles[i]) for i in train_idx]
    mols_valid = [Chem.MolFromSmiles(smiles[i]) for i in valid_idx]
    if any(m is None for m in mols_train) or any(m is None for m in mols_valid):
        raise SystemExit("fingerprint input: unparsable SMILES")

    fp = ECFPFingerprint(**FINGERPRINT_KWARGS)
    t_fp = time.perf_counter()
    X_train = np.asarray(fp.transform(mols_train), dtype=np.float32)
    X_valid = np.asarray(fp.transform(mols_valid), dtype=np.float32)
    fp_seconds = time.perf_counter() - t_fp

    # determinism / purity checks on a fixed subset
    subset = mols_train[:200]
    X_sub_a = np.asarray(fp.transform(subset), dtype=np.float32)
    X_sub_b = np.asarray(fp.transform(subset), dtype=np.float32)
    fp_purity = {
        "recompute_identical": bool(np.array_equal(X_sub_a, X_sub_b)),
        "subset_matches_full_rows": bool(np.array_equal(X_sub_a, X_train[:200])),
        "finite": bool(np.isfinite(X_train).all() and np.isfinite(X_valid).all()),
        "non_negative_integer": bool(
            (X_train >= 0).all() and (X_valid >= 0).all() and np.array_equal(X_train, np.round(X_train))
        ),
    }
    if not all(fp_purity.values()):
        raise SystemExit(f"fingerprint purity gate failed: {fp_purity}")

    feature_meta = {
        "shape_train": list(X_train.shape),
        "shape_valid": list(X_valid.shape),
        "dtype": str(X_train.dtype),
        "nnz_train": int((X_train > 0).sum()),
        "nnz_valid": int((X_valid > 0).sum()),
        "max_count_train": float(X_train.max()),
        "max_count_valid": float(X_valid.max()),
        "empty_feature_rows_train": int((X_train.sum(axis=1) == 0).sum()),
        "empty_feature_rows_valid": int((X_valid.sum(axis=1) == 0).sum()),
        "expected_shape": [10873, 2048],
        "shape_ok": list(X_train.shape) == [10873, 2048] and list(X_valid.shape) == [2331, 2048],
    }
    if not feature_meta["shape_ok"]:
        raise SystemExit(f"fingerprint shape gate failed: {feature_meta}")

    from sklearn.metrics import average_precision_score
    from lightgbm import LGBMClassifier

    per_label: list[dict[str, Any]] = []
    probs = np.zeros((X_valid.shape[0], C.N_LABELS), dtype=np.float64)
    importances = np.zeros((C.N_LABELS, X_train.shape[1]), dtype=np.float64)
    label_times: list[float] = []
    t_lgbm = time.perf_counter()
    for j, name in enumerate(C.LABELS):
        t0 = time.perf_counter()
        clf = LGBMClassifier(**LGBM_KWARGS)
        clf.fit(X_train, y_train[:, j])
        p_valid = clf.predict_proba(X_valid)[:, 1]
        elapsed = time.perf_counter() - t0
        label_times.append(elapsed)
        if p_valid.shape != (X_valid.shape[0],):
            raise RuntimeError(f"{name}: probability shape {p_valid.shape}")
        if not (np.isfinite(p_valid).all() and (p_valid >= 0).all() and (p_valid <= 1).all()):
            raise RuntimeError(f"{name}: probabilities outside [0,1] or non-finite")
        probs[:, j] = p_valid
        ap = float(average_precision_score(y_valid[:, j], p_valid))
        ap_recheck = float(average_precision_score(y_valid[:, j].astype(np.int64), probs[:, j]))
        if abs(ap - ap_recheck) > 1e-15:
            raise RuntimeError(f"{name}: AP recomputation mismatch")
        importances[j] = np.asarray(clf.feature_importances_, dtype=np.float64)
        tr_pos = int(y_train[:, j].sum())
        va_pos = int(y_valid[:, j].sum())
        va_rate = va_pos / y_valid.shape[0]
        per_label.append(
            {
                "label_index": j,
                "label": name,
                "n_train": int(X_train.shape[0]),
                "train_pos": tr_pos,
                "train_neg": int(X_train.shape[0]) - tr_pos,
                "train_pos_rate": tr_pos / X_train.shape[0],
                "n_valid": int(X_valid.shape[0]),
                "valid_pos": va_pos,
                "valid_neg": int(X_valid.shape[0]) - va_pos,
                "valid_pos_rate": va_rate,
                "valid_ap": ap,
                "valid_ap_pct": 100.0 * ap,
                "valid_ap_minus_valid_pos_rate": ap - va_rate,
                "valid_constant_baseline_ap": va_rate,
                "prob_min": float(p_valid.min()),
                "prob_max": float(p_valid.max()),
                "prob_mean": float(p_valid.mean()),
                "gain_importance_top1_pct": 100.0
                * float(importances[j].max() / max(importances[j].sum(), 1e-12)),
                "fit_seconds": elapsed,
            }
        )
        C.log(f"{name:<26} valid AP {100*ap:6.2f}%  (baseline {100*va_rate:5.2f}%)  fit {elapsed:6.1f}s")
    lgbm_seconds = time.perf_counter() - t_lgbm

    aps = np.asarray([row["valid_ap"] for row in per_label], dtype=np.float64)
    macro = float(aps.mean())
    macro_recheck = float(
        np.mean([average_precision_score(y_valid[:, j], probs[:, j]) for j in range(C.N_LABELS)])
    )
    if abs(macro - macro_recheck) > 1e-12:
        raise RuntimeError("macro AP recomputation mismatch")

    baseline_macro = float(np.mean([row["valid_constant_baseline_ap"] for row in per_label]))
    total_seconds = time.perf_counter() - t_start

    summary = {
        "dataset": C.DATASET,
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "baseline": "Count ECFP4 (radius 2, 2048 bits) + LightGBM, one model per label",
        "training_split": "official Train only",
        "evaluation_split": "official Validation only",
        "n_train": int(X_train.shape[0]),
        "n_valid": int(X_valid.shape[0]),
        "n_labels": C.N_LABELS,
        "valid_macro_ap": macro,
        "valid_macro_ap_pct": 100.0 * macro,
        "valid_macro_ap_recheck": macro_recheck,
        "valid_macro_constant_baseline_ap": baseline_macro,
        "valid_macro_constant_baseline_ap_pct": 100.0 * baseline_macro,
        "valid_macro_ap_minus_baseline_pct": 100.0 * (macro - baseline_macro),
        "macro_ap_recheck_delta": abs(macro - macro_recheck),
        "fingerprint_seconds": fp_seconds,
        "lgbm_seconds": lgbm_seconds,
        "total_seconds": total_seconds,
        "test_rows_read": 0,
        "test_metrics_computed": False,
    }
    write_csv(C.RESULTS / "ecfp_validation.csv", [summary])
    write_csv(C.RESULTS / "ecfp_per_label.csv", per_label)
    write_csv(C.RESULTS / "ecfp_gain_importance.csv", [
        {"feature_index": f, "label": C.LABELS[j], "label_index": j, "gain": float(importances[j, f])}
        for j in range(C.N_LABELS)
        for f in np.argsort(importances[j])[::-1][:20]
    ])
    np.savez_compressed(
        C.RESULTS / "ecfp_valid_predictions.npz",
        probs=probs,
        y_true=y_valid,
        valid_row_indices=valid_idx,
        labels=np.asarray(C.LABELS),
    )
    manifest = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "entry_point": "code/diag_p1_ecfp.py",
        "interpreter": {
            "path": "/home/calendar/.cache/research-venvs/peptides-ecfp/bin/python",
            "note": "isolated CPU venv; the repo's pinned GPU environment is not modified",
            "versions": versions(),
        },
        "data": {
            "smiles_loader": "skfp.datasets.lrgb.load_peptides_func(mol_type='SMILES')",
            "split_loader": "skfp.datasets.lrgb.load_lrgb_mol_splits('Peptides-func')",
            "local_cache": C.track_rel(C.SKFP_DATA),
            "smiles_csv_sha256": C.sha256_file(C.PEPTIDES_CSV),
            "splits_json_sha256": C.sha256_file(C.PEPTIDES_SPLITS),
            "train_index_sha256": C.sha256_text(",".join(str(i) for i in train_idx)),
            "valid_index_sha256": C.sha256_text(",".join(str(i) for i in valid_idx)),
            "train_index_first_last": [int(train_idx[0]), int(train_idx[-1])],
            "valid_index_first_last": [int(valid_idx[0]), int(valid_idx[-1])],
            "split_sizes": {"train": int(train_idx.size), "valid": int(valid_idx.size)},
        },
        "checks": checks,
        "fingerprint": {
            "implementation": "skfp.fingerprints.ECFPFingerprint",
            "kwargs": FINGERPRINT_KWARGS,
            "atom_invariants": "RDKit defaults (no pharmacophore invariants, no explicit chirality flag)",
            "extra_descriptors": "none",
            "purity": fp_purity,
            "features": feature_meta,
        },
        "classifier": {
            "implementation": "lightgbm.LGBMClassifier",
            "kwargs": LGBM_KWARGS,
            "other_parameters": "LightGBM defaults",
            "tuning": "none",
            "tree_count_search": "none",
        },
        "metrics": {
            "per_label": "sklearn.metrics.average_precision_score (positive class probability)",
            "macro": "unweighted mean over the 10 labels",
            "macro_ap": macro,
            "macro_ap_recheck_delta": abs(macro - macro_recheck),
            "constant_baseline_macro_ap": baseline_macro,
        },
        "timing_seconds": {
            "load_and_fingerprint": fp_seconds,
            "lightgbm_total": lgbm_seconds,
            "total": total_seconds,
            "per_label": label_times,
        },
        "artifacts": {
            "summary": "results/ecfp_validation.csv",
            "per_label": "results/ecfp_per_label.csv",
            "gain_importance": "results/ecfp_gain_importance.csv",
            "valid_predictions": "results/ecfp_valid_predictions.npz",
        },
        "environment": C.environment(),
        "test_rows_read": 0,
    }
    C.write_json(C.ANALYSIS / "ecfp_manifest.json", manifest)
    C.log(f"ECFP4+LightGBM Validation macro AP {100*macro:.2f}% (constant baseline {100*baseline_macro:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
