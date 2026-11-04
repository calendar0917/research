"""P0: reproduce the historical Count-ECFP4 + LightGBM baseline.

Run inside the isolated CPU fingerprint environment (scikit-fingerprints 2.1.0,
lightgbm 4.7.0, rdkit 2026.3.5, scikit-learn 1.9.1).  The historical ECFP
baseline is SMILES-based: the official CSV contains isotope/explicit-H
information that the 9-column PyG graph does not carry.  The streaming loader
below parses only the SMILES field of the official Train/Validation split rows;
it never parses a label column and never retains or evaluates Test rows.

No Test labels, Test graphs or Test predictions are read.
"""

from __future__ import annotations

import json
import time

import numpy as np

import pc_common as C

ECFP_FORBIDDEN = tuple(
    name
    for name in C.TEST_FORBIDDEN
    if name not in ("LRGB_Peptides-func", "lrgb_splits_peptides_func.json")
)
# The mixed official CSV is allowed only through the row-filtered loader below.
GUARD = C.install_test_access_guard(forbidden=ECFP_FORBIDDEN)

FINGERPRINT_KWARGS = {"count": True, "radius": 2, "fp_size": 2048}
PEPTIDES_CSV = C.DATA / "lrgb_skfp/LRGB_Peptides-func/peptides_func.csv"
PEPTIDES_SPLITS = C.DATA / "lrgb_skfp/LRGB_Peptides-func/lrgb_splits_peptides_func.json"
HISTORICAL_VALID_PREDICTIONS = (
    C.REPO / "tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/results/ecfp_valid_predictions.npz"
)
LGBM_KWARGS = {
    "n_estimators": 500,
    "is_unbalance": True,
    "importance_type": "gain",
    "random_state": 0,
    "n_jobs": 8,
    "verbose": -1,
}
HISTORICAL_MACRO_AP_PCT = 72.98
HISTORICAL_ACCEPT_PP = 0.01


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


def load_split_indices() -> tuple[list[int], list[int]]:
    with open(PEPTIDES_SPLITS, "r", encoding="utf-8") as fh:
        split = json.load(fh)
    train_idx = [int(i) for i in split["train"]]
    valid_idx = [int(i) for i in split["valid"]]
    with np.load(HISTORICAL_VALID_PREDICTIONS) as hist:
        hist_valid = [int(i) for i in hist["valid_row_indices"]]
    if valid_idx != hist_valid:
        raise SystemExit("validation row indices differ from the historical ECFP result")
    if len(train_idx) != C.SPLIT_SIZES["train"] or len(valid_idx) != C.SPLIT_SIZES["valid"]:
        raise SystemExit("split size mismatch")
    return train_idx, valid_idx


def load_smiles_rows(indices: list[int]) -> list[str]:
    """Parse only the SMILES field of the requested CSV rows.

    The official CSV is a mixed Train/Validation/Test file.  For a requested
    row we take the substring before the first comma (the CSV SMILES column is
    first and SMILES contains no comma).  No label column is ever parsed.  Rows
    not requested are discarded without parsing.
    """
    wanted = set(int(i) for i in indices)
    out: dict[int, str] = {}
    with open(PEPTIDES_CSV, "r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header.startswith("SMILES,"):
            raise SystemExit("unexpected CSV header")
        for i, line in enumerate(fh):
            if i in wanted:
                out[i] = line.split(",", 1)[0]
    missing = wanted - set(out)
    if missing:
        raise SystemExit(f"missing SMILES rows: {len(missing)}")
    return [out[int(i)] for i in indices]


def main() -> int:
    from rdkit import Chem, RDLogger
    from skfp.fingerprints import ECFPFingerprint
    from sklearn.metrics import average_precision_score
    from lightgbm import LGBMClassifier

    RDLogger.DisableLog("rdApp.*")
    t_start = time.perf_counter()
    train_idx, valid_idx = load_split_indices()
    y_train = np.load(C.PREV_ANALYSIS / "peptides_func_y_train.npy").astype(np.int64)
    y_valid = np.load(C.PREV_ANALYSIS / "peptides_func_y_valid.npy").astype(np.int64)
    if y_train.shape != (len(train_idx), C.N_LABELS) or y_valid.shape != (len(valid_idx), C.N_LABELS):
        raise SystemExit("label shape gate failed")

    t_read = time.perf_counter()
    smiles_train = load_smiles_rows(train_idx)
    smiles_valid = load_smiles_rows(valid_idx)
    read_seconds = time.perf_counter() - t_read

    t_mol = time.perf_counter()
    mols_train = [Chem.MolFromSmiles(s) for s in smiles_train]
    mols_valid = [Chem.MolFromSmiles(s) for s in smiles_valid]
    if any(m is None for m in mols_train) or any(m is None for m in mols_valid):
        raise SystemExit("unparsable SMILES")
    mol_seconds = time.perf_counter() - t_mol

    fp = ECFPFingerprint(**FINGERPRINT_KWARGS)
    t_fp = time.perf_counter()
    X_train = np.asarray(fp.transform(mols_train), dtype=np.float32)
    X_valid = np.asarray(fp.transform(mols_valid), dtype=np.float32)
    fp_seconds = time.perf_counter() - t_fp

    subset = mols_train[:200]
    X_sub_a = np.asarray(fp.transform(subset), dtype=np.float32)
    X_sub_b = np.asarray(fp.transform(subset), dtype=np.float32)
    purity = {
        "recompute_identical": bool(np.array_equal(X_sub_a, X_sub_b)),
        "subset_matches_full_rows": bool(np.array_equal(X_sub_a, X_train[:200])),
        "finite": bool(np.isfinite(X_train).all() and np.isfinite(X_valid).all()),
        "non_negative_integer": bool(
            (X_train >= 0).all()
            and (X_valid >= 0).all()
            and np.array_equal(X_train, np.round(X_train))
            and np.array_equal(X_valid, np.round(X_valid))
        ),
    }
    if not all(purity.values()):
        raise SystemExit(f"fingerprint purity gate failed: {purity}")
    if list(X_train.shape) != [10873, 2048] or list(X_valid.shape) != [2331, 2048]:
        raise SystemExit(f"fingerprint shape gate failed: {X_train.shape} / {X_valid.shape}")

    per_label: list[dict] = []
    probs = np.zeros((X_valid.shape[0], C.N_LABELS), dtype=np.float64)
    importances = np.zeros((C.N_LABELS, X_train.shape[1]), dtype=np.float64)
    fit_seconds: list[float] = []
    for j, name in enumerate(C.LABELS):
        t0 = time.perf_counter()
        clf = LGBMClassifier(**LGBM_KWARGS)
        clf.fit(X_train, y_train[:, j])
        p = clf.predict_proba(X_valid)[:, 1]
        elapsed = time.perf_counter() - t0
        fit_seconds.append(elapsed)
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            raise SystemExit(f"{name}: invalid probabilities")
        probs[:, j] = p
        ap = float(average_precision_score(y_valid[:, j], p))
        importances[j] = np.asarray(clf.feature_importances_, dtype=np.float64)
        per_label.append(
            {
                "label_index": j,
                "label": name,
                "train_pos": int(y_train[:, j].sum()),
                "valid_pos": int(y_valid[:, j].sum()),
                "valid_pos_rate": float(y_valid[:, j].mean()),
                "valid_ap": ap,
                "valid_ap_pct": 100.0 * ap,
                "fit_seconds": elapsed,
            }
        )
        C.log(f"{name:<26} valid AP {100 * ap:6.2f}%")
    macro = float(np.mean([row["valid_ap"] for row in per_label]))
    macro_recheck = float(
        np.mean([average_precision_score(y_valid[:, j], probs[:, j]) for j in range(C.N_LABELS)])
    )
    if abs(macro - macro_recheck) > 1e-12:
        raise SystemExit("macro AP recheck mismatch")
    macro_pct = 100.0 * macro
    delta_vs_history = macro_pct - HISTORICAL_MACRO_AP_PCT
    reproduction_ok = abs(delta_vs_history) <= HISTORICAL_ACCEPT_PP

    summary = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "generated_utc": C.utcnow(),
        "entry_point": "code/pc_p0_ecfp.py",
        "interpreter": "/home/calendar/.cache/research-venvs/peptides-ecfp/bin/python",
        "versions": versions(),
        "n_train": int(X_train.shape[0]),
        "n_valid": int(X_valid.shape[0]),
        "n_labels": C.N_LABELS,
        "valid_macro_ap": macro,
        "valid_macro_ap_pct": macro_pct,
        "valid_macro_ap_recheck_delta": abs(macro - macro_recheck),
        "historical_macro_ap_pct": HISTORICAL_MACRO_AP_PCT,
        "delta_vs_historical_pp": delta_vs_history,
        "historical_accept_pp": HISTORICAL_ACCEPT_PP,
        "reproduction_ok": bool(reproduction_ok),
        "per_label": per_label,
        "fingerprint": {
            "implementation": "skfp.fingerprints.ECFPFingerprint",
            "kwargs": FINGERPRINT_KWARGS,
            "input": "official CSV SMILES for the Train / Validation split rows only",
            "row_loader": "streaming CSV reader; SMILES field only; no label column parsed",
            "purity": purity,
            "shape_train": list(X_train.shape),
            "shape_valid": list(X_valid.shape),
            "nnz_train": int((X_train > 0).sum()),
            "nnz_valid": int((X_valid > 0).sum()),
            "max_count_train": float(X_train.max()),
            "max_count_valid": float(X_valid.max()),
            "empty_rows_train": int((X_train.sum(axis=1) == 0).sum()),
            "empty_rows_valid": int((X_valid.sum(axis=1) == 0).sum()),
            "labels_used_as_features": False,
            "raw_graph_rebuild_note": (
                "the 9-column PyG graph omits SMILES isotope/explicit-H information; "
                "ECFP must use the historical SMILES input to reproduce 72.98%"
            ),
        },
        "mixed_csv_access": {
            "path": C.track_rel(PEPTIDES_CSV),
            "sha256_at_start": C.sha256_file(PEPTIDES_CSV),
            "split_json": C.track_rel(PEPTIDES_SPLITS),
            "split_json_sha256_at_start": C.sha256_file(PEPTIDES_SPLITS),
            "rows_requested": int(len(train_idx) + len(valid_idx)),
            "train_rows_requested": int(len(train_idx)),
            "valid_rows_requested": int(len(valid_idx)),
            "label_columns_parsed": False,
            "test_rows_parsed": 0,
            "test_rows_retained": 0,
            "test_labels_read": False,
        },
        "classifier": {
            "implementation": "lightgbm.LGBMClassifier",
            "kwargs": LGBM_KWARGS,
            "other_parameters": "LightGBM defaults",
            "tuning": "none",
            "tree_count_search": "none",
        },
        "timing_seconds": {
            "csv_smiles_read": float(read_seconds),
            "mol_parse": float(mol_seconds),
            "fingerprints": float(fp_seconds),
            "lightgbm_total": float(sum(fit_seconds)),
            "lightgbm_per_label": fit_seconds,
            "total": float(time.perf_counter() - t_start),
        },
        "test_access": {"attempts": list(GUARD["attempts"]), "strict": GUARD["strict"]},
        "test_rows_read": 0,
        "environment": C.environment(),
    }
    C.write_json(C.ANALYSIS / "ecfp_validation.json", summary)
    np.savez_compressed(
        C.ANALYSIS / "ecfp_valid_predictions.npz",
        probs=probs,
        y_true=y_valid,
        labels=np.asarray(C.LABELS),
    )
    C.log(
        f"ECFP4+LightGBM Validation macro AP {macro_pct:.4f}% "
        f"(historical {HISTORICAL_MACRO_AP_PCT:.2f}%, delta {delta_vs_history:+.4f} pp)"
    )
    if not reproduction_ok:
        C.log("ECFP_REPRODUCTION_BLOCKED: do not enter the GPU stage")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
