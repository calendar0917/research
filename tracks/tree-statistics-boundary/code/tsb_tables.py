"""Report tables A-D for ``tsb-lipo-feature-ladder-v1``.

Pure post-processing of recorded fit/test artefacts: no model is fitted and no
feature is rebuilt here.  Every number in ``REPORT.md`` comes from one of these
tables, and each table records the artefact hashes it was derived from.
"""

from __future__ import annotations

import csv
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import tsb_common as C
import tsb_data as D
import tsb_runner as R


def _fits() -> dict[str, dict[str, Any]]:
    return {
        path.stem: C.read_json(path)
        for path in sorted(R.FITS_DIR.glob("*.json"))
    }


def _test_records() -> dict[str, dict[str, Any]]:
    if not R.TEST_DIR.is_dir():
        return {}
    return {
        path.stem: C.read_json(path)
        for path in sorted(R.TEST_DIR.glob("*.json"))
        if path.stem != "test_summary"
    }


def _write_csv(path, rows: Sequence[Mapping[str, Any]]) -> None:
    C.ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# ---------------------------------------------------------------------------
# table A
# ---------------------------------------------------------------------------


def table_a(fits: Mapping[str, Mapping[str, Any]], tests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for suffix in C.SPLIT_SUFFIXES:
        for arm in C.ARMS:
            key = R.fit_key(suffix, arm, C.FULL_SIZE)
            fit = fits.get(key)
            if fit is None:
                continue
            test = tests.get(key)
            rows.append(
                {
                    "split": f"Lipop.random{suffix}",
                    "arm": arm,
                    "n_train": fit["n_train"],
                    "n_columns": fit["n_columns"],
                    "train_rmse": fit["metrics"]["train"]["rmse"],
                    "valid_rmse": fit["metrics"]["valid"]["rmse"],
                    "test_rmse": test["metrics"]["test"]["rmse"] if test else None,
                    "valid_mae": fit["metrics"]["valid"]["mae"],
                    "test_mae": test["metrics"]["test"]["mae"] if test else None,
                    "valid_r2": fit["metrics"]["valid"]["r2"],
                    "test_r2": test["metrics"]["test"]["r2"] if test else None,
                    "train_valid_gap_rmse": fit["generalisation_gap_rmse"],
                    "fit_seconds": fit["fit_seconds"],
                    "feature_assembly_seconds": fit["feature_assembly_seconds"],
                    "model_sha256": fit["model"]["sha256"],
                }
            )
    summary: list[dict[str, Any]] = []
    for arm in C.ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        entry: dict[str, Any] = {"arm": arm, "n_splits": len(arm_rows)}
        for metric in ("train_rmse", "valid_rmse", "test_rmse", "valid_mae",
                       "test_mae", "train_valid_gap_rmse", "fit_seconds",
                       "feature_assembly_seconds"):
            values = [row[metric] for row in arm_rows if row[metric] is not None]
            if values:
                entry[f"{metric}_mean"] = float(np.mean(values))
                entry[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(entry)
    note = (
        "the three repeats share the same 4,200 molecules (only the random "
        "partition changes), so the mean/std across repeats is NOT a mean/std "
        "across three independent external datasets"
    )
    C.write_json(R.TABLES_DIR / "table_a.json",
                 {"schema": "tsb-table-a-v1", "rows": rows, "summary": summary, "note": note})
    _write_csv(R.TABLES_DIR / "table_a.csv", rows)
    _write_csv(R.TABLES_DIR / "table_a_summary.csv", summary)
    return {"rows": rows, "summary": summary, "note": note}


# ---------------------------------------------------------------------------
# table B
# ---------------------------------------------------------------------------


def table_b(fits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _label, left, right in C.CONTRASTS:
        deltas: list[float] = []
        for suffix in C.SPLIT_SUFFIXES:
            key_left = R.fit_key(suffix, left, C.FULL_SIZE)
            key_right = R.fit_key(suffix, right, C.FULL_SIZE)
            if key_left not in fits or key_right not in fits:
                continue
            split = D.split(suffix)
            labels = D.split_labels(suffix)
            pred_left = _load_pred(fits[key_left]["predictions"]["valid"]["path"])
            pred_right = _load_pred(fits[key_right]["predictions"]["valid"]["path"])
            delta = C.paired_bootstrap_delta(
                pred_left["pred"], pred_right["pred"], labels["valid"]
            )
            deltas.append(delta["delta_rmse"])
            rows.append(
                {
                    "contrast": left + " - " + right,
                    "left": left,
                    "right": right,
                    "split": f"Lipop.random{suffix}",
                    "valid_rmse_left": float(np.sqrt(np.mean((pred_left["pred"] - labels["valid"]) ** 2))),
                    "valid_rmse_right": float(np.sqrt(np.mean((pred_right["pred"] - labels["valid"]) ** 2))),
                    "delta_rmse": delta["delta_rmse"],
                    "ci95_low": delta["ci95"][0],
                    "ci95_high": delta["ci95"][1],
                    "bootstrap_replicates": delta["replicates"],
                    "bootstrap_seed": delta["seed"],
                }
            )
        if deltas:
            same_sign = all(d < 0 for d in deltas) or all(d > 0 for d in deltas)
            for row in rows:
                if row["left"] == left and row["right"] == right:
                    row["direction_consistent_across_3_splits"] = bool(same_sign)
            rows.append(
                {
                    "contrast": left + " - " + right,
                    "left": left,
                    "right": right,
                    "split": "ALL",
                    "delta_rmse": float(np.mean(deltas)),
                    "delta_rmse_std_across_splits": float(np.std(deltas, ddof=0)),
                    "direction_consistent_across_3_splits": bool(same_sign),
                }
            )
    note = (
        "delta = RMSE(left) - RMSE(right) on the Validation rows; lower RMSE is "
        "better, so a NEGATIVE delta means the left arm is better. The bootstrap "
        "interval is a molecule-sampling interval for one frozen seed and is not "
        "an estimate of training-seed variance."
    )
    C.write_json(R.TABLES_DIR / "table_b.json",
                 {"schema": "tsb-table-b-v1", "rows": rows, "note": note})
    _write_csv(R.TABLES_DIR / "table_b.csv", rows)
    return {"rows": rows, "note": note}


def _load_pred(rel_path: str) -> dict[str, np.ndarray]:
    with np.load(C.REPO / rel_path, allow_pickle=False) as handle:
        return {
            "rows": np.asarray(handle["rows"], dtype=np.int64),
            "pred": np.asarray(handle["pred"], dtype=np.float64),
            "y_true": np.asarray(handle["y_true"], dtype=np.float64),
        }


# ---------------------------------------------------------------------------
# table C
# ---------------------------------------------------------------------------


def table_c(fits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for arm in C.ARMS:
        for suffix in C.SPLIT_SUFFIXES:
            for n in C.SAMPLE_SIZES:
                fit = fits.get(R.fit_key(suffix, arm, n))
                if fit is None:
                    continue
                rows.append(
                    {
                        "arm": arm,
                        "split": f"Lipop.random{suffix}",
                        "n_train": n,
                        "valid_rmse": fit["metrics"]["valid"]["rmse"],
                        "valid_mae": fit["metrics"]["valid"]["mae"],
                        "train_rmse": fit["metrics"]["train"]["rmse"],
                        "train_valid_gap_rmse": fit["generalisation_gap_rmse"],
                        "fit_seconds": fit["fit_seconds"],
                    }
                )
    curves: dict[str, Any] = {}
    for arm in C.ARMS:
        per_split = {}
        for suffix in C.SPLIT_SUFFIXES:
            series = [
                {"n_train": n, "valid_rmse": fits[R.fit_key(suffix, arm, n)]["metrics"]["valid"]["rmse"]}
                for n in C.SAMPLE_SIZES
                if R.fit_key(suffix, arm, n) in fits
            ]
            per_split[str(suffix)] = series
        curves[arm] = {
            "per_split": per_split,
            "mean_across_splits": [
                {
                    "n_train": n,
                    "valid_rmse_mean": float(
                        np.mean([
                            fits[R.fit_key(suffix, arm, n)]["metrics"]["valid"]["rmse"]
                            for suffix in C.SPLIT_SUFFIXES
                            if R.fit_key(suffix, arm, n) in fits
                        ])
                    ),
                }
                for n in C.SAMPLE_SIZES
            ],
        }
    note = (
        "smaller sizes are nested prefixes of the frozen order "
        "sha256('20261104:<original row id>'); the typed-WL vocabularies always "
        "come from the full official Train, so this curve is a labelled-sample-size "
        "curve, not a 'only N graphs exist' curve"
    )
    C.write_json(R.TABLES_DIR / "table_c.json",
                 {"schema": "tsb-table-c-v1", "rows": rows, "curves": curves, "note": note})
    _write_csv(R.TABLES_DIR / "table_c.csv", rows)
    return {"rows": rows, "curves": curves, "note": note}


# ---------------------------------------------------------------------------
# table D
# ---------------------------------------------------------------------------


def _role_coverage(csr, rows: np.ndarray) -> dict[str, float]:
    """OOV and column usage of one round block restricted to a row subset.

    Derived from the frozen cached CSR only: the last column of a typed-WL
    block *is* the OOV bucket, so the statistics are exact and do not need a
    second tokenisation pass.
    """
    block = csr[np.asarray(rows, dtype=np.int64)]
    total = float(block.sum())
    oov = float(block[:, -1].sum())
    rows_with_oov = int(np.count_nonzero(block[:, -1].toarray().reshape(-1)))
    used = np.zeros(block.shape[1], dtype=bool)
    used[np.unique(block.indices)] = True
    return {
        "n_rows": int(block.shape[0]),
        "token_occurrences": int(total),
        "oov_occurrences": int(oov),
        "oov_fraction": float(oov / total) if total else 0.0,
        "rows_with_oov": rows_with_oov,
        "rows_with_oov_fraction": float(rows_with_oov / block.shape[0]),
        "vocab_tokens_used": int(used[:-1].sum()),
        "vocab_coverage_fraction": float(used[:-1].sum() / (block.shape[1] - 1)),
        "nonzero_column_fraction": float(used.sum() / block.shape[1]),
        "density": float(block.nnz / (block.shape[0] * block.shape[1])),
    }


def table_d(feature_manifest: Mapping[str, Any], fits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    import tsb_features as F

    rows: list[dict[str, Any]] = []
    for suffix in C.SPLIT_SUFFIXES:
        split = D.split(suffix)
        for round_index in range(max(C.WL_ROUNDS["WL3"]) + 1):
            entry = feature_manifest["wl"][str(suffix)][f"r{round_index}"]
            block = F.load_block_counts(suffix, round_index, binary=False)
            for role in ("train", "valid", "test"):
                stats = _role_coverage(block, split["rows"][role])
                rows.append(
                    {
                        "split": f"Lipop.random{suffix}",
                        "round": int(round_index),
                        "role": role,
                        "n_rows": stats["n_rows"],
                        "n_columns": int(block.shape[1]),
                        "token_occurrences": stats["token_occurrences"],
                        "oov_occurrences": stats["oov_occurrences"],
                        "oov_fraction": stats["oov_fraction"],
                        "rows_with_oov_fraction": stats["rows_with_oov_fraction"],
                        "vocab_tokens_used": stats["vocab_tokens_used"],
                        "vocab_coverage_fraction": stats["vocab_coverage_fraction"],
                        "nonzero_column_fraction": stats["nonzero_column_fraction"],
                        "density": stats["density"],
                        "vocab_sha256": entry["vocab_sha256"],
                    }
                )
    cost: list[dict[str, Any]] = []
    for suffix in C.SPLIT_SUFFIXES:
        for arm in C.ARMS:
            fit = fits.get(R.fit_key(suffix, arm, C.FULL_SIZE))
            if fit is None:
                continue
            cost.append(
                {
                    "split": f"Lipop.random{suffix}",
                    "arm": arm,
                    "n_columns": fit["n_columns"],
                    "feature_assembly_seconds": fit["feature_assembly_seconds"],
                    "fit_seconds": fit["fit_seconds"],
                    "model_bytes": fit["model"]["size_bytes"],
                }
            )
    timing = feature_manifest["timing"]
    global_cost = {
        "tokenization_seconds_one_time": timing["tokenization_seconds"],
        "total_feature_stage_seconds": timing["total_feature_stage_seconds"],
        "g0_build_seconds": feature_manifest["g0"]["seconds"],
        "morgan_build_seconds": feature_manifest["morgan"]["seconds"],
        "morgan_mean_raw_environments": feature_manifest["morgan"]["mean_raw_environments"],
        "morgan_mean_buckets_used": feature_manifest["morgan"]["mean_buckets_used"],
        "morgan_mean_collision_buckets": feature_manifest["morgan"]["mean_collision_buckets"],
        "morgan_rows_with_collision": feature_manifest["morgan"]["rows_with_collision"],
        "peak_rss_mb_feature_stage": feature_manifest["peak_rss_mb"],
    }
    note = (
        "coverage is computed per role directly from the frozen cached CSR blocks "
        "(the last column of each typed-WL block is the OOV bucket); the `coverage` "
        "block inside cache/features_manifest.json is the all-4,200-molecule aggregate "
        "and is NOT the per-role number. feature_assembly_seconds is the time to "
        "assemble one arm/split matrix from cached label-free blocks (hstack + dtype); "
        "the one-time tokenisation cost is reported separately because all three "
        "splits share it."
    )
    C.write_json(
        R.TABLES_DIR / "table_d.json",
        {"schema": "tsb-table-d-v1", "coverage": rows, "cost": cost,
         "global_cost": global_cost, "note": note},
    )
    _write_csv(R.TABLES_DIR / "table_d_coverage.csv", rows)
    _write_csv(R.TABLES_DIR / "table_d_cost.csv", cost)
    return {"coverage": rows, "cost": cost, "global_cost": global_cost, "note": note}


# ---------------------------------------------------------------------------
# external, read-only reference
# ---------------------------------------------------------------------------


def external_reference() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    try:
        payload = C.read_json(C.REPO / C.EXTERNAL_REFERENCE["sources"][122])
        test = C.read_json(
            C.REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/baseline/Lipophilicity_test.json"
        )
        entries["122"] = {
            "valid_rmse": payload["metrics"]["valid"]["rmse"],
            "valid_mae": payload["metrics"]["valid"]["mae"],
            "test_rmse": test["metrics"]["test"]["rmse"],
            "test_mae": test["metrics"]["test"]["mae"],
            "train_rmse": None,
            "source": C.EXTERNAL_REFERENCE["sources"][122],
        }
        for suffix in (123, 124):
            fit = C.read_json(C.REPO / C.EXTERNAL_REFERENCE["sources"][suffix])
            test = C.read_json(
                C.REPO / C.EXTERNAL_REFERENCE["sources"][suffix].replace(".json", "_test.json")
            )
            entries[str(suffix)] = {
                "valid_rmse": fit["valid"]["rmse"],
                "valid_mae": fit["valid"]["mae"],
                "test_rmse": test["test"]["rmse"],
                "test_mae": test["test"]["mae"],
                "train_rmse": fit["train"]["rmse"],
                "source": C.EXTERNAL_REFERENCE["sources"][suffix],
            }
    except Exception as exc:  # pragma: no cover - reference files are read-only inputs
        return {"label": C.EXTERNAL_REFERENCE["label"], "error": repr(exc)}
    valid = [entry["valid_rmse"] for entry in entries.values()]
    test = [entry["test_rmse"] for entry in entries.values()]
    payload = {
        "label": C.EXTERNAL_REFERENCE["label"],
        "note": (
            "READ-ONLY external reference: ECFP4 *binary* + 217 RDKit descriptors + "
            "XGBoost with that round's own Train-only imputation. Different input block "
            "and different preprocessing; NOT an arm of this protocol and never used for "
            "selection here."
        ),
        "per_split": entries,
        "valid_rmse_mean": float(np.mean(valid)),
        "valid_rmse_std": float(np.std(valid, ddof=0)),
        "test_rmse_mean": float(np.mean(test)),
        "test_rmse_std": float(np.std(test, ddof=0)),
    }
    C.write_json(R.TABLES_DIR / "external_reference.json", payload)
    return payload


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_all() -> dict[str, Any]:
    C.ensure_dir(R.TABLES_DIR)
    fits = _fits()
    tests = _test_records()
    a = table_a(fits, tests)
    b = table_b(fits)
    c = table_c(fits)
    d = table_d(C.read_json(R.FITS_DIR.parent / "p0" / "features_manifest.json")
                if (R.FITS_DIR.parent / "p0" / "features_manifest.json").is_file()
                else _load_feature_manifest(), fits)
    reference = external_reference()
    payload = {
        "schema": "tsb-tables-v1",
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "tables": ["table_a", "table_b", "table_c", "table_d", "external_reference"],
        "n_fits": len(fits),
        "test_available": bool(tests),
        "n_test_models": len(tests),
        "external_reference_label": reference.get("label"),
        "table_a_note": a["note"],
        "table_b_note": b["note"],
        "table_c_note": c["note"],
        "table_d_note": d["note"],
    }
    C.write_json(R.TABLES_DIR / "tables_manifest.json", payload)
    return payload


def _load_feature_manifest() -> dict[str, Any]:
    import tsb_features as F

    return F.load_manifest()
