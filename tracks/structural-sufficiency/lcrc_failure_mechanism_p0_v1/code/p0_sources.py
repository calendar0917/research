"""Read-only loaders, strict alignment and integrity audit for the P0 round.

Two evaluation groups are kept strictly separate:

``A_random42``
    Lipophilicity random42 split (``torch random_split`` seed 42), frozen
    Validation420 predictions of LCRC-A-R3 and PNA-native.  Test is not touched.

``B_lipo_random122``
    Lipophilicity random122 "published" split from Sprint 02, frozen
    Validation420/Test420 predictions of BASE / CHEM8 / MULTICONTRIB /
    COMBO from ``lcrc_performance_longrun_v1``.  Test is decoded only for the
    explicitly labelled ``POST_HOC_EXPOSED_DISCOVERY`` part.

Nothing here retrains, re-infers or selects a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from p0_common import (
    A_ROOT,
    B_ROOT,
    CHEMBENCH_V1,
    CHEMBENCH_V2_ROOT,
    TOL,
    read_json,
    relpath,
    row_sha,
    sha256_file,
    sha256_float64,
    sha256_text,
)
from p0_stats import metrics


class AlignmentError(RuntimeError):
    """Raised when predictions are not aligned to the frozen row order."""


# ---------------------------------------------------------------------------
# frozen input paths
# ---------------------------------------------------------------------------

A_SPLIT = A_ROOT / "data/split_esa_random42_v1.json"
A_LABELS = A_ROOT / "data/labels_train_valid_v1.json"
A_METRICS = A_ROOT / "analysis/metrics_validation.json"
A_RUNS: dict[str, str] = {
    "lcrc_a_r3_s0": "LCRC-A-R3-scratch",
    "pna_native_s0": "PNA-native",
}

B_CSV = CHEMBENCH_V1 / "Lipop/Lipop.csv.gz"
B_SPLIT = CHEMBENCH_V1 / "Lipop/Lipop.random122.ind.pkl"
B_SPLIT_MANIFEST = CHEMBENCH_V2_ROOT / "analysis/split_manifest_Lipophilicity.json"
B_DATASET_MANIFEST = CHEMBENCH_V2_ROOT / "analysis/dataset_manifest.json"
B_RUNS: dict[tuple[str, int], str] = {
    ("base", 0): "p0_base_lipo_parity",
    ("chem8", 0): "p1_chem8_lipo",
    ("multicontrib", 0): "p1_contrib_lipo",
    ("combo_chem_contrib", 0): "p2b_combo_chem_contrib_lipo",
    ("base", 1): "p5_base_lipo_s1",
    ("combo_chem_contrib", 1): "p5b_combo_lipo_s1",
    ("base", 2): "p5_base_lipo_s2",
    ("combo_chem_contrib", 2): "p5b_combo_lipo_s2",
}
B_ARMS = ("base", "chem8", "multicontrib", "combo_chem_contrib")
B_SEEDS = (0, 1, 2)
B_TABLES = (
    B_ROOT / "analysis/p1_table.csv",
    B_ROOT / "analysis/p2b_table.csv",
    B_ROOT / "analysis/p5_table.csv",
    B_ROOT / "analysis/p5b_table.csv",
)

#: runs that exist but are excluded from the four-arm comparison and why
B_EXCLUDED = {
    "p1_rho_lipo": "rhoshape is outside the pre-declared four-arm BASE/CHEM8/MULTICONTRIB/COMBO set",
    "p2_combo_rho_contrib_lipo": "combo_rho_contrib is outside the pre-declared four-arm set",
    "p3_base_scalem_lipo": "scale-M variant outside the pre-declared four-arm set",
    "p3_rho_scalem_lipo": "rho + scale-M variant outside the pre-declared four-arm set",
    "p4_rho_lipo123": "random123: no matching BASE per-molecule prediction file",
    "p4_rho_lipo124": "random124: no matching BASE per-molecule prediction file",
    "p4b_combo_lipo123": "random123: COMBO exists but no matching BASE per-molecule prediction file",
    "p4b_combo_lipo124": "random124: COMBO exists but no matching BASE per-molecule prediction file",
}


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    key: str
    arm: str
    seed: int
    rows: np.ndarray
    pred: np.ndarray
    y: np.ndarray
    exposure: str
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class Group:
    key: str
    dataset: str
    split: str
    split_seed: int | None
    rows: dict[str, np.ndarray]
    smiles: list[str]
    labels: dict[str, np.ndarray]
    predictions: dict[str, Prediction]
    audit: dict[str, Any]


# ---------------------------------------------------------------------------
# alignment helpers
# ---------------------------------------------------------------------------


def strict_align(
    rows: Sequence[int],
    values: Sequence[float],
    target_rows: Sequence[int],
    *,
    label: str,
) -> np.ndarray:
    """Require byte-identical row order; used by the integrity audit."""
    rows_arr = np.asarray(rows, dtype=np.int64).reshape(-1)
    target = np.asarray(target_rows, dtype=np.int64).reshape(-1)
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if rows_arr.shape != target.shape:
        raise AlignmentError(f"{label}: {len(rows_arr)} rows vs {len(target)} target rows")
    if not np.array_equal(rows_arr, target):
        first = int(np.flatnonzero(rows_arr != target)[0]) if np.any(rows_arr != target) else -1
        raise AlignmentError(f"{label}: row order differs from the frozen split (first at {first})")
    if values_arr.shape != rows_arr.shape:
        raise AlignmentError(f"{label}: values/prediction length mismatch")
    return values_arr


def align_by_row_id(
    rows: Sequence[int],
    values: Sequence[float],
    target_rows: Sequence[int],
    *,
    label: str,
) -> np.ndarray:
    """Re-index by row id, so input row order cannot change an alignment statistic.

    Duplicate row ids, missing rows or extra rows are rejected.  The output is
    always in ``target_rows`` order.
    """
    rows_arr = np.asarray(rows, dtype=np.int64).reshape(-1)
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    target = np.asarray(target_rows, dtype=np.int64).reshape(-1)
    if rows_arr.shape != values_arr.shape:
        raise AlignmentError(f"{label}: rows/values length mismatch")
    if len(np.unique(rows_arr)) != len(rows_arr):
        raise AlignmentError(f"{label}: duplicate row ids")
    if len(np.unique(target)) != len(target):
        raise AlignmentError(f"{label}: duplicate target row ids")
    if set(rows_arr.tolist()) != set(target.tolist()):
        missing = len(set(target.tolist()) - set(rows_arr.tolist()))
        extra = len(set(rows_arr.tolist()) - set(target.tolist()))
        raise AlignmentError(f"{label}: row set differs (missing {missing}, extra {extra})")
    lookup = {int(r): float(v) for r, v in zip(rows_arr.tolist(), values_arr.tolist())}
    return np.asarray([lookup[int(r)] for r in target.tolist()], dtype=np.float64)


def _metric_check(
    name: str, reported: float, recomputed: float, *, tol: float = TOL
) -> dict[str, Any]:
    diff = abs(float(reported) - float(recomputed))
    return {
        "metric": name,
        "reported": float(reported),
        "recomputed": float(recomputed),
        "abs_diff": diff,
        "tolerance": float(tol),
        "passed": bool(diff <= tol),
    }


def _require(checks: list[dict[str, Any]], *, context: str) -> None:
    failed = [c for c in checks if not c.get("passed", False)]
    if failed:
        raise RuntimeError(f"{context}: blocking integrity failure(s): {failed}")


# ---------------------------------------------------------------------------
# group A
# ---------------------------------------------------------------------------


def load_group_a() -> Group:
    """Load random42 Validation420 predictions, labels and split metadata."""
    checks: list[dict[str, Any]] = []
    split_file_sha = sha256_file(A_SPLIT)
    payload = read_json(A_SPLIT)

    checks.append(
        {
            "check": "split_n_molecules",
            "passed": int(payload["n_molecules"]) == 4200,
            "detail": int(payload["n_molecules"]),
        }
    )
    checks.append(
        {
            "check": "split_smiles_sha256",
            "passed": sha256_text(payload["smiles"]) == payload["smiles_sha256"],
            "detail": payload["smiles_sha256"],
        }
    )

    rows: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        arr = np.asarray(payload["rows"][split], dtype=np.int64)
        rows[split] = arr
        checks.append(
            {
                "check": f"split_rows_sha256_{split}",
                "passed": row_sha(arr) == payload["sha256"][split],
                "detail": row_sha(arr),
            }
        )
        checks.append(
            {
                "check": f"split_rows_unique_{split}",
                "passed": len(np.unique(arr)) == len(arr),
                "detail": {"n": int(len(arr)), "n_unique": int(len(np.unique(arr)))},
            }
        )
        checks.append(
            {
                "check": f"split_rows_in_range_{split}",
                "passed": bool(arr.min() >= 0 and arr.max() < 4200),
                "detail": [int(arr.min()), int(arr.max())],
            }
        )

    overlap = {
        "train_valid": int(len(set(rows["train"]) & set(rows["valid"]))),
        "train_test": int(len(set(rows["train"]) & set(rows["test"]))),
        "valid_test": int(len(set(rows["valid"]) & set(rows["test"]))),
        "union": int(len(set(rows["train"]) | set(rows["valid"]) | set(rows["test"]))),
    }
    checks.append(
        {
            "check": "split_row_isolation",
            "passed": all(v == 0 for k, v in overlap.items() if k != "union") and overlap["union"] == 4200,
            "detail": overlap,
        }
    )
    checks.append(
        {
            "check": "split_declared_sizes",
            "passed": payload["sizes"] == {"train": 3360, "valid": 420, "test": 420},
            "detail": payload["sizes"],
        }
    )
    checks.append(
        {
            "check": "split_method_random42",
            "passed": "manual_seed(42)" in payload["method"] and payload["seed"] == 42,
            "detail": payload["method"],
        }
    )

    labels_payload = read_json(A_LABELS)
    labels_sha = sha256_file(A_LABELS)
    checks.append(
        {
            "check": "labels_scope_train_valid_only",
            "passed": labels_payload["scope"] == "train_and_valid_only"
            and labels_payload["test_labels_included"] is False,
            "detail": {
                "scope": labels_payload["scope"],
                "test_labels_included": labels_payload["test_labels_included"],
            },
        }
    )
    checks.append(
        {
            "check": "labels_split_file_sha256",
            "passed": labels_payload["split_file_sha256"] == split_file_sha,
            "detail": labels_payload["split_file_sha256"],
        }
    )
    labels: dict[str, np.ndarray] = {}
    for split in ("train", "valid"):
        arr = np.asarray(labels_payload["rows"][split], dtype=np.int64)
        values = np.asarray(labels_payload["values"][split], dtype=np.float64)
        labels[split] = values
        checks.append(
            {
                "check": f"labels_rows_match_split_{split}",
                "passed": np.array_equal(arr, rows[split]),
                "detail": {"n": int(len(arr))},
            }
        )
        checks.append(
            {
                "check": f"labels_row_sha256_{split}",
                "passed": row_sha(arr) == labels_payload["row_sha256_int64"][split],
                "detail": labels_payload["row_sha256_int64"][split],
            }
        )
        checks.append(
            {
                "check": f"labels_vector_sha256_{split}",
                "passed": sha256_float64(values) == labels_payload["vector_sha256"][split],
                "detail": labels_payload["vector_sha256"][split],
            }
        )

    metric_doc = read_json(A_METRICS)
    reported = {entry["model"]: entry for entry in metric_doc["results"]}
    predictions: dict[str, Prediction] = {}
    model_audit: list[dict[str, Any]] = []
    for run, model_name in A_RUNS.items():
        path = A_ROOT / "analysis" / "runs" / run / "valid_predictions.npz"
        payload_npz = np.load(path, allow_pickle=False)
        rows_npz = np.asarray(payload_npz["rows"], dtype=np.int64)
        pred = strict_align(rows_npz, payload_npz["pred"], rows["valid"], label=f"A/{run}")
        y = labels["valid"]
        checks.append(
            {
                "check": f"predictions_finite_{run}",
                "passed": bool(np.isfinite(pred).all()),
                "detail": int(np.sum(~np.isfinite(pred))),
            }
        )
        rec = metrics(y, pred)
        reported_entry = reported[model_name]
        reported_metrics = reported_entry["valid_metrics"]
        mchecks = [
            _metric_check("rmse", reported_metrics["rmse"], rec["rmse"]),
            _metric_check("mae", reported_metrics["mae"], rec["mae"]),
            _metric_check(
                "sse",
                float(reported_metrics.get("n", len(y))) * float(reported_metrics["rmse"]) ** 2,
                rec["sse"],
            ),
        ]
        checks.extend({**m, "check": f"{run}_{m['metric']}"} for m in mchecks)
        checks.append(
            {
                "check": f"prediction_file_sha256_{run}",
                "passed": sha256_file(path) == reported_entry["prediction_sha256"],
                "detail": sha256_file(path),
            }
        )
        checks.append(
            {
                "check": f"prediction_rows_sha256_{run}",
                "passed": row_sha(rows_npz) == reported_entry["valid_rows_sha256"],
                "detail": row_sha(rows_npz),
            }
        )
        model_audit.append(
            {
                "run": run,
                "model": model_name,
                "prediction_file": relpath(path),
                "prediction_sha256": sha256_file(path),
                "rows_sha256": row_sha(rows_npz),
                "n": int(len(rows_npz)),
                "strict_row_order_match": True,
                "recomputed": rec,
                "reported_valid_metrics": reported_metrics,
                "metric_checks": mchecks,
                "checkpoint_rule": reported_entry.get("checkpoint_rule"),
                "best_epoch": reported_entry.get("best_epoch"),
                "parameters_total": reported_entry.get("parameters_total"),
                "seed": reported_entry.get("seed"),
            }
        )
        predictions[run] = Prediction(
            key=run,
            arm=run,
            seed=0,
            rows=rows_npz,
            pred=pred,
            y=y,
            exposure="FROZEN_VALIDATION_ONLY",
            source={"path": relpath(path), "sha256": sha256_file(path)},
        )

    checks.append(
        {
            "check": "group_a_valid_row_sets_identical",
            "passed": len({tuple(p.rows.tolist()) for p in predictions.values()}) == 1,
            "detail": {k: len(v.rows) for k, v in predictions.items()},
        }
    )
    _require(checks, context="group A")

    audit = {
        "group": "A_random42",
        "dataset": "Lipophilicity",
        "split": "random42",
        "split_seed": 42,
        "split_file": {
            "path": relpath(A_SPLIT),
            "sha256": split_file_sha,
            "n_molecules": 4200,
            "smiles_sha256": payload["smiles_sha256"],
            "row_sha256": {k: payload["sha256"][k] for k in ("train", "valid", "test")},
            "sizes": payload["sizes"],
            "method": payload["method"],
        },
        "metric_file": {
            "path": relpath(A_METRICS),
            "sha256": sha256_file(A_METRICS),
            "definition": "frozen valid RMSE/MAE for LCRC-A-R3 and PNA-native; SSE derived as n*RMSE^2 for the reproduction check",
        },
        "labels": {
            "path": relpath(A_LABELS),
            "sha256": labels_sha,
            "scope": labels_payload["scope"],
            "vector_sha256": labels_payload["vector_sha256"],
        },
        "models": model_audit,
        "row_overlap": overlap,
        "checks": checks,
    }
    return Group(
        key="A_random42",
        dataset="Lipophilicity",
        split="random42",
        split_seed=42,
        rows={k: np.asarray(v, dtype=np.int64) for k, v in rows.items()},
        smiles=list(payload["smiles"]),
        labels=labels,
        predictions=predictions,
        audit=audit,
    )


# ---------------------------------------------------------------------------
# group B
# ---------------------------------------------------------------------------


def _read_pickle_split(path: Path) -> list[list[int]]:
    series = pd.read_pickle(path)
    parts = [list(part) for part in series.to_list()]
    if len(parts) != 3:
        raise ValueError(f"{path}: expected 3 parts, got {len(parts)}")
    return [[int(x) for x in part] for part in parts]


def load_group_b() -> Group:
    """Load Lipophilicity random122 BASE/CHEM8/MULTICONTRIB/COMBO predictions."""
    checks: list[dict[str, Any]] = []
    dataset_manifest = read_json(B_DATASET_MANIFEST)["datasets"]["Lipophilicity"]

    split_file_sha = sha256_file(B_SPLIT)
    csv_sha = sha256_file(B_CSV)
    checks.append(
        {
            "check": "b_split_file_sha256",
            "passed": split_file_sha == dataset_manifest["split_sha256"],
            "detail": split_file_sha,
        }
    )
    checks.append(
        {
            "check": "b_csv_sha256",
            "passed": csv_sha == dataset_manifest["csv_sha256"],
            "detail": csv_sha,
        }
    )

    parts = _read_pickle_split(B_SPLIT)
    rows = {name: np.asarray(part, dtype=np.int64) for name, part in zip(("train", "valid", "test"), parts)}
    expected_rows = dataset_manifest["rows_sha256"]
    for split in ("train", "valid", "test"):
        checks.append(
            {
                "check": f"b_split_rows_sha256_{split}",
                "passed": row_sha(rows[split]) == expected_rows[split],
                "detail": row_sha(rows[split]),
            }
        )
        checks.append(
            {
                "check": f"b_split_rows_unique_{split}",
                "passed": len(np.unique(rows[split])) == len(rows[split]),
                "detail": int(len(rows[split])),
            }
        )
    overlap = {
        "train_valid": int(len(set(rows["train"]) & set(rows["valid"]))),
        "train_test": int(len(set(rows["train"]) & set(rows["test"]))),
        "valid_test": int(len(set(rows["valid"]) & set(rows["test"]))),
        "union": int(len(set(rows["train"]) | set(rows["valid"]) | set(rows["test"]))),
    }
    checks.append(
        {
            "check": "b_split_row_isolation",
            "passed": all(v == 0 for k, v in overlap.items() if k != "union") and overlap["union"] == 4200,
            "detail": overlap,
        }
    )
    checks.append(
        {
            "check": "b_split_sizes",
            "passed": [len(rows[k]) for k in ("train", "valid", "test")] == [3360, 420, 420],
            "detail": {k: int(len(rows[k])) for k in ("train", "valid", "test")},
        }
    )

    csv_df = pd.read_csv(B_CSV)
    checks.append(
        {
            "check": "b_csv_columns_and_rows",
            "passed": list(csv_df.columns) == ["smiles", "exp"] and len(csv_df) == 4200,
            "detail": {"columns": list(csv_df.columns), "n": int(len(csv_df))},
        }
    )
    smiles = [str(s) for s in csv_df["smiles"].tolist()]
    csv_labels = np.asarray(csv_df["exp"].to_numpy(), dtype=np.float64)
    labels = {
        "train": csv_labels[rows["train"]],
        "valid": csv_labels[rows["valid"]],
        "test": csv_labels[rows["test"]],
    }

    predictions: dict[str, Prediction] = {}
    run_audit: list[dict[str, Any]] = []
    target_norms: list[dict[str, Any]] = []
    for (arm, seed), run in B_RUNS.items():
        run_dir = B_ROOT / "analysis" / "runs" / run
        train_json = read_json(run_dir / "train.json")
        train_sha = sha256_file(run_dir / "train.json")
        checks.append(
            {
                "check": f"b_run_identity_{run}",
                "passed": train_json["dataset"] == "Lipophilicity"
                and train_json["split"] == "published"
                and train_json["arm"] == arm
                and int(train_json["seed"]) == int(seed),
                "detail": {
                    "dataset": train_json["dataset"],
                    "split": train_json["split"],
                    "arm": train_json["arm"],
                    "seed": int(train_json["seed"]),
                },
            }
        )
        norm = train_json["data"]["target_norm"]
        target_norms.append(
            {"run": run, "mean": float(norm["mean"]), "scale": float(norm["scale"]), "n": int(norm["n"])}
        )
        checks.append(
            {
                "check": f"b_target_norm_train_only_{run}",
                "passed": norm["source"] == "train_split_only" and norm["valid_or_test_used"] is False,
                "detail": {"source": norm["source"], "valid_or_test_used": norm["valid_or_test_used"]},
            }
        )

        valid_npz = np.load(run_dir / "valid_predictions.npz", allow_pickle=False)
        test_npz = np.load(run_dir / "test_predictions.npz", allow_pickle=False)
        valid_rows = np.asarray(valid_npz["rows"], dtype=np.int64)
        test_rows = np.asarray(test_npz["rows"], dtype=np.int64)
        raw_valid = strict_align(valid_rows, valid_npz["raw"], rows["valid"], label=f"B/{run}/valid")
        raw_test = strict_align(test_rows, test_npz["raw"], rows["test"], label=f"B/{run}/test")
        y_valid_npz = np.asarray(valid_npz["y_true"], dtype=np.float64).reshape(-1)
        y_test_npz = np.asarray(test_npz["y_true"], dtype=np.float64).reshape(-1)
        pred_valid = raw_valid * float(norm["scale"]) + float(norm["mean"])
        pred_test_given = np.asarray(test_npz["pred"], dtype=np.float64).reshape(-1)
        pred_test = raw_test * float(norm["scale"]) + float(norm["mean"])

        checks.append(
            {
                "check": f"b_valid_labels_match_source_{run}",
                "passed": bool(np.array_equal(y_valid_npz, labels["valid"])),
                "detail": {"max_abs_diff": float(np.max(np.abs(y_valid_npz - labels["valid"])))},
            }
        )
        checks.append(
            {
                "check": f"b_test_labels_match_source_{run}",
                "passed": bool(np.array_equal(y_test_npz, labels["test"])),
                "detail": {"max_abs_diff": float(np.max(np.abs(y_test_npz - labels["test"])))},
            }
        )
        checks.append(
            {
                "check": f"b_test_pred_reconstruction_{run}",
                "passed": float(np.max(np.abs(pred_test_given - pred_test))) <= 1e-12,
                "detail": float(np.max(np.abs(pred_test_given - pred_test))),
            }
        )
        checks.append(
            {
                "check": f"b_predictions_finite_{run}",
                "passed": bool(np.isfinite(pred_valid).all() and np.isfinite(pred_test).all()),
                "detail": {
                    "n_nonfinite_valid": int(np.sum(~np.isfinite(pred_valid))),
                    "n_nonfinite_test": int(np.sum(~np.isfinite(pred_test))),
                },
            }
        )
        rec_valid = metrics(labels["valid"], pred_valid)
        rec_test = metrics(labels["test"], pred_test)
        mchecks = [
            _metric_check("valid_rmse", train_json["metrics"]["valid"]["rmse"], rec_valid["rmse"]),
            _metric_check("valid_mae", train_json["metrics"]["valid"]["mae"], rec_valid["mae"]),
            _metric_check(
                "valid_sse",
                float(train_json["metrics"]["valid"].get("n", len(labels["valid"])))
                * float(train_json["metrics"]["valid"]["rmse"]) ** 2,
                rec_valid["sse"],
            ),
            _metric_check("test_rmse", train_json["metrics"]["test"]["rmse"], rec_test["rmse"]),
            _metric_check("test_mae", train_json["metrics"]["test"]["mae"], rec_test["mae"]),
            _metric_check(
                "test_sse",
                float(train_json["metrics"]["test"].get("n", len(labels["test"])))
                * float(train_json["metrics"]["test"]["rmse"]) ** 2,
                rec_test["sse"],
            ),
        ]
        checks.extend({**m, "check": f"{run}_{m['metric']}"} for m in mchecks)

        test_eval = read_json(run_dir / "test_eval.json")
        checks.append(
            {
                "check": f"b_test_rows_sha256_{run}",
                "passed": row_sha(test_rows) == test_eval["test_rows_sha256"],
                "detail": row_sha(test_rows),
            }
        )
        checks.append(
            {
                "check": f"b_valid_file_sha256_{run}",
                "passed": sha256_file(run_dir / "valid_predictions.npz")
                == train_json["predictions"]["valid_sha256"],
                "detail": sha256_file(run_dir / "valid_predictions.npz"),
            }
        )
        checks.append(
            {
                "check": f"b_test_file_sha256_{run}",
                "passed": sha256_file(run_dir / "test_predictions.npz") == test_eval["test_predictions_sha256"],
                "detail": sha256_file(run_dir / "test_predictions.npz"),
            }
        )

        run_audit.append(
            {
                "run": run,
                "arm": arm,
                "seed": int(seed),
                "train_json": relpath(run_dir / "train.json"),
                "train_json_sha256": train_sha,
                "valid_predictions": relpath(run_dir / "valid_predictions.npz"),
                "valid_predictions_sha256": sha256_file(run_dir / "valid_predictions.npz"),
                "test_predictions": relpath(run_dir / "test_predictions.npz"),
                "test_predictions_sha256": sha256_file(run_dir / "test_predictions.npz"),
                "rows_sha256": {"valid": row_sha(valid_rows), "test": row_sha(test_rows)},
                "target_norm": {"mean": float(norm["mean"]), "scale": float(norm["scale"]), "n": int(norm["n"])},
                "recomputed": {"valid": rec_valid, "test": rec_test},
                "reported": {
                    "valid": train_json["metrics"]["valid"],
                    "test": train_json["metrics"]["test"],
                },
                "metric_checks": mchecks,
                "checkpoint": {
                    "best_epoch": train_json["checkpoint"]["best_epoch"],
                    "best_metric_name": train_json["checkpoint"]["best_metric_name"],
                    "selection_uses_test": train_json["checkpoint"]["selection_uses_test"],
                    "epochs_run": train_json["checkpoint"]["epochs_run"],
                },
                "parameters_total": train_json["model"]["parameters"],
                "exposure": {
                    "valid": "FROZEN_VALIDATION_ONLY",
                    "test": "POST_HOC_EXPOSED_DISCOVERY",
                },
            }
        )
        predictions[f"{arm}_s{seed}"] = Prediction(
            key=f"{arm}_s{seed}",
            arm=arm,
            seed=int(seed),
            rows=valid_rows,
            pred=pred_valid,
            y=labels["valid"],
            exposure="FROZEN_VALIDATION_ONLY",
            source={"path": relpath(run_dir / "valid_predictions.npz")},
        )
        predictions[f"{arm}_s{seed}_test"] = Prediction(
            key=f"{arm}_s{seed}_test",
            arm=arm,
            seed=int(seed),
            rows=test_rows,
            pred=pred_test,
            y=labels["test"],
            exposure="POST_HOC_EXPOSED_DISCOVERY",
            source={"path": relpath(run_dir / "test_predictions.npz")},
        )

    checks.append(
        {
            "check": "b_target_norm_identical_across_runs",
            "passed": len({(round(t["mean"], 12), round(t["scale"], 12), t["n"]) for t in target_norms}) == 1,
            "detail": target_norms,
        }
    )
    for split in ("valid", "test"):
        is_test = split == "test"
        selected = [p for p in predictions.values() if p.key.endswith("_test") == is_test]
        sets = {tuple(p.rows.tolist()) for p in selected}
        checks.append(
            {
                "check": f"b_{split}_row_sets_identical_across_runs",
                "passed": len(sets) == 1,
                "detail": {"n_distinct": len(sets), "n_runs": len(selected)},
            }
        )

    # ---- cross-check against the historical stage tables (where a row exists)
    table_matches: list[dict[str, Any]] = []
    for table_path in B_TABLES:
        if not table_path.exists():
            continue
        table = pd.read_csv(table_path)
        if "dataset" not in table.columns:
            continue
        table = table[table["dataset"] == "Lipophilicity"]
        table = table.drop_duplicates(subset=["key"])
        for _, row in table.iterrows():
            key = str(row["key"])
            if key in B_RUNS.values():
                audit_entry = next(a for a in run_audit if a["run"] == key)
                for metric_name, column in (("valid", "valid"), ("test", "test")):
                    reported = float(row[column])
                    recomputed = audit_entry["recomputed"][metric_name]["rmse"]
                    table_matches.append(
                        {
                            "table": relpath(table_path),
                            "key": key,
                            "metric": f"{metric_name}_rmse",
                            "reported": reported,
                            "recomputed": recomputed,
                            "abs_diff": abs(reported - recomputed),
                            "passed": abs(reported - recomputed) <= TOL,
                        }
                    )
            ref_key = str(row.get("ref_key", ""))
            if ref_key in B_RUNS.values() and "ref_valid" in table.columns:
                audit_entry = next(a for a in run_audit if a["run"] == ref_key)
                for metric_name, column in (("valid", "ref_valid"), ("test", "ref_test")):
                    if column not in table.columns or pd.isna(row[column]):
                        continue
                    reported = float(row[column])
                    recomputed = audit_entry["recomputed"][metric_name]["rmse"]
                    table_matches.append(
                        {
                            "table": relpath(table_path),
                            "key": f"{ref_key} (reference)",
                            "metric": f"{metric_name}_rmse",
                            "reported": reported,
                            "recomputed": recomputed,
                            "abs_diff": abs(reported - recomputed),
                            "passed": abs(reported - recomputed) <= TOL,
                        }
                    )
    for match in table_matches:
        checks.append({**match, "check": f"table_{match['key']}_{match['metric']}"})

    _require(checks, context="group B")

    missing_cells = [
        {"arm": arm, "seed": seed, "reason": "no per-molecule prediction file in the frozen long-run"}
        for arm in B_ARMS
        for seed in B_SEEDS
        if (arm, seed) not in B_RUNS
    ]
    audit = {
        "group": "B_lipo_random122",
        "dataset": "Lipophilicity",
        "split": "random122(published)",
        "split_seed": None,
        "split_seed_note": (
            "The published split file is named random122; the frozen train.json does not "
            "record a split seed, so none is asserted here."
        ),
        "split_file": {
            "path": relpath(B_SPLIT),
            "sha256": split_file_sha,
            "row_sha256": {k: row_sha(rows[k]) for k in ("train", "valid", "test")},
            "sizes": {k: int(len(rows[k])) for k in ("train", "valid", "test")},
        },
        "labels_source": {
            "path": relpath(B_CSV),
            "sha256": csv_sha,
            "n": int(len(csv_df)),
            "test_labels_used_for": "POST_HOC_EXPOSED_DISCOVERY evaluation only",
        },
        "runs": run_audit,
        "missing_cells": missing_cells,
        "excluded_runs": [
            {"run": run, "reason": reason} for run, reason in B_EXCLUDED.items()
        ],
        "split_manifest": relpath(B_SPLIT_MANIFEST),
        "row_overlap": overlap,
        "table_cross_checks": table_matches,
        "checks": checks,
    }
    return Group(
        key="B_lipo_random122",
        dataset="Lipophilicity",
        split="random122(published)",
        split_seed=None,
        rows=rows,
        smiles=smiles,
        labels=labels,
        predictions=predictions,
        audit=audit,
    )


# ---------------------------------------------------------------------------
# group separation
# ---------------------------------------------------------------------------


def separation_record(a: Group, b: Group) -> dict[str, Any]:
    """Record that random42 and random122 are different evaluation sets."""
    a_valid = set(a.rows["valid"].tolist())
    b_valid = set(b.rows["valid"].tolist())
    a_test = set(a.rows["test"].tolist())
    b_test = set(b.rows["test"].tolist())
    return {
        "group_a": a.key,
        "group_b": b.key,
        "a_split": a.split,
        "b_split": b.split,
        "a_valid_rows_sha256": row_sha(a.rows["valid"]),
        "b_valid_rows_sha256": row_sha(b.rows["valid"]),
        "same_train_row_set": set(a.rows["train"].tolist()) == set(b.rows["train"].tolist()),
        "same_valid_row_set": a_valid == b_valid,
        "same_valid_row_order": bool(np.array_equal(a.rows["valid"], b.rows["valid"])),
        "valid_row_overlap_n": int(len(a_valid & b_valid)),
        "valid_row_overlap_frac_of_a": float(len(a_valid & b_valid) / len(a_valid)),
        "a_test_rows_sha256": row_sha(a.rows["test"]),
        "b_test_rows_sha256": row_sha(b.rows["test"]),
        "same_test_row_set": a_test == b_test,
        "treated_as_same_evaluation_set": False,
        "pooled_across_groups": False,
        "per_molecule_pairing_across_groups": False,
        "policy": (
            "random42 and random122 are different random partitions of the same 4,200-molecule "
            "Lipophilicity table; no per-molecule pairing or pooled metric is computed across them."
        ),
    }
