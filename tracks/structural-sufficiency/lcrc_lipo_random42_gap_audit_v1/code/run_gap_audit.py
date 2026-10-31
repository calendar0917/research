#!/usr/bin/env python3
"""Read-only audit of the frozen random42 Validation420 LCRC-vs-PNA gap.

This script only reads the frozen random42 split, the committed train/valid
labels, and the two historical Validation420 prediction files.  It never opens
Test labels, Test predictions, Test statistics, or historical per-molecule Test
tables.  It does not train, fine-tune, or select any model.

Outputs (under ``analysis/``):
    per_molecule_valid.csv
    subgroups.csv
    train_neighbor_support.json
    summary.json
    figures/01_gap_concentration.png
    figures/02_group_rmse.png
    figures/03_train_neighbor_support.png
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

TRACK = Path(__file__).resolve().parents[1]
REPO = TRACK.parents[2]
SOURCE = TRACK.parent / "lcrc_lipo_esa_random42_v1"
ANALYSIS = TRACK / "analysis"
FIGURES = ANALYSIS / "figures"

SPLIT_PATH = SOURCE / "data/split_esa_random42_v1.json"
LABELS_PATH = SOURCE / "data/labels_train_valid_v1.json"
METRICS_PATH = SOURCE / "analysis/metrics_validation.json"

MODELS = {
    "lcrc": {
        "name": "LCRC-A-R3-scratch",
        "run": "lcrc_a_r3_s0",
        "pred_path": SOURCE / "analysis/runs/lcrc_a_r3_s0/valid_predictions.npz",
        "expected_rmse": 0.5890072081930064,
        "expected_mae": 0.4208143415034744,
    },
    "pna": {
        "name": "PNA-native",
        "run": "pna_native_s0",
        "pred_path": SOURCE / "analysis/runs/pna_native_s0/valid_predictions.npz",
        "expected_rmse": 0.5540342318996931,
        "expected_mae": 0.4007184723844865,
    },
}

TOL = 1e-5
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
NEIGHBOR_SIM_THRESHOLD = 0.80
STABLE_N = 30


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_int64(values: Iterable[int]) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_float64(values: Iterable[float]) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_text(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def jsonable(value: Any) -> Any:
    """Recursively convert numpy/float values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = pred - target
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "sse": float(np.sum(error**2)),
    }


def dist_stats(values: Iterable[float], *, prefix: str = "") -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    quantiles = np.quantile(array, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    keys = ["p00", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100"]
    payload: dict[str, Any] = {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    for key, value in zip(keys, quantiles, strict=True):
        payload[f"{prefix}{key}"] = float(value)
    return payload


def histogram(values: Iterable[float], edges: Iterable[float]) -> list[dict[str, Any]]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    edge_array = np.asarray(list(edges), dtype=np.float64)
    counts, _ = np.histogram(array, bins=edge_array)
    payload = []
    for left, right, count in zip(edge_array[:-1], edge_array[1:], counts, strict=True):
        payload.append(
            {
                "bin_left": float(left) if math.isfinite(float(left)) else None,
                "bin_right": float(right) if math.isfinite(float(right)) else None,
                "count": int(count),
                "fraction": float(count / array.size) if array.size else None,
            }
        )
    return payload


def quantile_edges(values: np.ndarray) -> np.ndarray:
    return np.quantile(np.asarray(values, dtype=np.float64), [0.0, 0.25, 0.50, 0.75, 1.0])


def assign_quartiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign four quantile bins with right-closed inner boundaries.

    Inner boundaries are the sample 25/50/75 percentiles.  Values equal to a
    boundary are assigned to the lower bin (``searchsorted(..., side="left")``).
    This is deterministic and does not use any outcome information.
    """
    values = np.asarray(values, dtype=np.float64)
    edges = quantile_edges(values)
    inner = edges[1:-1]
    group = np.searchsorted(inner, values, side="left")
    group = np.clip(group, 0, 3).astype(np.int64)
    return group, edges


def quartile_labels() -> list[str]:
    return ["Q1_lowest", "Q2", "Q3", "Q4_highest"]


def scaffold_and_fallback(mol: Chem.Mol) -> tuple[str, bool]:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:  # RDKit can fail on unusual valence; fallback is defined.
        scaffold = None
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), True
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True), False


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    required_paths = [
        SPLIT_PATH,
        LABELS_PATH,
        METRICS_PATH,
        MODELS["lcrc"]["pred_path"],
        MODELS["pna"]["pred_path"],
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        stop(f"missing required frozen artifacts: {missing}")

    source_hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in required_paths}

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    labels_payload = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    metrics_payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            stop(f"failed check {name}: {detail}")

    # ------------------------------------------------------------------
    # A. Frozen data alignment and hash verification
    # ------------------------------------------------------------------
    check("split_size_train", split["sizes"]["train"] == 3360, str(split["sizes"]))
    check("split_size_valid", split["sizes"]["valid"] == 420, str(split["sizes"]))
    check(
        "split_file_hash_matches_labels_manifest",
        source_hashes[str(SPLIT_PATH.relative_to(REPO))] == labels_payload["split_file_sha256"],
        source_hashes[str(SPLIT_PATH.relative_to(REPO))],
    )

    train_rows = np.asarray(split["rows"]["train"], dtype=np.int64)
    valid_rows = np.asarray(split["rows"]["valid"], dtype=np.int64)
    valid_rows_sha = sha256_int64(valid_rows)
    train_rows_sha = sha256_int64(train_rows)

    check("valid_rows_length", valid_rows.size == 420, str(valid_rows.size))
    check("train_rows_length", train_rows.size == 3360, str(train_rows.size))
    check("valid_rows_unique", np.unique(valid_rows).size == 420, str(np.unique(valid_rows).size))
    check("train_rows_unique", np.unique(train_rows).size == 3360, str(np.unique(train_rows).size))
    check(
        "train_valid_disjoint",
        np.intersect1d(train_rows, valid_rows).size == 0,
        str(np.intersect1d(train_rows, valid_rows).size),
    )
    check(
        "valid_rows_hash_matches_split",
        valid_rows_sha == split["sha256"]["valid"],
        f"{valid_rows_sha} vs frozen {split['sha256']['valid']}",
    )
    check(
        "train_rows_hash_matches_split",
        train_rows_sha == split["sha256"]["train"],
        f"{train_rows_sha} vs frozen {split['sha256']['train']}",
    )
    check(
        "valid_rows_hash_matches_labels",
        valid_rows_sha == labels_payload["row_sha256_int64"]["valid"],
        valid_rows_sha,
    )
    check(
        "train_rows_hash_matches_labels",
        train_rows_sha == labels_payload["row_sha256_int64"]["train"],
        train_rows_sha,
    )

    train_row_list = [int(row) for row in split["rows"]["train"]]
    valid_row_list = [int(row) for row in split["rows"]["valid"]]
    check(
        "labels_rows_equal_split_valid",
        np.array_equal(np.asarray(labels_payload["rows"]["valid"], dtype=np.int64), valid_rows),
        "valid label row order",
    )
    check(
        "labels_rows_equal_split_train",
        np.array_equal(np.asarray(labels_payload["rows"]["train"], dtype=np.int64), train_rows),
        "train label row order",
    )

    y_valid = np.asarray(labels_payload["values"]["valid"], dtype=np.float64)
    y_train = np.asarray(labels_payload["values"]["train"], dtype=np.float64)
    check("labels_valid_length", y_valid.size == 420, str(y_valid.size))
    check("labels_train_length", y_train.size == 3360, str(y_train.size))
    check("labels_valid_finite", bool(np.isfinite(y_valid).all()), "finite")
    check("labels_train_finite", bool(np.isfinite(y_train).all()), "finite")
    check(
        "valid_label_vector_hash",
        sha256_float64(y_valid) == labels_payload["vector_sha256"]["valid"],
        sha256_float64(y_valid),
    )
    check(
        "train_label_vector_hash",
        sha256_float64(y_train) == labels_payload["vector_sha256"]["train"],
        sha256_float64(y_train),
    )

    # Only train and valid SMILES are materialised below; Test entries are not
    # dereferenced or written anywhere.
    all_smiles = split["smiles"]
    train_smiles = [all_smiles[row] for row in train_row_list]
    valid_smiles = [all_smiles[row] for row in valid_row_list]
    check(
        "train_smiles_hash",
        sha256_text(train_smiles) == labels_payload["smiles_sha256"]["train"],
        sha256_text(train_smiles),
    )
    check(
        "valid_smiles_hash",
        sha256_text(valid_smiles) == labels_payload["smiles_sha256"]["valid"],
        sha256_text(valid_smiles),
    )
    check("train_smiles_unique", len(set(train_smiles)) == 3360, str(len(set(train_smiles))))
    check("valid_smiles_unique", len(set(valid_smiles)) == 420, str(len(set(valid_smiles))))

    # ------------------------------------------------------------------
    # Historical metric reproduction
    # ------------------------------------------------------------------
    metrics_by_model = {record["model"]: record for record in metrics_payload["results"]}
    predictions: dict[str, np.ndarray] = {}
    reproduction: dict[str, Any] = {}

    for key, spec in MODELS.items():
        check(
            f"{key}_historical_record_present",
            spec["name"] in metrics_by_model,
            str(list(metrics_by_model)),
        )
        historical_record = metrics_by_model[spec["name"]]
        check(
            f"{key}_historical_n_valid",
            historical_record["n_valid"] == 420,
            str(historical_record["n_valid"]),
        )
        check(
            f"{key}_prediction_file_hash",
            source_hashes[str(spec["pred_path"].relative_to(REPO))]
            == historical_record["prediction_sha256"],
            source_hashes[str(spec["pred_path"].relative_to(REPO))],
        )
        check(
            f"{key}_historical_valid_rows_hash",
            historical_record["valid_rows_sha256"] == valid_rows_sha,
            historical_record["valid_rows_sha256"],
        )
        check(
            f"{key}_historical_rmse_matches_task",
            abs(historical_record["valid_metrics"]["rmse"] - spec["expected_rmse"]) <= 1e-12,
            f"{historical_record['valid_metrics']['rmse']} vs {spec['expected_rmse']}",
        )
        check(
            f"{key}_historical_mae_matches_task",
            abs(historical_record["valid_metrics"]["mae"] - spec["expected_mae"]) <= 1e-12,
            f"{historical_record['valid_metrics']['mae']} vs {spec['expected_mae']}",
        )

        with np.load(spec["pred_path"]) as archive:
            check(f"{key}_prediction_keys", {"rows", "pred"}.issubset(archive.files), str(archive.files))
            pred_rows = np.asarray(archive["rows"], dtype=np.int64)
            pred = np.asarray(archive["pred"], dtype=np.float64)

        check(f"{key}_prediction_shape", pred.shape == (420,), str(pred.shape))
        check(f"{key}_prediction_finite", bool(np.isfinite(pred).all()), "finite")
        check(
            f"{key}_prediction_rows_exact_frozen_order",
            np.array_equal(pred_rows, valid_rows),
            f"first_rows={pred_rows[:5].tolist()}",
        )
        check(
            f"{key}_prediction_rows_hash",
            sha256_int64(pred_rows) == valid_rows_sha,
            sha256_int64(pred_rows),
        )

        scores = metrics(pred, y_valid)
        rmse_diff = abs(scores["rmse"] - spec["expected_rmse"])
        mae_diff = abs(scores["mae"] - spec["expected_mae"])
        check(
            f"{key}_rmse_reproduction_{TOL:g}",
            rmse_diff <= TOL,
            f"recomputed={scores['rmse']:.17g} historical={spec['expected_rmse']:.17g} diff={rmse_diff:.3e}",
        )
        check(
            f"{key}_mae_reproduction_{TOL:g}",
            mae_diff <= TOL,
            f"recomputed={scores['mae']:.17g} historical={spec['expected_mae']:.17g} diff={mae_diff:.3e}",
        )

        predictions[key] = pred
        reproduction[key] = {
            "model": spec["name"],
            "run": spec["run"],
            "rmse_recomputed": scores["rmse"],
            "rmse_historical": spec["expected_rmse"],
            "rmse_abs_diff": rmse_diff,
            "mae_recomputed": scores["mae"],
            "mae_historical": spec["expected_mae"],
            "mae_abs_diff": mae_diff,
            "sse_recomputed": scores["sse"],
            "n_valid": 420,
            "passed": True,
        }
        print(
            f"[A] {spec['name']}: RMSE={scores['rmse']:.16f} "
            f"(diff={rmse_diff:.2e}), MAE={scores['mae']:.16f} (diff={mae_diff:.2e})"
        )

    # ------------------------------------------------------------------
    # B. Per-molecule paired error gap
    # ------------------------------------------------------------------
    pred_lcrc = predictions["lcrc"]
    pred_pna = predictions["pna"]
    abs_lcrc = np.abs(pred_lcrc - y_valid)
    abs_pna = np.abs(pred_pna - y_valid)
    sq_lcrc = (pred_lcrc - y_valid) ** 2
    sq_pna = (pred_pna - y_valid) ** 2
    d = sq_lcrc - sq_pna

    sse_lcrc = float(np.sum(sq_lcrc))
    sse_pna = float(np.sum(sq_pna))
    sse_diff = float(np.sum(d))
    rmse_lcrc = float(np.sqrt(np.mean(sq_lcrc)))
    rmse_pna = float(np.sqrt(np.mean(sq_pna)))
    rmse_diff = float(rmse_lcrc - rmse_pna)

    n_pna_better = int(np.sum(d > 0.0))
    n_lcrc_better = int(np.sum(d < 0.0))
    n_tie = int(np.sum(d == 0.0))
    sum_positive_d = float(np.sum(d[d > 0.0]))
    sum_negative_d = float(np.sum(d[d < 0.0]))

    order = np.argsort(-d, kind="stable")
    rank_d = np.empty(420, dtype=np.int64)
    rank_d[order] = np.arange(1, 421)

    top_rows = []
    for rank, index in enumerate(order[:20], start=1):
        top_rows.append(
            {
                "rank": rank,
                "row_id": int(valid_rows[index]),
                "y_logD": float(y_valid[index]),
                "pred_lcrc": float(pred_lcrc[index]),
                "pred_pna": float(pred_pna[index]),
                "abs_error_lcrc": float(abs_lcrc[index]),
                "abs_error_pna": float(abs_pna[index]),
                "sq_error_lcrc": float(sq_lcrc[index]),
                "sq_error_pna": float(sq_pna[index]),
                "d_lcrc_minus_pna": float(d[index]),
            }
        )

    topk_payload: dict[str, Any] = {}
    for k in (5, 10, 20):
        selected = order[:k]
        keep = np.ones(420, dtype=bool)
        keep[selected] = False
        sum_d = float(np.sum(d[selected]))
        rmse_lcrc_after = float(np.sqrt(np.mean(sq_lcrc[keep])))
        rmse_pna_after = float(np.sqrt(np.mean(sq_pna[keep])))
        topk_payload[str(k)] = {
            "k": k,
            "row_ids": [int(valid_rows[index]) for index in selected],
            "sum_d": sum_d,
            "share_of_net_sse_gap": float(sum_d / sse_diff) if sse_diff != 0.0 else None,
            "share_of_positive_d_mass": float(sum_d / sum_positive_d) if sum_positive_d != 0.0 else None,
            "sse_gap_after_removal": float(sse_diff - sum_d),
            "remaining_n": int(np.sum(keep)),
            "rmse_lcrc_after_removal": rmse_lcrc_after,
            "rmse_pna_after_removal": rmse_pna_after,
            "rmse_diff_after_removal": float(rmse_lcrc_after - rmse_pna_after),
        }

    gap_summary = {
        "n_valid": 420,
        "sse_lcrc": sse_lcrc,
        "sse_pna": sse_pna,
        "sse_diff_lcrc_minus_pna": sse_diff,
        "relative_sse_reduction_pna": float(sse_diff / sse_lcrc),
        "rmse_lcrc": rmse_lcrc,
        "rmse_pna": rmse_pna,
        "rmse_diff_lcrc_minus_pna": rmse_diff,
        "relative_rmse_reduction_pna": float(rmse_diff / rmse_lcrc),
        "n_pna_better": n_pna_better,
        "n_lcrc_better": n_lcrc_better,
        "n_tie": n_tie,
        "fraction_pna_better": float(n_pna_better / 420.0),
        "sum_positive_d": sum_positive_d,
        "sum_negative_d": sum_negative_d,
        "mean_d": float(np.mean(d)),
        "median_d": float(np.median(d)),
        "std_d": float(np.std(d, ddof=0)),
        "d_distribution": dist_stats(d),
        "top_positive_d": topk_payload,
        "top_20_rows_by_d": top_rows,
    }
    print(
        f"[B] SSE gap={sse_diff:.12f}; RMSE gap={rmse_diff:.12f}; "
        f"PNA better={n_pna_better}/420; LCRC better={n_lcrc_better}/420"
    )
    for k in (5, 10, 20):
        record = topk_payload[str(k)]
        print(
            f"    top{k:>2}: sum_d={record['sum_d']:.6f} "
            f"({100 * record['share_of_net_sse_gap']:.1f}% of net gap); "
            f"RMSE after removal LCRC={record['rmse_lcrc_after_removal']:.9f}, "
            f"PNA={record['rmse_pna_after_removal']:.9f}, "
            f"diff={record['rmse_diff_after_removal']:+.9f}"
        )

    # ------------------------------------------------------------------
    # C. RDKit fingerprints, train similarity, scaffolds, descriptors
    # ------------------------------------------------------------------
    train_mols = [Chem.MolFromSmiles(smiles) for smiles in train_smiles]
    valid_mols = [Chem.MolFromSmiles(smiles) for smiles in valid_smiles]
    check("all_train_smiles_parse", all(mol is not None for mol in train_mols), "RDKit parse")
    check("all_valid_smiles_parse", all(mol is not None for mol in valid_mols), "RDKit parse")

    train_fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
        )
        for mol in train_mols
    ]
    valid_fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
        )
        for mol in valid_mols
    ]

    valid_to_train = np.empty((420, 3360), dtype=np.float64)
    for index, fingerprint in enumerate(valid_fps):
        valid_to_train[index] = DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps)
    max_train_sim = np.max(valid_to_train, axis=1)
    nearest_train_index = np.argmax(valid_to_train, axis=1)
    nearest_train_row = train_rows[nearest_train_index]

    # Resolution caveat (measured, not assumed): a Morgan(2, 2048) Tanimoto of
    # exactly 1.0 does NOT imply an identical molecule, because bits are hashed
    # and radius-2 environments can coincide for homologues / symmetric swaps.
    # Every such pair is therefore reported as a fingerprint-equivalent pair.
    train_canonical = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in train_mols]
    valid_canonical = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in valid_mols]
    train_canonical_set = set(train_canonical)
    check(
        "train_canonical_smiles_unique",
        len(train_canonical_set) == 3360,
        f"{len(train_canonical_set)} unique canonical Train3360 molecules",
    )
    # Section E is defined on Train3360 canonical SMILES, so it uses its own
    # fingerprint set rather than the split-order strings used by section C.
    train_canonical_mols = [Chem.MolFromSmiles(smiles) for smiles in train_canonical]
    check(
        "all_train_canonical_smiles_parse",
        all(mol is not None for mol in train_canonical_mols),
        "RDKit parse of canonical Train3360 SMILES",
    )
    train_canonical_fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
        )
        for mol in train_canonical_mols
    ]
    check(
        "canonical_and_split_train_fingerprints_identical",
        all(
            DataStructs.TanimotoSimilarity(left, right) >= 1.0 - 1e-12
            for left, right in zip(train_canonical_fps, train_fps, strict=True)
        ),
        "Morgan(2,2048) bit sets are invariant to the split SMILES atom order",
    )
    fp_identical_valid = max_train_sim >= 1.0 - 1e-12
    valid_identical_canonical = np.asarray(
        [
            bool(flag) and (valid_canonical[index] not in train_canonical_set)
            for index, flag in enumerate(fp_identical_valid)
        ],
        dtype=bool,
    )
    valid_exact_duplicate = np.asarray(
        [
            bool(flag) and (valid_canonical[index] in train_canonical_set)
            for index, flag in enumerate(fp_identical_valid)
        ],
        dtype=bool,
    )
    fingerprint_resolution = {
        "definition": (
            "max_train_sim == 1.0 groups: (a) valid molecule is also an exact "
            "canonical-SMILES duplicate of some Train3360 molecule, or (b) the "
            "Morgan(2,2048) bit set is identical to a structurally different "
            "Train3360 molecule (hashed-bit coincidence)."
        ),
        "n_valid_max_sim_eq_1": int(np.sum(fp_identical_valid)),
        "n_valid_exact_canonical_duplicate_in_train": int(np.sum(valid_exact_duplicate)),
        "n_valid_fingerprint_equivalent_but_nonidentical": int(
            np.sum(valid_identical_canonical)
        ),
        "valid_max_sim_eq_1": {
            "n": int(np.sum(fp_identical_valid)),
            "rmse_lcrc": (
                float(np.sqrt(np.mean(sq_lcrc[fp_identical_valid])))
                if np.any(fp_identical_valid)
                else None
            ),
            "rmse_pna": (
                float(np.sqrt(np.mean(sq_pna[fp_identical_valid])))
                if np.any(fp_identical_valid)
                else None
            ),
            "sse_diff_lcrc_minus_pna": (
                float(np.sum(d[fp_identical_valid])) if np.any(fp_identical_valid) else None
            ),
            "row_ids": [int(row) for row in valid_rows[fp_identical_valid]],
        },
    }

    train_to_train = np.empty((3360, 3360), dtype=np.float32)
    for index, fingerprint in enumerate(train_canonical_fps):
        train_to_train[index] = DataStructs.BulkTanimotoSimilarity(
            fingerprint, train_canonical_fps
        )
    np.fill_diagonal(train_to_train, -1.0)
    train_max_neighbor_sim = np.max(train_to_train, axis=1)

    train_scaffold = []
    valid_scaffold = []
    valid_scaffold_fallback = []
    train_scaffold_fallback = []
    for mol in train_mols:
        scaffold, fallback = scaffold_and_fallback(mol)
        train_scaffold.append(scaffold)
        train_scaffold_fallback.append(fallback)
    for mol in valid_mols:
        scaffold, fallback = scaffold_and_fallback(mol)
        valid_scaffold.append(scaffold)
        valid_scaffold_fallback.append(fallback)

    train_scaffold_set = set(train_scaffold)
    scaffold_seen = np.asarray([scaffold in train_scaffold_set for scaffold in valid_scaffold], dtype=bool)

    heavy_atoms = np.asarray([mol.GetNumHeavyAtoms() for mol in valid_mols], dtype=np.int64)
    ring_count = np.asarray([rdMolDescriptors.CalcNumRings(mol) for mol in valid_mols], dtype=np.int64)
    aromatic_atom_fraction = np.asarray(
        [
            sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1 and atom.GetIsAromatic())
            / max(mol.GetNumHeavyAtoms(), 1)
            for mol in valid_mols
        ],
        dtype=np.float64,
    )
    tpsa = np.asarray([Descriptors.TPSA(mol) for mol in valid_mols], dtype=np.float64)
    hbd = np.asarray([rdMolDescriptors.CalcNumHBD(mol) for mol in valid_mols], dtype=np.int64)
    hba = np.asarray([rdMolDescriptors.CalcNumHBA(mol) for mol in valid_mols], dtype=np.int64)
    mol_logp = np.asarray([Descriptors.MolLogP(mol) for mol in valid_mols], dtype=np.float64)

    sim_group, sim_edges = assign_quartiles(max_train_sim)
    size_group, size_edges = assign_quartiles(heavy_atoms)
    tpsa_group, tpsa_edges = assign_quartiles(tpsa)
    logp_group, logp_edges = assign_quartiles(mol_logp)

    def subgroup_record(
        family: str,
        group: str,
        mask: np.ndarray,
        values: np.ndarray | None = None,
    ) -> dict[str, Any]:
        n = int(np.sum(mask))
        if n == 0:
            stop(f"empty subgroup {family}/{group}")
        group_rmse_lcrc = float(np.sqrt(np.mean(sq_lcrc[mask])))
        group_rmse_pna = float(np.sqrt(np.mean(sq_pna[mask])))
        group_sse_lcrc = float(np.sum(sq_lcrc[mask]))
        group_sse_pna = float(np.sum(sq_pna[mask]))
        group_sse_diff = float(np.sum(d[mask]))
        record = {
            "family": family,
            "group": group,
            "n": n,
            "group_min": float(np.min(values[mask])) if values is not None else None,
            "group_median": float(np.median(values[mask])) if values is not None else None,
            "group_max": float(np.max(values[mask])) if values is not None else None,
            "rmse_lcrc": group_rmse_lcrc,
            "rmse_pna": group_rmse_pna,
            "rmse_diff_lcrc_minus_pna": float(group_rmse_lcrc - group_rmse_pna),
            "mae_lcrc": float(np.mean(abs_lcrc[mask])),
            "mae_pna": float(np.mean(abs_pna[mask])),
            "sse_lcrc": group_sse_lcrc,
            "sse_pna": group_sse_pna,
            "sse_diff_lcrc_minus_pna": group_sse_diff,
            "d_mean": float(np.mean(d[mask])),
            "d_median": float(np.median(d[mask])),
            "n_pna_better": int(np.sum(d[mask] > 0.0)),
            "fraction_pna_better": float(np.mean(d[mask] > 0.0)),
            "share_of_total_d_sum": float(group_sse_diff / sse_diff) if sse_diff != 0.0 else None,
            "stability": "stable" if n >= STABLE_N else f"small_n_unstable_n<{STABLE_N}",
        }
        return record

    subgroup_records: list[dict[str, Any]] = [
        subgroup_record("all", "all", np.ones(420, dtype=bool))
    ]

    sim_labels = quartile_labels()
    for group_index, group_label in enumerate(sim_labels):
        subgroup_records.append(
            subgroup_record(
                "train_similarity_quartile",
                group_label,
                sim_group == group_index,
                max_train_sim,
            )
        )
    for group_label, mask in (
        ("seen", scaffold_seen),
        ("unseen", ~scaffold_seen),
    ):
        subgroup_records.append(subgroup_record("scaffold", group_label, mask))

    for family, group_array, values in (
        ("size_quartile", size_group, heavy_atoms),
        ("tpsa_quartile", tpsa_group, tpsa),
        ("logp_quartile", logp_group, mol_logp),
    ):
        for group_index, group_label in enumerate(quartile_labels()):
            subgroup_records.append(
                subgroup_record(family, group_label, group_array == group_index, values)
            )

    subgroup_df = pd.DataFrame(subgroup_records)
    subgroup_df.to_csv(ANALYSIS / "subgroups.csv", index=False)

    # ------------------------------------------------------------------
    # E. Train-only local neighbor support (threshold >= 0.80)
    # ------------------------------------------------------------------
    neighbor_count = np.zeros(3360, dtype=np.int64)
    neighbor_label_std = np.full(3360, np.nan, dtype=np.float64)
    neighbor_label_std_sample = np.full(3360, np.nan, dtype=np.float64)
    nearest_threshold_row = np.full(3360, -1, dtype=np.int64)
    nearest_threshold_similarity = np.full(3360, np.nan, dtype=np.float64)
    label_diff_threshold = np.full(3360, np.nan, dtype=np.float64)
    nearest_any_row = np.zeros(3360, dtype=np.int64)
    nearest_any_similarity = np.zeros(3360, dtype=np.float64)
    label_diff_any = np.zeros(3360, dtype=np.float64)

    for index in range(3360):
        similarities = train_to_train[index]
        mask = similarities >= NEIGHBOR_SIM_THRESHOLD
        mask[index] = False
        neighbor_indices = np.flatnonzero(mask)
        count = int(neighbor_indices.size)
        neighbor_count[index] = count

        if count:
            neighbor_labels = y_train[neighbor_indices]
            neighbor_label_std[index] = float(np.std(neighbor_labels, ddof=0))
            if count >= 2:
                neighbor_label_std_sample[index] = float(np.std(neighbor_labels, ddof=1))
            nearest_local_offset = int(np.argmax(similarities[neighbor_indices]))
            nearest_index = int(neighbor_indices[nearest_local_offset])
            nearest_threshold_row[index] = int(train_rows[nearest_index])
            nearest_threshold_similarity[index] = float(similarities[nearest_index])
            label_diff_threshold[index] = float(y_train[index] - y_train[nearest_index])

        nearest_index_any = int(np.argmax(similarities))
        nearest_any_row[index] = int(train_rows[nearest_index_any])
        nearest_any_similarity[index] = float(similarities[nearest_index_any])
        label_diff_any[index] = float(y_train[index] - y_train[nearest_index_any])

    support_mask = neighbor_count >= 1
    support_ge3 = neighbor_count >= 3
    support_ge5 = neighbor_count >= 5

    def support_level(level_mask: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(np.sum(level_mask)),
            "neighbor_label_std_ddof0": dist_stats(neighbor_label_std[level_mask]),
            "abs_label_diff_to_nearest_threshold_neighbor": dist_stats(
                np.abs(label_diff_threshold[level_mask])
            ),
            "signed_label_diff_to_nearest_threshold_neighbor": dist_stats(
                label_diff_threshold[level_mask]
            ),
        }

    abs_diff_threshold = np.abs(label_diff_threshold[support_mask])
    abs_diff_any = np.abs(label_diff_any)

    neighbor_count_hist = {
        "0": int(np.sum(neighbor_count == 0)),
        "1": int(np.sum(neighbor_count == 1)),
        "2": int(np.sum(neighbor_count == 2)),
        "3-4": int(np.sum((neighbor_count >= 3) & (neighbor_count <= 4))),
        "5-9": int(np.sum((neighbor_count >= 5) & (neighbor_count <= 9))),
        "10+": int(np.sum(neighbor_count >= 10)),
    }

    sim1_train_mask = train_max_neighbor_sim >= 1.0 - 1e-12
    neighbor_support = {
        "scope": "Train3360 canonical SMILES and Train3360 labels only; Validation/Test not used",
        "parameters": {
            "train_smiles": "RDKit canonical SMILES (isomericSmiles=True) from the frozen Train3360 split SMILES",
            "fingerprint": "Morgan",
            "radius": FINGERPRINT_RADIUS,
            "n_bits": FINGERPRINT_BITS,
            "similarity": "Tanimoto",
            "neighbor_threshold": NEIGHBOR_SIM_THRESHOLD,
            "self_excluded": True,
            "nearest_tie_break": "first occurrence in frozen Train3360 order",
        },
        "n_train": 3360,
        "counts": {
            "n_zero_neighbors": int(np.sum(neighbor_count == 0)),
            "n_at_least_1_neighbor": int(np.sum(neighbor_count >= 1)),
            "n_at_least_3_neighbors": int(np.sum(neighbor_count >= 3)),
            "n_at_least_5_neighbors": int(np.sum(neighbor_count >= 5)),
            "n_at_least_10_neighbors": int(np.sum(neighbor_count >= 10)),
            "fraction_at_least_1": float(np.mean(neighbor_count >= 1)),
            "fraction_at_least_3": float(np.mean(neighbor_count >= 3)),
            "fraction_at_least_5": float(np.mean(neighbor_count >= 5)),
        },
        "neighbor_count_distribution": {
            "summary": dist_stats(neighbor_count),
            "categories": neighbor_count_hist,
        },
        "neighbor_label_std": {
            "definition": "population standard deviation (ddof=0) over all >=0.80 neighbors; exactly one neighbor gives 0.0",
            "n_with_at_least_1": int(np.sum(support_mask)),
            "n_with_exactly_1": int(np.sum(neighbor_count == 1)),
            "all_with_at_least_1": dist_stats(neighbor_label_std[support_mask]),
            "with_at_least_2": dist_stats(neighbor_label_std[neighbor_count >= 2]),
            "sample_std_ddof1_with_at_least_2": dist_stats(neighbor_label_std_sample[neighbor_count >= 2]),
        },
        "label_diff_to_nearest_neighbor": {
            "definition_signed": "y_i - y_nearest; threshold set is the >=0.80 neighbor set",
            "threshold_set": {
                "n": int(np.sum(support_mask)),
                "signed": dist_stats(label_diff_threshold[support_mask]),
                "absolute": dist_stats(abs_diff_threshold),
                "signed_histogram": histogram(
                    label_diff_threshold[support_mask],
                    [-math.inf, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, math.inf],
                ),
                "absolute_histogram": histogram(
                    abs_diff_threshold,
                    [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, math.inf],
                ),
            },
            "all_other_train_nearest": {
                "n": 3360,
                "signed": dist_stats(label_diff_any),
                "absolute": dist_stats(abs_diff_any),
                "signed_histogram": histogram(
                    label_diff_any,
                    [-math.inf, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, math.inf],
                ),
                "absolute_histogram": histogram(
                    abs_diff_any,
                    [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, math.inf],
                ),
            },
        },
        "support_levels": {
            "at_least_1": support_level(support_mask),
            "at_least_3": support_level(support_ge3),
            "at_least_5": support_level(support_ge5),
        },
        "resolution_caveats": {
            "max_other_train_similarity": dist_stats(train_max_neighbor_sim),
            "n_train_with_a_sim_eq_1_neighbor": int(np.sum(sim1_train_mask)),
            "n_train_fingerprint_equivalent_pairs": int(
                np.sum(train_to_train >= 1.0 - 1e-12) // 2
            ),
            "sim_eq_1_subset": {
                "n": int(np.sum(sim1_train_mask)),
                "abs_label_diff_to_nearest_ge080": dist_stats(
                    np.abs(label_diff_threshold[sim1_train_mask])
                ),
                "n_with_abs_label_diff_eq_0": int(
                    np.sum(np.abs(label_diff_threshold[sim1_train_mask]) == 0.0)
                ),
            },
            "note": (
                "Tanimoto 1.0 on hashed Morgan bits is fingerprint equivalence, not "
                "molecular identity; it must not be read as a single-group edit."
            ),
        },
        "per_molecule": [
            {
                "row_id": int(train_rows[index]),
                "n_neighbors_ge080": int(neighbor_count[index]),
                "neighbor_label_std_ddof0": (
                    float(neighbor_label_std[index])
                    if np.isfinite(neighbor_label_std[index])
                    else None
                ),
                "nearest_ge080_row": (
                    int(nearest_threshold_row[index])
                    if nearest_threshold_row[index] >= 0
                    else None
                ),
                "nearest_ge080_similarity": (
                    float(nearest_threshold_similarity[index])
                    if np.isfinite(nearest_threshold_similarity[index])
                    else None
                ),
                "label_diff_to_nearest_ge080": (
                    float(label_diff_threshold[index])
                    if np.isfinite(label_diff_threshold[index])
                    else None
                ),
                "nearest_any_row": int(nearest_any_row[index]),
                "nearest_any_similarity": float(nearest_any_similarity[index]),
                "label_diff_to_nearest_any": float(label_diff_any[index]),
            }
            for index in range(3360)
        ],
        "hashes": {
            "train_rows_sha256_int64": train_rows_sha,
            "train_labels_sha256_float64": sha256_float64(y_train),
            "train_smiles_sha256_text": sha256_text(train_smiles),
            "train_canonical_smiles_sha256_text": sha256_text(train_canonical),
        },
    }
    write_json(ANALYSIS / "train_neighbor_support.json", neighbor_support)

    print(
        "[E] train support >=0.80: "
        f">=1 {np.sum(support_mask)}/3360, >=3 {np.sum(support_ge3)}/3360, "
        f">=5 {np.sum(support_ge5)}/3360"
    )

    # ------------------------------------------------------------------
    # Per-molecule table
    # ------------------------------------------------------------------
    top_indices = {
        5: set(int(index) for index in order[:5]),
        10: set(int(index) for index in order[:10]),
        20: set(int(index) for index in order[:20]),
    }
    per_molecule = pd.DataFrame(
        {
            "row_id": valid_rows,
            "smiles": valid_smiles,
            "y_logD": y_valid,
            "pred_lcrc": pred_lcrc,
            "pred_pna": pred_pna,
            "abs_err_lcrc": abs_lcrc,
            "abs_err_pna": abs_pna,
            "sq_err_lcrc": sq_lcrc,
            "sq_err_pna": sq_pna,
            "d_sq_err_lcrc_minus_pna": d,
            "better_model": [
                "PNA" if value > 0.0 else ("LCRC" if value < 0.0 else "tie") for value in d
            ],
            "rank_d_desc": rank_d,
            "in_top5_positive_d": [index in top_indices[5] for index in range(420)],
            "in_top10_positive_d": [index in top_indices[10] for index in range(420)],
            "in_top20_positive_d": [index in top_indices[20] for index in range(420)],
            "max_train_tanimoto": max_train_sim,
            "nearest_train_row": nearest_train_row,
            "sim_quartile": [sim_labels[index] for index in sim_group],
            "scaffold": valid_scaffold,
            "scaffold_seen": scaffold_seen,
            "heavy_atoms": heavy_atoms,
            "ring_count": ring_count,
            "aromatic_atom_fraction": aromatic_atom_fraction,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "mol_logp": mol_logp,
            "size_quartile": [quartile_labels()[index] for index in size_group],
            "tpsa_quartile": [quartile_labels()[index] for index in tpsa_group],
            "logp_quartile": [quartile_labels()[index] for index in logp_group],
        }
    )
    per_molecule.to_csv(
        ANALYSIS / "per_molecule_valid.csv",
        index=False,
        float_format="%.17g",
    )

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    sim_definitions = []
    for group_index, group_label in enumerate(sim_labels):
        mask = sim_group == group_index
        sim_definitions.append(
            {
                "group": group_label,
                "n": int(np.sum(mask)),
                "min": float(np.min(max_train_sim[mask])),
                "median": float(np.median(max_train_sim[mask])),
                "max": float(np.max(max_train_sim[mask])),
            }
        )

    descriptor_edges = {
        "heavy_atoms": [float(value) for value in size_edges],
        "tpsa": [float(value) for value in tpsa_edges],
        "mol_logp": [float(value) for value in logp_edges],
    }

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rdkit": Chem.rdBase.rdkitVersion,
        "matplotlib": matplotlib.__version__,
        "gpu_used": False,
        "training_or_finetuning": False,
        "test_artifacts_used": False,
    }

    summary = {
        "audit_id": "lcrc-lipo-random42-gap-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": (
            "Frozen random42 Validation420 only. Read-only audit of the two existing "
            "prediction files; no training, no checkpoint selection, no Test labels, "
            "no Test predictions, no Test statistics, and no historical per-molecule "
            "Test analysis were opened."
        ),
        "environment": environment,
        "inputs": {
            "source_track": str(SOURCE.relative_to(REPO)),
            "files": source_hashes,
            "verified_hashes": {
                "valid_rows_sha256_int64": valid_rows_sha,
                "train_rows_sha256_int64": train_rows_sha,
                "valid_labels_sha256_float64": sha256_float64(y_valid),
                "train_labels_sha256_float64": sha256_float64(y_train),
                "valid_smiles_sha256_text": sha256_text(valid_smiles),
                "train_smiles_sha256_text": sha256_text(train_smiles),
                "train_canonical_smiles_sha256_text": sha256_text(train_canonical),
            },
            "n_train": 3360,
            "n_valid": 420,
        },
        "checks": checks,
        "metrics_reproduction": {
            "tolerance": TOL,
            "models": reproduction,
            "all_passed": True,
        },
        "paired_error_gap": gap_summary,
        "similarity_and_scaffold": {
            "fingerprint": {
                "type": "Morgan",
                "radius": FINGERPRINT_RADIUS,
                "n_bits": FINGERPRINT_BITS,
                "similarity": "Tanimoto",
            },
            "max_train_similarity_summary": dist_stats(max_train_sim),
            "quartile_edges": [float(value) for value in sim_edges],
            "quartile_definitions": sim_definitions,
            "fingerprint_resolution": fingerprint_resolution,
            "scaffold": {
                "definition": (
                    "Bemis-Murcko scaffold canonical SMILES; empty scaffold falls back "
                    "to the molecule's canonical SMILES"
                ),
                "n_train_unique_scaffolds": int(len(train_scaffold_set)),
                "n_valid_unique_scaffolds": int(len(set(valid_scaffold))),
                "n_valid_empty_scaffold_fallback": int(sum(valid_scaffold_fallback)),
                "n_train_empty_scaffold_fallback": int(sum(train_scaffold_fallback)),
                "n_valid_scaffold_seen": int(np.sum(scaffold_seen)),
                "n_valid_scaffold_unseen": int(np.sum(~scaffold_seen)),
            },
        },
        "descriptor_groups": {
            "definitions": {
                "heavy_atoms": "RDKit MolFromSmiles (no explicit H); GetNumHeavyAtoms()",
                "ring_count": "rdMolDescriptors.CalcNumRings()",
                "aromatic_atom_fraction": "aromatic heavy atoms / heavy atoms",
                "tpsa": "Descriptors.TPSA()",
                "hbd": "rdMolDescriptors.CalcNumHBD()",
                "hba": "rdMolDescriptors.CalcNumHBA()",
                "mol_logp": "Descriptors.MolLogP()",
                "quartile_rule": (
                    "sample 25/50/75 percentiles; inner boundaries are right-closed; "
                    "values equal to a boundary go to the lower quartile"
                ),
            },
            "quartile_edges": descriptor_edges,
            "descriptor_summaries": {
                "heavy_atoms": dist_stats(heavy_atoms),
                "ring_count": dist_stats(ring_count),
                "aromatic_atom_fraction": dist_stats(aromatic_atom_fraction),
                "tpsa": dist_stats(tpsa),
                "hbd": dist_stats(hbd),
                "hba": dist_stats(hba),
                "mol_logp": dist_stats(mol_logp),
            },
        },
        "subgroups": subgroup_records,
        "train_neighbor_support": {
            "path": "analysis/train_neighbor_support.json",
            "parameters": neighbor_support["parameters"],
            "counts": neighbor_support["counts"],
            "neighbor_count_distribution": neighbor_support["neighbor_count_distribution"],
            "neighbor_label_std": neighbor_support["neighbor_label_std"],
            "label_diff_to_nearest_neighbor": neighbor_support["label_diff_to_nearest_neighbor"],
            "support_levels": neighbor_support["support_levels"],
            "resolution_caveats": neighbor_support["resolution_caveats"],
        },
        "code": {
            "path": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(ANALYSIS / "summary.json", summary)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    if "--no-figures" not in sys.argv:
        make_figures(
            d=d,
            order=order,
            sse_diff=sse_diff,
            sq_lcrc=sq_lcrc,
            sq_pna=sq_pna,
            subgroup_df=subgroup_df,
            neighbor_count=neighbor_count,
            neighbor_label_std=neighbor_label_std,
            label_diff_threshold=label_diff_threshold,
            support_mask=support_mask,
            support_ge3=support_ge3,
            support_ge5=support_ge5,
        )
        figure_hashes = {
            str(path.relative_to(TRACK)): sha256_file(path)
            for path in sorted(FIGURES.glob("*.png"))
        }
        summary["outputs"] = {
            "per_molecule_valid_csv": str((ANALYSIS / "per_molecule_valid.csv").relative_to(TRACK)),
            "subgroups_csv": str((ANALYSIS / "subgroups.csv").relative_to(TRACK)),
            "train_neighbor_support_json": str(
                (ANALYSIS / "train_neighbor_support.json").relative_to(TRACK)
            ),
            "figures": figure_hashes,
            "sha256": {
                "analysis/per_molecule_valid.csv": sha256_file(ANALYSIS / "per_molecule_valid.csv"),
                "analysis/subgroups.csv": sha256_file(ANALYSIS / "subgroups.csv"),
                "analysis/train_neighbor_support.json": sha256_file(
                    ANALYSIS / "train_neighbor_support.json"
                ),
            },
        }
        write_json(ANALYSIS / "summary.json", summary)

    print("[DONE] wrote analysis/per_molecule_valid.csv, analysis/subgroups.csv,")
    print("       analysis/train_neighbor_support.json, analysis/summary.json")
    print("       and figures/")


def make_figures(
    *,
    d: np.ndarray,
    order: np.ndarray,
    sse_diff: float,
    sq_lcrc: np.ndarray,
    sq_pna: np.ndarray,
    subgroup_df: pd.DataFrame,
    neighbor_count: np.ndarray,
    neighbor_label_std: np.ndarray,
    label_diff_threshold: np.ndarray,
    support_mask: np.ndarray,
    support_ge3: np.ndarray,
    support_ge5: np.ndarray,
) -> None:
    n = d.size

    # Figure 1: concentration of the net gap and leave-top-k RMSE.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cumulative = np.concatenate([[0.0], np.cumsum(d[order])])
    x = np.arange(n + 1)
    axes[0].plot(x, cumulative, color="#1f77b4", linewidth=2)
    axes[0].axhline(sse_diff, color="#d62728", linestyle="--", linewidth=1.5, label="net SSE gap")
    for k in (5, 10, 20):
        axes[0].axvline(k, color="0.65", linestyle=":", linewidth=1)
        axes[0].text(k, cumulative.max() * 0.97, f" top {k}", fontsize=8, rotation=90, va="top")
    axes[0].set_xlim(0, n)
    axes[0].set_xlabel("Number of highest-d Validation420 molecules removed")
    axes[0].set_ylabel("Cumulative sum of d (SSE units)")
    axes[0].set_title("PNA advantage mass is concentrated")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].grid(alpha=0.25)

    removed = np.arange(0, n)
    remaining = n - removed
    cum_lcrc = np.concatenate([[0.0], np.cumsum(sq_lcrc[order])])
    cum_pna = np.concatenate([[0.0], np.cumsum(sq_pna[order])])
    rmse_lcrc_removed = np.sqrt((sq_lcrc.sum() - cum_lcrc[:n]) / remaining)
    rmse_pna_removed = np.sqrt((sq_pna.sum() - cum_pna[:n]) / remaining)
    axes[1].plot(removed, rmse_lcrc_removed, label="LCRC-A-R3", color="#ff7f0e", linewidth=2)
    axes[1].plot(removed, rmse_pna_removed, label="PNA-native", color="#2ca02c", linewidth=2)
    for k in (5, 10, 20):
        axes[1].axvline(k, color="0.65", linestyle=":", linewidth=1)
    axes[1].set_xlim(0, 120)
    axes[1].set_xlabel("Number of highest-d Validation420 molecules removed")
    axes[1].set_ylabel("RMSE on remaining molecules (logD)")
    axes[1].set_title("Removing top-d molecules reverses the RMSE ranking")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_gap_concentration.png", dpi=180)
    plt.close(fig)

    # Figure 2: RMSE by requested subgroup families.
    families = [
        ("Train-similarity quartile", "train_similarity_quartile"),
        ("Scaffold status", "scaffold"),
        ("Heavy-atom quartile", "size_quartile"),
        ("TPSA quartile", "tpsa_quartile"),
        ("MolLogP quartile", "logp_quartile"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    flat_axes = axes.ravel()
    for ax, (title, family) in zip(flat_axes, families, strict=False):
        frame = subgroup_df[subgroup_df["family"] == family].reset_index(drop=True)
        xpos = np.arange(len(frame))
        width = 0.38
        ax.bar(xpos - width / 2, frame["rmse_lcrc"], width, label="LCRC-A-R3", color="#ff7f0e")
        ax.bar(xpos + width / 2, frame["rmse_pna"], width, label="PNA-native", color="#2ca02c")
        ax.set_xticks(xpos)
        ax.set_xticklabels(
            [f"{group}\n(n={count})" for group, count in zip(frame["group"], frame["n"])],
            fontsize=8,
        )
        upper = max(float(frame["rmse_lcrc"].max()), float(frame["rmse_pna"].max()))
        ax.set_ylim(0, upper * 1.22)
        ax.set_ylabel("RMSE (logD)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    flat_axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_group_rmse.png", dpi=180)
    plt.close(fig)

    # Figure 3: train-only neighbor support.
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    ax = axes[0, 0]
    levels = [">=1", ">=3", ">=5"]
    counts = [int(np.sum(support_mask)), int(np.sum(support_ge3)), int(np.sum(support_ge5))]
    ax.bar(levels, counts, color="#1f77b4")
    for xpos, count in zip(levels, counts):
        ax.text(xpos, count + 20, f"{count}/3360", ha="center", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_ylabel("Training molecules")
    ax.set_title("Molecules with a >=0.80 Morgan neighbor")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    max_count = int(neighbor_count.max())
    bins = np.arange(0, max_count + 2) - 0.5
    ax.hist(neighbor_count, bins=bins, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Number of >=0.80 training neighbors")
    ax.set_ylabel("Training molecules")
    ax.set_title("Neighbor-count distribution")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    abs_diff = np.abs(label_diff_threshold[support_mask])
    ax.hist(abs_diff, bins=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0], color="#2ca02c", edgecolor="white")
    ax.set_xlabel("|y_i - y_nearest| among >=0.80 neighbors")
    ax.set_ylabel("Training molecules")
    ax.set_title("Label difference to nearest qualifying neighbor")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    box_data = [
        neighbor_label_std[support_mask],
        neighbor_label_std[support_ge3],
        neighbor_label_std[support_ge5],
    ]
    ax.boxplot(box_data, tick_labels=levels, showfliers=False)
    ax.set_xlabel("Support level")
    ax.set_ylabel("Neighbor label std (ddof=0)")
    ax.set_title("Neighbor-label dispersion by support level")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_train_neighbor_support.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
