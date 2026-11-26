"""Bemis--Murcko scaffolds of the official Train molecules and the balanced weights.

The scaffold definition is the *verbatim* previous-round implementation
(``lcrc_structural_failure_localization_v0/code/sfl_common.py::scaffold_key``),
imported read-only so the rule cannot drift:

    Chem.MolFromSmiles(smiles) -> MurckoScaffold.GetScaffoldForMol -> Chem.MolToSmiles

``MolToSmiles`` keeps the default ``isomericSmiles=True``, so stereo descriptors
that survive the scaffold extraction remain part of the scaffold key.  Empty
scaffolds (acyclic molecules) all share the single group ``""``; multi-component
SMILES are used as parsed (no salt stripping, no largest-fragment rule).

Nothing here accepts a label: the weights are a function of the Train scaffold
partition and the two integers ``N_train`` and ``K_train`` only.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import sb_common as C

import sfl_common as SFL  # noqa: E402  (read-only previous-round scaffold rule)


# ---------------------------------------------------------------------------
# official rows
# ---------------------------------------------------------------------------


def load_official_rows() -> dict[str, Any]:
    """Official Train/Valid row ids, SMILES and labels (Test split file never opened)."""
    mol = pd.read_csv(C.MOL_CSV).reset_index().rename(columns={"index": "row_id"})
    if len(mol) != 4200:
        raise RuntimeError(f"mol.csv rows {len(mol)} != 4200")
    train = pd.read_csv(C.SPLIT_DIR / "train.csv.gz", header=None)[0].to_numpy(dtype=np.int64)
    valid = pd.read_csv(C.SPLIT_DIR / "valid.csv.gz", header=None)[0].to_numpy(dtype=np.int64)
    if train.size != C.N_TRAIN or valid.size != C.N_VALID:
        raise RuntimeError(f"split sizes {train.size}/{valid.size} != {C.N_TRAIN}/{C.N_VALID}")
    union = np.sort(np.concatenate([train, valid]))
    if union.size != np.unique(union).size:
        raise RuntimeError("Train/Valid row ids overlap")
    smiles = dict(zip(mol["row_id"].astype(int), mol["smiles"].astype(str)))
    labels = dict(zip(mol["row_id"].astype(int), mol["exp"].astype(float)))
    return {
        "train_rows": train,
        "valid_rows": valid,
        "smiles": smiles,
        "labels": labels,
        "mol_csv_sha256": C.sha256_file(C.MOL_CSV),
        "split_sha256": {
            "train": C.sha256_file(C.SPLIT_DIR / "train.csv.gz"),
            "valid": C.sha256_file(C.SPLIT_DIR / "valid.csv.gz"),
        },
        "sfl_scaffold_sha256": C.sha256_file(C.SFL_CODE / "sfl_common.py"),
        "test_split_file_opened": False,
    }


def scaffold_of_rows(smiles: Mapping[int, str], rows: Sequence[int]) -> dict[int, str]:
    """Scaffold string per official row id using the frozen previous-round rule."""
    return {int(r): SFL.scaffold_key(str(smiles[int(r)])) for r in rows}


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def scaffold_counts(scaffold_by_row: Mapping[int, str], rows: Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        key = scaffold_by_row[int(r)]
        counts[key] = counts.get(key, 0) + 1
    return counts


def balanced_weights(
    scaffold_by_row: Mapping[int, str],
    rows: Sequence[int],
    *,
    n_train: int,
    k_train: int,
) -> dict[int, float]:
    """``weight_i = N_train / (K_train * n_s(i))`` — structure only, no labels."""
    counts = scaffold_counts(scaffold_by_row, rows)
    if int(n_train) != len(rows):
        raise ValueError(f"n_train {n_train} != number of rows {len(rows)}")
    if int(k_train) != len(counts):
        raise ValueError(f"k_train {k_train} != number of distinct scaffolds {len(counts)}")
    base = float(n_train) / float(k_train)
    return {int(r): base / float(counts[scaffold_by_row[int(r)]]) for r in rows}


def weight_audit(
    scaffold_by_row: Mapping[int, str],
    rows: Sequence[int],
    weights: Mapping[int, float],
) -> dict[str, Any]:
    """Frozen checks 4-5: per-scaffold totals, mean, quantiles, effective sample size."""
    counts = scaffold_counts(scaffold_by_row, rows)
    k = len(counts)
    n = len(rows)
    target = float(n) / float(k)
    keys = np.asarray([scaffold_by_row[int(r)] for r in rows], dtype=object)
    w = np.asarray([float(weights[int(r)]) for r in rows], dtype=np.float64)
    uniq, inverse = np.unique(keys.astype(str), return_inverse=True)
    totals_arr = np.bincount(inverse, weights=w, minlength=uniq.size)
    if uniq.size != k:
        raise RuntimeError(f"group count mismatch during the weight audit: {uniq.size} != {k}")
    ess = C.effective_sample_size(w)
    return {
        "n_train": int(n),
        "k_scaffolds": int(k),
        "mean_weight": float(w.mean()),
        "sum_weight": float(w.sum()),
        "expected_total_per_scaffold": target,
        "per_scaffold_total_min": float(totals_arr.min()),
        "per_scaffold_total_max": float(totals_arr.max()),
        "per_scaffold_total_max_abs_deviation": float(np.max(np.abs(totals_arr - target))),
        "per_scaffold_total_relative_deviation": float(np.max(np.abs(totals_arr - target)) / target),
        "all_per_scaffold_totals_equal": bool(np.max(np.abs(totals_arr - target)) <= 1e-9 * target),
        "weight_summary": C.percentile_summary(w),
        "effective_sample_size": ess,
        "effective_sample_size_fraction": float(ess / n),
        "uniform_weight_scaffold_total": {   # what the original equal-per-molecule objective gives
            "min": float(min(counts.values())) / n,
            "max": float(max(counts.values())) / n,
            "max_over_min": float(max(counts.values())) / float(min(counts.values())),
        },
        "weight_sha256": C.sha256_array(np.asarray([float(weights[int(r)]) for r in sorted(rows)])),
    }


def scaffold_distribution(
    scaffold_by_row: Mapping[int, str],
    train_rows: Sequence[int],
    valid_rows: Sequence[int],
) -> dict[str, Any]:
    """Section 4 statistics of the Train scaffold partition."""
    counts = scaffold_counts(scaffold_by_row, train_rows)
    sizes = np.asarray(sorted(counts.values(), reverse=True), dtype=np.int64)
    n = int(len(train_rows))
    k = int(len(counts))
    valid_keys = {scaffold_by_row[int(r)] for r in valid_rows}
    hist = {int(s): int((sizes == s).sum()) for s in sorted(set(int(x) for x in sizes))}
    empty = counts.get("", 0)
    empty_rows = [int(r) for r in train_rows if scaffold_by_row[int(r)] == ""]
    multi = [int(r) for r in train_rows if "." in str(scaffold_by_row[int(r)])]
    return {
        "n_train_molecules": n,
        "K_distinct_train_scaffolds": k,
        "mean_scaffold_size": float(n / k),
        "size_summary": C.percentile_summary(sizes.astype(np.float64)),
        "size_histogram": hist,
        "n_singleton_scaffolds": int((sizes == 1).sum()),
        "singleton_fraction_of_scaffolds": float((sizes == 1).mean()),
        "singleton_fraction_of_molecules": float(sizes[sizes == 1].sum() / n),
        "largest_scaffold_size": int(sizes.max()),
        "largest_scaffold_share_of_train": float(sizes.max() / n),
        "n_scaffolds_size_ge_10": int((sizes >= 10).sum()),
        "molecules_in_scaffolds_size_ge_10": int(sizes[sizes >= 10].sum()),
        "empty_scaffold": {
            "n_molecules": int(empty),
            "rows": empty_rows,
            "rule": "acyclic molecules all share the single group \"\" (not split into singletons)",
        },
        "multi_component": {
            "n_train_rows_with_dot": int(len(multi)),
            "rule": "scaffold computed on the whole parsed molecule; no salt stripping / largest fragment",
        },
        "valid": {
            "n_molecules": int(len(valid_rows)),
            "n_distinct_scaffolds": int(len(valid_keys)),
            "n_valid_scaffolds_present_in_train": int(len(valid_keys & set(counts))),
            "note": "Valid scaffolds only verify the official split; they never enter the weights",
        },
        "original_equal_weight_total_per_scaffold": {
            "description": "n_s / N_train — the total optimisation weight a scaffold gets under plain per-molecule MSE",
            "min": float(sizes.min() / n),
            "max": float(sizes.max() / n),
            "max_over_min": float(sizes.max() / sizes.min()),
        },
        "balanced_total_per_scaffold": {
            "description": "N_train / K_train — the total optimisation weight under the balanced rule",
            "value": float(n / k),
            "max_over_min": 1.0,
        },
        "contrast": {
            "per_molecule_weight_max_over_min": float(sizes.max() / sizes.min()),
            "gate": dict(C.CONTRAST_GATE),
        },
    }


def weight_gate(dist: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    """INSUFFICIENT_WEIGHT_CONTRAST gate (frozen in PROTOCOL section 4)."""
    ratio = float(dist["contrast"]["per_molecule_weight_max_over_min"])
    ess_frac = float(audit["effective_sample_size_fraction"])
    insufficient = bool(
        ratio < float(C.CONTRAST_GATE["min_weight_ratio"])
        and ess_frac > float(C.CONTRAST_GATE["max_ess_fraction"])
    )
    return {
        "max_over_min_weight": ratio,
        "min_weight_ratio_threshold": float(C.CONTRAST_GATE["min_weight_ratio"]),
        "ess_fraction": ess_frac,
        "max_ess_fraction_threshold": float(C.CONTRAST_GATE["max_ess_fraction"]),
        "rule": "INSUFFICIENT_WEIGHT_CONTRAST iff max/min weight < 1.5 AND ESS/N > 0.9",
        "insufficient_weight_contrast": insufficient,
    }


__all__ = [
    "SFL",
    "balanced_weights",
    "load_official_rows",
    "scaffold_counts",
    "scaffold_distribution",
    "scaffold_of_rows",
    "weight_audit",
    "weight_gate",
]
