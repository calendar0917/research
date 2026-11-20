"""Connect the frozen structure catalogue to labels and frozen predictions.

This module reads labels/predictions only after verifying the structure freeze
written by ``tt_catalogue.py``.  It never trains, selects a checkpoint, changes
a view/aggregate/fallback/dictionary/alpha/projector/split or redefines a pair.
All Test numbers are DEVELOPMENT_TEST evaluation only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tt_common as TC  # noqa: E402

CELL_ORDER = (
    "Train-Train", "Train-Valid", "Train-Test",
    "Valid-Valid", "Valid-Test", "Test-Test",
)
SET_ORDER = ("E_T", "E_HH_T", "E_STRICT", "E_STRICT_C_ONLY")
MODELS = TC.MODELS
N_BOOT = 2000
BOOT_SEED = 1703


# ---------------------------------------------------------------------------
# loading and identity checks
# ---------------------------------------------------------------------------


def _cast_pair(row: Mapping[str, str]) -> dict[str, Any]:
    int_fields = (
        "row_a", "row_b", "row_lo", "row_hi", "component_id", "core_uid",
        "cut_id_a", "cut_id_b", "core_size", "cut_bond_type", "cut_bond_stereo",
        "cut_bond_conjugated",
        "cut_edge_a", "cut_edge_b", "s_size_a", "s_size_b", "max_s_size",
        "stereo_tokens_a", "stereo_tokens_b", "old_stereo_ok_both",
        "n_candidate_definitions", "core_seen_train_full", "core_seen_train_c_only",
        "transform_seen_train_other_component", "transform_train_other_components",
        "s_in_inventory_a", "s_in_inventory_b", "shared_selected_core_view",
        "is_train_a", "is_train_b", "is_test_a", "is_test_b", "is_valid_a", "is_valid_b",
        "e_t", "e_hh_t", "e_strict_full", "e_strict_c_only",
    )
    out: dict[str, Any] = dict(row)
    for key in int_fields:
        out[key] = int(row[key])
    return out


def load_catalogue() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    freeze_path = TC.STRUCTURE_DIR / "structure_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("structure_freeze.json missing; run tt_catalogue.py first")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for name, expected in freeze["files"].items():
        path = TC.STRUCTURE_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"structure file missing: {path}")
        actual = TC.sha256_file(path)
        if actual != expected:
            raise ValueError(f"structure freeze mismatch for {name}: {actual} != {expected}")
    if TC.sha256_json(freeze["files"]) != freeze["structure_hash"]:
        raise ValueError("structure_hash does not match the file hash map")
    if TC.sha256_file(TC.CONFIG_PATH) != freeze["config_sha256"]:
        raise ValueError("config changed after the structure freeze")
    if TC.sha256_file(TC.PROTOCOL_PATH) != freeze["protocol_sha256"]:
        raise ValueError("PROTOCOL.md changed after the structure freeze")
    rows = [
        _cast_pair(r)
        for r in csv.DictReader((TC.STRUCTURE_DIR / "catalogue_pairs.csv").open(newline="", encoding="utf-8"))
    ]
    if len(rows) != freeze["n_pairs"]:
        raise ValueError("catalogue length differs from the freeze")
    return rows, freeze


def input_hashes() -> dict[str, str]:
    paths = {
        "official_cache": TC.RC.DATA_DIR / "ogbg_mollipo_sq-pilot-cache-v1.npz",
        "fragment_cache": TC.RC.DATA_DIR / "ogbg_mollipo_rd_frag_v1.npz",
        "RESP_predictions": TC.NATIVE_PREDICTION["RESP"],
        "ADD_predictions": TC.NATIVE_PREDICTION["ADD"],
        "JOINT_predictions": TC.NATIVE_PREDICTION["JOINT"],
        "dictionary_predictions": TC.DICT_PREDICTION,
        "old_pairs_csv": TC.OLD_PAIRS_CSV,
        "old_pair_mappings": TC.OLD_MAPPINGS_JSON,
        "old_followup_main_table": TC.FU_DIR / "main_table.csv",
        "old_explanation_full_metrics": TC.EC_RESULTS / "projection_full_f_metrics.csv",
        "old_RESP_checkpoint": TC.S0_RESULTS / "lipo_RESP" / "best.pt",
        "old_ADD_checkpoint": TC.S0_RESULTS / "lipo_ADD" / "best.pt",
        "old_JOINT_checkpoint": TC.FU_RESULTS / "lipo_JOINT" / "best.pt",
    }
    return {name: TC.sha256_file(path) for name, path in paths.items() if Path(path).is_file()}


# ---------------------------------------------------------------------------
# support / scope helpers
# ---------------------------------------------------------------------------


def _scope_indices(pairs: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[int]]]:
    scopes: list[tuple[str, list[int]]] = [
        ("ALL", list(range(len(pairs))))
    ]
    for cell in CELL_ORDER:
        scopes.append((cell, [i for i, p in enumerate(pairs) if str(p["cell"]) == cell]))
    scopes.append(("E_T", [i for i, p in enumerate(pairs) if int(p["e_t"])]))
    scopes.append(("E_HH_T", [i for i, p in enumerate(pairs) if int(p["e_hh_t"])]))
    scopes.append(("E_STRICT", [i for i, p in enumerate(pairs) if int(p["e_strict_full"])]))
    scopes.append(("E_STRICT_C_ONLY", [i for i, p in enumerate(pairs) if int(p["e_strict_c_only"])]))
    return scopes


def _support_stats(pairs: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> dict[str, Any]:
    rows: dict[int, str] = {}
    comps: set[int] = set()
    cores: set[int] = set()
    transforms: set[str] = set()
    shared = 0
    both_inv = 0
    one_inv = 0
    none_inv = 0
    for i in indices:
        p = pairs[i]
        a, b = int(p["row_a"]), int(p["row_b"])
        rows[a] = str(p["split_a"])
        rows[b] = str(p["split_b"])
        comps.add(int(p["component_id"]))
        cores.add(int(p["core_uid"]))
        transforms.add(str(p["transform_id"]))
        shared += int(p["shared_selected_core_view"])
        ia = int(p["s_in_inventory_a"])
        ib = int(p["s_in_inventory_b"])
        if ia and ib:
            both_inv += 1
        elif ia or ib:
            one_inv += 1
        else:
            none_inv += 1
    n_non_train = sum(1 for split in rows.values() if split != "train")
    return {
        "n_pairs": int(len(indices)),
        "n_molecules": int(len(rows)),
        "n_non_train_molecules": int(n_non_train),
        "n_test_molecules": int(sum(1 for split in rows.values() if split == "test")),
        "n_valid_molecules": int(sum(1 for split in rows.values() if split == "valid")),
        "n_train_molecules": int(len(rows) - n_non_train),
        "n_components": int(len(comps)),
        "n_cores": int(len(cores)),
        "n_transforms": int(len(transforms)),
        "n_pairs_with_shared_selected_core": int(shared),
        "n_pairs_s_both_in_inventory": int(both_inv),
        "n_pairs_s_one_in_inventory": int(one_inv),
        "n_pairs_s_none_in_inventory": int(none_inv),
    }


# ---------------------------------------------------------------------------
# pair table and references
# ---------------------------------------------------------------------------


def build_pair_table(pairs, payload, predictions) -> dict[str, Any]:
    labels = np.asarray(payload["labels"], dtype=np.float64)
    n = len(pairs)
    delta_y = np.empty(n, dtype=np.float64)
    delta_pred: dict[str, np.ndarray] = {}
    for model in MODELS:
        delta_pred[model] = np.empty(n, dtype=np.float64)
    for i, p in enumerate(pairs):
        a, b = int(p["row_a"]), int(p["row_b"])
        delta_y[i] = labels[b] - labels[a]
        for model in MODELS:
            value = predictions[model]["pred"]
            delta_pred[model][i] = float(value[b] - value[a])

    delta_pred[TC.ZERO] = np.zeros(n, dtype=np.float64)

    # TRAIN_TRANSFORM_MEAN: source = Train-Train only; component-equal average,
    # excluding the evaluated pair's entire component.
    source: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for i, p in enumerate(pairs):
        if str(p["split_a"]) == "train" and str(p["split_b"]) == "train":
            source[str(p["transform_id"])][int(p["component_id"])].append(float(delta_y[i]))
    source_means: dict[str, dict[int, float]] = {}
    source_counts: dict[str, dict[int, int]] = {}
    for transform, by_comp in source.items():
        source_means[transform] = {c: float(np.mean(v)) for c, v in by_comp.items()}
        source_counts[transform] = {c: int(len(v)) for c, v in by_comp.items()}

    table = np.full(n, np.nan, dtype=np.float64)
    table_supported = np.zeros(n, dtype=bool)
    table_n_comp = np.zeros(n, dtype=np.int64)
    table_n_pairs = np.zeros(n, dtype=np.int64)
    for i, p in enumerate(pairs):
        transform = str(p["transform_id"])
        comp = int(p["component_id"])
        means = {
            c: m for c, m in source_means.get(transform, {}).items() if int(c) != comp
        }
        if not means:
            continue
        table[i] = float(np.mean(list(means.values())))
        table_supported[i] = True
        table_n_comp[i] = len(means)
        table_n_pairs[i] = int(sum(
            source_counts[transform][c] for c in means
        ))
    delta_pred[TC.TABLE] = table

    errors = {
        model: (delta_pred[model] - delta_y) for model in list(MODELS) + [TC.ZERO, TC.TABLE]
    }
    return {
        "labels": labels,
        "delta_y": delta_y,
        "delta_pred": delta_pred,
        "errors": errors,
        "abs_errors": {m: np.abs(errors[m]) for m in errors},
        "sq_errors": {m: errors[m] ** 2 for m in errors},
        "table_supported": table_supported,
        "table_n_comp": table_n_comp,
        "table_n_pairs": table_n_pairs,
    }


def write_pair_table(pairs, table, path: Path) -> None:
    columns = [
        "pair_id", "row_a", "row_b", "row_lo", "row_hi", "split_a", "split_b", "cell",
        "component_id", "core_id", "core_uid", "core_size", "cut_bond_type", "cut_bond_stereo",
        "cut_bond_conjugated", "s_key_a", "s_key_b", "s_size_a", "s_size_b",
        "max_s_size", "s_bin", "transform_id", "stereo_tokens_a", "stereo_tokens_b",
        "old_stereo_ok_both", "old_pair_status",
        "core_seen_train_full", "core_seen_train_c_only",
        "transform_seen_train_other_component", "transform_train_other_components",
        "s_in_inventory_a", "s_in_inventory_b", "shared_selected_core_view",
        "e_t", "e_hh_t", "e_strict_full", "e_strict_c_only",
        "y_a", "y_b", "delta_y",
        "delta_pred_ZERO", "delta_pred_RESP", "delta_pred_JOINT", "delta_pred_ADD",
        "delta_pred_D_RESPONSE", "delta_pred_D_LATENT", "delta_pred_D_FREQ",
        "delta_pred_TRAIN_TRANSFORM_MEAN", "table_supported",
        "table_source_components", "table_source_pairs",
        "error_ZERO", "error_RESP", "error_JOINT", "error_ADD",
        "error_D_RESPONSE", "error_D_LATENT", "error_D_FREQ",
        "error_TRAIN_TRANSFORM_MEAN",
        "abs_error_ZERO", "abs_error_RESP", "abs_error_JOINT", "abs_error_ADD",
        "abs_error_D_RESPONSE", "abs_error_D_LATENT", "abs_error_D_FREQ",
        "abs_error_TRAIN_TRANSFORM_MEAN",
    ]
    rows = []
    for i, p in enumerate(pairs):
        rec = {k: p.get(k, "") for k in columns if k in p}
        a, b = int(p["row_a"]), int(p["row_b"])
        rec.update({
            "y_a": float(table["labels"][a]),
            "y_b": float(table["labels"][b]),
            "delta_y": float(table["delta_y"][i]),
            "table_supported": int(table["table_supported"][i]),
            "table_source_components": int(table["table_n_comp"][i]),
            "table_source_pairs": int(table["table_n_pairs"][i]),
        })
        for model in list(MODELS) + [TC.ZERO, TC.TABLE]:
            rec[f"delta_pred_{model}"] = (
                float(table["delta_pred"][model][i])
                if np.isfinite(table["delta_pred"][model][i]) else float("nan")
            )
            rec[f"error_{model}"] = (
                float(table["errors"][model][i])
                if np.isfinite(table["errors"][model][i]) else float("nan")
            )
            rec[f"abs_error_{model}"] = (
                float(table["abs_errors"][model][i])
                if np.isfinite(table["abs_errors"][model][i]) else float("nan")
            )
        rows.append(rec)
    TC.write_csv(path, rows, columns)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _pair_mae(abs_errors: np.ndarray, idx: np.ndarray) -> float:
    return float(np.mean(abs_errors[idx])) if idx.size else float("nan")


def _pair_rmse(sq_errors: np.ndarray, idx: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sq_errors[idx]))) if idx.size else float("nan")


def _component_equal_mae(abs_errors: np.ndarray, sq_errors: np.ndarray, components: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    if not idx.size:
        return float("nan"), float("nan")
    _, inverse = np.unique(components[idx], return_inverse=True)
    counts = np.bincount(inverse)
    mae_terms = np.bincount(inverse, weights=abs_errors[idx]) / counts
    mse_terms = np.bincount(inverse, weights=sq_errors[idx]) / counts
    return float(np.mean(mae_terms)), float(np.sqrt(np.mean(mse_terms)))


def _scope_component_means(
    values: np.ndarray, components: np.ndarray, idx: np.ndarray, comp_ids: np.ndarray
) -> np.ndarray:
    local = np.searchsorted(comp_ids, components[idx])
    counts = np.bincount(local, minlength=len(comp_ids))
    sums = np.bincount(local, weights=values[idx], minlength=len(comp_ids))
    return sums / counts


def _metric_rows(
    scope: str,
    indices: list[int],
    models: Sequence[str],
    abs_errors: Mapping[str, np.ndarray],
    sq_errors: Mapping[str, np.ndarray],
    components: np.ndarray,
    support: Mapping[str, Any],
) -> list[dict[str, Any]]:
    idx = np.asarray(indices, dtype=np.int64)
    out = []
    for model in models:
        ce_mae, ce_rmse = _component_equal_mae(
            abs_errors[model], sq_errors[model], components, idx
        )
        out.append({
            "scope": scope,
            "model": model,
            **support,
            "pair_mae": _pair_mae(abs_errors[model], idx),
            "pair_rmse": _pair_rmse(sq_errors[model], idx),
            "component_equal_mae": ce_mae,
            "component_equal_rmse": ce_rmse,
        })
    return out


def metrics_by_scope(pairs, table) -> list[dict[str, Any]]:
    components = np.asarray([int(p["component_id"]) for p in pairs], dtype=np.int64)
    abs_errors = table["abs_errors"]
    sq_errors = table["sq_errors"]
    scopes = _scope_indices(pairs)
    rows = []
    for scope, idx in scopes:
        support = _support_stats(pairs, idx)
        rows.extend(_metric_rows(
            scope, idx, list(MODELS) + [TC.ZERO], abs_errors, sq_errors, components, support
        ))
    return rows


def metrics_table_subset(pairs, table) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    components = np.asarray([int(p["component_id"]) for p in pairs], dtype=np.int64)
    supported = np.flatnonzero(table["table_supported"]).tolist()
    scopes = _scope_indices(pairs)
    rows = []
    coverage = {}
    for scope, scope_idx in scopes:
        idx = [i for i in scope_idx if bool(table["table_supported"][i])]
        support = _support_stats(pairs, idx)
        n_scope = len(scope_idx)
        cov = float(len(idx) / n_scope) if n_scope else float("nan")
        coverage[scope] = {
            "n_pairs_scope": int(n_scope),
            "n_pairs_supported": int(len(idx)),
            "coverage": cov,
            **support,
        }
        if idx:
            abs_errors = {m: table["abs_errors"][m] for m in list(MODELS) + [TC.ZERO, TC.TABLE]}
            sq_errors = {m: table["sq_errors"][m] for m in list(MODELS) + [TC.ZERO, TC.TABLE]}
            base_rows = _metric_rows(
                scope, idx, list(MODELS) + [TC.ZERO, TC.TABLE],
                abs_errors, sq_errors, components, support,
            )
            for row in base_rows:
                row.update({
                    "n_pairs_scope": int(n_scope),
                    "n_pairs_supported": int(len(idx)),
                    "coverage": cov,
                    "table_source_components_min": int(np.min(table["table_n_comp"][idx])),
                    "table_source_components_median": float(np.median(table["table_n_comp"][idx])),
                    "table_source_components_max": int(np.max(table["table_n_comp"][idx])),
                    "table_source_pairs_min": int(np.min(table["table_n_pairs"][idx])),
                    "table_source_pairs_median": float(np.median(table["table_n_pairs"][idx])),
                    "table_source_pairs_max": int(np.max(table["table_n_pairs"][idx])),
                })
            rows.extend(base_rows)
        else:
            for model in list(MODELS) + [TC.ZERO, TC.TABLE]:
                rows.append({
                    "scope": scope, "model": model,
                    "n_pairs_scope": int(n_scope), "n_pairs_supported": 0,
                    "coverage": cov, **support,
                    "pair_mae": float("nan"), "pair_rmse": float("nan"),
                    "component_equal_mae": float("nan"),
                    "component_equal_rmse": float("nan"),
                    "table_source_components_min": 0,
                    "table_source_components_median": float("nan"),
                    "table_source_components_max": 0,
                    "table_source_pairs_min": 0,
                    "table_source_pairs_median": float("nan"),
                    "table_source_pairs_max": 0,
                })
    return rows, coverage


# ---------------------------------------------------------------------------
# bootstrap and paired gains
# ---------------------------------------------------------------------------


def _bootstrap_for_scope(
    scope: str,
    indices: list[int],
    components: np.ndarray,
    abs_errors: Mapping[str, np.ndarray],
    sq_errors: Mapping[str, np.ndarray],
    comparisons: Sequence[tuple[str, str, str]],
    rng: np.random.Generator,
    models: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return [], {"scope": scope, "n_pairs": 0, "n_components": 0, "samples_sha256": None}
    comp_ids = sorted(set(components[idx].tolist()))
    K = len(comp_ids)
    comp_index = {c: i for i, c in enumerate(comp_ids)}
    pair_comp = np.asarray([comp_index[int(components[i])] for i in idx], dtype=np.int64)
    samples = rng.integers(0, K, size=(N_BOOT, K), dtype=np.int64)
    counts = np.zeros((N_BOOT, K), dtype=np.int64)
    for b in range(N_BOOT):
        counts[b] = np.bincount(samples[b], minlength=K)
    weights = counts[:, pair_comp]
    denom = weights.sum(axis=1)
    samples_hash = hashlib.sha256(samples.tobytes()).hexdigest()

    needed = set(models) | {c[1] for c in comparisons} | {c[2] for c in comparisons}
    comp_ids_arr = np.asarray(comp_ids, dtype=np.int64)
    comp_means = {
        m: _scope_component_means(abs_errors[m], components, idx, comp_ids_arr)
        for m in needed
    }
    rows: list[dict[str, Any]] = []
    for name, ref, cand in comparisons:
        gain_pair = abs_errors[ref][idx] - abs_errors[cand][idx]
        pair_gain_draws = (weights @ gain_pair) / denom
        comp_gain_draws = (counts @ (comp_means[ref] - comp_means[cand])) / K
        rmse_ref_draws = np.sqrt((weights @ sq_errors[ref][idx]) / denom)
        rmse_cand_draws = np.sqrt((weights @ sq_errors[cand][idx]) / denom)
        rmse_gain_draws = rmse_ref_draws - rmse_cand_draws
        pair_point = float(np.mean(gain_pair))
        ce_point = float(np.mean(comp_means[ref] - comp_means[cand]))
        rmse_point = float(
            np.sqrt(np.mean(sq_errors[ref][idx])) - np.sqrt(np.mean(sq_errors[cand][idx]))
        )
        ci_ok = K >= 2
        rows.append({
            "scope": scope,
            "comparison": name,
            "reference": ref,
            "candidate": cand,
            "n_pairs": int(idx.size),
            "n_components": int(K),
            "delta_mae": pair_point,
            "delta_mae_ci_low": float(np.percentile(pair_gain_draws, 2.5)) if ci_ok else float("nan"),
            "delta_mae_ci_high": float(np.percentile(pair_gain_draws, 97.5)) if ci_ok else float("nan"),
            "delta_component_equal_mae": ce_point,
            "delta_component_equal_mae_ci_low": float(np.percentile(comp_gain_draws, 2.5)) if ci_ok else float("nan"),
            "delta_component_equal_mae_ci_high": float(np.percentile(comp_gain_draws, 97.5)) if ci_ok else float("nan"),
            "delta_rmse": rmse_point,
            "delta_rmse_ci_low": float(np.percentile(rmse_gain_draws, 2.5)) if ci_ok else float("nan"),
            "delta_rmse_ci_high": float(np.percentile(rmse_gain_draws, 97.5)) if ci_ok else float("nan"),
            "ci_available": int(ci_ok),
            "n_bootstrap": int(N_BOOT),
            "seed": int(BOOT_SEED),
        })
    plan = {
        "scope": scope,
        "n_pairs": int(idx.size),
        "n_components": int(K),
        "samples_sha256": samples_hash,
    }
    return rows, plan


def paired_gains(pairs, table) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    components = np.asarray([int(p["component_id"]) for p in pairs], dtype=np.int64)
    abs_errors = table["abs_errors"]
    sq_errors = table["sq_errors"]
    rng = np.random.default_rng(BOOT_SEED)
    full_comparisons = [
        ("RESP_vs_JOINT", "JOINT", "RESP"),
        ("RESP_vs_ADD", "ADD", "RESP"),
        ("D_RESPONSE_vs_RESP", "RESP", "D_RESPONSE"),
        ("D_RESPONSE_vs_D_LATENT", "D_LATENT", "D_RESPONSE"),
        ("D_RESPONSE_vs_D_FREQ", "D_FREQ", "D_RESPONSE"),
    ]
    table_comparisons = [("RESP_vs_TRAIN_TRANSFORM_MEAN", TC.TABLE, "RESP")]
    rows: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    for scope, idx in _scope_indices(pairs):
        got, plan = _bootstrap_for_scope(
            scope, idx, components, abs_errors, sq_errors, full_comparisons, rng,
            list(MODELS) + [TC.ZERO, TC.TABLE],
        )
        rows.extend(got)
        plans.append(plan)
    # The table reference lives only on its own support subset; this is a
    # separate scope family and is sampled after the full scopes with the same
    # seed-1703 stream.
    for scope, scope_idx in _scope_indices(pairs):
        idx = [i for i in scope_idx if bool(table["table_supported"][i])]
        if not idx:
            plans.append({
                "scope": f"{scope}__table_supported",
                "n_pairs": 0,
                "n_components": 0,
                "samples_sha256": None,
            })
            continue
        got, plan = _bootstrap_for_scope(
            f"{scope}__table_supported", idx, components, abs_errors, sq_errors,
            table_comparisons, rng, list(MODELS) + [TC.ZERO, TC.TABLE],
        )
        rows.extend(got)
        plans.append(plan)
    return rows, plans


# ---------------------------------------------------------------------------
# dictionary fidelity / label table
# ---------------------------------------------------------------------------


def dictionary_delta_table(pairs, table) -> list[dict[str, Any]]:
    components = np.asarray([int(p["component_id"]) for p in pairs], dtype=np.int64)
    rows = []
    for scope, idx in _scope_indices(pairs):
        if not idx:
            continue
        for model in ("RESP",) + tuple(TC.DICTIONARIES):
            delta = table["delta_pred"][model][idx]
            resp = table["delta_pred"]["RESP"][idx]
            true = table["delta_y"][idx]
            rows.append({
                "scope": scope,
                "model": model,
                "n_pairs": int(len(idx)),
                "n_components": int(len(set(components[idx].tolist()))),
                "fidelity_rmse_to_RESP": float(np.sqrt(np.mean((delta - resp) ** 2))),
                "fidelity_mae_to_RESP": float(np.mean(np.abs(delta - resp))),
                "label_rmse_to_delta_y": float(np.sqrt(np.mean((delta - true) ** 2))),
                "label_mae_to_delta_y": float(np.mean(np.abs(delta - true))),
                "is_native_RESP": int(model == "RESP"),
            })
    return rows


# ---------------------------------------------------------------------------
# influence
# ---------------------------------------------------------------------------


def influence_E_T(pairs, payload, table) -> dict[str, Any]:
    official = TC.official_split(payload)
    e_t = [i for i, p in enumerate(pairs) if int(p["e_t"])]
    test_rows = sorted({
        int(p["row_a"]) for i, p in enumerate(pairs) if i in set(e_t) and str(p["split_a"]) == "test"
    } | {
        int(p["row_b"]) for i, p in enumerate(pairs) if i in set(e_t) and str(p["split_b"]) == "test"
    })
    comparisons = {
        "RESP_vs_JOINT": ("JOINT", "RESP"),
        "RESP_vs_ADD": ("ADD", "RESP"),
        "D_RESPONSE_vs_RESP": ("RESP", "D_RESPONSE"),
        "RESP_vs_TRAIN_TRANSFORM_MEAN": (TC.TABLE, "RESP"),
    }

    def gains(indices: list[int]) -> dict[str, float]:
        out = {}
        for name, (ref, cand) in comparisons.items():
            if name == "RESP_vs_TRAIN_TRANSFORM_MEAN":
                sub = [i for i in indices if bool(table["table_supported"][i])]
            else:
                sub = indices
            if not sub:
                out[name] = float("nan")
                continue
            out[name] = float(np.mean(
                table["abs_errors"][ref][sub] - table["abs_errors"][cand][sub]
            ))
        return out

    full = gains(e_t)
    full_comps = len({int(pairs[i]["component_id"]) for i in e_t})
    removed = []
    for row in test_rows:
        remain = [i for i in e_t if row not in (int(pairs[i]["row_a"]), int(pairs[i]["row_b"]))]
        g = gains(remain)
        removed.append({
            "test_row": int(row),
            "split": str(official[row]),
            "n_pairs_removed": int(len(e_t) - len(remain)),
            "n_pairs_remaining": int(len(remain)),
            "n_components_remaining": int(len({int(pairs[i]["component_id"]) for i in remain})),
            **{f"gain_{k}": v for k, v in g.items()},
        })
    summary = {}
    for name in comparisons:
        vals = [r[f"gain_{name}"] for r in removed if math.isfinite(r[f"gain_{name}"])]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            full_val = full[name]
            sign_reversals = int(sum(
                1 for v in vals
                if math.isfinite(full_val) and ((full_val > 0 and v <= 0) or (full_val < 0 and v >= 0) or (full_val == 0 and v != 0))
            ))
            worst = removed[int(np.argmin([r[f"gain_{name}"] if math.isfinite(r[f"gain_{name}"]) else float("inf") for r in removed]))]
            best = removed[int(np.argmax([r[f"gain_{name}"] if math.isfinite(r[f"gain_{name}"]) else -float("inf") for r in removed]))]
            summary[name] = {
                "full_gain": float(full_val),
                "min_gain": float(np.min(arr)),
                "max_gain": float(np.max(arr)),
                "n_sign_reversals": sign_reversals,
                "most_adverse_test_row": int(worst["test_row"]),
                "most_adverse_gain": float(worst[f"gain_{name}"]),
                "most_favourable_test_row": int(best["test_row"]),
                "most_favourable_gain": float(best[f"gain_{name}"]),
            }
        else:
            summary[name] = {"full_gain": float(full[name])}
    return {
        "scope": "E_T",
        "n_pairs": int(len(e_t)),
        "n_components": int(full_comps),
        "n_test_molecules": int(len(test_rows)),
        "full_gains": {k: float(v) for k, v in full.items()},
        "removed": removed,
        "summary": summary,
        "note": ("one Test molecule and all its E_T pairs are removed at a time; "
                 "this is an influence check on the fixed gains, not a new subpopulation"),
    }


# ---------------------------------------------------------------------------
# frozen full-molecule identity check
# ---------------------------------------------------------------------------


def _expected_full_metrics() -> dict[tuple[str, str], dict[str, float]]:
    expected: dict[tuple[str, str], dict[str, float]] = {}
    with (TC.FU_DIR / "main_table.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arm = str(row["arm"])
            if str(row["state"]) != "best" or arm not in ("RESP", "ADD", "JOINT"):
                continue
            for split in TC.SPLITS:
                expected[(arm, split)] = {
                    "rmse": float(row[f"{split}_rmse_raw"]),
                    "mae": float(row[f"{split}_mae_raw"]),
                    "source": "response_dictionary_lipo_followup_s0_v1/main_table.csv",
                }
    with (TC.EC_RESULTS / "projection_full_f_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row["version"]) != "HP_CERTIFIED":
                continue
            name = str(row["dictionary"])
            if name not in TC.DICTIONARIES:
                continue
            split = str(row["split"])
            expected[(name, split)] = {
                "rmse": float(row["rmse_raw"]),
                "mae": float(row["mae_raw"]),
                "source": "response_dictionary_explanation_check_s0_v1/results/projection_full_f_metrics.csv",
            }
    return expected


def frozen_full_metrics(payload, predictions) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = np.asarray(payload["labels"], dtype=np.float64)
    expected = _expected_full_metrics()
    rows = []
    max_diff = 0.0
    all_pass = True
    for model in list(MODELS):
        pred = predictions[model]["pred"]
        for split in TC.SPLITS:
            idx = np.asarray(payload[f"{split}_idx"], dtype=np.int64)
            r = TC.rmse(pred[idx], labels[idx])
            m = TC.mae(pred[idx], labels[idx])
            exp = expected[(model, split)]
            dr = abs(r - exp["rmse"])
            dm = abs(m - exp["mae"])
            max_diff = max(max_diff, dr, dm)
            passed = dr <= 1e-5 and dm <= 1e-5
            all_pass = all_pass and passed
            rows.append({
                "model": model,
                "split": split,
                "rmse": r,
                "mae": m,
                "expected_rmse": exp["rmse"],
                "expected_mae": exp["mae"],
                "abs_diff_rmse": dr,
                "abs_diff_mae": dm,
                "pass_1e-5": int(passed),
                "expected_source": exp["source"],
            })
    checks = {
        "all_pass_1e-5": int(all_pass),
        "max_abs_diff": float(max_diff),
        "n_checks": int(len(rows)),
        "models": list(MODELS),
        "splits": list(TC.SPLITS),
        "prediction_epochs": {m: predictions[m]["epoch"] for m in MODELS},
        "prediction_hashes": {m: predictions[m]["sha256"] for m in MODELS},
        "prediction_sources": {m: predictions[m]["path"] for m in MODELS},
        "dictionary_version": TC.DICT_VERSION,
    }
    return rows, checks


# ---------------------------------------------------------------------------
# summary / entry
# ---------------------------------------------------------------------------


def evaluate() -> dict[str, Any]:
    pairs, freeze = load_catalogue()
    payload = TC.load_payload()
    predictions = TC.prediction_arrays()
    table = build_pair_table(pairs, payload, predictions)

    TC.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_pair_table(pairs, table, TC.ANALYSIS_DIR / "pairs.csv")

    rows = metrics_by_scope(pairs, table)
    TC.write_csv(TC.ANALYSIS_DIR / "metrics_by_scope.csv", rows)
    table_rows, coverage = metrics_table_subset(pairs, table)
    TC.write_csv(TC.ANALYSIS_DIR / "metrics_table_mean_subset.csv", table_rows)
    gain_rows, bootstrap_plan = paired_gains(pairs, table)
    TC.write_csv(TC.ANALYSIS_DIR / "paired_gains.csv", gain_rows)
    TC.write_json(TC.ANALYSIS_DIR / "bootstrap_plan.json", {
        "n_bootstrap": int(N_BOOT),
        "seed": int(BOOT_SEED),
        "method": ("component bootstrap; pair-equal metrics keep all pairs of sampled "
                   "components; component-equal metrics average sampled component means; "
                   "one draw matrix per scope shared by every model and comparison"),
        "scope_order": [p["scope"] for p in bootstrap_plan],
        "plans": bootstrap_plan,
    })
    dict_rows = dictionary_delta_table(pairs, table)
    TC.write_csv(TC.ANALYSIS_DIR / "dictionary_delta_table.csv", dict_rows)
    influence = influence_E_T(pairs, payload, table)
    TC.write_json(TC.ANALYSIS_DIR / "influence_E_T.json", influence)
    full_rows, full_checks = frozen_full_metrics(payload, predictions)
    TC.write_csv(TC.ANALYSIS_DIR / "frozen_full_metrics.csv", full_rows)
    TC.write_json(TC.ANALYSIS_DIR / "frozen_checks.json", full_checks)

    # convenience extracts
    n_pairs = len(pairs)
    def _subset_table(selected: list[int]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in table.items():
            if isinstance(value, dict):
                out[key] = {m: a[selected] for m, a in value.items()}
            elif isinstance(value, np.ndarray) and value.shape[:1] == (n_pairs,):
                out[key] = value[selected]
            else:
                out[key] = value
        return out
    for name, flag in (("E_HH_T", "e_hh_t"), ("E_STRICT", "e_strict_full")):
        selected = [i for i, p in enumerate(pairs) if int(p[flag])]
        if selected:
            write_pair_table(
                [pairs[i] for i in selected],
                _subset_table(selected),
                TC.ANALYSIS_DIR / f"{name}_pairs.csv",
            )
        else:
            TC.write_csv(TC.ANALYSIS_DIR / f"{name}_pairs.csv", [], [])

    support = json.loads((TC.STRUCTURE_DIR / "support.json").read_text(encoding="utf-8"))
    summary = {
        "round_id": TC.load_config()["round_id"],
        "protocol_hash": TC.protocol_hash(),
        "structure_freeze": freeze,
        "inputs": input_hashes(),
        "frozen_full_checks": full_checks,
        "counts": support["counts"],
        "scopes": support["scopes"],
        "table_mean_coverage": coverage,
        "paired_gains": gain_rows,
        "influence_E_T": influence,
        "frozen_full_metrics": full_rows,
        "test_policy": "DEVELOPMENT_TEST",
        "test_use": "evaluation only; no checkpoint, dictionary, alpha, subset or reference was selected by Test",
        "note": ("This round reports descriptive edit-response evidence. It does not "
                 "establish a chemical task law; Train-Train pairs are not held-out evidence."),
    }
    TC.write_json(TC.ANALYSIS_DIR / "summary.json", summary)
    return {
        "summary": summary,
        "metrics_rows": rows,
        "table_rows": table_rows,
        "gain_rows": gain_rows,
        "dict_rows": dict_rows,
        "full_checks": full_checks,
        "n_pairs": len(pairs),
        "structure_hash": freeze["structure_hash"],
    }


def main(argv=None) -> int:
    result = evaluate()
    print(json.dumps({
        "n_pairs": result["n_pairs"],
        "structure_hash": result["structure_hash"],
        "frozen_full_all_pass": result["full_checks"]["all_pass_1e-5"],
        "frozen_full_max_abs_diff": result["full_checks"]["max_abs_diff"],
        "E_T": result["summary"]["scopes"]["E_T"],
        "E_HH_T": result["summary"]["scopes"]["E_HH_T"],
        "E_STRICT": result["summary"]["scopes"]["E_STRICT"],
        "table_E_T_coverage": result["summary"]["table_mean_coverage"].get("E_T"),
        "influence_E_T": result["summary"]["influence_E_T"]["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
