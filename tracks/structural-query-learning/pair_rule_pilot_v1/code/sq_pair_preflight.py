"""Frozen preflight checks (round brief section 4, items 1-4 plus smoke hooks).

The checks are written to ``runs/prep/preflight.json`` with explicit errors and
scale numbers.  A failed check stops the stage before any formal worker runs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sq_pair_common as C
import sq_pair_data as PD
import sq_pair_fista as F
import sq_pair_rules as R

import sq_data as D


def _features_signature(features: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "named": dict(features["named"]),
        "s2": {key: int(value["count"]) for key, value in features["s2"].items()},
        "bin_counts": dict(features["bin_counts"]),
        "type_counts": dict(features["type_counts"]),
        "ext_counts": dict(features["ext_counts"]),
    }


def check_small_graphs() -> dict[str, Any]:
    chain = PD.chain_graph(3)
    chain_features = PD.pair_features(chain)
    long_chain = PD.chain_graph(11)
    long_features = PD.pair_features(long_chain)
    isolated = PD.chain_with_isolated(3)
    isolated_features = PD.pair_features(isolated)
    s2_counts = [entry["count"] for entry in chain_features["s2"].values()]
    result = {
        "chain3": {
            "n_atoms": chain.n_atoms,
            "n_bonds": chain.n_bonds,
            "n_components": chain.n_components,
            "cycle_rank": chain.cycle_rank,
            "T": chain_features["checks"]["total_pairs"],
            "H_1": chain_features["bin_counts"]["1"],
            "H_2": chain_features["bin_counts"]["2"],
            "H_unreachable": chain_features["bin_counts"]["unreachable"],
            "S2_total": int(sum(s2_counts)),
        },
        "long_chain11": {
            "H_9plus": long_features["bin_counts"]["9+"],
            "H_1": long_features["bin_counts"]["1"],
            "T": long_features["checks"]["total_pairs"],
            "max_finite_distance": int(long_chain.n_atoms - 1),
        },
        "chain3_plus_isolated": {
            "n_atoms": isolated.n_atoms,
            "n_bonds": isolated.n_bonds,
            "n_components": isolated.n_components,
            "H_1": isolated_features["bin_counts"]["1"],
            "H_2": isolated_features["bin_counts"]["2"],
            "H_unreachable": isolated_features["bin_counts"]["unreachable"],
            "T": isolated_features["checks"]["total_pairs"],
            "checks_ok": bool(isolated_features["checks"]["ok"]),
        },
    }
    ok = bool(
        result["chain3"]["n_bonds"] == 2
        and result["chain3"]["n_components"] == 1
        and result["chain3"]["T"] == 3
        and result["chain3"]["H_1"] == 2
        and result["chain3"]["H_2"] == 1
        and result["chain3"]["S2_total"] == 1
        and result["long_chain11"]["H_9plus"] == 3
        and result["chain3_plus_isolated"]["n_components"] == 2
        and result["chain3_plus_isolated"]["H_unreachable"] == 3
        and result["chain3_plus_isolated"]["T"] == 6
        and result["chain3_plus_isolated"]["checks_ok"]
    )
    return {"checks": result, "pass": ok}


def check_relation_positive_control() -> dict[str, Any]:
    g6 = PD.ring_graph(18, oxygen_position=0, nitrogen_position=6)
    g8 = PD.ring_graph(18, oxygen_position=0, nitrogen_position=8)
    f6 = PD.pair_features(g6)
    f8 = PD.pair_features(g8)
    types6 = g6.types
    types8 = g8.types
    tp6 = PD.type_pair(types6, 0, 6)
    tp8 = PD.type_pair(types8, 0, 8)
    tp_name_6 = PD.type_pair_name(tp6)
    tp_name_8 = PD.type_pair_name(tp8)
    if tp_name_6 != tp_name_8:
        raise ValueError("positive control: O/N type pair changed between the two rings")
    slot6 = R.relation_slot_key(tp_name_6, "6")
    slot8 = R.relation_slot_key(tp_name_8, "8")
    slot = {"tp_name": tp_name_6, "bin": "6"}
    null6 = R.null_columns_for_row(
        n_pairs=f6["checks"]["total_pairs"],
        bin_counts=f6["bin_counts"],
        type_counts=f6["type_counts"],
        slots=[slot],
    )
    null8 = R.null_columns_for_row(
        n_pairs=f8["checks"]["total_pairs"],
        bin_counts=f8["bin_counts"],
        type_counts=f8["type_counts"],
        slots=[slot],
    )
    named_equal = f6["named"] == f8["named"]
    bins_equal = f6["bin_counts"] == f8["bin_counts"]
    types_equal = f6["type_counts"] == f8["type_counts"]
    s2_equal = {k: v["count"] for k, v in f6["s2"].items()} == {
        k: v["count"] for k, v in f8["s2"].items()
    }
    count6 = int(f6["ext_counts"].get(slot6, 0))
    count8 = int(f8["ext_counts"].get(slot6, 0))
    check8 = int(f8["ext_counts"].get(slot8, 0))
    result = {
        "graph_a": {"O": 0, "N": 6, "frozen_slot": slot6, "count": count6, "own_bin_count": int(f6["ext_counts"].get(slot6, 0))},
        "graph_b": {"O": 0, "N": 8, "frozen_slot": slot6, "count": count8, "own_bin_count": check8},
        "named_marginal_equal": named_equal,
        "distance_marginal_equal": bins_equal,
        "type_pair_marginal_equal": types_equal,
        "local_s2_equal": s2_equal,
        "null_columns_a": null6.tolist(),
        "null_columns_b": null8.tolist(),
        "null_equal": bool(np.allclose(null6, null8, atol=0.0, rtol=0.0)),
        "manual_vocabulary_slot": slot,
    }
    result["pass"] = bool(
        count6 == 1
        and count8 == 0
        and named_equal
        and bins_equal
        and types_equal
        and s2_equal
        and np.allclose(null6, null8, atol=0.0, rtol=0.0)
    )
    return result


def check_relabel_invariance(payload: Mapping[str, Any], *, n_rows: int = 8) -> dict[str, Any]:
    ids = R.sorted_split_ids(payload, "train")[: int(n_rows)]
    rng = np.random.default_rng(0)
    rows = []
    ok = True
    for row in ids.tolist():
        g = PD.graph_from_payload(payload, int(row))
        perm = rng.permutation(g.n_atoms)
        relabelled = PD.relabel(g, perm)
        a = _features_signature(PD.pair_features(g))
        b = _features_signature(PD.pair_features(relabelled))
        same = a == b
        ok = ok and same
        rows.append(
            {
                "row_id": int(row),
                "n_atoms": int(g.n_atoms),
                "perm_hash": C.sha256_array(np.asarray(perm, dtype=np.int64)),
                "identical": bool(same),
            }
        )
    return {
        "n_rows": int(ids.size),
        "rows": rows,
        "all_identical": bool(ok),
        "pass": bool(ok),
    }


def check_label_scaler(payload: Mapping[str, Any]) -> dict[str, Any]:
    train_ids = R.sorted_split_ids(payload, "train")
    raw = np.asarray(payload["labels"], dtype=np.float64)[train_ids]
    scaler = C.SC.train_target_scaler(raw)
    z = (raw - scaler["mean"]) / scaler["std"]
    roundtrip = z * scaler["std"] + scaler["mean"]
    return {
        "fit_split": "train",
        "n": int(train_ids.size),
        "mean": scaler["mean"],
        "std": scaler["std"],
        "ddof": 0,
        "roundtrip_max_abs_error": float(np.abs(roundtrip - raw).max()),
        "pass": bool(np.abs(roundtrip - raw).max() <= 1e-12),
    }


def check_train_only_scales(payload: Mapping[str, Any], prep_dir: Path) -> dict[str, Any]:
    """A Train-only scale cannot move when Val rows are perturbed."""
    scales = C.read_npz(Path(prep_dir) / "scales.npz")
    train = C.read_npz(Path(prep_dir) / "matrices" / "train.npz")
    valid = C.read_npz(Path(prep_dir) / "matrices" / "valid.npz")
    base_train = train["base_raw"].astype(np.float64)
    scale_train = C.scales_for(base_train)
    combined = np.concatenate([base_train, valid["base_raw"].astype(np.float64)], axis=0)
    combined[base_train.shape[0] :, :] += 7.0
    scale_combined = C.scales_for(combined)
    return {
        "train_only_scale_matches_stored": bool(np.allclose(scale_train, np.asarray(scales["common_scale"]), atol=1e-15, rtol=1e-15)),
        "val_perturbation_changes_combined_scale": bool(np.any(np.abs(scale_combined - scale_train) > 1e-12)),
        "train_rows": int(base_train.shape[0]),
        "valid_rows": int(valid["base_raw"].shape[0]),
        "pass": bool(
            np.allclose(scale_train, np.asarray(scales["common_scale"]), atol=1e-15, rtol=1e-15)
            and np.any(np.abs(scale_combined - scale_train) > 1e-12)
        ),
    }


def check_graph_population(prep_dir: Path) -> dict[str, Any]:
    summary = C.read_json(Path(prep_dir) / "summary.json")
    stats = summary["split_check_stats"]
    total = int(sum(stats[split]["n_graphs"] for split in C.SPLITS))
    maxima = {
        key: max(int(stats[split][key]) for split in C.SPLITS)
        for key in ("max_total_pairs_error", "max_bin_sum_error", "max_type_sum_error", "max_ext_sum_error")
    }
    failures = int(sum(stats[split]["n_graphs_with_check_failure"] for split in C.SPLITS))
    return {
        "n_graphs_checked": total,
        "expected": C.EXPECTED_TOTAL_ROWS,
        "maxima": maxima,
        "n_graphs_with_check_failure": failures,
        "per_split": stats,
        "pass": bool(total == C.EXPECTED_TOTAL_ROWS and failures == 0 and all(v == 0 for v in maxima.values())),
    }


def check_identity(payload: Mapping[str, Any], prep_dir: Path, bundle_identity: Mapping[str, Any]) -> dict[str, Any]:
    identity = R.verify_identity(payload)
    scaler = check_label_scaler(payload)
    train_only = check_train_only_scales(payload, prep_dir)
    labels_ok = True
    for split in C.SPLITS:
        ids = R.sorted_split_ids(payload, split)
        if identity["split_label_sha256"][split] != C.sha256_array(
            np.asarray(payload["labels"], dtype=np.float64)[ids]
        ):
            labels_ok = False
    payload_sizes = {split: int(R.sorted_split_ids(payload, split).size) for split in C.SPLITS}
    result = {
        "cache_sha256_expected": C.EXPECTED_CACHE_SHA256,
        "cache_sha256_actual": {
            "graph": identity["graph_cache_sha256"],
            "lipo_raw": identity["lipo_raw_sha256"],
        },
        "split_sizes": payload_sizes,
        "split_row_id_sha256": identity["split_row_id_sha256"],
        "label_hash_all_rows": identity["label_hash_all_rows"],
        "row_partition_ok": identity["row_partition_ok"],
        "labels_item_by_item_ok": labels_ok,
        "bundle": bundle_identity,
        "label_scaler": scaler,
        "train_only_scales": train_only,
        "bond_validation": identity["bond_validation"],
        "atom_validation": identity["atom_validation"],
    }
    result["pass"] = bool(
        payload_sizes == C.EXPECTED_SPLIT_SIZES
        and identity["row_partition_ok"]
        and labels_ok
        and scaler["pass"]
        and train_only["pass"]
        and bundle_identity.get("scaler_match")
        and bundle_identity.get("targets_match_rows")
        and identity["bond_validation"]["valid"]
    )
    return result


def run_preflight(
    *,
    payload: Mapping[str, Any],
    prep_dir: Path,
    bundle_identity: Mapping[str, Any],
    mlp_smoke: Mapping[str, Any] | None,
    ledger=None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    checks: dict[str, Any] = {}
    checks["1_row_and_split_identity"] = check_identity(payload, prep_dir, bundle_identity)
    checks["2_small_graphs"] = check_small_graphs()
    checks["2_graph_population"] = check_graph_population(prep_dir)
    checks["2_relabel_invariance"] = check_relabel_invariance(payload)
    checks["3_relation_positive_control"] = check_relation_positive_control()
    checks["4_null_hypergeom"] = R.check_null_hypergeom()
    checks["4_proximal_closed_form"] = F.check_proximal_closed_form()
    checks["4_bias_and_budget"] = F.check_bias_and_budget()
    vocab = C.read_json(Path(prep_dir) / "rules_vocabulary.json")
    checks["4_real_coverage"] = {
        "coverage": vocab["coverage"],
        "pass": True,
    }
    if mlp_smoke is not None:
        checks["5_mlp_smoke"] = dict(mlp_smoke)
    errors = [
        {"check": name, "detail": payload_check}
        for name, payload_check in checks.items()
        if isinstance(payload_check, Mapping) and not bool(payload_check.get("pass", False))
    ]
    result = {
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "checks": checks,
        "errors": errors,
        "n_checks": len(checks),
        "n_failed": len(errors),
        "pass": bool(not errors),
        "seconds": time.perf_counter() - t0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scale": {
            "n_graphs": C.EXPECTED_TOTAL_ROWS,
            "n_train": C.EXPECTED_SPLIT_SIZES["train"],
            "max_abs_error_observed": {
                "label_roundtrip": checks["1_row_and_split_identity"]["label_scaler"]["roundtrip_max_abs_error"],
                "pair_total": checks["2_graph_population"]["maxima"]["max_total_pairs_error"],
                "bin_sum": checks["2_graph_population"]["maxima"]["max_bin_sum_error"],
                "type_sum": checks["2_graph_population"]["maxima"]["max_type_sum_error"],
                "ext_sum": checks["2_graph_population"]["maxima"]["max_ext_sum_error"],
                "positive_control_null": float(
                    np.abs(
                        np.asarray(checks["3_relation_positive_control"]["null_columns_a"])
                        - np.asarray(checks["3_relation_positive_control"]["null_columns_b"])
                    ).max()
                ),
                "proximal": checks["4_proximal_closed_form"]["max_abs_error"],
            },
        },
    }
    C.write_json(Path(prep_dir) / "preflight.json", result)
    if ledger is not None:
        ledger.record("preflight", result["seconds"], pass_=result["pass"], n_failed=result["n_failed"])
    if not result["pass"]:
        raise RuntimeError(f"preflight failed: {errors}")
    return result
