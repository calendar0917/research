"""Rule-library construction and the frozen raw feature matrices.

Pipeline (all Train-only decisions happen before any Val/Test label is read):

1. fixed B query bank (``sq_bank`` default cap 512 / min 8 Train molecules);
2. typed local two-bond paths (S2), kept if >=16 Train molecules support them
   and their Train counts are non-constant, capped at 256;
3. actual anchor-pair relation slots ``(type_pair, finite_distance_bin)`` with
   bins 3..9+, kept by the same support / non-constancy rules and capped at 512;
4. type-pair totals ``C_t`` for the ``t`` appearing in the final slots and the
   whole-graph distance marginals ``H_d`` for all ten bins;
5. the NULL counterpart of every stored slot from ``C_t, H_d, T`` only.

The final column order is: four named counts, B bank (original order), S2
(canonical-string order), type-pair totals (canonical-string order), distance
marginals (bin order), then the locked relation slots (support desc, key asc).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import hypergeom

import sq_pair_common as C

import sq_ast as A
import sq_bank as B
import sq_data as D

import sq_pair_data as PD

relation_slot_key = PD.relation_slot_key


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def sorted_split_ids(payload: Mapping[str, Any], split: str) -> np.ndarray:
    ids = np.asarray(D.split_indices(payload, split), dtype=np.int64)
    return np.sort(ids, kind="stable")


def verify_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Cache fingerprint, split partition and per-row label identity."""
    graph_path = D.cache_path(C.DATASET)
    graph_sha = C.sha256_file(graph_path)
    lipo_path = D.lipo_raw_path(C.DATASET)
    lipo_sha = C.sha256_file(lipo_path)
    if graph_sha != C.EXPECTED_CACHE_SHA256["graph"]:
        raise ValueError(f"graph cache sha256 {graph_sha} != pinned {C.EXPECTED_CACHE_SHA256['graph']}")
    if lipo_sha != C.EXPECTED_CACHE_SHA256["lipo_raw"]:
        raise ValueError(f"lipo raw sha256 {lipo_sha} != pinned {C.EXPECTED_CACHE_SHA256['lipo_raw']}")
    sizes = {split: int(sorted_split_ids(payload, split).size) for split in C.SPLITS}
    if sizes != C.EXPECTED_SPLIT_SIZES:
        raise ValueError(f"official split sizes {sizes} != {C.EXPECTED_SPLIT_SIZES}")
    ids = {split: sorted_split_ids(payload, split) for split in C.SPLITS}
    union = np.concatenate([ids[split] for split in C.SPLITS])
    if union.size != C.EXPECTED_TOTAL_ROWS:
        raise ValueError(f"split union has {union.size} rows, expected {C.EXPECTED_TOTAL_ROWS}")
    if not np.array_equal(np.sort(union), np.arange(C.EXPECTED_TOTAL_ROWS, dtype=np.int64)):
        raise ValueError("official split partition does not cover 0..4199 exactly once")
    for a in range(len(C.SPLITS)):
        for b in range(a + 1, len(C.SPLITS)):
            if np.intersect1d(ids[C.SPLITS[a]], ids[C.SPLITS[b]]).size:
                raise ValueError(f"splits {C.SPLITS[a]} and {C.SPLITS[b]} overlap")
    labels = np.asarray(payload["labels"], dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("non-finite raw label")
    decoding = PD.ogb_decoding()
    bond_validation = PD.validate_bond_columns(np.asarray(payload["bond_feat"]), decoding)
    atom_cols = np.asarray(payload["atom_feat"])
    atom_validation = {
        "maxima": [int(atom_cols[:, j].max()) for j in (0, 3, 7, 8)],
        "limits": [
            len(decoding["atom"]["atomic_num"]),
            len(decoding["atom"]["formal_charge"]),
            len(decoding["atom"]["is_aromatic"]),
            len(decoding["atom"]["is_in_ring"]),
        ],
    }
    if any(atom_validation["maxima"][k] >= atom_validation["limits"][k] for k in range(4)):
        raise ValueError(f"atom category index outside the OGB decoding table: {atom_validation}")
    return {
        "dataset": C.DATASET,
        "graph_cache_path": str(graph_path),
        "graph_cache_sha256": graph_sha,
        "lipo_raw_path": str(lipo_path),
        "lipo_raw_sha256": lipo_sha,
        "split_sizes": sizes,
        "split_row_id_sha256": {split: C.sha256_array(ids[split]) for split in C.SPLITS},
        "split_label_sha256": {
            split: C.sha256_array(labels[ids[split]]) for split in C.SPLITS
        },
        "label_hash_all_rows": C.sha256_array(labels),
        "row_partition_ok": True,
        "n_rows": int(union.size),
        "ogb_decoding": decoding,
        "bond_validation": bond_validation,
        "atom_validation": atom_validation,
        "bond_column_note": (
            "cache column names are reused from sq_data, whose third bond label says "
            "bond_is_in_ring; the OGB value is the conjugation code "
            "(possible_bond_is_conjugated_list)"
        ),
    }


def load_bundle_identity() -> dict[str, Any]:
    """Use ``gr_data.bundle`` as the single identity source (prepare only)."""
    import gr_common as GC
    import gr_data as GD

    b = GD.bundle(C.DATASET, verify=True)
    payload = b.payload
    decoder = PD.ogb_decoding()
    out: dict[str, Any] = {
        "bundle_dataset": b.dataset,
        "bundle_scaler": {k: b.scaler[k] for k in ("mean", "std", "n")},
        "bundle_n_train": int(b.caches["train"]["n_molecules"]),
        "bundle_n_valid": int(b.caches["valid"]["n_molecules"]),
        "bundle_n_test": int(b.caches["test"]["n_molecules"]),
        "bundle_split_sizes": b.split_sizes,
        "bundle_graph_cache_sha256": payload.get("cache_sha256"),
        "bundle_lipo_raw_sha256": b.lipo_raw.get("cache_sha256"),
    }
    sq_scaler = C.SC.train_target_scaler(np.asarray(payload["labels"])[sorted_split_ids(payload, "train")])
    if abs(sq_scaler["mean"] - b.scaler["mean"]) > 1e-12 or abs(sq_scaler["std"] - b.scaler["std"]) > 1e-12:
        raise ValueError("bundle target scaler differs from the Train-only sq scaler")
    out["sq_scaler"] = {k: sq_scaler[k] for k in ("mean", "std", "n")}
    out["scaler_match"] = True
    for split in C.SPLITS:
        order, ids = b.row_ids_sorted(split)
        targets = np.asarray(b.targets_raw[split], dtype=np.float64)[order]
        expected = np.asarray(payload["labels"], dtype=np.float64)[ids]
        if not np.array_equal(targets, expected):
            raise ValueError(f"bundle {split} targets do not match the official row labels")
        if not np.array_equal(ids, sorted_split_ids(payload, split)):
            raise ValueError(f"bundle {split} row ids are not the ascending official ids")
    out["targets_match_rows"] = True
    out["decoding_double_bond_index"] = decoder["double_bond_index"]
    out["recipe_constants"] = {
        "C1_recipe": GC.RECIPES[GC.RECIPE_ORDER[1]] if False else GC.RECIPES["Rf"],
        "epochs": C.EPOCHS,
        "batch_size": C.BATCH_SIZE,
        "eval_batch_size": C.EVAL_BATCH_SIZE,
        "dropout": GC.DROPOUT,
    }
    return out


def load_payload_verified() -> dict[str, Any]:
    graph_path = D.cache_path(C.DATASET)
    payload = D.load_cache(C.DATASET)
    if C.sha256_file(graph_path) != C.EXPECTED_CACHE_SHA256["graph"]:
        raise ValueError("graph cache checksum mismatch")
    return payload


# ---------------------------------------------------------------------------
# B bank
# ---------------------------------------------------------------------------


def ast_to_json(ast: tuple) -> dict[str, Any]:
    kind = ast[0]
    if kind in A.NODE_KINDS:
        if kind == "ATOM_EQ":
            return {"op": "ATOM_EQ", "atom_column": int(ast[1]), "value": int(ast[2])}
        if kind in ("DEG_EQ", "DEG_GE", "MIN_RING_EQ", "MIN_RING_GE"):
            return {"op": kind, "value": int(ast[1])}
        return {"op": kind}
    if kind == "AND":
        return {"op": "AND", "children": [ast_to_json(child) for child in ast[1:]]}
    if kind == "NBR":
        return {
            "op": "NBR",
            "parent": ast_to_json(ast[1]),
            "bond": _bond_to_json(ast[2]),
            "m": int(ast[3]),
        }
    if kind == "BRANCH":
        return {
            "op": "BRANCH",
            "parent_p": ast_to_json(ast[1]),
            "parent_q": ast_to_json(ast[2]),
            "bond_p": _bond_to_json(ast[3]),
            "bond_q": _bond_to_json(ast[4]),
            "distinct_neighbours_required": True,
        }
    raise ValueError(f"unknown AST kind {kind!r}")


def _bond_to_json(bond: tuple) -> dict[str, Any]:
    if bond[0] == "ANY":
        return {"op": "ANY"}
    return {"op": "BOND_EQ", "bond_column": int(bond[1]), "value": int(bond[2])}


def build_b_bank(payload: Mapping[str, Any]) -> dict[str, Any]:
    packed_train = D.packed_split(payload, "train")
    vocabulary = D.train_vocabulary(payload)
    records, stats = B.build_base_bank(
        packed_train,
        vocabulary,
        cap=C.BANK_CAP,
        min_molecules=C.BANK_MIN_MOLECULES,
    )
    return {
        "records": records,
        "stats": stats,
        "vocabulary": vocabulary,
        "packed_train": packed_train,
    }


def b_counts_for_split(payload: Mapping[str, Any], split: str, records: Sequence[B.QueryRecord]) -> tuple[np.ndarray, np.ndarray]:
    packed = D.packed_split(payload, split)
    masks = B.MaskCache()
    if records:
        blocks = [
            np.asarray(B.query_columns(packed, record.ast, cache=masks)[:, 0], dtype=np.int32)
            for record in records
        ]
        counts = np.stack(blocks, axis=1)
    else:
        counts = np.zeros((packed.n_graphs, 0), dtype=np.int32)
    order = np.argsort(packed.row_ids, kind="stable")
    ids = np.asarray(packed.row_ids, dtype=np.int64)[order]
    return counts[order], ids


# ---------------------------------------------------------------------------
# candidate collection (Train only)
# ---------------------------------------------------------------------------


def collect_train_candidates(payload: Mapping[str, Any]) -> dict[str, Any]:
    ids = sorted_split_ids(payload, "train")
    n_train = int(ids.size)
    s2_vectors: dict[str, np.ndarray] = {}
    s2_meta: dict[str, dict[str, Any]] = {}
    ext_vectors: dict[str, np.ndarray] = {}
    ext_meta: dict[str, dict[str, Any]] = {}
    check_failures: list[dict[str, Any]] = []
    for local, row in enumerate(ids.tolist()):
        g = PD.graph_from_payload(payload, int(row))
        features = PD.pair_features(g)
        if not features["checks"]["ok"]:
            check_failures.append({"row_id": int(row), "checks": features["checks"]})
            if len(check_failures) > 5:
                raise ValueError(f"pair-count check failed on Train rows: {check_failures[:5]}")
        for name, entry in features["s2"].items():
            vector = s2_vectors.get(name)
            if vector is None:
                vector = np.zeros(n_train, dtype=np.int16)
                s2_vectors[name] = vector
                s2_meta[name] = entry["meta"]
            vector[local] = int(entry["count"])
        for key, count in features["ext_counts"].items():
            vector = ext_vectors.get(key)
            if vector is None:
                vector = np.zeros(n_train, dtype=np.int16)
                ext_vectors[key] = vector
                ext_meta[key] = {"key": key, **features["type_meta"].get(key.rpartition("|")[0], {})}
            vector[local] = int(count)
    if check_failures:
        raise ValueError(f"pair-count checks failed: {check_failures[:5]}")
    return {
        "n_train": n_train,
        "s2_vectors": s2_vectors,
        "s2_meta": s2_meta,
        "ext_vectors": ext_vectors,
        "ext_meta": ext_meta,
    }


def _candidate_row(vector: np.ndarray) -> dict[str, Any]:
    support = int((vector > 0).sum())
    return {
        "train_support": support,
        "train_count_min": int(vector.min()) if vector.size else 0,
        "train_count_max": int(vector.max()) if vector.size else 0,
        "train_nonconstant": bool(vector.size and int(vector.min()) != int(vector.max())),
    }


def select_candidates(candidates: Mapping[str, Any]) -> dict[str, Any]:
    s2_rows = []
    for name, vector in candidates["s2_vectors"].items():
        stat = _candidate_row(vector)
        if stat["train_support"] >= C.S2_MIN_MOLECULES and stat["train_nonconstant"]:
            s2_rows.append({"name": name, "meta": candidates["s2_meta"][name], **stat})
    s2_rows.sort(key=lambda row: (-row["train_support"], row["name"]))
    s2_excluded_by_cap = max(0, len(s2_rows) - C.S2_CAP)
    s2_selected = s2_rows[: C.S2_CAP]

    rel_rows = []
    for key, vector in candidates["ext_vectors"].items():
        stat = _candidate_row(vector)
        if stat["train_support"] >= C.RELATION_MIN_MOLECULES and stat["train_nonconstant"]:
            tp_name, bin_name = key.rpartition("|")[0], key.rpartition("|")[2]
            rel_rows.append(
                {
                    "key": key,
                    "tp_name": tp_name,
                    "bin": bin_name,
                    "meta": candidates["ext_meta"][key],
                    **stat,
                }
            )
    rel_rows.sort(key=lambda row: (-row["train_support"], row["key"]))
    rel_excluded_by_cap = max(0, len(rel_rows) - C.RELATION_CAP)
    rel_selected = rel_rows[: C.RELATION_CAP]
    return {
        "s2_selected": s2_selected,
        "s2_excluded_by_cap": int(s2_excluded_by_cap),
        "n_s2_survivors": int(len(s2_rows)),
        "relation_selected": rel_selected,
        "relation_excluded_by_cap": int(rel_excluded_by_cap),
        "n_relation_survivors": int(len(rel_rows)),
        "s2_min_molecules": C.S2_MIN_MOLECULES,
        "relation_min_molecules": C.RELATION_MIN_MOLECULES,
    }


# ---------------------------------------------------------------------------
# frozen groups
# ---------------------------------------------------------------------------


def build_common_groups(
    *,
    b_records: Sequence[B.QueryRecord],
    s2_selected: Sequence[Mapping[str, Any]],
    relation_selected: Sequence[Mapping[str, Any]],
    decoding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    def add(kind: str, key: str, name: str, meta: Mapping[str, Any]) -> None:
        groups.append(
            {
                "group_index": len(groups),
                "kind": kind,
                "key": str(key),
                "name": str(name),
                "meta": dict(meta),
            }
        )

    for name in C.NAMED_GRAPH_COUNTS:
        add("graph_count", name, name, {"count_name": name})
    for record in b_records:
        add(
            "b_query",
            record.label,
            f"B::{record.label}",
            {
                "label": record.label,
                "ast_canon": A.canon(record.ast),
                "ast_json": ast_to_json(record.ast),
                "train_support": int(record.train_support),
            },
        )
    for row in s2_selected:
        add("s2_path", row["name"], row["name"], row["meta"])
    final_types = sorted({row["tp_name"] for row in relation_selected})
    type_meta: dict[str, dict[str, Any]] = {}
    for row in relation_selected:
        if row["tp_name"] not in type_meta:
            type_meta[row["tp_name"]] = row["meta"]
    for tp_name in final_types:
        meta = type_meta.get(tp_name, {"name": tp_name})
        decoded = _decode_type_pair(decoding, meta)
        add("type_pair_total", tp_name, tp_name, {**meta, "decoded": decoded})
    for bin_name in C.DISTANCE_BINS:
        add("distance_bin", bin_name, f"D::{bin_name}", {"bin": bin_name})
    return groups


def _decode_type_pair(decoding: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key in ("type_a", "type_b"):
        if key in meta:
            out[key] = PD.decode_atom_type(decoding, meta[key])
    return out


def build_relation_groups(
    relation_selected: Sequence[Mapping[str, Any]],
    decoding: Mapping[str, Any],
    *,
    offset: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k, row in enumerate(relation_selected):
        meta = dict(row["meta"])
        meta.update(
            {
                "tp_name": row["tp_name"],
                "distance_bin": row["bin"],
                "train_support": row["train_support"],
                "train_count_min": row["train_count_min"],
                "train_count_max": row["train_count_max"],
                "decoded": _decode_type_pair(decoding, meta),
            }
        )
        out.append(
            {
                "group_index": int(offset + k),
                "slot_index": int(k),
                "kind": "relation_real",
                "key": row["key"],
                "name": f"REL::{row['key']}",
                "tp_name": row["tp_name"],
                "bin": row["bin"],
                "meta": meta,
            }
        )
    return out


# ---------------------------------------------------------------------------
# NULL marginal expectation
# ---------------------------------------------------------------------------


def null_columns_for_row(
    *,
    n_pairs: int,
    bin_counts: Mapping[str, int],
    type_counts: Mapping[str, int],
    slots: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Exact expectations of ``[min(K,1), max(K-1,0)]`` under Hypergeom(T,H_d,C_t)."""
    n_slots = len(slots)
    out = np.zeros((n_slots, 2), dtype=np.float64)
    if n_slots == 0:
        return out
    if n_pairs <= 0:
        return out
    c_arr = np.asarray([int(type_counts.get(slot["tp_name"], 0)) for slot in slots], dtype=np.int64)
    h_arr = np.asarray([int(bin_counts.get(slot["bin"], 0)) for slot in slots], dtype=np.int64)
    valid = (c_arr > 0) & (h_arr > 0)
    mean = np.zeros(n_slots, dtype=np.float64)
    p_hit = np.zeros(n_slots, dtype=np.float64)
    mean[valid] = c_arr[valid].astype(np.float64) * h_arr[valid].astype(np.float64) / float(n_pairs)
    if valid.any():
        p_hit[valid] = np.asarray(
            hypergeom.sf(0, int(n_pairs), h_arr[valid], c_arr[valid]), dtype=np.float64
        )
    second = mean - p_hit
    if np.any(second < -C.NEG_CLAMP_EPS):
        raise ValueError(f"NULL second column is materially negative: {float(second.min())}")
    second = np.where(second < 0.0, 0.0, second)
    if np.any(p_hit < -1e-12) or np.any(p_hit > 1.0 + 1e-12):
        raise ValueError("NULL first column left [0,1]")
    out[:, 0] = p_hit
    out[:, 1] = second
    return out


def check_null_hypergeom() -> dict[str, Any]:
    from scipy.stats import hypergeom as hg

    T, H, C = 4, 2, 2
    p_hit = float(hg.sf(0, T, H, C))
    mean = C * H / T
    # The second basis column is P(K >= 2), i.e. the "subsequent hit" mass.
    repeat = float(hg.pmf(2, T, H, C))
    columns = null_columns_for_row(
        n_pairs=T,
        bin_counts={"6": H},
        type_counts={"TP|0|0": C},
        slots=[{"tp_name": "TP|0|0", "bin": "6"}],
    )
    return {
        "T": T,
        "H_d": H,
        "C_t": C,
        "p_hit_expected": 5.0 / 6.0,
        "p_hit": p_hit,
        "second_column_expected": 1.0 / 6.0,
        "second_column": repeat,
        "mean_K_expected": mean,
        "mean_K": float(C * H / T),
        "columns": columns[0].tolist(),
        "column_sum": float(columns[0].sum()),
        "first_in_unit_interval": bool(0.0 <= columns[0, 0] <= 1.0),
        "sum_equals_mean": bool(abs(float(columns[0].sum()) - mean) <= 1e-12),
        "pass": bool(
            abs(p_hit - 5.0 / 6.0) < 1e-12
            and abs(repeat - 1.0 / 6.0) < 1e-12
            and abs(float(columns[0, 0]) - 5.0 / 6.0) < 1e-12
            and abs(float(columns[0, 1]) - 1.0 / 6.0) < 1e-12
            and abs(float(columns[0].sum()) - mean) <= 1e-12
        ),
    }


# ---------------------------------------------------------------------------
# matrices
# ---------------------------------------------------------------------------


def _common_counts_for_graph(
    *,
    features: Mapping[str, Any],
    b_row: np.ndarray,
    common_groups: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    counts = np.zeros(len(common_groups), dtype=np.int32)
    for k, group in enumerate(common_groups):
        kind = group["kind"]
        key = group["key"]
        if kind == "graph_count":
            value = int(features["named"][key])
        elif kind == "b_query":
            b_index = int(group["meta"]["b_index"])
            value = int(b_row[b_index])
        elif kind == "s2_path":
            entry = features["s2"].get(key)
            value = 0 if entry is None else int(entry["count"])
        elif kind == "type_pair_total":
            value = int(features["type_counts"].get(key, 0))
        elif kind == "distance_bin":
            value = int(features["bin_counts"][key])
        else:
            raise ValueError(f"unknown common group kind {kind!r}")
        counts[k] = value
    return counts


def build_split_matrices(
    *,
    payload: Mapping[str, Any],
    split: str,
    common_groups: Sequence[Mapping[str, Any]],
    relation_slots: Sequence[Mapping[str, Any]],
    b_records: Sequence[B.QueryRecord],
) -> dict[str, Any]:
    ids = sorted_split_ids(payload, split)
    n_rows = int(ids.size)
    base_raw = np.zeros((n_rows, 2 * len(common_groups)), dtype=np.int32)
    real_raw = np.zeros((n_rows, 2 * len(relation_slots)), dtype=np.int32)
    null_raw = np.zeros((n_rows, 2 * len(relation_slots)), dtype=np.float64)
    b_counts, b_ids = b_counts_for_split(payload, split, b_records)
    if not np.array_equal(b_ids, ids):
        raise ValueError(f"{split}: B-count row order does not match ascending official row ids")
    check_stats = {
        "n_graphs": n_rows,
        "max_total_pairs_error": 0,
        "max_bin_sum_error": 0,
        "max_type_sum_error": 0,
        "max_ext_sum_error": 0,
        "n_graphs_with_check_failure": 0,
    }
    for local, row in enumerate(ids.tolist()):
        g = PD.graph_from_payload(payload, int(row))
        features = PD.pair_features(g)
        checks = features["checks"]
        err_total = abs(int(checks["sum_bins"]) - int(checks["total_pairs"]))
        err_type = abs(int(checks["sum_type_pairs"]) - int(checks["total_pairs"]))
        err_ext = abs(int(checks["sum_ext"]) - int(checks["finite_margin"]))
        err_bin = abs(int(checks["sum_bins"]) - int(checks["finite_margin"]) - int(features["bin_counts"]["1"]) - int(features["bin_counts"]["2"]) - int(features["bin_counts"]["unreachable"]))
        check_stats["max_total_pairs_error"] = max(check_stats["max_total_pairs_error"], err_total)
        check_stats["max_bin_sum_error"] = max(check_stats["max_bin_sum_error"], err_bin)
        check_stats["max_type_sum_error"] = max(check_stats["max_type_sum_error"], err_type)
        check_stats["max_ext_sum_error"] = max(check_stats["max_ext_sum_error"], err_ext)
        if not checks["ok"]:
            check_stats["n_graphs_with_check_failure"] += 1
        counts_common = _common_counts_for_graph(
            features=features, b_row=b_counts[local], common_groups=common_groups
        )
        base_raw[local] = C.u_basis(counts_common).reshape(-1).astype(np.int32)
        if relation_slots:
            counts_rel = np.asarray(
                [int(features["ext_counts"].get(slot["key"], 0)) for slot in relation_slots],
                dtype=np.int32,
            )
            real_raw[local] = C.u_basis(counts_rel).reshape(-1).astype(np.int32)
            null_raw[local] = null_columns_for_row(
                n_pairs=int(checks["total_pairs"]),
                bin_counts=features["bin_counts"],
                type_counts=features["type_counts"],
                slots=relation_slots,
            ).reshape(-1)
    if check_stats["n_graphs_with_check_failure"]:
        raise ValueError(f"{split}: structural pair-count checks failed: {check_stats}")
    return {
        "ids": ids,
        "base_raw": base_raw,
        "real_raw": real_raw,
        "null_raw": null_raw,
        "check_stats": check_stats,
        "b_counts": b_counts,
    }


def coverage_counts(real_raw: np.ndarray, null_raw: np.ndarray) -> dict[str, Any]:
    if real_raw.size == 0 or null_raw.size == 0:
        return {
            "n_slots": 0,
            "n_rows": int(real_raw.shape[0]),
            "n_rows_with_extension_difference": 0,
            "max_abs_difference": 0.0,
            "all_zero": True,
        }
    diff = np.abs(real_raw.astype(np.float64) - null_raw)
    row_has = np.any(diff > C.RAW_DIFF_EPS, axis=1)
    return {
        "n_slots": int(real_raw.shape[1] // 2),
        "n_rows": int(real_raw.shape[0]),
        "n_rows_with_extension_difference": int(row_has.sum()),
        "max_abs_difference": float(diff.max()),
        "all_zero": bool(not row_has.any()),
    }


# ---------------------------------------------------------------------------
# full prepare artefact writer
# ---------------------------------------------------------------------------


def build_all(payload: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    out = C.ensure_dir(out_dir)
    matrices_dir = C.ensure_dir(out / "matrices")
    b_bank = build_b_bank(payload)
    b_records = b_bank["records"]
    candidates = collect_train_candidates(payload)
    selected = select_candidates(candidates)
    decoding = PD.ogb_decoding()
    common_groups = build_common_groups(
        b_records=b_records,
        s2_selected=selected["s2_selected"],
        relation_selected=selected["relation_selected"],
        decoding=decoding,
    )
    # attach the B index to the group metadata (stable original bank order)
    b_index = 0
    for group in common_groups:
        if group["kind"] == "b_query":
            group["meta"]["b_index"] = b_index
            b_index += 1
    relation_groups = build_relation_groups(
        selected["relation_selected"], decoding, offset=len(common_groups)
    )

    split_payload: dict[str, Any] = {}
    for split in C.SPLITS:
        built = build_split_matrices(
            payload=payload,
            split=split,
            common_groups=common_groups,
            relation_slots=relation_groups,
            b_records=b_records,
        )
        split_payload[split] = built
        C.save_npz(
            matrices_dir / f"{split}.npz",
            {
                "row_id": built["ids"],
                "y_raw": np.asarray(payload["labels"], dtype=np.float64)[built["ids"]],
                "base_raw": built["base_raw"],
                "real_raw": built["real_raw"],
                "null_raw": built["null_raw"],
            },
        )

    train_base = split_payload["train"]["base_raw"]
    train_real = split_payload["train"]["real_raw"]
    train_null = split_payload["train"]["null_raw"]
    common_scale = C.scales_for(train_base)
    real_scale = C.scales_for(train_real)
    null_scale = C.scales_for(train_null)
    C.save_npz(
        out / "scales.npz",
        {
            "common_scale": common_scale,
            "real_scale": real_scale,
            "null_scale": null_scale,
        },
    )

    train_labels = np.asarray(payload["labels"], dtype=np.float64)[sorted_split_ids(payload, "train")]
    scaler = C.SC.train_target_scaler(train_labels)
    coverage = {
        split: coverage_counts(
            split_payload[split]["real_raw"], split_payload[split]["null_raw"]
        )
        for split in C.SPLITS
    }
    column_order = {
        "common": [
            f"{group['name']}::{suffix}"
            for group in common_groups
            for suffix in ("pos", "excess")
        ],
        "real_extension": [
            f"{group['name']}::{suffix}"
            for group in relation_groups
            for suffix in ("pos", "excess")
        ],
        "null_extension": [
            f"{group['name']}::{suffix}"
            for group in relation_groups
            for suffix in ("p_hit", "mean_minus_p_hit")
        ],
        "n_common_groups": len(common_groups),
        "n_relation_slots": len(relation_groups),
        "n_common_columns": 2 * len(common_groups),
        "n_extension_columns": 2 * len(relation_groups),
    }
    vocab_payload = {
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "design_basis_commit": C.DESIGN_BASIS_COMMIT,
        "constants": {
            "bank_cap": C.BANK_CAP,
            "bank_min_molecules": C.BANK_MIN_MOLECULES,
            "s2_min_molecules": C.S2_MIN_MOLECULES,
            "s2_cap": C.S2_CAP,
            "relation_min_molecules": C.RELATION_MIN_MOLECULES,
            "relation_cap": C.RELATION_CAP,
            "extension_bins": list(C.EXTENSION_BINS),
            "distance_bins": list(C.DISTANCE_BINS),
            "named_graph_counts": list(C.NAMED_GRAPH_COUNTS),
            "lambda_fraction": C.LAMBDA_FRACTION,
            "scale_floor": C.SCALE_FLOOR,
            "raw_diff_eps": C.RAW_DIFF_EPS,
            "neg_clamp_eps": C.NEG_CLAMP_EPS,
        },
        "b_bank": {
            "stats": b_bank["stats"],
            "vocabulary": {
                "atom_values": {str(k): v for k, v in b_bank["vocabulary"]["atom_values"].items()},
                "bond_values": {str(k): v for k, v in b_bank["vocabulary"]["bond_values"].items()},
                "degree_values": b_bank["vocabulary"]["degree_values"],
                "ring_values": b_bank["vocabulary"]["ring_values"],
                "element_values": b_bank["vocabulary"]["element_values"],
            },
            "records": [
                {
                    "label": record.label,
                    "ast_json": ast_to_json(record.ast),
                    "ast_canon": A.canon(record.ast),
                    "train_support": int(record.train_support),
                    "source": record.source,
                    "outer_round": int(record.outer_round),
                }
                for record in b_records
            ],
        },
        "common_groups": common_groups,
        "relation_slots": relation_groups,
        "column_order": column_order,
        "ogb_decoding": decoding,
        "candidate_selection": {
            "s2_selected": [
                {
                    "name": row["name"],
                    "train_support": row["train_support"],
                    "train_count_min": row["train_count_min"],
                    "train_count_max": row["train_count_max"],
                    "meta": row["meta"],
                }
                for row in selected["s2_selected"]
            ],
            "s2_excluded_by_cap": selected["s2_excluded_by_cap"],
            "n_s2_survivors": selected["n_s2_survivors"],
            "relation_selected": [
                {
                    "key": row["key"],
                    "tp_name": row["tp_name"],
                    "bin": row["bin"],
                    "train_support": row["train_support"],
                    "train_count_min": row["train_count_min"],
                    "train_count_max": row["train_count_max"],
                    "meta": row["meta"],
                }
                for row in selected["relation_selected"]
            ],
            "relation_excluded_by_cap": selected["relation_excluded_by_cap"],
            "n_relation_survivors": selected["n_relation_survivors"],
        },
        "target_scaler": scaler,
        "coverage": coverage,
        "split_check_stats": {
            split: split_payload[split]["check_stats"] for split in C.SPLITS
        },
        "split_row_id_sha256": {
            split: C.sha256_array(split_payload[split]["ids"]) for split in C.SPLITS
        },
        "split_label_sha256": {
            split: C.sha256_array(np.asarray(payload["labels"], dtype=np.float64)[split_payload[split]["ids"]])
            for split in C.SPLITS
        },
    }
    C.write_json(out / "rules_vocabulary.json", vocab_payload)
    C.write_json(out / "column_order.json", column_order)
    hashes = {
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "graph_cache_sha256": C.EXPECTED_CACHE_SHA256["graph"],
        "lipo_raw_sha256": C.EXPECTED_CACHE_SHA256["lipo_raw"],
        "rules_vocabulary_sha256": C.sha256_file(out / "rules_vocabulary.json"),
        "column_order_sha256": C.sha256_file(out / "column_order.json"),
        "scales_sha256": C.sha256_file(out / "scales.npz"),
        "matrices_sha256": {
            split: C.sha256_file(matrices_dir / f"{split}.npz") for split in C.SPLITS
        },
        "code_sha256": C.code_hashes(),
        "protocol_sha256": C.protocol_hash(),
        "git_commit": C.git_commit(),
    }
    C.write_json(out / "hashes.json", hashes)
    summary = {
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "n_b_records": len(b_records),
        "b_bank_stats": b_bank["stats"],
        "n_s2_selected": len(selected["s2_selected"]),
        "n_relation_slots": len(relation_groups),
        "n_final_types": len({row["tp_name"] for row in selected["relation_selected"]}),
        "n_common_groups": len(common_groups),
        "n_common_columns": int(2 * len(common_groups)),
        "n_extension_columns": int(2 * len(relation_groups)),
        "coverage": coverage,
        "target_scaler": scaler,
        "scales": {
            "common_min": float(common_scale.min()) if common_scale.size else None,
            "common_max": float(common_scale.max()) if common_scale.size else None,
            "real_min": float(real_scale.min()) if real_scale.size else None,
            "real_max": float(real_scale.max()) if real_scale.size else None,
            "null_min": float(null_scale.min()) if null_scale.size else None,
            "null_max": float(null_scale.max()) if null_scale.size else None,
        },
        "split_check_stats": {split: split_payload[split]["check_stats"] for split in C.SPLITS},
        "matrices": {
            split: {
                "n_rows": int(split_payload[split]["ids"].size),
                "base_shape": list(split_payload[split]["base_raw"].shape),
                "real_shape": list(split_payload[split]["real_raw"].shape),
                "null_shape": list(split_payload[split]["null_raw"].shape),
            }
            for split in C.SPLITS
        },
    }
    C.write_json(out / "summary.json", summary)
    return summary
