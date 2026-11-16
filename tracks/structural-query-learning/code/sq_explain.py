"""Exact export and explanation of the query model.

Two artefacts leave this module:

``queries.json``
    every query of the final bank: AST (structured + canonical string), the four
    graph-level transforms, cost/depth/support, the discovery round and the
    Train gain.  No anonymous embedding dimension is given a chemical name.

``native_trees.json`` + ``replay.json``
    the actual boosting trees of the Val-selected state and the audit that the
    native executor reproduces the XGBoost predictions (raw units) to
    ``<= 1e-5`` absolute error on all Val/Test molecules and up to 256 Train
    molecules.

The three deterministic case studies walk the real decision paths, list the
matched roots (with structural witnesses for composite queries) and summarise
the tree contributions with an explicit remainder.  Nothing here uses SHAP, and
split gain is never presented as a chemical effect size.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import sq_ast as A
import sq_bank as B
import sq_common as C
import sq_xgb as X
from sq_ast import PackedGraph

REPLAY_MAX_TRAIN = 256
REPLAY_TOL = 1e-5


# ---------------------------------------------------------------------------
# queries.json
# ---------------------------------------------------------------------------


def export_queries(
    records: Sequence[B.QueryRecord],
    *,
    dataset: str,
    base_count: int,
    discovered_count: int,
    feature_names: Sequence[str] | None = None,
    scaler: Mapping[str, float] | None = None,
    selected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    queries = []
    for record in records:
        queries.append(
            {
                **record.to_dict(),
                "ast_tree": ast_to_json(record.ast),
                "transforms": list(A.FEATURE_TRANSFORMS),
                "feature_columns": [f"{record.label}::{t}" for t in A.FEATURE_TRANSFORMS],
                "operator_shape": A.operator_shape(record.ast),
            }
        )
    return {
        "round_id": C.ROUND_ID,
        "dataset": dataset,
        "n_queries": len(records),
        "n_base": int(base_count),
        "n_discovered": int(discovered_count),
        "feature_transforms": {
            "matched_roots": "c = number of atoms matching the query",
            "density": "p = c / n_atoms",
            "exists": "e = 1[c > 0]",
            "log1p_count": "l = log1p(c)",
        },
        "node_budget": {"max_nodes": A.MAX_NODES, "max_nbr_branch_hops": A.MAX_HOPS},
        "base_columns": list(B.BASE_COLUMN_NAMES),
        "feature_names": list(feature_names or []),
        "target_scaler": dict(scaler or {}),
        "selected_state": dict(selected or {}),
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "replay_recipe": (
            "x_i = features from ast_tree (values are c, p=c/n_atoms, e=1[c>0], log1p(c)); "
            "columns follow feature_names; margin = base_score + sum over trees of "
            "leaf value reached with 'value < split_condition' (float32, left child if true); "
            "prediction_raw = margin * target_scaler.std + target_scaler.mean"
        ),
        "queries": queries,
    }


def ast_to_json(ast: tuple) -> dict[str, Any]:
    kind = ast[0]
    if kind in A.NODE_KINDS:
        if kind == "ATOM_EQ":
            return {"op": "ATOM_EQ", "atom_column": ast[1], "value": ast[2]}
        if kind in ("DEG_EQ", "DEG_GE", "MIN_RING_EQ", "MIN_RING_GE"):
            return {"op": kind, "value": ast[1]}
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
    return {"op": "BOND_EQ", "bond_column": bond[1], "value": bond[2]}


# ---------------------------------------------------------------------------
# native export + replay audit
# ---------------------------------------------------------------------------


def export_and_audit(
    *,
    booster,
    feature_names: Sequence[str],
    matrices: Mapping[str, np.ndarray],
    scaler: Mapping[str, float],
    row_ids: Mapping[str, np.ndarray],
    rounds: int,
    max_train: int = REPLAY_MAX_TRAIN,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trees = X.export_trees(booster, feature_names, rounds=rounds)
    audit: dict[str, Any] = {
        "n_trees": int(trees["n_trees"]),
        "tolerance_abs": REPLAY_TOL,
        "splits": {},
        "max_train_molecules": int(max_train),
    }
    worst_abs = 0.0
    worst_rel = 0.0
    for split in ("train", "valid", "test"):
        design = matrices[split]
        n_rows = design.shape[0]
        rows = np.arange(n_rows)
        if split == "train" and n_rows > max_train:
            rows = rows[:max_train]
        sub = design[rows]
        native = X.replay_native(trees, sub, feature_names, scaler)
        mean = float(scaler["mean"])
        std = float(scaler["std"])
        xgb_std = booster.predict(
            dmatrix(sub, feature_names), iteration_range=(0, int(rounds))
        )
        xgb_raw = np.asarray(xgb_std, dtype=np.float64) * std + mean
        abs_err = np.abs(native - xgb_raw)
        denom = np.maximum(np.abs(xgb_raw), 1e-12)
        rel_err = abs_err / denom
        audit["splits"][split] = {
            "n_rows_checked": int(rows.size),
            "max_abs_error": float(abs_err.max()),
            "max_rel_error": float(rel_err.max()),
            "mean_abs_error": float(abs_err.mean()),
        }
        worst_abs = max(worst_abs, float(abs_err.max()))
        worst_rel = max(worst_rel, float(rel_err.max()))
        if split == "test":
            audit["row_id_checksum"] = C.sha256_array(np.asarray(row_ids[split], dtype=np.int64))
    audit["max_abs_error_overall"] = worst_abs
    audit["max_rel_error_overall"] = worst_rel
    audit["pass"] = bool(worst_abs <= REPLAY_TOL)
    return trees, audit


def dmatrix(design: np.ndarray, feature_names: Sequence[str]):
    import xgboost as xgb

    return xgb.DMatrix(design, feature_names=list(feature_names))


# ---------------------------------------------------------------------------
# case studies
# ---------------------------------------------------------------------------


def design_row(design: np.ndarray, names: Sequence[str], row: int) -> dict[str, float]:
    return {name: float(design[row, j]) for j, name in enumerate(names)}


def select_cases(
    *,
    pred_a: np.ndarray,
    pred_f: np.ndarray,
    target: np.ndarray,
    row_ids: np.ndarray,
) -> list[dict[str, Any]]:
    """Three deterministic Val cases: median error, max gain, max loss vs F."""
    err_a = np.abs(pred_a - target)
    err_f = np.abs(pred_f - target)
    order = np.lexsort((row_ids, err_a))
    median_position = order[len(order) // 2]
    gain = err_f - err_a
    loss = err_a - err_f
    cases: list[dict[str, Any]] = []
    used: set[int] = set()

    def pick(metric: np.ndarray, prefer_high: bool) -> int:
        ranked = np.lexsort((row_ids, -metric if prefer_high else metric))
        for index in ranked:
            if int(index) not in used:
                used.add(int(index))
                return int(index)
        raise RuntimeError("no distinct case left")

    cases.append({"kind": "median_abs_error", "row": int(median_position)})
    cases.append({"kind": "max_improvement_vs_F", "row": pick(gain, True)})
    cases.append({"kind": "max_degradation_vs_F", "row": pick(loss, True)})
    for case in cases:
        row = case["row"]
        case.update(
            {
                "row_id": int(row_ids[row]),
                "target": float(target[row]),
                "pred_A": float(pred_a[row]),
                "pred_F": float(pred_f[row]),
                "abs_error_A": float(err_a[row]),
                "abs_error_F": float(err_f[row]),
                "error_delta_A_minus_F": float(err_a[row] - err_f[row]),
            }
        )
    return cases


def explain_molecule(
    *,
    packed: PackedGraph,
    records: Sequence[B.QueryRecord],
    trees: Mapping[str, Any],
    feature_names: Sequence[str],
    design_row: Mapping[str, float],
    scaler: Mapping[str, float],
    case: Mapping[str, Any],
    cache: B.MaskCache | None,
    max_queries_listed: int = 8,
    top_trees: int = 8,
) -> dict[str, Any]:
    """Structure, matched roots, witnesses, real tree paths and contributions."""
    paths = X.tree_paths(trees, design_row, feature_names)
    graph = int(np.searchsorted(np.cumsum(packed.graph_n_atoms), case["row"], side="right"))
    atom_start = int(np.concatenate([[0], np.cumsum(packed.graph_n_atoms)])[graph])
    atom_end = atom_start + int(packed.graph_n_atoms[graph])
    nodes = np.arange(atom_start, atom_end)
    bonds = [
        {
            "u": int(packed.bond_u[idx] - atom_start),
            "v": int(packed.bond_v[idx] - atom_start),
            "features": [int(v) for v in packed.bond_feat[idx]],
        }
        for idx in range(packed.n_bonds)
        if atom_start <= packed.bond_u[idx] < atom_end
    ]
    structure = {
        "smiles": packed.smiles[graph],
        "n_atoms": int(packed.graph_n_atoms[graph]),
        "n_physical_bonds": int(packed.graph_n_bonds[graph]),
        "n_components": int(packed.graph_n_components[graph]),
        "cycle_rank": int(packed.graph_cycle_rank[graph]),
        "atoms": [
            {
                "local_index": int(node - atom_start),
                "atom_features": [int(v) for v in packed.atom_feat[node]],
                "degree": int(packed.degree[node]),
                "min_ring": int(packed.min_ring[node]),
                "in_cycle": bool(packed.in_cycle[node]),
            }
            for node in nodes
        ],
        "bonds": bonds,
    }

    # Matched roots for every query with c > 0 on this molecule.
    queries_used: list[dict[str, Any]] = []
    for record in records:
        value = float(design_row.get(f"{record.label}::matched_roots", 0.0))
        if value <= 0:
            continue
        mask = B.root_mask(packed, record.ast, cache)
        roots = [int(node - atom_start) for node in np.nonzero(mask[atom_start:atom_end])[0] + atom_start]
        entry = {
            "label": record.label,
            "ast": A.canon(record.ast),
            "source": record.source,
            "matched_roots_local": roots,
            "n_matched_roots_in_molecule": len(roots),
            "n_matched_roots_in_split": int(value),
            "features": {
                transform: float(design_row.get(f"{record.label}::{transform}", 0.0))
                for transform in A.FEATURE_TRANSFORMS
            },
            "tree_uses": sorted(
                {
                    name
                    for tree in paths["trees"]
                    for step in tree["steps"]
                    for name in [step["feature"]]
                    if name.startswith(f"{record.label}::")
                }
            ),
        }
        if not A.is_node_primitive(record.ast) and roots:
            entry["witness"] = A.witness(packed, record.ast, roots[0] + atom_start)
        queries_used.append(entry)
    queries_used.sort(key=lambda item: (-item["n_matched_roots_in_molecule"], item["label"]))

    contributions = sorted(
        (
            {
                "tree": entry["tree"],
                "leaf_value_std": entry["leaf_value_std"],
                "path_length": entry["path_length"],
                "steps": entry["steps"],
            }
            for entry in paths["trees"]
        ),
        key=lambda item: (-abs(item["leaf_value_std"]), item["tree"]),
    )
    head = contributions[:top_trees]
    remainder = float(sum(item["leaf_value_std"] for item in contributions[top_trees:]))
    features_on_paths = sorted({step["feature"] for item in contributions for step in item["steps"]})
    explanation = {
        "case": dict(case),
        "structure": structure,
        "queries_with_matches": queries_used[:max_queries_listed],
        "n_queries_with_matches": len(queries_used),
        "padding_note": "matched roots are 0-indexed within this molecule",
        "tree_paths": {
            "n_trees": len(contributions),
            "n_trees_using_a_query_feature": int(
                sum(
                    1
                    for item in contributions
                    if any(step["feature"].startswith(("A", "B", "R")) for step in item["steps"])
                )
            ),
            "n_features_on_any_path": len(features_on_paths),
            "total_path_length": int(paths["total_path_length"]),
            "mean_path_length": float(paths["total_path_length"] / max(1, len(contributions))),
            "top_contributions": head,
            "remainder_leaf_value_std": remainder,
            "total_leaf_value_std": float(paths["margin_std"]),
            "reconstructed_raw_prediction": float(paths["margin_std"]) * float(scaler["std"])
            + float(scaler["mean"]),
            "top_features": [
                {
                    "feature": name,
                    "value": float(design_row[name]),
                }
                for name in features_on_paths[:20]
            ],
        },
    }
    return explanation


def tree_complexity(trees: Mapping[str, Any]) -> dict[str, Any]:
    """Node/leaf counts of the exported model (a complexity fact, not a claim)."""
    n_nodes = 0
    n_leaves = 0
    depth_max = 0
    per_tree_depths: list[int] = []
    for tree in trees["trees"]:
        n_nodes += len(tree["left"])
        n_leaves += int(sum(1 for v in tree["left"] if v == -1))
        depth_max = max(depth_max, _depth(tree))
        per_tree_depths.append(_depth(tree))
    return {
        "n_trees": int(trees["n_trees"]),
        "n_nodes": int(n_nodes),
        "n_leaves": int(n_leaves),
        "max_depth": int(depth_max),
        "mean_depth": float(np.mean(per_tree_depths)) if per_tree_depths else 0.0,
    }


def _depth(tree: Mapping[str, Any]) -> int:
    best = 0
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if tree["left"][node] == -1:
            best = max(best, depth)
            continue
        stack.append((tree["left"][node], depth + 1))
        stack.append((tree["right"][node], depth + 1))
    return best
