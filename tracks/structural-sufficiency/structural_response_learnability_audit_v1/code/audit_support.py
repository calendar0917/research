"""Task 1: independent support of real structural replacement types."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

import audit_common as A


SCOPE_KEYS = ("train_sub", "cross", "dev_dev")


def _counts_for_positions(pairs: A.PairTable, pos: np.ndarray,
                          inp: A.Inputs) -> dict[str, int]:
    if pos.size == 0:
        return {
            "pairs": 0, "molecules": 0, "ctx_full": 0, "ctx_r1": 0, "ctx_r2": 0,
            "scaffolds": 0, "components": 0,
        }
    mols = A.union_molecules(pairs.a, pairs.b, pos)
    return {
        "pairs": int(pos.size),
        "molecules": int(len(mols)),
        "ctx_full": int(len(set(int(x) for x in pairs.C[pos].tolist()))),
        "ctx_r1": int(len(set(int(x) for x in pairs.r1e[pos].tolist()))),
        "ctx_r2": int(len(set(int(x) for x in pairs.r2e[pos].tolist()))),
        "scaffolds": int(len({inp.scaffold_of_mol[int(m)] for m in mols})),
        "components": int(len(A.union_components(pairs.comp_train_nodes, pairs.a, pairs.b, pos))),
    }


def type_table(pairs: A.PairTable, inp: A.Inputs, keys: np.ndarray,
               label: str) -> list[dict[str, Any]]:
    """Per-type support counts, using Train-sub for training support."""
    groups_all = A.group_positions(keys, np.arange(keys.shape[0]))
    groups_train = A.group_positions(keys, np.nonzero(pairs.scope == 0)[0])
    rows: list[dict[str, Any]] = []
    for key in sorted(groups_all, key=lambda x: (str(type(x)), str(x))):
        pos_all = groups_all[key]
        pos_train = groups_train.get(key, np.zeros(0, np.int64))
        rec: dict[str, Any] = {
            "key": list(key) if isinstance(key, tuple) else int(key),
            "pairs_train": int(pos_train.size),
            "pairs_all": int(pos_all.size),
            "scope_counts": {SCOPE_KEYS[s]: int((pairs.scope[pos_all] == s).sum()) for s in range(3)},
        }
        if pos_train.size:
            c = _counts_for_positions(pairs, pos_train, inp)
            rec.update({
                "molecules_train": c["molecules"],
                "ctx_full_train": c["ctx_full"],
                "ctx_r1_train": c["ctx_r1"],
                "ctx_r2_train": c["ctx_r2"],
                "scaffolds_train": c["scaffolds"],
                "components_train": c["components"],
            })
        else:
            rec.update({
                "molecules_train": 0, "ctx_full_train": 0, "ctx_r1_train": 0,
                "ctx_r2_train": 0, "scaffolds_train": 0, "components_train": 0,
            })
        if pos_all.size:
            mols = A.union_molecules(pairs.a, pairs.b, pos_all)
            rec.update({
                "molecules_all": int(len(mols)),
                "ctx_full_all": int(len(set(int(x) for x in pairs.C[pos_all].tolist()))),
                "ctx_r1_all": int(len(set(int(x) for x in pairs.r1e[pos_all].tolist()))),
                "ctx_r2_all": int(len(set(int(x) for x in pairs.r2e[pos_all].tolist()))),
                "scaffolds_all": int(len({inp.scaffold_of_mol[int(m)] for m in mols})),
                "components_all": int(len(A.union_components(pairs.comp_all_nodes, pairs.a, pairs.b, pos_all))),
            })
        else:
            rec.update({
                "molecules_all": 0, "ctx_full_all": 0, "ctx_r1_all": 0,
                "ctx_r2_all": 0, "scaffolds_all": 0, "components_all": 0,
            })
        rows.append(rec)
    return rows


def _threshold_report(rows: Sequence[Mapping[str, Any]], field: str,
                      thresholds: Sequence[int] = (2, 3, 5)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in thresholds:
        selected = [r for r in rows if int(r.get(field, 0)) >= int(k)]
        out[str(k)] = {
            "n_types": len(selected),
            "n_pairs_train": int(sum(int(r["pairs_train"]) for r in selected)),
            "n_pairs_all": int(sum(int(r["pairs_all"]) for r in selected)),
            "n_molecule_incidences": int(sum(int(r["molecules_train"]) for r in selected)),
            "n_scaffold_incidences": int(sum(int(r["scaffolds_train"]) for r in selected)),
            "n_component_incidences": int(sum(int(r["components_train"]) for r in selected)),
        }
    return out


def _unique_units(pairs: A.PairTable, inp: A.Inputs, rows: Sequence[Mapping[str, Any]],
                  keys: np.ndarray, field: str,
                  thresholds: Sequence[int] = (2, 3, 5)) -> dict[str, Any]:
    train = np.nonzero(pairs.scope == 0)[0]
    out: dict[str, Any] = {}
    for k in thresholds:
        selected = [r for r in rows if int(r.get(field, 0)) >= int(k)]
        keyset = {A.hashable_key(r["key"]) for r in selected}
        pos = np.asarray([j for j in train.tolist()
                          if A.hashable_key(keys[j]) in keyset], np.int64)
        if pos.size == 0:
            out[str(k)] = {"n_types": 0, "n_pairs": 0}
            continue
        mols = A.union_molecules(pairs.a, pairs.b, pos)
        cells = set(zip((A.hashable_key(keys[j]) for j in pos.tolist()),
                        (int(pairs.C[j]) for j in pos.tolist())))
        out[str(k)] = {
            "n_types": len(selected),
            "n_pairs": int(pos.size),
            "n_type_context_cells": int(len(cells)),
            "n_molecules": int(len(mols)),
            "n_scaffolds": int(len({inp.scaffold_of_mol[int(m)] for m in mols})),
            "n_components": int(len({int(pairs.comp_train_nodes[int(m)]) for m in mols})),
            "pairs_per_type_context_cell": float(pos.size / max(1, len(cells))),
        }
    return out


def _joint_threshold_report(rows: Sequence[Mapping[str, Any]], ctx_field: str,
                            other_field: str, thresholds: Sequence[int] = (2, 3, 5)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in thresholds:
        selected = [r for r in rows
                    if int(r.get(ctx_field, 0)) >= int(k) and int(r.get(other_field, 0)) >= 2]
        out[str(k)] = {
            "n_types": len(selected),
            "n_pairs_train": int(sum(int(r["pairs_train"]) for r in selected)),
        }
    return out


def _strict_independence(rows: Sequence[Mapping[str, Any]],
                         thresholds: Sequence[int] = (2, 3, 5)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in thresholds:
        selected = [r for r in rows
                    if int(r.get("ctx_full_train", 0)) >= int(k)
                    and int(r.get("components_train", 0)) >= 2
                    and int(r.get("scaffolds_train", 0)) >= 2]
        out[str(k)] = {
            "n_types": len(selected),
            "n_pairs_train": int(sum(int(r["pairs_train"]) for r in selected)),
        }
    return out


def _component_summary(comp: np.ndarray, nodes: np.ndarray) -> dict[str, Any]:
    counts = Counter()
    for x in np.asarray(nodes, np.int64).tolist():
        counts[int(comp[int(x)])] += 1
    sizes = np.asarray(list(counts.values()), np.int64) if counts else np.zeros(0, np.int64)
    return {
        "n_components": int(len(counts)),
        "n_molecules_in_components": int(sizes.sum()) if sizes.size else 0,
        "size_histogram": dict(sorted(Counter(sizes.tolist()).items())) if sizes.size else {},
        "max_size": int(sizes.max()) if sizes.size else 0,
        "median_size": float(np.median(sizes)) if sizes.size else 0.0,
    }


def _duplicate_checks(pairs: A.PairTable) -> dict[str, Any]:
    n = pairs.a.size
    unordered = np.stack([np.minimum(pairs.a, pairs.b), np.maximum(pairs.a, pairs.b)], 1)
    uniq, inv, cnt = np.unique(unordered, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    orientation = (pairs.a > pairs.b).astype(np.int8)
    flips = 0
    for g in range(uniq.shape[0]):
        if len(set(orientation[inv == g].tolist())) > 1:
            flips += 1
    incidence = Counter(np.concatenate([pairs.a, pairs.b]).tolist())
    inc = np.asarray(list(incidence.values()), np.int64)
    cell = Counter(zip(pairs.T.tolist(), pairs.C.tolist()))
    full_key = list(zip(pairs.a.tolist(), pairs.b.tolist(), pairs.Sa.tolist(),
                        pairs.Sb.tolist(), pairs.C.tolist(), pairs.bond_row.tolist()))
    full_set = set(full_key)
    full_counts = Counter(full_key)
    reverse_rows = sum(
        1 for k in full_key
        if (k[1], k[0], k[3], k[2], k[4], k[5]) in full_set)
    return {
        "n_pairs": int(n),
        "n_unique_unordered_molecule_pairs": int(uniq.shape[0]),
        "molecule_pair_multiplicity_histogram": dict(sorted(Counter(cnt.tolist()).items())),
        "n_molecule_pairs_with_both_molecule_id_orientations": int(flips),
        "n_exact_reverse_pair_rows": int(reverse_rows // 2),
        "n_molecules_in_pairs": int(len(incidence)),
        "pairs_per_molecule_incidence": {
            "mean": float(inc.mean()) if inc.size else 0.0,
            "median": float(np.median(inc)) if inc.size else 0.0,
            "max": int(inc.max()) if inc.size else 0,
            "histogram": dict(sorted(Counter(inc.tolist()).items())),
        },
        "max_pairs_per_exact_type_full_context": int(max(cell.values())) if cell else 0,
        "n_exact_type_full_context_cells": int(len(cell)),
        "n_unique_full_pair_rows": int(len(full_counts)),
        "n_repeated_full_pair_rows": int(n - len(full_counts)),
        "n_parallel_edges_same_molecule_pair": int(n - uniq.shape[0]),
        "full_pair_multiplicity_histogram": dict(sorted(Counter(full_counts.values()).items())),
    }


def _cross_split_components(pairs: A.PairTable, inp: A.Inputs) -> dict[str, Any]:
    cross = pairs.scope == 1
    dd = pairs.scope == 2
    train = pairs.scope == 0
    comp_train = pairs.comp_train_nodes
    comp_all = pairs.comp_all_nodes

    all_nodes = np.unique(np.concatenate([pairs.a, pairs.b]))
    nodes_train = np.asarray([m for m in all_nodes.tolist() if inp.train_sub[int(m)]], np.int64)
    nodes_dev = np.asarray([m for m in all_nodes.tolist() if inp.dev[int(m)]], np.int64)
    dev_nodes_cross = set(int(m) for m in nodes_dev.tolist()
                          if int(comp_all[int(m)]) in set(int(comp_all[int(pairs.a[j])]) for j in np.nonzero(cross)[0].tolist()))
    comps_cross = set(int(comp_all[int(pairs.a[j])]) for j in np.nonzero(cross)[0].tolist())

    train_edge_nodes = np.unique(np.concatenate([pairs.a[train], pairs.b[train]]))
    train_nodes_all = set(int(m) for m in nodes_train.tolist())
    train_edge_set = set(int(m) for m in train_edge_nodes.tolist())
    train_comp_set = set(int(x) for x in comp_all[nodes_train].tolist()) if nodes_train.size else set()
    dev_comp_set = set(int(x) for x in comp_all[nodes_dev].tolist()) if nodes_dev.size else set()
    comps_with_both = train_comp_set & dev_comp_set

    rows: list[dict[str, Any]] = []
    for c in sorted(comps_with_both):
        pos_cross = np.nonzero(cross)[0]
        pos_dd = np.nonzero(dd)[0]
        pos_train = np.nonzero(train)[0]
        pos_cross = [int(j) for j in pos_cross.tolist() if int(pairs.comp_all[j]) == c]
        pos_dd = [int(j) for j in pos_dd.tolist() if int(pairs.comp_all[j]) == c]
        pos_train = [int(j) for j in pos_train.tolist() if int(pairs.comp_all[j]) == c]
        dev_mols = set()
        for j in pos_cross + pos_dd:
            for m in (int(pairs.a[j]), int(pairs.b[j])):
                if inp.dev[m]:
                    dev_mols.add(m)
        rows.append({
            "component": int(c),
            "n_cross_pairs": len(pos_cross),
            "n_dev_dev_pairs": len(pos_dd),
            "n_train_pairs": len(pos_train),
            "n_dev_molecules": len(dev_mols),
        })
    rows.sort(key=lambda r: (-r["n_cross_pairs"], -r["n_dev_molecules"], r["component"]))

    dd_in_cross = sum(1 for j in np.nonzero(dd)[0].tolist() if int(pairs.comp_all[j]) in comps_with_both)
    train_in_cross = sum(1 for j in np.nonzero(train)[0].tolist() if int(pairs.comp_all[j]) in comps_with_both)

    return {
        "n_components_train_edges": int(len(set(comp_train[train_edge_nodes].tolist()))),
        "n_train_sub_molecules_in_any_pair": int(len(train_nodes_all)),
        "n_train_sub_molecules_in_train_edges": int(len(train_edge_set)),
        "n_train_sub_molecules_only_in_cross_edges": int(len(train_nodes_all - train_edge_set)),
        "n_components_all_graph_on_train_nodes": int(len(set(comp_all[nodes_train].tolist()))),
        "n_components_all_train_pairs": int(len(set(comp_all[all_nodes].tolist()))),
        "n_components_with_train_and_dev": int(len(comps_with_both)),
        "n_cross_pairs": int(cross.sum()),
        "n_dev_dev_pairs": int(dd.sum()),
        "n_dev_dev_pairs_in_cross_components": int(dd_in_cross),
        "n_train_pairs_in_cross_components": int(train_in_cross),
        "frac_train_pairs_in_cross_components": float(train_in_cross / max(1, train.sum())),
        "n_dev_molecules": int(inp.dev.sum()),
        "n_dev_molecules_in_cross_components": int(len(dev_nodes_cross)),
        "n_dev_molecules_in_dev_only_or_other": int(inp.dev.sum() - len(dev_nodes_cross)),
        "n_cross_components": int(len(comps_cross)),
        "top_cross_components": rows[:10],
    }


def _coverage(pairs: A.PairTable, inp: A.Inputs, rows_exact: Sequence[Mapping[str, Any]],
              scope: int) -> dict[str, Any]:
    pos = np.nonzero(pairs.scope == scope)[0]
    tr = np.nonzero(pairs.scope == 0)[0]
    seen_types = set(int(x) for x in pairs.T[tr].tolist())
    seen_ctx = set(int(x) for x in pairs.C[tr].tolist())
    seen_r1 = set(int(x) for x in pairs.r1e[tr].tolist())
    seen_r2 = set(int(x) for x in pairs.r2e[tr].tolist())
    ctx2_types = {int(r["key"]) for r in rows_exact
                  if int(r["pairs_train"]) > 0 and int(r["ctx_full_train"]) >= 2}
    t_seen = np.asarray([int(pairs.T[j]) in seen_types for j in pos.tolist()], bool) if pos.size else np.zeros(0, bool)
    c_seen = np.asarray([int(pairs.C[j]) in seen_ctx for j in pos.tolist()], bool) if pos.size else np.zeros(0, bool)
    r1_seen = np.asarray([int(pairs.r1e[j]) in seen_r1 for j in pos.tolist()], bool) if pos.size else np.zeros(0, bool)
    r2_seen = np.asarray([int(pairs.r2e[j]) in seen_r2 for j in pos.tolist()], bool) if pos.size else np.zeros(0, bool)
    t_ctx2 = np.asarray([int(pairs.T[j]) in ctx2_types for j in pos.tolist()], bool) if pos.size else np.zeros(0, bool)
    if scope == 1:
        clusters = np.asarray([int(pairs.a[j]) if inp.dev[int(pairs.a[j])] else int(pairs.b[j])
                               for j in pos.tolist()], np.int64)
        cluster_unit = "Dev molecule"
    else:
        clusters = np.asarray([int(pairs.comp_all[j]) for j in pos.tolist()], np.int64)
        cluster_unit = "full Train pair component"
    return {
        "n_pairs": int(pos.size),
        "n_clusters": int(np.unique(clusters).size),
        "cluster_unit": cluster_unit,
        "type_seen": int(t_seen.sum()),
        "type_seen_frac": float(t_seen.mean()) if pos.size else 0.0,
        "type_seen_ctx2": int(t_ctx2.sum()),
        "type_seen_ctx2_frac": float(t_ctx2.mean()) if pos.size else 0.0,
        "ctx_full_seen": int(c_seen.sum()),
        "ctx_full_seen_frac": float(c_seen.mean()) if pos.size else 0.0,
        "ctx_r1_seen": int(r1_seen.sum()),
        "ctx_r1_seen_frac": float(r1_seen.mean()) if pos.size else 0.0,
        "ctx_r2_seen": int(r2_seen.sum()),
        "ctx_r2_seen_frac": float(r2_seen.mean()) if pos.size else 0.0,
        "type_and_ctx_full_seen": int((t_seen & c_seen).sum()),
        "type_seen_ctx_full_unseen": int((t_seen & ~c_seen).sum()),
        "type_unseen_ctx_full_seen": int((~t_seen & c_seen).sum()),
        "both_unseen": int((~t_seen & ~c_seen).sum()),
    }


def run(inp: A.Inputs, pairs: A.PairTable) -> dict[str, Any]:
    exact_rows = type_table(pairs, inp, pairs.T, "exact")
    coarse_rows = type_table(pairs, inp, pairs.coarse, "coarse")

    thresholds = {
        "exact": {
            "context_full": _threshold_report(exact_rows, "ctx_full_train"),
            "context_r2": _threshold_report(exact_rows, "ctx_r2_train"),
            "context_r1": _threshold_report(exact_rows, "ctx_r1_train"),
            "scaffolds": _threshold_report(exact_rows, "scaffolds_train"),
            "pair_components": _threshold_report(exact_rows, "components_train"),
            "context_full_and_components2": _joint_threshold_report(
                exact_rows, "ctx_full_train", "components_train"),
            "context_full_and_scaffolds2": _joint_threshold_report(
                exact_rows, "ctx_full_train", "scaffolds_train"),
        },
        "coarse": {
            "context_full": _threshold_report(coarse_rows, "ctx_full_train"),
            "context_r2": _threshold_report(coarse_rows, "ctx_r2_train"),
            "context_r1": _threshold_report(coarse_rows, "ctx_r1_train"),
            "scaffolds": _threshold_report(coarse_rows, "scaffolds_train"),
            "pair_components": _threshold_report(coarse_rows, "components_train"),
            "context_full_and_components2": _joint_threshold_report(
                coarse_rows, "ctx_full_train", "components_train"),
            "context_full_and_scaffolds2": _joint_threshold_report(
                coarse_rows, "ctx_full_train", "scaffolds_train"),
        },
    }

    train_nodes = np.unique(np.concatenate([pairs.a[pairs.scope == 0], pairs.b[pairs.scope == 0]]))
    all_nodes = np.unique(np.concatenate([pairs.a, pairs.b]))
    duplicate = _duplicate_checks(pairs)
    cross = _cross_split_components(pairs, inp)

    payload = {
        "meta": A.meta_block(inp, "structural-response-learnability-audit-v1/support", {
            "source_pairs": A.relative(A.TRC_CACHE),
            "source_identities": A.relative(A.ACRA_CACHE),
        }),
        "definitions": {
            "exact_type": "(word_a, word_b, cut bond raw 3-column attributes); the pair directory's pair_transform",
            "coarse_type": "(radius-1 identity of word_a, radius-1 identity of word_b, cut bond raw attributes); ACRA local identity of the attachment site",
            "context_full": "exact environment fragment C (the matched context; canonical coloured rooted fragment, up to 8 atoms)",
            "context_r2": "ACRA C identity: radius-2 ball around the environment attachment atom in G - cut bond",
            "context_r1": "ACRA S identity of the reverse directed event: radius-1 ball around the environment attachment atom in G - cut bond",
            "independent_unit": "one distinct full context; pair components and scaffolds are reported as molecule-level dependence checks",
            "label_scope": "support is label-free and restricted to official Train rows",
        },
        "totals": {
            "n_pairs_analyzed": int(pairs.a.size),
            "n_pairs_train_sub": int((pairs.scope == 0).sum()),
            "n_pairs_cross": int((pairs.scope == 1).sum()),
            "n_pairs_dev_dev": int((pairs.scope == 2).sum()),
            "n_molecules_in_pairs": int(len(all_nodes)),
            "n_molecules_train_sub_in_train_sub_pairs": int(len(train_nodes)),
            "n_molecules_dev": int(inp.dev.sum()),
            "n_exact_types_all": len(exact_rows),
            "n_exact_types_with_train_support": int(sum(1 for r in exact_rows if r["pairs_train"] > 0)),
            "n_exact_types_train_only_singletons": int(sum(1 for r in exact_rows if r["pairs_train"] == 1)),
            "n_coarse_types_all": len(coarse_rows),
            "n_coarse_types_with_train_support": int(sum(1 for r in coarse_rows if r["pairs_train"] > 0)),
            "n_coarse_types_train_only_singletons": int(sum(1 for r in coarse_rows if r["pairs_train"] == 1)),
            "n_full_contexts_train": int(len(set(int(x) for x in pairs.C[pairs.scope == 0].tolist()))),
            "n_r1_contexts_train": int(len(set(int(x) for x in pairs.r1e[pairs.scope == 0].tolist()))),
            "n_r2_contexts_train": int(len(set(int(x) for x in pairs.r2e[pairs.scope == 0].tolist()))),
            "n_scaffolds_train_sub": int(inp.split.get("n_scaffolds_train_sub", -1)),
            "n_scaffolds_dev": int(inp.split.get("n_scaffolds_dev", -1)),
            "n_pair_components_train_graph": int(len(set(pairs.comp_train_nodes[train_nodes].tolist()))),
            "n_pair_components_all_train_pairs": int(len(set(pairs.comp_all_nodes[all_nodes].tolist()))),
        },
        "thresholds_train_support": {
            **thresholds,
            "note": ("n_molecule/scaffold/component_incidences are per-type counts summed across "
                     "types and are not unique across types; use unique_units_train_support for "
                     "the deduplicated counts."),
        },
        "unique_units_train_support": {
            "exact": _unique_units(pairs, inp, exact_rows, pairs.T, "ctx_full_train"),
            "coarse": _unique_units(pairs, inp, coarse_rows, pairs.coarse, "ctx_full_train"),
        },
        "strict_independence_train_support": {
            "exact": _strict_independence(exact_rows),
            "coarse": _strict_independence(coarse_rows),
            "definition": "contexts >= k, pair components >= 2, and scaffolds >= 2 within the type",
        },
        "independence": {
            "duplicates": duplicate,
            "train_pair_graph": _component_summary(pairs.comp_train_nodes, train_nodes),
            "all_train_pair_graph": _component_summary(pairs.comp_all_nodes, all_nodes),
            "cross_split": cross,
        },
        "evaluation_coverage": {
            "cross": _coverage(pairs, inp, exact_rows, 1),
            "dev_dev": _coverage(pairs, inp, exact_rows, 2),
        },
        "types": {
            "exact": exact_rows,
            "coarse": coarse_rows,
        },
        "notes": [
            "The full context C is the matched environment and the finest independence unit; radius-1 and radius-2 identities are nested coarsenings.",
            "For exact types, max pairs per (type, full context) is 1, so full contexts and pairs coincide; for coarse types a coarse type can contain several exact types under different full contexts.",
            "Pair components are connected components of the matched-pair graph containing at least one pair endpoint; isolated molecules are not counted.",
            "No Valid/Test row or label is used; the scope is the official Train set only.",
        ],
    }
    return payload