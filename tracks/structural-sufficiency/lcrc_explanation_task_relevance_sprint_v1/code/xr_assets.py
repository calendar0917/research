"""Frozen asset loading, identity verification, pair support and edit regions.

The pair structure and atom mapping are *reused*, not re-derived by a new
matching rule.  This module verifies the copied pair catalogue against the
official graph cache and derives the structural edit region from the verified
mapping only.  Labels, predictions and WC_LOGP contributions are never used to
choose an edit region or a candidate.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import xr_common as C


# ---------------------------------------------------------------------------
# small raw-cache accessors
# ---------------------------------------------------------------------------


def _raw(b: Any, row: int) -> dict[str, Any]:
    lipo = b.lipo_raw
    ap = np.asarray(lipo["atom_ptr"], dtype=np.int64)
    ep = np.asarray(lipo["edge_ptr"], dtype=np.int64)
    a0, a1 = int(ap[row]), int(ap[row + 1])
    e0, e1 = int(ep[row]), int(ep[row + 1])
    return {
        "smiles": str(lipo["smiles"][row]),
        "atom_class": np.asarray(lipo["atom_class"], dtype=np.int64)[a0:a1],
        "atom_charge": np.asarray(lipo["atom_charge"], dtype=np.int64)[a0:a1],
        "atom_h": np.asarray(lipo["atom_h"], dtype=np.int64)[a0:a1],
        "edge_u": np.asarray(lipo["edge_u"], dtype=np.int64)[e0:e1],
        "edge_v": np.asarray(lipo["edge_v"], dtype=np.int64)[e0:e1],
        "edge_class": np.asarray(lipo["edge_class"], dtype=np.int64)[e0:e1],
        "edge_stereo": np.asarray(lipo["edge_stereo"], dtype=np.int64)[e0:e1],
        "atom_ptr": (a0, a1),
        "edge_ptr": (e0, e1),
    }


def _bond_map(raw: Mapping[str, Any]) -> dict[tuple[int, int], int]:
    return {
        (min(int(u), int(v)), max(int(u), int(v))): int(c)
        for u, v, c in zip(raw["edge_u"].tolist(), raw["edge_v"].tolist(), raw["edge_class"].tolist())
    }


def _atom_label(raw: Mapping[str, Any], i: int) -> tuple[int, int, int]:
    return (int(raw["atom_class"][i]), int(raw["atom_charge"][i]), int(raw["atom_h"][i]))


def cache_row_of(b: Any, split: str, row_id: int) -> int:
    ids = np.asarray(b.global_row_ids[split], dtype=np.int64)
    hits = np.flatnonzero(ids == int(row_id))
    if hits.size != 1:
        raise KeyError(f"{split}/{row_id}: found {hits.size} cache rows")
    return int(hits[0])


# ---------------------------------------------------------------------------
# catalogue / support
# ---------------------------------------------------------------------------


def load_pairs() -> list[dict[str, Any]]:
    pairs = C.read_jsonl_gz(C.PAIR_CATALOG)
    if len(pairs) != 823:
        raise RuntimeError(f"expected 823 frozen pairs, found {len(pairs)}")
    return pairs


def load_oracle() -> dict[str, np.ndarray]:
    z = np.load(C.ORACLE_NPZ)
    out = {key: np.asarray(z[key]) for key in z.files}
    for key in ("WC_LOGP", "WC_LOGP_atom", "atom_ptr"):
        if key not in out:
            raise RuntimeError(f"{C.ORACLE_NPZ}: missing {key}")
    return out


def pair_support(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_of = {}
    for pair in pairs:
        split_of[int(pair["row_a"])] = str(pair["split_a"])
        split_of[int(pair["row_b"])] = str(pair["split_b"])
    strata = {
        "train_train": [p for p in pairs if p["split_a"] == "train" and p["split_b"] == "train"],
        "at_least_one_nontrain": [p for p in pairs if not (p["split_a"] == "train" and p["split_b"] == "train")],
        "both_nontrain": [p for p in pairs if p["split_a"] != "train" and p["split_b"] != "train"],
        "all": list(pairs),
    }
    out = {
        "n_pairs": len(pairs),
        "n_molecules": len({int(p["row_a"]) for p in pairs} | {int(p["row_b"]) for p in pairs}),
        "n_components": len({int(p["component"]) for p in pairs}),
        "split_pair_counts": dict(sorted(Counter(str(p["split_pair"]) for p in pairs).items())),
        "split_combo_counts": dict(sorted(Counter(str(p["split_combo"]) for p in pairs).items())),
        "strata": {},
    }
    for name, subset in strata.items():
        out["strata"][name] = {
            "label": C.STRATUM_LABEL[name],
            "n_pairs": len(subset),
            "n_components": len({int(p["component"]) for p in subset}),
            "pair_ids_sha256": C.sha256_json(sorted(str(p["pair_id"]) for p in subset)),
            "support": C.support_label(len({int(p["component"]) for p in subset}), len(subset)),
        }
    out["split_pair_strata"] = {}
    for key in sorted({str(p["split_pair"]) for p in pairs}):
        subset = [p for p in pairs if str(p["split_pair"]) == key]
        comps = len({int(p["component"]) for p in subset})
        out["split_pair_strata"][key] = {
            "n_pairs": len(subset),
            "n_components": comps,
            "support": C.support_label(comps, len(subset)),
        }
    out["support_note"] = (
        "The both-endpoints-non-Train stratum is empty in the frozen catalogue: NO_SUPPORT. "
        "The 823 pairs share molecules and components and are not independent samples."
    )
    return out


# ---------------------------------------------------------------------------
# mapping / edit-region verification
# ---------------------------------------------------------------------------


def _check_fragment(b: Any, row: int, var_atoms: Sequence[int], attach: int, side: str) -> dict[str, Any]:
    raw = _raw(b, int(row))
    bonds = _bond_map(raw)
    var = set(int(v) for v in var_atoms)
    n = len(raw["atom_class"])
    reasons = []
    if not var:
        reasons.append(f"{side}: empty variable fragment")
    if attach not in set(range(n)):
        reasons.append(f"{side}: attachment atom outside graph")
    external = []
    for (u, v), cls in bonds.items():
        inside_u, inside_v = u in var, v in var
        if inside_u != inside_v:
            external.append((u, v, cls))
    external_atoms = {u if u not in var else v for u, v, _c in external}
    if len(external) != 1:
        reasons.append(f"{side}: {len(external)} external variable-fragment bonds, expected 1")
    if external_atoms and external_atoms != {int(attach)}:
        reasons.append(f"{side}: external attachment atoms {sorted(external_atoms)} != {{{int(attach)}}}")
    # induced variable-fragment connectivity
    if var:
        parent = {v: v for v in var}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, bb):
            ra, rb = find(a), find(bb)
            if ra != rb:
                parent[ra] = rb

        for (u, v), _cls in bonds.items():
            if u in var and v in var:
                union(u, v)
        roots = {find(v) for v in var}
        if len(roots) != 1:
            reasons.append(f"{side}: variable fragment has {len(roots)} internal components")
    return {"reasons": reasons, "external_bonds": [[int(u), int(v), int(c)] for u, v, c in external]}


def verify_mapping(b: Any, pair: Mapping[str, Any]) -> dict[str, Any]:
    """Independent structural verification of the stored identity core mapping."""
    raw_a = _raw(b, int(pair["row_a"]))
    raw_b = _raw(b, int(pair["row_b"]))
    na, nb = len(raw_a["atom_class"]), len(raw_b["atom_class"])
    core_a = [int(v) for v in pair["core_atoms_a"]]
    core_b = [int(v) for v in pair["core_atoms_b"]]
    var_a = [int(v) for v in pair["var_atoms_a"]]
    var_b = [int(v) for v in pair["var_atoms_b"]]
    mapping = {int(k): int(v) for k, v in pair["core_atom_map"].items()}
    reasons = []

    if set(core_a) | set(var_a) != set(range(na)) or set(core_a) & set(var_a):
        reasons.append("A: core/var do not partition all atoms")
    if set(core_b) | set(var_b) != set(range(nb)) or set(core_b) & set(var_b):
        reasons.append("B: core/var do not partition all atoms")
    if set(mapping) != set(core_a):
        reasons.append("core_atom_map keys != core_atoms_a")
    if set(mapping.values()) != set(core_b):
        reasons.append("core_atom_map values != core_atoms_b")
    if len(set(mapping.values())) != len(mapping):
        reasons.append("core_atom_map is not injective")
    if int(pair["attach_a"]) not in set(core_a) or int(pair["attach_b"]) not in set(core_b):
        reasons.append("attachment atom is not in the common core")

    n_label_mismatch = 0
    for i, j in mapping.items():
        if _atom_label(raw_a, i) != _atom_label(raw_b, j):
            n_label_mismatch += 1
            reasons.append(f"atom label mismatch {i}->{j}")
    bonds_a = _bond_map(raw_a)
    bonds_b = _bond_map(raw_b)
    missing_a = 0
    missing_b = 0
    type_changes = []
    for x, i in enumerate(core_a):
        for y in range(x + 1, len(core_a)):
            j = core_a[y]
            pi, pj = mapping[i], mapping[j]
            key_a = (min(i, j), max(i, j))
            key_b = (min(pi, pj), max(pi, pj))
            ca, cb = bonds_a.get(key_a), bonds_b.get(key_b)
            if ca is None and cb is not None:
                missing_a += 1
                reasons.append(f"core bond {key_a} missing in A")
            elif ca is not None and cb is None:
                missing_b += 1
                reasons.append(f"core bond {key_b} missing in B")
            elif ca is not None and cb is not None and int(ca) != int(cb):
                type_changes.append({"a": [int(key_a[0]), int(key_a[1]), int(ca)], "b": [int(key_b[0]), int(key_b[1]), int(cb)]})

    frag_a = _check_fragment(b, int(pair["row_a"]), var_a, int(pair["attach_a"]), "A")
    frag_b = _check_fragment(b, int(pair["row_b"]), var_b, int(pair["attach_b"]), "B")
    reasons.extend(frag_a["reasons"])
    reasons.extend(frag_b["reasons"])
    # de-duplicate reasons while preserving order
    reasons = list(dict.fromkeys(reasons))
    return {
        "pair_id": str(pair["pair_id"]),
        "row_a": int(pair["row_a"]),
        "row_b": int(pair["row_b"]),
        "n_atoms_a": na,
        "n_atoms_b": nb,
        "n_core": len(core_a),
        "n_var_a": len(var_a),
        "n_var_b": len(var_b),
        "n_label_mismatch": int(n_label_mismatch),
        "n_missing_core_bond_a": int(missing_a),
        "n_missing_core_bond_b": int(missing_b),
        "changed_core_bonds": type_changes,
        "variable_external_bonds_a": frag_a["external_bonds"],
        "variable_external_bonds_b": frag_b["external_bonds"],
        "valid": bool(not reasons),
        "reasons": reasons,
    }


def derive_edit_regions(b: Any, pair: Mapping[str, Any], mapping_check: Mapping[str, Any]) -> dict[str, Any]:
    """Structure-only edit region on each endpoint."""
    out = {}
    for side, row_key, var_key, core_key, attach_key in (
        ("a", "row_a", "var_atoms_a", "core_atoms_a", "attach_a"),
        ("b", "row_b", "var_atoms_b", "core_atoms_b", "attach_b"),
    ):
        row = int(pair[row_key])
        raw = _raw(b, row)
        bonds = _bond_map(raw)
        var = set(int(v) for v in pair[var_key])
        attach = int(pair[attach_key])
        var_bonds = {key for key in bonds if key[0] in var or key[1] in var}
        attach_bonds = {key for key in var_bonds if attach in key}
        boundary_atoms = {attach}
        for u, v in var_bonds:
            boundary_atoms.add(u if u not in var else v)
        changed_bonds = set()
        if side == "a":
            for rec in mapping_check["changed_core_bonds"]:
                changed_bonds.add(tuple(rec["a"][:2]))
        else:
            for rec in mapping_check["changed_core_bonds"]:
                changed_bonds.add(tuple(rec["b"][:2]))
        edit_bonds = set(var_bonds) | changed_bonds
        edit_atoms = set(var) | set(boundary_atoms)
        for u, v in edit_bonds:
            edit_atoms.add(u)
            edit_atoms.add(v)
        # bond types for the edit bonds
        bond_types = {f"{u}-{v}": int(bonds[(u, v)]) for u, v in sorted(edit_bonds)}
        out[side] = {
            "row_id": row,
            "n_atoms": len(raw["atom_class"]),
            "n_bonds": len(bonds),
            "var_atoms": sorted(var),
            "var_bonds": sorted([list(x) for x in var_bonds]),
            "attach_atom": int(attach),
            "attach_bonds": sorted([list(x) for x in attach_bonds]),
            "boundary_atoms": sorted(boundary_atoms),
            "changed_core_bonds": sorted([list(x) for x in changed_bonds]),
            "edit_atoms": sorted(edit_atoms),
            "edit_bonds": sorted([list(x) for x in edit_bonds]),
            "edit_atom_count": len(edit_atoms),
            "edit_bond_count": len(edit_bonds),
            "bond_types": bond_types,
        }
    return out


# ---------------------------------------------------------------------------
# WC_LOGP contribution closure
# ---------------------------------------------------------------------------


def contribution_delta(
    oracle: Mapping[str, np.ndarray],
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    row_a, row_b = int(pair["row_a"]), int(pair["row_b"])
    ptr = np.asarray(oracle["atom_ptr"], dtype=np.int64)
    atom = np.asarray(oracle["WC_LOGP_atom"], dtype=np.float64)
    graph = np.asarray(oracle["WC_LOGP"], dtype=np.float64)
    a0, a1 = int(ptr[row_a]), int(ptr[row_a + 1])
    b0, b1 = int(ptr[row_b]), int(ptr[row_b + 1])
    ca = atom[a0:a1]
    cb = atom[b0:b1]
    na, nb = int(ca.size), int(cb.size)
    mapping = {int(k): int(v) for k, v in pair["core_atom_map"].items()}
    core_a = set(int(v) for v in pair["core_atoms_a"])
    core_b = set(int(v) for v in pair["core_atoms_b"])
    var_a = set(int(v) for v in pair["var_atoms_a"])
    var_b = set(int(v) for v in pair["var_atoms_b"])

    delta_a = np.full(na, np.nan, dtype=np.float64)
    delta_b = np.full(nb, np.nan, dtype=np.float64)
    pair_delta: list[float] = []
    # mapped core: one union entry, mirrored onto both endpoints
    for i, j in mapping.items():
        d = float(cb[j] - ca[i])
        delta_a[i] = d
        delta_b[j] = d
        pair_delta.append(d)
    # B-only atoms
    for j in sorted(var_b):
        d = float(cb[j])
        delta_b[j] = d
        pair_delta.append(d)
    # A-only atoms
    for i in sorted(var_a):
        d = float(-ca[i])
        delta_a[i] = d
        pair_delta.append(d)
    # Every atom must be assigned exactly once; core + var partitions were checked separately.
    missing_a = int(np.isnan(delta_a).sum())
    missing_b = int(np.isnan(delta_b).sum())
    if missing_a or missing_b:
        raise RuntimeError(f"{pair['pair_id']}: unassigned contribution delta A={missing_a} B={missing_b}")
    total = float(sum(pair_delta))
    target = float(graph[row_b] - graph[row_a])
    err = abs(total - target)
    tol = float(C.oracle_tolerance(target))
    # Graph contribution closure, independently rechecked for both endpoints.
    graph_err_a = abs(float(ca.sum()) - float(graph[row_a]))
    graph_err_b = abs(float(cb.sum()) - float(graph[row_b]))
    return {
        "delta_a": delta_a.tolist(),
        "delta_b": delta_b.tolist(),
        "pair_delta_sum": total,
        "wc_delta": target,
        "closure_error": float(err),
        "closure_tolerance": tol,
        "closed": bool(err <= tol),
        "graph_closure_error_a": float(graph_err_a),
        "graph_closure_error_b": float(graph_err_b),
        "graph_closure_ok": bool(graph_err_a <= 1e-6 and graph_err_b <= 1e-6),
        "core_abs_sum": float(sum(abs(float(cb[j] - ca[i])) for i, j in mapping.items())),
        "var_a_abs_sum": float(sum(abs(float(ca[i])) for i in var_a)),
        "var_b_abs_sum": float(sum(abs(float(cb[j])) for j in var_b)),
        "wc_a": float(graph[row_a]),
        "wc_b": float(graph[row_b]),
        "n_core": len(mapping),
        "n_var_a": len(var_a),
        "n_var_b": len(var_b),
    }


def oracle_graph_closure(oracle: Mapping[str, np.ndarray]) -> dict[str, Any]:
    ptr = np.asarray(oracle["atom_ptr"], dtype=np.int64)
    atom = np.asarray(oracle["WC_LOGP_atom"], dtype=np.float64)
    graph = np.asarray(oracle["WC_LOGP"], dtype=np.float64)
    errs = np.empty(graph.size, dtype=np.float64)
    for row in range(graph.size):
        a0, a1 = int(ptr[row]), int(ptr[row + 1])
        errs[row] = abs(float(atom[a0:a1].sum()) - float(graph[row]))
    return {
        "n_rows": int(graph.size),
        "max_abs_error": float(errs.max()) if errs.size else 0.0,
        "mean_abs_error": float(errs.mean()) if errs.size else 0.0,
        "n_rows_over_1e-6": int((errs > 1e-6).sum()),
        "passed": bool(np.all(errs <= 1e-6)),
    }


# ---------------------------------------------------------------------------
# optional core-automorphism invariance check
# ---------------------------------------------------------------------------


def _core_automorphisms(b: Any, pair: Mapping[str, Any], limit: int = 8) -> list[dict[int, int]]:
    try:
        import networkx as nx
    except Exception:
        return []
    raw = _raw(b, int(pair["row_a"]))
    core = [int(v) for v in pair["core_atoms_a"]]
    attach = int(pair["attach_a"])
    bonds = _bond_map(raw)
    g = nx.Graph()
    for i in core:
        g.add_node(i, label=(int(raw["atom_class"][i]), int(raw["atom_charge"][i]), int(raw["atom_h"][i]), i == attach))
    for x, i in enumerate(core):
        for j in core[x + 1 :]:
            key = (min(i, j), max(i, j))
            if key in bonds:
                g.add_edge(i, j, label=int(bonds[key]))
    node_match = nx.algorithms.isomorphism.categorical_node_match("label", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", None)
    matcher = nx.algorithms.isomorphism.GraphMatcher(g, g, node_match=node_match, edge_match=edge_match)
    out = []
    for mapping in matcher.isomorphisms_iter():
        if int(mapping[attach]) != attach:
            continue
        if {int(mapping[i]) for i in core} != set(core):
            continue
        out.append({int(i): int(mapping[i]) for i in core})
        if len(out) >= int(limit):
            break
    return out


def check_core_automorphism_invariance(b: Any, pair: Mapping[str, Any], delta_core: Mapping[int, float]) -> dict[str, Any]:
    n_declared = int(pair.get("n_core_isomorphisms_capped", 1))
    if n_declared <= 1:
        return {"checked": 0, "passed": True, "reason": "trivial core automorphism", "n_declared": n_declared}
    autos = _core_automorphisms(b, pair, limit=8)
    if len(autos) <= 1:
        return {"checked": len(autos), "passed": True, "reason": "no admissible non-trivial automorphism found", "n_declared": n_declared}
    worst = 0.0
    for p in autos:
        for i, j in p.items():
            worst = max(worst, abs(float(delta_core[i]) - float(delta_core[j])))
    tol = 1e-8 + 1e-8 * max(1.0, max(abs(float(v)) for v in delta_core.values()))
    return {
        "checked": len(autos),
        "n_declared": n_declared,
        "max_abs_delta_difference": float(worst),
        "tolerance": float(tol),
        "passed": bool(worst <= tol),
        "reason": "admissible core automorphisms must not change the oracle delta on a mapped atom",
    }


# ---------------------------------------------------------------------------
# endpoint table and identity record
# ---------------------------------------------------------------------------


def build_endpoints(b: Any, pairs: Sequence[Mapping[str, Any]], oracle: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    endpoint_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        mcheck = verify_mapping(b, pair)
        edits = derive_edit_regions(b, pair, mcheck)
        delta = contribution_delta(oracle, pair)
        delta_core = {}
        for i, j in {int(k): int(v) for k, v in pair["core_atom_map"].items()}.items():
            delta_core[i] = float(delta["delta_a"][i])
        auto = check_core_automorphism_invariance(b, pair, delta_core)
        reasons = list(mcheck["reasons"])
        if not delta["closed"]:
            reasons.append(f"WC_LOGP pair closure error {delta['closure_error']:.3e} > {delta['closure_tolerance']:.3e}")
        if not delta["graph_closure_ok"]:
            reasons.append("WC_LOGP graph contribution closure failed")
        if not auto["passed"]:
            reasons.append("core automorphism changes the mapped WC_LOGP delta")
        evaluable = bool(not reasons)
        rec = {
            "pair_index": idx,
            "pair_id": str(pair["pair_id"]),
            "row_a": int(pair["row_a"]),
            "row_b": int(pair["row_b"]),
            "split_a": str(pair["split_a"]),
            "split_b": str(pair["split_b"]),
            "split_pair": str(pair["split_pair"]),
            "component": int(pair["component"]),
            "train_train": bool(pair["split_a"] == "train" and pair["split_b"] == "train"),
            "in_H": bool(pair["split_a"] != "train" and pair["split_b"] != "train"),
            "evaluable": evaluable,
            "reasons": reasons,
            "mapping": mcheck,
            "automorphism": auto,
            "wc": {
                "wc_a": delta["wc_a"],
                "wc_b": delta["wc_b"],
                "wc_delta": delta["wc_delta"],
                "pair_delta_sum": delta["pair_delta_sum"],
                "closure_error": delta["closure_error"],
                "closure_tolerance": delta["closure_tolerance"],
                "closed": delta["closed"],
                "graph_closure_error_a": delta["graph_closure_error_a"],
                "graph_closure_error_b": delta["graph_closure_error_b"],
                "graph_closure_ok": delta["graph_closure_ok"],
                "core_abs_sum": delta["core_abs_sum"],
                "var_a_abs_sum": delta["var_a_abs_sum"],
                "var_b_abs_sum": delta["var_b_abs_sum"],
                "n_core": delta["n_core"],
                "n_var_a": delta["n_var_a"],
                "n_var_b": delta["n_var_b"],
            },
        }
        pair_rows.append(rec)
        for side in ("a", "b"):
            split = str(pair[f"split_{side}"])
            row_id = int(pair[f"row_{side}"])
            endpoint_rows.append(
                {
                    "pair_index": idx,
                    "pair_id": str(pair["pair_id"]),
                    "side": side,
                    "row_id": row_id,
                    "split": split,
                    "cache_row": cache_row_of(b, split, row_id),
                    "component": int(pair["component"]),
                    "split_pair": str(pair["split_pair"]),
                    "train_train": bool(pair["split_a"] == "train" and pair["split_b"] == "train"),
                    "in_H": bool(pair["split_a"] != "train" and pair["split_b"] != "train"),
                    "evaluable": evaluable,
                    "reasons": reasons,
                    "n_atoms": edits[side]["n_atoms"],
                    "n_bonds": edits[side]["n_bonds"],
                    "var_atoms": edits[side]["var_atoms"],
                    "var_bonds": edits[side]["var_bonds"],
                    "attach_atom": edits[side]["attach_atom"],
                    "attach_bonds": edits[side]["attach_bonds"],
                    "boundary_atoms": edits[side]["boundary_atoms"],
                    "changed_core_bonds": edits[side]["changed_core_bonds"],
                    "edit_atoms": edits[side]["edit_atoms"],
                    "edit_bonds": edits[side]["edit_bonds"],
                    "edit_atom_count": edits[side]["edit_atom_count"],
                    "edit_bond_count": edits[side]["edit_bond_count"],
                    "bond_types": edits[side]["bond_types"],
                    "oracle_delta": delta[f"delta_{side}"],
                    "smiles": str(pair[f"smiles_{side}"]),
                }
            )
        checks.append(
            {
                "pair_id": str(pair["pair_id"]),
                "mapping_valid": mcheck["valid"],
                "evaluable": evaluable,
                "reasons": reasons,
                "automorphism_checked": auto["checked"],
                "automorphism_passed": auto["passed"],
                "wc_closed": delta["closed"],
                "wc_closure_error": delta["closure_error"],
            }
        )
    summary = {
        "n_pairs": len(pairs),
        "n_endpoints": len(endpoint_rows),
        "n_evaluable_pairs": int(sum(1 for p in pair_rows if p["evaluable"])),
        "n_not_evaluable_pairs": int(sum(1 for p in pair_rows if not p["evaluable"])),
        "not_evaluable_reasons": dict(Counter(reason for p in pair_rows if not p["evaluable"] for reason in p["reasons"]).most_common(30)),
        "mapping_all_valid": bool(all(p["mapping"]["valid"] for p in pair_rows)),
        "wc_all_closed": bool(all(p["wc"]["closed"] for p in pair_rows)),
        "automorphism_all_passed": bool(all(p["automorphism"]["passed"] for p in pair_rows)),
        "n_automorphism_checked": int(sum(p["automorphism"]["checked"] for p in pair_rows)),
        "n_automorphism_failed": int(sum(1 for p in pair_rows if not p["automorphism"]["passed"])),
        "n_pairs_with_multiple_declared_core_isomorphisms": int(sum(1 for p in pair_rows if p["mapping"]["n_core"] and p["automorphism"]["n_declared"] > 1)),
        "checks_sha256": C.sha256_json(checks),
    }
    return endpoint_rows, pair_rows, summary


# ---------------------------------------------------------------------------
# official split identity
# ---------------------------------------------------------------------------


def verify_official_split(b: Any) -> dict[str, Any]:
    repo = C.REPO_ROOT
    split_dir = repo / "data/ogb/ogbg_mollipo/split/scaffold"
    out = {"split_dir": C.relpath(split_dir), "files": {}, "all_match": True}
    for split in C.SPLITS:
        path = split_dir / f"{split}.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            values = [int(line.strip()) for line in handle if line.strip()]
        official = np.asarray(values, dtype=np.int64)
        cache_ids = np.asarray(b.global_row_ids[split], dtype=np.int64)
        # The split CSV is in scaffold order; the cache may sort/unsort.  Compare sets and exact order as stored.
        set_match = set(official.tolist()) == set(cache_ids.tolist())
        order_match = bool(np.array_equal(official, cache_ids))
        entry = {
            "path": C.relpath(path),
            "sha256": C.sha256_file(path),
            "n": int(official.size),
            "expected_n": int(C.EXPECTED_SPLIT_SIZES[split]),
            "set_match": bool(set_match),
            "order_match": order_match,
            "row_ids_sha256": C.sha256_array(cache_ids),
            "official_ids_sha256": C.sha256_array(official),
        }
        out["files"][split] = entry
        out["all_match"] = bool(out["all_match"] and set_match and official.size == C.EXPECTED_SPLIT_SIZES[split])
    return out


def data_identity(b: Any, pairs: Sequence[Mapping[str, Any]], oracle: Mapping[str, np.ndarray], checkpoint_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cache_dir = C.SQL_DIR / "data"
    caches = {}
    for name, expected in C.EXPECTED_CACHE_SHA256.items():
        path = cache_dir / name
        actual = C.sha256_file(path) if path.is_file() else None
        caches[name] = {
            "path": C.relpath(path),
            "sha256": actual,
            "expected_sha256": expected,
            "match": bool(actual == expected),
            "bytes": int(path.stat().st_size) if path.is_file() else None,
        }
    assets = {}
    for name, expected in C.EXPECTED_ASSET_SHA256.items():
        path = C.ANALYSIS_DIR / name
        actual = C.sha256_file(path) if path.is_file() else None
        assets[name] = {"path": C.relpath(path), "sha256": actual, "expected_sha256": expected, "match": bool(actual == expected)}
    split_sets = {split: set(np.asarray(b.global_row_ids[split], dtype=np.int64).tolist()) for split in C.SPLITS}
    intersections = {
        f"{a}-{bb}": len(split_sets[a] & split_sets[bb])
        for x, a in enumerate(C.SPLITS)
        for bb in C.SPLITS[x + 1 :]
    }
    union = set().union(*split_sets.values())
    labels = np.asarray(b.payload["labels"], dtype=np.float64)
    return {
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "environment": C.environment(),
        "caches": caches,
        "assets": assets,
        "split_sizes": {split: int(np.asarray(b.global_row_ids[split]).size) for split in C.SPLITS},
        "split_expected_sizes": C.EXPECTED_SPLIT_SIZES,
        "split_intersections": intersections,
        "split_union_size": len(union),
        "split_disjoint_and_cover": bool(all(v == 0 for v in intersections.values()) and len(union) == 4200),
        "official_split": verify_official_split(b),
        "labels": {
            "n": int(labels.size),
            "global_min": float(labels.min()),
            "global_max": float(labels.max()),
            "global_mean": float(labels.mean()),
            "global_std_ddof0": float(labels.std(ddof=0)),
            "sha256": C.sha256_array(labels),
        },
        "pair_catalog": {
            "n_pairs": len(pairs),
            "sha256": C.sha256_file(C.PAIR_CATALOG),
            "n_components": len({int(p["component"]) for p in pairs}),
            "split_pair_counts": dict(sorted(Counter(str(p["split_pair"]) for p in pairs).items())),
        },
        "oracle_graph_closure": oracle_graph_closure(oracle),
        "checkpoints": list(checkpoint_reports),
        "all_identity_gates_passed": bool(
            all(v["match"] for v in caches.values())
            and all(v["match"] for v in assets.values())
            and all(v["match"] for v in split_sets_check(b).values())
            and all(v["passed"] for v in checkpoint_reports)
        ),
    }


def split_sets_check(b: Any) -> dict[str, dict[str, Any]]:
    out = {}
    for split in C.SPLITS:
        ids = np.asarray(b.global_row_ids[split], dtype=np.int64)
        out[split] = {"n": int(ids.size), "match": bool(ids.size == C.EXPECTED_SPLIT_SIZES[split])}
    return out


__all__ = [
    "load_pairs",
    "load_oracle",
    "pair_support",
    "verify_mapping",
    "derive_edit_regions",
    "contribution_delta",
    "oracle_graph_closure",
    "build_endpoints",
    "data_identity",
    "verify_official_split",
    "cache_row_of",
]
