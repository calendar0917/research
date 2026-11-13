"""Label-free view construction: delete exactly one physical bond, rebuild inputs.

A *view* is the original molecule with one undirected physical bond removed:

``O``
    original raw row (zero cuts); must reproduce the frozen structure cache and
    the frozen checkpoint predictions.
``T``
    remove ``c-n`` (the C(=O)-N link).
``K``
    remove ``c-acyl_c`` (the neighbouring C-C link); the C(=O)-N link stays.
``R``
    remove one minimal-cost remote ``q-r`` C-N bridge.

Everything else is kept byte-for-byte: atoms, ``atom_class`` / ``atom_charge``
/ ``atom_h``, all remaining directed bond records with their ``edge_class`` and
``edge_stereo``.  No hydrogen is added, no atom is deleted, no cap is written,
no SMILES is re-parsed into the model input and nothing is re-sanitised.  Only
``n_components`` of the copied raw row is updated, then the frozen raw-graph
adapter and the frozen ``lipo_v1`` feature builder rebuild every input block
(AF51 / BF13 / scalars9 / S1 / S2 / global42) from the cut graph.

The original SMILES is carried as metadata only (plotting / provenance).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import motif as M

import molecular_graph as MG
import lcrc_common as PREP_COMMON
import lcrc_data as PREP_DATA
from feature_spec import FeatureSpec


# ---------------------------------------------------------------------------
# raw rows
# ---------------------------------------------------------------------------


def row_from_artifact(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one artifact row into plain ``dict[str, np.ndarray]`` form."""
    return {
        "smiles": str(row["smiles"]),
        "canonical_smiles": str(row["canonical_smiles"]),
        "mol_index": int(row["mol_index"]),
        "atom_class": np.asarray(row["atom_class"], dtype=np.int64).reshape(-1).copy(),
        "atom_charge": np.asarray(row["atom_charge"], dtype=np.int64).reshape(-1).copy(),
        "atom_h": np.asarray(row["atom_h"], dtype=np.int64).reshape(-1).copy(),
        "edge_index": np.asarray(row["edge_index"], dtype=np.int64).reshape(2, -1).copy(),
        "edge_class": np.asarray(row["edge_class"], dtype=np.int64).reshape(-1).copy(),
        "edge_stereo": np.asarray(row["edge_stereo"], dtype=np.int64).reshape(-1).copy(),
        "n_components": int(row["n_components"]),
    }


def n_components_of_row(row: Mapping[str, Any]) -> int:
    bu, bv, _bc, _bs = M.physical_bonds(row)
    n = int(np.asarray(row["atom_class"]).shape[0])
    comp = M.component_ids(n, bu, bv)
    return int(comp.max()) + 1 if n else 0


def cut_row(row: Mapping[str, Any], a: int, b: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copy of ``row`` with the physical bond ``{a,b}`` deleted.

    Both directed records are removed together with their ``edge_class`` and
    ``edge_stereo`` entries.  The atoms and every other record are unchanged.
    ``n_components`` is increased by one (the cut bond must be a bridge; the
    caller verifies this on the graph).
    """
    a, b = int(a), int(b)
    if a == b:
        raise ValueError("cannot cut a self-loop")
    ei = np.asarray(row["edge_index"], dtype=np.int64)
    ec = np.asarray(row["edge_class"], dtype=np.int64).reshape(-1)
    es = np.asarray(row["edge_stereo"], dtype=np.int64).reshape(-1)
    keep = ~(((ei[0] == a) & (ei[1] == b)) | ((ei[0] == b) & (ei[1] == a)))
    removed = int((~keep).sum())
    if removed != 2:
        raise ValueError(f"cut {a}-{b}: removed {removed} directed records, expected exactly 2")
    if not np.any((ei[0] == a) & (ei[1] == b)) or not np.any((ei[0] == b) & (ei[1] == a)):
        raise ValueError(f"cut {a}-{b}: directed record pair is incomplete")
    out = {
        "smiles": str(row["smiles"]),
        "canonical_smiles": str(row["canonical_smiles"]),
        "mol_index": int(row["mol_index"]),
        "atom_class": np.asarray(row["atom_class"], dtype=np.int64).reshape(-1).copy(),
        "atom_charge": np.asarray(row["atom_charge"], dtype=np.int64).reshape(-1).copy(),
        "atom_h": np.asarray(row["atom_h"], dtype=np.int64).reshape(-1).copy(),
        "edge_index": ei[:, keep].copy(),
        "edge_class": ec[keep].copy(),
        "edge_stereo": es[keep].copy(),
        "n_components": int(row["n_components"]) + 1,
    }
    return out, {
        "cut_a": a,
        "cut_b": b,
        "removed_directed_records": removed,
        "n_directed_before": int(ei.shape[1]),
        "n_directed_after": int(out["edge_index"].shape[1]),
        "n_components_before": int(row["n_components"]),
        "n_components_after": int(out["n_components"]),
    }


def permute_row(row: Mapping[str, Any], perm: Sequence[int]) -> dict[str, Any]:
    """Relabel atoms so that old atom ``i`` becomes new atom ``perm[i]``."""
    n = int(np.asarray(row["atom_class"]).shape[0])
    perm = np.asarray(perm, dtype=np.int64).reshape(-1)
    if perm.shape != (n,) or sorted(perm.tolist()) != list(range(n)):
        raise ValueError("perm must be a permutation of range(n_atoms)")
    inv = np.argsort(perm)  # new index j holds old atom inv[j]
    ei = np.asarray(row["edge_index"], dtype=np.int64)
    out = {
        "smiles": str(row["smiles"]),
        "canonical_smiles": str(row["canonical_smiles"]),
        "mol_index": int(row["mol_index"]),
        "atom_class": np.asarray(row["atom_class"], dtype=np.int64).reshape(-1)[inv].copy(),
        "atom_charge": np.asarray(row["atom_charge"], dtype=np.int64).reshape(-1)[inv].copy(),
        "atom_h": np.asarray(row["atom_h"], dtype=np.int64).reshape(-1)[inv].copy(),
        "edge_index": perm[ei].copy(),
        "edge_class": np.asarray(row["edge_class"], dtype=np.int64).reshape(-1).copy(),
        "edge_stereo": np.asarray(row["edge_stereo"], dtype=np.int64).reshape(-1).copy(),
        "n_components": int(row["n_components"]),
    }
    return out


# ---------------------------------------------------------------------------
# frozen adapter + cache
# ---------------------------------------------------------------------------


def view_graph(row: Mapping[str, Any], spec: FeatureSpec) -> MG.MolecularGraph:
    """Frozen raw-graph adapter (rebuilds AF51/BF13 from the cut graph)."""
    return C.RD.raw_row_to_graph(row, int(row["mol_index"]), spec)


def build_cache(
    entries: Sequence[tuple[str, Mapping[str, Any]]], spec: FeatureSpec, *, split: str
) -> dict[str, Any]:
    """Build one label-free structure cache from ``(view_id, raw_row)`` entries."""
    graphs = [view_graph(row, spec) for _vid, row in entries]
    for g, (vid, _row) in zip(graphs, entries):
        if int(g.molecule_id) != int(vid.split(":")[0]):
            raise ValueError(f"view {vid}: graph molecule id {g.molecule_id} != row id")
    cache = PREP_DATA.build_cache(graphs, spec, split=split)
    cache["view_ids"] = [str(vid) for vid, _row in entries]
    cache["rows"] = np.asarray([int(vid.split(":")[0]) for vid, _row in entries], dtype=np.int64)
    cache["n_molecules"] = int(len(entries))
    cache["meta"]["labels_present"] = False
    cache["meta"]["view_cache"] = True
    PREP_DATA.validate_cache(cache, spec)
    cache["meta"]["content_fingerprint"] = PREP_COMMON.content_fingerprint(cache)
    return cache


def cache_slice(cache: Mapping[str, Any], i: int) -> dict[str, Any]:
    """Extract one molecule's arrays from a flat cache (local indices)."""
    i = int(i)
    a0, a1 = int(cache["atom_ptr"][i]), int(cache["atom_ptr"][i + 1])
    b0, b1 = int(cache["bond_ptr"][i]), int(cache["bond_ptr"][i + 1])
    p0, p1 = int(cache["path_ptr"][a0]), int(cache["path_ptr"][a1])
    n0, n1 = int(cache["nbr_ptr"][a0]), int(cache["nbr_ptr"][a1])
    return {
        "n_atoms": a1 - a0,
        "n_bonds": b1 - b0,
        "atom_feat": np.asarray(cache["atom_feat"])[a0:a1].copy(),
        "atom_type": np.asarray(cache["atom_type"])[a0:a1].copy(),
        "atom_scalar": np.asarray(cache["atom_scalar"])[a0:a1].copy(),
        "bond_feat": np.asarray(cache["bond_feat"])[b0:b1].copy(),
        "bond_type": np.asarray(cache["bond_type"])[b0:b1].copy(),
        "bond_index": np.asarray(cache["bond_index"])[b0:b1].copy(),
        "nbr_ptr": (np.asarray(cache["nbr_ptr"])[a0 : a1 + 1] - n0).copy(),
        "nbr_code": np.asarray(cache["nbr_code"])[n0:n1].copy(),
        "path_ptr": (np.asarray(cache["path_ptr"])[a0 : a1 + 1] - p0).copy(),
        "path_cat": np.asarray(cache["path_cat"])[p0:p1].copy(),
        "path_end63": np.asarray(cache["path_end63"])[p0:p1].copy(),
        "path_end": np.asarray(cache["path_end"])[p0:p1].copy(),
        "path_closure": np.asarray(cache["path_closure"])[p0:p1].copy(),
        "global": np.asarray(cache["global"])[i].copy(),
        "n_components": int(cache["n_components"][i]),
        "n_junctions": int(cache["n_junctions"][i]),
        "n_ring_systems": int(cache["n_ring_systems"][i]),
    }


_ARRAY_KEYS = (
    "atom_feat",
    "atom_type",
    "atom_scalar",
    "bond_feat",
    "bond_type",
    "bond_index",
    "nbr_ptr",
    "nbr_code",
    "path_ptr",
    "path_cat",
    "path_end63",
    "path_end",
    "path_closure",
    "global",
)


def compare_cached_molecule(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Byte-level comparison of two cache slices (used by the zero-cut gate)."""
    diff = {}
    for key in _ARRAY_KEYS:
        a = np.asarray(actual[key])
        b = np.asarray(expected[key])
        diff[key] = bool(a.shape == b.shape and np.array_equal(a, b))
    for key in ("n_atoms", "n_bonds", "n_components", "n_junctions", "n_ring_systems"):
        diff[key] = bool(int(actual[key]) == int(expected[key]))
    return {"all_equal": all(diff.values()), "fields": diff}


# ---------------------------------------------------------------------------
# structural verification of one view (protocol section 8, items 2 and 3)
# ---------------------------------------------------------------------------


def s2_endpoint_map(graph: MG.MolecularGraph) -> dict[tuple[int, int], set[int]]:
    """``(root, middle) -> {end}`` for the non-backtracking 2-path universe."""
    analysis = MG.analyse_graph(graph)
    records = MG.enumerate_s2(graph.atom_type, analysis.adj, C.DC.N_BOND_TYPES, analysis.dist)
    out: dict[tuple[int, int], set[int]] = {}
    for root, lst in enumerate(records):
        for rec in lst:
            out.setdefault((int(root), int(rec["middle"])), set()).add(int(rec["end"]))
    return out


def verify_cut(
    row: Mapping[str, Any],
    view: Mapping[str, Any],
    *,
    cut_a: int,
    cut_b: int,
    spec: FeatureSpec,
) -> dict[str, Any]:
    """Prove that ``view`` is ``row`` with exactly one physical bond removed."""
    bu0, bv0, bc0, bs0 = M.physical_bonds(row)
    bu1, bv1, bc1, bs1 = M.physical_bonds(view)
    bi0 = {(int(u), int(v)): (int(c), int(s)) for u, v, c, s in zip(bu0, bv0, bc0, bs0)}
    bi1 = {(int(u), int(v)): (int(c), int(s)) for u, v, c, s in zip(bu1, bv1, bc1, bs1)}
    removed = sorted(set(bi0) - set(bi1))
    added = sorted(set(bi1) - set(bi0))
    changed = sorted(k for k in set(bi0) & set(bi1) if bi0[k] != bi1[k])
    atoms_equal = {
        key: bool(np.array_equal(np.asarray(row[key], dtype=np.int64), np.asarray(view[key], dtype=np.int64)))
        for key in ("atom_class", "atom_charge", "atom_h")
    }
    directed_ok = int(np.asarray(view["edge_index"]).shape[1]) + 2 == int(
        np.asarray(row["edge_index"]).shape[1]
    )
    n_comp_before = n_components_of_row(row)
    n_comp_after = n_components_of_row(view)
    g0 = view_graph(row, spec)
    g1 = view_graph(view, spec)
    a0 = MG.analyse_graph(g0)
    a1 = MG.analyse_graph(g1)
    s2_0 = s2_endpoint_map(g0)
    s2_1 = s2_endpoint_map(g1)
    return {
        "removed_physical_bonds": removed,
        "removed_is_requested_cut": removed == [(min(int(cut_a), int(cut_b)), max(int(cut_a), int(cut_b)))],
        "added_physical_bonds": added,
        "changed_bond_attributes": changed,
        "n_physical_bonds_before": len(bi0),
        "n_physical_bonds_after": len(bi1),
        "directed_records_after_equal_before_minus_2": directed_ok,
        "atom_attributes_unchanged": atoms_equal,
        "n_components_before": n_comp_before,
        "n_components_after": n_comp_after,
        "components_increased_by_one": n_comp_after == n_comp_before + 1,
        "raw_n_components_matches_graph": int(view["n_components"]) == n_comp_after,
        "s1_degree_sum_matches_bonds": int(a1.degree.sum()) == 2 * int(a1.n_bonds),
        "s2_total_before": sum(len(v) for v in s2_0.values()),
        "s2_total_after": sum(len(v) for v in s2_1.values()),
        "s2_map_before": s2_0,
        "s2_map_after": s2_1,
        "dist_before": a0.dist,
        "dist_after": a1.dist,
    }


def cut_effect_summary(
    check: Mapping[str, Any], *, kind: str, c: int, n: int, o: int, acyl_c: int
) -> dict[str, Any]:
    """Role-specific expectations for the T / K cut (protocol section 8 item 2)."""
    d1 = np.asarray(check["dist_after"])
    d0 = np.asarray(check["dist_before"])
    s20 = check["s2_map_before"]
    s21 = check["s2_map_after"]
    o_c_n_before = n in s20.get((o, c), set())
    o_c_n_after = n in s21.get((o, c), set())
    n_c_o_before = o in s20.get((n, c), set())
    n_c_o_after = o in s21.get((n, c), set())
    out = {
        "o_c_n_path_before": bool(o_c_n_before),
        "o_c_n_path_after": bool(o_c_n_after),
        "n_c_o_path_before": bool(n_c_o_before),
        "n_c_o_path_after": bool(n_c_o_after),
        "dist_c_n_before": int(d0[c, n]),
        "dist_c_n_after": int(d1[c, n]),
        "dist_c_acyl_before": int(d0[c, acyl_c]),
        "dist_c_acyl_after": int(d1[c, acyl_c]),
    }
    if kind == "T":
        out["expected"] = (
            bool(out["o_c_n_path_before"])
            and bool(out["n_c_o_path_before"])
            and not out["o_c_n_path_after"]
            and not out["n_c_o_path_after"]
            and out["dist_c_n_after"] == M.UNREACHABLE
            and out["dist_c_acyl_after"] == 1
        )
    elif kind == "K":
        out["expected"] = (
            bool(out["o_c_n_path_before"])
            and bool(out["o_c_n_path_after"])
            and bool(out["n_c_o_path_after"])
            and out["dist_c_n_after"] == 1
            and out["dist_c_acyl_after"] == M.UNREACHABLE
        )
    else:  # R
        out["expected"] = out["dist_c_n_after"] == 1 and out["dist_c_acyl_after"] == 1
    out["pass"] = bool(out["expected"])
    return out


__all__ = [
    "row_from_artifact",
    "n_components_of_row",
    "cut_row",
    "permute_row",
    "view_graph",
    "build_cache",
    "cache_slice",
    "compare_cached_molecule",
    "s2_endpoint_map",
    "verify_cut",
    "cut_effect_summary",
]
