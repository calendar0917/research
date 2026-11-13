"""Frozen raw-graph adapter and structure-cache helpers (no model, no labels).

The edited raw row is pushed through the *frozen* adapter
(``mb_rdkit.raw_row_to_graph``) and the *frozen* ``lipo_v1`` feature builder, so
every input block (AF51 / BF13 / scalars9 / S1 codes / S2 path codes / global42)
is recomputed from the edited graph.  Nothing here is fitted or cached across
rounds; the frozen Train caches produced by the previous pipeline are only *read*
for the zero-rewire identity gate and for the Train input-range definition.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import graphlib as G

import lcrc_common as PREP_COMMON
import lcrc_data as PREP_DATA
import molecular_graph as MG
from feature_spec import FeatureSpec


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


def view_graph(row: Mapping[str, Any], spec: FeatureSpec) -> MG.MolecularGraph:
    """Frozen raw-graph adapter (rebuilds AF51/BF13 from the given graph)."""
    return C.RD.raw_row_to_graph(row, int(row["mol_index"]), spec)


def build_cache(
    entries: Sequence[tuple[str, Mapping[str, Any]]], spec: FeatureSpec, *, split: str
) -> dict[str, Any]:
    """Build one label-free structure cache from ``(view_key, raw_row)`` entries."""
    graphs = [view_graph(row, spec) for _vk, row in entries]
    for g, (vk, _row) in zip(graphs, entries):
        if int(g.molecule_id) != int(vk.split(":")[0]):
            raise ValueError(f"view {vk}: graph molecule id {g.molecule_id} != row id")
    cache = PREP_DATA.build_cache(graphs, spec, split=split)
    cache["view_ids"] = [str(vk) for vk, _row in entries]
    cache["rows"] = np.asarray([int(vk.split(":")[0]) for vk, _row in entries], dtype=np.int64)
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
    """Byte-level comparison of two cache slices (used by the zero-rewire gate)."""
    diff = {}
    for key in _ARRAY_KEYS:
        a = np.asarray(actual[key])
        b = np.asarray(expected[key])
        diff[key] = bool(a.shape == b.shape and np.array_equal(a, b))
    for key in ("n_atoms", "n_bonds", "n_components", "n_junctions", "n_ring_systems"):
        diff[key] = bool(int(actual[key]) == int(expected[key]))
    return {"all_equal": all(diff.values()), "fields": diff}


def af51_difference(cache: Mapping[str, Any], i_o: int, i_view: int) -> dict[str, Any]:
    """Per-atom AF51 / atom-type / atom-scalar difference between two views."""
    a = cache_slice(cache, i_o)
    b = cache_slice(cache, i_view)
    if a["n_atoms"] != b["n_atoms"]:
        return {"same_atom_count": False, "max_abs_atom_feat_diff": float("inf"), "max_abs_atom_scalar_diff": float("inf")}
    da = np.abs(np.asarray(a["atom_feat"], dtype=np.float64) - np.asarray(b["atom_feat"], dtype=np.float64))
    ds = np.abs(np.asarray(a["atom_scalar"], dtype=np.float64) - np.asarray(b["atom_scalar"], dtype=np.float64))
    na = np.asarray(a["nbr_code"])
    nb = np.asarray(b["nbr_code"])
    return {
        "same_atom_count": True,
        "max_abs_atom_feat_diff": float(da.max()) if da.size else 0.0,
        "max_abs_atom_scalar_diff": float(ds.max()) if ds.size else 0.0,
        "atom_feat_equal": bool(np.array_equal(np.asarray(a["atom_feat"]), np.asarray(b["atom_feat"]))),
        "atom_scalar_equal": bool(np.array_equal(np.asarray(a["atom_scalar"]), np.asarray(b["atom_scalar"]))),
        "atom_type_equal": bool(np.array_equal(np.asarray(a["atom_type"]), np.asarray(b["atom_type"]))),
        "n_atoms_differing_af51": int((da.max(axis=1) > 0).sum()) if da.size else 0,
        "s1_nbr_code_equal": bool(na.shape == nb.shape and np.array_equal(na, nb)),
        "s1_n_differing_entries": int((na != nb).sum()) if na.shape == nb.shape else None,
        "s2_path_code_equal": bool(
            np.asarray(a["path_cat"]).shape == np.asarray(b["path_cat"]).shape
            and np.array_equal(np.asarray(a["path_cat"]), np.asarray(b["path_cat"]))
        ),
        "global_equal": bool(np.array_equal(np.asarray(a["global"]), np.asarray(b["global"]))),
    }


__all__ = [
    "row_from_artifact",
    "view_graph",
    "build_cache",
    "cache_slice",
    "compare_cached_molecule",
    "af51_difference",
    "MG",
    "FeatureSpec",
]
