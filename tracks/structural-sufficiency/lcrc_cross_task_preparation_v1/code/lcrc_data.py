"""Schema-driven LCRC-A data assembly, batching and Fit-only normalisation.

The module has four deliberately separate layers:

1. :func:`build_molecule` turns one canonical :class:`MolecularGraph` into the
   label-free LCRC-A arrays (S1 joint codes, S2 typed paths, atom scalars,
   global statistics).
2. :func:`concat_molecules` concatenates per-molecule arrays with explicit
   pointers.  Bond indices remain molecule-local until collation.
3. :func:`collate` offsets every index into a batch and returns only structural
   tensors.  No label field ever exists in a cache or a batch.
4. :func:`fit_norm` computes standardisation from an explicit Fit cache only.

S1 is a genuine joint statistic: the code is
``atom_type[neighbour] * n_bond_types + (bond_type - 1)`` for the same real
incident bond.  S2 is a joint typed-path statistic:
``atom_type[middle] * n_bond_types**2 + (bond1 - 1) * n_bond_types + (bond2 - 1)``
plus the joint ``(end type, bond2)`` code and the real triangle-closure flag.
Neither is a concatenation of marginals.

All structural codes use ``int32`` (or wider).  The lipo_v1 path vocabulary has
400 categories, so the historical ``uint8`` representation would wrap.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lcrc_common as C
import molecular_graph as MG
from adapters import CHARGE_UNK, H_UNK
from feature_spec import FeatureSpec

# ---------------------------------------------------------------------------
# per-molecule assembly
# ---------------------------------------------------------------------------


def _feature_block_offsets(spec: FeatureSpec) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    off = 0
    for f in spec.atom_feature_fields:
        out[f.name] = (off, off + f.dim)
        off += f.dim
    return out


def _legacy_scalar_from_blocks(name: str, graph: MG.MolecularGraph, blocks: Mapping[str, tuple[int, int]]) -> np.ndarray:
    af = graph.atom_feat
    if name == "in_ring":
        lo, hi = blocks["in_ring"]
        return af[:, lo:hi].reshape(-1)
    if name == "min_ring_size_over_8":
        lo, hi = blocks["min_ring_size"]
        onehot = af[:, lo:hi]
        size = np.sum(onehot * np.asarray([3, 4, 5, 6, 7, 8], dtype=np.float32).reshape(1, -1), axis=1)
        return size / 8.0
    if name == "ring_bond_count_over_4":
        lo, _hi = blocks["ring_bonds"]
        return af[:, lo] / 4.0
    if name == "terminal":
        lo, _hi = blocks["roles"]
        return af[:, lo]
    if name == "fused_or_spiro":
        lo, _hi = blocks["ring_bonds"]
        return (af[:, lo] >= 3.0).astype(np.float32)
    raise KeyError(name)


def _atom_scalars(
    graph: MG.MolecularGraph,
    analysis: MG.GraphAnalysis,
    s2_records: Sequence[Sequence[Mapping[str, int]]],
    spec: FeatureSpec,
) -> np.ndarray:
    n = analysis.n_atoms
    path_counts = np.asarray([len(r) for r in s2_records], dtype=np.float32)
    closure_counts = np.asarray([sum(int(x["closure"]) for x in r) for r in s2_records], dtype=np.float32)
    closure_ratio = np.divide(closure_counts, np.maximum(path_counts, 1.0))
    n_d2 = np.sum(analysis.dist == 2, axis=1).astype(np.float32)
    blocks = _feature_block_offsets(spec)
    columns: list[np.ndarray] = []
    for f in spec.atom_scalar_fields:
        name = f.name
        if name == "log1p_n_d2":
            v = np.log1p(n_d2)
        elif name == "log1p_n_paths2":
            v = np.log1p(path_counts)
        elif name == "closure_ratio":
            v = closure_ratio
        elif name == "degree_over_4":
            v = analysis.degree.astype(np.float32) / 4.0
        elif name == "log1p_degree":
            v = np.log1p(analysis.degree.astype(np.float32))
        elif name == "in_ring":
            v = _legacy_scalar_from_blocks(name, graph, blocks) if spec.name == "legacy_zinc" else analysis.ring_atom.astype(np.float32)
        elif name == "min_ring_size_over_8":
            v = _legacy_scalar_from_blocks(name, graph, blocks) if spec.name == "legacy_zinc" else analysis.min_ring_size.astype(np.float32) / 8.0
        elif name == "ring_bond_count_over_4":
            v = _legacy_scalar_from_blocks(name, graph, blocks) if spec.name == "legacy_zinc" else analysis.ring_bond_count.astype(np.float32) / 4.0
        elif name == "terminal" or name == "is_terminal":
            v = _legacy_scalar_from_blocks("terminal", graph, blocks) if spec.name == "legacy_zinc" else (analysis.degree == 1).astype(np.float32)
        elif name == "fused_or_spiro" or name == "is_fused_or_spiro":
            v = _legacy_scalar_from_blocks("fused_or_spiro", graph, blocks) if spec.name == "legacy_zinc" else (analysis.ring_bond_count >= 3).astype(np.float32)
        else:
            raise KeyError(f"unknown atom scalar {name!r}")
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape != (n,):
            raise ValueError(f"scalar {name!r} has shape {v.shape}, expected {(n,)}")
        columns.append(v)
    out = np.stack(columns, axis=1).astype(np.float32)
    if out.shape != (n, spec.n_scalars):
        raise ValueError(f"atom scalar shape {out.shape} != {(n, spec.n_scalars)}")
    return out


def _lipo_global(graph: MG.MolecularGraph, analysis: MG.GraphAnalysis, spec: FeatureSpec) -> np.ndarray:
    n = int(analysis.n_atoms)
    m = int(analysis.n_bonds)
    deg = analysis.degree.astype(np.float64)
    dist = analysis.dist
    iu = np.triu_indices(n, 1)
    pair_dist = dist[iu] if n > 1 else np.zeros(0, dtype=np.int64)
    finite = pair_dist >= 0
    unreachable = pair_dist < 0
    finite_dist = pair_dist[finite]
    n_pairs = int(pair_dist.size)
    diameter = float(finite_dist.max()) if finite_dist.size else 0.0
    mean_finite = float(finite_dist.mean()) if finite_dist.size else 0.0
    frac_unreachable = float(unreachable.sum() / n_pairs) if n_pairs else 0.0
    frac_finite_ge3 = float(((finite_dist >= 3).sum()) / n_pairs) if n_pairs else 0.0

    ring_atom = analysis.ring_atom
    ring_bond = analysis.is_ring_bond
    n_ring_atoms = int(ring_atom.sum())
    n_junctions = int(analysis.n_junctions)
    ring_size_counts = np.zeros(6, dtype=np.float64)
    for atom in np.flatnonzero(ring_atom):
        size = int(analysis.min_ring_size[atom])
        if size <= 0:
            continue
        slot = min(max(size - 3, 0), 5)
        ring_size_counts[slot] += 1.0
    if n_ring_atoms:
        ring_size_counts /= float(n_ring_atoms)

    bond_type = graph.bond_type
    bond_counts = np.zeros(5, dtype=np.float64)
    if m:
        for t in bond_type.tolist():
            bond_counts[int(t) - 1] += 1.0
        bond_counts /= float(m)
    atom_type = graph.atom_type
    is_aromatic_type = np.asarray(
        [bool(meta.get("aromatic", False)) for meta in spec.atom_type_metadata],
        dtype=bool,
    )
    aromatic_atom_fraction = float(is_aromatic_type[atom_type].mean()) if n else 0.0
    unknown_atom_fraction = float((atom_type == spec.atom_unk_index).mean()) if n and spec.atom_unk_index is not None else 0.0

    charges = np.asarray(graph.raw_atom_charge, dtype=np.int64)
    hs = np.asarray(graph.raw_atom_h, dtype=np.int64)
    known_charge = charges != CHARGE_UNK
    known_h = hs != H_UNK
    charged_atom_fraction = float(((charges != 0) & known_charge).mean()) if n else 0.0
    mean_formal_charge = float(np.where(known_charge, charges, 0).mean()) if n else 0.0
    mean_hydrogens = float(np.where(known_h, hs, 0).mean()) if n else 0.0
    max_abs_formal_charge = float(np.abs(np.where(known_charge, charges, 0)).max()) if n else 0.0
    unknown_charge_fraction = float((~known_charge).mean()) if n else 0.0
    unknown_h_fraction = float((~known_h).mean()) if n else 0.0

    junction_deg2 = 0
    junction_deg3plus = 0
    junction_ring = 0
    for v in range(n):
        d = int(deg[v])
        if d < 2:
            continue
        count = d * (d - 1) // 2
        if d == 2:
            junction_deg2 += count
        else:
            junction_deg3plus += count
        if bool(ring_atom[v]):
            junction_ring += count

    values = {
        "log1p_n_atoms": np.log1p(n),
        "log1p_n_bonds": np.log1p(m),
        "log1p_n_junctions": np.log1p(n_junctions),
        "cyclomatic_number": float(m - n + analysis.n_components),
        "n_components": float(analysis.n_components),
        "ring_atom_fraction": float(ring_atom.mean()) if n else 0.0,
        "ring_bond_fraction": float(ring_bond.mean()) if m else 0.0,
        "n_ring_systems": float(analysis.n_ring_systems),
        "diameter_max_finite": diameter,
        "mean_finite_pair_distance": mean_finite,
        "frac_unreachable_pairs": frac_unreachable,
        "frac_finite_pairs_ge3": frac_finite_ge3,
        "bonds_per_atom": float(m) / max(n, 1),
        "log1p_n_ring_atoms": np.log1p(n_ring_atoms),
        "max_degree": float(deg.max()) if n else 0.0,
        "terminal_atom_fraction": float((deg == 1).mean()) if n else 0.0,
        "junction_per_bond": float(n_junctions) / max(m, 1),
        "frac_junction_centers_ring": (float(junction_ring) / n_junctions) if n_junctions else 0.0,
        "frac_deg2_junctions": (float(junction_deg2) / n_junctions) if n_junctions else 0.0,
        "frac_deg3plus_junctions": (float(junction_deg3plus) / n_junctions) if n_junctions else 0.0,
        "bond_single_fraction": float(bond_counts[0]),
        "bond_double_fraction": float(bond_counts[1]),
        "bond_triple_fraction": float(bond_counts[2]),
        "bond_aromatic_fraction": float(bond_counts[3]),
        "bond_unknown_fraction": float(bond_counts[4]),
        "ring_size3_fraction": float(ring_size_counts[0]),
        "ring_size4_fraction": float(ring_size_counts[1]),
        "ring_size5_fraction": float(ring_size_counts[2]),
        "ring_size6_fraction": float(ring_size_counts[3]),
        "ring_size7_fraction": float(ring_size_counts[4]),
        "ring_size8plus_fraction": float(ring_size_counts[5]),
        "fused_ring_atom_fraction": float((analysis.ring_bond_count >= 3).mean()) if n else 0.0,
        "aromatic_atom_fraction": aromatic_atom_fraction,
        "unknown_atom_fraction": unknown_atom_fraction,
        "charged_atom_fraction": charged_atom_fraction,
        "mean_formal_charge": mean_formal_charge,
        "mean_hydrogens": mean_hydrogens,
        "distinct_atom_types": float(len(set(int(x) for x in atom_type.tolist()))),
        "distinct_bond_types": float(len(set(int(x) for x in bond_type.tolist()))),
        "max_abs_formal_charge": max_abs_formal_charge,
        "unknown_charge_fraction": unknown_charge_fraction,
        "unknown_hydrogen_fraction": unknown_h_fraction,
    }
    out = np.asarray([values[f.name] for f in spec.global_feature_fields], dtype=np.float32)
    if out.shape != (spec.global_dim,):
        raise ValueError(f"global shape {out.shape} != {(spec.global_dim,)}")
    if not np.isfinite(out).all():
        raise ValueError("non-finite lipo_v1 global feature")
    return out


def build_molecule(graph: MG.MolecularGraph, spec: FeatureSpec, *, validate: bool = True) -> dict[str, Any]:
    """Build one molecule's label-free LCRC-A arrays."""
    analysis = MG.analyse_graph(graph)
    if not spec.supports_disconnected and analysis.n_components != 1:
        raise ValueError(
            f"molecule {graph.molecule_id}: {analysis.n_components} components but {spec.name} is connected-only"
        )
    s1_records = MG.enumerate_s1(graph.atom_type, analysis.adj, spec.n_bond_types)
    s2_records = MG.enumerate_s2(graph.atom_type, analysis.adj, spec.n_bond_types, analysis.dist)

    nbr_counts = np.asarray([len(r) for r in s1_records], dtype=np.int64)
    nbr_ptr = np.concatenate([[0], np.cumsum(nbr_counts)]).astype(np.int64)
    nbr_code = (
        np.asarray([code for row in s1_records for code in row], dtype=np.int32)
        if any(s1_records)
        else np.zeros(0, dtype=np.int32)
    )
    path_counts = np.asarray([len(r) for r in s2_records], dtype=np.int64)
    path_ptr = np.concatenate([[0], np.cumsum(path_counts)]).astype(np.int64)
    if int(path_ptr[-1]):
        flat = [rec for row in s2_records for rec in row]
        path_cat = np.asarray([rec["path_cat"] for rec in flat], dtype=np.int32)
        path_end63 = np.asarray([rec["end63"] for rec in flat], dtype=np.int32)
        path_end = np.asarray([rec["end"] for rec in flat], dtype=np.int64)
        path_closure = np.asarray([rec["closure"] for rec in flat], dtype=np.int8)
    else:
        path_cat = np.zeros(0, dtype=np.int32)
        path_end63 = np.zeros(0, dtype=np.int32)
        path_end = np.zeros(0, dtype=np.int64)
        path_closure = np.zeros(0, dtype=np.int8)

    atom_scalar = _atom_scalars(graph, analysis, s2_records, spec)
    if spec.global_feature_source.startswith("frozen P3"):
        if graph.global_feat is None:
            raise ValueError("legacy graph lacks the frozen global33 vector")
        global_feature = np.asarray(graph.global_feat, dtype=np.float32).reshape(-1)
    else:
        global_feature = _lipo_global(graph, analysis, spec)

    mol: dict[str, Any] = {
        "n_atoms": np.int64(analysis.n_atoms),
        "n_bonds": np.int64(analysis.n_bonds),
        "atom_feat": np.asarray(graph.atom_feat, dtype=np.float32),
        "atom_type": np.asarray(graph.atom_type, dtype=np.int64),
        "atom_scalar": atom_scalar,
        "bond_feat": np.asarray(graph.bond_feat, dtype=np.float32),
        "bond_type": np.asarray(graph.bond_type, dtype=np.int64),
        "bond_index": np.stack([analysis.bond_u, analysis.bond_v], axis=1).astype(np.int64)
        if analysis.n_bonds
        else np.zeros((0, 2), dtype=np.int64),
        "nbr_ptr": nbr_ptr,
        "nbr_code": nbr_code,
        "path_ptr": path_ptr,
        "path_cat": path_cat,
        "path_end63": path_end63,
        "path_end": path_end,
        "path_closure": path_closure,
        "global": global_feature,
        "n_components": np.int64(analysis.n_components),
        "n_junctions": np.int64(analysis.n_junctions),
        "n_ring_systems": np.int64(analysis.n_ring_systems),
        "molecule_id": str(graph.molecule_id),
        "smiles": str(graph.smiles),
    }
    if validate:
        validate_molecule(mol, spec)
    return mol


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate_molecule(mol: Mapping[str, Any], spec: FeatureSpec) -> None:
    n = int(mol["n_atoms"])
    m = int(mol["n_bonds"])
    if n < 1:
        raise ValueError("molecule has no atoms")
    atom_feat = np.asarray(mol["atom_feat"])
    bond_feat = np.asarray(mol["bond_feat"])
    atom_type = np.asarray(mol["atom_type"])
    bond_type = np.asarray(mol["bond_type"])
    atom_scalar = np.asarray(mol["atom_scalar"])
    bond_index = np.asarray(mol["bond_index"])
    nbr_ptr = np.asarray(mol["nbr_ptr"])
    nbr_code = np.asarray(mol["nbr_code"])
    path_ptr = np.asarray(mol["path_ptr"])
    path_cat = np.asarray(mol["path_cat"])
    path_end63 = np.asarray(mol["path_end63"])
    path_end = np.asarray(mol["path_end"])
    path_closure = np.asarray(mol["path_closure"])
    global_feature = np.asarray(mol["global"])

    if atom_feat.shape != (n, spec.atom_feat_dim):
        raise ValueError(f"atom_feat shape {atom_feat.shape}")
    if bond_feat.shape != (m, spec.bond_feat_dim):
        raise ValueError(f"bond_feat shape {bond_feat.shape}")
    if atom_type.shape != (n,):
        raise ValueError(f"atom_type shape {atom_type.shape}")
    if bond_type.shape != (m,):
        raise ValueError(f"bond_type shape {bond_type.shape}")
    if atom_scalar.shape != (n, spec.n_scalars):
        raise ValueError(f"atom_scalar shape {atom_scalar.shape}")
    if global_feature.shape != (spec.global_dim,):
        raise ValueError(f"global shape {global_feature.shape}")
    if bond_index.shape != (m, 2):
        raise ValueError(f"bond_index shape {bond_index.shape}")
    if nbr_ptr.shape != (n + 1,) or path_ptr.shape != (n + 1,):
        raise ValueError("pointer shape mismatch")
    if nbr_code.shape != (int(nbr_ptr[-1]),):
        raise ValueError("nbr_code length mismatch")
    if not (path_cat.shape == path_end63.shape == path_end.shape == path_closure.shape == (int(path_ptr[-1]),)):
        raise ValueError("path array length mismatch")
    for arr, name in (
        (atom_feat, "atom_feat"),
        (bond_feat, "bond_feat"),
        (atom_scalar, "atom_scalar"),
        (global_feature, "global"),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf")
    if np.any(nbr_ptr[1:] < nbr_ptr[:-1]) or np.any(path_ptr[1:] < path_ptr[:-1]):
        raise ValueError("non-monotonic pointer")
    if nbr_ptr[0] != 0 or path_ptr[0] != 0:
        raise ValueError("pointer does not start at zero")
    if np.any(atom_type < 0) or np.any(atom_type >= spec.n_atom_types):
        raise ValueError("atom_type outside vocabulary")
    if m and (np.any(bond_type < 1) or np.any(bond_type > spec.n_bond_types)):
        raise ValueError("bond_type outside vocabulary")
    if np.any(nbr_code < 0) or np.any(nbr_code >= spec.n_nbr_codes):
        raise ValueError("nbr_code outside range")
    if np.any(path_cat < 0) or np.any(path_cat >= spec.n_path_codes):
        raise ValueError("path_cat outside range")
    if np.any(path_end63 < 0) or np.any(path_end63 >= spec.n_nbr_codes):
        raise ValueError("path_end63 outside range")
    if np.any(path_end < 0) or np.any(path_end >= n):
        raise ValueError("path_end outside atom range")
    if np.any((path_closure < 0) | (path_closure > 1)):
        raise ValueError("path_closure outside {0,1}")
    if m:
        if np.any(bond_index < 0) or np.any(bond_index >= n):
            raise ValueError("bond_index outside atom range")
        if np.any(bond_index[:, 0] >= bond_index[:, 1]):
            raise ValueError("bond_index must satisfy u < v")
        if len(set(map(tuple, bond_index.tolist()))) != m:
            raise ValueError("duplicate physical bond")
        if int(m) != int(nbr_ptr[-1]) // 2:
            raise ValueError("physical bond count does not match S1 degree sum / 2")
    else:
        if int(nbr_ptr[-1]) != 0:
            raise ValueError("bondless molecule has S1 codes")
    # Path count identity: sum over roots of sum_{w in N(v)} (deg(w) - 1).
    # Reconstruct degrees from the S1 pointer counts; in a simple graph the
    # number of incident bonds is the degree.
    degree = np.diff(nbr_ptr)
    expected_paths = 0
    # A path record is rooted at v; the root degree is nbr_ptr[v+1]-nbr_ptr[v].
    # The production arrays do not store the middle atom, so use the canonical
    # bond list to compute the exact identity independently.
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in bond_index.tolist():
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    for v in range(n):
        expected_paths += sum(max(len(adj[w]) - 1, 0) for w in adj[v])
    if int(path_ptr[-1]) != int(expected_paths):
        raise ValueError(f"path count {int(path_ptr[-1])} != expected {expected_paths}")
    if int(degree.sum()) != 2 * m:
        raise ValueError("degree sum != 2 * physical bond count")
    if not spec.supports_disconnected and int(mol["n_components"]) != 1:
        raise ValueError("connected-only schema received a disconnected molecule")


# ---------------------------------------------------------------------------
# concatenation
# ---------------------------------------------------------------------------


def _cat(parts: Sequence[np.ndarray], dtype) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=dtype)
    return np.concatenate([np.asarray(p, dtype=dtype) for p in parts], axis=0)


def concat_molecules(mols: Sequence[Mapping[str, Any]], spec: FeatureSpec, *, split: str = "") -> dict[str, Any]:
    if not mols:
        raise ValueError("cannot concatenate an empty molecule list")
    atom_counts = [int(m["n_atoms"]) for m in mols]
    bond_counts = [int(m["n_bonds"]) for m in mols]
    atom_ptr = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int64)
    bond_ptr = np.concatenate([[0], np.cumsum(bond_counts)]).astype(np.int64)

    nbr_ptr_parts: list[np.ndarray] = []
    path_ptr_parts: list[np.ndarray] = []
    nbr_offset = 0
    path_offset = 0
    for m in mols:
        nbr_ptr_parts.append(np.asarray(m["nbr_ptr"][:-1], dtype=np.int64) + nbr_offset)
        nbr_offset += int(len(m["nbr_code"]))
        path_ptr_parts.append(np.asarray(m["path_ptr"][:-1], dtype=np.int64) + path_offset)
        path_offset += int(len(m["path_cat"]))
    nbr_ptr_parts.append(np.asarray([nbr_offset], dtype=np.int64))
    path_ptr_parts.append(np.asarray([path_offset], dtype=np.int64))
    nbr_ptr = np.concatenate(nbr_ptr_parts).astype(np.int64)
    path_ptr = np.concatenate(path_ptr_parts).astype(np.int64)

    bond_index_parts = [np.asarray(m["bond_index"], dtype=np.int64) for m in mols]
    bond_index = (
        np.concatenate(bond_index_parts, axis=0) if bond_index_parts else np.zeros((0, 2), dtype=np.int64)
    )
    cache: dict[str, Any] = {
        "atom_ptr": atom_ptr,
        "bond_ptr": bond_ptr,
        "nbr_ptr": nbr_ptr,
        "path_ptr": path_ptr,
        "atom_feat": _cat([m["atom_feat"] for m in mols], np.float32),
        "atom_type": _cat([m["atom_type"] for m in mols], np.int64),
        "atom_scalar": _cat([m["atom_scalar"] for m in mols], np.float32),
        "bond_feat": _cat([m["bond_feat"] for m in mols], np.float32),
        "bond_type": _cat([m["bond_type"] for m in mols], np.int64),
        "bond_index": bond_index.astype(np.int64),
        "nbr_code": _cat([m["nbr_code"] for m in mols], np.int32),
        "path_cat": _cat([m["path_cat"] for m in mols], np.int32),
        "path_end63": _cat([m["path_end63"] for m in mols], np.int32),
        "path_end": _cat([m["path_end"] for m in mols], np.int64),
        "path_closure": _cat([m["path_closure"] for m in mols], np.int8),
        "global": np.stack([np.asarray(m["global"], dtype=np.float32) for m in mols], axis=0),
        "n_components": np.asarray([int(m["n_components"]) for m in mols], dtype=np.int64),
        "n_junctions": np.asarray([int(m["n_junctions"]) for m in mols], dtype=np.int64),
        "n_ring_systems": np.asarray([int(m["n_ring_systems"]) for m in mols], dtype=np.int64),
        "molecule_ids": [str(m["molecule_id"]) for m in mols],
        "smiles": [str(m["smiles"]) for m in mols],
        "n_molecules": int(len(mols)),
        "split": str(split),
        "meta": {
            "feature_spec_name": spec.name,
            "feature_spec_hash": spec.hash(),
            "n_molecules": int(len(mols)),
            "n_atoms_total": int(atom_ptr[-1]),
            "n_bonds_total": int(bond_ptr[-1]),
            "n_s1_codes": int(nbr_ptr[-1]),
            "n_s2_paths": int(path_ptr[-1]),
            "split": str(split),
            "labels_present": False,
            "created_utc": C.utcnow(),
        },
    }
    return cache


def build_cache(graphs: Sequence[MG.MolecularGraph], spec: FeatureSpec, *, split: str = "") -> dict[str, Any]:
    mols = []
    t0 = time.perf_counter()
    for i, graph in enumerate(graphs):
        mols.append(build_molecule(graph, spec))
        if (i + 1) % 1000 == 0:
            C.log(f"[build {split}] {i + 1}/{len(graphs)} molecules {time.perf_counter() - t0:.1f}s")
    cache = concat_molecules(mols, spec, split=split)
    cache["meta"]["build_seconds"] = float(time.perf_counter() - t0)
    return cache


# ---------------------------------------------------------------------------
# cache-level validation
# ---------------------------------------------------------------------------


def validate_cache(cache: Mapping[str, Any], spec: FeatureSpec, *, max_molecules: int | None = None) -> dict[str, Any]:
    n_mol = int(cache["n_molecules"])
    if max_molecules is not None:
        n_mol = min(n_mol, int(max_molecules))
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    if atom_ptr.shape != (cache["n_molecules"] + 1,) or bond_ptr.shape != (cache["n_molecules"] + 1,):
        raise ValueError("molecule pointer shape mismatch")
    if nbr_ptr.shape != (int(atom_ptr[-1]) + 1,) or path_ptr.shape != (int(atom_ptr[-1]) + 1,):
        raise ValueError("per-atom pointer shape mismatch")
    for arr, name in (
        (cache["atom_feat"], "atom_feat"),
        (cache["atom_scalar"], "atom_scalar"),
        (cache["bond_feat"], "bond_feat"),
        (cache["global"], "global"),
    ):
        if not np.isfinite(np.asarray(arr)).all():
            raise ValueError(f"{name} contains NaN/Inf")
    if np.any(atom_ptr[1:] < atom_ptr[:-1]) or np.any(bond_ptr[1:] < bond_ptr[:-1]):
        raise ValueError("non-monotonic molecule pointer")
    if np.any(nbr_ptr[1:] < nbr_ptr[:-1]) or np.any(path_ptr[1:] < path_ptr[:-1]):
        raise ValueError("non-monotonic per-atom pointer")
    if np.any(np.asarray(cache["atom_type"]) < 0) or np.any(np.asarray(cache["atom_type"]) >= spec.n_atom_types):
        raise ValueError("cache atom_type outside vocabulary")
    if np.any(np.asarray(cache["bond_type"]) < 1) or np.any(np.asarray(cache["bond_type"]) > spec.n_bond_types):
        raise ValueError("cache bond_type outside vocabulary")
    if np.any(np.asarray(cache["nbr_code"]) < 0) or np.any(np.asarray(cache["nbr_code"]) >= spec.n_nbr_codes):
        raise ValueError("cache nbr_code outside range")
    if np.any(np.asarray(cache["path_cat"]) < 0) or np.any(np.asarray(cache["path_cat"]) >= spec.n_path_codes):
        raise ValueError("cache path_cat outside range")
    if np.any(np.asarray(cache["path_end63"]) < 0) or np.any(np.asarray(cache["path_end63"]) >= spec.n_nbr_codes):
        raise ValueError("cache path_end63 outside range")
    if np.any((np.asarray(cache["path_closure"]) < 0) | (np.asarray(cache["path_closure"]) > 1)):
        raise ValueError("cache path_closure outside {0,1}")
    for g in range(n_mol):
        a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        n_atoms = a1 - a0
        if n_atoms < 1:
            raise ValueError(f"molecule {g} has no atoms")
        bi = np.asarray(cache["bond_index"])[b0:b1]
        if bi.size:
            if np.any(bi < 0) or np.any(bi >= n_atoms):
                raise ValueError(f"molecule {g}: bond endpoint outside local atom range")
            if np.any(bi[:, 0] >= bi[:, 1]):
                raise ValueError(f"molecule {g}: bond u >= v")
        pe = np.asarray(cache["path_end"])[int(path_ptr[a0]) : int(path_ptr[a1])] if a1 > a0 else np.zeros(0, np.int64)
        # path_end is molecule-local in the cache; check the local range.
        if pe.size and (np.any(pe < 0) or np.any(pe >= n_atoms)):
            raise ValueError(f"molecule {g}: path_end outside local atom range")
        if int(nbr_ptr[a1] - nbr_ptr[a0]) != 2 * (b1 - b0):
            raise ValueError(f"molecule {g}: S1 degree sum != 2 * bonds")
    return {
        "n_molecules_checked": int(n_mol),
        "n_atoms": int(atom_ptr[n_mol]),
        "n_bonds": int(bond_ptr[n_mol]),
        "n_s1_codes": int(nbr_ptr[atom_ptr[n_mol]]),
        "n_s2_paths": int(path_ptr[atom_ptr[n_mol]]),
        "finite": True,
        "index_ranges_valid": True,
        "labels_present": False,
    }


# ---------------------------------------------------------------------------
# Fit-only normalisation
# ---------------------------------------------------------------------------


def _scale_from_std(std: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Fit-only scale with an explicit constant-feature policy.

    A feature that is constant on Fit has no meaningful standard deviation.
    Dividing by the legacy ``1e-6`` epsilon would turn any out-of-fit value
    (for example ``n_components == 2`` or an UNK fraction) into an enormous
    input.  Constant features therefore use scale ``1.0`` and are recorded in
    the norm metadata; non-constant features keep the legacy epsilon.
    """
    std = np.asarray(std, dtype=np.float32)
    const = np.flatnonzero(std <= np.float32(1e-6)).tolist()
    scale = np.where(std > np.float32(1e-6), std, np.float32(1.0)).astype(np.float32)
    return scale, const


def fit_norm(cache: Mapping[str, Any], spec: FeatureSpec, *, source: str | None = None) -> dict[str, Any]:
    scalars = np.asarray(cache["atom_scalar"], dtype=np.float64)
    globals_ = np.asarray(cache["global"], dtype=np.float64)
    if scalars.shape[1] != spec.n_scalars or globals_.shape[1] != spec.global_dim:
        raise ValueError("cache dimensions do not match feature spec")
    scalar_mean = scalars.mean(0).astype(np.float32)
    scalar_scale, scalar_const = _scale_from_std(scalars.std(0))
    global_mean = globals_.mean(0).astype(np.float32)
    global_scale, global_const = _scale_from_std(globals_.std(0))
    return {
        "source": str(source or f"computed_from_{cache.get('split', 'fit')}_structures"),
        "fit_split": str(cache.get("split", "")),
        "valid_or_test_used": False,
        "zero_variance_policy": "constant_fit_features_use_scale_1.0",
        "scalar_constant_indices": scalar_const,
        "global_constant_indices": global_const,
        "scalar_mean": scalar_mean,
        "scalar_scale": scalar_scale,
        "global_mean": global_mean,
        "global_scale": global_scale,
    }


def norm_fingerprint(norm: Mapping[str, Any]) -> str:
    parts = []
    for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale"):
        parts.append(np.asarray(norm[k], dtype=np.float32).tobytes())
    policy = str(norm.get("zero_variance_policy", ""))
    return C.sha256_bytes(b"".join(parts) + str(norm.get("source", "")).encode("utf-8") + policy.encode("utf-8"))


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def _ranges(ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    starts = ptr[rows].astype(np.int64)
    ends = ptr[rows + 1].astype(np.int64)
    counts = ends - starts
    if len(rows) == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)])
    batch = np.repeat(np.arange(len(rows), dtype=np.int64), counts)
    local = np.concatenate([np.arange(c, dtype=np.int64) for c in counts])
    return idx, batch, local, counts


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("rows must be 1-D")
    n_molecules = int(cache["n_molecules"])
    if np.any(rows < 0) or np.any(rows >= n_molecules):
        raise ValueError("row index outside cache")
    if len(rows) != len(set(int(x) for x in rows.tolist())):
        raise ValueError("duplicate row in batch")
    B = int(len(rows))
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    atom_idx, atom_batch, atom_local, atom_counts = _ranges(atom_ptr, rows)
    bond_idx, bond_batch, _bond_local, bond_counts = _ranges(bond_ptr, rows)
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64) if B else np.zeros(0, np.int64)

    bi = np.asarray(cache["bond_index"])[bond_idx].astype(np.int64) if len(bond_idx) else np.zeros((0, 2), np.int64)
    if len(bi):
        bi = bi + atom_off[bond_batch].reshape(-1, 1)

    nbr_ptr = np.asarray(cache["nbr_ptr"])
    nbr_counts = nbr_ptr[atom_idx + 1] - nbr_ptr[atom_idx] if len(atom_idx) else np.zeros(0, np.int64)
    atom_local_idx = np.arange(len(atom_idx), dtype=np.int64)
    if len(atom_idx):
        nbr_idx = np.concatenate(
            [np.arange(nbr_ptr[a], nbr_ptr[a] + c, dtype=np.int64) for a, c in zip(atom_idx, nbr_counts)]
        )
    else:
        nbr_idx = np.zeros(0, dtype=np.int64)
    nbr_atom = np.repeat(atom_local_idx, nbr_counts) if len(atom_idx) else np.zeros(0, np.int64)

    path_ptr = np.asarray(cache["path_ptr"])
    path_counts = path_ptr[atom_idx + 1] - path_ptr[atom_idx] if len(atom_idx) else np.zeros(0, np.int64)
    if len(atom_idx):
        path_idx = np.concatenate(
            [np.arange(path_ptr[a], path_ptr[a] + c, dtype=np.int64) for a, c in zip(atom_idx, path_counts)]
        )
    else:
        path_idx = np.zeros(0, dtype=np.int64)
    path_graph = np.repeat(atom_batch, path_counts) if len(atom_idx) else np.zeros(0, np.int64)
    path_atom = np.repeat(atom_local_idx, path_counts) if len(atom_idx) else np.zeros(0, np.int64)
    if len(path_idx):
        path_end = np.asarray(cache["path_end"])[path_idx].astype(np.int64) + atom_off[path_graph]
    else:
        path_end = np.zeros(0, dtype=np.int64)

    def _t(arr, dtype):
        return torch.from_numpy(np.asarray(arr, dtype=dtype))

    out = {
        "n_graphs": B,
        "atom_feat": _t(np.asarray(cache["atom_feat"])[atom_idx], np.float32),
        "atom_type": _t(np.asarray(cache["atom_type"])[atom_idx], np.int64),
        "atom_scalar": _t(np.asarray(cache["atom_scalar"])[atom_idx], np.float32),
        "atom_batch": _t(atom_batch, np.int64),
        "atom_off": _t(atom_off, np.int64),
        "bond_feat": _t(np.asarray(cache["bond_feat"])[bond_idx], np.float32),
        "bond_type": _t(np.asarray(cache["bond_type"])[bond_idx], np.int64),
        "bond_index": _t(bi, np.int64),
        "bond_batch": _t(bond_batch, np.int64),
        "nbr_atom": _t(nbr_atom, np.int64),
        "nbr_code": _t(np.asarray(cache["nbr_code"])[nbr_idx], np.int64),
        "path_atom": _t(path_atom, np.int64),
        "path_cat": _t(np.asarray(cache["path_cat"])[path_idx], np.int64),
        "path_end63": _t(np.asarray(cache["path_end63"])[path_idx], np.int64),
        "path_end": _t(path_end, np.int64),
        "path_closure": _t(np.asarray(cache["path_closure"])[path_idx], np.int64),
        "path_graph": _t(path_graph, np.int64),
        "global": _t(np.asarray(cache["global"])[rows], np.float32),
        "atom_counts": _t(atom_counts, np.int64),
        "bond_counts": _t(bond_counts, np.int64),
    }
    return out


def batch_meta(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    return {
        "n_atoms": (atom_ptr[rows + 1] - atom_ptr[rows]).astype(np.int64),
        "n_bonds": (bond_ptr[rows + 1] - bond_ptr[rows]).astype(np.int64),
    }


# ---------------------------------------------------------------------------
# disk cache (git-ignored)
# ---------------------------------------------------------------------------


def save_cache(cache: Mapping[str, Any], path) -> str:
    p = Path(path)
    C.ensure_dir(p.parent)
    torch.save(dict(cache), p)
    return C.sha256_file(p)


def load_cache(path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


__all__ = [
    "build_molecule",
    "validate_molecule",
    "concat_molecules",
    "build_cache",
    "validate_cache",
    "fit_norm",
    "norm_fingerprint",
    "collate",
    "batch_meta",
    "save_cache",
    "load_cache",
]
