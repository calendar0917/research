"""Feature specifications and feature builders for the three task families."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import bp_graph as BG
from feature_spec import (
    FEATURE_DTYPE,
    SCHEMA_VERSION,
    STRUCTURAL_CODE_DTYPE,
    STRUCTURAL_INDEX_DTYPE,
    FeatureField,
    FeatureSpec,
    TensorSpec,
)

# The 9 per-node topology scalars (identical names/order to the historical TU
# schema in ``lcrc_tu_quick_screen_v1/code/tu_common.py``).
SCALAR_FIELDS: tuple[tuple[str, str], ...] = (
    ("log1p_n_d2", "log1p(number of nodes at graph distance exactly 2)"),
    ("log1p_n_paths2", "log1p(number of ordered non-backtracking 2-paths rooted at the node)"),
    ("closure_ratio", "closure-count / path-count (0 when there is no rooted 2-path)"),
    ("log1p_degree", "log1p(degree)"),
    ("in_cycle", "node lies on at least one cycle"),
    ("log1p_min_cycle_size", "log1p(smallest cycle through the node); 0 if acyclic"),
    ("log1p_cycle_degree", "log1p(number of incident cycle edges)"),
    ("is_terminal", "degree == 1"),
    ("cycle_degree_ge3", "cycle_degree >= 3"),
)
SCALAR_DIM = len(SCALAR_FIELDS)

# The 28 global topology fields (identical names/order to the historical TU
# schema; chemical fields are not used).
GLOBAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("log1p_n_atoms", "log1p(number of nodes)"),
    ("log1p_n_bonds", "log1p(number of physical edges after cleaning)"),
    ("log1p_n_junctions", "log1p(number of unordered neighbour pairs at a centre)"),
    ("cyclomatic_number", "n_bonds - n_atoms + n_components (first Betti number)"),
    ("n_components", "number of connected components (isolated nodes included)"),
    ("ring_atom_fraction", "fraction of nodes on a cycle"),
    ("ring_bond_fraction", "fraction of edges on a cycle"),
    ("n_ring_systems", "number of biconnected blocks with at least three edges"),
    ("diameter_max_finite", "maximum finite shortest-path distance inside a component (0 for a single node)"),
    ("mean_finite_pair_distance", "mean finite unordered-pair distance; unreachable pairs excluded"),
    ("frac_unreachable_pairs", "unreachable unordered pairs / all unordered pairs"),
    ("frac_finite_pairs_ge3", "finite unordered pairs at distance >= 3 / all unordered pairs"),
    ("bonds_per_atom", "n_bonds / n_atoms"),
    ("log1p_n_ring_atoms", "log1p(number of nodes on a cycle)"),
    ("max_degree", "maximum node degree"),
    ("terminal_atom_fraction", "fraction of degree-1 nodes"),
    ("junction_per_bond", "n_junctions / n_bonds"),
    ("frac_junction_centers_ring", "fraction of junction centres on a cycle"),
    ("frac_deg2_junctions", "fraction of junctions whose centre has degree 2"),
    ("frac_deg3plus_junctions", "fraction of junctions whose centre has degree >= 3"),
    ("ring_size3_fraction", "cycle-node fraction with minimum cycle length 3"),
    ("ring_size4_fraction", "cycle-node fraction with minimum cycle length 4"),
    ("ring_size5_fraction", "cycle-node fraction with minimum cycle length 5"),
    ("ring_size6_fraction", "cycle-node fraction with minimum cycle length 6"),
    ("ring_size7_fraction", "cycle-node fraction with minimum cycle length 7"),
    ("ring_size8plus_fraction", "cycle-node fraction with minimum cycle length >= 8"),
    ("fused_ring_atom_fraction", "fraction of nodes with cycle_degree >= 3"),
    ("distinct_atom_types", "number of distinct atom types present"),
)
GLOBAL_DIM = len(GLOBAL_FIELDS)

DEGREE_BUCKET_MAX = 16
N_DEGREE_BUCKETS = DEGREE_BUCKET_MAX + 1

PEPTIDE_ATOM_VOCABS: tuple[int, ...] = (119, 4, 12, 12, 6, 6, 6, 2, 2)
PEPTIDE_BOND_VOCABS: tuple[int, ...] = (5, 6, 2)
PEPTIDE_ATOM_NAMES: tuple[str, ...] = (
    "atomic_number",
    "chirality",
    "degree",
    "formal_charge",
    "num_hydrogens",
    "num_radical_electrons",
    "hybridization",
    "aromatic",
    "ring",
)
PEPTIDE_BOND_NAMES: tuple[str, ...] = ("bond_type", "stereo", "conjugated")


def _topology_atom_fields() -> tuple[FeatureField, ...]:
    return (
        FeatureField("degree_bucket", N_DEGREE_BUCKETS, FEATURE_DTYPE, "degree one-hot 0..15,>=16 from the cleaned physical graph", "graph"),
        FeatureField("in_cycle", 1, FEATURE_DTYPE, "node lies on at least one cycle", "biconnected-block analysis"),
        FeatureField("min_cycle_size", 7, FEATURE_DTYPE, "one-hot acyclic,3,4,5,6,7,>=8 minimum cycle length", "smallest cycle through the node"),
        FeatureField("log1p_cycle_degree", 1, FEATURE_DTYPE, "log1p(number of incident cycle edges)", "biconnected-block analysis"),
        FeatureField("cycle_degree_fraction", 1, FEATURE_DTYPE, "cycle_degree / max(degree, 1)", "biconnected-block analysis"),
        FeatureField("is_terminal", 1, FEATURE_DTYPE, "degree == 1", "graph"),
        FeatureField("cycle_degree_ge3", 1, FEATURE_DTYPE, "cycle_degree >= 3", "biconnected-block analysis"),
    )


TOPOLOGY_ATOM_DIM = 29


def _cycle_size_slot(min_cycle_size: np.ndarray) -> np.ndarray:
    m = np.asarray(min_cycle_size, dtype=np.int64)
    slot = np.zeros(m.shape, dtype=np.int64)
    non_zero = m > 0
    slot[non_zero] = np.where(m[non_zero] >= 8, 6, m[non_zero] - 2)
    return slot


def topology_atom_features(analysis: BG.FastAnalysis) -> np.ndarray:
    n = int(analysis.n_atoms)
    out = np.zeros((n, TOPOLOGY_ATOM_DIM), dtype=np.float32)
    bucket = np.minimum(np.asarray(analysis.degree, dtype=np.int64), DEGREE_BUCKET_MAX)
    out[np.arange(n), bucket] = 1.0
    off = N_DEGREE_BUCKETS
    out[:, off] = analysis.ring_atom.astype(np.float32)
    slot = _cycle_size_slot(analysis.min_ring_size)
    out[np.arange(n), off + 1 + slot] = 1.0
    cycle_degree = np.asarray(analysis.ring_bond_count, dtype=np.float64)
    degree = np.asarray(analysis.degree, dtype=np.float64)
    base = off + 1 + 7
    out[:, base + 0] = np.log1p(cycle_degree).astype(np.float32)
    out[:, base + 1] = (cycle_degree / np.maximum(degree, 1.0)).astype(np.float32)
    out[:, base + 2] = (degree == 1).astype(np.float32)
    out[:, base + 3] = (cycle_degree >= 3).astype(np.float32)
    return out


def bond_topology_features(analysis: BG.FastAnalysis) -> np.ndarray:
    m = int(analysis.n_bonds)
    out = np.zeros((m, 1), dtype=np.float32)
    if m:
        out[:, 0] = analysis.is_ring_bond.astype(np.float32)
    return out


def scalar_features(analysis: BG.FastAnalysis, s2_counts: np.ndarray, closure_counts: np.ndarray) -> np.ndarray:
    n = int(analysis.n_atoms)
    path_counts = np.asarray(s2_counts, dtype=np.float64)
    closure_counts = np.asarray(closure_counts, dtype=np.float64)
    closure_ratio = closure_counts / np.maximum(path_counts, 1.0)
    degree = np.asarray(analysis.degree, dtype=np.float64)
    cycle_degree = np.asarray(analysis.ring_bond_count, dtype=np.float64)
    min_cycle = np.asarray(analysis.min_ring_size, dtype=np.float64)
    cols = [
        np.log1p(analysis.n_d2.astype(np.float64)),
        np.log1p(path_counts),
        closure_ratio,
        np.log1p(degree),
        analysis.ring_atom.astype(np.float64),
        np.log1p(min_cycle),
        np.log1p(cycle_degree),
        (degree == 1).astype(np.float64),
        (cycle_degree >= 3).astype(np.float64),
    ]
    out = np.stack(cols, axis=1).astype(np.float32)
    if out.shape != (n, SCALAR_DIM):
        raise RuntimeError(f"scalar shape {out.shape} != {(n, SCALAR_DIM)}")
    return out


def global_features(analysis: BG.FastAnalysis, atom_type: np.ndarray) -> np.ndarray:
    n = int(analysis.n_atoms)
    m = int(analysis.n_bonds)
    deg = np.asarray(analysis.degree, dtype=np.float64)
    dist = analysis.distance
    diameter = float(dist["diameter"])
    mean_finite = float(dist["mean_finite_distance"])
    frac_unreachable = float(dist["frac_unreachable"])
    frac_finite_ge3 = float(dist["frac_finite_ge3"])

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
        "ring_size3_fraction": float(ring_size_counts[0]),
        "ring_size4_fraction": float(ring_size_counts[1]),
        "ring_size5_fraction": float(ring_size_counts[2]),
        "ring_size6_fraction": float(ring_size_counts[3]),
        "ring_size7_fraction": float(ring_size_counts[4]),
        "ring_size8plus_fraction": float(ring_size_counts[5]),
        "fused_ring_atom_fraction": float((analysis.ring_bond_count >= 3).mean()) if n else 0.0,
        "distinct_atom_types": float(len(set(int(x) for x in np.asarray(atom_type).tolist()))),
    }
    out = np.asarray([values[name] for name, _desc in GLOBAL_FIELDS], dtype=np.float32)
    if out.shape != (GLOBAL_DIM,):
        raise RuntimeError(f"global shape {out.shape} != {(GLOBAL_DIM,)}")
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite global feature")
    return out


def _scalar_fields() -> tuple[FeatureField, ...]:
    return tuple(FeatureField(name, 1, FEATURE_DTYPE, desc, "topology") for name, desc in SCALAR_FIELDS)


def _global_fields() -> tuple[FeatureField, ...]:
    return tuple(FeatureField(name, 1, FEATURE_DTYPE, desc, "graph statistics") for name, desc in GLOBAL_FIELDS)


def _output_tensors(atom_dim: int, bond_dim: int, global_dim: int) -> tuple[TensorSpec, ...]:
    code = STRUCTURAL_CODE_DTYPE
    idx = STRUCTURAL_INDEX_DTYPE
    f32 = FEATURE_DTYPE
    return (
        TensorSpec("atom_feat", f"[N, {atom_dim}]", f32, "per-node feature matrix"),
        TensorSpec("atom_type", "[N]", idx, "discrete node type id (0-based)"),
        TensorSpec("atom_scalar", f"[N, {SCALAR_DIM}]", f32, "per-node topology scalars before Train-only standardisation"),
        TensorSpec("bond_feat", f"[M, {bond_dim}]", f32, "per-physical-edge feature matrix"),
        TensorSpec("bond_type", "[M]", idx, "edge type id (1-based)"),
        TensorSpec("bond_index", "[M, 2]", idx, "physical edge endpoints, u < v, sorted"),
        TensorSpec("nbr_code", "[nnz_S1]", code, "joint (neighbour node type, edge type) code"),
        TensorSpec("path_cat", "[nnz_S2]", code, "joint (middle type, edge1, edge2) code"),
        TensorSpec("path_end63", "[nnz_S2]", code, "joint (end node type, edge2) code"),
        TensorSpec("path_end", "[nnz_S2]", idx, "absolute end-node index"),
        TensorSpec("path_closure", "[nnz_S2]", "int8", "1 iff the end node is adjacent to the root"),
        TensorSpec("global", f"[B, {global_dim}]", f32, "graph-level topology statistics before Train-only standardisation"),
    )


def tu_spec(
    dataset: str,
    atom_type_names: Sequence[str],
    atom_unk_index: int | None,
    bond_type_names: Sequence[str],
    bond_unk_id: int | None,
    *,
    continuous_atom_dim: int,
    continuous_atom_names: Sequence[str] = (),
    edge_attribute_dim: int = 0,
    edge_attribute_names: Sequence[str] = (),
    source_provenance: Mapping[str, Any] | None = None,
) -> FeatureSpec:
    atom_fields: list[FeatureField] = [
        FeatureField(
            "node_type",
            len(atom_type_names),
            FEATURE_DTYPE,
            "one-hot discrete node type (raw TU node label vocabulary + reserved UNK, or degree bucket)",
            "TU node label / degree bucket",
        )
    ]
    atom_fields.extend(_topology_atom_fields())
    if continuous_atom_dim:
        atom_fields.append(
            FeatureField(
                "continuous_node_attributes",
                int(continuous_atom_dim),
                FEATURE_DTYPE,
                "raw continuous TU node attributes retained verbatim; standardised on the fold Train split only",
                "TU node_attributes.txt",
            )
        )
    bond_fields: list[FeatureField] = [
        FeatureField("constant_one", 1, FEATURE_DTYPE, "always 1", "adapter"),
        FeatureField("is_cycle_edge", 1, FEATURE_DTYPE, "edge lies on at least one cycle", "biconnected-block analysis"),
    ]
    if edge_attribute_dim:
        bond_fields.append(
            FeatureField(
                "continuous_edge_attributes",
                int(edge_attribute_dim),
                FEATURE_DTYPE,
                "raw continuous TU edge attributes retained verbatim; standardised on the fold Train split only",
                "TU edge_attributes.txt",
            )
        )
    nbt = len(bond_type_names)
    spec = FeatureSpec(
        schema_version=SCHEMA_VERSION,
        name=f"tu_bench_{dataset.lower().replace('-', '_')}",
        description=(
            f"Fast positioning TU schema for {dataset}: discrete node labels/degree buckets, generic "
            "cycle/degree descriptors, retained continuous node attributes, and exact S1/S2 topology. "
            "No target label or fitted statistic is part of the schema."
        ),
        atom_type_names=tuple(str(x) for x in atom_type_names),
        atom_type_metadata=tuple(
            {"index": i, "name": str(name), "is_unknown": i == atom_unk_index, "source": "TU raw node label"}
            for i, name in enumerate(atom_type_names)
        ),
        atom_unk_index=atom_unk_index,
        atom_unknown_policy="map_to_UNK: raw node label absent from the fixed dataset vocabulary maps to the reserved UNK index",
        atom_type_encoding="atom_type = rank of the raw node label in the sorted dataset-wide vocabulary; degree datasets use min(degree,16); UNK reserved",
        bond_type_names=tuple(str(x) for x in bond_type_names),
        bond_type_metadata=tuple(
            {"id": i + 1, "name": str(name), "is_unknown": (i + 1) == bond_unk_id, "source": "TU raw edge label"}
            for i, name in enumerate(bond_type_names)
        ),
        bond_unk_id=bond_unk_id,
        bond_unknown_policy="map_to_UNK: observed raw edge labels map to 1..B-1; the reserved UNK id is B",
        bond_type_encoding="bond_type = 1 + rank of the raw edge label in the sorted dataset-wide vocabulary; one-based",
        atom_feature_fields=tuple(atom_fields),
        bond_feature_fields=tuple(bond_fields),
        atom_scalar_fields=_scalar_fields(),
        global_feature_fields=_global_fields(),
        nbr_code_formula=f"node_type[neighbour] * {nbt} + (edge_type - 1)",
        path_code_formula=f"node_type[middle] * {nbt * nbt} + (bt1 - 1) * {nbt} + (bt2 - 1)",
        path_end_code_formula=f"node_type[end] * {nbt} + (bt2 - 1)",
        structural_code_dtype=STRUCTURAL_CODE_DTYPE,
        structural_index_dtype=STRUCTURAL_INDEX_DTYPE,
        feature_dtype=FEATURE_DTYPE,
        supports_disconnected=True,
        disconnected_policy=(
            "supported with finite statistics: unreachable pairs (shortest-path sentinel -1) are excluded from "
            "diameter/mean distance, reported via frac_unreachable_pairs, and never counted as finite distance >= 3"
        ),
        atom_feature_source="TU raw node labels/attributes + cleaned graph topology",
        bond_feature_source="TU raw edge labels + cleaned graph topology",
        global_feature_source="label-free graph statistics with explicit disconnected handling",
        source_provenance=dict(source_provenance or {}),
        output_tensors=_output_tensors(
            len(atom_type_names) + TOPOLOGY_ATOM_DIM + int(continuous_atom_dim),
            2 + int(edge_attribute_dim),
            GLOBAL_DIM,
        ),
    )
    spec.validate()
    return spec


def peptide_spec(source_provenance: Mapping[str, Any] | None = None) -> FeatureSpec:
    atom_fields: list[FeatureField] = [
        FeatureField(
            PEPTIDE_ATOM_NAMES[i],
            int(PEPTIDE_ATOM_VOCABS[i]),
            FEATURE_DTYPE,
            f"one-hot legal category set for Peptides-func atom feature column {i}",
            "LRGB Peptides-func x",
        )
        for i in range(len(PEPTIDE_ATOM_VOCABS))
    ]
    atom_fields.extend(_topology_atom_fields())
    bond_fields: list[FeatureField] = [
        FeatureField(
            PEPTIDE_BOND_NAMES[i],
            int(PEPTIDE_BOND_VOCABS[i]),
            FEATURE_DTYPE,
            f"one-hot legal category set for Peptides-func bond feature column {i}",
            "LRGB Peptides-func edge_attr",
        )
        for i in range(len(PEPTIDE_BOND_VOCABS))
    ]
    bond_fields.append(FeatureField("is_cycle_edge", 1, FEATURE_DTYPE, "edge lies on at least one cycle", "graph"))
    n_at = int(PEPTIDE_ATOM_VOCABS[0])
    nbt = int(PEPTIDE_BOND_VOCABS[0])
    atom_dim = int(sum(PEPTIDE_ATOM_VOCABS)) + TOPOLOGY_ATOM_DIM
    bond_dim = int(sum(PEPTIDE_BOND_VOCABS)) + 1
    spec = FeatureSpec(
        schema_version=SCHEMA_VERSION,
        name="peptides_func_bench",
        description=(
            "Peptides-func LCRC input: all 9 OGB-style discrete atom columns and 3 discrete bond "
            "columns one-hot encoded with their legal category counts, plus the historical LCRC "
            "topology descriptors.  S1/S2 type codes use atom column 0 and bond column 0."
        ),
        atom_type_names=tuple(f"ATOMIC_NUMBER_{i}" for i in range(n_at)),
        atom_type_metadata=tuple(
            {"index": i, "name": f"ATOMIC_NUMBER_{i}", "is_unknown": False, "source": "Peptides-func atom column 0"}
            for i in range(n_at)
        ),
        atom_unk_index=None,
        atom_unknown_policy="reject: every atom column 0 value must be a legal category in [0,119)",
        atom_type_encoding="atom_type = atom feature column 0 (atomic number category), 0-based",
        bond_type_names=tuple(f"BOND_TYPE_{i}" for i in range(nbt)),
        bond_type_metadata=tuple(
            {"id": i + 1, "name": f"BOND_TYPE_{i}", "is_unknown": False, "source": "Peptides-func bond column 0"}
            for i in range(nbt)
        ),
        bond_unk_id=None,
        bond_unknown_policy="reject: every bond column 0 value must be a legal category in [0,5)",
        bond_type_encoding="bond_type = bond feature column 0 + 1, one-based",
        atom_feature_fields=tuple(atom_fields),
        bond_feature_fields=tuple(bond_fields),
        atom_scalar_fields=_scalar_fields(),
        global_feature_fields=_global_fields(),
        nbr_code_formula=f"node_type[neighbour] * {nbt} + (edge_type - 1)",
        path_code_formula=f"node_type[middle] * {nbt * nbt} + (bt1 - 1) * {nbt} + (bt2 - 1)",
        path_end_code_formula=f"node_type[end] * {nbt} + (bt2 - 1)",
        structural_code_dtype=STRUCTURAL_CODE_DTYPE,
        structural_index_dtype=STRUCTURAL_INDEX_DTYPE,
        feature_dtype=FEATURE_DTYPE,
        supports_disconnected=True,
        disconnected_policy=(
            "supported with finite statistics: unreachable pairs (shortest-path sentinel -1) are excluded from "
            "diameter/mean distance and reported via frac_unreachable_pairs"
        ),
        atom_feature_source="Peptides-func 9-column discrete atom features, one-hot by legal category count",
        bond_feature_source="Peptides-func 3-column discrete bond features, one-hot by legal category count",
        global_feature_source="label-free graph statistics with explicit disconnected handling",
        source_provenance=dict(source_provenance or {}),
        output_tensors=_output_tensors(atom_dim, bond_dim, GLOBAL_DIM),
    )
    spec.validate()
    return spec


def peptide_features(
    atom_raw: np.ndarray,
    bond_raw: np.ndarray,
    atom_type: np.ndarray,
    bond_type: np.ndarray,
    analysis: BG.FastAnalysis,
) -> tuple[np.ndarray, np.ndarray]:
    atom_raw = np.asarray(atom_raw, dtype=np.int64)
    bond_raw = np.asarray(bond_raw, dtype=np.int64)
    n = atom_raw.shape[0]
    m = bond_raw.shape[0]
    if atom_raw.shape != (n, len(PEPTIDE_ATOM_VOCABS)):
        raise ValueError("peptide atom feature width")
    if bond_raw.shape != (m, len(PEPTIDE_BOND_VOCABS)):
        raise ValueError("peptide bond feature width")
    atom_parts = []
    for i, k in enumerate(PEPTIDE_ATOM_VOCABS):
        col = atom_raw[:, i]
        if np.any(col < 0) or np.any(col >= k):
            raise ValueError(f"peptide atom column {i} outside legal range [0,{k})")
        atom_parts.append(np.eye(k, dtype=np.float32)[col])
    atom_parts.append(topology_atom_features(analysis))
    bond_parts = []
    for i, k in enumerate(PEPTIDE_BOND_VOCABS):
        col = bond_raw[:, i]
        if np.any(col < 0) or np.any(col >= k):
            raise ValueError(f"peptide bond column {i} outside legal range [0,{k})")
        bond_parts.append(np.eye(k, dtype=np.float32)[col])
    bond_parts.append(bond_topology_features(analysis))
    return np.concatenate(atom_parts, axis=1).astype(np.float32), np.concatenate(bond_parts, axis=1).astype(np.float32)
