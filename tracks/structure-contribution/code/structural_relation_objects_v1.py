"""BJOC structural-relation objects (cscl-structural-relation-composer-v1).

Protocol: ``configs/structural_relation_composer_v1.yaml``.

This module turns a raw molecular graph (PyG ZINC atoms/bonds only: ``x`` =
atom type index, ``edge_index`` + ``edge_attr`` = PyG doubled directed bonds)
into a **one-shot** object hierarchy:

* **atom objects**        ``a_i``  — raw atom label + label-blind local
  chemistry/topology statistics computed by fixed graph algorithms
  (degree, ring membership/min ring size, bond-type/neighbour-type
  histograms, radius-2 atom-type histogram, local cycles/triangles).
* **physical bond objects** ``(u, v, bond_type)`` — every *undirected* bond
  exactly once (PyG stores each bond twice; the direction pair is validated
  and canonicalised by :func:`cscl_units.unique_undirected_bonds`).
* **junction objects**    ``(i, j, k)`` — a shared centre atom ``j`` and two
  distinct neighbours ``i < k``; every junction exactly once, endpoints
  unordered, centre explicit.
* **incidence**           ``junction -> (bond_ij, bond_jk)`` — which two real
  bonds each junction touches.  This is the object that carries *structural
  relation ownership* into the model.

Discipline
----------
* Label-blind: only ``x`` / ``edge_index`` / ``edge_attr`` are read.  ``y``
  never enters this module.
* Deterministic: no RNG, no learned component, no data-set statistics.
* No message passing: every feature is a fixed function of a rooted
  neighbourhood; nothing is updated iteratively.
* Atom/bond/junction indices are emitted in canonical order so that a pure
  atom relabelling permutes rows/indices and nothing else.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_units as cu  # noqa: E402  (unique_undirected_bonds + biconnected helpers)

# ---------------------------------------------------------------------------
# frozen feature layout
# ---------------------------------------------------------------------------

ATOM_CATEGORIES = 28  # raw ZINC subset x index vocabulary (max observed 20)
BOND_CATEGORIES = 4   # single / double / triple / aromatic

#: atom features
ATOM_DEGREE_SLOTS = 5            # degree 0..4 (max observed 4)
RING_SIZE_SLOTS = 6              # min ring size 3,4,5,6,7,8+
ATOM_FEAT_DIM = (
    ATOM_CATEGORIES              # own atom type
    + ATOM_DEGREE_SLOTS          # own degree
    + 1                          # in any ring
    + RING_SIZE_SLOTS            # smallest ring through atom (zero if acyclic)
    + 2                          # n ring bonds, ring-bond fraction
    + BOND_CATEGORIES            # incident bond-type histogram
    + ATOM_CATEGORIES            # radius-1 neighbour-type histogram
    + ATOM_CATEGORIES            # radius-2 atom-type histogram
    + 4                          # log1p(n_atoms d=2), log1p(2-paths), triangles, local cycles
    + 2                          # is_terminal, ring-system membership flag
)

#: raw bond features (endpoint embeddings are concatenated in the model)
BOND_FEAT_DIM = BOND_CATEGORIES + 3  # type one-hot, is_ring, exactly_one_in_ring, both_in_ring

#: raw junction features (atom/bond embeddings are concatenated in the model)
JUNC_FEAT_DIM = (
    1    # centre degree
    + 1  # endpoints i,k are bonded to each other
    + 1  # log1p(common neighbours of i,k)
    + 1  # number of the two bonds that are ring bonds
    + 1  # both bonds are ring bonds
    + 1  # centre atom is in a ring
    + RING_SIZE_SLOTS  # centre min ring size (zero if acyclic)
    + BOND_CATEGORIES  # symmetric sum of the two bond-type one-hots
    + BOND_CATEGORIES * BOND_CATEGORIES  # symmetric outer sum
)

#: label-blind global topology vector (fixed layout; standardised fit-only)
GLOBAL_FEAT_NAMES = (
    "log1p_n_atoms",
    "log1p_n_bonds",
    "log1p_n_junctions",
    "cyclomatic_rank",
    "ring_atom_fraction",
    "ring_bond_fraction",
    "n_ring_systems",
    "diameter",
    "mean_distance",
    "frac_dist_ge3",
    "bonds_per_atom",
    "log1p_n_ring_atoms",
    "max_degree",
    "terminal_atom_fraction",
    "junction_per_bond",
    "frac_junction_centers_ring",
    "frac_deg2_junctions",
    "frac_deg3_junctions",
    "frac_deg4_junctions",
    "bond_type0_frac",
    "bond_type1_frac",
    "bond_type2_frac",
    "bond_type3_frac",
    "ring_size3_frac",
    "ring_size4_frac",
    "ring_size5_frac",
    "ring_size6_frac",
    "ring_size7_frac",
    "ring_size8plus_frac",
    "fused_ring_atom_fraction",
    "mean_pair_distance",
    "distinct_bond_types",
    "distinct_atom_types",
)
GLOBAL_DIM = len(GLOBAL_FEAT_NAMES)

#: atom-feature block layout (name -> slice), used by masking diagnostics
def atom_blocks() -> dict[str, slice]:
    off = 0
    out: dict[str, slice] = {}
    out["atom_type"] = slice(off, off + ATOM_CATEGORIES); off += ATOM_CATEGORIES
    out["degree"] = slice(off, off + ATOM_DEGREE_SLOTS); off += ATOM_DEGREE_SLOTS
    out["in_ring"] = slice(off, off + 1); off += 1
    out["min_ring_size"] = slice(off, off + RING_SIZE_SLOTS); off += RING_SIZE_SLOTS
    out["ring_bonds"] = slice(off, off + 2); off += 2
    out["bond_type_counts"] = slice(off, off + BOND_CATEGORIES); off += BOND_CATEGORIES
    out["neighbour_types"] = slice(off, off + ATOM_CATEGORIES); off += ATOM_CATEGORIES
    out["radius2_types"] = slice(off, off + ATOM_CATEGORIES); off += ATOM_CATEGORIES
    out["radius2_stats"] = slice(off, off + 4); off += 4
    out["roles"] = slice(off, off + 2); off += 2
    assert off == ATOM_FEAT_DIM, (off, ATOM_FEAT_DIM)
    return out


# ---------------------------------------------------------------------------
# per-molecule object container
# ---------------------------------------------------------------------------


@dataclass
class MolObjects:
    """One-shot atom / bond / junction object tensors for one molecule."""

    mol_index: int
    smiles: str
    n_atoms: int

    atom_feat: np.ndarray        # [n_atoms, ATOM_FEAT_DIM] float32
    bond_index: np.ndarray       # [n_bonds, 2] int64, u < v
    bond_type: np.ndarray        # [n_bonds] int64
    bond_feat: np.ndarray        # [n_bonds, BOND_FEAT_DIM] float32
    junc_index: np.ndarray       # [n_junc, 3] int64, (i, j, k) with i < k
    junc_bond: np.ndarray        # [n_junc, 2] int64, global bond ids
    junc_feat: np.ndarray        # [n_junc, JUNC_FEAT_DIM] float32
    global_feat: np.ndarray      # [GLOBAL_DIM] float32


# ---------------------------------------------------------------------------
# fixed graph statistics
# ---------------------------------------------------------------------------


def _adjacency(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> list[list[tuple[int, int]]]:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for u, v, bt in bonds:
        adj[u].append((v, int(bt)))
        adj[v].append((u, int(bt)))
    for u in range(n_atoms):
        adj[u].sort()
    return adj


def _biconnected_blocks(n_atoms: int, bonds: Sequence[tuple[int, int, int]]):
    """Biconnected components (node set, edge list); fixed Tarjan (networkx)."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    graph.add_edges_from((u, v) for u, v, _ in bonds)
    blocks = []
    for nodes in nx.biconnected_components(graph):
        nodes = set(nodes)
        edges = [(u, v, bt) for u, v, bt in bonds if u in nodes and v in nodes]
        blocks.append((nodes, edges))
    return blocks


def _all_pairs_distances(n_atoms: int, adj: list[list[tuple[int, int]]]) -> np.ndarray:
    """Unweighted shortest-path matrix (BFS from every node; fixed algorithm)."""
    inf = np.int64(1 << 20)
    dist = np.full((n_atoms, n_atoms), inf, dtype=np.int64)
    for s in range(n_atoms):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt: list[int] = []
            for u in frontier:
                for v, _ in adj[u]:
                    if dist[s, v] == inf:
                        dist[s, v] = d
                        nxt.append(v)
            frontier = nxt
    return dist


def _min_ring_size_per_atom(n_atoms: int, adj: list[list[tuple[int, int]]]) -> np.ndarray:
    """Smallest cycle through each atom (0 if the atom is acyclic).

    A cycle through ``a`` is a shortest path between two distinct neighbours
    of ``a`` that avoids ``a``; ring size = path length + 2.
    """
    result = np.zeros(n_atoms, dtype=np.int64)
    for a in range(n_atoms):
        nbrs = [v for v, _ in adj[a]]
        if len(nbrs) < 2:
            continue
        best = 0
        blocked = np.zeros(n_atoms, dtype=bool)
        blocked[a] = True
        for x in range(len(nbrs)):
            for y in range(x + 1, len(nbrs)):
                s, t = nbrs[x], nbrs[y]
                # BFS from s avoiding a
                inf = 1 << 20
                seen = np.full(n_atoms, inf, dtype=np.int64)
                seen[s] = 0
                frontier = [s]
                d = 0
                found = False
                while frontier and not found:
                    d += 1
                    nxt = []
                    for u in frontier:
                        for v, _ in adj[u]:
                            if blocked[v] or seen[v] != inf:
                                continue
                            seen[v] = d
                            if v == t:
                                found = True
                                break
                            nxt.append(v)
                        if found:
                            break
                    frontier = nxt
                if found:
                    size = int(seen[t]) + 2
                    if best == 0 or size < best:
                        best = size
        result[a] = best
    return result


def _ring_bond_flags(n_atoms: int, bonds: Sequence[tuple[int, int, int]]):
    """is_ring_bond per bond; ring-atom mask.

    A bond lies on a cycle iff its biconnected block contains >= 3 edges.
    An atom lies on a cycle iff it is in a block with >= 3 edges and has
    degree >= 2 inside that block.
    """
    blocks = _biconnected_blocks(n_atoms, bonds)
    bond_key = {(min(u, v), max(u, v)): k for k, (u, v, _) in enumerate(bonds)}
    is_ring_bond = np.zeros(len(bonds), dtype=bool)
    atom_deg_cycle = np.zeros(n_atoms, dtype=np.int64)
    ring_systems = 0
    for nodes, edges in blocks:
        if len(edges) >= 3:
            ring_systems += 1
            for u, v, _ in edges:
                is_ring_bond[bond_key[(min(u, v), max(u, v))]] = True
                atom_deg_cycle[u] += 1
                atom_deg_cycle[v] += 1
    ring_atom = atom_deg_cycle >= 2
    return is_ring_bond, ring_atom, ring_systems


def _ring_size_bucket(size: int) -> int:
    """3,4,5,6,7,8+ -> 0..5 (0 elsewhere)."""
    if size <= 0:
        return -1
    if size >= 8:
        return 5
    if size < 3:
        return -1
    return size - 3


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------


def build_mol_objects(
    mol_index: int,
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    smiles: str = "",
) -> MolObjects:
    """Raw PyG molecule -> one-shot BJOC object tensors (label-blind)."""
    atom_types = [int(v) for v in np.asarray(x).reshape(-1)]
    ei = np.asarray(edge_index)
    directed = [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])]
    directed_types = [int(t) for t in np.asarray(edge_attr).reshape(-1)]
    bonds = cu.unique_undirected_bonds([(a, b) for a, b in directed], directed_types)
    # cu returns (u, v, bt) sorted by (u, v) with u < v
    n_atoms = len(atom_types)
    n_bonds = len(bonds)
    adj = _adjacency(n_atoms, bonds)

    is_ring_bond, ring_atom, n_ring_systems = _ring_bond_flags(n_atoms, bonds)
    min_ring = _min_ring_size_per_atom(n_atoms, adj)
    dist = _all_pairs_distances(n_atoms, adj)

    # ---- atom features -----------------------------------------------------
    atom_feat = np.zeros((n_atoms, ATOM_FEAT_DIM), dtype=np.float32)
    bond_ij: dict[tuple[int, int], int] = {}
    for k, (u, v, _) in enumerate(bonds):
        bond_ij[(u, v)] = k
    degree = np.array([len(adj[i]) for i in range(n_atoms)], dtype=np.int64)
    ring_bond_count = np.zeros(n_atoms, dtype=np.int64)
    for k, (u, v, _) in enumerate(bonds):
        if is_ring_bond[k]:
            ring_bond_count[u] += 1
            ring_bond_count[v] += 1

    off = 0
    for i in range(n_atoms):
        at = atom_types[i]
        if not (0 <= at < ATOM_CATEGORIES):
            raise ValueError(f"atom type {at} outside frozen vocabulary {ATOM_CATEGORIES}")
        atom_feat[i, off + at] = 1.0
    off += ATOM_CATEGORIES
    for i in range(n_atoms):
        d = int(degree[i])
        if d >= ATOM_DEGREE_SLOTS:
            d = ATOM_DEGREE_SLOTS - 1
        atom_feat[i, off + d] = 1.0
    off += ATOM_DEGREE_SLOTS
    atom_feat[:, off] = ring_atom.astype(np.float32)
    off += 1
    for i in range(n_atoms):
        slot = _ring_size_bucket(int(min_ring[i]))
        if slot >= 0:
            atom_feat[i, off + slot] = 1.0
    off += RING_SIZE_SLOTS
    atom_feat[:, off] = ring_bond_count.astype(np.float32)
    atom_feat[:, off + 1] = np.where(degree > 0, ring_bond_count / np.maximum(degree, 1), 0.0)
    off += 2
    for k, (u, v, bt) in enumerate(bonds):
        atom_feat[u, off + bt] += 1.0
        atom_feat[v, off + bt] += 1.0
    off += BOND_CATEGORIES
    for i in range(n_atoms):
        for v, _ in adj[i]:
            atom_feat[i, off + atom_types[v]] += 1.0
    off += ATOM_CATEGORIES
    n_d2 = np.zeros(n_atoms, dtype=np.float32)
    paths2 = np.zeros(n_atoms, dtype=np.float32)
    triangles = np.zeros(n_atoms, dtype=np.float32)
    for i in range(n_atoms):
        for v in range(n_atoms):
            if v != i and dist[i, v] == 2:
                atom_feat[i, off + atom_types[v]] += 1.0
                n_d2[i] += 1.0
        for v, _ in adj[i]:
            paths2[i] += max(int(degree[v]) - 1, 0)
        for v in sorted(nbr for nbr, _ in adj[i]):
            for w, _ in adj[v]:
                if w > v and w != i and dist[i, w] == 1:
                    triangles[i] += 1.0
    off2 = off + ATOM_CATEGORIES
    atom_feat[:, off2] = np.log1p(n_d2)
    atom_feat[:, off2 + 1] = np.log1p(paths2)
    atom_feat[:, off2 + 2] = triangles
    atom_feat[:, off2 + 3] = (ring_bond_count >= 3).astype(np.float32)  # fused/spiro centre
    off2 += 4
    atom_feat[:, off2] = (degree == 1).astype(np.float32)
    atom_feat[:, off2 + 1] = (((ring_bond_count >= 3) | (min_ring >= 3) & (min_ring <= 4))).astype(np.float32)
    assert off2 + 2 == ATOM_FEAT_DIM, (off2 + 2, ATOM_FEAT_DIM)

    # ---- bond features -----------------------------------------------------
    bond_feat = np.zeros((n_bonds, BOND_FEAT_DIM), dtype=np.float32)
    for k, (u, v, bt) in enumerate(bonds):
        bond_feat[k, bt] = 1.0
        bond_feat[k, BOND_CATEGORIES] = float(is_ring_bond[k])
        ru, rv = bool(ring_atom[u]), bool(ring_atom[v])
        bond_feat[k, BOND_CATEGORIES + 1] = float(ru != rv)
        bond_feat[k, BOND_CATEGORIES + 2] = float(ru and rv)

    # ---- junctions ---------------------------------------------------------
    neighbours: list[list[int]] = [[v for v, _ in adj[i]] for i in range(n_atoms)]
    junc_i: list[tuple[int, int, int]] = []
    junc_bonds: list[tuple[int, int]] = []
    for j in range(n_atoms):
        nbrs = sorted(neighbours[j])
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                junc_i.append((i, j, k))
                junc_bonds.append((bond_ij[(min(i, j), max(i, j))], bond_ij[(min(j, k), max(j, k))]))
    n_junc = len(junc_i)
    junc_index = np.asarray(junc_i, dtype=np.int64).reshape(n_junc, 3)
    junc_bond = np.asarray(junc_bonds, dtype=np.int64).reshape(n_junc, 2)
    junc_feat = np.zeros((n_junc, JUNC_FEAT_DIM), dtype=np.float32)
    for idx, (i, j, k) in enumerate(junc_i):
        b1, b2 = junc_bonds[idx]
        t1, t2 = bonds[b1][2], bonds[b2][2]
        off = 0
        junc_feat[idx, off] = degree[j]
        off += 1
        junc_feat[idx, off] = float((min(i, k), max(i, k)) in bond_ij)
        off += 1
        common = set(neighbours[i]) & set(neighbours[k])
        common.discard(j)
        junc_feat[idx, off] = np.log1p(len(common))
        off += 1
        r = float(is_ring_bond[b1]) + float(is_ring_bond[b2])
        junc_feat[idx, off] = r
        junc_feat[idx, off + 1] = float(is_ring_bond[b1] and is_ring_bond[b2])
        junc_feat[idx, off + 2] = float(ring_atom[j])
        off += 3
        slot = _ring_size_bucket(int(min_ring[j]))
        if slot >= 0:
            junc_feat[idx, off + slot] = 1.0
        off += RING_SIZE_SLOTS
        junc_feat[idx, off + t1] += 1.0
        junc_feat[idx, off + t2] += 1.0
        off += BOND_CATEGORIES
        junc_feat[idx, off + t1 * BOND_CATEGORIES + t2] += 1.0
        junc_feat[idx, off + t2 * BOND_CATEGORIES + t1] += 1.0
        off += BOND_CATEGORIES * BOND_CATEGORIES
        assert off == JUNC_FEAT_DIM, (off, JUNC_FEAT_DIM)

    global_feat = _global_features(
        n_atoms, bonds, is_ring_bond, ring_atom, min_ring, junc_index, n_ring_systems, dist, atom_types
    )
    return MolObjects(
        mol_index=int(mol_index),
        smiles=smiles,
        n_atoms=n_atoms,
        atom_feat=atom_feat,
        bond_index=np.asarray([(u, v) for u, v, _ in bonds], dtype=np.int64).reshape(n_bonds, 2),
        bond_type=np.asarray([bt for _, _, bt in bonds], dtype=np.int64),
        bond_feat=bond_feat,
        junc_index=junc_index,
        junc_bond=junc_bond,
        junc_feat=junc_feat,
        global_feat=global_feat,
    )


def _global_features(
    n_atoms: int,
    bonds,
    is_ring_bond: np.ndarray,
    ring_atom: np.ndarray,
    min_ring: np.ndarray,
    junc_index: np.ndarray,
    n_ring_systems: int,
    dist: np.ndarray,
    atom_types: Sequence[int],
) -> np.ndarray:
    n_bonds = len(bonds)
    n_junc = int(junc_index.shape[0])
    finite = dist[dist < (1 << 20)]
    diameter = float(dist[dist < (1 << 20)].max()) if n_atoms > 1 else 0.0
    # exclude diagonal from mean pair distance
    off_diag = dist[~np.eye(n_atoms, dtype=bool)]
    off_diag = off_diag[off_diag < (1 << 20)]
    mean_distance = float(off_diag.mean()) if off_diag.size else 0.0
    frac_ge3 = float((dist >= 3).sum() / max(n_atoms * n_atoms - n_atoms, 1))
    degree = np.zeros(n_atoms, dtype=np.int64)
    ring_bond_count = np.zeros(n_atoms, dtype=np.int64)
    for k, (u, v, _) in enumerate(bonds):
        degree[u] += 1
        degree[v] += 1
        if is_ring_bond[k]:
            ring_bond_count[u] += 1
            ring_bond_count[v] += 1
    ring_size_counts = np.zeros(RING_SIZE_SLOTS, dtype=np.float64)
    for i in range(n_atoms):
        slot = _ring_size_bucket(int(min_ring[i]))
        if slot >= 0 and ring_atom[i]:
            ring_size_counts[slot] += 1.0
    if ring_atom.any():
        ring_size_counts /= float(ring_atom.sum())
    bond_type_counts = np.zeros(BOND_CATEGORIES, dtype=np.float64)
    for _, _, bt in bonds:
        bond_type_counts[bt] += 1.0
    if n_bonds:
        bond_type_counts /= float(n_bonds)
    center_deg = degree[junc_index[:, 1]] if n_junc else np.zeros(0, dtype=np.int64)
    fused = float((ring_bond_count >= 3).mean()) if n_atoms else 0.0
    values = [
        np.log1p(n_atoms),
        np.log1p(n_bonds),
        np.log1p(n_junc),
        float(n_bonds - n_atoms + 1),
        float(ring_atom.mean()) if n_atoms else 0.0,
        float(is_ring_bond.mean()) if n_bonds else 0.0,
        float(n_ring_systems),
        diameter,
        mean_distance,
        frac_ge3,
        float(n_bonds) / max(n_atoms, 1),
        np.log1p(float(ring_atom.sum())),
        float(degree.max()) if n_atoms else 0.0,
        float((degree == 1).mean()) if n_atoms else 0.0,
        float(n_junc) / max(n_bonds, 1),
        float(((degree[junc_index[:, 1]] > 0) & ring_atom[junc_index[:, 1]]).mean()) if n_junc else 0.0,
        float((center_deg == 2).mean()) if n_junc else 0.0,
        float((center_deg == 3).mean()) if n_junc else 0.0,
        float((center_deg >= 4).mean()) if n_junc else 0.0,
        *bond_type_counts.tolist(),
        *ring_size_counts.tolist(),
        fused,
        mean_distance,
        float(len(set(int(b) for _, _, b in bonds))),
        float(len(set(int(t) for t in atom_types))),
    ]
    out = np.asarray(values, dtype=np.float32)
    assert out.shape[0] == GLOBAL_DIM, (out.shape[0], GLOBAL_DIM)
    return out


# ---------------------------------------------------------------------------
# batched tensor assembly (one-shot gathers; no message passing)
# ---------------------------------------------------------------------------


def batch_objects(objects: Sequence[MolObjects]):
    """Concatenate per-molecule object tensors into a flat batch.

    Returns a dict with a molecule index for every atom/bond/junction, so all
    downstream operations are one-shot gathers / scatter-adds.
    """
    import torch

    atom_rows, atom_batch = [], []
    bond_rows, bond_batch, bond_index_all = [], [], []
    junc_rows, junc_batch, junc_index_all, junc_bond_all = [], [], [], []
    global_rows = []
    a_off = b_off = 0
    for g, obj in enumerate(objects):
        n_a, n_b, n_j = obj.n_atoms, obj.bond_index.shape[0], obj.junc_index.shape[0]
        atom_rows.append(torch.as_tensor(obj.atom_feat, dtype=torch.float32))
        atom_batch.append(torch.full((n_a,), g, dtype=torch.long))
        bond_rows.append(torch.as_tensor(obj.bond_feat, dtype=torch.float32))
        bond_batch.append(torch.full((n_b,), g, dtype=torch.long))
        bond_index_all.append(torch.as_tensor(obj.bond_index, dtype=torch.long) + a_off)
        junc_rows.append(torch.as_tensor(obj.junc_feat, dtype=torch.float32))
        junc_batch.append(torch.full((n_j,), g, dtype=torch.long))
        junc_index_all.append(torch.as_tensor(obj.junc_index, dtype=torch.long) + a_off)
        junc_bond_all.append(torch.as_tensor(obj.junc_bond, dtype=torch.long) + b_off)
        global_rows.append(torch.as_tensor(obj.global_feat, dtype=torch.float32))
        a_off += n_a
        b_off += n_b
    out = {
        "n_graphs": len(objects),
        "atom_feat": torch.cat(atom_rows, 0) if atom_rows else torch.zeros(0, ATOM_FEAT_DIM),
        "atom_batch": torch.cat(atom_batch, 0) if atom_batch else torch.zeros(0, dtype=torch.long),
        "bond_feat": torch.cat(bond_rows, 0) if bond_rows else torch.zeros(0, BOND_FEAT_DIM),
        "bond_batch": torch.cat(bond_batch, 0) if bond_batch else torch.zeros(0, dtype=torch.long),
        "bond_index": torch.cat(bond_index_all, 0) if bond_index_all else torch.zeros(0, 2, dtype=torch.long),
        "junc_feat": torch.cat(junc_rows, 0) if junc_rows else torch.zeros(0, JUNC_FEAT_DIM),
        "junc_batch": torch.cat(junc_batch, 0) if junc_batch else torch.zeros(0, dtype=torch.long),
        "junc_index": torch.cat(junc_index_all, 0) if junc_index_all else torch.zeros(0, 3, dtype=torch.long),
        "junc_bond": torch.cat(junc_bond_all, 0) if junc_bond_all else torch.zeros(0, 2, dtype=torch.long),
        "global_feat": torch.stack(global_rows, 0) if global_rows else torch.zeros(0, GLOBAL_DIM),
    }
    return out
