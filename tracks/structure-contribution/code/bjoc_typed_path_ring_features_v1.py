"""Typed short-path + ring-block structural descriptors (BJOC representation upgrade).

Protocol: ``configs/bjoc_representation_optimization_v1.yaml``.

This module adds **label-blind, fixed-graph-algorithm** description of two
structural aspects that the original BJOC atom/bond objects do not express:

* **typed bond paths (length 2 and 3)** rooted at every atom: the ordered
  bond-type sequence ``(t1, t2[, t3])`` traversed from the root outwards,
  how many of the traversed bonds are ring bonds, the count of paths, and
  (length 3 only) the far-endpoint atom-type histogram.  Enumeration is over
  *simple* paths of the raw molecular graph and is a pure function of the
  rooted neighbourhood: no atom index, no SMILES fragment id, no label.
* **ring-block descriptors**: the Tarjan biconnected decomposition of the
  molecular graph, with per-block size / bond count / local cycle rank /
  bond-type composition / number of external attachments / fused-edge and
  spiro roles, aggregated onto every atom and every physical bond.

Nothing here is learned, iterated or refreshed from a previous forward pass;
this module is a *feature extractor*, so the BJOC dependence DAG is unchanged
(atom features -> atom embedding -> bond embedding -> ...).

Discipline
----------
* reads only ``x`` / ``edge_index`` / ``edge_attr`` (plus the already-derived
  physical bond list); ``y`` never enters.
* deterministic, no RNG.
* permutation invariant: atom relabelling permutes the atom rows; bond rows
  follow the canonical ``(min, max)`` bond order of the base object builder.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import structural_relation_objects_v1 as sro  # noqa: E402

# ---------------------------------------------------------------------------
# frozen typed-path layout
# ---------------------------------------------------------------------------

PATH2_SEQ = 16   # (t1, t2) in 4 x 4 bond types
PATH3_SEQ = 64   # (t1, t2, t3) in 4 x 4 x 4 bond types
PATH2_RING = 3   # one-hot: number of ring bonds on the length-2 path (0,1,2)
PATH3_RING = 4   # one-hot: number of ring bonds on the length-3 path (0..3)
PATH3_ENDTYPE = sro.ATOM_CATEGORIES   # far-endpoint atom type histogram (length-3)
PATH_TOTALS = 2  # log1p(total length-2 paths), log1p(total length-3 paths)

PATH_ATOM_DIM = PATH2_SEQ + PATH3_SEQ + PATH2_RING + PATH3_RING + PATH3_ENDTYPE + PATH_TOTALS

# ---------------------------------------------------------------------------
# frozen ring-block layout
# ---------------------------------------------------------------------------

RING_ATOM_NBLK = 4        # one-hot: cyclic blocks containing the atom (0,1,2,3+)
RING_ATOM_DIM = (
    RING_ATOM_NBLK        # number of cyclic biconnected blocks containing the atom
    + 1                   # log1p(max local cycle rank over those blocks)
    + 1                   # log1p(max block n_atoms)
    + 1                   # log1p(min block n_atoms)
    + 1                   # log1p(sum local cycle rank)
    + 1                   # fraction of its blocks in which the atom has an external bond
    + 1                   # log1p(max number of external attachment atoms of its blocks)
    + 1                   # is a spiro centre (belongs to >= 2 cyclic blocks)
    + 1                   # is incident to >= 1 fused (>= 2 cycles) edge
    + 1                   # belongs to a monocyclic block (rank == 1)
    + 1                   # belongs to a polycyclic block (rank >= 2)
    + sro.BOND_CATEGORIES  # mean bond-type composition of its blocks
)

RING_BOND_DIM = (
    1                     # bridge / non-cyclic edge
    + 1                   # log1p(block n_atoms)
    + 1                   # log1p(block n_edges)
    + 1                   # log1p(block local cycle rank)
    + 1                   # log1p(number of block atoms with an external bond)
    + sro.BOND_CATEGORIES  # bond-type composition of the block
    + 1                   # block is monocyclic (rank == 1)
    + 1                   # block is polycyclic (rank >= 2)
    + 1                   # fused edge (lies on >= 2 distinct simple cycles)
)

NEW_ATOM_DIM = PATH_ATOM_DIM + RING_ATOM_DIM
NEW_BOND_DIM = RING_BOND_DIM

PATH_BLOCK_NAMES = (
    "path2_seq",
    "path3_seq",
    "path2_ring_count",
    "path3_ring_count",
    "path3_endpoint_type",
    "path_totals",
)
RING_ATOM_BLOCK_NAMES = (
    "n_cyclic_blocks",
    "max_block_cycle_rank",
    "max_block_n_atoms",
    "min_block_n_atoms",
    "sum_block_cycle_rank",
    "frac_blocks_cut_vertex",
    "max_block_attachments",
    "is_spiro_center",
    "is_fused_center",
    "in_monocyclic_block",
    "in_polycyclic_block",
    "block_bondtype_frac",
)
RING_BOND_BLOCK_NAMES = (
    "is_bridge",
    "block_n_atoms",
    "block_n_edges",
    "block_cycle_rank",
    "block_attachments",
    "block_bondtype_frac",
    "block_monocyclic",
    "block_polycyclic",
    "is_fused_edge",
)


def atom_layout() -> dict[str, slice]:
    """Name -> slice of the new atom channel (order is frozen)."""
    off = 0
    out: dict[str, slice] = {}
    out["path2_seq"] = slice(off, off + PATH2_SEQ); off += PATH2_SEQ
    out["path3_seq"] = slice(off, off + PATH3_SEQ); off += PATH3_SEQ
    out["path2_ring_count"] = slice(off, off + PATH2_RING); off += PATH2_RING
    out["path3_ring_count"] = slice(off, off + PATH3_RING); off += PATH3_RING
    out["path3_endpoint_type"] = slice(off, off + PATH3_ENDTYPE); off += PATH3_ENDTYPE
    out["path_totals"] = slice(off, off + PATH_TOTALS); off += PATH_TOTALS
    out["ring_n_cyclic_blocks"] = slice(off, off + RING_ATOM_NBLK); off += RING_ATOM_NBLK
    out["ring_max_block_cycle_rank"] = slice(off, off + 1); off += 1
    out["ring_max_block_n_atoms"] = slice(off, off + 1); off += 1
    out["ring_min_block_n_atoms"] = slice(off, off + 1); off += 1
    out["ring_sum_block_cycle_rank"] = slice(off, off + 1); off += 1
    out["ring_frac_blocks_cut_vertex"] = slice(off, off + 1); off += 1
    out["ring_max_block_attachments"] = slice(off, off + 1); off += 1
    out["ring_is_spiro_center"] = slice(off, off + 1); off += 1
    out["ring_is_fused_center"] = slice(off, off + 1); off += 1
    out["ring_in_monocyclic_block"] = slice(off, off + 1); off += 1
    out["ring_in_polycyclic_block"] = slice(off, off + 1); off += 1
    out["ring_block_bondtype_frac"] = slice(off, off + sro.BOND_CATEGORIES); off += sro.BOND_CATEGORIES
    assert off == NEW_ATOM_DIM, (off, NEW_ATOM_DIM)
    return out


def bond_layout() -> dict[str, slice]:
    off = 0
    out: dict[str, slice] = {}
    out["is_bridge"] = slice(off, off + 1); off += 1
    out["block_n_atoms"] = slice(off, off + 1); off += 1
    out["block_n_edges"] = slice(off, off + 1); off += 1
    out["block_cycle_rank"] = slice(off, off + 1); off += 1
    out["block_attachments"] = slice(off, off + 1); off += 1
    out["block_bondtype_frac"] = slice(off, off + sro.BOND_CATEGORIES); off += sro.BOND_CATEGORIES
    out["block_monocyclic"] = slice(off, off + 1); off += 1
    out["block_polycyclic"] = slice(off, off + 1); off += 1
    out["is_fused_edge"] = slice(off, off + 1); off += 1
    assert off == NEW_BOND_DIM, (off, NEW_BOND_DIM)
    return out


# ---------------------------------------------------------------------------
# object container
# ---------------------------------------------------------------------------


@dataclass
class ExtendedMolObjects:
    """Base BJOC objects plus the raw (unstandardised) new channels."""

    base: sro.MolObjects
    atom_new: np.ndarray   # [n_atoms, NEW_ATOM_DIM] float32, raw
    bond_new: np.ndarray   # [n_bonds, NEW_BOND_DIM] float32, raw

    @property
    def n_atoms(self) -> int:
        return self.base.n_atoms

    @property
    def mol_index(self) -> int:
        return self.base.mol_index

    @property
    def smiles(self) -> str:
        return self.base.smiles


# ---------------------------------------------------------------------------
# typed paths
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ring blocks
# ---------------------------------------------------------------------------


def _block_cycle_rank(n_nodes: int, n_edges: int) -> int:
    return int(n_edges - n_nodes + 1)


def _simple_path_count_capped(
    start: int, target: int, adj: dict[int, list[int]], blocked_edge: tuple[int, int], cap: int = 2
) -> int:
    """Count simple ``start -> target`` paths that avoid ``blocked_edge`` (capped)."""
    count = 0
    stack: list[tuple[int, set[int]]] = [(start, {start})]
    while stack:
        node, visited = stack.pop()
        if node == target:
            count += 1
            if count >= cap:
                return count
            continue
        for nxt in adj.get(node, ()):  # type: ignore[arg-type]
            if nxt in visited:
                continue
            e = (min(node, nxt), max(node, nxt))
            if e == blocked_edge:
                continue
            stack.append((nxt, visited | {nxt}))
    return count


def ring_block_blocks(
    n_atoms: int,
    bonds: Sequence[tuple[int, int, int]],
    *,
    fused_check_max_block_atoms: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Per-atom and per-bond ring-block descriptors.

    ``bonds`` is the canonical physical-bond list (``u < v`` sorted), matching
    :func:`structural_relation_objects_v1.build_mol_objects`.
    """
    bond_key = {(min(u, v), max(u, v)): k for k, (u, v, _) in enumerate(bonds)}
    blocks = sro._biconnected_blocks(n_atoms, bonds)
    atom_out = np.zeros((n_atoms, RING_ATOM_DIM), dtype=np.float32)
    bond_out = np.zeros((len(bonds), RING_BOND_DIM), dtype=np.float32)
    bond_acc: list[list[float]] = [[] for _ in range(len(bonds))]

    atom_cyclic_blocks: list[list[int]] = [[] for _ in range(n_atoms)]
    atom_fused_edges = np.zeros(n_atoms, dtype=np.int64)

    cyclic_blocks: list[dict[str, object]] = []
    adj_all = _adj_list(bonds, n_atoms)
    for nodes, edges in blocks:
        if len(edges) < 3:
            for u, v, _ in edges:
                k = bond_key[(min(u, v), max(u, v))]
                bond_out[k, 0] = 1.0  # bridge / non-cyclic edge
            continue
        nodes = set(nodes)
        rank = _block_cycle_rank(len(nodes), len(edges))
        block_id = len(cyclic_blocks)
        for u in nodes:
            atom_cyclic_blocks[u].append(block_id)
        # external attachment atoms: block atoms that have a bond leaving the block
        attachments = 0
        for u in nodes:
            if any(v not in nodes for v, _ in adj_all[u]):
                attachments += 1
        type_counts = np.zeros(sro.BOND_CATEGORIES, dtype=np.float64)
        for u, v, t in edges:
            type_counts[t] += 1.0
        type_frac = type_counts / max(len(edges), 1.0)
        fused_edges = _fused_edges_within_block(nodes, edges, adj_all, fused_check_max_block_atoms)
        cyclic_blocks.append(
            {
                "nodes": nodes,
                "n_atoms": len(nodes),
                "n_edges": len(edges),
                "rank": rank,
                "attachments": attachments,
                "type_frac": type_frac,
                "fused_edges": fused_edges,
            }
        )
        for u, v, _ in edges:
            k = bond_key[(min(u, v), max(u, v))]
            fused = 1.0 if (min(u, v), max(u, v)) in fused_edges else 0.0
            bond_acc[k] = [
                0.0,
                float(np.log1p(len(nodes))),
                float(np.log1p(len(edges))),
                float(np.log1p(rank)),
                float(np.log1p(attachments)),
                *type_frac.tolist(),
                1.0 if rank == 1 else 0.0,
                1.0 if rank >= 2 else 0.0,
                fused,
            ]
            if fused:
                atom_fused_edges[u] += 1
                atom_fused_edges[v] += 1

    for k, acc in enumerate(bond_acc):
        if acc:
            bond_out[k] = np.asarray(acc, dtype=np.float32)

    for a in range(n_atoms):
        ids = atom_cyclic_blocks[a]
        nblk = len(ids)
        slot = min(nblk, RING_ATOM_NBLK - 1)
        off = 0
        atom_out[a, off + slot] = 1.0
        off += RING_ATOM_NBLK
        if nblk == 0:
            continue
        ranks = [int(cyclic_blocks[i]["rank"]) for i in ids]  # type: ignore[arg-type]
        sizes = [int(cyclic_blocks[i]["n_atoms"]) for i in ids]  # type: ignore[arg-type]
        attach = [int(cyclic_blocks[i]["attachments"]) for i in ids]  # type: ignore[arg-type]
        type_fracs = np.stack([np.asarray(cyclic_blocks[i]["type_frac"], dtype=np.float64) for i in ids], 0)  # type: ignore[arg-type]
        cut = 0
        for i in ids:
            nodes = cyclic_blocks[i]["nodes"]
            if any(v not in nodes for v, _ in adj_all[a]):  # type: ignore[operator]
                cut += 1
        atom_out[a, off] = float(np.log1p(max(ranks))); off += 1
        atom_out[a, off] = float(np.log1p(max(sizes))); off += 1
        atom_out[a, off] = float(np.log1p(min(sizes))); off += 1
        atom_out[a, off] = float(np.log1p(sum(ranks))); off += 1
        atom_out[a, off] = float(cut) / float(nblk); off += 1
        atom_out[a, off] = float(np.log1p(max(attach))); off += 1
        atom_out[a, off] = 1.0 if nblk >= 2 else 0.0; off += 1
        atom_out[a, off] = 1.0 if atom_fused_edges[a] >= 1 else 0.0; off += 1
        atom_out[a, off] = 1.0 if any(r == 1 for r in ranks) else 0.0; off += 1
        atom_out[a, off] = 1.0 if any(r >= 2 for r in ranks) else 0.0; off += 1
        atom_out[a, off : off + sro.BOND_CATEGORIES] = type_fracs.mean(0).astype(np.float32)
        off += sro.BOND_CATEGORIES
        assert off == RING_ATOM_DIM, (off, RING_ATOM_DIM)

    meta = {
        "n_cyclic_blocks": len(cyclic_blocks),
        "n_bridge_edges": int((bond_out[:, 0] > 0.5).sum()),
        "n_fused_edges": int((bond_out[:, RING_BOND_DIM - 1] > 0.5).sum()),
        "n_spiro_centers": int((atom_out[:, RING_ATOM_NBLK + 7] > 0.5).sum()),
    }
    return atom_out, bond_out, meta


def _adj_list(bonds: Sequence[tuple[int, int, int]], n_atoms: int) -> list[list[tuple[int, int]]]:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for u, v, t in bonds:
        adj[u].append((v, int(t)))
        adj[v].append((u, int(t)))
    return adj


def _fused_edges_within_block(
    nodes: set[int],
    edges: Sequence[tuple[int, int, int]],
    adj_all: Sequence[Sequence[tuple[int, int]]],
    max_block_atoms: int,
) -> set[tuple[int, int]]:
    """Edges of a cyclic block that lie on >= 2 distinct simple cycles.

    An edge lies on >= 2 cycles iff its endpoints are joined by >= 2 distinct
    simple paths in the block without that edge.  Paths are counted with a
    cap of 2 on a small graph (blocks are tiny in ZINC), so this is a fixed,
    label-blind graph algorithm.
    """
    if len(nodes) > max_block_atoms:
        # Defensive: fall back to the block cycle rank criterion.
        return {(min(u, v), max(u, v)) for u, v, _ in edges} if len(edges) - len(nodes) + 1 >= 2 else set()
    adj: dict[int, list[int]] = {u: [] for u in nodes}
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)
    fused: set[tuple[int, int]] = set()
    for u, v, _ in edges:
        key = (min(u, v), max(u, v))
        if _simple_path_count_capped(u, v, adj, key, cap=2) >= 2:
            fused.add(key)
    return fused


# ---------------------------------------------------------------------------
# full extended builder
# ---------------------------------------------------------------------------


def build_extended_objects(
    mol_index: int,
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    smiles: str = "",
) -> ExtendedMolObjects:
    """Raw PyG molecule -> base BJOC objects + raw new typed-path/ring-block channels."""
    base = sro.build_mol_objects(mol_index, x, edge_index, edge_attr, smiles)
    bonds = [(int(u), int(v), int(t)) for (u, v), t in zip(base.bond_index, base.bond_type)]
    adj = _adj_list(bonds, base.n_atoms)
    is_ring_bond, _ring_atom, _n_ring_systems = sro._ring_bond_flags(base.n_atoms, bonds)

    # adjacency carrying the global physical-bond id (matches bond_key ordering)
    adj_ids: list[list[tuple[int, int, int]]] = [[] for _ in range(base.n_atoms)]
    for k, (u, v, t) in enumerate(bonds):
        adj_ids[u].append((v, k, t))
        adj_ids[v].append((u, k, t))
    for u in range(base.n_atoms):
        adj_ids[u].sort()

    atom_new = np.zeros((base.n_atoms, NEW_ATOM_DIM), dtype=np.float32)
    # ---- typed paths (root-wise, local order; no atom index dependence) ----
    atom_types = [int(v) for v in np.asarray(x).reshape(-1)]
    for root in range(base.n_atoms):
        n2 = 0
        n3 = 0
        for a, ka, t1 in adj_ids[root]:
            for b, kb, t2 in adj_ids[a]:
                if b == root:
                    continue
                n2 += 1
                atom_new[root, t1 * 4 + t2] += 1.0
                ring2 = float(is_ring_bond[ka]) + float(is_ring_bond[kb])
                atom_new[root, PATH2_SEQ + PATH3_SEQ + int(ring2)] += 1.0
                for c, kc, t3 in adj_ids[b]:
                    if c == root or c == a:
                        continue
                    n3 += 1
                    seq3 = (t1 * 4 + t2) * 4 + t3
                    atom_new[root, PATH2_SEQ + seq3] += 1.0
                    base3 = PATH2_SEQ + PATH3_SEQ + PATH2_RING
                    atom_new[root, base3 + int(ring2 + float(is_ring_bond[kc]))] += 1.0
                    atom_new[root, base3 + PATH3_RING + int(atom_types[c])] += 1.0
        b_off = PATH2_SEQ + PATH3_SEQ + PATH2_RING + PATH3_RING + PATH3_ENDTYPE
        atom_new[root, b_off] = np.log1p(n2)
        atom_new[root, b_off + 1] = np.log1p(n3)
    atom_new[:, : PATH2_SEQ + PATH3_SEQ] = np.log1p(atom_new[:, : PATH2_SEQ + PATH3_SEQ])
    ring_slice = slice(PATH2_SEQ + PATH3_SEQ, PATH2_SEQ + PATH3_SEQ + PATH2_RING + PATH3_RING)
    atom_new[:, ring_slice] = np.minimum(atom_new[:, ring_slice], 1.0)
    end_slice = slice(
        PATH2_SEQ + PATH3_SEQ + PATH2_RING + PATH3_RING,
        PATH2_SEQ + PATH3_SEQ + PATH2_RING + PATH3_RING + PATH3_ENDTYPE,
    )
    atom_new[:, end_slice] = np.log1p(atom_new[:, end_slice])

    # ---- ring blocks ----
    rb_atom, rb_bond, _meta = ring_block_blocks(base.n_atoms, bonds)
    atom_new[:, PATH_ATOM_DIM:] = rb_atom
    return ExtendedMolObjects(base=base, atom_new=atom_new, bond_new=rb_bond)


# ---------------------------------------------------------------------------
# batching + standardisation
# ---------------------------------------------------------------------------


def standardise_stats(
    objects: Sequence[ExtendedMolObjects],
    fit_rows: Sequence[int],
) -> dict[str, object]:
    """Fit-only mean/std of the new channels (atom rows and bond rows separately)."""
    atoms = np.concatenate([objects[int(i)].atom_new for i in fit_rows], 0).astype(np.float64)
    bond_count = sum(int(objects[int(i)].bond_new.shape[0]) for i in fit_rows)
    if bond_count:
        bonds = np.concatenate([objects[int(i)].bond_new for i in fit_rows], 0).astype(np.float64)
    else:
        bonds = np.zeros((0, NEW_BOND_DIM), dtype=np.float64)
    out = {}
    for name, arr in (("atom", atoms), ("bond", bonds)):
        mean = arr.mean(0)
        std = arr.std(0)
        std = np.where(std < 1e-8, 1.0, std)
        out[name] = {"mean": mean, "scale": std, "n_rows": int(arr.shape[0])}
    return out


def apply_standardisation(
    objects: Sequence[ExtendedMolObjects],
    stats: dict[str, object],
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack standardised new channels for a list of molecules.

    ``mode`` is ``"real"`` (standardised values) or ``"zero"`` (structural
    padding: exactly zero *after* standardisation, so the arm receives no
    typed-path/ring-block content but keeps the identical architecture).
    """
    if mode not in ("real", "zero"):
        raise ValueError(mode)
    if mode == "zero":
        n_atoms = sum(int(o.base.n_atoms) for o in objects)
        n_bonds = sum(int(o.base.bond_index.shape[0]) for o in objects)
        return (
            np.zeros((n_atoms, NEW_ATOM_DIM), dtype=np.float32),
            np.zeros((n_bonds, NEW_BOND_DIM), dtype=np.float32),
        )
    am = np.asarray(stats["atom"]["mean"], dtype=np.float64)  # type: ignore[index]
    asc = np.asarray(stats["atom"]["scale"], dtype=np.float64)  # type: ignore[index]
    bm = np.asarray(stats["bond"]["mean"], dtype=np.float64)  # type: ignore[index]
    bsc = np.asarray(stats["bond"]["scale"], dtype=np.float64)  # type: ignore[index]
    atoms = np.concatenate([o.atom_new for o in objects], 0).astype(np.float64)
    bonds = np.concatenate([o.bond_new for o in objects], 0).astype(np.float64)
    return (
        ((atoms - am) / asc).astype(np.float32),
        ((bonds - bm) / bsc).astype(np.float32),
    )
