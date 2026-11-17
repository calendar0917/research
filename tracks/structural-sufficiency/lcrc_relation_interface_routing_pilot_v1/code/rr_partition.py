"""Frozen topology-block partition of the physical bond graph.

Rule (declared in ``PROTOCOL.md`` before any statistic was computed; no SMARTS,
no functional-group list, no label, no atom index):

1. Find every **bridge** of the undirected physical bond graph: a bond whose
   deletion increases the number of connected components.
2. The non-bridge bonds form **cycle blocks**: each connected component of the
   non-bridge subgraph is one block (fused/spiro rings that share an atom land in
   the same block; rings sharing no atom land in different blocks).
3. The bridge bonds form a forest.  A maximal **chain block** is a maximal path
   whose *interior* vertices satisfy ``deg_bridge(v) == 2`` and ``v`` is not an
   atom of a cycle block.  A path stops at a branch vertex, at a ring atom, or at
   a leaf.  Every bridge bond belongs to exactly one chain block.
4. Every atom with no physical bond forms its own **zero-bond block**.
5. Every physical bond is owned by exactly one block; each block's bond set is
   connected; different blocks may share connection atoms but never share a bond.

Atom aggregation weight for atom ``v`` and block ``i``::

    w(v, i) = #incident bonds owned by block i / degree(v)

so ``sum_i w(v,i) = 1`` for every non-isolated atom (and ``w(v,i)=1`` for the
single zero-bond block of an isolated atom).  This is an *information routing*
weight; it does not claim that a chemical atom is uniquely apportioned to
functional groups.

Block ordering is intrinsic.  Blocks are sorted by a key computed only from the
block's own labelled structure (kind, bond/atom counts, a 3-round
Weisfeiler-Lehman hash of the labelled block subgraph with per-atom
block/external incidence roles, and the sorted per-atom label multiset).  Two
blocks with identical keys are isomorphic and indistinguishable to every
downstream symmetric function, so the readout is invariant to their index order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BLOCK_KIND_RANK = {"cycle": 0, "chain": 1, "isolated_atom": 2}
WL_ROUNDS = 3


# ---------------------------------------------------------------------------
# graph primitives
# ---------------------------------------------------------------------------


def bridge_mask(n_atoms: int, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    """Iterative Tarjan bridge finding (no recursion; safe for deep trees)."""
    m = len(edges)
    if m == 0:
        return np.zeros(0, dtype=bool)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(int(n_atoms))]
    for e, (u, v) in enumerate(edges):
        adj[u].append((v, e))
        adj[v].append((u, e))
    disc = np.full(int(n_atoms), -1, dtype=np.int64)
    low = np.zeros(int(n_atoms), dtype=np.int64)
    is_bridge = np.zeros(m, dtype=bool)
    clock = 0
    for start in range(int(n_atoms)):
        if disc[start] != -1:
            continue
        disc[start] = low[start] = clock
        clock += 1
        stack: list[list[int]] = [[start, -1, 0]]      # [vertex, parent_edge, next_index]
        while stack:
            frame = stack[-1]
            v, parent_edge, pos = frame
            if pos < len(adj[v]):
                w, e = adj[v][pos]
                frame[2] = pos + 1
                if e == parent_edge:
                    continue
                if disc[w] == -1:
                    disc[w] = low[w] = clock
                    clock += 1
                    stack.append([w, e, 0])
                else:
                    low[v] = min(int(low[v]), int(disc[w]))
            else:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(int(low[u]), int(low[v]))
                    if int(low[v]) > int(disc[u]):
                        is_bridge[parent_edge] = True
    return is_bridge


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(int(n)))

    def find(self, a: int) -> int:
        parent = self.parent
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


# ---------------------------------------------------------------------------
# the partition
# ---------------------------------------------------------------------------


@dataclass
class Partition:
    """One molecule's frozen block decomposition."""

    n_atoms: int
    n_bonds: int
    bond_block: np.ndarray                       # (m,) block id per physical bond
    blocks: list[dict[str, Any]]                 # ordered; each has bonds/atoms/kind
    bridge: np.ndarray                           # (m,) bool
    degrees: np.ndarray                          # (n,) physical degree
    n_blocks: int = 0

    def block_bonds(self) -> list[list[int]]:
        return [list(b["bonds"]) for b in self.blocks]

    def block_atoms(self) -> list[list[int]]:
        return [list(b["atoms"]) for b in self.blocks]

    def weights(self) -> list[list[tuple[int, float]]]:
        """Per-atom ``(block, weight)`` memberships (sorted by block id)."""
        out: list[list[tuple[int, float]]] = [[] for _ in range(self.n_atoms)]
        deg = self.degrees.astype(np.float64)
        for block_id, block in enumerate(self.blocks):
            for v in block["atoms"]:
                if int(self.degrees[v]) == 0:
                    w = 1.0
                else:
                    owned = int(block["atom_bond_counts"][v])
                    w = owned / float(deg[v])
                out[int(v)].append((block_id, float(w)))
        for v in range(self.n_atoms):
            out[v].sort()
        return out


# ---------------------------------------------------------------------------
# intrinsic block key (permutation invariant)
# ---------------------------------------------------------------------------


def _hash_pairs(*parts: Any) -> str:
    text = "|".join(str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _wl_block_hash(
    bonds: Sequence[int],
    edges: Sequence[tuple[int, int]],
    atom_type: np.ndarray,
    bond_type: np.ndarray,
    atom_block_degree: Mapping[int, int],
    atom_external_degree: Mapping[int, int],
) -> tuple[str, list[str]]:
    """3-round WL hash of the labelled block subgraph plus per-atom signatures."""
    blockedges = [int(e) for e in bonds]
    nodes = sorted({int(v) for e in blockedges for v in edges[e]})
    local = {v: i for i, v in enumerate(nodes)}
    labels = []
    for v in nodes:
        labels.append(
            _hash_pairs(
                "atom", int(atom_type[v]),
                int(atom_block_degree.get(v, 0)),
                int(atom_external_degree.get(v, 0)),
            )
        )
    nbrs: list[list[tuple[str, str]]] = [[] for _ in nodes]
    for e in blockedges:
        u, v = edges[e]
        bt = _hash_pairs("bond", int(bond_type[e]))
        nbrs[local[u]].append((bt, labels[local[v]]))
        nbrs[local[v]].append((bt, labels[local[u]]))
    for _ in range(WL_ROUNDS):
        new = []
        for i, lab in enumerate(labels):
            new.append(_hash_pairs("wl", lab, *sorted(f"{bt}:{hl}" for bt, hl in nbrs[i])))
        labels = new
        nbrs = [[] for _ in nodes]
        for e in blockedges:
            u, v = edges[e]
            bt = _hash_pairs("bond", int(bond_type[e]))
            nbrs[local[u]].append((bt, labels[local[v]]))
            nbrs[local[v]].append((bt, labels[local[u]]))
    atom_sig = sorted(labels)
    return _hash_pairs("block", *atom_sig), atom_sig


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def partition_molecule(
    n_atoms: int,
    edges: Sequence[tuple[int, int]],
    atom_type: Sequence[int],
    bond_type: Sequence[int],
) -> Partition:
    """Deterministic, label-free, SMARTS-free topology partition of one molecule."""
    n = int(n_atoms)
    edges = [(int(u), int(v)) for u, v in edges]
    m = len(edges)
    atom_type = np.asarray(atom_type, dtype=np.int64).reshape(-1)
    bond_type = np.asarray(bond_type, dtype=np.int64).reshape(-1)
    if atom_type.shape != (n,):
        raise ValueError(f"atom_type shape {atom_type.shape} != {(n,)}")
    if bond_type.shape != (m,):
        raise ValueError(f"bond_type shape {bond_type.shape} != {(m,)}")
    bridge = bridge_mask(n, edges)

    deg = np.zeros(n, dtype=np.int64)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e, (u, v) in enumerate(edges):
        deg[u] += 1
        deg[v] += 1
        adj[u].append((v, e))
        adj[v].append((u, e))

    # ---- cycle blocks: components of the non-bridge subgraph --------------
    non_bridge = [e for e in range(m) if not bridge[e]]
    uf = _UnionFind(n)
    for e in non_bridge:
        u, v = edges[e]
        uf.union(u, v)
    groups: dict[int, list[int]] = {}
    for e in non_bridge:
        groups.setdefault(uf.find(edges[e][0]), []).append(int(e))
    cycle_bond_sets = [sorted(g) for _, g in sorted(groups.items())]
    cycle_atoms: set[int] = set()
    for bonds in cycle_bond_sets:
        for e in bonds:
            cycle_atoms.add(edges[e][0])
            cycle_atoms.add(edges[e][1])

    # ---- chain blocks: maximal bridge paths ------------------------------
    bridge_adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e in range(m):
        if bridge[e]:
            u, v = edges[e]
            bridge_adj[u].append((v, int(e)))
            bridge_adj[v].append((u, int(e)))
    deg_bridge = np.asarray([len(row) for row in bridge_adj], dtype=np.int64)
    passthrough = [(int(deg_bridge[v]) == 2) and (v not in cycle_atoms) for v in range(n)]

    seen: set[int] = set()
    chain_bond_sets: list[list[int]] = []
    for e0 in range(m):
        if not bridge[e0] or e0 in seen:
            continue
        group = {int(e0)}
        seen.add(int(e0))
        for side in (0, 1):
            cur_vertex = edges[e0][side]
            cur_edge = int(e0)
            while passthrough[cur_vertex]:
                options = [(w, ee) for w, ee in bridge_adj[cur_vertex] if ee != cur_edge]
                if len(options) != 1:
                    break
                w, ee = options[0]
                if ee in group:
                    break
                group.add(int(ee))
                seen.add(int(ee))
                cur_edge = int(ee)
                cur_vertex = int(w)
        chain_bond_sets.append(sorted(group))
    chain_bond_sets.sort()

    # ---- isolated atoms: zero-bond blocks --------------------------------
    connected = np.zeros(n, dtype=bool)
    for e, (u, v) in enumerate(edges):
        connected[u] = True
        connected[v] = True
    isolated_atoms = [int(v) for v in range(n) if not connected[v]]

    # ---- assemble, key, and order ---------------------------------------
    raw_blocks: list[dict[str, Any]] = []
    for bonds in cycle_bond_sets:
        raw_blocks.append({"kind": "cycle", "bonds": bonds})
    for bonds in chain_bond_sets:
        raw_blocks.append({"kind": "chain", "bonds": bonds})
    for v in isolated_atoms:
        raw_blocks.append({"kind": "isolated_atom", "bonds": [], "isolated_atom": int(v)})

    for block in raw_blocks:
        bonds = block["bonds"]
        if bonds:
            atoms = sorted({int(v) for e in bonds for v in edges[e]})
            counts: dict[int, int] = {}
            for e in bonds:
                for v in edges[e]:
                    counts[int(v)] = counts.get(int(v), 0) + 1
            block["atoms"] = atoms
            block["atom_bond_counts"] = counts
            atom_deg_b = {int(v): int(counts.get(int(v), 0)) for v in atoms}
            atom_deg_ext = {int(v): int(deg[v] - counts.get(int(v), 0)) for v in atoms}
            key_hash, atom_sig = _wl_block_hash(
                bonds, edges, atom_type, bond_type, atom_deg_b, atom_deg_ext
            )
            block["wl_hash"] = key_hash
            block["atom_signatures"] = atom_sig
        else:
            v = int(block["isolated_atom"])
            block["atoms"] = [v]
            block["atom_bond_counts"] = {}
            key_hash, _ = _wl_block_hash([], edges, atom_type, bond_type, {}, {})
            block["wl_hash"] = key_hash
            block["atom_signatures"] = [f"isolated:{int(atom_type[v])}"]
        block["n_bonds"] = len(bonds)
        block["n_atoms"] = len(block["atoms"])
        block["key"] = (
            int(BLOCK_KIND_RANK[block["kind"]]),
            int(block["n_bonds"]),
            int(block["n_atoms"]),
            block["wl_hash"],
            tuple(block["atom_signatures"]),
        )

    order = sorted(range(len(raw_blocks)), key=lambda i: raw_blocks[i]["key"])
    blocks = [raw_blocks[i] for i in order]
    for index, block in enumerate(blocks):
        block["index"] = index
    # zero is a valid block id, so ownership is counted explicitly (never inferred)
    bond_block = np.zeros(m, dtype=np.int64)
    owned = np.zeros(m, dtype=np.int64)
    for index, block in enumerate(blocks):
        for e in block["bonds"]:
            bond_block[e] = index
            owned[e] += 1
    if m and int(owned.min()) != 1:
        raise RuntimeError("a physical bond is not owned exactly once")
    return Partition(
        n_atoms=n,
        n_bonds=m,
        bond_block=bond_block,
        blocks=blocks,
        bridge=bridge,
        degrees=deg,
        n_blocks=len(blocks),
    )


# ---------------------------------------------------------------------------
# structural checks
# ---------------------------------------------------------------------------


def check_partition(part: Partition, edges: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Gates 1, 2 and 5 for one molecule."""
    m = part.n_bonds
    owns = np.zeros(m, dtype=np.int64)
    for block in part.blocks:
        for e in block["bonds"]:
            owns[int(e)] += 1
    unique = bool(m == 0 or int(owns.min()) == 1)
    over = int((owns > 1).sum())

    connected_ok = True
    for block in part.blocks:
        bonds = list(block["bonds"])
        if not bonds:
            if len(block["atoms"]) != 1:
                connected_ok = False
            continue
        uf = _UnionFind(part.n_atoms)
        for e in bonds:
            u, v = edges[e]
            uf.union(u, v)
        roots = {uf.find(v) for e in bonds for v in edges[e]}
        if len(roots) != 1:
            connected_ok = False

    weights = part.weights()
    weight_sums = []
    for v, memberships in enumerate(weights):
        total = float(sum(w for _, w in memberships))
        if part.degrees[v] == 0:
            ok = len(memberships) == 1 and abs(total - 1.0) < 1e-12
        else:
            ok = abs(total - 1.0) < 1e-12
        weight_sums.append((ok, total, len(memberships)))
    weights_ok = all(ok for ok, _, _ in weight_sums)
    return {
        "bond_unique_ownership": unique,
        "n_bonds_owned_twice": over,
        "blocks_connected": bool(connected_ok),
        "weights_sum_to_one": bool(weights_ok),
        "n_blocks": int(part.n_blocks),
        "n_zero_bond_blocks": int(sum(1 for b in part.blocks if b["n_bonds"] == 0)),
        "max_atom_block_multiplicity": int(max((c for _, _, c in weight_sums), default=0)),
    }


__all__ = [
    "BLOCK_KIND_RANK",
    "Partition",
    "bridge_mask",
    "partition_molecule",
    "check_partition",
]
