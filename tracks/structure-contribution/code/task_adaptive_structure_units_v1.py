"""Task-adaptive structure units (cscl-task-adaptive-structure-units-v1).

Track: ``tracks/structure-contribution`` · protocol:
``configs/task_adaptive_structure_units_v1.yaml``.

Research question
-----------------
Can structural-unit boundaries be *learned* from the downstream molecule
property loss, starting from the existing Atom/Bond encoders and the real
molecular graph, while keeping every unit connected, non-overlapping and
traceable to original atom/bond ids?

This module implements the frozen prototype:

1. **Physical-bond canonicalisation** — PyG ZINC stores every bond twice;
   :func:`cscl_units.unique_undirected_bonds` reduces it to one undirected
   record with a stable id (BUG-2 of ``cscl-correctness-v1`` must not recur).
2. **Base blocks** — bridges (edges whose removal disconnects) and the
   connected components after deleting every bridge (2-edge-connected
   blocks).  Each atom is in exactly one block; every block is connected;
   ring bonds are never bridges.  Contracting blocks yields a tree.
3. **Learnable merge decisions** — one scalar gate per bridge from a shared
   small MLP over base-block summaries (sum pooling, no message passing).
   Hard forward (threshold 0.5, union of merged blocks) with a
   straight-through surrogate so the property loss can update the gates.
4. **Explicit unit objects** — every final unit gets a 96-D representation
   from pooled atom/bond embeddings (+ size descriptors); every retained
   bridge becomes a 96-D undirected unit-relation object.  Every physical
   bond has exactly one role: unit-internal or unit-relation.
5. **2x2 readouts** — additive (``y = beta + sum(alpha) + sum(gamma)``,
   numerically exact per sample) or opaque (reader MLP over the two sums).

Discipline
----------
* No GNN / Transformer / attention / learned iterative message passing.
* No Junction encoder in the supervised model (the junction tensor is never
  read); only the Atom/Bond encoders are reused from the frozen masked SSL.
* ``y`` is read only by the supervised loop and scoring; topology and all
  audits are label-blind.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_units as cu  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-task-adaptive-structure-units-v1"

#: arms of the frozen 2x2 matrix
PARTITION_FIXED = "fixed"
PARTITION_LEARNED = "learned"
READOUT_ADDITIVE = "additive"
READOUT_OPAQUE = "opaque"
PARTITION_MODES = (PARTITION_FIXED, PARTITION_LEARNED)
READOUT_MODES = (READOUT_ADDITIVE, READOUT_OPAQUE)
ARMS = ("F-A", "L-A", "F-O", "L-O")


def arm_modes(arm: str) -> tuple[str, str]:
    """Return (partition_mode, readout_mode) for one of the four arms."""
    table = {
        "F-A": (PARTITION_FIXED, READOUT_ADDITIVE),
        "L-A": (PARTITION_LEARNED, READOUT_ADDITIVE),
        "F-O": (PARTITION_FIXED, READOUT_OPAQUE),
        "L-O": (PARTITION_LEARNED, READOUT_OPAQUE),
    }
    if arm not in table:
        raise ValueError(f"unknown arm {arm!r}")
    return table[arm]


FORBIDDEN_TOKENS = (
    "MessagePassing",
    "GCN",
    "GIN",
    "GAT",
    "SAGEConv",
    "Transformer",
    "MultiheadAttention",
    "nn.MultiheadAttention",
    "self_attention",
    "TransformerEncoder",
    "GRU",
    "LSTM",
    "RNN",
    "GraphConv",
    "ChebConv",
    "TAGConv",
)

#: frozen merge-scorer initialisation (before GPU): zero last-layer weight and
#: a fixed negative bias; the gate probability at init is sigmoid(-0.5).
MERGE_INIT_BIAS = -0.5

#: components of the unit descriptor H_k
UNIT_DESCRIPTOR_DIM = 2 * 96 + 2  # sum atoms, sum internal bonds, log1p counts


# ---------------------------------------------------------------------------
# topology: physical bonds, bridges, 2-edge-connected blocks
# ---------------------------------------------------------------------------


def reference_bridges(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> set[tuple[int, int]]:
    """Independent Tarjan-style bridge finder (no networkx).

    ``bonds`` is a list of unique undirected ``(u, v, bond_type)`` records.
    Returns the set of bridge keys ``(min(u,v), max(u,v))``.
    """
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for k, (u, v, _) in enumerate(bonds):
        adj[u].append((v, k))
        adj[v].append((u, k))
    disc = [-1] * n_atoms
    low = [0] * n_atoms
    bridges: set[tuple[int, int]] = set()
    timer = 0

    for source in range(n_atoms):
        if disc[source] != -1:
            continue
        stack: list[tuple[int, int, int]] = [(source, -1, 0)]
        disc[source] = low[source] = timer
        timer += 1
        while stack:
            node, parent_edge, child_index = stack[-1]
            if child_index < len(adj[node]):
                stack[-1] = (node, parent_edge, child_index + 1)
                nxt, edge_id = adj[node][child_index]
                if edge_id == parent_edge:
                    continue
                if disc[nxt] == -1:
                    disc[nxt] = low[nxt] = timer
                    timer += 1
                    stack.append((nxt, edge_id, 0))
                else:
                    low[node] = min(low[node], disc[nxt])
            else:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[node])
                    if low[node] > disc[parent]:
                        u, v, _ = bonds[parent_edge]
                        bridges.add((min(u, v), max(u, v)))
    return bridges


def brute_force_bridges(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> set[tuple[int, int]]:
    """O(E * (V+E)) connectivity check per edge (audit reference)."""
    bridges: set[tuple[int, int]] = set()
    for skip, (u, v, _) in enumerate(bonds):
        adj: list[list[int]] = [[] for _ in range(n_atoms)]
        for k, (a, b, _) in enumerate(bonds):
            if k == skip:
                continue
            adj[a].append(b)
            adj[b].append(a)
        seen = {u}
        frontier = [u]
        while frontier:
            nxt: list[int] = []
            for x in frontier:
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        nxt.append(y)
            frontier = nxt
        if v not in seen:
            bridges.add((min(u, v), max(u, v)))
    return bridges


@dataclass
class StructureTopology:
    """Static, label-blind block/bridge/path structure of one molecule."""

    mol_index: int
    n_atoms: int
    n_bonds: int
    n_blocks: int
    block_of_atom: np.ndarray        # [n_atoms] local block id 0..B-1
    bond_block_u: np.ndarray         # [n_bonds]
    bond_block_v: np.ndarray         # [n_bonds]
    is_bridge: np.ndarray            # [n_bonds] bool
    bridge_bond: np.ndarray          # [n_bridge] local bond ids
    bridge_block_u: np.ndarray       # [n_bridge]
    bridge_block_v: np.ndarray       # [n_bridge]
    tree_edge: np.ndarray            # [n_bridge, 2] block endpoints (local ids)
    pair_u: np.ndarray               # [B*B] ordered pairs (local ids)
    pair_v: np.ndarray               # [B*B]
    pair_edge_offsets: np.ndarray    # [B*B+1] CSR into pair_edge_ids
    pair_edge_ids: np.ndarray        # [E] bridge index on the path
    block_atom_ids: list[np.ndarray] = field(default_factory=list)

    def hard_components(self, gate_merge: np.ndarray) -> tuple[np.ndarray, int]:
        """Union-find components over bridge merges; returns (rep_of_block, n_units).

        ``gate_merge`` is a bool array over bridges (True = merge).  The
        representative of a component is its minimum local block id.
        """
        parent = list(range(self.n_blocks))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for k in range(len(self.bridge_bond)):
            if bool(gate_merge[k]):
                a = find(int(self.bridge_block_u[k]))
                b = find(int(self.bridge_block_v[k]))
                if a != b:
                    if a > b:
                        a, b = b, a
                    parent[b] = a
        reps = np.asarray([find(b) for b in range(self.n_blocks)], dtype=np.int64)
        return reps, int(len(np.unique(reps)))

    def to_json(self) -> dict[str, Any]:
        return {
            "mol_index": int(self.mol_index),
            "n_atoms": int(self.n_atoms),
            "n_bonds": int(self.n_bonds),
            "n_blocks": int(self.n_blocks),
            "n_bridges": int(len(self.bridge_bond)),
            "block_sizes": [int(len(x)) for x in self.block_atom_ids],
        }


def build_structure_topology(
    mol_index: int,
    bonds: Sequence[tuple[int, int, int]],
    *,
    check_reference: bool = True,
) -> StructureTopology:
    """Build the frozen block/bridge topology of one connected molecule.

    ``bonds``: unique undirected physical bonds ``(u, v, bond_type)`` with
    ``u < v`` (canonicalised by :func:`cscl_units.unique_undirected_bonds`).
    """
    bonds = [(int(u), int(v), int(bt)) for u, v, bt in bonds]
    n_atoms = max((max(u, v) for u, v, _ in bonds), default=-1) + 1
    n_bonds = len(bonds)
    if n_atoms == 0:
        raise ValueError("empty molecule")
    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    graph.add_edges_from((u, v) for u, v, _ in bonds)
    if not nx.is_connected(graph):
        raise ValueError(f"molecule {mol_index} is disconnected: prototype requires connected graphs")
    bridges_nx = {tuple(sorted(e)) for e in nx.bridges(graph)}
    if check_reference:
        ref = reference_bridges(n_atoms, bonds)
        if ref != bridges_nx:
            raise AssertionError(f"bridge reference mismatch: nx={bridges_nx} ref={ref}")

    stripped = graph.copy()
    stripped.remove_edges_from(bridges_nx)
    components = [set(c) for c in nx.connected_components(stripped)]
    components.sort(key=lambda s: min(s))
    block_of_atom = np.full(n_atoms, -1, dtype=np.int64)
    for b, atoms in enumerate(components):
        for a in sorted(atoms):
            block_of_atom[a] = b
    if (block_of_atom < 0).any():
        raise AssertionError("atom without a 2-edge-connected block")
    n_blocks = len(components)

    bond_block_u = np.empty(n_bonds, dtype=np.int64)
    bond_block_v = np.empty(n_bonds, dtype=np.int64)
    is_bridge = np.zeros(n_bonds, dtype=bool)
    for k, (u, v, _) in enumerate(bonds):
        bu, bv = int(block_of_atom[u]), int(block_of_atom[v])
        bond_block_u[k], bond_block_v[k] = bu, bv
        if bu != bv:
            is_bridge[k] = True
        if is_bridge[k] != ((u, v) in bridges_nx):
            raise AssertionError("cross-block bond set differs from the bridge set")

    bridge_bond = np.asarray([k for k in range(n_bonds) if is_bridge[k]], dtype=np.int64)
    bridge_block_u = bond_block_u[bridge_bond]
    bridge_block_v = bond_block_v[bridge_bond]
    tree_edge = np.stack([bridge_block_u, bridge_block_v], axis=1) if len(bridge_bond) else np.zeros((0, 2), dtype=np.int64)

    # ordered paths between all block pairs (B is small; BFS per source)
    block_adj: list[list[tuple[int, int]]] = [[] for _ in range(n_blocks)]
    for e, (u, v) in enumerate(zip(bridge_block_u.tolist(), bridge_block_v.tolist())):
        block_adj[u].append((v, e))
        block_adj[v].append((u, e))
    pair_u, pair_v = [], []
    pair_edge_offsets = [0]
    pair_edge_ids: list[int] = []
    for s in range(n_blocks):
        parent = [-1] * n_blocks
        parent_edge = [-1] * n_blocks
        seen = [False] * n_blocks
        seen[s] = True
        frontier = [s]
        while frontier:
            nxt: list[int] = []
            for node in frontier:
                for nbr, e in block_adj[node]:
                    if not seen[nbr]:
                        seen[nbr] = True
                        parent[nbr] = node
                        parent_edge[nbr] = e
                        nxt.append(nbr)
            frontier = nxt
        for t in range(n_blocks):
            pair_u.append(s)
            pair_v.append(t)
            path: list[int] = []
            cur = t
            while cur != s:
                if parent_edge[cur] < 0:
                    raise AssertionError("block tree is not connected")
                path.append(int(parent_edge[cur]))
                cur = parent[cur]
            path.reverse()
            pair_edge_ids.extend(path)
            pair_edge_offsets.append(len(pair_edge_ids))
    block_atom_ids = [np.asarray(sorted(atoms), dtype=np.int64) for atoms in components]
    return StructureTopology(
        mol_index=int(mol_index),
        n_atoms=int(n_atoms),
        n_bonds=int(n_bonds),
        n_blocks=int(n_blocks),
        block_of_atom=block_of_atom,
        bond_block_u=bond_block_u,
        bond_block_v=bond_block_v,
        is_bridge=is_bridge,
        bridge_bond=bridge_bond,
        bridge_block_u=bridge_block_u,
        bridge_block_v=bridge_block_v,
        tree_edge=tree_edge,
        pair_u=np.asarray(pair_u, dtype=np.int64),
        pair_v=np.asarray(pair_v, dtype=np.int64),
        pair_edge_offsets=np.asarray(pair_edge_offsets, dtype=np.int64),
        pair_edge_ids=np.asarray(pair_edge_ids, dtype=np.int64),
        block_atom_ids=block_atom_ids,
    )


def topology_from_objects(obj: sro.MolObjects, x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> StructureTopology:
    """Convenience: raw PyG row -> unique physical bonds -> topology."""
    directed = [(int(a), int(b)) for a, b in np.asarray(edge_index).T]
    types = [int(t) for t in np.asarray(edge_attr).reshape(-1)]
    bonds = cu.unique_undirected_bonds(directed, types)
    return build_structure_topology(int(obj.mol_index), bonds)


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def batch_atom_bond(objs: Sequence[sro.MolObjects]) -> dict[str, Any]:
    """Lightweight batch of only the Atom/Bond/global fields plus offsets.

    Junction tensors are deliberately **not** built: the supervised model
    never reads a Junction object.
    """
    atom_feat, atom_batch = [], []
    bond_feat, bond_batch, bond_index = [], [], []
    global_feat = []
    atom_off = bond_off = 0
    for g, obj in enumerate(objs):
        n_a = int(obj.n_atoms)
        n_b = int(obj.bond_index.shape[0])
        atom_feat.append(torch.as_tensor(obj.atom_feat, dtype=torch.float32))
        atom_batch.append(torch.full((n_a,), g, dtype=torch.long))
        bond_feat.append(torch.as_tensor(obj.bond_feat, dtype=torch.float32))
        bond_batch.append(torch.full((n_b,), g, dtype=torch.long))
        bond_index.append(torch.as_tensor(obj.bond_index, dtype=torch.long) + atom_off)
        global_feat.append(torch.as_tensor(obj.global_feat, dtype=torch.float32))
        atom_off += n_a
        bond_off += n_b
    return {
        "n_graphs": len(objs),
        "atom_feat": torch.cat(atom_feat, 0) if atom_feat else torch.zeros(0, sro.ATOM_FEAT_DIM),
        "atom_batch": torch.cat(atom_batch, 0) if atom_batch else torch.zeros(0, dtype=torch.long),
        "bond_feat": torch.cat(bond_feat, 0) if bond_feat else torch.zeros(0, sro.BOND_FEAT_DIM),
        "bond_batch": torch.cat(bond_batch, 0) if bond_batch else torch.zeros(0, dtype=torch.long),
        "bond_index": torch.cat(bond_index, 0) if bond_index else torch.zeros(0, 2, dtype=torch.long),
        "global_feat": torch.stack(global_feat, 0) if global_feat else torch.zeros(0, sro.GLOBAL_DIM),
    }


def batch_structure_topology(
    objs: Sequence[sro.MolObjects],
    topos: Sequence[StructureTopology],
) -> dict[str, Any]:
    """Assemble the ragged block/bridge/path tensors for one batch.

    The returned tensors use global indices (blocks, bonds, bridges) so the
    model forward is pure gather / scatter-add; no per-molecule Python loop.
    Path entries reference global bridge ids so the merge gates can be
    gathered directly.
    """
    if len(objs) != len(topos):
        raise ValueError("objects and topologies must align")
    n_graphs = len(objs)
    blk_off = np.zeros(n_graphs + 1, dtype=np.int64)
    bond_off = np.zeros(n_graphs + 1, dtype=np.int64)
    bridge_off = np.zeros(n_graphs + 1, dtype=np.int64)
    for g, (obj, topo) in enumerate(zip(objs, topos)):
        if int(topo.n_atoms) != int(obj.n_atoms) or int(topo.n_bonds) != int(obj.bond_index.shape[0]):
            raise ValueError(f"topology/object mismatch at batch position {g}")
        blk_off[g + 1] = blk_off[g] + topo.n_blocks
        bond_off[g + 1] = bond_off[g] + topo.n_bonds
        bridge_off[g + 1] = bridge_off[g] + len(topo.bridge_bond)

    atom_graph, atom_block_local, block_of_atom = [], [], []
    bond_graph, bond_block_u_local, bond_block_v_local = [], [], []
    bond_block_u, bond_block_v = [], []
    bridge_bond_global, bridge_graph, bridge_u_local, bridge_v_local = [], [], [], []
    pair_graph, pair_u_local, pair_v_local, pair_edge_ids = [], [], [], []
    pair_offsets = [0]
    for g, topo in enumerate(topos):
        n_a = int(topo.n_atoms)
        atom_graph.append(torch.full((n_a,), g, dtype=torch.long))
        atom_block_local.append(torch.as_tensor(topo.block_of_atom, dtype=torch.long))
        block_of_atom.append(torch.as_tensor(topo.block_of_atom + blk_off[g], dtype=torch.long))
        bond_graph.append(torch.full((topo.n_bonds,), g, dtype=torch.long))
        bond_block_u_local.append(torch.as_tensor(topo.bond_block_u, dtype=torch.long))
        bond_block_v_local.append(torch.as_tensor(topo.bond_block_v, dtype=torch.long))
        bond_block_u.append(torch.as_tensor(topo.bond_block_u + blk_off[g], dtype=torch.long))
        bond_block_v.append(torch.as_tensor(topo.bond_block_v + blk_off[g], dtype=torch.long))
        if len(topo.bridge_bond):
            bridge_bond_global.append(torch.as_tensor(topo.bridge_bond + bond_off[g], dtype=torch.long))
            bridge_graph.append(torch.full((len(topo.bridge_bond),), g, dtype=torch.long))
            bridge_u_local.append(torch.as_tensor(topo.bridge_block_u, dtype=torch.long))
            bridge_v_local.append(torch.as_tensor(topo.bridge_block_v, dtype=torch.long))
        n_pairs = int(topo.n_blocks) * int(topo.n_blocks)
        pair_graph.append(torch.full((n_pairs,), g, dtype=torch.long))
        pair_u_local.append(torch.as_tensor(topo.pair_u, dtype=torch.long))
        pair_v_local.append(torch.as_tensor(topo.pair_v, dtype=torch.long))
        if len(topo.pair_edge_ids):
            pair_edge_ids.append(torch.as_tensor(topo.pair_edge_ids + bridge_off[g], dtype=torch.long))
        pair_offsets.extend((topo.pair_edge_offsets[1:] + pair_offsets[-1]).tolist())

    def cat_or_empty(parts: Sequence[torch.Tensor], empty: torch.Tensor) -> torch.Tensor:
        return torch.cat(parts, 0) if parts else empty

    batch: dict[str, Any] = {
        "n_graphs": n_graphs,
        "n_blocks_pad": int(max((int(t.n_blocks) for t in topos), default=0)),
        "blk_off": torch.as_tensor(blk_off[:-1], dtype=torch.long),
        "blk_batch": torch.repeat_interleave(
            torch.arange(n_graphs, dtype=torch.long),
            torch.as_tensor([int(t.n_blocks) for t in topos], dtype=torch.long),
        ),
        "blk_slot": cat_or_empty(
            [torch.arange(int(t.n_blocks), dtype=torch.long) for t in topos], torch.zeros(0, dtype=torch.long)
        ),
        "block_of_atom": cat_or_empty(block_of_atom, torch.zeros(0, dtype=torch.long)),
        "atom_block_local": cat_or_empty(atom_block_local, torch.zeros(0, dtype=torch.long)),
        "atom_graph": cat_or_empty(atom_graph, torch.zeros(0, dtype=torch.long)),
        "bond_graph": cat_or_empty(bond_graph, torch.zeros(0, dtype=torch.long)),
        "bond_block_u_local": cat_or_empty(bond_block_u_local, torch.zeros(0, dtype=torch.long)),
        "bond_block_v_local": cat_or_empty(bond_block_v_local, torch.zeros(0, dtype=torch.long)),
        "bond_block_u": cat_or_empty(bond_block_u, torch.zeros(0, dtype=torch.long)),
        "bond_block_v": cat_or_empty(bond_block_v, torch.zeros(0, dtype=torch.long)),
        "bridge_bond": cat_or_empty(bridge_bond_global, torch.zeros(0, dtype=torch.long)),
        "bridge_graph": cat_or_empty(bridge_graph, torch.zeros(0, dtype=torch.long)),
        "bridge_u_local": cat_or_empty(bridge_u_local, torch.zeros(0, dtype=torch.long)),
        "bridge_v_local": cat_or_empty(bridge_v_local, torch.zeros(0, dtype=torch.long)),
        "pair_graph": cat_or_empty(pair_graph, torch.zeros(0, dtype=torch.long)),
        "pair_u_local": cat_or_empty(pair_u_local, torch.zeros(0, dtype=torch.long)),
        "pair_v_local": cat_or_empty(pair_v_local, torch.zeros(0, dtype=torch.long)),
        "pair_offsets": torch.as_tensor(pair_offsets, dtype=torch.long),
        "pair_edge_ids": cat_or_empty(pair_edge_ids, torch.zeros(0, dtype=torch.long)),
    }
    lengths = batch["pair_offsets"][1:] - batch["pair_offsets"][:-1]
    batch["pair_lengths"] = lengths
    if int(lengths.numel()):
        batch["pair_ids_flat"] = torch.repeat_interleave(torch.arange(int(lengths.numel()), dtype=torch.long), lengths)
        entry_pos = torch.arange(int(batch["pair_edge_ids"].shape[0]), dtype=torch.long)
        batch["pair_entry_pos"] = entry_pos - torch.repeat_interleave(batch["pair_offsets"][:-1], lengths)
        batch["pair_max_len"] = max(int(lengths.max()), 1)
        batch["max_path_len"] = int(lengths.max())
    else:
        batch["pair_entry_pos"] = torch.zeros(0, dtype=torch.long)
        batch["pair_max_len"] = 1
        batch["max_path_len"] = 0
    return batch


# ---------------------------------------------------------------------------
# synthetic known-truth molecules (CPU feasibility)
# ---------------------------------------------------------------------------


@dataclass
class SynthMolecule:
    mol_index: int
    atom_types: list[int]
    bonds: list[tuple[int, int, int]]     # unique undirected (u<v)
    true_merge: list[bool]                # per bridge (topology order)
    unit_of_atom: list[int]               # oracle unit id per atom
    atom_weight: list[float]              # planted per-atom weight
    y: float
    n_units_oracle: int
    tree_edge: list[tuple[int, int]]      # block endpoints per bridge (topology order)
    block_atom_ids: list[list[int]]       # atoms of every base block
    block_affinity: list[tuple[int, int]]  # planted structure label per block


def synth_raw_arrays(mol: SynthMolecule) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, edge_index, edge_attr) in PyG doubled-directed convention."""
    n = len(mol.atom_types)
    x = np.asarray(mol.atom_types, dtype=np.int64).reshape(n, 1)
    rows = []
    types = []
    for u, v, bt in mol.bonds:
        rows.extend([(u, v), (v, u)])
        types.extend([bt, bt])
    edge_index = np.asarray(rows, dtype=np.int64).T if rows else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.asarray(types, dtype=np.int64)
    return x, edge_index, edge_attr


def block_affinity(atom_types: Sequence[int], atoms: Sequence[int]) -> tuple[int, int]:
    """Deterministic structure label of one base block (learnable from H_k).

    ``(kind, bucket)`` with ``kind = 1`` for rings and ``0`` for single
    atoms; ``bucket`` is the rounded mean raw atom type (only meaningful
    inside a kind).  The planted merge rule is ``affinity(u) == affinity(v)``.
    """
    kind = 1 if len(atoms) > 1 else 0
    mean_type = float(np.mean([int(atom_types[a]) for a in atoms]))
    return (kind, int(round(mean_type * 2.0)))


def make_synthetic_molecule(
    rng: np.random.Generator,
    mol_index: int,
    *,
    max_blocks: int = 6,
    ring_probability: float = 0.45,
    max_attempts: int = 50,
) -> SynthMolecule:
    """Random connected molecule with explicit base blocks and merge labels.

    Base blocks are single atoms or simple rings; block-tree bridges carry a
    planted true merge/cut label that is a **deterministic function of the
    adjacent block compositions** (``affinity(u) == affinity(v)``), so the
    boundary is learnable from the same information the scorer sees.  The
    oracle partition is the connected component of the true-merge edges and
    the target is nonlinear in that partition (:func:`synthetic_y`).
    """
    for _attempt in range(int(max_attempts)):
        n_blocks = int(rng.integers(2, max_blocks + 1))
        atom_types: list[int] = []
        bonds: list[tuple[int, int, int]] = []
        block_atoms: list[list[int]] = []
        for _ in range(n_blocks):
            if rng.random() < ring_probability:
                size = int(rng.integers(3, 7))
                ids = list(range(len(atom_types), len(atom_types) + size))
                atom_types.extend(int(rng.integers(0, 4)) for _ in range(size))
                for i in range(size):
                    u, v = ids[i], ids[(i + 1) % size]
                    bonds.append((min(u, v), max(u, v), int(rng.integers(0, 3))))
                block_atoms.append(ids)
            else:
                a = len(atom_types)
                atom_types.append(int(rng.integers(0, 4)))
                block_atoms.append([a])
        bridge_spec: list[tuple[int, int, int, int, int]] = []
        for b in range(1, n_blocks):
            parent = int(rng.integers(0, b))
            u = int(rng.choice(block_atoms[parent]))
            v = int(rng.choice(block_atoms[b]))
            bridge_spec.append((parent, b, u, v, int(rng.integers(0, 3))))
        for (_, _, u, v, bt) in bridge_spec:
            bonds.append((min(u, v), max(u, v), bt))
        bonds = sorted(set(bonds))
        try:
            topo = build_structure_topology(mol_index, bonds, check_reference=False)
        except ValueError:
            continue
        if int(topo.n_blocks) != n_blocks:
            continue
        affinities = [block_affinity(atom_types, atoms) for atoms in topo.block_atom_ids]
        merge_by_edge = {
            (int(min(bu, bv)), int(max(bu, bv))): affinities[int(bu)] == affinities[int(bv)]
            for bu, bv in zip(topo.bridge_block_u.tolist(), topo.bridge_block_v.tolist())
        }
        ordered_merge = [merge_by_edge[(int(min(a, b)), int(max(a, b)))] for a, b in topo.tree_edge]
        if all(ordered_merge) or not any(ordered_merge):
            continue
        reps, n_units = topo.hard_components(np.asarray(ordered_merge, dtype=bool))
        unit_of_atom = [int(reps[int(topo.block_of_atom[a])]) for a in range(topo.n_atoms)]
        atom_weight = [float(round(rng.uniform(0.2, 1.0), 4)) for _ in range(topo.n_atoms)]
        mol = SynthMolecule(
            mol_index=mol_index,
            atom_types=atom_types,
            bonds=bonds,
            true_merge=ordered_merge,
            unit_of_atom=unit_of_atom,
            atom_weight=atom_weight,
            y=0.0,
            n_units_oracle=int(n_units),
            tree_edge=[(int(a), int(b)) for a, b in topo.tree_edge],
            block_atom_ids=[list(map(int, ids)) for ids in topo.block_atom_ids],
            block_affinity=[(int(k), int(b)) for k, b in affinities],
        )
        mol.y = synthetic_y(mol)
        return mol
    raise RuntimeError(f"synthetic molecule {mol_index}: could not sample a non-degenerate merge rule")


SYNTH_SCALE_UNARY_LINEAR = 0.4
SYNTH_SCALE_UNARY_SQUARE = 2.5   # (sum w)^2 / 10
SYNTH_SCALE_RELATION = 0.7
SYNTH_BIAS = 0.3
SYNTH_NOISE = 0.02


def synthetic_y(mol: SynthMolecule, noise_rng: np.random.Generator | None = None) -> float:
    """Planted nonlinear target that depends on the oracle partition.

    ``y = bias + 0.4 * sum_k (sum_{i in U_k} w_i)
           + 2.5 * sum_k (sum_{i in U_k} w_i)^2 / 10
           + 0.7 * sum_{cut bridges} (w_left * w_right) / 10 + noise``.

    The squared unary term and the cut-bridge term make a wrong partition
    (e.g. always cutting) non-representable by the fixed partition without
    losing the cross terms that only a merged unit can see.
    """
    units: dict[int, list[int]] = {}
    for a, u in enumerate(mol.unit_of_atom):
        units.setdefault(u, []).append(a)
    value = SYNTH_BIAS
    for atoms in units.values():
        total = sum(mol.atom_weight[a] for a in atoms)
        value += SYNTH_SCALE_UNARY_LINEAR * float(total)
        value += SYNTH_SCALE_UNARY_SQUARE * float(total**2) / 10.0
    for k, merged in enumerate(mol.true_merge):
        if merged:
            continue
        bu, bv = mol.tree_edge[k]
        wu = sum(mol.atom_weight[a] for a in mol.block_atom_ids[bu])
        wv = sum(mol.atom_weight[a] for a in mol.block_atom_ids[bv])
        value += SYNTH_SCALE_RELATION * float(wu * wv) / 10.0
    if noise_rng is not None:
        value += float(noise_rng.normal(0.0, SYNTH_NOISE))
    return float(value)


def synthetic_dataset(
    n_molecules: int,
    *,
    seed: int,
    max_blocks: int = 6,
    noise: bool = True,
) -> list[SynthMolecule]:
    rng = np.random.default_rng(int(seed))
    noise_rng = np.random.default_rng(int(seed) + 977)
    out = []
    for i in range(int(n_molecules)):
        mol = make_synthetic_molecule(rng, i, max_blocks=max_blocks)
        if noise:
            mol.y = synthetic_y(mol, noise_rng)
        out.append(mol)
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TASUConfig:
    """Frozen widths of the task-adaptive structure-unit model."""

    atom_hidden: int = 128
    atom_out: int = 96
    bond_hidden: int = 160
    bond_out: int = 96
    unit_hidden: int = 160
    unit_out: int = 96
    relation_hidden: int = 160
    relation_out: int = 96
    scorer_hidden: int = 160
    contrib_hidden: int = 64
    #: opaque reader is capacity-matched to the two additive contribution heads
    #: (12.4k vs 12.5k params) so the readout contrast is not a capacity contrast
    reader_hidden: int = 64
    reader_layers: int = 1
    dropout: float = 0.0

    @property
    def atom_in(self) -> int:
        return sro.ATOM_FEAT_DIM

    @property
    def bond_in(self) -> int:
        return 3 * self.atom_out + sro.BOND_FEAT_DIM

    @property
    def unit_in(self) -> int:
        return 2 * self.unit_out + 2

    @property
    def relation_in(self) -> int:
        return 4 * self.unit_out

    @property
    def scorer_in(self) -> int:
        return 4 * self.unit_out

    @property
    def reader_in(self) -> int:
        return 2 * self.unit_out

    def inventory(self) -> dict[str, int]:
        def mlp(i: int, h: int, o: int) -> int:
            return i * h + h + h * o + o

        def lin(i: int, o: int) -> int:
            return i * o + o

        atom = mlp(self.atom_in, self.atom_hidden, self.atom_out)
        bond = mlp(self.bond_in, self.bond_hidden, self.bond_out)
        unit = mlp(self.unit_in, self.unit_hidden, self.unit_out)
        relation = mlp(self.relation_in, self.relation_hidden, self.relation_out)
        scorer = mlp(self.scorer_in, self.scorer_hidden, 1)
        unit_head = mlp(self.unit_out, self.contrib_hidden, 1)
        rel_head = mlp(self.relation_out, self.contrib_hidden, 1)
        reader = lin(self.reader_in, self.reader_hidden)
        for _ in range(self.reader_layers - 1):
            reader += lin(self.reader_hidden, self.reader_hidden)
        reader += lin(self.reader_hidden, 1)
        return {
            "atom_encoder": atom,
            "bond_encoder": bond,
            "unit_encoder": unit,
            "relation_encoder": relation,
            "merge_scorer": scorer,
            "unit_contrib": unit_head,
            "relation_contrib": rel_head,
            "reader": reader,
            "bias": 1,
            "total": atom + bond + unit + relation + scorer + unit_head + rel_head + reader + 1,
        }


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_hidden: int = 1, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * n_hidden + [out_dim]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
                if self.dropout > 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TaskAdaptiveStructureUnits(nn.Module):
    """One-shot atom/bond -> base blocks -> learned units -> readout model."""

    def __init__(
        self,
        config: TASUConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        *,
        partition_mode: str = PARTITION_FIXED,
        readout_mode: str = READOUT_ADDITIVE,
        merge_init_bias: float = MERGE_INIT_BIAS,
    ) -> None:
        super().__init__()
        if partition_mode not in PARTITION_MODES:
            raise ValueError(f"unknown partition_mode {partition_mode!r}")
        if readout_mode not in READOUT_MODES:
            raise ValueError(f"unknown readout_mode {readout_mode!r}")
        self.config = config
        self.partition_mode = partition_mode
        self.readout_mode = readout_mode
        self.atom_encoder = _MLP(config.atom_in, config.atom_hidden, config.atom_out, 1, config.dropout)
        self.bond_encoder = _MLP(config.bond_in, config.bond_hidden, config.bond_out, 1, config.dropout)
        self.unit_encoder = _MLP(config.unit_in, config.unit_hidden, config.unit_out, 1, config.dropout)
        self.relation_encoder = _MLP(config.relation_in, config.relation_hidden, config.relation_out, 1, config.dropout)
        self.merge_scorer = _MLP(config.scorer_in, config.scorer_hidden, 1, 1, config.dropout)
        self.unit_contrib = _MLP(config.unit_out, config.contrib_hidden, 1, 1, config.dropout)
        self.relation_contrib = _MLP(config.relation_out, config.contrib_hidden, 1, 1, config.dropout)
        self.reader = _MLP(config.reader_in, config.reader_hidden, 1, config.reader_layers, config.dropout)
        self.bias = nn.Parameter(torch.zeros(1))
        #: when set (CPU synthetic controls only), the forward uses these hard
        #: gates instead of the scalar scorer (all-merge / oracle / all-cut arms)
        self.forced_gates: torch.Tensor | None = None
        with torch.no_grad():
            last = self.merge_scorer.lins[-1]
            last.weight.zero_()
            last.bias.fill_(float(merge_init_bias))
        self.register_buffer("global_mean", torch.as_tensor(global_mean, dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(global_scale, dtype=torch.float32))

    # -- encoders -----------------------------------------------------------
    def encode_atoms(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.atom_encoder(batch["atom_feat"])

    def encode_bonds(self, a: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        feats = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        return self.bond_encoder(feats)

    # -- base block summaries (sum pooling; no message passing) --------------
    def base_block_features(
        self, a: torch.Tensor, b: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        dev = a.device
        dtype = a.dtype
        n_blocks = int(batch["blk_batch"].shape[0])
        block_of_atom = batch["block_of_atom"]
        bond_block_u, bond_block_v = batch["bond_block_u"], batch["bond_block_v"]
        internal = bond_block_u == bond_block_v
        atom_sum = torch.zeros((n_blocks, a.shape[1]), device=dev, dtype=dtype)
        atom_sum.index_add_(0, block_of_atom, a)
        atom_n = torch.zeros(n_blocks, device=dev, dtype=dtype)
        atom_n.index_add_(0, block_of_atom, torch.ones_like(block_of_atom, dtype=dtype))
        bond_sum = torch.zeros((n_blocks, b.shape[1]), device=dev, dtype=dtype)
        bond_n = torch.zeros(n_blocks, device=dev, dtype=dtype)
        if internal.any():
            bu = bond_block_u[internal]
            bond_sum.index_add_(0, bu, b[internal])
            bond_n.index_add_(0, bu, torch.ones_like(bu, dtype=dtype))
        h = torch.cat(
            [atom_sum, bond_sum, torch.log1p(atom_n).unsqueeze(1), torch.log1p(bond_n).unsqueeze(1)], dim=1
        )
        return {"h": self.unit_encoder(h), "atom_sum": atom_sum, "bond_sum": bond_sum}

    # -- merge gates ---------------------------------------------------------
    def merge_logits(
        self, h_blocks: torch.Tensor, b: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Merge logits over bridges (empty tensor when the batch has none)."""
        bridge_u, bridge_v = batch["bridge_u_local"], batch["bridge_v_local"]
        if bridge_u.numel() == 0:
            return torch.zeros(0, device=h_blocks.device, dtype=h_blocks.dtype)
        hu, hv = h_blocks[bridge_u], h_blocks[bridge_v]
        bridge_bond = batch["bridge_bond"]
        feats = torch.cat(
            [hu + hv, (hu - hv).abs(), hu * hv, b[bridge_bond]], dim=1
        )
        return self.merge_scorer(feats).view(-1)

    def straight_through_gate(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        hard = (p > 0.5).to(p.dtype)
        return hard.detach() - p.detach() + p

    def bridges_u_v(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        blk_off = batch["blk_off"]
        graph = batch["bridge_graph"]
        return (
            blk_off[graph] + batch["bridge_u_local"],
            blk_off[graph] + batch["bridge_v_local"],
        )

    # -- membership and pooling ---------------------------------------------
    def membership(
        self, gates: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Differentiable path-product membership and canonical unit selector.

        Hard forward: ``q[b, k] = 1`` iff ``k`` is the minimum-index block of
        ``b``'s merged component.  Backward: gate-aware products of the gates
        on the block-tree path.
        """
        n_graphs = int(batch["n_graphs"])
        bpad = int(batch["n_blocks_pad"])
        dev = gates.device if gates.numel() else batch["blk_batch"].device
        n_pairs = int(batch["pair_graph"].shape[0])
        if n_pairs == 0:
            raise ValueError("empty batch")
        max_len = int(batch["pair_max_len"])
        padded = torch.ones((n_pairs, max_len), device=dev, dtype=torch.float32)
        if gates.numel():
            padded[batch["pair_ids_flat"], batch["pair_entry_pos"]] = gates[batch["pair_edge_ids"]]
        p_flat = padded.prod(dim=1)
        p_dense = torch.zeros((n_graphs, bpad, bpad), device=dev, dtype=p_flat.dtype)
        flat_idx = (
            batch["pair_graph"] * (bpad * bpad)
            + batch["pair_u_local"] * bpad
            + batch["pair_v_local"]
        )
        p_dense.view(-1).index_copy_(0, flat_idx, p_flat)
        one_minus = 1.0 - p_dense
        cum = torch.cumprod(one_minus, dim=2)
        prefix = torch.ones_like(cum)
        if bpad > 1:
            prefix[:, :, 1:] = cum[:, :, :-1]
        q = p_dense * prefix
        s = q.max(dim=1).values  # [n_graphs, bpad]
        return {"P": p_dense, "q": q, "s": s}

    def _q_flat(self, q: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return q.reshape(-1)

    def unit_representations(
        self,
        q: torch.Tensor,
        s: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Pool atom/bond embeddings per unit (soft in backward, exact hard forward)."""
        dev, dtype = a.device, a.dtype
        n_graphs = int(batch["n_graphs"])
        bpad = int(batch["n_blocks_pad"])
        n_blocks = int(batch["blk_batch"].shape[0])
        q_flat = q.reshape(-1)
        k_range = torch.arange(bpad, device=dev, dtype=torch.long)

        def gather_q(graph: torch.Tensor, block_local: torch.Tensor) -> torch.Tensor:
            idx = graph[:, None] * (bpad * bpad) + block_local[:, None] * bpad + k_range[None, :]
            return q_flat[idx]

        q_atoms = gather_q(batch["atom_graph"], batch["atom_block_local"])       # [n_atoms, bpad]
        q_bu = gather_q(batch["bond_graph"], batch["bond_block_u_local"])        # [n_bonds, bpad]
        q_bv = gather_q(batch["bond_graph"], batch["bond_block_v_local"])
        q_bonds = q_bu * q_bv                                                    # internal-bond weight

        unit_of_atom = batch["blk_off"][batch["atom_graph"]][:, None] + k_range[None, :]
        unit_of_bond = batch["blk_off"][batch["bond_graph"]][:, None] + k_range[None, :]
        # padded block slots (k >= B_g) have q = 0 weights; clamp them so the
        # scatter index stays inside the global block range
        unit_of_atom = unit_of_atom.clamp(max=n_blocks - 1)
        unit_of_bond = unit_of_bond.clamp(max=n_blocks - 1)

        atom_sum = torch.zeros((n_blocks, a.shape[1]), device=dev, dtype=dtype)
        atom_n = torch.zeros(n_blocks, device=dev, dtype=dtype)
        atom_sum.index_add_(0, unit_of_atom.reshape(-1), (q_atoms.unsqueeze(2) * a.unsqueeze(1)).reshape(-1, a.shape[1]))
        atom_n.index_add_(0, unit_of_atom.reshape(-1), q_atoms.reshape(-1))
        bond_sum = torch.zeros((n_blocks, b.shape[1]), device=dev, dtype=dtype)
        bond_n = torch.zeros(n_blocks, device=dev, dtype=dtype)
        if b.shape[0]:
            bond_sum.index_add_(0, unit_of_bond.reshape(-1), (q_bonds.unsqueeze(2) * b.unsqueeze(1)).reshape(-1, b.shape[1]))
            bond_n.index_add_(0, unit_of_bond.reshape(-1), q_bonds.reshape(-1))
        h = torch.cat(
            [atom_sum, bond_sum, torch.log1p(atom_n.clamp(min=0)).unsqueeze(1), torch.log1p(bond_n.clamp(min=0)).unsqueeze(1)],
            dim=1,
        )
        u = self.unit_encoder(h)
        s_flat = s[batch["blk_batch"], batch["blk_slot"]]
        u = u * s_flat.unsqueeze(1)
        return {
            "unit_atom_sum": atom_sum,
            "unit_bond_sum": bond_sum,
            "unit_atom_n": atom_n,
            "unit_bond_n": bond_n,
            "unit_repr": u,
            "q_atoms": q_atoms,
            "q_bonds": q_bonds,
        }

    def unit_relations(
        self, u: torch.Tensor, b: torch.Tensor, q: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Relation objects for retained bridges (endpoint-symmetric)."""
        n_graphs = int(batch["n_graphs"])
        bpad = int(batch["n_blocks_pad"])
        bridge_u, bridge_v = self.bridges_u_v(batch)
        graph = batch["bridge_graph"]
        k_range = torch.arange(bpad, device=u.device, dtype=torch.long)
        q_flat = q.reshape(-1)
        u_dense = torch.zeros((n_graphs, bpad, u.shape[1]), device=u.device, dtype=u.dtype)
        u_dense.view(-1, u.shape[1]).index_copy_(
            0, batch["blk_batch"] * bpad + batch["blk_slot"], u
        )
        u_soft = torch.bmm(q, u_dense)  # [n_graphs, bpad, out]
        if bridge_u.numel() == 0:
            empty = torch.zeros(0, u.shape[1], device=u.device, dtype=u.dtype)
            return {"relation_repr": empty, "rel_weight": torch.zeros(0, device=u.device, dtype=u.dtype), "u_soft": u_soft}
        left = u_soft[graph, batch["bridge_u_local"]]
        right = u_soft[graph, batch["bridge_v_local"]]
        feats = torch.cat(
            [left + right, (left - right).abs(), left * right, b[batch["bridge_bond"]]], dim=1
        )
        rel = self.relation_encoder(feats)
        q_rows_u = q_flat[(graph * (bpad * bpad) + batch["bridge_u_local"] * bpad)[:, None] + k_range[None, :]]
        q_rows_v = q_flat[(graph * (bpad * bpad) + batch["bridge_v_local"] * bpad)[:, None] + k_range[None, :]]
        rel_weight = 1.0 - (q_rows_u * q_rows_v).sum(dim=1)
        return {"relation_repr": rel, "rel_weight": rel_weight, "u_soft": u_soft, "left": left, "right": right}

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        return_features: bool = False,
        partition_mode: str | None = None,
        readout_mode: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pmode = self.partition_mode if partition_mode is None else partition_mode
        rmode = self.readout_mode if readout_mode is None else readout_mode
        if pmode not in PARTITION_MODES or rmode not in READOUT_MODES:
            raise ValueError("unknown mode")
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        block_feats = self.base_block_features(a, b, batch)
        h_blocks = block_feats["h"]
        if pmode == PARTITION_LEARNED:
            logits = self.merge_logits(h_blocks, b, batch)
            if self.forced_gates is not None:
                gates = self.forced_gates.to(device=a.device, dtype=a.dtype)
            else:
                gates = self.straight_through_gate(logits)
        else:
            logits = torch.zeros(0, device=a.device, dtype=a.dtype)
            gates = torch.zeros(int(batch["bridge_bond"].shape[0]), device=a.device, dtype=a.dtype)
        mem = self.membership(gates, batch)
        pooled = self.unit_representations(mem["q"], mem["s"], a, b, batch)
        u = pooled["unit_repr"]
        relations = self.unit_relations(u, b, mem["q"], batch)
        n_graphs = int(batch["n_graphs"])

        def scatter_rows(x: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
            out = torch.zeros((n, x.shape[1]), device=x.device, dtype=x.dtype)
            if x.shape[0]:
                out.index_add_(0, index, x)
            return out

        z_u = scatter_rows(u, batch["blk_batch"], n_graphs)
        rel_repr = relations["relation_repr"]
        rel_weight = relations["rel_weight"]
        z_r = scatter_rows(rel_weight.unsqueeze(1) * rel_repr, batch["bridge_graph"], n_graphs)
        if rmode == READOUT_ADDITIVE:
            alpha = self.unit_contrib(u)                     # [n_blocks, 1]
            gamma = self.relation_contrib(rel_repr) * rel_weight.unsqueeze(1)
            alpha_sum = scatter_rows(alpha, batch["blk_batch"], n_graphs)
            gamma_sum = scatter_rows(gamma, batch["bridge_graph"], n_graphs)
            pred = (self.bias + alpha_sum + gamma_sum).view(-1)
        else:
            reader_in = torch.cat([z_u, z_r], dim=1)
            pred = self.reader(reader_in).view(-1)
        feats = {
            "a": a,
            "b": b,
            "h_blocks": h_blocks,
            "merge_logits": logits,
            "gates": gates,
            "P": mem["P"],
            "q": mem["q"],
            "s": mem["s"],
            "unit_repr": u,
            "unit_atom_sum": pooled["unit_atom_sum"],
            "unit_bond_sum": pooled["unit_bond_sum"],
            "unit_atom_n": pooled["unit_atom_n"],
            "unit_bond_n": pooled["unit_bond_n"],
            "relation_repr": rel_repr,
            "rel_weight": rel_weight,
            "z_u": z_u,
            "z_r": z_r,
        }
        if rmode == READOUT_ADDITIVE:
            feats.update({"alpha": alpha, "gamma": gamma})
        if return_features:
            return pred, feats
        return pred


# ---------------------------------------------------------------------------
# builders / audits
# ---------------------------------------------------------------------------


def build_model(
    config: TASUConfig,
    *,
    seed: int,
    partition_mode: str,
    readout_mode: str,
    global_mean: np.ndarray,
    global_scale: np.ndarray,
    device: torch.device | str = "cpu",
) -> TaskAdaptiveStructureUnits:
    """Deterministic construction; the four arms share identical parameters."""
    torch.manual_seed(int(seed))
    model = TaskAdaptiveStructureUnits(
        config,
        np.asarray(global_mean, dtype=np.float32),
        np.asarray(global_scale, dtype=np.float32),
        partition_mode=partition_mode,
        readout_mode=readout_mode,
    )
    return model.to(device)


def load_ssl_encoders(
    model: TaskAdaptiveStructureUnits,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device | str = "cpu",
) -> None:
    """Load only the Atom/Bond encoders from a masked-SSL checkpoint state."""
    model.atom_encoder.load_state_dict({k: v.to(device) for k, v in ssl_state["atom_encoder"].items()})
    model.bond_encoder.load_state_dict({k: v.to(device) for k, v in ssl_state["bond_encoder"].items()})


def parameter_audit(config: TASUConfig, model: nn.Module) -> dict[str, Any]:
    expected = config.inventory()
    actual = int(sum(p.numel() for p in model.parameters()))
    named = {
        "atom_encoder": sum(p.numel() for p in model.atom_encoder.parameters()),
        "bond_encoder": sum(p.numel() for p in model.bond_encoder.parameters()),
        "unit_encoder": sum(p.numel() for p in model.unit_encoder.parameters()),
        "relation_encoder": sum(p.numel() for p in model.relation_encoder.parameters()),
        "merge_scorer": sum(p.numel() for p in model.merge_scorer.parameters()),
        "unit_contrib": sum(p.numel() for p in model.unit_contrib.parameters()),
        "relation_contrib": sum(p.numel() for p in model.relation_contrib.parameters()),
        "reader": sum(p.numel() for p in model.reader.parameters()),
        "bias": int(model.bias.numel()),
    }
    return {
        "expected": expected,
        "named": named,
        "actual_parameters": actual,
        "parameter_exact": actual == expected["total"] == sum(named.values()),
    }


def state_dict_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.endswith("global_mean") or key.endswith("global_scale"):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def encoder_group_hash(model: TaskAdaptiveStructureUnits) -> str:
    h = hashlib.sha256()
    for name in ("atom_encoder", "bond_encoder"):
        module = getattr(model, name)
        for k, v in sorted(module.state_dict().items()):
            h.update(f"{name}.{k}".encode())
            h.update(np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def other_group_hash(model: TaskAdaptiveStructureUnits) -> str:
    h = hashlib.sha256()
    for name in ("unit_encoder", "relation_encoder", "merge_scorer", "unit_contrib", "relation_contrib", "reader"):
        module = getattr(model, name)
        for k, v in sorted(module.state_dict().items()):
            h.update(f"{name}.{k}".encode())
            h.update(np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(model.bias.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def forbidden_module_scan(model: nn.Module) -> list[str]:
    found = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        for token in FORBIDDEN_TOKENS:
            if token in cls:
                found.append(f"{name}:{cls}")
    return found


# ---------------------------------------------------------------------------
# decomposition and partition utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_batch(
    model: TaskAdaptiveStructureUnits,
    batch: Mapping[str, torch.Tensor],
    *,
    partition_mode: str | None = None,
    readout_mode: str | None = None,
) -> np.ndarray:
    model.eval()
    pred = model(batch, partition_mode=partition_mode, readout_mode=readout_mode)
    return pred.detach().cpu().numpy()


def hard_partition_gates(model: TaskAdaptiveStructureUnits, batch: Mapping[str, torch.Tensor]) -> np.ndarray:
    """Return hard merge decisions for the batch (True = merge)."""
    if model.partition_mode != PARTITION_LEARNED:
        n = int(batch["bridge_bond"].shape[0])
        return np.zeros(n, dtype=bool)
    model.eval()
    with torch.no_grad():
        a = model.encode_atoms(batch)
        b = model.encode_bonds(a, batch)
        h = model.base_block_features(a, b, batch)["h"]
        logits = model.merge_logits(h, b, batch)
    return (torch.sigmoid(logits) > 0.5).detach().cpu().numpy()


def unit_assignment_from_gates(topo: StructureTopology, merge: np.ndarray) -> np.ndarray:
    """Map each atom to its final unit representative (min block id)."""
    reps, _ = topo.hard_components(np.asarray(merge, dtype=bool))
    return reps[topo.block_of_atom]


# ---------------------------------------------------------------------------
# synthetic training loops (CPU feasibility: S1 and S2)
# ---------------------------------------------------------------------------


def synthetic_objects_topos(
    mols: Sequence[SynthMolecule],
) -> tuple[list[sro.MolObjects], list[StructureTopology], np.ndarray]:
    objs: list[sro.MolObjects] = []
    topos: list[StructureTopology] = []
    y = np.zeros(len(mols), dtype=np.float64)
    for i, mol in enumerate(mols):
        x, edge_index, edge_attr = synth_raw_arrays(mol)
        obj = sro.build_mol_objects(mol.mol_index, x, edge_index, edge_attr)
        objs.append(obj)
        topos.append(build_structure_topology(mol.mol_index, mol.bonds))
        y[i] = mol.y
    return objs, topos, y


def synthetic_merge_labels(mols: Sequence[SynthMolecule]) -> np.ndarray:
    out = []
    for mol in mols:
        out.extend([1.0 if v else 0.0 for v in mol.true_merge])
    return np.asarray(out, dtype=np.float32)


def synthetic_batch(
    objs: Sequence[sro.MolObjects], topos: Sequence[StructureTopology]
) -> dict[str, torch.Tensor]:
    batch = batch_atom_bond(objs)
    batch.update(batch_structure_topology(objs, topos))
    return batch


def train_synthetic_s1(
    *,
    seed: int,
    train_mols: Sequence[SynthMolecule],
    test_mols: Sequence[SynthMolecule],
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    global_mean: np.ndarray | None = None,
    global_scale: np.ndarray | None = None,
) -> dict[str, Any]:
    """S1: direct bridge-boundary supervision (merge/cut labels only)."""
    dev = torch.device(device)
    train_objs, train_topos, _ = synthetic_objects_topos(list(train_mols))
    test_objs, test_topos, _ = synthetic_objects_topos(list(test_mols))
    config = TASUConfig()
    if global_mean is None:
        global_mean = np.zeros(sro.GLOBAL_DIM, dtype=np.float32)
    if global_scale is None:
        global_scale = np.ones(sro.GLOBAL_DIM, dtype=np.float32)
    model = build_model(
        config,
        seed=seed,
        partition_mode=PARTITION_LEARNED,
        readout_mode=READOUT_ADDITIVE,
        global_mean=global_mean,
        global_scale=global_scale,
        device=dev,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    gen = torch.Generator().manual_seed(int(seed) + 17)
    order = torch.randperm(len(train_mols), generator=gen).tolist()
    trace = []
    gate_grad_norms = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total, n_seen = 0.0, 0
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch = synthetic_batch([train_objs[i] for i in idx], [train_topos[i] for i in idx])
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            yb = torch.as_tensor(
                np.concatenate([np.asarray(train_mols[i].true_merge, dtype=np.float32) for i in idx]),
                dtype=torch.float32,
                device=dev,
            )
            a = model.encode_atoms(batch)
            b_enc = model.encode_bonds(a, batch)
            h = model.base_block_features(a, b_enc, batch)["h"]
            logits = model.merge_logits(h, b_enc, batch)
            if logits.numel() == 0:
                loss = torch.zeros((), device=dev)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            if model.merge_scorer.lins[-1].weight.grad is not None:
                gate_grad_norms.append(float(model.merge_scorer.lins[-1].weight.grad.norm()))
            opt.step()
            total += float(loss) * len(idx)
            n_seen += len(idx)
        stats = synthetic_partition_stats(model, train_objs, train_topos, train_mols, dev)
        trace.append({"epoch": epoch, "bce": total / max(n_seen, 1), **stats})
    final = synthetic_partition_stats(model, test_objs, test_topos, test_mols, dev)
    return {
        "protocol": PROTOCOL_ID,
        "test": "S1_direct_boundary_supervision",
        "epochs": int(epochs),
        "trace": trace,
        "test_stats": final,
        "gate_grad_norm_last": gate_grad_norms[-1] if gate_grad_norms else None,
        "gate_grad_norm_first": gate_grad_norms[0] if gate_grad_norms else None,
        "model": model,
    }


def synthetic_partition_stats(
    model: TaskAdaptiveStructureUnits,
    objs: Sequence[sro.MolObjects],
    topos: Sequence[StructureTopology],
    mols: Sequence[SynthMolecule],
    device: torch.device | str,
) -> dict[str, Any]:
    """Hard-partition agreement vs the synthetic oracle."""
    dev = torch.device(device)
    correct_bridges = total_bridges = 0
    exact_molecules = 0
    unit_counts = []
    for i in range(0, len(objs), 32):
        chunk_objs = list(objs[i : i + 32])
        chunk_topos = list(topos[i : i + 32])
        batch = synthetic_batch(chunk_objs, chunk_topos)
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        merge = hard_partition_gates(model, batch)
        off = 0
        for j, topo in enumerate(chunk_topos):
            n_b = len(topo.bridge_bond)
            pred_m = merge[off : off + n_b]
            off += n_b
            truth = np.asarray(mols[i + j].true_merge, dtype=bool)
            correct_bridges += int((pred_m == truth).sum())
            total_bridges += int(n_b)
            reps, n_units = topo.hard_components(pred_m)
            unit_counts.append(int(n_units))
            # exact partition agreement: atom->unit partition equal as a set partition
            oracle = np.asarray(mols[i + j].unit_of_atom, dtype=np.int64)
            pred_atoms = reps[topo.block_of_atom]
            if set(oracle.tolist()) == set(pred_atoms.tolist()) and all(
                (oracle == oracle[k])[pred_atoms == pred_atoms[k]].all() for k in range(len(oracle))
            ):
                exact_molecules += 1
    return {
        "bridge_accuracy": correct_bridges / max(total_bridges, 1),
        "exact_partition_accuracy": exact_molecules / max(len(mols), 1),
        "mean_units": float(np.mean(unit_counts)) if unit_counts else 0.0,
    }


def forced_gates_for_batch(
    mols: Sequence[SynthMolecule], mode: str
) -> np.ndarray:
    """Hard gates for the synthetic controls: 'none' (fixed), 'all', 'oracle'."""
    out: list[float] = []
    for mol in mols:
        n = len(mol.true_merge)
        if mode == "none":
            out.extend([0.0] * n)
        elif mode == "all":
            out.extend([1.0] * n)
        elif mode == "oracle":
            out.extend([1.0 if v else 0.0 for v in mol.true_merge])
        else:
            raise ValueError(f"unknown forced mode {mode!r}")
    return np.asarray(out, dtype=np.float32)


def train_synthetic_s2(
    *,
    seed: int,
    train_mols: Sequence[SynthMolecule],
    test_mols: Sequence[SynthMolecule],
    partition_mode: str,
    readout_mode: str = READOUT_ADDITIVE,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    forced: str | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """S2: y-only synthetic task; compare fixed vs learned partition.

    ``forced`` (CPU controls): ``None`` uses the arm's scorer; ``"none"``,
    ``"all"`` and ``"oracle"`` force the hard partition to all-cut, all-merge
    and the planted oracle without training or using the scorer.
    """
    dev = torch.device(device)
    train_objs, train_topos, train_y = synthetic_objects_topos(list(train_mols))
    test_objs, test_topos, test_y = synthetic_objects_topos(list(test_mols))
    config = TASUConfig()
    global_mean = np.zeros(sro.GLOBAL_DIM, dtype=np.float32)
    global_scale = np.ones(sro.GLOBAL_DIM, dtype=np.float32)
    model = build_model(
        config,
        seed=seed,
        partition_mode=partition_mode,
        readout_mode=readout_mode,
        global_mean=global_mean,
        global_scale=global_scale,
        device=dev,
    )

    def set_forced(mols: Sequence[SynthMolecule]) -> None:
        if forced is None:
            model.forced_gates = None
        else:
            model.forced_gates = torch.as_tensor(forced_gates_for_batch(mols, forced), dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    gen = torch.Generator().manual_seed(int(seed) + 31)
    curve = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(len(train_mols), generator=gen).tolist()
        total, n_seen = 0.0, 0
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            chunk_mols = [train_mols[i] for i in idx]
            batch = synthetic_batch([train_objs[i] for i in idx], [train_topos[i] for i in idx])
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            set_forced(chunk_mols)
            yb = torch.as_tensor(train_y[idx], dtype=torch.float32, device=dev)
            pred = model(batch)
            loss = F.l1_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
            n_seen += len(idx)
        curve.append({"epoch": epoch, "train_mae": total / max(n_seen, 1)})
    model.eval()
    with torch.no_grad():
        test_pred = []
        for i in range(0, len(test_objs), 64):
            chunk_mols = list(test_mols[i : i + 64])
            batch = synthetic_batch(list(test_objs[i : i + 64]), list(test_topos[i : i + 64]))
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            set_forced(chunk_mols)
            test_pred.append(model(batch).detach().cpu().numpy())
        test_pred = np.concatenate(test_pred, 0) if test_pred else np.zeros(0)
    model.forced_gates = None
    stats = synthetic_partition_stats(model, test_objs, test_topos, test_mols, dev)
    oracle = synthetic_oracle_contributions(test_mols)
    return {
        "protocol": PROTOCOL_ID,
        "test": "S2_y_only",
        "partition_mode": partition_mode,
        "readout_mode": readout_mode,
        "forced": forced,
        "epochs": int(epochs),
        "curve": curve,
        "test_mae": float(np.abs(test_pred - test_y).mean()) if len(test_y) else None,
        "test_pred": test_pred.tolist(),
        "test_y": test_y.tolist(),
        "partition_stats": stats,
        "oracle": oracle,
        "model": model,
    }


def synthetic_oracle_contributions(mols: Sequence[SynthMolecule]) -> dict[str, np.ndarray]:
    """Oracle unit/relation contributions (before the model sees them)."""
    per_mol_y = np.asarray([m.y for m in mols], dtype=np.float64)
    unit_scale = np.asarray([SYNTH_SCALE_UNARY_LINEAR for _ in mols], dtype=np.float64)
    return {"y": per_mol_y, "unit_scale": unit_scale, "bias": np.asarray([SYNTH_BIAS], dtype=np.float64)}


# ---------------------------------------------------------------------------
# supervised training (ZINC split D; recipe R3 frozen before GPU)
# ---------------------------------------------------------------------------

RECIPE_R3: dict[str, Any] = {
    "epochs": 360,
    "schedule": "cosine",
    "lr": 1.0e-3,
    "lr_min": 1.0e-5,
    "weight_decay": 1.0e-5,
    "batch_size": 128,
    "grad_clip": 5.0,
    "soup_last": 5,
    "shuffle_offset": 101,
}


def schedule_lr(recipe: Mapping[str, Any], step: int, total: int) -> float:
    if recipe["schedule"] == "constant":
        return float(recipe["lr"])
    if recipe["schedule"] == "cosine":
        import math

        return float(recipe["lr_min"] + 0.5 * (recipe["lr"] - recipe["lr_min"]) * (1.0 + math.cos(math.pi * step / max(total, 1))))
    raise ValueError(f"unknown schedule {recipe['schedule']!r}")


def shuffle_batches(n: int, batch_size: int, generator: torch.Generator) -> list[list[int]]:
    order = torch.randperm(n, generator=generator).tolist()
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def batch_order_hash(seed: int, n: int, batch_size: int, epochs: int) -> str:
    gen = torch.Generator().manual_seed(int(seed) + RECIPE_R3["shuffle_offset"])
    h = hashlib.sha256()
    for _ in range(int(epochs)):
        for batch in shuffle_batches(n, batch_size, gen):
            h.update(np.ascontiguousarray(np.asarray(batch, dtype=np.int64)).tobytes())
    return h.hexdigest()


@torch.no_grad()
def partition_probe_stats(
    model: TaskAdaptiveStructureUnits,
    objs: Sequence[sro.MolObjects],
    topos: Sequence[StructureTopology],
    device: torch.device | str = "cpu",
    chunk: int = 128,
) -> dict[str, Any]:
    """Label-blind hard-partition statistics of the current scorer state."""
    dev = torch.device(device)
    was_training = model.training
    model.eval()
    merged = []
    n_bridges = []
    logits_all = []
    units = []
    n_blocks_total = 0
    for i in range(0, len(objs), int(chunk)):
        batch = batch_atom_bond(list(objs[i : i + int(chunk)]))
        batch.update(batch_structure_topology(list(objs[i : i + int(chunk)]), list(topos[i : i + int(chunk)])))
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        a = model.encode_atoms(batch)
        b = model.encode_bonds(a, batch)
        h = model.base_block_features(a, b, batch)["h"]
        logits = model.merge_logits(h, b, batch)
        logits_all.append(logits.detach().cpu().numpy())
        hard = (torch.sigmoid(logits) > 0.5).detach().cpu().numpy() if logits.numel() else np.zeros(0, dtype=bool)
        off = 0
        for j, topo in enumerate(topos[i : i + int(chunk)]):
            nb = len(topo.bridge_bond)
            m = hard[off : off + nb]
            off += nb
            if nb:
                merged.append(float(m.mean()))
                n_bridges.append(nb)
            reps, nu = topo.hard_components(m)
            units.append(int(nu))
            n_blocks_total += int(topo.n_blocks)
    if was_training:
        model.train()
    logits_cat = np.concatenate(logits_all, 0) if logits_all else np.zeros(0)
    return {
        "n_molecules": int(len(objs)),
        "n_bridges": int(sum(n_bridges)),
        "n_blocks": int(n_blocks_total),
        "mean_merge_fraction": float(np.mean(merged)) if merged else 0.0,
        "frac_bridges_merged": float(np.concatenate([np.full(1, m) for m in merged]).mean()) if merged else 0.0,
        "mean_units": float(np.mean(units)) if units else 0.0,
        "logit_mean": float(logits_cat.mean()) if logits_cat.size else 0.0,
        "logit_std": float(logits_cat.std()) if logits_cat.size else 0.0,
        "prob_mean": float(torch.sigmoid(torch.as_tensor(logits_cat)).mean()) if logits_cat.size else 0.0,
        "prob_lt_0p5": float((torch.sigmoid(torch.as_tensor(logits_cat)) <= 0.5).float().mean()) if logits_cat.size else 0.0,
    }


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def last_soup(states: Sequence[Mapping[str, torch.Tensor]], soup_last: int) -> dict[str, torch.Tensor]:
    members = list(states)[-int(soup_last) :]
    keys = sorted(members[0].keys())
    soup = {
        k: torch.stack([m[k].float() for m in members], 0).mean(0) for k in keys
    }
    return soup


def train_supervised_tasu(
    *,
    arm: str,
    seed: int,
    train_objs: Sequence[sro.MolObjects],
    train_topos: Sequence[StructureTopology],
    y_fit: np.ndarray,
    device: torch.device | str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    recipe: Mapping[str, Any] | None = None,
    probe_objs: Sequence[sro.MolObjects] | None = None,
    probe_topos: Sequence[StructureTopology] | None = None,
    probe_every: int = 20,
    epochs: int | None = None,
    log_fn: Any = print,
) -> tuple[TaskAdaptiveStructureUnits, dict[str, Any]]:
    """One frozen-R3 supervised trajectory of one arm.

    All four arms share seed, initialisation, batch order and recipe; only
    ``partition_mode``/``readout_mode`` differ.  The scorer receives the
    straight-through surrogate gradient from the raw-y L1 loss.
    """
    rec = dict(RECIPE_R3 if recipe is None else recipe)
    n_epochs = int(rec["epochs"] if epochs is None else epochs)
    batch_size = int(rec["batch_size"])
    part_mode, read_mode = arm_modes(arm)
    dev = torch.device(device)
    config = TASUConfig()
    model = build_model(
        config,
        seed=seed,
        partition_mode=part_mode,
        readout_mode=read_mode,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
        device=dev,
    )
    init_full_sha = state_dict_hash(model)
    init_encoder_sha = encoder_group_hash(model)
    init_other_sha = other_group_hash(model)
    if ssl_state is not None:
        load_ssl_encoders(model, ssl_state, dev)
    encoder_used_sha = encoder_group_hash(model)
    per_step_lr = rec["schedule"] == "cosine"
    n = int(len(train_objs))
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_updates = n_epochs * steps_per_epoch
    opt = torch.optim.Adam(model.parameters(), lr=float(rec["lr"]), weight_decay=float(rec["weight_decay"]))
    gen = torch.Generator().manual_seed(int(seed) + int(rec["shuffle_offset"]))
    states: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    probe_trace: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    gate_grad_norms: list[float] = []
    import time as _time

    started = _time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        total, n_seen, steps = 0.0, 0, 0
        t_ep = _time.perf_counter()
        for indices in shuffle_batches(n, batch_size, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = batch_atom_bond([train_objs[i] for i in indices])
            batch.update(batch_structure_topology([train_objs[i] for i in indices], [train_topos[i] for i in indices]))
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            yb = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=dev)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), yb.view(-1))
            if per_step_lr:
                lr = schedule_lr(rec, steps + (epoch - 1) * steps_per_epoch, total_updates)
                for group in opt.param_groups:
                    group["lr"] = lr
            opt.zero_grad()
            loss.backward()
            if part_mode == PARTITION_LEARNED and model.merge_scorer.lins[-1].weight.grad is not None:
                gate_grad_norms.append(float(model.merge_scorer.lins[-1].weight.grad.detach().norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(rec["grad_clip"]))
            opt.step()
            total += float((pred.view(-1) - yb.view(-1)).abs().sum())
            n_seen += len(indices)
            steps += 1
        states.append(state_dict_to_cpu(model))
        curve.append(
            {
                "epoch": epoch,
                "train_mae": total / max(n_seen, 1),
                "lr": float(opt.param_groups[0]["lr"]),
                "seconds": _time.perf_counter() - t_ep,
            }
        )
        if probe_objs is not None and (epoch % int(probe_every) == 0 or epoch == 1 or epoch == n_epochs):
            probe = partition_probe_stats(model, probe_objs, probe_topos or [], device=dev)
            probe_trace.append({"epoch": epoch, **probe})
        if epoch % 20 == 0 or epoch == 1 or epoch == n_epochs:
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} lr={curve[-1]['lr']:.3e} {curve[-1]['seconds']:.1f}s")
    wall = _time.perf_counter() - started
    soup = last_soup(states, int(rec["soup_last"]))
    model.load_state_dict({k: v.to(dev) for k, v in soup.items()})
    model.eval()
    with torch.no_grad():
        fit_pred = []
        for i in range(0, n, 256):
            batch = batch_atom_bond(list(train_objs[i : i + 256]))
            batch.update(batch_structure_topology(list(train_objs[i : i + 256]), list(train_topos[i : i + 256])))
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            fit_pred.append(model(batch).detach().cpu().numpy())
        fit_pred = np.concatenate(fit_pred, 0) if fit_pred else np.zeros(0)
    info = {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "partition_mode": part_mode,
        "readout_mode": read_mode,
        "seed": int(seed),
        "recipe": dict(rec),
        "epochs": n_epochs,
        "updates": int(total_updates),
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {rec['soup_last']} epoch states",
        "init_full_sha256": init_full_sha,
        "init_encoder_sha256": init_encoder_sha,
        "init_other_sha256": init_other_sha,
        "encoder_used_sha256": encoder_used_sha,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "batch_order_hash": order_hash.hexdigest(),
        "batch_order_hash_independent": batch_order_hash(seed, n, batch_size, n_epochs),
        "curve": curve,
        "probe_trace": probe_trace,
        "gate_grad_norm_first": gate_grad_norms[0] if gate_grad_norms else None,
        "gate_grad_norm_last": gate_grad_norms[-1] if gate_grad_norms else None,
        "gate_grad_norm_mean": float(np.mean(gate_grad_norms)) if gate_grad_norms else None,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()) if len(fit_pred) else None,
        "wall_seconds": wall,
        "device": str(dev),
        "labels_used": "Fit rows raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    return model, info
