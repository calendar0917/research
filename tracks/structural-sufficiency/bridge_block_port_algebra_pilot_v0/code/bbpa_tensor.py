"""Batched tensor assembly for the BBPA decomposition.

The expensive graph work (bridges, blocks, ports, distances, DP schedule) is a
pure CPU function of one molecule and is precomputed once per row.  A batch
collate only concatenates molecule-local arrays with offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import bbpa_structure as S


@dataclass
class MoleculeTensors:
    row: int
    n_atoms: int
    n_bonds: int
    n_blocks: int
    n_bridges: int
    n_ports: int
    n_edges: int
    # blocks
    block_atoms: np.ndarray          # local atom ids, flat
    block_atom_ptr: np.ndarray       # n_blocks+1
    block_bonds: np.ndarray          # local bond ids, flat
    block_bond_ptr: np.ndarray       # n_blocks+1
    block_bond_u: np.ndarray         # local atom ids, flat
    block_bond_v: np.ndarray
    block_n_ports: np.ndarray
    block_cycle_rank: np.ndarray
    # ports
    port_block: np.ndarray
    port_atom: np.ndarray            # local atom id
    port_bridge: np.ndarray
    port_atom_degree: np.ndarray
    port_atom_n_ports: np.ndarray
    port_bucket_atom_atom: np.ndarray
    port_bucket_atom_seg: np.ndarray  # port*6+bucket
    port_bucket_bond_bond: np.ndarray
    port_bucket_bond_seg: np.ndarray
    port_bucket_atom_count: np.ndarray  # [P*6]
    port_bucket_bond_count: np.ndarray  # [P*6]
    # bridges
    bridge_bond: np.ndarray
    bridge_block_u: np.ndarray
    bridge_block_v: np.ndarray
    bridge_port_u: np.ndarray
    bridge_port_v: np.ndarray
    # directed edges and dependencies
    edge_step: np.ndarray
    edge_source: np.ndarray
    edge_target: np.ndarray
    edge_port: np.ndarray
    edge_bridge: np.ndarray
    dep_edge: np.ndarray
    dep_port: np.ndarray
    dep_bridge: np.ndarray
    dep_port_feat: np.ndarray
    dep_dist_bucket: np.ndarray
    dep_target_edge: np.ndarray
    max_step: int
    # per-molecule segment helpers
    block_seg: np.ndarray            # block -> molecule index (inside this molecule all same)
    bridge_seg: np.ndarray


def _port_bucket_stats(decomp: S.Decomp) -> tuple[np.ndarray, ...]:
    """Distance-bucketed atom/bond occurrences for every port.

    Bucket 0..4 are exact graph distances; bucket 5 is ``5+``.  A bond is put
    in the bucket of its closer endpoint (``min(d_u, d_v)``), which is a
    deterministic function of the rooted block graph.
    """
    n_ports = len(decomp.ports)
    atom_ids: list[int] = []
    atom_segs: list[int] = []
    bond_ids: list[int] = []
    bond_segs: list[int] = []
    atom_counts = np.zeros(n_ports * 6, np.int64)
    bond_counts = np.zeros(n_ports * 6, np.int64)
    for port in decomp.ports:
        b = decomp.blocks[port.block]
        atom_to_local = {int(a): i for i, a in enumerate(b.atoms.tolist())}
        bu_local = np.asarray([atom_to_local[int(x)] for x in decomp.bu[b.bonds].tolist()],
                              np.int64)
        bv_local = np.asarray([atom_to_local[int(x)] for x in decomp.bv[b.bonds].tolist()],
                              np.int64)
        dist = S.bfs_dist(len(b.atoms), bu_local, bv_local, atom_to_local[int(port.atom)])
        if np.any(dist < 0):
            raise AssertionError("block internal graph is disconnected")
        buckets = np.minimum(dist, 5)
        for local_idx, bucket in enumerate(buckets.tolist()):
            atom_ids.append(int(b.atoms[local_idx]))
            atom_segs.append(port.port_id * 6 + int(bucket))
            atom_counts[port.port_id * 6 + int(bucket)] += 1
        # bond bucket = closer endpoint's distance bucket
        bdist = np.minimum(buckets[bu_local], buckets[bv_local])
        for local_idx, bucket in enumerate(bdist.tolist()):
            bond_ids.append(int(b.bonds[local_idx]))
            bond_segs.append(port.port_id * 6 + int(bucket))
            bond_counts[port.port_id * 6 + int(bucket)] += 1
    return (np.asarray(atom_ids, np.int64), np.asarray(atom_segs, np.int64),
            np.asarray(bond_ids, np.int64), np.asarray(bond_segs, np.int64),
            atom_counts, bond_counts)


def build_molecule_tensors(decomp: S.Decomp, row: int) -> MoleculeTensors:
    n_blocks = len(decomp.blocks)
    n_bridges = len(decomp.bridges)
    n_ports = len(decomp.ports)
    n_edges = n_bridges * 2

    block_atoms = np.concatenate([b.atoms for b in decomp.blocks]) if n_blocks else np.zeros(0, np.int64)
    block_atom_ptr = np.zeros(n_blocks + 1, np.int64)
    block_atom_ptr[1:] = np.cumsum([len(b.atoms) for b in decomp.blocks])
    block_bonds = np.concatenate([b.bonds for b in decomp.blocks]) if n_blocks else np.zeros(0, np.int64)
    block_bond_ptr = np.zeros(n_blocks + 1, np.int64)
    block_bond_ptr[1:] = np.cumsum([len(b.bonds) for b in decomp.blocks])
    block_bond_u = np.concatenate([decomp.bu[b.bonds] for b in decomp.blocks]) if n_blocks else np.zeros(0, np.int64)
    block_bond_v = np.concatenate([decomp.bv[b.bonds] for b in decomp.blocks]) if n_blocks else np.zeros(0, np.int64)
    block_n_ports = np.asarray([len(b.ports) for b in decomp.blocks], np.int64)
    block_cycle_rank = np.asarray([max(0, len(b.bonds) - len(b.atoms) + 1)
                                   for b in decomp.blocks], np.int64)

    port_block = np.asarray([p.block for p in decomp.ports], np.int64)
    port_atom = np.asarray([p.atom for p in decomp.ports], np.int64)
    port_bridge = np.asarray([p.bridge for p in decomp.ports], np.int64)
    port_atom_degree = np.zeros(n_ports, np.int64)
    port_atom_n_ports = np.zeros(n_ports, np.int64)
    for port in decomp.ports:
        b = decomp.blocks[port.block]
        atom = int(port.atom)
        port_atom_degree[port.port_id] = int(np.sum((decomp.bu[b.bonds] == atom) |
                                                    (decomp.bv[b.bonds] == atom)))
        port_atom_n_ports[port.port_id] = sum(1 for q in b.ports
                                              if int(decomp.ports[q].atom) == atom)
    atom_ids, atom_segs, bond_ids, bond_segs, atom_counts, bond_counts = _port_bucket_stats(decomp)

    bridge_bond = np.asarray([b.bond for b in decomp.bridges], np.int64)
    bridge_block_u = np.asarray([b.block_u for b in decomp.bridges], np.int64)
    bridge_block_v = np.asarray([b.block_v for b in decomp.bridges], np.int64)
    bridge_port_u = np.asarray([b.port_u for b in decomp.bridges], np.int64)
    bridge_port_v = np.asarray([b.port_v for b in decomp.bridges], np.int64)

    dep_edge: list[int] = []
    dep_port: list[int] = []
    dep_bridge: list[int] = []
    dep_port_feat: list[int] = []
    dep_dist: list[int] = []
    dep_target: list[int] = []
    for e, deps in enumerate(decomp.edge_deps):
        for d in deps:
            dep_edge.append(int(d["dep_edge"]))
            dep_port.append(int(d["dep_port"]))
            dep_bridge.append(int(d["bridge"]))
            dep_port_feat.append(int(d["port"]))
            dep_dist.append(int(d["dist_bucket"]))
            dep_target.append(e)

    return MoleculeTensors(
        row=int(row), n_atoms=int(decomp.n_atoms), n_bonds=int(decomp.n_bonds),
        n_blocks=int(n_blocks), n_bridges=int(n_bridges), n_ports=int(n_ports),
        n_edges=int(n_edges),
        block_atoms=block_atoms, block_atom_ptr=block_atom_ptr,
        block_bonds=block_bonds, block_bond_ptr=block_bond_ptr,
        block_bond_u=block_bond_u, block_bond_v=block_bond_v,
        block_n_ports=block_n_ports, block_cycle_rank=block_cycle_rank,
        port_block=port_block, port_atom=port_atom, port_bridge=port_bridge,
        port_atom_degree=port_atom_degree, port_atom_n_ports=port_atom_n_ports,
        port_bucket_atom_atom=atom_ids.astype(np.int64),
        port_bucket_atom_seg=atom_segs.astype(np.int64),
        port_bucket_bond_bond=bond_ids.astype(np.int64),
        port_bucket_bond_seg=bond_segs.astype(np.int64),
        port_bucket_atom_count=atom_counts, port_bucket_bond_count=bond_counts,
        bridge_bond=bridge_bond, bridge_block_u=bridge_block_u,
        bridge_block_v=bridge_block_v, bridge_port_u=bridge_port_u,
        bridge_port_v=bridge_port_v,
        edge_step=np.asarray(decomp.edge_step, np.int64),
        edge_source=np.asarray(decomp.edge_source, np.int64),
        edge_target=np.asarray(decomp.edge_target, np.int64),
        edge_port=np.asarray(decomp.edge_port, np.int64),
        edge_bridge=np.asarray(decomp.edge_bridge, np.int64),
        dep_edge=np.asarray(dep_edge, np.int64),
        dep_port=np.asarray(dep_port, np.int64),
        dep_bridge=np.asarray(dep_bridge, np.int64),
        dep_port_feat=np.asarray(dep_port_feat, np.int64),
        dep_dist_bucket=np.asarray(dep_dist, np.int64),
        dep_target_edge=np.asarray(dep_target, np.int64),
        max_step=int(decomp.max_step),
        block_seg=np.zeros(n_blocks, np.int64),
        bridge_seg=np.zeros(n_bridges, np.int64),
    )


def build_dataset(raw: Mapping[str, Any], rows: Sequence[int]) -> dict[int, MoleculeTensors]:
    from bbpa_data import molecule_view

    out: dict[int, MoleculeTensors] = {}
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    bu_all = np.asarray(raw["bond_u"], np.int64)
    bv_all = np.asarray(raw["bond_v"], np.int64)
    for row in np.asarray(rows, np.int64).tolist():
        view = molecule_view(raw, row)
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        d = S.decompose(int(view["n_atoms"]), bu_all[b0:b1], bv_all[b0:b1])
        out[int(row)] = build_molecule_tensors(d, int(row))
    return out


def _cat(arrays: Sequence[np.ndarray], dtype=np.int64) -> np.ndarray:
    if not arrays:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(arrays).astype(dtype, copy=False)


def collate(rows: Sequence[int], mols: Mapping[int, MoleculeTensors],
            raw: Mapping[str, Any], device: Any = "cpu") -> dict[str, torch.Tensor]:
    rows = [int(r) for r in rows]
    if not rows:
        raise ValueError("empty batch")
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_start = node_ptr[np.asarray(rows, np.int64)]
    bond_start = bond_ptr[np.asarray(rows, np.int64)]

    atom_parts, bond_parts = [], []
    block_atom_idx, block_atom_seg = [], []
    block_bond_idx, block_bond_seg = [], []
    block_bond_u, block_bond_v = [], []
    block_n_atoms, block_n_bonds, block_n_ports, block_cycle_rank = [], [], [], []
    block_seg = []
    port_block, port_atom_idx, port_atom_degree, port_atom_n_ports = [], [], [], []
    port_bucket_atom_idx, port_bucket_atom_seg = [], []
    port_bucket_bond_idx, port_bucket_bond_seg = [], []
    port_bucket_atom_count, port_bucket_bond_count = [], []
    bridge_bond, bridge_block_u, bridge_block_v, bridge_port_u, bridge_port_v = [], [], [], [], []
    bridge_seg = []
    edge_step, edge_source, edge_target, edge_port, edge_bridge = [], [], [], [], []
    dep_edge, dep_port, dep_bridge, dep_port_feat, dep_dist, dep_target = [], [], [], [], [], []
    atom_mol, bond_mol = [], []
    mol_block_ptr = [0]
    mol_bridge_ptr = [0]
    atom_off = 0
    bond_off = 0
    block_off = 0
    port_off = 0
    bridge_off = 0
    edge_off = 0

    for m, row in enumerate(rows):
        mt = mols[row]
        a0 = int(atom_start[m])
        b0 = int(bond_start[m])
        atom_parts.append(np.arange(a0, a0 + mt.n_atoms, dtype=np.int64))
        bond_parts.append(np.arange(b0, b0 + mt.n_bonds, dtype=np.int64))
        atom_mol.append(np.full(mt.n_atoms, m, np.int64))
        bond_mol.append(np.full(mt.n_bonds, m, np.int64))

        # block atoms / bonds (batch-local embedding indices)
        ba0 = atom_off
        bb0 = bond_off
        block_atom_idx.append(mt.block_atoms + ba0)
        block_atom_seg.append(np.repeat(np.arange(block_off, block_off + mt.n_blocks),
                                        np.diff(mt.block_atom_ptr)))
        block_bond_idx.append(mt.block_bonds + bb0)
        block_bond_seg.append(np.repeat(np.arange(block_off, block_off + mt.n_blocks),
                                        np.diff(mt.block_bond_ptr)))
        block_bond_u.append(mt.block_bond_u + ba0)
        block_bond_v.append(mt.block_bond_v + ba0)
        block_n_atoms.append(np.diff(mt.block_atom_ptr))
        block_n_bonds.append(np.diff(mt.block_bond_ptr))
        block_n_ports.append(mt.block_n_ports)
        block_cycle_rank.append(mt.block_cycle_rank)
        block_seg.append(np.full(mt.n_blocks, m, np.int64))

        # ports
        port_block.append(mt.port_block + block_off)
        port_atom_idx.append(mt.port_atom + ba0)
        port_atom_degree.append(mt.port_atom_degree)
        port_atom_n_ports.append(mt.port_atom_n_ports)
        port_bucket_atom_idx.append(mt.port_bucket_atom_atom + ba0)
        port_bucket_atom_seg.append(mt.port_bucket_atom_seg + port_off * 6)
        port_bucket_bond_idx.append(mt.port_bucket_bond_bond + bb0)
        port_bucket_bond_seg.append(mt.port_bucket_bond_seg + port_off * 6)
        port_bucket_atom_count.append(mt.port_bucket_atom_count)
        port_bucket_bond_count.append(mt.port_bucket_bond_count)

        # bridges
        bridge_bond.append(mt.bridge_bond + bb0)
        bridge_block_u.append(mt.bridge_block_u + block_off)
        bridge_block_v.append(mt.bridge_block_v + block_off)
        bridge_port_u.append(mt.bridge_port_u + port_off)
        bridge_port_v.append(mt.bridge_port_v + port_off)
        bridge_seg.append(np.full(mt.n_bridges, m, np.int64))

        # directed edges
        edge_step.append(mt.edge_step)
        edge_source.append(mt.edge_source + block_off)
        edge_target.append(mt.edge_target + block_off)
        edge_port.append(mt.edge_port + port_off)
        edge_bridge.append(mt.edge_bridge + bridge_off)
        dep_edge.append(mt.dep_edge + edge_off)
        dep_port.append(mt.dep_port + port_off)
        dep_bridge.append(mt.dep_bridge + bridge_off)
        dep_port_feat.append(mt.dep_port_feat + port_off)
        dep_dist.append(mt.dep_dist_bucket)
        dep_target.append(mt.dep_target_edge + edge_off)

        atom_off += mt.n_atoms
        bond_off += mt.n_bonds
        block_off += mt.n_blocks
        port_off += mt.n_ports
        bridge_off += mt.n_bridges
        edge_off += mt.n_edges
        mol_block_ptr.append(block_off)
        mol_bridge_ptr.append(bridge_off)

    atom_feat = np.concatenate([np.asarray(raw["atom_feat"])[p] for p in atom_parts]) \
        if atom_parts else np.zeros((0, 9), np.int64)
    bond_feat = np.concatenate([np.asarray(raw["bond_feat"])[p] for p in bond_parts]) \
        if bond_parts else np.zeros((0, 3), np.int64)
    atom_feat = atom_feat.astype(np.int64, copy=False)
    bond_feat = bond_feat.astype(np.int64, copy=False)

    block_atom_idx = _cat(block_atom_idx)
    block_atom_seg = _cat(block_atom_seg)
    block_bond_idx = _cat(block_bond_idx)
    block_bond_seg = _cat(block_bond_seg)
    block_bond_u = _cat(block_bond_u)
    block_bond_v = _cat(block_bond_v)
    block_n_atoms = _cat(block_n_atoms)
    block_n_bonds = _cat(block_n_bonds)
    block_n_ports = _cat(block_n_ports)
    block_cycle_rank = _cat(block_cycle_rank)
    block_seg = _cat(block_seg)
    port_block = _cat(port_block)
    port_atom_idx = _cat(port_atom_idx)
    port_atom_degree = _cat(port_atom_degree)
    port_atom_n_ports = _cat(port_atom_n_ports)
    port_bucket_atom_idx = _cat(port_bucket_atom_idx)
    port_bucket_atom_seg = _cat(port_bucket_atom_seg)
    port_bucket_bond_idx = _cat(port_bucket_bond_idx)
    port_bucket_bond_seg = _cat(port_bucket_bond_seg)
    port_bucket_atom_count = _cat(port_bucket_atom_count)
    port_bucket_bond_count = _cat(port_bucket_bond_count)
    bridge_bond = _cat(bridge_bond)
    bridge_block_u = _cat(bridge_block_u)
    bridge_block_v = _cat(bridge_block_v)
    bridge_port_u = _cat(bridge_port_u)
    bridge_port_v = _cat(bridge_port_v)
    bridge_seg = _cat(bridge_seg)
    edge_step = _cat(edge_step)
    edge_source = _cat(edge_source)
    edge_target = _cat(edge_target)
    edge_port = _cat(edge_port)
    edge_bridge = _cat(edge_bridge)
    dep_edge = _cat(dep_edge)
    dep_port = _cat(dep_port)
    dep_bridge = _cat(dep_bridge)
    dep_port_feat = _cat(dep_port_feat)
    dep_dist = _cat(dep_dist)
    dep_target = _cat(dep_target)
    atom_mol = _cat(atom_mol)
    bond_mol = _cat(bond_mol)

    n_mol = len(rows)
    n_edges = int(edge_step.size)
    n_bridges = int(bridge_bond.size)
    max_step = int(edge_step.max()) if n_edges else -1

    # step schedule: edges grouped by step, deps grouped by their target step
    step_edge_ptr = np.zeros(max_step + 2, np.int64)
    if n_edges:
        order_e = np.argsort(edge_step, kind="stable")
        counts_e = np.bincount(edge_step, minlength=max_step + 1)
        step_edge_ptr[1:] = np.cumsum(counts_e)
        edge_order = order_e
        # rank of each edge within its step
        edge_rank = np.empty(n_edges, np.int64)
        for s in range(max_step + 1):
            lo, hi = step_edge_ptr[s], step_edge_ptr[s + 1]
            edge_rank[edge_order[lo:hi]] = np.arange(hi - lo)
    else:
        edge_order = np.zeros(0, np.int64)
        edge_rank = np.zeros(0, np.int64)
    dep_step = edge_step[dep_target] if dep_target.size else np.zeros(0, np.int64)
    step_dep_ptr = np.zeros(max_step + 2, np.int64)
    if dep_target.size:
        dep_order = np.argsort(dep_step, kind="stable")
        counts_d = np.bincount(dep_step, minlength=max_step + 1)
        step_dep_ptr[1:] = np.cumsum(counts_d)
        dep_target_local = edge_rank[dep_target]
    else:
        dep_order = np.zeros(0, np.int64)
        dep_target_local = np.zeros(0, np.int64)

    out = {
        "atom_feat": torch.as_tensor(atom_feat, dtype=torch.long),
        "bond_feat": torch.as_tensor(bond_feat, dtype=torch.long),
        "atom_mol": torch.as_tensor(atom_mol, dtype=torch.long),
        "bond_mol": torch.as_tensor(bond_mol, dtype=torch.long),
        "n_molecules": int(n_mol),
        "n_atoms": int(atom_feat.shape[0]),
        "n_bonds": int(bond_feat.shape[0]),
        "n_blocks": int(block_off),
        "n_ports": int(port_off),
        "n_bridges": int(n_bridges),
        "n_edges": int(n_edges),
        "max_step": int(max_step),
        "block_atom_idx": torch.as_tensor(block_atom_idx, dtype=torch.long),
        "block_atom_seg": torch.as_tensor(block_atom_seg, dtype=torch.long),
        "block_bond_idx": torch.as_tensor(block_bond_idx, dtype=torch.long),
        "block_bond_seg": torch.as_tensor(block_bond_seg, dtype=torch.long),
        "block_bond_u": torch.as_tensor(block_bond_u, dtype=torch.long),
        "block_bond_v": torch.as_tensor(block_bond_v, dtype=torch.long),
        "block_n_atoms": torch.as_tensor(block_n_atoms, dtype=torch.long),
        "block_n_bonds": torch.as_tensor(block_n_bonds, dtype=torch.long),
        "block_n_ports": torch.as_tensor(block_n_ports, dtype=torch.long),
        "block_cycle_rank": torch.as_tensor(block_cycle_rank, dtype=torch.long),
        "block_seg": torch.as_tensor(block_seg, dtype=torch.long),
        "port_block": torch.as_tensor(port_block, dtype=torch.long),
        "port_atom_idx": torch.as_tensor(port_atom_idx, dtype=torch.long),
        "port_atom_degree": torch.as_tensor(port_atom_degree, dtype=torch.long),
        "port_atom_n_ports": torch.as_tensor(port_atom_n_ports, dtype=torch.long),
        "port_bucket_atom_idx": torch.as_tensor(port_bucket_atom_idx, dtype=torch.long),
        "port_bucket_atom_seg": torch.as_tensor(port_bucket_atom_seg, dtype=torch.long),
        "port_bucket_bond_idx": torch.as_tensor(port_bucket_bond_idx, dtype=torch.long),
        "port_bucket_bond_seg": torch.as_tensor(port_bucket_bond_seg, dtype=torch.long),
        "port_bucket_atom_count": torch.as_tensor(port_bucket_atom_count, dtype=torch.long),
        "port_bucket_bond_count": torch.as_tensor(port_bucket_bond_count, dtype=torch.long),
        "bridge_bond": torch.as_tensor(bridge_bond, dtype=torch.long),
        "bridge_block_u": torch.as_tensor(bridge_block_u, dtype=torch.long),
        "bridge_block_v": torch.as_tensor(bridge_block_v, dtype=torch.long),
        "bridge_port_u": torch.as_tensor(bridge_port_u, dtype=torch.long),
        "bridge_port_v": torch.as_tensor(bridge_port_v, dtype=torch.long),
        "bridge_seg": torch.as_tensor(bridge_seg, dtype=torch.long),
        "edge_step": torch.as_tensor(edge_step, dtype=torch.long),
        "edge_source": torch.as_tensor(edge_source, dtype=torch.long),
        "edge_target": torch.as_tensor(edge_target, dtype=torch.long),
        "edge_port": torch.as_tensor(edge_port, dtype=torch.long),
        "edge_bridge": torch.as_tensor(edge_bridge, dtype=torch.long),
        "edge_order": torch.as_tensor(edge_order, dtype=torch.long),
        "step_edge_ptr": torch.as_tensor(step_edge_ptr, dtype=torch.long),
        "dep_edge": torch.as_tensor(dep_edge, dtype=torch.long),
        "dep_port": torch.as_tensor(dep_port, dtype=torch.long),
        "dep_bridge": torch.as_tensor(dep_bridge, dtype=torch.long),
        "dep_port_feat": torch.as_tensor(dep_port_feat, dtype=torch.long),
        "dep_dist_bucket": torch.as_tensor(dep_dist, dtype=torch.long),
        "dep_order": torch.as_tensor(dep_order, dtype=torch.long),
        "step_dep_ptr": torch.as_tensor(step_dep_ptr, dtype=torch.long),
        "dep_target_local": torch.as_tensor(dep_target_local, dtype=torch.long),
        "mol_block_ptr": torch.as_tensor(np.asarray(mol_block_ptr), dtype=torch.long),
        "mol_bridge_ptr": torch.as_tensor(np.asarray(mol_bridge_ptr), dtype=torch.long),
        "rows": np.asarray(rows, np.int64),
    }
    if device != "cpu":
        out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
    return out