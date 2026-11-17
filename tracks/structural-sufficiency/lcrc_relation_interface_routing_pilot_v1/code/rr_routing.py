"""S1 / S2 occurrence enumeration with bond identity, and the routing cache.

The enumeration is a *routing wrapper* around the frozen C1 occurrence semantics:

* S1 per root atom in sorted-adjacency order, exactly ``molecular_graph.enumerate_s1``
  (joint ``(neighbour type, bond type)`` code) — plus the physical bond id;
* S2 ordered non-backtracking 2-paths ``root -> middle -> end`` with
  ``path_cat``/``end63``/``closure`` computed by the *same* formulas as
  ``molecular_graph.enumerate_s2`` — plus the two physical bond ids ``e1,e2``.

Both wrappers are verified occurrence-by-occurrence against the frozen caches
(``analysis/routing_invariants.json``) before any model is trained.

Routing rule (frozen):

* S1 occurrence of physical bond ``e`` at root ``v`` -> block ``owner(e)``;
* S2 occurrence with ``owner(e1) == owner(e2) = i`` -> internal S2 of block ``i``
  (whole occurrence: type, end, joint, closure, content);
* S2 occurrence with ``owner(e1) = i != j = owner(e2)`` -> interface ``{i,j}``;
* each occurrence is recorded exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import rr_common as C
import rr_partition as P


# ---------------------------------------------------------------------------
# occurrence enumeration (same order as the frozen enumerators)
# ---------------------------------------------------------------------------


def sorted_adjacency(
    n_atoms: int,
    edges: Sequence[tuple[int, int]],
    bond_type: Sequence[int],
) -> list[list[tuple[int, int, int]]]:
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(int(n_atoms))]
    for e, (u, v) in enumerate(edges):
        t = int(bond_type[e])
        adj[int(u)].append((int(v), t, int(e)))
        adj[int(v)].append((int(u), t, int(e)))
    for row in adj:
        row.sort()
    return adj


def enumerate_s1_routed(
    atom_type: np.ndarray,
    adj: Sequence[Sequence[tuple[int, int, int]]],
    n_bond_types: int,
) -> dict[str, np.ndarray]:
    roots: list[int] = []
    codes: list[int] = []
    bonds: list[int] = []
    for root, row in enumerate(adj):
        for nb, bt, e in row:
            roots.append(int(root))
            codes.append(int(atom_type[int(nb)]) * int(n_bond_types) + (int(bt) - 1))
            bonds.append(int(e))
    return {
        "root": np.asarray(roots, dtype=np.int64),
        "code": np.asarray(codes, dtype=np.int64),
        "bond": np.asarray(bonds, dtype=np.int64),
    }


def enumerate_s2_routed(
    atom_type: np.ndarray,
    adj: Sequence[Sequence[tuple[int, int, int]]],
    n_bond_types: int,
    edges: Sequence[tuple[int, int]],
) -> dict[str, np.ndarray]:
    nb = int(n_bond_types)
    adjacency_of = [set() for _ in range(len(adj))]
    for u, v in edges:
        adjacency_of[int(u)].add(int(v))
        adjacency_of[int(v)].add(int(u))
    root_l: list[int] = []
    end_l: list[int] = []
    cat_l: list[int] = []
    end63_l: list[int] = []
    close_l: list[int] = []
    e1_l: list[int] = []
    e2_l: list[int] = []
    for root, row in enumerate(adj):
        for middle, bt1, e1 in row:
            type_middle = int(atom_type[int(middle)])
            for end, bt2, e2 in adj[int(middle)]:
                end = int(end)
                if end == root:
                    continue
                root_l.append(int(root))
                end_l.append(end)
                cat_l.append(type_middle * nb * nb + (int(bt1) - 1) * nb + (int(bt2) - 1))
                end63_l.append(int(atom_type[end]) * nb + (int(bt2) - 1))
                close_l.append(1 if end in adjacency_of[int(root)] else 0)
                e1_l.append(int(e1))
                e2_l.append(int(e2))
    return {
        "root": np.asarray(root_l, dtype=np.int64),
        "end": np.asarray(end_l, dtype=np.int64),
        "path_cat": np.asarray(cat_l, dtype=np.int64),
        "end63": np.asarray(end63_l, dtype=np.int64),
        "closure": np.asarray(close_l, dtype=np.int8),
        "e1": np.asarray(e1_l, dtype=np.int64),
        "e2": np.asarray(e2_l, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# per-split routing cache
# ---------------------------------------------------------------------------


@dataclass
class SplitRouting:
    split: str
    arrays: dict[str, np.ndarray]
    stats: dict[str, Any]
    fingerprint: str


def _build_belongs(
    part: P.Partition,
) -> tuple[list[tuple[int, int, float]], dict[tuple[int, int], int], np.ndarray]:
    """CSR-ordered ``(atom, block, weight)`` triples plus their index and counts."""
    triples: list[tuple[int, int, float]] = []
    index: dict[tuple[int, int], int] = {}
    counts = np.zeros(part.n_atoms, dtype=np.int64)
    for v, memberships in enumerate(part.weights()):
        counts[v] = len(memberships)
        for block, weight in memberships:
            index[(int(v), int(block))] = len(triples)
            triples.append((int(v), int(block), float(weight)))
    return triples, index, counts


def build_split_routing(
    cache: Mapping[str, Any],
    split: str,
    *,
    n_bond_types: int,
    verify: bool = True,
) -> SplitRouting:
    """Build the routing arrays for every molecule of one split cache."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_type_all = np.asarray(cache["atom_type"], dtype=np.int64)
    bond_type_all = np.asarray(cache["bond_type"], dtype=np.int64)
    bond_index_all = np.asarray(cache["bond_index"], dtype=np.int64)
    nbr_code_all = np.asarray(cache["nbr_code"], dtype=np.int64)
    path_cat_all = np.asarray(cache["path_cat"], dtype=np.int64)
    path_end63_all = np.asarray(cache["path_end63"], dtype=np.int64)
    path_end_all = np.asarray(cache["path_end"], dtype=np.int64)
    path_closure_all = np.asarray(cache["path_closure"], dtype=np.int64)
    n_mol = int(cache["n_molecules"])

    # block level
    block_ptr = np.zeros(n_mol + 1, dtype=np.int64)
    blk_bond_ptr: list[int] = [0]
    blk_bond: list[int] = []
    blk_n_bonds: list[int] = []
    blk_n_atoms: list[int] = []
    blk_kind: list[int] = []

    # atom -> block membership CSR
    belong_ptr = np.zeros(int(atom_ptr[-1]) + 1, dtype=np.int64)
    belong_atom: list[int] = []
    belong_block: list[int] = []
    belong_weight: list[float] = []

    # per-occurrence routing
    nbr_bucket = np.zeros(int(nbr_ptr[-1]), dtype=np.int64)
    path_bucket = np.full(int(path_ptr[-1]), -1, dtype=np.int64)
    path_owner1 = np.zeros(int(path_ptr[-1]), dtype=np.int64)
    path_owner2 = np.zeros(int(path_ptr[-1]), dtype=np.int64)
    path_cross = np.zeros(int(path_ptr[-1]), dtype=np.int8)
    path_iface = np.full(int(path_ptr[-1]), -1, dtype=np.int64)

    # bonds -> (endpoint, owner) bucket
    bond_bucket_u = np.zeros(int(bond_ptr[-1]), dtype=np.int64)
    bond_bucket_v = np.zeros(int(bond_ptr[-1]), dtype=np.int64)

    # interfaces per molecule
    iface_ptr = np.zeros(n_mol + 1, dtype=np.int64)
    iface_a: list[int] = []
    iface_b: list[int] = []

    # statistics
    n_cross = 0
    n_internal = 0
    max_blocks = 0
    n_single_block = 0
    n_zero_bond_blocks = 0
    n_isolated_atoms = 0
    block_sizes: list[int] = []
    cycle_atoms_total = 0
    bond_mismatch: list[int] = []
    s1_mismatch: list[int] = []
    s2_mismatch: list[int] = []
    atoms_with_multiplicity = 0

    for row in range(n_mol):
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = a1 - a0
        edges = [(int(bond_index_all[j, 0]), int(bond_index_all[j, 1])) for j in range(b0, b1)]
        atom_type = atom_type_all[a0:a1]
        bond_type = bond_type_all[b0:b1]
        part = P.partition_molecule(n, edges, atom_type, bond_type)

        checks = P.check_partition(part, edges)
        if not (
            checks["bond_unique_ownership"]
            and checks["blocks_connected"]
            and checks["weights_sum_to_one"]
        ):
            raise RuntimeError(f"{split}: partition check failed on molecule row {row}: {checks}")
        n_zero_bond_blocks += int(checks["n_zero_bond_blocks"])
        n_isolated_atoms += int(sum(1 for v in range(n) if int(part.degrees[v]) == 0))

        triples, belong_index, atom_counts = _build_belongs(part)
        for v, block, weight in triples:
            belong_atom.append(int(v))
            belong_block.append(block)
            belong_weight.append(weight)
        belong_ptr[a0 + 1 : a1 + 1] = belong_ptr[a0] + np.cumsum(atom_counts)
        per_atom_count: dict[int, int] = {}
        for v, _b, _w in triples:
            per_atom_count[v] = per_atom_count.get(v, 0) + 1
        atoms_with_multiplicity += int(sum(1 for c in per_atom_count.values() if c > 1))

        block_ptr[row + 1] = block_ptr[row] + part.n_blocks
        max_blocks = max(max_blocks, int(part.n_blocks))
        if part.n_blocks < 2:
            n_single_block += 1
        for block in part.blocks:
            blk_n_bonds.append(int(block["n_bonds"]))
            blk_n_atoms.append(int(block["n_atoms"]))
            blk_kind.append(int(P.BLOCK_KIND_RANK[block["kind"]]))
            block_sizes.append(int(block["n_bonds"]))
            for e in block["bonds"]:
                blk_bond.append(int(b0 + e))
            blk_bond_ptr.append(len(blk_bond))
            if block["kind"] == "cycle":
                cycle_atoms_total += int(block["n_atoms"])

        # ---- S1 routing ---------------------------------------------------
        adj = sorted_adjacency(n, edges, bond_type)
        s1 = enumerate_s1_routed(atom_type, adj, n_bond_types)
        s0, s1_end = int(nbr_ptr[a0]), int(nbr_ptr[a1])
        if s1["code"].size != s1_end - s0:
            s1_mismatch.append(row)
        elif verify and not np.array_equal(s1["code"], nbr_code_all[s0:s1_end]):
            s1_mismatch.append(row)
        for k in range(s1["code"].size):
            root = int(s1["root"][k])
            owner = int(part.bond_block[int(s1["bond"][k])])
            nbr_bucket[s0 + k] = belong_index[(root, owner)]

        # ---- S2 routing ---------------------------------------------------
        s2 = enumerate_s2_routed(atom_type, adj, n_bond_types, edges)
        p0, p1 = int(path_ptr[a0]), int(path_ptr[a1])
        if s2["path_cat"].size != p1 - p0:
            s2_mismatch.append(row)
        elif verify and not (
            np.array_equal(s2["path_cat"], path_cat_all[p0:p1])
            and np.array_equal(s2["end63"], path_end63_all[p0:p1])
            and np.array_equal(s2["end"], path_end_all[p0:p1])
            and np.array_equal(s2["closure"], path_closure_all[p0:p1].astype(np.int64))
        ):
            s2_mismatch.append(row)

        pairs: dict[tuple[int, int], int] = {}
        local_iface: list[int] = []
        for k in range(s2["path_cat"].size):
            root = int(s2["root"][k])
            o1 = int(part.bond_block[int(s2["e1"][k])])
            o2 = int(part.bond_block[int(s2["e2"][k])])
            path_owner1[p0 + k] = o1
            path_owner2[p0 + k] = o2
            if o1 == o2:
                path_cross[p0 + k] = 0
                path_bucket[p0 + k] = belong_index[(root, o1)]
                local_iface.append(-1)
                n_internal += 1
            else:
                path_cross[p0 + k] = 1
                key = (min(o1, o2), max(o1, o2))
                slot = pairs.get(key)
                if slot is None:
                    slot = len(pairs)
                    pairs[key] = slot
                local_iface.append(slot)
                n_cross += 1
        if pairs:
            ordered = sorted(pairs)
            remap = {pairs[key]: i for i, key in enumerate(ordered)}
            for k, slot in enumerate(local_iface):
                if slot >= 0:
                    path_iface[p0 + k] = iface_ptr[row] + remap[slot]
        iface_ptr[row + 1] = iface_ptr[row] + len(pairs)
        for a, b in sorted(pairs):
            iface_a.append(int(block_ptr[row] + a))
            iface_b.append(int(block_ptr[row] + b))
            if a == b:
                raise RuntimeError("an interface must join two different blocks")

        # ---- bond endpoints -> bucket ------------------------------------
        for e in range(len(edges)):
            u, v = edges[e]
            owner = int(part.bond_block[e])
            bond_bucket_u[b0 + e] = belong_index[(int(u), owner)]
            bond_bucket_v[b0 + e] = belong_index[(int(v), owner)]

    arrays = {
        "block_ptr": block_ptr,
        "blk_bond_ptr": np.asarray(blk_bond_ptr, dtype=np.int64),
        "blk_bond": np.asarray(blk_bond, dtype=np.int64),
        "blk_n_bonds": np.asarray(blk_n_bonds, dtype=np.int64),
        "blk_n_atoms": np.asarray(blk_n_atoms, dtype=np.int64),
        "blk_kind": np.asarray(blk_kind, dtype=np.int64),
        "belong_ptr": belong_ptr,
        "belong_atom": np.asarray(belong_atom, dtype=np.int64),
        "belong_block": np.asarray(belong_block, dtype=np.int64),
        "belong_weight": np.asarray(belong_weight, dtype=np.float32),
        "nbr_bucket": nbr_bucket,
        "path_bucket": path_bucket,
        "path_cross": path_cross,
        "path_owner1": path_owner1,
        "path_owner2": path_owner2,
        "path_iface": path_iface,
        "bond_bucket_u": bond_bucket_u,
        "bond_bucket_v": bond_bucket_v,
        "iface_ptr": iface_ptr,
        "iface_a": np.asarray(iface_a, dtype=np.int64),
        "iface_b": np.asarray(iface_b, dtype=np.int64),
    }
    n_blocks = int(block_ptr[-1])
    n_atoms = int(atom_ptr[-1])
    n_s2 = int(path_ptr[-1])
    if nbr_bucket.shape[0] != 2 * int(bond_ptr[-1]):
        raise RuntimeError(f"{split}: S1 occurrences {nbr_bucket.shape[0]} != 2*bonds")
    if arrays["blk_bond"].shape[0] != int(bond_ptr[-1]):
        raise RuntimeError(f"{split}: block bond ownership does not cover every bond exactly once")
    if n_s2 and not (int((arrays["path_cross"] == 0).sum()) == n_internal):
        raise RuntimeError(f"{split}: internal S2 accounting mismatch")
    if n_s2 and int((arrays["path_cross"] == 1).sum()) != n_cross:
        raise RuntimeError(f"{split}: cross S2 accounting mismatch")
    if n_internal + n_cross != n_s2:
        raise RuntimeError(f"{split}: S2 conservation violated: {n_internal}+{n_cross} != {n_s2}")
    if s1_mismatch or s2_mismatch:
        raise RuntimeError(
            f"{split}: occurrence reconstruction disagrees with the frozen cache on molecules "
            f"S1={s1_mismatch[:5]} S2={s2_mismatch[:5]}"
        )

    fingerprint = C.sha256_json(
        {key: C.sha256_array(np.ascontiguousarray(value)) for key, value in sorted(arrays.items())}
    )
    stats = {
        "split": split,
        "n_molecules": n_mol,
        "n_atoms": n_atoms,
        "n_bonds": int(bond_ptr[-1]),
        "n_blocks": n_blocks,
        "n_interfaces": int(iface_ptr[-1]),
        "mean_blocks_per_molecule": float(np.mean(np.diff(block_ptr))),
        "max_blocks_per_molecule": int(max_blocks),
        "n_molecules_single_block": int(n_single_block),
        "n_molecules_multi_block": int(n_mol - n_single_block),
        "n_zero_bond_blocks": int(n_zero_bond_blocks),
        "n_isolated_atoms": int(n_isolated_atoms),
        "n_atoms_in_multiple_blocks": int(atoms_with_multiplicity),
        "cycle_atoms": int(cycle_atoms_total),
        "n_s1_occurrences": int(nbr_bucket.shape[0]),
        "n_s2_total": int(n_s2),
        "n_s2_internal": int(n_internal),
        "n_s2_cross": int(n_cross),
        "s2_internal_plus_cross_equals_total": bool(n_internal + n_cross == n_s2),
        "mean_interfaces_per_molecule": float(np.mean(np.diff(iface_ptr))),
        "block_bond_size_histogram": _hist(block_sizes, int(bond_ptr[-1])),
        "mean_block_bonds": float(np.mean(blk_n_bonds)) if blk_n_bonds else 0.0,
        "s1_occurrences_match_frozen_cache": True,
        "s2_occurrences_match_frozen_cache": True,
        "routing_fingerprint": fingerprint,
    }
    return SplitRouting(split=split, arrays=arrays, stats=stats, fingerprint=fingerprint)


def _hist(values: Sequence[int], cap: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        key = str(int(v))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def cross_boundary_molecule_stats(
    arrays: Mapping[str, np.ndarray], cache: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Per-molecule block / cross-boundary-S2 availability (label free)."""
    block_ptr = np.asarray(arrays["block_ptr"])
    path_cross = np.asarray(arrays["path_cross"])
    path_ptr = np.asarray(cache["path_ptr"])
    atom_ptr = np.asarray(cache["atom_ptr"])
    iface_ptr = np.asarray(arrays["iface_ptr"])
    n_mol = int(block_ptr.size) - 1
    n_blocks = np.diff(block_ptr)
    cross = np.zeros(n_mol, dtype=np.int64)
    for row in range(n_mol):
        p0 = int(path_ptr[int(atom_ptr[row])])
        p1 = int(path_ptr[int(atom_ptr[row + 1])])
        cross[row] = int(path_cross[p0:p1].sum())
    return {
        "n_blocks": n_blocks,
        "n_cross_s2": cross,
        "n_interfaces": np.diff(iface_ptr),
        "multi_block": n_blocks >= 2,
        "has_cross_s2": cross > 0,
        "usable": (n_blocks >= 2) & (cross > 0),
    }


__all__ = [
    "sorted_adjacency",
    "enumerate_s1_routed",
    "enumerate_s2_routed",
    "SplitRouting",
    "build_split_routing",
    "cross_boundary_molecule_stats",
]
