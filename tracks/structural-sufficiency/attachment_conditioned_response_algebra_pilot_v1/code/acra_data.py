"""Local structural objects for ACRA-v1.

Design fixed by the round brief (see ``PROTOCOL.md``):

* one **directed connection event** ``u -> v`` per orientation of every real
  undirected bond;
* ``S`` = radius-1 ball around ``u`` in ``G - {u,v}`` (the attachment site);
* ``C`` = radius-2 ball around ``v`` in ``G - {u,v}`` (the local context);
* the removed bond keeps its raw 3-column attributes and its bridge / ring
  flags; every atom and bond of ``G`` participates through at least one event,
  and ring bonds are *not* treated as cuts that remove the rest of the ring;
* within a local ball, enumerate every connected induced subset of 1..4 atoms
  containing the root, canonicalise it with the parent round's root-fixed frame
  LUT (root pinned to slot 0, all winning permutations kept) and compress equal
  frames to ``unique frame + multiplicity``.  This compression is lossless for
  the shared ``FrameEncoder`` because the key contains exactly its inputs.

Nothing here reads a label.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import acra_common as C

PD = C.PD

#: per-(event, side) local scalar features, order frozen
LOCAL_SCALARS = (
    "log1p_n_atoms",
    "log1p_n_bonds",
    "cycle_rank",
    "n_ring_bonds",
    "ring_bond_frac",
    "root_degree",
    "root_in_ring",
    "has_ring",
)
N_LOCAL_SCALARS = len(LOCAL_SCALARS)


# ---------------------------------------------------------------------------
# raw access
# ---------------------------------------------------------------------------


def load_raw(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Official OGB cache (9 atom columns, 3 bond columns), sha-checked.

    SMILES strings are added from the same label-free file for reporting only.
    """
    raw = PD.load_raw_cache(cfg, "lipo")
    path = C.REPO_ROOT / str(cfg["raw_cache"])
    with np.load(path, allow_pickle=False) as handle:
        if "smiles" in handle.files:
            raw["smiles"] = np.asarray(handle["smiles"])
        if "degree" in handle.files:
            raw["degree"] = np.asarray(handle["degree"])
    return raw


def raw_molecule(raw: Mapping[str, Any], row: int) -> PD.Molecule:
    return PD.raw_molecule(raw, row)


def official_splits(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "train": np.asarray(raw["train_idx"], np.int64),
        "valid": np.asarray(raw["valid_idx"], np.int64),
        "test": np.asarray(raw["test_idx"], np.int64),
    }


# ---------------------------------------------------------------------------
# graph primitives
# ---------------------------------------------------------------------------


def adjacency(mol: PD.Molecule) -> list[list[int]]:
    return mol.adjacency_lists()


def bond_neighbour_lists(mol: PD.Molecule) -> list[list[tuple[int, int]]]:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(mol.n_atoms)]
    for e, (a, b) in enumerate(zip(mol.bond_u.tolist(), mol.bond_v.tolist())):
        adj[int(a)].append((int(b), e))
        adj[int(b)].append((int(a), e))
    return adj


def neighbours_within(adj: Sequence[Sequence[int]], root: int, radius: int,
                      banned: tuple[int, int] | None = None) -> list[int]:
    """Atoms at graph distance ``<= radius`` from ``root`` in ``G - banned``."""
    dist = {int(root): 0}
    frontier = [int(root)]
    for _ in range(int(radius)):
        nxt: list[int] = []
        for u in frontier:
            for w in adj[u]:
                w = int(w)
                if banned is not None and ((u == banned[0] and w == banned[1])
                                           or (u == banned[1] and w == banned[0])):
                    continue
                if w not in dist:
                    dist[w] = dist[u] + 1
                    nxt.append(w)
        frontier = nxt
    return sorted(dist)


def induced_bonds(mol: PD.Molecule, atoms: Sequence[int]) -> list[tuple[int, int, np.ndarray]]:
    """All real bonds inside ``atoms`` (indices are positions in ``atoms``)."""
    pos = {int(a): i for i, a in enumerate(atoms)}
    out: list[tuple[int, int, np.ndarray]] = []
    for e, (a, b) in enumerate(zip(mol.bond_u.tolist(), mol.bond_v.tolist())):
        a, b = int(a), int(b)
        if a in pos and b in pos:
            i, j = pos[a], pos[b]
            if i > j:
                i, j = j, i
            out.append((i, j, mol.bond_feat[e]))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def biconnected_bond_blocks(n_atoms: int,
                            adj_bonds: Sequence[Sequence[tuple[int, int]]]) -> tuple[np.ndarray, int]:
    """Vertex-biconnected block id per bond (Tarjan, edge stack).

    Used for the exact ``shared ring`` relation: two distinct bonds lie on a
    common cycle iff they belong to the same biconnected block and that block
    is not a bridge.
    """
    n_edges = sum(len(x) for x in adj_bonds) // 2
    comp = np.full(int(n_edges), -1, np.int32)
    disc = [-1] * int(n_atoms)
    low = [0] * int(n_atoms)
    stack: list[int] = []
    timer = 0
    n_blocks = 0

    def dfs(u: int, parent_edge: int) -> None:
        nonlocal timer, n_blocks
        disc[u] = low[u] = timer
        timer += 1
        for w, eid in adj_bonds[u]:
            if eid == parent_edge:
                continue
            if disc[w] < 0:
                stack.append(eid)
                dfs(w, eid)
                low[u] = min(low[u], low[w])
                if low[w] >= disc[u]:
                    while True:
                        x = stack.pop()
                        comp[x] = n_blocks
                        if x == eid:
                            break
                    n_blocks += 1
            elif disc[w] < disc[u]:
                stack.append(eid)
                low[u] = min(low[u], disc[w])

    for u in range(int(n_atoms)):
        if disc[u] < 0:
            dfs(u, -1)
    while stack:                    # defensive: should never happen
        comp[stack.pop()] = n_blocks
        n_blocks += 1
    return comp, n_blocks


def all_pairs_distances(n_atoms: int, adj: Sequence[Sequence[int]]) -> np.ndarray:
    dist = np.full((int(n_atoms), int(n_atoms)), -1, np.int16)
    for source in range(int(n_atoms)):
        dist[source, source] = 0
        frontier = [source]
        while frontier:
            nxt: list[int] = []
            for u in frontier:
                for w in adj[u]:
                    w = int(w)
                    if dist[source, w] < 0:
                        dist[source, w] = dist[source, u] + 1
                        nxt.append(w)
            frontier = nxt
    return dist


def bond_geometry(mol: PD.Molecule, adj: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Bridge / ring / biconnected-block / bond-pair relation data."""
    nb = bond_neighbour_lists(mol)
    block, n_blocks = biconnected_bond_blocks(mol.n_atoms, nb)
    block_size = np.bincount(block, minlength=n_blocks).astype(np.int32)
    is_bridge = block_size[block] == 1
    in_ring = ~is_bridge

    m = int(mol.bond_u.size)
    pair_e: list[int] = []
    pair_f: list[int] = []
    pair_bucket: list[int] = []
    pair_shared: list[int] = []
    if m >= 2:
        dist = all_pairs_distances(mol.n_atoms, adj)
        eu = mol.bond_u.astype(np.int64)
        ev = mol.bond_v.astype(np.int64)
        for e, f in itertools.combinations(range(m), 2):
            d = min(int(dist[eu[e], eu[f]]), int(dist[eu[e], ev[f]]),
                    int(dist[ev[e], eu[f]]), int(dist[ev[e], ev[f]]))
            # bond separation = shortest path length in the line graph
            # = 1 + (minimum endpoint distance); two distinct bonds are always
            # at least 1 apart, so the buckets are exactly 1,2,3,4+.
            if d < 0:
                bucket = 4                      # unreachable => 4+ bucket
            else:
                bucket = int(min(d + 1, 4))
            shared = int(block[e] == block[f] and not is_bridge[e] and not is_bridge[f])
            pair_e.append(int(e))
            pair_f.append(int(f))
            pair_bucket.append(bucket)
            pair_shared.append(shared)
    return {
        "block": block,
        "n_blocks": int(n_blocks),
        "block_size": block_size,
        "is_bridge": is_bridge,
        "in_ring": in_ring,
        "pair_e": np.asarray(pair_e, np.int32),
        "pair_f": np.asarray(pair_f, np.int32),
        "pair_bucket": np.asarray(pair_bucket, np.uint8),
        "pair_shared_ring": np.asarray(pair_shared, np.uint8),
    }


# ---------------------------------------------------------------------------
# local subgraph features
# ---------------------------------------------------------------------------


def rooted_connected_subsets(adj: Sequence[Sequence[int]], allowed: Iterable[int],
                             root: int, max_k: int = C.MAX_OBJECT_ATOMS
                             ) -> list[tuple[int, tuple[int, ...]]]:
    """Every connected induced subset of ``allowed`` with size <= max_k that
    contains ``root`` (the parent round's enumeration restricted to one root)."""
    vset = set(int(a) for a in allowed)
    root = int(root)
    if root not in vset:
        raise ValueError("root outside the allowed set")
    out: list[tuple[int, tuple[int, ...]]] = [(root, (root,))]
    frontier: list[frozenset[int]] = [frozenset((root,))]
    for _size in range(2, int(max_k) + 1):
        nxt: set[frozenset[int]] = set()
        for s in frontier:
            for u in s:
                for w in adj[u]:
                    w = int(w)
                    if w in vset and w not in s:
                        nxt.add(s | {w})
        frontier = list(nxt)
        out.extend((root, tuple(sorted(s))) for s in frontier)
    return out


def local_graph_stats(n_atoms: int, n_bonds: int, adj: Sequence[Sequence[int]],
                      root: int) -> np.ndarray:
    """Local scalars (counts and ring attributes) for one S/C ball."""
    comp = np.full(int(n_atoms), -1, np.int32)
    label = 0
    for start in range(int(n_atoms)):
        if comp[start] >= 0:
            continue
        comp[start] = label
        stack = [start]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if comp[w] < 0:
                    comp[w] = label
                    stack.append(w)
        label += 1
    cycle_rank = int(n_bonds) - int(n_atoms) + int(label)
    nb = bond_neighbour_lists_local(n_atoms, adj)
    block, n_blocks = biconnected_bond_blocks(n_atoms, nb)
    block_size = np.bincount(block, minlength=n_blocks) if n_blocks else np.zeros(0, np.int32)
    if n_blocks:
        is_bridge = block_size[block] == 1
        n_ring_bonds = int((~is_bridge).sum())
    else:
        is_bridge = np.zeros(0, bool)
        n_ring_bonds = 0
    root_deg = int(len(adj[root]))
    root_in_ring = 0
    if n_ring_bonds:
        bond_ids = [eid for _w, eid in nb[root]]
        root_in_ring = int(any(not bool(is_bridge[b]) for b in bond_ids))
    return np.asarray([
        np.log1p(float(n_atoms)),
        np.log1p(float(n_bonds)),
        float(cycle_rank),
        float(n_ring_bonds),
        float(n_ring_bonds) / max(1.0, float(n_bonds)),
        float(root_deg),
        float(root_in_ring),
        float(cycle_rank > 0),
    ], np.float32)


def bond_neighbour_lists_local(n_atoms: int,
                               adj: Sequence[Sequence[int]]) -> list[list[tuple[int, int]]]:
    out: list[list[tuple[int, int]]] = [[] for _ in range(int(n_atoms))]
    eid = 0
    for u in range(int(n_atoms)):
        for w in adj[u]:
            if int(u) < int(w):
                out[u].append((int(w), eid))
                out[w].append((int(u), eid))
                eid += 1
    return out


def local_identity(mol: PD.Molecule, atoms: Sequence[int], root: int,
                   adjacency_pairs: Sequence[tuple[int, int, np.ndarray]],
                   colors: PD.ColorSpace) -> str:
    atoms = list(atoms)
    port = atoms.index(int(root))          # canonical_key wants a *local* index
    return PD.canonical_key([mol.atom_feat[int(a)] for a in atoms], adjacency_pairs,
                            port, colors)


def frame_patterns(mol: PD.Molecule, atoms: Sequence[int], root: int,
                   luts: Mapping[int, Mapping[str, np.ndarray]],
                   adj: Sequence[Sequence[int]] | None = None) -> dict[int, dict[tuple, int]]:
    """Rooted 1-4 frames of one local ball, deduplicated losslessly.

    Returns ``{k: {pattern_key: multiplicity}}``.  ``pattern_key`` contains the
    canonical adjacency bits and every raw attribute the shared ``FrameEncoder``
    consumes, so merging equal keys cannot change any downstream number.
    """
    atoms = tuple(sorted(int(a) for a in atoms))
    if int(root) not in atoms:
        raise ValueError("root not in local subgraph")
    adj = adjacency(mol) if adj is None else adj
    objects = rooted_connected_subsets(adj, atoms, int(root))
    # frames_for_objects groups consecutive equal sizes
    objects.sort(key=lambda t: len(t[1]))
    atom_row = np.arange(mol.n_atoms, dtype=np.int32)
    bond_row = np.full((mol.n_atoms, mol.n_atoms), -1, np.int32)
    for e, (a, b) in enumerate(zip(mol.bond_u.tolist(), mol.bond_v.tolist())):
        bond_row[int(a), int(b)] = e
        bond_row[int(b), int(a)] = e
    frames = PD.frames_for_objects(objects, atom_row, bond_row, luts, C.MAX_OBJECT_ATOMS)
    out: dict[int, dict[tuple, int]] = {k: {} for k in range(1, C.MAX_OBJECT_ATOMS + 1)}
    slot_pairs = list(C.FRAME_PAIRS)
    atom_pad = np.zeros((C.MAX_OBJECT_ATOMS, C.ATOM_COLUMNS), np.uint8)
    bond_pad = np.zeros((len(slot_pairs), C.BOND_COLUMNS), np.uint8)
    for f in range(frames.frame_k.shape[0]):
        k = int(frames.frame_k[f])
        bits = 0
        a_vals = atom_pad.copy()
        b_vals = bond_pad.copy()
        for slot, (i, j) in enumerate(slot_pairs):
            bid = int(frames.bond_rows[f, slot])
            if bid >= 0:
                bits |= 1 << slot
                b_vals[slot] = mol.bond_feat[bid]
        a_idx = frames.atom_rows[f, :k]
        if np.any(a_idx < 0):
            raise AssertionError("negative atom slot")
        a_vals[:k] = mol.atom_feat[a_idx]
        key = (int(k), int(bits), a_vals.tobytes(), b_vals.tobytes())
        out[k][key] = out[k].get(key, 0) + 1
    return out


def local_structure(mol: PD.Molecule, atoms: Sequence[int], root: int,
                    luts, colors, adj=None) -> dict[str, Any]:
    atoms = tuple(sorted(int(a) for a in atoms))
    pos = {a: i for i, a in enumerate(atoms)}
    pairs = induced_bonds(mol, atoms)
    local_adj: list[list[int]] = [[] for _ in atoms]
    for i, j, _b in pairs:
        local_adj[i].append(j)
        local_adj[j].append(i)
    stats = local_graph_stats(len(atoms), len(pairs), local_adj, pos[int(root)])
    return {
        "identity": local_identity(mol, atoms, int(root), pairs, colors),
        "frames": frame_patterns(mol, atoms, int(root), luts, adj=adj),
        "scalars": stats,
        "n_atoms": int(len(atoms)),
        "n_bonds": int(len(pairs)),
    }


# ---------------------------------------------------------------------------
# directed events
# ---------------------------------------------------------------------------


def event_balls(mol: PD.Molecule, adj: Sequence[Sequence[int]], bond: int, direction: int = 0
                ) -> tuple[tuple[int, tuple[int, ...]], tuple[int, tuple[int, ...]]]:
    """(S, C) of the directed event ``bond_u -> bond_v`` (or the reverse).

    ``S`` is always the radius-1 ball around the *source* atom and ``C`` the
    radius-2 ball around the *target* atom, both in ``G - bond``.
    """
    if int(direction) not in (0, 1):
        raise ValueError("direction must be 0 or 1")
    u = int(mol.bond_u[bond])
    v = int(mol.bond_v[bond])
    source, target = (u, v) if int(direction) == 0 else (v, u)
    s_atoms = neighbours_within(adj, source, 1, (u, v))
    c_atoms = neighbours_within(adj, target, 2, (u, v))
    return (source, tuple(s_atoms)), (target, tuple(c_atoms))


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def _concat_ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Concatenate ``arange(s, e)`` for every pair (contiguous per segment)."""
    lengths = np.asarray(ends, np.int64) - np.asarray(starts, np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    out = np.empty(total, np.int64)
    offset = 0
    for s, n in zip(np.asarray(starts, np.int64).tolist(), lengths.tolist()):
        if n:
            out[offset:offset + n] = np.arange(s, s + n, dtype=np.int64)
            offset += n
    return out


class BatchSource:
    """Index tensors for a set of molecules, built from the frozen pack.

    The pack stores every per-molecule array contiguously, so a batch is a
    concatenation of short contiguous slices (no padding, no truncation).
    """

    def __init__(self, pack: Mapping[str, Any], device: Any) -> None:
        import torch

        self.pack = pack
        self.device = device
        names = ("atom_ptr", "atom_feat", "ev_ptr", "ev_bond_attr", "ev_bond", "ev_dir",
                 "ev_bridge", "ev_in_ring",
                 "ev_key_s", "ev_key_c", "ev_scalars_s", "ev_scalars_c",
                 "blk_ptr", "blk_count", "ent_pat", "ent_mult",
                 "pat_k", "pat_bits", "pat_atom", "pat_bond",
                 "pair_ptr", "pair_e", "pair_f", "pair_bucket", "pair_shared_ring",
                 "mol_n_bonds")
        self.t: dict[str, Any] = {}
        for name in names:
            arr = np.ascontiguousarray(pack[name])
            tensor = torch.as_tensor(arr)
            if name in ("ev_scalars_s", "ev_scalars_c"):
                tensor = tensor.float()
            else:
                tensor = tensor.long()
            self.t[name] = tensor.to(device)
        self.n_molecules = int(pack["n_molecules"])

    def make(self, rows: Sequence[int]) -> dict[str, Any]:
        import torch

        rows = np.asarray(rows, np.int64)
        t = self.t
        starts_atom = self.pack["atom_ptr"][rows]
        ends_atom = self.pack["atom_ptr"][rows + 1]
        atom_idx = _concat_ranges(starts_atom, ends_atom)
        starts_ev = self.pack["ev_ptr"][rows]
        ends_ev = self.pack["ev_ptr"][rows + 1]
        ev_idx = _concat_ranges(starts_ev, ends_ev)
        starts_pair = self.pack["pair_ptr"][rows]
        ends_pair = self.pack["pair_ptr"][rows + 1]
        pair_idx = _concat_ranges(starts_pair, ends_pair)
        starts_blk = starts_ev * 8
        ends_blk = ends_ev * 8
        blk_idx = _concat_ranges(starts_blk, ends_blk)
        blk_ptr = self.pack["blk_ptr"]
        starts_ent = blk_ptr[starts_blk]
        ends_ent = blk_ptr[ends_blk]
        ent_idx = _concat_ranges(starts_ent, ends_ent)

        ent_counts = blk_ptr[blk_idx + 1] - blk_ptr[blk_idx]
        if int(ent_counts.sum()) != int(ent_idx.size):
            raise AssertionError("entry slices lost entries")
        blk_local = np.repeat(np.arange(blk_idx.size, dtype=np.int64), ent_counts)
        ev_counts = ends_ev - starts_ev
        ev_mol = np.repeat(np.arange(rows.size, dtype=np.int64), ev_counts)
        atom_mol = np.repeat(np.arange(rows.size, dtype=np.int64), ends_atom - starts_atom)
        pair_counts = ends_pair - starts_pair
        pair_mol = np.repeat(np.arange(rows.size, dtype=np.int64), pair_counts)

        n_atoms = ends_atom - starts_atom
        n_bonds = self.pack["mol_n_bonds"][rows].astype(np.int64)
        bond_off = np.concatenate([[0], np.cumsum(n_bonds)[:-1]]) if rows.size else np.zeros(0, np.int64)
        pair_e = self.pack["pair_e"][pair_idx].astype(np.int64) + bond_off[pair_mol]
        pair_f = self.pack["pair_f"][pair_idx].astype(np.int64) + bond_off[pair_mol]

        def ten(arr, dtype=None):
            out = torch.as_tensor(np.ascontiguousarray(arr))
            if dtype is not None:
                out = out.to(dtype)
            return out.to(self.device)

        return {
            "rows": rows,
            "n_molecules": int(rows.size),
            "atom_idx": ten(atom_idx),
            "atom_mol": ten(atom_mol),
            "ev_idx": ten(ev_idx),
            "ev_mol": ten(ev_mol),
            "blk_idx": ten(blk_idx),
            "blk_local": ten(blk_local),
            "ent_idx": ten(ent_idx),
            "pair_idx": ten(pair_idx),
            "pair_mol": ten(pair_mol),
            "pair_e": ten(pair_e),
            "pair_f": ten(pair_f),
            "n_bonds": ten(n_bonds),
            "n_atoms_per_mol": ten(n_atoms),
            "ent_pat": t["ent_pat"][ten(ent_idx, torch.long)],
            "ent_mult": t["ent_mult"][ten(ent_idx, torch.long)],
            "blk_count_batch": t["blk_count"][ten(blk_idx, torch.long)],
            "n_events": int(ev_idx.size),
            "n_atoms": int(atom_idx.size),
            "n_pairs": int(pair_idx.size),
            "n_blocks": int(blk_idx.size),
            "n_entries": int(ent_idx.size),
        }


def pack_meta(pack: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(str(pack["meta_json"]))