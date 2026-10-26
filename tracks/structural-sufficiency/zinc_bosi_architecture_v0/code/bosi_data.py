"""BOSI v0: label-free structural relation construction + batch collation.

The only structure-dependent object this module creates is, for every directed
half-edge ``(v, e)`` of a physical bond ``e = {v, u}``, the partition of its
partner half-edges ``(v, f)``, ``f != e`` into the four mutually exclusive
classes of DESIGN.md §2.3:

    T  the two far endpoints (u and w) are directly bonded  (triangle closure)
    Q  else they share a neighbour other than v             (extra common nbr)
    R  else e and f lie in the same biconnected block with >= 3 edges
    N  else                                                 (no closure)

plus the raw closure counts that priority would otherwise discard.

Everything is a deterministic function of the raw graph (atom indices, bond
endpoints, bond types).  No label, no dataset statistic, no RNG.

Relation cache layout (flat, one file per split):
    half_ptr   [M+1]   offsets into the half-edge arrays
    half_atom  [H]     centre atom (molecule-local row in the flat atom arrays)
    half_bond  [E*2]   physical bond row of the half-edge
    partner_ptr[H+1]   offsets into the partner arrays
    partner_half[P]    half-edge row of the partner (same centre)
    partner_rel [P]    relation id, 0=T 1=Q 2=R 3=N
    closure    [H,4]   raw closure summary (see ``_closure`` below)
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

N_REL = 4
REL_NAMES = ("T", "Q", "R", "N")
CLOSURE_DIM = 4


# ---------------------------------------------------------------------------
# one molecule
# ---------------------------------------------------------------------------


def _biconnected_bond_blocks(n_atoms: int, bond_index: np.ndarray) -> np.ndarray:
    """Block id per bond, -1 for bonds outside every >=3-edge block.

    ``networkx.biconnected_component_edges`` is the reference implementation
    already used by the frozen ``structural_relation_objects_v1`` object
    builder, so the block definition cannot drift from the historical one.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    for u, v in bond_index:
        graph.add_edge(int(u), int(v))
    block_of = np.full(len(bond_index), -1, dtype=np.int64)
    key_to_bond = {(min(int(u), int(v)), max(int(u), int(v))): k
                   for k, (u, v) in enumerate(bond_index)}
    bid = 0
    for edges in nx.biconnected_component_edges(graph):
        if len(edges) < 3:
            continue
        for u, v in edges:
            key = (min(int(u), int(v)), max(int(u), int(v)))
            block_of[key_to_bond[key]] = bid
        bid += 1
    return block_of


def relation_arrays(n_atoms: int, bond_index: np.ndarray,
                    bond_type: np.ndarray) -> dict:
    """Per-molecule relation arrays for every directed half-edge.

    ``bond_index`` keeps the caller's storage order and direction; the returned
    half-edge rows are ``2*e`` (centre = bond_index[e, 0]) and ``2*e + 1``
    (centre = bond_index[e, 1]).
    """
    bond_index = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    n_bonds = len(bond_index)
    bond_set = set()
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for e, (u, v) in enumerate(bond_index):
        u, v = int(u), int(v)
        bond_set.add((min(u, v), max(u, v)))
        adj[u].append(e)
        adj[v].append(e)
    nbr_sets = []
    for a in range(n_atoms):
        s = set()
        for f in adj[a]:
            u, v = int(bond_index[f][0]), int(bond_index[f][1])
            s.add(v if u == a else u)
        nbr_sets.append(s)

    block_of = _biconnected_bond_blocks(n_atoms, bond_index)
    half_of = np.empty((n_bonds, 2), dtype=np.int64)
    half_of[:, 0] = 2 * np.arange(n_bonds)
    half_of[:, 1] = 2 * np.arange(n_bonds) + 1

    half_atom = np.zeros(2 * n_bonds, dtype=np.int64)
    half_bond = np.zeros(2 * n_bonds, dtype=np.int64)
    half_deg = np.zeros(2 * n_bonds, dtype=np.int64)
    partner_half: list[int] = []
    partner_rel: list[int] = []
    partner_ptr = np.zeros(2 * n_bonds + 1, dtype=np.int64)
    closure = np.zeros((2 * n_bonds, CLOSURE_DIM), dtype=np.float32)

    for e in range(n_bonds):
        u0, v0 = int(bond_index[e][0]), int(bond_index[e][1])
        for side, (centre, other) in enumerate(((u0, v0), (v0, u0))):
            h = 2 * e + side
            half_atom[h] = centre
            half_bond[h] = e
            half_deg[h] = len(adj[centre])
            c_list = []
            for f in adj[centre]:
                if f == e:
                    continue
                w = int(bond_index[f][0]) + int(bond_index[f][1]) - centre
                common = nbr_sets[other] & nbr_sets[w]
                common.discard(centre)
                c = len(common)
                c_list.append(c)
                if (min(other, w), max(other, w)) in bond_set:
                    rel = 0  # T
                elif c > 0:
                    rel = 1  # Q
                elif block_of[e] >= 0 and block_of[e] == block_of[f]:
                    rel = 2  # R
                else:
                    rel = 3  # N
                partner_half.append(2 * f + (0 if int(bond_index[f][0]) == centre else 1))
                partner_rel.append(rel)
            partner_ptr[h + 1] = len(partner_half)
            # raw closure summary: priority would otherwise discard these
            if c_list:
                c_arr = np.asarray(c_list, dtype=np.float64)
                closure[h, 0] = float(np.log1p(c_arr).sum())
                closure[h, 1] = float(np.log1p(c_arr.max()))
                closure[h, 2] = float(np.log1p((c_arr >= 1).sum()))
                closure[h, 3] = float(np.log1p((c_arr >= 2).sum()))

    return {
        "n_atoms": int(n_atoms),
        "n_bonds": int(n_bonds),
        "half_atom": half_atom,
        "half_bond": half_bond,
        "half_deg": half_deg,
        "half_of_bond": half_of,
        "partner_ptr": partner_ptr,
        "partner_half": np.asarray(partner_half, dtype=np.int64),
        "partner_rel": np.asarray(partner_rel, dtype=np.int64),
        "closure": closure,
        "block_of": block_of,
    }


# ---------------------------------------------------------------------------
# whole split
# ---------------------------------------------------------------------------


def build_relation_cache(cache: Mapping[str, Any]) -> dict:
    """Relation arrays for every molecule of a frozen split cache."""
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    n = len(ap) - 1

    half_ptr = np.zeros(n + 1, dtype=np.int64)
    bond_ptr = bp.astype(np.int64)
    parts: dict[str, list[np.ndarray]] = {
        "half_atom": [], "half_bond": [], "half_deg": [], "half_of_bond_local": [],
        "partner_ptr": [], "partner_half": [], "partner_rel": [], "closure": [],
        "bond_block": [],
    }
    for i in range(n):
        a0, a1 = int(ap[i]), int(ap[i + 1])
        b0, b1 = int(bp[i]), int(bp[i + 1])
        r = relation_arrays(a1 - a0, bi[b0:b1], bt[b0:b1])
        half_ptr[i + 1] = half_ptr[i] + 2 * (b1 - b0)
        parts["half_atom"].append(r["half_atom"] + a0)
        parts["half_bond"].append(r["half_bond"] + b0)
        parts["half_deg"].append(r["half_deg"])
        parts["half_of_bond_local"].append(r["half_of_bond"] + (half_ptr[i]))
        if not parts["partner_ptr"]:
            parts["partner_ptr"].append(r["partner_ptr"])
        else:
            # drop the duplicated boundary element of each per-molecule ptr
            parts["partner_ptr"].append(r["partner_ptr"][1:] + parts["partner_ptr"][-1][-1])
        parts["partner_half"].append(r["partner_half"] + half_ptr[i])
        parts["partner_rel"].append(r["partner_rel"])
        parts["closure"].append(r["closure"])
        parts["bond_block"].append(r["block_of"])

    out = {
        "half_ptr": half_ptr,
        "n_molecules": n,
    }
    out["half_atom_raw"] = np.concatenate(parts["half_atom"]) if n else np.zeros(0, np.int64)
    out["half_bond_raw"] = np.concatenate(parts["half_bond"]) if n else np.zeros(0, np.int64)
    out["half_deg"] = np.concatenate(parts["half_deg"]) if n else np.zeros(0, np.int64)
    out["half_of_bond"] = np.concatenate(parts["half_of_bond_local"]) if n else np.zeros((0, 2), np.int64)
    out["partner_ptr"] = np.concatenate(parts["partner_ptr"]) if n else np.zeros(1, np.int64)
    out["partner_half_raw"] = np.concatenate(parts["partner_half"]) if n else np.zeros(0, np.int64)
    out["partner_rel"] = np.concatenate(parts["partner_rel"]) if n else np.zeros(0, np.int64)
    out["closure"] = np.concatenate(parts["closure"]) if n else np.zeros((0, CLOSURE_DIM), np.float32)
    return out


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def relation_content_hash(rel: Mapping[str, Any]) -> str:
    """Machine-independent hash of the relation cache content."""
    import hashlib

    h = hashlib.sha256()
    for key in sorted(rel):
        value = rel[key]
        h.update(key.encode())
        if isinstance(value, np.ndarray):
            h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
        else:
            h.update(str(value).encode())
    return h.hexdigest()


def _ranges(ptr: np.ndarray, rows: np.ndarray):
    starts = np.asarray(ptr, dtype=np.int64)[rows]
    stops = np.asarray(ptr, dtype=np.int64)[rows + 1]
    counts = stops - starts
    idx = np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts)]) \
        if len(rows) else np.zeros(0, dtype=np.int64)
    batch = np.repeat(np.arange(len(rows)), counts).astype(np.int64)
    offs = np.zeros(len(rows), dtype=np.int64)
    if len(rows) > 1:
        offs[1:] = np.cumsum(counts)[:-1]
    return idx, batch, offs, counts


def collate(cache: Mapping[str, Any], rel: Mapping[str, Any],
            rows: Sequence[int]) -> dict:
    """Assemble one batch.  Pure index arithmetic, one numpy pass."""
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)
    atom_idx, atom_batch, atom_off, _ = _ranges(np.asarray(cache["atom_ptr"]), rows)
    bond_idx, bond_batch, bond_off, _ = _ranges(np.asarray(cache["bond_ptr"]), rows)
    half_idx, half_batch, half_off, _ = _ranges(np.asarray(rel["half_ptr"]), rows)

    bi = np.asarray(cache["bond_index"], dtype=np.int64)[bond_idx] + atom_off[bond_batch].reshape(-1, 1)
    half_atom = np.asarray(rel["half_atom_raw"], dtype=np.int64)[half_idx]
    half_atom = half_atom - np.asarray(cache["atom_ptr"], dtype=np.int64)[rows][half_batch] \
        + atom_off[half_batch]
    half_bond = np.asarray(rel["half_bond_raw"], dtype=np.int64)[half_idx]
    half_bond = half_bond - np.asarray(cache["bond_ptr"], dtype=np.int64)[rows][half_batch] \
        + bond_off[half_batch]
    half_of_bond = np.asarray(rel["half_of_bond"], dtype=np.int64)[bond_idx]
    half_of_bond = half_of_bond - np.asarray(rel["half_ptr"], dtype=np.int64)[rows][bond_batch].reshape(-1, 1) \
        + half_off[bond_batch].reshape(-1, 1)

    hp = np.asarray(rel["partner_ptr"], dtype=np.int64)
    hp_counts = np.diff(hp)[half_idx]
    if len(half_idx):
        partner_rows = np.concatenate([np.arange(hp[h], hp[h + 1]) for h in half_idx])
    else:
        partner_rows = np.zeros(0, dtype=np.int64)
    half_local = half_idx - np.asarray(rel["half_ptr"], dtype=np.int64)[rows][half_batch] \
        + half_off[half_batch]
    tgt_full = np.repeat(half_local, hp_counts)
    partner_batch = np.repeat(half_batch, hp_counts)
    assert len(tgt_full) == len(partner_rows)
    partner_half = np.asarray(rel["partner_half_raw"], dtype=np.int64)[partner_rows] \
        - np.asarray(rel["half_ptr"], dtype=np.int64)[rows][partner_batch] + half_off[partner_batch]

    def T(x, dtype=None):
        t = torch.from_numpy(np.ascontiguousarray(x))
        if dtype is not None:
            t = t.to(dtype)
        return t

    return {
        "n_graphs": B,
        "rows": T(rows),
        "atom_feat": torch.from_numpy(np.asarray(cache["atom_feat"])[atom_idx].astype(np.float32)),
        "atom_batch": T(atom_batch),
        "bond_feat": torch.from_numpy(np.asarray(cache["bond_feat"])[bond_idx].astype(np.float32)),
        "bond_type": T(np.asarray(cache["bond_type"])[bond_idx].astype(np.int64)),
        "bond_batch": T(bond_batch),
        "bond_index": T(bi),
        "half_atom": T(half_atom),
        "half_bond": T(half_bond),
        "half_deg": T(np.asarray(rel["half_deg"])[half_idx].astype(np.float32)),
        "half_of_bond": T(half_of_bond),
        "partner_half": T(partner_half),
        "partner_tgt": T(tgt_full),
        "partner_rel": T(np.asarray(rel["partner_rel"])[partner_rows].astype(np.int64)),
        "closure": torch.from_numpy(np.asarray(rel["closure"])[half_idx].astype(np.float32)),
        "global": torch.from_numpy(np.asarray(cache["global_feat"])[rows].astype(np.float32)),
        "n_half": int(len(half_idx)),
        "n_partners": int(len(partner_rows)),
    }


def batch_from_specs(specs: Sequence[Mapping[str, Any]]) -> dict:
    """Batch from raw per-molecule arrays (CPU gates / synthetic molecules).

    Each spec: ``atom_feat`` [n,108], ``bond_index`` [E,2], ``bond_feat`` [E,7],
    ``global`` [33].  Bond direction / order are used exactly as given, which
    is what makes the symmetry gates meaningful.
    """
    import torch

    atoms, bonds, bfeat, globs = [], [], [], []
    half_atom, half_bond, half_deg, half_of_bond = [], [], [], []
    partner_half, partner_rel, partner_tgt, closure = [], [], [], []
    atom_batch, bond_batch, half_batch = [], [], []
    bond_index = []
    a_off = b_off = h_off = 0
    for g, s in enumerate(specs):
        af = np.asarray(s["atom_feat"], dtype=np.float32)
        bi = np.asarray(s["bond_index"], dtype=np.int64).reshape(-1, 2)
        bf = np.asarray(s["bond_feat"], dtype=np.float32)
        r = relation_arrays(len(af), bi, np.zeros(len(bi), dtype=np.int64))
        n_h = len(r["half_atom"])
        atoms.append(af)
        bonds.append(bf)
        globs.append(np.asarray(s["global"], dtype=np.float32))
        bond_index.append(bi + a_off)
        atom_batch.append(np.full(len(af), g, dtype=np.int64))
        bond_batch.append(np.full(len(bi), g, dtype=np.int64))
        half_batch.append(np.full(n_h, g, dtype=np.int64))
        half_atom.append(r["half_atom"] + a_off)
        half_bond.append(r["half_bond"] + b_off)
        half_deg.append(r["half_deg"])
        half_of_bond.append(r["half_of_bond"] + h_off)
        partner_half.append(r["partner_half"] + h_off)
        partner_rel.append(r["partner_rel"])
        counts = np.diff(r["partner_ptr"])
        partner_tgt.append(np.repeat(np.arange(n_h) + h_off, counts))
        closure.append(r["closure"])
        a_off += len(af)
        b_off += len(bi)
        h_off += n_h

    def cat(x, dtype=None):
        t = torch.from_numpy(np.ascontiguousarray(np.concatenate(x) if len(x) else
                                                 np.zeros(0, dtype=np.int64)))
        return t.to(dtype) if dtype is not None else t

    return {
        "n_graphs": len(specs),
        "atom_feat": torch.from_numpy(np.concatenate(atoms, 0)),
        "atom_batch": cat(atom_batch),
        "bond_feat": torch.from_numpy(np.concatenate(bonds, 0)),
        "bond_type": cat([np.zeros(len(b), dtype=np.int64) for b in bonds]),
        "bond_batch": cat(bond_batch),
        "bond_index": cat(bond_index, torch.int64),
        "half_atom": cat(half_atom, torch.int64),
        "half_bond": cat(half_bond, torch.int64),
        "half_deg": cat(half_deg, torch.float32),
        "half_of_bond": cat(half_of_bond, torch.int64),
        "partner_half": cat(partner_half, torch.int64),
        "partner_tgt": cat(partner_tgt, torch.int64),
        "partner_rel": cat(partner_rel, torch.int64),
        "closure": torch.from_numpy(np.concatenate(closure, 0)),
        "global": torch.from_numpy(np.stack(globs, 0)),
        "n_half": int(sum(len(h) for h in half_atom)),
        "n_partners": int(sum(len(p) for p in partner_rel)),
    }
