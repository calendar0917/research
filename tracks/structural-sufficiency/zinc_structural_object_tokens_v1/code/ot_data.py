"""Static object enumeration + batching for B_object_tokens.

Three physically real object families, all extracted from the raw atom types
and the real physical bonds only (no labels, no y-derived quantities, no
sampling, no split-dependent truncation):

``A``   every atom of the molecule, exactly one token per atom;
``S3``  BF's rooted S3 object: root + three distinct neighbours;
``S4``  BF's rooted S4 object: root + four distinct neighbours;
``F4``  BF's rooted F4 object: neighbour ``u`` + two distinct ``w`` in
        ``N(u)\\{root}``, node set ``{root, u, w1, w2}``;
``C``   every chordless simple cycle of length 3..8 in the raw graph.

Every real object keeps its **actual atom set**, so two objects with the same
canonical content but a different atom set stay two tokens (true
multiplicity).  A ``(root, family, node set)`` triple exists once; different
families are independent.  An empty branch/cycle set is an exact empty set, no
dummy token is fabricated.

Content canonicalisation
------------------------
Branch content reuses ``bf_data.canonical_key`` exactly: the root is fixed at
slot 0 and the remaining nodes take the lexicographic-minimum permutation of
the complete typed induced adjacency (atom category per slot + pair type per
unordered pair, ``0 = no real edge``, ``1..3 = raw bond category``).

A cycle has no semantic root.  Its content is the lexicographic minimum over
all ``2L`` rotations / directions of the actual node sequence of

    atom categories (L slots) + bond types along the cycle (L slots)

where ``bond[i]`` is the real bond between sequence node ``i`` and node
``(i + 1) % L``.  The atom array and the bond array are rotated/reversed
together, so a reversed atom sequence can never be paired with a
non-reversed bond sequence.

Cached tensors are static inputs only (content, members, static relations).
No hidden state, no phi output, no label is cached.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_BF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import bf_data as BF  # noqa: E402

EXP = _HERE.parent
CACHE = EXP / "cache"

FAMILIES = BF.FAMILIES  # S3, S4, F4
FAMILY_INDEX = BF.FAMILY_INDEX
MAX_NODES = BF.MAX_NODES  # 5
N_PAIRS = BF.N_PAIRS  # 10
PAIR_J = BF.PAIR_J
MAX_MEMBERS = 8  # largest real object (chordless ring of length 8)
MAX_BRANCH_MEMBERS = 5  # S4 root + 4 neighbours
MAX_RING = 8

TOKEN_TYPES = ("A", "S3", "S4", "F4", "C")
TYPE_A, TYPE_C = 0, 4
N_TOKEN_TYPES = 5
GROUP_ATOM, GROUP_BRANCH, GROUP_CYCLE = 0, 1, 2

_CACHE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------


def enumerate_branching(obj) -> list[tuple[int, int, tuple[int, ...], bytes]]:
    """Every real ``(family, root, node set)`` object with its content key.

    Mirrors ``bf_data.enumerate_rooted_branching`` object-for-object, but keeps
    the actual member atoms.  Counts are audited against the frozen BF cache
    metadata in ``ot_checks``.
    """
    n = int(obj.n_atoms)
    cats = BF.atom_categories(obj)
    bmap = BF.bond_map(obj)
    adj = BF.adjacency(n, bmap)
    out: list[tuple[int, int, tuple[int, ...], bytes]] = []
    for root in range(n):
        nbrs = adj[root]
        node_sets: tuple[set, set, set] = (set(), set(), set())
        if len(nbrs) >= 3:
            for combo in itertools.combinations(nbrs, 3):
                node_sets[0].add(combo)
        if len(nbrs) >= 4:
            for combo in itertools.combinations(nbrs, 4):
                node_sets[1].add(combo)
        for u in nbrs:
            rest = [w for w in adj[u] if w != root]
            for combo in itertools.combinations(rest, 2):
                node_sets[2].add(tuple(sorted((u,) + combo)))
        for fi, sets in enumerate(node_sets):
            for others in sets:
                key = BF.canonical_key(cats, bmap, root, others)
                members = tuple(sorted((root,) + tuple(others)))
                out.append((fi, root, members, key))
    return out


def enumerate_cycles(obj) -> list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    """Every chordless simple cycle of length 3..8, once, with canonical content.

    Returns ``(L, members_sorted, canonical_atoms, canonical_bonds)``.  The
    enumeration is exhaustive (no sampling, no id truncation): for each minimum
    atom id ``s`` all induced paths starting at ``s`` are followed and closed
    back to ``s``.  Every chordless cycle has a unique node set, so the node-set
    set removes the two directions exactly once.
    """
    n = int(obj.n_atoms)
    cats = BF.atom_categories(obj)
    bmap = BF.bond_map(obj)
    adj = BF.adjacency(n, bmap)
    found: dict[int, dict[tuple[int, ...], tuple[int, ...]]] = {
        L: {} for L in range(3, MAX_RING + 1)
    }
    for s in range(n):
        stack: list[tuple[int, list[int]]] = [(s, [s])]
        while stack:
            last, path = stack.pop()
            L = len(path)
            if L >= 3 and (last, s) in bmap:
                # chordless: s may only touch path[1] and the closing node
                if not any((s, x) in bmap for x in path[2:-1]):
                    nodes = tuple(sorted(path))
                    found[L].setdefault(nodes, tuple(path))
            if L >= MAX_RING:
                continue
            for w in adj[last]:
                if w <= s or w in path:
                    continue
                # induced path: the new node may only touch the last node
                # (the start s is the future closing edge and is allowed)
                if any((w, x) in bmap for x in path[1:-1]):
                    continue
                stack.append((w, path + [w]))
    out = []
    for L in range(3, MAX_RING + 1):
        for nodes, order in found[L].items():
            atoms, bonds = canonical_cycle(cats, bmap, list(order))
            out.append((L, nodes, atoms, bonds))
    return out


def canonical_cycle(cats, bmap, order):
    """Lexicographic-min typed content over rotations and both directions."""
    L = len(order)
    best = None
    for direction in (0, 1):
        base = list(order) if direction == 0 else list(reversed(order))
        for r in range(L):
            seq = base[r:] + base[:r]
            atoms = tuple(int(cats[v]) for v in seq)
            bonds = tuple(
                int(bmap[(seq[i], seq[(i + 1) % L])]) for i in range(L)
            )
            key = atoms + bonds
            if best is None or key < best[0]:
                best = (key, atoms, bonds)
    return best[1], best[2]


def atom_graph_matrices(obj):
    """Label-free ``n x n`` hop distance and direct-bond-type matrices."""
    n = int(obj.n_atoms)
    bmap = BF.bond_map(obj)
    adj = BF.adjacency(n, bmap)
    big = np.int16(32767)
    dist = np.full((n, n), big, dtype=np.int16)
    for s in range(n):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if dist[s, v] == big:
                        dist[s, v] = d
                        nxt.append(v)
            frontier = nxt
    bond = np.zeros((n, n), dtype=np.uint8)
    for (u, v), t in bmap.items():
        bond[u, v] = np.uint8(t)
        bond[v, u] = np.uint8(t)
    return dist, bond


# ---------------------------------------------------------------------------
# cache construction
# ---------------------------------------------------------------------------


def cache_path(split: str) -> Path:
    return CACHE / f"ot_{split}.pt"


def build_from_objects(objs, split: str, source: dict, *, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    n_mol = len(objs)

    tok_type: list[np.ndarray] = []
    tok_root: list[np.ndarray] = []
    tok_members = np.zeros((0, MAX_MEMBERS), dtype=np.uint8)
    tok_nmembers: list[np.ndarray] = []
    tok_family: list[np.ndarray] = []
    tok_atoms = np.zeros((0, MAX_NODES), dtype=np.uint8)
    tok_pairs = np.zeros((0, N_PAIRS), dtype=np.uint8)
    ring_atoms = np.zeros((0, MAX_RING), dtype=np.uint8)
    ring_bonds = np.zeros((0, MAX_RING), dtype=np.uint8)
    ring_len: list[np.ndarray] = []
    tok_content: list[np.ndarray] = []

    br_keys: dict[tuple[int, bytes], int] = {}
    br_rows: list[tuple[int, bytes, int]] = []  # (family, key, pattern length)
    ring_keys: dict[bytes, int] = {}
    ring_rows: list[bytes] = []

    tok_ptr = np.zeros((n_mol + 1,), dtype=np.int64)
    n_atoms_mol = np.zeros((n_mol,), dtype=np.int32)
    n_branch_mol = np.zeros((n_mol,), dtype=np.int32)
    n_cycle_mol = np.zeros((n_mol,), dtype=np.int32)
    dist_ptr = np.zeros((n_mol + 1,), dtype=np.int64)
    dist_parts: list[np.ndarray] = []
    bond_parts: list[np.ndarray] = []
    cycle_by_len = {L: 0 for L in range(3, MAX_RING + 1)}
    branch_by_family = {f: 0 for f in FAMILIES}

    for m, obj in enumerate(objs):
        n = int(obj.n_atoms)
        n_atoms_mol[m] = n
        branches = enumerate_branching(obj)
        cycles = enumerate_cycles(obj)
        n_branch_mol[m] = len(branches)
        n_cycle_mol[m] = len(cycles)

        # --- atoms -------------------------------------------------------
        typ = np.zeros((n,), dtype=np.uint8)
        root = np.arange(n, dtype=np.int16)
        members = np.zeros((n, MAX_MEMBERS), dtype=np.uint8)
        members[:, 0] = np.arange(n, dtype=np.uint8)
        nmem = np.ones((n,), dtype=np.uint8)
        fam = np.full((n,), 255, dtype=np.uint8)
        atoms = np.zeros((n, MAX_NODES), dtype=np.uint8)
        pairs = np.zeros((n, N_PAIRS), dtype=np.uint8)
        r_atoms = np.zeros((n, MAX_RING), dtype=np.uint8)
        r_bonds = np.zeros((n, MAX_RING), dtype=np.uint8)
        r_len = np.zeros((n,), dtype=np.uint8)
        content = np.zeros((n,), dtype=np.int16)

        # --- branches ----------------------------------------------------
        if branches:
            nb = len(branches)
            b_typ = np.zeros((nb,), dtype=np.uint8)
            b_root = np.zeros((nb,), dtype=np.int16)
            b_mem = np.zeros((nb, MAX_BRANCH_MEMBERS), dtype=np.uint8)
            b_nmem = np.zeros((nb,), dtype=np.uint8)
            b_fam = np.zeros((nb,), dtype=np.uint8)
            b_atoms = np.zeros((nb, MAX_NODES), dtype=np.uint8)
            b_pairs = np.zeros((nb, N_PAIRS), dtype=np.uint8)
            b_content = np.zeros((nb,), dtype=np.int16)
            for i, (fi, r, mem, key) in enumerate(branches):
                b_typ[i] = fi + 1
                b_root[i] = r
                b_mem[i, : len(mem)] = mem
                b_nmem[i] = len(mem)
                b_fam[i] = fi
                raw = np.frombuffer(key, dtype=np.uint8)
                b_atoms[i] = raw[:MAX_NODES]
                b_pairs[i] = raw[MAX_NODES:]
                reg = (fi, key)
                pid = br_keys.get(reg)
                if pid is None:
                    pid = len(br_keys)
                    br_keys[reg] = pid
                    br_rows.append((fi, key, 4 if fi != FAMILY_INDEX["S4"] else 5))
                b_content[i] = pid
                branch_by_family[FAMILIES[fi]] += 1
        else:
            b_typ = np.zeros((0,), np.uint8)
            b_root = np.zeros((0,), np.int16)
            b_mem = np.zeros((0, MAX_BRANCH_MEMBERS), np.uint8)
            b_nmem = np.zeros((0,), np.uint8)
            b_fam = np.zeros((0,), np.uint8)
            b_atoms = np.zeros((0, MAX_NODES), np.uint8)
            b_pairs = np.zeros((0, N_PAIRS), np.uint8)
            b_content = np.zeros((0,), np.int16)

        # --- cycles ------------------------------------------------------
        if cycles:
            nc = len(cycles)
            c_mem = np.zeros((nc, MAX_MEMBERS), dtype=np.uint8)
            c_nmem = np.zeros((nc,), dtype=np.uint8)
            c_atoms = np.zeros((nc, MAX_RING), dtype=np.uint8)
            c_bonds = np.zeros((nc, MAX_RING), dtype=np.uint8)
            c_len = np.zeros((nc,), dtype=np.uint8)
            c_content = np.zeros((nc,), dtype=np.int16)
            for i, (L, mem, at, bt) in enumerate(cycles):
                c_mem[i, :L] = mem
                c_nmem[i] = L
                c_atoms[i, :L] = at
                c_bonds[i, :L] = bt
                c_len[i] = L
                key = bytes(at) + bytes(bt) + bytes([L])
                pid = ring_keys.get(key)
                if pid is None:
                    pid = len(ring_keys)
                    ring_keys[key] = pid
                    ring_rows.append(key)
                c_content[i] = pid
                cycle_by_len[L] += 1
        else:
            c_mem = np.zeros((0, MAX_MEMBERS), np.uint8)
            c_nmem = np.zeros((0,), np.uint8)
            c_atoms = np.zeros((0, MAX_RING), np.uint8)
            c_bonds = np.zeros((0, MAX_RING), np.uint8)
            c_len = np.zeros((0,), np.uint8)
            c_content = np.zeros((0,), np.int16)

        tok_type.append(np.concatenate([typ, b_typ, np.full((len(cycles),), TYPE_C, np.uint8)]))
        tok_root.append(np.concatenate([root, b_root, np.full((len(cycles),), -1, np.int16)]))
        b_mem_full = np.zeros((len(branches), MAX_MEMBERS), dtype=np.uint8)
        b_mem_full[:, :MAX_BRANCH_MEMBERS] = b_mem
        tok_members = np.concatenate([tok_members, members, b_mem_full, c_mem], axis=0)
        tok_nmembers.append(np.concatenate([nmem, b_nmem, c_nmem]))
        tok_family.append(np.concatenate([fam, b_fam, np.full((len(cycles),), 255, np.uint8)]))
        tok_atoms = np.concatenate([tok_atoms, atoms, b_atoms, np.zeros((len(cycles), MAX_NODES), np.uint8)])
        tok_pairs = np.concatenate([tok_pairs, pairs, b_pairs, np.zeros((len(cycles), N_PAIRS), np.uint8)])
        ring_atoms = np.concatenate([ring_atoms, r_atoms, np.zeros((len(branches), MAX_RING), np.uint8), c_atoms])
        ring_bonds = np.concatenate([ring_bonds, r_bonds, np.zeros((len(branches), MAX_RING), np.uint8), c_bonds])
        ring_len.append(np.concatenate([r_len, np.zeros((len(branches),), np.uint8), c_len]))
        tok_content.append(np.concatenate([content, b_content, c_content]))

        tok_ptr[m + 1] = tok_ptr[m] + n + len(branches) + len(cycles)

        dist, bond = atom_graph_matrices(obj)
        dist_parts.append(dist.reshape(-1))
        bond_parts.append(bond.reshape(-1))
        dist_ptr[m + 1] = dist_ptr[m] + n * n
        if verbose and (m + 1) % 2000 == 0:
            C.log(f"[ot-prep {split}] {m + 1}/{n_mol} molecules {time.perf_counter() - t0:.1f}s")

    pat_atom = np.zeros((len(br_rows), MAX_NODES), dtype=np.uint8)
    pat_pair = np.zeros((len(br_rows), N_PAIRS), dtype=np.uint8)
    pat_family = np.zeros((len(br_rows),), dtype=np.uint8)
    pat_len = np.zeros((len(br_rows),), dtype=np.uint8)
    for pid, (fam, key, length) in enumerate(br_rows):
        raw = np.frombuffer(key, dtype=np.uint8)
        pat_atom[pid] = raw[:MAX_NODES]
        pat_pair[pid] = raw[MAX_NODES:]
        pat_len[pid] = length
        # families are independent (BF keeps one table per family), so the
        # content address is (family, typed content)
        pat_family[pid] = fam

    r_atoms = np.zeros((len(ring_rows), MAX_RING), dtype=np.uint8)
    r_bonds = np.zeros((len(ring_rows), MAX_RING), dtype=np.uint8)
    r_length = np.zeros((len(ring_rows),), dtype=np.uint8)
    for pid, key in enumerate(ring_rows):
        raw = np.frombuffer(key, dtype=np.uint8)
        L = int(raw[-1])
        r_atoms[pid, :L] = raw[:L]
        r_bonds[pid, :L] = raw[L:2 * L]
        r_length[pid] = L

    n_tokens = int(tok_ptr[-1])
    data = {
        "split": split,
        "families": FAMILIES,
        "token_types": TOKEN_TYPES,
        "max_nodes": MAX_NODES,
        "n_pairs": N_PAIRS,
        "max_members": MAX_MEMBERS,
        "max_branch_members": MAX_BRANCH_MEMBERS,
        "max_ring": MAX_RING,
        "tok_type": torch.from_numpy(np.concatenate(tok_type)),
        "tok_root": torch.from_numpy(np.concatenate(tok_root).astype(np.int64)),
        "tok_members": torch.from_numpy(tok_members.astype(np.int64)),
        "tok_nmembers": torch.from_numpy(np.concatenate(tok_nmembers).astype(np.int64)),
        "tok_family": torch.from_numpy(np.concatenate(tok_family).astype(np.int64)),
        "tok_atoms": torch.from_numpy(tok_atoms.astype(np.int64)),
        "tok_pairs": torch.from_numpy(tok_pairs.astype(np.int64)),
        "ring_atoms": torch.from_numpy(ring_atoms.astype(np.int64)),
        "ring_bonds": torch.from_numpy(ring_bonds.astype(np.int64)),
        "ring_len": torch.from_numpy(np.concatenate(ring_len).astype(np.int64)),
        "tok_content": torch.from_numpy(np.concatenate(tok_content).astype(np.int64)),
        "br_pat_atom": torch.from_numpy(pat_atom.astype(np.int64)),
        "br_pat_pair": torch.from_numpy(pat_pair.astype(np.int64)),
        "br_pat_family": torch.from_numpy(pat_family.astype(np.int64)),
        "br_pat_len": torch.from_numpy(pat_len.astype(np.int64)),
        "ring_pat_atom": torch.from_numpy(r_atoms.astype(np.int64)),
        "ring_pat_bond": torch.from_numpy(r_bonds.astype(np.int64)),
        "ring_pat_len": torch.from_numpy(r_length.astype(np.int64)),
        "tok_ptr": torch.from_numpy(tok_ptr),
        "n_atoms_per_mol": torch.from_numpy(n_atoms_mol),
        "n_branch_per_mol": torch.from_numpy(n_branch_mol),
        "n_cycle_per_mol": torch.from_numpy(n_cycle_mol),
        "dist_ptr": torch.from_numpy(dist_ptr),
        "dist_flat": torch.from_numpy(np.concatenate(dist_parts).astype(np.int16)),
        "bond_flat": torch.from_numpy(np.concatenate(bond_parts).astype(np.uint8)),
        "meta": {
            "split": split,
            "object_source": source,
            "n_molecules": int(n_mol),
            "n_atoms": int(n_atoms_mol.sum()),
            "n_tokens": int(n_tokens),
            "n_branch_tokens": int(n_branch_mol.sum()),
            "n_cycle_tokens": int(n_cycle_mol.sum()),
            "branch_by_family": {f: int(branch_by_family[f]) for f in FAMILIES},
            "branch_unique_contents": int(len(br_rows)),
            "cycle_by_length": {str(L): int(cycle_by_len[L]) for L in range(3, MAX_RING + 1)},
            "cycle_unique_contents": int(len(ring_rows)),
            "max_tokens_per_molecule": int((tok_ptr[1:] - tok_ptr[:-1]).max()),
            "max_atoms_per_molecule": int(n_atoms_mol.max()),
            "mean_atoms_per_molecule": float(n_atoms_mol.mean()),
            "mean_branch_tokens_per_molecule": float(n_branch_mol.mean()),
            "mean_cycle_tokens_per_molecule": float(n_cycle_mol.mean()),
            "definition": (
                "A: every atom; S3/S4/F4: BF rooted branching objects kept per "
                "actual node set; C: every chordless simple cycle of length 3..8"
            ),
            "cycle_definition": (
                "chordless = no extra edge between non-adjacent cycle nodes; "
                "canonical content = lexicographic-min over 2L rotations/directions "
                "with the bond array rotated/reversed together"
            ),
            "multiplicity": (
                "same canonical content on different actual node sets stays a "
                "separate token; (root, family, node set) exists once"
            ),
            "canonicalisation": "BF canonical_key for branches; 2L min for cycles",
            "labels_used": False,
            "test_used": split == "test",
            "created": C.utcnow(),
            "prep_seconds": float(time.perf_counter() - t0),
        },
    }
    return data


def build_split(split: str, *, verbose: bool = True) -> dict:
    objs, source = D.load_p3_objects(split)
    return build_from_objects(objs, split, source, verbose=verbose)


def save_split(split: str) -> Path:
    data = build_split(split)
    path = cache_path(split)
    C.atomic_torch_save(data, path)
    C.log(f"[ot-prep] wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    C.log(f"[ot-prep] meta {data['meta']}")
    return path


def load_cache(split: str) -> dict:
    if split not in _CACHE:
        path = cache_path(split)
        if not path.exists():
            raise FileNotFoundError(f"missing object cache {path}; run `ot_data.py prep`")
        _CACHE[split] = torch.load(path, map_location="cpu", weights_only=False)
    return _CACHE[split]


def register_cache(split: str, data: dict) -> None:
    _CACHE[split] = data


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "audit"])
    p.add_argument("--split", action="append", default=None)
    args = p.parse_args(argv)
    splits = args.split or ["train", "valid"]
    if args.command == "prep":
        CACHE.mkdir(parents=True, exist_ok=True)
        for split in splits:
            save_split(split)
    else:
        for split in splits:
            C.log(load_cache(split)["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def collate_ot(cache, rows, candidate: str = "OT", *, valid: bool = False) -> dict:
    """Batch the ad cache plus the static object-token structure.

    The batch keeps, per molecule, ``T = n_atoms + n_branch + n_cycle`` real
    tokens padded to the batch ``Tmax``.  Everything the model needs for the
    static relation bias is assembled here from cached static arrays; no label,
    prediction or training hidden state is involved.
    """
    import ad_train as T1

    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)
    pc = load_cache(str(cache["split"]))

    atom_ptr = np.asarray(cache["atom_ptr"])
    atom_counts = np.diff(atom_ptr)[rows].astype(np.int64)
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64)
    atom_idx = T1._ranges(atom_ptr, rows)[0]
    # B only needs raw atom descriptors and the global vector; nothing from the
    # P3 bond / junction object channels is built or batched here.
    base = {
        "n_graphs": B,
        "atom_feat": torch.from_numpy(
            np.asarray(cache["atom_feat"])[atom_idx].astype(np.float32)),
        "global": torch.from_numpy(
            np.asarray(cache["global_feat"])[rows].astype(np.float32)),
    }

    tok_ptr = np.asarray(pc["tok_ptr"])
    tok_counts = (tok_ptr[rows + 1] - tok_ptr[rows]).astype(np.int64)
    Tmax = int(tok_counts.max(initial=0))
    Nmax = int(atom_counts.max(initial=0))
    tok_off = np.concatenate([[0], np.cumsum(tok_counts)[:-1]]).astype(np.int64)
    tok_idx = np.concatenate(
        [np.arange(int(s), int(e), dtype=np.int64) for s, e in zip(tok_ptr[rows], tok_ptr[rows + 1])]
    ) if int(tok_counts.sum()) else np.zeros((0,), dtype=np.int64)
    tok_batch = np.repeat(np.arange(B, dtype=np.int64), tok_counts)
    tok_local = T1._ranges(tok_ptr, rows)[2] if int(tok_counts.sum()) else np.zeros((0,), dtype=np.int64)

    # ---- padded static token descriptors --------------------------------
    flat = tok_batch * Tmax + tok_local
    n_flat = B * Tmax
    tok_type_pad = np.zeros((n_flat,), dtype=np.int64)
    tok_root_pad = np.full((n_flat,), -1, dtype=np.int64)
    tok_members_pad = np.zeros((n_flat, MAX_MEMBERS), dtype=np.int64)
    tok_member_mask = np.zeros((n_flat, MAX_MEMBERS), dtype=bool)
    tok_valid = np.zeros((n_flat,), dtype=bool)

    typ = np.asarray(pc["tok_type"])[tok_idx].astype(np.int64)
    rt = np.asarray(pc["tok_root"])[tok_idx].astype(np.int64)
    mem = np.asarray(pc["tok_members"])[tok_idx].astype(np.int64)
    nmem = np.asarray(pc["tok_nmembers"])[tok_idx].astype(np.int64)
    valid_tok = np.ones((len(tok_idx),), dtype=bool)
    tok_type_pad[flat] = typ
    tok_root_pad[flat] = rt
    tok_members_pad[flat] = mem
    for k in range(MAX_MEMBERS):
        tok_member_mask[flat, k] = nmem > k
    tok_valid[flat] = valid_tok

    # ---- containers -----------------------------------------------------
    is_atom = typ == TYPE_A
    is_branch = (typ >= 1) & (typ <= 3)
    is_cycle = typ == TYPE_C
    atom_tok_pos = flat[is_atom]
    branch_tok_pos = flat[is_branch]
    ring_tok_pos = flat[is_cycle]

    content = np.asarray(pc["tok_content"])[tok_idx].astype(np.int64)
    br_pat = np.where(is_branch, content, 0)
    ring_pat = np.where(is_cycle, content, 0)
    br_uniq, br_inv = np.unique(br_pat[is_branch], return_inverse=True) if is_branch.any() else (
        np.zeros((0,), np.int64), np.zeros((0,), np.int64))
    r_uniq, r_inv = np.unique(ring_pat[is_cycle], return_inverse=True) if is_cycle.any() else (
        np.zeros((0,), np.int64), np.zeros((0,), np.int64))

    fam = np.asarray(pc["tok_family"])[tok_idx][is_branch].astype(np.int64)
    b_members_local = mem[is_branch][:, :MAX_BRANCH_MEMBERS]
    r_members_local = mem[is_cycle]
    r_len = np.asarray(pc["ring_len"])[tok_idx][is_cycle].astype(np.int64)

    # global atom indices (batch-local) for pooling the live atom encoder
    b_batch = tok_batch[is_branch]
    r_batch = tok_batch[is_cycle]
    b_members_global = np.where(
        np.arange(MAX_BRANCH_MEMBERS)[None, :] < nmem[is_branch][:, None],
        b_members_local + atom_off[b_batch][:, None], 0,
    )
    r_members_global = np.where(
        np.arange(MAX_MEMBERS)[None, :] < r_len[:, None],
        r_members_local + atom_off[r_batch][:, None], 0,
    )
    a_batch = tok_batch[is_atom]
    atom_tok_global = atom_off[a_batch] + np.arange(len(atom_tok_pos), dtype=np.int64)  # not used; see below

    # ---- static relation inputs ----------------------------------------
    dist_ptr = np.asarray(pc["dist_ptr"])
    dist_flat = np.asarray(pc["dist_flat"])
    bond_flat = np.asarray(pc["bond_flat"])
    dist_pad = np.zeros((B, Nmax, Nmax), dtype=np.int64)
    bond_pad = np.zeros((B, Nmax, Nmax), dtype=np.int64)
    for i, row in enumerate(rows):
        n = int(atom_counts[i])
        s, e = int(dist_ptr[row]), int(dist_ptr[row + 1])
        dist_pad[i, :n, :n] = dist_flat[s:e].reshape(n, n)
        bond_pad[i, :n, :n] = bond_flat[s:e].reshape(n, n)

    base.update({
        "n_graphs": B,
        "Nmax": Nmax,
        "Tmax": Tmax,
        "n_tokens_total": int(len(tok_idx)),
        "tok_type_pad": torch.from_numpy(tok_type_pad),
        "tok_root_pad": torch.from_numpy(tok_root_pad),
        "tok_members_pad": torch.from_numpy(tok_members_pad),
        "tok_member_mask": torch.from_numpy(tok_member_mask),
        "tok_valid": torch.from_numpy(tok_valid),
        "atom_dist": torch.from_numpy(dist_pad),
        "atom_bond": torch.from_numpy(bond_pad),
        "atom_tok_pos": torch.from_numpy(atom_tok_pos),
        "branch_tok_pos": torch.from_numpy(branch_tok_pos),
        "branch_root": torch.from_numpy(
            (rt[is_branch] + atom_off[b_batch]).astype(np.int64) if is_branch.any()
            else np.zeros((0,), np.int64)),
        "branch_members": torch.from_numpy(np.ascontiguousarray(b_members_global)),
        "branch_nmembers": torch.from_numpy(nmem[is_branch].astype(np.int64)),
        "branch_family": torch.from_numpy(fam),
        "branch_pat": torch.from_numpy(br_inv.astype(np.int64)),
        "ring_tok_pos": torch.from_numpy(ring_tok_pos),
        "ring_members": torch.from_numpy(np.ascontiguousarray(r_members_global)),
        "ring_len": torch.from_numpy(r_len),
        "ring_pat": torch.from_numpy(r_inv.astype(np.int64)),
        "br_pat_atom": torch.from_numpy(np.asarray(pc["br_pat_atom"])[br_uniq].astype(np.int64)),
        "br_pat_pair": torch.from_numpy(np.asarray(pc["br_pat_pair"])[br_uniq].astype(np.int64)),
        "br_pat_family": torch.from_numpy(np.asarray(pc["br_pat_family"])[br_uniq].astype(np.int64)),
        "br_pat_len": torch.from_numpy(np.asarray(pc["br_pat_len"])[br_uniq].astype(np.int64)),
        "ring_pat_atom": torch.from_numpy(np.asarray(pc["ring_pat_atom"])[r_uniq].astype(np.int64)),
        "ring_pat_bond": torch.from_numpy(np.asarray(pc["ring_pat_bond"])[r_uniq].astype(np.int64)),
        "ring_pat_len": torch.from_numpy(np.asarray(pc["ring_pat_len"])[r_uniq].astype(np.int64)),
        "token_batch": torch.from_numpy(tok_batch),
        "token_type": torch.from_numpy(typ),
        "group_counts": torch.from_numpy(np.stack([
            atom_counts,
            np.asarray(pc["n_branch_per_mol"])[rows].astype(np.int64),
            np.asarray(pc["n_cycle_per_mol"])[rows].astype(np.int64),
        ], axis=1).astype(np.float32)),
    })
    del atom_tok_global
    return base
