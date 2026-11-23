"""Root-conditioned connected structure words.

For every atom ``v`` of a molecule the *root context* is the full radius-2
induced subgraph

    R2(v) = {t : dist(v, t) <= 2} with every real induced bond.

A *root-conditioned structure word* is a distinct atom support ``S`` with

    v in S,  1 <= |S| <= 4,  all atoms of S within distance 2 of v,
    G[S] connected,  and every real bond with both endpoints in S.

The internal bond set of ``S`` is exactly "all real bonds whose two endpoints
are both in ``S``" (a triangle's closing bond is therefore always read).  The
support is stored with the root in slot 0 and the remaining atoms in ascending
molecule-local order, so ``atoms[:, 0]`` *is* the root.

For the Mobius recursion this module also precomputes, for every support of size
k >= 3, the indices of its connected root-containing proper subsets.  Subsets
that are disconnected are not stored: their ``kappa`` is provably zero and the
recursion only needs non-zero support.

Atom ids are molecule-local enumeration keys only; they are never features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

MAX_SIZE = 4
MAX_BONDS = 6


@dataclass
class MoleculeStruct:
    row: int
    n_atoms: int
    n_bonds: int
    atom_feat: np.ndarray                  # [n, 9]
    bond_feat: np.ndarray                  # [m, 3]
    bond_u: np.ndarray                     # [m]
    bond_v: np.ndarray                     # [m]
    graph_scalars: np.ndarray              # [8]
    atoms: dict[int, np.ndarray]           # k -> [nk, 4] int32, root in slot 0, -1 pad
    bonds: dict[int, np.ndarray]           # k -> [nk, 6] int32 local bond ids, -1 pad
    bslot: dict[int, np.ndarray]           # k -> [nk, 6, 2] int8 slot pairs, -1 pad
    sub2: dict[int, np.ndarray]            # k=3 -> [n3, 2]; k=4 -> [n4, 3] int32
    sub3: dict[int, np.ndarray]            # k=4 -> [n4, 3] int32
    word_type: dict[int, np.ndarray] = field(default_factory=dict)   # filled by analysis

    @property
    def n_words(self) -> int:
        return int(sum(self.atoms[k].shape[0] for k in self.atoms))

    def counts(self) -> dict[int, int]:
        return {k: int(self.atoms[k].shape[0]) for k in sorted(self.atoms)}


def _adjacency(n_atoms: int, n_bonds: int, bond_u: np.ndarray,
               bond_v: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(int(n_atoms))]
    bond_of: dict[tuple[int, int], int] = {}
    for bid in range(int(n_bonds)):
        u = int(bond_u[bid])
        v = int(bond_v[bid])
        adj[u].append((v, bid))
        adj[v].append((u, bid))
        bond_of[(min(u, v), max(u, v))] = bid
    return adj, bond_of


def _bfs_dist(n_atoms: int, adj: Sequence[Sequence[tuple[int, int]]], source: int) -> np.ndarray:
    dist = np.full(int(n_atoms), -1, np.int16)
    dist[source] = 0
    frontier = [int(source)]
    d = 0
    while frontier:
        d += 1
        nxt: list[int] = []
        for x in frontier:
            for y, _ in adj[x]:
                if dist[y] < 0:
                    dist[y] = d
                    nxt.append(int(y))
        frontier = nxt
    return dist


def _enumerate_connected(root: int, dist: np.ndarray, n_atoms: int,
                         adj: Sequence[Sequence[tuple[int, int]]]) -> dict[int, list[tuple[int, ...]]]:
    """All connected supports of size 1..4 containing ``root`` inside R2(root)."""
    out: dict[int, set[tuple[int, ...]]] = {1: {(int(root),)}, 2: set(), 3: set(), 4: set()}
    stack: list[tuple[int, ...]] = [(int(root),)]
    allowed = [bool(0 <= dist[i] <= 2) for i in range(int(n_atoms))]
    while stack:
        cur = stack.pop()
        size = len(cur)
        if size >= 2:
            out[size].add(tuple(sorted(cur)))
        if size >= MAX_SIZE:
            continue
        present = set(cur)
        for x in cur:
            for y, _ in adj[x]:
                if allowed[y] and y not in present:
                    stack.append(tuple(sorted(cur + (int(y),))))
    return {k: sorted(v) for k, v in out.items()}


def build_molecule(mol: Mapping[str, Any]) -> MoleculeStruct:
    n_atoms = int(mol["n_atoms"])
    n_bonds = int(mol["n_bonds"])
    bond_u = np.asarray(mol["bond_u"], np.int64)
    bond_v = np.asarray(mol["bond_v"], np.int64)
    bond_feat = np.asarray(mol["bond_feat"], np.int64).reshape(n_bonds, 3)
    atom_feat = np.asarray(mol["atom_feat"], np.int64).reshape(n_atoms, 9)
    adj, bond_of = _adjacency(n_atoms, n_bonds, bond_u, bond_v)

    rows: dict[int, list[tuple[int, ...]]] = {2: [], 3: [], 4: []}
    slot_rows: dict[int, list[list[int]]] = {2: [], 3: [], 4: []}
    bond_rows: dict[int, list[list[int]]] = {2: [], 3: [], 4: []}
    slot_pair_rows: dict[int, list[list[int]]] = {2: [], 3: [], 4: []}
    index: dict[tuple[int, tuple[int, ...]], tuple[int, int]] = {}

    for root in range(n_atoms):
        dist = _bfs_dist(n_atoms, adj, root)
        found = _enumerate_connected(root, dist, n_atoms, adj)
        for size in (2, 3, 4):
            for support in found[size]:
                slots = (int(root),) + tuple(x for x in support if x != root)
                # internal bonds: every real bond with both endpoints inside S
                internal = []
                for i in range(size):
                    for j in range(i + 1, size):
                        key = (min(slots[i], slots[j]), max(slots[i], slots[j]))
                        bid = bond_of.get(key)
                        if bid is not None:
                            internal.append((i, j, int(bid)))
                internal.sort(key=lambda t: (t[0], t[1]))
                local = len(rows[size])
                index[(int(root), tuple(sorted(slots)))] = (size, local)
                rows[size].append(slots)
                slot_rows[size].append(list(slots))
                bond_rows[size].append([t[2] for t in internal])
                pairs: list[int] = []
                for i, j, _ in internal:
                    pairs.extend([i, j])
                slot_pair_rows[size].append(pairs)

    atoms = {}
    bonds = {}
    bslot = {}
    for size in (2, 3, 4):
        n = len(rows[size])
        a = np.full((n, MAX_SIZE), -1, np.int32)
        b = np.full((n, MAX_BONDS), -1, np.int32)
        s = np.full((n, MAX_BONDS, 2), -1, np.int8)
        for i in range(n):
            a[i, :size] = np.asarray(slot_rows[size][i], np.int32)
            bl = bond_rows[size][i]
            if bl:
                b[i, :len(bl)] = np.asarray(bl, np.int32)
            pr = slot_pair_rows[size][i]
            if pr:
                s[i, :len(pr) // 2] = np.asarray(pr, np.int8).reshape(-1, 2)
        atoms[size] = a
        bonds[size] = b
        bslot[size] = s

    # connected root-containing proper subsets, by size
    sub2 = {3: np.full((len(rows[3]), 2), -1, np.int32),
            4: np.full((len(rows[4]), 3), -1, np.int32)}
    sub3 = {4: np.full((len(rows[4]), 3), -1, np.int32)}
    for size in (3, 4):
        for i, slots in enumerate(rows[size]):
            root = slots[0]
            others = slots[1:]
            col = 0
            for a in others:
                key = (root, tuple(sorted((root, a))))
                hit = index.get(key)
                if hit is not None and hit[0] == 2:
                    sub2[size][i, col] = hit[1]
                    col += 1
            if size == 4:
                col = 0
                for a_i in range(len(others)):
                    for b_i in range(a_i + 1, len(others)):
                        key = (root, tuple(sorted((root, others[a_i], others[b_i]))))
                        hit = index.get(key)
                        if hit is not None and hit[0] == 3:
                            sub3[4][i, col] = hit[1]
                            col += 1

    return MoleculeStruct(
        row=int(mol["row"]), n_atoms=n_atoms, n_bonds=n_bonds,
        atom_feat=atom_feat, bond_feat=bond_feat, bond_u=bond_u, bond_v=bond_v,
        graph_scalars=np.asarray(mol["graph_scalars"], np.float64),
        atoms=atoms, bonds=bonds, bslot=bslot, sub2=sub2, sub3=sub3)


def build_all(bundle: Any, rows: Sequence[int]) -> dict[int, MoleculeStruct]:
    import rice_data as D

    return {int(r): build_molecule(D.molecule_arrays(bundle, int(r))) for r in rows}


def structure_stats(structs: Mapping[int, MoleculeStruct]) -> dict[str, Any]:
    n_roots = 0
    counts = {2: 0, 3: 0, 4: 0}
    roots_with = {2: 0, 3: 0, 4: 0}
    per_root_hist: dict[int, dict[int, int]] = {2: {}, 3: {}, 4: {}}
    per_mol_total = 0
    max_mol_total = 0
    n_bonds_internal = {2: 0, 3: 0, 4: 0}
    for st in structs.values():
        n_roots += st.n_atoms
        total = 0
        c = st.counts()
        for size in (2, 3, 4):
            counts[size] += c[size]
            total += c[size]
            roots_with[size] += int(np.count_nonzero(
                np.bincount(st.atoms[size][:, 0], minlength=st.n_atoms)))
            hist = per_root_hist[size]
            for k, v in zip(*np.unique(
                    np.bincount(st.atoms[size][:, 0], minlength=st.n_atoms), return_counts=True)):
                hist[int(k)] = hist.get(int(k), 0) + int(v)
            if c[size]:
                n_bonds_internal[size] += int(np.count_nonzero(st.bonds[size] >= 0))
        per_mol_total += total
        max_mol_total = max(max_mol_total, total)
    return {
        "molecules": len(structs), "roots": int(n_roots),
        "words_by_size": {str(k): int(v) for k, v in counts.items()},
        "words_total": int(sum(counts.values())),
        "roots_with_size": {str(k): int(v) for k, v in roots_with.items()},
        "root_fraction_with_size": {str(k): float(roots_with[k]) / max(1, n_roots)
                                    for k in (2, 3, 4)},
        "root_count_histogram": {str(k): {str(a): int(b) for a, b in sorted(hist.items())}
                                 for k, hist in per_root_hist.items()},
        "internal_bonds_total": {str(k): int(v) for k, v in n_bonds_internal.items()},
        "mean_words_per_molecule": float(per_mol_total) / max(1, len(structs)),
        "max_words_per_molecule": int(max_mol_total),
    }


# ---------------------------------------------------------------------------
# root-marked attributed-graph canonical certificate
# ---------------------------------------------------------------------------

def _mixed_radix(values: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    out = np.zeros(values.shape[0], np.int64)
    for i, dim in enumerate(dims):
        out = out * int(dim) + values[:, i].astype(np.int64)
    return out


_PERMS: dict[int, list[tuple[int, ...]]] = {}


def _perms_for(size: int) -> list[tuple[int, ...]]:
    if size not in _PERMS:
        import itertools

        _PERMS[size] = [(0,) + p for p in itertools.permutations(range(1, size))]
    return _PERMS[size]


_PAIR_ORDER = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def canonical_signature(st: MoleculeStruct, size: int,
                        atom_dims: Sequence[int], bond_dims: Sequence[int]
                        ) -> np.ndarray:
    """Exact canonical 10-int certificate of every root-marked word of one size.

    Layout ``[node_label(4 slots), edge_label(6 slot pairs)]`` after the
    lexicographically minimal relabelling of the non-root slots.  Node labels and
    edge labels are mixed-radix encodings of the *full* OGB categorical fields;
    an absent internal bond is ``-1``.  The certificate is a complete invariant of
    the root-marked attributed induced subgraph, so equal certificates <=> same
    root-conditioned structure word type.
    """
    atoms = st.atoms[size]
    n = int(atoms.shape[0])
    out = np.zeros((n, 10), np.int64)
    if n == 0:
        return out
    node_label = _mixed_radix(st.atom_feat, atom_dims)                     # [n_atoms]
    bond_label = _mixed_radix(st.bond_feat, bond_dims)                      # [n_bonds]
    bond_code = bond_label + 1                                             # 0 => absent
    chunk = 200_000
    for start in range(0, n, chunk):
        part = atoms[start:start + chunk]
        m = part.shape[0]
        node = np.where(part >= 0, node_label[np.abs(part)], -1)           # [m,4]
        edge = np.full((m, 4, 4), -1, np.int64)
        bl = st.bonds[size][start:start + chunk]
        sl = st.bslot[size][start:start + chunk]
        valid = bl >= 0
        ecode = np.where(valid, bond_code[np.abs(bl)], 0)
        for term in range(sl.shape[1]):
            i = sl[:, term, 0].astype(np.int64)
            j = sl[:, term, 1].astype(np.int64)
            ok = valid[:, term]
            if not ok.any():
                continue
            idx = np.nonzero(ok)[0]
            edge[idx, i[idx], j[idx]] = ecode[idx, term]
            edge[idx, j[idx], i[idx]] = ecode[idx, term]
        best = np.full((m, 10), np.iinfo(np.int64).max, np.int64)
        for perm in _perms_for(size):
            cand = np.empty((m, 10), np.int64)
            for slot in range(4):
                cand[:, slot] = node[:, perm[slot]] if slot < size else -1
            for t, (a, b) in enumerate(_PAIR_ORDER):
                if a < size and b < size:
                    cand[:, 4 + t] = edge[:, perm[a], perm[b]]
                else:
                    cand[:, 4 + t] = -1
            undecided = np.ones(m, bool)
            better = np.zeros(m, bool)
            for col in range(10):
                diff = cand[:, col] < best[:, col]
                better |= undecided & diff
                undecided &= cand[:, col] == best[:, col]
                if not undecided.any():
                    break
            best[better] = cand[better]
        out[start:start + chunk] = best
    return out


def decode_signature(row: Sequence[int], size: int, atom_dims: Sequence[int],
                     bond_dims: Sequence[int]) -> dict[str, Any]:
    """Human-readable description of one canonical certificate."""
    def unmix(value: int, dims: Sequence[int]) -> list[int]:
        digits = []
        for dim in reversed(list(dims)):
            digits.append(int(value % int(dim)))
            value //= int(dim)
        return list(reversed(digits))

    nodes = [unmix(int(row[i]), atom_dims) for i in range(size)]
    edges = []
    for t, (a, b) in enumerate(_PAIR_ORDER):
        if a < size and b < size:
            code = int(row[4 + t])
            if code >= 0:
                edges.append({"slots": [a, b],
                              "fields": unmix(code - 1, bond_dims)})
    return {"size": int(size), "node_fields": nodes, "internal_bonds": edges,
            "root_slot": 0}


def certificate_key(row: Sequence[int]) -> bytes:
    return np.asarray(row, np.int64).tobytes()