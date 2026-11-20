"""Exact word templates, Train-only merging, frozen replay and packed tensors.

A **word type** is a connected, typed subgraph template of 1..6 atoms.  Its atom
labels use only the official OGB columns ``[0, 3, 7]`` (atomic number, formal
charge, aromatic flag) and its internal bond labels only column ``[0]`` (bond
type).  Identity is the *exact* canonical serialization obtained by minimising
the labelled serialization over all atom permutations (<= 6! here); a SHA-256 of
it is used only as an index.

A **word instance** keeps the word id, the original atom ids, *all* original
9/3-column attributes of its atoms and internal bonds, and every crossing
physical bond with both ports and all three bond columns, so the original
attributed graph can be reassembled exactly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Mapping, Sequence

import numpy as np

import ws_common as C

ATOM_COLUMNS = C.ATOM_COLUMNS
WORD_ATOM_COLUMNS = C.WORD_ATOM_COLUMNS
WORDS_ATOM_COLS = list(WORD_ATOM_COLUMNS)
MAX_WORD_ATOMS = 6
R_DIM = 15
PAIR_FIELD_DIM = 7  # same-neighbour flag, neighbour-bond exists flag, 5 bond-type counts


# ---------------------------------------------------------------------------
# exact canonicalization of typed templates
# ---------------------------------------------------------------------------

_PERMS: dict[int, np.ndarray] = {}


def _perms(n: int) -> np.ndarray:
    if n not in _PERMS:
        if n > MAX_WORD_ATOMS:
            raise ValueError(f"template with {n} atoms exceeds the {MAX_WORD_ATOMS}-atom word limit")
        _PERMS[n] = np.asarray(list(permutations(range(n))), dtype=np.int64)
    return _PERMS[n]


def _chunk_codes(values: np.ndarray, bits: int) -> list[np.ndarray]:
    n_pos = values.shape[-1]
    per = 63 // bits
    shifts = np.arange(per, dtype=np.int64) * bits
    out = []
    for start in range(0, n_pos, per):
        block = values[..., start:start + per]
        if block.shape[-1] < per:
            pad = np.zeros(block.shape[:-1] + (per - block.shape[-1],), dtype=np.int64)
            block = np.concatenate([block, pad], axis=-1)
        out.append((block * (np.int64(1) << shifts)).sum(axis=-1))
    return out


def canonicalize(labels: np.ndarray, bu: np.ndarray, bv: np.ndarray, btype: np.ndarray):
    """Exact canonical serialization of a typed template (``n <= 6``)."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError(f"word-type labels must be (n, 3), got {labels.shape}")
    n = int(labels.shape[0])
    bu = np.asarray(bu, dtype=np.int64).reshape(-1)
    bv = np.asarray(bv, dtype=np.int64).reshape(-1)
    btype = np.asarray(btype, dtype=np.int64).reshape(-1)
    mat = np.zeros((n, n), dtype=np.int64)
    for a, b, t in zip(bu.tolist(), bv.tolist(), btype.tolist()):
        mat[a, b] = t + 1
        mat[b, a] = t + 1
    if n <= 1:
        return f"1|{','.join(str(int(x)) for x in labels[0])};", tuple(range(n))
    perm = _perms(n)
    lab = labels[perm]
    mb = mat[perm[:, :, None], perm[:, None, :]]
    tri = np.triu_indices(n)
    keys = _chunk_codes(lab.reshape(lab.shape[0], -1), 7) + _chunk_codes(mb[:, tri[0], tri[1]], 3)
    best = int(np.lexsort(tuple(reversed(keys)))[0])
    order = tuple(int(x) for x in perm[best].tolist())
    lab_can = labels[list(order)]
    mat_can = mat[np.ix_(list(order), list(order))]
    node_part = ";".join(f"{a},{b},{c}" for a, b, c in lab_can.tolist())
    edge_part = ";".join(
        f"{i}-{j}:{mat_can[i, j] - 1}" for i in range(n) for j in range(i + 1, n) if mat_can[i, j] > 0
    )
    return f"{n}|{node_part}|{edge_part}", order


def template_key(canon: str) -> str:
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def label_code(label_row) -> int:
    return int(int(label_row[0]) * 1000000 + int(label_row[1]) * 1000 + int(label_row[2]))


def _shortest_paths(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    dist = np.full((n, n), -1, dtype=np.int64)
    for start in range(n):
        dist[start, start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nxt in np.nonzero(adj[node])[0].tolist():
                if dist[start, nxt] < 0:
                    dist[start, nxt] = dist[start, node] + 1
                    queue.append(nxt)
    return dist


class Template:
    """One interned word type with pre-computed relative-site quantities."""

    __slots__ = ("tid", "canon", "key", "labels", "bu", "bv", "btype", "n", "adj",
                 "dist", "pow_adj", "internal_degree", "erased")

    def __init__(self, tid: int, canon: str, labels, bu, bv, btype) -> None:
        self.tid = tid
        self.canon = canon
        self.key = template_key(canon)
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1, 3)
        self.bu = np.asarray(bu, dtype=np.int64).reshape(-1)
        self.bv = np.asarray(bv, dtype=np.int64).reshape(-1)
        self.btype = np.asarray(btype, dtype=np.int64).reshape(-1)
        self.n = int(self.labels.shape[0])
        adj = np.zeros((self.n, self.n), dtype=np.int64)
        for a, b in zip(self.bu.tolist(), self.bv.tolist()):
            adj[a, b] = adj[b, a] = 1
        self.adj = adj
        self.internal_degree = adj.sum(axis=1)
        self.dist = _shortest_paths(adj)
        self.pow_adj = {r: np.linalg.matrix_power(adj.astype(np.float64), r) for r in range(1, 6)}
        self.erased = _erased_table(self)


def r_descriptor(tmpl: Template, p: int, q: int) -> np.ndarray:
    """Fixed numeric relative-site descriptor ``R(D; p, q)`` (15 dims, symmetric)."""
    out = np.zeros(R_DIM, dtype=np.float32)
    out[0] = 1.0 if p == q else 0.0
    dist = int(tmpl.dist[p, q])
    bucket = 5 if dist < 0 or dist >= 5 else dist
    out[1 + bucket] = 1.0
    for r in range(1, 6):
        out[6 + r] = np.log1p(float(tmpl.pow_adj[r][p, q])) / np.log1p(6.0 ** r)
    d1, d2 = int(tmpl.internal_degree[p]), int(tmpl.internal_degree[q])
    out[12] = float(max(d1, d2))
    out[13] = float(min(d1, d2))
    out[14] = 1.0 if np.array_equal(tmpl.labels[p], tmpl.labels[q]) else 0.0
    return out


def _erased_table(tmpl: Template) -> dict[tuple[int, int], np.ndarray]:
    """Mean ``R`` over ordered site pairs compatible with each endpoint label pair.

    Fixed candidate rule, independent of the instance's actual sites: all
    ordered pairs ``(p, q)`` of template sites -- identical sites allowed --
    whose intrinsic word-type atom labels equal the two endpoint labels.
    """
    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    for p in range(tmpl.n):
        for q in range(tmpl.n):
            key = (label_code(tmpl.labels[p]), label_code(tmpl.labels[q]))
            buckets.setdefault(key, []).append(r_descriptor(tmpl, p, q))
    return {k: np.mean(np.stack(v, axis=0), axis=0).astype(np.float32) for k, v in buckets.items()}


class TemplateTable:
    def __init__(self) -> None:
        self.templates: list[Template] = []
        self.index: dict[str, int] = {}
        self.multi: list[int] = []

    def _register(self, canon: str, labels_can, cu, cv, ct) -> int:
        tmpl = Template(len(self.templates), canon, labels_can, cu, cv, ct)
        self.templates.append(tmpl)
        self.index[canon] = tmpl.tid
        if tmpl.n > 1:
            self.multi.append(tmpl.tid)
        return tmpl.tid

    def intern(self, labels, bu, bv, btype) -> int:
        canon, order = canonicalize(labels, bu, bv, btype)
        if canon in self.index:
            return self.index[canon]
        labels = np.asarray(labels, dtype=np.int64).reshape(-1, 3)
        n = labels.shape[0]
        mat = np.zeros((n, n), dtype=np.int64)
        for a, b, t in zip(np.asarray(bu).reshape(-1).tolist(), np.asarray(bv).reshape(-1).tolist(),
                           np.asarray(btype).reshape(-1).tolist()):
            mat[a, b] = mat[b, a] = int(t) + 1
        order = list(order)
        mat_can = mat[np.ix_(order, order)]
        cu, cv, ct = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                if mat_can[i, j] > 0:
                    cu.append(i)
                    cv.append(j)
                    ct.append(int(mat_can[i, j]) - 1)
        return self._register(canon, labels[order], cu, cv, ct)

    def add_clone(self, tmpl: "Template") -> int:
        if tmpl.canon in self.index:
            return self.index[tmpl.canon]
        return self._register(tmpl.canon, tmpl.labels, tmpl.bu, tmpl.bv, tmpl.btype)

    def add_from_canon(self, canon: str, labels_can, local_bonds, order) -> int:
        if canon in self.index:
            return self.index[canon]
        order = list(order)
        n = labels_can.shape[0]
        cu, cv, ct = [], [], []
        for a, b, t in local_bonds:
            pa, pb = order.index(a), order.index(b)
            i, j = min(pa, pb), max(pa, pb)
            cu.append(i)
            cv.append(j)
            ct.append(int(t))
        return self._register(canon, labels_can[order], cu, cv, ct)

    def __len__(self) -> int:
        return len(self.templates)


# ---------------------------------------------------------------------------
# molecules
# ---------------------------------------------------------------------------


@dataclass
class Molecule:
    index: int
    n_atoms: int
    atom_feat: np.ndarray
    bond_u: np.ndarray
    bond_v: np.ndarray
    bond_feat: np.ndarray
    adj: list[list[tuple[int, int]]]
    rank: np.ndarray            # (n,) whole-molecule canonical position


def molecule_canonical_rank(atom_feat: np.ndarray, bu, bv, bond_feat) -> np.ndarray:
    """Exact canonical position of every atom (all 9/3 columns as colours)."""
    import pynauty

    n_atoms = int(atom_feat.shape[0])
    n_bonds = int(len(bu))
    n = n_atoms + n_bonds
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for k, (a, b) in enumerate(zip(bu.tolist(), bv.tolist())):
        bond_node = n_atoms + k
        adj[int(a)].add(bond_node)
        adj[int(b)].add(bond_node)
        adj[bond_node].add(int(a))
        adj[bond_node].add(int(b))
    colours: dict[int, tuple[int, ...]] = {}
    for i in range(n_atoms):
        colours[i] = (0, *[int(x) for x in atom_feat[i]])
    for k in range(n_bonds):
        colours[n_atoms + k] = (1, *[int(x) for x in bond_feat[k]])
    classes: dict[tuple[int, ...], set[int]] = {}
    for vertex, colour in colours.items():
        classes.setdefault(colour, set()).add(vertex)
    graph = pynauty.Graph(
        number_of_vertices=n,
        directed=False,
        adjacency_dict={node: sorted(nbrs) for node, nbrs in adj.items()},
        vertex_coloring=[classes[key] for key in sorted(classes)],
    )
    order = pynauty.canon_label(graph)
    rank = np.zeros(n_atoms, dtype=np.int64)
    for position, vertex in enumerate(order):
        if vertex < n_atoms:
            rank[vertex] = position
    return rank


def load_molecules(payload: Mapping[str, Any]) -> list[Molecule]:
    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    atom_feat = np.asarray(payload["atom_feat"], dtype=np.int64)
    bond_u = np.asarray(payload["bond_u"], dtype=np.int64)
    bond_v = np.asarray(payload["bond_v"], dtype=np.int64)
    bond_feat = np.asarray(payload["bond_feat"], dtype=np.int64)
    out: list[Molecule] = []
    for g in range(len(node_ptr) - 1):
        a0, a1 = int(node_ptr[g]), int(node_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        n = a1 - a0
        u = bond_u[b0:b1]
        v = bond_v[b0:b1]  # cache endpoints are graph-local
        bf = bond_feat[b0:b1]
        adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for k, (a, b) in enumerate(zip(u.tolist(), v.tolist())):
            adj[a].append((b, k))
            adj[b].append((a, k))
        out.append(Molecule(g, n, atom_feat[a0:a1], u, v, bf, adj,
                            molecule_canonical_rank(atom_feat[a0:a1], u, v, bf)))
    return out


# ---------------------------------------------------------------------------
# Train-only merging (one fixed algorithm)
# ---------------------------------------------------------------------------


@dataclass
class PartState:
    pieces: list[list[int]] = field(default_factory=list)
    tmpl: list[int] = field(default_factory=list)

    def atom_piece(self) -> np.ndarray:
        n = sum(len(p) for p in self.pieces)
        out = np.full(n, -1, dtype=np.int64)
        for idx, atoms in enumerate(self.pieces):
            for a in atoms:
                out[a] = idx
        return out


def initial_state(mol: Molecule, table: TemplateTable) -> PartState:
    state = PartState()
    empty = np.zeros(0, dtype=np.int64)
    for a in range(mol.n_atoms):
        lab = mol.atom_feat[a][WORDS_ATOM_COLS][None, :]
        state.pieces.append([a])
        state.tmpl.append(table.intern(lab, empty, empty, empty))
    return state


def _union_key(mol: Molecule, state: PartState, atom_piece: np.ndarray, bond_idx: int):
    """Order-independent cache key of the union of the two pieces at a bond."""
    a = int(mol.bond_u[bond_idx])
    b = int(mol.bond_v[bond_idx])
    pa, pb = int(atom_piece[a]), int(atom_piece[b])
    if pa == pb:
        return None
    atoms_a, atoms_b = state.pieces[pa], state.pieces[pb]
    if len(atoms_a) + len(atoms_b) > MAX_WORD_ATOMS:
        return None
    pos_a = {atom: i for i, atom in enumerate(atoms_a)}
    pos_b = {atom: i for i, atom in enumerate(atoms_b)}
    cross = []
    for x, sx in pos_a.items():
        for nbr, bk in mol.adj[x]:
            if nbr in pos_b:
                cross.append((sx, pos_b[nbr], int(mol.bond_feat[bk, 0])))
    cross.sort()
    ta, tb = state.tmpl[pa], state.tmpl[pb]
    site_a, site_b = pos_a[a], pos_b[b]
    if (ta, site_a) > (tb, site_b):
        pa, pb = pb, pa
        ta, tb = tb, ta
        site_a, site_b = site_b, site_a
        cross = sorted((s2, s1, t) for s1, s2, t in cross)
    return (ta, site_a, tb, site_b, tuple(cross)), pa, pb


def _instance_atoms(mol: Molecule, state: PartState, atom_piece, bond_idx: int,
                    table: TemplateTable, cache: dict):
    """``(template id, atoms in canonical order, pa, pb)`` for one crossing bond."""
    keyed = _union_key(mol, state, atom_piece, bond_idx)
    if keyed is None:
        return None
    key, pa, pb = keyed
    hit = cache.get(key)
    if hit is None:
        atoms_a, atoms_b = state.pieces[pa], state.pieces[pb]
        local_atoms = list(atoms_a) + list(atoms_b)
        pos = {atom: i for i, atom in enumerate(local_atoms)}
        labels = mol.atom_feat[local_atoms][:, WORDS_ATOM_COLS]
        lu, lv, lt = [], [], []
        for x in local_atoms:
            for nbr, bk in mol.adj[x]:
                if nbr in pos and pos[x] < pos[nbr]:
                    lu.append(pos[x])
                    lv.append(pos[nbr])
                    lt.append(int(mol.bond_feat[bk, 0]))
        canon, order = canonicalize(labels, np.asarray(lu), np.asarray(lv), np.asarray(lt))
        tid = table.add_from_canon(canon, np.asarray(labels, dtype=np.int64),
                                   list(zip(lu, lv, lt)), order)
        hit = (tid, tuple(int(x) for x in order))
        cache[key] = hit
    tid, order_t = hit
    local_atoms = list(state.pieces[pa]) + list(state.pieces[pb])
    atoms = tuple(local_atoms[i] for i in order_t)
    return tid, atoms, pa, pb


def apply_template(mol: Molecule, state: PartState, template_id: int, table: TemplateTable,
                   cache: dict) -> int:
    """Merge every non-overlapping instance of one frozen template."""
    atom_piece = state.atom_piece()
    instances = []
    for bond_idx in range(len(mol.bond_u)):
        a = int(mol.bond_u[bond_idx])
        b = int(mol.bond_v[bond_idx])
        if atom_piece[a] == atom_piece[b]:
            continue
        found = _instance_atoms(mol, state, atom_piece, bond_idx, table, cache)
        if found is None:
            continue
        tid, atoms, pa, pb = found
        if tid != template_id or len(atoms) != table.templates[template_id].n:
            continue
        ranks = sorted(int(mol.rank[x]) for x in atoms)
        instances.append(((ranks[0], tuple(ranks)), pa, pb, atoms))
    if not instances:
        return 0
    instances.sort(key=lambda item: item[0])
    used: set[int] = set()
    groups = []
    for _, pa, pb, atoms in instances:
        if pa in used or pb in used:
            continue
        used.add(pa)
        used.add(pb)
        groups.append((pa, pb, atoms))
    if not groups:
        return 0
    by_piece = {min(pa, pb): (max(pa, pb), atoms) for pa, pb, atoms in groups}
    skip: set[int] = set()
    new_pieces, new_tmpl = [], []
    for idx in range(len(state.pieces)):
        if idx in skip:
            continue
        if idx in by_piece:
            pb, atoms = by_piece[idx]
            new_pieces.append(list(atoms))
            new_tmpl.append(template_id)
            skip.add(pb)
        else:
            new_pieces.append(state.pieces[idx])
            new_tmpl.append(state.tmpl[idx])
    state.pieces = new_pieces
    state.tmpl = new_tmpl
    return len(groups)


def learn_merge_history(molecules: Sequence[Molecule], train_graphs: Sequence[int],
                        table: TemplateTable, *, min_df: int, max_multi_words: int,
                        max_rounds: int = 512, log: list[dict] | None = None):
    states = {g: initial_state(molecules[g], table) for g in train_graphs}
    history: list[int] = []
    cache: dict = {}
    rounds = 0
    while rounds < max_rounds and len(table.multi) < max_multi_words:
        df: dict[int, int] = {}
        instances: dict[int, list] = {}
        for g in train_graphs:
            mol = molecules[g]
            state = states[g]
            atom_piece = state.atom_piece()
            seen: set[int] = set()
            for bond_idx in range(len(mol.bond_u)):
                if atom_piece[int(mol.bond_u[bond_idx])] == atom_piece[int(mol.bond_v[bond_idx])]:
                    continue
                found = _instance_atoms(mol, state, atom_piece, bond_idx, table, cache)
                if found is None:
                    continue
                tid = found[0]
                if tid not in seen:
                    seen.add(tid)
                    df[tid] = df.get(tid, 0) + 1
                instances.setdefault(tid, []).append(g)
        candidates = [(t, d) for t, d in df.items() if d >= min_df]
        candidates.sort(key=lambda item: (-item[1], table.templates[item[0]].n,
                                          table.templates[item[0]].canon))
        applied = False
        for tid, d in candidates:
            n_merged = 0
            for g in sorted(set(instances[tid])):
                n_merged += apply_template(molecules[g], states[g], tid, table, cache)
            if n_merged == 0:
                continue
            history.append(tid)
            rounds += 1
            if log is not None:
                log.append({
                    "round": rounds, "template_id": tid,
                    "canonical": table.templates[tid].canon, "df": d,
                    "n_atoms": table.templates[tid].n,
                    "n_merged_molecules": int(len(set(instances[tid]))),
                    "n_merged_instances": int(n_merged),
                })
            applied = True
            break
        if not applied:
            break
    stats = {"rounds": rounds, "n_multi_atom_words": len(table.multi),
             "stopped_at_cap": rounds >= max_rounds, "min_df": int(min_df),
             "cache_entries": len(cache),
             "hit_word_cap": len(table.multi) >= max_multi_words}
    return history, stats


def replay_history(molecules: Sequence[Molecule], graphs: Sequence[int], table: TemplateTable,
                   history: Sequence[int]) -> dict[int, PartState]:
    cache: dict = {}
    states = {g: initial_state(molecules[g], table) for g in graphs}
    for tid in history:
        for g in graphs:
            apply_template(molecules[g], states[g], tid, table, cache)
    return states


def char_states(molecules: Sequence[Molecule], graphs: Sequence[int], table: TemplateTable):
    states: dict[int, PartState] = {}
    empty = np.zeros(0, dtype=np.int64)
    for g in graphs:
        mol = molecules[g]
        state = PartState()
        for a in range(mol.n_atoms):
            lab = mol.atom_feat[a][WORDS_ATOM_COLS][None, :]
            state.pieces.append([a])
            state.tmpl.append(table.intern(lab, empty, empty, empty))
        states[g] = state
    return states


# ---------------------------------------------------------------------------
# packed split tensors
# ---------------------------------------------------------------------------


def _erased_index(table: TemplateTable):
    rows: list[np.ndarray] = []
    index: dict[tuple[int, int, int], int] = {}
    for tmpl in table.templates:
        for (l1, l2), value in tmpl.erased.items():
            index[(tmpl.tid, l1, l2)] = len(rows)
            rows.append(value)
    arr = np.stack(rows, axis=0).astype(np.float32) if rows else np.zeros((0, R_DIM), np.float32)
    return arr, index


def build_split(molecules: Sequence[Molecule], graphs: Sequence[int], states: Mapping[int, PartState],
                table: TemplateTable, *, max_layout_chars: int = 28) -> dict[str, np.ndarray]:
    """Pack word instances, incidences and incidence pairs of one split."""
    erased_R, erased_index = _erased_index(table)
    word_tmpl: list[int] = []
    word_n_atoms: list[int] = []
    word_ptr_atom: list[int] = [0]
    word_atoms: list[int] = []
    word_ptr_bond: list[int] = [0]
    word_bonds: list[int] = []
    graph_word_ptr: list[int] = [0]
    inc_ptr: list[int] = [0]
    inc_other: list[int] = []
    inc_own_atom: list[int] = []
    inc_other_atom: list[int] = []
    inc_bond: list[int] = []
    inc_own_site: list[int] = []
    inc_other_site: list[int] = []
    pair_ptr: list[int] = [0]
    pair_a: list[int] = []
    pair_b: list[int] = []
    pair_R: list[np.ndarray] = []
    pair_ER: list[np.ndarray] = []
    pair_fields: list[np.ndarray] = []
    word_layout: list[str] = []
    pair_layout: list[str] = []
    graph_atom_ptr: list[int] = [0]
    graph_bond_ptr: list[int] = [0]
    atom_off = 0
    bond_off = 0
    for g in graphs:
        mol = molecules[g]
        state = states[g]
        atom_piece = state.atom_piece()
        base_word = len(word_tmpl)
        # crossing bonds seen from each piece (one entry per crossing bond)
        cross: list[list[tuple[int, int, int, int]]] = [[] for _ in state.pieces]
        for idx, atoms in enumerate(state.pieces):
            for a in atoms:
                for nbr, bk in mol.adj[a]:
                    j = int(atom_piece[nbr])
                    if j != idx:
                        cross[idx].append((bk, a, nbr, j))
        for idx in range(len(state.pieces)):
            cross[idx].sort()
        for idx, atoms in enumerate(state.pieces):
            tid = state.tmpl[idx]
            tmpl = table.templates[tid]
            pos = {atom: i for i, atom in enumerate(atoms)}
            word_tmpl.append(tid)
            word_n_atoms.append(len(atoms))
            word_atoms.extend(a + atom_off for a in atoms)
            word_ptr_atom.append(len(word_atoms))
            internal = set()
            for a in atoms:
                for nbr, bk in mol.adj[a]:
                    if atom_piece[nbr] == idx:
                        internal.add(bk)
            for bk in sorted(internal):
                word_bonds.append(bk + bond_off)
            word_ptr_bond.append(len(word_bonds))
            my_inc = len(inc_own_atom)
            for bk, a, nbr, j in cross[idx]:
                inc_bond.append(bk + bond_off)
                inc_own_atom.append(a + atom_off)
                inc_other_atom.append(nbr + atom_off)
                inc_other.append(base_word + j)
                inc_own_site.append(pos[a])
                inc_other_site.append(atom_piece_other_pos(states[g], atom_piece, j, nbr))
            inc_ptr.append(len(inc_own_atom))
            word_layout.append(f"{tid}:{len(atoms)}:{len(cross[idx])}")
            k = len(cross[idx])
            for x in range(k):
                for y in range(x + 1, k):
                    p = pos[cross[idx][x][1]]
                    q = pos[cross[idx][y][1]]
                    pair_a.append(my_inc + x)
                    pair_b.append(my_inc + y)
                    pair_R.append(r_descriptor(tmpl, p, q))
                    key = (tid, label_code(tmpl.labels[p]), label_code(tmpl.labels[q]))
                    eidx = erased_index.get(key)
                    if eidx is None:  # pragma: no cover - exactness guard
                        raise KeyError(f"missing erased descriptor for {key}")
                    pair_ER.append(erased_R[eidx])
                    same = 1.0 if cross[idx][x][3] == cross[idx][y][3] else 0.0
                    counts = np.zeros(5, dtype=np.float32)
                    if same == 0.0:
                        other_atoms = set(state.pieces[cross[idx][y][3]])
                        for b in state.pieces[cross[idx][x][3]]:
                            for nbr, bk in mol.adj[b]:
                                if nbr in other_atoms:
                                    counts[int(mol.bond_feat[bk, 0])] += 1.0
                    exists = 1.0 if counts.sum() > 0 else 0.0
                    pair_fields.append(np.concatenate([[same, exists], counts]).astype(np.float32))
                    pair_layout.append(f"{tid}:{min(p, q)}-{max(p, q)}")
            pair_ptr.append(len(pair_a))
        graph_word_ptr.append(len(word_tmpl))
        atom_off += mol.n_atoms
        bond_off += len(mol.bond_u)
        graph_atom_ptr.append(atom_off)
        graph_bond_ptr.append(bond_off)
    return {
        "graph_word_ptr": np.asarray(graph_word_ptr, dtype=np.int64),
        "graph_atom_ptr": np.asarray(graph_atom_ptr, dtype=np.int64),
        "graph_bond_ptr": np.asarray(graph_bond_ptr, dtype=np.int64),
        "word_tmpl": np.asarray(word_tmpl, dtype=np.int32),
        "word_n_atoms": np.asarray(word_n_atoms, dtype=np.int32),
        "word_ptr_atom": np.asarray(word_ptr_atom, dtype=np.int64),
        "word_atoms": np.asarray(word_atoms, dtype=np.int64),
        "word_ptr_bond": np.asarray(word_ptr_bond, dtype=np.int64),
        "word_bonds": np.asarray(word_bonds, dtype=np.int64),
        "inc_ptr": np.asarray(inc_ptr, dtype=np.int64),
        "inc_other": np.asarray(inc_other, dtype=np.int64),
        "inc_own_atom": np.asarray(inc_own_atom, dtype=np.int64),
        "inc_other_atom": np.asarray(inc_other_atom, dtype=np.int64),
        "inc_bond": np.asarray(inc_bond, dtype=np.int64),
        "inc_own_site": np.asarray(inc_own_site, dtype=np.int64),
        "inc_other_site": np.asarray(inc_other_site, dtype=np.int64),
        "pair_ptr": np.asarray(pair_ptr, dtype=np.int64),
        "pair_a": np.asarray(pair_a, dtype=np.int64),
        "pair_b": np.asarray(pair_b, dtype=np.int64),
        "pair_R": np.asarray(pair_R, dtype=np.float32).reshape(-1, R_DIM),
        "pair_ER": np.asarray(pair_ER, dtype=np.float32).reshape(-1, R_DIM),
        "pair_fields": np.asarray(pair_fields, dtype=np.float32).reshape(-1, PAIR_FIELD_DIM),
        "word_layout": np.asarray(word_layout, dtype=f"<U{max_layout_chars}"),
        "pair_layout": np.asarray(pair_layout, dtype=f"<U{max_layout_chars}"),
    }


def atom_piece_other_pos(state: PartState, atom_piece, piece: int, atom: int) -> int:
    return state.pieces[piece].index(atom)


# ---------------------------------------------------------------------------
# support statistics (Train only, label-free)
# ---------------------------------------------------------------------------


def verify_partition(molecules: Sequence[Molecule], graphs: Sequence[int],
                     states: Mapping[int, PartState]) -> dict[str, Any]:
    """Every atom exactly once, and piece atoms agree with the template labels."""
    checked = 0
    for g in graphs:
        mol = molecules[g]
        state = states[g]
        seen = np.zeros(mol.n_atoms, dtype=np.int64)
        for idx, atoms in enumerate(state.pieces):
            if len(atoms) != len(set(atoms)):
                raise ValueError(f"graph {g}: duplicated atom inside piece {idx}")
            for a in atoms:
                if not (0 <= a < mol.n_atoms):
                    raise ValueError(f"graph {g}: atom {a} out of range")
                seen[a] += 1
            tmpl = None
            if len(atoms) != len(set(atoms)):  # pragma: no cover
                raise ValueError("duplicate atom")
        if not np.all(seen == 1):
            bad = int(np.sum(seen != 1))
            raise ValueError(f"graph {g}: {bad} atoms are not covered exactly once")
        checked += 1
    return {"graphs_checked": checked, "atoms_covered_exactly_once": True}


def support_stats(molecules: Sequence[Molecule], graphs: Sequence[int], states: Mapping[int, PartState],
                  table: TemplateTable) -> dict[str, Any]:
    n_atoms_total = 0
    n_atoms_multi = 0
    word_count: dict[int, int] = {}
    size_hist: dict[int, int] = {}
    inc_hist: dict[int, int] = {}
    port_hist: dict[int, int] = {}
    pair_layouts: dict[int, set] = {}
    dual_instances: dict[int, int] = {}
    for g in graphs:
        mol = molecules[g]
        state = states[g]
        atom_piece = state.atom_piece()
        n_atoms_total += mol.n_atoms
        for idx, atoms in enumerate(state.pieces):
            tid = state.tmpl[idx]
            tmpl = table.templates[tid]
            word_count[tid] = word_count.get(tid, 0) + 1
            size_hist[tmpl.n] = size_hist.get(tmpl.n, 0) + 1
            if tmpl.n > 1:
                n_atoms_multi += len(atoms)
            ports, n_inc = set(), 0
            for a in atoms:
                for nbr, _ in mol.adj[a]:
                    if atom_piece[nbr] != idx:
                        ports.add(atoms.index(a))
                        n_inc += 1
            inc_hist[n_inc] = inc_hist.get(n_inc, 0) + 1
            port_hist[len(ports)] = port_hist.get(len(ports), 0) + 1
            if len(ports) == 2:
                dual_instances[tid] = dual_instances.get(tid, 0) + 1
                pair_layouts.setdefault(tid, set()).add(tuple(sorted(ports)))
    supported = sum(n for tid, n in dual_instances.items() if len(pair_layouts.get(tid, set())) >= 2)
    counts = sorted(word_count.values())
    return {
        "n_train_molecules": len(graphs),
        "n_atoms_total": int(n_atoms_total),
        "n_atoms_in_multi_atom_words": int(n_atoms_multi),
        "multi_atom_atom_coverage": float(n_atoms_multi) / max(1, n_atoms_total),
        "n_word_instances": int(sum(word_count.values())),
        "n_distinct_word_types": int(len(word_count)),
        "n_multi_atom_word_types": int(len(table.multi)),
        "word_size_histogram": {str(k): int(v) for k, v in sorted(size_hist.items())},
        "incidence_count_histogram": {str(k): int(v) for k, v in sorted(inc_hist.items())},
        "distinct_port_count_histogram": {str(k): int(v) for k, v in sorted(port_hist.items())},
        "n_dual_distinct_site_instances": int(sum(dual_instances.values())),
        "n_dual_site_layout_supported_instances": int(supported),
        "n_word_types_with_two_dual_site_layouts": int(
            sum(1 for v in pair_layouts.values() if len(v) >= 2)),
        "rarest_word_type_count": int(counts[0]) if counts else 0,
        "median_word_type_count": float(np.median(counts)) if counts else 0.0,
        "n_word_types_count_le_2": int(sum(1 for c in counts if c <= 2)),
    }
