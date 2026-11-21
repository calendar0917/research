"""Structure library and exhaustive per-molecule enumeration.

Objects (fixed by the round brief, not searched):

* ``char``   = a raw atom and a raw physical bond (the official 9 / 3 columns).
* ``word``   = ``(G, root=v, U)`` with ``1 <= |U| <= 4``, ``v in U`` and the
  **induced** subgraph ``G[U]`` connected.  All such instances are enumerated;
  they may overlap freely.  The structural type is the induced-subgraph
  isomorphism class with the root kept at slot 0 -- ignoring attributes.  The
  rooted library therefore has ``1, 1, 3, 11`` types for ``k = 1..4`` (16 total).
* ``sentence`` = ``(G, root=v, S)`` with ``|S| in {5, 6}`` that can be covered by
  two words sharing ``v``::

      C(S, v) = { {U, V} : v in U ∩ V, |U|,|V| in {3,4}, U ∪ V = S,
                  U ⊄ V, V ⊄ U, G[U] and G[V] connected }

  ``U ∩ V`` need not be connected.  A sentence exists iff ``C(S, v)`` is
  non-empty; different covers of the same sentence are averaged.
* ``segment`` = the whole molecule (read out over all words and sentences).

Nothing is truncated, sampled or shrunk: every instance is enumerated.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

import swf_common as C

EDGE_PAIRS = {k: tuple((i, j) for i in range(k) for j in range(i + 1, k)) for k in range(0, 5)}


def _key_from_slot_adjacency(adj: Sequence[Sequence[int]], k: int) -> str:
    return "".join(str(int(adj[i][j])) for (i, j) in EDGE_PAIRS[k])


def _connected_slot(adj: Sequence[Sequence[int]], k: int) -> bool:
    if k <= 1:
        return True
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for w in range(k):
            if adj[u][w] and w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == k


@dataclass(frozen=True)
class Template:
    template_id: int
    k: int
    key: str
    adjacency: tuple[tuple[int, ...], ...]
    automorphisms: tuple[tuple[int, ...], ...]  # root-preserving slot permutations

    @property
    def n_frames(self) -> int:
        return len(self.automorphisms)

    @property
    def node_mask(self) -> tuple[int, ...]:
        return tuple(1 if i < self.k else 0 for i in range(C.MAX_WORD_ATOMS))

    @property
    def edge_mask(self) -> tuple[int, ...]:
        return tuple(
            1 if (i < self.k and j < self.k and self.adjacency[i][j]) else 0
            for (i, j) in C.WORD_EDGE_PAIRS
        )

    @property
    def adjacency_upper(self) -> tuple[int, ...]:
        return tuple(
            int(self.adjacency[i][j]) if (i < self.k and j < self.k) else 0
            for (i, j) in C.WORD_EDGE_PAIRS
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "k": self.k,
            "canonical_key": self.key,
            "adjacency": [list(row) for row in self.adjacency],
            "edge_slots": [
                [int(i), int(j)] for (i, j) in C.WORD_EDGE_PAIRS
                if i < self.k and j < self.k and self.adjacency[i][j]
            ],
            "root_automorphisms": [list(p) for p in self.automorphisms],
            "n_frames": self.n_frames,
        }


def enumerate_templates(max_k: int = 4) -> tuple[dict[int, Template], dict[str, int]]:
    """All rooted connected unlabeled graphs on 1..4 vertices (16 of them)."""
    templates: dict[int, Template] = {}
    by_key: dict[str, int] = {}
    for k in range(1, max_k + 1):
        pairs = EDGE_PAIRS[k]
        found: dict[str, tuple[tuple[int, ...], ...]] = {}
        for mask in range(1 << len(pairs)):
            adj = [[0] * k for _ in range(k)]
            for bit, (i, j) in enumerate(pairs):
                if (mask >> bit) & 1:
                    adj[i][j] = adj[j][i] = 1
            if not _connected_slot(adj, k):
                continue
            best_key = None
            best_adj = None
            for perm in itertools.permutations(range(1, k)):
                p = (0,) + perm
                reshuffled = tuple(tuple(adj[p[i]][p[j]] for j in range(k)) for i in range(k))
                key = _key_from_slot_adjacency(reshuffled, k)
                if best_key is None or key < best_key:
                    best_key, best_adj = key, reshuffled
            assert best_key is not None and best_adj is not None
            found.setdefault(best_key, best_adj)
        for key in sorted(found):
            adj = found[key]
            autos = tuple(
                p
                for p in itertools.permutations(range(k))
                if p[0] == 0 and all(adj[p[i]][p[j]] == adj[i][j] for i in range(k) for j in range(k))
            )
            tid = len(templates)
            templates[tid] = Template(tid, k, key, adj, autos)
            by_key[key] = tid
    counts = {str(k): sum(1 for t in templates.values() if t.k == k) for k in range(1, max_k + 1)}
    if counts != {"1": 1, "2": 1, "3": 3, "4": 11}:
        raise AssertionError(f"rooted template library is not 1/1/3/11: {counts}")
    return templates, by_key


TEMPLATES, TEMPLATE_BY_KEY = enumerate_templates()
N_TEMPLATES = len(TEMPLATES)
TEMPLATE_NFRAMES = np.asarray([TEMPLATES[t].n_frames for t in range(N_TEMPLATES)], dtype=np.int32)
TEMPLATE_K = np.asarray([TEMPLATES[t].k for t in range(N_TEMPLATES)], dtype=np.int32)
TEMPLATE_NODE_MASK = np.asarray([TEMPLATES[t].node_mask for t in range(N_TEMPLATES)], dtype=np.uint8)
TEMPLATE_EDGE_MASK = np.asarray([TEMPLATES[t].edge_mask for t in range(N_TEMPLATES)], dtype=np.uint8)
TEMPLATE_ADJ = np.asarray([TEMPLATES[t].adjacency_upper for t in range(N_TEMPLATES)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# molecule container
# ---------------------------------------------------------------------------


@dataclass
class Molecule:
    """One molecule with physical (u < v) bonds and the raw attribute columns."""

    row_id: int
    n_atoms: int
    atom_feat: np.ndarray            # [n, 9] uint8
    bond_u: np.ndarray               # [e] int32, u < v
    bond_v: np.ndarray               # [e] int32
    bond_feat: np.ndarray            # [e, 3] uint8

    def adjacency_lists(self) -> list[list[int]]:
        adj = [[] for _ in range(self.n_atoms)]
        for a, b in zip(self.bond_u.tolist(), self.bond_v.tolist()):
            adj[a].append(b)
            adj[b].append(a)
        return adj

    def adjacency_matrix_bool(self) -> list[list[bool]]:
        adj = [[False] * self.n_atoms for _ in range(self.n_atoms)]
        for a, b in zip(self.bond_u.tolist(), self.bond_v.tolist()):
            adj[a][b] = adj[b][a] = True
        return adj

    def bond_index(self) -> dict[tuple[int, int], int]:
        out: dict[tuple[int, int], int] = {}
        for k, (a, b) in enumerate(zip(self.bond_u.tolist(), self.bond_v.tolist())):
            out[(a, b)] = k
            out[(b, a)] = k
        return out

    def permute(self, perm: Sequence[int]) -> "Molecule":
        """Relabel atoms by ``perm`` (old atom -> new atom); bonds follow."""
        perm_arr = np.asarray(perm, dtype=np.int64)
        new_atom = np.zeros_like(self.atom_feat)
        new_atom[perm_arr] = self.atom_feat
        nu = perm_arr[self.bond_u.astype(np.int64)]
        nv = perm_arr[self.bond_v.astype(np.int64)]
        lo = np.minimum(nu, nv).astype(np.int32)
        hi = np.maximum(nu, nv).astype(np.int32)
        order = np.lexsort((hi, lo))
        return Molecule(
            row_id=self.row_id,
            n_atoms=self.n_atoms,
            atom_feat=new_atom,
            bond_u=lo[order],
            bond_v=hi[order],
            bond_feat=self.bond_feat[order],
        )


# ---------------------------------------------------------------------------
# raw cache
# ---------------------------------------------------------------------------


def load_raw_cache(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = C.load_config() if cfg is None else cfg
    path = C.REPO_ROOT / str(cfg["raw_cache"])
    if not path.is_file():
        raise FileNotFoundError(f"raw label-free cache missing: {path}")
    digest = C.sha256_file(path)
    expected = str(cfg["raw_cache_sha256"])
    if digest != expected:
        raise ValueError(f"raw cache sha256 mismatch: {digest} != {expected}")
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["cache_sha256"] = digest
    payload["cache_path"] = str(path)
    return payload


def raw_molecule(raw: Mapping[str, Any], row: int) -> Molecule:
    """Raw ``bond_u/bond_v`` are **molecule-local**; never subtract an offset twice."""
    node_ptr = np.asarray(raw["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], dtype=np.int64)
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    return Molecule(
        row_id=int(np.asarray(raw["row_ids"])[row]),
        n_atoms=int(a1 - a0),
        atom_feat=np.asarray(raw["atom_feat"])[a0:a1].astype(np.uint8, copy=True),
        bond_u=np.asarray(raw["bond_u"])[b0:b1].astype(np.int32, copy=True),
        bond_v=np.asarray(raw["bond_v"])[b0:b1].astype(np.int32, copy=True),
        bond_feat=np.asarray(raw["bond_feat"])[b0:b1].astype(np.uint8, copy=True),
    )


# ---------------------------------------------------------------------------
# attribute code tables (pure dedup; no learned id for a full attribute record)
# ---------------------------------------------------------------------------


class CodeTables:
    """Exact dedup of raw attribute rows (only used for gather indices)."""

    def __init__(self) -> None:
        self.atom_rows: list[bytes] = []
        self.atom_id: dict[bytes, int] = {}
        self.bond_rows: list[bytes] = []
        self.bond_id: dict[bytes, int] = {}

    def atom(self, row: np.ndarray) -> int:
        key = np.asarray(row, dtype=np.uint8).tobytes()
        got = self.atom_id.get(key)
        if got is None:
            got = len(self.atom_rows)
            self.atom_id[key] = got
            self.atom_rows.append(key)
        return got

    def bond(self, row: np.ndarray) -> int:
        key = np.asarray(row, dtype=np.uint8).tobytes()
        got = self.bond_id.get(key)
        if got is None:
            got = len(self.bond_rows)
            self.bond_id[key] = got
            self.bond_rows.append(key)
        return got

    def atom_matrix(self) -> np.ndarray:
        return np.frombuffer(b"".join(self.atom_rows), np.uint8).reshape(-1, C.ATOM_COLUMNS).copy()

    def bond_matrix(self) -> np.ndarray:
        return np.frombuffer(b"".join(self.bond_rows), np.uint8).reshape(-1, C.BOND_COLUMNS).copy()


# ---------------------------------------------------------------------------
# enumeration of one molecule
# ---------------------------------------------------------------------------


def connected_words(adj: Sequence[Sequence[int]], n_atoms: int, max_k: int = 4) -> list[tuple[int, frozenset[int]]]:
    """Every (root, atom_set) with a connected induced subgraph, |set| <= max_k."""
    out: list[tuple[int, frozenset[int]]] = []
    for root in range(n_atoms):
        frontier = [frozenset((root,))]
        subsets: list[frozenset[int]] = [frontier[0]]
        for _size in range(2, max_k + 1):
            nxt: set[frozenset[int]] = set()
            for s in frontier:
                for u in s:
                    for w in adj[u]:
                        if w not in s:
                            nxt.add(s | {w})
            frontier = list(nxt)
            subsets.extend(frontier)
        for s in subsets:
            out.append((root, s))
    return out


def frames_for_word(
    root: int,
    atom_set: frozenset[int],
    adjacency: Sequence[Sequence[bool]],
) -> tuple[int, tuple[tuple[int, ...], ...]]:
    """Template id and every root-fixed slot assignment (the local frames)."""
    verts = sorted(atom_set)
    k = len(verts)
    others = [x for x in verts if x != root]
    candidates: list[tuple[str, tuple[int, ...]]] = []
    for perm in itertools.permutations(others) if k > 1 else [()]:
        slots = (root,) + tuple(perm)
        adj = tuple(
            tuple(1 if (i != j and adjacency[slots[i]][slots[j]]) else 0 for j in range(k))
            for i in range(k)
        )
        candidates.append((_key_from_slot_adjacency(adj, k), slots))
    best_key = min(key for key, _ in candidates)
    tid = TEMPLATE_BY_KEY[best_key]
    frames = tuple(slots for key, slots in candidates if key == best_key)
    if len(frames) != TEMPLATES[tid].n_frames:
        raise AssertionError("frame count does not match |Aut_root(T)|")
    return tid, frames


@dataclass
class MoleculeStructure:
    """All words / sentences / covers / frame pairs of one molecule (local ids)."""

    row_id: int
    n_atoms: int
    n_bonds: int
    # words
    word_root: np.ndarray          # [W] int32
    word_n: np.ndarray             # [W] uint8
    word_tmpl: np.ndarray          # [W] uint8
    word_atoms: np.ndarray         # [W, 4] int32
    word_frame_ptr: np.ndarray     # [W+1] int64
    frame_slots: np.ndarray        # [F, 4] int32
    frame_win_payload: list[bytes] = field(default_factory=list)   # per frame
    # sentences
    sent_root: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    sent_n: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
    sent_atoms: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), np.int32))
    sent_ncovers: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint16))
    cover_sent: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    cover_u: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    cover_v: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    #: ``2 * n_rows`` relation payloads, ordered ``[row0_uv, row0_vu, row1_uv, ...]``
    rel_payload: list[bytes] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rel_payload) // 2


def relation_payload(
    tmpl_u: int, tmpl_v: int, k_u: int, k_v: int,
    e_bits: int, c_bits: int, bond_code_row: Sequence[int],
) -> bytes:
    return b"".join([
        bytes([tmpl_u & 0xFF, tmpl_v & 0xFF, k_u & 0xFF, k_v & 0xFF]),
        np.uint16(e_bits).tobytes(),
        np.uint16(c_bits).tobytes(),
        np.asarray(list(bond_code_row), dtype=np.uint16).tobytes(),
    ])


def enumerate_molecule(
    mol: Molecule,
    codes: CodeTables,
    *,
    sent_sizes: Sequence[int] = (5, 6),
    cover_sizes: Sequence[int] = (3, 4),
    max_word_atoms: int = 4,
) -> MoleculeStructure:
    n = mol.n_atoms
    nb = int(mol.bond_u.size)
    adj_bool = mol.adjacency_matrix_bool()
    adj_list = mol.adjacency_lists()
    bindex = mol.bond_index()
    atom_code_row = [codes.atom(mol.atom_feat[i]) for i in range(n)]
    bond_code_row = [codes.bond(mol.bond_feat[e]) for e in range(nb)]

    words = connected_words(adj_list, n, max_word_atoms)
    nw = len(words)
    word_root = np.zeros(nw, np.int32)
    word_n = np.zeros(nw, np.uint8)
    word_tmpl = np.zeros(nw, np.uint8)
    word_atoms = np.zeros((nw, max_word_atoms), np.int32)
    word_frame_ptr = np.zeros(nw + 1, np.int64)
    frame_slots_all: list[tuple[int, ...]] = []
    frame_payload: list[bytes] = []

    for wi, (root, atom_set) in enumerate(words):
        tid, frames = frames_for_word(root, atom_set, adj_bool)
        tmpl = TEMPLATES[tid]
        k = tmpl.k
        word_root[wi] = root
        word_n[wi] = k
        word_tmpl[wi] = tid
        word_atoms[wi, :k] = sorted(atom_set)
        word_frame_ptr[wi + 1] = word_frame_ptr[wi] + len(frames)
        for slots in frames:
            frame_slots_all.append(slots + (0,) * (max_word_atoms - k))
            atom_rows = [atom_code_row[slots[i]] for i in range(k)]
            bond_rows = []
            for (i, j) in C.WORD_EDGE_PAIRS:
                if i < k and j < k and tmpl.adjacency[i][j]:
                    bond_rows.append(bond_code_row[bindex[(slots[i], slots[j])]])
                else:
                    bond_rows.append(0xFFFF)
            frame_payload.append(
                np.asarray(
                    [tid] + atom_rows + [0] * (max_word_atoms - k) + bond_rows,
                    dtype=np.uint16,
                ).tobytes()
            )
    frame_slots = np.asarray(frame_slots_all, np.int32).reshape(-1, max_word_atoms)

    # ---- sentences and covers -------------------------------------------------
    by_root: dict[int, list[int]] = {}
    for wi in range(nw):
        if int(word_n[wi]) in cover_sizes:
            by_root.setdefault(int(word_root[wi]), []).append(wi)

    sent_index: dict[tuple[tuple[int, ...], int], int] = {}
    sent_root: list[int] = []
    sent_n: list[int] = []
    sent_atoms: list[tuple[int, ...]] = []
    sent_covers: list[list[tuple[int, int]]] = []
    for root in sorted(by_root):
        lst = by_root[root]
        m = len(lst)
        for i in range(m):
            for j in range(i + 1, m):
                wi, wj = lst[i], lst[j]
                union = sorted(
                    set(word_atoms[wi, : int(word_n[wi])].tolist())
                    | set(word_atoms[wj, : int(word_n[wj])].tolist())
                )
                if len(union) not in sent_sizes:
                    continue
                key = (tuple(union), root)
                sid = sent_index.get(key)
                if sid is None:
                    sid = len(sent_root)
                    sent_index[key] = sid
                    sent_root.append(root)
                    sent_n.append(len(union))
                    sent_atoms.append(tuple(union + [0] * (6 - len(union))))
                    sent_covers.append([])
                sent_covers[sid].append((wi, wj))

    ns = len(sent_root)
    cover_sent: list[int] = []
    cover_u: list[int] = []
    cover_v: list[int] = []
    rel_payload: list[bytes] = []
    for sid in range(ns):
        for (wi, wj) in sent_covers[sid]:
            cover_sent.append(sid)
            cover_u.append(wi)
            cover_v.append(wj)
            ku, kv = int(word_n[wi]), int(word_n[wj])
            tu, tv = int(word_tmpl[wi]), int(word_tmpl[wj])
            fu0, fu1 = int(word_frame_ptr[wi]), int(word_frame_ptr[wi + 1])
            fv0, fv1 = int(word_frame_ptr[wj]), int(word_frame_ptr[wj + 1])
            for fa in range(fu0, fu1):
                sa = frame_slots[fa]
                for fb in range(fv0, fv1):
                    sb = frame_slots[fb]
                    for (swap, sx, sy, kx, ky, tx, ty) in (
                        (False, sa, sb, ku, kv, tu, tv),
                        (True, sb, sa, kv, ku, tv, tu),
                    ):
                        e_bits = 0
                        c_bits = 0
                        bond_codes = [0] * 16
                        for p in range(kx):
                            for q in range(ky):
                                ax, ay = int(sx[p]), int(sy[q])
                                if ax == ay:
                                    e_bits |= 1 << (p * 4 + q)
                                else:
                                    be = bindex.get((ax, ay))
                                    if be is not None:
                                        c_bits |= 1 << (p * 4 + q)
                                        bond_codes[p * 4 + q] = bond_code_row[be]
                        rel_payload.append(
                            relation_payload(tx, ty, kx, ky, e_bits, c_bits, bond_codes)
                        )

    return MoleculeStructure(
        row_id=mol.row_id,
        n_atoms=n,
        n_bonds=nb,
        word_root=word_root,
        word_n=word_n,
        word_tmpl=word_tmpl,
        word_atoms=word_atoms,
        word_frame_ptr=word_frame_ptr,
        frame_slots=frame_slots,
        frame_win_payload=frame_payload,
        sent_root=np.asarray(sent_root, np.int32) if ns else np.zeros(0, np.int32),
        sent_n=np.asarray(sent_n, np.uint8) if ns else np.zeros(0, np.uint8),
        sent_atoms=np.asarray(sent_atoms, np.int32) if ns else np.zeros((0, 6), np.int32),
        sent_ncovers=np.asarray([len(sent_covers[s]) for s in range(ns)], np.uint16) if ns else np.zeros(0, np.uint16),
        cover_sent=np.asarray(cover_sent, np.int32) if cover_sent else np.zeros(0, np.int32),
        cover_u=np.asarray(cover_u, np.int32) if cover_u else np.zeros(0, np.int32),
        cover_v=np.asarray(cover_v, np.int32) if cover_v else np.zeros(0, np.int32),
        rel_payload=rel_payload,
    )
