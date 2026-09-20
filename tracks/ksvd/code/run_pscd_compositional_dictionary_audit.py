#!/usr/bin/env python
"""PSCD-v0 — Port-Structured Compositional Graph Dictionary (ZINC).

Question (representation object, Stage A; capacity probe, Stage B)
------------------------------------------------------------------
Can a molecular graph be coded as a small number of **learned subgraph
primitives reused across graphs** plus an explicit, **port-aware composition
graph**?  The sparse code is the structured object::

    G  <->  (D, C_G)

with ``D = {d_1..d_K}`` a shared dictionary of connected, attributed, typed-bond
mesoscale subgraphs learned by data-driven graph BPE (no task supervision), and
``C_G`` a graph-specific composition graph whose nodes are motif occurrences and
whose hyper-edges record ``(source occ, source port, bond type, target occ,
target port)``.

Stage A (feasibility): learn a frozen 64-rule / max-size-8 BPE vocabulary from
train only; tokenize train + valid with the frozen sequence; require **exact**
100 % decode of every graph; audit permutation invariance; build a collision
ladder (bag / no-port / independent-port / full PSCD); report compression,
motif-size, vocabulary-reuse, port-complexity and an MDL-style proxy.

Stage B (capacity probe, only if all Stage-A gates pass): a tiny matched reader
(d=32, L=2) on (1) the composition graph, (2) the motif bag, (3) the raw atom
graph; compare train/valid MAE.

Discipline
----------
* Only official ZINC ``train`` (10 000) and ``valid`` (1 000) are loaded.  The
  official ``test`` split is never read, instantiated or referenced.
* Motif vocabulary / merge rules are learned from **train only**; valid uses the
  frozen sequence.
* ``y`` is used only in the Stage-B capacity probe; Stage A never touches it.
* Exact colored-incidence canonicalization (``pynauty``) is reused from the
  repository; WL hashes are never used as final identity.
* No task-driven dictionary / learned tokenizer / attention / deep GNN / GW-FGW.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code.run_aiom_representation_audit import (  # noqa: E402
    Mol,
    canonical_key_bytes,
    iso_key_hex,
    mol_from_pyg,
    permute_mol,
    repo_loader,
)
from tracks.ksvd.code.run_wholegraph_canonical_registration_audit import (  # noqa: E402
    _pynauty,
    canonical_atom_order,
)

# ---------------------------------------------------------------------------
# Frozen configuration (no sweep)
# ---------------------------------------------------------------------------
MERGE_RULES = 64
MAX_MOTIF = 8
D_MODEL = 32
N_LAYERS = 2
SEED = 0
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/pscd_compositional"

INV_GRAPHS = 500
INV_RELABELS = 20

# Stage B training protocol (identical across the three readers)
B_EPOCHS = 300
B_PATIENCE = 40
B_BATCH = 128
B_LR = 1e-3
B_WD = 0.0
B_CLIP = 5.0


# ===========================================================================
# 0. Loaders
# ===========================================================================
def load_mols_and_y(root: Path, split: str, limit: int | None = None):
    if split == "test":
        raise RuntimeError("PSCD never loads the official test split")
    load_zinc, data_to_graph = repo_loader()
    pyg_split = "val" if split == "valid" else split
    dataset = load_zinc(root, pyg_split)
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    mols: list[Mol] = []
    ys: list[float] = []
    for i in range(n):
        data = dataset[i]
        mols.append(mol_from_pyg(data, data_to_graph))
        ys.append(float(data.y.reshape(-1)[0]))
    return mols, np.asarray(ys, dtype=np.float64)


# ===========================================================================
# 1. Exact canonicalization helpers
# ===========================================================================
class _SG:
    """Lightweight graph wrapper so we can reuse ``canonical_atom_order``."""

    __slots__ = ("_n", "_e")

    def __init__(self, n: int, edges: list[tuple[int, int]]):
        self._n = n
        self._e = edges

    @property
    def n(self) -> int:
        return self._n

    def edges(self) -> list[tuple[int, int]]:
        return self._e


def mol_rank(mol: Mol) -> list[int]:
    """Exact canonical atom rank ``rho_G`` (rank[atom] = canonical slot)."""
    order, _key = canonical_atom_order(
        _SG(mol.n, list(mol.bonds)), mol.node_types,
        {b: int(mol.bond_types[i]) for i, b in enumerate(mol.bonds)},
    )
    rank = [0] * mol.n
    for slot, atom in enumerate(order):
        rank[int(atom)] = slot
    return rank


def canon_subgraph(atoms: Sequence[int], bonds_with_types, node_types):
    """Exact canonical key + canonical form of an induced subgraph.

    ``bonds_with_types`` is an iterable of ``(a, b, t)`` over the whole molecule
    (atom ids are global).  Returns ``(key, canon_node_types, canon_bonds)``
    where ``canon_node_types[s]`` is the atom category in canonical slot ``s``
    and ``canon_bonds`` is a sorted tuple of ``(slot_a, slot_b, bond_type)``.
    """
    atoms = sorted(int(a) for a in atoms)
    idx = {a: i for i, a in enumerate(atoms)}
    k = len(atoms)
    nt = [int(node_types[a]) for a in atoms]
    edges: list[tuple[int, int]] = []
    et: dict[tuple[int, int], int] = {}
    for (a, b, t) in bonds_with_types:
        ia = idx.get(int(a))
        ib = idx.get(int(b))
        if ia is None or ib is None:
            continue
        if ia > ib:
            ia, ib = ib, ia
        edges.append((ia, ib))
        et[(ia, ib)] = int(t)
    order, key = canonical_atom_order(_SG(k, edges), nt, et)
    rank = [0] * k
    for slot, a in enumerate(order):
        rank[int(a)] = slot
    canon_nt = tuple(nt[int(order[s])] for s in range(k))
    canon_bonds = tuple(sorted(
        (min(rank[a], rank[b]), max(rank[a], rank[b]), int(t)) for (a, b), t in et.items()
    ))
    return key, canon_nt, canon_bonds


def motif_orbits(canon_nt: Sequence[int], canon_bonds: Sequence[tuple[int, int, int]]):
    """Automorphism orbit id per canonical slot (colored incidence via pynauty)."""
    pynauty = _pynauty()
    L = len(canon_nt)
    nb = len(canon_bonds)
    nv = L + nb
    adj: dict[int, list[int]] = {v: [] for v in range(nv)}
    color: list[tuple[Any, ...]] = [("n", int(canon_nt[i])) for i in range(L)]
    for e, (a, b, t) in enumerate(canon_bonds):
        ev = L + e
        adj[int(a)].append(ev)
        adj[int(b)].append(ev)
        adj[ev] = [int(a), int(b)]
        color.append(("e", int(t)))
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for v in range(nv):
        groups[color[v]].append(v)
    cell_keys = sorted(groups, key=repr)
    coloring = [set(groups[k]) for k in cell_keys]
    pg = pynauty.Graph(number_of_vertices=nv, directed=False,
                       adjacency_dict={k: list(v) for k, v in adj.items()},
                       vertex_coloring=coloring)
    _gens, _g1, _g2, orbits, _norb = pynauty.autgrp(pg)
    return tuple(int(orbits[i]) for i in range(L))


def colored_graph_key(node_colors: Sequence[Any], edges: Sequence[tuple[int, int]]) -> bytes:
    """Exact canonical key of a simple colored graph.

    The pynauty certificate alone encodes only adjacency, so the canonical
    colour sequence is appended (same convention as ``canonical_atom_order``).
    """
    pynauty = _pynauty()
    nv = len(node_colors)
    adj: dict[int, list[int]] = {v: [] for v in range(nv)}
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    groups: dict[Any, list[int]] = defaultdict(list)
    for v in range(nv):
        groups[node_colors[v]].append(v)
    cell_keys = sorted(groups, key=repr)
    coloring = [set(groups[k]) for k in cell_keys]
    pg = pynauty.Graph(number_of_vertices=nv, directed=False,
                       adjacency_dict={k: list(v) for k, v in adj.items()},
                       vertex_coloring=coloring)
    certificate = bytes(pynauty.certificate(pg))
    label = [int(v) for v in pynauty.canon_label(pg)]
    colour_sequence = tuple(node_colors[v] for v in label)
    return certificate + b"|" + repr(colour_sequence).encode()


# ===========================================================================
# 2. Motif types and vocabulary
# ===========================================================================
@dataclass
class MotifType:
    mid: int
    key: bytes
    size: int
    canon_nt: tuple[int, ...]
    canon_bonds: tuple[tuple[int, int, int], ...]
    orbits: tuple[int, ...]
    singleton: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "mid": self.mid,
            "size": self.size,
            "key": self.key.hex(),
            "node_types": list(self.canon_nt),
            "bonds": [list(b) for b in self.canon_bonds],
            "orbits": list(self.orbits),
            "singleton": self.singleton,
        }


def singleton_key(node_type: int) -> bytes:
    return b"s" + int(node_type).to_bytes(2, "little")


class Vocabulary:
    def __init__(self) -> None:
        self.by_key: dict[bytes, MotifType] = {}
        self.by_id: dict[int, MotifType] = {}
        self._next = 0

    def add(self, key: bytes, size: int, canon_nt, canon_bonds, singleton: bool) -> MotifType:
        if key in self.by_key:
            return self.by_key[key]
        mid = self._next
        self._next += 1
        mt = MotifType(mid=mid, key=key, size=size, canon_nt=tuple(canon_nt),
                       canon_bonds=tuple(tuple(b) for b in canon_bonds),
                       orbits=motif_orbits(canon_nt, canon_bonds), singleton=singleton)
        self.by_key[key] = mt
        self.by_id[mid] = mt
        return mt

    def add_singleton(self, node_type: int) -> MotifType:
        return self.add(singleton_key(node_type), 1, (int(node_type),), (), True)

    def __len__(self) -> int:
        return self._next


# ===========================================================================
# 3. Graph-BPE candidate enumeration and deterministic application
# ===========================================================================
@dataclass
class Token:
    atoms: tuple[int, ...]
    key: bytes
    size: int


def graph_bonds(mol: Mol):
    return [(int(a), int(b), int(mol.bond_types[i])) for i, (a, b) in enumerate(mol.bonds)]


def init_tokens(vocab: Vocabulary, mol: Mol) -> list[Token]:
    return [Token((v,), vocab.add_singleton(int(mol.node_types[v])).key, 1)
            for v in range(mol.n)]


def atom_to_token(tokens: Sequence[Token]) -> dict[int, int]:
    m: dict[int, int] = {}
    for i, tok in enumerate(tokens):
        for a in tok.atoms:
            m[a] = i
    return m


def enumerate_candidates(tokens: Sequence[Token], bonds, node_types,
                         atom_index: dict[int, int], max_motif: int = MAX_MOTIF):
    """All distinct adjacent-token union candidates for one graph.

    Each candidate is ``(ta, tb, atoms_tuple, key, canon_nt, canon_bonds)``.
    """
    seen: set[tuple[int, int]] = set()
    out = []
    for (a, b, t) in bonds:
        ta = atom_index[int(a)]
        tb = atom_index[int(b)]
        if ta == tb:
            continue
        pk = (ta, tb) if ta < tb else (tb, ta)
        if pk in seen:
            continue
        seen.add(pk)
        atoms = tuple(sorted(set(tokens[ta].atoms) | set(tokens[tb].atoms)))
        if len(atoms) > max_motif:
            continue
        key, canon_nt, canon_bonds = canon_subgraph(atoms, bonds, node_types)
        out.append((ta, tb, atoms, key, canon_nt, canon_bonds))
    return out


def apply_rule(tokens: list[Token], candidates, target_key: bytes, rank) -> tuple[list[Token], int]:
    """Deterministic greedy maximal matching of candidates with ``key == target``."""
    hits = [c for c in candidates if c[3] == target_key]
    if not hits:
        return tokens, 0

    def sort_key(c):
        r = sorted(rank[a] for a in c[2])
        return (r[0], tuple(r))

    hits.sort(key=sort_key)
    used: set[int] = set()
    parent = list(range(len(tokens)))
    n_merges = 0
    for (ta, tb, atoms, key, _cn, _cb) in hits:
        if ta in used or tb in used:
            continue
        if ta == tb:
            continue
        used.add(ta)
        used.add(tb)
        parent[max(ta, tb)] = min(ta, tb)
        n_merges += 1
    if n_merges == 0:
        return tokens, 0

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(tokens)):
        root = i
        while parent[root] != root:
            root = parent[root]
        groups[root].append(i)

    new_tokens: list[Token] = []
    # preserve order by smallest member for determinism
    for root in sorted(groups):
        members = groups[root]
        if len(members) == 1:
            new_tokens.append(tokens[members[0]])
        else:
            atoms = tuple(sorted(a for i in members for a in tokens[i].atoms))
            new_tokens.append(Token(atoms, target_key, len(atoms)))
    return new_tokens, n_merges


def learn_vocabulary(train_mols: Sequence[Mol], vocab: Vocabulary,
                     n_rules: int = MERGE_RULES, log=print):
    bonds = [graph_bonds(m) for m in train_mols]
    tokens = [init_tokens(vocab, m) for m in train_mols]
    ranks = [mol_rank(m) for m in train_mols]
    introduced: set[bytes] = set()
    rounds: list[dict[str, Any]] = []

    for r in range(1, n_rules + 1):
        tally: Counter[bytes] = Counter()
        support: dict[bytes, set[int]] = defaultdict(set)
        example: dict[bytes, tuple] = {}
        per_graph_cands = []
        for gi, mol in enumerate(train_mols):
            idx = atom_to_token(tokens[gi])
            cands = enumerate_candidates(tokens[gi], bonds[gi], mol.node_types, idx)
            per_graph_cands.append(cands)
            for (_ta, _tb, _atoms, key, cn, cb) in cands:
                if key in introduced:
                    continue
                tally[key] += 1
                support[key].add(gi)
                if key not in example:
                    example[key] = (len(_atoms), cn, cb)
        if not tally:
            log(f"[BPE] round {r}: no new candidates; stopping")
            break
        best_key = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        size, cn, cb = example[best_key]
        vocab.add(best_key, size, cn, cb, singleton=False)
        introduced.add(best_key)

        total_merges = 0
        for gi in range(len(train_mols)):
            tokens[gi], nm = apply_rule(tokens[gi], per_graph_cands[gi], best_key, ranks[gi])
            total_merges += nm
        remaining = int(sum(len(tk) for tk in tokens))
        rounds.append({
            "round": r,
            "key": best_key.hex(),
            "size": int(size),
            "candidates": int(sum(1 for g in per_graph_cands for c in g if c[3] == best_key)),
            "occurrence_freq": int(tally[best_key]),
            "graph_support": int(len(support[best_key])),
            "merges": int(total_merges),
            "remaining_tokens": remaining,
        })
        log(f"[BPE] r{r:02d} size={size} occ={tally[best_key]} "
            f"support={len(support[best_key])} merges={total_merges} "
            f"remaining={remaining}")

    return tokens, rounds


def tokenize_frozen(mol: Mol, vocab: Vocabulary, sequence: Sequence[bytes], rank):
    """Apply a frozen merge sequence to one molecule (no new motifs)."""
    tokens = init_tokens(vocab, mol)
    bonds = graph_bonds(mol)
    for key in sequence:
        idx = atom_to_token(tokens)
        cands = enumerate_candidates(tokens, bonds, mol.node_types, idx)
        tokens, _ = apply_rule(tokens, cands, key, rank)
    return tokens


# ===========================================================================
# 4. Occurrence -> dictionary mapping (lex-min isomorphism) and structured code
# ===========================================================================
def lexmin_slot_mapping(atoms, bonds, node_types, mt: MotifType, rank):
    """Lex-min isomorphism ``psi: canonical slots -> occurrence atoms``.

    Returns ``slot_of`` mapping occurrence atom -> canonical slot, or ``None`` if
    no isomorphism exists (a consistency failure).
    """
    occ = sorted(int(a) for a in atoms)
    L = mt.size
    occ_set = set(occ)
    # local adjacency among occurrence atoms: type of bond between two atoms
    bond_type: dict[tuple[int, int], int] = {}
    for (a, b, t) in bonds:
        a, b = int(a), int(b)
        if a in occ_set and b in occ_set:
            key = (a, b) if a < b else (b, a)
            bond_type[key] = int(t)
    motif_edge: dict[tuple[int, int], int] = {}
    for (a, b, t) in mt.canon_bonds:
        motif_edge[(a, b)] = int(t)

    cand_by_type: dict[int, list[int]] = defaultdict(list)
    for a in occ:
        cand_by_type[int(node_types[a])].append(a)
    for v in cand_by_type:
        cand_by_type[v].sort(key=lambda a: rank[a])

    psi = [-1] * L

    def consistent(slot: int, atom: int) -> bool:
        for s2 in range(slot):
            a2 = psi[s2]
            key = (atom, a2) if atom < a2 else (a2, atom)
            mkey = (slot, s2) if slot < s2 else (s2, slot)
            want = motif_edge.get(mkey)
            have = bond_type.get(key)
            if want is None and have is not None:
                return False
            if want is not None and have != want:
                return False
        return True

    def dfs(slot: int, used: set) -> bool:
        if slot == L:
            return True
        need = mt.canon_nt[slot]
        for atom in cand_by_type.get(need, []):
            if atom in used:
                continue
            if not consistent(slot, atom):
                continue
            psi[slot] = atom
            used.add(atom)
            if dfs(slot + 1, used):
                return True
            used.discard(atom)
            psi[slot] = -1
        return False

    if not dfs(0, set()):
        return None
    slot_of = {atom: s for s, atom in enumerate(psi)}
    return slot_of


@dataclass
class StructuredCode:
    n: int
    m: int
    occ_motif_ids: list[int]
    comp_edges: list[tuple[int, int, int, int, int]]  # (i, p_i, t, j, p_j)

    def port_states(self) -> list[list[tuple[int, int]]]:
        """Per occurrence: list of (port, bond_type) over external bonds."""
        states: list[list[tuple[int, int]]] = [[] for _ in self.occ_motif_ids]
        for (i, pi, t, j, pj) in self.comp_edges:
            states[i].append((pi, t))
            states[j].append((pj, t))
        for s in states:
            s.sort()
        return states

    def port_slots(self) -> list[set[int]]:
        out = [set() for _ in self.occ_motif_ids]
        for (i, pi, t, j, pj) in self.comp_edges:
            out[i].add(pi)
            out[j].add(pj)
        return out


def build_structured_code(mol: Mol, tokens: Sequence[Token], vocab: Vocabulary,
                          rank) -> dict:
    """Return a structured code dict.  Occurrences are ordered by canonical
    atom-set key so the object is permutation invariant."""
    bonds = graph_bonds(mol)
    atom_to_occ: dict[int, int] = {}
    occs = []
    for tok in tokens:
        atoms = tuple(sorted(tok.atoms))
        r = sorted(rank[a] for a in atoms)
        occs.append((r[0], tuple(r), atoms, tok.key))
    occs.sort(key=lambda z: (z[0], z[1]))

    slot_maps: list[dict[int, int]] = []
    occ_motif_ids: list[int] = []
    for (_k0, _k1, atoms, key) in occs:
        mt = vocab.by_key[key]
        occ_motif_ids.append(mt.mid)
        for a in atoms:
            atom_to_occ[a] = len(occ_motif_ids) - 1
        if mt.size == 1:
            slot_maps.append({atoms[0]: 0})
        else:
            sm = lexmin_slot_mapping(atoms, bonds, mol.node_types, mt, rank)
            if sm is None:
                raise RuntimeError("occurrence has no isomorphism to its motif type")
            slot_maps.append(sm)

    comp_edges = []
    for (a, b, t) in bonds:
        oa = atom_to_occ[int(a)]
        ob = atom_to_occ[int(b)]
        if oa == ob:
            continue
        pi = slot_maps[oa][int(a)]
        pj = slot_maps[ob][int(b)]
        comp_edges.append((oa, pi, int(t), ob, pj))
    comp_edges.sort()
    return {"code": StructuredCode(mol.n, mol.m, occ_motif_ids, comp_edges),
            "slot_maps": slot_maps, "atom_to_occ": atom_to_occ}


def decode(code: StructuredCode, vocab: Vocabulary) -> Mol:
    """Exact reconstruction of the molecule from (dictionary, composition)."""
    n = 0
    offsets = []
    node_types = []
    for mid in code.occ_motif_ids:
        offsets.append(n)
        mt = vocab.by_id[mid]
        node_types.extend(mt.canon_nt)
        n += mt.size
    bonds: list[tuple[int, int]] = []
    bond_types: list[int] = []
    for occ_i, mid in enumerate(code.occ_motif_ids):
        mt = vocab.by_id[mid]
        for (a, b, t) in mt.canon_bonds:
            bonds.append((offsets[occ_i] + a, offsets[occ_i] + b))
            bond_types.append(int(t))
    for (i, pi, t, j, pj) in code.comp_edges:
        u = offsets[i] + pi
        v = offsets[j] + pj
        bonds.append((u, v) if u < v else (v, u))
        bond_types.append(int(t))
    return Mol(n, bonds, np.asarray(bond_types, dtype=np.int64),
               np.asarray(node_types, dtype=np.int64))


def mol_iso_key(mol: Mol) -> bytes:
    """Exact canonical key of a Mol, normalising bond endpoints first."""
    bonds = []
    bts = []
    for i, (a, b) in enumerate(mol.bonds):
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        bonds.append((a, b))
        bts.append(int(mol.bond_types[i]))
    nm = Mol(mol.n, bonds, np.asarray(bts, dtype=np.int64),
             np.asarray(mol.node_types, dtype=np.int64))
    return canonical_key_bytes(nm)


# ===========================================================================
# 5. Collision-ladder keys (A/B/C/D)
# ===========================================================================
def ladder_keys(sc: StructuredCode, vocab: Vocabulary) -> dict[str, bytes]:
    mid = sc.occ_motif_ids
    mt_list = [vocab.by_id[k] for k in mid]

    # A: motif bag (occurrence-node colors only)
    colors_a = [("m", vocab.by_id[k].key.hex()) for k in mid]
    key_a = colored_graph_key(colors_a, [])

    # B: occurrence nodes + per-bond nodes (no ports)
    colors_b = [("m", vocab.by_id[k].key.hex()) for k in mid]
    edges_b: list[tuple[int, int]] = []
    for idx, (i, pi, t, j, pj) in enumerate(sc.comp_edges):
        bnode = len(colors_b)
        colors_b.append(("b", int(t)))
        edges_b.append((i, bnode))
        edges_b.append((bnode, j))
    key_b = colored_graph_key(colors_b, edges_b)

    # C: + port nodes colored by Aut-orbit id
    colors_c = [("m", vocab.by_id[k].key.hex()) for k in mid]
    edges_c: list[tuple[int, int]] = []
    for (i, pi, t, j, pj) in sc.comp_edges:
        bnode = len(colors_c)
        colors_c.append(("b", int(t)))
        oi = mt_list[i].orbits[pi]
        oj = mt_list[j].orbits[pj]
        pi_node = len(colors_c)
        colors_c.append(("p", ("orb", int(oi))))
        pj_node = len(colors_c)
        colors_c.append(("p", ("orb", int(oj))))
        edges_c.extend([(i, pi_node), (pi_node, bnode), (bnode, pj_node), (pj_node, j)])
    key_c = colored_graph_key(colors_c, edges_c)

    # D: full PSCD (exact canonical port slots)
    colors_d = [("m", vocab.by_id[k].key.hex()) for k in mid]
    edges_d: list[tuple[int, int]] = []
    for (i, pi, t, j, pj) in sc.comp_edges:
        bnode = len(colors_d)
        colors_d.append(("b", int(t)))
        pi_node = len(colors_d)
        colors_d.append(("p", int(pi)))
        pj_node = len(colors_d)
        colors_d.append(("p", int(pj)))
        edges_d.extend([(i, pi_node), (pi_node, bnode), (bnode, pj_node), (pj_node, j)])
    key_d = colored_graph_key(colors_d, edges_d)
    return {"A": key_a, "B": key_b, "C": key_c, "D": key_d}


# ===========================================================================
# 6. Metrics
# ===========================================================================
def _q(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"mean": 0.0, "median": 0.0, "p5": 0.0, "p95": 0.0}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p5": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
    }


def compression_metrics(codes: Sequence[StructuredCode]) -> dict[str, Any]:
    n = np.asarray([c.n for c in codes], dtype=np.float64)
    m = np.asarray([len(c.occ_motif_ids) for c in codes], dtype=np.float64)
    r = m / np.maximum(n, 1.0)
    return {
        "n_mean": float(n.mean()),
        "M_mean": float(m.mean()),
        "ratio": _q(r.tolist()),
        "compression": _q((n / np.maximum(m, 1.0)).tolist()),
        "mean_ratio": float(r.mean()),
        "mean_compression": float((n / np.maximum(m, 1.0)).mean()),
    }


def motif_size_metrics(code: StructuredCode, vocab: Vocabulary) -> dict[str, Any]:
    sizes = [vocab.by_id[k].size for k in code.occ_motif_ids]
    if not sizes:
        return {"occ_sizes": [], "atom_covered_ge2": 0}
    return {"occ_sizes": sizes, "atom_covered_ge2": sum(s for s in sizes if s >= 2)}


def aggregate_motif_size(codes: Sequence[StructuredCode], vocab: Vocabulary) -> dict[str, Any]:
    occ_sizes: list[int] = []
    n_occ = 0
    n_singleton_occ = 0
    covered_atoms = 0
    total_atoms = 0
    for c in codes:
        total_atoms += c.n
        for k in c.occ_motif_ids:
            s = vocab.by_id[k].size
            occ_sizes.append(s)
            n_occ += 1
            covered_atoms += s
            if s == 1:
                n_singleton_occ += 1
    arr = np.asarray(occ_sizes, dtype=np.float64)
    hist = Counter(occ_sizes)
    return {
        "occ_size": _q(arr.tolist()),
        "hist": {int(k): int(v) for k, v in sorted(hist.items())},
        "atom_frac_size_ge2": float(sum(s for s in occ_sizes if s >= 2) / max(total_atoms, 1)),
        "occurrence_frac_singleton": float(n_singleton_occ / max(n_occ, 1)),
    }


def vocabulary_reuse(codes: Sequence[StructuredCode], vocab: Vocabulary) -> dict[str, Any]:
    occ_counter: Counter[int] = Counter()
    graph_support: Counter[int] = Counter()
    for c in codes:
        seen = set()
        for k in c.occ_motif_ids:
            occ_counter[k] += 1
            seen.add(k)
        for k in seen:
            graph_support[k] += 1
    learned = [mt for mt in vocab.by_id.values() if not mt.singleton]
    used = [mt for mt in learned if occ_counter.get(mt.mid, 0) > 0]
    unused = [mt for mt in learned if occ_counter.get(mt.mid, 0) == 0]
    freqs = sorted((occ_counter.get(mt.mid, 0) for mt in learned), reverse=True)
    support_hist = Counter(graph_support.get(mt.mid, 0) for mt in learned)
    frag = sum(1 for f in freqs if f < 10)
    return {
        "learned_types": len(learned),
        "used_types": len(used),
        "unused_types": len(unused),
        "low_freq_lt10": int(frag),
        "low_freq_frac": float(frag / max(len(learned), 1)),
        "support_hist": {int(k): int(v) for k, v in sorted(support_hist.items())},
        "top_coverage": {
            "top16": float(sum(freqs[:16]) / max(sum(freqs), 1)),
            "top32": float(sum(freqs[:32]) / max(sum(freqs), 1)),
            "top64": float(sum(freqs[:64]) / max(sum(freqs), 1)),
        },
        "motif_occ": {int(mt.mid): int(occ_counter.get(mt.mid, 0)) for mt in learned},
        "motif_support": {int(mt.mid): int(graph_support.get(mt.mid, 0)) for mt in learned},
    }


def port_complexity(codes: Sequence[StructuredCode], vocab: Vocabulary) -> dict[str, Any]:
    d_list: list[int] = []
    p_list: list[int] = []
    ratio: list[float] = []
    ratio_ge4: list[float] = []
    size_ge4_occ = 0
    near_all_ports_ge4 = 0
    joint_states: dict[int, set] = defaultdict(set)
    for c in codes:
        states = c.port_states()
        slots = c.port_slots()
        for occ, k in enumerate(c.occ_motif_ids):
            size = vocab.by_id[k].size
            dj = len(states[occ])
            pj = len(slots[occ])
            d_list.append(dj)
            p_list.append(pj)
            ratio.append(pj / size)
            joint_states[k].add(tuple(sorted(states[occ])))
            if size >= 4:
                size_ge4_occ += 1
                ratio_ge4.append(pj / size)
                if pj >= max(size - 1, 1):
                    near_all_ports_ge4 += 1
    diversity = {int(k): len(v) for k, v in joint_states.items()}
    return {
        "d_mean": float(np.mean(d_list)) if d_list else 0.0,
        "p_mean": float(np.mean(p_list)) if p_list else 0.0,
        "p_p95": float(np.percentile(p_list, 95)) if p_list else 0.0,
        "p_over_size": _q(ratio),
        "p_over_size_ge4_mean": float(np.mean(ratio_ge4)) if ratio_ge4 else 0.0,
        "size_ge4_occ": int(size_ge4_occ),
        "near_all_ports_frac_ge4": float(near_all_ports_ge4 / max(size_ge4_occ, 1)),
        "joint_state_diversity_mean": float(np.mean(list(diversity.values()))) if diversity else 0.0,
        "joint_state_diversity": diversity,
    }


def mdl_proxy(train_codes, valid_codes, train_mols, valid_mols, vocab: Vocabulary):
    """Transparent empirical description-length proxy (not full GraphMDL)."""
    def logp(counter, total):
        return {k: -math.log2(v / total) for k, v in counter.items() if v > 0}

    # empirical symbol distributions from train
    atom_c = Counter(int(x) for mol in train_mols for x in mol.node_types.tolist())
    bond_c = Counter(int(x) for mol in train_mols for x in mol.bond_types.tolist())
    atom_lp = {k: -math.log2(v / sum(atom_c.values())) for k, v in atom_c.items()}
    bond_lp = {k: -math.log2(v / sum(bond_c.values())) for k, v in bond_c.items()}
    occs = [len(c.occ_motif_ids) for c in train_codes]
    occ_c = Counter(occs)
    occ_lp = {k: -math.log2(v / max(sum(occ_c.values()), 1)) for k, v in occ_c.items()}
    edges_counts = [len(c.comp_edges) for c in train_codes]
    ecount_c = Counter(edges_counts)
    ecount_lp = {k: -math.log2(v / max(sum(ecount_c.values()), 1)) for k, v in ecount_c.items()}
    motif_use = Counter(k for c in train_codes for k in c.occ_motif_ids)
    motif_lp = {k: -math.log2(v / max(sum(motif_use.values()), 1)) for k, v in motif_use.items()}
    # port distribution conditioned on motif type and slot
    port_c: dict[int, Counter] = defaultdict(Counter)
    for c in train_codes:
        for (i, pi, t, j, pj) in c.comp_edges:
            port_c[c.occ_motif_ids[i]][pi] += 1
            port_c[c.occ_motif_ids[j]][pj] += 1
    port_lp = {k: {s: -math.log2(v / sum(cnt.values())) for s, v in cnt.items()}
               for k, cnt in port_c.items()}

    def sym_lp(table, key, default=8.0):
        return table.get(key, default)

    # ---- dictionary cost ----
    learned = [mt for mt in vocab.by_id.values() if not mt.singleton]
    sizes_c = Counter(mt.size for mt in learned)
    dict_cost = 0.0
    for mt in learned:
        dict_cost += -math.log2(sizes_c[mt.size] / max(len(learned), 1))
        for c in mt.canon_nt:
            dict_cost += sym_lp(atom_lp, int(c))
        for (_a, _b, t) in mt.canon_bonds:
            dict_cost += sym_lp(bond_lp, int(t))
        npairs = mt.size * (mt.size - 1) // 2
        dict_cost += npairs  # 1 bit per internal pair: edge / no-edge symbol

    def atom_singleton_cost(mol: Mol) -> float:
        cost = 0.0
        for x in mol.node_types.tolist():
            cost += sym_lp(atom_lp, int(x))
        # encode the full bond pattern as an unordered-pair symbol (none/type)
        bg = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in mol.bonds)
        for i in range(mol.n):
            for j in range(i + 1, mol.n):
                if (i, j) in bg:
                    cost += 1.0 + 2.0  # edge present + type ~2 bits
                else:
                    cost += 1.0
        return cost

    def pscd_graph_cost(c: StructuredCode) -> float:
        cost = sym_lp(occ_lp, len(c.occ_motif_ids), 12.0)
        cost += sym_lp(ecount_lp, len(c.comp_edges), 12.0)
        for k in c.occ_motif_ids:
            cost += sym_lp(motif_lp, k, 12.0)
        for (i, pi, t, j, pj) in c.comp_edges:
            cost += sym_lp(bond_lp, int(t))
            cost += port_lp.get(c.occ_motif_ids[i], {}).get(pi, 3.0)
            cost += port_lp.get(c.occ_motif_ids[j], {}).get(pj, 3.0)
        return cost

    n_train = max(len(train_codes), 1)
    pscd_vals = [pscd_graph_cost(c) for c in valid_codes]
    atom_vals = [atom_singleton_cost(m) for m in valid_mols]
    return {
        "dictionary_cost_bits": float(dict_cost),
        "dictionary_cost_per_graph_bits": float(dict_cost / n_train),
        "pscd_graph_code_mean_bits": float(np.mean(pscd_vals)) if pscd_vals else 0.0,
        "pscd_total_mean_bits": float(np.mean(pscd_vals) + dict_cost / n_train) if pscd_vals else 0.0,
        "atom_singleton_mean_bits": float(np.mean(atom_vals)) if atom_vals else 0.0,
        "ratio_total_over_atom": float((np.mean(pscd_vals) + dict_cost / n_train)
                                       / max(np.mean(atom_vals), 1e-9)) if pscd_vals else 0.0,
        "ratio_graph_over_atom": float(np.mean(pscd_vals) / max(np.mean(atom_vals), 1e-9))
        if pscd_vals else 0.0,
    }


def motif_details(vocab: Vocabulary, train_codes, valid_codes, top: int = 20):
    """Per-motif statistics + top hand-off details for the dictionary report."""
    occ = Counter()
    support: dict[int, set] = defaultdict(set)
    states: dict[int, Counter] = defaultdict(Counter)
    port_counts: dict[int, list[int]] = defaultdict(list)
    valid_occ = Counter()
    for c in train_codes:
        st = c.port_states()
        seen = set()
        for j, k in enumerate(c.occ_motif_ids):
            occ[k] += 1
            seen.add(k)
            ps = tuple(sorted(st[j]))
            states[k][ps] += 1
            port_counts[k].append(len(set(p for p, _t in st[j])))
        for k in seen:
            support[k].add(id(c))
    for c in valid_codes:
        for k in c.occ_motif_ids:
            valid_occ[k] += 1
    rows = []
    for mt in vocab.by_id.values():
        if mt.singleton:
            continue
        avg_ports = float(np.mean(port_counts[mt.mid])) if port_counts[mt.mid] else 0.0
        common = states[mt.mid].most_common(3)
        rows.append({
            "mid": mt.mid,
            "size": mt.size,
            "node_types": list(mt.canon_nt),
            "bonds": [list(b) for b in mt.canon_bonds],
            "occ_train": int(occ.get(mt.mid, 0)),
            "support_train": int(len(support.get(mt.mid, ()))),
            "occ_valid": int(valid_occ.get(mt.mid, 0)),
            "avg_ports": avg_ports,
            "joint_states": [[list(s), int(n)] for s, n in common],
        })
    rows.sort(key=lambda r: (-r["occ_train"], -r["support_train"], r["mid"]))
    return rows[:top], rows


def write_dictionary_artifacts(out: Path, vocab: Vocabulary, train_codes, valid_codes, log=print):
    top, allrows = motif_details(vocab, train_codes, valid_codes, top=20)
    write_json(out / "dictionary_top20.json", {
        "note": "learned reusable graph primitives (no chemistry names assigned)",
        "top20": top,
        "n_learned": len(allrows),
    })
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        n = len(top)
        cols = 4
        rows_n = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
        axes = np.atleast_1d(axes).reshape(-1)
        cats = sorted({c for r in allrows for c in r["node_types"]})
        cmap = plt.get_cmap("tab20")
        cidx = {c: i for i, c in enumerate(cats)}
        bond_colors = {1: "#333333", 2: "#1f77b4", 3: "#d62728"}
        for ax, r in zip(axes, top):
            g = nx.Graph()
            for s in range(r["size"]):
                g.add_node(s)
            for (a, b, t) in r["bonds"]:
                g.add_edge(a, b, t=t)
            pos = nx.spring_layout(g, seed=0)
            node_colors = [cmap(cidx[c] % 20) for c in r["node_types"]]
            nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=380,
                                   edgecolors="black")
            nx.draw_networkx_labels(g, pos, ax=ax, font_size=7)
            for (a, b, t) in r["bonds"]:
                nx.draw_networkx_edges(g, pos, edgelist=[(a, b)], ax=ax, width=2.0,
                                       edge_color=bond_colors.get(int(t), "gray"))
            ax.set_title(f"mid={r['mid']} size={r['size']}\nocc={r['occ_train']} sup={r['support_train']} "
                         f"ports={r['avg_ports']:.1f}", fontsize=8)
            ax.axis("off")
        for ax in axes[len(top):]:
            ax.axis("off")
        fig.suptitle("PSCD-v0 learned reusable graph primitives (top-20 by occurrence) "
                     "| node colour = atom category, edge colour = bond type", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "dictionary_top20.png", dpi=130)
        plt.close(fig)
        log("[A] wrote dictionary_top20.json + .png")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        log(f"[A] dictionary plot skipped: {exc!r}")
    return top


# ===========================================================================
# 7. Stage B — tiny matched readers
# ===========================================================================
def _torch():
    import torch

    return torch


def build_comp_arrays(sc: StructuredCode, vocab: Vocabulary):
    """Node/edge arrays for the bipartite composition graph."""
    M = len(sc.occ_motif_ids)
    Ec = len(sc.comp_edges)
    node_feat_kind = np.zeros(M + Ec, dtype=np.int64)   # 0 motif, 1 bond
    node_feat_id = np.zeros(M + Ec, dtype=np.int64)
    node_feat_id[:M] = sc.occ_motif_ids
    node_feat_id[M:] = [t for (_i, _pi, t, _j, _pj) in sc.comp_edges]
    src, dst, port = [], [], []
    for idx, (i, pi, t, j, pj) in enumerate(sc.comp_edges):
        c = M + idx
        for (s_node, d_node, p) in ((i, c, pi), (c, i, pi), (j, c, pj), (c, j, pj)):
            src.append(s_node)
            dst.append(d_node)
            port.append(p)
    return {
        "node_kind": node_feat_kind,
        "node_id": node_feat_id,
        "n_motif": M,
        "edges": (np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)),
        "ports": np.asarray(port, dtype=np.int64),
    }


def build_raw_arrays(mol: Mol):
    src, dst, bt = [], [], []
    for i, (a, b) in enumerate(mol.bonds):
        t = int(mol.bond_types[i])
        src.extend([a, b])
        dst.extend([b, a])
        bt.extend([t, t])
    return {
        "node_type": mol.node_types.astype(np.int64),
        "n_nodes": mol.n,
        "edges": (np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)),
        "bond_types": np.asarray(bt, dtype=np.int64),
    }


def make_module(kind: str, n_motif: int, n_bond: int, n_port: int,
                d: int = D_MODEL, layers: int = N_LAYERS):
    torch = _torch()
    nn = torch.nn

    class _Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.d = d
            self.layers = layers
            if kind in ("comp", "bag"):
                self.E_M = nn.Embedding(n_motif, d)
            if kind == "comp":
                self.E_B = nn.Embedding(n_bond, d)
                self.E_P = nn.Embedding(n_port, d)
            if kind == "raw":
                self.E_A = nn.Embedding(n_motif, d)
                self.E_B = nn.Embedding(n_bond, d)
            if kind in ("comp", "raw"):
                self.W_self = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.W_msg = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.W_edge = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))

        def _mp(self, h, edge_index, edge_feat, n_nodes):
            torch = _torch()
            for li in range(self.layers):
                src, dst = edge_index
                msg = self.W_msg[li](h)[src] + self.W_edge[li](edge_feat)
                agg = torch.zeros(n_nodes, self.d, device=h.device, dtype=h.dtype)
                agg.index_add_(0, dst, msg)
                h = torch.nn.functional.silu(self.W_self[li](h) + agg)
            return h

        def _pool(self, h, pool_index, n_graphs):
            torch = _torch()
            out = torch.zeros(n_graphs, self.d, device=h.device, dtype=h.dtype)
            if pool_index.numel():
                out.index_add_(0, pool_index, h)
            return out

        def forward(self, batch):
            torch = _torch()
            n_graphs = batch["n_graphs"]
            if self.kind == "bag":
                h = self.E_M(batch["occ_ids"])
                z = self._pool(h, batch["pool_index"], n_graphs)
                return self.head(z).squeeze(-1)
            if self.kind == "comp":
                n_nodes = batch["n_nodes"]
                h = torch.zeros(n_nodes, self.d, device=batch["occ_ids"].device)
                M = batch["occ_ids"].shape[0]
                h[:M] = self.E_M(batch["occ_ids"])
                h[M:] = self.E_B(batch["conn_ids"])
                h = self._mp(h, batch["edge_index"], self.E_P(batch["ports"]), n_nodes)
                z = self._pool(h[:M], batch["pool_index"], n_graphs)
                return self.head(z).squeeze(-1)
            if self.kind == "raw":
                h = self.E_A(batch["atom_types"])
                edge_feat = self.E_B(batch["bond_types"])
                h = self._mp(h, batch["edge_index"], edge_feat, batch["atom_types"].shape[0])
                z = self._pool(h, batch["pool_index"], n_graphs)
                return self.head(z).squeeze(-1)
            raise ValueError(kind)

    return _Reader()


def collate(items, vocab, atom_index, bond_index, kind, device):
    """Padded/concatenated batch with a per-graph pooling index."""
    torch = _torch()
    dev = device
    n_graphs = len(items)
    if kind == "bag":
        ids, pool = [], []
        for gi, sc in enumerate(items):
            for k in sc.occ_motif_ids:
                ids.append(int(k))
                pool.append(gi)
        return {
            "occ_ids": torch.as_tensor(ids, dtype=torch.long, device=dev),
            "pool_index": torch.as_tensor(pool, dtype=torch.long, device=dev),
            "n_graphs": n_graphs,
        }
    if kind == "comp":
        occ_all, conn_all, src, dst, port, pool = [], [], [], [], [], []
        off = 0
        for gi, sc in enumerate(items):
            a = build_comp_arrays(sc, vocab)
            M = a["n_motif"]
            occ_all.extend(a["node_id"][:M].tolist())
            conn_all.extend(int(bond_index[int(t)]) for t in a["node_id"][M:].tolist())
            pool.extend([gi] * M)
            s, d = a["edges"]
            if s.size:
                src.extend((s + off).tolist())
                dst.extend((d + off).tolist())
                port.extend(a["ports"].tolist())
            off += len(a["node_id"])
        return {
            "occ_ids": torch.as_tensor(occ_all, dtype=torch.long, device=dev),
            "conn_ids": torch.as_tensor(conn_all, dtype=torch.long, device=dev),
            "n_nodes": off,
            "ports": torch.as_tensor(port, dtype=torch.long, device=dev),
            "edge_index": torch.as_tensor([src, dst], dtype=torch.long, device=dev),
            "pool_index": torch.as_tensor(pool, dtype=torch.long, device=dev),
            "n_graphs": n_graphs,
        }
    if kind == "raw":
        atom_all, bt_all, src, dst, pool = [], [], [], [], []
        off = 0
        for gi, mol in enumerate(items):  # items are Mol objects
            a = build_raw_arrays(mol)
            atom_all.extend(int(atom_index[int(x)]) for x in a["node_type"].tolist())
            bt_all.extend(int(bond_index[int(t)]) for t in a["bond_types"].tolist())
            pool.extend([gi] * mol.n)
            s, d = a["edges"]
            if s.size:
                src.extend((s + off).tolist())
                dst.extend((d + off).tolist())
            off += a["n_nodes"]
        return {
            "atom_types": torch.as_tensor(atom_all, dtype=torch.long, device=dev),
            "bond_types": torch.as_tensor(bt_all, dtype=torch.long, device=dev),
            "edge_index": torch.as_tensor([src, dst], dtype=torch.long, device=dev),
            "pool_index": torch.as_tensor(pool, dtype=torch.long, device=dev),
            "n_graphs": n_graphs,
        }
    raise ValueError(kind)


def train_reader(kind, train_items, valid_items, y_train, y_valid,
                 n_motif, n_bond, n_port, atom_index=None, bond_index=None,
                 vocab=None, device="cpu", seed=SEED, log=print):
    torch = _torch()
    torch.manual_seed(seed)
    model = make_module(kind, n_motif, n_bond, n_port).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=B_LR, weight_decay=B_WD)
    ytr = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    yva = torch.as_tensor(y_valid, dtype=torch.float32, device=device)
    n = len(train_items)
    g = torch.Generator().manual_seed(seed + 4242)
    best = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def forward(items, batch_i, y):
        batch = collate(items, vocab, atom_index, bond_index, kind, device)
        pred = model(batch)
        return torch.nn.functional.l1_loss(pred, y[batch_i])

    for epoch in range(1, B_EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=g).tolist()
        for start in range(0, n, B_BATCH):
            idx = order[start:start + B_BATCH]
            items = [train_items[i] for i in idx]
            loss = forward(items, torch.as_tensor(idx, dtype=torch.long, device=device), ytr)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), B_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            vpred = []
            for start in range(0, len(valid_items), B_BATCH):
                vi = list(range(start, min(start + B_BATCH, len(valid_items))))
                batch = collate([valid_items[i] for i in vi], vocab, atom_index,
                                   bond_index, kind, device)
                vpred.append(model(batch))
            vpred = torch.cat(vpred)
            vmae = float((vpred - yva).abs().mean())
        if vmae < best:
            best = vmae
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 20 == 0:
            log(f"[B:{kind}] epoch {epoch} valid_mae={vmae:.4f} best={best:.4f}")
        if stale >= B_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        trpred, vapred = [], []
        for start in range(0, n, B_BATCH):
            ti = list(range(start, min(start + B_BATCH, n)))
            batch = collate([train_items[i] for i in ti], vocab, atom_index,
                               bond_index, kind, device)
            trpred.append(model(batch))
        trpred = torch.cat(trpred)
        for start in range(0, len(valid_items), B_BATCH):
            vi = list(range(start, min(start + B_BATCH, len(valid_items))))
            batch = collate([valid_items[i] for i in vi], vocab, atom_index,
                               bond_index, kind, device)
            vapred.append(model(batch))
        vapred = torch.cat(vapred)
    peak = float(torch.cuda.max_memory_allocated() / 1e6) if device.startswith("cuda") else 0.0
    return {
        "kind": kind,
        "train_mae": float((trpred - ytr).abs().mean()),
        "valid_mae": float((vapred - yva).abs().mean()),
        "params": int(sum(p.numel() for p in model.parameters())),
        "epochs": int(best_epoch),
        "peak_gpu_mb": peak,
    }


# ===========================================================================
# 8. Orchestration
# ===========================================================================
def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
                    encoding="utf-8")


def collision_report(keys: dict[str, list[bytes]], iso_ids: list[str]):
    report = {}
    for name, kl in keys.items():
        buckets: dict[bytes, set[str]] = defaultdict(set)
        for k, iso in zip(kl, iso_ids):
            buckets[k].add(iso)
        colliding_classes = sum(len(v) for v in buckets.values() if len(v) > 1)
        pairs = 0
        for v in buckets.values():
            c = len(v)
            if c > 1:
                pairs += c * (c - 1) // 2
        report[name] = {
            "n_keys": len(buckets),
            "colliding_iso_classes": int(colliding_classes),
            "colliding_class_pairs": int(pairs),
        }
    return report


def stage_a(train_mols, valid_mols, out, log):
    vocab = Vocabulary()
    t0 = time.time()
    log("[A] learning vocabulary (BPE, train only)")
    train_tokens, rounds = learn_vocabulary(train_mols, vocab, MERGE_RULES, log=log)
    log(f"[A] vocabulary learned in {time.time() - t0:.1f}s, |D|={len(vocab)}")

    sequence = [bytes.fromhex(r["key"]) for r in rounds]
    log("[A] tokenizing valid with the frozen sequence")
    valid_tokens = []
    for mol in valid_mols:
        valid_tokens.append(tokenize_frozen(mol, vocab, sequence, mol_rank(mol)))

    train_ranks = [mol_rank(m) for m in train_mols]
    valid_ranks = [mol_rank(m) for m in valid_mols]

    log("[A] building structured codes")
    train_codes = [build_structured_code(m, t, vocab, r)["code"]
                   for m, t, r in zip(train_mols, train_tokens, train_ranks)]
    valid_codes = [build_structured_code(m, t, vocab, r)["code"]
                   for m, t, r in zip(valid_mols, valid_tokens, valid_ranks)]

    log("[A] exact reconstruction check")
    recon = {"train": 0, "valid": 0}
    recon_fail: list[int] = []
    for split, mols, codes in (("train", train_mols, train_codes),
                               ("valid", valid_mols, valid_codes)):
        for idx, (mol, code) in enumerate(zip(mols, codes)):
            ghat = decode(code, vocab)
            if mol_iso_key(ghat) == mol_iso_key(mol):
                recon[split] += 1
            elif len(recon_fail) < 5:
                recon_fail.append(idx)
    recon["train_frac"] = recon["train"] / max(len(train_mols), 1)
    recon["valid_frac"] = recon["valid"] / max(len(valid_mols), 1)
    recon["fail_examples"] = recon_fail
    log(f"[A] exact reconstruct train={recon['train_frac']:.6f} "
        f"valid={recon['valid_frac']:.6f}")

    log("[A] collision ladder")
    keys = {"A": [], "B": [], "C": [], "D": []}
    iso_ids = []
    for mol, code in list(zip(train_mols, train_codes)) + list(zip(valid_mols, valid_codes)):
        lk = ladder_keys(code, vocab)
        for k in keys:
            keys[k].append(lk[k])
        iso_ids.append(iso_key_hex(mol))
    collisions = collision_report(keys, iso_ids)
    log(f"[A] collisions {collisions}")

    compression = {"train": compression_metrics(train_codes),
                   "valid": compression_metrics(valid_codes)}
    motif_size = {"train": aggregate_motif_size(train_codes, vocab),
                  "valid": aggregate_motif_size(valid_codes, vocab)}
    reuse = vocabulary_reuse(train_codes, vocab)
    ports = port_complexity(list(train_codes) + list(valid_codes), vocab)
    mdl = mdl_proxy(train_codes, valid_codes, train_mols, valid_mols, vocab)
    top_motifs = write_dictionary_artifacts(out, vocab, train_codes, valid_codes, log=log)

    return {
        "vocab": vocab,
        "sequence": sequence,
        "rounds": rounds,
        "train_codes": train_codes,
        "valid_codes": valid_codes,
        "reconstruction": recon,
        "collisions": collisions,
        "compression": compression,
        "motif_size": motif_size,
        "reuse": reuse,
        "ports": ports,
        "mdl": mdl,
        "top_motifs": top_motifs,
    }


def stage_a2_invariance(train_mols, vocab, sequence, n_graphs=INV_GRAPHS,
                        n_relabels=INV_RELABELS, seed=SEED, log=print):
    torch_free = np.random.default_rng(seed + 7)
    picks = torch_free.permutation(len(train_mols))[:n_graphs]
    mismatches = 0
    decode_fail = 0
    checks = 0
    for gi in picks:
        mol = train_mols[int(gi)]
        base = build_structured_code(mol, tokenize_frozen(mol, vocab, sequence, mol_rank(mol)),
                                     vocab, mol_rank(mol))
        base_key = ladder_keys(base["code"], vocab)["D"]
        n = mol.n
        m = mol.m
        for _ in range(n_relabels):
            perm_atom = np.random.default_rng().permutation(n)
            perm_bond = np.random.default_rng().permutation(m)
            pm = permute_mol(mol, perm_atom, perm_bond)
            toks = tokenize_frozen(pm, vocab, sequence, mol_rank(pm))
            code = build_structured_code(pm, toks, vocab, mol_rank(pm))
            key = ladder_keys(code["code"], vocab)["D"]
            checks += 1
            if key != base_key:
                mismatches += 1
            ghat = decode(code["code"], vocab)
            if mol_iso_key(ghat) != mol_iso_key(pm):
                decode_fail += 1
    return {"checks": checks, "representation_mismatches": int(mismatches),
            "decode_failures": int(decode_fail), "graphs": int(len(picks)),
            "relabels": int(n_relabels)}


def stage_b(data, train_mols, valid_mols, y_train, y_valid, device, log):
    vocab = data["vocab"]
    train_codes = data["train_codes"]
    valid_codes = data["valid_codes"]
    atom_cats = sorted({int(x) for mol in train_mols for x in mol.node_types.tolist()})
    bond_cats = sorted({int(x) for mol in train_mols for x in mol.bond_types.tolist()})
    atom_index = {c: i for i, c in enumerate(atom_cats)}
    bond_index = {c: i for i, c in enumerate(bond_cats)}
    n_motif = len(vocab)
    n_bond = len(bond_cats)
    n_port = MAX_MOTIF
    results = {}
    log("[B] training composition reader")
    results["comp"] = train_reader("comp", train_codes, valid_codes, y_train, y_valid,
                                   n_motif, n_bond, n_port, atom_index=atom_index,
                                   bond_index=bond_index, vocab=vocab,
                                   device=device, log=log)
    log("[B] training motif-bag control")
    results["bag"] = train_reader("bag", train_codes, valid_codes, y_train, y_valid,
                                  n_motif, n_bond, n_port, atom_index=atom_index,
                                  bond_index=bond_index, vocab=vocab,
                                  device=device, log=log)
    log("[B] training raw atom-graph control")
    results["raw"] = train_reader("raw", train_mols, valid_mols, y_train, y_valid,
                                  n_motif, n_bond, n_port, atom_index=atom_index,
                                  bond_index=bond_index, vocab=vocab,
                                  device=device, log=log)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--stage", choices=["A", "B", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-invariance", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="smoke override only")
    parser.add_argument("--patience", type=int, default=None, help="smoke override only")
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log("=== PSCD-v0 start ===")
    provenance = {
        "round": "PSCD-v0",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "data_root": str(args.data_root),
        "seed": args.seed,
        "stage": args.stage,
        "limit": args.limit,
        "merge_rules": MERGE_RULES,
        "max_motif": MAX_MOTIF,
        "device": args.device,
    }
    write_json(out / "provenance.json", provenance)

    limit = args.limit
    if args.smoke and limit is None:
        limit = 300
    global B_EPOCHS, B_PATIENCE
    if args.epochs is not None:
        B_EPOCHS = args.epochs
    if args.patience is not None:
        B_PATIENCE = args.patience
    train_mols, y_train = load_mols_and_y(args.data_root, "train", limit)
    valid_mols, y_valid = load_mols_and_y(args.data_root, "valid",
                                          None if limit is None else max(100, limit // 3))
    log(f"loaded train={len(train_mols)} valid={len(valid_mols)} (official test never loaded)")

    # asserted graph semantics
    assert len({int(x) for mol in train_mols for x in mol.node_types.tolist()}) <= 21
    log("graph semantics: undirected bonds, categorical atom/bond labels")

    summary: dict[str, Any] = {"provenance": provenance}

    if args.stage in ("A", "both"):
        data = stage_a(train_mols, valid_mols, out, log)
        # drop heavy vocab object from JSON
        summary["stage_a"] = {
            "rounds": data["rounds"],
            "reconstruction": data["reconstruction"],
            "collisions": data["collisions"],
            "compression": data["compression"],
            "motif_size": data["motif_size"],
            "reuse": data["reuse"],
            "ports": data["ports"],
            "mdl": data["mdl"],
            "top_motifs": data["top_motifs"],
            "n_motif_types": len(data["vocab"]),
        }
        if not args.skip_invariance:
            log("[A] permutation invariance test")
            inv = stage_a2_invariance(train_mols, data["vocab"], data["sequence"],
                                      log=log)
            summary["stage_a"]["invariance"] = inv
            log(f"[A] invariance {inv}")
        # gates
        mean_ratio = data["compression"]["train"]["mean_ratio"]
        gates = {
            "A1_exactness": bool(data["reconstruction"]["train_frac"] == 1.0
                                 and data["reconstruction"]["valid_frac"] == 1.0),
            "A3_compression": bool(mean_ratio <= 0.55),
            "A4_reuse": bool(data["reuse"]["low_freq_frac"] <= 0.5),
            "A5_port_complexity": bool(data["ports"]["p_over_size_ge4_mean"] <= 0.75),
            "A6_no_collision": bool(data["collisions"]["D"]["colliding_class_pairs"] == 0),
        }
        if "invariance" in summary["stage_a"]:
            inv = summary["stage_a"]["invariance"]
            gates["A2_invariance"] = bool(inv["representation_mismatches"] == 0
                                          and inv["decode_failures"] == 0)
        gates["all_pass"] = all(gates.values())
        summary["stage_a"]["gates"] = gates
        log(f"[A] gates {gates}")

        with (out / "stageA.pkl").open("wb") as fh:
            pickle.dump({"vocab": data["vocab"], "sequence": data["sequence"],
                         "train_codes": data["train_codes"],
                         "valid_codes": data["valid_codes"], "rounds": data["rounds"]}, fh)

    if args.stage == "B":
        with (out / "stageA.pkl").open("rb") as fh:
            data = pickle.load(fh)
    elif args.stage == "both":
        pass

    if args.stage in ("B", "both"):
        summary.setdefault("stage_a", {})
        gates = summary.get("stage_a", {}).get("gates", {})
        if args.stage == "both" and not gates.get("all_pass", False):
            log("[B] Stage A gates failed -> Stage B is NOT run (pre-registered)")
            summary["stage_b"] = {"skipped": True, "reason": "stage A gate failure"}
        else:
            log("[B] running capacity probe")
            results = stage_b(data, train_mols, valid_mols, y_train, y_valid,
                              args.device, log)
            summary["stage_b"] = results
            mean_ratio = float(np.mean([len(c.occ_motif_ids) / max(c.n, 1)
                                        for c in data["train_codes"]]))
            comp = results["comp"]["valid_mae"]
            raw = results["raw"]["valid_mae"]
            bag = results["bag"]["valid_mae"]
            summary["stage_b"]["interpretation"] = {
                "strong": bool(comp <= raw + 0.03 and mean_ratio <= 0.55),
                "composition_matters": bool(bag - comp >= 0.02),
                "weak_abstraction": bool(comp > raw + 0.08),
            }

    write_json(out / "SUMMARY.json", summary)
    log("=== PSCD-v0 done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
