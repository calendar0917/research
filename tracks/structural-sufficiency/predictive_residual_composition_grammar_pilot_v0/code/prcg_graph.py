"""Discrete graph-word grammar core for PRCG-v0.

A *word* is a connected induced atom support of a real molecule graph.  Words
are formed by merging two adjacent words along the real bonds between them,
starting from single atoms.  Everything in this module is label-free, exact and
independent of atom numbering:

* ``canonical_ranks`` gives an exact individualisation-refinement canonical
  order of a labelled molecular graph.  It is used only to turn the *structure*
  into a deterministic linear order; the raw input index never breaks a tie.
* ``word_signature`` computes a true canonical certificate of an induced
  subgraph by enumerating all ``k! <= 120`` orderings (no hashing assumption),
  together with the automorphism-orbit class and orbit size of every atom.
* ``Grammar`` is an ordered list of merge rules; applying it to a molecule is a
  deterministic greedy maximal matching of the rule's occurrences, ordered by
  the canonical atom ranks.

The word *type* uses only the frozen core fields (atomic number, formal charge,
aromaticity for atoms; bond type, conjugation for bonds).  Every other OGB field
stays available as a model context feature and is never part of a type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MAX_WORD_ATOMS = 5
ATOM_CORE_COLUMNS = (0, 3, 7)   # atomic number, formal charge, aromatic
BOND_CORE_COLUMNS = (0, 2)      # bond type, conjugated


class CanonicalBudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# exact canonical order of a labelled graph
# ---------------------------------------------------------------------------
def _intern(signatures: Sequence[Any]) -> list[int]:
    """Deterministic dense relabelling: ids follow the sorted signature order."""
    uniq = sorted(set(signatures))
    index = {sig: i for i, sig in enumerate(uniq)}
    return [index[sig] for sig in signatures]


def canonical_ranks(node_labels: Sequence[int], edge_sets: Sequence[Sequence[tuple[int, int]]],
                    n_nodes: int, budget: int = 20000) -> list[int]:
    """Exact canonical labelling by individualisation + colour refinement.

    ``edge_sets[v]`` lists ``(neighbour, edge_label)``.  Returns ``rank[v]`` =
    position of ``v`` in a canonical order (0 = first).  The search is
deterministic and label-independent: the target cell is always the first
    non-singleton colour class (an invariant choice) and every vertex of it is
    individualised, keeping the lexicographically smallest adjacency string.
    The resulting order is therefore a graph invariant up to automorphism
    (``rank_2[pi(i)] == rank_1[sigma(i)]`` for some automorphism ``sigma``),
    which is all that is needed: ``tests/test_prcg.py`` verifies this against
    brute force and the renumbering check verifies the induced map is an
    automorphism.  Raises ``CanonicalBudgetExceeded`` instead of silently
    returning an unverifiable order.
    """
    if n_nodes == 0:
        return []
    nbrs: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for v in range(n_nodes):
        for u, lab in edge_sets[v]:
            if u == v:
                raise ValueError("self loop")
            nbrs[v].append((int(u), int(lab)))
    for v in range(n_nodes):
        nbrs[v].sort()
    counter = [0]
    cache: dict[tuple[int, ...], tuple[Any, list[int]]] = {}

    def refine(colours: list[int]) -> list[int]:
        while True:
            sigs = []
            for v in range(n_nodes):
                nb = sorted((lab, colours[u]) for u, lab in nbrs[v])
                sigs.append((colours[v], tuple(nb)))
            new = _intern(sigs)
            if len(set(new)) == len(set(colours)):
                return new
            colours = new

    def search(colours: list[int]) -> tuple[Any, list[int]]:
        key = tuple(colours)
        hit = cache.get(key)
        if hit is not None:
            return hit
        counter[0] += 1
        if counter[0] > budget:
            raise CanonicalBudgetExceeded(f"budget {budget} exceeded on {n_nodes} nodes")
        colours = refine(colours)
        counts: dict[int, int] = {}
        for c in colours:
            counts[c] = counts.get(c, 0) + 1
        cells = sorted(c for c, s in counts.items() if s > 1)
        if not cells:
            seq = sorted(range(n_nodes), key=lambda v: colours[v])
            out = (_canonical_string(seq, node_labels, nbrs, n_nodes), seq)
            cache[key] = out
            return out
        best: tuple[Any, list[int]] | None = None
        new_colour = max(colours) + 1
        cell = cells[0]
        for v in [x for x in range(n_nodes) if colours[x] == cell]:
            trial = list(colours)
            trial[v] = new_colour
            sig, seq = search(trial)
            if best is None or sig < best[0]:
                best = (sig, seq)
        assert best is not None
        cache[key] = best
        return best

    _, seq = search(_intern([int(x) for x in node_labels]))
    ranks = [0] * n_nodes
    for pos, v in enumerate(seq):
        ranks[v] = pos
    return ranks


def _canonical_string(order: Sequence[int], node_labels: Sequence[int],
                      nbrs: Sequence[Sequence[tuple[int, int]]], n_nodes: int) -> tuple:
    pos = [0] * n_nodes
    for i, v in enumerate(order):
        pos[v] = i
    atoms = tuple(int(node_labels[v]) for v in order)
    edges = []
    for v in order:
        for u, lab in nbrs[v]:
            if pos[u] > pos[v]:
                edges.append((pos[v], pos[u], int(lab)))
    edges.sort()
    return atoms + tuple(x for e in edges for x in e) + (len(edges),)


# ---------------------------------------------------------------------------
# one molecule
# ---------------------------------------------------------------------------
@dataclass
class Molecule:
    row: int
    n_atoms: int
    atom_core: np.ndarray          # [n, 3] int64 (atomic number, formal charge, aromatic)
    atom_intrinsic: np.ndarray     # [n, 4] int64 (chirality, numH, radical, hybridisation)
    atom_context: np.ndarray       # [n, 5] float (degree, min_ring, in_cycle, log1p deg, deg==0)
    bond_u: np.ndarray             # [m]
    bond_v: np.ndarray
    bond_core: np.ndarray          # [m, 2] int64 (bond type, conjugated)
    bond_stereo: np.ndarray        # [m]
    adj: list[list[tuple[int, int]]]     # (neighbour atom, bond index)
    ranks: list[int]               # canonical atom ranks
    n_components: int = 1
    _word_cache: dict[tuple[int, ...], "WordSignature"] = field(default_factory=dict)

    @property
    def n_bonds(self) -> int:
        return int(self.bond_u.size)

    def bonds_between(self, a: Iterable[int], b: Iterable[int]) -> list[int]:
        sa = set(int(x) for x in a)
        out = []
        for bidx in range(self.n_bonds):
            u = int(self.bond_u[bidx])
            v = int(self.bond_v[bidx])
            if (u in sa) != (v in sa):
                other = v if u in sa else u
                if other in set(int(x) for x in b):
                    out.append(bidx)
        return sorted(out)

    def internal_bonds(self, support: Sequence[int]) -> list[int]:
        s = set(int(x) for x in support)
        out = []
        for bidx in range(self.n_bonds):
            if int(self.bond_u[bidx]) in s and int(self.bond_v[bidx]) in s:
                out.append(bidx)
        return out

    def is_connected(self, support: Sequence[int]) -> bool:
        s = [int(x) for x in support]
        if len(s) <= 1:
            return True
        allowed = set(s)
        seen = {s[0]}
        stack = [s[0]]
        while stack:
            x = stack.pop()
            for y, _ in self.adj[x]:
                if y in allowed and y not in seen:
                    seen.add(y)
                    stack.append(y)
        return len(seen) == len(s)


@dataclass(frozen=True)
class WordSignature:
    """Exact certificate of one induced connected support."""
    cert: str                         # canonical type certificate (core fields only)
    order: tuple[int, ...]            # atom ids in the canonical order
    orbit_rep: tuple[int, ...]        # per canonical slot: representative slot of its orbit
    orbit_size: tuple[int, ...]       # per canonical slot: size of its orbit
    atom_codes: tuple[int, ...]       # per canonical slot: core atom code
    bond_codes: tuple[int, ...]       # canonical upper-triangle bond codes (0 = no bond)

    @property
    def size(self) -> int:
        return len(self.order)


def _atom_code(core: Sequence[int]) -> int:
    return int(core[0]) * 1024 + int(core[1]) * 8 + int(core[2])


def _bond_code(bcore: Sequence[int]) -> int:
    return int(bcore[0]) * 4 + int(bcore[1]) + 1


def word_signature(mol: Molecule, support: Sequence[int]) -> WordSignature:
    """Exact canonical certificate + automorphism orbits of one induced support."""
    key = tuple(sorted(int(x) for x in support))
    cached = mol._word_cache.get(key)
    if cached is not None:
        return cached
    k = len(key)
    if k > MAX_WORD_ATOMS:
        raise ValueError(f"word size {k} exceeds {MAX_WORD_ATOMS}")
    if not mol.is_connected(key):
        raise ValueError("word support is not connected")
    codes = [_atom_code(mol.atom_core[a]) for a in key]
    mat = [[0] * k for _ in range(k)]
    bidx: dict[tuple[int, int], int] = {}
    for b in mol.internal_bonds(key):
        u, v = int(mol.bond_u[b]), int(mol.bond_v[b])
        i, j = key.index(u), key.index(v)
        code = _bond_code(mol.bond_core[b])
        mat[i][j] = mat[j][i] = code
        bidx[(min(i, j), max(i, j))] = b
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    best: tuple | None = None
    aut: list[tuple[int, ...]] = []
    for perm in permutations(range(k)):
        flat = tuple(mat[perm[i]][perm[j]] for i, j in pairs)
        sig = tuple(codes[perm[i]] for i in range(k)) + flat
        if best is None or sig < best:
            best = sig
            aut = [perm]
        elif sig == best:
            aut.append(perm)
    assert best is not None
    # orbit of each canonical slot, expressed as the set of positions it can
    # occupy inside the support; orbit ids are the *smallest canonical slot* of
    # the orbit, so they are independent of the atom numbering
    members = [frozenset(perm[slot] for perm in aut) for slot in range(k)]
    rep = [min(j for j in range(k) if members[j] == members[slot]) for slot in range(k)]
    size = [len(members[slot]) for slot in range(k)]
    # certificate: atoms then bonds, one token per item, exact (no hashing)
    cert_atoms = ".".join(str(x) for x in best[:k])
    cert_bonds = ".".join(f"{i}-{j}:{v}" for (i, j), v in zip(pairs, best[k:]) if v)
    cert = f"k{k}[{cert_atoms}][{cert_bonds}]"
    out = WordSignature(cert=cert, order=tuple(key[a] for a in aut[0]),
                        orbit_rep=tuple(rep), orbit_size=tuple(size),
                        atom_codes=tuple(best[:k]), bond_codes=tuple(best[k:]))
    mol._word_cache[key] = out
    return out


# ---------------------------------------------------------------------------
# partitions and merges
# ---------------------------------------------------------------------------
@dataclass
class Word:
    support: tuple[int, ...]          # sorted atom ids
    sig: WordSignature
    rule: int = -1                    # -1 = atom primitive, >=0 = grammar rule id
    children: tuple[int, int] = (-1, -1)   # word indices in the partition history
    index: int = -1

    @property
    def size(self) -> int:
        return len(self.support)

    @property
    def cert(self) -> str:
        return self.sig.cert


@dataclass
class MergeOccurrence:
    left: int
    right: int
    left_support: tuple[int, ...]
    right_support: tuple[int, ...]
    crossing: tuple[int, ...]         # bond indices
    pattern: str                      # canonical attachment-pattern key
    key: tuple                        # deterministic, structure-derived order key
    left_cert: str = ""
    right_cert: str = ""


def _port_slot(sig: WordSignature, atom: int) -> int:
    """Canonical slot of ``atom`` inside ``sig`` under the canonical order."""
    return sig.order.index(int(atom))


def _port_role_id(sig: WordSignature, atom: int) -> tuple[int, int]:
    slot = _port_slot(sig, atom)
    return (int(sig.orbit_rep[slot]), int(sig.orbit_size[slot]))


def _pattern_key(mol: Molecule, left: Word, right: Word,
                 crossing: Sequence[int]) -> str:
    """Canonical attachment pattern of one merge (independent of atom ids)."""
    items = []
    for bidx in crossing:
        b = int(bidx)
        u, v = int(mol.bond_u[b]), int(mol.bond_v[b])
        if u in left.support:
            pa, pb = u, v
        else:
            pa, pb = v, u
        ra = _port_role_id(left.sig, pa)
        rb = _port_role_id(right.sig, pb)
        items.append((int(mol.bond_core[b][0]), int(mol.bond_core[b][1]),
                      ra[0], rb[0], ra[1], rb[1]))
    items.sort()
    body = "|".join(f"b{t}c{j}:L{a}R{b}:{sa}x{sb}" for t, j, a, b, sa, sb in items)
    # the rule identity is (child type A, child type B, attachment pattern)
    return f"{left.cert}@@{right.cert}@@{body}"


def merge_occurrences(mol: Molecule, words: Sequence[Word],
                      left_cert: str | None = None,
                      right_cert: str | None = None) -> list[MergeOccurrence]:
    """All legal adjacent word pairs (|A|+|B| <= MAX_WORD_ATOMS).

    ``left_cert``/``right_cert`` restrict the enumeration to one ordered child
    type pair; the orientation is canonical (the lexicographically smaller
    certificate is the left child) so a rule is orientation-free.
    """
    out: list[MergeOccurrence] = []
    atom_owner: dict[int, int] = {}
    for wi, w in enumerate(words):
        for a in w.support:
            atom_owner[a] = wi
    by_pair: dict[tuple[int, int], list[int]] = {}
    for bidx in range(mol.n_bonds):
        u, v = int(mol.bond_u[bidx]), int(mol.bond_v[bidx])
        wi, wj = atom_owner[u], atom_owner[v]
        if wi == wj:
            continue
        key = (min(wi, wj), max(wi, wj))
        by_pair.setdefault(key, []).append(bidx)
    for (wi, wj), bonds in sorted(by_pair.items()):
        wa, wb = words[wi], words[wj]
        if wa.size + wb.size > MAX_WORD_ATOMS:
            continue
        if wa.cert <= wb.cert:
            left, right, crossing = wa, wb, bonds
        else:
            left, right, crossing = wb, wa, bonds
        if left_cert is not None and left.cert != left_cert:
            continue
        if right_cert is not None and right.cert != right_cert:
            continue
        pattern = _pattern_key(mol, left, right, crossing)
        ranks = mol.ranks
        key = (min(ranks[a] for a in left.support), min(ranks[a] for a in right.support))
        key = (min(key), max(key)) + tuple(sorted(ranks[a] for a in left.support + right.support))
        out.append(MergeOccurrence(left=left.index, right=right.index,
                                  left_support=left.support, right_support=right.support,
                                  crossing=tuple(sorted(int(b) for b in crossing)),
                                  pattern=pattern, key=key,
                                  left_cert=left.cert, right_cert=right.cert))
    out.sort(key=lambda o: o.key)
    return out


def parent_signature(mol: Molecule, occ: MergeOccurrence) -> WordSignature:
    support = tuple(sorted(occ.left_support + occ.right_support))
    if len(support) > MAX_WORD_ATOMS:
        raise ValueError("parent too large")
    if len(set(support)) != len(support):
        raise ValueError("overlapping children")
    return word_signature(mol, support)


def greedy_select(occurrences: Sequence[MergeOccurrence]) -> list[MergeOccurrence]:
    """Deterministic maximal matching: occurrences are pre-sorted by a
    structure-derived key, so the outcome never depends on atom numbering."""
    used: set[int] = set()
    picked: list[MergeOccurrence] = []
    for occ in occurrences:
        if occ.left in used or occ.right in used:
            continue
        used.add(occ.left)
        used.add(occ.right)
        picked.append(occ)
    return picked


# ---------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------
@dataclass
class Rule:
    index: int
    left_cert: str
    right_cert: str
    pattern: str
    parent_cert: str
    parent_size: int
    n_crossing: int
    origin: str = "predictive"

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "left_cert": self.left_cert,
                "right_cert": self.right_cert, "pattern": self.pattern,
                "parent_cert": self.parent_cert, "parent_size": self.parent_size,
                "n_crossing_bonds": self.n_crossing, "origin": self.origin,
                "key": f"{self.left_cert} + {self.right_cert} @ {self.pattern} -> {self.parent_cert}"}


@dataclass
class Partition:
    words: list[Word]
    history: list[Word]
    applied: list[int]

    def copy(self) -> "Partition":
        return Partition(words=list(self.words), history=list(self.history),
                         applied=list(self.applied))


def initial_partition(mol: Molecule) -> Partition:
    words: list[Word] = []
    for a in range(mol.n_atoms):
        sig = word_signature(mol, (a,))
        w = Word(support=(a,), sig=sig, rule=-1, children=(-1, -1), index=len(words))
        words.append(w)
    return Partition(words=words, history=list(words), applied=[])


def apply_rule(mol: Molecule, part: Partition, rule: Rule,
               signed: bool = True) -> tuple[Partition, int, dict[str, bool]]:
    """Apply one rule greedily to the current partition.  Returns the new
    partition, the number of applications and small consistency flags."""
    occ = merge_occurrences(mol, part.words, rule.left_cert, rule.right_cert)
    occ = [o for o in occ if o.pattern == rule.pattern]
    picked = greedy_select(occ)
    flags = {"parent_cert_ok": True}
    if not picked:
        return part.copy(), 0, flags
    new_words: list[Word] = []
    history = list(part.history)
    replaced: dict[int, Word] = {}
    for o in picked:
        sig = parent_signature(mol, o)
        if signed and sig.cert != rule.parent_cert:
            flags["parent_cert_ok"] = False
        w = Word(support=tuple(sorted(o.left_support + o.right_support)), sig=sig,
                 rule=rule.index, children=(o.left, o.right), index=len(history))
        history.append(w)
        replaced[o.left] = w
        replaced[o.right] = w
    emitted: set[int] = set()
    for w in part.words:
        if w.index in replaced:
            new = replaced[w.index]
            if new.index not in emitted:
                emitted.add(new.index)
                new_words.append(new)
        else:
            new_words.append(w)
    # word indices always refer to the molecule-wide history, never to a position
    new_part = Partition(words=new_words, history=history,
                         applied=part.applied + [rule.index])
    return new_part, len(picked), flags


class Grammar:
    def __init__(self, rules: Sequence[Rule] | None = None) -> None:
        self.rules: list[Rule] = list(rules or [])

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def parent_certs(self) -> set[str]:
        return {r.parent_cert for r in self.rules}

    def add(self, rule: Rule) -> None:
        self.rules.append(rule)

    def apply(self, mol: Molecule, record_log: bool = False) -> tuple[Partition, dict[str, Any]]:
        part = initial_partition(mol)
        applied: list[int] = []
        bad = []
        for rule in self.rules:
            part, n, flags = apply_rule(mol, part, rule)
            applied.append(n)
            if not flags["parent_cert_ok"]:
                bad.append(rule.index)
        log = {"n_applied": applied, "parent_cert_mismatch": bad,
               "n_words": len(part.words)} if record_log else {}
        return part, log

    def to_dict(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.rules]


def count_pattern_occurrences(mol: Molecule, part: Partition, pattern: str,
                              left_cert: str, right_cert: str) -> int:
    occ = [o for o in merge_occurrences(mol, part.words, left_cert, right_cert)
           if o.pattern == pattern]
    return len(greedy_select(occ))