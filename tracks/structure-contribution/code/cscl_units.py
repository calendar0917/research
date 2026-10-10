"""CSCL RINGCHAIN unit partition (deterministic, label-blind).

Research track: ``structure-contribution`` (see notes/research_core.md and
notes/v0_protocol.md §2).

**Version: cscl-correctness-v1.**  This module fixes two confirmed cscl-v0
defects (each reproduced with minimal counterexamples before fixing; see
``notes/correctness_information_audit_v1.md``):

1. *Signature*: v0 dense-remapped WL colours by first-appearance order, which
   (a) broke relabel invariance and (b) erased atom/bond types from the
   signature entirely (C–O, C–N, C–F chains collided; C–C/C=C/C#C collided).
   v1 uses hash-based labelled 1-WL (content-hash colours, sorted multiset).
2. *Bond accounting*: v0 fed PyG's double-stored directed edges into the
   partition, double counting every physical bond (six-membered ring:
   cyclomatic 7 instead of 1; relation histograms doubled; ``is_terminal``
   broken).  v1 canonicalizes to unique undirected bonds first
   (:func:`unique_undirected_bonds`).  The partition (atom ownership)
   algorithm itself is unchanged.

Design contract (frozen in v0_protocol.md):

* **Non-overlapping**: every atom belongs to exactly one unit (ring system or
  chain unit).
* **Every real chemical bond** is either intra-unit (both endpoints in the same
  unit) or an inter-unit connection; no bond is lost or double counted.
* **Ring preserving**: ring atoms are grouped into maximal ring systems
  (fused/spiro/bridged rings merged through shared atoms); acyclic remainder
  atoms form chain units (whole-molecule acyclic compounds are a single chain
  unit).
* **Deterministic and label-blind**: only atom categories (28) and bond
  categories (4) are read; never the target ``y``.
* **Cross-molecule type alignment**: unit identity is a canonical signature
  computed from the unit-internal attributed graph (4-round hash-based
  labelled WL with bond-type edge labels plus a ring/chain kind bit and
  size).  Invariant to atom relabeling; distinct non-isomorphic units may
  collide in rare cases (1-WL limitation, documented; equality is not proof
  of isomorphism).

The vocabulary (signature -> id) is built on **fit data only** (v0_protocol
§2.4): signatures occurring < ``min_count`` times map to (kind, size-bucket)
UNK buckets so that unseen structures have an explicit, comparable handling.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx

ATOM_CATEGORIES = 28
BOND_CATEGORIES = 4

#: size buckets for UNK fallback typing (v0_protocol §2.4)
SIZE_BUCKET_EDGES = (1, 2, 3, 4, 6, 8, 10)

KIND_RING = 0
KIND_CHAIN = 1

#: WL refinement rounds for the unit signature (v0_protocol §2.3)
WL_ROUNDS = 4

#: reserved global UNK id (dev signature whose (kind, size-bucket) was never
#: seen even as an UNK bucket during fit)
GLOBAL_UNK_KEY = "UNK|GLOBAL"


# ---------------------------------------------------------------------------
# physical-bond canonicalization (PyG ZINC stores every bond twice)
# ---------------------------------------------------------------------------


def unique_undirected_bonds(
    edge_index: list[tuple[int, int]], edge_types: list[int]
) -> list[tuple[int, int, int]]:
    """Canonicalize PyG-style directed edge lists to unique undirected bonds.

    PyG ZINC builds ``edge_index`` as ``symmetric_adj.nonzero()`` (verified on
    the local ``data/ZINC`` snapshot: every physical bond appears exactly
    twice, once per direction, with consistent bond types; no self-loops; no
    parallel multi-edges).  Treating both directions as independent edges —
    as cscl-v0 did — double counts every bond in intra/inter histograms,
    inflates ``cyclomatic = E - V + 1`` (six-membered ring: 12 stored edges →
    v0 reported 7 instead of 1), corrupts relation bond histograms, and breaks
    the ``is_terminal`` descriptor (attach counts doubled).

    Rules (v1):
    * each physical bond is emitted exactly once with ``u < v``;
    * both directions must carry the same bond type, else ``ValueError``;
    * self-loops, duplicate same-direction entries and >2 occurrences of one
      physical bond raise ``ValueError`` (record nothing silently).
    """
    counts: dict[tuple[int, int], list[int]] = {}
    for (u, v), bt in zip(edge_index, edge_types):
        u, v, bt = int(u), int(v), int(bt)
        if u == v:
            raise ValueError(f"self-loop at atom {u}: not a physical bond")
        key = (u, v) if u < v else (v, u)
        entry = counts.setdefault(key, [])
        entry.append(bt)
    bonds: list[tuple[int, int, int]] = []
    for (u, v), bts in sorted(counts.items()):
        if len(bts) > 2:
            raise ValueError(f"bond ({u},{v}) occurs {len(bts)} times: parallel multi-edge unsupported")
        if len(bts) == 2 and bts[0] != bts[1]:
            raise ValueError(f"inconsistent bond type across directions for ({u},{v}): {bts}")
        bonds.append((u, v, bts[0]))
    return bonds


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    """One structural unit: a unique-ownership partition of atoms."""

    unit_id: int
    kind: int  # KIND_RING or KIND_CHAIN
    atoms: list[int]
    intra_bonds: list[tuple[int, int, int]]  # (u, v, bond_type) within unit
    inter_bonds: list[tuple[int, int, int]]  # (u, v, bond_type) to other units
    signature: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)


@dataclass
class UnitPartition:
    units: list[Unit]
    n_atoms: int
    #: unit id per atom (len == n_atoms); unit ids are 0..n_units-1 by
    #: (kind, min atom index) ordering for determinism
    atom_unit: list[int] = field(default_factory=list)

    @property
    def n_units(self) -> int:
        return len(self.units)

    def inter_relations(self) -> dict[tuple[int, int], list[int]]:
        """Unordered unit-pair -> list of bond types (each real bond once)."""
        rel: dict[tuple[int, int], list[int]] = {}
        for unit in self.units:
            if not unit.inter_bonds:
                continue
            atoms_set = set(unit.atoms)
            for u, v, bt in unit.inter_bonds:
                other_id = self.atom_unit[v] if u in atoms_set else self.atom_unit[u]
                key = (min(unit.unit_id, other_id), max(unit.unit_id, other_id))
                rel.setdefault(key, []).append(bt)
        return rel


def _biconnected_blocks(n_atoms: int, edges: list[tuple[int, int, int]]) -> list[tuple[set[int], list[tuple[int, int, int]]]]:
    """Biconnected components (atoms set, edge list) of the molecular graph."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    graph.add_edges_from((u, v) for u, v, _ in edges)
    blocks: list[tuple[set[int], list[tuple[int, int, int]]]] = []
    for comp in nx.biconnected_components(graph):
        comp = set(comp)
        block_edges = [(u, v, bt) for u, v, bt in edges if u in comp and v in comp]
        blocks.append((comp, block_edges))
    return blocks


def partition_units(atom_types: list[int], edges: list[tuple[int, int, int]]) -> UnitPartition:
    """RINGCHAIN-v0 partition (v0_protocol §2).

    ``edges`` must be **unique undirected physical bonds** (one entry per
    bond, arbitrary direction; see :func:`unique_undirected_bonds`).  Ring
    systems = maximal unions of cycle blocks (biconnected blocks with
    >= 3 edges) sharing atoms; chain units = connected components of the
    remaining acyclic atoms under the acyclic induced subgraph.
    """
    blocks = _biconnected_blocks(len(atom_types), edges)
    cycle_atoms: set[int] = set()
    cycle_edges: list[tuple[int, int, int]] = []
    for atoms, bedges in blocks:
        if len(bedges) >= 3:  # 2-connected block containing a cycle
            cycle_atoms |= atoms
            cycle_edges.extend(bedges)

    # merge cycle blocks sharing atoms -> ring systems (union-find over atoms)
    parent = {a: a for a in cycle_atoms}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, _ in cycle_edges:
        if u in parent and v in parent:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv

    ring_groups: dict[int, list[int]] = {}
    for a in sorted(cycle_atoms):
        ring_groups.setdefault(find(a), []).append(a)

    # chain units: connected components of non-ring atoms (acyclic edges only)
    chain_atoms = [a for a in range(len(atom_types)) if a not in cycle_atoms]
    chain_set = set(chain_atoms)
    adj: dict[int, list[int]] = {a: [] for a in chain_atoms}
    for u, v, _ in edges:
        if u in chain_set and v in chain_set:
            adj[u].append(v)
            adj[v].append(u)

    chain_groups: list[list[int]] = []
    seen: set[int] = set()
    for a in chain_atoms:
        if a in seen:
            continue
        comp: list[int] = []
        stack = [a]
        seen.add(a)
        while stack:
            x = stack.pop()
            comp.append(x)
            for nb in adj[x]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        chain_groups.append(sorted(comp))

    # canonical unit ordering: ring systems first (by smallest atom index),
    # then chain units (by smallest atom index)
    ring_units = sorted([sorted(g) for g in ring_groups.values()], key=lambda g: g[0])
    chain_units_sorted = sorted(chain_groups, key=lambda g: g[0])
    ordered = [(KIND_RING, g) for g in ring_units] + [(KIND_CHAIN, g) for g in chain_units_sorted]

    atom_unit = [0] * len(atom_types)
    for uid, (_, atoms) in enumerate(ordered):
        for a in atoms:
            atom_unit[a] = uid

    units: list[Unit] = []
    for uid, (kind, atoms) in enumerate(ordered):
        atoms_set = set(atoms)
        intra, inter = [], []
        for u, v, bt in edges:
            if u in atoms_set and v in atoms_set:
                intra.append((u, v, bt))
            elif u in atoms_set or v in atoms_set:
                # bond from this unit to a different unit; stored once from the
                # smaller unit id's perspective (each real bond counted once)
                other = v if u in atoms_set else u
                if uid < atom_unit[other]:
                    inter.append((u, v, bt))
        units.append(Unit(unit_id=uid, kind=kind, atoms=atoms, intra_bonds=intra, inter_bonds=inter))

    return UnitPartition(units=units, n_atoms=len(atom_types), atom_unit=atom_unit)


# ---------------------------------------------------------------------------
# signature (cross-molecule type alignment)
# ---------------------------------------------------------------------------


def _size_bucket(n_atoms: int) -> int:
    for i, edge in enumerate(SIZE_BUCKET_EDGES):
        if n_atoms <= edge:
            return i
    return len(SIZE_BUCKET_EDGES)


def _h32(payload: str) -> str:
    """Process-stable content hash (sha256 truncated; never Python hash())."""
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def unit_signature(kind: int, atoms: list[int], atom_types: list[int], intra_bonds: list[tuple[int, int, int]]) -> str:
    """Canonical (relabel-invariant) signature of a unit-internal graph.

    v1 (cscl-correctness-v1): **hash-based 1-WL**.  Node colours are content
    hashes of the full labelled local structure — never dense re-numberings.
    Initial colour = atom category; every refinement round hashes
    ``(own colour, sorted multiset of (bond type, neighbour colour))``.
    The signature is ``kind | n_atoms | hash(sorted final colour multiset)``.

    Consequences (each covered by tests in
    ``tests/test_cscl_correctness_v1.py``):

    * relabel invariance: any permutation of atom indices yields the same
      colour multiset at every round (cscl-v0's dense first-appearance
      remap violated this — e.g. an N–C–N chain got two different
      signatures under two labelings);
    * chemistry preservation: atom categories and bond categories stay
      inside every colour string (v0 erased them: C–O / C–N / C–F chains
      and C–C / C=C / C#C all collided);
    * determinism across processes (sha256 only, no ``hash()``).

    Known theoretical limit (unchanged from v0, documented): 1-WL cannot
    distinguish *all* non-isomorphic labelled graphs (e.g. certain regular
    graphs); equal signatures therefore do not prove isomorphism.  Tests use
    attributed-graph isomorphism (VF2) as a sampled soundness check only.
    """
    local = {a: i for i, a in enumerate(sorted(atoms))}
    n = len(atoms)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, bt in intra_bonds:
        iu, iv = local[u], local[v]
        adj[iu].append((int(bt), iv))
        adj[iv].append((int(bt), iu))
    colors = [f"a{int(atom_types[a])}" for a in atoms]
    for _ in range(WL_ROUNDS):
        colors = [
            _h32(f"{colors[i]}#{';'.join(sorted(f'{bt}:{colors[j]}' for bt, j in adj[i]))}")
            for i in range(n)
        ]
    core = ";".join(sorted(colors))
    return f"{kind}|{n}|{_h32(core)}"


def unk_signature(kind: int, n_atoms: int) -> str:
    """UNK bucket key for rare/unseen units (v0_protocol §2.4)."""
    return f"UNK|{kind}|{ _size_bucket(n_atoms)}"


# ---------------------------------------------------------------------------
# vocabulary (fit-only)
# ---------------------------------------------------------------------------


@dataclass
class UnitVocab:
    """Signature -> type id mapping, built on fit data only."""

    sig_to_id: dict[str, int] = field(default_factory=dict)
    known_sigs: set[str] = field(default_factory=set)
    min_count: int = 3

    def fit(self, signatures: list[str]) -> "UnitVocab":
        counts = Counter(signatures)
        known = sorted([s for s, c in counts.items() if c >= self.min_count])
        self.known_sigs = set(known)
        self.sig_to_id = {s: i for i, s in enumerate(known)}
        # rare fit signatures collapse into their (kind, size-bucket) UNK key;
        # unseen-at-dev signatures route to the same keys (or UNK|GLOBAL).
        unk_keys = set()
        for s in counts:
            if s not in self.known_sigs:
                parts = s.split("|")
                if len(parts) == 3:
                    try:
                        unk_keys.add(unk_signature(int(parts[0]), int(parts[1])))
                        continue
                    except ValueError:
                        pass
                unk_keys.add(s)  # non-standard signature: raw string key
        unk_keys.add(GLOBAL_UNK_KEY)
        for key in sorted(unk_keys):
            self.sig_to_id[key] = len(self.sig_to_id)
        return self

    def to_id(self, signature: str, kind: int, n_atoms: int) -> int:
        """Type id; unseen signatures (dev-time) map to their UNK bucket id."""
        if signature in self.sig_to_id:
            return self.sig_to_id[signature]
        key = unk_signature(kind, n_atoms)
        if key in self.sig_to_id:
            return self.sig_to_id[key]
        return self.sig_to_id[GLOBAL_UNK_KEY]

    def is_known(self, signature: str) -> bool:
        return signature in self.known_sigs

    @property
    def n_known(self) -> int:
        return len(self.known_sigs)

    @property
    def n_total(self) -> int:
        return len(self.sig_to_id)


def build_partition(atom_types: list[int], edge_index: list[tuple[int, int]], edge_types: list[int]) -> UnitPartition:
    """Partition from PyG-style input; physical bonds canonicalized first
    (:func:`unique_undirected_bonds`), so both-direction storage can never
    double count bonds anywhere downstream."""
    edges = unique_undirected_bonds(edge_index, edge_types)
    part = partition_units(atom_types, edges)
    for unit in part.units:
        unit.signature = unit_signature(unit.kind, unit.atoms, atom_types, unit.intra_bonds)
    return part
