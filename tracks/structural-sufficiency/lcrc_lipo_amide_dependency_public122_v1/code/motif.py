"""Structure-only definitions for the restricted amide link ``acyl_c - c(=O) - n``.

Nothing in this module reads a target, a prediction or an intervention effect.
Every decision is taken on the raw undirected physical graph of the original
molecule (the label-free ``raw_attributes_v1`` row), using the frozen ``lipo_v1``
raw vocabulary:

====================  =============================================
field                 meaning
====================  =============================================
``atom_class``        index in the 15-entry (element, aromatic) table
``atom_charge``       charge bucket (-2,-1,0,1,2) -> 0..4, 5 = UNK
``atom_h``            H-count bucket 0..4 (4 = "4+"), 5 = UNK
``edge_class``        0 SINGLE, 1 DOUBLE, 2 TRIPLE, 3 AROMATIC, 4 UNK
====================  =============================================

Target instance (protocol section 3), all five conditions required:

1. ``c``    : atom_class 2 (C, non-aromatic), charge bucket 2 (neutral),
              raw atom_h 0, heavy-atom degree 3;
2. ``o``    : atom_class 9 (O), charge 2, atom_h 0, degree 1, ``c-o`` DOUBLE;
3. ``n``    : atom_class 7 (N, non-aromatic), charge 2, every incident original
              bond SINGLE, ``degree(n) + atom_h(n) == 3``, ``c-n`` SINGLE **and
              a bridge**;
4. ``acyl_c``: atom_class in {2, 3}, ``c-acyl_c`` SINGLE **and a bridge**;
5. the three neighbours of ``c`` are exactly ``{o, n, acyl_c}``.

Because a neighbour is classified by its own atom class (9 / 7 / {2,3}) and the
three classes are disjoint, condition 5 is checked by requiring one neighbour
from each class.

``MATCHED`` additionally requires ``degree(n) == degree(acyl_c)`` and identical
original ring-atom flags.  A ring atom is an atom with at least one incident
non-bridge bond (equivalently: an atom that lies on a cycle).  ``BALANCED``
keeps instances whose two cut fragments have heavy-atom counts within 2.

The remote control ``R`` looks for a different ``q-r`` bond of the *same*
element/bond type (neutral non-aromatic C - neutral non-aromatic N, single,
bridge, ``r`` fully saturated with ``degree+atom_h == 3``) inside the same
original connected component, disjoint from ``{c, o, n}`` and with the same
``r`` degree / H count as ``n``.  Candidates are ranked by the structural cost
tuple ``(|deg(q)-deg(c)|, |atom_h(q)-atom_h(c)|, |s(q-r)-s(c-n)|)`` where ``s``
is the heavy-atom count of the smaller side of the original component after
removing that bridge; **all** minimal-cost ties are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

import common as C

UNREACHABLE = -1


# ---------------------------------------------------------------------------
# raw row -> undirected physical graph
# ---------------------------------------------------------------------------


def row_arrays(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    atom_class = np.asarray(row["atom_class"], dtype=np.int64).reshape(-1)
    atom_charge = np.asarray(row["atom_charge"], dtype=np.int64).reshape(-1)
    atom_h = np.asarray(row["atom_h"], dtype=np.int64).reshape(-1)
    if not (atom_class.shape == atom_charge.shape == atom_h.shape):
        raise ValueError("atom attribute length mismatch")
    return {"atom_class": atom_class, "atom_charge": atom_charge, "atom_h": atom_h}


def physical_bonds(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate the directed records into sorted ``u < v`` physical bonds."""
    ei = np.asarray(row["edge_index"], dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError("edge_index must be [2, 2M]")
    ec = np.asarray(row["edge_class"], dtype=np.int64).reshape(-1)
    es = np.asarray(row["edge_stereo"], dtype=np.int64).reshape(-1)
    if ei.shape[1] != ec.shape[0] or ec.shape[0] != es.shape[0]:
        raise ValueError("directed edge attribute length mismatch")
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u == v:
            raise ValueError("self-loop in raw row")
        key = (min(u, v), max(u, v))
        val = (int(ec[k]), int(es[k]))
        if key in seen and seen[key] != val:
            raise ValueError(f"inconsistent directed records for {key}")
        seen[key] = val
    keys = sorted(seen)
    bu = np.asarray([k[0] for k in keys], dtype=np.int64)
    bv = np.asarray([k[1] for k in keys], dtype=np.int64)
    bc = np.asarray([seen[k][0] for k in keys], dtype=np.int64)
    bs = np.asarray([seen[k][1] for k in keys], dtype=np.int64)
    return bu, bv, bc, bs


def adjacency(
    n_atoms: int, bu: np.ndarray, bv: np.ndarray, bclass: np.ndarray
) -> tuple[list[list[tuple[int, int, int]]], dict[tuple[int, int], int]]:
    """Per-atom ``(neighbour, bond_class, bond_id)`` and a ``(u,v) -> bond_id`` map."""
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(int(n_atoms))]
    lut: dict[tuple[int, int], int] = {}
    for e, (u, v, t) in enumerate(zip(bu.tolist(), bv.tolist(), bclass.tolist())):
        adj[int(u)].append((int(v), int(t), int(e)))
        adj[int(v)].append((int(u), int(t), int(e)))
        lut[(int(u), int(v))] = int(e)
    for row in adj:
        row.sort()
    return adj, lut


def degrees(n_atoms: int, bu: np.ndarray, bv: np.ndarray) -> np.ndarray:
    deg = np.zeros(int(n_atoms), dtype=np.int64)
    if len(bu):
        np.add.at(deg, bu, 1)
        np.add.at(deg, bv, 1)
    return deg


def component_ids(n_atoms: int, bu: np.ndarray, bv: np.ndarray) -> np.ndarray:
    n = int(n_atoms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(bu.tolist(), bv.tolist()):
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    roots = [find(i) for i in range(n)]
    unique = {r: i for i, r in enumerate(sorted(set(roots)))}
    return np.asarray([unique[r] for r in roots], dtype=np.int64)


def bridges(n_atoms: int, bu: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """Tarjan bridge flags per physical bond (undirected simple graph)."""
    n = int(n_atoms)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e, (u, v) in enumerate(zip(bu.tolist(), bv.tolist())):
        adj[int(u)].append((int(v), int(e)))
        adj[int(v)].append((int(u), int(e)))
    out = np.zeros(len(bu), dtype=bool)
    disc = np.full(n, -1, dtype=np.int64)
    low = np.zeros(n, dtype=np.int64)
    timer = 0
    for start in range(n):
        if disc[start] != -1:
            continue
        disc[start] = low[start] = timer
        timer += 1
        stack: list[tuple[int, int, Any]] = [(start, -1, iter(adj[start]))]
        while stack:
            node, parent_edge, it = stack[-1]
            advanced = False
            for nxt, e in it:
                if e == parent_edge:
                    continue
                if disc[nxt] == -1:
                    disc[nxt] = low[nxt] = timer
                    timer += 1
                    stack.append((nxt, e, iter(adj[nxt])))
                    advanced = True
                    break
                low[node] = min(low[node], disc[nxt])
            if advanced:
                continue
            stack.pop()
            if stack:
                pnode = stack[-1][0]
                low[pnode] = min(low[pnode], low[node])
                if low[node] > disc[pnode]:
                    out[parent_edge] = True
    return out


def ring_atom_flags(n_atoms: int, bu: np.ndarray, bv: np.ndarray, is_bridge: np.ndarray) -> np.ndarray:
    """An atom is a ring atom iff it has at least one incident non-bridge bond."""
    flags = np.zeros(int(n_atoms), dtype=bool)
    for e, (u, v) in enumerate(zip(bu.tolist(), bv.tolist())):
        if not bool(is_bridge[e]):
            flags[int(u)] = True
            flags[int(v)] = True
    return flags


def reachable_count(
    n_atoms: int,
    adj: Sequence[Sequence[tuple[int, int, int]]],
    start: int,
    blocked_bond: int,
) -> int:
    """Number of atoms reachable from ``start`` after deleting one bond."""
    n = int(n_atoms)
    seen = np.zeros(n, dtype=bool)
    seen[int(start)] = True
    frontier = [int(start)]
    count = 0
    while frontier:
        count += len(frontier)
        nxt: list[int] = []
        for u in frontier:
            for v, _cls, e in adj[u]:
                if e == blocked_bond:
                    continue
                if not seen[v]:
                    seen[v] = True
                    nxt.append(int(v))
        frontier = nxt
    return count


def distances(n_atoms: int, adj: Sequence[Sequence[tuple[int, int, int]]], source: int) -> np.ndarray:
    n = int(n_atoms)
    dist = np.full(n, UNREACHABLE, dtype=np.int64)
    dist[int(source)] = 0
    frontier = [int(source)]
    d = 0
    while frontier:
        d += 1
        nxt: list[int] = []
        for u in frontier:
            for v, _cls, _e in adj[u]:
                if dist[v] == UNREACHABLE:
                    dist[v] = d
                    nxt.append(int(v))
        frontier = nxt
    return dist


# ---------------------------------------------------------------------------
# graph summary (structure only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawGraph:
    row_id: int
    n_atoms: int
    atom_class: np.ndarray
    atom_charge: np.ndarray
    atom_h: np.ndarray
    bu: np.ndarray
    bv: np.ndarray
    bclass: np.ndarray
    bstereo: np.ndarray
    degree: np.ndarray
    adj: tuple[tuple[tuple[int, int, int], ...], ...]
    lut: dict[tuple[int, int], int] = field(default_factory=dict)
    is_bridge: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    ring_atom: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    component: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    component_size: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def n_bonds(self) -> int:
        return int(self.bu.shape[0])

    def bond_class(self, a: int, b: int) -> int:
        return int(self.bclass[self.lut[(min(a, b), max(a, b))]])

    def bond_id(self, a: int, b: int) -> int:
        return int(self.lut[(min(a, b), max(a, b))])


def summarise_raw(row: Mapping[str, Any], row_id: int) -> RawGraph:
    atoms = row_arrays(row)
    bu, bv, bc, bs = physical_bonds(row)
    n = int(atoms["atom_class"].shape[0])
    deg = degrees(n, bu, bv)
    adj, lut = adjacency(n, bu, bv, bc)
    is_bridge = bridges(n, bu, bv)
    ring = ring_atom_flags(n, bu, bv, is_bridge)
    comp = component_ids(n, bu, bv)
    counts = np.bincount(comp, minlength=int(comp.max()) + 1 if n else 1).astype(np.int64)
    comp_size = counts[comp] if n else np.zeros(0, dtype=np.int64)
    return RawGraph(
        row_id=int(row_id),
        n_atoms=n,
        atom_class=atoms["atom_class"],
        atom_charge=atoms["atom_charge"],
        atom_h=atoms["atom_h"],
        bu=bu,
        bv=bv,
        bclass=bc,
        bstereo=bs,
        degree=deg,
        adj=tuple(tuple(r) for r in adj),
        lut=lut,
        is_bridge=is_bridge,
        ring_atom=ring,
        component=comp,
        component_size=comp_size,
    )


# ---------------------------------------------------------------------------
# target instances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    row_id: int
    c: int
    o: int
    n: int
    acyl_c: int
    degree_n: int
    degree_acyl: int
    ring_n: bool
    ring_acyl: bool
    matched: bool
    n_side: int  # heavy atoms on the n side after cutting c-n
    acyl_side: int  # heavy atoms on the acyl_c side after cutting c-acyl_c
    component_size: int
    small_side_target: int  # min(side of c-n, rest of the original component)
    bond_cn: int
    bond_c_acyl: int
    bond_c_o: int

    def as_row(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "c": self.c,
            "o": self.o,
            "n": self.n,
            "acyl_c": self.acyl_c,
            "degree_n": self.degree_n,
            "degree_acyl": self.degree_acyl,
            "ring_n": int(self.ring_n),
            "ring_acyl": int(self.ring_acyl),
            "matched": int(self.matched),
            "n_side": self.n_side,
            "acyl_side": self.acyl_side,
            "component_size": self.component_size,
            "side_diff": abs(self.n_side - self.acyl_side),
            "balanced": int(abs(self.n_side - self.acyl_side) <= C.MATCHED_SIDE_TOL),
            "small_side_target": self.small_side_target,
            "bond_cn": self.bond_cn,
            "bond_c_acyl": self.bond_c_acyl,
            "bond_c_o": self.bond_c_o,
        }


def _all_single_incident(g: RawGraph, atom: int) -> bool:
    return all(int(cls) == C.MF_SINGLE_BOND for _v, cls, _e in g.adj[atom])


def find_instances(g: RawGraph) -> list[Instance]:
    """All restricted amide-link instances in one original molecule."""
    if g.n_atoms == 0:
        return []
    out: list[Instance] = []
    for c in range(g.n_atoms):
        if int(g.atom_class[c]) != C.MF_C_CLASS:
            continue
        if int(g.atom_charge[c]) != C.MF_NEUTRAL_CHARGE or int(g.atom_h[c]) != 0:
            continue
        if int(g.degree[c]) != 3:
            continue
        nbrs = [int(v) for v, _cls, _e in g.adj[c]]
        if len(nbrs) != 3:
            continue
        by_class = {"o": [], "n": [], "acyl": []}
        for v in nbrs:
            vc = int(g.atom_class[v])
            if vc == C.MF_O_CLASS:
                by_class["o"].append(v)
            elif vc == C.MF_N_CLASS:
                by_class["n"].append(v)
            elif vc in (C.MF_C_CLASS, C.MF_AROMATIC_C_CLASS):
                by_class["acyl"].append(v)
        if len(by_class["o"]) != 1 or len(by_class["n"]) != 1 or len(by_class["acyl"]) != 1:
            continue
        o = by_class["o"][0]
        n = by_class["n"][0]
        acyl = by_class["acyl"][0]
        # o
        if not (
            int(g.atom_charge[o]) == C.MF_NEUTRAL_CHARGE
            and int(g.atom_h[o]) == 0
            and int(g.degree[o]) == 1
            and g.bond_class(c, o) == C.MF_DOUBLE_BOND
        ):
            continue
        # n
        if int(g.atom_charge[n]) != C.MF_NEUTRAL_CHARGE:
            continue
        if not _all_single_incident(g, n):
            continue
        if int(g.degree[n]) + int(g.atom_h[n]) != 3:
            continue
        e_cn = g.bond_id(c, n)
        if g.bond_class(c, n) != C.MF_SINGLE_BOND or not bool(g.is_bridge[e_cn]):
            continue
        # acyl_c
        e_ca = g.bond_id(c, acyl)
        if g.bond_class(c, acyl) != C.MF_SINGLE_BOND or not bool(g.is_bridge[e_ca]):
            continue
        # sides
        comp = int(g.component[c])
        comp_size = int(np.sum(g.component == comp))
        n_side = reachable_count(g.n_atoms, g.adj, n, e_cn)
        acyl_side = reachable_count(g.n_atoms, g.adj, acyl, e_ca)
        small_side = min(n_side, comp_size - n_side)
        matched = bool(
            int(g.degree[n]) == int(g.degree[acyl]) and bool(g.ring_atom[n]) == bool(g.ring_atom[acyl])
        )
        out.append(
            Instance(
                row_id=g.row_id,
                c=int(c),
                o=int(o),
                n=int(n),
                acyl_c=int(acyl),
                degree_n=int(g.degree[n]),
                degree_acyl=int(g.degree[acyl]),
                ring_n=bool(g.ring_atom[n]),
                ring_acyl=bool(g.ring_atom[acyl]),
                matched=matched,
                n_side=int(n_side),
                acyl_side=int(acyl_side),
                component_size=int(comp_size),
                small_side_target=int(small_side),
                bond_cn=int(e_cn),
                bond_c_acyl=int(e_ca),
                bond_c_o=int(g.bond_id(c, o)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# remote same-element / same-bond-type control R
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteCandidate:
    q: int
    r: int
    bond: int
    cost: tuple[int, int, int]
    is_amide_like: bool
    degree_q: int
    atom_h_q: int
    side_small: int
    side_q: int

    def as_row(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "r": self.r,
            "bond": self.bond,
            "cost_degree": self.cost[0],
            "cost_h": self.cost[1],
            "cost_size": self.cost[2],
            "is_amide_like": int(self.is_amide_like),
            "degree_q": self.degree_q,
            "atom_h_q": self.atom_h_q,
            "side_small": self.side_small,
            "side_q": self.side_q,
        }


def _is_amide_like(g: RawGraph, q: int, r: int) -> bool:
    """``q`` carries a C=O and is singly bonded to the N ``r`` (loose pattern)."""
    if g.bond_class(q, r) != C.MF_SINGLE_BOND:
        return False
    for v, cls, _e in g.adj[q]:
        if int(g.atom_class[v]) == C.MF_O_CLASS and int(cls) == C.MF_DOUBLE_BOND:
            return True
    return False


def remote_candidates(g: RawGraph, inst: Instance, *, same_component: bool = True) -> dict[str, Any]:
    """All minimal-cost remote ``q-r`` bonds for one target instance.

    ``same_component=True`` is the protocol definition (search only inside the
    original connected component that contains the target instance);
    ``same_component=False`` is reported once as a coverage diagnostic.
    """
    target_component = int(g.component[inst.c])
    excluded = {inst.c, inst.o, inst.n}
    candidates: list[RemoteCandidate] = []
    for e, (u, v) in enumerate(zip(g.bu.tolist(), g.bv.tolist())):
        if not bool(g.is_bridge[e]):
            continue
        if int(g.bclass[e]) != C.MF_SINGLE_BOND:
            continue
        for q, r in ((int(u), int(v)), (int(v), int(u))):
            if int(g.atom_class[q]) != C.MF_C_CLASS or int(g.atom_charge[q]) != C.MF_NEUTRAL_CHARGE:
                continue
            if int(g.atom_class[r]) != C.MF_N_CLASS or int(g.atom_charge[r]) != C.MF_NEUTRAL_CHARGE:
                continue
            if {q, r} & excluded:
                continue
            if same_component and int(g.component[q]) != target_component:
                continue
            if not _all_single_incident(g, r):
                continue
            if int(g.degree[r]) + int(g.atom_h[r]) != 3:
                continue
            if int(g.degree[r]) != inst.degree_n or int(g.atom_h[r]) != int(g.atom_h[inst.n]):
                continue
            side_q = reachable_count(g.n_atoms, g.adj, q, e)
            side_small = min(side_q, inst.component_size - side_q)
            cost = (
                abs(int(g.degree[q]) - int(g.degree[inst.c])),
                abs(int(g.atom_h[q]) - int(g.atom_h[inst.c])),
                abs(int(side_small) - int(inst.small_side_target)),
            )
            candidates.append(
                RemoteCandidate(
                    q=q,
                    r=r,
                    bond=int(e),
                    cost=cost,
                    is_amide_like=_is_amide_like(g, q, r),
                    degree_q=int(g.degree[q]),
                    atom_h_q=int(g.atom_h[q]),
                    side_small=int(side_small),
                    side_q=int(side_q),
                )
            )
    if not candidates:
        return {"n_candidates": 0, "best_cost": None, "ties": [], "n_ties": 0}
    best = min(c.cost for c in candidates)
    ties = sorted([c for c in candidates if c.cost == best], key=lambda c: (c.q, c.r))
    return {
        "n_candidates": len(candidates),
        "best_cost": tuple(int(x) for x in best),
        "ties": ties,
        "n_ties": len(ties),
    }


# ---------------------------------------------------------------------------
# Bemis-Murcko scaffold key (local RDKit; analysis clustering only)
# ---------------------------------------------------------------------------


def bemis_murcko_key(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return str(scaffold)
    # Acyclic molecules share the empty scaffold: fall back to the molecule's own
    # non-isomeric canonical SMILES so they are not merged into a single cluster.
    return str(Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True))


def canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    return str(Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True))


__all__ = [
    "UNREACHABLE",
    "row_arrays",
    "physical_bonds",
    "adjacency",
    "degrees",
    "component_ids",
    "bridges",
    "ring_atom_flags",
    "reachable_count",
    "distances",
    "RawGraph",
    "summarise_raw",
    "Instance",
    "find_instances",
    "RemoteCandidate",
    "remote_candidates",
    "bemis_murcko_key",
    "canonical_smiles",
]
