"""RDKit reconstruction and verified legal Kekule enumeration.

Scope (frozen for this round)
-----------------------------
* start from the committed official PyG graph;
* fix atom correspondence, connectivity, element, formal charge and the
  hydrogen state implied by the original Kekule graph;
* identify the aromatic system on the **whole molecule** with one fixed RDKit
  version / aromaticity model;
* enumerate only aromatic-bond single/double re-assignments that are legal
  Kekule matchings under the fixed atom valences;
* verify every candidate re-sanitises to the same aromatic graph and atom
  attributes.

No tautomer, protonation, charge, non-aromatic bond, or random bond change is
allowed.  Molecules that cannot be verified are returned as ``unsupported``;
they are never silently dropped.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "tracks/structure-contribution/code"))

import cscl_smiles_graph as CSG  # noqa: E402
from rdkit import Chem  # noqa: E402

BOND_TYPES = {
    "SINGLE": Chem.BondType.SINGLE,
    "DOUBLE": Chem.BondType.DOUBLE,
    "TRIPLE": Chem.BondType.TRIPLE,
}
AROMATIC_TOKEN = 4
ENUM_LIMIT = 9  # one more than the requested maximum, so truncation is detectable

_ATOM_WORDS, _BOND_WORDS = CSG._zinc_dicts()
_PT = Chem.GetPeriodicTable()


def rdkit_version() -> str:
    return str(Chem.rdBase.rdkitVersion)


def atom_word(idx: int) -> str:
    return _ATOM_WORDS[int(idx)]


def atom_element_h_charge(idx: int) -> tuple[str, int, int]:
    return CSG.parse_atom_word(_ATOM_WORDS[int(idx)])


def bond_word(idx: int) -> str:
    return _BOND_WORDS[int(idx)]


# ---------------------------------------------------------------------------
# RDKit reconstruction from the PyG graph
# ---------------------------------------------------------------------------


def build_mol_from_graph(cats: np.ndarray, edges: Sequence[tuple[int, int]],
                         orders: np.ndarray) -> Chem.Mol:
    """Reconstruct the attributed RDKit molecule from the PyG graph.

    The committed ZINC atom dictionary stores element + explicit H + charge.
    Neutral N/O/S H counts are *not* representable for every valence state, so
    RDKit is allowed to infer implicit H here.  The inferred values are audited
    and then frozen by :func:`verify_candidate`; they are never taken from an
    external SMILES.
    """
    rw = Chem.RWMol()
    for t in np.asarray(cats).reshape(-1):
        el, h, q = CSG.parse_atom_word(_ATOM_WORDS[int(t)])
        a = Chem.Atom(el)
        a.SetFormalCharge(int(q))
        a.SetNumExplicitHs(int(h))
        a.SetNoImplicit(False)
        rw.AddAtom(a)
    for (u, v), o in zip(edges, orders):
        u, v, o = int(u), int(v), int(o)
        word = _BOND_WORDS[o]
        if word == "NONE":
            continue
        if word not in BOND_TYPES:
            raise ValueError(f"unsupported bond word {word!r}")
        rw.AddBond(u, v, BOND_TYPES[word])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def dictionary_h(cats: np.ndarray) -> dict[int, int]:
    return {i: int(CSG.parse_atom_word(_ATOM_WORDS[int(t)])[1])
            for i, t in enumerate(np.asarray(cats).reshape(-1))}


def aromatic_info(mol: Chem.Mol) -> tuple[set[int], set[tuple[int, int]]]:
    atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
    bonds: set[tuple[int, int]] = set()
    for b in mol.GetBonds():
        if b.GetIsAromatic():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bonds.add((min(u, v), max(u, v)))
    return atoms, bonds


def aromatic_capacities(mol: Chem.Mol) -> dict[int, int]:
    """Number of aromatic double bonds each aromatic atom needs in a Kekule form."""
    out: dict[int, int] = {}
    for a in mol.GetAtoms():
        if not a.GetIsAromatic():
            continue
        sigma = int(a.GetDegree())
        exo_pi = 0.0
        for b in a.GetBonds():
            if b.GetIsAromatic():
                continue
            order = float(b.GetBondTypeAsDouble())
            if order > 1.0:
                exo_pi += order - 1.0
        cap = (_PT.GetDefaultValence(a.GetAtomicNum())
               + int(a.GetFormalCharge())
               - sigma
               - int(a.GetTotalNumHs())
               - exo_pi)
        out[a.GetIdx()] = int(round(cap))
    return out


# ---------------------------------------------------------------------------
# Kekule matching enumeration
# ---------------------------------------------------------------------------


def enumerate_kekule(mol: Chem.Mol, limit: int = ENUM_LIMIT
                     ) -> tuple[list[tuple[int, int]], dict[int, int], list[tuple[int, ...]]]:
    """Enumerate legal Kekule matchings of the aromatic bonds.

    Returns ``(aromatic_edges, capacities, solutions)`` where each solution is
    an order vector (1/2) aligned with ``aromatic_edges``.  At most ``limit``
    solutions are generated; a returned list of length ``limit`` means the true
    count is at least ``limit`` (truncation).
    """
    caps = aromatic_capacities(mol)
    if any(c not in (0, 1) for c in caps.values()):
        raise ValueError(f"aromatic capacity outside {{0,1}}: {sorted(set(caps.values()))}")
    ar_edges: list[tuple[int, int]] = []
    for b in mol.GetBonds():
        if b.GetIsAromatic():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ar_edges.append((min(u, v), max(u, v)))
    ar_edges.sort()
    if not ar_edges:
        return [], caps, [tuple()]

    inc: dict[int, list[int]] = {a: [] for a in caps}
    for k, (u, v) in enumerate(ar_edges):
        inc[u].append(k)
        inc[v].append(k)

    sols: list[tuple[int, ...]] = []
    chosen: list[int | None] = [None] * len(ar_edges)
    rem = dict(caps)

    def rec() -> None:
        if len(sols) >= limit:
            return
        changed: list[int] = []
        # forced propagation: if an atom has exactly as many unassigned
        # incident bonds as its remaining capacity, all of them must be double
        while True:
            forced = None
            for a, r in rem.items():
                un = [k for k in inc[a] if chosen[k] is None]
                if r < 0 or r > len(un) or r > 1:
                    for k in changed:
                        u, v = ar_edges[k]
                        rem[u] += 1
                        rem[v] += 1
                        chosen[k] = None
                    return
                if r == len(un) and r > 0:
                    forced = (a, un)
                    break
            if forced is None:
                break
            _, un = forced
            for k in un:
                u, v = ar_edges[k]
                chosen[k] = 2
                rem[u] -= 1
                rem[v] -= 1
                changed.append(k)
        if any(r < 0 for r in rem.values()):
            for k in changed:
                u, v = ar_edges[k]
                rem[u] += 1
                rem[v] += 1
                chosen[k] = None
            return
        if all(r == 0 for r in rem.values()):
            sols.append(tuple(2 if x == 2 else 1 for x in chosen))
            for k in changed:
                u, v = ar_edges[k]
                rem[u] += 1
                rem[v] += 1
                chosen[k] = None
            return
        # branch on the most constrained atom with remaining capacity
        cand = [(len([k for k in inc[a] if chosen[k] is None]), a)
                for a, r in rem.items() if r > 0]
        _, a = min(cand)
        for k in [k for k in inc[a] if chosen[k] is None]:
            u, v = ar_edges[k]
            chosen[k] = 2
            rem[u] -= 1
            rem[v] -= 1
            rec()
            rem[u] += 1
            rem[v] += 1
            chosen[k] = None
            if len(sols) >= limit:
                break
        for k in changed:
            u, v = ar_edges[k]
            rem[u] += 1
            rem[v] += 1
            chosen[k] = None

    rec()
    return ar_edges, caps, sols


# ---------------------------------------------------------------------------
# candidate verification
# ---------------------------------------------------------------------------


def verify_candidate(orig: Chem.Mol, cats: np.ndarray,
                     edges: Sequence[tuple[int, int]],
                     orig_orders: np.ndarray,
                     candidate_orders: np.ndarray,
                     aromatic_edges: set[tuple[int, int]]) -> tuple[bool, str]:
    """Rebuild a candidate with frozen H / charge and verify it is the same molecule.

    The checks are deliberately strict: same atom attributes and total H, same
    connectivity, same aromatic atom/bond sets, same canonical aromatic SMILES,
    no radicals, and no non-aromatic bond-order change.
    """
    rw = Chem.RWMol()
    for a in orig.GetAtoms():
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(int(a.GetFormalCharge()))
        na.SetNumExplicitHs(int(a.GetTotalNumHs()))
        na.SetNoImplicit(True)
        rw.AddAtom(na)
    for (u, v), o in zip(edges, candidate_orders):
        word = _BOND_WORDS[int(o)]
        if word not in BOND_TYPES:
            return False, f"bad bond word {word}"
        rw.AddBond(int(u), int(v), BOND_TYPES[word])
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit raises a variety of valence errors
        return False, f"sanitize:{type(exc).__name__}"

    if mol.GetNumAtoms() != orig.GetNumAtoms():
        return False, "atom_count"
    for a, b in zip(orig.GetAtoms(), mol.GetAtoms()):
        if a.GetAtomicNum() != b.GetAtomicNum():
            return False, "element"
        if a.GetFormalCharge() != b.GetFormalCharge():
            return False, "charge"
        if int(a.GetTotalNumHs()) != int(b.GetTotalNumHs()):
            return False, "hydrogen"
        if a.GetDegree() != b.GetDegree():
            return False, "degree"
        if a.GetIsAromatic() != b.GetIsAromatic():
            return False, "aromatic_atom"
        if b.GetNumRadicalElectrons() != 0:
            return False, "radical"

    orig_ar_atoms, orig_ar_bonds = aromatic_info(orig)
    cand_ar_atoms, cand_ar_bonds = aromatic_info(mol)
    if orig_ar_atoms != cand_ar_atoms:
        return False, "aromatic_atom_set"
    if orig_ar_bonds != cand_ar_bonds:
        return False, "aromatic_bond_set"
    if set(map(tuple, map(sorted, aromatic_edges))) != orig_ar_bonds:
        return False, "declared_aromatic_set"
    if Chem.MolToSmiles(orig, isomericSmiles=False) != Chem.MolToSmiles(
            mol, isomericSmiles=False):
        return False, "canonical_smiles"

    index = {tuple(map(int, e)): k for k, e in enumerate(edges)}
    for e in edges:
        if e in aromatic_edges:
            continue
        if int(candidate_orders[index[e]]) != int(orig_orders[index[e]]):
            return False, "non_aromatic_change"
    return True, "ok"


# ---------------------------------------------------------------------------
# per-molecule representation generation
# ---------------------------------------------------------------------------


@dataclass
class RepResult:
    status: str                      # exhaustive / no_alternative / truncated / unsupported
    reps: list[np.ndarray] = field(default_factory=list)
    n_total: int = -1
    truncated: bool = False
    aromatic_edges: list[tuple[int, int]] = field(default_factory=list)
    n_inferred_h: int = 0
    inferred_h_details: list[dict] = field(default_factory=list)
    reason: str = ""
    h_audit: dict = field(default_factory=dict)
    total_h: list[int] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return self.status != "unsupported"


def generate_representations(cats: np.ndarray, edges: Sequence[tuple[int, int]],
                             orig_orders: np.ndarray, *,
                             max_reps: int = 8,
                             audit_explicit_h: bool = True) -> RepResult:
    """Generate up to ``max_reps`` verified legal Kekule representations."""
    cats = np.asarray(cats).reshape(-1)
    edges = [tuple(map(int, e)) for e in edges]
    orig_orders = np.asarray(orig_orders, dtype=np.int64)
    try:
        mol = build_mol_from_graph(cats, edges, orig_orders)
    except Exception as exc:
        return RepResult(status="unsupported", reason=f"reconstruct:{type(exc).__name__}:{exc}")

    ar_atoms, ar_bonds = aromatic_info(mol)
    total_h = [int(a.GetTotalNumHs()) for a in mol.GetAtoms()]
    dict_h = dictionary_h(cats)
    inferred = []
    for a in mol.GetAtoms():
        if not a.GetIsAromatic():
            continue
        dh = int(dict_h[a.GetIdx()])
        th = int(a.GetTotalNumHs())
        if th != dh:
            inferred.append({
                "atom": int(a.GetIdx()),
                "word": atom_word(int(cats[a.GetIdx()])),
                "dictionary_h": dh,
                "rdkit_total_h": th,
                "degree": int(a.GetDegree()),
                "charge": int(a.GetFormalCharge()),
            })

    try:
        ar_edges, caps, sols = enumerate_kekule(mol, limit=ENUM_LIMIT)
    except Exception as exc:
        return RepResult(status="unsupported", reason=f"enumerate:{type(exc).__name__}:{exc}",
                         n_inferred_h=len(inferred), inferred_h_details=inferred,
                         total_h=total_h)

    if not ar_edges:
        return RepResult(status="no_alternative", reps=[orig_orders.copy()],
                         n_total=1, truncated=False, aromatic_edges=[],
                         n_inferred_h=len(inferred), inferred_h_details=inferred,
                         total_h=total_h)

    edge_index = {tuple(e): k for k, e in enumerate(edges)}
    orig_ar = tuple(int(orig_orders[edge_index[e]]) for e in ar_edges)
    if any(o not in (1, 2) for o in orig_ar):
        return RepResult(status="unsupported", reason="original_aromatic_order_not_1_or_2",
                         n_inferred_h=len(inferred), inferred_h_details=inferred,
                         total_h=total_h)
    if orig_ar not in sols:
        return RepResult(status="unsupported", reason="original_kekule_not_enumerated",
                         n_inferred_h=len(inferred), inferred_h_details=inferred,
                         aromatic_edges=list(ar_edges), total_h=total_h)

    truncated = len(sols) >= ENUM_LIMIT
    n_total = ENUM_LIMIT if truncated else len(sols)
    others = sorted(set(sols) - {orig_ar})
    selected = [orig_ar] + others
    selected = selected[:max_reps]

    reps = []
    for cand_ar in selected:
        full = orig_orders.copy()
        for e, o in zip(ar_edges, cand_ar):
            full[edge_index[e]] = int(o)
        ok, why = verify_candidate(mol, cats, edges, orig_orders, full, ar_bonds)
        if not ok:
            return RepResult(status="unsupported", reason=f"candidate_verify:{why}",
                             n_inferred_h=len(inferred), inferred_h_details=inferred,
                             aromatic_edges=list(ar_edges), total_h=total_h)
        reps.append(full)

    h_audit = {}
    if audit_explicit_h and inferred:
        h_audit = explicit_h_only_audit(cats, edges, orig_orders, ar_edges, orig_ar)
    status = "truncated" if truncated else ("no_alternative" if len(reps) == 1 else "exhaustive")
    return RepResult(status=status, reps=reps, n_total=n_total, truncated=truncated,
                     aromatic_edges=list(ar_edges), n_inferred_h=len(inferred),
                     inferred_h_details=inferred, h_audit=h_audit, total_h=total_h)


def explicit_h_only_audit(cats: np.ndarray, edges: Sequence[tuple[int, int]],
                          orig_orders: np.ndarray,
                          ar_edges: Sequence[tuple[int, int]],
                          orig_ar: tuple[int, ...],
                          limit: int = 50) -> dict:
    """Show why the dictionary's explicit-H-only view is insufficient.

    Re-enumerates with capacities computed from the dictionary H count (not the
    RDKit-inferred total H) and reports (a) no legal matching, (b) the original
    Kekule missing, (c) candidates that would put a double bond on an atom whose
    H is only inferred.  This is an audit only; those candidates are never used.
    """
    mol = build_mol_from_graph(cats, edges, orig_orders)
    dict_h = dictionary_h(cats)
    caps = {}
    inferred = set()
    for a in mol.GetAtoms():
        if not a.GetIsAromatic():
            continue
        if int(a.GetTotalNumHs()) != dict_h[a.GetIdx()]:
            inferred.add(a.GetIdx())
        sigma = int(a.GetDegree())
        exo_pi = sum(float(b.GetBondTypeAsDouble()) - 1.0 for b in a.GetBonds()
                     if (not b.GetIsAromatic()) and float(b.GetBondTypeAsDouble()) > 1.0)
        caps[a.GetIdx()] = int(round(_PT.GetDefaultValence(a.GetAtomicNum())
                                     + int(a.GetFormalCharge()) - sigma
                                     - dict_h[a.GetIdx()] - exo_pi))
    if any(c not in (0, 1) for c in caps.values()):
        return {"status": "capacity_outside_0_1"}
    # reuse the matching engine by temporarily substituting capacities
    inc: dict[int, list[int]] = {a: [] for a in caps}
    for k, (u, v) in enumerate(ar_edges):
        inc[u].append(k)
        inc[v].append(k)
    sols: list[tuple[int, ...]] = []
    chosen: list[int | None] = [None] * len(ar_edges)
    rem = dict(caps)

    def rec() -> None:
        if len(sols) >= limit:
            return
        changed: list[int] = []
        while True:
            forced = None
            for a, r in rem.items():
                un = [k for k in inc[a] if chosen[k] is None]
                if r < 0 or r > len(un) or r > 1:
                    for k in changed:
                        u, v = ar_edges[k]
                        rem[u] += 1
                        rem[v] += 1
                        chosen[k] = None
                    return
                if r == len(un) and r > 0:
                    forced = (a, un)
                    break
            if forced is None:
                break
            _, un = forced
            for k in un:
                u, v = ar_edges[k]
                chosen[k] = 2
                rem[u] -= 1
                rem[v] -= 1
                changed.append(k)
        if any(r < 0 for r in rem.values()):
            for k in changed:
                u, v = ar_edges[k]
                rem[u] += 1
                rem[v] += 1
                chosen[k] = None
            return
        if all(r == 0 for r in rem.values()):
            sols.append(tuple(2 if x == 2 else 1 for x in chosen))
            for k in changed:
                u, v = ar_edges[k]
                rem[u] += 1
                rem[v] += 1
                chosen[k] = None
            return
        cand = [(len([k for k in inc[a] if chosen[k] is None]), a)
                for a, r in rem.items() if r > 0]
        _, a = min(cand)
        for k in [k for k in inc[a] if chosen[k] is None]:
            u, v = ar_edges[k]
            chosen[k] = 2
            rem[u] -= 1
            rem[v] -= 1
            rec()
            rem[u] += 1
            rem[v] += 1
            chosen[k] = None
            if len(sols) >= limit:
                break
        for k in changed:
            u, v = ar_edges[k]
            rem[u] += 1
            rem[v] += 1
            chosen[k] = None

    rec()
    h_changing = 0
    for s in sols:
        if any(s[k] == 2 for k, (u, v) in enumerate(ar_edges)
               if u in inferred or v in inferred):
            h_changing += 1
    return {
        "status": "ok",
        "inferred_h_atoms": sorted(int(x) for x in inferred),
        "explicit_h_no_solution": int(len(sols) == 0),
        "explicit_h_original_missing": int(bool(sols) and orig_ar not in sols),
        "explicit_h_h_changing_candidates": int(h_changing),
        "explicit_h_n_solutions_seen": int(len(sols)),
    }


# ---------------------------------------------------------------------------
# readable key / example helpers
# ---------------------------------------------------------------------------


def key_digest(key: tuple, max_items: int = 6) -> str:
    """Compact, deterministic text digest of a radius-2 key (for reports only)."""
    def cat(x: int) -> str:
        return f"a{x}"

    def bt(x: int) -> str:
        return {1: "S", 2: "D", 3: "T", AROMATIC_TOKEN: "A"}.get(int(x), f"b{x}")

    def sig(node) -> str:
        c, nbrs = node
        items = [f"{bt(t)}:{cat(u)}" for t, u in nbrs]
        if len(items) > max_items:
            items = items[:max_items] + ["..."]
        return f"{cat(c)}({','.join(items)})"

    c, nbrs = key
    items = [f"{bt(t)}:{sig(u)}" for t, u in nbrs]
    if len(items) > max_items:
        items = items[:max_items] + ["..."]
    return f"{cat(c)}({','.join(items)})"
