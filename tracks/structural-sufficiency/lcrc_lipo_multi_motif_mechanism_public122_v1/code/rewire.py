"""Unified connected two-bond rewiring for the six families.

For one main instance ``(center, target, control)`` and one remote **donor**
bridge ``u-v`` this round builds the same pair of views for every family:

``T`` (destroy the target connection)
    delete ``center-target`` and ``u-v``; add a single bond from ``center`` to
    one donor end and from ``target`` to the other.  Exactly one of the two
    crossings keeps the whole graph connected.  The target family key must
    disappear and that family's global count must drop by exactly one.
``K0`` (keep the target connection)
    delete ``center-control`` and the **same** ``u-v``; add a single bond from
    ``center`` to one donor end and from ``control`` to the other.  The target
    key must still hold and the family count must be unchanged.

Donor rule (structure only; no distance / label / prediction ranking):

* ``u`` and ``v`` are both neutral carbon, ``atom_class`` in ``{2, 3}`` -- the
  carbonyl/sulfonyl pilot allowed only non-aromatic (class 2) carbons; this
  round uniformly also allows **aromatic** carbon (class 3) on either end.
  This is a pre-prediction design change and does not touch the old experiment;
* ``u-v`` is an original SINGLE bond, a bridge, ``edge_stereo == 0``;
* ``{u, v}`` is disjoint from the instance core atoms (center, target, control
  and the terminal oxygens of the carbonyl / sulfonyl environment);
* neither ``u`` nor ``v`` is a carbonyl carbon (no DOUBLE bond to a class-9 O).

Both views remove two physical bonds and add two physical bonds, so atoms,
per-atom attributes, per-atom degree, bond-type totals, rings and the component
count are preserved exactly; only the typed-neighbourhood (S1), the typed-2-path
(S2) statistics and the distance-dependent global statistics can change.
Every invariant is checked on the rebuilt graph, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import graphlib as G
import families as FAM


# ---------------------------------------------------------------------------
# label-free row editing
# ---------------------------------------------------------------------------


def edit_row(
    row: Mapping[str, Any],
    remove: Sequence[tuple[int, int]],
    add: Sequence[tuple[int, int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy ``row`` with ``remove`` physical bonds deleted and ``add`` inserted."""
    ei = np.asarray(row["edge_index"], dtype=np.int64)
    ec = np.asarray(row["edge_class"], dtype=np.int64).reshape(-1)
    es = np.asarray(row["edge_stereo"], dtype=np.int64).reshape(-1)
    keep = np.ones(ei.shape[1], dtype=bool)
    rem_info: list[dict[str, int]] = []
    for a, b in remove:
        a, b = int(a), int(b)
        if a == b:
            raise ValueError("cannot remove a self-loop")
        mask = ((ei[0] == a) & (ei[1] == b)) | ((ei[0] == b) & (ei[1] == a))
        n_hit = int(mask.sum())
        if n_hit != 2:
            raise ValueError(f"remove {a}-{b}: {n_hit} directed records, expected exactly 2")
        keep &= ~mask
        rem_info.append({"a": a, "b": b, "removed_directed_records": n_hit})
    removed_keys = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in remove}
    existing = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in zip(ei[0].tolist(), ei[1].tolist())
        if (min(int(u), int(v)), max(int(u), int(v))) not in removed_keys
    }
    new_cols: list[tuple[int, int, int, int]] = []
    for a, b in add:
        a, b = int(a), int(b)
        if a == b:
            raise ValueError(f"add {a}-{b}: self-loop")
        key = (min(a, b), max(a, b))
        if key in existing:
            raise ValueError(f"add {a}-{b}: duplicate physical bond")
        existing.add(key)
        new_cols.append((a, b, C.MF_SINGLE_BOND, C.MF_NO_STEREO))
        new_cols.append((b, a, C.MF_SINGLE_BOND, C.MF_NO_STEREO))
    ei_new = ei[:, keep]
    ec_new = ec[keep]
    es_new = es[keep]
    if new_cols:
        add_idx = np.asarray([[c[0] for c in new_cols], [c[1] for c in new_cols]], dtype=np.int64)
        add_cls = np.asarray([c[2] for c in new_cols], dtype=np.int64)
        add_ste = np.asarray([c[3] for c in new_cols], dtype=np.int64)
        ei_new = np.concatenate([ei_new, add_idx], axis=1)
        ec_new = np.concatenate([ec_new, add_cls])
        es_new = np.concatenate([es_new, add_ste])
    out = {
        "smiles": str(row["smiles"]),
        "canonical_smiles": str(row["canonical_smiles"]),
        "mol_index": int(row["mol_index"]),
        "atom_class": np.asarray(row["atom_class"], dtype=np.int64).reshape(-1).copy(),
        "atom_charge": np.asarray(row["atom_charge"], dtype=np.int64).reshape(-1).copy(),
        "atom_h": np.asarray(row["atom_h"], dtype=np.int64).reshape(-1).copy(),
        "edge_index": ei_new,
        "edge_class": ec_new,
        "edge_stereo": es_new,
        "n_components": 0,
    }
    bu, bv, _bc, _bs = G.physical_bonds(out)
    out["n_components"] = G.n_components(int(out["atom_class"].shape[0]), bu, bv)
    info = {
        "removed": rem_info,
        "added": [
            {"a": int(a), "b": int(b), "class": int(C.MF_SINGLE_BOND), "stereo": int(C.MF_NO_STEREO)}
            for a, b in add
        ],
        "n_directed_before": int(ei.shape[1]),
        "n_directed_after": int(ei_new.shape[1]),
        "n_components_after": int(out["n_components"]),
    }
    return out, info


# ---------------------------------------------------------------------------
# donors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Donor:
    u: int
    v: int
    bond_id: int
    class_u: int
    class_v: int
    degree_u: int
    degree_v: int
    atom_h_u: int
    atom_h_v: int
    ring_u: bool
    ring_v: bool
    distance_to_center: int

    @property
    def pair_key(self) -> str:
        return f"{min(self.u, self.v)}-{max(self.u, self.v)}"

    def as_row(self) -> dict[str, Any]:
        return {
            "donor_u": self.u,
            "donor_v": self.v,
            "donor_bond_id": self.bond_id,
            "donor_class_u": self.class_u,
            "donor_class_v": self.class_v,
            "donor_degree_u": self.degree_u,
            "donor_degree_v": self.degree_v,
            "donor_atom_h_u": self.atom_h_u,
            "donor_atom_h_v": self.atom_h_v,
            "donor_ring_u": int(self.ring_u),
            "donor_ring_v": int(self.ring_v),
            "donor_distance_to_center": self.distance_to_center,
        }


def donor_bonds(g: G.RawGraph, inst: FAM.Instance) -> list[Donor]:
    """Every structural donor bridge of one instance (no ranking, no selection)."""
    excluded = {int(inst.center), int(inst.target), int(inst.control)} | {int(t) for t in inst.terminals}
    dist = G.distances(g.n_atoms, g.adj, inst.center)
    out: list[Donor] = []
    for e, (u, v) in enumerate(zip(g.bu.tolist(), g.bv.tolist())):
        a, b = int(min(u, v)), int(max(u, v))
        if int(g.bclass[e]) != C.MF_SINGLE_BOND:
            continue
        if not bool(g.is_bridge[e]):
            continue
        if int(g.bstereo[e]) != C.MF_NO_STEREO:
            continue
        for x in (a, b):
            if int(g.atom_class[x]) not in (C.MF_C_CLASS, C.MF_AROMATIC_C_CLASS):
                break
            if int(g.atom_charge[x]) != C.MF_NEUTRAL_CHARGE:
                break
            if FAM.is_carbonyl_carbon(g, x):
                break
        else:
            if {a, b} & excluded:
                continue
            out.append(
                Donor(
                    u=a,
                    v=b,
                    bond_id=int(e),
                    class_u=int(g.atom_class[a]),
                    class_v=int(g.atom_class[b]),
                    degree_u=int(g.degree[a]),
                    degree_v=int(g.degree[b]),
                    atom_h_u=int(g.atom_h[a]),
                    atom_h_v=int(g.atom_h[b]),
                    ring_u=bool(g.ring_atom[a]),
                    ring_v=bool(g.ring_atom[b]),
                    distance_to_center=int(min(dist[a], dist[b])),
                )
            )
    out.sort(key=lambda d: (d.u, d.v))
    return out


# ---------------------------------------------------------------------------
# crossings
# ---------------------------------------------------------------------------


def _connected_with_edges(n_atoms: int, edges: set[tuple[int, int]]) -> bool:
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = 0
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adj.get(x, []):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == int(n_atoms)


@dataclass(frozen=True)
class Crossing:
    added: tuple[tuple[int, int], tuple[int, int]]
    connected: bool
    duplicates: tuple[str, ...]

    @property
    def legal(self) -> bool:
        return bool(self.connected) and not self.duplicates


def crossings(g: G.RawGraph, *, anchor: int, other: int, donor: Donor) -> list[Crossing]:
    """The two candidate cross-connections after cutting ``anchor-other`` and ``u-v``."""
    base = {
        (int(u), int(v))
        for u, v in zip(g.bu.tolist(), g.bv.tolist())
        if {int(u), int(v)} != {int(anchor), int(other)}
        and {int(u), int(v)} != {int(donor.u), int(donor.v)}
    }
    options = [
        ((int(anchor), int(donor.u)), (int(other), int(donor.v))),
        ((int(anchor), int(donor.v)), (int(other), int(donor.u))),
    ]
    out: list[Crossing] = []
    for add in options:
        edges = set(base)
        dup: list[str] = []
        for a, b in add:
            key = (min(a, b), max(a, b))
            if a == b:
                dup.append(f"self:{a}")
            elif key in edges:
                dup.append(f"duplicate:{key[0]}-{key[1]}")
            else:
                edges.add(key)
        out.append(
            Crossing(
                added=tuple((int(a), int(b)) for a, b in add),
                connected=_connected_with_edges(g.n_atoms, edges),
                duplicates=tuple(dup),
            )
        )
    return out


# ---------------------------------------------------------------------------
# view identity
# ---------------------------------------------------------------------------


def _bonds_code(bonds: Sequence[tuple[int, int]]) -> str:
    return ".".join(f"{min(int(a), int(b))}-{max(int(a), int(b))}" for a, b in bonds)


def view_key(row_id: int, kind: str, family: str, remove: Sequence[tuple[int, int]], add: Sequence[tuple[int, int]]) -> str:
    """View identity: row id + view kind + family + removed set + added set."""
    rem = _bonds_code(sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in remove))
    add_code = _bonds_code(sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in add))
    return f"{int(row_id)}:{kind}:{family}:R{rem}:A{add_code}"


def o_view_key(row_id: int) -> str:
    return f"{int(row_id)}:O"


def split_view_key(view_key: str) -> tuple[int, str, str]:
    rid, kind, family, *_rest = str(view_key).split(":")
    return int(rid), str(kind), str(family)


# ---------------------------------------------------------------------------
# structural verification
# ---------------------------------------------------------------------------


def verify_view_structure(
    g0: G.RawGraph,
    view_row: Mapping[str, Any],
    *,
    remove: Sequence[tuple[int, int]],
    add: Sequence[tuple[int, int]],
    kind: str,
    inst: FAM.Instance,
) -> dict[str, Any]:
    """All structural invariants of one rewired view against the original graph."""
    g1 = G.summarise_raw(view_row, g0.row_id)
    b0 = {(int(u), int(v)): (int(c), int(s)) for u, v, c, s in zip(g0.bu, g0.bv, g0.bclass, g0.bstereo)}
    b1 = {(int(u), int(v)): (int(c), int(s)) for u, v, c, s in zip(g1.bu, g1.bv, g1.bclass, g1.bstereo)}
    want_rem = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in remove}
    want_add = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in add}
    removed = set(b0) - set(b1)
    added = set(b1) - set(b0)
    changed = sorted(k for k in set(b0) & set(b1) if b0[k] != b1[k])

    atoms_equal = {
        key: bool(
            np.array_equal(
                np.asarray(view_row[key], dtype=np.int64), np.asarray(getattr(g0, key), dtype=np.int64)
            )
        )
        for key in ("atom_class", "atom_charge", "atom_h")
    }
    degrees_equal = bool(np.array_equal(g0.degree, g1.degree))
    ring_flags_equal = bool(np.array_equal(g0.ring_atom, g1.ring_atom))
    class_before = np.bincount(g0.bclass.astype(np.int64), minlength=5)
    class_after = np.bincount(g1.bclass.astype(np.int64), minlength=5)
    per_atom_class_equal = True
    for a in range(g0.n_atoms):
        c0 = np.bincount(np.asarray([c for _v, c, _e in g0.adj[a]], dtype=np.int64), minlength=5)
        c1 = np.bincount(np.asarray([c for _v, c, _e in g1.adj[a]], dtype=np.int64), minlength=5)
        if not np.array_equal(c0, c1):
            per_atom_class_equal = False
            break
    ring_bonds_before = {
        (int(u), int(v)) for u, v, e in zip(g0.bu, g0.bv, range(len(g0.bu))) if not bool(g0.is_bridge[e])
    }
    ring_bonds_after = {
        (int(u), int(v)) for u, v, e in zip(g1.bu, g1.bv, range(len(g1.bu))) if not bool(g1.is_bridge[e])
    }
    ring_bonds_preserved = ring_bonds_before == ring_bonds_after
    added_are_bridges = all(bool(g1.is_bridge[g1.lut[(min(a, b), max(a, b))]]) for a, b in want_add)

    keys0 = set(FAM.family_keys(g0, inst.family))
    keys1 = set(FAM.family_keys(g1, inst.family))
    key = (int(inst.center), int(inst.target))
    n_directed = int(np.asarray(view_row["edge_index"]).shape[1])
    uniq_keys = len({(int(u), int(v)) for u, v in zip(g1.bu, g1.bv)})
    new_self_loops = int(np.sum(g1.bu == g1.bv)) if g1.n_bonds else 0
    d_center_target = G.distances(g1.n_atoms, g1.adj, inst.center)

    out: dict[str, Any] = {
        "kind": kind,
        "family": inst.family,
        "removed_physical_bonds": sorted(removed),
        "added_physical_bonds": sorted(added),
        "removed_is_requested": removed == want_rem,
        "added_is_requested": added == want_add,
        "changed_bond_attributes": changed,
        "n_physical_bonds_before": len(b0),
        "n_physical_bonds_after": len(b1),
        "bond_count_preserved": len(b0) == len(b1),
        "directed_records_before": int(np.asarray(g0.bu).size * 2),
        "directed_records_after": n_directed,
        "directed_records_preserved": n_directed == int(np.asarray(g0.bu).size * 2),
        "directed_records_unique_physical": uniq_keys == len(b1),
        "no_self_loops": new_self_loops == 0,
        "atom_attributes_unchanged": atoms_equal,
        "degrees_unchanged": degrees_equal,
        "bond_class_totals_unchanged": bool(np.array_equal(class_before, class_after)),
        "per_atom_incident_class_counts_unchanged": bool(per_atom_class_equal),
        "ring_atom_flags_unchanged": ring_flags_equal,
        "ring_bonds_unchanged": bool(ring_bonds_preserved),
        "added_bonds_are_bridges": bool(added_are_bridges),
        "n_components_before": g0.n_components,
        "n_components_after": g1.n_components,
        "single_component_preserved": g1.n_components == 1 == g0.n_components,
        "raw_n_components_matches_graph": int(view_row["n_components"]) == g1.n_components,
        "family_count_before": len(keys0),
        "family_count_after": len(keys1),
        "family_count_delta": int(len(keys1) - len(keys0)),
        "target_key": list(key),
        "target_key_present_before": key in keys0,
        "target_key_present_after": key in keys1,
        "center_target_bond_after": g1.has_bond(inst.center, inst.target),
        "center_target_distance_after": int(d_center_target[inst.target]),
    }
    if kind == "T":
        out["expected"] = bool(
            out["target_key_present_before"]
            and not out["target_key_present_after"]
            and out["family_count_delta"] == -1
            and not out["center_target_bond_after"]
            and out["center_target_distance_after"] > 1
        )
    else:
        out["expected"] = bool(
            out["target_key_present_before"]
            and out["target_key_present_after"]
            and out["family_count_delta"] == 0
            and out["center_target_bond_after"]
            and out["center_target_distance_after"] == 1
        )
    out["pass"] = bool(
        out["expected"]
        and out["removed_is_requested"]
        and out["added_is_requested"]
        and not out["changed_bond_attributes"]
        and out["bond_count_preserved"]
        and out["directed_records_preserved"]
        and out["directed_records_unique_physical"]
        and out["no_self_loops"]
        and all(out["atom_attributes_unchanged"].values())
        and out["degrees_unchanged"]
        and out["bond_class_totals_unchanged"]
        and out["per_atom_incident_class_counts_unchanged"]
        and out["ring_atom_flags_unchanged"]
        and out["ring_bonds_unchanged"]
        and out["added_bonds_are_bridges"]
        and out["single_component_preserved"]
        and out["raw_n_components_matches_graph"]
    )
    return out


# ---------------------------------------------------------------------------
# pair construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairViews:
    row_id: int
    family: str
    center: int
    target: int
    control: int
    donor: Donor
    t_remove: tuple[tuple[int, int], ...]
    t_add: tuple[tuple[int, int], ...]
    k_remove: tuple[tuple[int, int], ...]
    k_add: tuple[tuple[int, int], ...]
    rejections: tuple[str, ...]
    n_legal_T: int
    n_legal_K: int
    n_cross_T: int
    n_cross_K: int

    @property
    def ok(self) -> bool:
        return not self.rejections

    @property
    def t_key(self) -> str:
        return view_key(self.row_id, "T", self.family, self.t_remove, self.t_add)

    @property
    def k_key(self) -> str:
        return view_key(self.row_id, "K0", self.family, self.k_remove, self.k_add)


def build_pair(g: G.RawGraph, inst: FAM.Instance, donor: Donor) -> PairViews:
    """Construct the T / K0 crossing for one (instance, donor); never selects by effect."""
    t_cross = crossings(g, anchor=inst.center, other=inst.target, donor=donor)
    k_cross = crossings(g, anchor=inst.center, other=inst.control, donor=donor)
    t_legal = [c for c in t_cross if c.legal]
    k_legal = [c for c in k_cross if c.legal]
    rejections: list[str] = []
    if len(t_legal) != 1:
        rejections.append(f"T_legal_crossings={len(t_legal)}")
    if len(k_legal) != 1:
        rejections.append(f"K_legal_crossings={len(k_legal)}")
    t_add = t_legal[0].added if len(t_legal) == 1 else tuple((0, 0) for _ in range(2))
    k_add = k_legal[0].added if len(k_legal) == 1 else tuple((0, 0) for _ in range(2))
    return PairViews(
        row_id=int(g.row_id),
        family=inst.family,
        center=int(inst.center),
        target=int(inst.target),
        control=int(inst.control),
        donor=donor,
        t_remove=((int(inst.center), int(inst.target)), (int(donor.u), int(donor.v))),
        t_add=tuple((int(a), int(b)) for a, b in t_add),
        k_remove=((int(inst.center), int(inst.control)), (int(donor.u), int(donor.v))),
        k_add=tuple((int(a), int(b)) for a, b in k_add),
        rejections=tuple(rejections),
        n_legal_T=len(t_legal),
        n_legal_K=len(k_legal),
        n_cross_T=len(t_cross),
        n_cross_K=len(k_cross),
    )


__all__ = [
    "edit_row",
    "Donor",
    "donor_bonds",
    "Crossing",
    "crossings",
    "view_key",
    "o_view_key",
    "split_view_key",
    "verify_view_structure",
    "PairViews",
    "build_pair",
]
