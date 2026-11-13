"""Connected two-bond rewiring: donors, T/K construction and invariant checks.

For one target instance ``(c, o, n, acyl_c)`` and one remote **donor** bridge
``u-v`` we build a pair of views:

``T`` (destroy the pattern)
    delete ``c-n`` and ``u-v``; add a single bond from ``c`` to one donor end and
    from ``n`` to the other.  Exactly one of the two crossings keeps the graph
    connected; the other one splits it.  ``T`` loses the ``o-c-n`` two-bond path
    and the restricted-amide instance count drops by exactly one.
``K`` (keep the pattern)
    delete ``c-acyl_c`` and the **same** ``u-v``; add a single bond from ``c`` to
    one donor end and from ``acyl_c`` to the other.  The ``o-c-n`` amide link
    survives; the third neighbour of ``c`` is a fresh non-aromatic carbon that
    carries the acyl role.

Donor rule (structure only, no distance/label/prediction ranking):

* ``u`` and ``v`` are both neutral ``atom_class == 2`` (non-aromatic carbon);
* ``u-v`` is an original SINGLE bond, a bridge, ``edge_stereo == 0``;
* ``{u, v}`` is disjoint from ``{c, o, n, acyl_c}``;
* neither ``u`` nor ``v`` carries a double bond to an oxygen (no carbonyl carbon).

Because both removed bonds are bridges and the four endpoints are distinct, the
two-bridge cut leaves a small tree of fragments and (apart from the degenerate
"donor sits exactly between the two target fragments" case) exactly one crossing
reconnects them.  Both views remove two physical bonds and add two physical
bonds, so atoms, per-atom degree, bond-type totals, ring membership, components
and the per-atom AF51 feature are preserved exactly; only the typed
neighbourhood (S1) / typed 2-path (S2) statistics and the distance-dependent
global statistics can change.  This is checked, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import graphlib as G


# ---------------------------------------------------------------------------
# label-free row editing
# ---------------------------------------------------------------------------


def edit_row(
    row: Mapping[str, Any],
    remove: Sequence[tuple[int, int]],
    add: Sequence[tuple[int, int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy ``row`` with ``remove`` physical bonds deleted and ``add`` inserted.

    Every removed physical bond must be present with **both** directed records;
    every added bond is a plain SINGLE bond with ``edge_stereo = 0`` recorded in
    both directions.  ``n_components`` of the copied row is recomputed from the
    edited graph.  No atom is added or deleted, no attribute is rewritten.
    """
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
    existing = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in zip(ei[0].tolist(), ei[1].tolist())
        if (min(int(u), int(v)), max(int(u), int(v))) not in {(min(int(a), int(b)), max(int(a), int(b))) for a, b in remove}
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
        "added": [{"a": int(a), "b": int(b), "class": int(C.MF_SINGLE_BOND), "stereo": int(C.MF_NO_STEREO)} for a, b in add],
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
    degree_u: int
    degree_v: int
    atom_h_u: int
    atom_h_v: int
    ring_u: bool
    ring_v: bool
    distance_to_c: int

    @property
    def pair_key(self) -> str:
        return f"{min(self.u, self.v)}-{max(self.u, self.v)}"

    def as_row(self) -> dict[str, Any]:
        return {
            "donor_u": self.u,
            "donor_v": self.v,
            "donor_bond_id": self.bond_id,
            "donor_degree_u": self.degree_u,
            "donor_degree_v": self.degree_v,
            "donor_atom_h_u": self.atom_h_u,
            "donor_atom_h_v": self.atom_h_v,
            "donor_ring_u": int(self.ring_u),
            "donor_ring_v": int(self.ring_v),
            "donor_distance_to_c": self.distance_to_c,
        }


def _is_carbonyl_carbon(g: G.RawGraph, atom: int) -> bool:
    """True if ``atom`` has a DOUBLE bond to a class-9 oxygen."""
    for v, cls, _e in g.adj[atom]:
        if int(g.atom_class[v]) == C.MF_O_CLASS and int(cls) == C.MF_DOUBLE_BOND:
            return True
    return False


def donor_bonds(g: G.RawGraph, inst: G.Instance) -> list[Donor]:
    """Every structural donor bridge of one instance (no ranking, no selection)."""
    excluded = {inst.c, inst.o, inst.n, inst.acyl_c}
    dist_c = G.distances(g.n_atoms, g.adj, inst.c)
    out: list[Donor] = []
    for e, (u, v) in enumerate(zip(g.bu.tolist(), g.bv.tolist())):
        a, b = int(min(u, v)), int(max(u, v))
        if int(g.bclass[e]) != C.MF_SINGLE_BOND:
            continue
        if not bool(g.is_bridge[e]):
            continue
        if int(g.bstereo[e]) != C.MF_NO_STEREO:
            continue
        if int(g.atom_class[a]) != C.MF_C_CLASS or int(g.atom_charge[a]) != C.MF_NEUTRAL_CHARGE:
            continue
        if int(g.atom_class[b]) != C.MF_C_CLASS or int(g.atom_charge[b]) != C.MF_NEUTRAL_CHARGE:
            continue
        if {a, b} & excluded:
            continue
        if _is_carbonyl_carbon(g, a) or _is_carbonyl_carbon(g, b):
            continue
        out.append(
            Donor(
                u=a,
                v=b,
                bond_id=int(e),
                degree_u=int(g.degree[a]),
                degree_v=int(g.degree[b]),
                atom_h_u=int(g.atom_h[a]),
                atom_h_v=int(g.atom_h[b]),
                ring_u=bool(g.ring_atom[a]),
                ring_v=bool(g.ring_atom[b]),
                distance_to_c=int(min(dist_c[a], dist_c[b])),
            )
        )
    out.sort(key=lambda d: (d.u, d.v))
    return out


# ---------------------------------------------------------------------------
# T / K crossings
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


def crossings(
    g: G.RawGraph,
    *,
    anchor: int,
    other: int,
    donor: Donor,
) -> list[Crossing]:
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
# view identity and verification
# ---------------------------------------------------------------------------


def _bonds_code(bonds: Sequence[tuple[int, int]]) -> str:
    return ".".join(f"{min(int(a), int(b))}-{max(int(a), int(b))}" for a, b in bonds)


def view_key(
    row_id: int,
    remove: Sequence[tuple[int, int]],
    add: Sequence[tuple[int, int]],
) -> str:
    """View identity: row id + removed physical bond set + added physical bond set.

    The frozen previous round used only the cut set (``row:Xa-b``); a rewire must
    also pin the bonds that were *written*, so both sets are part of the key.
    The scope is carried as a separate column of every prediction/analysis row.
    """
    rem = _bonds_code(sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in remove))
    add_code = _bonds_code(sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in add))
    return f"{int(row_id)}:R{rem}:A{add_code}"


def o_view_key(row_id: int) -> str:
    return f"{int(row_id)}:O"


def verify_view_structure(
    g0: G.RawGraph,
    view_row: Mapping[str, Any],
    *,
    remove: Sequence[tuple[int, int]],
    add: Sequence[tuple[int, int]],
    kind: str,
    inst: G.Instance,
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
    # original ring bonds stay ring bonds; the new bonds are bridges
    ring_bonds_before = {
        (int(u), int(v)) for u, v, e in zip(g0.bu, g0.bv, range(len(g0.bu))) if not bool(g0.is_bridge[e])
    }
    ring_bonds_after = {
        (int(u), int(v)) for u, v, e in zip(g1.bu, g1.bv, range(len(g1.bu))) if not bool(g1.is_bridge[e])
    }
    ring_bonds_preserved = ring_bonds_before == ring_bonds_after
    added_are_bridges = all(bool(g1.is_bridge[b1_index(g1, a, b)]) for a, b in want_add)

    insts0 = G.find_instances(g0)
    insts1 = G.find_instances(g1)
    ids0 = {(i.c, i.o, i.n, i.acyl_c) for i in insts0}
    ids1 = {(i.c, i.o, i.n, i.acyl_c) for i in insts1}
    d1_cn = G.distances(g1.n_atoms, g1.adj, inst.c)
    new_self_loops = int(np.sum(g1.bu == g1.bv)) if g1.n_bonds else 0
    n_directed = int(np.asarray(view_row["edge_index"]).shape[1])
    uniq_keys = len({(int(u), int(v)) for u, v in zip(g1.bu, g1.bv)})

    out: dict[str, Any] = {
        "kind": kind,
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
        "instance_count_before": len(insts0),
        "instance_count_after": len(insts1),
        "instances_before": sorted(ids0),
        "instances_after": sorted(ids1),
        "target_instance_present_before": (inst.c, inst.o, inst.n, inst.acyl_c) in ids0,
        "target_instance_present_after": (inst.c, inst.o, inst.n, inst.acyl_c) in ids1,
        "o_c_n_path_after": inst.n in {v for v, _c, _e in g1.adj[inst.c]},
        "c_n_distance_after": int(d1_cn[inst.n]),
    }
    if kind == "T":
        out["expected"] = bool(
            out["target_instance_present_before"]
            and not out["target_instance_present_after"]
            and out["instance_count_after"] == out["instance_count_before"] - 1
            and not out["o_c_n_path_after"]
            and out["c_n_distance_after"] > 1
        )
    else:
        out["expected"] = bool(
            out["target_instance_present_before"]
            and out["o_c_n_path_after"]
            and out["c_n_distance_after"] == 1
            and out["instance_count_after"] == out["instance_count_before"]
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


def b1_index(g: G.RawGraph, a: int, b: int) -> int:
    return int(g.lut[(min(int(a), int(b)), max(int(a), int(b)))])


# ---------------------------------------------------------------------------
# pair construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairViews:
    row_id: int
    donor: Donor
    t_remove: tuple[tuple[int, int], tuple[int, int]]
    t_add: tuple[tuple[int, int], tuple[int, int]]
    k_remove: tuple[tuple[int, int], tuple[int, int]]
    k_add: tuple[tuple[int, int], tuple[int, int]]
    rejections: tuple[str, ...]
    n_legal_T: int
    n_legal_K: int

    @property
    def ok(self) -> bool:
        return not self.rejections

    @property
    def t_key(self) -> str:
        return view_key(self.row_id, self.t_remove, self.t_add)

    @property
    def k_key(self) -> str:
        return view_key(self.row_id, self.k_remove, self.k_add)


def build_pair(g: G.RawGraph, inst: G.Instance, donor: Donor) -> PairViews:
    """Construct the T/K crossing for one (instance, donor); never selects by effect."""
    t_cross = crossings(g, anchor=inst.c, other=inst.n, donor=donor)
    k_cross = crossings(g, anchor=inst.c, other=inst.acyl_c, donor=donor)
    t_legal = [c for c in t_cross if c.legal]
    k_legal = [c for c in k_cross if c.legal]
    rejections: list[str] = []
    if len(t_legal) != 1:
        rejections.append(f"T_legal_crossings={len(t_legal)}")
    if len(k_legal) != 1:
        rejections.append(f"K_legal_crossings={len(k_legal)}")
    t_add = t_legal[0].added if len(t_legal) == 1 else ((0, 0), (0, 0))
    k_add = k_legal[0].added if len(k_legal) == 1 else ((0, 0), (0, 0))
    return PairViews(
        row_id=int(g.row_id),
        donor=donor,
        t_remove=((int(inst.c), int(inst.n)), (int(donor.u), int(donor.v))),
        t_add=tuple((int(a), int(b)) for a, b in t_add),
        k_remove=((int(inst.c), int(inst.acyl_c)), (int(donor.u), int(donor.v))),
        k_add=tuple((int(a), int(b)) for a, b in k_add),
        rejections=tuple(rejections),
        n_legal_T=len(t_legal),
        n_legal_K=len(k_legal),
    )


__all__ = [
    "edit_row",
    "Donor",
    "donor_bonds",
    "Crossing",
    "crossings",
    "view_key",
    "o_view_key",
    "verify_view_structure",
    "b1_index",
    "PairViews",
    "build_pair",
]
