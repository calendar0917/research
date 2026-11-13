"""The six pre-registered connection families (structure only, frozen).

Three connection environments x (N / O) hetero atom:

===============  ==========================  =======================
family           target environment          key
===============  ==========================  =======================
amide            C(=O)-N                     (center, h)
ester            C(=O)-O-C                   (center, h)
sulfonamide      S(=O)2-N                    (center, h)
sulfonate        S(=O)2-O-C                  (center, h)
arylamine        Ar-N-C_alkyl                (h, aromatic a) bond
aryl_ether       Ar-O-C_alkyl                (h, aromatic a) bond
===============  ==========================  =======================

Two different catalogues are kept strictly apart (protocol section 2):

``family keys`` (``family_keys``)
    the *global* count used for the "the count drops by exactly one" / "the
    count is unchanged" checks.  For the carbonyl / sulfonyl families a key is
    ``(center, h)`` with the full structural constraints of the instance; for
    the aromatic families a key is each qualifying aromatic ``C-N`` / ``C-O``
    bond ``(h, a)`` -- no control ``k`` and no "exactly one aromatic neighbour"
    is required.
``main instances`` (``main_instances``)
    the *rewirable* subset: the same constraints **plus** the requirement that
    the target bond ``center-target`` is a bridge with a usable carbon control
    ``k`` (``h-k`` a bridge, ``k`` non-aromatic carbon), i.e. the instances this
    round can actually rewire.  For the aromatic families this additionally
    requires ``h`` to have exactly one aromatic carbon neighbour.

Nothing here reads a label, a prediction or a fitted quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import common as C
import graphlib as G


# ---------------------------------------------------------------------------
# atom-level predicates
# ---------------------------------------------------------------------------


def is_carbonyl_carbon(g: G.RawGraph, atom: int) -> bool:
    """True iff ``atom`` carries a DOUBLE bond to a class-9 (O) atom."""
    for v, cls, _e in g.adj[atom]:
        if int(g.atom_class[v]) == C.MF_O_CLASS and int(cls) == C.MF_DOUBLE_BOND:
            return True
    return False


def all_incident_single(g: G.RawGraph, atom: int) -> bool:
    return all(int(cls) == C.MF_SINGLE_BOND for _v, cls, _e in g.adj[atom])


def _terminal_oxygen(g: G.RawGraph, atom: int) -> bool:
    return bool(
        int(g.atom_class[atom]) == C.MF_O_CLASS
        and int(g.atom_charge[atom]) == C.MF_NEUTRAL_CHARGE
        and int(g.atom_h[atom]) == 0
        and int(g.degree[atom]) == 1
    )


def amide_like_nitrogen(g: G.RawGraph, atom: int) -> bool:
    """Neutral N, every incident bond single, deg + rawH == 3."""
    return bool(
        int(g.atom_class[atom]) == C.MF_N_CLASS
        and int(g.atom_charge[atom]) == C.MF_NEUTRAL_CHARGE
        and all_incident_single(g, atom)
        and int(g.degree[atom]) + int(g.atom_h[atom]) == 3
    )


def ester_like_oxygen(g: G.RawGraph, atom: int, center: int) -> bool:
    """Neutral O, deg 2, H 0, all single, other neighbour a non-carbonyl C."""
    if not (
        int(g.atom_class[atom]) == C.MF_O_CLASS
        and int(g.atom_charge[atom]) == C.MF_NEUTRAL_CHARGE
        and int(g.atom_h[atom]) == 0
        and int(g.degree[atom]) == 2
        and all_incident_single(g, atom)
    ):
        return False
    others = [int(v) for v, _c, _e in g.adj[atom] if int(v) != int(center)]
    if len(others) != 1:
        return False
    other = others[0]
    if int(g.atom_class[other]) not in (C.MF_C_CLASS, C.MF_AROMATIC_C_CLASS):
        return False
    if is_carbonyl_carbon(g, other):
        return False
    return True


def aromatic_hetero_ok(g: G.RawGraph, atom: int) -> bool:
    """Neutral non-aromatic N (deg+H 3) or O (deg 2, H 0), all-carbon single nbrs."""
    cls = int(g.atom_class[atom])
    if int(g.atom_charge[atom]) != C.MF_NEUTRAL_CHARGE:
        return False
    if not all_incident_single(g, atom):
        return False
    if cls == C.MF_N_CLASS:
        if int(g.degree[atom]) + int(g.atom_h[atom]) != 3:
            return False
    elif cls == C.MF_O_CLASS:
        if int(g.degree[atom]) != 2 or int(g.atom_h[atom]) != 0:
            return False
    else:
        return False
    for v, _c, _e in g.adj[atom]:
        if int(g.atom_class[v]) not in (C.MF_C_CLASS, C.MF_AROMATIC_C_CLASS):
            return False
        if is_carbonyl_carbon(g, v):
            return False
    return True


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One rewirable target instance of one family in one molecule."""

    row_id: int
    family: str
    center: int
    target: int
    control: int
    terminals: tuple[int, ...]
    bond_center_target: int
    bond_center_control: int
    degree_target: int
    degree_control: int
    ring_target: bool
    ring_control: bool
    n_aromatic_nbrs: int
    component_size: int

    @property
    def instance_id(self) -> str:
        return f"{self.family}:{self.row_id}:{self.center}:{self.target}"

    @property
    def key(self) -> tuple[int, ...]:
        return family_key(self.family, self.center, self.target)

    def as_row(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "family": self.family,
            "row_id": self.row_id,
            "center": self.center,
            "target": self.target,
            "control": self.control,
            "terminals": "-".join(str(t) for t in self.terminals),
            "bond_center_target": self.bond_center_target,
            "bond_center_control": self.bond_center_control,
            "degree_target": self.degree_target,
            "degree_control": self.degree_control,
            "ring_target": int(self.ring_target),
            "ring_control": int(self.ring_control),
            "n_aromatic_nbrs": self.n_aromatic_nbrs,
            "component_size": self.component_size,
        }


def family_key(family: str, center: int, target: int) -> tuple[int, ...]:
    return (int(center), int(target))


# ---------------------------------------------------------------------------
# carbonyl / sulfonyl families
# ---------------------------------------------------------------------------


def _instance_from_center(
    g: G.RawGraph,
    family: str,
    center: int,
    terminals: Sequence[int],
    target: int,
    control: int,
    *,
    n_aromatic_nbrs: int = 0,
) -> Instance:
    return Instance(
        row_id=int(g.row_id),
        family=str(family),
        center=int(center),
        target=int(target),
        control=int(control),
        terminals=tuple(int(t) for t in terminals),
        bond_center_target=int(g.bond_id(center, target)),
        bond_center_control=int(g.bond_id(center, control)),
        degree_target=int(g.degree[target]),
        degree_control=int(g.degree[control]),
        ring_target=bool(g.ring_atom[target]),
        ring_control=bool(g.ring_atom[control]),
        n_aromatic_nbrs=int(n_aromatic_nbrs),
        component_size=int(g.component_size[center]),
    )


def carbonyl_instances(g: G.RawGraph, family: str) -> list[Instance]:
    """Restricted amide (N) / ester (O) instances ``center - h`` with carbon control."""
    if family not in C.CARBONYL_FAMILIES:
        raise KeyError(family)
    out: list[Instance] = []
    for center in range(g.n_atoms):
        if int(g.atom_class[center]) != C.MF_C_CLASS:
            continue
        if int(g.atom_charge[center]) != C.MF_NEUTRAL_CHARGE or int(g.atom_h[center]) != 0:
            continue
        if int(g.degree[center]) != 3:
            continue
        nbrs = [int(v) for v, _c, _e in g.adj[center]]
        oxy = [v for v in nbrs if _terminal_oxygen(g, v) and g.bond_class(center, v) == C.MF_DOUBLE_BOND]
        if len(oxy) != 1:
            continue
        o = oxy[0]
        rest = [v for v in nbrs if v != o]
        if len(rest) != 2:
            continue
        heteros: list[int] = []
        carbons: list[int] = []
        for v in rest:
            if int(g.atom_class[v]) == C.MF_N_CLASS and family == "amide":
                if amide_like_nitrogen(g, v):
                    heteros.append(v)
            elif int(g.atom_class[v]) == C.MF_O_CLASS and family == "ester":
                if ester_like_oxygen(g, v, center):
                    heteros.append(v)
            elif int(g.atom_class[v]) in (C.MF_C_CLASS, C.MF_AROMATIC_C_CLASS):
                carbons.append(v)
        if len(heteros) != 1 or len(carbons) != 1:
            continue
        h, k = heteros[0], carbons[0]
        e_h = g.bond_id(center, h)
        e_k = g.bond_id(center, k)
        if g.bond_class(center, h) != C.MF_SINGLE_BOND or not bool(g.is_bridge[e_h]):
            continue
        if g.bond_class(center, k) != C.MF_SINGLE_BOND or not bool(g.is_bridge[e_k]):
            continue
        out.append(_instance_from_center(g, family, center, (o,), h, k))
    return out


def sulfonyl_instances(g: G.RawGraph, family: str) -> list[Instance]:
    """Restricted sulfonamide (N) / sulfonate (O) instances ``s - h`` with carbon control."""
    if family not in C.SULFONYL_FAMILIES:
        raise KeyError(family)
    out: list[Instance] = []
    for center in range(g.n_atoms):
        if int(g.atom_class[center]) != C.MF_S_CLASS:
            continue
        if int(g.atom_charge[center]) != C.MF_NEUTRAL_CHARGE or int(g.atom_h[center]) != 0:
            continue
        if int(g.degree[center]) != 4:
            continue
        nbrs = [int(v) for v, _c, _e in g.adj[center]]
        oxy = [v for v in nbrs if _terminal_oxygen(g, v) and g.bond_class(center, v) == C.MF_DOUBLE_BOND]
        if len(oxy) != 2:
            continue
        rest = [v for v in nbrs if v not in oxy]
        if len(rest) != 2:
            continue
        heteros: list[int] = []
        carbons: list[int] = []
        for v in rest:
            if int(g.atom_class[v]) == C.MF_N_CLASS and family == "sulfonamide":
                if amide_like_nitrogen(g, v):
                    heteros.append(v)
            elif int(g.atom_class[v]) == C.MF_O_CLASS and family == "sulfonate":
                if ester_like_oxygen(g, v, center):
                    heteros.append(v)
            elif int(g.atom_class[v]) in (C.MF_C_CLASS, C.MF_AROMATIC_C_CLASS):
                carbons.append(v)
        if len(heteros) != 1 or len(carbons) != 1:
            continue
        h, k = heteros[0], carbons[0]
        e_h = g.bond_id(center, h)
        e_k = g.bond_id(center, k)
        if g.bond_class(center, h) != C.MF_SINGLE_BOND or not bool(g.is_bridge[e_h]):
            continue
        if g.bond_class(center, k) != C.MF_SINGLE_BOND or not bool(g.is_bridge[e_k]):
            continue
        out.append(_instance_from_center(g, family, center, tuple(oxy), h, k))
    return out


# ---------------------------------------------------------------------------
# aromatic families
# ---------------------------------------------------------------------------


def aromatic_keys(g: G.RawGraph, family: str) -> list[tuple[int, int]]:
    """Every qualifying aromatic ``C-N`` / ``C-O`` bond as a global family key."""
    if family not in C.AROMATIC_FAMILIES:
        raise KeyError(family)
    want_class = C.MF_N_CLASS if family == "arylamine" else C.MF_O_CLASS
    out: list[tuple[int, int]] = []
    for h in range(g.n_atoms):
        if int(g.atom_class[h]) != want_class:
            continue
        if not aromatic_hetero_ok(g, h):
            continue
        for a, _cls, _e in g.adj[h]:
            if int(g.atom_class[a]) == C.MF_AROMATIC_C_CLASS:
                out.append((int(h), int(a)))
    return sorted(out)


def aromatic_instances(g: G.RawGraph, family: str) -> list[Instance]:
    """Rewirable aromatic instances: exactly one aromatic neighbour (a bridge) + carbon control(s)."""
    if family not in C.AROMATIC_FAMILIES:
        raise KeyError(family)
    want_class = C.MF_N_CLASS if family == "arylamine" else C.MF_O_CLASS
    out: list[Instance] = []
    for h in range(g.n_atoms):
        if int(g.atom_class[h]) != want_class:
            continue
        if not aromatic_hetero_ok(g, h):
            continue
        aromatic = [int(a) for a, _c, _e in g.adj[h] if int(g.atom_class[a]) == C.MF_AROMATIC_C_CLASS]
        if len(aromatic) != 1:
            continue
        a = aromatic[0]
        e_ha = g.bond_id(h, a)
        if g.bond_class(h, a) != C.MF_SINGLE_BOND or not bool(g.is_bridge[e_ha]):
            continue
        controls = [
            int(k)
            for k, cls, _e in g.adj[h]
            if int(g.atom_class[k]) == C.MF_C_CLASS
            and int(cls) == C.MF_SINGLE_BOND
            and bool(g.is_bridge[g.bond_id(h, k)])
        ]
        for k in sorted(controls):
            out.append(
                _instance_from_center(
                    g, family, h, (), a, k, n_aromatic_nbrs=len(aromatic)
                )
            )
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def main_instances(g: G.RawGraph, family: str) -> list[Instance]:
    """The rewirable instance directory of one family in one molecule."""
    if family in C.CARBONYL_FAMILIES:
        return carbonyl_instances(g, family)
    if family in C.SULFONYL_FAMILIES:
        return sulfonyl_instances(g, family)
    return aromatic_instances(g, family)


def family_keys(g: G.RawGraph, family: str) -> list[tuple[int, int]]:
    """The global family-count keys of one molecule (never mixed with instances)."""
    if family in C.AROMATIC_FAMILIES:
        return aromatic_keys(g, family)
    return sorted({(int(i.center), int(i.target)) for i in main_instances(g, family)})


def count_family(g: G.RawGraph, family: str) -> int:
    return len(family_keys(g, family))


def target_key_holds(g: G.RawGraph, family: str, center: int, target: int) -> bool:
    return (int(center), int(target)) in set(family_keys(g, family))


__all__ = [
    "is_carbonyl_carbon",
    "all_incident_single",
    "amide_like_nitrogen",
    "ester_like_oxygen",
    "aromatic_hetero_ok",
    "Instance",
    "family_key",
    "carbonyl_instances",
    "sulfonyl_instances",
    "aromatic_keys",
    "aromatic_instances",
    "main_instances",
    "family_keys",
    "count_family",
    "target_key_holds",
]
