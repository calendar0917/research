"""Independent RDKit validation of the rewired molecules (no model, no labels).

The model input is *never* re-parsed from SMILES: it is always the edited
label-free raw graph pushed through the frozen adapter.  RDKit is used here only
as an independent valence / graph legality check and, later, for drawing.

Three checks per molecule:

1. **SMILES -> raw atom mapping.**  ``smiles_to_raw_row`` (the frozen vocabulary
   function that produced the artifact) must reproduce the artifact row exactly;
   if the atom order differs, a typed / bond-typed graph isomorphism between the
   artifact graph and the RDKit molecule supplies the mapping.  No mapping means
   the molecule is rejected.
2. **Rewired copy sanitises.**  The same two deleted and two added bonds are
   applied to a copy of the RDKit molecule and ``Chem.SanitizeMol`` must succeed.
3. **Rewired copy equals the rewired raw graph.**  Heavy atoms, per-atom formal
   charge bucket, per-atom H-count bucket, aromatic flags, the physical bond set
   with its bond-type codes and the component count must all match the edited raw
   row.  Anything else is a rejection reason; nothing is repaired, capped or
   hydrogenated to make a case pass.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import graphlib as G
import rewire as R

RD = C.RD


# ---------------------------------------------------------------------------
# 1. SMILES -> raw atom mapping
# ---------------------------------------------------------------------------


def _typed_graph(row: Mapping[str, Any]) -> Any:
    import networkx as nx

    bu, bv, bc, _bs = G.physical_bonds(row)
    ac = np.asarray(row["atom_class"], dtype=np.int64)
    ch = np.asarray(row["atom_charge"], dtype=np.int64)
    ah = np.asarray(row["atom_h"], dtype=np.int64)
    g = nx.Graph()
    for i in range(int(ac.shape[0])):
        g.add_node(int(i), cls=int(ac[i]), charge=int(ch[i]), h=int(ah[i]))
    for u, v, c in zip(bu.tolist(), bv.tolist(), bc.tolist()):
        g.add_edge(int(u), int(v), cls=int(c))
    return g


def rdkit_derived_row(smiles: str, row_index: int) -> dict[str, Any]:
    """Raw attributes in RDKit parse order (frozen vocabulary function)."""
    return RD.smiles_to_raw_row(str(smiles), int(row_index))


def smiles_mapping(smiles: str, raw_row: Mapping[str, Any], row_id: int) -> dict[str, Any]:
    """Prove the SMILES and the artifact row are the same typed graph.

    Returns ``raw_to_rdkit`` (list: artifact atom index -> RDKit atom index) and
    the checks that were used.  The identity mapping is preferred (the artifact
    was produced by the same vocabulary function); a typed graph isomorphism is
    the fallback and is reported as such.
    """
    from rdkit import Chem

    derived = rdkit_derived_row(smiles, row_id)
    identity = {
        "n_atoms": int(len(derived["atom_class"])) == int(len(raw_row["atom_class"])),
        "atom_class": np.array_equal(
            np.asarray(derived["atom_class"]), np.asarray(raw_row["atom_class"], dtype=np.int64)
        ),
        "atom_charge": np.array_equal(
            np.asarray(derived["atom_charge"]), np.asarray(raw_row["atom_charge"], dtype=np.int64)
        ),
        "atom_h": np.array_equal(
            np.asarray(derived["atom_h"]), np.asarray(raw_row["atom_h"], dtype=np.int64)
        ),
        "bond_pairs": {tuple(sorted(p)) for p in np.asarray(derived["edge_index"]).T.tolist()}
        == {tuple(sorted(p)) for p in np.asarray(raw_row["edge_index"]).T.tolist()},
        "bond_classes": sorted(np.asarray(derived["edge_class"]).tolist())
        == sorted(np.asarray(raw_row["edge_class"]).tolist()),
        "n_components": int(derived["n_components"]) == int(raw_row["n_components"]),
    }
    n = int(np.asarray(raw_row["atom_class"]).shape[0])
    if all(identity.values()):
        return {
            "method": "identity",
            "raw_to_rdkit": list(range(n)),
            "identity_checks": identity,
            "isomorphic": True,
        }
    # fallback: typed graph isomorphism (element+aromatic class, charge bucket, H bucket, bond class)
    import networkx as nx

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {"method": "none", "raw_to_rdkit": [], "identity_checks": identity, "isomorphic": False,
                "reason": "rdkit_parse_failed"}
    Chem.SanitizeMol(mol)
    g_raw = _typed_graph(raw_row)
    g_rd = _typed_graph(derived)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        g_raw, g_rd, node_match=lambda a, b: a == b, edge_match=lambda a, b: a == b
    )
    if not matcher.is_isomorphic():
        return {"method": "none", "raw_to_rdkit": [], "identity_checks": identity, "isomorphic": False,
                "reason": "typed_graph_not_isomorphic"}
    mapping = matcher.mapping
    return {
        "method": "typed_isomorphism",
        "raw_to_rdkit": [int(mapping[i]) for i in range(n)],
        "identity_checks": identity,
        "isomorphic": True,
    }


# ---------------------------------------------------------------------------
# 2./3. rewired RDKit copy: sanitise and compare with the rewired raw graph
# ---------------------------------------------------------------------------


def _bond_class_of(bond: Any) -> int:
    from rdkit import Chem

    bt = bond.GetBondType()
    return {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }.get(bt, 4)


def _stereo_of(bond: Any) -> int:
    from rdkit import Chem

    return {
        Chem.BondStereo.STEREONONE: 0,
        Chem.BondStereo.STEREOE: 1,
        Chem.BondStereo.STEREOZ: 2,
    }.get(bond.GetStereo(), 3)


def check_rewired_rdkit(
    smiles: str,
    row_id: int,
    raw_row: Mapping[str, Any],
    view_row: Mapping[str, Any],
    mapping: Sequence[int],
    *,
    remove: Sequence[tuple[int, int]],
    add: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Apply the same edits to an RDKit copy; sanitise; compare attributes.

    ``raw_to_rdkit`` maps artifact atom indices to RDKit atom indices; every
    rejection is a recorded reason, never silently dropped.
    """
    from rdkit import Chem

    out: dict[str, Any] = {"remove": [list(map(int, b)) for b in remove],
                           "add": [list(map(int, b)) for b in add]}
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        out.update({"pass": False, "reason": "rdkit_parse_failed"})
        return out
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # pragma: no cover - defensive
        out.update({"pass": False, "reason": f"rdkit_original_sanitize_failed:{exc}"})
        return out
    if int(mol.GetNumAtoms()) != len(mapping):
        out.update({"pass": False, "reason": f"atom_count_mismatch:{mol.GetNumAtoms()}!={len(mapping)}"})
        return out
    rw = Chem.RWMol(mol)
    try:
        for a, b in remove:
            ia, ib = int(mapping[int(a)]), int(mapping[int(b)])
            bond = rw.GetBondBetweenAtoms(ia, ib)
            if bond is None:
                out.update({"pass": False, "reason": f"rdkit_missing_bond:{a}-{b}"})
                return out
            rw.RemoveBond(ia, ib)
        for a, b in add:
            ia, ib = int(mapping[int(a)]), int(mapping[int(b)])
            if rw.GetBondBetweenAtoms(ia, ib) is not None:
                out.update({"pass": False, "reason": f"rdkit_duplicate_bond:{a}-{b}"})
                return out
            rw.AddBond(ia, ib, Chem.BondType.SINGLE)
        edited = rw.GetMol()
        Chem.SanitizeMol(edited)
    except Exception as exc:
        out.update({"pass": False, "reason": f"rdkit_sanitize_failed:{type(exc).__name__}:{exc}"})
        return out

    # compare the edited RDKit molecule with the edited raw graph
    n = int(mol.GetNumAtoms())
    cls = np.zeros(n, dtype=np.int64)
    charge = np.zeros(n, dtype=np.int64)
    hcount = np.zeros(n, dtype=np.int64)
    for atom in edited.GetAtoms():
        i = int(atom.GetIdx())
        cls[i] = RD._ELEMENT_TO_CLASS.get((atom.GetSymbol(), bool(atom.GetIsAromatic())), RD.ATOM_UNK)
        charge[i] = RD._charge_code(int(atom.GetFormalCharge()))
        hcount[i] = RD._h_code(int(atom.GetTotalNumHs()))
    rd_pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for bond in edited.GetBonds():
        i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        rd_pairs[(min(i, j), max(i, j))] = (_bond_class_of(bond), _stereo_of(bond))

    raw_cls = np.asarray(view_row["atom_class"], dtype=np.int64)
    raw_charge = np.asarray(view_row["atom_charge"], dtype=np.int64)
    raw_h = np.asarray(view_row["atom_h"], dtype=np.int64)
    bu, bv, bc, bs = G.physical_bonds(view_row)
    raw_pairs = {
        (int(u), int(v)): (int(c), int(s)) for u, v, c, s in zip(bu.tolist(), bv.tolist(), bc.tolist(), bs.tolist())
    }
    # map artifact atom indices to RDKit indices for the comparison
    idx = np.asarray(mapping, dtype=np.int64)  # artifact atom index -> RDKit atom index
    raw_cls_m = np.zeros(n, dtype=np.int64)
    raw_cls_m[idx] = raw_cls
    raw_charge_m = np.zeros(n, dtype=np.int64)
    raw_charge_m[idx] = raw_charge
    raw_h_m = np.zeros(n, dtype=np.int64)
    raw_h_m[idx] = raw_h
    raw_pairs_m = {
        (min(int(mapping[u]), int(mapping[v])), max(int(mapping[u]), int(mapping[v]))): val
        for (u, v), val in raw_pairs.items()
    }
    checks = {
        "atom_class": bool(np.array_equal(cls, raw_cls_m)),
        "formal_charge": bool(np.array_equal(charge, raw_charge_m)),
        "hydrogen_count": bool(np.array_equal(hcount, raw_h_m)),
        "bond_pairs": set(rd_pairs) == set(raw_pairs_m),
        "bond_classes": all(rd_pairs[k][0] == raw_pairs_m[k][0] for k in set(rd_pairs) & set(raw_pairs_m)),
        "bond_stereo": all(rd_pairs[k][1] == raw_pairs_m[k][1] for k in set(rd_pairs) & set(raw_pairs_m)),
        "single_component": len(Chem.GetMolFrags(edited)) == 1,
    }
    out.update(
        {
            "checks": checks,
            "n_atoms": int(n),
            "n_bonds": int(len(rd_pairs)),
            "pass": bool(all(checks.values())),
            "reason": "" if all(checks.values()) else "rdkit_attribute_mismatch:" + ",".join(
                k for k, v in checks.items() if not v
            ),
        }
    )
    return out


__all__ = [
    "rdkit_derived_row",
    "smiles_mapping",
    "check_rewired_rdkit",
]
