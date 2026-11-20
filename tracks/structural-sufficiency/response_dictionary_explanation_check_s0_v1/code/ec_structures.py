"""Exact fragment structure identifiers and small RDKit helpers (display only).

The fragment cache already stores every atom/bond attribute and the exact
coloured-graph key.  This module only rebuilds a readable fragment SMILES with a
``*`` dummy at the cut port; it never changes a structure, a key or a number.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ogb.utils.features import allowable_features as OGB_FEATURES

ATOMIC_NUM = OGB_FEATURES["possible_atomic_num_list"]
FORMAL_CHARGE = OGB_FEATURES["possible_formal_charge_list"]
NUM_H = OGB_FEATURES["possible_numH_list"]
BOND_TYPE = OGB_FEATURES["possible_bond_type_list"]
HYBRID = OGB_FEATURES["possible_hybridization_list"]


def _as_scalar(value):
    return value if not isinstance(value, str) else None


def element_symbol(atomic_number: int) -> str:
    from rdkit.Chem import GetPeriodicTable

    return GetPeriodicTable().GetElementSymbol(int(atomic_number))


def fragment_atoms(payload: Mapping[str, Any], fid: int):
    a0 = int(payload["frag_atom_ptr"][fid])
    a1 = int(payload["frag_atom_ptr"][fid + 1])
    b0 = int(payload["frag_bond_ptr"][fid])
    b1 = int(payload["frag_bond_ptr"][fid + 1])
    return (np.asarray(payload["frag_atom_feat"][a0:a1]),
            np.asarray(payload["frag_bond_u"][b0:b1]),
            np.asarray(payload["frag_bond_v"][b0:b1]),
            np.asarray(payload["frag_bond_feat"][b0:b1]))


def fragment_smiles(payload: Mapping[str, Any], fid: int) -> str:
    """Readable SMILES with ``*`` at the cut; falls back to an exact graph string."""
    from rdkit import Chem
    from rdkit.Chem import Atom, BondType, RWMol

    atom_feat, bu, bv, bf = fragment_atoms(payload, fid)
    port = int(payload["frag_port_local"][fid])
    if port < 0:
        return "NO_PORT"
    mol = RWMol()
    for row in atom_feat:
        z = int(ATOMIC_NUM[int(row[0])])
        atom = Atom(z)
        charge = _as_scalar(FORMAL_CHARGE[int(row[3])])
        atom.SetFormalCharge(int(charge) if charge is not None else 0)
        h = _as_scalar(NUM_H[int(row[4])])
        atom.SetNumExplicitHs(int(h) if h is not None else 0)
        atom.SetNoImplicit(True)
        hyb = HYBRID[int(row[6])]
        atom.SetHybridization({
            "SP": Chem.HybridizationType.SP,
            "SP2": Chem.HybridizationType.SP2,
            "SP3": Chem.HybridizationType.SP3,
            "SP3D": Chem.HybridizationType.SP3D,
            "SP3D2": Chem.HybridizationType.SP3D2,
        }.get(hyb, Chem.HybridizationType.UNSPECIFIED))
        if int(row[7]):
            atom.SetIsAromatic(True)
        mol.AddAtom(atom)
    for k in range(bu.size):
        kind = BOND_TYPE[int(bf[k, 0])]
        mol.AddBond(int(bu[k]), int(bv[k]), {
            "SINGLE": BondType.SINGLE,
            "DOUBLE": BondType.DOUBLE,
            "TRIPLE": BondType.TRIPLE,
            "AROMATIC": BondType.AROMATIC,
        }.get(kind, BondType.SINGLE))
    dummy = mol.AddAtom(Atom(0))
    mol.AddBond(port, dummy, BondType.SINGLE)
    try:
        out = mol.GetMol()
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    except Exception:
        return _graph_signature(payload, fid)


def _graph_signature(payload: Mapping[str, Any], fid: int) -> str:
    atom_feat, bu, bv, bf = fragment_atoms(payload, fid)
    port = int(payload["frag_port_local"][fid])
    atoms = []
    for i, row in enumerate(atom_feat):
        z = int(ATOMIC_NUM[int(row[0])])
        mark = "*" if i == port else ""
        atoms.append(f"{i}:{element_symbol(z)}{mark}")
    bonds = [f"{int(a)}-{int(b)}:{BOND_TYPE[int(f[0])]}" for a, b, f in zip(bu, bv, bf)]
    return "[" + ",".join(atoms) + "|" + ",".join(bonds) + "]"


def port_info(payload: Mapping[str, Any], fid: int) -> dict[str, Any]:
    atom_feat, _, _, _ = fragment_atoms(payload, fid)
    port = int(payload["frag_port_local"][fid])
    if port < 0:
        return {"port_local": -1, "port_element": "none", "port_atomic_number": -1}
    z = int(ATOMIC_NUM[int(atom_feat[port][0])])
    return {"port_local": port, "port_element": element_symbol(z), "port_atomic_number": z}


def bond_label(triplet) -> str:
    kind = BOND_TYPE[int(triplet[0])]
    return f"{kind}/stereo={int(triplet[1])}/conj={int(triplet[2])}"
