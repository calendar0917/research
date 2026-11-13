"""RDKit oracle labels for the two computational tasks (label-blind).

Two explicit public functions are used, exactly as specified by the round brief
and the official RDKit API documentation:

``WC_LOGP``
    ``rdMolDescriptors.CalcCrippenDescriptors(mol, includeHs=True, force=True)[0]``
    -- a *computational* Wildman-Crippen logP reference.  It is not
    experimental logD and not a newly measured logP.
``MOLWT``
    ``Descriptors.MolWt(mol)`` -- average molecular weight.  Because the frozen
    rewiring keeps element / isotope / formal charge / hydrogen count, the
    MolWt reference change under every legal rewire is exactly 0 (invariant
    control).

The model input is *never* re-parsed from SMILES: it is always the frozen raw
graph rebuilt from the label-free artifact.  The RDKit molecule here is an
independent oracle path only.  A possible descriptor cache on a copied molecule
is cleared (``ClearComputedProps``), re-sanitised and re-computed with
``force=True``; for a fixed set of views the new canonical SMILES is parsed from
scratch as an independent check that the rewire label is not the original
molecule's cached value.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import round_common as RC


def parse_smiles(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    Chem.SanitizeMol(mol)
    return mol


def fresh_copy(mol):
    """Copy the molecule, drop any cached/computed properties and re-sanitise."""
    from rdkit import Chem

    m = Chem.Mol(mol)
    m.ClearComputedProps()
    m.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(m)
    return m


def wc_logp(mol) -> float:
    from rdkit.Chem import rdMolDescriptors

    return float(rdMolDescriptors.CalcCrippenDescriptors(fresh_copy(mol), includeHs=True, force=True)[0])


def molwt(mol) -> float:
    from rdkit.Chem import Descriptors

    return float(Descriptors.MolWt(fresh_copy(mol)))


def canonical_isomeric(mol) -> str:
    from rdkit import Chem

    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def canonical_non_isomeric(mol) -> str:
    from rdkit import Chem

    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))


def atom_signature(mol) -> list[tuple]:
    """Per-atom (symbol, isotope, formal charge, total H, aromatic) in atom order."""
    out: list[tuple] = []
    for atom in mol.GetAtoms():
        out.append(
            (
                str(atom.GetSymbol()),
                int(atom.GetIsotope()),
                int(atom.GetFormalCharge()),
                int(atom.GetTotalNumHs()),
                bool(atom.GetIsAromatic()),
            )
        )
    return out


def bond_signature(mol) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for bond in mol.GetBonds():
        i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        bt = str(bond.GetBondType())
        code = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}.get(bt, 4)
        out[(min(i, j), max(i, j))] = code
    return out


def edited_rdkit_mol(mol, mapping: Sequence[int], remove: Sequence[tuple[int, int]], add: Sequence[tuple[int, int]]):
    """Apply the exact two-bond deletion/addition to an RDKit copy and sanitise.

    ``mapping`` maps frozen raw atom indices to RDKit atom indices (identity for
    every row of this round; verified by ``chemcheck.smiles_mapping``).
    """
    from rdkit import Chem

    rw = Chem.RWMol(Chem.Mol(mol))
    for a, b in remove:
        ia, ib = int(mapping[int(a)]), int(mapping[int(b)])
        if rw.GetBondBetweenAtoms(ia, ib) is None:
            raise ValueError(f"remove bond {a}-{b} not present in RDKit copy")
        rw.RemoveBond(ia, ib)
    for a, b in add:
        ia, ib = int(mapping[int(a)]), int(mapping[int(b)])
        if rw.GetBondBetweenAtoms(ia, ib) is not None:
            raise ValueError(f"add bond {a}-{b} already present in RDKit copy")
        rw.AddBond(ia, ib, Chem.BondType.SINGLE)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m


def compute_both(mol) -> dict[str, float]:
    """Both oracle values for one RDKit molecule (fresh, forced, cache-cleared)."""
    return {"wc_logp": wc_logp(mol), "molwt": molwt(mol)}


def independent_reparse_values(mol) -> dict[str, float]:
    """Parse the molecule's new canonical isomeric SMILES from scratch and recompute."""
    smi = canonical_isomeric(mol)
    reparsed = parse_smiles(smi)
    return {"smiles": smi, **compute_both(reparsed)}


def verify_edit_identity(
    original_mol,
    edited_mol,
    raw_view_row: Mapping[str, Any],
    mapping: Sequence[int],
    *,
    tol: float = RC.ORACLE_INVARIANT_TOL,
) -> dict[str, Any]:
    """Verify the edited RDKit molecule keeps atoms/isotopes/charge/H and the raw bond set.

    ``raw_view_row`` is the edited frozen raw row (not RDKit-derived).  Elements,
    isotopes, formal charges and H counts are compared atom-wise; the physical
    bond set with bond-type codes and the single-component status are checked.
    """
    n = int(len(mapping))
    if int(edited_mol.GetNumAtoms()) != n:
        return {"pass": False, "reason": f"atom_count:{edited_mol.GetNumAtoms()}!={n}"}
    sig_o = atom_signature(original_mol)
    sig_v = atom_signature(edited_mol)
    # edited atom order is the original RDKit atom order (bonds only change)
    atoms_preserved = sig_o == sig_v
    charge_ok = all(a[2] == b[2] for a, b in zip(sig_o, sig_v))
    isotope_ok = all(a[1] == b[1] for a, b in zip(sig_o, sig_v))
    h_ok = all(a[3] == b[3] for a, b in zip(sig_o, sig_v))
    element_ok = all(a[0] == b[0] for a, b in zip(sig_o, sig_v))
    # compare bond set against the frozen raw row (mapped to RDKit indices)
    bu, bv, bc, _bs = RC.G.physical_bonds(raw_view_row)
    raw_pairs = {}
    for u, v, c in zip(bu.tolist(), bv.tolist(), bc.tolist()):
        a, b = int(mapping[int(u)]), int(mapping[int(v)])
        raw_pairs[(min(a, b), max(a, b))] = int(c)
    rd_pairs = bond_signature(edited_mol)
    bond_pairs_ok = set(raw_pairs) == set(rd_pairs)
    bond_classes_ok = bond_pairs_ok and all(rd_pairs[k] == raw_pairs[k] for k in raw_pairs)
    from rdkit import Chem

    single_component = len(Chem.GetMolFrags(edited_mol)) == 1
    mw_delta = abs(molwt(edited_mol) - molwt(original_mol))
    checks = {
        "atoms_preserved": bool(atoms_preserved),
        "element": bool(element_ok),
        "isotope": bool(isotope_ok),
        "formal_charge": bool(charge_ok),
        "hydrogen_count": bool(h_ok),
        "bond_pairs": bool(bond_pairs_ok),
        "bond_classes": bool(bond_classes_ok),
        "single_component": bool(single_component),
        "molwt_delta": float(mw_delta),
        "molwt_invariant": bool(mw_delta <= tol),
    }
    return {"pass": bool(all(v for k, v in checks.items() if isinstance(v, bool))), "checks": checks}


__all__ = [
    "parse_smiles",
    "fresh_copy",
    "wc_logp",
    "molwt",
    "canonical_isomeric",
    "canonical_non_isomeric",
    "atom_signature",
    "bond_signature",
    "edited_rdkit_mol",
    "compute_both",
    "independent_reparse_values",
    "verify_edit_identity",
]
