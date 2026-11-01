"""RDKit SMILES -> frozen raw atom/bond attributes -> ``lipo_v1`` molecular graph.

The ``lipo_v1`` FeatureSpec was defined on a *frozen raw attribute vocabulary*
(see ``lcrc_cross_task_preparation_v1/analysis/feature_spec_lipo.json``).  The
new MoleculeNet datasets are distributed as SMILES, so this module reproduces
exactly the same raw attribute convention that the Lipophilicity artifact used
(``tracks/structure-contribution/code/bjoc_lipophilicity_adapter_v1.py``):

======================  ==============================================
raw field               definition
======================  ==============================================
``atom_class``          index of ``(element, is_aromatic)`` in the frozen
                        15-entry table, else 15 (UNK)
``atom_charge``         index of the formal charge in ``(-2,-1,0,1,2)``,
                        else 5 (UNK)
``atom_h``              index of ``GetTotalNumHs()`` in ``(0,1,2,3,4)``;
                        counts above 4 map to bucket 4 (``4+``); else 5 (UNK)
``edge_class``          ``SINGLE/DOUBLE/TRIPLE/AROMATIC`` -> 0..3, else 4 (UNK)
``edge_stereo``         ``NONE/E/Z`` -> 0..2, else 3 (OTHER)
======================  ==============================================

Nothing is fitted, masked or imputed here.  Molecules that RDKit cannot parse
are reported as failures rather than dropped; the caller decides (and the P0
audit blocks the task rather than silently changing the sample set).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import mb_common as C  # noqa: I001,F401  (adds the frozen prep code dir to sys.path)

import adapters as PREP_ADAPTERS
import molecular_graph as MG
from feature_spec import FeatureSpec

#: Frozen ``lipo_v1`` (element, aromatic) vocabulary, 0..14; 15 is UNK.
ATOM_CLASSES: tuple[tuple[str, bool], ...] = (
    ("B", False),
    ("Br", False),
    ("C", False),
    ("C", True),
    ("Cl", False),
    ("F", False),
    ("I", False),
    ("N", False),
    ("N", True),
    ("O", False),
    ("O", True),
    ("P", False),
    ("S", False),
    ("S", True),
    ("Si", False),
)
ATOM_UNK = len(ATOM_CLASSES)  # 15

BOND_CLASSES: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
BOND_UNK = 4

CHARGE_BUCKETS: tuple[int, ...] = (-2, -1, 0, 1, 2)
CHARGE_UNK = 5

H_BUCKETS: tuple[int, ...] = (0, 1, 2, 3, 4)
H_UNK = 5

_ELEMENT_TO_CLASS: dict[tuple[str, bool], int] = {pair: k for k, pair in enumerate(ATOM_CLASSES)}


def _charge_code(value: int) -> int:
    if int(value) in CHARGE_BUCKETS:
        return CHARGE_BUCKETS.index(int(value))
    return CHARGE_UNK


def _h_code(value: int) -> int:
    value = int(value)
    if value in H_BUCKETS:
        return H_BUCKETS.index(value)
    if value > H_BUCKETS[-1]:
        return len(H_BUCKETS) - 1  # explicit 4+ bucket, never silently dropped
    return H_UNK


def _stereo_code(stereo: Any) -> int:
    from rdkit import Chem

    mapping = {
        Chem.BondStereo.STEREONONE: 0,
        Chem.BondStereo.STEREOE: 1,
        Chem.BondStereo.STEREOZ: 2,
    }
    return mapping.get(stereo, 3)


def smiles_to_raw_row(smiles: str, row_index: int) -> dict[str, Any]:
    """Deterministic RDKit parse -> frozen raw-attribute arrays (label-blind)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    Chem.SanitizeMol(mol)
    n = int(mol.GetNumAtoms())
    if n < 1:
        raise ValueError(f"SMILES {smiles!r} parsed to 0 atoms")
    atom_class = np.full(n, ATOM_UNK, dtype=np.int64)
    atom_charge = np.full(n, CHARGE_UNK, dtype=np.int64)
    atom_h = np.full(n, H_UNK, dtype=np.int64)
    for atom in mol.GetAtoms():
        i = int(atom.GetIdx())
        atom_class[i] = _ELEMENT_TO_CLASS.get((atom.GetSymbol(), bool(atom.GetIsAromatic())), ATOM_UNK)
        atom_charge[i] = _charge_code(int(atom.GetFormalCharge()))
        atom_h[i] = _h_code(int(atom.GetTotalNumHs()))
    edges: list[tuple[int, int]] = []
    classes: list[int] = []
    stereo: list[int] = []
    for bond in mol.GetBonds():
        i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        bt = str(bond.GetBondType())
        code = BOND_CLASSES.index(bt) if bt in BOND_CLASSES else BOND_UNK
        st = _stereo_code(bond.GetStereo())
        edges.extend([(i, j), (j, i)])
        classes.extend([code, code])
        stereo.extend([st, st])
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return {
        "smiles": str(smiles),
        "canonical_smiles": str(canonical),
        "atom_class": atom_class,
        "atom_charge": atom_charge,
        "atom_h": atom_h,
        "edge_index": edge_index,
        "edge_class": np.asarray(classes, dtype=np.int64),
        "edge_stereo": np.asarray(stereo, dtype=np.int64),
        "n_components": int(len(Chem.GetMolFrags(mol))),
        "mol_index": int(row_index),
    }


def raw_row_to_graph(row: Mapping[str, Any], row_index: int, spec: FeatureSpec) -> MG.MolecularGraph:
    """Frozen ``lipo_v1`` graph builder (upstream adapter, unchanged)."""
    return PREP_ADAPTERS.lipo_row_to_graph(row, int(row_index), spec)


def atom_attribute_audit(raw_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Vocabulary coverage of the *raw* attributes (before any model input)."""
    atom_class = np.concatenate([np.asarray(r["atom_class"], dtype=np.int64) for r in raw_rows])
    charge = np.concatenate([np.asarray(r["atom_charge"], dtype=np.int64) for r in raw_rows])
    h = np.concatenate([np.asarray(r["atom_h"], dtype=np.int64) for r in raw_rows])
    bond_class = np.concatenate(
        [np.asarray(r["edge_class"], dtype=np.int64) for r in raw_rows if np.asarray(r["edge_class"]).size]
    )
    stereo = np.concatenate(
        [np.asarray(r["edge_stereo"], dtype=np.int64) for r in raw_rows if np.asarray(r["edge_stereo"]).size]
    )
    n_components = np.asarray([int(r["n_components"]) for r in raw_rows], dtype=np.int64)
    class_counts = np.bincount(atom_class, minlength=ATOM_UNK + 1)
    bond_counts = np.bincount(bond_class, minlength=BOND_UNK + 1)
    return {
        "n_atoms_total": int(atom_class.size),
        "n_directed_bond_records": int(bond_class.size),
        "n_molecules": int(len(raw_rows)),
        "atom_class_counts": {str(i): int(c) for i, c in enumerate(class_counts.tolist())},
        "atom_unk_count": int(class_counts[ATOM_UNK]),
        "atom_unk_fraction": float(class_counts[ATOM_UNK] / max(atom_class.size, 1)),
        "charge_counts": {str(i): int(c) for i, c in enumerate(np.bincount(charge, minlength=CHARGE_UNK + 1).tolist())},
        "charge_unk_count": int((charge == CHARGE_UNK).sum()),
        "h_counts": {str(i): int(c) for i, c in enumerate(np.bincount(h, minlength=H_UNK + 1).tolist())},
        "h_unk_count": int((h == H_UNK).sum()),
        "bond_class_counts": {str(i): int(c) for i, c in enumerate(bond_counts.tolist())},
        "bond_unk_count": int(bond_counts[BOND_UNK]),
        "stereo_counts": {str(i): int(c) for i, c in enumerate(np.bincount(stereo, minlength=5).tolist())},
        "n_multi_component_molecules": int((n_components > 1).sum()),
        "max_components": int(n_components.max()) if n_components.size else 0,
    }


__all__ = [
    "ATOM_CLASSES",
    "ATOM_UNK",
    "BOND_CLASSES",
    "BOND_UNK",
    "CHARGE_BUCKETS",
    "CHARGE_UNK",
    "H_BUCKETS",
    "H_UNK",
    "smiles_to_raw_row",
    "raw_row_to_graph",
    "atom_attribute_audit",
]
