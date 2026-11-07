"""Frozen 8-family motif vocabulary (RDKit ``ChemicalFeatures`` + 2 SMARTS).

A motif is exactly

    ``(family_id, tuple(sorted(unique atom indices)))``

for one molecule, where ``family_id`` is one of the eight frozen families in
``rie_common.MOTIF_FAMILIES``.  Duplicate matches of the *same*
``(family_id, atom_indices)`` are collapsed; different families may share atoms.

Nothing here uses the historical 8192-bucket hashed S1/S2 vocabulary, and no
family may be added, removed or re-tuned in this round.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Any, Iterable

import rie_common as C

#: RDKit families whose ``GetFamily()`` string maps onto the frozen ids 0-5.
_CHEMICAL_FEATURE_FAMILIES = ("Donor", "Acceptor", "Aromatic", "PosIonizable", "NegIonizable", "Hydrophobe")


def _rdkit():
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures

    return Chem, RDConfig, ChemicalFeatures


@lru_cache(maxsize=1)
def feature_factory() -> tuple[Any, dict[str, Any]]:
    """Build the RDKit chemical-feature factory once per process."""
    _Chem, RDConfig, ChemicalFeatures = _rdkit()
    fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    if not os.path.isfile(fdef_path):
        raise RuntimeError(f"RDKit BaseFeatures.fdef is missing: {fdef_path}")
    with open(fdef_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest != C.RDKIT_FDEF_SHA256:
        raise RuntimeError(
            f"BaseFeatures.fdef sha256 {digest} != frozen {C.RDKIT_FDEF_SHA256}; the motif "
            "definition source changed"
        )
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
    info = {
        "fdef_path": fdef_path,
        "fdef_sha256": digest,
        "rdkit_version": _rdkit_version(),
        "chemical_feature_families": list(_CHEMICAL_FEATURE_FAMILIES),
        "smarts": {C.FAMILY_NAMES[i]: C.MOTIF_FAMILIES[i]["smarts"] for i in (6, 7)},
    }
    return factory, info


@lru_cache(maxsize=1)
def _smarts_queries() -> tuple[Any, Any]:
    Chem, _RDConfig, _ChemicalFeatures = _rdkit()
    carbonyl = Chem.MolFromSmarts(C.CARBONYL_SMARTS)
    halogen = Chem.MolFromSmarts(C.HALOGEN_SMARTS)
    if carbonyl is None or halogen is None:
        raise RuntimeError("frozen SMARTS did not compile")
    return carbonyl, halogen


def _rdkit_version() -> str:
    from rdkit import rdBase

    return str(rdBase.rdkitVersion)


def feature_definition_audit() -> dict[str, Any]:
    """Versioned description of the frozen motif definition (label blind)."""
    _factory, info = feature_factory()
    out = {
        "families": [dict(f) for f in C.MOTIF_FAMILIES],
        "n_families": int(C.N_MOTIF_FAMILIES),
        "family_pair_columns": int(C.N_FAMILY_PAIRS),
        "rdkit_version": info["rdkit_version"],
        "base_features_fdef": info["fdef_path"],
        "base_features_fdef_sha256": info["fdef_sha256"],
        "smarts": info["smarts"],
        "dedupe_rule": "same (family_id, sorted atom indices) kept once; families may share atoms",
        "excluded_rdkit_families": ["LumpedHydrophobe", "ZnBinder"],
        "no_hashed_vocabulary": True,
    }
    return out


def parse_smiles(smiles: str) -> tuple[Any | None, str | None]:
    """Strict RDKit parse.  Returns ``(mol, None)`` or ``(None, reason)``."""
    Chem, _RDConfig, _ChemicalFeatures = _rdkit()
    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
        return None, f"{type(exc).__name__}: {exc}"
    if mol is None:
        return None, "Chem.MolFromSmiles returned None (strict sanitisation failed)"
    if mol.GetNumAtoms() < 1:
        return None, f"SMILES parsed to {mol.GetNumAtoms()} atoms"
    return mol, None


def motifs_for_mol(mol: Any) -> list[tuple[int, tuple[int, ...]]]:
    """Frozen motif set of one already-parsed RDKit molecule (canonically sorted)."""
    factory, _info = feature_factory()
    carbonyl, halogen = _smarts_queries()
    n_atoms = int(mol.GetNumAtoms())
    found: set[tuple[int, tuple[int, ...]]] = set()

    for feature in factory.GetFeaturesForMol(mol):
        family = str(feature.GetFamily())
        if family not in C.FAMILY_NAME_TO_ID:
            continue  # LumpedHydrophobe / ZnBinder are outside the frozen 8
        atoms = tuple(sorted({int(a) for a in feature.GetAtomIds()}))
        if not atoms:
            continue
        found.add((int(C.FAMILY_NAME_TO_ID[family]), atoms))

    for match in mol.GetSubstructMatches(carbonyl):
        atoms = tuple(sorted({int(a) for a in match}))
        if len(atoms) != 2:
            raise RuntimeError(f"carbonyl match {match} does not have exactly two atoms")
        found.add((int(C.FAMILY_NAME_TO_ID["Carbonyl"]), atoms))

    for match in mol.GetSubstructMatches(halogen):
        atoms = tuple(sorted({int(a) for a in match}))
        if len(atoms) != 1:
            raise RuntimeError(f"halogen match {match} does not have exactly one atom")
        found.add((int(C.FAMILY_NAME_TO_ID["Halogen"]), atoms))

    for _family, atoms in found:
        if atoms[-1] >= n_atoms:
            raise RuntimeError(f"motif atom index {atoms[-1]} outside molecule of {n_atoms} atoms")
    return sorted(found, key=lambda item: (int(item[0]), tuple(int(a) for a in item[1])))


def motifs_for_smiles(smiles: str) -> tuple[list[tuple[int, tuple[int, ...]]], str | None]:
    """Parse + motif extraction in one call (used by the structure builder)."""
    mol, reason = parse_smiles(smiles)
    if mol is None:
        return [], reason
    return motifs_for_mol(mol), None


def motif_statistics(motifs: Iterable[tuple[int, tuple[int, ...]]]) -> dict[str, Any]:
    """Small descriptive block used by coverage audits and tests."""
    motifs = list(motifs)
    per_family = [0] * C.N_MOTIF_FAMILIES
    sizes: dict[int, list[int]] = {i: [] for i in range(C.N_MOTIF_FAMILIES)}
    for family, atoms in motifs:
        per_family[int(family)] += 1
        sizes[int(family)].append(len(atoms))
    return {
        "n_motifs": len(motifs),
        "per_family_counts": per_family,
        "per_family_mean_atoms": {
            C.FAMILY_NAMES[i]: (float(sum(v) / len(v)) if v else None) for i, v in sizes.items()
        },
        "has_motifs": bool(motifs),
        "families_present": [C.FAMILY_NAMES[i] for i, c in enumerate(per_family) if c],
    }


__all__ = [
    "feature_definition_audit",
    "feature_factory",
    "motif_statistics",
    "motifs_for_mol",
    "motifs_for_smiles",
    "parse_smiles",
]
