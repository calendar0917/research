"""Extract chemically anchored, single-substituent changes from molecules.

The earlier MolHIV work represents one molecule as a bag of static local
patches.  This module deliberately changes the object being represented:
given two molecules with the same Bemis--Murcko core, it finds the case where
all but one attached branch agree and returns the *signed replacement*
``negative branch -> positive branch``.

There are two important restrictions here.

* A branch is represented together with its attachment atom and one step of
  core context.  Thus it contains atom and bond information rather than only
  topology.
* Only a single branch replacement is accepted.  A pair with several
  simultaneous substitutions is useful chemistry, but is not a clean first
  test of whether a reusable local change has label signal.

Nothing in this file reads a class label.  Labels are supplied by the caller
only after the branch decomposition has been fixed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class AnchoredBranch:
    """One non-core connected component and its chemical attachment context."""

    anchor_signature: str
    exact_key: str
    fingerprint: np.ndarray
    fragment_smiles: str


@dataclass(frozen=True)
class MoleculeBranches:
    """Label-free decomposition of a molecule around its Murcko core."""

    scaffold: str
    branches: tuple[AnchoredBranch, ...]


@dataclass(frozen=True)
class Replacement:
    """A clean branch replacement between two members of one core family."""

    scaffold: str
    positive: AnchoredBranch
    negative: AnchoredBranch


def _bond_text(bond: Chem.Bond) -> str:
    return str(bond.GetBondType()) + (":aromatic" if bond.GetIsAromatic() else "")


def _atom_text(atom: Chem.Atom) -> str:
    return ":".join(
        [
            atom.GetSymbol(),
            str(atom.GetFormalCharge()),
            "ar" if atom.GetIsAromatic() else "al",
            str(atom.GetTotalDegree()),
        ]
    )


def _connected_components(molecule: Chem.Mol, atoms: set[int]) -> list[set[int]]:
    remaining = set(atoms)
    output: list[set[int]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors():
                other = neighbor.GetIdx()
                if other in remaining:
                    remaining.remove(other)
                    component.add(other)
                    stack.append(other)
        output.append(component)
    return output


def _canonical_core_match(molecule: Chem.Mol, core: Chem.Mol) -> tuple[int, ...] | None:
    """Choose a deterministic mapping when a symmetric core has many matches.

    The choice is only used to find *external* components.  The externally
    exposed anchor signature below never includes the atom's original index,
    so the returned representation is not a node-ID feature.
    """
    matches = molecule.GetSubstructMatches(core, uniquify=True)
    if not matches:
        return None
    return min(tuple(int(value) for value in match) for match in matches)


def _anchor_signature(
    molecule: Chem.Mol,
    anchor: int,
    external: int,
    core_atoms: set[int],
) -> str:
    anchor_atom = molecule.GetAtomWithIdx(anchor)
    bond = molecule.GetBondBetweenAtoms(anchor, external)
    assert bond is not None
    core_neighbors = []
    for neighbor in anchor_atom.GetNeighbors():
        index = neighbor.GetIdx()
        if index in core_atoms and index != external:
            edge = molecule.GetBondBetweenAtoms(anchor, index)
            assert edge is not None
            core_neighbors.append(f"{_bond_text(edge)}>{_atom_text(neighbor)}")
    return "|".join(
        [
            f"anchor={_atom_text(anchor_atom)}",
            f"attach={_bond_text(bond)}",
            "core-neighbors=" + ",".join(sorted(core_neighbors)),
        ]
    )


@lru_cache(maxsize=4)
def _fingerprint_generator(n_bits: int, radius: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _fingerprint(
    molecule: Chem.Mol,
    n_bits: int,
    radius: int,
    *,
    from_atoms: list[int] | None = None,
) -> np.ndarray:
    values = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(
        _fingerprint_generator(n_bits, radius).GetFingerprint(molecule, fromAtoms=from_atoms), values
    )
    return values


def decompose_molecule(
    smiles: str,
    *,
    n_bits: int = 512,
    radius: int = 2,
) -> MoleculeBranches | None:
    """Return anchored non-core branches, or ``None`` if no usable core exists."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(molecule)
    if core.GetNumAtoms() == 0 or core.GetNumAtoms() >= molecule.GetNumAtoms():
        return None
    match = _canonical_core_match(molecule, core)
    if match is None:
        return None
    core_atoms = set(match)
    external_atoms = set(range(molecule.GetNumAtoms())) - core_atoms
    branches: list[AnchoredBranch] = []
    for component in _connected_components(molecule, external_atoms):
        connections = [
            (atom, neighbor.GetIdx())
            for atom in component
            for neighbor in molecule.GetAtomWithIdx(atom).GetNeighbors()
            if neighbor.GetIdx() in core_atoms
        ]
        # Fused/ring-changing components can have multiple core attachments.
        # They are intentionally excluded from this first, one-site route.
        if len(connections) != 1:
            continue
        external, anchor = connections[0]
        # Canonical branch text is used only to decide whether two branches
        # are identical.  The branch itself is a connected component, so this
        # string has complete rings and can be canonicalized safely.
        selected = sorted(component)
        fragment_smiles = Chem.MolFragmentToSmiles(
            molecule, atomsToUse=selected, canonical=True, isomericSmiles=True
        )
        anchor_signature = _anchor_signature(molecule, anchor, external, core_atoms)
        exact_key = anchor_signature + "||" + fragment_smiles
        branches.append(
            AnchoredBranch(
                anchor_signature=anchor_signature,
                exact_key=exact_key,
                # Fingerprint environments are seeded at the branch and its
                # attachment atom in the original valid molecule.  Thus a
                # radius-2 vector includes the bond and local core context
                # without having to manufacture an invalid partial-aromatic
                # fragment molecule.
                fingerprint=_fingerprint(
                    molecule, n_bits, radius, from_atoms=sorted(component | {anchor})
                ),
                fragment_smiles=fragment_smiles,
            )
        )
    if not branches:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    if not scaffold:
        return None
    return MoleculeBranches(scaffold=scaffold, branches=tuple(branches))


def scaffold_from_smiles(smiles: str) -> str | None:
    """Return a non-empty Murcko scaffold without retaining molecular data."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    return scaffold or None


def one_branch_replacement(
    positive: MoleculeBranches,
    negative: MoleculeBranches,
) -> Replacement | None:
    """Find the unique same-site replacement, if the two branch bags differ once."""
    if positive.scaffold != negative.scaffold:
        return None
    pos_by_key: dict[str, list[AnchoredBranch]] = {}
    neg_by_key: dict[str, list[AnchoredBranch]] = {}
    for branch in positive.branches:
        pos_by_key.setdefault(branch.exact_key, []).append(branch)
    for branch in negative.branches:
        neg_by_key.setdefault(branch.exact_key, []).append(branch)
    remaining_pos: list[AnchoredBranch] = []
    remaining_neg: list[AnchoredBranch] = []
    for key in set(pos_by_key) | set(neg_by_key):
        left, right = pos_by_key.get(key, []), neg_by_key.get(key, [])
        matched = min(len(left), len(right))
        remaining_pos.extend(left[matched:])
        remaining_neg.extend(right[matched:])
    if len(remaining_pos) != 1 or len(remaining_neg) != 1:
        return None
    if remaining_pos[0].anchor_signature != remaining_neg[0].anchor_signature:
        return None
    return Replacement(
        scaffold=positive.scaffold,
        positive=remaining_pos[0],
        negative=remaining_neg[0],
    )


def replacement_vector(replacement: Replacement) -> np.ndarray:
    """Signed ``negative -> positive`` chemical change vector."""
    return (
        replacement.positive.fingerprint.astype(np.float64)
        - replacement.negative.fingerprint.astype(np.float64)
    )


def branch_key_counts(value: MoleculeBranches) -> Counter[str]:
    """Small inspection helper used only by tests/audits."""
    return Counter(branch.exact_key for branch in value.branches)


def replacements_for_labelled_group(
    members: Iterable[tuple[MoleculeBranches, int]],
    *,
    seed: int,
) -> list[Replacement]:
    """Sample non-overlapping positive/negative pairs within one core group.

    Sampling is deterministic and uses each member at most once.  This avoids
    very large medicinal-chemistry series dominating the mechanism audit.
    """
    positive = [item for item in members if item[1] == 1]
    negative = [item for item in members if item[1] == 0]
    if not positive or not negative:
        return []
    rng = np.random.default_rng(seed)
    pos_order = rng.permutation(len(positive))
    neg_order = rng.permutation(len(negative))
    output: list[Replacement] = []
    for pos_index, neg_index in zip(pos_order, neg_order):
        candidate = one_branch_replacement(
            positive[int(pos_index)][0], negative[int(neg_index)][0]
        )
        if candidate is not None:
            output.append(candidate)
    return output
