"""The single molecular graph used by every feature arm.

Both ``G0`` and the typed-WL arms read this module, so "the same RDKit graph
and the same atom/bond attributes" is a structural property of the code rather
than a promise: there is exactly one place where an RDKit ``Mol`` is turned
into ``(atom attributes, bond attributes, edges)``.

Atom/bond attribute values are names or small integers produced by RDKit's
categorical getters.  Chirality is the CIP code (R/S/none) rather than the raw
``CHI_TETRAHEDRAL_CW/CCW`` tag, because the raw tag is defined relative to the
atom's bond order and is therefore not stable under atom renumbering.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

import tsb_common as C


def parse_smiles(smiles: str):
    """Strict RDKit parse; a failure is a hard error, never a zero vector."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    if mol.GetNumAtoms() == 0:
        raise ValueError(f"RDKit produced an empty molecule for SMILES: {smiles!r}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    Chem.GetSymmSSSR(mol)
    return mol


def relabel(mol, permutation: Sequence[int]):
    """Return the same molecule with atom indices permuted (diagnostics only)."""
    from rdkit import Chem

    return Chem.RenumberAtoms(mol, [int(i) for i in permutation])


def _atom_attributes(atom) -> tuple[Any, ...]:
    cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "none"
    return (
        int(atom.GetAtomicNum()),
        int(atom.GetFormalCharge()),
        int(bool(atom.GetIsAromatic())),
        int(atom.GetTotalNumHs(includeNeighbors=True)),
        str(atom.GetHybridization()),
        str(cip),
    )


def _bond_attributes(bond) -> tuple[Any, ...]:
    return (
        str(bond.GetBondType()),
        int(bool(bond.GetIsAromatic())),
        int(bool(bond.GetIsConjugated())),
        str(bond.GetStereo()),
    )


def as_graph(mol) -> dict[str, Any]:
    """Canonical, index-free view of the molecular graph.

    ``atoms[i]`` is the attribute tuple of atom ``i``; ``bonds`` holds
    ``(i, j, attr_tuple)`` with ``i < j``.  Nothing else in the track inspects
    the RDKit object, which keeps G0 and typed-WL on identical inputs.
    """
    atoms = [_atom_attributes(a) for a in mol.GetAtoms()]
    bonds: list[tuple[int, int, tuple[Any, ...]]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i > j:
            i, j = j, i
        bonds.append((int(i), int(j), _bond_attributes(bond)))
    rings = [tuple(sorted(int(a) for a in ring)) for ring in mol.GetRingInfo().AtomRings()]
    bond_rings = [tuple(sorted(int(b) for b in ring)) for ring in mol.GetRingInfo().BondRings()]
    return {
        "n_atoms": len(atoms),
        "atoms": atoms,
        "bonds": bonds,
        "rings": sorted(rings),
        "bond_rings": sorted(bond_rings),
    }


def atom_token(attributes: Sequence[Any]) -> str:
    """Canonical round-0 token: the attribute vector in frozen order."""
    fields = ",".join(f"{name}={value}" for name, value in zip(C.WL_ATOM_ATTRIBUTES, attributes))
    return f"A({fields})"


def bond_token(attributes: Sequence[Any], neighbor_token: str) -> str:
    """Canonical edge descriptor used by every refinement round."""
    fields = ",".join(f"{name}={value}" for name, value in zip(C.WL_BOND_ATTRIBUTES, attributes))
    return f"E({fields})>{neighbor_token}"


def refine_tokens(tokens: Sequence[str], bonds: Iterable[tuple[int, int, tuple[Any, ...]]], round_index: int) -> list[str]:
    """One typed-WL refinement step.

    ``token_{k+1}(v) = R{k+1}(token_k(v))[sorted multiset of
    (bond attributes, token_k(u)) over neighbours u]``.  The neighbour
    descriptors are sorted as strings, so the result is independent of atom
    numbering and of bond direction.  Only string concatenation is used --
    never Python's ``hash()`` -- so the token is a stable identifier across
    processes and machines.
    """
    neighbours: list[list[str]] = [[] for _ in tokens]
    for i, j, attrs in bonds:
        neighbours[i].append(bond_token(attrs, tokens[j]))
        neighbours[j].append(bond_token(attrs, tokens[i]))
    return [
        f"R{int(round_index)}({tokens[v]})[" + "|".join(sorted(neighbours[v])) + "]"
        for v in range(len(tokens))
    ]


def molecule_tokens(smiles: str, rounds: int = 3):
    """Parsed graph plus the atom token list for every round ``0..rounds``."""
    mol = parse_smiles(smiles)
    return molecule_tokens_from_mol(mol, rounds=rounds)


def molecule_tokens_from_mol(mol, rounds: int = 3):
    """Same as :func:`molecule_tokens` for an already parsed/relabelled ``Mol``."""
    _assign_stereo(mol)
    graph = as_graph(mol)
    tokens = [atom_token(a) for a in graph["atoms"]]
    per_round = [list(tokens)]
    for k in range(1, int(rounds) + 1):
        tokens = refine_tokens(tokens, graph["bonds"], k)
        per_round.append(list(tokens))
    return mol, graph, per_round


def _assign_stereo(mol) -> None:
    """Stereochemistry/SSSR assignment required before attribute extraction."""
    from rdkit import Chem

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    Chem.GetSymmSSSR(mol)


def count_tokens(tokens: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def graph_degree_sequence(graph: dict[str, Any]) -> np.ndarray:
    degree = np.zeros(graph["n_atoms"], dtype=np.int64)
    for i, j, _attrs in graph["bonds"]:
        degree[i] += 1
        degree[j] += 1
    return degree
