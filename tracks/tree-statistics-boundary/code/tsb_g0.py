"""``G0``: base global composition / topology statistics.

What G0 is, exactly
-------------------
Every column is a count (or a ratio of counts) over the *label-free* graph
view in :mod:`tsb_mol`: atom species, formal charge, aromaticity, degree,
hydrogen count, hybridization, CIP chirality, bond types, ring sizes and
ring/cyclomatic topology.

What G0 is **not** (deliberately excluded, see ``PROTOCOL.md`` §3.1)
--------------------------------------------------------------
No Crippen logP / MR, no TPSA, no QED, no synthetic-accessibility score, no
BCUT/EState/Morgan-descriptor block, and none of the 217 RDKit 2D
descriptors.  The point of G0 is to ask "how far do plain composition and
topology counts get?" before any named local pattern is introduced.

Column order is frozen by :data:`COLUMN_NAMES`; ``feature_spec.json`` records
the same list, and the correctness gates assert the two agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Fixed vocabularies.  A species/type outside the list lands in ``*_other`` so
# the dimension never depends on the data.
ELEMENTS: tuple[str, ...] = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se")
BOND_TYPES: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
HYBRIDIZATIONS: tuple[str, ...] = ("S", "SP", "SP2", "SP3", "SP3D", "SP3D2")
CHIRALITIES: tuple[str, ...] = ("R", "S")
DEGREES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
H_COUNTS: tuple[int, ...] = (0, 1, 2, 3)
RING_SIZES: tuple[int, ...] = (3, 4, 5, 6, 7, 8)


def _columns() -> list[str]:
    names: list[str] = [
        # --- size / connectivity -------------------------------------------
        "n_atoms",
        "n_bonds",
        "n_components",
        "cyclomatic_number",
        "n_rings",
        "n_ring_atoms",
        "n_ring_bonds",
        "n_aromatic_atoms",
        "n_aromatic_bonds",
        "n_heteroatoms",
        "n_charged_pos",
        "n_charged_neg",
        "n_degree1_atoms",
        "n_aromatic_rings",
        "n_nonaromatic_rings",
    ]
    names += [f"el_{e}" for e in ELEMENTS] + ["el_other"]
    names += [f"deg_{d}" for d in DEGREES] + ["deg_7plus"]
    names += [f"bond_{b}" for b in BOND_TYPES] + ["bond_other"]
    names += ["charge_m2", "charge_m1", "charge_0", "charge_p1", "charge_p2", "charge_other"]
    names += [f"h_{h}" for h in H_COUNTS] + ["h_4plus"]
    names += [f"hyb_{h}" for h in HYBRIDIZATIONS] + ["hyb_other"]
    names += [f"chir_{c}" for c in CHIRALITIES] + ["chir_none", "chir_other"]
    names += [f"ring_size_{s}" for s in RING_SIZES] + ["ring_size_9plus"]
    names += [
        "ratio_ring_atoms",
        "ratio_ring_bonds",
        "ratio_aromatic_atoms",
        "ratio_aromatic_bonds",
        "ratio_heteroatoms",
        "ratio_degree1_atoms",
        "rings_per_atom",
    ]
    return names


COLUMN_NAMES: tuple[str, ...] = tuple(_columns())
DIM = len(COLUMN_NAMES)


def _components(n_atoms: int, bonds) -> int:
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _attrs in bonds:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return len({find(i) for i in range(n_atoms)})


def vector(graph: dict[str, Any]) -> np.ndarray:
    """G0 feature vector of one parsed molecule (label-free)."""
    n_atoms = int(graph["n_atoms"])
    bonds = graph["bonds"]
    n_bonds = len(bonds)
    counts: dict[str, float] = {}

    counts["n_atoms"] = n_atoms
    counts["n_bonds"] = n_bonds
    components = _components(n_atoms, bonds) if n_atoms else 0
    counts["n_components"] = components
    counts["cyclomatic_number"] = n_bonds - n_atoms + components

    ring_atoms = {a for ring in graph["rings"] for a in ring}
    ring_bonds = {b for ring in graph["bond_rings"] for b in ring}
    counts["n_rings"] = len(graph["rings"])
    counts["n_ring_atoms"] = len(ring_atoms)
    counts["n_ring_bonds"] = len(ring_bonds)

    arom_atoms = sum(1 for a in graph["atoms"] if a[2])
    arom_bonds = sum(1 for _i, _j, attrs in bonds if attrs[1])
    counts["n_aromatic_atoms"] = arom_atoms
    counts["n_aromatic_bonds"] = arom_bonds
    counts["n_heteroatoms"] = sum(1 for a in graph["atoms"] if a[0] != 6)
    counts["n_charged_pos"] = sum(1 for a in graph["atoms"] if a[1] > 0)
    counts["n_charged_neg"] = sum(1 for a in graph["atoms"] if a[1] < 0)

    degree = np.zeros(n_atoms, dtype=np.int64)
    for i, j, _attrs in bonds:
        degree[i] += 1
        degree[j] += 1
    counts["n_degree1_atoms"] = int(np.sum(degree == 1))

    aromatic_rings = 0
    for ring in graph["rings"]:
        if all(graph["atoms"][a][2] for a in ring):
            aromatic_rings += 1
    counts["n_aromatic_rings"] = aromatic_rings
    counts["n_nonaromatic_rings"] = len(graph["rings"]) - aromatic_rings

    for element in ELEMENTS:
        counts[f"el_{element}"] = 0
    counts["el_other"] = 0
    symbol_of_z = {
        6: "C", 7: "N", 8: "O", 16: "S", 9: "F", 17: "Cl", 35: "Br",
        53: "I", 15: "P", 5: "B", 14: "Si", 34: "Se",
    }
    for atom in graph["atoms"]:
        symbol = symbol_of_z.get(int(atom[0]), "other")
        key = f"el_{symbol}" if symbol != "other" else "el_other"
        counts[key] += 1

    for d in DEGREES:
        counts[f"deg_{d}"] = int(np.sum(degree == d))
    counts["deg_7plus"] = int(np.sum(degree >= 7))

    for b in BOND_TYPES:
        counts[f"bond_{b}"] = sum(1 for _i, _j, attrs in bonds if attrs[0] == b)
    counts["bond_other"] = n_bonds - sum(counts[f"bond_{b}"] for b in BOND_TYPES)

    counts["charge_m2"] = sum(1 for a in graph["atoms"] if a[1] <= -2)
    counts["charge_m1"] = sum(1 for a in graph["atoms"] if a[1] == -1)
    counts["charge_0"] = sum(1 for a in graph["atoms"] if a[1] == 0)
    counts["charge_p1"] = sum(1 for a in graph["atoms"] if a[1] == 1)
    counts["charge_p2"] = sum(1 for a in graph["atoms"] if a[1] >= 2)
    counts["charge_other"] = 0  # the five bins above partition every integer charge

    for h in H_COUNTS:
        counts[f"h_{h}"] = sum(1 for a in graph["atoms"] if a[3] == h)
    counts["h_4plus"] = sum(1 for a in graph["atoms"] if a[3] >= 4)

    for hyb in HYBRIDIZATIONS:
        counts[f"hyb_{hyb}"] = sum(1 for a in graph["atoms"] if a[4] == hyb)
    counts["hyb_other"] = n_atoms - sum(counts[f"hyb_{hyb}"] for hyb in HYBRIDIZATIONS)

    for chir in CHIRALITIES:
        counts[f"chir_{chir}"] = sum(1 for a in graph["atoms"] if a[5] == chir)
    counts["chir_none"] = sum(1 for a in graph["atoms"] if a[5] == "none")
    counts["chir_other"] = n_atoms - sum(counts[f"chir_{c}"] for c in CHIRALITIES) - counts["chir_none"]

    sizes = [len(ring) for ring in graph["rings"]]
    for s in RING_SIZES:
        counts[f"ring_size_{s}"] = sum(1 for size in sizes if size == s)
    counts["ring_size_9plus"] = sum(1 for size in sizes if size >= 9)

    denom = float(n_atoms) if n_atoms else float("nan")
    bond_denom = float(n_bonds) if n_bonds else float("nan")
    counts["ratio_ring_atoms"] = counts["n_ring_atoms"] / denom
    counts["ratio_ring_bonds"] = counts["n_ring_bonds"] / bond_denom
    counts["ratio_aromatic_atoms"] = arom_atoms / denom
    counts["ratio_aromatic_bonds"] = arom_bonds / bond_denom
    counts["ratio_heteroatoms"] = counts["n_heteroatoms"] / denom
    counts["ratio_degree1_atoms"] = counts["n_degree1_atoms"] / denom
    counts["rings_per_atom"] = counts["n_rings"] / denom

    missing = [name for name in COLUMN_NAMES if name not in counts]
    if missing:
        raise AssertionError(f"G0 columns not populated: {missing[:5]}")
    extra = [name for name in counts if name not in COLUMN_NAMES]
    if extra:
        raise AssertionError(f"G0 produced undeclared columns: {extra[:5]}")
    return np.asarray([counts[name] for name in COLUMN_NAMES], dtype=np.float32)


def matrix(graphs: list[dict[str, Any]]) -> np.ndarray:
    out = np.empty((len(graphs), DIM), dtype=np.float32)
    for i, graph in enumerate(graphs):
        out[i] = vector(graph)
    return out


def spec() -> dict[str, Any]:
    """Machine-readable description of G0 for the frozen feature spec."""
    return {
        "arm_block": "G0",
        "dim": DIM,
        "column_names": list(COLUMN_NAMES),
        "construction": "counts over the label-free tsb_mol graph view; ratios are count/atoms or count/bonds",
        "excluded_on_purpose": [
            "Crippen MolLogP / MolMR",
            "TPSA",
            "synthetic accessibility score",
            "QED",
            "all 217 RDKit 2D descriptors (including BCUT / EState / VSA / fr_* / ring-count helpers)",
            "Morgan / ECFP bits",
        ],
        "label_used": False,
    }
