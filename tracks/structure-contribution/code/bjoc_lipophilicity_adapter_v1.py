"""BJOC Lipophilicity input adapter (cscl-bjoc-cross-task-lipophilicity-v1).

Protocol: ``configs/bjoc_cross_task_lipophilicity_v1.yaml``.

This module adapts a **new** molecular-property dataset (MoleculeNet
Lipophilicity, RDKit SMILES) onto the *unchanged* BJOC one-shot object model
(Atom -> Bond -> Junction -> static incidence -> Null composer -> whole-graph
reader).  It does **not** modify any frozen ZINC module; it imports the
vocabulary-independent graph algorithms from
``structural_relation_objects_v1`` and re-declares only the raw chemistry
vocabulary:

raw atom fields (all hidden together by the SSL mask)
    ``atom_class``  = (element, aromatic) class from the frozen table below
    ``atom_charge`` = formal-charge bucket
    ``atom_h``      = total H count bucket
    ``atom_chirality`` = RDKit chiral tag bucket

raw bond fields (hidden together by the SSL mask)
    ``edge_class``  = single / double / triple / aromatic
    ``edge_stereo`` = none / E / Z / other

Derived object features (rebuilt deterministically from the *masked* raw
graph, never patched incrementally): atom/bond/junction/global blocks described
by :func:`atom_blocks`, ``BOND_FEAT_DIM``, ``JUNC_FEAT_DIM`` and
``GLOBAL_FEAT_NAMES``.

Frozen vocabulary provenance (Stage A, structure only, DeepChem-2.8.0 scaffold
split TRAIN partition of the 4200 Lipophilicity molecules): 15 observed
(element, aromatic) pairs; the single valid-only ``Se`` aromatic atom maps to
``ATOM_UNK`` and no test atom is outside the table.  ``ATOM_MASK`` / ``BOND_MASK``
are dedicated slots that never collide with a real class or with the UNK slot
(correctness gate 3).

Discipline
----------
* label-blind: nothing in this module reads or accepts a property label.
* deterministic: no RNG except the frozen mask sampler; no learned component.
* one-shot: no message passing anywhere; the model itself lives in
  :mod:`structural_relation_composer_v1`.
* multi-component molecules (salts / counter-ions) are kept whole as
  disconnected graphs; the cyclomatic rank is ``E - V + C`` and the global
  vector carries ``n_components`` / ``frac_disconnected_pairs``.
"""

from __future__ import annotations

import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch.nn as nn

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_units as cu  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-cross-task-lipophilicity-v1"

# ---------------------------------------------------------------------------
# frozen vocabulary (Stage A audit -> frozen in the pre-registration)
# ---------------------------------------------------------------------------

#: (element, aromatic) classes observed on the DeepChem-2.8.0 scaffold-split
#: TRAIN partition (3360 molecules), sorted lexicographically.  Structure-only.
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
ATOM_UNK = len(ATOM_CLASSES)          # 15
ATOM_MASK = len(ATOM_CLASSES) + 1     # 16
N_ATOM_SLOTS = len(ATOM_CLASSES) + 2  # 17 input one-hot width
N_ATOM_TARGETS = len(ATOM_CLASSES)    # 15 pretext classes (never UNK/MASK)
_ELEMENT_TO_CLASS: dict[tuple[str, bool], int] = {pair: k for k, pair in enumerate(ATOM_CLASSES)}

#: raw bond classes; UNK/MASK are dedicated slots.
BOND_CLASSES: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
BOND_UNK = 4
BOND_MASK = 5
N_BOND_SLOTS = 6
N_BOND_TARGETS = 4

#: formal-charge buckets (-2, -1, 0, +1, >=+2); UNK/MASK dedicated slots.
CHARGE_BUCKETS: tuple[int, ...] = (-2, -1, 0, 1, 2)
CHARGE_UNK = 5
CHARGE_MASK = 6
N_CHARGE_SLOTS = 7

#: total-H buckets (0..4+); UNK/MASK dedicated slots.
H_BUCKETS: tuple[int, ...] = (0, 1, 2, 3, 4)
H_UNK = 5
H_MASK = 6
N_H_SLOTS = 7

#: RDKit chiral-tag buckets; UNK/MASK dedicated slots.
CHIRAL_BUCKETS: tuple[str, ...] = ("UNSPECIFIED", "CW", "CCW", "OTHER")
CHIRAL_UNK = 4
CHIRAL_MASK = 5
N_CHIRAL_SLOTS = 6

#: bond-stereo buckets; UNK/MASK dedicated slots.
STEREO_BUCKETS: tuple[str, ...] = ("NONE", "E", "Z", "OTHER")
STEREO_UNK = 4
STEREO_MASK = 5
N_STEREO_SLOTS = 6

#: fixed mask ratios (frozen; same value as the ZINC round)
ATOM_MASK_RATIO = 0.15
BOND_MASK_RATIO = 0.15

#: model-side derived dims
ATOM_DEGREE_SLOTS = 7            # degree 0..6 (max observed 4; headroom for salts)
RING_SIZE_SLOTS = 6              # min ring size 3,4,5,6,7,8+
ATOM_FEAT_DIM = (
    N_ATOM_SLOTS                 # atom class one-hot (real + UNK + MASK)
    + ATOM_DEGREE_SLOTS          # own degree
    + N_CHARGE_SLOTS             # formal charge bucket
    + N_H_SLOTS                  # total H bucket
    + N_CHIRAL_SLOTS             # chirality bucket
    + 1                          # in any ring
    + RING_SIZE_SLOTS            # smallest ring through atom (zero if acyclic)
    + 2                          # n ring bonds, ring-bond fraction
    + N_BOND_SLOTS               # incident bond-class histogram
    + N_ATOM_SLOTS               # radius-1 neighbour-class histogram
    + N_ATOM_SLOTS               # radius-2 class histogram
    + 4                          # log1p(n d=2), log1p(2-paths), triangles, fused/spiro
    + 2                          # is_terminal, small-ring flag
)

BOND_FEAT_DIM = N_BOND_SLOTS + 3 + N_STEREO_SLOTS  # class one-hot, 3 ring flags, stereo

JUNC_FEAT_DIM = (
    1
    + 1
    + 1
    + 1
    + 1
    + 1
    + RING_SIZE_SLOTS
    + N_BOND_SLOTS
    + N_BOND_SLOTS * N_BOND_SLOTS
)

GLOBAL_FEAT_NAMES = (
    "log1p_n_atoms",
    "log1p_n_bonds",
    "log1p_n_junctions",
    "cyclomatic_rank_corrected",   # E - V + n_components
    "n_components",
    "ring_atom_fraction",
    "ring_bond_fraction",
    "n_ring_systems",
    "diameter_finite",
    "mean_distance_finite",
    "frac_finite_pairs_dist_ge3",
    "frac_disconnected_pairs",
    "bonds_per_atom",
    "log1p_n_ring_atoms",
    "max_degree",
    "terminal_atom_fraction",
    "junction_per_bond",
    "frac_junction_centers_ring",
    "frac_deg2_junctions",
    "frac_deg3_junctions",
    "frac_deg4_junctions",
    "bond_type0_frac",
    "bond_type1_frac",
    "bond_type2_frac",
    "bond_type3_frac",
    "bond_type4_unk_frac",
    "bond_type5_mask_frac",
    "ring_size3_frac",
    "ring_size4_frac",
    "ring_size5_frac",
    "ring_size6_frac",
    "ring_size7_frac",
    "ring_size8plus_frac",
    "fused_ring_atom_fraction",
    "distinct_bond_types",
    "distinct_atom_types",
    "charged_atom_fraction",
    "mean_abs_formal_charge",
)
GLOBAL_DIM = len(GLOBAL_FEAT_NAMES)


def atom_blocks() -> dict[str, slice]:
    """Atom-feature block layout (name -> slice); used by the leak audits."""
    off = 0
    out: dict[str, slice] = {}
    out["atom_class"] = slice(off, off + N_ATOM_SLOTS)
    off += N_ATOM_SLOTS
    out["degree"] = slice(off, off + ATOM_DEGREE_SLOTS)
    off += ATOM_DEGREE_SLOTS
    out["charge"] = slice(off, off + N_CHARGE_SLOTS)
    off += N_CHARGE_SLOTS
    out["hcount"] = slice(off, off + N_H_SLOTS)
    off += N_H_SLOTS
    out["chirality"] = slice(off, off + N_CHIRAL_SLOTS)
    off += N_CHIRAL_SLOTS
    out["in_ring"] = slice(off, off + 1)
    off += 1
    out["min_ring_size"] = slice(off, off + RING_SIZE_SLOTS)
    off += RING_SIZE_SLOTS
    out["ring_bonds"] = slice(off, off + 2)
    off += 2
    out["bond_type_counts"] = slice(off, off + N_BOND_SLOTS)
    off += N_BOND_SLOTS
    out["neighbour_types"] = slice(off, off + N_ATOM_SLOTS)
    off += N_ATOM_SLOTS
    out["radius2_types"] = slice(off, off + N_ATOM_SLOTS)
    off += N_ATOM_SLOTS
    out["radius2_stats"] = slice(off, off + 4)
    off += 4
    out["roles"] = slice(off, off + 2)
    off += 2
    assert off == ATOM_FEAT_DIM, (off, ATOM_FEAT_DIM)
    return out


# ---------------------------------------------------------------------------
# raw molecule container
# ---------------------------------------------------------------------------


@dataclass
class RawMol:
    """Label-free raw molecular graph in the frozen Lipophilicity vocabulary."""

    mol_index: int
    compound_id: str
    smiles: str
    atom_class: np.ndarray       # [n] int64, 0..ATOM_UNK
    atom_charge: np.ndarray      # [n] int64, 0..CHARGE_UNK
    atom_h: np.ndarray           # [n] int64, 0..H_UNK
    atom_chirality: np.ndarray   # [n] int64, 0..CHIRAL_UNK
    edge_index: np.ndarray       # [2, 2E] int64 directed (both directions)
    edge_class: np.ndarray       # [2E] int64, 0..BOND_UNK
    edge_stereo: np.ndarray      # [2E] int64, 0..STEREO_UNK
    n_components: int

    @property
    def n_atoms(self) -> int:
        return int(self.atom_class.shape[0])

    @property
    def n_physical_bonds(self) -> int:
        return int(self.edge_class.shape[0]) // 2


# ---------------------------------------------------------------------------
# RDKit -> raw graph
# ---------------------------------------------------------------------------


def _charge_code(value: int) -> int:
    if value in CHARGE_BUCKETS:
        return CHARGE_BUCKETS.index(int(value))
    return CHARGE_UNK


def _h_code(value: int) -> int:
    if int(value) in H_BUCKETS:
        return H_BUCKETS.index(int(value))
    if int(value) > H_BUCKETS[-1]:
        return len(H_BUCKETS) - 1  # 4+
    return H_UNK


def _chirality_code(tag: Any) -> int:
    from rdkit import Chem

    mapping = {
        Chem.ChiralType.CHI_UNSPECIFIED: 0,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    }
    return mapping.get(tag, 3)


def _stereo_code(stereo: Any) -> int:
    from rdkit import Chem

    mapping = {
        Chem.BondStereo.STEREONONE: 0,
        Chem.BondStereo.STEREOE: 1,
        Chem.BondStereo.STEREOZ: 2,
    }
    return mapping.get(stereo, 3)


def raw_from_smiles(smiles: str, mol_index: int, compound_id: str = "") -> RawMol:
    """Deterministic RDKit parse -> frozen raw-graph arrays (label-blind)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    Chem.SanitizeMol(mol)
    n = mol.GetNumAtoms()
    atom_class = np.full(n, ATOM_UNK, dtype=np.int64)
    atom_charge = np.full(n, CHARGE_UNK, dtype=np.int64)
    atom_h = np.full(n, H_UNK, dtype=np.int64)
    atom_chiral = np.full(n, CHIRAL_UNK, dtype=np.int64)
    for a in mol.GetAtoms():
        i = int(a.GetIdx())
        atom_class[i] = _ELEMENT_TO_CLASS.get((a.GetSymbol(), bool(a.GetIsAromatic())), ATOM_UNK)
        atom_charge[i] = _charge_code(int(a.GetFormalCharge()))
        atom_h[i] = _h_code(int(a.GetTotalNumHs()))
        atom_chiral[i] = _chirality_code(a.GetChiralTag())
    edges: list[tuple[int, int]] = []
    classes: list[int] = []
    stereo: list[int] = []
    for b in mol.GetBonds():
        i, j = int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())
        bt = str(b.GetBondType())
        code = BOND_CLASSES.index(bt) if bt in BOND_CLASSES else BOND_UNK
        st = _stereo_code(b.GetStereo())
        edges.extend([(i, j), (j, i)])
        classes.extend([code, code])
        stereo.extend([st, st])
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return RawMol(
        mol_index=int(mol_index),
        compound_id=str(compound_id),
        smiles=str(smiles),
        atom_class=atom_class,
        atom_charge=atom_charge,
        atom_h=atom_h,
        atom_chirality=atom_chiral,
        edge_index=edge_index,
        edge_class=np.asarray(classes, dtype=np.int64),
        edge_stereo=np.asarray(stereo, dtype=np.int64),
        n_components=int(len(Chem.GetMolFrags(mol))),
    )


def raw_from_record(rec: Mapping[str, Any]) -> RawMol:
    return RawMol(
        mol_index=int(rec["mol_index"]),
        compound_id=str(rec["compound_id"]),
        smiles=str(rec["smiles"]),
        atom_class=np.asarray(rec["atom_class"], dtype=np.int64),
        atom_charge=np.asarray(rec["atom_charge"], dtype=np.int64),
        atom_h=np.asarray(rec["atom_h"], dtype=np.int64),
        atom_chirality=np.asarray(rec["atom_chirality"], dtype=np.int64),
        edge_index=np.asarray(rec["edge_index"], dtype=np.int64).reshape(2, -1),
        edge_class=np.asarray(rec["edge_class"], dtype=np.int64),
        edge_stereo=np.asarray(rec["edge_stereo"], dtype=np.int64),
        n_components=int(rec["n_components"]),
    )


def raw_to_record(raw: RawMol) -> dict[str, Any]:
    return {
        "mol_index": int(raw.mol_index),
        "compound_id": raw.compound_id,
        "smiles": raw.smiles,
        "atom_class": [int(v) for v in raw.atom_class],
        "atom_charge": [int(v) for v in raw.atom_charge],
        "atom_h": [int(v) for v in raw.atom_h],
        "atom_chirality": [int(v) for v in raw.atom_chirality],
        "edge_index": [[int(u) for u in row] for row in raw.edge_index],
        "edge_class": [int(v) for v in raw.edge_class],
        "edge_stereo": [int(v) for v in raw.edge_stereo],
        "n_components": int(raw.n_components),
    }


def save_raw_dataset(path: Path | str, raws: Sequence[RawMol], meta: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "bjoc-lipo-rawgraph-v1",
        "vocab": vocab_manifest(),
        "meta": {str(k): v for k, v in meta.items()},
        "rows": [raw_to_record(r) for r in raws],
    }
    with gzip.open(Path(path), "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)


def load_raw_dataset(path: Path | str) -> tuple[list[RawMol], dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") != "bjoc-lipo-rawgraph-v1":
        raise ValueError(f"unexpected raw dataset schema {payload.get('schema')!r}")
    raws = [raw_from_record(r) for r in payload["rows"]]
    return raws, payload


def vocab_manifest() -> dict[str, Any]:
    return {
        "atom_classes": [list(p) for p in ATOM_CLASSES],
        "atom_unk": ATOM_UNK,
        "atom_mask": ATOM_MASK,
        "n_atom_slots": N_ATOM_SLOTS,
        "bond_classes": list(BOND_CLASSES),
        "bond_unk": BOND_UNK,
        "bond_mask": BOND_MASK,
        "n_bond_slots": N_BOND_SLOTS,
        "charge_buckets": list(CHARGE_BUCKETS),
        "charge_unk": CHARGE_UNK,
        "charge_mask": CHARGE_MASK,
        "h_buckets": list(H_BUCKETS),
        "h_unk": H_UNK,
        "h_mask": H_MASK,
        "chiral_buckets": list(CHIRAL_BUCKETS),
        "chiral_unk": CHIRAL_UNK,
        "chiral_mask": CHIRAL_MASK,
        "stereo_buckets": list(STEREO_BUCKETS),
        "stereo_unk": STEREO_UNK,
        "stereo_mask": STEREO_MASK,
        "mask_ratios": {"atom": ATOM_MASK_RATIO, "bond": BOND_MASK_RATIO},
    }


# ---------------------------------------------------------------------------
# object feature construction (adapted from structural_relation_objects_v1)
# ---------------------------------------------------------------------------


def _ring_size_bucket(size: int) -> int:
    return sro._ring_size_bucket(size)


def _components(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> int:
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, _ in bonds:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    return len({find(i) for i in range(n_atoms)})


def build_lipo_objects(raw: RawMol) -> sro.MolObjects:
    """Raw Lipophilicity graph -> one-shot BJOC object tensors (label-blind)."""
    atom_class = [int(v) for v in np.asarray(raw.atom_class).reshape(-1)]
    atom_charge = [int(v) for v in np.asarray(raw.atom_charge).reshape(-1)]
    atom_h = [int(v) for v in np.asarray(raw.atom_h).reshape(-1)]
    atom_chiral = [int(v) for v in np.asarray(raw.atom_chirality).reshape(-1)]
    ei = np.asarray(raw.edge_index)
    directed = [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])]
    directed_types = [int(t) for t in np.asarray(raw.edge_class).reshape(-1)]
    directed_stereo = [int(t) for t in np.asarray(raw.edge_stereo).reshape(-1)]
    bonds = cu.unique_undirected_bonds(directed, directed_types)
    # stereo consistency across the two directed records (per physical bond)
    stereo_by_bond: dict[tuple[int, int], list[int]] = {}
    for (u, v), st in zip(directed, directed_stereo):
        stereo_by_bond.setdefault((min(u, v), max(u, v)), []).append(int(st))
    for key, values in stereo_by_bond.items():
        if len(set(values)) != 1:
            raise ValueError(f"inconsistent bond stereo across directions for {key}: {values}")
    bond_stereo = [stereo_by_bond[(u, v)][0] for u, v, _ in bonds]

    n_atoms = len(atom_class)
    n_bonds = len(bonds)
    n_components = _components(n_atoms, bonds)
    adj = sro._adjacency(n_atoms, bonds)
    is_ring_bond, ring_atom, n_ring_systems = sro._ring_bond_flags(n_atoms, bonds)
    min_ring = sro._min_ring_size_per_atom(n_atoms, adj)
    dist = sro._all_pairs_distances(n_atoms, adj)

    for code in atom_class:
        if not (0 <= code <= ATOM_MASK):
            raise ValueError(f"atom class code {code} outside [0,{ATOM_MASK}]")
    for code in directed_types:
        if not (0 <= code <= BOND_MASK):
            raise ValueError(f"bond class code {code} outside [0,{BOND_MASK}]")

    degree = np.array([len(adj[i]) for i in range(n_atoms)], dtype=np.int64)
    ring_bond_count = np.zeros(n_atoms, dtype=np.int64)
    for k, (u, v, _) in enumerate(bonds):
        if is_ring_bond[k]:
            ring_bond_count[u] += 1
            ring_bond_count[v] += 1

    atom_feat = np.zeros((n_atoms, ATOM_FEAT_DIM), dtype=np.float32)
    off = 0
    for i in range(n_atoms):
        atom_feat[i, off + atom_class[i]] = 1.0
    off += N_ATOM_SLOTS
    for i in range(n_atoms):
        d = int(degree[i])
        if d >= ATOM_DEGREE_SLOTS:
            d = ATOM_DEGREE_SLOTS - 1
        atom_feat[i, off + d] = 1.0
    off += ATOM_DEGREE_SLOTS
    for i in range(n_atoms):
        atom_feat[i, off + atom_charge[i]] = 1.0
    off += N_CHARGE_SLOTS
    for i in range(n_atoms):
        atom_feat[i, off + atom_h[i]] = 1.0
    off += N_H_SLOTS
    for i in range(n_atoms):
        atom_feat[i, off + atom_chiral[i]] = 1.0
    off += N_CHIRAL_SLOTS
    atom_feat[:, off] = ring_atom.astype(np.float32)
    off += 1
    for i in range(n_atoms):
        slot = _ring_size_bucket(int(min_ring[i]))
        if slot >= 0:
            atom_feat[i, off + slot] = 1.0
    off += RING_SIZE_SLOTS
    atom_feat[:, off] = ring_bond_count.astype(np.float32)
    atom_feat[:, off + 1] = np.where(degree > 0, ring_bond_count / np.maximum(degree, 1), 0.0)
    off += 2
    for k, (u, v, bt) in enumerate(bonds):
        atom_feat[u, off + bt] += 1.0
        atom_feat[v, off + bt] += 1.0
    off += N_BOND_SLOTS
    for i in range(n_atoms):
        for v, _ in adj[i]:
            atom_feat[i, off + atom_class[v]] += 1.0
    off += N_ATOM_SLOTS
    n_d2 = np.zeros(n_atoms, dtype=np.float32)
    paths2 = np.zeros(n_atoms, dtype=np.float32)
    triangles = np.zeros(n_atoms, dtype=np.float32)
    for i in range(n_atoms):
        for v in range(n_atoms):
            if v != i and dist[i, v] == 2:
                atom_feat[i, off + atom_class[v]] += 1.0
                n_d2[i] += 1.0
        for v, _ in adj[i]:
            paths2[i] += max(int(degree[v]) - 1, 0)
        for v in sorted(nbr for nbr, _ in adj[i]):
            for w, _ in adj[v]:
                if w > v and w != i and dist[i, w] == 1:
                    triangles[i] += 1.0
    off2 = off + N_ATOM_SLOTS
    atom_feat[:, off2] = np.log1p(n_d2)
    atom_feat[:, off2 + 1] = np.log1p(paths2)
    atom_feat[:, off2 + 2] = triangles
    atom_feat[:, off2 + 3] = (ring_bond_count >= 3).astype(np.float32)
    off2 += 4
    atom_feat[:, off2] = (degree == 1).astype(np.float32)
    atom_feat[:, off2 + 1] = (((ring_bond_count >= 3) | (min_ring >= 3) & (min_ring <= 4))).astype(np.float32)
    assert off2 + 2 == ATOM_FEAT_DIM, (off2 + 2, ATOM_FEAT_DIM)

    bond_feat = np.zeros((n_bonds, BOND_FEAT_DIM), dtype=np.float32)
    for k, (u, v, bt) in enumerate(bonds):
        bond_feat[k, bt] = 1.0
        bond_feat[k, N_BOND_SLOTS] = float(is_ring_bond[k])
        ru, rv = bool(ring_atom[u]), bool(ring_atom[v])
        bond_feat[k, N_BOND_SLOTS + 1] = float(ru != rv)
        bond_feat[k, N_BOND_SLOTS + 2] = float(ru and rv)
        bond_feat[k, N_BOND_SLOTS + 3 + bond_stereo[k]] = 1.0

    bond_ij: dict[tuple[int, int], int] = {}
    for k, (u, v, _) in enumerate(bonds):
        bond_ij[(u, v)] = k
    neighbours: list[list[int]] = [[v for v, _ in adj[i]] for i in range(n_atoms)]
    junc_i: list[tuple[int, int, int]] = []
    junc_bonds: list[tuple[int, int]] = []
    for j in range(n_atoms):
        nbrs = sorted(neighbours[j])
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                junc_i.append((i, j, k))
                junc_bonds.append((bond_ij[(min(i, j), max(i, j))], bond_ij[(min(j, k), max(j, k))]))
    n_junc = len(junc_i)
    junc_index = np.asarray(junc_i, dtype=np.int64).reshape(n_junc, 3)
    junc_bond = np.asarray(junc_bonds, dtype=np.int64).reshape(n_junc, 2)
    junc_feat = np.zeros((n_junc, JUNC_FEAT_DIM), dtype=np.float32)
    for idx, (i, j, k) in enumerate(junc_i):
        b1, b2 = junc_bonds[idx]
        t1, t2 = bonds[b1][2], bonds[b2][2]
        off = 0
        junc_feat[idx, off] = degree[j]
        off += 1
        junc_feat[idx, off] = float((min(i, k), max(i, k)) in bond_ij)
        off += 1
        common = set(neighbours[i]) & set(neighbours[k])
        common.discard(j)
        junc_feat[idx, off] = np.log1p(len(common))
        off += 1
        r = float(is_ring_bond[b1]) + float(is_ring_bond[b2])
        junc_feat[idx, off] = r
        junc_feat[idx, off + 1] = float(is_ring_bond[b1] and is_ring_bond[b2])
        junc_feat[idx, off + 2] = float(ring_atom[j])
        off += 3
        slot = _ring_size_bucket(int(min_ring[j]))
        if slot >= 0:
            junc_feat[idx, off + slot] = 1.0
        off += RING_SIZE_SLOTS
        junc_feat[idx, off + t1] += 1.0
        junc_feat[idx, off + t2] += 1.0
        off += N_BOND_SLOTS
        junc_feat[idx, off + t1 * N_BOND_SLOTS + t2] += 1.0
        junc_feat[idx, off + t2 * N_BOND_SLOTS + t1] += 1.0
        off += N_BOND_SLOTS * N_BOND_SLOTS
        assert off == JUNC_FEAT_DIM, (off, JUNC_FEAT_DIM)

    global_feat = _global_features(
        n_atoms, bonds, is_ring_bond, ring_atom, min_ring, junc_index, n_ring_systems,
        dist, atom_class, atom_charge, bond_stereo, n_components,
    )
    return sro.MolObjects(
        mol_index=int(raw.mol_index),
        smiles=str(raw.smiles),
        n_atoms=n_atoms,
        atom_feat=atom_feat,
        bond_index=np.asarray([(u, v) for u, v, _ in bonds], dtype=np.int64).reshape(n_bonds, 2),
        bond_type=np.asarray([bt for _, _, bt in bonds], dtype=np.int64),
        bond_feat=bond_feat,
        junc_index=junc_index,
        junc_bond=junc_bond,
        junc_feat=junc_feat,
        global_feat=global_feat,
    )


def _global_features(
    n_atoms: int,
    bonds,
    is_ring_bond: np.ndarray,
    ring_atom: np.ndarray,
    min_ring: np.ndarray,
    junc_index: np.ndarray,
    n_ring_systems: int,
    dist: np.ndarray,
    atom_class: Sequence[int],
    atom_charge: Sequence[int],
    bond_stereo: Sequence[int],
    n_components: int,
) -> np.ndarray:
    n_bonds = len(bonds)
    n_junc = int(junc_index.shape[0])
    inf = 1 << 20
    all_pairs = max(n_atoms * (n_atoms - 1), 1)
    finite_mask = (dist < inf) & ~np.eye(n_atoms, dtype=bool)
    finite_vals = dist[finite_mask]
    diameter = float(finite_vals.max()) if finite_vals.size else 0.0
    mean_distance = float(finite_vals.mean()) if finite_vals.size else 0.0
    frac_ge3 = float((finite_vals >= 3).sum() / max(finite_vals.size, 1))
    frac_disconnected = float((dist >= inf).sum() / all_pairs)
    degree = np.zeros(n_atoms, dtype=np.int64)
    ring_bond_count = np.zeros(n_atoms, dtype=np.int64)
    for k, (u, v, _) in enumerate(bonds):
        degree[u] += 1
        degree[v] += 1
        if is_ring_bond[k]:
            ring_bond_count[u] += 1
            ring_bond_count[v] += 1
    ring_size_counts = np.zeros(RING_SIZE_SLOTS, dtype=np.float64)
    for i in range(n_atoms):
        slot = _ring_size_bucket(int(min_ring[i]))
        if slot >= 0 and ring_atom[i]:
            ring_size_counts[slot] += 1.0
    if ring_atom.any():
        ring_size_counts /= float(ring_atom.sum())
    bond_type_counts = np.zeros(N_BOND_SLOTS, dtype=np.float64)
    for _, _, bt in bonds:
        bond_type_counts[bt] += 1.0
    if n_bonds:
        bond_type_counts /= float(n_bonds)
    stereo_counts = np.zeros(N_STEREO_SLOTS, dtype=np.float64)
    for st in bond_stereo:
        stereo_counts[st] += 1.0
    center_deg = degree[junc_index[:, 1]] if n_junc else np.zeros(0, dtype=np.int64)
    fused = float((ring_bond_count >= 3).mean()) if n_atoms else 0.0
    charge_values = np.zeros(n_atoms, dtype=np.float64)
    charged = 0.0
    for i, code in enumerate(atom_charge):
        if 0 <= code < len(CHARGE_BUCKETS):
            charge_values[i] = abs(float(CHARGE_BUCKETS[code]))
            charged += float(CHARGE_BUCKETS[code] != 0)
    mean_abs_charge = float(charge_values.mean()) if n_atoms else 0.0
    values = [
        np.log1p(n_atoms),
        np.log1p(n_bonds),
        np.log1p(n_junc),
        float(n_bonds - n_atoms + n_components),
        float(n_components),
        float(ring_atom.mean()) if n_atoms else 0.0,
        float(is_ring_bond.mean()) if n_bonds else 0.0,
        float(n_ring_systems),
        diameter,
        mean_distance,
        frac_ge3,
        frac_disconnected,
        float(n_bonds) / max(n_atoms, 1),
        np.log1p(float(ring_atom.sum())),
        float(degree.max()) if n_atoms else 0.0,
        float((degree == 1).mean()) if n_atoms else 0.0,
        float(n_junc) / max(n_bonds, 1),
        float(((degree[junc_index[:, 1]] > 0) & ring_atom[junc_index[:, 1]]).mean()) if n_junc else 0.0,
        float((center_deg == 2).mean()) if n_junc else 0.0,
        float((center_deg == 3).mean()) if n_junc else 0.0,
        float((center_deg >= 4).mean()) if n_junc else 0.0,
        *bond_type_counts.tolist(),
        *ring_size_counts.tolist(),
        fused,
        float(len(set(int(b) for _, _, b in bonds))),
        float(len(set(int(t) for t in atom_class))),
        float(charged / max(n_atoms, 1)),
        mean_abs_charge,
    ]
    out = np.asarray(values, dtype=np.float32)
    assert out.shape[0] == GLOBAL_DIM, (out.shape[0], GLOBAL_DIM)
    return out


# ---------------------------------------------------------------------------
# batched tensor assembly (one-shot gathers; no message passing)
# ---------------------------------------------------------------------------


def batch_lipo_objects(objects: Sequence[sro.MolObjects]):
    import torch

    atom_rows, atom_batch = [], []
    bond_rows, bond_batch, bond_index_all = [], [], []
    junc_rows, junc_batch, junc_index_all, junc_bond_all = [], [], [], []
    global_rows = []
    a_off = b_off = 0
    for g, obj in enumerate(objects):
        n_a, n_b, n_j = obj.n_atoms, obj.bond_index.shape[0], obj.junc_index.shape[0]
        atom_rows.append(torch.as_tensor(obj.atom_feat, dtype=torch.float32))
        atom_batch.append(torch.full((n_a,), g, dtype=torch.long))
        bond_rows.append(torch.as_tensor(obj.bond_feat, dtype=torch.float32))
        bond_batch.append(torch.full((n_b,), g, dtype=torch.long))
        bond_index_all.append(torch.as_tensor(obj.bond_index, dtype=torch.long) + a_off)
        junc_rows.append(torch.as_tensor(obj.junc_feat, dtype=torch.float32))
        junc_batch.append(torch.full((n_j,), g, dtype=torch.long))
        junc_index_all.append(torch.as_tensor(obj.junc_index, dtype=torch.long) + a_off)
        junc_bond_all.append(torch.as_tensor(obj.junc_bond, dtype=torch.long) + b_off)
        global_rows.append(torch.as_tensor(obj.global_feat, dtype=torch.float32))
        a_off += n_a
        b_off += n_b
    return {
        "n_graphs": len(objects),
        "atom_feat": torch.cat(atom_rows, 0) if atom_rows else torch.zeros(0, ATOM_FEAT_DIM),
        "atom_batch": torch.cat(atom_batch, 0) if atom_batch else torch.zeros(0, dtype=torch.long),
        "bond_feat": torch.cat(bond_rows, 0) if bond_rows else torch.zeros(0, BOND_FEAT_DIM),
        "bond_batch": torch.cat(bond_batch, 0) if bond_batch else torch.zeros(0, dtype=torch.long),
        "bond_index": torch.cat(bond_index_all, 0) if bond_index_all else torch.zeros(0, 2, dtype=torch.long),
        "junc_feat": torch.cat(junc_rows, 0) if junc_rows else torch.zeros(0, JUNC_FEAT_DIM),
        "junc_batch": torch.cat(junc_batch, 0) if junc_batch else torch.zeros(0, dtype=torch.long),
        "junc_index": torch.cat(junc_index_all, 0) if junc_index_all else torch.zeros(0, 3, dtype=torch.long),
        "junc_bond": torch.cat(junc_bond_all, 0) if junc_bond_all else torch.zeros(0, 2, dtype=torch.long),
        "global_feat": torch.stack(global_rows, 0) if global_rows else torch.zeros(0, GLOBAL_DIM),
    }


# ---------------------------------------------------------------------------
# model config (input widths only; hidden widths are the frozen BJOC values)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LipoConfig(src.BJOCConfig):
    """Frozen BJOC widths with the Lipophilicity input vocabulary."""

    @property
    def atom_in(self) -> int:
        return ATOM_FEAT_DIM

    @property
    def bond_in(self) -> int:
        return 3 * self.atom_out + BOND_FEAT_DIM

    @property
    def junc_in(self) -> int:
        return 4 * self.atom_out + 2 * self.bond_out + JUNC_FEAT_DIM

    @property
    def compose_in(self) -> int:
        return self.bond_out + 2 * self.junc_out + 1 + BOND_FEAT_DIM

    @property
    def reader_in(self) -> int:
        return self.compose_out + 2 * self.atom_out + GLOBAL_DIM


def build_lipo_model(
    config: LipoConfig,
    *,
    seed: int,
    organization_mode: str,
    global_mean: np.ndarray,
    global_scale: np.ndarray,
    device: Any = "cpu",
) -> src.BJOC:
    """Deterministic builder; mode never touches initialisation."""
    import torch

    torch.manual_seed(int(seed))
    model = src.BJOC(config, global_mean, global_scale, organization_mode)
    return model.to(device)


# ---------------------------------------------------------------------------
# masked SSL (frozen logic; adapted vocabulary)
# ---------------------------------------------------------------------------


def mask_seed_sequence(seed: int, mol_id: int, visit_index: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed), int(mol_id), int(visit_index)])


def sample_mask(seed: int, mol_id: int, visit_index: int, n_atoms: int, n_bonds: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(mask_seed_sequence(seed, mol_id, visit_index))
    n_atoms, n_bonds = int(n_atoms), int(n_bonds)
    if n_atoms > 0:
        k_a = max(1, int(round(ATOM_MASK_RATIO * n_atoms)))
        atom_ids = np.sort(rng.choice(n_atoms, size=k_a, replace=False)).astype(np.int64)
    else:
        atom_ids = np.zeros(0, dtype=np.int64)
    if n_bonds > 0:
        k_b = max(1, int(round(BOND_MASK_RATIO * n_bonds)))
        bond_ids = np.sort(rng.choice(n_bonds, size=k_b, replace=False)).astype(np.int64)
    else:
        bond_ids = np.zeros(0, dtype=np.int64)
    return atom_ids, bond_ids


def mask_raw(raw: RawMol, atom_ids: np.ndarray, bond_ids: np.ndarray) -> RawMol:
    """Replace every raw field of the selected atoms/bonds by the mask token.

    Atoms: class / charge / H / chirality are all masked together (no residual
    identity field can leak the hidden class).  Bonds: both directed records of
    each masked physical bond get class MASK and stereo MASK.
    """
    bond_index = cu.unique_undirected_bonds(
        [(int(raw.edge_index[0, k]), int(raw.edge_index[1, k])) for k in range(raw.edge_index.shape[1])],
        [int(v) for v in raw.edge_class],
    )
    masked = RawMol(
        mol_index=raw.mol_index,
        compound_id=raw.compound_id,
        smiles=raw.smiles,
        atom_class=raw.atom_class.copy(),
        atom_charge=raw.atom_charge.copy(),
        atom_h=raw.atom_h.copy(),
        atom_chirality=raw.atom_chirality.copy(),
        edge_index=raw.edge_index.copy(),
        edge_class=raw.edge_class.copy(),
        edge_stereo=raw.edge_stereo.copy(),
        n_components=raw.n_components,
    )
    ids = np.asarray(atom_ids, dtype=np.int64)
    masked.atom_class[ids] = ATOM_MASK
    masked.atom_charge[ids] = CHARGE_MASK
    masked.atom_h[ids] = H_MASK
    masked.atom_chirality[ids] = CHIRAL_MASK
    if len(bond_ids):
        directed = [(int(masked.edge_index[0, k]), int(masked.edge_index[1, k])) for k in range(masked.edge_index.shape[1])]
        mapping: dict[tuple[int, int], list[int]] = {}
        for col, (u, v) in enumerate(directed):
            mapping.setdefault((min(u, v), max(u, v)), []).append(col)
        for k in np.asarray(bond_ids, dtype=np.int64):
            u, v, _ = bond_index[int(k)]
            for col in mapping[(min(u, v), max(u, v))]:
                masked.edge_class[col] = BOND_MASK
                masked.edge_stereo[col] = STEREO_MASK
    return masked


@dataclass
class PretextSample:
    """Masked object plus hidden truth kept *outside* the model input."""

    masked: sro.MolObjects
    atom_ids: np.ndarray
    atom_classes: np.ndarray
    bond_ids: np.ndarray
    bond_classes: np.ndarray


def build_masked_sample(raw: RawMol, *, seed: int, visit_index: int) -> PretextSample:
    """Full-rebuild masked sample (rebuilds every object feature, no patching)."""
    bond_index = cu.unique_undirected_bonds(
        [(int(raw.edge_index[0, k]), int(raw.edge_index[1, k])) for k in range(raw.edge_index.shape[1])],
        [int(v) for v in raw.edge_class],
    )
    n_bonds = len(bond_index)
    atom_ids, bond_ids = sample_mask(seed, raw.mol_index, visit_index, raw.n_atoms, n_bonds)
    masked_raw = mask_raw(raw, atom_ids, bond_ids)
    masked = build_lipo_objects(masked_raw)
    atom_classes = np.asarray([int(raw.atom_class[i]) for i in atom_ids], dtype=np.int64)
    bond_classes = np.asarray([int(bond_index[k][2]) for k in bond_ids], dtype=np.int64)
    return PretextSample(
        masked=masked,
        atom_ids=atom_ids,
        atom_classes=atom_classes,
        bond_ids=bond_ids,
        bond_classes=bond_classes,
    )


class LipoPretextModel(nn.Module):
    """The published BJOC encoders + two discarded reconstruction heads.

    Same static-incidence context as the ZINC masked-pretraining round; only
    the head output vocabularies differ (15 atom classes, 4 bond classes).
    """

    def __init__(self, backbone: src.BJOC, head_hidden: int = 160) -> None:
        super().__init__()
        self.backbone = backbone
        atom_out = int(backbone.config.atom_out)
        bond_out = int(backbone.config.bond_out)
        junc_out = int(backbone.config.junc_out)
        self.atom_head = src._MLP(4 * atom_out, head_hidden, N_ATOM_TARGETS, 1, 0.0)
        self.bond_head = src._MLP(4 * bond_out + junc_out, head_hidden, N_BOND_TARGETS, 1, 0.0)

    @staticmethod
    def _incidence(batch: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        bi = batch["bond_index"]
        n_bonds = int(bi.shape[0])
        ji = batch["junc_index"]
        n_junc = int(ji.shape[0])
        dev = bi.device
        bond_range = torch.arange(n_bonds, device=dev, dtype=torch.long)
        junc_range = torch.arange(n_junc, device=dev, dtype=torch.long)
        return {
            "hole_atom": torch.cat([bi[:, 0], bi[:, 1]]),
            "hole_bond": torch.cat([bond_range, bond_range]),
            "junc_center_atom": ji[:, 1],
            "junc_endpoint_atom": torch.cat([ji[:, 0], ji[:, 2]]),
            "junc_endpoint_row": torch.cat([junc_range, junc_range]),
            "jb1": batch["junc_bond"][:, 0],
            "jb2": batch["junc_bond"][:, 1],
        }

    def forward(self, batch: Mapping[str, Any], masked_atom_ids: Any, masked_bond_ids: Any) -> dict[str, Any]:
        import torch

        backbone = self.backbone
        a = backbone.encode_atoms(batch)
        b = backbone.encode_bonds(a, batch)
        w = backbone.encode_junctions(a, b, batch)
        inc = self._incidence(batch)
        n_atoms = int(a.shape[0])
        n_bonds = int(b.shape[0])

        sum_b = torch.zeros((n_atoms, b.shape[1]), device=a.device, dtype=a.dtype)
        sum_b = sum_b.index_add(0, inc["hole_atom"], b[inc["hole_bond"]])
        sum_wc = torch.zeros((n_atoms, w.shape[1]), device=a.device, dtype=a.dtype)
        sum_wc = sum_wc.index_add(0, inc["junc_center_atom"], w)
        sum_we = torch.zeros((n_atoms, w.shape[1]), device=a.device, dtype=a.dtype)
        sum_we = sum_we.index_add(0, inc["junc_endpoint_atom"], w[inc["junc_endpoint_row"]])
        cat_atom = torch.cat([a, sum_b, sum_wc, sum_we], dim=1)

        sum_wb = torch.zeros((n_bonds, w.shape[1]), device=a.device, dtype=a.dtype)
        sum_wb = sum_wb.index_add(0, inc["jb1"], w)
        sum_wb = sum_wb.index_add(0, inc["jb2"], w)
        bi = batch["bond_index"]
        au, av = a[bi[:, 0]], a[bi[:, 1]]
        cat_bond = torch.cat([b, au + av, (au - av).abs(), au * av, sum_wb], dim=1)

        return {
            "a": a,
            "b": b,
            "w": w,
            "atom_logits": self.atom_head(cat_atom[masked_atom_ids]),
            "bond_logits": self.bond_head(cat_bond[masked_bond_ids]),
        }

    def loss(self, batch: Mapping[str, Any], atom_targets: Any, bond_targets: Any) -> tuple[Any, dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        logits = self(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])
        ce_atom = F.cross_entropy(logits["atom_logits"], atom_targets) if atom_targets.numel() else None
        ce_bond = F.cross_entropy(logits["bond_logits"], bond_targets) if bond_targets.numel() else None
        if ce_atom is None and ce_bond is None:
            raise ValueError("empty pretext batch")
        loss = torch.zeros((), device=logits["a"].device, dtype=logits["a"].dtype)
        out: dict[str, Any] = {}
        if ce_atom is not None:
            loss = loss + ce_atom
            out["ce_atom"] = ce_atom
        if ce_bond is not None:
            loss = loss + ce_bond
            out["ce_bond"] = ce_bond
        out["loss"] = loss
        return loss, out


def batch_pretext_samples(samples: Sequence[PretextSample]) -> tuple[dict[str, Any], Any, Any, np.ndarray, np.ndarray]:
    import torch

    batch = batch_lipo_objects([s.masked for s in samples])
    a_off, b_off = 0, 0
    atom_rows, atom_targets = [], []
    bond_rows, bond_targets = [], []
    for s in samples:
        atom_rows.append(torch.as_tensor(s.atom_ids + a_off, dtype=torch.long))
        atom_targets.append(torch.as_tensor(s.atom_classes, dtype=torch.long))
        bond_rows.append(torch.as_tensor(s.bond_ids + b_off, dtype=torch.long))
        bond_targets.append(torch.as_tensor(s.bond_classes, dtype=torch.long))
        a_off += s.masked.n_atoms
        b_off += s.masked.bond_index.shape[0]
    batch["masked_atom_ids"] = torch.cat(atom_rows, 0) if atom_rows else torch.zeros(0, dtype=torch.long)
    batch["masked_bond_ids"] = torch.cat(bond_rows, 0) if bond_rows else torch.zeros(0, dtype=torch.long)
    atom_targets_t = torch.cat(atom_targets, 0) if atom_targets else torch.zeros(0, dtype=torch.long)
    bond_targets_t = torch.cat(bond_targets, 0) if bond_targets else torch.zeros(0, dtype=torch.long)
    return (
        batch,
        atom_targets_t,
        bond_targets_t,
        batch["masked_atom_ids"].numpy().copy(),
        batch["masked_bond_ids"].numpy().copy(),
    )


def class_prior_stats(atom_codes: np.ndarray, bond_codes: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, codes, n_classes in (
        ("atom", np.asarray(atom_codes, dtype=np.int64), N_ATOM_TARGETS),
        ("bond", np.asarray(bond_codes, dtype=np.int64), N_BOND_TARGETS),
    ):
        counts = np.bincount(codes, minlength=n_classes).astype(np.float64)
        p = counts / max(counts.sum(), 1.0)
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        out[name] = {
            "n": int(counts.sum()),
            "counts": counts.astype(int).tolist(),
            "entropy_nats": entropy,
            "majority_class": int(np.argmax(counts)),
            "majority_fraction": float(p.max()),
            "marginal_ce_nats": entropy,
            "majority_accuracy": float(p.max()),
        }
    return out


# ---------------------------------------------------------------------------
# fit-only standardiser (global topology vector)
# ---------------------------------------------------------------------------


def fit_standardizer(objs: Sequence[sro.MolObjects]) -> dict[str, np.ndarray]:
    feats = np.stack([o.global_feat for o in objs], 0).astype(np.float64)
    mean = feats.mean(0)
    scale = feats.std(0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32), "n_fit": len(objs)}


# ---------------------------------------------------------------------------
# DeepChem 2.8.0 scaffold split (verbatim algorithm; structures only)
# ---------------------------------------------------------------------------


def generate_scaffold_dc280(smiles: str) -> str | None:
    """Bemis-Murcko scaffold exactly as ``deepchem.splits.splitters._generate_scaffold`` 2.8.0."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def generate_scaffold_sets_dc280(ids: Sequence[str]) -> list[list[int]]:
    """``ScaffoldSplitter.generate_scaffolds`` verbatim (DeepChem 2.8.0)."""
    scaffolds: dict[str, list[int]] = {}
    for ind, smiles in enumerate(ids):
        scaffold = generate_scaffold_dc280(smiles)
        if scaffold is not None:
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split_dc280(
    ids: Sequence[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    """``ScaffoldSplitter.split`` verbatim (DeepChem 2.8.0)."""
    if abs(frac_train + frac_valid + frac_test - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    scaffold_sets = generate_scaffold_sets_dc280(ids)
    n = len(ids)
    train_cutoff = frac_train * n
    valid_cutoff = (frac_train + frac_valid) * n
    train_inds: list[int] = []
    valid_inds: list[int] = []
    test_inds: list[int] = []
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def canonical_smiles_and_hash(smiles: str) -> tuple[str, str]:
    """RDKit canonical SMILES + sha256 of its UTF-8 bytes (structure-only)."""
    import hashlib

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot canonicalise {smiles!r}")
    can = Chem.MolToSmiles(mol)
    return can, hashlib.sha256(can.encode("utf-8")).hexdigest()
