"""CSCL-v0 feature assembly: molecules -> unit/relation tensors + fit-only stats.

Protocol source: notes/v0_protocol.md §1 (data/splits) and §2 (units/vocab).

Label discipline:
* Only the raw ``y`` of the official-train split is used as supervision.
* No ``g/ell/s/c`` auxiliary labels, no frozen Q heads, no historical model
  outputs ever enter the pipeline (asserted by tests).
* Type vocabulary, per-type descriptor means (centering), y-scaler, and any
  other data-driven statistic are fit on **fit_inner only**.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cscl_units import (  # noqa: E402
    ATOM_CATEGORIES,
    BOND_CATEGORIES,
    KIND_CHAIN,
    KIND_RING,
    UnitVocab,
    build_partition,
)

ZINC_ROOT = REPO_ROOT / "data/ZINC"
SMILES_TABLE = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"

SPLIT_SEED = "cscl-v0"
INNER_SEED = "cscl-v0-inner"

#: descriptor layout
D_HIST = ATOM_CATEGORIES + BOND_CATEGORIES  # atom hist (28) + internal bond hist (4)
D_SCALARS = 8  # log1p(n_atoms), log1p(n_intra_bonds), cyclomatic, is_ring,
#               n_attachments, n_ring_neighbors, n_chain_neighbors, is_terminal
DESC_DIM = D_HIST + D_SCALARS
REL_DIM = BOND_CATEGORIES + 1  # inter-unit bond hist (4) + log1p(count)

FIT_N = 8000
DEV_N = 2000
INNER_FIT_N = 7200
INNER_MON_N = 800


# ---------------------------------------------------------------------------
# official split (train only; official valid/test never loaded this round)
# ---------------------------------------------------------------------------


def load_official_train():
    """Canonical PyG ZINC subset=True official **train** split only."""
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

    return _load_zinc(ZINC_ROOT, "train")


def load_canonical_smiles() -> np.ndarray:
    """Committed canonical SMILES table (row i = official-train subset_index i)."""
    with np.load(SMILES_TABLE, allow_pickle=True) as z:
        return np.asarray(z["smiles"], dtype=object)


def _grouped_split(smiles: list[str], seed: str, n_first: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic grouped split (v0_protocol §1): duplicate molecules never
    cross the two sides.  Groups ordered by sha256('seed|' + smiles); rows
    within a group by row index; greedy fill of the first side to n_first rows.
    """
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(s, []).append(i)
    keys = sorted(groups, key=lambda s: hashlib.sha256(f"{seed}|{s}".encode()).hexdigest())
    first: list[int] = []
    second: list[int] = []
    for key in keys:
        rows = sorted(groups[key])
        if len(first) < n_first:
            first.extend(rows)
        else:
            second.extend(rows)
    return np.array(sorted(first), dtype=np.int64), np.array(sorted(second), dtype=np.int64)


def build_split_indices(smiles: list[str]) -> dict[str, np.ndarray]:
    """fit 8000 / dev 2000 (seed cscl-v0); fit -> inner 7200 / monitor 800
    (seed cscl-v0-inner).  Grouping: identical SMILES never cross sides."""
    fit, dev = _grouped_split(smiles, SPLIT_SEED, FIT_N)
    fit_smiles = [smiles[i] for i in fit]
    fit_in, mon = _grouped_split(fit_smiles, INNER_SEED, INNER_FIT_N)
    fit_in = fit[fit_in]
    mon = fit[mon]
    assert len(fit_in) == INNER_FIT_N and len(mon) == INNER_MON_N
    assert len(fit) == FIT_N and len(dev) == DEV_N
    return {"fit_inner": fit_in, "monitor": mon, "dev": dev, "fit_all": fit}


# ---------------------------------------------------------------------------
# per-molecule unit feature extraction
# ---------------------------------------------------------------------------


@dataclass
class MolUnits:
    """Unit/relation summary of one molecule (label-free)."""

    mol_index: int  # row in the official-train table
    unit_sigs: list[str]
    unit_kinds: list[int]
    unit_atoms: list[list[int]]
    desc: np.ndarray  # [n_units, DESC_DIM]
    rel_pairs: list[tuple[int, int]]  # unordered unit-id pairs
    rel_feat: np.ndarray  # [n_rel, REL_DIM]
    n_atoms: int
    smiles: str = ""


def molecule_units(mol_index: int, x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray, smiles: str = "") -> MolUnits:
    atom_types = [int(v) for v in x.reshape(-1)]
    edges = [(int(edge_index[0, k]), int(edge_index[1, k]), int(edge_attr.reshape(-1)[k])) for k in range(edge_index.shape[1])]
    part = build_partition(atom_types, [(u, v) for u, v, _ in edges], [bt for _, _, bt in edges])

    n_units = part.n_units
    # neighbour kinds per unit (over inter-unit bonds)
    nb_ring = [0] * n_units
    nb_chain = [0] * n_units
    rel_map = part.inter_relations()
    attach_count = [0] * n_units
    for (a, b), bts in rel_map.items():
        attach_count[a] += len(bts)
        attach_count[b] += len(bts)
        for target in (a, b):
            if part.units[target].kind == KIND_RING:
                nb_ring[target] += 1
            else:
                nb_chain[target] += 1

    desc = np.zeros((n_units, DESC_DIM), dtype=np.float32)
    for unit in part.units:
        i = unit.unit_id
        for a in unit.atoms:
            desc[i, int(atom_types[a])] += 1.0
        for _, _, bt in unit.intra_bonds:
            desc[i, ATOM_CATEGORIES + int(bt)] += 1.0
        cyc = len(unit.intra_bonds) - len(unit.atoms) + 1  # connected => cyclomatic number
        scalars = [
            np.log1p(len(unit.atoms)),
            np.log1p(len(unit.intra_bonds)),
            float(max(cyc, 0)),
            1.0 if unit.kind == KIND_RING else 0.0,
            float(attach_count[i]),
            float(nb_ring[i]),
            float(nb_chain[i]),
            1.0 if attach_count[i] <= 1 else 0.0,
        ]
        desc[i, D_HIST:] = np.asarray(scalars, dtype=np.float32)

    rel_pairs = sorted(rel_map.keys())
    rel_feat = np.zeros((len(rel_pairs), REL_DIM), dtype=np.float32)
    for k, (a, b) in enumerate(rel_pairs):
        for bt in rel_map[(a, b)]:
            rel_feat[k, int(bt)] += 1.0
        rel_feat[k, BOND_CATEGORIES] = np.log1p(len(rel_map[(a, b)]))

    return MolUnits(
        mol_index=mol_index,
        unit_sigs=[u.signature for u in part.units],
        unit_kinds=[u.kind for u in part.units],
        unit_atoms=[list(u.atoms) for u in part.units],
        desc=desc,
        rel_pairs=rel_pairs,
        rel_feat=rel_feat,
        n_atoms=part.n_atoms,
        smiles=smiles,
    )


def extract_all() -> tuple[list[MolUnits], np.ndarray, list[str]]:
    """Extract unit summaries for all 10000 official-train molecules.

    Label-free: reads x/edge_index/edge_attr; y is returned verbatim but only
    used downstream as supervision.
    """
    import torch  # local import: keep module import cheap for CPU audits

    dataset = load_official_train()
    smiles = [str(s) for s in load_canonical_smiles()]
    assert len(dataset) == len(smiles) == 10000
    mols: list[MolUnits] = []
    ys = np.zeros((len(dataset), 1), dtype=np.float32)
    for i, data in enumerate(dataset):
        mols.append(
            molecule_units(
                i,
                data.x.numpy(),
                data.edge_index.numpy(),
                data.edge_attr.numpy(),
                smiles[i],
            )
        )
        ys[i, 0] = float(data.y.reshape(-1)[0])
    return mols, ys, smiles


# ---------------------------------------------------------------------------
# fit-only statistics (vocab, centering, y-scaler)
# ---------------------------------------------------------------------------


@dataclass
class FitStats:
    """Fit-only statistics: vocabulary, per-type centering means, y-scaler."""

    vocab: UnitVocab = field(default_factory=UnitVocab)
    type_means: dict[int, np.ndarray] = field(default_factory=dict)  # type id -> DESC_DIM
    global_mean: np.ndarray | None = None
    y_mean: float = 0.0
    y_std: float = 1.0
    rel_mean: np.ndarray | None = None

    def __init__(self, min_count: int = 3) -> None:
        self.vocab = UnitVocab(min_count=min_count)
        self.type_means = {}
        self.global_mean = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.rel_mean = None

    def fit(self, mols: list[MolUnits], ys: np.ndarray) -> "FitStats":
        sigs = [s for m in mols for s in m.unit_sigs]
        self.vocab.fit(sigs)

        # per-type descriptor means (centering; v0_protocol §3)
        acc: dict[int, list[np.ndarray]] = {}
        for m in mols:
            for k, sig in enumerate(m.unit_sigs):
                tid = self.vocab.to_id(sig, m.unit_kinds[k], len(m.unit_atoms[k]))
                acc.setdefault(tid, []).append(m.desc[k])
        self.type_means = {t: np.mean(np.stack(v), axis=0) for t, v in acc.items()}
        self.global_mean = np.mean(np.concatenate([m.desc for m in mols], axis=0), axis=0)

        all_rel = np.concatenate([m.rel_feat for m in mols if len(m.rel_pairs)], axis=0)
        self.rel_mean = all_rel.mean(axis=0) if len(all_rel) else np.zeros(REL_DIM, dtype=np.float32)

        self.y_mean = float(ys[mol_indices(mols)].mean())
        self.y_std = float(ys[mol_indices(mols)].std() + 1e-12)
        return self

    def center_desc(self, tid: int, desc_row: np.ndarray) -> np.ndarray:
        mu = self.type_means.get(tid, self.global_mean)
        return desc_row - mu

    def type_id(self, mol: MolUnits, k: int) -> int:
        return self.vocab.to_id(mol.unit_sigs[k], mol.unit_kinds[k], len(mol.unit_atoms[k]))


def mol_indices(mols: list[MolUnits]) -> np.ndarray:
    return np.array([m.mol_index for m in mols], dtype=np.int64)


def y_only_targets(raw_y: np.ndarray) -> np.ndarray:
    """Guard: supervision is the raw ``y`` column and nothing else.

    Any future auxiliary target must extend this function explicitly and
    update the protocol; tests assert the output is exactly the raw y.
    """
    if raw_y.ndim != 2 or raw_y.shape[1] != 1:
        raise ValueError(f"y must be [N,1] raw targets, got {raw_y.shape}")
    return raw_y[:, 0].copy()
