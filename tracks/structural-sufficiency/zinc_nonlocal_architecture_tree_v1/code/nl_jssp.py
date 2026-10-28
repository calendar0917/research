"""Joint-Span Structural Pretraining (JSSP) mask mechanism.

Only the mask sampler differs from the frozen masked atom/bond SSL task: a
training example hides one or more contiguous 2-bond spans of the real graph
(their atoms and physical bonds) instead of independent 15%/15% atoms/bonds.
Every derived object feature is rebuilt from the masked raw graph with the
frozen ``structural_relation_objects_v1.build_mol_objects``.  Hidden truth is
kept outside the model input, exactly as in the frozen code.

The span budget is fixed before training: ``max(1, round(0.05 * n_atoms))``
spans, each covering up to 3 atoms and 2 physical bonds.  This matches the
frozen atom mask budget (~15% atoms) and deliberately does not search ratios.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[3]
SC = REPO / "tracks/structure-contribution/code"
for p in (str(SC), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

SPAN_ATOM_RATIO = 0.05          # spans per atom; each span ~3 atoms
MASK_STREAM_TAG = 20261029      # independent from the old independent-mask RNG


def _adjacency(n_atoms: int, bond_index: np.ndarray):
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for e, (u, v) in enumerate(np.asarray(bond_index, dtype=np.int64)):
        adj[int(u)].append((int(v), e))
        adj[int(v)].append((int(u), e))
    return adj


def enumerate_spans(bond_index: np.ndarray) -> list[tuple[int, int]]:
    """All unordered 2-bond paths (pairs of distinct bonds sharing one atom)."""
    bi = np.asarray(bond_index, dtype=np.int64)
    n = int(bi.max()) + 1 if len(bi) else 0
    adj = _adjacency(n, bi)
    out: set[tuple[int, int]] = set()
    for e, (u, v) in enumerate(bi):
        for w, e2 in adj[int(v)]:
            if w == int(u) or e2 == e:
                continue
            out.add((min(e, e2), max(e, e2)))
        for w, e2 in adj[int(u)]:
            if w == int(v) or e2 == e:
                continue
            out.add((min(e, e2), max(e, e2)))
    return sorted(out)


def sample_joint_span_mask(seed: int, mol_id: int, visit_index: int, n_atoms: int,
                           bond_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bi = np.asarray(bond_index, dtype=np.int64)
    m = len(bi)
    if m == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(mol_id), int(visit_index), MASK_STREAM_TAG])
    )
    k = max(1, int(round(SPAN_ATOM_RATIO * int(n_atoms))))
    spans = enumerate_spans(bi)
    chosen: list[tuple[int, int]] = []
    if spans:
        if k <= len(spans):
            pick = rng.choice(len(spans), size=k, replace=False)
        else:
            pick = rng.choice(len(spans), size=k, replace=True)
        chosen = [spans[int(i)] for i in pick]
    else:
        # one-bond molecule: a degenerate span hides that bond and its atoms
        e = int(rng.integers(0, m))
        chosen = [(e, e)]

    atoms: set[int] = set()
    bonds: set[int] = set()
    for e1, e2 in chosen:
        bonds.add(int(e1)); bonds.add(int(e2))
        atoms.add(int(bi[e1, 0])); atoms.add(int(bi[e1, 1]))
        atoms.add(int(bi[e2, 0])); atoms.add(int(bi[e2, 1]))
    return (np.asarray(sorted(atoms), dtype=np.int64),
            np.asarray(sorted(bonds), dtype=np.int64))


def build_masked_sample(base: sro.MolObjects, x: np.ndarray, edge_index: np.ndarray,
                        edge_attr: np.ndarray, *, seed: int, visit_index: int) -> mp.PretextSample:
    bond_index = np.asarray(base.bond_index, dtype=np.int64)
    atom_ids, bond_ids = sample_joint_span_mask(
        seed, base.mol_index, visit_index, base.n_atoms, bond_index
    )
    x_m, ea_m = mp.mask_raw_arrays(x, edge_index, edge_attr, bond_index, atom_ids, bond_ids)
    masked = sro.build_mol_objects(base.mol_index, x_m, edge_index, ea_m, smiles="")
    xf = np.asarray(x).reshape(-1)
    atom_types = np.asarray([int(xf[i]) for i in atom_ids], dtype=np.int64)
    bond_types = np.asarray([int(base.bond_type[i]) for i in bond_ids], dtype=np.int64)
    return mp.PretextSample(masked=masked, atom_ids=atom_ids, atom_types=atom_types,
                            bond_ids=bond_ids, bond_types=bond_types)


def mask_stats(bases, raw, n_mol: int = 200, seed: int = 0, visit_index: int = 0) -> dict:
    """Label-free coverage / parity statistics vs the frozen independent mask."""
    old_a, old_b, new_a, new_b = [], [], [], []
    for j in range(min(n_mol, len(bases))):
        o = bases[j]
        d = raw[int(o.mol_index)]
        a_ids, b_ids = mp.sample_mask(seed, o.mol_index, visit_index, o.n_atoms,
                                      len(o.bond_index))
        old_a.append(len(a_ids) / max(o.n_atoms, 1))
        old_b.append(len(b_ids) / max(len(o.bond_index), 1))
        a2, b2 = sample_joint_span_mask(seed, o.mol_index, visit_index, o.n_atoms,
                                       o.bond_index)
        new_a.append(len(a2) / max(o.n_atoms, 1))
        new_b.append(len(b2) / max(len(o.bond_index), 1))
    return {
        "n_molecules": len(old_a),
        "old_atom_fraction_mean": float(np.mean(old_a)),
        "old_bond_fraction_mean": float(np.mean(old_b)),
        "span_atom_fraction_mean": float(np.mean(new_a)),
        "span_bond_fraction_mean": float(np.mean(new_b)),
        "old_atom_fraction_min": float(np.min(old_a)),
        "span_atom_fraction_min": float(np.min(new_a)),
    }


__all__ = ["SPAN_ATOM_RATIO", "enumerate_spans", "sample_joint_span_mask",
           "build_masked_sample", "mask_stats"]
