"""Control A: matched random connected subgraphs from the *same* molecule.

Frozen rules (protocol §6):

* same original molecule;
* exactly as many physical bonds as the Top-1 discovery subgraph;
* not identical to the Top-1 bond set;
* the number of S2 occurrences it touches is within a 25% relative difference
  (denominator ``max(1, n_s2_top1)``);
* RNG seed :data:`as_common.CONTROL_SEED`, at most 512 growth attempts, accepted
  candidates taken in deterministic generation order, first 16;
* fewer than 8 acceptable candidates -> ``INSUFFICIENT_MATCHED_NULL`` and the
  matched comparison for that molecule is reported as missing.

A key-count-only random set is recorded separately and is **never** mixed into
the primary matched statistic.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import as_common as C
import as_gate as G


def grow_connected_subgraph(mol: G.SingleMolecule, k: int, rng: np.random.Generator) -> tuple[int, ...] | None:
    """Grow a uniformly-chosen random connected subgraph with exactly ``k`` bonds."""
    k = int(k)
    if k <= 0 or k > mol.n_bonds:
        return None
    degree = np.zeros(mol.n_atoms, dtype=np.int64)
    np.add.at(degree, mol.bond_u, 1)
    np.add.at(degree, mol.bond_v, 1)
    candidates = np.where(degree > 0)[0]
    if candidates.size == 0:
        return None
    start = int(rng.choice(candidates))
    covered = {start}
    chosen: list[int] = []
    while len(chosen) < k:
        frontier = [
            e
            for e in range(mol.n_bonds)
            if e not in chosen and (int(mol.bond_u[e]) in covered or int(mol.bond_v[e]) in covered)
        ]
        if not frontier:
            return None
        pick = int(rng.choice(np.asarray(frontier, dtype=np.int64)))
        chosen.append(pick)
        covered.add(int(mol.bond_u[pick]))
        covered.add(int(mol.bond_v[pick]))
    return tuple(sorted(chosen))


def matched_random_controls(
    mol: G.SingleMolecule,
    top1_bonds: Sequence[int],
    rng: np.random.Generator,
    *,
    n_target: int = C.N_MATCHED_RANDOM,
    max_attempts: int = C.MAX_GROWTH_ATTEMPTS,
    s2_rel_tol: float = C.S2_MATCH_REL_TOL,
    n_secondary: int = C.N_MATCHED_RANDOM,
) -> dict[str, Any]:
    """Matched random connected subgraphs for one molecule's Top-1 subgraph."""
    top1 = tuple(sorted(int(b) for b in top1_bonds))
    k = len(top1)
    target_s2 = mol.s2_count(top1)
    target_s2_either = mol.s2_count_either(top1)
    denom = max(1, target_s2)

    matched: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = {top1}
    attempts = 0
    exhausted = False
    while attempts < int(max_attempts) and len(matched) < int(n_target):
        attempts += 1
        cand = grow_connected_subgraph(mol, k, rng)
        if cand is None:
            exhausted = True
            break
        if cand in seen:
            continue
        seen.add(cand)
        if not mol.atoms_connected(cand):        # growth guarantees this; assert anyway
            continue
        s2 = mol.s2_count(cand)
        rel = abs(s2 - target_s2) / denom
        rec = {
            "bonds": list(cand),
            "atoms": mol.bond_atoms(cand),
            "n_bonds": len(cand),
            "n_s2_occurrences": int(s2),
            "s2_rel_diff": float(rel),
        }
        if rel <= float(s2_rel_tol) + 1e-12:
            matched.append(rec)
        elif len(secondary) < int(n_secondary):
            secondary.append(rec)

    status = "OK" if len(matched) >= int(C.MIN_MATCHED_RANDOM) else "INSUFFICIENT_MATCHED_NULL"
    return {
        "status": status,
        "rule": {
            "same_molecule": True,
            "same_bond_count": k,
            "exclude_identical_to_top1": True,
            "s2_occurrence_rel_tol": float(s2_rel_tol),
            "s2_denominator": "max(1, n_s2_top1)",
            "n_target": int(n_target),
            "min_acceptable": int(C.MIN_MATCHED_RANDOM),
            "max_growth_attempts": int(max_attempts),
            "seed": int(C.CONTROL_SEED),
        },
        "top1": {
            "bonds": list(top1),
            "n_bonds": k,
            "n_s2_occurrences": int(target_s2),
            "n_s2_occurrences_either_bond": int(target_s2_either),
        },
        "attempts": int(attempts),
        "growth_exhausted": bool(exhausted),
        "n_matched": len(matched),
        "matched": matched,
        "n_secondary_keycount_only": len(secondary),
        "secondary_keycount_only": secondary,
        "note": (
            "key-count-only random sets are a secondary diagnostic and never enter "
            "the primary matched statistic"
        ),
    }


__all__ = ["grow_connected_subgraph", "matched_random_controls"]
