"""Candidate generation, the shared candidate pool, selector scores and controls.

Label-blind by construction: every function in this module receives only the
molecule's physical-bond graph, the frozen model's gradient / local-output
fields, or an S2 occurrence count.  No label, edit region, pair partner or
oracle contribution is an argument, and the label-blind caller test in
``tests/`` checks that shuffling them leaves the pool byte-identical.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import lfa_common as C


def _mods():
    C.install_source_paths()
    return C.source_modules()


def rank_key(score: float, bonds: Sequence[int]) -> tuple[float, int, tuple[int, ...]]:
    """Frozen tie-break: score descending, fewer bonds, sorted bond-id tuple."""
    return (-float(score), len(bonds), tuple(sorted(int(b) for b in bonds)))


def topo_score(bonds: Sequence[int], mol) -> float:
    """``n_s2_affected(S)``: S2 occurrences with at least one selected bond."""
    return float(mol.s2_count_either([int(b) for b in bonds]))


def score_functions(mol, fields: Mapping[str, Any]) -> dict[str, Callable[[Sequence[int]], float]]:
    AS = _mods()["as_search"]
    grad_l = np.asarray(fields["lcrc_grad"], dtype=np.float64)
    grad_g = np.asarray(fields["gine_grad"], dtype=np.float64)
    local = np.abs(np.asarray(fields["lcrc_local"], dtype=np.float64))
    return {
        "LCRC_GRAD": lambda bonds: AS.score_discovery(bonds, grad_l),
        "LCRC_LOCAL": lambda bonds: AS.score_local_heuristic(bonds, local, mol),
        "GINE_GRAD": lambda bonds: AS.score_discovery(bonds, grad_g),
        "TOPO_S2": lambda bonds: topo_score(bonds, mol),
    }


def generate_families(mol, fields: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Run the inherited frozen search once per generator.

    k=2 enumerates every pair of bonds sharing an endpoint; k=3..6 expand every
    retained level-(k-1) candidate by one adjacent physical bond and keep at
    most ``BEAM_WIDTH`` per level.  The generator's own candidate set is the
    union of those retained level beams.
    """
    AS = _mods()["as_search"]
    out: dict[str, dict[str, Any]] = {}
    for name, fn in score_functions(mol, fields).items():
        res = AS.search_subgraphs(mol, fn, score_name=name, keep_log=False)
        cands: dict[tuple[int, ...], float] = {}
        for k in range(C.SEARCH_MIN_K, C.SEARCH_MAX_K + 1):
            for cand in res.levels.get(k, []):
                key = tuple(sorted(int(b) for b in cand.bonds))
                cands[key] = float(cand.score)
        out[name] = {"scores": cands, "n_evaluated": int(res.n_unique_evaluated)}
    return out


def shared_pool(families: Mapping[str, Mapping[str, Any]]) -> list[tuple[int, ...]]:
    """Union of the four generators' retained candidates, canonically sorted."""
    keys: set[tuple[int, ...]] = set()
    for fam in families.values():
        keys.update(fam["scores"].keys())
    return sorted(keys, key=lambda b: (len(b), b))


def candidate_atoms(mol, bonds: Sequence[int]) -> list[int]:
    return [int(v) for v in mol.bond_atoms([int(b) for b in bonds])]


def s2_counts(mol, bonds: Sequence[int]) -> dict[str, int]:
    internal = int(mol.s2_count([int(b) for b in bonds]))
    affected = int(mol.s2_count_either([int(b) for b in bonds]))
    return {
        "n_s2_internal": internal,
        "n_s2_boundary": int(affected - internal),
        "n_s2_affected": affected,
        "n_s1_codes": int(mol.s1_count([int(b) for b in bonds])),
    }


def select_top1(pool: Sequence[tuple[int, ...]], scores: Sequence[float]) -> int | None:
    order = sorted(range(len(pool)), key=lambda i: rank_key(float(scores[i]), pool[i]))
    return int(order[0]) if order else None


# ---------------------------------------------------------------------------
# matched random connected controls
# ---------------------------------------------------------------------------


def _incidence(mol) -> list[list[int]]:
    inc: list[list[int]] = [[] for _ in range(mol.n_atoms)]
    for e in range(mol.n_bonds):
        inc[int(mol.bond_u[e])].append(e)
        inc[int(mol.bond_v[e])].append(e)
    return inc


def grow_matched(mol, k: int, target_atoms: int, rng: np.random.Generator,
                 inc: Sequence[Sequence[int]] | None = None) -> tuple[int, ...] | None:
    """Grow one connected physical-bond set with exactly ``k`` bonds and
    ``target_atoms`` involved atoms.

    A connected k-bond subgraph on ``a`` atoms has exactly ``k - a + 1``
    independent cycle closures, so the growth keeps that count in view instead
    of hoping a cycle appears: this only changes the *proposal* distribution,
    never the acceptance rule (same bond count, same atom count, connected,
    not identical to the Top-1).
    """
    k, target_atoms = int(k), int(target_atoms)
    if k <= 0 or target_atoms <= 0 or target_atoms > k + 1:
        return None
    inc = _incidence(mol) if inc is None else inc
    starts = [v for v in range(mol.n_atoms) if inc[v]]
    if not starts:
        return None
    covered = {int(rng.choice(np.asarray(starts, dtype=np.int64)))}
    chosen: list[int] = []
    closes_needed = k - target_atoms + 1
    closes_done = 0
    while len(chosen) < k:
        remaining = k - len(chosen)
        ext_needed = target_atoms - len(covered)
        closes_left = closes_needed - closes_done
        if ext_needed < 0 or ext_needed > remaining or closes_left < 0 or closes_left > remaining:
            return None
        closing: list[int] = []
        extending: list[int] = []
        for atom in covered:
            for e in inc[atom]:
                if e in chosen:
                    continue
                u, v = int(mol.bond_u[e]), int(mol.bond_v[e])
                if u in covered and v in covered:
                    if e not in closing:
                        closing.append(e)
                elif u in covered or v in covered:
                    if e not in extending:
                        extending.append(e)
        if not closing and not extending:
            return None
        if ext_needed == 0:
            pick_from = closing
        elif closes_left == 0 or not closing:
            pick_from = extending
        elif remaining == closes_left:
            pick_from = closing
        elif remaining == ext_needed:
            pick_from = extending
        else:
            pick_from = closing + extending
        if not pick_from:
            # dead end: the remaining budget cannot be spent without either
            # adding an atom that the target size forbids or closing a cycle
            # that does not exist.  ``None`` is a rejected proposal, not a
            # relaxation of the matching rule.
            return None
        e = int(rng.choice(np.asarray(pick_from, dtype=np.int64)))
        u, v = int(mol.bond_u[e]), int(mol.bond_v[e])
        before = len(covered)
        covered.update((u, v))
        if len(covered) == before:
            closes_done += 1
        chosen.append(e)
    if len(covered) != target_atoms:
        return None
    return tuple(sorted(chosen))


def matched_controls(mol, top1_bonds: Sequence[int], *, seed: Sequence[int],
                     n_target: int = C.N_RANDOM, min_acceptable: int = C.MIN_RANDOM,
                     max_attempts: int = C.MAX_GROWTH_ATTEMPTS) -> dict[str, Any]:
    """Same-molecule, same-bond-count, same-atom-count random connected sets."""
    top1 = tuple(sorted(int(b) for b in top1_bonds))
    k = len(top1)
    target_atoms = len(mol.bond_atoms(list(top1)))
    rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
    inc = _incidence(mol)
    accepted: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = {top1}
    attempts = 0
    while attempts < int(max_attempts) and len(accepted) < int(n_target):
        attempts += 1
        cand = grow_matched(mol, k, target_atoms, rng, inc)
        if cand is None or cand in seen:
            continue
        seen.add(cand)
        if not mol.atoms_connected(cand):
            continue
        accepted.append(cand)
    status = "OK" if len(accepted) >= int(min_acceptable) else "INSUFFICIENT_RANDOM"
    return {
        "status": status,
        "n_target": int(n_target),
        "min_acceptable": int(min_acceptable),
        "n_random": len(accepted),
        "attempts": int(attempts),
        "k": k,
        "n_atoms": target_atoms,
        "seed": [int(s) for s in seed],
        "controls": [
            {"bonds": [int(b) for b in cand], "n_bonds": len(cand),
             "n_atoms": len(mol.bond_atoms(list(cand)))}
            for cand in accepted
        ],
    }


def control_seed(split: str, row_id: int, selector: str) -> list[int]:
    return [C.CONTROL_SEED, C.SPLIT_CODE[str(split)], int(row_id), C.SELECTORS.index(selector) + 1]


__all__ = [
    "rank_key", "topo_score", "score_functions", "generate_families", "shared_pool",
    "candidate_atoms", "s2_counts", "select_top1", "grow_matched", "matched_controls",
    "control_seed",
]
