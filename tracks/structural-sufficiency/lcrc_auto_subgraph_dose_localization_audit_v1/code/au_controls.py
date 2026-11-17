"""Corrected random null: matched on the *affected* S2 dose, not the internal one.

Frozen corrected rule (protocol §3):

* same molecule;
* exactly as many physical bonds as the pilot's Top-1 subgraph (the bond set is
  reused verbatim, never re-selected);
* not identical to the Top-1 bond set;
* ``|n_s2_affected(candidate) - n_s2_affected(Top-1)| / max(1, n_s2_affected(Top-1)) <= 0.25``;
* RNG seed 20261117 with the pilot's per-molecule stream derivation;
* at most 512 growth attempts;
* the first 16 accepted, distinct candidates, in generation order;
* fewer than 8 accepted candidates -> ``INSUFFICIENT_MATCHED_NULL`` (no backfill,
  no relaxation, no substitution).

The connected-subgraph *generator* is the pilot's own
``grow_connected_subgraph``; only the acceptance criterion changes.  Because the
generator consumes randomness independently of whether a draw is accepted, the
two rounds see the same underlying draw sequence for a molecule, which isolates
"criterion changed" from "random draws changed".
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import au_common as C
import au_s2 as S2

import as_controls as OC          # read-only reuse of the pilot's growth generator
import as_gate as G

MATCH_REASONS: dict[int, str] = {0: "accepted", 1: "affected_rel_diff", 2: "duplicate"}


def _atomic_record(mol: G.SingleMolecule, bonds: Sequence[int]) -> dict[str, Any]:
    acc = S2.dose_accounting(mol, bonds)
    rec = acc.record()
    return {
        "bonds": [int(b) for b in bonds],
        "atoms": mol.bond_atoms(bonds),
        "n_bonds": int(acc.n_bonds),
        **rec,
    }


def matched_random_controls(
    mol: G.SingleMolecule,
    top1_bonds: Sequence[int],
    rng: np.random.Generator,
    *,
    n_target: int = C.N_MATCHED_RANDOM,
    max_attempts: int = C.MAX_GROWTH_ATTEMPTS,
    rel_tol: float = C.S2_AFFECTED_REL_TOL,
) -> dict[str, Any]:
    """Affected-dose-matched random connected subgraphs for one molecule."""
    top1 = tuple(sorted(int(b) for b in top1_bonds))
    if not top1:
        raise ValueError("the corrected null needs a non-empty Top-1 bond set")
    k = len(top1)
    target = S2.dose_accounting(mol, top1)
    denom = max(1, int(target.n_s2_affected))
    denom_internal = max(1, int(target.n_s2_internal))

    matched: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = {top1}
    attempts = 0
    n_duplicates = 0
    exhausted = False
    while attempts < int(max_attempts) and len(matched) < int(n_target):
        attempts += 1
        cand = OC.grow_connected_subgraph(mol, k, rng)
        if cand is None:
            exhausted = True
            break
        if tuple(cand) in seen:
            n_duplicates += 1
            ledger.append({"bonds": list(cand), "reason": 2, "aff_bonds": 0, "affected": 0, "affected_rel": float("nan")})
            continue
        seen.add(tuple(cand))
        acc = S2.dose_accounting(mol, cand)
        rel = abs(int(acc.n_s2_affected) - int(target.n_s2_affected)) / denom
        rec = {
            "bonds": list(cand),
            "atoms": mol.bond_atoms(cand),
            "n_bonds": int(acc.n_bonds),
            "n_s2_internal": int(acc.n_s2_internal),
            "n_s2_boundary": int(acc.n_s2_boundary),
            "n_s2_affected": int(acc.n_s2_affected),
            "affected_rel_diff": float(rel),
            "internal_rel_diff": abs(int(acc.n_s2_internal) - int(target.n_s2_internal)) / denom_internal,
            "boundary_over_affected": acc.boundary_over_affected,
            "affected_over_total": acc.affected_over_total,
            "affected_root_outside_fraction": acc.affected_root_outside_fraction,
            "accepted": bool(rel <= float(rel_tol) + 1e-12),
            "kind": "matched_random",
        }
        ledger.append(
            {
                "bonds": list(cand),
                "reason": 0 if rec["accepted"] else 1,
                "aff_bonds": int(acc.n_s2_internal),
                "affected": int(acc.n_s2_affected),
                "affected_rel": float(rel),
            }
        )
        if rec["accepted"]:
            rec["index"] = len(matched)
            matched.append(rec)

    status = "OK" if len(matched) >= int(C.MIN_MATCHED_RANDOM) else "INSUFFICIENT_MATCHED_NULL"
    return {
        "status": status,
        "rule": {
            "match_quantity": "n_s2_affected",
            "same_molecule": True,
            "same_bond_count": int(k),
            "exclude_identical_to_top1": True,
            "rel_tol": float(rel_tol),
            "denominator": "max(1, n_s2_affected(top1))",
            "n_target": int(n_target),
            "min_acceptable": int(C.MIN_MATCHED_RANDOM),
            "max_growth_attempts": int(max_attempts),
            "seed": int(C.CONTROL_SEED),
            "rng": "numpy SeedSequence([20261117, dataset_index, split_index, row_id])",
        },
        "top1": _atomic_record(mol, top1),
        "attempts": int(attempts),
        "growth_exhausted": bool(exhausted),
        "n_matched": len(matched),
        "n_duplicate_draws": int(n_duplicates),
        "n_candidates": int(len(ledger)),
        "matched": matched,
        "ledger": ledger,
    }


def old_control_dose(
    mol: G.SingleMolecule,
    top1_bonds: Sequence[int],
    old_control_bonds: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Re-account the pilot's already accepted controls on the corrected quantity.

    The pilot's bond sets are read verbatim; nothing is re-drawn or re-ranked.  This
    is the measurement behind "the pilot matched the wrong quantity", and it never
    replaces the pilot's stored effects.
    """
    top1 = [int(b) for b in top1_bonds]
    target = S2.dose_accounting(mol, top1)
    denom = max(1, int(target.n_s2_affected))
    denom_internal = max(1, int(target.n_s2_internal))
    out: list[dict[str, Any]] = []
    for i, bonds in enumerate(old_control_bonds):
        acc = S2.dose_accounting(mol, bonds)
        out.append(
            {
                "index": int(i),
                "bonds": [int(b) for b in bonds],
                "n_bonds": int(acc.n_bonds),
                "n_s2_internal": int(acc.n_s2_internal),
                "n_s2_boundary": int(acc.n_s2_boundary),
                "n_s2_affected": int(acc.n_s2_affected),
                "internal_rel_diff": abs(int(acc.n_s2_internal) - int(target.n_s2_internal)) / denom_internal,
                "affected_rel_diff": abs(int(acc.n_s2_affected) - int(target.n_s2_affected)) / denom,
                "affected_minus_internal": int(acc.n_s2_affected - acc.n_s2_internal),
                "top1_n_s2_internal": int(target.n_s2_internal),
                "top1_n_s2_affected": int(target.n_s2_affected),
                "would_pass_affected_tolerance": bool(
                    abs(int(acc.n_s2_affected) - int(target.n_s2_affected)) / denom <= C.S2_AFFECTED_REL_TOL
                ),
            }
        )
    return out


def ledger_arrays(entries: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Compact ``.npz`` encoding of every generated candidate of one split."""
    mol_index = np.asarray([int(e["mol_index"]) for e in entries], dtype=np.int32)
    reason = np.asarray([int(e["reason"]) for e in entries], dtype=np.int8)
    k = np.asarray([len(e["bonds"]) for e in entries], dtype=np.int8)
    internal = np.asarray([int(e["aff_bonds"]) for e in entries], dtype=np.int32)
    affected = np.asarray([int(e["affected"]) for e in entries], dtype=np.int32)
    rel = np.asarray([float(e["affected_rel"]) for e in entries], dtype=np.float64)
    ptr = np.concatenate([[0], np.cumsum(k)]).astype(np.int32) if entries else np.zeros(1, dtype=np.int32)
    flat = np.asarray([int(b) for e in entries for b in e["bonds"]], dtype=np.int16)
    return {
        "mol_index": mol_index,
        "reason": reason,
        "k": k,
        "n_s2_internal": internal,
        "n_s2_affected": affected,
        "affected_rel_diff": rel,
        "bond_ptr": ptr,
        "bonds": flat,
    }


def decode_ledger(payload: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    flat = np.asarray(payload["bonds"], dtype=np.int64)
    out = []
    for i in range(int(ptr.size) - 1):
        out.append(
            {
                "mol_index": int(payload["mol_index"][i]),
                "reason": MATCH_REASONS.get(int(payload["reason"][i]), "unknown"),
                "bonds": flat[ptr[i]:ptr[i + 1]].tolist(),
                "n_s2_internal": int(payload["n_s2_internal"][i]),
                "n_s2_affected": int(payload["n_s2_affected"][i]),
                "affected_rel_diff": float(payload["affected_rel_diff"][i]),
            }
        )
    return out


__all__ = [
    "MATCH_REASONS",
    "matched_random_controls",
    "old_control_dose",
    "ledger_arrays",
    "decode_ledger",
]
