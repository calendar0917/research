"""Correct S2 dose accounting for the ``CONDITIONAL_S1_S2_EVIDENCE_MASK``.

The pilot matched its random null on ``n_s2_internal(S)``:

    e1 in S AND e2 in S

That is the number of non-backtracking 2-path occurrences *inside* the subgraph.
It is **not** the number of occurrences the frozen gate actually removes.  The
S2 weight is

    w[p] = m[e1] * m[e2]

so an occurrence disappears as soon as *either* of its two physical bonds is
masked.  The quantity the intervention acts on is therefore

    n_s2_affected(S) = #{ p : e1 in S OR e2 in S }
                     = n_s2_internal(S) + n_s2_boundary(S)

with

    n_s2_boundary(S) = #{ p : exactly one of e1, e2 in S }.

This module computes all three, splits the boundary occurrences by which of the
two bonds is masked (which decides whether the occurrence's *root atom* is
inside the highlighted subgraph), and proves on the real frozen function that
``n_s2_affected`` is exactly the number of occurrences whose weight changes from
1 to 0 — by counting the zeroed weights and by comparing the masked forward with
a forward in which precisely those occurrences were deleted from the path list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import au_common as C

import as_data as D
import as_gate as G


def _ids(bonds: Sequence[int]) -> np.ndarray:
    return np.asarray([int(b) for b in bonds], dtype=np.int64).reshape(-1)


@dataclass(frozen=True)
class DoseAccounting:
    """Every S1/S2 quantity touched by masking one physical bond set ``S``."""

    n_bonds: int
    n_atoms_inside: int
    n_atoms_outside: int
    n_s2_total: int
    n_s2_internal: int
    n_s2_boundary: int
    n_s2_affected: int
    n_s2_unaffected: int
    #: boundary occurrences whose *first* bond is masked (their root is an endpoint
    #: of a masked bond, i.e. inside the highlighted atoms) vs whose second bond is
    #: masked (root can be outside the highlighted atoms).
    n_s2_boundary_first_masked: int
    n_s2_boundary_second_masked: int
    #: affected occurrences grouped by whether the occurrence's root atom is one of
    #: the highlighted atoms.  This is the mechanism behind an outside response.
    n_s2_affected_root_inside: int
    n_s2_affected_root_outside: int
    n_s1_codes: int

    # -- descriptive ratios (never used as a hidden selection rule) ----------
    @property
    def boundary_over_affected(self) -> float:
        return float(self.n_s2_boundary) / float(self.n_s2_affected) if self.n_s2_affected else 0.0

    @property
    def affected_over_total(self) -> float:
        return float(self.n_s2_affected) / float(self.n_s2_total) if self.n_s2_total else 0.0

    @property
    def internal_over_total(self) -> float:
        return float(self.n_s2_internal) / float(self.n_s2_total) if self.n_s2_total else 0.0

    @property
    def boundary_over_total(self) -> float:
        return float(self.n_s2_boundary) / float(self.n_s2_total) if self.n_s2_total else 0.0

    @property
    def affected_root_outside_fraction(self) -> float:
        return (
            float(self.n_s2_affected_root_outside) / float(self.n_s2_affected)
            if self.n_s2_affected else 0.0
        )

    @property
    def affected_over_internal(self) -> float:
        return float(self.n_s2_affected) / float(self.n_s2_internal) if self.n_s2_internal else float("nan")

    def record(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(
            {
                "boundary_over_affected": self.boundary_over_affected,
                "affected_over_total": self.affected_over_total,
                "internal_over_total": self.internal_over_total,
                "boundary_over_total": self.boundary_over_total,
                "affected_root_outside_fraction": self.affected_root_outside_fraction,
                "affected_over_internal": self.affected_over_internal,
                "identity_internal_plus_boundary_eq_affected": bool(
                    self.n_s2_internal + self.n_s2_boundary == self.n_s2_affected
                ),
                "identity_affected_plus_unaffected_eq_total": bool(
                    self.n_s2_affected + self.n_s2_unaffected == self.n_s2_total
                ),
            }
        )
        return out


def s2_partition(occ: Any, bonds: Sequence[int]) -> dict[str, np.ndarray]:
    """Boolean occurrence masks for one bond set (molecule-local ``OccMap``)."""
    ids = _ids(bonds)
    b1 = np.asarray(occ.path_bond1, dtype=np.int64)
    b2 = np.asarray(occ.path_bond2, dtype=np.int64)
    if b1.size == 0:
        empty = np.zeros(0, dtype=bool)
        return {
            "first_masked": empty, "second_masked": empty, "internal": empty,
            "boundary": empty, "affected": empty, "unaffected": empty,
            "boundary_first_only": empty, "boundary_second_only": empty,
        }
    in1 = np.isin(b1, ids)
    in2 = np.isin(b2, ids)
    internal = in1 & in2
    boundary = in1 ^ in2
    return {
        "first_masked": in1,
        "second_masked": in2,
        "internal": internal,
        "boundary": boundary,
        "affected": in1 | in2,
        "unaffected": ~(in1 | in2),
        "boundary_first_only": in1 & ~in2,
        "boundary_second_only": ~in1 & in2,
    }


def dose_accounting(mol: G.SingleMolecule, bonds: Sequence[int]) -> DoseAccounting:
    """All three S2 counts, the boundary split and the root split for one bond set."""
    bonds = [int(b) for b in bonds]
    if not bonds:
        raise ValueError("dose accounting needs a non-empty bond set")
    part = s2_partition(mol.occ, bonds)
    inside_atoms = np.zeros(mol.n_atoms, dtype=bool)
    inside_atoms[np.asarray(mol.bond_atoms(bonds), dtype=np.int64)] = True
    root = np.asarray(mol.occ.path_root, dtype=np.int64)
    affected = part["affected"]
    root_inside = inside_atoms[root] if root.size else np.zeros(0, dtype=bool)
    n_s1 = int(np.isin(np.asarray(mol.occ.nbr_bond, dtype=np.int64), _ids(bonds)).sum())
    n_inside = int(inside_atoms.sum())
    return DoseAccounting(
        n_bonds=len(bonds),
        n_atoms_inside=n_inside,
        n_atoms_outside=int(mol.n_atoms - n_inside),
        n_s2_total=int(affected.size),
        n_s2_internal=int(part["internal"].sum()),
        n_s2_boundary=int(part["boundary"].sum()),
        n_s2_affected=int(affected.sum()),
        n_s2_unaffected=int(part["unaffected"].sum()),
        n_s2_boundary_first_masked=int(part["boundary_first_only"].sum()),
        n_s2_boundary_second_masked=int(part["boundary_second_only"].sum()),
        n_s2_affected_root_inside=int((affected & root_inside).sum()),
        n_s2_affected_root_outside=int((affected & ~root_inside).sum()),
        n_s1_codes=n_s1,
    )


def s2_weight_audit(gb: D.GatedBatch, mask: np.ndarray, accounting: DoseAccounting) -> dict[str, Any]:
    """Count the S2 weights that the real gate turns from 1 into 0.

    This reads the *same* ``gb.s2_weight`` tensor the frozen forward consumes, so
    a mismatch against :func:`dose_accounting` is a real accounting error and not
    a re-implementation artefact.
    """
    M = gb.gate_matrix([mask])
    with torch.no_grad():
        w = gb.s2_weight(M).detach().cpu().numpy().astype(np.float64)
    zeroed = int(np.sum(w == 0.0))
    kept = int(np.sum(w == 1.0))
    fractional = int(w.size - zeroed - kept)
    return {
        "n_s2_occurrences_in_batch": int(w.size),
        "n_weight_zeroed_by_gate": zeroed,
        "n_weight_kept": kept,
        "n_weight_fractional": fractional,
        "accounting_n_s2_affected": int(accounting.n_s2_affected),
        "accounting_n_s2_unaffected": int(accounting.n_s2_unaffected),
        "count_matches_accounting": bool(
            zeroed == int(accounting.n_s2_affected) and kept == int(accounting.n_s2_unaffected)
        ),
        "passed": bool(
            zeroed == int(accounting.n_s2_affected)
            and kept == int(accounting.n_s2_unaffected)
            and fractional == 0
        ),
    }


@torch.no_grad()
def gate_drop_equivalence(
    arm: torch.nn.Module,
    cache: Mapping[str, Any],
    row: int,
    split_occ: D.SplitOcc,
    mol: G.SingleMolecule,
    bonds: Sequence[int],
    accounting: DoseAccounting,
    *,
    sigma: float,
    mu: float,
) -> dict[str, Any]:
    """S2 mask == deleting exactly the affected occurrences, on the frozen function.

    The S1 side is left fully open, so this isolates the S2 arithmetic:
    ``m[e]=0`` on ``S`` must be *identical* to running the unmodified model on the
    same molecule with every occurrence ``e1 in S OR e2 in S`` removed from the
    path list.  An occurrence with both bonds masked must therefore be removed
    once, not twice, and an unaffected occurrence must not move at all.
    """
    n = int(mol.n_bonds)
    gb = D.collate_gated(cache, [int(row)], split_occ)
    ones = gb.gate_ones()
    s2_mask = G.bond_mask(n, bonds, 0.0)
    parts_open = G.gated_parts(arm, gb, ones)
    parts_masked = G.gated_parts(arm, gb, ones, gb.gate_matrix([s2_mask]))
    p_open = float(parts_open["pred_std"].detach().cpu().numpy().astype(np.float64)[0])
    p_masked = float(parts_masked["pred_std"].detach().cpu().numpy().astype(np.float64)[0])

    part = s2_partition(mol.occ, bonds)
    dropped = part["affected"] if part["affected"].size else np.zeros(0, dtype=bool)
    mol_arrays = {
        "atom_feat": mol.mol["atom_feat"],
        "atom_type": mol.mol["atom_type"],
        "atom_scalar": mol.mol["atom_scalar"],
        "bond_feat": mol.mol["bond_feat"],
        "bond_type": mol.mol["bond_type"],
        "bond_index": mol.mol["bond_index"],
        "global": mol.mol["global"],
        "smiles": mol.mol["smiles"],
        "molecule_id": mol.mol["molecule_id"],
    }
    cache2, occ2 = D.molecule_cache(mol_arrays, mol.occ, drop_paths=dropped)
    gb2 = D.collate_gated(cache2, [0], occ2)
    parts_drop = G.gated_parts(arm, gb2, gb2.gate_ones())
    p_drop = float(parts_drop["pred_std"].detach().cpu().numpy().astype(np.float64)[0])

    raw_open = p_open * float(sigma) + float(mu)
    raw_masked = p_masked * float(sigma) + float(mu)
    raw_drop = p_drop * float(sigma) + float(mu)
    err = abs(raw_masked - raw_drop)
    tol = float(C.trace_tolerance(raw_open))
    audit = s2_weight_audit(gb, s2_mask, accounting)
    n_dropped = int(dropped.sum()) if dropped.size else 0
    n_dropped_both = int(part["internal"].sum())
    n_dropped_one = int(part["boundary"].sum())
    return {
        "pred_raw_open": raw_open,
        "pred_raw_s2_masked": raw_masked,
        "pred_raw_drop_affected_occurrences": raw_drop,
        "abs_err_raw": float(err),
        "tolerance_raw": tol,
        "n_occurrences_dropped": n_dropped,
        "n_occurrences_dropped_both_bonds_masked": n_dropped_both,
        "n_occurrences_dropped_one_bond_masked": n_dropped_one,
        "accounting_n_s2_affected": int(accounting.n_s2_affected),
        "n_dropped_matches_accounting": bool(n_dropped == int(accounting.n_s2_affected)),
        "weight_audit": audit,
        "passed": bool(err <= tol and n_dropped == int(accounting.n_s2_affected) and audit["passed"]),
    }


__all__ = [
    "DoseAccounting",
    "s2_partition",
    "dose_accounting",
    "s2_weight_audit",
    "gate_drop_equivalence",
]
