"""Batched, memory-bounded re-evaluation of the frozen function under one mask.

Only two things are measured here, both with the frozen model in ``eval()`` mode
and ``torch.no_grad()``:

1. the raw prediction / ``local_sum`` / ``rho`` after the
   ``CONDITIONAL_S1_S2_EVIDENCE_MASK`` — the same quantity the pilot stored, used
   to (a) re-identify the pilot's Top-1 effect and (b) score the corrected null;
2. the per-root-node local output ``c_v = f(h_v)`` before and after the mask, so
   the masked response can be attributed to the *roots* that produced it.

No parameter is touched, no gradient is formed, no loss is defined.  Variants of
one molecule are replicated into a single small batch and evaluated in chunks
(:data:`au_common.MAX_VARIANTS_PER_FORWARD`) so peak memory stays flat regardless
of how many controls a molecule has.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import au_common as C
import au_s2 as S2

import as_data as D
import as_gate as G


def mask_vector(n_bonds: int, bonds: Sequence[int] | None, value: float = 0.0) -> np.ndarray:
    if bonds is None:
        return np.ones(int(n_bonds), dtype=np.float64)
    return G.bond_mask(int(n_bonds), bonds, float(value))


def variant(name: str, kind: str, bonds: Sequence[int], **extra: Any) -> dict[str, Any]:
    return {"name": str(name), "kind": str(kind), "bonds": [int(b) for b in bonds], **extra}


@torch.no_grad()
def run_variants(
    arm: torch.nn.Module,
    cache: Mapping[str, Any],
    row: int,
    occ: D.SplitOcc,
    variants: Sequence[Mapping[str, Any]],
    *,
    sigma: float,
    mu: float,
    chunk_size: int = C.MAX_VARIANTS_PER_FORWARD,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every variant of one molecule, chunked, and keep per-atom outputs."""
    if arm.training:
        raise RuntimeError("the audit re-evaluates the frozen model in eval() mode only")
    if not variants:
        return []
    n_bonds = int(cache["bond_ptr"][int(row) + 1] - cache["bond_ptr"][int(row)])
    n_atoms = int(cache["atom_ptr"][int(row) + 1] - cache["atom_ptr"][int(row)])
    chunk_size = max(1, min(int(chunk_size), int(C.MAX_VARIANTS_PER_FORWARD)))
    out: list[dict[str, Any]] = []
    for start in range(0, len(variants), chunk_size):
        chunk = list(variants[start:start + chunk_size])
        rep_cache, rep_occ = D.replicate_molecule(cache, int(row), len(chunk), occ)
        gb = D.collate_gated(rep_cache, list(range(len(chunk))), rep_occ)
        if device is not None:
            gb = gb.to(device)
        n_atoms_batch = int(gb.batch["atom_feat"].shape[0])
        n_bonds_batch = int(gb.batch["bond_index"].shape[0])
        if n_atoms_batch > C.MAX_FORWARD_ATOMS or n_bonds_batch > C.MAX_FORWARD_BONDS:
            raise RuntimeError(
                f"refusing a forward of {n_atoms_batch} atoms / {n_bonds_batch} bonds "
                f"(guards {C.MAX_FORWARD_ATOMS}/{C.MAX_FORWARD_BONDS}); lower "
                "MAX_VARIANTS_PER_FORWARD instead of raising the guard"
            )
        masks = [mask_vector(n_bonds, v["bonds"], 0.0) for v in chunk]
        M = gb.gate_matrix(masks, device=device)
        parts = G.gated_parts(arm, gb, M)
        pred = parts["pred_std"].detach().cpu().numpy().astype(np.float64) * float(sigma) + float(mu)
        local = parts["local_sum"].detach().cpu().numpy().astype(np.float64) * float(sigma)
        rho = parts["rho"].detach().cpu().numpy().astype(np.float64) * float(sigma)
        contrib = (
            parts["contrib_atom"].detach().cpu().numpy().astype(np.float64).reshape(len(chunk), n_atoms)
            * float(sigma)
        )
        for i, v in enumerate(chunk):
            out.append(
                {
                    "name": v["name"],
                    "kind": v["kind"],
                    "bonds": [int(b) for b in v["bonds"]],
                    "pred_raw": float(pred[i]),
                    "local_sum_raw": float(local[i]),
                    "rho_raw": float(rho[i]),
                    "contrib_atom_raw": contrib[i].copy(),
                    **{k: v[k] for k in v if k not in ("name", "kind", "bonds")},
                }
            )
        del parts, gb, rep_cache, rep_occ, M
    return out


def effect_record(base: Mapping[str, Any], after: Mapping[str, Any], *, train_std: float) -> dict[str, Any]:
    """``delta = y0 - yS`` in raw target units, split into local sum and rho."""
    delta_total = float(base["pred_raw"]) - float(after["pred_raw"])
    delta_l = float(base["local_sum_raw"]) - float(after["local_sum_raw"])
    delta_rho = float(base["rho_raw"]) - float(after["rho_raw"])
    closure = abs(delta_total - (delta_l + delta_rho))
    tol = float(C.trace_tolerance(delta_total))
    return {
        "pred_raw": float(after["pred_raw"]),
        "local_sum_raw": float(after["local_sum_raw"]),
        "rho_raw": float(after["rho_raw"]),
        "delta_total": delta_total,
        "delta_L": delta_l,
        "delta_rho": delta_rho,
        "abs_delta_total": abs(delta_total),
        "abs_delta_L": abs(delta_l),
        "abs_delta_rho": abs(delta_rho),
        "delta_total_over_train_std": delta_total / float(train_std),
        "abs_delta_total_over_train_std": abs(delta_total) / float(train_std),
        "closure_total_error": closure,
        "closure_total_tolerance": tol,
        "closure_total_passed": bool(closure <= tol),
    }


def localization_record(
    base: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    inside_atoms: np.ndarray,
    accounting: S2.DoseAccounting,
    train_std: float,
) -> dict[str, Any]:
    """Root-node localisation of the masked response for one molecule.

    ``delta_c_v = c_v(original) - c_v(masked)`` for every atom root ``v``.  The
    inside/outside split is a statement about *where the local readout moves*, not
    an independent or unique chemical contribution assignment.
    """
    inside = np.asarray(inside_atoms, dtype=bool)
    delta_c = np.asarray(base["contrib_atom_raw"], dtype=np.float64) - np.asarray(
        after["contrib_atom_raw"], dtype=np.float64
    )
    if delta_c.size != inside.size:
        raise ValueError(f"atom mask has {inside.size} entries for {delta_c.size} roots")
    abs_dc = np.abs(delta_c)
    inside_sum = float(abs_dc[inside].sum())
    outside_sum = float(abs_dc[~inside].sum())
    total = inside_sum + outside_sum
    eff = effect_record(base, after, train_std=train_std)
    local_closure = abs(float(delta_c.sum()) - eff["delta_L"])
    local_tol = float(C.trace_tolerance(eff["delta_L"]))
    return {
        "n_atoms": int(inside.size),
        "n_atoms_inside": int(inside.sum()),
        "n_atoms_outside": int((~inside).sum()),
        "inside_abs_response": inside_sum,
        "outside_abs_response": outside_sum,
        "total_abs_response": total,
        "outside_response_fraction": outside_sum / (total + 1e-15) if total > 0 else float("nan"),
        "inside_signed_response": float(delta_c[inside].sum()),
        "outside_signed_response": float(delta_c[~inside].sum()),
        "n_root_inside_moved": int((abs_dc[inside] > 0).sum()),
        "n_root_outside_moved": int((abs_dc[~inside] > 0).sum()),
        "max_abs_delta_c_inside": float(abs_dc[inside].max()) if inside.any() else 0.0,
        "max_abs_delta_c_outside": float(abs_dc[~inside].max()) if (~inside).any() else 0.0,
        "closure_local_sum_error": float(local_closure),
        "closure_local_sum_tolerance": local_tol,
        "closure_local_sum_passed": bool(local_closure <= local_tol),
        "delta_total": eff["delta_total"],
        "delta_L": eff["delta_L"],
        "delta_rho": eff["delta_rho"],
        "abs_delta_total": eff["abs_delta_total"],
        "closure_total_error": eff["closure_total_error"],
        "closure_total_tolerance": eff["closure_total_tolerance"],
        "closure_total_passed": eff["closure_total_passed"],
        "n_s2_internal": int(accounting.n_s2_internal),
        "n_s2_boundary": int(accounting.n_s2_boundary),
        "n_s2_affected": int(accounting.n_s2_affected),
        "n_s2_total": int(accounting.n_s2_total),
        "boundary_over_affected": accounting.boundary_over_affected,
        "affected_root_outside_fraction": accounting.affected_root_outside_fraction,
        "n_s2_affected_root_inside": int(accounting.n_s2_affected_root_inside),
        "n_s2_affected_root_outside": int(accounting.n_s2_affected_root_outside),
    }


__all__ = ["mask_vector", "variant", "run_variants", "effect_record", "localization_record"]
