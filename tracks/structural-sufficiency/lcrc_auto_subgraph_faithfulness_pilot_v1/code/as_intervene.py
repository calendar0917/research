"""Real forward-response measurement of the ``CONDITIONAL_S1_S2_EVIDENCE_MASK``.

Every number produced here comes from the frozen arm-A function evaluated again
with explicit physical-bond gates -- never from the gradient, never from an
attribution heuristic.  All variants of one molecule are evaluated in a single
batched forward so the cost per molecule stays at one forward pass.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import as_common as C
import as_data as D
import as_gate as G


# ---------------------------------------------------------------------------
# variant construction
# ---------------------------------------------------------------------------


def _gate(n_bonds: int, bonds: Sequence[int] | None, value: float) -> np.ndarray:
    if bonds is None:
        return np.ones(int(n_bonds), dtype=np.float64)
    return G.bond_mask(n_bonds, bonds, value)


def build_variants(
    mol: G.SingleMolecule,
    topk: Sequence[Sequence[int]],
    *,
    local_heuristic_bonds: Sequence[int] | None,
    matched_random: Sequence[Sequence[int]],
    keycount_only: Sequence[Sequence[int]],
    dose: float = C.DOSE_GATE,
) -> list[dict[str, Any]]:
    """The fixed, ordered list of intervention variants for one molecule."""
    n = mol.n_bonds
    ones = np.ones(n, dtype=np.float64)
    variants: list[dict[str, Any]] = []

    def add(name: str, s1: np.ndarray, s2: np.ndarray, bonds: Sequence[int], kind: str, **extra) -> None:
        variants.append(
            {
                "name": name,
                "kind": kind,
                "bonds": [int(b) for b in bonds],
                "gate_s1": np.asarray(s1, dtype=np.float64),
                "gate_s2": np.asarray(s2, dtype=np.float64),
                **extra,
            }
        )

    add("BASE", ones, ones, (), "baseline")

    for rank, bonds in enumerate(topk, start=1):
        m0 = _gate(n, bonds, 0.0)
        add(f"TOP{rank}", m0, m0, bonds, "discovery_mask", rank=rank)

    if topk:
        top1 = topk[0]
        m_half = _gate(n, top1, float(dose))
        add("TOP1_DOSE", m_half, m_half, top1, "dose", dose=float(dose))
        m0 = _gate(n, top1, 0.0)
        add("TOP1_S1_ONLY", m0, ones, top1, "s1_only", rank=1)
        add("TOP1_S2_ONLY", ones, m0, top1, "s2_only", rank=1)
        add("TOP1_S1_S2", m0, m0, top1, "s1_s2", rank=1)

    if local_heuristic_bonds is not None:
        m0 = _gate(n, local_heuristic_bonds, 0.0)
        add("LOCAL_HEURISTIC", m0, m0, local_heuristic_bonds, "local_heuristic")

    for i, bonds in enumerate(matched_random):
        m0 = _gate(n, bonds, 0.0)
        add(f"MATCHED_RANDOM_{i:02d}", m0, m0, bonds, "matched_random", index=i)

    for i, bonds in enumerate(keycount_only):
        m0 = _gate(n, bonds, 0.0)
        add(f"KEYCOUNT_ONLY_{i:02d}", m0, m0, bonds, "keycount_only", index=i)

    return variants


# ---------------------------------------------------------------------------
# batched execution
# ---------------------------------------------------------------------------


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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every variant of one molecule in a single batched forward."""
    rep_cache, rep_occ = D.replicate_molecule(cache, int(row), len(variants), occ)
    gb = D.collate_gated(rep_cache, list(range(len(variants))), rep_occ)
    M1 = gb.gate_matrix([v["gate_s1"] for v in variants])
    M2 = gb.gate_matrix([v["gate_s2"] for v in variants])
    parts = G.gated_parts(arm, gb, M1, M2)
    pred_raw = parts["pred_std"].detach().cpu().numpy().astype(np.float64) * float(sigma) + float(mu)
    local_raw = parts["local_sum"].detach().cpu().numpy().astype(np.float64) * float(sigma)
    rho_raw = parts["rho"].detach().cpu().numpy().astype(np.float64) * float(sigma)
    out = []
    for i, v in enumerate(variants):
        out.append(
            {
                "name": v["name"],
                "kind": v["kind"],
                "bonds": [int(b) for b in v["bonds"]],
                "pred_raw": float(pred_raw[i]),
                "local_sum_raw": float(local_raw[i]),
                "rho_raw": float(rho_raw[i]),
                **{k: v[k] for k in v if k not in ("name", "kind", "bonds", "gate_s1", "gate_s2")},
            }
        )
    # batching must not change the base prediction (cross-checked in tests too)
    base = next(o for o in out if o["name"] == "BASE")
    return out, {"n_variants": len(variants), "base_pred_raw": base["pred_raw"]}


def effect_record(
    base: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    train_std: float,
    tol_scale: float = 1.0,
) -> dict[str, Any]:
    """``delta = y0 - yS`` in raw units plus the two readout components."""
    delta = float(base["pred_raw"]) - float(after["pred_raw"])
    delta_l = float(base["local_sum_raw"]) - float(after["local_sum_raw"])
    delta_p = float(base["rho_raw"]) - float(after["rho_raw"])
    total_abs = abs(delta_l) + abs(delta_p)
    return {
        "pred_raw": float(after["pred_raw"]),
        "local_sum_raw": float(after["local_sum_raw"]),
        "rho_raw": float(after["rho_raw"]),
        "signed_delta": delta,
        "abs_delta": abs(delta),
        "delta_over_train_std": delta / float(train_std),
        "abs_delta_over_train_std": abs(delta) / float(train_std),
        "delta_L": delta_l,
        "delta_P": delta_p,
        "abs_delta_L": abs(delta_l),
        "abs_delta_P": abs(delta_p),
        "rho_share": abs(delta_p) / (total_abs + 1e-12),
        "cancellation": (total_abs - abs(delta)) / (total_abs + 1e-12),
        "closure_error": abs(delta - (delta_l + delta_p)),
        "closure_tolerance": float(C.trace_tolerance(delta)) * float(tol_scale),
    }


def decomposition_record(
    base: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    *,
    train_std: float,
) -> dict[str, Any]:
    """S1/S2 interaction diagnostic on the raw prediction scale."""
    out: dict[str, Any] = {}
    for key, name in (("s1_only", "TOP1_S1_ONLY"), ("s2_only", "TOP1_S2_ONLY"), ("s1_s2", "TOP1_S1_S2")):
        rec = effect_record(base, variants[name], train_std=train_std)
        out[key] = rec
    d_s1 = out["s1_only"]["signed_delta"]
    d_s2 = out["s2_only"]["signed_delta"]
    d_both = out["s1_s2"]["signed_delta"]
    out["interaction_delta"] = d_both - d_s1 - d_s2
    out["interaction_note"] = (
        "conditional interaction of the frozen function under this mask; not an "
        "independent chemical interaction energy"
    )
    return out


__all__ = ["build_variants", "run_variants", "effect_record", "decomposition_record"]
