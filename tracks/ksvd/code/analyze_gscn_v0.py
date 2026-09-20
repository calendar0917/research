#!/usr/bin/env python
"""GSCN-v0 result analysis (valid-only; official test never loaded).

Reads ``tracks/ksvd/results/gscn_v0/`` summaries (produced on the A100 by
``run_gscn_v0.py``), applies the pre-registered gates and prints/writes the
frozen-format verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "tracks/ksvd/results/gscn_v0"

M_FULL = 0.119818           # canonical B-Full reference (seed 0 soup)
M_BNULL_CANON = 0.123028    # canonical B-Null reference (seed 0 soup)

ARMS = ["bnull_raw", "generic", "nothresh", "sparse"]


def load_json(p: Path) -> Any:
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=RESULTS / "analysis.json")
    args = ap.parse_args()

    summaries = {}
    for arm in ARMS:
        p = RESULTS / f"summary_{arm}.json"
        summaries[arm] = load_json(p) if p.exists() else None

    def metric(arm: str, key: str = "soup_valid"):
        s = summaries.get(arm)
        return None if s is None else float(s[key])

    M_null = metric("bnull_raw")
    M_generic = metric("generic")
    M_nothresh = metric("nothresh")
    M_sparse = metric("sparse")

    missing = [a for a in ARMS if summaries[a] is None]
    if missing:
        print(f"missing summaries: {missing}")
        return 1

    best = {a: float(summaries[a]["best_valid"]) for a in ARMS}
    soup = {a: float(summaries[a]["soup_valid"]) for a in ARMS}

    d_null = M_null - M_sparse
    d_cap = M_generic - M_sparse
    d_sparse = M_nothresh - M_sparse

    strong_positive = (M_sparse <= M_null - 0.02
                       and M_sparse <= M_generic - 0.01
                       and M_sparse <= M_nothresh - 0.01)
    gap_full = M_null - M_FULL
    rho = (d_null / gap_full) if abs(gap_full) >= 0.02 else None
    capacity_only = (M_generic <= M_sparse) or (abs(d_cap) < 0.01)
    sparsity_not_needed = (M_nothresh <= M_sparse) and abs(d_sparse) >= 0.01
    ambiguous = (0.005 < d_null < 0.02) or (abs(d_cap) < 0.01) or (abs(d_sparse) < 0.01)

    smoke = load_json(RESULTS / "smoke_gate.json") if (RESULTS / "smoke_gate.json").exists() else None
    mech = {
        "stage0_all_passed": bool(load_json(RESULTS / "stage0_tests.json")["tests"]["all_passed"])
        if (RESULTS / "stage0_tests.json").exists() else None,
        "smoke_gate_passed": None if smoke is None else bool(smoke["gate"]["passed"]),
        "smoke_pooled_atoms_used": None if smoke is None else smoke.get("pooled_atoms_used"),
    }

    verdict = _verdict(strong_positive, capacity_only, sparsity_not_needed, mech)

    payload = {
        "official_test_loaded": False,
        "M_Null_raw": M_null,
        "M_Generic": M_generic,
        "M_NoThresh": M_nothresh,
        "M_Sparse": M_sparse,
        "M_Full_reference": M_FULL,
        "M_BNull_canonical_reference": M_BNULL_CANON,
        "best_valid": best,
        "soup_valid": soup,
        "delta_null": d_null,
        "delta_capacity": d_cap,
        "delta_sparsity": d_sparse,
        "handcrafted_gap_M_null_minus_M_full": gap_full,
        "rho_recovery": rho,
        "gates": {
            "strong_positive": strong_positive,
            "capacity_only_failure": capacity_only,
            "sparsity_not_needed_failure": sparsity_not_needed,
            "ambiguous_positive_requires_seed1": ambiguous,
        },
        "mechanism": mech,
        "verdict_first_line": verdict,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _verdict(strong, capacity_only, sparsity_not_needed, mech) -> str:
    if mech.get("stage0_all_passed") is False or mech.get("smoke_gate_passed") is False:
        return "Inconclusive due to baseline or implementation integrity failure."
    if strong:
        return "Task-driven sparse dictionary coding is viable as the core graph representation update."
    if sparsity_not_needed:
        return "Dictionary factorization helps, but explicit sparsity is not supported."
    if capacity_only:
        return "The gain is explained by generic model capacity rather than sparse dictionary learning."
    return "Inconclusive: no pre-registered gate fired cleanly (see deltas)."


if __name__ == "__main__":
    raise SystemExit(main())
