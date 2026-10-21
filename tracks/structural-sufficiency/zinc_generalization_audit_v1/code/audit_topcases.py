"""Readable review of the <=10 molecules with the largest cross-model error."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import audit_common as C  # noqa: E402

ANALYSIS = C.ANALYSIS
CATS = [
    "C", "O", "N", "F", "C(H1)", "S", "Cl", "O-", "N(H1)+", "Br", "N(H3)+",
    "N(H2)+", "N+", "N-", "S-", "I", "P", "O(H1)+", "N(H1)-", "O+", "S+",
]


def main() -> int:
    rows = list(csv.DictReader(open(ANALYSIS / "per_molecule.csv", encoding="utf-8")))
    y = np.asarray([float(r["y_raw"]) for r in rows])
    names = list(C.MODELS)
    pred = {k: np.asarray([float(r[f"pred_{k}"]) for r in rows]) for k in names}
    ae = {k: np.asarray([float(r[f"abs_error_{k}"]) for r in rows]) for k in names}
    med = np.median(np.stack([ae[k] for k in names], axis=0), axis=0)
    top = np.argsort(med)[-10:][::-1]

    cases = []
    for i in top:
        r = rows[i]
        atom_counts = [int(x) for x in r["atom_counts_21"].split("|")]
        atom_desc = {CATS[j]: int(v) for j, v in enumerate(atom_counts) if v > 0}
        bond_counts = [int(x) for x in r["bond_counts_1_3"].split("|")]
        cases.append(
            {
                "row_id": int(i),
                "graph_sha256": r["graph_sha256"],
                "y_raw": float(y[i]),
                "n_atoms": int(r["n_atoms"]),
                "n_bonds": int(r["n_bonds"]),
                "n_components": int(r["n_components"]),
                "cycle_rank": int(r["cycle_rank"]),
                "n_rings": int(r["n_rings"]),
                "max_ring_size": int(r["max_ring_size"]),
                "degree_max": int(r["degree_max"]),
                "atom_type_counts_readable": atom_desc,
                "bond_type_counts_1_2_3": bond_counts,
                "median_abs_error_all7": float(med[i]),
                "mean_abs_error_all7": float(np.mean([ae[k][i] for k in names])),
                "P3B_worst10_member": bool(int(r["P3B_worst10"])),
                "predictions": {k: float(pred[k][i]) for k in names},
                "signed_errors": {k: float(pred[k][i] - y[i]) for k in names},
                "abs_errors": {k: float(ae[k][i]) for k in names},
                "same_error_direction_all_models": bool(
                    len(set(np.sign([pred[k][i] - y[i] for k in names]).tolist())) == 1
                ),
                "j_unseen_occurrence_mass": float(r["j_unseen_mass"]),
                "j_lowfreq_1_4_occurrence_mass": float(r["j_lowfreq_mass"]),
                "radius2_unseen_atom_fraction": float(r["r2_unseen_atom_frac"]),
                "radius2_low4_atom_fraction": float(r["r2_low4_atom_frac"]),
                "cache_fidelity": "verified for all 1000 valid rows against official processed val.pt (atom/edge/label exact)",
            }
        )

    out = {
        "selection": "top-10 by median absolute error across the 7 fixed models (robust to one bad seed); no row deletion",
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "n_same_error_direction_all_models": int(sum(c["same_error_direction_all_models"] for c in cases)),
            "n_with_radius2_unseen_atoms": int(sum(c["radius2_unseen_atom_fraction"] > 0 for c in cases)),
            "note": "These are descriptive hard cases, not a held-out test and not evidence that the target is unlearnable.",
        },
    }
    C.write_json(ANALYSIS / "top_cases.json", out)
    print(json.dumps(out, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
