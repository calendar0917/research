"""Stage D read-only integration: where does the C1 -> GINE gap live?

No training, no GPU, no new model.  Uses only
* the frozen replay predictions of the sprint round (C1 = ``L_lipophilicity``,
  GINE-G0 = ``G0_lipophilicity``), and
* the frozen three-bond-path cache for the structural covariates.

The four slices are pre-registered in ``analysis/stage_d_plan.json`` *before* the
structure of the gap was inspected in this round.  Every row reports the slice
RMSE of both models, the gap, and a paired molecule bootstrap CI of the gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import bc_analysis as A
import bc_common as C
import bc_data as D
import bc_path3 as P3


def _quantile_bins(values: np.ndarray, n: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.int64)
    ranks[order] = np.arange(values.size)
    return np.minimum(ranks * n // values.size, n - 1)


def _ring_fraction(cache, n_molecules: int) -> np.ndarray:
    """Fraction of atoms that sit in a ring, straight from the bond table."""
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    bond_type = np.asarray(cache["bond_feat"]) if "bond_feat" in cache else None
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    n_atoms = np.diff(atom_ptr).astype(np.int64)
    # a bond is in a ring iff removing it leaves its endpoints connected: use the
    # cheapest sufficient test, 2-core membership of the bond graph
    out = np.zeros(n_molecules, dtype=np.float64)
    for g in range(n_molecules):
        lo, hi = int(bond_ptr[g]), int(bond_ptr[g + 1])
        if hi <= lo:
            continue
        edges = bond_index[lo:hi]
        n_a = int(n_atoms[g])
        deg = np.bincount(edges.ravel(), minlength=n_a)
        alive = deg > 0
        core = np.zeros(n_a, dtype=bool)
        frontier = np.where(alive & (deg <= 1))[0]
        deg_work = deg.copy()
        while frontier.size:
            node = int(frontier[0])
            frontier = frontier[1:]
            if not alive[node]:
                continue
            alive[node] = False
            for nb in edges[(edges[:, 0] == node) | (edges[:, 1] == node)].ravel():
                if nb != node and alive[nb]:
                    deg_work[nb] -= 1
                    if deg_work[nb] <= 1:
                        frontier = np.append(frontier, nb)
        core = alive
        out[g] = float(core.sum() / max(1, n_atoms[g]))
    return out


def build() -> dict[str, Any]:
    b = D.bundle("lipo")
    hist = {name: A.hist_predictions(name) for name in (A.HIST_LIPO, A.HIST_GINE)}
    p3 = D.build_paths3(b)
    out: dict[str, Any] = {"round_id": C.ROUND_ID, "source": "read-only historical predictions"}
    cache = b.caches["valid"]
    n_molecules = int(cache["n_molecules"])
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    atom_count = np.diff(atom_ptr).astype(np.float64)
    n_paths3 = np.asarray(p3["valid"].counts, dtype=np.float64)
    ring_fraction = _ring_fraction(cache, n_molecules)

    for split in ("valid", "test"):
        c1 = hist[A.HIST_LIPO][split]
        g0 = hist[A.HIST_GINE][split]
        err_c1 = c1["pred_raw"] - c1["y_raw"]
        err_g0 = g0["pred_raw"] - g0["y_raw"]
        structural = split == "valid" or split == "test"
        cov = {}
        if structural:
            if split != "valid":
                cache_s = b.caches[split]
                atom_ptr_s = np.asarray(cache_s["atom_ptr"], dtype=np.int64)
                cov["atom_count"] = np.diff(atom_ptr_s).astype(np.float64)
                cov["paths3"] = np.asarray(p3[split].counts, dtype=np.float64)
                cov["ring_fraction"] = _ring_fraction(cache_s, int(cache_s["n_molecules"]))
            else:
                cov["atom_count"] = atom_count
                cov["paths3"] = n_paths3
                cov["ring_fraction"] = ring_fraction
        slices: dict[str, list[dict[str, Any]]] = {}
        definitions: dict[str, np.ndarray] = {}
        definitions["atom_count_quartile"] = _quantile_bins(cov["atom_count"], 4)
        definitions["paths3_quartile"] = _quantile_bins(cov["paths3"], 4)
        rf = cov["ring_fraction"]
        definitions["ring_atom_fraction_bin"] = np.digitize(rf, [1e-9, 0.3, 0.6]) - 1
        definitions["c1_error_decile"] = _quantile_bins(np.abs(err_c1), 10)
        for name, labels in definitions.items():
            rows = []
            for value in np.unique(labels):
                mask = labels == value
                if mask.sum() < 5:
                    continue
                boot = A.bootstrap_paired(err_g0[mask], err_c1[mask])
                rows.append(
                    {
                        "slice": name,
                        "bin": int(value),
                        "n": int(mask.sum()),
                        "rmse_c1": boot["rmse_b"],
                        "rmse_gine": boot["rmse_a"],
                        "gap_c1_minus_gine": boot["delta"],
                        "ci95": boot["ci95"],
                        "prob_gine_better": boot["prob_a_better"],
                    }
                )
            slices[name] = rows
        out[split] = {
            "overall": {
                "n": int(err_c1.size),
                "rmse_c1": float(np.sqrt(np.mean(err_c1**2))),
                "rmse_gine": float(np.sqrt(np.mean(err_g0**2))),
                "gap": float(np.sqrt(np.mean(err_c1**2)) - np.sqrt(np.mean(err_g0**2))),
                "paired": A.bootstrap_paired(err_g0, err_c1),
            },
            "slices": slices,
        }
    C.atomic_write_json(C.ANALYSIS_DIR / "stage_d_error_localisation.json", out)
    return out


def to_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# Stage D — C1 -> GINE gap localisation (read-only)\n"]
    for split in ("valid", "test"):
        block = payload[split]
        o = block["overall"]
        lines.append(f"## {split}\n")
        lines.append(f"overall n={o['n']} C1={o['rmse_c1']:.4f} GINE={o['rmse_gine']:.4f} "
                     f"gap={o['gap']:+.4f} (95% CI {o['paired']['ci95'][0]:+.4f}, {o['paired']['ci95'][1]:+.4f})\n")
        for name, rows in block["slices"].items():
            lines.append(f"\n### {name}\n")
            lines.append("| bin | n | C1 | GINE | gap | 95% CI | P(GINE better) |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in rows:
                lines.append(
                    f"| {r['bin']} | {r['n']} | {r['rmse_c1']:.4f} | {r['rmse_gine']:.4f} | "
                    f"{r['gap_c1_minus_gine']:+.4f} | [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] | "
                    f"{r['prob_gine_better']:.2f} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = build()
    (C.ANALYSIS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (C.ANALYSIS_DIR / "tables" / "stage_d.md").write_text(to_markdown(payload))
    v = payload["valid"]["overall"]
    print(f"valid: C1 {v['rmse_c1']:.4f} GINE {v['rmse_gine']:.4f} gap {v['gap']:+.4f}")


if __name__ == "__main__":
    main()
