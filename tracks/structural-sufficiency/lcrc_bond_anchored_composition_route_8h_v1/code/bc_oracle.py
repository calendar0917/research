"""C5 — external calibration against the atom-additive WC_LOGP oracle.

``WC_LOGP(G) = sum_v oracle_c_v`` is an *externally computed* additive table
(RDKit Wildman-Crippen atom contributions, never a chemical cause).  That makes it
the one reference in this round that does not come from the model itself, which is
exactly what the round's third question needs:

    capture(P) = sum_{v in atoms(P)} |oracle_c_v| / sum_v |oracle_c_v|

``P* = argmax_P |c_P|`` is the pre-registered, label-free discovery rule.  The
enrichment is ``capture(P*)`` minus the capture of matched random events drawn from
the *same* molecule's candidate pool (up to 16 controls, seed 20261118), with a
2 000-draw molecule-level bootstrap.

Primary evaluation: the WC_LOGP models (the task whose label *is* the oracle).
A Lipo model is reported as a *transfer* row and never as the primary result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import bc_common as C
import bc_data as D
import bc_explain as E
import bc_path3 as P3

N_CONTROLS = 16


def _oracle(b) -> tuple[np.ndarray, np.ndarray]:
    XR = C.modules()["xr_common"]
    with np.load(XR.ORACLE_NPZ, allow_pickle=False) as mesh:
        atom = np.asarray(mesh["WC_LOGP_atom"], dtype=np.float64)
        atom_ptr = np.asarray(mesh["atom_ptr"], dtype=np.int64)
    return atom, atom_ptr


def capture_table(
    c_P: np.ndarray,
    graph: np.ndarray,
    p3s: P3.Path3Split,
    rows: Sequence[int],
    b,
    *,
    seed: int = C.ORACLE_CONTROL_SEED,
    n_controls: int = N_CONTROLS,
) -> dict[str, Any]:
    """Capture of the top event vs matched random events, per molecule."""
    atom, atom_ptr = _oracle(b)
    ids = np.asarray(b.global_row_ids["valid"], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    out_top: list[float] = []
    out_random: list[float] = []
    out_random_max: list[float] = []
    per_molecule: list[dict[str, Any]] = []
    for position, molecule in enumerate(rows):
        gid = int(ids[int(molecule)])
        a0, a1 = int(atom_ptr[gid]), int(atom_ptr[gid + 1])
        weight = np.abs(atom[a0:a1])
        denom = float(weight.sum())
        lo, hi = int(p3s.ptr[int(molecule)]), int(p3s.ptr[int(molecule) + 1])
        if hi <= lo or denom <= 0.0:
            continue
        events = np.arange(lo, hi)
        in_batch = graph == position
        if not in_batch.any():
            continue
        local = c_P[in_batch]
        best = int(np.argmax(np.abs(local)))
        atoms_best = p3s.atoms[lo + best]
        cap_best = float(weight[atoms_best].sum() / denom)
        draws = []
        for _ in range(int(n_controls)):
            pick = int(rng.integers(0, events.size))
            while events.size > 1 and pick == best:
                pick = int(rng.integers(0, events.size))
            atoms_pick = p3s.atoms[lo + pick]
            draws.append(float(weight[atoms_pick].sum() / denom))
        out_top.append(cap_best)
        out_random.append(float(np.mean(draws)))
        out_random_max.append(float(np.max(draws)))
        per_molecule.append(
            {
                "module_row": int(molecule),
                "global_row_id": gid,
                "capture_top": cap_best,
                "capture_random_mean": float(np.mean(draws)),
                "n_events": int(events.size),
                "abs_c_top": float(np.abs(local[best])),
            }
        )
    top = np.asarray(out_top)
    rand = np.asarray(out_random)
    rand_max = np.asarray(out_random_max)
    rng2 = np.random.default_rng(int(seed))
    diff = top - rand
    draws = np.empty(C.BOOTSTRAP_DRAWS)
    for d in range(C.BOOTSTRAP_DRAWS):
        idx = rng2.integers(0, diff.size, diff.size)
        draws[d] = float(np.mean(diff[idx]))
    return {
        "n_molecules": int(top.size),
        "capture_top_mean": float(top.mean()),
        "capture_random_mean": float(rand.mean()),
        "capture_random_max_mean": float(rand_max.mean()),
        "enrichment": float(diff.mean()),
        "enrichment_ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "prob_enrichment_positive": float(np.mean(draws > 0.0)),
        "n_controls": int(n_controls),
        "seed": int(seed),
        "capture_top_median": float(np.median(top)),
        "capture_random_median": float(np.median(rand)),
        "fraction_molecules_top_above_random": float(np.mean(top > rand)),
        "per_molecule": per_molecule,
    }


def evaluate_unit(unit: str, *, split: str = "valid") -> dict[str, Any]:
    device = torch.device("cpu")
    model, b, payload, meta = E._load_unit(unit, device)
    p3 = D.build_paths3(b)
    p3s = p3[split]
    order, _, _ = D.eval_order(b, split)
    bt = D.make_batch(b, split, order, p3s).to(device)
    with torch.no_grad():
        parts = model(bt.batch, bt.p3, return_parts=True)
    c = parts["c_P"].detach().cpu().numpy().astype(np.float64)
    graph = bt.p3["p3_graph"].numpy().astype(np.int64)
    report = {
        "unit": payload.get("unit", meta["checkpoint"]),
        "task": payload["task"],
        "variant": payload["variant"],
        "institution": payload["institution"],
        "seed": int(payload["seed"]),
        "selected_epoch": int(payload["epoch"]),
        "split": split,
    }
    report["capture"] = capture_table(c, graph, p3s, order, b)

    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="C5 WC_LOGP oracle localisation")
    ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--out", default=str(C.ANALYSIS_DIR / "oracle_localisation.json"))
    args = ap.parse_args(argv)
    rows = []
    for unit in args.units:
        report = evaluate_unit(unit)
        cap = report["capture"]
        cap.pop("per_molecule", None)
        rows.append(report)
        print(f"{report['task']:8s} {report['variant']:8s} epoch {report['selected_epoch']:3d} "
              f"capture_top {cap['capture_top_mean']:.4f} random {cap['capture_random_mean']:.4f} "
              f"enrichment {cap['enrichment']:+.4f} "
              f"[{cap['enrichment_ci95'][0]:+.4f},{cap['enrichment_ci95'][1]:+.4f}] "
              f"P>0 {cap['prob_enrichment_positive']:.3f}")
    C.atomic_write_json(Path(args.out), {"round_id": C.ROUND_ID, "units": rows,
                                         "n_controls": N_CONTROLS,
                                         "control_seed": C.ORACLE_CONTROL_SEED,
                                         "statistic": "capture(top |c_P| event) - capture(matched random event of the same molecule)"})
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
