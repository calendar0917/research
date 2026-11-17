"""Stage 0: preflight — path3 cache, support statistics, the 14 gates, the freeze.

CPU only.  Nothing is trained here.  The output is
``analysis/path3_support.json``, ``analysis/preflight.json`` and the freeze files
written by :mod:`bc_prepare_freeze`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import bc_common as C
import bc_data as D
import bc_path3 as P3


def path3_support() -> dict:
    b = D.bundle("lipo")
    p3 = D.build_paths3(b, write=True)
    b_wc = D.bundle("wc_logp")
    p3_wc = D.build_paths3(b_wc, write=True)
    out = {"task": "structural_support_of_the_three_bond_path_object",
           "dataset": C.DATASET, "splits": {}, "wc_logp_splits": {},
           "n_bond_types": int(b.spec.n_bond_types),
           "definitions": {
               "event": "one undirected simple three-bond path (a,b,c,d); the anchor bond is (b,c)",
               "closure4": "1 iff the terminal atoms (a,d) are additionally bonded",
               "distinguishable_event": "the molecule has >= 2 events whose second S2 occurrence id differs",
               "no_sampling": "every enumerated path is kept; large molecules are never truncated",
           }}
    gate = {}
    for split in C.SPLITS:
        stats = P3.support_statistics(p3[split], b.caches[split])
        stats.update(P3.clause_statistics(p3[split]))
        out["splits"][split] = stats
        out["wc_logp_splits"][split] = P3.support_statistics(p3_wc[split], b_wc.caches[split])
        gate[split] = float(stats["fraction_molecules_distinguishable"])
    out["pairing_support"] = gate
    out["min_valid_support"] = float(gate["valid"])
    out["threshold"] = C.MIN_PAIRING_SUPPORT
    out["status"] = (
        "PAIRING_SUPPORT_OK" if float(gate["valid"]) >= C.MIN_PAIRING_SUPPORT else "LOW_PAIRING_SUPPORT"
    )
    out["paths3_sha256"] = {split: p3[split].sha256 for split in C.SPLITS}
    out["paths3_wc_logp_sha256"] = {split: p3_wc[split].sha256 for split in C.SPLITS}
    return out


def main() -> None:
    t0 = time.time()
    C.ensure_dir(C.ANALYSIS_DIR)
    support = path3_support()
    C.atomic_write_json(C.ANALYSIS_DIR / "path3_support.json", support)
    print("[path3] support:", support["pairing_support"], "status:", support["status"], flush=True)
    for split, stats in support["splits"].items():
        print(
            f"[path3] {split}: mols={stats['n_molecules']} events={stats['n_events']} "
            f"p50={stats['events_per_molecule_p50']:.1f} p95={stats['events_per_molecule_p95']:.1f} "
            f"max={stats['events_per_molecule_max']} with_event={stats['fraction_molecules_with_event']:.3f} "
            f"distinguishable={stats['fraction_molecules_distinguishable']:.3f} "
            f"closure4={stats['closure4_fraction']:.3f}",
            flush=True,
        )
    import bc_checks as K

    preflight = K.run_all(quick=False)
    preflight["pairing_support_status"] = support["status"]
    preflight["pairing_support_min_valid"] = support["min_valid_support"]
    preflight["stage0_seconds"] = time.time() - t0
    C.atomic_write_json(C.ANALYSIS_DIR / "preflight.json", preflight)
    print("[preflight] failed gates:", preflight["failed"], flush=True)
    print("[preflight] all_pass:", preflight["all_pass"], f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
