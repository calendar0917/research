"""Post-freeze descriptive coverage / k-sensitivity audit for cscl-bjoc-signature-conditional-value-v1.

Created AFTER the single frozen validation comparison.  It changes no model,
table, score rule, floor, shrinkage, gate, threshold or decision.  It only
recomputes the frozen scores to report, descriptively:

* junction-weighted vs type-level signature coverage (candidate vs below-floor
  vs never-seen) and the fallback share of the selected top-4,
* context-unit coverage of the Arm-C top-4,
* k-sensitivity contrasts of the frozen merged per-molecule arrays,
* readable words for the top positive-gain driver classes.

Output: ``results/bjoc_signature_conditional_value_v1/post_freeze_descriptive_coverage.json``.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_signature_conditional_value_v1 as bcv  # noqa: E402
import bjoc_signature_transfer_v1 as bst  # noqa: E402
import run_bjoc_signature_conditional_value_v1 as runner  # noqa: E402

OUT = bcv.OUT
RESIDUAL_TABLES = OUT / "residual_tables.json"
ROW_METRICS = OUT / "row_metrics_v2.npz"
METRICS = OUT / "metrics.json"
TARGET = OUT / "post_freeze_descriptive_coverage.json"
KS = (1, 2, 4, 8)


def main() -> int:
    tables = bcv.read_json(RESIDUAL_TABLES)
    metrics = bcv.read_json(METRICS)
    rows = runner.load_validation_key_rows()
    raw = {s: np.load(bcv.PRIOR / f"validation_raw_seed{s}.npz", allow_pickle=True) for s in (0, 1)}
    mols = {s: list(raw[s]["mols_valid"]) for s in (0, 1)}

    # ---- signature coverage ----
    candidate_keys = {k for k, v in tables["sig"].items() if int(v["n_molecules"]) >= bcv.SIGNATURE_FLOOR_MOLECULES}
    below_floor_keys = {k for k, v in tables["sig"].items() if int(v["n_molecules"]) < bcv.SIGNATURE_FLOOR_MOLECULES}
    disc_keys = candidate_keys | below_floor_keys
    type_counts = Counter(k for per_row in rows for k in per_row)
    n_junctions = int(sum(type_counts.values()))
    never = {k for k in type_counts if k not in disc_keys}
    j_never = int(sum(type_counts[k] for k in never))
    j_below = int(sum(type_counts[k] for k in type_counts if k in below_floor_keys))
    j_cand = int(sum(type_counts[k] for k in type_counts if k in candidate_keys))

    # ---- selection coverage of the frozen top-4 ----
    sel_stats = {}
    for arm in ("A", "B", "C"):
        hits = {"candidate": 0, "below_floor": 0, "never_seen": 0, "ctx_applied": 0, "ctx_known": 0, "total": 0}
        per_seed = {}
        for seed in (0, 1):
            local = dict(hits)
            for i, mol in enumerate(mols[seed]):
                keys = rows[int(mol["row"])]
                degree = np.asarray(mol["degree"], dtype=np.int64)
                erc = np.asarray(mol["endpoint_ring_count"], dtype=np.int64)
                d = bcv.percentile(np.asarray(mol["dev"], dtype=np.float64))
                sig_b, ctx_b = bcv.bonus_arrays(tables, keys, degree, erc)
                score = bcv.arm_score(d, sig_b, ctx_b, arm)
                for j in bcv.top_k_indices(score, 4):
                    local["total"] += 1
                    k = keys[j]
                    if k in candidate_keys:
                        local["candidate"] += 1
                    elif k in below_floor_keys:
                        local["below_floor"] += 1
                    else:
                        local["never_seen"] += 1
                    if arm == "C":
                        ck = bcv.context_key(k, int(degree[j]), int(erc[j]))
                        if ck in tables["ctx"]:
                            local["ctx_known"] += 1
                            if int(tables["ctx"][ck]["n_molecules"]) >= bcv.CONTEXT_FLOOR_MOLECULES:
                                local["ctx_applied"] += 1
            per_seed[str(seed)] = local
            for key in hits:
                hits[key] += local[key]
        sel_stats[arm] = {
            "pooled": {k: int(v) for k, v in hits.items()},
            "fractions": {
                "candidate": hits["candidate"] / max(hits["total"], 1),
                "below_floor": hits["below_floor"] / max(hits["total"], 1),
                "never_seen": hits["never_seen"] / max(hits["total"], 1),
                **(
                    {
                        "ctx_applied": hits["ctx_applied"] / max(hits["total"], 1),
                        "ctx_known": hits["ctx_known"] / max(hits["total"], 1),
                    }
                    if arm == "C"
                    else {}
                ),
            },
            "per_seed": per_seed,
        }

    # ---- k-sensitivity contrasts (frozen arrays) ----
    with np.load(ROW_METRICS, allow_pickle=True) as z:
        k_contrasts = {}
        for k in KS:
            merged = {a: (z[f"seed0_{a}{k}"] + z[f"seed1_{a}{k}"]) / 2.0 for a in ("A", "B", "C")}
            k_contrasts[f"k{k}"] = {
                "mean_A": float(merged["A"].mean()),
                "mean_B": float(merged["B"].mean()),
                "mean_C": float(merged["C"].mean()),
                "B_minus_A": bcv.contrast(merged["B"], merged["A"]),
                "C_minus_B": bcv.contrast(merged["C"], merged["B"]),
            }

    # ---- words for the top driver classes ----
    awords, bwords = bst.zinc_words()
    top = []
    for rec in metrics["concentration"]["top5_driver_shares"]:
        parts = [int(x) for x in rec["driver"].split("|")]
        key = (parts[0], (parts[1], parts[2]), (parts[3], parts[4]), parts[5], parts[6])
        top.append(
            {
                **rec,
                "words": bst.signature_words(key, awords, bwords),
                "R_shrunk": float(tables["sig"][rec["driver"]]["R_shrunk"]) if rec["driver"] in tables["sig"] else None,
                "support_molecules": int(tables["sig"][rec["driver"]]["n_molecules"]) if rec["driver"] in tables["sig"] else None,
            }
        )

    payload: dict[str, Any] = {
        "protocol": bcv.PROTOCOL_ID,
        "note": (
            "POST-FREEZE descriptive audit created after the single frozen validation comparison; "
            "no score, table, floor, shrinkage, gate or decision was changed."
        ),
        "signature_coverage": {
            "n_validation_junctions": n_junctions,
            "n_distinct_types": int(len(type_counts)),
            "n_candidate_types": int(len(candidate_keys)),
            "n_below_floor_types": int(len(below_floor_keys)),
            "n_never_seen_types": int(len(never)),
            "junction_fraction_candidate": j_cand / max(n_junctions, 1),
            "junction_fraction_below_floor": j_below / max(n_junctions, 1),
            "junction_fraction_never_seen": j_never / max(n_junctions, 1),
            "type_fraction_never_seen": len(never) / max(len(type_counts), 1),
        },
        "selection_coverage": sel_stats,
        "k_sensitivity_contrasts": k_contrasts,
        "top_driver_words": top,
    }
    bcv.write_json(TARGET, payload)
    print("[posthoc-coverage] wrote", TARGET)
    print(
        f"[posthoc-coverage] junctions: candidate {payload['signature_coverage']['junction_fraction_candidate']:.4f}, "
        f"below-floor {payload['signature_coverage']['junction_fraction_below_floor']:.4f}, "
        f"never-seen {payload['signature_coverage']['junction_fraction_never_seen']:.4f}"
    )
    for k in KS:
        c = k_contrasts[f"k{k}"]["B_minus_A"]
        print(f"[posthoc-coverage] k={k} B-A {c['mean']:+.5f} [{c['ci95_lower']:+.5f},{c['ci95_upper']:+.5f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
