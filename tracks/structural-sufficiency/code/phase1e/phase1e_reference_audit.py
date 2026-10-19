"""Phase 1e Stage B1 — protocol-separated performance reference audit.

Three things are done here, all CPU-only:

1. Contamination audit.  For every historical strong-model checkpoint that could
   plausibly serve as a reference, look up the rows it was *supervised* on and
   intersect them with the SSRC Monitor 1,000.  A checkpoint trained on Monitor
   rows cannot produce an independent Monitor number for this project.
2. Protocol separation.  Internal Fit-8k / Monitor-1k numbers, the exposed
   official-test number of the frozen BJOC-P3, and the published official
   ZINC-12k leaderboard numbers are recorded in separate tables and are never
   ranked against each other.
3. Budget estimate for the optional single-GPU strong baseline (Stage B2).

Nothing here trains, selects, or reads a new label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (HERE, REPO / "tracks/structural-sufficiency/code/phase1b"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import phase1e_common as C  # noqa: E402

OUT = C.RESULTS_1E / "reference_audit.json"
SC = REPO / "tracks/structure-contribution/results"
BJOC_ARCHIVE = SC / "bjoc_performance_mechanism_long_v1"
BJOC_MASKED = SC / "bjoc_masked_pretraining_v1"
BJOC_F10K = SC / "bjoc_full10k_pretraining_transfer_v1"
BJOC_RECIPE = SC / "bjoc_recipe_tuning_v1"
BJOC_TERMINAL = SC / "bjoc_r3_frozen_terminal_test_v1"

#: Published official ZINC-12k results, copied from the track's literature
#: boundary note (`notes/prior_work_boundary.md`) unless marked otherwise.
#: Protocol: official 10k/1k/1k split, official test MAE, single model where the
#: source reports one.  These numbers are NEVER compared with internal Monitor
#: numbers in this round; they are listed so that the distance between the
#: project's internal protocol and the literature is explicit.
LITERATURE_ZINC12K: list[dict[str, Any]] = [
    {
        "name": "ESA+",
        "mae": 0.052,
        "pm": 0.001,
        "params": "n/a",
        "source": "Buterez et al. 2025, Nat Commun, DOI 10.1038/s41467-025-60252-z, Table 6 (RWSE + ring features)",
        "verified_in_repo": "notes/prior_work_boundary.md §0",
    },
    {
        "name": "CSA-rings",
        "mae": 0.056,
        "pm": 0.002,
        "params": "~?",
        "source": "arXiv 2304.10933",
        "verified_in_repo": "notes/prior_work_boundary.md",
    },
    {
        "name": "MoSE + GRIT",
        "mae": 0.056,
        "pm": 0.001,
        "params": "n/a",
        "source": "arXiv 2410.18676 (ICML 2025)",
        "verified_in_repo": "notes/prior_work_boundary.md",
    },
    {
        "name": "GRIT (RRWP)",
        "mae": 0.059,
        "pm": 0.002,
        "params": "~500k",
        "source": "ICML 2023, PMLR v202:23321",
        "verified_in_repo": "notes/prior_work_boundary.md",
    },
    {
        "name": "SPSE + GRIT",
        "mae": 0.059,
        "pm": 0.001,
        "params": "n/a",
        "source": "arXiv 2502.09365",
        "verified_in_repo": "notes/prior_work_boundary.md",
    },
    {
        "name": "GPM (Neural Graph Pattern Machine)",
        "mae": 0.064,
        "pm": 0.004,
        "params": "n/a",
        "source": "ICML 2025, PMLR v267",
        "verified_in_repo": "notes/prior_work_boundary.md",
    },
    {
        "name": "GraphGPS",
        "mae": 0.070,
        "pm": 0.004,
        "params": "~500k",
        "source": "Rampášek et al. 2022, GraphGPS; number quoted in results/phase1b/REPORT.md and phase0 REPORT",
        "verified_in_repo": "results/phase1b/REPORT.md:203, results/phase0/REPORT.md:103 (secondary citation, not re-verified this round)",
    },
    {
        "name": "GSN (substructure counts)",
        "mae": 0.101,
        "pm": 0.010,
        "params": "~500k",
        "source": "arXiv 2006.09252 (TPAMI 2022)",
        "verified_in_repo": "notes/prior_work_boundary.md",
    },
]


def _as_int_list(values: Sequence[Any]) -> list[int]:
    return [int(v) for v in values]


def _overlap_payload(fit_rows: Sequence[int], monitor: np.ndarray, reserved: np.ndarray) -> dict[str, Any]:
    f = set(_as_int_list(fit_rows))
    m = set(int(x) for x in monitor)
    r = set(int(x) for x in reserved)
    return {
        "n_fit_rows": len(f),
        "monitor_overlap": len(f & m),
        "monitor_overlap_fraction": len(f & m) / max(len(m), 1),
        "reserved_overlap": len(f & r),
        "reserved_overlap_fraction": len(f & r) / max(len(r), 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)

    split = C.read_json(C.SPLIT_PATH)
    monitor = np.asarray(split["rows"]["monitor"], dtype=np.int64)
    reserved = np.asarray(split["rows"]["reserved"], dtype=np.int64)
    fit = np.asarray(split["rows"]["fit"], dtype=np.int64)

    # ---------------------------------------------------------------- B1.1
    checkpoint_audit: list[dict[str, Any]] = []

    def add(
        name: str,
        description: str,
        supervised_rows: Sequence[int] | None,
        provenance: str,
        note: str = "",
    ) -> None:
        entry: dict[str, Any] = {
            "name": name,
            "description": description,
            "provenance": provenance,
            "note": note,
        }
        if supervised_rows is None:
            entry["supervised_rows"] = "ALL_OFFICIAL_TRAIN_10000"
            entry["contamination"] = {
                "n_fit_rows": 10000,
                "monitor_overlap": int(monitor.size),
                "monitor_overlap_fraction": 1.0,
                "reserved_overlap": int(reserved.size),
                "reserved_overlap_fraction": 1.0,
            }
        else:
            entry["contamination"] = _overlap_payload(supervised_rows, monitor, reserved)
        entry["usable_for_internal_monitor"] = bool(
            entry["contamination"]["monitor_overlap"] == 0
        )
        checkpoint_audit.append(entry)

    add(
        "BJOC-P3 full-10k (exposed official test 0.069822)",
        "masked-SSL encoder on all 10k official train + R3 360-epoch cosine, last-5 soup, 413,505 params",
        None,
        "tracks/structure-contribution/results/bjoc_r3_frozen_terminal_test_v1/REPORT.md",
        "supervised on every official-train row, therefore on every SSRC Monitor row",
    )
    add(
        "BJOC-P0 full-10k",
        "same but R0 240-epoch constant LR",
        None,
        "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/REPORT.md",
    )
    if (BJOC_ARCHIVE / "split_B.json").exists():
        sb = C.read_json(BJOC_ARCHIVE / "split_B.json")
        add(
            "BJOC-P3 split-B (internal 8k/1k/1k)",
            "masked-SSL (Fit-B only) + R3; Monitor-B MAE 0.114942 (2-seed mean)",
            sb["fit_rows"],
            "tracks/structure-contribution/results/bjoc_performance_mechanism_long_v1/REPORT.md §B2",
            "different split salt and different Monitor molecules from SSRC; usable only as a cross-split reference",
        )
    if (BJOC_MASKED / "split_a.json").exists():
        sa = C.read_json(BJOC_MASKED / "split_a.json")
        add(
            "BJOC masked-pretraining split-A arms (P/B on Fit-A)",
            "masked SSL + supervised BJOC-Null on the history split-A 8k fit block",
            sa["fit_rows"],
            "tracks/structure-contribution/results/bjoc_masked_pretraining_v1/",
        )
    if (SC / "bjoc_predictive_bottleneck_v1" / "split_a.json").exists():
        sa2 = C.read_json(SC / "bjoc_predictive_bottleneck_v1" / "split_a.json")
        add(
            "BJOC predictive-bottleneck N8/O8 arms",
            "BJOC-Null on the bottleneck split-A 8k fit block",
            sa2["fit_rows"],
            "tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1/REPORT.md",
        )

    any_usable = [c for c in checkpoint_audit if c["usable_for_internal_monitor"]]

    # ---------------------------------------------------------------- B1.2
    internal = {
        "protocol": "SSRC internal: official-train 10k, frozen permutation split 8000/1000/1000, label-free preprocessing, Monitor scored once per frozen run",
        "monitor_rows": int(monitor.size),
        "rows": {
            "G_ref_seed0": 0.165290,
            "R1_size_seed0": 0.236575,
            "R1_size_seed1": 0.281039,
            "R1_size_cap_seed0": 0.193475,
            "R_incidence_seed0": 0.279662,
            "S_cap_size_seed0": 0.289477,
            "R1_marginal_size_seed0": 0.304730,
            "M_keep_edge_seed0": 0.237006,
        },
        "route_conditioned_post_hoc": {"rdkit_large_ring_to_R1_size_cap": 0.192190},
        "comparable_with": ["itself", "the Stage B2 strong baseline, if it runs on this exact split"],
    }

    bjoc_splitb = None
    if (BJOC_ARCHIVE / "stage_b_monitor_B.json").exists():
        sbj = C.read_json(BJOC_ARCHIVE / "stage_b_monitor_B.json")
        bjoc_splitb = {
            "protocol": "BJOC internal split-B: same 10k official train, different grouped salt, Fit 8000 / Monitor 1000 / Locked 1000, Fit-only SSL",
            "monitor_rows": 1000,
            "seed_mean_mae": {
                "B0_scratch_R0": sbj["contrasts"]["P0_minus_B0"]["mae_ref_seed_mean"],
                "B3_scratch_R3": sbj["contrasts"]["B3_minus_B0"]["mae_new_seed_mean"],
                "P0_ssl_R0": sbj["contrasts"]["P3_minus_P0"]["mae_ref_seed_mean"],
                "P3_ssl_R3": sbj["contrasts"]["P3_minus_B3"]["mae_new_seed_mean"],
            },
            "comparable_with": ["itself"],
            "why_not_directly_comparable": "different Monitor molecules (121/1000 overlap), different split salt; same data regime only",
            "overlap_with_ssrc_monitor": 121,
        }

    exposed = {
        "protocol": "BJOC frozen P3 on the official ZINC subset test split (1000 rows), single frozen terminal evaluation",
        "exposure": "test already read earlier in this project (KSVD track 2026-10-03); this is an exposed-test measurement, not a blind test",
        "rows": {
            "P3_seed0": 0.069304,
            "P3_seed1": 0.070340,
            "P3_seed_mean": 0.069822,
            "P0_seed_mean": 0.073957,
        },
        "training_rows": "all 10,000 official-train rows + 9,996-row structure-only SSL",
        "comparable_with": ["the literature official-test column only in protocol, not in split realisation"],
    }

    # ---------------------------------------------------------------- B1.3
    b2_budget = {
        "recipe": {
            "name": "BJOC-Null + masked SSL + R3 360-epoch cosine",
            "params": 413505,
            "ssl_updates": 7560,
            "supervised_updates_8k": 22680,
            "source": "tracks/structure-contribution/results/bjoc_performance_mechanism_long_v1/",
            "seed_replication": "2 seeds on split-B, SSL gain replicated across two data regimes and two recipes",
        },
        "measured_gpu_seconds": {
            "ssl_pretrain_8k": [1630.36, 1639.53],
            "supervised_360ep_8k": [510.82, 515.63],
        },
        "projected_gpu_hours_single_seed": (1630.36 + 510.82) / 3600.0,
        "cap_gpu_hours": 2.0,
        "fits_cap": bool((1630.36 + 510.82) / 3600.0 < 2.0),
        "isolation_requirements": [
            "objects/standardiser built from SSRC Fit 8,000 rows only",
            "masked SSL encoder trained on SSRC Fit 8,000 rows only",
            "supervised training on SSRC Fit 8,000 rows only, raw y",
            "Monitor predictions hashed before the single Monitor label read",
            "no official valid/test, no internal Reserved, no early stopping on Monitor",
        ],
        "why_execute": (
            "no existing checkpoint can be scored on the SSRC Monitor without "
            "label contamination, and the internal G_ref is a 6-layer local GINE "
            "that is ~0.05-0.07 MAE weaker than a published strong model on the "
            "official protocol; without a matched strong baseline the 'gap to "
            "competitiveness' question cannot be answered inside one protocol"
        ),
    }

    # Stage B2 result, when it exists, joins the *internal* table only.
    strong_path = C.RESULTS_1E / "strong_baseline.json"
    if strong_path.exists():
        strong = C.read_json(strong_path)
        internal["rows"]["P3E_strong_baseline_seed0"] = strong["monitor_mae"]
        internal["stage_b2_arm"] = {
            "recipe": strong["recipe"],
            "params": strong["params"],
            "seed": strong["seed"],
            "prediction_sha256": strong["prediction_sha256"],
            "information_regime": (
                "full graph + junction objects + structure-only global descriptors "
                "(incl. ring-size fractions); Fit-only masked SSL"
            ),
            "not_like_for_like_with_ssrc_arms": True,
        }

    payload = {
        "protocol_id": "ssrc-phase1e-zinc-v1",
        "stage": "B1_reference_audit",
        "scope": {"reserved_labels_read": 0, "official_valid_test_read": 0},
        "contamination_audit": {
            "ssrc_monitor_rows": int(monitor.size),
            "ssrc_reserved_rows": int(reserved.size),
            "ssrc_fit_rows": int(fit.size),
            "checkpoints": checkpoint_audit,
            "n_usable_for_internal_monitor": len(any_usable),
            "usable_names": [c["name"] for c in any_usable],
            "conclusion": (
                "no historical strong-model checkpoint can be scored on the SSRC "
                "Monitor without label contamination"
                if not any_usable
                else "a historical checkpoint with zero Monitor overlap exists"
            ),
        },
        "protocol_tables": {
            "internal_ssrc_monitor": internal,
            "bjoc_internal_split_b": bjoc_splitb,
            "bjoc_exposed_official_test": exposed,
            "literature_official_zinc12k": LITERATURE_ZINC12K,
        },
        "ranking_policy": (
            "the four tables above use different splits, different training-set sizes "
            "and different exposure histories. No number from one table may be "
            "subtracted from, ratioed with, or ranked against a number from another."
        ),
        "b2_budget": b2_budget,
        "b2_status": "EXECUTED" if strong_path.exists() else "NOT_EVALUATED",
    }
    C.write_json(OUT, payload)
    print(f"[B1] checkpoints audited: {len(checkpoint_audit)}; usable: {len(any_usable)}")
    for c in checkpoint_audit:
        print(
            f"[B1]   {c['name'][:52]:52s} monitor_overlap="
            f"{c['contamination']['monitor_overlap']:4d} "
            f"({c['contamination']['monitor_overlap_fraction']:.3f})"
        )
    print(
        f"[B1] projected B2 cost {b2_budget['projected_gpu_hours_single_seed']:.3f} GPU-h "
        f"(cap {b2_budget['cap_gpu_hours']})"
    )
    print(f"[B1] wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
