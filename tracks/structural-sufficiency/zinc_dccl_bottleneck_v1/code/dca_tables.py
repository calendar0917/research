"""Render the phase-A tables to markdown (machine-generated from the JSON artifacts)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import dca_common as C


def main() -> int:
    g = json.loads((C.ANALYSIS / "groups.json").read_text())
    p2 = json.loads((C.ANALYSIS / "analysis_part2.json").read_text())
    cov = json.loads((C.ANALYSIS / "coverage_projection.json").read_text())
    pm = pd.read_csv(C.ANALYSIS / "per_molecule_features.csv")
    lines: list[str] = []
    add = lines.append

    add("# Phase A — machine-generated tables (DCC-L, official valid 1,000)\n")
    o = g["overall"]
    add(f"Full-sample MAE (seed mean) = **{o['mae_seed_mean']:.5f}** "
        f"(per seed {', '.join(f'{x:.5f}' for x in o['mae_per_seed'])}); "
        f"median AE {o['median_ae']:.4f}; p90 AE {o['p90_ae']:.4f}; "
        f"top-10 rows = {100 * o['top10_ae_share']:.1f}% of total AE, "
        f"top-100 = {100 * o['top100_ae_share']:.1f}%.\n")

    add("## Table 1 (MECE, 12 cells) — radius-2 support × extreme label × ring complexity\n")
    add("`U` = molecule has ≥1 atom whose exact typed radius-2 signature never occurs in train 10k; "
        "`T` = y ≤ −3; `R` = ring count. MAE is the seed mean; shares are of total absolute error.\n")
    add("| U | T | rings | n | n% | MAE | AE share | signed | seeds same sign | per-seed MAE |")
    add("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for c in g["tables_mece"]["U_x_T_x_R"]:
        add(f"| {'unseen' if c['r2_unseen'] else 'seen'} | {'y≤-3' if c['tail'] else 'ordinary'} | "
            f"{c['rings']} | {c['n']} | {100 * c['n_frac']:.1f}% | {c['mae']:.4f} | "
            f"{100 * c['ae_share']:.1f}% | {c['signed']:+.4f} | "
            f"{'yes' if c['seed_sign_consistent'] else 'no'} | "
            f"{' / '.join(f'{x:.3f}' for x in c['mae_per_seed'])} |")

    add("\n## Table 2 (MECE, 9 cells) — atom-count tertile (train cuts) × environment support tier\n")
    add("support tier: `full` = every atom's radius-2 signature has ≥5 train molecules; "
        "`low4` = none unseen but ≥1 atom with 1–4 train molecules; `unseen` = ≥1 atom with 0.\n")
    add("| size | support | n | MAE | AE share | signed |")
    add("|---|---|---:|---:|---:|---:|")
    for c in g["tables_mece"]["S_x_support"]:
        add(f"| {c['size']} | {c['support']} | {c['n']} | {c['mae']:.4f} | "
            f"{100 * c['ae_share']:.1f}% | {c['signed']:+.4f} |")

    add("\n## Table 3 (MECE, 8 cells) — radius-2 support × extreme label × size\n")
    add(f"size split at the train median n_atoms = {g['checks']['size_cuts_train_terciles'][1]:.0f}.\n")
    add("| U | T | large | n | MAE | AE share | signed |")
    add("|---|---|---|---:|---:|---:|---:|")
    for c in g["tables_mece"]["U_x_T_x_size"]:
        add(f"| {'unseen' if c['r2_unseen'] else 'seen'} | {'y≤-3' if c['tail'] else 'ordinary'} | "
            f"{c['large']} | {c['n']} | {c['mae']:.4f} | {100 * c['ae_share']:.1f}% | "
            f"{c['signed']:+.4f} |")

    add("\n## Dose response\n")
    add("| mean per-atom unseen fraction | n | MAE | AE share | signed | bias²/MSE |")
    add("|---|---:|---:|---:|---:|---:|")
    for r in p2["dose_response_unseen_frac"]:
        add(f"| {r['bin']} | {r['n']} | {r['mae']:.4f} | {100 * r['ae_share']:.1f}% | "
            f"{r['signed']:+.4f} | {r['bias_share_of_mse']:.4f} |")
    add("\n| fraction of atoms with ≥5 train examples | n | MAE | AE share |")
    add("|---|---:|---:|---:|")
    for r in p2["dose_response_supported_frac"]:
        add(f"| {r['bin']} | {r['n']} | {r['mae']:.4f} | {100 * r['ae_share']:.1f}% |")

    add("\n## Confounder control (radius-2 novelty vs absolute error)\n")
    c = g["confounders"]
    add("| statistic | value |")
    add("|---|---:|")
    for k in ("spearman_raw", "spearman_partial_n_atoms", "spearman_partial_natoms_cyclerank",
              "spearman_partial_natoms_cyclerank_absy", "delta_standardised_by_size_ring"):
        add(f"| {k} | {c[k]:+.4f} |")
    add(f"| per-seed ΔMAE (unseen − seen) | "
        f"{' / '.join(f'{x:+.4f}' for x in c['per_seed_delta_unseen_minus_seen'])} |")
    add(f"| ΔMAE after dropping rows with AE>1 | {c['drop_ae_gt1']['delta_unseen_minus_seen']:+.4f} "
        f"(n={c['drop_ae_gt1']['n']}) |")

    add("\n### Size × ring strata (unseen − seen), cells with ≥10 molecules on both sides\n")
    add("| size bin | ring bin | n unseen | n seen | MAE unseen | MAE seen | Δ |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for s in c["strata_delta_unseen_minus_seen"]:
        if s.get("delta") is None:
            continue
        add(f"| {s['size_bin']} | {s['ring_bin']} | {s['n_unseen']} | {s['n_seen']} | "
            f"{s['mae_unseen']:.4f} | {s['mae_seen']:.4f} | {s['delta']:+.4f} |")

    add("\n## Seed structure of the signed error\n")
    s = g["seed_structure"]
    add(f"Per-seed signed-error Pearson correlations (s0..s3): "
        f"min {s['min_pairwise_corr']:.3f}, mean {s['mean_pairwise_corr']:.3f}. "
        f"Fraction of rows where all four seeds share the error sign: "
        f"{100 * s['global_sign_agree_frac']:.1f}%.\n")
    add("| cell | n | mean signed error per seed | 4/4 same sign | per-row sign agreement |")
    add("|---|---:|---|---:|---:|")
    for c in g["tables_mece"]["U_x_T_x_R"]:
        lab = (f"{'unseen' if c['r2_unseen'] else 'seen'}/"
               f"{'y≤-3' if c['tail'] else 'ord'}/{c['rings']}")
        add(f"| {lab} | {c['n']} | "
            f"{' / '.join(f'{x:+.3f}' for x in c['signed_per_seed'])} | "
            f"{'yes' if c['seed_sign_consistent'] else 'no'} | {c['sign_agree_frac']:.2f} |")

    add("\n## Pre-fixed target-free support (whole-molecule similarity)\n")
    for k, v in p2["similarity"].items():
        add(f"**{k}** — {v['definition']}")
        add("")
        add("| quantity | value |")
        add("|---|---:|")
        for kk in ("knn_reference_mae", "median_abs_global", "spearman_top1_vs_nn_label_gap",
                   "spearman_simtopk_ae", "spearman_simtopk_ae_partial",
                   "spearman_nnystd_ae", "spearman_nnystd_ae_partial"):
            add(f"| {kk} | {v[kk]:+.4f} |")
        add(f"| top-1 similarity quantiles (0/10/50/90/100%) | "
            f"{', '.join(f'{x:.3f}' for x in v['sim_top1_quantiles'])} |")
        add("")
    add("### Support × neighbour-label dispersion (terciles, primary similarity)\n")
    add("| support tercile | dispersion tercile | n | MAE | AE share | kNN MAE | mean sim | mean nn y-std | unseen share |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in p2["support_x_dispersion"]["cells"]:
        add(f"| {c['support_bin']} | {c['dispersion_bin']} | {c['n']} | {c['mae']:.4f} | "
            f"{100 * c['ae_share']:.1f}% | {c['knn_mae']:.3f} | {c['sim_topk']:.3f} | "
            f"{c['nn_ystd']:.3f} | {100 * c['unseen_frac_within']:.0f}% |")
    add("\n### kNN reference vs DCC-L\n")
    add("| group | n | DCC-L MAE | kNN MAE (WL-IDF) | kNN MAE (Tanimoto) |")
    add("|---|---:|---:|---:|---:|")
    for r in p2["knn_vs_dcc"]:
        add(f"| {r['group']} | {r['n']} | {r['dcc_mae']:.4f} | {r['knn_mae']:.4f} | "
            f"{r['knn_mae_tan']:.4f} |")

    add("\n## Headroom arithmetic\n")
    h = p2["headroom"]
    add(f"- total MAE {h['mae_total']:.5f}; seen (749) {h['mae_seen']:.5f}; "
        f"unseen (251) {h['mae_unseen']:.5f}; unseen share of AE {100 * h['unseen_ae_share']:.1f}%")
    add(f"- if the unseen group matched the seen group, total MAE would be "
        f"{h['mae_if_unseen_equal_seen']:.5f}")
    add(f"- holding the seen group at {h['mae_seen']:.5f}, reaching 0.060 needs unseen = "
        f"{h['unseen_mae_needed_for_total_0.060_holding_seen']:.5f}")

    add("\n## Coverage projection (zero-GPU, structure only)\n")
    for k in ("fit4k", "fit8k", "train10k"):
        cc = cov["coverage"][k]
        add(f"- fit={k} ({cc['fit_molecules']} molecules): official-valid atom tiers "
            f"unseen {100 * cc['valid_atom_tier_share']['unseen']:.2f}% / "
            f"low4 {100 * cc['valid_atom_tier_share']['low4']:.2f}% / "
            f"supported {100 * cc['valid_atom_tier_share']['supported']:.2f}%; "
            f"molecules with any unseen environment {100 * cc['valid_molecule_any_unseen_frac']:.1f}%; "
            f"locked-dev atom unseen {100 * cc['locked_atom_tier_share']['unseen']:.2f}%")
    pr = cov["coverage_projection"]
    add(f"- fixed-tier-error projection 4k→8k on official valid: "
        f"{pr['fit4k']['fixed_tier_mae']:.5f} → {pr['fit8k']['fixed_tier_mae']:.5f} "
        f"(Δ {pr['projected_delta_8k_minus_4k_fixed_tier_error']:+.5f})")

    (C.ANALYSIS / "report_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {C.ANALYSIS / 'report_tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
