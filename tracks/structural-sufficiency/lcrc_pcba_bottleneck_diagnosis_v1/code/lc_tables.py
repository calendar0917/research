"""Compact report tables (Markdown) built from the analysis JSON/CSV outputs.

Only pre-computed artifacts are read; no new statistic is introduced here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import lc_common as C

OUT_PATH = C.ANALYSIS / "report_tables.md"


def _f(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):+.{nd}f}" if nd else f"{float(x):.{nd}f}"


def _pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{100.0 * float(x):.1f}%"


def build() -> str:
    inv = C.read_json(C.ANALYSIS / "prediction_inventory.json")
    sim = C.read_json(C.ANALYSIS / "prediction_similarity_summary.json")
    nov = C.read_json(C.ANALYSIS / "novelty_vs_gain_summary.json")
    macro = C.read_json(C.ANALYSIS / "macro_ap_decomposition.json")
    struct = C.read_json(C.ANALYSIS / "structure_support_summary.json")
    task_sets = C.read_json(C.ANALYSIS / "task_sets.json")

    lines: list[str] = []
    lines.append("# Report tables (auto-generated)\n")
    lines.append(f"- round: `{C.ROUND_ID}`")
    lines.append(f"- label: `{C.RESULT_LABEL}`\n")

    # T1 inventory
    lines.append("## T1. Frozen checkpoints and replay gates\n")
    lines.append("| model | epoch | Valid AP (replay / recorded) | Test AP (replay / recorded) | Valid Δ | Test Δ |")
    lines.append("|---|---|---|---|---|---|")
    for label in C.MODEL_ORDER:
        m = inv["models"][label]
        lines.append(
            f"| {label} | {m['checkpoint_epoch']} | {m['valid']['ap_official']:.9f} / "
            f"{m['valid']['ap_recorded']:.9f} | {m['test']['ap_official']:.9f} / "
            f"{m['test']['ap_recorded']:.9f} | {m['valid']['ap_replay_abs_delta']:.1e} | "
            f"{m['test']['ap_replay_abs_delta']:.1e} |"
        )
    lines.append("")

    # T2 similarity
    lines.append("## T2. P1 prediction similarity (HIGH_SUPPORT tasks)\n")
    lines.append(
        "| pair | split | median rho [q1, q3] | median Jaccard K1 | median Jaccard K5 | "
        "pos in K1 (both / only new / only ref) | median Δpct of positives |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for pkey, p in sim["pairs"].items():
        for split in ("valid", "test"):
            s = p[split]["spearman"]["spearman_high_support"]
            t = p[split]["topk"]
            hs = t["high_support_sums"]
            lines.append(
                f"| {pkey} | {split} | {s['median']:.3f} [{s['q1']:.3f}, {s['q3']:.3f}] | "
                f"{t['jaccard_k1_high_support']['median']:.3f} | {t['jaccard_k5_high_support']['median']:.3f} | "
                f"{hs['both_pos_k1']} / {hs['only_new_pos_k1']} / {hs['only_reference_pos_k1']} | "
                f"{_f(t['high_support_median_delta_pct']['median'], 4)} |"
            )
    lines.append("")
    lines.append("Verdicts: " + "; ".join(f"`{k}` → `{v['verdict']['label']}`" for k, v in sim["pairs"].items()))
    lines.append("")

    # T3 novelty groups
    lines.append("## T3. P2 Train-only WL2 novelty groups\n")
    lines.append("| split | group | molecules | positives | median heavy atoms | tasks with group-pos ≥20 |")
    lines.append("|---|---|---|---|---|---|")
    for split in ("valid", "test"):
        df = pd.read_csv(C.ANALYSIS / f"structure_support_{split}.csv")
        # labels are re-read through the frozen loaders (no new information)
        import lc_data as D

        yy = D.split_labels(split)
        groups = df["novelty_group"].to_numpy()
        for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL"):
            m = groups == g
            pos = np.nansum(np.asarray(yy[m], dtype=np.float64), axis=0)
            lines.append(
                f"| {split} | {g} | {int(m.sum())} | {int(np.nansum(pos))} | "
                f"{np.median(df['n_heavy_atoms'].to_numpy()[m]):.0f} | {int((pos >= 20).sum())} |"
            )
    lines.append("")

    # T4 novelty vs gain (shared tasks)
    lines.append("## T4. P3 novelty vs gain (task-matched shared main cells)\n")
    lines.append("| pair | group | shared tasks | mean ΔAP Valid | mean ΔAP Test | decay (Test−Valid) | improved Valid → Test |")
    lines.append("|---|---|---|---|---|---|---|")
    for pkey, p in nov["pairs"].items():
        t = p["trend_shared_tasks"]
        for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL"):
            c = t["per_group"][g]
            lines.append(
                f"| {pkey} | {g} | {c['n_shared_tasks']} | {_f(c['mean_delta_valid'])} | "
                f"{_f(c['mean_delta_test'])} | {_f(c['decay_test_minus_valid'])} | "
                f"{c['n_improved_valid']}/{c['n_shared_tasks']} → {c['n_improved_test']}/{c['n_shared_tasks']} |"
            )
    lines.append("")
    lines.append("Verdicts: " + "; ".join(f"`{k}` → `{v['verdict']['label']}`" for k, v in nov["pairs"].items()))
    lines.append("")

    # T5 macro decomposition
    lines.append("## T5. P4 macro-AP decomposition (Test unless stated)\n")
    lines.append(
        "| pair | mean ΔAP Valid | mean ΔAP Test | high-support (≥20) mean ΔAP | improved (high) | "
        "low-support (≤19) sum ΔAP | low-support share of |Δ| | Valid+ → Test− | single largest ±ΔAP |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for pkey, p in macro["pairs"].items():
        v = p["splits"]["valid"]
        t = p["splits"]["test"]
        sa = p["support_asymmetry"]
        low_sum = sa["test_low_support_sum"]
        rev = macro["valid_test_reversals"][pkey]
        top_pos = t["top_effects"]["top_positive"][0]
        top_neg = t["top_effects"]["top_negative"][0]
        lines.append(
            f"| {pkey} | {_f(v['mean_delta_ap'])} | {_f(t['mean_delta_ap'])} | "
            f"{_f(t['strata']['ge20']['mean_delta_ap'])} | "
            f"{t['strata']['ge20']['n_improved']}/{t['strata']['ge20']['n_tasks']} | "
            f"{_f(low_sum)} | {_pct(sa['test_low_support_share_of_abs_movement'])} | "
            f"{rev['n_valid_positive_test_negative']}/{rev['n_tasks_both']} | "
            f"+{top_pos['delta_ap']:.3f} (t{top_pos['task']}, n={top_pos['positive']}) / "
            f"{top_neg['delta_ap']:.3f} (t{top_neg['task']}, n={top_neg['positive']}) |"
        )
    lines.append("")
    lines.append(
        "Low-support stratum = Test positives 5–19 (18 tasks) + 1–4 (1 task). "
        "Top support-asymmetry labels: "
        + "; ".join(f"`{k}` → `{v['support_asymmetry']['label']}`" for k, v in macro["pairs"].items())
    )
    lines.append("")

    # T6 task-set counts
    lines.append("## T6. Frozen task sets\n")
    lines.append("| set | n |")
    lines.append("|---|---|")
    for key, val in task_sets["n"].items():
        lines.append(f"| {key} | {val} |")
    lines.append("")
    lines.append(
        f"Tasks excluded from HIGH_SUPPORT although Test positives ≥20: "
        f"{task_sets['TEST_HIGH_BUT_VALID_LOW_SUPPORT']} (Validation positives 13–14)."
    )
    lines.append("")
    lines.append("## T7. Structural support build\n")
    ts = struct["train_support_meta"]
    lines.append(
        f"- Train molecules: {ts['n_train_molecules']}, Train atoms: {ts['n_train_atoms']}, "
        f"distinct WL2: {ts['n_distinct_wl2']} ({ts['distinct_with_support_1']} appear in one molecule, "
        f"{ts['distinct_with_support_le_4']} in ≤4), build {ts['build_seconds']:.0f}s"
    )
    lines.append(f"- parse modes (Train): {ts['parse_modes']}")
    lines.append(f"- labels read: {struct['labels_read']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    text = build()
    OUT_PATH.write_text(text, encoding="utf-8")
    print(text[:1200])
    print(f"...\n[report_tables] -> {C.rel(OUT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
