"""Render the report tables (markdown) from the frozen analysis JSONs."""

from __future__ import annotations

import json
from typing import Any, Mapping

import psc_common as C
import psc_analysis as A

OUT = A.ANALYSIS / "tables.md"


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "nan"
    return f"{v:.{nd}f}"


def ci(d: Mapping[str, Any]) -> str:
    lo, hi = d["ci95"]
    return f"{f(d['point'], 4)} [{f(lo, 4)}, {f(hi, 4)}]"


def main() -> int:
    summary = json.loads((A.RESULTS / "summary.json").read_text(encoding="utf-8"))
    lock = C.read_json(A.ANALYSIS / "selection_lock.json")
    boot = C.read_json(A.ANALYSIS / "bootstrap.json") if (A.ANALYSIS / "bootstrap.json").is_file() else {}
    boot_d1 = (C.read_json(A.ANALYSIS / "bootstrap_D1.json")
               if (A.ANALYSIS / "bootstrap_D1.json").is_file() else {})
    valid = (C.read_json(A.ANALYSIS / "final_valid.json")
             if (A.ANALYSIS / "final_valid.json").is_file() else None)
    upm = (C.read_json(A.ANALYSIS / "unique_pair_metrics.json")
           if (A.ANALYSIS / "unique_pair_metrics.json").is_file() else None)
    decomp = (C.read_json(A.ANALYSIS / "pair_residual_decomposition.json")
              if (A.ANALYSIS / "pair_residual_decomposition.json").is_file() else None)
    ledger = (C.read_json(A.ANALYSIS / "sum_edit_ledger.json")
              if (A.ANALYSIS / "sum_edit_ledger.json").is_file() else None)
    sizes = (C.read_json(A.ANALYSIS / "size_strata.json")
             if (A.ANALYSIS / "size_strata.json").is_file() else None)
    budget = (C.read_json(A.ANALYSIS / "compute_budget.json")
              if (A.ANALYSIS / "compute_budget.json").is_file() else None)
    lines: list[str] = ["# PSC result tables (auto-generated)", ""]

    def table(header: list[str], rows: list[list[str]]) -> None:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(r) + " |")
        lines.append("")

    ref = lock["ACRA_B_star"]
    for split in ("D0", "D1"):
        agg = summary["aggregate"].get(split, {})
        if not agg:
            continue
        lines.append(f"## {split} held-out whole-molecule accuracy")
        lines.append("")
        rows = []
        order = [a for a in A.PSC_ARMS] + ["ACRA_REF", "ACRA_PAIR", "GINE_REF"] + \
                list(A.CPU_MODELS) + [a for a in A.ARM_SPEC
                                      if ARM_SPEC_TARGET(a) == "DIAG"]
        for arm in order:
            if arm not in agg:
                continue
            v = agg[arm]
            spec = A.ARM_SPEC.get(arm, {})
            delta = ""
            key = f"{arm}_vs_{ref}"
            contrasts = boot.get("contrasts", {}) if split == "D0" else boot_d1.get("contrasts", {})
            if key in contrasts:
                delta = ci(contrasts[key])
            rows.append([arm, spec.get("input", "–"), spec.get("readout", "–"),
                         spec.get("target", "–"), str(v["n_seeds"]),
                         f"{f(v['rmse_mean'])} ± {f(v['rmse_sd'])}",
                         f(v["mae_mean"]), delta])
        table(["arm", "input", "readout", "target", "seeds", "RMSE mean ± sd", "MAE",
               f"RMSE {ref} − arm [95% CI]"], rows)
        lines.append(f"Reference for the last column: {ref}. "
                     "Positive Δ means the arm is better than the reference.")
        lines.append("")

    if valid:
        lines.append("## Official Valid (scored once after the freeze)")
        lines.append("")
        rows = []
        for m, v in valid["models"].items():
            if not v.get("seeds"):
                rows.append([m, str(v.get("status")), "–", "–", "–", "–"])
                continue
            b = v.get("bootstrap_vs_ACRA_B")
            rows.append([m, str(v["n_seeds"]), f"{f(v['rmse_mean'])} ± {f(v['rmse_sd'])}",
                         f(v["mae_mean"]),
                         ci(b) if b else "–",
                         ", ".join(f(s["rmse"]) for s in v["seeds"])])
        table(["model", "seeds", "RMSE mean ± sd", "MAE", "ACRA_B* − model [95% CI]",
               "per-seed RMSE"], rows)
        lines.append("")

    for name, data in (("D0", boot), ("D1", boot_d1)):
        if not data.get("contrasts"):
            continue
        lines.append(f"## Pre-specified contrasts, {name} (paired molecule bootstrap)")
        lines.append("")
        rows = []
        for key, v in data["contrasts"].items():
            rows.append([v["question"], f"{v['candidate']} vs {v['reference']}",
                         ci(v), f(v["p_two_sided"], 3)])
        table(["question", "candidate vs reference", "Δ RMSE [95% CI]", "p"], rows)
        sc = data.get("scaffold_cluster_bootstrap") or {}
        if sc and "error" not in sc:
            lines.append(f"Scaffold-cluster bootstrap ({name}, one resample per Bemis–Murcko "
                         "scaffold cluster, same statistic):")
            lines.append("")
            rows = [[k, ci(v), str(v.get("n_clusters"))] for k, v in sc.items()]
            table(["contrast", "Δ RMSE [95% CI]", "clusters"], rows)
        lines.append("")

    if upm:
        for split in ("D0", "D1"):
            d = upm["splits"].get(split)
            if not d:
                continue
            lines.append(f"## Pair-level metrics, {split}")
            lines.append("")
            lines.append("### By matched-pair scope (unique unordered pairs, unweighted)")
            lines.append("")
            models = [m for m in d["scopes"].get("heldout_heldout", {})
                      if m not in ("n_pairs", "n_components")]
            rows = []
            for scope, v in d["scopes"].items():
                for m in models:
                    got = v.get(m, {})
                    ms = got.get("mean_over_seeds", {})
                    rows.append([scope, m, str(got.get("n_scored", "–")),
                                 f(ms.get("mae")), f(ms.get("rmse")),
                                 f(ms.get("direction_accuracy"), 3)])
            table(["scope", "model", "pairs scored", "Δ MAE", "Δ RMSE", "direction acc."], rows)
            lines.append("### By support class (heldout × heldout, cut-row weighted)")
            lines.append("")
            rows = []
            for cls, v in d["support_classes"].items():
                for m in models:
                    if m in v and isinstance(v[m], dict) and "mae" in v[m]:
                        rows.append([cls, str(v["n_pairs_any"]), f(v["weight_mass"], 1), m,
                                     f(v[m]["mae"]), f(v[m].get("rmse")),
                                     f(v[m].get("direction_accuracy"), 3)])
            table(["support class", "pairs", "weight", "model", "Δ MAE", "Δ RMSE",
                   "direction acc."], rows)
            for scope in ("fit_fit", "heldout_heldout"):
                nov = d["novelty"].get(scope)
                if not nov:
                    continue
                for label, buckets in nov.items():
                    lines.append(f"### Novelty ({label}) — {scope}, {split}")
                    lines.append("")
                    rows = []
                    for bname, v in buckets.items():
                        for m in models:
                            if m in v and isinstance(v[m], dict) and "mae" in v[m]:
                                rows.append([bname, str(v["n_pairs"]), str(v["n_components"]),
                                             m, f(v[m]["mae"]), f(v[m].get("rmse"))])
                    table(["bucket", "pairs", "components", "model", "Δ MAE", "Δ RMSE"], rows)
            s = d.get("delta_sensitivity_0p25", {})
            if s:
                lines.append(f"### |Δ| ≥ 0.25 sensitivity ({split} heldout pairs)")
                lines.append("")
                rows = [[m, f(v["mae"]), f(v["rmse"]), f(v.get("direction_accuracy"), 3)]
                        for m, v in s.items() if isinstance(v, dict)]
                table(["model", "Δ MAE", "Δ RMSE", "direction acc."], rows)
            lines.append(f"Counts: {json.dumps(d['counts'])}")
            lines.append("")

    if decomp:
        lines.append("## Pair-residual decomposition (D0 heldout × heldout, component weights)")
        lines.append("")
        rows = []
        for m, v in decomp["models"].items():
            mean = v["mean"]
            rows.append([m, f(mean["mse_delta"]), f(mean["mean_ea2"]), f(mean["mean_eb2"]),
                         f(mean["cross"]), f(mean.get("corr_endpoint_residuals"), 3),
                         f(v["check_identity"], 1e-9) if False else f"{v['check_identity']:.1e}"])
        table(["model", "E[(e_b−e_a)²]", "E[e_a²]", "E[e_b²]", "E[e_a e_b]",
               "corr(e_a,e_b)", "identity err"], rows)
        lines.append("")

    if sizes:
        lines.append("## Molecule-size strata (D0 dev, seed 0) and worst-error sensitivity")
        lines.append("")
        rows = []
        for m, v in sizes["models"].items():
            rows.append([m, f(v["overall_rmse"]), f(v["overall_mae"])]
                        + [f(v[f"q{i}"]["rmse"]) for i in range(1, 5)]
                        + [f(v["top5_removed_rmse"])])
        table(["model", "overall RMSE", "MAE", "Q1", "Q2", "Q3", "Q4",
               "top-5 removed"], rows)
        lines.append(f"Quartile edges (fit atom counts): {sizes['quartile_edges']}")
        uni = sizes.get("top5_union_removed")
        if uni:
            lines.append("")
            rows = [[m, f(v["rmse"])] for m, v in uni["models"].items()]
            table([f"model (union of {uni['n_removed']} worst removed)", "RMSE"], rows)
        lines.append("")

    if ledger and ledger.get("examples"):
        lines.append("## SUM edit ledger cases")
        lines.append("")
        rows = []
        for e in ledger["examples"]:
            d = e.get("decomposition") or {}
            rows.append([e["rule"], str(e["a"]), str(e["b"]), f(e["delta_true"]),
                         f(e["delta_ref_pred"]), f(e.get("eta"), 3),
                         str(e.get("T_seen")), str(e.get("frag_seen")),
                         f(d.get("delta_F0_S")), f(d.get("delta_F0_C")),
                         f(d.get("delta_X"))])
        table(["rule", "a", "b", "true Δ", "ACRA_B* Δ", "eta", "T seen", "frags seen",
               "ΔF_S", "ΔF_C", "ΔF_X"], rows)
        lines.append(f"Ledger model: {ledger['ledger_model']} "
                     f"(champion {ledger['champion']}); identity error "
                     f"{ledger['identity_max_abs_std']:.2e} (standardised units).")
        lines.append("")

    if budget:
        lines.append("## Compute budget")
        lines.append("")
        t = budget["totals"]
        table(["item", "value"], [
            ["GPU-hours (all launched jobs)", f(t["gpu_hours_including_attempts"], 3)],
            ["GPU-hours (completed runs)", f(t["gpu_hours_completed_runs"], 3)],
            ["train-loop hours", f(t["train_loop_hours"], 3)],
            ["train-loop fraction of GPU wall", f(t["train_loop_fraction"], 3)],
            ["CPU-hours (closed-form baselines)", f(t["cpu_hours_closed_form"], 3)],
            ["budget", "16 GPU-hours"],
        ])
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


def ARM_SPEC_TARGET(arm: str) -> str:
    return A.ARM_SPEC.get(arm, {}).get("target", "")


if __name__ == "__main__":
    raise SystemExit(main())