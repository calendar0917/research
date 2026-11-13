"""Stage C: cross-part summary tables and the three main figures.

Reads only the committed Part-A and Part-B outputs; produces:

* ``summary_tables.json`` — the extracted headline numbers (Part A claims,
  WC_LOGP gain matrix, MOLWT invariance) plus the source hashes.
* ``fig_partA_mechanism.png`` — Part A same-population mechanism contrast.
* ``fig_wc_logp_delta.png`` — WC_LOGP delta_pred vs delta_ref (Test, main CF).
* ``fig_task_response.png`` — scale-normalised response of both tasks on the
  identical CF set, with the MOLWT invariant band.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import round_common as RC


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        s = str(x).strip()
        return float(s) if s not in ("", "nan", "None") else d
    except Exception:
        return d


def _load() -> dict[str, Any]:
    out: dict[str, Any] = {"partA": RC.read_json(RC.NEW_ANALYSIS / "partA_clarifications.json")}
    for task in RC.TASKS:
        mpath = RC.NEW_ANALYSIS / f"partB_{task}_metrics.json"
        ppath = RC.NEW_ANALYSIS / f"partB_{task}_pairs.csv"
        out[task] = {"metrics": RC.read_json(mpath) if mpath.exists() else None,
                     "pairs": RC.read_csv(ppath) if ppath.exists() else []}
    return out


def summary_table(data: dict[str, Any]) -> dict[str, Any]:
    partA = data["partA"]
    summary: dict[str, Any] = {
        "round_id": RC.ROUND_ID,
        "partA": {
            "coverage_total": partA["A1_coverage_and_claim_matrix"]["_total"],
            "three_tables_amide_test": {
                k: partA["A2_same_population_three_tables"][f"amide/test/{k}"]
                for k in ("ALL_K0", "RETAINED_K0", "RETAINED_C")
            },
            "recovery_residual_max": partA["A3_conditional_recovery"]["arithmetic_residual_max"],
            "recovery_pass": partA["A3_conditional_recovery"]["arithmetic_pass"],
        },
        "partB": {},
    }
    for task in RC.TASKS:
        m = data[task]["metrics"]
        if m is None:
            summary["partB"][task] = {"available": False}
            continue
        fams = {}
        for fam in RC.SC.FAMILIES:
            b = m["family_stats"].get(f"{task}/test/{fam}")
            if b and "gain" in b:
                fams[fam] = {
                    "n_molecules": b["n_molecules"], "n_pairs": b["n_pairs"],
                    "delta_mae": b["delta_mae"], "baseline": b["baseline_zero_change"],
                    "gain": b["gain"], "gain_ci95": b["gain_ci95"],
                    "gain_bonferroni": b.get("gain_bonferroni"),
                    "sign_agreement": b["sign_agreement"],
                    "new_graph_mae": b["new_graph_mae_vs_own_oracle"],
                }
        summary["partB"][task] = {
            "available": True,
            "units": m["units"],
            "original_graph_metrics": m["original_graph_metrics"],
            "test_families": fams,
            "bonferroni": m["bonferroni"],
        }
    return summary


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_partA(data: dict[str, Any], path: Path) -> None:
    agg = data["partA"]["A3_conditional_recovery"]["aggregate"]
    fams = [f for f in RC.SC.FAMILIES if agg.get(f"{f}/test/RETAINED_C", {}).get("n_pairs", 0) > 0]
    labels = ["R3 scalar recovery (H)", "R2 S2 recovery (H)"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(fams))
    w = 0.38
    for i, q in enumerate(("R3_T_H", "R2_T_H")):
        pts = [agg[f"{f}/test/RETAINED_C"][q]["point"] for f in fams]
        lo = [agg[f"{f}/test/RETAINED_C"][q]["ci95"][0] for f in fams]
        hi = [agg[f"{f}/test/RETAINED_C"][q]["ci95"][1] for f in fams]
        ax.bar(x + (i - 0.5) * w, pts, w, label=labels[i],
               yerr=[np.asarray(pts) - np.asarray(lo), np.asarray(hi) - np.asarray(pts)], capsize=3)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=20, ha="right")
    ax.set_ylabel("H-space recovery (|p- p0| units)")
    ax.set_title("Part A: conditional scalar vs S2 recovery (Test, T vs matched C)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_wc_delta(data: dict[str, Any], path: Path) -> None:
    pairs = data["wc_logp"]["pairs"]
    sel = [r for r in pairs if r["split"] == "test" and str(r["main_cf"]) == "1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, view in zip(axes, ("T", "K")):
        xs = np.asarray([_f(r[f"delta_ref_{view}"]) for r in sel])
        ys = np.asarray([_f(r[f"delta_pred_{view}"]) for r in sel])
        ax.scatter(xs, ys, s=10, alpha=0.5)
        lim = [min(xs.min(), ys.min()) - 0.1, max(xs.max(), ys.max()) + 0.1]
        ax.plot(lim, lim, "k--", lw=1, label="y = x")
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel(f"delta_ref (WC_LOGP, {view})"); ax.set_ylabel(f"delta_pred ({view})")
        ax.set_title(f"WC_LOGP {view}: model response vs oracle change")
        ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_task_response(data: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # left: WC_LOGP gain per family
    m = data["wc_logp"]["metrics"]
    fams = [f for f in RC.SC.FAMILIES if f"wc_logp/test/{f}" in m["family_stats"]
            and "gain" in m["family_stats"][f"wc_logp/test/{f}"]]
    ax = axes[0]
    pts = [m["family_stats"][f"wc_logp/test/{f}"]["gain"] for f in fams]
    lo = [m["family_stats"][f"wc_logp/test/{f}"]["gain_ci95"][0] for f in fams]
    hi = [m["family_stats"][f"wc_logp/test/{f}"]["gain_ci95"][1] for f in fams]
    ax.errorbar(range(len(fams)), pts, yerr=[np.asarray(pts) - np.asarray(lo), np.asarray(hi) - np.asarray(pts)],
                fmt="o", capsize=3)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=20, ha="right")
    ax.set_ylabel("WC_LOGP delta-MAE gain (baseline - model)")
    ax.set_title("WC_LOGP: gain over the zero-change baseline")
    # right: MOLWT |delta_pred| / target std distribution
    ax = axes[1]
    mm = data["molwt"]["metrics"]
    fams2 = [f for f in RC.SC.FAMILIES if f"molwt/test/{f}" in mm["family_stats"]
             and "abs_delta_pred" in mm["family_stats"][f"molwt/test/{f}"]]
    med = [mm["family_stats"][f"molwt/test/{f}"]["abs_delta_pred"]["median_over_target_std"] for f in fams2]
    p95 = [mm["family_stats"][f"molwt/test/{f}"]["abs_delta_pred"]["p95_over_target_std"] for f in fams2]
    x = np.arange(len(fams2))
    ax.bar(x - 0.2, med, 0.4, label="median |delta_pred| / target std")
    ax.bar(x + 0.2, p95, 0.4, label="p95 |delta_pred| / target std")
    ax.set_xticks(x); ax.set_xticklabels(fams2, rotation=20, ha="right")
    ax.set_ylabel("scale-normalised response")
    ax.set_title("MOLWT (invariant): model response scale")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=str(RC.NEW_ANALYSIS))
    args = ap.parse_args(argv)
    data = _load()
    RC.write_json(RC.NEW_ANALYSIS / "summary_tables.json", summary_table(data))
    fig_partA(data, RC.NEW_ANALYSIS / "fig_partA_mechanism.png")
    if data["wc_logp"]["pairs"]:
        fig_wc_delta(data, RC.NEW_ANALYSIS / "fig_wc_logp_delta.png")
    if data["molwt"]["metrics"] is not None and data["wc_logp"]["metrics"] is not None:
        fig_task_response(data, RC.NEW_ANALYSIS / "fig_task_response.png")
    RC.log("figures done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
