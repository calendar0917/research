"""Supplement analysis for ``zinc_shared_factor_supplement_v1`` (T vs N at seed 1).

Reads the artifacts of the supplement runs (``SF_T_s1``, ``SF_N_s1``) plus the
already-frozen seed-0 artifacts (``SF_T_s0``, ``SF_D_s0``) written by the same
runner, and the frozen grouping reference of ``zinc_restricted_adaptation_v1``.
Nothing is trained here and nothing official-test related is read.

Pre-registered questions and the decision rule live in ``PROTOCOL.md`` §4; this
script evaluates exactly those, with no extra threshold.

Outputs (``analysis/``):
  summary.json           metrics, contrasts, cells, Q1-Q3 numbers
  main_table.md          the small tables the report quotes
  preds_valid_TN_s1.csv  per-molecule valid predictions of the four arms
  valid_curves.csv       full per-epoch valid MAE curve of the four arms
  curves.png             the same curves, plotted
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
EXP = _HERE.parent
_TRACKS = EXP.parent
_SF = _TRACKS / "zinc_shared_factor_v1"
for _p in (str(_SF / "code"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sf_analysis as A  # noqa: E402  (shared metrics/contrasts/group helpers)

ANALYSIS = EXP / "analysis"
REF = _TRACKS / "zinc_restricted_adaptation_v1" / "analysis" / "reference_preds_valid.csv"
ARMS = [("D", 0), ("T", 0), ("T", 1), ("N", 1)]        # D0/T0 are frozen context
KEYS = [f"{a}_s{s}" for a, s in ARMS]
CELLS = ("seen_ordinary", "seen_tail", "unseen_ordinary", "unseen_tail")
ABSOLUTE_BAR = 0.075                                  # reference line only, not a re-gate
P3_CELL_MAE = 0.133939597                             # historical P3 on unseen+ordinary (n=225)
CONTRASTS = [("T1_minus_N1", "T_s1", "N_s1"), ("T1_minus_T0", "T_s1", "T_s0"),
             ("N1_minus_T0", "N_s1", "T_s0"), ("T0_minus_D0", "T_s0", "D_s0"),
             ("N1_minus_D0", "N_s1", "D_s0")]


# ---------------------------------------------------------------------------
# reference (P3) predictions and groupings
# ---------------------------------------------------------------------------


def load_reference() -> dict:
    with open(REF, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "row": np.asarray([int(r["row"]) for r in rows], dtype=np.int64),
        "y_raw": np.asarray([float(r["y_raw"]) for r in rows]),
        "pred_P3B_s0": np.asarray([float(r["pred_P3B_s0"]) for r in rows]),
        "r2_unseen_atom_frac": np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows]),
        "signature_unseen": np.asarray([int(r["signature_unseen"]) for r in rows]),
        "ordinary": np.asarray([int(r["ordinary_y_gt_minus3"]) for r in rows]),
    }


def align(ref: dict, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(ref["row"])
    g_row = ref["row"][order]
    pos = np.searchsorted(g_row, rows)
    if not np.array_equal(g_row[pos], rows):
        raise RuntimeError("reference rows do not match the prediction rows")
    return order[pos]


def cell_masks(ref: dict, pos: np.ndarray) -> dict[str, np.ndarray]:
    unseen = ref["r2_unseen_atom_frac"][pos] > 0.0
    ordinary = ref["ordinary"][pos] > 0
    return {"seen_ordinary": (~unseen) & ordinary, "seen_tail": (~unseen) & (~ordinary),
            "unseen_ordinary": unseen & ordinary, "unseen_tail": unseen & (~ordinary)}


# ---------------------------------------------------------------------------
# cells with an explicit contribution decomposition
# ---------------------------------------------------------------------------


def cell_table(preds: dict, ref: dict) -> dict:
    base = preds[KEYS[0]]
    rows = base["row"]
    pos = align(ref, rows)
    masks = cell_masks(ref, pos)
    n_total = rows.size
    ae = {k: np.abs(p["pred"] - p["y"]) for k, p in preds.items()}
    present = [k for k in KEYS if k in ae]
    out: dict = {"n_total": int(n_total), "cells": {}}
    for name in CELLS:
        m = masks[name]
        n = int(m.sum())
        cell: dict = {"n": n, "share": n / n_total}
        for k in present:
            cell.setdefault("mae", {})[k] = float(ae[k][m].mean())
            cell.setdefault("contribution", {})[k] = float(ae[k][m].sum() / n_total)
        cell["delta_mae"] = {}
        cell["delta_contribution"] = {}
        for label, a, b in CONTRASTS:
            if a not in ae or b not in ae:
                continue
            d = float(ae[a][m].mean() - ae[b][m].mean())
            cell["delta_mae"][label] = d
            cell["delta_contribution"][label] = float(d * n / n_total)
        cell["p3_descriptive_mae"] = float(np.abs(ref["pred_P3B_s0"][pos][m]
                                                 - ref["y_raw"][pos][m]).mean())
        out["cells"][name] = cell
    # exact decomposition check: sum of per-cell contributions == overall MAE
    out["checks"] = {}
    for k in present:
        total = float(sum(out["cells"][c]["contribution"][k] for c in CELLS))
        out["checks"][k] = {"overall_mae": float(ae[k].mean()),
                            "sum_of_cell_contributions": total,
                            "abs_diff": abs(total - float(ae[k].mean()))}
    return out


# ---------------------------------------------------------------------------
# pre-registered Q1 conditions
# ---------------------------------------------------------------------------


def pre_registered_verdict(runs: dict) -> dict:
    t, n = runs.get("T_s1"), runs.get("N_s1")
    if not t or not n:
        return {"available": False}
    cond = {
        "best_valid_T_lt_N": {"T": t["best_valid"], "N": n["best_valid"],
                              "pass": bool(t["best_valid"] < n["best_valid"])},
        "best20_median_T_lt_N": {"T": t["best20_window_median"], "N": n["best20_window_median"],
                                 "pass": bool(t["best20_window_median"]
                                              < n["best20_window_median"])},
    }
    return cond


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="analyse the seed-1 T-vs-N supplement")
    ap.add_argument("--out", default=str(ANALYSIS))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict] = {}
    preds: dict[str, dict] = {}
    curves: dict[str, np.ndarray] = {}
    for arm, seed in ARMS:
        m = A.run_metrics(arm, seed)
        if m is None:
            continue
        key = f"{arm}_s{seed}"
        m["state"] = A.state_diagnostics(arm, seed)
        runs[key] = m
        p = A.load_preds(arm, seed, "best")
        if p is not None:
            preds[key] = p
        curves[key] = A.read_curve(A.RUNS / f"SF_{arm}_s{seed}" / "curve.csv")

    rng = np.random.default_rng(A.BOOT_SEED)
    contrasts = {label: A.contrast(preds.get(a), preds.get(b), rng)
                 for label, a, b in CONTRASTS}

    ref = load_reference()
    cells = cell_table(preds, ref) if preds else {}
    cover = A.type_coverage() if preds else {}
    extremes = A.extreme_diagnostics(preds, ("T_s1", "N_s1")) if {"T_s1", "N_s1"} <= preds.keys() else {}
    extremes_t0 = A.extreme_diagnostics(preds, ("T_s0", "T_s1")) if {"T_s0", "T_s1"} <= preds.keys() else {}

    verdict = pre_registered_verdict(runs)
    cond3 = contrasts.get("T1_minus_N1", {})
    if verdict.get("available") is not False:
        verdict["paired_delta_mae_T_lt_N"] = {
            "delta_mae": cond3.get("delta_mae"),
            "pass": bool(cond3.get("delta_mae") is not None and cond3["delta_mae"] < 0)}
        verdict["n_conditions_passed"] = int(sum(v["pass"] for v in verdict.values()
                                                 if isinstance(v, dict) and "pass" in v))
        verdict["supports_sharing"] = bool(all(v["pass"] for v in verdict.values()
                                               if isinstance(v, dict) and "pass" in v))

    summary = {
        "protocol": "zinc-shared-factor-supplement-v1",
        "kind": "supplement; does not modify the original A_FAIL verdict or its thresholds",
        "runs": runs,
        "contrasts_best_checkpoint": contrasts,
        "cells_radius2_x_ordinary": cells,
        "type_coverage": {k: v for k, v in (cover or {}).items() if not isinstance(v, np.ndarray)},
        "q1_pre_registered": verdict,
        "q2_absolute_T_across_seeds": {
            "T_s0_best": runs.get("T_s0", {}).get("best_valid"),
            "T_s1_best": runs.get("T_s1", {}).get("best_valid"),
            "reference_bar": ABSOLUTE_BAR,
            "T_s0_below_bar": (bool(runs["T_s0"]["best_valid"] <= ABSOLUTE_BAR)
                               if "T_s0" in runs else None),
            "T_s1_below_bar": (bool(runs["T_s1"]["best_valid"] <= ABSOLUTE_BAR)
                               if "T_s1" in runs else None),
        },
        "q3_unseen_ordinary": {
            "cell": cells.get("cells", {}).get("unseen_ordinary"),
            "historical_P3_descriptive_mae": P3_CELL_MAE,
        },
        "extreme_diagnostics_T1_vs_N1": extremes,
        "extreme_diagnostics_T1_vs_T0": extremes_t0,
        "notes": [
            "molecule-level bootstrap intervals describe molecule sampling only",
            "two seeds are NOT a seed-variance estimate",
            "D seed 1 was not run: T1's absolute level cannot be separated from seed difficulty",
            "historical P3 numbers are descriptive references, not same-round paired baselines",
            "no official test data was read in this study",
        ],
    }
    A.write_json(out_dir / "summary.json", summary)

    # ---- full curves ------------------------------------------------------
    if curves:
        with open(out_dir / "valid_curves.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch"] + KEYS)
            n_ep = min(len(c["valid_mae"]) for c in curves.values())
            for i in range(n_ep):
                w.writerow([int(i + 1)] + [f"{float(curves[k]['valid_mae'][i]):.9f}"
                                           for k in KEYS if k in curves])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=130)
            for k in KEYS:
                if k not in curves:
                    continue
                v = curves[k]["valid_mae"]
                ax.plot(np.arange(1, len(v) + 1), v, lw=1.0, label=k)
                ax.axhline(ABSOLUTE_BAR, color="0.6", lw=0.8, ls=":")
            ax.set_xlabel("epoch")
            ax.set_ylabel("valid MAE (all 1000 molecules)")
            ax.set_title("zinc_shared_factor_supplement_v1: T vs N (seed 1)")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "curves.png")
            plt.close(fig)
        except Exception as exc:  # plotting is optional
            print(f"[warn] curve plot skipped: {exc}")

    # ---- per-molecule predictions ----------------------------------------
    if preds:
        base = preds[KEYS[0]]
        pos = align(ref, base["row"])
        cols: dict[str, np.ndarray] = {"row": base["row"], "y_raw": base["y"]}
        for k in KEYS:
            if k in preds:
                cols[f"pred_{k}"] = preds[k]["pred"]
        for name in ("r2_unseen_atom_frac", "signature_unseen", "ordinary"):
            cols[name] = ref[name][pos]
        with open(out_dir / "preds_valid_TN_s1.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(list(cols))
            for i in range(base["row"].size):
                w.writerow([f"{cols[k][i]:.9f}" if np.issubdtype(np.asarray(cols[k]).dtype,
                                                                np.floating) else int(cols[k][i])
                            for k in cols])

    # ---- markdown ---------------------------------------------------------
    lines = ["# zinc_shared_factor_supplement_v1 (T vs N, seed 1)", "",
             "| arm | seed | params | best valid | best ep | train@best | final valid"
             " | best20 median | tail100 median | s/epoch | GPU-h |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, m in runs.items():
        lines.append(
            f"| {m['arm_name']} | {m['seed']} | {m['params']} | {m['best_valid']:.6f} | "
            f"{m['best_epoch']} | {m.get('eval_train_at_best', float('nan')):.6f} | "
            f"{m['final_valid']:.6f} | {m['best20_window_median']:.6f} | "
            f"{m['tail100_median']:.6f} | {m['seconds_per_epoch']:.2f} | "
            f"{(m['gpu_hours'] or 0):.3f} |")

    lines += ["", "## paired per-molecule contrasts (best checkpoint)", "",
              "| pair | ΔMAE | 95% CI (molecules only) | frac improved | median ΔAE |",
              "| --- | --- | --- | --- | --- |"]
    for label, ct in contrasts.items():
        if not ct.get("available"):
            lines.append(f"| {label} | n/a | n/a | n/a | n/a |")
            continue
        lo, hi = ct["ci95"]
        lines.append(f"| {label} | {ct['delta_mae']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | "
                     f"{ct['frac_improved']:.3f} | {ct['median_delta_ae']:+.6f} |")

    keep = [k for k in KEYS if k in preds] or KEYS
    if cells:
        n_total = cells["n_total"]
        lines += ["", "## groups: radius-2 x ordinary (all 1000 molecules)", "",
                  "| cell | n | " + " | ".join(keep) + " | Δ(T1−N1) | Δ(T1−T0) | P3 (desc.) |",
                  "| --- | --- | " + " | ".join("---" for _ in keep)
                  + " | --- | --- | --- |"]
        for name in CELLS:
            c = cells["cells"][name]
            lines.append(f"| {name} | {c['n']} | "
                         + " | ".join(f"{c['mae'].get(k, float('nan')):.4f}" for k in keep)
                         + f" | {c['delta_mae'].get('T1_minus_N1', float('nan')):+.4f}"
                         + f" | {c['delta_mae'].get('T1_minus_T0', float('nan')):+.4f}"
                         + f" | {c['p3_descriptive_mae']:.4f} |")
        lines += ["", "### contribution decomposition (n/1000 x cell ΔMAE)", "",
                  "| cell | n/1000 | Δ(T1−N1) | contributes | Δ(T1−T0) | contributes |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name in CELLS:
            c = cells["cells"][name]
            lines.append(f"| {name} | {c['n'] / n_total:.3f} | "
                         f"{c['delta_mae'].get('T1_minus_N1', float('nan')):+.6f} | "
                         f"{c['delta_contribution'].get('T1_minus_N1', float('nan')):+.6f} | "
                         f"{c['delta_mae'].get('T1_minus_T0', float('nan')):+.6f} | "
                         f"{c['delta_contribution'].get('T1_minus_T0', float('nan')):+.6f} |")
        chk = cells["checks"].get("T_s1") or next(iter(cells["checks"].values()))
        lines.append(f"| **sum** | 1.000 | | "
                     f"{sum(cells['cells'][c]['delta_contribution'].get('T1_minus_N1', 0.0) for c in CELLS):+.6f} |"
                     f" | {sum(cells['cells'][c]['delta_contribution'].get('T1_minus_T0', 0.0) for c in CELLS):+.6f} |")
        lines += ["", f"decomposition check (T_s1): overall {chk['overall_mae']:.6f} vs "
                      f"sum of cells {chk['sum_of_cell_contributions']:.6f} "
                      f"(|diff| {chk['abs_diff']:.2e})"]

    lines += ["", "## pre-registered Q1 (T vs N at seed 1)", "", "```json",
              json.dumps(verdict, indent=1), "```"]
    q2 = summary["q2_absolute_T_across_seeds"]
    fmt = lambda v: "n/a" if v is None else f"{v:.6f}"  # noqa: E731
    lines += ["", "## Q2 (T absolute level across supervised seeds)", "",
              f"- T seed0 best {fmt(q2['T_s0_best'])}, T seed1 best {fmt(q2['T_s1_best'])}, "
              f"reference bar {ABSOLUTE_BAR}: below at s0={q2['T_s0_below_bar']}, "
              f"below at s1={q2['T_s1_below_bar']}"]
    u = summary["q3_unseen_ordinary"]["cell"] or {}
    if u:
        lines += ["", "## Q3 (225 unseen+ordinary molecules)", "",
                  f"- n={u['n']}; MAE: " + ", ".join(
                      f"{k} {u['mae'].get(k, float('nan')):.6f}" for k in keep)
                  + f"; historical P3 {P3_CELL_MAE:.9f} (descriptive)",
                  f"- Δ(T1−N1) {u['delta_mae'].get('T1_minus_N1', float('nan')):+.6f} "
                  f"(contributes {u['delta_contribution'].get('T1_minus_N1', float('nan')):+.6f} "
                  f"of overall MAE), Δ(T1−T0) "
                  f"{u['delta_mae'].get('T1_minus_T0', float('nan')):+.6f} "
                  f"(contributes {u['delta_contribution'].get('T1_minus_T0', float('nan')):+.6f})"]
    lines.append("")
    (out_dir / "main_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[supplement] -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
