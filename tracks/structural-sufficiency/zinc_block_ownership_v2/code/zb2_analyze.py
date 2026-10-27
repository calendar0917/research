"""Analysis and pre-registered decision for ``zinc_block_ownership_v2``.

Conventions (identical to v1 so the numbers are comparable):

* ``delta = AE(other) - AE(ref)`` per molecule, so **positive means the
  reference arm is better**;
* ``d_best = other.best_valid - ref.best_valid``;
* for the ownership contrast the reference is **R** and the other arm is **S**,
  hence ``d_RS > 0`` means true multi-bond-block ownership is better.

Phases:

* B  -- R (v1, frozen) vs S (v2 run), seed 0.  Decides whether a Phase C
  confirmation purchase is justified.
* C  -- R/N/S at seed 1, only if Phase B proceeded.  Reports the two-seed
  mechanism picture and the execution-variance context.

Official test is never read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zb2_common as Z  # noqa: E402

BC = Z.BC
V1RUNS = Z.V1ROUND / "runs"


def _find_run(ctx: dict, arm: str, seed: int) -> dict | None:
    for base in (Z.RUNS, V1RUNS):
        d = base / f"{arm}_s{seed}"
        if (d / "train.json").exists():
            return Z.load_run(d, ctx)
    return None


def _stratified_did(delta: np.ndarray, treat: np.ndarray, ctrl: np.ndarray,
                    n_boot: int = Z.BOOT) -> dict:
    """Difference-in-differences vs the input-equivalence control set.

    ``n_multi <= 1`` molecules have identical R and S block content, so any
    R-vs-S difference there is a pure model/training-path difference, not a
    content difference.  Subtracting it is the honest "ownership-only" estimate.
    Secondary and non-gating: the control set is small (111 rows) and the 95% CI
    is reported, never used as a threshold.
    """
    d_t, d_c = delta[treat], delta[ctrl]
    point = float(d_t.mean() - d_c.mean())
    rng = np.random.default_rng(Z.BOOT_SEED + 7)
    it = rng.integers(0, len(d_t), size=(int(n_boot), len(d_t)))
    ic = rng.integers(0, len(d_c), size=(int(n_boot), len(d_c)))
    boot = d_t[it].mean(axis=1) - d_c[ic].mean(axis=1)
    return {
        "delta_treat_n_multi_ge_2": float(d_t.mean()),
        "delta_control_n_multi_le_1": float(d_c.mean()),
        "did": point,
        "did_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n_treat": int(len(d_t)), "n_control": int(len(d_c)),
        "note": ("secondary, non-gating; the control set has identical R/S block "
                 "content so it absorbs global model differences"),
    }


def compare(ref: dict, other: dict, ctx: dict) -> dict:
    """Paired comparison; positive delta => the reference arm is better."""
    y = ctx["y"]
    n = len(y)
    ae_r, ae_o = ref["ae"]["best"], other["ae"]["best"]
    delta = ae_o - ae_r
    groups = {k: Z.group_stat(delta, m, ae_r, ae_o)
              for k, m in ctx["groups"].items()}

    row172 = np.zeros(n, dtype=bool)
    row172[172] = True
    fixed = np.asarray(ctx["fixed_mask"], dtype=bool)
    absd = np.abs(delta)
    order_abs = np.argsort(-absd, kind="stable")
    ae_order = np.argsort(-ae_r, kind="stable")

    def drop(idx) -> np.ndarray:
        m = np.ones(n, dtype=bool)
        m[np.asarray(list(idx), dtype=np.int64)] = False
        return m

    variants = {}
    for name, mask in {
        "full": np.ones(n, dtype=bool),
        "drop_row172": ~row172,
        "drop_fixed_worst10": ~fixed,
        "drop_row172_and_fixed_worst10": ~(fixed | row172),
        "drop_top1_abs_delta": drop(order_abs[:1]),
        "drop_top5_abs_delta": drop(order_abs[:5]),
        "drop_top10_abs_delta": drop(order_abs[:10]),
        "drop_top10_ae_ref": drop(ae_order[:10]),
    }.items():
        d = delta[mask]
        variants[name] = {"n": int(mask.sum()), "delta_mae": float(d.mean()),
                          "ci95": Z.bootstrap_ci(d)}

    sums = float(delta.sum())
    loo = (sums - delta) / (n - 1)
    total = float(delta.mean())
    single_row = variants["drop_top1_abs_delta"]["delta_mae"]
    # "not single-row driven" per PLAN.md §2.1
    single_row_ok = {
        "top1_contribution_lt_50pct": bool(total != 0 and abs(delta[order_abs[0]]) / abs(sums) < 0.5),
        "top1_contribution_fraction": (float(abs(delta[order_abs[0]]) / abs(sums))
                                       if sums != 0 else None),
        "min_delta_after_removals_ge_half": bool(min(
            variants["drop_row172"]["delta_mae"],
            variants["drop_fixed_worst10"]["delta_mae"],
            variants["drop_top1_abs_delta"]["delta_mae"]) >= 0.5 * total),
        "values_after_removals": {
            "drop_row172": variants["drop_row172"]["delta_mae"],
            "drop_fixed_worst10": variants["drop_fixed_worst10"]["delta_mae"],
            "drop_top1_abs_delta": variants["drop_top1_abs_delta"]["delta_mae"]},
    }
    single_row_ok["not_single_row_driven"] = bool(
        single_row_ok["top1_contribution_lt_50pct"]
        and single_row_ok["min_delta_after_removals_ge_half"])

    return {
        "ref_arm": ref["arm"], "ref_seed": ref["seed"],
        "other_arm": other["arm"], "other_seed": other["seed"],
        "n": n,
        "ref_best_valid": ref["best_valid"], "other_best_valid": other["best_valid"],
        "d_best": float(other["best_valid"] - ref["best_valid"]),
        "d_W20": float(other["W20"] - ref["W20"]),
        "d_T100": float(other["T100"] - ref["T100"]),
        "d_final": float(other["final_valid"] - ref["final_valid"]),
        "delta_mean": total,
        "delta_ci95": Z.bootstrap_ci(delta),
        "win_rate": float((delta > 0).mean()),
        "groups": groups,
        "drop_variants": variants,
        "single_row_check": single_row_ok,
        "leave_one_out": {
            "min": float(loo.min()), "argmin": int(loo.argmin()),
            "max": float(loo.max()), "argmax": int(loo.argmax())},
        "n_rows_delta_gt_0.01": int((delta > 0.01).sum()),
        "n_rows_delta_lt_-0.01": int((delta < -0.01).sum()),
        "did_vs_n_multi_control": _stratified_did(
            delta, ctx["n_multi"] >= 2, ctx["n_multi"] <= 1),
        "train_at_best": {"ref": ref["train_at_best"], "other": other["train_at_best"]},
        "mechanism": {"ref": ref["mechanism"], "other": other["mechanism"]},
        "mechanism_active": {"ref": ref["mechanism_active"],
                             "other": other["mechanism_active"]},
    }


def phase_b_decision(cmp_rs: dict, cfg: dict) -> dict:
    th = cfg["phase_b_decision"]["proceed_thresholds"]
    checks = {
        "d_best_ge_0.003": cmp_rs["d_best"] >= th["d_RS_min"],
        "d_W20_ge_-0.002": cmp_rs["d_W20"] >= th["d_RS_W20_min"],
        "d_T100_ge_-0.002": cmp_rs["d_T100"] >= th["d_RS_T100_min"],
        "not_single_row_driven": cmp_rs["single_row_check"]["not_single_row_driven"],
        "mechanism_active_S": bool(cmp_rs["mechanism_active"]["other"]),
    }
    proceed = all(checks.values())
    if proceed:
        label = "MULTI_BLOCK_OWNERSHIP_SUPPORTED_SEED0"
    elif not checks["mechanism_active_S"]:
        label = "MECHANISM_INACTIVE_S"
    elif not checks["d_best_ge_0.003"]:
        label = "SINGLETON_CONTENT_EXPLAINS_THE_GAIN"
    else:
        label = "SINGLE_ROW_DRIVEN"
    return {"checks": checks, "proceed_to_phase_c": bool(proceed),
            "phase_b_label": label,
            "wording_if_stop": ("multi-block ownership did not obtain evidence "
                                "sufficient to buy a confirmation experiment"),
            "note": ("this is NOT 'multi-block ownership is proven useless'; both "
                     "v1's R-N and v2's R-S are single-seed magnitudes")}


def render(cmp: dict, ctx: dict, tag: str) -> list[str]:
    lines = [f"### {tag}", "",
             "| quantity | value |", "|---|---:|",
             f"| ref / other | {cmp['ref_arm']}_s{cmp['ref_seed']} / {cmp['other_arm']}_s{cmp['other_seed']} |",
             f"| best-valid ref / other | {cmp['ref_best_valid']:.6f} / {cmp['other_best_valid']:.6f} |",
             f"| **d_best** | **{cmp['d_best']:+.6f}** |",
             f"| d_W20 / d_T100 / d_final | {cmp['d_W20']:+.6f} / {cmp['d_T100']:+.6f} / {cmp['d_final']:+.6f} |",
             f"| paired ΔMAE (mean) | {cmp['delta_mean']:+.6f} [{cmp['delta_ci95'][0]:+.6f}, {cmp['delta_ci95'][1]:+.6f}] |",
             f"| win rate | {cmp['win_rate']:.3f} |",
             f"| train@best ref / other | {cmp['train_at_best']['ref']:.6f} / {cmp['train_at_best']['other']:.6f} |", ""]
    keys = ["ordinary_y_gt_-3", "tail_y_le_-3", "support_all_supported",
            "support_low_support", "support_any_unseen", "rings_ge_4",
            "n_multi_1", "n_multi_2", "n_multi_3plus", "fixed_worst10",
            "block_sig_combo_novel", "block_sig_combo_seen"]
    lines += ["| group | n | ΔMAE | 95% CI |", "|---|---:|---:|---|"]
    for k in keys:
        g = cmp["groups"].get(k)
        if g:
            lines.append(f"| {k} | {g['n']} | {g['delta_mean']:+.6f} | "
                         f"[{g['delta_ci95'][0]:+.6f}, {g['delta_ci95'][1]:+.6f}] |")
    lines += ["", "| drop variant | n | ΔMAE | 95% CI |", "|---|---:|---:|---|"]
    for k, v in cmp["drop_variants"].items():
        lines.append(f"| {k} | {v['n']} | {v['delta_mae']:+.6f} | "
                     f"[{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] |")
    did = cmp["did_vs_n_multi_control"]
    lines += ["", f"secondary (non-gating) DiD vs the n_multi<=1 control: "
                  f"Δ(n_multi>=2) {did['delta_treat_n_multi_ge_2']:+.6f} − "
                  f"Δ(n_multi<=1) {did['delta_control_n_multi_le_1']:+.6f} = "
                  f"**{did['did']:+.6f}** [{did['did_ci95'][0]:+.6f}, "
                  f"{did['did_ci95'][1]:+.6f}], n={did['n_treat']}/{did['n_control']}"]
    sr = cmp["single_row_check"]
    lines += ["", f"single-row check: top-1 share {sr['top1_contribution_fraction']:.3f}, "
                  f"not_single_row_driven={sr['not_single_row_driven']}, "
                  f"LOO [{cmp['leave_one_out']['min']:+.6f}, "
                  f"{cmp['leave_one_out']['max']:+.6f}]", ""]
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Z.ANALYSIS / "summary.json"))
    p.add_argument("--tables", default=str(Z.ANALYSIS / "tables.md"))
    p.add_argument("--utc", default=None)
    args = p.parse_args(argv)

    cfg = Z.read_json(Z.CONFIG)
    ctx = Z.valid_context()
    runs = {(arm, seed): _find_run(ctx, arm, seed)
            for arm in ("R", "N", "S") for seed in (0, 1)}
    runs = {k: v for k, v in runs.items() if v is not None}
    Z.log(f"[zb2 analyze] runs found: {sorted(runs)}")

    payload = {
        "round": Z.ROUND_ID, "written": args.utc or Z.now(),
        "official_test_loaded": False,
        "config_sha256": BC.sha256_file(Z.CONFIG),
        "runs": {f"{a}_s{s}": {"dir": r["dir"], "best_valid": r["best_valid"],
                               "W20": r["W20"], "T100": r["T100"],
                               "final_valid": r["final_valid"],
                               "gpu_hours": r["gpu_hours"],
                               "mechanism_active": r["mechanism_active"],
                               "initial_state_sha256": r["initial_state_sha256"],
                               "git_head": r["git_head"],
                               "official_test_loaded": r["official_test_loaded"],
                               "pred_cpu_recompute_max_abs_diff":
                                   r["pred_cpu_recompute_max_abs_diff"]}
                 for (a, s), r in runs.items()},
    }
    lines = ["# zbo-v2 — tables", "",
             f"generated {payload['written']} · official test never loaded", ""]
    lines += ["## Run-level metrics", "",
              "| run | best@ep | W20 | T100 | final | GPU·h | mech active |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for (a, s), r in sorted(runs.items()):
        lines.append(f"| {a}_s{s} | {r['best_valid']:.6f}@{r['best_epoch']} | "
                     f"{r['W20']:.6f} | {r['T100']:.6f} | {r['final_valid']:.6f} | "
                     f"{r['gpu_hours']:.3f} | {r['mechanism_active']} |")
    lines.append("")

    # ---- Phase B: R vs S at seed 0 -------------------------------------
    if ("R", 0) in runs and ("S", 0) in runs:
        cmp_rs = compare(runs[("R", 0)], runs[("S", 0)], ctx)
        payload["phase_b"] = {"R_vs_S_seed0": cmp_rs}
        payload["phase_b_mechanism"] = {
            "S_model_content_probe": {
                src: Z.probe_contents(runs[("S", 0)], ctx, "S", dst)
                for src, dst in (("S", "R"), ("S", "N"))},
            "R_model_content_probe_R_to_S": payload_marker(ctx, runs),
        }
        dec = phase_b_decision(cmp_rs, cfg)
        payload["phase_b"].update(dec)
        payload["phase_b_label"] = dec["phase_b_label"]
        payload["proceed_to_phase_c"] = dec["proceed_to_phase_c"]
        lines += render(cmp_rs, ctx, "Phase B — R vs S, seed 0 (positive = R better)")
        lines += ["**decision**", "",
                  f"* checks: {dec['checks']}",
                  f"* label: `{dec['phase_b_label']}`",
                  f"* proceed_to_phase_c: **{dec['proceed_to_phase_c']}**", ""]

    # ---- reference: R vs N (v1, unchanged) ------------------------------
    if ("R", 0) in runs and ("N", 0) in runs:
        cmp_rn = compare(runs[("R", 0)], runs[("N", 0)], ctx)
        payload["reference_R_vs_N_seed0"] = cmp_rn
        payload["reference_R_vs_N_seed0"]["v1_reported_d_best"] = 0.004884399
        payload["reference_R_vs_N_seed0"]["matches_v1_d_best"] = bool(
            abs(cmp_rn["d_best"] - 0.004884399) < 5e-7)
        lines += render(cmp_rn, ctx, "Reference — R vs N, seed 0 (v1 numbers, re-derived)")
    payload["_context"] = {"n_groups": len(ctx["groups"]),
                           "groups": {k: int(m.sum()) for k, m in ctx["groups"].items()}}

    # ---- Phase C (only if the runs exist) ------------------------------
    if ("R", 1) in runs and ("N", 1) in runs:
        cmp_rn1 = compare(runs[("R", 1)], runs[("N", 1)], ctx)
        payload["phase_c"] = {"R_vs_N_seed1": cmp_rn1}
        lines += render(cmp_rn1, ctx, "Phase C — R vs N, seed 1")
        rn0 = payload["reference_R_vs_N_seed0"]
        payload["phase_c"]["R_vs_N_two_seed"] = {
            "d_best_seed0": rn0["d_best"], "d_best_seed1": cmp_rn1["d_best"],
            "mean_d_best": float((rn0["d_best"] + cmp_rn1["d_best"]) / 2),
            "same_sign": bool(rn0["d_best"] * cmp_rn1["d_best"] > 0),
            "mean_delta": float((rn0["delta_mean"] + cmp_rn1["delta_mean"]) / 2),
        }
        if ("S", 1) in runs:
            cmp_rs1 = compare(runs[("R", 1)], runs[("S", 1)], ctx)
            payload["phase_c"]["R_vs_S_seed1"] = cmp_rs1
            lines += render(cmp_rs1, ctx, "Phase C — R vs S, seed 1")
            rs0 = payload["phase_b"]["R_vs_S_seed0"]
            two = {
                "d_best_seed0": rs0["d_best"], "d_best_seed1": cmp_rs1["d_best"],
                "mean_d_best": float((rs0["d_best"] + cmp_rs1["d_best"]) / 2),
                "same_sign": bool(rs0["d_best"] * cmp_rs1["d_best"] > 0),
                "mean_delta": float((rs0["delta_mean"] + cmp_rs1["delta_mean"]) / 2),
                "both_not_single_row": bool(rs0["single_row_check"]["not_single_row_driven"]
                                            and cmp_rs1["single_row_check"]["not_single_row_driven"]),
                "ordinary_mean": float((rs0["groups"]["ordinary_y_gt_-3"]["delta_mean"]
                                        + cmp_rs1["groups"]["ordinary_y_gt_-3"]["delta_mean"]) / 2),
                "any_unseen_mean": float((rs0["groups"]["support_any_unseen"]["delta_mean"]
                                          + cmp_rs1["groups"]["support_any_unseen"]["delta_mean"]) / 2),
                "R_seed_mean_valid": float((runs[("R", 0)]["best_valid"]
                                            + runs[("R", 1)]["best_valid"]) / 2),
            }
            payload["phase_c"]["R_vs_S_two_seed"] = two
            if (two["mean_d_best"] >= 0.003 and two["same_sign"]
                    and two["both_not_single_row"] and two["ordinary_mean"] >= 0):
                label = "MULTI_BLOCK_OWNERSHIP_CONFIRMED_TWO_SEEDS"
            elif two["same_sign"] and two["mean_d_best"] > 0:
                label = "MULTI_BLOCK_OWNERSHIP_PARTIAL"
            else:
                label = "SINGLETON_CONTENT_ONLY_CONFIRMED"
            payload["phase_c"]["label"] = label
            lines += [f"**two-seed R vs S**: mean d_best {two['mean_d_best']:+.6f}, "
                      f"same sign {two['same_sign']}, ordinary {two['ordinary_mean']:+.6f}, "
                      f"any_unseen {two['any_unseen_mean']:+.6f} → `{label}`", ""]

    Z.write_json(args.out, Z.jsonable(payload))
    Path(args.tables).write_text("\n".join(lines) + "\n")
    Z.log(f"[zb2 analyze] -> {BC.rel_or_abs(args.out)} and {BC.rel_or_abs(args.tables)}")
    if "phase_b_label" in payload:
        Z.log(f"[zb2 analyze] phase B label {payload['phase_b_label']} "
              f"proceed={payload['proceed_to_phase_c']}")
    return 0


def payload_marker(ctx: dict, runs: dict) -> dict:
    """R-model R->S probe, recomputed here so the summary is self-contained."""
    return Z.probe_contents(runs[("R", 0)], ctx, "R", "S")


if __name__ == "__main__":
    raise SystemExit(main())
