"""Analysis for the closeout round: results tables, bootstrap, comparisons, budget.

Reads the pulled probe runs and the source audit; writes machine quantities to
``analysis/{results,learning_path,comparisons,budget}.json`` and the curve
figures.  All predictions are de-standardised exactly once here.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import rc_common as R


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _grid(npz_path: Path) -> dict[str, np.ndarray]:
    z = dict(np.load(npz_path, allow_pickle=False))
    return z


def _rmse(pred_raw, y_raw) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred_raw, np.float64) - np.asarray(y_raw, np.float64)) ** 2)))


def load_arm(rule: str) -> dict[str, Any] | None:
    out = R.RUNS / rule
    tp = out / "train.json"
    gp = out / "grid_predictions.npz"
    if not tp.exists() or not gp.exists():
        return None
    payload = R.read_json(tp)
    z = _grid(gp)
    tnorm = payload["target_norm"]
    inv = lambda p: R.denorm(p, tnorm)  # noqa: E731
    gu = np.asarray(z["grid_updates"], np.int64).tolist()
    y = {"train": np.asarray(z["fit_y"], np.float64), "valid": np.asarray(z["valid_y"], np.float64),
         "test": np.asarray(z["test_y"], np.float64)}
    key = {"train": "train_fit", "valid": "valid", "test": "test"}
    pred_raw = {s: inv(np.asarray(z[key[s]], np.float64)) for s in R.SPLITS}  # (n_grid, n_rows)
    return {"rule": rule, "payload": payload, "grid_updates": gu, "tnorm": tnorm,
            "y": y, "pred_raw": pred_raw, "grid_std": {s: np.asarray(z[key[s]], np.float64) for s in R.SPLITS}}


def step0_index(gu: list[int]) -> int:
    return gu.index(0)


def selected_index(gu: list[int], update: int) -> int:
    return gu.index(int(update))


def metrics_at(arm: dict[str, Any], update: int) -> dict[str, Any]:
    i = selected_index(arm["grid_updates"], update)
    out = {"update": int(update)}
    for s in R.SPLITS:
        p = arm["pred_raw"][s][i]
        y = arm["y"][s]
        out[f"{s}_rmse"] = _rmse(p, y)
        out[f"{s}_mae"] = float(np.mean(np.abs(p - y)))
    return out


# ---------------------------------------------------------------------------
# paired row bootstrap (fixed predictions)
# ---------------------------------------------------------------------------


def _boot_rmse_delta(pred_a, pred_b, y, *, replicates=R.BOOTSTRAP_REPLICATES, seed=R.BOOTSTRAP_SEED):
    rng = np.random.default_rng(int(seed))
    n = len(y)
    a = np.asarray(pred_a, np.float64)
    b = np.asarray(pred_b, np.float64)
    y = np.asarray(y, np.float64)
    deltas = np.empty(replicates, np.float64)
    for r in range(replicates):
        idx = rng.integers(0, n, size=n)
        ra = math.sqrt(np.mean((a[idx] - y[idx]) ** 2))
        rb = math.sqrt(np.mean((b[idx] - y[idx]) ** 2))
        deltas[r] = ra - rb
    point = math.sqrt(np.mean((a - y) ** 2)) - math.sqrt(np.mean((b - y) ** 2))
    return {
        "delta_point": float(point),
        "delta_mean": float(np.mean(deltas)),
        "delta_ci_lo": float(np.percentile(deltas, 2.5)),
        "delta_ci_hi": float(np.percentile(deltas, 97.5)),
        "frac_positive": float(np.mean(deltas > 0)),
        "replicates": int(replicates), "seed": int(seed),
        "note": "paired row bootstrap; conditional on these weights and single seed, sample uncertainty only",
    }


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def build_results(source: dict[str, Any]) -> dict[str, Any]:
    arms = {r: load_arm(r) for r in R.PROBE_ARMS}
    if any(a is None for a in arms.values()):
        missing = [r for r, a in arms.items() if a is None]
        raise FileNotFoundError(f"missing probe runs: {missing}")

    warm = arms["P_WARM"]
    i0 = step0_index(warm["grid_updates"])
    step0 = {"update": 0}
    for s in R.SPLITS:
        step0[f"{s}_rmse"] = _rmse(warm["pred_raw"][s][i0], warm["y"][s])
        step0[f"{s}_mae"] = float(np.mean(np.abs(warm["pred_raw"][s][i0] - warm["y"][s])))

    table = []
    for r in R.PROBE_ARMS:
        a = arms[r]
        rules = a["payload"]["selection"]["rules"]
        row = {"probe": r, "new_params": a["payload"]["new_params"],
               "step0": {s: {"rmse": step0[f"{s}_rmse"], "mae": step0[f"{s}_mae"]} for s in R.SPLITS},
               "r_sel": {}, "a_sel": {}, "w_sel": {}}
        for rule_key, short in (("R_SEL", "r_sel"), ("A_SEL", "a_sel"), ("W_SEL", "w_sel")):
            rec = rules[rule_key]
            upd = rec.get("update")
            if upd is None:
                row[short] = {"update": None}
            else:
                m = metrics_at(a, upd)
                row[short] = m
        table.append(row)

    # gains vs WARM R_SEL and vs common step0
    warm_r = next(t for t in table if t["probe"] == "P_WARM")["r_sel"]
    gains = {}
    for t in table:
        r = t["probe"]
        if t["r_sel"].get("update") is None:
            gains[r] = None
            continue
        g = {"R_SEL_update": t["r_sel"]["update"]}
        for s in ("valid", "test"):
            g[f"vs_warm_{s}_delta"] = t["r_sel"][f"{s}_rmse"] - warm_r[f"{s}_rmse"]
            g[f"vs_warm_{s}_rel_gain"] = (warm_r[f"{s}_rmse"] - t["r_sel"][f"{s}_rmse"]) / warm_r[f"{s}_rmse"]
            g[f"vs_step0_{s}_delta"] = t["r_sel"][f"{s}_rmse"] - step0[f"{s}_rmse"]
            g[f"vs_step0_{s}_rel_gain"] = (step0[f"{s}_rmse"] - t["r_sel"][f"{s}_rmse"]) / step0[f"{s}_rmse"]
        gains[r] = g

    # purchase threshold: a NEW probe (POST/PRE) improves Valid AND Test by >=1%
    # vs BOTH P_WARM and common step0, under R_SEL
    purchase = {}
    for r in ("P_POST", "P_PRE"):
        g = gains.get(r)
        if not g:
            purchase[r] = {"passes": False, "reason": "no R_SEL state"}
            continue
        ok = all(g[f"{base}_{s}_rel_gain"] >= R.PURCHASE_RELATIVE_GAIN
                 for base in ("vs_warm", "vs_step0") for s in ("valid", "test"))
        purchase[r] = {"passes": bool(ok), "relative_gain": {
            "vs_warm_valid": g["vs_warm_valid_rel_gain"], "vs_warm_test": g["vs_warm_test_rel_gain"],
            "vs_step0_valid": g["vs_step0_valid_rel_gain"], "vs_step0_test": g["vs_step0_test_rel_gain"]}}

    # bootstrap: each arm's R_SEL predictions vs WARM R_SEL (per split) and vs step0
    boot = {}
    for r in ("P_POST", "P_PRE"):
        a = arms[r]
        upd = a["payload"]["selection"]["rules"]["R_SEL"]["update"]
        if upd is None:
            boot[r] = {"status": "no R_SEL state"}
            continue
        ia = selected_index(a["grid_updates"], upd)
        iw = selected_index(warm["grid_updates"], warm["payload"]["selection"]["rules"]["R_SEL"]["update"])
        rec = {}
        for s in ("valid", "test"):
            rec[f"vs_warm_{s}"] = _boot_rmse_delta(a["pred_raw"][s][ia], warm["pred_raw"][s][iw], warm["y"][s])
            rec[f"vs_step0_{s}"] = _boot_rmse_delta(a["pred_raw"][s][ia], warm["pred_raw"][s][i0], warm["y"][s])
        boot[r] = rec

    # PRE vs POST closeness
    post = arms["P_POST"]
    pre = arms["P_PRE"]
    ip = selected_index(post["grid_updates"], post["payload"]["selection"]["rules"]["R_SEL"]["update"])
    ie = selected_index(pre["grid_updates"], pre["payload"]["selection"]["rules"]["R_SEL"]["update"])
    pre_vs_post = {}
    for s in R.SPLITS:
        d = np.abs(pre["pred_raw"][s][ie] - post["pred_raw"][s][ip])
        pre_vs_post[s] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}
    # max prediction difference over the whole grid, vs WARM at matching updates
    def max_diff_vs_warm(arm):
        out = {}
        gu_a = arm["grid_updates"]
        gu_w = warm["grid_updates"]
        for s in R.SPLITS:
            ds = []
            for u in gu_a:
                if u in gu_w:
                    ds.append(float(np.abs(arm["pred_raw"][s][gu_a.index(u)] - warm["pred_raw"][s][gu_w.index(u)]).max()))
            out[s] = {"max_abs": float(max(ds)) if ds else 0.0}
        return out
    branch_presence = {"P_POST_vs_WARM": max_diff_vs_warm(post),
                       "P_PRE_vs_WARM": max_diff_vs_warm(pre),
                       "POST_vs_PRE_R_SEL": pre_vs_post}

    # interpretation layer
    lp = aggregate_learning_path()
    layer = classify(layer_name=None, lp=lp, purchase=purchase, branch=branch_presence)

    return {
        "schema": "lcrc-readout-closeout-results-v1", "round_id": R.ROUND_ID,
        "status_tag": R.STATUS_TAG,
        "source": {"arm": source["source"]["arm"], "internal_update": source["source"]["internal_update"],
                   "state_hash": source["source"]["state_hash"], "verdict": source.get("verdict")},
        "target_norm": warm["tnorm"],
        "step0_common": step0,
        "table": table,
        "gains": gains,
        "purchase_threshold": {"rule": f">={R.PURCHASE_RELATIVE_GAIN:.0%} Valid AND Test RMSE gain vs P_WARM and common step0 (R_SEL)",
                               "result": purchase, "passed": any(v.get("passes") for v in purchase.values())},
        "bootstrap": boot,
        "branch_presence": branch_presence,
        "interpretation_layer": layer,
        "learning_path_summary": lp.get("summary"),
        "cost": {"gpu_hours_charged": sum((arms[r]["payload"]["cost"]["gpu_hours_charged"] for r in R.PROBE_ARMS)),
                 "wall_seconds": sum(arms[r]["payload"]["cost"]["wall_seconds"] for r in R.PROBE_ARMS)},
    }


# ---------------------------------------------------------------------------
# learning path aggregation
# ---------------------------------------------------------------------------


def aggregate_learning_path() -> dict[str, Any]:
    out = {"probes": {}, "summary": {}}
    for r in R.PROBE_ARMS:
        p = R.RUNS / r / "learning_path.json"
        if p.exists():
            out["probes"][r] = R.read_json(p)
    # summarize branch channel use at 1800 for POST/PRE
    summary = {}
    for r in ("P_POST", "P_PRE"):
        recs = out["probes"].get(r, {}).get("records", [])
        if not recs:
            continue
        last = recs[-1]
        b = last.get("branch", {})
        summary[r] = {
            "update": last["update"],
            "grad_norm_preclip": last["grad_norm_preclip"],
            "relative_param_displacement": last["relative_param_displacement"],
            "channel_response_relative_to_full_delta": b.get("delta_channel_response", {}).get("relative_to_full_delta"),
            "channel_response_raw_rms": b.get("delta_channel_response", {}).get("raw_rms"),
            "mean_psi_vs_psi_mean_relative_to_extra_rms": b.get("mean_psi_vs_psi_mean", {}).get("relative_to_extra_rms"),
            "extra_feature_rms": b.get("extra_feature_rms"),
            "psi_grad_nonzero_by_1800": (last["grad_norm_preclip"].get("psi", 0.0) or 0.0) > 0.0,
            "delta_first_grad_nonzero_by_1800": (last["grad_norm_preclip"].get("delta_first", 0.0) or 0.0) > 0.0,
        }
    out["summary"] = summary
    return out


def classify(*, layer_name, lp: dict[str, Any], purchase: dict[str, Any], branch: dict[str, Any]) -> dict[str, Any]:
    """Assign the pre-registered interpretation layer A/B/C/D."""
    # A: implementation invalid handled upstream (source/cache mismatch => no formal run)
    summary = lp.get("summary", {})
    weak = []
    strong = []
    for r, s in summary.items():
        rel = s.get("channel_response_relative_to_full_delta")
        psi_ok = s.get("psi_grad_nonzero_by_1800")
        if rel is None:
            continue
        # "weak" = channel response is < 5% of the full delta output at 1800
        (weak if (rel < 0.05 or not psi_ok) else strong).append(r)
    passes = any(v.get("passes") for v in purchase.values())
    if passes:
        layer = "PURCHASE_CANDIDATE"
    elif summary and len(weak) == len(summary):
        layer = "EXTRA_CHANNEL_WEAKLY_USED_THIS_CONFIG"
    elif summary:
        layer = "NO_READOUT_GAIN_THIS_CONFIG"
    else:
        layer = "INCONCLUSIVE"
    return {
        "layer": layer,
        "weak_channel_arms": weak, "responsive_channel_arms": strong,
        "note": ("PRE vs POST closeness alone does not prove the branch did not train; "
                 "the per-update gradient/displacement/channel-response measurements decide."),
    }


# ---------------------------------------------------------------------------
# comparisons (upstream same-round numbers, existing predictions only)
# ---------------------------------------------------------------------------


def build_comparisons(results: dict[str, Any]) -> dict[str, Any]:
    up = R.UPSTREAM_TRACK / "analysis/synthesis"
    db = R.read_json(up / "data_budget.json")["cells"]
    ro = R.read_json(up / "readout.json")

    # Lipo E1 R_SEL for the structural arms + DMPNN, and BACE fullN
    def lipo_e1(cell_key, n):
        cells = db.get(cell_key, [])
        for c in cells:
            if int(c.get("N", -1)) == int(n):
                return {"arm_id": c.get("arm_id"), "N": c.get("N"), "U0": c.get("U0"),
                        "E1_R_SEL_test": c.get("E1_R_SEL_test"),
                        "E1_last10_test_median": c.get("E1_last10_test_median"),
                        "E1_R_SEL_valid": c.get("E1_R_SEL_valid")}
        return None

    structural = {
        "C1_R1@Lipophilicity (F100)": lipo_e1("C1_R1@Lipophilicity", 3360),
        "C1_R3@Lipophilicity (F100)": lipo_e1("C1_R3@Lipophilicity", 3360),
        "CHEM8_R1@Lipophilicity (F100)": lipo_e1("CHEM8_R1@Lipophilicity", 3360),
        "K_R1@Lipophilicity (F100)": lipo_e1("K_R1@Lipophilicity", 3360),
        "DMPNN_D2F@Lipophilicity (F100)": lipo_e1("DMPNN_D2F@Lipophilicity", 3360),
    }
    bace = {
        "C1_R1@BACE (F100)": lipo_e1("C1_R1@BACE", 1210),
        "DMPNN_D2F@BACE (F100)": lipo_e1("DMPNN_D2F@BACE", 1210),
    }

    old = {}
    lipo_ro = ro.get("lipo", {})
    for r in R.PROBE_ARMS:
        if r in lipo_ro:
            sel = lipo_ro[r]["selected"]["R_SEL"]
            old[r] = {"source_update": 9720, "R_SEL_update": sel.get("update"),
                      "valid_rmse": sel.get("valid_rmse"), "test_rmse": sel.get("test_rmse"),
                      "new_params": lipo_ro[r]["new_params"]}
    old_oracle = ro.get("probe_oracle", {})
    old_meta = {"branch_displacement_vs_P_WARM_max_abs": old_oracle.get("branch_displacement_vs_P_WARM_max_abs"),
                "pre_vs_post_max_abs": old_oracle.get("pre_vs_post_max_abs")}

    new = {}
    for row in results["table"]:
        r = row["probe"]
        rsel = row["r_sel"]
        new[r] = {"source_update": results["source"]["internal_update"],
                  "R_SEL_update": rsel.get("update"),
                  "valid_rmse": rsel.get("valid_rmse"), "test_rmse": rsel.get("test_rmse"),
                  "new_params": row["new_params"]}

    return {
        "schema": "lcrc-readout-closeout-comparisons-v1",
        "note": "existing upstream predictions only; not recomputed here",
        "structural_fullN_E1_R_SEL": structural,
        "bace_fullN_E1_R_SEL": bace,
        "old_probes_u0_snapshot_9720": {"probes": old, "meta": old_meta},
        "new_probes_update_8165": {"probes": new},
        "caveat": ("New probe gain over R1 and whether it exceeds the cheaper R3 are two different "
                   "questions; a probe result does not re-rank the structural arms."),
    }


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def build_budget() -> dict[str, Any]:
    per_arm = {}
    for r in R.PROBE_ARMS:
        p = R.RUNS / r / "train.json"
        if p.exists():
            pay = R.read_json(p)
            per_arm[r] = {"wall_seconds": pay["cost"]["wall_seconds"],
                          "process_wall_seconds": pay["cost"]["process_wall_seconds"],
                          "gpu_hours_charged": pay["cost"]["gpu_hours_charged"]}
    audit = R.read_json(R.ANALYSIS / "source_audit.json") if (R.ANALYSIS / "source_audit.json").exists() else {}
    ledger = R.read_json(R.ANALYSIS / "launch_ledger.json") if (R.ANALYSIS / "launch_ledger.json").exists() else {}
    total_gpu = sum(v["gpu_hours_charged"] for v in per_arm.values())
    return {
        "schema": "lcrc-readout-closeout-budget-v1", "round_id": R.ROUND_ID,
        "caps": {"gpu_soft_hours": R.GPU_SOFT_CAP_HOURS, "gpu_hard_hours": R.GPU_HARD_CAP_HOURS,
                 "wall_target_hours": R.WALL_TARGET_HOURS, "wall_hard_hours": R.WALL_HARD_HOURS},
        "per_arm": per_arm,
        "gpu_hours_charged_total": float(total_gpu),
        "within_gpu_soft_cap": bool(total_gpu <= R.GPU_SOFT_CAP_HOURS),
        "source_audit": {"wall_seconds": (audit.get("cost") or {}).get("wall_seconds"),
                         "device": (audit.get("cost") or {}).get("device")},
        "launch_ledger": (ledger.get("jobs") if ledger else {}),
        "formal_trajectories": len(per_arm),
        "note": "GPU hours are the per-process card wall reported by each probe run; audit + cache are eval-only.",
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures() -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        R.log(f"figures skipped: {exc}")
        return []
    fig_dir = R.ANALYSIS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    colors = {"P_WARM": "tab:blue", "P_POST": "tab:orange", "P_PRE": "tab:green"}
    # per-arm curves
    curves = {}
    for r in R.PROBE_ARMS:
        cp = R.RUNS / r / "curve.csv"
        if cp.exists():
            with open(cp, newline="") as fh:
                curves[r] = list(csv.DictReader(fh))
    if curves:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, split in zip(axes, ("train_fit", "valid", "test")):
            for r, rows in curves.items():
                xs = [int(x["update"]) for x in rows]
                ys = [float(x[f"{split}_rmse"]) for x in rows]
                ax.plot(xs, ys, label=r, color=colors[r], lw=1.3)
            ax.set_title(f"{split} RMSE (raw)")
            ax.set_xlabel("update")
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle("Readout probes on recovered update-8165 source")
        fig.tight_layout()
        p = fig_dir / "probe_curves.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(R.track_rel(p))
    return made


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="closeout analysis")
    ap.parse_args(argv)
    source = R.read_json(R.ANALYSIS / "source_audit.json")
    results = build_results(source)
    R.write_json(R.ANALYSIS / "results.json", results)
    R.write_json(R.ANALYSIS / "learning_path.json", aggregate_learning_path())
    R.write_json(R.ANALYSIS / "comparisons.json", build_comparisons(results))
    R.write_json(R.ANALYSIS / "budget.json", build_budget())
    figs = make_figures()
    R.log(f"analysis done: purchase_passed={results['purchase_threshold']['passed']} "
          f"layer={results['interpretation_layer']['layer']} figures={figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
