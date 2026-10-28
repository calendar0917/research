"""Stage-1 / stage-2 analysis for the second-hop composition pilot.

Reads only the run artefacts (``runs/<run>/train.json`` + ``predictions/*.npz``)
and the frozen LCRC-A predictions.  The primary metric is the official-valid
full raw-y MAE (1,000 rows) at each run's strict-argmin best epoch.

Sign convention: every reported ``X-Y`` delta is ``MAE(X) - MAE(Y)``, so a
negative value means X is better.  The pre-registered purchase thresholds use
the same convention (PROTOCOL.md §4).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sh_common as S

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261106
ROW172 = 172


# ---------------------------------------------------------------------------
# frozen groups / references
# ---------------------------------------------------------------------------

def load_anchor() -> dict:
    with open(S.ANCHOR_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["row_id"]))
    y = np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64)
    unseen = np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows], dtype=np.float64) > 0
    n_rings = np.asarray([int(float(r["n_rings"])) for r in rows], dtype=np.int64)
    worst = S.read_json_or_none(S.WORST10_JSON) or {}
    worst_rows = np.zeros(len(y), dtype=bool)
    for r in worst.get("row_ids", S.GROUPS["fixed_worst10"]):
        worst_rows[int(r)] = True
    idx = np.arange(len(y))
    masks = {
        "all": np.ones(len(y), dtype=bool),
        "ordinary": y > -3.0,
        "tail": y <= -3.0,
        "row172": idx == ROW172,
        "all_minus_row172": idx != ROW172,
        "fixed_worst10": worst_rows,
        "all_minus_fixed_worst10": ~worst_rows,
        "r2_unseen": unseen,
        "r2_unseen_ordinary": unseen & (y > -3.0),
        "seen_ordinary": (~unseen) & (y > -3.0),
        "ring_rich": n_rings >= 4,
        "ring_rich_unseen": (n_rings >= 4) & unseen,
    }
    return {"y": y, "r2_unseen": unseen, "n_rings": n_rings, "masks": masks,
            "worst10_rows": worst_rows, "source": S.track_rel(S.ANCHOR_CSV),
            "worst10_source": S.track_rel(S.WORST10_JSON)}


def variant_masks(anchor: Mapping[str, Any]) -> dict[str, np.ndarray]:
    m = anchor["masks"]
    return {
        "full": m["all"],
        "drop_row172": m["all_minus_row172"],
        "drop_fixed_worst10": m["all_minus_fixed_worst10"],
    }


def load_a_reference(seed: int) -> dict:
    run = S.historical_dir(seed)
    npz = np.load(run / "predictions" / "valid_best.npz")
    row = np.asarray(npz["row"], dtype=np.int64)
    pred = np.asarray(npz["pred"], dtype=np.float64)
    train_json = S.L.read_json(run / "train.json")
    ckpt = torch_load_sha(run / "best.pt")
    return {
        "name": f"A_s{seed}",
        "arm": "A",
        "seed": int(seed),
        "row": row,
        "pred": pred,
        "train_json": train_json,
        "checkpoint_sha256": ckpt,
        "best_valid_mae_epoch_loop": train_json["best"]["valid_mae"],
        "best_valid_mae_replay": train_json["best_valid_mae_eval"],
        "best_epoch": train_json["best"]["epoch"],
        "W20": train_json.get("W20"),
        "T100": train_json.get("T100"),
        "train_mae_eval": train_json.get("best_train_mae_eval"),
        "params": train_json["model_info"]["parameters"],
        "gpu_hours": train_json.get("gpu_hours_estimate"),
        "wall_seconds": train_json.get("wall_seconds"),
        "git_commit": train_json.get("git_commit"),
    }


def torch_load_sha(path: Path) -> str | None:
    return S.L.C.sha256_file(path) if path.exists() else None


def load_run(name: str, anchor: Mapping[str, Any]) -> dict:
    run_dir = S.RUNS / name
    train_json = run_dir / "train.json"
    if not train_json.exists():
        return {}
    payload = S.L.read_json(train_json)
    npz = np.load(run_dir / "predictions" / "valid_best.npz")
    row = np.asarray(npz["row"], dtype=np.int64)
    pred = np.asarray(npz["pred"], dtype=np.float64)
    y = anchor["y"][row]
    mae = float(np.mean(np.abs(pred - y)))
    return {
        "name": name,
        "arm": payload["arm"],
        "seed": int(payload["seed"]),
        "run_dir": S.track_rel(run_dir),
        "row": row,
        "pred": pred,
        "train_json": payload,
        "mae_recomputed": mae,
        "mae_recorded_epoch_loop": payload["best"]["valid_mae"],
        "mae_recorded_replay": payload["best_valid_mae_eval"],
        "delta_recomputed_vs_replay": mae - payload["best_valid_mae_eval"],
        "best_epoch": payload["best"]["epoch"],
        "best_train_mae_eval": payload["best_train_mae_eval"],
        "final_train_mae_eval": payload["final_train_mae_eval"],
        "final_valid_mae": payload["final_valid_mae"],
        "W20": payload["W20"],
        "T100": payload["T100"],
        "params": payload["model_info"]["parameters"],
        "initial_state_sha256": payload["model_info"]["initial_state_sha256"],
        "projections": payload["model_info"].get("projections"),
        "gradient_audit_first_batch": payload["gradient_audit_first_batch"],
        "init_diagnostics": payload.get("init_diagnostics"),
        "step1_weights": payload.get("step1_weights"),
        "best_diagnostics": payload.get("best_diagnostics"),
        "wall_seconds": payload["wall_seconds"],
        "gpu_hours": payload["gpu_hours_estimate"],
        "git_commit": payload["git_commit"],
        "device_name": payload.get("device_name"),
        "environment": payload["environment"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "data": payload["data"],
        "curve": payload["curve"],
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def mae_on(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(pred[mask] - y[mask])))


def model_metrics(model: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict:
    row = model["row"]
    y = anchor["y"][row]
    pred = model["pred"]
    masks = anchor["masks"]
    group_mae = {name: mae_on(pred, y, mask[row]) for name, mask in masks.items() if mask[row].any()}
    variants = {name: mae_on(pred, y, mask[row])
                for name, mask in variant_masks(anchor).items()}
    return {"mae_full": group_mae["all"], "variants": variants, "groups": group_mae,
            "n_rows": int(len(row))}


def contrast(first: Mapping[str, Any], second: Mapping[str, Any],
             anchor: Mapping[str, Any], *, seed: int = BOOTSTRAP_SEED) -> dict:
    """``first - second`` on paired rows: negative => first is better."""
    row = first["row"]
    if not np.array_equal(row, second["row"]):
        raise RuntimeError(f"row mismatch: {first['name']} vs {second['name']}")
    y = anchor["y"][row]
    ae1 = np.abs(first["pred"] - y)
    ae2 = np.abs(second["pred"] - y)
    d = ae1 - ae2                      # per-row delta of absolute error
    masks = anchor["masks"]
    out: dict[str, Any] = {
        "first": first["name"], "second": second["name"],
        "delta_full": float(d.mean()),
        "delta_by_variant": {},
        "delta_by_group": {},
        "win_rate": float(np.mean(d < 0)),
        "sum_abs_delta": float(np.abs(d).sum()),
        "max_single_row_abs_delta": float(np.abs(d).max()),
        "max_row_id": int(np.argmax(np.abs(d))),
        "single_row_share": float(np.abs(d).max() / max(np.abs(d.sum()), 1e-12)),
    }
    for name, mask in variant_masks(anchor).items():
        m = mask[row]
        out["delta_by_variant"][name] = float(d[m].mean())
    for name, mask in masks.items():
        m = mask[row]
        if m.any():
            out["delta_by_group"][name] = float(d[m].mean())
    rng = np.random.default_rng(seed)
    n = len(d)
    boot = np.empty(BOOTSTRAP_N, dtype=np.float64)
    for i in range(BOOTSTRAP_N):
        take = rng.integers(0, n, n)
        boot[i] = d[take].mean()
    out["bootstrap_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    out["bootstrap_n"] = BOOTSTRAP_N
    # leave-one-row-out sensitivity of the full delta
    loo = (d.sum() - d) / (n - 1)
    out["loo_delta_min"] = float(loo.min())
    out["loo_delta_max"] = float(loo.max())
    return out


# ---------------------------------------------------------------------------
# stage gates
# ---------------------------------------------------------------------------

def evaluate_stage1(runs: Mapping[str, Mapping], anchor: Mapping[str, Any]) -> dict:
    a = load_a_reference(0)
    a_metrics = model_metrics(a, anchor)
    real = runs.get("Real_s0", {})
    null = runs.get("Null_s0", {})
    if not real or not null:
        return {"ready": False, "A_s0": a_metrics,
                "reason": "Real_s0 / Null_s0 not both present"}
    real_m = model_metrics(real, anchor)
    null_m = model_metrics(null, anchor)
    c_rn = contrast(real, null, anchor)
    c_ra = contrast(real, a, anchor)
    c_na = contrast(null, a, anchor)

    mae_a = a_metrics["mae_full"]
    c0 = max(real_m["mae_full"], null_m["mae_full"]) <= mae_a + 0.005

    signal_a_hits = []
    for label, m in (("Real", real_m), ("Null", null_m)):
        for v, val in m["variants"].items():
            delta = val - a_metrics["variants"][v]
            if delta <= -0.003 and m["mae_full"] - mae_a <= 0.002:
                signal_a_hits.append({"arm": label, "variant": v, "delta_vs_A": delta,
                                      "delta_full_vs_A": m["mae_full"] - mae_a})
    signal_b_hits = []
    for v in ("full", "drop_row172", "drop_fixed_worst10"):
        delta = real_m["variants"][v] - null_m["variants"][v]
        if delta <= -0.003 and real_m["mae_full"] - null_m["mae_full"] <= 0.002:
            signal_b_hits.append({"variant": v, "delta": delta,
                                  "delta_full": real_m["mae_full"] - null_m["mae_full"]})
    c = anchor["masks"]
    row = real["row"]
    d_r2 = float(np.mean((np.abs(real["pred"] - anchor["y"][row])
                          - np.abs(null["pred"] - anchor["y"][row]))[c["r2_unseen_ordinary"][row]]))
    d_seen = float(np.mean((np.abs(real["pred"] - anchor["y"][row])
                            - np.abs(null["pred"] - anchor["y"][row]))[c["seen_ordinary"][row]]))
    signal_c = bool(d_r2 <= -0.010 and d_seen <= 0.002)

    return {
        "ready": True,
        "A_s0": a_metrics,
        "Real_s0": real_m,
        "Null_s0": null_m,
        "contrasts": {"Real_minus_Null": c_rn, "Real_minus_A": c_ra, "Null_minus_A": c_na},
        "c0_no_break": {"passed": bool(c0), "threshold": mae_a + 0.005,
                        "max_new_full": max(real_m["mae_full"], null_m["mae_full"])},
        "signal_A": {"passed": bool(signal_a_hits), "hits": signal_a_hits,
                     "rule": "variant delta vs A_s0 <= -0.003 with full delta <= +0.002"},
        "signal_B": {"passed": bool(signal_b_hits), "hits": signal_b_hits,
                     "rule": "Real-Null variant delta <= -0.003 with full delta <= +0.002"},
        "signal_C": {"passed": signal_c,
                     "r2_unseen_ordinary_delta": d_r2,
                     "seen_ordinary_delta": d_seen,
                     "rule": "r2_unseen_ordinary <= -0.010 and seen_ordinary <= +0.002"},
        "buy_stage2": bool(c0 and (signal_a_hits or signal_b_hits or signal_c)),
    }


def evaluate_stage2(runs: Mapping[str, Mapping], stage1: Mapping[str, Any],
                    anchor: Mapping[str, Any]) -> dict:
    have = [k for k in ("Real_s0", "Null_s0", "Real_s1", "Null_s1") if k in runs]
    if len(have) < 4:
        return {"ready": False, "available": have}
    a0, a1 = load_a_reference(0), load_a_reference(1)
    per_seed = {}
    for seed, (rname, nname, aref) in enumerate((("Real_s0", "Null_s0", a0),
                                                 ("Real_s1", "Null_s1", a1))):
        real, null = runs[rname], runs[nname]
        rm, nm = model_metrics(real, anchor), model_metrics(null, anchor)
        am = model_metrics(aref, anchor)
        per_seed[str(seed)] = {
            "A": am, "Real": rm, "Null": nm,
            "Real_minus_Null": contrast(real, null, anchor),
            "Real_minus_A": contrast(real, aref, anchor),
            "Null_minus_A": contrast(null, aref, anchor),
        }
    rn = [per_seed[s]["Real_minus_Null"]["delta_full"] for s in ("0", "1")]
    ra = [per_seed[s]["Real_minus_A"]["delta_full"] for s in ("0", "1")]
    na = [per_seed[s]["Null_minus_A"]["delta_full"] for s in ("0", "1")]
    same_sign_rn = (rn[0] < 0) == (rn[1] < 0)
    shares = [per_seed[s]["Real_minus_Null"]["single_row_share"] for s in ("0", "1")]
    me_rn, me_ra, me_na = float(np.mean(rn)), float(np.mean(ra)), float(np.mean(na))
    if not same_sign_rn or max(shares) >= 0.5:
        label = "INSUFFICIENT_EVIDENCE"
    elif me_rn <= -0.001 and me_ra <= -0.001:
        label = "REAL_PAIRING_SUPPORTED_PILOT"
    elif me_ra <= -0.001 and me_na <= -0.001:
        label = "SECOND_HOP_CONTENT_ONLY"
    elif me_ra > -0.001 and me_na > -0.001:
        label = "NO_PREDICTIVE_GAIN_THIS_CONFIG"
    else:
        label = "INSUFFICIENT_EVIDENCE"
    return {
        "ready": True,
        "per_seed": per_seed,
        "mean_Real_minus_Null": me_rn,
        "mean_Real_minus_A": me_ra,
        "mean_Null_minus_A": me_na,
        "Real_minus_Null_same_sign": bool(same_sign_rn),
        "single_row_shares": shares,
        "label": label,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt(x, nd=6):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:+.{nd}f}" if abs(x) < 10 else f"{x:.{nd}f}"
    return str(x)


def write_tables(payload: Mapping[str, Any], anchor: Mapping[str, Any]) -> None:
    lines = ["# Second-hop composition pilot — valid tables", ""]
    stage1 = payload.get("stage1", {})
    if stage1.get("ready"):
        lines += ["## Stage 1 (seed 0)", "",
                  "| model | full | drop row172 | drop worst10 | ordinary | r2-unseen ord | seen ord | tail | best ep | W20 | T100 | train MAE |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key in ("A_s0", "Real_s0", "Null_s0"):
            m = stage1[key]
            if key in payload["runs"]:
                tj = payload["runs"][key]["train_json"]
                best_epoch, w20, t100 = tj["best"]["epoch"], tj["W20"], tj["T100"]
                train_mae = tj.get("best_train_mae_eval", float("nan"))
            else:
                tj = payload["A_refs"]["0"]
                best_epoch, w20, t100 = tj["best_epoch"], tj["W20"], tj["T100"]
                train_mae = tj.get("train_mae_eval", float("nan"))
            g = m["groups"]
            lines.append(
                f"| {key} | {m['mae_full']:.6f} | {m['variants']['drop_row172']:.6f} | "
                f"{m['variants']['drop_fixed_worst10']:.6f} | {g['ordinary']:.6f} | "
                f"{g['r2_unseen_ordinary']:.6f} | {g['seen_ordinary']:.6f} | {g['tail']:.6f} | "
                f"{best_epoch} | {w20:.6f} | {t100:.6f} | {train_mae:.6f} |")
        lines += ["", "## Stage-1 paired contrasts (negative = first is better)", "",
                  "| contrast | full | drop row172 | drop worst10 | CI95 | win rate | max row share | r2-unseen ord | seen ord |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, c in stage1["contrasts"].items():
            v = c["delta_by_variant"]
            g = c["delta_by_group"]
            lines.append(
                f"| {name} | {v['full']:+.6f} | {v['drop_row172']:+.6f} | "
                f"{v['drop_fixed_worst10']:+.6f} | [{c['bootstrap_ci95'][0]:+.6f}, "
                f"{c['bootstrap_ci95'][1]:+.6f}] | {c['win_rate']:.3f} | "
                f"{c['single_row_share']:.3f} | {g.get('r2_unseen_ordinary', float('nan')):+.6f} | "
                f"{g.get('seen_ordinary', float('nan')):+.6f} |")
        lines += ["", "## Stage-1 purchase gate", "",
                  f"* c0 no-break: {stage1['c0_no_break']['passed']} "
                  f"(max new full {stage1['c0_no_break']['max_new_full']:.6f} vs limit "
                  f"{stage1['c0_no_break']['threshold']:.6f})",
                  f"* signal A (vs A): {stage1['signal_A']['passed']} {stage1['signal_A']['hits']}",
                  f"* signal B (Real vs Null): {stage1['signal_B']['passed']} {stage1['signal_B']['hits']}",
                  f"* signal C (r2-unseen ordinary): {stage1['signal_C']['passed']} "
                  f"(r2-unseen ord {stage1['signal_C']['r2_unseen_ordinary_delta']:+.6f}, "
                  f"seen ord {stage1['signal_C']['seen_ordinary_delta']:+.6f})",
                  f"* **buy stage 2: {stage1['buy_stage2']}**"]
    stage2 = payload.get("stage2", {})
    if stage2.get("ready"):
        lines += ["", "## Stage 2 (both seeds)", "",
                  "| seed | Real-Null | Real-A | Null-A | Real-Null CI95 | label |",
                  "|---|---|---|---|---|---|"]
        for seed in ("0", "1"):
            d = stage2["per_seed"][seed]
            lines.append(
                f"| {seed} | {d['Real_minus_Null']['delta_full']:+.6f} | "
                f"{d['Real_minus_A']['delta_full']:+.6f} | "
                f"{d['Null_minus_A']['delta_full']:+.6f} | "
                f"[{d['Real_minus_Null']['bootstrap_ci95'][0]:+.6f}, "
                f"{d['Real_minus_Null']['bootstrap_ci95'][1]:+.6f}] | |")
        lines += ["", f"mean Real-Null {stage2['mean_Real_minus_Null']:+.6f}, "
                      f"mean Real-A {stage2['mean_Real_minus_A']:+.6f}, "
                      f"mean Null-A {stage2['mean_Null_minus_A']:+.6f}, "
                      f"same sign Real-Null {stage2['Real_minus_Null_same_sign']}, "
                      f"single-row shares {stage2['single_row_shares']}",
                  "", f"**label: {stage2['label']}**"]
    lines += ["", "## Mechanism (frozen best checkpoints)", "",
              "| run | W_new/W_old | g_new/g_old @step1 | C-switch mean|Δpred| | C-switch ΔMAE | h PR |",
              "|---|---|---|---|---|---|"]
    for name, run in payload["runs"].items():
        if not run.get("best_diagnostics") or "first_layer" not in run.get(
                "gradient_audit_first_batch", {}):
            continue
        wd = run["best_diagnostics"]["weights"]
        gd = run["gradient_audit_first_batch"]["first_layer"]
        sw = run["best_diagnostics"]["c_switch"]
        hp = run["best_diagnostics"]["h"]["h_participation_ratio"]
        lines.append(f"| {name} | {wd['W_new_over_old']:.4f} | {gd['grad_new_over_old']:.4f} | "
                     f"{sw['mean_abs_pred_change']:.6f} | {sw['delta_mae']:+.6f} | {hp:.2f} |")
    lines += ["", "Rows are the official valid split (1,000 molecules). "
                  "`A_s0`/`A_s1` are the frozen LCRC-A predictions; the paired comparison uses "
                  "the stored per-molecule predictions, not re-training."]
    (S.ANALYSIS / "tables.md").write_text("\n".join(lines) + "\n")


def per_molecule_csv(runs: Mapping[str, Mapping], anchor: Mapping[str, Any]) -> None:
    rows = np.arange(len(anchor["y"]), dtype=np.int64)
    name = "per_molecule_valid.csv"
    fields = ["row_id", "y"] + [f"pred_{k}" for k in runs] + [f"ae_{k}" for k in runs]
    index_maps = {k: {int(r): j for j, r in enumerate(run["row"])} for k, run in runs.items()}
    out_rows = []
    for i in rows:
        entry = {"row_id": int(i), "y": float(anchor["y"][i])}
        for k, run in runs.items():
            pred = run["pred"][index_maps[k][int(i)]]
            entry[f"pred_{k}"] = float(pred)
            entry[f"ae_{k}"] = float(abs(pred - anchor["y"][i]))
        out_rows.append(entry)
    with open(S.ANALYSIS / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="second-hop pilot analysis")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--runs", default=None,
                        help="comma separated run dir names (default: shReal_s0,shNull_s0[,shReal_s1,shNull_s1])")
    args = parser.parse_args(argv)
    S.L.ensure_dir(S.ANALYSIS)

    anchor = load_anchor()
    if args.runs:
        names = [n.strip() for n in args.runs.split(",") if n.strip()]
    else:
        names = ["shReal_s0", "shNull_s0"]
        if args.stage >= 2:
            names += ["shReal_s1", "shNull_s1"]
    keyed = {}
    for name in names:
        run = load_run(name, anchor)
        if run is None:
            continue
        keyed[f"{run['arm']}_s{run['seed']}"] = run
    a_refs = {"0": load_a_reference(0), "1": load_a_reference(1)}
    for seed, ref in a_refs.items():
        ref["row"] = ref["row"]
        ref["pred"] = ref["pred"]

    # sanity: recomputed MAE of each run matches its recorded replay value
    sanity = {k: {"delta_recomputed_vs_replay": v["delta_recomputed_vs_replay"],
                  "mae_recomputed": v["mae_recomputed"],
                  "mae_recorded": v["mae_recorded_replay"]} for k, v in keyed.items()}
    if any(abs(v["delta_recomputed_vs_replay"]) > 1e-6 for v in sanity.values()):
        raise RuntimeError(f"prediction file does not reproduce the recorded MAE: {sanity}")

    stage1 = evaluate_stage1(keyed, anchor)
    stage2 = evaluate_stage2(keyed, stage1, anchor) if args.stage >= 2 else {"ready": False}
    payload = {
        "round": S.ROUND_ID,
        "created": S.L.now(),
        "git_commit": S.L.C.git_commit(),
        "anchor": {"source": anchor["source"], "worst10_source": anchor["worst10_source"],
                   "n_valid": int(len(anchor["y"])),
                   "group_sizes": {k: int(v.sum()) for k, v in anchor["masks"].items()}},
        "A_refs": {s: {k: v for k, v in ref.items() if k not in ("row", "pred")}
                   for s, ref in a_refs.items()},
        "runs": keyed,
        "run_sanity": sanity,
        "stage1": stage1,
        "stage2": stage2,
        "sign_convention": "all X-Y deltas are MAE(X)-MAE(Y); negative = X better",
        "official_test_loaded": False,
    }
    S.L.write_json(S.ANALYSIS / "summary.json", payload)
    write_tables(payload, anchor)
    per_molecule_csv(keyed, anchor)
    if stage1.get("ready"):
        S.L.write_json(S.ANALYSIS / "stage1_gate.json", stage1)
    if stage2.get("ready"):
        S.L.write_json(S.ANALYSIS / "stage2_summary.json", stage2)
    S.L.log(f"analysis written: stage1 ready={stage1.get('ready')} "
            f"buy_stage2={stage1.get('buy_stage2')} stage2={stage2.get('label')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
