"""Valid-only analysis for LCRC-v1 (no official test access).

Produces:
  analysis/main_table.{json,md}
  analysis/contrasts.{json,md}
  analysis/groups.{json,md}
  analysis/per_molecule_valid.csv
  analysis/mechanism.json
  analysis/summary.json

The DCC-L anchor is historical: the frozen performance-search valid
predictions (B-R0 online, seeds 0..3) and the reported 4-seed valid mean
0.075990.  It is not retrained in this round.  The fixed subgroup masks
(radius-2 unseen, P3B worst-10) come from the earlier generalization audit and
were frozen before this round.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import lc_common as L
import lc_data as D
import lc_models as M
import lc_train as T

ANCHOR_CSV = L.REPO / "tracks/structural-sufficiency/zinc_performance_search_v1/analysis/per_molecule_valid.csv"
WORST10_JSON = L.REPO / "tracks/structural-sufficiency/zinc_generalization_audit_v1/analysis/fixed_worst10.json"
DCCL_4SEED_VALID = 0.075990
DCCL_S0_VALID = 0.078537
ROW172 = 172


def _read_anchor():
    with open(ANCHOR_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["row_id"]))
    y = np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64)
    unseen = np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows], dtype=np.float64) > 0
    n_rings = np.asarray([int(float(r["n_rings"])) for r in rows], dtype=np.int64)
    preds = {}
    for seed in range(4):
        key = f"pred_B-R0-online-s{seed}"
        preds[seed] = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return {"y": y, "unseen": unseen, "n_rings": n_rings, "dcc_pred": preds,
            "row_id": np.asarray([int(r["row_id"]) for r in rows])}


def _fixed_masks(anchor):
    worst = json.load(open(WORST10_JSON))
    worst_rows = np.zeros(len(anchor["y"]), dtype=bool)
    for r in worst["row_ids"]:
        worst_rows[int(r)] = True
    y = anchor["y"]
    masks = {
        "all": np.ones(len(y), dtype=bool),
        "ordinary_y_gt_-3": y > -3.0,
        "tail_y_le_-3": y <= -3.0,
        "row172": np.arange(len(y)) == ROW172,
        "r2_unseen": anchor["unseen"],
        "r2_unseen_ordinary": anchor["unseen"] & (y > -3.0),
        "ring_rich_ge4": anchor["n_rings"] >= 4,
        "ring_rich_ge4_unseen": (anchor["n_rings"] >= 4) & anchor["unseen"],
        "dcc_worst10_fixed": worst_rows,
        "all_minus_row172": np.arange(len(y)) != ROW172,
        "all_minus_worst10": ~worst_rows,
        "tail_minus_row172": (y <= -3.0) & (np.arange(len(y)) != ROW172),
    }
    return masks, worst_rows


def _mae(pred, y, mask=None):
    ae = np.abs(np.asarray(pred, np.float64) - np.asarray(y, np.float64))
    if mask is not None:
        ae = ae[mask]
    return float(ae.mean()) if len(ae) else float("nan")


def _bootstrap_delta(ae_first, ae_second, *, n_boot=10000, seed=20261108):
    d = np.asarray(ae_first, np.float64) - np.asarray(ae_second, np.float64)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "win_rate_first_better": float((d < 0).mean()),
        "n": int(n),
    }


def _contrast(ae_first, ae_second, worst_rows, y):
    out = _bootstrap_delta(ae_first, ae_second)
    d = ae_first - ae_second
    order = np.argsort(-np.abs(d))
    drop = np.ones(len(d), dtype=bool)
    drop[order[:10]] = False
    out["drop_top10_abs_delta"] = float(d[drop].mean())
    keep = np.arange(len(d)) != ROW172
    out["drop_row172"] = float(d[keep].mean())
    keep = ~worst_rows
    out["drop_fixed_dcc_worst10"] = float(d[keep].mean())
    return out


def _load_run(run_dir: Path):
    info = L.read_json(run_dir / "train.json")
    arm = info["arm"]
    seed = int(info["seed"])
    valid = D.load_cache("valid")
    y = np.asarray(valid["labels"], np.float64)
    best_npz = np.load(run_dir / "predictions" / "valid_best.npz")
    pred_best = np.asarray(best_npz["pred"], np.float64)
    rows = np.asarray(best_npz["row"], np.int64)
    if not np.array_equal(rows, np.arange(len(y))):
        raise RuntimeError(f"{run_dir}: valid rows are not 0..n-1")
    recomputed = _mae(pred_best, y)
    if abs(recomputed - float(info["best"]["valid_mae"])) > 1e-6:
        raise RuntimeError(f"{run_dir}: best-valid replay mismatch {recomputed} vs {info['best']}")
    final_npz = np.load(run_dir / "predictions" / "valid_final.npz")
    pred_final = np.asarray(final_npz["pred"], np.float64)
    return {
        "run": run_dir.name,
        "arm": arm,
        "seed": seed,
        "info": info,
        "y": y,
        "pred_best": pred_best,
        "pred_final": pred_final,
        "best_valid": recomputed,
        "ae_best": np.abs(pred_best - y),
        "ae_final": np.abs(pred_final - y),
    }


def _discover_runs(explicit: list[str] | None):
    if explicit:
        return [L.RUNS / r for r in explicit]
    out = []
    for p in sorted(L.RUNS.glob("lcrc*_s*")):
        if (p / "train.json").exists() and (p / "predictions" / "valid_best.npz").exists():
            out.append(p)
    return out


def _load_model(arm, state):
    norm = D.load_norm()
    return M.load_model_from_state(arm, norm, state, "cpu")


def _model_probe(run):
    arm = run["arm"]
    state_path = L.RUNS / run["run"] / "best.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)["state_dict"]
    model = _load_model(arm, state)
    valid = D.load_cache("valid")
    rows = np.arange(len(run["y"]))
    pred, mae, aux = T.evaluate(model, valid, rows, device="cpu", batch_size=256, return_aux=True)
    out: dict[str, Any] = {
        "run": run["run"],
        "arm": arm,
        "seed": run["seed"],
        "best_valid": mae,
        "checkpoint_sha256": L.C.sha256_file(state_path),
        "delta_mean_abs": float(np.mean(np.abs(aux["delta"]))),
        "delta_max_abs": float(np.max(np.abs(aux["delta"]))),
        "contrib_mean_abs": float(np.mean(np.abs(aux["contrib"]))),
        "rho_mean_abs": float(np.mean(np.abs(aux["rho"]))),
        "rho_over_contrib_median": float(np.median(np.abs(aux["rho"]) / np.maximum(np.abs(aux["contrib"]), 1e-8))),
        "local_share_median": float(np.median(np.abs(aux["contrib"]) / np.maximum(np.abs(aux["contrib"]) + np.abs(aux["rho"]), 1e-8))),
    }
    # h effective dimension at the selected checkpoint
    H = np.asarray(aux["mean_h"], np.float64)
    Hc = H - H.mean(0, keepdims=True)
    cov = Hc.T @ Hc / max(len(Hc) - 1, 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0, None)
    out["h_participation_ratio"] = float((ev.sum() ** 2) / max((ev ** 2).sum(), 1e-12))
    out["h_top1_energy_fraction"] = float(ev[-1] / max(ev.sum(), 1e-12))
    if arm in ("R", "N"):
        rel = model.rel
        out["rel_head_norm"] = float(torch.linalg.vector_norm(rel.head.weight).item())
        out["cat_emb_norm"] = float(torch.linalg.vector_norm(rel.cat_emb.weight).item())
        # same weights, forced counterpart mode: isolates the correspondence
        orig = model.arm
        model.arm = "N" if arm == "R" else "R"
        _, _, aux_other = T.evaluate(model, valid, rows, device="cpu", batch_size=256, return_aux=True)
        model.arm = orig
        diff = np.linalg.norm(np.asarray(aux["M"]) - np.asarray(aux_other["M"]), axis=2)
        ex = np.asarray(aux["exists"]) > 0.5
        out["same_weights_other_mode"] = {
            "fraction_molecules_content_diff": float(np.mean([(diff[i][ex[i]] > 1e-6).any() for i in range(len(diff))])),
            "mean_l2_over_nonempty": float(np.mean([diff[i][ex[i]].mean() if ex[i].any() else 0.0 for i in range(len(diff))])),
            "delta_mae_other_mode": float(_mae(aux_other["delta"] + aux["contrib"] + aux["rho"], run["y"])),
            "delta_mae_normal": float(_mae(aux["contrib"] + aux["rho"] + aux["delta"], run["y"])),
        }
        # frozen-weight shutdown of the relation path
        zero = copy.deepcopy(model)
        with torch.no_grad():
            zero.rel.head.weight.zero_()
            zero.rel.head.bias.zero_()
        pred0, mae0 = T.evaluate(zero, valid, rows, device="cpu", batch_size=256)
        out["head_zero"] = {
            "valid_mae": float(mae0),
            "delta_mae": float(mae0 - mae),
            "mean_abs_pred_change": float(np.mean(np.abs(pred0 - pred))),
            "max_abs_pred_change": float(np.max(np.abs(pred0 - pred))),
        }
        # per-category content coverage / RMS
        n = np.asarray(aux["n"])
        out["category_nonempty_fraction"] = ex.mean(0).tolist()
        out["category_mean_count"] = n.mean(0).tolist()
        out["content_rms_per_category"] = np.sqrt((np.asarray(aux["M"]) ** 2).mean(axis=(0, 2))).tolist()
    return out, aux


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    args = parser.parse_args(argv)
    run_dirs = _discover_runs(args.runs)
    if not run_dirs:
        raise SystemExit("no completed LCRC runs found")
    anchor = _read_anchor()
    masks, worst_rows = _fixed_masks(anchor)
    runs = [_load_run(p) for p in run_dirs]
    runs.sort(key=lambda r: (r["arm"], r["seed"]))
    if not np.allclose(anchor["y"], runs[0]["y"], atol=1e-5, rtol=0.0):
        raise RuntimeError("historical valid labels do not match the LCRC valid cache")
    L.ensure_dir(L.ANALYSIS)

    # ---------------- main table ----------------
    main_rows = []
    for r in runs:
        info = r["info"]
        main_rows.append({
            "run": r["run"],
            "arm": r["arm"],
            "seed": r["seed"],
            "params": int(info["model_info"]["parameters"]),
            "epochs": int(info["epochs"]),
            "updates": int(info["total_updates"]),
            "best_valid": float(r["best_valid"]),
            "best_epoch": int(info["best"]["epoch"]),
            "W20": info.get("W20"),
            "T100": info.get("T100"),
            "final_valid": float(info["final_valid_mae"]),
            "train_eval_best": float(info["best_train_mae_eval"]),
            "train_eval_final": float(info["final_train_mae_eval"]),
            "wall_hours": float(info["gpu_hours_estimate"]),
            "git_commit": info.get("git_commit"),
            "init_state_sha256": info["model_info"]["initial_state_sha256"],
            "checkpoint_sha256": info["checkpoint_sha256"]["best"],
            "data_order_seed": info.get("data_order_seed"),
            "first_batch_rows_sha256": L.C.sha256_array(np.asarray(info["first_batch_rows"], np.int64)),
        })
    L.write_json(L.ANALYSIS / "main_table.json", {"rows": main_rows, "anchor": {
        "DCC_L_valid_4seed_mean": DCCL_4SEED_VALID,
        "DCC_L_valid_seed0": DCCL_S0_VALID,
        "source": "historical zinc_performance_search_v1 (not retrained)",
    }})
    lines = ["# LCRC-v1 main table (official valid-1000, raw-y MAE)", "",
             "| run | arm | seed | params | updates | best valid | ep | W20 | T100 | final valid | train@best | GPU-h |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in main_rows:
        lines.append(
            f"| {m['run']} | {m['arm']} | {m['seed']} | {m['params']:,} | {m['updates']:,} | "
            f"{m['best_valid']:.6f} | {m['best_epoch']} | {m['W20']:.6f} | {m['T100']:.6f} | "
            f"{m['final_valid']:.6f} | {m['train_eval_best']:.6f} | {m['wall_hours']:.3f} |"
        )
    lines.append("")
    lines.append(f"Historical DCC-L anchor: valid 4-seed mean {DCCL_4SEED_VALID:.6f}, seed0 {DCCL_S0_VALID:.6f}.")
    (L.ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n")

    # ---------------- per-molecule + contrasts ----------------
    per = {"row": np.arange(len(runs[0]["y"])), "y": runs[0]["y"],
           "r2_unseen_atom_frac": anchor["unseen"].astype(float), "n_rings": anchor["n_rings"],
           "dcc_s0_ae": np.abs(anchor["dcc_pred"][0] - runs[0]["y"])}
    for r in runs:
        per[f"pred_{r['run']}"] = r["pred_best"]
        per[f"ae_{r['run']}"] = r["ae_best"]

    contrast_rows = []
    by_key = {(r["arm"], r["seed"]): r for r in runs}
    for seed in sorted({r["seed"] for r in runs}):
        for a, b in (("R", "A"), ("N", "A"), ("R", "N")):
            if (a, seed) in by_key and (b, seed) in by_key:
                ra, rb = by_key[(a, seed)], by_key[(b, seed)]
                c = _contrast(ra["ae_best"], rb["ae_best"], worst_rows, ra["y"])
                c.update({"first": ra["run"], "second": rb["run"], "seed": seed,
                          "mae_first": ra["best_valid"], "mae_second": rb["best_valid"]})
                contrast_rows.append(c)
    for r in runs:
        c = _contrast(r["ae_best"], per["dcc_s0_ae"], worst_rows, r["y"])
        c.update({"first": r["run"], "second": "DCC-L_s0_historical", "seed": r["seed"],
                  "mae_first": r["best_valid"], "mae_second": DCCL_S0_VALID})
        contrast_rows.append(c)
    L.write_json(L.ANALYSIS / "contrasts.json", {"rows": contrast_rows,
                                                "bootstrap": {"n": 10000, "seed": 20261108,
                                                              "semantics": "molecule resampling only, not training seeds"}})
    clines = ["# LCRC-v1 paired contrasts (valid-1000)", "",
              "Delta = mean AE(first) - mean AE(second); negative = first better. "
              "Bootstrap resamples molecules only.", "",
              "| first | second | seed | MAE first | MAE second | Delta | 95% CI | first better | drop row172 | drop fixed worst10 | drop top10 abs Delta |",
              "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for c in contrast_rows:
        clines.append(
            f"| {c['first']} | {c['second']} | {c['seed']} | {c['mae_first']:.6f} | {c['mae_second']:.6f} | "
            f"{c['mean_delta']:+.6f} | [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | {c['win_rate_first_better']:.3f} | "
            f"{c['drop_row172']:+.6f} | {c['drop_fixed_dcc_worst10']:+.6f} | {c['drop_top10_abs_delta']:+.6f} |"
        )
    (L.ANALYSIS / "contrasts.md").write_text("\n".join(clines) + "\n")

    # ---------------- groups ----------------
    group_rows = {}
    for name, mask in masks.items():
        group_rows[name] = {"n": int(mask.sum()), "mae": {}}
        for r in runs:
            group_rows[name]["mae"][r["run"]] = _mae(r["pred_best"], r["y"], mask)
        group_rows[name]["mae"]["DCC-L_s0_historical"] = _mae(anchor["dcc_pred"][0], r["y"], mask)
    L.write_json(L.ANALYSIS / "groups.json", group_rows)
    glines = ["# LCRC-v1 fixed-group valid MAE", "",
              "| group | n | " + " | ".join(r["run"] for r in runs) + " | DCC-L s0 |",
              "|---|---:|" + "---:|" * len(runs) + "---:|"]
    for name, mask in masks.items():
        vals = " | ".join(f"{group_rows[name]['mae'][r['run']]:.6f}" for r in runs)
        glines.append(f"| {name} | {int(mask.sum())} | {vals} | {group_rows[name]['mae']['DCC-L_s0_historical']:.6f} |")
    (L.ANALYSIS / "groups.md").write_text("\n".join(glines) + "\n")

    # ---------------- mechanism ----------------
    mechanism = []
    for r in runs:
        try:
            m, aux = _model_probe(r)
            mechanism.append(m)
        except Exception as exc:  # noqa: BLE001
            mechanism.append({"run": r["run"], "error": f"{type(exc).__name__}: {exc}"})
    L.write_json(L.ANALYSIS / "mechanism.json", {"rows": mechanism})

    # ---------------- per-molecule csv ----------------
    keys = list(per.keys())
    with open(L.ANALYSIS / "per_molecule_valid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for i in range(len(per["row"])):
            w.writerow([per[k][i] for k in keys])

    # ---------------- summary ----------------
    summary = {
        "round": L.ROUND_ID,
        "git_commit": L.C.git_commit(),
        "runs": [r["run"] for r in runs],
        "main_table": main_rows,
        "contrasts": contrast_rows,
        "groups": group_rows,
        "mechanism": mechanism,
        "anchor": {"DCC_L_valid_4seed_mean": DCCL_4SEED_VALID,
                   "DCC_L_valid_seed0": DCCL_S0_VALID},
        "official_test_loaded": False,
        "created": L.now(),
    }
    L.write_json(L.ANALYSIS / "summary.json", summary)
    L.log(f"analyzed {len(runs)} runs -> {L.track_rel(L.ANALYSIS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
