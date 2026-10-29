"""Analysis / frozen metrics for the SSRC research tree v1.

Subcommands
-----------
``freeze``   freeze the historical baseline numbers, the fixed worst-10 Monitor
             rows and the structural-support tiers (no GPU, no Locked labels).
``tables``   build ``analysis/main_table.csv``, ``analysis/slices.csv``,
             ``analysis/contrasts.json`` and ``analysis/budget.json`` from the
             run directories.
``decision`` evaluate the frozen Stage III rules on the runs that exist.
``locked``   one-shot Locked1k read, only after ``analysis/locked_freeze.json``
             exists (frozen checkpoints + hashes).

Nothing here loads or scores the official test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "cma", HERE.parent / "spc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import xs_common as C  # noqa: E402
import xs_metrics as MET  # noqa: E402

DCCL_FIT8K = {0: C.DCCL_FIT8K_MONITOR[0], 1: C.DCCL_FIT8K_MONITOR[1]}


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def freeze() -> dict:
    split = C.split_rows()
    C.verify_split_hashes(split["rows"])
    cache = C.load_cache("train")
    y = np.asarray(cache["labels"], dtype=np.float64)
    mon_rows = split["rows"]["monitor"]
    y_mon = y[mon_rows]
    fit_rows = split["rows"]["fit8k"]

    dcc = {}
    for seed in (0, 1):
        path = C.DCCL_RUNS / f"fit8k_s{seed}" / "predictions_best_monitor.pt"
        obj = torch.load(path, map_location="cpu", weights_only=False)
        rows = np.asarray(obj["monitor_rows"])
        if not np.array_equal(rows, mon_rows):
            raise RuntimeError(f"DCC-L monitor rows mismatch for seed {seed}")
        pred = np.asarray(obj["monitor_pred"], dtype=np.float64)
        dcc[seed] = {
            "monitor_mae_recomputed": MET.mae(pred, y_mon),
            "monitor_mae_train_json": DCCL_FIT8K[seed],
            "source": str(path.relative_to(C.REPO)),
            "sha256": C.sha256_file(path),
            "pred": pred,
        }
    # fixed worst-10 from the frozen DCC-L seed-0 best-Monitor checkpoint
    err = np.abs(dcc[0]["pred"] - y_mon)
    worst10_pos = np.argsort(-err, kind="stable")[:10]
    worst10_rows = [int(mon_rows[i]) for i in worst10_pos]

    tiers = MET.support_tiers(cache, fit_rows, cache, mon_rows)
    row172_pos = 172
    if row172_pos >= len(mon_rows):
        raise RuntimeError("Monitor list shorter than 172 rows")
    row172_row = int(mon_rows[row172_pos])

    payload = {
        "round": C.ROUND_ID,
        "created": C.now(),
        "git_commit": C.git_commit(),
        "recipes": dict(C.RECIPE),
        "parameter_cap": C.PARAM_CAP,
        "data_boundary": {
            "official_test_loaded": False,
            "official_valid_loaded": False,
            "caches_used": ["ad_train.pt (10k official-train)"],
            "internal_split": "bjoc_predictive_bottleneck_v1/split_a.json",
            "split_hashes": split["hashes"],
            "label_hashes": C.label_hashes(cache, split["rows"]),
        },
        "historical_reference": {
            "dcc_l_fit8k_monitor_mae": {str(k): v["monitor_mae_train_json"] for k, v in dcc.items()},
            "dcc_l_fit8k_monitor_mae_recomputed": {str(k): v["monitor_mae_recomputed"] for k, v in dcc.items()},
            "dcc_l_fit8k_prediction_files": {str(k): {"path": v["source"], "sha256": v["sha256"]}
                                             for k, v in dcc.items()},
            "note": "same split / same recipe / same regime (res A100-40GB) reference",
        },
        "fixed_worst10_monitor": {
            "selection": "top-10 |error| of the frozen DCC-L fit8k seed-0 best-Monitor "
                         "checkpoint, fixed before any run of this round",
            "positions": [int(i) for i in worst10_pos],
            "rows": worst10_rows,
            "abs_errors": [float(err[i]) for i in worst10_pos],
            "y": [float(y_mon[i]) for i in worst10_pos],
        },
        "fixed_row172": {
            "definition": "row at fixed position 172 of the frozen Monitor row list "
                          "(identity by row number, never chosen by y)",
            "position": row172_pos,
            "row": row172_row,
            "y": float(y_mon[row172_pos]),
        },
        "monitor_slices": {
            "n": int(len(mon_rows)),
            "n_ordinary": int(tiers["ordinary"].sum()),
            "n_tail": int(tiers["tail"].sum()),
            "n_any_unseen": int(tiers["any_unseen"].sum()),
            "n_unseen_ordinary": int(tiers["unseen_ordinary"].sum()),
            "n_seen_ordinary": int(tiers["seen_ordinary"].sum()),
            "fit8k_radius2_vocab_size": int(tiers["fit_vocab_size"]),
            "unseen_frac_mean": float(tiers["unseen_frac"].mean()),
        },
    }
    C.write_json(C.ANALYSIS / "frozen_baseline.json", payload)

    np.savez(
        C.ANALYSIS / "monitor_slices.npz",
        rows=np.asarray(mon_rows),
        y=y_mon,
        ordinary=tiers["ordinary"],
        tail=tiers["tail"],
        any_unseen=tiers["any_unseen"],
        unseen_ordinary=tiers["unseen_ordinary"],
        seen_ordinary=tiers["seen_ordinary"],
        unseen_frac=tiers["unseen_frac"],
        fixed_worst10_positions=np.asarray(worst10_pos),
        row172_position=np.asarray([row172_pos]),
    )
    C.log(f"[freeze] DCC-L monitor {dcc[0]['monitor_mae_recomputed']:.6f}/"
          f"{dcc[1]['monitor_mae_recomputed']:.6f}; monitor n={len(mon_rows)} "
          f"ordinary={int(tiers['ordinary'].sum())} any_unseen={int(tiers['any_unseen'].sum())} "
          f"unseen_ordinary={int(tiers['unseen_ordinary'].sum())}")
    return payload


def _slices() -> dict:
    path = C.ANALYSIS / "monitor_slices.npz"
    if not path.exists():
        raise FileNotFoundError("run `freeze` first")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _valid_slices() -> dict:
    """Same frozen slice *definitions* applied to the official valid1k rows.

    Built once, deterministically, from the frozen caches: support (radius-2
    environment vocabulary) is measured against Fit8k, exactly as for Monitor.
    The two Monitor-only slices (fixed worst-10, fixed row172) are not defined
    here and are reported as N/A for valid-monitor runs.
    """
    path = C.ANALYSIS / "valid_slices.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    split = C.split_rows()
    fit_rows, mon_rows = split["rows"]["fit8k"], split["rows"]["monitor"]
    train_cache = C.load_cache("train")
    valid_cache = C.load_cache("valid")
    rows = np.arange(len(valid_cache["labels"]), dtype=np.int64)
    y = np.asarray(valid_cache["labels"], dtype=np.float64)[rows]
    tiers = MET.support_tiers(train_cache, fit_rows, valid_cache, rows)
    # the internal Monitor rows are legitimate *support* for valid-row novelty:
    # they are part of the official train subset, and no labels are used.
    tiers_mon = MET.support_tiers(train_cache, np.union1d(fit_rows, mon_rows), valid_cache, rows)
    C.write_json(C.ANALYSIS / "valid_freeze.json", {
        "created": C.now(),
        "n": int(len(rows)),
        "label_sha256": C.sha256_array(y.astype(np.float32)),
        "n_ordinary": int(tiers["ordinary"].sum()),
        "n_tail": int(tiers["tail"].sum()),
        "n_any_unseen": int(tiers["any_unseen"].sum()),
        "n_unseen_ordinary": int(tiers["unseen_ordinary"].sum()),
        "support_set": "fit8k",
        "n_any_unseen_support_fit8k_plus_monitor": int(tiers_mon["any_unseen"].sum()),
        "official_test_loaded": False,
    })
    np.savez(
        path,
        rows=rows, y=y,
        ordinary=tiers["ordinary"], tail=tiers["tail"],
        any_unseen=tiers["any_unseen"], unseen_ordinary=tiers["unseen_ordinary"],
        seen_ordinary=tiers["seen_ordinary"], unseen_frac=tiers["unseen_frac"],
    )
    C.log(f"[freeze] valid1k n={len(rows)} ordinary={int(tiers['ordinary'].sum())} "
          f"any_unseen={int(tiers['any_unseen'].sum())} "
          f"unseen_ordinary={int(tiers['unseen_ordinary'].sum())}")
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _runs() -> list[dict]:
    """Formal runs only: directory name must start with ``stage`` (smoke/timing
    runs are named ``stage*`` too but are marked ``smoke`` and skipped below)."""
    out = []
    for d in sorted(C.RUNS.iterdir()):
        if not d.is_dir() or not d.name.startswith("stage"):
            continue
        path = d / "train.json"
        if not path.exists():
            continue
        info = C.read_json(path)
        info["run_dir"] = d.name
        out.append(info)
    return out


def _stage_of(run: str) -> str:
    for pref, stage in (("stage1", "I_cma_seed0"), ("stage2", "II_spc_seed0"),
                        ("stage3", "III_seed1"), ("stage4", "IV_train10k")):
        if run.startswith(pref):
            return stage
    return "other"


def tables() -> dict:
    sl0 = _slices()
    packs = {"monitor": sl0, "valid": _valid_slices()}
    tier_keys = ("ordinary", "tail", "any_unseen", "unseen_ordinary", "seen_ordinary")

    rows_main, rows_slice, contrasts = [], [], []
    # timing / smoke jobs are not ranked but their cost is part of the budget
    budget = {"timing_smoke": {"runs": [], "gpu_hours": 0.0, "wall_seconds": 0.0}}
    for info in _runs():
        if not info.get("smoke"):
            continue
        budget["timing_smoke"]["runs"].append(info["run_dir"])
        budget["timing_smoke"]["gpu_hours"] += float(info["gpu_hours"])
        budget["timing_smoke"]["wall_seconds"] += float(info["wall_seconds"])
    for info in _runs():
        run = info["run_dir"]
        if info.get("smoke"):
            continue
        best = torch.load(C.RUNS / run / "predictions_best_monitor.pt", map_location="cpu",
                          weights_only=False)
        final = torch.load(C.RUNS / run / "predictions_final.pt", map_location="cpu",
                           weights_only=False)
        sl = packs[info["monitor_split"]]
        y = sl["y"]
        tiers = {k: sl[k] for k in tier_keys}
        mon_rows = sl["rows"]
        if not np.array_equal(np.asarray(best["rows"]), mon_rows):
            raise RuntimeError(f"{run}: monitor rows differ from the frozen list")
        pred_b = np.asarray(best["pred"], dtype=np.float64)
        pred_f = np.asarray(final["pred"], dtype=np.float64)
        stats = MET.curve_stats(info["curve"])
        entry = {
            "run": run,
            "family": info["family"],
            "arm": info["arm"],
            "seed": info["seed"],
            "fit": info["fit"],
            "monitor_split": info["monitor_split"],
            "stage": _stage_of(run),
            "parameters": info["parameters"],
            "best_monitor_mae": info["best_monitor"]["monitor_mae"],
            "best_epoch": info["best_monitor"]["epoch"],
            "rolling20_min_median": stats["rolling20_min_median"],
            "tail100_median": stats["tail100_median"],
            "tail20_median": stats["tail20_median"],
            "final_monitor_mae": info["final_monitor_mae"],
            "eval_train_mae_best": info["reads"]["best_monitor"]["eval_train_mae"],
            "eval_train_mae_final": info["reads"]["final"]["eval_train_mae"],
            "gpu_hours": info["gpu_hours"],
            "git_commit": info["git_commit"],
            "checkpoint_best_sha256": info["checkpoint_sha256"]["best_monitor"],
            "started_at": info["started_at"],
            "finished_at": info["finished_at"],
        }
        # selection-relevant slices on the best-Monitor checkpoint
        for tag, pred in (("best", pred_b), ("final", pred_f)):
            base = MET.slice_table(pred, y, tiers=tiers)
            named = dict(base)
            if info["monitor_split"] == "monitor":
                worst10 = sl["fixed_worst10_positions"]
                row172 = int(sl["row172_position"][0])
                drop = MET.slice_table(pred, y, tiers=tiers, drop_rows=worst10)
                drop_r = MET.slice_table(pred, y, tiers=tiers, row_index=row172)
                named["minus_fixed_worst10"] = drop["full"]
                named["minus_row172"] = drop_r["minus_row172"]
            else:  # Monitor-only slices are not defined on valid rows
                named["minus_fixed_worst10"] = None
                named["minus_row172"] = None
            if tag == "best":
                entry.update(named)
            for k, v in named.items():
                rows_slice.append({"run": run, "checkpoint": tag, "slice": k, "n": None, "mae": v})
        rows_main.append(entry)
        stage = entry["stage"]
        budget.setdefault(stage, {"runs": [], "gpu_hours": 0.0, "wall_seconds": 0.0})
        budget[stage]["runs"].append(run)
        budget[stage]["gpu_hours"] += float(info["gpu_hours"])
        budget[stage]["wall_seconds"] += float(info["wall_seconds"])

    # matched REAL - NULL contrasts (same family / seed / fit / monitor)
    by_key: dict[tuple, dict] = {}
    for info in _runs():
        if info.get("smoke"):
            continue
        key = (info["family"], info["seed"], info["fit"], info["monitor_split"])
        by_key.setdefault(key, {})[info["arm"]] = info
    for key, pair in sorted(by_key.items()):
        if set(pair) != {"REAL", "MEAN"}:
            continue
        family, seed, fit, monsplit = key
        pb = torch.load(C.RUNS / pair["REAL"]["run_dir"] / "predictions_best_monitor.pt",
                        map_location="cpu", weights_only=False)
        pm = torch.load(C.RUNS / pair["MEAN"]["run_dir"] / "predictions_best_monitor.pt",
                        map_location="cpu", weights_only=False)
        a = np.asarray(pb["pred"], dtype=np.float64)
        b = np.asarray(pm["pred"], dtype=np.float64)
        sl = packs[monsplit]
        y = sl["y"]
        tiers = {k: sl[k] for k in tier_keys}
        masks = {
            "full": np.ones(len(y), bool),
            "ordinary": tiers["ordinary"],
            "unseen_ordinary": tiers["unseen_ordinary"],
            "seen_ordinary": tiers["seen_ordinary"],
            "tail": tiers["tail"],
        }
        if monsplit == "monitor":
            keep_w10 = np.ones(len(y), bool); keep_w10[sl["fixed_worst10_positions"]] = False
            keep_172 = np.ones(len(y), bool); keep_172[int(sl["row172_position"][0])] = False
            masks["minus_fixed_worst10"] = keep_w10
            masks["minus_row172"] = keep_172
        rec = {"family": family, "seed": int(seed), "fit": fit, "monitor_split": monsplit,
               "real_run": pair["REAL"]["run_dir"], "mean_run": pair["MEAN"]["run_dir"],
               "delta_real_minus_mean": {}}
        for name, mask in masks.items():
            rec["delta_real_minus_mean"][name] = MET.paired_delta(a, b, y, mask)
        # tail-only check: improvement present in ordinary molecules?
        contrast = MET.paired_delta(a, b, y, tiers["ordinary"])["delta"]
        contrast_tail = MET.paired_delta(a, b, y, tiers["tail"])["delta"]
        rec["tail_only"] = bool(contrast is not None and contrast_tail is not None
                                and contrast >= 0 and contrast_tail < 0)
        contrasts.append(rec)
        if fit == "fit8k":
            budget.setdefault("contrasts", {"runs": [], "gpu_hours": 0.0, "wall_seconds": 0.0})
            budget["contrasts"]["runs"].append(f"{family}_s{seed}")

    # write csv
    C.ensure_dir(C.ANALYSIS)
    if rows_main:
        keys = list(rows_main[0].keys())
        with open(C.ANALYSIS / "main_table.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows_main:
                w.writerow(r)
    with open(C.ANALYSIS / "slices.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "checkpoint", "slice", "mae"])
        w.writeheader()
        for r in rows_slice:
            w.writerow({"run": r["run"], "checkpoint": r["checkpoint"], "slice": r["slice"],
                        "mae": r["mae"]})
    C.write_json(C.ANALYSIS / "contrasts.json", {
        "definition": "delta = MAE(REAL) - MAE(MEAN); negative means REAL is better; "
                      "paired bootstrap 5000 resamples over Monitor rows",
        "contrasts": contrasts,
    })

    caps = {"I_cma_seed0": 4.0, "II_spc_seed0": 4.0, "III_seed1": 3.0, "IV_train10k": 3.0,
            "reserve": 2.0}
    total = sum(v["gpu_hours"] for v in budget.values() if isinstance(v, dict))
    budget_doc = {
        "round": C.ROUND_ID,
        "created": C.now(),
        "soft_cap_gpu_hours": 14.0,
        "hard_cap_gpu_hours": 16.0,
        "caps_by_stage": caps,
        "stages": {k: {"runs": v["runs"], "gpu_hours": round(v["gpu_hours"], 4),
                       "wall_seconds": round(v["wall_seconds"], 1),
                       "cap_gpu_hours": caps.get(k)}
                   for k, v in budget.items()},
        "total_gpu_hours": round(total, 4),
        "total_wall_seconds": round(sum(v["wall_seconds"] for v in budget.values()), 1),
        "within_soft_cap": bool(total <= 14.0),
        "within_hard_cap": bool(total <= 16.0),
    }
    C.write_json(C.ANALYSIS / "budget.json", budget_doc)
    C.log(f"[tables] {len(rows_main)} runs, {len(contrasts)} matched contrasts, "
          f"{budget_doc['total_gpu_hours']:.3f} GPU-h")
    return budget_doc


# ---------------------------------------------------------------------------
# decision inputs (Stage III rules, frozen in PROTOCOL.md)
# ---------------------------------------------------------------------------


def decision() -> dict:
    frozen = C.read_json(C.ANALYSIS / "frozen_baseline.json")
    contrasts = C.read_json(C.ANALYSIS / "contrasts.json")["contrasts"]
    main = list(csv.DictReader(open(C.ANALYSIS / "main_table.csv")))
    by_run = {r["run"]: r for r in main}
    dccl = {int(k): float(v) for k, v in frozen["historical_reference"]["dcc_l_fit8k_monitor_mae"].items()}
    out = {"rules": {
        "matched_full_delta_max": -0.004,
        "matched_ordinary_delta_max": 0.0,
        "matched_unseen_ordinary_delta_max": -0.004,
        "absolute_monitor_max_above_dccl": 0.010,
        "smoothing_must_agree": True,
        "mechanism_only_delta": -0.010,
    }, "families": {}}
    for c in contrasts:
        if c["fit"] != "fit8k" or c["seed"] != 0:
            continue
        fam = c["family"]
        real = by_run[c["real_run"]]
        d = c["delta_real_minus_mean"]
        checks = {
            "full": d["full"]["delta"] <= -0.004,
            "ordinary": d["ordinary"]["delta"] <= 0.0,
            "unseen_ordinary": d["unseen_ordinary"]["delta"] <= -0.004,
            "absolute": float(real["best_monitor_mae"]) <= dccl[0] + 0.010,
            "smoothing": (float(real["rolling20_min_median"]) - float(real["best_monitor_mae"])) <= 0.0,
        }
        out["families"][fam] = {
            "checks": checks,
            "passed": bool(all(checks.values())),
            "real_monitor": float(real["best_monitor_mae"]),
            "mean_monitor": float(by_run[c["mean_run"]]["best_monitor_mae"]),
            "matched_delta": d["full"],
        }
    C.write_json(C.ANALYSIS / "decision_inputs.json", out)
    C.log(f"[decision] {json.dumps({k: v['passed'] for k, v in out['families'].items()})}")
    return out


# ---------------------------------------------------------------------------
# one-shot Locked1k read (Stage III, only after the freeze file exists)
# ---------------------------------------------------------------------------


def locked() -> dict:
    from xs_train import SplitData, predict

    freeze_path = C.ANALYSIS / "locked_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("locked_freeze.json must exist before any Locked label is read")
    frozen = C.read_json(freeze_path)
    split = C.split_rows()
    cache = C.load_cache("train")
    locked_rows = split["rows"]["locked"]
    norm_by_fit = {}
    out = {"frozen_at": frozen["frozen_at"], "entries": []}
    for entry in frozen["checkpoints"]:
        run = entry["run"]
        info = C.read_json(C.RUNS / run / "train.json")
        if C.sha256_file(C.RUNS / run / "best_monitor.pt") != entry["checkpoint_sha256"]:
            raise RuntimeError(f"{run}: checkpoint hash changed after freeze")
        family = info["family"]
        M = __import__(f"{family}_model")
        if info["fit"] == "fit8k":
            norm = info["norm"]
            norm = {"global_mean": np.asarray(norm["global_mean"], np.float32),
                    "global_scale": np.asarray(norm["global_scale"], np.float32)}
        else:
            norm = C.fit_only_norm(cache, np.arange(int(cache["n_molecules"])))
        model, _ = M.build_model(norm, info["seed"], arm=info["arm"], device="cpu",
                                 ssl_seed=info["seed"])
        state = torch.load(C.RUNS / run / "best_monitor.pt", map_location="cpu",
                           weights_only=False)["state_dict"]
        C.load_state(model, state, "cpu")
        ds = SplitData(cache, family, "train", "cpu")
        pred = predict(model, ds, locked_rows, norm, device="cpu")
        y = np.asarray(cache["labels"], np.float64)[locked_rows]
        err = np.abs(pred - y)
        out["entries"].append({
            "run": run, "family": family, "arm": info["arm"], "seed": info["seed"],
            "locked_mae": float(err.mean()), "locked_pred": pred.tolist(),
            "locked_rows": locked_rows.tolist(),
        })
    C.write_json(C.ANALYSIS / "locked_read.json", out)
    C.log(f"[locked] {[(e['run'], round(e['locked_mae'], 6)) for e in out['entries']]}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["freeze", "tables", "decision", "locked"])
    args = p.parse_args(argv)
    {"freeze": freeze, "tables": tables, "decision": decision, "locked": locked}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
