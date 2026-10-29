"""Valid-only analysis for CJP-SSL v1 (official test is never loaded).

Writes
------
``analysis/pretext_summary.json``
``analysis/supervised_summary.json``
``analysis/paired_comparisons.json``
``analysis/per_molecule_valid.csv``

The paired contrasts are fixed before the runs:

1. ``A-REPLAY - SCRATCH``   value of the historical atom/bond masked SSL
2. ``MARG - A-REPLAY``      value of pretraining LocalContent
3. ``JOINT - MARG``         value of the true joint conditional
4. ``JOINT - A-REPLAY``     net effect

All contrasts are same-seed, same valid molecules, paired per row.  Subgroups
(ordinary/tail/radius-2 unseen, row172 removed, fixed historical worst-10
removed) are diagnostic and never replace the full-VM primary metric.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cjp_common as C

L = C.L

ANCHOR_CSV = C.REPO / "tracks/structural-sufficiency/zinc_performance_search_v1/analysis/per_molecule_valid.csv"
FROZEN_CONFIG = C.CONFIGS / "cjp_ssl_v1.json"
ROW172 = 172

PAIRS = (
    ("A-REPLAY", "SCRATCH"),
    ("MARG-SSL", "A-REPLAY"),
    ("JOINT-SSL", "MARG-SSL"),
    ("JOINT-SSL", "A-REPLAY"),
)
ARMS = ("A-REPLAY", "SCRATCH", "MARG-SSL", "JOINT-SSL")
PREFIX = {"A-REPLAY": "A_replay", "SCRATCH": "A_scratch",
          "MARG-SSL": "A_margssl", "JOINT-SSL": "A_jointssl"}
PRETRAIN = {"MARG-SSL": "pretrain_marg", "JOINT-SSL": "pretrain_joint"}

GROUPS = (
    "full",
    "minus_row172",
    "minus_historical_lca_worst10",
    "ordinary_y_gt_-3",
    "tail_y_le_-3",
    "r2_unseen_ordinary",
    "r2_unseen",
    "seen_ordinary",
    "ring_rich_ge4_unseen",
)


# ---------------------------------------------------------------------------
# anchor rows and fixed masks
# ---------------------------------------------------------------------------


def read_anchor() -> dict[str, Any]:
    with open(ANCHOR_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["row_id"]))
    return {
        "row_id": np.asarray([int(r["row_id"]) for r in rows], dtype=np.int64),
        "y": np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64),
        "unseen": np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows]) > 0,
        "n_rings": np.asarray([int(float(r["n_rings"])) for r in rows], dtype=np.int64),
    }


def fixed_masks(anchor: Mapping[str, Any]) -> dict[str, np.ndarray]:
    cfg = json.loads(FROZEN_CONFIG.read_text())
    worst_rows = np.zeros(len(anchor["y"]), dtype=bool)
    for r in cfg["diagnostic_rows"]["historical_lca_s0_worst10"]:
        worst_rows[int(r)] = True
    y = anchor["y"]
    unseen = anchor["unseen"]
    n = np.arange(len(y))
    return {
        "full": np.ones(len(y), dtype=bool),
        "minus_row172": n != ROW172,
        "minus_historical_lca_worst10": ~worst_rows,
        "ordinary_y_gt_-3": y > -3.0,
        "tail_y_le_-3": y <= -3.0,
        "r2_unseen_ordinary": unseen & (y > -3.0),
        "r2_unseen": unseen,
        "seen_ordinary": (~unseen) & (y > -3.0),
        "ring_rich_ge4_unseen": (anchor["n_rings"] >= 4) & unseen,
        "row172": n == ROW172,
        "historical_lca_worst10": worst_rows,
    }


# ---------------------------------------------------------------------------
# run loading
# ---------------------------------------------------------------------------


def load_supervised(run_dir: Path, anchor: Mapping[str, Any]) -> dict[str, Any] | None:
    train_json = run_dir / "train.json"
    pred_path = run_dir / "predictions/valid_best.npz"
    if not train_json.exists() or not pred_path.exists():
        return None
    payload = C.read_json(train_json)
    pred = np.load(pred_path)
    rows = np.asarray(pred["row"], dtype=np.int64)
    p = np.asarray(pred["pred"], dtype=np.float64)
    if not np.array_equal(rows, anchor["row_id"]):
        raise RuntimeError(f"{run_dir}: prediction rows are not the anchor rows")
    ae = np.abs(p - anchor["y"])
    return {"dir": run_dir, "payload": payload, "pred": p, "ae": ae,
            "best_valid_mae": float(ae.mean())}


def _window(curve: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    vals = [float(c["valid_mae"]) for c in curve]
    w20 = None
    if len(vals) >= 20:
        w20 = min(float(np.median(vals[i:i + 20])) for i in range(len(vals) - 19))
    t100 = float(np.median(vals[-min(100, len(vals)):])) if vals else None
    return {"W20": w20, "T100": t100, "final": float(vals[-1]) if vals else None}


def supervised_summary(anchor, masks, runs) -> dict[str, Any]:
    out = {"primary_metric": "official valid 1000 raw-y MAE (lower is better)",
           "official_test_loaded": False, "rows": [], "coverage": {}}
    for arm in ARMS:
        arm_rows = []
        for seed in (0, 1):
            key = (arm, seed)
            if key not in runs:
                continue
            r = runs[key]
            payload = r["payload"]
            window = _window(payload["curve"])
            groups = {g: float(r["ae"][masks[g]].mean()) for g in GROUPS}
            groups["row172"] = float(r["ae"][masks["row172"]].mean())
            arm_rows.append({
                "arm": arm,
                "seed": int(seed),
                "run": r["dir"].name,
                "best_valid_mae": float(r["best_valid_mae"]),
                "best_epoch": int(payload["best"]["epoch"]),
                "W20": window["W20"],
                "T100": window["T100"],
                "final_valid_mae": window["final"],
                "train_mae_eval_at_best": float(payload["best_train_mae_eval"]),
                "train_mae_eval_at_final": float(payload["final_train_mae_eval"]),
                "parameters": int(payload["model_info"]["parameters"]),
                "updates": int(payload["total_updates"]),
                "epochs": int(payload["epochs"]),
                "wall_seconds": float(payload["wall_seconds"]),
                "gpu_hours": float(payload["gpu_hours_estimate"]),
                "best_checkpoint_sha256": payload["checkpoint_sha256"]["best"],
                "final_checkpoint_sha256": payload["checkpoint_sha256"]["final"],
                "init_state_sha256": payload["model_info"]["initial_state_sha256"],
                "group_mae": groups,
                "cjp": payload.get("cjp", {}),
                "git_commit": payload.get("git_commit"),
            })
        if arm_rows:
            out["rows"].extend(arm_rows)
            out["coverage"][arm] = {
                "seeds": [r["seed"] for r in arm_rows],
                "mean_best_valid_mae": float(np.mean([r["best_valid_mae"] for r in arm_rows])),
            }
    return out


# ---------------------------------------------------------------------------
# paired contrasts
# ---------------------------------------------------------------------------


def bootstrap_delta(d: np.ndarray, *, n_boot: int = 10000, seed: int = 20261108) -> dict[str, Any]:
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_delta": float(d.mean()), "ci95": [float(lo), float(hi)], "n": int(n)}


def contrast(ae_first: np.ndarray, ae_second: np.ndarray, masks) -> dict[str, Any]:
    d = np.asarray(ae_first, np.float64) - np.asarray(ae_second, np.float64)
    base = bootstrap_delta(d)
    order = np.argsort(-np.abs(d))
    drop = np.ones(len(d), dtype=bool)
    drop[order[:10]] = False
    out = {
        "mean_delta": base["mean_delta"],
        "ci95": base["ci95"],
        "n": base["n"],
        "first_better_fraction": float((d < 0).mean()),
        "median_abs_error_delta": float(np.median(d)),
        "drop_top10_abs_delta": float(d[drop].mean()),
        "group_mean_delta": {g: float(d[masks[g]].mean()) for g in GROUPS},
        "group_mae_first": {g: float(np.asarray(ae_first)[masks[g]].mean()) for g in GROUPS},
        "group_mae_second": {g: float(np.asarray(ae_second)[masks[g]].mean()) for g in GROUPS},
    }
    return out


def paired_comparisons(runs, masks) -> dict[str, Any]:
    out = {"pairs": {}, "official_test_loaded": False}
    for first, second in PAIRS:
        entry: dict[str, Any] = {"per_seed": {}, "seed_mean": None,
                                 "sign_convention": "delta = MAE(first) - MAE(second); "
                                                    "negative means first is better"}
        per_seed_delta = []
        for seed in (0, 1):
            if (first, seed) in runs and (second, seed) in runs:
                c = contrast(runs[(first, seed)]["ae"], runs[(second, seed)]["ae"], masks)
                c["first_best_valid_mae"] = runs[(first, seed)]["best_valid_mae"]
                c["second_best_valid_mae"] = runs[(second, seed)]["best_valid_mae"]
                entry["per_seed"][int(seed)] = c
                per_seed_delta.append(np.asarray(runs[(first, seed)]["ae"])
                                      - np.asarray(runs[(second, seed)]["ae"]))
        if per_seed_delta:
            mean_d = np.mean(np.stack(per_seed_delta, axis=0), axis=0)
            sm = bootstrap_delta(mean_d)
            sm["group_mean_delta"] = {g: float(mean_d[masks[g]].mean()) for g in GROUPS}
            sm["seeds"] = len(per_seed_delta)
            entry["seed_mean"] = sm
        out["pairs"][f"{first}_minus_{second}"] = entry
    return out


# ---------------------------------------------------------------------------
# pretext summary
# ---------------------------------------------------------------------------


def load_pretrain() -> dict[str, Any]:
    out: dict[str, Any] = {"runs": {}, "same_samples_across_arms": {},
                           "official_test_loaded": False}
    for arm, prefix in PRETRAIN.items():
        for seed in (0, 1):
            d = C.RUNS / f"{prefix}_s{seed}"
            path = d / "pretrain.json"
            if not path.exists():
                continue
            payload = C.read_json(path)
            out["runs"][f"{arm}_s{seed}"] = {
                "arm": arm,
                "seed": seed,
                "run": d.name,
                "updates_done": payload["updates_done"],
                "masked_batch_sha256": payload["data"]["masked_batch_sha256"],
                "first_batch_query_sha256": payload["data"]["first_batch_query_sha256"],
                "split": payload["data"]["split"],
                "prior_stats": payload["prior_stats"],
                "diagnostics_before": payload["diagnostics_before"],
                "diagnostics_after": payload["diagnostics_after"],
                "train": payload["train"],
                "encoder_bit_identical": payload["encoder_bit_identical"],
                "local_state": payload["local_state"],
                "pretext_head": payload["pretext_head"],
                "timing": payload["timing"],
                "model_info": payload["model_info"],
                "git_commit": payload["git_commit"],
            }
    for seed in (0, 1):
        a = out["runs"].get(f"MARG-SSL_s{seed}")
        b = out["runs"].get(f"JOINT-SSL_s{seed}")
        if a and b:
            out["same_samples_across_arms"][int(seed)] = {
                "masked_batch_sha256_equal": a["masked_batch_sha256"] == b["masked_batch_sha256"],
                "first_batch_query_sha256_equal":
                    a["first_batch_query_sha256"] == b["first_batch_query_sha256"],
                "split_equal": a["split"] == b["split"],
                "mask_coverage_equal":
                    a["train"]["mask_coverage"] == b["train"]["mask_coverage"],
            }
    return out


# ---------------------------------------------------------------------------
# pre-registered verdicts (§11 of the task book)
# ---------------------------------------------------------------------------


def _delta(pairs, first, second, seed=None, group="full") -> float | None:
    key = f"{first}_minus_{second}"
    if key not in pairs["pairs"]:
        return None
    entry = pairs["pairs"][key]
    if seed is None:
        return None if entry["seed_mean"] is None else entry["seed_mean"]["group_mean_delta"][group]
    c = entry["per_seed"].get(int(seed))
    return None if c is None else c["group_mean_delta"][group]


def _pretext_diag_ce(pretext, arm: str, seed: int) -> Optional[float]:
    p = (pretext or {}).get("runs", {}).get(f"{arm}_s{seed}")
    if not p:
        return None
    try:
        return float(p["diagnostics_after"]["diag"]["molecule_mean_ce"])
    except Exception:  # pragma: no cover
        return None


def verdicts(pairs, pretext=None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("MARG-SSL", "JOINT-SSL"):
        d0 = _delta(pairs, arm, "A-REPLAY", 0)
        d1 = _delta(pairs, arm, "A-REPLAY", 1)
        dm = _delta(pairs, arm, "A-REPLAY", group="full")
        out[arm] = {
            "vs_A_REPLAY": {
                "seed0_delta": d0, "seed1_delta": d1, "seed_mean_delta": dm,
                "both_seeds_improve": bool(d0 is not None and d1 is not None
                                           and d0 < 0 and d1 < 0),
                "seed_mean_gain_ge_0.003": bool(dm is not None and dm <= -0.003),
                "ordinary_delta_seed_mean": _delta(pairs, arm, "A-REPLAY",
                                                   group="ordinary_y_gt_-3"),
                "minus_row172_delta_seed_mean": _delta(pairs, arm, "A-REPLAY",
                                                       group="minus_row172"),
                "r2_unseen_ordinary_delta_seed_mean": _delta(
                    pairs, arm, "A-REPLAY", group="r2_unseen_ordinary"),
            }
        }
    j0 = _delta(pairs, "JOINT-SSL", "MARG-SSL", 0)
    j1 = _delta(pairs, "JOINT-SSL", "MARG-SSL", 1)
    jm = _delta(pairs, "JOINT-SSL", "MARG-SSL", group="full")
    out["JOINT_vs_MARG"] = {
        "seed0_delta": j0, "seed1_delta": j1, "seed_mean_delta": jm,
        "same_direction": bool(j0 is not None and j1 is not None and j0 * j1 > 0),
        "seed_mean_gain_ge_0.002": bool(jm is not None and jm <= -0.002),
        "ordinary_delta_seed_mean": _delta(pairs, "JOINT-SSL", "MARG-SSL",
                                           group="ordinary_y_gt_-3"),
        "r2_unseen_ordinary_delta_seed_mean": _delta(
            pairs, "JOINT-SSL", "MARG-SSL", group="r2_unseen_ordinary"),
    }
    # absolute thresholds as fixed by the task book (§11)
    local_supported = []
    for arm in ("MARG-SSL", "JOINT-SSL"):
        v = out[arm]["vs_A_REPLAY"]
        ok = (v["both_seeds_improve"] and v["seed_mean_gain_ge_0.003"]
              and v["ordinary_delta_seed_mean"] is not None
              and v["ordinary_delta_seed_mean"] < 0
              and v["minus_row172_delta_seed_mean"] is not None
              and v["minus_row172_delta_seed_mean"] < 0
              and v["r2_unseen_ordinary_delta_seed_mean"] is not None
              and v["r2_unseen_ordinary_delta_seed_mean"] <= 0.002)
        if ok:
            local_supported.append(arm)
    joint = out["JOINT_vs_MARG"]
    joint_ok = (joint["same_direction"] and joint["seed_mean_gain_ge_0.002"]
                and joint["ordinary_delta_seed_mean"] is not None
                and joint["ordinary_delta_seed_mean"] <= 0
                and joint["r2_unseen_ordinary_delta_seed_mean"] is not None
                and joint["r2_unseen_ordinary_delta_seed_mean"] <= 0.002)
    out["CONCLUSION_A_LOCAL_SSL_SUPPORTED"] = bool(local_supported)
    out["CONCLUSION_A_arms"] = local_supported
    out["CONCLUSION_B_JOINT_BINDING_SUPPORTED"] = bool(
        "JOINT-SSL" in local_supported and joint_ok)
    out["CONCLUSION_C_MARG_ONLY_GAIN"] = bool(
        "MARG-SSL" in local_supported and not out["CONCLUSION_B_JOINT_BINDING_SUPPORTED"])
    out["CONCLUSION_D_NO_GAIN"] = bool(not local_supported)
    joint_ce = [_pretext_diag_ce(pretext, "JOINT-SSL", s) for s in (0, 1)]
    marg_ce = [_pretext_diag_ce(pretext, "MARG-SSL", s) for s in (0, 1)]
    joint_ce_mean = (None if any(v is None for v in joint_ce)
                     else float(sum(joint_ce) / 2))
    marg_ce_mean = (None if any(v is None for v in marg_ce)
                    else float(sum(marg_ce) / 2))
    out["JOINT_vs_MARG"].update({
        "joint_pretext_ce_seed_mean": joint_ce_mean,
        "marg_pretext_ce_seed_mean": marg_ce_mean,
        "joint_pretext_ce_better": bool(joint_ce_mean is not None
                                        and marg_ce_mean is not None
                                        and joint_ce_mean < marg_ce_mean),
    })
    out["JOINT_PRETEXT_LEARNED_NO_DOWNSTREAM_GAIN"] = bool(
        joint["seed_mean_delta"] is not None and joint["seed_mean_delta"] >= -0.002
        and out["JOINT_vs_MARG"]["joint_pretext_ce_better"])
    out["thresholds"] = {
        "A_seed_mean_gain": 0.003, "B_seed_mean_gain": 0.002,
        "unseen_not_worse_tolerance": 0.002,
        "note": "diagnostic thresholds; full-VM paired numbers are always reported",
    }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def write_per_molecule(anchor, runs, path: Path) -> None:
    header = ["row_id", "y_raw", "r2_unseen"] + [f"{a}_s{s}" for a in ARMS for s in (0, 1)]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(len(anchor["y"])):
            row = [int(anchor["row_id"][i]), float(anchor["y"][i]), int(anchor["unseen"][i])]
            for arm in ARMS:
                for seed in (0, 1):
                    r = runs.get((arm, seed))
                    row.append("" if r is None else f"{r['pred'][i]:.9f}")
            w.writerow(row)


# ---------------------------------------------------------------------------
# transfer / provenance checks
# ---------------------------------------------------------------------------


def transfer_checks(runs, pretext) -> dict[str, Any]:
    out: dict[str, Any] = {"pretrain_to_supervised": {}, "cross_arm_module_sharing": {},
                           "historical_ssl_checkpoint": {}}
    for arm, prefix in PRETRAIN.items():
        for seed in (0, 1):
            r = runs.get((arm, seed))
            p = pretext["runs"].get(f"{arm}_s{seed}")
            if not r or not p:
                continue
            info = r["payload"].get("cjp", {}).get("model_info", {})
            sup_local = info.get("module_sha256", {}).get("local")
            out["pretrain_to_supervised"][f"{arm}_s{seed}"] = {
                "pretrain_local_state_sha256": p["local_state"]["state_sha256"],
                "supervised_local_module_sha256": sup_local,
                "equal": bool(sup_local == p["local_state"]["state_sha256"]),
                "pretrain_encoder_bit_identical": p["encoder_bit_identical"],
            }
    for seed in (0, 1):
        present = {a: runs[(a, seed)]["payload"].get("cjp", {}).get("model_info", {})
                   for a in ARMS if (a, seed) in runs}
        if len(present) < 2:
            continue
        ref = present.get("A-REPLAY")
        entry: dict[str, Any] = {}
        for arm, info in present.items():
            if arm == "A-REPLAY":
                continue
            entry[arm] = {
                k: bool(info.get("module_sha256", {}).get(k) == ref.get("module_sha256", {}).get(k))
                for k in ("atom_encoder", "bond_encoder", "f", "rho", "local")
            } if ref else {}
        if "MARG-SSL" in present and "JOINT-SSL" in present:
            m = present["MARG-SSL"].get("module_sha256", {})
            j = present["JOINT-SSL"].get("module_sha256", {})
            entry["MARG_vs_JOINT"] = {
                "only_local_differs": bool(m.get("local") != j.get("local")
                                           and all(m.get(k) == j.get(k)
                                                   for k in m if k != "local")),
                "module_sha256": {"MARG": m, "JOINT": j},
            }
        out["cross_arm_module_sharing"][int(seed)] = entry
    for seed in (0, 1):
        try:
            ssl = C.M.ssl_path(seed)
            out["historical_ssl_checkpoint"][int(seed)] = {
                "path": C.track_rel(ssl),
                "sha256": C.sha256_file(ssl),
            }
        except Exception:  # pragma: no cover
            pass
    return out


# ---------------------------------------------------------------------------
# markdown summary (human-readable mirror of the JSON artifacts)
# ---------------------------------------------------------------------------


def _fmt(x, nd=6):
    return "" if x is None else f"{x:.{nd}f}"


def write_markdown(sup, pairs, pretext, path: Path) -> None:
    lines = ["# CJP-SSL v1 — valid-only summary", "",
             "Primary metric: official valid 1,000 raw-y MAE (lower is better).",
             "Official test was never loaded.", "", "## Supervised arms", "",
             "| arm | seed | best MAE | best ep | W20 | T100 | final | params | "
             "ordinary | tail | r2-unseen ord | -row172 | -A-worst10 | train@best |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sup["rows"]:
        g = r["group_mae"]
        lines.append(
            f"| {r['arm']} | {r['seed']} | {_fmt(r['best_valid_mae'])} | {r['best_epoch']} | "
            f"{_fmt(r['W20'])} | {_fmt(r['T100'])} | {_fmt(r['final_valid_mae'])} | "
            f"{r['parameters']} | {_fmt(g['ordinary_y_gt_-3'])} | {_fmt(g['tail_y_le_-3'])} | "
            f"{_fmt(g['r2_unseen_ordinary'])} | {_fmt(g['minus_row172'])} | "
            f"{_fmt(g['minus_historical_lca_worst10'])} | {_fmt(r['train_mae_eval_at_best'])} |"
        )
    lines += ["", "## Paired contrasts (delta = first - second; negative = first better)", "",
              "| contrast | seed | full delta | CI95 | ordinary | r2-unseen ord | -row172 | "
              "first-better frac | median delta |", "|---|---|---|---|---|---|---|---|---|"]
    for name, entry in pairs["pairs"].items():
        for seed, c in entry["per_seed"].items():
            lines.append(
                f"| {name.replace('_minus_', ' - ')} | {seed} | {_fmt(c['mean_delta'])} | "
                f"[{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | "
                f"{_fmt(c['group_mean_delta']['ordinary_y_gt_-3'])} | "
                f"{_fmt(c['group_mean_delta']['r2_unseen_ordinary'])} | "
                f"{_fmt(c['group_mean_delta']['minus_row172'])} | "
                f"{c['first_better_fraction']:.3f} | {_fmt(c['median_abs_error_delta'])} |"
            )
        sm = entry.get("seed_mean")
        if sm:
            lines.append(
                f"| {name.replace('_minus_', ' - ')} | mean | {_fmt(sm['mean_delta'])} | "
                f"[{sm['ci95'][0]:+.6f}, {sm['ci95'][1]:+.6f}] | "
                f"{_fmt(sm['group_mean_delta']['ordinary_y_gt_-3'])} | "
                f"{_fmt(sm['group_mean_delta']['r2_unseen_ordinary'])} | "
                f"{_fmt(sm['group_mean_delta']['minus_row172'])} | | |"
            )
    lines += ["", "## Pretext diagnostics", "",
              "| run | updates | diag CE before | diag CE after | class prior CE | "
              "cond prior CE | joint interaction RMS | eff. rank before | eff. rank after | "
              "atom mask frac | bond mask frac | paths/mol | swap delta | shuffle delta |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in pretext["runs"].items():
        b = r["diagnostics_before"]["diag"]
        a = r["diagnostics_after"]["diag"]
        cov = r["train"]["mask_coverage"]
        it = r["train"]["interventions"]
        lines.append(
            f"| {key} | {r['updates_done']} | {_fmt(b['molecule_mean_ce'])} | "
            f"{_fmt(a['molecule_mean_ce'])} | {_fmt(a.get('class_prior_ce'))} | "
            f"{_fmt(a.get('conditional_prior_ce'))} | {_fmt(a['interaction_rms'])} | "
            f"{_fmt(b['representation']['effective_rank'], 3)} | "
            f"{_fmt(a['representation']['effective_rank'], 3)} | "
            f"{cov['atom_mask_fraction']:.4f} | {cov['bond_mask_fraction']:.4f} | "
            f"{cov['paths_per_molecule']:.3f} | {_fmt(it['swap_v_w_delta'])} | "
            f"{_fmt(it['cross_molecule_shuffle_delta'])} |"
        )
    lines += ["", "## Same-samples check across arms", "",
              "```json", json.dumps(pretext["same_samples_across_arms"], indent=1), "```", ""]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CJP-SSL valid-only analysis")
    args = parser.parse_args(argv)
    anchor = read_anchor()
    masks = fixed_masks(anchor)

    runs = {}
    for arm in ARMS:
        for seed in (0, 1):
            d = C.RUNS / f"{PREFIX[arm]}_s{seed}"
            r = load_supervised(d, anchor)
            if r is not None:
                runs[(arm, seed)] = r

    sup = supervised_summary(anchor, masks, runs)
    pairs = paired_comparisons(runs, masks)
    pretext = load_pretrain()
    verdict = verdicts(pairs, pretext) if pairs["pairs"] else {}
    sup["verdicts"] = verdict
    sup["transfer_checks"] = transfer_checks(runs, pretext)
    sup["coverage"]["n_runs"] = len(runs)
    sup["coverage"]["expected_runs"] = 8
    sup["all_runs_present"] = len(runs) == 8

    C.write_json(C.ANALYSIS / "supervised_summary.json", sup)
    C.write_json(C.ANALYSIS / "paired_comparisons.json", pairs)
    C.write_json(C.ANALYSIS / "pretext_summary.json", pretext)
    write_per_molecule(anchor, runs, C.ANALYSIS / "per_molecule_valid.csv")
    write_markdown(sup, pairs, pretext, C.ANALYSIS / "summary.md")
    C.log(f"runs={len(runs)} pairs={list(pairs['pairs'])}")
    for arm in ARMS:
        cov = sup["coverage"].get(arm)
        if cov:
            C.log(f"{arm}: {cov['mean_best_valid_mae']:.6f}")
    C.log(f"verdicts: {json.dumps({k: v for k, v in verdict.items() if k.startswith('CONCLUSION')})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
