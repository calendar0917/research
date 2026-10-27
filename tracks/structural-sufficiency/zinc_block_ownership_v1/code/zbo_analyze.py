"""Frozen analysis for ``zinc_block_ownership_v1``.

Reads only official train/valid caches and the formal run directories.  The
official test split is never opened.  Produces ``analysis/zbo_analysis.json``
with:

* per-run best/final/W20/T100/train@best/train@final and provenance;
* paired per-molecule ΔMAE (positive = reference arm better) with bootstrap CI;
* ordinary / tail / radius-2 support / rings>=4 / fixed worst-10 / drop-top-10 /
  drop-row172 / block-count / novelty groups;
* the pre-registered seed-0 and two-seed decision checks.

Usage:
    python .../zbo_analyze.py
    python .../zbo_analyze.py --run-dir runs --out analysis/zbo_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zbo_common as BC  # noqa: E402
import zbo_models as ZM  # noqa: E402

BOOT = 2000
BOOT_SEED = 20261027


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def bootstrap_ci(values: np.ndarray, n: int = BOOT) -> list[float]:
    # Local deterministic RNG: the CI for a given vector does not depend on
    # how many other groups were evaluated earlier in the script.
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def group_stat(delta: np.ndarray, mask: np.ndarray,
               ae_ref: np.ndarray, ae_other: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    d = delta[mask]
    return {
        "n": n,
        "delta_mean": float(d.mean()),
        "delta_ci95": bootstrap_ci(d),
        "delta_median": float(np.median(d)),
        "win_rate": float((d > 0).mean()),
        "ae_ref_mean": float(ae_ref[mask].mean()),
        "ae_other_mean": float(ae_other[mask].mean()),
    }


def by_bins(values: np.ndarray, edges: Sequence[float], prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    out = {}
    for lo, hi, label in edges:
        mask = (values >= lo) & (values <= hi)
        out[f"{prefix}_{label}"] = mask
    return out


def nblock_mask(n: np.ndarray, bins: Sequence[int]) -> dict:
    n = np.asarray(n, dtype=np.int64)
    a, b, c = [int(x) for x in bins]
    return {
        f"n_blocks_le_{a}": n <= a,
        f"n_blocks_{a + 1}_{b}": (n > a) & (n <= b),
        f"n_blocks_{b + 1}_{c}": (n > b) & (n <= c),
        f"n_blocks_ge_{c + 1}": n > c,
    }


def n_multi_mask(n: np.ndarray) -> dict:
    n = np.asarray(n, dtype=np.int64)
    return {
        "n_multi_0": n == 0,
        "n_multi_1": n == 1,
        "n_multi_2": n == 2,
        "n_multi_3plus": n >= 3,
    }


def load_valid_context(cfg: Mapping[str, Any]):
    vcache = BC.load_cache("valid")
    y = np.asarray(vcache["labels"], dtype=np.float64)
    vblocks = BC.load_blocks("valid")
    train_blocks = BC.load_blocks("train")
    trf = BC.block_row_features(train_blocks)
    vaf = BC.block_row_features(vblocks)
    train_graphs = BC.molecules_from_cache(BC.load_cache("train"))
    valid_graphs = BC.molecules_from_cache(vcache)
    support = BC.radius2_support(train_graphs, valid_graphs)
    tiers = BC.support_tier(support["unseen_frac"], support["low4_frac"])
    rings = np.asarray([BC.paton_max_ring(g) for g in valid_graphs], dtype=np.int64)
    cycle_rank = np.asarray([BC.cycle_rank(g) for g in valid_graphs], dtype=np.int64)
    train_size_vocab = set(trf["block_size_combo"])
    train_sig_vocab = set(trf["block_sig_combo"])
    size_novel = np.asarray([c not in train_size_vocab for c in vaf["block_size_combo"]])
    sig_novel = np.asarray([c not in train_sig_vocab for c in vaf["block_sig_combo"]])
    fixed_rows = [int(r) for r in cfg["fixed_worst10_rows"]]
    fixed_mask = np.zeros(len(y), dtype=bool)
    fixed_mask[fixed_rows] = True
    row172 = np.zeros(len(y), dtype=bool)
    if 0 <= 172 < len(y):
        row172[172] = True
    groups = {
        "ordinary_y_gt_-3": y > -3.0,
        "tail_y_le_-3": y <= -3.0,
        "support_all_supported": tiers == "all_supported",
        "support_low_support": tiers == "low_support",
        "support_any_unseen": tiers == "any_unseen",
        "rings_ge_4": rings >= 4,
        "penalized_proxy_ring_ge_7": rings >= 7,
        "fixed_worst10": fixed_mask,
        "drop_fixed_worst10": ~fixed_mask,
        "drop_row172": ~row172,
        **n_multi_mask(vaf["n_multi_blocks"]),
        **nblock_mask(vaf["n_blocks"],
                      cfg["group_definitions"]["n_blocks_quantiles_train"]),
        "block_size_combo_novel": size_novel,
        "block_size_combo_seen": ~size_novel,
        "block_sig_combo_novel": sig_novel,
        "block_sig_combo_seen": ~sig_novel,
    }
    context = {
        "vcache": vcache,
        "y": y,
        "vblocks": vblocks,
        "fixed_rows": fixed_rows,
        "fixed_mask": fixed_mask,
        "groups": groups,
        "support": support,
        "tiers": tiers,
        "rings": rings,
        "cycle_rank": cycle_rank,
        "size_novel": size_novel,
        "sig_novel": sig_novel,
    }
    return context


def load_run(run_dir: Path, norm: Mapping[str, Any],
             vcache: Mapping[str, Any], vrel: Mapping[str, Any],
             vblocks: Mapping[str, Any], cache: Mapping[str, Any],
             rel: Mapping[str, Any], blocks: Mapping[str, Any]) -> dict | None:
    tj = run_dir / "train.json"
    if not tj.exists():
        return None
    info = BC.read_json(tj)
    arm = str(info["arm"])
    seed = int(info["seed"])
    model = ZM.BlockBOSI(norm, block_arm=arm)
    model.eval()
    preds = {}
    for tag in ("best", "final"):
        ck = run_dir / f"{tag}.pt"
        if not ck.exists():
            raise FileNotFoundError(ck)
        payload = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        pv = BC.predict(model, vcache, vrel, vblocks,
                        np.arange(len(vcache["labels"]), dtype=np.int64),
                        device="cpu")
        pt = BC.predict(model, cache, rel, blocks,
                        np.arange(10_000, dtype=np.int64), device="cpu")
        preds[tag] = {"valid": pv, "train": pt}
    y = np.asarray(vcache["labels"], dtype=np.float64)
    y_train = np.asarray(cache["labels"], dtype=np.float64)
    best = preds["best"]
    final = preds["final"]
    run = {
        "arm": arm,
        "seed": seed,
        "run_dir": BC.rel_or_abs(run_dir),
        "parameters": int(info["parameters"]),
        "best_valid": float(BC.mae(best["valid"], y)),
        "best_valid_logged": float(info["best"]["valid_mae"]),
        "best_epoch": int(info["best"]["epoch"]),
        "best_step": int(info["best"]["step"]),
        "final_valid": float(BC.mae(final["valid"], y)),
        "train_at_best": float(BC.mae(best["train"], y_train)),
        "train_at_final": float(BC.mae(final["train"], y_train)),
        "W20": float(info["W20"]),
        "T100": float(info["T100"]),
        "median_AE_best": float(np.median(np.abs(best["valid"] - y))),
        "p90_AE_best": float(np.percentile(np.abs(best["valid"] - y), 90)),
        "median_AE_final": float(np.median(np.abs(final["valid"] - y))),
        "p90_AE_final": float(np.percentile(np.abs(final["valid"] - y), 90)),
        "mechanism": info.get("mechanism", {}),
        "mechanism_active": bool(info.get("mechanism_active", False)),
        "gpu_hours": float(info.get("gpu_hours", 0.0)),
        "wall_seconds": float(info.get("wall_seconds", 0.0)),
        "initial_state_sha256": info.get("initial", {}).get("initial_state_sha256"),
        "checkpoint_sha256": info.get("checkpoint_sha256", {}),
        "git_head": info.get("git_head"),
        "git_dirty": info.get("git_dirty"),
        "config_sha256": info.get("config_sha256"),
        "first_batch_rows_sha256": info.get("first_batch_rows_sha256"),
        "epochs_run": int(info.get("epochs_run", 0)),
        "total_updates": int(info.get("total_updates", 0)),
        "smoke": bool(info.get("smoke", False)),
        "valid_pred_sha256": {
            "best": info.get("reads", {}).get("best", {}).get("valid_pred_sha256"),
            "final": info.get("reads", {}).get("final", {}).get("valid_pred_sha256"),
        },
        "_preds": preds,
    }
    for tag in ("best", "final"):
        saved = np.load(run_dir / f"valid_{tag}.npz")
        saved_pred = np.asarray(saved["pred"], dtype=np.float64)
        run[f"valid_pred_max_abs_diff_vs_saved_{tag}"] = float(
            np.max(np.abs(saved_pred - np.asarray(preds[tag]["valid"], dtype=np.float64))))
    # Recompute-vs-logged prediction hash check (float32 exact).
    for tag in ("best", "final"):
        arr = np.asarray(preds[tag]["valid"], dtype=np.float32)
        import hashlib
        h = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        run[f"valid_pred_sha256_recomputed_{tag}"] = h
        run[f"valid_pred_sha256_match_{tag}"] = (
            h == run["valid_pred_sha256"][tag])
    return run


def compare(ref: Mapping[str, Any], other: Mapping[str, Any],
            context: Mapping[str, Any], tag: str) -> dict:
    y = context["y"]
    ae_ref = np.abs(ref["_preds"]["best"]["valid"] - y)
    ae_other = np.abs(other["_preds"]["best"]["valid"] - y)
    delta = ae_other - ae_ref  # positive => reference arm better
    groups = {k: group_stat(delta, m, ae_ref, ae_other)
              for k, m in context["groups"].items()}
    order = np.argsort(-ae_ref, kind="stable")[:10]
    drop_top = np.ones(len(y), dtype=bool)
    drop_top[order] = False
    groups["drop_top10_ae_ref"] = group_stat(delta, drop_top, ae_ref, ae_other)
    total = float(delta.sum())
    max_row = float(delta.max())
    ae_ref_f = np.abs(ref["_preds"]["final"]["valid"] - y)
    ae_other_f = np.abs(other["_preds"]["final"]["valid"] - y)
    delta_f = ae_other_f - ae_ref_f
    final_groups = {
        k: group_stat(delta_f, m, ae_ref_f, ae_other_f)
        for k, m in context["groups"].items()
    }
    out = {
        "seed": int(ref["seed"]),
        "ref_arm": ref["arm"],
        "other_arm": other["arm"],
        "n": int(len(y)),
        "ref_best_valid": float(ref["best_valid"]),
        "other_best_valid": float(other["best_valid"]),
        "d_best": float(other["best_valid"] - ref["best_valid"]),
        "d_W20": float(other["W20"] - ref["W20"]),
        "d_T100": float(other["T100"] - ref["T100"]),
        "delta_mean": float(delta.mean()),
        "delta_ci95": bootstrap_ci(delta),
        "delta_median": float(np.median(delta)),
        "win_rate": float((delta > 0).mean()),
        "ref_final_valid": float(ref["final_valid"]),
        "other_final_valid": float(other["final_valid"]),
        "d_final": float(other["final_valid"] - ref["final_valid"]),
        "delta_final_mean": float(delta_f.mean()),
        "delta_final_ci95": bootstrap_ci(delta_f),
        "win_rate_final": float((delta_f > 0).mean()),
        "final_groups": final_groups,
        "single_row_max_contribution_fraction_of_net": (
            max_row / total if total > 0 else None),
        "single_row_max_delta": max_row,
        "groups": groups,
    }
    return out


def decision_summary(runs: Mapping[tuple[str, int], Mapping[str, Any]],
                     comparisons: Mapping[str, Mapping[str, Any]],
                     cfg: Mapping[str, Any]) -> dict:
    out = {"seed0": {}, "two_seed": {}, "label": None, "reasons": []}
    r0 = runs.get(("R", 0))
    n0 = runs.get(("N", 0))
    if r0 and n0:
        c = comparisons["R_vs_N"]["0"]
        mech = bool(r0["mechanism_active"])
        gate = cfg["seed0_gate"]
        ordinary = c["groups"]["ordinary_y_gt_-3"]
        drop_fixed = c["groups"]["drop_fixed_worst10"]
        checks = {
            "d_best_ge_0.005": c["d_best"] >= float(gate["d_best_min"]),
            "R_best_le_0.082": c["ref_best_valid"] <= float(gate["R_best_max"]),
            "d_W20_ge_-0.002": c["d_W20"] >= float(gate["d_W20_min"]),
            "d_T100_ge_-0.002": c["d_T100"] >= float(gate["d_T100_min"]),
            "W20_or_T100_positive": (c["d_W20"] > 0) or (c["d_T100"] > 0),
            "ordinary_delta_ge_0": ordinary.get("delta_mean", -1) >= 0.0,
            "drop_fixed_worst10_delta_gt_0": drop_fixed.get("delta_mean", -1) > 0.0,
            "single_row_fraction_lt_0.5": (
                c["single_row_max_contribution_fraction_of_net"] is not None
                and c["single_row_max_contribution_fraction_of_net"] < 0.5),
            "mechanism_active": mech,
        }
        out["seed0"] = {"pass": bool(all(checks.values())), "checks": checks,
                        "comparison": c, "mechanism": r0["mechanism"]}
        if not mech:
            out["label"] = "MECHANISM_INACTIVE"
            out["reasons"].append("R seed-0 block channel inactive")
        elif not out["seed0"]["pass"]:
            failed = [k for k, v in checks.items() if not v]
            out["gate_failure"] = failed
            out["status"] = "stopped_after_seed0_gate"
            # Labels 1-4 are defined with two seeds.  With one seed we may only
            # claim OWNERSHIP_NOT_USEFUL when the evidence is clearly negative;
            # a positive-but-below-floor point estimate with a CI spanning zero
            # is the pre-registered UNRESOLVED_DUE_TO_VARIANCE case.
            ci = c["delta_ci95"]
            clearly_negative = (c["d_best"] <= 0.0 and ci[1] < 0.0)
            out["label"] = ("OWNERSHIP_NOT_USEFUL" if clearly_negative
                            else "UNRESOLVED_DUE_TO_VARIANCE")
            out["reasons"].append(
                "seed-0 R-N gate failed: " + ",".join(failed))
    else:
        out["seed0"] = {"pass": False, "checks": {}, "reason": "missing R_s0/N_s0"}
        out["reasons"].append("missing R_s0/N_s0")

    r1 = runs.get(("R", 1))
    n1 = runs.get(("N", 1))
    if r1 and n1:
        c1 = comparisons["R_vs_N"]["1"]
        both = (out["seed0"].get("pass", False)
                and c1["d_best"] >= float(cfg["two_seed_gate"]["both_d_best_min"]))
        same = (np.sign(c1["d_best"]) == np.sign(
            comparisons["R_vs_N"]["0"]["d_best"]))
        out["two_seed"] = {
            "both_seed_d_best_ge_0.005": bool(both),
            "same_direction": bool(same),
            "seed1_d_best": c1["d_best"],
            "seed1_R_best": c1["ref_best_valid"],
            "pass": bool(both and same),
        }
        if out["label"] not in ("MECHANISM_INACTIVE", "OWNERSHIP_NOT_USEFUL"):
            if not out["two_seed"]["pass"]:
                out["label"] = "UNRESOLVED_DUE_TO_VARIANCE"
                out["reasons"].append("two-seed R-N gate failed or direction conflict")
    else:
        out["two_seed"] = {"pass": False, "reason": "missing R_s1/N_s1"}

    # S comparisons, only if present.
    for seed in (0, 1):
        rs = runs.get(("S", seed))
        rr = runs.get(("R", seed))
        if rs and rr:
            c = comparisons["R_vs_S"][str(seed)]
            out.setdefault("R_vs_S", {})[str(seed)] = c
    if out["label"] not in ("MECHANISM_INACTIVE", "OWNERSHIP_NOT_USEFUL",
                            "UNRESOLVED_DUE_TO_VARIANCE"):
        s_means = [v["delta_mean"] for v in out.get("R_vs_S", {}).values()]
        r_best = [runs[("R", s)]["best_valid"] for s in (0, 1)
                  if ("R", s) in runs]
        if len(s_means) == 2:
            s_mean = float(np.mean(s_means))
            r_mean = float(np.mean(r_best)) if len(r_best) == 2 else None
            out["R_S_two_seed_mean"] = s_mean
            out["R_seed_mean_best"] = r_mean
            if s_mean < 0.003:
                out["label"] = "SINGLETON_NONLINEARITY_ONLY"
                out["reasons"].append("R-S mean below 0.003")
            elif r_mean is not None and r_mean > 0.077:
                out["label"] = "STRUCTURAL_GAIN_BUT_NOT_COMPETITIVE"
                out["reasons"].append("R seed-mean valid > 0.077")
            elif r_mean is not None:
                out["label"] = "PROMISING_COMPOSITION_MECHANISM"
                out["reasons"].append("R-S mean >= 0.003 and R seed-mean <= 0.077")
        else:
            out["reasons"].append("S runs missing or incomplete")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default=str(BC.RUNS))
    p.add_argument("--out", default=str(BC.ANALYSIS / "zbo_analysis.json"))
    p.add_argument("--utc", default=None,
                   help="fixed analysis timestamp for a reproducible artifact")
    args = p.parse_args(argv)
    cfg = BC.read_json(BC.CONFIG)
    norm = BC.load_norm()
    cache = BC.load_cache("train")
    rel = BC.load_rel("train")
    blocks = BC.load_blocks("train")
    vcache = BC.load_cache("valid")
    vrel = BC.load_rel("valid")
    vblocks = BC.load_blocks("valid")
    context = load_valid_context(cfg)

    run_root = Path(args.run_dir)
    runs: dict[tuple[str, int], dict] = {}
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        try:
            run = load_run(run_dir, norm, vcache, vrel, vblocks,
                           cache, rel, blocks)
        except Exception as exc:
            BC.log(f"[analysis] skip {run_dir.name}: {exc!r}")
            continue
        if run is None:
            continue
        if run["smoke"]:
            continue
        key = (run["arm"], run["seed"])
        if key in runs:
            raise RuntimeError(f"duplicate run for {key}: {run_dir}")
        runs[key] = run

    comparisons = {"R_vs_N": {}, "R_vs_S": {}}
    for seed in (0, 1):
        if ("R", seed) in runs and ("N", seed) in runs:
            comparisons["R_vs_N"][str(seed)] = compare(
                runs[("R", seed)], runs[("N", seed)], context, "R_vs_N")
        if ("R", seed) in runs and ("S", seed) in runs:
            comparisons["R_vs_S"][str(seed)] = compare(
                runs[("R", seed)], runs[("S", seed)], context, "R_vs_S")

    # Compact run table without the in-memory prediction arrays.
    run_table = {}
    for (arm, seed), run in sorted(runs.items()):
        run_table[f"{arm}_s{seed}"] = {
            k: v for k, v in run.items() if not k.startswith("_")
        }
    payload = {
        "round": BC.ROUND_ID,
        "analysis_utc": args.utc or BC.now(),
        "config_sha256": BC.sha256_file(BC.CONFIG),
        "official_test_loaded": False,
        "n_valid": int(len(context["y"])),
        "n_train": 10000,
        "fixed_worst10_rows": context["fixed_rows"],
        "runs": run_table,
        "comparisons": comparisons,
        "support_audit": {
            "train_vocab_size": int(context["support"]["train_vocab_size"]),
            "fraction_valid_any_unseen": float((context["tiers"] == "any_unseen").mean()),
            "fraction_valid_low_support": float((context["tiers"] == "low_support").mean()),
            "fraction_valid_all_supported": float((context["tiers"] == "all_supported").mean()),
            "fraction_valid_rings_ge4": float((context["rings"] >= 4).mean()),
            "fraction_valid_penalized_proxy": float((context["rings"] >= 7).mean()),
            "fraction_valid_size_combo_novel": float(context["size_novel"].mean()),
            "fraction_valid_sig_combo_novel": float(context["sig_novel"].mean()),
        },
        "decision": decision_summary(runs, comparisons, cfg),
        "claim_limits": cfg["claim_limits"],
        "forbidden": cfg["forbidden"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    BC.write_json(out, jsonable(payload))
    BC.log(f"[analysis] wrote {BC.rel_or_abs(out)} runs={len(runs)} "
           f"comparisons={list(comparisons)} label={payload['decision']['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
