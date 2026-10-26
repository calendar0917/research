"""Analysis for ``zinc_bosi_joint_moment_v1`` — official-valid only, test untouched.

Reads ``analysis/freeze.json`` (written by ``bjm_freeze.py``) and the frozen
per-run ``valid_{best,final}.npz`` arrays, and writes

* ``analysis/summary.json``  — main table, paired contrasts, groups, decision
* ``analysis/tables.md``     — the same, human readable

Nothing here retrains, reselects an epoch, or opens the official test cache.
Every label used is an official-valid label, and the valid split has been read
many times in this project: all intervals below describe molecule sampling
uncertainty only, never seed-level reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bjm_common as BC  # noqa: E402

N_BOOT = 10_000
SEED0_GATE = {
    "A": "JM_s0.best_valid <= 0.075",
    "B": "(JM_s0.best_valid - B_s0.best_valid) <= -0.003 and JM_s0.best_valid <= 0.078",
    "guards": ["ordinary group (y > -3) shows no clear regression",
               "W20 shows no large regression in the opposite direction"],
}
HISTORICAL = {
    "dccl_valid_4seed_mean": 0.075990,
    "dccl_valid_seed0": 0.078537,
    "p3b_valid_seed0": 0.076224,
    "note": ("same official protocol, historical values; not retrained and not a "
             "matched control of this round"),
}


def load_pred(run_dir: Path, ckpt: str) -> np.ndarray:
    blob = np.load(run_dir / f"valid_{ckpt}.npz")
    return np.asarray(blob["pred"], dtype=np.float64)


def boot_ci(delta: np.ndarray, rng) -> tuple[float, float]:
    n = len(delta)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = delta[idx].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def groups_for(valid_cache, train_cache) -> dict:
    """Frozen, label-free-where-possible grouping of the official-valid rows."""
    y = np.asarray(valid_cache["labels"], dtype=np.float64)
    vgraphs = BC.molecules_from_cache(valid_cache)
    tgraphs = BC.molecules_from_cache(train_cache)
    support = BC.radius2_support(tgraphs, vgraphs)
    tier = BC.support_tier(support["unseen_frac"], support["low4_frac"])
    rings = np.asarray([BC.cycle_rank(g) >= 4 for g in vgraphs])
    return {
        "masks": {
            "overall": np.ones(len(y), dtype=bool),
            "ordinary_y_gt_-3": y > -3.0,
            "tail_y_le_-3": y <= -3.0,
            "radius2_all_supported": tier == "all_supported",
            "radius2_low_support": tier == "low_support",
            "radius2_any_unseen": tier == "any_unseen",
            "rings_ge4": rings,
        },
        "support": {
            "train_vocab_size": support["train_vocab_size"],
            "unseen_frac_mean": float(support["unseen_frac"].mean()),
            "mean_unseen_atom_fraction": float(support["unseen_frac"].mean()),
        },
        "y": y,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    freeze_path = Path(args.freeze) if args.freeze else BC.ANALYSIS / "freeze.json"
    if not freeze_path.exists():
        raise SystemExit(f"missing {freeze_path}; run bjm_freeze.py first")
    freeze = BC.read_json(freeze_path)
    if BC.sha256_file(BC.CONFIG) != freeze["config_sha256"]:
        raise SystemExit("config changed after the freeze; refusing to analyse")
    if BC.sha256_file(BC.ANALYSIS / "cpu_checks.json") != freeze["cpu_gates_sha256"]:
        raise SystemExit("cpu_checks.json changed after the freeze; refusing to analyse")

    vcache = BC.load_cache("valid")
    tcache = BC.load_cache("train")
    grp = groups_for(vcache, tcache)
    y = grp["y"]
    masks = grp["masks"]

    runs = freeze["runs"]
    table = {}
    aerr = {}
    for tag, info in runs.items():
        d = BC.resolve_run(tag)
        for ckpt in ("best", "final"):
            pred = load_pred(d, ckpt)
            ae = np.abs(pred - y)
            aerr[(tag, ckpt)] = ae
            entry = table.setdefault(tag, {"arm": info["arm"], "seed": info["seed"],
                                          "parameters": info["parameters"],
                                          "gpu_hours": info["gpu_hours"],
                                          "best_epoch": info["best"]["epoch"],
                                          "W20": info["W20"], "T100": info["T100"],
                                          "best_valid": info["best"]["valid_mae"]})
            entry[ckpt] = {
                "valid_mae": float(ae.mean()),
                "groups": {g: float(ae[m].mean()) for g, m in masks.items() if m.any()},
                "group_n": {g: int(m.sum()) for g, m in masks.items() if m.any()},
                "train_mae_eval_mode": info["checkpoints"][ckpt]["train_mae_eval_mode"],
                "median_ae": float(np.median(ae)),
                "top10_share_of_abs_error": float(np.sort(ae)[-10:].sum() / ae.sum()),
                "prediction_sha256": info["checkpoints"][ckpt]["prediction_sha256"],
                "checkpoint_sha256": info["checkpoints"][ckpt]["checkpoint_sha256"],
            }

    # ---- paired contrasts ---------------------------------------------------
    rng = np.random.default_rng(20261028)
    contrasts = {}

    def add_contrast(name: str, a: tuple, b: tuple, ckpt: str):
        ae_a, ae_b = aerr[(a, ckpt)], aerr[(b, ckpt)]
        delta = ae_a - ae_b
        lo, hi = boot_ci(delta, rng)
        order = np.argsort(-np.maximum(ae_a, ae_b))[:10]
        keep = np.ones(len(delta), dtype=bool)
        keep[order] = False
        contrasts[f"{name}|{ckpt}"] = {
            "first": a, "second": b, "checkpoint": ckpt,
            "delta_mae": float(delta.mean()),
            "ci95": [lo, hi],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "first_better_fraction": float(np.mean(delta < 0)),
            "delta_mae_drop_top10": float(delta[keep].mean()),
            "delta_by_group": {g: float(delta[m].mean()) for g, m in masks.items()
                               if m.any()},
        }

    def seeds_of(arm: str) -> list[int]:
        return sorted(i["seed"] for i in runs.values() if i["arm"] == arm)

    by_arm = {arm: seeds_of(arm) for arm in ("B", "JM", "CM")}
    for seed in by_arm["JM"]:
        if seed in by_arm["B"]:
            add_contrast("JM_minus_B", f"JM_s{seed}", f"B_s{seed}", "best")
            add_contrast("JM_minus_B", f"JM_s{seed}", f"B_s{seed}", "final")
        if seed in by_arm["CM"]:
            add_contrast("JM_minus_CM", f"JM_s{seed}", f"CM_s{seed}", "best")
            add_contrast("JM_minus_CM", f"JM_s{seed}", f"CM_s{seed}", "final")

    # ---- seed aggregation ---------------------------------------------------
    seed_means = {}
    for arm, seeds in by_arm.items():
        if not seeds:
            continue
        seed_means[arm] = {
            "seeds": seeds,
            "best_valid_mean": float(np.mean([table[f"{arm}_s{s}"]["best_valid"]
                                              for s in seeds])),
            "W20_mean": float(np.mean([table[f"{arm}_s{s}"]["W20"] for s in seeds])),
            "T100_mean": float(np.mean([table[f"{arm}_s{s}"]["T100"] for s in seeds])),
            "final_valid_mean": float(np.mean([table[f"{arm}_s{s}"]["final"]["valid_mae"]
                                               for s in seeds])),
        }

    # ---- seed0 gate ---------------------------------------------------------
    decision = {"rule": SEED0_GATE, "stage": "seed0"}
    if "B_s0" in table and "JM_s0" in table:
        jm0 = table["JM_s0"]["best"]["valid_mae"]
        b0 = table["B_s0"]["best"]["valid_mae"]
        d0 = jm0 - b0
        cond_a = bool(jm0 <= 0.075)
        cond_b = bool(d0 <= -0.003 and jm0 <= 0.078)
        ordinary_delta = contrasts["JM_minus_B|best"]["delta_by_group"]["ordinary_y_gt_-3"]
        w20_delta = table["JM_s0"]["W20"] - table["B_s0"]["W20"]
        guards_ok = bool(ordinary_delta <= 0.002 and w20_delta <= 0.002)
        decision.update({
            "JM_s0_best_valid": jm0, "B_s0_best_valid": b0, "delta": d0,
            "condition_A": cond_a, "condition_B": cond_b,
            "ordinary_delta": ordinary_delta, "W20_delta": w20_delta,
            "guards_ok": guards_ok,
            "buy_seed1": bool((cond_a or cond_b) and guards_ok),
        })
        if "B_s1" in table and "JM_s1" in table:
            d1 = table["JM_s1"]["best"]["valid_mae"] - table["B_s1"]["best"]["valid_mae"]
            decision.update({
                "stage": "seed1",
                "JM_s1_best_valid": table["JM_s1"]["best"]["valid_mae"],
                "B_s1_best_valid": table["B_s1"]["best"]["valid_mae"],
                "delta_s1": d1,
                "seed_mean_delta": (d0 + d1) / 2.0,
                "both_seeds_same_direction": bool(np.sign(d0) == np.sign(d1)),
                "run_mechanism_control": bool(
                    np.sign(d0) == np.sign(d1) and (d0 + d1) / 2.0 <= -0.003),
            })
    if "CM_s0" in table and "JM_s0" in table:
        cm_delta = contrasts["JM_minus_CM|best"]["delta_mae"]
        decision["jm_minus_cm_s0"] = cm_delta
        decision["jm_minus_cm_favours_pairing"] = bool(cm_delta < 0)

    # ---- per-molecule AE table ---------------------------------------------
    csv_path = BC.ANALYSIS / "per_molecule_valid.csv"
    tags = list(table)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "y"] + [f"AE_{t}_best" for t in tags]
                   + [f"AE_{t}_final" for t in tags])
        for i in range(len(y)):
            w.writerow([i, f"{y[i]:.6f}"]
                       + [f"{aerr[(t, 'best')][i]:.6f}" for t in tags]
                       + [f"{aerr[(t, 'final')][i]:.6f}" for t in tags])

    # ---- curves / gaps ------------------------------------------------------
    gaps = {}
    for tag, info in runs.items():
        d = BC.resolve_run(tag)
        curve = json.loads((d / "train.json").read_text())["curve"]
        v = np.asarray([e["valid_mae"] for e in curve])
        tr = np.asarray([e["online_train_mae"] for e in curve])
        starts = [0, max(0, len(v) // 4), max(0, len(v) // 2), max(0, 3 * len(v) // 4)]
        gaps[tag] = {
            "valid_first": float(v[0]), "valid_last": float(v[-1]),
            "valid_min": float(v.min()), "valid_argmin_epoch": int(v.argmin() + 1),
            "eval_train_at_best": info["checkpoints"]["best"]["train_mae_eval_mode"],
            "eval_train_at_final": info["checkpoints"]["final"]["train_mae_eval_mode"],
            "train_valid_gap_at_best": float(
                info["checkpoints"]["best"]["train_mae_eval_mode"]
                - info["best"]["valid_mae"]),
            "train_valid_gap_at_final": float(
                info["checkpoints"]["final"]["train_mae_eval_mode"]
                - table[tag]["final"]["valid_mae"]),
            "valid_quartile_medians": [float(np.median(v[s:])) for s in starts],
            "online_train_quartile_medians": [
                float(np.median(tr[s:s + len(v) // 4])) if len(v) >= 4 else float(np.median(tr))
                for s in starts],
            "epochs": int(len(v)),
        }

    payload = {
        "round": BC.ROUND_ID,
        "written": BC.now(),
        "freeze": BC.rel_or_abs(freeze_path),
        "official_valid_loaded": True,
        "official_test_loaded": False,
        "n_valid": int(len(y)),
        "metric": "raw-y MAE",
        "main_table": table,
        "seed_means": seed_means,
        "contrasts": contrasts,
        "groups": {"masks": {g: int(m.sum()) for g, m in masks.items()},
                   "support": grp["support"],
                   "definitions": {
                       "ordinary_y_gt_-3": "y > -3",
                       "tail_y_le_-3": "y <= -3",
                       "radius2_*": "exact typed radius-2 rooted signatures counted on the official train 10k; unseen = contains an atom environment with 0 train molecules, low_support = no unseen but some 1-4, all_supported = every atom >= 5",
                       "rings_ge4": "cycle rank E - V + 1 >= 4"}},
        "gaps_and_curves": gaps,
        "decision": decision,
        "historical_references": HISTORICAL,
        "gpu_hours_total": float(sum(r["gpu_hours"] for r in runs.values())),
        "bootstrap": {"n": N_BOOT, "kind": "paired over fixed valid molecules, descriptive only"},
    }
    out = Path(args.out) if args.out else BC.ANALYSIS / "summary.json"
    BC.write_json(out, payload)
    BC.write_json(BC.ANALYSIS / "decision.json", decision)
    BC.write_json(BC.ANALYSIS / "groups.json",
                  {"masks": payload["groups"]["masks"],
                   "support": grp["support"]})
    _tables(payload)
    BC.log(f"[bjm-analyze] wrote {BC.rel_or_abs(out)}")
    return 0


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.6f}"


def _tables(payload: dict) -> None:
    lines = ["# BOSI-JM v1 — frozen official-valid tables", "",
             f"written {payload['written']} · n_valid = {payload['n_valid']} · "
             "official test never loaded", "",
             "## Main table (raw-y MAE)", "",
             "| run | arm | seed | params | GPU-h | best_valid | best_ep | W20 | T100 "
             "| final_valid | train@best | train@final |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, r in payload["main_table"].items():
        lines.append(
            f"| {tag} | {r['arm']} | {r['seed']} | {r['parameters']} | "
            f"{r['gpu_hours']:.3f} | {_fmt(r['best_valid'])} | {r['best_epoch']} | "
            f"{_fmt(r['W20'])} | {_fmt(r['T100'])} | {_fmt(r['final']['valid_mae'])} | "
            f"{_fmt(r['best']['train_mae_eval_mode'])} | "
            f"{_fmt(r['final']['train_mae_eval_mode'])} |")
    lines += ["", "## Paired contrasts (negative = first arm better)", "",
              "| contrast | ckpt | ΔMAE | 95% CI | CI excludes 0 | first better frac "
              "| Δ drop-top10 | ordinary | tail | unseen | rings≥4 |",
              "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, c in payload["contrasts"].items():
        g = c["delta_by_group"]
        lines.append(
            f"| {c['first']} − {c['second']} | {c['checkpoint']} | "
            f"{c['delta_mae']:+.6f} | [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | "
            f"{c['ci_excludes_zero']} | {c['first_better_fraction']:.3f} | "
            f"{c['delta_mae_drop_top10']:+.6f} | "
            f"{g.get('ordinary_y_gt_-3', float('nan')):+.6f} | "
            f"{g.get('tail_y_le_-3', float('nan')):+.6f} | "
            f"{g.get('radius2_any_unseen', float('nan')):+.6f} | "
            f"{g.get('rings_ge4', float('nan')):+.6f} |")
    lines += ["", "## Group sizes", ""]
    for g, n in payload["groups"]["masks"].items():
        lines.append(f"* `{g}`: {n}")
    lines += ["", "## Decision (pre-registered)", "", "```",
              json.dumps(payload["decision"], indent=2, default=str), "```", ""]
    (BC.ANALYSIS / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
