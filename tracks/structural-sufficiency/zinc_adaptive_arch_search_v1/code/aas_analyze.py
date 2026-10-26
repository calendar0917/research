"""Analysis for ``zinc_adaptive_arch_search_v1`` — official-valid only, test untouched.

Reads ``analysis/freeze_<stage>.json`` and the frozen per-run
``valid_{best,final}.npz`` arrays (plus the frozen BOSI-B seed-0 reference),
and writes ``analysis/summary_<stage>.json``, ``analysis/tables_<stage>.md``,
``analysis/groups_<stage>.md`` and ``analysis/per_molecule_<stage>.csv``.

Pre-registered decision rules are evaluated verbatim (no post-hoc thresholds):

  D1  A1 − B_s0 <= -0.003  and  A1 − A0 <= -0.001        -> DEEPEN_A
  D2  A1 − B_s0 <= -0.003  and |A1 − A0| <  0.001        -> NONLINEAR_ONLY
  D3  A1 − B_s0 >= +0.005, or W20 and T100 both >= +0.003 -> STOP_A
  otherwise                                              -> AMBIGUOUS

Guards: ordinary group (y > -3) must not regress by more than 0.002.
All intervals describe 1,000-molecule sampling uncertainty, never seed noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aas_common as AC  # noqa: E402

N_BOOT = 10_000
HISTORICAL = {
    "dccl_valid_seed0": 0.078537,
    "dccl_valid_4seed_mean": 0.075990,
    "p3b_valid_seed0": 0.076224,
    "note": "same official protocol, historical values; not retrained this round",
}


def boot_ci(delta: np.ndarray, rng) -> tuple[float, float]:
    n = len(delta)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = delta[idx].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def groups_for(valid_cache, train_cache) -> dict:
    y = np.asarray(valid_cache["labels"], dtype=np.float64)
    vgraphs = AC.molecules_from_cache(valid_cache)
    tgraphs = AC.molecules_from_cache(train_cache)
    support = AC.radius2_support(tgraphs, vgraphs)
    tier = AC.BC.support_tier(support["unseen_frac"], support["low4_frac"])
    rings = np.asarray([AC.BC.cycle_rank(g) >= 4 for g in vgraphs])
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
        "support": {"train_vocab_size": support["train_vocab_size"],
                    "unseen_frac_mean": float(support["unseen_frac"].mean())},
        "y": y,
    }


def load_pred(tag: str, ckpt: str, runs: dict, reference: dict) -> np.ndarray:
    if tag in runs:
        blob = np.load(AC.RUNS / tag / f"valid_{ckpt}.npz")
        return np.asarray(blob["pred"], dtype=np.float64)
    if tag == "B_s0":
        p = REFERENCE_ROOT / "B_s0" / f"valid_{ckpt}.npz"
        blob = np.load(p)
        return np.asarray(blob["pred"], dtype=np.float64)
    raise KeyError(tag)


REFERENCE_ROOT = AC.SUFF / "zinc_bosi_joint_moment_v1" / "runs"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="a")
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--pairs", default=None)
    args = p.parse_args(argv)

    freeze_path = AC.ANALYSIS / f"freeze_{args.stage}.json"
    if not freeze_path.exists():
        raise SystemExit(f"missing {freeze_path}; run aas_freeze.py first")
    freeze = AC.read_json(freeze_path)
    cfg = Path(args.config) if args.config else AC.CONFIG
    if AC.sha256_file(cfg) != freeze["config_sha256"]:
        raise SystemExit("config changed after the freeze; refusing to analyse")
    for tag, rec in freeze["runs"].items():
        if AC.sha256_file(AC.RUNS / tag / "train.json") != rec["train_json_sha256"]:
            raise SystemExit(f"{tag}: train.json changed after the freeze")

    vcache = AC.load_cache("valid")
    tcache = AC.load_cache("train")
    grp = groups_for(vcache, tcache)
    y = grp["y"]
    masks = dict(grp["masks"])
    rows = np.arange(len(y))

    runs = freeze["runs"]
    ref = freeze["reference"]
    tags = list(runs) + ["B_s0"]
    table, aerr = {}, {}
    for tag in tags:
        for ckpt in ("best", "final"):
            pred = load_pred(tag, ckpt, runs, ref)
            ae = np.abs(pred - y)
            aerr[(tag, ckpt)] = ae
            entry = table.setdefault(tag, {})
            if tag in runs:
                r = runs[tag]
                entry.update({"arm": r["arm"], "seed": r["seed"],
                              "parameters": r["parameters"],
                              "gpu_hours": r["gpu_hours"],
                              "best_epoch": r["best"]["epoch"],
                              "W20": r["W20"], "T100": r["T100"],
                              "best_valid": r["best"]["valid_mae"]})
            else:
                entry.update({"arm": "B", "seed": 0,
                              "parameters": ref["parameters"],
                              "gpu_hours": ref["gpu_hours"],
                              "best_epoch": ref["best_epoch"],
                              "W20": ref["W20"], "T100": ref["T100"],
                              "best_valid": ref["best_valid"]})
            entry[ckpt] = {
                "valid_mae": float(ae.mean()),
                "groups": {g: float(ae[m].mean()) for g, m in masks.items() if m.any()},
                "group_n": {g: int(m.sum()) for g, m in masks.items() if m.any()},
                "median_ae": float(np.median(ae)),
                "p90_ae": float(np.quantile(ae, 0.90)),
                "top10_share_of_abs_error": float(np.sort(ae)[-10:].sum() / ae.sum()),
                "train_mae_eval_mode": (
                    runs[tag]["checkpoints"][ckpt]["train_mae_eval_mode"]
                    if tag in runs else
                    (ref["train_at_best"] if ckpt == "best" else ref["train_at_final"])),
            }

    # fixed reference worst-10 and the same-10-complement, defined by B_s0
    ref_ae = aerr[("B_s0", "best")]
    worst10 = np.argsort(-ref_ae)[:10]
    keep990 = np.ones(len(y), dtype=bool)
    keep990[worst10] = False
    masks["reference_worst10"] = np.zeros(len(y), dtype=bool)
    masks["reference_worst10"][worst10] = True
    masks["reference_complement_990"] = keep990
    masks["drop_row172"] = rows != 172
    for tag in tags:
        for ckpt in ("best", "final"):
            ae = aerr[(tag, ckpt)]
            table[tag][ckpt]["groups"].update({
                g: float(ae[m].mean()) for g, m in masks.items() if m.any()})

    # ---- paired contrasts ---------------------------------------------------
    rng = np.random.default_rng(20261028)
    contrasts = {}

    def add(name, a, b, ckpt):
        ae_a, ae_b = aerr[(a, ckpt)], aerr[(b, ckpt)]
        delta = ae_a - ae_b
        lo, hi = boot_ci(delta, rng)
        order = np.argsort(-np.maximum(ae_a, ae_b))[:10]
        keep = np.ones(len(delta), dtype=bool)
        keep[order] = False
        contrasts[f"{name}|{ckpt}"] = {
            "first": a, "second": b, "checkpoint": ckpt,
            "delta_mae": float(delta.mean()), "ci95": [lo, hi],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "first_better_fraction": float(np.mean(delta < 0)),
            "delta_mae_drop_top10": float(delta[keep].mean()),
            "delta_by_group": {g: float(delta[m].mean()) for g, m in masks.items()
                               if m.any()},
            "row172_delta": float(delta[172]),
            "n_rows_improved_gt_0.01": int((delta < -0.01).sum()),
            "n_rows_worse_gt_0.01": int((delta > 0.01).sum()),
            "sum_improvement": float(np.maximum(-delta, 0).sum()),
            "sum_degradation": float(np.maximum(delta, 0).sum()),
            "top_contributor_row": int(np.argmax(np.abs(delta))),
            "top_contributor_delta": float(delta[int(np.argmax(np.abs(delta)))]),
            "delta_drop_top_contributor": float(np.delete(delta, int(np.argmax(np.abs(delta)))).mean()),
            "delta_drop_row172": float(np.delete(delta, 172).mean()),
        }

    defaults = ["A1:A0", "A1:B_s0", "A0:B_s0", "A2:A1", "A3:A1", "A2:B_s0",
                "A3:B_s0", "S1:S0", "S1:B_s0", "S0:B_s0", "S1:A1", "A1:B_s0"]
    pairs = defaults if not args.pairs else [x.strip() for x in args.pairs.split(",") if x.strip()]
    for pair in pairs:
        a_arm, b_arm = pair.split(":")
        for seed in sorted({r["seed"] for r in runs.values()}):
            have = {r["arm"]: tag for tag, r in runs.items() if r["seed"] == seed}
            if a_arm not in have:
                continue
            if b_arm == "B_s0":
                if seed != 0:
                    continue
                btag = "B_s0"
            else:
                btag = have.get(b_arm)
                if btag is None:
                    continue
            bname = "B" if b_arm == "B_s0" else b_arm
            add(f"{a_arm}_minus_{bname}", have[a_arm], btag, "best")
            add(f"{a_arm}_minus_{bname}", have[a_arm], btag, "final")

    # ---- seed aggregation ---------------------------------------------------
    seed_means = {}
    for arm in ("A1", "A0"):
        its = [r for r in runs.values() if r["arm"] == arm]
        if not its:
            continue
        seed_means[arm] = {
            "seeds": sorted(r["seed"] for r in its),
            "best_valid_mean": float(np.mean([r["best"]["valid_mae"] for r in its])),
            "W20_mean": float(np.mean([r["W20"] for r in its])),
            "T100_mean": float(np.mean([r["T100"] for r in its])),
            "final_valid_mean": float(np.mean([r["final_valid"] for r in its])),
        }

    # ---- pre-registered decision -------------------------------------------
    decision = {"rules": {
        "DEEPEN_A": "A1 - B_s0 <= -0.003 and A1 - A0 <= -0.001",
        "NONLINEAR_ONLY": "A1 - B_s0 <= -0.003 and |A1 - A0| < 0.001",
        "STOP_A": "A1 - B_s0 >= +0.005, or (W20 and T100 both >= B_s0 + 0.003)",
        "AMBIGUOUS": "otherwise: at most one confirmation purchase",
        "guard": "ordinary group (y > -3) must not regress by more than 0.002"}}
    if "A1_s0" in table:
        a1b = contrasts["A1_minus_B|best"]["delta_mae"]
        w20d = table["A1_s0"]["W20"] - table["B_s0"]["W20"]
        t100d = table["A1_s0"]["T100"] - table["B_s0"]["T100"]
        ordinary = contrasts["A1_minus_B|best"]["delta_by_group"]["ordinary_y_gt_-3"]
        a1a0 = contrasts.get("A1_minus_A0|best", {}).get("delta_mae")
        decision.update({
            "A1_s0_best_valid": table["A1_s0"]["best"]["valid_mae"],
            "A1_s0_best_epoch": table["A1_s0"]["best_epoch"],
            "B_s0_best_valid": table["B_s0"]["best_valid"],
            "delta_A1_minus_B_best": a1b,
            "delta_A1_minus_B_final": contrasts["A1_minus_B|final"]["delta_mae"],
            "delta_A1_minus_A0_best": a1a0,
            "W20_delta": w20d, "T100_delta": t100d,
            "ordinary_delta": ordinary,
        })
        deepen = (a1b <= -0.003 and a1a0 is not None and a1a0 <= -0.001)
        nonlinear = (a1b <= -0.003 and a1a0 is not None and abs(a1a0) < 0.001)
        stop = (a1b >= 0.005 or (w20d >= 0.003 and t100d >= 0.003))
        decision["verdict"] = (
            "DEEPEN_A" if deepen else
            "NONLINEAR_ONLY" if nonlinear else
            "STOP_A" if stop else "AMBIGUOUS")
        decision["guard_ok"] = bool(ordinary <= 0.002)
        if "A0_s0" in table:
            decision["A0_s0_best_valid"] = table["A0_s0"]["best"]["valid_mae"]
            decision["delta_A0_minus_B_best"] = contrasts["A0_minus_B|best"]["delta_mae"]

    # ---- seed-aware reading of the same contrasts ---------------------------
    # The pre-registered block above pairs the LAST trained A1 run against the
    # frozen B_s0 reference, so once a second seed exists that verdict is a
    # seed-mismatched comparison.  This block is what the decisions cite: it
    # only uses same-seed pairs, and it states whether the effect survives the
    # body's own seed-to-seed spread.
    by_seed = {}
    for seed in sorted({r["seed"] for r in runs.values()}):
        have = {r["arm"]: tag for tag, r in runs.items() if r["seed"] == seed}
        for name, (x, y_) in {"A1_minus_A0": ("A1", "A0"),
                              "A1_minus_B": ("A1", "B"),
                              "A0_minus_B": ("A0", "B"),
                              "A0M0_minus_A0": ("A0M0", "A0")}.items():
            if x not in have:
                continue
            if y_ == "B" and seed == 0 and "B" not in have:
                btag = "B_s0"
            else:
                btag = have.get(y_)
            if btag is None:
                continue
            d = aerr[(have[x], "best")] - aerr[(btag, "best")]
            base = contrasts.get(f"{x}_minus_{'B' if y_ == 'B' else y_}|best")
            ci = boot_ci(d, np.random.default_rng(20261027))
            by_seed.setdefault(name, {})[str(seed)] = {
                "delta_mae_best": float(d.mean()),
                "ci95": [float(ci[0]), float(ci[1])],
                "excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                "delta_drop_row172": float(np.delete(d, 172).mean()),
                "ordinary": float(d[masks["ordinary_y_gt_-3"]].mean()),
                "rings_ge4": float(d[masks["rings_ge4"]].mean()),
                "unseen": float(d[masks["radius2_any_unseen"]].mean()),
                "tag_x": have[x], "tag_y": btag,
            }
        have.pop("A1", None); have.pop("A0", None)
    decision["paired_by_seed"] = by_seed
    spreads = {}
    for arm in ("B", "A1", "A0", "S1", "S2", "A0M0"):
        v = [r["best"]["valid_mae"] for r in runs.values() if r["arm"] == arm]
        if len(v) > 1:
            spreads[arm] = {"n": len(v), "values": [float(x) for x in v],
                            "range": float(max(v) - min(v)),
                            "mean": float(np.mean(v))}
    decision["same_arm_seed_spread"] = spreads
    verdicts = {}
    for name, per_seed in by_seed.items():
        signs = {np.sign(v["delta_mae_best"]) for v in per_seed.values()}
        if len(per_seed) > 1 and len(signs) > 1:
            verdicts[name] = "SIGN_FLIP_ACROSS_SEEDS"
        elif all(not v["excludes_zero"] for v in per_seed.values()):
            verdicts[name] = "NO_EFFECT_CI_INCLUDES_ZERO"
        elif all(v["excludes_zero"] for v in per_seed.values()):
            verdicts[name] = "CONSISTENT"
        else:
            verdicts[name] = "MIXED"
    decision["contradiction_verdicts"] = verdicts

    # ---- per-molecule table -------------------------------------------------
    csv_path = AC.ANALYSIS / f"per_molecule_{args.stage}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "y"]
                   + [f"AE_{t}_{c}" for t in tags for c in ("best", "final")])
        for i in range(len(y)):
            w.writerow([i, f"{y[i]:.6f}"]
                       + [f"{aerr[(t, c)][i]:.6f}" for t in tags for c in ("best", "final")])

    # ---- curves -------------------------------------------------------------
    gaps = {}
    for tag, r in runs.items():
        curve = AC.read_json(AC.RUNS / tag / "train.json")["curve"]
        v = np.asarray([e["valid_mae"] for e in curve])
        tr = np.asarray([e["online_train_mae"] for e in curve])
        q = [0, len(v) // 4, len(v) // 2, 3 * len(v) // 4]
        gaps[tag] = {
            "valid_first": float(v[0]), "valid_last": float(v[-1]),
            "valid_min": float(v.min()), "valid_argmin_epoch": int(v.argmin() + 1),
            "eval_train_at_best": r["checkpoints"]["best"]["train_mae_eval_mode"],
            "eval_train_at_final": r["checkpoints"]["final"]["train_mae_eval_mode"],
            "train_valid_gap_at_best": float(
                r["checkpoints"]["best"]["train_mae_eval_mode"] - r["best"]["valid_mae"]),
            "valid_quartile_medians": [float(np.median(v[s:])) for s in q],
            "online_train_quartile_medians": [
                float(np.median(tr[s:s + len(v) // 4])) for s in q],
            "epochs": int(len(v)), "total_updates": int(r["total_updates"]),
            "seconds_per_epoch_mean": float(np.mean([e["seconds"] for e in curve])),
        }

    payload = {
        "round": AC.ROUND_ID, "stage": args.stage, "written": AC.now(),
        "official_valid_loaded": True, "official_test_loaded": False,
        "n_valid": int(len(y)), "metric": "raw-y MAE",
        "main_table": table,
        "seed_means": seed_means,
        "contrasts": contrasts,
        "groups": {"masks": {g: int(m.sum()) for g, m in masks.items()},
                   "support": grp["support"],
                   "definitions": {
                       "ordinary_y_gt_-3": "y > -3",
                       "tail_y_le_-3": "y <= -3",
                       "radius2_*": "exact typed radius-2 rooted signatures counted on the official train 10k",
                       "rings_ge4": "cycle rank E - V + 1 >= 4",
                       "reference_worst10": "the 10 valid rows with the largest |AE| under the frozen BOSI-B seed-0 best checkpoint",
                       "reference_complement_990": "the other 990 rows",
                       "drop_row172": "diagnostic only; row 172 retained in every headline"}},
        "gaps_and_curves": gaps,
        "decision": decision,
        "historical_references": HISTORICAL,
        "gpu_hours_total": float(sum(r["gpu_hours"] for r in runs.values())),
        "bootstrap": {"n": N_BOOT, "kind": "paired over fixed valid molecules, descriptive"},
    }
    out = Path(args.out) if args.out else AC.ANALYSIS / f"summary_{args.stage}.json"
    AC.write_json(out, payload)
    _tables(payload)
    AC.log(f"[aas-analyze {args.stage}] verdict={decision.get('verdict')} "
           f"A1-B={decision.get('delta_A1_minus_B_best')} "
           f"A1-A0={decision.get('delta_A1_minus_A0_best')} -> {AC.rel_or_abs(out)}")
    return 0


def _f(x) -> str:
    return "—" if x is None else f"{x:.6f}"


def _tables(payload: dict) -> None:
    lines = [f"# AAS v1 stage {payload['stage']} — official-valid tables", "",
             f"written {payload['written']} · n_valid={payload['n_valid']} · "
             "official test never loaded", "",
             "## Main table (raw-y MAE)", "",
             "| run | params | GPU-h | best_valid | best_ep | W20 | T100 | final_valid "
             "| train@best | train@final | median AE | P90 AE | top10 share |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, r in payload["main_table"].items():
        b, f = r["best"], r["final"]
        lines.append(
            f"| {tag} | {r['parameters']} | {r['gpu_hours']:.3f} | {_f(r['best_valid'])} "
            f"| {r['best_epoch']} | {_f(r['W20'])} | {_f(r['T100'])} "
            f"| {_f(f['valid_mae'])} | {_f(b['train_mae_eval_mode'])} "
            f"| {_f(f['train_mae_eval_mode'])} | {b['median_ae']:.6f} "
            f"| {b['p90_ae']:.6f} | {b['top10_share_of_abs_error']:.3f} |")
    lines += ["", "## Paired contrasts (negative = first arm better)", "",
              "| contrast | ckpt | ΔMAE | 95% CI | excludes 0 | first better | Δ drop-top10 "
              "| ordinary | tail | all_sup | low_sup | unseen | rings≥4 | worst10 | drop172 "
              "| drop top-1 contributor |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, c in payload["contrasts"].items():
        g = c["delta_by_group"]
        lines.append(
            f"| {c['first']} − {c['second']} | {c['checkpoint']} | {c['delta_mae']:+.6f} "
            f"| [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | {c['ci_excludes_zero']} "
            f"| {c['first_better_fraction']:.3f} | {c['delta_mae_drop_top10']:+.6f} "
            f"| {g.get('ordinary_y_gt_-3', float('nan')):+.6f} "
            f"| {g.get('tail_y_le_-3', float('nan')):+.6f} "
            f"| {g.get('radius2_all_supported', float('nan')):+.6f} "
            f"| {g.get('radius2_low_support', float('nan')):+.6f} "
            f"| {g.get('radius2_any_unseen', float('nan')):+.6f} "
            f"| {g.get('rings_ge4', float('nan')):+.6f} "
            f"| {g.get('reference_worst10', float('nan')):+.6f} "
            f"| {g.get('drop_row172', float('nan')):+.6f} "
            f"| {c['delta_drop_top_contributor']:+.6f} (row {c['top_contributor_row']}) |")
    lines += ["", "## Group sizes", ""]
    for g, n in payload["groups"]["masks"].items():
        lines.append(f"* `{g}`: {n}")
    lines += ["", "## Decision (pre-registered)", "", "```",
              json.dumps(payload["decision"], indent=2, default=str), "```", ""]
    (AC.ANALYSIS / f"tables_{payload['stage']}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
