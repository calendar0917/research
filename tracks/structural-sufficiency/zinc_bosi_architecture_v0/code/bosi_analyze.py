"""Frozen analysis and decision for BOSI v0.

Consumes only ``analysis/freeze.json``, ``analysis/locked_read.json`` and the
frozen historical DCC-L Locked predictions.  Writes ``analysis/summary.json``,
``analysis/groups.json``, ``analysis/decision.json`` and ``analysis/tables.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bosi_common as BC  # noqa: E402

DCCL_RUNS = BC.DCC_RUNS
N_BOOT = 10_000


def _load_pred(path: Path) -> np.ndarray:
    z = np.load(path)
    return np.asarray(z["pred"], dtype=np.float64)


def _mae(p, y):
    return float(np.mean(np.abs(p - y)))


def _boot_delta(ae_a, ae_b, seed):
    rng = np.random.default_rng(seed)
    d = ae_a - ae_b
    n = len(d)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    return {
        "delta": float(d.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "fraction_better": float(np.mean(d < 0)),
        "median_delta": float(np.median(d)),
    }


def _top10_excluded(ae_a, ae_b):
    worst = np.argsort(-np.maximum(ae_a, ae_b))[:10]
    mask = np.ones(len(ae_a), dtype=bool)
    mask[worst] = False
    return float(ae_a[mask].mean() - ae_b[mask].mean()), int(mask.sum())


def main() -> int:
    freeze = json.loads((BC.ANALYSIS / "freeze.json").read_text())
    locked = json.loads((BC.ANALYSIS / "locked_read.json").read_text())
    rows = BC.split_rows()
    lock_rows = rows["locked"]
    cache = BC.load_train_cache()
    y = BC.labels(cache, lock_rows)

    # DCC-L historical Locked predictions (never retrained)
    dcc = {}
    for seed in (0, 1):
        path = DCCL_RUNS / f"fit8k_s{seed}" / "predictions_best_monitor.pt"
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not np.array_equal(np.asarray(blob["locked_rows"]), lock_rows):
            raise SystemExit("DCC-L locked rows do not match the frozen split")
        dcc[seed] = {"pred": np.asarray(blob["locked_pred"], dtype=np.float64),
                     "path": BC.rel_or_abs(path),
                     "sha256": BC.sha256_file(path)}

    # frozen structural grouping (Fit8k-only support)
    graphs = BC.molecules_from_cache(cache)
    fit_graphs = [graphs[int(r)] for r in rows["fit8k"]]
    lock_graphs = [graphs[int(r)] for r in lock_rows]
    sup = BC.radius2_support(fit_graphs, lock_graphs)
    tier = BC.support_tier(sup["unseen_frac"], sup["low4_frac"])
    n_atoms = np.asarray([len(g[0]) for g in lock_graphs])

    # per-molecule relation usage (static, label-free)
    rel = torch.load(BC.CACHE / "bosi_rel_train.pt", map_location="cpu", weights_only=False)
    hp = np.asarray(rel["half_ptr"], dtype=np.int64)
    pp = np.asarray(rel["partner_ptr"], dtype=np.int64)
    pr = np.asarray(rel["partner_rel"], dtype=np.int64)
    alloc_frac = np.zeros(len(lock_rows))
    tq_frac = np.zeros(len(lock_rows))
    ring_frac = np.zeros(len(lock_rows))
    for i, r in enumerate(lock_rows):
        hs = np.arange(hp[r], hp[r + 1])
        if not len(hs):
            continue
        counts = np.diff(pp)[hs]
        tgt = np.repeat(np.arange(len(hs)), counts)
        part = np.concatenate([np.arange(pp[h], pp[h + 1]) for h in hs]) if counts.sum() else \
            np.zeros(0, np.int64)
        onehot = np.zeros((len(hs), 4), dtype=np.int64)
        if len(part):
            np.add.at(onehot, (tgt, pr[part]), 1)
        alloc_frac[i] = float(np.mean((onehot > 0).sum(1) >= 2))
        tq_frac[i] = float(np.mean((onehot[:, 0] + onehot[:, 1]) > 0))
        ring_frac[i] = float(np.mean(onehot[:, 2] > 0))
    alloc_q = np.quantile(alloc_frac, [0.25, 0.5, 0.75])

    runs = {}
    for tag, per_ckpt in locked["predictions"].items():
        arm = freeze["runs"][tag]["arm"]
        seed = int(freeze["runs"][tag]["seed"])
        for ckpt, meta in per_ckpt.items():
            p = _load_pred(BC.resolve_path(meta["file"]))
            runs[(tag, ckpt)] = {"arm": arm, "seed": seed, "pred": p, "tag": tag,
                                 "locked_mae": _mae(p, y),
                                 "top10": float(np.sort(np.abs(p - y))[-1]),
                                 "train_gpu_hours": freeze["runs"][tag]["gpu_hours"],
                                 "best_monitor": freeze["runs"][tag]["best_monitor"]}

    tags = sorted({t for t, _ in runs})
    seeds = sorted({v["seed"] for (t, c), v in runs.items() if c == "best_monitor"})

    groups = {
        "all": np.ones(len(y), dtype=bool),
        "ordinary": y > -3.0,
        "tail": y <= -3.0,
        "low_support": tier == "low_support",
        "any_unseen": tier == "any_unseen",
        "all_supported": tier == "all_supported",
        "has_T_or_Q": tq_frac > 0,
        "alloc_defined_top_quartile": alloc_frac >= alloc_q[-1],
        "alloc_defined_zero": alloc_frac == 0,
        "rings_ge4": np.asarray([_cycle_rank(g) >= 4 for g in lock_graphs]),
    }

    table = {}
    for tag in tags:
        entry = runs[(tag, "best_monitor")]
        p = entry["pred"]
        ae = np.abs(p - y)
        table[tag] = {"arm": entry["arm"], "seed": entry["seed"],
                      "locked_mae": entry["locked_mae"],
                      "monitor_mae": entry["best_monitor"]["monitor_mae"],
                      "groups": {g: _mae(p[m], y[m]) for g, m in groups.items() if m.sum()},
                      "group_n": {g: int(m.sum()) for g, m in groups.items()},
                      "tail_fraction_of_abs_error": float(ae[groups["tail"]].sum() / ae.sum())
                      if groups["tail"].any() else 0.0}

    contrasts = {}

    def add(name, a, b):
        contrasts[name] = {
            "a": a, "b": b,
            "overall": _boot_delta(np.abs(a - y), np.abs(b - y), 12345),
            "top10_excluded_delta": _top10_excluded(np.abs(a - y), np.abs(b - y))[0],
            "group_delta": {g: float(np.abs(a[m] - y[m]).mean() - np.abs(b[m] - y[m]).mean())
                            for g, m in groups.items() if m.sum()},
        }

    for seed in seeds:
        if f"R_s{seed}" in tags and f"B_s{seed}" in tags:
            add(f"R_s{seed}_minus_B_s{seed}", runs[(f"R_s{seed}", "best_monitor")]["pred"],
                runs[(f"B_s{seed}", "best_monitor")]["pred"])
        if f"R_s{seed}" in tags and seed in dcc:
            add(f"R_s{seed}_minus_DCCL_s{seed}", runs[(f"R_s{seed}", "best_monitor")]["pred"],
                dcc[seed]["pred"])
        if f"B_s{seed}" in tags and seed in dcc:
            add(f"B_s{seed}_minus_DCCL_s{seed}", runs[(f"B_s{seed}", "best_monitor")]["pred"],
                dcc[seed]["pred"])

    # seed means
    seed_mean = {}
    for arm in ("R", "B"):
        ts = [t for t in tags if t.startswith(f"{arm}_s")]
        if ts:
            seed_mean[arm] = float(np.mean([runs[(t, "best_monitor")]["locked_mae"] for t in ts]))
    dcc_mean = float(np.mean([_mae(dcc[s]["pred"], y) for s in sorted(dcc)]))
    dcc_same_seed = {seed: _mae(dcc[seed]["pred"], y) for seed in seeds if seed in dcc}

    rb = {seed: contrasts[f"R_s{seed}_minus_B_s{seed}"]["overall"]["delta"]
          for seed in seeds if f"R_s{seed}_minus_B_s{seed}" in contrasts}
    rd = {seed: contrasts[f"R_s{seed}_minus_DCCL_s{seed}"]["overall"]["delta"]
          for seed in seeds if f"R_s{seed}_minus_DCCL_s{seed}" in contrasts}
    bd = {seed: contrasts[f"B_s{seed}_minus_DCCL_s{seed}"]["overall"]["delta"]
          for seed in seeds if f"B_s{seed}_minus_DCCL_s{seed}" in contrasts}
    r_minus_dcc_same_seed_mean = float(np.mean(list(rd.values()))) if rd else float("nan")
    r_minus_b_mean = float(np.mean(list(rb.values()))) if rb else float("nan")

    cond = {}
    if len(seeds) >= 2 and len(rb) == len(seeds) and len(rd) == len(seeds):
        r_better_b_both = all(v < 0 for v in rb.values())
        r_better_dcc_both = all(v < 0 for v in rd.values())
        ordinary_ok = all(contrasts[f"R_s{s}_minus_DCCL_s{s}"]["group_delta"]["ordinary"] <= 0
                          for s in seeds)
        unseen_ok = all(contrasts[f"R_s{s}_minus_DCCL_s{s}"]["group_delta"]["any_unseen"] <= 0
                        for s in seeds)
        top10_ok = all(contrasts[f"R_s{s}_minus_DCCL_s{s}"]["top10_excluded_delta"] < 0
                       for s in seeds)
        cond = {
            "r_better_than_b_both_seeds": bool(r_better_b_both),
            "r_better_than_dcc_both_seeds": bool(r_better_dcc_both),
            "r_minus_b_seed_mean": r_minus_b_mean,
            "r_minus_dcc_same_seed_mean": r_minus_dcc_same_seed_mean,
            "locked_gain_ge_0.005": bool(-r_minus_dcc_same_seed_mean >= 0.005),
            "ordinary_non_negative_both": bool(ordinary_ok),
            "any_unseen_non_negative_both": bool(unseen_ok),
            "top10_excluded_positive_both": bool(top10_ok),
        }
        if (r_better_b_both and r_better_dcc_both
                and -r_minus_dcc_same_seed_mean >= 0.005
                and ordinary_ok and unseen_ok and top10_ok):
            decision = "PERFORMANCE_PROMISING"
        elif r_better_b_both and -r_minus_b_mean >= 0.003:
            decision = "STRUCTURE_MECHANISM_ONLY"
        elif (not r_better_b_both) or r_minus_dcc_same_seed_mean >= 0.005 or (not top10_ok):
            decision = "NO_GO"
        else:
            decision = "INCONCLUSIVE"
    else:
        cond = {"seeds_available": len(seeds),
                "note": "seed-1 was not bought (pre-registered stop) or not frozen"}
        if rb and all(v < 0 for v in rb.values()) and -r_minus_b_mean >= 0.003:
            decision = "STRUCTURE_MECHANISM_ONLY_SINGLE_SEED"
        elif rb and min(rb.values()) > 0:
            decision = "NO_GO_SINGLE_SEED"
        else:
            decision = "INCONCLUSIVE_SINGLE_SEED"

    summary = {
        "round": BC.ROUND_ID,
        "written": BC.now(),
        "n_locked": int(len(y)),
        "runs": {f"{t}/{c}": {"arm": v["arm"], "seed": v["seed"],
                              "locked_mae": v["locked_mae"]}
                 for (t, c), v in runs.items()},
        "table": table,
        "contrasts": contrasts,
        "seed_mean_locked": {"R": seed_mean.get("R"), "B": seed_mean.get("B"),
                             "DCCL_same_seeds": dcc_same_seed_mean(dcc, y, seeds),
                             "DCCL_two_seed_mean": dcc_mean},
        "decision_conditions": cond,
        "decision": decision,
        "interventions": locked["interventions"],
        "historical_dcc": {str(s): {"locked_mae": _mae(dcc[s]["pred"], y),
                                    "file": dcc[s]["path"],
                                    "sha256": dcc[s]["sha256"]} for s in dcc},
    }
    groups_out = {
        "split_hashes": rows["hashes"],
        "support_source": "Fit8k-only exact typed radius-2 signatures",
        "fit_vocab_size": int(sup["fit_vocab_size"]),
        "tier_n": {t: int((tier == t).sum()) for t in ("all_supported", "low_support", "any_unseen")},
        "group_n": {g: int(m.sum()) for g, m in groups.items()},
        "alloc_defined_fraction_quantiles": {
            "q25": float(alloc_q[0]), "q50": float(alloc_q[1]), "q75": float(alloc_q[2]),
            "mean": float(alloc_frac.mean())},
        "tq_present_molecules": int((tq_frac > 0).sum()),
        "ring_relation_present_molecules": int((ring_frac > 0).sum()),
        "per_molecule": {
            "unseen_frac": sup["unseen_frac"].tolist(),
            "low4_frac": sup["low4_frac"].tolist(),
            "supported_frac": sup["supported_frac"].tolist(),
            "tier": tier.tolist(),
            "allocation_defined_fraction": alloc_frac.tolist(),
            "tq_fraction": tq_frac.tolist(),
            "ring_fraction": ring_frac.tolist(),
            "n_atoms": n_atoms.tolist(),
            "y": y.tolist(),
        },
    }
    BC.write_json(BC.ANALYSIS / "summary.json", summary)
    BC.write_json(BC.ANALYSIS / "groups.json", groups_out)
    BC.write_json(BC.ANALYSIS / "decision.json",
                  {"decision": decision, "conditions": cond,
                   "written": summary["written"]})
    _tables(summary, groups_out)
    BC.log(f"[bosi-analyze] decision = {decision}")
    return 0


def dcc_same_seed_mean(dcc, y, seeds):
    vals = [_mae(dcc[s]["pred"], y) for s in seeds if s in dcc]
    return float(np.mean(vals)) if vals else None


def _cycle_rank(graph):
    atoms, bonds = graph
    return len(bonds) - len(atoms) + 1


def _tables(summary, groups_out):
    lines = ["# BOSI v0 — frozen Locked tables", "",
             f"generated {summary['written']} · n_locked = {summary['n_locked']}", "",
             "## Runs (best-monitor checkpoint)", "",
             "| run | arm | seed | Locked MAE | Monitor MAE |", "|---|---|---:|---:|---:|"]
    for k, v in summary["runs"].items():
        t = summary["table"][k.split("/")[0]]
        lines.append(f"| {k} | {v['arm']} | {v['seed']} | {v['locked_mae']:.6f} "
                     f"| {t['monitor_mae']:.6f} |")
    lines += ["", "## Frozen groups", "",
              "| group | n |", "|---|---:|"]
    for g, n in groups_out["group_n"].items():
        lines.append(f"| {g} | {n} |")
    lines += ["", "## Per-group MAE (best-monitor)", ""]
    tags = sorted(summary["table"])
    cols = " | ".join(tags)
    lines += [f"| group | {cols} |", "|---|" + "---:|" * len(tags)]
    for g in groups_out["group_n"]:
        vals = " | ".join(f"{summary['table'][t]['groups'].get(g, float('nan')):.6f}"
                          for t in tags)
        lines.append(f"| {g} | {vals} |")
    lines += ["", "## Paired contrasts (Locked, negative = first is better)", "",
              "| contrast | ΔMAE | 95% CI | better frac | top-10 dropped |",
              "|---|---:|---|---:|---:|"]
    for name, c in summary["contrasts"].items():
        ci = c["overall"]["ci95"]
        lines.append(f"| {name} | {c['overall']['delta']:+.6f} "
                     f"| [{ci[0]:+.6f}, {ci[1]:+.6f}] | {c['overall']['fraction_better']:.3f} "
                     f"| {c['top10_excluded_delta']:+.6f} |")
    lines += ["", "## Frozen-weight interventions (best-monitor)", "",
              "| run | swap-to-arm MAE | zero-stats MAE | side-zero MAE | rel-perm mean \\|Δy\\| |",
              "|---|---:|---:|---:|---:|"]
    for tag, v in summary["interventions"].items():
        lines.append(f"| {tag} | {v['swap_locked_mae']:.6f} | {v['zero_stats_locked_mae']:.6f} "
                     f"| {v['side_zero_locked_mae']:.6f} | {v['rel_perm_mean_abs_delta']:.6f} |")
    lines += ["", f"**decision = `{summary['decision']}`**", ""]
    (BC.ANALYSIS / "tables.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
