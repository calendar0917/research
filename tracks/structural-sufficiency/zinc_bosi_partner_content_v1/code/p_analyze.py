"""Frozen analysis and decision for the partner-content necessity experiment.

Consumes only ``analysis/freeze.json``, ``analysis/locked_read.json`` and the
reused, hash-verified v0 / DCC-L Locked prediction files.  Writes
``analysis/{summary.json,groups.json,decision.json,tables.md}``.

Primary contrast: ``Delta = MAE(P) - MAE(B)``; positive means the partner
content channel is useful.  The bootstrap is paired over the fixed Locked
molecules and therefore describes molecule sampling only -- it says nothing
about training randomness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bosi_common as BC  # noqa: E402

N_BOOT = 10_000

def _v0path(text):
    """Resolve a path recorded by the v0 round (relative to that round's root)."""
    p = Path(text)
    return p if p.is_absolute() else (BC.V0 / p)

PRIMARY_GROUPS = ("all", "ordinary", "tail", "low_support", "any_unseen",
                  "all_supported", "has_T_or_Q", "rings_ge4",
                  "alloc_defined_top_quartile")


def _mae(p, y):
    return float(np.mean(np.abs(np.asarray(p, np.float64) - np.asarray(y, np.float64))))


def _boot(ae_a, ae_b, seed):
    rng = np.random.default_rng(seed)
    d = np.asarray(ae_a, np.float64) - np.asarray(ae_b, np.float64)
    n = len(d)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    return {
        "delta": float(d.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "fraction_a_better": float(np.mean(d < 0)),
        "median_delta": float(np.median(d)),
        "sign_consistent_ci_excludes_zero": bool(
            np.percentile(means, 2.5) > 0 or np.percentile(means, 97.5) < 0),
    }


def _boot_group(ae_a, ae_b, seed, n_boot=2000):
    """Paired bootstrap inside a frozen subgroup (descriptive only)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(ae_a, np.float64) - np.asarray(ae_b, np.float64)
    n = len(d)
    if n == 0:
        return {"delta": None, "ci95": None, "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    return {"delta": float(d.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n": int(n)}


def _top10_excluded(ae_a, ae_b):
    worst = np.argsort(-np.maximum(ae_a, ae_b))[:10]
    mask = np.ones(len(ae_a), dtype=bool)
    mask[worst] = False
    return float(ae_a[mask].mean() - ae_b[mask].mean()), int(mask.sum())


def _pair(entry_a, entry_b, y, groups, tag):
    ae_a = np.abs(entry_a["pred"] - y)
    ae_b = np.abs(entry_b["pred"] - y)
    overall = _boot(ae_a, ae_b, 1234)
    top10, n_top = _top10_excluded(ae_a, ae_b)
    return {
        "a": entry_a["tag"], "b": entry_b["tag"], "name": tag,
        "overall": overall,
        "top10_excluded_delta": top10,
        "top10_excluded_n": n_top,
        "group_delta": {g: float(ae_a[m].mean() - ae_b[m].mean())
                        for g, m in groups.items()},
        "group_n": {g: int(m.sum()) for g, m in groups.items()},
    }


def _cycle_rank(graph):
    atoms, bonds = graph
    return len(bonds) - len(atoms) + 1


def main() -> int:
    freeze = json.loads((BC.ANALYSIS / "freeze.json").read_text())
    locked = json.loads((BC.ANALYSIS / "locked_read.json").read_text())
    rows = BC.split_rows()
    lock_rows = rows["locked"]
    cache = BC.load_train_cache()
    y = BC.labels(cache, lock_rows)

    # ---- reused, hash-verified predictions (never recomputed) --------------
    reused = {}
    for tag in ("B_s0", "B_s1", "R_s0", "R_s1"):
        meta = freeze["reused"][f"locked_pred/{tag}"]["best_monitor"]
        path = _v0path(meta["file"])
        if BC.sha256_file(path) != meta["sha256"]:
            raise SystemExit(f"reused prediction changed: {tag}")
        z = np.load(path)
        if not np.array_equal(np.asarray(z["row"]), lock_rows):
            raise SystemExit(f"{tag}: locked rows do not match the frozen split")
        if not np.allclose(np.asarray(z["y"], np.float64), y):
            raise SystemExit(f"{tag}: locked labels do not match")
        reused[tag] = {"pred": np.asarray(z["pred"], np.float64), "tag": tag,
                       "arm": freeze["runs"][tag]["arm"],
                       "seed": int(freeze["runs"][tag]["seed"]),
                       "source": str(path), "sha256": meta["sha256"],
                       "locked_mae": meta["locked_mae"]}

    dcc = {}
    for seed in ("0", "1"):
        meta = freeze["reused"][f"dccl/{seed}"]
        path = _v0path(meta["file"])
        if BC.sha256_file(path) != meta["sha256"]:
            raise SystemExit(f"DCC-L prediction changed: seed {seed}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not np.array_equal(np.asarray(blob["locked_rows"]), lock_rows):
            raise SystemExit("DCC-L locked rows do not match the frozen split")
        dcc[int(seed)] = {"pred": np.asarray(blob["locked_pred"], np.float64),
                          "tag": f"DCC-L_s{seed}", "arm": "DCC-L", "seed": int(seed),
                          "source": str(path), "sha256": meta["sha256"],
                          "locked_mae": meta["locked_mae"]}

    runs = {}
    for tag, per_ckpt in locked["predictions"].items():
        info = freeze["runs"][tag]
        for ckpt, meta in per_ckpt.items():
            path = BC.resolve_path(meta["file"])
            if BC.sha256_file(path) != meta["sha256"]:
                raise SystemExit(f"locked prediction changed: {tag}/{ckpt}")
            z = np.load(path)
            runs[(tag, ckpt)] = {
                "tag": tag, "arm": info["arm"], "seed": int(info["seed"]),
                "pred": np.asarray(z["pred"], np.float64), "checkpoint": ckpt,
                "locked_mae": meta["locked_mae"],
                "monitor_mae": info["checkpoints"][ckpt]["monitor_mae_recomputed"],
                "fit_mae": info["checkpoints"][ckpt]["fit_mae"],
                "gpu_hours": info.get("gpu_hours"),
                "sha256": meta["sha256"],
            }
    for tag, entry in reused.items():
        runs[(tag, "best_monitor")] = dict(entry, checkpoint="best_monitor",
                                          locked_mae=entry["locked_mae"],
                                          monitor_mae=freeze["runs"][tag]["checkpoints"]
                                          ["best_monitor"]["monitor_mae_recomputed"],
                                          fit_mae=freeze["runs"][tag]["checkpoints"]
                                          ["best_monitor"]["fit_mae"],
                                          gpu_hours=None)
        fin = freeze["runs"][tag]["checkpoints"]["final"]
        # the reused final prediction is not needed for the primary contrast;
        # its Monitor/Fit readings stay in the table below.
        runs[(tag, "final")] = dict(entry, checkpoint="final", pred=None,
                                    locked_mae=None,
                                    monitor_mae=fin["monitor_mae_recomputed"],
                                    fit_mae=fin["fit_mae"], gpu_hours=None)

    # ---- frozen structural groups (Fit8k-only support, v0 construction) ----
    graphs = BC.molecules_from_cache(cache)
    fit_graphs = [graphs[int(r)] for r in rows["fit8k"]]
    lock_graphs = [graphs[int(r)] for r in lock_rows]
    sup = BC.radius2_support(fit_graphs, lock_graphs)
    tier = BC.support_tier(sup["unseen_frac"], sup["low4_frac"])

    rel = torch.load(BC.CACHE / "bosi_rel_train.pt", map_location="cpu", weights_only=False)
    hp = np.asarray(rel["half_ptr"], np.int64)
    pp = np.asarray(rel["partner_ptr"], np.int64)
    pr = np.asarray(rel["partner_rel"], np.int64)
    alloc_frac = np.zeros(len(lock_rows))
    tq_frac = np.zeros(len(lock_rows))
    for i, r in enumerate(lock_rows):
        hs = np.arange(hp[r], hp[r + 1])
        if not len(hs):
            continue
        counts = np.diff(pp)[hs]
        tgt = np.repeat(np.arange(len(hs)), counts)
        part = (np.concatenate([np.arange(pp[h], pp[h + 1]) for h in hs])
                if counts.sum() else np.zeros(0, np.int64))
        onehot = np.zeros((len(hs), 4), np.int64)
        if len(part):
            np.add.at(onehot, (tgt, pr[part]), 1)
        alloc_frac[i] = float(np.mean((onehot > 0).sum(1) >= 2))
        tq_frac[i] = float(np.mean((onehot[:, 0] + onehot[:, 1]) > 0))
    alloc_q = np.quantile(alloc_frac, [0.25, 0.5, 0.75])

    groups = {
        "all": np.ones(len(y), bool),
        "ordinary": y > -3.0,
        "tail": y <= -3.0,
        "low_support": tier == "low_support",
        "any_unseen": tier == "any_unseen",
        "all_supported": tier == "all_supported",
        "has_T_or_Q": tq_frac > 0,
        "alloc_defined_top_quartile": alloc_frac >= alloc_q[-1],
        "rings_ge4": np.asarray([_cycle_rank(g) >= 4 for g in lock_graphs]),
    }

    seeds = (0, 1)
    contrasts = {}
    for seed in seeds:
        contrasts[f"P_minus_B_s{seed}"] = _pair(
            runs[(f"P_s{seed}", "best_monitor")], runs[(f"B_s{seed}", "best_monitor")],
            y, groups, f"P_minus_B_s{seed}")
        contrasts[f"P_minus_R_s{seed}"] = _pair(
            runs[(f"P_s{seed}", "best_monitor")], runs[(f"R_s{seed}", "best_monitor")],
            y, groups, f"P_minus_R_s{seed}")
        contrasts[f"P_minus_DCCL_s{seed}"] = _pair(
            runs[(f"P_s{seed}", "best_monitor")], dcc[seed], y, groups,
            f"P_minus_DCCL_s{seed}")
        contrasts[f"B_minus_DCCL_s{seed}"] = _pair(
            runs[(f"B_s{seed}", "best_monitor")], dcc[seed], y, groups,
            f"B_minus_DCCL_s{seed}")
    for seed in seeds:
        a = runs[(f"P_s{seed}", "final")]
        b = runs[(f"B_s{seed}", "final")]
        contrasts[f"P_minus_B_s{seed}_final"] = {
            "a": a["tag"], "b": b["tag"], "name": f"P_minus_B_s{seed}_final",
            "locked_delta": None if a["pred"] is None else float("nan"),
            "note": "final checkpoints of the reused B/R runs are not scored on Locked "
                    "in this round; Monitor and Fit finals are in the table",
            "monitor_delta": float(a["monitor_mae"] - b["monitor_mae"]),
            "fit_delta": float(a["fit_mae"] - b["fit_mae"]),
        }

    # ---- pre-registered subgroups: CI and relative size (descriptive) ------
    for s in seeds:
        pa = runs[(f"P_s{s}", "best_monitor")]["pred"]
        pb = runs[(f"B_s{s}", "best_monitor")]["pred"]
        ae_p, ae_b = np.abs(pa - y), np.abs(pb - y)
        c = contrasts[f"P_minus_B_s{s}"]
        c["group_ci95"] = {g: _boot_group(ae_p[groups[g]], ae_b[groups[g]], 7000 + 13 * s + i)
                           for i, g in enumerate(PRIMARY_GROUPS)}
        c["group_relative_delta"] = {
            g: float(c["group_delta"][g] / ae_b[groups[g]].mean())
            if ae_b[groups[g]].mean() > 0 else None
            for g in PRIMARY_GROUPS}
        c["group_mae_B"] = {g: float(ae_b[groups[g]].mean()) for g in PRIMARY_GROUPS}

    # ---- decision (pre-registered) ----------------------------------------
    d = {s: contrasts[f"P_minus_B_s{s}"]["overall"]["delta"] for s in seeds}
    both_pos = all(v > 0 for v in d.values())
    both_neg = all(v < 0 for v in d.values())
    mean_d = float(np.mean(list(d.values())))
    ordinary_ok = all(contrasts[f"P_minus_B_s{s}"]["group_delta"]["ordinary"] > 0
                      for s in seeds)
    top10_ok = all(contrasts[f"P_minus_B_s{s}"]["top10_excluded_delta"] > 0
                   for s in seeds)
    unseen_ok = all(contrasts[f"P_minus_B_s{s}"]["group_delta"]["any_unseen"] > 0
                    for s in seeds)
    ci_excludes = {s: contrasts[f"P_minus_B_s{s}"]["overall"]
                   ["sign_consistent_ci_excludes_zero"] for s in seeds}
    cond = {
        "delta_definition": "MAE(P) - MAE(B); positive = partner content useful",
        "delta_per_seed": {str(s): d[s] for s in seeds},
        "delta_seed_mean": mean_d,
        "p_worse_than_b_both_seeds": bool(both_pos),
        "p_better_than_b_both_seeds": bool(both_neg),
        "ordinary_positive_both": bool(ordinary_ok),
        "top10_excluded_positive_both": bool(top10_ok),
        "any_unseen_positive_both": bool(unseen_ok),
        "ci_excludes_zero_per_seed": {str(s): bool(ci_excludes[s]) for s in seeds},
        "seed_mean_ge_0.005": bool(abs(mean_d) >= 0.005),
        "monitor_delta_per_seed": {str(s): float(
            runs[(f"P_s{s}", "best_monitor")]["monitor_mae"]
            - runs[(f"B_s{s}", "best_monitor")]["monitor_mae"]) for s in seeds},
        "fit_delta_per_seed": {str(s): float(
            runs[(f"P_s{s}", "best_monitor")]["fit_mae"]
            - runs[(f"B_s{s}", "best_monitor")]["fit_mae"]) for s in seeds},
    }
    if both_pos and mean_d >= 0.005 and ordinary_ok and top10_ok:
        decision = "PARTNER_CONTENT_SUPPORTED"
    elif both_neg and (-mean_d) >= 0.005 and (not top10_ok):
        decision = "SIMPLER_BODY_PROMISING"
    elif both_neg and (-mean_d) >= 0.005 and top10_ok:
        decision = "SIMPLER_BODY_PROMISING"
    else:
        decision = "PARTNER_CONTENT_NOT_ESTABLISHED"
    cond["interpretation_note"] = (
        "bootstrap CIs describe molecule sampling only; training randomness enters "
        "through the seed-to-seed spread, which is reported directly")

    # ---- write ------------------------------------------------------------
    table = {}
    for tag in sorted({t for t, _ in runs}):
        per = {}
        for ckpt in ("best_monitor", "final"):
            e = runs.get((tag, ckpt))
            if e is None:
                continue
            per[ckpt] = {"monitor_mae": e["monitor_mae"], "fit_mae": e["fit_mae"],
                         "locked_mae": e["locked_mae"],
                         "step": freeze["runs"][tag]["checkpoints"][ckpt]["step"],
                         "epoch": freeze["runs"][tag]["checkpoints"][ckpt]["epoch"]}
        table[tag] = {"arm": runs[(tag, "best_monitor")]["arm"],
                      "seed": runs[(tag, "best_monitor")]["seed"],
                      "checkpoints": per,
                      "gpu_hours": freeze["runs"][tag].get("gpu_hours")
                      or freeze["runs"][tag].get("source")}

    summary = {
        "round": BC.ROUND_ID,
        "written": BC.now(),
        "n_locked": int(len(y)),
        "runs": table,
        "contrasts": contrasts,
        "interventions": locked["interventions"],
        "v0_interventions_source": freeze["reused"]["v0_locked_read"]["sha256"],
        "seed_mean_locked": {
            "P": float(np.mean([_mae(runs[(f"P_s{s}", "best_monitor")]["pred"], y)
                                for s in seeds])),
            "B": float(np.mean([_mae(runs[(f"B_s{s}", "best_monitor")]["pred"], y)
                                for s in seeds])),
            "R": float(np.mean([_mae(runs[(f"R_s{s}", "best_monitor")]["pred"], y)
                                for s in seeds])),
            "DCCL": float(np.mean([_mae(dcc[s]["pred"], y) for s in seeds])),
        },
        "decision_conditions": cond,
        "decision": decision,
        "gpu_hours_total": freeze.get("gpu_hours_total"),
    }
    groups_out = {
        "split_hashes": rows["hashes"],
        "group_n": {g: int(m.sum()) for g, m in groups.items()},
        "tier_n": {t: int((tier == t).sum())
                   for t in ("all_supported", "low_support", "any_unseen")},
        "alloc_defined_fraction_quantiles": {"q25": float(alloc_q[0]),
                                            "q50": float(alloc_q[1]),
                                            "q75": float(alloc_q[2]),
                                            "mean": float(alloc_frac.mean())},
        "tq_present_molecules": int((tq_frac > 0).sum()),
    }
    BC.write_json(BC.ANALYSIS / "summary.json", summary)
    BC.write_json(BC.ANALYSIS / "groups.json", groups_out)
    BC.write_json(BC.ANALYSIS / "decision.json",
                  {"decision": decision, "conditions": cond,
                   "written": summary["written"]})
    _tables(summary, groups_out)
    BC.log(f"[p-analyze] decision = {decision}")
    return 0


def _tables(summary, groups_out):
    L = ["# BOSI-P v1 — frozen Locked tables", "",
         f"generated {summary['written']} · n_locked = {summary['n_locked']}", "",
         "## Runs (Monitor and Fit from the freeze, Locked from the single read)", "",
         "| run | arm | seed | checkpoint | step | Fit MAE | Monitor MAE | Locked MAE |",
         "|---|---|---|---|---:|---:|---:|---:|"]
    for tag, e in summary["runs"].items():
        for ckpt, v in e["checkpoints"].items():
            lock = "—" if v["locked_mae"] is None else f"{v['locked_mae']:.6f}"
            L.append(f"| {tag} | {e['arm']} | {e['seed']} | {ckpt} | {v['step']} "
                     f"| {v['fit_mae']:.6f} | {v['monitor_mae']:.6f} | {lock} |")
    L += ["", "## Seed means (Locked, best-monitor)", "",
          "| arm | Locked MAE |", "|---|---:|"]
    for k, v in summary["seed_mean_locked"].items():
        L.append(f"| {k} | {v:.6f} |")
    L += ["", "## Paired contrasts (Delta = first − second; positive = first worse)", "",
          "| contrast | ΔMAE | 95% CI | frac first better | median | top-10 dropped |",
          "|---|---:|---|---:|---:|---:|"]
    for name, c in summary["contrasts"].items():
        if "overall" not in c:
            continue
        ci = c["overall"]["ci95"]
        L.append(f"| {name} | {c['overall']['delta']:+.6f} "
                 f"| [{ci[0]:+.6f}, {ci[1]:+.6f}] | {c['overall']['fraction_a_better']:.3f} "
                 f"| {c['overall']['median_delta']:+.6f} | {c['top10_excluded_delta']:+.6f} |")
    L += ["", "## Group deltas for the primary contrast (positive = P worse)", "",
          "| group | n | s0 | s1 |", "|---|---:|---:|---:|"]
    for g, n in groups_out["group_n"].items():
        v0 = summary["contrasts"]["P_minus_B_s0"]["group_delta"][g]
        v1 = summary["contrasts"]["P_minus_B_s1"]["group_delta"][g]
        L.append(f"| {g} | {n} | {v0:+.6f} | {v1:+.6f} |")
    L += ["", "## Pre-registered subgroups, primary contrast (Delta = P − B)", "",
          "| group | n | s0 Δ | s0 CI | s0 rel | s1 Δ | s1 CI | s1 rel |",
          "|---|---:|---:|---|---:|---:|---|---:|"]
    for g, n in groups_out["group_n"].items():
        if g not in PRIMARY_GROUPS:
            continue
        cells = []
        for s in (0, 1):
            c = summary["contrasts"][f"P_minus_B_s{s}"]
            ci = c["group_ci95"][g]["ci95"]
            rel = c["group_relative_delta"][g]
            cells += [f"{c['group_delta'][g]:+.6f}",
                      f"[{ci[0]:+.6f}, {ci[1]:+.6f}]" if ci else "—",
                      f"{rel * 100:.1f}%" if rel is not None else "—"]
        L.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
    L += ["", "## Frozen-weight interventions (P, best-monitor)", "",
          "| run | swap→B | swap→R | M zeroed | side zeroed |", "|---|---:|---:|---:|---:|"]
    for tag, v in summary["interventions"].items():
        L.append(f"| {tag} | {v['swap_to_B_locked_mae']:.6f} "
                 f"| {v['swap_to_R_locked_mae']:.6f} | {v['zero_stats_locked_mae']:.6f} "
                 f"| {v['side_zero_locked_mae']:.6f} |")
    L += ["", f"**decision = `{summary['decision']}`**", ""]
    (BC.ANALYSIS / "tables.md").write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
