"""Frozen analysis and decision for the statistics-only SSL round.

Consumes only ``analysis/freeze.json``, ``analysis/locked_read.json`` and the
hash-verified BOSI-B Locked predictions of v0.  Writes ``summary.json``,
``groups.json``, ``tables.md`` and ``decision.json``.

Decision vocabulary (PROTOCOL.md §6):
    SSL_PROMISING, GAIN_FROM_EXTRA_COMPUTE_ONLY, NO_GAIN, INCONCLUSIVE,
    PRETEXT_NO_GO.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s_common as SC  # noqa: E402

N_BOOT = 10_000
SSL_GAIN_BAR = 0.005        # seed-mean Locked MAE improvement required
C_GAIN_BAR = 0.003          # independent gain over the compute control
MONITOR_BAR = 0.004         # seed-0 purchase gate (PROTOCOL.md §5)


def _pred(path: Path) -> np.ndarray:
    z = np.load(path)
    return np.asarray(z["pred"], dtype=np.float64)


def _mae(p, y):
    return float(np.mean(np.abs(p - y)))


def _boot(ae_a, ae_b, seed):
    rng = np.random.default_rng(seed)
    d = ae_a - ae_b
    n = len(d)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    return {"delta": float(d.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "fraction_first_better": float(np.mean(d < 0)),
            "median_delta": float(np.median(d))}


def _top10(ae_a, ae_b):
    worst = np.argsort(-np.maximum(ae_a, ae_b))[:10]
    mask = np.ones(len(ae_a), dtype=bool)
    mask[worst] = False
    return float(ae_a[mask].mean() - ae_b[mask].mean()), int(mask.sum())


def _contrast(pa, pb, y, groups, seed_tag, label):
    ae_a, ae_b = np.abs(pa - y), np.abs(pb - y)
    out = {"label": label, "seed": seed_tag, **_boot(ae_a, ae_b, 0),
           "rmse_a": float(np.sqrt(np.mean((pa - y) ** 2))),
           "rmse_b": float(np.sqrt(np.mean((pb - y) ** 2)))}
    out["top10_removed_delta"], out["top10_removed_n"] = _top10(ae_a, ae_b)
    out["groups"] = {}
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        out["groups"][name] = {
            "n": int(mask.sum()),
            "mae_a": float(ae_a[mask].mean()),
            "mae_b": float(ae_b[mask].mean()),
            "delta": float(ae_a[mask].mean() - ae_b[mask].mean()),
        }
    return out


def main() -> int:
    freeze = json.loads((SC.ANALYSIS / "freeze.json").read_text())
    locked = json.loads((SC.ANALYSIS / "locked_read.json").read_text())
    rows = SC.split_rows()
    lock_rows = rows["locked"]
    cache = SC.load_train_cache()
    rel = SC.load_relation_cache()
    y = SC.labels(cache, lock_rows)
    groups = SC.group_definitions(lock_rows, cache, rel)

    preds: dict[str, dict[str, np.ndarray]] = {}
    meta: dict[str, dict] = {}
    for tag, per_ckpt in locked["predictions"].items():
        preds[tag] = {}
        meta[tag] = {"kind": "trained", "seed": int(freeze["runs"][tag]["seed"]),
                     "init": freeze["runs"][tag]["init"]["init"]}
        for ckpt, m in per_ckpt.items():
            preds[tag][ckpt] = _pred(SC.resolve_path(m["file"]))
    for seed in (0, 1):
        tag = f"B_s{seed}"
        preds[tag] = {}
        meta[tag] = {"kind": "reused", "seed": seed, "init": "historical"}
        for ckpt, m in freeze["reused"]["locked_pred"].items():
            if not ckpt.startswith(tag + "/"):
                continue
            key = ckpt.split("/")[1]
            preds[tag][key] = _pred(Path(m["file"]))
            if not np.array_equal(np.load(Path(m["file"]))["row"], lock_rows):
                raise SystemExit("reused BOSI-B locked rows do not match the frozen split")

    summary = {
        "round": SC.ROUND_ID,
        "analysis_utc": SC.now(),
        "n_locked": int(len(y)),
        "locked_rows_sha256": rows["hashes"]["locked"],
        "n_boot": N_BOOT,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "runs": {},
        "contrasts": {},
        "interventions": locked["interventions"],
        "pilot": freeze.get("pilot_decision"),
        "gpu_ledger": freeze.get("gpu_ledger"),
    }
    for tag in sorted(preds):
        for ckpt in sorted(preds[tag]):
            summary["runs"][f"{tag}/{ckpt}"] = {
                **meta[tag], "checkpoint": ckpt,
                "locked_mae": _mae(preds[tag][ckpt], y),
                "rmse": float(np.sqrt(np.mean((preds[tag][ckpt] - y) ** 2))),
            }

    ssl_tags = sorted(t for t in preds if t.startswith("S_s"))
    c_tags = sorted(t for t in preds if t.startswith("C_s"))
    for tag in ssl_tags + c_tags:
        seed = meta[tag]["seed"]
        base = f"B_s{seed}"
        if base not in preds:
            continue
        summary["contrasts"][f"{tag} - {base}"] = _contrast(
            preds[tag]["best_monitor"], preds[base]["best_monitor"], y, groups,
            seed, f"{tag} - {base}")
    for tag in ssl_tags:
        seed = meta[tag]["seed"]
        ctag = f"C_s{seed}"
        if ctag in preds:
            summary["contrasts"][f"{tag} - {ctag}"] = _contrast(
                preds[tag]["best_monitor"], preds[ctag]["best_monitor"], y, groups,
                seed, f"{tag} - {ctag}")
    for tag in c_tags:
        seed = meta[tag]["seed"]
        base = f"B_s{seed}"
        summary["contrasts"][f"{tag} - {base}"] = _contrast(
            preds[tag]["best_monitor"], preds[base]["best_monitor"], y, groups,
            seed, f"{tag} - {base}")

    # ---- decision -------------------------------------------------------
    decision = _decide(freeze, summary, ssl_tags, c_tags, meta)
    summary["decision"] = decision
    summary["groups"] = {k: int(v.sum()) for k, v in groups.items()}

    SC.write_json(SC.ANALYSIS / "summary.json", summary)
    SC.write_json(SC.ANALYSIS / "groups.json",
                  {k: {"n": int(v.sum()), "definition": k} for k, v in groups.items()})
    SC.write_json(SC.ANALYSIS / "decision.json", decision)
    (SC.ANALYSIS / "tables.md").write_text(_tables(summary))
    SC.log(f"[s-analyze] decision {decision['verdict']}: {decision['reason']}")
    return 0


def _decide(freeze, summary, ssl_tags, c_tags, meta) -> dict:
    checks: dict[str, object] = {}
    pilot = freeze.get("pilot_decision") or {}
    d_ce = pilot.get("probe_delta_ce") or {}
    # the pilot gate itself is re-evaluated here from the recorded probe numbers
    atom_vs_prior = d_ce.get("atom_vs_marginal")
    bond_vs_prior = d_ce.get("bond_vs_marginal")
    checks["pilot_atom_beats_prior"] = bool(atom_vs_prior is not None and atom_vs_prior >= 0.10)
    checks["pilot_bond_beats_prior"] = bool(bond_vs_prior is not None and bond_vs_prior >= 0.02)
    checks["pilot_atom_uses_partner_stats"] = bool(
        d_ce.get("atom_shuffled_partner") is not None
        and d_ce["atom_shuffled_partner"] >= 0.02)
    checks["pilot_bond_uses_partner_stats"] = bool(
        d_ce.get("bond_shuffled_partner") is not None
        and d_ce["bond_shuffled_partner"] >= 0.02)
    if not all(v for k, v in checks.items() if k.startswith("pilot_")):
        return {"verdict": "PRETEXT_NO_GO",
                "reason": ("the pilot pretext did not learn beyond the marginal prior or "
                           "the partner-statistics channel was not used"),
                "checks": checks, "frozen": True}

    deltas = {t: summary["contrasts"].get(f"{t} - B_s{meta[t]['seed']}", {}).get("delta")
              for t in ssl_tags}
    avail = {t: d for t, d in deltas.items() if d is not None}
    checks["ssl_seeds_run"] = sorted(int(meta[t]["seed"]) for t in ssl_tags)
    checks["monitor_deltas"] = {
        t: (freeze["runs"][t]["best_monitor"]["monitor_mae"]
            - SC.HISTORICAL_B_MONITOR[int(meta[t]["seed"])]) for t in ssl_tags}
    checks["seed0_monitor_gate"] = bool(
        checks["monitor_deltas"].get("S_s0", 1.0) <= -MONITOR_BAR)
    if len(avail) < 2:
        return {"verdict": "NO_GAIN",
                "reason": ("stopped at the pre-registered seed-0 Monitor gate; "
                           "a single seed cannot satisfy the two-seed rule"),
                "checks": checks, "locked_deltas": avail, "frozen": True}

    mean_delta = float(np.mean(list(avail.values())))
    checks["seed_mean_locked_delta"] = mean_delta
    checks["both_seeds_better"] = bool(all(d < 0 for d in avail.values()))
    checks["seed_mean_le_minus_0p005"] = bool(mean_delta <= -SSL_GAIN_BAR)

    ord_d = {t: summary["contrasts"][f"{t} - B_s{meta[t]['seed']}"]["groups"]["ordinary"]["delta"]
             for t in avail}
    top_d = {t: summary["contrasts"][f"{t} - B_s{meta[t]['seed']}"]["top10_removed_delta"]
             for t in avail}
    uns_d = {t: summary["contrasts"][f"{t} - B_s{meta[t]['seed']}"]["groups"]["any_unseen"]["delta"]
             for t in avail}
    checks["ordinary_deltas"] = ord_d
    checks["top10_removed_deltas"] = top_d
    checks["any_unseen_deltas"] = uns_d
    checks["ordinary_all_negative"] = bool(all(d < 0 for d in ord_d.values()))
    checks["top10_all_negative"] = bool(all(d < 0 for d in top_d.values()))
    checks["any_unseen_not_worse"] = bool(
        float(np.mean(list(uns_d.values()))) <= 0.0 and all(d <= 0.002 for d in uns_d.values()))

    needed = ("both_seeds_better", "seed_mean_le_minus_0p005", "ordinary_all_negative",
              "top10_all_negative", "any_unseen_not_worse")
    if not all(checks[k] for k in needed):
        mixed = len({d < 0 for d in avail.values()}) > 1
        checks["direction_mixed_across_seeds"] = bool(mixed)
        verdict = "INCONCLUSIVE" if (mean_delta < 0 and mixed) else "NO_GAIN"
        return {"verdict": verdict,
                "reason": ("the frozen two-seed Locked conditions are not met"
                           + (" (seed directions disagree)" if mixed else "")),
                "checks": checks, "locked_deltas": avail, "frozen": True}

    if not c_tags:
        return {"verdict": "INCONCLUSIVE",
                "reason": "the compute control C was not purchased",
                "checks": checks, "locked_deltas": avail, "frozen": True}
    c_d = {t: summary["contrasts"][f"{t} - C_s{meta[t]['seed']}"]["delta"]
           for t in avail if f"{t} - C_s{meta[t]['seed']}" in summary["contrasts"]}
    cb_d = {t: summary["contrasts"].get(f"C_s{meta[t]['seed']} - B_s{meta[t]['seed']}", {})
            .get("delta") for t in avail}
    checks["ssl_minus_c_deltas"] = c_d
    checks["c_minus_b_deltas"] = cb_d
    checks["ssl_beats_c"] = bool(c_d and float(np.mean(list(c_d.values()))) <= -C_GAIN_BAR
                                 and all(d < 0 for d in c_d.values()))
    if checks["ssl_beats_c"]:
        return {"verdict": "SSL_PROMISING",
                "reason": ("two-seed Locked improvement over BOSI-B with the frozen "
                           "conditions, plus an independent gain over the "
                           "compute-matched control C"),
                "checks": checks, "locked_deltas": avail, "frozen": True}
    return {"verdict": "GAIN_FROM_EXTRA_COMPUTE_ONLY",
            "reason": ("the gain over BOSI-B survives, but the compute-matched control C "
                       "explains it"),
            "checks": checks, "locked_deltas": avail, "frozen": True}


def _tables(summary) -> str:
    L = ["# BOSI statistics-only SSL v1 — frozen tables", ""]
    L.append("## Locked MAE (one read, n=%d)" % summary["n_locked"])
    L.append("")
    L.append("| run | init | seed | ckpt | Locked MAE | RMSE |")
    L.append("|---|---|---:|---|---:|---:|")
    for k, v in summary["runs"].items():
        L.append(f"| {k} | {v['init']} | {v['seed']} | {v['checkpoint']} | "
                 f"{v['locked_mae']:.6f} | {v['rmse']:.4f} |")
    L.append("")
    L.append("## Paired contrasts (negative = first better)")
    L.append("")
    L.append("| contrast | ΔMAE | CI95 | better frac | top-10 removed | ordinary | any-unseen |")
    L.append("|---|---:|---|---:|---:|---:|---:|")
    for k, v in summary["contrasts"].items():
        g = v["groups"]
        L.append(f"| {k} | {v['delta']:+.6f} | [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] | "
                 f"{v['fraction_first_better']:.3f} | {v['top10_removed_delta']:+.6f} | "
                 f"{g['ordinary']['delta']:+.6f} | "
                 f"{(g['any_unseen']['delta'] if 'any_unseen' in g else float('nan')):+.6f} |")
    L.append("")
    d = summary["decision"]
    L.append(f"## Decision: `{d['verdict']}`")
    L.append("")
    L.append(d["reason"])
    L.append("")
    for k, v in d["checks"].items():
        L.append(f"- `{k}`: {json.dumps(v)}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
