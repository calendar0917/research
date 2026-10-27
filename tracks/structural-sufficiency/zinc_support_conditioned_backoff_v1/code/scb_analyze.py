"""Phase 1 analysis and pre-registered decision for SCSB v1.

Reads only the frozen Locked scores/predictions produced by ``scb_score.py``.
No official valid, no official test.  All gates are the ones frozen in
``configs/scb_v1.json`` and ``PROTOCOL.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import scb_common as S
import ad_common as C

ARMS = ("L", "S", "P")
SEEDS = (0, 1)
CONTRASTS = (("S", "L"), ("S", "P"), ("L", "P"))
GROUPS_FOR_TABLE = ("primary", "unseen_all", "low4_nopen", "full_nopen",
                    "ordinary_seen", "ordinary_label", "tail", "penalty_all", "all")


def load_pred(arm: str, seed: int, tag: str = "best_monitor"):
    run = S.RUNS / f"{arm}_fit8k_s{seed}"
    p = run / f"predictions_locked_{tag}.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    obj = torch.load(p, map_location="cpu", weights_only=False)
    return {
        "pred": np.asarray(obj["locked_pred"], dtype=np.float64),
        "rows": np.asarray(obj["locked_rows"], dtype=np.int64),
        "y": np.asarray(obj["y"], dtype=np.float64),
        "meta": {k: obj[k] for k in ("arm", "seed", "fit", "checkpoint")},
        "sha256": S.sha256_file(p),
    }


def mae(pred, y, mask=None):
    d = np.abs(pred - y)
    if mask is not None:
        d = d[mask]
    return float(d.mean())


def natural_from_ae(ae, tier):
    total = 0.0
    for k, w in S.NATURAL_WEIGHTS.items():
        total += w * float(ae[tier == k].mean())
    return float(total)


def bootstrap_rows(delta, n_boot=10000, seed=20261028):
    return S.bootstrap_mean(delta, n_boot=n_boot, seed=seed)


def bootstrap_natural(delta, tier, n_boot=10000, seed=20261028):
    rng = np.random.default_rng(seed)
    idx = {k: np.flatnonzero(tier == k) for k in S.NATURAL_WEIGHTS}
    vals = np.zeros(int(n_boot), dtype=np.float64)
    for k, w in S.NATURAL_WEIGHTS.items():
        ii = idx[k]
        draw = rng.integers(0, len(ii), size=(int(n_boot), len(ii)))
        vals += w * delta[ii][draw].mean(axis=1)
    return {"mean": float(sum(w * delta[idx[k]].mean() for k, w in S.NATURAL_WEIGHTS.items())),
            "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "n_boot": int(n_boot), "seed": int(seed)}


def contrast(name, a, b, preds, locked):
    y = locked["y"]
    masks = locked["masks"]
    tier = locked["tier"]
    ae_a = {s: np.abs(preds[a][s] - y) for s in SEEDS}
    ae_b = {s: np.abs(preds[b][s] - y) for s in SEEDS}
    delta = {s: ae_a[s] - ae_b[s] for s in SEEDS}
    delta_mean = np.mean([delta[s] for s in SEEDS], axis=0)
    out = {"contrast": f"{a}-{b}", "a": a, "b": b, "seed_deltas": {}, "gates": {}}

    for g in GROUPS_FOR_TABLE:
        m = masks[g]
        if not m.any():
            continue
        out.setdefault("groups", {})[g] = {
            "n": int(m.sum()),
            "mae_a": {str(s): mae(preds[a][s], y, m) for s in SEEDS},
            "mae_b": {str(s): mae(preds[b][s], y, m) for s in SEEDS},
            "delta_seed": {str(s): float(delta[s][m].mean()) for s in SEEDS},
            "delta_seed_mean": float(delta_mean[m].mean()),
        }

    natural_seed = {s: float(natural_from_ae(ae_a[s], tier) - natural_from_ae(ae_b[s], tier))
                    for s in SEEDS}
    natural_mean = float(np.mean([natural_seed[s] for s in SEEDS]))
    out["natural"] = {
        "delta_seed": {str(s): float(natural_seed[s]) for s in SEEDS},
        "delta_seed_mean": natural_mean,
        "ci": bootstrap_natural(delta_mean, tier),
    }

    prim = masks["primary"]
    d_prim = delta_mean[prim]
    prim_mean = float(d_prim.mean())
    out["primary"] = {
        "n": int(prim.sum()),
        "mae_a": {str(s): mae(preds[a][s], y, prim) for s in SEEDS},
        "mae_b": {str(s): mae(preds[b][s], y, prim) for s in SEEDS},
        "delta_seed": {str(s): float(delta[s][prim].mean()) for s in SEEDS},
        "delta_seed_mean": prim_mean,
        "bootstrap": bootstrap_rows(d_prim),
        "drop_top1": S.drop_top_k(d_prim, 1),
        "drop_top10": S.drop_top_k(d_prim, 10),
        "key_cluster": S.cluster_bootstrap_mean(
            d_prim, locked["clusters"][prim], n_boot=10000, seed=20261028),
    }
    out["seed_stability"] = {
        "max_minus_min": float(max(delta[s][prim].mean() for s in SEEDS)
                               - min(delta[s][prim].mean() for s in SEEDS)),
        "same_sign": bool(not (any(delta[s][prim].mean() < 0 for s in SEEDS)
                               and any(delta[s][prim].mean() > 0 for s in SEEDS))),
    }
    out["safety"] = {
        "tail_max_seed_delta": float(max(delta[s][masks["tail"]].mean() for s in SEEDS)),
        "ordinary_label_max_seed_delta": float(max(delta[s][masks["ordinary_label"]].mean()
                                                   for s in SEEDS)),
        "ordinary_seen_max_seed_delta": float(max(delta[s][masks["ordinary_seen"]].mean()
                                                  for s in SEEDS)),
    }
    return out


def judge(c, cfg):
    g = cfg["phase1"]["gates"]
    p = c["primary"]
    n = c["natural"]
    st = c["seed_stability"]
    sf = c["safety"]
    gates = {
        "primary_effect": bool(p["delta_seed_mean"] <= -g["primary_effect"]
                               and all(v < 0 for v in p["delta_seed"].values())
                               and p["bootstrap"]["ci95"][1] < 0),
        "overall_effect": bool(n["delta_seed_mean"] <= -g["overall_effect"]
                               and all(v < 0 for v in n["delta_seed"].values())
                               and n["ci"]["ci95"][1] < 0),
        "ordinary_loss_cap": bool(sf["ordinary_seen_max_seed_delta"] <= g["ordinary_loss_cap"]),
        "seed_stability": bool(st["same_sign"] and st["max_minus_min"] <= g["seed_stability_max_spread"]),
        "drop_top_k": bool(p["drop_top1"]["mean"] <= -g["drop_top_k_min_effect"]
                           and p["drop_top10"]["mean"] <= -g["drop_top_k_min_effect"]
                           and p["drop_top1"]["mean"] * p["delta_seed_mean"] > 0
                           and p["drop_top10"]["mean"] * p["delta_seed_mean"] > 0),
        "key_cluster": bool(p["key_cluster"]["ci95"][1] < 0
                            and p["key_cluster"]["mean"] * p["delta_seed_mean"] > 0),
        "safety_tail": bool(sf["tail_max_seed_delta"] <= g["safety_tail_loss_cap"]),
        "safety_ordinary_label": bool(sf["ordinary_label_max_seed_delta"]
                                      <= g["safety_ordinary_label_loss_cap"]),
    }
    return {
        "gates": gates,
        "pass": bool(all(gates.values())),
        "safety_pass": bool(gates["safety_tail"] and gates["safety_ordinary_label"]
                            and gates["ordinary_loss_cap"]),
        "near_miss": bool(-0.015 < p["delta_seed_mean"] <= -0.010
                          and all(v < 0 for v in p["delta_seed"].values())
                          and p["bootstrap"]["ci95"][1] < 0),
    }


def decide(judged):
    sl = judged["S-L"]
    sp = judged["S-P"]
    if not (sl["safety_pass"] and sp["safety_pass"]):
        return "STOP_UNSAFE", "ordinary/tail safety cap failed"
    if not (sl["gates"]["seed_stability"] and sp["gates"]["seed_stability"]):
        return "STOP_UNSTABLE", "seed signs or seed spread"
    if sl["pass"] and sp["pass"]:
        return "PASS", "both total and placement gates pass"
    if sl["near_miss"] and sp["near_miss"]:
        return "NEAR_MISS_STOP", "primary in (-0.015,-0.010], same sign, CI<0"
    if sl["pass"] and not sp["pass"]:
        return "STOP_MECHANISM_NOT_SUPPORTED", "S-L passes but S-P does not: gain is not placed"
    if sp["pass"] and not sl["pass"]:
        return "STOP_MECHANISM_NOT_SUPPORTED", "S-P passes but S-L does not: no net total gain"
    if sl["near_miss"] or sp["near_miss"]:
        return "NEAR_MISS_STOP", "one contrast in near-miss band"
    return "STOP_NO_EFFECT", "primary/overall effect below threshold"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="best_monitor")
    args = p.parse_args(argv)
    scores = S.read_json(S.ANALYSIS / "phase1_scores.json")
    cfg = S.load_cfg()

    cache = S.load_train_cache()
    rows = S.split_rows()
    _, per_atom = S.build_signatures(cache)
    support = S.build_support(cache, rows["fit"], per_atom)
    locked = S.group_payload(cache, support, rows["locked"], "train")
    locked["clusters"] = S.key_clusters(support, per_atom, rows["locked"])

    preds = {a: {} for a in ARMS}
    metas = {}
    for a in ARMS:
        for s in SEEDS:
            obj = load_pred(a, s, args.tag)
            if not np.array_equal(obj["rows"], rows["locked"]):
                raise RuntimeError(f"{a} s{s}: locked row order")
            if not np.allclose(obj["y"], locked["y"]):
                raise RuntimeError(f"{a} s{s}: y mismatch")
            preds[a][s] = obj["pred"]
            metas[f"{a}_s{s}"] = {"sha256": obj["sha256"], **obj["meta"]}

    contrasts = {}
    for a, b in CONTRASTS:
        name = f"{a}-{b}"
        contrasts[name] = contrast(name, a, b, preds, locked)
        contrasts[name].update(judge(contrasts[name], cfg))
    decision, reason = decide(contrasts)

    arm_table = {}
    for a in ARMS:
        for s in SEEDS:
            sc = scores["scores"][f"{a}_s{s}"]["reads"][args.tag]
            arm_table[f"{a}_s{s}"] = {
                "primary": sc["groups"]["primary"]["mae"],
                "natural_overall": sc["natural_overall"],
                "locked_mae": sc["locked_mae"],
                "ordinary_seen": sc["groups"]["ordinary_seen"]["mae"],
                "low4_nopen": sc["groups"]["low4_nopen"]["mae"],
                "full_nopen": sc["groups"]["full_nopen"]["mae"],
                "unseen_all": sc["groups"]["unseen_all"]["mae"],
                "tail": sc["groups"]["tail"]["mae"],
                "penalty_all": sc["groups"]["penalty_all"]["mae"],
                "train_mae_at_best": sc["train_mae_at_best"],
                "w20": sc["curve"]["w20"],
                "t100": sc["curve"]["t100"],
                "best_epoch": sc["epoch"],
                "parameters": scores["scores"][f"{a}_s{s}"]["parameters"],
            }

    payload = {
        "protocol": cfg["protocol_id"],
        "utc": C.utcnow(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(S.REPO), "rev-parse", "HEAD"], text=True).strip(),
        "tag": args.tag,
        "decision": decision,
        "reason": reason,
        "phase2_unlocked": bool(decision == "PASS"),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "official_test_policy": "not constructed, not read, not scored",
        "locked_rows_sha256": S.sha256_array(rows["locked"]),
        "group_counts": dict(locked["counts"]),
        "arm_table": arm_table,
        "contrasts": contrasts,
        "prediction_files": metas,
        "alpha_manifest": S.read_json(S.CACHE_DIR / "alpha_manifest.json"),
        "protocol_gates": cfg["phase1"]["gates"],
    }
    S.write_json(S.ANALYSIS / "phase1_decision.json", payload)
    print(json.dumps({"decision": decision, "reason": reason,
                      "S-L": {k: contrasts["S-L"][k] for k in ("pass", "near_miss")},
                      "S-P": {k: contrasts["S-P"][k] for k in ("pass", "near_miss")}},
                     indent=2))
    return 0 if decision in ("PASS", "NEAR_MISS_STOP", "STOP_NO_EFFECT",
                             "STOP_UNSTABLE", "STOP_UNSAFE",
                             "STOP_MECHANISM_NOT_SUPPORTED") else 1


if __name__ == "__main__":
    sys.exit(main())
