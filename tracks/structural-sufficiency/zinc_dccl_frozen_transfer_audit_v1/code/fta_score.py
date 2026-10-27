"""One-shot Monitor scoring of every frozen probe cell.

Runs only after ``analysis/probe_freeze.json`` exists.  This is the single
diagnostic read of the Monitor1k labels.  Official valid/test and the internal
Locked1k are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

import fta_common as K

DIM = 513
HIDDEN = 64


def build_head_from_state(state: dict) -> torch.nn.Sequential:
    head = torch.nn.Sequential(
        torch.nn.Linear(DIM, HIDDEN),
        torch.nn.SiLU(),
        torch.nn.Linear(HIDDEN, 1),
    )
    head.load_state_dict({k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                          for k, v in state.items()})
    return head


def monitor_groups(cache, rows, y_mon, raw4k_pred, cfg) -> tuple[dict, list[dict]]:
    g_mon = K.molecule_graphs(cache, rows["monitor"])
    g4 = K.molecule_graphs(cache, rows["fit4k"])
    g8 = K.molecule_graphs(cache, rows["fit8k"])
    freq4 = K.support_frequency(g4)
    freq8 = K.support_frequency(g8)
    tier4 = K.support_tiers(g_mon, freq4)
    tier8 = K.support_tiers(g_mon, freq8)
    rings = [K.ring_stats(g) for g in g_mon]
    max_ring = np.asarray([r["max_ring_size"] for r in rings])
    ordinary = max_ring <= 6
    unseen4 = tier4["unseen_frac"] > 0
    unseen8 = tier8["unseen_frac"] > 0
    low4 = (~unseen4) & (tier4["supported_frac"] < 0.999)
    low8 = (~unseen8) & (tier8["supported_frac"] < 0.999)
    full4 = tier4["supported_frac"] >= 0.999
    full8 = tier8["supported_frac"] >= 0.999

    # "fixed influence" rows: top-10 raw Fit4k error, seed-mean, final checkpoint
    top10 = np.argsort(-np.abs(raw4k_pred - y_mon), kind="stable")[:10]
    drop_mask = np.ones(len(y_mon), dtype=bool)
    drop_mask[top10] = False

    masks = {
        "overall": np.ones(len(y_mon), dtype=bool),
        "ordinary_nopenalty": ordinary,
        "penalised_ring_gt6": ~ordinary,
        "any_unseen_fit4k": unseen4 & ordinary,
        "any_unseen_fit8k": unseen8 & ordinary,
        "low_support_fit4k": low4,
        "low_support_fit8k": low8,
        "full_support_fit4k": full4,
        "full_support_fit8k": full8,
        "newly_covered_4k_to_8k": unseen4 & ~unseen8,
        "still_unseen_both": unseen4 & unseen8,
        "extreme_tail": y_mon <= -3.0,
        "non_tail": y_mon > -3.0,
        "drop_fixed_top10": drop_mask,
    }
    groups = {
        "definitions": {
            "support_rule": cfg["groups"]["support_rule"],
            "no_penalty_rule": cfg["groups"]["no_penalty_rule"],
            "extreme_tail_rule": cfg["groups"]["extreme_tail_rule"],
            "drop_fixed_top10_rule": cfg["groups"]["drop_fixed_top10_rule"],
        },
        "n": {k: int(v.sum()) for k, v in masks.items()},
        "drop_fixed_top10_rows": [int(rows["monitor"][i]) for i in top10],
        "fit8k_unique_environments": int(len(freq8)),
        "fit4k_unique_environments": int(len(freq4)),
        "n_penalised_ring_gt6": int((~ordinary).sum()),
    }
    annotated = []
    for i in range(len(y_mon)):
        annotated.append({
            "monitor_index": i,
            "row_id": int(rows["monitor"][i]),
            "y": float(y_mon[i]),
            "n_atoms": rings[i]["n_atoms"],
            "n_rings": rings[i]["n_rings"],
            "cycle_rank": rings[i]["cycle_rank"],
            "max_ring_size": rings[i]["max_ring_size"],
            "cycle_excess": rings[i]["cycle_excess"],
            "ordinary_nopenalty": bool(ordinary[i]),
            "unseen_frac_fit4k": float(tier4["unseen_frac"][i]),
            "low4_frac_fit4k": float(tier4["low4_frac"][i]),
            "supported_frac_fit4k": float(tier4["supported_frac"][i]),
            "min_support_fit4k": int(tier4["min_support"][i]),
            "unseen_frac_fit8k": float(tier8["unseen_frac"][i]),
            "low4_frac_fit8k": float(tier8["low4_frac"][i]),
            "supported_frac_fit8k": float(tier8["supported_frac"][i]),
            "min_support_fit8k": int(tier8["min_support"][i]),
            "any_unseen_fit4k": bool(unseen4[i]),
            "any_unseen_fit8k": bool(unseen8[i]),
            "in_drop_fixed_top10": bool(not drop_mask[i]),
        })
    return masks, groups, annotated


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    t_start = time.perf_counter()

    out_json = K.ANALYSIS / "probe_monitor_metrics.json"
    if out_json.exists() and not args.force:
        raise SystemExit(f"{out_json} exists; one-shot scoring already done (use --force to log a re-read)")

    cfg = K.load_config()
    freeze = json.loads((K.ANALYSIS / "representation_freeze.json").read_text())
    probe = json.loads((K.ANALYSIS / "probe_freeze.json").read_text())
    cache = K.load_train_cache()
    rows = K.split_rows()
    pf, ps, split_info = K.probe_split(cache, rows["fit4k"])
    export_rows = np.concatenate([pf, ps, rows["monitor"]])
    assert K.row_hash(export_rows) == freeze["export_rows"]["sha256"]
    n_fit, n_sel = len(pf), len(ps)
    pos_mon = np.arange(n_fit + n_sel, len(export_rows))

    y_export = np.asarray(cache["labels"], dtype=np.float64)[export_rows]
    y_mon = y_export[pos_mon]

    zs = {f"{f}_s{s}": np.load(K.Z_DIR / f"{f}_s{s}.npz") for f, s in K.CONDITIONS}
    raw_pred_mon = {cid: zs[cid]["pred"][pos_mon].astype(np.float64) for cid in zs}

    # ---------- E5: exported raw Monitor MAE must equal the frozen trajectory value
    e5 = {}
    for fit, seed in K.CONDITIONS:
        cid = f"{fit}_s{seed}"
        _, info_path = K.trunk_paths(fit, seed)
        frozen_mae = float(json.loads(info_path.read_text())["eval_reads"]["final"]["monitor_mae"])
        got = float(np.abs(raw_pred_mon[cid] - y_mon).mean())
        e5[cid] = {"exported_monitor_mae": got, "frozen_train_json_mae": frozen_mae,
                   "abs_diff": abs(got - frozen_mae)}
        if abs(got - frozen_mae) > 1e-6:
            raise RuntimeError(f"E5 failed for {cid}: {e5[cid]}")

    masks, groups, annotated = monitor_groups(
        cache, rows, y_mon,
        raw4k_pred=np.mean([raw_pred_mon["fit4k_s0"], raw_pred_mon["fit4k_s1"]], axis=0),
        cfg=cfg)

    # ---------- score every frozen cell
    preds, cells = {}, []
    for c in probe["cells"]:
        cid = c["condition"]
        z_all = zs[cid]["z"]
        mu, sd = K.standardizer(z_all[:n_fit])
        z_mon = K.apply_standardizer(z_all[pos_mon], mu, sd)
        z_mon = np.ascontiguousarray(z_mon * K.block_mask(c["block"]).astype(np.float32))
        zt = torch.from_numpy(z_mon)
        for tag in ("best", "final"):
            path = K.HEAD_DIR / f"{c['id']}.{tag}.pt"
            if K.sha256_file(path) != c[f"{tag}_state_sha256"]:
                raise RuntimeError(f"{c['id']} {tag} state hash drift")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            head = build_head_from_state(ck["state_dict"])
            head.eval()
            with torch.no_grad():
                pred = head(zt).numpy().reshape(-1).astype(np.float64)
            key = f"{c['id']}__{tag}"
            preds[key] = pred
            cells.append({
                "cell": c["id"], "checkpoint": tag, "kind": c["kind"],
                "condition": cid, "block": c["block"], "head_init": c["head_init"],
                "target": c["target"],
                "monitor_mae": float(np.abs(pred - y_mon).mean()),
                "probe_sel_l1_frozen": float(c["best"]["probe_sel_l1"] if tag == "best"
                                             else c["final"]["probe_sel_l1"]),
                "pred_sha256": hashlib.sha256(pred.astype(np.float32).tobytes()).hexdigest(),
            })
        K.log(f"scored {c['id']}")

    raw_metrics = {}
    for cid, p in raw_pred_mon.items():
        raw_metrics[cid] = {"monitor_mae": float(np.abs(p - y_mon).mean())}
    raw_gain = float(np.mean([raw_metrics[f"fit4k_s{s}"]["monitor_mae"]
                              - raw_metrics[f"fit8k_s{s}"]["monitor_mae"] for s in (0, 1)]))

    pred_sha = K.atomic_npz(K.ANALYSIS / "probe_monitor_predictions.npz",
                            **{k: v.astype(np.float32) for k, v in preds.items()},
                            y=y_mon, monitor_rows=rows["monitor"],
                            raw_fit4k_s0=raw_pred_mon["fit4k_s0"].astype(np.float32),
                            raw_fit4k_s1=raw_pred_mon["fit4k_s1"].astype(np.float32),
                            raw_fit8k_s0=raw_pred_mon["fit8k_s0"].astype(np.float32),
                            raw_fit8k_s1=raw_pred_mon["fit8k_s1"].astype(np.float32))

    # ---------- per-molecule csv (single diagnostic table)
    import csv

    keys_best = [f"{c['id']}__best" for c in probe["cells"]]
    header = ["monitor_index", "row_id", "y", "n_atoms", "n_rings", "cycle_rank",
              "max_ring_size", "cycle_excess", "ordinary_nopenalty",
              "unseen_frac_fit4k", "supported_frac_fit4k", "min_support_fit4k",
              "unseen_frac_fit8k", "supported_frac_fit8k", "min_support_fit8k",
              "any_unseen_fit4k", "any_unseen_fit8k", "in_drop_fixed_top10",
              "raw_ae_fit4k_s0", "raw_ae_fit4k_s1", "raw_ae_fit8k_s0", "raw_ae_fit8k_s1"]
    header += [f"pred__{k}" for k in keys_best] + [f"ae__{k}" for k in keys_best]
    with open(K.ANALYSIS / "per_molecule_monitor.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, a in enumerate(annotated):
            row = [a[k] for k in header[:18]]
            row += [float(abs(raw_pred_mon[cid][i] - y_mon[i])) for cid in
                    ("fit4k_s0", "fit4k_s1", "fit8k_s0", "fit8k_s1")]
            row += [float(preds[k][i]) for k in keys_best]
            row += [float(abs(preds[k][i] - y_mon[i])) for k in keys_best]
            w.writerow(row)

    payload = {
        "round": K.ROUND_ID,
        "protocol_sha256": K.sha256_file(K.ROUND / "PROTOCOL.md"),
        "probe_freeze_sha256": K.sha256_file(K.ANALYSIS / "probe_freeze.json"),
        "representation_freeze_sha256": K.sha256_file(K.ANALYSIS / "representation_freeze.json"),
        "e5_export_vs_frozen_trajectory": e5,
        "groups": groups,
        "raw": raw_metrics,
        "raw_gain_fit4k_minus_fit8k_monitor": raw_gain,
        "raw_gain_reference_locked_best_monitor": {
            "fit4k_s0": 0.1212907578186132, "fit4k_s1": 0.12029905317956582,
            "fit8k_s0": 0.08357349264295771, "fit8k_s1": 0.08342303324723616,
            "source": "zinc_dccl_bottleneck_v1 train.json eval_reads.best_monitor.locked_mae",
            "note": "quoted frozen numbers; Locked1k is not loaded in this round",
        },
        "cells": cells,
        "predictions_sha256": pred_sha,
        "monitor_labels_read_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_loaded": False,
        "wall_seconds": float(time.perf_counter() - t_start),
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    K.write_json(out_json, payload)
    K.write_json(K.ANALYSIS / "monitor_groups.json", groups)
    K.ledger_append({"stage": "score", "wall_seconds": payload["wall_seconds"],
                     "n_cells": len(cells), "device": "cpu"})
    K.log(f"raw Monitor MAE 4k={[raw_metrics[f'fit4k_s{s}']['monitor_mae'] for s in (0,1)]} "
          f"8k={[raw_metrics[f'fit8k_s{s}']['monitor_mae'] for s in (0,1)]} gain={raw_gain:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
