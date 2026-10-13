"""Seed-aware locked freeze + final locked scoring for protocol
`cscl-bjoc-performance-mechanism-long-v1`.

The frozen driver's `locked-freeze`/`locked-score` assume every arm exists for
both seeds; the Stage-E arm A1 is seed-0 only by design, so this standalone
script mirrors those two steps while taking the available seeds per arm from the
frozen prediction manifest.  Everything else is identical: the same manifest
hash checks, the same one-shot locked label read (lock + saved y), the same
metric and contrast definitions from `bjoc_perfmech_long_v1`/the driver.

It is a separate file so the frozen driver hash stays valid.

Usage:
  python locked_finalize.py --tag B --mode freeze
  python locked_finalize.py --tag B --mode score     # reads locked labels once
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_bjoc_performance_mechanism_long_v1 as drv  # noqa: E402


def _available(manifest: dict, arm: str) -> list[int]:
    seeds = []
    for seed in drv.SEEDS:
        if f"locked:{arm}:{seed}" in manifest["entries"]:
            seeds.append(int(seed))
    return seeds


def _arm_plan(tag: str) -> list[tuple[str, list[int]]]:
    manifest = drv._verify_prediction_manifest(tag)
    arms = ["B0", "B3", "P0", "P3"]
    c1 = drv.OUT / f"stage_c_monitor_{tag}.json"
    if c1.exists() and drv.read_json(c1)["decision"] == "C1_MONITOR_GAIN_SUPPORTED":
        arms.append("C1")
    a1 = drv.OUT / f"stage_e_monitor_{tag}.json"
    if a1.exists():
        arms.append("A1")
    plan = [(arm, _available(manifest, arm)) for arm in arms]
    for arm, seeds in plan:
        if not seeds:
            raise SystemExit(f"no locked predictions for arm {arm}")
    return plan


def cmd_freeze(tag: str) -> int:
    manifest = drv._verify_prediction_manifest(tag)
    plan = _arm_plan(tag)
    entries = {}
    for arm, seeds in plan:
        for seed in seeds:
            key = f"locked:{arm}:{seed}"
            entries[key] = manifest["entries"][key]
    payload = {
        "protocol": drv.PROTOCOL_ID,
        "git_commit": drv.git_commit(),
        "split_tag": tag,
        "arms": [arm for arm, _ in plan],
        "seeds_per_arm": {arm: seeds for arm, seeds in plan},
        "entries": entries,
        "monitor_decision_sha256": drv.bp.file_sha256(drv.OUT / f"stage_b_monitor_{tag}.json"),
        "frozen_by": "locked_finalize.py",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    drv.write_json(drv.OUT / f"locked_manifest_{tag}.json", payload)
    drv.log(f"[locked-freeze {tag}] arms {payload['arms']} seeds {payload['seeds_per_arm']}")
    return 0


def cmd_score(tag: str) -> int:
    locked_manifest = drv.read_json(drv.OUT / f"locked_manifest_{tag}.json")
    for key, rec in locked_manifest["entries"].items():
        p = drv.REPO_ROOT / rec["path"]
        if not p.exists() or drv.bp.file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"locked prediction drift for {key}")
    split = drv.read_json(drv.split_paths(tag)["split"])
    y = drv._read_locked_y_once(tag, split["locked_rows"])
    seeds_per_arm = locked_manifest["seeds_per_arm"]
    preds: dict[str, dict[int, np.ndarray]] = {}
    for arm, seeds in seeds_per_arm.items():
        preds[arm] = {int(s): drv._read_npz_pred(drv.pred_path("locked", tag, arm, int(s)), len(y)) for s in seeds}
    metrics = {}
    for arm, seeds in preds.items():
        metrics[arm] = {"per_seed_mae": {str(s): float(drv._err(preds[arm][s], y).mean()) for s in seeds}}
        m = np.mean([drv._err(preds[arm][s], y) for s in seeds], axis=0)
        metrics[arm]["seed_mean_mae"] = float(m.mean())
        metrics[arm]["median_abs"] = float(np.median(m))
        metrics[arm]["p90_abs"] = float(np.quantile(m, 0.90))
        metrics[arm]["rmse"] = float(np.sqrt(np.mean(np.mean([(preds[arm][s] - y) ** 2 for s in seeds], axis=0))))
    contrasts = {}
    for new_arm, ref_arm in (("P3", "P0"), ("B3", "B0"), ("P3", "B3"), ("P0", "B0"), ("C1", "P3"), ("A1", "P3")):
        if new_arm in preds and ref_arm in preds:
            common = sorted(set(preds[new_arm]) & set(preds[ref_arm]))
            if common:
                contrasts[f"{new_arm}_minus_{ref_arm}"] = drv._contrast(
                    {s: preds[new_arm][s] for s in common}, {s: preds[ref_arm][s] for s in common}, y, f"{new_arm}-{ref_arm}"
                )
    payload = {
        "protocol": drv.PROTOCOL_ID,
        "git_commit": drv.git_commit(),
        "split_tag": tag,
        "metrics": metrics,
        "contrasts": contrasts,
        "scored_by": "locked_finalize.py",
        "official_test_loaded": False,
    }
    drv.write_json(drv.OUT / f"locked_final_{tag}.json", payload)
    drv.log(f"[locked-score {tag}] done")
    for arm in preds:
        drv.log(
            f"  {arm}: seed-mean locked MAE {metrics[arm]['seed_mean_mae']:.6f} "
            f"per-seed {metrics[arm]['per_seed_mae']} rmse {metrics[arm]['rmse']:.6f}"
        )
    for key, c in contrasts.items():
        drv.log(
            f"  {key:12s} delta {c['delta_seed_mean']:+.5f} CI95 "
            f"[{c['bootstrap']['ci_lo']:+.5f},{c['bootstrap']['ci_hi']:+.5f}] per-seed {c['per_seed_delta']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="B", choices=sorted(drv.SPLIT_SALTS))
    ap.add_argument("--mode", choices=["freeze", "score"], required=True)
    args = ap.parse_args(argv)
    drv.verify_freeze(strict_code=False)
    return cmd_freeze(args.tag) if args.mode == "freeze" else cmd_score(args.tag)


if __name__ == "__main__":
    raise SystemExit(main())
