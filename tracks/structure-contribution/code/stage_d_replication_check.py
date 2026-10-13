"""Stage D — second-split replication check for protocol
`cscl-bjoc-performance-mechanism-long-v1`.

Scores only the pre-registered P0/P3 arms on a new group-aware split (default
`C`) against its frozen monitor block.  The monitor labels are read exactly once
under the same lock discipline as Stage B; predictions must match the frozen
`prediction_manifest_<tag>.json` hashes.

Purchase rule (frozen in the config): Stage D runs iff the Stage-B recipe gate
passed.  This script only *scores* what was already trained; it never starts
training.  It is intentionally a separate file so the frozen driver hash used by
`verify_freeze(strict_code=True)` does not drift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_bjoc_performance_mechanism_long_v1 as drv  # noqa: E402


def score(tag: str, arms: tuple[str, ...], label: str) -> int:
    drv.verify_freeze(strict_code=False)
    paths = drv.split_paths(tag)
    split = drv.read_json(paths["split"])
    manifest = drv._verify_prediction_manifest(tag)
    for arm in arms:
        for seed in drv.SEEDS:
            key = f"monitor:{arm}:{seed}"
            if key not in manifest["entries"]:
                raise SystemExit(f"missing frozen monitor prediction {key}")
    y = drv._read_monitor_y_once(tag, split["monitor_rows"])
    preds = {
        arm: {s: drv._read_npz_pred(drv.pred_path("monitor", tag, arm, s), len(y)) for s in drv.SEEDS}
        for arm in arms
    }
    contrasts = {}
    if "P3" in preds and "P0" in preds:
        contrasts["recipe_P3_minus_P0"] = drv._contrast(preds["P3"], preds["P0"], y, "P3-P0")
    if "B3" in preds and "B0" in preds:
        contrasts["recipe_scratch_B3_minus_B0"] = drv._contrast(preds["B3"], preds["B0"], y, "B3-B0")
    recipe = contrasts.get("recipe_P3_minus_P0")
    gate = bool(
        recipe is not None
        and recipe["delta_seed_mean"] <= -0.003
        and all(v < 0 for v in recipe["per_seed_delta"].values())
    )
    payload = {
        "protocol": drv.PROTOCOL_ID,
        "stage": "D",
        "label": label,
        "git_commit": drv.git_commit(),
        "split_tag": tag,
        "arms": list(arms),
        "contrasts": contrasts,
        "recipe_gate_pass": gate,
        "decision": "RECIPE_GAIN_REPLICATES_ON_SECOND_SPLIT" if gate else "RECIPE_GAIN_NOT_ROBUST_ACROSS_SPLITS",
        "official_test_loaded": False,
    }
    drv.write_json(drv.OUT / f"stage_d_monitor_{tag}.json", payload)
    drv.log(f"[stage-d {tag}] decision {payload['decision']}")
    for key, c in contrasts.items():
        drv.log(
            f"  {key:24s} delta {c['delta_seed_mean']:+.5f} "
            f"CI95 [{c['bootstrap']['ci_lo']:+.5f},{c['bootstrap']['ci_hi']:+.5f}] "
            f"per-seed {c['per_seed_delta']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="C", choices=sorted(drv.SPLIT_SALTS))
    ap.add_argument("--arms", nargs="*", default=["P0", "P3"])
    ap.add_argument("--label", default="stage-d")
    args = ap.parse_args(argv)
    return score(args.tag, tuple(args.arms), args.label)


if __name__ == "__main__":
    raise SystemExit(main())
