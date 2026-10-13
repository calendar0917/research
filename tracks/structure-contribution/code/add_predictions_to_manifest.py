"""Register already-written label-blind predictions of an arm into the split's
`prediction_manifest_<tag>.json`.

Why this exists: `stage-b-predict` writes one manifest per call, and the A1 arm
is seed-0-only, so calling it for all Stage-B arms at once aborts on the missing
seed-1 soup after having written all available predictions.  This helper appends
the A1 entries (with file hashes and cross-checked soup hashes) without touching
the frozen driver, keeping `verify_freeze(strict_code=True)` valid for training.

Usage:
    python add_predictions_to_manifest.py --tag B --arm A1 --seeds 0 [--drop]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_bjoc_performance_mechanism_long_v1 as drv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="B", choices=sorted(drv.SPLIT_SALTS))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--drop", action="store_true", help="drop existing entries of this arm first")
    args = ap.parse_args(argv)

    drv.verify_freeze(strict_code=False)
    tag, arm = args.tag, args.arm
    split = drv.read_json(drv.split_paths(tag)["split"])
    manifest_path = drv.PRED_MANIFEST[tag]
    manifest = drv.read_json(manifest_path)
    for key, rec in manifest["entries"].items():
        p = drv.REPO_ROOT / rec["path"]
        if not p.exists() or drv.bp.file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"prediction drift for {key}")
    if args.drop:
        manifest["entries"] = {k: v for k, v in manifest["entries"].items() if f":{arm}:" not in k}
    for seed in args.seeds:
        soup = drv.soup_path(tag, arm, seed)
        train_json = drv.train_json_path(tag, arm, seed)
        if not soup.exists() or not train_json.exists():
            raise SystemExit(f"missing soup/train json for {arm} s{seed}")
        recorded = drv.read_json(train_json)["soup_sha256"]
        actual = drv.bp.file_sha256(soup)
        if recorded != actual:
            raise SystemExit(f"soup hash mismatch for {arm} s{seed}: {recorded[:12]} != {actual[:12]}")
        for zone in ("monitor", "locked"):
            out = drv.pred_path(zone, tag, arm, seed)
            if not out.exists():
                raise SystemExit(f"missing prediction {out}")
            n = len(split[zone + "_rows"])
            drv._read_npz_pred(out, n)  # shape/row/finiteness sanity check
            key = f"{zone}:{arm}:{seed}"
            if key in manifest["entries"]:
                raise SystemExit(f"{key} already registered; pass --drop to replace")
            manifest["entries"][key] = {
                "path": str(out.relative_to(drv.REPO_ROOT)),
                "sha256": drv.bp.file_sha256(out),
                "soup_sha256": actual,
            }
            print(f"[manifest] + {key} -> {out.name}")
    drv.write_json(manifest_path, manifest)
    print(f"[manifest] {manifest_path.name} now has {len(manifest['entries'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
