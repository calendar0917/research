"""One-shot preparation: splits, structure caches, fingerprints, pair plans.

Run once per host (CPU only, no GPU is charged).  Everything it writes is a pure
function of the frozen inputs and the frozen salts.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import lr_common as C
import lr_data as D
import lr_pairs as P
import lr_split as S


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="prepare caches, fingerprints, splits and pair plans")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-plans", action="store_true")
    args = ap.parse_args(argv)

    if args.rebuild or not all(D.SPLIT_FILES[s].exists() for s in ("s0", "s1", "s2", "sg", "ft")):
        C.log("[prepare] building split payloads")
        S.main([])
    manifest: dict = {"schema": "lcrc-lipo-route-prepare-v1", "created_utc": C.utcnow(),
                      "git_commit": C.git_commit()}
    C.log("[prepare] structure caches")
    manifest["caches"] = D.build_all(rebuild=args.rebuild)
    C.log("[prepare] fingerprints")
    manifest["fingerprints"] = D.build_fingerprint_caches(rebuild=args.rebuild)
    if not args.skip_plans:
        C.log("[prepare] pair plans")
        plans = {}
        for split in C.SPLITS + ("ft",):
            steps = C.FT_UPDATES if split == "ft" else C.FORMAL_UPDATES
            for seed in C.SEEDS:
                plan = P.get_plan(split, seed, steps, rebuild=args.rebuild)
                plans[f"{split}_s{seed}"] = {
                    "hash": plan["hash"], "n_steps": int(plan["n_steps"]),
                    "mean_non_mmp_fraction": plan["mean_non_mmp_fraction"],
                    "min_non_mmp_fraction": plan["min_non_mmp_fraction"],
                    "n_shortfall_steps": plan["n_shortfall_steps"],
                    "max_redraws": int(plan["max_redraws"]),
                }
        manifest["pair_plans"] = plans
    manifest["splits"] = {
        s: {"status": D.split_payload(s)["status"],
            "sha256": C.sha256_file(D.SPLIT_FILES[s]),
            "partitions": {p: {k: v for k, v in D.split_payload(s)["partitions"][p].items()
                               if k in ("n_rows", "n_pairs", "n_pair_molecules", "n_pair_components", "n_classes")}
                           for p in C.PARTITIONS}}
        for s in C.SPLITS + ("ft",)
    }
    C.write_json(C.ANALYSIS / "prepare_manifest.json", manifest)
    C.log("[prepare] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
