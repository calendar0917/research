"""P0: build TYPE-S2 / FULL-S2 / NULL-S2 count matrices for Train and Validation.

Only the official Train and Validation splits are materialised.  The frozen
Fast v1 cache is read-only.  Labels are never passed to the feature builder.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pc_common as C

GUARD = C.install_test_access_guard()

import pc_data as D  # noqa: E402
import pc_s2 as S  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(C.ANALYSIS))
    parser.add_argument("--limit", type=int, default=None, help="engineering smoke only")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    manifest = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "generated_utc": C.utcnow(),
        "entry_point": "code/pc_p0_features.py",
        "limit": args.limit,
        "splits": {},
        "test_access": {"attempts": GUARD["attempts"], "strict": GUARD["strict"]},
        "labels_used": False,
        "environment": C.environment(),
    }
    for split in C.ALLOWED_SPLITS:
        C.log(f"building features for {split}")
        stats = S.build_split_features(split, limit=args.limit, verbose=True)
        npz_path = out / f"features_{split}.npz"
        json_path = out / f"features_{split}.json"
        stats = S.save_split_features(split, stats, npz_path, json_path)
        raw_path = out / f"raw_{split}.npz"
        raw = D.export_raw_arrays(split, raw_path)
        manifest["splits"][split] = {
            "features_npz": stats["npz_path"],
            "features_npz_sha256": stats["npz_sha256"],
            "features_npz_bytes": stats["npz_bytes"],
            "features_json": stats["json_path"],
            "features_json_sha256": stats["json_sha256"],
            "raw_npz": raw["path"],
            "raw_npz_sha256": raw["sha256"],
            "raw_n_graphs": raw["n_graphs"],
            "raw_n_atoms": raw["n_atoms"],
            "raw_n_bonds": raw["n_bonds"],
            "n_graphs": stats["n_graphs"],
            "n_paths_total": stats["n_paths_total"],
            "n_paths_per_graph": stats["n_paths_per_graph"],
            "global": {kind: stats["features"][kind]["global"] for kind in ("type", "full", "null")},
            "full_binary": stats["full_binary"],
            "null_audit": stats["null_audit"],
            "counts_type_sha256": stats["counts_type_sha256"],
            "counts_full_sha256": stats["counts_full_sha256"],
            "counts_null_sha256": stats["counts_null_sha256"],
            "seconds": stats["seconds"],
        }
        C.log(
            f"{split}: {stats['n_graphs']} graphs / {stats['n_paths_total']} paths; "
            f"unique type/full/null = "
            f"{stats['features']['type']['global']['n_unique_patterns']}/"
            f"{stats['features']['full']['global']['n_unique_patterns']}/"
            f"{stats['features']['null']['global']['n_unique_patterns']}"
        )
    manifest["total_seconds"] = float(time.perf_counter() - t_start)
    manifest["test_access"]["attempts"] = list(GUARD["attempts"])
    C.write_json(out / "p0_features_manifest.json", manifest)
    C.log(f"P0 feature build complete in {manifest['total_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
