"""Prepare the four TU datasets: raw manifest, frozen split, caches, Train norm.

Run once before the 12 formal trainings.  It never trains and never reads a Test
label for fitting: the standardisation is computed from the Train cache only and
is audited against the Train means.
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import tu_common as C
import tu_data as D


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(sorted(C.DATASETS)))
    parser.add_argument("--force", action="store_true", help="rebuild caches/specs from raw files")
    parser.add_argument("--skip-gates", action="store_true")
    args = parser.parse_args(argv)
    datasets = [d for d in args.datasets.split(",") if d]
    prepared = {}
    for ds in datasets:
        C.log(f"preparing {ds}")
        prepared[ds] = D.prepare_dataset(ds, force=bool(args.force))
        p = prepared[ds]
        C.log(
            f"{ds}: {p['dataset_stats']['n_graphs_loaded']} graphs, "
            f"sizes {[p['split'][s]['n'] for s in ('train', 'valid', 'test')]}, "
            f"node types {p['n_node_types']}, atom_feat {p['atom_feat_dim']}, "
            f"S2 paths {sum(p['caches'][s]['n_s2_paths'] for s in ('train', 'valid', 'test'))}, "
            f"cache seconds "
            f"{sum(p['local_cache_files'][s]['build_seconds'] for s in ('train', 'valid', 'test')):.1f}"
        )
    D.write_split_file(prepared)
    D.write_round_manifest(prepared)
    host = os.uname().nodename if hasattr(os, "uname") else "unknown"
    D.write_local_prep_report(prepared, host)
    summary = {
        ds: {
            "n_graphs": p["dataset_stats"]["n_graphs_loaded"],
            "n_train": p["split"]["train"]["n"],
            "n_valid": p["split"]["valid"]["n"],
            "n_test": p["split"]["test"]["n"],
            "class_counts_train": p["split"]["train"]["class_counts"],
            "feature_spec_hash": p["feature_spec_hash"],
            "atom_feat_dim": p["atom_feat_dim"],
            "n_node_types": p["n_node_types"],
            "node_type_mode": p["node_type_mode"],
            "norm_fingerprint": p["norm_fingerprint"],
            "partition_sha256": p["split"]["partition_sha256"],
            "cell_cleaning": p["dataset_stats"]["clean_totals"],
        }
        for ds, p in prepared.items()
    }
    C.write_json(C.ANALYSIS / "prep_summary.json", summary)
    print(C.stable_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
