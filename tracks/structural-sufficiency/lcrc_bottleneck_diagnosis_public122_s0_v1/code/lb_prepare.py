"""Prepare: subsets, shared norms, caches, fingerprints, protocol manifest (CPU).

Writes only into this track's ``cache/`` and ``analysis/``.  No upstream file is
overwritten.  This is where the label discipline is materialised:

* the nested subsets are recomputed from the published split and hash-verified;
* the structure norm consumes **only official Train structures**;
* the regression target norm consumes **only the smallest fit subset's labels**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import lb_common as C
import lb_data as D


def prepare() -> dict[str, Any]:
    C.CACHE.mkdir(parents=True, exist_ok=True)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"created_utc": C.utcnow(), "round_id": C.ROUND_ID,
                           "datasets": {}, "norms": {}, "subsets": {}}
    for dataset in C.DATASETS:
        meta = D.dataset_meta(dataset)
        fracs = list(C.FIT_N[dataset].keys())
        consistency = D.check_nested_consistency(dataset, fracs)
        out["subsets"][dataset] = {
            "fractions": fracs, "consistency": consistency,
            "detail": {f: {k: v for k, v in D.nested_subset_positions(dataset, f).items()
                           if k in ("n", "positions_sha256", "global_rows_sha256")}
                       for f in fracs},
        }
        snorm = D.structure_norm(dataset)
        tnorm = D.shared_target_norm(dataset)
        out["norms"][dataset] = {
            "structure_norm_sha256": C.sha256_text(C.stable_json(
                {k: np.asarray(v).tolist() for k, v in snorm.items()})),
            "structure_norm_source": "all_official_train_structures",
            "target_norm": tnorm["target_norm"],
        }
        # cache fingerprints (read-only upstream structure caches)
        tgt = D.targets(dataset, include_test=True)
        out["datasets"][dataset] = {
            "task": meta["task"], "split_method": meta["split_method"],
            "sizes": list(meta["sizes"]),
            "rows_sha256": {s: C.row_sha256(tgt["rows"][s]) for s in C.SPLITS},
        }
    # D-MPNN feature cache presence (materialised upstream; consumes no labels)
    feat = {}
    for dataset in C.DATASETS:
        import lcpr_dmpnn as PDN
        p = PDN.featurize_path(dataset)
        feat[dataset] = {"path": C.track_rel(p), "exists": p.exists(),
                         "sha256": C.sha256_file(p) if p.exists() else None}
    out["dmpnn_features"] = feat
    out["label_access"] = {
        "structure_norm_uses": "official Train structures only (label-free)",
        "target_norm_uses": "smallest fit subset labels only",
        "official_valid_test": "never used to fit any normaliser / selector / probe",
        "test_used_for_selection": False,
    }
    C.write_json(C.ANALYSIS / "prepare_manifest.json", out)
    C.log(f"prepare complete: {len(out['datasets'])} datasets")
    return out


def main(argv=None) -> int:
    prepare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
