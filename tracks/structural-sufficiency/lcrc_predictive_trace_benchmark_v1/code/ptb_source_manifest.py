"""Write ``analysis/source_manifest.json``: every input this round read, with hashes.

Covers the dataset sources, the frozen upstream rounds that are cited or reused,
the feature spec, and the frozen structure caches.  Nothing here trains or scores;
it is the provenance sheet for ``REPORT.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import ptb_common as C
import ptb_data as D

UPSTREAM_FILES = [
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/runs/Lipophilicity_s0/checkpoint.pt",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/literature_matrix.csv",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/analysis/tables/variant_means.csv",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_followup_v1/analysis/corrected_tables/corrected_contrasts_valid.json",
    "tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/analysis/validation_table.json",
    "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2/analysis/cv_results_v2.json",
    "tracks/structural-sufficiency/zinc_dccl_bottleneck_v1/analysis/decision.json",
    "tracks/structural-sufficiency/zinc_error_mechanism_audit_v1/analysis/audit.json",
    "tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/code/tu_model.py",
    "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/PROTOCOL.md",
    "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/analysis/probe_freeze.json",
    "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/analysis/probe_report.json",
]


def _dataset_entry(view) -> dict[str, Any]:
    audit = view.audit()
    return {
        "task": view.task,
        "label_columns": list(view.label_columns),
        "units": view.units,
        "n_train": audit["n_train"],
        "n_valid": audit["n_valid"],
        "train_cache_fingerprint": audit["train_cache_fingerprint"],
        "valid_cache_fingerprint": audit["valid_cache_fingerprint"],
        "train_norm_fingerprint": audit["train_norm_fingerprint"],
        "rows": audit["rows"],
        "source": audit["source"],
        "labels_in_structure_cache": audit["labels_in_structure_cache"],
    }


def run(datasets: Sequence[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "lcrc-ptb-source-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "code_hashes": C.code_hashes(),
        "reused_module_hashes": C.reused_reused() if hasattr(C, "reused_reused") else C.reused_module_hashes(),
        "feature_spec_hash": None,
        "datasets": {},
        "upstream_files": {},
        "test_split_policy": (
            "Test structure caches and Test labels are opened only by code/ptb_score.py "
            "after analysis/checkpoint_manifest.json freezes the units; the TU outer folds "
            "are scored once per unit inside the training script after the checkpoint hash is written"
        ),
    }
    for dataset in datasets:
        view = D.molecule_view(dataset)
        payload["feature_spec_hash"] = view.spec.hash()
        entry = _dataset_entry(view)
        entry["epochs"] = int(C.MOLECULE_DATASETS[dataset]["epochs"])
        entry["primary_metric"] = C.MOLECULE_DATASETS[dataset]["primary_metric"]
        payload["datasets"][dataset] = entry
    for path in UPSTREAM_FILES:
        p = C.REPO / path
        payload["upstream_files"][path] = (
            {"sha256": C.sha256_file(p), "bytes": int(p.stat().st_size)} if p.exists() else {"missing": True}
        )
    tu_meta = {}
    for dataset in C.TU_DATASETS:
        entry = dict(D._tu_meta(dataset)) if hasattr(D, "_tu_meta") else {}
        tu_meta[dataset] = {
            "meta": entry,
            "epochs": int(C.TU_EPOCHS),
            "model_names": C.TU_MODEL_NAME,
        }
    payload["tu_datasets"] = tu_meta
    payload["cache_policy"] = (
        "structure caches are rebuildable and gitignored; their content fingerprints are "
        "recorded above so a rebuild can be checked"
    )
    C.write_json(C.ANALYSIS / "source_manifest.json", payload)
    C.log(f"source manifest written for {len(payload['datasets'])} molecule and {len(tu_meta)} TU datasets")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(C.MOLECULE_ORDER))
    args = ap.parse_args(argv)
    run(args.datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
