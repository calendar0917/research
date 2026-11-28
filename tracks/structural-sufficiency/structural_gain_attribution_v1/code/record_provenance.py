"""Provenance ledger: data identity, code, environment, execution regime, budget.

Reads the frozen artefacts, the run JSONs and ``analysis/remote_jobs.json`` (the
``rr status --json`` snapshot of every remote job) and writes
``analysis/provenance.json``.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
from typing import Any

import numpy as np

import sga_common as K

K.add_prev_code_path()

GPU_POOLS = ("res-gpu0", "res-gpu1")


def main() -> None:
    K.ensure_dirs()
    jobs = json.loads((K.ANALYSIS / "remote_jobs.json").read_text())
    for j in jobs:
        for key in ("started_at", "completed_at"):
            if j.get(key):
                j[key] = dt.datetime.fromisoformat(j[key]).timestamp()
    gpu_seconds = 0.0
    for j in jobs:
        if j.get("pool") in GPU_POOLS and j.get("started_at") and j.get("completed_at"):
            gpu_seconds += max(0.0, j["completed_at"] - j["started_at"])

    runs = {}
    for path in sorted(glob.glob(str(K.RUNS / "phase*" / "*.json"))):
        if path.endswith("_model.json"):
            continue
        payload = json.loads(open(path).read())
        arm = payload.get("arm", path.split("/")[-1])
        runs[arm] = {
            "phase": payload.get("phase"),
            "arm": arm,
            "extra_blocks": payload.get("extra_blocks"),
            "recipe": payload.get("recipe"),
            "metrics_recomputed_from": f"analysis/predictions/{arm}.npz",
            "seconds_on_device": payload.get("seconds"),
            "best_epoch": payload.get("best_epoch"),
            "device": payload.get("device"),
            "shared_init_hash": payload.get("shared_init_hash"),
            "parameters": payload.get("parameters"),
            "recorded_metrics": {k: v.get("roc_auc", v.get("mae")) for k, v in
                                 (payload.get("metrics") or {}).items()},
        }

    payload = {
        "round_id": K.ROUND_ID,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {
            "code_revision_used_by_all_remote_runs": "8b68632dbd346dd3b1f5b10e1b3603cc3d6f2036",
            "note": "every rr job was launched from that committed revision (rr refuses dirty "
                    "revisions); the analysis/report files in this directory were produced "
                    "locally afterwards, from the pulled per-molecule predictions",
            "working_tree_at_generation": K.git_state(),
        },
        "environment_local": K.environment(),
        "data_identity": {
            "molhiv": {
                "source": "data/ogb/ogbg_molhiv (official scaffold split)",
                "splits": K.split_fingerprint(K.molhiv_splits()),
            },
            "zinc": json.loads((K.CACHE / "zinc_design_meta.json").read_text())["split"]
            | {"source": "PyG ZINC(subset=True) processed/train|val|test.pt"},
            "input_bundle": {
                "path": K.rel(K.INPUT_BUNDLE),
                "sha256": K.sha256_file(K.INPUT_BUNDLE),
                "size_bytes": int(K.INPUT_BUNDLE.stat().st_size),
                "meta_sha256": K.sha256_file(K.INPUT_BUNDLE_META),
            },
            "frozen_predecessor_assets": {
                "phaseA_instances_sha256": K.sha256_file(K.MSDS_DATA / "phaseA_instances.npz"),
                "design_CF_sha256": K.sha256_file(K.MSDS_DATA / "design_CF.npz"),
                "vocab_sha256": K.sha256_file(K.MSDS_DATA / "vocab.npz"),
                "phaseD_xgb_predictions_sha256": K.sha256_file(
                    K.MSDS_DATA / "phaseD_xgb_struct_predictions.npz"),
                "historical_features_sha256": K.sha256_file(K.HIST_FEATURES),
                "historical_manifest_sha256": K.sha256_file(K.HIST_MANIFEST),
            },
            "zinc_design": {
                "path": K.rel(K.CACHE / "zinc_design.npz"),
                "sha256": K.sha256_file(K.CACHE / "zinc_design.npz"),
            },
        },
        "execution": {
            "jobs": jobs,
            "n_jobs": len(jobs),
            "gpu_seconds_total": gpu_seconds,
            "gpu_hours_total": gpu_seconds / 3600.0,
            "regime": "res / a100-2; pools res-local (CPU), res-gpu0 + res-gpu1 (1 A100 each); "
                      "driver 550.163.01; torch 2.5.1+cu124",
            "note": "every job ran on the same host+dependency regime, so all numbers in this "
                    "round are mutually comparable",
        },
        "budget": {
            "new_xgboost_models": 8,
            "new_mlp_models": 5,
            "reused_models_not_refitted": ["X0 (historical XGBoost)", "X-CF (MSDS-v0 XGB_STRUCT)"],
            "input_caches_built_once": ["bundle/sav1_inputs.npz", "data/zinc_design.npz"],
            "caps": {"wall_clock_hours": 4, "gpu_hours": 4, "new_xgb": 8, "new_mlp": 5},
            "gpu_hours_used": gpu_seconds / 3600.0,
        },
        "runs": runs,
        "engineering_deviations": [
            {
                "what": "`rr run ... -- python -m <module>` could not import the round's modules",
                "cause": "the remote job runs from the repository root, so the round code dir is "
                         "not on sys.path",
                "fix": "commands run as `env PYTHONPATH=<round>/code python -m <module>`",
                "affected_results": "none (the affected launches failed before writing anything)",
            },
            {
                "what": "remote checkout had an untracked MSDS-v0 Phase-E directory that blocked deploy",
                "cause": "MSDS-v0 wrote Phase-E outputs into the remote checkout",
                "fix": "moved to /home/hxy/sga1_orphans/ (preserved, never deleted); the same files "
                       "already exist locally under .rr/pulled/phaseE-*",
                "affected_results": "none",
            },
            {
                "what": "ZINC FULL->COARSE projection verification was initially a topology comparison",
                "cause": "FULL and COARSE canonical forms use different node orderings, so comparing "
                         "edge index lists is not a valid projection test",
                "fix": "replaced by the predecessor's conflict check: a FULL key may only ever map to "
                       "one projected COARSE key (measured conflicts = 0)",
                "affected_results": "none (the model features were already built from the projected key)",
            },
            {
                "what": "ZINC type identity is keyed by a 64-bit digest of the canonical key",
                "cause": "worker payload size (2.4 M instances) and deterministic tie-breaking",
                "fix": "documented in the ZINC prep meta; the digest is a projection of the complete "
                       "invariant, used as a dictionary key and tie-break only",
                "affected_results": "negligible hash-collision probability (~1e-8 for 2.3 k types); "
                                    "no observed anomaly",
            },
        ],
    }
    K.write_json(K.ANALYSIS / "provenance.json", payload)
    print(f"[provenance] jobs={len(jobs)} gpu_hours={gpu_seconds / 3600:.3f} "
          f"runs={len(runs)} bundle={K.sha256_file(K.INPUT_BUNDLE)[:12]}", flush=True)


if __name__ == "__main__":
    main()
