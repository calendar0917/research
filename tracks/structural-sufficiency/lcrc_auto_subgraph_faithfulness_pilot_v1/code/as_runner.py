"""Runner ``lcrc-auto-subgraph-faithfulness-pilot-v1``.

Stages
------
``preflight``      data + checkpoint identity and the full ENGINEERING_GATE for
                   one dataset (or both), no explanation computed;
``smoke``          the first 8 selected Valid molecules of a dataset, proving the
                   whole discovery + intervention path;
``dataset_worker`` the formal 3 x 64-molecule run for one dataset on one GPU;
``aggregate``      CPU-only deterministic tables, bootstrap, figures, verdicts
                   and REPORT.md / DECISION.md from the pulled artefacts.

Only engineering knobs are overridable (stage, dataset, limit).  Every
scientific constant is frozen in :mod:`as_common` and hashed into
``analysis/protocol_freeze.json``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import as_common as C

DEFAULT_CONFIG = C.ROUND_DIR / "configs" / "experiment.yaml"


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name=C.ROUND_ID,
        default_config=DEFAULT_CONFIG,
        default_study=C.STUDY_ID,
        description=(
            "Automatic connected-subgraph explanation of the frozen LCRC-C1 (arm A) on "
            "official ogbg-molesol / ogbg-mollipo: gradient discovery + real "
            "CONDITIONAL_S1_S2_EVIDENCE_MASK forward response + matched random controls "
            "(test_policy=always, single reused checkpoint)"
        ),
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    import cr_data as SD

    payload: dict[str, Any] = {"round_id": C.ROUND_ID, "freeze_hash": C.freeze_hash()}
    for dataset in C.DATASETS:
        cache = C.SQL_DATA / SD.cache_filename(dataset)
        raw = C.SQL_DATA / SD.lipo_raw_filename(dataset)
        splits = {}
        for split in C.SPLITS:
            mesh = np.load(cache, allow_pickle=False)
            try:
                idx = np.asarray(mesh[C.SC.SPLIT_FILE[split]], dtype=np.int64)
            finally:
                mesh.close()
            splits[split] = {"n": int(idx.size), "sha256": C.sha256_array(idx)}
        payload[dataset] = {
            "graph_cache_sha256": C.sha256_file(cache) if cache.is_file() else "missing",
            "lipo_raw_sha256": C.sha256_file(raw) if raw.is_file() else "missing",
            "splits": splits,
            "checkpoint_sha256": C.EXPECTED_CHECKPOINT_SHA256[dataset],
        }
    split_fingerprint = C.sha256_json(
        {d: payload[d]["splits"] for d in C.DATASETS if isinstance(payload.get(d), dict)}
    )
    return payload, split_fingerprint


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(config: Mapping[str, Any], context) -> Any:
    from ksvd_research.runtime.manifest import RunResult

    if context.test_access != "granted":
        raise RuntimeError(
            "this round requires test_policy=always; the control plane blocked official test access"
        )
    stage = str(config.get("stage", "dataset_worker"))
    ledger_dir = Path(context.artifact_dir)
    t0 = time.perf_counter()
    if stage == "preflight":
        import as_worker as W

        datasets = _datasets(config)
        summary = {"datasets": {}}
        for dataset in datasets:
            pf = W.preflight_dataset(dataset)
            out_dir = C.RESULTS_DIR / dataset
            C.ensure_dir(out_dir)
            C.atomic_write_json(out_dir / "preflight.json", pf)
            C.atomic_write_json(out_dir / "training_source.json", pf["training_source"])
            summary["datasets"][dataset] = {
                "engineering_gate": pf["engineering_gate"],
                "failed_gates": [g["gate"] for g in pf["gates"] if not g["passed"]],
            }
        metrics = summary
    elif stage == "smoke":
        import as_worker as W

        dataset = _one_dataset(config)
        metrics = W.run_dataset(dataset, stage="smoke", limit=config.get("limit"))
    elif stage == "dataset_worker":
        import as_worker as W

        dataset = _one_dataset(config)
        metrics = W.run_dataset(dataset, stage="dataset_worker", limit=config.get("limit"))
    elif stage == "aggregate":
        import as_aggregate

        metrics = as_aggregate.run(datasets=list(C.DATASETS))
        import as_report

        metrics["reports"] = as_report.write_reports()
        # aggregate re-reads every per-molecule record; keep the ledger small
        metrics.pop("rows", None)
        metrics.pop("main", None)
        metrics.pop("diag", None)
        metrics.pop("manifest", None)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    metrics = {**metrics, "stage": stage, "wall_seconds": time.perf_counter() - t0}
    C.atomic_write_json(ledger_dir / f"{stage}_summary.json", metrics)
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=[f"artifacts/{stage}_summary.json"],
    )


def _datasets(config: Mapping[str, Any]) -> list[str]:
    raw = config.get("dataset", "all")
    names = [str(x) for x in raw] if isinstance(raw, (list, tuple)) else [str(raw)]
    if "all" in names:
        return list(C.DATASETS)
    bad = [n for n in names if n not in C.DATASETS]
    if bad:
        raise ValueError(f"unknown dataset(s) {bad}; allowed {C.DATASETS} or 'all'")
    return names


def _one_dataset(config: Mapping[str, Any]) -> str:
    datasets = _datasets(config)
    if len(datasets) != 1:
        raise ValueError("this stage runs exactly one dataset per worker")
    return datasets[0]


__all__ = ["build_runner", "fingerprints", "run", "DEFAULT_CONFIG"]
