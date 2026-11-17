"""Runner ``lcrc-auto-subgraph-dose-localization-audit-v1``.

Stages
------
``preflight``  data / sample / checkpoint / Top-1-source identity plus a full-path
              smoke audit of the first 8 selected Valid molecules (never aggregated);
``audit``      the formal 3 x 64-molecule corrected audit for one dataset (CPU by
              default; a GPU is used only if one is visible, and only for the
              frozen forward pass);
``aggregate``  CPU-only deterministic tables, bootstrap, verdicts and
              REPORT.md / DECISION.md from the per-molecule records.

Only engineering knobs are overridable (stage, dataset, device, limit).  Every
scientific constant is frozen in :mod:`au_common` and hashed into
``analysis/protocol_freeze.json``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import au_common as C

DEFAULT_CONFIG = C.ROUND_DIR / "configs" / "experiment.yaml"


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name=C.ROUND_ID,
        default_config=DEFAULT_CONFIG,
        default_study=C.STUDY_ID,
        description=(
            "Read-only audit of lcrc-auto-subgraph-faithfulness-pilot-v1: correct S2 dose "
            "accounting (n_s2_affected, not n_s2_internal), affected-matched random null, "
            "and root-node localisation of the masked response of the frozen LCRC-C1 "
            "(no training, no re-selection, single reused checkpoint)"
        ),
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = {"round_id": C.ROUND_ID, "freeze_hash": C.freeze_hash()}
    for dataset in C.DATASETS:
        cache = C.SQL_DATA / C.SD.cache_filename(dataset)
        raw = C.SQL_DATA / C.SD.lipo_raw_filename(dataset)
        splits = {}
        for split in C.SPLITS:
            with np.load(cache, allow_pickle=False) as mesh:
                idx = np.asarray(mesh[C.SC.SPLIT_FILE[split]], dtype=np.int64)
            splits[split] = {"n": int(idx.size), "sha256": C.sha256_array(idx)}
        paths = C.upstream_paths(dataset)
        payload[dataset] = {
            "graph_cache_sha256": C.sha256_file(cache) if cache.is_file() else "missing",
            "lipo_raw_sha256": C.sha256_file(raw) if raw.is_file() else "missing",
            "splits": splits,
            "checkpoint_sha256": C.EXPECTED_CHECKPOINT_SHA256[dataset],
            "pilot_records_sha256": {s: C.sha256_file(paths[s]) for s in C.SPLITS},
            "pilot_samples_sha256": C.sha256_file(paths["samples"]),
        }
    split_fingerprint = C.sha256_json({d: payload[d]["splits"] for d in C.DATASETS})
    return payload, split_fingerprint


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(config: Mapping[str, Any], context) -> Any:
    from ksvd_research.runtime.manifest import RunResult

    if context.test_access != "granted":
        raise RuntimeError(
            "this round needs official Test for its bounded corrected comparison; "
            "test_policy=always was not granted"
        )
    stage = str(config.get("stage", "audit"))
    ledger_dir = Path(context.artifact_dir)
    C.ensure_dir(ledger_dir)
    t0 = time.perf_counter()
    if stage in ("preflight", "audit"):
        import au_worker as W

        datasets = _datasets(config)
        if stage == "audit" and len(datasets) != 1:
            raise RuntimeError("the audit stage runs exactly one dataset per worker")
        summary: dict[str, Any] = {"stage": stage, "datasets": {}}
        for dataset in datasets:
            result = W.run_dataset(
                dataset,
                stage="preflight" if stage == "preflight" else "audit",
                limit=config.get("limit"),
            )
            summary["datasets"][dataset] = result
            if stage == "preflight" and not result["pass"]:
                raise RuntimeError(f"{dataset}: preflight failed -> {result['failures']}")
        metrics = summary
    elif stage == "aggregate":
        import au_aggregate

        metrics = au_aggregate.run(datasets=list(C.DATASETS))
        import au_report

        metrics["reports"] = au_report.write_reports()
        for key in ("detail", "localization_detail", "old_vs_corrected_rows",
                    "localization_rows", "preflight", "decision", "manifest", "summary"):
            metrics.pop(key, None)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    metrics = {**metrics, "wall_seconds": time.perf_counter() - t0, "stage": stage}
    C.atomic_write_json(ledger_dir / f"{stage}_summary.json", C.jsonable(metrics))
    return RunResult(
        metrics=C.jsonable(metrics),
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
        raise ValueError(f"unknown dataset(s) {bad}; allowed {list(C.DATASETS)} or 'all'")
    return names


__all__ = ["build_runner", "fingerprints", "run", "DEFAULT_CONFIG"]
