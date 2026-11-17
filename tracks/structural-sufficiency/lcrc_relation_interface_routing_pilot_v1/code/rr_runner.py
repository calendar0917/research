"""Runner ``lcrc-relation-interface-routing-pilot-v1``.

One run = one stage.  The four formal training units are

    --set stage=train --set task=lipo    --set model=R1   (pool res-gpu0)
    --set stage=train --set task=wc_logp --set model=R1   (pool res-gpu0)
    --set stage=train --set task=lipo    --set model=R2   (pool res-gpu1)
    --set stage=train --set task=wc_logp --set model=R2   (pool res-gpu1)

Stages:

* ``prepare``  – CPU only: freeze the topology partition, build the three routing
  caches, print their SHA-256, write the support statistics, the parameter
  budget and the protocol freeze;
* ``checks``   – CPU only: the twelve correctness gates;
* ``train``    – one GPU training unit (Rf recipe, 360 epochs);
* ``smoke``    – same code path with a few epochs; never reported;
* ``aggregate``– CPU only: predictions, bootstrap, oracle localisation, ledgers,
  decision, report.

Only engineering knobs are overridable (stage / task / model / device / resume /
smoke epochs).  Every scientific constant lives in the code modules and is
hashed into ``analysis/protocol_freeze.json``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import rr_common as C
import rr_data as D
import rr_train as T
from rr_common import Ledger, stage_timer

DEFAULT_CONFIG = C.FROZEN_CONFIG


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name=C.ROUND_ID,
        default_config=DEFAULT_CONFIG,
        default_study=C.STUDY_ID,
        description=(
            "LCRC relation-interface routing feasibility pilot: topology-block tokens + "
            "internal/cross-boundary S2 routing; R1 vs R2 (interface context) on official "
            "ogbg-mollipo (real label) and WC_LOGP; seed 0, Rf recipe, Test=DEVELOPMENT_TEST"
        ),
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# identity / fingerprints
# ---------------------------------------------------------------------------


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    from rr_prepare import partition_support

    payload: dict[str, Any] = {"round_id": C.ROUND_ID}
    for name, expected in C.EXPECTED_CACHE_SHA256.items():
        path = C.SQL_DATA / name
        payload[name] = C.sha256_file(path) if path.is_file() else "missing"
    split_payload: dict[str, Any] = {}
    for split in C.SPLITS:
        path = C.SQL_DATA / f"ogbg_mollipo_sq-pilot-cache-v1.npz"
        mesh = np.load(path, allow_pickle=False)
        try:
            idx = np.asarray(mesh[C.SPLIT_FILE[split]], dtype=np.int64)
        finally:
            mesh.close()
        split_payload[split] = {"n": int(idx.size), "sha256": C.sha256_array(idx)}
    payload["splits"] = split_payload
    payload["routing_cache_file_sha256"] = {
        split: (C.sha256_file(C.routing_cache_path(split)) if C.routing_cache_path(split).is_file() else "missing")
        for split in C.SPLITS
    }
    payload["targets_sha256"] = {
        task: {split: C.sha256_array(_target_array(task, split)) for split in C.SPLITS}
        for task in C.TASKS
    }
    return payload, C.sha256_json(split_payload)


def _target_array(task: str, split: str) -> np.ndarray:
    payload = np.load(C.SQL_DATA / "ogbg_mollipo_sq-pilot-cache-v1.npz", allow_pickle=False)
    try:
        ids = np.asarray(payload[C.SPLIT_FILE[split]], dtype=np.int64)
        if task == "lipo":
            labels = np.asarray(payload["labels"], dtype=np.float64)
            return labels[ids]
    finally:
        payload.close()
    import csv

    table: dict[int, float] = {}
    with C.WC_LOG_PRED_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_kind") != "original" or str(row.get("task", "")).lower() != "wc_logp":
                continue
            table[int(row["row_id"])] = float(row["value"])
    return np.asarray([table[int(i)] for i in ids.tolist()], dtype=np.float64)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def run(config: Mapping[str, Any], context):
    from ksvd_research.runtime.manifest import RunResult

    if context.test_access != "granted":
        raise RuntimeError(
            "this round evaluates the official Test split every epoch as DEVELOPMENT_TEST; "
            "the control plane blocked test access"
        )
    stage = str(config.get("stage", "prepare"))
    ledger = Ledger(Path(context.artifact_dir) / "ledger.jsonl")
    t0 = time.perf_counter()
    if stage == "prepare":
        metrics = _stage_prepare(config, ledger)
    elif stage == "checks":
        metrics = _stage_checks(config, ledger)
    elif stage in ("train", "smoke"):
        metrics = _stage_train(config, ledger, smoke=(stage == "smoke"))
    elif stage == "aggregate":
        metrics = _stage_aggregate(config, ledger)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    metrics["wall_seconds"] = time.perf_counter() - t0
    return RunResult(status="completed", metrics=metrics, artifacts=[])


def _stage_prepare(config: Mapping[str, Any], ledger) -> dict[str, Any]:
    from rr_prepare import prepare_all

    with stage_timer(ledger, "prepare", device="cpu", purpose="freeze"):
        out = prepare_all(force=bool(config.get("force", False)))
    return out


def _stage_checks(config: Mapping[str, Any], ledger) -> dict[str, Any]:
    import rr_checks as K
    from rr_prepare import prepare_all

    with stage_timer(ledger, "checks", device="cpu", purpose="gates"):
        prepare_all()
        bundle = D.build_bundle()
        out = K.run_all(bundle)
    C.atomic_write_json(C.ANALYSIS_DIR / "routing_invariants.json", out)
    return {
        "stage": "checks",
        "pass": out["pass"],
        "failing": out["failing"],
        "n_gates": len(out["gates"]),
    }


def _stage_train(config: Mapping[str, Any], ledger, *, smoke: bool) -> dict[str, Any]:
    task = str(config.get("task"))
    model_name = str(config.get("model"))
    if task not in C.TASKS:
        raise ValueError(f"task must be one of {C.TASKS}")
    if model_name not in C.MODELS:
        raise ValueError(f"model must be one of {C.MODELS}")
    bundle = D.build_bundle()
    out_dir = C.task_results_dir(task, model_name)
    if smoke:
        out_dir = C.RESULTS_DIR / "_smoke" / task / model_name
    epochs = int(config.get("smoke_epochs", 3)) if smoke else int(config.get("epochs", C.RECIPE["epochs"]))
    device = str(config.get("device", "cuda:0"))
    summary = T.run_trajectory(
        bundle,
        task,
        model_name,
        out_dir=out_dir,
        device=device,
        epochs=epochs,
        resume=bool(config.get("resume", True)),
        ledger=ledger,
        purpose="smoke" if smoke else "formal",
    )
    C.atomic_write_json(
        C.ANALYSIS_DIR / f"trajectory_{task}_{model_name}{'_smoke' if smoke else ''}.json",
        {
            "task": task,
            "model": model_name,
            "purpose": "smoke" if smoke else "formal",
            "selected": summary.get("selected"),
            "final": summary.get("final"),
            "last20": summary.get("last20"),
            "n_parameters": summary.get("n_parameters"),
            "signature": summary.get("signature"),
            "status": summary.get("status"),
            "device": summary.get("device"),
            "environment": summary.get("environment"),
        },
    )
    return {
        "stage": "smoke" if smoke else "train",
        "task": task,
        "model": model_name,
        "epochs": epochs,
        "selected_epoch": int(summary["selected"]["epoch"]),
        "valid_rmse_raw": float(summary["selected"]["valid_rmse_raw"]),
        "test_rmse_raw": float(summary["selected"]["test_rmse_raw"]),
        "n_parameters": int(summary["n_parameters"]),
        "train_seconds": float(summary["train_seconds"]),
        "status": summary.get("status"),
    }


def _stage_aggregate(config: Mapping[str, Any], ledger) -> dict[str, Any]:
    import rr_analysis as A
    import rr_report as RP

    with stage_timer(ledger, "aggregate", device="cpu", purpose="analysis"):
        out = A.aggregate()
    out["reports"] = RP.write_reports()
    return out


__all__ = ["build_runner", "run", "fingerprints", "DEFAULT_CONFIG"]
