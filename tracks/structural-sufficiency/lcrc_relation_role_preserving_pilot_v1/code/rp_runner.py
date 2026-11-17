"""Runner ``lcrc-relation-role-preserving-pilot-v1``.

One run = one stage.  The four formal training units are

    --set stage=train --set task=lipo    --set model=RR_REAL     (pool res-gpu0)
    --set stage=train --set task=wc_logp --set model=RR_REAL     (pool res-gpu0)
    --set stage=train --set task=lipo    --set model=RR_CONTROL  (pool res-gpu1)
    --set stage=train --set task=wc_logp --set model=RR_CONTROL  (pool res-gpu1)

Stages:

* ``prepare``  – CPU only: re-verify the frozen routing cache, freeze the
  CONTROL S2-type hash table, print the role group sizes, the parameter budget
  and the protocol freeze;
* ``checks``   – CPU only: the correctness gates (all of them, including the
  trained-state gates when the four checkpoints exist);
* ``train``    – one GPU training unit (Rf recipe, 360 epochs);
* ``smoke``    – same code path with a few epochs; never reported;
* ``aggregate``– predictions, paired bootstrap, gamma ablation, WC_LOGP oracle
  localisation, decision and report.

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

import rp_common as C
import rp_data as D
import rp_train as T
from rp_common import Ledger, stage_timer

DEFAULT_CONFIG = C.FROZEN_CONFIG


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name=C.ROUND_ID,
        default_config=DEFAULT_CONFIG,
        default_study=C.STUDY_ID,
        description=(
            "LCRC role-preserving C1 pilot: the frozen C1 atom/bond/S1/S2/LocalMLP/"
            "atom-sum/rho function, with the S2 columns split into internal vs cross-boundary "
            "(RR_REAL) or an arbitrary label-free S2 type hash (RR_CONTROL) and rescaled by two "
            "trainable per-channel gamma vectors; official ogbg-mollipo, seed 0, Rf 360 epochs"
        ),
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# identity / fingerprints
# ---------------------------------------------------------------------------


def _target_array(task: str, split: str) -> np.ndarray:
    import csv

    path = C.SQL_DATA / "ogbg_mollipo_sq-pilot-cache-v1.npz"
    with np.load(path, allow_pickle=False) as payload:
        ids = np.asarray(payload[f"{split}_idx"], dtype=np.int64)
        if task == "lipo":
            labels = np.asarray(payload["labels"], dtype=np.float64)
            return labels[ids]
    table: dict[int, float] = {}
    with C.WC_LOG_PRED_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_kind") != "original" or str(row.get("task", "")).lower() != "wc_logp":
                continue
            table[int(row["row_id"])] = float(row["value"])
    return np.asarray([table[int(i)] for i in ids.tolist()], dtype=np.float64)


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Dataset / split / routing / target / control-table fingerprint of one run."""
    path = C.SQL_DATA / "ogbg_mollipo_sq-pilot-cache-v1.npz"
    payload: dict[str, Any] = {"round_id": C.ROUND_ID}
    payload["graph_cache_sha256"] = C.sha256_file(path) if path.is_file() else "missing"
    split_payload: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as mesh:
        for split in C.SPLITS:
            idx = np.asarray(mesh[f"{split}_idx"], dtype=np.int64)
            split_payload[split] = {"n": int(idx.size), "sha256": C.sha256_array(idx)}
    payload["splits"] = split_payload
    payload["routing_cache_file_sha256"] = {
        split: (
            C.sha256_file(C.PRED_CACHE / f"ogbg_mollipo_{split}_topology-routing-cache-v1.npz")
            if (C.PRED_CACHE / f"ogbg_mollipo_{split}_topology-routing-cache-v1.npz").is_file()
            else "missing"
        )
        for split in C.SPLITS
    }
    payload["targets_sha256"] = {
        task: {split: C.sha256_array(_target_array(task, split)) for split in C.SPLITS}
        for task in C.TASKS
    }
    payload["control_table_sha256"] = (
        C.sha256_file(C.control_table_path()) if C.control_table_path().is_file() else "missing"
    )
    return payload, C.sha256_json(split_payload)


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
    from rp_prepare import prepare_all

    with stage_timer(ledger, "prepare", device="cpu", purpose="freeze"):
        out = prepare_all(force=bool(config.get("force", False)))
    return out


def _stage_checks(config: Mapping[str, Any], ledger) -> dict[str, Any]:
    import json

    import rp_checks as K
    from rp_prepare import prepare_all

    with stage_timer(ledger, "checks", device="cpu", purpose="gates"):
        prepare_all()
        bundle = D.build_bundle()
        checkpoints = K.discover_checkpoints()
        phase = "post_training" if checkpoints else "pre_training"
        previous_path = C.ANALYSIS_DIR / "routing_invariants.json"
        previous = C.read_json(previous_path) if previous_path.is_file() else {}
        out = K.run_all(bundle, checkpoint_paths=checkpoints or None, phase=phase)
        if previous and previous.get("phase") == "pre_training" and phase == "post_training":
            out["pre_training"] = previous
    C.atomic_write_json(C.ANALYSIS_DIR / "routing_invariants.json", out)
    return {
        "stage": "checks",
        "phase": phase,
        "pass": out["pass"],
        "failing": out["failing"],
        "n_gates": len(out["gates"]),
        "n_checkpoints": len(checkpoints),
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
    import rp_analysis as A
    import rp_report as RP

    analysis_device = str(config.get("analysis_device", "cpu"))
    with stage_timer(ledger, "aggregate", device=analysis_device, purpose="analysis"):
        out = A.aggregate(device=analysis_device)
    out["reports"] = RP.write_reports()
    return out


__all__ = ["build_runner", "run", "fingerprints", "DEFAULT_CONFIG"]
