"""Control-plane runner for ``lcrc-lipo-context-small-s0-v1``.

Stages (``--set stage=...``):

``prepare``    build/verify the label-free structure caches, the train-only
               structure normalisation, the train-only target normalisation and
               the shared epoch plan; record every hash in ``analysis/prepare.json``
``preflight``  run the targeted preflight sections (a-f) on this device and write
               ``analysis/preflight.json``
``train``      one formal arm (``--set arm=B|S|R1|R2|D2|G``); see ``lp_train.py``
``analyze``    assemble the main/sensitivity/group/intervention tables from the
               pulled per-arm outputs (CPU, no training)

The official Test split is a declared development-evaluation target of this
study (``protocols/lcrc-lipo-context-small-s0-v1.yaml``); selection stays on
Validation and the runner refuses to let any stage select on Test.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

import lp_common as C

from ksvd_research.runner_api import Runner, RunnerError
from ksvd_research.runtime.manifest import RunResult

NAME = "lcrc_lipo_context_small_v1"
CONFIG_PATH = C.CONFIGS / f"{C.PROTOCOL_ID}.yaml"
STUDY = C.PROTOCOL_ID

PREPARE_PATH = C.ANALYSIS / "prepare.json"
PREFLIGHT_PATH = C.ANALYSIS / "preflight.json"
OPERATIONAL_KEYS = ("stage", "arm", "sections", "out_dir", "f_epochs", "steps", "device_pref")


# ---------------------------------------------------------------------------
# frozen-config enforcement
# ---------------------------------------------------------------------------


def _frozen_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "protocol_id": C.PROTOCOL_ID,
        "seed": int(C.TRAIN_SEED),
        "dataset": C.DATASET,
        "split_method": C.OGB_SPLIT_METHOD,
        "arms": list(C.ARMS),
        "epochs": dict(C.EPOCHS),
        "c1_recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in C.C1_RECIPE.items()},
        "dmpnn_recipe": dict(C.DMPNN_RECIPE),
        "expected_parameters": dict(C.EXPECTED_PARAMS),
        "promise": {
            "rel_improvement": C.PROMISE_REL_IMPROVEMENT,
            "mae_tolerance": C.PROMISE_MAE_TOLERANCE,
            "tail_trim_fractions": list(C.TAIL_TRIM_FRACTIONS),
        },
        "groups": {
            "gap_threshold": C.GROUP_GAP_THRESHOLD,
            "min_side": C.GROUP_MIN_SIDE,
            "min_bin_side": C.GROUP_MIN_BIN_SIDE,
            "min_bins": C.GROUP_MIN_BINS,
            "support_min": C.GROUP_SUPPORT_MIN,
            "morgan_radius": C.MORGAN_RADIUS,
        },
        "bootstrap": {
            "replicates": C.BOOTSTRAP_REPLICATES,
            "seed": C.BOOTSTRAP_SEED,
            "weight_dist": C.BOOTSTRAP_WEIGHT_DIST,
        },
        "budget": {
            "soft_gpu_hours": C.GPU_SOFT_CAP_HOURS,
            "hard_gpu_hours": C.GPU_HARD_CAP_HOURS,
        },
    }
    for key, expected in required.items():
        if key not in config:
            raise RunnerError(f"config is missing the frozen key {key!r}")
        if config[key] != expected:
            raise RunnerError(
                f"config key {key!r} drifted from the frozen protocol:\n"
                f"  config   = {config[key]}\n  protocol = {expected}"
            )
    return config


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in OPERATIONAL_KEYS}


def code_hashes() -> dict[str, str]:
    return C.code_hashes()


def environment() -> dict[str, Any]:
    return C.env_info()


def fingerprints(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Dataset + split fingerprints (must not depend on the operational stage)."""
    manifest_path = C.DATA_MANIFEST
    if not manifest_path.is_file():
        raise RunnerError(f"missing {C.track_rel(manifest_path)}; run lp_source.py first")
    manifest = C.read_json(manifest_path)
    split = C.read_json(C.SPLIT_JSON)
    payload = {
        "dataset": C.DATASET,
        "split_method": C.OGB_SPLIT_METHOD,
        "ogb_archive_sha256": manifest["ogb_archive"]["sha256"],
        "mol_csv_sha256": manifest["files"]["mol_csv"]["sha256"],
        "raw_artifact_sha256": manifest["artifacts"]["raw_attributes"]["sha256"],
        "labels_sha256": manifest["artifacts"]["labels"]["sha256"],
        "split_file_sha256": split["source_file_sha256"],
        "split_sizes": split["sizes"],
        "n_molecules": C.N_MOLECULES,
    }
    split_fingerprint = C.sha256_json({"rows_sha256": split["rows_sha256"], "method": C.OGB_SPLIT_METHOD})
    return payload, split_fingerprint


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_prepare(config: dict[str, Any], context) -> RunResult:
    import lp_data as D
    import lp_source as S

    C.setup_determinism()
    started = time.perf_counter()
    S.check()
    view = D.build_view()
    plan = D.epoch_plan(int(view.caches["train"]["n_molecules"]), int(C.EPOCHS["B"]), int(C.TRAIN_SEED))
    dm_plan = D.epoch_plan(int(view.caches["train"]["n_molecules"]), int(C.EPOCHS["G"]), int(C.TRAIN_SEED))
    payload = {
        "schema": "lcrc-lipo-context-small-prepare-v1",
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "data": view.audit(),
        "data_manifest_sha256": C.sha256_file(C.DATA_MANIFEST),
        "epoch_plan": {
            "c1": {k: plan[k] for k in ("sha256", "shape", "seed", "shuffle_offset", "generator")},
            "dmpnn": {k: dm_plan[k] for k in ("sha256", "shape", "seed", "shuffle_offset", "generator")},
            "note": (
                "one row plan per (n_train, epochs) generated from seed+917000; every arm with "
                "the same n_train and epoch count consumes exactly the same permutation sequence"
            ),
        },
        "cache_files": {
            split: {"path": C.track_rel(D.cache_path(split)), "sha256": C.sha256_file(D.cache_path(split))}
            for split in C.SPLITS
        },
        "seconds": float(time.perf_counter() - started),
        "environment": environment(),
    }
    C.write_json(PREPARE_PATH, payload)
    return RunResult(
        metrics={
            "n_train": int(view.caches["train"]["n_molecules"]),
            "n_valid": int(view.caches["valid"]["n_molecules"]),
            "n_test": int(view.caches["test"]["n_molecules"]),
            "target_mean": view.target_norm["mean"],
            "target_scale": view.target_norm["scale"],
            "epoch_plan_sha256": plan["sha256"],
            "cache_train_fingerprint": view.audit()["cache_fingerprints"]["train"],
        },
        artifacts=[C.track_rel(PREPARE_PATH)],
    )


def stage_preflight(config: dict[str, Any], context) -> RunResult:
    import torch

    import lp_data as D
    import lp_preflight as P

    device = torch.device(str(config.get("device_pref", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    C.setup_determinism()
    view = D.build_view()
    sections = list(config.get("sections") or P.SECTIONS)
    payload = P.run_sections(
        view,
        device,
        sections,
        steps=int(config.get("steps", 100)),
        f_epochs=int(config.get("f_epochs", 2)),
    )
    payload["environment"] = C.env_info()
    payload["git_commit"] = C.git_commit()
    payload["git_dirty_tracked"] = C.git_dirty()
    payload["data"] = view.audit()
    checks = P.collect_checks(payload)
    payload["checks"] = checks
    payload["all_passed"] = all(checks.values())
    payload["sections_run"] = sections
    C.write_json(PREFLIGHT_PATH, payload)
    return RunResult(
        status="completed" if payload["all_passed"] else "failed",
        metrics={
            "all_passed": payload["all_passed"],
            "n_checks": len(checks),
            "failed_checks": [k for k, v in checks.items() if not v],
            "device": str(device),
        },
        artifacts=[C.track_rel(PREFLIGHT_PATH)],
        error=None if payload["all_passed"] else "preflight checks failed",
    )


def stage_train(config: dict[str, Any], context) -> RunResult:
    import torch

    import lp_train as T

    arm = str(config.get("arm") or "")
    if arm not in C.ARMS:
        raise RunnerError(f"stage=train needs --set arm=<{'|'.join(C.ARMS)}>")
    device = torch.device(str(config.get("device_pref", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RunnerError("CUDA requested but unavailable on this host")
    out_dir = Path(config.get("out_dir") or (C.RUNS / f"arm_{arm}_s{int(C.TRAIN_SEED)}"))
    payload = T.train_run(arm=arm, device=device, out_dir=out_dir)
    summary = payload["summary"]
    return RunResult(
        metrics={
            "arm": arm,
            "best_epoch": summary["best_valid"]["epoch"],
            "best_valid_rmse": summary["best_valid"]["valid"]["rmse"],
            "best_valid_test_rmse": summary["best_valid"]["test"]["rmse"],
            "last_valid_rmse": summary["last_epoch"]["valid"]["rmse"],
            "last_test_rmse": summary["last_epoch"]["test"]["rmse"],
            "parameters": payload["model"]["parameters"],
            "gpu_hours": payload["timing"]["gpu_hours_charged"],
        },
        artifacts=[
            C.track_rel(out_dir / "metrics.json"),
            C.track_rel(out_dir / "curve.csv"),
        ],
    )


def stage_analyze(config: dict[str, Any], context) -> RunResult:
    import lp_analysis as A

    started = time.perf_counter()
    payload = A.run_all(make_figures=True)
    return RunResult(
        metrics={**payload.get("metrics", {}), "seconds": float(time.perf_counter() - started)},
        artifacts=payload.get("artifacts", []),
    )


STAGES = {
    "prepare": stage_prepare,
    "preflight": stage_preflight,
    "train": stage_train,
    "analyze": stage_analyze,
}


def run(config: dict[str, Any], context) -> RunResult:
    config = _frozen_config(config)
    stage = str(config.get("stage") or "")
    if stage not in STAGES:
        raise RunnerError(f"stage must be one of {sorted(STAGES)}")
    return STAGES[stage](config, context)


def build_runner() -> Runner:
    import sys

    return Runner(
        name=NAME,
        default_config=CONFIG_PATH,
        default_study=STUDY,
        description=(
            "LCRC small-data Lipophilicity round: B/S/R1/R2/D2 context arms + D-MPNN reference, "
            "official OGB scaffold split, seed 0, development-evaluation Test access."
        ),
        run_module=sys.modules[__name__],
    )
