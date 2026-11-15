"""Control-plane runner for ``lcrc-semantic-analysis-repair-bw0-s0-v1``.

Stages (``--set stage=...``):

``prepare``    input identity + corrected support statistics (read-only on the
               predecessor rounds)
``preflight``  the focused checks for exactly the repaired code paths
``train``      the single new formal training ``B_N100_lipo_W0``
``smoke``      a short adaptation-replay of that training (never a result)
``analyze``    every CPU repair section through :mod:`rp_analysis`

Test is a declared development-evaluation target of this round; selection stays
on Validation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import rp_common as RP

from ksvd_research.runner_api import Runner, RunnerError
from ksvd_research.runtime.manifest import RunResult

NAME = RP.PROTOCOL_ID
CONFIG_PATH = RP.CONFIGS / f"{RP.PROTOCOL_ID}.yaml"
STUDY = RP.STUDY

OPERATIONAL_KEYS = ("stage", "device_pref", "out_dir", "sections", "smoke", "epochs", "resume", "purpose")
FROZEN = {
    "protocol_id": RP.PROTOCOL_ID,
    "seed": 0,
    "dataset_name": "ogbg-mollipo",
    "split_method": "scaffold",
    "split_sizes": {"train": 3360, "valid": 420, "test": 420},
    "new_training": {"family": "B", "recipe": "W0", "dataset": "lipo", "budget": "N100", "seed": 0},
    "recipes": {"W0": {"weight_decay": 0.0}},
    "reference_template": {"arm": "B_N100_lipo_Rwd", "weight_decay": 0.01},
    "parameters": {"C1": RP.C1_PARAMS},
    "bootstrap": {"replicates": RP.BOOTSTRAP_REPLICATES, "seed": RP.BOOTSTRAP_SEED},
    "kernel": {"seed": RP.KERNEL_SEED, "rank": RP.RANK, "max_codes": 64},
    "tolerances": {"replay_max_abs_delta": RP.REPRO_TOL, "kernel_coupling": RP.KERNEL_TOL},
    "low_support_min_components": RP.LOW_SUPPORT_MIN_COMPONENTS,
}


def _frozen_config(config: dict[str, Any]) -> dict[str, Any]:
    for key, expected in FROZEN.items():
        if key not in config:
            raise RunnerError(f"config missing frozen key {key!r}")
        if config[key] != expected:
            raise RunnerError(f"config key {key!r} drifted:\n  config={config[key]}\n  frozen={expected}")
    return config


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in OPERATIONAL_KEYS}


def fingerprints(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    payload = {
        "protocol_id": RP.PROTOCOL_ID,
        "code_sha256": RP.code_hashes(),
        "git_commit": RP.git_commit(),
        "frozen": FROZEN,
        "scientific": scientific_config(config),
    }
    return payload, RP.sha256_bytes(repr(sorted(payload.items())).encode("utf-8"))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_prepare(config, context) -> RunResult:
    import rp_source as SC
    import rp_support as SU

    started = time.perf_counter()
    src = SC.run()
    sup = SU.run()
    return RunResult(
        metrics={"metric_rows": src["n_metric_rows"], "metric_rows_match": src["n_metric_rows_match"],
                 "n100_df": sup["n100"]["df_reference_size"],
                 "train_path_occurrences": sup["n100"]["train_occurrences"]["path"],
                 "risk_bins": sup["n100"]["risk_bins"]["status"],
                 "seconds": round(time.perf_counter() - started, 1)},
        artifacts=src["artifacts"] + sup["artifacts"],
    )


def stage_preflight(config, context) -> RunResult:
    import rp_checks as CK

    started = time.perf_counter()
    payload = CK.run()
    return RunResult(
        status="completed" if payload["all_ok"] else "failed",
        metrics={"all_ok": payload["all_ok"], "failed": payload["failed"],
                 "seconds": round(time.perf_counter() - started, 1)},
        artifacts=payload["artifacts"],
        error=None if payload["all_ok"] else f"focused checks failed: {payload['failed']}",
    )


def stage_train(config, context) -> RunResult:
    import rp_train as RT

    device = str(config.get("device_pref", "cuda"))
    payload = RT.run(device, resume=bool(config.get("resume")))
    ok = payload.get("status") == "completed"
    return RunResult(
        status="completed" if ok else "failed",
        metrics={k: payload.get(k) for k in
                 ("best_epoch", "best_valid_rmse", "best_test_rmse", "grid_valid_rmse",
                  "grid_test_rmse", "parameters", "gpu_hours", "init_state_hash")},
        artifacts=[RP.track_rel(Path(payload["out_dir"]) / "metrics.json"),
                   RP.track_rel(Path(payload["out_dir"]) / "curve.csv")],
        error=None if ok else f"training status {payload.get('status')}",
    )


def stage_smoke(config, context) -> RunResult:
    import rp_train as RT

    started = time.perf_counter()
    payload = RT.run(str(config.get("device_pref", "cuda")), smoke=True,
                     epochs=int(config.get("epochs") or 2))
    return RunResult(
        metrics={"smoke": True, "status": payload.get("status"),
                 "seconds": round(time.perf_counter() - started, 1),
                 "init_state_hash": payload.get("init_state_hash")},
        artifacts=[RP.track_rel(Path(payload["out_dir"]) / "metrics.json")],
    )


def stage_analyze(config, context) -> RunResult:
    import rp_analysis as A

    sections = config.get("sections") or None
    if isinstance(sections, str):
        sections = [s.strip() for s in sections.split(",") if s.strip()] or None
    payload = A.run_all(sections)
    failed = [k for k, v in payload["sections"].items() if isinstance(v, dict) and "error" in v]
    artifacts: list[str] = [RP.track_rel(RP.ANALYSIS / "analysis_run.json")]
    for v in payload["sections"].values():
        if isinstance(v, dict):
            artifacts.extend(v.get("artifacts", []) or [])
    return RunResult(
        status="completed" if not failed else "failed",
        metrics={"sections": list(payload["sections"]), "failed": failed, "seconds": payload["seconds"]},
        artifacts=artifacts,
        error=None if not failed else f"sections failed: {failed}",
    )


STAGES = {
    "prepare": stage_prepare,
    "preflight": stage_preflight,
    "train": stage_train,
    "smoke": stage_smoke,
    "analyze": stage_analyze,
}


def run(config, context) -> RunResult:
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
        description="LCRC analysis repair (reader/support/explicit/kernel/export/edit) + the single B_N100_lipo_W0 control.",
        run_module=sys.modules[__name__],
    )
