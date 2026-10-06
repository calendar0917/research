"""Runner: ZINC CSSD nonlinear binding v1 (node/edge joint-vs-separate 2x2).

``model.stage``:

* ``build-objects``  stage-0 phase A: fold + targets + tuple payload +
                     kappa_M + prep (fit-only; local CPU).
* ``cssd-refit``     stage-0: CSSD q1 basis refit on the fit split (CPU,
                     320 epochs, last-5 average).
* ``train-q``        stage-0: one shared Q(topology25) head on fit (CPU).
* ``checks``         focused pre-training correctness checks (CPU).
* ``smoke``          short GPU plumbing smoke on real fit batches (all arms).
* ``train``          one 240-epoch body run (``model.arm``, ``model.seed``).
* ``select-eval``    Stage A: full-y raw/cal MAE on select + contrasts +
                     frozen decision rules (seed 0 primary).
* ``confirm-eval``   Stage B: one-shot confirm evaluation of the frozen roster.
* ``interventions``  post-training N/E pairing-shuffle diagnostics.

Official ZINC **valid** and **test** are never instantiated; a granted
``test_access`` is refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from ksvd_research.runtime.fingerprints import zinc_fingerprints
from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, resolve_path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ksvd_research.runner_api import RunnerError  # noqa: E402

from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_cssd_nonlinear_binding_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_cssd_nonlinear_binding_v1.yaml"

STAGES = (
    "build-objects",
    "cssd-refit",
    "train-q",
    "checks",
    "smoke",
    "train",
    "select-eval",
    "confirm-eval",
    "interventions",
)

#: one entry per pre-registered arm (A = strength reference with the original
#: product binding slots; C** = the 2x2 joint/separate factorial)
ARM_TABLE = {
    "A": {"n": "original-product", "e": "original-product"},
    "C00": {"n": "sep", "e": "sep"},
    "C10": {"n": "joint", "e": "sep"},
    "C01": {"n": "sep", "e": "joint"},
    "C11": {"n": "joint", "e": "joint"},
}


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_cssd_nonlinear_binding_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Frozen-CSSD pre-pooling joint vs separate node/edge structure-semantics binding: "
            "strength reference A (original product binding slots restored) plus the C00/C10/C01/C11 "
            "2x2 factorial; identical body skeleton/recipe, fit-only basis+Q, fixed last-5 soup, "
            "fit-median y bias; select primary / confirm one-shot; official valid+test blocked"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "zinc_cssd_nonlinear_binding_v1 is non-terminal; it must never run with test access granted"
        )
    model_cfg = config.get("model", {})
    stage = str(model_cfg.get("stage", "checks"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}, got {stage!r}")
    runtime = config.get("runtime", {})
    device_name = str(runtime.get("device", "cpu"))
    arm = str(model_cfg.get("arm", "C00"))
    if arm not in ARM_TABLE:
        raise ValueError(f"model.arm must be one of {sorted(ARM_TABLE)}, got {arm!r}")
    seed = int(model_cfg.get("seed", stages.SEED))
    out_dir = stages.RESULTS_DIR
    threads = int(runtime.get("torch_threads", 8))
    print(f"[cssd-bind-v1] stage={stage} arm={arm} seed={seed} device={device_name}", flush=True)

    def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
        base = {
            "measure": f"zinc_cssd_nonlinear_binding_v1::{stage}",
            "stage": stage,
            "arm": arm,
            "seed": seed,
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        base.update({k: v for k, v in payload.items() if k in (
            "all_passed", "all_finite", "seconds", "steps_done", "stopped_reason",
            "recommendation", "seed1_arms",
        )})
        if "fit_diagnostics" in payload:
            base["fit_y_raw_mae"] = float(payload["fit_diagnostics"]["fit_y_raw_mae"])
        if "seed0_mae_raw" in payload:
            base["seed0_mae_raw"] = payload["seed0_mae_raw"]
        return base

    if stage == "build-objects":
        result = stages.phase_build_objects(out_dir=out_dir)
        _write_json(context.artifact_dir / "build_objects.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/build_objects.json"])

    if stage == "cssd-refit":
        result = stages.cssd_refit(out_dir=out_dir, threads=threads)
        _write_json(context.artifact_dir / "cssd_refit.json", result)
        metrics = _metrics(result)
        metrics["fit_monitor_mae_last"] = result["fit_monitor_mae_last"]
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/cssd_refit.json"])

    if stage == "train-q":
        result = stages.train_q(out_dir=out_dir)
        _write_json(context.artifact_dir / "Q_meta_stage.json", result)
        metrics = _metrics(result)
        metrics["last_train_mae"] = result["last_train_mae"]
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/Q_meta_stage.json"])

    if stage == "checks":
        checks = stages.run_checks(
            n_graphs=int(model_cfg.get("check_graphs", 24)), out_dir=out_dir
        )
        _write_json(context.artifact_dir / "checks.json", checks)
        metrics = _metrics(checks)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        print(f"[cssd-bind-v1] checks all_passed={checks['all_passed']}", flush=True)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/checks.json"])

    if stage == "smoke":
        smoke = stages.run_smoke(
            device_name=device_name,
            n_batches=int(model_cfg.get("smoke_batches", 4)),
            out_dir=out_dir,
        )
        _write_json(context.artifact_dir / "smoke.json", smoke)
        metrics = _metrics(smoke)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/smoke.json"])

    if stage == "train":
        import torch

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable; no silent CPU fallback")
        epochs = int(model_cfg.get("epochs", stages.EPOCHS))
        result = stages.train_arm(
            arm,
            seed=seed,
            device_name=device_name,
            out_dir=out_dir,
            epochs=epochs,
            max_steps=None if model_cfg.get("max_steps") is None else int(model_cfg["max_steps"]),
        )
        _write_json(context.artifact_dir / "train_manifest.json", result)
        metrics = _metrics(result)
        metrics["soup_state_sha256"] = result["soup_state_sha256"]
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/train_manifest.json"])

    if stage == "select-eval":
        seeds = [int(s) for s in model_cfg.get("seeds", [stages.SEED])]
        result = stages.select_eval(
            seeds=seeds, device_name=device_name, out_dir=out_dir
        )
        _write_json(context.artifact_dir / "select_eval.json", result)
        metrics = _metrics(result)
        if "frozen_decision" in result:
            metrics["recommendation"] = result["frozen_decision"]["recommendation"]
            metrics["seed1_arms"] = result["frozen_decision"]["seed1_arms"]
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/select_eval.json"])

    if stage == "confirm-eval":
        roster_path = out_dir / "confirm_roster.json"
        if not roster_path.exists():
            raise RuntimeError(
                "confirm_roster.json missing: freeze the roster (select-eval decision) before confirm"
            )
        result = stages.confirm_eval(device_name=device_name, out_dir=out_dir)
        _write_json(context.artifact_dir / "confirm_eval.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/confirm_eval.json"])

    if stage == "interventions":
        seeds = [int(s) for s in model_cfg.get("seeds", [stages.SEED])]
        result = stages.run_interventions(device_name=device_name, out_dir=out_dir, seeds=seeds)
        _write_json(context.artifact_dir / "interventions.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/interventions.json"])

    raise RunnerError(f"unhandled stage {stage}")
