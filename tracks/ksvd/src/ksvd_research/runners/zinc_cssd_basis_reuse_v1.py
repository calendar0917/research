"""Runner: ZINC CSSD basis reuse v1 (frozen source-domain basis on a size gap).

``model.stage``:

* ``build-objects``  stage 0: domain fold (n_nodes threshold + canonical-SMILES
                     group-hash rules) + T_fit-refit targets/payload/kappa_M/
                     prep (CPU; reads the committed canonical-SMILES table).
* ``cssd-refit``     one CSSD basis refit (``model.arm`` = SOURCE on S_fit |
                     TARGET on T_fit; CPU, 320 epochs, last-5 average, equal
                     molecule counts asserted).
* ``train-q``        one shared Q(topology25) head on T_fit only (CPU).
* ``checks``         focused pre-training correctness checks (CPU).
* ``smoke``          short GPU plumbing smoke on real T_fit batches (both arms).
* ``train``          one 240-epoch body run (``model.arm``, ``model.seed`` in
                     {0, 1}; identical skeleton/recipe, only the frozen basis
                     package differs).
* ``terminal-eval``  the one-shot terminal stage: roster -> coverage -> the
                     unique dictionary intervention -> T_eval scoring of all
                     four runs -> SMILES-group-paired bootstrap -> frozen
                     decision rules (A-F).

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
    zinc_cssd_basis_reuse_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_cssd_basis_reuse_v1.yaml"

STAGES = (
    "build-objects",
    "cssd-refit",
    "train-q",
    "checks",
    "smoke",
    "train",
    "terminal-eval",
)

#: one entry per arm (SOURCE = frozen basis fit on S_fit small molecules;
#: TARGET = the identical-method refit on T_fit, the matched control)
ARM_TABLE = {
    "SOURCE": {"basis_rows": "s_fit", "consumer_fit": "t_fit"},
    "TARGET": {"basis_rows": "t_fit", "consumer_fit": "t_fit"},
}


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_cssd_basis_reuse_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Frozen-CSSD basis reuse across a molecule-size domain gap: the whole basis "
            "package (U/common_rms/D) fit on the SOURCE domain (n_nodes <= 23) vs the "
            "identical-method refit on the TARGET domain, consumed by the identical "
            "Full446 / M_COMP+Q skeleton trained on T_fit with one shared Q head and one "
            "recipe; body seeds 0/1; T_eval = sealed canonical-SMILES group prefix, opened "
            "once at terminal-eval with coverage, the unique alpha-mean dictionary "
            "intervention, SMILES-group-paired bootstrap and frozen A-F decision rules; "
            "official valid+test blocked"
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
            "zinc_cssd_basis_reuse_v1 is non-terminal; it must never run with test access granted"
        )
    model_cfg = config.get("model", {})
    stage = str(model_cfg.get("stage", "checks"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}, got {stage!r}")
    runtime = config.get("runtime", {})
    device_name = str(runtime.get("device", "cpu"))
    arm = str(model_cfg.get("arm", ""))
    if stage in ("cssd-refit", "train") and arm not in ARM_TABLE:
        raise RunnerError(f"stage {stage} requires model.arm in {sorted(ARM_TABLE)}, got {arm!r}")
    seed = int(model_cfg.get("seed", stages.SEED))
    if stage == "train" and seed not in stages.REUSE_SEEDS:
        raise RunnerError(
            f"stage train requires model.seed in {list(stages.REUSE_SEEDS)}, got {seed}"
        )
    out_dir = stages.RESULTS_DIR
    threads = int(runtime.get("torch_threads", 8))
    print(f"[cssd-reuse-v1] stage={stage} arm={arm or '-'} seed={seed} device={device_name}", flush=True)

    def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
        base = {
            "measure": f"zinc_cssd_basis_reuse_v1::{stage}",
            "stage": stage,
            "arm": arm or None,
            "seed": seed,
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        base.update({k: v for k, v in payload.items() if k in (
            "all_passed", "all_finite", "seconds", "steps_done", "stopped_reason",
            "recommendation", "branch", "one_shot", "n_fit_molecules",
        )})
        if "fit_diagnostics" in payload:
            base["fit_y_raw_mae"] = float(payload["fit_diagnostics"]["fit_y_raw_mae"])
        if "decision" in payload:
            base["branch"] = payload["decision"]["branch"]
            base["recommendation"] = payload["decision"]["recommendation"]
        return base

    if stage == "build-objects":
        result = stages.phase_build_objects(out_dir=out_dir)
        _write_json(context.artifact_dir / "build_objects.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/build_objects.json"])

    if stage == "cssd-refit":
        result = stages.cssd_refit_domain(arm, out_dir=out_dir, threads=threads)
        _write_json(context.artifact_dir / f"cssd_refit_{arm}.json", result)
        metrics = _metrics(result)
        metrics["fit_monitor_mae_last"] = result["fit_monitor_mae_last"]
        metrics["n_fit_molecules"] = result["n_fit_molecules"]
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=[f"artifacts/cssd_refit_{arm}.json"])

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
        metrics["all_passed"] = True
        _write_json(context.artifact_dir / "metrics.json", metrics)
        print("[cssd-reuse-v1] checks all_passed=True", flush=True)
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
        result = stages.train_reuse_arm(
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

    if stage == "terminal-eval":
        result = stages.terminal_eval(device_name=device_name, out_dir=out_dir)
        _write_json(context.artifact_dir / "terminal_eval.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/terminal_eval.json"])

    raise RunnerError(f"unhandled stage {stage}")
