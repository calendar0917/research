"""Runner: ZINC CSSD consumer generalization v1 (frozen basis + access; can a
limited training arrangement improve the DICT consumer's g generalization?).

``model.stage``:

* ``source-checks``         verify every read-only reused object and the
                           historical DICT_s0/s1 run artifacts (hashes,
                           schedule prefix reproduction, init reproduction);
                           CPU.
* ``checkpoint-diagnostics`` stage A per seed (``model.seed`` in {0, 1}):
                           read-only forward export + pre-registered
                           analysis of the historical epoch40/epoch120/
                           epoch240/soup checkpoints on fit/dev; GPU or CPU.
* ``freeze-roster``         apply the frozen routing rule to the two
                           diagnosis files and write the one-shot
                           roster_lock.json; CPU.
* ``smoke``                 short end-to-end GPU plumbing smoke (1-epoch
                           trajectory with a SMOKE estimator).
* ``train-trajectory``      one from-scratch DICT trajectory per seed with
                           deterministic member capture and the frozen
                           estimator construction (``model.seed`` in {0, 1}).
* ``terminal-eval``         the one-shot terminal stage: roster verification,
                           full fit/dev scoring of every estimator, paired
                           deltas vs CTRL240, the historical reproduction
                           gap, shared-group bootstrap, noise bound, the
                           unique alpha-mean intervention, the frozen
                           retention gate + tie-break.

Official ZINC **valid** and **test** are never instantiated; a granted
``test_access`` is refused.
"""

from __future__ import annotations

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
    zinc_cssd_consumer_generalization_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_cssd_consumer_generalization_v1.yaml"

STAGES = (
    "source-checks",
    "checkpoint-diagnostics",
    "freeze-roster",
    "smoke",
    "train-trajectory",
    "terminal-eval",
)
SEED_STAGES = ("checkpoint-diagnostics", "train-trajectory")


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_cssd_consumer_generalization_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Frozen-CSSD basis + access held fixed (the verified DICT consumer of "
            "zinc_cssd_consumer_replacement_v1): Stage A read-only checkpoint "
            "diagnostics (epoch40/120/240/soup on fit/dev, pre-registered component "
            "and k=0 group analyses) drive a frozen routing rule (OVERFIT / "
            "STILL_IMPROVING / FLAT_OR_MIXED) into at most two new from-scratch "
            "trajectories (seeds 0/1, <=360 epochs) whose fixed estimators "
            "(CTRL240=mean(236..240) control; candidates WIDE5_240=mean(200,210,220,"
            "230,240) and route-conditional EARLY120/LONG360) are equal-weight FP32 "
            "means of five full member states; one-shot terminal eval with "
            "group-paired bootstrap (2000 draws, seed 20261007), the unique "
            "alpha-mean intervention on every candidate and a frozen retention gate "
            "+ tie-break; dev = the historical development rows (also used for "
            "branch selection — never an independent confirm); official "
            "valid+test blocked"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "zinc_cssd_consumer_generalization_v1 is non-terminal; it must never run with test access granted"
        )
    model_cfg = config.get("model", {})
    stage = str(model_cfg.get("stage", "source-checks"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}, got {stage!r}")
    runtime = config.get("runtime", {})
    device_name = str(runtime.get("device", "cpu"))
    seed = int(model_cfg.get("seed", stages.SEEDS[0]))
    if stage in SEED_STAGES and seed not in stages.SEEDS:
        raise RunnerError(f"stage {stage} requires model.seed in {list(stages.SEEDS)}, got {seed}")
    out_dir = stages.RESULTS_DIR
    print(
        f"[cssd-consumer-generalization-v1] stage={stage} seed={seed} device={device_name}",
        flush=True,
    )

    def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
        base = {
            "measure": f"zinc_cssd_consumer_generalization_v1::{stage}",
            "stage": stage,
            "seed": seed,
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        base.update({k: v for k, v in payload.items() if k in (
            "all_passed", "seconds", "steps_done", "stopped_reason", "one_shot",
            "fit_n", "dev_n", "curve_seconds_total", "wall_clock_s",
        )})
        if "route" in payload:
            base["route"] = payload["route"]["route"]
        if "decision" in payload:
            base["passed_candidates"] = payload["decision"]["passed_candidates"]
            base["winner"] = payload["decision"]["winner"]
        return base

    if stage == "source-checks":
        result = stages.source_checks(out_dir=out_dir)
        _write_json(context.artifact_dir / "source_manifest.json", result)
        metrics = _metrics(result)
        metrics["fit_n"] = result["fit_n"]
        metrics["dev_n"] = result["dev_n"]
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/source_manifest.json"])

    if stage == "checkpoint-diagnostics":
        result = stages.checkpoint_diagnostics(seed, device_name=device_name, out_dir=out_dir)
        _write_json(context.artifact_dir / "diagnosis.json", result)
        metrics = _metrics(result)
        metrics["fit_n"] = result["n_fit"]
        metrics["dev_n"] = result["n_dev"]
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/diagnosis.json"])

    if stage == "freeze-roster":
        result = stages.freeze_roster(out_dir=out_dir)
        _write_json(context.artifact_dir / "roster_lock.json", result)
        metrics = _metrics(result)
        metrics["route"] = result["route"]["route"]
        metrics["epochs"] = result["epochs"]
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/roster_lock.json"])

    if stage == "smoke":
        result = stages.run_smoke(device_name=device_name, out_dir=out_dir)
        _write_json(context.artifact_dir / "smoke.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/smoke.json"])

    if stage == "train-trajectory":
        import torch

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable; no silent CPU fallback")
        result = stages.train_trajectory(seed, device_name=device_name, out_dir=out_dir)
        _write_json(context.artifact_dir / "train_manifest.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/train_manifest.json"])

    if stage == "terminal-eval":
        result = stages.terminal_eval(device_name=device_name, out_dir=out_dir)
        _write_json(context.artifact_dir / "terminal_eval.json", result)
        metrics = _metrics(result)
        metrics["one_shot"] = True
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/terminal_eval.json"])

    raise RunnerError(f"unhandled stage {stage}")
