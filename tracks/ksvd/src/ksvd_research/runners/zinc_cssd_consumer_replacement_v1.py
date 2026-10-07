"""Runner: ZINC CSSD consumer replacement v1 (frozen basis supplies the M_COMP
consumer's local phi65 interface).

``model.stage``:

* ``source-objects`` verify and record every read-only reused object (fold,
                   targets, payload, kappa_M, prep, CSSD basis, frozen Q) from
                   the source round's results directory; build the round fold
                   view (fit 8001 / dev = sorted union(select, confirm)); CPU.
* ``checks``       focused pre-training correctness checks (CPU).
* ``probe``        fit-only interface probe (<=256 molecules by gid; the
                   historical M_COMP soup read-only diagnostic; CPU).
* ``smoke``        short GPU plumbing smoke (both arms x both seeds, <=4
                   batches per arm).
* ``train``        one 240-epoch body run (``model.arm`` in {RAW, DICT},
                   ``model.seed`` in {0, 1}; identical consumer and recipe,
                   only the local structure input differs).
* ``terminal-eval`` the one-shot terminal stage: roster -> dev scoring of all
                   four runs -> shared-group paired bootstrap -> FP32 noise
                   bound -> the unique alpha-mean dictionary intervention ->
                   (conditional) read-only localisation -> frozen A-D rules.

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
    zinc_cssd_consumer_replacement_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_cssd_consumer_replacement_v1.yaml"

STAGES = (
    "source-objects",
    "checks",
    "probe",
    "smoke",
    "train",
    "terminal-eval",
)


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_cssd_consumer_replacement_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Frozen-CSSD basis supplies the local phi65 interface of the strong M_COMP "
            "consumer: RAW (original phi) vs DICT (frozen full reconstruction "
            "c@U.T + alpha@Dbar.T), identical 297,539-parameter consumer and recipe, "
            "body seeds 0/1, four runs on the historical 8001-row fit fold; dev = sorted "
            "union of the old select/confirm rows (development comparison, not a new "
            "confirm); y_raw = ell_hat + s_hat + Q_raw with the frozen shared Q; terminal "
            "stage = one-shot dev scoring, shared-group paired bootstrap, the unique "
            "alpha-mean intervention, conditional localisation and frozen A-D rules; "
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
            "zinc_cssd_consumer_replacement_v1 is non-terminal; it must never run with test access granted"
        )
    model_cfg = config.get("model", {})
    stage = str(model_cfg.get("stage", "source-objects"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}, got {stage!r}")
    runtime = config.get("runtime", {})
    device_name = str(runtime.get("device", "cpu"))
    arm = str(model_cfg.get("arm", ""))
    if stage == "train" and arm not in stages.ARMS:
        raise RunnerError(f"stage train requires model.arm in {list(stages.ARMS)}, got {arm!r}")
    seed = int(model_cfg.get("seed", stages.SEED))
    if stage == "train" and seed not in stages.SEEDS:
        raise RunnerError(f"stage train requires model.seed in {list(stages.SEEDS)}, got {seed}")
    out_dir = stages.RESULTS_DIR
    threads = int(runtime.get("torch_threads", 8))
    print(
        f"[cssd-consumer-replacement-v1] stage={stage} arm={arm or '-'} "
        f"seed={seed} device={device_name}",
        flush=True,
    )

    def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
        base = {
            "measure": f"zinc_cssd_consumer_replacement_v1::{stage}",
            "stage": stage,
            "arm": arm or None,
            "seed": seed,
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        base.update({k: v for k, v in payload.items() if k in (
            "all_passed", "all_finite", "seconds", "steps_done", "stopped_reason",
            "one_shot", "fit_n", "dev_n",
        )})
        if "fit_diagnostics" in payload:
            base["fit_y_raw_mae"] = float(payload["fit_diagnostics"]["fit_y_raw_mae"])
        if "decision" in payload:
            base["branch"] = payload["decision"]["branch"]
            base["recommendation"] = payload["decision"]["recommendation"]
        return base

    if stage == "source-objects":
        result = stages.phase_source_manifest(out_dir=out_dir)
        _write_json(context.artifact_dir / "source_manifest.json", result)
        metrics = _metrics(result)
        metrics["fit_n"] = result["fold_view"]["fit"]["n"]
        metrics["dev_n"] = result["fold_view"]["dev"]["n"]
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/source_manifest.json"])

    if stage == "checks":
        checks = stages.run_checks(out_dir=out_dir)
        _write_json(context.artifact_dir / "checks.json", checks)
        metrics = _metrics(checks)
        metrics["all_passed"] = True
        _write_json(context.artifact_dir / "metrics.json", metrics)
        print("[cssd-consumer-replacement-v1] checks all_passed=True", flush=True)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/checks.json"])

    if stage == "probe":
        probe = stages.interface_probe(out_dir=out_dir)
        _write_json(context.artifact_dir / "interface_probe.json", probe)
        metrics = _metrics(probe)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/interface_probe.json"])

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
            seed,
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
