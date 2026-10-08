"""Runner: ZINC CSSD pair joint residual v1 (keeping C's shared endpoint
projection and the existing pair path, does a SHARED, endpoint-swap-symmetric
joint residual reading BOTH full endpoints and the existing relation improve
the complete y_raw and g?  Single body seed 0, TWO arms, one training each).

``model.stage``:

* ``source-checks``      verify every read-only reused object, the historical
                         DICT seed-0 run, the previous round's SHARED_MLP s0
                         anchor, the schedule/init contracts, the dimension/
                         bucket contracts and the per-arm parameter contracts
                         (CTRL_C reproduces the previous round's C s0 init
                         bit-for-bit; JOINT shares every non-joint tensor and
                         attaches its module under a fork_rng private seed
                         202610081 with W2 exactly zero); CPU.
* ``pretrain-checks``    the meaningful fit-only pre-training checks (CTRL_C
                         forward parity bit-exact vs the previous C factory,
                         joint module contracts, endpoint-swap symmetry, the
                         2-D collision demo, step-0 agreement, invariances,
                         mask/fill wiring of the new branch, gradient smoke,
                         save/reload replay, the joint_enabled=False switch
                         equivalence); CPU.
* ``smoke``              one 1-epoch end-to-end trajectory per arm through the
                         full train path (fit rows only); GPU.
* ``train``              one from-scratch 240-epoch training of one arm x
                         seed 0 (``model.arm`` in {CTRL_C, JOINT_RESIDUAL});
                         15,120 steps asserted; members 236..240 captured;
                         estimator = their equal-weight FP32 mean; GPU.
* ``terminal-eval``      the ONE-SHOT frozen comparison: roster verification,
                         full fit+dev scoring with per-row exports, the single
                         paired contrast JOINT-CTRL_C on y_raw/g (dev + fit,
                         relative deltas, gaps), the group-paired bootstrap
                         (2000 draws, seed 20261008, shared picks across both
                         arms and both metrics), the noise bound, the
                         alpha-mean dictionary intervention on both soups and
                         the joint_enabled=False toggle on the JOINT soup
                         (CTRL_C's switch no-op verified), then the frozen
                         single-seed exploratory reading.

Single body seed: a performance-lead screen only — never a repeatability,
mechanism or paired-information-recovery claim; the historical 0.003 gates
and A-E classes are NOT inherited.  Official ZINC **valid** and **test** are
never instantiated; a granted ``test_access`` is refused.
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

from ksvd_research.runner_api import Runner, RunnerError  # noqa: E402

from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_cssd_pair_joint_residual_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_cssd_pair_joint_residual_v1.yaml"

STAGES = (
    "source-checks",
    "pretrain-checks",
    "smoke",
    "train",
    "terminal-eval",
)
SEED_STAGES = ("train",)
ARM_STAGES = ("train",)


def build_runner():
    return Runner(
        name="zinc_cssd_pair_joint_residual_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Frozen-CSSD basis + access held fixed; keeping C's shared endpoint "
            "projection and the existing pair path, does a SHARED, "
            "endpoint-swap-symmetric joint residual reading BOTH full endpoints "
            "and the existing relation (psi: no-bias 144->64->48, "
            "delta=0.5*(psi([l,r,rel])+psi([r,l,rel])), h_pair=h_old+delta, "
            "W2 init exactly zero, private seed 202610081) improve the complete "
            "y and g?  TWO arms x body seed 0, ONE from-scratch 240-epoch "
            "training each (CTRL_C = the previous round's untrained C factory, "
            "325,187 params; JOINT_RESIDUAL = +12,288 params = 337,475, +3.78%; "
            "same factory/data/schedule; old h_old path verbatim; no new pool, "
            "no pair->node write-back).  Single estimator mean(236..240); "
            "one-shot terminal eval with per-row fit/dev exports, the single "
            "paired contrast JOINT-CTRL_C on y_raw and g, group-paired "
            "bootstrap (2000 draws, seed 20261008, shared picks), the noise "
            "bound, the alpha-mean dictionary intervention on both soups and "
            "the joint_enabled=False toggle (delta:=0), then the frozen "
            "single-seed exploratory reading (performance-lead screen only — "
            "no repeatability/mechanism claim, the 0.003 gates and A-E classes "
            "NOT inherited); dev = the historical development rows "
            "(repeatedly used — never an independent confirm); official "
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
            "zinc_cssd_pair_joint_residual_v1 is non-terminal; it must never run with test access granted"
        )
    model_cfg = config.get("model", {})
    stage = str(model_cfg.get("stage", "source-checks"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}, got {stage!r}")
    runtime = config.get("runtime", {})
    device_name = str(runtime.get("device", "cpu"))
    seed = int(model_cfg.get("seed", stages.SEEDS[0]))
    arm = str(model_cfg.get("arm", stages.CONTROL))
    if stage in SEED_STAGES and seed not in stages.SEEDS:
        raise RunnerError(f"stage {stage} requires model.seed in {list(stages.SEEDS)}, got {seed}")
    if stage in ARM_STAGES and arm not in stages.ARMS:
        raise RunnerError(f"stage {stage} requires model.arm in {list(stages.ARMS)}, got {arm}")
    out_dir = stages.RESULTS_DIR
    print(
        f"[cssd-pair-joint-residual-v1] stage={stage} arm={arm} seed={seed} device={device_name}",
        flush=True,
    )

    def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
        base = {
            "measure": f"zinc_cssd_pair_joint_residual_v1::{stage}",
            "stage": stage,
            "arm": arm,
            "seed": seed,
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        base.update({k: v for k, v in payload.items() if k in (
            "all_passed", "seconds", "steps_done", "stopped_reason", "one_shot",
            "fit_n", "dev_n", "curve_seconds_total", "wall_clock_s",
        )})
        if "exploratory_reading" in payload:
            base["reading_branch"] = payload["exploratory_reading"]["branch"]
        return base

    if stage == "source-checks":
        result = stages.source_checks(out_dir=out_dir)
        _write_json(context.artifact_dir / "source_manifest.json", result)
        metrics = _metrics(result)
        metrics["fit_n"] = result["dim_contract"]["fit_n"]
        metrics["dev_n"] = result["dim_contract"]["dev_n"]
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/source_manifest.json"])

    if stage == "pretrain-checks":
        result = stages.pretrain_checks(out_dir=out_dir)
        _write_json(context.artifact_dir / "pretrain_checks.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/pretrain_checks.json"])

    if stage == "smoke":
        result = stages.run_smoke(device_name=device_name, out_dir=out_dir)
        _write_json(context.artifact_dir / "smoke.json", result)
        metrics = _metrics(result)
        _write_json(context.artifact_dir / "metrics.json", metrics)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/smoke.json"])

    if stage == "train":
        import torch

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable; no silent CPU fallback")
        result = stages.train_run(arm, seed, device_name=device_name, out_dir=out_dir)
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
