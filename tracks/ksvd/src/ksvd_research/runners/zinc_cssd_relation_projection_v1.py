"""Runner: ZINC CSSD relation projection v1 (frozen basis + access; does
letting the EXISTING distance buckets select the endpoint projection — before
the local representation is compressed — improve the DICT consumer's complete
y and g?  Single body seed 0, three arms, one training each).

``model.stage``:

* ``source-checks``      verify every read-only reused object, the historical
                         DICT seed-0 run, the schedule/init contracts, the
                         dimension/bucket contracts and the per-arm parameter
                         contracts (BASE reproduces the historical init
                         bit-for-bit; C/R share every non-projection tensor
                         and swap only the endpoint projection under a
                         fork_rng private seed 20261008); CPU.
* ``pretrain-checks``    the meaningful fit-only pre-training checks (BASE
                         forward parity bit-exact vs the untouched factory,
                         C(E)=P0E identity init, R=P0 bitwise copies, step-0
                         agreement, endpoint-swap/pair-order/node-relabel/
                         batch invariances, gradient smoke, save/reload
                         replay, route-override wiring); CPU.
* ``smoke``              one 1-epoch end-to-end trajectory per arm through the
                         full train path (fit rows only); GPU.
* ``train``              one from-scratch 240-epoch training of one arm x
                         seed 0 (``model.arm`` in {BASE, SHARED_MLP,
                         BUCKET_LINEAR}); 15,120 steps asserted; members
                         236..240 captured; estimator = their equal-weight
                         FP32 mean; GPU.
* ``terminal-eval``      the ONE-SHOT frozen comparison: roster verification,
                         full fit+dev scoring with per-row exports, the three
                         paired contrasts (R-BASE, C-BASE, R-C) on y_raw/g,
                         group-paired bootstrap (2000 draws, seed 20261008,
                         shared picks across arms and metrics), the noise
                         bound (repeat + save/reload replay), the unique
                         alpha-mean dictionary intervention on all three
                         soups and the route-shuffle intervention with ONE
                         fixed override set shared by all arms, then the
                         frozen single-seed exploratory reading.

Single body seed: a directional signal only — never a repeatability or
method-stability claim.  Official ZINC **valid** and **test** are never
instantiated; a granted ``test_access`` is refused.
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
    zinc_cssd_relation_projection_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_cssd_relation_projection_v1.yaml"

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
        name="zinc_cssd_relation_projection_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Frozen-CSSD basis + access held fixed (the verified DICT consumer): does "
            "the relation side read the endpoint projection BEFORE the local "
            "representation is compressed?  Three arms x body seed 0, ONE from-scratch "
            "240-epoch training each (BASE = the historical shared no-bias "
            "Linear(144,48), 297,539 params; SHARED_MLP = a shared no-bias "
            "144->180->48 SiLU MLP with a fixed function-preserving init C(E)=P0E, "
            "325,187 params; BUCKET_LINEAR = five no-bias P_b[48,144], one per "
            "EXISTING distance bucket, each a bitwise copy of P0, 325,187 params; the "
            "relation input, distance gate, pair readout and Q keep data.pair_bucket; "
            "only the endpoint views are routed).  Single estimator mean(236..240); "
            "one-shot terminal eval with per-row fit/dev exports, the paired contrasts "
            "R-BASE / C-BASE / R-C on y_raw and g, group-paired bootstrap (2000 draws, "
            "seed 20261008, shared picks), the noise bound, the unique alpha-mean "
            "dictionary intervention and the route-shuffle intervention (one fixed "
            "override set for all arms), then the frozen single-seed exploratory "
            "reading (directional signal only — no repeatability claim, the B72 A-E "
            "gates are NOT inherited); dev = the historical development rows "
            "(repeatedly used — never an independent confirm); official valid+test "
            "blocked"
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
            "zinc_cssd_relation_projection_v1 is non-terminal; it must never run with test access granted"
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
        f"[cssd-relation-projection-v1] stage={stage} arm={arm} seed={seed} device={device_name}",
        flush=True,
    )

    def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
        base = {
            "measure": f"zinc_cssd_relation_projection_v1::{stage}",
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
