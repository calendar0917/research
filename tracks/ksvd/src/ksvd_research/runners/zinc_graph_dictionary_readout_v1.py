"""Runner: zinc_graph_dictionary_readout_v1 (frozen Full prototype readout).

One candidate only: the **unchanged** existing Full m=3 seed-0 parameter soup
(408,651 parameters).  The old MLP reader is replaced, for the screen, by a
fixed graph-level prototype dictionary plus a certified convex L2-regularised
MAE head fitted on the frozen 814-D reader input ``R``.  No backbone parameter
is trained, no second backbone exists, no old prediction is added.

``model.stage=fit`` exports the frozen official-train representation and fits
the head (the official valid cache is not accepted by the fit stage).
``model.stage=evaluate`` opens the frozen official valid exactly once for the
paired direct screen, runs the positive-signal deployment acceptance and the
analysis.  The official ZINC **test** split is never instantiated; a granted
``test_access`` is refused.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping

from ksvd_research.runtime.fingerprints import zinc_fingerprints
from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, resolve_path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_graph_dictionary_readout_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_graph_dictionary_readout_v1.yaml"

ARTIFACTS = (
    "preflight.json",
    "evaluated.json",
    "evaluated.predictions.npz",
    "summary.json",
    "REPORT.md",
    "DECISION.md",
)


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_graph_dictionary_readout_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Frozen Full (408,651 params) + fixed K=256 graph-level prototype dictionary "
            "readout on the captured 814-D reader input; CPU-only, test blocked"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _stage(config: Mapping[str, Any]) -> str:
    stage = str(config.get("model", {}).get("stage", "fit"))
    if stage not in ("preflight", "export", "fit", "evaluate", "smoke", "deploy", "analysis", "chain"):
        raise ValueError(f"unsupported model.stage={stage!r}")
    return stage


def _copy_artifacts(context: RunContext) -> list[str]:
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ARTIFACTS:
        source = stages.RESULTS_DIR / name
        if not source.exists():
            continue
        target = context.artifact_dir / name
        if source.suffix == ".npz":
            target.write_bytes(source.read_bytes())
        else:
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(f"artifacts/{name}")
    for name, source in (
        ("provenance_train.json", stages.TRAIN_CACHE.with_suffix(".provenance.json")),
        ("provenance_valid.json", stages.VALID_CACHE.with_suffix(".provenance.json")),
        ("FIT.json", stages.FIT_JSON),
        ("scaffold_smoke.json", stages.SCAFFOLD_SMOKE_JSON),
        ("deploy.json", stages.DEPLOY_JSON),
    ):
        if source.exists():
            (context.artifact_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            copied.append(f"artifacts/{name}")
    return copied


def _metrics(stage: str) -> dict[str, Any]:
    summary_path = stages.RESULTS_DIR / "summary.json"
    eval_path = stages.EVAL_JSON
    metrics: dict[str, Any] = {
        "measure": "valid_mae",
        "stage": stage,
        "seed": int(stages.SEED),
        "device": "cpu",
        "split_sizes": {"train": 10_000, "valid": 1_000, "test": None},
        "official_valid_loaded": bool(eval_path.exists()),
        "official_test_loaded": False,
    }
    if eval_path.exists():
        evaluated = stages._read_json(eval_path)
        metrics.update(
            {
                "verdict": str(evaluated.get("verdict", "INCOMPLETE")),
                "completed": True,
                "original_valid_mae": float(evaluated.get("original_MAE", float("nan"))),
                "graph_dictionary_valid_mae": float(
                    evaluated.get("graph_dictionary_MAE", float("nan"))
                ),
                "valid_mae": float(evaluated.get("graph_dictionary_MAE", float("nan"))),
                "gain": float(evaluated.get("gain", float("nan"))),
                "better_fraction": float(evaluated.get("better_fraction", float("nan"))),
            }
        )
    else:
        metrics["verdict"] = "INCOMPLETE"
    if summary_path.exists():
        summary = stages._read_json(summary_path)
        metrics.update(
            {
                "selected_lambda": float(summary.get("selected_lambda", float("nan"))),
                "n_prototypes": int(summary.get("n_prototypes", 0)),
                "five_ID_bins_improved": int(summary.get("five_ID_bins_improved", 0)),
                "valid_replay_gate_passed": bool(summary.get("valid_replay_gate_passed", False)),
                "deployment_acceptance": str(summary.get("deployment_acceptance", "")),
                "passed": bool(summary.get("passed", False)),
            }
        )
    deploy_path = stages.DEPLOY_JSON
    if deploy_path.exists():
        deploy = stages._read_json(deploy_path)
        metrics["deployment_acceptance"] = str(
            deploy.get("deployment_acceptance", metrics.get("deployment_acceptance", ""))
        )
    return metrics


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "graph-dictionary-readout-v1 is a non-terminal screen; it must never run with "
            "test access granted"
        )
    stage = _stage(config)
    runtime = config.get("runtime", {})
    stages.configure(
        threads=int(runtime.get("torch_threads", stages.THREADS)),
        device=str(runtime.get("device", "cpu")),
    )
    print(f"[graph-dict] stage={stage} commit={stages._git_commit()}", flush=True)
    stages.run_stage(stage)
    artifacts = _copy_artifacts(context)
    metrics = _metrics(stage)
    (context.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    artifacts.append("artifacts/metrics.json")
    print(f"[graph-dict] verdict={metrics['verdict']}", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=artifacts)


__all__ = ["build_runner", "fingerprints", "run", "stages"]