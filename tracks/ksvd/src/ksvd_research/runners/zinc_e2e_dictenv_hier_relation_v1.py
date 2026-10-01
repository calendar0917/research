"""Runner: E2E-DictEnv-Hier-Relation-v1 (single-model hierarchical relation dictionary).

One candidate only: the node dictionary ``D_N`` (712 x 128, all coordinates kept,
no top-k, no message passing, no node write-back) feeding a within-molecule
same-atom-type shared environment ``mu`` / deviation ``delta`` relation object
(228 dims) into a sparse relation dictionary ``D_R`` (228 x 64, tied-IHT s=8) and
one shared graph readout (586 -> 64 -> 32 -> 1).

``model.stage=prepare`` builds the train-only node input, the unique physical
edge structure, the frozen ``P`` / scalers and the spectral dictionary
initialisation (scratch use).  ``model.stage=screen`` runs the correctness gates,
the smoke, the formal seed-0 320-epoch CPU training, the endpoint interventions
and the analysis.

This round has **no control arm**: no dense/PCA arm, no random or shuffled
dictionary arm, no second seed.  The official ZINC **test** split is never
instantiated; a granted ``test_access`` is refused by this runner.
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
    zinc_e2e_dictenv_hier_relation_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_e2e_dictenv_hier_relation_v1.yaml"

ARTIFACTS = (
    "source_verify.json",
    "hier_extra_scaler.json",
    "hier_node_input.json",
    "hier_structure.json",
    "align_check.json",
    "hier_relation_scaler.json",
    "hier_init.json",
    "correctness.json",
    "smoke.json",
    "run_HIER-RELATION.json",
    "curve_HIER-RELATION.csv",
    "soup_HIER-RELATION.json",
    "interventions.json",
    "summary.json",
    "REPORT.md",
    "DECISION.md",
)


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_e2e_dictenv_hier_relation_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Hier-Relation-v1: single candidate, one model, node dictionary (712x128) + "
            "within-molecule shared/deviation relation object (228) + sparse relation "
            "dictionary (228x64, tied-IHT s=8) + one 586-wide graph readout, CPU, test blocked"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _config_stage(config: Mapping[str, Any]) -> str:
    model = config.get("model", {})
    stage = str(model.get("stage", "screen"))
    if stage not in ("prepare", "screen", "all"):
        raise ValueError(f"model.stage must be 'prepare', 'screen' or 'all', got {stage!r}")
    return stage


def _copy_artifacts(context: RunContext) -> list[str]:
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ARTIFACTS:
        source = stages.RESULTS_DIR / name
        if not source.exists():
            continue
        (context.artifact_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(f"artifacts/{name}")
    return copied


def _metrics(stage: str, device: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "measure": "soup_valid_mae",
        "stage": stage,
        "seed": int(stages.SEED),
        "device": str(device),
        "d_x": int(stages.core.D_X),
        "node_atoms": int(stages.core.NODE_ATOMS),
        "relation_input_dim": int(stages.core.REL_INPUT_DIM),
        "relation_atoms": int(stages.core.REL_ATOMS),
        "relation_sparsity": int(stages.core.REL_SPARSITY),
        "relation_iht_steps": int(stages.core.REL_IHT_STEPS),
        "readout_dim": int(stages.core.READOUT_DIM),
        "split_sizes": {"train": 10_000, "valid": 1_000, "test": None},
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    summary_path = stages.RESULTS_DIR / "summary.json"
    if summary_path.exists():
        summary = stages._read_json(summary_path)
        band = summary.get("band", {})
        mechanism = summary.get("mechanism", {})
        metrics.update(
            {
                "verdict": str(summary.get("verdict", "INCOMPLETE")),
                "epochs": int(summary.get("epochs", stages.TRAIN_EPOCHS)),
                "epochs_run": int(summary.get("epochs_run", 0)),
                "completed": bool(summary.get("completed", False)),
                "valid_mae": float(summary.get("soup_valid_mae", float("nan"))),
                "valid_soup_mae": float(summary.get("soup_valid_mae", float("nan"))),
                "valid_best_mae": float(summary.get("best_valid_mae", float("nan"))),
                "best_epoch": int(summary.get("best_epoch", 0)),
                "absolute_band": str(band.get("band", "unknown")),
                "absolute_recommendation": str(band.get("recommendation", "")),
                "soup_train_mae": float(summary.get("soup_train_mae", float("nan"))),
                "train_valid_gap": float(summary.get("train_valid_gap", float("nan"))),
                "zero_relation_code_delta": float(
                    mechanism.get("interventions", {}).get("modes", {}).get("zero_beta", {}).get(
                        "delta_mae", float("nan")
                    )
                ),
                "zero_node_pool_delta": float(
                    mechanism.get("interventions", {})
                    .get("modes", {})
                    .get("zero_node_pool", {})
                    .get("delta_mae", float("nan"))
                ),
                "zero_delta_relation_delta": float(
                    mechanism.get("interventions", {})
                    .get("modes", {})
                    .get("zero_delta_rel", {})
                    .get("delta_mae", float("nan"))
                ),
                "node_channel_load_bearing": bool(mechanism.get("node_channel_load_bearing", False)),
                "relation_channel_load_bearing": bool(
                    mechanism.get("any_relation_channel_load_bearing", False)
                ),
                "valid_r_node": float(
                    summary.get("reconstruction", {}).get("valid_r_node", float("nan"))
                ),
                "valid_r_rel": float(
                    summary.get("reconstruction", {}).get("valid_r_rel", float("nan"))
                ),
                "correctness_all_passed": bool(summary.get("identity", {}).get("correctness_all_passed", False)),
                "smoke_passed": bool(summary.get("identity", {}).get("smoke_passed", False)),
                "trainable_parameters": int(summary.get("parameters", {}).get("trainable", 0)),
                "total_parameters": int(summary.get("parameters", {}).get("total_parameters", 0)),
                "wall_clock_s": float(summary.get("wall_clock_s", float("nan"))),
                "seconds_per_epoch": float(summary.get("seconds_per_epoch", float("nan"))),
                "peak_rss_bytes": int(summary.get("peak_rss_bytes", 0)),
            }
        )
    else:
        metrics["verdict"] = "PREPARE_OR_INCOMPLETE"
    return metrics


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "Hier-Relation-v1 is a non-terminal screen; it must never run with test access granted"
        )
    stage = _config_stage(config)
    runtime = config.get("runtime", {})
    device = str(runtime.get("device", "cpu"))
    configured = stages.configure(
        epochs=int(config.get("model", {}).get("epochs", stages.TRAIN_EPOCHS)),
        threads=int(runtime.get("torch_threads", stages.THREADS)),
        device=device,
    )
    print(
        f"[hier-rel] stage={stage} device={configured['device']} "
        f"epochs={configured['epochs']} threads={configured['threads']} "
        f"commit={stages._git_commit()}",
        flush=True,
    )
    if stage in ("prepare", "all"):
        stages.run_stages("prepare")
    if stage in ("screen", "all"):
        stages.run_stages("screen")
    artifacts = _copy_artifacts(context)
    metrics = _metrics(stage, configured["device"])
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    artifacts.append("artifacts/metrics.json")
    print(f"[hier-rel] verdict={metrics['verdict']}", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=artifacts)


__all__ = ["build_runner", "fingerprints", "run", "stages"]
