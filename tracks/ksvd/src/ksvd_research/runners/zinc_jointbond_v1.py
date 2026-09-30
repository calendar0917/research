"""Runner: JointBond-v1 key-level joint structure-semantics fusion on ZINC.

Implements ``tracks/ksvd/notes/e2e_dictenv_jointbond_v1_preregistration.md``: the
frozen ``CSSD-Sem108`` parent plus one additive residual branch per real bond
that binds **that endpoint's** residual dictionary code to **that endpoint's**
atom type and fuses both endpoints with the bond type before the shellpair
``index_add_`` aggregation.

This runner is the control-plane entry point: it drives the round's stage module
(``tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_jointbond_v1.py``) exactly as
pre-registered (Phase-A audit -> preflight -> correctness -> <=64-step smoke ->
one seed-0 320-epoch run -> frozen inference probes -> analysis).

CPU only.  The official ZINC **test** split is never loaded; a granted
``test_access`` is refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from ksvd_research.runtime.fingerprints import zinc_fingerprints
from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, resolve_path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_jointbond_v1 as jb  # noqa: E402
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_jointbond_v1 as stages  # noqa: E402

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_jointbond_v1.yaml"


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_jointbond_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description="JointBond-v1 key-level joint structure-semantics fusion (ZINC, CPU)",
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "JointBond-v1 is a validation-only round; it must never run with test access granted"
        )
    if not torch.cuda.is_available():
        torch.set_num_threads(int(config.get("runtime", {}).get("torch_threads", stages.THREADS)))

    stages.stage_references()
    stages.stage_audit()
    stages.stage_preflight()
    stages.stage_correctness()
    stages.stage_smoke()
    stages.stage_train()
    stages.stage_interventions()
    summary = stages.stage_analysis()

    preflight = stages._read_json(stages.RESULTS_DIR / "preflight.json")
    health = stages._read_json(stages.MECHANISM_DIR / "dictionary_health.json")
    correctness = stages._read_json(stages.RESULTS_DIR / "correctness.json")
    smoke = stages._read_json(stages.RESULTS_DIR / "smoke.json")
    run_seed0 = stages._read_json(stages.RESULTS_DIR / "run_seed0.json")

    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    (context.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for source, name in (
        (stages.RESULTS_DIR / "correctness.json", "correctness.json"),
        (stages.RESULTS_DIR / "smoke.json", "smoke.json"),
        (stages.RESULTS_DIR / "preflight.json", "preflight.json"),
        (stages.RESULTS_DIR / "parameter_audit.json", "parameter_audit.json"),
        (stages.RESULTS_DIR / "soup.json", "soup.json"),
        (stages.RESULTS_DIR / "REPORT.md", "REPORT.md"),
        (stages.RESULTS_DIR / "DECISION.md", "DECISION.md"),
    ):
        if source.exists():
            (context.artifact_dir / name).write_text(
                Path(source).read_text(encoding="utf-8"), encoding="utf-8"
            )

    metrics: dict[str, Any] = {
        "valid_mae": float(summary["M_S"]),
        "valid_soup_mae": float(summary["M_S"]),
        "valid_best_mae": float(summary["best_valid_mae"]),
        "best_epoch": int(summary["best_epoch"]),
        "epochs_run": int(summary["epochs"]),
        "parameters": int(summary["parameters"]["candidate"]),
        "new_parameters": int(summary["parameters"]["new"]),
        "parent_parameters": int(summary["parameters"]["parent"]),
        "runtime_seconds": float(summary["wall_clock_s"]),
        "performance_band": str(summary["performance_band"]),
        "G_branch_off": float(summary["branch_mechanism"]["G_branch_off"]),
        "G_joint_alpha_shuffle": float(summary["branch_mechanism"]["G_joint_alpha_shuffle"]),
        "G_joint_atom_shuffle": float(summary["branch_mechanism"]["G_joint_atom_shuffle"]),
        "G_joint_alpha0": float(summary["branch_mechanism"]["G_joint_alpha0"]),
        "G_dict0": float(summary["parent_mechanism"]["G_dict0"]),
        "G_node": float(summary["parent_mechanism"]["G_node"]),
        "G_edge": float(summary["parent_mechanism"]["G_edge"]),
        "endpoint_correspondence": str(summary["branch_mechanism"]["endpoint_correspondence"]),
        "branch_used": bool(summary["gates"]["branch_used"]),
        "branch_dead": bool(summary["gates"]["branch_dead"]),
        "node_dictionary_binding_dead": bool(
            summary["parent_mechanism"]["node_dictionary_binding_dead"]
        ),
        "buy_matched_control": bool(summary["decision"]["buy_matched_control"]),
        "alpha_active_atoms": int(health["active_atoms"]),
        "alpha_effective_atoms": float(health["effective_atoms"]),
        "alpha_usage_top1_share": float(health["usage_top1_share"]),
        "correctness_all_passed": bool(correctness["all_passed"]),
        "smoke_passed": bool(smoke["passed"]),
        "parameter_contract_exact": bool(
            preflight["parameter_audit"]["new_params_exact"]
            and preflight["parameter_audit"]["total_exact"]
        ),
        "split_sizes": {"train": 10_000, "valid": 1_000, "test": None},
        "test_access": context.test_access,
        "official_test_loaded": False,
        "seed": int(summary["seed"]),
        "soup_members": list(summary["soup_members"]),
    }
    (context.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        f"[jointbond] M_S={metrics['valid_soup_mae']:.6f} band={metrics['performance_band']} "
        f"G_off={metrics['G_branch_off']:+.6f} buy={metrics['buy_matched_control']}",
        flush=True,
    )
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=[
            "artifacts/summary.json",
            "artifacts/metrics.json",
            "artifacts/correctness.json",
            "artifacts/smoke.json",
            "artifacts/preflight.json",
            "artifacts/parameter_audit.json",
            "artifacts/soup.json",
            "artifacts/REPORT.md",
            "artifacts/DECISION.md",
        ],
    )


__all__ = ["build_runner", "fingerprints", "run", "jb"]
