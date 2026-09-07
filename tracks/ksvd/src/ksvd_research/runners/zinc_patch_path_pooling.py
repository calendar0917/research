"""Strangler runner for the active ZINC patch--path pooling experiment.

The scientific algorithm remains the legacy module
``tracks/ksvd/experiments/luyin16/zinc_patch_path_pooling``; this wrapper
owns only the *plumbing* contract: it receives an already-resolved config and
a :class:`RunContext` and returns a :class:`RunResult`.  The CLI never passes
output paths in the config — this runner injects them into the legacy
temporary config so legacy code keeps working unchanged.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping

from ksvd_research.runtime.config import apply_override
from ksvd_research.runtime.fingerprints import zinc_fingerprints
from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, resolve_path
from ksvd_research.runtime.serialization import write_yaml_atomic

PATCH_RADIUS_2_SHELL_WIDTH = 143
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_patch_path_pooling.yaml"
EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}


def _legacy_module():
    """Import the legacy experiment module independent of ``cwd``."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling

    return zinc_patch_path_pooling


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_patch_path_pooling",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "ZINC exact rooted typed patch tokens with shortest-path-conditioned "
            "pair pooling (legacy luyin16 experiment)"
        ),
        run_module=sys.modules[__name__],
    )


def check_test_access_blocked(protocol: Mapping[str, Any] | None, mode: str) -> bool:
    """Deprecated: kept only for legacy callers of the old semantics.

    The policy now belongs to the control plane
    (``ksvd_research.runtime.policy.resolve_test_access``).  The runner only
    obeys ``context.test_access``; it no longer inspects test_policy.
    """
    if not protocol:
        return False
    return protocol.get("test_policy") == "terminal" and mode != "terminal"


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _inject_policy(config: dict[str, Any], blocked: bool) -> dict[str, Any]:
    if blocked:
        return apply_override(config, "test_policy", "no_test")
    return config


def _metrics_from_legacy(result: Mapping[str, Any], blocked: bool) -> dict[str, Any]:
    evaluation = result["evaluation"]
    test = evaluation["test_after_train_valid_refit"]
    metrics: dict[str, Any] = {
        "valid_mae": float(evaluation["valid"]["mae"]),
        "valid_best_mae": (
            None
            if evaluation["valid"]["best_mae"] is None
            else float(evaluation["valid"]["best_mae"])
        ),
        "valid_selected_epoch": int(evaluation["valid"]["selected_epoch"]),
        "valid_epochs_run": int(evaluation["valid"]["epochs_run"]),
        "parameters": int(evaluation["parameters"]),
        "runtime_seconds": float(result["runtime"]["seconds"]),
        "split_sizes": result["data"]["sizes"],
        "test_access": "blocked" if blocked else "granted",
    }
    if test.get("mae") is not None:
        metrics["test_after_train_valid_refit_mae"] = float(test["mae"])
        metrics["test_epochs_run"] = int(test["epochs_run"])
    return metrics


def run(config: dict[str, Any], context: RunContext) -> RunResult:
    # The control plane has already resolved access (terminal rule, mode,
    # protocol); the runner only obeys the instruction recorded in context.
    blocked = context.test_access == "blocked"
    legacy_config = _inject_policy(copy.deepcopy(config), blocked)
    legacy_config = apply_override(legacy_config, "output.json", str(context.artifact_dir / "legacy_full_result.json"))
    legacy_config = apply_override(legacy_config, "output.markdown", str(context.artifact_dir / "legacy_result.md"))
    legacy_input = Path(context.run_dir) / "legacy_input.yaml"
    write_yaml_atomic(legacy_input, legacy_config)
    legacy = _legacy_module()
    result = legacy.run(legacy_input)
    metrics = _metrics_from_legacy(result, blocked)
    return RunResult(
        metrics=metrics,
        status="completed" if result["status"] == "completed" else "failed",
        artifacts=[
            "legacy_input.yaml",
            "artifacts/legacy_full_result.json",
            "artifacts/legacy_result.md",
        ],
    )
