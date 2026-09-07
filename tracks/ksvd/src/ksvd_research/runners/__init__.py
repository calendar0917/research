"""Runner registry.

A runner wraps one scientific entry point (typically a legacy experiment with
its own CLI) behind the uniform contract::

    run(config: dict, context: RunContext) -> RunResult

The CLI owns argparse, config loading, output paths, manifests, environment
capture and run directories; the runner owns the science.  Runners must not
import other runners (enforced by the architecture test in
``test_runner_dependencies.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ksvd_research.runtime.manifest import RunContext, RunResult


class RunnerError(RuntimeError):
    """Raised when a runner cannot execute for policy or setup reasons."""


class Runner:
    def __init__(
        self,
        name: str,
        default_config: Path,
        default_study: str | None,
        description: str,
        run_module: Any,
    ) -> None:
        self.name = name
        self.default_config = default_config
        self.default_study = default_study
        self.description = description
        self._run_module = run_module

    def run(self, config: dict[str, Any], context: RunContext) -> RunResult:
        return self._run_module.run(config, context)

    def fingerprints(self, config: dict[str, Any]) -> tuple[dict[str, Any], str]:
        return self._run_module.fingerprints(config)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "default_config": str(self.default_config),
            "default_study": self.default_study,
            "description": self.description,
        }


def _registry() -> dict[str, Runner]:
    from . import zinc_patch_path_pooling

    return {runner.name: runner for runner in (zinc_patch_path_pooling.build_runner(),)}


def get_runner(name: str) -> Runner:
    try:
        return _registry()[name]
    except KeyError:
        available = ", ".join(sorted(_registry()))
        raise RunnerError(f"unknown runner {name!r}; available: {available}") from None


def list_runners() -> list[dict[str, Any]]:
    return [runner.info() for runner in _registry().values()]
