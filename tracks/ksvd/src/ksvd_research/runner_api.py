"""The runner contract, imported by concrete runners.

A runner wraps one scientific entry point behind the uniform contract::

    run(config: dict, context: RunContext) -> RunResult

The CLI owns argparse, config loading, output paths, manifests, environment
capture and run directories; the runner owns the science.  Concrete runners
must not import ``ksvd_research.runners`` (the registry) — the dependency
direction is CLI -> registry -> runner, so registering a runner can never
pull another runner's science into this one.
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
