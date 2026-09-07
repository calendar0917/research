"""Runner registry.

The registry maps names to concrete runners.  The contract classes
(:class:`Runner`, :class:`RunnerError`) live in ``ksvd_research.runner_api``
so concrete runners can import them without importing the registry itself
(runners never import other runners).
"""

from __future__ import annotations

from typing import Any

from ksvd_research.runner_api import Runner, RunnerError


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


__all__ = ["Runner", "RunnerError", "get_runner", "list_runners"]
