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
    from . import (
        zinc_bond_anchored_triple_joint_tune_v1_mainline,
        zinc_bond_anchored_triple_v1_mainline,
        zinc_e2e_dictenv_hier_relation_v1,
        zinc_e2e_dictenv_joint709_absolute_v1,
        zinc_e2e_dictenv_latent_bridge_v1,
        zinc_e2e_dictenv_rolecorr_increment_v2,
        zinc_e2e_dictenv_rolecorr_v1,
        zinc_e2e_dictenv_scale_v1,
        zinc_icrate,
        zinc_jointbond_decay_diagnostic_v1,
        zinc_jointbond_v1,
        zinc_patch_path_pooling,
        zinc_wg_icsc,
    )

    return {
        runner.name: runner
        for runner in (
            zinc_patch_path_pooling.build_runner(),
            zinc_wg_icsc.build_runner(),
            zinc_icrate.build_runner(),
            zinc_jointbond_v1.build_runner(),
            zinc_jointbond_decay_diagnostic_v1.build_runner(),
            zinc_bond_anchored_triple_v1_mainline.build_runner(),
            zinc_bond_anchored_triple_joint_tune_v1_mainline.build_runner(),
            zinc_e2e_dictenv_rolecorr_v1.build_runner(),
            zinc_e2e_dictenv_rolecorr_increment_v2.build_runner(),
            zinc_e2e_dictenv_joint709_absolute_v1.build_runner(),
            zinc_e2e_dictenv_hier_relation_v1.build_runner(),
            zinc_e2e_dictenv_latent_bridge_v1.build_runner(),
            zinc_e2e_dictenv_scale_v1.build_runner(),
        )
    }


def get_runner(name: str) -> Runner:
    try:
        return _registry()[name]
    except KeyError:
        available = ", ".join(sorted(_registry()))
        raise RunnerError(f"unknown runner {name!r}; available: {available}") from None


def list_runners() -> list[dict[str, Any]]:
    return [runner.info() for runner in _registry().values()]


__all__ = ["Runner", "RunnerError", "get_runner", "list_runners"]
