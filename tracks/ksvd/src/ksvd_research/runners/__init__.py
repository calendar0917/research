"""Runner registry.

The registry maps names to concrete runners.  The contract classes
(:class:`Runner`, :class:`RunnerError`) live in ``ksvd_research.runner_api``
so concrete runners can import them without importing the registry itself
(runners never import other runners).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from ksvd_research.runner_api import Runner, RunnerError
from ksvd_research.runtime.paths import REPO_ROOT


def _extra_runners() -> list[Runner]:
    """Runners contributed by another track (opt-in, default empty).

    ``RESEARCH_EXTRA_RUNNER_PATHS`` is an ``os.pathsep``-separated list of
    directories (absolute or repository-relative) prepended to ``sys.path``;
    ``RESEARCH_EXTRA_RUNNERS`` is a comma-separated list of module names that
    expose ``build_runner() -> Runner``.  Both unset (the default) reproduces
    the historical KSVD-only registry exactly.
    """
    for raw in os.environ.get("RESEARCH_EXTRA_RUNNER_PATHS", "").split(os.pathsep):
        entry = raw.strip()
        if not entry:
            continue
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    runners: list[Runner] = []
    for module_name in os.environ.get("RESEARCH_EXTRA_RUNNERS", "").split(","):
        name = module_name.strip()
        if not name:
            continue
        module = importlib.import_module(name)
        runners.append(module.build_runner())
    return runners


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
        zinc_cssd_basis_reuse_v1,
        zinc_cssd_bridge_compression_v1,
        zinc_cssd_consumer_generalization_v1,
        zinc_cssd_consumer_replacement_v1,
        zinc_cssd_g_component_attribution_v1,
        zinc_cssd_graph_information_sufficiency_v1,
        zinc_cssd_nonlinear_binding_v1,
        zinc_cssd_pair_joint_residual_v1,
        zinc_cssd_performance_triage_v1,
        zinc_cssd_relation_projection_v1,
        zinc_full_bottleneck_audit_v1,
        zinc_full_nonlinear_probe_v1,
        zinc_full_reader_wd_pilot_v1,
        zinc_graph_dictionary_readout_v1,
        zinc_icrate,
        zinc_jointbond_decay_diagnostic_v1,
        zinc_jointbond_v1,
        zinc_node_binding_residual_pair_v1,
        zinc_overnight_bottleneck_v1,
        zinc_patch_path_pooling,
        zinc_e2e_dictenv_typed_cycle_v1,
        zinc_upstream_portfolio_v1,
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
            zinc_cssd_basis_reuse_v1.build_runner(),
            zinc_cssd_bridge_compression_v1.build_runner(),
            zinc_cssd_consumer_generalization_v1.build_runner(),
            zinc_cssd_consumer_replacement_v1.build_runner(),
            zinc_cssd_g_component_attribution_v1.build_runner(),
            zinc_cssd_graph_information_sufficiency_v1.build_runner(),
            zinc_cssd_nonlinear_binding_v1.build_runner(),
            zinc_cssd_pair_joint_residual_v1.build_runner(),
            zinc_cssd_performance_triage_v1.build_runner(),
            zinc_cssd_relation_projection_v1.build_runner(),
            zinc_full_bottleneck_audit_v1.build_runner(),
            zinc_full_nonlinear_probe_v1.build_runner(),
            zinc_full_reader_wd_pilot_v1.build_runner(),
            zinc_e2e_dictenv_typed_cycle_v1.build_runner(),
            zinc_graph_dictionary_readout_v1.build_runner(),
            zinc_upstream_portfolio_v1.build_runner(),
            zinc_node_binding_residual_pair_v1.build_runner(),
            zinc_overnight_bottleneck_v1.build_runner(),
        ) + tuple(_extra_runners())
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
