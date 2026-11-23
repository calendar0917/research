"""FGCR-v0 exact response extraction on top of the vendored BAMF junction tree.

The vendored ``bamf_inference.py`` is byte-identical to the validated BAMF-v1
source (SHA-256 recorded in PROTOCOL.md).  This module adds exactly one thing:
a single call to ``_exact_beliefs`` from which both

    log Z_r(G)              (per graph, per channel)
    logw[v, r, 0/1]         (per atom, per channel, unnormalised log weights)

are derived.  There is no second inference pass, no truncation, no sampling and
no approximate message passing.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch

import bamf_inference as B
from bamf_inference import (  # re-exported for the rest of FGCR
    BatchPlan,
    GlobalPlan,
    build_global_plan,
    edge_log_potential,
    min_fill_width,
    _exact_beliefs,
)

__all__ = [
    "BatchPlan",
    "GlobalPlan",
    "build_corpus_plan",
    "edge_log_potential",
    "min_fill_width",
    "exact_responses",
    "logz_from_belief",
    "node_logweights_from_belief",
    "plan_stats",
    "BELIEF_CALLS",
]

#: Diagnostic counter.  ``exact_responses`` must increment this by exactly one.
BELIEF_CALLS = {"count": 0}


def build_corpus_plan(graphs: Sequence[tuple[int, Sequence[tuple[int, int]]]],
                      *, smax: int = 4, verbose: bool = False) -> GlobalPlan:
    """Build one exact junction plan per graph over the whole corpus."""
    t0 = time.perf_counter()
    plan = build_global_plan(list(graphs), smax=int(smax), verbose=verbose)
    plan.build_seconds = time.perf_counter() - t0  # type: ignore[attr-defined]
    return plan


def logz_from_belief(belief: torch.Tensor, valid: torch.Tensor,
                     bp: BatchPlan) -> torch.Tensor:
    """Per-graph exact log partition function, [G, C].

    A calibrated root clique belief is the log joint factor over that root
    clique and its component.  Its log-sum-exp over valid assignments is the
    component partition function; independent components add in log space.
    """
    if bp.n_cliques == 0:
        return belief.new_zeros((bp.n_graphs, belief.shape[1]))
    root = (bp.parent < 0).nonzero(as_tuple=False).reshape(-1)
    if root.numel() == 0:
        return belief.new_zeros((bp.n_graphs, belief.shape[1]))
    b = belief.index_select(0, root)
    v = valid.index_select(0, root)
    neg = torch.finfo(b.dtype).min
    root_logz = torch.logsumexp(b.masked_fill(~v.unsqueeze(1), neg), dim=-1)
    out = belief.new_zeros((bp.n_graphs, belief.shape[1]))
    out.index_add_(0, bp.clique_graph.index_select(0, root), root_logz)
    return out


def node_logweights_from_belief(belief: torch.Tensor, valid: torch.Tensor,
                                states: torch.Tensor,
                                bp: BatchPlan) -> torch.Tensor:
    """Per-atom unnormalised log weights, [N, C, 2].

    The two entries are ``logw(0)`` and ``logw(1)`` from the calibrated clique
    that owns each atom.  Their difference is the exact posterior log-odds.
    """
    if bp.n_atoms == 0:
        return belief.new_zeros((0, belief.shape[1], 2))
    c = bp.node_clique.to(torch.long)
    b = belief.index_select(0, c)
    v = valid.index_select(0, c)
    bit1 = ((states.unsqueeze(0) >> bp.node_pos.unsqueeze(1)) & 1).bool()
    m1 = bit1 & v
    m0 = (~bit1) & v
    neg = torch.finfo(b.dtype).min
    lw1 = torch.logsumexp(b.masked_fill(~m1.unsqueeze(1), neg), dim=-1)
    lw0 = torch.logsumexp(b.masked_fill(~m0.unsqueeze(1), neg), dim=-1)
    return torch.stack([lw0, lw1], dim=-1)


def exact_responses(log_phi: torch.Tensor, log_psi: torch.Tensor,
                    bp: BatchPlan) -> dict[str, torch.Tensor]:
    """One exact belief pass -> logZ and node log weights.

    Parameters
    ----------
    log_phi : [N, C, 2] unary log potentials
    log_psi : [M, C, 2, 2] symmetric edge log potentials
    bp : BatchPlan

    Returns
    -------
    dict with ``belief`` [K,C,S], ``logz`` [G,C], ``logw`` [N,C,2].
    """
    belief, valid, states = _exact_beliefs(log_phi, log_psi, bp)
    BELIEF_CALLS["count"] += 1
    return {
        "belief": belief,
        "logz": logz_from_belief(belief, valid, bp),
        "logw": node_logweights_from_belief(belief, valid, states, bp),
    }


def plan_stats(plan: GlobalPlan) -> dict[str, Any]:
    widths = np.asarray(plan.widths.detach().cpu().numpy(), dtype=np.int64)
    scope_sizes = np.asarray(plan.clique_scope_size.detach().cpu().numpy())
    n_cliques = int(plan.graph_clique_ptr[-1].item())
    n_atoms = int(plan.graph_node_ptr[-1].item())
    n_bonds = int(plan.graph_edge_ptr[-1].item())
    return {
        "n_graphs": int(plan.n_graphs),
        "n_cliques": n_cliques,
        "n_atoms": n_atoms,
        "n_physical_bonds": n_bonds,
        "smax": int(plan.smax),
        "max_clique_size": int(plan.max_clique_size),
        "max_width": int(plan.max_width),
        "width_median": float(np.median(widths)),
        "width_p95": float(np.percentile(widths, 95)),
        "width_p99": float(np.percentile(widths, 99)),
        "width_max": int(widths.max()),
        "width_counts": {int(k): int(v) for k, v in
                         zip(*np.unique(widths, return_counts=True))},
        "clique_state_table_max": int(2 ** int(scope_sizes.max())),
        "clique_state_table_smax": int(2 ** int(plan.smax)),
    }