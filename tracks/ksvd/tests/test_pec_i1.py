"""Focused CPU tests for PEC-I1 (``pec_i1``).

The G0-G8 correctness gates live in ``pec_i1`` so the runner's ``correctness``
stage and this pytest module exercise exactly the same code.  Data-free: no
ZINC data, no checkpoints, no official test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import pec_i1 as i1
from tracks.ksvd.experiments.luyin16 import pec_v0 as pec
from tracks.ksvd.experiments.luyin16.pec_v0_gate0 import (
    _chain,
    _chem,
    _dictionary,
    _ring,
)

I1_GATES = i1.I1_GATES


@pytest.mark.parametrize("name", sorted(I1_GATES))
def test_i1_gates(name: str) -> None:
    result = I1_GATES[name]()
    if isinstance(result, dict):
        assert result.get("verdict") == "PASS"


def test_parameter_matching_within_one_percent() -> None:
    payload = i1.parameter_accounting(94049)
    assert payload["baseline_total_params"] == 94049
    assert payload["relative_diff"] <= 0.01
    assert payload["within_tolerance"]
    assert payload["reader_hidden"] == i1.I1_READER_HIDDEN
    assert payload["absolute_diff"] == 96


def test_s0_bucket_reindex() -> None:
    """PEC's 8-class distance one-hot re-indexes to S0's 5 buckets."""
    rho = torch.zeros((8, pec.TOPO_PAIR_DIM))
    for index in range(8):
        rho[index, index] = 1.0
    bucket = i1.pair_bucket_from_rho(rho)
    assert bucket.tolist() == [0, 1, 2, 3, 4, 4, 4, 4]


def test_pooling_empty_bucket_convention() -> None:
    values = torch.zeros((0, 3))
    index = torch.zeros(0, dtype=torch.long)
    pooled = i1.pool_mean_std_count(values, index, 2)
    assert pooled.shape == (2, 7)
    assert torch.allclose(pooled[0, :3], torch.zeros(3))
    assert torch.allclose(pooled[0, 3:6], torch.full((3,), float(np.sqrt(1e-8))))
    assert float(pooled[0, 6]) == 0.0


def test_environment_parameters_bit_identical_to_pec() -> None:
    d_node, d_edge = _dictionary(seed=0)
    old = pec.build_model("dense", d_node=d_node, d_edge=d_edge, seed=0)
    new = i1.build_i1_model("dense", d_node=d_node, d_edge=d_edge, seed=0)
    env_diff, pair_diff = i1.environment_pair_max_abs_diff(old, new)
    assert env_diff == 0.0
    assert pair_diff == 0.0


def test_stage_a_recoverability_chemistry_blocks_exact() -> None:
    graph = _ring(6)
    atom_types, edge_types = _chem(graph, (0, 1, 2, 3), 1)
    sample = pec.build_sample(
        graph, [atom_types[node] for node in graph.nodes], edge_types
    )
    records = [(graph, np.asarray([atom_types[n] for n in graph.nodes]), edge_types, sample)]
    report = i1.stage_a_recoverability(records)
    for name in ("atom_shell", "bond_shell", "root_atom", "incident"):
        block = report["blocks"][name]
        assert block["unexplained_coordinates"] == 0, (name, block)
        assert block["exact_fraction"] == 1.0
    # S0 (0,0) is structurally empty and absent from PEC's taxonomy
    assert report["totals"]["s0_00_edges"] == 0
    assert len(report["s0_shell_pairs"]) == 6
    assert len(report["pec_shell_pairs"]) == 5


def test_reader_width_constant() -> None:
    assert i1.i1_reader_input_width() == 590
    assert i1.I1_UNARY_WIDTH == 97
    assert i1.I1_PAIR_BUCKET_WIDTH == 97
