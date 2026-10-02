"""Focused tests for the typed-cycle static object (round typed_cycle_v1).

Pure layout / collate / parameter-contract checks run without data.  The model
containment test needs the frozen dictionary + common subspace and is skipped
when those local artifacts are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_typed_cycle_v1 as tc  # noqa: E402
from tracks.ksvd.experiments.luyin16 import typed_cycle_data_v1 as tcd  # noqa: E402
from tracks.ksvd.experiments.luyin16.typed_cycle_reference.v1_20261002 import (  # noqa: E402
    typed_cycle_reference as tcr,
)


def _ring_graph(sequence):
    n = len(sequence)
    edges = [(min(i, (i + 1) % n), max(i, (i + 1) % n), 0) for i in range(n)]
    return tcr.Graph(tuple(int(a) for a in sequence), tuple(sorted(edges)))


def test_assemble_views_matches_reference_exactly():
    graph = _ring_graph([0, 1, 2, 3, 1, 0])
    cycle = tcr.chordless_cycles(graph)[0]
    atom, bond, mask, length = tc.dihedral_view_components(graph, cycle)
    assembled = tc.assemble_views(
        torch.as_tensor(atom),
        torch.as_tensor(bond),
        torch.as_tensor(mask),
        torch.as_tensor(length),
    )
    reference = torch.as_tensor(tcr.dihedral_views(graph, cycle), dtype=torch.float32)
    assert assembled.shape == (2 * len(cycle), tc.RING_INPUT_DIM)
    assert torch.equal(assembled, reference)


def test_parameter_contract_closed_form():
    assert tc.RING_INPUT_DIM == 338
    assert tc.RING_POOL_DIM == 97
    assert tc.TYPED_READER_IN == 399
    assert tc.RING_ENCODER_PARAMETERS == 24816
    assert tc.READER_INCREMENT == 1261
    assert tc.EXPECTED_PARAMETERS == 133002


def test_typed_cycle_collate_offsets():
    from torch_geometric.data import Data

    def _molecule(n_nodes, n_cycles, views_per_cycle):
        data = Data()
        data.num_nodes = n_nodes
        data.env_occ_node = torch.zeros(0, dtype=torch.long)
        data.env_bond_root = torch.zeros(0, dtype=torch.long)
        total_views = n_cycles * views_per_cycle
        data.ring_view_atom = torch.zeros(total_views, tc.MAX_RING_LENGTH, dtype=torch.long)
        data.ring_view_bond = torch.zeros(total_views, tc.MAX_RING_LENGTH, dtype=torch.long)
        data.ring_view_mask = torch.ones(total_views, tc.MAX_RING_LENGTH)
        data.ring_view_length = torch.full((total_views,), 3, dtype=torch.long)
        data.ring_view_cycle = torch.arange(total_views, dtype=torch.long) // views_per_cycle
        data.ring_cycle_anchor = torch.zeros(n_cycles, dtype=torch.long)
        data.ring_cycle_length = torch.full((n_cycles,), 3, dtype=torch.long)
        return data

    a = _molecule(4, 2, 6)
    b = _molecule(5, 1, 6)
    batch = tc.typed_cycle_collate([a, b])
    # cycle ids: molecule a 0,1 ; molecule b 2
    assert batch.ring_view_cycle.tolist()[:6] == [0] * 6
    assert batch.ring_view_cycle.tolist()[6:12] == [1] * 6
    assert batch.ring_view_cycle.tolist()[12:] == [2] * 6
    # anchors: molecule a node offset 0, molecule b node offset 4
    assert batch.ring_cycle_anchor.tolist() == [0, 0, 4]


def _has_local_artifacts() -> bool:
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as base

    try:
        base._dictionary_tensor()
        base._load_parent_subspace()
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _has_local_artifacts(), reason="local dictionary / subspace artifacts missing")
def test_model_containment_and_zero_ring():
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as base

    subspace = base._load_parent_subspace()
    dictionary = base._dictionary_tensor()
    small = lb.build_latent_bridge_model(dictionary, 0, subspace).eval()
    typed = tc.build_typed_cycle_model(dictionary, 0, subspace).eval()
    assert sum(p.numel() for p in typed.parameters()) == tc.EXPECTED_PARAMETERS
    state = small.state_dict()
    for key, value in state.items():
        if key.startswith("reader."):
            continue
        assert torch.equal(typed.state_dict()[key], value), key
    # new reader columns start at zero.
    assert float(typed.reader.net[0].weight[:, tc.SMALL_READER_IN :].abs().max()) == 0.0
