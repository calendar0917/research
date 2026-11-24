"""Correctness tests for GSRO v0.

The heavy numerical checks live in ``gsro_preflight.run_checks`` (the same code
that produces the formal ``checks.json``); these tests make sure that gate stays
green and that the reused data identity is what the protocol names.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import gsro_common as C
import gsro_data as D
from gsro_model import GSROModel, full_response, local2_response, masked_diag
from gsro_preflight import _numpy_reference, prediction_decomposition, run_checks


@pytest.fixture(scope="module")
def config():
    return C.load_config()


@pytest.fixture(scope="module")
def prep(config):
    return D.load_prepared(config)


def test_data_identity(prep, config):
    assert prep.payload_sha256 == config["cache_sha256"]
    assert [len(prep.splits[name].graphs) for name in D.SPLITS] == [3360, 420, 420]
    assert prep.atom_vocab_sizes[0] == 12  # 11 Train atomic numbers + unseen
    assert len(prep.joint_pairs) == 22
    assert prep.n_joint_features == 23
    assert 26.0 < prep.n_bar < 28.0
    # Train-only target standardisation.
    train_y = prep.splits["train"].y_std
    assert abs(float(np.mean(train_y))) < 1e-12
    assert abs(float(np.std(train_y)) - 1.0) < 1e-6


def test_full_matches_independent_fp64(prep, config):
    model = GSROModel(prep, config, "full").to(dtype=torch.float64)
    graph = sorted(prep.splits["train"].graphs, key=lambda g: (g.n, g.row_id))[1]
    batch = D.single_graph(graph, "cpu", dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        s, r, w = model.generate(batch)
        q = full_response(
            s, r, w, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
        )
        q_local = local2_response(
            s, r, w, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
        )
    reference = _numpy_reference(
        s.numpy()[0], r.numpy()[0], w.numpy()[0], graph.bu, graph.bv
    )
    assert np.allclose(q.numpy()[0], reference["q_full"], rtol=0, atol=1e-10)
    assert np.allclose(q_local.numpy()[0], reference["q_local2"], rtol=0, atol=1e-10)
    # Q0/Q1/Q2 are the three polynomial orders.
    _, parts = local2_response(
        s, r, w, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"],
        return_parts=True,
    )
    assert np.allclose(parts["Q0"].numpy()[0], reference["Q0"], atol=1e-10)
    assert np.allclose(parts["Q1"].numpy()[0], reference["Q1"], atol=1e-10)
    assert np.allclose(parts["Q2"].numpy()[0], reference["Q2"], atol=1e-10)


def test_w_zero_full_equals_local2(prep, config):
    model = GSROModel(prep, config, "full").to(dtype=torch.float64)
    graph = sorted(prep.splits["train"].graphs, key=lambda g: (g.n, g.row_id))[3]
    batch = D.single_graph(graph, "cpu", dtype=torch.float64)
    with torch.no_grad():
        s, r, _ = model.generate(batch)
        w0 = torch.zeros_like(_)
        q_full = full_response(
            s, r, w0, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
        )
        q_local = local2_response(
            s, r, w0, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
        )
    assert torch.allclose(q_full, q_local, atol=1e-12, rtol=0)


def test_decomposition_matches_prediction(prep, config):
    model = GSROModel(prep, config, "full").to(dtype=torch.float64)
    graph = sorted(prep.splits["train"].graphs, key=lambda g: (g.n, g.row_id))[5]
    batch = D.single_graph(graph, "cpu", dtype=torch.float64)
    decomp = prediction_decomposition(model, batch)
    assert float(decomp["reconstruction_abs_error"].abs().max()) < 1e-9
    # The padded/identity conventions keep padding inert.
    padded = D.collate(
        [prep.splits["train"].graphs[i] for i in (0, 2, 4)], "cpu"
    )
    with torch.no_grad():
        s, r, w = model.generate(padded)
        d, _, _ = masked_diag(
            r, w, padded["bu"], padded["bv"], padded["node_mask"], padded["edge_mask"]
        )
    mask = padded["node_mask"]
    assert torch.allclose(d[~mask], torch.ones_like(d[~mask]))
    assert torch.all(d[mask] >= 0.5)


def test_preflight_checks_all_pass(config):
    result = run_checks(config, "cpu")
    assert result["all_passed"], {
        name: entry for name, entry in result["checks"].items() if not entry.get("passed")
    }
    assert result["checks"]["shared_initialisation"]["param_count_full"] == result[
        "checks"
    ]["shared_initialisation"]["param_count_local2"]
