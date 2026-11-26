"""Tests for the MEANFIELD distribution pilot (CPU only, no OGB access).

Run:  uv run pytest -q tracks/structural-sufficiency/fixed_source_structural_distribution_pilot_v0/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import mf_checks as CK  # noqa: E402
import mf_common as R  # noqa: E402
import mf_model as MF  # noqa: E402
import ds_common as C  # noqa: E402
import ds_model as DM  # noqa: E402


@pytest.fixture(scope="module")
def spec():
    return CK.load_spec()


def test_parameter_and_init_parity(spec):
    report = CK.check_parameter_parity(spec)
    assert report["passed"]
    assert report["parameters_meanfield"] == R.PARAMS_TOTAL
    assert report["n_state_tensors_differing"] == 0


def test_group_isolation(spec):
    report = CK.check_group_isolation(CK._double_model(spec), CK.synthetic_graph())
    assert report["passed"]
    assert report["n_groups"] == 3


def test_conservation_and_counts(spec):
    report = CK.check_conservation_and_counts(CK._double_model(spec), CK.synthetic_graph())
    assert report["passed"]
    assert report["max_relative_conservation_error"] < 1e-12
    assert report["per_centre_in_degree_preserved"]
    assert report["per_centre_bond_type_counts_preserved"]


def test_binding_witness(spec):
    report = CK.check_binding_witness(spec)
    assert report["passed"]
    assert report["max_abs_p_diff_between_two_sources_fixed1"] > 1e-3
    assert report["max_abs_p_diff_after_meanfield"] < 1e-12


def test_invariances(spec):
    report = CK.check_invariances(spec)
    assert report["passed"]
    assert report["meanfield_content_direction_symmetric"]
    assert report["fixed1_content_max_abs_asymmetry_between_directions"] > 1e-3


def test_empty_graph_and_gradients(spec):
    report = CK.check_empty_graph_and_gradients(spec)
    assert report["passed"]
    assert report["aggregation_is_exactly_zero"]


def test_test_split_is_rejected_at_runtime():
    import mf_train as T

    with pytest.raises(RuntimeError):
        T.evaluate(model=None, task=None, device=torch.device("cpu"), split="test", reason="guard")


def test_static_test_isolation():
    report = CK.check_test_isolation()
    assert report["passed"]
    assert not report["split_calls_with_test"]


def test_group_mean_matches_naive_reference(spec):
    """The vectorised segment reduce equals an explicit per-group mean."""
    model = CK._double_model(spec)
    batch = CK.synthetic_graph(dtype=torch.float64)
    h0 = batch["h0"]
    src, dst, edge_msg = DM.directed_bonds(model, batch)
    bf_dir, gb_dir = MF.directed_bond_features(batch)
    gid, n_groups, _ = MF.group_index(bf_dir, gb_dir)
    z = model.blocks[0].norm(h0)
    m, p, p_mf = MF.meanfield_message(z, src, dst, edge_msg, gid, n_groups)
    expected_m = torch.zeros_like(z)
    for g in range(n_groups):
        members = (gid == g).nonzero(as_tuple=False).flatten()
        mean = p.index_select(0, members).mean(dim=0)
        targets = dst.index_select(0, members)
        expected_m.index_add_(0, targets, mean.expand(len(members), z.shape[1]))
    assert torch.allclose(m, expected_m, atol=1e-12, rtol=0)


def test_zero_initialised_fc2_reproduces_untrained_w2(spec):
    """At initialisation both blocks are pure residuals, so h2 == h0."""
    model, _ = MF.build_mf_model(spec, CK.fake_norm(spec), C.N_TASKS, seed=0, verify=False)
    batch = CK.synthetic_graph(dtype=torch.float32)
    h0 = batch["h0"]
    h1, h2, _info = model.structural_layers(h0, batch)
    assert torch.allclose(h1, h0, atol=0, rtol=0)
    assert torch.allclose(h2, h0, atol=0, rtol=0)
