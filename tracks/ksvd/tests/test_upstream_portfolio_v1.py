"""Focused wiring tests for the upstream portfolio arms (no full suite)."""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_upstream_portfolio_v1 as up
from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as z
from tracks.ksvd.experiments.luyin16.upstream_portfolio_v1_reference import (
    upstream_reference as ref,
)

torch.set_num_threads(4)


def test_expected_parameter_counts_match_the_frozen_table():
    full = {arm: up.expected_parameters(arm, sc.FULL) for arm in up.ARMS}
    small = {arm: up.expected_parameters(arm, sc.SMALL) for arm in up.ARMS}
    assert full == {"control": 408651, "code": 431115, "drop": 408651, "r3": 420963}
    assert small == {"control": 106925, "code": 109421, "drop": 106925, "r3": 111029}


def test_small_parameter_audit_is_exact():
    audit = z.small_parameter_audit()
    assert audit["all_exact"] is True


def test_shell3_features_match_the_vendored_reference():
    # a small asymmetric graph with a shell-3 atom and both (2,3)/(3,3) bonds.
    atoms = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1, 1), (1, 2, 0), (2, 3, 1), (3, 4, 2), (4, 5, 3), (1, 5, 0)]
    got = up.shell3_features(len(atoms), atoms, edges)
    _old, extra = ref.shell_semantics(atoms, edges)
    assert np.array_equal(extra, got)
    assert up.R3_DIM == 36


def test_undirected_edges_dedup_two_directed_copies():
    edge_index = np.array([[0, 1, 1, 0], [1, 0, 2, 2]])
    edge_attr = np.array([3, 3, 1, 1])
    edges = up.undirected_edges(edge_index, edge_attr)
    assert sorted(edges) == [(0, 1, 3), (0, 2, 1), (1, 2, 1)]


def test_r3_block_scales_zero_block_is_one():
    features = torch.zeros(4, 36)
    features[:, 0] = 2.0
    scales = up.r3_block_scales(features)
    assert abs(scales[0] - float(np.sqrt(4.0 * 4.0 / (4.0 * 28.0)))) < 1e-9
    assert scales[1] == 1.0 and scales[2] == 1.0


def test_arms_reproduce_the_parent_with_parent_weights():
    subspace = z._load_parent_subspace()
    dictionary = z._dictionary_tensor()
    soup = z.load_parent_soup_state()
    train = z.load_split("control", "train", subset=16)
    batch = next(iter(p1.make_env_loader(train, 16, False, 0)))
    # R3 wiring test without the raw-data cache: zero features prove that the
    # zero-initialised new columns leave the parent function untouched.
    for item in train:
        item.r3_features = torch.zeros(int(item.num_nodes), 36, dtype=torch.int16)
    batch = next(iter(p1.make_env_loader(train, 16, False, 0)))

    parent = up.build_arm_model("control", dictionary, 0, subspace, sc.FULL)
    parent.load_state_dict(soup)
    parent.eval()
    with torch.no_grad():
        parent_prediction = parent(batch, mask=cm.C6_MASK).view(-1).double()

    for arm in up.ARMS:
        torch.manual_seed(0)
        model = z.make_model(arm, dictionary, subspace, r3_scales=(1.0, 1.0, 1.0), drop_p=up.DROP_P)
        up.load_parent_soup(model, arm, soup)
        assert up.arm_parameter_audit(model, arm)["parameter_exact"] is True
        model.eval()
        with torch.no_grad():
            prediction = model(batch, mask=cm.C6_MASK).view(-1).double()
        assert float((prediction - parent_prediction).abs().max()) <= 1.0e-4


def test_code_moments_are_the_single_bridge_code_and_are_appended():
    subspace = z._load_parent_subspace()
    dictionary = z._dictionary_tensor()
    soup = z.load_parent_soup_state()
    train = z.load_split("control", "train", subset=16)
    batch = next(iter(p1.make_env_loader(train, 16, False, 0)))

    torch.manual_seed(0)
    model = z.make_model("code", dictionary, subspace, r3_scales=(1.0, 1.0, 1.0), drop_p=up.DROP_P)
    up.load_parent_soup(model, "code", soup)
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    handle = model.reader.register_forward_pre_hook(
        lambda _m, inputs: captured.__setitem__("unified", inputs[0].detach())
    )
    with torch.no_grad():
        model(batch, mask=cm.C6_MASK)
    handle.remove()
    moments = model._graph_code_moments(torch.device("cpu"), torch.float32)
    unified = captured["unified"]
    assert unified.shape[1] == sc.FULL.reader_input + 2 * sc.FULL.k
    assert float((unified[:, -moments.shape[1] :] - moments).abs().max()) == 0.0


def test_drop_arm_eval_identity_and_training_mask_shared_across_rows():
    subspace = z._load_parent_subspace()
    dictionary = z._dictionary_tensor()
    soup = z.load_parent_soup_state()
    train = z.load_split("control", "train", subset=16)
    batch = next(iter(p1.make_env_loader(train, 16, False, 0)))

    torch.manual_seed(0)
    model = z.make_model("drop", dictionary, subspace, r3_scales=(1.0, 1.0, 1.0), drop_p=up.DROP_P)
    up.load_parent_soup(model, "drop", soup)
    parent = up.build_arm_model("control", dictionary, 0, subspace, sc.FULL)
    parent.load_state_dict(soup)
    parent.eval()
    model.eval()
    with torch.no_grad():
        parent_prediction = parent(batch, mask=cm.C6_MASK).view(-1).double()
        eval_prediction = model(batch, mask=cm.C6_MASK).view(-1).double()
    assert float((eval_prediction - parent_prediction).abs().max()) == 0.0
    model.train()
    with torch.no_grad():
        train_prediction = model(batch, mask=cm.C6_MASK).view(-1).double()
    assert float((train_prediction - parent_prediction).abs().max()) > 0.0
