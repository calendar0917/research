"""Targeted tests for the ZINC B-Null S/A/B path ablation.

Fast, self-contained: no ZINC data, no training, official test never loaded.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_bnull_sab_path_ablation as sab
from tracks.ksvd.experiments.luyin16 import zinc_local_token_null as ltn
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _valid_descriptor(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 6, size=(3, 28)).astype(np.float64)
    counts[0, 3] = 1.0
    atom_shell = (counts / counts.sum()).astype(np.float32)
    bond_counts = rng.integers(0, 4, size=(6, 4)).astype(np.float64)
    bond_counts[0, 1] = 1.0
    bond_shell = (bond_counts / bond_counts.sum()).astype(np.float32)
    root = np.zeros(28, dtype=np.float32)
    root[int(rng.integers(0, 28))] = 1.0
    incident = np.zeros(4, dtype=np.float32)
    incident[int(rng.integers(0, 4))] = 1.0
    scalars = rng.normal(size=6).astype(np.float32)
    return np.concatenate(
        [atom_shell.reshape(-1), bond_shell.reshape(-1), root, incident, scalars]
    )


def _pair_relation(seed: int = 0, adjacent: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    relation = np.zeros(sab.RELATION_WIDTH, dtype=np.float32)
    relation[0:5] = 0.0
    relation[int(rng.integers(0, 5))] = 1.0
    relation[5] = 1.0
    relation[6:14] = rng.random(8).astype(np.float32)
    path = rng.random(4).astype(np.float32)
    relation[14:18] = path / path.sum()
    relation[18] = 0.5
    if adjacent:
        relation[19 + int(rng.integers(0, 4))] = 1.0
    return relation


def _batch() -> Data:
    from tracks.ksvd.experiments.luyin16 import (
        zinc_compact_v4_identity_capacity_control as ic,
    )

    batch = ic._synthetic_batch()
    rng = torch.Generator().manual_seed(7)
    batch.parent_token = torch.randint(
        0, 32, (batch.patch_cont.shape[0],), generator=rng
    )
    return batch


# ---------------------------------------------------------------------------
# feature map
# ---------------------------------------------------------------------------


def test_feature_map_matches_code_constants():
    feature = sab.feature_map()
    assert feature["patch_cont"]["width"] == int(zpp.SHELL_WIDTH) == 146
    assert feature["pair_relation"]["width"] == int(zpp.RELATION_WIDTH) == 23
    assert sab.ATOM_CATEGORIES == 28 and sab.BOND_CATEGORIES == 4
    # blocks tile the whole width exactly once
    covered = []
    for block in feature["patch_cont"]["blocks"]:
        covered.extend(range(block["columns"][0][0], block["columns"][-1][1]))
    assert covered == list(range(146))
    covered_pair = []
    for block in feature["pair_relation"]["blocks"]:
        covered_pair.extend(range(block["columns"][0][0], block["columns"][-1][1]))
    assert covered_pair == list(range(23))


# ---------------------------------------------------------------------------
# patch-B marginalization
# ---------------------------------------------------------------------------


def test_patch_marginalization_preserves_marginals_and_scalars():
    descriptor = _valid_descriptor(3)
    out = sab.marginalize_patch_descriptor(descriptor)
    assert out.shape == descriptor.shape
    atom = descriptor[0:84].reshape(3, 28)
    out_atom = out[0:84].reshape(3, 28)
    assert np.allclose(atom.sum(axis=0), out_atom.sum(axis=0), atol=1e-5)
    assert np.allclose(atom.sum(axis=1), out_atom.sum(axis=1), atol=1e-5)
    bond = descriptor[84:108].reshape(6, 4)
    out_bond = out[84:108].reshape(6, 4)
    assert np.allclose(bond.sum(axis=0), out_bond.sum(axis=0), atol=1e-5)
    assert np.allclose(bond.sum(axis=1), out_bond.sum(axis=1), atol=1e-5)
    # root / incident chemistry replaced by the patch marginals
    assert np.allclose(out[108:136], out_atom.sum(axis=0), atol=1e-5)
    assert np.allclose(out[136:140], out_bond.sum(axis=0), atol=1e-5)
    # pure-S scalars bit-identical
    assert np.array_equal(descriptor[140:146], out[140:146])


def test_patch_marginalization_destroys_shell_assignment():
    descriptor = _valid_descriptor(11)
    out = sab.marginalize_patch_descriptor(descriptor)
    atom = descriptor[0:84].reshape(3, 28)
    out_atom = out[0:84].reshape(3, 28)
    # conditional distribution over atom types is now shell-independent
    conditional = out_atom / np.maximum(out_atom.sum(axis=1, keepdims=True), 1e-12)
    assert np.allclose(conditional[0], conditional[1], atol=1e-5)
    assert not np.allclose(atom, out_atom, atol=1e-6)


def test_patch_marginalization_idempotent_on_marginals():
    descriptor = _valid_descriptor(5)
    out = sab.marginalize_patch_descriptor(descriptor)
    assert np.allclose(
        out, sab.marginalize_patch_descriptor(out), atol=1e-6
    )


# ---------------------------------------------------------------------------
# pair-B marginalization
# ---------------------------------------------------------------------------


def test_pair_marginalization_pure_s_bit_identical():
    relation = _pair_relation(1)
    marginal = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out = sab.marginalize_pair_relation(relation, marginal)
    for start, stop in sab.PAIR_S_SLICES:
        assert np.array_equal(relation[start:stop], out[start:stop])


def test_pair_marginalization_preserves_adjacency_mass_and_destroys_bond():
    relation = _pair_relation(2, adjacent=True)
    marginal = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out = sab.marginalize_pair_relation(relation, marginal)
    assert np.isclose(relation[19:23].sum(), out[19:23].sum())
    assert np.allclose(out[14:18], marginal)
    assert np.allclose(out[19:23], marginal)
    non_adjacent = _pair_relation(2, adjacent=False)
    out_non = sab.marginalize_pair_relation(non_adjacent, marginal)
    assert np.allclose(out_non[19:23], 0.0)


def test_graph_bond_marginal_is_normalized_counts():
    relation = np.zeros((3, sab.RELATION_WIDTH), dtype=np.float32)
    relation[0, 19] = 1.0  # bond type 0
    relation[1, 19] = 1.0
    relation[2, 21] = 1.0  # bond type 2
    record = type("R", (), {"pair_relation": relation})()
    marginal = sab.graph_bond_marginal(record)
    assert np.isclose(marginal.sum(), 1.0)
    assert np.isclose(marginal[0], 2.0 / 3.0)
    assert np.isclose(marginal[2], 1.0 / 3.0)
    assert np.allclose(marginal[[1, 3]], 0.0)


# ---------------------------------------------------------------------------
# retrained-ticket builders
# ---------------------------------------------------------------------------


def test_noparent_keeps_params_and_zeroes_parent_output():
    model = sab.build_noparent(0)
    assert sab._n_params(model) == ltn.NULL_TOTAL_PARAMS == 49343
    assert all(not p.requires_grad for p in model.parent_embedding.parameters())
    batch = _batch()
    with torch.no_grad():
        parent = model.parent_embedding(batch.parent_token)
    assert torch.equal(parent, torch.zeros_like(parent))
    # shared backbone tensors are bit-identical to the canonical B-Null init
    null = ltn.build_null(0)
    for key, value in null.state_dict().items():
        if key.startswith("parent_embedding"):
            continue
        assert torch.equal(value, model.state_dict()[key]), key


def test_t1_builder_has_same_params_and_one_round():
    model = sab.build_t1(0)
    assert sab._n_params(model) == ltn.NULL_TOTAL_PARAMS
    assert model.recurrence_rounds == 1
    batch = _batch()
    counts = model.module_call_counts(batch)
    assert counts["pair_encoder"] == 1
    assert counts["center_update"] == 1
    null = ltn.build_null(0)
    t2 = null.module_call_counts(batch)
    assert t2["pair_encoder"] == 2
    assert t2["center_update"] == 2


def test_t1_frozen_forward_is_single_round_readout():
    model = ltn.build_null(0).eval()
    batch = _batch()
    with torch.no_grad():
        unified = model.encode_original(batch)
        expected = model.head(unified).view(-1)
    assert unified.ndim == 2
    assert torch.equal(expected, model.head(unified).view(-1))


# ---------------------------------------------------------------------------
# frozen interventions
# ---------------------------------------------------------------------------


def test_zero_parent_embedding_context_manager_zeroes_only_parent():
    model = ltn.build_null(0).eval()
    batch = _batch()
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, inputs, _output):
        captured["input"] = inputs[0].detach().clone()

    handle = model.patch_encoder.register_forward_hook(capture)
    with torch.no_grad():
        model(batch)
    handle.remove()
    base = captured["input"].clone()

    handle = model.patch_encoder.register_forward_hook(capture)
    with torch.no_grad(), sab.zero_parent_embedding(model):
        model(batch)
    handle.remove()
    after = captured["input"].clone()

    parent_start = int(model.shell_width) + int(model.context_width) + int(
        model.token_width
    )
    parent_width = int(model.parent_width)
    parent_slice = slice(parent_start, parent_start + parent_width)
    assert torch.equal(after[:, parent_slice], torch.zeros_like(after[:, parent_slice]))
    others = [
        index for index in range(base.shape[1]) if not (parent_start <= index < parent_start + parent_width)
    ]
    assert torch.equal(base[:, others], after[:, others])
