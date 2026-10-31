"""Focused tests for the corrected distance aggregation and the checkpoint rule.

These tests check the *content* of the aggregation, not just its shape:

* a hand-computed 6-atom chain (0-1-2-3-4-5) with ``h = [0, 1, 4, 9, 16, 25]``
  must give ``c34_0 = 12.5`` and ``c5_0 = 25``;
* empty buckets must be exactly zero;
* a multi-molecule batch must not mix molecules;
* atom renumbering and batch reordering must not change predictions;
* gradients must reach the real target atoms (and only the containing roots);
* ``DIST-TRUE-FIXED`` and ``DIST-BLIND`` must have bitwise identical
  initialisation for the same seed;
* the historical aggregation is reconstructed and shown to be a rescaled copy
  of the root atom's own representation;
* the best-epoch tracker keeps the earlier epoch on a tie and returns exactly
  the state it recorded.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import be_common as C
import be_data as D
import be_models as M
import be_train as T

CHAIN_H = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
CHAIN_D34 = {0: [3, 4], 1: [4, 5], 2: [5], 3: [0], 4: [0, 1], 5: [1, 2]}
CHAIN_D5 = {0: [5], 1: [], 2: [], 3: [], 4: [], 5: [0]}


@lru_cache(maxsize=1)
def views():
    return D.load_views(include_test=False)


def chain_batch(atom_batch: torch.Tensor | None = None) -> dict:
    d34_idx = torch.as_tensor([j for i in range(6) for j in CHAIN_D34[i]], dtype=torch.long)
    d5_idx = torch.as_tensor([j for i in range(6) for j in CHAIN_D5[i]], dtype=torch.long)
    return {
        "d34_idx": d34_idx,
        "d5_idx": d5_idx,
        "d34_counts": torch.as_tensor([len(CHAIN_D34[i]) for i in range(6)], dtype=torch.long),
        "d5_counts": torch.as_tensor([len(CHAIN_D5[i]) for i in range(6)], dtype=torch.long),
        "atom_batch": torch.zeros(6, dtype=torch.long) if atom_batch is None else atom_batch,
    }


# ---------------------------------------------------------------------------
# aggregation content
# ---------------------------------------------------------------------------


def test_chain_bucket_means_hand_computed():
    batch = chain_batch()
    h = torch.as_tensor(CHAIN_H, dtype=torch.float64).unsqueeze(1)
    c34, c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    assert [float(x) for x in c34.squeeze(1)] == [12.5, 20.5, 25.0, 0.0, 0.5, 2.5]
    assert [float(x) for x in c5.squeeze(1)] == [25.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # the definition asked for in the protocol, spelled out
    assert float(c34[0, 0]) == (9.0 + 16.0) / 2.0
    assert float(c5[0, 0]) == 25.0


def test_chain_bucket_membership_matches_definition():
    reference = D.independent_bucket_from_adjacency([[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]])
    assert reference["d34"] == [CHAIN_D34[i] for i in range(6)]
    assert reference["d5"] == [CHAIN_D5[i] for i in range(6)]


def test_empty_bucket_is_exactly_zero_not_self_representation():
    h = torch.randn(6, C.WIDTH)
    batch = chain_batch()
    c34, c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    assert torch.equal(c5[1:5], torch.zeros(4, C.WIDTH))
    model, _meta = M.build_family("dist_true_fixed", views()["spec"], views()["train_norm"], 0, "cpu")
    with torch.no_grad():
        model.dist.W_out.bias.fill_(0.25)
    empty = dict(chain_batch())
    empty["d34_idx"] = torch.zeros(0, dtype=torch.long)
    empty["d5_idx"] = torch.zeros(0, dtype=torch.long)
    empty["d34_counts"] = torch.zeros(6, dtype=torch.long)
    empty["d5_counts"] = torch.zeros(6, dtype=torch.long)
    change, ctx = model.dist(h, empty, 1, "true")
    assert torch.equal(change, torch.zeros_like(change))
    assert torch.equal(ctx["c34"], torch.zeros_like(ctx["c34"]))


def test_aggregation_matches_independent_reference():
    batch = D.collate(views()["valid_cache"], list(range(12)))
    torch.manual_seed(3)
    h = torch.randn(int(batch["atom_batch"].numel()), C.WIDTH, dtype=torch.float64)
    c34, c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    ref34, ref5 = D.reference_bucket_means(batch, h.numpy())
    assert np.abs(c34.numpy() - ref34).max() < 1e-6
    assert np.abs(c5.numpy() - ref5).max() < 1e-6


def test_historical_aggregation_is_scaled_self_representation():
    batch = D.collate(views()["valid_cache"], list(range(8)))
    torch.manual_seed(5)
    h = torch.randn(int(batch["atom_batch"].numel()), C.WIDTH, dtype=torch.float64)
    n_atoms = int(h.shape[0])
    buggy = h.new_zeros((n_atoms, C.WIDTH)).index_add_(
        0, batch["d34_idx"], h.index_select(0, batch["d34_idx"])
    ) / batch["d34_counts"].clamp(min=1).unsqueeze(1)
    in_degree = h.new_zeros(n_atoms).index_add_(
        0, batch["d34_idx"], torch.ones_like(batch["d34_idx"], dtype=h.dtype)
    )
    ratio = in_degree / batch["d34_counts"].clamp(min=1).to(h.dtype)
    # equal up to the different floating-point order of the two expressions
    assert float((buggy - h * ratio.unsqueeze(1)).abs().max()) < 1e-12
    fixed, _c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    assert float((fixed - buggy).abs().max()) > 1e-2


def test_bucket_index_arrays_match_bfs_definition():
    batch = D.collate(views()["valid_cache"], list(range(4)))
    ref = D.reference_buckets(batch)
    assert torch.equal(batch["d34_idx"], ref["d34_targets"])
    assert torch.equal(batch["d5_idx"], ref["d5_targets"])
    assert torch.equal(batch["d34_counts"], ref["d34_counts"])
    assert torch.equal(batch["d5_counts"], ref["d5_counts"])


# ---------------------------------------------------------------------------
# batching / invariance
# ---------------------------------------------------------------------------


def test_multi_molecule_batch_has_no_cross_talk():
    spec, norm = views()["spec"], views()["train_norm"]
    model, _meta = M.build_family("dist_true_fixed", spec, norm, 0, "cpu")
    model.eval()
    rows = [7, 13, 21]
    batch = D.collate(views()["valid_cache"], rows)
    atom_batch = batch["atom_batch"]
    with torch.no_grad():
        y_batch, aux_batch = model(batch, return_aux=True)
        for g, row in enumerate(rows):
            single = D.collate(views()["valid_cache"], [row])
            y_single, aux_single = model(single, return_aux=True)
            mask = atom_batch == g
            assert torch.allclose(aux_batch["c34"][mask], aux_single["c34"], atol=1e-6)
            assert torch.allclose(aux_batch["c5"][mask], aux_single["c5"], atol=1e-6)
            assert abs(float(y_batch[g] - y_single[0])) < 1e-6
    feat = batch["atom_feat"].clone()
    feat[atom_batch == 0] = 0.0
    perturbed = dict(batch)
    perturbed["atom_feat"] = feat
    with torch.no_grad():
        y_pert = model(perturbed)
    assert abs(float(y_batch[1] - y_pert[1])) < 1e-6


def test_atom_renumbering_preserves_prediction():
    spec, norm = views()["spec"], views()["train_norm"]
    model, _meta = M.build_family("dist_true_fixed", spec, norm, 0, "cpu")
    model.eval()
    batch = D.collate(views()["valid_cache"], [2, 9, 17, 30])
    generator = torch.Generator().manual_seed(77)
    permuted = D.permute_batch_atoms(batch, generator)
    with torch.no_grad():
        y = model(batch).numpy().reshape(-1)
        y_perm = model(permuted).numpy().reshape(-1)
    assert np.abs(y - y_perm).max() < 1e-5


def test_batch_order_preserves_per_molecule_prediction():
    spec, norm = views()["spec"], views()["train_norm"]
    model, _meta = M.build_family("dist_true_fixed", spec, norm, 0, "cpu")
    model.eval()
    rows = [4, 8, 15, 16]
    with torch.no_grad():
        y = model(D.collate(views()["valid_cache"], rows)).numpy().reshape(-1)
        y_rev = model(D.collate(views()["valid_cache"], list(reversed(rows)))).numpy().reshape(-1)
    assert np.abs(y - y_rev[::-1]).max() < 1e-6


# ---------------------------------------------------------------------------
# gradients / locality
# ---------------------------------------------------------------------------


def test_gradient_flows_to_true_target_atoms():
    batch = D.collate(views()["valid_cache"], list(range(3)))
    n_atoms = int(batch["atom_batch"].numel())
    torch.manual_seed(9)
    h = torch.randn(n_atoms, C.WIDTH, dtype=torch.float64, requires_grad=True)
    c34, c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    (c34.sum() + c5.sum()).backward()
    grad = h.grad
    ref = D.reference_buckets(batch)
    expected = torch.zeros(n_atoms, dtype=torch.bool)
    for bucket in ref["d34"] + ref["d5"]:
        if bucket:
            expected[torch.as_tensor(bucket, dtype=torch.long)] = True
    got = grad.abs().sum(dim=1) > 0
    assert torch.equal(expected, got)
    assert float(grad.abs().max()) > 0.0


def test_changing_target_content_changes_containing_roots_only():
    batch = D.collate(views()["valid_cache"], list(range(2)))
    n_atoms = int(batch["atom_batch"].numel())
    torch.manual_seed(13)
    h = torch.randn(n_atoms, C.WIDTH, dtype=torch.float64)
    c34, c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    ref = D.reference_buckets(batch)
    for j in range(min(n_atoms, 16)):
        h2 = h.clone()
        h2[j] += 1.0
        c34b, c5b = D.bucket_means(h2, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
        assert ((c34b - c34).abs().sum(1) > 0).tolist() == [j in bucket for bucket in ref["d34"]]
        assert ((c5b - c5).abs().sum(1) > 0).tolist() == [j in bucket for bucket in ref["d5"]]


# ---------------------------------------------------------------------------
# model / init parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_dist_true_fixed_and_dist_blind_have_identical_init(seed):
    spec, norm = views()["spec"], views()["train_norm"]
    true_model, true_meta = M.build_family("dist_true_fixed", spec, norm, seed, "cpu")
    blind_model, blind_meta = M.build_family("dist_blind", spec, norm, seed, "cpu")
    true_state = true_model.state_dict()
    blind_state = blind_model.state_dict()
    assert set(true_state) == set(blind_state)
    for key in true_state:
        assert torch.equal(true_state[key], blind_state[key]), key
    assert true_meta["parameters"] == blind_meta["parameters"]
    assert true_meta["init_state_hash"] == blind_meta["init_state_hash"]


def test_all_families_share_the_base_initialisation():
    import x1_models as X1M

    spec, norm = views()["spec"], views()["train_norm"]
    base, base_meta = M.build_family("base", spec, norm, 0, "cpu")
    for family in C.FAMILIES:
        if family == "base":
            continue
        model, meta = M.build_family(family, spec, norm, 0, "cpu")
        audit = X1M.shared_init_audit(base, model)
        assert audit["all_shared_bitwise_identical"], (family, audit["mismatched"])
        assert meta["base_init_state_hash"] == base_meta["init_state_hash"]


def test_zeroed_correction_recovers_base():
    spec, norm = views()["spec"], views()["train_norm"]
    batch = M.to_device(D.collate(views()["valid_cache"], list(range(10))), "cpu")
    base, _ = M.build_family("base", spec, norm, 0, "cpu")
    base.eval()
    with torch.no_grad():
        y_base = base(batch).numpy().reshape(-1)
    for family in ("roleblind", "g2l_loo", "g2l_all", "dist_true_fixed", "dist_blind"):
        model, _meta = M.build_family(family, spec, norm, 0, "cpu")
        out = M.correction_output(model, family)
        with torch.no_grad():
            out.weight.zero_()
            out.bias.zero_()
        model.eval()
        with torch.no_grad():
            y = model(batch).numpy().reshape(-1)
        assert np.abs(y - y_base).max() < 1e-6, family


# ---------------------------------------------------------------------------
# best-epoch rule
# ---------------------------------------------------------------------------


def test_best_epoch_tracker_tie_keeps_earlier_epoch():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = T.BestEpochTracker()
    for epoch, rmse, mae in [(1, 0.60, 0.50), (2, 0.55, 0.45), (3, 0.55, 0.40), (4, 0.58, 0.42)]:
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        tracker.update(epoch, rmse, mae, model)
    assert tracker.best_epoch == 2
    assert tracker.best_rmse == 0.55
    assert tracker.best_mae == 0.45
    assert float(tracker.state["weight"]) == 2.0
    assert tracker.updates == [1, 2]


def test_best_epoch_tracker_state_is_immutable():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = T.BestEpochTracker()
    with torch.no_grad():
        model.weight.fill_(1.0)
    tracker.update(1, 0.5, 0.4, model)
    with torch.no_grad():
        model.weight.fill_(9.0)
    assert float(tracker.state["weight"]) == 1.0


def test_recipe_constants_are_the_frozen_r3_recipe():
    assert C.TRAIN_RECIPE["batch_size"] == 128
    assert C.TRAIN_RECIPE["epochs"] == 360
    assert C.TRAIN_RECIPE["peak_lr"] == 3.0e-4
    assert C.TRAIN_RECIPE["min_lr"] == 1.0e-5
    assert C.TRAIN_RECIPE["weight_decay"] == 1.0e-3
    assert C.TRAIN_RECIPE["gradient_clip"] == 5.0
    assert C.TRAIN_RECIPE["dropout"] == 0.10
    assert C.TRAIN_RECIPE["checkpoint"] == "single_best_validation420_rmse_epoch"
    assert C.TRAIN_RECIPE["early_stopping"] is False
    assert len(C.ARMS) == 18
    assert len(set(C.ARMS)) == 18
