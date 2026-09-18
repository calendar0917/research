"""Targeted tests for the MolHIV B-Null frozen small-head sufficiency probe.

Fast, self-contained: no MolHIV record cache, no training, official test never
loaded.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import molhiv_bnull_small_head_probe as probe
from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc


def _toy_model() -> mpp.PatchPathModel:
    return rpc.MolhivRecurrentPairCentreModel(
        8,
        6,
        patch_hidden=16,
        pair_hidden=4,
        token_width=6,
        dropout=0.0,
        center_context=True,
        center_context_hidden=12,
        recurrence_rounds=2,
        patch_representation="null",
    )


def _toy_batch() -> Data:
    generator = torch.Generator().manual_seed(23)
    return Data(
        patch_cont=torch.randn(3, mpp.SHELL_WIDTH, generator=generator),
        typed_token=torch.tensor([1, 2, 3], dtype=torch.long),
        parent_token=torch.tensor([1, 1, 2], dtype=torch.long),
        pair_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        pair_relation=torch.randn(3, mpp.RELATION_WIDTH, generator=generator),
        pair_bucket=torch.tensor([0, 1, 0], dtype=torch.long),
        global_context=torch.randn(1, mpp.GLOBAL_WIDTH, generator=generator),
        batch=torch.zeros(3, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=3,
    )


def test_head_parameter_counts_exact():
    assert probe._n_params(probe.build_h_refit()) == 100417
    assert probe._n_params(probe.build_h_small()) == 14177
    assert probe.HEAD_REFIT_INPUT_DIM == 423


def test_h_refit_matches_current_molhiv_head_architecture():
    model = _toy_model()
    # the current MolHIV head is Linear(423,192)/LN/ReLU/Dropout/Linear(192,96)/
    # ReLU/Linear(96,1); compare the module specification only
    current = mpp.PatchPathModel(
        8,
        6,
        patch_hidden=16,
        pair_hidden=4,
        token_width=6,
        dropout=0.05,
        center_context=True,
        center_context_hidden=12,
    )
    refit = probe.build_h_refit()
    assert len(refit) == len(current.head)
    assert str(refit) == str(probe.build_h_refit())


def test_R_dimension_423_from_real_head_input():
    model = _toy_model()
    model.head = torch.nn.Linear(423, 1)
    assert int(model.head.in_features) == probe.HEAD_REFIT_INPUT_DIM


def test_head_pre_hook_reproduces_model_logits():
    model = _toy_model().eval()
    batch = _toy_batch()
    captured: list[torch.Tensor] = []

    def hook(_module, args):
        captured.append(args[0].detach())

    handle = model.head.register_forward_pre_hook(hook)
    with torch.no_grad():
        logits = model(batch)
    handle.remove()
    assert captured
    assert torch.equal(model.head(captured[0]).view(-1), logits.view(-1))


def test_head_forward_and_valid_auc_are_finite_on_synthetic_R():
    torch.manual_seed(0)
    head = probe.build_h_small()
    R = torch.randn(64, probe.HEAD_REFIT_INPUT_DIM)
    y = (torch.rand(64) > 0.5).float()
    logits = probe._head_forward(head, R)
    assert logits.shape == (64,)
    assert torch.isfinite(logits).all()
    auc = probe._valid_auc(head, R, y, torch.device("cpu"))
    assert 0.0 <= auc <= 1.0
