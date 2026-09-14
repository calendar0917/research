"""Tests for the MolHIV H96 recurrent pair--centre transfer module.

Static / unit tests only: the T=1 reduction is bit-identical to the MolHIV
one-shot patch--path model, the T=2 recurrence applies the shared pair/centre
modules exactly twice, gradients are finite/non-zero, and the frozen selection
rules are pre-registered.  No training is performed and official test is never
loaded.
"""

from __future__ import annotations

import inspect

import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc


def _synthetic_batch(n_graphs: int = 3):
    datasets: list[Data] = []
    for g in range(n_graphs):
        n = 5 + g
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        buckets = torch.tensor(
            [min(i % (mpp.DISTANCE_BUCKETS - 1), mpp.DISTANCE_BUCKETS - 1) for i in range(len(pairs))],
            dtype=torch.long,
        )
        datasets.append(
            Data(
                patch_cont=torch.zeros(n, int(mpp.SHELL_WIDTH)),
                typed_token=torch.zeros(n, dtype=torch.long),
                parent_token=torch.zeros(n, dtype=torch.long),
                pair_index=pair_index,
                pair_relation=torch.zeros(len(pairs), int(mpp.RELATION_WIDTH)),
                pair_bucket=buckets,
                global_context=torch.zeros(1, int(mpp.GLOBAL_WIDTH)),
                y=torch.tensor([float(g % 2)]),
                num_nodes=n,
            )
        )
    return next(iter(mpp._make_loader(datasets, 4, False, 0)))


def _build(rounds: int):
    return rpc.MolhivRecurrentPairCentreModel(
        64,
        8,
        patch_hidden=rpc.H_DIM,
        pair_hidden=rpc.Q_DIM,
        token_width=rpc.TOKEN_WIDTH,
        dropout=rpc.DROPOUT,
        center_context=True,
        center_context_hidden=rpc.CENTER_CONTEXT_HIDDEN,
        recurrence_rounds=rounds,
    )


def test_01_t1_matches_base_patch_path_model():
    torch.manual_seed(0)
    recurrent = _build(1)
    torch.manual_seed(0)
    base = mpp.PatchPathModel(
        64,
        8,
        patch_hidden=rpc.H_DIM,
        pair_hidden=rpc.Q_DIM,
        token_width=rpc.TOKEN_WIDTH,
        dropout=rpc.DROPOUT,
        center_context=True,
        center_context_hidden=rpc.CENTER_CONTEXT_HIDDEN,
    )
    base.load_state_dict(recurrent.state_dict())
    batch = _synthetic_batch()
    recurrent.eval()
    base.eval()
    with torch.no_grad():
        assert float((recurrent(batch) - base(batch)).abs().max()) == 0.0


def test_02_t2_applies_shared_modules_twice():
    batch = _synthetic_batch()
    model = _build(2)
    counts = model.module_call_counts(batch)
    assert counts == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}
    assert int(model.recurrence_rounds) == 2


def test_03_t2_forward_backward_finite_nonzero():
    batch = _synthetic_batch()
    model = _build(2)
    model.train()
    logits = model(batch)
    assert bool(torch.isfinite(logits).all())
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.y.view(-1))
    loss.backward()
    assert any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in model.parameters()
    )
    assert all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )


def test_04_weight_tied_parameter_count_unchanged_vs_one_shot():
    counts = {}
    for rounds in (1, 2):
        model = _build(rounds)
        counts[rounds] = int(sum(p.numel() for p in model.parameters()))
    assert counts[1] == counts[2]


def test_05_frozen_selection_rules():
    assert rpc.SOUP_K == 5
    assert rpc.SEEDS == (0, 1)
    assert rpc.RECURRENCE_ROUNDS == 2
    # raw = best validation checkpoint; soup = top-5 validation checkpoints.
    rows = [
        {"epoch": 1, "valid_auc": 0.5, "path": "a"},
        {"epoch": 2, "valid_auc": 0.7, "path": "b"},
        {"epoch": 3, "valid_auc": 0.7, "path": "c"},
        {"epoch": 4, "valid_auc": 0.6, "path": "d"},
        {"epoch": 5, "valid_auc": 0.4, "path": "e"},
        {"epoch": 6, "valid_auc": 0.3, "path": "f"},
    ]
    top = rpc._topk(rows, 5)
    # highest AUC first; ties resolved by earliest epoch
    assert [row["epoch"] for row in top[:3]] == [2, 3, 4]
    assert rpc._topk(rows, 1)[0]["epoch"] == 2


def test_06_requires_center_context():
    try:
        rpc.MolhivRecurrentPairCentreModel(
            64,
            8,
            patch_hidden=rpc.H_DIM,
            pair_hidden=rpc.Q_DIM,
            token_width=rpc.TOKEN_WIDTH,
            dropout=0.0,
            center_context=False,
            recurrence_rounds=2,
        )
    except ValueError as error:
        assert "center_context" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when center_context is disabled")


def test_07_freeze_refuses_before_runs_exist():
    # The frozen rule may only be written from completed seed runs; the test
    # asserts the guard exists in code rather than touching experiment state.
    source = inspect.getsource(rpc.freeze)
    assert "missing run for seed" in source
    assert "official_test_unlock" in source
