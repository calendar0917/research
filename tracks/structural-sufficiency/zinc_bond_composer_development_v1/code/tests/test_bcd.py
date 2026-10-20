"""Lightweight CPU checks for the BCD development round."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import bcd_common as C  # noqa: E402
import bcd_models as M  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402


def test_branch_parameter_inventory():
    inv = M.branch_parameter_inventory()
    assert inv["input_projection"] == 4656
    assert inv["block"] == 18960
    assert inv["output_projection"] == 4704
    assert inv["total"] == 28320
    assert inv["mix_b1"] == 9408
    assert inv["mix_b2"] == 9408


def test_branch_parameter_counts_without_data():
    b1 = M.BondBranch("B1")
    b2 = M.BondBranch("B2")
    assert sum(p.numel() for p in b1.parameters()) == 28320
    assert sum(p.numel() for p in b2.parameters()) == 28320


@pytest.mark.skipif(
    not all(
        (C.CACHE / name).exists()
        for name in ("objects_fit.pt", "objects_monitor.pt", "pretrain_s0.pt")
    ),
    reason="frozen local BCD cache is absent",
)
def test_zero_branch_and_batch_invariance():
    norm = {
        "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    }
    ssl, _ = C.load_ssl_state()
    models = {}
    for kind in M.KINDS:
        _, model, _ = M.build_model(kind, seed=0, norm=norm, ssl_state=ssl, device="cpu")
        models[kind] = model.eval()
    # Import lazily to reuse the same synthetic builder as the gates.
    import bcd_runner as R

    objs = R._synthetic_objects()
    b = R._batch(objs)
    with torch.no_grad():
        pred = {k: models[k](b).detach() for k in M.KINDS}
        reader = {k: models[k].forward_features(b)["reader_in"].detach() for k in M.KINDS}
    assert torch.allclose(pred["B1"], pred["B0"], atol=1e-6, rtol=0)
    assert torch.allclose(pred["B2"], pred["B0"], atol=1e-6, rtol=0)
    assert torch.allclose(reader["B1"], reader["B0"], atol=1e-6, rtol=0)
    assert torch.allclose(reader["B2"], reader["B0"], atol=1e-6, rtol=0)
    assert reader["B0"].shape[1] == 321
    # Per-molecule evaluation equals batch evaluation for the data-independent
    # synthetic molecules.
    for kind, model in models.items():
        singles = torch.cat([model(R._batch([o])).detach() for o in objs])
        assert torch.allclose(singles, pred[kind], atol=1e-4, rtol=0)


@pytest.mark.skipif(
    not all(
        (C.CACHE / name).exists()
        for name in ("objects_fit.pt", "objects_monitor.pt", "pretrain_s0.pt")
    ),
    reason="frozen local BCD cache is absent",
)
def test_branch_common_init_and_two_step_gradients():
    import bcd_runner as R

    norm = {
        "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    }
    ssl, _ = C.load_ssl_state()
    _, b1, i1 = M.build_model("B1", seed=0, norm=norm, ssl_state=ssl, device="cpu")
    _, b2, i2 = M.build_model("B2", seed=0, norm=norm, ssl_state=ssl, device="cpu")
    assert i1["trunk_param_sha256"] == i2["trunk_param_sha256"]
    assert i1["branch_common_param_sha256"] == i2["branch_common_param_sha256"]
    objs = R._synthetic_objects()
    batch = R._batch(objs)
    y = torch.zeros(batch["n_graphs"])
    for model in (b1, b2):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        grads = []
        for _ in range(2):
            model.train()
            opt.zero_grad()
            loss = torch.nn.functional.l1_loss(model(batch).view(-1), y)
            loss.backward()
            grads.append(R._grad_norms(model))
            opt.step()
        assert grads[0]["branch.output_proj"] > 0
        assert grads[1]["branch.input_proj"] > 0
        assert all(v > 0 for k, v in grads[1].items() if k.startswith("branch."))
