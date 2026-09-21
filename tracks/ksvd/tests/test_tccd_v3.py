"""Targeted correctness tests for TCCD-v3 Strong-Local Bridge."""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v3 as V
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like


def _records(n_graphs: int = 4):
    layout = T.PatchLayout(capacity=8, n_atom=5, n_bond=3)
    atom_index = {c: i for i, c in enumerate(range(5))}
    bond_index = {c: i for i, c in enumerate(range(3))}
    out = []
    for seed in range(n_graphs):
        rec = T.build_mol_record(synthetic_zinc_like(seed, 10 + seed), layout, atom_index, bond_index)
        rng = np.random.default_rng(100 + seed)
        rec["X"] = rng.standard_normal((int(rec["n"]), V.STRONG_LOCAL_WIDTH)).astype(np.float32)
        rec["y"] = float(seed) / 3.0
        out.append(rec)
    return out


def test_gate0_passes_locality_permutation_batch_and_freeze():
    checks = V.gate0_checks()
    assert checks["official_test_loaded"] is False
    assert checks["pure_local"] is True
    assert checks["permutation_invariant"] is True
    assert checks["batch_invariant"] is True
    assert checks["deterministic"] is True
    assert checks["frozen_encoder"] is True
    assert checks["all_pass"] is True


def test_dense_bridge_adapter_and_reader_receive_task_gradient():
    torch.manual_seed(0)
    records = _records()
    model = V.DenseBridgeModel(seed=0)
    batch = V.make_batch(records, [0, 1, 2], "cpu")
    pred, _, h = model.forward_padded(batch)
    loss = (pred - batch["y"]).abs().mean()
    loss.backward()
    assert model.adapter.weight.grad is not None
    assert float(model.adapter.weight.grad.abs().sum()) > 0.0
    assert model.head.weight.grad is not None
    assert float(model.head.weight.grad.abs().sum()) > 0.0
    assert h.shape[1] == T.h_dim(V.K_PROTO, V.N_REL)
    assert model.uses_latent_bypass is False
    assert model.graph_feature_kind == "dense_latent_composition_only"


def test_prototype_bridge_gradients_and_composition_sensitivity():
    torch.manual_seed(1)
    records = _records()
    model = V.PrototypeBridgeModel(seed=0)
    batch = V.make_batch(records, [0, 1, 2], "cpu")
    pred, C, h = model.forward_padded(batch)
    loss = (pred - batch["y"]).abs().mean()
    loss.backward()
    assert model.adapter.weight.grad is not None
    assert float(model.adapter.weight.grad.abs().sum()) > 0.0
    assert model.P.grad is not None
    assert float(model.P.grad.abs().sum()) > 0.0
    assert model.temp_logit.grad is not None
    assert float(model.temp_logit.grad.abs().sum()) > 0.0
    assert model.head.weight.grad is not None
    assert float(model.head.weight.grad.abs().sum()) > 0.0
    assert model.uses_latent_bypass is False
    assert model.graph_feature_kind == "assignment_composition_only"
    with torch.no_grad():
        h_shuffle = model.graph_features(
            C,
            batch,
            shuffle=True,
            indices=[0, 1, 2],
            records=records,
            seed=0,
        )
    assert float((h - h_shuffle).abs().max()) > 1.0e-6


def test_prototype_reader_does_not_depend_on_continuous_latent_bypass():
    records = _records(2)
    model = V.PrototypeBridgeModel(seed=0)
    batch = V.make_batch(records, [0, 1], "cpu")
    with torch.no_grad():
        pred0, C0, _ = model.forward_padded(batch)
        Z0 = model.encode(batch["X_pad"].reshape(-1, V.STRONG_LOCAL_WIDTH))
        C3 = C0.reshape(batch["X_pad"].shape[0], batch["X_pad"].shape[1], V.K_PROTO)
        h0 = model.graph_features(C0, batch)
        h1 = model.graph_features(C3.reshape(-1, V.K_PROTO), batch)
    assert float((pred0 - model.head(h0).reshape(-1)).abs().max()) == 0.0
    assert float((h0 - h1).abs().max()) == 0.0
    assert Z0.shape[1] == V.D_LOCAL
