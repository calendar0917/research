"""Targeted correctness tests for TCCD-v5 occurrence-preserving pair audit."""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v5 as V
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like


def _records(n_graphs: int = 4, seed: int = 0):
    layout = T.PatchLayout(capacity=8, n_atom=5, n_bond=3)
    frozen_dim = V.V2.frozen_layout().feature_dim
    ai = {c: i for i, c in enumerate(range(5))}
    bi = {c: i for i, c in enumerate(range(3))}
    out = []
    for s in range(n_graphs):
        rec = T.build_mol_record(synthetic_zinc_like(seed + s, 10 + s), layout, ai, bi)
        rec["X"] = np.random.default_rng(100 + s).standard_normal(
            (int(rec["n"]), frozen_dim)
        ).astype(np.float32)
        rec["y"] = float(s) / 3.0
        out.append(rec)
    return out, layout


def test_pair_geometry_and_phi_parameters():
    assert V.pair_descriptor_width() == 3 * V.K_PROTO + V.N_REL == 197
    assert V.frozen_base_dim() == T.h_dim(V.K_PROTO, V.N_REL) == 10464
    assert V.base_representation_dim() == 10464 + V.PHI_OUT
    assert V.pair_mlp_parameter_count() == 197 * 64 + 64 + 64 * 16 + 16


def test_gate0_checks_pass():
    checks = V.gate0_checks()
    for key in (
        "pair_swap_invariant",
        "pair_order_invariant",
        "linear_phi_pre_equals_post",
        "nonlinear_separation",
        "relabel_invariant_pre",
        "relabel_invariant_post",
        "batching_invariant",
        "empty_pair_zero",
        "target_independent",
        "official_test_blocked",
        "all_pass",
    ):
        assert checks[key] is True, key
    assert checks["official_test_loaded"] is False


def test_gradient_scope_stage_a():
    grads = V.gradient_checks()
    assert grads["stage_a_ok"] is True
    assert grads["stage_a_branch_grad"] > 0.0
    assert grads["stage_a_reader_grad"] > 0.0


def test_pair_cache_round_trip(tmp_path, monkeypatch):
    records, _ = _records(3)
    monkeypatch.setattr(V, "CACHE_DIR", tmp_path)
    cache = V.build_or_load_pair_cache(records, [0, 1, 2], "unit", "cpu", force=True)
    stats = cache.pair_statistics()
    assert stats["n_graphs"] == 3
    assert stats["relation_count"] == V.N_REL
    assert stats["pair_descriptor_width"] == 197
    assert stats["total_pairs"] == sum(int(r["n"]) * (int(r["n"]) - 1) // 2 for r in records)
    cache2 = V.build_or_load_pair_cache(records, [0, 1, 2], "unit", "cpu", force=False)
    assert np.array_equal(cache.C, cache2.C)
    assert np.array_equal(cache.pair_rel, cache2.pair_rel)


def test_stage_a_pre_and_post_differ_and_shuffle_changes():
    records, _ = _records(6)
    cache = V.build_or_load_pair_cache(records, list(range(6)), "unitB", "cpu", force=True)
    train = np.arange(4)
    dev = np.arange(4, 6)
    post = V.train_stage_a(cache, records, train, dev, "cpu", "post", seed=0, max_epochs=3, patience=3)
    pre = V.train_stage_a(cache, records, train, dev, "cpu", "pre", seed=0, max_epochs=3, patience=3)
    assert post.soup_valid is not None and pre.soup_valid is not None
    pre_model = V.StageAModelFactory.build("pre", seed=0)
    pre_model.load_state_dict(pre.state_best)
    real = V._evaluate_stage_a(pre_model, cache, records, dev, "cpu")
    shuf = V._evaluate_stage_a(pre_model, cache, records, dev, "cpu", shuffle=True)
    assert np.isfinite(real) and np.isfinite(shuf)


def test_e2e_model_gradients_and_dim():
    records, _ = _records(4)
    model = V.build_e2e_model("pre", seed=0)
    batch = V.make_e2e_batch(records, [0, 1, 2], "cpu")
    pred, C, h = model.forward_padded(batch)
    assert h.shape[1] == V.model_feature_dim()
    (pred - batch["y"]).abs().mean().backward()
    for name in ("W", "P", "temp_logit"):
        grad = getattr(model, name).grad
        assert grad is not None and float(grad.abs().sum()) > 0.0, name
    assert float(model.branch.fc1.weight.grad.abs().sum()) > 0.0
    assert float(model.head.weight.grad.abs().sum()) > 0.0
    assert model.uses_latent_bypass is False


def test_e2e_post_and_pre_are_distinct_modules():
    records, _ = _records(3)
    post = V.build_e2e_model("post", seed=0)
    pre = V.build_e2e_model("pre", seed=0)
    batch = V.make_e2e_batch(records, [0, 1, 2], "cpu")
    with torch.no_grad():
        h_post = post.forward_padded(batch)[2]
        h_pre = pre.forward_padded(batch)[2]
    base = V.V2.compose_padded(
        post.assign(batch["X_pad"].reshape(-1, post.F)).reshape(batch["X_pad"].shape[0], batch["X_pad"].shape[1], V.K_PROTO),
        batch["R_pad"],
        batch["valid"],
        batch["iu0"],
        batch["iu1"],
    )
    assert float((h_post[:, : V.frozen_base_dim()] - base).abs().max()) <= 1e-5
    assert float((h_pre[:, : V.frozen_base_dim()] - base).abs().max()) <= 1e-5
    assert float((h_post[:, V.frozen_base_dim():] - h_pre[:, V.frozen_base_dim():]).abs().max()) > 0.0


def test_stage_a_decision_rules():
    assert V.stage_a_decision(0.02, 0.02, 0.02, 0) == "PASS"
    assert V.stage_a_decision(0.004, 0.02, 0.02, 0) == "FAIL"
    assert V.stage_a_decision(0.008, 0.02, 0.02, 0) == "AMBIGUOUS_NEEDS_SEED1"
    seed0 = {"delta_place": 0.012, "delta_add": 0.012, "delta_shuffle": 0.012}
    assert V.stage_a_decision(0.012, 0.012, 0.012, 1, seed0=seed0) == "PASS"


def test_stage_b_decision_rules():
    assert V.stage_b_decision(0.03, 0.03, 0) == "PASS"
    assert V.stage_b_decision(0.001, 0.03, 0) == "FAIL"
    assert V.stage_b_decision(0.01, 0.03, 0) == "AMBIGUOUS_NEEDS_SEED1"
    seed0 = {"delta_place_e2e": 0.012, "delta_base_e2e": 0.012}
    assert V.stage_b_decision(0.012, 0.012, 1, seed0=seed0) == "PASS"


def test_official_authorization_routes():
    assert V.official_authorized(0.22) is True
    assert V.official_authorized(0.25) is False
    assert V.official_authorized(0.25, base_soup=0.29) is True
