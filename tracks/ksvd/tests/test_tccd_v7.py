"""Targeted correctness tests for the TCCD-v7 normalized-moment representation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V2
from tracks.ksvd.code import tccd_v7 as V
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _records(n_graphs: int = 4, capacity: int = 8):
    layout = T.PatchLayout(capacity=capacity, n_atom=5, n_bond=3)
    ai = {c: i for i, c in enumerate(range(5))}
    bi = {c: i for i, c in enumerate(range(3))}
    out = []
    for seed in range(n_graphs):
        rec = T.build_mol_record(synthetic_zinc_like(seed, 10 + seed), layout, ai, bi)
        rec["y"] = float(seed)
        out.append(rec)
    return out, layout


def _tiny_nm_model(layout, *, with_moments: bool = True, seed: int = 0):
    init = np.random.default_rng(seed).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    return V.NMPrototypeModelFactory.build(
        layout.feature_dim,
        V.D_LOCAL,
        V.K_PROTO,
        V.N_REL,
        init,
        with_moments=with_moments,
        proto_seed=V.PROTO_INIT_SEED,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# geometry / layout / parameters
# ---------------------------------------------------------------------------
def test_dimensions_and_layout():
    assert V.K_PROTO == 64 and V.N_REL == 5
    assert V.TRI == 2080
    assert V.RAW_DIM == T.h_dim(V.K_PROTO, V.N_REL) == 10464
    assert V.NORM_DIM == 64 + 5 * 2080 + 1 + 5 == 10470
    assert V.NORM_MOM_DIM == 64 + 64 + 5 * 2080 + 1 + 5 == 10534
    assert V.representation_dim(False) == 10470
    assert V.representation_dim(True) == 10534
    assert V.reader_parameter_count(False) == 10471
    assert V.reader_parameter_count(True) == 10535

    layout_n = V.feature_layout(False)
    layout_nm = V.feature_layout(True)
    assert [b["name"] for b in layout_n] == ["mu", "Mhat_rel", "log_size", "log_mass"]
    assert [b["name"] for b in layout_nm] == ["mu", "v", "Mhat_rel", "log_size", "log_mass"]
    assert sum(b["dim"] for b in layout_n) == V.NORM_DIM
    assert sum(b["dim"] for b in layout_nm) == V.NORM_MOM_DIM
    assert "m2" not in [b["name"] for b in layout_nm]  # no deterministic redundancy

    assert V.parameter_accounting(True) == {
        "encoder": 45696,
        "prototypes": 4096,
        "temperature": 1,
        "reader": 10535,
        "total": 60328,
    }


# ---------------------------------------------------------------------------
# Gate 0 A: permutation invariance
# ---------------------------------------------------------------------------
def test_representation_is_permutation_invariant():
    C, R = V.synthetic_graph(9, seed=3)
    perm = np.asarray([4, 0, 8, 1, 7, 2, 6, 3, 5], dtype=np.int64)
    b0 = V._tensor_batch([C], [R], "cpu")
    b1 = V._tensor_batch([C[perm]], [R[:, perm][:, :, perm]], "cpu")
    for with_moments in (False, True):
        h0 = V.nm_graph_features(b0["C3"], b0["R_pad"], b0["valid"], b0["iu0"], b0["iu1"], with_moments=with_moments)
        h1 = V.nm_graph_features(b1["C3"], b1["R_pad"], b1["valid"], b1["iu0"], b1["iu1"], with_moments=with_moments)
        assert float((h0 - h1).abs().max()) <= 1e-6


def test_model_prediction_is_permutation_invariant():
    torch = T._torch()
    records, layout = _records(3)
    model = _tiny_nm_model(layout)
    model.eval()
    b = V2.make_batch(records, [0], "cpu")
    n = int(records[0]["n"])
    rng = np.random.default_rng(5)
    perm = rng.permutation(n)
    with torch.no_grad():
        p0, c0, h0 = model.forward_padded(b)
        Xp = b["X_pad"][:, :n].index_select(1, torch.as_tensor(perm))
        Rp = b["R_pad"][:, :, :n, :n].index_select(2, torch.as_tensor(perm)).index_select(3, torch.as_tensor(perm))
        bp = dict(b)
        bp["X_pad"] = torch.nn.functional.pad(Xp, (0, 0, 0, b["X_pad"].shape[1] - n))
        bp["R_pad"] = torch.zeros_like(b["R_pad"])
        bp["R_pad"][:, :, :n, :n] = Rp
        p1, c1, h1 = model.forward_padded(bp)
    assert float((p0 - p1).abs().max()) <= 1e-5
    assert float((h0 - h1).abs().max()) <= 1e-5


# ---------------------------------------------------------------------------
# Gate 0 B: batching / padding invariance
# ---------------------------------------------------------------------------
def test_batching_and_padding_invariance():
    C_list, R_list = ([], [])
    for i in range(4):
        C, R = V.synthetic_graph(5 + 2 * i, seed=90 + i)
        C_list.append(C)
        R_list.append(R)
    b = V._tensor_batch(C_list, R_list, "cpu")
    h_batched = V.nm_graph_features(b["C3"], b["R_pad"], b["valid"], b["iu0"], b["iu1"], with_moments=True)
    for i, (Ci, Ri) in enumerate(zip(C_list, R_list)):
        bi = V._tensor_batch([Ci], [Ri], "cpu")
        hi = V.nm_graph_features(bi["C3"], bi["R_pad"], bi["valid"], bi["iu0"], bi["iu1"], with_moments=True)
        assert float((h_batched[i] - hi[0]).abs().max()) <= 1e-6


# ---------------------------------------------------------------------------
# Gate 0 C: algebra
# ---------------------------------------------------------------------------
def test_mu_simplex_and_variance_identity():
    C, R = V.synthetic_graph(11, seed=8)
    b = V._tensor_batch([C], [R], "cpu")
    h = V.nm_graph_features(b["C3"], b["R_pad"], b["valid"], b["iu0"], b["iu1"], with_moments=True)
    mu = h[0, : V.MEAN_DIM].detach().cpu().numpy()
    v = h[0, V.MEAN_DIM : V.MEAN_DIM + V.MOM_DIM].detach().cpu().numpy()
    assert abs(float(mu.sum()) - 1.0) <= 1e-6
    Cn = C.astype(np.float64)
    mu_ref = Cn.sum(axis=0) / Cn.shape[0]
    m2_ref = (Cn**2).sum(axis=0) / Cn.shape[0]
    v_ref = np.maximum(m2_ref - mu_ref**2, 0.0)
    assert np.abs(v - v_ref).max() <= 1e-7
    assert v.min() >= 0.0


def test_variance_clamp_and_zero_mass_relation():
    C, R = V.synthetic_graph(7, seed=12, zero_relation=2)
    b = V._tensor_batch([C], [R], "cpu")
    h = V.nm_graph_features(b["C3"], b["R_pad"], b["valid"], b["iu0"], b["iu1"], with_moments=True)
    assert bool(T._torch().isfinite(h).all())
    vec = h[0, V.MEAN_DIM + V.MOM_DIM : V.MEAN_DIM + V.MOM_DIM + V.REL_DIM].detach().cpu().numpy()
    zero_block = vec[2 * V.TRI : 3 * V.TRI]
    assert float(np.abs(zero_block).max()) == 0.0
    other = np.concatenate([vec[: 2 * V.TRI], vec[3 * V.TRI :]])
    assert float(np.abs(other).max()) > 0.0


# ---------------------------------------------------------------------------
# Gate 0 D: normalized composition mass (full symmetric rebuild)
# ---------------------------------------------------------------------------
def test_normalized_mass_equals_one_on_full_symmetric_rebuild():
    C, R = V.synthetic_graph(9, seed=21)
    b = V._tensor_batch([C], [R], "cpu")
    for with_moments in (False, True):
        h = V.nm_graph_features(b["C3"], b["R_pad"], b["valid"], b["iu0"], b["iu1"], with_moments=with_moments)
        off = V.MEAN_DIM + (V.MOM_DIM if with_moments else 0)
        vec = h[0, off : off + V.REL_DIM].detach().cpu().numpy()
        for r in range(V.N_REL):
            M = V.full_symmetric_from_upper(vec[r * V.TRI : (r + 1) * V.TRI])
            assert abs(float(M.sum()) - 1.0) <= 1e-4
    # the storage really is the upper triangle only: rebuilding is not idempotent
    vec = h[0, V.MEAN_DIM + V.MOM_DIM : V.MEAN_DIM + V.MOM_DIM + V.REL_DIM].detach().cpu().numpy()
    M0 = V.full_symmetric_from_upper(vec[: V.TRI])
    assert float(M0.sum()) > float(vec[: V.TRI].sum())


def test_reference_numpy_implementation_matches_tensor_path():
    C, R = V.synthetic_graph(13, seed=31, scale=1.2)
    b = V._tensor_batch([C], [R], "cpu")
    for with_moments in (False, True):
        h = V.nm_graph_features(b["C3"], b["R_pad"], b["valid"], b["iu0"], b["iu1"], with_moments=with_moments)
        ref = V.reference_features_np(C, R, with_moments=with_moments)
        got = h[0].detach().cpu().numpy().astype(np.float64)
        assert np.abs(got - ref).max() <= 1e-6 * max(np.abs(ref).max(), 1.0)


def test_relation_mass_and_mhat_denominator_are_consistent():
    C, R = V.synthetic_graph(10, seed=41)
    s = R.astype(np.float64).sum(axis=(1, 2))
    assert np.all(s > 0)
    Cn = C.astype(np.float64)
    iu0, iu1 = T.sym_indices(V.K_PROTO)
    for r in range(V.N_REL):
        M = Cn.T @ R[r].astype(np.float64) @ Cn
        assert abs(float(M.sum()) - float(s[r])) <= 1e-6 * max(float(s[r]), 1.0)
        Mhat = M / max(float(s[r]), V.EPS)
        assert abs(float(Mhat.sum()) - 1.0) <= 1e-6
        stored = Mhat[iu0, iu1]
        assert abs(float(V.full_symmetric_from_upper(stored).sum()) - 1.0) <= 1e-6


# ---------------------------------------------------------------------------
# Gate 0 F: no target leakage
# ---------------------------------------------------------------------------
def test_features_do_not_depend_on_labels():
    C, R = V.synthetic_graph(8, seed=51)
    b = V._tensor_batch([C], [R], "cpu")
    h_a = V.nm_graph_features(b["C3"], b["R_pad"], b["valid"], b["iu0"], b["iu1"], with_moments=True)
    h_b = V.nm_graph_features(b["C3"], b["R_pad"], b["valid"], b["iu0"], b["iu1"], with_moments=True)
    assert float((h_a - h_b).abs().max()) == 0.0

    import inspect

    for fn in (V.nm_graph_features, V.reference_features_np, V.stage_a_features):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"y", "target", "label", "labels", "targets"})


def test_stage_a_features_never_read_labels():
    """The Stage-A builder must not touch any label-bearing attribute."""
    cache = _tiny_cache()

    class _NoLabelCache(type(cache)):  # type: ignore[misc]
        @property
        def y(self):  # pragma: no cover - only reached on a contract violation
            raise AssertionError("stage_a_features must not read labels")

        @property
        def pair_y(self):  # pragma: no cover
            raise AssertionError("stage_a_features must not read labels")

        @property
        def labels(self):  # pragma: no cover
            raise AssertionError("stage_a_features must not read labels")

    fields = [f for f in cache.__dataclass_fields__.values() if f.init]
    guarded = _NoLabelCache(**{f.name: getattr(cache, f.name) for f in fields})
    feats, diag = V.stage_a_features(guarded, with_moments=True)
    assert feats.shape == (cache.n_graphs, V.NORM_MOM_DIM)
    assert diag["target_not_read"] is True
    assert diag["mu_row_sum_max_abs_dev"] <= 1e-5
    plain, _ = V.stage_a_features(cache, with_moments=True)
    assert np.array_equal(feats, plain)


# ---------------------------------------------------------------------------
# cache <-> tensor equivalence
# ---------------------------------------------------------------------------
def _tiny_cache(n_graphs: int = 3, seed: int = 0):
    """Cache whose base is built exactly from C and R (real relation conventions)."""
    from tracks.ksvd.code.tccd_v6 import LabelFreeCache

    C_list, i0_list, i1_list, rel_list, base_rows = [], [], [], [], []
    iu0, iu1 = T.sym_indices(V.K_PROTO)
    for g in range(n_graphs):
        C, R = V.synthetic_graph(6 + g, seed=seed + g)
        C_list.append(C)
        n = C.shape[0]
        a0, a1 = np.triu_indices(n, 1)
        i0_list.append(a0.astype(np.int64))
        i1_list.append(a1.astype(np.int64))
        rel_list.append(np.ascontiguousarray(R[:, a0, a1].T, dtype=np.float32))
        blocks = [C.sum(axis=0)]
        for r in range(V.N_REL):
            M = C.T.astype(np.float64) @ R[r].astype(np.float64) @ C.astype(np.float64)
            blocks.append(M[iu0, iu1])
        base_rows.append(np.concatenate(blocks).astype(np.float32))
    n_patches = np.asarray([c.shape[0] for c in C_list])
    n_pairs = np.asarray([len(a) for a in i0_list])
    return LabelFreeCache(
        base=np.stack(base_rows, axis=0),
        C=np.concatenate(C_list),
        c_offsets=np.concatenate([[0], np.cumsum(n_patches)]).astype(np.int64),
        pair_i0=np.concatenate(i0_list),
        pair_i1=np.concatenate(i1_list),
        pair_rel=np.concatenate(rel_list),
        pair_offsets=np.concatenate([[0], np.cumsum(n_pairs)]).astype(np.int64),
        graph_index=np.arange(n_graphs, dtype=np.int64),
        meta={"protocol": "test", "official_test_loaded": False},
    )


def test_cache_and_tensor_feature_paths_agree():
    cache = _tiny_cache(3)
    rows = list(range(cache.n_graphs))
    C_list, R_list = V.tensor_inputs_from_cache(cache, rows)
    b = V._tensor_batch(C_list, R_list, "cpu")
    for with_moments in (False, True):
        h = V.nm_graph_features(b["C3"], b["R_pad"], b["valid"], b["iu0"], b["iu1"], with_moments=with_moments)
        feats, _ = V.stage_a_features(cache, with_moments=with_moments, rows=rows)
        assert np.abs(h.detach().cpu().numpy() - feats).max() <= 1e-5


def test_relation_mass_from_cache_matches_direct_matrix_sum():
    cache = _tiny_cache(2, seed=7)
    C_list, R_list = V.tensor_inputs_from_cache(cache, list(range(cache.n_graphs)))
    s_cache = V.relation_masses_from_cache(cache)
    for g, R in enumerate(R_list):
        s_direct = R.astype(np.float64).sum(axis=(1, 2))
        assert np.abs(s_cache[g] - s_direct).max() <= 1e-6


def test_raw_features_are_the_frozen_base():
    cache = _tiny_cache(2)
    assert np.array_equal(V.raw_features(cache), cache.base.astype(np.float32))


# ---------------------------------------------------------------------------
# Stage-B model contract
# ---------------------------------------------------------------------------
def test_nm_model_parameterization_and_representation_dim():
    torch = T._torch()
    records, layout = _records(3)
    model = _tiny_nm_model(layout)
    assert model.graph_feature_kind == "normalized_moment_representation_v1"
    assert model.uses_latent_bypass is False
    assert model.head.in_features == V.NORM_MOM_DIM
    assert model.head.out_features == 1
    assert model.W.shape == (layout.feature_dim, V.D_LOCAL)
    assert model.P.shape == (V.D_LOCAL, V.K_PROTO)
    b = V2.make_batch(records, [0, 1], "cpu")
    with torch.no_grad():
        pred, C, h = model.forward_padded(b)
    assert h.shape[1] == V.NORM_MOM_DIM
    assert pred.shape == (2,)
    assert V2.TEMP_MIN < float(model.temperature()) < 1.0
    assert abs(float(model.temperature()) - V2.TEMP_INIT) < 0.1


def test_nm_model_norm_variant_has_no_variance_block():
    records, layout = _records(2)
    model_plain = _tiny_nm_model(layout, with_moments=False)
    assert model_plain.head.in_features == V.NORM_DIM
    b = V2.make_batch(records, [0], "cpu")
    with T._torch().no_grad():
        _, _, h = model_plain.forward_padded(b)
    assert h.shape[1] == V.NORM_DIM


def test_task_gradient_reaches_encoder_prototypes_temperature_and_reader():
    records, layout = _records(3)
    model = _tiny_nm_model(layout)
    b = V2.make_batch(records, [0, 1, 2], "cpu")
    pred, _, _ = model.forward_padded(b)
    (pred - b["y"]).abs().mean().backward()
    assert float(model.W.grad.abs().sum()) > 0
    assert float(model.P.grad.abs().sum()) > 0
    assert float(model.temp_logit.grad.abs().sum()) > 0
    assert float(model.head.weight.grad.abs().sum()) > 0


def test_end_to_end_train_step_reduces_train_loss_and_keeps_rows_simplex():
    torch = T._torch()
    records, layout = _records(4)
    model = _tiny_nm_model(layout)
    b = V2.make_batch(records, [0, 1, 2, 3], "cpu")
    reg = V2._initial_regularization(model, records, [0, 1, 2, 3], "cpu", batch=2, log=lambda *a: None)
    assert reg["lambda_local"] > 0 and reg["lambda_balance"] > 0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(6):
        opt.zero_grad(set_to_none=True)
        pred, C, _ = model.forward_padded(b)
        valid = b["valid"].reshape(-1)
        loss = (
            (pred - b["y"]).abs().mean()
            + reg["lambda_local"] * V2.local_entropy(C, valid)
            + reg["lambda_balance"] * V2.balance_kl(C, valid)
        )
        loss.backward()
        opt.step()
        model.renormalize_()
        if first is None:
            first = float(loss.detach())
    with torch.no_grad():
        _, C, _ = model.forward_padded(b)
    rows = C[b["valid"].reshape(-1)]
    assert float((rows.sum(1) - 1).abs().max()) <= 1e-6
    assert float(rows.min()) >= -1e-7
    assert T._torch().isfinite(model.head.weight).all()


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------
def test_stage_a_decision_pass_fail_and_drift():
    passed = V.stage_a_decision(
        raw_rescreen_soup=0.2783, norm_soup=0.2560, nm_soup=0.2530
    )
    assert passed["verdict"] == "PASS" and passed["pass"] is True
    assert abs(passed["delta_norm"] - (V.RAW_SOUP_MAE - 0.2530)) < 1e-12
    assert passed["delta_mom"] == pytest.approx(0.003, abs=1e-6)
    assert passed["variance_diagnostic"] == V.VARIANCE_NEGLIGIBLE

    material = V.stage_a_decision(raw_rescreen_soup=0.2783, norm_soup=0.2650, nm_soup=0.2530)
    assert material["variance_diagnostic"] == V.VARIANCE_MATERIAL

    # absolute gate: NM soup above 0.255 is a FAIL even with a large delta
    too_big = V.stage_a_decision(raw_rescreen_soup=0.2783, norm_soup=0.2620, nm_soup=0.2560)
    assert too_big["verdict"] == "FAIL"
    assert too_big["checks"]["nm_soup_abs_ok"] is False

    # small delta is a FAIL
    small = V.stage_a_decision(raw_rescreen_soup=0.2783, norm_soup=0.2750, nm_soup=0.2700)
    assert small["verdict"] == "FAIL" and small["checks"]["delta_norm_ok"] is False

    # protocol drift blocks Stage B even if the numbers look good
    drift = V.stage_a_decision(raw_rescreen_soup=0.2710, norm_soup=0.2560, nm_soup=0.2530)
    assert drift["verdict"] == "PROTOCOL_DRIFT"
    assert drift["stage_b_authorized"] is False


def test_stage_b_cases():
    assert V.stage_b_decision(0.2400)["case"] == "S"
    assert V.stage_b_decision(0.2400)["stop"] is True
    assert V.stage_b_decision(0.2301)["case"] == "S"
    assert V.stage_b_decision(0.2300)["case"] == "P"
    assert V.stage_b_decision(0.2100)["case"] == "P"
    assert V.stage_b_decision(0.2100)["official_valid_authorized"] is True
    assert V.stage_b_decision(0.2000)["case"] == "G"
    assert V.stage_b_decision(0.1500)["case"] == "G"


def test_official_bands():
    assert V.official_band(0.1400) == "COMPETITIVE_ISH"
    assert V.official_band(0.1500) == "COMPETITIVE_ISH"
    assert V.official_band(0.1501) == "SUBSTANTIAL_BUT_INCOMPLETE"
    assert V.official_band(0.2000) == "SUBSTANTIAL_BUT_INCOMPLETE"
    assert V.official_band(0.2001) == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# execution discipline
# ---------------------------------------------------------------------------
def test_official_test_is_unreachable_and_references_are_frozen():
    from tracks.ksvd.code import run_tccd_v7 as R

    source = Path(R.__file__).read_text(encoding="utf-8")
    for forbidden in ("official_test", "load_mols(args.data_root, \"test\")", "\"test\""):
        if forbidden == "official_test":
            assert "official_test_loaded" in source  # only the false flag is present
            continue
        assert forbidden not in source.replace("'test'", "")
    assert R.V.RAW_SOUP_MAE == 0.2783639132976532
    assert R.V.RAW_BEST_MAE == 0.28791576623916626
    assert R.V.V2_INTERNAL_BEST_MAE == 0.2862437069416046
    assert R.V.V2_INTERNAL_SOUP_MAE == 0.26635152101516724
    assert R.V.CANONICAL_GPU1_BASELINE == 0.119818
    assert R.V.STAGE_A_NM_ABS_MAX == 0.255
    assert R.V.STAGE_A_DELTA_NORM_MIN == 0.015
    assert R.V.DELTA_MOM_MATERIAL == 0.005
    assert R.V.STAGE_B_STOP_GT == 0.23
    assert R.V.STAGE_B_PARTIAL_GT == 0.20
    with pytest.raises(RuntimeError):
        T.load_mols(Path("/nonexistent"), "test")


def test_gate0_data_free_checks_pass():
    checks = V.gate0_data_free_checks("cpu")
    assert checks["data_free_all_pass"] is True
    assert checks["dims_ok"] is True
    assert checks["official_test_loaded"] is False
    assert checks["target_not_read"] is True


def test_stage_a_reference_artifact_is_frozen():
    if not (V.V5_STAGE_A_JSON.exists() and V.V2_BEST_CHECKPOINT.exists()):
        pytest.skip("frozen TCCD-v5/v2 artifacts are not available in this checkout")
    assert V._sha256_file(V.V5_STAGE_A_JSON) == V.V5_STAGE_A_SHA256
    payload = json.loads(V.V5_STAGE_A_JSON.read_text(encoding="utf-8"))
    base = payload["results"]["BASE"]
    assert int(base["feature_dim"]) == V.RAW_DIM
    assert abs(float(base["soup_valid_mae"]) - V.RAW_SOUP_MAE) < 1e-12
    assert abs(float(base["best_valid_mae"]) - V.RAW_BEST_MAE) < 1e-12
    assert V.V2_BEST_CHECKPOINT.exists()
    assert V._sha256_file(V.V2_BEST_CHECKPOINT) == V.V2_BEST_CHECKPOINT_SHA256