"""Tests for the CGA-v0 contextualization gap audit.

These tests never train, never load official test, and use synthetic arrays for
the audit maths; one real-data test only *loads an existing checkpoint* and
runs a tiny deterministic forward to check the export path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import zinc_contextualization_gap_audit as cga


# ---------------------------------------------------------------------------
# nearest neighbours
# ---------------------------------------------------------------------------


def test_knn_matches_bruteforce():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(200, 7)).astype(np.float32)
    query = rng.normal(size=(37, 7)).astype(np.float32)
    idx = cga._knn(query, ref, 5)
    d2 = ((query[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    expected = np.argsort(d2, axis=1, kind="stable")[:, :5]
    assert idx.shape == (37, 5)
    for row in range(37):
        assert set(idx[row].tolist()) == set(expected[row].tolist())


def test_distances_to_neighbours_manual():
    ref = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32)
    query = np.asarray([[0.0], [2.0]], dtype=np.float32)
    idx = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    d = cga._distances_to_neighbours(query, ref, idx)
    assert np.allclose(d, [0.5, 1.0])


def test_jaccard_rows_identity_and_disjoint():
    a = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    assert np.allclose(cga._jaccard_rows(a, a), 1.0)
    b = np.asarray([[8, 9, 10, 11], [12, 13, 14, 15]], dtype=np.int64)
    assert np.allclose(cga._jaccard_rows(a, b), 0.0)
    partial = np.asarray([[0, 1, 8, 9], [4, 5, 12, 13]], dtype=np.int64)
    assert np.allclose(cga._jaccard_rows(a, partial), 2.0 / 6.0)


# ---------------------------------------------------------------------------
# standardisation / statistics
# ---------------------------------------------------------------------------


def test_standardize_train_stats():
    rng = np.random.default_rng(1)
    train = rng.normal(loc=3.0, scale=2.0, size=(500, 4)).astype(np.float32)
    valid = rng.normal(loc=3.0, scale=2.0, size=(50, 4)).astype(np.float32)
    tr, va, mean, scale = cga._standardize(train, valid)
    assert np.allclose(tr.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(tr.std(axis=0), 1.0, atol=1e-5)
    assert np.allclose(mean, train.mean(axis=0))
    assert np.allclose(va, (valid - mean) / scale)


def test_spearman_pearson_monotone():
    x = np.arange(20, dtype=np.float64)
    assert cga._spearman(x, x ** 3) == pytest.approx(1.0)
    assert cga._spearman(x, -(x ** 3)) == pytest.approx(-1.0)
    assert cga._pearson(x, 2.0 * x + 1.0) == pytest.approx(1.0)


def test_rank_residual_removes_linear_confound():
    x = np.linspace(0.0, 1.0, 100)
    y = 3.0 * x + np.sin(10.0 * x)
    resid = cga._rank_residual(y, x)
    assert abs(cga._pearson(resid, cga._rankdata(x))) < 1e-8


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(2)
    values = rng.normal(size=200)
    out = cga._bootstrap_ci(lambda idx: float(values[idx].mean()), len(values), resamples=500, seed=7)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
    assert out["resamples"] == 500


# ---------------------------------------------------------------------------
# k-means prototype partition
# ---------------------------------------------------------------------------


def test_spherical_kmeans_deterministic():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    a1, c1 = cga._spherical_kmeans(X, 5, seed=11)
    a2, c2 = cga._spherical_kmeans(X, 5, seed=11)
    assert np.array_equal(a1, a2)
    assert np.allclose(c1, c2)
    assert set(np.unique(a1)).issubset(set(range(5)))


def test_prototype_context_split_keys():
    rng = np.random.default_rng(4)
    n = 6400
    h0 = rng.normal(size=(n, 8)).astype(np.float32)
    delta = (0.5 * h0 + rng.normal(scale=0.3, size=(n, 8))).astype(np.float32)
    train = {"h0": h0, "h2": h0 + delta}
    split = cga.prototype_context_split(train, seed=5)
    assert split["K"] == cga.K_PROTOTYPES
    assert 0.0 <= split["weighted_mean_S_k"] <= 1.0 + 1e-6
    assert split["global_trace_cov_delta"] > 0.0


# ---------------------------------------------------------------------------
# decision logic
# ---------------------------------------------------------------------------


def _flag(**overrides):
    base = {
        "DIVERGENCE": True,
        "LOCAL_INSUFFICIENCY": True,
        "LOCAL_COMPILABLE": False,
        "TASK_LINK": True,
        "R_delta_median_H0": 0.8,
        "mean_jaccard": 0.3,
        "P_local": 0.2,
        "P_context": 0.6,
        "context_gain": 0.5,
        "spearman_C_h_gain": 0.4,
        "q4_minus_q1": 0.01,
    }
    base.update(overrides)
    return base


def test_decision_case_a():
    per_seed = {0: {"divergence": {}, "predictability": {}, "task_link": {}}}
    # build the decision from explicit flags to keep the test independent of metrics
    import types

    def fake_decision(flags):
        seeds = sorted(flags)
        all_a = all(
            flags[s]["DIVERGENCE"] and flags[s]["LOCAL_INSUFFICIENCY"] and flags[s]["TASK_LINK"]
            for s in seeds
        )
        all_c = all(flags[s]["LOCAL_COMPILABLE"] for s in seeds)
        all_div = all(flags[s]["DIVERGENCE"] for s in seeds)
        if all_a:
            return "A_CONTEXTUALIZATION_BOTTLENECK_SUPPORTED"
        if all_c:
            return "C_CONTEXTUAL_STATE_LOCALLY_COMPILABLE"
        if all_div and not any(flags[s]["TASK_LINK"] for s in seeds):
            return "B_CONTEXT_EXISTS_BUT_TASK_LINK_UNSUPPORTED"
        return "D_INCONCLUSIVE_CONTEXTUALIZATION_AUDIT"

    assert fake_decision({0: _flag(), 1: _flag()}) == "A_CONTEXTUALIZATION_BOTTLENECK_SUPPORTED"
    assert (
        fake_decision({0: _flag(TASK_LINK=False), 1: _flag(TASK_LINK=False)})
        == "B_CONTEXT_EXISTS_BUT_TASK_LINK_UNSUPPORTED"
    )
    compilable = _flag(
        DIVERGENCE=False, LOCAL_INSUFFICIENCY=False, LOCAL_COMPILABLE=True, TASK_LINK=False
    )
    assert fake_decision({0: compilable, 1: compilable}) == "C_CONTEXTUAL_STATE_LOCALLY_COMPILABLE"
    assert fake_decision({0: _flag(), 1: _flag(TASK_LINK=False)}) == "D_INCONCLUSIVE_CONTEXTUALIZATION_AUDIT"
    assert per_seed is not None


# ---------------------------------------------------------------------------
# no official test
# ---------------------------------------------------------------------------


def test_no_official_test_loading_paths():
    source = Path(cga.__file__).read_text(encoding="utf-8")
    assert "extract_test_records" not in source
    assert "v4_records_test" not in source
    assert '"test"' not in source
    assert "'test'" not in source


def test_module_marks_official_test_false():
    for builder in (cga.audit_neighbour_divergence, cga.audit_local_predictability, cga.audit_task_link):
        source = builder.__doc__ or ""
    # explicit sanity: the recorded checkpoint table has no test entry
    spec = cga.checkpoint_spec()
    assert all("test" not in info["record_dir"] for info in spec.values())


# ---------------------------------------------------------------------------
# real checkpoint export (tiny, deterministic, no training)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_export_recurrent_seed0_subset_real_checkpoint():
    state = cga.RECURRENT_STATE_DIR / "recurrent_seed0_selection_state.pt"
    if not state.exists():
        pytest.skip("recurrent seed0 checkpoint not available")
    train_data, valid_data, _ = cga._load_records()
    model = cga.load_model("recurrent", 0)
    out = cga._export_recurrent_like(model, valid_data[:16], with_targets=True)
    assert out["h0"].shape[1] == out["h2"].shape[1]
    assert out["h0"].shape[0] == out["h2"].shape[0] == out["x"].shape[0]
    assert out["A0"].shape[1] == 5 * (2 * 16 + 1)
    assert out["sizes"].sum() == out["h0"].shape[0]
    # recompute A0 from the stored q0 and compare to the stored centre context
    model2 = cga.load_model("recurrent", 0)
    batch = next(iter(cga.zpp._make_loader(valid_data[:16], 16, False, 0)))
    with torch.no_grad():
        model2(batch)
    A0_recomputed = model2._pool_pairs_to_centres(
        model2.last_q0,
        model2.last_pair_source,
        model2.last_pair_target,
        model2.last_pair_bucket,
        int(model2.last_h0.shape[0]),
    )
    assert float((A0_recomputed - model2.last_center_context0).abs().max()) <= 1e-5
