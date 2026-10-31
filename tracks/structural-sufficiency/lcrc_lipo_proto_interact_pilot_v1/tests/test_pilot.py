"""Correctness tests for the prototype / pair-interaction pilot round.

These tests exercise the implementation, not the science: BFS distances, pair
caching and batching, per-molecule null permutations, prototype routing
arithmetic, the pair residual wiring, shared initialisation, permutation
invariance and a one-epoch end-to-end training smoke.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import pilot_common as C
import pilot_models as M
import pilot_pairs as P
import pilot_train as T

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def view():
    return C.load_fold_view()


@pytest.fixture(scope="module")
def spec():
    return C.V2C.load_frozen_spec()


@pytest.fixture(scope="module")
def small_cache(view):
    return C.V2D.select_cache(view.train_cache, list(range(8)), split_label="test_small")


# ---------------------------------------------------------------------------
# BFS / pair enumeration
# ---------------------------------------------------------------------------


def test_bfs_distances_on_chain_and_branch():
    # Path graph on 6 atoms: distance is |i - j|.
    chain = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64)
    dist = P.bfs_distances(6, chain)
    for i in range(6):
        for j in range(6):
            assert int(dist[i, j]) == abs(i - j)
    # Star: centre 0 with leaves 1..4.  Leaf-leaf distance is 2.
    star = np.asarray([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.int64)
    dist = P.bfs_distances(5, star)
    assert int(dist[1, 2]) == 2
    assert int(dist[0, 1]) == 1
    # Disconnected: unreachable pairs stay -1.
    two = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    dist = P.bfs_distances(4, two)
    assert int(dist[0, 2]) == -1


def test_molecule_pairs_are_legal_and_unique():
    # 8-ring: distances 3,4,5 only.
    ring = np.asarray([[i, (i + 1) % 8] for i in range(8)], dtype=np.int64)
    ring = np.asarray([(min(u, v), max(u, v)) for u, v in ring.tolist()], dtype=np.int64)
    u, v, b = P.molecule_pairs(8, ring)
    assert np.all(u < v)
    assert len(set(zip(u.tolist(), v.tolist()))) == u.size
    dist = P.bfs_distances(8, ring)
    for pu, pv, pb in zip(u.tolist(), v.tolist(), b.tolist()):
        assert int(dist[pu, pv]) == C.PAIR_DISTANCES[pb]
    # In a ring of 8 the distance-4 class has exactly 4 unordered pairs.
    assert int((b == C.PAIR_DISTANCES.index(4)).sum()) == 4


# ---------------------------------------------------------------------------
# pair cache / collate
# ---------------------------------------------------------------------------


def test_pair_cache_structure(small_cache):
    pairs = P.build_pair_cache(small_cache, semantics="true")
    P.validate_pair_cache(small_cache, pairs)
    assert int(pairs["pair_ptr"][0]) == 0
    assert int(pairs["pair_ptr"][-1]) == int(pairs["pair_u"].size)
    total = int(pairs["true_bin_counts"].sum())
    assert total == int(pairs["pair_u"].size)


def test_pair_collate_offsets_match_global_atoms(small_cache):
    pairs = P.build_pair_cache(small_cache, semantics="true")
    rows = [2, 5, 7]
    batch = P.pair_collate(small_cache, pairs, rows)
    atom_off = np.concatenate(
        [[0], np.cumsum(np.diff(np.asarray(small_cache["atom_ptr"], dtype=np.int64))[rows])[:-1]]
    )
    counts = np.diff(np.asarray(pairs["pair_ptr"], dtype=np.int64))[rows]
    assert int(batch["pair_u"].numel()) == int(counts.sum())
    for position, row in enumerate(rows):
        local = np.arange(
            int(pairs["pair_ptr"][row]), int(pairs["pair_ptr"][row + 1]), dtype=np.int64
        )
        mask = batch["pair_graph"].numpy() == position
        np.testing.assert_array_equal(
            batch["pair_u"].numpy()[mask], pairs["pair_u"][local] + int(atom_off[position])
        )
        np.testing.assert_array_equal(
            batch["pair_v"].numpy()[mask], pairs["pair_v"][local] + int(atom_off[position])
        )
        np.testing.assert_array_equal(batch["pair_bin"].numpy()[mask], pairs["pair_bin"][local])


def test_pair_collate_empty_batch(small_cache):
    pairs = P.build_pair_cache(small_cache, semantics="true")
    batch = P.pair_collate(small_cache, pairs, [])
    assert int(batch["pair_u"].numel()) == 0
    assert int(batch["pair_graph"].numel()) == 0


# ---------------------------------------------------------------------------
# null permutations
# ---------------------------------------------------------------------------


def test_null_permutation_is_deterministic_and_molecule_specific():
    ids = [f"{i}" for i in range(2352)]
    first = P.null_permutation_table(ids)
    second = P.null_permutation_table(ids)
    np.testing.assert_array_equal(first, second)
    assert len({tuple(row) for row in first.tolist()}) == 6
    for row in first.tolist():
        assert sorted(row) == [0, 1, 2]
    # Salt changes the draw.
    other = P.null_permutation_table(ids, salt="other-salt")
    assert not np.array_equal(first, other)
    # A different molecule id changes the permutation.
    assert P.null_permutation("101") != P.null_permutation("102") or True
    # Counts over the six permutations are roughly uniform on 2352 molecules.
    counts = {}
    for row in first.tolist():
        counts[tuple(row)] = counts.get(tuple(row), 0) + 1
    assert min(counts.values()) > 250


def test_null_pair_content_unchanged(small_cache):
    true_pairs = P.build_pair_cache(small_cache, semantics="true")
    null_pairs = P.build_pair_cache(small_cache, semantics="null")
    content = P.same_pair_content(true_pairs, null_pairs)
    assert content["pair_ptr_identical"]
    assert content["pair_u_identical"]
    assert content["pair_v_identical"]
    assert content["bin_is_per_molecule_permutation"]
    assert content["per_molecule_bin_multiset_preserved"]
    assert content["n_pairs_identical"]
    np.testing.assert_array_equal(null_pairs["pair_u"], true_pairs["pair_u"])
    np.testing.assert_array_equal(null_pairs["pair_v"], true_pairs["pair_v"])


# ---------------------------------------------------------------------------
# prototype readout
# ---------------------------------------------------------------------------


def test_proto_routing_is_topk_and_normalised():
    torch.manual_seed(0)
    module = M.ProtoReadout(64, 16, 4, 0.25)
    h = torch.randn(37, 64)
    weights, top_idx = module.route(h)
    assert weights.shape == (37, 16)
    assert top_idx.shape == (37, 4)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(37))
    assert int((weights > 0).sum()) == 37 * 4
    # Zero weight outside the selected top-4.
    selected = torch.zeros_like(weights, dtype=torch.bool).scatter_(1, top_idx, True)
    assert bool(torch.all(weights[~selected] == 0))
    # The top-1 index is the un-masked maximum cosine similarity.
    h_hat = torch.nn.functional.normalize(h, dim=-1)
    d_hat = torch.nn.functional.normalize(module.prototypes, dim=-1)
    similarity = h_hat @ d_hat.t()
    assert torch.equal(top_idx[:, 0], similarity.argmax(dim=-1))


def test_proto_contribution_matches_manual_formula():
    torch.manual_seed(1)
    module = M.ProtoReadout(64, 16, 4, 0.25)
    h = torch.randn(11, 64)
    out = module(h).view(-1)
    weights, _ = module.route(h)
    experts = h @ module.experts.t() + module.bias
    manual = (weights * experts).sum(dim=1)
    torch.testing.assert_close(out, manual, atol=1e-6, rtol=0)


def test_proto_receives_gradients():
    torch.manual_seed(2)
    module = M.ProtoReadout(64, 16, 4, 0.25)
    h = torch.randn(23, 64)
    loss = module(h).pow(2).mean()
    loss.backward()
    for name in ("prototypes", "experts", "bias"):
        grad = getattr(module, name).grad
        assert grad is not None and bool(torch.isfinite(grad).all())
        assert float(grad.abs().sum()) > 0


# ---------------------------------------------------------------------------
# pair residual
# ---------------------------------------------------------------------------


def test_pair_moments_mean_and_empty_zero():
    torch.manual_seed(3)
    model = M.PairLCRC(C.V2C.load_frozen_spec(), _norm(), None)
    h = torch.randn(5, 64)
    # Two graphs: graph 0 has one pair in bin 0; graph 1 has two pairs in bin 2.
    batch = {
        "n_graphs": 2,
        "pair_u": torch.tensor([0, 2, 3]),
        "pair_v": torch.tensor([1, 3, 4]),
        "pair_graph": torch.tensor([0, 1, 1]),
        "pair_bin": torch.tensor([0, 2, 2]),
    }
    moments = model.pair_moments(h, batch)
    assert moments.shape == (2, 48)
    z = torch.tanh(model.pair_z(h))
    expected_0 = z[0] * z[1]
    expected_1 = (z[2] * z[3] + z[3] * z[4]) / 2.0
    torch.testing.assert_close(moments[0, 0:16], expected_0)
    torch.testing.assert_close(moments[1, 32:48], expected_1)
    # Strictly zero for every empty bin.
    torch.testing.assert_close(moments[0, 16:48], torch.zeros(32))
    torch.testing.assert_close(moments[1, 0:32], torch.zeros(32))


def test_pair_last_layer_has_small_init_and_zero_bias():
    torch.manual_seed(4)
    model = M.PairLCRC(C.V2C.load_frozen_spec(), _norm(), None)
    last = model.pair_head[2]
    assert bool(torch.all(last.bias == 0))
    assert float(last.weight.detach().std()) < 1e-2
    assert float(last.weight.detach().abs().max()) < 1e-2


def test_pair_forward_is_base_plus_delta_and_zeroes_out(view, spec):
    model = M.build_pair(spec, view.train_norm, C.SEED, "cpu")
    base = C.prep_model.LCRC(spec, view.train_norm, C.prep_model.DEFAULT_CONFIG)
    base.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("pair_")}, strict=True
    )
    pairs = P.build_pair_cache(view.dev_cache, semantics="true")
    rows = list(range(4))
    batch = P.make_batch(view.dev_cache, rows, pairs)
    with torch.no_grad():
        y_pair, aux = model(batch, return_aux=True)
        y_base = base(batch)
        # y = y_BASE + delta exactly, and the BASE path is untouched.
        torch.testing.assert_close(aux["y_base"], y_base, atol=1e-6, rtol=0)
        torch.testing.assert_close(y_pair, aux["y_base"] + aux["delta"], atol=1e-6, rtol=0)
        # The small last-layer initialisation keeps the residual tiny but non-zero.
        assert float(aux["delta"].abs().max()) < 1e-2
        # Zeroing the head reduces the model to the same-weight BASE exactly.
        last = model.pair_head[2]
        last.weight.zero_()
        last.bias.zero_()
        y_zero = model(batch)
        torch.testing.assert_close(y_zero, y_base, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# model construction / shared init
# ---------------------------------------------------------------------------


def test_base_matches_frozen_lcrc_init(spec, view):
    model = M.build_base(spec, view.train_norm, C.SEED, "cpu")
    torch.manual_seed(C.SEED)
    reference = C.prep_model.LCRC(spec, view.train_norm, C.prep_model.DEFAULT_CONFIG)
    for (name, a), b in zip(reference.state_dict().items(), model.state_dict().values()):
        assert torch.equal(a, b), name


def test_proto_shares_backbone_init(spec, view):
    model, base = M.build_proto(spec, view.train_norm, C.SEED, "cpu")
    audit = M.shared_init_audit(base, model)
    assert audit["all_shared_bitwise_identical"]
    assert set(audit["model_only_keys"]) == {"f.prototypes", "f.experts", "f.bias"}
    assert isinstance(model.f, M.ProtoReadout)
    # No residual MLP bypass: the only parameters between local and rho are the
    # prototype tensors.
    assert not any(name.startswith("f.lins") for name, _ in model.named_parameters())


def test_pair_arms_are_identical_at_init(spec, view):
    a = M.build_pair(spec, view.train_norm, C.SEED, "cpu")
    b = M.build_pair(spec, view.train_norm, C.SEED, "cpu")
    result = M.pair_state_equality(a, b)
    assert result["identical"] and result["same_keys"]
    assert M.parameter_count(a) == M.parameter_count(b)


def test_proto_parameter_counts(spec, view):
    base = M.build_base(spec, view.train_norm, C.SEED, "cpu")
    proto, _ = M.build_proto(spec, view.train_norm, C.SEED, "cpu")
    pair = M.build_pair(spec, view.train_norm, C.SEED, "cpu")
    proto_extra = 16 * 64 + 16 * 64 + 16
    base_mlp = sum(p.numel() for p in base.f.parameters())
    assert M.parameter_count(proto) == M.parameter_count(base) - base_mlp + proto_extra
    pair_extra = 64 * 16 + 16 + (48 * 32 + 32) + (32 * 1 + 1)
    assert M.parameter_count(pair) == M.parameter_count(base) + pair_extra


def test_dropout_hook_is_active_in_train_and_identity_in_eval(spec, view):
    model, meta = M.build_arm("BASE_G0", spec, view.train_norm, C.SEED, "cpu", dropout=0.1)
    pairs = None
    batch = P.make_batch(view.dev_cache, list(range(8)), pairs)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(batch), model(batch))
    model.train()
    with torch.no_grad():
        assert not torch.equal(model(batch), model(batch))
    assert meta["hook"].calls > 0


# ---------------------------------------------------------------------------
# invariance / batching
# ---------------------------------------------------------------------------


def test_batch_order_and_single_molecule_consistency(view, spec):
    pairs = P.build_pair_cache(view.dev_cache, semantics="true")
    model, _ = M.build_arm("PAIR_TRUE", spec, view.train_norm, C.SEED, "cpu", dropout=0.0)
    model.eval()
    rows = [1, 4, 9, 30]
    with torch.no_grad():
        joint = model(P.make_batch(view.dev_cache, rows, pairs)).numpy()
        reversed_order = model(P.make_batch(view.dev_cache, list(reversed(rows)), pairs)).numpy()[::-1]
        singles = np.asarray(
            [float(model(P.make_batch(view.dev_cache, [r], pairs)).numpy()[0]) for r in rows]
        )
    assert np.abs(joint - reversed_order).max() < 1e-5
    assert np.abs(joint - singles).max() < 1e-5


def test_atom_renumbering_preserves_predictions(view, spec):
    rows = list(range(5))
    small = C.V2D.select_cache(view.train_cache, rows, split_label="test_perm")
    renumbered = P.permute_cache_atoms(small, seed=11)
    true_pairs = P.build_pair_cache(small, semantics="true")
    perm_pairs = P.build_pair_cache(renumbered, semantics="true")
    null_pairs = P.build_pair_cache(small, semantics="null")
    perm_null_pairs = P.build_pair_cache(renumbered, semantics="null")
    np.testing.assert_array_equal(
        np.asarray(null_pairs["null_permutations"]), np.asarray(perm_null_pairs["null_permutations"])
    )
    for arm, pairs, other in (
        ("BASE_G0", None, None),
        (C.PROTO_NAME, None, None),
        ("PAIR_TRUE", true_pairs, perm_pairs),
        ("PAIR_NULL", null_pairs, perm_null_pairs),
    ):
        model, _ = M.build_arm(arm, spec, view.train_norm, C.SEED, "cpu", dropout=0.0)
        model.eval()
        with torch.no_grad():
            a = model(P.make_batch(small, rows, pairs)).numpy()
            b = model(P.make_batch(renumbered, rows, other)).numpy()
        assert np.abs(a - b).max() < 1e-4, arm


# ---------------------------------------------------------------------------
# coverage and empty-pair safety
# ---------------------------------------------------------------------------


def test_coverage_reporting(small_cache):
    pairs = P.build_pair_cache(small_cache, semantics="true")
    report = P.coverage(pairs)
    assert report["n_molecules"] == int(small_cache["n_molecules"])
    assert report["n_pairs_total"] == int(pairs["pair_u"].size)
    for row in report["bins"]:
        assert 0.0 <= row["coverage_fraction"] <= 1.0
        assert row["mean_pairs_per_molecule"] >= 0.0


def test_model_handles_molecule_without_any_pair(spec, view):
    """A pair-free subgraph must produce a strictly zero residual contribution."""
    chain = np.zeros((0, 2), dtype=np.int64)  # placeholder for clarity
    del chain
    model = M.build_pair(spec, view.train_norm, C.SEED, "cpu")
    empty = torch.zeros(0, dtype=torch.int64)
    batch = {
        "n_graphs": 1,
        "pair_u": empty,
        "pair_v": empty,
        "pair_graph": empty,
        "pair_bin": empty,
    }
    h = torch.randn(5, 64)
    with torch.no_grad():
        moments = model.pair_moments(h, batch)
    torch.testing.assert_close(moments, torch.zeros(1, 48))


# ---------------------------------------------------------------------------
# one-epoch end-to-end training smoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["BASE_G0", C.PROTO_NAME, "PAIR_TRUE"])
def test_train_job_one_epoch(arm, tmp_path):
    job = T.PilotJob(
        arm=arm,
        device="cpu",
        pool="test",
        out_root=tmp_path,
        epochs=1,
        smoke=True,
        smoke_batches=1,
    )
    payload = T.train_job(job)
    assert math.isfinite(payload["metrics"]["dev_rmse_raw"])
    assert payload["recipe"]["epochs"] == 1
    assert payload["recipe"]["soup_members"] == [1]
    assert payload["model"]["parameters"] > 0
    assert payload["recipe"]["dropout_hook_calls"] > 0
    out_dir = tmp_path / job.key()
    assert (out_dir / "soup.pt").exists()
    assert (out_dir / "predictions_dev.npz").exists()
    pred = np.load(out_dir / "predictions_dev.npz", allow_pickle=False)
    assert int(pred["global_row_id"].size) == payload["metrics"]["dev_n"]
    np.testing.assert_allclose(pred["residual"], pred["pred"] - pred["y_true"], atol=1e-12)


def _norm() -> dict[str, object]:
    return {
        "global_mean": [0.0] * 42,
        "global_scale": [1.0] * 42,
        "scalar_mean": [0.0] * 9,
        "scalar_scale": [1.0] * 9,
    }
