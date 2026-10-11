"""cscl-strong-nongnn-predictor-v1 correctness and liveness tests.

Fast tests (no data / no checkpoints) cover the triple construction, the
operator's exchange symmetry, ablations, pooling and static protocol guards.
Slow tests (marked ``slow``) load the ZINC caches / the frozen Full-Y soup and
check: initialisation identity with the historical Full-Y, gradient liveness
into the new branch, batch/label invariances, save/load equality and the
read-only soup replay tolerance.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import static_triplet_operator as stp  # noqa: E402


# ---------------------------------------------------------------------------
# fast: triplet construction
# ---------------------------------------------------------------------------


def _bonds(edges: list[tuple[int, int]], types: list[int] | None = None, span: int | None = None):
    lo = torch.tensor([min(a, b) for a, b in edges], dtype=torch.long)
    hi = torch.tensor([max(a, b) for a, b in edges], dtype=torch.long)
    tp = torch.tensor(types if types is not None else [0] * len(edges), dtype=torch.long)
    span = int(span) if span is not None else int(max(max(a, b) for a, b in edges) + 1)
    key = lo * span + hi
    order = torch.argsort(key)
    return {"lo": lo[order], "hi": hi[order], "type": tp[order], "key": key[order]}, span


def test_wedge_enumeration_unique_complete_and_closed():
    # star 0-(1,2,3,4): C(4,2)=6 wedges centred at 0
    star, span = _bonds([(0, 1), (0, 2), (0, 3), (0, 4)])
    w = stp.build_wedges(star, num_nodes=span)
    assert w["center"].numel() == 6
    assert set(w["center"].tolist()) == {0}
    pairs = {frozenset((int(i), int(k))) for i, k in zip(w["end_i"], w["end_k"])}
    assert len(pairs) == 6
    # path 0-1-2-3: only centre 1 and 2 have degree 2 -> 2 wedges
    path, span = _bonds([(0, 1), (1, 2), (2, 3)])
    w2 = stp.build_wedges(path, num_nodes=span)
    assert w2["center"].numel() == 2
    assert sorted(w2["center"].tolist()) == [1, 2]
    # triangle 0-1-2: one wedge per centre, closure = 1
    tri, span = _bonds([(0, 1), (1, 2), (0, 2)])
    w3 = stp.build_wedges(tri, num_nodes=span)
    assert w3["center"].numel() == 3
    assert float(w3["closure"].sum()) == 3.0
    # chain 0-1-2 closure 0
    w4 = stp.build_wedges(path, num_nodes=4)
    assert float(w4["closure"].sum()) == 0.0


def test_wedge_relabel_invariance_and_exchange_symmetry():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    bonds, span = _bonds(edges)
    w = stp.build_wedges(bonds, num_nodes=span)
    # relabel permutation
    perm = torch.tensor([2, 0, 3, 1])
    relabelled, span2 = _bonds([(int(perm[a]), int(perm[b])) for a, b in edges])
    w2 = stp.build_wedges(relabelled, num_nodes=span2)
    def canon(ww):
        return sorted(
            (int(c), tuple(sorted((int(i), int(k)))))
            for c, i, k in zip(ww["center"], ww["end_i"], ww["end_k"])
        )

    inverse = torch.argsort(perm)
    w2_back = {
        "center": inverse[w2["center"]],
        "end_i": inverse[w2["end_i"]],
        "end_k": inverse[w2["end_k"]],
    }
    assert canon(w) == canon(w2_back)
    # closure flags must also be permutation invariant
    assert sorted(w["closure"].tolist()) == sorted(w2["closure"].tolist())
    # endpoint swap: same multiset of (centre, unordered pair)
    swap = sorted(
        (int(c), tuple(sorted((int(i), int(k)))))
        for c, i, k in zip(w["center"], w["end_k"], w["end_i"])
    )
    assert canon(w) == swap
    # no self-loops / duplicate unordered pairs
    pairs = [(int(c), tuple(sorted((int(i), int(k))))) for c, i, k in zip(w["center"], w["end_i"], w["end_k"])]
    assert len(pairs) == len(set(pairs))
    assert all(i != k and i != c and k != c for c, (i, k) in pairs)


def test_wedges_empty_graphs_return_empty():
    empty = {
        "lo": torch.zeros(0, dtype=torch.long),
        "hi": torch.zeros(0, dtype=torch.long),
        "type": torch.zeros(0, dtype=torch.long),
        "key": torch.zeros(0, dtype=torch.long),
    }
    w = stp.build_wedges(empty, num_nodes=1)
    assert w["center"].numel() == 0 and w["closure"].numel() == 0
    # single bond: no wedge
    one, span = _bonds([(0, 1)])
    w1 = stp.build_wedges(one, num_nodes=span)
    assert w1["center"].numel() == 0


def test_unique_bonds_from_env_dedups_directions_and_roots():
    u = torch.tensor([3, 1, 0, 3])
    v = torch.tensor([1, 0, 1, 1])
    t = torch.tensor([2, 1, 1, 2])
    bonds = stp.unique_bonds_from_env(u, v, t, num_nodes=4)
    assert bonds["lo"].tolist() == [0, 1]
    assert bonds["hi"].tolist() == [1, 3]
    assert bonds["type"].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# fast: operator symmetry / ablations / pooling
# ---------------------------------------------------------------------------


def _random_triplet_inputs(n: int = 5, env_dim: int = 6, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    E = torch.randn(n, env_dim, generator=gen)
    center = torch.tensor([0, 1, 2, 0, 2])
    end_i = torch.tensor([1, 0, 0, 2, 1])
    end_k = torch.tensor([2, 3, 3, 3, 3])
    bond_1 = torch.tensor([0, 1, 2, 3, 0])
    bond_2 = torch.tensor([1, 2, 3, 0, 1])
    atom = torch.tensor([0, 5, 27, 13, 2])
    closure = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    return E, center, end_i, end_k, bond_1, bond_2, atom, closure


def test_operator_endpoint_exchange_symmetry():
    op = stp.StaticTripletOperator(env_dim=6, seed=0)
    E, center, end_i, end_k, b1, b2, atom, closure = _random_triplet_inputs()
    q = op(E, center, end_i, end_k, b1, b2, atom, closure)
    q_swap = op(E, center, end_k, end_i, b2, b1, atom, closure)
    assert torch.allclose(q, q_swap, atol=1e-6)
    assert q.shape == (5, stp.TRIPLET_OUT)


def test_operator_ablations_change_features_but_stay_finite():
    op = stp.StaticTripletOperator(env_dim=6, seed=0)
    E, center, end_i, end_k, b1, b2, atom, closure = _random_triplet_inputs()
    base = op(E, center, end_i, end_k, b1, b2, atom, closure)
    for ablation in ("zero_endpoint_env", "zero_center_env", "zero_atom_type", "zero_closure", "zero_bond_types", "cyclic_bond_types"):
        alt = op(E, center, end_i, end_k, b1, b2, atom, closure, ablation=ablation)
        assert torch.isfinite(alt).all()
        assert not torch.allclose(base, alt), ablation
    with pytest.raises(ValueError):
        op(E, center, end_i, end_k, b1, b2, atom, closure, ablation="nope")


def test_pool_triplet_moments_empty_is_zero():
    q = torch.zeros((0, stp.TRIPLET_OUT))
    graph = torch.zeros(0, dtype=torch.long)
    out = stp.pool_triplet_moments(q, graph, n_graphs=3)
    assert out.shape == (3, stp.TRIPLET_SUMMARY_DIM)
    assert torch.count_nonzero(out) == 0


def test_pool_triplet_moments_values():
    q = torch.tensor([[1.0, -2.0], [3.0, 4.0], [0.5, 0.5]])
    graph = torch.tensor([0, 0, 1])
    out = stp.pool_triplet_moments(q, graph, n_graphs=2)
    assert torch.allclose(out[0, :2], torch.tensor([4.0, 2.0]))
    assert torch.allclose(out[0, 2:], torch.tensor([10.0, 20.0]))
    assert torch.allclose(out[1, :2], torch.tensor([0.5, 0.5]))
    assert torch.allclose(out[1, 2:], torch.tensor([0.25, 0.25]))


def test_triplet_feature_dim_accounting():
    op = stp.StaticTripletOperator(env_dim=7, seed=0)
    assert op.in_dim == stp.triplet_feature_dim()
    assert op.in_dim == 3 * 32 + 32 + 4 + 16 + 28 + 1


# ---------------------------------------------------------------------------
# fast: protocol and static guards
# ---------------------------------------------------------------------------


def test_protocol_yaml_frozen():
    text = (REPO_ROOT / "tracks/structure-contribution/configs/strong_nongnn_predictor_v1.yaml").read_text()
    for key in (
        "cscl-strong-nongnn-predictor-v1",
        "FullY-StaticTriplet",
        "case_A",
        "0.095",
        "0.106",
        "240",
        "top-5",
        "zero-initialised",
        "centre",
        "real bond",
        "soup",
    ):
        assert key in text, key


def test_driver_seed_zero_only():
    text = (REPO_ROOT / "tracks/structure-contribution/code/run_strong_predictor_v1.py").read_text()
    assert "choices=[0]" in text


def test_no_official_or_decomp_access_in_new_code():
    for name in ("run_strong_predictor_v1.py", "static_triplet_operator.py"):
        text = (_CODE_DIR / name).read_text()
        for forbidden in (
            "encoded_valid",
            "load_test_data",
            'split="test"',
            'split="valid"',
            "target_decomposition",
            "official_test_loader",
        ):
            assert forbidden not in text, (name, forbidden)


def test_operator_has_no_gnn_or_attention_modules():
    op = stp.StaticTripletOperator(env_dim=6, seed=0)
    names = {type(m).__name__ for m in op.modules()}
    banned = {
        "MessagePassing", "GCNConv", "GINConv", "GINEConv", "GraphConv",
        "TransformerConv", "GATConv", "MultiheadAttention", "Transformer",
    }
    assert not (names & banned)
    assert {type(m).__name__ for m in op.modules()} == {"StaticTripletOperator", "Linear", "Sequential", "SiLU"}


def test_embedding_or_parameter_shapes_forbidden():
    """The operator may not contain per-molecule tables or graph state."""
    op = stp.StaticTripletOperator(env_dim=6, seed=0)
    for name, param in op.named_parameters():
        assert param.dim() in (1, 2), (name, param.dim())
    assert "atom_embedding" not in {name for name, _ in op.named_parameters()}


# ---------------------------------------------------------------------------
# slow: data-backed identity and liveness
# ---------------------------------------------------------------------------

slow = pytest.mark.slow


@pytest.fixture(scope="module")
def smoke_env():
    import cscl_features as cf
    import run_prediction_feasibility_v2 as drv2

    quiet = lambda *a, **k: None  # noqa: E731
    tmp = tempfile.mkdtemp(prefix="strong-v1-test-")
    train = drv2.load_train_rows(log=quiet)
    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
    drv2.verify_row_alignment(train, log=quiet)
    path, meta = drv2.build_prep_fit7200(train, idx, Path(tmp) / "prep", smoke=True, log=quiet)
    prep = {k: z for k, z in np.load(path, allow_pickle=False).items()}
    blob10 = {k: z for k, z in np.load(drv2.LEGACY_PREP, allow_pickle=False).items()}
    drv2.apply_prep_fit7200(train, prep, blob10)
    rows = [int(i) for i in idx["dev"][:16]]
    data = [train[i] for i in rows]
    return drv2, train, idx, prep, data, blob10


@slow
def test_bonds_equal_pair_bucket_on_sample(smoke_env):
    _drv2, train, idx, _prep, _data, _blob = smoke_env
    rows = [int(i) for i in idx["fit_inner"][:24]] + [int(i) for i in idx["dev"][:24]]
    for row in rows:
        d = train[row]
        env_bonds = stp.unique_bonds_from_env(
            d.env_bond_u, d.env_bond_v, d.env_bond_type, int(d.dict_phi.shape[0])
        )
        env_pairs = set(zip(env_bonds["lo"].tolist(), env_bonds["hi"].tolist()))
        mask = d.pair_bucket == 0
        pair_pairs = set(zip(d.pair_index[0][mask].tolist(), d.pair_index[1][mask].tolist()))
        assert env_pairs == pair_pairs, row


@slow
def test_initial_function_equals_full_y(smoke_env):
    drv2, _train, _idx, prep, data, _blob = smoke_env
    ref = drv2.build_model(prep, seed=0)
    new = stp.build_triplet_model(np.asarray(prep["D_fit"], np.float32), 0, drv2.zjd.fold_subspace(prep))
    shared = new.shared_with_full_y(ref.state_dict())
    assert shared["identical_shared_parameters"], shared
    ref_pred = drv2.predict_raw(ref, data, torch.device("cpu"))
    new_pred = drv2.predict_raw(new, data, torch.device("cpu"))
    assert float(np.abs(ref_pred - new_pred).max()) <= 5e-5
    inventory = stp.triplet_parameter_inventory(new)
    assert inventory["triplet_total"] == inventory["triplet_env_proj"] + inventory["triplet_mlp"]
    assert inventory["full_model_parameters"] == inventory["original_full_y_parameters"] + inventory["triplet_total"] + inventory["reader_new_columns"]


@slow
def test_gradient_liveness_and_sensitivity(smoke_env):
    drv2, _train, _idx, prep, data, _blob = smoke_env
    model = stp.build_triplet_model(np.asarray(prep["D_fit"], np.float32), 0, drv2.zjd.fold_subspace(prep))
    model.train()
    y = torch.tensor([float(d.y.reshape(-1)[0]) for d in data], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = drv2._batch(data, list(range(len(data))), y, torch.device("cpu"))
    pred, aux = model(batch, mask=drv2.cm.C6_MASK, return_aux=True)
    loss = torch.nn.functional.l1_loss(pred.view(-1), batch.y.view(-1)) + drv2.H1_LAMBDA * model.reconstruction_loss(
        aux["phi"], aux["coord"]
    )
    opt.zero_grad()
    loss.backward()
    grad_new = model.reader.net[0].weight.grad[:, model.reader_input_original :]
    assert float(grad_new.abs().sum()) > 0.0
    opt.step()
    assert float(model.reader.net[0].weight[:, model.reader_input_original :].abs().sum()) > 0.0
    # second step reaches the triplet module
    pred2, aux2 = model(batch, mask=drv2.cm.C6_MASK, return_aux=True)
    loss2 = torch.nn.functional.l1_loss(pred2.view(-1), batch.y.view(-1)) + drv2.H1_LAMBDA * model.reconstruction_loss(
        aux2["phi"], aux2["coord"]
    )
    opt.zero_grad()
    loss2.backward()
    assert float(model.triplet_op.env_proj.weight.grad.abs().sum()) > 0.0
    assert float(model.triplet_op.mlp[0].weight.grad.abs().sum()) > 0.0
    opt.step()
    # prediction must now depend on the triplet summary
    model.eval()
    with torch.no_grad():
        base = drv2.predict_raw(model, data, torch.device("cpu"))
        model.triplet_ablation = "zero_summary"
        ablated = drv2.predict_raw(model, data, torch.device("cpu"))
        model.triplet_ablation = "none"
    assert float(np.abs(base - ablated).max()) > 1e-6


@slow
def test_label_independence_and_y_change(smoke_env):
    drv2, _train, _idx, prep, data, _blob = smoke_env
    model = stp.build_triplet_model(np.asarray(prep["D_fit"], np.float32), 0, drv2.zjd.fold_subspace(prep))
    with torch.no_grad():
        base = drv2.predict_raw(model, data, torch.device("cpu"))
    saved = [d.y.clone() for d in data]
    for d in data:
        d.y = torch.tensor([1e4])
    with torch.no_grad():
        changed = drv2.predict_raw(model, data, torch.device("cpu"))
    for d, s in zip(data, saved):
        d.y = s
    assert np.array_equal(base, changed)


@slow
def test_save_load_equality(smoke_env):
    drv2, _train, _idx, prep, data, _blob = smoke_env
    model = stp.build_triplet_model(np.asarray(prep["D_fit"], np.float32), 0, drv2.zjd.fold_subspace(prep))
    with torch.no_grad():
        base = drv2.predict_raw(model, data, torch.device("cpu"))
    state = {k: v.clone() for k, v in model.state_dict().items()}
    f = Path(tempfile.mkdtemp()) / "s.pt"
    torch.save(state, f)
    other = stp.build_triplet_model(np.asarray(prep["D_fit"], np.float32), 5, drv2.zjd.fold_subspace(prep))
    other.load_state_dict(torch.load(f, weights_only=True))
    with torch.no_grad():
        loaded = drv2.predict_raw(other, data, torch.device("cpu"))
    assert np.array_equal(base, loaded)


@slow
def test_read_only_soup_replay_matches_history(smoke_env):
    drv2, _train_shared, idx, _prep_smoke, _data, _blob = smoke_env
    prep_path = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/prep_fit7200.npz"
    soup_path = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_soup_state.pt"
    if not prep_path.exists() or not soup_path.exists():
        pytest.skip("historical Full-Y artifacts not present locally")
    quiet = lambda *a, **k: None  # noqa: E731
    train = drv2.load_train_rows(log=quiet)  # fresh (unstandardized) rows
    prep = {k: z for k, z in np.load(prep_path, allow_pickle=False).items()}
    blob10 = {k: z for k, z in np.load(drv2.LEGACY_PREP, allow_pickle=False).items()}
    drv2.apply_prep_fit7200(train, prep, blob10)
    rows = [int(i) for i in idx["dev"][:16]]
    data = [train[i] for i in rows]
    state = torch.load(soup_path, map_location="cpu", weights_only=False)
    ref = drv2.build_model(prep, seed=0)
    ref.load_state_dict(state)
    new = stp.build_triplet_model(np.asarray(prep["D_fit"], np.float32), 0, drv2.zjd.fold_subspace(prep))
    stp.load_full_y_state_into_triplet(new, state)
    with torch.no_grad():
        ref_pred = drv2.predict_raw(ref, data, torch.device("cpu"))
        new_pred = drv2.predict_raw(new, data, torch.device("cpu"))
    assert float(np.abs(ref_pred - new_pred).max()) <= 5e-5


@slow
def test_static_triplet_module_is_the_only_new_component(smoke_env):
    drv2, _train, _idx, prep, _data, _blob = smoke_env
    ref = drv2.build_model(prep, seed=0)
    new = stp.build_triplet_model(np.asarray(prep["D_fit"], np.float32), 0, drv2.zjd.fold_subspace(prep))
    ref_keys = set(ref.state_dict())
    new_keys = set(new.state_dict())
    triplet_keys = sorted(k for k in new_keys if k.startswith("triplet_op."))
    added = sorted(new_keys - ref_keys)
    assert added == triplet_keys, (added, triplet_keys)
    assert len(triplet_keys) == 6, triplet_keys
    # reader widened: same key names, new parameter width
    assert new.reader.net[0].weight.shape[1] == ref.reader.net[0].weight.shape[1] + stp.TRIPLET_SUMMARY_DIM
    assert new.reader_input_original == ref.reader.net[0].weight.shape[1]
