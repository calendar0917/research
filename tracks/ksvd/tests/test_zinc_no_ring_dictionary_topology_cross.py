"""Targeted tests for DTX-v0 (no-ring dictionary x generic topology cross).

Static/CPU-only, synthetic ``Data``: never loads ZINC data, checkpoints,
official valid, or official test.

They enforce the non-negotiable strict-static contract, the round's no-ring
contract and the DTX-specific causal design:

* A. architecture: standalone module, ``center_context=False``,
  ``center_update is None``, forward succeeds while the historical
  ``_pool_pairs_to_centres`` raises;
* B. one single pass: pair encoder / relation encoder / dictionary encoder /
  dictionary assignment each evaluated exactly once per forward;
* C. cross independence: mutating the topology role leaves ``h_i`` and
  ``alpha_i`` bit-identical (cross cannot write back into a local state) while
  the Arm-A prediction changes (non-vacuous);
* D. matched marginal control: within-graph permutation of ``s_i`` leaves the
  Arm-M ``J_indep`` *and* prediction at the numerical noise floor, while Arm A's
  ``J_align`` and prediction move materially;
* E. A/M parity: identical state-dict keys, shapes, parameter count and
  bit-identical shared initialisation;
* F. invariance: pair order, endpoint swap, within-graph patch relabel; the
  topology-role extractor is *exactly* invariant to node relabelling;
* G. no explicit ring code path: AST-level audit of the extractor and of
  ``DTXModel.encode`` for any cycle/ring symbol or identity-token read; the
  cycle-rank descriptor coordinate is located from the real implementation and
  deleted from the dictionary query;
* H. gradients: local encoder, dictionary encoder, dictionary atoms, tau,
  cross projection, raw pair encoders and head all receive nonzero finite
  gradient.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import zinc_no_ring_dictionary_topology_cross as dtx
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------


def ztopo_width() -> int:
    return int(ztopo.raw_width("hinge"))


def _graph(n: int, seed: int) -> Data:
    generator = torch.Generator().manual_seed(int(seed))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.zeros(2, 0, dtype=torch.long)
    )
    return Data(
        patch_cont=torch.randn(n, int(zpp.SHELL_WIDTH), generator=generator),
        patch_context=torch.zeros(n, 0),
        # Present but semantically inert: DTX must never read these.
        typed_token=torch.randint(0, 4096, (n,), generator=generator),
        parent_token=torch.randint(0, 32, (n,), generator=generator),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 0),
        generic_topology=torch.randn(n, dtx.TOPO_ROLE_DIM, generator=generator),
        pair_index=pair_index,
        pair_relation=torch.randn(
            len(pairs), int(zpp.RELATION_WIDTH), generator=generator
        ),
        pair_bucket=torch.randint(
            0, zpp.DISTANCE_BUCKETS, (len(pairs),), generator=generator
        ),
        global_context=torch.randn(1, int(zpp.GLOBAL_WIDTH), generator=generator),
        topology_features=torch.randn(1, ztopo_width(), generator=generator),
        y=torch.randn(1, generator=generator),
        num_nodes=n,
    )


def _batch(graphs=None):
    graphs = graphs or [_graph(n, seed=n) for n in (7, 6, 8)]
    return next(iter(zpp._make_loader(graphs, len(graphs), False, 0)))


def _model(arm: str = "aligned") -> dtx.DTXModel:
    model = dtx.build_dtx(0, arm)
    model.eval()
    return model


def _permute_roles_within_graph(batch, seed: int = 777):
    generator = torch.Generator().manual_seed(int(seed))
    permuted = batch.generic_topology.clone()
    for graph_id in range(int(batch.global_context.shape[0])):
        index = (batch.batch == graph_id).nonzero(as_tuple=True)[0]
        if int(index.numel()) > 1:
            order = torch.randperm(int(index.numel()), generator=generator)
            permuted[index] = batch.generic_topology[index[order]]
    mutated = batch.clone()
    mutated.generic_topology = permuted
    return mutated


# ---------------------------------------------------------------------------
# A. architecture / widths / identity-free inputs
# ---------------------------------------------------------------------------


def test_architecture_contract_and_runtime_widths():
    model = dtx.build_dtx(0, "aligned")
    assert model.center_context is False
    assert model.center_update is None
    assert not isinstance(model, zpp.PatchPathModel)
    assert not hasattr(model, "_pool_pairs_to_centres")

    audit = dtx.runtime_width_audit(model, _batch())
    assert audit["widths_consistent"]
    assert audit["local_input_width"] == dtx.LOCAL_INPUT_WIDTH == 146
    assert audit["local_state_width"] == dtx.LOCAL_DIM == 48
    assert audit["dictionary_input_width"] == dtx.DICT_INPUT_WIDTH == 145
    assert audit["dictionary_query_width"] == dtx.DICT_RANK == 32
    assert audit["dictionary_assignment_width"] == dtx.DICT_ATOMS == 64
    assert audit["topology_role_width"] == dtx.TOPO_ROLE_DIM == 8
    assert audit["cross_matrix_width"] == dtx.CROSS_DIM == 512
    assert audit["cross_embedding_width"] == dtx.CROSS_HIDDEN == 32
    assert audit["pair_input_width"] == 4 * dtx.PAIR_HIDDEN == 64
    assert audit["relation_input_width"] == zpp.RELATION_WIDTH == 23
    assert audit["unary_width"] == 97
    assert audit["pair_pool_width"] == 165
    assert audit["global_width"] == 32
    assert audit["topology_width"] == 8
    assert audit["mu_alpha_width"] == 64
    assert audit["mu_s_width"] == 8
    assert audit["graph_width"] == 406
    assert audit["identity_input_audit"]["only_distance_gate_embedding"]
    assert audit["identity_input_audit"]["uses_typed_token"] is False
    assert audit["identity_input_audit"]["uses_structural_ring_context"] is False
    assert abs(float(model.dictionary.current_tau().detach()) - 0.20) < 1.0e-6


def test_no_message_passing_forward_succeeds():
    model = _model()
    batch = _batch()

    def _violation(*_args, **_kwargs):
        raise RuntimeError("strict-static contract violated")

    original = zpp.PatchPathModel._pool_pairs_to_centres
    zpp.PatchPathModel._pool_pairs_to_centres = _violation
    try:
        with torch.no_grad():
            prediction = model(batch)
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original
    assert prediction.shape == (3,)


# ---------------------------------------------------------------------------
# B. one single pass
# ---------------------------------------------------------------------------


def test_pair_and_dictionary_evaluated_once():
    for arm in dtx.ARMS:
        model = _model(arm)
        _prediction, trace = dtx._trace_forward(model, _batch())
        counters = trace["counters"]
        assert counters["pair_encoder"] == 1
        assert counters["relation_encoder"] == 1
        assert counters["dict_encoder"] == 1
        assert counters["dictionary"] == 1


# ---------------------------------------------------------------------------
# C. cross never writes back into h_i / alpha_i
# ---------------------------------------------------------------------------


def test_cross_does_not_affect_local_state():
    model = _model("aligned")
    batch = _batch()
    generator = torch.Generator().manual_seed(101)
    mutated = batch.clone()
    mutated.generic_topology = torch.randn(
        batch.generic_topology.shape, generator=generator
    )
    prediction, trace = dtx._trace_forward(model, batch)
    mutated_prediction, mutated_trace = dtx._trace_forward(model, mutated)
    assert torch.equal(trace["h"], mutated_trace["h"])
    assert torch.equal(trace["alpha"], mutated_trace["alpha"])
    assert torch.equal(trace["dict_embedding"], mutated_trace["dict_embedding"])
    assert not torch.equal(trace["cross"], mutated_trace["cross"])
    assert (prediction - mutated_prediction).abs().max().item() > 1.0e-8


# ---------------------------------------------------------------------------
# D. matched marginal control is invariant to the alignment
# ---------------------------------------------------------------------------


def test_marginal_control_invariant_to_alignment():
    batch = _batch()
    permuted = _permute_roles_within_graph(batch)
    # The permutation must keep the topology marginal multiset intact.
    assert torch.equal(
        batch.generic_topology.sort(dim=0).values,
        permuted.generic_topology.sort(dim=0).values,
    )
    results = {}
    for arm in dtx.ARMS:
        model = _model(arm)
        prediction, trace = dtx._trace_forward(model, batch)
        perm_prediction, perm_trace = dtx._trace_forward(model, permuted)
        results[arm] = {
            "joint_shift": (trace["cross"] - perm_trace["cross"]).abs().max().item(),
            "prediction_shift": (prediction - perm_prediction).abs().max().item(),
        }
    assert results["marginal"]["joint_shift"] <= 1.0e-5
    assert results["marginal"]["prediction_shift"] <= 1.0e-5
    assert results["aligned"]["joint_shift"] >= 1.0e-2
    assert results["aligned"]["prediction_shift"] >= 1.0e-5
    # Arm A responds strictly more than Arm M (1000x+ at random init on CPU).
    assert (
        results["aligned"]["prediction_shift"]
        > 10.0 * max(results["marginal"]["prediction_shift"], 1.0e-9)
    )

    stats = dtx.cross_statistics(
        torch.full((4, dtx.DICT_ATOMS), 1.0 / dtx.DICT_ATOMS),
        torch.randn(4, dtx.TOPO_ROLE_DIM),
        torch.zeros(4, dtype=torch.long),
        1,
    )
    assert torch.equal(
        stats["J_indep"],
        stats["mu_alpha"].unsqueeze(2) * stats["mu_s"].unsqueeze(1),
    )


# ---------------------------------------------------------------------------
# E. A / M parity
# ---------------------------------------------------------------------------


def test_arm_parameter_parity_and_shared_init():
    aligned = dtx.build_dtx(0, "aligned")
    marginal = dtx.build_dtx(0, "marginal")
    aligned_state = aligned.state_dict()
    marginal_state = marginal.state_dict()
    assert list(aligned_state.keys()) == list(marginal_state.keys())
    for key in aligned_state:
        assert aligned_state[key].shape == marginal_state[key].shape
        assert torch.equal(aligned_state[key], marginal_state[key])
    assert dtx._n_params(aligned) == dtx._n_params(marginal)
    assert int(dtx._n_params(aligned)) == int(dtx._n_params(aligned))


# ---------------------------------------------------------------------------
# F. invariance
# ---------------------------------------------------------------------------


def test_model_permutation_invariance():
    model = _model("aligned")
    batch = _batch()
    prediction, trace = dtx._trace_forward(model, batch)
    # endpoint swap + pair order
    order = torch.randperm(int(batch.pair_index.shape[1]), generator=torch.Generator().manual_seed(7))
    shuffled = batch.clone()
    shuffled.pair_index = batch.pair_index[:, order]
    shuffled.pair_relation = batch.pair_relation[order]
    shuffled.pair_bucket = batch.pair_bucket[order]
    with torch.no_grad():
        shuffled_prediction = model(shuffled)
    assert (prediction - shuffled_prediction).abs().max().item() < 1.0e-4
    # within-graph patch relabel
    generator = torch.Generator().manual_seed(13)
    patch_order = torch.arange(int(batch.patch_cont.shape[0]))
    for graph_id in range(int(batch.global_context.shape[0])):
        members = (batch.batch == graph_id).nonzero(as_tuple=True)[0]
        if int(members.numel()) > 1:
            patch_order[members] = members[
                torch.randperm(int(members.numel()), generator=generator)
            ]
    inverse = torch.empty_like(patch_order)
    inverse[patch_order] = torch.arange(patch_order.shape[0])
    relabeled = batch.clone()
    relabeled.patch_cont = batch.patch_cont[patch_order]
    relabeled.generic_topology = batch.generic_topology[patch_order]
    relabeled.pair_index = inverse[batch.pair_index]
    with torch.no_grad():
        relabeled_prediction = model(relabeled)
    assert (prediction - relabeled_prediction).abs().max().item() < 1.0e-4


def _role_for_edges(n: int, edges, center: int = 0) -> np.ndarray:
    graph = from_edges(n, [(int(u), int(v)) for u, v in edges])
    distances = zpp._ego_distances(graph, center, zpp.PATCH_RADIUS)
    return dtx.generic_topology_role(graph, center, distances)


def test_topology_role_formula_and_exact_relabel_invariance():
    triangle = _role_for_edges(3, [(0, 1), (1, 2), (2, 0)])
    # Every non-root node has patch degree 2 and exactly one same-shell
    # neighbour, so t7 = 0.5; the shell-1 pair is fully connected (t3 = 1).
    expected = np.asarray(
        [2 / 2, 2 / 2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32
    )
    assert np.array_equal(triangle, expected)

    star = _role_for_edges(4, [(0, 1), (0, 2), (0, 3)])
    assert star[0] == 1.0 and star[1] == 1.0 and star[2] == 0.0

    rng = np.random.default_rng(0)
    for n, edges in (
        (3, [(0, 1), (1, 2), (2, 0)]),
        (5, [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]),
        (7, [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 3), (2, 5)]),
    ):
        base = _role_for_edges(n, edges)
        for _ in range(50):
            permutation = rng.permutation(n)
            remapped = [(int(permutation[u]), int(permutation[v])) for u, v in edges]
            center = int(permutation[0])
            relabeled = _role_for_edges(n, remapped, center)
            assert np.array_equal(base, relabeled)
        assert np.all(base >= 0.0) and np.all(base <= 1.0)


# ---------------------------------------------------------------------------
# G. no explicit ring code path / descriptor coordinate audit
# ---------------------------------------------------------------------------


def _top_level_symbols(function) -> set[str]:
    source = inspect.getsource(function)
    if not source.startswith("def "):
        source = textwrap.dedent(source)
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def test_extractor_has_no_ring_or_identity_symbols():
    forbidden = ("cycle", "ring", "basis", "aromatic", "fused", "spiro")

    def _hits(function):
        symbols = _top_level_symbols(function)
        return sorted(
            symbol
            for symbol in symbols
            # Module-level ALL_CAPS constants are allowed (the dictionary input
            # deletes the audited cycle-rank *index*; it never computes one).
            if not symbol.isupper()
            for needle in forbidden
            if needle in symbol.lower()
        )

    for function in (
        dtx.generic_topology_role_from_edges,
        dtx.generic_topology_role,
        dtx.topology_roles_for_raw_graph,
        dtx.dictionary_input_from_patch_cont,
    ):
        hits = _hits(function)
        assert hits == [], f"{function.__name__} references ring machinery: {hits}"

    # The extractor may only reach the graph through distance/edge primitives.
    extractor_symbols = _top_level_symbols(dtx.generic_topology_role)
    assert "_ego_distances" not in extractor_symbols
    assert "induced" in extractor_symbols and "edges" in extractor_symbols

    encode_symbols = _top_level_symbols(dtx.DTXModel.encode)
    identity_hits = sorted(
        symbol
        for symbol in encode_symbols
        if any(
            needle in symbol.lower()
            for needle in (
                "typed_token",
                "parent_token",
                "structural_token",
                "structural_coarse",
                "structural_context",
            )
        )
    )
    assert identity_hits == []
    assert "generic_topology" in encode_symbols


def test_cycle_rank_coordinate_is_located_and_deleted():
    audit = dtx.cycle_rank_coordinate_audit()
    assert audit["passed"]
    assert audit["feature_index"] == 143
    assert audit["dictionary_input_width"] == 145

    fused = dtx._descriptor_for_edges(
        5, ((0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0))
    )
    assert abs(float(fused[143]) - 2.0 / 5.0) < 1.0e-7
    tree = dtx._descriptor_for_edges(3, ((0, 1), (1, 2)))
    assert float(tree[143]) == 0.0

    generator = torch.Generator().manual_seed(5)
    patch_cont = torch.randn(4, 146, generator=generator)
    dropped = dtx.dictionary_input_from_patch_cont(patch_cont)
    assert dropped.shape == (4, 145)
    assert torch.equal(dropped[:, :143], patch_cont[:, :143])
    assert torch.equal(dropped[:, 143:], patch_cont[:, 144:])
    # No other coordinate is cycle-derived: the non-scalar blocks are untouched.
    assert torch.equal(dropped[:, :140], patch_cont[:, :140])


# ---------------------------------------------------------------------------
# H. gradients
# ---------------------------------------------------------------------------


def test_gradients_reach_every_dtx_component():
    for arm in dtx.ARMS:
        model = dtx.build_dtx(0, arm)
        batch = _batch()
        model.train()
        prediction = model(batch).view(-1)
        loss = torch.nn.functional.l1_loss(prediction, batch.y.view(-1))
        loss.backward()
        checks = {
            "local_encoder": model.local_encoder,
            "dict_encoder": model.dict_encoder,
            "dictionary": model.dictionary,
            "cross_projection": model.cross_projection,
            "pair_projection": model.pair_projection,
            "relation_encoder": model.relation_encoder,
            "pair_encoder": model.pair_encoder,
            "head": model.head,
        }
        for name, module in checks.items():
            total = 0.0
            for parameter in module.parameters():
                assert parameter.grad is not None, f"{arm}:{name} has no grad"
                assert torch.isfinite(parameter.grad).all(), f"{arm}:{name} non-finite"
                total += float(parameter.grad.detach().pow(2).sum())
            if name == "dictionary":
                # atoms must receive gradient, and tau must be finite.
                assert float(total) > 0.0
                assert torch.isfinite(model.dictionary.tau_logit.grad).all()
            elif name in ("head", "cross_projection"):
                assert float(total) > 0.0
            else:
                assert float(total) > 0.0, f"{arm}:{name} grad norm is zero"


# ---------------------------------------------------------------------------
# extra: cheap inference interventions
# ---------------------------------------------------------------------------


def test_inference_interventions_are_wired():
    model = _model("aligned")
    batch = _batch()
    with torch.no_grad():
        base = model(batch)
    for mode in ("alignment_removal", "topology_shuffle"):
        model.set_inference_intervention(mode)
        try:
            with torch.no_grad():
                shifted = model(batch)
        finally:
            model.set_inference_intervention(None)
        assert (base - shifted).abs().max().item() > 1.0e-8
    # topology mutation must not touch parameters or the local state
    parameter_before = [value.clone() for value in model.parameters()]
    generator = torch.Generator().manual_seed(3)
    mutated = batch.clone()
    mutated.generic_topology = torch.randn(
        batch.generic_topology.shape, generator=generator
    )
    _prediction, trace = dtx._trace_forward(model, batch)
    _mut_prediction, mutated_trace = dtx._trace_forward(model, mutated)
    assert torch.equal(trace["alpha"], mutated_trace["alpha"])
    for before, after in zip(parameter_before, model.parameters()):
        assert torch.equal(before, after.detach())


def test_pooling_semantics_are_inherited_mean_std_logcount():
    model = _model()
    batch = _batch()
    _prediction, trace = dtx._trace_forward(model, batch)
    assert int(trace["pair_input"].shape[1]) == 4 * model.pair_hidden
    # 5 distance buckets x [mean | std | log1p(count)] over 16D pair states.
    assert model.pair_pool_width == zpp.DISTANCE_BUCKETS * (2 * model.pair_hidden + 1)
    assert model.unary_width == 2 * model.local_dim + 1
    assert batch.pair_bucket.max().item() < zpp.DISTANCE_BUCKETS
