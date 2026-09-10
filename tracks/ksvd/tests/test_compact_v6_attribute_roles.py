"""Correctness tests for Compact-v6: topology-attribute factorization.

Covers the hard gates from the task brief:

1. permutation of node IDs -> identical role-pooled attribute primitives
   (>= 100 permutations per graph, several topology classes);
2. same topology + same attributes under permutation -> identical;
3. same topology + different (structurally non-equivalent) placement ->
   factorized primitives differ;
4. same attribute multiset but different role placement -> factorized
   primitives differ (the adversarial example);
5. count-control primitives are invariant to attribute placement;
6. baseline mode reproduces the historical compact-v4-hinge model exactly;
7. the v6 path performs no exact typed-token (corrected) embedding lookup;
8. the role definition is versioned / fingerprint-safe.
"""

from __future__ import annotations

import numpy as np
import torch
import yaml

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import compact_v6_attribute_roles as v6
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module

SELECTION_TYPED_VOCAB = 6785
PARENT_VOCAB = 32
V4_HINGE_TOTAL = 99_613
V6_DELTA = 1_470


# --------------------------------------------------------------------------
# toy graphs + relabelling helpers
# --------------------------------------------------------------------------

def _branched() -> tuple:
    """Root 0 -- 1 -- {2, 3}; 0 -- 4 (branch + heteroatom + multiple bonds)."""
    graph = from_edges(5, [(0, 1), (1, 2), (1, 3), (0, 4)])
    node_types = np.asarray([0, 1, 2, 0, 1], dtype=np.int64)
    edge_types = {(0, 1): 0, (1, 2): 1, (1, 3): 2, (0, 4): 3}
    return graph, node_types, edge_types


def _cyclic() -> tuple:
    graph = from_edges(
        7,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)],
    )
    node_types = np.asarray([0, 1, 2, 3, 0, 1, 2], dtype=np.int64)
    edge_types = {
        (0, 1): 0, (1, 2): 1, (2, 3): 0, (3, 4): 2,
        (4, 5): 0, (0, 5): 1, (0, 3): 3,
    }
    return graph, node_types, edge_types


def _symmetric_star() -> tuple:
    graph = from_edges(5, [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)])
    node_types = np.asarray([0, 1, 1, 1, 2], dtype=np.int64)
    edge_types = {(0, 1): 0, (0, 2): 0, (0, 3): 0, (1, 4): 1, (2, 4): 1, (3, 4): 1}
    return graph, node_types, edge_types


def _collision_pair() -> tuple:
    """Two patches sharing the historical (uncolored) certificate but differing
    in atom placement along a path a-b-c."""
    graph = from_edges(3, [(0, 1), (1, 2)])
    p1 = np.asarray([0, 1, 0], dtype=np.int64)  # N at middle
    p2 = np.asarray([0, 0, 1], dtype=np.int64)  # N at leaf
    edge_types = {(0, 1): 0, (1, 2): 0}
    return graph, p1, p2, edge_types


def _relabel(graph, node_types, edge_types, permutation):
    nodes = graph.nodes
    mapping = {int(node): int(permutation[index]) for index, node in enumerate(nodes)}
    new_edges = [(mapping[int(u)], mapping[int(v)]) for u, v in graph.edges()]
    new_graph = from_edges(len(nodes), new_edges)
    new_node_types = np.zeros_like(node_types)
    for node in nodes:
        new_node_types[mapping[int(node)]] = node_types[int(node)]
    new_edge_types: dict[tuple[int, int], int] = {}
    for (u, v), value in edge_types.items():
        new_edge_types[new_graph.edge_key(mapping[int(u)], mapping[int(v)])] = int(value)
    return new_graph, new_node_types, new_edge_types, mapping


def _atom_rows(primitives: v6.PatchRolePrimitives, *, use_role: bool):
    rows = []
    for index in range(int(primitives.atom_types.shape[0])):
        role = (
            tuple(np.round(primitives.atom_roles[index], 6).tolist())
            if use_role
            else ()
        )
        rows.append((int(primitives.atom_types[index]), role))
    return sorted(rows)


def _bond_rows(primitives: v6.PatchRolePrimitives, *, use_role: bool):
    rows = []
    for index in range(int(primitives.bond_types.shape[0])):
        if use_role:
            left = tuple(np.round(primitives.bond_role_left[index], 6).tolist())
            right = tuple(np.round(primitives.bond_role_right[index], 6).tolist())
        else:
            left = right = ()
        rows.append((int(primitives.bond_types[index]), left, right))
    return sorted(rows)


def _pooled(graph, center, node_types, edge_types, radius=2, *, use_role=True):
    primitives = v6.patch_role_primitives(
        graph, int(center), node_types, edge_types, radius
    )
    return (
        _atom_rows(primitives, use_role=use_role),
        _bond_rows(primitives, use_role=use_role),
    )


# --------------------------------------------------------------------------
# 1 / 2 -- permutation invariance
# --------------------------------------------------------------------------

def test_role_pooled_primitives_permutation_invariant() -> None:
    rng = np.random.default_rng(20260912)
    builders = [_branched, _cyclic, _symmetric_star]
    for build in builders:
        graph, node_types, edge_types = build()
        n = len(graph.nodes)
        for center in range(n):
            reference = _pooled(graph, center, node_types, edge_types)
            for _ in range(120):
                permutation = rng.permutation(n).tolist()
                rg, rnt, ret, mapping = _relabel(
                    graph, node_types, edge_types, permutation
                )
                other = _pooled(rg, mapping[center], rnt, ret)
                assert other == reference, (build.__name__, center, permutation)


def test_same_topology_same_attributes_under_permutation() -> None:
    graph, node_types, edge_types = _cyclic()
    reference = _pooled(graph, 0, node_types, edge_types)
    for permutation in ([1, 0, 2, 3, 4, 5, 6], [3, 2, 1, 0, 6, 5, 4]):
        rg, rnt, ret, mapping = _relabel(graph, node_types, edge_types, permutation)
        assert _pooled(rg, mapping[0], rnt, ret) == reference


# --------------------------------------------------------------------------
# 3 / 4 / 5 -- placement sensitivity and count-control invariance
# --------------------------------------------------------------------------

def _adversarial_pair():
    """Same topology, same atom and bond type multisets, but the N atom sits at
    two structurally non-equivalent roles (leaf vs branch point)."""
    graph = from_edges(5, [(0, 1), (1, 2), (1, 3), (0, 4)])
    # P1: N at node 2 (distance 2, degree 1 leaf)
    p1 = np.asarray([0, 0, 1, 0, 0], dtype=np.int64)
    # P2: N at node 1 (distance 1, degree 3 branch)
    p2 = np.asarray([0, 1, 0, 0, 0], dtype=np.int64)
    edge_types = {(0, 1): 0, (1, 2): 0, (1, 3): 0, (0, 4): 0}
    return graph, p1, p2, edge_types


def test_placement_changes_factorized_outside_role() -> None:
    graph, p1, p2, edge_types = _adversarial_pair()
    # same topology -> same role descriptors; different type placement
    roles1 = v6.patch_role_primitives(graph, 1, p1, edge_types, 2)
    roles2 = v6.patch_role_primitives(graph, 1, p2, edge_types, 2)
    assert _pooled(graph, 1, p1, edge_types, use_role=True) != _pooled(
        graph, 1, p2, edge_types, use_role=True
    )
    assert roles1.n_orbits == roles2.n_orbits


def test_adversarial_count_same_factorized_differs() -> None:
    graph, p1, p2, edge_types = _adversarial_pair()
    count1 = _pooled(graph, 1, p1, edge_types, use_role=False)
    count2 = _pooled(graph, 1, p2, edge_types, use_role=False)
    assert count1 == count2, "count-control must ignore placement"
    factor1 = _pooled(graph, 1, p1, edge_types, use_role=True)
    factor2 = _pooled(graph, 1, p2, edge_types, use_role=True)
    assert factor1 != factor2, "factorized-role must express placement"


def test_historical_collision_case_permutation_invariant() -> None:
    graph, p1, p2, edge_types = _collision_pair()
    # The historical uncolored certificate is the same for both placements.
    for node_types in (p1, p2):
        reference = _pooled(graph, 1, node_types, edge_types)
        for permutation in ([0, 1, 2], [1, 0, 2], [2, 1, 0]):
            rg, rnt, ret, mapping = _relabel(
                graph, node_types, edge_types, permutation
            )
            assert _pooled(rg, mapping[1], rnt, ret) == reference


def test_count_control_invariant_to_attribute_assignment_permutation() -> None:
    graph, _, edge_types = _cyclic()
    base = np.asarray([0, 1, 2, 3, 0, 1, 2], dtype=np.int64)
    reference = _pooled(graph, 0, base, edge_types, use_role=False)
    # any reassignment of the same type multiset must not change the count view
    shuffled = np.asarray([2, 3, 0, 1, 0, 1, 2], dtype=np.int64)
    assert sorted(base.tolist()) == sorted(shuffled.tolist())
    assert _pooled(graph, 0, shuffled, edge_types, use_role=False) == reference


# --------------------------------------------------------------------------
# 6 / 7 -- baseline reproduction and no exact-token lookup
# --------------------------------------------------------------------------

def _v4_hinge_config() -> dict:
    path = (
        module.REPO_ROOT
        / "tracks/ksvd/configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model(attribute_mode: str, *, typed_vocab: int = SELECTION_TYPED_VOCAB):
    config = _v4_hinge_config()["model"]
    return module.PatchPathModel(
        typed_vocab,
        PARENT_VOCAB,
        patch_hidden=int(config["patch_hidden"]),
        pair_hidden=int(config["pair_hidden"]),
        token_width=int(config["token_width"]),
        dropout=float(config["dropout"]),
        embedding_mode=str(config["embedding_mode"]),
        embedding_rank=int(config["embedding_rank"]),
        hybrid_full_typed_tokens=int(config["hybrid_full_typed_tokens"]),
        hybrid_full_parent_tokens=int(config["hybrid_full_parent_tokens"]),
        center_context=bool(config["center_context"]),
        center_context_hidden=int(config["center_context_hidden"]),
        graph_head_hidden_0=int(config["graph_head_hidden_0"]),
        graph_head_hidden_1=int(config["graph_head_hidden_1"]),
        shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
        context_width=0,
        topology_mode="hinge",
        topology_input_width=25,
        topology_hidden_dim=int(config["topology_hidden_dim"]),
        topology_out_dim=int(config["topology_out_dim"]),
        attribute_mode=attribute_mode,
    )


def _attribute_data() -> module.Data:
    n = 6
    return module.Data(
        patch_cont=torch.randn(n, module._shell_width_for_radius(module.PATCH_RADIUS)),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.tensor([0, 1, 767, 768, 769, 100], dtype=torch.long),
        parent_token=torch.tensor([0, 1, 2, 3, 31, 31], dtype=torch.long),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 4),
        pair_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
        pair_relation=torch.randn(5, module.RELATION_WIDTH),
        pair_bucket=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        global_context=torch.randn(1, module.GLOBAL_WIDTH),
        batch=torch.zeros(n, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=n,
        topology_features=torch.zeros(1, 25),
        attribute_atom_type=torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.long),
        attribute_atom_role=torch.rand(7, v6.ROLE_DIM),
        attribute_atom_patch_index=torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.long),
        attribute_bond_type=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        attribute_bond_role_left=torch.rand(4, v6.ROLE_DIM),
        attribute_bond_role_right=torch.rand(4, v6.ROLE_DIM),
        attribute_bond_patch_index=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )


def test_baseline_mode_reproduces_v4_hinge() -> None:
    model = _model("none")
    assert model.attribute_encoder is None
    assert int(model.attribute_input_width) == 0
    assert (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        == V4_HINGE_TOTAL
    )


def test_attribute_modes_have_equal_parameters() -> None:
    totals = {}
    for mode in ("factorized_role", "count_control", "capacity_control"):
        model = _model(mode)
        totals[mode] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    assert len(set(totals.values())) == 1
    assert totals["factorized_role"] - V4_HINGE_TOTAL == V6_DELTA
    assert totals["factorized_role"] <= 105_000


def test_factorized_starts_equal_to_baseline_and_gets_gradients() -> None:
    torch.manual_seed(0)
    factor = _model("factorized_role")
    torch.manual_seed(0)
    base = _model("none")
    base_state = base.state_dict()
    factor_state = factor.state_dict()
    for key, value in base_state.items():
        if key not in factor_state:
            continue
        if key == "patch_encoder.layers.0.weight":
            factor_state[key][:, : value.shape[1]] = value.clone()
        elif factor_state[key].shape == value.shape:
            factor_state[key] = value.clone()
    factor.load_state_dict(factor_state)
    data = _attribute_data()
    factor.eval()
    base.eval()
    with torch.no_grad():
        out_factor = factor(data)
        out_base = base(data)
    assert torch.allclose(out_factor, out_base, atol=1e-6)
    factor.train()
    factor(data).sum().backward()
    # the fusion output layer receives gradient immediately; the upstream
    # layers get it from step 2 onwards (zero-initialised fusion = one-step
    # delayed, not permanently dead).
    fusion_grads = [
        p.grad
        for name, p in factor.named_parameters()
        if name.startswith("attribute_encoder.fusion") and p.grad is not None
    ]
    assert fusion_grads and all(torch.isfinite(g).all() for g in fusion_grads)
    assert any(float(g.abs().sum()) > 0 for g in fusion_grads)
    step = torch.optim.SGD(factor.parameters(), lr=0.1)
    step.step()
    step.zero_grad()
    factor(data).sum().backward()
    upstream = [
        p.grad
        for name, p in factor.named_parameters()
        if name == "attribute_encoder.atom_mlp.0.weight" and p.grad is not None
    ]
    assert upstream and any(float(g.abs().sum()) > 0 for g in upstream)


def test_v6_path_uses_no_exact_typed_token_embedding() -> None:
    model = _model("factorized_role")
    encoder = model.attribute_encoder
    assert encoder is not None
    assert int(encoder.atom_embedding.num_embeddings) == v6.ATOM_CATEGORIES
    assert int(encoder.bond_embedding.num_embeddings) == v6.BOND_CATEGORIES
    # the large corrected-type vocabulary is only ever the coarse token table
    assert int(model.typed_embedding.vocabulary_size) == SELECTION_TYPED_VOCAB
    # the attribute branch never references a corrected canonical key
    source = module._AttributeEncoder.forward.__code__.co_names
    assert "corrected_canonical_key" not in source


def test_role_fingerprint_is_versioned_and_stable() -> None:
    fingerprint = v6.attribute_role_fingerprint(2)
    assert fingerprint == v6.attribute_role_fingerprint(2)
    assert fingerprint != v6.attribute_role_fingerprint(3)
    assert v6.ROLE_DIM == 8
    assert v6.ATOM_CATEGORIES == module.ATOM_CATEGORIES
    assert v6.BOND_CATEGORIES == module.BOND_CATEGORIES
