"""Correctness tests for the Binding Composition Encoder (BCE).

Fast unit / static tests only: no full training, no official valid, no official
test data.

Coverage (numbered as in the BCE brief, section 17):

A. node relabel invariance (single edge / path-3 / triangle / branch / 4-cycle /
   ambiguous-decomposition support)
B. edge-order invariance
C. decomposition-order invariance (shuffle the decomposition list of a support)
D. exact induced support (support edges == original induced edges)
E. complete decomposition coverage vs an independent brute-force reference
F. provenance (every higher-order object traces back to atom/bond primitives)
G. binding sensitivity (atom / bond / endpoint attributes)
H. connectivity sensitivity
I. no GAT / scalar attention (static code audit)
J. gradient viability for every BCE module
K. batch invariance

plus target-free parameter-accounting checks.
"""

from __future__ import annotations

import inspect

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_binding_composition_encoder as bce,
)
from tracks.ksvd.experiments.luyin16 import structural_patch_encoder
from tracks.ksvd.experiments.luyin16.binding_composition_support import (
    brute_force_decompositions,
    build_chart,
)

# ---------------------------------------------------------------------------
# graph fixtures (single patch, root = 0 unless stated)
# ---------------------------------------------------------------------------

GRAPHS = {
    "single_edge": dict(
        atom=[0, 1], root_index=0, dist=[0, 1], edges=[(0, 1)], bond_types=[0]
    ),
    "path3": dict(
        atom=[0, 1, 2],
        root_index=0,
        dist=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        bond_types=[0, 1],
    ),
    "triangle": dict(
        atom=[0, 1, 2],
        root_index=0,
        dist=[0, 1, 1],
        edges=[(0, 1), (0, 2), (1, 2)],
        bond_types=[0, 1, 2],
    ),
    "branch": dict(
        atom=[0, 1, 2, 3, 4],
        root_index=0,
        dist=[0, 1, 1, 2, 2],
        edges=[(0, 1), (0, 2), (1, 3), (2, 4)],
        bond_types=[0, 1, 0, 2],
    ),
    "four_cycle": dict(
        atom=[0, 1, 2, 3],
        root_index=0,
        dist=[0, 1, 1, 2],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        bond_types=[0, 1, 2, 0],
    ),
    "ambiguous": dict(
        atom=[0, 1, 2, 3],
        root_index=0,
        dist=[0, 1, 1, 2],
        edges=[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
        bond_types=[0, 1, 2, 0, 1],
    ),
    "star4": dict(
        atom=[0, 1, 2, 3],
        root_index=0,
        dist=[0, 1, 1, 1],
        edges=[(0, 1), (0, 2), (0, 3)],
        bond_types=[0, 1, 2],
    ),
}


def _graph(name: str):
    return bce.synthetic_graph(**GRAPHS[name])


def _encoder(seed: int = 0) -> bce.BindingCompositionEncoder:
    return bce._encoder(seed)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _support_nodes(graph) -> list[frozenset[int]]:
    ptr = graph.sup_node_ptr.numpy()
    idx = graph.sup_node_idx.numpy()
    return [
        frozenset(int(v) for v in idx[ptr[i] : ptr[i + 1]])
        for i in range(len(ptr) - 1)
    ]


def _adjacency(name: str) -> list[set[int]]:
    spec = GRAPHS[name]
    n = len(spec["atom"])
    adj: list[set[int]] = [set() for _ in range(n)]
    for u, v in spec["edges"]:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _permute_decompositions(graph, perm: torch.Tensor):
    """Rebuild a graph with its decomposition rows reordered by ``perm``."""
    n_dec = int(graph.dec_child.numel())
    op = graph.dec_overlap_ptr.numpy().astype(np.int64)
    oi = graph.dec_overlap_idx.numpy().astype(np.int64)
    cp = graph.dec_cross_ptr.numpy().astype(np.int64)
    ci = graph.dec_cross_idx.numpy().astype(np.int64)
    order = perm.numpy().astype(np.int64)
    overlap_rows = [oi[op[i] : op[i + 1]] for i in order]
    cross_rows = [ci[cp[i] : cp[i + 1]] for i in order]
    new_overlap_idx = np.concatenate(overlap_rows) if overlap_rows else np.zeros(0, np.int64)
    new_cross_idx = np.concatenate(cross_rows) if cross_rows else np.zeros(0, np.int64)
    new_overlap_ptr = np.concatenate(
        [[0], np.cumsum([len(r) for r in overlap_rows])]
    ).astype(np.int64)
    new_cross_ptr = np.concatenate(
        [[0], np.cumsum([len(r) for r in cross_rows])]
    ).astype(np.int64)
    fields = {key: getattr(graph, key) for key in bce._STRUCT_KEYS}
    fields["dec_child"] = graph.dec_child[perm]
    fields["dec_parent_a"] = graph.dec_parent_a[perm]
    fields["dec_parent_b"] = graph.dec_parent_b[perm]
    fields["dec_overlap_ptr"] = torch.from_numpy(new_overlap_ptr)
    fields["dec_overlap_idx"] = torch.from_numpy(new_overlap_idx)
    fields["dec_cross_ptr"] = torch.from_numpy(new_cross_ptr)
    fields["dec_cross_idx"] = torch.from_numpy(new_cross_idx)
    assert n_dec == len(order)
    return bce._PlainGraph(**fields)


# ---------------------------------------------------------------------------
# A. node relabel invariance
# ---------------------------------------------------------------------------


def test_node_relabel_invariance_all_fixtures():
    encoder = _encoder(0)
    for name in GRAPHS:
        graph = _graph(name)
        n = int(graph.struct_atom.numel())
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(1234 + n))
        relabelled = bce.relabel_graph(graph, perm)
        with torch.no_grad():
            reference = encoder(graph)
            moved = encoder(relabelled)
        assert float((reference - moved).abs().max()) < 1.0e-5, name
        # support / provenance mapping back to the original labels
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(n)
        original_supports = _support_nodes(graph)
        moved_supports = _support_nodes(relabelled)
        # new label i corresponds to old label perm[i]
        mapped_back = {
            frozenset(int(perm[v]) for v in support)
            for support in moved_supports
        }
        assert mapped_back == set(original_supports), name


def test_ambiguous_support_has_multiple_decompositions():
    graph = _graph("ambiguous")
    support_nodes = _support_nodes(graph)
    counts: dict[frozenset[int], int] = {}
    for child in graph.dec_child.tolist():
        key = support_nodes[child]
        counts[key] = counts.get(key, 0) + 1
    assert any(count > 1 for count in counts.values())


# ---------------------------------------------------------------------------
# B. edge-order invariance
# ---------------------------------------------------------------------------


def test_edge_order_invariance():
    encoder = _encoder(1)
    for name in GRAPHS:
        graph = _graph(name)
        reordered = bce.reorder_edges(graph)
        with torch.no_grad():
            reference = encoder(graph)
            moved = encoder(reordered)
        assert float((reference - moved).abs().max()) < 1.0e-6, name


# ---------------------------------------------------------------------------
# C. decomposition-order invariance
# ---------------------------------------------------------------------------


def test_decomposition_order_invariance():
    encoder = _encoder(2)
    for name in ("path3", "triangle", "ambiguous", "four_cycle", "star4"):
        graph = _graph(name)
        n_dec = int(graph.dec_child.numel())
        if n_dec < 2:
            continue
        perm = torch.randperm(n_dec, generator=torch.Generator().manual_seed(99))
        reordered = _permute_decompositions(graph, perm)
        with torch.no_grad():
            reference = encoder(graph)
            moved = encoder(reordered)
            encoder(graph)
            h_reference = encoder.last_support_states.clone()
            encoder(reordered)
            h_moved = encoder.last_support_states.clone()
        # every support state must be unchanged (mean/std aggregation)
        assert float((h_reference - h_moved).abs().max()) < 1.0e-5, name
        assert float((reference - moved).abs().max()) < 1.0e-5, name


# ---------------------------------------------------------------------------
# D. exact induced support
# ---------------------------------------------------------------------------


def test_exact_induced_support_edges():
    for name, spec in GRAPHS.items():
        adj = _adjacency(name)
        graph = _graph(name)
        ptr = graph.sup_edge_ptr.numpy()
        idx = graph.sup_edge_idx.numpy()
        n_bonds = int(graph.struct_bond.numel()) // 2
        src = graph.struct_src.numpy().reshape(n_bonds, 2)[:, 0]
        dst = graph.struct_dst.numpy().reshape(n_bonds, 2)[:, 0]
        bond_list = [
            (int(min(u, v)), int(max(u, v))) for u, v in zip(src, dst)
        ]
        support_nodes = _support_nodes(graph)
        for index, support in enumerate(support_nodes):
            expected = {
                (u, v)
                for u in support
                for v in adj[u]
                if v in support and u < v
            }
            actual = {
                bond_list[b] for b in idx[ptr[index] : ptr[index + 1]]
            }
            assert actual == expected, (name, support)


# ---------------------------------------------------------------------------
# E. complete decomposition coverage (brute force)
# ---------------------------------------------------------------------------


def test_complete_decomposition_coverage_brute_force():
    for name, spec in GRAPHS.items():
        adj = _adjacency(name)
        graph = _graph(name)
        support_nodes = _support_nodes(graph)
        support_set = set(support_nodes)
        index_of = {support: i for i, support in enumerate(support_nodes)}
        # chart decompositions
        chart: dict[frozenset[int], set] = {}
        for child, parent_a, parent_b in zip(
            graph.dec_child.tolist(),
            graph.dec_parent_a.tolist(),
            graph.dec_parent_b.tolist(),
        ):
            pair = tuple(
                sorted(
                    (support_nodes[parent_a], support_nodes[parent_b]),
                    key=lambda s: tuple(sorted(s)),
                )
            )
            chart.setdefault(support_nodes[child], set()).add(pair)
        for support in support_nodes:
            if len(support) < 3:
                assert support not in chart
                continue
            reference = brute_force_decompositions(support, support_set, adj)
            assert chart.get(support, set()) == reference, (name, support)
            # no reversed duplicates
            assert len(chart[support]) == len(
                {tuple(sorted(pair, key=lambda s: tuple(sorted(s)))) for pair in chart[support]}
            )


# ---------------------------------------------------------------------------
# F. provenance
# ---------------------------------------------------------------------------


def test_provenance_recurses_to_primitives():
    for name in ("ambiguous", "four_cycle", "star4"):
        graph = _graph(name)
        support_nodes = _support_nodes(graph)
        sizes = graph.sup_size.tolist()
        parents_of: dict[int, list[tuple[int, int]]] = {}
        for child, parent_a, parent_b in zip(
            graph.dec_child.tolist(),
            graph.dec_parent_a.tolist(),
            graph.dec_parent_b.tolist(),
        ):
            parents_of.setdefault(child, []).append((parent_a, parent_b))

        def leaves(index: int) -> set[int]:
            if sizes[index] <= 2:
                node_set = support_nodes[index]
                assert sizes[index] == len(node_set)
                return set(node_set)
            assert index in parents_of, (name, index)
            found: set[int] = set()
            for parent_a, parent_b in parents_of[index]:
                assert sizes[parent_a] < sizes[index]
                assert sizes[parent_b] < sizes[index]
                found |= leaves(parent_a) | leaves(parent_b)
            return found

        for index, size in enumerate(sizes):
            if size == 4:
                found = leaves(index)
                assert found == set(support_nodes[index])


# ---------------------------------------------------------------------------
# G. binding sensitivity
# ---------------------------------------------------------------------------


def test_binding_sensitivity():
    encoder = _encoder(3)
    base = bce.synthetic_graph(**GRAPHS["path3"])
    atom_changed = bce.synthetic_graph(
        atom=[0, 3, 2],
        root_index=0,
        dist=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        bond_types=[0, 1],
    )
    bond_changed = bce.synthetic_graph(
        atom=[0, 1, 2],
        root_index=0,
        dist=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        bond_types=[2, 1],
    )
    # endpoint attribute permutation must move the symmetric binding state too
    endpoint_permuted = bce.synthetic_graph(
        atom=[2, 0, 1],
        root_index=0,
        dist=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        bond_types=[0, 1],
    )
    with torch.no_grad():
        e_base = encoder(base)
        e_atom = encoder(atom_changed)
        e_bond = encoder(bond_changed)
        e_perm = encoder(endpoint_permuted)
    assert float((e_base - e_atom).abs().max()) > 1.0e-6
    assert float((e_base - e_bond).abs().max()) > 1.0e-6
    assert float((e_base - e_perm).abs().max()) > 1.0e-6


# ---------------------------------------------------------------------------
# H. connectivity sensitivity
# ---------------------------------------------------------------------------


def test_connectivity_sensitivity():
    encoder = _encoder(4)
    # identical atom / root / distance multisets and bond-type multiset,
    # different real connectivity (3 attaches through 1 vs through 2)
    left = bce.synthetic_graph(
        atom=[0, 1, 2, 3],
        root_index=0,
        dist=[0, 1, 1, 2],
        edges=[(0, 1), (0, 2), (1, 3)],
        bond_types=[0, 0, 0],
    )
    right = bce.synthetic_graph(
        atom=[0, 1, 2, 3],
        root_index=0,
        dist=[0, 1, 1, 2],
        edges=[(0, 1), (0, 2), (2, 3)],
        bond_types=[0, 0, 0],
    )
    with torch.no_grad():
        e_left = encoder(left)
        e_right = encoder(right)
    assert float((e_left - e_right).abs().max()) > 1.0e-6


# ---------------------------------------------------------------------------
# I. no GAT / scalar attention (static audit)
# ---------------------------------------------------------------------------


def test_no_attention_or_scalar_gating():
    import ast
    import textwrap

    source = textwrap.dedent(
        inspect.getsource(structural_patch_encoder.BindingCompositionEncoder)
    )
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    for token in (
        "softmax",
        "Softmax",
        "attention",
        "Attention",
        "gumbel",
        "topk",
        "sigmoid",
        "MultiheadAttention",
        "sigmoid_",
    ):
        assert token not in names, token
    encoder = _encoder(0)
    module_names = [name for name, _ in encoder.named_modules()]
    assert not any("attn" in name or "gate" in name for name in module_names)
    # the aggregation over decompositions is mean + std, no learned weighting
    assert "candidate_mean" in source and "candidate_std" in source


# ---------------------------------------------------------------------------
# J. gradient viability
# ---------------------------------------------------------------------------


def test_gradient_viability():
    encoder = _encoder(5)
    encoder.train()
    graph = _graph("ambiguous")
    loss = encoder(graph).pow(2).mean()
    loss.backward()
    for name in (
        "atom_mlp",
        "bind_mlp",
        "compose",
        "support_update",
        "fusion",
    ):
        module = getattr(encoder, name)
        total = sum(
            float(parameter.grad.detach().pow(2).sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        assert total > 0.0, name
        for parameter in module.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()


# ---------------------------------------------------------------------------
# K. batch invariance
# ---------------------------------------------------------------------------


def test_batch_invariance():
    encoder = _encoder(6)
    graphs = [_graph(name) for name in ("ambiguous", "path3", "four_cycle")]
    with torch.no_grad():
        batched = encoder(bce.synthetic_batch(graphs))
        singles = torch.cat([encoder(graph) for graph in graphs])
    assert float((batched - singles).abs().max()) < 1.0e-5


# ---------------------------------------------------------------------------
# parameter accounting / no vocabulary
# ---------------------------------------------------------------------------


def test_encoder_has_no_vocabulary_and_fits_budget():
    encoder = _encoder(0)
    total = sum(parameter.numel() for parameter in encoder.parameters())
    assert total <= 120000
    for name, parameter in encoder.named_parameters():
        if parameter.ndim >= 2 and "embedding" in name:
            assert parameter.shape[0] <= 64, name
    # no attention / gate submodules
    assert not any(
        isinstance(module, torch.nn.MultiheadAttention)
        for module in encoder.modules()
    )
