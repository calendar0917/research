"""CPU-only correctness tests for the SSRC Phase 0 calibration.

Run:

    uv run pytest -q tracks/structural-sufficiency/results/phase0/tests
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import ssrc_toy as toy  # noqa: E402


def test_cycle_distances_are_exact() -> None:
    g6 = toy.cycle_graph(24, 6)
    g9 = toy.cycle_graph(24, 9)
    assert toy.graph_distance(g6, *toy.markers(g6)) == 6
    assert toy.graph_distance(g9, *toy.markers(g9)) == 9


def test_caterpillar_and_grid_distances_are_exact() -> None:
    for graph, d in [
        (toy.caterpillar_graph(n_spine=40, distance=6, margin=6), 6),
        (toy.caterpillar_graph(n_spine=40, distance=9, margin=6), 9),
        (toy.grid_graph(distance=6), 6),
        (toy.grid_graph(distance=9), 9),
    ]:
        assert toy.graph_distance(graph, *toy.markers(graph)) == d


def test_cycle_e0_e1_e2_collide_and_e3_separates() -> None:
    g6 = toy.cycle_graph(24, 6)
    g9 = toy.cycle_graph(24, 9)
    for r in (0, 1, 2):
        assert toy.encoding(g6, r) == toy.encoding(g9, r)
        assert toy.both_marks_count(g6, r) == 0
        assert toy.both_marks_count(g9, r) == 0
    assert toy.encoding(g6, 3) != toy.encoding(g9, 3)
    assert toy.both_marks_count(g6, 3) == 1
    assert toy.both_marks_count(g9, 3) == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda d: toy.cycle_graph(24, d),
        lambda d: toy.caterpillar_graph(n_spine=40, distance=d, margin=6),
        lambda d: toy.grid_graph(width=20, height=9, distance=d, a_x=4, a_y=4),
    ],
)
def test_second_and_third_family_e2_collide_e3_separates(factory) -> None:
    g6, g9 = factory(6), factory(9)
    assert toy.encoding(g6, 2) == toy.encoding(g9, 2)
    assert toy.encoding(g6, 3) != toy.encoding(g9, 3)
    assert toy.both_marks_count(g6, 2) == 0
    assert toy.both_marks_count(g9, 2) == 0
    assert toy.both_marks_count(g6, 3) == 1
    assert toy.both_marks_count(g9, 3) == 0


def test_canonical_key_matches_exhaustive_and_networkx_on_c24() -> None:
    for d in (6, 9):
        graph = toy.cycle_graph(24, d)
        for r in range(0, 4):
            patches = toy.all_rooted_balls(graph, r)
            for p1, p2 in itertools.combinations(patches, 2):
                a = toy.exhaustive_canonical(p1) == toy.exhaustive_canonical(p2)
                b = toy.pynauty_key(p1) == toy.pynauty_key(p2)
                c = toy.nx_isomorphic_rooted(p1, p2)
                assert a == b == c


def test_canonical_key_matches_networkx_on_larger_patches() -> None:
    rng = random.Random(20261018)
    for graph, r in [
        (toy.caterpillar_graph(n_spine=40, distance=6, margin=6), 3),
        (toy.grid_graph(distance=9), 3),
    ]:
        patches = toy.all_rooted_balls(graph, r)
        for _ in range(100):
            p1, p2 = rng.sample(patches, 2)
            assert (toy.pynauty_key(p1) == toy.pynauty_key(p2)) == toy.nx_isomorphic_rooted(p1, p2)


def test_encoding_is_invariant_to_node_relabelling() -> None:
    cases = [
        toy.cycle_graph(24, 6),
        toy.cycle_graph(24, 9),
        toy.caterpillar_graph(n_spine=40, distance=6, margin=6),
        toy.grid_graph(distance=9),
    ]
    for graph in cases:
        result = toy.check_relabel_invariance(graph, radii=range(0, 5), n_permutations=30, seed=0)
        assert result["passed"], result["failures"]


def test_encoding_is_invariant_to_edge_order() -> None:
    graph = toy.grid_graph(distance=6)
    rng = random.Random(1)
    shuffled = list(graph.edges)
    rng.shuffle(shuffled)
    other = toy.AttributedGraph(n=graph.n, node_types=graph.node_types, edges=tuple(sorted(shuffled)))
    for r in range(0, 4):
        assert toy.encoding(graph, r) == toy.encoding(other, r)


def test_encoder_does_not_receive_labels_or_seed() -> None:
    """The encoder signature is structural only; labels cannot change E_r."""
    import inspect

    sig = inspect.signature(toy.encoding)
    assert list(sig.parameters) == ["graph", "radius"]
    graph = toy.cycle_graph(24, 6)
    # A label array is external; changing it cannot change the structural call.
    labels_a = [0] * graph.n
    labels_b = [1] * graph.n
    assert labels_a != labels_b
    assert toy.encoding(graph, 2) == toy.encoding(graph, 2)


def test_collision_stats_are_task_relative() -> None:
    """E_2 collides for d=6/9 but not for the binary label [d <= 4]."""
    enc = {d: toy.encoding(toy.cycle_graph(24, d), 2) for d in range(1, 13)}
    stats = toy.collision_stats(enc, label_threshold=4)
    assert stats["same_6_9"] is True
    assert stats["n_collision_pairs"] > 0
    assert stats["label_mixed_groups"] == 0
    group_with_6 = next(g["d_values"] for g in stats["groups"] if 6 in g["d_values"])
    assert 9 in group_with_6


def test_positive_control_r2_is_perfect_but_r0_r1_are_not() -> None:
    for factory, distances in [
        (lambda d: toy.cycle_graph(24, d), range(1, 13)),
        (lambda d: toy.caterpillar_graph(n_spine=40, distance=d, margin=6), range(1, 13)),
        (lambda d: toy.grid_graph(width=20, height=9, distance=d, a_x=4, a_y=4), range(1, 10)),
    ]:
        for r, expected in [(0, False), (1, False), (2, True)]:
            correct = sum(toy.oracle_positive_control(factory(d), r, 4)[0] == toy.oracle_positive_control(factory(d), r, 4)[1] for d in distances)
            total = len(list(distances))
            if expected:
                assert correct == total
            else:
                assert correct < total


def test_dynamic_oracle_needs_depth_3_for_d6_and_depth_5_for_d9() -> None:
    for graph in [toy.cycle_graph(24, 6), toy.grid_graph(distance=6)]:
        assert toy.dynamic_oracle_detect(graph, 2) is False
        assert toy.dynamic_oracle_detect(graph, 3) is True
    for graph in [toy.cycle_graph(24, 9), toy.grid_graph(distance=9)]:
        assert toy.dynamic_oracle_detect(graph, 4) is False
        assert toy.dynamic_oracle_detect(graph, 5) is True


def test_noniterative_oracle_separates_the_pair() -> None:
    for graph in [toy.cycle_graph(24, 6), toy.caterpillar_graph(n_spine=40, distance=6, margin=6), toy.grid_graph(distance=6)]:
        assert toy.noniter_distance_oracle(graph) == 6
    for graph in [toy.cycle_graph(24, 9), toy.caterpillar_graph(n_spine=40, distance=9, margin=6), toy.grid_graph(distance=9)]:
        assert toy.noniter_distance_oracle(graph) == 9


def test_encoding_is_a_multiset_not_a_set() -> None:
    graph = toy.cycle_graph(24, 6)
    enc = toy.encoding(graph, 2)
    assert len(enc) == graph.n
    assert len(set(enc)) < len(enc)  # symmetric plain patches repeat


def test_node_and_edge_colours_are_part_of_the_key() -> None:
    base = toy.rooted_ball(toy.cycle_graph(24, 6), 10, 2)  # plain C root
    changed_node = toy.RootedPatch(
        node_types=(toy.TYPE_A,) + base.node_types[1:],
        dists=base.dists,
        edges=base.edges,
        original_nodes=base.original_nodes,
    )
    assert toy.pynauty_key(base) != toy.pynauty_key(changed_node)
    assert not toy.nx_isomorphic_rooted(base, changed_node)

    u, v, t = base.edges[0]
    changed_edge = toy.RootedPatch(
        node_types=base.node_types,
        dists=base.dists,
        edges=((u, v, t + 10),) + base.edges[1:],
        original_nodes=base.original_nodes,
    )
    assert toy.pynauty_key(base) != toy.pynauty_key(changed_edge)
    assert not toy.nx_isomorphic_rooted(base, changed_edge)
