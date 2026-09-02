"""Pytest smoke test for sampling/evaluation interoperability."""

from __future__ import annotations

from ksvd_research.core.graph import from_edges
from ksvd_research.evaluation.metrics import evaluate_bundle
from ksvd_research.sampling.sample import SampleConfig, sample_B0


def test_sample_bundle_can_be_evaluated() -> None:
    graph = from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    bundle = sample_B0(graph, SampleConfig(max_nodes=3, seed=11))
    metrics = evaluate_bundle(graph, bundle)
    assert metrics["method"] == "B0"
    assert metrics["n_subgraphs"] == graph.n
    assert 0.0 <= metrics["edge_cover"] <= 1.0
