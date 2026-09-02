"""Relabel-invariance audit for discrete-canonical, continuous-content ENZYMES Beam8."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data_tud import load_tud
from .pipeline import graph_from_edge_index
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_node_incidence_feasibility import (
    node_incidence_features,
    orbit_safe_features,
)
from .run_beam8_nci1_chain_classification import ROOT, _atomic_json, _atomic_text
from .run_beam8_nci1_compact_relation_followup import _compact_feature
from .run_enzymes_beam8_conditional_controls import (
    DEFAULT_ROOT,
    _binary_typed,
    _continuous_feature_colors,
)


PROTOCOL = "tracks/ksvd/docs/KSVD_ENZYMES_BEAM8_FEATURE_CANONICAL_INVARIANCE_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
DEFAULT_JSON = RESULT_DIR / "enzymes_beam8_feature_canonical_invariance_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ENZYMES_BEAM8_FEATURE_CANONICAL_INVARIANCE_20260814.md"


def _graph(adjacency: np.ndarray) -> Any:
    left, right = np.nonzero(adjacency)
    return graph_from_edge_index(
        len(adjacency), np.asarray([left, right], dtype=np.int64)
    )


def _tokens(item: Any) -> np.ndarray:
    return np.concatenate([item.vectors, item.node_histograms], axis=1)


def _sorted_rows(values: np.ndarray) -> list[tuple[float, ...]]:
    return sorted(tuple(float(value) for value in row) for row in values)


def _safe_incidence(
    item: Any,
    tokens: np.ndarray,
    typed: np.ndarray,
    content: np.ndarray,
    discrete: np.ndarray,
    canonical_types: np.ndarray,
) -> np.ndarray:
    direct, _diagnostic = node_incidence_features(
        item, typed.shape[0], tokens=tokens, include_relation=False
    )
    safe, _orbits = orbit_safe_features(
        direct,
        typed,
        content,
        canonical_node_features=discrete,
        canonical_node_types=canonical_types,
    )
    return safe


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, content_features, metadata = load_tud(
        "ENZYMES", args.dataset_root, use_node_attr=True
    )
    _graphs_discrete, labels_discrete, discrete_features, discrete_meta = load_tud(
        "ENZYMES", args.dataset_root, use_node_attr=False
    )
    if not np.array_equal(labels, labels_discrete):
        raise RuntimeError("ENZYMES loaders disagree")
    trials = []
    count = min(args.graphs, len(graphs))
    for index, (graph, content, discrete) in enumerate(
        zip(graphs[:count], content_features[:count], discrete_features[:count])
    ):
        adjacency = _binary_typed(graph)
        canonical_types = _continuous_feature_colors(np.asarray(content))
        base = prepare_attributed_beam_graph(
            index,
            graph,
            int(labels[index]),
            content,
            adjacency,
            edge_dim=1,
            canonical_node_features=discrete,
            canonical_node_types=canonical_types,
        )
        base_tokens = _tokens(base)
        base_readout = _compact_feature(base, base_tokens)
        base_incidence = _safe_incidence(
            base,
            base_tokens,
            adjacency,
            np.asarray(content),
            np.asarray(discrete),
            canonical_types,
        )
        for trial in range(args.permutations):
            permutation = np.random.default_rng(
                20260814 + index * 1009 + trial
            ).permutation(graph.n)
            rel_adjacency = adjacency[np.ix_(permutation, permutation)]
            rel_content = np.asarray(content)[permutation]
            rel_discrete = np.asarray(discrete)[permutation]
            rel_canonical_types = canonical_types[permutation]
            rel = prepare_attributed_beam_graph(
                index,
                _graph(rel_adjacency),
                int(labels[index]),
                rel_content,
                rel_adjacency,
                edge_dim=1,
                canonical_node_features=rel_discrete,
                canonical_node_types=rel_canonical_types,
            )
            rel_tokens = _tokens(rel)
            rel_readout = _compact_feature(rel, rel_tokens)
            rel_incidence = _safe_incidence(
                rel,
                rel_tokens,
                rel_adjacency,
                rel_content,
                rel_discrete,
                rel_canonical_types,
            )
            mapped_sets = tuple(
                frozenset(int(permutation[node]) for node in nodes)
                for nodes in rel.node_sets
            )
            mapped_incidence = np.zeros_like(rel_incidence)
            mapped_incidence[permutation] = rel_incidence
            trials.append(
                {
                    "graph_index": index,
                    "permutation": trial,
                    "chain_exact_match": mapped_sets == base.node_sets,
                    "token_row_match": base_tokens.shape == rel_tokens.shape
                    and np.allclose(base_tokens, rel_tokens, atol=1e-12, rtol=0),
                    "token_multiset_match": base_tokens.shape == rel_tokens.shape
                    and _sorted_rows(base_tokens) == _sorted_rows(rel_tokens),
                    "readout_match": base_readout.shape == rel_readout.shape
                    and np.allclose(base_readout, rel_readout, atol=1e-12, rtol=0),
                    "incidence_equivariance": base_incidence.shape
                    == mapped_incidence.shape
                    and np.allclose(
                        base_incidence, mapped_incidence, atol=1e-12, rtol=0
                    ),
                }
            )
        if (index + 1) % 16 == 0 or index + 1 == count:
            print(f"{index + 1}/{count}", flush=True)
    keys = (
        "chain_exact_match",
        "token_row_match",
        "token_multiset_match",
        "readout_match",
        "incidence_equivariance",
    )
    rates = {key: float(np.mean([row[key] for row in trials])) for key in keys}
    required = (
        "token_row_match",
        "token_multiset_match",
        "readout_match",
        "incidence_equivariance",
    )
    passed = all(rates[key] == 1.0 for key in required)
    return {
        "protocol": PROTOCOL,
        "dataset": metadata
        | {"canonical_feature_dim": discrete_meta["node_feat_dim"]},
        "config": {"graphs": count, "permutations": args.permutations},
        "rates": rates,
        "failure_examples": {
            key: [row for row in trials if not row[key]][:20] for key in keys
        },
        "decision": (
            "ENZYMES_FEATURE_CANONICAL_CONTINUOUS_CONTENT_INVARIANCE_PASS"
            if passed
            else "ENZYMES_BEAM8_INVARIANCE_FAILED"
        ),
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# ENZYMES full-feature-canonical Beam8 relabel-invariance audit",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']}`",
        "",
        "| invariant | match rate | role |",
        "|---|---:|---:|",
    ]
    for key, value in payload["rates"].items():
        role = "diagnostic" if key == "chain_exact_match" else "required"
        lines.append(f"| {key} | {value:.6f} | {role} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- complete feature-row colors define attributed automorphisms; structural vectors still use discrete labels.",
            "- token/readout invariance and mapped node-incidence equivariance are required.",
            "- Passing confirms that continuous patch content did not break relabel legality.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--graphs", type=int, default=64)
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
