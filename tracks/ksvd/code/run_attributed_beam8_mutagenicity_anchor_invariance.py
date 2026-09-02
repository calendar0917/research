"""Relabel-invariance audit for attributed Beam8 typed anchors."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data_tud import load_tud
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_mutagenicity_anchor import attributed_anchor_item
from .run_attributed_beam8_mutagenicity_invariance import _graph, _sorted_rows, _tokens
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _adjacency,
    _atomic_json,
    _atomic_text,
)
from .run_beam8_nci1_compact_relation_followup import _compact_feature
from .run_luyin14_edge_aware_joint import _load_edge_pyg


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_MUTAGENICITY_ANCHOR_INVARIANCE_PROTOCOL_20260813.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/attributed_beam8_mutagenicity_anchor_invariance_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_MUTAGENICITY_ANCHOR_INVARIANCE_20260813.md"


def _prepare(
    index: int,
    graph: Any,
    node_features: np.ndarray,
    typed: np.ndarray,
    edge_dim: int,
) -> tuple[Any, Any]:
    base = prepare_attributed_beam_graph(
        index, graph, 0, node_features, typed, edge_dim=edge_dim
    )
    anchor, _count = attributed_anchor_item(
        base, graph, node_features, typed, edge_dim=edge_dim
    )
    return base, anchor


def _matches(original: Any, relabeled: Any, permutation: np.ndarray) -> dict[str, bool]:
    original_tokens = _tokens(original)
    relabeled_tokens = _tokens(relabeled)
    mapped = tuple(
        frozenset(int(permutation[node]) for node in nodes)
        for nodes in relabeled.node_sets
    )
    return {
        "chain_exact_match": mapped == original.node_sets,
        "token_row_match": original_tokens.shape == relabeled_tokens.shape
        and np.array_equal(original_tokens, relabeled_tokens),
        "token_multiset_match": original_tokens.shape == relabeled_tokens.shape
        and _sorted_rows(original_tokens) == _sorted_rows(relabeled_tokens),
        "readout_match": np.allclose(
            _compact_feature(original, original_tokens),
            _compact_feature(relabeled, relabeled_tokens),
            atol=1e-12,
            rtol=0,
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, _labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, _raw_labels, _classes, edge_dim = _load_edge_pyg(
        "Mutagenicity", args.dataset_root
    )
    rows = []
    count = min(args.graphs, len(graphs))
    for index, (graph, node_features, data) in enumerate(
        zip(graphs[:count], features[:count], raw[:count])
    ):
        adjacency = _adjacency(graph)
        typed = _typed_adjacency(data, edge_dim)
        original_base, original_anchor = _prepare(
            index, graph, node_features, typed, edge_dim
        )
        for trial in range(args.permutations):
            permutation = np.random.default_rng(
                930000 + index * 1009 + trial
            ).permutation(graph.n)
            relabeled_graph = _graph(adjacency[np.ix_(permutation, permutation)])
            relabeled_typed = typed[np.ix_(permutation, permutation)]
            relabeled_features = np.asarray(node_features)[permutation]
            relabeled_base, relabeled_anchor = _prepare(
                index,
                relabeled_graph,
                relabeled_features,
                relabeled_typed,
                edge_dim,
            )
            rows.append(
                {
                    "graph_index": index,
                    "permutation": trial,
                    "base": _matches(original_base, relabeled_base, permutation),
                    "anchor": _matches(original_anchor, relabeled_anchor, permutation),
                }
            )
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"{index + 1}/{count}", flush=True)
    rates = {
        family: {
            key: float(np.mean([row[family][key] for row in rows]))
            for key in (
                "chain_exact_match",
                "token_row_match",
                "token_multiset_match",
                "readout_match",
            )
        }
        for family in ("base", "anchor")
    }
    required = ("token_row_match", "token_multiset_match", "readout_match")
    gate = all(rates[family][key] == 1.0 for family in rates for key in required)
    failures = [
        row
        for row in rows
        if any(not row[family][key] for family in ("base", "anchor") for key in required)
    ][:20]
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "config": {"graphs": count, "permutations": args.permutations},
        "rates": rates,
        "failure_examples": failures,
        "decision": (
            "ATTRIBUTED_ANCHOR_INVARIANCE_PASS"
            if gate
            else "ATTRIBUTED_ANCHOR_INVARIANCE_FAILED_INVALIDATE_CLASSIFICATION"
        ),
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Attributed Beam8/Mutagenicity anchor invariance audit",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']}`",
        "",
        "| family | invariant | match rate | gate |",
        "|---|---|---:|---:|",
    ]
    for family, rates in payload["rates"].items():
        for key, value in rates.items():
            lines.append(
                f"| {family} | {key} | {value:.6f} | {'diagnostic' if key == 'chain_exact_match' else 'required'} |"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- concrete node IDs may differ inside attributed automorphisms; tokens and compact readout may not.",
            "- anchor gate fails时，classification 只保留为实现诊断，不作为方法证据。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--graphs", type=int, default=512)
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
