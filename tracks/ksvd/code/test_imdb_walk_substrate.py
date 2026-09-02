"""Self-tests for the IMDB R0 WALK substrate audit utilities."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .imdb_walk_substrate import (
    exact_isomorphism_groups,
    extract_walk_patch_graphs,
    isomorphism_summary,
    load_tu_structure_text,
    mapped_trajectory_invariance_audit,
    patch_substrate_summary,
)


def _write_tiny_tu(root: Path) -> None:
    raw = root / "TINY" / "raw"
    raw.mkdir(parents=True)
    # Two relabeled 6-cycles with different labels, plus a 6-node path.
    graphs = [
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        [(0, 2), (2, 4), (4, 1), (1, 5), (5, 3), (3, 0)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    ]
    indicators = []
    edges = []
    offset = 0
    for graph_id, graph_edges in enumerate(graphs, start=1):
        indicators.extend([graph_id] * 6)
        for left, right in graph_edges:
            edges.append((offset + left + 1, offset + right + 1))
            edges.append((offset + right + 1, offset + left + 1))
        offset += 6
    np.savetxt(raw / "TINY_graph_indicator.txt", indicators, fmt="%d")
    np.savetxt(raw / "TINY_graph_labels.txt", [-1, 1, -1], fmt="%d")
    np.savetxt(raw / "TINY_A.txt", edges, fmt="%d", delimiter=",")


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_tiny_tu(root)
        graphs = load_tu_structure_text(root / "TINY")
        assert len(graphs) == 3
        assert [graph.label for graph in graphs] == [0, 1, 0]
        assert all(graph.adjacency.shape == (6, 6) for graph in graphs)

        groups = exact_isomorphism_groups(graphs)
        summary = isomorphism_summary(groups, len(graphs))
        assert summary["exact_structure_group_count"] == 2
        assert summary["label_conflict_group_count"] == 1
        assert summary["graphs_in_label_conflict_groups"] == 2

        patches = extract_walk_patch_graphs(
            graphs, sampling_seed=17, patch_size=6, max_patches_per_graph=6
        )
        substrate = patch_substrate_summary(patches)
        assert substrate["patch_count"] == 18
        assert substrate["root_coverage"]["full_coverage_graph_count"] == 3
        assert substrate["tree_fraction"] > 0.0
        mapped = mapped_trajectory_invariance_audit(
            graphs, patches, seed=17, graph_limit=3
        )
        assert mapped["passes_exact_mapped_trajectory_gate"]
    print("imdb_walk_substrate self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
