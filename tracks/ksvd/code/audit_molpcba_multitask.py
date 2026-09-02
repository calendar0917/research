"""Read-only suitability audit for the MolPCBA multi-task route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _task_summary(labels: np.ndarray, indices: np.ndarray) -> list[dict[str, int | float]]:
    output = []
    for task in range(labels.shape[1]):
        values = labels[indices, task]
        known = values[~np.isnan(values)]
        positive = int(np.sum(known > 0.5))
        negative = int(np.sum(known <= 0.5))
        output.append({
            "task": task,
            "known": int(len(known)),
            "positive": positive,
            "negative": negative,
            "positive_rate": float(positive / max(len(known), 1)),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ogb")
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph-sample", type=int, default=20000)
    args = parser.parse_args()
    # OGB's processed cache is a trusted local pickle.  PyTorch 2.6 changed
    # the default to weights_only=True, so match the compatibility guard used
    # by the existing MolHIV loader before opening it.
    import torch
    original_load = torch.load
    if not getattr(original_load, "_ksvd_ogb_patch", False):
        def compatible_load(*values, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.setdefault("weights_only", False)
            return original_load(*values, **kwargs)
        compatible_load._ksvd_ogb_patch = True  # type: ignore[attr-defined]
        torch.load = compatible_load  # type: ignore[assignment]
    from ogb.graphproppred import GraphPropPredDataset

    dataset = GraphPropPredDataset(name="ogbg-molpcba", root=args.root)
    labels = np.asarray(dataset.labels, dtype=np.float64)
    split = dataset.get_idx_split()
    per_split = {name: _task_summary(labels, np.asarray(indices, dtype=np.int64)) for name, indices in split.items()}
    all_summary = _task_summary(labels, np.arange(len(labels), dtype=np.int64))
    order = sorted(all_summary, key=lambda row: (-int(row["positive"]), int(row["task"])))
    rng = np.random.default_rng(0)
    sample = rng.choice(len(dataset), size=min(args.graph_sample, len(dataset)), replace=False)
    nodes, edges = [], []
    for index in sample:
        graph, _ = dataset[int(index)]
        nodes.append(int(graph["num_nodes"]))
        edges.append(int(np.asarray(graph["edge_index"]).shape[1] // 2))
    result = {
        "dataset": "ogbg-molpcba",
        "n_graphs": int(len(dataset)),
        "n_tasks": int(labels.shape[1]),
        "split_sizes": {name: int(len(indices)) for name, indices in split.items()},
        "all_tasks": all_summary,
        "tasks_by_positive_count": order,
        "per_split": per_split,
        "sampled_graph_size": {
            "n_graphs": int(len(sample)),
            "nodes_mean": float(np.mean(nodes)), "nodes_p95": float(np.quantile(nodes, 0.95)), "nodes_max": int(np.max(nodes)),
            "edges_mean": float(np.mean(edges)), "edges_p95": float(np.quantile(edges, 0.95)), "edges_max": int(np.max(edges)),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_graphs": result["n_graphs"], "n_tasks": result["n_tasks"],
        "split_sizes": result["split_sizes"], "sampled_graph_size": result["sampled_graph_size"],
        "top_tasks": order[:12],
    }, indent=2))


if __name__ == "__main__":
    main()
