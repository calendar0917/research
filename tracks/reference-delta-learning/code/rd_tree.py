"""Exact incremental evaluation of a trained XGBoost forest.

The protocol forbids re-walking the whole forest every round ("avoid repeating
the full-forest traversal per round") but requires every round's three-split
metric.  The compromise is the standard one: extract every tree once from the
booster's JSON, then add one tree's contribution per round to a float32
accumulator, in tree order, exactly like XGBoost's own ``PredictBatch``
(``bst_float`` accumulation).  A full-prefix ``booster.predict`` cross-check is
run at the pre-registered checkpoints (see ``rd_xgb``).

The pair feature layout is why this module exists: ``z(G,R) = [x_G, x_R,
x_G - x_R]`` is stored as three feature-major blocks ``(2059, n_rows)`` instead
of one wide row-major matrix.  A tree split on column ``c`` therefore reads
``blocks[c // 2059][c % 2059]`` with a sign, and the reverse pair
``z(R,G) = [x_R, x_G, x_R - x_G]`` is the same blocks with X1/X2 swapped and X3
negated -- no extra 10 GB permutation copy is ever materialised.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import rd_common as C


def extract_trees(booster, feature_names: Sequence[str], *, rounds: int | None = None) -> list[dict[str, np.ndarray]]:
    """Per-tree float32 arrays: left/right/split_index/condition/default_left."""
    payload = C.SQX.export_trees(booster, feature_names, rounds=rounds or int(booster.num_boosted_rounds()))
    trees: list[dict[str, np.ndarray]] = []
    for entry in payload["trees"]:
        trees.append(
            {
                "left": np.asarray(entry["left"], dtype=np.int64),
                "right": np.asarray(entry["right"], dtype=np.int64),
                "feature": np.asarray(entry["split_index"], dtype=np.int64),
                "condition": np.asarray(entry["split_condition"], dtype=np.float32),
                "default_left": np.asarray(entry["default_left"], dtype=np.int64),
                "split_type": np.asarray(entry["split_type"], dtype=np.int64),
            }
        )
    for tree in trees:
        if tree["split_type"].size and int(tree["split_type"].max()) != 0:
            raise ValueError("categorical split found; the frozen recipe has numeric splits only")
    return trees


class BlockLayout:
    """Accessor for a feature-major multi-block pair matrix."""

    def __init__(self, blocks: Sequence[np.ndarray], n_features: int, *, reverse: bool = False) -> None:
        if len(blocks) != 3:
            raise ValueError("expected three feature blocks")
        self.blocks = blocks
        self.n_features = int(n_features)
        # forward: [X1, X2, X3]; reverse: [X2, X1, -X3]
        if reverse:
            self.order = (1, 0, 2)
            self.sign = (1.0, 1.0, -1.0)
        else:
            self.order = (0, 1, 2)
            self.sign = (1.0, 1.0, 1.0)

    def gather(self, columns: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
        for column in np.unique(columns):
            selected = columns == column
            block_index = int(column) // self.n_features
            offset = int(column) - block_index * self.n_features
            source = self.blocks[self.order[block_index]]
            values = source[offset, rows[selected]]
            if self.sign[block_index] < 0:
                # ``out[boolean_mask]`` is a copy, so negate into a local first
                values = np.negative(values)
            out[selected] = values


class DenseLayout:
    """Accessor for a plain dense design matrix (the D arm)."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.n_features = int(matrix.shape[1])

    def gather(self, columns: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
        for column in np.unique(columns):
            selected = columns == column
            out[selected] = self.matrix[rows[selected], int(column)]


def tree_leaf_values(tree: Mapping[str, np.ndarray], layout: Any, n_rows: int) -> np.ndarray:
    """Leaf value of one tree for every row (float32, matching XGBoost)."""
    left = tree["left"]
    right = tree["right"]
    feature = tree["feature"]
    condition = tree["condition"]
    node = np.zeros(n_rows, dtype=np.int64)
    out = np.zeros(n_rows, dtype=np.float32)
    active = np.arange(n_rows, dtype=np.int64)
    while active.size:
        index = node[active]
        is_leaf = left[index] == -1
        if is_leaf.any():
            leaves = active[is_leaf]
            out[leaves] = condition[index[is_leaf]]
            keep = ~is_leaf
            active = active[keep]
            index = index[keep]
            if active.size == 0:
                break
        columns = feature[index]
        values = np.empty(active.size, dtype=np.float32)
        layout.gather(columns, active, values)
        go_left = values < condition[index]
        node[active] = np.where(go_left, left[index], right[index])
    return out


def accumulate(
    accumulator: np.ndarray,
    trees: Sequence[Mapping[str, np.ndarray]],
    layout: Any,
    *,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    """Add trees ``[start, stop)`` to ``accumulator`` in order (float32)."""
    if accumulator.dtype != np.float32:
        raise ValueError("accumulator must be float32 to match XGBoost's own summation")
    upper = len(trees) if stop is None else int(stop)
    n_rows = int(accumulator.size)
    for tree in trees[int(start):upper]:
        np.add(accumulator, tree_leaf_values(tree, layout, n_rows), out=accumulator)
    return accumulator


def static_leaf_values(tree: Mapping[str, np.ndarray], layout: Any, n_rows: int) -> np.ndarray:
    """Alias kept for readability at call sites that evaluate a single round."""
    return tree_leaf_values(tree, layout, n_rows)
