"""Evaluation metrics and graph-level KSVD encoding helpers."""

from .graph_level import (
    GraphLevelConfig,
    bundle_to_Y,
    encode_dataset,
    encode_graph,
    learn_shared_D_graph_level,
    sparse_code_patch_matrix,
    sparse_code_readouts,
)
from .metrics import compare_methods, evaluate_bundle

__all__ = [
    "GraphLevelConfig",
    "bundle_to_Y",
    "learn_shared_D_graph_level",
    "sparse_code_patch_matrix",
    "sparse_code_readouts",
    "encode_graph",
    "encode_dataset",
    "evaluate_bundle",
    "compare_methods",
]
