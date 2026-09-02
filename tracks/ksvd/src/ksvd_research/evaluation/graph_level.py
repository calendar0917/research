"""Graph-level KSVD pipeline API backed by the historical implementation."""

from .._legacy import import_legacy


_legacy = import_legacy("graph_level")
GraphLevelConfig = _legacy.GraphLevelConfig
bundle_to_Y = _legacy.bundle_to_Y
collect_train_Y = _legacy.collect_train_Y
encode_dataset = _legacy.encode_dataset
encode_graph = _legacy.encode_graph
encode_patch_matrix = _legacy.encode_patch_matrix
learn_shared_D_graph_level = _legacy.learn_shared_D_graph_level
sample_patches_B0 = _legacy.sample_patches_B0
sample_patches_graph_level = _legacy.sample_patches_graph_level
sparse_code_patch_matrix = _legacy.sparse_code_patch_matrix
sparse_code_readouts = _legacy.sparse_code_readouts

__all__ = [
    "GraphLevelConfig",
    "sample_patches_graph_level",
    "sample_patches_B0",
    "bundle_to_Y",
    "collect_train_Y",
    "learn_shared_D_graph_level",
    "sparse_code_patch_matrix",
    "sparse_code_readouts",
    "encode_patch_matrix",
    "encode_graph",
    "encode_dataset",
]
