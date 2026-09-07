"""Training-horizon audit for Compact-Hybrid-v2 (60 -> 120 epoch ceiling).

Single-variable experiment: the horizon config must differ from the
Compact-v2 baseline config ONLY in ``model.epochs`` (60 -> 120), keeping the
model, representation, optimizer, learning-rate rule (no scheduler), and the
patience-12 early-stopping rule identical.  These tests enforce that
invariance programmatically so a run cannot silently become a different
experiment.
"""

from __future__ import annotations

import yaml

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module

CONFIGS = module.REPO_ROOT / "tracks/ksvd/configs/luyin16"
V2_PATH = CONFIGS / "zinc_hierarchical_patch_relation_context_compact_hybrid_v2_budget_reallocation.yaml"
HORIZON_PATH = CONFIGS / "zinc_hierarchical_patch_relation_context_compact_hybrid_v2_horizon120.yaml"

# Allowed differences: candidate identity (protocol_id), plumbing output
# paths, and the single experimental variable (model.epochs 60 -> 120).
# Every other leaf must be identical.
ALLOWED_KEY_PATHS = {"protocol_id", "output.json", "output.markdown", "model.epochs"}

SELECTION_TYPED_VOCAB = 6785
REFIT_TYPED_VOCAB = 7051
PARENT_VOCAB = 32


def _leaves(mapping: dict, prefix: str = "") -> dict[str, object]:
    leaves: dict[str, object] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            leaves.update(_leaves(value, path))
        else:
            leaves[path] = value
    return leaves


def test_horizon_config_diff_is_only_epochs() -> None:
    v2 = yaml.safe_load(V2_PATH.read_text(encoding="utf-8"))
    horizon = yaml.safe_load(HORIZON_PATH.read_text(encoding="utf-8"))
    left = _leaves(v2)
    right = _leaves(horizon)
    differing = {
        path: (left[path], right[path])
        for path in left.keys() | right.keys()
        if left.get(path) != right.get(path)
    }
    allowed = {path: value for path, value in differing.items() if path in ALLOWED_KEY_PATHS}
    unexpected = {path: value for path, value in differing.items() if path not in ALLOWED_KEY_PATHS}
    assert allowed, "expected at least identity/output differences"
    assert not unexpected, f"unexpected config differences: {unexpected}"
    assert left["model.epochs"] == 60
    assert right["model.epochs"] == 120
    # Every other training/model/representation leaf is byte-identical.
    for path in (
        "seed",
        "model.patch_hidden",
        "model.pair_hidden",
        "model.token_width",
        "model.embedding_mode",
        "model.embedding_rank",
        "model.hybrid_full_typed_tokens",
        "model.hybrid_full_parent_tokens",
        "model.center_context",
        "model.center_context_hidden",
        "model.graph_head_hidden_0",
        "model.graph_head_hidden_1",
        "model.dropout",
        "model.batch_size",
        "model.learning_rate",
        "model.weight_decay",
        "model.patience",
        "model.parameter_audit",
        "model.expected_max_trainable_params",
        "representation.max_typed_tokens",
        "representation.minimum_typed_frequency",
        "representation.max_parent_tokens",
        "representation.minimum_parent_frequency",
        "runtime.torch_threads",
    ):
        assert left[path] == right[path], path


def _model_from_config(config: dict, typed_vocab: int) -> module.PatchPathModel:
    model = config["model"]
    return module.PatchPathModel(
        typed_vocab,
        PARENT_VOCAB,
        patch_hidden=int(model["patch_hidden"]),
        pair_hidden=int(model["pair_hidden"]),
        token_width=int(model["token_width"]),
        dropout=float(model["dropout"]),
        embedding_mode=str(model["embedding_mode"]),
        embedding_rank=int(model["embedding_rank"]),
        hybrid_full_typed_tokens=int(model["hybrid_full_typed_tokens"]),
        hybrid_full_parent_tokens=int(model["hybrid_full_parent_tokens"]),
        center_context=bool(model["center_context"]),
        center_context_hidden=int(model["center_context_hidden"]),
        graph_head_hidden_0=int(model["graph_head_hidden_0"]),
        graph_head_hidden_1=int(model["graph_head_hidden_1"]),
        shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
        context_width=0,
    )


def test_horizon_model_identical_parameters_and_representation() -> None:
    v2_config = yaml.safe_load(V2_PATH.read_text(encoding="utf-8"))
    horizon_config = yaml.safe_load(HORIZON_PATH.read_text(encoding="utf-8"))
    for typed_vocab in (SELECTION_TYPED_VOCAB, REFIT_TYPED_VOCAB):
        model_v2 = _model_from_config(v2_config, typed_vocab)
        model_h = _model_from_config(horizon_config, typed_vocab)
        audit_v2 = module.audit_parameters(model_v2)
        audit_h = module.audit_parameters(model_h)
        assert audit_h["total_trainable"] == audit_v2["total_trainable"] == (
            98_549 if typed_vocab == SELECTION_TYPED_VOCAB else 99_613
        )
        assert audit_h["blocks"] == audit_v2["blocks"]
        assert audit_h["dimensions"] == audit_v2["dimensions"]
        assert audit_h["embedding"] == audit_v2["embedding"]
        assert audit_h["dimensions"]["patch_descriptor_dim"] == 146
        assert audit_h["dimensions"]["relation_descriptor_dim"] == 23
        assert audit_h["dimensions"]["distance_buckets"] == 5
        assert audit_h["dimensions"]["typed_token_width"] == 16
        assert audit_h["dimensions"]["patch_hidden_dim"] == 48
        assert audit_h["dimensions"]["pair_hidden_dim"] == 16
        assert audit_h["dimensions"]["center_context_hidden_dim"] == 60
        assert audit_h["dimensions"]["graph_head_hidden_dims"] == [64, 32]
        # Identical parameterization: exact same number of parameters in every
        # block; the total must equal the global sum.
        assert audit_h["total_trainable"] == sum(
            p.numel() for p in model_h.parameters() if p.requires_grad
        )
