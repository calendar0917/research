"""Static verification for the Compact-Hybrid-v1 parameterization.

Builds the real PatchPathModel with the official ZINC vocabulary sizes
(train-only fit and train+valid refit fit), checks the per-module parameter
audit, the hybrid embedding boundary/identity semantics, and runs a forward +
backward pass with the real descriptor widths.  No training is performed.

Run with::

    uv run python -m tracks.ksvd.code.test_compact_hybrid_v1_static
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module  # noqa: E402

# Official ZINC vocabulary sizes produced by _fit_vocabulary on the same
# extraction code path (recorded by the 277053-parameter baseline run):
# selection phase fits on official train (6784 known + OOV = 6785);
# refit phase fits on official train+valid (7050 known + OOV = 7051).
SELECTION_TYPED_VOCAB = 6785
REFIT_TYPED_VOCAB = 7051
PARENT_VOCAB = 32  # 31 known + OOV

COMPACT = dict(
    patch_hidden=48,
    pair_hidden=16,
    token_width=16,
    embedding_mode="hybrid",
    embedding_rank=2,
    hybrid_full_typed_tokens=768,
    hybrid_full_parent_tokens=32,
    center_context=True,
    center_context_hidden=60,
)


def _model(typed_vocab: int) -> module.PatchPathModel:
    return module.PatchPathModel(
        typed_vocab,
        PARENT_VOCAB,
        dropout=0.05,
        shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
        context_width=0,
        **COMPACT,
    )


def _batch(typed_vocab: int, token_width: int, parent_width: int) -> Data:
    generator = torch.Generator().manual_seed(101)
    n = 6
    return Data(
        patch_cont=torch.randn(n, module._shell_width_for_radius(2), generator=generator),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.tensor([0, 1, 767, 768, 769, typed_vocab - 1], dtype=torch.long),
        parent_token=torch.tensor([0, 1, 2, 3, 31, 31], dtype=torch.long),
        pair_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
        pair_relation=torch.randn(5, module.RELATION_WIDTH, generator=generator),
        pair_bucket=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        global_context=torch.randn(1, module.GLOBAL_WIDTH, generator=generator),
        batch=torch.zeros(n, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=n,
    )


def main() -> int:
    checks = 0

    for label, typed_vocab, expected_total in (
        ("validation-selection", SELECTION_TYPED_VOCAB, 98_579),
        ("train-valid-refit", REFIT_TYPED_VOCAB, 99_111),
    ):
        model = _model(typed_vocab)
        audit = module.audit_parameters(model)
        module._print_parameter_audit(model, label)
        total = audit["total_trainable"]
        assert total == sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        assert total <= 100_000, (label, total)
        assert total == expected_total, (label, total, expected_total)
        checks += 1

        # Forward + backward on real descriptor widths.
        model.train()
        data = _batch(typed_vocab, model.token_width, model.parent_width)
        prediction = model(data)
        assert prediction.shape == (1,)
        loss = torch.nn.functional.l1_loss(prediction, data.y)
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        print(f"[static] phase={label} forward/backward OK (loss={loss.item():.6f})")
        checks += 1

        # Hybrid mapping semantics: IDs [0, 768) hit the full table, IDs
        # >= 768 hit distinct rank-2 factorized rows (no merging).
        embedding = model.typed_embedding
        assert embedding.full_count == 768
        assert embedding.rare is not None
        token = torch.tensor([0, 1, 767, 768, 769, typed_vocab - 1], dtype=torch.long)
        with torch.no_grad():
            values = embedding(token)
        assert values.shape == (6, 16)
        assert not torch.allclose(values[3], values[4])  # rare rows distinct
        assert not torch.allclose(values[4], values[5])
        # Full table row count and rare row count.
        assert embedding.full.weight.shape == (768, 16)
        assert embedding.rare.embedding.weight.shape == (typed_vocab - 768, 2)
        assert embedding.rare.projection.weight.shape == (16, 2)
        checks += 1

    # Parent vocabulary fully covered by the full table (31 known + OOV).
    parent = _model(SELECTION_TYPED_VOCAB).parent_embedding
    assert parent.full_count == PARENT_VOCAB
    assert parent.rare is None
    assert parent.full.weight.shape == (PARENT_VOCAB, 8)
    checks += 1

    print(f"[static] all checks passed ({checks})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
