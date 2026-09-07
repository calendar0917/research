"""Static verification for Compact-Hybrid-v2 (budget reallocation).

Compares Compact-v1 and Compact-v2 side by side on the real ZINC vocabulary
sizes (train-only selection fit and train+valid refit fit):

* per-module parameter audit (v1 vs v2, with deltas)
* budget PASS/FAIL for both phases (selection and refit)
* representation dimensions agreed to be unchanged (146/23/5/16/48/16/60/768)
* forward + backward with finite loss and gradients
* hybrid rank-4 exact-identity mapping (full <=> rare boundary, distinct
  rare rows, no merging / hashing / UNK collapse)

No training is performed.

Run with::

    uv run python -m tracks.ksvd.code.test_compact_hybrid_v2_static
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module  # noqa: E402

# Official ZINC vocabulary sizes recorded by the 277053-parameter baseline
# and Compact-v1 runs: selection fits on official train, refit on train+valid.
SELECTION_TYPED_VOCAB = 6785
REFIT_TYPED_VOCAB = 7051
PARENT_VOCAB = 32
BUDGET = 100_000

V1 = dict(
    patch_hidden=48,
    pair_hidden=16,
    token_width=16,
    embedding_mode="hybrid",
    embedding_rank=2,
    hybrid_full_typed_tokens=768,
    hybrid_full_parent_tokens=32,
    center_context=True,
    center_context_hidden=60,
    graph_head_hidden_0=None,
    graph_head_hidden_1=None,
)
V2 = dict(V1, embedding_rank=4, graph_head_hidden_0=64, graph_head_hidden_1=32)

EXPECTED_V1_TOTAL = {"validation-selection": 98_579, "train-valid-refit": 99_111}
EXPECTED_V2_TOTAL = {"validation-selection": 98_549, "train-valid-refit": 99_613}


def _model(spec: dict, typed_vocab: int) -> module.PatchPathModel:
    return module.PatchPathModel(
        typed_vocab,
        PARENT_VOCAB,
        dropout=0.05,
        shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
        context_width=0,
        **spec,
    )


def _batch(typed_vocab: int) -> module.Data:
    generator = torch.Generator().manual_seed(101)
    n = 6
    return module.Data(
        patch_cont=torch.randn(n, module._shell_width_for_radius(2), generator=generator),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.tensor(
            [0, 1, 767, 768, 769, typed_vocab - 1], dtype=torch.long
        ),
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
    audited: dict[str, dict[str, dict[str, int]]] = {}
    for phase, typed_vocab, expected in (
        ("validation-selection", SELECTION_TYPED_VOCAB, EXPECTED_V1_TOTAL["validation-selection"]),
        ("train-valid-refit", REFIT_TYPED_VOCAB, EXPECTED_V1_TOTAL["train-valid-refit"]),
    ):
        v1_model = _model(V1, typed_vocab)
        v2_model = _model(V2, typed_vocab)
        audit_v1 = module.audit_parameters(v1_model)
        audit_v2 = module.audit_parameters(v2_model)
        for label, audit in (("Compact-v1", audit_v1), ("Compact-v2", audit_v2)):
            total = audit["total_trainable"]
            assert total == sum(
                p.numel() for p in _model(V1 if label == "Compact-v1" else V2, typed_vocab).parameters() if p.requires_grad
            ), (phase, label)
        assert audit_v1["total_trainable"] == expected, (phase, audit_v1["total_trainable"])
        assert audit_v2["total_trainable"] == EXPECTED_V2_TOTAL[phase], phase
        assert audit_v1["total_trainable"] <= BUDGET
        assert audit_v2["total_trainable"] <= BUDGET
        audited[phase] = {
            "v1": audit_v1,
            "v2": audit_v2,
        }
        checks += 1

        # Representation dimensions must be identical between v1 and v2.
        for key in (
            "patch_descriptor_dim",
            "context_descriptor_dim",
            "typed_token_width",
            "parent_token_width",
            "patch_hidden_dim",
            "relation_descriptor_dim",
            "pair_hidden_dim",
            "distance_buckets",
            "unary_pooled_dim",
            "pair_pooled_dim",
            "graph_readout_dim",
            "center_context_hidden_dim",
            "global_encoder_width",
        ):
            assert audit_v1["dimensions"][key] == audit_v2["dimensions"][key], key
        assert audit_v1["dimensions"]["patch_descriptor_dim"] == 146
        assert audit_v1["dimensions"]["relation_descriptor_dim"] == 23
        assert audit_v1["dimensions"]["distance_buckets"] == 5
        assert audit_v1["dimensions"]["typed_token_width"] == 16
        assert audit_v1["dimensions"]["patch_hidden_dim"] == 48
        assert audit_v1["dimensions"]["pair_hidden_dim"] == 16
        assert audit_v1["dimensions"]["center_context_hidden_dim"] == 60
        assert audit_v1["embedding"]["typed"]["full_count"] == 768
        assert audit_v2["embedding"]["typed"]["full_count"] == 768
        assert audit_v1["embedding"]["typed"]["rank"] == 2
        assert audit_v2["embedding"]["typed"]["rank"] == 4
        checks += 1

        # Forward + backward with real descriptor widths.
        v2_model.train()
        data = _batch(typed_vocab)
        prediction = v2_model(data)
        assert prediction.shape == (1,)
        loss = torch.nn.functional.l1_loss(prediction, data.y)
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in v2_model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        checks += 1

        # Hybrid mapping semantics: IDs [0, 768) hit the full table, IDs
        # >= 768 hit distinct rank-4 factorized rows (no merging).
        embedding = v2_model.typed_embedding
        assert embedding.full_count == 768
        assert embedding.rare is not None
        token = torch.tensor(
            [0, 1, 767, 768, 769, typed_vocab - 1], dtype=torch.long
        )
        with torch.no_grad():
            values = embedding(token)
        assert values.shape == (6, 16)
        assert not torch.allclose(values[3], values[4])
        assert not torch.allclose(values[4], values[5])
        assert embedding.full.weight.shape == (768, 16)
        assert embedding.rare.embedding.weight.shape == (typed_vocab - 768, 4)
        assert embedding.rare.projection.weight.shape == (16, 4)
        checks += 1

    print("=" * 78)
    print("Compact-v1 vs Compact-v2 static parameter audit (real ZINC vocab)")
    print("=" * 78)
    for phase in ("validation-selection", "train-valid-refit"):
        v1 = audited[phase]["v1"]
        v2 = audited[phase]["v2"]
        print(f"\n--- phase: {phase} (typed vocab {v1['embedding']['typed']['vocabulary_size']}) ---")
        print(f"{'module':<26} {'v1':>10} {'v2':>10} {'delta':>10}   pct of budget shift")
        for label, _attribute in module.PARAMETER_AUDIT_BLOCKS:
            a = int(v1["blocks"].get(label, 0))
            b = int(v2["blocks"].get(label, 0))
            print(f"{label:<26} {a:>10} {b:>10} {b - a:>+10}")
        print(f"{'TOTAL':<26} {v1['total_trainable']:>10} {v2['total_trainable']:>10} {v2['total_trainable'] - v1['total_trainable']:>+10}")
        print(
            f"budget: v1={v1['total_trainable']} <= {BUDGET} "
            f"({'PASS' if v1['total_trainable'] <= BUDGET else 'FAIL'}); "
            f"v2={v2['total_trainable']} <= {BUDGET} "
            f"({'PASS' if v2['total_trainable'] <= BUDGET else 'FAIL'})"
        )
        for key in ("patch_descriptor_dim", "typed_token_width", "parent_token_width",
                    "patch_hidden_dim", "relation_descriptor_dim", "pair_hidden_dim",
                    "distance_buckets", "unary_pooled_dim", "pair_pooled_dim",
                    "graph_readout_dim", "center_context_hidden_dim"):
            print(
                f"rep {key}: v1={v1['dimensions'][key]} v2={v2['dimensions'][key]} "
                f"({'unchanged' if v1['dimensions'][key] == v2['dimensions'][key] else 'CHANGED!'})"
            )
        print(
            f"embedding: v1 rank={v1['embedding']['typed']['rank']} "
            f"full={v1['embedding']['typed']['full_count']} | "
            f"v2 rank={v2['embedding']['typed']['rank']} "
            f"full={v2['embedding']['typed']['full_count']}"
        )
        print(
            f"graph head: v1 dims={v1['dimensions']['graph_head_hidden_dims']} "
            f"params={v1['blocks']['graph_head']} | "
            f"v2 dims={v2['dimensions']['graph_head_hidden_dims']} "
            f"params={v2['blocks']['graph_head']}"
        )

    print(f"\n[static] all checks passed ({checks})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
