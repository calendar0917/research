"""Audit delta magnitude of the v3 conditioning during real training.

Correction note: the model's `_condition_patch` is invoked inside `forward`;
we instrument by recomputing delta from the intermediate tensors would require
hooking.  Instead we train a few epochs and record:

* mean ||delta|| / ||e_patch|| over ring patches (via a forward hook),
* train L1 per epoch,
* valid MAE per epoch,
side by side with compact-v2 for the same seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch
import yaml

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as m
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc, global_feature_views
from torch_geometric.loader import DataLoader


def build(config_path: Path, mode: str, vocab: int, typed_vocab: int, parent_vocab: int):
    config = yaml.safe_load(config_path.read_text())
    mc = config["model"]
    return m.PatchPathModel(
        typed_vocab,
        parent_vocab,
        patch_hidden=int(mc["patch_hidden"]),
        pair_hidden=int(mc["pair_hidden"]),
        token_width=int(mc["token_width"]),
        dropout=float(mc["dropout"]),
        embedding_mode=str(mc["embedding_mode"]),
        embedding_rank=int(mc["embedding_rank"]),
        hybrid_full_typed_tokens=min(int(mc["hybrid_full_typed_tokens"]), typed_vocab),
        hybrid_full_parent_tokens=min(int(mc["hybrid_full_parent_tokens"]), parent_vocab),
        center_context=bool(mc["center_context"]),
        center_context_hidden=int(mc["center_context_hidden"]),
        graph_head_hidden_0=64,
        graph_head_hidden_1=32,
        shell_width=m._shell_width_for_radius(2),
        context_width=0,
        structural_context_mode=mode,
        structural_context_fusion="condition",
        structural_context_dim=4,
        structural_context_embedding_rank=1,
        structural_context_condition_rank=8,
        structural_context_vocabulary_size=vocab,
    )


def main() -> None:
    torch.manual_seed(0)
    root = Path("data/ZINC")
    train = _load_zinc(root, "train")
    valid = _load_zinc(root, "val")
    contexts = global_feature_views(train)["global_all"]
    valid_contexts = global_feature_views(valid)["global_all"]
    cache: dict[bytes, bytes] = {}
    records = [
        m._graph_record(d, contexts[i], cache, structural_mode="typed_ring", max_cycle_len=10)
        for i, d in enumerate(train[:2000])
    ]
    valid_records = [
        m._graph_record(d, valid_contexts[i], cache, structural_mode="typed_ring", max_cycle_len=10)
        for i, d in enumerate(valid[:300])
    ]
    config = yaml.safe_load(Path("tracks/ksvd/configs/luyin16/zinc_compact_v3_typed_ring_condition.yaml").read_text())
    train_data, _, audit = m._phase_data(records, records, config=config)
    vocab_size = audit["structural_context_vocabulary_size"]
    typed_vocab = audit["typed_vocabulary_size_with_oov"]
    valid_data, _, _ = m._phase_data(valid_records, valid_records, config=config)

    model = build(
        Path("tracks/ksvd/configs/luyin16/zinc_compact_v3_typed_ring_condition.yaml"),
        "typed_ring",
        vocab_size,
        typed_vocab,
        audit["parent_vocabulary_size_with_oov"],
    )
    # fix the typed vocabulary size for a real-size model
    loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(91011))
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ratios = []
    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in loader:
            pred = model(batch)
            loss = torch.nn.functional.l1_loss(pred, batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss) * len(batch.y)
            seen += len(batch.y)
        model.eval()
        with torch.no_grad():
            va = m._evaluate(model, valid_loader, torch.device("cpu"))
        # measure delta ratio on the first train batch
        with torch.no_grad():
            batch = train_data[0]
            e_patch = model.typed_embedding(batch.typed_token)
            e_ctx = model._structural_context_embedding_value(batch)
            mask = model._structural_no_ring_mask(batch)
            delta = model.context_delta_output(
                torch.tanh(model.context_patch_projection(e_patch))
                * torch.tanh(model.context_condition_projection(e_ctx))
            ) * mask
            denom = e_patch.norm(dim=1).clamp_min(1e-6)
            ratio = (delta.norm(dim=1) / denom)
            ratios.append(float(ratio.mean()))
        print(
            f"epoch {epoch}: train_l1={total_loss/max(seen,1):.5f} valid={va:.5f} "
            f"mean||delta||/||e_patch||={ratio.mean():.4f} (max {ratio.max():.4f})",
            flush=True,
        )
        if epoch == 8:
            print("W_out norm:", model.context_delta_output.weight.norm().item())
            print("W_p grad norm:", model.context_patch_projection.weight.grad.norm().item() if model.context_patch_projection.weight.grad is not None else None)


if __name__ == "__main__":
    main()
