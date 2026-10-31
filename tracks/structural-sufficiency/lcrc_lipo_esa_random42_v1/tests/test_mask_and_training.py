"""Mask-semantics and training-loop tests (the executable evidence that the
``torch`` attention backend keeps the upstream xformers mask semantics)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_mask_backends_agree_on_the_same_adjacency(structures):
    import r42_upstream as U

    mods = U.install()
    masked = mods["masked_layers"]
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(structures[:6])
    kw = dict(
        edge_index=batch.edge_index,
        batch_size=batch.num_graphs,
        max_items=128,
        batch_mapping=batch.batch,
        use_bfloat16=False,
    )
    torch_mask = masked.get_adj_mask_from_edge_index_edge(xformers_or_torch_attn="torch", **kw)
    xf_mask = masked.get_adj_mask_from_edge_index_edge(xformers_or_torch_attn="xformers", **kw)
    torch_allowed = torch_mask[:, 0] if torch_mask.dim() == 4 else torch_mask
    xf_allowed = (xf_mask[:, 0] == 0) if xf_mask.dim() == 4 else (xf_mask == 0)
    assert torch.equal(torch_allowed, xf_allowed)


def test_mask_matches_adjacency_reference(structures):
    import r42_upstream as U

    mods = U.install()
    masked = mods["masked_layers"]
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(structures[:4])
    mask = masked.get_adj_mask_from_edge_index_edge(
        edge_index=batch.edge_index,
        batch_size=batch.num_graphs,
        max_items=128,
        batch_mapping=batch.batch,
        xformers_or_torch_attn="torch",
        use_bfloat16=False,
    )
    allowed = mask[:, 0]
    edges_per_graph = torch.bincount(batch.batch.index_select(0, batch.edge_index[0]), minlength=batch.num_graphs)
    offset = 0
    for g in range(batch.num_graphs):
        n_e = int(edges_per_graph[g])
        block = batch.edge_index[:, offset : offset + n_e]
        offset += n_e
        src, dst = block[0].unsqueeze(1), block[1].unsqueeze(1)
        ref = (src == src.t()) | (src == dst.t()) | (dst == src.t()) | (dst == dst.t())
        ref.fill_diagonal_(False)
        assert torch.equal(allowed[g][:n_e, :n_e], ref)
        # Padding never attends.
        assert int(allowed[g][n_e:, :].sum()) == 0
        assert int(allowed[g][:, n_e:].sum()) == 0


def test_attention_scores_respect_the_mask():
    """An unmasked pair must not influence a masked query (math kernel, CPU)."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    q = torch.randn(1, 2, 1, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    mask_a = torch.tensor([[[True, False, False, False]]])
    mask_b = torch.tensor([[[True, True, True, True]]])
    out_a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_a)
    out_b = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_b)
    # With a single allowed key the output equals that key's projected value.
    assert torch.allclose(out_a.expand(-1, -1, 1, -1).reshape(-1), v[:, :, 0, :].reshape(-1), atol=1e-6)
    assert not torch.allclose(out_a, out_b)


def test_train_lcrc_smoke_writes_manifest(split, labels, tmp_path):
    import r42_common as C
    import r42_lcrc_train as LT

    payload = LT.train_lcrc(
        cfg=C.load_config()["models"]["lcrc"],
        split=split,
        labels=labels,
        device=torch.device("cpu"),
        run_dir=tmp_path,
        limit_train=32,
        limit_valid=16,
        max_epochs=2,
        soup_last=2,
        label="test",
    )
    assert payload["training"]["checkpoint_rule"] == "average_last_5_epoch_end_state_dicts"
    assert payload["training"]["soup_member_count"] == 2
    assert payload["training"]["early_stopping"] is False
    assert payload["test_labels_read"] is False
    assert payload["metrics"]["valid"]["units"] == "raw_logD"
    assert (tmp_path / "checkpoint.pt").exists()
    assert payload["data"]["n_train"] == 32


def test_train_native_pna_smoke(split, labels, tmp_path):
    import r42_common as C
    import r42_native_train as NT

    payload = NT.train_native(
        kind="pna",
        cfg=C.load_config()["models"]["pna"],
        split=split,
        labels=labels,
        device=torch.device("cpu"),
        run_dir=tmp_path,
        limit_train=32,
        limit_valid=16,
        max_epochs=2,
        label="test-pna",
    )
    assert payload["training"]["checkpoint_rule"] == "best_validation_loss"
    assert payload["training"]["best_epoch"] in (1, 2)
    assert payload["env"]["optimizer"] == "torch.optim.AdamW"
    assert payload["test_labels_read"] is False


def test_train_native_esa_smoke(split, labels, tmp_path):
    import r42_common as C
    import r42_native_train as NT

    payload = NT.train_native(
        kind="esa",
        cfg=C.load_config()["models"]["esa"],
        split=split,
        labels=labels,
        device=torch.device("cpu"),
        run_dir=tmp_path,
        limit_train=16,
        limit_valid=8,
        max_epochs=1,
        label="test-esa",
        precision_override="fp32",
    )
    assert payload["metrics"]["valid"]["units"] == "raw_logD"
    assert payload["env"]["attention_backend"] == "torch"
    assert payload["test_labels_read"] is False
