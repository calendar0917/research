"""Model-fidelity tests: the three neural models are the frozen/upstream ones."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _batch(structures, rows):
    from torch_geometric.data import Batch

    return Batch.from_data_list([structures[i] for i in rows])


def test_lcrc_parameter_count_and_mechanism(cfg, split, labels):
    import r42_common as C
    import r42_data as D
    import lcrc_data
    import lcrc_model

    views = D.build_lcrc_views(split, labels)
    torch.manual_seed(0)
    model = lcrc_model.LCRC(views["spec"], views["train_norm"], lcrc_model.DEFAULT_CONFIG)
    assert int(sum(p.numel() for p in model.parameters())) == 247106
    names = [n for n, _ in model.named_modules()]
    assert not any("attention" in n for n in names)
    assert not any("junction" in n for n in names)
    assert not any("relation" in n for n in names)
    assert model.local.__class__.__name__ == "LocalContent"
    batch = lcrc_data.collate(views["train_cache"], list(range(8)))
    out = model(batch, return_aux=True)
    y, aux = out
    assert y.shape == (8,)
    assert aux["contrib"].shape == (8,) and aux["rho"].shape == (8,)
    assert "delta" in aux
    assert aux["h"].shape[1] == 64  # width


def test_lcrc_r3_recipe_constants(cfg):
    lcrc = cfg["models"]["lcrc"]
    assert lcrc["seed"] == 0
    assert lcrc["epochs"] == 360
    assert lcrc["batch_size"] == 128
    assert lcrc["peak_lr"] == 3e-4
    assert lcrc["min_lr"] == 1e-5
    assert lcrc["weight_decay"] == 1e-3
    assert lcrc["dropout"] == 0.10
    assert lcrc["gradient_clip"] == 5.0
    assert lcrc["scheduler"] == "per_step_cosine"
    assert lcrc["checkpoint"] == "average_last_5_epoch_end_state_dicts"
    assert lcrc["early_stopping"] is False
    assert lcrc["ssl"] is False


def test_pna_is_upstream_class_with_frozen_config(cfg, structures):
    import r42_common as C
    import r42_native_train as NT

    run_dir = C.RUNS / "test_pna_build"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = structures[:16]
    model, degree_info = NT.build_pna(cfg["models"]["pna"], torch.device("cpu"), run_dir, data, 79, 13)
    assert type(model).__module__ == "gnn.graph_models"
    assert type(model).__name__ == "Estimator"
    pna = model.gnn_model
    assert type(pna).__module__ == "gnn.graph_models" and type(pna).__name__ == "PNA"
    conv = pna.convs[0]
    assert type(conv).__module__ == "gnn.graph_models" and type(conv).__name__ == "PNAConvSC"
    layer = conv.gnn_layer
    # PyG stores the aggregator/scaler spec on the DegreeScalerAggregation module.
    assert type(layer.aggr_module).__name__ == "DegreeScalerAggregation"
    aggr_names = [type(a).__name__ for a in layer.aggr_module.aggr.aggrs]
    assert aggr_names == ["MeanAggregation", "MinAggregation", "MaxAggregation", "StdAggregation"]
    pna_cfg = cfg["models"]["pna"]
    assert list(layer.aggr_module.scaler) == list(pna_cfg["scalers"])
    assert layer.towers == int(pna_cfg["towers"])
    assert layer.divide_input is False
    assert int(layer.edge_dim) == 13
    assert int(pna.intermediate_dim) % 5 == 0 and int(pna.out_channels) % 5 == 0
    assert degree_info["loaded_by_model_verified"] is True
    # Each upstream layer block is (residual conv, BatchRenorm, Mish).
    n_layers = int(cfg["models"]["pna"]["num_layers"])
    assert len(pna.convs) == 3 * n_layers
    assert [type(m).__name__ for m in pna.convs][0::3] == ["PNAConvSC"] * n_layers
    assert [type(m).__name__ for m in pna.convs][1::3] == ["BN"] * n_layers
    assert [type(m).__name__ for m in pna.convs][2::3] == ["Mish"] * n_layers
    assert type(pna).__name__ == "PNA" and pna.original_out_channels == cfg["models"]["pna"]["output_node_dim"]
    assert int(pna.intermediate_dim) == 255 and int(pna.out_channels) == 255  # 256 -> nearest multiple of 5
    batch8 = _batch(structures, range(8))
    out = model(batch8.x, batch8.edge_index, batch8.batch, batch8.edge_attr)
    assert out[2].shape == (8,)
    assert out[1].shape == (8, model.output_intermediate_dim)
    assert model.output_intermediate_dim == 3 * pna.original_out_channels


def test_esa_is_upstream_class_with_frozen_stack(cfg, structures):
    import r42_common as C
    import r42_native_train as NT

    run_dir = C.RUNS / "test_esa_build"
    run_dir.mkdir(parents=True, exist_ok=True)
    model, extra = NT.build_esa(cfg["models"]["esa"], torch.device("cpu"), run_dir, 438, 79, 13)
    assert type(model).__module__ == "esa.models"
    assert type(model.st_fast).__module__ == "esa.masked_layers"
    assert list(model.st_fast.layer_types) == ["S", "S", "M", "M", "M", "S", "P"]
    assert list(model.st_fast.dim_hidden) == [256] * 7
    assert list(model.st_fast.num_heads) == [8] * 7
    assert model.st_fast.node_or_edge == "edge"
    assert model.st_fast.pre_or_post == "post"
    assert model.graph_dim == 512
    assert model.output_mlp.in_features == 512
    # PMA seeding: 32 outputs.
    pma = model.st_fast.decoder[0].pma
    assert int(pma.S.shape[1]) == 32
    assert type(pma).__module__ == "esa.mha"
    batch = _batch(structures, range(8))
    num_max_items = int(__import__("esa.models", fromlist=["x"]).nearest_multiple_of_8(438 + 1))
    pred = model(batch.x.float(), batch.edge_index, batch.batch, batch.edge_attr.float(), num_max_items, batch)
    assert pred.shape == (8,)
    assert extra["attention_backend"] == "torch"


def test_esa_attention_backend_is_torch_and_no_posenc(cfg):
    esa = cfg["models"]["esa"]
    assert esa["attention_backend"] == "torch"
    assert esa["positional_encoding"] == "none"
    assert esa["external_pretraining"] is False
    assert esa["apply_attention_on"] == "edge"
    assert esa["mlp_type"] == "gated_mlp"
    assert esa["mlp_layers"] == 2
    assert esa["pma_seeds"] == 32


def test_upstream_vendored_files_are_byte_identical_to_commit():
    import r42_common as C
    import r42_upstream as U

    hashes = U.upstream_file_hashes()
    assert "esa/models.py" in hashes and "gnn/graph_models.py" in hashes
    provenance = C.read_json(C.ANALYSIS / "upstream_provenance.json") if (C.ANALYSIS / "upstream_provenance.json").exists() else None
    if provenance is not None:
        for rel, digest in provenance["files"].items():
            assert hashes[rel] == digest
