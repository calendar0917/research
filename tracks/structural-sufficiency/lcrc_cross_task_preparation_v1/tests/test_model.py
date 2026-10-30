from __future__ import annotations

import numpy as np
import torch

import cpu_gates as G
import lcrc_data as LD


def test_lcrc_forward_decomposition_and_shapes(spec, sample_cache, sample_norm, sample_model):
    rows = list(range(min(16, int(sample_cache["n_molecules"]))))
    report = G.gate_model_forward(sample_cache, spec, sample_norm, seed=0, rows=rows)
    assert report["status"] == "PASS"
    assert report["checks"]["decomposition"] is True
    assert report["checks"]["contrib_grouping"] is True
    assert report["checks"]["no_relation_head"] is True
    assert report["checks"]["no_junction_module"] is True
    assert report["checks"]["no_attention_module"] is True
    assert report["model_info"]["parameters"] > 0


def test_model_state_dict_has_no_forbidden_modules(sample_model):
    keys = list(sample_model.state_dict())
    assert not any("rel." in k for k in keys)
    assert not any("junction" in k for k in keys)
    assert not any("attention" in k for k in keys)
    assert not any("pair" in k for k in keys)


def test_edge_cases_are_finite_and_decompose(spec, sample_model):
    report = G.gate_edge_cases(spec, sample_model)
    assert report["status"] == "PASS"
    for name, case in report["cases"].items():
        assert case["finite"] is True
        assert case["decomposition_error"] <= 1e-6
        assert case["pass"] is True


def test_label_injection_does_not_change_batch(spec, sample_cache):
    rows = list(range(min(8, int(sample_cache["n_molecules"]))))
    report = G.gate_label_independence(sample_cache, spec, rows=rows)
    assert report["status"] == "PASS"
    assert report["checks"]["dummy_label_injection_ignored"] is True
    assert report["checks"]["no_label_keys_in_batch"] is True
    assert report["checks"]["cache_meta_labels_false"] is True


def test_forward_is_finite_under_no_grad(spec, sample_cache, sample_norm, sample_model):
    rows = list(range(min(8, int(sample_cache["n_molecules"]))))
    batch = LD.collate(sample_cache, rows)
    sample_model.eval()
    with torch.no_grad():
        y, aux = sample_model(batch, return_aux=True)
    assert torch.isfinite(y).all()
    assert torch.isfinite(aux["h"]).all()
    assert torch.max(torch.abs(y - (aux["contrib"] + aux["rho"]))).item() <= 1e-6
    assert tuple(aux["h"].shape) == (int(batch["atom_feat"].shape[0]), int(sample_model.cfg["width"]))
    assert y.shape == (len(rows),)


def test_parameter_count_is_schema_dependent(spec, legacy_spec, sample_norm):
    import lcrc_model as LM

    lipo_model, lipo_info = LM.build_model(spec, sample_norm, seed=0, device="cpu")
    # A legacy norm is dimensionally incompatible with lipo_v1; build a
    # legacy-compatible norm only to exercise the schema-driven width.
    legacy_norm = {
        "global_mean": np.zeros(legacy_spec.global_dim, dtype=np.float32),
        "global_scale": np.ones(legacy_spec.global_dim, dtype=np.float32),
        "scalar_mean": np.zeros(legacy_spec.n_scalars, dtype=np.float32),
        "scalar_scale": np.ones(legacy_spec.n_scalars, dtype=np.float32),
        "source": "test",
    }
    legacy_model, legacy_info = LM.build_model(legacy_spec, legacy_norm, seed=0, device="cpu")
    assert lipo_info["parameters"] != legacy_info["parameters"]
    assert lipo_info["parameters"] == LM.parameter_count(lipo_model)
    assert legacy_info["parameters"] == LM.parameter_count(legacy_model)
