from __future__ import annotations

import numpy as np

import lcrc_common as C
import lcrc_data as LD


def test_sample_cache_validation(spec, sample_cache):
    report = LD.validate_cache(sample_cache, spec)
    assert report["finite"] is True
    assert report["index_ranges_valid"] is True
    assert report["labels_present"] is False
    assert report["n_molecules_checked"] == int(sample_cache["n_molecules"])
    assert np.asarray(sample_cache["path_cat"]).dtype.itemsize >= 4
    assert np.asarray(sample_cache["nbr_code"]).dtype.itemsize >= 4


def test_cache_determinism_and_metadata(spec, sample_graphs):
    a = LD.build_cache(sample_graphs, spec, split="fit_sample")
    b = LD.build_cache(sample_graphs, spec, split="fit_sample")
    assert C.content_fingerprint(a) == C.content_fingerprint(b)
    assert a["meta"]["labels_present"] is False
    assert a["meta"]["feature_spec_hash"] == spec.hash()
    assert a["meta"]["n_molecules"] == len(sample_graphs)


def test_save_load_roundtrip(tmp_path, sample_cache):
    path = tmp_path / "sample.pt"
    sha = LD.save_cache(sample_cache, path)
    loaded = LD.load_cache(path)
    assert C.sha256_file(path) == sha
    assert C.content_fingerprint(loaded) == C.content_fingerprint(sample_cache)
    assert loaded["meta"]["labels_present"] is False


def test_fit_norm_metadata_and_finite_values(spec, sample_cache):
    norm = LD.fit_norm(sample_cache, spec, source="fit_sample_only")
    assert norm["fit_split"] == "fit_sample"
    assert norm["valid_or_test_used"] is False
    assert norm["source"] == "fit_sample_only"
    assert norm["zero_variance_policy"] == "constant_fit_features_use_scale_1.0"
    for key in ("global_mean", "global_scale", "scalar_mean", "scalar_scale"):
        arr = np.asarray(norm[key])
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
    assert (np.asarray(norm["global_scale"]) > 0).all()
    assert (np.asarray(norm["scalar_scale"]) > 0).all()
    assert LD.norm_fingerprint(norm) == LD.norm_fingerprint(dict(norm))


def test_cache_has_no_label_keys(tmp_path, sample_cache):
    for key in sample_cache:
        assert "label" not in key.lower()
        assert key.lower() != "exp"
    path = tmp_path / "sample.pt"
    LD.save_cache(sample_cache, path)
    loaded = LD.load_cache(path)
    assert "labels" not in loaded
    assert "exp" not in loaded
