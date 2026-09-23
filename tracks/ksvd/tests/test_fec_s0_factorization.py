"""Targeted tests for FEC-S0 (factorized environment-composition equivalence).

These tests are an audit, not a training run: they load the historical
strict-static S0 checkpoint and verify that the factorized feature
construction reproduces the historical inputs, intermediate states and
predictions.  Official test is never loaded.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from tracks.ksvd.experiments.luyin16.fec_s0_factorization import (
    TOL_INTERMEDIATE,
    TOL_PREDICTION,
    TOL_RAW,
    TOL_STANDARDIZED,
    FactorizedFeatureTransform,
    _capture,
    _ensure_batch,
    build_fits,
    factorized_record,
    load_s0_model,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import _extract_v4_records
from tracks.ksvd.experiments.luyin16.zinc_static_dictionary_pair import (
    load_encoded,
    static_contract_checks,
)

_SETUP: dict | None = None


def _setup() -> dict:
    global _SETUP
    if _SETUP is None:
        train_records, valid_records = _extract_v4_records()
        _train_encoded, valid_encoded, _audit = load_encoded()
        fits = build_fits(train_records)
        valid_raw = list(_load_zinc(__import__("pathlib").Path("data/ZINC"), "val"))[:4]
        records = list(valid_records)[:4]
        encoded = list(valid_encoded)[:4]
        factorized = [factorized_record(data) for data in valid_raw]
        _SETUP = {
            "fits": fits,
            "transform": FactorizedFeatureTransform(fits),
            "model": load_s0_model(),
            "valid_raw": valid_raw,
            "records": records,
            "encoded": encoded,
            "factorized": factorized,
        }
    return _SETUP


def test_factorized_shell_descriptor_bit_identical() -> None:
    setup = _setup()
    for record, fact in zip(setup["records"], setup["factorized"]):
        historical = np.stack([patch.shell_descriptor for patch in record.patches])
        assert historical.shape == fact.patch_cont_raw.shape
        assert np.array_equal(historical, fact.patch_cont_raw)
        assert float(np.abs(historical - fact.patch_cont_raw).max()) <= TOL_RAW


def test_factorized_pair_relation_bit_identical() -> None:
    setup = _setup()
    for record, fact in zip(setup["records"], setup["factorized"]):
        assert np.array_equal(record.pair_relation, fact.pair_relation_raw)
        assert float(np.abs(record.pair_relation - fact.pair_relation_raw).max()) <= TOL_RAW


def test_factorized_global_context_bit_identical() -> None:
    setup = _setup()
    for record, fact in zip(setup["records"], setup["factorized"]):
        assert np.array_equal(record.global_context, fact.global_raw)


def test_typed_key_reconstructs_historical_certificate() -> None:
    setup = _setup()
    for record, fact in zip(setup["records"], setup["factorized"]):
        assert tuple(p.typed_certificate for p in record.patches) == fact.typed_keys
        assert tuple(p.parent_certificate for p in record.patches) == fact.parent_keys


def test_transform_matches_cached_encoded_inputs() -> None:
    setup = _setup()
    transform: FactorizedFeatureTransform = setup["transform"]
    for raw, record, encoded, fact in zip(
        setup["valid_raw"], setup["records"], setup["encoded"], setup["factorized"]
    ):
        data = transform.build(
            raw,
            float(raw.y.view(-1)[0]),
            topology_raw=record.topology_features,
            raw=fact,
        )
        assert torch.allclose(encoded.patch_cont, data.patch_cont, atol=TOL_STANDARDIZED)
        assert torch.equal(encoded.patch_cont, data.patch_cont)
        assert torch.allclose(encoded.global_context, data.global_context, atol=TOL_STANDARDIZED)
        assert torch.equal(encoded.typed_token, data.typed_token)
        assert torch.equal(encoded.parent_token, data.parent_token)
        assert torch.allclose(encoded.topology_features, data.topology_features, atol=TOL_STANDARDIZED)


def test_forward_and_prediction_bit_identical() -> None:
    setup = _setup()
    transform: FactorizedFeatureTransform = setup["transform"]
    model = setup["model"]
    for raw, record, encoded, fact in zip(
        setup["valid_raw"], setup["records"], setup["encoded"], setup["factorized"]
    ):
        data = transform.build(
            raw,
            float(raw.y.view(-1)[0]),
            topology_raw=record.topology_features,
            raw=fact,
        )
        historical = _capture(model, _ensure_batch(encoded))
        candidate = _capture(model, data)
        for name in ("patch", "pair_value", "relation", "graph_hidden", "R"):
            assert torch.equal(historical[name], candidate[name]), name
        assert torch.allclose(historical["pred"], candidate["pred"], atol=TOL_PREDICTION)
        assert float((historical["pred"] - candidate["pred"]).abs().max()) <= TOL_INTERMEDIATE


def test_purity_contract_no_pair_to_centre_no_writeback() -> None:
    setup = _setup()
    transform: FactorizedFeatureTransform = setup["transform"]
    model = setup["model"]
    data = []
    for raw, record, fact in zip(setup["valid_raw"], setup["records"], setup["factorized"]):
        data.append(
            transform.build(
                raw,
                float(raw.y.view(-1)[0]),
                topology_raw=record.topology_features,
                raw=fact,
            )
        )
    batch = Batch.from_data_list(data)
    contract = static_contract_checks(model, batch)
    assert contract["center_update_is_none"]
    assert contract["forward_without_pair_to_center"]
    assert contract["pair_encoder_calls_per_forward"] == 1
    assert contract["h_identical_under_relation_mutation"]
    assert contract["passed"]
