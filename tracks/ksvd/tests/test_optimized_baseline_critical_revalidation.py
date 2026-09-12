"""Tests for the optimized-baseline critical revalidation audit.

These tests enforce the *protocol discipline* of the revalidation, not any
numeric outcome:

1.  every new variant uses the frozen optimized training protocol;
2.  Part A keeps the historical compact-v2 architecture except topology;
3.  Part B switches only tokenizer-related representation;
4.  the tokenizer version is explicit in configs and cache fingerprints;
5.  the seed0 decision gate blocks the seed1 it does not need;
6.  seed2/seed3 cannot be scheduled by this experiment;
7.  official test cannot be loaded;
8.  historical / optimized reference records carry the correct fingerprint;
9.  the serial deterministic execution policy is recorded;
10. parameter counts come from the real model, not a hardcoded constant.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from tracks.ksvd.experiments.luyin16 import (
    zinc_optimized_baseline_critical_revalidation as m,
)
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (
    TYPED_TOKENIZER_V1_HISTORICAL,
    TYPED_TOKENIZER_V2_CORRECTED,
    typed_tokenizer_fingerprint,
)

PROTOCOL_KEYS = (
    "epochs",
    "patience",
    "batch_size",
    "learning_rate",
    "weight_decay",
)


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _strip_protocol(model: dict) -> dict:
    return {key: value for key, value in model.items() if key not in PROTOCOL_KEYS}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Test 1 -- every new variant uses the frozen optimized protocol
# ---------------------------------------------------------------------------


def test_new_variants_use_frozen_optimized_protocol() -> None:
    runs = sorted((m.RESULTS_DIR / "runs").glob("*.json"))
    assert runs, "no revalidation runs found"
    for path in runs:
        run = _load(path)
        assert run["protocol"] == m.OPTIMIZED_PROTOCOL
        assert run["protocol"]["max_epochs"] == 240
        assert run["protocol"]["patience"] == 40
        assert run["protocol"]["batch_size"] == 128
        assert run["protocol"]["learning_rate"] == 1.0e-3
        assert run["protocol"]["weight_decay"] == 1.0e-5
        assert run["protocol"]["scheduler"] == "none"
        assert run["protocol"]["loss"].startswith("L1")
    # the two configs also carry the optimized horizon / patience
    for path in (m.V2_CONFIG_PATH, m.CORR_CONFIG_PATH):
        config = _config(path)
        assert config["model"]["epochs"] == 240
        assert config["model"]["patience"] == 40
        assert config["model"]["learning_rate"] == 1.0e-3
        assert config["model"]["weight_decay"] == 1.0e-5


# ---------------------------------------------------------------------------
# Test 2 -- Part A keeps the historical compact-v2 architecture except topology
# ---------------------------------------------------------------------------


def test_part_a_matches_historical_v2_except_topology() -> None:
    new = _config(m.V2_CONFIG_PATH)
    old = _config(m.HISTORICAL_V2_CONFIG_PATH)
    # tokenizer pinned explicitly (new) and equal to historical default (old)
    assert new["representation"]["typed_tokenizer_version"] == TYPED_TOKENIZER_V1_HISTORICAL
    assert "typed_tokenizer_version" not in old["representation"]
    new_rep = dict(new["representation"])
    new_rep.pop("typed_tokenizer_version")
    assert new_rep == old["representation"]
    # model: only protocol fields move; no topology channel is added
    assert _strip_protocol(new["model"]) == _strip_protocol(old["model"])
    assert "topology_mode" not in new["model"]
    assert new["protocol_id"] == old["protocol_id"]
    run = _load(m.RESULTS_DIR / "runs/v2_hist_seed0.json")
    assert run["topology_mode"] == "none"


# ---------------------------------------------------------------------------
# Test 3 -- Part B switches only tokenizer-related representation
# ---------------------------------------------------------------------------


def test_part_b_switches_only_tokenizer() -> None:
    corrected = _config(m.CORR_CONFIG_PATH)
    historical_corrected = _config(m.HISTORICAL_CORR_CONFIG_PATH)
    # compared with its own historical corrected-token config: only protocol moves
    assert _strip_protocol(corrected["model"]) == _strip_protocol(
        historical_corrected["model"]
    )
    assert corrected["representation"] == historical_corrected["representation"]
    assert corrected["representation"]["typed_tokenizer_version"] == TYPED_TOKENIZER_V2_CORRECTED
    # compared with the non-corrected v4 config: only tokenizer + budget + protocol
    plain_v4 = _config(m.HISTORICAL_V4_CONFIG_PATH)
    model_delta = {
        key: (plain_v4["model"].get(key), corrected["model"].get(key))
        for key in set(plain_v4["model"]) | set(corrected["model"])
        if plain_v4["model"].get(key) != corrected["model"].get(key)
    }
    # only the parameter budget and training-protocol fields may move
    assert set(model_delta) <= {"expected_max_trainable_params", *PROTOCOL_KEYS}
    assert "expected_max_trainable_params" in model_delta
    rep_delta = {
        key
        for key in set(plain_v4["representation"]) | set(corrected["representation"])
        if plain_v4["representation"].get(key) != corrected["representation"].get(key)
    }
    assert rep_delta == {"typed_tokenizer_version"}


# ---------------------------------------------------------------------------
# Test 4 -- tokenizer version explicit in config + cache fingerprint
# ---------------------------------------------------------------------------


def test_tokenizer_version_explicit_and_in_cache_fingerprint() -> None:
    for tag, expected in (
        ("v2_hist", TYPED_TOKENIZER_V1_HISTORICAL),
        ("v4_corr", TYPED_TOKENIZER_V2_CORRECTED),
    ):
        _, _, meta = m._cache_paths(tag)
        payload = _load(meta)
        assert payload["typed_tokenizer_version"] == expected
        assert payload["typed_tokenizer_fingerprint"] == typed_tokenizer_fingerprint(
            expected, m.PATCH_RADIUS
        )
    assert _config(m.V2_CONFIG_PATH)["representation"]["typed_tokenizer_version"] == (
        TYPED_TOKENIZER_V1_HISTORICAL
    )
    assert _config(m.CORR_CONFIG_PATH)["representation"]["typed_tokenizer_version"] == (
        TYPED_TOKENIZER_V2_CORRECTED
    )
    # a mismatched-cache guard exists
    assert "cache tokenizer mismatch" in inspect.getsource(m.extract_records)


# ---------------------------------------------------------------------------
# Test 5 -- the seed0 gate blocks the seed1 it does not need
# ---------------------------------------------------------------------------


def test_seed0_gate_blocks_unnecessary_seed1() -> None:
    a_decision = _load(m.RESULTS_DIR / "part_a_decision.json")
    a_seed0 = _load(m.RESULTS_DIR / "part_a_v2_seed0.json")
    delta = a_seed0["delta_topo_v2_minus_v4"]
    assert delta >= m.A_STRONG
    assert a_decision["verdict"] == "TOPO_STRONG_SURVIVES_SEED0"
    assert a_decision["need_seed1"] is False
    with pytest.raises(RuntimeError):
        m.seed1_a()

    b_decision = _load(m.RESULTS_DIR / "part_b_decision.json")
    b_seed0 = _load(m.RESULTS_DIR / "part_b_corrected_seed0.json")
    assert b_seed0["corrected_minus_historical"] >= m.B_CLEAR_NOGO
    assert b_decision["verdict"] == "CORRECTED_CLEAR_NO_GO"
    assert b_decision["need_seed1"] is False
    with pytest.raises(RuntimeError):
        m.seed1_b()


# ---------------------------------------------------------------------------
# Test 6 -- seed2 / seed3 cannot be scheduled
# ---------------------------------------------------------------------------


def test_seed2_seed3_not_schedulable() -> None:
    main_source = inspect.getsource(m.main)
    assert "seed2" not in main_source
    assert "seed3" not in main_source
    assert "seed1_a" in main_source and "seed1_b" in main_source
    # only seed 0/1 can ever produce a run record
    for path in (m.RESULTS_DIR / "runs").glob("*.json"):
        assert int(_load(path)["seed"]) in (0, 1)


# ---------------------------------------------------------------------------
# Test 7 -- official test cannot be loaded
# ---------------------------------------------------------------------------


def test_official_test_not_loaded() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert "extract_test_records" not in source
    assert '_load_zinc(ZINC_ROOT, "test")' not in source
    # extract_records only ever loads the train and val splits
    body = source.split("def extract_records")[1].split("def build_encoded")[0]
    assert '_load_zinc(ZINC_ROOT, "train")' in body
    assert '_load_zinc(ZINC_ROOT, "val")' in body
    assert '"test"' not in body
    for path in (m.RESULTS_DIR / "runs").glob("*.json"):
        assert _load(path)["official_test_loaded"] is False
    # no test artifact is produced
    assert not (m.RESULTS_DIR / "test_results.csv").exists()


# ---------------------------------------------------------------------------
# Test 8 -- historical / optimized reference fingerprints are correct
# ---------------------------------------------------------------------------


def test_reference_fingerprints_correct() -> None:
    inv = _load(m.RESULTS_DIR / "checkpoint_inventory.json")
    ref = inv["optimized_v4_reference"]["optimized_v4_seed0"]
    assert ref["verified"] is True
    assert ref["best_valid_mae"] == pytest.approx(m.OPTIMIZED_V4_SEED0_VALID)
    assert ref["best_epoch"] == m.OPTIMIZED_V4_SEED0_EPOCH
    assert ref["parameters"] == m.OPTIMIZED_V4_PARAMS
    assert ref["typed_tokenizer_version"] == TYPED_TOKENIZER_V1_HISTORICAL
    assert ref["typed_tokenizer_fingerprint"] == typed_tokenizer_fingerprint(
        TYPED_TOKENIZER_V1_HISTORICAL, m.PATCH_RADIUS
    )
    # matches the sufficiency protocol lock tokenizer fingerprint
    lock = _load(m.CANONICAL_SUFFICIENCY_DIR / "final_training_protocol_lock.json")
    assert lock["tokenizer_fingerprint"] == typed_tokenizer_fingerprint(
        TYPED_TOKENIZER_V1_HISTORICAL, m.PATCH_RADIUS
    )
    assert lock["model_parameter_count"] == m.OPTIMIZED_V4_PARAMS


# ---------------------------------------------------------------------------
# Test 9 -- serial deterministic execution policy recorded
# ---------------------------------------------------------------------------


def test_serial_deterministic_policy_recorded() -> None:
    lock = _load(m.RESULTS_DIR / "protocol_lock.json")
    det = lock["determinism"]
    assert det["serial_execution"] is True
    assert det["fixed_global_rng"] is True
    assert det["torch_threads"] == 4
    run = _load(m.RESULTS_DIR / "runs/v2_hist_seed0.json")
    assert run["environment"]["configured_torch_threads"] == 4
    assert run["environment"]["device"] == "cpu"
    assert "environment" in inspect.getsource(m.train_variant)
    assert lock["official_test_policy"]["loaded"] is False


# ---------------------------------------------------------------------------
# Test 10 -- parameter counts come from the real model
# ---------------------------------------------------------------------------


def test_parameter_counts_from_real_model() -> None:
    for tag, expected in (("v2_hist", 98549), ("v4_corr", 107201)):
        run = _load(m.RESULTS_DIR / f"runs/{tag}_seed0.json")
        state_path = Path(run["state_path"])
        assert state_path.exists()
        # count the real tensors stored in the checkpoint; do NOT trust a literal
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        real = int(sum(value.numel() for value in state.values()))
        assert real == expected
        assert int(run["parameters"]) == real
    # the training code derives the count from the model, not a constant
    source = inspect.getsource(m.train_variant)
    assert 'int(phase["parameters"])' in source
