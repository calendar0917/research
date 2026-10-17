"""Fast CPU tests for protocol ``cscl-p3-slim-v1``.

No ZINC, no labels, no GPU.  These tests mirror the hard G1-G8 gates and the
frozen R3 loop equivalence check.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import p3_slim_v1 as sl  # noqa: E402
import run_p3_slim_v1 as driver  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402


def _norm():
    return {
        "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    }


def test_config_matches_frozen_constants():
    with (_CODE.parent / "configs" / "p3_slim_v1.yaml").open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["protocol"] == sl.PROTOCOL_ID
    assert cfg["split_tag"] == sl.SPLIT_TAG
    assert cfg["salt"] == sl.SPLIT_SALT
    assert tuple(cfg["seeds"]) == sl.SEEDS
    assert tuple(cfg["arms"]) == sl.ARMS
    assert cfg["model"]["p3_reader_hidden"] == 256
    assert cfg["model"]["slim_reader_hidden"] == 128
    assert cfg["model"]["p3_parameters"] == 413_505
    assert cfg["model"]["slim_parameters"] == 322_881
    assert cfg["model"]["p3_reader_parameters"] == 148_481
    assert cfg["model"]["slim_reader_parameters"] == 57_857
    assert cfg["model"]["shared_parameters"] == 265_024
    assert cfg["recipe"]["key"] == "R3"
    assert cfg["recipe"]["epochs"] == bp.RECIPES["R3"]["epochs"]
    assert cfg["recipe"]["batch_size"] == bp.BATCH_SIZE
    assert cfg["ssl"]["updates"] == bp.PRETRAIN_UPDATES
    assert cfg["gate"]["meaningful_delta"] == sl.MEANINGFUL_DELTA
    assert cfg["budget"]["gpu_hours"] == sl.GPU_BUDGET_CAP_HOURS


def test_parameter_inventory_and_actual_counts():
    inv = sl.parameter_inventory()
    assert inv["all_expected"]
    p3, slim = sl.build_pair(0, _norm(), "cpu")
    assert sl.model_parameters(p3) == 413_505
    assert sl.model_parameters(slim) == 322_881
    assert sum(p.numel() for p in p3.reader.parameters()) == 148_481
    assert sum(p.numel() for p in slim.reader.parameters()) == 57_857
    assert inv["reader_reduction"] == 90_624
    assert inv["shared_parameters"] == 265_024


def test_shared_modules_bitwise_equal_reader_distinct():
    p3, slim = sl.build_pair(0, _norm(), "cpu")
    assert sl.shared_hash(p3) == sl.shared_hash(slim)
    assert sl.encoder_hash(p3) == sl.encoder_hash(slim)
    assert sl.reader_hash(p3) != sl.reader_hash(slim)
    assert sl.full_hash(p3) != sl.full_hash(slim)
    for name in sl.SHARED_MODULE_NAMES:
        assert sl.module_state_hash(getattr(p3, name)) == sl.module_state_hash(getattr(slim, name))
    assert p3.config.reader_in == slim.config.reader_in == 321
    assert slim.config.reader_hidden == 128


def test_reader_input_identical_before_training():
    p3, slim = sl.build_pair(0, _norm(), "cpu")
    objs = sl.synthetic_objects()[:3]
    batch = sro.batch_objects(objs)
    with torch.no_grad():
        f1 = p3.forward_features(batch)
        f2 = slim.forward_features(batch)
    assert tuple(f1["reader_in"].shape) == (3, 321)
    assert tuple(f2["reader_in"].shape) == (3, 321)
    assert torch.equal(f1["reader_in"], f2["reader_in"])
    assert float((f1["reader_in"] - f2["reader_in"]).abs().max()) == 0.0


def test_new_loop_p3_matches_frozen_trainer():
    norm = _norm()
    objs = sl.synthetic_objects()
    y = np.asarray([0.5, -0.3, 0.1, 0.9, -0.7, 0.2, 0.4, -0.1], dtype=np.float64)
    frozen = bp.train_supervised(
        arm="P3",
        seed=125,
        recipe_key="R3",
        init="scratch",
        train_objs=objs,
        rows=[o.mol_index for o in objs],
        y_fit=y,
        norm=norm,
        device=torch.device("cpu"),
        smoke_epochs=2,
        log_fn=lambda _m: None,
    )
    model = sl.build_p3(125, norm, "cpu")
    new = sl.train_supervised_model(
        arm="P3",
        seed=125,
        model=model,
        train_objs=objs,
        y_fit=y,
        device=torch.device("cpu"),
        epochs=2,
        log_fn=lambda _m: None,
    )
    diff = max(float((frozen["soup_state"][k] - new["soup_state"][k]).abs().max()) for k in frozen["soup_state"])
    assert diff <= 1e-7
    assert frozen["batch_order"]["full_hash"] == new["batch_order"]["full_hash"]
    assert frozen["updates"] == new["updates"] == 2


def test_last5_soup_and_batch_order():
    norm = _norm()
    objs = sl.synthetic_objects()
    y = np.asarray([0.5, -0.3, 0.1, 0.9, -0.7, 0.2, 0.4, -0.1], dtype=np.float64)
    model = sl.build_p3(126, norm, "cpu")
    info = sl.train_supervised_model(
        arm="P3",
        seed=126,
        model=model,
        train_objs=objs,
        y_fit=y,
        device=torch.device("cpu"),
        epochs=6,
        collect_epoch_states=True,
        log_fn=lambda _m: None,
    )
    states = info.pop("epoch_states")
    assert info["soup_members"] == [2, 3, 4, 5, 6]
    expected = {k: torch.stack([states[i - 1][k].float() for i in (2, 3, 4, 5, 6)]).mean(0) for k in states[0]}
    assert max(float((info["soup_state"][k] - expected[k]).abs().max()) for k in expected) <= 1e-7
    assert info["updates"] == 6
    assert info["batch_order"]["full_matches"]


def test_gate_arithmetic_requires_all_seven_conditions():
    good = {
        "delta_seed_mean": -0.005,
        "per_seed_delta": {"0": -0.004, "1": -0.006},
        "bootstrap": {"ci_hi": -0.001},
        "ordinary_delta": -0.004,
        "top10_removed_delta": -0.003,
        "rmse_new": 0.180,
        "rmse_ref": 0.183,
        "tail_delta": 0.001,
    }
    assert driver._gate(good)["passed"]
    bad = dict(good)
    bad["per_seed_delta"] = {"0": -0.004, "1": 0.001}
    assert not driver._gate(bad)["passed"]
    bad = dict(good)
    bad["tail_delta"] = 0.006
    assert not driver._gate(bad)["passed"]
    bad = dict(good)
    bad["top10_removed_delta"] = 0.0
    assert not driver._gate(bad)["passed"]


def test_fit_update_budget_for_7996():
    assert math.ceil(7996 / bp.BATCH_SIZE) * bp.RECIPES["R3"]["epochs"] == 22_680
    assert bp.batch_order_hash(0, 7996, bp.BATCH_SIZE, 360) != bp.batch_order_hash(1, 7996, bp.BATCH_SIZE, 360)
