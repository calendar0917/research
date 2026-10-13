"""Fast correctness tests for cscl-bjoc-performance-mechanism-long-v1.

No ZINC data, no checkpoints, no GPU.  Covers the frozen recipe / arm
registry, the schedule and batch-order identity with the recipe round, the
group-aware split builder, the C1 parameter budget and zero-column identity,
C1 permutation / endpoint / batch invariance on synthetic molecules, and the
no-official-test discipline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402
import run_bjoc_performance_mechanism_long_v1 as drv  # noqa: E402
import run_bjoc_recipe_tuning_v1 as rt  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

ZERO_MEAN = np.zeros(sro.GLOBAL_DIM, dtype=np.float32)
ONE_SCALE = np.ones(sro.GLOBAL_DIM, dtype=np.float32)
NORM = {"global_mean": ZERO_MEAN, "global_scale": ONE_SCALE}


def _cfg():
    return yaml.safe_load(drv.CONFIG_PATH.read_text(encoding="utf-8"))


def _mol(atom_types, bonds):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    edge_index = np.asarray(ei, dtype=np.int64).T
    return sro.build_mol_objects(-1, x, edge_index, np.asarray(ea, dtype=np.int64))


def _small_objects():
    return [
        _mol([0, 1, 2, 3], [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 0)]),
        _mol([4, 5, 6], [(0, 1, 0), (1, 2, 2)]),
        _mol([7, 7, 8, 9, 10], [(0, 1, 0), (1, 2, 0), (2, 3, 1), (3, 4, 0), (4, 1, 0)]),
    ]


# ---------------------------------------------------------------------------
# frozen registry / config agreement
# ---------------------------------------------------------------------------


def test_config_and_arm_registry_agree():
    cfg = _cfg()
    assert cfg["protocol"] == bp.PROTOCOL_ID == drv.PROTOCOL_ID
    assert cfg["official_test"] == "never_read"
    assert cfg["stage_b"]["split_salt"] == drv.SPLIT_SALTS["B"]
    assert cfg["stage_d"]["split_salt"] == drv.SPLIT_SALTS["C"]
    assert set(cfg["stage_b"]["arms"]) == {"B0", "B3", "P0", "P3"}
    assert set(drv.STAGE_ARMS) == {"B0", "B3", "P0", "P3", "A1", "C1"}
    for arm, spec in cfg["stage_b"]["arms"].items():
        assert drv.STAGE_ARMS[arm]["init"] == spec["init"]
        assert drv.STAGE_ARMS[arm]["recipe"] == spec["recipe"]
    assert bp.RECIPES["R0"] == {"epochs": 240, "schedule": "constant", "lr": 1.0e-3, "lr_min": None, "weight_decay": 1.0e-5}
    assert bp.RECIPES["R3"] == {"epochs": 360, "schedule": "cosine", "lr": 1.0e-3, "lr_min": 1.0e-5, "weight_decay": 1.0e-5}
    assert cfg["stage_c"]["model"]["total_parameter_count"] == bp.C1_EXPECTED_PARAMETERS == 438081
    assert cfg["stage_c"]["model"]["new_parameter_count"] == 24576
    assert cfg["frozen"]["expected_parameters"] == 413505
    assert cfg["bans"][0] == "official test access in any form"


def test_gate_thresholds_frozen():
    cfg = _cfg()
    assert cfg["stage_b"]["monitor_gates"]["recipe_effect"] == "P3 - P0 seed-mean <= -0.003 and both per-seed deltas < 0"
    assert cfg["stage_b"]["monitor_gates"]["ssl_effect_r3"] == "P3 - B3 seed-mean <= -0.005 and both per-seed deltas < 0"
    assert cfg["stage_c"]["cpu_gate"]["identity_max_abs_diff_max"] == 1.0e-4
    assert cfg["stage_c"]["gpu_gate"]["seed_0_gate"].startswith("Monitor MAE(C1) - MAE(P3) <= -0.003")
    assert cfg["stage_d"]["purchase_rule"].startswith("run iff Stage-B recipe_effect gate passes")
    assert cfg["budget"]["gpu_hours_max"] == 7.0


# ---------------------------------------------------------------------------
# schedule + batch order identity with the frozen recipe round
# ---------------------------------------------------------------------------


def test_cosine_matches_recipe_round_function():
    for step in (0, 1, 123, 28439, 28440):
        assert bp.cosine_lr(step, 28440, 1.0e-3, 1.0e-5) == rt.cosine_lr(step, 28440, 1.0e-3, 1.0e-5)


def test_schedule_lr_constant_and_cosine():
    assert bp.schedule_lr(bp.RECIPES["R0"], 100, 18960) == 1.0e-3
    assert abs(bp.schedule_lr(bp.RECIPES["R3"], 0, 28440) - 1.0e-3) < 1e-15
    assert abs(bp.schedule_lr(bp.RECIPES["R3"], 28440, 28440) - 1.0e-5) < 1e-12


def test_batch_order_hash_matches_full10k_round():
    for seed in (0, 1):
        assert bp.batch_order_hash(seed, 10000, 128, 360) == f10k.batch_order_hash(seed, 10000, 128, 360)
    # the 8k prefix matches the frozen masked-pretraining round generator rule
    assert bp.batch_order_hash(0, 8000, 128, 240) == f10k.batch_order_hash(0, 8000, 128, 240)


# ---------------------------------------------------------------------------
# split builder
# ---------------------------------------------------------------------------


def test_split_builder_is_deterministic_and_group_safe():
    smiles = [f"C{i % 37}" for i in range(500)] + ["C0", "C1", "C2"]
    a = bp.build_split(smiles, "salt-x", fit_target=300, monitor_target=50)
    b = bp.build_split(smiles, "salt-x", fit_target=300, monitor_target=50)
    for key in ("fit", "monitor", "locked"):
        assert np.array_equal(a[key], b[key])
    union = set(map(int, a["fit"])) | set(map(int, a["monitor"])) | set(map(int, a["locked"]))
    assert union == set(range(len(smiles)))
    assert a["duplicate_groups_cross_sides"] == 0
    c = bp.build_split(smiles, "salt-y", fit_target=300, monitor_target=50)
    assert not np.array_equal(a["fit"], c["fit"])


# ---------------------------------------------------------------------------
# C1 model
# ---------------------------------------------------------------------------


def test_c1_parameter_budget_and_identity():
    base, c1, info = bp.build_c1_from_init(0, NORM, "cpu", None)
    assert sum(p.numel() for p in base.parameters()) == bp.EXPECTED_PARAMETERS == 413505
    assert sum(p.numel() for p in c1.parameters()) == bp.C1_EXPECTED_PARAMETERS == 438081
    assert info["c1_new_parameters"] == 24576
    batch = sro.batch_objects(_small_objects())
    with torch.no_grad():
        yb = base(batch)
        yc = c1(batch)
    assert float((yb - yc).abs().max()) < 1.0e-5
    # new columns are exactly zero
    co = int(src.BJOCConfig().compose_out)
    ao = int(src.BJOCConfig().atom_out)
    w = c1.reader.lins[0].weight.detach()
    assert torch.equal(w[:, co : co + ao], torch.zeros_like(w[:, co : co + ao]))
    # all other parameters equal the base
    for key, value in base.state_dict().items():
        if key == "reader.lins.0.weight":
            continue
        assert torch.equal(value.cpu(), c1.state_dict()[key].cpu()), key


def test_c1_forward_is_permutation_and_endpoint_invariant():
    _, c1, _ = bp.build_c1_from_init(0, NORM, "cpu", None)
    objs = _small_objects()
    batch = sro.batch_objects(objs)
    with torch.no_grad():
        y0 = c1(batch).numpy()
        y_single = np.asarray([c1(sro.batch_objects([o])).item() for o in objs])
    assert np.abs(y0 - y_single).max() < 1e-5

    # atom permutation: relabel raw arrays, rebuild, same prediction
    topo = ([0, 1, 2, 3], [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 0)])
    perm = np.asarray([2, 0, 3, 1])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    x = np.asarray(topo[0], dtype=np.int64).reshape(-1, 1)
    ei, ea = [], []
    for u, v, bt in topo[1]:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    ei = np.asarray(ei, dtype=np.int64).T
    p1 = sro.build_mol_objects(-1, x, ei, np.asarray(ea, dtype=np.int64))
    p2 = sro.build_mol_objects(-1, x[perm], inv[ei], np.asarray(ea, dtype=np.int64))
    with torch.no_grad():
        a = c1(sro.batch_objects([p1])).item()
        b = c1(sro.batch_objects([p2])).item()
    assert abs(a - b) < 1e-5

    # endpoint swap: reverse the doubled directed edge list
    p3 = sro.build_mol_objects(-1, x, ei[::-1], np.asarray(ea, dtype=np.int64))
    with torch.no_grad():
        c = c1(sro.batch_objects([p3])).item()
    assert abs(a - c) < 1e-5


def test_c1_no_forbidden_modules():
    _, c1, _ = bp.build_c1_from_init(0, NORM, "cpu", None)
    assert src.forbidden_module_scan(c1) == []


# ---------------------------------------------------------------------------
# discipline
# ---------------------------------------------------------------------------


def test_driver_has_no_official_test_path():
    text = (Path(_CODE_DIR) / "run_bjoc_performance_mechanism_long_v1.py").read_text(encoding="utf-8")
    for token in ("encoded_test", "load_official_test", '"official_test_loaded": True', "test_read"):
        assert token not in text, token
    assert text.count('"official_test_loaded": False') >= 4


def test_module_pretraining_never_reads_labels():
    text = (Path(_CODE_DIR) / "bjoc_perfmech_long_v1.py").read_text(encoding="utf-8")
    body = text.split("def pretrain_masked", 1)[1].split("\ndef ", 1)[0]
    assert "load_y_rows" not in body
    assert '\"labels_read\": False' in text
