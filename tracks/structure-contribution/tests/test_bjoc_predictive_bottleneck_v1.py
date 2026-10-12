"""Tests for cscl-predictive-bottleneck-diagnosis-v1.

Data-free tests run in a fresh clone on CPU (split/step arithmetic, model
identity, invariance, discipline scans).  Result-dependent checks skip when the
gitignored result JSONs are absent.  Official valid / official test are never
referenced or loaded.

Slow tests (marked ``slow``) load the real object cache / result artefacts.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import run_bjoc_predictive_bottleneck_v1 as drv  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bjoc_predictive_bottleneck_v1.yaml"
RESULTS = ROOT / "results" / "bjoc_predictive_bottleneck_v1"


# ---------------------------------------------------------------------------
# tiny data-free graphs
# ---------------------------------------------------------------------------


def _tiny_objects():
    out = []
    # molecule 0: triangle (ring), atom types 0,1,2, single bonds
    x = np.asarray([[0], [1], [2]], dtype=np.int64)
    ei = np.asarray([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
    ea = np.asarray([1, 1, 1, 1, 1, 1], dtype=np.int64)
    out.append(sro.build_mol_objects(0, x, ei, ea, smiles="C1CC1"))
    # molecule 1: chain of 4 with a branch, mixed bond types
    x = np.asarray([[0], [1], [3], [2]], dtype=np.int64)
    ei = np.asarray([[0, 1, 1, 2, 1, 3], [1, 0, 2, 1, 3, 1]], dtype=np.int64)
    ea = np.asarray([1, 1, 2, 2, 1, 1], dtype=np.int64)
    out.append(sro.build_mol_objects(1, x, ei, ea, smiles="CC(C)N"))
    return out


# ---------------------------------------------------------------------------
# split / steps arithmetic
# ---------------------------------------------------------------------------


def test_config_has_frozen_block_and_vocabulary():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["protocol"] == "cscl-predictive-bottleneck-diagnosis-v1"
    assert cfg["status"] == "frozen"
    frozen = cfg["frozen"]
    assert frozen["T"] == 240 * frozen["steps_per_epoch_8k"]
    assert len(frozen["n4_short_snapshot_steps"]) == 5
    assert frozen["n4_stepmatched_snapshot_steps"] == frozen["n8_o8_soup_steps"]
    vocab = set(cfg["final_decision_vocabulary"])
    assert "SAMPLE_EFFICIENCY_BOTTLENECK_SUPPORTED" in vocab
    assert "OWNERSHIP_GENERALIZATION_SIGNAL" in vocab
    assert "ARCHITECTURAL_INDUCTIVE_BIAS_SUSPECTED" in vocab
    assert "INPUT_OBSERVABILITY_LIMITATION_FOUND" in vocab
    assert "BOTTLENECK_NOT_IDENTIFIED" in vocab


def test_build_split_grouping_and_determinism(monkeypatch):
    monkeypatch.setattr(drv, "FIT_TARGET", 10)
    monkeypatch.setattr(drv, "MONITOR_TARGET", 5)
    smiles = [f"C{i}" for i in range(40)]
    smiles[3] = smiles[0]  # duplicate group of size 2
    a = drv.build_split(smiles, "test-salt")
    b = drv.build_split(smiles, "test-salt")
    assert a["fit"] == b["fit"] and a["monitor"] == b["monitor"] and a["locked"] == b["locked"]
    assert a["overlap"] == {"fit_monitor": 0, "fit_locked": 0, "monitor_locked": 0, "union": 40}
    for left, right in (("fit", "monitor"), ("fit", "locked"), ("monitor", "locked")):
        assert set(a[left]).isdisjoint(a[right])
    # duplicated smiles never cross groups
    for group in ([0, 3],):
        side = [("fit" if i in a["fit"] else "monitor" if i in a["monitor"] else "locked") for i in group]
        assert len(set(side)) == 1


def test_build_fit4k_nested_and_label_blind(monkeypatch):
    monkeypatch.setattr(drv, "FIT_TARGET", 20)
    monkeypatch.setattr(drv, "MONITOR_TARGET", 5)
    smiles = [f"C{i}" for i in range(60)]
    split = drv.build_split(smiles, "test-salt")
    fit4k = drv.build_fit4k(smiles, "test-salt", split["fit"], target=8)
    assert 8 <= len(fit4k) <= 16
    assert set(fit4k) <= set(split["fit"])
    assert not (set(fit4k) & set(split["monitor"]))
    # deterministic and insensitive to a y array (never passed)
    assert fit4k == drv.build_fit4k(smiles, "test-salt", split["fit"], target=8)
    sig = inspect.signature(drv.build_fit4k)
    assert "y" not in sig.parameters and "y" not in (sig.parameters.keys())


def test_compute_steps():
    s = drv.compute_steps(8000, 4000)
    assert s["steps_per_epoch_8k"] == 63 and s["steps_per_epoch_4k"] == 32
    assert s["T"] == 15120
    assert s["n4_short_snapshot_steps"] == [7552, 7584, 7616, 7648, 7680]
    assert s["n4_stepmatched_snapshot_steps"] == [14868, 14931, 14994, 15057, 15120]
    assert max(s["n4_short_snapshot_steps"]) < s["T"]
    assert s["n4_stepmatched_approx_epochs"] == 472.5
    # 8001 rows still 63 steps/epoch (the freeze is robust to a +1 group overshoot)
    assert drv.compute_steps(8001, 4001)["T"] == 15120


# ---------------------------------------------------------------------------
# model identity / invariance (CPU, data-free)
# ---------------------------------------------------------------------------


def _prep():
    return {"global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float64),
            "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float64)}


def test_parameter_count_and_init_identity():
    cpu = torch.device("cpu")
    n = drv.model_for("N", _prep(), 0, cpu)
    o = drv.model_for("O", _prep(), 0, cpu)
    assert sum(p.numel() for p in n.parameters()) == 413505
    assert sum(p.numel() for p in o.parameters()) == 413505
    assert src.state_dict_hash(n) == src.state_dict_hash(o)
    assert n.organization_mode == src.NULL and o.organization_mode == src.ORGANIZED


def test_no_forbidden_modules_and_no_hidden_arch_difference():
    cpu = torch.device("cpu")
    n = drv.model_for("N", _prep(), 0, cpu)
    assert src.forbidden_module_scan(n) == []
    o = drv.model_for("O", _prep(), 0, cpu)
    assert list(n.state_dict()) == list(o.state_dict())
    assert all(n.state_dict()[k].shape == o.state_dict()[k].shape for k in n.state_dict())


def test_predict_shapes_batch_consistency_and_invariance():
    cpu = torch.device("cpu")
    model = drv.model_for("N", _prep(), 0, cpu)
    objs = _tiny_objects()
    p_batch = drv.predict_bjoc(model, objs, [0, 1], cpu)
    p_single = np.concatenate([drv.predict_bjoc(model, [objs[0]], [0], cpu),
                               drv.predict_bjoc(model, [objs[1]], [0], cpu)])
    assert p_batch.shape == (2,)
    assert np.max(np.abs(p_batch - p_single)) < 1e-5
    rng = np.random.default_rng(0)
    perm = rng.permutation(objs[0].n_atoms)
    relab = drv.relabel_objects(objs[0], perm)
    p_relab = drv.predict_bjoc(model, [relab], [0], cpu)
    assert abs(p_relab[0] - p_single[0]) < 1e-5


def test_soup_from_mean_and_hash_helpers():
    states = {}
    for step in (7, 8):
        states[step] = {"w": torch.full((2, 2), float(step))}
    soup = drv.soup_from(states, [7, 8])
    assert torch.allclose(soup["w"], torch.full((2, 2), 7.5))
    with pytest.raises(AssertionError):
        drv.soup_from(states, [7, 9])
    a = np.arange(4, dtype=np.float64)
    assert drv.array_sha256(a) == hashlib.sha256(a.tobytes()).hexdigest()
    assert drv.state_sha256(states[7]) == drv.state_sha256(states[7])


def test_top_k_removed_semantics():
    e_new = np.asarray([1.0, 100.0, 2.0, 3.0])
    e_ref = np.asarray([5.0, 100.0, 4.0, 4.0])
    d = e_new - e_ref
    assert drv._top_k_removed(d, e_new, e_ref, 1) == pytest.approx(d[[0, 2, 3]].mean())
    assert drv._top_k_removed(d, e_new, e_ref, 0) == pytest.approx(d.mean())


def test_source_discipline_scans():
    src_text = Path(drv.__file__).read_text(encoding="utf-8")
    assert not [m.group(0) for m in __import__("re").finditer(r"_load_zinc\([^)]*\)", src_text)
                if '"train"' not in m.group(0)]
    for token in ("attribute" + "_atom_type", "topology" + "_features", "global" + "_context",
                  "teach" + "er"):
        assert token not in src_text


def test_readouts_cover_trajectories():
    assert set(drv.READOUT_NAMES) == {"N4short", "N4stepmatched", "N8", "O8"}
    assert drv.READOUT_SPECS["N4short"]["arm"] == "N"
    assert drv.READOUT_SPECS["O8"]["arm"] == "O"
    # exactly six formal trajectories produce the eight readout models
    tris = {("N", "fit4k", s) for s in drv.SEEDS} | {("N", "fit8k", s) for s in drv.SEEDS} | \
           {("O", "fit8k", s) for s in drv.SEEDS}
    assert len(tris) == 6


# ---------------------------------------------------------------------------
# result-dependent (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_frozen_split_matches_generated_files():
    if not (RESULTS / "split_a.json").exists():
        pytest.skip("result artefacts absent")
    cfg = yaml.safe_load(CONFIG.read_text())
    split = json.loads((RESULTS / "split_a.json").read_text())
    fit4k = json.loads((RESULTS / "fit4k_subset.json").read_text())
    assert split["fit8k_sha256_rows"] == cfg["frozen"]["fit8k_sha256_rows"]
    assert split["monitor_sha256_rows"] == cfg["frozen"]["monitor_sha256_rows"]
    assert split["locked_sha256_rows"] == cfg["frozen"]["locked_sha256_rows"]
    assert fit4k["fit4k_sha256_rows"] == cfg["frozen"]["fit4k_sha256_rows"]
    assert set(fit4k["rows"]) <= set(split["fit_rows"])
    assert not (set(fit4k["rows"]) & set(split["monitor_rows"]))
    assert not (set(fit4k["rows"]) & set(split["locked_rows"]))


@pytest.mark.slow
def test_gates_all_pass_if_present():
    path = RESULTS / "correctness_gates.json"
    if not path.exists():
        pytest.skip("gates artefact absent")
    gates = json.loads(path.read_text())
    assert gates["all_pass"] is True
    assert all(v["pass"] for v in gates["gates"].values())
