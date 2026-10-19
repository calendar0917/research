"""Phase 2b CPU tests: frozen config, metrics, promotion rule and isolation.

No ZINC data or trained checkpoint is required for most tests.  The two
artefact-dependent checks skip when the historical files are unavailable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REPO = CODE.parents[3]
for p in (CODE, REPO / "tracks/structure-contribution/code", REPO / "tracks/structural-sufficiency/code/phase2a"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import phase2b_common as C  # noqa: E402
import phase2b_runner as R  # noqa: E402
import phase2a_models as M  # noqa: E402
import phase2a_runner as R2A  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def _mols():
    return [o for _, o in R2A._synthetic_molecules()]


# ---------------------------------------------------------------------------
# frozen configuration
# ---------------------------------------------------------------------------


def test_frozen_config():
    cfg = C.load_config()
    assert cfg["protocol_id"] == "ssrc-phase2b-zinc-v1"
    assert cfg["data"]["train_rows"] == 10000
    assert cfg["recipe"]["epochs"] == 360
    assert cfg["recipe"]["updates"] == 28440
    assert cfg["recipe"]["batch_size"] == 128
    assert cfg["recipe"]["shuffle_offset"] == 101
    assert cfg["promotion_rule"]["practical_gain_mean_mae"] == 0.003
    assert cfg["promotion_rule"]["robust_gain_after_drop_top1_favourable"] == 0.002
    assert cfg["promotion_rule"]["mp_separation_mean_mae"] == 0.002
    assert cfg["promotion_rule"]["max_finalists"] == 1
    assert cfg["test_rule"]["one_shot"] is True
    assert cfg["budget"]["gpu_hours_cap"] == 2.5
    assert len(cfg["audit_expectations"]["p3_soup_sha256"]) == 2


def test_protocol_file_exists_and_mentions_no_monitor_in_official_table():
    text = (C.OUT / "PROTOCOL.md").read_text(encoding="utf-8")
    table_lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    assert not any("0.085345" in ln for ln in table_lines)
    assert "ZINC-Full" in text and "0.011" in text
    assert "STANDARD_ZINC_GAIN_CONFIRMED" in text
    assert "VALID_GAIN_NOT_REPLICATED" in text


def test_recipe_records_are_the_historical_full10k_r3():
    for seed in C.SEEDS:
        rec = json.loads((C.RECIPE / f"train_R3_seed{seed}.json").read_text())
        assert rec["epochs"] == 360
        assert rec["hyper"]["batch_size"] == 128
        assert rec["hyper"]["lr"] == 1e-3
        assert rec["hyper"]["weight_decay"] == 1e-5
        assert rec["hyper"]["grad_clip"] == 5.0
        assert rec["soup_members"] == [356, 357, 358, 359, 360]
        assert rec["hyper"]["train_shuffle_seed"] == seed + 101


# ---------------------------------------------------------------------------
# arms unchanged
# ---------------------------------------------------------------------------


def test_arm_parameter_counts():
    counts = {}
    for arm, kind in C.ARM_KINDS.items():
        _, model, _ = M.build_local_model(kind, seed=0, norm=NORM, device="cpu")
        counts[arm] = int(sum(p.numel() for p in model.parameters()))
    assert counts == {"P3": 413505, "F": 463489, "M": 464257}
    assert M.local_parameters(M.KIND_NODE_FFN) == 49984
    assert M.local_parameters(M.KIND_LOCAL_MP) == 50752


def test_f_is_adjacency_blind_m_is_not():
    objs = _mols()
    batch = sro.batch_objects(objs)
    rolled = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    rolled["bond_index"] = torch.roll(batch["bond_index"], shifts=1, dims=0)
    rolled_feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    rolled_feat["bond_feat"] = torch.roll(batch["bond_feat"], shifts=3, dims=0)
    for arm, kind, blind in (("F", M.KIND_NODE_FFN, True), ("M", M.KIND_LOCAL_MP, False)):
        _, model, _ = M.build_local_model(kind, seed=0, norm=NORM, device="cpu")
        model.eval()
        with torch.no_grad():
            a0 = model.encode_atoms(batch)
            d_idx = float((model.encode_atoms(rolled) - a0).abs().max())
            d_feat = float((model.encode_atoms(rolled_feat) - a0).abs().max())
        if blind:
            assert d_idx == 0.0 and d_feat == 0.0
        else:
            assert d_idx > 1e-4 and d_feat > 1e-4


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_abs_metrics_and_bootstrap():
    y = np.zeros(1000)
    pred = np.full(1000, 0.01)
    m = C.abs_metrics(pred, y)
    assert m["mae"] == pytest.approx(0.01)
    assert m["rmse"] == pytest.approx(0.01)
    d = np.full(1000, 0.004)
    b = C.paired_bootstrap(d, n=500)
    assert b["mean_diff"] == pytest.approx(0.004)
    assert b["ci95_low"] == pytest.approx(0.004)
    assert b["better_fraction"] == 1.0


def test_seed_averaged_delta_is_seed_mean_mae_gain():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    ref = {0: y + 0.01, 1: y - 0.02}
    cand = {0: y + 0.004, 1: y - 0.016}
    d = C.seed_averaged_delta(cand, ref, y)
    expected = 0.5 * (np.abs(ref[0] - y).mean() - np.abs(cand[0] - y).mean())
    expected += 0.5 * (np.abs(ref[1] - y).mean() - np.abs(cand[1] - y).mean())
    assert d.mean() == pytest.approx(expected)


def test_top_removed_gain():
    d = np.array([4.0] + [0.0] * 999)
    assert C.top_removed_gain(d, [0]) == 0.0
    assert C.top_removed_gain(np.full(1000, 0.004), [0]) == pytest.approx(0.004)


def test_cycle_proxy():
    ring6 = _mols()[0]
    chain7 = _mols()[1]
    longest, penalty = C.cycle_proxy([ring6, chain7])
    assert longest[0] == 6 and not penalty[0]
    assert longest[1] == 0 and not penalty[1]
    # an explicit 8-cycle is > 6
    n = 8
    ei = []
    for i in range(n):
        ei += [[i, (i + 1) % n], [(i + 1) % n, i]]
    obj = sro.build_mol_objects(
        0,
        np.zeros(n, dtype=np.int64),
        np.asarray(ei, dtype=np.int64).T,
        np.zeros(len(ei), dtype=np.int64),
    )
    longest, penalty = C.cycle_proxy([obj])
    assert longest[0] == 8 and penalty[0]


# ---------------------------------------------------------------------------
# promotion rule
# ---------------------------------------------------------------------------


def _cfg():
    return C.load_config()


def _preds(p3, cand, seed_shift=0.0):
    y = np.zeros(1000)
    return (
        y,
        {
            "P3": {0: p3, 1: p3 + seed_shift},
            "F": {0: cand, 1: cand + seed_shift},
            "M": {0: cand, 1: cand + seed_shift},
        },
    )


def test_candidate_eval_uniform_gain_qualifies():
    y, preds = _preds(np.full(1000, 0.01), np.full(1000, 0.006))
    masks = {"penalty_complement": np.ones(1000, dtype=bool)}
    e = R._candidate_eval("F", preds, y, masks, _cfg())
    assert e["mean_gain"] == pytest.approx(0.004)
    assert e["per_seed_gain"]["0"] == pytest.approx(0.004)
    assert e["checks"]["both_seeds_positive"]
    assert e["qualifies"]


def test_candidate_eval_gain_from_one_molecule_does_not_qualify():
    p3 = np.zeros(1000)
    p3[0] = 4.0  # one very bad P3 row
    y, preds = _preds(p3, np.zeros(1000))
    masks = {"penalty_complement": np.ones(1000, dtype=bool)}
    e = R._candidate_eval("F", preds, y, masks, _cfg())
    assert e["mean_gain"] == pytest.approx(0.004)
    assert e["drop_top1_favourable_gain"] == 0.0
    assert not e["qualifies"]


def test_candidate_eval_below_band_does_not_qualify():
    y, preds = _preds(np.full(1000, 0.01), np.full(1000, 0.008))
    masks = {"penalty_complement": np.ones(1000, dtype=bool)}
    e = R._candidate_eval("F", preds, y, masks, _cfg())
    assert e["mean_gain"] == pytest.approx(0.002)
    assert not e["qualifies"]


def test_select_finalist_never_returns_two():
    def ev(f, m):
        return {
            "F": {"qualifies": f},
            "M": {"qualifies": m},
        }

    assert R._select_finalist(ev(False, False), {"separated": False})[0] is None
    assert R._select_finalist(ev(True, False), {"separated": False})[0] == "F"
    assert R._select_finalist(ev(False, True), {"separated": False})[0] == "M"
    assert R._select_finalist(ev(True, True), {"separated": True})[0] == "M"
    assert R._select_finalist(ev(True, True), {"separated": False})[0] == "F"


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------


def test_label_guard_blocks_labels():
    class Data:
        x = torch.zeros(1)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.long)
        y = torch.zeros(1)

    guard = R._LabelGuard([Data()])
    item = guard[0]
    assert hasattr(item, "x") and hasattr(item, "edge_index") and hasattr(item, "edge_attr")
    with pytest.raises(AttributeError):
        _ = guard.y


def test_test_stage_refuses_without_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(R.C, "OUT", tmp_path)
    (tmp_path / "valid_decision.json").write_text(json.dumps({"finalist": None, "promotion_authorized": False}))
    with pytest.raises(SystemExit):
        R._finalist_from_decision()


def test_frozen_hash_fields_are_well_formed():
    cfg = C.load_config()
    for key in ("objects_train10k", "objects_valid", "prep_train10k", "soup_R3_seed0", "soup_R3_seed1"):
        assert len(cfg["frozen_inputs"][key]["sha256"]) == 64
    for seed in (0, 1):
        assert len(cfg["audit_expectations"]["batch_order_sha256"][str(seed)]) == 64
        assert len(cfg["audit_expectations"]["encoder_used_sha256"][str(seed)]) == 64
