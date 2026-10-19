"""Phase 2a CPU tests: the two arms, the insertion point and the decision rules.

The tests are about mechanism, not numbers: no ZINC data and no checkpoint is
needed for the model tests, and artefact-dependent checks skip when the run has
not produced them yet.
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
for p in (CODE, REPO / "tracks/structure-contribution/code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import phase2a_common as C  # noqa: E402
import phase2a_models as M  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def _mols():
    from phase2a_runner import _synthetic_molecules

    return [o for _, o in _synthetic_molecules()]


def _batch(objs):
    return sro.batch_objects(objs)


# ---------------------------------------------------------------------------
# parameters / initialisation
# ---------------------------------------------------------------------------


def test_parameter_counts_and_inventory():
    base, _, _ = M.build_local_model(M.KIND_NONE, seed=0, norm=NORM)
    _, ffn, _ = M.build_local_model(M.KIND_NODE_FFN, seed=0, norm=NORM)
    _, mp, _ = M.build_local_model(M.KIND_LOCAL_MP, seed=0, norm=NORM)
    assert sum(p.numel() for p in base.parameters()) == 413_505
    assert sum(p.numel() for p in ffn.parameters()) == 463_489
    assert sum(p.numel() for p in mp.parameters()) == 464_257
    for model in (base, ffn, mp):
        assert M.parameter_audit(model)["parameter_exact"]
    delta = 464_257 - 463_489
    assert delta / M.local_parameters(M.KIND_LOCAL_MP) <= 0.05


def test_trunk_init_is_bit_identical_to_p3e():
    import bjoc_perfmech_long_v1 as bp

    base = bp.build_bjoc_null(0, NORM, "cpu")
    ref = M.src.state_dict_hash(base)
    for kind in (M.KIND_NODE_FFN, M.KIND_LOCAL_MP):
        _, model, info = M.build_local_model(kind, seed=0, norm=NORM)
        assert info["trunk_matches_base_init"]
        assert M.trunk_param_sha256(model) == ref


def test_local_init_is_deterministic_and_shared():
    _, a, _ = M.build_local_model(M.KIND_NODE_FFN, seed=0, norm=NORM)
    _, b, _ = M.build_local_model(M.KIND_NODE_FFN, seed=0, norm=NORM)
    _, c, _ = M.build_local_model(M.KIND_LOCAL_MP, seed=0, norm=NORM)
    assert M.local_state_sha256(a.local) == M.local_state_sha256(b.local)
    assert M.local_block_state_sha256(a.local) == M.local_block_state_sha256(c.local)
    # a different seed changes the local module (the generator is not ignored)
    _, d, _ = M.build_local_model(M.KIND_NODE_FFN, seed=1, norm=NORM)
    assert M.local_state_sha256(d.local) != M.local_state_sha256(a.local)


# ---------------------------------------------------------------------------
# arm semantics
# ---------------------------------------------------------------------------


def test_node_ffn_ignores_adjacency():
    objs = _mols()
    batch = _batch(objs)
    _, model, _ = M.build_local_model(M.KIND_NODE_FFN, seed=0, norm=NORM)
    model.eval()
    with torch.no_grad():
        a = model.atom_encoder(batch["atom_feat"])
        ref = model.local(a, batch)
        for key, roll in (("bond_index", 1), ("bond_feat", 3)):
            other = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            other[key] = torch.roll(other[key], shifts=roll, dims=0)
            assert torch.equal(model.local(a, other), ref)


def test_local_mp_uses_edges_and_bond_features():
    objs = _mols()
    batch = _batch(objs)
    _, model, _ = M.build_local_model(M.KIND_LOCAL_MP, seed=0, norm=NORM)
    model.eval()
    with torch.no_grad():
        a = model.atom_encoder(batch["atom_feat"])
        ref = model.local(a, batch)
        idx = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        idx["bond_index"] = torch.roll(idx["bond_index"], shifts=1, dims=0)
        assert not torch.allclose(model.local(a, idx), ref)
        feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        feat["bond_feat"] = torch.roll(feat["bond_feat"], shifts=3, dims=0)
        assert not torch.allclose(model.local(a, feat), ref)
        perm = torch.randperm(a.shape[0], generator=torch.Generator().manual_seed(5))
        assert not torch.allclose(model.local(a[perm], batch), ref)


def test_directed_edges_cover_both_orientations():
    batch = _batch(_mols())
    _, model, _ = M.build_local_model(M.KIND_LOCAL_MP, seed=0, norm=NORM)
    src, dst = model.local.directed_edges(batch)
    bi = batch["bond_index"]
    n = bi.shape[0]
    assert torch.equal(src[:n], bi[:, 0]) and torch.equal(dst[:n], bi[:, 1])
    assert torch.equal(src[n:], bi[:, 1]) and torch.equal(dst[n:], bi[:, 0])


def test_new_parameters_receive_gradient():
    batch = _batch(_mols())
    for kind in (M.KIND_NODE_FFN, M.KIND_LOCAL_MP):
        _, model, _ = M.build_local_model(kind, seed=0, norm=NORM)
        model.zero_grad(set_to_none=True)
        (model(batch) ** 2).mean().backward()
        new = {n: p.grad for n, p in model.named_parameters() if n.startswith("local.")}
        assert new, kind
        assert all(g is not None and float(g.abs().sum()) > 0 for g in new.values()), kind


def test_branch_is_not_identity():
    batch = _batch(_mols())
    _, ffn, _ = M.build_local_model(M.KIND_NODE_FFN, seed=0, norm=NORM)
    _, mp, _ = M.build_local_model(M.KIND_LOCAL_MP, seed=0, norm=NORM)
    ffn.eval()
    mp.eval()
    with torch.no_grad():
        a = ffn.atom_encoder(batch["atom_feat"])
        for model in (ffn, mp):
            shift = M.local_branch_shift(a, model.local(a, batch))
            assert shift["relative_shift"] > 1e-2
            assert shift["cosine"] < 0.999


def test_reader_layout_is_unchanged():
    batch = _batch(_mols())
    _, model, _ = M.build_local_model(M.KIND_NODE_FFN, seed=0, norm=NORM)
    feats = model.forward_features(batch)
    assert feats["reader_in"].shape[1] == 96 + 2 * 96 + sro.GLOBAL_DIM


def test_identity_arm_matches_published_bjoc():
    import structural_relation_composer_v1 as src

    batch = _batch(_mols())
    _, model, _ = M.build_local_model(M.KIND_NONE, seed=0, norm=NORM)
    plain = src.BJOC(src.BJOCConfig(), NORM["global_mean"], NORM["global_scale"], src.NULL)
    plain.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(plain(batch), model(batch))


# ---------------------------------------------------------------------------
# decision rules
# ---------------------------------------------------------------------------


def test_decision_strong_increment():
    d = C.decide(
        mae_f=0.0900,
        mae_m=0.0860,
        rmse_m=C.P3E_RMSE,
        complement_gain=0.004,
        flops_ratio_f=1.19,
        flops_ratio_m=1.19,
        optimization_ok=True,
    )
    assert d["status"] == "STRONG_BACKBONE_MP_INCREMENT"
    assert d["buy_seed1"]


def test_decision_capacity_only():
    d = C.decide(
        mae_f=0.0860,
        mae_m=0.0865,
        rmse_m=C.P3E_RMSE,
        complement_gain=0.004,
        flops_ratio_f=1.19,
        flops_ratio_m=1.19,
        optimization_ok=True,
    )
    assert d["status"] == "CAPACITY_ONLY_INCREMENT"
    assert not d["buy_seed1"]


def test_decision_no_gain_and_optimization_guard():
    d = C.decide(
        mae_f=0.0920,
        mae_m=0.0925,
        rmse_m=C.P3E_RMSE,
        complement_gain=-0.001,
        flops_ratio_f=1.19,
        flops_ratio_m=1.19,
        optimization_ok=True,
    )
    assert d["status"] == "NO_PRACTICAL_GAIN_OVER_P3E"
    bad = C.decide(
        mae_f=0.0800,
        mae_m=0.0700,
        rmse_m=C.P3E_RMSE,
        complement_gain=0.01,
        flops_ratio_f=1.19,
        flops_ratio_m=1.19,
        optimization_ok=False,
    )
    assert bad["status"] == "OPTIMIZATION_UNRESOLVED"
    assert not bad["buy_seed1"]


def test_decision_mps_gain_concentrated_in_penalised_group_is_not_bought():
    d = C.decide(
        mae_f=0.0900,
        mae_m=0.0830,
        rmse_m=C.P3E_RMSE,
        complement_gain=0.001,
        flops_ratio_f=1.19,
        flops_ratio_m=1.19,
        optimization_ok=True,
    )
    assert d["status"] != "STRONG_BACKBONE_MP_INCREMENT"
    assert not d["buy_seed1"]


# ---------------------------------------------------------------------------
# artefacts (skip before the run exists)
# ---------------------------------------------------------------------------


def _load(name: str):
    path = C.RESULTS_2A / name
    if not path.exists():
        pytest.skip(f"{name} has not been produced yet")
    return C.read_json(path)


def test_cpu_gates_pass():
    payload = _load("cpu_gates_phase2a.json")
    assert payload["status"] == "PASS"
    assert payload["n_passed"] == payload["n_gates"]


def test_replay_reproduces_frozen_p3e():
    payload = _load("replay.json")
    assert payload["max_abs_prediction_diff"] <= 1e-6
    assert abs(payload["replayed_monitor_mae"] - C.P3E_MAE) < 1e-9
    assert payload["retrained_p3e"] is False


def test_train_records_never_read_monitor():
    for path in sorted(C.RESULTS_2A.glob("train_*_s*.json")):
        payload = json.loads(path.read_text())
        assert payload["monitor_read_at_train"] is False
        assert payload["official_test_loaded"] is False
        assert payload["labels_used"] == "SSRC Fit rows raw y only"


def test_frozen_predictions_match_their_hashes():
    for path in sorted(C.RESULTS_2A.glob("monitor_freeze_*.json")):
        freeze = json.loads(path.read_text())
        assert freeze["monitor_labels_read_before_freeze"] is False
        pred = C.RESULTS_2A / path.name.replace("monitor_freeze_", "monitor_pred_").replace(".json", ".npz")
        if pred.exists():
            assert C.sha256_file(pred) == freeze["prediction_sha256"]


def test_score_read_is_one_shot():
    payload = _load("analysis_phase2a.json")
    assert payload["monitor_labels_read_count"] == 1
    assert payload["reserved_labels_read"] == 0
    assert payload["official_valid_test_loaded"] is False
    assert payload["model_selection"].startswith("none")
