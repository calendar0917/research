"""Phase 1d CPU tests: additive incidence control and frozen capacity arms."""

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
for p in (CODE, CODE / "phase1b", CODE / "phase1c"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phase1d_gates import (  # noqa: E402
    edge_batch,
    mol_chain,
    one_mol_batch,
    permute_mol,
    rewiring_pair,
)
from ssrc_models import build_model, parameter_count  # noqa: E402
from ssrc_models_p1c import P1CConfig, register_phase1c_models  # noqa: E402
from ssrc_models_p1d import (  # noqa: E402
    AdditiveIncidenceRelation,
    ModelRIncidence,
    P1DConfig,
    base_incidence_config,
    incidence_capacity_config,
    r1_size_capacity_config,
    r1_size_edge_aggregate,
    register_phase1d_models,
)
from ssrc_train import MODEL_KEYS  # noqa: E402

ATOM_DIM = 21
EDGE_DIM = 4


def _small_cfg() -> P1DConfig:
    return P1DConfig(hidden=16, ffn_hidden=32, rel_hidden=32, edge_readout_hidden=16, incidence_rel_hidden=31)


def test_base_parameters_and_capacity_match_frozen_contract() -> None:
    register_phase1c_models()
    register_phase1d_models()
    torch.manual_seed(0)
    r1 = build_model("R1_size", ATOM_DIM, EDGE_DIM, P1CConfig())
    torch.manual_seed(0)
    inc = build_model("R_incidence", ATOM_DIM, EDGE_DIM, base_incidence_config())
    assert parameter_count(r1) == 115_977
    assert parameter_count(inc) == 115_952
    assert abs(parameter_count(inc) - parameter_count(r1)) / parameter_count(r1) < 0.01
    assert parameter_count(build_model("R1_size_cap", ATOM_DIM, EDGE_DIM, r1_size_capacity_config())) == 453_129
    assert parameter_count(build_model("R_incidence_cap", ATOM_DIM, EDGE_DIM, incidence_capacity_config())) == 436_841
    for n in (453_129, 436_841):
        assert abs(n - 439_559) / 439_559 < 0.05


def test_incidence_and_r1_size_use_identical_batch_keys() -> None:
    register_phase1c_models()
    register_phase1d_models()
    assert MODEL_KEYS["R_incidence"] == MODEL_KEYS["R1_size"]
    assert MODEL_KEYS["R1_size_cap"] == MODEL_KEYS["R1_size"]
    assert MODEL_KEYS["R_incidence_cap"] == MODEL_KEYS["R1_size"]


def test_incidence_relation_has_no_joint_endpoint_input() -> None:
    cfg = base_incidence_config()
    model = ModelRIncidence(ATOM_DIM + 4, EDGE_DIM, cfg)
    assert isinstance(model.rel, AdditiveIncidenceRelation)
    assert model.rel.net[0].in_features == cfg.hidden + EDGE_DIM
    assert model.rel.net[0].in_features != 3 * cfg.hidden + EDGE_DIM


def test_incidence_edge_aggregate_invariant_but_r1_size_changes_under_valid_rewiring() -> None:
    g1, g2 = rewiring_pair()
    b1, b2 = edge_batch(g1), edge_batch(g2)
    cfg = _small_cfg()
    inc_diffs, r1_diffs = [], []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        inc = ModelRIncidence(8, 2, cfg).eval()
        torch.manual_seed(seed)
        r1 = build_model("R1_size", 4, 2, cfg).eval()
        tok = torch.randn(6, 16)
        with torch.no_grad():
            im1, is1, _, _ = inc.edge_aggregate(tok, b1)
            im2, is2, _, _ = inc.edge_aggregate(tok, b2)
            rm1, rs1, _, _ = r1_size_edge_aggregate(r1, tok, b1)
            rm2, rs2, _, _ = r1_size_edge_aggregate(r1, tok, b2)
        inc_diffs.append(float(max((im1 - im2).abs().max(), (is1 - is2).abs().max())))
        r1_diffs.append(float(max((rm1 - rm2).abs().max(), (rs1 - rs2).abs().max())))
    assert max(inc_diffs) < 1e-6
    assert min(r1_diffs) > 1e-5


def test_permutation_and_edge_direction_invariance() -> None:
    cfg = _small_cfg()
    torch.manual_seed(0)
    inc = ModelRIncidence(8, 2, cfg).eval()
    torch.manual_seed(0)
    r1 = build_model("R1_size", 4, 2, cfg).eval()
    mol = mol_chain(6, d=4)
    b0 = one_mol_batch(mol)
    perm = np.asarray([3, 0, 5, 1, 4, 2], dtype=np.int64)
    bp = one_mol_batch(permute_mol(mol, perm))
    b_rev = dict(b0)
    b_rev["root_edge_index"] = b0["root_edge_index"].flip(0)
    with torch.no_grad():
        assert torch.allclose(inc(b0), inc(bp), atol=1e-6)
        assert torch.allclose(r1(b0), r1(bp), atol=1e-6)
        assert torch.allclose(inc(b0), inc(b_rev), atol=1e-6)
        assert torch.allclose(r1(b0), r1(b_rev), atol=1e-6)


def test_incidence_forward_backward_and_both_slots_live() -> None:
    cfg = base_incidence_config()
    register_phase1d_models()
    from ssrc_data import CACHE_DIR, LabelVault, gather_batch, load_cache
    from ssrc_train import _model_batch

    cache = load_cache(CACHE_DIR / "zinc_phase1b.pt")
    size = torch.load(
        REPO / "tracks/structural-sufficiency/results/phase1c/zinc_size_features.pt",
        map_location="cpu",
        weights_only=False,
    )["size_feat"].float()
    cache["mol_size_feat"] = size
    split = json.loads((REPO / "tracks/structural-sufficiency/results/phase1b/split_zinc_phase1b.json").read_text())
    rows = [int(x) for x in split["rows"]["fit"][:8]]
    vault = LabelVault(
        REPO / "tracks/structural-sufficiency/results/phase1b/zinc_labels_fit_monitor.json",
        [int(x) for x in split["rows"]["fit"]],
        [int(x) for x in split["rows"]["monitor"]],
        [int(x) for x in split["rows"]["reserved"]],
    )
    vault.load()
    y = vault.labels(rows, "train")
    b = _model_batch("R_incidence", gather_batch(cache, rows, y, include_size=True))
    torch.manual_seed(0)
    model = build_model("R_incidence", ATOM_DIM, EDGE_DIM, cfg)
    pred = model(b)
    loss = torch.nn.functional.l1_loss(pred, b["y"])
    loss.backward()
    for n, p in model.named_parameters():
        if n.startswith(("rel.", "trunk.", "readout.", "edge_readout.", "head.")):
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
            assert float(p.grad.abs().max()) > 0
    with torch.no_grad():
        tok = model.trunk(b)
        src, dst = b["root_edge_index"]
        e = b["root_edge_attr"]
        base = model.rel(tok[src], e) + model.rel(tok[dst], e)
        r_src = model.rel(torch.zeros_like(tok[src]), e) + model.rel(tok[dst], e)
        r_dst = model.rel(tok[src], e) + model.rel(torch.zeros_like(tok[dst]), e)
    assert float((base - r_src).abs().max()) > 0
    assert float((base - r_dst).abs().max()) > 0


def test_structure_feature_table_is_structure_only() -> None:
    path = REPO / "tracks/structural-sufficiency/results/phase1d/structural_features.npz"
    manifest = REPO / "tracks/structural-sufficiency/results/phase1d/structural_features_manifest.json"
    if not path.exists() or not manifest.exists():
        pytest.skip("run phase1d_features.py first")
    z = np.load(path, allow_pickle=False)
    m = json.loads(manifest.read_text())
    assert z["rows"].size == 10000
    assert m["scope"]["structure_only"] is True
    assert m["scope"]["labels_read"] is False
    assert m["scope"]["official_valid_test_read"] is False
    assert m["scope"]["internal_reserved_read"] is False
    assert "y" not in z.files
