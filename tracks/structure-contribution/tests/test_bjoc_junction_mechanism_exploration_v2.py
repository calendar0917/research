"""Fast correctness tests for cscl-bjoc-junction-mechanism-exploration-v2.

No ZINC dataset, no checkpoints and no GPU: tiny synthetic molecules only.
Covers the six forward modes, the arm-C/E identities, the D reader expansion,
the type-conditioned witness and reference loop, gradient behaviour, the
training-loop equivalence with the frozen ``bp.train_supervised`` body, and
the label/official-test discipline.
"""

from __future__ import annotations

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

import bjoc_junction_mechanism_models_v2 as mm  # noqa: E402
import bjoc_junction_sufficiency_models_v1 as jsm  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/bjoc_junction_mechanism_exploration_v2.yaml"
NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def _mol(idx, atom_types, bonds):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return sro.build_mol_objects(idx, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))


def _tiny_world(n=6):
    specs = [
        ([0, 1, 2, 3], [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 1)]),
        ([4, 5, 6], [(0, 1, 1), (1, 2, 3)]),
        ([7, 7, 8, 9, 10], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 1, 1)]),
        ([1, 2, 3, 4, 5], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 0, 1)]),
        ([2, 2, 3], [(0, 1, 2), (1, 2, 1)]),
        ([0, 1, 2, 3, 4], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 0, 2)]),
    ]
    return [_mol(i, specs[i % len(specs)][0], specs[i % len(specs)][1]) for i in range(n)]


def _from_init(mode, seed=0):
    base, model, info = mm.build_mech_from_init(seed, NORM, "cpu", mode, ssl_state=None)
    return base, model, info


def _batch(objs):
    return sro.batch_objects(objs)


# ---------------------------------------------------------------------------
# registry / parameters
# ---------------------------------------------------------------------------


def test_mode_registry_and_names():
    assert mm.MODES == ("A0", "A1", "A2", "C", "D", "E")
    assert mm.ARM_MODES["D"] == mm.MODE_D
    assert mm.ARM_NAMES["E"] == "J_TYPE_CONDITIONED_POOL"
    assert mm._PARAMS_BASE == 413505
    assert mm._PARAMS_D == 462657


def test_parameter_counts_and_shapes():
    for mode in ("A0", "A1", "A2", "C", "E"):
        _base, model, info = _from_init(mode)
        assert info["parameter_count"] == 413505
        assert info["reader_input_dim"] == 321
    _base, model, info = _from_init(mm.MODE_D)
    assert info["parameter_count"] == 462657
    assert info["reader_input_dim"] == 513
    assert info["reader_expansion"]["new_columns"] == 192
    assert info["reader_expansion"]["new_columns_zero"] is True
    named = {k: int(v.numel()) for k, v in model.named_parameters()}
    assert sum(named.values()) == 462657
    assert named["reader.lins.0.weight"] == 256 * 513


def test_a2_equals_parent_bitwise():
    base, model, _info = _from_init(mm.MODE_A2)
    objs = _tiny_world(6)
    batch = _batch(objs)
    with torch.no_grad():
        p = base(batch)
        q = model(batch)
        f = model.forward_features(batch)
    assert torch.equal(p, q)
    assert float((f["W"] - base.forward_features(batch)["W"]).abs().max()) == 0.0


def test_a1_loop_matches_sufficiency_module_on_synthetic():
    objs = _tiny_world(8)
    y = np.linspace(-2.0, 2.0, len(objs))
    rows = list(range(len(objs)))
    ref = jsm.train_supervised_suff(
        arm="A1",
        seed=0,
        recipe_key="R3",
        init="scratch",
        train_objs=objs,
        rows=rows,
        y_fit=y,
        norm=NORM,
        device=torch.device("cpu"),
        mode=jsm.MODE_A1,
        ssl_state=None,
        smoke_epochs=1,
        log_fn=lambda _m: None,
    )
    new = mm.train_supervised_mech(
        arm="A1",
        seed=0,
        recipe_key="R3",
        init="scratch",
        train_objs=objs,
        rows=rows,
        y_fit=y,
        norm=NORM,
        device=torch.device("cpu"),
        mode=mm.MODE_A1,
        ssl_state=None,
        smoke_epochs=1,
        log_fn=lambda _m: None,
    )
    assert new["batch_order"]["full_hash"] == ref["batch_order"]["full_hash"]
    assert new["init_full_sha256"] == ref["init_full_sha256"]
    worst = max(float((new["soup_state"][k] - ref["soup_state"][k]).abs().max()) for k in ref["soup_state"])
    assert worst == 0.0


# ---------------------------------------------------------------------------
# arm C / E identities
# ---------------------------------------------------------------------------


def test_c_count_only_semantics():
    _base, model, _info = _from_init(mm.MODE_C)
    objs = _tiny_world(5)
    batch = _batch(objs)
    with torch.no_grad():
        f = model.forward_features(batch)
    assert float(f["W"].abs().max()) == 0.0
    assert float(f["V"].abs().max()) == 0.0
    assert float((f["q"] - f["n_b"]).abs().max()) == 0.0
    jb = batch["junc_bond"].numpy()
    n_expected = np.bincount(jb.reshape(-1), minlength=int(batch["bond_feat"].shape[0]))
    assert np.array_equal(f["n_b"].numpy(), n_expected.astype(np.float32))


def test_c_junction_encoder_gets_no_gradient():
    _base, model, _info = _from_init(mm.MODE_C)
    objs = _tiny_world(8)
    y = np.linspace(-1.0, 1.0, len(objs))
    rec = mm.gradient_audit_mech(model, objs=objs, y=y, device="cpu", mode=mm.MODE_C)
    j = rec["modules"]["junction_encoder"]
    assert j["grad_norm_total"] in (None, 0.0)
    assert int(j["active_entries_nonzero_grad"]) == 0


def test_e_init_identical_to_a2_and_conservation():
    _base_e, e, _info_e = _from_init(mm.MODE_E)
    _base_a2, a2, _info_a2 = _from_init(mm.MODE_A2)
    se, sa = e.state_dict(), a2.state_dict()
    assert sorted(se) == sorted(sa)
    assert max(float((se[k].float() - sa[k].float()).abs().max()) for k in se) == 0.0
    objs = _tiny_world(6)
    batch = _batch(objs)
    with torch.no_grad():
        f = e.forward_features(batch)
    ng = int(batch["n_graphs"])
    jb = batch["junc_batch"]
    bb = batch["bond_batch"]
    w, W, V = f["w"], f["W"], f["V"]

    def per_graph_sum(x, idx):
        out = torch.zeros((ng, x.shape[1]), dtype=x.dtype)
        out.index_add_(0, idx, x)
        return out

    lhs_w = per_graph_sum(W, bb)
    rhs_w = 2 * per_graph_sum(w, jb)
    lhs_v = per_graph_sum(V, bb)
    rhs_v = 2 * per_graph_sum(w * w, jb)
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max().clamp(min=1.0))
    assert rel(lhs_w, rhs_w) <= 1e-4
    assert rel(lhs_v, rhs_v) <= 1e-4


def test_e_reference_loop_equality():
    _base, e, _info = _from_init(mm.MODE_E)
    batch = _batch(_tiny_world(5))
    with torch.no_grad():
        f = e.forward_features(batch)
    W_ref, V_ref, n_ref = mm.type_conditioned_grouping_reference(f["w"], batch)
    assert float((W_ref - f["W"]).abs().max()) <= 1e-4
    assert float((V_ref - f["V"]).abs().max()) <= 1e-4
    assert float((n_ref - f["n_b"]).abs().max()) == 0.0


def test_type_witness_a2_blind_e_sensitive():
    ba, bb = mm.type_witness_batch()
    w = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    Wa, Va, na = mm.null_grouping(w, ba)
    Wb, Vb, nb = mm.null_grouping(w, bb)
    assert float((Wa - Wb).abs().max()) == 0.0
    assert float((Va - Vb).abs().max()) == 0.0
    assert float((na - nb).abs().max()) == 0.0
    We_a, Ve_a, _, _, _ = mm.type_conditioned_grouping(w, ba)
    We_b, Ve_b, _, _, _ = mm.type_conditioned_grouping(w, bb)
    assert float((We_a - We_b).abs().max()) > 0.0
    assert float((Ve_a - Ve_b).abs().max()) > 0.0


def test_type_relabel_consistency():
    _base, e, _info = _from_init(mm.MODE_E)
    batch = _batch(_tiny_world(5))
    with torch.no_grad():
        f = e.forward_features(batch)
    perm = torch.tensor([2, 0, 3, 1])
    batch2 = dict(batch)
    bf = batch["bond_feat"].clone()
    bf[:, :4] = batch["bond_feat"][:, perm]
    batch2["bond_feat"] = bf
    W1, V1, _, _, _ = mm.type_conditioned_grouping(f["w"], batch)
    W2, V2, _, _, _ = mm.type_conditioned_grouping(f["w"], batch2)
    assert float((W1 - W2).abs().max()) == 0.0
    assert float((V1 - V2).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# arm D expansion / gradients
# ---------------------------------------------------------------------------


def test_d_initial_forward_equals_a0():
    _base_a0, a0, _i0 = _from_init(mm.MODE_A0)
    _base_d, d, _id = _from_init(mm.MODE_D)
    batch = _batch(_tiny_world(6))
    with torch.no_grad():
        p0 = a0(batch)
        p1 = d(batch)
        f = d.forward_features(batch)
    assert float((p0 - p1).abs().max()) <= 1e-6
    assert f["reader_in"].shape[1] == 513
    old = float((d.reader.lins[0].weight[:, :321] - a0.reader.lins[0].weight).abs().max())
    new = float(d.reader.lins[0].weight[:, 321:].abs().max())
    assert old == 0.0 and new == 0.0


def test_d_gradient_flows_into_junction_and_new_columns():
    _base, d, _info = _from_init(mm.MODE_D)
    objs = _tiny_world(8)
    y = np.linspace(-1.0, 1.0, len(objs))
    rec = mm.gradient_audit_after_optimizer_step(d, objs=objs, y=y, device="cpu", mode=mm.MODE_D)
    first = rec["at_init"]["modules"]["junction_encoder"]["grad_norm_total"]
    after = rec["after_one_step"]["modules"]["junction_encoder"]["grad_norm_total"]
    assert first in (None, 0.0)  # zero-init readout gate, by construction
    assert after > 0.0
    assert rec["after_one_step"]["reader_first_layer_column_grad_max_abs"]["new_max_abs"] > 0.0


def test_d_uses_junction_moments_in_reader_input():
    _base, d, _info = _from_init(mm.MODE_D)
    objs = _tiny_world(4)
    batch = _batch(objs)
    with torch.no_grad():
        f = d.forward_features(batch)
    mu, nu = mm.graph_junction_moments(f["w"], batch)
    assert float((f["reader_in"][:, 321:417] - mu).abs().max()) == 0.0
    assert float((f["reader_in"][:, 417:513] - nu).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# invariances / save-load / edge cases / discipline
# ---------------------------------------------------------------------------


def test_junction_row_permutation_invariance_all_modes():
    from run_bjoc_junction_mechanism_exploration_v2 import _permute_junctions

    objs = _tiny_world(5)
    perm = _permute_junctions(objs, seed=11)
    for mode in mm.MODES:
        _base, model, _info = _from_init(mode)
        with torch.no_grad():
            p0 = model(_batch(objs))
            p1 = model(_batch(perm))
        assert float((p0 - p1).abs().max()) <= 1e-6


def test_save_load_replay(tmp_path):
    _base, model, _info = _from_init(mm.MODE_D)
    objs = _tiny_world(3)
    path = tmp_path / "d.pt"
    torch.save(model.state_dict(), path)
    fresh = mm.BJOCMechV2(mm.DirectReadoutConfig(), NORM["global_mean"], NORM["global_scale"], mode=mm.MODE_D)
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    with torch.no_grad():
        assert torch.equal(model(_batch(objs)), fresh(_batch(objs)))


def test_empty_edge_cases_finite():
    from run_bjoc_junction_mechanism_exploration_v2 import _make_edge_case_objects

    batch = _batch(_make_edge_case_objects())
    for mode in mm.MODES:
        _base, model, _info = _from_init(mode)
        with torch.no_grad():
            p = model(batch)
        assert np.isfinite(p.numpy()).all()


def test_forbidden_module_scan_clean():
    for mode in mm.MODES:
        _base, model, _info = _from_init(mode)
        assert src.forbidden_module_scan(model) == []


def test_no_official_valid_test_loader_tokens():
    for name in ("run_bjoc_junction_mechanism_exploration_v2.py", "bjoc_junction_mechanism_models_v2.py"):
        text = (REPO_ROOT / "tracks/structure-contribution/code" / name).read_text(encoding="utf-8")
        assert "load_official_valid" not in text
        assert "load_official_test" not in text


def test_config_frozen_words_and_budget():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    vocab = cfg["decision_vocabulary"]
    for token in (
        "COUNT_ONLY_NEAR_A2",
        "COUNT_ONLY_INSUFFICIENT",
        "J_DIRECT_READOUT_USEFUL_EXPLORATORY",
        "J_DIRECT_READOUT_NO_GAIN",
        "J_TYPE_POOL_USEFUL_EXPLORATORY",
        "J_TYPE_POOL_NO_GAIN",
        "J_TYPE_POOL_FIT_ONLY",
    ):
        assert token in vocab
    assert float(cfg["budget"]["gpu_hard_limit_hours"]) == 2.0
    assert cfg["label_access"]["locked"] == "never read in this round"
    assert cfg["label_access"]["official_test"] == "never loaded"
    assert cfg["training_matrix"]["seed1"]["not_run"] == ["A0"]


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        mm.BJOCMechV2(src.BJOCConfig(), NORM["global_mean"], NORM["global_scale"], mode="Z")
