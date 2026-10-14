"""Fast correctness tests for cscl-bjoc-junction-sufficiency-audit-v1.

No ZINC dataset, no checkpoints and no GPU: tiny synthetic molecules only.
Covers the three forward modes, the invariance identities, gradient/active
parameter behaviour, the training-loop equivalence with the frozen
``bp.train_supervised`` body, the object-level ownership counterexample and
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

import bjoc_junction_sufficiency_models_v1 as jsm  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import run_bjoc_junction_sufficiency_audit_v1 as drv  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/bjoc_junction_sufficiency_audit_v1.yaml"
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


def _parent(seed=0):
    return src.build_bjoc(
        src.BJOCConfig(),
        seed=seed,
        organization_mode=src.NULL,
        global_mean=NORM["global_mean"],
        global_scale=NORM["global_scale"],
        device="cpu",
    )


def _suff(mode, seed=0):
    parent = _parent(seed)
    model = jsm.BJOCSufficiency(
        src.BJOCConfig(),
        NORM["global_mean"],
        NORM["global_scale"],
        mode=mode,
    )
    model.load_state_dict({k: v.detach().clone() for k, v in parent.state_dict().items()})
    return parent, model


def _batch(objs):
    return sro.batch_objects(objs)


# ---------------------------------------------------------------------------
# forward-mode identities
# ---------------------------------------------------------------------------


def test_mode_registry_and_names():
    assert jsm.MODES == ("A0", "A1", "A2")
    assert jsm.ARM_MODES["A2"] == jsm.MODE_A2
    assert set(jsm.MODE_NAMES) == set(jsm.MODES)


def test_a2_equals_parent_bitwise():
    parent, model = _suff(jsm.MODE_A2)
    objs = _tiny_world(6)
    batch = _batch(objs)
    with torch.no_grad():
        parent_pred = parent(batch)
        model_pred = model(batch)
        model_feats = model.forward_features(batch)
    assert torch.equal(parent_pred, model_pred)
    pfeats = parent.forward_features(batch)
    assert torch.equal(model_feats["reader_in"], pfeats["reader_in"])
    assert torch.equal(model_feats["W"], pfeats["W"])
    assert torch.equal(model_feats["n_b"], pfeats["n_b"])


def test_a0_zero_inputs_and_junction_insensitivity():
    _parent_model, model = _suff(jsm.MODE_A0)
    objs = _tiny_world(4)
    batch = _batch(objs)
    with torch.no_grad():
        feats = model.forward_features(batch)
        pred = model(batch)
    assert float(feats["W"].abs().max()) == 0.0
    assert float(feats["V"].abs().max()) == 0.0
    assert float(feats["n_b"].abs().max()) == 0.0
    perm = drv._perturb_junctions(objs, seed=0)
    with torch.no_grad():
        pred_pert = model(_batch(perm))
    assert float((pred - pred_pert).abs().max()) == 0.0


def test_a1_uniform_within_molecule_and_zero_counts():
    _parent_model, model = _suff(jsm.MODE_A1)
    objs = _tiny_world(5)
    batch = _batch(objs)
    with torch.no_grad():
        feats = model.forward_features(batch)
    assert float(feats["n_b"].abs().max()) == 0.0
    bond_batch = batch["bond_batch"]
    for g in range(int(batch["n_graphs"])):
        m = bond_batch == g
        if int(m.sum()) == 0:
            continue
        assert float((feats["W"][m].max(0).values - feats["W"][m].min(0).values).abs().max()) == 0.0
        assert float((feats["V"][m].max(0).values - feats["V"][m].min(0).values).abs().max()) == 0.0


def test_a2_counts_match_independent_bincount():
    _parent_model, model = _suff(jsm.MODE_A2)
    objs = _tiny_world(5)
    batch = _batch(objs)
    with torch.no_grad():
        feats = model.forward_features(batch)
    jb = batch["junc_bond"].numpy()
    n_expected = np.bincount(jb.reshape(-1), minlength=int(batch["bond_feat"].shape[0]))
    assert np.array_equal(feats["n_b"].numpy(), n_expected.astype(np.float32))


def test_junction_row_permutation_invariance_all_modes():
    objs = _tiny_world(5)
    perm = drv._permute_junctions(objs, seed=11)
    for mode in jsm.MODES:
        _parent_model, model = _suff(mode)
        with torch.no_grad():
            p0 = model(_batch(objs))
            p1 = model(_batch(perm))
        assert float((p0 - p1).abs().max()) <= 1e-6


def test_parameter_tensors_identical_to_parent():
    parent, model = _suff(jsm.MODE_A0)
    assert sorted(model.state_dict().keys()) == sorted(parent.state_dict().keys())
    total = int(sum(p.numel() for p in model.parameters()))
    assert total == 413505 == bp.EXPECTED_PARAMETERS


def test_save_load_replay(tmp_path):
    _parent_model, model = _suff(jsm.MODE_A1)
    objs = _tiny_world(3)
    path = tmp_path / "s.pt"
    torch.save(model.state_dict(), path)
    fresh = jsm.BJOCSufficiency(src.BJOCConfig(), NORM["global_mean"], NORM["global_scale"], mode=jsm.MODE_A1)
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    with torch.no_grad():
        a = model(_batch(objs))
        b = fresh(_batch(objs))
    assert torch.equal(a, b)


def test_forbidden_module_scan_clean():
    for mode in jsm.MODES:
        _parent_model, model = _suff(mode)
        assert src.forbidden_module_scan(model) == []


# ---------------------------------------------------------------------------
# gradient / active capacity
# ---------------------------------------------------------------------------


def test_gradient_audit_active_capacity_ordering():
    objs = _tiny_world(8)
    y = np.linspace(-1.0, 1.0, len(objs))
    records = {}
    for mode in jsm.MODES:
        _parent_model, model = _suff(mode)
        records[mode] = jsm.gradient_audit(model, objs=objs, y=y, device="cpu", mode=mode)
    j0 = records[jsm.MODE_A0]["modules"]["junction_encoder"]
    j1 = records[jsm.MODE_A1]["modules"]["junction_encoder"]
    j2 = records[jsm.MODE_A2]["modules"]["junction_encoder"]
    assert j0["grad_norm_total"] in (None, 0.0)
    assert int(j0["active_entries_nonzero_grad"]) == 0
    assert j1["grad_norm_total"] > 0.0 and j2["grad_norm_total"] > 0.0
    assert records[jsm.MODE_A0]["active_entries_nonzero_grad"] < records[jsm.MODE_A1]["active_entries_nonzero_grad"]
    cols0 = records[jsm.MODE_A0]["composer_first_layer_column_grad_max_abs"]
    assert cols0["W"] == 0.0 and cols0["V"] == 0.0 and cols0["q"] == 0.0
    cols1 = records[jsm.MODE_A1]["composer_first_layer_column_grad_max_abs"]
    assert cols1["W"] > 0.0 and cols1["V"] > 0.0 and cols1["q"] == 0.0


def test_activation_audit_runs():
    _parent_model, model = _suff(jsm.MODE_A2)
    rec = jsm.activation_audit(model, objs=_tiny_world(4), device="cpu", mode=jsm.MODE_A2)
    assert rec["composer_hidden"]["units"] == 160
    assert rec["junction_encoder_hidden"]["units"] == 160


# ---------------------------------------------------------------------------
# training-loop equivalence (A2 smoke vs the frozen bp loop)
# ---------------------------------------------------------------------------


def test_train_loop_equivalence_a2_smoke():
    objs = _tiny_world(12)
    y = np.linspace(-2.0, 2.0, len(objs))
    rows = list(range(len(objs)))
    ref = bp.train_supervised(
        arm="P3",
        seed=0,
        recipe_key="R3",
        init="scratch",
        train_objs=objs,
        rows=rows,
        y_fit=y,
        norm=NORM,
        device=torch.device("cpu"),
        ssl_state=None,
        model_kind="base",
        smoke_epochs=1,
        log_fn=lambda _m: None,
    )
    new = jsm.train_supervised_suff(
        arm="A2",
        seed=0,
        recipe_key="R3",
        init="scratch",
        train_objs=objs,
        rows=rows,
        y_fit=y,
        norm=NORM,
        device=torch.device("cpu"),
        mode=jsm.MODE_A2,
        ssl_state=None,
        smoke_epochs=1,
        log_fn=lambda _m: None,
    )
    assert new["batch_order"]["full_hash"] == ref["batch_order"]["full_hash"]
    assert new["init_full_sha256"] == ref["init_full_sha256"]
    worst = max(float((new["soup_state"][k] - ref["soup_state"][k]).abs().max()) for k in ref["soup_state"])
    assert worst == 0.0


# ---------------------------------------------------------------------------
# object-level ownership counterexample
# ---------------------------------------------------------------------------


def test_object_counterexample_null_invariant_organized_differs():
    a_inst, b_inst = src.counterexample_object_pair()
    assert src.object_marginal_signature(a_inst) == src.object_marginal_signature(b_inst)
    assert src.per_bond_junction_tokens(a_inst) != src.per_bond_junction_tokens(b_inst)
    batch = src.batch_object_instances([a_inst, b_inst])
    for mode in jsm.MODES:
        _parent_model, model = _suff(mode)
        with torch.no_grad():
            feats = model.forward_features(batch, mode=mode)
            pred = model(batch, mode=mode)
        assert float((feats["W"][:4] - feats["W"][4:]).abs().max()) == 0.0
        assert float((feats["V"][:4] - feats["V"][4:]).abs().max()) == 0.0
        # prediction invariance is expected only to float32 reduction noise
        assert float((pred[0] - pred[1]).abs()) <= 1e-6
    _parent_model, model = _suff(jsm.MODE_A2)
    with torch.no_grad():
        org = model.forward_features(batch, organization_mode=src.ORGANIZED)
    assert float((org["W"][:4] - org["W"][4:]).abs().max()) > 0.0


def test_object_counterexample_targets():
    a_inst, b_inst = src.counterexample_object_pair()
    assert a_inst.target_marginal == b_inst.target_marginal
    assert a_inst.target_incidence != b_inst.target_incidence


# ---------------------------------------------------------------------------
# structural signatures / discipline
# ---------------------------------------------------------------------------


def test_structure_signature_and_exact_key_deterministic():
    objs = _tiny_world(4)
    sig = drv._structure_signatures(objs)
    assert sig["global"].shape == (4, sro.GLOBAL_DIM)
    assert sig["atom_hist"].shape == (4, 28)
    keys0 = [drv._exact_key(sig, i) for i in range(4)]
    keys1 = [drv._exact_key(sig, i) for i in range(4)]
    assert keys0 == keys1
    # duplicated identical molecule -> identical exact key, different row
    dup = _mol(99, [0, 1, 2, 3], [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 1)])
    sig2 = drv._structure_signatures([objs[0], dup])
    assert drv._exact_key(sig2, 0) == drv._exact_key(sig2, 1)


def test_config_frozen_words_and_budget():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    vocab = cfg["decision_vocabulary"]
    for token in (
        "J_MOMENTS_USEFUL_EXPLORATORY",
        "J_MOMENTS_NO_CLEAR_GAIN",
        "COUNT_CONDITIONING_USEFUL_EXPLORATORY",
        "COUNT_CONDITIONING_NO_CLEAR_GAIN",
        "NULL_OWNERSHIP_BLINDNESS_VERIFIED",
        "REAL_REPRESENTATION_COLLISION_FOUND",
        "REAL_COLLISION_NOT_FOUND_WITHIN_BUDGET",
        "INCONCLUSIVE_SINGLE_SEED",
    ):
        assert token in vocab
    assert cfg["budget"]["gpu_hard_limit_hours"] <= 1.5
    assert cfg["label_access"]["locked"] == "never read in this round"
    assert cfg["label_access"]["official_test"] == "never loaded"
    assert cfg["seed"] == 0


def test_no_official_valid_test_loader_tokens():
    tokens = ["load_official_" + "valid", "load_official_" + "test"]
    for name in ("run_bjoc_junction_sufficiency_audit_v1.py", "bjoc_junction_sufficiency_models_v1.py"):
        text = (REPO_ROOT / "tracks/structure-contribution/code" / name).read_text(encoding="utf-8")
        for token in tokens:
            assert token not in text


def test_zero_injection_fails_config_or_unknown_mode():
    with pytest.raises(ValueError):
        jsm.BJOCSufficiency(src.BJOCConfig(), NORM["global_mean"], NORM["global_scale"], mode="A9")
