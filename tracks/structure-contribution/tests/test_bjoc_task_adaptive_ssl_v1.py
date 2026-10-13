"""Fast correctness tests for cscl-bjoc-task-adaptive-ssl-v1.

No ZINC dataset, no checkpoints and no GPU: tiny synthetic molecules only.
Covers the frozen-loop replay equivalence, the SSL head-capture wrapper,
the auxiliary branch mechanics, atom-context identity, gradient diagnostics,
masked-view audits, invariance, gate arithmetic and the no-official-test
discipline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bjoc_task_adaptive_ssl_v1 as tat  # noqa: E402
import run_bjoc_task_adaptive_ssl_v1 as drv  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/bjoc_task_adaptive_ssl_v1.yaml"
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


def _raw_from_spec(atom_types, bonds):
    x = np.asarray(atom_types, dtype=np.int64).reshape(-1, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return SimpleNamespace(
        x=torch.as_tensor(x),
        edge_index=torch.as_tensor(np.asarray(ei, dtype=np.int64).T),
        edge_attr=torch.as_tensor(np.asarray(ea, dtype=np.int64)),
    )


def _tiny_world(n=6):
    # bond types use the real ZINC vocabulary {1, 2, 3}; bond mask token 0 is unused
    specs = [
        ([0, 1, 2, 3], [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 1)]),
        ([4, 5, 6], [(0, 1, 1), (1, 2, 3)]),
        ([7, 7, 8, 9, 10], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 1, 1)]),
        ([0, 0, 0, 0, 0, 0], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)]),
        ([1, 2, 3, 4, 5], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 0, 1)]),
        ([2, 2, 3], [(0, 1, 2), (1, 2, 1)]),
    ]
    atom_specs = [specs[i % len(specs)][0] for i in range(n)]
    bond_specs = [specs[i % len(specs)][1] for i in range(n)]
    objs = [_mol(i, atom_specs[i], bond_specs[i]) for i in range(n)]
    raws = [_raw_from_spec(atom_specs[i], bond_specs[i]) for i in range(n)]
    return objs, raws


def _cfg():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# frozen definitions
# ---------------------------------------------------------------------------


def test_config_and_constants_agree():
    cfg = _cfg()
    assert cfg["protocol"] == tat.PROTOCOL_ID == drv.PROTOCOL_ID
    assert cfg["official_test"] == "never_read"
    assert cfg["split"]["salt"] == tat.SPLIT_SALT
    assert cfg["split"]["fingerprints"] == tat.SPLIT_FINGERPRINTS
    assert cfg["split"]["fingerprints"]["fit_sha256_rows"] == (
        "9468656affcc2b7c69409f03698b1b6287b3a42dc6e164efd19dfb0190254525"
    )
    assert cfg["candidate"]["schedule"]["rule"].startswith("L_t = L_property + 0.1")
    assert cfg["candidate"]["schedule"]["aux_updates"] == 5670
    assert cfg["candidate"]["schedule"]["lambda"] == tat.AUX_LAMBDA == 0.1
    assert cfg["candidate"]["aux_mask"]["atom_ratio"] == mp.ATOM_MASK_RATIO
    assert cfg["candidate"]["aux_task"].startswith("masked atom-type CE only")
    assert cfg["candidate"]["soup"].startswith("parameter mean of the last 5")
    assert cfg["baseline"]["recipe"]["epochs"] == bp.RECIPES["R3"]["epochs"] == 360
    assert cfg["monitor_gate"]["bootstrap"]["n_resamples"] == bp.BOOTSTRAP_N
    assert cfg["budget"]["gpu_hours_max"] == 3.0
    assert "TASSL_GAIN_PROMISING" in cfg["monitor_gate"]["decision_vocabulary"]
    assert "TASSL_INVALID" in cfg["monitor_gate"]["decision_vocabulary"]
    assert cfg["bans"][0] == "official test access in any form"
    assert tat.CONTROL_UPDATES == 28350
    assert tat.CONTROL_EPOCHS * (8000 // bp.BATCH_SIZE + (8000 % bp.BATCH_SIZE > 0)) == 28350
    assert cfg["compute_control"]["purchase_rule"].startswith("run only if")


def test_atom_head_matches_pretext_architecture():
    cfg = src.BJOCConfig()
    head = tat.build_atom_head(cfg)
    assert isinstance(head, src._MLP)
    assert head.lins[0].in_features == 4 * cfg.atom_out
    assert head.lins[-1].out_features == sro.ATOM_CATEGORIES
    assert sum(p.numel() for p in head.parameters()) == tat.atom_head_parameters(cfg)


# ---------------------------------------------------------------------------
# frozen-loop replay
# ---------------------------------------------------------------------------


def test_adaptive_loop_without_aux_matches_frozen_loop():
    objs, _ = _tiny_world()
    rows = list(range(len(objs)))
    y = np.asarray([0.5 * i - 1.0 for i in range(len(objs))], dtype=np.float64)
    kw = dict(
        seed=0,
        recipe_key="R3",
        init="scratch",
        train_objs=objs,
        rows=rows,
        y_fit=y,
        norm=NORM,
        device=torch.device("cpu"),
        smoke_epochs=2,
    )
    a = bp.train_supervised(arm="a", **kw)
    b = tat.train_task_adaptive(arm="b", aux_every=None, **kw)
    assert tat.state_sha256(a["soup_state"]) == tat.state_sha256(b["soup_state"])
    assert a["batch_order"]["full_hash"] == b["batch_order"]["full_hash"]
    assert a["init_full_sha256"] == b["init_full_sha256"]
    for ca, cb in zip(a["lr_curve"], b["lr_curve"]):
        assert ca["train_mae"] == cb["train_mae"]
        assert ca["lr_first"] == cb["lr_first"]
        assert ca["lr_last"] == cb["lr_last"]
    assert a["soup_fit_mae"] == b["soup_fit_mae"]


# ---------------------------------------------------------------------------
# SSL wrapper
# ---------------------------------------------------------------------------


def test_compare_loops_noise_floor_instrument():
    # The A1 amendment: on a non-bitwise-deterministic machine the new loop must
    # stay within the frozen loop's own run-to-run noise floor.
    curve = [{"train_mae": 1.0, "lr_first": 1.0e-3, "lr_last": 1.0e-5}]
    base = {"w": torch.zeros(8)}
    tiny = {"w": torch.full((8,), 5.0e-7)}
    far = {"w": torch.full((8,), 5.0e-3)}

    def payload(state, soup_mae=0.0):
        return {
            "soup_state": state,
            "batch_order": {"full_hash": "order"},
            "lr_curve": curve,
            "soup_members": [1],
            "soup_fit_mae": soup_mae,
            "updates": 1,
        }

    ok = drv._compare_loops(payload(base), payload(tiny), payload(tiny))
    assert ok["soup_bit_identical"] is False
    assert ok["bp_repeat_max_abs_diff"] == pytest.approx(5.0e-7)
    assert ok["equivalent_within_noise"] is True
    assert ok["equivalence_tolerance"] > ok["new_vs_bp_max_abs_diff"]

    bad = drv._compare_loops(payload(base), payload(far), payload(tiny))
    assert bad["equivalent_within_noise"] is False

    identical = drv._compare_loops(payload(base), payload(base), payload(base))
    assert identical["soup_bit_identical"] is True
    assert identical["equivalent_within_noise"] is True


def test_ssl_wrapper_captures_head_without_changing_encoders():
    objs, raws = _tiny_world()
    rows = list(range(len(objs)))
    a = bp.pretrain_masked(
        seed=0, fit_objs=objs, fit_rows=rows, raw_dataset=raws, device=torch.device("cpu"),
        mode="atom_bond", smoke_updates=3, log_fn=lambda m: None,
    )
    b = tat.pretrain_masked_with_head(
        seed=0, fit_objs=objs, fit_rows=rows, raw_dataset=raws, device=torch.device("cpu"),
        mode="atom_bond", smoke_updates=3, log_fn=lambda m: None,
    )
    assert a["encoder_final_sha256"] == b["encoder_final_sha256"]
    assert a["loss_at_steps"] == b["loss_at_steps"]
    assert b["labels_read"] is False
    assert b["atom_head_parameters"] == tat.atom_head_parameters()
    assert set(b["atom_head_state"]) == set(tat.build_atom_head().state_dict())
    assert tat.state_sha256(b["atom_head_state"]) == b["atom_head_sha256"]


# ---------------------------------------------------------------------------
# auxiliary branch
# ---------------------------------------------------------------------------


def test_atom_pretext_logits_identical_to_pretext_model():
    objs, raws = _tiny_world()
    backbone = src.build_bjoc(
        src.BJOCConfig(), seed=3, organization_mode=src.NULL,
        global_mean=NORM["global_mean"], global_scale=NORM["global_scale"],
    )
    pretext = mp.PretextModel(backbone)
    head = tat.build_atom_head()
    head.load_state_dict(pretext.atom_head.state_dict())
    sample = mp.build_masked_sample(objs[0], raws[0].x.numpy(), raws[0].edge_index.numpy(), raws[0].edge_attr.numpy(), seed=1, visit_index=0)
    batch, _, _, _, _ = mp.batch_pretext_samples([sample])
    with torch.no_grad():
        ref = pretext(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])["atom_logits"]
        new = tat.atom_pretext_logits(backbone, batch, batch["masked_atom_ids"], head)
    assert torch.equal(ref, new)


def test_aux_branch_single_step_and_head_gradient():
    objs, raws = _tiny_world()
    rows = list(range(len(objs)))
    y = np.asarray([0.3 * i - 0.7 for i in range(len(objs))], dtype=np.float64)
    ssl = tat.pretrain_masked_with_head(
        seed=0, fit_objs=objs, fit_rows=rows, raw_dataset=raws, device=torch.device("cpu"),
        mode="atom_bond", smoke_updates=2, log_fn=lambda m: None,
    )
    out = tat.train_task_adaptive(
        arm="TA", seed=0, recipe_key="R3", init="ssl", train_objs=objs, rows=rows, y_fit=y,
        norm=NORM, device=torch.device("cpu"), ssl_state=ssl["encoder_state"],
        atom_head_state=ssl["atom_head_state"], aux_every=2, aux_lambda=0.1,
        raw_dataset=raws, smoke_epochs=1, log_fn=lambda m: None,
    )
    assert out["head_info"]["inference_parameters"] == bp.EXPECTED_PARAMETERS
    assert out["head_info"]["training_parameters"] == bp.EXPECTED_PARAMETERS + tat.atom_head_parameters()
    assert out["aux_stats"]["aux_updates"] == (out["updates"] + 1) // 2
    assert out["aux_stats"]["aux_ce_first"] is not None and np.isfinite(out["aux_stats"]["aux_ce_first"])
    assert out["anomalies"]["aux_head_grad_norm_max"] > 0
    assert out["batch_order"]["full_matches"]
    assert tat.state_sha256(out["final_atom_head_state"]) != out["head_info"]["atom_head_sha256"]
    assert "atom_head" not in out["soup_state"]


def test_gradient_diagnostics_head_isolated_from_property():
    objs, raws = _tiny_world()
    rows = list(range(len(objs)))
    y = np.asarray([0.1 * i for i in range(len(objs))], dtype=np.float64)
    ssl = tat.pretrain_masked_with_head(
        seed=0, fit_objs=objs, fit_rows=rows, raw_dataset=raws, device=torch.device("cpu"),
        mode="atom_bond", smoke_updates=2, log_fn=lambda m: None,
    )
    diag = tat.gradient_diagnostics(
        seed=0, train_objs=objs, rows=rows, y_fit=y, norm=NORM, device=torch.device("cpu"),
        raw_dataset=raws, ssl_state=ssl["encoder_state"], atom_head_state=ssl["atom_head_state"],
        n_batches=2, batch_size=3, visit_index=0,
    )
    assert diag["all_finite"]
    assert diag["head_receives_atom_gradient"]
    assert diag["head_grad_on_property_is_none_all"]
    assert diag["mean_grad_norm_ratio_atom_over_property"] > 0
    for rec in diag["per_batch"]:
        assert set(rec["per_group_property"]) == {"atom_encoder", "bond_encoder", "junction_encoder"}
        assert set(rec["per_group_atom"]) == {"atom_encoder", "bond_encoder", "junction_encoder"}


# ---------------------------------------------------------------------------
# audits / invariance
# ---------------------------------------------------------------------------


def test_mask_swap_and_rebuild_audits_on_tiny_molecules():
    objs, raws = _tiny_world(n=8)
    raw_rows = [
        (int(o.mol_index), raws[i].x.numpy(), raws[i].edge_index.numpy(), raws[i].edge_attr.numpy())
        for i, o in enumerate(objs)
    ]
    swap = tat.mask_swap_audit(raw_rows, n_trials=6, positive_cases=2)
    assert swap["verdict"] == "MASK_SWAP_INVARIANT"
    assert swap["atom_swap_passed"] == swap["atom_swap_cases"] > 0
    assert swap["bond_swap_passed"] == swap["bond_swap_cases"] > 0
    rebuild = tat.rebuild_audit(raw_rows, n_cases=4)
    assert rebuild["verdict"] == "MASKED_REBUILD_LEAK_FREE"
    assert rebuild["hidden_truth_absent_from_own_slot"]
    assert rebuild["both_directed_records_masked"]


def test_relabel_and_endpoint_invariance():
    objs, raws = _tiny_world(n=4)
    model = bp.build_bjoc_null(0, NORM, torch.device("cpu"))
    base = bp.predict_model(model, objs, torch.device("cpu"))
    relabeled = tat.relabel_objects(objs, raws, seed=2)
    swapped = tat.endpoint_swap_objects(objs, raws)
    assert np.abs(bp.predict_model(model, relabeled, torch.device("cpu")) - base).max() <= 1.0e-5
    assert np.abs(bp.predict_model(model, swapped, torch.device("cpu")) - base).max() <= 1.0e-5
    inv = tat.batch_invariance(model, objs, torch.device("cpu"))
    assert inv["batch_membership_max_abs_diff"] <= 1.0e-5


# ---------------------------------------------------------------------------
# gate arithmetic / driver discipline
# ---------------------------------------------------------------------------


def test_monitor_gate_mapping_matches_frozen_vocabulary():
    y = np.linspace(-4.0, 4.0, 200)
    ref = {0: y + 0.10, 1: y + 0.12}
    good = {0: y + 0.10 - 0.005, 1: y + 0.12 - 0.006}
    contrast = drv.ab.compute_contrast(good, ref, y)
    gate = drv.ab.monitor_gate(contrast, two_seed=True)
    assert gate["passed"]
    bad = {0: y + 0.10 + 0.002, 1: y + 0.12 + 0.001}
    c2 = drv.ab.compute_contrast(bad, ref, y)
    assert drv.ab.monitor_gate(c2, two_seed=True)["passed"] is False
    assert c2["delta_seed_mean"] >= 0


def test_driver_has_no_official_data_path():
    text = (Path(_CODE_DIR) / "run_bjoc_task_adaptive_ssl_v1.py").read_text(encoding="utf-8")
    for token in ("encoded_test", "encoded_valid", "load_official_test", "load_official_valid", 'split="test"'):
        assert token not in text, token
    assert "official_test" in text and "official_valid" in text
    assert text.count('"official_test_loaded": False') >= 3
    assert "locked_read_authorized" in text


def test_module_auxiliary_path_reads_no_labels():
    text = (Path(_CODE_DIR) / "bjoc_task_adaptive_ssl_v1.py").read_text(encoding="utf-8")
    aux_block = text.split("def aux_masked_batch", 1)[1].split("def train_task_adaptive", 1)[0]
    assert "d.x" in aux_block and "d.edge_index" in aux_block and "d.edge_attr" in aux_block
    assert ".y" not in aux_block
    train_block = text.split("def train_task_adaptive", 1)[1].split("def _group_norms", 1)[0]
    assert "raw_dataset[int(o.mol_index)].y" not in train_block
    assert "official_test_loaded" in text


def test_driver_paths_are_inside_results_dir():
    assert drv.OUT.name == "bjoc_task_adaptive_ssl_v1"
    assert drv.SPLIT_TAG == "D"
    assert drv.ARMS == ("P3", "TA")
    assert drv.SEEDS == (0, 1)
