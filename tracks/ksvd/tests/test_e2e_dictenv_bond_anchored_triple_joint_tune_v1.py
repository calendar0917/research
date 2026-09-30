"""Focused tests for ``e2e_dictenv_bond_anchored_triple_joint_tune_v1`` (BATJ-v1).

Local CPU only; the official ZINC test split is never touched.  Covers the
pre-registered checks: the exact frozen/trainable whitelist and parameter
accounting, the warm-start value equality against both source checkpoints, the
online hot-start reproduction of the BAT-v1 soup valid MAE, the online-vs-cached
path agreement, the structure cache cross-check against the BAT-v1 cache, batch
triple mapping safety, the mode/dropout discipline, gradient connectivity and
optimizer-set identity, the frozen-coordinate hash invariance, and the verdict
thresholds.
"""

from __future__ import annotations

import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_bond_anchored_triple_joint_tune_v1 as jt
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_bond_anchored_triple_v1 as bat
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import (
    zinc_bond_anchored_triple_joint_tune_v1_mainline as stages,
)


def _valid_split_and_structure():
    return p1run.load_split("valid"), jt.build_or_load_structure("valid")


def test_parameter_accounting_matches_preregistration() -> None:
    model = jt.build_joint_model(seed=0)
    accounting = jt.configure_trainable(model)
    assert accounting["passed"] is True
    assert accounting["trainable_tensors"] == 34
    assert accounting["trainable_params"] == 82_805
    assert accounting["total_registered_params"] == 103_757
    breakdown = jt.parameter_breakdown(model)
    expected = {
        "edge_binding": 4_944,
        "edge_encoder": 3_920,
        "fusion": 56_478,
        "pair_projection": 768,
        "relation_encoder": 1_104,
        "distance_gate": 80,
        "pair_encoder": 5_328,
        "F": 5_216,
        "reader": 4_967,
    }
    assert {key: value["numel"] for key, value in breakdown["groups"].items()} == expected


def test_sources_load_value_for_value_and_old_reader_is_gone() -> None:
    model = jt.build_joint_model(seed=0)
    sem_state = bat.load_parent_state()
    bat_state = torch.load(jt.BAT_SOUP_PATH, map_location="cpu", weights_only=False)
    state = model.state_dict()
    names = set(dict(model.named_parameters()))
    assert not any(name.startswith("backbone.reader") for name in names)
    assert "reader.0.weight" in names and "F.0.weight" in names
    for key, value in sem_state.items():
        if key.startswith("reader."):
            continue
        assert torch.equal(state[f"backbone.{key}"], value.float())
    for key, value in bat_state.items():
        assert torch.equal(state[key], value.float())


def test_frozen_and_trainable_whitelist_is_exact() -> None:
    model = jt.build_joint_model(seed=0)
    jt.configure_trainable(model)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 34
    for name in trainable:
        assert any(name.startswith(prefix) for prefix in jt.TRAINABLE_PREFIXES)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert not any(name.startswith(prefix) for prefix in jt.TRAINABLE_PREFIXES)
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    assert frozen  # the dictionary / bindings / encoders are present and frozen


def test_fixed_coordinate_hash_ignores_trainable_updates() -> None:
    model = jt.build_joint_model(seed=0)
    jt.configure_trainable(model)
    before = jt.fixed_coordinate_hash(model)["fixed_coordinate_sha256"]
    with torch.no_grad():
        model.backbone.W_E_S.add_(0.25)
        model.reader[0].weight.mul_(0.5)
    after = jt.fixed_coordinate_hash(model)["fixed_coordinate_sha256"]
    assert before == after
    frozen_before = jt.frozen_subset_hash(model)
    with torch.no_grad():
        model.backbone.W_A_S.add_(0.25)
    frozen_after = jt.frozen_subset_hash(model)
    assert frozen_before["frozen_params_sha256"] != frozen_after["frozen_params_sha256"]
    assert frozen_before["fixed_coordinate_sha256"] == frozen_after["fixed_coordinate_sha256"]


def test_hot_start_reproduces_bat_v1_soup_valid_mae() -> None:
    model = jt.build_joint_model(seed=0)
    jt.configure_trainable(model)
    valid_data, valid_structure = _valid_split_and_structure()
    payload = jt.hot_start_check(model, valid_data, valid_structure)
    assert payload["passed"] is True
    assert payload["abs_diff"] <= 1.0e-6


def test_online_path_matches_cached_bat_path() -> None:
    model = jt.build_joint_model(seed=0)
    jt.configure_trainable(model)
    _, valid_structure = _valid_split_and_structure()
    payload = jt.online_vs_cached_check(model, valid_structure, n_graphs=16)
    assert payload["passed"] is True
    assert payload["max_prediction_diff"] <= 2.0e-5
    assert payload["max_z3_diff"] <= 2.0e-5


def test_structure_cache_matches_bat_v1_counts_and_rows() -> None:
    for split in ("train", "valid"):
        structure = jt.build_or_load_structure(split)
        cache = bat.build_or_load_cache(split)
        assert structure.n_pairs == cache.n_pairs
        assert structure.n_triples == cache.n_triples
        assert torch.equal(structure.triple_ij.long(), cache.triple_ij.long())
        assert torch.equal(structure.triple_ik.long(), cache.triple_ik.long())
        assert torch.equal(structure.triple_jk.long(), cache.triple_jk.long())
        audit = jt.structure_audit(structure, brute_force_graphs=8)
        assert audit["passed"] is True


def test_batch_triple_mapping_stays_inside_graphs() -> None:
    valid_data, valid_structure = _valid_split_and_structure()
    payload = jt.batch_mapping_check(valid_structure, valid_data, batch_size=32)
    assert payload["passed"] is True
    assert payload["pair_buckets_match_cache"] is True


def test_shuffled_batch_order_matches_repository_loader() -> None:
    """The new raw-graph index loader reproduces the repository protocol order."""
    valid_data, valid_structure = _valid_split_and_structure()
    seed = 0 + jt.TRAIN_SHUFFLE_OFFSET
    reference = p1.make_env_loader(valid_data, 1000, True, seed)
    batch = next(iter(jt.make_index_loader(valid_data, valid_structure, 1000, True, seed)))
    reference_batch = next(iter(reference))
    assert torch.equal(batch["data"].y, reference_batch.y)
    assert torch.equal(batch["data"].ptr, reference_batch.ptr)
    assert torch.equal(batch["data"].dict_phi, reference_batch.dict_phi)


def test_mode_discipline_keeps_backbone_eval_and_matches_eval_forward() -> None:
    model = jt.build_joint_model(seed=0)
    jt.configure_trainable(model)
    train_data = p1run.load_split("train")
    train_structure = jt.build_or_load_structure("train")
    payload = jt.mode_forward_check(model, train_structure, train_data)
    assert payload["passed"] is True
    assert payload["max_abs_diff_train_vs_eval"] == 0.0
    assert payload["backbone_modules_in_train_mode"] == []
    assert payload["dropout_modules_in_train_mode"] == []


def test_one_batch_gradient_reaches_upstream_and_frozen_stay_empty() -> None:
    model = jt.build_joint_model(seed=0)
    jt.configure_trainable(model)
    train_data = p1run.load_split("train")
    train_structure = jt.build_or_load_structure("train")
    payload = jt.one_batch_gradient_check(model, train_structure, train_data)
    assert payload["passed"] is True
    assert payload["frozen_grads_none"] is True
    assert payload["optimizer_set_equals_whitelist"] is True
    assert payload["group_grad_norms"]["edge_binding"] > 0.0
    assert payload["group_grad_norms"]["edge_encoder"] > 0.0
    assert payload["group_grad_norms"]["fusion"] > 0.0
    assert payload["group_grad_norms"]["pair_encoder"] > 0.0
    assert payload["group_grad_norms"]["F"] > 0.0
    assert payload["group_grad_norms"]["reader"] > 0.0
    assert payload["grads_cleared"] is True


def test_provenance_sources_are_consistent() -> None:
    parent = jt.parent_provenance()
    assert parent["canonical_state_sha256"] == bat.PARENT_SOUP_SHA256
    assert parent["params"] == 97_709
    m1 = jt.m1_soup_provenance()
    assert m1["soup_members"] == list(jt.BAT_SOUP_MEMBERS)
    assert abs(m1["soup_valid_mae"] - jt.M_START_EXPECTED) < 1.0e-12
    assert all(m1["buffers_match_standardizers_json"].values())


def test_verdict_thresholds_are_frozen() -> None:
    perfect = {"all_passed": True}
    assert stages._verdict(0.119, True, perfect, True) == jt.VERDICT_PROMISING
    assert stages._verdict(0.120, True, perfect, True) == jt.VERDICT_PROMISING
    assert stages._verdict(0.1201, True, perfect, True) == jt.VERDICT_NO_SIGNAL
    assert stages._verdict(0.119, False, perfect, True) == jt.VERDICT_INVALID
    assert stages._verdict(0.119, True, {"all_passed": False}, False) == jt.VERDICT_INVALID
