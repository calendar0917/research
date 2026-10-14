"""Correctness tests for cscl-bjoc-cross-task-lipophilicity-v1.

These tests are deliberately RDKit-free: Lipophilicity structures are consumed
from the committed label-free raw-graph artifact, so the suite runs in the
GPU/torch environment as well as locally.  Anything that needs the official
CSV is exercised by the driver's ``prepare-dataset`` command instead.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

TRACK = Path(__file__).resolve().parents[1]
CODE = TRACK / "code"
OUT = TRACK / "results/bjoc_cross_task_lipophilicity_v1"
CONFIG_PATH = TRACK / "configs" / "bjoc_cross_task_lipophilicity_v1.yaml"
RAW_PATH = OUT / "dataset" / "lipo_raw_v1.json.gz"
SPLIT_PATH = OUT / "split_lipo_v1.json"
VOCAB_PATH = OUT / "vocab_v1.json"
REFERENCE_PATH = OUT / "upstream" / "deepchem_2.8.0_reference_split.json"

if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import bjoc_lipophilicity_adapter_v1 as lip  # noqa: E402
import run_bjoc_cross_task_lipophilicity_v1 as driver  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _raws():
    if not RAW_PATH.exists():
        pytest.skip("raw graph artifact missing; run prepare-dataset")
    raws, meta = lip.load_raw_dataset(RAW_PATH)
    return raws, meta


def _objects(raws, rows):
    return [lip.build_lipo_objects(raws[int(r)]) for r in rows]


def _model():
    return lip.build_lipo_model(
        lip.LipoConfig(), seed=0, organization_mode="null",
        global_mean=np.zeros(lip.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(lip.GLOBAL_DIM, dtype=np.float32),
        device="cpu",
    )


def _predict(model, objs):
    model.eval()
    with torch.no_grad():
        return model(lip.batch_lipo_objects(objs)).cpu().numpy()


# ---------------------------------------------------------------------------
# protocol / config consistency
# ---------------------------------------------------------------------------


def test_config_matches_module_constants():
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["protocol_id"] == lip.PROTOCOL_ID == driver.PROTOCOL_ID
    assert cfg["dataset"]["sha256"] == "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"
    assert cfg["split"]["expected_sizes"] == [3360, 420, 420]
    assert cfg["model"]["expected_inference_parameters"] == 419713
    assert cfg["supervised"]["epochs"] == driver.EPOCHS == 360
    assert cfg["supervised"]["batch_size"] == driver.BATCH_SIZE == 128
    assert cfg["ssl"]["updates"] == driver.ssl_updates(3360) == 2520
    assert cfg["budget"]["gpu_hours_max"] == 4.0
    assert cfg["promotion_gate"]["conditions"]["relative_rmse_drop_min"] == driver.PROMOTE_REL_RMSE_DROP
    assert cfg["promotion_gate"]["conditions"]["absolute_rmse_drop_min"] == driver.PROMOTE_ABS_RMSE_DROP
    assert tuple(cfg["conditional_confirmation"]["extra_seeds"]) == driver.SEEDS_CONFIRM


def test_schedule_counts_are_preregistered():
    assert driver.ssl_updates(3360) == 2520
    assert driver.EPOCHS * math.ceil(3360 / driver.BATCH_SIZE) == 9720


# ---------------------------------------------------------------------------
# dataset / split provenance
# ---------------------------------------------------------------------------


def test_raw_dataset_is_label_free_with_frozen_schema():
    with gzip.open(RAW_PATH, "rt", encoding="utf-8") as fh:
        payload = json.loads(fh.read())
    expected = {
        "mol_index", "compound_id", "smiles", "atom_class", "atom_charge", "atom_h",
        "atom_chirality", "edge_index", "edge_class", "edge_stereo", "n_components",
    }
    assert len(payload["rows"]) == 4200
    assert {tuple(sorted(r.keys())) for r in payload["rows"]} == {tuple(sorted(expected))}
    assert payload["meta"]["label_free"] is True
    assert "exp" not in {k.lower() for k in payload["meta"]}
    assert payload["meta"]["csv_sha256"] == "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"


def test_split_partition_and_reference_agreement():
    split = _load_json(SPLIT_PATH)
    assert (split["n_train"], split["n_valid"], split["n_test"]) == (3360, 420, 420)
    train = [int(r) for r in split["train_rows"]]
    valid = [int(r) for r in split["valid_rows"]]
    test = [int(r) for r in split["test_rows"]]
    assert sorted(train + valid + test) == list(range(4200))
    assert not (set(train) & set(valid)) and not (set(train) & set(test)) and not (set(valid) & set(test))
    assert len(split["train_y"]) == 3360
    assert "valid_y" not in split and "test_y" not in split
    assert split["canonical_smiles_cross_split"] == 0
    assert split["scaffold_groups_cross_split"] == 0
    assert split["train_sha256_rows"] == driver.bp.rows_sha256(train)
    assert split["valid_sha256_rows"] == driver.bp.rows_sha256(valid)
    assert split["test_sha256_rows"] == driver.bp.rows_sha256(test)
    reference = _load_json(REFERENCE_PATH)
    assert reference["train_idx"] == train
    assert reference["valid_idx"] == valid
    assert reference["test_idx"] == test
    assert reference["deepchem_version"] == "2.8.0"


def test_reference_split_has_no_labels():
    reference = _load_json(REFERENCE_PATH)
    assert not any("y" == k or "label" in k.lower() for k in reference)


# ---------------------------------------------------------------------------
# vocabulary / masks
# ---------------------------------------------------------------------------


def test_vocabulary_slots_and_manifest():
    assert lip.ATOM_UNK == len(lip.ATOM_CLASSES)
    assert lip.ATOM_MASK not in range(len(lip.ATOM_CLASSES)) and lip.ATOM_MASK != lip.ATOM_UNK
    assert lip.BOND_MASK not in range(len(lip.BOND_CLASSES)) and lip.BOND_MASK != lip.BOND_UNK
    for real, unk, mask in (
        (lip.CHARGE_BUCKETS, lip.CHARGE_UNK, lip.CHARGE_MASK),
        (lip.H_BUCKETS, lip.H_UNK, lip.H_MASK),
        (lip.CHIRAL_BUCKETS, lip.CHIRAL_UNK, lip.CHIRAL_MASK),
        (lip.STEREO_BUCKETS, lip.STEREO_UNK, lip.STEREO_MASK),
    ):
        assert mask not in range(len(real)) and mask != unk
    manifest = lip.vocab_manifest()
    assert manifest["atom_classes"] == [[e, aromatic] for e, aromatic in lip.ATOM_CLASSES]
    assert len(manifest["atom_classes"]) == lip.N_ATOM_TARGETS == 15
    assert manifest["n_atom_slots"] == lip.N_ATOM_SLOTS == 17
    assert len(manifest["bond_classes"]) == lip.N_BOND_TARGETS == 4
    assert manifest["n_bond_slots"] == lip.N_BOND_SLOTS == 6
    assert manifest["mask_ratios"] == {"atom": 0.15, "bond": 0.15}
    if VOCAB_PATH.exists():
        assert manifest == _load_json(VOCAB_PATH)


def test_feature_dims_match_config():
    raws, _ = _raws()
    obj = lip.build_lipo_objects(raws[0])
    assert obj.atom_feat.shape[1] == lip.ATOM_FEAT_DIM
    assert obj.bond_feat.shape[1] == lip.BOND_FEAT_DIM
    assert obj.junc_feat.shape[1] == lip.JUNC_FEAT_DIM
    assert obj.global_feat.shape[0] == lip.GLOBAL_DIM
    config = lip.LipoConfig()
    assert config.atom_in == lip.ATOM_FEAT_DIM
    assert config.bond_in == 3 * config.atom_out + lip.BOND_FEAT_DIM
    assert config.junc_in == 4 * config.atom_out + 2 * config.bond_out + lip.JUNC_FEAT_DIM
    assert config.reader_in == config.compose_out + 2 * config.atom_out + lip.GLOBAL_DIM


# ---------------------------------------------------------------------------
# object construction invariants
# ---------------------------------------------------------------------------


def test_object_build_is_deterministic():
    raws, _ = _raws()
    a = lip.build_lipo_objects(raws[17])
    b = lip.build_lipo_objects(raws[17])
    assert driver._objects_equal(a, b)


def test_relabelling_and_endpoint_swap_are_invariant():
    raws, _ = _raws()
    raw = raws[17]
    model = _model()
    base = lip.build_lipo_objects(raw)
    perm = np.random.default_rng(0).permutation(raw.n_atoms)
    relabelled = lip.build_lipo_objects(driver._relabel_raw(raw, perm))
    swapped = lip.build_lipo_objects(driver._endpoint_swap_raw(raw))
    preds = _predict(model, [base, relabelled, swapped])
    assert np.abs(preds[1] - preds[0]).max() <= 1e-5
    assert np.abs(preds[2] - preds[0]).max() <= 1e-5


def test_masked_rebuild_hides_hidden_truth():
    raws, _ = _raws()
    raw = raws[42]
    sample = lip.build_masked_sample(raw, seed=0, visit_index=3)
    assert len(sample.atom_ids) >= 1
    # every masked atom slot must carry the mask token in every raw field
    masked_raw = lip.mask_raw(raw, sample.atom_ids, sample.bond_ids)
    assert (masked_raw.atom_class[sample.atom_ids] == lip.ATOM_MASK).all()
    assert (masked_raw.atom_charge[sample.atom_ids] == lip.CHARGE_MASK).all()
    assert (masked_raw.atom_h[sample.atom_ids] == lip.H_MASK).all()
    assert (masked_raw.atom_chirality[sample.atom_ids] == lip.CHIRAL_MASK).all()
    # hidden-truth swap: changing the removed classes cannot change the input
    altered = driver._copy_raw(raw)
    altered.atom_class[sample.atom_ids] = (altered.atom_class[sample.atom_ids] + 1) % lip.N_ATOM_TARGETS
    other = lip.build_masked_sample(altered, seed=0, visit_index=3)
    assert np.array_equal(sample.atom_ids, other.atom_ids)
    assert driver._objects_equal(sample.masked, other.masked)
    assert not np.array_equal(sample.atom_classes, other.atom_classes)


def test_supervised_raw_graphs_never_contain_mask_tokens():
    split = _load_json(SPLIT_PATH)
    raws, _ = _raws()
    for zone in ("train", "valid", "test"):
        for row in split[f"{zone}_rows"][:200]:
            raw = raws[int(row)]
            assert int(raw.atom_class.max()) <= lip.ATOM_UNK
            assert int(raw.edge_class.max()) <= lip.BOND_UNK


# ---------------------------------------------------------------------------
# model / initialisation
# ---------------------------------------------------------------------------


def test_parameter_budget_and_no_forbidden_modules():
    import structural_relation_composer_v1 as src

    model = _model()
    audit = src.parameter_audit(lip.LipoConfig(), model)
    assert audit["parameter_exact"] is True
    assert audit["actual_parameters"] == 419713
    assert src.forbidden_module_scan(model) == []
    assert not hasattr(model, "atom_head") and not hasattr(model, "bond_head")


def test_init_is_seeded_and_scratch_ssl_share_it():
    import structural_relation_composer_v1 as src

    a = _model()
    b = _model()
    assert src.state_dict_hash(a) == src.state_dict_hash(b)
    assert driver.bp.encoder_group_hash(a) == driver.bp.encoder_group_hash(b)
    c = lip.build_lipo_model(
        lip.LipoConfig(), seed=1, organization_mode="null",
        global_mean=np.zeros(lip.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(lip.GLOBAL_DIM, dtype=np.float32), device="cpu",
    )
    assert src.state_dict_hash(c) != src.state_dict_hash(a)


def test_pretext_model_only_adds_heads_and_shares_encoders():
    model = _model()
    pretext = lip.LipoPretextModel(model)
    assert pretext.backbone is model
    assert pretext.backbone.atom_encoder is model.atom_encoder
    head_params = [n for n, _ in pretext.named_parameters() if n.startswith(("atom_head", "bond_head"))]
    assert head_params
    infer_params = {n for n, _ in model.named_parameters()}
    assert not any(n.startswith(("atom_head", "bond_head")) for n in infer_params)


def test_ssl_state_transfer_touches_encoders_only():
    import bjoc_masked_pretraining_v1 as mp

    model = _model()
    composer_hash = mp.group_state_hash(model)
    state = mp.encoder_state(model)
    for group in state.values():
        for key in group:
            group[key] = group[key] + 1.0
    mp.load_encoder_state(model, state)
    assert mp.group_state_hash(model) == composer_hash
    assert driver.bp.encoder_group_hash(model) != driver.bp.encoder_group_hash(_model())


# ---------------------------------------------------------------------------
# metrics / promotion rule
# ---------------------------------------------------------------------------


def test_metrics_on_known_case():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    pred = np.array([0.0, 1.0, 0.0, 1.0])
    out = driver._metrics(pred, y)
    assert out["rmse"] == pytest.approx(math.sqrt(0.5))
    assert out["mae"] == pytest.approx(0.5)
    assert out["median_abs"] == pytest.approx(0.5)
    assert out["n"] == 4


def test_promotion_gate_accepts_a_clean_gain():
    per_seed = {"0": {"S": {"rmse": 0.80, "mae": 0.55}, "P": {"rmse": 0.75, "mae": 0.52}},
                "1": {"S": {"rmse": 0.82, "mae": 0.56}, "P": {"rmse": 0.76, "mae": 0.53}}}
    seed_mean = {"S": {"rmse": 0.81, "mae": 0.555}, "P": {"rmse": 0.755, "mae": 0.525}}
    boot = {"delta": -0.055, "ci95": [-0.09, -0.02]}
    diagnostics = {"top1_improvement_share": 0.05, "top10_improvement_share": 0.2}
    verdict = driver.evaluate_promotion(per_seed, seed_mean, boot, diagnostics,
                                        audits_pass=True, baselines_done=True)
    assert verdict["decision"] == "TRANSFER_SSL_GAIN_PROMISING"
    assert verdict["test_authorized"] is True
    assert verdict["seeds_to_confirm"] == [2, 3, 4]


def test_promotion_gate_rejects_small_or_unconfirmed_gains():
    base_seed = {"S": {"rmse": 0.80, "mae": 0.55}, "P": {"rmse": 0.792, "mae": 0.548}}
    seed_mean = {"S": {"rmse": 0.80, "mae": 0.55}, "P": {"rmse": 0.792, "mae": 0.548}}
    diagnostics = {"top1_improvement_share": 0.05, "top10_improvement_share": 0.2}
    verdict = driver.evaluate_promotion({"0": base_seed, "1": base_seed}, seed_mean,
                                        {"delta": -0.008, "ci95": [-0.03, 0.005]}, diagnostics,
                                        audits_pass=True, baselines_done=True)
    assert verdict["conditions"]["relative_rmse_drop_min"] is False
    assert verdict["decision"] == "TRANSFER_SSL_PROMISING_UNCONFIRMED"
    assert verdict["test_authorized"] is False


def test_promotion_gate_rejects_harm_and_outlier_driven_gains():
    per_seed = {"0": {"S": {"rmse": 0.80, "mae": 0.55}, "P": {"rmse": 0.83, "mae": 0.57}},
                "1": {"S": {"rmse": 0.80, "mae": 0.55}, "P": {"rmse": 0.81, "mae": 0.56}}}
    seed_mean = {"S": {"rmse": 0.80, "mae": 0.55}, "P": {"rmse": 0.82, "mae": 0.565}}
    diagnostics = {"top1_improvement_share": 0.05, "top10_improvement_share": 0.2}
    verdict = driver.evaluate_promotion(per_seed, seed_mean, {"delta": 0.02, "ci95": [0.005, 0.04]},
                                        diagnostics, audits_pass=True, baselines_done=True)
    assert verdict["decision"] == "TRANSFER_SSL_NO_CONFIRMED_GAIN"
    assert verdict["test_authorized"] is False

    good_seed = {"S": {"rmse": 0.80, "mae": 0.55}, "P": {"rmse": 0.74, "mae": 0.52}}
    verdict = driver.evaluate_promotion(
        {"0": good_seed, "1": good_seed}, {"S": {"rmse": 0.80, "mae": 0.55}, "P": {"rmse": 0.74, "mae": 0.52}},
        {"delta": -0.06, "ci95": [-0.1, -0.02]},
        {"top1_improvement_share": 0.9, "top10_improvement_share": 0.95},
        audits_pass=True, baselines_done=True,
    )
    assert verdict["conditions"]["not_single_outlier_driven"] is False
    assert verdict["decision"] == "TRANSFER_SSL_NO_CONFIRMED_GAIN"

    # a crossing CI always maps to the unconfirmed label, even with a mixed sign
    verdict = driver.evaluate_promotion(per_seed, seed_mean, {"delta": 0.01, "ci95": [-0.02, 0.03]},
                                        diagnostics, audits_pass=True, baselines_done=True)
    assert verdict["decision"] == "TRANSFER_SSL_PROMISING_UNCONFIRMED"


def test_bootstrap_is_deterministic_and_zero_for_identical_arms():
    split = _load_json(SPLIT_PATH)
    n = split["n_valid"] if "n_valid" in split else 420
    rng = np.random.default_rng(0)
    y = rng.normal(2.0, 1.0, size=n)
    pred = y + rng.normal(0, 0.1, size=n)
    preds = {("S", 0): pred, ("P", 0): pred}
    out = driver._paired_bootstrap_mean(preds, [0], y, "rmse")
    assert out["delta"] == pytest.approx(0.0, abs=1e-12)
    assert out["ci95"][0] == pytest.approx(0.0, abs=1e-12)
    assert out["ci95"][1] == pytest.approx(0.0, abs=1e-12)
    again = driver._paired_bootstrap_mean(preds, [0], y, "rmse")
    assert out == again


def test_inverse_transform_check():
    result = driver._inverse_and_metric_check()
    assert result["match"] is True


# ---------------------------------------------------------------------------
# label-isolation discipline in the driver source
# ---------------------------------------------------------------------------


def test_driver_blocks_score_labels_and_test_reads():
    source = (CODE / "run_bjoc_cross_task_lipophilicity_v1.py").read_text(encoding="utf-8")
    assert source.count("assert_score_labels_absent()") >= 4
    assert "validation_decision.json" in source
    assert "test_authorized" in source
    assert "read_label_vault" in source
    # test prediction must be gated before any test row is touched
    predict_body = source.split("def cmd_predict")[1].split("def _pred_denormalizer")[0]
    assert "test_authorized" in predict_body


def test_freeze_binds_code_and_artifacts():
    freeze = driver.read_json(driver.FREEZE)
    assert freeze["protocol"] == lip.PROTOCOL_ID
    assert freeze["config_sha256"] == driver.file_sha256(driver.CONFIG_PATH)
    for name, digest in freeze["artifact_sha256"].items():
        path = driver.OUT / freeze["artifact_paths"][name]
        assert path.exists(), path
        assert driver.file_sha256(path) == digest
    assert freeze["n_train"] == 3360 and freeze["n_valid"] == 420 and freeze["n_test"] == 420
    assert freeze["ssl_updates"] == 2520
    assert freeze["supervised_updates"] == 9720
    assert freeze["official_valid_loaded"] is False and freeze["official_test_loaded"] is False
