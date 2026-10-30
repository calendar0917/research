from __future__ import annotations

import pytest

import adapters as A
import lcrc_common as C
import zinc_legacy as Z


pytestmark = pytest.mark.skipif(
    not (C.P3_TRAIN.exists() and C.LEGACY_CKPT.exists() and C.LEGACY_NORM.exists()),
    reason="frozen ZINC P3 objects/checkpoint/norm are not all available",
)


def test_legacy_p3_input_replay_sample(legacy_spec):
    report = Z.compare_p3_objects(legacy_spec, "train", max_molecules=32, progress_every=0)
    assert report["status"] == "PASS"
    assert report["all_fields_exact"] is True
    assert report["n_molecules"] == 32
    assert report["first_mismatch"] is None


def test_legacy_collate_replay_sample(legacy_spec):
    report = Z.compare_collate(legacy_spec, n_molecules=16)
    assert report["status"] == "PASS"
    assert report["all_fields_exact"] is True
    assert not any(v["exact"] is False for v in report["fields"].values())


def test_legacy_checkpoint_cpu_replay_sample(legacy_spec):
    norm = A.load_legacy_norm()
    report = Z.replay_frozen_checkpoint(legacy_spec, norm, n_molecules=16)
    assert report["status"] == "PASS"
    assert report["state_keys_equal"] is True
    assert report["parameters_old"] == report["parameters_new"] == 237519
    assert report["state_hash_old"] == report["state_hash_new"]
    assert report["decomposition_max_abs_error"] <= 1e-6
    assert report["training"] is False
    assert report["optimizer_steps"] == 0
    assert report["labels_read"] is False
    assert all(v["exact"] for v in report["fields"].values())


def test_legacy_spec_rejects_disconnected_construction_input(legacy_spec):
    # The legacy schema is connected-only; its policy is explicit rather than
    # silently coercing a disconnected molecule.
    assert legacy_spec.supports_disconnected is False
    assert "reject" in legacy_spec.disconnected_policy.lower()
