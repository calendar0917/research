"""Protocol requirement 4: training/validation entry points cannot read Test labels."""

from __future__ import annotations

import numpy as np
import pytest


def test_train_valid_labels_reject_test(labels):
    import r42_labels as L

    assert labels.y_train.shape == (3360,)
    assert labels.y_valid.shape == (420,)
    with pytest.raises(L.TestLabelAccessError):
        labels.get("test")
    with pytest.raises(L.TestLabelAccessError):
        labels["test"]
    with pytest.raises(KeyError):
        labels.get("reserved")


def test_assert_train_valid_only_rejects_mappings_with_test(labels):
    import r42_labels as L

    L.assert_train_valid_only(labels)
    with pytest.raises(L.TestLabelAccessError):
        L.assert_train_valid_only({"train": labels.y_train, "valid": labels.y_valid, "test": np.zeros(420)})
    with pytest.raises(L.TestLabelAccessError):
        L.assert_train_valid_only(labels.y_train)


def test_label_artifact_has_no_test_values():
    import r42_common as C

    payload = C.read_json(C.LABELS_TRAIN_VALID)
    assert payload["scope"] == "train_and_valid_only"
    assert payload["test_labels_included"] is False
    assert set(payload["rows"]) == {"train", "valid"}
    assert set(payload["values"]) == {"train", "valid"}
    text = C.LABELS_TRAIN_VALID.read_text(encoding="utf-8")
    assert '"test"' not in text


def test_trainers_require_train_valid_labels_object(split):
    import torch
    import r42_common as C
    import r42_lcrc_train as LT
    import r42_native_train as NT
    import r42_labels as L

    with pytest.raises(L.TestLabelAccessError):
        LT.train_lcrc(cfg=C.load_config()["models"]["lcrc"], split=split, labels={"train": [0.0]}, device=torch.device("cpu"), run_dir=C.RUNS / "t")
    with pytest.raises(L.TestLabelAccessError):
        NT.train_native(kind="pna", cfg=C.load_config()["models"]["pna"], split=split, labels={"train": [0.0]}, device=torch.device("cpu"), run_dir=C.RUNS / "t")


def test_xgb_requires_train_valid_labels_object(split):
    import r42_common as C
    import r42_labels as L
    import r42_xgb as X

    with pytest.raises(L.TestLabelAccessError):
        X.train_xgb(C.load_config()["models"]["xgb"], split, {"train": np.zeros(3360)}, C.RUNS / "t_xgb")


def test_test_vault_refuses_without_manifest(tmp_path):
    import r42_labels as L

    vault = L.TestLabelVault(tmp_path / "missing.json", tmp_path / "lock.json")
    with pytest.raises(PermissionError):
        vault.read(expected_manifest_sha256="0" * 64)


def test_test_vault_is_one_shot(tmp_path):
    import r42_common as C
    import r42_labels as L

    manifest = tmp_path / "pre_test_manifest.json"
    C.write_json(manifest, {"protocol_id": C.PROTOCOL_ID, "dummy": True})
    digest = C.sha256_file(manifest)
    lock = tmp_path / "test_read.lock"
    vault = L.TestLabelVault(manifest, lock)
    y = vault.read(expected_manifest_sha256=digest)
    assert y.shape == (420,)
    vault.write_lock({"protocol_id": C.PROTOCOL_ID})
    with pytest.raises(PermissionError):
        L.TestLabelVault(manifest, lock).read(expected_manifest_sha256=digest)
    assert lock.exists()


def test_split_excludes_nothing_and_keeps_all_rows(split):
    import r42_common as C

    union = set(split["rows"]["train"]) | set(split["rows"]["valid"]) | set(split["rows"]["test"])
    assert union == set(range(C.N_MOLECULES))
    assert C.MULTI_COMPONENT_ROW in union
