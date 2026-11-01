"""Source and split integrity tests (no training, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

import mb_common as C
import mb_source as S


def test_source_manifest_is_hash_verified():
    manifest = S.load_source_manifest(verify=True)
    assert manifest["upstream"]["commit"] == S.CHEMBENCH_COMMIT
    assert len(manifest["files"]) == 16
    for key in C.DATASET_ORDER:
        entry = manifest["datasets"][key]
        assert entry["selected_repeat_index"] == 0
        assert entry["selected_repeat_file_index"] == 122


def test_manifest_detects_tampering(tmp_path, monkeypatch):
    import shutil

    root = tmp_path / "chembench"
    shutil.copytree(C.CHEMBENCH_DIR, root)
    path = root / "ESOL/ESOL.csv.gz"
    path.write_bytes(path.read_bytes() + b"\x00")
    monkeypatch.setattr(C, "CHEMBENCH_DIR", root)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        S.load_source_manifest(verify=True)


@pytest.mark.parametrize("key", C.DATASET_ORDER)
def test_published_split_partitions_the_sample(key):
    src = S.load_dataset(key, labels="train_valid")
    rows = src["rows"]
    n = len(src["smiles"])
    assert n == C.DATASETS[key]["expected_rows"]
    parts = [set(rows["train"]), set(rows["valid"]), set(rows["test"])]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert len(parts[0] | parts[1] | parts[2]) == n
    assert [len(parts[0]), len(parts[1]), len(parts[2])] == list(C.DATASETS[key]["expected_split_sizes"])
    assert set(src["y"]) == {"train", "valid"}


def test_repeat_zero_is_the_lexicographically_first_file():
    src = S.load_dataset("ESOL", labels="none")
    assert src["selected_repeat"]["file"].endswith("122.ind.pkl")
    assert src["selected_repeat"] == src["repeats"][0]
    # repeats are genuinely different for the random splits
    assert len({r["rows_sha256"][0] for r in src["repeats"]}) == 3


def test_bace_repeats_are_identical_scaffold_files():
    src = S.load_dataset("BACE", labels="none")
    assert len({r["sha256"] for r in src["repeats"]}) == 1


def test_no_test_labels_are_decoded_when_labels_none():
    src = S.load_dataset("ESOL", labels="none")
    assert src["y"] is None
    assert list(src["dataframe"].columns) == ["smiles"]


def test_train_valid_labels_scope_excludes_test():
    src = S.load_dataset("BACE", labels="train_valid")
    assert set(src["y"]) == {"train", "valid"}
    assert len(src["y"]["train"]) == 1210
    assert len(src["y"]["valid"]) == 151


def test_row_hash_is_stable():
    src = S.load_dataset("FreeSolv", labels="none")
    assert C.row_sha256(src["rows"]["train"]) == src["selected_repeat"]["rows_sha256"][0]
    assert np.asarray(src["rows"]["test"]).dtype == np.int64
