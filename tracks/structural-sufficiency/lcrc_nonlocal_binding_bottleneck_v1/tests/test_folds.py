"""Split integrity: scaffold grouping, no leakage, determinism, feasibility."""

from __future__ import annotations

import numpy as np
import pytest

import nb_common as C
import nb_data as D


@pytest.fixture(scope="module")
def train():
    data = D.load_train_only()
    scaf = D.scaffolds(data["smiles"])
    y = np.asarray(data["labels"], dtype=np.int64)
    folds = D.build_folds(y, scaf["codes"])
    return {"data": data, "scaf": scaf, "y": y, "folds": folds}


def test_official_train_size_and_label_balance(train):
    audit = train["data"]["audit"]
    assert audit["n_train"] == C.OGB_TRAIN_ROWS == 32901
    assert audit["rows_match_frozen_benchmark"] is True
    assert set(np.unique(train["y"]).tolist()) == {0, 1}
    assert audit["n_positive"] + audit["n_negative"] == audit["n_train"]
    assert audit["valid_test_index_files_opened"] is False
    assert audit["valid_test_labels_read"] is False


def test_scaffold_groups(train):
    scaf = train["scaf"]
    assert scaf["n_groups"] > 1000
    assert all(g != "" for g in scaf["groups"])
    assert set(scaf["kind_counts"]) <= {"murcko", "canonical_smiles_fallback"}
    assert scaf["kind_counts"].get("murcko", 0) > 30000


def test_folds_are_feasible_and_partition_the_train_set(train):
    folds = train["folds"]
    assert folds["status"] == "OK"
    assert folds["feasible"] is True
    assert folds["outer_valid_partition_covers_all_rows"] is True
    seen: list[int] = []
    for f in folds["folds"]:
        assert f["train"]["n_classes"] == 2
        assert f["valid"]["n_classes"] == 2
        assert f["valid"]["n_positive"] > 0
        seen.extend(f["valid_rows"])
    assert sorted(seen) == list(range(train["y"].size))
    assert len(seen) == len(set(seen))


def test_no_scaffold_crosses_any_outer_or_inner_boundary(train):
    codes = train["scaf"]["codes"]
    for f in train["folds"]["folds"]:
        tr = set(np.unique(codes[np.asarray(f["train_rows"])]).tolist())
        va = set(np.unique(codes[np.asarray(f["valid_rows"])]).tolist())
        assert tr.isdisjoint(va), f["outer_fold"]
        for inner in f["inner"]:
            itr = set(np.unique(codes[np.asarray(inner["train_rows"])]).tolist())
            iva = set(np.unique(codes[np.asarray(inner["valid_rows"])]).tolist())
            assert itr.isdisjoint(iva), (f["outer_fold"], inner["inner_fold"])
            assert itr | iva == tr  # inner split is strictly inside the outer Train


def test_inner_folds_cover_the_outer_train_exactly(train):
    for f in train["folds"]["folds"]:
        outer_train = set(f["train_rows"])
        union: set[int] = set()
        for inner in f["inner"]:
            union |= set(inner["train_rows"]) | set(inner["valid_rows"])
        assert union == outer_train


def test_fold_records_are_hashed_consistently(train):
    for f in train["folds"]["folds"]:
        assert C.rows_sha256(sorted(f["train_rows"])) == f["train"]["index_sha256"]
        assert C.rows_sha256(sorted(f["valid_rows"])) == f["valid"]["index_sha256"]
        assert f["inner_seed"] == C.SPLIT_SEED + f["outer_fold"]


def test_fold_construction_is_deterministic(train):
    again = D.build_folds(train["y"], train["scaf"]["codes"])
    for a, b in zip(train["folds"]["folds"], again["folds"]):
        assert a["train"]["index_sha256"] == b["train"]["index_sha256"]
        assert a["valid"]["index_sha256"] == b["valid"]["index_sha256"]
        for ia, ib in zip(a["inner"], b["inner"]):
            assert ia["train"]["index_sha256"] == ib["train"]["index_sha256"]
            assert ia["valid"]["index_sha256"] == ib["valid"]["index_sha256"]


def test_different_seed_gives_a_different_partition(train):
    from sklearn.model_selection import StratifiedGroupKFold

    codes = train["scaf"]["codes"]
    other = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=C.SPLIT_SEED + 1)
    folds = list(other.split(np.zeros((codes.size, 1)), train["y"], codes))
    ours = [f["valid"]["index_sha256"] for f in train["folds"]["folds"]]
    theirs = [C.rows_sha256(sorted(int(i) for i in va)) for _tr, va in folds]
    assert ours != theirs


def test_single_class_fold_is_reported_infeasible():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 2)
    codes = np.arange(y.size)  # singleton scaffolds
    y_bad = y.copy()
    y_bad[:4] = 0  # first four rows all negative, own scaffolds
    folds = D.build_folds(y_bad, codes)
    # one-class folds may or may not appear; the contract is that they are reported
    if folds["status"] != "OK":
        assert folds["feasible"] is False
        assert folds["infeasible_reasons"]
    else:  # pragma: no cover - split happened to stay balanced
        assert folds["feasible"] is True


def test_streamed_labels_keep_only_the_requested_positions(tmp_path):
    """Synthetic label file: the streaming loader must keep only Train positions."""
    import gzip

    values = [0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
    path = tmp_path / "labels.csv.gz"
    with gzip.open(path, "wt") as fh:
        for v in values:
            fh.write(f"{v}\n")
    rows = [3, 5, 9]
    out = D._stream_train_labels(rows, path)
    assert out.tolist() == [values[r] for r in rows]
    with pytest.raises(ValueError):
        D._stream_train_labels([3, 99], path)


def test_real_label_stream_returns_train_rows_only(train):
    out = D._stream_train_labels(train["data"]["rows"])
    assert out.shape == (C.OGB_TRAIN_ROWS,)
    assert set(out.tolist()) <= {0, 1}
    assert out.tolist() == train["data"]["labels"].tolist()
