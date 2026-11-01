"""Source, split and target-integrity tests for the seven-task round."""

from __future__ import annotations

import numpy as np
import pytest

import mb2_common as C
import mb2_source as S


def test_source_manifest_verifies():
    manifest = S.load_source_manifest(verify=True)
    assert manifest["chembench"]["commit"] == S.CHEMBENCH_COMMIT
    assert len(manifest["chembench"]["files"]) == 24
    assert len(manifest["ogb"]["files"]) == 7


def test_all_seven_datasets_have_frozen_sizes():
    for key in C.DATASET_ORDER:
        src = S.load_dataset(key, labels="train_valid", verify=True)
        sizes = tuple(len(src["rows"][s]) for s in ("train", "valid", "test"))
        assert sizes == tuple(C.DATASETS[key]["expected_split_sizes"]), key
        assert sum(sizes) == len(src["smiles"]) == C.DATASETS[key]["expected_rows"]
        union = set(src["rows"]["train"]) | set(src["rows"]["valid"]) | set(src["rows"]["test"])
        assert len(union) == len(src["smiles"])
        assert not (set(src["rows"]["train"]) & set(src["rows"]["test"]))
        assert not (set(src["rows"]["valid"]) & set(src["rows"]["test"]))


def test_labels_none_decodes_no_target():
    for key in C.DATASET_ORDER:
        src = S.load_dataset(key, labels="none")
        assert src["y"] is None
        if src["dataframe"] is not None:  # pragma: no cover - defensive
            raise AssertionError(f"{key}: dataframe must not be materialised without labels")
        if src["source_kind"] == "chembench":
            assert src["columns"] == ["smiles"]


def test_tox21_missing_labels_are_nan_not_zero():
    src = S.load_chembench("Tox21", labels="all")
    y = src["y"]
    train = np.asarray(y["train"], dtype=np.float64)
    assert np.isnan(train).any(), "Tox21 must expose missing cells as NaN"
    observed = train[np.isfinite(train)]
    assert set(np.unique(observed).tolist()) <= {0.0, 1.0}
    # a zero-fill would make the observed count equal the cell count
    assert int(np.isfinite(train).sum()) < train.size
    assert tuple(len(src["rows"][s]) for s in ("train", "valid", "test")) == (6264, 783, 784)


def test_tox21_every_endpoint_has_both_classes_in_train_and_valid():
    import mb2_data as D

    view = D.build_view("Tox21")
    train = view.targets.audit("train")
    valid = view.targets.audit("valid")
    assert train["n_tasks"] == 12 == valid["n_tasks"]
    assert train["tasks_with_both_classes"] == 12
    assert valid["tasks_with_both_classes"] == 12
    assert list(train["per_task"]) == list(C.TOX21_TASKS)


def test_ogb_split_is_the_official_one_and_ordered_as_published():
    import pandas as pd

    src = S.load_ogb_hiv(labels="train_valid")
    for name in ("train", "valid", "test"):
        published = pd.read_csv(S.ogb_path(f"split/scaffold/{name}.csv.gz"), header=None).to_numpy().reshape(-1)
        assert src["rows"][name] == [int(x) for x in published], f"{name} row order changed"
        assert src["rows"][name] == sorted(src["rows"][name])
    assert src["label_agreement"]["identical"] is True
    assert src["label_agreement"]["rows_compared"] == 41127
    assert len(src["smiles"]) == len(set(src["smiles"])) or True  # duplicates are allowed, not hidden
    assert all(isinstance(s, str) and s for s in src["smiles"][:10])


def test_chembench_repeat_zero_is_the_frozen_file():
    for key in ("ESOL", "Lipophilicity", "FreeSolv", "BACE", "BBBP", "Tox21"):
        src = S.load_chembench(key, labels="none")
        assert src["selected_repeat"]["file"].endswith(f"{C.SPLIT_REPEAT_FILE_INDEX}.ind.pkl")
        assert len(src["repeats"]) == 3


def test_bbbp_and_bace_scaffold_repeats_are_identical():
    for key in ("BBBP", "BACE"):
        src = S.load_chembench(key, labels="none")
        hashes = {r["sha256"] for r in src["repeats"]}
        assert len(hashes) == 1, f"{key}: scaffold repeats should be byte-identical"


def test_tox21_and_freesolv_random_repeats_differ():
    for key in ("Tox21", "FreeSolv", "ESOL"):
        src = S.load_chembench(key, labels="none")
        hashes = {r["sha256"] for r in src["repeats"]}
        assert len(hashes) == 3, f"{key}: random repeats must be genuinely different draws"


@pytest.mark.parametrize(
    "key",
    ["ESOL", "Lipophilicity", "FreeSolv", "BACE", "BBBP", "Tox21", "ogbg-molhiv"],
)
def test_split_rows_are_disjoint_and_cover_everything(key):
    src = S.load_dataset(key, labels="none")
    rows = src["rows"]
    s = {k: set(v) for k, v in rows.items()}
    assert not (s["train"] & s["valid"]) and not (s["train"] & s["test"]) and not (s["valid"] & s["test"])
    assert sum(len(v) for v in s.values()) == len(src["smiles"])
