"""Official split, isolated Test labels, and the official AP definition."""

import numpy as np
import pytest

import rd_common as C
import rd_data as D

pytestmark = pytest.mark.slow


def test_split_partition(task):
    audit = D.split_partition_audit(task)
    assert audit["counts"] == {"train": 350343, "valid": 43793, "test": 43793}
    assert audit["counts_match"] is True
    assert audit["train_valid_overlap"] == 0
    assert audit["train_test_overlap"] == 0
    assert audit["valid_test_overlap"] == 0
    assert audit["union_unique"] == audit["union_size"] == C.N_MOLECULES
    assert audit["covers_all_molecules"] is True
    assert audit["split_partition_sha256"] == C.SOURCE["split_partition_sha256"]


def test_frozen_artifact_keeps_test_rows_nan(task):
    assert np.isfinite(task.labels[task.test_rows]).sum() == 0
    assert np.isfinite(task.labels[task.train_rows]).any()
    assert np.isfinite(task.labels[task.valid_rows]).any()
    assert task.label_artifact_sha256 == C.SOURCE["label_artifact_sha256"]


def test_official_raw_labels_align_and_test_is_binary(task, test_labels):
    audit = test_labels["audit"]
    assert audit["shape"] == [C.N_MOLECULES, C.N_TASKS]
    assert audit["rows_read"] == C.N_MOLECULES
    assert audit["train_valid_alignment_verified"] is True
    assert audit["test_labels_enter_training"] is False
    assert audit["test_shape"] == [C.N_TEST, C.N_TASKS]
    y = test_labels["labels"]
    finite = np.isfinite(y)
    assert finite.any()
    assert np.isin(y[finite], (0.0, 1.0)).all()
    assert audit["test_observed_entries"] == int(finite.sum())
    assert audit["test_positive_entries"] == int((y[finite] == 1.0).sum())


def test_test_labels_differ_from_the_frozen_nan_block(task, test_labels):
    """The Test labels are real labels and do not come from the training artifact."""
    y = test_labels["labels"]
    if np.isfinite(y).all():
        assert not np.isfinite(task.labels[task.test_rows]).any()


def test_task45_is_excluded_from_the_validation_macro_ap(splits):
    assert splits["valid"].positive[45] == 0
    rng = np.random.default_rng(0)
    y = splits["valid"].labels
    probs = rng.random(y.shape).astype(np.float32)
    report = C.MET.per_task_ap(y, probs)
    assert report["per_task"][45] is None
    assert report["n_tasks_scored"] == 127
    official = C.MET.official_ap(y, probs)
    assert official["value"] == pytest.approx(report["mean_over_scored_tasks"], abs=1e-12)


def test_official_ap_and_sklearn_agree_on_all_splits(splits):
    rng = np.random.default_rng(7)
    for name in C.SPLITS:
        y = splits[name].labels[:3000]
        probs = rng.random(y.shape).astype(np.float32)
        cross = D.cross_check_ap(y, probs)
        assert cross["abs_delta"] <= 1e-12, (name, cross)


def test_evaluate_split_covers_every_row_once(task, splits, models):
    model = models["W3_WD10"][0]
    tiny = D.SplitData(
        name="valid",
        rows=splits["valid"].rows[:64],
        labels=np.ascontiguousarray(splits["valid"].labels[:64]),
        observed=splits["valid"].observed,
        positive=splits["valid"].positive,
        negative=splits["valid"].negative,
    )
    rep = D.evaluate_split(model, task, tiny, "cpu", batch_size=16)
    assert rep["rows"] == 64
    assert len(rep["per_task_ap"]) == C.N_TASKS
    assert len(rep["observed"]) == C.N_TASKS
    assert sum(rep["observed"]) == int(np.isfinite(tiny.labels).sum())
    assert rep["logits_sha256"] != rep["probs_sha256"]
    assert rep["ap_source"].startswith("OGB official Evaluator")


def test_training_batches_never_contain_test_or_valid_rows(task, plan):
    isolation = C.SD.plan_summary(plan, task)["isolation"]
    assert isolation["passed"] is True
    assert len(isolation["checks"]) == 40
    for check in isolation["checks"]:
        assert check["test_rows_in_epoch"] == 0
        assert check["valid_rows_in_epoch"] == 0
        assert check["is_train_permutation"] is True
