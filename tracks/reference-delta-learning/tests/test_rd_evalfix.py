"""Regression tests for the rd-evalfix-v1 evaluation bug.

The original constructor always read ``task.x["train"]``.  These tests use a
sentinel matrix where Train, Val and Test values are distinguishable, so the
old implementation cannot pass by accident.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import rd_arms as A
import rd_common as C
import rd_data as D
import rd_eval as E
import rd_pairs as P
import rd_reeval as RE
import rd_refs as R
import rd_tree as T
from helpers import make_task


def _sentinel_task():
    return SimpleNamespace(
        x={
            "train": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "valid": np.array([[11, 12]], dtype=np.float32),
            "test": np.array([[21, 22]], dtype=np.float32),
        }
    )


def _legacy_pair_matrix(task, src_local, ref_local):
    """The exact pre-repair implementation, kept here as the failing baseline."""
    x = task.x["train"]
    s = np.asarray(src_local, dtype=np.int64)
    r = np.asarray(ref_local, dtype=np.int64)
    head = x[s]
    tail = x[r]
    return np.concatenate([head, tail, head - tail], axis=1)


def test_explicit_domains_reproduce_the_sentinel_values():
    task = _sentinel_task()
    q = np.array([0])
    r = np.array([1])
    train = P.pair_matrix(
        query_x=task.x["train"],
        reference_x=task.x["train"],
        query_local=q,
        reference_local=r,
    )
    valid = P.pair_matrix(
        query_x=task.x["valid"],
        reference_x=task.x["train"],
        query_local=q,
        reference_local=r,
    )
    test = P.pair_matrix(
        query_x=task.x["test"],
        reference_x=task.x["train"],
        query_local=q,
        reference_local=r,
    )
    assert train.tolist() == [[1.0, 2.0, 3.0, 4.0, -2.0, -2.0]]
    assert valid.tolist() == [[11.0, 12.0, 3.0, 4.0, 8.0, 8.0]]
    assert test.tolist() == [[21.0, 22.0, 3.0, 4.0, 18.0, 18.0]]

    valid_rev = P.pair_matrix(
        query_x=task.x["train"],
        reference_x=task.x["valid"],
        query_local=r,
        reference_local=q,
    )
    test_rev = P.pair_matrix(
        query_x=task.x["train"],
        reference_x=task.x["test"],
        query_local=r,
        reference_local=q,
    )
    assert valid_rev.tolist() == [[3.0, 4.0, 11.0, 12.0, -8.0, -8.0]]
    assert test_rev.tolist() == [[3.0, 4.0, 21.0, 22.0, -18.0, -18.0]]

    blocks = P.pair_feature_blocks(
        query_x=task.x["valid"],
        reference_x=task.x["train"],
        query_local=q,
        reference_local=r,
    )
    assert blocks[0][:, 0].tolist() == [11.0, 12.0]
    assert blocks[1][:, 0].tolist() == [3.0, 4.0]
    assert blocks[2][:, 0].tolist() == [8.0, 8.0]


def test_the_old_train_only_constructor_fails_the_sentinel():
    task = _sentinel_task()
    q = np.array([0])
    r = np.array([1])
    old_valid = _legacy_pair_matrix(task, q, r)
    old_test = _legacy_pair_matrix(task, q, r)
    assert old_valid.tolist() == [[1.0, 2.0, 3.0, 4.0, -2.0, -2.0]]
    assert old_test.tolist() == old_valid.tolist()
    assert old_valid.tolist() != [[11.0, 12.0, 3.0, 4.0, 8.0, 8.0]]
    assert old_test.tolist() != [[21.0, 22.0, 3.0, 4.0, 18.0, 18.0]]


def test_real_evaluation_chain_passes_the_evaluated_split(monkeypatch):
    task = make_task(n_train=16, n_valid=3, n_test=3, n_bits=8, n_descriptors=2)
    ctx = A.ArmContext.build(task, R.reference_table(task))
    spec = P.build_pairs(task, P.eligible_order_train(task))
    seen: list[tuple[int, int, int]] = []
    original = P.pair_feature_blocks

    def spy(*, query_x, reference_x, query_local, reference_local, chunk=4096):
        seen.append((id(query_x), id(reference_x), int(np.asarray(query_local).size)))
        return original(
            query_x=query_x,
            reference_x=reference_x,
            query_local=query_local,
            reference_local=reference_local,
            chunk=chunk,
        )

    monkeypatch.setattr(P, "pair_feature_blocks", spy)
    A.run_c(ctx, spec, preset="H1", device="cpu", rounds=2)

    # The formal evaluator must ask for Val queries in the Val domain and Test
    # queries in the Test domain, always against the Train reference domain.
    for split in ("valid", "test"):
        assert any(
            query_id == id(task.x[split]) and reference_id == id(task.x["train"])
            for query_id, reference_id, _ in seen
        ), split
    # No call may use a non-Train reference domain.
    assert all(reference_id == id(task.x["train"]) for _, reference_id, _ in seen)
    # The old implementation called the constructor positionally with only the
    # Train matrix; this spy would have raised TypeError, so the regression test
    # cannot pass against it.


def test_oracle_restores_query_y_on_all_splits_not_train_first_row():
    train = np.array([[4.0, 1.0], [7.0, 2.0], [10.0, 3.0]], dtype=np.float32)
    valid = np.array([[11.0, 5.0]], dtype=np.float32)
    test = np.array([[21.0, 6.0]], dtype=np.float32)
    y_train = train[:, 0].astype(np.float64) + 7.0
    y_valid = valid[:, 0].astype(np.float64) + 7.0
    y_test = test[:, 0].astype(np.float64) + 7.0
    task = D.TaskData(
        dataset="oracle",
        x={"train": train, "valid": valid, "test": test},
        bits={"train": train[:, :1], "valid": valid[:, :1], "test": test[:, :1]},
        y={"train": y_train, "valid": y_valid, "test": y_test},
        row_id={
            "train": np.array([0, 1, 2]),
            "valid": np.array([10]),
            "test": np.array([20]),
        },
        smiles={"train": ["a", "b", "c"], "valid": ["v"], "test": ["t"]},
        scaler={"mean": 123.0, "std": 3.0},
        feature_names=["x0", "x1"],
        design_stats={},
        cache_sha256="oracle",
        ecfp_sha256="oracle",
        graph_cache={},
        ecfp_cache={},
    )
    refs = {
        "train": {
            "ref_local": np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
            "n_kept": np.array([2, 2, 2], dtype=np.int32),
        },
        "valid": {
            "ref_local": np.array([[1, 2]], dtype=np.int32),
            "n_kept": np.array([2], dtype=np.int32),
        },
        "test": {
            "ref_local": np.array([[0, 2]], dtype=np.int32),
            "n_kept": np.array([2], dtype=np.int32),
        },
    }
    sigma = float(task.scaler["std"])
    weights = np.array([[0.5, 0.5]], dtype=np.float64)
    for split, expected in (("valid", 18.0), ("test", 28.0)):
        query, ref = P.evaluation_pairs(refs[split])
        fwd = P.pair_feature_blocks(
            query_x=task.x[split],
            reference_x=task.x["train"],
            query_local=query,
            reference_local=ref,
        )
        rev = P.pair_feature_blocks(
            query_x=task.x["train"],
            reference_x=task.x[split],
            query_local=ref,
            reference_local=query,
        )
        # Known comparator h(z(q,r)) = (x_q[0] - x_r[0]) / sigma.
        h_fwd = (fwd[0][0] - fwd[1][0]) / sigma
        h_rev = (rev[0][0] - rev[1][0]) / sigma
        delta = E.delta_from_margins(h_fwd.reshape(1, -1), h_rev.reshape(1, -1), sigma)
        y_ref = E.reference_targets(task, refs[split]["ref_local"])
        pred = E.aggregate_c(weights, y_ref, delta)
        assert pred[0] == pytest.approx(expected, abs=1e-6)
        assert pred[0] != pytest.approx(y_train[0], abs=1e-6)
        # mu is deliberately non-zero and must never enter delta.
        assert pred[0] != pytest.approx(expected + task.scaler["mean"], abs=1e-6)


def test_training_pair_input_is_bit_identical_to_the_pre_fix_train_use():
    task = make_task(n_train=24, n_valid=4, n_test=4, n_bits=8, n_descriptors=2)
    spec = P.build_pairs(task, P.eligible_order_train(task))
    report = RE.check_training_input_unchanged(task, spec, count=128)
    assert report["pass"] is True
    assert report["bitwise_identical"] is True
    assert report["max_abs_dense_diff"] == 0.0
    assert report["max_abs_block_diff"] == 0.0


def test_input_identity_check_rejects_the_old_train_only_blocks():
    task = make_task(n_train=24, n_valid=4, n_test=4, n_bits=8, n_descriptors=2)
    ref_table = R.reference_table(task)
    ctx = A.ArmContext.build(task, ref_table)
    blocks = A.build_eval_blocks(ctx)
    sample = {split: RE.sample_legal_slots(ref_table, split, count=8, seed=1703) for split in C.SPLITS}
    good = RE.check_input_identity(task, ref_table, blocks, sample)
    assert good["pass"] is True
    wrong = {}
    for split in C.SPLITS:
        query, ref = P.evaluation_pairs(ref_table["splits"][split])
        wrong[split] = P.pair_feature_blocks(
            query_x=task.x["train"],
            reference_x=task.x["train"],
            query_local=query,
            reference_local=ref,
        )
    bad = RE.check_input_identity(task, ref_table, wrong, sample)
    assert bad["pass"] is False
    assert any(
        entry["max_abs_forward_pair_diff"] > 0.0
        for split, entry in bad["splits"].items()
        if split in ("valid", "test")
    )


def test_evaluation_pair_constructor_has_no_train_default():
    task = _sentinel_task()
    with pytest.raises(TypeError):
        P.pair_matrix(task.x["valid"], task.x["train"], np.array([0]), np.array([1]))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        P.pair_feature_blocks(  # type: ignore[call-arg]
            query_x=task.x["valid"],
            reference_x=task.x["train"],
            query_local=np.array([0]),
        )


def test_reevaluation_path_never_calls_a_training_entry():
    import xgboost as xgb

    task = make_task(n_train=12, n_valid=3, n_test=3, n_bits=6, n_descriptors=1)
    ctx = A.ArmContext.build(task, R.reference_table(task))
    spec = P.build_pairs(task, P.eligible_order_train(task))
    src = np.asarray(spec["src_local"], dtype=np.int64)[:200]
    ref = np.asarray(spec["ref_local"], dtype=np.int64)[:200]
    matrix = P.pair_matrix(
        query_x=task.x["train"],
        reference_x=task.x["train"],
        query_local=src,
        reference_local=ref,
    )
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu",
        "max_depth": 2,
        "eta": 0.3,
        "base_score": 0.0,
        "seed": 0,
        "nthread": 2,
    }
    booster = xgb.train(params, xgb.DMatrix(matrix, label=spec["target_std"][:200]), num_boost_round=2)
    trees = T.extract_trees(booster, [f"c{i}" for i in range(matrix.shape[1])], rounds=2)
    with RE.forbid_training() as calls:
        run = A._evaluate_c(
            ctx,
            spec,
            preset="H1",
            booster=booster,
            trees=trees,
            seconds=0.0,
            include_pairs=False,
        )
    assert all(int(value) == 0 for value in calls.values())
    assert len(run.curves) == 2 * 3 * len(C.CONFIGS)
