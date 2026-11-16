"""Reference retrieval and pair construction: the two label-free structures."""

from __future__ import annotations

import numpy as np
import pytest

import rd_common as C
import rd_pairs as P
import rd_refs as R
from helpers import make_task


def test_tanimoto_hand_examples():
    a = np.array([[1, 1, 0, 0]], dtype=np.float32)
    b = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=np.float32)
    sim = R.tanimoto_matrix(a, b)
    assert sim.shape == (1, 3)
    assert sim[0, 0] == pytest.approx(1.0 / 3.0)
    # both all-zero -> union == 0 -> frozen convention is 0, not NaN
    sim_zero = R.tanimoto_matrix(np.zeros((1, 4), np.float32), np.zeros((1, 4), np.float32))
    assert sim_zero[0, 0] == 0.0
    assert sim[0, 2] == pytest.approx(1.0)


def test_tanimoto_survives_intersections_above_255():
    n_bits = 2048
    ones = np.zeros(n_bits, dtype=np.float32)
    ones[:400] = 1.0
    query = ones.reshape(1, -1)
    refs = np.stack([ones, np.roll(ones, 1)])
    sim = R.tanimoto_matrix(query, refs)
    assert sim[0, 0] == pytest.approx(1.0)
    # 399 shared of 401 union
    assert sim[0, 1] == pytest.approx(399.0 / 401.0)


def test_tanimoto_matches_integer_reference_on_random_bits():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, size=(7, 64)).astype(np.float32)
    b = rng.integers(0, 2, size=(11, 64)).astype(np.float64)
    ours = R.tanimoto_matrix(a, b.astype(np.float32))
    inter = a.astype(np.int64) @ b.astype(np.int64).T
    union = a.sum(1, dtype=np.int64)[:, None] + b.sum(1, dtype=np.int64)[None, :] - inter
    expected = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    assert np.allclose(ours, expected.astype(np.float32), rtol=0.0, atol=0.0)


def test_reference_table_excludes_self_and_identical_smiles():
    task = make_task(n_train=12, n_valid=3, n_test=3, n_bits=8)
    table = R.reference_table(task)
    train = table["splits"]["train"]
    duplicate_pair = (0, task.n("train") - 1)
    assert task.smiles["train"][duplicate_pair[0]] == task.smiles["train"][duplicate_pair[1]]
    for query in range(task.n("train")):
        refs = train["ref_local"][query, : train["n_kept"][query]]
        assert query not in set(refs.tolist())
        for ref in refs.tolist():
            assert task.smiles["train"][ref] != task.smiles["train"][query]
    # the duplicated Train molecule is counted as an excluded same-SMILES row
    assert train["same_smiles_excluded"][0] == 1
    assert train["same_smiles_excluded"][duplicate_pair[1]] == 1


def test_reference_table_is_sorted_by_similarity_then_row_id():
    task = make_task(n_train=9, n_valid=2, n_test=2, n_bits=4)
    # force exact similarity ties: every Train bit vector identical
    for split in C.SPLITS:
        task.bits[split][:] = 0.0
        task.bits[split][:, 0] = 1.0
    table = R.reference_table(task)
    entry = table["splits"]["train"]
    for query in range(task.n("train")):
        kept = entry["ref_local"][query, : entry["n_kept"][query]]
        row_ids = task.row_id["train"][kept]
        assert np.all(np.diff(row_ids) > 0), "ties must break by ascending Train row_id"
        assert np.allclose(entry["sim"][query, : entry["n_kept"][query]], 1.0)


def test_k_eff_and_weight_rows_sum_to_one():
    task = make_task(n_train=10, n_valid=3, n_test=3)
    table = R.reference_table(task)
    for split in C.SPLITS:
        entry = table["splits"][split]
        for config in C.CONFIGS:
            weights = R.config_weights(entry, config["K"], config["weights"])
            assert np.allclose(weights.sum(axis=1), 1.0, rtol=1e-12, atol=0.0)
            nonzero = (weights > 0).sum(axis=1)
            expected = np.minimum(entry["eligible_count"], config["K"])
            assert np.array_equal(nonzero, expected)
            # no weight outside the retained Top64, none on an excluded row
            ref = entry["ref_local"]
            assert np.all(weights[ref < 0] == 0)


def test_references_and_pairs_do_not_depend_on_labels():
    task = make_task(n_train=12, n_valid=3, n_test=3)
    order_a = P.eligible_order_train(task)
    refs_a = R.reference_table(task)
    pairs_a = P.build_pairs(task, order_a)

    shuffled = make_task(n_train=12, n_valid=3, n_test=3)
    for split in C.SPLITS:
        shuffled.y[split] = shuffled.y[split][::-1].copy()
    order_b = P.eligible_order_train(shuffled)
    refs_b = R.reference_table(shuffled)
    pairs_b = P.build_pairs(shuffled, order_b)

    assert all(np.array_equal(a, b) for a, b in zip(order_a, order_b, strict=True))
    for split in C.SPLITS:
        assert np.array_equal(refs_a["splits"][split]["ref_local"], refs_b["splits"][split]["ref_local"])
        assert np.array_equal(refs_a["splits"][split]["sim"], refs_b["splits"][split]["sim"])
    assert np.array_equal(pairs_a["src_local"], pairs_b["src_local"])
    assert np.array_equal(pairs_a["ref_local"], pairs_b["ref_local"])
    assert not np.allclose(pairs_a["target_std"], pairs_b["target_std"])


def test_pair_records_have_frozen_signs_weights_and_counts():
    task = make_task(n_train=34, n_valid=4, n_test=4)
    order = P.eligible_order_train(task)
    spec = P.build_pairs(task, order)
    stats = spec["stats"]
    assert stats["n_records"] == 2 * 34 * C.MAX_PAIRS
    assert stats["n_records_expected"] == 2 * 34 * C.MAX_PAIRS
    assert stats["matches_expected_count"] is True
    assert stats["weight_sum_equals_n_train"] is True
    assert stats["q_min"] == C.MAX_PAIRS and stats["q_max"] == C.MAX_PAIRS
    sigma = task.scaler["std"]
    src = spec["src_local"]
    ref = spec["ref_local"]
    target = spec["target_std"]
    sign = spec["sign"]
    weight = spec["weight"]
    y = task.y["train"]
    origin = spec["query_local"]
    for index in range(0, 200):
        other = ref[index] if sign[index] > 0 else src[index]
        expected = sign[index] * (y[origin[index]] - y[other]) / sigma
        assert target[index] == pytest.approx(expected, rel=0, abs=1e-12)
        assert weight[index] == pytest.approx(1.0 / (2.0 * C.MAX_PAIRS), rel=0, abs=0.0)
    # the reverse record of a pair carries the negated target
    assert target[0] == pytest.approx(-target[1], rel=0, abs=1e-12)
    assert src[0] == ref[1] and ref[0] == src[1]
    per_query = np.zeros(34)
    np.add.at(per_query, spec["query_local"], weight)
    assert np.allclose(per_query, 1.0, rtol=1e-12, atol=0.0)


def test_pair_matrix_difference_column_is_signed():
    task = make_task(n_train=34, n_valid=2, n_test=2)
    order = P.eligible_order_train(task)
    spec = P.build_pairs(task, order)
    src = spec["src_local"][:4].astype(np.int64)
    ref = spec["ref_local"][:4].astype(np.int64)
    matrix = P.pair_matrix(query_x=task.x["train"], reference_x=task.x["train"], query_local=src, reference_local=ref)
    n = task.n_features
    assert matrix.shape == (4, 3 * n)
    assert np.array_equal(matrix[:, :n], task.x["train"][src])
    assert np.array_equal(matrix[:, n:2 * n], task.x["train"][ref])
    assert np.array_equal(matrix[:, 2 * n:], task.x["train"][src] - task.x["train"][ref])
    reverse = P.pair_matrix(query_x=task.x["train"], reference_x=task.x["train"], query_local=ref, reference_local=src)
    assert np.array_equal(reverse[:, 2 * n:], -matrix[:, 2 * n:])
    assert np.array_equal(reverse[:, :n], matrix[:, n:2 * n])


def test_evaluation_pairs_pad_missing_slots_with_the_query_itself():
    entry = {
        "ref_local": np.array([[3, 4, -1, -1]], dtype=np.int32),
        "n_kept": np.array([2], dtype=np.int32),
    }
    query, ref = P.evaluation_pairs(entry)
    assert query.tolist() == [0, 0, 0, 0]
    assert ref.tolist() == [3, 4, 0, 0]
