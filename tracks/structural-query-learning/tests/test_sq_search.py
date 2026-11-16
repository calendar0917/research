"""Residual-stump scoring and the synthetic known-pool check.

The synthetic experiment proves two mechanical facts: (a) the frozen residual
score ranks a query that really carries target information above a useless one,
and (b) the read-out can actually use that feature.  It is *not* evidence that
the automatic search discovers unknown rules.
"""

from __future__ import annotations

import numpy as np
import pytest

import sq_ast as A
import sq_bank as B
import sq_search as S
import sq_xgb as X
from helpers import make_packed, random_tree_molecules


def _synthetic(n_molecules: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    graphs = random_tree_molecules(rng, n_molecules)
    packed = make_packed(graphs)
    counts_b = packed.graph_counts(A.evaluate(packed, A.ATOM_EQ(0, 7))).astype(np.float64)
    counts_o = packed.graph_counts(
        A.evaluate(packed, A.NBR(A.ATOM_EQ(0, 8), A.ANY_BOND_PRED(), 1))
    ).astype(np.float64)
    noise = rng.normal(scale=0.05, size=n_molecules)
    y = 2.0 * counts_b + 1.0 * counts_o + noise
    packed.labels = y.astype(np.float32)
    return packed, y


def test_best_stump_finds_the_informative_split():
    x = np.asarray([0, 0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=np.float64)
    r = x * 2.0
    stump = S.best_stump(x, r, counts_column=True, min_molecules=2)
    assert stump is not None
    assert stump["threshold"] in (0.0, 1.0, 2.0)
    assert stump["gain"] > 0
    # a constant feature has no valid split
    assert S.best_stump(np.zeros(10), r, counts_column=True, min_molecules=2) is None
    # both sides must keep the minimum number of molecules
    assert S.best_stump(x, r, counts_column=True, min_molecules=6) is None


def test_residual_score_ranks_the_true_query_first():
    packed, y = _synthetic()
    residual = y - y.mean()
    true_queries = [
        A.ATOM_EQ(0, 7),
        A.NBR(A.ATOM_EQ(0, 8), A.ANY_BOND_PRED(), 1),
    ]
    distractors = [
        A.ATOM_EQ(0, 5),
        A.ATOM_EQ(0, 6),
        A.DEG_EQ(1),
        A.IN_CYCLE(),
        A.MIN_RING_EQ(6),
        A.NBR(A.ATOM_EQ(0, 6), A.ANY_BOND_PRED(), 2),
    ]
    scores = {
        A.canon(ast): S.score_query(packed, ast, residual, cache=B.MaskCache(limit_bytes=8 << 20))
        for ast in true_queries + distractors
    }
    for ast in true_queries:
        assert scores[A.canon(ast)]["score"] > 0
        assert scores[A.canon(ast)]["reject_reason"] is None
    ranked = sorted(scores.values(), key=lambda item: -item["score"])
    assert ranked[0]["ast"] in true_queries
    assert scores[A.canon(A.ATOM_EQ(0, 5))]["reject_reason"] == "support"
    assert scores[A.canon(A.IN_CYCLE())]["reject_reason"] == "support"


def test_constant_feature_is_rejected():
    """Identical molecules give a constant feature even for a supported query."""
    packed = make_packed([([6, 6], [(0, 1)])] * 10)
    residual = np.linspace(-1.0, 1.0, 10)
    result = S.score_query(packed, A.TRUE(), residual)
    assert result["reject_reason"] == "constant"


def test_complexity_penalty_reduces_the_score_of_a_bigger_ast():
    packed, y = _synthetic()
    residual = y - y.mean()
    small = A.ATOM_EQ(0, 7)
    big = A.AND(A.ATOM_EQ(0, 7), A.DEG_EQ(1))
    cache = B.MaskCache(limit_bytes=8 << 20)
    s_small = S.score_query(packed, small, residual, cache=cache)
    s_big = S.score_query(packed, big, residual, cache=cache)
    if s_big["gain"] > 0:
        assert s_big["score"] <= s_big["gain"] / (1.0 + 0.05 * (A.ast_nodes(big) - 1)) + 1e-12
        assert s_big["score"] < s_small["score"] or s_big["gain"] <= s_small["gain"]


def test_readout_uses_the_discovered_feature():
    """The GBM head improves when the informative query columns are present."""
    packed, y = _synthetic(n_molecules=160)
    scaler = {"mean": float(y.mean()), "std": float(y.std(ddof=0))}
    y_std = (y - scaler["mean"]) / scaler["std"]
    base_only = B.design_matrix(packed, [], include_base=True)[0]
    queries = [
        B.QueryRecord(ast=A.ATOM_EQ(0, 7), label="Q000", source="B", outer_round=0, train_support=0),
        B.QueryRecord(
            ast=A.NBR(A.ATOM_EQ(0, 8), A.ANY_BOND_PRED(), 1),
            label="Q001",
            source="B",
            outer_round=0,
            train_support=0,
        ),
    ]
    with_queries, names = B.design_matrix(packed, queries, include_base=True)
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu",
        "max_depth": 3,
        "eta": 0.1,
        "min_child_weight": 2,
        "reg_lambda": 1.0,
        "base_score": 0.0,
        "seed": 0,
        "nthread": 2,
    }
    import xgboost as xgb

    def train_rmse(matrix):
        dmat = xgb.DMatrix(matrix, label=y_std)
        booster = xgb.train(params, dmat, num_boost_round=64, verbose_eval=False)
        pred = booster.predict(dmat)
        return float(np.sqrt(np.mean((pred - y_std) ** 2)))

    rmse_base = train_rmse(base_only)
    rmse_queries = train_rmse(with_queries)
    assert rmse_queries < rmse_base
    assert rmse_queries < 0.15


def test_candidate_generation_respects_grammar_and_budget():
    packed, y = _synthetic(n_molecules=40)
    vocab = {
        "atom_values": {j: [6, 7, 8] for j in range(9)},
        "bond_values": {j: [0] for j in range(3)},
        "degree_values": [0, 1, 2, 3],
        "ring_values": [0],
        "element_values": [6, 7, 8],
    }
    primitives = B.node_primitives(vocab)
    bonds = B.bond_predicates(vocab)
    rng = np.random.Generator(np.random.PCG64(1701))
    candidates, stats = S.generate_candidates(primitives[:8], primitives, bonds, rng=rng, existing=set())
    assert candidates
    assert all(A.validates(ast) for ast in candidates)
    assert len({A.canon(ast) for ast in candidates}) == len(candidates)
    assert stats.and_primitive + stats.and_pair + stats.nbr + stats.branch == len(candidates)
    assert stats.accepted == len(candidates)


def test_random_library_preserves_shape_and_is_label_free():
    packed, y = _synthetic(n_molecules=120)
    vocab = {
        "atom_values": {j: [6, 7, 8] for j in range(9)},
        "bond_values": {j: [0] for j in range(3)},
        "degree_values": [0, 1, 2, 3],
        "ring_values": [0],
        "element_values": [6, 7, 8],
    }
    masks = B.MaskCache(limit_bytes=8 << 20)
    base, _ = B.build_base_bank(packed, vocab, cache=masks)
    discovered = [
        B.QueryRecord(
            ast=A.NBR(A.NBR(A.ATOM_EQ(0, 7), A.ANY_BOND_PRED(), 1), A.ANY_BOND_PRED(), 2),
            label="A000",
            source="discovered",
            outer_round=1,
            train_support=10,
            train_gain=0.1,
        )
    ]
    result = S.build_random_library(
        discovered=discovered,
        train_packed=packed,
        vocabulary=vocab,
        existing_records=base,
        cache=masks,
    )
    assert result["n_requested"] == 1
    assert result["n_filled"] == 1
    assert result["all_shapes_match"]
    original_shape = S.ast_shape(discovered[0].ast)
    mapped_shape = S.ast_shape(result["records"][0].ast)
    assert original_shape["counts"] == mapped_shape["counts"]
    assert original_shape["m_values"] == mapped_shape["m_values"]
    assert original_shape["nodes"] == mapped_shape["nodes"]
    # remapping never introduces a value outside the Train vocabulary
    for kind, leaf in A.leaf_params(result["records"][0].ast):
        if kind == "node" and leaf[0] == "ATOM_EQ":
            assert leaf[2] in vocab["atom_values"][leaf[1]]
