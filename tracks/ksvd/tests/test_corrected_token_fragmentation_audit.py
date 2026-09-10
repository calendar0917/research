"""Unit tests for the corrected-token fragmentation audit helpers.

The audit itself is a read-only diagnostic over frozen run artifacts; these
tests pin the pure, deterministic pieces (transition classification, split
maps, group assignment, quintile ladders) so the mechanism table cannot drift
silently.
"""

from __future__ import annotations

import numpy as np

from tracks.ksvd.experiments.luyin16.zinc_corrected_token_fragmentation_audit import (
    assign_groups,
    classification,
    fit_vocabulary,
    partial_spearman,
    quintile_table,
    spearman,
    tier_of,
    token_split_map,
)


def test_classification_transitions():
    assert classification(0, 3) == "historical_oov"
    assert classification(4, 0) == "newly_oov"
    assert classification(10, 10) == "stable_common"
    assert classification(10, 3) == "newly_rare"
    assert classification(3, 3) == "already_rare"
    assert classification(2, 9) == "rare_to_common"


def test_token_split_map_refinement_and_metrics():
    # historical token 0 splits into corrected 0 and 1; historical 1 is not split.
    hist = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    corr = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    mapping = token_split_map(hist, corr)
    assert mapping[0]["historical_count"] == 4
    assert mapping[0]["split_multiplicity"] == 2
    assert sorted(mapping[0]["corrected_child_ids"]) == [0, 1]
    assert sorted(mapping[0]["corrected_child_counts"]) == [2, 2]
    assert mapping[1]["split_multiplicity"] == 1
    # entropy of a 2/2 split is log(2)
    assert abs(mapping[0]["split_entropy"] - np.log(2)) < 1e-9


def test_token_split_map_rejects_non_refinement():
    # corrected key 0 appears under two historical tokens -> refinement violated
    hist = np.array([0, 1], dtype=np.int64)
    corr = np.array([0, 0], dtype=np.int64)
    try:
        token_split_map(hist, corr)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for a non-refining map")


def test_fit_vocabulary_matches_frequency_order():
    ids = np.array([5, 5, 5, 7, 7, 9], dtype=np.int64)
    vocab = fit_vocabulary(ids, maximum=8192)
    assert vocab[5] == 1
    assert vocab[7] == 2
    assert vocab[9] == 3
    assert tier_of(0, 768) == "full"  # OOV shares the full table row
    assert tier_of(767, 768) == "full"
    assert tier_of(768, 768) == "lowrank"
    assert tier_of(1_000, 768) == "lowrank"


def test_assign_groups_priority_and_median_split():
    features = {
        "fraction_newly_rare": np.array([0, 0, 0, 1, 0, 0], dtype=np.float64),
        "fraction_newly_oov": np.array([0, 0, 1, 0, 0, 0], dtype=np.float64),
        "fraction_split_common": np.array([0.1, 0.9, 0.5, 0.5, 0.2, 0.8], dtype=np.float64),
        "fraction_split": np.array([1, 1, 1, 1, 1, 1], dtype=np.float64),
    }
    groups = assign_groups(features)
    group = groups["group"]
    assert group[2] == "G4"  # priority over newly-rare
    assert group[3] == "G3"
    supported = np.array([True, True, False, False, True, True])
    threshold = float(np.median(features["fraction_split_common"][supported]))
    assert group[0] == "G1" and group[1] == "G2"
    assert threshold == 0.5


def test_spearman_and_partial_are_defined():
    rng = np.random.default_rng(0)
    z = rng.normal(size=400)
    x = z + 0.05 * rng.normal(size=400)
    y = z + 0.05 * rng.normal(size=400)
    assert spearman(x, y) > 0.8  # shared confound drives a strong raw association
    assert abs(partial_spearman(x, y, z)) < 0.4  # ...which the partial removes


def test_quintile_table_is_monotone_for_a_monotone_signal():
    rng = np.random.default_rng(1)
    n = 500
    metric = rng.normal(size=n)
    base = metric  # degradation rises with the metric
    err_hist = np.zeros((4, n))
    err_corr = np.tile(base, (4, 1))
    degradation = {"err_hist": err_hist, "err_corr": err_corr, "delta": err_corr - err_hist}
    rows = quintile_table(metric, degradation, n_bins=5)
    degradations = [row["degradation"] for row in rows]
    assert degradations == sorted(degradations)
