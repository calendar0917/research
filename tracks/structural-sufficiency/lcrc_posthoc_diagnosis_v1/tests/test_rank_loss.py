import numpy as np
from sklearn.metrics import roc_auc_score

import ph_rank as R


def test_rank_identity_with_ties():
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(20, 200))
        y = rng.integers(0, 2, size=n)
        if y.min() == y.max():
            continue
        scores = rng.integers(0, 4, size=n).astype(float)  # many ties
        pos = R.positive_rank_losses(scores, y)
        neg = R.negative_rank_losses(scores, y)
        auc = roc_auc_score(y, scores)
        assert abs((1.0 - pos.mean()) - auc) < 1e-12
        assert abs((1.0 - neg.mean()) - auc) < 1e-12
        per = R.per_sample_rank_loss(scores, y)
        assert abs(per[y == 1].mean() - pos.mean()) < 1e-12
        assert abs(per[y == 0].mean() - neg.mean()) < 1e-12


def test_all_tied_scores_are_half_auc():
    y = np.array([1, 1, 0, 0, 0])
    s = np.zeros(5)
    summ = R.rank_summary(s, y)
    assert abs(summ["roc_auc"] - 0.5) < 1e-12
    assert abs(summ["positive_rank_loss_mean"] - 0.5) < 1e-12
    assert summ["rank_identity_ok"]


def test_perfect_ranking_has_zero_loss():
    y = np.array([1, 1, 0, 0])
    s = np.array([3.0, 2.0, 1.0, 0.0])
    summ = R.rank_summary(s, y)
    assert summ["roc_auc"] == 1.0
    assert abs(summ["positive_rank_loss_mean"]) < 1e-12
