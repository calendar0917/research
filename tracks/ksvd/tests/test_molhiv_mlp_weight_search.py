from __future__ import annotations

from tracks.ksvd.experiments.luyin16.molhiv_mlp_weight_search import _select_candidate


def test_select_candidate_uses_validation_and_lower_weight_tie_break() -> None:
    candidates = {
        "2": {"pos_weight": 2.0, "valid": {"best_auc": 0.8}},
        "5": {"pos_weight": 5.0, "valid": {"best_auc": 0.8}},
        "10": {"pos_weight": 10.0, "valid": {"best_auc": 0.81}},
    }
    key, selected = _select_candidate(candidates)
    assert key == "10"
    assert selected["pos_weight"] == 10.0

    tied = {
        "2": {"pos_weight": 2.0, "valid": {"best_auc": 0.8}},
        "5": {"pos_weight": 5.0, "valid": {"best_auc": 0.8}},
    }
    key, selected = _select_candidate(tied)
    assert key == "2"
    assert selected["pos_weight"] == 2.0
