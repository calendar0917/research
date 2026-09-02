"""Self-tests for rooted-canonical downstream attribution."""
from __future__ import annotations

from .run_rooted_canonical_downstream_attribution import BRANCHES, classify_folds


def _fold(index: int, values: dict[str, float]) -> dict:
    return {
        "fold_index": index,
        "branches": {
            branch: {
                "joint_balanced_accuracy": values[branch],
                "family_balanced_accuracy": values[branch],
                "degree_balanced_accuracy": values[branch],
            }
            for branch in BRANCHES
        },
        "stability": {
            branch: {"cosine_similarity": 0.95, "relative_l2_difference": 0.1}
            for branch in BRANCHES
        },
        "invariants": {"passed": True},
    }


def main() -> int:
    values = {
        "GLOBAL_STATS": 0.70,
        "ROOTED_RAW_BAG": 0.68,
        "ROOTED_RAW_TRUE_RELATION": 0.72,
        "ROOTED_KSVD_BAG": 0.69,
        "ROOTED_KSVD_TRUE_RELATION": 0.76,
        "ROOTED_KSVD_SHUFFLED_RELATION": 0.70,
    }
    decision = classify_folds([_fold(i, values) for i in range(3)])
    assert decision["classification"] == "ROOTED_KSVD_RELATIONS_ADD_DOWNSTREAM_VALUE"
    assert decision["relation_fold_wins"] == [True, True, True]

    values["GLOBAL_STATS"] = 0.77
    direct = classify_folds([_fold(i, values) for i in range(3)])
    assert direct["classification"] == "DIRECT_STATS_SUFFICIENT_FOR_SYNTHETIC_FACTORS"
    print("rooted_canonical_downstream_attribution self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
