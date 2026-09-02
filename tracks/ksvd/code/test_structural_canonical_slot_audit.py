"""Self-tests for structural canonical slot audit decisions."""
from __future__ import annotations

from .run_structural_canonical_slot_audit import BRANCHES, classify_branches


def _branch(observed: float, full: float, *, stable: bool = True) -> dict:
    value = 1.0 if stable else 0.8
    return {
        "mean_reconstruction": {
            "observed_pair_rmse": observed,
            "full_adjacency_rmse": full,
        },
        "mean_mapped_stability": {
            "patch_vector_row_match": value,
            "matched_patch_code_cosine": value,
            "graph_embedding_cosine": value,
        },
        "all_invariants_passed": True,
    }


def main() -> int:
    branches = {branch: _branch(0.35, 0.34, stable=False) for branch in BRANCHES}
    branches["CONSTRUCTION"] = _branch(0.35, 0.34)
    branches["CANONICAL"] = _branch(0.351, 0.339)
    branches["ROOTED_CANONICAL"] = _branch(0.349, 0.338)
    branches["OVERLAP_CANONICAL"] = _branch(0.348, 0.337)
    decision = classify_branches(branches)
    assert decision["classification"] == "ADOPT_STRUCTURAL_CANONICAL_SLOTS"
    assert decision["selected_branch"] == "OVERLAP_CANONICAL"
    assert decision["passing_branches"] == [
        "CANONICAL",
        "ROOTED_CANONICAL",
        "OVERLAP_CANONICAL",
    ]

    for branch in ("CANONICAL", "ROOTED_CANONICAL", "OVERLAP_CANONICAL"):
        branches[branch] = _branch(0.40, 0.40)
    failed = classify_branches(branches)
    assert failed["classification"] == "STRUCTURAL_CANONICAL_SLOT_ROUTE_FAILS_JOINT_GATE"
    assert failed["selected_branch"] is None
    print("structural_canonical_slot_audit self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
