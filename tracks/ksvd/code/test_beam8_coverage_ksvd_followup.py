"""Self-tests for Beam8 coverage KSVD follow-up decisions."""
from __future__ import annotations

from .run_beam8_coverage_ksvd_followup import classify


def _metrics(full: float, corrected: float, observed: float, recall: float) -> dict:
    return {
        "uncorrected": {
            "full_adjacency_rmse": full,
            "observed_pair_rmse": observed,
            "full_edge_recall": recall,
        },
        "residual_corrected": {"full_adjacency_rmse": corrected},
    }


def _branch(stage: dict, seeds: tuple[int, ...]) -> dict:
    return {
        "seeds": {
            str(seed): {"mean_stages": {"final": stage}} for seed in seeds
        },
        "mean_stages": {"final": stage},
        "dictionary_scalars": 100,
        "code_scalars_per_graph": 20.0,
    }


def main() -> int:
    seeds = (1, 2, 3)
    branches = {}
    for geometry in ("s8_o2", "s10_o3", "s12_o4"):
        branches[f"{geometry}_BASE"] = _branch(
            _metrics(0.40, 0.30, 0.30, 0.70), seeds
        )
        branches[f"{geometry}_FAIR95"] = _branch(
            _metrics(0.35, 0.301, 0.303, 0.75), seeds
        )
    decision = classify(branches, seeds)
    assert decision["classification"] == (
        "ADOPT_FAIR95_OPERATING_POINT_GEOMETRY_UNRESOLVED"
    )
    assert decision["passing_fair95_geometries"] == [
        "s8_o2",
        "s10_o3",
        "s12_o4",
    ]
    print("beam8_coverage_ksvd_followup self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
