"""Self-tests for invariant-space KSVD capacity selection."""
from __future__ import annotations

from .run_invariant_space_ksvd_capacity_audit import classify_cells


def _cell(cell_id: str, n_atoms: int, sparsity: int, rmse: float, *, passing: bool) -> dict:
    return {
        "cell_id": cell_id,
        "n_atoms": n_atoms,
        "sparsity": sparsity,
        "decision": {
            "classification": (
                "ADOPT_INVARIANT_SPACE_KSVD_TOKEN"
                if passing
                else "INVARIANT_KSVD_STABLE_BUT_TOO_LOSSY"
            ),
            "mean_branches": {
                "INV_KSVD_TRUE_RELATION": {"overall_rmse": rmse}
            },
        },
    }


def main() -> int:
    cells = [
        _cell("K16_T3", 16, 3, 0.123, passing=False),
        _cell("K24_T3", 24, 3, 0.124, passing=True),
        _cell("K32_T3", 32, 3, 0.120, passing=True),
        _cell("K16_T4", 16, 4, 0.119, passing=True),
    ]
    decision = classify_cells(cells)
    assert decision["classification"] == "ADOPT_INVARIANT_KSVD_CAPACITY"
    assert decision["passing_cell_ids"] == ["K24_T3", "K32_T3", "K16_T4"]
    assert decision["selected_cell_id"] == "K24_T3"

    failed = classify_cells(
        [_cell("K16_T3", 16, 3, 0.130, passing=False)]
    )
    assert failed["classification"] == "INVARIANT_KSVD_CAPACITY_GRID_FAILS_ERROR_GATE"
    assert failed["passing_cell_ids"] == []
    assert failed["selected_cell_id"] is None
    print("invariant_space_ksvd_capacity_audit self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
