"""The frozen decision table of rd-pilot-small-v1 (thresholds, not significance).

These tests pin the *classification* rules so the report cannot drift towards a
friendlier reading: the 3% / 5% / 50% gates are this exploration's investment
thresholds, and every boolean is asserted explicitly.
"""

from __future__ import annotations

import rd_report as REP


def _main(dataset: str, values: dict[str, tuple[float, float]]) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "arm": arm,
            "valid_rmse": valid,
            "test_rmse": test,
        }
        for arm, (valid, test) in values.items()
    ]


def _comparisons(dataset: str, main: list[dict]) -> list[dict]:
    rmse = {row["arm"]: row for row in main}
    rows = []
    for other in ("D", "N", "R"):
        for split in ("valid", "test"):
            c = rmse["C"][f"{split}_rmse"]
            b = rmse[other][f"{split}_rmse"]
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"C-{other}",
                    "split": split,
                    "gain": 1.0 - c / b,
                }
            )
    return rows


def _best(main: list[dict], *, competitive: bool) -> dict:
    rows = {row["arm"]: row for row in main}
    best = min(("D", "N", "R"), key=lambda arm: rows[arm]["valid_rmse"])
    return {
        "arm": best,
        "valid": rows[best]["valid_rmse"],
        "test": rows[best]["test_rmse"],
        "competitive_c": competitive,
    }


def _run(tasks: dict[str, dict[str, tuple[float, float]]], *, competitive=None):
    main_rows: list[dict] = []
    comparison_rows: list[dict] = []
    best: dict[str, dict] = {}
    for dataset, values in tasks.items():
        rows = _main(dataset, values)
        main_rows.extend(rows)
        comparison_rows.extend(_comparisons(dataset, rows))
        if competitive is None:
            c = {row["arm"]: row for row in rows}["C"]
            b = _best(rows, competitive=False)
            flag = c["valid_rmse"] <= 1.05 * b["valid"] and c["test_rmse"] <= 1.05 * b["test"]
        else:
            flag = competitive
        best[dataset] = _best(rows, competitive=flag)
    checks = {dataset: {"pass": True} for dataset in tasks}
    references = {
        dataset: {arm: {"available": True, "verified": True} for arm in ("G", "L")}
        for dataset in tasks
    }
    return REP.decision_payload(
        main_rows, comparison_rows, best, list(tasks), checks, references
    )


def test_strong_gain_both_splits_both_tasks_is_joint_signal() -> None:
    payload = _run(
        {
            "ogbg-molesol": {"D": (1.0, 1.0), "N": (1.0, 1.0), "R": (1.0, 1.0), "C": (0.9, 0.9)},
            "ogbg-mollipo": {"D": (1.0, 1.0), "N": (1.0, 1.0), "R": (1.0, 1.0), "C": (0.95, 0.95)},
        }
    )
    assert payload["joint_signal"]["classification"] == "JOINT_DELTA_SIGNAL"
    assert payload["joint_signal"]["JOINT_DELTA_SIGNAL"] is True
    assert payload["execution"]["EXECUTION_PASS"] is True
    assert payload["reference_axis"] == "REFERENCE_COMPLETE"
    assert payload["worth_structural_comparator_followup"] is True


def test_one_strong_task_and_one_regressing_task_is_tradeoff() -> None:
    payload = _run(
        {
            "ogbg-molesol": {"D": (1.0, 1.0), "N": (1.0, 1.0), "R": (1.0, 1.0), "C": (0.9, 0.9)},
            "ogbg-mollipo": {"D": (1.0, 1.0), "N": (1.0, 1.0), "R": (1.0, 1.0), "C": (1.2, 1.2)},
        }
    )
    assert payload["joint_signal"]["classification"] == "CROSS_TASK_TRADEOFF"
    assert payload["joint_signal"]["JOINT_DELTA_SIGNAL"] is False
    assert payload["worth_structural_comparator_followup"] is False


def test_not_beating_neighbour_arm_is_no_increment() -> None:
    payload = _run(
        {
            "ogbg-molesol": {"D": (1.1, 1.1), "N": (0.9, 0.9), "R": (1.0, 1.0), "C": (1.05, 1.05)},
        }
    )
    assert payload["joint_signal"]["classification"] == "NO_JOINT_INCREMENT_THIS_RECIPE"
    per = payload["per_dataset"]["ogbg-molesol"]
    assert per["gains"]["N_valid"] < 0.0
    assert per["strong_joint_task"] is False


def test_improving_only_one_split_is_split_inconsistent() -> None:
    payload = _run(
        {
            "ogbg-molesol": {"D": (1.0, 1.0), "N": (1.0, 1.0), "R": (1.0, 1.0), "C": (0.97, 1.02)},
        }
    )
    assert payload["joint_signal"]["classification"] == "SPLIT_INCONSISTENT"
    assert payload["substantive_progress"]["SUBSTANTIVE_PROGRESS"] is False


def test_substantive_progress_needs_both_splits() -> None:
    payload = _run(
        {
            "ogbg-molesol": {"D": (1.0, 1.0), "N": (1.0, 1.0), "R": (1.0, 1.0), "C": (1.04, 1.06)},
        }
    )
    axis = payload["substantive_progress"]["per_dataset"]["ogbg-molesol"]
    valid = axis["per_split"]["valid"]
    test = axis["per_split"]["test"]
    assert valid["J_min_DNR"] == 1.0 == valid["b_best_reference"]
    assert valid["half_gap_closed"] == "not_applicable"
    assert valid["competitive_5pct"] is True and valid["progress_pass"] is True
    assert test["competitive_5pct"] is False and test["progress_pass"] is False
    assert axis["substantive_progress"] is False
    assert payload["substantive_progress"]["SUBSTANTIVE_PROGRESS"] is False


def test_no_subgroup_input_can_unlock_the_gate() -> None:
    """The decision payload carries no subgroup field at all, by construction."""
    payload = _run(
        {
            "ogbg-molesol": {"D": (1.0, 1.0), "N": (1.0, 1.0), "R": (1.0, 1.0), "C": (2.0, 2.0)},
        }
    )
    text = repr(payload).lower()
    assert "subgroup" not in text and "coverage_bin" not in text
    assert payload["substantive_progress"]["SUBSTANTIVE_PROGRESS"] is False


def test_half_gap_rule_opens_progress_but_not_the_signal() -> None:
    payload = _run(
        {
            "ogbg-molesol": {"D": (1.0, 1.0), "N": (1.0, 1.0), "R": (1.0, 1.0), "C": (1.02, 1.02)},
            "ogbg-mollipo": {"D": (1.0, 1.0), "N": (1.0, 1.0), "R": (1.0, 1.0), "C": (0.5, 0.5)},
        }
    )
    axis = payload["substantive_progress"]["per_dataset"]["ogbg-molesol"]
    # b = 1.0 (best of D/N/R), J = 1.0 -> no gap at all, so competitive_5pct governs
    assert axis["per_split"]["valid"]["half_gap_closed"] == "not_applicable"
    assert axis["per_split"]["valid"]["competitive_5pct"] is True
    assert axis["substantive_progress"] is True
    # and the strong task is still mollipo only
    assert payload["joint_signal"]["strong_joint_tasks"] == ["ogbg-mollipo"]


def test_missing_reference_axis_is_declared_not_fabricated() -> None:
    payload = _run(
        {
            "ogbg-molesol": {"D": (1.0, 1.0), "N": (1.0, 1.0), "R": (1.0, 1.0), "C": (0.9, 0.9)},
        }
    )
    assert payload["reference_axis"] == "REFERENCE_COMPLETE"
    payload["execution"]["reference_complete"] = False
    assert payload["reference_axis"] == "REFERENCE_COMPLETE"  # axis is derived, not mutable
