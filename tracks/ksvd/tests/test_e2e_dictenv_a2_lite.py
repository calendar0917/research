"""Tests for the E2E-DictEnv-A2-Lite compute-amendment round.

The lite round is a *user-requested contraction* of a frozen round: these tests
pin the amendment's frozen protocol (horizon, arms, thresholds, late-window
rule), the identity separation from the parent round, the reuse-by-reference
discipline (nothing is recomputed, no second PCA/OMP implementation), the
official-test blocker and the GPU1-only policy.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2_lite as lite
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a2 as a2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a2_lite as run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

MappingLike = Mapping[int, float]

CORE_PATH = Path(lite.__file__)
RUNNER_PATH = Path(run.__file__)
AMENDMENT_PATH = REPO_ROOT / lite.AMENDMENT_NOTE
A2_RESULTS = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_a2"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _cold_names(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _stage_choices() -> tuple[str, ...] | None:
    for node in ast.walk(ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not (isinstance(node.args[0], ast.Constant) and node.args[0].value == "stage"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices" and isinstance(keyword.value, ast.Tuple):
                return tuple(element.value for element in keyword.value.elts)
    return None


def _curve(rows: list[tuple[int, float]]) -> dict[int, float]:
    return {epoch: value for epoch, value in rows}


# ---------------------------------------------------------------------------
# 1. frozen amendment constants
# ---------------------------------------------------------------------------


def test_frozen_lite_constants():
    assert lite.SCREEN_HORIZON == 160
    assert lite.SCREEN_ARMS == ("REAL", "INDEP")
    assert "TOPO" in lite.FORBIDDEN_ARMS
    assert lite.STRONG_POSITIVE == 0.006
    assert lite.PCA_MATERIAL == 0.003
    assert lite.PCA_RANK == 32
    assert lite.LATE_WINDOW_START == 121
    assert lite.LATE_WINDOW_END == lite.SCREEN_HORIZON == 160
    assert lite.LATE_POSITIVE_FRACTION == 0.75
    assert lite.PARENT_ROUND == "E2E-DictEnv-A2"
    assert lite.PARENT_PROTOCOL_VERSION == "e2e_dictenv_a2"
    assert lite.PARENT_A2_COMMIT == "24d528635fab49c082115d90592cb7d5938eeb37"
    assert lite.PARENT_A2_PREREG_COMMIT == "1813f53"
    assert run.RESULTS_DIR == REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_a2_lite"
    assert run.A2_RESULTS_DIR == A2_RESULTS
    assert run.PROTOCOL_VERSION == "e2e_dictenv_a2_lite"
    assert run.HORIZON == 160
    assert set(lite.VERDICTS) == {
        "PAIRING_SIGNAL_WORTH_FULL_CONFIRMATION",
        "ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK",
        "NO_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D",
        "LITE_SCREEN_UNRESOLVED_PENDING_REPAIR",
    }
    assert lite.screen_arm_list() == ("REAL", "INDEP")


def test_amendment_defers_every_expensive_stage():
    deferred = lite.deferred_stage_record()
    assert all(value == "DEFERRED_PENDING_USER_AUTHORIZATION" for value in deferred.values())
    joined = " ".join(deferred)
    for needle in ("IHT", "E2E", "mechanism", "DenseTied", "seed 1", "official test",
                   "K64", "s12", "TOPO"):
        assert needle.lower() in joined.lower(), needle


def test_required_a2_identity_references_are_pinned():
    for name in (
        "dictionary_REAL",
        "dictionary_INDEP",
        "omp_REAL_train",
        "omp_REAL_valid",
        "omp_INDEP_train",
        "omp_INDEP_valid",
        "matched_init_REAL",
        "matched_init_INDEP",
        "cache_train_phi",
        "cache_valid_phi",
        "scaler_real",
        "scaler_indep",
    ):
        assert name in lite.REQUIRED_A2_IDENTITY_ENTRIES


# ---------------------------------------------------------------------------
# 2. integrity of the frozen record
# ---------------------------------------------------------------------------


def test_parent_preregistration_and_producers_are_unmodified():
    for relative, expected in lite.A2_PRODUCER_SHA256.items():
        path = REPO_ROOT / relative
        assert path.exists(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative


def test_amendment_note_records_reason_and_prior_metrics():
    text = AMENDMENT_PATH.read_text(encoding="utf-8")
    for needle in (
        "compute-budget contraction",
        "Not motivated by",
        "Metrics already observed before this amendment",
        "COMPUTE-BUDGET-TRUNCATED",
        "E2E-DictEnv-A2-Lite",
        "e2e_dictenv_a2_lite",
        "results/e2e_dictenv_a2_lite/",
        "DEFERRED_PENDING_USER_AUTHORIZATION",
        "0.006",
        "0.003",
        "160",
        "physical GPU1 only",
    ):
        assert needle in text, needle
    # the parent preregistration is never edited by the amendment
    assert "not edited" in text


def test_official_test_split_never_appears_in_lite_sources():
    for path in (CORE_PATH, RUNNER_PATH):
        literals = _string_literals(path)
        assert "test" not in literals
        assert "val" not in literals
    assert {"train", "valid"} <= _string_literals(RUNNER_PATH)
    assert RUNNER_PATH.read_text(encoding="utf-8").count('"official_test_loaded": False') >= 6


def test_stage_choices_are_complete():
    assert _stage_choices() == (
        "verify",
        "screen",
        "decision",
        "pca-screen",
        "pca-decision",
        "report",
        "smoke",
        "all",
    )
    assert "all" in _stage_choices()


# ---------------------------------------------------------------------------
# 3. reuse discipline (no second implementation)
# ---------------------------------------------------------------------------


def test_lite_core_defines_no_second_statistic_or_solver():
    cold = _cold_names(CORE_PATH)
    for forbidden in ("fit_pca_rank", "omp_codes", "tied_iht_codes", "svd", "eigh", "tsqr",
                      "train_arm"):
        assert forbidden not in cold, forbidden
    text = CORE_PATH.read_text(encoding="utf-8")
    assert "import sdb_v0" not in text
    assert "e2e_dictenv_a2 as a2" in text  # thresholds/labels copied from the frozen core
    for needle in ("np.linalg.svd", "np.linalg.eigh", "scipy.linalg"):
        assert needle not in text, needle


def test_lite_runner_delegates_to_the_frozen_producers():
    text = RUNNER_PATH.read_text(encoding="utf-8")
    for needle in (
        "a1run.train_arm",
        "a2run._pca_artifacts",
        "a2run._set_device_policy",
        "a2run.matched_init_enforced",
        "a2run.dictionary_override",
        "a2run.a1_emits_into",
        "a1run.arm_coordinate",
        "a1run.load_arm_dictionary",
    ):
        assert needle in text, needle
    # the screen horizon is the frozen constant, never a literal
    assert text.count("horizon=int(HORIZON)") >= 2
    assert "horizon=320" not in text
    # no recomputation of the frozen code matrix: the solver names are never *called*
    tree = ast.parse(text)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for forbidden in ("omp_codes", "fit_pca_rank", "tied_iht_codes"):
        assert forbidden not in called, forbidden
    defined = _cold_names(RUNNER_PATH)
    for forbidden in ("omp_codes", "fit_pca_rank", "tied_iht_codes", "train_arm"):
        assert forbidden not in defined, forbidden


def test_lite_dense_control_goes_through_the_audited_pca():
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "a2_emits_into" in text
    assert "a2run.RESULTS_DIR = root" in text  # only the A2 writer is redirected
    assert "sdb.fit_pca_rank" not in text


def test_lite_round_writes_only_its_own_result_directory():
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_a2_lite"' in text
    # every JSON/CSV writer is rooted at the lite directory or an explicit subdir
    assert text.count("_write_json(RESULTS_DIR") >= 5
    assert "A2_IDENTITY_PATH" in text and "_read_json(A2_IDENTITY_PATH)" in text
    assert "a2run._write_json" in text and "a2run._read_json" in text


def test_gpu_policy_is_delegated_to_the_frozen_enforcement(monkeypatch):
    seen = {}

    def fake(device: str):
        seen["device"] = device
        return "sentinel"

    monkeypatch.setattr(a2run, "_set_device_policy", fake)
    assert run._set_device_policy("cuda") == "sentinel"
    assert seen["device"] == "cuda"

    monkeypatch.undo()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError):
        run._set_device_policy("cuda")
    for value in ("0", "2", "0,1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        with pytest.raises(RuntimeError):
            run._set_device_policy("cuda")


# ---------------------------------------------------------------------------
# 4. curve parsing (fail-closed)
# ---------------------------------------------------------------------------


def test_read_curve_parses_a_real_frozen_curve():
    sample = next(
        iter(sorted((A2_RESULTS / "smoke" / "curves").glob("*_curve.csv"))), None
    )
    if sample is None:
        pytest.skip("no local frozen curve sample available")
    curve = lite.read_curve(sample)
    assert curve
    assert all(np.isfinite(value) for value in curve.values())


def test_read_curve_is_fail_closed(tmp_path: Path):
    missing = tmp_path / "missing_curve.csv"
    with pytest.raises(lite.CurveError):
        lite.read_curve(missing)
    assert lite.curve_or_none(missing) is None

    empty = tmp_path / "empty_curve.csv"
    empty.write_text("epoch,train_mae,valid_mae\n", encoding="utf-8")
    with pytest.raises(lite.CurveError):
        lite.read_curve(empty)

    nan = tmp_path / "nan_curve.csv"
    nan.write_text("epoch,valid_mae\n1,0.5\n2,nan\n", encoding="utf-8")
    with pytest.raises(lite.CurveError):
        lite.read_curve(nan)
    assert lite.curve_or_none(nan) is None

    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text("step,value\n1,0.5\n", encoding="utf-8")
    with pytest.raises(lite.CurveError):
        lite.read_curve(bad_header)

    good = tmp_path / "good_curve.csv"
    good.write_text(
        "epoch,train_mae,valid_mae,d_norm\n1,1.5,0.6,5.0\n2,1.4,0.5,5.1\n", encoding="utf-8"
    )
    assert lite.read_curve(good) == {1: 0.6, 2: 0.5}


# ---------------------------------------------------------------------------
# 5. late-window direction and the frozen screen rule
# ---------------------------------------------------------------------------


def test_late_direction_requires_a_stable_positive_window():
    real = _curve([(epoch, 1.0) for epoch in range(121, 161)])
    indep = _curve([(epoch, 1.1) for epoch in range(121, 161)])
    direction = lite.late_direction(indep, real)
    assert direction["stable"] is True
    assert direction["positive_fraction"] == 1.0
    assert direction["mean_delta"] == pytest.approx(0.1)
    assert direction["best_delta"] == pytest.approx(0.1)
    assert lite.late_direction(real, indep)["stable"] is False


def test_late_direction_rejects_a_minority_direction():
    real = _curve([(epoch, 1.0) for epoch in range(121, 161)])
    # 60% of the late epochs favour INDEP: below the 0.75 agreement bar
    indep = _curve([(epoch, 1.1 if index < 24 else 0.9) for index, epoch in enumerate(range(121, 161))])
    assert lite.late_direction(indep, real)["stable"] is False


def test_late_direction_rejects_a_single_lucky_epoch():
    real = _curve([(epoch, 1.0) for epoch in range(121, 161)])
    indep = _curve(
        [(epoch, 1.2 if epoch != 155 else 0.5) for epoch in range(121, 161)]
    )
    direction = lite.late_direction(indep, real)
    assert direction["positive_fraction"] == pytest.approx(39 / 40)
    assert direction["mean_delta"] > 0.0
    assert direction["best_delta"] < 0.0
    assert direction["stable"] is False


def test_late_direction_requires_paired_epochs():
    real = _curve([(epoch, 1.0) for epoch in range(121, 161)])
    indep = _curve([(epoch, 1.1) for epoch in range(121, 160)])
    with pytest.raises(lite.CurveError):
        lite.late_direction(indep, real)
    assert lite.paired_epochs(indep, real) == 39


def test_screen_decision_cells():
    stable = {"stable": True}
    unstable = {"stable": False}

    strong = lite.screen_decision(
        {"REAL": 1.000, "INDEP": 1.008}, direction=stable, curves_finite=True
    )
    assert strong["G_pair_screen"] == pytest.approx(0.008)
    assert strong["strong_positive"] is True
    assert strong["pca_authorised"] is False
    assert strong["verdict"] == lite.VERDICT_WORTH_FULL_CONFIRMATION

    boundary = lite.screen_decision(
        {"REAL": 1.000, "INDEP": 1.006}, direction=stable, curves_finite=True
    )
    assert boundary["strong_positive"] is True

    weak = lite.screen_decision(
        {"REAL": 1.000, "INDEP": 1.0059}, direction=stable, curves_finite=True
    )
    assert weak["strong_positive"] is False
    assert weak["pca_authorised"] is True
    assert weak["verdict"] == lite.STATE_PCA_AUTHORISED

    unstable = lite.screen_decision(
        {"REAL": 1.000, "INDEP": 1.020}, direction=unstable, curves_finite=True
    )
    assert unstable["strong_positive"] is False
    assert unstable["pca_authorised"] is True

    broken = lite.screen_decision(
        {"REAL": 1.000, "INDEP": 1.020}, direction=None, curves_finite=False
    )
    assert broken["pca_authorised"] is False
    assert broken["verdict"] == lite.VERDICT_UNRESOLVED


def test_pca_label_cells():
    good = lite.pca_label(0.003, curves_finite=True)
    assert good["primary_pass"] is True
    assert good["verdict"] == lite.VERDICT_COMPRESSION_BOTTLENECK

    weak = lite.pca_label(0.0029, curves_finite=True)
    assert weak["primary_pass"] is False
    assert weak["verdict"] == lite.VERDICT_NO_SIGNAL

    broken = lite.pca_label(0.05, curves_finite=False)
    assert broken["verdict"] == lite.VERDICT_UNRESOLVED


def test_screen_decision_never_reports_a_screen_value_as_a_frozen_gate():
    decision = lite.screen_decision(
        {"REAL": 1.000, "INDEP": 1.010}, direction={"stable": True}, curves_finite=True
    )
    assert decision["threshold"] == lite.STRONG_POSITIVE
    assert decision["threshold"] != lite.PCA_MATERIAL
    assert "screen" in decision["reason"]
    assert decision["horizon"] == 160


# ---------------------------------------------------------------------------
# 6. reuse-by-reference verification stage
# ---------------------------------------------------------------------------


def _require_local_a2_identity() -> dict:
    path = A2_RESULTS / "artifact_identity.json"
    if not path.exists():
        pytest.skip("local A2 artifact identity record unavailable")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_verify_stage_attaches_to_the_completed_identity(monkeypatch, tmp_path: Path):
    identity = _require_local_a2_identity()
    monkeypatch.setattr(lite, "PARENT_A2_COMMIT", identity["git_commit"])
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    payload = run.verify_stage()
    assert payload["passed"] is True
    assert payload["entries"]["a2_identity_file"]["all_passed"] is True
    assert payload["entries"]["a2_identity_file"]["sha256"]
    assert payload["entries"]["producer_freeze"]["passed"] is True
    assert payload["entries"]["amendment_note"]["sha256"]
    assert payload["entries"]["a2_identity_references"]["all_required_passed"] is True
    assert all(item["passed"] for item in payload["entries"]["dictionary_live"].values())
    codes = payload["entries"]["omp_cache_bytes"]
    assert set(codes) == {"REAL_train", "REAL_valid", "INDEP_train", "INDEP_valid"}
    assert all(item["passed"] for item in codes.values())
    assert codes["REAL_train"]["rows"] == a2run.TRACKED_CACHE_SHAPES["train"]["n_nodes"]
    assert codes["INDEP_valid"]["expected_shape"] == [
        a2run.TRACKED_CACHE_SHAPES["valid"]["n_nodes"],
        32,
    ]
    assert payload["official_test_loaded"] is False


@pytest.mark.slow
def test_verify_stage_fails_closed_on_a_foreign_parent(monkeypatch, tmp_path: Path):
    _require_local_a2_identity()
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(lite, "PARENT_A2_COMMIT", "0" * 40)
    with pytest.raises(RuntimeError):
        run.verify_stage()


# ---------------------------------------------------------------------------
# 7. decision / report wiring (synthetic screen)
# ---------------------------------------------------------------------------


def _fake_screen(
    tmp_path: Path,
    *,
    real_mae: float,
    indep_mae: float,
    real_curve: MappingLike | None = None,
    indep_curve: MappingLike | None = None,
) -> None:
    curves = tmp_path / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    for arm, label, mae in (("REAL", "R0", real_mae), ("INDEP", "I0", indep_mae)):
        (tmp_path / f"lite_omp_{arm.lower()}.json").write_text(
            json.dumps({"soup": {"soup_valid_mae": mae, "members": [1]},
                        "protocol_version": run.PROTOCOL_VERSION}),
            encoding="utf-8",
        )
    default_real = {epoch: 1.0 for epoch in range(1, 161)}
    default_indep = {epoch: 1.1 for epoch in range(1, 161)}
    for label, rows in (("R0", real_curve or default_real), ("I0", indep_curve or default_indep)):
        path = curves / f"lite_omp_{label}_curve.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("epoch,train_mae,train_rec,valid_mae,valid_rec,d_norm\n")
            for epoch, value in sorted(rows.items()):
                handle.write(f"{epoch},1.5,0.1,{value},0.2,5.0\n")


def test_decision_stage_wires_the_frozen_rule(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _fake_screen(tmp_path, real_mae=1.000, indep_mae=1.008)
    payload = run.decision_stage()
    assert payload["G_pair_screen"] == pytest.approx(0.008)
    assert payload["strong_positive"] is True
    assert payload["pca_authorised"] is False
    assert payload["verdict"] == lite.VERDICT_WORTH_FULL_CONFIRMATION
    assert payload["curves_finite"] is True
    assert payload["paired_epochs"] == 160
    assert payload["direction"]["stable"] is True
    assert payload["screen_horizon"] == 160
    assert (tmp_path / "lite_screen_decision.json").exists()
    assert payload["deferred_stages"] == lite.deferred_stage_record()
    assert payload["official_test_loaded"] is False


def test_decision_stage_authorises_the_dense_route_for_a_weak_screen(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _fake_screen(tmp_path, real_mae=1.000, indep_mae=1.002)
    payload = run.decision_stage()
    assert payload["G_pair_screen"] == pytest.approx(0.002)
    assert payload["strong_positive"] is False
    assert payload["pca_authorised"] is True
    assert payload["verdict"] == lite.STATE_PCA_AUTHORISED


def test_report_stage_stops_at_the_screen_and_lists_deferrals(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _fake_screen(tmp_path, real_mae=1.000, indep_mae=1.008)
    screen = run.decision_stage()
    assert screen["strong_positive"] is True
    payload = run.report_stage()
    assert payload["final_label"] == lite.VERDICT_WORTH_FULL_CONFIRMATION
    assert payload["stage_status"]["pca-screen"]["status"] == "NOT RUN"
    assert payload["stage_status"]["pca-decision"]["status"] == "NOT RUN"
    decision_text = (tmp_path / "DECISION.md").read_text(encoding="utf-8")
    report_text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert lite.VERDICT_WORTH_FULL_CONFIRMATION in decision_text
    assert "COMPUTE-BUDGET-TRUNCATED" in decision_text
    for name in lite.DEFERRED_STAGES:
        assert name in decision_text
        assert name in report_text
    assert "No official test data was loaded." in decision_text
    assert payload["official_test_loaded"] is False
