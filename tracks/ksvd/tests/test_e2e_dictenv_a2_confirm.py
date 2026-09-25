"""Tests for the E2E-DictEnv-A2-Confirm round.

The confirmation round is a *frozen confirmation*: these tests pin its frozen
preregistration (parent horizon 320, arms REAL/INDEP only, material bar 0.003,
TOPO practical tolerance 0.003, the four-label decision matrix), the
continuation rule (exact resume only if provable for every arm, never mixed),
the reuse-by-reference discipline (nothing is recomputed), the official-test
blocker and the GPU1-only policy.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Mapping

import pytest

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2 as a2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2_confirm as confirm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2_lite as lite
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a1 as a1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a2 as a2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a2_confirm as run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

MappingLike = Mapping[int, float]

CORE_PATH = Path(confirm.__file__)
RUNNER_PATH = Path(run.__file__)
PREREG_PATH = REPO_ROOT / confirm.PREREGISTRATION_NOTE
A2_RESULTS = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_a2"
LITE_RESULTS = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_a2_lite"

EXPECTED_STAGE_CHOICES = (
    "verify",
    "continuation",
    "screen",
    "arm",
    "decision",
    "report",
    "smoke",
    "all",
)


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


def _flat_curve(horizon: int, value: float, *, start: int = 1) -> dict[int, float]:
    return {epoch: value for epoch in range(int(start), int(horizon) + 1)}


def _write_curve(path: Path, curve: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,valid_mae"]
    lines += [f"{epoch},{float(curve[epoch])!r}" for epoch in sorted(curve)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_identity(tmp_path: Path, *, topo_soup: float = confirm.TOPO_320_SOUP,
                   passed: bool = True) -> None:
    payload = {
        "passed": bool(passed),
        "entries": {
            "topo_320_reference": {
                "source": "fake/topo.json",
                "sha256": "0" * 64,
                "soup_valid_mae": float(topo_soup),
                "best_valid_mae": float(confirm.TOPO_320_BEST),
                "best_epoch": 314,
                "horizon": int(confirm.TOPO_320_HORIZON),
                "never_retrained": True,
            }
        },
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifact_identity.json").write_text(json.dumps(payload), encoding="utf-8")


def _fake_arm(tmp_path: Path, arm: str, soup: float, *, best: float | None = None,
              mem_start: int = 300) -> dict[str, float]:
    payload = {
        "round": confirm.ROUND,
        "protocol_version": confirm.PROTOCOL_VERSION,
        "stage": "omp",
        "arm": arm,
        "tag": f"{arm.lower()}_320",
        "horizon": int(confirm.CONFIRM_HORIZON),
        "soup": {
            "members": list(range(mem_start, mem_start + 5)),
            "member_valid_mae": [float(soup)] * 5,
            "soup_valid_mae": float(soup),
        },
        "best_valid_mae": float(soup if best is None else best),
        "best_epoch": int(mem_start),
        "frozen_dictionary": True,
        "dictionary_sha256_f32": a2run.EXPECTED_DICT_SHA[arm],
        "seconds": 1.0,
        "official_test_loaded": False,
    }
    (tmp_path / f"{arm.lower()}_320.json").write_text(json.dumps(payload), encoding="utf-8")
    return {"soup": float(soup)}


def _fake_continuation(tmp_path: Path, mode: str = confirm.CONTINUATION_FRESH) -> None:
    payload = {
        "mode": str(mode),
        "mixing_prohibited": True,
        "resume_authorised": mode == confirm.CONTINUATION_RESUME,
        "missing": {},
        "reason": "fake continuation record for tests",
    }
    (tmp_path / "continuation_mode.json").write_text(json.dumps(payload), encoding="utf-8")


def _prepare_decision_dir(
    tmp_path: Path, *, real: float, indep: float, topo: float = confirm.TOPO_320_SOUP,
    mode: str = confirm.CONTINUATION_FRESH,
) -> None:
    _fake_identity(tmp_path, topo_soup=topo)
    _fake_continuation(tmp_path, mode=mode)
    _fake_arm(tmp_path, "REAL", real, best=real + 0.006, mem_start=310)
    _fake_arm(tmp_path, "INDEP", indep, best=indep + 0.006, mem_start=300)
    curves = tmp_path / "curves"
    _write_curve(curves / "real_320_curve.csv", _flat_curve(320, float(real) + 0.05))
    _write_curve(curves / "indep_320_curve.csv", _flat_curve(320, float(indep) + 0.05))


# ---------------------------------------------------------------------------
# 1. frozen preregistration constants
# ---------------------------------------------------------------------------


def test_frozen_confirm_constants():
    assert confirm.ROUND == "E2E-DictEnv-A2-Confirm"
    assert confirm.PROTOCOL_VERSION == "e2e_dictenv_a2_confirm"
    assert confirm.CONFIRM_ARMS == ("REAL", "INDEP")
    assert confirm.CONFIRM_LABEL == {"REAL": "R0", "INDEP": "I0"}
    assert "TOPO" in confirm.FORBIDDEN_ARMS
    assert confirm.PARENT_ROUND == a2.ROUND
    assert confirm.PREDECESSOR_ROUND == lite.ROUND
    assert confirm.PARENT_A2_COMMIT == lite.PARENT_A2_COMMIT
    assert confirm.PARENT_A2_PREREG_COMMIT == "1813f53"


def test_confirmation_horizon_is_the_parent_horizon():
    assert confirm.CONFIRM_HORIZON == 320
    assert confirm.CONFIRM_HORIZON == int(a1run.HORIZON)
    assert confirm.LATE_WINDOW_START == 241
    assert confirm.LATE_WINDOW_END == confirm.CONFIRM_HORIZON


def test_thresholds_are_the_frozen_values():
    assert confirm.MATERIAL == 0.003
    assert confirm.MATERIAL == float(a2.MATERIAL)
    assert confirm.TOPO_TOLERANCE == 0.003
    assert confirm.GAP_CHANGE_TOLERANCE == 0.001


def test_topo_reference_pins_match_the_completed_parent_arm():
    assert confirm.TOPO_320_SOUP == 0.12681294702464949
    assert confirm.TOPO_320_BEST == 0.13136472144449363
    assert confirm.TOPO_320_HORIZON == 320
    assert confirm.TOPO_320_ARM == "TOPO"
    payload = json.loads((A2_RESULTS / "omp_screen_topo.json").read_text(encoding="utf-8"))
    assert float(payload["soup"]["soup_valid_mae"]) == confirm.TOPO_320_SOUP
    assert float(payload["best_valid_mae"]) == confirm.TOPO_320_BEST
    assert int(payload["horizon"]) == confirm.TOPO_320_HORIZON
    assert payload["arm"] == "TOPO"
    assert payload["frozen_dictionary"] is True


def test_lite_160_reference_pins_match_the_completed_screen():
    assert confirm.LITE_160_HORIZON == 160
    assert confirm.LITE_160_SOUP == {
        "REAL": 0.15437116196932038,
        "INDEP": 0.15993232336913935,
    }
    for arm in confirm.CONFIRM_ARMS:
        tag = confirm.LITE_160_TAG[arm]
        payload = json.loads((LITE_RESULTS / f"{tag}.json").read_text(encoding="utf-8"))
        assert int(payload["horizon"]) == confirm.LITE_160_HORIZON
        assert float(payload["soup"]["soup_valid_mae"]) == confirm.LITE_160_SOUP[arm]
    gap_160 = confirm.LITE_160_SOUP["INDEP"] - confirm.LITE_160_SOUP["REAL"]
    assert gap_160 == pytest.approx(0.005561161399818965, abs=0.0)


def test_verdict_set_is_the_frozen_four():
    assert set(confirm.VERDICTS) == {
        "ATTRIBUTED_PAIRING_SUPPORTED_AND_COMPETITIVE",
        "ATTRIBUTED_PAIRING_SUPPORTED_BUT_NOT_COMPETITIVE",
        "NO_CONFIRMED_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D",
        "ARTIFACT_IDENTITY_FAILURE",
    }
    assert confirm.VERDICT_UNUSABLE not in confirm.VERDICTS


def test_deferred_stages_cover_the_preregistration_list():
    joined = " ".join(confirm.DEFERRED_STAGES).lower()
    for needle in ("seed 1", "pca32", "continuity-v2", "iht", "e2e",
                   "code-pairing-removal", "node-only", "edge-only", "densetied",
                   "topo", "k64", "s12", "official test", "sweep"):
        assert needle.lower() in joined, needle
    record = confirm.deferred_stage_record()
    assert set(record) == set(confirm.DEFERRED_STAGES)
    assert all(value == "DEFERRED_PENDING_USER_AUTHORIZATION" for value in record.values())


def test_required_resume_state_covers_the_preregistration_items():
    assert set(confirm.REQUIRED_RESUME_STATE) == {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "current_epoch",
        "rng_states",
        "loader_order_state",
        "soup_member_states_for_full_trajectory",
    }
    assert set(confirm.CONTINUATION_MODES) == {"fresh_matched_320", "exact_resume_161_320"}


def test_confirm_arm_list_blocks_topo(monkeypatch):
    assert confirm.confirm_arm_list() == ("REAL", "INDEP")
    monkeypatch.setattr(confirm, "CONFIRM_ARMS", ("REAL", "TOPO"))
    with pytest.raises(RuntimeError):
        confirm.confirm_arm_list()


def test_producer_pins_and_preregistration_exist():
    assert PREREG_PATH.exists()
    assert set(confirm.A2_PRODUCER_SHA256) == set(lite.A2_PRODUCER_SHA256)
    for relative in confirm.A2_PRODUCER_SHA256:
        assert (REPO_ROOT / relative).exists()


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
        assert name in confirm.REQUIRED_A2_IDENTITY_ENTRIES, name


# ---------------------------------------------------------------------------
# 2. sign conventions
# ---------------------------------------------------------------------------


def test_topo_delta_sign_convention():
    assert confirm.topo_delta(0.130, 0.120) == pytest.approx(0.010)
    assert confirm.topo_delta(0.130, 0.120) > 0.0  # REAL worse than TOPO
    assert confirm.topo_delta(0.100, 0.120) == pytest.approx(-0.020)
    assert confirm.topo_delta(0.100, 0.120) < 0.0  # REAL better than TOPO
    assert confirm.topo_delta(0.120, 0.120) == 0.0


def test_g_pair_sign_convention_is_indep_minus_real():
    decision = confirm.confirm_decision(
        0.004, 0.0, identity_ok=True, curves_finite=True
    )
    assert decision["G_pair_320"] == pytest.approx(0.004)
    assert decision["primary_pass"] is True
    negative = confirm.confirm_decision(
        -0.002, 0.0, identity_ok=True, curves_finite=True
    )
    assert negative["primary_pass"] is False
    assert negative["verdict"] == confirm.VERDICT_NO_SIGNAL


# ---------------------------------------------------------------------------
# 3. the frozen decision matrix
# ---------------------------------------------------------------------------


def test_decision_matrix_case_a():
    decision = confirm.confirm_decision(0.0029, 0.010, identity_ok=True, curves_finite=True)
    assert decision["case"] == "A"
    assert decision["verdict"] == confirm.VERDICT_NO_SIGNAL
    assert decision["primary_pass"] is False
    assert decision["competitive"] is None


def test_decision_matrix_case_b():
    decision = confirm.confirm_decision(0.0031, 0.0030, identity_ok=True, curves_finite=True)
    assert decision["case"] == "B"
    assert decision["verdict"] == confirm.VERDICT_COMPETITIVE
    assert decision["primary_pass"] is True
    assert decision["competitive"] is True


def test_decision_matrix_case_c():
    decision = confirm.confirm_decision(0.0111, 0.0031, identity_ok=True, curves_finite=True)
    assert decision["case"] == "C"
    assert decision["verdict"] == confirm.VERDICT_NOT_COMPETITIVE
    assert decision["primary_pass"] is True
    assert decision["competitive"] is False


def test_decision_matrix_boundaries_are_inclusive():
    # exactly at the material bar: still material
    at_bar = confirm.confirm_decision(0.003, 0.0, identity_ok=True, curves_finite=True)
    assert at_bar["primary_pass"] is True
    assert at_bar["verdict"] == confirm.VERDICT_COMPETITIVE
    # exactly at the practical tolerance: still competitive
    at_tolerance = confirm.confirm_decision(0.004, 0.003, identity_ok=True, curves_finite=True)
    assert at_tolerance["competitive"] is True
    assert at_tolerance["verdict"] == confirm.VERDICT_COMPETITIVE
    # just above the tolerance: not competitive
    just_above = confirm.confirm_decision(0.004, 0.0030001, identity_ok=True, curves_finite=True)
    assert just_above["competitive"] is False
    assert just_above["verdict"] == confirm.VERDICT_NOT_COMPETITIVE


def test_decision_matrix_identity_failure():
    decision = confirm.confirm_decision(0.5, -0.5, identity_ok=False, curves_finite=True)
    assert decision["verdict"] == confirm.VERDICT_IDENTITY_FAILURE
    assert decision["primary_pass"] is False
    assert decision["case"] == "IDENTITY"


def test_decision_matrix_unusable_curves_is_not_a_scientific_label():
    decision = confirm.confirm_decision(0.5, -0.5, identity_ok=True, curves_finite=False)
    assert decision["verdict"] == confirm.VERDICT_UNUSABLE
    assert decision["verdict"] not in confirm.VERDICTS
    assert decision["primary_pass"] is False


# ---------------------------------------------------------------------------
# 4. diagnostics
# ---------------------------------------------------------------------------


def test_late_window_stats_values():
    curve_real = _flat_curve(320, 0.20, start=241)
    curve_indep = _flat_curve(320, 0.20, start=241)
    for epoch in range(300, 310):  # ten epochs favour REAL by 0.01
        curve_indep[epoch] = 0.21
    curve_indep[320] = 0.215  # last epoch favours REAL by 0.015
    stats = confirm.late_window_stats(curve_indep, curve_real)
    assert stats["window"] == [241, 320]
    assert stats["n_epochs"] == 80
    assert stats["positive_fraction"] == pytest.approx(11 / 80)
    assert stats["mean_delta"] == pytest.approx((10 * 0.01 + 0.015) / 80)
    assert stats["delta_first"] == pytest.approx(0.0)
    assert stats["delta_last"] == pytest.approx(0.015)
    assert stats["median_delta"] == pytest.approx(0.0)
    assert stats["min_delta"] == pytest.approx(0.0)
    assert stats["max_delta"] == pytest.approx(0.015)
    assert "diagnostic only" in stats["note"]
    # a window that favours INDEP throughout keeps the sign convention
    negative = confirm.late_window_stats(_flat_curve(320, 0.19, start=241), curve_real)
    assert negative["positive_fraction"] == 0.0
    assert negative["mean_delta"] == pytest.approx(-0.01)
    assert negative["median_delta"] == pytest.approx(-0.01)


def test_late_window_stats_fail_closed():
    curve_real = _flat_curve(320, 0.20, start=241)
    with pytest.raises(lite.CurveError):
        confirm.late_window_stats(_flat_curve(319, 0.20, start=241), curve_real)
    broken = _flat_curve(320, 0.20, start=241)
    broken[300] = float("nan")
    with pytest.raises(lite.CurveError):
        confirm.late_window_stats(broken, curve_real)


def test_gap_change_classifications():
    assert confirm.gap_change(0.005, 0.007)["classification"] == "WIDENED"
    assert confirm.gap_change(0.005, 0.003)["classification"] == "SHRUNK"
    assert confirm.gap_change(0.005, 0.0055)["classification"] == "KEPT"
    assert confirm.gap_change(0.005, 0.005)["classification"] == "KEPT"
    assert confirm.gap_change(0.005, 0.007)["change"] == pytest.approx(0.002)


def test_gap_change_sign_reversal():
    reversed_gap = confirm.gap_change(0.005561, -0.001)
    assert reversed_gap["sign_reversed"] is True
    assert reversed_gap["classification"] == "SHRUNK"
    kept_sign = confirm.gap_change(0.005561, 0.009)
    assert kept_sign["sign_reversed"] is False
    assert kept_sign["classification"] == "WIDENED"


# ---------------------------------------------------------------------------
# 5. continuation regime (no mixing)
# ---------------------------------------------------------------------------


def _complete_evidence() -> dict[str, dict[str, object]]:
    return {
        item: {"present_and_provable": True} for item in confirm.REQUIRED_RESUME_STATE
    }


def test_continuation_decision_requires_every_item_for_every_arm():
    both = {arm: _complete_evidence() for arm in confirm.CONFIRM_ARMS}
    assert confirm.continuation_decision(both)["mode"] == confirm.CONTINUATION_RESUME
    one_missing = {arm: _complete_evidence() for arm in confirm.CONFIRM_ARMS}
    one_missing["INDEP"]["optimizer_state"] = {"present_and_provable": False}
    decision = confirm.continuation_decision(one_missing)
    assert decision["mode"] == confirm.CONTINUATION_FRESH
    assert decision["missing"]["INDEP"] == ["optimizer_state"]
    assert decision["missing"]["REAL"] == []
    empty = confirm.continuation_decision({})
    assert empty["mode"] == confirm.CONTINUATION_FRESH
    assert set(empty["missing"]) == set(confirm.CONFIRM_ARMS)


def test_continuation_decision_never_mixes_regimes():
    both = {arm: _complete_evidence() for arm in confirm.CONFIRM_ARMS}
    decision = confirm.continuation_decision(both)
    assert decision["mode"] in confirm.CONTINUATION_MODES
    assert decision["mixing_prohibited"] is True
    assert isinstance(decision["reason"], str) and decision["reason"]
    # a single mode is returned for both arms by construction (no per-arm field)
    assert not any(key.startswith("mode_") for key in decision)


def test_complete_curve_is_fail_closed(tmp_path: Path):
    good = tmp_path / "good.csv"
    _write_curve(good, _flat_curve(320, 0.1))
    assert run._complete_curve(good, 320) is True
    assert run._complete_curve(good, 321) is False
    short = tmp_path / "short.csv"
    _write_curve(short, _flat_curve(319, 0.1))
    assert run._complete_curve(short, 320) is False
    empty = tmp_path / "empty.csv"
    empty.write_text("epoch,valid_mae\n", encoding="utf-8")
    assert run._complete_curve(empty, 320) is False
    assert run._complete_curve(tmp_path / "missing.csv", 320) is False
    nan = tmp_path / "nan.csv"
    _write_curve(nan, {**_flat_curve(320, 0.1), 5: float("nan")})
    assert run._complete_curve(nan, 320) is False


def test_arm_tag_uses_the_confirm_horizon():
    assert run._arm_tag("REAL") == "real_320"
    assert run._arm_tag("INDEP") == "indep_320"


# ---------------------------------------------------------------------------
# 6. stage wiring, reuse discipline and safety
# ---------------------------------------------------------------------------


def test_arm_stage_is_guarded_and_single_arm(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _fake_identity(tmp_path)
    _fake_continuation(tmp_path)
    with pytest.raises(RuntimeError, match="not authorised"):
        run.arm_stage("TOPO", device="cuda")
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "unverified")
    (tmp_path / "unverified").mkdir()
    with pytest.raises(RuntimeError, match="verify stage"):
        run.arm_stage("INDEP", device="cuda")
    # the parallel entry point shares the no-mixing gate with the sequential one
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    _fake_continuation(tmp_path, mode=confirm.CONTINUATION_RESUME)
    with pytest.raises(RuntimeError, match="mix regimes"):
        run.arm_stage("INDEP", device="cuda")
    assert "arm" in EXPECTED_STAGE_CHOICES


def test_screen_refuses_to_mix_regimes(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _fake_identity(tmp_path)
    _fake_continuation(tmp_path, mode=confirm.CONTINUATION_RESUME)
    with pytest.raises(RuntimeError, match="mix regimes"):
        run.screen_stage(device="cuda")


def test_screen_refuses_without_verified_identity(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    with pytest.raises(RuntimeError, match="verify stage"):
        run.screen_stage(device="cuda")


def test_decision_stage_refuses_unusable_curves(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _prepare_decision_dir(tmp_path, real=0.15, indep=0.16)
    (tmp_path / "curves" / "indep_320_curve.csv").unlink()
    with pytest.raises(RuntimeError):
        run.decision_stage()
    assert not (tmp_path / "decision.json").exists()


def test_decision_stage_wires_the_frozen_matrix(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _prepare_decision_dir(tmp_path, real=0.100, indep=0.104, topo=0.098)
    payload = run.decision_stage()
    assert payload["G_pair_320"] == pytest.approx(0.004)
    assert payload["Delta_vs_TOPO"] == pytest.approx(0.002)
    assert payload["verdict"] == confirm.VERDICT_COMPETITIVE
    assert payload["case"] == "B"
    assert payload["official_test_loaded"] is False
    paired = json.loads((tmp_path / "paired_analysis.json").read_text(encoding="utf-8"))
    assert paired["G_pair_320"] == pytest.approx(0.004)
    assert paired["Delta_vs_TOPO"] == pytest.approx(0.002)
    assert paired["topo_reference"]["never_retrained"] is True
    assert paired["late_window"]["n_epochs"] == 80
    assert paired["from_160_to_320"]["G_pair"]["G_pair_320"] == pytest.approx(0.004)
    assert paired["continuation_mode"] == confirm.CONTINUATION_FRESH


def test_decision_stage_refuses_a_foreign_dictionary(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _prepare_decision_dir(tmp_path, real=0.100, indep=0.104)
    payload = json.loads((tmp_path / "real_320.json").read_text(encoding="utf-8"))
    payload["dictionary_sha256_f32"] = "0" * 64
    (tmp_path / "real_320.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen dictionary"):
        run.decision_stage()


def test_report_stage_records_the_verdict_and_defers_everything_else(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _prepare_decision_dir(tmp_path, real=0.100, indep=0.104, topo=0.098)
    decision = run.decision_stage()
    payload = run.report_stage()
    assert payload["verdict"] == decision["verdict"]
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "DECISION.md").exists()
    text = (tmp_path / "DECISION.md").read_text(encoding="utf-8")
    assert confirm.VERDICT_COMPETITIVE in text
    assert "DEFERRED_PENDING_USER_AUTHORIZATION" in text
    status = json.loads((tmp_path / "stage_status.json").read_text(encoding="utf-8"))
    assert status["official_test_loaded"] is False
    deferred = [
        name for name, item in status["stages"].items()
        if item.get("status") == "NOT RUN"
    ]
    assert len(deferred) >= 10
    assert "seed 1 replication" in deferred


def test_gpu_policy_is_delegated_and_enforced(monkeypatch):
    monkeypatch.setattr(a2run, "_set_device_policy", lambda device: ("delegated", device))
    assert run._set_device_policy("cuda") == ("delegated", "cuda")
    monkeypatch.undo()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError):
        run._set_device_policy("cuda")
    for value in ("0", "0,1", "2", ""):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        with pytest.raises(RuntimeError):
            run._set_device_policy("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    device = run._set_device_policy("cpu")
    assert str(device) == "cpu"


def test_official_test_split_never_appears_in_confirm_sources():
    for path in (CORE_PATH, RUNNER_PATH):
        literals = _string_literals(path)
        assert "test" not in literals
        assert "val" not in literals
    assert {"train", "valid"} <= _string_literals(RUNNER_PATH)
    assert RUNNER_PATH.read_text(encoding="utf-8").count('"official_test_loaded": False') >= 6


def test_stage_choices_are_complete():
    assert _stage_choices() == EXPECTED_STAGE_CHOICES


def test_no_second_implementation_of_frozen_machinery():
    names = _cold_names(RUNNER_PATH) | _cold_names(CORE_PATH)
    for forbidden in (
        "train_arm", "fit_dictionary", "fit_ksvd", "fit_pca", "fit_pca_rank",
        "omp_codes", "tied_iht_codes", "build_model", "evaluate", "make_env_loader",
        "normalized_dictionary", "train_e2e",
    ):
        assert forbidden not in names, forbidden
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "a1run.train_arm(" in text
    assert "a2run._set_device_policy" in text
    assert "a2run.matched_init_enforced" in text
    assert "fit_pca" not in text


def test_fresh_regime_is_the_implemented_one():
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "fresh matched 320" in text
    assert "no pseudo-resume" in text or "no pseudo-resume" in CORE_PATH.read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 7. the local artifact chain (slow: reads the completed rounds)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_verify_stage_attaches_to_completed_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    payload = run.verify_stage()
    assert payload["passed"] is True
    entries = payload["entries"]
    assert entries["a2_identity_file"]["all_passed"] is True
    assert entries["a2_identity_references"]["all_required_passed"] is True
    assert entries["producer_freeze"]["passed"] is True
    assert entries["topo_320_reference"]["passed"] is True
    assert entries["topo_320_reference"]["curve_complete"] is True
    assert all(item["passed"] for item in entries["dictionary_live"].values())
    assert all(item["passed"] for item in entries["omp_cache_bytes"].values())
    assert all(item["passed"] for item in entries["lite_160_reference"].values())
    assert payload["official_test_loaded"] is False


@pytest.mark.slow
def test_verify_stage_fails_closed_on_a_moved_topo_reference(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    monkeypatch.setattr(confirm, "TOPO_320_SOUP", 0.0)
    with pytest.raises(RuntimeError, match="ARTIFACT_IDENTITY_FAILURE"):
        run.verify_stage()
    payload = json.loads((tmp_path / "artifact_identity.json").read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["entries"]["topo_320_reference"]["passed"] is False


@pytest.mark.slow
def test_continuation_stage_records_the_fresh_regime(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "STAGE_STATUS", {})
    _fake_identity(tmp_path)
    payload = run.continuation_stage()
    assert payload["mode"] == confirm.CONTINUATION_FRESH
    assert payload["resume_authorised"] is False
    assert payload["mixing_prohibited"] is True
    for arm in confirm.CONFIRM_ARMS:
        missing = payload["missing"][arm]
        assert "model_state" not in missing
        assert "current_epoch" not in missing
        assert "optimizer_state" in missing
        assert "rng_states" in missing
        assert "soup_member_states_for_full_trajectory" in missing
        detail = payload["per_arm"][arm]["evidence"]
        assert detail["model_state"]["present_and_provable"] is True
        assert detail["optimizer_state"]["present_and_provable"] is False
        assert payload["per_arm"][arm]["curve"]["final_epoch"] == 160
    trainer = payload["trainer_capability"]
    assert trainer["accepts_resume_argument"] is False
    assert trainer["writes_optimizer_state"] is False
    assert trainer["consistent_with_missing_state"] is True
    assert payload["official_test_loaded"] is False
    assert (tmp_path / "continuation_mode.json").exists()
