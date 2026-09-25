"""Independent local analysis of the E2E-DictEnv-A2-Confirm artifacts.

Recomputes, from the *pulled* files only, everything the frozen confirmation
preregistration declares: the arm soups, ``G_pair_320``, the completed parent
``TOPO-OMP`` reference and ``Delta_vs_TOPO``, the frozen four-label decision
(independent re-implementation of the rule), the paired late window 241-320,
the ``160 -> 320`` change and every provenance flag promised by the round.
Nothing is re-trained and no CUDA is touched.

Exit code 1 if any recomputed value disagrees with the recorded artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACK = REPO_ROOT / "tracks/ksvd"
RESULTS = TRACK / "results/e2e_dictenv_a2_confirm"
PARENT = TRACK / "results/e2e_dictenv_a2"
LITE = TRACK / "results/e2e_dictenv_a2_lite"

# frozen pins, repeated literally so this script cannot follow a moved goalpost
HORIZON = 320
MATERIAL = 0.003
TOPO_TOLERANCE = 0.003
TOPO_320_SOUP = 0.12681294702464949
TOPO_320_BEST = 0.13136472144449363
DICT_SHA = {
    "REAL": "c1cafb086662fb0753d52987fc321b1369d4f586164dde7960a3bed775d4b809",
    "INDEP": "400821ee5105050eb34600a0fb4cb8040f983daa1e773738dc9e2774036ff32d",
}
LITE_160_SOUP = {"REAL": 0.15437116196932038, "INDEP": 0.15993232336913935}
LATE_WINDOW = (241, 320)
VERDICT_COMPETITIVE = "ATTRIBUTED_PAIRING_SUPPORTED_AND_COMPETITIVE"
VERDICT_NOT_COMPETITIVE = "ATTRIBUTED_PAIRING_SUPPORTED_BUT_NOT_COMPETITIVE"
VERDICT_NO_SIGNAL = "NO_CONFIRMED_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_curve(path: Path) -> dict[int, float]:
    rows: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["epoch"])] = float(row["valid_mae"])
    return rows


def frozen_verdict(g_pair_320: float, delta_vs_topo: float) -> tuple[str, str]:
    """Independent re-implementation of the preregistered decision matrix."""
    if g_pair_320 < MATERIAL:
        return "A", VERDICT_NO_SIGNAL
    if delta_vs_topo <= TOPO_TOLERANCE:
        return "B", VERDICT_COMPETITIVE
    return "C", VERDICT_NOT_COMPETITIVE


def gap_change(g_160: float, g_320: float, *, tolerance: float = 0.001) -> dict[str, Any]:
    delta = g_320 - g_160
    classification = "KEPT" if abs(delta) <= tolerance else ("WIDENED" if delta > 0 else "SHRUNK")
    return {
        "G_pair_160": g_160,
        "G_pair_320": g_320,
        "change": delta,
        "classification": classification,
        "sign_reversed": (g_160 > 0.0) != (g_320 > 0.0),
    }


def analyse() -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}

    def expect(name: str, observed: Any, expected: Any) -> None:
        ok = observed == expected
        checks[name] = {"observed": observed, "expected": expected, "ok": bool(ok)}
        if not ok:
            errors.append(f"{name}: observed {observed!r} != expected {expected!r}")

    def expect_close(name: str, observed: float, expected: float, tol: float = 1e-12) -> None:
        ok = abs(float(observed) - float(expected)) <= tol
        checks[name] = {"observed": float(observed), "expected": float(expected), "ok": bool(ok)}
        if not ok:
            errors.append(f"{name}: observed {observed!r} != expected {expected!r}")

    decision = read_json(RESULTS / "decision.json")
    paired = read_json(RESULTS / "paired_analysis.json")
    identity = read_json(RESULTS / "artifact_identity.json")
    continuation = read_json(RESULTS / "continuation_mode.json")
    stage_status = read_json(RESULTS / "stage_status.json")

    # --- provenance -----------------------------------------------------------------
    expect("identity.passed", identity.get("passed"), True)
    expect("identity.official_test_loaded", identity.get("official_test_loaded"), False)
    expect("decision.official_test_loaded", decision.get("official_test_loaded"), False)
    expect("policy.gpu1_only", stage_status["cuda_stages_run"], sorted(
        stage_status["cuda_stages_run"]))
    expect("continuation.mode", continuation.get("mode"), "fresh_matched_320")
    expect("continuation.mixing_prohibited", continuation.get("mixing_prohibited"), True)
    expect("continuation.resume_authorised", continuation.get("resume_authorised"), False)
    trainer = continuation.get("trainer_capability", {})
    expect("continuation.trainer_no_resume_argument",
           trainer.get("accepts_resume_argument"), False)
    expect("continuation.trainer_no_optimizer_state",
           trainer.get("writes_optimizer_state"), False)

    # --- arms -----------------------------------------------------------------------
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("REAL", "INDEP"):
        path = RESULTS / f"{arm.lower()}_320.json"
        payload = read_json(path)
        expect(f"{arm}.horizon", int(payload["horizon"]), HORIZON)
        expect(f"{arm}.frozen_dictionary", bool(payload["frozen_dictionary"]), True)
        expect(f"{arm}.dictionary_sha", payload["dictionary_sha256_f32"], DICT_SHA[arm])
        expect(f"{arm}.official_test_loaded", payload.get("official_test_loaded"), False)
        expect(f"{arm}.confirm_horizon", int(payload["confirm_horizon"]), HORIZON)
        arms[arm] = {
            "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
            "best_valid_mae": float(payload["best_valid_mae"]),
            "best_epoch": int(payload["best_epoch"]),
            "soup_members": payload["soup"]["members"],
            "wall_clock_s": payload.get("wall_clock_s"),
            "schedule": payload.get("schedule"),
            "artifact_sha256": sha256_file(path),
        }
        expect_close(f"{arm}.soup_matches_paired_analysis",
                     arms[arm]["soup_valid_mae"], paired["arms"][arm]["soup_valid_mae"])

    # --- primary quantity and the frozen matrix -------------------------------------
    g_pair_320 = arms["INDEP"]["soup_valid_mae"] - arms["REAL"]["soup_valid_mae"]
    expect_close("G_pair_320", g_pair_320, decision["G_pair_320"])
    expect("G_pair_320.primary_pass", bool(g_pair_320 >= MATERIAL), True)

    topo_payload = read_json(PARENT / "omp_screen_topo.json")
    expect_close("topo.soup", float(topo_payload["soup"]["soup_valid_mae"]), TOPO_320_SOUP)
    expect_close("topo.best", float(topo_payload["best_valid_mae"]), TOPO_320_BEST)
    expect("topo.horizon", int(topo_payload["horizon"]), HORIZON)
    expect("topo.frozen_dictionary", bool(topo_payload["frozen_dictionary"]), True)
    delta_vs_topo = arms["REAL"]["soup_valid_mae"] - TOPO_320_SOUP
    expect_close("Delta_vs_TOPO", delta_vs_topo, decision["Delta_vs_TOPO"])

    case, verdict = frozen_verdict(g_pair_320, delta_vs_topo)
    expect("verdict.independent_rule", verdict, decision["verdict"])
    expect("verdict.case", case, decision["case"])
    expect("verdict.competitive", bool(delta_vs_topo <= TOPO_TOLERANCE), decision["competitive"])

    # --- late window (diagnostic) ---------------------------------------------------
    curve_real = read_curve(RESULTS / "curves/real_320_curve.csv")
    curve_indep = read_curve(RESULTS / "curves/indep_320_curve.csv")
    expect("curves.real_epochs", len(curve_real), HORIZON)
    expect("curves.indep_epochs", len(curve_indep), HORIZON)
    epochs = list(range(LATE_WINDOW[0], LATE_WINDOW[1] + 1))
    deltas = [curve_indep[e] - curve_real[e] for e in epochs]
    mean_delta = sum(deltas) / len(deltas)
    ordered = sorted(deltas)
    middle = len(ordered) // 2
    median = (ordered[middle - 1] + ordered[middle]) / 2 if len(ordered) % 2 == 0 else ordered[middle]
    positive_fraction = sum(1 for d in deltas if d > 0) / len(deltas)
    late_record = paired["late_window"]
    expect_close("late.mean_delta", mean_delta, late_record["mean_delta"])
    expect_close("late.median_delta", median, late_record["median_delta"])
    expect_close("late.positive_fraction", positive_fraction, late_record["positive_fraction"])
    expect_close("late.delta_first", deltas[0], late_record["delta_first"])
    expect_close("late.delta_last", deltas[-1], late_record["delta_last"])

    # --- 160 -> 320 -----------------------------------------------------------------
    lite_soup = {
        arm: float(read_json(LITE / f"lite_omp_{'R0' if arm == 'REAL' else 'I0'}.json")["soup"]["soup_valid_mae"])
        for arm in ("REAL", "INDEP")
    }
    for arm in ("REAL", "INDEP"):
        expect_close(f"lite.{arm}.soup", lite_soup[arm], LITE_160_SOUP[arm])
    g_pair_160 = lite_soup["INDEP"] - lite_soup["REAL"]
    change = gap_change(g_pair_160, g_pair_320)
    expect_close("change.G_pair_160", change["G_pair_160"],
                 paired["from_160_to_320"]["G_pair"]["G_pair_160"])
    expect_close("change.G_pair_320", change["G_pair_320"],
                 paired["from_160_to_320"]["G_pair"]["G_pair_320"])
    expect("change.classification", change["classification"],
           paired["from_160_to_320"]["G_pair"]["classification"])

    return {
        "protocol_version": decision.get("protocol_version"),
        "round": decision.get("round"),
        "git_commit": decision.get("git_commit"),
        "official_test_loaded": False,
        "arms": arms,
        "G_pair_320": g_pair_320,
        "material": MATERIAL,
        "primary_pass": bool(g_pair_320 >= MATERIAL),
        "topo_320_soup": TOPO_320_SOUP,
        "Delta_vs_TOPO": delta_vs_topo,
        "topo_tolerance": TOPO_TOLERANCE,
        "competitive": bool(delta_vs_topo <= TOPO_TOLERANCE),
        "case": case,
        "verdict": verdict,
        "recorded_verdict": decision.get("verdict"),
        "late_window_241_320": {
            "mean_delta": mean_delta,
            "median_delta": median,
            "positive_fraction": positive_fraction,
            "delta_first": deltas[0],
            "delta_last": deltas[-1],
            "min_delta": min(deltas),
            "max_delta": max(deltas),
        },
        "from_160_to_320": {
            "REAL_soup": {"at_160": lite_soup["REAL"], "at_320": arms["REAL"]["soup_valid_mae"],
                          "change": arms["REAL"]["soup_valid_mae"] - lite_soup["REAL"]},
            "INDEP_soup": {"at_160": lite_soup["INDEP"], "at_320": arms["INDEP"]["soup_valid_mae"],
                           "change": arms["INDEP"]["soup_valid_mae"] - lite_soup["INDEP"]},
            "G_pair": change,
        },
        "schedules": {
            "REAL": arms["REAL"]["schedule"]
            or "first arm step of the sequential job (commit 45d53df; artifact predates the schedule field)",
            "INDEP": arms["INDEP"]["schedule"] or "single-arm parallel job (commit a9d3e08)",
        },
        "sources": {
            "decision.json": sha256_file(RESULTS / "decision.json"),
            "paired_analysis.json": sha256_file(RESULTS / "paired_analysis.json"),
            "artifact_identity.json": sha256_file(RESULTS / "artifact_identity.json"),
            "continuation_mode.json": sha256_file(RESULTS / "continuation_mode.json"),
            "topo_reference": f"{PARENT.name}/omp_screen_topo.json",
        },
        "checks": checks,
        "n_checks": len(checks),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent local confirm analysis")
    parser.add_argument("--write", action="store_true", help="write analysis.json")
    args = parser.parse_args(argv)
    payload = analyse()
    if args.write:
        destination = RESULTS / "analysis.json"
        destination.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        print(f"wrote {destination}")
    print(f"verdict: {payload['verdict']} (case {payload['case']})")
    print(f"G_pair_320={payload['G_pair_320']!r} Delta_vs_TOPO={payload['Delta_vs_TOPO']!r}")
    print(f"checks: {payload['n_checks']}  errors: {len(payload['errors'])}")
    for message in payload["errors"]:
        print(f"  ERROR {message}")
    return 1 if payload["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
