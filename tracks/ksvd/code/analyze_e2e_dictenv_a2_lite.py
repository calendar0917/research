#!/usr/bin/env python
"""Independent local analysis of the E2E-DictEnv-A2-Lite screen (CPU, nothing trained).

Recomputes the frozen lite quantities from the *pulled* artifacts and compares
them with the values the remote runner recorded:

* ``G_pair_screen = MAE(INDEP-OMP) - MAE(REAL-OMP)`` and, when present,
  ``G_pair_PCA = MAE(INDEP-PCA32) - MAE(REAL-PCA32)``;
* the paired late-window direction on epochs 121-160 (>= 75 % agreement, mean and
  best-epoch deltas > 0);
* the frozen label mapping (0.006 strong bar; 0.003 dense bar; unusable curves
  stop the round);
* provenance flags: horizon 160, frozen dictionaries, frozen OMP codes, the
  parent-round reference, ``official_test_loaded = false``.

The statistic here is implemented independently of the runner (a cross-check,
not a second production path): any disagreement with the recorded decision is
reported as an error.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

STRONG_POSITIVE = 0.006
PCA_MATERIAL = 0.003
LATE_START, LATE_END = 121, 160
MIN_POSITIVE_FRACTION = 0.75

DEFAULT_DIR = Path("tracks/ksvd/results/e2e_dictenv_a2_lite")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_valid_curve(path: Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        curve = {int(row["epoch"]): float(row["valid_mae"]) for row in rows}
    if not curve:
        raise SystemExit(f"empty curve: {path}")
    return curve


def late_direction(indep: dict[int, float], real: dict[int, float]) -> dict[str, Any]:
    epochs = [epoch for epoch in range(LATE_START, LATE_END + 1)]
    missing = [epoch for epoch in epochs if epoch not in indep or epoch not in real]
    if missing:
        raise SystemExit(f"late window epochs missing: {missing[:8]}")
    deltas = [indep[epoch] - real[epoch] for epoch in epochs]
    fraction = sum(delta > 0.0 for delta in deltas) / len(deltas)
    mean_delta = sum(deltas) / len(deltas)
    best_delta = min(indep[epoch] for epoch in epochs) - min(real[epoch] for epoch in epochs)
    return {
        "n_epochs": len(epochs),
        "positive_fraction": fraction,
        "mean_delta": mean_delta,
        "best_delta": best_delta,
        "stable": bool(fraction >= MIN_POSITIVE_FRACTION and mean_delta > 0.0 and best_delta > 0.0),
    }


def expected_screen_label(g_pair: float, stable: bool) -> str:
    if g_pair >= STRONG_POSITIVE and stable:
        return "PAIRING_SIGNAL_WORTH_FULL_CONFIRMATION"
    return "PCA_DIAGNOSTIC_AUTHORISED"


def expected_pca_label(g_pair: float) -> str:
    return (
        "ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK"
        if g_pair >= PCA_MATERIAL
        else "NO_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D"
    )


def analyze(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    out: dict[str, Any] = {"root": str(root), "errors": errors}

    screen_arms = {}
    for arm in ("REAL", "INDEP"):
        payload = _read_json(root / f"lite_omp_{arm.lower()}.json")
        screen_arms[arm] = {
            "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
            "best_valid_mae": float(payload["best_valid_mae"]),
            "best_epoch": int(payload["best_epoch"]),
            "soup_members": list(payload["soup"]["members"]),
            "horizon": int(payload["horizon"]),
            "stage": payload["stage"],
            "coding_mode": payload.get("coding_mode"),
            "frozen_dictionary": payload.get("frozen_dictionary"),
            "dictionary_sha256_f32": payload.get("dictionary_sha256_f32"),
            "official_test_loaded": payload.get("official_test_loaded"),
            "protocol_version": payload.get("protocol_version"),
            "git_commit": payload.get("git_commit"),
        }
        if screen_arms[arm]["horizon"] != 160:
            errors.append(f"{arm}: horizon {screen_arms[arm]['horizon']} != 160")
        if screen_arms[arm]["frozen_dictionary"] is not True:
            errors.append(f"{arm}: dictionary was not frozen")
        if screen_arms[arm]["official_test_loaded"] is not False:
            errors.append(f"{arm}: official_test_loaded not false")
    g_pair = screen_arms["INDEP"]["soup_valid_mae"] - screen_arms["REAL"]["soup_valid_mae"]

    curves = {}
    for label, arm in (("R0", "REAL"), ("I0", "INDEP")):
        curves[arm] = read_valid_curve(root / "curves" / f"lite_omp_{label}_curve.csv")
    direction = late_direction(curves["INDEP"], curves["REAL"])
    screen_label = expected_screen_label(g_pair, direction["stable"])

    recorded = _read_json(root / "lite_screen_decision.json")
    recorded_g = float(recorded["G_pair_screen"])
    if abs(recorded_g - g_pair) > 1e-12:
        errors.append(f"recorded G_pair_screen {recorded_g} != recomputed {g_pair}")
    if float(recorded["soup_valid_mae"]["REAL"]) != screen_arms["REAL"]["soup_valid_mae"]:
        errors.append("recorded REAL soup value disagrees with the arm artifact")
    if float(recorded["soup_valid_mae"]["INDEP"]) != screen_arms["INDEP"]["soup_valid_mae"]:
        errors.append("recorded INDEP soup value disagrees with the arm artifact")
    recorded_direction = recorded.get("direction") or {}
    for key in ("positive_fraction", "mean_delta", "best_delta", "stable"):
        if abs(float(recorded_direction[key]) - float(direction[key])) > 1e-12:
            errors.append(f"recorded direction[{key}] disagrees with the recomputation")
    if recorded["verdict"] != screen_label:
        errors.append(
            f"recorded screen verdict {recorded['verdict']} != frozen rule {screen_label}"
        )
    if bool(recorded["strong_positive"]) != (screen_label == "PAIRING_SIGNAL_WORTH_FULL_CONFIRMATION"):
        errors.append("recorded strong_positive disagrees with the frozen rule")
    if bool(recorded["pca_authorised"]) != (screen_label != "PAIRING_SIGNAL_WORTH_FULL_CONFIRMATION"):
        errors.append("recorded pca_authorised disagrees with the frozen rule")

    out["screen"] = {
        "arms": screen_arms,
        "G_pair_screen": g_pair,
        "direction": direction,
        "frozen_label": screen_label,
        "recorded_label": recorded["verdict"],
        "curves_finite": bool(recorded.get("curves_finite")),
        "paired_epochs": len(set(curves["INDEP"]) & set(curves["REAL"])),
    }

    pca_path = root / "lite_pca_decision.json"
    if pca_path.exists():
        pca_arms = {}
        for arm in ("REAL", "INDEP"):
            payload = _read_json(root / f"lite_pca_{arm.lower()}.json")
            pca_arms[arm] = {
                "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
                "best_valid_mae": float(payload["best_valid_mae"]),
                "horizon": int(payload["horizon"]),
                "coding_mode": payload.get("coding_mode"),
                "frozen_dictionary": payload.get("frozen_dictionary"),
                "pca_components_sha256_f32": payload.get("pca_components_sha256_f32"),
                "official_test_loaded": payload.get("official_test_loaded"),
            }
            if pca_arms[arm]["horizon"] != 160:
                errors.append(f"PCA {arm}: horizon {pca_arms[arm]['horizon']} != 160")
            if pca_arms[arm]["official_test_loaded"] is not False:
                errors.append(f"PCA {arm}: official_test_loaded not false")
        g_pair_pca = pca_arms["INDEP"]["soup_valid_mae"] - pca_arms["REAL"]["soup_valid_mae"]
        pca_label = expected_pca_label(g_pair_pca)
        recorded_pca = _read_json(pca_path)
        if abs(float(recorded_pca["G_pair_PCA"]) - g_pair_pca) > 1e-12:
            errors.append(
                f"recorded G_pair_PCA {recorded_pca['G_pair_PCA']} != recomputed {g_pair_pca}"
            )
        if recorded_pca["verdict"] != pca_label:
            errors.append(
                f"recorded PCA verdict {recorded_pca['verdict']} != frozen rule {pca_label}"
            )
        out["pca"] = {
            "arms": pca_arms,
            "G_pair_PCA": g_pair_pca,
            "frozen_label": pca_label,
            "recorded_label": recorded_pca["verdict"],
            "rank": recorded_pca.get("rank"),
        }

    report = _read_json(root / "lite_report.json")
    out["final_label"] = report["final_label"]
    expected_final = (
        out["pca"]["frozen_label"] if "pca" in out else out["screen"]["frozen_label"]
    )
    if out["final_label"] != expected_final:
        errors.append(f"final label {out['final_label']} != frozen rule {expected_final}")
    out["deferred_stages"] = report.get("deferred_stages")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--write", action="store_true", help="write analysis.json next to the artifacts")
    args = parser.parse_args(argv)

    payload = analyze(args.dir)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.write:
        (args.dir / "analysis.json").write_text(text + "\n", encoding="utf-8")
    if payload["errors"]:
        for error in payload["errors"]:
            print(f"ANALYSIS ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
