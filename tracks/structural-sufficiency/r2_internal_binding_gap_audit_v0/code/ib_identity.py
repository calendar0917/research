"""Audit the frozen D1 split identity and every D1 run record.

This script is read-only.  It re-derives the D1 identity from the frozen split
file, compares every manifest/checkpoint/prediction to that identity, and records
whether the parent summary-field discrepancy is text-only or a real row-set error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import ib_common as C


def _manifest_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    split_field = manifest.get("split")
    if isinstance(split_field, dict):
        split = split_field.get("name") or manifest.get("split_name")
        identity = split_field.get("identity_sha256")
        fit = split_field.get("fit_rows_sha256")
        eval_ = split_field.get("eval_rows_sha256")
        scaler = split_field.get("scaler")
    else:
        split = split_field or manifest.get("split_name")
        data = manifest.get("data") or {}
        identity = data.get("identity_sha256")
        fit = data.get("fit_rows_sha256")
        eval_ = data.get("eval_rows_sha256")
        scaler = data.get("scaler")
    data_hashes = manifest.get("data_hashes") or {}
    data_split = data_hashes.get("split") or {}
    if identity is None:
        identity = data_split.get("sha256") or data_split.get("identity_sha256")
    if fit is None:
        fit = data_split.get("fit_rows_sha256")
    if eval_ is None:
        eval_ = data_split.get("eval_rows_sha256")
    return {
        "split": str(split) if split is not None else None,
        "identity_sha256": identity,
        "fit_rows_sha256": fit or manifest.get("fit_rows_sha256"),
        "eval_rows_sha256": eval_ or manifest.get("dev_rows_sha256"),
        "scaler": scaler or manifest.get("scaler"),
        "top_level_fit_rows_sha256": manifest.get("fit_rows_sha256"),
        "top_level_dev_rows_sha256": manifest.get("dev_rows_sha256"),
        "top_level_scaler": manifest.get("scaler"),
    }


def _scaler_ok(found: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(found, dict):
        return False
    try:
        return bool(
            abs(float(found["mean"]) - float(expected["mean"])) <= 1e-12
            and abs(float(found["std"]) - float(expected["std"])) <= 1e-12
            and int(found["n"]) == int(expected["n"])
        )
    except Exception:
        return False


def main() -> None:
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    inputs = C.assert_inputs()
    bundles = {split: C.load_bundle(split) for split in C.SPLITS}
    identities = {split: C.split_identity(bundles[split]) for split in C.SPLITS}

    runs: dict[str, Any] = {}
    failures: list[str] = []
    manifest_paths = sorted(C.SR_RESULTS.glob("*/manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError("no parent run manifests found")
    for manifest_path in manifest_paths:
        run_id = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_field = manifest.get("split")
        if isinstance(split_field, dict):
            split = split_field.get("name") or manifest.get("split_name")
        else:
            split = split_field or manifest.get("split_name")
        if split not in C.SPLITS:
            continue
        seed = int(manifest.get("seed", -1))
        arm = str(manifest.get("arm", "unknown"))
        fields = _manifest_fields(manifest)
        expected = C.EXPECTED_IDENTITY[split]
        checks = {
            "manifest_split_ok": fields["split"] == split,
            "manifest_identity_ok": fields["identity_sha256"] == expected,
            "manifest_fit_rows_ok": fields["fit_rows_sha256"] == C.EXPECTED_FIT[split],
            "manifest_eval_rows_ok": fields["eval_rows_sha256"] == C.EXPECTED_EVAL[split],
            "manifest_scaler_ok": _scaler_ok(fields["scaler"], C.EXPECTED_SCALER[split]),
            "top_level_fit_rows_ok": (
                fields["top_level_fit_rows_sha256"] is None
                or fields["top_level_fit_rows_sha256"] == C.EXPECTED_FIT[split]
            ),
            "top_level_dev_rows_ok": (
                fields["top_level_dev_rows_sha256"] is None
                or fields["top_level_dev_rows_sha256"] == C.EXPECTED_EVAL[split]
            ),
            "top_level_scaler_ok": (
                fields["top_level_scaler"] is None
                or _scaler_ok(fields["top_level_scaler"], C.EXPECTED_SCALER[split])
            ),
        }
        try:
            pred = C.load_predictions(run_id, bundles[split])
            checks["predictions_rows_ok"] = bool(
                np.array_equal(pred["row_id"], bundles[split].eval_rows)
            )
            checks["predictions_labels_ok"] = bool(
                np.allclose(pred["label"],
                            bundles[split].labels[bundles[split].eval_rows],
                            rtol=0.0, atol=1e-9)
            )
        except Exception as exc:  # noqa: BLE001
            checks["predictions_rows_ok"] = False
            checks["predictions_labels_ok"] = False
            checks["predictions_error"] = str(exc)
        try:
            ck = C.verify_checkpoint_identity(run_id, split)
            checks["checkpoint_epoch_ok"] = True
            checks["checkpoint_identity_ok"] = bool(
                ck["identity_sha256"] == expected
            )
        except Exception as exc:  # noqa: BLE001
            checks["checkpoint_epoch_ok"] = False
            checks["checkpoint_identity_ok"] = False
            checks["checkpoint_error"] = str(exc)
        ok = all(bool(v) for k, v in checks.items()
                 if k.endswith("_ok"))
        if not ok:
            failures.append(run_id)
        runs[run_id] = {
            "split": split,
            "seed": int(seed),
            "arm": arm,
            "smoke": bool(manifest.get("smoke", False)),
            "manifest_fields": fields,
            "checks": checks,
            "all_checks_ok": bool(ok),
        }

    # Parent summary fields that contain the known discrepancy.
    queue_lock_path = C.SR_ROUND / "analysis" / "queue_lock.json"
    queue_lock = json.loads(queue_lock_path.read_text(encoding="utf-8"))
    d1_recorded = (queue_lock.get("d1_data") or {}).get("identity_sha256")
    report_text = (C.SR_ROUND / "REPORT.md").read_text(encoding="utf-8")
    d1_report_lines = [
        line.strip() for line in report_text.splitlines()
        if "D1 identity/fit/eval" in line
    ]
    protocol_text = (C.SR_ROUND / "PROTOCOL.md").read_text(encoding="utf-8")
    sprint_text = (C.SR_ROUND / "configs" / "sprint.json").read_text(encoding="utf-8")
    checks_path = C.SR_ROUND / "analysis" / "checks.json"
    checks_json = json.loads(checks_path.read_text(encoding="utf-8"))

    summary_fields = {
        "queue_lock_d1_identity": d1_recorded,
        "queue_lock_expected": C.EXPECTED_IDENTITY["D1"],
        "queue_lock_status": (
            "correct" if d1_recorded == C.EXPECTED_IDENTITY["D1"]
            else "records_D0_identity"
        ),
        "report_d1_identity_lines": d1_report_lines,
        "report_status": (
            "correct" if any(C.EXPECTED_IDENTITY["D1"] in line
                             for line in d1_report_lines)
            else "records_D0_identity"
        ),
        "protocol_contains_expected_d1": C.EXPECTED_IDENTITY["D1"] in protocol_text,
        "sprint_contains_expected_d1": C.EXPECTED_IDENTITY["D1"] in sprint_text,
        "checks_json_d1_identity": (
            checks_json.get("checks", [{}])[0].get("identities", {})
            .get("D1", {}).get("identity_sha256")
        ),
    }
    summary_fields["checks_json_d1_status"] = (
        "correct" if summary_fields["checks_json_d1_identity"]
        == C.EXPECTED_IDENTITY["D1"] else "missing_or_wrong"
    )

    replay_path = C.ANALYSIS_DIR / "replay.json"
    replay = (json.loads(replay_path.read_text(encoding="utf-8"))
              if replay_path.is_file() else None)
    replay_d1_ok = (
        bool(replay["cells"]["D1_S0"]["R2"]["ok"]
             and replay["cells"]["D1_S0"]["EGO2"]["ok"])
        if replay is not None and "D1_S0" in replay.get("cells", {}) else None
    )

    d1_runs = [rid for rid, rec in runs.items() if rec["split"] == "D1"]
    d1_all_ok = all(runs[rid]["all_checks_ok"] for rid in d1_runs)
    d0_all_ok = all(rec["all_checks_ok"] for rec in runs.values()
                    if rec["split"] == "D0")
    text_only = bool(
        d1_all_ok and summary_fields["queue_lock_status"] == "records_D0_identity"
        and summary_fields["report_status"] == "records_D0_identity"
        and (replay_d1_ok is not False)
    )
    if not d1_all_ok:
        conclusion = "RUN_LEVEL_D1_IDENTITY_FAILURE"
    elif text_only:
        conclusion = "TEXT_ONLY_IDENTITY_ERRATA"
    else:
        conclusion = "NO_ERRATUM_REQUIRED"

    payload = {
        "schema": "r2-internal-binding-gap-audit-identity-v1",
        "created_utc": C.now_utc(),
        "expected": {
            split: {
                "identity_sha256": C.EXPECTED_IDENTITY[split],
                "fit_rows_sha256": C.EXPECTED_FIT[split],
                "eval_rows_sha256": C.EXPECTED_EVAL[split],
                "scaler": C.EXPECTED_SCALER[split],
                "recomputed": identities[split],
            }
            for split in C.SPLITS
        },
        "runs": runs,
        "d1_run_ids": d1_runs,
        "d1_all_run_checks_ok": bool(d1_all_ok),
        "d0_all_run_checks_ok": bool(d0_all_ok),
        "d1_replay_ok": replay_d1_ok,
        "replay": replay,
        "failed_runs": failures,
        "summary_fields": summary_fields,
        "conclusion": conclusion,
        "errata_required": conclusion == "TEXT_ONLY_IDENTITY_ERRATA",
        "retraining_required": False,
        "inputs": inputs,
    }
    C.write_json(C.ANALYSIS_DIR / "identity_audit.json", payload)

    print(f"identity audit: {conclusion}")
    for rid in d1_runs:
        print(f"  D1 {rid}: all_checks_ok={runs[rid]['all_checks_ok']}")
    print(f"  queue_lock D1 identity: {summary_fields['queue_lock_status']}")
    print(f"  REPORT D1 identity line: {summary_fields['report_status']}")
    print(f"  D0 all run checks ok: {d0_all_ok}")
    print(f"  failures: {failures}")
    if conclusion == "RUN_LEVEL_D1_IDENTITY_FAILURE":
        raise SystemExit("D1 run-level identity failure; stop D1 comparisons")


if __name__ == "__main__":
    main()