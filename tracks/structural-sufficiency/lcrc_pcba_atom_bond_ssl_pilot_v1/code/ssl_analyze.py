"""Assemble the frozen analysis artifacts from the pretrain and supervised runs.

Everything here is derived from files the runs already wrote; no training is
redone and no official Test target is ever touched.  The decision label comes
from ``ssl_common.decision_label`` so it cannot drift from the protocol.

Usage::

    uv run python code/ssl_analyze.py
    uv run python code/ssl_analyze.py --skip-data-audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import ssl_common as C

HISTORICAL_PER_TASK = (
    C.NMP_SPRINT / "runs" / "phaseC_seed0_A" / "per_task_ap.csv"
)
HISTORICAL_EPOCH = 40
SURPORT_BUCKETS = (
    ("support_ge20", 20, None),
    ("support_5_19", 5, 20),
    ("support_lt5", None, 5),
)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _rows_for_epoch(rows: Iterable[Mapping[str, str]], epoch: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if int(row["epoch"]) != epoch:
            continue
        item: dict[str, Any] = dict(row)
        for key in ("task", "observed", "positive", "negative", "train_observed", "train_positive"):
            if item.get(key) not in (None, ""):
                item[key] = int(item[key])
        item["ap"] = None if item.get("ap") in (None, "") else float(item["ap"])
        out.append(item)
    return sorted(out, key=lambda r: r["task"])


def _curve(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    out = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                item[key] = None
                continue
            try:
                item[key] = float(value) if ("." in value or "e" in value.lower()) else int(value)
            except ValueError:
                item[key] = value
        out.append(item)
    return out


def _sha(path: Path) -> str | None:
    return C.sha256_file(path) if Path(path).exists() else None


# ---------------------------------------------------------------------------
# artifact builders
# ---------------------------------------------------------------------------


def build_data_audit(pretrain: Mapping[str, Any] | None) -> dict[str, Any]:
    import ssl_data as D

    st = D.load_structure()
    audit = dict(st.audit)
    audit["schema"] = "lcrc-pcba-atom-bond-ssl-data-audit-v1"
    audit["round_id"] = C.ROUND_ID
    audit["explicit_lipophilicity_source"] = "lipo_v1 (51/13/9 feature spec)"
    audit["test_rows_used_for_pretraining"] = 0
    audit["test_labels_read"] = False
    audit["test_labels_note"] = (
        "the frozen label artifact stores NaN for official Test rows and is never indexed "
        "by this track"
    )
    audit["pretrain_label_ledger"] = None if pretrain is None else pretrain.get("label_access")
    return audit


def build_probe(pretrain: Mapping[str, Any], curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probe_rows = [r for r in curve if r.get("probe_atom_ce") is not None]
    init = probe_rows[0] if probe_rows else {}
    final = probe_rows[-1] if probe_rows else {}
    gate = pretrain.get("pretext_gate") or {}
    margins = {
        "atom_prior_ce_minus_ce": None
        if final.get("probe_atom_ce") is None
        else float(final["probe_atom_prior_ce"] - final["probe_atom_ce"]),
        "bond_prior_ce_minus_ce": None
        if final.get("probe_bond_ce") is None
        else float(final["probe_bond_prior_ce"] - final["probe_bond_ce"]),
    }
    return {
        "schema": "lcrc-pcba-atom-bond-ssl-pretrain-probe-v1",
        "round_id": C.ROUND_ID,
        "partition": pretrain.get("probe_partition"),
        "n_probe_molecules": C.N_PROBE,
        "mask_visit_index": int(C.PROBE_VISIT_INDEX),
        "n_eval_points": len(probe_rows),
        "init": {
            "update": init.get("update"),
            "atom_ce": init.get("probe_atom_ce"),
            "bond_ce": init.get("probe_bond_ce"),
            "atom_accuracy": init.get("probe_atom_accuracy"),
            "bond_accuracy": init.get("probe_bond_accuracy"),
            "atom_prior_ce": init.get("probe_atom_prior_ce"),
            "bond_prior_ce": init.get("probe_bond_prior_ce"),
            "masked_atoms": init.get("probe_masked_atoms"),
            "masked_bonds": init.get("probe_masked_bonds"),
        },
        "final": {
            "update": final.get("update"),
            "atom_ce": final.get("probe_atom_ce"),
            "bond_ce": final.get("probe_bond_ce"),
            "atom_accuracy": final.get("probe_atom_accuracy"),
            "bond_accuracy": final.get("probe_bond_accuracy"),
            "atom_prior_ce": final.get("probe_atom_prior_ce"),
            "bond_prior_ce": final.get("probe_bond_prior_ce"),
            "atom_prior_ce_masked_targets": final.get("probe_atom_prior_ce_masked_targets"),
            "bond_prior_ce_masked_targets": final.get("probe_bond_prior_ce_masked_targets"),
            "masked_atoms": final.get("probe_masked_atoms"),
            "masked_bonds": final.get("probe_masked_bonds"),
            "atom_class_counts": final.get("probe_atom_class_counts"),
            "bond_class_counts": final.get("probe_bond_class_counts"),
        },
        "margins": margins,
        "gate": gate,
        "curve": [
            {k: r.get(k) for k in (
                "update", "probe_atom_ce", "probe_bond_ce",
                "probe_atom_accuracy", "probe_bond_accuracy",
                "probe_atom_prior_ce", "probe_bond_prior_ce",
            )}
            for r in probe_rows
        ],
    }


def build_pretrain_run(pretrain: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    keys = (
        "schema",
        "round_id",
        "stage",
        "status",
        "stop_reason",
        "updates_requested",
        "updates_total",
        "updates_completed",
        "resumed",
        "pretrain_config",
        "optimizer_audit",
        "pretext_audit",
        "class_priors",
        "norm_fingerprint",
        "curve_path",
        "curve_sha256",
        "final_checkpoint",
        "final_checkpoint_sha256",
        "encoder_initial_fingerprint",
        "encoder_final_fingerprint",
        "encoder_change",
        "pretext_gate",
        "shuffle",
        "data",
        "rng_state_hashes_start",
        "rng_state_hashes_end",
        "environment",
        "git_commit",
        "started_utc",
        "finished_utc",
        "gpu_seconds",
        "gpu_hours",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
        "non_finite_events",
        "test_evaluated",
        "test_policy",
    )
    out = {k: pretrain.get(k) for k in keys}
    out["checkpoint"] = C.track_rel(checkpoint) if checkpoint.exists() else None
    out["checkpoint_sha256"] = _sha(checkpoint)
    out["updates_completed_is_final"] = bool(
        int(pretrain.get("updates_completed", -1)) == int(pretrain.get("updates_requested", -2))
        == int(C.PRETRAIN["final_update_for_export"])
    )
    return out


def _encoder_sha_table(state: Mapping[str, Any]) -> dict[str, str]:
    table = {}
    for key, tensor in state.items():
        if not key.startswith(C.TRANSFERRED_PREFIXES):
            continue
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        table[key] = hashlib.sha256(array.tobytes()).hexdigest()
    return dict(sorted(table.items()))


def build_transfer_audit(supervised: Mapping[str, Any], ssl_payload: Mapping[str, Any]) -> dict[str, Any]:
    audit = dict(supervised["transfer_audit"])
    audit["schema"] = "lcrc-pcba-atom-bond-ssl-encoder-transfer-v1"
    audit["round_id"] = C.ROUND_ID
    audit["ssl_checkpoint"] = supervised.get("ssl_checkpoint")
    audit["ssl_checkpoint_sha256"] = supervised.get("ssl_checkpoint_sha256")
    audit["ssl_updates"] = supervised.get("ssl_updates")
    audit["ssl_encoder_fingerprint"] = supervised.get("ssl_checkpoint_encoder_fingerprint")
    audit["historical_parity"] = dict(C.HISTORICAL_PARITY)
    audit["historical_a0"] = dict(C.HISTORICAL_A0)
    audit["transferred_prefixes"] = list(C.TRANSFERRED_PREFIXES)
    audit["frozen_prefixes"] = list(C.NON_TRANSFERRED_PREFIXES)
    audit["non_transferred_tensors_are_historical_seed0"] = audit.get(
        "non_transferred_tensors_all_equal_to_seed0_scratch"
    )
    audit["ssl_encoder_tensor_sha256"] = _encoder_sha_table(ssl_payload["encoder_state"])
    audit["reachable_without_pretext_heads"] = True
    audit["inference_parameters"] = int(supervised["parameters"])
    return audit


def _bucket_stats(compared: Sequence[Mapping[str, Any]], lo: int | None, hi: int | None) -> dict[str, Any]:
    rows = [
        r
        for r in compared
        if r["ap"] is not None
        and (lo is None or r["positive"] >= lo)
        and (hi is None or r["positive"] < hi)
    ]
    deltas = [r["delta"] for r in rows]
    return {
        "n_tasks": len(rows),
        "mean_ap": float(statistics.fmean([r["ap"] for r in rows])) if rows else None,
        "mean_historical_ap": float(statistics.fmean([r["historical_ap"] for r in rows]))
        if rows
        else None,
        "mean_delta": float(statistics.fmean(deltas)) if deltas else None,
        "median_delta": float(statistics.median(deltas)) if deltas else None,
        "n_improved": sum(1 for d in deltas if d > 0),
        "n_regressed": sum(1 for d in deltas if d < 0),
        "n_equal": sum(1 for d in deltas if d == 0),
    }


def build_per_task(
    mine: Sequence[Mapping[str, Any]], historical: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hist = {r["task"]: r for r in historical}
    compared: list[dict[str, Any]] = []
    for row in mine:
        h = hist.get(row["task"])
        if h is None:
            raise RuntimeError(f"task {row['task']} missing from the historical per-task table")
        delta = None if row["ap"] is None else float(row["ap"] - h["ap"])
        compared.append(
            {
                "task": int(row["task"]),
                "ap": row["ap"],
                "historical_ap": h["ap"],
                "delta": delta,
                "observed": int(row["observed"]),
                "positive": int(row["positive"]),
                "negative": int(row["negative"]),
                "train_observed": int(row["train_observed"]),
                "train_positive": int(row["train_positive"]),
                "stratum": row.get("stratum") or h.get("stratum"),
            }
        )
    return compared, mine


def build_supervised(
    supervised: Mapping[str, Any],
    curve: Sequence[Mapping[str, Any]],
    compared: Sequence[Mapping[str, Any]],
    historical: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ap40 = float(supervised["validation_ap40"])
    delta = float(ap40 - C.AP40_A0)
    label = C.decision_label(delta)
    train_eval = supervised.get("train_eval_final") or {}
    buckets = {
        name: _bucket_stats(compared, lo, hi) for name, lo, hi in SURPORT_BUCKETS
    }
    scored = [r for r in compared if r["ap"] is not None]
    summary = {
        "n_tasks": len(compared),
        "n_tasks_scored": len(scored),
        "n_improved": sum(1 for r in scored if r["delta"] > 0),
        "n_regressed": sum(1 for r in scored if r["delta"] < 0),
        "n_equal": sum(1 for r in scored if r["delta"] == 0),
        "mean_delta": float(statistics.fmean([r["delta"] for r in scored])),
        "median_delta": float(statistics.median([r["delta"] for r in scored])),
        "min_delta": float(min(r["delta"] for r in scored)),
        "max_delta": float(max(r["delta"] for r in scored)),
    }
    return {
        "schema": "lcrc-pcba-atom-bond-ssl-supervised-epoch40-v1",
        "round_id": C.ROUND_ID,
        "model_name": C.MODEL_NAME,
        "seed": int(C.SSL_SEED),
        "status": supervised.get("status"),
        "stop_reason": supervised.get("stop_reason"),
        "epochs_run": supervised.get("epochs_run"),
        "global_steps_run": supervised.get("global_steps_run"),
        "parameter_count": int(supervised["parameters"]),
        "base_parameters": int(supervised["base_parameters"]),
        "new_parameters": int(supervised["new_parameters"]),
        "historical_a0_ap40": C.AP40_A0,
        "validation_ap": {
            str(epoch): {
                "ap": report["ap"],
                "n_tasks_scored": report.get("n_tasks_scored"),
                "valid_bce": report.get("valid_bce"),
            }
            for epoch, report in sorted(
                ((int(k), v) for k, v in supervised["fixed_epoch_reports"].items())
            )
        },
        "validation_ap40": ap40,
        "delta_ap40_vs_historical_a0": delta,
        "decision_label": label,
        "train_eval_final": {
            "ap40": train_eval.get("ap"),
            "bce": train_eval.get("bce"),
            "epoch": train_eval.get("epoch"),
            "evaluated_rows": train_eval.get("evaluated_rows"),
            "n_tasks_scored": train_eval.get("n_tasks_scored"),
        },
        "train_valid_ap_gap": supervised.get("train_valid_ap_gap"),
        "valid_bce": (
            supervised["fixed_epoch_reports"][str(C.TOTAL_EPOCHS)].get("valid_bce")
            if str(C.TOTAL_EPOCHS) in supervised.get("fixed_epoch_reports", {})
            else None
        ),
        "per_task_summary": summary,
        "support_buckets": buckets,
        "flops": supervised.get("flops"),
        "gpu_seconds": supervised.get("gpu_seconds"),
        "gpu_hours": supervised.get("gpu_hours"),
        "gpu_peak_allocated_bytes": supervised.get("gpu_peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": supervised.get("gpu_peak_reserved_bytes"),
        "grad_clip_events": supervised.get("grad_clip_events"),
        "non_finite_events": supervised.get("non_finite_events"),
        "checkpoint_epoch40": supervised.get("final_checkpoint"),
        "checkpoint_epoch40_sha256": supervised.get("final_checkpoint_sha256"),
        "per_task_ap_path": supervised.get("per_task_ap_path"),
        "per_task_ap_sha256": supervised.get("per_task_ap_sha256"),
        "curve": supervised.get("curve"),
        "curve_sha256": supervised.get("curve_sha256"),
        "batch_plan": supervised.get("batch_plan"),
        "training_order_summary": supervised.get("training_order_summary"),
        "git_commit": supervised.get("git_commit"),
        "environment": supervised.get("environment"),
        "recipe": supervised.get("recipe"),
        "started_utc": supervised.get("started_utc"),
        "finished_utc": supervised.get("finished_utc"),
        "test_evaluated": False,
    }


def build_decision(supervised: Mapping[str, Any], pretrain: Mapping[str, Any]) -> dict[str, Any]:
    delta = float(supervised["validation_ap40"] - C.AP40_A0)
    return {
        "schema": "lcrc-pcba-atom-bond-ssl-decision-v1",
        "round_id": C.ROUND_ID,
        "model_name": C.MODEL_NAME,
        "seed": int(C.SSL_SEED),
        "validation_ap40": float(supervised["validation_ap40"]),
        "historical_a0_validation_ap40": C.AP40_A0,
        "delta_ap40": delta,
        "decision_label": C.decision_label(delta),
        "bands": [
            *[{"min_delta": float(t), "label": label} for t, label in C.DECISION_BANDS],
            {"max_delta": -0.002, "label": "NEGATIVE_TRANSFER"},
        ],
        "pretext_gate": {
            "label": (pretrain.get("pretext_gate") or {}).get("label"),
            "min_prior_ce_margin_nats": C.PRETEXT_GATE["min_prior_ce_margin_nats"],
            "rule": C.PRETEXT_GATE["rule"],
            "passed": bool((pretrain.get("pretext_gate") or {}).get("passed")),
        },
        "comparison_is_paired": False,
        "caveats": [
            "no same-round A0 run: implementation/environment drift cannot be excluded",
            "SSL pretraining adds compute, so any gain cannot be attributed to self-supervision alone",
            "a well-learned pretext with no validation gain is a valid negative transfer result",
            "the historical Test-facing A0 number is not a target of this round",
            "the exploratory bands are resource-allocation labels, not significance tests",
        ],
        "test_evaluated": False,
    }


def build_registry(
    pretrain: Mapping[str, Any],
    supervised: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    runs = []
    for entry in (pretrain, supervised):
        runs.append(
            {
                "stage": entry.get("stage"),
                "run_id": entry.get("run_id") or C.ROUND_ID,
                "status": entry.get("status"),
                "stop_reason": entry.get("stop_reason"),
                "started_utc": entry.get("started_utc"),
                "finished_utc": entry.get("finished_utc"),
                "gpu_hours": entry.get("gpu_hours"),
                "gpu_seconds": entry.get("gpu_seconds"),
                "git_commit": entry.get("git_commit"),
                "hostname": (entry.get("environment") or {}).get("hostname"),
                "device": (entry.get("environment") or {}).get("cuda_device_name"),
                "parameters": entry.get("parameters"),
                "checkpoint": entry.get("final_checkpoint") or entry.get("checkpoint"),
                "checkpoint_sha256": entry.get("final_checkpoint_sha256")
                or entry.get("checkpoint_sha256"),
                "job_id": entry.get("job_id"),
            }
        )
    return {
        "schema": "lcrc-pcba-atom-bond-ssl-run-registry-v1",
        "round_id": C.ROUND_ID,
        "runs": runs,
        "preflight_audit": None
        if audit is None
        else {
            "all_passed": audit.get("all_passed"),
            "device": audit.get("device"),
            "checks": [c["name"] for c in audit.get("checks", [])],
        },
        "rfcs": {
            "pretrain": 1,
            "supervised": 1,
            "seeds": [int(C.SSL_SEED)],
            "test_evaluated": False,
        },
    }


def build_ledger(pretrain: Mapping[str, Any], supervised: Mapping[str, Any], extras: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries = []
    for entry in (pretrain, supervised):
        entries.append(
            {
                "stage": entry.get("stage"),
                "status": entry.get("status"),
                "gpu_seconds": entry.get("gpu_seconds"),
                "gpu_hours": entry.get("gpu_hours"),
                "hostname": (entry.get("environment") or {}).get("hostname"),
                "device": (entry.get("environment") or {}).get("cuda_device_name"),
            }
        )
    total = sum(float(e.get("gpu_hours") or 0.0) for e in entries)
    extras_total = sum(float(e.get("gpu_hours") or 0.0) for e in extras)
    return {
        "schema": "lcrc-pcba-atom-bond-ssl-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "budget_target_gpu_hours": float(C.BUDGET["gpu_target_hours"]),
        "budget_hard_gpu_hours": float(C.BUDGET["gpu_hard_hours"]),
        "runs": entries,
        "extras": list(extras),
        "total_gpu_hours": total + extras_total,
        "training_gpu_hours": total,
        "within_target": (total + extras_total) <= float(C.BUDGET["gpu_target_hours"]),
        "within_hard_cap": (total + extras_total) <= float(C.BUDGET["gpu_hard_hours"]),
        "note": "includes preflight, failed attempts and the two formal runs",
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default=str(C.ANALYSIS))
    parser.add_argument("--runs-dir", default=str(C.RUNS))
    parser.add_argument("--skip-data-audit", action="store_true")
    parser.add_argument("--extras", default=None, help="JSON list of extra GPU ledger entries")
    args = parser.parse_args(argv)

    analysis = Path(args.analysis_dir)
    runs = Path(args.runs_dir)
    pretrain_dir = runs / "pretrain"
    supervised_dir = runs / "supervised"
    pretrain = _read_json(pretrain_dir / "run.json")
    supervised = _read_json(supervised_dir / "run.json")
    if pretrain.get("stage") != "pretrain":
        raise RuntimeError("runs/pretrain/run.json is not a pretrain run")
    if supervised.get("stage") != "supervised":
        raise RuntimeError("runs/supervised/run.json is not a supervised run")

    pretrain_curve = _curve(pretrain_dir / "curve.csv")
    supervised_curve = _curve(supervised_dir / "train_curve.csv")
    ssl_checkpoint = pretrain_dir / "checkpoint_final.pt"
    import torch

    ssl_payload = torch.load(ssl_checkpoint, map_location="cpu", weights_only=False)
    if int(ssl_payload.get("update", -1)) != int(C.PRETRAIN["final_update_for_export"]):
        raise RuntimeError("runs/pretrain/checkpoint_final.pt is not the update-10000 export")

    if not args.skip_data_audit:
        data_audit = build_data_audit(pretrain)
        C.write_json(analysis / "data_audit.json", data_audit)

    probe = build_probe(pretrain, pretrain_curve)
    C.write_json(analysis / "pretrain_probe.json", probe)

    pretrain_run = build_pretrain_run(pretrain, ssl_checkpoint)
    pretrain_run["curve_rows"] = len(pretrain_curve)
    pretrain_run["final_train_row"] = pretrain_curve[-1] if pretrain_curve else None
    C.write_json(analysis / "pretrain_run.json", pretrain_run)

    transfer = build_transfer_audit(supervised, ssl_payload)
    C.write_json(analysis / "encoder_transfer_audit.json", transfer)

    mine = _rows_for_epoch(_read_csv(supervised_dir / "per_task_ap.csv"), int(supervised["final_epoch"]))
    historical = _rows_for_epoch(_read_csv(HISTORICAL_PER_TASK), HISTORICAL_EPOCH)
    compared, _ = build_per_task(mine, historical)

    per_task_path = analysis / "per_task_ap.csv"
    per_task_path.parent.mkdir(parents=True, exist_ok=True)
    with open(per_task_path, "w", newline="", encoding="utf-8") as fh:
        fields = [
            "task", "ap", "historical_ap", "delta", "observed", "positive",
            "negative", "train_observed", "train_positive", "stratum",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in compared:
            writer.writerow(
                {
                    k: ("" if row.get(k) is None else row[k])
                    for k in fields
                }
            )

    supervised_summary = build_supervised(supervised, supervised_curve, compared, historical)
    supervised_summary["per_task_table"] = C.track_rel(per_task_path)
    supervised_summary["per_task_table_sha256"] = C.sha256_file(per_task_path)
    C.write_json(analysis / "supervised_epoch40.json", supervised_summary)

    decision = build_decision(supervised, pretrain)
    C.write_json(analysis / "decision.json", decision)

    audit = None
    audit_path = analysis / "mask_leakage_audit.json"
    if audit_path.exists():
        audit = _read_json(audit_path)
    if audit is not None and not audit.get("all_passed"):
        raise RuntimeError("mask/leakage audit did not pass; refusing to register the round")

    C.write_json(analysis / "run_registry.json", build_registry(pretrain, supervised, audit))

    extras = [] if args.extras is None else _read_json(Path(args.extras))
    C.write_json(analysis / "gpu_ledger.json", build_ledger(pretrain, supervised, extras))

    C.log(
        f"[ssl-analyze] AP40={decision['validation_ap40']:.10f} "
        f"delta={decision['delta_ap40']:+.10f} label={decision['decision_label']} "
        f"pertask={supervised_summary['per_task_summary']['n_improved']}"
        f"/{supervised_summary['per_task_summary']['n_regressed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
