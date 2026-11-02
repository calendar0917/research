"""Freeze the checkpoints (and the TU folds) before any Test label is read.

``--molecule`` writes ``analysis/checkpoint_manifest.json`` for every molecule
dataset whose six units are all complete, and refuses to freeze a dataset with a
missing or failed unit.  ``--tu`` writes ``analysis/split_manifest.json`` with the
fold partitions and their hashes, and the per-run checkpoint hashes once the TU
units finish.

A dataset that is frozen here may not be re-trained in this round.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import ptb_common as C
import ptb_data as D


def molecule_manifest(dataset: str) -> dict[str, Any]:
    units = []
    missing = []
    for model in C.MODELS:
        for seed in C.MOLECULE_SEEDS:
            run_dir = C.mol_run_dir(dataset, model, seed)
            run_json = run_dir / "run.json"
            sel_json = run_dir / "selection.json"
            ckpt = run_dir / "checkpoint.pt"
            if not (run_json.exists() and sel_json.exists() and ckpt.exists()):
                missing.append(C.mol_run_name(dataset, model, seed))
                continue
            rec = C.read_json(run_json)
            sel = C.read_json(sel_json)
            sha = C.sha256_file(ckpt)
            if sha != sel["checkpoint_sha256"] or sha != rec["checkpoint"]["sha256"]:
                raise RuntimeError(f"{dataset}/{model}/s{seed}: checkpoint hash does not match the frozen record")
            if bool(sel.get("test_evaluated")):
                raise RuntimeError(f"{dataset}/{model}/s{seed}: Test was already evaluated before freezing")
            units.append(
                {
                    "unit": C.mol_run_name(dataset, model, seed),
                    "dataset": dataset,
                    "variant": model,
                    "seed": int(seed),
                    "epochs_run": int(rec["epochs_run"]),
                    "best_epoch": int(rec["checkpoint"]["best_epoch"]),
                    "best_metric_name": rec["validation"]["metrics"] and sel["best_metric_name"],
                    "best_valid_value": float(sel["best_metric_value"]),
                    "valid_metrics": rec["validation"]["metrics"],
                    "train_metrics": rec["train"]["metrics"],
                    "parameters": int(rec["parameters"]),
                    "n_train": int(rec["n_train"]),
                    "n_valid": int(rec["n_valid"]),
                    "checkpoint_path": C.track_rel(ckpt),
                    "checkpoint_sha256": sha,
                    "best_state_hash": rec["checkpoint"]["best_state_hash"],
                    "curve_sha256": sel["curve_sha256"],
                    "gpu_hours": float(rec["timing"]["gpu_hours_estimate"]),
                    "budget_limited_possible": bool(rec.get("budget_limited_possible", False)),
                    "result_label": rec["result_label"],
                }
            )
    record = {
        "dataset": dataset,
        "n_units_expected": len(C.MODELS) * len(C.MOLECULE_SEEDS),
        "n_units_present": len(units),
        "missing_units": missing,
        "frozen": not missing,
        "frozen_utc": C.utcnow(),
        "frozen_before_test_access": True,
        "units": units,
    }
    record["manifest_sha256"] = C.sha256_text(C.stable_json(record))
    return record


def tu_manifest(protocol) -> dict[str, Any]:
    scheme = protocol["tu_experiment"]["scheme"]
    folds = []
    for dataset in C.TU_DATASETS:
        for fold in range(int(scheme["n_splits"])):
            sig = D.fold_signature(dataset, int(scheme["n_splits"]), fold, int(scheme["fold_seed"]))
            folds.append(sig)
    units = []
    missing = 0
    for dataset in C.TU_DATASETS:
        for model in C.MODELS:
            for fold in range(int(scheme["n_splits"])):
                for seed in scheme["seeds"]:
                    run_dir = C.tu_run_dir(dataset, model, fold, int(seed))
                    run_json = run_dir / "run.json"
                    if not run_json.exists():
                        missing += 1
                        continue
                    rec = C.read_json(run_json)
                    ckpt = run_dir / "checkpoint.pt"
                    units.append(
                        {
                            "unit": C.tu_run_name(dataset, model, fold, int(seed)),
                            "dataset": dataset,
                            "variant": model,
                            "fold": int(fold),
                            "seed": int(seed),
                            "best_epoch": int(rec["checkpoint"]["best_epoch"]),
                            "valid_accuracy": float(rec["checkpoint"]["best_valid_accuracy"]),
                            "test_accuracy": None if rec["test"] is None else float(rec["test"]["accuracy"]),
                            "parameters": int(rec["parameters"]),
                            "checkpoint_sha256": C.sha256_file(ckpt) if ckpt.exists() else None,
                            "partition": rec["partition"],
                            "feature_spec_hash": rec["feature_spec_hash"],
                            "gpu_hours": float(rec["timing"]["gpu_hours_estimate"]),
                        }
                    )
    payload = {
        "schema": "lcrc-ptb-split-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "scheme": scheme,
        "scheme_name": protocol["tu_experiment"]["scheme_name"],
        "n_folds": len(folds),
        "folds": folds,
        "folds_sha256": C.sha256_text(C.stable_json(folds)),
        "n_units_expected": len(C.TU_DATASETS) * len(C.MODELS) * int(scheme["n_splits"]) * len(scheme["seeds"]),
        "n_units_present": len(units),
        "n_units_missing": missing,
        "units": units,
    }
    payload["manifest_sha256"] = C.sha256_text(C.stable_json({k: payload[k] for k in ("folds_sha256", "units")}))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--molecule", action="store_true")
    ap.add_argument("--tu", action="store_true")
    ap.add_argument("--dataset", default=None, help="freeze one molecule dataset only")
    ap.add_argument("--protocol", default=str(C.CONFIGS / "protocol.json"))
    args = ap.parse_args(argv)
    if not (args.molecule or args.tu):
        ap.error("choose --molecule and/or --tu")
    if args.molecule:
        path = C.ANALYSIS / "checkpoint_manifest.json"
        payload = C.read_json(path) if path.exists() else {"schema": "lcrc-ptb-checkpoint-manifest-v1", "datasets": {}}
        payload["schema"] = "lcrc-ptb-checkpoint-manifest-v1"
        payload["round_id"] = C.ROUND_ID
        datasets = [args.dataset] if args.dataset else list(C.MOLECULE_ORDER)
        for dataset in datasets:
            try:
                rec = molecule_manifest(dataset)
            except RuntimeError as exc:
                existing = payload["datasets"].get(dataset)
                # a dataset that was already frozen *and* scored keeps its frozen entry:
                # re-freezing is impossible by design (the Test read is one-shot), so the
                # only honest action is to carry the recorded entry forward unchanged.
                if existing and existing.get("frozen") and "already evaluated" in str(exc):
                    rec = existing
                    C.log(f"{dataset}: already frozen and scored -> entry preserved ({rec['n_units_present']}/6)")
                    continue
                raise
            C.log(
                f"{dataset}: {rec['n_units_present']}/{rec['n_units_expected']} units "
                f"{'FROZEN' if rec['frozen'] else 'INCOMPLETE -> ' + str(rec['missing_units'])}"
            )
        payload["updated_utc"] = C.utcnow()
        payload["frozen_datasets"] = [d for d, r in payload["datasets"].items() if r["frozen"]]
        C.write_json(path, payload)
        back = C.read_json(path)
        for dataset in datasets:
            if back["datasets"][dataset]["n_units_present"] != payload["datasets"][dataset]["n_units_present"]:
                raise RuntimeError(f"{dataset}: manifest write did not persist")
        C.log(f"manifest written: {sorted(back['frozen_datasets'])}")
    if args.tu:
        protocol = C.read_json(args.protocol)
        payload = tu_manifest(protocol)
        C.write_json(C.ANALYSIS / "split_manifest.json", payload)
        C.log(
            f"TU folds frozen ({payload['n_folds']} folds, {payload['n_units_present']}/"
            f"{payload['n_units_expected']} units present)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
