"""P1 -- one-shot Test scoring of the two new Lipophilicity repeats.

Gates (all enforced in code, none optional):

1. ``analysis/checkpoint_manifest.json`` exists and contains the run;
2. the checkpoint sha256 still equals the frozen value;
3. the run has not been scored before (the ledger is append-only);
4. the Test row ids equal the frozen published repeat.

Only the two Lipophilicity repeats are scored in this round.  Their Test labels
belong to a dataset whose Test was already read in Sprint 01, so the result is
labelled ``PUBLIC_SPLIT_ROBUSTNESS`` -- a repeat-robustness statement, not an
external blind test.

Usage
-----
    uv run python code/ta_eval.py --run lipo_r123
    uv run python code/ta_eval.py --run lipo_r123 --verify
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ta_common as C  # noqa: I001

import mb2_data as D
import mb2_metrics as MET
import mb2_model as M
import ta_data as TD

SCORABLE = {"lipo_r123": 123, "lipo_r124": 124}


def _ledger() -> dict[str, Any]:
    if C.TEST_LEDGER.exists():
        return C.read_json(C.TEST_LEDGER)
    return {"schema": "lcrc-taskaware-test-access-v1", "round_id": C.ROUND_ID, "reads": []}


def _frozen_entry(key: str) -> dict[str, Any]:
    if not C.CHECKPOINT_MANIFEST.exists():
        raise SystemExit("checkpoint manifest is missing; run ta_freeze.py first")
    manifest = C.read_json(C.CHECKPOINT_MANIFEST)
    if key not in manifest["runs"]:
        raise SystemExit(f"run {key!r} is not in the frozen checkpoint manifest")
    entry = manifest["runs"][key]
    path = C.REPO / entry["artifacts"]["checkpoint"]
    if C.sha256_file(path) != entry["artifacts"]["checkpoint_sha256"]:
        raise SystemExit(f"checkpoint {C.track_rel(path)} changed after the freeze")
    return entry


def evaluate(key: str) -> dict[str, Any]:
    suffix = SCORABLE[key]
    entry = _frozen_entry(key)
    ledger = _ledger()
    if any(read.get("run") == key for read in ledger["reads"]):
        raise SystemExit(f"{key} was already scored; a second read is forbidden")
    repeat = TD.lipo_repeat_split(suffix)
    if entry["split_rows_sha256"].get("train") != repeat["rows_sha256"]["train"]:
        raise SystemExit("frozen split rows differ from the published repeat")

    spec = D.frozen_spec()
    checkpoint_path = C.REPO / entry["artifacts"]["checkpoint"]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    view = TD.build_view_rows(
        "Lipophilicity",
        {"train": repeat["rows"]["train"], "valid": repeat["rows"]["valid"]},
        tag=f"lipo_r{suffix}",
        split_meta={"kind": "published_repeat", "suffix": suffix},
    )
    model = M.load_checkpoint_model(
        spec,
        view.train_norm,
        ckpt["state_dict"],
        torch.device("cpu"),
        dropout=float(C.recipe()["dropout"]),
        n_tasks=int(ckpt.get("n_outputs", 1)),
    )
    test_cache, y_test, test_rows = TD.build_test_view(
        "Lipophilicity", repeat["rows"]["test"], tag=f"lipo_r{suffix}", spec=spec
    )
    if test_rows != [int(x) for x in repeat["rows"]["test"]]:
        raise SystemExit("Test row ids differ from the frozen published repeat")
    if C.row_sha(test_rows) != repeat["rows_sha256"]["test"]:
        raise SystemExit("Test row hash differs from the frozen published repeat")
    raw = D.predict_matrix(model, test_cache, torch.device("cpu")).reshape(-1)
    target_norm = ckpt["target_norm"]
    pred = raw * float(target_norm["scale"]) + float(target_norm["mean"])
    metrics = MET.regression_metrics(pred, y_test)

    run_dir = C.run_dir(key)
    np.savez(run_dir / "test_predictions.npz", rows=np.asarray(test_rows, dtype=np.int64), raw=raw, pred=pred, y_true=y_test)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "run": key,
        "dataset": "Lipophilicity",
        "repeat_suffix": int(suffix),
        "split_file": repeat["file"],
        "split_file_sha256": repeat["sha256"],
        "split_rows_sha256": repeat["rows_sha256"],
        "test": metrics,
        "valid": entry["valid_metrics"],
        "best_epoch": entry["best_epoch"],
        "checkpoint_sha256": entry["artifacts"]["checkpoint_sha256"],
        "checkpoint_path": entry["artifacts"]["checkpoint"],
        "target_norm": target_norm,
        "test_predictions_sha256": C.sha256_file(run_dir / "test_predictions.npz"),
        "test_labels_read": True,
        "test_used_for_any_selection": False,
        "test_read_utc": C.utcnow(),
        "exposure": "PUBLIC_SPLIT_ROBUSTNESS (Lipophilicity Test labels were already read in Sprint 01)",
        "n_test": int(y_test.size),
    }
    C.write_json(run_dir / "test_eval.json", payload)
    ledger["reads"].append(
        {
            "run": key,
            "dataset": "Lipophilicity",
            "repeat_suffix": int(suffix),
            "test_rows_sha256": repeat["rows_sha256"]["test"],
            "n_test": int(y_test.size),
            "checkpoint_manifest_sha256": C.sha256_file(C.CHECKPOINT_MANIFEST),
            "checkpoint_sha256": entry["artifacts"]["checkpoint_sha256"],
            "test_eval_sha256": C.sha256_file(run_dir / "test_eval.json"),
            "metric": float(metrics["rmse"]),
            "read_utc": C.utcnow(),
            "used_for_selection": False,
        }
    )
    ledger["n_reads"] = len(ledger["reads"])
    ledger["policy"] = "one read per (run, split); append-only; never used for any selection"
    C.write_json(C.TEST_LEDGER, ledger)
    C.log(f"[{key}] Test RMSE {metrics['rmse']:.5f} (n={y_test.size}) recorded once")
    return payload


def verify(key: str) -> dict[str, Any]:
    run_dir = C.run_dir(key)
    payload = C.read_json(run_dir / "test_eval.json")
    with np.load(run_dir / "test_predictions.npz") as handle:
        pred = np.asarray(handle["pred"], dtype=np.float64)
        y = np.asarray(handle["y_true"], dtype=np.float64)
    recomputed = MET.regression_metrics(pred, y)
    diff = abs(float(recomputed["rmse"]) - float(payload["test"]["rmse"]))
    out = {
        "run": key,
        "stored_rmse": float(payload["test"]["rmse"]),
        "recomputed_rmse": float(recomputed["rmse"]),
        "abs_diff": float(diff),
        "consistent": bool(diff <= 1e-12),
    }
    print(out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 one-shot Test scoring (Lipophilicity repeats)")
    parser.add_argument("--run", required=True, choices=sorted(SCORABLE))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        verify(args.run)
    else:
        print(evaluate(args.run)["test"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
