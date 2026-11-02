"""Score the frozen Test split for one frozen stage.

Discipline
----------
* Only call this after :mod:`dissect_freeze` has written the stage freeze.
* One model/seed/repeat is scored exactly once; a second call refuses unless the
  stored predictions are missing (``--repair``, which is recorded).
* Every Test number is labelled ``EXPOSED_TEST_EXPLORATORY``: the three public
  ChemBench repeats have already been used in earlier rounds, so these are
  descriptive numbers, not a blind confirmation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import dissect_common as C
import dissect_data as D
import dissect_model as M

ACCESS_LOG = C.ANALYSIS / "test_access_log.json"


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = pred - y
    ss_res = float((err * err).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(y.size),
        "rmse": float(math.sqrt(ss_res / y.size)),
        "mae": float(np.abs(err).mean()),
        "r2": float(1.0 - ss_res / ss_tot),
        "bias": float(err.mean()),
    }


def _append_access(entry: Mapping[str, Any]) -> None:
    log = C.read_json(ACCESS_LOG) if ACCESS_LOG.exists() else {"schema": "lcrc-lipo-dissection-test-access-v1", "entries": []}
    log["entries"].append(dict(entry))
    C.write_json(ACCESS_LOG, log)


def score_freeze(freeze_path: Path, *, repair: bool = False) -> dict[str, Any]:
    freeze = C.read_json(freeze_path)
    repeat = int(freeze["repeat"])
    cache, y_test, smiles, rows = D.load_test(repeat)
    if C.row_sha256(rows) != C.SPLIT_ROWS_SHA256[repeat][2]:
        raise RuntimeError("Test row ids do not match the frozen split bytes")
    spec = D.frozen_spec()
    scored: list[dict[str, Any]] = []
    skipped: list[str] = []
    for entry in freeze["entries"]:
        run_dir = C.run_dir(repeat, entry["variant"], entry["seed"])
        eval_path = run_dir / "test_eval.json"
        pred_path = run_dir / "test_predictions.npz"
        ckpt_sha = entry["checkpoint_sha256"]
        if eval_path.exists() and pred_path.exists():
            prev = C.read_json(eval_path)
            if prev.get("checkpoint_sha256") == ckpt_sha and not repair:
                skipped.append(entry["run"])
                C.log(f"[score] {entry['run']} already scored once; skipping")
                continue
            if not repair:
                raise SystemExit(
                    f"{entry['run']}: existing Test evaluation belongs to a different checkpoint "
                    f"({prev.get('checkpoint_sha256')} != {ckpt_sha}); pass --repair only if the stored "
                    "predictions are unusable, and record why"
                )
        ckpt = torch.load(C.REPO / entry["checkpoint_path"], map_location="cpu", weights_only=False)
        if C.sha256_file(C.REPO / entry["checkpoint_path"]) != ckpt_sha:
            raise SystemExit(f"{entry['run']}: checkpoint bytes changed after the freeze")
        model = M.load_checkpoint_model(
            spec, D.build_view(repeat).train_norm, entry["variant"], ckpt["state_dict"], torch.device("cpu")
        )
        pred_std = D.predict_scores(model, cache, torch.device("cpu"))
        pred = pred_std * float(ckpt["target_norm"]["scale"]) + float(ckpt["target_norm"]["mean"])
        # Persist predictions before metrics so a metric failure never forces a re-read.
        np.savez(pred_path, rows=np.asarray(rows, dtype=np.int64), raw_standardised=pred_std, raw=pred, y_true=y_test)
        metrics = _metrics(pred, y_test)
        payload = {
            "schema": "lcrc-lipo-dissection-test-eval-v1",
            "status": "EXPOSED_TEST_EXPLORATORY",
            "run": entry["run"],
            "repeat": repeat,
            "variant": entry["variant"],
            "variant_name": entry["variant_name"],
            "seed": int(entry["seed"]),
            "device": "cpu",
            "deterministic": True,
            "checkpoint_path": entry["checkpoint_path"],
            "checkpoint_sha256": ckpt_sha,
            "freeze_path": C.track_rel(freeze_path),
            "freeze_sha256": C.sha256_file(freeze_path),
            "best_epoch": int(entry["best_epoch"]),
            "target_norm": dict(ckpt["target_norm"]),
            "test_rows_sha256": C.row_sha256(rows),
            "split_file_sha256": entry["split_file_sha256"],
            "metrics": metrics,
            "predictions": {
                "path": C.track_rel(pred_path),
                "sha256": C.sha256_file(pred_path),
            },
            "test_labels_read": True,
            "test_access_count_for_checkpoint": 1,
            "note": (
                "EXPOSED_TEST_EXPLORATORY: the public ChemBench repeats were already used in earlier rounds. "
                "Descriptive only; not an independent confirmation, not used for any selection."
            ),
        }
        C.write_json(eval_path, payload)
        _append_access(
            {
                "utc": C.utcnow(),
                "run": entry["run"],
                "repeat": repeat,
                "variant": entry["variant"],
                "seed": int(entry["seed"]),
                "checkpoint_sha256": ckpt_sha,
                "freeze_tag": freeze["tag"],
                "status": "scored",
                "repair": bool(repair),
                "test_rmse": metrics["rmse"],
            }
        )
        scored.append({"run": entry["run"], "test_rmse": metrics["rmse"], "test_mae": metrics["mae"]})
        C.log(f"[score {freeze['tag']}] {entry['run']} test_rmse={metrics['rmse']:.6f}")

    summary = {
        "schema": "lcrc-lipo-dissection-score-summary-v1",
        "freeze_tag": freeze["tag"],
        "freeze_path": C.track_rel(freeze_path),
        "freeze_sha256": C.sha256_file(freeze_path),
        "repeat": repeat,
        "created_utc": C.utcnow(),
        "n_scored": len(scored),
        "n_skipped": len(skipped),
        "scored": scored,
        "skipped": skipped,
        "status": "EXPOSED_TEST_EXPLORATORY",
    }
    C.write_json(C.ANALYSIS / "freeze" / f"score_{freeze['tag']}.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score frozen Test checkpoints (one read per model/seed/repeat)")
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--repair", action="store_true", help="re-score only when stored predictions are unusable")
    args = parser.parse_args(argv)
    summary = score_freeze(Path(args.freeze), repair=bool(args.repair))
    C.log(f"scored {summary['n_scored']} runs, skipped {summary['n_skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
