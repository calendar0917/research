"""Terminal, reporting-only official-test scoring of the frozen DBT checkpoint.

Order of operations (the freeze is written *before* any test label is read):

1. load ``best.pt`` (the strict-``<`` best-valid checkpoint of the one declared
   seed-0 run) and reproduce its valid MAE on the full 1,000 valid molecules;
2. write ``analysis/test_freeze.json`` with the checkpoint sha256, epoch,
   valid MAE, recipe and parameter count — this file is written first and is
   never rewritten;
3. only then build/load the label-free test *relation* cache, read the
   official test labels once, and compute the descriptive test metrics.

No test number is used to pick a checkpoint, an epoch, a recipe or a direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import dbt_data as BD  # noqa: E402
import dbt_train as BT  # noqa: E402

ANALYSIS = _EXP / "analysis"
RUN = _EXP / "runs" / "dbt_s0"


def score(model, cache, rel, rows, device, batch_size=256) -> np.ndarray:
    pred, _ = BT.evaluate(model, cache, rel, np.asarray(rows), device=device,
                          batch_size=batch_size)
    return pred


def describe(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ae = np.abs(pred.astype(np.float64) - y.astype(np.float64))
    signed = pred.astype(np.float64) - y.astype(np.float64)
    return {
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((signed ** 2).mean())),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "p95_ae": float(np.quantile(ae, 0.95)),
        "max_ae": float(ae.max()),
        "mean_signed": float(signed.mean()),
        "n": int(len(y)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--best", default=str(RUN / "best.pt"))
    p.add_argument("--run", default=str(RUN))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--freeze-out", default=str(ANALYSIS / "test_freeze.json"))
    p.add_argument("--no-test", action="store_true",
                   help="write the freeze and stop before any test label is read")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    run = Path(args.run)
    train_info = C.read_json(run / "train.json")

    model, info = BT.build(int(train_info["seed"]), device,
                           dropout=float(train_info["recipe"]["dropout"]))
    state = torch.load(args.best, map_location="cpu", weights_only=False)
    if int(state["epoch"]) != int(train_info["best_valid"]["epoch"]):
        raise RuntimeError("checkpoint epoch does not match the recorded best-valid epoch")
    C.load_state(model, state["state_dict"], device)

    valid_cache = D.load_cache("valid")
    valid_rel = BD.load_relation_split("valid")
    rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)
    pred_valid = score(model, valid_cache, valid_rel, rows_valid, device, args.batch_size)
    valid_mae = C.mae(pred_valid, np.asarray(valid_cache["labels"], np.float64))
    if abs(valid_mae - float(train_info["best_valid"]["valid_mae"])) > 1e-6:
        raise RuntimeError(
            f"valid reproduction failed: {valid_mae} vs {train_info['best_valid']['valid_mae']}"
        )

    freeze = {
        "protocol": "zinc-directed-bond-transformer-v1-test-reporting",
        "checkpoint": str(Path(args.best).resolve().relative_to(C.REPO)),
        "checkpoint_sha256": C.sha256_file(args.best),
        "best_valid_epoch": int(state["epoch"]),
        "best_valid_mae": float(state["valid_mae"]),
        "valid_mae_recomputed": float(valid_mae),
        "parameter_count": int(train_info["parameter_audit"]["parameters"]),
        "recipe": train_info["recipe"],
        "git_commit": C.git_commit(),
        "selection": "argmin(valid_mae) over all 1,500 epochs, strict <",
        "test_loaded_before_this_file": False,
        "frozen_at": C.utcnow(),
    }
    C.write_json(Path(args.freeze_out), freeze)
    C.log(f"[freeze] {freeze['checkpoint']} epoch {freeze['best_valid_epoch']} "
          f"valid {valid_mae:.9f} sha256 {freeze['checkpoint_sha256'][:12]}")
    if args.no_test:
        C.log("[freeze] --no-test: stopping before any test label is read")
        return 0

    # -- only now is the test split built / read ------------------------------
    path = BD.cache_path("test")
    if path.exists():
        test_rel = BD.load_relation_split("test")
    else:
        test_rel = BD.build_relation_split("test")
        BD.save_relation_split(test_rel)
    test_cache = D.load_cache("test")
    if len(test_cache["labels"]) != 1000:
        raise RuntimeError(f"unexpected test size {len(test_cache['labels'])}")
    labels_pyg = D.official_labels("test")
    rows_test = np.arange(len(labels_pyg), dtype=np.int64)
    pred_test = score(model, test_cache, test_rel, rows_test, device, args.batch_size)
    test_metrics = describe(pred_test, labels_pyg)
    valid_metrics = describe(pred_valid, np.asarray(valid_cache["labels"], np.float64))

    pred_dir = run / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez(pred_dir / "test_best.npz", row=rows_test, pred=pred_test, y=labels_pyg)

    report = {
        "protocol": freeze["protocol"],
        "freeze": freeze,
        "valid": valid_metrics,
        "test": test_metrics,
        "test_label_sha256": C.sha256_array(np.asarray(labels_pyg, np.float64)),
        "test_cache_label_sha256": C.sha256_array(
            np.asarray(test_cache["labels"], np.float64)),
        "test_relation_cache": {"meta": test_rel["meta"],
                                "sha256": C.sha256_file(BD.cache_path("test"))},
        "note": ("the official ZINC test split is already spent in this repository; "
                 "these numbers are descriptive reporting of one frozen checkpoint, "
                 "not a pristine blind evaluation"),
    }
    C.write_json(ANALYSIS / "test_report.json", report)
    C.log(json.dumps({"valid": valid_metrics, "test": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
