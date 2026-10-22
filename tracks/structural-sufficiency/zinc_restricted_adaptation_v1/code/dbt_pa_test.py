"""Freeze the best-valid DBT-PA checkpoint, then read test once.

Order of operations:

1. load ``best.pt`` (strict-``<`` best-valid checkpoint of the formal run);
2. reproduce its valid MAE from the frozen-feature cache;
3. write ``analysis/test_freeze.json`` with checkpoint/recipe/SSL/cache hashes;
4. only then build the test object/relation/feature caches and read labels;
5. write descriptive test metrics, paired contrasts and predictions.

No test number is used to pick an epoch, checkpoint or recipe.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = _SSUF / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import dbt_data as BD  # noqa: E402
import dbt_pa_cache as PC  # noqa: E402
import dbt_pa_models as PM  # noqa: E402
import dbt_pa_train as BT  # noqa: E402

ANALYSIS = _EXP / "analysis"
RUN = _EXP / "runs" / "pa_s0"


def _scales() -> dict:
    path = ANALYSIS / "graph_scales.json"
    if path.exists():
        return C.read_json(path)
    return C.read_json(_SSUF / "zinc_directed_bond_transformer_v1" / "analysis" / "graph_scales.json")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def _score(model, obj, rel, feat, rows, device, batch_size=256) -> np.ndarray:
    pred, _ = BT.evaluate(model, obj, rel, feat, np.asarray(rows), device=device,
                          batch_size=batch_size)
    return pred


def paired_bootstrap(ae_a: np.ndarray, ae_b: np.ndarray, *, reps: int = 10000,
                     seed: int = 20261022) -> dict[str, float]:
    delta = np.asarray(ae_a, dtype=np.float64) - np.asarray(ae_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(reps, dtype=np.float64)
    chunk = max(1, min(reps, 2000))
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done:done + m] = delta[idx].mean(axis=1)
        done += m
    return {
        "mean_delta_ae": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "fraction_candidate_better": float((delta < 0).mean()),
        "reps": int(reps),
        "seed": int(seed),
        "delta_definition": "AE(candidate) - AE(reference); negative = candidate better",
    }


def _freeze(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        old = C.read_json(path)
        same = (old.get("checkpoint_sha256") == payload["checkpoint_sha256"]
                and abs(float(old.get("best_valid_mae", -1))
                        - float(payload["best_valid_mae"])) < 1e-12)
        if not same:
            raise RuntimeError(f"existing freeze {path} describes a different checkpoint")
        C.log(f"[freeze] reuse identical freeze {path}")
        return
    C.write_json(path, payload)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--best", default=str(RUN / "best.pt"))
    p.add_argument("--run", default=str(RUN))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--freeze-out", default=str(ANALYSIS / "test_freeze.json"))
    args = p.parse_args(argv)
    device = torch.device(args.device)
    run = Path(args.run)
    train_info = C.read_json(run / "train.json")
    if train_info.get("timing_only"):
        raise RuntimeError("refusing to test a timing-only run")
    if train_info.get("status") != "COMPLETE":
        raise RuntimeError(f"formal run is not complete: {train_info.get('status')}")

    model, info = PM.build_model_pa(
        D.load_norm(), int(train_info["seed"]), device, graph_scales=_scales(),
        dropout=float(train_info["recipe"]["dropout"]),
        adapter_rank=int(train_info["recipe"]["adapter_rank"]),
    )
    state = torch.load(args.best, map_location="cpu", weights_only=False)
    if int(state["epoch"]) != int(train_info["best_valid"]["epoch"]):
        raise RuntimeError("checkpoint epoch does not match the recorded best-valid epoch")
    C.load_state(model, state["state_dict"], device)

    valid_obj = BD.load_object_cache("valid")
    valid_rel = BD.load_relation_split("valid")
    valid_struct = PC.structural_fingerprint(valid_obj)
    valid_feat = PC.load_feature_split("valid", model,
                                       ssl_sha256=str(train_info["model_info"]["ssl"]["sha256"]),
                                       encoder_sha256=str(train_info["model_info"]["encoder_sha256"]),
                                       structure_sha256=valid_struct, require=True)
    rows_valid = np.arange(len(valid_obj["labels"]), dtype=np.int64)
    pred_valid = _score(model, valid_obj, valid_rel, valid_feat, rows_valid,
                        device, args.batch_size)
    valid_mae = C.mae(pred_valid, np.asarray(valid_obj["labels"], np.float64))
    if abs(valid_mae - float(train_info["best_valid"]["valid_mae"])) > 1e-6:
        raise RuntimeError(
            f"valid reproduction failed: {valid_mae} vs {train_info['best_valid']['valid_mae']}"
        )

    checkpoint_sha = C.sha256_file(args.best)
    freeze = {
        "protocol": "zinc-restricted-adaptation-v1-test-reporting",
        "candidate": "DBT-PA",
        "checkpoint": str(Path(args.best).resolve().relative_to(C.REPO)),
        "checkpoint_sha256": checkpoint_sha,
        "best_valid_epoch": int(state["epoch"]),
        "best_valid_mae": float(state["valid_mae"]),
        "valid_mae_recomputed": float(valid_mae),
        "parameter_count": int(train_info["parameter_audit"]["parameters_total_inference"]),
        "trainable_parameters": int(train_info["parameter_audit"]["parameters_trainable"]),
        "frozen_parameters": int(train_info["parameter_audit"]["parameters_frozen"]),
        "encoder_sha256": train_info["model_info"]["encoder_sha256"],
        "ssl_sha256": train_info["model_info"]["ssl"]["sha256"],
        "ssl_source": train_info["model_info"]["ssl"]["source"],
        "valid_feature_cache_sha256": valid_feat["_sha256"],
        "recipe": train_info["recipe"],
        "git_commit": C.git_commit(),
        "selection": f"argmin(valid_mae) over {train_info['recipe']['epochs']} epochs, strict <",
        "test_loaded_before_this_file": False,
        "frozen_at": C.utcnow(),
    }
    _freeze(Path(args.freeze_out), freeze)
    C.log(f"[freeze] {freeze['checkpoint']} epoch {freeze['best_valid_epoch']} "
          f"valid {valid_mae:.9f} sha {checkpoint_sha[:12]}")
    C.write_json(ANALYSIS / "valid_predictions_meta.json", {
        "checkpoint_sha256": checkpoint_sha,
        "best_valid_epoch": int(state["epoch"]),
        "valid_mae": float(valid_mae),
        "prediction_path": str((run / "predictions" / "valid_best.npz").relative_to(C.REPO)),
        "prediction_sha256": C.sha256_array(pred_valid),
    })

    # -- test is built and read only after the freeze ------------------------
    test_rel = BD.load_relation_split("test") if BD.cache_path("test").exists() else None
    if test_rel is None:
        test_rel = BD.build_relation_split("test")
        BD.save_relation_split(test_rel)
    test_obj = BD.load_object_cache("test")
    if len(test_obj["labels"]) != 1000:
        raise RuntimeError(f"unexpected test size {len(test_obj['labels'])}")
    test_struct = PC.structural_fingerprint(test_obj)
    test_feat, test_feat_path = PC.build_feature_split(
        "test", model, device,
        ssl_sha256=str(train_info["model_info"]["ssl"]["sha256"]),
        encoder_sha256=str(train_info["model_info"]["encoder_sha256"]),
        batch_size=128,
    )
    test_check = PC.verify_feature_split(model, "test", test_feat,
                                         [0, 172, 500, 999], device)
    if not test_check["all_pass"]:
        raise RuntimeError(f"test feature cache verification failed: {test_check}")
    rows_test = np.arange(1000, dtype=np.int64)
    pred_test = _score(model, test_obj, test_rel, test_feat, rows_test,
                       device, args.batch_size)
    labels_pyg = D.official_labels("test")
    labels_cache = np.asarray(test_obj["labels"], dtype=np.float64)
    if not np.allclose(labels_pyg, labels_cache, atol=1e-6, rtol=0):
        raise RuntimeError("official test labels disagree with the compact object cache")
    test_metrics = describe(pred_test, labels_pyg)
    valid_metrics = describe(pred_valid, np.asarray(valid_obj["labels"], np.float64))

    pred_dir = run / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez(pred_dir / "test_best.npz", row=rows_test, pred=pred_test, y=labels_pyg)

    # -- frozen old references ----------------------------------------------
    ref_meta = C.read_json(ANALYSIS / "reference_preds_meta.json")
    test_rows = _read_csv(ANALYSIS / "reference_preds_test.csv")
    if len(test_rows) != 1000:
        raise RuntimeError("reference test CSV is not 1000 rows")
    ref_y = np.asarray([float(r["y_raw"]) for r in test_rows])
    if not np.allclose(ref_y, labels_pyg, atol=1e-6, rtol=0):
        raise RuntimeError("reference test y is not aligned with official labels")
    ae_pa = np.abs(pred_test - labels_pyg)
    ae = {}
    for name in ("P3B_s0", "L_s0", "DBT_s0"):
        pred = np.asarray([float(r[f"pred_{name}"]) for r in test_rows], dtype=np.float64)
        if abs(float(np.abs(pred - ref_y).mean()) - float(ref_meta["test"]["mae"][name])) > 1e-6:
            raise RuntimeError(f"reference {name} test MAE does not reproduce")
        ae[name] = np.abs(pred - labels_pyg)
    test_contrasts = {name: paired_bootstrap(ae_pa, ae[name])
                      for name in ("P3B_s0", "L_s0", "DBT_s0")}

    valid_rows = _read_csv(ANALYSIS / "reference_preds_valid.csv")
    valid_y = np.asarray([float(r["y_raw"]) for r in valid_rows])
    if not np.allclose(valid_y, np.asarray(valid_obj["labels"], np.float64), atol=1e-6, rtol=0):
        raise RuntimeError("reference valid y is not aligned")
    ae_pa_v = np.abs(pred_valid - valid_y)
    valid_contrasts = {}
    for name in ("P3B_s0", "L_s0", "DBT_s0"):
        pred = np.asarray([float(r[f"pred_{name}"]) for r in valid_rows], dtype=np.float64)
        if abs(float(np.abs(pred - valid_y).mean()) - float(ref_meta["valid"]["mae"][name])) > 1e-6:
            raise RuntimeError(f"reference {name} valid MAE does not reproduce")
        valid_contrasts[name] = paired_bootstrap(ae_pa_v, np.abs(pred - valid_y))

    report = {
        "protocol": freeze["protocol"],
        "freeze": freeze,
        "valid": valid_metrics,
        "test": test_metrics,
        "test_label_sha256": C.sha256_array(np.asarray(labels_pyg, np.float64)),
        "valid/feature_cache": {
            "path": str(valid_feat["_path"]),
            "sha256": valid_feat["_sha256"],
            "identity": {k: valid_feat["meta"][k] for k in
                         ("feature_version", "split", "ssl_sha256",
                          "encoder_sha256", "structure_sha256")},
        },
        "test/feature_cache": {
            "path": str(test_feat_path),
            "sha256": C.sha256_file(test_feat_path),
            "identity": {k: test_feat["meta"][k] for k in
                         ("feature_version", "split", "ssl_sha256",
                          "encoder_sha256", "structure_sha256")},
            "verification": test_check,
        },
        "object_caches": {
            split: {"path": str(BD.object_cache_path(split).relative_to(C.REPO)),
                    "sha256": C.sha256_file(BD.object_cache_path(split))}
            for split in ("valid", "test")
        },
        "test_contrasts": test_contrasts,
        "valid_contrasts": valid_contrasts,
        "reference_meta": ref_meta,
        "note": ("the official ZINC test split is already spent in this repository; "
                 "these numbers are descriptive reporting of one frozen checkpoint, "
                 "not a pristine blind evaluation"),
    }
    C.write_json(ANALYSIS / "test_report.json", report)

    # Compact main table for the report.
    main = {
        "rows": [
            {"model": "P3B_s0", "params": 413505, "best_valid": 0.076223754,
             "test_mae": 0.069864841, "status": "reused"},
            {"model": "L_s0", "params": 494193, "best_valid": 0.078537152,
             "test_mae": 0.066266555, "status": "reused"},
            {"model": "DBT_s0", "params": 452523, "best_valid": 0.077768402,
             "best_epoch": 949, "eval_train_at_best": 0.015964,
             "test_mae": 0.069401494, "updates": 235500, "status": "reused"},
            {"model": "DBT-PA_s0", "params": int(train_info["parameter_audit"]["parameters_total_inference"]),
             "best_valid": float(train_info["best_valid"]["valid_mae"]),
             "best_epoch": int(train_info["best_valid"]["epoch"]),
             "eval_train_at_best": float(train_info["fixed_objects"]["best"]["train_mae_eval_mode"]),
             "test_mae": float(test_metrics["mae"]),
             "updates": int(train_info["recipe"]["total_updates"]),
             "gpu_hours": train_info["gpu_hours"], "status": "this round"},
        ],
        "test_contrasts": test_contrasts,
        "valid_contrasts": valid_contrasts,
    }
    C.write_json(ANALYSIS / "main_table.json", main)
    C.log(json.dumps({"valid": valid_metrics, "test": test_metrics,
                      "test_contrasts": test_contrasts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
