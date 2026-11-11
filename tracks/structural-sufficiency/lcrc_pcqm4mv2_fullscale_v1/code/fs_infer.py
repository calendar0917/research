"""Model selection, official Test-dev feature build, inference and submission.

Stage order (enforced by the CLI):

1. ``select``  — after both formal runs are COMPLETED, pick the single model by
   lowest best Validation MAE (tie -> fewer parameters) and freeze its identity.
2. ``features`` — official ``split['test-dev']`` rows in original order,
   SMILES taken from the raw CSV's ``smiles`` column only (labels are never
   read), frozen ``lipo_v1`` adapter, merged mmap cache.
3. ``predict`` — load the frozen checkpoint, one inference pass, inverse
   normalisation to eV, ``PCQM4Mv2Evaluator.save_test_submission(mode="test-dev")``.
4. ``end2end`` — features + predict in one process, timed from raw SMILES.
5. ``verify-submission`` — independent re-open of the ``.npz``.

No Test label is read, inferred or scored anywhere in this module.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import fs_common as C
import fs_build as B
import fs_data as D

import torch

TESTDEV = "test-dev"
CACHE_NAME = "pcqm_testdev"
TEST_MANIFEST = C.ANALYSIS / "fs_testdev_cache_manifest.json"
SELECTION = C.ANALYSIS / "fs_model_selection.json"
SUBMISSION_DIR = C.RUNS / "PCQM_TESTDEV_SUBMISSION"
REPORT = C.ANALYSIS / "fs_testdev_inference.json"


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def select_model() -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for run in C.RUN_ORDER:
        path = C.RUNS / run / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"{run}: run.json missing ({path})")
        payload = C.read_json(path)
        if payload.get("status") != "COMPLETED" or int(payload.get("epochs_run", 0)) != int(C.TOTAL_EPOCHS):
            raise RuntimeError(
                f"{run} is not complete ({payload.get('status')}, "
                f"{payload.get('epochs_run')}/{C.TOTAL_EPOCHS} epochs); selection is not allowed"
            )
        ckpt = Path(C.REPO / payload["best_checkpoint"])
        if not ckpt.is_file():
            raise FileNotFoundError(f"{run}: best checkpoint missing ({ckpt})")
        sha = C.sha256_file(ckpt)
        if str(payload.get("best_checkpoint_sha256")) != sha:
            raise RuntimeError(f"{run}: checkpoint hash does not match run.json")
        runs[run] = {
            "run": run,
            "model": payload["model"],
            "scale": int(payload["scale"]),
            "parameters": int(payload["parameters"]),
            "best_epoch": int(payload["best_epoch"]),
            "best_valid_mae_eV": float(payload["best_valid_mae_eV"]),
            "final_epoch_valid_mae_eV": float(payload["final_epoch_valid_mae_eV"]),
            "final_epoch_monitor_mae_eV": float(payload["final_epoch_monitor_mae_eV"]),
            "checkpoint": payload["best_checkpoint"],
            "checkpoint_sha256": sha,
            "gpu_hours": float(payload["gpu_hours"]),
            "steps_total": int(payload["steps_total"]),
            "epochs": int(payload["epochs"]),
            "train_size": int(payload["train_size"]),
            "row_plan_sha256": payload["row_plan"]["rows_sha256"],
            "feature_norm_fingerprint": payload["feature_norm"]["norm_fingerprint"],
            "label_norm_mean_eV": float(payload["label_norm"]["mean_eV"]),
            "label_norm_std_eV": float(payload["label_norm"]["std_eV"]),
            "git_commit": payload["git_commit"],
            "curve_sha256": payload["curve_sha256"],
        }
    # lowest Validation MAE; tie -> fewer parameters; final tie -> RUN_ORDER
    order = sorted(
        runs.values(),
        key=lambda r: (
            float(r["best_valid_mae_eV"]),
            int(r["parameters"]),
            C.RUN_ORDER.index(r["run"]),
        ),
    )
    chosen = order[0]
    split_payload = D.load_split_dict()
    dev_rows = np.asarray(split_payload["split"][TESTDEV], dtype=np.int64)
    payload = {
        "schema": "lcrc-pcqm4mv2-fullscale-selection-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "selection_rule": "lowest best Validation MAE; tie -> fewer parameters; then RUN_ORDER",
        "runs": runs,
        "ranking": [r["run"] for r in order],
        "selected_run": chosen["run"],
        "selected_model": chosen["model"],
        "selected_scale": int(chosen["scale"]),
        "selected_parameters": int(chosen["parameters"]),
        "selected_best_epoch": int(chosen["best_epoch"]),
        "selected_best_valid_mae_eV": float(chosen["best_valid_mae_eV"]),
        "selected_checkpoint": chosen["checkpoint"],
        "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
        "frozen": {
            "feature_spec_hash": C.PQ_DATA.FS.load_spec(C.PARENT_ANALYSIS / "pcqm_spec.json").hash(),
            "feature_norm_sha256": C.sha256_file(C.PARENT_ANALYSIS / "pcqm_feature_norm.json"),
            "label_norm_sha256": C.sha256_file(C.PARENT_ANALYSIS / "pcqm_label_norm.json"),
            "label_norm_mean_eV": float(chosen["label_norm_mean_eV"]),
            "label_norm_std_eV": float(chosen["label_norm_std_eV"]),
        },
        "testdev_rows": {
            "n": int(dev_rows.size),
            "index_sha256": C.index_hash(dev_rows),
            "first": int(dev_rows[0]),
            "last": int(dev_rows[-1]),
            "split_dict_sha256": split_payload["split_dict"]["sha256"],
        },
        "test_mae": "PENDING_OFFICIAL_EVALUATION",
        "test_labels_read": False,
    }
    C.write_json(SELECTION, payload)
    C.log(
        f"[select] {chosen['run']} best_valid={chosen['best_valid_mae_eV']:.6f} eV "
        f"epoch={chosen['best_epoch']} params={chosen['parameters']}"
    )
    return payload


def load_selection() -> dict[str, Any]:
    if not SELECTION.is_file():
        raise FileNotFoundError(f"selection missing: {SELECTION}")
    payload = C.read_json(SELECTION)
    ckpt = Path(C.REPO / payload["selected_checkpoint"])
    sha = C.sha256_file(ckpt)
    if sha != payload["selected_checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint hash changed since selection")
    return payload


# ---------------------------------------------------------------------------
# official Test-dev features
# ---------------------------------------------------------------------------


def official_smiles_rows(rows: Sequence[int], *, csv_path: Path | None = None) -> dict[str, Any]:
    """SMILES of dataset row ids, read from the raw CSV's ``smiles`` column only.

    ``usecols=['idx','smiles']`` deliberately excludes the label column, so no
    Test label is parsed even in memory.
    """
    import pandas as pd

    path = Path(csv_path) if csv_path else (C.PQ.PCQM_ROOT / "raw" / "data.csv.gz")
    if not path.is_file():
        raise FileNotFoundError(f"official raw CSV missing: {path}")
    frame = pd.read_csv(path, usecols=["idx", "smiles"])
    idx = frame["idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(idx, np.arange(idx.size, dtype=np.int64)):
        raise RuntimeError("raw CSV idx column is not the dataset row order")
    smiles = frame["smiles"].astype(str).tolist()
    rows = [int(r) for r in rows]
    if min(rows) < 0 or max(rows) >= len(smiles):
        raise RuntimeError("Test-dev row outside the dataset")
    return {
        "smiles": [smiles[r] for r in rows],
        "csv_path": C.track_rel(path),
        "csv_sha256": C.sha256_file(path),
        "csv_rows": int(len(smiles)),
        "labels_column_read": False,
    }


def build_testdev_cache(*, workers: int = 32, chunk_size: int = 4000) -> dict[str, Any]:
    t0 = time.perf_counter()
    split_payload = D.load_split_dict(verify_with_ogb=False)
    rows = np.asarray(split_payload["split"][TESTDEV], dtype=np.int64)
    if rows.size != C.TEST_DEV_ROWS:
        raise RuntimeError(f"test-dev split has {rows.size} rows != {C.TEST_DEV_ROWS}")
    if not np.all(np.diff(rows) > 0):
        raise RuntimeError("test-dev split is not sorted ascending (original order)")

    # hard guard: never touch train/valid rows
    train_ids = np.asarray(split_payload["split"]["train"], dtype=np.int64)
    valid_ids = np.asarray(split_payload["split"]["valid"], dtype=np.int64)
    if int(np.intersect1d(rows, train_ids).size) or int(np.intersect1d(rows, valid_ids).size):
        raise RuntimeError("test-dev rows overlap Train or Valid")

    fetched = official_smiles_rows(rows)
    # The feature builder indexes a SMILES list; build with local positions
    # 0..147036 (same original order) and write the official row ids as gids
    # after the merge, so the cache keeps the training caches' gid semantics.
    local_rows = np.arange(rows.size, dtype=np.int64)
    shard_dir = C.CACHE / "shards_testdev"
    summary_path = C.ANALYSIS / "fs_testdev_cache_build.json"
    ledger_path = C.ANALYSIS / "fs_testdev_parse_ledger.json"
    failures_path = C.ANALYSIS / "fs_testdev_adapter_failures.json"
    build = B.build_shards_full(
        rows=local_rows,
        smiles_all=fetched["smiles"],
        shard_dir=shard_dir,
        tag="pcqm_testdev",
        workers=int(workers),
        chunk_size=int(chunk_size),
        summary_path=summary_path,
        ledger_path=ledger_path,
        failures_path=failures_path,
    )
    if build.get("status") != "OK":
        raise RuntimeError(f"test-dev feature build status {build.get('status')} != OK")
    spec = C.PQ_DATA.FS.load_spec(C.PARENT_ANALYSIS / "pcqm_spec.json")
    out_dir = C.CACHE / CACHE_NAME
    cache = C.PQ_DATA.merge_shards(
        [C.REPO / s["path"] for s in build["shards"]],
        spec,
        out_dir=out_dir,
        split=TESTDEV,
        dataset=C.DATASET,
        tag="pcqm_testdev",
    )
    merged_gids = np.load(out_dir / "gids.npy")
    if not np.array_equal(np.asarray(merged_gids), local_rows):
        raise RuntimeError("merged test-dev cache did not keep the local build order")
    np.save(out_dir / "gids.npy", rows)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "round_id": C.ROUND_ID,
            "tag": "pcqm_testdev",
            "split": TESTDEV,
            "labels_present": False,
            "labels_in_cache": False,
            "uses_3d_geometry": False,
            "row_order": "official split['test-dev'] ascending (original order)",
            "rows_sha256": C.index_hash(rows),
            "build_summary": C.track_rel(summary_path),
            "raw_csv": fetched["csv_path"],
            "raw_csv_sha256": fetched["csv_sha256"],
            "labels_column_read": False,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    tree_sha = C.sha256_tree(out_dir)
    loaded = D.load_cache(out_dir)
    checks = {
        "n_molecules": int(loaded["n_molecules"]) == C.TEST_DEV_ROWS,
        "gids_equal_split": bool(np.array_equal(np.asarray(loaded["gids"]), rows)),
        "spec_hash": str(loaded["meta"]["feature_spec_hash"]) == spec.hash(),
        "labels_absent": not bool(loaded["meta"].get("labels_present", True)),
        "labels_column_read": not bool(loaded["meta"].get("labels_column_read", True)),
        "no_train_overlap": int(np.intersect1d(np.asarray(loaded["gids"]), train_ids).size) == 0,
        "no_valid_overlap": int(np.intersect1d(np.asarray(loaded["gids"]), valid_ids).size) == 0,
    }
    failed = [k for k, v in checks.items() if not v]
    manifest = {
        "schema": "lcrc-pcqm4mv2-fullscale-testdev-cache-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "cache_path": C.track_rel(out_dir),
        "cache_sha256": tree_sha,
        "cache_bytes": int(sum(p.stat().st_size for p in out_dir.glob("*") if p.is_file())),
        "n_molecules": int(loaded["n_molecules"]),
        "n_atoms_total": int(np.asarray(loaded["atom_ptr"])[-1]),
        "n_bonds_total": int(np.asarray(loaded["bond_ptr"])[-1]),
        "n_s1_codes": int(np.asarray(loaded["nbr_ptr"])[-1]),
        "n_s2_paths": int(np.asarray(loaded["path_ptr"])[-1]),
        "rows_sha256": C.index_hash(rows),
        "split_dict_sha256": split_payload["split_dict"]["sha256"],
        "raw_csv": fetched["csv_path"],
        "raw_csv_sha256": fetched["csv_sha256"],
        "labels_column_read": False,
        "test_mae": "PENDING_OFFICIAL_EVALUATION",
        "checks": checks,
        "build_seconds": float(build["wall_seconds"]),
        "merge_seconds": float(cache["merge_seconds"]),
        "total_seconds": float(time.perf_counter() - t0),
        "parse_modes": build["parse_modes"],
        "n_fallback_rows": int(build["n_fallback_rows"]),
        "status": "OK" if not failed else "FAILED",
    }
    C.write_json(TEST_MANIFEST, manifest)
    if failed:
        raise RuntimeError(f"test-dev cache checks failed: {failed}")
    import shutil

    if shard_dir.is_dir():
        shutil.rmtree(shard_dir)
    C.log(
        f"[testdev] cache OK rows={manifest['n_molecules']} {tree_sha[:12]} "
        f"build+merge={manifest['total_seconds']:.1f}s"
    )
    return manifest


def load_testdev_cache(*, verify: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    if not TEST_MANIFEST.is_file():
        raise FileNotFoundError(f"test-dev manifest missing: {TEST_MANIFEST}")
    manifest = C.read_json(TEST_MANIFEST)
    if manifest.get("status") != "OK":
        raise RuntimeError(f"test-dev cache manifest status {manifest.get('status')} != OK")
    out_dir = C.CACHE / CACHE_NAME
    if verify and C.sha256_tree(out_dir) != manifest["cache_sha256"]:
        raise RuntimeError("test-dev cache tree hash changed")
    cache = D.load_cache(out_dir)
    return cache, manifest


# ---------------------------------------------------------------------------
# inference + submission
# ---------------------------------------------------------------------------


def predict_testdev(*, device: torch.device, batch_size: int = 512) -> dict[str, Any]:
    selection = load_selection()
    cache, manifest = load_testdev_cache(verify=True)
    rows = np.asarray(cache["gids"], dtype=np.int64)
    positions = np.arange(rows.size, dtype=np.int64)
    if rows.size != C.TEST_DEV_ROWS:
        raise RuntimeError("test-dev cache size changed")
    if C.index_hash(rows) != selection["testdev_rows"]["index_sha256"]:
        raise RuntimeError("test-dev row order differs from the frozen selection")

    ckpt_path = Path(C.REPO / selection["selected_checkpoint"])
    if C.sha256_file(ckpt_path) != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint hash changed")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if checkpoint.get("run") != selection["selected_run"]:
        raise RuntimeError("checkpoint run id differs from selection")
    if int(checkpoint.get("scale", -1)) != int(selection["selected_scale"]):
        raise RuntimeError("checkpoint width differs from selection")
    if str(checkpoint.get("feature_spec_hash")) != str(selection["frozen"]["feature_spec_hash"]):
        raise RuntimeError("checkpoint feature spec differs from the frozen one")
    if float(checkpoint["label_norm"]["mean_eV"]) != float(selection["frozen"]["label_norm_mean_eV"]):
        raise RuntimeError("checkpoint label normalisation differs from the frozen one")

    spec = C.PQ_DATA.FS.load_spec(C.PARENT_ANALYSIS / "pcqm_spec.json")
    feature_norm = C.read_json(C.PARENT_ANALYSIS / "pcqm_feature_norm.json")
    norm = {
        "global_mean": np.asarray(feature_norm["global_mean_list"], dtype=np.float32),
        "global_scale": np.asarray(feature_norm["global_scale_list"], dtype=np.float32),
        "scalar_mean": np.asarray(feature_norm["scalar_mean_list"], dtype=np.float32),
        "scalar_scale": np.asarray(feature_norm["scalar_scale_list"], dtype=np.float32),
    }
    label_norm = C.read_json(C.PARENT_ANALYSIS / "pcqm_label_norm.json")
    C.apply_deterministic_settings()
    C.seed_everything(int(C.MODEL_SEED))
    model, audit = C.EW_MODEL.build_model(
        spec, norm, scale=int(selection["selected_scale"]), seed=int(C.MODEL_SEED), device=device
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    n = int(rows.size)
    pred_z = np.empty(n, dtype=np.float32)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, n, int(batch_size)):
            chunk = positions[i : i + int(batch_size)]
            raw = C.EW_COLLATE.collate_v2(cache, chunk.tolist())
            batch = {k: (v.to(device, non_blocking=False) if torch.is_tensor(v) else v) for k, v in raw.items()}
            pred = C.EW_MODEL.predict_z(model, batch)
            pred_z[i : i + chunk.size] = pred.detach().float().cpu().numpy()
    inference_seconds = float(time.perf_counter() - t0)
    pred_eV = C.PQ_DATA.from_z(pred_z, label_norm).astype(np.float64)
    if pred_eV.shape != (C.TEST_DEV_ROWS,):
        raise RuntimeError(f"prediction shape {pred_eV.shape} != ({C.TEST_DEV_ROWS},)")
    if not np.isfinite(pred_eV).all():
        raise RuntimeError("non-finite Test-dev prediction")

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    evaluator = C.PQ_DATA._evaluator()
    evaluator.save_test_submission(
        input_dict={"y_pred": pred_eV}, dir_path=str(SUBMISSION_DIR), mode="test-dev"
    )
    npz_path = SUBMISSION_DIR / "y_pred_pcqm4m-v2_test-dev.npz"
    if not npz_path.is_file():
        raise RuntimeError(f"official submission file not written: {npz_path}")

    # independent re-open and strict checks
    with np.load(npz_path) as data:
        saved = np.asarray(data["y_pred"])
    checks = {
        "key_y_pred": True,
        "shape_is_147037": tuple(saved.shape) == (C.TEST_DEV_ROWS,),
        "dtype_float32": saved.dtype == np.float32,
        "all_finite": bool(np.isfinite(saved).all()),
        "matches_float32_predictions": bool(np.array_equal(saved, pred_eV.astype(np.float32))),
        "row_order_matches_split": bool(
            np.array_equal(rows, np.asarray(load_split_rows(), dtype=np.int64))
        ),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"submission verification failed: {failed}")

    raw_npy = SUBMISSION_DIR / "predictions_testdev_eV.npy"
    np.save(raw_npy, pred_eV)
    payload = {
        "schema": "lcrc-pcqm4mv2-fullscale-testdev-inference-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "selected_run": selection["selected_run"],
        "selected_model": selection["selected_model"],
        "selected_scale": int(selection["selected_scale"]),
        "selected_parameters": int(selection["selected_parameters"]),
        "selected_best_epoch": int(selection["selected_best_epoch"]),
        "selected_best_valid_mae_eV": float(selection["selected_best_valid_mae_eV"]),
        "checkpoint": selection["selected_checkpoint"],
        "checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "testdev_rows_sha256": C.index_hash(rows),
        "testdev_rows_n": int(rows.size),
        "testdev_rows_first": int(rows[0]),
        "testdev_rows_last": int(rows[-1]),
        "testdev_cache_sha256": manifest["cache_sha256"],
        "prediction_dtype": str(pred_eV.dtype),
        "prediction_sha256_float64": C.sha256_bytes(np.ascontiguousarray(pred_eV, dtype=np.float64).tobytes()),
        "prediction_sha256_float32": C.sha256_bytes(np.ascontiguousarray(pred_eV, dtype=np.float32).tobytes()),
        "prediction_min_eV": float(pred_eV.min()),
        "prediction_max_eV": float(pred_eV.max()),
        "prediction_mean_eV": float(pred_eV.mean()),
        "prediction_std_eV": float(pred_eV.std()),
        "inference_seconds": inference_seconds,
        "molecules_per_second": float(n / inference_seconds) if inference_seconds > 0 else None,
        "batch_size": int(batch_size),
        "device": str(device),
        "submission_file": C.track_rel(npz_path),
        "submission_sha256": C.sha256_file(npz_path),
        "submission_bytes": int(npz_path.stat().st_size),
        "raw_prediction_file": C.track_rel(raw_npy),
        "raw_prediction_sha256": C.sha256_file(raw_npy),
        "checks": checks,
        "test_mae": "PENDING_OFFICIAL_EVALUATION",
        "test_mae_note": "no Test-dev label was read; only the official evaluator can score this file",
        "test_labels_read": False,
        "environment": C.environment(),
        "host_facts": C.host_facts(),
    }
    C.write_json(REPORT, payload)
    C.log(
        f"[predict] {payload['selected_run']} rows={n} {inference_seconds:.1f}s "
        f"({payload['molecules_per_second']:.0f} mol/s) file={C.track_rel(npz_path)} "
        f"sha={payload['submission_sha256'][:12]}"
    )
    return payload


def load_split_rows() -> np.ndarray:
    return np.asarray(D.load_split_dict()["split"][TESTDEV], dtype=np.int64)


def verify_submission() -> dict[str, Any]:
    selection = load_selection()
    npz_path = SUBMISSION_DIR / "y_pred_pcqm4m-v2_test-dev.npz"
    with np.load(npz_path) as data:
        saved = np.asarray(data["y_pred"])
    rows = load_split_rows()
    checks = {
        "shape": tuple(saved.shape) == (C.TEST_DEV_ROWS,),
        "dtype_float32": saved.dtype == np.float32,
        "all_finite": bool(np.isfinite(saved).all()),
        "row_order": C.index_hash(rows) == selection["testdev_rows"]["index_sha256"],
        "checkpoint_sha256": C.sha256_file(Path(C.REPO / selection["selected_checkpoint"]))
        == selection["selected_checkpoint_sha256"],
    }
    payload = {
        "schema": "lcrc-pcqm4mv2-fullscale-testdev-submission-verify-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "submission_file": C.track_rel(npz_path),
        "submission_sha256": C.sha256_file(npz_path),
        "submission_bytes": int(npz_path.stat().st_size),
        "n": int(saved.size),
        "dtype": str(saved.dtype),
        "min_eV": float(saved.min()),
        "max_eV": float(saved.max()),
        "finite": bool(np.isfinite(saved).all()),
        "row_order_sha256": C.index_hash(rows),
        "checks": checks,
        "test_mae": "PENDING_OFFICIAL_EVALUATION",
        "status": "SUBMISSION_READY_NOT_SCORED" if all(checks.values()) else "FAILED",
    }
    C.write_json(C.ANALYSIS / "fs_testdev_submission_verify.json", payload)
    C.log(f"[verify-submission] {payload['status']} {payload['submission_sha256'][:12]}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("select", "features", "predict", "end2end", "verify-submission"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    if args.stage == "select":
        select_model()
        return 0
    if args.stage == "features":
        manifest = build_testdev_cache(workers=int(args.workers))
        C.log(f"features done in {manifest['total_seconds']:.1f}s")
        return 0
    if args.stage == "predict":
        payload = predict_testdev(device=torch.device(args.device), batch_size=int(args.batch_size))
        C.log(f"predict done in {payload['inference_seconds']:.1f}s")
        return 0
    if args.stage == "end2end":
        load_selection()
        manifest = build_testdev_cache(workers=int(args.workers))
        payload = predict_testdev(device=torch.device(args.device), batch_size=int(args.batch_size))
        total = float(time.perf_counter() - t0)
        payload["end_to_end"] = {
            "definition": "raw official SMILES (test-dev rows) -> cache build -> merge -> GPU inference -> .npz",
            "feature_build_seconds": float(manifest["build_seconds"]),
            "feature_merge_seconds": float(manifest["merge_seconds"]),
            "feature_total_seconds": float(manifest["total_seconds"]),
            "inference_seconds": float(payload["inference_seconds"]),
            "script_total_seconds": total,
            "official_limit_hours": 4.0,
            "within_limit": bool(total <= 4 * 3600.0),
            "device": str(args.device),
            "cpu_workers": int(args.workers),
        }
        C.write_json(REPORT, payload)
        C.log(f"[end2end] total {total:.1f}s within 4h={payload['end_to_end']['within_limit']}")
        return 0
    verify = verify_submission()
    return 0 if verify["status"] != "FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
