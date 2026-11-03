"""The single one-shot Test pass for the sparse-identity round.

Preconditions (all verified before any label is read):

* ``analysis/checkpoint_manifest.json`` exists and its own SHA-256 recomputes;
* every frozen code file still hashes to the manifest value (so the scoring uses
  the frozen training/model/feature code);
* every checkpoint file still hashes to the manifest value.

Then — and only then — the label-free Test structure cache is encoded with the
frozen vocabulary, the WIDE_NULL Test randomisation is created with seed
20261105, the Test labels are read once (``ptb_data.molecule_test``), and all 12
frozen checkpoints plus SPARSE_ONLY are scored exactly once.  Nothing is
re-selected, re-ranked or retrained afterwards; the access is appended to
``analysis/test_access_ledger.json`` and the result is tagged
``EXPOSED_TEST_EXPLORATORY``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import si_common as C
import si_features as FEAT
import si_model as M

import mb2_metrics as MET
import ptb_data as PD


def _verify_manifest(manifest: Mapping[str, Any]) -> None:
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if C.sha256_text(C.stable_json(payload)) != manifest["manifest_sha256"]:
        raise RuntimeError("checkpoint manifest SHA-256 does not recompute; the manifest was modified")
    if not manifest.get("frozen_before_test_access"):
        raise RuntimeError("manifest is not marked as frozen before Test access")
    drifted = []
    for rel, sha in manifest["code_hashes"].items():
        path = _resolve_code_path(rel)
        current = C.sha256_file(path) if path.exists() else None
        if current != sha:
            drifted.append({"file": rel, "frozen": sha, "current": current})
    if drifted:
        raise RuntimeError(f"frozen code changed after the freeze: {drifted}")


def _resolve_code_path(rel: str) -> Path:
    if rel.startswith("code/"):
        return C.CODE_DIR / rel.split("/", 1)[1]
    if rel.startswith("tests/"):
        return C.TESTS / rel.split("/", 1)[1]
    if rel.startswith("reused/"):
        name = rel.split("/", 1)[1]
        return C.REUSED_MODULES[name]
    return C.TRACK / rel


def _ledger_append(events: list[dict[str, Any]]) -> None:
    path = C.ANALYSIS / "test_access_ledger.json"
    payload = C.read_json(path) if path.exists() else {
        "schema": "lcrc-sparse-identity-test-access-v1",
        "round_id": C.ROUND_ID,
        "events": [],
    }
    payload["events"].extend(events)
    payload["n_events"] = len(payload["events"])
    payload["updated_utc"] = C.utcnow()
    C.write_json(path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One-shot Test scoring of every frozen model")
    ap.parse_args(argv)
    out_path = C.ANALYSIS / "test_scores.json"
    if out_path.exists():
        raise SystemExit(f"{C.track_rel(out_path)} exists; Test is scored exactly once")
    manifest = C.read_json(C.ANALYSIS / "checkpoint_manifest.json")
    _verify_manifest(manifest)

    view = PD.molecule_view(C.DATASET)
    spec = view.spec
    vocab = C.read_json(C.ANALYSIS / "feature_vocabulary.json")

    # ---- label-free Test structures ----------------------------------------
    test_cache, y_test_tmp, test_rows = PD.molecule_test(C.DATASET)
    test_feats = FEAT.encode_split(test_cache, spec, vocab)
    fold = FEAT.fold_buckets(vocab)
    x_true = test_feats["x"]
    z_fold = FEAT.project_fold64(x_true, fold["bucket_of_col"])
    null_binary, null_stats = FEAT.double_edge_swap(test_cache_binary(test_feats), C.NULL_SEED["test"])
    active = null_binary.sum(axis=1).astype(np.int64)
    scale = (1.0 / np.sqrt(np.maximum(1, active).astype(np.float64))).astype(np.float32)
    x_null = (null_binary.astype(np.float32) * scale[:, None]).astype(np.float32)

    rows_now = [int(x) for x in np.asarray(test_cache["rows"]).tolist()]
    if rows_now != [int(x) for x in test_rows]:
        raise RuntimeError("Test cache rows differ from the published split")
    if C.index_sha256(rows_now) != manifest["test_rows_sha256"]:
        raise RuntimeError("Test row ids changed since the freeze")
    y_test = np.asarray(y_test_tmp, dtype=np.float64).reshape(-1)
    if y_test.size != len(rows_now):
        raise RuntimeError("Test label length does not match the frozen rows")

    branch_inputs = {
        C.MODEL_TRUE: x_true,
        C.MODEL_NULL: x_null,
        C.MODEL_FOLD: z_fold,
    }
    device = torch.device("cpu")
    scores: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for unit in manifest["units"]:
        arm = str(unit["arm"])
        seed = int(unit["seed"])
        ckpt_path = C.REPO / unit["checkpoint_path"]
        sha = C.sha256_file(ckpt_path)
        if sha != unit["checkpoint_sha256"]:
            raise RuntimeError(f"{unit['unit']}: checkpoint changed after the freeze")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = M.load_model(arm, spec, view.train_norm, ckpt["state_dict"], device, in_dim=C.BRANCH_DIM[arm])
        feats = None if arm == C.MODEL_A0 else branch_inputs[arm]
        pred = _predict(model, test_cache, feats, device)
        metrics = MET.task_metrics(C.TASK, pred, y_test, view.target_norm, ["HIV_active"])
        official = MET.ogb_evaluator_roc_auc(y_test, pred.reshape(-1))
        contrib = None
        if arm in C.WIDE:
            contrib = M.contribution_stats(model.branch_contribution(feats, device))
        np.savez(
            C.PREDICTIONS / f"{arm}_s{seed}_test.npz",
            rows=np.asarray(rows_now, dtype=np.int64),
            pred=pred,
            y_true=y_test,
        )
        scores[unit["unit"]] = {
            "arm": arm,
            "seed": seed,
            "checkpoint_sha256": sha,
            "metrics": metrics,
            "official_ogb": official,
            "sklearn_roc_auc_check": {
                "official": official["ogb_rocauc"],
                "sklearn": official["sklearn_roc_auc"],
                "abs_diff": official["abs_diff"],
                "consistent": official["consistent"],
            },
            "branch_contribution": contrib,
            "branch_norms": unit["branch"],
            "best_epoch": int(unit["best_epoch"]),
            "result_label": C.RESULT_LABEL,
        }
        events.append(
            {
                "utc": C.utcnow(),
                "unit": unit["unit"],
                "arm": arm,
                "seed": seed,
                "n_test": int(len(rows_now)),
                "checkpoint_sha256": sha,
                "manifest_sha256": manifest["manifest_sha256"],
                "frozen_before_access": True,
                "selection_used_test": False,
                "result_label": C.RESULT_LABEL,
            }
        )

    sparse = manifest["sparse_only"]
    with np.load(C.REPO / sparse["model_path"]) as handle:
        coef = np.asarray(handle["coef"], dtype=np.float64)
        intercept = float(np.asarray(handle["intercept"], dtype=np.float64).reshape(-1)[0])
    if C.array_hash_f32(coef.astype(np.float32)) != sparse["coef_sha256"]:
        raise RuntimeError("SPARSE_ONLY coefficients changed after the freeze")
    sparse_logits = (x_true.astype(np.float64) @ coef + intercept).astype(np.float64)
    sparse_metrics = MET.task_metrics(C.TASK, sparse_logits.reshape(-1, 1), y_test, view.target_norm, ["HIV_active"])
    sparse_official = MET.ogb_evaluator_roc_auc(y_test, sparse_logits)
    np.savez(
        C.PREDICTIONS / f"{C.MODEL_SPARSE}_test.npz",
        rows=np.asarray(rows_now, dtype=np.int64),
        pred=sparse_logits.reshape(-1, 1),
        y_true=y_test,
    )
    scores[C.MODEL_SPARSE] = {
        "arm": C.MODEL_SPARSE,
        "seed": int(C.SPARSE_RANDOM_STATE),
        "model_sha256": sparse["model_sha256"],
        "metrics": sparse_metrics,
        "official_ogb": sparse_official,
        "sklearn_roc_auc_check": {
            "official": sparse_official["ogb_rocauc"],
            "sklearn": sparse_official["sklearn_roc_auc"],
            "abs_diff": sparse_official["abs_diff"],
            "consistent": sparse_official["consistent"],
        },
        "coef_l1": float(sparse["coef_l1"]),
        "coef_l2": float(sparse["coef_l2"]),
        "result_label": C.RESULT_LABEL,
    }
    events.append(
        {
            "utc": C.utcnow(),
            "unit": C.MODEL_SPARSE,
            "arm": C.MODEL_SPARSE,
            "seed": int(C.SPARSE_RANDOM_STATE),
            "n_test": int(len(rows_now)),
            "model_sha256": sparse["model_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "frozen_before_access": True,
            "result_label": C.RESULT_LABEL,
        }
    )

    payload = {
        "schema": "lcrc-sparse-identity-test-scores-v1",
        "round_id": C.ROUND_ID,
        "scored_utc": C.utcnow(),
        "manifest_sha256": manifest["manifest_sha256"],
        "n_units": len(scores),
        "n_test": int(len(rows_now)),
        "test_rows_sha256": C.index_sha256(rows_now),
        "test_structure_sha256": test_feats["stats"]["binary_sha256"],
        "test_vector_sha256": test_feats["stats"]["vector_sha256"],
        "test_feature_stats": test_feats["stats"],
        "test_rows_in_vocabulary_coverage": test_feats["stats"]["molecule_coverage"],
        "test_oov_path_ratio": test_feats["stats"]["oov_path_ratio"],
        "test_null_stats": {
            k: v
            for k, v in null_stats.items()
            if not isinstance(v, (list, dict)) and k not in ("seed",)
        },
        "test_null_seed": int(C.NULL_SEED["test"]),
        "execution_regime": {
            "device": "cpu",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "hostname": C.env_info()["hostname"],
            "python": C.env_info()["python"],
            "git_commit": C.git_commit(),
            "note": "Test forward passes were executed on CPU float32; the frozen checkpoints are "
            "embedded in the manifest by SHA-256, and Validation numbers come from the training-time GPU "
            "forward of the same checkpoints",
        },
        "scores": scores,
        "result_label": C.RESULT_LABEL,
        "test_exposure": "EXPOSED_TEST_EXPLORATORY",
        "post_test_actions": "none: no re-selection, no re-ranking, no retraining, no new seed",
    }
    C.write_json(out_path, payload)
    _ledger_append(events)
    for name, entry in scores.items():
        C.log(
            f"TEST {name}: roc_auc={entry['metrics']['roc_auc']:.6f} ap={entry['metrics']['pr_auc']:.6f} "
            f"ogb={entry['official_ogb']['ogb_rocauc']:.6f}"
        )
    C.log(f"Test scored once for {len(scores)} models; ledger appended ({len(events)} events)")
    return 0


def test_cache_binary(test_feats: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(test_feats["binary"], dtype=np.uint8)


def _predict(model, cache, feats, device) -> np.ndarray:
    import si_train as TRAIN

    return TRAIN.predict(model, cache, feats, device)


if __name__ == "__main__":
    raise SystemExit(main())
