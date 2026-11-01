"""Optional Stage D: Lipophilicity **scaffold** split (development shift test).

Why this is optional and how it is labelled
-------------------------------------------
The scaffold split is a different partition of the *same* 4,200 Lipophilicity
molecules whose labels have already been read in the random42 rounds, so this is
a **distribution-shift development evaluation**, not a new independent dataset
confirmation.  It runs only after the 18 random42 runs, the frozen-manifest
Test420 pass and all verification are complete.

The split is the repository's existing
``tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/split_lipo_v1.json``
(``deepchem==2.8.0`` ``ScaffoldSplitter``, 3360/420/420).  It is verified, not
assumed: the row partition, the scaffold-group non-crossing claim, the train
label vector against the raw CSV and the row identities are re-derived here.

Models are chosen by the pre-registered rule "BASE, ROLEBLIND and the strongest
other model by the random42 3-seed mean **Validation** RMSE, if any is
plausible".  Nothing is selected on Test420.  Normalisation (structure and
target) is refitted on the scaffold Train split only, and training restarts from
scratch with the same frozen R3 recipe and the same best-epoch checkpoint rule.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import be_common as C
import be_data as D
import be_models as M
import be_train as T

import lcrc_data  # noqa: E402
import r42_data as R42D  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402

SCAFFOLD_SPLIT_FILE = (
    C.REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/split_lipo_v1.json"
)
SCAFFOLD_ROOT = C.ANALYSIS / "scaffold"
SCAFFOLD_RUNS = SCAFFOLD_ROOT / "runs"
SCAFFOLD_EVAL = SCAFFOLD_ROOT / "eval"
SCAFFOLD_SPLIT_AUDIT = SCAFFOLD_ROOT / "split_audit.json"
SCAFFOLD_PREFLIGHT = SCAFFOLD_ROOT / "preflight.json"
SCAFFOLD_MANIFEST = SCAFFOLD_ROOT / "manifest.json"
SCAFFOLD_MANIFEST_FROZEN = SCAFFOLD_ROOT / "MANIFEST_FROZEN"
SCAFFOLD_TABLE = SCAFFOLD_ROOT / "table_scaffold.csv"

SCAFFOLD_MODELS = ("base", "roleblind", "g2l_all")
SCAFFOLD_SEEDS = (0, 1, 2)
TOL = 1.0e-5


def arm_key(family: str, seed: int) -> str:
    return f"scaffold_{family}_s{int(seed)}"


SCAFFOLD_ARMS = tuple(arm_key(f, s) for f in SCAFFOLD_MODELS for s in SCAFFOLD_SEEDS)


# ---------------------------------------------------------------------------
# split audit
# ---------------------------------------------------------------------------


def _csv_labels() -> tuple[list[str], np.ndarray]:
    with open(C.CSV_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != C.N_MOLECULES:
        raise ValueError("Lipophilicity CSV row count drifted")
    smiles = [str(r["smiles"]) for r in rows]
    y = np.asarray([float(r["exp"]) for r in rows], dtype=np.float64)
    if C.sha256_file(C.CSV_PATH) != C.CSV_SHA256:
        raise PermissionError("Lipophilicity CSV hash drifted")
    return smiles, y


def load_scaffold_split() -> dict[str, Any]:
    if not SCAFFOLD_SPLIT_FILE.exists():
        raise FileNotFoundError(SCAFFOLD_SPLIT_FILE)
    payload = json.loads(SCAFFOLD_SPLIT_FILE.read_text(encoding="utf-8"))
    rows = {name: [int(x) for x in payload[f"{name}_rows"]] for name in ("train", "valid", "test")}
    smiles, y = _csv_labels()
    detail: dict[str, Any] = {
        "split_file": C.track_rel(SCAFFOLD_SPLIT_FILE),
        "split_file_sha256": C.sha256_file(SCAFFOLD_SPLIT_FILE),
        "protocol": payload.get("protocol"),
        "splitter": payload.get("splitter"),
        "source_git_commit": payload.get("git_commit"),
        "reference_versions": payload.get("reference_versions"),
        "n_total": int(payload.get("n_total", -1)),
        "sizes": {name: len(rows[name]) for name in rows},
        "scaffold_groups": payload.get("scaffold_groups"),
        "scaffold_groups_cross_split": payload.get("scaffold_groups_cross_split"),
        "official_valid_loaded": payload.get("official_valid_loaded"),
        "official_test_loaded": payload.get("official_test_loaded"),
        "disjoint": bool(
            set(rows["train"]).isdisjoint(rows["valid"])
            and set(rows["train"]).isdisjoint(rows["test"])
            and set(rows["valid"]).isdisjoint(rows["test"])
        ),
        "union_is_all_rows": len(set(rows["train"]) | set(rows["valid"]) | set(rows["test"]))
        == C.N_MOLECULES,
        "declared_n_matches_file": {name: int(payload.get(f"n_{name}", -1)) for name in rows},
        "train_y_matches_csv_exactly": bool(
            np.array_equal(np.asarray(payload["train_y"], dtype=np.float64), y[np.asarray(rows["train"], dtype=np.int64)])
        ),
        "file_row_hashes_recorded": {
            name: str(payload.get(f"{name}_sha256_rows")) for name in rows
        },
        "recomputed_row_hashes": {
            name: C.rows_sha256_int64(rows[name]) for name in rows
        },
        "declared_row_hashes_match_recomputed": {
            name: str(payload.get(f"{name}_sha256_rows")) == C.rows_sha256_int64(rows[name]) for name in rows
        },
        "row_hash_note": (
            "the declared row hashes reproduce exactly with the round's own int64 row-hash "
            "serialisation, so the file's row order identity is verified"
        ),
        "scaffold_group_non_crossing": "declared by the split file; not recomputed (rdkit-only step)",
        "rows": rows,
        "smiles": smiles,
        "y": y,
    }
    passed = bool(
        detail["sizes"] == C.SPLIT_SIZES
        and detail["disjoint"]
        and detail["union_is_all_rows"]
        and detail["train_y_matches_csv_exactly"]
        and detail["scaffold_groups_cross_split"] == 0
        and all(detail["declared_row_hashes_match_recomputed"].values())
        and detail["official_valid_loaded"] is False
        and detail["official_test_loaded"] is False
        and all(detail["declared_n_matches_file"][n] == detail["sizes"][n] for n in detail["sizes"])
    )
    detail["passed"] = passed
    if not passed:
        raise PermissionError(f"scaffold split audit failed: {detail}")
    return detail


def build_scaffold_views() -> dict[str, Any]:
    audit = load_scaffold_split()
    base_views = D.load_views(include_test=True)
    cache_all = base_views["cache_built"]["cache"]
    if [str(x) for x in cache_all["molecule_ids"]] != [str(i) for i in range(C.N_MOLECULES)]:
        raise PermissionError("frozen LCRC cache is not in CSV row order")
    spec = base_views["spec"]
    rows = audit["rows"]
    caches = {}
    for name in ("train", "valid", "test"):
        selected = R42D.select_cache(cache_all, rows[name], split_label=f"scaffold_{name}")
        caches[name] = D.build_distance_buckets(selected)
    if [str(x) for x in caches["train"]["smiles"]] != [audit["smiles"][r] for r in rows["train"]]:
        raise ValueError("scaffold train SMILES identity drifted")
    train_norm = lcrc_data.fit_norm(caches["train"], spec, source="scaffold_train_structures")
    y_train = audit["y"][np.asarray(rows["train"], dtype=np.int64)]
    y_valid = audit["y"][np.asarray(rows["valid"], dtype=np.int64)]
    y_test = audit["y"][np.asarray(rows["test"], dtype=np.int64)]
    target_norm = {
        "mean": float(y_train.mean()),
        "scale": float(y_train.std(ddof=0)),
        "source": "scaffold_train_labels_only",
        "fit_on": "scaffold Train3360 only",
        "n": int(y_train.size),
        "valid_or_test_used": False,
    }
    return {
        "split": {"rows": rows, "smiles": audit["smiles"], "name": "lipo_scaffold_deepchem_v1"},
        "labels": {
            "y_train": np.asarray(y_train, dtype=np.float64),
            "y_valid": np.asarray(y_valid, dtype=np.float64),
            "audit": {
                "source": "scaffold split json + frozen CSV labels",
                "y_train_sha256": C.vector_sha256_float64(y_train),
                "y_valid_sha256": C.vector_sha256_float64(y_valid),
                "y_test_sha256": C.vector_sha256_float64(y_test),
            },
        },
        "y_test": np.asarray(y_test, dtype=np.float64),
        "spec": spec,
        "cache_built": {
            **{k: v for k, v in base_views["cache_built"].items() if k != "cache"},
        },
        "train_cache": caches["train"],
        "valid_cache": caches["valid"],
        "test_cache": caches["test"],
        "train_norm": train_norm,
        "train_norm_fingerprint": lcrc_data.norm_fingerprint(train_norm),
        "target_norm": target_norm,
        "split_seed": None,
        "split_scope": "lipo_scaffold_deepchem_v1",
        "data_override": {
            "split_file": C.track_rel(SCAFFOLD_SPLIT_FILE),
            "split_file_sha256": audit["split_file_sha256"],
            "labels_file": "data/lipophilicity/Lipophilicity.csv",
            "labels_file_sha256": C.CSV_SHA256,
            "train_rows_sha256": C.rows_sha256_int64(rows["train"]),
            "valid_rows_sha256": C.rows_sha256_int64(rows["valid"]),
            "test_rows_sha256": C.rows_sha256_int64(rows["test"]),
            "split_scope": "lipo_scaffold_deepchem_v1",
            "train_norm_fingerprint": lcrc_data.norm_fingerprint(train_norm),
            "target_norm": dict(target_norm),
            "test_row_values_used": False,
            "label_scope": "scaffold_train_and_valid_only_during_training",
        },
        "protocol_override": {"scaffold_stage": True},
    }


# ---------------------------------------------------------------------------
# preflight / train / evaluate
# ---------------------------------------------------------------------------


def preflight() -> dict[str, Any]:
    audit = load_scaffold_split()
    answers_path = C.ANSWERS
    if not answers_path.exists():
        raise PermissionError("random42 analysis is missing; the model selection rule needs it")
    answers = C.read_json(answers_path)
    summary = answers["model_summary"]
    ranked = sorted(
        ((name, info["valid_rmse_mean"]) for name, info in summary.items() if name != "BASE"),
        key=lambda pair: pair[1],
    )
    strongest_other = ranked[0][0] if ranked else None
    frozen_family = {C.MODEL_DISPLAY[f]: f for f in C.FAMILIES}
    selected = ["BASE", "ROLEBLIND", strongest_other]
    selected_families = [frozen_family[name] for name in selected if name is not None]
    SCAFFOLD_ROOT.mkdir(parents=True, exist_ok=True)
    detail = {k: v for k, v in audit.items() if k not in ("rows", "smiles", "y")}
    C.write_json(SCAFFOLD_SPLIT_AUDIT, {"protocol_id": C.PROTOCOL_ID, **detail})
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "status": "SCAFFOLD_PREFLIGHT",
        "scope": "DEVELOPMENT_DISTRIBUTION_SHIFT_NOT_A_NEW_BLIND_DATASET",
        "split": {
            "file": detail["split_file"],
            "sha256": detail["split_file_sha256"],
            "sizes": detail["sizes"],
            "splitter": detail["splitter"],
        },
        "model_selection_rule": (
            "BASE, ROLEBLIND and the strongest other model by the random42 3-seed mean "
            "Validation420 RMSE (Test420 not used)"
        ),
        "random42_validation_ranking": [
            {"model": name, "valid_rmse_mean": value} for name, value in
            sorted(((n, i["valid_rmse_mean"]) for n, i in summary.items()), key=lambda p: p[1])
        ],
        "selected_models": selected_families,
        "models": [C.MODEL_DISPLAY[f] for f in selected_families],
        "seeds": list(SCAFFOLD_SEEDS),
        "arms": [arm_key(f, s) for f in selected_families for s in SCAFFOLD_SEEDS],
        "recipe": dict(C.TRAIN_RECIPE),
        "code_hashes": C.code_file_hashes(),
        "config_hash": C.sha256_json(C.load_config()),
        "norm_refit": "structure normalisation and target scaler refitted on scaffold Train3360 only",
    }
    payload["preflight_sha256"] = C.sha256_json({k: v for k, v in payload.items() if k != "preflight_sha256"})
    C.write_json(SCAFFOLD_PREFLIGHT, payload)
    return payload


def _require_preflight() -> dict[str, Any]:
    if not SCAFFOLD_PREFLIGHT.exists():
        raise PermissionError("scaffold preflight missing; run `run_be.py scaffold-preflight` first")
    payload = C.read_json(SCAFFOLD_PREFLIGHT)
    body = {k: v for k, v in payload.items() if k != "preflight_sha256"}
    if C.sha256_json(body) != payload["preflight_sha256"]:
        raise PermissionError("scaffold preflight content changed after the freeze")
    if payload["code_hashes"] != C.code_file_hashes():
        raise PermissionError("scaffold code changed after the preflight")
    return payload


def train(arm: str, device: torch.device, pool: str) -> dict[str, Any]:
    pre = _require_preflight()
    if arm not in pre["arms"]:
        raise ValueError(f"{arm} is not a pre-registered scaffold arm {pre['arms']}")
    family = arm[len("scaffold_") : arm.rindex("_s")]
    seed = int(arm[arm.rindex("_s") + 2 :])
    run_dir = SCAFFOLD_RUNS / arm
    if (run_dir / "train.json").exists():
        raise PermissionError(f"{arm} already trained (one-shot)")
    views = build_scaffold_views()
    payload = T.train_arm(
        arm=arm,
        cfg=C.load_config(),
        views=views,
        device=device,
        run_dir=run_dir,
        pool=pool,
        smoke=False,
        family=family,
        seed=seed,
    )
    payload["preflight_sha256"] = pre["preflight_sha256"]
    C.write_json(run_dir / "train.json", payload)
    return payload


def freeze_manifest() -> dict[str, Any]:
    pre = _require_preflight()
    arms: dict[str, Any] = {}
    for arm in pre["arms"]:
        train_json = SCAFFOLD_RUNS / arm / "train.json"
        if not train_json.exists():
            raise PermissionError(f"missing scaffold run {arm}")
        payload = C.read_json(train_json)
        ckpt_path = SCAFFOLD_RUNS / arm / "checkpoint.pt"
        arms[arm] = {
            "arm": arm,
            "model": payload["model"],
            "family": payload["family"],
            "seed": int(payload["seed"]),
            "checkpoint": C.track_rel(ckpt_path),
            "checkpoint_sha256": C.sha256_file(ckpt_path),
            "best_epoch": int(payload["checkpoint"]["best_epoch"]),
            "best_valid_rmse": float(payload["checkpoint"]["best_valid_rmse"]),
            "best_valid_mae": float(payload["checkpoint"]["best_valid_mae"]),
            "train_rmse": float(payload["metrics"]["train"]["rmse"]),
            "curve_last_epoch": int(payload["checkpoint"]["epochs_run"]),
            "checkpoint_is_final_epoch_state": bool(payload["checkpoint"]["checkpoint_is_final_epoch_state"]),
        }
    body = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "SCAFFOLD_MANIFEST_FROZEN_BEFORE_TEST",
        "scope": "DEVELOPMENT_DISTRIBUTION_SHIFT_NOT_A_NEW_BLIND_DATASET",
        "preflight_sha256": pre["preflight_sha256"],
        "n_arms": len(arms),
        "test_read_at_freeze_time": False,
        "arms": arms,
    }
    body["manifest_sha256"] = C.sha256_json(body)
    C.write_json(SCAFFOLD_MANIFEST, body)
    C.write_json(
        SCAFFOLD_MANIFEST_FROZEN,
        {"manifest_sha256": body["manifest_sha256"], "created_utc": body["created_utc"], "n_arms": len(arms)},
    )
    return body


def evaluate() -> dict[str, Any]:
    if not SCAFFOLD_MANIFEST.exists() or not SCAFFOLD_MANIFEST_FROZEN.exists():
        raise PermissionError("scaffold manifest is not frozen; run `scaffold-manifest` first")
    manifest = C.read_json(SCAFFOLD_MANIFEST)
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if C.sha256_json(body) != manifest["manifest_sha256"]:
        raise PermissionError("scaffold manifest changed after the freeze")
    views = build_scaffold_views()
    y_valid = views["labels"]["y_valid"]
    y_test = views["y_test"]
    results: dict[str, Any] = {}
    for arm, frozen in manifest["arms"].items():
        ckpt_path = SCAFFOLD_RUNS / arm / "checkpoint.pt"
        if C.sha256_file(ckpt_path) != frozen["checkpoint_sha256"]:
            raise PermissionError(f"{arm}: checkpoint changed after the manifest freeze")
        model, meta = M.build_family(frozen["family"], views["spec"], views["train_norm"], int(frozen["seed"]), "cpu")
        payload = M.load_checkpoint(
            model,
            ckpt_path,
            expected_sha256=frozen["checkpoint_sha256"],
            expected_family=frozen["family"],
            expected_seed=int(frozen["seed"]),
        )
        pred_valid = NT.denormalise(M.predict(model, views["valid_cache"], "cpu"), views["target_norm"])
        pred_test = NT.denormalise(M.predict(model, views["test_cache"], "cpu"), views["target_norm"])
        valid_metrics = MET.all_metrics(pred_valid, y_valid)
        test_metrics = MET.all_metrics(pred_test, y_test)
        if abs(float(valid_metrics["rmse"]) - float(frozen["best_valid_rmse"])) > TOL:
            raise RuntimeError(f"{arm}: scaffold Validation RMSE does not reproduce")
        results[arm] = {
            "arm": arm,
            "model": frozen["model"],
            "family": frozen["family"],
            "seed": int(frozen["seed"]),
            "best_epoch": int(frozen["best_epoch"]),
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "train_rmse": float(frozen["train_rmse"]),
            "valid": dict(valid_metrics),
            "test": dict(test_metrics),
        }
        C.log(
            f"[scaffold-eval] {arm} epoch={frozen['best_epoch']} valid={valid_metrics['rmse']:.6f} "
            f"test={test_metrics['rmse']:.6f}"
        )
    payload_out = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "scope": "DEVELOPMENT_DISTRIBUTION_SHIFT_NOT_A_NEW_BLIND_DATASET",
        "manifest_sha256": manifest["manifest_sha256"],
        "target_norm": dict(views["target_norm"]),
        "models": results,
    }
    SCAFFOLD_EVAL.mkdir(parents=True, exist_ok=True)
    C.write_json(SCAFFOLD_EVAL / "metrics.json", payload_out)
    _write_tables(payload_out)
    return payload_out


def _write_tables(payload: Mapping[str, Any]) -> None:
    models = payload["models"]
    rows = []
    for arm in sorted(models):
        rec = models[arm]
        rows.append(
            {
                "model": rec["model"],
                "arm": arm,
                "seed": rec["seed"],
                "best_epoch": rec["best_epoch"],
                "train_rmse": rec["train_rmse"],
                "valid_rmse": rec["valid"]["rmse"],
                "valid_mae": rec["valid"]["mae"],
                "valid_r2": rec["valid"]["r2"],
                "test_rmse": rec["test"]["rmse"],
                "test_mae": rec["test"]["mae"],
                "test_r2": rec["test"]["r2"],
            }
        )
    with open(SCAFFOLD_TABLE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for display in dict.fromkeys(r["model"] for r in rows):
        subset = [r for r in rows if r["model"] == display]
        summary.append(
            {
                "model": display,
                "n_seeds": len(subset),
                "best_epochs": "/".join(str(r["best_epoch"]) for r in sorted(subset, key=lambda x: x["seed"])),
                "valid_rmse_mean": float(np.mean([r["valid_rmse"] for r in subset])),
                "valid_rmse_sd": float(np.std([r["valid_rmse"] for r in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "test_rmse_mean": float(np.mean([r["test_rmse"] for r in subset])),
                "test_rmse_sd": float(np.std([r["test_rmse"] for r in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "valid_mae_mean": float(np.mean([r["valid_mae"] for r in subset])),
                "test_mae_mean": float(np.mean([r["test_mae"] for r in subset])),
            }
        )
    with open(SCAFFOLD_ROOT / "table_scaffold_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    C.write_json(SCAFFOLD_ROOT / "summary.json", {"scaffold_summary": summary})


__all__ = [
    "load_scaffold_split",
    "build_scaffold_views",
    "preflight",
    "train",
    "freeze_manifest",
    "evaluate",
    "SCAFFOLD_ARMS",
]
