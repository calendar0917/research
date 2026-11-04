"""Control-plane runner for ``tsb-lipo-feature-ladder-v1`` (CPU only).

Stages
------
``features``  build the label-free feature blocks and the three Train
              vocabularies, write ``analysis/p0/*``
``p0``        run every CPU correctness gate and freeze the feature spec
``p1``        the 15 full-Train models (3 splits x 5 arms, N=3,360)
``p2``        the 30 small-sample models (3 splits x 5 arms x {1,000, 2,000})
``freeze``    hash-freeze the 15 full-Train models *before* any Test label is read
``test``      one-shot Test scoring of exactly those 15 models
``report``    assemble the four report tables from recorded metrics

The runner refuses to touch Test labels in any stage except ``test``, and the
``test`` stage requires the control plane to have granted terminal access
(``context.test_access == "granted"``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import tsb_common as C
import tsb_data as D
import tsb_features as F
import tsb_model as M

from ksvd_research.runner_api import Runner, RunnerError
from ksvd_research.runtime.manifest import RunResult

NAME = "tsb_lipo_feature_ladder_v1"
CONFIG_PATH = C.CONFIGS / f"{C.PROTOCOL_ID}.yaml"
STUDY = "tsb-lipo-feature-ladder-v1"

P0_DIR = C.ANALYSIS / "p0"
FITS_DIR = C.ANALYSIS / "fits"
MODELS_DIR = C.ANALYSIS / "models"
PRED_DIR = C.ANALYSIS / "predictions"
FREEZE_DIR = C.ANALYSIS / "freeze"
TEST_DIR = C.ANALYSIS / "test"
TEST_PRED_DIR = TEST_DIR / "predictions"
TABLES_DIR = C.ANALYSIS / "tables"
FREEZE_PATH = FREEZE_DIR / "pre_test_freeze.json"
FREEZE_POINTER = FREEZE_DIR / "freeze_pointer.json"


# ---------------------------------------------------------------------------
# frozen-config enforcement
# ---------------------------------------------------------------------------


def _frozen_config(config: dict[str, Any]) -> dict[str, Any]:
    """Assert that the YAML config still encodes the frozen protocol."""
    required = {
        "protocol_id": C.PROTOCOL_ID,
        "seed": C.SEED,
        "arms": list(C.ARMS),
        "splits": list(C.SPLIT_SUFFIXES),
        "sample_sizes": list(C.SAMPLE_SIZES),
        "wl_vocab_size": C.WL_VOCAB_SIZE,
        "wl_atom_attributes": list(C.WL_ATOM_ATTRIBUTES),
        "wl_bond_attributes": list(C.WL_BOND_ATTRIBUTES),
        "morgan": {
            "radius": C.MORGAN_RADIUS,
            "n_bits": C.MORGAN_NBITS,
            "use_counts": C.MORGAN_USE_COUNTS,
        },
        "xgboost": dict(C.XGB_PARAMS),
        "bootstrap": {"replicates": C.BOOTSTRAP_REPLICATES, "seed": C.BOOTSTRAP_SEED},
    }
    for key, expected in required.items():
        if key not in config:
            raise RunnerError(f"config is missing the frozen key {key!r}")
        if config[key] != expected:
            raise RunnerError(
                f"config key {key!r} drifted from the frozen protocol:\n"
                f"  config   = {config[key]}\n  protocol = {expected}"
            )
    return config


OPERATIONAL_KEYS = ("stage", "rebuild_features")


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    """Config without the operational stage selector.

    ``stage`` says *which phase* runs; it is not part of the frozen scientific
    configuration, so freeze and Test must compare the same scientific config
    even though they are different stages.
    """
    return {key: value for key, value in config.items() if key not in OPERATIONAL_KEYS}


def code_hashes() -> dict[str, str]:
    files = sorted(C.CODE.glob("*.py")) + [CONFIG_PATH]
    return {C.track_rel(path): C.sha256_file(path) for path in files if path.is_file()}


def environment() -> dict[str, Any]:
    import platform
    import sys

    import pandas
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "xgboost": M.version(),
        "rdkit": _rdkit_version(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty_tracked(),
    }


def _rdkit_version() -> str:
    import rdkit

    return str(rdkit.__version__)


def fit_key(suffix: int, arm: str, n_train: int) -> str:
    return f"s{int(suffix)}__{arm}__n{int(n_train)}"


def _fit_path(key: str) -> Path:
    return FITS_DIR / f"{key}.json"


def _model_path(key: str) -> Path:
    return MODELS_DIR / f"{key}.ubj"


def _pred_paths(key: str) -> dict[str, Path]:
    return {
        "train": PRED_DIR / f"{key}__train.npz",
        "valid": PRED_DIR / f"{key}__valid.npz",
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_features(config: dict[str, Any], context) -> RunResult:
    if context.test_access == "granted":
        raise RunnerError("the features stage must never run with Test access granted")
    started = time.perf_counter()
    manifest = F.build(rebuild=bool(config.get("rebuild_features", False)))
    spec = build_feature_spec(manifest)
    C.write_json(P0_DIR / "features_manifest.json", manifest)
    C.write_json(C.CONFIGS / "feature_spec.json", spec)
    C.write_json(P0_DIR / "feature_spec.json", spec)
    seconds = time.perf_counter() - started
    metrics = {
        "n_molecules": manifest["n_molecules"],
        "n_parse_failed": manifest["n_parse_failed"],
        "g0_dim": manifest["g0"]["dim"],
        "tokenization_seconds": manifest["timing"]["tokenization_seconds"],
        "feature_stage_seconds": seconds,
        "peak_rss_mb": C.peak_rss_mb(),
        "test_labels_read": False,
    }
    for arm, arm_meta in spec["arms"].items():
        metrics[f"dim_{arm}"] = arm_meta["n_columns"]
    return RunResult(
        metrics=metrics,
        artifacts=[
            C.track_rel(P0_DIR / "features_manifest.json"),
            C.track_rel(C.CONFIGS / "feature_spec.json"),
            C.track_rel(F.MANIFEST_PATH),
        ],
    )


def build_feature_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    """Frozen, data-dependent column definition of all five arms.

    WL column names embed the token hash, so they are split-specific (each
    published split owns its Train-only vocabulary).  The spec therefore stores
    the arm shape once and one column-list hash per (arm, split).
    """
    import tsb_g0
    import tsb_morgan
    import tsb_wl

    arms: dict[str, Any] = {}
    for arm in C.ARMS:
        blocks = F.arm_blocks(arm)
        block_meta: list[dict[str, Any]] = []
        for kind, name in blocks:
            if kind == "g0":
                names = list(tsb_g0.COLUMN_NAMES)
            elif kind == "morgan":
                names = tsb_morgan.column_names()
            else:
                round_index = int(name.split(":")[0].lstrip("r"))
                names = tsb_wl.column_names([], round_index)
            block_meta.append(
                {
                    "kind": kind,
                    "name": name,
                    "dim": len(names),
                    "column_name_pattern": _column_pattern(kind, name),
                }
            )
        per_split: dict[str, Any] = {}
        for suffix in C.SPLIT_SUFFIXES:
            _matrix, columns = F.arm_matrix(suffix, arm)
            per_split[str(suffix)] = {
                "n_columns": len(columns),
                "column_list_sha256": C.sha256_json(columns),
            }
        arms[arm] = {
            "id": arm,
            "n_columns_by_split": {
                suffix: entry["n_columns"] for suffix, entry in per_split.items()
            },
            "n_columns": max(entry["n_columns"] for entry in per_split.values()),
            "blocks": block_meta,
            "per_split": per_split,
            "label_used": False,
            "per_split_dimension_note": (
                "round-0 vocabulary is min(512, distinct Train atom signatures) = 48-49 tokens, "
                "so arm width can differ by a few columns across published splits; within one "
                "split all arms and every sample size share exactly the same columns"
            ),
        }
    return {
        "schema": "tsb-feature-spec-v1",
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "note": (
            "WL column names are per published split because each split owns its Train-only "
            "vocabulary; the vocabulary files under cache/wl/ are hashed in the feature manifest"
        ),
        "graph_contract": manifest["graph_contract"],
        "g0": manifest["g0"]["spec"],
        "morgan": manifest["morgan"]["spec"],
        "wl": {
            "rounds": {name: list(rounds) for name, rounds in C.WL_ROUNDS.items()},
            "vocab_size_per_round": C.WL_VOCAB_SIZE,
            "oov_column_per_round": True,
            "dim_per_round": C.WL_VOCAB_SIZE + 1,
            "atom_attributes": list(C.WL_ATOM_ATTRIBUTES),
            "bond_attributes": list(C.WL_BOND_ATTRIBUTES),
            "vocabulary_manifest": {
                str(suffix): {
                    f"r{round_index}": {
                        "vocab_sha256": entry["vocab_sha256"],
                        "n_tokens": int(entry["dim"]) - 1,
                    }
                    for round_index, entry in payload.items()
                }
                for suffix, payload in manifest["wl"].items()
            },
        },
        "arms": arms,
    }


def _column_pattern(kind: str, name: str) -> str:
    if kind == "g0":
        return "g0_<g0_column>"
    if kind == "morgan":
        return "morgan_r2_b<AAAA>"
    round_index = int(name.split(":")[0].lstrip("r"))
    return f"wl{round_index}_t<RRR>_<token_sha256[:12]> | wl{round_index}_oov"


def stage_p0(config: dict[str, Any], context) -> RunResult:
    if context.test_access == "granted":
        raise RunnerError("the P0 gate stage must never run with Test access granted")
    import tsb_gates

    started = time.perf_counter()
    manifest = F.load_manifest()
    spec_path = C.CONFIGS / "feature_spec.json"
    if not spec_path.is_file():
        C.write_json(spec_path, build_feature_spec(manifest))
    spec = C.read_json(spec_path)
    for suffix in C.SPLIT_SUFFIXES:
        for arm in C.ARMS:
            _matrix, names = F.arm_matrix(suffix, arm)
            frozen = spec["arms"][arm]["per_split"][str(suffix)]
            if len(names) != frozen["n_columns"]:
                raise RunnerError(
                    f"{arm}/{suffix}: live dim {len(names)} != frozen {frozen['n_columns']}"
                )
            if C.sha256_json(names) != frozen["column_list_sha256"]:
                raise RunnerError(f"{arm}/{suffix}: live column list hash != frozen feature spec")
        dims = {arm: spec["arms"][arm]["per_split"][str(suffix)]["n_columns"] for arm in C.ARMS}
        if dims["WL2_BIN"] != dims["WL2_COUNT"]:
            raise RunnerError(f"{suffix}: WL2_BIN dim {dims['WL2_BIN']} != WL2_COUNT dim {dims['WL2_COUNT']}")
        round3_dim = int(manifest["wl"][str(suffix)]["r3"]["dim"])
        if dims["WL3_COUNT"] != dims["WL2_COUNT"] + round3_dim:
            raise RunnerError(f"{suffix}: WL3 does not equal WL2 + round-3 block")
        if dims["MORGAN_COUNT"] != dims["G0"] + C.MORGAN_NBITS:
            raise RunnerError(f"{suffix}: MORGAN_COUNT dim != G0 + {C.MORGAN_NBITS}")
    gates = tsb_gates.run_all(include_fit_gates=bool(list(FITS_DIR.glob("*.json"))))
    payload = {
        "schema": "tsb-p0-gates-v1",
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "data_manifest": D.data_manifest(),
        "feature_manifest_sha256": C.sha256_file(F.MANIFEST_PATH),
        "feature_spec_sha256": C.sha256_file(spec_path),
        "gates": gates,
        "all_passed": all(bool(gate["passed"]) for gate in gates),
        "test_labels_read": False,
        "seconds": time.perf_counter() - started,
    }
    C.write_json(P0_DIR / "p0_gates.json", payload)
    if not payload["all_passed"]:
        failed = [gate["name"] for gate in gates if not gate["passed"]]
        raise RunnerError(f"P0 correctness gates failed: {failed}")
    return RunResult(
        metrics={
            "n_gates": len(gates),
            "gates_passed": sum(1 for gate in gates if gate["passed"]),
            "synthetic_wl_boundary_passed": next(
                gate["passed"] for gate in gates if gate["name"].startswith("11_")
            ),
            "peak_rss_mb": C.peak_rss_mb(),
            "seconds": payload["seconds"],
            "test_labels_read": False,
        },
        artifacts=[
            C.track_rel(P0_DIR / "p0_gates.json"),
            C.track_rel(C.CONFIGS / "feature_spec.json"),
        ],
    )


def _fit_one(
    *,
    suffix: int,
    arm: str,
    n_train: int,
    matrix,
    columns: list[str],
    feature_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    key = fit_key(suffix, arm, n_train)
    existing = _fit_path(key)
    if existing.is_file():
        record = C.read_json(existing)
        if record.get("config_sha256") != C.sha256_json(config):
            raise RunnerError(f"{key}: existing fit was produced by a different config")
        return record

    split = D.split(suffix)
    labels = D.split_labels(suffix)
    train_rows = D.nested_prefix(split["rows"]["train"], n_train)
    train_index = _index_of(split["rows"]["train"], train_rows)
    valid_rows = np.asarray(split["rows"]["valid"], dtype=np.int64)
    y_train = np.asarray(labels["train"], dtype=np.float64)[train_index]
    y_valid = np.asarray(labels["valid"], dtype=np.float64)

    X_train = matrix[train_rows]
    X_valid = matrix[valid_rows]
    model, fit_seconds = M.fit(X_train, y_train, config["xgboost"])
    train_pred = M.predict(model, X_train)
    valid_pred = M.predict(model, X_valid)
    model_info = M.save(model, _model_path(key))

    pred_paths = _pred_paths(key)
    C.ensure_dir(PRED_DIR)
    np.savez(
        pred_paths["train"],
        rows=train_rows,
        pred=train_pred,
        y_true=y_train,
    )
    np.savez(
        pred_paths["valid"],
        rows=valid_rows,
        pred=valid_pred,
        y_true=y_valid,
    )

    train_metrics = C.regression_metrics(train_pred, y_train)
    valid_metrics = C.regression_metrics(valid_pred, y_valid)
    record = {
        "schema": "tsb-fit-v1",
        "protocol_id": C.PROTOCOL_ID,
        "key": key,
        "split_suffix": int(suffix),
        "split_file": split["file"],
        "split_file_sha256": split["file_sha256"],
        "arm": arm,
        "n_train": int(n_train),
        "seed": C.SEED,
        "config_sha256": C.sha256_json(config),
        "xgboost_params": dict(config["xgboost"]),
        "xgboost_params_sha256": M.params_sha256(),
        "n_columns": int(matrix.shape[1]),
        "column_list_sha256": C.sha256_json(columns),
        "feature_assembly_seconds": float(feature_seconds),
        "fit_seconds": float(fit_seconds),
        "train_rows_sha256": C.row_sha256(train_rows),
        "valid_rows_sha256": C.row_sha256(valid_rows),
        "test_rows_sha256": split["rows_sha256"]["test"],
        "y_train_sha256": C.sha256_bytes(np.ascontiguousarray(y_train).tobytes()),
        "y_valid_sha256": C.sha256_bytes(np.ascontiguousarray(y_valid).tobytes()),
        "model": model_info,
        "predictions": {
            role: {
                "path": C.track_rel(path),
                "sha256": C.sha256_file(path),
            }
            for role, path in pred_paths.items()
        },
        "metrics": {"train": train_metrics, "valid": valid_metrics},
        "generalisation_gap_rmse": valid_metrics["rmse"] - train_metrics["rmse"],
        "test_scored": False,
        "test_labels_read": False,
        "environment": environment(),
        "saved_utc": C.utcnow(),
    }
    C.write_json(existing, record)
    return record


def _index_of(train_rows: np.ndarray, subset: np.ndarray) -> np.ndarray:
    lookup = {int(row): i for i, row in enumerate(train_rows)}
    return np.asarray([lookup[int(row)] for row in subset], dtype=np.int64)


def _run_fits(
    suffix: int,
    arm: str,
    sizes: Sequence[int],
    config: dict[str, Any],
    cache: dict[tuple[int, str], tuple[Any, list[str], float]],
) -> list[dict[str, Any]]:
    assembled = cache.get((suffix, arm))
    if assembled is None:
        t0 = time.perf_counter()
        matrix, columns = F.arm_matrix(suffix, arm)
        seconds = time.perf_counter() - t0
        assembled = (matrix, columns, seconds)
        cache[(suffix, arm)] = assembled
    matrix, columns, seconds = assembled
    return [
        _fit_one(
            suffix=suffix,
            arm=arm,
            n_train=n,
            matrix=matrix,
            columns=columns,
            feature_seconds=seconds,
            config=config,
        )
        for n in sizes
    ]


def _stage_fits(config: dict[str, Any], context, *, stage: str) -> RunResult:
    if context.test_access == "granted":
        raise RunnerError(f"the {stage} stage must never run with Test access granted")
    sizes = [C.FULL_SIZE] if stage == "p1" else [n for n in C.SAMPLE_SIZES if n != C.FULL_SIZE]
    started = time.perf_counter()
    cache: dict[tuple[int, str], tuple[Any, list[str], float]] = {}
    records: list[dict[str, Any]] = []
    for suffix in C.SPLIT_SUFFIXES:
        for arm in C.ARMS:
            records.extend(_run_fits(suffix, arm, sizes, config, cache))
    metrics: dict[str, Any] = {
        "stage": stage,
        "n_fits": len(records),
        "sizes": sizes,
        "fit_seconds_total": float(sum(r["fit_seconds"] for r in records)),
        "stage_seconds": float(time.perf_counter() - started),
        "peak_rss_mb": C.peak_rss_mb(),
        "test_labels_read": False,
    }
    for record in records:
        metrics[f"valid_rmse__{record['key']}"] = record["metrics"]["valid"]["rmse"]
    return RunResult(
        metrics=metrics,
        artifacts=[C.track_rel(path) for path in sorted(FITS_DIR.glob("*.json"))],
    )


def stage_p1(config: dict[str, Any], context) -> RunResult:
    return _stage_fits(config, context, stage="p1")


def stage_p2(config: dict[str, Any], context) -> RunResult:
    return _stage_fits(config, context, stage="p2")


def stage_freeze(config: dict[str, Any], context) -> RunResult:
    if context.test_access == "granted":
        raise RunnerError("the freeze stage must run with Test access blocked")
    if TEST_DIR.exists() and any(TEST_DIR.glob("*.json")) and not (TEST_DIR / "test_summary.json").is_file():
        raise RunnerError("partial Test artefacts exist before the freeze; refusing to freeze")
    started = time.perf_counter()
    entries: dict[str, Any] = {}
    missing: list[str] = []
    for suffix in C.SPLIT_SUFFIXES:
        for arm in C.ARMS:
            key = fit_key(suffix, arm, C.FULL_SIZE)
            path = _fit_path(key)
            if not path.is_file():
                missing.append(key)
                continue
            record = C.read_json(path)
            model_file = C.REPO / record["model"]["path"]
            if C.sha256_file(model_file) != record["model"]["sha256"]:
                raise RunnerError(f"{key}: model file changed since the fit")
            for role, payload in record["predictions"].items():
                if C.sha256_file(C.REPO / payload["path"]) != payload["sha256"]:
                    raise RunnerError(f"{key}: {role} predictions changed since the fit")
            if record.get("test_scored"):
                raise RunnerError(f"{key}: already Test-scored before the freeze")
            entries[key] = {
                "split_suffix": int(suffix),
                "arm": arm,
                "n_train": int(record["n_train"]),
                "config_sha256": record["config_sha256"],
                "xgboost_params_sha256": record["xgboost_params_sha256"],
                "n_columns": int(record["n_columns"]),
                "column_list_sha256": record["column_list_sha256"],
                "train_rows_sha256": record["train_rows_sha256"],
                "model_path": record["model"]["path"],
                "model_sha256": record["model"]["sha256"],
                "valid_predictions": record["predictions"]["valid"],
                "train_predictions": record["predictions"]["train"],
                "metrics_train": record["metrics"]["train"],
                "metrics_valid": record["metrics"]["valid"],
                "fit_seconds": record["fit_seconds"],
                "feature_assembly_seconds": record["feature_assembly_seconds"],
            }
    if missing:
        raise RunnerError(f"cannot freeze: {len(missing)} full-Train fits missing: {missing[:5]}")
    feature_manifest = F.load_manifest()
    payload = {
        "schema": "tsb-pre-test-freeze-v1",
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty_tracked(),
        "config": config,
        "config_sha256": C.sha256_json(config),
        "scientific_config": scientific_config(config),
        "scientific_config_sha256": C.sha256_json(scientific_config(config)),
        "operational_keys_excluded_from_scientific_config": list(OPERATIONAL_KEYS),
        "protocol_snapshot": _protocol_snapshot(context),
        "code_hashes": code_hashes(),
        "environment": environment(),
        "feature_manifest_path": C.track_rel(F.MANIFEST_PATH),
        "feature_manifest_sha256": C.sha256_file(F.MANIFEST_PATH),
        "feature_spec_path": C.track_rel(C.CONFIGS / "feature_spec.json"),
        "feature_spec_sha256": C.sha256_file(C.CONFIGS / "feature_spec.json"),
        "n_models": len(entries),
        "models": entries,
        "test_not_read": True,
        "test_policy": {
            "rule": "all 15 full-Train models, their predictions and this manifest are hashed "
                    "before the first Test label is read; Test scores are reported for every "
                    "pre-registered model, never used for selection",
            "exposure": C.TEST_EXPOSURE_LABEL,
        },
        "seconds": float(time.perf_counter() - started),
    }
    C.write_json(FREEZE_PATH, payload)
    C.ensure_dir(FREEZE_DIR)
    pointer = {
        "schema": "tsb-freeze-pointer-v1",
        "protocol_id": C.PROTOCOL_ID,
        "freeze_path": C.track_rel(FREEZE_PATH),
        "freeze_sha256": C.sha256_file(FREEZE_PATH),
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
    }
    C.write_json(FREEZE_POINTER, pointer)
    return RunResult(
        metrics={
            "n_models_frozen": len(entries),
            "freeze_sha256_prefix": pointer["freeze_sha256"][:16],
            "fit_seconds_total": float(sum(v["fit_seconds"] for v in entries.values())),
            "test_labels_read": False,
        },
        artifacts=[C.track_rel(FREEZE_PATH), C.track_rel(FREEZE_POINTER)],
    )


def _protocol_snapshot(context) -> dict[str, Any]:
    path = Path(context.run_dir) / "protocol.snapshot.yaml"
    if path.is_file():
        return {"path": C.track_rel(path), "sha256": C.sha256_file(path)}
    return {"path": None, "sha256": None}


def stage_test(config: dict[str, Any], context) -> RunResult:
    if context.test_access != "granted":
        raise RunnerError("Test access was not granted by the control plane; refusing to score")
    if not FREEZE_POINTER.is_file():
        raise RunnerError("freeze pointer missing: run the freeze stage first")
    pointer = C.read_json(FREEZE_POINTER)
    if C.sha256_file(FREEZE_PATH) != pointer["freeze_sha256"]:
        raise RunnerError("freeze manifest changed after the freeze")
    freeze = C.read_json(FREEZE_PATH)
    if freeze["scientific_config_sha256"] != C.sha256_json(scientific_config(config)):
        raise RunnerError("the Test run scientific config differs from the frozen config")
    if freeze["code_hashes"] != code_hashes():
        raise RunnerError("code changed between the freeze and the Test stage")
    if freeze["feature_manifest_sha256"] != C.sha256_file(F.MANIFEST_PATH):
        raise RunnerError("feature cache changed between the freeze and the Test stage")
    if freeze["feature_spec_sha256"] != C.sha256_file(C.CONFIGS / "feature_spec.json"):
        raise RunnerError("feature spec changed between the freeze and the Test stage")

    started = time.perf_counter()
    summary_path = TEST_DIR / "test_summary.json"
    entries: dict[str, Any] = {}
    cache: dict[tuple[int, str], tuple[Any, list[str], float]] = {}
    for key, frozen in sorted(freeze["models"].items()):
        suffix = int(frozen["split_suffix"])
        arm = str(frozen["arm"])
        record_path = TEST_DIR / f"{key}.json"
        pred_path = TEST_PRED_DIR / f"{key}__test.npz"
        if record_path.is_file():
            record = C.read_json(record_path)
            if record["model_sha256"] != frozen["model_sha256"]:
                raise RunnerError(f"{key}: recorded Test model hash != frozen hash")
            entries[key] = record
            continue
        assembled = cache.get((suffix, arm))
        if assembled is None:
            matrix, columns = F.arm_matrix(suffix, arm)
            assembled = (matrix, columns, 0.0)
            cache[(suffix, arm)] = assembled
        matrix, columns, _seconds = assembled
        if C.sha256_json(columns) != frozen["column_list_sha256"]:
            raise RunnerError(f"{key}: column definition changed since the freeze")
        split = D.split(suffix)
        test_rows = np.asarray(split["rows"]["test"], dtype=np.int64)
        if C.row_sha256(test_rows) != split["rows_sha256"]["test"]:
            raise RunnerError(f"{key}: Test row ids changed")
        model_path = C.REPO / frozen["model_path"]
        if C.sha256_file(model_path) != frozen["model_sha256"]:
            raise RunnerError(f"{key}: model file changed since the freeze")
        model = M.load(model_path)
        pred = M.predict(model, matrix[test_rows])
        labels = D.split_labels(suffix, include_test=True)
        y_test = np.asarray(labels["test"], dtype=np.float64)
        C.ensure_dir(TEST_PRED_DIR)
        np.savez(pred_path, rows=test_rows, pred=pred, y_true=y_test)
        metrics = C.regression_metrics(pred, y_test)
        record = {
            "schema": "tsb-test-v1",
            "protocol_id": C.PROTOCOL_ID,
            "key": key,
            "split_suffix": suffix,
            "arm": arm,
            "n_test": int(test_rows.size),
            "test_rows_sha256": C.row_sha256(test_rows),
            "model_path": frozen["model_path"],
            "model_sha256": frozen["model_sha256"],
            "freeze_sha256": pointer["freeze_sha256"],
            "predictions": {"path": C.track_rel(pred_path), "sha256": C.sha256_file(pred_path)},
            "metrics": {"train": frozen["metrics_train"], "valid": frozen["metrics_valid"],
                        "test": metrics},
            "exposure": C.TEST_EXPOSURE_LABEL,
            "combined_with_valid_split": False,
            "test_used_for_any_choice": False,
            "scored_utc": C.utcnow(),
        }
        C.write_json(record_path, record)
        entries[key] = record
    if summary_path.is_file():
        raise RunnerError("Test summary already exists: this evaluation is one-shot")
    summary = {
        "schema": "tsb-test-summary-v1",
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "freeze_sha256": pointer["freeze_sha256"],
        "n_models": len(entries),
        "exposure": C.TEST_EXPOSURE_LABEL,
        "exposure_note": (
            "Lipophilicity Test labels were already read by earlier rounds of this repository "
            "(structural-sufficiency Sprint 01 and later repeats); this is not a first blind read"
        ),
        "test_used_for_any_choice": False,
        "models": {
            key: {
                "split_suffix": entry["split_suffix"],
                "arm": entry["arm"],
                "metrics": entry["metrics"],
                "model_sha256": entry["model_sha256"],
                "predictions_sha256": entry["predictions"]["sha256"],
            }
            for key, entry in sorted(entries.items())
        },
        "seconds": float(time.perf_counter() - started),
    }
    C.write_json(summary_path, summary)
    metrics = {"n_models": len(entries), "exposure": C.TEST_EXPOSURE_LABEL,
               "seconds": summary["seconds"]}
    for key, entry in sorted(entries.items()):
        metrics[f"test_rmse__{key}"] = entry["metrics"]["test"]["rmse"]
    return RunResult(
        metrics=metrics,
        artifacts=[C.track_rel(summary_path)] + [
            C.track_rel(TEST_DIR / f"{key}.json") for key in sorted(entries)
        ],
    )


def stage_report(config: dict[str, Any], context) -> RunResult:
    import tsb_tables

    payload = tsb_tables.build_all()
    return RunResult(
        metrics={
            "tables": list(payload["tables"]),
            "n_fits": payload["n_fits"],
            "test_available": payload["test_available"],
        },
        artifacts=[C.track_rel(path) for path in sorted(TABLES_DIR.glob("*"))],
    )


STAGE_HANDLERS = {
    "features": stage_features,
    "p0": stage_p0,
    "p1": stage_p1,
    "p2": stage_p2,
    "freeze": stage_freeze,
    "test": stage_test,
    "report": stage_report,
}


# ---------------------------------------------------------------------------
# runner contract
# ---------------------------------------------------------------------------


def run(config: dict[str, Any], context) -> RunResult:
    _frozen_config(config)
    stage = str(config.get("stage") or "")
    if stage not in STAGE_HANDLERS:
        raise RunnerError(f"unknown stage {stage!r}; expected one of {sorted(STAGE_HANDLERS)}")
    return STAGE_HANDLERS[stage](config, context)


def fingerprints(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from ksvd_research.runtime.serialization import sha256_text

    splits = D.load_splits()
    payload = {
        "dataset": C.DATASET,
        "source_csv_sha256": C.CHEMBENCH_CSV_SHA256,
        "split_files": {
            str(suffix): {
                "file": meta["file"],
                "sha256": meta["file_sha256"],
                "rows_sha256": meta["rows_sha256"],
            }
            for suffix, meta in splits.items()
        },
        "n_rows_total": 4200,
    }
    split_fingerprint = sha256_text(
        "|".join(
            f"{suffix}:{meta['file_sha256']}:{meta['rows_sha256']['train']}:"
            f"{meta['rows_sha256']['valid']}:{meta['rows_sha256']['test']}"
            for suffix, meta in sorted(splits.items())
        )
    )
    return payload, split_fingerprint


def build_runner() -> Runner:
    return Runner(
        name=NAME,
        default_config=CONFIG_PATH,
        default_study=STUDY,
        description=(
            "Tree models on explicit graph statistics: Lipophilicity feature ladder "
            "(G0 / WL2_BIN / WL2_COUNT / WL3_COUNT / MORGAN_COUNT), XGBoost, CPU"
        ),
        run_module=__import__(__name__),
    )
