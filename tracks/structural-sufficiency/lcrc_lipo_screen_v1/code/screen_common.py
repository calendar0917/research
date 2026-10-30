"""Frozen constants, paths and artifact verification for the LCRC Lipophilicity screen.

This round is a *downstream* consumer of the frozen Phase 1a split and of the
frozen ``lcrc_cross_task_preparation_v1`` FeatureSpec / structural caches.  It
never modifies them.  All imports of the preparation code go through this
module so that there is exactly one place where the frozen code directory is
placed on ``sys.path``.

Scope discipline (Phase 2 ``lcrc_lipo_screen_v1``)
--------------------------------------------------
The only permitted supervision is:

* Fit rows 2352 -- training and target standardisation;
* Monitor rows 504 -- development evaluation *after* predictions are frozen;
* Reserved rows 504 -- no labels, no scoring, not even structures in this round;
* official valid/test -- no labels, no scoring, structures not used.

The label file is opened only by :mod:`fit_labels` (Fit keys) and by
:mod:`monitor_labels` (Monitor keys, after the prediction freeze).  Every other
module in this round must go through those two entry points.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROUND_ID = "lcrc-lipo-screen-v1"
PROTOCOL_ID = "lcrc-lipo-screen-v1"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
PREP_ANALYSIS = PREP / "analysis"
PREP_CACHE = PREP / "cache"

ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
CACHE = EXP_DIR / "cache"
RUNS = ANALYSIS / "runs"
PREDICTIONS = ANALYSIS / "predictions"
CONFIG_PATH = CONFIGS / "lcrc_lipo_screen_v1.yaml"

#: Frozen Phase 1a artifacts (structure-only split + Fit/Monitor label artifact).
LIPO_SPLIT = REPO / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"
LIPO_LABELS = REPO / "tracks/structural-sufficiency/results/phase1a/labels_fit_monitor.json"
LIPO_RAW = (
    REPO
    / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz"
)
LIPO_ROW_MANIFEST = (
    REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/row_manifest.json"
)
#: Frozen sources that this round must never open (label-bearing or official).
LIPO_CSV = REPO / "data/lipophilicity/Lipophilicity.csv"
LIPO_OFFICIAL_SPLIT = (
    REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/split_lipo_v1.json"
)
FORBIDDEN_FILES = (
    LIPO_CSV,
    LIPO_OFFICIAL_SPLIT,
    REPO / "data/lipophilicity/labels.npy",
    REPO / "data/lipophilicity/labels_meta.json",
    PREP_CACHE / "lipo_reserved.pt",
)

# ---------------------------------------------------------------------------
# Frozen hashes (copied from the preparation round REPORT / cpu_audit.json)
# ---------------------------------------------------------------------------

EXPECTED_SHA256 = {
    "split_phase1a": "9fa96d49efd94cd2ff381922f83a07f963d832064c75469657b4ce5f68c332b6",
    "labels_fit_monitor": "0260a3909eec30900df646bc760e65360315f4a1398c7092c7c72d04a4e404f7",
    "lipo_raw": "8416837ea3e56e3e5218865391c88913bb16948b48b7a61be1ee890ea4dda6dd",
    "row_manifest": "2c638e4814cb6d21ccec9d8c8af357d28ce210fb2c20c3d09b9c19afd476073d",
    "feature_spec_lipo": "a65c67f9595d9190e7c913d2594ba984a07b51300d63916de865dccca434206c",
    "norm_lipo": "084c129c1731ea92a67d5e9bdce0d8d3bb335cf75ef45595ee690da288394d77",
    "cache_fit_file": "907509ae5fba887061501865ea1e4e62883182a9eeef3586553989fa705d00eb",
    "cache_monitor_file": "2abd1aef16348241568b0f9dc2c79cb55fd6c23d8744f0c58046f1d1e2ff8b54",
}

EXPECTED_ROW_HASHES = {
    "fit": "c474e38cdecf10071965527409901403ec8b670b96fe4a3c42452531dd0210e9",
    "monitor": "fe81cc7904a6da9378f5bf5ae7452ac71629adad83a2a98b917f87299a85f39f",
    "reserved": "bc28c5d37c2fe2fbe39c01d5bf2ab9c56630eee2d0a4454a108d150ab2f42d86",
}

EXPECTED_FEATURE_SPEC_HASH = "7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee"
EXPECTED_NORM_FINGERPRINT = "9c54f31ca26b6bc4b303958dfb465f16512fb6c576d23817aff2117d872d4286"
EXPECTED_LABELS_ROW_HASH = "2ccf20f5293b16dacc5fbaa4542d38ea4bbf2dc3eea68e20f75197ba0fbd8493"

#: ``content_fingerprint`` of the frozen structural caches (labels absent).
EXPECTED_CACHE_FINGERPRINT = {
    "fit": "0a73407fe3b20bb5158b521bcd8dd94bac69d0857ec5c5bd31314c378e9990d9",
    "monitor": "db48a9c237d628025185f76497a024dac35537c77413c511b1d76fb3721919b8",
    "reserved": "8018c9f56ce713a15e82e72dc5a5e0d0ddf32ffd7ed4b30634d3068ec1b7c7a4",
}

SPLIT_SIZES = {"fit": 2352, "monitor": 504, "reserved": 504}

# ---------------------------------------------------------------------------
# Frozen training recipe / run registry
# ---------------------------------------------------------------------------

SEEDS = (0, 1, 2)
NN_ARMS = ("A", "B", "C")
BASELINE_ARMS = ("D",)
BASELINE_REPLICATES = (0, 1, 2)  # XGBoost ``random_state``; replicate 0 is primary
PRIMARY_BASELINE_REPLICATE = 0

BATCH_SIZE = 128
EPOCHS = 360
LR = 1.0e-3
LR_MIN = 1.0e-5
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
DROPOUT = 0.0
SOUP_LAST = 5
#: Offset added to the init seed for the independent batch-order generator.
TRAIN_SHUFFLE_OFFSET = 917_000
#: Steps per epoch for Fit 2352 at batch 128 (last batch has 48 molecules).
STEPS_PER_EPOCH = (SPLIT_SIZES["fit"] + BATCH_SIZE - 1) // BATCH_SIZE  # 19
TOTAL_UPDATES = STEPS_PER_EPOCH * EPOCHS  # 6840

#: GINE parameter matching: ``hidden`` chosen by a deterministic rule (see PROTOCOL).
GINE_LAYERS = 3
GINE_HIDDEN = 170
GINE_RHO_HIDDEN = 64
LCRC_PARAMS_REFERENCE = 247_106
GINE_PARAMS_REFERENCE = 245_305  # verified at runtime; see analysis/parameters.json

#: Fit-only target standardisation is written to this file by ``freeze_protocol``.
NORM_TARGET = ANALYSIS / "norm_target.json"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"

#: Metric / decision constants frozen *before* any GPU run (see PROTOCOL.md).
MATERIALITY_RMSE = 0.03  # raw logD units
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

#: Bootstrap predictions manifest / lock used to gate Monitor label access.
PREDICTION_MANIFEST = PREDICTIONS / "MANIFEST.json"
PREDICTION_LOCK = PREDICTIONS / "MANIFEST.sha256"


# ---------------------------------------------------------------------------
# Import the frozen preparation modules (read-only)
# ---------------------------------------------------------------------------

if str(PREP_CODE) not in sys.path:
    sys.path.insert(0, str(PREP_CODE))

import feature_spec as _feature_spec  # noqa: E402
import lcrc_common as _prep_common  # noqa: E402
import lcrc_data as _prep_data  # noqa: E402
import lcrc_model as _prep_model  # noqa: E402

feature_spec = _feature_spec
prep_common = _prep_common
prep_data = _prep_data
prep_model = _prep_model


# ---------------------------------------------------------------------------
# hashing / json helpers
# ---------------------------------------------------------------------------

sha256_file = _prep_common.sha256_file
read_json = _prep_common.read_json
track_rel = _prep_common.track_rel
utcnow = _prep_common.utcnow
git_commit = _prep_common.git_commit
git_dirty = _prep_common.git_dirty


def source_dirty() -> bool:
    """True iff a tracked *source* file (code/tests/configs) is modified.

    Generated analysis artifacts are deliberately excluded: the freeze step
    rewrites them, and they are committed separately.  Formal runs still use
    ``git_dirty()``, which requires the whole tracked worktree to be clean.
    """
    import subprocess

    rel = track_rel(EXP_DIR)
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO),
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                f"{rel}/code",
                f"{rel}/tests",
                f"{rel}/configs",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return True
peak_rss_mb = _prep_common.peak_rss_mb


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True, default=_json_default)
    p.write_text(text + "\n", encoding="utf-8")
    return p


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str) -> None:
    print(msg, flush=True)


def file_record(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": track_rel(p), "exists": False}
    return {
        "path": track_rel(p),
        "exists": True,
        "bytes": int(p.stat().st_size),
        "sha256": sha256_file(p),
    }


def verify_sha256(path: Path | str, expected: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected:
        raise PermissionError(f"{label}: sha256 {actual} != frozen {expected}")
    return {"label": label, "path": track_rel(path), "sha256": actual, "matches": True}


def env_info() -> dict[str, Any]:
    import platform

    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "round_id": ROUND_ID,
        "timestamp": utcnow(),
    }


# ---------------------------------------------------------------------------
# frozen spec / norm / split access
# ---------------------------------------------------------------------------


def load_frozen_spec():
    """Load ``lipo_v1`` from the frozen first-round JSON and verify identity."""
    path = PREP_ANALYSIS / "feature_spec_lipo.json"
    verify_sha256(path, EXPECTED_SHA256["feature_spec_lipo"], "feature_spec_lipo")
    spec = feature_spec.load_spec(path)
    if spec.name != "lipo_v1":
        raise ValueError(f"unexpected feature spec {spec.name!r}")
    if spec.hash() != EXPECTED_FEATURE_SPEC_HASH:
        raise ValueError(f"feature spec hash {spec.hash()} != frozen {EXPECTED_FEATURE_SPEC_HASH}")
    if (spec.atom_feat_dim, spec.bond_feat_dim, spec.global_dim, spec.n_scalars) != (51, 13, 42, 9):
        raise ValueError("feature spec dimensions drifted")
    return spec


def load_frozen_norm() -> dict[str, Any]:
    path = PREP_ANALYSIS / "norm_lipo.json"
    verify_sha256(path, EXPECTED_SHA256["norm_lipo"], "norm_lipo")
    norm = read_json(path)
    fingerprint = prep_data.norm_fingerprint(norm)
    if fingerprint != EXPECTED_NORM_FINGERPRINT:
        raise ValueError(f"norm fingerprint {fingerprint} != frozen {EXPECTED_NORM_FINGERPRINT}")
    if norm.get("valid_or_test_used") is not False or norm.get("fit_split") != "fit":
        raise PermissionError("norm artifact is not certified Fit-only")
    return norm


def load_frozen_split() -> dict[str, Any]:
    """Read the structure-only Phase 1a split and verify row hashes.

    Only the row-index arrays and the structure-only scope flags are read.  The
    split artifact contains no labels.
    """
    verify_sha256(LIPO_SPLIT, EXPECTED_SHA256["split_phase1a"], "split_phase1a")
    payload = read_json(LIPO_SPLIT)
    if payload.get("protocol_id") != "ssrc-phase1a-lipo-v1":
        raise ValueError("unexpected split protocol id")
    rows = payload["rows"]
    split = {k: [int(x) for x in rows[k]] for k in ("fit", "monitor", "reserved")}
    for key, n in SPLIT_SIZES.items():
        if len(split[key]) != n or len(set(split[key])) != n:
            raise ValueError(f"{key}: expected {n} unique rows, got {len(split[key])}")
        actual = prep_common.row_sha256(split[key])
        if actual != EXPECTED_ROW_HASHES[key]:
            raise ValueError(f"{key}: row hash {actual} != frozen {EXPECTED_ROW_HASHES[key]}")
    for a in ("fit", "monitor", "reserved"):
        for b in ("fit", "monitor", "reserved"):
            if a < b and (set(split[a]) & set(split[b])):
                raise ValueError(f"row overlap between {a} and {b}")
    scope = payload.get("scope", {})
    if scope.get("labels_read") is not False or scope.get("structure_only") is not True:
        raise PermissionError("split artifact is not certified structure-only")
    if scope.get("official_valid_test_labels_read") is not False:
        raise PermissionError("split artifact does not certify official valid/test label exclusion")
    return {
        "split": split,
        "payload": payload,
        "hashes": {k: prep_common.row_sha256(split[k]) for k in split},
    }


# ---------------------------------------------------------------------------
# frozen structural cache access
# ---------------------------------------------------------------------------


def _cache_path(split: str) -> Path:
    return PREP_CACHE / f"lipo_{split}.pt"


def load_frozen_cache(split: str, spec=None) -> dict[str, Any]:
    """Load a frozen structural cache and verify fingerprints and row order.

    The cache is read-only here.  If the local (git-ignored) cache file is
    absent -- e.g. in a fresh clone or on the remote compute checkout -- it is
    rebuilt deterministically from the label-free raw artifact into *this*
    round's ``cache/`` directory and must reproduce the frozen fingerprint
    exactly.  The first round's cache directory is never written.
    """
    if split not in SPLIT_SIZES:
        raise ValueError(f"unknown split {split!r}")
    spec = spec or load_frozen_spec()
    frozen_split = load_frozen_split()
    rows = frozen_split["split"][split]

    path = _cache_path(split)
    rebuilt = False
    if path.exists():
        cache = prep_data.load_cache(path)
        bytes_sha = sha256_file(path)
        expected_bytes = EXPECTED_SHA256.get(f"cache_{split}_file")
        if expected_bytes and bytes_sha != expected_bytes:
            raise PermissionError(f"{split} cache file sha256 {bytes_sha} != frozen {expected_bytes}")
    else:
        rebuilt = True
        cache = _rebuild_cache(split, rows, spec)
        path = CACHE / f"lipo_{split}.pt"
        ensure_dir(CACHE)
        prep_data.save_cache(cache, path)

    fingerprint = prep_common.content_fingerprint(cache)
    if fingerprint != EXPECTED_CACHE_FINGERPRINT[split]:
        raise ValueError(f"{split} cache content fingerprint {fingerprint} != frozen")
    meta = cache.get("meta", {})
    if meta.get("labels_present") is not False:
        raise PermissionError(f"{split} cache is not certified label-free")
    if meta.get("feature_spec_hash") != EXPECTED_FEATURE_SPEC_HASH:
        raise ValueError(f"{split} cache feature spec hash drifted")
    if int(cache["n_molecules"]) != SPLIT_SIZES[split]:
        raise ValueError(f"{split} cache molecule count drifted")
    ids = [str(x) for x in cache["molecule_ids"]]
    expected_ids = [str(int(r)) for r in rows]
    if ids != expected_ids:
        raise ValueError(f"{split} cache row order does not match the frozen split")
    return {
        "cache": cache,
        "path": path,
        "rebuilt": rebuilt,
        "content_fingerprint": fingerprint,
        "file_sha256": sha256_file(path),
        "rows": rows,
        "n_molecules": int(cache["n_molecules"]),
    }


def _rebuild_cache(split: str, rows, spec):
    import adapters as prep_adapters

    raw = prep_adapters.load_lipo_raw_artifact(LIPO_RAW)
    graphs = [prep_adapters.lipo_row_to_graph(raw["rows"][int(r)], int(r), spec) for r in rows]
    return prep_data.build_cache(graphs, spec, split=split)


# ---------------------------------------------------------------------------
# config access and validation
# ---------------------------------------------------------------------------


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def config_hash(cfg: Mapping[str, Any]) -> str:
    return sha256_text(stable_json(cfg))


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Hard-check every scientific constant against the frozen code values."""
    def need(actual, expected, name):
        if actual != expected:
            raise ValueError(f"config drift at {name}: {actual!r} != frozen {expected!r}")

    need(cfg.get("protocol_id"), PROTOCOL_ID, "protocol_id")
    need(cfg.get("round_id"), ROUND_ID, "round_id")
    train = cfg["training"]
    need(int(train["seeds"][0]), SEEDS[0], "training.seeds[0]")
    need(tuple(int(x) for x in train["seeds"]), SEEDS, "training.seeds")
    need(int(train["batch_size"]), BATCH_SIZE, "training.batch_size")
    need(int(train["epochs"]), EPOCHS, "training.epochs")
    need(float(train["peak_lr"]), LR, "training.peak_lr")
    need(float(train["min_lr"]), LR_MIN, "training.min_lr")
    need(float(train["weight_decay"]), WEIGHT_DECAY, "training.weight_decay")
    need(float(train["grad_clip"]), GRAD_CLIP, "training.grad_clip")
    need(float(train["dropout"]), DROPOUT, "training.dropout")
    need(int(train["soup_last"]), SOUP_LAST, "training.soup_last")
    need(train["optimizer"], "AdamW", "training.optimizer")
    need(train["scheduler"], "per_step_cosine", "training.scheduler")
    need(train["checkpoint"], "mean_of_last_5_epoch_end_state_dicts", "training.checkpoint")
    need(train["loss"], "mse_on_fit_standardised_target", "training.loss")
    need(train["target_normalisation"], "fit_only", "training.target_normalisation")
    need(bool(train["monitor_during_training"]), False, "training.monitor_during_training")
    need(bool(train["checkpoint_selection_by_monitor"]), False, "training.checkpoint_selection_by_monitor")
    need(bool(train["early_stopping"]), False, "training.early_stopping")
    need(bool(train["arm_specific_epochs"]), False, "training.arm_specific_epochs")
    need(bool(train["ssl"]), False, "training.ssl")
    need(int(train["shuffle_offset"]), TRAIN_SHUFFLE_OFFSET, "training.shuffle_offset")

    arms = cfg["arms"]
    need(tuple(arms["nn"]), NN_ARMS, "arms.nn")
    need(tuple(arms["baseline"]), BASELINE_ARMS, "arms.baseline")
    need(bool(arms["B"]["s2_off_zeroes_localmlp_input"]), True, "arms.B.s2_off_zeroes_localmlp_input")
    need(bool(arms["B"]["s2_off_train_and_eval"]), True, "arms.B.s2_off_train_and_eval")
    need(str(arms["B"]["model"]), "lcrc_a_lipo_v1", "arms.B.model")
    need(str(arms["C"]["model"]), "gine_lipo_v1", "arms.C.model")
    need(str(arms["D"]["model"]), "ecfp4_xgb", "arms.D.model")

    gine = cfg["models"]["gine"]
    need(int(gine["layers"]), GINE_LAYERS, "models.gine.layers")
    need(int(gine["hidden"]), GINE_HIDDEN, "models.gine.hidden")
    need(int(gine["rho_hidden"]), GINE_RHO_HIDDEN, "models.gine.rho_hidden")
    need(bool(gine["message_passing_bidirectional"]), True, "gine.bidirectional")
    need(str(arms["C"]["atom_feature"]), "lipo_v1.atom51", "arms.C.atom_feature")
    need(str(arms["C"]["bond_feature"]), "lipo_v1.bond13", "arms.C.bond_feature")
    need(str(arms["C"]["adjacency"]), "real_physical_bonds", "arms.C.adjacency")

    baseline = cfg["baseline"]
    need(baseline["fingerprint"], "ecfp4_morgan_r2_2048", "baseline.fingerprint")
    need(baseline["refit_on"], "fit2352", "baseline.refit_on")
    need(baseline["score_on"], "monitor504", "baseline.score_on")
    need(bool(baseline["reuse_previous_bjoc_numbers"]), False, "baseline.reuse_previous_bjoc_numbers")
    need(int(baseline["fp_radius"]), 2, "baseline.fp_radius")
    need(int(baseline["fp_bits"]), 2048, "baseline.fp_bits")
    need(bool(baseline["fp_use_features"]), False, "baseline.fp_use_features")
    xgb = baseline["xgboost"]
    need(int(xgb["n_estimators"]), 500, "xgb.n_estimators")
    need(int(xgb["max_depth"]), 6, "xgb.max_depth")
    need(float(xgb["learning_rate"]), 0.05, "xgb.learning_rate")
    need(float(xgb["subsample"]), 0.8, "xgb.subsample")
    need(float(xgb["colsample_bytree"]), 0.8, "xgb.colsample_bytree")
    need(float(xgb["reg_lambda"]), 1.0, "xgb.reg_lambda")
    need(int(xgb["min_child_weight"]), 1, "xgb.min_child_weight")
    need(str(xgb["tree_method"]), "hist", "xgb.tree_method")
    need(tuple(int(x) for x in baseline["random_states"]), BASELINE_REPLICATES, "baseline.random_states")
    need(int(baseline["primary_random_state"]), PRIMARY_BASELINE_REPLICATE, "baseline.primary_random_state")
    need(bool(baseline["hyperparameter_search"]), False, "baseline.hyperparameter_search")

    ev = cfg["evaluation"]
    need(ev["primary_metric"], "raw_rmse", "evaluation.primary_metric")
    need(tuple(ev["secondary_metrics"]), ("raw_mae", "paired_mse", "paired_mae"), "evaluation.secondary_metrics")
    need(ev["monitor_split"], "monitor", "evaluation.monitor_split")
    need(bool(ev["reserved_labels_read"]), False, "evaluation.reserved_labels_read")
    need(bool(ev["official_valid_test_read"]), False, "evaluation.official_valid_test_read")
    need(float(ev["materiality_rmse"]), MATERIALITY_RMSE, "evaluation.materiality_rmse")
    need(int(ev["bootstrap_resamples"]), BOOTSTRAP_RESAMPLES, "evaluation.bootstrap_resamples")
    need(int(ev["bootstrap_seed"]), BOOTSTRAP_SEED, "evaluation.bootstrap_seed")
    need(bool(ev["seed_and_molecule_uncertainty_separate"]), True, "evaluation.seed_and_molecule_uncertainty_separate")
    need(bool(ev["report_best_seed_only"]), False, "evaluation.report_best_seed_only")
    need(bool(ev["monitor_used_for_selection"]), False, "evaluation.monitor_used_for_selection")


# ---------------------------------------------------------------------------
# protocol freeze
# ---------------------------------------------------------------------------


def code_file_hashes() -> dict[str, str]:
    out = {}
    for path in sorted(list((EXP_DIR / "code").glob("*.py")) + list((EXP_DIR / "tests").glob("*.py"))):
        out[track_rel(path)] = sha256_file(path)
    out[track_rel(CONFIG_PATH)] = sha256_file(CONFIG_PATH)
    return out


def build_protocol_freeze(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "round_id": ROUND_ID,
        "generated_utc": utcnow(),
        "git_commit": git_commit(),
        "git_dirty_tracked": source_dirty(),
        "git_dirty_worktree": git_dirty(),
        "config_hash": config_hash(cfg),
        "config_path": track_rel(CONFIG_PATH),
        "code_hashes": code_file_hashes(),
        "artifacts": {
            k: file_record(p)
            for k, p in {
                "split_phase1a": LIPO_SPLIT,
                "labels_fit_monitor": LIPO_LABELS,
                "lipo_raw": LIPO_RAW,
                "row_manifest": LIPO_ROW_MANIFEST,
                "feature_spec_lipo": PREP_ANALYSIS / "feature_spec_lipo.json",
                "norm_lipo": PREP_ANALYSIS / "norm_lipo.json",
            }.items()
        },
        "expected_hashes": dict(EXPECTED_SHA256),
        "expected_row_hashes": dict(EXPECTED_ROW_HASHES),
        "expected_cache_fingerprints": dict(EXPECTED_CACHE_FINGERPRINT),
        "feature_spec_hash": EXPECTED_FEATURE_SPEC_HASH,
        "norm_fingerprint": EXPECTED_NORM_FINGERPRINT,
        "recipe": {
            "seeds": list(SEEDS),
            "arms": list(NN_ARMS),
            "baseline_arms": list(BASELINE_ARMS),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "total_updates": TOTAL_UPDATES,
            "lr": LR,
            "lr_min": LR_MIN,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "dropout": DROPOUT,
            "soup_last": SOUP_LAST,
            "optimizer": "AdamW",
            "scheduler": "per_step_cosine",
            "loss": "mse_on_fit_standardised_target",
            "target_normalisation": "fit_only",
            "monitor_during_training": False,
            "ssl": False,
        },
        "decision_constants": {
            "materiality_rmse": MATERIALITY_RMSE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "gpu_authorized": False,
        "gpu_hours_used": 0.0,
    }


def verify_protocol_freeze(require_clean: bool = False) -> dict[str, Any]:
    """Refuse to run the formal recipe if the frozen record no longer matches."""
    if not PROTOCOL_FREEZE.exists():
        raise PermissionError(f"missing protocol freeze {track_rel(PROTOCOL_FREEZE)}")
    frozen = read_json(PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("protocol freeze has the wrong protocol id")
    cfg = load_config()
    if frozen["config_hash"] != config_hash(cfg):
        raise PermissionError("config changed after the protocol freeze")
    current = code_file_hashes()
    if current != frozen["code_hashes"]:
        changed = sorted(k for k in set(current) | set(frozen["code_hashes"]) if current.get(k) != frozen["code_hashes"].get(k))
        raise PermissionError(f"code changed after the protocol freeze: {changed}")
    if require_clean and git_dirty():
        raise PermissionError("formal run requires a clean committed worktree")
    return frozen


__all__ = [
    "ROUND_ID",
    "PROTOCOL_ID",
    "CODE_DIR",
    "EXP_DIR",
    "REPO",
    "PREP",
    "PREP_CODE",
    "PREP_ANALYSIS",
    "PREP_CACHE",
    "ANALYSIS",
    "CONFIGS",
    "CACHE",
    "RUNS",
    "PREDICTIONS",
    "CONFIG_PATH",
    "LIPO_SPLIT",
    "LIPO_LABELS",
    "LIPO_RAW",
    "LIPO_ROW_MANIFEST",
    "LIPO_CSV",
    "LIPO_OFFICIAL_SPLIT",
    "FORBIDDEN_FILES",
    "EXPECTED_SHA256",
    "EXPECTED_ROW_HASHES",
    "EXPECTED_FEATURE_SPEC_HASH",
    "EXPECTED_NORM_FINGERPRINT",
    "EXPECTED_LABELS_ROW_HASH",
    "EXPECTED_CACHE_FINGERPRINT",
    "SPLIT_SIZES",
    "SEEDS",
    "NN_ARMS",
    "BASELINE_ARMS",
    "BASELINE_REPLICATES",
    "PRIMARY_BASELINE_REPLICATE",
    "BATCH_SIZE",
    "EPOCHS",
    "LR",
    "LR_MIN",
    "WEIGHT_DECAY",
    "GRAD_CLIP",
    "DROPOUT",
    "SOUP_LAST",
    "TRAIN_SHUFFLE_OFFSET",
    "STEPS_PER_EPOCH",
    "TOTAL_UPDATES",
    "GINE_LAYERS",
    "GINE_HIDDEN",
    "GINE_RHO_HIDDEN",
    "LCRC_PARAMS_REFERENCE",
    "GINE_PARAMS_REFERENCE",
    "NORM_TARGET",
    "PROTOCOL_FREEZE",
    "MATERIALITY_RMSE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "PREDICTION_MANIFEST",
    "PREDICTION_LOCK",
    "feature_spec",
    "prep_common",
    "prep_data",
    "prep_model",
    "sha256_file",
    "sha256_text",
    "read_json",
    "write_json",
    "stable_json",
    "track_rel",
    "utcnow",
    "git_commit",
    "git_dirty",
    "peak_rss_mb",
    "ensure_dir",
    "log",
    "file_record",
    "verify_sha256",
    "env_info",
    "load_frozen_spec",
    "load_frozen_norm",
    "load_frozen_split",
    "load_frozen_cache",
    "load_config",
    "config_hash",
    "validate_config",
    "code_file_hashes",
    "build_protocol_freeze",
    "verify_protocol_freeze",
]
