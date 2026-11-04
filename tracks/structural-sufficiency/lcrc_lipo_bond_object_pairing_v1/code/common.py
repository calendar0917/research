"""Frozen constants, artifact verification and provenance for the bond-object pilot.

This round is a read-only downstream consumer of the frozen Phase 1a split and
of the ``lcrc_cross_task_preparation_v1`` FeatureSpec / structural caches.  It
never writes a historical directory and never loads the Reserved cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROUND_ID = "lcrc-lipo-bond-object-pairing-v1"
PROTOCOL_ID = "lcrc-lipo-bond-object-pairing-v1"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]
PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
PREP_ANALYSIS = PREP / "analysis"
PREP_CACHE = PREP / "cache"
HIST = REPO / "tracks/structural-sufficiency/lcrc_lipo_screen_v1"
HIST_ANALYSIS = HIST / "analysis"
HIST_PREDICTIONS = HIST_ANALYSIS / "predictions"
HIST_MANIFEST = HIST_PREDICTIONS / "MANIFEST.json"
HIST_LOCK = HIST_PREDICTIONS / "MANIFEST.sha256"
HIST_METRICS = HIST_ANALYSIS / "monitor_metrics.json"
HIST_PARAMETERS = HIST_ANALYSIS / "parameters.json"

ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
CACHE = EXP_DIR / "cache"
RUNS = EXP_DIR / "runs"
RESULTS = EXP_DIR / "results"
PREDICTIONS = RESULTS / "monitor_predictions"
CONFIG_PATH = CONFIGS / "protocol.json"

#: Frozen Phase 1a artifacts (structure-only split + Fit/Monitor label artifact).
LIPO_SPLIT = REPO / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"
LIPO_LABELS = REPO / "tracks/structural-sufficiency/results/phase1a/labels_fit_monitor.json"
LIPO_RAW = REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz"
LIPO_ROW_MANIFEST = REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/row_manifest.json"
LIPO_CSV = REPO / "data/lipophilicity/Lipophilicity.csv"
LIPO_OFFICIAL_SPLIT = REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/split_lipo_v1.json"
FORBIDDEN_FILES = (
    LIPO_CSV,
    LIPO_OFFICIAL_SPLIT,
    REPO / "data/lipophilicity/labels.npy",
    REPO / "data/lipophilicity/labels_meta.json",
    PREP_CACHE / "lipo_reserved.pt",
)

# ---------------------------------------------------------------------------
# Frozen hashes (same values as the historical screen; independently rechecked)
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

EXPECTED_CACHE_FINGERPRINT = {
    "fit": "0a73407fe3b20bb5158b521bcd8dd94bac69d0857ec5c5bd31314c378e9990d9",
    "monitor": "db48a9c237d628025185f76497a024dac35537c77413c511b1d76fb3721919b8",
    "reserved": "8018c9f56ce713a15e82e72dc5a5e0d0ddf32ffd7ed4b30634d3068ec1b7c7a4",
}

SPLIT_SIZES = {"fit": 2352, "monitor": 504, "reserved": 504}

# ---------------------------------------------------------------------------
# Frozen recipe / experiment registry
# ---------------------------------------------------------------------------

SEEDS = (0, 1, 2)
ARMS = ("R", "N")             # R = REAL_PAIRING, N = MEANFIELD_NULL
HIST_ARM = "A"
RUN_KEYS = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS)

BATCH_SIZE = 128
EPOCHS = 360
LR = 1.0e-3
LR_MIN = 1.0e-5
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
DROPOUT = 0.0
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 917_000
STEPS_PER_EPOCH = (SPLIT_SIZES["fit"] + BATCH_SIZE - 1) // BATCH_SIZE  # 19
TOTAL_UPDATES = STEPS_PER_EPOCH * EPOCHS  # 6840

LCRC_PARAMS_REFERENCE = 247_106
NEW_PARAMS = 12_449
TOTAL_PARAMS_REFERENCE = 259_555
NEW_PARAM_BREAKDOWN = {
    "atom_context_projection": 64 * 32 + 32,
    "edge_hidden_128_64": 128 * 64 + 64,
    "edge_hidden_64_32": 64 * 32 + 32,
    "edge_score_32_1": 32 * 1 + 1,
}

#: Historical LCRC-A initial state hashes by seed (analysis/parameters.json).
EXPECTED_TRUNK_INIT_HASH = {
    0: "e7a688ff91f29489e33568065f5fb85e17dfad5bc09b364a1ae8a08d76ecba6d",
    1: "de28d5f425a2f7f851226391edfaaf6835b7c0e9c2b43164076233d7a2ec64ba",
    2: "08ed4e6dcc895af0c4372abc587a3cc64863e42a640f9d534c50b374d4351e07",
}
#: Historical frozen A Monitor RMSE (display only; formal metric uses predictions).
HISTORICAL_A_RMSE = {0: 0.9343236801196487, 1: 0.8806613918897982, 2: 0.9150863075696288}

MATERIALITY_RMSE = 0.03
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 0
GPU_HOUR_CAP = 1.5
RUN_WALL_CAP_SECONDS = 1_200

#: Bond-count grouping fixed before evaluation.
BOND_COUNT_BINS = (
    ("0", 0, 0),
    ("1-4", 1, 4),
    ("5-8", 5, 8),
    ("9-12", 9, 12),
    ("13+", 13, 10**9),
)

PREDICTION_MANIFEST = ANALYSIS / "prediction_manifest.json"
PREDICTION_LOCK = ANALYSIS / "prediction_manifest.sha256"
PROTOCOL_FREEZE = ANALYSIS / "preflight.json"

# ---------------------------------------------------------------------------
# Import frozen preparation modules (read-only)
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

sha256_file = _prep_common.sha256_file
read_json = _prep_common.read_json
track_rel = _prep_common.track_rel
utcnow = _prep_common.utcnow
row_sha256 = _prep_common.row_sha256
content_fingerprint = _prep_common.content_fingerprint


# ---------------------------------------------------------------------------
# JSON / provenance helpers
# ---------------------------------------------------------------------------


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


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_branch() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True)
        return bool(out.strip())
    except Exception:
        return True


def source_dirty() -> bool:
    rel = track_rel(EXP_DIR)
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no", "--", f"{rel}/code", f"{rel}/tests", f"{rel}/configs"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return True


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def file_record(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": track_rel(p), "exists": False}
    return {"path": track_rel(p), "exists": True, "bytes": int(p.stat().st_size), "sha256": sha256_file(p)}


def verify_sha256(path: Path | str, expected: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected:
        raise PermissionError(f"{label}: sha256 {actual} != frozen {expected}")
    return {"label": label, "path": track_rel(path), "sha256": actual, "matches": True}


def env_info() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty_tracked": git_dirty(),
        "round_id": ROUND_ID,
        "timestamp": utcnow(),
        "pid": os.getpid(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


# ---------------------------------------------------------------------------
# Frozen spec / norm / split / cache
# ---------------------------------------------------------------------------


def load_frozen_spec():
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
    verify_sha256(LIPO_SPLIT, EXPECTED_SHA256["split_phase1a"], "split_phase1a")
    payload = read_json(LIPO_SPLIT)
    if payload.get("protocol_id") != "ssrc-phase1a-lipo-v1":
        raise ValueError("unexpected split protocol id")
    rows = payload["rows"]
    split = {k: [int(x) for x in rows[k]] for k in ("fit", "monitor", "reserved")}
    for key, n in SPLIT_SIZES.items():
        if len(split[key]) != n or len(set(split[key])) != n:
            raise ValueError(f"{key}: expected {n} unique rows, got {len(split[key])}")
        actual = row_sha256(split[key])
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
        "hashes": {k: row_sha256(split[k]) for k in split},
    }


def _cache_path(split: str) -> Path:
    return PREP_CACHE / f"lipo_{split}.pt"


def load_frozen_cache(split: str, spec=None) -> dict[str, Any]:
    """Load a frozen structural cache; if absent, rebuild into *this* track's cache."""
    if split not in ("fit", "monitor"):
        raise ValueError("this round may only load the Fit and Monitor caches")
    spec = spec or load_frozen_spec()
    frozen_split = load_frozen_split()
    rows = frozen_split["split"][split]

    path = _cache_path(split)
    rebuilt = False
    if not path.exists():
        local_path = CACHE / f"lipo_{split}.pt"
        if local_path.exists():
            path = local_path
    if path.exists():
        cache = prep_data.load_cache(path)
        bytes_sha = sha256_file(path)
        expected_bytes = EXPECTED_SHA256.get(f"cache_{split}_file")
        file_sha_matches = (expected_bytes is None) or (bytes_sha == expected_bytes)
        if not file_sha_matches and path.parent != CACHE:
            raise PermissionError(f"{split} cache file sha256 {bytes_sha} != frozen {expected_bytes}")
    else:
        rebuilt = True
        cache = _rebuild_cache(split, rows, spec)
        path = CACHE / f"lipo_{split}.pt"
        ensure_dir(CACHE)
        prep_data.save_cache(cache, path)
        bytes_sha = sha256_file(path)
        expected_bytes = EXPECTED_SHA256.get(f"cache_{split}_file")
        file_sha_matches = (expected_bytes is None) or (bytes_sha == expected_bytes)

    fingerprint = content_fingerprint(cache)
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
        "content_fingerprint_matches_frozen": fingerprint == EXPECTED_CACHE_FINGERPRINT[split],
        "file_sha256": sha256_file(path),
        "file_sha256_expected_from_historical_serialization": EXPECTED_SHA256.get(f"cache_{split}_file"),
        "file_sha256_matches_historical_serialization": file_sha_matches,
        "consistency_basis": "all array bytes (content_fingerprint) match; torch serialization file bytes may differ across rebuilds",
        "rows": rows,
        "n_molecules": int(cache["n_molecules"]),
    }


def _rebuild_cache(split: str, rows, spec):
    import adapters as prep_adapters

    raw = prep_adapters.load_lipo_raw_artifact(LIPO_RAW)
    graphs = [prep_adapters.lipo_row_to_graph(raw["rows"][int(r)], int(r), spec) for r in rows]
    return prep_data.build_cache(graphs, spec, split=split)


# ---------------------------------------------------------------------------
# Config / protocol freeze
# ---------------------------------------------------------------------------


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    cfg = read_json(path)
    validate_config(cfg)
    return cfg


def config_hash(cfg: Mapping[str, Any]) -> str:
    return sha256_text(stable_json(cfg))


def validate_config(cfg: Mapping[str, Any]) -> None:
    def need(actual, expected, name):
        if actual != expected:
            raise ValueError(f"config drift at {name}: {actual!r} != frozen {expected!r}")

    need(cfg.get("protocol_id"), PROTOCOL_ID, "protocol_id")
    need(cfg.get("round_id"), ROUND_ID, "round_id")
    tr = cfg["training"]
    need(tuple(int(x) for x in tr["seeds"]), SEEDS, "training.seeds")
    need(int(tr["batch_size"]), BATCH_SIZE, "training.batch_size")
    need(int(tr["epochs"]), EPOCHS, "training.epochs")
    need(float(tr["peak_lr"]), LR, "training.peak_lr")
    need(float(tr["min_lr"]), LR_MIN, "training.min_lr")
    need(float(tr["weight_decay"]), WEIGHT_DECAY, "training.weight_decay")
    need(float(tr["grad_clip"]), GRAD_CLIP, "training.grad_clip")
    need(float(tr["dropout"]), DROPOUT, "training.dropout")
    need(int(tr["soup_last"]), SOUP_LAST, "training.soup_last")
    need(int(tr["steps_per_epoch"]), STEPS_PER_EPOCH, "training.steps_per_epoch")
    need(int(tr["total_updates"]), TOTAL_UPDATES, "training.total_updates")
    need(tr["optimizer"], "AdamW", "training.optimizer")
    need(tr["scheduler"], "per_step_cosine", "training.scheduler")
    need(tr["checkpoint"], "mean_of_last_5_epoch_end_state_dicts", "training.checkpoint")
    need(tr["loss"], "mse_on_fit_standardised_target", "training.loss")
    need(int(tr["shuffle_offset"]), TRAIN_SHUFFLE_OFFSET, "training.shuffle_offset")
    need(bool(tr["monitor_during_training"]), False, "training.monitor_during_training")
    need(bool(tr["early_stopping"]), False, "training.early_stopping")

    model = cfg["model"]
    need(int(model["atom_out"]), 96, "model.atom_out")
    need(int(model["width"]), 64, "model.width")
    need(int(model["bond_out"]), 96, "model.bond_out")
    need(int(model["atom_context_dim"]), 32, "model.atom_context_dim")
    need(int(model["edge_hidden_1"]), 64, "model.edge_hidden_1")
    need(int(model["edge_hidden_2"]), 32, "model.edge_hidden_2")
    need(int(model["new_parameters"]), NEW_PARAMS, "model.new_parameters")
    need(int(model["total_parameters"]), TOTAL_PARAMS_REFERENCE, "model.total_parameters")

    ev = cfg["evaluation"]
    need(float(ev["materiality_rmse"]), MATERIALITY_RMSE, "evaluation.materiality_rmse")
    need(int(ev["bootstrap_resamples"]), BOOTSTRAP_RESAMPLES, "evaluation.bootstrap_resamples")
    need(int(ev["bootstrap_seed"]), BOOTSTRAP_SEED, "evaluation.bootstrap_seed")
    need(ev["primary_metric"], "raw_rmse", "evaluation.primary_metric")
    need(bool(ev["monitor_used_for_selection"]), False, "evaluation.monitor_used_for_selection")
    need(bool(ev["reserved_labels_read"]), False, "evaluation.reserved_labels_read")
    need(bool(ev["official_valid_test_read"]), False, "evaluation.official_valid_test_read")


def code_file_hashes() -> dict[str, str]:
    out = {}
    for path in sorted(list((EXP_DIR / "code").glob("*.py")) + list((EXP_DIR / "tests").glob("*.py"))):
        out[track_rel(path)] = sha256_file(path)
    out[track_rel(CONFIG_PATH)] = sha256_file(CONFIG_PATH)
    return out


def build_preflight_manifest() -> dict[str, Any]:
    cfg = load_config()
    return {
        "protocol_id": PROTOCOL_ID,
        "round_id": ROUND_ID,
        "generated_utc": utcnow(),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty_tracked": git_dirty(),
        "git_dirty_source": source_dirty(),
        "worktree": str(EXP_DIR),
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
                "historical_screen_manifest": HIST_MANIFEST,
                "historical_screen_lock": HIST_LOCK,
                "historical_screen_metrics": HIST_METRICS,
            }.items()
        },
        "expected_hashes": dict(EXPECTED_SHA256),
        "expected_row_hashes": dict(EXPECTED_ROW_HASHES),
        "expected_cache_fingerprints": dict(EXPECTED_CACHE_FINGERPRINT),
        "recipe": {
            "seeds": list(SEEDS),
            "arms": list(ARMS),
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
            "shuffle_offset": TRAIN_SHUFFLE_OFFSET,
            "optimizer": "AdamW",
            "scheduler": "per_step_cosine",
            "loss": "mse_on_fit_standardised_target",
            "target_normalisation": "fit_only",
            "ssl": False,
            "early_stopping": False,
            "monitor_during_training": False,
        },
        "parameters": {
            "trunk": LCRC_PARAMS_REFERENCE,
            "new": NEW_PARAMS,
            "total": TOTAL_PARAMS_REFERENCE,
            "new_breakdown": dict(NEW_PARAM_BREAKDOWN),
        },
        "decision_constants": {
            "materiality_rmse": MATERIALITY_RMSE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "gpu_hour_cap": GPU_HOUR_CAP,
        "run_wall_cap_seconds": RUN_WALL_CAP_SECONDS,
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
    }


def verify_protocol_freeze(require_clean: bool = False) -> dict[str, Any]:
    if not PROTOCOL_FREEZE.exists():
        raise PermissionError(f"missing protocol freeze {track_rel(PROTOCOL_FREEZE)}")
    frozen = read_json(PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("protocol freeze has the wrong protocol id")
    cfg = load_config()
    if frozen.get("config_hash") != config_hash(cfg):
        raise PermissionError("config changed after the protocol freeze")
    current = code_file_hashes()
    if current != frozen.get("code_hashes"):
        changed = sorted(k for k in set(current) | set(frozen.get("code_hashes", {})) if current.get(k) != frozen["code_hashes"].get(k))
        raise PermissionError(f"code changed after the protocol freeze: {changed}")
    if require_clean and git_dirty():
        raise PermissionError("formal run requires a clean committed worktree")
    return frozen


# ---------------------------------------------------------------------------
# Historical A predictions (read-only reference)
# ---------------------------------------------------------------------------


def verify_historical_a_predictions() -> dict[str, Any]:
    """Hash-check the three frozen A predictions and their manifest lock."""
    if not HIST_MANIFEST.exists() or not HIST_LOCK.exists():
        raise FileNotFoundError("historical A prediction manifest/lock missing")
    lock = read_json(HIST_LOCK)
    actual = sha256_file(HIST_MANIFEST)
    if lock.get("sha256") != actual:
        raise PermissionError("historical prediction manifest does not match its lock")
    manifest = read_json(HIST_MANIFEST)
    if manifest.get("status") != "FROZEN" or manifest.get("monitor_labels_read") is not False:
        raise PermissionError("historical prediction manifest is not a label-blind freeze")
    rows = [int(r) for r in load_frozen_split()["split"]["monitor"]]
    rows_sha = hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()
    if manifest.get("rows") != rows or manifest.get("rows_sha256") != rows_sha:
        raise PermissionError("historical manifest Monitor row order differs")
    out = {}
    for seed in SEEDS:
        key = f"A_s{seed}"
        entry = manifest["predictions"].get(key)
        if entry is None:
            raise PermissionError(f"historical prediction {key} missing")
        path = REPO / entry["file"]
        if not path.exists() or sha256_file(path) != entry["sha256"]:
            raise PermissionError(f"{key}: historical prediction hash mismatch")
        with np.load(path) as data:
            pred = np.asarray(data["pred"], dtype=np.float64)
            got_rows = [int(r) for r in data["rows"]]
        if got_rows != rows:
            raise PermissionError(f"{key}: prediction row order differs")
        if pred.shape != (SPLIT_SIZES["monitor"],) or not np.isfinite(pred).all():
            raise PermissionError(f"{key}: invalid prediction vector")
        out[key] = {
            "file": entry["file"],
            "sha256": entry["sha256"],
            "rows_sha256": entry["rows_sha256"],
            "soup_state_hash": entry.get("soup_state_hash"),
            "pred_sha256": hashlib.sha256(np.ascontiguousarray(pred, dtype="<f8").tobytes()).hexdigest(),
            "pred": pred,
            "rows": got_rows,
        }
    return {
        "manifest": track_rel(HIST_MANIFEST),
        "manifest_sha256": actual,
        "lock_sha256": lock.get("sha256"),
        "rows_sha256": rows_sha,
        "predictions": out,
        "monitor_labels_read": False,
    }


# ---------------------------------------------------------------------------
# GPU provenance (read-only nvidia-smi)
# ---------------------------------------------------------------------------


def _nvidia_smi(query: str) -> list[str]:
    try:
        out = subprocess.check_output(["nvidia-smi", f"--query-{query}", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def gpu_runtime_info() -> dict[str, Any]:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    info: dict[str, Any] = {
        "pid": os.getpid(),
        "CUDA_VISIBLE_DEVICES": visible,
        "cuda_available": bool(torch.cuda.is_available()),
        "nvidia_smi_gpus": _nvidia_smi("gpu=index,uuid,name,memory.total,memory.used,utilization.gpu"),
        "nvidia_smi_compute_apps": _nvidia_smi("compute-apps=gpu_uuid,pid,used_memory"),
    }
    if info["cuda_available"]:
        idx = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(idx)
        uuid = None
        try:
            uuid = str(props.uuid)
        except Exception:
            uuid = None
        physical_index = None
        physical_uuid = uuid
        if visible is not None and visible.strip().isdigit():
            physical_index = int(visible.strip())
            for line in info["nvidia_smi_gpus"]:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0] == str(physical_index):
                    physical_uuid = parts[1]
                    break
        elif visible and visible.startswith("GPU-"):
            physical_uuid = visible
        info.update(
            {
                "visible_device_index": idx,
                "device_name": torch.cuda.get_device_name(idx),
                "device_uuid": uuid,
                "physical_gpu_index": physical_index,
                "physical_gpu_uuid": physical_uuid,
                "total_memory_gb": float(props.total_memory) / 1024**3,
            }
        )
    return info


__all__ = [name for name in globals() if not name.startswith("_")]
