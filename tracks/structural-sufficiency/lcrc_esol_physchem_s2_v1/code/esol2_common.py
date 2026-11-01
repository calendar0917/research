"""Shared paths, frozen constants and provenance helpers for ``lcrc_esol_physchem_s2_v1``.

Round question (two *independent* experiments on the *same* ESOL split):

H1 ``ESOL-BASE`` vs ``ESOL-PHYSCHEM6``
    does giving the global ``rho`` branch six explicitly recomputed
    physicochemical descriptors (MolWt, MolLogP, TPSA, HBD, HBA,
    RotatableBonds) improve small-sample water-solubility prediction?

H2 ``ESOL-BASE`` vs ``ESOL-S2BLIND-E``
    does the *typed* two-hop S2 path content carry predictive value that an
    S1/local-context network with matched effective capacity cannot replace?

Everything structural, the dataset, the split, the BASE checkpoints and the
historical E control are **reused read-only**.  This module only defines paths,
pins and the frozen recipe; nothing in a historical directory is written.

Read-only imports
-----------------
* ``lcrc_esol_quickstart_v1/code/esol_common.py``  — frozen constants, helpers
* ``lcrc_esol_quickstart_v1/code/esol_data.py``    — label-free cache / views
* ``lcrc_esol_quickstart_v1/code/esol_raw.py``     — CSV + artifacts (no RDKit)
* ``lcrc_esol_quickstart_v1/code/esol_split.py``   — frozen split loader
* ``lcrc_cross_task_preparation_v1/code/lcrc_*.py``— schema, cache, LCRC-A
* ``lcrc_lipo_mechanism_generalization_ssl_v2/code/v2_models.py``
                                       — the validated capacity-matched E control
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

V1 = REPO / "tracks/structural-sufficiency/lcrc_esol_quickstart_v1"
V1_CODE = V1 / "code"
V1_ANALYSIS = V1 / "analysis"
V2 = REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2"
V2_CODE = V2 / "code"
PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"

for _path in (str(CODE_DIR), str(V1_CODE), str(V2_CODE), str(PREP_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: v1 module.  Importing it also puts the frozen prep / lipophilicity code on
#: ``sys.path``; everything ESOL-specific that we reuse comes from here.
import esol_common as V1C  # noqa: E402
import esol_data as V1D  # noqa: E402,F401  (re-exported for the data module)
import esol_raw as V1R  # noqa: E402,F401
import esol_split as V1S  # noqa: E402,F401
import lcrc_model as PREP_MODEL  # noqa: E402,F401
import v2_models as V2M  # noqa: E402,F401

# ---------------------------------------------------------------------------
# round identity
# ---------------------------------------------------------------------------

ROUND_ID = "lcrc-esol-physchem-s2-v1"
PROTOCOL_ID = ROUND_ID

#: Frozen ESOL split (unchanged, hash-pinned below).
SPLIT_SEED = int(V1C.SPLIT_SEED)
SEEDS = (0, 1, 2)

# ---------------------------------------------------------------------------
# paths (this round only)
# ---------------------------------------------------------------------------

ANALYSIS = EXP_DIR / "analysis"
RUNS = ANALYSIS / "runs"
SMOKE = ANALYSIS / "smoke"
CONFIGS = EXP_DIR / "configs"
CACHE = EXP_DIR / "cache"

CONFIG_PATH = CONFIGS / "esol2.yaml"
P0_GATES = ANALYSIS / "p0_gates.json"
P0_ROW_ALIGNMENT = ANALYSIS / "p0_row_alignment.csv"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
BASE_REPLAY = ANALYSIS / "base_replay.json"
P2_INTERVENTION = ANALYSIS / "p2_intervention.json"
VALID_COMPARISON = ANALYSIS / "valid_comparison.csv"
PAIRED_VALID = ANALYSIS / "paired_valid.csv"
PAIRED_SUMMARY = ANALYSIS / "paired_summary.json"
PHYSCHEM_USAGE = ANALYSIS / "physchem_usage.json"
MANIFEST = ANALYSIS / "manifest.json"
GPU_LEDGER = ANALYSIS / "gpu_ledger.json"
COVERAGE = ANALYSIS / "coverage.json"

#: The derived structural cache is redirected here so that the read-only v1
#: directory is never written (the remote hosts rebuild it from the committed
#: label-free raw-graph artifact).
CACHE_FILE = CACHE / "lcrc_esol_all1128.pt"

# ---------------------------------------------------------------------------
# read-only inputs from the previous ESOL round (hash-pinned)
# ---------------------------------------------------------------------------

CSV_PATH = V1C.CSV_PATH
RAW_GRAPH = V1C.RAW_GRAPH
PHYSCHEM_FILE = V1C.PHYSCHEM_FILE
SCAFFOLD_FILE = V1C.SCAFFOLD_FILE
SPLIT_MANIFEST = V1C.SPLIT_MANIFEST
SPLIT_AUDIT = V1C.SPLIT_AUDIT
V1_RUNS = V1_ANALYSIS / "runs"
V1_MANIFEST = V1_ANALYSIS / "manifest.json"

CSV_SHA256 = V1C.CSV_SHA256
RAW_GRAPH_SHA256 = "4c704b4983efce332b62a25fc80267f622f4674a7a59aa4448a4e3a6452c249b"
PHYSCHEM_SHA256 = "b3be094811805287993d9ce8fd4a7924e460a4585e5ef72946f803c0824ae599"
SPLIT_MANIFEST_SHA256 = "47db06adfd4f5206fa6df3930a2522010c9fc6b51d29a550da3dd83ca8a71a43"
SPLIT_AUDIT_SHA256 = "8e473293e47394821d6fbf81782902cf9368963b67a2dd4b2e970627d0fe7b7e"
LCRC_CACHE_CONTENT_FINGERPRINT = "30f2bf97be4c78bd261b67b277f46df5f3c5d2dddbaf35d09a82a79bf5d353e3"
V1_MANIFEST_SHA256 = "81f6508cb4c9cce7cc5ff9a124d4c779b4364840e7fdad04b1e4d0cae4455e98"  # internal self-digest
V1_MANIFEST_FILE_SHA256 = "8d8eda2a774c7fef58a7da37e9ebef95893de28bb2ca76c73d2854da65df1f23"  # file bytes
V1_TRAIN_NORM_FINGERPRINT = "ba7bf26ea84390b9f81139d233a021e574c48123086da3135128fda0cabad11c"

N_TRAIN, N_VALID, N_TEST = 902, 113, 113

#: Frozen BASE evidence (Validation only; the Test numbers of the old round are
#: deliberately not restated here — this round never evaluates Test).
V1_BASE_VALID_RMSE = {0: 0.7204751406662424, 1: 0.7669908516208004, 2: 0.66122157828635}
V1_BASE_TRAIN_RMSE = {0: 0.4001862859899487, 1: 0.43156937372119036, 2: 0.4196118956083745}
V1_BASE_BEST_EPOCH = {0: 172, 1: 157, 2: 159}
V1_BASE_BATCH_ORDER_SHA256 = {
    0: "007413ff0c6538e99890153695be811a5f96c66ff688ac69ab80b0bcf393e939",
    1: "70f7db8ae2773896a2dff76fe30f8d16da6373bdb51604c0d8e0ae33c5889f02",
    2: "16182097c036fcebee581224c8dd50901d62f0a019490c53e1c634f5bca724b4",
}
#: Validation RMSE agreement required from the BASE replay (v1's own reload
#: check was <= 2.1e-8; one micro-unit is a generous but still strict bound).
BASE_REPLAY_TOL = 1.0e-6

PHYSCHEM_NAMES = tuple(V1C.PHYSCHEM_NAMES)
N_PHYSCHEM = len(PHYSCHEM_NAMES)

# ---------------------------------------------------------------------------
# models / arms
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "base": "ESOL-BASE",
    "physchem6": "ESOL-PHYSCHEM6",
    "s2blind_e": "ESOL-S2BLIND-E",
}
#: Both new arms use the frozen ESOL-BASE regularisation (dropout 0.10).
MODEL_DROPOUT = {"base": 0.10, "physchem6": 0.10, "s2blind_e": 0.10}
CANDIDATES = ("physchem6", "s2blind_e")
NEW_ARMS = tuple(f"{m}_s{s}" for m in CANDIDATES for s in SEEDS)


def arm_key(model: str, seed: int) -> str:
    return f"{model}_s{int(seed)}"


def arm_parts(arm: str) -> tuple[str, int]:
    if "_s" not in arm:
        raise ValueError(f"bad arm name {arm!r}")
    model, seed = arm.rsplit("_s", 1)
    if model not in MODEL_DISPLAY:
        raise ValueError(f"unknown model {model!r}")
    return model, int(seed)


# ---------------------------------------------------------------------------
# frozen training recipe (inherited verbatim from lcrc_esol_quickstart_v1)
# ---------------------------------------------------------------------------

TRAIN_RECIPE = dict(V1C.TRAIN_RECIPE)
if float(TRAIN_RECIPE["dropout"]) != 0.10:
    raise ValueError("inherited dropout is not 0.10")
if float(TRAIN_RECIPE["gradient_clip"]) != 5.0:
    raise ValueError("inherited gradient clip is not 5.0")
if int(TRAIN_RECIPE["epochs"]) != 360:
    raise ValueError("inherited epoch budget is not 360")
SHUFFLE_OFFSET = int(V1C.SHUFFLE_OFFSET)
GPU_HARD_CAP_HOURS = 0.6

#: Model sizes (used by the capacity audit; recomputed, not trusted).
BASE_PARAMETERS = 247_106
E_NOMINAL_PARAMETERS = 285_449
E_EFFECTIVE_PARAMETERS = 247_049
E_DEAD_LOCALMLP_COLUMNS = 38_400

# ---------------------------------------------------------------------------
# helpers re-exported from the frozen v1 module (identical conventions)
# ---------------------------------------------------------------------------

sha256_file = V1C.sha256_file
sha256_bytes = V1C.sha256_bytes
sha256_json = V1C.sha256_json
read_json = V1C.read_json
write_json = V1C.write_json
log = V1C.log
utcnow = V1C.utcnow
track_rel = V1C.track_rel
rows_sha256 = V1C.rows_sha256
vector_sha256_float64 = V1C.vector_sha256_float64
state_dict_fingerprint = V1C.state_dict_fingerprint
git_commit = V1C.git_commit
git_dirty = V1C.git_dirty
all_metrics = V1C.all_metrics
env_info = V1C.env_info
spec = V1C.spec


def config_sha256() -> str:
    return sha256_file(CONFIG_PATH)


def code_hashes() -> dict[str, str]:
    """This round's code plus every read-only module it imports."""
    out = {f"esol2/{path.name}": sha256_file(path) for path in sorted(CODE_DIR.glob("*.py"))}
    for name in (
        "esol_common.py",
        "esol_data.py",
        "esol_models.py",
        "esol_raw.py",
        "esol_split.py",
    ):
        out[f"v1/{name}"] = sha256_file(V1_CODE / name)
    for name in ("feature_spec.py", "adapters.py", "lcrc_data.py", "lcrc_model.py"):
        out[f"prep/{name}"] = sha256_file(PREP_CODE / name)
    out["v2/v2_models.py"] = sha256_file(V2_CODE / "v2_models.py")
    out["v2/v2_common.py"] = sha256_file(V2_CODE / "v2_common.py")
    out["lipo/bjoc_lipophilicity_adapter_v1.py"] = sha256_file(
        V1C.LIPOPHILICITY_CODE / "bjoc_lipophilicity_adapter_v1.py"
    )
    return out


def load_config() -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if str(cfg.get("protocol_id")) != PROTOCOL_ID:
        raise ValueError("esol2 config protocol_id drift")
    if int(cfg.get("split_seed", -1)) != SPLIT_SEED:
        raise ValueError("esol2 config split_seed drift")
    if [str(x) for x in cfg.get("seeds", [])] != [str(s) for s in SEEDS]:
        raise ValueError("esol2 config seeds drift")
    if [str(x) for x in cfg.get("models", [])] != list(CANDIDATES):
        raise ValueError("esol2 config models drift")
    training = dict(cfg["training"])
    for key, expected in TRAIN_RECIPE.items():
        if key not in training:
            raise ValueError(f"config missing training.{key}")
        if training[key] != expected:
            raise ValueError(f"config training.{key} drift: {training[key]!r} != {expected!r}")
    if float(cfg["budget"]["gpu_hard_cap_hours"]) != GPU_HARD_CAP_HOURS:
        raise ValueError("config GPU cap drift")
    if int(cfg["budget"]["max_formal_runs"]) != 6:
        raise ValueError("config formal-run cap drift")
    if [str(x) for x in cfg.get("descriptors", [])] != list(PHYSCHEM_NAMES):
        raise ValueError("config descriptor list drift")
    return cfg


def run_dir(arm: str) -> Path:
    return RUNS / str(arm)


def v1_run_dir(arm: str) -> Path:
    return V1_RUNS / str(arm)


def arms_with_runs() -> list[str]:
    if not RUNS.exists():
        return []
    out = []
    for path in sorted(RUNS.iterdir()):
        if (path / "train.json").exists():
            arm_parts(path.name)
            out.append(path.name)
    return out


# ---------------------------------------------------------------------------
# read-only verification of the frozen v1 evidence
# ---------------------------------------------------------------------------


def v1_manifest() -> dict[str, Any]:
    """Verify the v1 manifest's own digest and return it."""
    payload = read_json(V1_MANIFEST)
    body = {k: v for k, v in payload.items() if k != "manifest_sha256"}
    if sha256_json(body) != payload["manifest_sha256"]:
        raise PermissionError("v1 manifest self-digest mismatch")
    if payload["manifest_sha256"] != V1_MANIFEST_SHA256:
        raise PermissionError("v1 manifest sha256 differs from the pinned value")
    return payload


def v1_base_checkpoint_pins() -> dict[int, dict[str, Any]]:
    manifest = v1_manifest()
    out: dict[int, dict[str, Any]] = {}
    for arm, rec in manifest["arms"].items():
        # v1's manifest also contains the (closed) DROP20 arm; only BASE is reused.
        name, seed_text = arm.rsplit("_s", 1)
        if name != "base":
            continue
        seed = int(seed_text)
        if str(rec["family"]) != "base":
            raise PermissionError(f"v1 manifest arm {arm} is not BASE")
        out[seed] = {
            "arm": arm,
            "checkpoint": V1_RUNS / arm / "checkpoint.pt",
            "checkpoint_sha256": str(rec["checkpoint_sha256"]),
            "best_epoch": int(rec["best_epoch"]),
            "valid_rmse": float(rec["best_inner_valid_rmse"]),
            "train_json": V1_RUNS / arm / "train.json",
        }
    if sorted(out) != list(SEEDS):
        raise PermissionError("v1 manifest does not contain the three BASE seeds")
    return out


def verify_readonly_inputs() -> dict[str, Any]:
    """Re-verify every frozen input this round consumes."""
    pins = {
        "csv": (CSV_PATH, CSV_SHA256),
        "raw_graph": (RAW_GRAPH, RAW_GRAPH_SHA256),
        "physchem": (PHYSCHEM_FILE, PHYSCHEM_SHA256),
        "split_manifest": (SPLIT_MANIFEST, SPLIT_MANIFEST_SHA256),
        "split_audit": (SPLIT_AUDIT, SPLIT_AUDIT_SHA256),
        "v1_manifest": (V1_MANIFEST, V1_MANIFEST_FILE_SHA256),
    }
    out: dict[str, Any] = {}
    bad: list[str] = []
    for name, (path, expected) in pins.items():
        if not Path(path).exists():
            out[name] = {"path": track_rel(path), "exists": False, "sha256_matches": False}
            bad.append(name)
            continue
        actual = sha256_file(path)
        ok = actual == expected
        out[name] = {
            "path": track_rel(path),
            "exists": True,
            "sha256": actual,
            "expected_sha256": expected,
            "sha256_matches": bool(ok),
        }
        if not ok:
            bad.append(name)
    for seed, rec in v1_base_checkpoint_pins().items():
        path = rec["checkpoint"]
        if not path.exists():
            out[f"base_s{seed}"] = {"path": track_rel(path), "exists": False, "sha256_matches": False}
            bad.append(f"base_s{seed}")
            continue
        actual = sha256_file(path)
        ok = actual == rec["checkpoint_sha256"]
        out[f"base_s{seed}"] = {
            "path": track_rel(path),
            "exists": True,
            "sha256": actual,
            "expected_sha256": rec["checkpoint_sha256"],
            "sha256_matches": bool(ok),
            "readonly": True,
        }
        if not ok:
            bad.append(f"base_s{seed}")
    return {"pins": out, "all_matched": not bad, "mismatched": bad}


def require_freeze() -> dict[str, Any]:
    if not PROTOCOL_FREEZE.exists():
        raise PermissionError("protocol freeze missing; run `run_esol2.py freeze` first")
    frozen = read_json(PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("protocol freeze has the wrong protocol id")
    if frozen.get("config_sha256") != config_sha256():
        raise PermissionError("config changed after the freeze")
    if frozen.get("code_hashes") != code_hashes():
        raise PermissionError("code changed after the freeze")
    if frozen.get("split_manifest_sha256") != sha256_file(SPLIT_MANIFEST):
        raise PermissionError("split manifest changed after the freeze")
    if frozen.get("base_replay", {}).get("passed") is not True:
        raise PermissionError("BASE replay did not pass; refusing to train")
    if frozen.get("p0_status") != "PASS":
        raise PermissionError("P0 gates did not pass; refusing to train")
    return frozen


def freeze_protocol(*, p0: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    if p0.get("status") != "PASS":
        raise PermissionError("refusing to freeze: P0 gates are not PASS")
    if replay.get("passed") is not True:
        raise PermissionError("refusing to freeze: BASE replay failed")
    cfg = load_config()
    payload = {
        "protocol_id": PROTOCOL_ID,
        "round_id": ROUND_ID,
        "created_utc": utcnow(),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "status": "FROZEN_BEFORE_NEW_TRAINING",
        "scope": (
            "ESOL / Delaney (1,128 molecules), the frozen lcrc_esol_quickstart_v1 "
            "scaffold partition (seed 20261103), two independent experiments "
            "(PHYSCHEM6 on rho, capacity-matched typed-S2-blind E)"
        ),
        "split_seed": SPLIT_SEED,
        "split_manifest": track_rel(SPLIT_MANIFEST),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "split_audit_sha256": sha256_file(SPLIT_AUDIT),
        "input_pins": {
            "csv_sha256": sha256_file(CSV_PATH),
            "raw_graph_sha256": sha256_file(RAW_GRAPH),
            "physchem_sha256": sha256_file(PHYSCHEM_FILE),
            "lcrc_cache_content_fingerprint": LCRC_CACHE_CONTENT_FINGERPRINT,
        },
        "config": track_rel(CONFIG_PATH),
        "config_sha256": config_sha256(),
        "code_hashes": code_hashes(),
        "training_recipe": dict(TRAIN_RECIPE),
        "model_dropout": dict(MODEL_DROPOUT),
        "models": ["base", "physchem6", "s2blind_e"],
        "new_arms": list(NEW_ARMS),
        "seeds": list(SEEDS),
        "checkpoint_rule": "single_best_validation_rmse_epoch",
        "base_checkpoints": {
            f"base_s{s}": {"sha256": rec["checkpoint_sha256"], "reused_not_retrained": True}
            for s, rec in v1_base_checkpoint_pins().items()
        },
        "p0_status": str(p0["status"]),
        "p0_gates_sha256": sha256_file(P0_GATES),
        "base_replay": {
            "passed": bool(replay["passed"]),
            "per_seed": {
                f"s{seed}": {
                    "replayed_valid_rmse": rec["replayed_valid_rmse"],
                    "recorded_valid_rmse": rec["recorded_valid_rmse"],
                    "abs_diff": rec["abs_diff"],
                }
                for seed, rec in replay["per_seed"].items()
            },
            "tolerance": BASE_REPLAY_TOL,
        },
        "test_policy": (
            "NO new Test evaluation in this round. The old round's Test split has already "
            "been read once and analysed, so it is not a blind set here. Validation "
            "(113 molecules, already used by several rounds) is the only comparison set "
            "and it is explicitly not a fresh confirmation set."
        ),
        "planned_runs": {"physchem6": 3, "s2blind_e": 3, "hard_max": 6, "base_retrained": 0},
        "gpu_hard_cap_hours": GPU_HARD_CAP_HOURS,
        "writes_to_v1_directory": False,
    }
    payload["freeze_sha256"] = sha256_json({k: v for k, v in payload.items() if k != "freeze_sha256"})
    write_json(PROTOCOL_FREEZE, payload)
    return payload


def batch_order_sha256(n_molecules: int, epochs: int, batch_size: int, seed: int) -> str:
    """Replay the exact batch order of the frozen recipe (no model needed)."""
    import torch

    generator = torch.Generator().manual_seed(int(seed) + SHUFFLE_OFFSET)
    digest = hashlib.sha256()
    for _ in range(int(epochs)):
        perm = torch.randperm(int(n_molecules), generator=generator).tolist()
        for start in range(0, len(perm), int(batch_size)):
            digest.update(np.ascontiguousarray(np.asarray(perm[start : start + batch_size], dtype=np.int64)).tobytes())
    return digest.hexdigest()


__all__ = [name for name in dir() if not name.startswith("_")]
