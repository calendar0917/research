"""Shared constants, paths, frozen-config validation and read-only imports.

Scope of the round (``lcrc_lipo_proto_interact_pilot_v1``)
---------------------------------------------------------
One seed (0), one frozen scaffold fold (v2 Fold 2), Fit2352 only.  The round
compares five arms that differ *only* in the atom-contribution / interaction
readout:

    BASE_G0, BASE_G1     the frozen ``lcrc_model.LCRC`` (``lipo_v1``, width 64)
    PROTO_K16_TOP4       BASE with ``self.f`` replaced by a shared-prototype readout
    PAIR_TRUE            BASE + a graph-level residual over distance-3/4/5 atom pairs
    PAIR_NULL            PAIR_TRUE with each molecule's distance labels permuted

Nothing in the historical rounds is written by this module.  Every historical
artefact is consumed read-only through :mod:`v2_common` (which in turn only
reads the frozen ``lcrc_lipo_screen_v1`` / ``lcrc_cross_task_preparation_v1``
files).  Monitor504, Reserved504 and the random42 Validation420/Test420 splits
are never opened.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

V2 = REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2"
V2_CODE = V2 / "code"

ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
RUNS = ANALYSIS / "runs"
PREDICTIONS = ANALYSIS / "predictions"
CONFIG_PATH = CONFIGS / "pilot.yaml"

PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
GPU_AUTHORIZATION = ANALYSIS / "gpu_authorization.json"
DATA_AUDIT = ANALYSIS / "data_audit.json"
CPU_GATES = ANALYSIS / "cpu_gates.json"
SUMMARY = ANALYSIS / "summary.json"
GPU_LEDGER = ANALYSIS / "gpu_ledger.json"

ROUND_ID = "lcrc-lipo-proto-interact-pilot-v1"
PROTOCOL_ID = ROUND_ID

if str(V2_CODE) not in sys.path:
    sys.path.insert(0, str(V2_CODE))

import v2_common as V2C  # noqa: E402
import v2_data as V2D  # noqa: E402
import v2_folds as V2F  # noqa: E402
import v2_models as V2M  # noqa: E402,F401  (used by pilot_models)

prep_model = V2C.prep_model
prep_data = V2C.prep_data
feature_spec = V2C.feature_spec

# ---------------------------------------------------------------------------
# frozen design constants (mirrored in configs/pilot.yaml and re-validated)
# ---------------------------------------------------------------------------

SEED = 0
FOLD = 2
N_TRAIN = 1568
N_DEV = 784

BATCH_SIZE = 128
EPOCHS = 360
SOUP_LAST = 5
GRAD_CLIP = 5.0
PEAK_LR = 3e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-3
DROPOUT = 0.10
SHUFFLE_OFFSET = 917_000
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8

PROTO_NAME = "PROTO_K16_TOP4"
N_PROTOTYPES = 16
PROTO_TOPK = 4
PROTO_TAU = 0.25

PAIR_DISTANCES = (3, 4, 5)
PAIR_Z_DIM = 16
PAIR_HIDDEN = 32
PAIR_NULL_SALT = "lipo-pair-null-v1"

PROTO_STRONG_GAIN = 0.020
PAIR_STRONG_GAIN = 0.020
PAIR_VS_NULL_GAIN = 0.010

GPU_HARD_CAP_HOURS = 0.5

ARMS = ("BASE_G0", "BASE_G1", PROTO_NAME, "PAIR_TRUE", "PAIR_NULL")
BASE_ARMS = ("BASE_G0", "BASE_G1")
PAIR_ARMS = ("PAIR_TRUE", "PAIR_NULL")
PROTO_ARMS = (PROTO_NAME,)

#: Fixed GPU plan.  ``BASE_G0``/``PROTO`` run sequentially on GPU 0,
#: ``BASE_G1``/``PAIR_TRUE``/``PAIR_NULL`` sequentially on GPU 1.  Inside each
#: process the visible device is ``cuda:0`` (rr sets ``CUDA_VISIBLE_DEVICES``).
ARM_POOL = {
    "BASE_G0": "res-gpu0",
    PROTO_NAME: "res-gpu0",
    "BASE_G1": "res-gpu1",
    "PAIR_TRUE": "res-gpu1",
    "PAIR_NULL": "res-gpu1",
}
ARM_BASELINE = {
    PROTO_NAME: "BASE_G0",
    "PAIR_TRUE": "BASE_G1",
    "PAIR_NULL": "PAIR_TRUE",
    "BASE_G1": "BASE_G0",
}

COMPARISONS = (
    ("BASE_G0", PROTO_NAME),
    ("BASE_G1", "PAIR_TRUE"),
    ("PAIR_TRUE", "PAIR_NULL"),
    ("BASE_G0", "BASE_G1"),
)

# ---------------------------------------------------------------------------
# hashing / json helpers (delegated to the historical, unchanged helpers)
# ---------------------------------------------------------------------------

sha256_file = V2C.sha256_file
sha256_text = V2C.sha256_text
read_json = V2C.read_json
write_json = V2C.write_json
stable_json = V2C.stable_json
track_rel = V2C.track_rel
utcnow = V2C.utcnow
git_commit = V2C.git_commit
git_dirty = V2C.git_dirty
ensure_dir = V2C.ensure_dir
log = V2C.log
file_record = V2C.file_record
env_info = V2C.env_info

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def config_hash(cfg: Mapping[str, Any]) -> str:
    return sha256_text(stable_json(cfg))


def _values(cfg: Mapping[str, Any]):
    tr = cfg["training"]
    yield "seed", int(cfg["seed"]), SEED
    yield "fold", int(cfg["fold"]), FOLD
    yield "data.n_train", int(cfg["data"]["n_train"]), N_TRAIN
    yield "data.n_dev", int(cfg["data"]["n_dev"]), N_DEV
    yield "training.peak_lr", float(tr["peak_lr"]), PEAK_LR
    yield "training.min_lr", float(tr["min_lr"]), MIN_LR
    yield "training.weight_decay", float(tr["weight_decay"]), WEIGHT_DECAY
    yield "training.batch_size", int(tr["batch_size"]), BATCH_SIZE
    yield "training.epochs", int(tr["epochs"]), EPOCHS
    yield "training.grad_clip", float(tr["grad_clip"]), GRAD_CLIP
    yield "training.dropout", float(tr["dropout"]), DROPOUT
    yield "training.soup_last", int(tr["soup_last"]), SOUP_LAST
    yield "training.shuffle_offset", int(tr["shuffle_offset"]), SHUFFLE_OFFSET
    yield "training.optimizer", str(tr["optimizer"]), "AdamW"
    yield "training.scheduler", str(tr["scheduler"]), "per_step_cosine"
    yield "training.loss", str(tr["loss"]), "mse_on_fold_standardised_target"
    yield "training.early_stopping", bool(tr["early_stopping"]), False
    yield "training.dev_checkpoint_selection", bool(tr["dev_checkpoint_selection"]), False
    yield "model.width", int(cfg["model"]["width"]), 64
    yield "proto.n_prototypes", int(cfg["proto"]["n_prototypes"]), N_PROTOTYPES
    yield "proto.topk", int(cfg["proto"]["topk"]), PROTO_TOPK
    yield "proto.tau", float(cfg["proto"]["tau"]), PROTO_TAU
    yield "proto.aux_losses", str(cfg["proto"]["aux_losses"]), "none"
    yield "pair.distances", tuple(int(x) for x in cfg["pair"]["distances"]), PAIR_DISTANCES
    yield "pair.z_dim", int(cfg["pair"]["z_dim"]), PAIR_Z_DIM
    yield "pair.hidden", int(cfg["pair"]["hidden"]), PAIR_HIDDEN
    yield "pair.aggregation", str(cfg["pair"]["aggregation"]), "mean_over_pairs"
    yield "pair.null_salt", str(cfg["pair"]["null_salt"]), PAIR_NULL_SALT
    yield "decision.proto_strong_gain", float(cfg["decision"]["proto_strong_gain"]), PROTO_STRONG_GAIN
    yield "decision.pair_strong_gain", float(cfg["decision"]["pair_strong_gain"]), PAIR_STRONG_GAIN
    yield "decision.pair_vs_null_gain", float(cfg["decision"]["pair_vs_null_gain"]), PAIR_VS_NULL_GAIN
    yield "budget.gpu_hard_cap_hours", float(cfg["budget"]["gpu_hard_cap_hours"]), GPU_HARD_CAP_HOURS


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"config protocol_id {cfg.get('protocol_id')!r} != {PROTOCOL_ID!r}")
    if cfg.get("round_id") != ROUND_ID:
        raise ValueError("config round_id drift")
    for name, actual, expected in _values(cfg):
        if actual != expected:
            raise ValueError(f"config drift at {name}: {actual!r} != frozen {expected!r}")
    if tuple(str(a) for a in cfg["arms"]) != ARMS:
        raise ValueError(f"config arms {cfg['arms']} != {list(ARMS)}")
    if tuple(cfg["pair"]["head"]) != ("Linear(48,32)", "SiLU", "Linear(32,1)"):
        raise ValueError("config pair head drift")
    if str(cfg["pair"]["last_layer_init"]) != "weight:normal(0,1e-3);bias:0":
        raise ValueError("config pair last-layer init drift")
    plan = {str(row["key"]): str(row["pool"]) for row in cfg["runs"]}
    if plan != ARM_POOL:
        raise ValueError(f"config GPU plan {plan} != {ARM_POOL}")


# ---------------------------------------------------------------------------
# folded data
# ---------------------------------------------------------------------------


def load_fold_view() -> Any:
    """Load the frozen v2 Fold-2 view (Fit2352 only, Train-only normalisation)."""
    cache = V2C.load_fit_cache()["cache"]
    fit = V2C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)
    if y.shape != (2352,):
        raise ValueError(f"Fit label vector has shape {y.shape}")
    folds = V2F.load_folds()
    view = V2D.build_fold_view(FOLD, cache=cache, y_fit=y, folds=folds)
    if len(view.train_positions) != N_TRAIN or len(view.dev_positions) != N_DEV:
        raise ValueError("fold view does not have the frozen 1568/784 split")
    if set(view.train_positions) & set(view.dev_positions):
        raise ValueError("fold train/dev overlap")
    if int(view.target_norm["n"]) != N_TRAIN:
        raise ValueError("target normalisation was not computed from the train fold")
    return view


def folds_payload() -> dict[str, Any]:
    return V2F.load_folds()


def row_meta(folds: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """global_row_id -> fold row record (scaffold, fit cache position, ...)."""
    return {int(k): v for k, v in folds["rows"].items()}


def dev_row_table(view: Any) -> list[dict[str, Any]]:
    """Per-molecule identity for the Dev fold, in cache order."""
    folds = folds_payload()
    rows = row_meta(folds)
    out = []
    for pos, row in enumerate(view.dev_positions):
        rec = rows[int(int(view.dev_cache["molecule_ids"][pos]))]
        if int(rec["fit_cache_position"]) != int(row):
            raise ValueError("fold row record does not match the Dev cache position")
        out.append(
            {
                "global_row_id": int(rec["global_row_id"]),
                "fit_cache_position": int(rec["fit_cache_position"]),
                "scaffold": str(rec["scaffold"]),
                "group_key_sha256": str(rec["group_key_sha256"]),
            }
        )
    return out


def train_row_table(view: Any) -> list[dict[str, Any]]:
    folds = folds_payload()
    rows = row_meta(folds)
    out = []
    for pos, row in enumerate(view.train_positions):
        rec = rows[int(int(view.train_cache["molecule_ids"][pos]))]
        if int(rec["fit_cache_position"]) != int(row):
            raise ValueError("fold row record does not match the Train cache position")
        out.append(
            {
                "global_row_id": int(rec["global_row_id"]),
                "fit_cache_position": int(rec["fit_cache_position"]),
                "scaffold": str(rec["scaffold"]),
                "group_key_sha256": str(rec["group_key_sha256"]),
            }
        )
    return out


def steps_per_epoch(n_train: int = N_TRAIN, batch_size: int = BATCH_SIZE) -> int:
    return math.ceil(int(n_train) / int(batch_size))


def code_file_hashes() -> dict[str, str]:
    out = {}
    for path in sorted(list((EXP_DIR / "code").glob("*.py")) + list((EXP_DIR / "tests").glob("*.py"))):
        out[track_rel(path)] = sha256_file(path)
    out[track_rel(CONFIG_PATH)] = sha256_file(CONFIG_PATH)
    return out


def jsonable(obj: Any) -> Any:
    return V2C.stable_json(obj)  # raises on non-serialisable input; used as a guard


def state_dict_fingerprint(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()
