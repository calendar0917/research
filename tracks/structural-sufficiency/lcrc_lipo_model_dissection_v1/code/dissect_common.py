"""Shared constants, paths, frozen hashes and the variant registry.

This round is a *dissection* of the frozen LCRC-A Lipophilicity model
(``public122 FULL seed0`` reference: Valid RMSE 0.7108803968, best epoch 240).
It re-implements the exact computational graph in this directory so that every
branch can be switched off/on explicitly, while the frozen upstream modules
(``lcrc_cross_task_preparation_v1`` and ``lcrc_molnet_benchmark_pilot_v1``) stay
read-only.

Nothing here is fitted, selected or tuned.  All scientific definitions
(repeats, seeds, variants, parameter counts, selection tolerances) are frozen
before the first formal run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROUND_ID = "lcrc-lipo-model-dissection-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
PILOT = REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1"
PILOT_CODE = PILOT / "code"

#: Frozen upstream modules consumed read-only (hashed into every run record).
REUSED_MODULES = (
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/feature_spec.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/molecular_graph.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/adapters.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_data.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_model.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_common.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_common.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_source.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_rdkit.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_raw_artifact.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_data.py",
)

for _p in (str(PREP_CODE), str(PILOT_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR = PILOT / "analysis/source"
CHEMBENCH_DIR = DATA_DIR / "chembench"
CHEMBENCH_LIPOP = CHEMBENCH_DIR / "Lipop"
SOURCE_MANIFEST = DATA_DIR / "chembench_source_manifest.json"

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
CACHE = TRACK / "cache"
TESTS = TRACK / "tests"
FREEZE = ANALYSIS / "freeze"

# ---------------------------------------------------------------------------
# frozen public data (ChemBench, commit 0b10d4c...) - hashes copied verbatim
# from the executed protocol; the round stops if they do not match the bytes.
# ---------------------------------------------------------------------------

CHEMBENCH_COMMIT = "0b10d4c1ac72b6929503694a5f60446824736391"
LIPOP_CSV = CHEMBENCH_LIPOP / "Lipop.csv.gz"
LIPOP_CSV_SHA256 = "34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758"

REPEATS = (122, 123, 124)
MAIN_REPEAT = 122
SEEDS = (0, 1, 2)

SPLIT_FILES = {
    122: "Lipop.random122.ind.pkl",
    123: "Lipop.random123.ind.pkl",
    124: "Lipop.random124.ind.pkl",
}
SPLIT_FILE_SHA256 = {
    122: "bf7acb177e080caf2be9ee220f479f8ea91e0039ff4867a7281753a2fce24a7d",
    123: "afaed4c6d95b64ac64e8fb760b6de516802008965befe1601b3664303119a0e8",
    124: "2b48a41ccc39b622bbb74f18331ca9d9513190eec2de29144ce2bc7d47ef7ef3",
}
SPLIT_ROWS_SHA256 = {
    122: (
        "4e8d42a95c0b705f43f932b3bbb265277c76a7aa65684b0e5e6ca028580aaf96",
        "31e3f28f31da69249f70922126f19795920bc1e32cc05d5e29e27507bd5aa365",
        "a4d92ad939bf268c00ff69afe77aa5d58c19a6f809b6a09f6dadc51cd648ec93",
    ),
    123: (
        "a3af8ce779db41a288e4e82a7d7ce3d07a556ecf8f2e54ef60f43ad480626db4",
        "a79e192e922d95bbfa9ff4be1564405d80a0d5ca4c1421fdb5abef5565170982",
        "6c335c844cb1a53273fb784de389d7df25281df86a057bbe22882f130aa06df2",
    ),
    124: (
        "dacdee79f1e34d258c55e3bdc6b7b87ecf9bd63039516b64df9f2b2a68cbd887",
        "e2100b1b236d236f7a3542b057f9026ff77f7bb920aeb662f64f924c3991ef94",
        "c4a5b71d8245ab426cffe475cd8862e11b16106d303b3ddb5dc409611fd0427d",
    ),
}
SPLIT_SIZES = (3360, 420, 420)
LABEL_COLUMN = "exp"
N_MOLECULES = 4200

#: Frozen feature-spec hash (lipo_v1) and input widths.
FEATURE_SPEC_HASH = "7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee"
ATOM_FEAT_DIM = 51
BOND_FEAT_DIM = 13
ATOM_SCALAR_DIM = 9
GLOBAL_DIM = 42
N_NBR_CODES = 80
N_PATH_CODES = 400
N_BOND_TYPES = 5

#: FULL local-input layout (zero-based, half-open), frozen by the protocol.
LOCAL_LAYOUT = {
    "a": (0, 96),
    "b_sum": (96, 192),
    "b_mean": (192, 288),
    "s1_joint": (288, 320),
    "s2_type": (320, 352),
    "s2_end": (352, 384),
    "s2_joint": (384, 416),
    "s2_closure": (416, 424),
    "s2_content": (424, 520),
    "atom_scalar": (520, 529),
}
LOCAL_DIM = 529
RHO_LAYOUT = {
    "g": (0, 42),
    "mean_h": (42, 106),
    "mean_h2": (106, 170),
    "log1p_n": (170, 171),
}
RHO_DIM = 171

#: Frozen R3 recipe (identical to ``mb_common.RECIPE``; asserted in tests).
RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-3,
    "batch_size": 128,
    "dropout": 0.10,
    "dropout_position": "final atom-level local representation h, before the per-atom contribution f and the global rho",
    "gradient_clip": 5.0,
    "epochs": 360,
    "scheduler": "per_step_cosine",
    "shuffle_offset": 917_000,
    "soup": False,
    "ema": False,
    "swa": False,
    "ssl": False,
    "early_stopping": False,
    "hyperparameter_search": False,
}
SHUFFLE_OFFSET = RECIPE["shuffle_offset"]

#: Frozen public122 FULL seed0 reference (read-only replay target).
REFERENCE = {
    "checkpoint": (
        "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/"
        "analysis/runs/Lipophilicity_s0/checkpoint.pt"
    ),
    "best_epoch": 240,
    "valid_rmse": 0.7108803968155553,
    "test_rmse": 0.5727470871023582,
}

#: Budget (A100 GPU-hours, one A100 per run, two concurrent at most).
GPU_SOFT_CAP_HOURS = 6.0
GPU_HARD_CAP_HOURS = 8.0

#: Evaluation-sample bootstrap.
BOOTSTRAP_SEED = 20261102
BOOTSTRAP_REPLICATES = 2000

#: Practical (not statistical) simplification-purchase tolerances, in logD.
TOL_MEAN_DR = 0.010
TOL_MAX_DR = 0.020
TOL_MEAN_DA = 0.010

REPRO_TOL = 1e-5
CPU_EQUIV_TOL = 1e-6

#: Fixed display cases for the attribution walk-through (0-based ranks over
#: 122 Valid absolute errors, ties by CSV row id ascending).
ATTRIB_RANKS = (0, 1, 209, 210, 416, 417, 418, 419)


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_default(obj: Any):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def row_sha256(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return ""


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def env_info() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for rel in REUSED_MODULES:
        out[rel] = sha256_file(REPO / rel)
    return out


def reused_module_hashes() -> dict[str, str]:
    return {rel: sha256_file(REPO / rel) for rel in REUSED_MODULES}


# ---------------------------------------------------------------------------
# variant registry
# ---------------------------------------------------------------------------

#: Execution order fixed by the protocol.  ``COMPACT`` tie-break order is this
#: list order.
VARIANT_ORDER = (
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
)

STAGE_VARIANTS = {
    "A": ("A0", "A1", "A2", "A3", "A4", "A5"),
    "B": ("B0", "B1", "B2", "B3", "B4", "B5", "B6"),
    "C": ("C0", "C1", "C2", "C3", "C4"),
}

#: Expected trainable parameter counts (pre-registered; a mismatch is a bug).
EXPECTED_PARAMS = {
    "A0": 247106,
    "A1": 2817,
    "A2": 244418,
    "A3": 242881,
    "A4": 240193,
    "A5": 236034,
    "B0": 176514,
    "B1": 224546,
    "B2": 228274,
    "B3": 205714,
    "B4": 247106,
    "B5": 247106,
    "B6": 245362,
    "C0": 238402,
    "C1": 228674,
    "C2": 245378,
    "C3": 243010,
    "C4": 247106,
}

VARIANT_NAMES = {
    "A0": "FULL",
    "A1": "G_ONLY",
    "A2": "NO_G",
    "A3": "NO_ADD",
    "A4": "POOL_ONLY",
    "A5": "ADD_ONLY",
    "B0": "TYPED_S2_OFF",
    "B1": "NO_JOINT",
    "B2": "NO_CONTENT",
    "B3": "NO_JOINT_CONTENT",
    "B4": "MARGINAL_ALL",
    "B5": "MARGINAL_BT2",
    "B6": "NO_CLOSURE_CHANNELS",
    "C0": "NO_S1JOINT",
    "C1": "NO_BOND_MEAN",
    "C2": "NO_SCALARS",
    "C3": "NO_M2",
    "C4": "MEAN_ADD",
}

#: Machine-readable variant definitions.  Every field is an explicit branch.
#: ``local_cols``/``rho_cols`` are derived in :mod:`dissect_model`.
VARIANT_DEFS: dict[str, dict[str, Any]] = {
    "A0": dict(
        stage="A",
        narrative="full frozen LCRC-A",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "A1": dict(
        stage="A",
        narrative="standardised g -> 64 -> 1; no local encoder, no f",
        use_local=False, use_f=False, use_rho=True, add_bias=False,
        s1_joint=False, b_mean=False, scalar_cols=(),
        s2_type=False, s2_end=False, s2_joint=False, s2_closure=False, s2_content=False,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=False, use_mean_h2=False, use_log1p_n=False,
    ),
    "A2": dict(
        stage="A",
        narrative="L + rho_h(mean_h, mean_h2, log1p n); no raw g",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=False, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "A3": dict(
        stage="A",
        narrative="rho(g, mean_h, mean_h2, log1p n); full local encoding, no f",
        use_local=True, use_f=False, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "A4": dict(
        stage="A",
        narrative="rho_h(mean_h, mean_h2, log1p n); no raw g, no f",
        use_local=True, use_f=False, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=False, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "A5": dict(
        stage="A",
        narrative="L + learnable global constant b0; no rho",
        use_local=True, use_f=True, use_rho=False, add_bias=True,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=False, use_mean_h=False, use_mean_h2=False, use_log1p_n=False,
    ),
    "B0": dict(
        stage="B",
        narrative="remove the whole 200-d typed-S2 block and its learned modules",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=False, s2_end=False, s2_joint=False, s2_closure=False, s2_content=False,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "B1": dict(
        stage="B",
        narrative="remove only the 32-d S2_joint and path_u/path_v/joint_proj",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=False, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "B2": dict(
        stage="B",
        narrative="remove only the 96-d S2_content and path_gate",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=False,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "B3": dict(
        stage="B",
        narrative="remove S2_joint and S2_content (path_gate included)",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=False, s2_closure=True, s2_content=False,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "B4": dict(
        stage="B",
        narrative="same modules/parameters; joint+content replaced by per-root marginal products",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="marginal", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "B5": dict(
        stage="B",
        narrative="same modules/parameters; marginal products within (root, bt2) groups",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="marginal_bt2", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "B6": dict(
        stage="B",
        narrative="remove the 8-d closure embedding block and scalar closure_ratio",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=(0, 1, 3, 4, 5, 6, 7, 8),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=False, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "C0": dict(
        stage="C",
        narrative="remove the 32-d S1joint block and nbr_emb",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=False, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "C1": dict(
        stage="C",
        narrative="remove the 96-d b_mean block (b_sum retained)",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=False, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "C2": dict(
        stage="C",
        narrative="remove all nine atom_scalar inputs",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=(),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
    "C3": dict(
        stage="C",
        narrative="remove mean_h2 from the rho input",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="sum",
        use_g=True, use_mean_h=True, use_mean_h2=False, use_log1p_n=True,
    ),
    "C4": dict(
        stage="C",
        narrative="L = mean_v f(h_v) instead of sum_v f(h_v)",
        use_local=True, use_f=True, use_rho=True, add_bias=False,
        s1_joint=True, b_mean=True, scalar_cols=tuple(range(9)),
        s2_type=True, s2_end=True, s2_joint=True, s2_closure=True, s2_content=True,
        s2_agg="full", add_reduction="mean",
        use_g=True, use_mean_h=True, use_mean_h2=True, use_log1p_n=True,
    ),
}

for _k in VARIANT_ORDER:
    assert VARIANT_DEFS[_k]["stage"] == _k[0], _k

#: stage -> variants, in frozen order (used by the sweep driver)
VARIANT_STAGE: dict[str, tuple[str, ...]] = {
    _s: tuple(_v for _v in VARIANT_ORDER if VARIANT_DEFS[_v]["stage"] == _s) for _s in ("A", "B", "C")
}


def run_name(repeat: int, variant: str, seed: int) -> str:
    return f"r{int(repeat)}_{variant}_s{int(seed)}"


def run_dir(repeat: int, variant: str, seed: int) -> Path:
    return RUNS / run_name(repeat, variant, seed)


def variant_cols(variant: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (local_cols, rho_cols) in the frozen FULL layouts."""
    v = VARIANT_DEFS[variant]
    local: list[int] = []
    if v["use_local"]:
        for key in ("a", "b_sum"):
            lo, hi = LOCAL_LAYOUT[key]
            local.extend(range(lo, hi))
        if v["b_mean"]:
            lo, hi = LOCAL_LAYOUT["b_mean"]
            local.extend(range(lo, hi))
        if v["s1_joint"]:
            lo, hi = LOCAL_LAYOUT["s1_joint"]
            local.extend(range(lo, hi))
        if v["s2_type"]:
            lo, hi = LOCAL_LAYOUT["s2_type"]
            local.extend(range(lo, hi))
        if v["s2_end"]:
            lo, hi = LOCAL_LAYOUT["s2_end"]
            local.extend(range(lo, hi))
        if v["s2_joint"]:
            lo, hi = LOCAL_LAYOUT["s2_joint"]
            local.extend(range(lo, hi))
        if v["s2_closure"]:
            lo, hi = LOCAL_LAYOUT["s2_closure"]
            local.extend(range(lo, hi))
        if v["s2_content"]:
            lo, hi = LOCAL_LAYOUT["s2_content"]
            local.extend(range(lo, hi))
        for i in v["scalar_cols"]:
            local.append(LOCAL_LAYOUT["atom_scalar"][0] + int(i))
    rho: list[int] = []
    if v["use_rho"]:
        for key, flag in (
            ("g", "use_g"),
            ("mean_h", "use_mean_h"),
            ("mean_h2", "use_mean_h2"),
            ("log1p_n", "use_log1p_n"),
        ):
            if v[flag]:
                lo, hi = RHO_LAYOUT[key]
                rho.extend(range(lo, hi))
    return tuple(local), tuple(rho)


__all__ = [
    "ROUND_ID",
    "PROTOCOL_ID",
    "CODE_DIR",
    "TRACK",
    "REPO",
    "PREP",
    "PREP_CODE",
    "PILOT",
    "PILOT_CODE",
    "REUSED_MODULES",
    "DATA_DIR",
    "CHEMBENCH_DIR",
    "CHEMBENCH_LIPOP",
    "SOURCE_MANIFEST",
    "ANALYSIS",
    "CONFIGS",
    "RUNS",
    "CACHE",
    "TESTS",
    "FREEZE",
    "CHEMBENCH_COMMIT",
    "LIPOP_CSV",
    "LIPOP_CSV_SHA256",
    "REPEATS",
    "MAIN_REPEAT",
    "SEEDS",
    "SPLIT_FILES",
    "SPLIT_FILE_SHA256",
    "SPLIT_ROWS_SHA256",
    "SPLIT_SIZES",
    "LABEL_COLUMN",
    "N_MOLECULES",
    "FEATURE_SPEC_HASH",
    "ATOM_FEAT_DIM",
    "BOND_FEAT_DIM",
    "ATOM_SCALAR_DIM",
    "GLOBAL_DIM",
    "N_NBR_CODES",
    "N_PATH_CODES",
    "N_BOND_TYPES",
    "LOCAL_LAYOUT",
    "LOCAL_DIM",
    "RHO_LAYOUT",
    "RHO_DIM",
    "RECIPE",
    "SHUFFLE_OFFSET",
    "REFERENCE",
    "GPU_SOFT_CAP_HOURS",
    "GPU_HARD_CAP_HOURS",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_REPLICATES",
    "TOL_MEAN_DR",
    "TOL_MAX_DR",
    "TOL_MEAN_DA",
    "REPRO_TOL",
    "CPU_EQUIV_TOL",
    "ATTRIB_RANKS",
    "utcnow",
    "sha256_file",
    "sha256_bytes",
    "stable_json",
    "write_json",
    "read_json",
    "track_rel",
    "row_sha256",
    "log",
    "git_commit",
    "git_dirty",
    "env_info",
    "code_hashes",
    "reused_module_hashes",
    "VARIANT_ORDER",
    "VARIANT_STAGE",
    "STAGE_VARIANTS",
    "EXPECTED_PARAMS",
    "VARIANT_NAMES",
    "VARIANT_DEFS",
    "run_name",
    "run_dir",
    "variant_cols",
]
