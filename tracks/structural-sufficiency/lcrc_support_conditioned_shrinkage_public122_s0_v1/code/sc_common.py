"""Frozen constants, paths and provenance helpers for the
support-conditioned shrinkage round.

Everything scientific in this module was written **before** any formal training
run of this round: the two candidate support keys, the frozen alpha rules, the
feasibility gates (E1-E5), the key-priority rule, the five arms, the recipes and
the selection rules.  It mirrors ``configs/experiment.yaml`` and ``PROTOCOL.md``.

Read-only upstream sources (hashed into every run record):

* ``lcrc_performance_route_8h_public122_s0_v1`` -- the frozen C1 model, the
  ``build_support_v2`` molecule-support implementation (ERRATA E2) and the
  ``ShrinkLocalContent`` joint-block contraction precedent.
* ``lcrc_bottleneck_diagnosis_public122_s0_v1`` -- the update-driven trainer,
  the label-blind nested fit subsets, the shared structure/target norms, the
  RNG-safe evaluation, the 50-point selection grid and the bootstrap helpers.
* ``lcrc_lipo_model_dissection_v1`` -- ``LocalContent::_joint_agg`` /
  ``_content_agg``, ``marginal_joint_agg`` and the frozen FULL(seed 0) -> C1
  state remap.

Nothing here re-tunes the model, the recipe, the split or the corruption
strength: the only new degree of freedom is a *frozen structural prior* ``a_v``
that has no trainable parameters.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-support-conditioned-shrinkage-public122-s0-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
CACHE = TRACK / "cache"
RUNS = TRACK / "runs"
TESTS = TRACK / "tests"

PERF = REPO / "tracks/structural-sufficiency/lcrc_performance_route_8h_public122_s0_v1"
DIAG = REPO / "tracks/structural-sufficiency/lcrc_bottleneck_diagnosis_public122_s0_v1"
DIAG_CODE = DIAG / "code"
DISSECT_CODE = REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code"

for _p in (str(DIAG_CODE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lb_common as LC  # noqa: E402  (frozen diagnosis-round constants + sys.path setup)
import lb_data as LD  # noqa: E402
import lb_selection as LS  # noqa: E402

PC = LC.PC          # lcpr_common
PD = LC.PD          # lcpr_data
PM = LC.PM          # lcpr_model
PT = LC.PT          # lcpr_train
DM = LC.DM          # dissect_model
PP = LC.DC          # dissect_common

DATASETS = LC.DATASETS
PRIMARY_DATASET = LC.PRIMARY_DATASET
SPLITS = LC.SPLITS
TRAIN_SEED = LC.TRAIN_SEED
N_BOND_TYPES = int(PC.N_BOND_TYPES)
N_PATH_CODES = int(PC.N_PATH_CODES)
N_NBR_CODES = int(PC.N_NBR_CODES)
ATOM_FEAT_DIM = int(PC.ATOM_FEAT_DIM)
C1_PARAMS = int(PC.C1_PARAMS)
EVAL_BATCH = int(LC.EVAL_BATCH)

C1_R3 = dict(LC.C1_R3)          # Lipo recipe of this round (frozen upstream R3)
C1_R1 = dict(LC.C1_R1)          # BACE recipe of this round (frozen upstream R1)
SELECTION_RULES = tuple(LC.SELECTION_RULES)   # R_SEL (primary) / A_SEL / W_SEL
R_SEL = LC.R_SEL
A_SEL = LC.A_SEL
W_SEL = LC.W_SEL
GRID_POINTS = int(LC.GRID_POINTS)             # 50 candidate updates per phase
WINDOW = int(LC.WINDOW)
BOOTSTRAP_REPLICATES = int(LC.BOOTSTRAP_REPLICATES)
BOOTSTRAP_SEED = int(LC.BOOTSTRAP_SEED)

# ---------------------------------------------------------------------------
# support keys (exactly two, frozen priority)
# ---------------------------------------------------------------------------

KEY_FINE = "K_FINE"
KEY_JOINT = "K_JOINT"
#: fixed priority: try K_FINE first, then K_JOINT; never a third granularity.
KEY_PRIORITY = (KEY_FINE, KEY_JOINT)
KEY_DEFINITIONS = {
    KEY_FINE: ("canonical_float32(atom_feat[v], 51 cols)",
               "sorted tuple(c, e, D_v(c,e))"),
    KEY_JOINT: ("(no atom columns)", "sorted tuple(c, e, D_v(c,e))"),
}
DIGEST_BYTES = 32               # sha256; used as an index only
CANON_FLOAT_DTYPE = "<f4"       # explicit byte order, -0.0 folded to +0.0

#: canonical-structure group identity (support unit) from the published SMILES
GROUP_RDKIT_SPEC = "MolFromSmiles(published_smiles) -> MolToSmiles(canonical=True, isomericSmiles=True)"
FOLD_SALT = "lcrc-support-fold-v1"      # fold = int(sha256(salt|canonical),16) % 5
N_FOLDS = 5

# ---------------------------------------------------------------------------
# frozen gate constants (intervention legibility, not significance)
# ---------------------------------------------------------------------------

UNIFORM_A = 0.70                 # frozen uniform contraction dose
GATE_SLOPE = 0.60                # frozen slope of the support gate
GATE_RANGE = (0.40, 1.00)        # a_v range for SELECT / REVERSE
GATE_TOL = 1e-10                 # |weighted mean a - 0.70| tolerance

E1_MIN_ACTIVE_GROUPS = 100       # distinct canonical fit groups with >=1 active root
E1_MIN_ACTIVE_KEYS = 20          # distinct keys on active roots
E2_MIN_SPREAD_FULLFIT = 0.20     # weighted q90-q10 of a_SELECT, full fit table
E3_MIN_SPREAD_OOF = 0.20         # same on merged out-of-fold queries
E4_MAX_UNKNOWN_SHARE = 0.60      # weighted share of c == 0 on OOF queries
E5_MIN_LOW_SHARE = 0.10          # weighted share of c <= 2
E5_MIN_HIGH_SHARE = 0.10         # weighted share of c >= 5
LOW_C_MAX = 2
HIGH_C_MIN = 5

# ---------------------------------------------------------------------------
# arms (five, frozen) and the prediction-level rules
# ---------------------------------------------------------------------------

ARM_BASE = "BASE"
ARM_UNIFORM = "UNIFORM"
ARM_UNIFORM_OOF = "UNIFORM_OOF"
ARM_SELECT = "SELECT"
ARM_REVERSE = "REVERSE"
ARMS = (ARM_BASE, ARM_UNIFORM, ARM_UNIFORM_OOF, ARM_SELECT, ARM_REVERSE)
#: launch priority: BASE and SELECT first, then the two uniform arms and REVERSE
ARM_PRIORITY = {ARM_BASE: 1, ARM_SELECT: 1, ARM_UNIFORM: 2, ARM_UNIFORM_OOF: 2, ARM_REVERSE: 3}
ARM_README = {
    ARM_BASE: "a_v = 1 everywhere: the frozen C1 operator, no gate module",
    ARM_UNIFORM: "active roots a = 0.70, inactive a = 1 (constant dose)",
    ARM_UNIFORM_OOF: "active roots a = a_oof (train-structure-only average dose), inactive a = 1",
    ARM_SELECT: "a = clip(0.70 + 0.60*(F_T(c_T(K_v)) - 0.5), 0.40, 1.00)",
    ARM_REVERSE: "a = clip(0.70 - 0.60*(F_T(c_T(K_v)) - 0.5), 0.40, 1.00)",
}

#: pre-registered comparisons (all four are SELECT vs a control, gain > 0 is better)
SELECT_COMPARISONS = (
    (ARM_SELECT, ARM_BASE),
    (ARM_SELECT, ARM_UNIFORM),
    (ARM_SELECT, ARM_UNIFORM_OOF),
    (ARM_SELECT, ARM_REVERSE),
)

# ---------------------------------------------------------------------------
# promotion / purchase thresholds (budget rulers, not significance)
# ---------------------------------------------------------------------------

LIPO_PROMOTE_REL_RMSE = 0.01     # SELECT vs new BASE, both splits
LIPO_PROMOTE_MAE_TOL = 0.01      # MAE must not degrade more than 1 %
BACE_PROMOTE_AUC_ABS = 0.005     # BACE: AUC gain vs BASE and vs both uniforms + REVERSE
STATUS_DEV_TEST = "DEVELOPMENT_TEST"
TEST_USE = "TEST_USED_FOR_ROUTE_SELECTION"

# ---------------------------------------------------------------------------
# budget (this round)
# ---------------------------------------------------------------------------

WALL_TARGET_HOURS = 3.0
WALL_HARD_HOURS = 4.0
GPU_SOFT_CAP_HOURS = 2.0
GPU_HARD_CAP_HOURS = 3.0
MAX_FORMAL_RUNS = 15
MAX_SMOKE_JOBS = 8
SMOKE_MAX_UPDATES = 8
GPU_POOLS = ("res-gpu0", "res-gpu1")

#: fit fractions this round could purchase (Lipo only; BACE always 100)
LIPO_FITS = ("100", "25")

# ---------------------------------------------------------------------------
# artifact locations
# ---------------------------------------------------------------------------

ROOTKEYS_PATH = CACHE / "rootkeys_{dataset}.npz"
GROUPS_PATH = CACHE / "groups_{dataset}.npz"
SUPPORT_PATH = CACHE / "support_{dataset}_F{frac}_{key}.npz"
NORMS_REF = {"Lipophilicity": DIAG / "cache/target_norm_Lipophilicity.npz"}
STRUCT_NORM_REF = {d: DIAG / f"cache/struct_norm_{d}.npz" for d in DATASETS}
TARGET_NORM_REF = {d: DIAG / f"cache/target_norm_{d}.npz" for d in ("Lipophilicity",)}


def rootkeys_path(dataset: str) -> Path:
    return Path(str(ROOTKEYS_PATH).format(dataset=dataset))


def groups_path(dataset: str) -> Path:
    return Path(str(GROUPS_PATH).format(dataset=dataset))


def support_path(dataset: str, fit: str, key: str) -> Path:
    return Path(str(SUPPORT_PATH).format(dataset=dataset, frac=fit, key=key))


# ---------------------------------------------------------------------------
# hashing / IO (same conventions as the upstream rounds)
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(hashlib.sha256(data).digest(), dtype=np.uint8).copy()


def row_sha256(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _json_default(obj: Any):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
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
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n",
                 encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover
        return ""


def git_dirty(paths: Sequence[str] = ("code", "configs", "PROTOCOL.md")) -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no",
                                       "--", *paths], cwd=TRACK, text=True,
                                      stderr=subprocess.DEVNULL)
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
    }


CODE_PATHS = ("code", "configs", "PROTOCOL.md")


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.rglob("*.py")):
        out[track_rel(p)] = sha256_file(p)
    for p in sorted(TESTS.glob("*.py")):
        out[f"tests/{p.name}"] = sha256_file(p)
    for p in sorted(CONFIGS.glob("*.yaml")):
        out[f"configs/{p.name}"] = sha256_file(p)
    if (TRACK / "PROTOCOL.md").exists():
        out["PROTOCOL.md"] = sha256_file(TRACK / "PROTOCOL.md")
    # read-only upstream modules whose semantics this round depends on
    for base, tag in ((PERF / "code", PERF), (DIAG / "code", DIAG),
                      (REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code",
                       REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1")):
        for p in sorted(Path(base).glob("*.py")):
            out[track_rel(p)] = sha256_file(p)
    return out


# ---------------------------------------------------------------------------
# dataset / fit helpers (thin wrappers over the frozen diagnosis round)
# ---------------------------------------------------------------------------


def dataset_meta(dataset: str) -> dict[str, Any]:
    return LD.dataset_meta(dataset)


def official_train_n(dataset: str) -> int:
    return LC.official_train_n(dataset)


def u0(dataset: str) -> int:
    """C1 update horizon U0 = 360 * ceil(n_official_train / 128)."""
    return LC.u0("C1_R1", dataset)


def fit_positions(dataset: str, frac: str) -> list[int]:
    return list(LD.nested_subset_positions(dataset, frac)["positions"])


def shared_norms(dataset: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The frozen shared structure norm and the shared target norm (hashes recorded)."""
    snorm = LD.structure_norm(dataset)
    tnorm = LD.shared_target_norm(dataset)
    return snorm, tnorm


# ---------------------------------------------------------------------------
# the frozen alpha rules (one implementation, used by prepare and by training)
# ---------------------------------------------------------------------------


def alpha_rule(arm: str, c: np.ndarray, F_table: np.ndarray, a_oof: float | None) -> np.ndarray:
    """Frozen closed-form alpha for *active* roots.

    ``c`` are integer support counts ``c_T(K_v)``; ``F_table[j]`` is
    ``F_T(j)`` for ``j = 0 .. len(F_table)-1`` (``F_table[0] == 0``).  Inactive
    roots are set to 1.0 by the caller, never here.
    """
    c = np.asarray(c, dtype=np.int64)
    if arm == ARM_BASE:
        return np.ones(c.shape, dtype=np.float64)
    if arm == ARM_UNIFORM:
        return np.full(c.shape, UNIFORM_A, dtype=np.float64)
    if arm == ARM_UNIFORM_OOF:
        if a_oof is None:
            raise ValueError("UNIFORM_OOF needs the frozen a_oof")
        return np.full(c.shape, float(a_oof), dtype=np.float64)
    F = mid_cdf_lookup(c, F_table)
    signed = GATE_SLOPE * (F - 0.5)
    lo, hi = GATE_RANGE
    if arm == ARM_SELECT:
        a = UNIFORM_A + signed
    elif arm == ARM_REVERSE:
        a = UNIFORM_A - signed
    else:
        raise KeyError(arm)
    return np.clip(a, lo, hi)


def mid_cdf_lookup(x: np.ndarray, F_table: np.ndarray) -> np.ndarray:
    """Read ``F_T(x)`` from a precomputed table; ``x == 0`` maps to 0."""
    x = np.asarray(x, dtype=np.int64)
    out = np.zeros(x.shape, dtype=np.float64)
    ok = x > 0
    if np.any(ok):
        idx = np.clip(x[ok], 0, len(F_table) - 1)
        out[ok] = np.asarray(F_table, dtype=np.float64)[idx]
    return out


def build_mid_cdf(values: np.ndarray, weights: np.ndarray, n_support: int) -> np.ndarray:
    """Frozen fixed weighted mid-CDF on integer counts ``0..n_support``.

    ``F(x) = [sum_v w_v 1(c_v < x) + 0.5 sum_v w_v 1(c_v = x)] / W`` with
    ``W = sum_v w_v``.  Float64, no interpolation, ``F(0) = 0`` exactly.
    """
    values = np.asarray(values, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    W = float(weights.sum())
    table = np.zeros(int(n_support) + 1, dtype=np.float64)
    if W <= 0 or values.size == 0:
        return table
    hist = np.bincount(values, weights=weights, minlength=int(n_support) + 1)
    hist = hist[: int(n_support) + 1]
    below = np.concatenate([[0.0], np.cumsum(hist)[:-1]])   # weight with c < j
    table = (below + 0.5 * hist) / W
    table[0] = 0.0
    return table


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Smallest observation whose cumulative weight first reaches ``q`` (no interpolation)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(values) & (weights > 0)
    values, weights = values[good], weights[good]
    if values.size == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    if total <= 0:
        return float("nan")
    idx = int(np.searchsorted(cw, float(q) * total, side="left"))
    return float(v[min(idx, v.size - 1)])


def weighted_spread(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    q10 = weighted_quantile(values, weights, 0.10)
    q90 = weighted_quantile(values, weights, 0.90)
    q50 = weighted_quantile(values, weights, 0.50)
    return {"q10": q10, "q50": q50, "q90": q90, "spread": q90 - q10}


__all__ = [n for n in dir() if not n.startswith("_")]
