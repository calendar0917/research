"""Shared constants, frozen paths, hashing and CPU accounting.

Round: ``lcrc-nonlocal-binding-bottleneck-v1``.

This module contains **no** scientific choice of its own: every number here is
copied from the pre-registered design (``PROTOCOL.md`` / ``configs/frozen_design.json``)
or from the frozen upstream modules listed in ``REUSED_MODULES``.

Read-only reuse
---------------
* ``lcrc_cross_task_preparation_v1/code/feature_spec.py``  -> ``lipo_v1`` spec.
* ``lcrc_cross_task_preparation_v1/code/lcrc_data.py``     -> ``build_molecule`` /
  ``collate`` / ``fit_norm`` (the production LCRC-A input assembler).
* ``lcrc_molnet_benchmark_pilot_v2/code/mb2_rdkit.py``     -> SMILES -> raw
  attributes -> ``lipo_v1`` graph (the production feature path).
* ``lcrc_lipo_model_dissection_v1/code/dissect_model.py``  -> ``DissectLCRC``
  (A0) and the frozen column-deletion map to C1.
* ``lcrc_predictive_trace_benchmark_v1/code/ptb_model.py`` -> ``PTBLCRC`` (the
  single-task forward delegates to the frozen parent unchanged).
* ``lcrc_taskaware_representation_pilot_v1/code/ta_collision.py`` -> reference
  pattern for the production-path consistency test (imported, not copied).

Nothing in those directories is edited.  No model is trained, and no GPU is
used, in this round.
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

ROUND_ID = "lcrc-nonlocal-binding-bottleneck-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]
STRUCT = REPO / "tracks/structural-sufficiency"

PREP = STRUCT / "lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
MOLNET_V1 = STRUCT / "lcrc_molnet_benchmark_pilot_v1"
MOLNET_V1_CODE = MOLNET_V1 / "code"
MOLNET_V2 = STRUCT / "lcrc_molnet_benchmark_pilot_v2"
MOLNET_V2_CODE = MOLNET_V2 / "code"
DISSECT = STRUCT / "lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"
PTB = STRUCT / "lcrc_predictive_trace_benchmark_v1"
PTB_CODE = PTB / "code"

for _p in (str(CODE_DIR), str(DISSECT_CODE), str(PTB_CODE), str(MOLNET_V2_CODE), str(MOLNET_V1_CODE), str(PREP_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dissect_common as DC  # noqa: E402  (frozen variant registry / layouts / paths)
import dissect_model as DM  # noqa: E402
import mb2_common as MC  # noqa: E402  (official OGB source root + task registry)
import mb2_data as MD  # noqa: E402  (frozen structure-cache loader)
import mb2_rdkit as RDK  # noqa: E402  (SMILES -> raw attributes, frozen)
import ptb_common as PC  # noqa: E402
import ptb_model as PM  # noqa: E402

import lcrc_common as PREP_COMMON  # noqa: E402
import lcrc_data as PREP_DATA  # noqa: E402
from feature_spec import FeatureSpec  # noqa: E402

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
TESTS = TRACK / "tests"
CACHE_DIR = TRACK / "cache"

#: Frozen upstream files consumed read-only.  Hashed into ``analysis/provenance.json``.
REUSED_MODULES = (
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/feature_spec.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_data.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_common.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/molecular_graph.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/adapters.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/code/mb2_rdkit.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/code/mb2_common.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/code/mb2_data.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py",
    "tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1/code/ptb_model.py",
    "tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1/code/ptb_common.py",
    "tracks/structural-sufficiency/lcrc_taskaware_representation_pilot_v1/code/ta_collision.py",
)

#: Frozen ``lipo_v1`` interface widths (asserted, never re-derived).
FEATURE_SPEC_HASH = DC.FEATURE_SPEC_HASH
GLOBAL_DIM = 42

# ---------------------------------------------------------------------------
# Phase A design (frozen)
# ---------------------------------------------------------------------------

RING_SIZE = 18
OXYGEN_INDEX = 0
NITROGEN_INDEX_A = 6
NITROGEN_INDEX_B = 8
EXPECTED_FORMULA = "C16H33NO"
EXPECTED_SMILES_A = "C1CCCCCNCCCCCOCCCCC1"
EXPECTED_SMILES_B = "C1CCCCNCCCCCCCOCCCC1"
EXPECTED_NO_DISTANCE_A = 6
EXPECTED_NO_DISTANCE_B = 8
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
FORWARD_VARIANTS = ("A0", "C1")
FORWARD_SEEDS = (0, 1, 2)
PREDICTION_TOL = 1e-5

# ---------------------------------------------------------------------------
# Phase B design (frozen)
# ---------------------------------------------------------------------------

#: Fixed six heavy-atom classes.  Order fixes the 21 unordered type pairs.
ELEMENT_CLASSES: tuple[str, ...] = ("C", "N", "O", "S", "X", "OTHER")
HALOGENS: frozenset[str] = frozenset({"F", "Cl", "Br", "I"})
#: ``TYPE_PAIRS[p] = (i, j)`` with ``i <= j``, enumerated i-major:
#: ``p = i * 6 - i * (i - 1) // 2 + (j - i)``.
TYPE_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (i, j) for i in range(len(ELEMENT_CLASSES)) for j in range(i, len(ELEMENT_CLASSES))
)
assert len(TYPE_PAIRS) == 21

#: Fixed distance bins: 5, 6, 7, 8, 9, >= 10.
DIST_BIN_EDGES: tuple[int, ...] = (5, 6, 7, 8, 9)
DIST_BIN_LOWER: tuple[int, ...] = (5, 6, 7, 8, 9, 10)
DIST_BIN_LABELS: tuple[str, ...] = ("5", "6", "7", "8", "9", ">=10")
MIN_PAIR_DISTANCE = 5
N_DIST_BINS = len(DIST_BIN_LABELS)
N_TYPE_PAIRS = len(TYPE_PAIRS)
N_PAIR_FEATURES = N_TYPE_PAIRS * N_DIST_BINS  # 126
PAIR_FEATURE_LAYOUT = tuple((p, k) for p in range(N_TYPE_PAIRS) for k in range(N_DIST_BINS))
assert N_PAIR_FEATURES == 126

#: Compare-three-models design (exactly three; no fourth model).
BRANCHES = ("BASE", "BASE_NULL", "BASE_REAL")

OUTER_SPLITS = 3
INNER_SPLITS = 2
SPLIT_SEED = 20261103
LOGREG = {
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": None,
    "fit_intercept": True,
    "tol": 1e-4,
    "max_iter": 4000,
    "random_state": 20261103,
}
C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0)
CLIP = 5.0
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_MAX_ATTEMPTS = 2000
BOOTSTRAP_SEED = 20261103

#: Decision thresholds (B7).  Never relaxed after seeing a result.
GATE_REAL_MINUS_NULL_AUC_MEAN = 0.015
GATE_REAL_MINUS_BASE_AUC_MEAN = 0.010
GATE_REAL_MINUS_NULL_AP_MEAN = 0.0
GATE_ALL_FOLD_DIFFS_POSITIVE = True

#: CPU budgets for this round.
CPU_SOFT_CAP_HOURS = 2.0
GPU_BUDGET_HOURS = 0.0

#: Frozen Lipophilicity train-only structural normalisation (the one the
#: executed dissection/PTB runs used).  Recomputing it is allowed; the
#: fingerprint is the gate.
LIPO_REPEAT = 122
LIPO_TRAIN_NORM_FINGERPRINT = "4e62dae70efa3be46ff8df631417abf051a95395de689e1b23bd0483ed1cc015"
LIPO_TRAIN_CACHE_FINGERPRINT = "d9a41ef2dd2f8ce8ebdd6bad2809bb604ad2b6c25f267a07e48e1db543f4148e"

#: Official ogbg-molhiv official split file (train only is ever opened).
OGB_TRAIN_SPLIT = "split/scaffold/train.csv.gz"
OGB_MAPPING = "mapping/mol.csv.gz"
OGB_GRAPH_LABEL = "raw/graph-label.csv.gz"
OGB_TOTAL_ROWS = 41127
OGB_TRAIN_ROWS = 32901


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
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def rows_sha256(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return ""


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def env_info() -> dict[str, Any]:
    import rdkit
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "pandas": __import__("pandas").__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "rdkit": rdkit.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": "cpu",
    }


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    return out


def reused_module_hashes() -> dict[str, str]:
    return {rel: sha256_file(REPO / rel) for rel in REUSED_MODULES}


def frozen_spec() -> FeatureSpec:
    """The frozen ``lipo_v1`` FeatureSpec (hash-checked against the executed one)."""
    spec = MD.frozen_spec()
    if spec.hash() != FEATURE_SPEC_HASH:
        raise ValueError(f"lipo_v1 spec hash {spec.hash()} != frozen {FEATURE_SPEC_HASH}")
    if spec.global_dim != GLOBAL_DIM:
        raise ValueError("global dimension changed")
    return spec


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# CPU ledger
# ---------------------------------------------------------------------------


class CpuLedger:
    """Wall-clock + process-CPU accounting for the CPU-only budget (2 h)."""

    def __init__(self) -> None:
        self.started = time.time()
        self.started_cpu = time.process_time()
        self.steps: list[dict[str, Any]] = []

    def step(self) -> "CpuLedger.Step":
        return CpuLedger.Step(self)

    class Step:
        def __init__(self, ledger: "CpuLedger") -> None:
            self.ledger = ledger
            self.t0 = time.time()
            self.c0 = time.process_time()

        def __enter__(self) -> "CpuLedger.Step":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            self.ledger.record(self.t0, self.c0, label=getattr(self, "label", ""), failed=exc_type is not None)
            return False

    def record(self, t0: float, c0: float, *, label: str = "", failed: bool = False) -> None:
        self.steps.append(
            {
                "label": label,
                "wall_seconds": round(time.time() - t0, 3),
                "cpu_seconds": round(time.process_time() - c0, 3),
                "failed": bool(failed),
            }
        )

    def timed(self, label: str):
        ledger = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.time()
                self.c0 = time.process_time()
                log(f"START {label}")
                return self

            def __exit__(self, exc_type, exc, tb):
                ledger.record(self.t0, self.c0, label=label, failed=exc_type is not None)
                step = ledger.steps[-1]
                if exc_type is None:
                    log(f"DONE  {label} wall={step['wall_seconds']}s cpu={step['cpu_seconds']}s")
                return False

        return _Ctx()

    def to_dict(self, **extra: Any) -> dict[str, Any]:
        wall = time.time() - self.started
        cpu = time.process_time() - self.started_cpu
        return {
            "round_id": ROUND_ID,
            "created_utc": utcnow(),
            "gpu_hours_used": 0.0,
            "gpu_budget_hours": GPU_BUDGET_HOURS,
            "cpu_wall_seconds_total": round(wall, 3),
            "cpu_process_seconds_total": round(cpu, 3),
            "cpu_wall_hours_total": round(wall / 3600.0, 6),
            "cpu_budget_hours": CPU_SOFT_CAP_HOURS,
            "budget_ok": bool(wall / 3600.0 <= CPU_SOFT_CAP_HOURS),
            "steps": self.steps,
            **extra,
        }


__all__ = [name for name in dir() if not name.startswith("_")]
