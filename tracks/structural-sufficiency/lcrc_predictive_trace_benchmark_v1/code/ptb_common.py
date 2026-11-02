"""Shared constants, paths, frozen recipe and hashes for the predictive-strength
& traceability benchmark round.

Round: ``lcrc-predictive-trace-benchmark-v1``.

Everything scientific (dataset list, seeds, epochs, folds, recipe, budget) is
declared here *before* any formal GPU run, and is copied verbatim into
``configs/protocol.json`` / ``analysis/protocol_freeze.json`` by
``ptb_protocol.py``.  Nothing here is chosen from a score.

Reused read-only
----------------
* ``lcrc_lipo_model_dissection_v1``: the A0/C1 computation graph, the frozen
  column-deletion mapping (``variant_cols`` / ``remap_state``) and the R3 recipe.
* ``lcrc_molnet_benchmark_pilot_v2``: ChemBench / OGB sources, structure caches,
  task metrics, missing-label handling and the official OGB evaluator.
* ``lcrc_tu_quick_screen_v1``: the TU feature adapter, the A0/C1 bodies and the
  graph-correctness checks.
* ``lcrc_cross_task_preparation_v1``: the ``lipo_v1`` FeatureSpec, graph builder
  and the frozen collate/normalisation layer.
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

ROUND_ID = "lcrc-predictive-trace-benchmark-v1"
PROTOCOL_ID = ROUND_ID
REFERENCE_HEAD = "9628ab4ca901aeecd2957b9e2c172a9fab485f7d"

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]
for _p in (str(CODE_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DISSECT = REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"
MOLNET_V2 = REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2"
MOLNET_V2_CODE = MOLNET_V2 / "code"
MOLNET_V1 = REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1"
TU_QS = REPO / "tracks/structural-sufficiency/lcrc_tu_quick_screen_v1"
TU_QS_CODE = TU_QS / "code"
EDIT = REPO / "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1"
PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"

for _p in (str(DISSECT_CODE), str(MOLNET_V2_CODE), str(MOLNET_V1 / "code"), str(TU_QS_CODE), str(PREP / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dissect_common as DC  # noqa: E402  (inserts the frozen prep/pilot paths)
import mb2_common as MC  # noqa: E402  (chembench/ogb sources + task registry)

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
CACHE = TRACK / "cache"
TESTS = TRACK / "tests"

#: every imported upstream file is hashed into ``analysis/source_manifest.json``
REUSED_MODULES: dict[str, Path] = {
    "dissect/dissect_common.py": DISSECT_CODE / "dissect_common.py",
    "dissect/dissect_model.py": DISSECT_CODE / "dissect_model.py",
    "dissect/dissect_data.py": DISSECT_CODE / "dissect_data.py",
    "dissect/dissect_source.py": DISSECT_CODE / "dissect_source.py",
    "molnet_v2/mb2_common.py": MOLNET_V2_CODE / "mb2_common.py",
    "molnet_v2/mb2_data.py": MOLNET_V2_CODE / "mb2_data.py",
    "molnet_v2/mb2_metrics.py": MOLNET_V2_CODE / "mb2_metrics.py",
    "molnet_v2/mb2_source.py": MOLNET_V2_CODE / "mb2_source.py",
    "molnet_v2/mb2_model.py": MOLNET_V2_CODE / "mb2_model.py",
    "tu/tu_common.py": TU_QS_CODE / "tu_common.py",
    "tu/tu_adapter.py": TU_QS_CODE / "tu_adapter.py",
    "tu/tu_data.py": TU_QS_CODE / "tu_data.py",
    "tu/tu_model.py": TU_QS_CODE / "tu_model.py",
    "prep/feature_spec.py": PREP / "code/feature_spec.py",
    "prep/molecular_graph.py": PREP / "code/molecular_graph.py",
    "prep/lcrc_data.py": PREP / "code/lcrc_data.py",
}

# ---------------------------------------------------------------------------
# models under test
# ---------------------------------------------------------------------------

MODEL_A0 = "A0"  # frozen LCRC-A (FULL)
MODEL_C1 = "C1"  # NO_BOND_MEAN, the pre-registered main model
MODELS = (MODEL_A0, MODEL_C1)
MODEL_LABELS = {MODEL_A0: "A0 / FULL", MODEL_C1: "C1 / NO_BOND_MEAN"}
#: single-task parameter counts (frozen); Tox21 widens both heads identically.
EXPECTED_PARAMS_SINGLE = {MODEL_A0: 247_106, MODEL_C1: 228_674}
EXPECTED_PARAM_DELTA = 18_432

MODEL_CONFIG = dict(__import__("dissect_model").DEFAULT_CONFIG)  # width 64, rank 32, local_hidden 192 ...

# ---------------------------------------------------------------------------
# experiment A: molecular tasks
# ---------------------------------------------------------------------------

MOLECULE_DATASETS: dict[str, dict[str, Any]] = {
    "ESOL": {"epochs": 360, "task": "regression", "primary_metric": "rmse", "direction": "lower"},
    "FreeSolv": {"epochs": 360, "task": "regression", "primary_metric": "rmse", "direction": "lower"},
    "BACE": {"epochs": 360, "task": "binary", "primary_metric": "roc_auc", "direction": "higher"},
    "BBBP": {"epochs": 360, "task": "binary", "primary_metric": "roc_auc", "direction": "higher"},
    "Tox21": {"epochs": 360, "task": "multitask", "primary_metric": "macro_roc_auc", "direction": "higher"},
    "ogbg-molhiv": {
        "epochs": 100,
        "task": "binary",
        "primary_metric": "roc_auc",
        "direction": "higher",
        "notes": "official OGB scaffold split + official evaluator; 100 epochs, never extended",
    },
}
MOLECULE_ORDER = ("ESOL", "FreeSolv", "BACE", "BBBP", "Tox21", "ogbg-molhiv")
MOLECULE_SEEDS = (0, 1, 2)
#: datasets that may be loaded for verification only (never trained in this round)
SUPPORT_VIEWS = ("Lipophilicity",)

#: Frozen R3 recipe (identical to the dissection round).  MolHIV only differs in
#: the epoch cap.
RECIPE: dict[str, Any] = dict(DC.RECIPE)
RECIPE["epochs_by_dataset"] = {k: int(v["epochs"]) for k, v in MOLECULE_DATASETS.items()}

RESULT_LABEL_MOL = "NEW_EXPOSED_TEST_EXPLORATORY"

# ---------------------------------------------------------------------------
# experiment B: TU graph classification (cross-validation)
# ---------------------------------------------------------------------------

TU_DATASETS = ("NCI1", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI")
TU_EPOCHS = 200
TU_SEED_POOL = (0, 1, 2)
TU_RESULT_LABEL = "NEW_INTERNAL_CV"

#: Candidate CV schemes, in the protocol's priority order.  Which one is executed
#: is chosen *only* from measured wall-clock/GPU cost (see ``ptb_protocol.py``).
TU_SCHEMES: dict[str, dict[str, Any]] = {
    "A_10x2": {"n_splits": 10, "seeds": (0, 1), "fold_seed": 20261102},
    "B_5x3": {"n_splits": 5, "seeds": (0, 1, 2), "fold_seed": 20261102},
    "C_5x2": {"n_splits": 5, "seeds": (0, 1), "fold_seed": 20261102, "degraded": True},
}
TU_INNER_VALID_FRACTION = 0.10
TU_INNER_VALID_SEED_BASE = 20261102  # + fold_id

# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

GPU_SOFT_CAP_HOURS = 14.0
GPU_HARD_CAP_HOURS = 16.0
WALL_CLOCK_CAP_HOURS = 8.0
REPORTING_RESERVE_HOURS = 1.0
#: reserve for cache rebuilds / scheduling jitter, added to measured estimates
COST_SAFETY_FACTOR = 1.20
#: protocol-mandated explicit reserve (A100 GPU-hours)
EXPLICIT_RESERVE_GPU_HOURS = 1.0

STATS_BOOTSTRAP_SEED = 20261102
STATS_BOOTSTRAP_REPLICATES = 2000

# ---------------------------------------------------------------------------
# frozen historical artefacts consumed read-only (contrast tables AFTER errata)
# ---------------------------------------------------------------------------


def _safe_rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


HISTORICAL = {
    "lipo_dissection_protocol": DISSECT / "analysis/protocol_freeze.json",
    "lipo_dissection_cpu_audit": DISSECT / "analysis/audit_cpu.json",
    "lipo_dissection_tables": DISSECT / "analysis/tables/summary.json",
    "lipo_dissection_compact": DISSECT / "analysis/compact_selection.json",
    "lipo_random42_controls": REPO / "tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2",
    "lipo_mechanism_ssl": REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2",
    "lipo_dissection_followup": REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_followup_v1",
    "zinc_dccl_bottleneck": REPO / "tracks/structural-sufficiency/zinc_dccl_bottleneck_v1",
    "zinc_error_mechanism": REPO / "tracks/structural-sufficiency/zinc_error_mechanism_audit_v1",
    "zinc_performance_search": REPO / "tracks/structural-sufficiency/zinc_performance_search_v1",
    "performance_longrun": REPO / "tracks/structural-sufficiency/lcrc_performance_longrun_v1",
    "molnet_v2_literature": MOLNET_V2 / "analysis/literature_matrix.csv",
    "molnet_v2_benchmark_table": MOLNET_V2 / "analysis/benchmark_table.csv",
    "molnet_v2_summary": MOLNET_V2 / "analysis/summary.json",
}


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def index_sha256(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def log(msg: str) -> None:
    print(f"[{ROUND_ID}] {utcnow()} {msg}", flush=True)


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
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def env_info() -> dict[str, Any]:
    import torch

    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        out["cuda_device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        out["cuda_driver"] = getattr(torch.cuda, "driver_version", lambda: None)()
    return out


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for name, path in REUSED_MODULES.items():
        out[f"reused/{name}"] = sha256_file(path)
    return out


def reused_module_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in REUSED_MODULES.items()}


# ---------------------------------------------------------------------------
# run identity
# ---------------------------------------------------------------------------


def mol_run_name(dataset: str, model: str, seed: int) -> str:
    return f"mol_{dataset}_s{int(seed)}_{model}"


def mol_run_dir(dataset: str, model: str, seed: int) -> Path:
    return RUNS / mol_run_name(dataset, model, seed)


def tu_run_name(dataset: str, model: str, fold: int, seed: int) -> str:
    return f"tu_{dataset}_f{int(fold)}_s{int(seed)}_{model}"


def tu_run_dir(dataset: str, model: str, fold: int, seed: int) -> Path:
    return RUNS / tu_run_name(dataset, model, fold, seed)


def mol_units(seeds: Sequence[int] = MOLECULE_SEEDS, datasets: Sequence[str] = MOLECULE_ORDER) -> list[tuple[str, str, int]]:
    return [(d, m, int(s)) for d in datasets for s in seeds for m in MODELS]


def tu_units(
    scheme: Mapping[str, Any], datasets: Sequence[str] = TU_DATASETS
) -> list[tuple[str, str, int, int]]:
    folds = range(int(scheme["n_splits"]))
    out: list[tuple[str, str, int, int]] = []
    for d in datasets:
        for f in folds:
            for s in scheme["seeds"]:
                for m in MODELS:
                    out.append((d, m, int(f), int(s)))
    return out


# ---------------------------------------------------------------------------
# compute ledger
# ---------------------------------------------------------------------------

LEDGER_PATH = ANALYSIS / "compute_ledger.json"


def ledger_read() -> dict[str, Any]:
    if LEDGER_PATH.exists():
        return read_json(LEDGER_PATH)
    return {
        "schema": "lcrc-ptb-compute-ledger-v1",
        "round_id": ROUND_ID,
        "gpu_soft_cap_hours": GPU_SOFT_CAP_HOURS,
        "gpu_hard_cap_hours": GPU_HARD_CAP_HOURS,
        "wall_clock_cap_hours": WALL_CLOCK_CAP_HOURS,
        "units": [],
        "total_gpu_hours": 0.0,
        "started_utc": utcnow(),
    }


def ledger_total() -> float:
    return float(ledger_read().get("total_gpu_hours", 0.0))


def ledger_add(unit: Mapping[str, Any]) -> None:
    payload = ledger_read()
    payload["units"].append(dict(unit))
    payload["total_gpu_hours"] = float(sum(float(u.get("gpu_hours", 0.0)) for u in payload["units"]))
    payload["updated_utc"] = utcnow()
    write_json(LEDGER_PATH, payload)


def elapsed_hours_since_start() -> float:
    payload = ledger_read()
    start = payload.get("started_utc")
    if not start:
        return 0.0
    import calendar

    t = calendar.timegm(time.strptime(start, "%Y-%m-%dT%H:%M:%SZ"))
    return (time.time() - t) / 3600.0


__all__ = [n for n in dir() if not n.startswith("_")]
