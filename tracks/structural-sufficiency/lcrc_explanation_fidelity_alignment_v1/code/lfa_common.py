"""Identity, frozen constants, paths, IO and statistics for
``lcrc-explanation-fidelity-alignment-v1``.

Every constant that can influence a reported number is listed in
:data:`FROZEN_CONSTANTS` and hashed into ``analysis/protocol_freeze.json``.
Nothing here decides a scientific value beyond the rules pre-registered in
``PROTOCOL.md``.

Read-only reuse: this module puts the frozen source rounds on ``sys.path`` and
never writes into them.  Model/checkpoint loading is delegated to
``xr_models`` so that the four frozen checkpoints keep one identity chain.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-explanation-fidelity-alignment-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

ANALYSIS_DIR = ROUND_DIR / "analysis"
RESULTS_DIR = ROUND_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"          # optional local copy
FIELDS_DIR = RESULTS_DIR / "fields"
POOL_DIR = RESULTS_DIR / "pool"
SELECT_DIR = RESULTS_DIR / "select"
REPLAY_DIR = RESULTS_DIR / "replay"

#: Read-only source rounds (imported, never written).
XR_DIR = TRACK_DIR / "lcrc_explanation_task_relevance_sprint_v1"
XR_CODE = XR_DIR / "code"
XR_RESULTS = XR_DIR / "results"
PILOT_DIR = TRACK_DIR / "lcrc_auto_subgraph_faithfulness_pilot_v1"
PILOT_CODE = PILOT_DIR / "code"
ROUTE_DIR = TRACK_DIR / "gnn_route_clarification_8h_v1"
ROUTE_CODE = ROUTE_DIR / "code"

DATASET = "ogbg-mollipo"
SPLITS: tuple[str, ...] = ("train", "valid", "test")
SPLIT_CODE = {"train": 0, "valid": 1, "test": 2}
EXPECTED_SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}

# ---------------------------------------------------------------------------
# main-experiment scope
# ---------------------------------------------------------------------------

#: Main WC_LOGP experiment: the complete official Valid/Test splits plus a
#: fixed descriptive Train subset (the inherited sha256 ordering rule).
MAIN_SPLITS: tuple[str, ...] = ("valid", "test")
DESCRIPTION_SPLIT = "train"
N_TRAIN_DESCRIPTION = 128
SCOPE_MAIN = "main"
SCOPE_AUX = "aux"

# ---------------------------------------------------------------------------
# frozen search / control / statistics constants
# ---------------------------------------------------------------------------

SAMPLE_SEED = 20261117
CONTROL_SEED = 20261117
BOOTSTRAP_SEED = 20261117
BOOTSTRAP_REPLICATES = 2000

SEARCH_MIN_K = 2
SEARCH_MAX_K = 6
BEAM_WIDTH = 12

N_RANDOM = 16
MIN_RANDOM = 8
MAX_GROWTH_ATTEMPTS = 512
DOSE_GATE = 0.5

FD_EPS = 1e-3
DOSE_PATH_STEPS = 64
GATE_CHECK_MOLECULES = 6

#: Pre-registered decision thresholds (see PROTOCOL.md §8).
ORACLE_ENRICHMENT_MIN = 0.02
FIDELITY_ADVANTAGE_STD_MIN = 0.01
PAIRED_IMPROVEMENT_MIN = 0.02
CONCENTRATION_TOP_FRACTION = 0.05
CONCENTRATION_MAX_SHARE = 0.50
MIN_EVALUABLE_FRACTION = 0.80
LOW_SUPPORT_COMPONENTS = 30

GENERATORS: tuple[str, ...] = ("LCRC_GRAD", "LCRC_LOCAL", "GINE_GRAD", "TOPO_S2")
SELECTORS: tuple[str, ...] = (
    "A_LCRC_GRAD",
    "B_LCRC_LOCAL",
    "C_LCRC_MASK50",
    "D_TOPOLOGY",
    "E_GINE_GRAD",
    "F_GINE_MASK50",
)
#: Which frozen model evaluates a selector's fidelity.  TOPOLOGY is model-free
#: and is evaluated by both models.
SELECTOR_MODEL = {
    "A_LCRC_GRAD": "LCRC",
    "B_LCRC_LOCAL": "LCRC",
    "C_LCRC_MASK50": "LCRC",
    "D_TOPOLOGY": "BOTH",
    "E_GINE_GRAD": "GINE",
    "F_GINE_MASK50": "GINE",
}
MODEL_SELECTORS = {
    "LCRC": ("A_LCRC_GRAD", "B_LCRC_LOCAL", "C_LCRC_MASK50", "D_TOPOLOGY"),
    "GINE": ("E_GINE_GRAD", "F_GINE_MASK50", "D_TOPOLOGY"),
}
#: The candidate set a selector would have seen on its own (for the
#: "hit its own original Top-1" audit).  Never used for selection.
OWN_POOL_GENERATORS = {
    "A_LCRC_GRAD": ("LCRC_GRAD",),
    "B_LCRC_LOCAL": ("LCRC_LOCAL",),
    "C_LCRC_MASK50": ("LCRC_GRAD", "LCRC_LOCAL"),
    "D_TOPOLOGY": ("TOPO_S2",),
    "E_GINE_GRAD": ("GINE_GRAD",),
    "F_GINE_MASK50": ("GINE_GRAD",),
}
GENERATOR_SELECTOR = {"LCRC_GRAD": "A_LCRC_GRAD", "LCRC_LOCAL": "B_LCRC_LOCAL",
                      "GINE_GRAD": "E_GINE_GRAD", "TOPO_S2": "D_TOPOLOGY"}

#: Frozen checkpoints, reused read-only from the task-relevance sprint.
CHECKPOINT_NAMES = {
    "LCRC_WC": "L_WC_LOGP",
    "GINE_WC": "T_WC_LOGP",
    "LCRC_LIPO": "L_lipophilicity",
    "GINE_LIPO": "G0_lipophilicity",
}
SOURCE_CHECKPOINT_PATHS = {
    "L_lipophilicity": "results/stage1/ogbg-mollipo/L/Rf/checkpoint_best_val.pt",
    "G0_lipophilicity": "results/stage1/ogbg-mollipo/G0/Rf/checkpoint_best_val.pt",
    "L_WC_LOGP": "results/oracle/ogbg-mollipo/L/WC_LOGP/selected/checkpoint_best_val.pt",
    "T_WC_LOGP": "results/oracle/ogbg-mollipo/T/WC_LOGP/selected/checkpoint_best_val.pt",
}

#: Frozen source hashes (identical to the task-relevance sprint records).
EXPECTED_CHECKPOINT_SHA256 = {
    "L_lipophilicity": "9ef542e591963cf8ed012e95036d0f6f95e9a3c3d142a1838f28581bbbed10bc",
    "G0_lipophilicity": "04282be531bdc5883e31684e79720d3b42f5c998b5442921b56e61aecfb559ca",
    "L_WC_LOGP": "4522a6361909b1b7d1aee317bdbead0798e07eee0dd3143a396ade64ea788a67",
    "T_WC_LOGP": "5e812d7e45813bbb9100e3b3b98ac9f0d320573d7370abbade8d5fa8417fc165",
}
EXPECTED_CHECKPOINT_META = {
    "L_lipophilicity": {"kind": "L", "epoch": 124, "valid_rmse_raw": 0.7068848410118609, "n_parameters": 228674},
    "G0_lipophilicity": {"kind": "G0", "epoch": 183, "valid_rmse_raw": 0.616077290013576, "n_parameters": 377350},
    "L_WC_LOGP": {"kind": "L", "epoch": 333, "valid_rmse_raw": 0.11440873206013732, "n_parameters": 228674},
    "T_WC_LOGP": {"kind": "T", "epoch": 355, "valid_rmse_raw": 0.1032791933377434, "n_parameters": 377350},
}
EXPECTED_ASSET_SHA256 = {
    "pair_catalog.jsonl.gz": "12376174e7d95a20a25c3f3ed5cca510850db1c71807b30cbd826005c21e11ba",
    "row_atom_mapping.jsonl.gz": "01e861f379dfdbdd83aaf4caee236eceb3bebd7e7d4568e191d55aa96043f64b",
    "ogbg-mollipo.npz": "5690c0e3c7ba0ed51fbe30541e4a83a49abf174c6440aaebfb46a3c5cbc48257",
    "pair_endpoints.jsonl.gz": "4eaf852f3caf9aee442952da44fe535366fad33f4f8f891aea4e5d4ded52cacb",
}

#: Effective number of bytes retained by one batched intervention forward.
#: An implementation guard, not a research rule; a molecule that exceeds it is
#: processed in smaller variant chunks instead of OOM-ing the host.
MAX_BATCH_ATOMS = 200_000
MAX_BATCH_BONDS = 200_000
CHUNK_ATOMS = 60_000
CHUNK_BONDS = 60_000


def checkpoint_paths() -> dict[str, Path]:
    """The four frozen checkpoints, resolved without duplicating bytes."""
    out: dict[str, Path] = {}
    for name in CHECKPOINT_NAMES.values():
        candidates = [
            CHECKPOINT_DIR / f"{name}_{'best' if 'WC' in name else 'Rf_best'}.pt",
            XR_RESULTS / "checkpoints" / f"{name}_{'best' if 'WC' in name else 'Rf_best'}.pt",
        ]
        hit = next((p for p in candidates if p.is_file()), None)
        out[name] = hit if hit is not None else candidates[0]
    return out


#: Human-readable selector labels used in tables and the report.
SELECTOR_LABEL = {
    "A_LCRC_GRAD": "A LCRC-GRAD",
    "B_LCRC_LOCAL": "B LCRC-LOCAL",
    "C_LCRC_MASK50": "C LCRC-MASK50",
    "D_TOPOLOGY": "D TOPOLOGY",
    "E_GINE_GRAD": "E GINE-GRAD",
    "F_GINE_MASK50": "F GINE-MASK50",
}

FROZEN_CONSTANTS: dict[str, Any] = {
    "round_id": ROUND_ID,
    "dataset": DATASET,
    "main_splits": list(MAIN_SPLITS),
    "description_split": DESCRIPTION_SPLIT,
    "n_train_description": N_TRAIN_DESCRIPTION,
    "search": {
        "min_k": SEARCH_MIN_K,
        "max_k": SEARCH_MAX_K,
        "beam_width": BEAM_WIDTH,
        "k2_rule": "enumerate every pair of physical bonds sharing an endpoint",
        "k3_to_k6_rule": "expand every retained level-(k-1) candidate by one adjacent physical bond",
        "tie_break": "score desc, then fewer bonds, then sorted physical-bond-id tuple",
        "top_k": 1,
    },
    "generators": list(GENERATORS),
    "generator_scores": {
        "LCRC_GRAD": "A(S)=|sum_e g_e|/sqrt(k) on the frozen LCRC all-ones gate gradient",
        "LCRC_LOCAL": "B(S)=sum_{v in atoms(S)} |f(h_v)|/sqrt(k) on the frozen LCRC local output",
        "GINE_GRAD": "A(S)=|sum_e g_e|/sqrt(k) on the frozen GINE physical-bond gate gradient",
        "TOPO_S2": "n_s2_affected(S) only (structure; no model, label or oracle)",
    },
    "pool": "union over the four generators of all retained level-k beams (k=2..6), deduplicated by sorted bond-id tuple",
    "selectors": list(SELECTORS),
    "selector_scores": {
        "A_LCRC_GRAD": "|sum_{e in S} g_e| / sqrt(k)  (LCRC)",
        "B_LCRC_LOCAL": "sum_{v in atoms(S)} |f(h_v)| / sqrt(k)  (LCRC)",
        "C_LCRC_MASK50": "|y(1) - y(m_S=0.5)|  (LCRC)",
        "D_TOPOLOGY": "n_s2_affected(S)  (structure only)",
        "E_GINE_GRAD": "|sum_{e in S} g_e| / sqrt(k)  (GINE)",
        "F_GINE_MASK50": "|y(1) - y(m_S=0.5)|  (GINE)",
    },
    "dose_gate_for_selection": DOSE_GATE,
    "fidelity_evaluation_dose": 0.0,
    "controls": {
        "n_target": N_RANDOM,
        "min_acceptable": MIN_RANDOM,
        "max_growth_attempts": MAX_GROWTH_ATTEMPTS,
        "match": "same molecule, same physical-bond count, same atom count, not identical to Top-1",
        "seed": CONTROL_SEED,
    },
    "statistics": {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "main_unit": "molecule (independent within an official split)",
        "aux_unit": "shared-endpoint connected component",
        "concentration_top_fraction": CONCENTRATION_TOP_FRACTION,
        "concentration_max_share": CONCENTRATION_MAX_SHARE,
    },
    "thresholds": {
        "oracle_enrichment_min": ORACLE_ENRICHMENT_MIN,
        "fidelity_advantage_std_min": FIDELITY_ADVANTAGE_STD_MIN,
        "paired_improvement_min": PAIRED_IMPROVEMENT_MIN,
        "min_evaluable_fraction": MIN_EVALUABLE_FRACTION,
        "low_support_components": LOW_SUPPORT_COMPONENTS,
    },
    "gates": {
        "fd_eps": FD_EPS,
        "dose_path_steps": DOSE_PATH_STEPS,
        "gate_check_molecules": GATE_CHECK_MOLECULES,
    },
}


# ---------------------------------------------------------------------------
# import discipline / read-only reuse
# ---------------------------------------------------------------------------

_PATHS_READY = False


def install_source_paths() -> None:
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (str(XR_CODE), str(PILOT_CODE), str(ROUTE_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_READY = True


def import_xr():
    install_source_paths()
    import xr_common  # noqa: F401

    xr_common.install_source_paths()
    return xr_common


XC = import_xr()

# hashing / io / environment / tolerances: one vocabulary, inherited verbatim
sha256_file = XC.sha256_file
sha256_state = XC.sha256_state
sha256_json = XC.sha256_json
sha256_array = XC.sha256_array
sha256_text = XC.sha256_text
read_json = XC.read_json
write_json = XC.write_json
write_text = XC.write_text
write_csv = XC.write_csv
read_jsonl_gz = XC.read_jsonl_gz
write_jsonl_gz = XC.write_jsonl_gz
append_jsonl = XC.append_jsonl
jsonable = XC.jsonable
rmse = XC.rmse
mae = XC.mae
trace_tolerance = XC.trace_tolerance
oracle_tolerance = XC.oracle_tolerance
environment = XC.environment
git_commit = XC.git_commit
git_branch = XC.git_branch
git_head_now = XC.git_commit
git_dirty_tracked = XC.git_dirty_tracked
component_bootstrap = XC.component_bootstrap
relpath = XC.relpath
utc_now = XC.utc_now
source_modules = XC.source_modules


def freeze_hash() -> str:
    return sha256_json(FROZEN_CONSTANTS)


def code_hashes() -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(CODE_DIR.glob("*.py"))}


def append_jsonl(path: Path | str, row: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def open_resumable_jsonl(path: Path | str, key_fields: Sequence[str]):
    """Open a JSONL file for append, truncating a torn last line.

    Returns ``(seen_keys, handle)``.  Every complete existing row contributes
    its key, so a restarted stage skips the molecules it already finished and
    never rewrites a row.
    """
    path = Path(path)
    ensure_dir(path.parent)
    seen: set[tuple] = set()
    keep: list[str] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.rstrip("\n")
                if not text.strip():
                    continue
                try:
                    rec = json.loads(text)
                except json.JSONDecodeError:
                    break
                keep.append(text)
                seen.add(tuple(int(rec[f]) if isinstance(rec[f], int) else str(rec[f]) for f in key_fields))
        with path.open("w", encoding="utf-8") as handle:
            for text in keep:
                handle.write(text + "\n")
    return seen, path.open("a", encoding="utf-8")


def molecule_key(rec: Mapping[str, Any]) -> tuple[str, int]:
    return (str(rec["split"]), int(rec["cache_row"]))


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as handle:
        return {key: handle[key] for key in handle.files}


def write_npz(path: Path | str, payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in payload.items()})
    tmp.replace(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def median(values: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"q{int(round(q * 100)):02d}": float("nan") for q in qs}
    return {f"q{int(round(q * 100)):02d}": float(np.quantile(arr, q)) for q in qs}


def support_label(n_units: int, n_components: int, *, min_components: int = LOW_SUPPORT_COMPONENTS) -> str:
    if int(n_units) == 0:
        return "NO_SUPPORT"
    if int(n_components) < int(min_components):
        return "LOW_SUPPORT"
    return "OK"


def safe_div(num: float, den: float) -> float | None:
    if den is None or not np.isfinite(den) or den == 0.0:
        return None
    return float(num) / float(den)


# ---------------------------------------------------------------------------
# bootstrap (molecule / component resampling)
# ---------------------------------------------------------------------------


def bootstrap_columns(
    values: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
    unit_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Percentile bootstrap of the mean and median of every named column.

    ``unit_ids`` is the resampling unit (molecule id by default, or a
    shared-endpoint component id for the auxiliary pair analysis).  When it is
    given, all rows of a drawn unit enter the replicate together (cluster
    bootstrap); otherwise each row is a unit.
    """
    mask = np.asarray(mask, dtype=bool)
    idx_all = np.flatnonzero(mask)
    cols = {key: np.asarray(val, dtype=np.float64)[idx_all] for key, val in values.items()}
    out: dict[str, Any] = {
        "n_rows": int(idx_all.size),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
        "point": {},
        "ci95": {},
    }
    if idx_all.size == 0:
        for key in cols:
            out["point"][key] = {"mean": None, "median": None}
            out["ci95"][key] = {"mean": [None, None], "median": [None, None]}
        return out
    for key, arr in cols.items():
        finite = arr[np.isfinite(arr)]
        out["point"][key] = {
            "mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "n_finite": int(finite.size),
        }

    rng = np.random.default_rng(int(seed))
    if unit_ids is None:
        draws = rng.integers(0, idx_all.size, size=(int(n_boot), idx_all.size))
        groups = [[int(v)] for v in range(idx_all.size)]
    else:
        units = np.asarray(unit_ids, dtype=np.int64)[idx_all]
        uniq = np.unique(units)
        pos = [np.flatnonzero(units == u) for u in uniq.tolist()]
        draws = rng.integers(0, uniq.size, size=(int(n_boot), uniq.size))
        groups = [[int(i) for i in p] for p in pos]
    for key, arr in cols.items():
        means, meds = [], []
        for row in draws:
            local = np.concatenate([groups[int(j)] for j in row]) if groups else np.zeros(0, dtype=np.int64)
            sub = arr[local]
            sub = sub[np.isfinite(sub)]
            means.append(float(sub.mean()) if sub.size else np.nan)
            meds.append(float(np.median(sub)) if sub.size else np.nan)
        means = np.asarray(means, dtype=np.float64)
        meds = np.asarray(meds, dtype=np.float64)
        out["ci95"][key] = {
            "mean": [float(np.nanquantile(means, alpha / 2.0)), float(np.nanquantile(means, 1 - alpha / 2.0))],
            "median": [float(np.nanquantile(meds, alpha / 2.0)), float(np.nanquantile(meds, 1 - alpha / 2.0))],
        }
    return out


def positive_concentration(values: Sequence[float], top_fraction: float = CONCENTRATION_TOP_FRACTION) -> dict[str, Any]:
    """How much of the total positive effect comes from the largest few units."""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    pos = np.sort(arr[arr > 0.0])[::-1]
    total = float(pos.sum())
    k = max(1, int(math.ceil(top_fraction * arr.size))) if arr.size else 0
    top = float(pos[:k].sum()) if k else 0.0
    return {
        "n_positive": int(pos.size),
        "n_negative": int(np.sum(arr < 0.0)),
        "n_zero": int(np.sum(arr == 0.0)),
        "total_positive": total,
        "top_fraction": float(top_fraction),
        "top_k": int(k),
        "top_share": (top / total) if total > 0 else None,
        "max_value": float(arr.max()) if arr.size else None,
    }


def gate_report(*, passed: bool, **payload: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **payload}
