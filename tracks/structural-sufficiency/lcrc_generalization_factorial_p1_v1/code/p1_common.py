"""Shared paths, frozen constants and hashing for ``lcrc-generalization-factorial-p1-v1``.

This round re-uses, read-only:

* the frozen ChemBench Lipophilicity splits ``random122`` / ``random123`` /
  ``random124`` (Train/Valid/Test = 3360/420/420, the same 4,200-molecule CSV);
* the frozen R3 training code and the committed per-run artefacts of
  ``lcrc_performance_longrun_v1`` (arm construction, initialisation, training
  recipe, checkpoint rule, evaluation units);
* the label-free molecule/support machinery of
  ``lcrc_failure_mechanism_p0_v1`` (rooted radius-2 signatures, Bemis-Murcko
  scaffolds, descriptors, Morgan Tanimoto).

Nothing in this module writes outside this round's own directory.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-generalization-factorial-p1-v1"
PROTOCOL_ID = ROUND_ID

CODE = Path(__file__).resolve().parent
ROUND = CODE.parent
REPO = ROUND.parents[2]
STRUCT = REPO / "tracks/structural-sufficiency"

LONGRUN = STRUCT / "lcrc_performance_longrun_v1"
LONGRUN_CODE = LONGRUN / "code"
LONGRUN_RUNS = LONGRUN / "analysis/runs"
LONGRUN_FREEZE = LONGRUN / "analysis/protocol_freeze.json"

P0 = STRUCT / "lcrc_failure_mechanism_p0_v1"
P0_CODE = P0 / "code"

CHEMBENCH = STRUCT / "lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench"
LIPOP_CSV = CHEMBENCH / "Lipop/Lipop.csv.gz"
LIPOP_SPLIT_DIR = CHEMBENCH / "Lipop"

ANALYSIS = ROUND / "analysis"
RUNS = ANALYSIS / "runs"
TESTS = ROUND / "tests"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"

#: split name -> (frozen longrun `--split` value, pickle file name)
SPLITS: dict[str, dict[str, str]] = {
    "random122": {"longrun_split": "published", "file": "Lipop.random122.ind.pkl"},
    "random123": {"longrun_split": "lipo-repeat:123", "file": "Lipop.random123.ind.pkl"},
    "random124": {"longrun_split": "lipo-repeat:124", "file": "Lipop.random124.ind.pkl"},
}
SPLIT_ORDER: tuple[str, ...] = ("random122", "random123", "random124")
PART_ORDER: tuple[str, ...] = ("train", "valid", "test")
EXPECTED_SIZES = {"train": 3360, "valid": 420, "test": 420}
N_MOLECULES = 4200

#: the four arms of the random122 factorial, in canonical order
ARMS: tuple[str, ...] = ("base", "chem8", "multicontrib", "combo_chem_contrib")
ARM_SHORT = {
    "base": "BASE",
    "chem8": "CHEM8",
    "multicontrib": "MULTICONTRIB8",
    "combo_chem_contrib": "COMBO",
}
#: Q2 only compares these two
Q2_ARMS: tuple[str, ...] = ("base", "combo_chem_contrib")

SEEDS: dict[str, tuple[int, ...]] = {
    "random122": (0, 1, 2),
    "random123": (0, 1),
    "random124": (0, 1),
}

#: frozen analysis constants (declared in PROTOCOL.md before any new run)
TOL = 1e-5                  # metric reproduction tolerance
N_BOOT = 2000               # scaffold-cluster bootstrap replicates
BOOTSTRAP_SEED = 20261104   # deterministic bootstrap stream for this round
CI_PERCENTILES = (2.5, 97.5)
TOP_FRACTION = 0.10         # tail fraction for CVaR90-AE and the BASE-defined tail
RARE_DF_MAX = 4             # df in [1, 4] counts as "rare" (P0 definition)
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
N_QUANTILE_BINS = 4
LOW_SUPPORT_N = 30

#: recipe knobs that must be identical for every reused/fresh unit (R3)
FROZEN_RECIPE_KEYS: tuple[str, ...] = (
    "optimizer",
    "adam_betas",
    "adam_eps",
    "peak_lr",
    "min_lr",
    "weight_decay",
    "batch_size",
    "dropout",
    "epochs_planned",
    "scheduler",
    "scheduler_horizon_epochs",
    "gradient_clip",
    "loss",
    "checkpoint_rule",
    "early_stopping",
    "soup",
    "ema",
    "swa",
    "ssl",
    "hyperparameter_search",
    "validation_used_for_checkpoint_selection",
    "test_used_for_any_selection",
)
SHUFFLE_OFFSET = 917000     # recorded in every frozen `train.json`; seed+offset


# ---------------------------------------------------------------------------
# units: (split, arm, seed) -> frozen artefact
# ---------------------------------------------------------------------------


def unit_key(split: str, arm: str, seed: int) -> str:
    short = {"base": "base", "chem8": "chem8", "multicontrib": "contrib", "combo_chem_contrib": "combo"}[arm]
    return f"{short}_{split}_s{seed}"


ARM_SHORT_KEY = {
    "base": "base",
    "chem8": "chem8",
    "multicontrib": "contrib",
    "combo_chem_contrib": "combo",
}


def fresh_unit_key(split: str, arm: str, seed: int) -> str:
    """Artefact directory name of a unit trained in this round."""
    return f"f1_{ARM_SHORT_KEY[arm]}_{split}_s{int(seed)}"


def _short_arm(arm: str) -> str:
    return ARM_SHORT_KEY[arm]


#: Units whose artefacts already exist in ``lcrc_performance_longrun_v1``.
#: ``key`` is this round's unit id; ``source_run`` is the frozen historical run.
REUSED_UNITS: tuple[dict[str, Any], ...] = (
    {"split": "random122", "arm": "base", "seed": 0, "source_run": "p0_base_lipo_parity"},
    {"split": "random122", "arm": "base", "seed": 1, "source_run": "p5_base_lipo_s1"},
    {"split": "random122", "arm": "base", "seed": 2, "source_run": "p5_base_lipo_s2"},
    {"split": "random122", "arm": "chem8", "seed": 0, "source_run": "p1_chem8_lipo"},
    {"split": "random122", "arm": "multicontrib", "seed": 0, "source_run": "p1_contrib_lipo"},
    {"split": "random122", "arm": "combo_chem_contrib", "seed": 0, "source_run": "p2b_combo_chem_contrib_lipo"},
    {"split": "random122", "arm": "combo_chem_contrib", "seed": 1, "source_run": "p5b_combo_lipo_s1"},
    {"split": "random122", "arm": "combo_chem_contrib", "seed": 2, "source_run": "p5b_combo_lipo_s2"},
    {"split": "random123", "arm": "combo_chem_contrib", "seed": 0, "source_run": "p4b_combo_lipo123"},
    {"split": "random124", "arm": "combo_chem_contrib", "seed": 0, "source_run": "p4b_combo_lipo124"},
)

#: Units this round must train (the only GPU work in the round).
FRESH_UNITS: tuple[dict[str, Any], ...] = (
    {"split": "random122", "arm": "chem8", "seed": 1},
    {"split": "random122", "arm": "chem8", "seed": 2},
    {"split": "random122", "arm": "multicontrib", "seed": 1},
    {"split": "random122", "arm": "multicontrib", "seed": 2},
    {"split": "random123", "arm": "base", "seed": 0},
    {"split": "random123", "arm": "base", "seed": 1},
    {"split": "random123", "arm": "combo_chem_contrib", "seed": 1},
    {"split": "random124", "arm": "base", "seed": 0},
    {"split": "random124", "arm": "base", "seed": 1},
    {"split": "random124", "arm": "combo_chem_contrib", "seed": 1},
)


def unit_registry() -> list[dict[str, Any]]:
    """All units of the round, with origin and on-disk artefact directory."""
    out: list[dict[str, Any]] = []
    for spec in REUSED_UNITS:
        key = unit_key(spec["split"], spec["arm"], spec["seed"])
        out.append(
            {
                "key": key,
                "split": spec["split"],
                "arm": spec["arm"],
                "seed": int(spec["seed"]),
                "origin": "reused",
                "source_run": spec["source_run"],
                "run_dir": LONGRUN_RUNS / spec["source_run"],
            }
        )
    for spec in FRESH_UNITS:
        key = fresh_unit_key(spec["split"], spec["arm"], spec["seed"])
        out.append(
            {
                "key": key,
                "split": spec["split"],
                "arm": spec["arm"],
                "seed": int(spec["seed"]),
                "origin": "fresh",
                "source_run": None,
                "run_dir": RUNS / key,
            }
        )
    return out


def expected_units() -> dict[tuple[str, str, int], dict[str, Any]]:
    return {(u["split"], u["arm"], u["seed"]): u for u in unit_registry()}


# ---------------------------------------------------------------------------
# hashing / io helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_sha(rows: Sequence[int]) -> str:
    arr = np.asarray([int(x) for x in rows], dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def sha256_text(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, allow_nan=True, default=_json_default)
        handle.write("\n")
    return target


def track_rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return None


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


# ---------------------------------------------------------------------------
# analysis-code freeze
# ---------------------------------------------------------------------------

ANALYSIS_CODE_FILES: tuple[str, ...] = (
    "code/p1_common.py",
    "code/p1_stats.py",
    "code/p1_sources.py",
    "code/p1_analysis.py",
    "code/run_analysis.py",
    "code/run_unit.py",
    "code/dispatch.py",
    "code/write_freeze.py",
)


def analysis_code_hashes() -> dict[str, str]:
    return {name: sha256_file(ROUND / name) for name in ANALYSIS_CODE_FILES}


def verify_analysis_freeze() -> dict[str, Any]:
    """Refuse to score unless the analysis code still matches the freeze."""
    if not PROTOCOL_FREEZE.exists():
        raise FileNotFoundError(f"{track_rel(PROTOCOL_FREEZE)} is missing; write it before scoring")
    freeze = read_json(PROTOCOL_FREEZE)
    recorded = freeze.get("analysis_code_hashes", {})
    actual = analysis_code_hashes()
    mismatched = {k: (recorded.get(k), v) for k, v in actual.items() if recorded.get(k) != v}
    if mismatched:
        raise RuntimeError(f"analysis code changed after the protocol freeze: {mismatched}")
    return {
        "protocol_freeze": track_rel(PROTOCOL_FREEZE),
        "protocol_freeze_sha256": sha256_file(PROTOCOL_FREEZE),
        "analysis_code_hashes_match": True,
    }


__all__ = [
    "ROUND_ID",
    "PROTOCOL_ID",
    "CODE",
    "ROUND",
    "REPO",
    "STRUCT",
    "LONGRUN",
    "LONGRUN_CODE",
    "LONGRUN_RUNS",
    "LONGRUN_FREEZE",
    "P0",
    "P0_CODE",
    "LIPOP_CSV",
    "LIPOP_SPLIT_DIR",
    "ANALYSIS",
    "RUNS",
    "TESTS",
    "PROTOCOL_FREEZE",
    "SPLITS",
    "SPLIT_ORDER",
    "PART_ORDER",
    "EXPECTED_SIZES",
    "N_MOLECULES",
    "ARMS",
    "ARM_SHORT",
    "Q2_ARMS",
    "SEEDS",
    "TOL",
    "N_BOOT",
    "BOOTSTRAP_SEED",
    "CI_PERCENTILES",
    "TOP_FRACTION",
    "RARE_DF_MAX",
    "MORGAN_RADIUS",
    "MORGAN_BITS",
    "N_QUANTILE_BINS",
    "LOW_SUPPORT_N",
    "FROZEN_RECIPE_KEYS",
    "SHUFFLE_OFFSET",
    "unit_key",
    "fresh_unit_key",
    "ARM_SHORT_KEY",
    "REUSED_UNITS",
    "FRESH_UNITS",
    "unit_registry",
    "expected_units",
    "sha256_file",
    "row_sha",
    "sha256_text",
    "read_json",
    "write_json",
    "track_rel",
    "utcnow",
    "git_commit",
    "git_dirty",
    "ANALYSIS_CODE_FILES",
    "analysis_code_hashes",
    "verify_analysis_freeze",
]
