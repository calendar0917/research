"""Round-local identity, paths and constants for the oracle task-relevance round.

Round / protocol: ``lcrc-lipo-oracle-task-relevance-public122-s0-v1``.

This round is a *clarification + two fixed supervised tasks* round that builds
on the executed multi-motif mechanism round (protocol ``0273cad6``, report
``a5febe96``):

* **Part A** (zero GPU, no new inference) clarifies the existing six-family
  results: coverage/claim matrix, the same-population three-table contrast
  (population change vs control change), and the conditional scalar/S2
  recovery effect from the already frozen eight-cell internal path.
* **Part B** adds two *computational* supervised tasks on the exact same 4200
  original molecules and the frozen rewiring catalogue:

  - ``WC_LOGP`` : RDKit Wildman-Crippen logP (a computational function
    reference, *not* experimental logD and *not* a new measured logP);
  - ``MOLWT``   : RDKit average molecular weight (composition-preserving
    rewiring keeps it invariant, so it is the task-invariant control).

Exactly two formal trainings (``WC_LOGP_s0`` on gpu0 and ``MOLWT_s0`` on gpu1 of
host ``res``), same LCRC-C1 / ordinary-MSE recipe / seed 0 / frozen recipe, no
other training.  The original experimental logD model, its labels, the official
split, the six-family catalogue, the graphs, controls and dose rules are reused
unchanged.

Everything here is IO / provenance plumbing; the frozen upstream modules are
imported read-only from the executed source round so this round is auditable
against the same bytes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# paths / identity
# ---------------------------------------------------------------------------

ROUND_ID = "lcrc-lipo-oracle-task-relevance-public122-s0-v1"
PROTOCOL_ID = ROUND_ID

NEW_CODE = Path(__file__).resolve().parent
NEW_TRACK = NEW_CODE.parent
REPO = NEW_TRACK.parents[2]

SS = REPO / "tracks/structural-sufficiency"
SOURCE_TRACK = SS / "lcrc_lipo_multi_motif_mechanism_public122_v1"
SOURCE_CODE = SOURCE_TRACK / "code"
SOURCE_ANALYSIS = SOURCE_TRACK / "analysis"

#: Frozen source identity (multi-motif mechanism round).
SOURCE_PROTOCOL_REVISION = "0273cad6"
SOURCE_REPORT_REVISION = "a5febe96"
SOURCE_EXECUTED_PREV = "lcrc_lipo_amide_connected_rewire_public122_v1"
SOURCE_EXECUTED_PREV_REV = "64bb1c9f"

NEW_ANALYSIS = NEW_TRACK / "analysis"
NEW_CONFIGS = NEW_TRACK / "configs"
NEW_TESTS = NEW_TRACK / "tests"
NEW_RUNS = NEW_TRACK / "runs"
NEW_CASES = NEW_ANALYSIS / "cases"

# The frozen upstream modules must be importable by their own module names.  The
# NEW_CODE directory must win so this round's own modules (analysis.py,
# figures.py, ...) shadow any same-named source-round module.
for _p in (str(SOURCE_CODE), str(NEW_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if sys.path[0] != str(NEW_CODE):
    sys.path.remove(str(NEW_CODE))
    sys.path.insert(0, str(NEW_CODE))

import common as SC  # noqa: E402  (executed multi-motif common: frozen data/model plumbing)
import chemcheck as CHEM  # noqa: E402
import graphlib as G  # noqa: E402
import families as FAM  # noqa: E402
import rewire as R  # noqa: E402
import feats as FEAT  # noqa: E402
import doses as DOSES  # noqa: E402
import infer as INF  # noqa: E402  (frozen cache builder / structural re-verification)

# The source round also has an ``analysis.py``; import it under a private name so
# it never shadows this round's ``analysis`` stage module.
def _load_source_analysis():
    import importlib.util

    name = "_lr_src_analysis"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCE_CODE / "analysis.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AN = _load_source_analysis()  # frozen Part-A statistics helpers

# ---------------------------------------------------------------------------
# two computational tasks
# ---------------------------------------------------------------------------

TASKS: tuple[str, ...] = ("wc_logp", "molwt")
TASK_DESCRIPTION = {
    "wc_logp": "RDKit Wildman-Crippen logP (CalcCrippenDescriptors ... [0]); computational reference, NOT experimental logD",
    "molwt": "RDKit average molecular weight (Descriptors.MolWt); composition-invariant control",
}
TASK_UNITS = {"wc_logp": "logP (Wildman-Crippen, dimensionless)", "molwt": "g/mol (average molecular weight)"}
#: Main CF scope (full new molecule re-encoding).  G_FIXED is an internal diagnostic only.
CF_SCOPE = "FULL"

# ---------------------------------------------------------------------------
# frozen ordinary-MSE recipe (identical to the executed A_s0 arm; no change)
# ---------------------------------------------------------------------------

RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "weight_decay": 1.0e-3,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "batch_size": 128,
    "dropout": 0.10,
    "gradient_clip": 5.0,
    "epochs": 360,
    "steps_per_epoch": 27,
    "total_optimizer_steps": 9720,
    "scheduler": "per_step_cosine",
    "shuffle_offset": 917_000,
    "early_stopping": False,
    "soup": False,
    "ema": False,
    "swa": False,
    "amp": False,
}
TRAIN_SEED = 0
SHUFFLE_SEED = int(RECIPE["shuffle_offset"])
EVAL_BATCH_SIZE = 256

BOOTSTRAP_SEED = 20261114
BOOTSTRAP_REPLICATES = 5000

#: Tolerances.
ORACLE_INVARIANT_TOL = 1e-8      # MolWt rewire delta must be <= this
WC_LOG_CHANGE_TOL = 1e-8         # "changed" threshold for WC_LOGP rewire values
REPRO_TOL = 1e-5

#: Analytic form (official RDKit docs).
WC_LOGP_CALL = "rdMolDescriptors.CalcCrippenDescriptors(mol, includeHs=True, force=True)[0]"
MOLWT_CALL = "Descriptors.MolWt(mol)"

N_PRIMARY_ITEMS = 6  # WC_LOGP families x one gain item -> 6-item Bonferroni
ALPHA = 0.05


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    return SC.sha256_file(path)


def sha256_bytes(data: bytes) -> str:
    return SC.sha256_bytes(data)


def write_json(path: Path | str, obj: Any) -> Path:
    return SC.write_json(path, obj)


def read_json(path: Path | str) -> Any:
    return SC.read_json(path)


def git_commit() -> str:
    return SC.git_commit()


def git_dirty() -> bool:
    return SC.git_dirty()


def utcnow() -> str:
    return SC.utcnow()


def log(msg: str) -> None:
    SC.log(msg)


def rdkit_version() -> str:
    return SC.rdkit_version()


def env_info() -> dict[str, Any]:
    return SC.env_info()


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _csv_value(row.get(k, "")) for k in columns})
    return p


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _csv_value(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, float) and not np.isfinite(v):
        return ""
    return v


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def parse_bonds(code: str) -> list[tuple[int, int]]:
    code = str(code).strip()
    if not code:
        return []
    return [tuple(int(x) for x in chunk.split("-")) for chunk in code.split(".")]


# ---------------------------------------------------------------------------
# frozen original (un-rewired) structure caches (reused verbatim)
# ---------------------------------------------------------------------------


def original_cache_path(split: str) -> Path:
    return SC.DD.cache_path(SC.DATA_REPEAT, split)


def load_original_cache(split: str) -> dict[str, Any]:
    import torch

    path = original_cache_path(split)
    if not path.exists():
        raise FileNotFoundError(f"MISSING_BASELINE: frozen {split} cache not found at {track_rel(path)}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    expected = [int(x) for x in SC.load_source("none")["rows"][split]]
    got = [int(x) for x in np.asarray(cache["rows"]).tolist()]
    if got != expected:
        raise RuntimeError(f"frozen {split} cache rows differ from the published split")
    return cache


def frozen_spec():
    return SC.DD.frozen_spec()


def frozen_norm() -> dict[str, np.ndarray]:
    ck = SC.load_checkpoint()
    return SC.frozen_norm(ck)


def original_target_norm() -> dict[str, float]:
    """The frozen *experimental logD* target mean/scale (unchanged; not used by Part B)."""
    ck = SC.load_checkpoint()
    return SC.target_norm(ck)


# ---------------------------------------------------------------------------
# code / config hashes (for provenance)
# ---------------------------------------------------------------------------


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(NEW_CODE.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for p in sorted(NEW_TESTS.glob("*.py")):
        out[f"tests/{p.name}"] = sha256_file(p)
    for p in sorted(NEW_CONFIGS.glob("*.yaml")):
        out[f"configs/{p.name}"] = sha256_file(p)
    for name in ("PROTOCOL.md", "REPORT.md"):
        p = NEW_TRACK / name
        if p.exists():
            out[name] = sha256_file(p)
    # executed source modules consumed read-only
    for name in (
        "common.py", "graphlib.py", "families.py", "rewire.py", "feats.py", "doses.py",
        "chemcheck.py", "analysis.py", "infer.py", "prepare.py",
    ):
        p = SOURCE_CODE / name
        out[f"source_code/{name}"] = sha256_file(p) if p.exists() else "missing"
    return out


__all__ = [name for name in dir() if not name.startswith("_")]
