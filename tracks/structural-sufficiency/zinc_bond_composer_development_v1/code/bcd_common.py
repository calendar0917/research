"""Shared data, hashing and atomic-write helpers for the BCD development round.

Scope
-----
The round is an internal architecture-development probe on the already exposed
SSRC Fit-8000 / Monitor-1000 split.  It reuses:

* the phase1b split and the Fit+Monitor-only label vault;
* the phase1e Fit-only objects and the seed-0 Fit-only masked-SSL encoder;
* the frozen P3 constructor and R3 recipe.

It never loads the internal Reserved rows, official ZINC valid or official ZINC
test.  Monitor labels are read repeatedly for development curves and metrics;
this is explicitly not a blind test.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = HERE.parents[3]
STRUCT_CODE = REPO / "tracks/structure-contribution/code"
STRUCT_SUFF = REPO / "tracks/structural-sufficiency"
for _p in (HERE, STRUCT_CODE, STRUCT_SUFF / "code/phase1b"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS_1B = STRUCT_SUFF / "results/phase1b"
RESULTS_1E = STRUCT_SUFF / "results/phase1e"
BASELINE_1E = RESULTS_1E / "baseline"
SPLIT_PATH = RESULTS_1B / "split_zinc_phase1b.json"
VAULT_PATH = RESULTS_1B / "zinc_labels_fit_monitor.json"
PREP_PATH = BASELINE_1E / "prep_fit.json"
CACHE = Path(os.environ.get("BCD_CACHE_DIR", PROJECT / "cache"))
RUNS = PROJECT / "runs"
ANALYSIS = PROJECT / "analysis"
CURVES = PROJECT / "curves"

PROTOCOL_ID = "zinc-bond-composer-development-v1"
SEED = 0
EXPECTED_FIT = 8000
EXPECTED_MONITOR = 1000
EXPECTED_PARAMS = {"B0": 413_505, "B1": 441_825, "B2": 441_825}
P3_PARAMS = 413_505
BRANCH_PARAMS = 28_320
NEW_GROUPS = {
    "input_projection": 4_656,
    "block": 18_960,
    "output_projection": 4_704,
    "total": 28_320,
}
MIX_PARAMS = {"B1": 9_408, "B2": 9_408}

# Frozen input provenance (same values used by phase1e/phase2a).
SPLIT_FILE_SHA = "fbca5fd5ac9c82c457bf4d78781fbe9936dd7068082377ff8cc2bf2309d72b49"
FIT_ROWS_SHA = "7f66a23e038c3ad4879540f23d1e5a6cd6b8a789bbee8950a7b2c1d8b345ab15"
MONITOR_ROWS_SHA = "dc6cc4e9cb9d3cebac038b487f7e699e6c0aa52e5075e45825975b034509304f"
OBJECTS_FIT_SHA = "bfbf96532029f8b5020e416bbf0f9fc7bdb257f5377265fc05b9c7c0f02d766c"
OBJECTS_MONITOR_SHA = "7a71850655ea606da51ee7bf27edb3c09340b25d701dee586b15a30b55652f0a"
SSL_SHA = "0d045e6090062ac6e919c79c82c2cbc9a223c7974122c91d5b68a244d265bc0c"
PREP_SHA = "b00addec172a0e71f8556760ae2896360e8770dcf1fe25e86226da18b531d04b"
#: phase1e train record: post-SSL encoder and composer/reader hashes.
CANONICAL_ENCODER_SHA = "37282569525d1e69a6172c23f8f2e357ba7f35892af6116ce980628c54c4d4d6"
CANONICAL_GROUP_SHA = "d9ffbd15356659ed06af4b43077bb7f27403ad0afc3e7ac00c4d34fc98c7415f"
R3_BATCH_ORDER_SHA = "d142e307339b2869f4ad2d3ff8472e28249f1dd1ff887a4ae55f5b837d61c81f"

R3 = {
    "epochs": 360,
    "batch_size": 128,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "weight_decay": 1e-5,
    "grad_clip": 5.0,
    "soup_last": 5,
    "eval_every": 20,
    "eval_batch_size": 256,
}


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray, dtype: str = "float32") -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    return hashlib.sha256(a.tobytes()).hexdigest()


def rows_sha256(rows: Sequence[int]) -> str:
    return sha256_array(np.asarray(rows, dtype=np.int64), "int64")


def read_json(path: Path | str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


@dataclass(frozen=True)
class Split:
    fit: np.ndarray
    monitor: np.ndarray
    sha256_file: str


def load_split() -> Split:
    split = read_json(SPLIT_PATH)
    fit = np.asarray(split["rows"]["fit"], dtype=np.int64)
    monitor = np.asarray(split["rows"]["monitor"], dtype=np.int64)
    reserved = np.asarray(split["rows"]["reserved"], dtype=np.int64)
    if (fit.size, monitor.size, reserved.size) != (EXPECTED_FIT, EXPECTED_MONITOR, 1000):
        raise SystemExit("unexpected SSRC split sizes")
    if set(fit.tolist()) & set(monitor.tolist()):
        raise SystemExit("Fit and Monitor overlap")
    if set(fit.tolist()) & set(reserved.tolist()) or set(monitor.tolist()) & set(reserved.tolist()):
        raise SystemExit("Reserved overlaps another split side")
    if rows_sha256(fit) != FIT_ROWS_SHA:
        raise SystemExit("Fit row hash mismatch")
    if rows_sha256(monitor) != MONITOR_ROWS_SHA:
        raise SystemExit("Monitor row hash mismatch")
    got_file = sha256_file(SPLIT_PATH)
    if got_file != SPLIT_FILE_SHA:
        raise SystemExit(f"split file hash mismatch: {got_file}")
    return Split(fit=fit, monitor=monitor, sha256_file=got_file)


def load_vault() -> dict[int, float]:
    vault = read_json(VAULT_PATH)
    scope = vault.get("scope", {})
    if scope.get("reserved_labels_present") or scope.get("official_valid_test_labels_present"):
        raise SystemExit("label vault is not Fit/Monitor-only")
    return {int(k): float(v) for k, v in vault["labels"].items()}


def row_labels(rows: Sequence[int], labels: Mapping[int, float]) -> np.ndarray:
    missing = [int(r) for r in rows if int(r) not in labels]
    if missing:
        raise SystemExit(f"labels missing for rows {missing[:5]}")
    return np.asarray([labels[int(r)] for r in rows], dtype=np.float64)


def _find(name: str, candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    raise SystemExit(f"required frozen input missing: {name}; tried {[str(p) for p in candidates]}")


def load_objects(which: str, rows: Sequence[int]) -> tuple[list[Any], dict[str, Any]]:
    import torch

    name = "objects_fit.pt" if which == "fit" else "objects_monitor.pt"
    expected = OBJECTS_FIT_SHA if which == "fit" else OBJECTS_MONITOR_SHA
    path = _find(name, [CACHE / name, BASELINE_1E / name])
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(f"{name} hash mismatch: {got} != {expected}")
    objs = torch.load(path, map_location="cpu", weights_only=False)
    want = [int(r) for r in rows]
    have = [int(o.mol_index) for o in objs]
    if have != want:
        raise SystemExit(f"{name} row order mismatch")
    return objs, {"path": str(path), "sha256": got, "rows_sha256": rows_sha256(want)}


def load_ssl_state() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import torch

    path = _find("pretrain_s0.pt", [CACHE / "pretrain_s0.pt", BASELINE_1E / "pretrain_s0.pt"])
    got = sha256_file(path)
    if got != SSL_SHA:
        raise SystemExit(f"SSL state hash mismatch: {got} != {SSL_SHA}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state, {"path": str(path), "sha256": got, "keys": sorted(state.keys())}


def load_standardiser() -> tuple[dict[str, Any], dict[str, Any]]:
    norm = read_json(PREP_PATH)
    if not norm.get("fit_only") or int(norm.get("n_fit", -1)) != EXPECTED_FIT:
        raise SystemExit("standardiser is not the frozen Fit-only 8000-row one")
    got = sha256_file(PREP_PATH)
    if got != PREP_SHA:
        raise SystemExit(f"standardiser file changed: {got}")
    return norm, {"path": str(PREP_PATH), "sha256": got, "fit_only": True}


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relative(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)
