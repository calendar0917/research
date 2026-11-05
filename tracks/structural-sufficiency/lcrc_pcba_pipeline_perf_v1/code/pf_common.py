"""Bootstrap, frozen-input access and the fixed measurement protocol.

Round: ``lcrc-pcba-pipeline-perf-v1``

This module does **not** build a new dataset, model or recipe.  It reads the
frozen ``lcrc-pcba-scale-pilot-v1`` artifacts (memory-mapped feature cache,
Train/Valid-only label artifact, official split, feature spec) and defines the
*measurement* protocol only:

* batch size 128, seed 0 -> exactly the batch order the seed-0 training loop
  produced (``torch.Generator().manual_seed(0)`` + ``randperm``),
* 20 warmup steps, 300 timed steps,
* the same frozen LCRC-C1 model, AdamW, per-step cosine scheduler and masked
  BCE loss as ``pc_train.py``.

Nothing here is fitted or tuned; changing a constant in this file changes what
was measured and must be recorded in ``PROTOCOL.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-pcba-pipeline-perf-v1"
RESULT_LABEL_ACCEPTED = "PERF_OPTIMIZATION_ACCEPTED"
RESULT_LABEL_NO_SPEEDUP = "NO_CLEAR_SPEEDUP"

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

#: The frozen round whose artifacts are measured (read-only, never modified).
PILOT = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
PILOT_CODE = PILOT / "code"
if str(PILOT_CODE) not in sys.path:
    sys.path.insert(0, str(PILOT_CODE))

import pc_common as PC  # noqa: E402  (side effect: puts bp/prep/tuq/mb2 code on sys.path)

ANALYSIS = ROUND / "analysis"
CACHE_PATH = PILOT / "cache" / "pcba_v1"
LABELS_PATH = PILOT / "analysis" / "pcba_labels_trainvalid.npy"
SPEC_PATH = PILOT / "analysis" / "pcba_spec.json"
SPLIT_PATH = PILOT / "analysis" / "pcba_split.json"
CACHE_MANIFEST = PILOT / "analysis" / "cache_manifest.json"
HOST_VERIFY = PILOT / "analysis" / "cache_host_verify.json"
PILOT_RUN_JSON = PILOT / "runs" / "pcba_seed0" / "run.json"

#: Fixed measurement protocol (declared before the first measurement).
BATCH_SIZE = int(PC.RECIPE["batch_size"])          # 128
SEED = int(PC.SEED)                                 # 0
WARMUP_STEPS = 20
TIMED_STEPS = 300
#: pure-data-pipeline micro-benchmark length (same batch order prefix)
DATALOADER_STEPS = 300
#: torch.profiler window (steps inside the timed pass)
PROFILER_STEPS = 20


# ---------------------------------------------------------------------------
# small IO helpers (kept local so this round never depends on pilot internals)
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_thread(rows: Sequence[np.ndarray]) -> str:
    h = hashlib.sha256()
    for arr in rows:
        h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.int64)).tobytes())
    return h.hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


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
            cwd=REPO, text=True, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def environment() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(cuda),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "utc": utcnow(),
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        out.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": int(props.total_memory),
                "gpu_multiprocessor_count": int(props.multi_processor_count),
                "gpu_capability": f"{props.major}.{props.minor}",
                "cudnn_version": int(torch.backends.cudnn.version() or 0),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            }
        )
    return out


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# fixed inputs
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    spec: Any
    cache: dict[str, Any]
    labels: np.ndarray
    split: dict[str, np.ndarray]
    train_rows: np.ndarray
    valid_rows: np.ndarray
    test_rows: np.ndarray
    norm: dict[str, Any]
    identity: dict[str, Any]


def verify_cache_identity() -> dict[str, Any]:
    """Cheap structural identity check (sizes + recorded tree hash, no re-hash).

    Re-hashing 4.5 GB would evict the page cache and change the first measured
    epoch, so the tree SHA-256 is taken from the pilot's own build/verify
    manifest and *file sizes* are checked live.  The limitation is recorded.
    """
    manifest = read_json(CACHE_MANIFEST)
    verify = read_json(HOST_VERIFY) if HOST_VERIFY.is_file() else {}
    merged = manifest["merged"]
    meta = read_json(CACHE_PATH / "meta.json")
    sizes = {p.name: int(p.stat().st_size) for p in sorted(CACHE_PATH.iterdir()) if p.is_file()}
    checks = {
        "meta_n_molecules": int(meta["n_molecules"]) == int(merged["n_molecules"]),
        "meta_n_atoms": int(meta["n_atoms_total"]) == int(merged["n_atoms_total"]),
        "meta_n_s2": int(meta["n_s2_paths"]) == int(merged["n_s2_paths"]),
        "meta_feature_spec_hash": str(meta["feature_spec_hash"]),
        "file_bytes_total": int(sum(sizes.values())) == int(merged["cache_bytes"]),
        "file_count": len(sizes) == int(merged["cache_files"]),
    }
    ok = all(v for k, v in checks.items() if isinstance(v, bool))
    return {
        "cache": track_rel(CACHE_PATH),
        "format": "npy-directory-v1",
        "cache_sha256_from_manifest": merged["cache_sha256"],
        "cache_sha256_recomputed_here": False,
        "cache_sha256_limitation": (
            "not re-hashed in this round: hashing 4.5 GB would evict the page cache "
            "and change the measured mmap read pattern"
        ),
        "pilot_host_verify_status": verify.get("status"),
        "n_molecules": int(meta["n_molecules"]),
        "n_atoms_total": int(meta["n_atoms_total"]),
        "n_bonds_total": int(meta["n_bonds_total"]),
        "n_s1_codes": int(meta["n_s1_codes"]),
        "n_s2_paths": int(meta["n_s2_paths"]),
        "feature_spec_hash": str(meta["feature_spec_hash"]),
        "bytes": int(sum(sizes.values())),
        "checks": {k: v for k, v in checks.items() if isinstance(v, bool)},
        "ok": bool(ok),
    }


def load_inputs() -> Inputs:
    import pc_data as DATA
    import bp_data as D
    from feature_spec import load_spec

    t0 = time.perf_counter()
    spec = load_spec(SPEC_PATH)
    split_payload = read_json(SPLIT_PATH)
    split = {k: np.asarray(split_payload["rows"][k], dtype=np.int64) for k in ("train", "valid", "test")}
    identity = verify_cache_identity()
    if not identity["ok"]:
        raise RuntimeError(f"cache identity check failed: {identity['checks']}")
    if identity["feature_spec_hash"] != spec.hash():
        raise RuntimeError("cache feature-spec hash != loaded spec hash")
    cache = DATA.load_cache(CACHE_PATH)
    labels = np.load(LABELS_PATH)
    if labels.shape[1] != PC.N_TASKS:
        raise RuntimeError(f"label artifact {labels.shape} is not [N,{PC.N_TASKS}]")
    # Test isolation, identical to pc_train.load_task.
    if np.isfinite(labels[split["test"]]).any():
        raise RuntimeError("label artifact exposes official Test targets")
    if not np.array_equal(np.asarray(cache["gids"]), np.arange(int(cache["n_molecules"]))):
        raise RuntimeError("cache rows are not the official row order")
    norm = D.fit_norm_rows(cache, spec, split["train"], source="pcba_official_train_only")
    inputs = Inputs(
        spec=spec,
        cache=cache,
        labels=labels,
        split=split,
        train_rows=split["train"],
        valid_rows=split["valid"],
        test_rows=split["test"],
        norm=norm,
        identity=identity,
    )
    log(f"inputs loaded in {time.perf_counter() - t0:.1f}s (train={len(split['train'])})")
    return inputs


def fixed_batches(
    train_rows: np.ndarray, n_batches: int, *, batch_size: int = BATCH_SIZE, seed: int = SEED
) -> list[np.ndarray]:
    """The first ``n_batches`` batches of the seed-0 training epoch order."""
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(train_rows), generator=generator).numpy()
    shuffled = np.asarray(train_rows, dtype=np.int64)[perm]
    batches = [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]
    return batches[:n_batches]


def batch_plan_fingerprint(batches: Sequence[np.ndarray]) -> dict[str, Any]:
    return {
        "n_batches": int(len(batches)),
        "batch_size": int(BATCH_SIZE),
        "seed": int(SEED),
        "rows_sha256": sha256_thread(batches),
        "first_batch": np.asarray(batches[0], dtype=np.int64).tolist()[:8],
        "last_batch_rows": int(len(batches[-1])),
        "source": (
            "torch.Generator().manual_seed(0) + torch.randperm(len(train_rows)) "
            "-> the exact prefix of the seed-0 epoch order in pc_train._batches"
        ),
    }


def batch_shape_table(cache: Mapping[str, Any], batches: Sequence[np.ndarray]) -> dict[str, list[int]]:
    """Per-batch atom / bond / S2 counts.

    ``atom_ptr`` and ``bond_ptr`` are per-molecule, ``path_ptr`` is per-atom, so
    the S2 count of a molecule is ``path_ptr[atom_ptr[m+1]] - path_ptr[atom_ptr[m]]``.
    """
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atoms, bonds, paths = [], [], []
    for rows in batches:
        r = np.asarray(rows, dtype=np.int64)
        atoms.append(int(atom_ptr[r + 1].sum() - atom_ptr[r].sum()))
        bonds.append(int(bond_ptr[r + 1].sum() - bond_ptr[r].sum()))
        paths.append(int(path_ptr[atom_ptr[r + 1]].sum() - path_ptr[atom_ptr[r]].sum()))
    return {"atoms": atoms, "bonds": bonds, "s2": paths}


def diverse_batches(cache: Mapping[str, Any], batches: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Fixed batches spanning different atom / bond / S2 counts."""
    table = batch_shape_table(cache, batches)
    atoms = np.asarray(table["atoms"])
    bonds = np.asarray(table["bonds"])
    s2 = np.asarray(table["s2"])
    order_atoms = np.argsort(atoms)
    candidates = [
        ("max_atoms", np.argsort(-atoms, kind="stable")),
        ("max_s2", np.argsort(-s2, kind="stable")),
        ("max_bonds", np.argsort(-bonds, kind="stable")),
        ("min_atoms", order_atoms),
        ("median_atoms", np.concatenate([order_atoms[len(atoms) // 2 :], order_atoms[: len(atoms) // 2]])),
    ]
    out: dict[str, np.ndarray] = {}
    used: set[int] = set()
    for name, order in candidates:
        idx = next(int(x) for x in order if int(x) not in used)
        out[name] = batches[idx]
        used.add(int(idx))
    out["typical"] = batches[0]
    return out


def fingerprint_like_batches(batches: Sequence[np.ndarray]) -> dict[str, Any]:
    """Shape statistics of the measured batches (diagnostic only)."""
    atoms = np.asarray([len(b) for b in batches], dtype=np.int64)
    return {
        "n_batches": int(len(batches)),
        "rows_per_batch_min": int(atoms.min()),
        "rows_per_batch_max": int(atoms.max()),
        "rows_total": int(atoms.sum()),
    }


__all__ = [
    "ANALYSIS",
    "BATCH_SIZE",
    "CACHE_PATH",
    "DATALOADER_STEPS",
    "Inputs",
    "LABELS_PATH",
    "PILOT",
    "PROFILER_STEPS",
    "REPO",
    "RESULT_LABEL_ACCEPTED",
    "RESULT_LABEL_NO_SPEEDUP",
    "ROUND",
    "ROUND_ID",
    "SEED",
    "SPEC_PATH",
    "SPLIT_PATH",
    "TIMED_STEPS",
    "WARMUP_STEPS",
    "batch_plan_fingerprint",
    "batch_shape_table",
    "diverse_batches",
    "environment",
    "fingerprint_like_batches",
    "fixed_batches",
    "git_commit",
    "load_inputs",
    "log",
    "read_json",
    "sha256_file",
    "sha256_thread",
    "track_rel",
    "utcnow",
    "verify_cache_identity",
    "write_json",
]
