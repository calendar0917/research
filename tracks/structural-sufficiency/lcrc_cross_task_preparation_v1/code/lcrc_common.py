"""Shared paths, hashing and provenance helpers for the LCRC cross-task
preparation round.

This module deliberately contains no chemistry, no feature construction and
no model code.  It is the only place that knows repository paths.

Scope discipline
----------------
The Lipophilicity side of this round reads only:

* the committed label-free raw-graph artifact
  ``tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz``;
* the frozen structure-only split ``results/phase1a/split_phase1a.json``;
* the label-free row manifest (compound ids, canonical-SMILES hashes, atom
  counts, component counts, scaffold hashes).

It never reads ``labels_fit_monitor.json``, the ``exp`` column of the CSV,
``labels.npy`` or any residue of ``split_lipo_v1.json`` after the row-index
arrays.  The CSV is hashed as a byte stream only, never parsed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROUND_ID = "lcrc-cross-task-preparation-v1"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

ANALYSIS = EXP_DIR / "analysis"
CACHE = EXP_DIR / "cache"
TESTS = EXP_DIR / "tests"

# --- frozen label-free Lipophilicity sources ---------------------------------
LIPO_RAW = (
    REPO
    / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz"
)
LIPO_SPLIT = REPO / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"
LIPO_ROW_MANIFEST = (
    REPO
    / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/row_manifest.json"
)
LIPO_OFFICIAL_SPLIT = (
    REPO
    / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/split_lipo_v1.json"
)
LIPO_CSV = REPO / "data/lipophilicity/Lipophilicity.csv"

# --- frozen ZINC legacy artifacts (read-only) -------------------------------
P3_TRAIN = REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/objects_train10k.pt"
P3_VALID = REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/objects_valid.pt"
LEGACY_ROUND = REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1"
LEGACY_CKPT = LEGACY_ROUND / "runs/lcrcA_s0/best.pt"
LEGACY_NORM = LEGACY_ROUND / "analysis/norm.json"
LEGACY_CODE = LEGACY_ROUND / "code"
LEGACY_CPU_CHECKS = LEGACY_ROUND / "analysis/cpu_checks.json"
P3_CODE = REPO / "tracks/structure-contribution/code"

#: Files that must never be opened by this round.
FORBIDDEN_LABEL_FILES = (
    REPO / "tracks/structural-sufficiency/results/phase1a/labels_fit_monitor.json",
    REPO / "data/lipophilicity/labels.npy",
    REPO / "data/lipophilicity/labels_meta.json",
)

#: Expected source hashes recorded by the frozen Phase 1a split.
EXPECTED_HASHES = {
    "lipo_raw": "8416837ea3e56e3e5218865391c88913bb16948b48b7a61be1ee890ea4dda6dd",
    "lipo_split": "9fa96d49efd94cd2ff381922f83a07f963d832064c75469657b4ce5f68c332b6",
    "row_manifest": "2c638e4814cb6d21ccec9d8c8af357d28ce210fb2c20c3d09b9c19afd476073d",
    "lipo_csv": "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd",
}

#: Frozen row hashes from the Phase 1a split.
EXPECTED_ROW_HASHES = {
    "fit": "c474e38cdecf10071965527409901403ec8b670b96fe4a3c42452531dd0210e9",
    "monitor": "fe81cc7904a6da9378f5bf5ae7452ac71629adad83a2a98b917f87299a85f39f",
    "reserved": "bc28c5d37c2fe2fbe39c01d5bf2ab9c56630eee2d0a4454a108d150ab2f42d86",
}

EXPECTED_ZINC_OBJECT_HASHES = {
    "train": "14326a7ae6f9bb673a5c8b35c7d54171a4008f77edb702f0591d3d1f21338e8d",
    "valid": "61fca36a097f859b72a29e5f656405bff87a54ecf09f74b2727aa9663fdb853c",
}


# ---------------------------------------------------------------------------
# deterministic I/O
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


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def row_sha256(rows: Iterable[int]) -> str:
    """Phase 1a row-index hash convention: comma-separated decimal string."""
    return hashlib.sha256(",".join(str(int(r)) for r in rows).encode("utf-8")).hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default)


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True, default=_json_default)
    p.write_text(text + "\n", encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# cache content hash
# ---------------------------------------------------------------------------


def content_fingerprint(cache: Mapping[str, Any]) -> str:
    """Stable hash over every array in a cache, ignoring metadata strings.

    The key, dtype, shape and raw bytes are hashed.  This is the value used to
    prove that two independent rebuilds of a split produced identical
    structure tensors.
    """
    h = hashlib.sha256()
    for key in sorted(cache):
        value = cache[key]
        if isinstance(value, np.ndarray):
            h.update(key.encode("utf-8"))
            h.update(str(value.dtype).encode("ascii"))
            h.update(str(tuple(value.shape)).encode("ascii"))
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def array_summary(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
    }


# ---------------------------------------------------------------------------
# safe row-index extraction from the official split JSON
# ---------------------------------------------------------------------------


def stream_json_array_fields(path: Path | str, fields: Iterable[str], chunk_size: int = 1024) -> dict[str, Any]:
    """Read selected top-level list fields without parsing later JSON fields.

    ``split_lipo_v1.json`` stores the official row-index arrays before the
    train-label array.  This reader stops as soon as the requested arrays have
    been decoded, so the label tail is never parsed.  It returns the arrays and
    an audit record of how many bytes the file object handed back.
    """
    fields = list(dict.fromkeys(str(f) for f in fields))
    requested = list(fields)
    out: dict[str, Any] = {}
    bytes_read = 0
    saw_train_y = False
    buf = ""
    eof = False
    pos = 0
    with open(path, "r", encoding="utf-8") as fh:
        while fields:
            found = False
            for field in list(fields):
                key = f'"{field}"'
                idx = buf.find(key, pos)
                if idx < 0:
                    continue
                colon = buf.find(":", idx + len(key))
                if colon < 0:
                    continue
                start = colon + 1
                while start < len(buf) and buf[start].isspace():
                    start += 1
                try:
                    value, end = json.JSONDecoder().raw_decode(buf, start)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, list):
                    raise ValueError(f"{field} is not a JSON list")
                out[field] = value
                fields.remove(field)
                pos = end
                found = True
                break
            if found:
                continue
            if eof:
                break
            chunk = fh.read(chunk_size)
            if not chunk:
                eof = True
                continue
            bytes_read += len(chunk)
            buf += chunk
            if '"train_y"' in buf:
                saw_train_y = True
    out["_audit"] = {
        "path": track_rel(path),
        "sha256": sha256_file(path),
        "fields": [k for k in out if not str(k).startswith("_")],
        "bytes_read_by_reader": int(bytes_read),
        "requested_fields": requested,
        "fields_not_found": list(fields),
        "buffer_contains_train_y_key": bool(saw_train_y),
        "label_values_parsed": False,
    }
    return out


# ---------------------------------------------------------------------------
# environment / repository provenance
# ---------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return True


def env_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "round_id": ROUND_ID,
        "timestamp": utcnow(),
    }
    try:
        import networkx

        info["networkx"] = networkx.__version__
    except Exception:
        info["networkx"] = None
    try:
        import rdkit

        info["rdkit"] = rdkit.__version__
    except Exception:
        info["rdkit"] = None
    return info


def peak_rss_mb() -> float:
    """Process peak resident set size in MiB (Linux ``ru_maxrss`` is KiB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_maxrss) / 1024.0


def file_info(path: Path | str, *, hash_file: bool = True) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": track_rel(p), "exists": False}
    out: dict[str, Any] = {
        "path": track_rel(p),
        "exists": True,
        "bytes": int(p.stat().st_size),
    }
    if hash_file:
        out["sha256"] = sha256_file(p)
    return out


__all__ = [
    "ROUND_ID",
    "CODE_DIR",
    "EXP_DIR",
    "REPO",
    "ANALYSIS",
    "CACHE",
    "TESTS",
    "LIPO_RAW",
    "LIPO_SPLIT",
    "LIPO_ROW_MANIFEST",
    "LIPO_OFFICIAL_SPLIT",
    "LIPO_CSV",
    "P3_TRAIN",
    "P3_VALID",
    "LEGACY_ROUND",
    "LEGACY_CKPT",
    "LEGACY_NORM",
    "LEGACY_CODE",
    "LEGACY_CPU_CHECKS",
    "P3_CODE",
    "FORBIDDEN_LABEL_FILES",
    "EXPECTED_HASHES",
    "EXPECTED_ROW_HASHES",
    "EXPECTED_ZINC_OBJECT_HASHES",
    "utcnow",
    "sha256_file",
    "sha256_bytes",
    "sha256_array",
    "row_sha256",
    "stable_json",
    "write_json",
    "read_json",
    "track_rel",
    "log",
    "ensure_dir",
    "content_fingerprint",
    "array_summary",
    "stream_json_array_fields",
    "git_commit",
    "git_dirty",
    "env_info",
    "peak_rss_mb",
    "file_info",
]
