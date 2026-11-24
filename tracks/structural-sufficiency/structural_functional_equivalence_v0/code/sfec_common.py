"""Shared paths, frozen config, identity and IO helpers for SFEC v0.

No scientific quantity is chosen here.  All thresholds live in
``configs/sfec_v0.json`` which was committed together with ``PROTOCOL.md``
before any label analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
CONFIG_PATH = ROUND_DIR / "configs" / "sfec_v0.json"
DATA_DIR = ROUND_DIR / "data"
CHUNK_DIR = ROUND_DIR / "chunks"
ANALYSIS_DIR = ROUND_DIR / "analysis"
for _d in (DATA_DIR, CHUNK_DIR, ANALYSIS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ZINC_ROOT = REPO_ROOT / "data" / "ZINC"
PCQM_ROOT = REPO_ROOT / "data" / "ogb" / "pcqm4m-v2"
PCQM_TRAIN_CACHE = (
    REPO_ROOT
    / "tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/cache/pcqm_train1m"
)
PCQM_VALID_CACHE = (
    REPO_ROOT
    / "tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/cache/pcqm_valid"
)
HIST_FRAGMENT_CODE = (
    REPO_ROOT
    / "tracks/structural-sufficiency/response_dictionary_small_s0_v1/code/rd_fragment.py"
)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_sanitize_json(payload), indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path | str, rows: Iterable[Mapping[str, Any]], columns=None) -> Path:
    import csv

    path = Path(path)
    rows = list(rows)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})
    tmp.replace(path)
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.10g}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return ""
    return value


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "UNKNOWN"


def environment_snapshot() -> dict[str, Any]:
    import rdkit

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "rdkit": rdkit.__version__,
        "git_commit": git_commit(),
        "pid": os.getpid(),
        "blas_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
    }


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# labels (only called after structural artifacts are frozen)
# ---------------------------------------------------------------------------


def load_pcqm_labels() -> np.ndarray:
    """Raw HOMO-LUMO gap by global row index, from the official CSV.

    The file has 3,746,620 data rows.  Only the label column is kept.
    """
    import gzip

    path = PCQM_ROOT / "raw" / "data.csv.gz"
    labels = np.full(3_746_620, np.nan, dtype=np.float64)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        assert header == ["idx", "smiles", "homolumogap"], header
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 3:
                continue
            idx = int(parts[0])
            if parts[2].strip():
                labels[idx] = float(parts[2])
    return labels


def load_pcqm_split() -> dict[str, np.ndarray]:
    import torch

    raw = torch.load(PCQM_ROOT / "split_dict.pt", map_location="cpu", weights_only=False)
    return {str(k): np.asarray(v, dtype=np.int64).reshape(-1) for k, v in raw.items()}


def load_zinc_split(split: str):
    from torch_geometric.datasets import ZINC

    pyg_split = "val" if split == "valid" else split
    return ZINC(root=str(ZINC_ROOT), subset=True, split=pyg_split)


def zinc_split_sizes() -> dict[str, int]:
    return {"train": 10000, "valid": 1000, "test": 1000}


def zinc_arrays(dataset) -> list[dict[str, Any]]:
    """Materialise the small official ZINC subset into plain numpy records."""
    out = []
    for i in range(len(dataset)):
        d = dataset[i]
        out.append(
            {
                "row_id": i,
                "x": d.x.reshape(-1).numpy().astype(np.int64),
                "edge_index": d.edge_index.numpy().astype(np.int64),
                "edge_attr": d.edge_attr.reshape(-1).numpy().astype(np.int64),
                "y": float(d.y.reshape(-1)[0]),
            }
        )
    return out


def read_pcqm_smiles(cache: Path, rows: np.ndarray) -> list[str]:
    """SMILES at the given cache row positions (cache order, not label row ids)."""
    import json as _json

    wanted = set(int(r) for r in rows)
    out = {}
    with (cache / "smiles.jsonl").open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i in wanted:
                out[i] = _json.loads(line)
                if len(out) == len(wanted):
                    break
    return [out[int(r)] for r in rows]


def pcqm_cache_gids(cache: Path) -> np.ndarray:
    return np.load(cache / "gids.npy").astype(np.int64)


def context_hash_hex(key: str, salt: int = 0) -> str:
    return hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()


def context_split(key: str, salt: int = 0, discovery_percent: int = 80) -> bool:
    """True for discovery, False for evaluation."""
    h = context_hash_hex(key, salt)
    return (int(h[:16], 16) % 100) < discovery_percent


def stable_key_order(keys: list[str] | np.ndarray) -> np.ndarray:
    return np.asarray(sorted(range(len(keys)), key=lambda i: keys[i]), dtype=np.int64)

def zinc_labels(split: str) -> np.ndarray:
    ds = load_zinc_split(split)
    return np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(len(ds))], dtype=np.float64)
