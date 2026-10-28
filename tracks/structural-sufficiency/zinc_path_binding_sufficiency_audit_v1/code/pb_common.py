"""Shared plumbing for the path-binding sufficiency audit (CPU only, no labels).

The audit is *read-only* on the LCRC-v1 round: it imports ``lc_common``,
``lc_data`` and ``lc_models`` from ``zinc_local_relation_composition_v1/code``
and never writes there.

Label discipline is enforced mechanically: :func:`load_cache` replaces
``cache["labels"]`` with :class:`_Poison`, so any code path that touches a label
raises at access time.  The official test cache is never loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

AUDIT_DIR = Path(__file__).resolve().parent.parent
REPO = AUDIT_DIR.parents[2]
LCRC = REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1"
LCRC_CODE = LCRC / "code"
if str(LCRC_CODE) not in sys.path:
    sys.path.insert(0, str(LCRC_CODE))

import lc_common as L  # noqa: E402
import lc_data as D  # noqa: E402
import lc_models as M  # noqa: E402

import torch  # noqa: E402

CACHE = AUDIT_DIR / "cache"
ANALYSIS = AUDIT_DIR / "analysis"
CODE = AUDIT_DIR / "code"

TORCH_THREADS = max(1, min(16, os.cpu_count() or 1))
torch.set_num_threads(TORCH_THREADS)

ROUND_ID = "zinc-path-binding-sufficiency-audit-v1"
FEATURE_VERSION = "pb-audit-v1-Treal-Tnull-D-type-control"
N_PROJ_DIM = 32
N_ATOM_OUT = 96
X_DIM = 529
S_DIM = X_DIM - N_ATOM_OUT  # 433
H_DIM = 64

#: fixed, label-free, never searched (see PROTOCOL.md §4)
PROJ_SEEDS = (20261030, 20261031, 20261032)
PRIMARY_SEED = PROJ_SEEDS[0]

#: train-internal monitor split for reconstructor selection (molecules)
MONITOR_SEED = 20261029
MONITOR_MOLS = 2000
#: target-row permutation control
PERM_SEED = 20261101
MLP_SEED = 20261030

#: support-coverage thresholds (radius-2 keys, counted in train molecules only)
SUP_LOW_MAX = 3  # support 1..3 -> low, 0 -> unseen, >=4 -> seen

CHECKPOINTS = {
    "lcrcA_s0": {
        "run": "lcrcA_s0",
        "sha256_prefix": "06340bcd9e45af03",
        "seed": 0,
        "best_valid": 0.07658163924043765,
    },
    "lcrcA_s1": {
        "run": "lcrcA_s1",
        "sha256_prefix": "7225fe11fce0a24d",
        "seed": 1,
        "best_valid": 0.08050377571111313,
    },
}

#: frozen cache fingerprints from LCRC-v1 configs/lcrc_v1.json (checked, not trusted)
EXPECTED_CACHE_SHA = {
    "train": "8777494f7745ff5a9a8e3d2913cae8102ad84d267e54fb65235d7eff4c3a6cb2",
    "valid": "aae2caa50d322a25ce75cbd8160e61a831b1488d43295e872db31d0bc7a6dbfe",
}
EXPECTED_NORM_FP = "8c3e88137f18ca6841758b5779b57d7362d9dd57c2c29db2067f822cafd1c89f"


class _Poison:
    """Any access to a label raises: the audit must be label-free."""

    def _boom(self, *a, **k):
        raise RuntimeError("LABEL_ACCESS: this audit is label-free (PROTOCOL.md §2)")

    __getitem__ = __iter__ = __len__ = __array__ = __bool__ = _boom
    shape = property(lambda self: self._boom())
    dtype = property(lambda self: self._boom())


class Plot:
    """Tiny stage timer."""

    def __init__(self):
        self.rows = []
        self.t0 = time.time()

    def mark(self, name: str, **extra) -> None:
        self.rows.append({"stage": name, "t": round(time.time() - self.t0, 3),
                          "since_start_s": round(time.time() - self.t0, 3), **extra})
        print(f"[pb {self.rows[-1]['t']:8.1f}s] {name}", flush=True)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_json(path: Path | str, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, ensure_ascii=False)
    tmp.replace(p)


def read_json(path: Path | str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "short": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        "tracked_dirty_files": (run("status", "--porcelain", "--untracked-files=no") or "").splitlines(),
    }


def env_state() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": TORCH_THREADS,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
    }


def load_cache(split: str) -> dict:
    """Load a frozen LCRC cache; labels are poisoned; sha256 is verified."""
    if split not in ("train", "valid"):
        raise ValueError(f"audit may only load train/valid, not {split!r}")
    path = L.CACHE / f"lc_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen LCRC cache {path}")
    sha = sha256_file(path)
    expected = EXPECTED_CACHE_SHA.get(split)
    if expected and sha != expected:
        raise RuntimeError(f"{path}: sha256 {sha} != frozen {expected}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data["meta"].get("feature_version") != L.FEATURE_VERSION:
        raise RuntimeError(f"{split}: unexpected feature version {data['meta'].get('feature_version')}")
    if "labels" in data:
        data["labels"] = _Poison()
    data["_sha256"] = sha
    data["_path"] = L.track_rel(path)
    return data


def load_norm() -> dict:
    norm = D.load_norm()
    fp = D.norm_fingerprint(norm)
    if fp != EXPECTED_NORM_FP:
        raise RuntimeError(f"norm fingerprint {fp} != frozen {EXPECTED_NORM_FP}")
    norm["_fingerprint"] = fp
    return norm


def load_frozen_a(name: str) -> tuple[M.LCRC, dict]:
    """Load one frozen LCRC-A checkpoint in eval mode, verifying its sha256."""
    if name not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint {name!r}")
    spec = CHECKPOINTS[name]
    path = L.REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1/runs" / spec["run"] / "best.pt"
    sha = sha256_file(path)
    if not sha.startswith(spec["sha256_prefix"]):
        raise RuntimeError(f"{name}: sha256 {sha} does not start with frozen {spec['sha256_prefix']}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["arm"] != "A":
        raise RuntimeError(f"{name}: arm {ckpt['arm']!r} is not A")
    model = M.load_model_from_state("A", load_norm(), ckpt["state_dict"], "cpu", cfg=ckpt["config"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    info = {
        "name": name,
        "path": L.track_rel(path),
        "sha256": sha,
        "seed": int(ckpt["seed"]),
        "epoch": int(ckpt["epoch"]),
        "valid_mae_frozen": float(ckpt["valid_mae"]),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "config": ckpt["config"],
    }
    return model, info


def projections(seed: int) -> dict[str, np.ndarray]:
    """Fixed random projections; Normal(0, 1/d_in), float64, label-free."""
    rng = np.random.default_rng(int(seed))
    wq = rng.normal(0.0, 1.0, size=(N_PROJ_DIM, 2 * N_ATOM_OUT)) / np.sqrt(2 * N_ATOM_OUT)
    wk = rng.normal(0.0, 1.0, size=(N_PROJ_DIM, 2 * N_ATOM_OUT)) / np.sqrt(2 * N_ATOM_OUT)
    wv = rng.normal(0.0, 1.0, size=(N_PROJ_DIM, N_ATOM_OUT)) / np.sqrt(N_ATOM_OUT)
    return {"W_Q": wq, "W_K": wk, "W_V": wv}


def proj_fingerprint(proj: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in ("W_Q", "W_K", "W_V"):
        h.update(np.ascontiguousarray(proj[k], dtype=np.float64).tobytes())
    return h.hexdigest()


def seg_sum(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, values.shape[1]), dtype=values.dtype)
    np.add.at(out, index, values)
    return out


__all__ = [
    "AUDIT_DIR", "REPO", "CACHE", "ANALYSIS", "CODE", "LCRC", "L", "D", "M", "torch",
    "ROUND_ID", "FEATURE_VERSION", "N_PROJ_DIM", "N_ATOM_OUT", "X_DIM", "S_DIM",
    "H_DIM", "PROJ_SEEDS", "PRIMARY_SEED", "MONITOR_SEED", "MONITOR_MOLS",
    "PERM_SEED", "MLP_SEED", "SUP_LOW_MAX", "CHECKPOINTS", "EXPECTED_CACHE_SHA",
    "EXPECTED_NORM_FP", "TORCH_THREADS", "Plot", "sha256_file", "sha256_array",
    "write_json", "read_json", "git_state", "env_state", "load_cache", "load_norm",
    "load_frozen_a", "projections", "proj_fingerprint", "seg_sum",
]
