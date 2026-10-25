"""Shared plumbing for RCC-v0 (relational contrast contraction).

Everything here is label-free except ``fit_labels`` / ``monitor_labels``, which
are only ever indexed with the corresponding row list.  Official valid and test
caches are never opened.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
TRACK = _HERE.parent
REPO = TRACK.parents[2]
_PS = REPO / "tracks/structural-sufficiency"
_V1 = _PS / "zinc_arch_discovery_v1" / "code"
_V2 = _PS / "zinc_recipe_trunk_v2" / "code"
_DCC = _PS / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_V2), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402

CONFIG_PATH = TRACK / "configs" / "rcc_v0.json"
ANALYSIS = TRACK / "analysis"
RUNS = TRACK / "runs"
CACHE = REPO / "tracks/structural-sufficiency/zinc_arch_discovery_v1/cache"
BANK_PATH = ANALYSIS / "kernel_bank.npz"

WIDTH = 96
RANK = 16
T_DIM = RANK * (RANK + 1) // 2
N_STRUCT = 5
BMAX = 41


def config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def cfg_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash(module: torch.nn.Module) -> str:
    return C.state_hash(module)


def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return C.cpu_state(module)


def log(msg: str) -> None:
    C.log(msg)


def git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def git_dirty() -> bool:
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout
    return bool(out.strip())


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# split E
# ---------------------------------------------------------------------------


def load_split() -> dict[str, Any]:
    """Split E with rows already mapped to official-train (cache) row ids."""
    raw = read_json(REPO / config()["split"]["path"])
    rows_sorted = np.asarray(raw["rows_sorted"], dtype=np.int64)
    out = {
        "protocol": raw["protocol"],
        "salt": raw["salt"],
        "rows_sorted": rows_sorted,
        "split_index": {},
        "cache_rows": {},
    }
    for zone in ("fit", "monitor", "locked"):
        # split_E rows are already official-train row ids (rows_sorted is only a
        # sorted list of the surviving ids, kept for the audit trail).
        idx = np.asarray(raw[f"{zone}_rows"], dtype=np.int64)
        out["split_index"][zone] = np.arange(len(idx))
        out["cache_rows"][zone] = idx
    allrows = np.concatenate([out["cache_rows"][z] for z in ("fit", "monitor", "locked")])
    if len(set(allrows.tolist())) != len(allrows):
        raise RuntimeError("split E zones overlap")
    out["excluded_rows"] = np.asarray(raw["excluded_rows"], dtype=np.int64)
    return out


def load_train_cache() -> dict[str, Any]:
    """The 10k-row official *train* cache.  Never loads the valid cache."""
    return torch.load(CACHE / "ad_train.pt", map_location="cpu", weights_only=False)


def labels(cache: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(cache["labels"], dtype=np.float64)


def assert_rows_allowed(rows: Sequence[int], split: Mapping[str, Any]) -> None:
    """Locked rows must never enter the training loop."""
    locked = set(int(r) for r in split["cache_rows"]["locked"])
    bad = [int(r) for r in rows if int(r) in locked]
    if bad:
        raise RuntimeError(f"locked rows used outside scoring: {bad[:5]}")


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def load_bank() -> tuple[np.ndarray, np.ndarray]:
    """(kpad [10000, BMAX, BMAX] float32, struct [10000, 5] float32)."""
    z = np.load(BANK_PATH)
    return z["kpad"], z["struct"]


def kernel_from_bonds(bond_index: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """float64 (K_R, d) for one molecule; see rcc_kernels.py for the audit copy."""
    import rcc_kernels as K

    bi = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    return K.build_kernel(bi, int(n_atoms))


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------


def make_collate(bank: tuple[np.ndarray, np.ndarray] | None = None):
    """Wrap the exact P3N layout and add the relation kernel, mask and stats."""
    kpad, struct = load_bank() if bank is None else bank
    n_rows = kpad.shape[0]
    max_bonds = kpad.shape[1]

    def collate(cache, rows, candidate="P3N", *, valid=False, walk_cols=None):
        batch = T1.collate(cache, rows, "P3N", valid=valid, walk_cols=walk_cols)
        r = np.asarray(rows, dtype=np.int64)
        if r.max() >= n_rows:
            raise IndexError("row outside the frozen kernel bank")
        bptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
        counts = (bptr[r + 1] - bptr[r]).astype(np.int64)
        bmax = int(counts.max()) if len(counts) else 0
        bmax = max(bmax, 1)
        if bmax > max_bonds:
            raise IndexError(f"bmax {bmax} > bank width {max_bonds}")
        batch["k_pad"] = torch.from_numpy(np.ascontiguousarray(kpad[r][:, :bmax, :bmax]))
        batch["bond_mask"] = (
            torch.arange(bmax).view(1, bmax) < torch.from_numpy(counts).view(-1, 1)
        )
        batch["struct"] = torch.from_numpy(np.ascontiguousarray(struct[r]))
        offs = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        batch["bond_local"] = torch.from_numpy(np.arange(int(counts.sum())) - np.repeat(offs, counts))
        batch["bond_counts"] = torch.from_numpy(counts)
        return batch

    return collate


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# recipe
# ---------------------------------------------------------------------------


def recipe() -> dict[str, Any]:
    return dict(config()["train"])


def steps_per_epoch(n_fit: int) -> int:
    return int(np.ceil(n_fit / int(recipe()["batch_size"])))


@torch.no_grad()
def predict(model, cache, rows, *, device, collate_fn, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = np.zeros(len(rows), dtype=np.float64)
    order = np.arange(len(rows))
    for start in range(0, len(order), batch_size):
        sel = order[start:start + batch_size]
        batch = to_device(collate_fn(cache, np.asarray(rows)[sel]), device)
        pred = model(batch)
        out[sel] = pred.detach().double().view(-1).cpu().numpy()
    return out


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)).mean())
