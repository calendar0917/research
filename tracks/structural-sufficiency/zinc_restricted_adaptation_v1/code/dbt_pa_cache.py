"""Frozen local-stack feature cache for DBT-PA.

The adapters sit after the *complete* P3 local stack, so the base outputs
``a0, b0, w0`` are a deterministic, label-free function of the graph and the
SSL weights.  Caching them lets the supervised short schedule train only the
adapters and the DBT downstream.

Cache identity
--------------
The file name and the payload metadata both contain:

* ``FEATURE_VERSION`` (preprocessing / semantics version),
* the SSL checkpoint sha256,
* the sha256 of the loaded encoder state tensors,
* a structural fingerprint of the graph batch (features, indices, pointers;
  labels are deliberately excluded).

A cache from a different SSL checkpoint, encoder state or graph batch cannot
be loaded silently.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = _SSUF / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_train as AT  # noqa: E402
import dbt_data as BD  # noqa: E402

FEATURE_VERSION = "dbt-pa-feat-v1"
CACHE = _EXP / "cache"
STRUCT_FIELDS = (
    "atom_ptr", "atom_feat", "atom_type",
    "bond_ptr", "bond_feat", "bond_type", "bond_index",
    "junc_ptr", "junc_feat", "junc_index", "junc_bond",
    "global_feat", "n_atoms_graph", "n_bonds_graph",
)


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def structural_fingerprint(object_cache: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(FEATURE_VERSION.encode())
    h.update(str(int(object_cache["n_molecules"])).encode())
    for key in STRUCT_FIELDS:
        arr = np.ascontiguousarray(np.asarray(object_cache[key]))
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(tuple(arr.shape)).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def cache_identity(split: str, *, ssl_sha256: str, encoder_sha256: str,
                   structure_sha256: str) -> dict[str, str]:
    return {
        "feature_version": FEATURE_VERSION,
        "split": str(split),
        "ssl_sha256": str(ssl_sha256),
        "encoder_sha256": str(encoder_sha256),
        "structure_sha256": str(structure_sha256),
    }


def feature_cache_path(identity: Mapping[str, str]) -> Path:
    return CACHE / (
        f"{identity['feature_version']}-"
        f"{identity['ssl_sha256'][:10]}-"
        f"{identity['encoder_sha256'][:10]}-"
        f"{identity['structure_sha256'][:10]}_"
        f"{identity['split']}.pt"
    )


def _split_fill(dest: np.ndarray, src: torch.Tensor, ptr: np.ndarray,
                rows: np.ndarray) -> None:
    arr = src.detach().float().cpu().numpy()
    pos = 0
    for row in rows:
        lo, hi = int(ptr[row]), int(ptr[row + 1])
        n = hi - lo
        if n and pos + n > len(arr):
            raise RuntimeError("batch feature block is shorter than the graph pointers")
        dest[lo:hi] = arr[pos:pos + n]
        pos += n
    if pos != len(arr):
        raise RuntimeError(f"feature block length {len(arr)} != consumed {pos}")


def build_feature_split(split: str, model, device, *,
                        ssl_sha256: str, encoder_sha256: str,
                        batch_size: int = 128) -> tuple[dict[str, Any], Path]:
    """Run the frozen local stack once over a split and save its outputs."""
    object_cache = BD.load_object_cache(split)
    # The compact DBT object cache is byte-identical to the earlier rounds'
    # ``ad_*`` cache on every field read here (verified by the DBT test
    # harness); it keeps the frozen-feature cache on the same graph objects.
    ap = np.asarray(object_cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(object_cache["bond_ptr"], dtype=np.int64)
    jp = np.asarray(object_cache["junc_ptr"], dtype=np.int64)
    n_mol = int(object_cache["n_molecules"])
    a = np.empty((int(ap[-1]), 96), dtype=np.float32)
    b = np.empty((int(bp[-1]), 96), dtype=np.float32)
    w = np.empty((int(jp[-1]), 96), dtype=np.float32)

    model.eval()
    for p in model.encoders.parameters():
        p.requires_grad_(False)
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n_mol, int(batch_size)):
            rows = np.arange(start, min(start + int(batch_size), n_mol), dtype=np.int64)
            batch = _to_device(AT.collate(object_cache, rows, "T", valid=split != "train"),
                               device)
            a_b, b_b, w_b = model.encoders(batch)
            _split_fill(a, a_b, ap, rows)
            _split_fill(b, b_b, bp, rows)
            _split_fill(w, w_b, jp, rows)
    seconds = time.perf_counter() - t0

    identity = cache_identity(
        split, ssl_sha256=ssl_sha256, encoder_sha256=encoder_sha256,
        structure_sha256=structural_fingerprint(object_cache),
    )
    data: dict[str, Any] = {
        "split": split,
        "n_molecules": n_mol,
        "a": a, "b": b, "w": w,
        "atom_ptr": ap, "bond_ptr": bp, "junc_ptr": jp,
        "meta": {
            **identity,
            "kind": "frozen_local_stack_outputs",
            "cache_inputs": "raw graph -> frozen P3 encoders (SSL initialised)",
            "stores": "base outputs a0/b0/w0, not adapted states and not labels",
            "dtype": "float32",
            "n_atoms_total": int(ap[-1]),
            "n_bonds_total": int(bp[-1]),
            "n_junctions_total": int(jp[-1]),
            "batch_size": int(batch_size),
            "build_seconds": float(seconds),
            "created": C.utcnow(),
        },
    }
    path = feature_cache_path(identity)
    CACHE.mkdir(parents=True, exist_ok=True)
    C.atomic_torch_save(data, path)
    data["_path"] = str(path)
    data["_sha256"] = C.sha256_file(path)
    return data, path


def load_feature_split(split: str, model, *, ssl_sha256: str,
                       encoder_sha256: str, structure_sha256: str,
                       require: bool = True) -> dict[str, Any] | None:
    identity = cache_identity(
        split, ssl_sha256=ssl_sha256, encoder_sha256=encoder_sha256,
        structure_sha256=structure_sha256,
    )
    path = feature_cache_path(identity)
    if not path.exists():
        if require:
            raise FileNotFoundError(
                f"missing frozen-feature cache {path}; run the 'prep' command"
            )
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    meta = data.get("meta", {})
    for key, value in identity.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"feature cache identity mismatch for {path}: {key} "
                f"{meta.get(key)!r} != {value!r}"
            )
    if int(data["n_molecules"]) < 1:
        raise RuntimeError(f"empty feature cache {path}")
    for key in ("a", "b", "w", "atom_ptr", "bond_ptr", "junc_ptr"):
        if key not in data:
            raise RuntimeError(f"feature cache {path} is missing {key}")
    for key, ptr_key, width in (("a", "atom_ptr", 96), ("b", "bond_ptr", 96),
                                ("w", "junc_ptr", 96)):
        arr = np.asarray(data[key])
        ptr = np.asarray(data[ptr_key])
        if arr.ndim != 2 or arr.shape[1] != width or arr.shape[0] != int(ptr[-1]):
            raise RuntimeError(
                f"feature cache {path}: {key} shape {arr.shape} != ({int(ptr[-1])}, {width})"
            )
    data["_path"] = str(path)
    data["_sha256"] = C.sha256_file(path)
    return data


def gather_cached_features(feature: Mapping[str, Any], rows: Sequence[int],
                           device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = np.asarray(rows, dtype=np.int64)
    out = []
    for key, ptr_key in (("a", "atom_ptr"), ("b", "bond_ptr"), ("w", "junc_ptr")):
        ptr = np.asarray(feature[ptr_key], dtype=np.int64)
        arr = np.asarray(feature[key])
        parts = [arr[int(ptr[r]):int(ptr[r + 1])] for r in rows]
        cat = np.concatenate(parts, axis=0) if parts else np.zeros((0, arr.shape[1]),
                                                                  dtype=arr.dtype)
        out.append(torch.from_numpy(np.ascontiguousarray(cat)).to(device))
    return out[0], out[1], out[2]


def verify_feature_split(model, split: str, feature: Mapping[str, Any],
                         rows: Sequence[int], device, *, atol: float = 1e-5) -> dict[str, Any]:
    """Compare cached base outputs with a live frozen-stack forward."""
    object_cache = BD.load_object_cache(split)
    rows = np.asarray(rows, dtype=np.int64)
    batch = _to_device(AT.collate(object_cache, rows, "T", valid=split != "train"), device)
    model.eval()
    with torch.no_grad():
        a, b, w = model.encoders(batch)
    ca, cb, cw = gather_cached_features(feature, rows, device)
    checks = {}
    for name, live, cached in (("a", a, ca), ("b", b, cb), ("w", w, cw)):
        if live.shape != cached.shape:
            raise RuntimeError(f"{split} {name} shape {live.shape} != {cached.shape}")
        delta = (live.detach() - cached.detach()).abs()
        checks[name] = {
            "max_abs_diff": float(delta.max().item()) if delta.numel() else 0.0,
            "mean_abs_diff": float(delta.mean().item()) if delta.numel() else 0.0,
            "allclose": bool(torch.allclose(live, cached, atol=atol, rtol=1e-5)),
            "shape": list(live.shape),
        }
    checks.update({
        "split": split,
        "rows": [int(x) for x in rows],
        "atol": float(atol),
        "feature_cache_sha256": feature["_sha256"],
        "all_pass": all(v["allclose"] for v in checks.values()
                        if isinstance(v, dict) and "allclose" in v),
    })
    return checks
