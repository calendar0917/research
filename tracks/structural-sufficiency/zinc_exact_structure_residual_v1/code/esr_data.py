"""Exact rooted-patch type residual — data / vocabulary layer.

Object
------
For every atom ``v`` of a raw ZINC molecule we build two rooted attributed
induced patches from the *raw* graph only (``x`` / ``edge_index`` /
``edge_attr``):

* ``r2(v)`` — all atoms with graph distance <= 2 from ``v``, the induced
  subgraph on them, every bond in the induced subgraph, atom types and bond
  types as colours, and the root / root-distance designation as colours.
* ``r1(v)`` — the same construction with radius 1.  This is exactly the r2
  patch truncated to distance <= 1 and re-canonicalised.

Type identity is the **corrected** ``typed_tokenizer_v2_corrected`` key, i.e.
a complete invariant of the coloured incidence graph (canonical adjacency +
canonical semantic colour sequence).  No string prefix, no hash distance and
no approximate collision rule is used.

Vocabulary
----------
* support ``n_t`` = number of **distinct training molecules** containing type
  ``t`` (not atom occurrences).
* r1 types are the ancestors of r2 types; the r0 correction is fixed to zero
  so raw atom types are not learnable types.
* the vocabulary is built from official train-10k only.  Ancestor closure is
  enforced.  If the parameter budget (<= 500,000 total inference parameters)
  binds, r2 types are dropped deterministically by ``(-support, key)`` and
  the cost of the mandatory r1 ancestor is charged with them.
* inference uses the deepest retained type: r2 -> r1 -> r0 (zero correction).

Everything in this module is label-free and deterministic.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = _CODE_DIR.parent
REPO = _CODE_DIR.parents[3]
for _p in (_CODE_DIR, str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from tracks.ksvd.code.graph import from_edges  # noqa: E402
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (  # noqa: E402
    TYPED_TOKENIZER_V2_CORRECTED,
    build_colored_incidence,
    corrected_canonical_key,
)

import structural_relation_objects_v1 as sro  # noqa: E402

TOKENIZER_VERSION = TYPED_TOKENIZER_V2_CORRECTED
RADIUS_DEEP = 2
RADIUS_PARENT = 1
ATOM_CATEGORIES = sro.ATOM_CATEGORIES
BASE_P3_PARAMETERS = 413_505
PARAM_CAP = 500_000

#: fixed, pre-registered evaluation-group thresholds (never fitted on valid)
LOW_SUPPORT_THRESHOLD = 10
FALLBACK_HIGH_THRESHOLD = 0.25
LOW_SUPPORT_MAJORITY_FRACTION = 0.5
TAIL_THRESHOLD = -3.0
LARGE_RING_MIN_SIZE = 6
GLOBAL_RING_SIZE_SLOTS = {"ring_size3_frac": 23, "ring_size4_frac": 24, "ring_size5_frac": 25, "ring_size6_frac": 26, "ring_size7_frac": 27}


class ESRDataError(RuntimeError):
    """Hard protocol violation."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ESRDataError(message)


# ---------------------------------------------------------------------------
# raw graph view (structure only)
# ---------------------------------------------------------------------------


def raw_atom_types(obj: sro.MolObjects) -> np.ndarray:
    """Recover the raw atom-type index from the frozen one-hot atom features."""
    block = np.asarray(obj.atom_feat[:, :ATOM_CATEGORIES], dtype=np.float64)
    require(bool(np.all(block >= 0.0)), "atom type block is not non-negative")
    require(bool(np.all(np.abs(block.sum(axis=1) - 1.0) < 1e-6)), "atom type block is not one-hot")
    types = np.argmax(block, axis=1).astype(np.int64)
    return types


def bond_type_map(obj: sro.MolObjects) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for (u, v), bt in zip(np.asarray(obj.bond_index), np.asarray(obj.bond_type)):
        u, v = int(u), int(v)
        out[(u, v) if u < v else (v, u)] = int(bt)
    return out


def graph_of(obj: sro.MolObjects):
    edges = [(int(u), int(v)) for u, v in np.asarray(obj.bond_index)]
    return from_edges(int(obj.n_atoms), edges)


def patch_key(graph, center: int, atom_types: np.ndarray, edge_types: Mapping[tuple[int, int], int], radius: int) -> bytes:
    inc = build_colored_incidence(graph, int(center), atom_types, edge_types, int(radius))
    return corrected_canonical_key(inc)


def molecule_keys(obj: sro.MolObjects) -> dict[str, list[bytes]]:
    """r2 and r1 keys for every atom, in atom-index order."""
    g = graph_of(obj)
    at = raw_atom_types(obj)
    et = bond_type_map(obj)
    n = int(obj.n_atoms)
    r2 = [patch_key(g, c, at, et, RADIUS_DEEP) for c in range(n)]
    r1 = [patch_key(g, c, at, et, RADIUS_PARENT) for c in range(n)]
    return {"r2": r2, "r1": r1}


def all_molecule_keys(objs: Sequence[sro.MolObjects], log_fn=None) -> list[dict[str, list[bytes]]]:
    out: list[dict[str, list[bytes]]] = []
    for i, obj in enumerate(objs):
        out.append(molecule_keys(obj))
        if log_fn is not None and (i + 1) % 1000 == 0:
            log_fn(f"  patch keys {i + 1}/{len(objs)}")
    return out


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------


def support_counts(records: Sequence[Mapping[str, Sequence[bytes]]], field: str) -> Counter:
    counts: Counter = Counter()
    for rec in records:
        for key in set(rec[field]):  # distinct molecules, not occurrences
            counts[key] += 1
    return counts


def _parent_and_root_maps(records: Sequence[Mapping[str, Sequence[bytes]]]) -> tuple[dict[bytes, bytes], dict[bytes, int]]:
    """One pass: r2 key -> r1 parent key and r2/r1 key -> root raw atom type."""
    parent: dict[bytes, bytes] = {}
    root_type: dict[bytes, int] = {}
    for rec in records:
        roots = rec.get("root_atom_type", [])
        for i, (k2, k1) in enumerate(zip(rec["r2"], rec["r1"])):
            parent[k2] = k1
            if i < len(roots):
                root_type.setdefault(k1, int(roots[i]))
                root_type.setdefault(k2, int(roots[i]))
    return parent, root_type


def build_vocab(
    records: Sequence[Mapping[str, Sequence[bytes]]],
    *,
    n_train_molecules: int,
    atom_out: int = 96,
    rank: int = 4,
    base_parameters: int = BASE_P3_PARAMETERS,
    param_cap: int = PARAM_CAP,
) -> dict[str, Any]:
    """Train-only vocabulary with ancestor closure and a hard parameter budget."""
    supp1 = support_counts(records, "r1")
    supp2 = support_counts(records, "r2")
    parent_map, root_type_map = _parent_and_root_maps(records)
    for key in supp2:
        require(key in parent_map, "r2 type has no recorded parent")
        require(parent_map[key] in supp1, "r2 type has no r1 ancestor inside the train data")

    params_per_type = int(rank)  # one beta row; P = atom_out * rank is counted once below
    fixed_cost = int(atom_out) * int(rank)
    editable_budget = int(param_cap) - int(base_parameters) - fixed_cost
    require(editable_budget > 0, "no parameter budget left for residual types")
    max_types = editable_budget // params_per_type

    r1_order = sorted(supp1, key=lambda k: (-supp1[k], k))
    r2_order = sorted(supp2, key=lambda k: (-supp2[k], k))

    kept_r1: list[bytes] = []
    kept_r2: list[bytes] = []
    dropped_r2: list[bytes] = []
    # r1 types cost 1 type each and are ancestors / fallback targets: keep them
    # first in support order; then add r2 with its (already kept) ancestor.
    for key in r1_order:
        if len(kept_r1) + len(kept_r2) + 1 > max_types:
            break
        kept_r1.append(key)
    kept_r1_set = set(kept_r1)
    for key in r2_order:
        if len(kept_r1) + len(kept_r2) + 1 > max_types:
            dropped_r2.append(key)
            continue
        # ancestor closure: the parent must be retained, otherwise the type is
        # not usable (and is charged together with its ancestor).
        if parent_map[key] not in kept_r1_set:
            dropped_r2.append(key)
            continue
        kept_r2.append(key)

    keys = list(kept_r1) + list(kept_r2)
    depth = np.asarray([1] * len(kept_r1) + [2] * len(kept_r2), dtype=np.int32)
    support = np.asarray([supp1[k] for k in kept_r1] + [supp2[k] for k in kept_r2], dtype=np.int32)
    index_of = {k: i for i, k in enumerate(keys)}
    parent = np.full(len(keys), -1, dtype=np.int32)
    for key in kept_r2:
        parent[index_of[key]] = index_of[parent_map[key]]
    raw_type = np.asarray([int(root_type_map.get(k, -1)) for k in keys], dtype=np.int32)

    total_parameters = int(base_parameters) + fixed_cost + params_per_type * len(keys)
    require(total_parameters <= int(param_cap), "vocabulary exceeds the inference parameter cap")
    require(len(kept_r1) + len(kept_r2) <= max_types, "vocabulary exceeds the editable type budget")
    for i, d in enumerate(depth):
        if d == 2:
            require(int(parent[i]) >= 0 and int(depth[int(parent[i])]) == 1, "ancestor closure broken")

    return {
        "keys": keys,
        "depth": depth,
        "support": support,
        "parent": parent,
        "root_atom_type": raw_type,
        "index": index_of,
        "n_r1": len(kept_r1),
        "n_r2": len(kept_r2),
        "n_types": len(keys),
        "dropped_r2": len(dropped_r2),
        "n_types_available": len(kept_r1) + len(kept_r2) + len(dropped_r2),
        "fixed_residual_parameters": fixed_cost,
        "params_per_type": params_per_type,
        "total_inference_parameters": total_parameters,
        "param_cap": int(param_cap),
        "max_types_under_cap": int(max_types),
        "support_histogram": {
            "r1": _support_histogram([supp1[k] for k in kept_r1]),
            "r2": _support_histogram([supp2[k] for k in kept_r2]),
        },
        "support_quantiles_r2": _quantiles([supp2[k] for k in kept_r2]),
        "support_quantiles_r1": _quantiles([supp1[k] for k in kept_r1]),
        "n_train_molecules": int(n_train_molecules),
        "uses_valid_data": False,
        "tokenizer": TOKENIZER_VERSION,
        "radii": {"deep": RADIUS_DEEP, "parent": RADIUS_PARENT},
        "r0_correction": "fixed_zero",
    }


def _support_histogram(values: Sequence[int]) -> dict[str, int]:
    bins = {"1": 0, "2-4": 0, "5-9": 0, "10-49": 0, "50-199": 0, ">=200": 0}
    for v in values:
        v = int(v)
        if v <= 1:
            bins["1"] += 1
        elif v < 5:
            bins["2-4"] += 1
        elif v < 10:
            bins["5-9"] += 1
        elif v < 50:
            bins["10-49"] += 1
        elif v < 200:
            bins["50-199"] += 1
        else:
            bins[">=200"] += 1
    return bins


def _quantiles(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {f"q{int(q * 100)}": float(np.quantile(a, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)}


def attach_root_types(records: list[dict[str, list[bytes]]], objs: Sequence[sro.MolObjects]) -> None:
    for rec, obj in zip(records, objs):
        rec["root_atom_type"] = [int(v) for v in raw_atom_types(obj)]


# ---------------------------------------------------------------------------
# inference-time type assignment
# ---------------------------------------------------------------------------


def assign_types(
    records: Sequence[Mapping[str, Sequence[bytes]]],
    vocab: Mapping[str, Any],
    support_of_train:
    Mapping[bytes, int] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Deepest-retained-type assignment and OOV fallback flags."""
    index = vocab["index"]
    depth = np.asarray(vocab["depth"])
    out: list[dict[str, np.ndarray]] = []
    for rec in records:
        r2_id = np.asarray([index.get(k, -1) for k in rec["r2"]], dtype=np.int32)
        r1_id = np.asarray([index.get(k, -1) for k in rec["r1"]], dtype=np.int32)
        t_star = np.where(r2_id >= 0, r2_id, r1_id).astype(np.int32)
        t_depth = np.where(t_star >= 0, depth[np.clip(t_star, 0, None)], 0).astype(np.int32)
        r2_oov = (r2_id < 0).astype(np.int8)
        zero_correction = (t_star < 0).astype(np.int8)
        out.append(
            {
                "r2_id": r2_id,
                "r1_id": r1_id,
                "t_star": t_star,
                "t_depth": t_depth,
                "r2_oov": r2_oov,
                "zero_correction": zero_correction,
            }
        )
    return out


def type_support_of_assignment(assigned: Sequence[Mapping[str, np.ndarray]], vocab: Mapping[str, Any]) -> list[np.ndarray]:
    support = np.asarray(vocab["support"], dtype=np.int32)
    out = []
    for a in assigned:
        t = a["t_star"]
        out.append(np.where(t >= 0, support[np.clip(t, 0, None)], 0).astype(np.int32))
    return out


# ---------------------------------------------------------------------------
# group definitions (fixed before any valid-label read)
# ---------------------------------------------------------------------------


def molecule_group_labels(
    obj: sro.MolObjects,
    assignment: Mapping[str, np.ndarray],
    support: np.ndarray,
    y: float,
    *,
    size_tercile_cuts: Sequence[float] | None = None,
) -> dict[str, Any]:
    n = int(obj.n_atoms)
    t = assignment["t_star"]
    low = int(np.sum((t >= 0) & (support < LOW_SUPPORT_THRESHOLD)))
    fallback = int(np.sum(assignment["r2_oov"] > 0))
    zero = int(np.sum(assignment["zero_correction"] > 0))
    low_frac = low / n
    fallback_frac = fallback / n
    g = np.asarray(obj.global_feat, dtype=np.float64)
    ring6 = float(g[GLOBAL_RING_SIZE_SLOTS["ring_size6_frac"]])
    ring7 = float(g[GLOBAL_RING_SIZE_SLOTS["ring_size7_frac"]])
    labels = {
        "ordinary": bool(y > TAIL_THRESHOLD),
        "tail": bool(y <= TAIL_THRESHOLD),
        "large_ring": bool((ring6 + ring7) > 0.0),
        "low_support_none": bool(low_frac == 0.0),
        "low_support_partial": bool(0.0 < low_frac < LOW_SUPPORT_MAJORITY_FRACTION),
        "low_support_majority": bool(low_frac >= LOW_SUPPORT_MAJORITY_FRACTION),
        "fallback_none": bool(fallback_frac == 0.0),
        "fallback_partial": bool(0.0 < fallback_frac < FALLBACK_HIGH_THRESHOLD),
        "fallback_high": bool(fallback_frac >= FALLBACK_HIGH_THRESHOLD),
    }
    if size_tercile_cuts is not None:
        c1, c2 = float(size_tercile_cuts[0]), float(size_tercile_cuts[1])
        labels["size_small"] = bool(n <= c1)
        labels["size_medium"] = bool(c1 < n <= c2)
        labels["size_large"] = bool(n > c2)
    labels.update(
        {
            "n_atoms": n,
            "low_support_fraction": low_frac,
            "fallback_fraction": fallback_frac,
            "zero_correction_fraction": zero / n,
            "n_fallback_atoms": fallback,
        }
    )
    return labels


def train_size_terciles(objs: Sequence[sro.MolObjects]) -> tuple[float, float]:
    sizes = np.asarray([int(o.n_atoms) for o in objs], dtype=np.float64)
    return (float(np.quantile(sizes, 1.0 / 3.0)), float(np.quantile(sizes, 2.0 / 3.0)))


# ---------------------------------------------------------------------------
# label-derived frozen scalars
# ---------------------------------------------------------------------------


def sy_from_labels(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    med = float(np.median(y))
    return {
        "sy": float(np.abs(y - med).mean()),
        "median": med,
        "n": int(y.size),
        "definition": "mean absolute deviation of train labels from the train median",
    }


def sh2_for_encoder(model, objs: Sequence[sro.MolObjects], device, *, batch_size: int = 256) -> dict[str, float]:
    """Mean squared norm of the SSL-initialised atom embeddings on train."""
    import torch

    was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(objs), batch_size):
            batch = sro.batch_objects(list(objs[i : i + batch_size]))
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            a = model.atom_encoder(batch["atom_feat"])
            total += float((a * a).sum().item())
            n += int(a.shape[0])
    if was_training:
        model.train()
    require(n > 0, "no atoms in Sh2 computation")
    return {"sh2": total / n, "n_atoms": n, "definition": "mean squared norm of SSL-initialised train atom embeddings"}


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def save_types(path: Path, split_records: Mapping[str, Sequence[Mapping[str, np.ndarray]]]) -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {}
    hashes: dict[str, Any] = {}
    for split, assigned in split_records.items():
        payload[split] = [
            {k: np.ascontiguousarray(v) for k, v in a.items()} for a in assigned
        ]
        hashes[split] = {
            "n_molecules": len(assigned),
            "t_star_sha256": hashlib.sha256(b"".join(a["t_star"].tobytes() for a in assigned)).hexdigest(),
            "n_atoms": int(sum(a["t_star"].size for a in assigned)),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return hashes


def load_types(path: Path) -> dict[str, list[dict[str, np.ndarray]]]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)
