"""Deterministic, label-free feature caches for the explicit-performance sprint.

Two offline caches, both built only from the frozen molecule graph / SMILES and
never from a label:

A2 -- ten RDKit physicochemical descriptors per molecule + a missing flag
      (``physchem_v1``).  Mean/std are fitted on official **Train** rows only.

A3 -- explicit rooted two-hop branch tokens, one per directed bond, hashed with
      a stable BLAKE2b digest into 8192 buckets (``branch_tokens_v1``).  The
      per-root-atom aggregation (sum and mean over that atom's out-edges) is the
      only thing a learned module sees; the graph itself is used exclusively
      here, in deterministic preprocessing.

Both builders record algorithm version, source hashes, row identity, per-row
coverage and (for A3) bucket-collision statistics.  Loaders re-verify the
recorded hash and refuse Test rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import ep_common as C

_MISSING = float("nan")


# ---------------------------------------------------------------------------
# shared source access
# ---------------------------------------------------------------------------


def cache_dir() -> Path:
    return C.PILOT / "cache" / "pcba_v1"


def smiles_path() -> Path:
    return cache_dir() / "smiles.jsonl"


def read_smiles() -> list[str]:
    path = smiles_path()
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if not s:
                continue
            out.append(json.loads(s) if s.startswith('"') else s)
    if len(out) != C.N_MOLECULES:
        raise RuntimeError(f"SMILES artifact has {len(out)} rows, expected {C.N_MOLECULES}")
    return out


def _mmap(name: str) -> np.ndarray:
    return np.load(cache_dir() / f"{name}.npy", mmap_mode="r")


# ---------------------------------------------------------------------------
# A2: RDKit physicochemical descriptors
# ---------------------------------------------------------------------------


def _rdkit_descriptors(smiles: str) -> tuple[list[float], str]:
    """Return the frozen 10 descriptors and an error string ('' on success)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], "MolFromSmiles returned None"
        rings = mol.GetRingInfo().AtomRings()
        max_ring = max((len(r) for r in rings), default=0)
        values = [
            float(rdMolDescriptors.CalcTPSA(mol)),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            float(rdMolDescriptors.CalcNumHBA(mol)),
            float(math.log1p(Descriptors.MolWt(mol))),
            float(Crippen.MolLogP(mol)),
            float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            float(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(max_ring),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
        ]
    except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
        return [], f"{type(exc).__name__}: {exc}"
    if len(values) != C.N_PHYSCHEM:
        return [], f"descriptor count {len(values)} != {C.N_PHYSCHEM}"
    if not all(np.isfinite(v) for v in values):
        return [], "non-finite descriptor value"
    return values, ""


def _physchem_chunk(args: tuple[list[int], list[str]]) -> dict[str, Any]:
    rows, smiles = args
    values = np.full((len(rows), C.N_PHYSCHEM), np.nan, dtype=np.float32)
    failures: list[dict[str, Any]] = []
    for i, (row, smi) in enumerate(zip(rows, smiles)):
        vals, err = _rdkit_descriptors(smi)
        if err:
            failures.append({"row": int(row), "error": err})
        else:
            values[i] = np.asarray(vals, dtype=np.float32)
    return {"rows": list(rows), "values": values, "failures": failures}


def build_physchem_cache(
    task: Any,
    *,
    workers: int = 8,
    out_cache: Path = C.CACHE_A2,
    out_meta: Path = C.CACHE_A2_META,
) -> dict[str, Any]:
    """Build the A2 cache for official Train + Valid rows only."""
    import rdkit

    rows = np.sort(np.concatenate([task.train_rows, task.valid_rows]).astype(np.int64))
    if np.intersect1d(rows, task.test_rows).size:
        raise RuntimeError("A2 build row set includes official Test rows")
    smiles_all = read_smiles()
    smiles = [smiles_all[int(r)] for r in rows]
    raw = np.full((C.N_MOLECULES, C.N_PHYSCHEM), np.nan, dtype=np.float32)
    built = np.zeros(C.N_MOLECULES, dtype=bool)
    failed = np.zeros(C.N_MOLECULES, dtype=bool)
    failures: list[dict[str, Any]] = []

    n = max(1, int(workers))
    idx = np.array_split(np.arange(len(rows), dtype=np.int64), n)
    payloads = [([int(rows[i]) for i in part], [smiles[int(i)] for i in part]) for part in idx if part.size]
    with ProcessPoolExecutor(max_workers=n) as pool:
        for res in pool.map(_physchem_chunk, payloads):
            rr = np.asarray(res["rows"], dtype=np.int64)
            raw[rr] = res["values"]
            ok = np.isfinite(res["values"]).all(axis=1)
            built[rr[ok]] = True
            failed[rr[~ok]] = True
            failures.extend(res["failures"])

    train_ok = np.asarray(task.train_rows)[built[task.train_rows] & ~failed[task.train_rows]]
    if train_ok.size < 0.99 * len(task.train_rows):
        raise RuntimeError(f"too few Train descriptors: {train_ok.size}/{len(task.train_rows)}")
    train_values = raw[train_ok].astype(np.float64)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    constant = np.flatnonzero((std == 0.0) | ~np.isfinite(std))
    scale = std.copy()
    scale[constant] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError("physchem Train-only normalisation is not finite/positive")

    out_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_cache,
        raw=raw,
        built=built,
        failed=failed,
        attempted_rows=rows,
        train_mean=mean.astype(np.float64),
        train_scale=scale.astype(np.float64),
    )
    norm_fp = C.sha256_bytes(
        np.ascontiguousarray(mean, dtype=np.float64).tobytes()
        + np.ascontiguousarray(scale, dtype=np.float64).tobytes()
    )
    meta = {
        "schema": "ep-physchem-cache-v1",
        "algorithm": C.PHYSCHEM_ALGO,
        "names": list(C.PHYSCHEM_NAMES),
        "dim": C.N_PHYSCHEM,
        "rdkit_version": str(rdkit.__version__),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "rows_attempted": int(rows.size),
        "rows_attempted_sha256": C.array_sha(rows, "int64"),
        "train_rows": int(len(task.train_rows)),
        "valid_rows": int(len(task.valid_rows)),
        "test_rows_built": 0,
        "built_rows": int(built.sum()),
        "failed_rows": int(failed.sum()),
        "coverage_attempted": float(built[rows].mean()),
        "coverage_train": float(built[task.train_rows].mean()),
        "coverage_valid": float(built[task.valid_rows].mean()),
        "failures": failures,
        "missing_policy": "attempted-and-failed rows keep NaN in raw and get Train mean + flag 1 at load time",
        "train_only_norm": {
            "n_fit_rows": int(train_ok.size),
            "mean": [float(x) for x in mean],
            "scale": [float(x) for x in scale],
            "constant_indices": [int(x) for x in constant],
            "zero_std_policy": C.PHYSCHEM_ZERO_STD_POLICY,
            "valid_or_test_used": False,
            "fingerprint": norm_fp,
        },
        "finite": bool(np.isfinite(raw[built]).all()),
        "source": {
            "smiles": C.track_rel(smiles_path()),
            "smiles_sha256": C.sha256_file(smiles_path()),
            "feature_spec_hash": C.SOURCE["feature_spec_hash"],
            "split_partition_sha256": C.SOURCE["split_partition_sha256"],
            "label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
            "labels_read": False,
        },
        "files": {"cache": C.track_rel(out_cache), "cache_sha256": C.sha256_file(out_cache)},
        "created_utc": C.utcnow(),
    }
    C.write_json(out_meta, meta)
    return meta


def load_physchem(task: Any, *, cache: Path = C.CACHE_A2, meta_path: Path = C.CACHE_A2_META) -> dict[str, Any]:
    """Load and standardise the A2 input; Test rows are never returned finite."""
    if not cache.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"missing A2 cache {cache} / {meta_path}; run ep_build_features --physchem")
    meta = C.read_json(meta_path)
    if meta.get("schema") != "ep-physchem-cache-v1" or meta.get("algorithm") != C.PHYSCHEM_ALGO:
        raise RuntimeError("A2 cache schema/algorithm drifted")
    if meta["files"]["cache_sha256"] != C.sha256_file(cache):
        raise RuntimeError("A2 cache file hash drifted")
    if meta["source"]["smiles_sha256"] != C.sha256_file(smiles_path()):
        raise RuntimeError("A2 source SMILES hash drifted")
    with np.load(cache) as payload:
        raw = np.asarray(payload["raw"], dtype=np.float32)
        built = np.asarray(payload["built"]).astype(bool)
        failed = np.asarray(payload["failed"]).astype(bool)
        attempted = np.asarray(payload["attempted_rows"], dtype=np.int64)
        mean = np.asarray(payload["train_mean"], dtype=np.float64)
        scale = np.asarray(payload["train_scale"], dtype=np.float64)
    if raw.shape != (C.N_MOLECULES, C.N_PHYSCHEM):
        raise RuntimeError(f"A2 raw shape {raw.shape}")
    if np.isfinite(raw[task.test_rows]).any():
        raise RuntimeError("A2 cache exposes finite values for official Test rows")
    if np.intersect1d(attempted, task.test_rows).size:
        raise RuntimeError("A2 cache attempted official Test rows")
    # Recompute the Train-only normalisation independently and compare.
    train_rows = np.asarray(task.train_rows)
    train_ok = train_rows[built[train_rows] & ~failed[train_rows]]
    recomputed_mean = raw[train_ok].astype(np.float64).mean(axis=0)
    recomputed_std = raw[train_ok].astype(np.float64).std(axis=0)
    recomputed_scale = recomputed_std.copy()
    recomputed_scale[recomputed_scale == 0.0] = 1.0
    if not np.allclose(recomputed_mean, mean, atol=1e-6, rtol=0) or not np.allclose(
        recomputed_scale, scale, atol=1e-6, rtol=0
    ):
        raise RuntimeError("A2 Train-only mean/scale do not reproduce from raw + Train rows")
    matrix = np.full((C.N_MOLECULES, C.PHYSCHEM_DIM), np.nan, dtype=np.float32)
    values = raw.copy()
    values[~built & ~failed] = np.nan
    values[failed] = mean.astype(np.float32)
    matrix[:, :C.N_PHYSCHEM] = ((values - mean) / scale).astype(np.float32)
    matrix[:, C.N_PHYSCHEM] = failed.astype(np.float32)
    matrix[~built & ~failed] = np.nan
    return {
        "matrix": matrix,
        "raw": raw,
        "built": built,
        "failed": failed,
        "attempted_rows": attempted,
        "norm": {"mean": mean, "scale": scale, "fingerprint": meta["train_only_norm"]["fingerprint"]},
        "meta": meta,
        "feature_hash": meta["files"]["cache_sha256"],
        "norm_fingerprint": meta["train_only_norm"]["fingerprint"],
    }


# ---------------------------------------------------------------------------
# A3: explicit rooted two-hop branch tokens
# ---------------------------------------------------------------------------

_ATOM_BASE = 16
_BOND_BASE = 4


def branch_key(bond_type: int, atom_type: int, is_edge: bool) -> int:
    """Injective integer code for (bond_type, atom_type(w), is_edge(v,w)).

    Monotone in the tuple order, so sorting these integers is exactly the
    lexicographic sort required by the protocol.
    """
    if not (1 <= int(bond_type) <= _BOND_BASE):
        raise ValueError(f"bond_type {bond_type} outside 1..{_BOND_BASE}")
    if not (0 <= int(atom_type) < _ATOM_BASE):
        raise ValueError(f"atom_type {atom_type} outside 0..{_ATOM_BASE - 1}")
    return ((int(bond_type) - 1) * _ATOM_BASE + int(atom_type)) * 2 + int(bool(is_edge))


def decode_branch_key(key: int) -> tuple[int, int, bool]:
    edge = int(key) % 2
    rest = int(key) // 2
    atom_type = rest % _ATOM_BASE
    bond_type = rest // _ATOM_BASE + 1
    return bond_type, atom_type, bool(edge)


def signature_bytes(atom_v: int, bond_vu: int, atom_u: int, keys: Sequence[int]) -> bytes:
    """Stable canonical byte string of one directed-bond branch signature."""
    if not (0 <= int(atom_v) < _ATOM_BASE) or not (0 <= int(atom_u) < _ATOM_BASE):
        raise ValueError("atom_type outside 0..15")
    if not (1 <= int(bond_vu) <= _BOND_BASE):
        raise ValueError("bond_type outside 1..4")
    ks = [int(k) for k in keys]
    if any(k < 0 or k > 127 for k in ks):
        raise ValueError("branch key outside 0..127")
    return struct.pack("<3H", int(atom_v), int(bond_vu), int(atom_u)) + struct.pack(
        f"<{len(ks)}H", *ks
    )


def signature_digest(atom_v: int, bond_vu: int, atom_u: int, keys: Sequence[int]) -> tuple[int, int]:
    """Return (64-bit digest, bucket) for one directed-bond branch signature."""
    payload = signature_bytes(atom_v, bond_vu, atom_u, keys)
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value, value % C.N_BRANCH_BUCKETS


def molecule_branch_tokens(
    atom_types: Sequence[int],
    bond_src: Sequence[int],
    bond_dst: Sequence[int],
    bond_types: Sequence[int],
) -> dict[str, Any]:
    """Per-atom out-edge token buckets for one molecule (pure function).

    ``bond_src/bond_dst`` are the frozen cache's **undirected** bonds (exactly
    one row per bond).  Every bond is expanded into both orientations here, so
    the token set is invariant to the stored orientation and each atom receives
    one token per incident bond.  Root atom = ``v``, intermediate = ``u``.
    """
    n = len(atom_types)
    src = [int(x) for x in bond_src]
    dst = [int(x) for x in bond_dst]
    bt = [int(x) for x in bond_types]
    if not (len(src) == len(dst) == len(bt)):
        raise ValueError("bond arrays have different lengths")
    at = [int(x) for x in atom_types]
    if any(x < 0 or x >= _ATOM_BASE for x in at):
        raise ValueError("atom_type outside 0..15")
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k in range(len(src)):
        if not (0 <= src[k] < n and 0 <= dst[k] < n):
            raise ValueError("bond endpoint outside the atom array")
        if src[k] == dst[k]:
            raise ValueError("self-loop in the bond list")
        adj[src[k]].append((dst[k], bt[k]))
        adj[dst[k]].append((src[k], bt[k]))
    per_atom: list[list[int]] = [[] for _ in range(n)]
    digests: list[int] = []
    bucket_of_edge: list[int] = []
    n_zero_branch = 0
    for v in range(n):
        v_nbrs = {w for w, _ in adj[v]}
        for u, b_vu in adj[v]:
            keys = sorted(
                branch_key(bt2, at[w], w in v_nbrs) for w, bt2 in adj[u] if w != v
            )
            if not keys:
                n_zero_branch += 1
            digest, bucket = signature_digest(at[v], b_vu, at[u], keys)
            per_atom[v].append(bucket)
            digests.append(digest)
            bucket_of_edge.append(bucket)
    return {
        "per_atom": per_atom,
        "digests": digests,
        "bucket_of_edge": bucket_of_edge,
        "zero_degree_atoms": [i for i in range(n) if not adj[i]],
        "tokens": sum(len(x) for x in per_atom),
        "empty_branch_lists": int(n_zero_branch),
    }


def _branch_chunk(
    args: tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> dict[str, Any]:
    rows, atom_ptr, bond_ptr, bond_index, atom_type, bond_type = args
    counts = np.zeros(len(rows), dtype=np.int64)
    atom_counts: list[np.ndarray] = []
    buckets: list[np.ndarray] = []
    digests: list[np.ndarray] = []
    n_zero_degree = 0
    for i, row in enumerate(rows):
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        tokens = molecule_branch_tokens(
            np.asarray(atom_type[a0:a1]).tolist(),
            np.asarray(bond_index[b0:b1, 0]).tolist(),
            np.asarray(bond_index[b0:b1, 1]).tolist(),
            np.asarray(bond_type[b0:b1]).tolist(),
        )
        ac = np.fromiter((len(x) for x in tokens["per_atom"]), dtype=np.int32, count=a1 - a0)
        flat = np.fromiter(
            (b for per in tokens["per_atom"] for b in per), dtype=np.int16, count=int(ac.sum())
        )
        counts[i] = int(ac.sum())
        atom_counts.append(ac)
        buckets.append(flat)
        digests.append(np.asarray(tokens["digests"], dtype=np.uint64))
        n_zero_degree += len(tokens["zero_degree_atoms"])
    return {
        "rows": rows,
        "row_counts": counts,
        "atom_counts": atom_counts,
        "buckets": buckets,
        "digests": digests,
        "zero_degree_atoms": int(n_zero_degree),
    }


def build_branch_cache(
    task: Any,
    *,
    workers: int = 8,
    out_cache: Path = C.CACHE_A3,
    out_meta: Path = C.CACHE_A3_META,
) -> dict[str, Any]:
    """Build the A3 branch-token cache for official Train + Valid rows only."""
    rows = np.sort(np.concatenate([task.train_rows, task.valid_rows]).astype(np.int64))
    if np.intersect1d(rows, task.test_rows).size:
        raise RuntimeError("A3 build row set includes official Test rows")
    atom_ptr = np.asarray(_mmap("atom_ptr"))
    bond_ptr = np.asarray(_mmap("bond_ptr"))
    bond_index = np.asarray(_mmap("bond_index"))
    atom_type = np.asarray(_mmap("atom_type"))
    bond_type = np.asarray(_mmap("bond_type"))

    n = max(1, int(workers))
    parts = [part for part in np.array_split(np.arange(len(rows), dtype=np.int64), n) if part.size]
    args = [
        ([int(rows[i]) for i in part], atom_ptr, bond_ptr, bond_index, atom_type, bond_type)
        for part in parts
    ]
    row_counts = np.zeros(len(rows), dtype=np.int64)
    total_row_tokens = np.empty(C.N_MOLECULES + 1, dtype=np.int64)
    total_row_tokens[0] = 0
    atom_counts_all = np.zeros(int(atom_ptr[-1]), dtype=np.int32)
    bucket_parts: list[np.ndarray] = []
    digest_parts: list[np.ndarray] = []
    zero_degree = 0
    cursor = 0
    with ProcessPoolExecutor(max_workers=n) as pool:
        for res in pool.map(_branch_chunk, args):
            rc = np.asarray(res["row_counts"], dtype=np.int64)
            row_counts[cursor : cursor + len(rc)] = rc
            chunk_atoms = np.concatenate(
                [np.arange(int(atom_ptr[r]), int(atom_ptr[r + 1]), dtype=np.int64) for r in res["rows"]]
            )
            atom_counts_all[chunk_atoms] = np.concatenate(res["atom_counts"])
            bucket_parts.append(np.concatenate(res["buckets"]))
            digest_parts.append(np.concatenate(res["digests"]))
            zero_degree += int(res["zero_degree_atoms"])
            cursor += len(rc)
    token_bucket = np.concatenate(bucket_parts).astype(np.int16)

    # Row pointer over *all* molecules: unbuilt (Test) rows have zero tokens.
    cum = 0
    row_map = {int(r): int(c) for r, c in zip(rows.tolist(), row_counts.tolist())}
    for r in range(C.N_MOLECULES):
        total_row_tokens[r] = cum
        cum += row_map.get(r, 0)
    total_row_tokens[C.N_MOLECULES] = cum
    if cum != token_bucket.size:
        raise RuntimeError(f"branch token layout {cum} != {token_bucket.size}")
    if int(atom_counts_all.sum()) != token_bucket.size:
        raise RuntimeError("A3 per-atom token counts do not sum to the token array")
    n_atoms_total = int(atom_ptr[-1])
    n_atoms_built = int(np.sum(atom_ptr[rows + 1] - atom_ptr[rows]))
    expected_zero = n_atoms_total - n_atoms_built + int(zero_degree)
    if int((atom_counts_all == 0).sum()) != expected_zero:
        raise RuntimeError(
            f"A3 zero token counts {(atom_counts_all == 0).sum()} != unbuilt atoms "
            f"{n_atoms_total - n_atoms_built} + zero-degree built atoms {zero_degree}"
        )
    if int((atom_counts_all > 0).sum()) > n_atoms_built:
        raise RuntimeError("A3 atom token count layout is impossible")
    digests = np.concatenate(digest_parts).astype(np.uint64)
    distinct_sig = int(np.unique(digests).size)
    bucket_hist = np.bincount(token_bucket.astype(np.int64), minlength=C.N_BRANCH_BUCKETS)
    used = int((bucket_hist > 0).sum())

    out_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_cache,
        row_token_ptr=total_row_tokens,
        token_bucket=token_bucket,
        atom_token_count=atom_counts_all,
    )
    meta = {
        "schema": "ep-branch-token-cache-v1",
        "algorithm": C.BRANCH_ALGO,
        "symbol": "(atom_type(v), bond_type(v,u), atom_type(u), sorted[(bond_type(u,w), atom_type(w), is_edge(v,w)) for w in N(u), w != v])",
        "hash": "blake2b(digest_size=8, little-endian uint64) over stable uint16 packing",
        "n_buckets": int(C.N_BRANCH_BUCKETS),
        "bucket_rule": "uint64_digest % 8192",
        "embedding_dim": int(C.BRANCH_TOKEN_DIM),
        "rows_attempted": int(rows.size),
        "rows_attempted_sha256": C.array_sha(rows, "int64"),
        "train_rows": int(len(task.train_rows)),
        "valid_rows": int(len(task.valid_rows)),
        "test_rows_built": 0,
        "n_atoms_total": n_atoms_total,
        "n_atoms_built": n_atoms_built,
        "n_tokens": int(token_bucket.size),
        "tokens_per_molecule_mean": float(token_bucket.size / rows.size),
        "max_tokens_per_atom": int(atom_counts_all.max()),
        "n_zero_degree_atoms_built": int(zero_degree),
        "n_atoms_with_tokens": int((atom_counts_all > 0).sum()),
        "collision_stats": {
            "n_tokens": int(token_bucket.size),
            "n_distinct_signatures": distinct_sig,
            "n_buckets": int(C.N_BRANCH_BUCKETS),
            "n_buckets_used": used,
            "n_buckets_empty": int(C.N_BRANCH_BUCKETS - used),
            "max_bucket_count": int(bucket_hist.max()),
            "top1_share": float(bucket_hist.max() / max(1, token_bucket.size)),
            "min_bucket_lower_bound": int(max(0, distinct_sig - used)),
            "note": "exact distinct 64-bit signatures; bucket collisions counted at bucket granularity",
        },
        "source": {
            "atom_type_sha256": C.sha256_file(cache_dir() / "atom_type.npy"),
            "bond_type_sha256": C.sha256_file(cache_dir() / "bond_type.npy"),
            "bond_index_sha256": C.sha256_file(cache_dir() / "bond_index.npy"),
            "atom_ptr_sha256": C.sha256_file(cache_dir() / "atom_ptr.npy"),
            "bond_ptr_sha256": C.sha256_file(cache_dir() / "bond_ptr.npy"),
            "feature_spec_hash": C.SOURCE["feature_spec_hash"],
            "cache_sha256_manifest": C.SOURCE["cache_sha256"],
            "labels_read": False,
        },
        "files": {"cache": C.track_rel(out_cache), "cache_sha256": C.sha256_file(out_cache)},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "created_utc": C.utcnow(),
    }
    C.write_json(out_meta, meta)
    return meta


def load_branch(task: Any, *, cache: Path = C.CACHE_A3, meta_path: Path = C.CACHE_A3_META) -> dict[str, Any]:
    """Load the A3 token cache; Test rows must carry zero tokens."""
    if not cache.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"missing A3 cache {cache} / {meta_path}; run ep_build_features --branch")
    meta = C.read_json(meta_path)
    if meta.get("schema") != "ep-branch-token-cache-v1" or meta.get("algorithm") != C.BRANCH_ALGO:
        raise RuntimeError("A3 cache schema/algorithm drifted")
    if meta["files"]["cache_sha256"] != C.sha256_file(cache):
        raise RuntimeError("A3 cache file hash drifted")
    if int(meta["n_buckets"]) != C.N_BRANCH_BUCKETS:
        raise RuntimeError("A3 bucket count drifted")
    with np.load(cache) as payload:
        row_ptr = np.asarray(payload["row_token_ptr"], dtype=np.int64)
        tokens = np.asarray(payload["token_bucket"], dtype=np.int16)
        atom_counts = np.asarray(payload["atom_token_count"], dtype=np.int32)
    atom_ptr = np.asarray(_mmap("atom_ptr"))
    n_atoms_total = int(atom_ptr[-1])
    if row_ptr.shape != (C.N_MOLECULES + 1,) or int(row_ptr[0]) != 0 or int(row_ptr[-1]) != tokens.size:
        raise RuntimeError("A3 row pointer drift")
    if atom_counts.shape != (n_atoms_total,) or int(atom_counts.sum()) != int(tokens.size):
        raise RuntimeError("A3 atom token count drift")
    if tokens.size and (int(tokens.min()) < 0 or int(tokens.max()) >= C.N_BRANCH_BUCKETS):
        raise RuntimeError("A3 bucket id outside the frozen range")
    if np.any(row_ptr[np.asarray(task.test_rows)] != row_ptr[np.asarray(task.test_rows) + 1]):
        raise RuntimeError("A3 cache contains tokens for official Test rows")
    if int((atom_counts > 0).sum()) != int(meta["n_atoms_with_tokens"]):
        raise RuntimeError("A3 atom coverage differs from the recorded build")
    expected_zero = n_atoms_total - int(meta["n_atoms_built"]) + int(meta["n_zero_degree_atoms_built"])
    if int((atom_counts == 0).sum()) != expected_zero:
        raise RuntimeError(
            f"A3 atom layout drift: zero counts {(atom_counts == 0).sum()} != unbuilt atoms "
            f"{n_atoms_total - int(meta['n_atoms_built'])} + zero-degree built atoms "
            f"{int(meta['n_zero_degree_atoms_built'])}"
        )
    if int(atom_counts.max(initial=0)) != int(meta["max_tokens_per_atom"]):
        raise RuntimeError("A3 max tokens per atom drift")
    return {
        "row_token_ptr": row_ptr,
        "token_bucket": tokens,
        "atom_token_count": atom_counts,
        "meta": meta,
        "feature_hash": meta["files"]["cache_sha256"],
    }


# ---------------------------------------------------------------------------
# builders CLI
# ---------------------------------------------------------------------------


def run_builder(which: str, *, workers: int = 8) -> dict[str, Any]:
    import ep_data as D

    task = D.load_task()
    out: dict[str, Any] = {"which": which, "workers": int(workers), "started_utc": C.utcnow()}
    if which in ("all", "physchem"):
        out["physchem"] = build_physchem_cache(task, workers=int(workers))
    if which in ("all", "branch"):
        out["branch"] = build_branch_cache(task, workers=int(workers))
    out["finished_utc"] = C.utcnow()
    out["test_rows_read"] = 0
    return out


if __name__ == "__main__":  # pragma: no cover
    import ep_build_features as B

    raise SystemExit(B.main())
