"""Fixed endpoint features, rooted-moment cache and batch collate for RMC.

Construction (CPU, float64, train-only statistics)
-------------------------------------------------
1. ``a_SSL`` = frozen SSL atom encoder applied to the *raw* 108-D atom feature
   (the same input the P3/SSL encoder was trained on; no extra normalisation).
   ``d_j = concat(raw_atom108_j, a_SSL_j)`` in R^204.
2. train column mean / ``max(std, 1e-3)`` standardisation of ``d`` (constant
   columns -> scale 1); valid reuses the train constants.
3. two fixed random views ``u_jv = tanh(ds_j @ R_v) / sqrt(12)``,
   ``R_v ~ N(0, 1/204)`` with generators 7301/7302.  Not trained, not searched.
4. for every root/scale: ``W = (I+A)^k``, ``p = W / rowsum``, then ``mu``, packed
   ``C2``, packed ``C3`` and ``log m`` in float64, validated, stored float32.

The cache is *only* a function of the frozen structure objects and the frozen
SSL atom encoder -- no label, no model prediction, no valid/test statistic.
"""

from __future__ import annotations

import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import ad_data as AD  # noqa: E402
import ad_models as M1  # noqa: E402
import ad_train as AT  # noqa: E402

import rmc_common as C  # noqa: E402
import rmc_moments as RM  # noqa: E402

#: fallback when ``rmc_norm.pt`` does not exist yet (local unit checks only)
SENTINEL_NORM = "rmc_norm_missing"


# ---------------------------------------------------------------------------
# frozen endpoint features
# ---------------------------------------------------------------------------


def load_memory_atom_encoder(ssl_path: Path) -> M1.MLP:
    """The frozen atom encoder of the shared train-only SSL checkpoint."""
    import torch

    state = torch.load(ssl_path, map_location="cpu", weights_only=False)
    if "atom_encoder" not in state:
        raise SystemExit(f"{ssl_path} has no atom_encoder")
    enc = M1.MLP(AD.ATOM_FEAT_DIM, 128, 96, 1, 0.0)
    enc.load_state_dict(state["atom_encoder"], strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def memory_encoder_parameters() -> int:
    enc = M1.MLP(AD.ATOM_FEAT_DIM, 128, 96, 1, 0.0)
    return int(sum(p.numel() for p in enc.parameters()))


def atom_ssl_features(atom_feat: np.ndarray, enc_module, *, batch: int = 8192) -> np.ndarray:
    """Run the frozen atom encoder over every atom row (eval mode, no grad).

    Single-threaded on purpose: CPU torch reductions are thread-count dependent
    in the last bit, and the cached endpoint features (hence the moment cache)
    must be bit-reproducible for a fixed torch/numpy build.
    """
    import torch

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    out = []
    try:
        x = torch.from_numpy(np.ascontiguousarray(atom_feat, dtype=np.float32))
        with torch.no_grad():
            for start in range(0, x.shape[0], int(batch)):
                out.append(enc_module(x[start : start + int(batch)]).numpy())
    finally:
        torch.set_num_threads(prev)
    return np.concatenate(out, 0)


def endpoint_matrix(atom_feat: np.ndarray, atom_ssl: np.ndarray) -> np.ndarray:
    d = np.concatenate([np.asarray(atom_feat, dtype=np.float64), np.asarray(atom_ssl, dtype=np.float64)], axis=1)
    if d.shape[1] != C.ENDPOINT_DIM:
        raise ValueError(f"endpoint dim {d.shape[1]} != {C.ENDPOINT_DIM}")
    return d


def projection_matrices() -> np.ndarray:
    """``(2, 204, 12)`` fixed Gaussian projections, generator seeds 7301/7302."""
    out = []
    for seed in C.R_SEEDS:
        g = np.random.default_rng(int(seed))
        out.append(g.normal(0.0, 1.0 / C.ENDPOINT_DIM, size=(C.ENDPOINT_DIM, C.R_PROJ)))
    return np.stack(out, 0)


def standardise_columns(d: np.ndarray, mean: np.ndarray | None = None, scale: np.ndarray | None = None):
    d = np.asarray(d, dtype=np.float64)
    if mean is None:
        mean = d.mean(axis=0)
        std = d.std(axis=0)
        scale = np.where(std > 0.0, np.maximum(std, 1e-3), 1.0)
    else:
        mean = np.asarray(mean, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
    return (d - mean) / scale, mean, scale


def view_features(ds: np.ndarray, R: np.ndarray) -> np.ndarray:
    """``(N, 2, 12)`` tanh-projected endpoints for both fixed views."""
    ds = np.asarray(ds, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 2:
        R = R[None]
    # einsum, not ``@``: keep the fixed views reproducible across BLAS thread counts
    proj = np.stack([np.einsum("nd,dr->nr", ds, R[v], optimize=False)
                     for v in range(R.shape[0])], axis=1)  # (N, V, 12)
    return np.ascontiguousarray(np.tanh(proj) / math.sqrt(C.R_PROJ))


# ---------------------------------------------------------------------------
# per-molecule moments
# ---------------------------------------------------------------------------


def molecule_bonds(cache: Mapping[str, Any], mol: int) -> np.ndarray:
    b0, b1 = int(cache["bond_ptr"][mol]), int(cache["bond_ptr"][mol + 1])
    return np.asarray(cache["bond_index"][b0:b1], dtype=np.int64)


def molecule_moments(
    u_views: np.ndarray, bonds: np.ndarray, *, scales: Sequence[int] = C.SCALES
) -> dict[str, np.ndarray]:
    """All rooted moments of one molecule for both views.

    ``u_views``: ``(n, 2, 12)``.  Returns arrays with shape
    ``(n, 8, 2, 12)`` (mu), ``(n, 8, 2, 78)`` (c2), ``(n, 8, 2, 364)`` (c3),
    ``(n, 8)`` (log m), plus the uniform-over-atoms moments ``(2,12/78/364)``
    used only by the optional U control.
    """
    n = int(u_views.shape[0])
    powers = RM.walk_powers(n, bonds, scales)
    mu = np.zeros((n, len(scales), C.N_VIEW, C.PROJ_DIM), dtype=np.float64)
    c2 = np.zeros((n, len(scales), C.N_VIEW, C.N_PAIR), dtype=np.float64)
    c3 = np.zeros((n, len(scales), C.N_VIEW, C.N_TRIPLE), dtype=np.float64)
    logm = np.zeros((n, len(scales)), dtype=np.float64)
    for si, W in enumerate(powers):
        for v in range(C.N_VIEW):
            m_u, m_c2, m_c3, m_logm = RM.moments_from_walk(u_views[:, v, :], W)
            mu[:, si, v] = m_u
            c2[:, si, v] = m_c2
            c3[:, si, v] = m_c3
            if v == 0:
                logm[:, si] = m_logm
    # k = 0 is the point mass at the root: exact zeros (mathematically exact)
    zero_scale = [i for i, s in enumerate(scales) if int(s) == 0]
    for si in zero_scale:
        c2[:, si, :, :] = 0.0
        c3[:, si, :, :] = 0.0
    uni_mu = np.zeros((C.N_VIEW, C.PROJ_DIM), dtype=np.float64)
    uni_c2 = np.zeros((C.N_VIEW, C.N_PAIR), dtype=np.float64)
    uni_c3 = np.zeros((C.N_VIEW, C.N_TRIPLE), dtype=np.float64)
    p = np.full(n, 1.0 / n, dtype=np.float64)
    for v in range(C.N_VIEW):
        a, b, c = RM.moments_direct(u_views[:, v, :], p)
        uni_mu[v], uni_c2[v], uni_c3[v] = a, b, c
    return {"mu": mu, "c2": c2, "c3": c3, "logm": logm,
            "uniform_mu": uni_mu, "uniform_c2": uni_c2, "uniform_c3": uni_c3}


# ---------------------------------------------------------------------------
# chunked parallel build
# ---------------------------------------------------------------------------

_BUILD: dict[str, Any] = {}


def _worker(rng: tuple[int, int]) -> dict[str, int | np.ndarray]:
    start, end = int(rng[0]), int(rng[1])
    u_views = _BUILD["u"]
    atom_ptr = _BUILD["atom_ptr"]
    bond_ptr = _BUILD["bond_ptr"]
    bond_index = _BUILD["bond_index"]
    n_scale, n_view = C.N_SCALE, C.N_VIEW
    counts = np.diff(atom_ptr[start : end + 1])
    n_atoms = int(counts.sum())
    mu = np.zeros((n_atoms, n_scale, n_view, C.PROJ_DIM), dtype=np.float32)
    c2 = np.zeros((n_atoms, n_scale, n_view, C.N_PAIR), dtype=np.float32)
    c3 = np.zeros((n_atoms, n_scale, n_view, C.N_TRIPLE), dtype=np.float32)
    logm = np.zeros((n_atoms, n_scale), dtype=np.float32)
    u_mu = np.zeros((end - start, n_view, C.PROJ_DIM), dtype=np.float32)
    u_c2 = np.zeros((end - start, n_view, C.N_PAIR), dtype=np.float32)
    u_c3 = np.zeros((end - start, n_view, C.N_TRIPLE), dtype=np.float32)
    off = 0
    for mi in range(start, end):
        a0, a1 = int(atom_ptr[mi]), int(atom_ptr[mi + 1])
        b0, b1 = int(bond_ptr[mi]), int(bond_ptr[mi + 1])
        out = molecule_moments(u_views[a0:a1], bond_index[b0:b1])
        n = a1 - a0
        mu[off : off + n] = out["mu"].astype(np.float32)
        c2[off : off + n] = out["c2"].astype(np.float32)
        c3[off : off + n] = out["c3"].astype(np.float32)
        logm[off : off + n] = out["logm"].astype(np.float32)
        u_mu[mi - start] = out["uniform_mu"].astype(np.float32)
        u_c2[mi - start] = out["uniform_c2"].astype(np.float32)
        u_c3[mi - start] = out["uniform_c3"].astype(np.float32)
        off += n
    return {"start": start, "end": end, "n_atoms": n_atoms, "mu": mu, "c2": c2, "c3": c3,
            "logm": logm, "uniform_mu": u_mu, "uniform_c2": u_c2, "uniform_c3": u_c3}


def build_split_moments(
    split: str,
    u: np.ndarray,
    *,
    workers: int = 8,
    chunk: int = 128,
    limit: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Build ``mu / C2 / C3 / log m`` for one split into preallocated arrays."""
    cache = AD.load_cache(split)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    n_mol = int(cache["n_molecules"])
    if limit is not None:
        n_mol = min(int(limit), n_mol)
        atom_ptr = atom_ptr[: n_mol + 1]
    n_atoms = int(atom_ptr[-1])
    log(f"[prep {split}] molecules={n_mol} atoms={n_atoms}")

    mu = np.zeros((n_atoms, C.N_SCALE, C.N_VIEW, C.PROJ_DIM), dtype=np.float32)
    c2 = np.zeros((n_atoms, C.N_SCALE, C.N_VIEW, C.N_PAIR), dtype=np.float32)
    c3 = np.zeros((n_atoms, C.N_SCALE, C.N_VIEW, C.N_TRIPLE), dtype=np.float32)
    logm = np.zeros((n_atoms, C.N_SCALE), dtype=np.float32)
    u_mu = np.zeros((n_mol, C.N_VIEW, C.PROJ_DIM), dtype=np.float32)
    u_c2 = np.zeros((n_mol, C.N_VIEW, C.N_PAIR), dtype=np.float32)
    u_c3 = np.zeros((n_mol, C.N_VIEW, C.N_TRIPLE), dtype=np.float32)

    _BUILD.clear()
    _BUILD.update(
        {
            "u": np.ascontiguousarray(u[:n_atoms], dtype=np.float64),
            "atom_ptr": atom_ptr,
            "bond_ptr": np.asarray(cache["bond_ptr"], dtype=np.int64),
            "bond_index": np.asarray(cache["bond_index"], dtype=np.int64),
        }
    )
    ranges = [(s, min(s + int(chunk), n_mol)) for s in range(0, n_mol, int(chunk))]
    t0 = time.time()
    ctx = get_context("fork") if int(workers) > 1 else None
    if ctx is None:
        results = map(_worker, ranges)
        pool = None
    else:
        pool = ctx.Pool(int(workers))
        results = pool.imap(_worker, ranges)
    done = 0
    for res in results:
        s, e = int(res["start"]), int(res["end"])
        a0, a1 = int(atom_ptr[s]), int(atom_ptr[e])
        mu[a0:a1] = res["mu"]  # type: ignore[arg-type]
        c2[a0:a1] = res["c2"]  # type: ignore[arg-type]
        c3[a0:a1] = res["c3"]  # type: ignore[arg-type]
        logm[a0:a1] = res["logm"]  # type: ignore[arg-type]
        u_mu[s:e] = res["uniform_mu"]  # type: ignore[arg-type]
        u_c2[s:e] = res["uniform_c2"]  # type: ignore[arg-type]
        u_c3[s:e] = res["uniform_c3"]  # type: ignore[arg-type]
        done += e - s
        if done % (20 * int(chunk)) == 0 or done == n_mol:
            log(f"[prep {split}] {done}/{n_mol} molecules  {time.time() - t0:.1f}s")
    if pool is not None:
        pool.close()
        pool.join()
    elapsed = time.time() - t0
    _BUILD.clear()
    return {
        "split": split,
        "mu": mu,
        "c2": c2,
        "c3": c3,
        "logm": logm,
        "uniform_mu": u_mu,
        "uniform_c2": u_c2,
        "uniform_c3": u_c3,
        "atom_ptr": atom_ptr,
        "n_atoms": n_atoms,
        "n_molecules": n_mol,
        "build_seconds_cpu": float(elapsed),
        "workers": int(workers),
        "chunk": int(chunk),
        "bytes_moments": int(mu.nbytes + c2.nbytes + c3.nbytes + logm.nbytes),
    }


def save_moments(payload: Mapping[str, Any], path: Path) -> str:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "mu": torch.from_numpy(np.ascontiguousarray(payload["mu"])),
        "c2": torch.from_numpy(np.ascontiguousarray(payload["c2"])),
        "c3": torch.from_numpy(np.ascontiguousarray(payload["c3"])),
        "logm": torch.from_numpy(np.ascontiguousarray(payload["logm"])),
        "uniform_mu": torch.from_numpy(np.ascontiguousarray(payload["uniform_mu"])),
        "uniform_c2": torch.from_numpy(np.ascontiguousarray(payload["uniform_c2"])),
        "uniform_c3": torch.from_numpy(np.ascontiguousarray(payload["uniform_c3"])),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(tensors), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def load_moments(path: Path, *, mmap: bool = True) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", mmap=mmap, weights_only=True)


def save_static(split: str, cache: Mapping[str, Any], n_mol: int | None = None) -> str:
    import torch

    n = int(cache["n_molecules"]) if n_mol is None else int(n_mol)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)[: n + 1]
    n_atoms = int(atom_ptr[-1])
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)[: n + 1]
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    deg = np.zeros(n_atoms, dtype=np.int64)
    # `bond_index` / `junc_index` are molecule-local in the frozen P3 cache;
    # lift them to split-global atom ids before counting.
    b_end = int(bond_ptr[-1])
    b_mol = np.repeat(np.arange(n), np.diff(bond_ptr))
    b_off = atom_ptr[b_mol]
    np.add.at(deg, bond_index[:b_end, 0] + b_off, 1)
    np.add.at(deg, bond_index[:b_end, 1] + b_off, 1)
    junc_ptr = np.asarray(cache["junc_ptr"], dtype=np.int64)[: n + 1]
    junc_index = np.asarray(cache["junc_index"], dtype=np.int64)
    njunc = np.zeros(n_atoms, dtype=np.int64)
    j_end = int(junc_ptr[-1])
    j_mol = np.repeat(np.arange(n), np.diff(junc_ptr))
    j_off = atom_ptr[j_mol]
    np.add.at(njunc, junc_index[:j_end, 1] + j_off, 1)
    n_atoms_graph = np.asarray(cache["n_atoms_graph"], dtype=np.int64)[:n]
    n_bonds_graph = np.asarray(cache["n_bonds_graph"], dtype=np.int64)[:n]
    payload = {
        "split": split,
        "atom_ptr": torch.from_numpy(atom_ptr.astype(np.int64)),
        "deg": torch.from_numpy(deg),
        "njunc_root": torch.from_numpy(njunc),
        "n_atoms_graph": torch.from_numpy(n_atoms_graph),
        "n_bonds_graph": torch.from_numpy(n_bonds_graph),
    }
    path = C.static_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os as _os

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    _os.replace(tmp, path)
    return C.sha256_file(path)


def load_static(split: str) -> dict[str, Any]:
    import torch

    path = C.static_path(split)
    if not path.exists():
        raise SystemExit(f"missing static cache {path}; run prep")
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# batch collate
# ---------------------------------------------------------------------------


def make_collate(statics: Mapping[str, Mapping[str, Any]], *, arm: str):
    """P3 batch (unchanged) plus atom-level cache indices for M/G/U.

    P keeps the exact nr_v1 P3 data path (no extra keys) so that the same-round
    control is the unchanged ``P3NullModel`` input pipeline.
    """
    import torch

    def collate(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
        batch = AT.collate(cache, rows, "P3N", valid=valid)
        if arm == "P":
            return batch
        split = str(cache["meta"].get("split"))
        if split not in statics:
            raise ValueError(f"no static cache for split {split!r}")
        st = statics[split]
        rows_arr = np.asarray(rows, dtype=np.int64)
        atom_ptr = np.asarray(st["atom_ptr"], dtype=np.int64)
        starts = atom_ptr[rows_arr]
        counts = atom_ptr[rows_arr + 1] - starts
        idx = np.concatenate([np.arange(s, s + c, dtype=np.int64) for s, c in zip(starts, counts)])
        atom_batch = np.repeat(np.arange(len(rows_arr), dtype=np.int64), counts)
        mol_idx = np.repeat(rows_arr, counts)
        batch.update(
            {
                "rmc_split": split,
                "rmc_moment_index": torch.from_numpy(idx),
                "rmc_mol_index": torch.from_numpy(mol_idx),
                "rmc_deg": torch.from_numpy(np.asarray(st["deg"], dtype=np.float32)[idx]),
                "rmc_njunc": torch.from_numpy(np.asarray(st["njunc_root"], dtype=np.float32)[idx]),
                "rmc_natoms": torch.from_numpy(
                    np.asarray(st["n_atoms_graph"], dtype=np.float32)[mol_idx]
                ),
                "rmc_nbonds": torch.from_numpy(
                    np.asarray(st["n_bonds_graph"], dtype=np.float32)[mol_idx]
                ),
            }
        )
        return batch

    return collate
