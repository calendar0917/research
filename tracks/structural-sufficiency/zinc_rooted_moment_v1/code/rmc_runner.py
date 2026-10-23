"""Runner for ``zinc_rooted_moment_v1`` (RMC).

Commands
--------
prep     frozen endpoint features + train-only statistics + rooted-moment cache
checks   CPU implementation / invariance / gradient gates (no training)
timing   GPU wall-clock measurement only (no valid score is read or used)
train    one formal arm/seed through the unchanged recipe-B trainer
score    per-run valid metrics, fixed groups, per-molecule table
analyze  paired contrasts, persistence endpoints, frozen decision, U trigger
probe    frozen-checkpoint C3 / q=0 diagnostics (no training)
test     terminal official-test read (only if the frozen trigger fires)

The official ZINC test split is never loaded by prep/checks/timing/train/score/
analyze/probe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import ad_common as AC  # noqa: E402
import ad_data as AD  # noqa: E402
import ad_train as AT  # noqa: E402

import rmc_common as C  # noqa: E402
import rmc_data as D  # noqa: E402
import rmc_models as M  # noqa: E402
import rmc_moments as RM  # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def load_ys() -> tuple[np.ndarray, np.ndarray]:
    train = AD.load_cache("train")
    valid = AD.load_cache("valid")
    return (
        np.asarray(train["labels"], dtype=np.float64),
        np.asarray(valid["labels"], dtype=np.float64),
    )


def _maxdiff(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _state_sha(path: Path) -> str:
    return C.sha256_state_dict(M.state_from_checkpoint(path))


def order_sequence_hash(seed: int, *, epochs: int = 600, n_rows: int = 10000) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(int(seed) + 101)
    h = __import__("hashlib").sha256()
    per_epoch = []
    for _ in range(int(epochs)):
        perm = torch.randperm(int(n_rows), generator=gen).numpy().astype(np.int64)
        h.update(perm.tobytes())
        per_epoch.append(__import__("hashlib").sha256(perm.tobytes()).hexdigest())
    return {
        "generator_seed": int(seed) + 101,
        "epochs": int(epochs),
        "rows": int(n_rows),
        "stream_sha256": h.hexdigest(),
        "first_epoch_sha256": per_epoch[0] if per_epoch else "",
        "last_epoch_sha256": per_epoch[-1] if per_epoch else "",
    }


def make_builder(arm: str, norm_arr: Mapping[str, Any], u_c3: bool | None = None):
    """Builder handed to the unchanged trainer.

    ``ad_train.train`` always passes the P3 ``norm.json`` dict.  P uses that
    dict verbatim (it only reads global33 constants); M/G/U need this round's
    frozen RMC normaliser, so the trainer-supplied dict is ignored for them.
    """

    def build(candidate, norm_p3, seed, device):
        chosen = norm_p3 if arm == "P" else norm_arr
        return M.build_arm(arm, chosen, seed, device, u_c3=u_c3)

    return build


def eval_valid(model, cache, rows, arm: str, collate, device, batch_size=256) -> np.ndarray:
    pred, _ = AT.evaluate(model, cache, rows, candidate=arm, device=device,
                          batch_size=int(batch_size), collate_fn=collate)
    return pred


def _statics_both() -> dict[str, dict[str, Any]]:
    return {s: D.load_static(s) for s in ("train", "valid")}


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def cmd_prep(args: argparse.Namespace) -> int:
    cfg = C.load_config()
    t_start = time.time()
    ssl_path = C.REPO / cfg["ssl"]["path"]
    if C.sha256_file(ssl_path) != cfg["ssl"]["sha256"]:
        raise SystemExit("SSL checkpoint hash mismatch")
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    limit = None if args.limit is None else int(args.limit)
    n_valid_limit = None if limit is None else max(1, min(limit // 8, int(limit)))

    norm_p3 = AD.load_norm()
    enc = D.load_memory_atom_encoder(ssl_path)
    mem_params = int(sum(p.numel() for p in enc.parameters()))
    R = D.projection_matrices()

    t0 = time.time()
    train_cache = AD.load_cache("train")
    valid_cache = AD.load_cache("valid")
    n_tr = int(train_cache["n_molecules"]) if limit is None else min(limit, int(train_cache["n_molecules"]))
    n_va = int(valid_cache["n_molecules"]) if n_valid_limit is None else min(
        n_valid_limit, int(valid_cache["n_molecules"])
    )
    atom_end = int(np.asarray(train_cache["atom_ptr"])[n_tr])
    a_ssl_tr = D.atom_ssl_features(np.asarray(train_cache["atom_feat"])[:atom_end], enc)
    d_tr = D.endpoint_matrix(np.asarray(train_cache["atom_feat"])[:atom_end], a_ssl_tr)
    ds_tr, d_mean, d_scale = D.standardise_columns(d_tr)
    u_tr = D.view_features(ds_tr, R)
    del a_ssl_tr, d_tr, ds_tr

    vatom_end = int(np.asarray(valid_cache["atom_ptr"])[n_va])
    a_ssl_va = D.atom_ssl_features(np.asarray(valid_cache["atom_feat"])[:vatom_end], enc)
    d_va = D.endpoint_matrix(np.asarray(valid_cache["atom_feat"])[:vatom_end], a_ssl_va)
    ds_va, _, _ = D.standardise_columns(d_va, d_mean, d_scale)
    u_va = D.view_features(ds_va, R)
    del a_ssl_va, d_va, ds_va
    encode_seconds = time.time() - t0

    train_m = D.build_split_moments(
        "train", u_tr, workers=int(args.workers), chunk=int(args.chunk), limit=n_tr, log=print
    )
    # train-only statistics, from the train roots only
    mu = train_m["mu"]
    mu_sum = mu.sum(axis=0, dtype=np.float64)
    mu_sq = np.square(mu, dtype=np.float64).sum(axis=0)
    mu_mean = mu_sum / train_m["n_atoms"]
    mu_var = mu_sq / train_m["n_atoms"] - np.square(mu_mean)
    mu_scale = np.maximum(np.sqrt(np.maximum(mu_var, 0.0)), 0.02)
    logm = train_m["logm"]
    logm_mean = logm.mean(axis=0, dtype=np.float64)
    logm_scale = np.maximum(logm.std(axis=0, dtype=np.float64), 1e-3)
    del mu, mu_sq, logm

    train_static_sha = D.save_static("train", train_cache, n_tr)
    valid_static_sha = D.save_static("valid", valid_cache, n_va)
    train_static = D.load_static("train")
    log_deg = np.log1p(np.asarray(train_static["deg"], dtype=np.float64))
    deg_mean = float(log_deg.mean())
    deg_scale = float(max(log_deg.std(), 1e-3))
    train_mean_n_atoms = float(np.asarray(train_static["n_atoms_graph"], dtype=np.float64).mean())

    train_sha = D.save_moments(train_m, C.moments_path("train"))
    train_bytes = int(train_m["bytes_moments"])
    train_seconds = train_m["build_seconds_cpu"]
    del train_m

    valid_m = D.build_split_moments(
        "valid", u_va, workers=int(args.workers), chunk=int(args.chunk), limit=n_va, log=print
    )
    valid_sha = D.save_moments(valid_m, C.moments_path("valid"))
    valid_bytes = int(valid_m["bytes_moments"])
    valid_seconds = valid_m["build_seconds_cpu"]
    del valid_m

    norm = {
        "protocol_id": C.PROTOCOL_ID,
        "source": "train-only statistics; no y, no valid/test statistic",
        "d_mean": d_mean.astype(np.float32).tolist(),
        "d_scale": d_scale.astype(np.float32).tolist(),
        "R": R.astype(np.float32).tolist(),
        "R_seeds": list(C.R_SEEDS),
        "R_sha256": C.sha256_array(R.astype(np.float32)),
        "mu_mean": mu_mean.astype(np.float32).tolist(),
        "mu_scale": mu_scale.astype(np.float32).tolist(),
        "mu_scale_floor": 0.02,
        "logm_mean": logm_mean.astype(np.float32).tolist(),
        "logm_scale": logm_scale.astype(np.float32).tolist(),
        "deg_mean": deg_mean,
        "deg_scale": deg_scale,
        "global_mean": np.asarray(norm_p3["global_mean"], dtype=np.float32).tolist(),
        "global_scale": np.asarray(norm_p3["global_scale"], dtype=np.float32).tolist(),
        "global_source": norm_p3.get("source"),
        "train_mean_n_atoms": train_mean_n_atoms,
        "eps": C.EPS,
        "scales": list(C.SCALES),
        "views": list(C.VIEWS),
        "proj_dim": C.PROJ_DIM,
        "memory_atom_encoder_parameters": mem_params,
        "train_rows": int(n_tr),
        "valid_rows": int(n_va),
        "limit": limit,
        "d_mean_sha256": C.sha256_array(np.asarray(d_mean, dtype=np.float32)),
        "mu_mean_sha256": C.sha256_array(mu_mean.astype(np.float32)),
    }
    norm_file = C.norm_path()
    C.write_json(norm_file, norm)
    norm_sha = C.sha256_file(norm_file)

    prep = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "ssl_sha256": C.sha256_file(ssl_path),
        "normalisation_sha256": norm_sha,
        "limit": limit,
        "workers": int(args.workers),
        "chunk": int(args.chunk),
        "train": {
            "rows": int(n_tr),
            "atoms": int(train_cache["atom_ptr"][n_tr]),
            "moments_sha256": train_sha,
            "static_sha256": train_static_sha,
            "bytes": train_bytes,
            "build_seconds_cpu": train_seconds,
        },
        "valid": {
            "rows": int(n_va),
            "atoms": int(valid_cache["atom_ptr"][n_va]),
            "moments_sha256": valid_sha,
            "static_sha256": valid_static_sha,
            "bytes": valid_bytes,
            "build_seconds_cpu": valid_seconds,
        },
        "encode_seconds_cpu": encode_seconds,
        "wall_seconds": time.time() - t_start,
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "cache_layout": {
            "mu": [int(train_cache["atom_ptr"][n_tr]), C.N_SCALE, C.N_VIEW, C.PROJ_DIM],
            "c2": [int(train_cache["atom_ptr"][n_tr]), C.N_SCALE, C.N_VIEW, C.N_PAIR],
            "c3": [int(train_cache["atom_ptr"][n_tr]), C.N_SCALE, C.N_VIEW, C.N_TRIPLE],
            "logm": [int(train_cache["atom_ptr"][n_tr]), C.N_SCALE],
            "uniform": ["n_molecules", C.N_VIEW, "12 / 78 / 364"],
        },
        "float_storage": "float32 (built and validated in float64)",
        "official_test_loaded": False,
    }
    C.write_json(C.ANALYSIS_DIR / ("prep.json" if limit is None else "prep_small.json"), prep)
    print(json.dumps({k: prep[k] for k in ("train", "valid", "peak_rss_mb", "wall_seconds")}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def _torch_direct_moments(u: torch.Tensor, p: torch.Tensor):
    """Independent torch central-moment computation (full tensors, packed at the end)."""
    pi = torch.as_tensor(RM.pair_arrays()[0])
    pj = torch.as_tensor(RM.pair_arrays()[1])
    ti, tj, tk = [torch.as_tensor(x) for x in RM.triple_arrays()]
    mu = (p[:, None] * u).sum(0)
    d = u - mu
    c2 = torch.einsum("n,na,nb->ab", p, d, d)
    c3 = torch.einsum("n,na,nb,nc->abc", p, d, d, d)
    return mu, c2[pi, pj], c3[ti, tj, tk]


def _walk_counts_bruteforce(n: int, bonds: np.ndarray, k: int) -> np.ndarray:
    adj = [set([i]) for i in range(n)]
    for u, v in np.asarray(bonds, dtype=np.int64):
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    out = np.zeros((n, n), dtype=np.int64)

    def rec(start: int, cur: int, left: int):
        if left == 0:
            out[start, cur] += 1
            return
        for nxt in adj[cur]:
            rec(start, nxt, left - 1)

    for s in range(n):
        rec(s, s, k)
    return out


def _stack_moments(u: torch.Tensor, prob: torch.Tensor, n_scale: int):
    """``(n, S, 12/78/364)`` independent central moments, one distribution per (root, scale)."""
    n = u.shape[0]
    mu = torch.zeros(n, n_scale, C.PROJ_DIM, dtype=u.dtype)
    c2 = torch.zeros(n, n_scale, C.N_PAIR, dtype=u.dtype)
    c3 = torch.zeros(n, n_scale, C.N_TRIPLE, dtype=u.dtype)
    for s in range(n_scale):
        for i in range(n):
            m, q2, q3 = _torch_direct_moments(u, prob[i, s])
            mu[i, s], c2[i, s], c3[i, s] = m, q2, q3
    return mu, c2, c3


def _checks_synthetic() -> dict[str, Any]:
    """Implementation-level truth gates.  Convention: mu (n,S,12), q (n,H,12)."""
    gates: dict[str, Any] = {}
    torch.manual_seed(20261109)

    # --- witness -------------------------------------------------------
    w = RM.witness()
    a = w["distributions"]["A"]
    b = w["distributions"]["B"]
    gates["witness"] = {
        "pass": bool(
            abs(a["F_moment_with_C3"] - 0.3846153846) < 1e-9
            and abs(b["F_moment_with_C3"] - 0.4526064213) < 1e-9
            and abs(a["F_gaussian_closure"] - 0.3846153846) < 1e-9
            and abs(b["F_gaussian_closure"] - 0.3846153846) < 1e-9
            and abs(a["third_central_moment"]) < 1e-15
            and abs(b["third_central_moment"] - 0.08838834765) < 1e-10
        ),
        "data": w,
    }

    # --- identity: readout vs direct weighted sum (float64) ------------
    errs = {}
    for case, (n, S) in {"small": (5, 3), "wide": (17, 8)}.items():
        u = torch.randn(n, C.PROJ_DIM, dtype=torch.float64)
        W = torch.rand(n, S, n, dtype=torch.float64) + 0.05
        prob = W / W.sum(-1, keepdim=True)
        mu, c2, c3 = _stack_moments(u, prob, S)
        q = torch.randn(n, 2, C.PROJ_DIM, dtype=torch.float64)
        F, Z = RM.readout_batched(mu, c2, c3, q, use_c3=True)
        worst_f, worst_z = 0.0, 0.0
        for i in range(n):
            for s in range(S):
                for h in range(2):
                    Fd, Zd = RM.readout_direct(u, torch.log(prob[i, s]), q[i, h].unsqueeze(0))
                    worst_f = max(worst_f, float((F[i, s, h] - Fd[0]).abs().max()))
                    worst_z = max(worst_z, float(abs(Z[i, s, h] - Zd[0])))
        errs[case] = {"max_abs_F": worst_f, "max_abs_Z": worst_z}
    gates["identity_readout_vs_direct_sum"] = {
        "pass": bool(all(v["max_abs_F"] < 1e-10 and v["max_abs_Z"] < 1e-10 for v in errs.values())),
        "data": errs,
        "tol": 1e-10,
    }

    # --- identity: q = 0 -> F = mu, Z = 1 + eps ------------------------
    n, S = 9, 3
    u = torch.randn(n, C.PROJ_DIM, dtype=torch.float64)
    W = torch.rand(n, S, n, dtype=torch.float64) + 0.05
    prob = W / W.sum(-1, keepdim=True)
    mu, c2, c3 = _stack_moments(u, prob, S)
    q0 = torch.zeros(n, 1, C.PROJ_DIM, dtype=torch.float64)
    F, Z = RM.readout_batched(mu, c2, c3, q0, use_c3=True)
    gates["identity_q_zero"] = {
        "pass": bool(
            float((F - mu.unsqueeze(2)).abs().max()) < 1e-12
            and float((Z - (1.0 + C.EPS)).abs().max()) < 1e-12
        ),
        "max_abs_F_minus_mu": float((F - mu.unsqueeze(2)).abs().max()),
        "max_abs_Z_minus_1peps": float((Z - (1.0 + C.EPS)).abs().max()),
    }

    # --- point mass (W = I): F = u exactly, Z = kappa ---------------
    n = 6
    u = torch.randn(n, C.PROJ_DIM, dtype=torch.float64)
    mu = u.unsqueeze(1)
    c2 = torch.zeros(n, 1, C.N_PAIR, dtype=torch.float64)
    c3 = torch.zeros(n, 1, C.N_TRIPLE, dtype=torch.float64)
    q = torch.randn(n, 1, C.PROJ_DIM, dtype=torch.float64)
    F, Z = RM.readout_batched(mu, c2, c3, q, use_c3=True)
    kappa = (1.0 + (q[:, 0] * u).sum(-1)) ** 2 + C.EPS
    gates["point_mass"] = {
        "pass": bool(float((F[:, 0, 0] - u).abs().max()) < 1e-12
                     and float((Z[:, 0, 0] - kappa).abs().max()) < 1e-12),
        "max_abs_F_minus_u": float((F[:, 0, 0] - u).abs().max()),
    }

    # --- single atom, all scales --------------------------------------
    u1 = torch.randn(1, C.PROJ_DIM, dtype=torch.float64)
    mu1 = u1.repeat(C.N_SCALE, 1).unsqueeze(0)
    c21 = torch.zeros(1, C.N_SCALE, C.N_PAIR, dtype=torch.float64)
    c31 = torch.zeros(1, C.N_SCALE, C.N_TRIPLE, dtype=torch.float64)
    q1 = torch.randn(1, 2, C.PROJ_DIM, dtype=torch.float64)
    F, Z = RM.readout_batched(mu1, c21, c31, q1)
    kappa1 = (1.0 + (q1[0] @ u1[0])) ** 2 + C.EPS
    gates["single_atom_all_scales"] = {
        "pass": bool(
            float((F[0, :, :, :] - u1[0]).abs().max()) < 1e-12
            and float((Z[0, :, :] - kappa1.unsqueeze(0)).abs().max()) < 1e-12
        ),
        "note": "every scale collapses to the root itself; Z identical across scales",
    }

    # --- packed vs full contraction -----------------------------------
    n = 11
    u = torch.randn(n, C.PROJ_DIM, dtype=torch.float64)
    p = torch.rand(n, dtype=torch.float64) + 0.05
    p = p / p.sum()
    mu, c2p, c3p = _torch_direct_moments(u, p)
    d = u - mu
    c2_full = torch.einsum("n,na,nb->ab", p, d, d)
    c3_full = torch.einsum("n,na,nb,nc->abc", p, d, d, d)
    q = torch.randn(1, 1, C.PROJ_DIM, dtype=torch.float64)
    Fp, Zp = RM.readout_batched(mu.unsqueeze(0).unsqueeze(0), c2p.unsqueeze(0).unsqueeze(0),
                                c3p.unsqueeze(0).unsqueeze(0), q)
    a = 1.0 + (q[0, 0] @ mu)
    Zf = a * a + q[0, 0] @ c2_full @ q[0, 0] + C.EPS
    c3qq = torch.einsum("abc,b,c->a", c3_full, q[0, 0], q[0, 0])
    Ff = mu + (2.0 * a * (c2_full @ q[0, 0]) + c3qq) / Zf
    gates["packed_vs_full"] = {
        "pass": bool(float((Fp[0, 0, 0] - Ff).abs().max()) < 1e-11
                     and float(abs(Zp[0, 0, 0] - Zf)) < 1e-11),
        "max_abs_F": float((Fp[0, 0, 0] - Ff).abs().max()),
        "max_abs_Z": float(abs(Zp[0, 0, 0] - Zf)),
    }

    # --- C2 positive semidefinite -------------------------------------
    n = 13
    u = torch.randn(n, C.PROJ_DIM, dtype=torch.float64)
    p = torch.rand(n, dtype=torch.float64) + 0.01
    p = p / p.sum()
    mu, _, _ = _torch_direct_moments(u, p)
    d = u - mu
    c2_full = torch.einsum("n,na,nb->ab", p, d, d)
    gates["c2_psd"] = {
        "pass": bool(float(torch.linalg.eigvalsh(c2_full).min()) > -1e-12),
        "min_eigenvalue": float(torch.linalg.eigvalsh(c2_full).min()),
    }

    # --- k = 0 exact zeros --------------------------------------------
    rng = np.random.default_rng(11)
    bonds = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    uu = rng.normal(size=(4, C.PROJ_DIM))
    out = D.molecule_moments(np.stack([uu, uu], 1), bonds)
    z = C.SCALES.index(0)
    gates["k0_exact_zero"] = {
        "pass": bool(np.all(out["c2"][:, z] == 0.0) and np.all(out["c3"][:, z] == 0.0)
                     and np.allclose(out["logm"][:, z], 0.0, atol=1e-15)),
        "max_abs_c2": float(np.abs(out["c2"][:, z]).max()),
        "max_abs_c3": float(np.abs(out["c3"][:, z]).max()),
        "max_abs_mu_minus_u": float(np.abs(out["mu"][:, z] - np.stack([uu, uu], 1)).max()),
    }

    # --- walk powers vs brute-force walk enumeration -------------------
    worst = 0.0
    for seed in range(4):
        rg = np.random.default_rng(seed)
        n = 5
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        keep = rg.random(len(pairs)) < 0.5
        bd = np.asarray([pp for pp, kk in zip(pairs, keep) if kk], dtype=np.int64)
        powers = RM.walk_powers(n, bd, (0, 1, 2, 3, 4))
        for k, W in zip((0, 1, 2, 3, 4), powers):
            worst = max(worst, float(np.abs(W - _walk_counts_bruteforce(n, bd, k)).max()))
    gates["walk_powers_bruteforce"] = {"pass": bool(worst == 0.0), "max_abs_diff": worst}

    # --- Z >= eps for large queries on *valid* moments -----------------
    n = 6
    u = torch.randn(n, C.PROJ_DIM, dtype=torch.float64)
    W = torch.rand(n, 1, n, dtype=torch.float64) + 0.05
    prob = W / W.sum(-1, keepdim=True)
    mu, c2p, c3p = _stack_moments(u, prob, 1)
    q = 50.0 * torch.randn(n, 1, C.PROJ_DIM, dtype=torch.float64)
    F, Z = RM.readout_batched(mu, c2p, c3p, q)
    Fd, Zd = RM.readout_direct(u, torch.log(prob[0, 0]), q[0, 0].unsqueeze(0))
    gates["z_floor_large_q"] = {
        "pass": bool(float(Z.min()) >= C.EPS - 1e-9 and bool(torch.isfinite(F).all())
                     and abs(float(Z[0, 0, 0]) - float(Zd[0])) < 1e-9),
        "min_Z": float(Z.min()),
        "eps": C.EPS,
        "direct_vs_moment_Z_at_large_q": abs(float(Z[0, 0, 0]) - float(Zd[0])),
        "note": "Z >= eps uses a^2 >= 0 and C2 PSD; no clamp is applied anywhere",
    }

    # --- query gradient: moment path vs direct sum ---------------------
    n = 7
    u = torch.randn(n, C.PROJ_DIM, dtype=torch.float64)
    p = torch.rand(n, dtype=torch.float64) + 0.05
    p = p / p.sum()
    mu, c2p, c3p = _torch_direct_moments(u, p)
    q = torch.randn(1, 1, C.PROJ_DIM, dtype=torch.float64, requires_grad=True)
    wt = torch.randn(C.PROJ_DIM, dtype=torch.float64)
    F, Z = RM.readout_batched(mu.unsqueeze(0).unsqueeze(0), c2p.unsqueeze(0).unsqueeze(0),
                              c3p.unsqueeze(0).unsqueeze(0), q)
    (F[0, 0, 0] * wt).sum().backward()
    g_moment = q.grad.clone()
    q2 = q.detach().clone().requires_grad_(True)
    Fd, Zd = RM.readout_direct(u, torch.log(p), q2[0, 0].unsqueeze(0))
    (Fd[0] * wt).sum().backward()
    gates["query_gradient"] = {
        "pass": bool(float((g_moment - q2.grad).abs().max()) < 1e-10),
        "max_abs_grad_diff": float((g_moment - q2.grad).abs().max()),
    }
    return gates


def _norm_for_model() -> dict[str, Any]:
    """Load ``rmc_norm.pt`` and convert the list-valued entries to float32 arrays."""
    norm = C.read_json(C.norm_path())
    out: dict[str, Any] = {}
    for k, v in norm.items():
        if isinstance(v, list) and k in ("R", "d_mean", "d_scale", "mu_mean", "mu_scale",
                                         "logm_mean", "logm_scale", "global_mean", "global_scale"):
            out[k] = np.asarray(v, dtype=np.float32)
        else:
            out[k] = v
    out["deg_mean"] = float(norm["deg_mean"])
    out["deg_scale"] = float(norm["deg_scale"])
    out["train_mean_n_atoms"] = float(norm["train_mean_n_atoms"])
    return out


def _checks_data(n_rows: int, workers: int) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    cfg = C.load_config()
    norm = C.read_json(C.norm_path())
    norm_arr = _norm_for_model()
    statics = _statics_both()
    cache_train = AD.load_cache("train")
    rng = np.random.default_rng(7)
    rows = np.arange(int(n_rows), dtype=np.int64)

    # --- parameter counts ---------------------------------------------
    p_model, p_info = M.build_arm("P", norm_arr, 0, "cpu")
    m_model, m_info = M.build_arm("M", norm_arr, 0, "cpu")
    g_model, g_info = M.build_arm("G", norm_arr, 0, "cpu")
    u_model, u_info = M.build_arm("U", norm_arr, 0, "cpu", u_c3=True)
    gates["parameter_counts"] = {
        "pass": bool(
            p_info["parameters"] == 413505
            and m_info["parameters"] == g_info["parameters"] == u_info["parameters"] == 415250
            and m_info["parameters_deployment"] == g_info["parameters_deployment"] == 441586
            and m_info["parameters_deployment"] < 500000
        ),
        "P": p_info["parameters"],
        "M": m_info["parameters"],
        "G": g_info["parameters"],
        "U_with_full_C3": u_info["parameters"],
        "memory_encoder": m_info["parameters_memory_encoder"],
        "deployment_M": m_info["parameters_deployment"],
        "breakdown_M": m_info["parameter_breakdown"],
        "cap": 500000,
    }

    # --- paired init ---------------------------------------------------
    enc_p = C.sha256_state_dict(p_model.encoders.state_dict())
    enc_m = C.sha256_state_dict(m_model.encoders.state_dict())
    enc_g = C.sha256_state_dict(g_model.encoders.state_dict())
    gates["paired_init"] = {
        "pass": bool(enc_p == enc_m == enc_g
                     and C.sha256_state_dict(m_model.state_dict()) == C.sha256_state_dict(g_model.state_dict())),
        "encoders_sha_P": enc_p,
        "encoders_sha_M": enc_m,
        "encoders_sha_G": enc_g,
        "state_sha_M": C.sha256_state_dict(m_model.state_dict()),
        "state_sha_G": C.sha256_state_dict(g_model.state_dict()),
        "flags": {"M_use_c3": m_model.use_c3, "G_use_c3": g_model.use_c3,
                  "M_global": m_model.global_measure, "G_global": g_model.global_measure,
                  "U_global": u_model.global_measure},
    }

    # --- one real batch through M / G / U ------------------------------
    collate_m = D.make_collate(statics, arm="M")
    batch = AT.collate(cache_train, rows, "P3N", valid=False)
    batch.update(collate_m(cache_train, rows, "M", valid=False))
    batch = AT._to_device(batch, "cpu")
    for mod in (p_model, m_model, g_model, u_model):
        mod.eval()
    with torch.no_grad():
        pred_p = p_model(batch)
        pred_m = m_model(batch)
        pred_g = g_model(batch)
        pred_u = u_model(batch)
        m_model.use_c3 = False
        pred_m0 = m_model(batch)
        m_model.use_c3 = True
    gates["m_vs_g_flag"] = {
        "pass": bool(float((pred_m0 - pred_g).abs().max()) == 0.0
                     and bool(torch.isfinite(pred_m).all()) and bool(torch.isfinite(pred_g).all())
                     and bool(torch.isfinite(pred_u).all())),
        "max_abs_M_forced_C3_zero_minus_G": float((pred_m0 - pred_g).abs().max()),
        "max_abs_M_minus_G_at_init": float((pred_m - pred_g).abs().max()),
        "max_abs_U_minus_M_at_init": float((pred_u - pred_m).abs().max()),
    }

    # --- encoder replay vs P ------------------------------------------
    with torch.no_grad():
        a_p, b_p, w_p = p_model.encoders(batch)
        a_m, b_m, w_m = m_model.encoders(batch)
    gates["encoder_replay"] = {
        "pass": bool(_maxdiff(a_p.numpy(), a_m.numpy()) == 0.0
                     and _maxdiff(b_p.numpy(), b_m.numpy()) == 0.0
                     and _maxdiff(w_p.numpy(), w_m.numpy()) == 0.0),
        "a": _maxdiff(a_p.numpy(), a_m.numpy()),
        "b": _maxdiff(b_p.numpy(), b_m.numpy()),
        "w": _maxdiff(w_p.numpy(), w_m.numpy()),
    }

    # --- empty membership sets are exactly zero ------------------------
    with torch.no_grad():
        assert isinstance(a_m, torch.Tensor)
        n_atoms = int(a_m.shape[0])
        bi = batch["bond_index"]
        deg = batch["rmc_deg"].to(a_m.dtype)
        sum_b = a_m.new_zeros((n_atoms, b_m.shape[1]))
        sum_b.index_add_(0, torch.cat([bi[:, 0], bi[:, 1]]), torch.cat([b_m, b_m], dim=0))
        mean_b = sum_b / deg.clamp_min(1.0).unsqueeze(1)
        ji = batch["junc_index"]
        nj = batch["rmc_njunc"].to(a_m.dtype)
        sum_w = a_m.new_zeros((n_atoms, w_m.shape[1]))
        sum_w.index_add_(0, ji[:, 1], w_m)
        mean_w = sum_w / nj.clamp_min(1.0).unsqueeze(1)
    zero_w = nj == 0
    zero_b = deg == 0
    gates["empty_membership_zero"] = {
        "pass": bool(
            (not bool(zero_w.any()) or float(mean_w[zero_w].abs().max()) == 0.0)
            and (not bool(zero_b.any()) or float(mean_b[zero_b].abs().max()) == 0.0)
            and float(mean_w[~zero_w].abs().max()) > 0.0
        ),
        "n_atoms_without_root_junction": int(zero_w.sum()),
        "n_isolated_atoms": int(zero_b.sum()),
        "max_abs_mean_w_where_empty": float(mean_w[zero_w].abs().max()) if bool(zero_w.any()) else 0.0,
        "max_abs_mean_b_where_empty": float(mean_b[zero_b].abs().max()) if bool(zero_b.any()) else 0.0,
        "n_molecules_without_any_junction_in_split": int(
            (np.diff(np.asarray(cache_train["junc_ptr"])) == 0).sum()
        ),
    }

    # --- memory encoder reproduces the cached k=0 endpoint -------------
    ssl_path = C.REPO / cfg["ssl"]["path"]
    enc = D.load_memory_atom_encoder(ssl_path)
    atom_ptr = np.asarray(cache_train["atom_ptr"])
    a_end = int(atom_ptr[int(n_rows)])
    a_ssl = D.atom_ssl_features(np.asarray(cache_train["atom_feat"])[:a_end], enc)
    d = D.endpoint_matrix(np.asarray(cache_train["atom_feat"])[:a_end], a_ssl)
    ds, _, _ = D.standardise_columns(d, np.asarray(norm_arr["d_mean"], dtype=np.float64),
                                     np.asarray(norm_arr["d_scale"], dtype=np.float64))
    R = np.asarray(norm["R"], dtype=np.float32).reshape(C.N_VIEW, C.ENDPOINT_DIM, C.R_PROJ)
    u = D.view_features(ds, R)
    z = C.SCALES.index(0)
    mu0 = np.asarray(mom_u_rows("train", 0, a_end), dtype=np.float64)
    gates["memory_encoder_cache"] = {
        "pass": bool(_maxdiff(mu0, u) < 2e-6),
        "max_abs_diff": _maxdiff(mu0, u),
        "float32_storage_tol": 2e-6,
        "parameters": int(sum(p.numel() for p in enc.parameters())),
        "ssl_sha256": C.sha256_file(ssl_path),
    }

    # --- q = 0 on real cached moments ---------------------------------
    idx = batch["rmc_moment_index"]
    cache = M.split_cache("train", "cpu")
    mu = cache["mu"][idx]
    q0 = torch.zeros(mu.shape[0], 1, C.PROJ_DIM)
    F0, Z0 = RM.readout_batched(mu[:, :, 0, :], cache["c2"][idx][:, :, 0, :],
                                cache["c3"][idx][:, :, 0, :], q0)
    F1, Z1 = RM.readout_batched(mu[:, :, 1, :], cache["c2"][idx][:, :, 1, :],
                                cache["c3"][idx][:, :, 1, :], q0)
    gates["q_zero_real_cache"] = {
        "pass": bool(
            float((F0[:, :, 0, :] - mu[:, :, 0, :]).abs().max()) < 1e-6
            and float((F1[:, :, 0, :] - mu[:, :, 1, :]).abs().max()) < 1e-6
            and float((torch.log(Z0[:, :, 0]) - math.log(1.0 + C.EPS)).abs().max()) < 1e-6
        ),
        "max_abs_F_minus_mu_view0": float((F0[:, :, 0, :] - mu[:, :, 0, :]).abs().max()),
        "max_abs_F_minus_mu_view1": float((F1[:, :, 0, :] - mu[:, :, 1, :]).abs().max()),
        "max_abs_logZ_minus_log1peps": float((torch.log(Z0[:, :, 0]) - math.log(1.0 + C.EPS)).abs().max()),
    }

    # --- cached C2 PSD and Z floor on real moments --------------------
    c2_full = cache["c2"][idx][..., RM.c2_full_index()].reshape(-1, C.PROJ_DIM, C.PROJ_DIM)
    eig_min = float(torch.linalg.eigvalsh(c2_full.double()).min())
    with torch.no_grad():
        e_sel = m_model.e_mlp(torch.cat([a_m, mean_b, mean_w,
                                         torch.log1p(deg).unsqueeze(1),
                                         torch.log1p(nj).unsqueeze(1)], 1))
        q_sel = 4.0 * torch.tanh(m_model.q_proj(e_sel)).reshape(-1, C.N_HEAD, C.PROJ_DIM) / math.sqrt(C.PROJ_DIM)
        F_sel, Z_sel = RM.readout_batched(mu[:, :, 0, :], cache["c2"][idx][:, :, 0, :],
                                          cache["c3"][idx][:, :, 0, :], q_sel[:, 0:2])
    gates["cache_c2_psd_and_z_floor"] = {
        "pass": bool(eig_min > -1e-6 and float(Z_sel.min()) >= C.EPS - 1e-5),
        "min_eigenvalue_float64": eig_min,
        "min_Z_with_model_query": float(Z_sel.min()),
        "eps": C.EPS,
        "query_norm_max": float(q_sel.norm(dim=-1).max()),
    }

    # --- batch / single / order / cross-molecule isolation -------------
    single = []
    with torch.no_grad():
        for r in rows:
            rr = np.asarray([r], dtype=np.int64)
            b1 = AT.collate(cache_train, rr, "P3N", valid=False)
            b1.update(collate_m(cache_train, rr, "M", valid=False))
            single.append(float(m_model(AT._to_device(b1, "cpu"))[0]))
    rev_rows = rows[::-1].copy()
    b_rev = AT.collate(cache_train, rev_rows, "P3N", valid=False)
    b_rev.update(collate_m(cache_train, rev_rows, "M", valid=False))
    with torch.no_grad():
        pred_rev = m_model(AT._to_device(b_rev, "cpu")).numpy()
    gates["batch_vs_single"] = {
        "pass": bool(_maxdiff(pred_m.numpy(), np.asarray(single)) < 2e-5
                     and _maxdiff(pred_rev[::-1], pred_m.numpy()) < 2e-5),
        "max_abs_batch_minus_single": _maxdiff(pred_m.numpy(), np.asarray(single)),
        "max_abs_reversed_batch_order": _maxdiff(pred_rev[::-1], pred_m.numpy()),
        "note": "same molecules, different padding/peer composition -> same prediction",
    }

    # --- bond direction / atom renumbering of the moment cache ---------
    mol = int(rows[0])
    b0, b1 = int(cache_train["bond_ptr"][mol]), int(cache_train["bond_ptr"][mol + 1])
    a0, a1 = int(cache_train["atom_ptr"][mol]), int(cache_train["atom_ptr"][mol + 1])
    bonds = np.asarray(cache_train["bond_index"][b0:b1], dtype=np.int64)
    u_atoms = mom_u_rows("train", a0, a1)
    base = D.molecule_moments(u_atoms, bonds)
    flipped = D.molecule_moments(u_atoms, bonds[:, ::-1])
    perm = rng.permutation(a1 - a0)
    inv = np.argsort(perm)
    perm_mom = D.molecule_moments(u_atoms[perm], inv[bonds])
    gates["renumbering_and_bond_direction"] = {
        "pass": bool(
            _maxdiff(base["mu"], flipped["mu"]) < 1e-12
            and _maxdiff(base["c3"], flipped["c3"]) < 1e-12
            and _maxdiff(base["mu"][perm], perm_mom["mu"]) < 1e-10
            and _maxdiff(base["c2"][perm], perm_mom["c2"]) < 1e-10
            and _maxdiff(base["c3"][perm], perm_mom["c3"]) < 1e-9
        ),
        "bond_direction_c3_max": _maxdiff(base["c3"], flipped["c3"]),
        "renumbering_mu_max": _maxdiff(base["mu"][perm], perm_mom["mu"]),
        "renumbering_c2_max": _maxdiff(base["c2"][perm], perm_mom["c2"]),
        "renumbering_c3_max": _maxdiff(base["c3"][perm], perm_mom["c3"]),
        "molecule": mol,
    }

    # --- gradients ------------------------------------------------------
    m_model.train()
    opt = torch.optim.AdamW(m_model.parameters(), lr=1e-4)
    y = torch.as_tensor(np.asarray(cache_train["labels"], dtype=np.float32)[rows])
    loss = torch.nn.functional.l1_loss(m_model(batch), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    groups = {
        "encoders": list(m_model.encoders.parameters()),
        "e_mlp": list(m_model.e_mlp.parameters()),
        "q_proj": list(m_model.q_proj.parameters()),
        "h_mlp": list(m_model.h_mlp.parameters()),
        "t_mlp": list(m_model.t_mlp.parameters()),
        "graph_bias": [m_model.graph_bias],
    }
    gstat = {}
    for name, params in groups.items():
        g = [p.grad for p in params if p.grad is not None]
        gstat[name] = {
            "tensors": len(g),
            "nonzero": int(sum(int(torch.count_nonzero(x).item() > 0) for x in g)),
            "finite": bool(all(bool(torch.isfinite(x).all()) for x in g)),
            "grad_norm": float(sum(float(x.norm()) ** 2 for x in g) ** 0.5),
        }
    # graph_bias can cancel under a symmetric L1 sign pattern; probe it with
    # a monotone sum instead of claiming a zero gradient means "dead".
    opt.zero_grad(set_to_none=True)
    m_model(batch).sum().backward()
    bias_grad = float(m_model.graph_bias.grad)
    gstat["graph_bias"]["nonzero_sum_loss"] = int(bias_grad != 0.0)
    gstat["graph_bias"]["grad_under_sum_loss"] = bias_grad
    opt_cover = int(sum(p.numel() for grp in opt.param_groups for p in grp["params"]))
    required = {k: v for k, v in gstat.items() if k != "graph_bias"}
    gates["gradients"] = {
        "pass": bool(all(v["tensors"] > 0 and v["nonzero"] > 0 and v["finite"] for v in required.values())
                     and gstat["graph_bias"]["finite"] and bias_grad != 0.0
                     and opt_cover == m_info["parameters"]),
        "groups": gstat,
        "loss_finite": bool(torch.isfinite(loss)),
        "optimizer_parameters": opt_cover,
        "model_parameters": m_info["parameters"],
        "optimizer_covers_model": bool(opt_cover == m_info["parameters"]),
        "memory_encoder_in_model": False,
    }

    # --- frozen statistics ----------------------------------------------
    meta = {
        "norm_sha256": C.sha256_file(C.norm_path()),
        "train_moments_sha256": C.sha256_file(C.moments_path("train")),
        "valid_moments_sha256": C.sha256_file(C.moments_path("valid")),
        "train_static_sha256": C.sha256_file(C.static_path("train")),
        "valid_static_sha256": C.sha256_file(C.static_path("valid")),
        "train_mean_n_atoms_recomputed": float(
            np.asarray(statics["train"]["n_atoms_graph"], dtype=np.float64).mean()
        ),
        "train_mean_n_atoms_norm": float(norm["train_mean_n_atoms"]),
        "mu_scale_min": float(np.asarray(norm["mu_scale"]).min()),
        "logm_scale_min": float(np.asarray(norm["logm_scale"]).min()),
        "deg_scale": float(norm["deg_scale"]),
        "scales": norm["scales"],
        "views": norm["views"],
        "eps": norm["eps"],
        "R_sha256": norm["R_sha256"],
        "R_coefficients": int(C.N_VIEW * C.ENDPOINT_DIM * C.R_PROJ),
    }
    gates["frozen_statistics"] = {
        "pass": bool(
            abs(meta["train_mean_n_atoms_recomputed"] - meta["train_mean_n_atoms_norm"]) < 1e-6
            and meta["mu_scale_min"] >= 0.02 - 1e-9
            and meta["logm_scale_min"] >= 1e-3 - 1e-12
            and meta["deg_scale"] >= 1e-3 - 1e-12
        ),
        "data": meta,
    }
    return gates


def mom_u_rows(split: str, a0: int, a1: int) -> np.ndarray:
    """Rebuild the k=0 endpoint features (mu at scale 0) for one molecule."""
    mom = D.load_moments(C.moments_path(split), mmap=True)
    z = C.SCALES.index(0)
    out = np.asarray(mom["mu"][a0:a1, z], dtype=np.float64)
    del mom
    return out


def cmd_checks(args: argparse.Namespace) -> int:
    t0 = time.time()
    gates = _checks_synthetic()
    small_tmp = None
    if args.limit is not None:
        small_tmp = tempfile.TemporaryDirectory(prefix="rmc_checks_")
        old = C.CACHE_DIR
        C.CACHE_DIR = Path(small_tmp.name)
        print(f"[checks] building a {args.limit}-molecule cache in {C.CACHE_DIR}")
        try:
            prep_args = argparse.Namespace(limit=int(args.limit), workers=int(args.workers), chunk=64)
            cmd_prep(prep_args)
            gates.update(_checks_data(min(8, int(args.limit)), int(args.workers)))
        finally:
            C.CACHE_DIR = old
    else:
        if args.rebuild:
            for path in (C.norm_path(), C.moments_path("train"), C.moments_path("valid"),
                         C.static_path("train"), C.static_path("valid")):
                Path(path).unlink(missing_ok=True)
        if not (C.norm_path().exists() and C.moments_path("train").exists()
                and C.moments_path("valid").exists()):
            print("[checks] full cache missing -> building it (CPU, float64)")
            cmd_prep(argparse.Namespace(limit=None, workers=int(args.workers), chunk=128))
        gates.update(_checks_data(8, int(args.workers)))
    all_pass = bool(all(bool(v.get("pass")) for v in gates.values()))
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "normalisation_sha256": C.sha256_file(C.norm_path()) if C.norm_path().exists() else None,
        "limit": None if args.limit is None else int(args.limit),
        "all_pass": all_pass,
        "n_gates": len(gates),
        "gates": gates,
        "wall_seconds": time.time() - t0,
        "gate_semantics": "implementation / invariance / gradient truth; NOT evidence of performance",
        "official_test_loaded": False,
    }
    out = C.ANALYSIS_DIR / ("cpu_checks.json" if args.limit is None else "cpu_checks_small.json")
    C.write_json(out, payload)
    if small_tmp is not None:
        small_tmp.cleanup()
    print(json.dumps({"all_pass": all_pass, "n_gates": len(gates),
                      "failed": [k for k, v in gates.items() if not v.get("pass")],
                      "wall_seconds": payload["wall_seconds"]}, indent=2))
    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# smoke (tiny end-to-end slice through the *real* trainer)
# ---------------------------------------------------------------------------

_SLICE_KEYS_1D = (
    "atom_ptr", "bond_ptr", "junc_ptr", "dist_ptr", "twohop_ptr",
    "block_ptr", "block_pair_ptr", "block_atom_ptr", "block_bond_ptr",
)
_SLICE_KEYS_ATOM = ("atom_feat", "atom_type")
_SLICE_KEYS_BOND = ("bond_feat", "bond_type")
_SLICE_KEYS_JUNC = ("junc_feat",)


def _sliced_cache(cache: Mapping[str, Any], n_mol: int) -> dict[str, Any]:
    """First ``n_mol`` molecules of a frozen P3 cache, for smoke runs only."""
    out = dict(cache)
    atom_ptr = np.asarray(cache["atom_ptr"])[: n_mol + 1]
    bond_ptr = np.asarray(cache["bond_ptr"])[: n_mol + 1]
    junc_ptr = np.asarray(cache["junc_ptr"])[: n_mol + 1]
    out["atom_ptr"], out["bond_ptr"], out["junc_ptr"] = atom_ptr, bond_ptr, junc_ptr
    n_atoms, n_bonds, n_junc = int(atom_ptr[-1]), int(bond_ptr[-1]), int(junc_ptr[-1])
    for k in _SLICE_KEYS_ATOM:
        if k in cache:
            out[k] = np.asarray(cache[k])[:n_atoms]
    for k in _SLICE_KEYS_BOND:
        if k in cache:
            out[k] = np.asarray(cache[k])[:n_bonds]
    for k in _SLICE_KEYS_JUNC:
        if k in cache:
            out[k] = np.asarray(cache[k])[:n_junc]
    for k in ("bond_index", "junc_index", "junc_bond"):
        if k in cache:
            out[k] = np.asarray(cache[k])[: n_bonds if k != "junc_index" else n_junc]
    out["junc_bond"] = np.asarray(cache["junc_bond"])[:n_junc]
    out["labels"] = np.asarray(cache["labels"])[:n_mol]
    out["global_feat"] = np.asarray(cache["global_feat"])[:n_mol]
    out["n_atoms_graph"] = np.asarray(cache["n_atoms_graph"])[:n_mol]
    out["n_bonds_graph"] = np.asarray(cache["n_bonds_graph"])[:n_mol]
    out["n_molecules"] = int(n_mol)
    out["meta"] = dict(cache.get("meta", {}), n_molecules=int(n_mol), smoke_slice=True)
    return out


def cmd_smoke(args: argparse.Namespace) -> int:
    """One tiny slice through prep + the unchanged trainer, CPU by default."""
    t0 = time.time()
    n_train = int(args.n_train)
    # prep builds ``max(1, limit // 8)`` valid molecules; keep the smoke slice aligned
    n_valid = max(1, n_train // 8)
    tmp = tempfile.TemporaryDirectory(prefix="rmc_smoke_")
    old_dir = C.CACHE_DIR
    C.CACHE_DIR = Path(tmp.name)
    orig_load = AD.load_cache

    def load_cache(split: str):
        cache = orig_load(split)
        if split == "train":
            return _sliced_cache(cache, n_train)
        if split == "valid":
            return _sliced_cache(cache, n_valid)
        raise RuntimeError("smoke runs never touch the test split")

    out: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "kind": "smoke", "runs": {}}
    try:
        cmd_prep(argparse.Namespace(limit=n_train, workers=max(1, args.workers), chunk=8))
        AD.load_cache = load_cache
        statics = _statics_both()
        norm = _norm_for_model()
        device = torch.device("cpu")
        for arm in args.arms:
            run = Path(tmp.name) / f"smoke_{arm}"
            collate = D.make_collate(statics, arm=arm)
            builder = make_builder(arm, norm)
            info = AT.train(arm, 0, device, epochs=int(args.epochs), recipe_variant="B",
                            model_builder=builder, out_dir=run, collate_fn=collate, force=True)
            out["runs"][arm] = {
                "parameters": info["parameter_count"],
                "best_valid": info["best_valid"],
                "final": info["fixed_objects"]["final"],
                "curve_epochs": len(info["curve"]),
                "updates": info["recipe"]["actual_global_steps"],
                "gradient_audit": info["gradient_audit"],
                "parameter_audit": info["parameter_audit"],
                "files": sorted(p.name for p in run.iterdir()),
            }
        out["pass"] = all(v["gradient_audit"].get("all_grad_finite") and
                          v["parameter_audit"].get("parameters_in_optimizer")
                          for v in out["runs"].values())
    finally:
        AD.load_cache = orig_load
        C.CACHE_DIR = old_dir
        tmp.cleanup()
    out["wall_seconds"] = time.time() - t0
    out["official_test_loaded"] = False
    C.write_json(C.ANALYSIS_DIR / "smoke.json", out)
    print(json.dumps(out, indent=2)[:3000])
    return 0 if out.get("pass") else 1


# ---------------------------------------------------------------------------
# timing (GPU wall clock only)
# ---------------------------------------------------------------------------


def cmd_timing(args: argparse.Namespace) -> int:
    cfg = C.load_config()
    device = torch.device(args.device)
    norm = C.read_json(C.norm_path())
    norm_arr = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else v) for k, v in norm.items()}
    statics = _statics_both()
    cache_train = AD.load_cache("train")
    y_train = np.asarray(cache_train["labels"], dtype=np.float64)
    rows = np.arange(len(y_train), dtype=np.int64)
    out: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "device": str(device),
                           "note": "wall clock only; no valid score is computed, read or stored"}
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
        out["device_name"] = torch.cuda.get_device_name(device)
        out["device_count"] = int(torch.cuda.device_count())
    for arm in args.arms:
        t_build0 = time.perf_counter()
        model, info = M.build_arm(arm, norm_arr, 0, device)
        collate = D.make_collate(statics, arm=arm)
        t_build = time.perf_counter() - t_build0
        # first forward triggers the split-cache load
        gen = torch.Generator().manual_seed(101)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        epoch_times = []
        first_load = None
        for ep in range(1, 4):
            model.train()
            order = torch.randperm(len(rows), generator=gen).numpy()
            t0 = time.perf_counter()
            for start in range(0, len(order), 128):
                b = order[start : start + 128]
                batch = AT._to_device(collate(cache_train, b, arm, valid=False), device)
                target = torch.as_tensor(y_train[b], dtype=torch.float32, device=device)
                pred = model(batch)
                loss = torch.nn.functional.l1_loss(pred, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if first_load is None:
                    torch.cuda.synchronize(device) if device.type == "cuda" else None
                    first_load = time.perf_counter() - t0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_times.append(time.perf_counter() - t0)
        # one valid pass (timing only)
        cache_valid = AD.load_cache("valid")
        t0 = time.perf_counter()
        AT.evaluate(model, cache_valid, np.arange(1000, dtype=np.int64), candidate=arm,
                    device=device, batch_size=256, collate_fn=collate)
        valid_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        torch.save({"state_dict": AC.cpu_state(model)}, Path("/tmp/rmc_timing_save.pt"))
        save_seconds = time.perf_counter() - t0
        steady = float(np.mean(epoch_times[1:])) + valid_seconds
        overhead = 3.0 * steady  # two eval-mode train passes + final evals + saves
        projection = (steady * 600 + overhead) * (1.0 + 0.15)
        out[arm] = {
            "build_seconds": t_build,
            "first_epoch_seconds": epoch_times[0],
            "epoch_seconds": epoch_times,
            "valid_pass_seconds": valid_seconds,
            "save_seconds": save_seconds,
            "steady_epoch_seconds_train_plus_valid": steady,
            "projected_600_epoch_seconds": projection,
            "projected_600_epoch_hours": projection / 3600.0,
            "parameters": info["parameters"],
            "parameters_deployment": info["parameters_deployment"],
            "cache_load_in_first_epoch_seconds": first_load,
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else None
            ),
        }
        del model, collate
        M.clear_split_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out["projected_total_seconds_all_arms"] = float(
        sum(v["projected_600_epoch_seconds"] for v in out.values()
            if isinstance(v, dict) and "projected_600_epoch_seconds" in v)
    )
    C.write_json(C.ANALYSIS_DIR / "timing.json", out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in
                          ("steady_epoch_seconds_train_plus_valid", "projected_600_epoch_hours",
                           "parameters", "peak_cuda_memory_mb")}
                      for k, v in out.items() if isinstance(v, dict) and "projected_600_epoch_hours" in v}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    cfg = C.load_config()
    arm, seed = str(args.arm), int(args.seed)
    if arm not in C.ARMS or seed not in C.SEEDS:
        raise SystemExit("arm must be P/M/G/U and seed 0/1")
    device = torch.device(args.device)
    norm = C.read_json(C.norm_path())
    norm_arr = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else v) for k, v in norm.items()}
    statics = _statics_both()
    cache_train = AD.load_cache("train")
    cache_valid = AD.load_cache("valid")
    y_train = np.asarray(cache_train["labels"], dtype=np.float64)
    y_valid = np.asarray(cache_valid["labels"], dtype=np.float64)
    prefix = "" if args.tag is None else f"{args.tag}_"
    out = C.RUNS_DIR / f"{prefix}{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    u_c3 = None if args.u_c3 is None else bool(int(args.u_c3))
    builder = make_builder(arm, norm_arr, u_c3)
    collate = D.make_collate(statics, arm=arm)
    info = AT.train(
        arm, seed, device, recipe_variant="B", model_builder=builder,
        out_dir=out, collate_fn=collate, force=args.force, epochs=args.epochs,
    )
    print(json.dumps({"best_valid": info["best_valid"],
                      "final": info["fixed_objects"]["final"],
                      "wall_seconds": info["wall_seconds"]}, indent=2))

    final_model, final_info = M.build_arm(arm, norm_arr, seed, device, u_c3=u_c3)
    M.load_checkpoint(final_model, out / "final.pt", device)
    rows_valid = np.arange(C.N_VALID, dtype=np.int64)
    final_pred = eval_valid(final_model, cache_valid, rows_valid, arm, collate, device, 256)
    (out / "predictions").mkdir(exist_ok=True)
    np.savez(out / "predictions/valid_final.npz", row=rows_valid, pred=final_pred.astype(np.float32))
    best_z = np.load(out / "predictions/valid_best.npz", allow_pickle=False)
    final_mae = C.mae(final_pred, y_valid)
    best_mae = C.mae(best_z["pred"].astype(np.float64), y_valid)
    if abs(best_mae - float(info["best_valid"]["valid_mae"])) > 5e-6:
        raise SystemExit(f"best-valid replay mismatch: {best_mae} vs {info['best_valid']['valid_mae']}")
    if abs(final_mae - float(info["fixed_objects"]["final"]["valid_mae"])) > 5e-6:
        raise SystemExit(f"final-valid replay mismatch: {final_mae} vs {info['fixed_objects']['final']['valid_mae']}")
    init_model, init_info = M.build_arm(arm, norm_arr, seed, "cpu", u_c3=u_c3)
    extra = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm, "seed": seed,
        "git_commit": C.git_commit(),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "normalisation_sha256": C.sha256_file(C.norm_path()),
        "moment_cache": {
            "train_sha256": C.sha256_file(C.moments_path("train")),
            "valid_sha256": C.sha256_file(C.moments_path("valid")),
        },
        "ssl": final_info["ssl"],
        "parameters": final_info["parameters"],
        "parameters_online": final_info["parameters_online"],
        "parameters_memory_encoder": final_info["parameters_memory_encoder"],
        "parameters_deployment": final_info["parameters_deployment"],
        "parameter_breakdown": final_info["parameter_breakdown"],
        "use_c3": final_info["use_c3"],
        "global_measure": final_info["global_measure"],
        "init_state_sha256": init_info["init_state_sha256"],
        "best_checkpoint_state_sha256": _state_sha(out / "best.pt"),
        "final_checkpoint_state_sha256": _state_sha(out / "final.pt"),
        "best_valid_replay_mae": best_mae,
        "final_valid_replay_mae": final_mae,
        "predictions": {
            "valid_best_sha256": C.sha256_file(out / "predictions/valid_best.npz"),
            "valid_final_sha256": C.sha256_file(out / "predictions/valid_final.npz"),
        },
        "labels": {
            "train_sha256": C.sha256_array(y_train),
            "valid_sha256": C.sha256_array(y_valid),
        },
        "data_order": order_sequence_hash(seed, epochs=int(cfg["recipe"]["epochs"]), n_rows=len(y_train)),
        "windows": {k: v for k, v in C.window_metrics(info["curve"]).items()
                    if k not in ("epochs", "valid_curve")},
        "official_test_loaded": False,
        "curve_sha256": C.sha256_file(out / "curve.csv"),
    }
    C.write_json(out / "run_extra.json", extra)
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def _load_pred(run: Path, which: str) -> np.ndarray:
    z = np.load(run / f"predictions/valid_{which}.npz", allow_pickle=False)
    pred = z["pred"].astype(np.float64)
    rows = z["row"].astype(np.int64)
    if len(pred) != C.N_VALID or not np.array_equal(rows, np.arange(C.N_VALID)):
        raise SystemExit(f"{run}: predictions are not the full 1000-row valid split")
    return pred


def cmd_score(args: argparse.Namespace) -> int:
    run = Path(args.run).expanduser().resolve()
    info = C.read_json(run / "train.json")
    extra = C.read_json(run / "run_extra.json")
    groups = C.load_valid_groups()
    y = groups["y"]
    assert np.allclose(y, np.asarray(AD.load_cache("valid")["labels"], dtype=np.float64),
                       rtol=0.0, atol=1e-6)
    arm, seed = str(info["candidate"]), int(info["seed"])
    out: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID, "arm": arm, "seed": seed,
        "git_commit": info["git_commit"], "parameters": info["parameter_count"],
        "parameters_deployment": extra["parameters_deployment"],
        "recipe": info["recipe"],
        "best_valid": {**info["best_valid"],
                       "train_mae_eval_mode": info["fixed_objects"]["best"]["train_mae_eval_mode"]},
        "final": info["fixed_objects"]["final"],
        "windows": extra["windows"], "run_extra": extra,
        "runs": C.rel(run),
    }
    for which in ("best", "final"):
        pred = _load_pred(run, which)
        ae = np.abs(pred - y)
        out[which + "_metrics"] = C.metric_block(pred, y)
        out[which + "_groups"] = C.group_table(ae, groups["groups"])
        out[which + "_tail_calibration"] = {
            "tail": C.tail_calibration(pred, y, groups["tail"]),
            "tail_drop_row172": C.tail_calibration(pred, y, groups["tail"], drop_row172=True),
        }
        out[which + "_p3b_comparison"] = {
            "gain_vs_p3b_reference": float(groups["p3b_ae"].mean() - ae.mean()),
            "bootstrap_vs_p3b": C.paired_bootstrap(ae, groups["p3b_ae"]),
            "row172_ae": float(ae[172]),
            "mae_drop_row172": float(np.delete(ae, 172).mean()),
        }
        pred_path = C.ANALYSIS_DIR / "per_molecule" / f"{arm}_s{seed}_{which}.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["row_id", "y", "pred", "ae", "p3b_pred", "p3b_ae", "delta_ae_vs_p3b",
                        "radius2_unseen", "ordinary", "tail", "ring_rich"])
            for i in range(C.N_VALID):
                w.writerow([i, y[i], pred[i], ae[i], groups["p3b_pred"][i], groups["p3b_ae"][i],
                            ae[i] - groups["p3b_ae"][i], int(groups["radius2_unseen"][i]),
                            int(groups["ordinary"][i]), int(groups["tail"][i]),
                            int(groups["n_rings"][i] >= 4)])
        out[which + "_per_molecule_sha256"] = C.sha256_file(pred_path)
    C.write_json(C.ANALYSIS_DIR / "scores" / f"{arm}_s{seed}.json", out)
    print(json.dumps({"arm": arm, "seed": seed, "best": out["best_metrics"]["mae"],
                      "final": out["final_metrics"]["mae"], "windows": out["windows"]}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _ae(run: Path, which: str, y: np.ndarray) -> np.ndarray:
    return np.abs(_load_pred(run, which) - y)


def _seed_mean(arrays: Mapping[int, np.ndarray]) -> np.ndarray:
    return np.mean(np.stack([arrays[s] for s in sorted(arrays)], 0), axis=0)


def _contrast(name: str, ae_first: Mapping[int, np.ndarray], ae_second: Mapping[int, np.ndarray],
              groups: Mapping[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {"contrast": name, "per_seed": {}, "group_per_seed": {}}
    for s in sorted(ae_first):
        d = ae_first[s] - ae_second[s]
        boot = C.paired_bootstrap(ae_first[s], ae_second[s])
        out["per_seed"][str(s)] = {
            "delta_mae": float(d.mean()),
            "median_delta_ae": float(np.median(d)),
            "better_fraction": float((d < 0).mean()),
            "bootstrap": boot,
        }
        out["group_per_seed"][str(s)] = {}
        for gname, mask in groups.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            out["group_per_seed"][str(s)][gname] = {
                "n": int(mask.sum()),
                "delta_mae": float(d[mask].mean()),
                "mae_first": float(ae_first[s][mask].mean()),
                "mae_second": float(ae_second[s][mask].mean()),
            }
    if len(ae_first) > 1 and len(ae_second) > 1:
        f = _seed_mean(ae_first)
        sec = _seed_mean(ae_second)
        gm: dict[str, float] = {}
        for gname, mask in groups.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            gm[gname] = float((f[mask] - sec[mask]).mean())
        out["seed_mean"] = {
            "delta_mae": float((f - sec).mean()),
            "bootstrap_seed_mean_molecule": C.paired_bootstrap(f, sec),
            "groups": gm,
        }
    return out


def _drop_top10(ae_first: np.ndarray, ae_second: np.ndarray, k: int = 10) -> dict[str, Any]:
    avg = 0.5 * (np.asarray(ae_first) + np.asarray(ae_second))
    rows = np.argsort(-avg, kind="stable")[:k]
    keep = np.ones(len(avg), dtype=bool)
    keep[rows] = False
    return {
        "rows": [int(r) for r in rows],
        "delta_mae_all": float((ae_first - ae_second).mean()),
        "delta_mae_drop_top10": float((ae_first[keep] - ae_second[keep]).mean()),
        "rule": "drop the 10 molecules with the largest mean AE across the two compared models",
    }


def cmd_analyze(args: argparse.Namespace) -> int:
    groups_all = C.load_valid_groups()
    y = groups_all["y"]
    groups = dict(groups_all["groups"])
    groups["ordinary_935"] = groups_all["ordinary"]
    score_files = sorted((C.ANALYSIS_DIR / "scores").glob("*.json"))
    if not score_files:
        raise SystemExit("no scores; run score first")
    scores = {}
    for f in score_files:
        d = C.read_json(f)
        scores[(d["arm"], int(d["seed"]))] = d
    arms = sorted({a for a, _ in scores})
    seeds = sorted({s for (_, s) in scores})
    rows = []
    main: dict[str, Any] = {}
    for (arm, seed), d in sorted(scores.items()):
        main.setdefault(arm, {})[str(seed)] = {
            "best_valid": d["best_valid"]["valid_mae"],
            "best_epoch": d["best_valid"]["epoch"],
            "final_valid": d["final"]["valid_mae"],
            "eval_train_best": d["best_valid"]["train_mae_eval_mode"],
            "eval_train_final": d["final"]["train_mae_eval_mode"],
            "W20": d["windows"]["W20"],
            "T100": d["windows"]["T100"],
            "parameters": d["parameters"],
            "parameters_deployment": d["parameters_deployment"],
            "wall_seconds": d["run_extra"].get("wall_seconds"),
            "wall_hours": d["recipe"]["epochs"] and None,
        }
    # archive the per-epoch curves next to the tables (runs/ is gitignored)
    curves_dir = C.ANALYSIS_DIR / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    for (arm, seed), d in scores.items():
        src = C.REPO / d["runs"] / "curve.csv"
        if src.exists():
            shutil.copyfile(src, curves_dir / f"{arm}_s{seed}.csv")

    # wall time per run from train.json
    for (arm, seed), d in scores.items():
        run = C.REPO / d["runs"]
        tj = C.read_json(run / "train.json")
        main[arm][str(seed)]["wall_seconds"] = float(tj["wall_seconds"])
        main[arm][str(seed)]["wall_hours"] = float(tj["wall_seconds"]) / 3600.0

    contrasts = {}
    pairs = [("M", "P"), ("G", "P"), ("M", "G")]
    if "U" in arms:
        pairs += [("U", "P"), ("U", "M") if "M" in arms else ("U", "G")]
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        for which in ("best", "final"):
            ae_a = {s: _ae(C.REPO / scores[(a, s)]["runs"], which, y) for s in seeds if (a, s) in scores}
            ae_b = {s: _ae(C.REPO / scores[(b, s)]["runs"], which, y) for s in seeds if (b, s) in scores}
            common = sorted(set(ae_a) & set(ae_b))
            if not common:
                continue
            c = _contrast(f"{a}-{b}:{which}", {s: ae_a[s] for s in common}, {s: ae_b[s] for s in common}, groups)
            for s in common:
                c.setdefault("drop_top10", {})[str(s)] = _drop_top10(ae_a[s], ae_b[s])
            contrasts[f"{a}-{b}:{which}"] = c

    # ---- frozen decision --------------------------------------------
    def have(arm: str, seed: int) -> bool:
        return arm in main and str(seed) in main[arm]

    def t100(arm: str, seed: int) -> float:
        return float(main[arm][str(seed)]["T100"])

    def t100_gain(a: str, b: str, seed: int) -> float:
        return t100(b, seed) - t100(a, seed)  # positive = a better

    def both_seeds_positive(a: str, b: str) -> bool:
        common = [s for s in seeds if have(a, s) and have(b, s)]
        return bool(common) and all(t100_gain(a, b, s) > 0 for s in common)

    def group_delta(a: str, b: str, which: str, key: str) -> dict[str, float]:
        outd = {}
        for s in seeds:
            if (a, s) not in scores or (b, s) not in scores:
                continue
            aa = _ae(C.REPO / scores[(a, s)]["runs"], which, y)
            bb = _ae(C.REPO / scores[(b, s)]["runs"], which, y)
            mask = np.asarray(groups[key], dtype=bool)
            outd[str(s)] = float(aa[mask].mean() - bb[mask].mean())
        return outd

    decision: dict[str, Any] = {"arms": arms, "seeds": seeds, "endpoint": "T100 (epochs 501..600)"}
    complete = [s for s in seeds if all(have(a, s) for a in ("P", "M", "G"))]
    if all(a in arms for a in ("P", "M", "G")) and len(complete) >= 2:
        sP_M = both_seeds_positive("M", "P")
        sP_G = both_seeds_positive("G", "P")
        sMG = all(t100_gain("M", "G", s) > 0 for s in complete)
        mean_gain_M = float(np.mean([t100_gain("M", "P", s) for s in complete]))
        mean_gain_G = float(np.mean([t100_gain("G", "P", s) for s in complete]))
        ord_best_M = group_delta("M", "P", "best", "ordinary_935")
        ord_final_M = group_delta("M", "P", "final", "ordinary_935")
        un_final_M = group_delta("M", "P", "final", "unseen_ordinary")
        decision.update({
            "T100_gain_vs_P": {
                "M": {str(s): t100_gain("M", "P", s) for s in seeds},
                "G": {str(s): t100_gain("G", "P", s) for s in seeds},
                "M_minus_G": {str(s): t100_gain("M", "G", s) for s in seeds},
            },
            "mean_T100_gain_vs_P": {"M": mean_gain_M, "G": mean_gain_G},
            "both_seeds_T100_M_better_P": sP_M,
            "both_seeds_T100_G_better_P": sP_G,
            "both_seeds_T100_M_better_G": sMG,
            "ordinary_best_M_minus_P": ord_best_M,
            "ordinary_final_M_minus_P": ord_final_M,
            "unseen_ordinary_final_M_minus_P": un_final_M,
        })
        h1 = bool(sP_M and mean_gain_M >= 0.003)
        h2 = bool(sMG and float(np.mean([t100_gain("M", "G", s) for s in complete])) >= 0.0)
        pos_ord = all(v < 0 for v in ord_final_M.values()) and all(v < 0 for v in ord_best_M.values())
        pos_un = all(v < 0 for v in un_final_M.values())
        decision.update({
            "h1_supported": h1,
            "h2_supported": h2,
            "h1_rule": "both seeds T100(M) < T100(P) and seed-mean gain >= 0.003",
            "h2_rule": "both seeds T100(M) < T100(G) and seed-mean gain >= 0.0",
            "subgroup_reading": {"ordinary_best_and_final_both_better": pos_ord,
                                 "unseen_ordinary_final_both_better": pos_un},
        })
        if h1 and h2:
            decision["verdict"] = (
                "POSITIVE_M_BETTER_THAN_P_AND_G" if (pos_ord and pos_un)
                else "POSITIVE_WITH_CONFLICTING_SUBGROUPS"
            )
        elif h1:
            decision["verdict"] = "BODY_PROGRESS_C3_NOT_REPRODUCED"
        elif sP_G and not sP_M:
            decision["verdict"] = "GAUSSIAN_BODY_PROGRESS_M_NOT_BETTER"
        elif h2:
            decision["verdict"] = "C3_LIMITED_EVIDENCE_BUT_BELOW_P"
        else:
            decision["verdict"] = "NO_STABLE_GAIN_OVER_P"
        # ---- U trigger (pre-declared purchase rule) ------------------
        cand = None
        if "G" in arms and "M" in arms:
            meanM = float(np.mean([t100("M", s) for s in complete]))
            meanG = float(np.mean([t100("G", s) for s in complete]))
            cand = "M" if meanM <= meanG else "G"  # prospective method = better mean T100
        elif "M" in arms:
            cand = "M"
        elif "G" in arms:
            cand = "G"
        trigger = {}
        if cand is not None:
            mean_gain = float(np.mean([t100_gain(cand, "P", s) for s in complete]))
            ord_b = group_delta(cand, "P", "best", "ordinary_935")
            ord_f = group_delta(cand, "P", "final", "ordinary_935")
            un_f = group_delta(cand, "P", "final", "unseen_ordinary")
            trigger = {
                "candidate_arm": cand,
                "mean_T100_gain_vs_P": mean_gain,
                "both_seeds_T100_positive": both_seeds_positive(cand, "P"),
                "ordinary_best_seed_mean_gain": -float(np.mean(list(ord_b.values()))),
                "ordinary_final_seed_mean_gain": -float(np.mean(list(ord_f.values()))),
                "unseen_ordinary_final_per_seed_gain": {k: -v for k, v in un_f.items()},
                "unseen_ordinary_final_seed_mean_gain": -float(np.mean(list(un_f.values()))),
                "rule": "both seeds T100>0, mean>=0.003, ordinary best+final seed-mean>0, unseen+ordinary final both seeds not worse and mean>=0.005, budget allows 2x600 epochs U",
                "c1": bool(both_seeds_positive(cand, "P") and mean_gain >= 0.003),
                "c2": bool(float(np.mean(list(ord_b.values()))) < 0 and float(np.mean(list(ord_f.values()))) < 0),
                "c3": bool(all(v <= 0 for v in un_f.values()) and (-float(np.mean(list(un_f.values())))) >= 0.005),
            }
            trigger["group_rule_satisfied"] = bool(trigger["c1"] and trigger["c2"] and trigger["c3"])
        decision["u_trigger"] = trigger
    else:
        decision["verdict"] = "INCOMPLETE"

    prep = C.read_json(C.ANALYSIS_DIR / "prep.json")
    summary = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "main": main,
        "contrasts": contrasts,
        "decision": decision,
        "bootstrap_note": "molecule-level paired bootstrap only; it does not estimate training-seed variance (2 seeds)",
        "parameters": {
            "P": 413505, "M": 415250, "G": 415250, "U": 415250,
            "frozen_memory_atom_encoder": 26336, "deployment": 441586, "cap": 500000,
        },
        "groups": {k: int(np.asarray(v, dtype=bool).sum()) for k, v in groups.items()},
        "reference": {
            "p3b_valid_mae": float(groups_all["p3b_valid_mae"]),
            "nr_v1_P_seed_best_range": [0.076428, 0.078611],
            "note": "nr_v1 same-architecture P, different round; only a noise-scale reference",
        },
        "cache": {
            "train_moments_sha256": prep["train"]["moments_sha256"],
            "valid_moments_sha256": prep["valid"]["moments_sha256"],
            "train_static_sha256": prep["train"]["static_sha256"],
            "train_moments_bytes": prep["train"]["bytes"],
            "valid_moments_bytes": prep["valid"]["bytes"],
            "norm_sha256": C.sha256_file(C.norm_path()),
        },
        "runs": {
            f"{arm}_s{seed}": {
                "git_commit": scores[(arm, seed)]["git_commit"],
                "recipe": scores[(arm, seed)]["recipe"],
                "moment_cache": scores[(arm, seed)]["run_extra"]["moment_cache"],
                "init_state_sha256": scores[(arm, seed)]["run_extra"]["init_state_sha256"],
                "best_state_sha256": scores[(arm, seed)]["run_extra"]["best_checkpoint_state_sha256"],
                "final_state_sha256": scores[(arm, seed)]["run_extra"]["final_checkpoint_state_sha256"],
                "best_valid_replay_mae": scores[(arm, seed)]["run_extra"]["best_valid_replay_mae"],
                "final_valid_replay_mae": scores[(arm, seed)]["run_extra"]["final_valid_replay_mae"],
                "data_order": scores[(arm, seed)]["run_extra"]["data_order"],
                "use_c3": scores[(arm, seed)]["run_extra"]["use_c3"],
                "wall_seconds": main[arm][str(seed)]["wall_seconds"],
            }
            for (arm, seed) in sorted(scores)
        },
        "official_test_loaded": False,
        "official_test_policy": "terminal only; not read because the frozen U/group trigger did not fire",
    }
    C.write_json(C.ANALYSIS_DIR / "summary.json", summary)
    _write_tables(main, contrasts, decision, groups_all)
    print(json.dumps({"verdict": decision.get("verdict"),
                      "T100_gain_vs_P": decision.get("T100_gain_vs_P"),
                      "u_trigger": {k: v for k, v in decision.get("u_trigger", {}).items()
                                    if k in ("candidate_arm", "mean_T100_gain_vs_P", "c1", "c2", "c3",
                                             "group_rule_satisfied")}}, indent=2))
    return 0


def _write_tables(main, contrasts, decision, groups_all) -> None:
    lines = ["# RMC main table (valid, raw-y MAE; lower is better)", ""]
    lines.append("| arm | seed | best-valid | epoch | final | W20 | T100 | train@best | train@final | params | wall h |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in sorted(main):
        for seed in sorted(main[arm], key=int):
            d = main[arm][seed]
            lines.append(
                f"| {arm} | {seed} | {d['best_valid']:.6f} | {d['best_epoch']} | {d['final_valid']:.6f} | "
                f"{d['W20']:.6f} | {d['T100']:.6f} | {d['eval_train_best']:.6f} | {d['eval_train_final']:.6f} | "
                f"{d['parameters']} | {d['wall_hours']:.3f} |"
            )
    lines += ["", "Contrasts use `delta = MAE(first) - MAE(second)`; negative = first better.", ""]
    for name, c in contrasts.items():
        lines.append(f"## {name}")
        lines.append("| seed | delta MAE | 95% CI | better frac | ordinary | unseen+ordinary |")
        lines.append("|---:|---:|---|---:|---:|---:|")
        for seed, v in c["per_seed"].items():
            ci = v["bootstrap"]["ci95"]
            g = c["group_per_seed"][seed]
            lines.append(
                f"| {seed} | {v['delta_mae']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | "
                f"{v['better_fraction']:.3f} | {g.get('ordinary_935', {}).get('delta_mae', float('nan')):+.6f} | "
                f"{g.get('unseen_ordinary', {}).get('delta_mae', float('nan')):+.6f} |"
            )
        if "seed_mean" in c:
            sm = c["seed_mean"]
            ci = sm["bootstrap_seed_mean_molecule"]["ci95"]
            lines.append(
                f"| mean | {sm['delta_mae']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | "
                f"{sm['bootstrap_seed_mean_molecule']['better_fraction']:.3f} | "
                f"{sm['groups'].get('ordinary_935', float('nan')):+.6f} | "
                f"{sm['groups'].get('unseen_ordinary', float('nan')):+.6f} |"
            )
        lines.append("")
    C.write_json(C.ANALYSIS_DIR / "main_table.json", main)
    (C.ANALYSIS_DIR / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (C.ANALYSIS_DIR / "decision.md").write_text(
        "# Frozen decision (RMC)\n\n```json\n" + json.dumps(decision, indent=2) + "\n```\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# probe (frozen checkpoints, no training)
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    groups_all = C.load_valid_groups()
    y = groups_all["y"]
    norm = C.read_json(C.norm_path())
    norm_arr = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else v) for k, v in norm.items()}
    statics = _statics_both()
    cache_valid = AD.load_cache("valid")
    device = torch.device(args.device)
    rows = np.arange(C.N_VALID, dtype=np.int64)
    out: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "device": str(device),
                           "note": "fixed-weight diagnostics; no gradient step, no checkpoint change"}
    runs = []
    for arm in ("M", "G", "U"):
        for seed in C.SEEDS:
            p = C.RUNS_DIR / f"{arm}_s{seed}"
            if (p / "train.json").exists():
                runs.append((arm, seed, p))
    for arm, seed, run in runs:
        info = C.read_json(run / "train.json")
        extra = C.read_json(run / "run_extra.json")
        model, minfo = M.build_arm(arm, norm_arr, seed, device,
                                   u_c3=bool(extra.get("use_c3", arm == "M")))
        collate = D.make_collate(statics, arm=arm)
        rec: dict[str, Any] = {"parameters": info["parameter_count"],
                               "use_c3": extra.get("use_c3"),
                               "global_measure": extra.get("global_measure")}
        for which in ("best", "final"):
            M.load_checkpoint(model, run / f"{which}.pt", device)
            base = eval_valid(model, cache_valid, rows, arm, collate, device, 128)
            ae = np.abs(base - y)
            rec[which] = {"mae": float(ae.mean()),
                          "state_sha256": _state_sha(run / f"{which}.pt")}
            if arm in ("M", "U") and bool(extra.get("use_c3", True)):
                model.use_c3 = False
                off = eval_valid(model, cache_valid, rows, arm, collate, device, 128)
                model.use_c3 = True
                d = np.abs(off - y) - np.abs(base - y)
                rec[which]["c3_off"] = {
                    "mae": float(np.abs(off - y).mean()),
                    "delta_mae_off_minus_on": float(d.mean()),
                    "per_molecule_delta_median": float(np.median(d)),
                    "pred_change_mean": float(np.abs(off - base).mean()),
                    "pred_change_max": float(np.abs(off - base).max()),
                }
        # query / C3 magnitude diagnostics on one fixed batch
        model.eval()
        batch_rows = rows[:128]
        batch = AT._to_device(collate(cache_valid, batch_rows, arm, valid=True), device)
        with torch.no_grad():
            a, b0, w0 = model.encoders(batch)
            n_atoms = int(a.shape[0])
            bi = batch["bond_index"]
            deg = batch["rmc_deg"].to(a.dtype)
            sum_b = a.new_zeros((n_atoms, b0.shape[1]))
            sum_b.index_add_(0, torch.cat([bi[:, 0], bi[:, 1]]), torch.cat([b0, b0], dim=0))
            mean_b = sum_b / deg.clamp_min(1.0).unsqueeze(1)
            ji = batch["junc_index"]
            nj = batch["rmc_njunc"].to(a.dtype)
            sum_w = a.new_zeros((n_atoms, w0.shape[1]))
            sum_w.index_add_(0, ji[:, 1], w0)
            mean_w = sum_w / nj.clamp_min(1.0).unsqueeze(1)
            e = model.e_mlp(torch.cat([a, mean_b, mean_w, torch.log1p(deg).unsqueeze(1),
                                       torch.log1p(nj).unsqueeze(1)], 1))
            q = 4.0 * torch.tanh(model.q_proj(e)).reshape(-1, C.N_HEAD, C.PROJ_DIM) / math.sqrt(C.PROJ_DIM)
            cache = M.split_cache("valid", device)
            idx = batch["rmc_moment_index"]
            mu = cache["mu"][idx]
            c2 = cache["c2"][idx]
            c3 = cache["c3"][idx]
            # q = 0 -> F = mu, logZ = log(1+eps)
            q0 = torch.zeros(mu.shape[0], 1, C.PROJ_DIM, device=mu.device, dtype=mu.dtype)
            F0, Z0 = RM.readout_batched(mu[:, :, 0, :], c2[:, :, 0, :], c3[:, :, 0, :], q0, use_c3=True)
            F1, _ = RM.readout_batched(mu[:, :, 1, :], c2[:, :, 1, :], c3[:, :, 1, :], q0, use_c3=True)
            rec["q_zero"] = {
                "max_abs_F_minus_mu": float(max((F0[:, :, 0, :] - mu[:, :, 0, :]).abs().max(),
                                                (F1[:, :, 0, :] - mu[:, :, 1, :]).abs().max())),
                "max_abs_logZ_minus_log1peps": float((torch.log(Z0) - math.log(1 + C.EPS)).abs().max()),
            }
            # C3 term vs 2 a C2 q
            qq = RM.gram_pairs(q)
            a_hs = 1.0 + torch.einsum("nsc,nhc->nsh", mu[:, :, 0, :], q[:, 0:2])
            c2q = torch.einsum("nsab,nhb->nsha", c2[:, :, 0, :][..., RM._torch_maps(mu.device, mu.dtype)["c2"]]
                               .reshape(-1, C.N_SCALE, C.PROJ_DIM, C.PROJ_DIM), q[:, 0:2])
            c3qq = torch.einsum("nsap,nhp->nsha",
                                c3[:, :, 0, :][..., RM._torch_maps(mu.device, mu.dtype)["c3"]]
                                .reshape(-1, C.N_SCALE, C.PROJ_DIM, C.N_PAIR), qq[:, 0:2])
            lin = 2.0 * a_hs.unsqueeze(-1) * c2q
            ratio = (c3qq.norm(dim=-1) / lin.norm(dim=-1).clamp_min(1e-9))
            rec["c3_magnitude"] = {
                "query_norm_mean": float(q.norm(dim=-1).mean()),
                "query_norm_per_head": [float(q[:, h].norm(dim=-1).mean()) for h in range(C.N_HEAD)],
                "c3_over_linear_norm_median": float(ratio.median()),
                "c3_over_linear_norm_mean": float(ratio.mean()),
                "c3_over_linear_norm_per_scale": [float(ratio[:, s].mean()) for s in range(C.N_SCALE)],
                "logZ_mean_per_scale": [
                    float(torch.log(a_hs[:, s] * a_hs[:, s]
                                    + torch.einsum("np,nhp->nh", c2[:, s, 0, :], qq[:, 0:2])
                                    + C.EPS).mean())
                    for s in range(C.N_SCALE)
                ],
            }
            rec["input_blocks"] = {
                "e_norm_mean": float(e.norm(dim=1).mean()),
                "F_abs_mean": float(RM.readout_batched(mu[:, :, 0, :], c2[:, :, 0, :], c3[:, :, 0, :],
                                                       q[:, 0:2], use_c3=bool(extra.get("use_c3", True)))[0].abs().mean()),
            }
        out[f"{arm}_s{seed}"] = rec
        del model
        M.clear_split_cache()
    C.write_json(C.ANALYSIS_DIR / "probe_summary.json", out)
    print(json.dumps({k: {"best_mae": v["best"]["mae"], "final_mae": v["final"]["mae"],
                          "c3_off": v["best"].get("c3_off"),
                          "q_zero": v["q_zero"]} for k, v in out.items() if k.startswith(("M_", "G_", "U_"))},
                     indent=2))
    return 0


# ---------------------------------------------------------------------------
# terminal test (only after the frozen trigger)
# ---------------------------------------------------------------------------


def cmd_test(args: argparse.Namespace) -> int:
    summary_path = C.ANALYSIS_DIR / "summary.json"
    if not summary_path.exists():
        raise SystemExit("no summary.json; analysis must be frozen before any test read")
    summary = C.read_json(summary_path)
    decision = summary["decision"]
    trigger = decision.get("u_trigger", {})
    if not trigger.get("group_rule_satisfied"):
        raise SystemExit("test trigger not satisfied; refusing to touch the official test split")
    out_path = C.ANALYSIS_DIR / "test_terminal.json"
    if out_path.exists():
        raise SystemExit("terminal test result already exists; refusing a second read")
    cfg = C.load_config()
    # build the test moment cache with the frozen train statistics (no labels)
    prep = C.read_json(C.ANALYSIS_DIR / "prep.json")
    ssl_path = C.REPO / cfg["ssl"]["path"]
    if C.sha256_file(ssl_path) != cfg["ssl"]["sha256"]:
        raise SystemExit("SSL checkpoint hash mismatch")
    norm = C.read_json(C.norm_path())
    enc = D.load_memory_atom_encoder(ssl_path)
    cache_test = AD.load_cache("test")
    a_ssl = D.atom_ssl_features(np.asarray(cache_test["atom_feat"]), enc)
    d = D.endpoint_matrix(np.asarray(cache_test["atom_feat"]), a_ssl)
    ds, _, _ = D.standardise_columns(d, np.asarray(norm["d_mean"]), np.asarray(norm["d_scale"]))
    R = np.asarray(norm["R"], dtype=np.float32).reshape(2, C.ENDPOINT_DIM, C.R_PROJ)
    u = D.view_features(ds, R)
    m = D.build_split_moments("test", u, workers=int(args.workers), chunk=256, log=print)
    test_sha = D.save_moments(m, C.moments_path("test"))
    static_sha = D.save_static("test", cache_test)
    del m
    statics = _statics_both()
    statics["test"] = D.load_static("test")
    norm_arr = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else v) for k, v in norm.items()}
    y = np.asarray(cache_test["labels"], dtype=np.float64)
    rows = np.arange(len(y), dtype=np.int64)
    device = torch.device(args.device)
    result: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "note": "single terminal read; selection frozen in analysis/summary.json",
        "candidate_family": trigger.get("candidate_arm"),
        "test_moments_sha256": test_sha,
        "test_static_sha256": static_sha,
        "runs": {},
    }
    for arm in trigger.get("candidate_arm", ""), "P":
        for seed in C.SEEDS:
            run = C.RUNS_DIR / f"{arm}_s{seed}"
            if not (run / "train.json").exists():
                continue
            extra = C.read_json(run / "run_extra.json")
            model, _ = M.build_arm(arm, norm_arr, seed, device, u_c3=bool(extra.get("use_c3", arm == "M")))
            collate = D.make_collate(statics, arm=arm)
            for which in ("best", "final"):
                M.load_checkpoint(model, run / f"{which}.pt", device)
                pred = eval_valid(model, cache_test, rows, arm, collate, device, 256)
                result["runs"][f"{arm}_s{seed}_{which}"] = {
                    "mae": float(np.abs(pred - y).mean()),
                    "state_sha256": _state_sha(run / f"{which}.pt"),
                }
                np.savez(run / f"predictions/test_terminal_{which}.npz", row=rows,
                         pred=pred.astype(np.float32))
            del model
            M.clear_split_cache()
    result["official_test_loaded"] = True
    C.write_json(out_path, result)
    print(json.dumps(result, indent=2))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rmc_runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prep")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--chunk", type=int, default=128)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_prep)

    s = sub.add_parser("checks")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--rebuild", action="store_true")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_checks)

    s = sub.add_parser("smoke")
    s.add_argument("--arms", nargs="+", default=["P", "M"])
    s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--n-train", type=int, default=16)
    s.add_argument("--n-valid", type=int, default=4)
    s.add_argument("--workers", type=int, default=2)
    s.set_defaults(func=cmd_smoke)

    s = sub.add_parser("timing")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--arms", nargs="+", default=["P", "M", "G"])
    s.set_defaults(func=cmd_timing)

    s = sub.add_parser("train")
    s.add_argument("--arm", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--tag", default=None)
    s.add_argument("--epochs", type=int, default=None)
    s.add_argument("--u-c3", type=int, default=None)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("analyze")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("probe")
    s.add_argument("--device", default="cuda:0")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("test")
    s.add_argument("--device", default="cpu")
    s.add_argument("--workers", type=int, default=16)
    s.set_defaults(func=cmd_test)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
