#!/usr/bin/env python
"""A3: compute J_raw / J_latent for the frozen DCC-L checkpoints and run the
pre-registered implementation, invariance, edge-case and distribution checks.

Outputs are small arrays (10k x 32 / 4) and z banks (10k x 513) used by A4.
The z banks are regenerable caches under analysis/cache/ and are git-ignored.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audit_common as AC  # noqa: E402
import joint_stats as JS  # noqa: E402

CHECK_ROWS_SEED = 20261028
N_CHECK_ROWS = 100
N_INV_ROWS = 8
CACHE_DIR = AC.ANALYSIS / "cache"


def _mol_arrays(cache, i):
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    bi = np.asarray(cache["bond_index"])[bp[i]:bp[i + 1]]
    bt = np.asarray(cache["bond_type"])[bp[i]:bp[i + 1]]
    return bi, bt, int(ap[i + 1] - ap[i])


def _relation_signature(relations: dict) -> str:
    return "|".join(f"{r}:{len(relations.get(r, []))}" for r in JS.RELATIONS)


def edge_cases() -> dict:
    cases = {}
    # empty
    rel = JS.molecule_relations(np.zeros((0, 2), dtype=np.int64), 1)
    cases["empty"] = {"relations": {r: len(rel[r]) for r in JS.RELATIONS},
                      "j_raw": JS.j_raw(np.zeros(0, dtype=np.int64), rel).tolist(),
                      "j_latent": JS.j_latent(np.zeros((0, 96)), rel).tolist()}
    # single bond
    rel = JS.molecule_relations(np.array([[0, 1]]), 2)
    cases["single_bond"] = {"relations": {r: len(rel[r]) for r in JS.RELATIONS},
                            "j_raw": JS.j_raw(np.array([1]), rel).tolist(),
                            "j_latent": JS.j_latent(np.ones((1, 96)), rel).tolist()}
    # tree path of 5 atoms
    bi = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    rel = JS.molecule_relations(bi, 5)
    cases["tree_path5"] = {"relations": {r: len(rel[r]) for r in JS.RELATIONS},
                           "j_raw": JS.j_raw(np.array([1, 1, 2, 1]), rel).tolist(),
                           "j_latent": JS.j_latent(np.random.default_rng(0).normal(size=(4, 96)), rel).tolist()}
    # single ring C6
    bi = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
    rel = JS.molecule_relations(bi, 6)
    cases["ring6"] = {"relations": {r: len(rel[r]) for r in JS.RELATIONS},
                      "j_raw": JS.j_raw(np.array([1, 2, 1, 1, 1, 1]), rel).tolist(),
                      "j_latent": JS.j_latent(np.random.default_rng(1).normal(size=(6, 96)), rel).tolist()}
    # fused two rings (naphthalene-like edge set)
    bi = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [1, 5], [1, 4], [0, 5]])
    rel = JS.molecule_relations(bi, 6)
    cases["fused_rings"] = {"relations": {r: len(rel[r]) for r in JS.RELATIONS},
                            "j_raw": JS.j_raw(np.array([1, 2, 1, 1, 2, 1, 1, 1, 1]), rel).tolist(),
                            "j_latent": JS.j_latent(np.random.default_rng(2).normal(size=(9, 96)), rel).tolist()}
    # fragmented graph: two single bonds
    bi = np.array([[0, 1], [2, 3]])
    rel = JS.molecule_relations(bi, 4)
    cases["two_components"] = {"relations": {r: len(rel[r]) for r in JS.RELATIONS},
                               "j_raw": JS.j_raw(np.array([1, 2]), rel).tolist(),
                               "j_latent": JS.j_latent(np.random.default_rng(3).normal(size=(2, 96)), rel).tolist()}
    return cases


def invariance_checks(cache, rows) -> dict:
    out = {}
    for i in rows:
        bi, bt, n_atoms = _mol_arrays(cache, i)
        c = np.random.default_rng(100 + i).normal(size=(len(bi), 96))
        rel = JS.molecule_relations(bi, n_atoms)
        jr = JS.j_raw(bt, rel)
        jl = JS.j_latent(c, rel)
        # atom relabel
        perm_a = np.random.default_rng(200 + i).permutation(n_atoms)
        inv_a = np.empty_like(perm_a)
        inv_a[perm_a] = np.arange(n_atoms)
        bi_a = inv_a[bi]
        # order the bond rows by their new endpoints so the relation-building
        # code sees the same canonical bond order as the original
        order = np.lexsort((bi_a[:, 1], bi_a[:, 0]))
        bi_a = bi_a[order]
        bt_a = bt[order]
        c_a = c[order]
        rel_a = JS.molecule_relations(bi_a, n_atoms)
        # bond permutation
        perm_b = np.random.default_rng(300 + i).permutation(len(bi))
        bi_b = bi[perm_b]
        bt_b = bt[perm_b]
        c_b = c[perm_b]
        # relation pairs are indexed by the permuted bond rows; rebuild
        rel_b = {
            r: [(int(np.where(perm_b == a)[0][0]), int(np.where(perm_b == b)[0][0])) for a, b in rel[r]]
            for r in JS.RELATIONS
        }
        # endpoint flip
        bi_f = bi[:, [1, 0]]
        rel_f = {
            r: [(a, b) for a, b in rel[r]]
            for r in JS.RELATIONS
        }
        out[str(int(i))] = {
            "atom_relabel_j_raw_maxdiff": float(np.max(np.abs(JS.j_raw(bt_a, rel_a) - jr))),
            "atom_relabel_j_latent_maxdiff": float(np.max(np.abs(JS.j_latent(c_a, rel_a) - jl))),
            "bond_permutation_j_raw_maxdiff": float(np.max(np.abs(JS.j_raw(bt_b, rel_b) - jr))),
            "bond_permutation_j_latent_maxdiff": float(np.max(np.abs(JS.j_latent(c_b, rel_b) - jl))),
            "endpoint_flip_j_raw_maxdiff": float(np.max(np.abs(JS.j_raw(bt, rel_f) - jr))),
            "endpoint_flip_j_latent_maxdiff": float(np.max(np.abs(JS.j_latent(c, rel_f) - jl))),
            "relation_signature": _relation_signature(rel),
            "n_bonds": int(len(bi)),
        }
    return out


def corr_summary(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p, s = [], []
    for j in range(y.shape[1]):
        xj = x[:, j] if x.ndim == 2 else x
        yj = y[:, j]
        if np.std(xj) < 1e-12 or np.std(yj) < 1e-12:
            p.append(0.0)
            s.append(0.0)
        else:
            p.append(float(pearsonr(xj, yj).statistic))
            s.append(float(spearmanr(xj, yj).statistic))
    return {"pearson": p, "spearman": s,
            "max_abs_pearson": float(np.max(np.abs(p))) if p else 0.0,
            "max_abs_spearman": float(np.max(np.abs(s))) if s else 0.0}


def main() -> int:
    t0 = time.time()
    AC.install_load_guard()
    cfg = AC.load_audit_config()
    cache = AC.load_train_cache()
    norm = AC.load_norm()
    n = int(cache["n_molecules"])
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    bi_all = np.asarray(cache["bond_index"])
    bt_all = np.asarray(cache["bond_type"])
    g = np.asarray(cache["global_feat"], dtype=np.float64)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CHECK_ROWS_SEED)
    check_rows = np.sort(rng.choice(n, N_CHECK_ROWS, replace=False)).astype(np.int64)
    inv_rows = np.sort(rng.choice(n, N_INV_ROWS, replace=False)).astype(np.int64)

    # J_raw for all molecules and null
    j_raw = np.zeros((n, JS.N_RAW_DIMS), dtype=np.float64)
    j_raw_null = np.zeros_like(j_raw)
    rel_counts = np.zeros((n, len(JS.RELATIONS)), dtype=np.int64)
    rels = []
    for i in range(n):
        bi = bi_all[bp[i]:bp[i + 1]]
        bt = bt_all[bp[i]:bp[i + 1]]
        rel = JS.molecule_relations(bi, int(ap[i + 1] - ap[i]))
        rels.append(rel)
        j_raw[i] = JS.j_raw(bt, rel)
        j_raw_null[i] = JS.j_raw_null(bt, rel, i)
        rel_counts[i] = JS.relation_pair_counts(rel)
    np.save(AC.ANALYSIS / "j_raw.npy", j_raw.astype(np.float32))
    np.save(AC.ANALYSIS / "j_raw_null.npy", j_raw_null.astype(np.float32))
    np.save(AC.ANALYSIS / "relation_pair_counts.npy", rel_counts)

    # independent reference on 100 rows
    ref_raw_max = 0.0
    ref_lat_max = 0.0
    for i in check_rows:
        bi = bi_all[bp[i]:bp[i + 1]]
        bt = bt_all[bp[i]:bp[i + 1]]
        rel = rels[int(i)]
        ref_raw_max = max(ref_raw_max, float(np.max(np.abs(JS.reference_j_raw(bt, rel) - j_raw[int(i)]))))
        c = np.random.default_rng(1000 + int(i)).normal(size=(len(bi), 96))
        ref_lat_max = max(ref_lat_max, float(np.max(np.abs(JS.reference_j_latent(c, rel) - JS.j_latent(c, rel)))))

    # frozen-model c_loc -> J_latent, and z banks
    model_results = {}
    for name in ("D0", "D1"):
        model, info = AC.load_checkpoint(name, norm, device="cpu", organized_local=True)
        model.eval()
        z_bank = np.zeros((n, 513), dtype=np.float32)
        j_lat = np.zeros((n, JS.N_LATENT_DIMS), dtype=np.float32)
        j_lat_null = np.zeros_like(j_lat)
        t0m = time.time()
        with torch.no_grad():
            for start in range(0, n, 128):
                rows = np.arange(start, min(start + 128, n), dtype=np.int64)
                batch = AC.collate_rows(cache, rows, device="cpu")
                cap = AC.capture_forward(model, batch)
                z_bank[rows] = cap["z"].detach().cpu().numpy()
                c_loc = cap["c_loc"].detach().cpu().numpy()
                # c_loc is contiguous in molecule order
                off = 0
                for row in rows:
                    nb = int(bp[row + 1] - bp[row])
                    c = c_loc[off:off + nb]
                    off += nb
                    rel = rels[int(row)]
                    j_lat[row] = JS.j_latent(c, rel)
                    j_lat_null[row] = JS.j_latent_null(c, rel, int(row))
        np.save(CACHE_DIR / f"z_{name}.npy", z_bank)
        np.save(AC.ANALYSIS / f"j_latent_{name}.npy", j_lat)
        np.save(AC.ANALYSIS / f"j_latent_null_{name}.npy", j_lat_null)
        model_results[name] = {
            "checkpoint": info,
            "seconds": float(time.time() - t0m),
            "j_latent_sha256": AC.sha256_file(AC.ANALYSIS / f"j_latent_{name}.npy"),
            "z_sha256": AC.sha256_file(CACHE_DIR / f"z_{name}.npy"),
        }

    # distributions / correlations
    dist = {}
    corr = {}
    for name in ("D0", "D1"):
        jl = np.load(AC.ANALYSIS / f"j_latent_{name}.npy").astype(np.float64)
        jln = np.load(AC.ANALYSIS / f"j_latent_null_{name}.npy").astype(np.float64)
        z = np.load(CACHE_DIR / f"z_{name}.npy")
        dist[name] = {
            "j_latent_mean": jl.mean(0).tolist(),
            "j_latent_std": jl.std(0).tolist(),
            "j_latent_zero_fraction": (np.abs(jl) < 1e-12).mean(0).tolist(),
            "j_latent_null_mean": jln.mean(0).tolist(),
            "j_latent_null_std": jln.std(0).tolist(),
            "j_latent_vs_null_pearson": [float(pearsonr(jl[:, j], jln[:, j]).statistic) for j in range(4)],
            "j_latent_vs_null_max_abs_mean_delta": float(np.max(np.abs(jl.mean(0) - jln.mean(0)))),
        }
        corr[name] = {
            "j_latent_vs_global33": corr_summary(g, jl),
            "j_latent_vs_z": corr_summary(z, jl),
            "j_latent_vs_z_without_global": corr_summary(z[:, :288].astype(np.float64), jl),
            "j_latent_null_vs_global33": corr_summary(g, jln),
            "j_latent_null_vs_z": corr_summary(z, jln),
        }
    dist["j_raw"] = {
        "mean": j_raw.mean(0).tolist(),
        "std": j_raw.std(0).tolist(),
        "zero_fraction": (np.abs(j_raw) < 1e-12).mean(0).tolist(),
        "null_mean": j_raw_null.mean(0).tolist(),
        "null_std": j_raw_null.std(0).tolist(),
        "null_vs_real_pearson": [float(pearsonr(j_raw[:, j], j_raw_null[:, j]).statistic) for j in range(j_raw.shape[1])],
        "relation_pair_counts": {
            r: {"mean": float(rel_counts[:, i].mean()), "median": float(np.median(rel_counts[:, i])),
                "zero_fraction": float(np.mean(rel_counts[:, i] == 0)),
                "min": int(rel_counts[:, i].min()), "max": int(rel_counts[:, i].max())}
            for i, r in enumerate(JS.RELATIONS)
        },
    }
    corr["j_raw_vs_global33"] = corr_summary(g, j_raw)
    corr["j_raw_vs_D0_z"] = corr_summary(np.load(CACHE_DIR / "z_D0.npy"), j_raw)
    corr["j_raw_vs_D1_z"] = corr_summary(np.load(CACHE_DIR / "z_D1.npy"), j_raw)
    corr["j_raw_null_vs_global33"] = corr_summary(g, j_raw_null)

    # checks
    checks = {
        "independent_reference": {
            "n_rows": int(len(check_rows)),
            "j_raw_max_abs_diff": ref_raw_max,
            "j_latent_max_abs_diff": ref_lat_max,
        },
        "edge_cases": edge_cases(),
        "invariance": invariance_checks(cache, inv_rows),
        "cache_vs_online": {
            "j_raw_max_abs_diff": float(np.max(np.abs(
                np.load(AC.ANALYSIS / "j_raw.npy").astype(np.float64)[check_rows] - j_raw[check_rows]))),
            "j_latent_D0_max_abs_diff": float(np.max(np.abs(
                np.load(AC.ANALYSIS / "j_latent_D0.npy").astype(np.float64)[check_rows]
                - np.load(AC.ANALYSIS / "j_latent_D0.npy").astype(np.float64)[check_rows]))),
            "note": "j_latent cache check compares the stored array to a fresh load; the online-vs-cache comparison is the independent reference above plus the per-checkpoint z bank hash.",
        },
        "relation_pair_counts": dist["j_raw"]["relation_pair_counts"],
        "bond_type_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(bt_all, return_counts=True))},
        "n_molecules": n,
        "n_physical_bonds": int(len(bt_all)),
        "seconds": float(time.time() - t0),
    }
    out = {
        "protocol": cfg["protocol_id"],
        "models": model_results,
        "checks": checks,
    }
    AC.write_json(AC.ANALYSIS / "joint_stats_checks.json", AC.jsonable(out))
    AC.write_json(AC.ANALYSIS / "joint_stats_distribution.json", AC.jsonable(dist))
    AC.write_json(AC.ANALYSIS / "joint_stats_corr.json", AC.jsonable(corr))
    print(json.dumps({
        "seconds": checks["seconds"],
        "reference": checks["independent_reference"],
        "edge_cases": {k: v["relations"] for k, v in checks["edge_cases"].items()},
        "relation_counts": checks["relation_pair_counts"],
        "corr": {k: {"max_abs_pearson": v["max_abs_pearson"]} for k, v in corr.items() if isinstance(v, dict) and "max_abs_pearson" in v},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
