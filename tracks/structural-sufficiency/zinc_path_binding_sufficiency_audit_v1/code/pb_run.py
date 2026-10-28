"""Orchestrator for the path-binding sufficiency audit (CPU only).

Stages (each writes one JSON under ``analysis/``):

    provenance   → provenance.json
    paths        → paths.json          (raw enumeration vs cache, invariance, bounds)
    math         → math.json           (flat ≡ factorised, MC null, kernel demo)
    binding      → binding.json        (T/T_null/D magnitudes, strata, swap mass)
    recon        → reconstruction.json (S/X/h → T/D, ridge + MLP, permutation control)
    graph        → graphlevel.json     (graph-level collapse / redundancy)
    support      → support.json        (radius-2 seen/low/unseen coverage)
    batch        → batch_invariance.json (historical N mean-field defect)
    decision     → decision.json       (mechanical application of PROTOCOL.md §8)

Usage: ``uv run python code/pb_run.py [--stages paths,math,...]``
"""

from __future__ import annotations

import argparse
import gc
from typing import Any, Mapping

import numpy as np
import torch

import pb_common as P
import pb_paths as PP
import pb_features as F
import pb_recon as R
import pb_support as S
import pb_graphlevel as GL
import pb_batch_invariance as BI

BLOCKS = {
    "T_real": (0, 32),
    "T_null": (32, 64),
    "D": (64, 96),
    "T_type": (96, 128),
    "D_type": (128, 160),
    "h": (160, 224),
}
INPUTS = ("S", "X", "h")
SPLITS = ("train", "valid")


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------

def vec_rms(a: np.ndarray) -> float:
    return float(np.sqrt((np.asarray(a, dtype=np.float64) ** 2).sum(1).mean()))


def block_stats(a: np.ndarray, n_paths: np.ndarray | None = None) -> dict:
    a = np.asarray(a, dtype=np.float64)
    centered = a - a.mean(0, keepdims=True)
    cov = np.cov(centered.T) if a.shape[1] > 1 else np.atleast_2d(centered.var(0))
    out = {
        "rms": vec_rms(a),
        "per_channel_mean": [float(v) for v in a.mean(0)],
        "per_channel_var": [float(v) for v in a.var(0)],
        **R.effective_rank(cov),
    }
    if n_paths is not None:
        out["rms_by_path_count"] = {
            str(int(u)): {"rms": vec_rms(a[n_paths == u]),
                          "n_atoms": int((n_paths == u).sum())}
            for u in np.unique(n_paths) if (n_paths == u).sum() >= 50
        }
    return out


def rho_stats(d: np.ndarray, train_rms: float) -> dict:
    r = np.linalg.norm(np.asarray(d, dtype=np.float64), axis=1)
    rho = r / max(train_rms, 1e-12)
    total = float((r ** 2).sum())
    order = np.argsort(r ** 2)[::-1]
    k = max(1, len(order) // 100)
    return {
        "rho_median": float(np.median(rho)),
        "rho_mean": float(rho.mean()),
        "rho_p01": float(np.percentile(rho, 1)),
        "rho_p10": float(np.percentile(rho, 10)),
        "rho_p25": float(np.percentile(rho, 25)),
        "rho_p75": float(np.percentile(rho, 75)),
        "rho_p90": float(np.percentile(rho, 90)),
        "rho_p99": float(np.percentile(rho, 99)),
        "rho_max": float(rho.max()),
        "frac_atoms_rho_ge_0.05": float((rho >= 0.05).mean()),
        "frac_atoms_rho_ge_0.10": float((rho >= 0.10).mean()),
        "frac_atoms_rho_eq_0": float((rho <= 1e-12).mean()),
        "top1pct_energy_share": float((r[order[:k]] ** 2).sum() / max(total, 1e-12)),
        "rms_norm": float(np.sqrt((r ** 2).mean())),
    }


def molecule_rho(d: np.ndarray, atom_mol: np.ndarray, n_mols: int, train_rms: float) -> dict:
    r = np.linalg.norm(np.asarray(d, dtype=np.float64), axis=1) / max(train_rms, 1e-12)
    med = np.zeros(n_mols)
    mx = np.zeros(n_mols)
    for m in range(n_mols):
        sel = atom_mol == m
        if not bool(sel.any()):
            continue
        med[m] = np.median(r[sel])
        mx[m] = r[sel].max()
    return {
        "median_mol_rho_median": float(np.median(med)),
        "median_mol_rho_p10": float(np.percentile(med, 10)),
        "frac_molecules_median_rho_ge_0.10": float((med >= 0.10).mean()),
        "frac_molecules_median_rho_ge_0.05": float((med >= 0.05).mean()),
        "frac_molecules_any_rho_ge_0.05": float((mx >= 0.05).mean()),
        "frac_molecules_max_rho_ge_0.5": float((mx >= 0.5).mean()),
    }


def strata_report(rho_vec: np.ndarray, energy: np.ndarray, atom_scalar: np.ndarray,
                  n_paths: np.ndarray) -> dict:
    degree = np.rint(np.asarray(atom_scalar[:, 3], dtype=np.float64) * 4.0).astype(int)
    in_ring = np.asarray(atom_scalar[:, 4], dtype=np.float64) > 0.5
    terminal = np.asarray(atom_scalar[:, 7], dtype=np.float64) > 0.5
    fused = np.asarray(atom_scalar[:, 8], dtype=np.float64) > 0.5
    closure = np.asarray(atom_scalar[:, 2], dtype=np.float64)
    total = float(energy.sum())

    def group(mask):
        if not bool(mask.any()):
            return {"n_atoms": 0}
        return {"n_atoms": int(mask.sum()), "frac_atoms": float(mask.mean()),
                "rho_median": float(np.median(rho_vec[mask])),
                "rho_p10": float(np.percentile(rho_vec[mask], 10)),
                "energy_share": float(energy[mask].sum() / max(total, 1e-12))}

    return {
        "by_degree": {str(d): group(degree == d) for d in sorted(set(degree.tolist()))
                      if (degree == d).sum() >= 50},
        "by_path_count": {str(k): group(n_paths == k) for k in sorted(set(n_paths.tolist()))
                          if (n_paths == k).sum() >= 50},
        "by_ring_atom": {"ring": group(in_ring), "chain": group(~in_ring)},
        "by_terminal": {"terminal": group(terminal), "non_terminal": group(~terminal)},
        "by_fused": {"fused": group(fused), "non_fused": group(~fused)},
        "by_closure": {"closure0": group(closure < 1e-9),
                       "closure_mid": group((closure > 1e-9) & (closure < 0.5)),
                       "closure_high": group(closure >= 0.5)},
    }


def gate_sum(model, cats: np.ndarray, contents: np.ndarray) -> np.ndarray:
    """``sum_p gate[cat_p] * a_{u_p}`` with the frozen gate table (float64)."""
    with torch.no_grad():
        gate = model.local.path_gate(torch.from_numpy(np.asarray(cats, dtype=np.int64))).numpy()
    return (gate.astype(np.float64) * np.asarray(contents, dtype=np.float64)).sum(0)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

class Audit:
    def __init__(self):
        self.plot = P.Plot()
        self.caches = {s: P.load_cache(s) for s in SPLITS}
        self.norm = P.load_norm()
        self.atom_mol = {s: F.atom_molecule_index(self.caches[s]) for s in SPLITS}
        self.n_mols = {s: int(self.caches[s]["n_molecules"]) for s in SPLITS}
        self.n_atoms = {s: int(np.asarray(self.caches[s]["atom_ptr"])[-1]) for s in SPLITS}
        self.paths: dict[str, dict] = {}
        self.enc: dict[str, dict] = {}
        self.models: dict[str, Any] = {}
        self.out: dict[str, dict] = {}

    # -- 0 ---------------------------------------------------------------
    def stage_provenance(self):
        cfg = P.read_json(P.LCRC / "configs" / "lcrc_v1.json")
        prov = {
            "round": P.ROUND_ID,
            "feature_version": P.FEATURE_VERSION,
            "device": "cpu",
            "gpu_hours": 0.0,
            "git": P.git_state(),
            "env": P.env_state(),
            "protocol_sha256": P.sha256_file(P.AUDIT_DIR / "PROTOCOL.md"),
            "data": {},
            "official_test_loaded": False,
            "labels_used": False,
            "normalisation": {"fingerprint": self.norm["_fingerprint"],
                              "source": self.norm["source"]},
            "checkpoints": {},
            "projections": {str(s): P.proj_fingerprint(P.projections(s)) for s in P.PROJ_SEEDS},
            "projection_definition": "Normal(0, 1/d_in), numpy default_rng(seed), float64; "
                                     "Q/K over [a_v; b_e1] (192 -> 32), V over a_u (96 -> 32)",
            "lrc_v1_config_reference": {
                "train_content_sha256": cfg["data"]["train_cache"]["content_sha256"],
                "valid_content_sha256": cfg["data"]["valid_cache"]["content_sha256"],
            },
        }
        for split in SPLITS:
            c = self.caches[split]
            prov["data"][split] = {
                "path": c["_path"],
                "sha256": c["_sha256"],
                "matches_frozen_config": c["_sha256"] == P.EXPECTED_CACHE_SHA[split],
                "content_sha256": P.D.content_fingerprint(c),
                "n_molecules": self.n_mols[split],
                "n_atoms": self.n_atoms[split],
                "n_bonds": int(np.asarray(c["bond_ptr"])[-1]),
                "n_paths": int(len(np.asarray(c["path_cat"]))),
                "labels_object": type(c["labels"]).__name__,
                "meta_feature_version": c["meta"]["feature_version"],
                "object_source": c["meta"]["object_source"],
            }
        for name in P.CHECKPOINTS:
            model, info = P.load_frozen_a(name)
            self.models[name] = model
            prov["checkpoints"][name] = info
        for split in SPLITS:
            self.paths[split] = PP.split_paths(self.caches[split])
        for ckpt in P.CHECKPOINTS:
            self.enc[ckpt] = F.encode(self.models[ckpt], self.caches["train"])
            self.enc[f"{ckpt}|valid"] = F.encode(self.models[ckpt], self.caches["valid"])
        P.write_json(P.ANALYSIS / "provenance.json", prov)
        self.plot.mark("provenance+encode")
        return prov

    def _ensure_encoded(self):
        for ckpt in P.CHECKPOINTS:
            if ckpt not in self.enc:
                model, _ = P.load_frozen_a(ckpt)
                self.models[ckpt] = model
                self.enc[ckpt] = F.encode(model, self.caches["train"])
                self.enc[f"{ckpt}|valid"] = F.encode(model, self.caches["valid"])
        for split in SPLITS:
            if split not in self.paths:
                self.paths[split] = PP.split_paths(self.caches[split])

    # -- 1 ---------------------------------------------------------------
    def stage_paths(self):
        self._ensure_encoded()
        out = {"definition": "ordered non-backtracking 2-paths v->w->u, u != v, carrying "
                             "first/second bond ids; closure decided on the raw adjacency",
               "elementwise_vs_frozen_cache": {}}
        for split in SPLITS:
            res = PP.crosscheck_split(self.caches[split], self.paths[split])
            out["elementwise_vs_frozen_cache"][split] = res
            if not res["passed"]:
                raise RuntimeError(f"{split}: raw path enumeration disagrees with the frozen cache")

        model = self.models["lcrcA_s0"]
        proj = P.projections(P.PRIMARY_SEED)
        va = self.caches["valid"]

        def t_of(cache):
            enc = F.encode(model, cache)
            paths = PP.split_paths(cache)
            q, k, z = F.path_factors(enc["a"], enc["b"], paths, proj)
            return enc, F.t_stats(paths, q, k, z, enc["n_atoms"])["T_real"]

        # (a) atom relabeling on 8 real valid molecules
        rng = np.random.default_rng(20261030)
        relabel, reverse = [], []
        for m in range(8):
            a0, a1 = int(va["atom_ptr"][m]), int(va["atom_ptr"][m + 1])
            b0, b1 = int(va["bond_ptr"][m]), int(va["bond_ptr"][m + 1])
            types = np.asarray(va["atom_type"][a0:a1], dtype=np.int64)
            bonds = [(int(u), int(v), int(bt)) for (u, v), bt in
                     zip(np.asarray(va["bond_index"][b0:b1]), np.asarray(va["bond_type"][b0:b1]))]
            perm = rng.permutation(len(types))
            inv_types = np.zeros_like(types)
            inv_types[perm] = types
            relabelled = PP.synthetic_cache([(inv_types, [(int(perm[u]), int(perm[v]), bt)
                                                          for u, v, bt in bonds])])
            _, t_ref = t_of(PP.synthetic_cache([(types, bonds)]))
            _, t_rel = t_of(relabelled)
            relabel.append(float(np.abs(np.sort(t_rel, axis=0) - np.sort(t_ref, axis=0)).max()))
            reversed_bonds = PP.synthetic_cache([(types, [(v, u, bt) for u, v, bt in bonds])])
            _, t_rev = t_of(reversed_bonds)
            reverse.append(float(np.abs(t_rev - t_ref).max()))
        # (b) batch size / order: same split encoded at two batch sizes
        enc_big = F.encode(model, va, batch_size=256)
        enc_small = F.encode(model, va, batch_size=7)
        batch_diff = {k: float(np.abs(enc_big[k] - enc_small[k]).max())
                      for k in ("a", "b", "h", "x", "s")}
        paths_va = self.paths["valid"]
        t_big = F.t_stats(paths_va, *F.path_factors(enc_big["a"], enc_big["b"], paths_va, proj),
                          enc_big["n_atoms"])["T_real"]
        t_small = F.t_stats(paths_va, *F.path_factors(enc_small["a"], enc_small["b"], paths_va, proj),
                            enc_small["n_atoms"])["T_real"]
        batch_diff["T_real"] = float(np.abs(t_big - t_small).max())
        # (c) boundary graphs
        syn = PP.synthetic_cache([
            ([0, 0], [(0, 1, 1)]),
            ([0, 0, 0], [(0, 1, 1), (1, 2, 1)]),
            ([0, 0, 0, 0, 0, 0], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 0, 1)]),
            ([0, 0, 0, 0, 0, 0], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)]),
        ])
        enc_syn, t_syn = t_of(syn)
        p_syn = PP.split_paths(syn)
        n_syn = np.bincount(p_syn["root"], minlength=enc_syn["n_atoms"])
        st_syn = F.t_stats(p_syn, *F.path_factors(enc_syn["a"], enc_syn["b"], p_syn, proj),
                           enc_syn["n_atoms"])
        out["invariance"] = {
            "atom_relabel_max_abs_T_diff": float(max(relabel)),
            "bond_direction_max_abs_T_diff": float(max(reverse)),
            "batch_size_256_vs_7_max_abs": batch_diff,
            "n_molecules_relabel": 8,
        }
        out["boundary"] = {
            "graphs": ["2-atom", "3-atom chain", "6-ring", "6-chain"],
            "path_counts_per_molecule": np.diff(p_syn["mol_ptr"]).tolist(),
            "zero_path_atoms": int((n_syn == 0).sum()),
            "single_path_atoms": int((n_syn == 1).sum()),
            "all_finite": bool(np.isfinite(st_syn["T_real"]).all()
                               and np.isfinite(st_syn["D"]).all()),
            "zero_path_T_is_zero": bool(np.allclose(st_syn["T_real"][n_syn == 0], 0.0)),
            "single_path_D_is_zero": bool(np.allclose(st_syn["D"][n_syn == 1], 0.0)),
            "synthetic_crosscheck": PP.crosscheck_split(syn, p_syn),
        }
        passed = (out["elementwise_vs_frozen_cache"]["train"]["passed"]
                  and out["elementwise_vs_frozen_cache"]["valid"]["passed"]
                  and max(relabel) < 1e-5 and max(reverse) < 1e-5
                  and out["boundary"]["all_finite"]
                  and out["boundary"]["single_path_D_is_zero"])
        out["passed"] = bool(passed)
        self.out["paths"] = out
        P.write_json(P.ANALYSIS / "paths.json", out)
        self.plot.mark("paths")

    # -- 2 ---------------------------------------------------------------
    def stage_math(self):
        self._ensure_encoded()
        va = self.caches["valid"]
        paths = self.paths["valid"]
        model = self.models["lcrcA_s0"]
        enc = self.enc["lcrcA_s0|valid"] if "lcrcA_s0|valid" in self.enc else \
            F.encode(model, va)
        proj = P.projections(P.PRIMARY_SEED)
        sub = {k: paths[k][:8000] for k in ("root", "mid", "end", "e1", "e2")}
        n_atoms = enc["n_atoms"]
        q, k, z = F.path_factors(enc["a"], enc["b"], sub, proj)
        t_stats = F.t_stats(sub, q, k, z, n_atoms)["T_real"]
        t_flat = F.flat_t_reference(enc["a"], enc["b"], sub, proj, n_atoms)
        t_fact = F.factorized_t_reference(enc["a"], enc["b"], sub, proj, n_atoms)
        eq = {
            "n_paths": int(len(sub["root"])),
            "flat_vs_factorized_max_abs": float(np.abs(t_flat - t_fact).max()),
            "stats_vs_flat_max_abs": float(np.abs(t_stats - t_flat).max()),
            "flat_rms": vec_rms(t_flat),
        }
        # Monte-Carlo: independent per-path q,k,z ⇒ E[T_real] = E[T_null]
        rng = np.random.default_rng(20261030)
        mc = {"draws": 4000, "per_n": {}}
        for n in (1, 2, 3, 5, 8, 13):
            d_sum = np.zeros(32)
            t_rms, tn_rms = [], []
            for _ in range(mc["draws"]):
                qq, kk, zz = (rng.normal(size=(n, 32)) for _ in range(3))
                tr = (qq * kk * zz).sum(0) / np.sqrt(n)
                tn = np.sqrt(n) * qq.mean(0) * kk.mean(0) * zz.mean(0)
                d_sum += tr - tn
                t_rms.append((tr ** 2).sum())
                tn_rms.append((tn ** 2).sum())
            sd = float(np.sqrt((d_sum ** 2).mean() / mc["draws"]))
            mc["per_n"][str(n)] = {
                "mean_D_over_draws_max_abs_z": float(np.abs(d_sum / mc["draws"]).max()
                                                     / max(sd, 1e-12)),
                "rms_T_real": float(np.sqrt(np.mean(t_rms))),
                "rms_T_null": float(np.sqrt(np.mean(tn_rms))),
            }
        # kernel demo: within-cell content re-assignment on a real root
        qf, kf, zf = F.path_factors(enc["a"], enc["b"], paths, proj)
        cat = np.asarray(va["path_cat"], dtype=np.int64)
        end63 = np.asarray(va["path_end63"], dtype=np.int64)
        clos = np.asarray(va["path_closure"], dtype=np.int64)
        cell = ((paths["root"].astype(np.int64) * 189 + cat) * 63 + end63) * 2 + clos
        uniq, inverse, counts = np.unique(cell, return_inverse=True, return_counts=True)
        demo = {}
        for ci in np.argsort(counts)[::-1]:
            members = np.flatnonzero(inverse == ci)
            root = int(paths["root"][members[0]])
            others = np.flatnonzero(paths["root"] == root)
            if len(members) < 2 or len(others) > 12:
                continue
            ends = paths["end"]
            pos = {int(p): i for i, p in enumerate(others)}
            local_members = [pos[int(p)] for p in members]
            perm_ends = ends[others].copy()
            perm_ends[local_members] = ends[members][::-1]
            content_real = gate_sum(model, cat[others], enc["a"][ends[others]])
            content_perm = gate_sum(model, cat[others], enc["a"][perm_ends])
            kk = kf[others].copy()
            zz = zf[others].copy()
            for src, dst in zip(members, members[::-1]):
                kk[pos[int(dst)]] = kf[int(src)]
                zz[pos[int(dst)]] = zf[int(src)]
            t_new = (qf[others] * kk * zz).sum(0) / np.sqrt(len(others))
            t_old = (qf[others] * kf[others] * zf[others]).sum(0) / np.sqrt(len(others))
            demo = {
                "root_atom": root,
                "n_paths": int(len(others)),
                "cell_multiplicity": int(counts[ci]),
                "cell": {"cat189": int(cat[members[0]]), "end63": int(end63[members[0]]),
                         "closure": int(clos[members[0]])},
                "content_agg_max_abs_diff_after_reassignment": float(
                    np.abs(content_real - content_perm).max()),
                "T_max_abs_diff_after_reassignment": float(np.abs(t_new - t_old).max()),
                "T_rel_diff": float(np.abs(t_new - t_old).max()
                                    / max(np.abs(t_old).max(), 1e-12)),
            }
            break
        # path-order permutation invariance of T
        perm = rng.permutation(len(sub["root"]))
        n = np.bincount(sub["root"][perm], minlength=n_atoms).astype(np.float64)
        t_perm = P.seg_sum(q[perm] * k[perm] * z[perm], sub["root"][perm], n_atoms) \
            / np.sqrt(np.maximum(n, 1))[:, None]
        self.out["math"] = {
            "flat_factorized": eq,
            "monte_carlo_independent_paths": mc,
            "kernel_demo_within_cell_reassignment": demo,
            "path_order_permutation_max_abs_diff": float(np.abs(t_perm - t_stats).max()),
            "statements": {
                "S1": "T_real is a statistic of the joint rooted path multiset (Q,K,Z are "
                      "per-path functions of (a_v,b_e1,a_w,b_e2,a_u)); no information beyond "
                      "the graph + frozen encoders is created.",
                "S2": "sum_paths(q*k*z) ≡ sum_w q(v,w) ⊙ sum_u(k(w,u)*z(u)): flat and "
                      "branch-first forms are the same function (float64 roundoff only).",
                "S3": "content_agg is a rank-≤96 linear map of the (path-cat × endpoint-content) "
                      "matrix, so within-cell re-pairings are invisible to X_v by construction; "
                      "the demo realises one with T_max_abs_diff > 0.",
                "S4": "X_v is therefore a non-injective compression of the path-content "
                      "configuration. No claim is made that the same kernel is realised by two "
                      "distinct ZINC molecules; realised irrecoverability is measured empirically.",
            },
        }
        P.write_json(P.ANALYSIS / "math.json", self.out["math"])
        self.plot.mark("math")

    # -- 3 ---------------------------------------------------------------
    def stage_binding(self):
        self._ensure_encoded()
        train = self.caches["train"]
        train_rows = np.arange(self.n_mols["train"])
        out = {"type_content_control": {}, "per_checkpoint_seed": {}}
        for ckpt in P.CHECKPOINTS:
            enc_tr = self.enc[ckpt]
            tabs = F.type_content_tables(enc_tr["a"], enc_tr["b"], train, train_rows)
            a_type, b_type = F.type_content_arrays(tabs)
            out["type_content_control"][ckpt] = {
                "definition": "train-only means: ā[atom type], b̄[unordered (type,type,bond)]",
                "n_missing_bond_keys": tabs["n_missing_bond_keys"],
                "n_bond_keys": tabs["n_bond_keys"],
                "atom_type_count_min": int(tabs["atom_cnt"].min()),
                "atom_type_count_max": int(tabs["atom_cnt"].max()),
            }
            for seed in P.PROJ_SEEDS:
                proj = P.projections(seed)
                key = f"{ckpt}|p{seed}"
                entry: dict[str, Any] = {}
                train_rms = None
                for split in SPLITS:
                    cache = self.caches[split]
                    enc_s = self.enc[ckpt] if split == "train" else self.enc[f"{ckpt}|valid"]
                    paths = self.paths[split]
                    q, k, z = F.path_factors(enc_s["a"], enc_s["b"], paths, proj)
                    st = F.t_stats(paths, q, k, z, enc_s["n_atoms"],
                                   rng=np.random.default_rng(20261030))
                    sw = F.t_swap_within_cell(paths, cache, q, k, z, enc_s["n_atoms"],
                                              seed=20261031)
                    qt, kt, zt = F.path_factors(a_type, b_type, paths, proj)
                    stt = F.t_stats(paths, qt, kt, zt, enc_s["n_atoms"])
                    u = st["T_real"] - sw["T_swap"]
                    rms_t = vec_rms(st["T_real"])
                    if split == "train":
                        train_rms = rms_t
                    scale = train_rms
                    blocks = {name: block_stats(st[name], st["n_paths"])
                              for name in ("T_real", "T_null", "D", "T_perm")}
                    blocks["T_type"] = block_stats(stt["T_real"], stt["n_paths"])
                    blocks["D_type"] = block_stats(stt["D"], stt["n_paths"])
                    blocks["U"] = block_stats(u)
                    energy = np.linalg.norm(st["D"], axis=1) ** 2
                    entry[split] = {
                        "rms_T_real": rms_t,
                        "rms_T_null": vec_rms(st["T_null"]),
                        "rms_D": vec_rms(st["D"]),
                        "rms_T_perm": vec_rms(st["T_perm"]),
                        "rms_D_perm": vec_rms(st["D_perm"]),
                        "rms_T_type": vec_rms(stt["T_real"]),
                        "rms_D_type": vec_rms(stt["D"]),
                        "rms_U_within_cell": vec_rms(u),
                        "rms_U_over_rms_T": vec_rms(u) / max(rms_t, 1e-12),
                        "rms_D_type_over_rms_D": vec_rms(stt["D"]) / max(vec_rms(st["D"]), 1e-12),
                        "scale_train_rms_T": scale,
                        "n_paths": {"mean": float(st["n_paths"].mean()),
                                    "median": float(np.median(st["n_paths"])),
                                    "min": int(st["n_paths"].min()),
                                    "max": int(st["n_paths"].max()),
                                    "frac_one": float((st["n_paths"] == 1).mean()),
                                    "frac_two": float((st["n_paths"] == 2).mean())},
                        "blocks": blocks,
                        "rho": rho_stats(st["D"], scale),
                        # frozen P5 form: the type-only residual measured against the REAL
                        # binding scale (absolute, content-scale confounded)
                        "rho_type": rho_stats(stt["D"], scale),
                        # same control measured against its OWN train scale (unit-free)
                        "rho_type_own": rho_stats(stt["D"], vec_rms(stt["T_real"])),
                        "molecules": molecule_rho(st["D"], self.atom_mol[split],
                                                  self.n_mols[split], scale),
                        "molecules_type": molecule_rho(stt["D"], self.atom_mol[split],
                                                       self.n_mols[split], scale),
                        "molecules_type_own": molecule_rho(
                            stt["D"], self.atom_mol[split], self.n_mols[split],
                            vec_rms(stt["T_real"])),
                        "swap_cells": {
                            "n_cells": sw["n_cells"],
                            "n_multi_cells": sw["n_multi_cells"],
                            "max_cell_multiplicity": sw["max_cell_multiplicity"],
                            "frac_atoms_with_multi_cell": float((sw["cell_multiplicity"] > 0).mean()),
                            "frac_atoms_U_nonzero": float((np.abs(u).max(1) > 1e-9).mean()),
                        },
                        "strata": strata_report(
                            np.linalg.norm(st["D"], axis=1) / max(scale, 1e-12), energy,
                            np.asarray(cache["atom_scalar"], dtype=np.float64), st["n_paths"]),
                    }
                out["per_checkpoint_seed"][key] = entry
                self.plot.mark(f"binding {key}")
        self.out["binding"] = out
        P.write_json(P.ANALYSIS / "binding.json", out)

    # -- 4 ---------------------------------------------------------------
    def stage_recon(self):
        self._ensure_encoded()
        train, valid = self.caches["train"], self.caches["valid"]
        mon_mols = np.random.RandomState(P.MONITOR_SEED).permutation(
            self.n_mols["train"])[: P.MONITOR_MOLS]
        fit_mask = ~np.isin(self.atom_mol["train"], mon_mols)
        mon_mask = ~fit_mask
        n_train_atoms = self.n_atoms["train"]
        out = {"protocol": {
            "monitor_seed": P.MONITOR_SEED, "monitor_molecules": P.MONITOR_MOLS,
            "perm_seed": P.PERM_SEED, "mlp_seed": P.MLP_SEED,
            "lambda_grid": list(R.LAMBDA_GRID),
            "inputs": {"S": "existing S1/S2 statistics + local scalars (433)",
                       "X": "full LocalContent.mlp input (529)",
                       "h": "trained LCRC-A local output (64)"},
            "targets": {k: list(v) for k, v in BLOCKS.items()},
            "note": "ridge λ and MLP stopping chosen on a train-internal molecule monitor; "
                    "valid scored once per fit; targets standardised with fit-subset stats",
        }, "runs": {}, "permutation_controls": {}}
        for ckpt in P.CHECKPOINTS:
            enc_tr, enc_va = self.enc[ckpt], self.enc[f"{ckpt}|valid"]
            tabs = F.type_content_tables(enc_tr["a"], enc_tr["b"], train,
                                         np.arange(self.n_mols["train"]))
            a_type, b_type = F.type_content_arrays(tabs)
            for seed in P.PROJ_SEEDS:
                proj = P.projections(seed)
                key = f"{ckpt}|p{seed}"
                Y = {}
                for tag, enc_s, paths in (("tr", enc_tr, self.paths["train"]),
                                          ("va", enc_va, self.paths["valid"])):
                    q, k, z = F.path_factors(enc_s["a"], enc_s["b"], paths, proj)
                    st = F.t_stats(paths, q, k, z, enc_s["n_atoms"])
                    qt, kt, zt = F.path_factors(a_type, b_type, paths, proj)
                    stt = F.t_stats(paths, qt, kt, zt, enc_s["n_atoms"])
                    Y[tag] = np.concatenate([st["T_real"], st["T_null"], st["D"],
                                             stt["T_real"], stt["D"], enc_s["h"]],
                                            axis=1).astype(np.float32)
                std_y = R.Standardizer.fit(Y["tr"][fit_mask])
                Ytr_s = std_y.transform(Y["tr"])
                Yva_s = std_y.transform(Y["va"])
                deg, near = R.degenerate_channels(std_y.raw_scale)
                out["runs"][key] = {
                    "n_fit_atoms": int(fit_mask.sum()),
                    "n_monitor_atoms": int(mon_mask.sum()),
                    "n_valid_atoms": int(len(Yva_s)),
                    "degenerate_channels": {k: [int(i) for i in np.flatnonzero(deg[a:b])]
                                            for k, (a, b) in BLOCKS.items()},
                    "near_constant_channels": {k: [int(i) for i in np.flatnonzero(near[a:b])]
                                               for k, (a, b) in BLOCKS.items()},
                    "valid_target_std": {k: float(np.sqrt((Yva_s[:, a:b] ** 2).mean()))
                                         for k, (a, b) in BLOCKS.items()},
                }
                for iname in INPUTS:
                    Xtr = enc_tr["s"] if iname == "S" else enc_tr[iname.lower()]
                    Xva = enc_va["s"] if iname == "S" else enc_va[iname.lower()]
                    std_x = R.Standardizer.fit(Xtr[fit_mask])
                    Xtr_s = std_x.transform(Xtr)
                    Xva_s = std_x.transform(Xva)
                    entry = {"ridge": {}, "mlp": {}}
                    pred_r, info_r = R.fit_predict_ridge(
                        Xtr_s[fit_mask], Ytr_s[fit_mask], Xtr_s[mon_mask],
                        Ytr_s[mon_mask], Xva_s)
                    entry["ridge"] = {"info": info_r,
                                      "blocks": R.block_metrics(pred_r, Yva_s, deg | near,
                                                                BLOCKS, std_y.raw_scale)}
                    if iname == "h":
                        entry["mlp"] = {"skipped": "identity target: h -> h"}
                    else:
                        model, info_m = R.mlp_fit(
                            Xtr_s[fit_mask], Ytr_s[fit_mask], Xtr_s[mon_mask],
                            Ytr_s[mon_mask], seed=P.MLP_SEED, max_epochs=60, patience=8)
                        pred_m = R.mlp_predict(model, Xva_s)
                        entry["mlp"] = {
                            "info": {k: v for k, v in info_m.items() if k != "curve"},
                            "curve_sampled": info_m["curve"][::5],
                            "blocks": R.block_metrics(pred_m, Yva_s, deg | near, BLOCKS,
                                                      std_y.raw_scale)}
                        del model
                        gc.collect()
                    out["runs"][key][iname] = entry
                    self.plot.mark(f"recon {key} {iname}")

                # permutation control: break the X↔Y row correspondence globally
                perm = np.random.RandomState(P.PERM_SEED).permutation(n_train_atoms)
                Ytr_perm = Ytr_s[perm]
                pentry = {}
                for iname in INPUTS:
                    Xtr = enc_tr["s"] if iname == "S" else enc_tr[iname.lower()]
                    Xva = enc_va["s"] if iname == "S" else enc_va[iname.lower()]
                    std_x = R.Standardizer.fit(Xtr[fit_mask])
                    Xtr_s = std_x.transform(Xtr)
                    Xva_s = std_x.transform(Xva)
                    pred_r, info_r = R.fit_predict_ridge(
                        Xtr_s[fit_mask], Ytr_perm[fit_mask], Xtr_s[mon_mask],
                        Ytr_perm[mon_mask], Xva_s)
                    pentry[iname] = {"ridge": {
                        "info": info_r,
                        "blocks": R.block_metrics(pred_r, Yva_s, deg | near, BLOCKS,
                                                  std_y.raw_scale)}}
                    if iname != "h" and seed == P.PRIMARY_SEED:
                        model, info_m = R.mlp_fit(
                            Xtr_s[fit_mask], Ytr_perm[fit_mask], Xtr_s[mon_mask],
                            Ytr_perm[mon_mask], seed=P.MLP_SEED, max_epochs=25, patience=6)
                        pentry[iname]["mlp"] = {
                            "info": {k: v for k, v in info_m.items() if k != "curve"},
                            "blocks": R.block_metrics(R.mlp_predict(model, Xva_s), Yva_s,
                                                      deg | near, BLOCKS, std_y.raw_scale)}
                        del model
                        gc.collect()
                out["permutation_controls"][key] = pentry
                self.plot.mark(f"recon perm {key}")
                gc.collect()
        self.out["recon"] = out
        P.write_json(P.ANALYSIS / "reconstruction.json", out)

    # -- 5 ---------------------------------------------------------------
    def stage_graph(self):
        self._ensure_encoded()
        train, valid = self.caches["train"], self.caches["valid"]
        out = {"blocks": {k: list(v) for k, v in BLOCKS.items()}, "runs": {}}
        for ckpt in P.CHECKPOINTS:
            enc_tr, enc_va = self.enc[ckpt], self.enc[f"{ckpt}|valid"]
            E_tr = GL.existing_graph_features(train, enc_tr, self.norm,
                                              self.atom_mol["train"], self.n_mols["train"])
            E_va = GL.existing_graph_features(valid, enc_va, self.norm,
                                              self.atom_mol["valid"], self.n_mols["valid"])
            tabs = F.type_content_tables(enc_tr["a"], enc_tr["b"], train,
                                         np.arange(self.n_mols["train"]))
            a_type, b_type = F.type_content_arrays(tabs)
            for seed in P.PROJ_SEEDS:
                proj = P.projections(seed)
                key = f"{ckpt}|p{seed}"
                parts = {}
                for tag, cache, enc_s, atom_mol, n_mols in (
                        ("tr", train, enc_tr, self.atom_mol["train"], self.n_mols["train"]),
                        ("va", valid, enc_va, self.atom_mol["valid"], self.n_mols["valid"])):
                    paths = self.paths["train" if tag == "tr" else "valid"]
                    q, k, z = F.path_factors(enc_s["a"], enc_s["b"], paths, proj)
                    st = F.t_stats(paths, q, k, z, enc_s["n_atoms"])
                    qt, kt, zt = F.path_factors(a_type, b_type, paths, proj)
                    stt = F.t_stats(paths, qt, kt, zt, enc_s["n_atoms"])
                    feats = GL.new_graph_features(
                        {"T_real": st["T_real"], "D": st["D"], "D_type": stt["D"]},
                        atom_mol, n_mols)
                    parts[tag] = {"G": np.concatenate([feats["T_real"], feats["D"],
                                                       feats["D_type"]], axis=1),
                                  "E": E_tr if tag == "tr" else E_va}
                blocks_g = {"G_T_real": (0, 96), "G_D": (96, 192), "G_D_type": (192, 288)}
                fwd = GL.reconstruct_graphs(parts["tr"]["E"], parts["tr"]["G"],
                                            parts["va"]["E"], parts["va"]["G"],
                                            blocks=blocks_g)
                rev = GL.reconstruct_graphs(parts["tr"]["G"], parts["tr"]["E"],
                                            parts["va"]["G"], parts["va"]["E"],
                                            blocks={"E": (0, parts["tr"]["E"].shape[1])})
                for direction in (fwd, rev):
                    for kind in ("ridge", "mlp"):
                        for blk in direction[kind]["blocks"].values():
                            blk.pop("per_channel_r2", None)
                out["runs"][key] = {"E_dim": int(parts["tr"]["E"].shape[1]),
                                    "G_dim": int(parts["tr"]["G"].shape[1]),
                                    "E_to_G": fwd, "G_to_E": rev}
                self.plot.mark(f"graph {key}")
        self.out["graph"] = out
        P.write_json(P.ANALYSIS / "graphlevel.json", out)

    # -- 6 ---------------------------------------------------------------
    def stage_support(self):
        self._ensure_encoded()
        keys = S.build_keys(self.caches)
        sup = S.support_tables(keys)
        vinfo = S.valid_atom_support(keys, sup["sup2"])
        ref = P.read_json(P.REPO / "tracks/structural-sufficiency/"
                                 "zinc_composition_transfer_v1/cache/keys_meta.json")
        out = {
            "definition": "radius-2 rooted key, recursion depth 2 over (type, sorted (bond,type)); "
                          "support = number of official-train molecules containing the key",
            "vocabulary": {"n_key1": int(keys["n_key1"]), "n_key2": int(keys["n_key2"]),
                           "reference_n_key1": int(ref["n_key1"]),
                           "reference_n_key2": int(ref["n_key2"]),
                           "matches_existing_definition": bool(
                               int(keys["n_key1"]) == int(ref["n_key1"])
                               and int(keys["n_key2"]) == int(ref["n_key2"]))},
            "support_thresholds": {"unseen": 0, "low": f"1..{P.SUP_LOW_MAX}",
                                   "seen": f">={P.SUP_LOW_MAX + 1}"},
            "valid_atom_counts": vinfo["counts"],
            "valid_atom_fractions": vinfo["fractions"],
            "support_quantiles": vinfo["support_quantiles"],
            "molecule_has_unseen": {"n": int(vinfo["molecule_any_unseen"].sum()),
                                    "fraction": float(vinfo["molecule_any_unseen"].mean())},
            "groups": {},
        }
        group = vinfo["atom_group"]
        for gi, gname in enumerate(S.GROUP_NAMES):
            out["groups"][gname] = {"n_atoms": int((group == gi).sum())}
        mon_mols = np.random.RandomState(P.MONITOR_SEED).permutation(
            self.n_mols["train"])[: P.MONITOR_MOLS]
        fit_mask = ~np.isin(self.atom_mol["train"], mon_mols)
        proj = P.projections(P.PRIMARY_SEED)
        for ckpt in P.CHECKPOINTS:
            enc_tr, enc_va = self.enc[ckpt], self.enc[f"{ckpt}|valid"]
            tabs = F.type_content_tables(enc_tr["a"], enc_tr["b"], self.caches["train"],
                                         np.arange(self.n_mols["train"]))
            a_type, b_type = F.type_content_arrays(tabs)
            Y = {}
            for tag, enc_s, paths in (("tr", enc_tr, self.paths["train"]),
                                      ("va", enc_va, self.paths["valid"])):
                q, k, z = F.path_factors(enc_s["a"], enc_s["b"], paths, proj)
                Y[tag] = F.t_stats(paths, q, k, z, enc_s["n_atoms"])
            std_x = R.Standardizer.fit(enc_tr["x"][fit_mask])
            Ytr_s = np.concatenate([Y["tr"]["T_real"], Y["tr"]["D"]], axis=1)
            Yva_s = np.concatenate([Y["va"]["T_real"], Y["va"]["D"]], axis=1)
            std_y = R.Standardizer.fit(Ytr_s[fit_mask])
            Xtr = std_x.transform(enc_tr["x"])
            Xva = std_x.transform(enc_va["x"])
            pred, _ = R.fit_predict_ridge(Xtr[fit_mask], std_y.transform(Ytr_s[fit_mask]),
                                         Xtr[~fit_mask], std_y.transform(Ytr_s[~fit_mask]), Xva)
            va_std = std_y.transform(Yva_s)
            rms_t = vec_rms(Y["va"]["T_real"])
            for gi, gname in enumerate(S.GROUP_NAMES):
                sel = group == gi
                if not bool(sel.any()):
                    continue
                rho = np.linalg.norm(Y["va"]["D"][sel], axis=1) / max(rms_t, 1e-12)
                resid = pred[sel] - va_std[sel]
                out["groups"][gname][ckpt] = {
                    "n_atoms": int(sel.sum()),
                    "rho_median": float(np.median(rho)),
                    "rho_p10": float(np.percentile(rho, 10)),
                    "rms_D": vec_rms(Y["va"]["D"][sel]),
                    "rms_D_over_rms_D_all": vec_rms(Y["va"]["D"][sel])
                                            / max(vec_rms(Y["va"]["D"]), 1e-12),
                    "ridge_r2_D": float(1.0 - ((resid[:, 32:64]) ** 2).sum()
                                        / max(((va_std[sel][:, 32:64]) ** 2).sum(), 1e-12)),
                    "ridge_r2_T_real": float(1.0 - ((resid[:, 0:32]) ** 2).sum()
                                             / max(((va_std[sel][:, 0:32]) ** 2).sum(), 1e-12)),
                }
            # molecule-level: has unseen atom vs not
            mol_flag = vinfo["molecule_any_unseen"]
            mol_of_atom = self.atom_mol["valid"]
            for name, flag in (("molecule_has_unseen", True), ("molecule_no_unseen", False)):
                sel = mol_flag[mol_of_atom] == flag
                if not bool(sel.any()):
                    continue
                rho = np.linalg.norm(Y["va"]["D"][sel], axis=1) / max(rms_t, 1e-12)
                resid = pred[sel] - va_std[sel]
                out["groups"].setdefault(name, {})[ckpt] = {
                    "n_atoms": int(sel.sum()),
                    "n_molecules": int((mol_flag == flag).sum()),
                    "rho_median": float(np.median(rho)),
                    "ridge_r2_D": float(1.0 - ((resid[:, 32:64]) ** 2).sum()
                                        / max(((va_std[sel][:, 32:64]) ** 2).sum(), 1e-12)),
                }
        self.out["support"] = out
        P.write_json(P.ANALYSIS / "support.json", out)
        self.plot.mark("support")

    # -- 7 ---------------------------------------------------------------
    def stage_batch(self):
        out = BI.run_batch_invariance_audit(self.caches["valid"], self.norm)
        out["interpretation_limit"] = (
            "N is batch-scoped: its mean field is the mean over every eligible pair of the "
            "batch, so R−N conflates loss of correspondence with cross-molecule leakage. The "
            "LCRC-v1 conclusion (R's relation channel collapsed to a constant) does not depend "
            "on this defect: it rests on R's own near-constant Δ_rel and on the fixed-weight "
            "R↔N content swap. No historical number is changed or reinterpreted."
        )
        self.out["batch"] = out
        P.write_json(P.ANALYSIS / "batch_invariance.json", out)
        self.plot.mark("batch")

    # -- 8 ---------------------------------------------------------------
    def stage_decision(self):
        binding = self.out.get("binding") or P.read_json(P.ANALYSIS / "binding.json")
        recon = self.out.get("recon") or P.read_json(P.ANALYSIS / "reconstruction.json")
        graph = self.out.get("graph") or P.read_json(P.ANALYSIS / "graphlevel.json")
        seeds = list(P.PROJ_SEEDS)
        ckpts = list(P.CHECKPOINTS)

        def broad_from(rho: dict, mol: dict) -> dict:
            crit = {
                "median_rho_ge_0.15": rho["rho_median"] >= 0.15,
                "p10_rho_ge_0.05": rho["rho_p10"] >= 0.05,
                "frac_atoms_rho_ge_0.05": rho["frac_atoms_rho_ge_0.05"] >= 0.90,
                "median_mol_rho_ge_0.10": mol["median_mol_rho_median"] >= 0.10,
                "top1pct_energy_share_le_0.25": rho["top1pct_energy_share"] <= 0.25,
            }
            return {"criteria": crit, "pass": bool(all(crit.values()))}

        broad_real, broad_type, broad_type_own = {}, {}, {}
        for ck in ckpts:
            for s in seeds:
                e = binding["per_checkpoint_seed"][f"{ck}|p{s}"]["valid"]
                broad_real[f"{ck}|p{s}"] = broad_from(e["rho"], e["molecules"])
                broad_type[f"{ck}|p{s}"] = broad_from(e["rho_type"], e["molecules_type"])
                broad_type_own[f"{ck}|p{s}"] = broad_from(e["rho_type_own"],
                                                          e["molecules_type_own"])

        def r2(key, iname, kind, blk):
            return (recon["runs"].get(key, {}).get(iname, {}).get(kind, {})
                    .get("blocks", {}).get(blk, {}).get("r2"))

        def r2p(key, iname, kind, blk):
            return (recon["permutation_controls"].get(key, {}).get(iname, {}).get(kind, {})
                    .get("blocks", {}).get(blk, {}).get("r2"))

        table = {}
        for ck in ckpts:
            for s in seeds:
                key = f"{ck}|p{s}"
                for iname in INPUTS:
                    for kind in ("ridge", "mlp"):
                        for blk in ("T_real", "T_null", "D", "T_type", "D_type", "h"):
                            v = r2(key, iname, kind, blk)
                            if v is not None:
                                table[f"{key}|{iname}|{kind}|{blk}"] = float(v)
        perm_table = {}
        for ck in ckpts:
            for s in seeds:
                key = f"{ck}|p{s}"
                for iname in INPUTS:
                    for kind in ("ridge", "mlp"):
                        v = r2p(key, iname, kind, "D")
                        if v is not None:
                            perm_table[f"{key}|{iname}|{kind}|D"] = float(v)
        gtable = {}
        for ck in ckpts:
            for s in seeds:
                key = f"{ck}|p{s}"
                for kind in ("ridge", "mlp"):
                    for blk in ("G_T_real", "G_D", "G_D_type"):
                        v = graph["runs"][key]["E_to_G"][kind]["blocks"].get(blk, {}).get("r2")
                        if v is not None:
                            gtable[f"{key}|{kind}|{blk}"] = float(v)
                    gtable[f"{key}|{kind}|G_to_E"] = float(
                        graph["runs"][key]["G_to_E"][kind]["blocks"]["E"]["r2"])

        def vals(pattern):
            return [v for k, v in table.items() if k.endswith(pattern)]

        def hard(blk):
            res = {}
            for ck in ckpts:
                x = [table[f"{ck}|p{s}|X|{kind}|{blk}"] for s in seeds for kind in ("ridge", "mlp")
                     if f"{ck}|p{s}|X|{kind}|{blk}" in table]
                hh = [table[f"{ck}|p{s}|h|{kind}|{blk}"] for s in seeds for kind in ("ridge", "mlp")
                      if f"{ck}|p{s}|h|{kind}|{blk}" in table]
                res[ck] = {"x_min": min(x) if x else None, "x_max": max(x) if x else None,
                           "h_min": min(hh) if hh else None, "h_max": max(hh) if hh else None,
                           "hard": bool(x and hh and min(x) < 0.80 and min(hh) < 0.70)}
            return res

        hard_real, hard_type = hard("D"), hard("D_type")
        consist = {}
        for inp in ("S", "X", "h"):
            for kind in ("ridge", "mlp"):
                for blk in ("T_real", "D", "D_type"):
                    vv = [table[f"{ck}|p{s}|{inp}|{kind}|{blk}"]
                          for ck in ckpts for s in seeds
                          if f"{ck}|p{s}|{inp}|{kind}|{blk}" in table]
                    if len(vv) >= 2:
                        consist[f"{inp}|{kind}|{blk}"] = {
                            "spread": float(max(vv) - min(vv)),
                            "min": float(min(vv)), "max": float(max(vv)),
                            "consistent": bool(max(vv) - min(vv) <= 0.15)}

        def gmax(blk):
            vv = [v for k, v in gtable.items() if k.endswith(f"|{blk}")]
            return float(max(vv)) if vv else None

        graph_D = {kind: max(gtable[f"{ck}|p{s}|{kind}|G_D"] for ck in ckpts for s in seeds)
                   for kind in ("ridge", "mlp")}
        graph_T = {kind: max(gtable[f"{ck}|p{s}|{kind}|G_T_real"] for ck in ckpts for s in seeds)
                   for kind in ("ridge", "mlp")}
        graph_Dt = {kind: max(gtable[f"{ck}|p{s}|{kind}|G_D_type"] for ck in ckpts for s in seeds)
                    for kind in ("ridge", "mlp")}
        graph_to_E = {kind: max(gtable[f"{ck}|p{s}|{kind}|G_to_E"] for ck in ckpts for s in seeds)
                      for kind in ("ridge", "mlp")}

        n_broad_real = sum(v["pass"] for v in broad_real.values())
        n_broad_type = sum(v["pass"] for v in broad_type.values())
        def broad_per_ckpt(broad) -> dict:
            return {ck: int(sum(broad[f"{ck}|p{s}"]["pass"] for s in seeds)) for ck in ckpts}

        per_ckpt_real = broad_per_ckpt(broad_real)
        per_ckpt_type = broad_per_ckpt(broad_type)
        per_ckpt_type_own = broad_per_ckpt(broad_type_own)
        n_broad_type_own = sum(v["pass"] for v in broad_type_own.values())
        # "BROAD for >=2 of 3 projection seeds and for both checkpoints"
        p1 = all(v >= 2 for v in per_ckpt_real.values())
        p2 = all(hard_real[ck]["hard"] for ck in ckpts)
        p3 = bool(consist) and all(v["consistent"] for v in consist.values())
        p4 = all(graph_D[k] < 0.80 for k in ("ridge", "mlp"))
        p5 = all(v >= 2 for v in per_ckpt_type.values()) and \
            all(hard_type[ck]["hard"] for ck in ckpts)
        p5_own = all(v >= 2 for v in per_ckpt_type_own.values()) and \
            all(hard_type[ck]["hard"] for ck in ckpts)
        verdict_if_p5_own_scale = (
            "REPRESENTATION GAP / PILOT-ELIGIBLE"
            if (p1 and p2 and p3 and p4 and p5_own) else "INCONCLUSIVE")
        criteria = {
            "P1_BROAD_real": {"pass": bool(p1), "n_pass": int(n_broad_real), "n_total": 6,
                              "per_checkpoint": per_ckpt_real,
                              "per_run": {k: v["pass"] for k, v in broad_real.items()}},
            "P2_HARD_X_and_h": {"pass": bool(p2), "detail": hard_real},
            "P3_CHECKPOINT_CONSISTENT": {"pass": bool(p3), "spread_le_0.15": consist},
            "P4_NO_GRAPH_COLLAPSE": {"pass": bool(p4), "E_to_G_D_max_r2": graph_D},
            "P5_TYPE_BINDING_PRESENT": {"pass": bool(p5), "n_broad": int(n_broad_type),
                                        "per_checkpoint": per_ckpt_type,
                                        "detail": hard_type},
            "P5_SENSITIVITY_own_scale": {"pass": bool(p5_own),
                                         "n_broad": int(n_broad_type_own),
                                         "per_checkpoint": per_ckpt_type_own,
                                         "note": "same control normalised by its own train RMS; "
                                                 "pre-registered rule used the real scale"},
        }
        n1 = not p1
        n2 = all(table.get(f"{ck}|p{s}|X|{kind}|D", -1.0) >= 0.90
                 for ck in ckpts for s in seeds for kind in ("ridge", "mlp"))
        n3 = all(graph_D[k] >= 0.90 for k in ("ridge", "mlp"))
        n4 = not p5
        if all(v["pass"] for v in criteria.values()):
            verdict = "REPRESENTATION GAP / PILOT-ELIGIBLE"
        elif n1 or n2 or n3 or n4:
            verdict = "REDUNDANT / NO-GO"
        else:
            verdict = "INCONCLUSIVE"
        borderline = []
        for k, v in table.items():
            if k.split("|")[-1] in ("D", "D_type") and k.split("|")[1] in ("X", "h"):
                for thr in (0.80, 0.90):
                    if abs(v - thr) / thr < 0.20:
                        borderline.append(f"{k}={v:.4f}")
        for k, v in gtable.items():
            if k.endswith("G_D"):
                for thr in (0.80, 0.90):
                    if abs(v - thr) / thr < 0.20:
                        borderline.append(f"{k}={v:.4f}")
        if borderline and verdict != "INCONCLUSIVE":
            verdict = "INCONCLUSIVE"
        out = {
            "criteria": criteria,
            "no_go_triggers": {"N1_not_broad": bool(n1),
                               "N2_D_recoverable_from_X": bool(n2),
                               "N3_graph_collapse": bool(n3),
                               "N4_type_binding_absent": bool(n4)},
            "graph_level": {"E_to_G_D_max": graph_D, "E_to_G_T_real_max": graph_T,
                            "E_to_G_D_type_max": graph_Dt, "G_to_E_max": graph_to_E},
            "r2_table": table,
            "permutation_control_D": perm_table,
            "borderline_hits": borderline[:60],
            "verdict": verdict,
            "verdict_if_P5_own_scale": verdict_if_p5_own_scale,
            "levels": {
                "L1_mathematical": "T is a statistic of the joint path multiset; flat ≡ "
                                   "factorised; X_v is a non-injective compression of the "
                                   "path-content configuration (rank-≤96 content map).",
                "L2_empirical": "criteria above (label-free rho and R² measurements).",
                "L3_predictive": "NOT MEASURED: no property label was read; nothing here "
                                 "bounds MAE or predicts a performance change.",
            },
            "gpu_recommendation": (
                "none: do not buy a DFPB GPU round"
                if verdict == "REDUNDANT / NO-GO" else
                ("at most one minimal discriminative GPU mechanism experiment"
                 if verdict.startswith("REPRESENTATION") else
                 "none: fix the representation proposition first")),
        }
        self.out["decision"] = out
        P.write_json(P.ANALYSIS / "decision.json", out)
        self.plot.mark("decision")
        print("VERDICT:", verdict, flush=True)

    # -- runner ----------------------------------------------------------
    def run(self, stages):
        if "provenance" in stages:
            self.stage_provenance()
        if "paths" in stages:
            self.stage_paths()
        if "math" in stages:
            self.stage_math()
        if "binding" in stages:
            self.stage_binding()
        if "recon" in stages:
            self.stage_recon()
        if "graph" in stages:
            self.stage_graph()
        if "support" in stages:
            self.stage_support()
        if "batch" in stages:
            self.stage_batch()
        if "decision" in stages:
            self.stage_decision()
        P.write_json(P.ANALYSIS / "timing.json", {"stages": self.plot.rows})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages",
                    default="provenance,paths,math,binding,recon,graph,support,batch,decision")
    args = ap.parse_args()
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    P.ANALYSIS.mkdir(parents=True, exist_ok=True)
    Audit().run(stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
