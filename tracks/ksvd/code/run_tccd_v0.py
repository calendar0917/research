#!/usr/bin/env python
"""TCCD-v0 runner — frozen gates from ``notes/tccd_v0_preregistration.md``.

Subcommands
-----------
``stage0``  data-free correctness (CPU; also used by pytest)
``gate0``   real-data correctness + GPU smoke (permutation / batching / relations
            / sensitivity / gradient sanity)
``gate1``   local dictionary domain audit (official train only, never uses y)
``gate2``   BAG / REL / REL-SHUFFLE composition test (D frozen)
``gate3``   FROZEN-D / TASK-D / DENSE task-coupling + uniqueness test
``gate4``   full canonical ZINC training vs the canonical strong baseline

Official test is never loaded; ``split == "test"`` is refused in the loader.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code import tccd_v0 as T  # noqa: E402

OUT_DIR = T.RESULTS_DIR
CACHE_DIR = T.CACHE_DIR

# --- Gate 1 frozen thresholds (§3 of the pre-registration) ------------------
G1_PERM_JACCARD_MEAN = 1.0
G1_PERM_JACCARD_FRAC = 0.99
G1_RECON_RATIO_PASS = 0.70
G1_RECON_RATIO_FAIL = 0.90
G1_DEAD_FRAC_MAX = 0.10
G1_REUSED_FRAC_MIN = 0.60
G1_TOP1_MASS_MAX = 0.20
G1_TOP8_MASS_MAX = 0.80
G1_CONT_AUC_PASS = 0.70
G1_CONT_AUC_FAIL = 0.60

KSVD_EPOCHS = 12
KSVD_CHUNK = 16384
CONT_POOL = 1500
CONT_NEAR_PAIRS = 800


# ===========================================================================
# record cache
# ===========================================================================
def records_cache_path(split: str, layout: T.PatchLayout, n: int) -> Path:
    return CACHE_DIR / (
        f"tccd_v0_{split}_r{T.RADIUS}_M{layout.capacity}_A{layout.n_atom}"
        f"_B{layout.n_bond}_n{n}.pkl"
    )


def load_or_build_records(split, mols, layout, atom_index, bond_index, ys, n, log=print):
    path = records_cache_path(split, layout, n)
    cached = T.load_records(path)
    if cached is not None and cached.get("meta", {}).get("construction") == T.CONSTRUCTION_FINGERPRINT:
        log(f"[{split}] loaded {len(cached['records'])} cached records from {path.name}")
        return cached["records"], True
    log(f"[{split}] building {n} records (exact canonicalization; this is the slow step)")
    t0 = time.time()
    records = T.build_records(mols, layout, atom_index, bond_index, ys=ys, log=log)
    T.save_records(
        path,
        records,
        {
            "construction": T.CONSTRUCTION_FINGERPRINT,
            "layout": layout.descriptor(),
            "split": split,
            "n": n,
            "wall_s": time.time() - t0,
            "fingerprint": T.records_fingerprint(records, layout),
        },
    )
    log(f"[{split}] cached -> {path} ({time.time() - t0:.0f}s)")
    return records, False


def internal_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(T.SPLIT_SEED)
    perm = rng.permutation(int(n))
    if n >= T.N_INTERNAL_TRAIN + T.N_INTERNAL_DEV:
        n_tr = T.N_INTERNAL_TRAIN
    else:
        n_tr = max(1, int(round(0.8 * n)))
    return perm[:n_tr], perm[n_tr:]


def stack_patches(records, indices) -> np.ndarray:
    return np.concatenate([np.asarray(records[i]["X"], dtype=np.float32) for i in indices], axis=0)


# ===========================================================================
# stage0 (data-free)
# ===========================================================================
def synthetic_zinc_like(seed: int = 0, n_atoms: int = 12):
    from tracks.ksvd.code.run_aiom_representation_audit import Mol  # type: ignore

    rng = np.random.default_rng(seed)
    atoms = rng.integers(0, 5, size=n_atoms)
    bonds = []
    btypes = []
    for v in range(1, n_atoms):
        u = int(rng.integers(0, v))
        bonds.append((min(u, v), max(u, v)))
        btypes.append(int(rng.integers(0, 3)))
    for _ in range(n_atoms // 3):
        a, b = sorted(rng.choice(n_atoms, size=2, replace=False).tolist())
        if (a, b) not in bonds:
            bonds.append((a, b))
            btypes.append(int(rng.integers(0, 3)))
    order = np.argsort([b[0] * 1000 + b[1] for b in bonds])
    bonds = [bonds[i] for i in order]
    btypes = [btypes[i] for i in order]
    return Mol(int(n_atoms), bonds, np.asarray(btypes, dtype=np.int64), np.asarray(atoms, dtype=np.int64))


def stage0(log=print) -> dict[str, Any]:
    torch = T._torch()
    from tracks.ksvd.code.graph import from_edges  # type: ignore

    results: dict[str, Any] = {}
    mol = synthetic_zinc_like(0, 12)
    layout = T.PatchLayout(capacity=8, n_atom=5, n_bond=3)
    atom_index = {c: i for i, c in enumerate(range(5))}
    bond_index = {c: i for i, c in enumerate(range(3))}

    rec = T.build_mol_record(mol, layout, atom_index, bond_index)
    rng = np.random.default_rng(1234)

    # 1. permutation invariance of x_v
    max_dx = 0.0
    for _ in range(20):
        perm_mol, perm_atom, _ = T.permute_mol_ids(mol, rng)
        rec2 = T.build_mol_record(perm_mol, layout, atom_index, bond_index)
        for v in range(mol.n):
            max_dx = max(max_dx, float(np.abs(rec["X"][v] - rec2["X"][perm_atom[v]]).max()))
    results["permutation_max_dx"] = max_dx
    assert max_dx <= 1e-6, f"x_v not permutation invariant: {max_dx}"

    # 2. relation contraction joint-permutation correctness (float64 check)
    Rint, Rb, Rgeo = T.relation_matrices(rec)
    X = torch.as_tensor(rec["X"])
    D = torch.as_tensor(T.random_normalized_dictionary(layout.feature_dim, T.K_DICT, 0), dtype=torch.float32)
    C = T.iht_codes(D, X)
    iu0, iu1 = T.sym_indices(T.K_DICT)
    # exact-arithmetic check in float64 (the float32 model path is unit tested
    # for structural equivariance, not for bit-exactness)
    C64 = C.double()
    iu0d, iu1d = iu0, iu1
    h = T.compose_torch(C64, torch.as_tensor(Rint).double(),
                        [torch.as_tensor(r).double() for r in Rb],
                        torch.as_tensor(Rgeo).double(), iu0d, iu1d)
    n = mol.n
    perm = rng.permutation(n)
    tperm = torch.as_tensor(perm)
    Cp = C64.index_select(0, tperm)
    Rp_int = torch.as_tensor(Rint).double().index_select(0, tperm).index_select(1, tperm)
    Rp_b = [torch.as_tensor(r).double().index_select(0, tperm).index_select(1, tperm) for r in Rb]
    Rp_geo = torch.as_tensor(Rgeo).double().index_select(0, tperm).index_select(1, tperm)
    hp = T.compose_torch(Cp, Rp_int, Rp_b, Rp_geo, iu0d, iu1d)
    joint_delta = float((h - hp).abs().max())
    results["joint_permutation_h_delta"] = joint_delta
    assert joint_delta <= 1e-6, f"C^T R C not permutation equivariant: {joint_delta}"

    # 3. sensitivity: permute C rows only, keep R fixed
    h_broken = T.compose_torch(Cp, torch.as_tensor(Rint).double(),
                               [torch.as_tensor(r).double() for r in Rb],
                               torch.as_tensor(Rgeo).double(), iu0d, iu1d)
    rel = float((h - h_broken).norm() / (h.norm() + 1e-12))
    results["row_permutation_relative_change"] = rel
    assert rel >= 0.05, f"composition insensitive to environment/topology decoupling: {rel}"

    # 4. exact sparsity of IHT
    nnz = (C.abs() > 1e-9).sum(dim=1)
    results["iht_nnz_min"] = int(nnz.min())
    results["iht_nnz_max"] = int(nnz.max())
    assert int(nnz.max()) == T.SPARSITY and int(nnz.min()) == T.SPARSITY

    # 5. gradient sanity of reconstruction + task w.r.t. D
    Dp = torch.as_tensor(T.random_normalized_dictionary(layout.feature_dim, T.K_DICT, 0),
                         dtype=torch.float32).clone().requires_grad_(True)
    Cg = T.iht_codes(Dp, X)
    loss_rec = (((X - Cg @ Dp.t()) ** 2).sum(dim=1) / ((X ** 2).sum(dim=1) + 1e-8)).mean()
    g_rec = torch.autograd.grad(loss_rec, Dp, retain_graph=True)[0]
    results["grad_rec_norm"] = float(g_rec.norm())
    assert float(g_rec.norm()) > 0
    head = torch.nn.Linear(T.h_dim(T.K_DICT), 1)
    hp2 = T.compose_torch(Cg, torch.as_tensor(Rint), [torch.as_tensor(r) for r in Rb],
                          torch.as_tensor(Rgeo), iu0, iu1)
    loss_task = (head(hp2) - 1.0).abs().mean()
    g_task = torch.autograd.grad(loss_task, Dp, retain_graph=True)[0]
    results["grad_task_norm"] = float(g_task.norm())
    assert float(g_task.norm()) > 0
    # dictionary update non-zero + no column collapse after a step
    opt = torch.optim.Adam([Dp], lr=1e-3)
    D_before = Dp.detach().clone()
    loss_rec.backward()
    opt.step()
    results["dict_update_norm"] = float((Dp.detach() - D_before).norm())
    assert results["dict_update_norm"] > 0
    results["col_norm_min"] = float(Dp.detach().norm(dim=0).min())

    results["ok"] = True
    log(f"[stage0] all data-free checks passed: {results}")
    return results


# ===========================================================================
# gate0
# ===========================================================================
def gate0(args, log=print) -> int:
    torch = T._torch()
    device = args.device
    mols, ys = T.load_mols(args.data_root, "train", limit=int(args.n_graphs))
    atom_index, bond_index = T.category_catalog(mols)
    M = T.choose_capacity(mols)
    layout = T.PatchLayout(capacity=M, n_atom=len(atom_index), n_bond=len(bond_index))
    log(f"[gate0] M={M} A={layout.n_atom} B={layout.n_bond} F={layout.feature_dim} device={device}")
    records = T.build_records(mols, layout, atom_index, bond_index, ys=ys, log=log)
    indices = list(range(len(records)))

    D0 = T.random_normalized_dictionary(layout.feature_dim, T.K_DICT, T.DICT_SEED)
    model = T.TCCDModel.build(layout.feature_dim, T.K_DICT, 5, D0, dense=False, frozen_D=False)
    model = model.to(device)

    out: dict[str, Any] = {
        "layout": layout.descriptor(),
        "n_graphs": len(records),
        "device": device,
        "official_test_loaded": False,
    }

    # 1. permutation invariance: x_v, code, C^T R C, prediction
    rng = np.random.default_rng(7)
    max_dx = 0.0
    jac = []
    max_dh = 0.0
    max_dpred = 0.0
    iu0, iu1 = T.sym_indices(T.K_DICT)
    for gi in list(range(0, len(records), max(1, len(records) // 40)))[:40]:
        rec = records[gi]
        rint, rb, rgeo = T.relation_matrices(rec)
        X = torch.as_tensor(rec["X"], device=device)
        with torch.no_grad():
            C = T.iht_codes(model.D, X)
            h = T.compose_torch(C.double(), torch.as_tensor(rint, device=device).double(),
                                [torch.as_tensor(r, device=device).double() for r in rb],
                                torch.as_tensor(rgeo, device=device).double(), iu0, iu1)
            p = float(model.head(h.float()))
        for _ in range(5):
            pm, perm_atom, _ = T.permute_mol_ids(mols[gi], rng)
            rec2 = T.build_mol_record(pm, layout, atom_index, bond_index)
            X2 = torch.as_tensor(rec2["X"], device=device)
            with torch.no_grad():
                C2 = T.iht_codes(model.D, X2)
            for v in range(rec["n"]):
                a = perm_atom[v]
                max_dx = max(max_dx, float((X[v] - X2[a]).abs().max()))
                jac.append(T.support_jaccard(C[v].cpu().numpy(), C2[a].cpu().numpy()))
            rint2, rb2, rgeo2 = T.relation_matrices(rec2)
            with torch.no_grad():
                h2 = T.compose_torch(C2.double(), torch.as_tensor(rint2, device=device).double(),
                                     [torch.as_tensor(r, device=device).double() for r in rb2],
                                     torch.as_tensor(rgeo2, device=device).double(), iu0, iu1)
                p2 = float(model.head(h2.float()))
            max_dh = max(max_dh, float((h - h2).abs().max()))
            max_dpred = max(max_dpred, abs(p - p2))
    out["permutation"] = {
        "max_dx": max_dx,
        "mean_support_jaccard": float(np.mean(jac)),
        "min_support_jaccard": float(np.min(jac)),
        "max_h_delta": max_dh,
        "max_pred_delta": max_dpred,
    }

    # 2. batching invariance
    sel = list(range(min(64, len(records))))
    b = T.make_batch(records, sel, device)
    with torch.no_grad():
        batched = model(b).cpu().numpy()
    single = []
    with torch.no_grad():
        for gi in sel:
            bs = T.make_batch(records, [gi], device)
            single.append(float(model(bs)[0]))
    batch_delta = float(np.abs(batched - np.asarray(single)).max())
    out["batching_max_delta"] = batch_delta

    # 3. gradient sanity (task + reconstruction reach D, exact sparsity)
    T.train_model(model, records, records, sel, sel, device, seed=0, max_epochs=3,
                  patience=3, batch=16, calibrate_rec=True, log=log)
    C_last = model.encode(torch.as_tensor(records[sel[0]]["X"], device=device)).detach()
    nnz = int((C_last.abs() > 1e-9).sum(dim=1).max().item())
    col_norms = model.D.detach().norm(dim=0)
    out["gradient"] = {
        "nnz_max": nnz,
        "col_norm_min": float(col_norms.min()),
        "col_norm_max": float(col_norms.max()),
        "D_changed": float((model.D.detach().cpu().numpy() - D0).__abs__().max()) > 0,
    }

    # evaluate pre-registered Gate 0 conditions
    checks = {
        "permutation_x": max_dx <= 1e-6,
        "permutation_code": float(np.mean(jac)) >= 0.999,
        "permutation_h": max_dh <= 1e-5,
        "permutation_pred": max_dpred <= 1e-5,
        "batching": batch_delta <= 1e-5,
        "gradient_sparsity": nnz == T.SPARSITY,
        "gradient_dict_changed": out["gradient"]["D_changed"],
    }
    # sensitivity (relation decoupling) on a real molecule
    rec = records[0]
    rint, rb, rgeo = T.relation_matrices(rec)
    X = torch.as_tensor(rec["X"], device=device)
    with torch.no_grad():
        C = T.iht_codes(model.D, X)
        iu0, iu1 = T.sym_indices(T.K_DICT)
        h = T.compose_torch(C.double(), torch.as_tensor(rint, device=device).double(),
                            [torch.as_tensor(r, device=device).double() for r in rb],
                            torch.as_tensor(rgeo, device=device).double(), iu0, iu1)
        perm = np.random.default_rng(11).permutation(rec["n"])
        if rec["n"] > 1 and np.array_equal(perm, np.arange(rec["n"])):
            perm = np.roll(perm, 1)
        hb = T.compose_torch(C[torch.as_tensor(perm, device=device)].double(),
                             torch.as_tensor(rint, device=device).double(),
                             [torch.as_tensor(r, device=device).double() for r in rb],
                             torch.as_tensor(rgeo, device=device).double(), iu0, iu1)
    sens = float((h - hb).norm() / (h.norm() + 1e-12))
    out["sensitivity_relative_change"] = sens
    checks["sensitivity"] = sens >= 0.05

    out["checks"] = checks
    out["gate0_pass"] = bool(all(checks.values()))
    T.write_json(OUT_DIR / "gate0.json", out)
    log(f"[gate0] {'PASS' if out['gate0_pass'] else 'FAIL'}: {checks}")
    return 0 if out["gate0_pass"] else 1


# ===========================================================================
# gate1
# ===========================================================================
def _patch_graph_for_wl(mol, root, adj):
    from tracks.ksvd.code.graph import from_edges  # type: ignore

    dist = T.bfs_distances(adj, root, T.RADIUS)
    orig = sorted(dist)
    local = {a: i for i, a in enumerate(orig)}
    edges, et = [], {}
    for e, (a, b) in enumerate(mol.bonds):
        a, b = int(a), int(b)
        if a in local and b in local:
            la, lb = sorted((local[a], local[b]))
            edges.append((la, lb))
            et[(la, lb)] = int(mol.bond_types[e])
    graph = from_edges(len(orig), edges)
    nt = np.asarray([int(mol.node_types[a]) for a in orig], dtype=np.int64)
    nt[local[root]] = nt[local[root]] + 1000  # mark the root by a distinct colour
    return graph, nt, et


def _auc_paired(near_d, rand_d) -> float:
    near_d = np.asarray(near_d)
    rand_d = np.asarray(rand_d)
    return float(np.mean(near_d < rand_d) + 0.5 * np.mean(near_d == rand_d))


def gate1(args, log=print) -> int:
    t_start = time.time()
    device = "cpu"  # Gate 1 is a detached, label-free domain audit
    mols, ys = T.load_mols(args.data_root, "train", limit=args.limit)
    atom_index, bond_index = T.category_catalog(mols)
    M = T.choose_capacity(mols)
    layout = T.PatchLayout(capacity=M, n_atom=len(atom_index), n_bond=len(bond_index))
    log(f"[gate1] M={M} A={layout.n_atom} B={layout.n_bond} F={layout.feature_dim}")
    records, _ = load_or_build_records("train", mols, layout, atom_index, bond_index, ys,
                                       len(mols), log=log)
    tr_idx, dev_idx = internal_split(len(records))
    out: dict[str, Any] = {
        "layout": layout.descriptor(),
        "n_total": len(records),
        "n_internal_train": int(len(tr_idx)),
        "n_internal_dev": int(len(dev_idx)),
        "truncations": int(sum(r["truncations"] for r in records)),
        "official_test_loaded": False,
        "official_valid_used": False,
        "y_used": False,
    }

    X_fit = stack_patches(records, tr_idx)
    X_dev = stack_patches(records, dev_idx)
    log(f"[gate1] X_fit {X_fit.shape} X_dev {X_dev.shape}")

    D0 = T.random_normalized_dictionary(layout.feature_dim, T.K_DICT, T.DICT_SEED)
    prog: dict[str, Any] = {}
    t0 = time.time()
    D, meta = T.ksvd_fit(X_fit, D0, s=T.SPARSITY, epochs=KSVD_EPOCHS, chunk=KSVD_CHUNK,
                         seed=T.DICT_SEED, log=log, progress=prog)
    out["ksvd"] = {"epochs": KSVD_EPOCHS, "history": meta["history"], "wall_s": time.time() - t0}
    import pickle

    with (OUT_DIR / "dictionary_gate1.pkl").open("wb") as fh:
        pickle.dump({"D": D, "layout": layout.descriptor(), "seed": T.DICT_SEED}, fh, protocol=4)

    log("[gate1] reconstruction audit")
    C_fit = T.omp_codes(D, X_fit)
    C_dev = T.omp_codes(D, X_dev)
    Cr_fit = T.omp_codes(D0, X_fit)
    Cr_dev = T.omp_codes(D0, X_dev)
    e = {
        "learned_fit": T.relative_reconstruction_error(D, X_fit, C_fit),
        "learned_dev": T.relative_reconstruction_error(D, X_dev, C_dev),
        "random_fit": T.relative_reconstruction_error(D0, X_fit, Cr_fit),
        "random_dev": T.relative_reconstruction_error(D0, X_dev, Cr_dev),
    }
    e["ratio_dev"] = e["learned_dev"] / max(e["random_dev"], 1e-12)
    out["reconstruction"] = e

    # -- permutation stability
    log("[gate1] permutation stability")
    rng = np.random.default_rng(T.SPLIT_SEED + 5)
    max_dx = 0.0
    jac: list[float] = []
    for gi in [int(i) for i in dev_idx[:100]]:
        rec = records[gi]
        for _ in range(20):
            pm, perm_atom, _ = T.permute_mol_ids(mols[gi], rng)
            rec2 = T.build_mol_record(pm, layout, atom_index, bond_index)
            max_dx = max(max_dx, float(np.abs(rec["X"] - rec2["X"][perm_atom]).max()))
            ca = T.omp_codes(D, rec["X"], chunk=10_000)
            cb = T.omp_codes(D, rec2["X"], chunk=10_000)
            for v in range(rec["n"]):
                jac.append(T.support_jaccard(ca[v], cb[perm_atom[v]]))
    out["permutation_stability"] = {
        "max_dx": max_dx,
        "mean_jaccard": float(np.mean(jac)),
        "frac_jaccard_one": float(np.mean(np.asarray(jac) >= 1.0 - 1e-9)),
        "n_checks": len(jac),
    }

    # -- reuse
    log("[gate1] reuse statistics")
    support = (np.abs(C_dev) > 1e-8)
    atom_support = support.sum(axis=0)
    atom_mass = np.abs(C_dev).sum(axis=0)
    mol_of_patch = np.concatenate([np.full(records[i]["n"], k) for k, i in enumerate(dev_idx)])
    mol_presence = np.zeros((support.shape[1],), dtype=np.int64)
    for k in range(support.shape[1]):
        mol_presence[k] = len(np.unique(mol_of_patch[support[:, k]]))
    dead = int(np.sum(atom_support < 5))
    reused = int(np.sum((atom_support >= 20) & (mol_presence >= 5)))
    total_mass = float(atom_mass.sum())
    order = np.argsort(atom_mass)[::-1]
    out["reuse"] = {
        "dead_atoms": dead,
        "dead_frac": dead / T.K_DICT,
        "reused_atoms": reused,
        "reused_frac": reused / T.K_DICT,
        "top1_mass_share": float(atom_mass[order[0]] / max(total_mass, 1e-12)),
        "top8_mass_share": float(atom_mass[order[:8]].sum() / max(total_mass, 1e-12)),
        "atom_support": atom_support.astype(int).tolist(),
        "atom_mol_presence": mol_presence.astype(int).tolist(),
        "mean_coeff_entropy": float(T.coefficient_entropy(C_dev).mean()),
    }

    # -- local-coordinate continuity
    log("[gate1] local-coordinate continuity")
    cont = continuity_audit(mols, records, dev_idx, D, C_dev, log=log)
    out["continuity"] = cont

    # -- atom semantics (report only)
    log("[gate1] atom semantics")
    out["atom_semantics"] = atom_semantics(records, dev_idx, C_dev, order[:8])

    # -- gate decision
    perm = out["permutation_stability"]
    reuse = out["reuse"]
    r = out["reconstruction"]
    checks = {
        "permutation_x": perm["max_dx"] <= 1e-6,
        "permutation_jaccard_mean": perm["mean_jaccard"] >= G1_PERM_JACCARD_MEAN - 1e-9,
        "permutation_jaccard_frac": perm["frac_jaccard_one"] >= G1_PERM_JACCARD_FRAC,
        "reconstruction_gain": r["ratio_dev"] <= G1_RECON_RATIO_PASS,
        "reuse_dead": reuse["dead_frac"] <= G1_DEAD_FRAC_MAX,
        "reuse_reused": reuse["reused_frac"] >= G1_REUSED_FRAC_MIN,
        "reuse_mass_top1": reuse["top1_mass_share"] <= G1_TOP1_MASS_MAX,
        "reuse_mass_top8": reuse["top8_mass_share"] <= G1_TOP8_MASS_MAX,
        "continuity_auc": cont["code_auc"] >= G1_CONT_AUC_PASS,
    }
    out["checks"] = checks
    out["gate1_pass"] = bool(all(checks.values()))
    out["stop_reasons"] = [k for k, v in checks.items() if not v]
    out["wall_s"] = time.time() - t_start
    T.write_json(OUT_DIR / "gate1.json", out)
    log(f"[gate1] {'PASS' if out['gate1_pass'] else 'FAIL'}: {checks}")
    if not out["gate1_pass"]:
        log(f"[gate1] STOP reasons: {out['stop_reasons']}")
    return 0 if out["gate1_pass"] else 1


def continuity_audit(mols, records, dev_idx, D, C_dev, log=print) -> dict[str, Any]:
    from tracks.ksvd.code.run_wholegraph_canonical_registration_audit import (  # type: ignore
        _histogram_matrix,
        attributed_wl_fingerprint,
    )

    rng = np.random.default_rng(T.SPLIT_SEED + 9)
    # pool of random (molecule, root) patches from the internal-dev molecules
    pool_graphs = [int(i) for i in dev_idx]
    patch_refs: list[tuple[int, int]] = []
    seen_roots = set()
    for _ in range(CONT_POOL * 8):
        gi = pool_graphs[int(rng.integers(len(pool_graphs)))]
        v = int(rng.integers(records[gi]["n"]))
        if (gi, v) in seen_roots:
            continue
        seen_roots.add((gi, v))
        patch_refs.append((gi, v))
        if len(patch_refs) >= CONT_POOL:
            break

    hists = []
    sizes = []
    rootcats = []
    keys = []
    for (gi, v) in patch_refs:
        mol = mols[gi]
        adj = T._adjacency(mol)
        graph, nt, et = _patch_graph_for_wl(mol, v, adj)
        hists.append(attributed_wl_fingerprint(graph, nt, et, rounds=3))
        sizes.append(int(graph.n))
        rootcats.append(int(mol.node_types[v]))
        slot_nodes, _, key = T.patch_slot_order(mol, v, adj=adj)
        keys.append(key)

    vocab = sorted({k for h in hists for k in h}, key=repr)
    Fm = _histogram_matrix(hists, vocab)
    Fm = Fm / (np.linalg.norm(Fm, axis=1, keepdims=True) + 1e-12)
    S = Fm @ Fm.T
    np.fill_diagonal(S, -1.0)

    # near pairs: top cosine, non-isomorphic
    flat = np.argsort(S, axis=None)[::-1]
    near: list[tuple[int, int, float]] = []
    near_tail: list[tuple[int, int, float]] = []
    seen = set()
    for idx in flat:
        i, j = divmod(int(idx), S.shape[0])
        if i >= j:
            continue
        if keys[i] == keys[j]:
            continue
        pair = (i, j)
        if pair in seen:
            continue
        seen.add(pair)
        if len(near) < CONT_NEAR_PAIRS:
            near.append((i, j, float(S[i, j])))
        elif len(near_tail) < CONT_NEAR_PAIRS:
            near_tail.append((i, j, float(S[i, j])))
        else:
            break

    # matched random pairs on (patch size, root atom category)
    by_key: dict[tuple[int, int], list[int]] = {}
    for idx in range(len(patch_refs)):
        by_key.setdefault((sizes[idx], rootcats[idx]), []).append(idx)
    random_pairs: list[tuple[int, int]] = []
    for (i, j, _s) in near:
        ok = False
        for _try in range(200):
            ai = int(rng.choice(by_key.get((sizes[i], rootcats[i]), [i])))
            bj = int(rng.choice(by_key.get((sizes[j], rootcats[j]), [j])))
            if ai != bj and keys[ai] != keys[bj]:
                random_pairs.append((ai, bj))
                ok = True
                break
        if not ok:
            random_pairs.append((i, j))

    # x-space and code-space distances
    idx_by_ref = {}
    for k, (gi, v) in enumerate(patch_refs):
        idx_by_ref[(gi, v)] = k
    # C_dev rows correspond to concatenated dev patches in dev_idx order
    row_offsets = {}
    off = 0
    for k, gi in enumerate([int(i) for i in dev_idx]):
        row_offsets[gi] = off
        off += records[gi]["n"]

    def dist(code_space: bool, i: int, j: int) -> float:
        gi, vi = patch_refs[i]
        gj, vj = patch_refs[j]
        if code_space:
            return float(np.linalg.norm(C_dev[row_offsets[gi] + vi] - C_dev[row_offsets[gj] + vj]))
        xi = np.asarray(records[gi]["X"][vi], dtype=np.float64)
        xj = np.asarray(records[gj]["X"][vj], dtype=np.float64)
        return float(np.linalg.norm(xi - xj))

    x_near = [dist(False, i, j) for i, j, _ in near]
    x_rand = [dist(False, i, j) for i, j in random_pairs]
    c_near = [dist(True, i, j) for i, j, _ in near]
    c_rand = [dist(True, i, j) for i, j in random_pairs]
    graded: dict[str, Any] = {}
    if near_tail:
        tail_pairs = [(i, j) for i, j, _ in near_tail][: len(near_tail)]
        tail_rand = []
        for (i, j) in tail_pairs:
            ai = int(rng.choice(by_key.get((sizes[i], rootcats[i]), [i])))
            bj = int(rng.choice(by_key.get((sizes[j], rootcats[j]), [j])))
            tail_rand.append((ai, bj))
        graded = {
            "near_tail_wl_cosine_mean": float(np.mean([s for _, _, s in near_tail])),
            "x_auc": _auc_paired([dist(False, i, j) for i, j in tail_pairs],
                                 [dist(False, i, j) for i, j in tail_rand]),
            "code_auc": _auc_paired([dist(True, i, j) for i, j in tail_pairs],
                                    [dist(True, i, j) for i, j in tail_rand]),
        }
    return {
        "n_pool": len(patch_refs),
        "n_near_pairs": len(near),
        "n_random_pairs": len(random_pairs),
        "x_auc": _auc_paired(x_near, x_rand),
        "code_auc": _auc_paired(c_near, c_rand),
        "x_near_mean": float(np.mean(x_near)),
        "x_random_mean": float(np.mean(x_rand)),
        "code_near_mean": float(np.mean(c_near)),
        "code_random_mean": float(np.mean(c_rand)),
        "near_wl_cosine_mean": float(np.mean([s for _, _, s in near])),
        "graded_tail": graded,
    }


def atom_semantics(records, dev_idx, C_dev, top_atoms) -> dict[str, Any]:
    keys_flat = []
    for gi in [int(i) for i in dev_idx]:
        keys_flat.extend(records[gi]["keys"])
    n_patches = len(keys_flat)
    key_id: dict[bytes, int] = {}
    keys_int = np.empty(n_patches, dtype=np.int64)
    for i, k in enumerate(keys_flat):
        idx = key_id.get(k)
        if idx is None:
            idx = len(key_id)
            key_id[k] = idx
        keys_int[i] = idx
    out: dict[str, Any] = {}
    for a in top_atoms:
        col = C_dev[:, int(a)]
        act = np.argsort(np.abs(col))[::-1][:50]
        modal = np.bincount(keys_int[act]).max() / 50.0
        rand_idx = np.random.default_rng(int(a)).choice(n_patches, size=50, replace=False)
        baseline = np.bincount(keys_int[rand_idx]).max() / 50.0
        out[str(int(a))] = {"top50_key_concentration": float(modal), "random_baseline": float(baseline)}
    vals = [v["top50_key_concentration"] for v in out.values()]
    base = [v["random_baseline"] for v in out.values()]
    out["_summary"] = {
        "mean_concentration": float(np.mean(vals)),
        "mean_baseline": float(np.mean(base)),
        "max_concentration": float(np.max(vals)),
    }
    return out


# ===========================================================================
# gate2 / gate3
# ===========================================================================
def _load_gate1_dictionary(records, layout) -> np.ndarray | None:
    import json
    import pickle

    p = OUT_DIR / "dictionary_gate1.pkl"
    if p.exists():
        with p.open("rb") as fh:
            return pickle.load(fh)["D"]
    return None


def _prepare(args, log):
    mols, ys = T.load_mols(args.data_root, "train", limit=args.limit)
    atom_index, bond_index = T.category_catalog(mols)
    M = T.choose_capacity(mols)
    layout = T.PatchLayout(capacity=M, n_atom=len(atom_index), n_bond=len(bond_index))
    records, _ = load_or_build_records("train", mols, layout, atom_index, bond_index, ys,
                                       len(mols), log=log)
    tr_idx, dev_idx = internal_split(len(records))
    return mols, records, layout, tr_idx, dev_idx


def _train_arm(arm: str, args, layout, records, tr_idx, dev_idx, device, D_init, log):
    dense = arm == "DENSE"
    frozen = arm == "FROZEN-D"
    readout_dim = T.K_DICT if arm == "BAG" else None
    model = T.TCCDModel.build(layout.feature_dim, T.K_DICT, 5, D_init, dense=dense,
                              frozen_D=frozen, readout_dim=readout_dim, seed=args.seed)
    if frozen:
        model.D.requires_grad_(False)
    model = model.to(device)
    shuffle = arm == "REL-SHUFFLE"
    res = T.train_model(
        model, records, records, list(tr_idx), list(dev_idx), device,
        seed=args.seed, shuffle_train=shuffle, shuffle_seed=args.seed,
        max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
        calibrate_rec=(arm == "TASK-D"), log=log,
    )
    return res


def gate2(args, log=print) -> int:
    device = args.device
    mols, records, layout, tr_idx, dev_idx = _prepare(args, log)
    D = _load_gate1_dictionary(records, layout)
    if D is None:
        raise SystemExit("gate1 dictionary not found; run gate1 first and save dictionary_gate1.pkl")
    out: dict[str, Any] = {"layout": layout.descriptor(), "seed": args.seed,
                           "official_test_loaded": False}
    for arm, shuffle in [("REL", False), ("REL-SHUFFLE", True), ("BAG", False)]:
        log(f"[gate2] arm={arm}")
        # BAG uses REL codes but a readout over m only
        res = _train_arm("BAG" if arm == "BAG" else arm, args, layout, records, tr_idx, dev_idx,
                         device, D, log)
        out[arm] = {"best_valid": res.best_valid, "best_epoch": res.best_epoch,
                    "soup_valid": res.soup_valid, "soup_members": res.soup_members}
    delta = out["REL-SHUFFLE"]["best_valid"] - out["REL"]["best_valid"]
    out["delta_shuffle_minus_rel"] = delta
    if delta >= 0.015:
        out["gate2"] = "PASS"
    elif delta < 0.005:
        out["gate2"] = "FAIL"
    else:
        out["gate2"] = "AMBIGUOUS_NEEDS_SEED1"
    T.write_json(OUT_DIR / f"gate2_seed{args.seed}.json", out)
    log(f"[gate2] delta={delta:.5f} -> {out['gate2']}")
    return 0 if out["gate2"] == "PASS" else 1


def gate3(args, log=print) -> int:
    device = args.device
    mols, records, layout, tr_idx, dev_idx = _prepare(args, log)
    D = _load_gate1_dictionary(records, layout)
    if D is None:
        raise SystemExit("gate1 dictionary not found; run gate1 first")
    out: dict[str, Any] = {"layout": layout.descriptor(), "seed": args.seed,
                           "official_test_loaded": False}
    for arm in ["FROZEN-D", "TASK-D", "DENSE"]:
        log(f"[gate3] arm={arm}")
        res = _train_arm(arm, args, layout, records, tr_idx, dev_idx, device, D, log)
        out[arm] = {"best_valid": res.best_valid, "best_epoch": res.best_epoch,
                    "soup_valid": res.soup_valid, "soup_members": res.soup_members,
                    "lam_rec": res.lam_rec}
    a = out["FROZEN-D"]["best_valid"] - out["TASK-D"]["best_valid"]
    b = out["TASK-D"]["best_valid"] - out["DENSE"]["best_valid"]
    out["test_A_task_coupling"] = a
    out["test_B_vs_dense"] = b
    out["gate3_task_coupling_pass"] = bool(a >= 0.005)
    if not out["gate3_task_coupling_pass"]:
        out["gate3"] = "STOP_NO_TASK_COUPLING"
    elif b > 0.005:
        out["gate3"] = "STOP_DICTIONARY_NO_ADVANTAGE"
    else:
        out["gate3"] = "PASS"
    T.write_json(OUT_DIR / f"gate3_seed{args.seed}.json", out)
    log(f"[gate3] A={a:.5f} B={b:.5f} -> {out['gate3']}")
    return 0 if out["gate3"] == "PASS" else 1


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gate", choices=["stage0", "gate0", "gate1", "gate2", "gate3"])
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-graphs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=T.BATCH)
    p.add_argument("--max-epochs", type=int, default=T.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=T.PATIENCE)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.gate == "stage0":
        stage0()
        return 0
    if args.gate == "gate0":
        return gate0(args)
    if args.gate == "gate1":
        return gate1(args)
    if args.gate == "gate2":
        return gate2(args)
    if args.gate == "gate3":
        return gate3(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
