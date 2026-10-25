"""Label-free preparation: physical-bond relation kernel bank + structure stats
+ the single global T scale.

Writes ``analysis/kernel_bank.npz`` (kpad [10000, 41, 41] float32, struct
[10000, 5] float32) and ``analysis/prep.json``.  No labels are read here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rcc_common as C  # noqa: E402
import rcc_kernels as K  # noqa: E402
import rcc_models as M  # noqa: E402


def main() -> int:
    cache = C.load_train_cache()
    n_rows = int(cache["n_molecules"])
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    junc_ptr = np.asarray(cache["junc_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)

    kpad = np.zeros((n_rows, C.BMAX, C.BMAX), dtype=np.float32)
    struct = np.zeros((n_rows, C.N_STRUCT), dtype=np.float32)
    n_bonds = np.diff(bond_ptr)
    sym_err = 0.0
    mass_err = 0.0
    min_eig_gap = []
    for r in range(n_rows):
        s, e = int(bond_ptr[r]), int(bond_ptr[r + 1])
        bi = bond_index[s:e]
        n_atoms = int(atom_ptr[r + 1] - atom_ptr[r])
        n_junc = int(junc_ptr[r + 1] - junc_ptr[r])
        Kmat, d = K.build_kernel(bi, n_atoms)
        n = Kmat.shape[0]
        assert n == n_bonds[r], (r, n, n_bonds[r])
        kpad[r, :n, :n] = Kmat.astype(np.float32)
        struct[r] = K.structure_stats(Kmat, n, n_atoms, n_junc)
        sym_err = max(sym_err, float(np.abs(Kmat - Kmat.T).max()) if n else 0.0)
        mass_err = max(mass_err, abs(float(d.sum()) - n))
        if r < 400:
            Kd, _, _ = K.null_from_row_sums(Kmat)
            B = Kmat - Kd
            ev = np.linalg.eigvalsh(0.5 * (B + B.T))
            min_eig_gap.append(float(np.abs(ev).max() / max(np.abs(ev).sum(), 1e-30)))
        if r % 2000 == 0:
            C.log(f"[prep] row {r}/{n_rows}")

    # --- the scale of T at the shared initial state (one scalar for both arms)
    norm = None
    import ad_data as D

    norm = D.load_norm()
    model, info = M.build_initial_state(norm, 0)
    collate = C.make_collate((kpad, struct))
    fit_rows = C.load_split()["cache_rows"]["fit"]
    ss = {"real": 0.0, "null": 0.0}
    n_val = 0
    ratios = []
    top1 = []
    with torch.no_grad():
        for start in range(0, len(fit_rows), 128):
            rows = fit_rows[start:start + 128]
            batch = collate(cache, rows)
            for mode in ("real", "null"):
                T = model.compute(batch, mode=mode)["T"].double()
                ss[mode] += float((T * T).sum())
                if mode == "real":
                    Tr = T
                else:
                    Td = T
            n_val += Tr.shape[0]
            if start < 128 * 8:
                for i in range(Tr.shape[0]):
                    a = Tr[i].numpy()
                    f = Td[i].numpy()
                    ratios.append(float(np.linalg.norm(a - f) / max(np.linalg.norm(a), 1e-30)))
                    tmat = np.zeros((C.RANK, C.RANK))
                    tmat[np.triu_indices(C.RANK)] = a
                    tmat = tmat + tmat.T - np.diag(np.diag(tmat))
                    ev = np.abs(np.linalg.eigvalsh(tmat))
                    top1.append(float(ev.max() / max(ev.sum(), 1e-30)))
    rms = {k: float(np.sqrt(v / (n_val * C.T_DIM))) for k, v in ss.items()}
    t_scale = float(np.sqrt(0.5 * (rms["real"] ** 2 + rms["null"] ** 2)))

    stats = {
        "protocol": "rcc-v0",
        "git_commit": C.git_commit(),
        "config_sha256": C.cfg_sha256(),
        "n_rows": n_rows,
        "bmax": C.BMAX,
        "rows_with_max_bonds": int(n_bonds.max()),
        "max_symmetry_error": sym_err,
        "max_mass_error_sum_d_minus_nbonds": mass_err,
        "b_top1_abs_eig_share_400rows": {
            "mean": float(np.mean(min_eig_gap)),
            "min": float(np.min(min_eig_gap)),
            "max": float(np.max(min_eig_gap)),
        },
        "init": {
            "seed": 0,
            "initial_state_sha256": info["initial_state_sha256"],
            "parameters": info["parameters"],
            "parameters_breakdown": info["parameters_breakdown"],
            "ssl_sha256": info["ssl_sha256"],
        },
        "t_scale": t_scale,
        "t_rms_at_init": rms,
        "t_real_vs_null": {
            "n": len(ratios),
            "median_rel_fro_diff": float(np.median(ratios)),
            "min": float(np.min(ratios)),
            "p05": float(np.percentile(ratios, 5)),
            "max": float(np.max(ratios)),
        },
        "t_real_top1_eig_share": {
            "n": len(top1),
            "median": float(np.median(top1)),
            "p95": float(np.percentile(top1, 95)),
            "max": float(np.max(top1)),
        },
        "kpad_sha256": None,
        "struct_sha256": None,
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.BANK_PATH, kpad=kpad, struct=struct)
    stats["kpad_sha256"] = C.sha256_file(C.BANK_PATH)
    stats["file_bytes"] = int(C.BANK_PATH.stat().st_size)
    C.write_json(C.ANALYSIS / "prep.json", stats)
    print(f"[prep] rms(T_real)={rms['real']:.4f} rms(T_null)={rms['null']:.4f} "
          f"t_scale={t_scale:.4f}")
    print(f"[prep] median ||T_R-T_D||/||T_R|| = {stats['t_real_vs_null']['median_rel_fro_diff']:.4f}, "
          f"median top1 eig share of T_R = {stats['t_real_top1_eig_share']['median']:.4f}")
    print(f"[prep] bank {stats['file_bytes'] / 1e6:.1f} MB sha {stats['kpad_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
