"""Post-hoc probe-capacity control and per-factor decomposition.

Not part of the frozen decision rule; it answers two questions the frozen rule
cannot resolve on its own:

1. **Is the reconstructor able to learn a target that is known to be present?**
   ``T_lin = X W`` (a fixed random linear projection of X) must come back with
   R² = 1 from ridge; ``h`` from ``X`` (a LayerNorm∘MLP) is the nonlinear control.
2. **Which factor of the triple product is missing from X?**
   ``T_Q   = sum_p Q_p / sqrt(n)``      (first hop only)
   ``T_QK  = sum_p Q_p*K_p / sqrt(n)``  (adds the second-hop bond embedding)
   ``T_real= sum_p Q_p*K_p*Z_p/sqrt(n)``(adds the endpoint content pairing)

Note ``T_Q`` is *not* exactly in X either: the number of two-paths through a
neighbour ``w`` is ``deg(w)-1``, so the path marginal is a degree-weighted sum of
incident bond embeddings, while LCRC-A stores only the unweighted ``b_sum``.
"""

from __future__ import annotations

import numpy as np

import pb_common as P
import pb_paths as PP
import pb_features as F
import pb_recon as R
import pb_run as RUN

BLOCKS = {"T_Q": (0, 32), "T_QK": (32, 64), "T_real": (64, 96), "T_lin": (96, 128)}


def build_targets(enc, paths, proj):
    q, k, z = F.path_factors(enc["a"], enc["b"], paths, proj)
    n = np.bincount(paths["root"], minlength=enc["n_atoms"]).astype(np.float64)
    inv = 1.0 / np.sqrt(np.maximum(n, 1.0))
    t_q = P.seg_sum(q, paths["root"], enc["n_atoms"]) * inv[:, None]
    t_qk = P.seg_sum(q * k, paths["root"], enc["n_atoms"]) * inv[:, None]
    t_real = P.seg_sum(q * k * z, paths["root"], enc["n_atoms"]) * inv[:, None]
    return np.concatenate([t_q, t_qk, t_real], axis=1)


def canonical_correlations(A_tr, B_tr, A_va, B_va, n_comp: int = 8) -> dict:
    """Scale-free relation between two feature sets: CCA fitted on train, scored on valid."""
    mu_a, sd_a = A_tr.mean(0), A_tr.std(0) + 1e-12
    mu_b, sd_b = B_tr.mean(0), B_tr.std(0) + 1e-12
    At, Bt = (A_tr - mu_a) / sd_a, (B_tr - mu_b) / sd_b
    Av, Bv = (A_va - mu_a) / sd_a, (B_va - mu_b) / sd_b
    # whiten each side with train covariance, then SVD of the train cross-covariance
    def whiten(X):
        cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
        ev, V = np.linalg.eigh(cov)
        return X @ (V / np.sqrt(np.clip(ev, 1e-12, None))) @ V.T
    Aw, Bw = whiten(At), whiten(Bt)
    U, sv, Vt = np.linalg.svd(Aw.T @ Bw / max(len(At) - 1, 1))
    k = min(n_comp, len(sv))
    za = (Av @ (np.cov(At.T) + 1e-6 * np.eye(At.shape[1])) ** 0) @ U[:, :k]
    zb = Bv @ Vt[:k].T
    # recompute the valid correlations of the train directions
    za = Av @ np.linalg.pinv(np.cov(At.T) + 1e-6 * np.eye(At.shape[1])) @ U[:, :k]
    corrs = [float(np.corrcoef(za[:, i], zb[:, i])[0, 1]) for i in range(k)]
    return {"train_singular_values_top": [float(v) for v in sv[:k]],
            "valid_canonical_correlations": corrs,
            "valid_top1": corrs[0] if corrs else None,
            "valid_mean": float(np.mean(corrs)) if corrs else None}


def linear_map_r2(src_tr, tgt_tr, src_va, tgt_va) -> float:
    """R² of the best linear map src → tgt, fitted on train and scored on valid."""
    mu_s, sd_s = src_tr.mean(0), src_tr.std(0) + 1e-12
    mu_t, sd_t = tgt_tr.mean(0), tgt_tr.std(0) + 1e-12
    Xtr = (src_tr - mu_s) / sd_s
    Ytr = (tgt_tr - mu_t) / sd_t
    Xva = (src_va - mu_s) / sd_s
    Yva = (tgt_va - mu_t) / sd_t
    lam = 1.0
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xva @ W
    return float(1.0 - ((pred - Yva) ** 2).sum() / max((Yva ** 2).sum(), 1e-12))


def main() -> int:
    audit = RUN.Audit.__new__(RUN.Audit)
    audit.plot = P.Plot()
    audit.caches = {s: P.load_cache(s) for s in RUN.SPLITS}
    audit.norm = P.load_norm()
    audit.atom_mol = {s: F.atom_molecule_index(audit.caches[s]) for s in RUN.SPLITS}
    audit.n_mols = {s: int(audit.caches[s]["n_molecules"]) for s in RUN.SPLITS}
    audit.paths = {s: PP.split_paths(audit.caches[s]) for s in RUN.SPLITS}
    audit.enc = {}
    audit.models = {}
    for ckpt in P.CHECKPOINTS:
        model, _ = P.load_frozen_a(ckpt)
        audit.models[ckpt] = model
        audit.enc[ckpt] = F.encode(model, audit.caches["train"])
        audit.enc[f"{ckpt}|valid"] = F.encode(model, audit.caches["valid"])

    # X+ : X enriched with the *marginal and pairwise* path sums of the same
    # content (still a fixed-width, one-pass summary -- no learning, no pairing).
    def x_plus(enc, paths, proj):
        n = enc["n_atoms"]
        q, k, z = F.path_factors(enc["a"], enc["b"], paths, proj)
        return np.concatenate([
            enc["x"].astype(np.float32),
            P.seg_sum(enc["b"][paths["e2"]], paths["root"], n).astype(np.float32),
            P.seg_sum(enc["a"][paths["end"]], paths["root"], n).astype(np.float32),
            P.seg_sum(q, paths["root"], n).astype(np.float32),
            P.seg_sum(k, paths["root"], n).astype(np.float32),
            P.seg_sum(z, paths["root"], n).astype(np.float32),
            P.seg_sum(q * k, paths["root"], n).astype(np.float32),
            P.seg_sum(k * z, paths["root"], n).astype(np.float32),
        ], axis=1)

    mon_mols = np.random.RandomState(P.MONITOR_SEED).permutation(
        audit.n_mols["train"])[: P.MONITOR_MOLS]
    fit = ~np.isin(audit.atom_mol["train"], mon_mols)
    proj = P.projections(P.PRIMARY_SEED)
    Wlin = np.random.default_rng(P.PRIMARY_SEED).normal(size=(P.X_DIM, 32)) / np.sqrt(P.X_DIM)
    out = {"projection_seed": P.PRIMARY_SEED, "blocks": {k: list(v) for k, v in BLOCKS.items()},
           "runs": {},
           "note": "post-hoc probe control; not part of the frozen decision rule"}
    for ckpt in P.CHECKPOINTS:
        etr, eva = audit.enc[ckpt], audit.enc[f"{ckpt}|valid"]
        Y = {"tr": [], "va": []}
        for tag, es, paths in (("tr", etr, audit.paths["train"]),
                               ("va", eva, audit.paths["valid"])):
            base = build_targets(es, paths, proj)
            t_lin = es["x"].astype(np.float64) @ Wlin
            Y[tag] = np.concatenate([base, t_lin], axis=1).astype(np.float32)
        # scale-free type-vs-learned binding relation (per checkpoint / seed)
        proj_t = P.projections(P.PRIMARY_SEED)
        tabs = F.type_content_tables(etr["a"], etr["b"], audit.caches["train"],
                                     np.arange(audit.n_mols["train"]))
        at, bt = F.type_content_arrays(tabs)
        stats = {}
        for tag, es, paths in (("tr", etr, audit.paths["train"]),
                               ("va", eva, audit.paths["valid"])):
            st = F.t_stats(paths, *F.path_factors(es["a"], es["b"], paths, proj_t), es["n_atoms"])
            stt = F.t_stats(paths, *F.path_factors(at, bt, paths, proj_t), es["n_atoms"])
            stats[tag] = {"D": st["D"], "T_real": st["T_real"],
                          "D_type": stt["D"], "T_type": stt["T_real"]}
        relation = {
            "r2_D_type_to_D": linear_map_r2(stats["tr"]["D_type"], stats["tr"]["D"],
                                            stats["va"]["D_type"], stats["va"]["D"]),
            "r2_D_to_D_type": linear_map_r2(stats["tr"]["D"], stats["tr"]["D_type"],
                                            stats["va"]["D"], stats["va"]["D_type"]),
            "r2_T_type_to_T_real": linear_map_r2(stats["tr"]["T_type"], stats["tr"]["T_real"],
                                                 stats["va"]["T_type"], stats["va"]["T_real"]),
            "cca_D_type_vs_D": canonical_correlations(stats["tr"]["D_type"], stats["tr"]["D"],
                                                      stats["va"]["D_type"], stats["va"]["D"]),
            "cca_T_type_vs_T_real": canonical_correlations(
                stats["tr"]["T_type"], stats["tr"]["T_real"],
                stats["va"]["T_type"], stats["va"]["T_real"]),
            "rms_D_type_over_rms_D": float(np.sqrt((stats["va"]["D_type"] ** 2).sum(1).mean())
                                           / np.sqrt((stats["va"]["D"] ** 2).sum(1).mean())),
            "rms_D_type_over_rms_D_type_own_scale": float(
                np.sqrt((stats["va"]["D_type"] ** 2).sum(1).mean())
                / np.sqrt((stats["va"]["T_type"] ** 2).sum(1).mean())),
            "rms_D_over_rms_T_real": float(np.sqrt((stats["va"]["D"] ** 2).sum(1).mean())
                                           / np.sqrt((stats["va"]["T_real"] ** 2).sum(1).mean())),
        }
        XP = {"tr": x_plus(etr, audit.paths["train"], proj),
              "va": x_plus(eva, audit.paths["valid"], proj)}
        std_y = R.Standardizer.fit(Y["tr"][fit])
        Ytr_s, Yva_s = std_y.transform(Y["tr"]), std_y.transform(Y["va"])
        deg, near = R.degenerate_channels(std_y.raw_scale)
        entry = {"type_vs_learned": relation}
        for iname in ("S", "X", "h", "X+"):
            if iname == "X+":
                Xtr, Xva = XP["tr"], XP["va"]
            else:
                Xtr = etr["s"] if iname == "S" else etr[iname.lower()]
                Xva = eva["s"] if iname == "S" else eva[iname.lower()]
            std_x = R.Standardizer.fit(Xtr[fit])
            Xtr_s, Xva_s = std_x.transform(Xtr), std_x.transform(Xva)
            pred, info = R.fit_predict_ridge(Xtr_s[fit], Ytr_s[fit], Xtr_s[~fit],
                                             Ytr_s[~fit], Xva_s)
            entry[iname] = {"ridge": {"info": info,
                                      "blocks": R.block_metrics(pred, Yva_s, deg | near,
                                                                BLOCKS, std_y.raw_scale)}}
            if iname in ("X", "X+"):
                model, mi = R.mlp_fit(Xtr_s[fit], Ytr_s[fit], Xtr_s[~fit], Ytr_s[~fit],
                                      max_epochs=60, patience=8)
                entry[iname]["mlp"] = {
                    "info": {k: v for k, v in mi.items() if k != "curve"},
                    "blocks": R.block_metrics(R.mlp_predict(model, Xva_s), Yva_s,
                                              deg | near, BLOCKS, std_y.raw_scale)}
                del model
            audit.plot.mark(f"probe {ckpt} {iname}")
        for iname in [k for k in entry if k != "type_vs_learned"]:
            for kind in entry[iname]:
                for blk in entry[iname][kind]["blocks"].values():
                    blk.pop("per_channel_r2", None)
        out["runs"][ckpt] = entry
    P.write_json(P.ANALYSIS / "probe_control.json", out)
    for ckpt, entry in out["runs"].items():
        for iname, kinds in entry.items():
            if iname == "type_vs_learned":
                print(ckpt, "type_vs_learned", {k: (round(v, 4) if isinstance(v, float) else v)
                                                for k, v in kinds.items()})
                continue
            for kind, res in kinds.items():
                print(ckpt, iname, kind,
                      {b: round(v["r2"], 4) for b, v in res["blocks"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
