"""Graph-level collapse audit (PROTOCOL.md §6).

Per-molecule aggregates of ``T_real``, ``D`` and ``D_type`` (sum, mean, mean of
squares) are regressed against LCRC-A's existing graph-level input
``E = [standardised global33, mean_v h, mean_v h², log(1+|V|)]`` and vice versa.
Both directions are reported: ``E -> G`` is the redundancy test, ``G -> E`` says
how much of the existing graph vector the new aggregates would carry.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

import pb_common as P
import pb_recon as R


def molecule_aggregates(values: np.ndarray, atom_mol: np.ndarray, n_mols: int,
                        chunk: int = 65536) -> dict[str, np.ndarray]:
    """``sum``, ``mean`` and ``mean of squares`` of a per-atom feature, per molecule."""
    d = values.shape[1]
    s = np.zeros((n_mols, d), dtype=np.float64)
    sq = np.zeros((n_mols, d), dtype=np.float64)
    cnt = np.bincount(atom_mol, minlength=n_mols).astype(np.float64)
    np.add.at(s, atom_mol, values.astype(np.float64))
    np.add.at(sq, atom_mol, values.astype(np.float64) ** 2)
    mean = s / np.maximum(cnt, 1)[:, None]
    return {"sum": s, "mean": mean, "mean_sq": sq / np.maximum(cnt, 1)[:, None]}


def new_graph_features(stats: Mapping[str, Mapping[str, np.ndarray]],
                       atom_mol: np.ndarray, n_mols: int) -> dict[str, np.ndarray]:
    """Concatenate (sum, mean, mean²) for each named per-atom statistic."""
    out = {}
    for name, values in stats.items():
        agg = molecule_aggregates(values, atom_mol, n_mols)
        out[name] = np.concatenate([agg["sum"], agg["mean"], agg["mean_sq"]], axis=1)
    return out


def existing_graph_features(cache: Mapping, enc: Mapping[str, np.ndarray],
                            norm: Mapping, atom_mol: np.ndarray, n_mols: int) -> np.ndarray:
    """Exactly LCRC-A's ``rho`` input: [g33, mean h, mean h², log(1+n_atoms)]."""
    g = (np.asarray(cache["global"], dtype=np.float64) - np.asarray(norm["global_mean"])) \
        / np.asarray(norm["global_scale"])
    h = enc["h"].astype(np.float64)
    cnt = np.bincount(atom_mol, minlength=n_mols).astype(np.float64)
    mean_h = np.zeros((n_mols, h.shape[1]))
    mean_h2 = np.zeros((n_mols, h.shape[1]))
    np.add.at(mean_h, atom_mol, h)
    np.add.at(mean_h2, atom_mol, h * h)
    mean_h /= np.maximum(cnt, 1)[:, None]
    mean_h2 /= np.maximum(cnt, 1)[:, None]
    return np.concatenate([g, mean_h, mean_h2, np.log1p(cnt)[:, None]], axis=1)


def reconstruct_graphs(E_tr, G_tr, E_va, G_va, *, seed: int = P.MLP_SEED,
                       blocks: Mapping[str, tuple[int, int]] | None = None) -> dict:
    """Ridge + MLP in one direction, with train-standardised features/targets."""
    std_e = R.Standardizer.fit(E_tr)
    std_g = R.Standardizer.fit(G_tr)
    Xtr = std_e.transform(E_tr)
    Ytr = std_g.transform(G_tr)
    Xva = std_e.transform(E_va)
    Yva = std_g.transform(G_va)
    n_mon = max(2, int(0.2 * len(Xtr)))
    mon = np.arange(n_mon)
    fit = np.arange(n_mon, len(Xtr))
    out = {"n_train": int(len(Xtr)), "n_valid": int(len(Xva)),
           "target_dim": int(G_tr.shape[1]), "input_dim": int(E_tr.shape[1])}
    pred_r, info_r = R.fit_predict_ridge(Xtr[fit], Ytr[fit], Xtr[mon], Ytr[mon], Xva)
    model, info_m = R.mlp_fit(Xtr[fit], Ytr[fit], Xtr[mon], Ytr[mon],
                              max_epochs=80, patience=10, seed=seed, hidden=(128, 64))
    pred_m = R.mlp_predict(model, Xva)
    train_std = std_g.raw_scale
    deg, near = R.degenerate_channels(train_std)
    out["ridge"] = {"info": info_r, "blocks": R.block_metrics(pred_r, Yva, deg | near,
                                                              blocks or {}, train_std)}
    out["mlp"] = {"info": {k: v for k, v in info_m.items() if k != "curve"},
                  "curve": info_m["curve"],
                  "blocks": R.block_metrics(pred_m, Yva, deg | near, blocks or {}, train_std)}
    out["degenerate_channels"] = int(deg.sum())
    out["near_constant_channels"] = int(near.sum())
    return out


__all__ = ["molecule_aggregates", "new_graph_features", "existing_graph_features",
           "reconstruct_graphs"]
