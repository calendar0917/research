"""Representation-compression diagnostics on the parent's TRC representations.

The parent round compresses every rooted structural object of a fragment into one
fragment embedding *before* the two sides meet.  This script measures what that
compression preserves, using the parent's own trained ``FACTOR`` checkpoint:

* how much of the pair-delta target is linearly recoverable from the compressed
  side embeddings alone (``[h_C, h_S]``, an additive probe),
* how much an explicit product basis adds (``h_C * h_S``), i.e. how much of the
  signal is interaction-shaped and therefore *not* reachable by an additive head
  on the same representation,
* the PCA spectrum of the fragment embedding (how much of the 64/128-d space is
  actually used).

Everything is out-of-fold (grouped 5-fold CV by transform) so the numbers are not
inflated by memorised pairs.  Descriptive only: no causal claim about the tokens.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np

import csoi_common as C
import csoi_data as D


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / max(ss, 1e-12))


def oof_r2(design: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray,
           lam: float = 10.0) -> dict[str, Any]:
    """Out-of-fold R^2 of each design block, grouped CV over transforms."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    fold_of = {int(t): int(i % 5) for i, t in enumerate(rng.permutation(uniq).tolist())}
    fold = np.asarray([fold_of[int(g)] for g in groups.tolist()])
    out = {}
    for name, X in design.items():
        pred = np.zeros_like(y)
        for f in range(5):
            tr = fold != f
            te = ~tr
            w = _ridge(X[tr], y[tr], lam)
            pred[te] = np.concatenate([np.ones((int(te.sum()), 1)), X[te]], 1) @ w
        out[name] = {"n_features": int(X.shape[1]), "oof_r2": _r2(y, pred),
                     "oof_rmse": float(np.sqrt(np.mean((y - pred) ** 2)))}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = D.load_pack(cfg, cfg["dataset_key"])
    parent_cfg = C.read_json(C.PARENT_ROUND / "configs" / "trc_s0.json")
    atom_dims, bond_dims = C.official_feature_dims()
    ckpt = torch.load(C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0" / "state_best.pt",
                      map_location="cpu", weights_only=False)
    model = C.PM.ResponseModel(parent_cfg, atom_dims, bond_dims, "FACTOR")
    model.load_pack(pack)
    model.load_state_dict(ckpt["state"])
    model = model.to(args.device).float().eval()
    n_frag = int(pack["n_fragments"])
    with torch.no_grad():
        h = model.encode_fragments(
            torch.arange(n_frag, device=args.device)).double().cpu().numpy()
    labels = np.asarray(pack["labels"], np.float64)
    train = np.asarray(pack["train_idx"], np.int64)
    store = C.PPAIR.PairStore(parent_cfg, pack, labels, None, train,
                              np.asarray(pack["row_ids"]), log=lambda *_a: None)
    a = np.asarray(pack["pair_mol_a"], np.int64)[store.index]
    b = np.asarray(pack["pair_mol_b"], np.int64)[store.index]
    pair_Sa = np.asarray(pack["pair_Sa"], np.int64)[store.index]
    pair_Sb = np.asarray(pack["pair_Sb"], np.int64)[store.index]
    pair_C = np.asarray(pack["pair_C"], np.int64)[store.index]
    bond = np.asarray(pack["pair_cut_bond_row"], np.int64)[store.index]
    # representation-level probe: it is about the frozen fragment embedding, not
    # about a prediction, so every non-Test pair is a legitimate probe row
    split = np.asarray(pack["molecule_split"], np.uint8)
    both = (split[a] != 2) & (split[b] != 2)
    y = (labels[b] - labels[a]) / float(store.sigma)
    groups = store.transform
    hs_a, hs_b, hc = h[pair_Sa], h[pair_Sb], h[pair_C]
    # the delta is antisymmetric in the two sides, so use the symmetric basis
    d = hs_b - hs_a
    prod = 0.5 * (hs_a * hc + hs_b * hc)
    sumz = 0.5 * (hs_a + hs_b)
    bond_oh = np.eye(int(np.max(bond)) + 1)[np.clip(bond, 0, None)]
    design = {
        "compressed_side_embeddings_only": np.concatenate([hc, d], 1),
        "plus_mean_S_embedding": np.concatenate([hc, d, sumz], 1),
        "plus_product_basis": np.concatenate([hc, d, sumz, prod], 1),
        "plus_product_and_cut_bond": np.concatenate([hc, d, sumz, prod, bond_oh], 1),
        "cut_bond_only": bond_oh,
    }
    res = {"protocol": cfg["protocol_id"],
           "note": ("descriptive diagnostic on the parent FACTOR representation; the "
                    "probe target is the standardised pair delta on Valid pairs"),
           "n_pairs_scored": int(both.sum()),
           "n_pairs_scored_note": ("every pair whose two molecules are outside the official "
                                   "Test split; grouped 5-fold CV by transform"),
           "embedding_dim": int(h.shape[1]),
           "pca_spectrum": [float(x) for x in
                            (np.linalg.svd(h - h.mean(0), compute_uv=False) ** 2
                             / max(1, h.shape[0] - 1))[:12]],
           "explained_variance_ratio_top8": [
               float(x) for x in (
                   (np.linalg.svd(h - h.mean(0), compute_uv=False) ** 2
                    / np.sum((h - h.mean(0)) ** 2))[:8])],
           "designs": oof_r2({k: v[both] for k, v in design.items()}, y[both],
                             groups[both]),
           "delta_std_units": float(np.std(y[both])),
           "interpretation_hint": ("if 'plus_product_basis' is materially above "
                                   "'compressed_side_embeddings_only', the compressed "
                                   "space does contain interaction signal that an "
                                   "additive head cannot reach"),
           "cache_content_sha256": pack["content_sha256"],
           "parent_checkpoint": str(C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0" / "state_best.pt")
           }
    C.write_json(C.results_root() / "compression_diagnostics.json", res)
    print(json.dumps(res["designs"], indent=1))
    print(f"-> {C.results_root() / 'compression_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
