"""CPU reference baselines: path-count ridge, path-count KRR, Morgan KRR, counts.

All features and vocabulary come from the current fit rows only.  Unknown path
columns get coefficient 0 in the ridge (unknown mass is reported) and the KRR
norm over evaluation molecules keeps the unseen key mass in the denominator
(that is what makes it the kernel of the same fixed feature map).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

import psc_common as C
import psc_data as D
import psc_paths as PP

OUT = C.ROUND_DIR / "results" / "cpu"
MORGAN_BINS = 4096


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


class PathFeatures:
    """Sparse path-count design matrix over the fit vocabulary."""

    def __init__(self, cache: PP.PathCache, fit_rows: Sequence[int], max_len: int,
                 with_counts: bool) -> None:
        vocab = D.fit_path_vocab(cache, fit_rows, max_len)
        self.ids = vocab.ids
        self.c_l = vocab.c_l
        self.max_len = int(max_len)
        self.with_counts = bool(with_counts)
        self.p = len(self.ids) + (int(max_len) + 1 if with_counts else 0)
        self.info = vocab.entropy

    def _matrix(self, cache: PP.PathCache, rows: Sequence[int]) -> sp.csr_matrix:
        pos = cache.positions(rows)
        starts = cache.mol_ptr[pos]
        ends = cache.mol_ptr[pos + 1]
        indptr = np.zeros(len(rows) + 1, np.int64)
        rows_idx: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
            ids = cache.ent_path[s:e]
            mult = cache.ent_mult[s:e]
            col = np.asarray([self.ids.get(int(p), -1) for p in ids.tolist()], np.int64)
            ok = col >= 0
            rows_idx.append(np.full(int(ok.sum()), i, np.int64))
            cols.append(col[ok])
            vals.append(mult[ok] * self.c_l[cache.path_len[ids[ok]]])
            indptr[i + 1] = indptr[i] + int(ok.sum())
        out = sp.csr_matrix((np.concatenate(vals) if vals else np.zeros(0),
                             (np.concatenate(rows_idx) if rows_idx else np.zeros(0, np.int64),
                              np.concatenate(cols) if cols else np.zeros(0, np.int64))),
                            shape=(len(rows), len(self.ids)))
        if self.with_counts:
            counts = cache.path_counts(rows).astype(np.float64) * self.c_l
            out = sp.hstack([out, sp.csr_matrix(counts)], format="csr")
        return out.astype(np.float64)

    def matrices(self, cache: PP.PathCache, fit_rows: Sequence[int],
                 eval_rows: Sequence[int]) -> dict[str, Any]:
        X_fit = self._matrix(cache, fit_rows)
        X_eval = self._matrix(cache, eval_rows)
        # unknown mass relative to the full eval path mass
        known = np.asarray(X_eval.sum(1)).ravel()
        full = np.zeros(len(eval_rows))
        pos = cache.positions(eval_rows)
        for i, (s, e) in enumerate(zip(cache.mol_ptr[pos].tolist(),
                                       cache.mol_ptr[pos + 1].tolist())):
            full[i] = float(np.sum(cache.ent_mult[s:e]
                                   * self.c_l[cache.path_len[cache.ent_path[s:e]]]))
        return {"X_fit": X_fit, "X_eval": X_eval,
                "unknown_mass_eval": float(1.0 - known.sum() / max(full.sum(), 1e-12)),
                "full_eval_mass": float(full.sum())}


def full_norm2(cache: PP.PathCache, rows: Sequence[int], c_l: np.ndarray) -> np.ndarray:
    """||x||^2 over *all* keys of the molecule, unseen ones included."""
    pos = cache.positions(rows)
    out = np.zeros(len(rows))
    for i, (s, e) in enumerate(zip(cache.mol_ptr[pos].tolist(),
                                   cache.mol_ptr[pos + 1].tolist())):
        mult = cache.ent_mult[s:e].astype(np.float64)
        cl = c_l[cache.path_len[cache.ent_path[s:e]]]
        out[i] = float(np.sum((mult * cl) ** 2))
    return out


def morgan_matrix(smiles: Sequence[str], radius: int) -> sp.csr_matrix:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator  # noqa: F401
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    rows, cols, vals = [], [], []
    for i, sm in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(sm))
        if mol is None:
            continue
        fp = AllChem.GetHashedMorganFingerprint(mol, int(radius), nBits=MORGAN_BINS,
                                                useChirality=True)
        nz = fp.GetNonzeroElements()
        for k, v in nz.items():
            rows.append(i)
            cols.append(int(k))
            vals.append(float(v))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(smiles), MORGAN_BINS),
                         dtype=np.float64)


def count_only_matrix(bundle: D.DataBundle, cache: PP.PathCache, rows: Sequence[int],
                      c_l: np.ndarray) -> np.ndarray:
    counts = cache.path_counts(rows).astype(np.float64) * c_l
    raw = bundle.raw
    atom = np.asarray(raw["graph_n_atoms"], np.float64)[rows]
    bond = np.asarray(raw["graph_n_bonds"], np.float64)[rows]
    cycle = np.asarray(raw["graph_cycle_rank"], np.float64)[rows]
    return np.column_stack([counts, atom, bond, cycle])


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def dual_ridge_predict(X_fit: sp.csr_matrix, X_eval: sp.csr_matrix, y_fit: np.ndarray,
                       alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    mean_fit = np.asarray(X_fit.mean(0)).ravel()
    m2 = float(mean_fit @ mean_fit)
    K = np.asarray((X_fit @ X_fit.T).todense())
    s_fit = np.asarray(X_fit @ mean_fit).ravel()
    Kc = K - s_fit[:, None] - s_fit[None, :] + m2
    Ke = np.asarray((X_eval @ X_fit.T).todense())
    s_eval = np.asarray(X_eval @ mean_fit).ravel()
    Kec = Ke - s_eval[:, None] - s_fit[None, :] + m2
    y_mean = float(np.mean(y_fit))
    yc = y_fit - y_mean
    a = np.linalg.solve(Kc + float(alpha) * np.eye(Kc.shape[0]), yc)
    pred = Kec @ a + y_mean
    cond = float(np.linalg.cond(Kc + float(alpha) * np.eye(Kc.shape[0])))
    return pred, {"cond": cond, "dual_dim": int(Kc.shape[0])}


def tanimoto_gram(A: sp.csr_matrix, B: sp.csr_matrix, nA: np.ndarray,
                  nB: np.ndarray) -> np.ndarray:
    dot = np.asarray((A @ B.T).todense())
    denom = nA[:, None] + nB[None, :] - dot
    return dot / np.maximum(denom, 1e-12)


def krr_predict(K_fit: np.ndarray, K_eval: np.ndarray, y_fit: np.ndarray,
                alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    y_mean = float(np.mean(y_fit))
    a = np.linalg.solve(K_fit + float(alpha) * np.eye(K_fit.shape[0]), y_fit - y_mean)
    return K_eval @ a + y_mean, {"cond": float(np.linalg.cond(
        K_fit + float(alpha) * np.eye(K_fit.shape[0])))}


def select_config(rows: Sequence[Mapping[str, Any]], tolerance: float = 1e-8
                  ) -> dict[str, Any]:
    """Minimum RMSE; within tolerance prefer larger alpha, then smaller radius,
    then lexicographic config."""
    best = min(float(r["rmse"]) for r in rows)
    cands = [r for r in rows if float(r["rmse"]) <= best + tolerance]
    cands.sort(key=lambda r: (-float(r["alpha"]), float(r.get("radius", 0)),
                              json.dumps(C.jsonable(r.get("config", {})), sort_keys=True)))
    return cands[0]


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def _maybe(fn, pred: np.ndarray, truth: np.ndarray):
    return None if not np.isfinite(truth).all() else float(fn(pred, truth))


def _save(name: str, payload: Mapping[str, Any], preds: Mapping[str, np.ndarray]) -> None:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    C.write_json(out / "manifest.json", payload)
    np.savez(out / "predictions.npz", **preds)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="D0", choices=("D0", "D1", "FULL"))
    ap.add_argument("--models", default="all")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--select-from", default="", help="D0 manifest to reuse hyper-parameters")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(int(args.threads)))
    cfg = C.load_config()
    L = int(args.max_len or cfg["paths"]["max_len"])
    bundle = D.load_bundle()
    f1 = None
    if args.split == "D1":
        f1 = json.loads(C.SPLIT_F1.read_text(encoding="utf-8"))
        f1["file_sha256"] = C.sha256_file(C.SPLIT_F1)
    split = C.split_spec(args.split, bundle.raw, f1=f1)
    fit_rows = np.asarray(split["fit_rows"], np.int64)
    eval_rows = np.asarray(split["eval_rows"], np.int64)
    y = np.asarray(bundle.labels[np.asarray(fit_rows)], np.float64)
    # official Valid labels stay sealed: FULL runs only produce predictions and
    # are scored exactly once by the frozen final evaluator.
    has_truth = split["name"] in ("D0", "D1")
    truth = (np.asarray(bundle.labels[eval_rows], np.float64) if has_truth
             else np.full(eval_rows.size, np.nan))
    root = np.union1d(bundle.official_train, bundle.official_valid)
    meta = PP.ensure_cache(bundle.raw, root, L)
    cache = PP.PathCache(meta, device="cpu")
    tag = args.tag or args.split
    models = args.models.split(",") if args.models != "all" else [
        "PATH_COUNT_RIDGE", "PATH_COUNT_KRR", "MORGAN_COUNT_KRR", "COUNT_ONLY_RIDGE"]
    fixed = {}
    if args.select_from:
        fixed = C.read_json(args.select_from)
    if split["name"] == "FULL" and not fixed:
        raise SystemExit("FULL runs require --select-from (hyper-parameters locked on D0)")
    rows_out: list[dict[str, Any]] = []
    t_all = time.time()
    for name in models:
        t0 = time.time()
        grid = []
        preds_by_cfg: dict[str, np.ndarray] = {}
        if name in ("PATH_COUNT_RIDGE", "PATH_COUNT_KRR"):
            feats = PathFeatures(cache, fit_rows, L, with_counts=(name == "PATH_COUNT_RIDGE"))
            mats = feats.matrices(cache, fit_rows, eval_rows)
            if name == "PATH_COUNT_RIDGE":
                ks = cfg["cpu"]["ridge_alpha"]
                for alpha in ks:
                    pred, info = dual_ridge_predict(mats["X_fit"], mats["X_eval"], y, alpha)
                    grid.append({"alpha": float(alpha), "rmse": C.rmse(pred, truth),
                                 "mae": C.mae(pred, truth), **info})
                    preds_by_cfg[f"alpha={alpha}"] = pred
            else:
                Xf, Xe = mats["X_fit"], mats["X_eval"]
                nf = np.asarray((Xf.multiply(Xf)).sum(1)).ravel()
                ne = full_norm2(cache, eval_rows, feats.c_l)
                Kf = tanimoto_gram(Xf, Xf, nf, nf)
                Ke = tanimoto_gram(Xe, Xf, ne, nf)
                for alpha in cfg["cpu"]["krr_alpha"]:
                    pred, info = krr_predict(Kf, Ke, y, alpha)
                    grid.append({"alpha": float(alpha), "rmse": C.rmse(pred, truth),
                                 "mae": C.mae(pred, truth),
                                 "symmetry": float(np.max(np.abs(Kf - Kf.T))),
                                 "self_sim_min": float(np.min(np.diag(Kf))), **info})
                    preds_by_cfg[f"alpha={alpha}"] = pred
            extra = {"features": feats.info, "p": int(feats.p),
                     "unknown_mass_eval": mats["unknown_mass_eval"]}
        elif name == "MORGAN_COUNT_KRR":
            smiles_fit = np.asarray(bundle.raw["smiles"])[fit_rows]
            smiles_eval = np.asarray(bundle.raw["smiles"])[eval_rows]
            grids = {}
            for radius in cfg["cpu"]["morgan_radius"]:
                Xf = morgan_matrix(smiles_fit, radius)
                Xe = morgan_matrix(smiles_eval, radius)
                nf = np.asarray((Xf.multiply(Xf)).sum(1)).ravel()
                ne = np.asarray((Xe.multiply(Xe)).sum(1)).ravel()
                Kf = tanimoto_gram(Xf, Xf, nf, nf)
                Ke = tanimoto_gram(Xe, Xf, ne, nf)
                for alpha in cfg["cpu"]["krr_alpha"]:
                    pred, info = krr_predict(Kf, Ke, y, alpha)
                    grids[f"radius={radius},alpha={alpha}"] = pred
                    grid.append({"radius": int(radius), "alpha": float(alpha),
                                 "rmse": C.rmse(pred, truth),
                                 "mae": C.mae(pred, truth), **info})
            preds_by_cfg = grids
            extra = {"radius": cfg["cpu"]["morgan_radius"], "bins": MORGAN_BINS,
                     "chirality": True,
                     "rdkit": __import__("rdkit").__version__}
        elif name == "COUNT_ONLY_RIDGE":
            Xf = sp.csr_matrix(count_only_matrix(bundle, cache, fit_rows, cache_c_l(cache, fit_rows)))
            Xe = sp.csr_matrix(count_only_matrix(bundle, cache, eval_rows, cache_c_l(cache, fit_rows)))
            for alpha in cfg["cpu"]["ridge_alpha"]:
                pred, info = dual_ridge_predict(Xf, Xe, y, alpha)
                grid.append({"alpha": float(alpha), "rmse": C.rmse(pred, truth),
                             "mae": C.mae(pred, truth), **info})
                preds_by_cfg[f"alpha={alpha}"] = pred
            extra = {"p": int(Xf.shape[1]), "features": "path counts by length + atom/bond/"
                                                      "cycle counts"}
        else:
            raise ValueError(name)
        if fixed:
            f = fixed[name]
            key = f.get("config_key")
            chosen = preds_by_cfg[key]
            sel = {"alpha": f["alpha"], "radius": f.get("radius", 0),
                   "config_key": key,
                   "rmse": _maybe(C.rmse, chosen, truth),
                   "mae": _maybe(C.mae, chosen, truth)}
            locked = f
        else:
            sel = select_config(grid)
            key = (f"radius={sel.get('radius')},alpha={sel['alpha']}" if "radius" in sel
                   else f"alpha={sel['alpha']}")
            chosen = preds_by_cfg[key]
            locked = {"alpha": sel["alpha"], "radius": sel.get("radius", 0),
                      "config_key": key, "rmse": sel["rmse"], "mae": sel["mae"],
                      "selection": "D0 minimum RMSE; ties prefer larger alpha, smaller "
                                   "radius, lexicographic config"}
        wall = time.time() - t0
        run_id = f"CPU_{name}_{tag}"
        payload = {
            "run_id": run_id, "model": name, "dataset": "ogbg-mollipo",
            "split": {k: split[k] for k in ("name", "sha256", "role")},
            "n_fit": int(fit_rows.size), "n_eval": int(eval_rows.size),
            "grid": grid, "n_configs": len(grid), "selected": locked,
            "eval_rmse": _maybe(C.rmse, chosen, truth),
            "eval_mae": _maybe(C.mae, chosen, truth),
            "eval_rmse_selected": sel["rmse"], "eval_mae_selected": sel["mae"],
            "labels_available_locally": has_truth,
            "wall_seconds": float(wall), "cpu_threads": int(args.threads),
            "train_labels_sha256": bundle.hashes["train_labels"]["sha256"],
            "path_cache_sha256": meta["cache_sha256"],
            "max_len": L, "environment": C.environment(),
            "provenance": C.code_provenance(), "argv": list(argv or []),
            "selection_split": "D0" if not fixed else f"locked from {args.select_from}",
            "extra": C.jsonable(extra),
            "finished_utc": C.now_utc(),
        }
        saved = {"row_id": bundle.row_id[eval_rows], "pred": chosen}
        if has_truth:
            saved["label"] = truth
        _save(run_id, payload, saved)
        rows_out.append({"run_id": run_id, "model": name, "split": split["name"],
                         "seed": "", "rmse": payload["eval_rmse"],
                         "mae": payload["eval_mae"],
                         "rmse_selected": sel["rmse"], "n_configs": len(grid),
                         "wall_seconds": wall})
        print(f"{run_id}: rmse={payload['eval_rmse']} mae={payload['eval_mae']} "
              f"selected={locked} configs={len(grid)} wall={wall:.1f}s")
    C.write_csv(OUT / f"cpu_{tag}.csv", rows_out)
    print(f"total cpu wall={time.time() - t_all:.1f}s")
    return 0


def cache_c_l(cache: PP.PathCache, fit_rows: Sequence[int]) -> np.ndarray:
    return PP.c_l_from_cache(cache, fit_rows)


if __name__ == "__main__":
    raise SystemExit(main())