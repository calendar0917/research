"""Performance-floor baselines on the candidate word space and on fingerprints.

``COUNT_RIDGE``
    exact attribute-word counts (rooted template + all ordered 9/3 attribute
    columns, minimised over the frames of the word) -> column-RMS-scaled ridge
    with an unpenalised intercept, fixed alpha grid, Valid-selected.

``FP_KRR``
    RDKit Morgan radius 2 / 2048 bits / ``useChirality=True``, binary Tanimoto
    kernel, fixed alpha grid, Valid-selected.

Both use **Train-only** vocabulary / scaler / kernel and record all three
splits for every alpha.  They have no epochs: one fit node = three split rows.
The ridge solve is exact (Cholesky of the primal or dual normal equations), so
there is no "LSQR did not converge" failure mode; the relative normal-equation
residual is recorded anyway.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

import swf_common as C
import swf_cache as K


# ---------------------------------------------------------------------------
# exact attribute-word keys and counts
# ---------------------------------------------------------------------------


def win_u16_matrix(base: Mapping[str, Any]) -> np.ndarray:
    """``[Uw, 11]`` exact ordered attribute record behind each dedup id."""
    n = int(np.asarray(base["win_tmpl"]).size)
    out = np.zeros((n, 11), np.int32)
    out[:, 0] = np.asarray(base["win_tmpl"]).astype(np.int32)
    out[:, 1:5] = np.asarray(base["win_atom_idx"]).astype(np.int32)
    bond = np.asarray(base["win_bond_idx"]).astype(np.int64)
    out[:, 5:11] = np.where(bond < 0, 0xFFFF, bond).astype(np.int32)
    return out


def canonical_word_keys(base: Mapping[str, Any]) -> np.ndarray:
    """Lexicographically minimal frame record per word (frame-invariant key)."""
    win_u16 = win_u16_matrix(base)
    frame_win = np.asarray(base["frame_win"], np.int64)
    word_frame_ptr = np.asarray(base["word_frame_ptr"], np.int64)
    nw = int(word_frame_ptr.size - 1)
    rows = win_u16[frame_win]                                    # [F, 11]
    word_of_frame = np.repeat(np.arange(nw, dtype=np.int64), np.diff(word_frame_ptr))
    order = np.lexsort(tuple([rows[:, j] for j in range(10, -1, -1)] + [word_of_frame]))
    ordered_words = word_of_frame[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = ordered_words[1:] != ordered_words[:-1]
    picked = order[first]
    out = np.zeros((nw, 11), np.int32)
    out[word_of_frame[picked]] = rows[picked]
    return out


def word_count_matrix(cache: K.Cache):
    """Sparse ``[n_molecules, n_unique_word_keys]`` exact attribute-word counts."""
    import scipy.sparse as sp

    base = cache.base
    keys = canonical_word_keys(base)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    word_mol = np.repeat(np.arange(cache.n_mol, dtype=np.int64), np.diff(cache.mol_word_ptr))
    mat = sp.csr_matrix(
        (np.ones(inv.size, np.float64), (word_mol, inv.astype(np.int64))),
        shape=(cache.n_mol, int(uniq.shape[0])),
    )
    df_all = np.asarray((mat[cache.split_rows["train"]] > 0).sum(axis=0)).reshape(-1).astype(np.int64)
    return {"matrix": mat, "keys": uniq, "inverse": inv, "df": df_all, "word_mol": word_mol}


# ---------------------------------------------------------------------------
# exact ridge (centred, unpenalised intercept)
# ---------------------------------------------------------------------------


def _solve_ridge(Xtr, ytr_c: np.ndarray, alpha: float) -> dict[str, Any]:
    import scipy.linalg as sla

    n, v = Xtr.shape
    if v <= n:
        G = (Xtr.T @ Xtr).toarray()
        b = np.asarray(Xtr.T @ ytr_c).reshape(-1)
        A = G + float(alpha) * np.eye(v)
        w = sla.cho_solve(sla.cho_factor(A, lower=True), b)
        resid = float(np.linalg.norm(A @ w - b) / max(1e-30, np.linalg.norm(b)))
        return {"w": w, "relative_residual": resid, "form": "primal", "dim": int(v)}
    gram = (Xtr @ Xtr.T).toarray()
    A = gram + float(alpha) * np.eye(n)
    a = sla.cho_solve(sla.cho_factor(A, lower=True), ytr_c)
    w = np.asarray(Xtr.T @ a).reshape(-1)
    resid = float(np.linalg.norm(A @ a - ytr_c) / max(1e-30, np.linalg.norm(ytr_c)))
    return {"w": w, "relative_residual": resid, "form": "dual", "dim": int(n)}


def count_ridge(*, cfg: Mapping[str, Any], cache: K.Cache) -> dict[str, Any]:
    import scipy.sparse as sp

    params = cfg["count_ridge"]
    scaler = cache.scaler
    rows_of = {split: cache.split_rows[split] for split in C.SPLITS}
    y_std = (cache.labels - float(scaler["mean"])) / float(scaler["std"])

    built = word_count_matrix(cache)
    full = built["matrix"]
    df = built["df"]
    keep = np.where(df >= int(params["min_df"]))[0]
    X = full[:, keep].tocsr()
    V = int(X.shape[1])

    extra = np.concatenate(
        [np.log1p(cache.n_atoms.astype(np.float64))[:, None],
         np.log1p(cache.n_bonds.astype(np.float64))[:, None]], axis=1
    )

    def build(rows: np.ndarray):
        Xr = X[rows]
        logX = Xr.copy()
        logX.data = np.log1p(logX.data)
        return sp.hstack([Xr, logX, sp.csr_matrix(extra[rows])], format="csr")

    mats = {split: build(rows_of[split]) for split in C.SPLITS}
    Xtr = mats["train"]
    sq = np.asarray(Xtr.multiply(Xtr).sum(axis=0)).reshape(-1) / max(1, Xtr.shape[0])
    rms = np.maximum(np.sqrt(sq), float(params["rms_floor"]))
    mats = {s: (m @ sp.diags(1.0 / rms)).tocsr() for s, m in mats.items()}
    Xtr = mats["train"]

    mean = np.asarray(Xtr.mean(axis=0)).reshape(-1)
    y_mean = float(y_std[rows_of["train"]].mean())
    y_c = y_std[rows_of["train"]] - y_mean

    # OOV rate: share of word instances whose exact attribute key is unseen / df<2
    in_keep = np.zeros(int(df.size), dtype=bool)
    in_keep[keep] = True
    oov = {}
    for split in C.SPLITS:
        mask = np.isin(built["word_mol"], rows_of[split])
        oov[split] = float(1.0 - in_keep[built["inverse"][mask]].mean()) if mask.any() else float("nan")

    results = []
    for alpha in params["alphas"]:
        t0 = time.perf_counter()
        sol = _solve_ridge(Xtr, y_c, float(alpha))
        w = sol["w"]
        intercept = y_mean - float(mean @ w)
        per_split: dict[str, dict[str, float]] = {}
        for split in C.SPLITS:
            rows = rows_of[split]
            pred_std = np.asarray(mats[split] @ w).reshape(-1) + intercept
            pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
            truth = cache.labels[rows]
            per_split[split] = {"rmse": C.rmse(pred_raw, truth), "mae": C.mae(pred_raw, truth)}
        results.append({
            "alpha": float(alpha),
            "form": sol["form"],
            "solve_dim": sol["dim"],
            "relative_residual": sol["relative_residual"],
            "converged": bool(sol["relative_residual"] < 1e-8),
            "fit_seconds": time.perf_counter() - t0,
            "splits": per_split,
        })
        print(
            f"[COUNT_RIDGE] alpha={alpha:g} valid={per_split['valid']['rmse']:.4f} "
            f"test={per_split['test']['rmse']:.4f} ({sol['form']} {sol['dim']}, "
            f"res={sol['relative_residual']:.1e})",
            flush=True,
        )
    converged = [r for r in results if r["converged"]]
    chosen = min(converged, key=lambda r: (r["splits"]["valid"]["rmse"], -r["alpha"]))
    return {
        "method": "COUNT_RIDGE",
        "n_unique_attribute_words": int(df.size),
        "n_keys_kept": V,
        "min_df": int(params["min_df"]),
        "n_columns": int(V * 2 + 2),
        "design_nnz_train": int(Xtr.nnz),
        "oov_rate": oov,
        "alphas": results,
        "selected_alpha": chosen["alpha"],
        "selected": chosen["splits"],
        "selected_converged": chosen["converged"],
        "scaler": dict(scaler),
        "feature_layout": "[counts, log1p(counts), log1p(n_atoms), log1p(n_bonds)] / Train RMS",
    }


# ---------------------------------------------------------------------------
# fingerprint kernel ridge
# ---------------------------------------------------------------------------


def morgan_bits(cfg: Mapping[str, Any]) -> tuple[np.ndarray, list[int]]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    import swf_data as D

    RDLogger.DisableLog("rdApp.*")
    raw = D.load_raw_cache(cfg)
    smiles = [str(s) for s in np.asarray(raw["smiles"]).tolist()]
    params = cfg["fp_krr"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(params["radius"]), fpSize=int(params["n_bits"])
    )
    bits = np.zeros((len(smiles), int(params["n_bits"])), np.float32)
    failed: list[int] = []
    for i, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text.strip())
        if mol is None:
            failed.append(i)
            continue
        fp = generator.GetFingerprint(mol)
        bits[i] = np.frombuffer(fp.ToBitString().encode("ascii"), np.uint8) - ord("0")
    return bits, failed


def tanimoto_kernel(bits: np.ndarray) -> np.ndarray:
    inter = bits.astype(np.float64) @ bits.astype(np.float64).T
    pop = bits.sum(axis=1).astype(np.float64)
    union = pop[:, None] + pop[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def fp_krr(*, cfg: Mapping[str, Any], cache: K.Cache) -> dict[str, Any]:
    import scipy.linalg as sla

    params = cfg["fp_krr"]
    scaler = cache.scaler
    rows_of = {split: cache.split_rows[split] for split in C.SPLITS}
    bits, failed = morgan_bits(cfg)
    if failed:
        raise RuntimeError(f"RDKit failed to parse {len(failed)} official molecules: {failed[:5]}")
    gram = tanimoto_kernel(bits)
    train_rows = rows_of["train"]
    y_std = (cache.labels - float(scaler["mean"])) / float(scaler["std"])
    ytr = y_std[train_rows]
    Ktt = gram[np.ix_(train_rows, train_rows)]
    results = []
    for alpha in params["alphas"]:
        t0 = time.perf_counter()
        A = Ktt + float(alpha) * np.eye(train_rows.size)
        a = sla.cho_solve(sla.cho_factor(A, lower=True), ytr)
        resid = float(np.linalg.norm(A @ a - ytr) / max(1e-30, np.linalg.norm(ytr)))
        per_split: dict[str, dict[str, float]] = {}
        for split in C.SPLITS:
            rows = rows_of[split]
            pred_std = gram[np.ix_(rows, train_rows)] @ a
            pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
            truth = cache.labels[rows]
            per_split[split] = {"rmse": C.rmse(pred_raw, truth), "mae": C.mae(pred_raw, truth)}
        results.append({
            "alpha": float(alpha),
            "relative_residual": resid,
            "converged": bool(resid < 1e-8),
            "fit_seconds": time.perf_counter() - t0,
            "splits": per_split,
        })
        print(
            f"[FP_KRR] alpha={alpha:g} valid={per_split['valid']['rmse']:.4f} "
            f"test={per_split['test']['rmse']:.4f}", flush=True,
        )
    chosen = min(results, key=lambda r: (r["splits"]["valid"]["rmse"], -r["alpha"]))
    return {
        "method": "FP_KRR",
        "radius": int(params["radius"]),
        "n_bits": int(params["n_bits"]),
        "use_chirality": bool(params["use_chirality"]),
        "kernel": "binary Tanimoto",
        "n_fit": int(train_rows.size),
        "alphas": results,
        "selected_alpha": chosen["alpha"],
        "selected": chosen["splits"],
        "selected_converged": chosen["converged"],
        "scaler": dict(scaler),
    }
