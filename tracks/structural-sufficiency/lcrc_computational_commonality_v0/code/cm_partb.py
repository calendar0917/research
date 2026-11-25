"""Part B: extract the frozen typed-S2 function, reconstruct it, replace it.

The frozen teacher is the W2 seed-0 best-Validation checkpoint.  Exactly three
local-MLP input blocks are touched (``s2_type``, ``s2_end``, ``s2_joint``):

    T(p, e) = W_type @ path_emb(p) + W_end @ end_emb(e)
              + W_joint @ joint_proj(path_u(p) * path_v(e))

is a function of the *type pair* alone, so it is tabulated exactly.  Every other
block (atom content, incident bonds, S1, S2 closure, S2 endpoint content,
path_gate, atom scalars, LayerNorm, the rest of the local MLP, all readouts) is
left untouched.

Stages (``uv run python code/cm_partb_run.py <stage>``)::

    stats    Train type occurrence / document frequency / component support
    fit      MEAN / CODE_ADD / SEM_1 / SEM_2 / NULL_1 / NULL_2 ridge fits
    verify   identity reconstruction of preactivation, h0 and logits
    replace  real forward replacement, official Valid/Test scoring
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import cm_common as C

N_ATOM = 16
N_BOND = 5
N_PATH = N_ATOM * N_BOND * N_BOND  # 400
N_END = N_ATOM * N_BOND  # 80
N_KEYS = N_PATH * N_END  # 32_000
RANK_LOCAL = 64
LOCAL_CHUNK = 20_000
LAMBDA = 1e-4
ALPHA_SHRINK = 100.0
FIT_KINDS = ("MEAN", "CODE_ADD", "SEM_1", "SEM_2", "NULL_1", "NULL_2")
REPLACE_ORDER = ("MEAN", "CODE_ADD", "SEM_1", "SEM_2", "NULL_2")

LOCAL = C.LOCAL / "partb"
STATS_PATH = LOCAL / "train_type_stats.npz"
FIT_PATH = LOCAL / "structural_function_fits.npz"
FIT_SUMMARY = LOCAL / "fit_summary.json"
TABLE_PATH = LOCAL / "replacement_tables.npz"
VERIFY_PATH = LOCAL / "identity_verification.json"


# ---------------------------------------------------------------------------
# teacher model and exact tables
# ---------------------------------------------------------------------------


def load_teacher(device: str = "cpu"):
    import bs_common as BC

    task = C.DC.SD.load_task()
    cfg = dict(BC.config_for_scale(2))
    model, audit = C.DM.build_w2(task.spec, task.norm, C.N_TASKS, seed=0, device="cpu")
    payload = torch.load(C.MODELS["W2_s0"]["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    import tu_model as TU

    layout = TU.local_layout(task.spec, "C1", cfg)
    return model, task, {"cfg": cfg, "layout": layout, "epoch": int(payload.get("epoch", -1)),
                         "checkpoint_sha256": C.sha256_file(C.MODELS["W2_s0"]["checkpoint"])}


def _blocks(local: torch.nn.Module, layout: Mapping[str, tuple[int, int]]) -> dict[str, torch.Tensor]:
    W = local.mlp.lins[0].weight.detach()
    return {name: W[:, int(layout[name][0]) : int(layout[name][1])].contiguous()
            for name in ("s2_type", "s2_end", "s2_joint")}


def teacher_tables(model: torch.nn.Module, layout: Mapping[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    local = model.local
    Wb = {k: v.double() for k, v in _blocks(local, layout).items()}
    path_emb = local.path_emb.weight.detach().double()
    end_emb = local.end_emb.weight.detach().double()
    path_u = local.path_u.weight.detach().double()
    path_v = local.path_v.weight.detach().double()
    jw = local.joint_proj.weight.detach().double()
    jb = local.joint_proj.bias.detach().double()
    t_type = path_emb @ Wb["s2_type"].T
    t_end = end_emb @ Wb["s2_end"].T
    prod = (path_u[:, None, :] * path_v[None, :, :]).reshape(N_KEYS, -1)
    t_joint3 = ((prod @ jw.T + jb) @ Wb["s2_joint"].T).reshape(N_PATH, N_END, -1)
    t_total = (t_type[:, None, :].numpy() + t_end[None, :, :].numpy() + t_joint3.numpy()).reshape(N_KEYS, -1)
    return {"T_type": t_type.numpy(), "T_end": t_end.numpy(),
            "T_joint": t_joint3.reshape(N_KEYS, -1).numpy(), "T_total": t_total}


# ---------------------------------------------------------------------------
# Train statistics over real typed-S2 paths
# ---------------------------------------------------------------------------


def _row_intervals(ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(ptr)[np.asarray(rows, dtype=np.int64)]
    q = np.asarray(ptr)[np.asarray(rows, dtype=np.int64) + 1]
    return p.astype(np.int64), q.astype(np.int64)


def path_keys(cache: Mapping[str, Any], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(keys, molecule_index) for every real typed-S2 path of ``rows``."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    a_lo, a_hi = _row_intervals(atom_ptr, rows)
    atom_counts = a_hi - a_lo
    if atom_counts.sum() == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int64)
    atom_idx = np.concatenate([np.arange(lo, hi, dtype=np.int64) for lo, hi in zip(a_lo, a_hi)])
    p_lo = path_ptr[atom_idx]
    p_counts = (path_ptr[atom_idx + 1] - p_lo).astype(np.int64)
    total = int(p_counts.sum())
    atom_mol = np.repeat(np.arange(len(rows), dtype=np.int64), atom_counts)
    path_mol = np.repeat(atom_mol, p_counts)
    starts = np.concatenate([[0], np.cumsum(p_counts)])[:-1]
    offsets = np.arange(total, dtype=np.int64) - np.repeat(starts, p_counts)
    path_idx = np.repeat(p_lo, p_counts) + offsets
    cat = np.asarray(cache["path_cat"])[path_idx].astype(np.int32)
    end = np.asarray(cache["path_end63"])[path_idx].astype(np.int32)
    return (cat * np.int32(N_END) + end).astype(np.int32), path_mol


def train_stats(cache: Mapping[str, Any], train_rows: np.ndarray) -> dict[str, Any]:
    occ = np.zeros(N_KEYS, dtype=np.int64)
    doc = np.zeros(N_KEYS, dtype=np.int64)
    bond_dup_ok = True
    n_paths = 0
    for start in range(0, len(train_rows), LOCAL_CHUNK):
        chunk = np.asarray(train_rows[start : start + LOCAL_CHUNK], dtype=np.int64)
        keys, mol = path_keys(cache, chunk)
        if keys.size == 0:
            continue
        occ += np.bincount(keys, minlength=N_KEYS)
        order = np.lexsort((keys, mol))
        ks, ms = keys[order], mol[order]
        new = np.ones(ks.size, dtype=bool)
        new[1:] = (ks[1:] != ks[:-1]) | (ms[1:] != ms[:-1])
        doc += np.bincount(ks[new], minlength=N_KEYS)
        n_paths += int(keys.size)
        bond_dup_ok = bond_dup_ok and bool(
            np.array_equal(ks % np.int32(N_BOND), (ks // np.int32(N_END)) % np.int32(N_BOND))
        )
    return {"occurrence": occ, "doc_freq": doc, "observed": occ > 0,
            "n_molecules_scanned": int(len(train_rows)), "s2_paths": n_paths,
            "bond2_field_duplicated_in_end_code": bool(bond_dup_ok)}


def components(keys: np.ndarray) -> np.ndarray:
    k = np.asarray(keys, dtype=np.int64)
    p, e = k // N_END, k % N_END
    return np.stack([p // 25, (p % 25) // 5, p % 5, e // 5], axis=-1).astype(np.int64)


COMP_NAMES = ("middle_atom", "bond1", "bond2", "end_atom")
COMP_SIZES = (N_ATOM, N_BOND, N_BOND, N_ATOM)


# ---------------------------------------------------------------------------
# features / ridge
# ---------------------------------------------------------------------------


def _onehot(x: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((x.size, size), dtype=np.float64)
    out[np.arange(x.size), x] = 1.0
    return out


def design(kind: str, keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    n = keys.size
    if kind == "MEAN":
        return np.zeros((n, 0), dtype=np.float64)
    if kind == "CODE_ADD":
        return np.concatenate([_onehot(keys // N_END, N_PATH), _onehot(keys % N_END, N_END)], axis=1)
    comp = components(keys)
    cols = [_onehot(comp[:, 0], N_ATOM), _onehot(comp[:, 1], N_BOND),
            _onehot(comp[:, 2], N_BOND), _onehot(comp[:, 3], N_ATOM)]
    if kind == "SEM_1":
        return np.concatenate(cols, axis=1)
    for a in range(4):
        for b in range(a + 1, 4):
            cols.append((cols[a][:, :, None] * cols[b][:, None, :]).reshape(n, -1))
    return np.concatenate(cols, axis=1)


def bucket_of(doc: np.ndarray) -> np.ndarray:
    return np.floor(np.log2(np.maximum(np.asarray(doc, dtype=np.float64), 1.0))).astype(np.int64)


def null_mapping(keys: np.ndarray, doc: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed random bijection inside log2(document-frequency) buckets.

    Returns the description keys assigned to every input key (same shape/order)
    plus a diagnosis of the intervention strength.
    """
    keys = np.asarray(keys, dtype=np.int64)
    bucket = bucket_of(doc[keys])
    rng = np.random.default_rng(int(seed))
    mapped = np.empty_like(keys)
    moved = 0
    n_bucket = 0
    small = 0
    order = np.lexsort((keys, bucket))
    bounds = np.flatnonzero(np.diff(bucket[order])) + 1
    for grp in np.split(order, bounds):
        if grp.size == 0:
            continue
        n_bucket += 1
        if grp.size < 2:
            small += 1
            mapped[grp] = keys[grp]
            continue
        draw = rng.permutation(grp.size)
        src = np.asarray(keys)[grp][draw]
        mapped[grp] = src
        moved += int((src != np.asarray(keys)[grp]).sum())
    return mapped, {"seed": int(seed), "n_keys": int(keys.size), "n_buckets": int(n_bucket),
                    "n_singleton_buckets": int(small), "n_keys_moved": int(moved),
                    "fraction_keys_moved": float(moved / max(keys.size, 1))}


def ridge_fit(X: np.ndarray, Y: np.ndarray, w: np.ndarray, *, lam: float = LAMBDA) -> dict[str, Any]:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = w / w.mean()
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    ybar = (w[:, None] * Y).sum(0) / w.sum()
    if X.shape[1] == 0:
        return {"beta": np.zeros((0, Y.shape[1])), "intercept": ybar, "rank": 0, "n_free_coefficients": 0}
    xbar = (w[:, None] * X).sum(0) / w.sum()
    Xc, Yc = X - xbar, Y - ybar
    XtWX = (Xc * w[:, None]).T @ Xc / w.sum()
    XtWY = (Xc * w[:, None]).T @ Yc / w.sum()
    beta = np.linalg.solve(XtWX + lam * np.eye(X.shape[1]), XtWY)
    return {"beta": beta, "intercept": ybar - xbar @ beta,
            "rank": int(np.linalg.matrix_rank(Xc)), "n_free_coefficients": int(X.shape[1])}


def predict(fit: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(fit["beta"]) + np.asarray(fit["intercept"])[None, :]


def recon_metrics(Y: np.ndarray, Yhat: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    Y = np.asarray(Y, dtype=np.float64)
    Yhat = np.asarray(Yhat, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = w / w.mean()
    res = Yhat - Y
    num = float((w * (res ** 2).sum(1)).sum())
    den = float((w * (Y ** 2).sum(1)).sum())
    yn, yhn = np.linalg.norm(Y, axis=1), np.linalg.norm(Yhat, axis=1)
    good = (yn > 0) & (yhn > 0)
    cos = float(np.mean((Y[good] * Yhat[good]).sum(1) / (yn[good] * yhn[good]))) if good.any() else None
    return {"relative_rms": float(np.sqrt(num / den)) if den > 0 else None,
            "rms": float(np.sqrt(num / (w.sum() * Y.shape[1]))),
            "target_rms": float(np.sqrt(den / (w.sum() * Y.shape[1]))),
            "cosine_mean": cos, "n_evaluated": int(Y.shape[0])}


def hash_order(keys: np.ndarray) -> np.ndarray:
    return np.asarray([int(hashlib.sha256(str(int(k)).encode()).hexdigest()[:12], 16) for k in keys], dtype=np.int64)


# ---------------------------------------------------------------------------
# fit stage
# ---------------------------------------------------------------------------


def run_fit(*, holdout_fraction: float = 0.2, min_doc: int = 3) -> dict[str, Any]:
    LOCAL.mkdir(parents=True, exist_ok=True)
    st = dict(np.load(STATS_PATH, allow_pickle=False))
    tab = dict(np.load(TABLE_PATH, allow_pickle=False)) if TABLE_PATH.is_file() else None
    if tab is None:
        raise RuntimeError("replacement/tables must exist before fit")
    occ, doc = st["occurrence"], st["doc_freq"]
    keys = np.flatnonzero(occ > 0).astype(np.int64)
    comp = components(keys)

    # ---- fixed type-level hold-out (>= min_doc Train molecules) -----------
    eligible = keys[doc[keys] >= min_doc]
    bucket = bucket_of(doc[eligible])
    ho = np.zeros(eligible.size, dtype=bool)
    order = np.lexsort((hash_order(eligible), bucket))
    bounds = np.flatnonzero(np.diff(bucket[order])) + 1
    for grp in np.split(order, bounds):
        if grp.size == 0:
            continue
        need = max(1, int(round(holdout_fraction * grp.size))) if grp.size >= 5 else 0
        ho[grp[np.argsort(hash_order(eligible)[grp])[:need]]] = True
    holdout_keys = eligible[ho]
    fit_keys = np.setdiff1d(keys, holdout_keys)
    fit_mask = np.zeros(N_KEYS, dtype=bool)
    fit_mask[fit_keys] = True
    ho_mask = np.zeros(N_KEYS, dtype=bool)
    ho_mask[holdout_keys] = True

    seen = [set(comp[fit_mask[keys], a].tolist()) for a in range(4)]
    ho_comp = components(holdout_keys)
    comp_seen = np.stack([np.isin(ho_comp[:, a], list(seen[a])) for a in range(4)], axis=1)
    subset_all_seen = ho_comp[np.all(comp_seen, axis=1)]
    subset_any_unseen = ho_comp[~np.all(comp_seen, axis=1)]

    # ---- null descriptions ------------------------------------------------
    null_keys_all, null_diag_all = null_mapping(keys, doc, seed=0)
    null_map_all = dict(zip(keys.tolist(), null_keys_all.tolist()))
    null_keys_fit, null_diag_fit = null_mapping(fit_keys, doc, seed=0)
    null_keys_ho, null_diag_ho = null_mapping(holdout_keys, doc, seed=0)

    targets = {"T_total": np.asarray(tab["T_total"], dtype=np.float64),
               "T_joint": np.asarray(tab["T_joint"], dtype=np.float64)}
    weight_schemes = {"type_equal": np.ones(keys.size),
                      "occurrence": occ[keys].astype(np.float64),
                      "molecule_support": doc[keys].astype(np.float64)}

    fits: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {"schema": "cc-structural-function-fit-v1", "created_utc": C.utcnow(),
                              "lambda": LAMBDA, "dtype": "float64", "intercept_penalised": False,
                              "n_types_observed": int(keys.size), "n_fit_types": int(fit_keys.size),
                              "n_holdout_types": int(holdout_keys.size),
                              "holdout": {"min_doc": min_doc, "fraction": holdout_fraction,
                                          "subset_all_components_seen": int(subset_all_seen.shape[0]),
                                          "subset_any_component_unseen": int(subset_any_unseen.shape[0])},
                              "null": {"all": null_diag_all, "fit": null_diag_fit, "holdout": null_diag_ho},
                              "full_known_types": {}, "type_holdout": {}, "T_joint_only": {}}
    null_map = {"keys": keys.astype(np.int32), "null_keys": null_keys_all.astype(np.int32),
                "fit_keys": fit_keys.astype(np.int32), "fit_null_keys": null_keys_fit.astype(np.int32),
                "holdout_keys": holdout_keys.astype(np.int32), "holdout_null_keys": null_keys_ho.astype(np.int32)}

    # ---- 12.1 full known-type fit ----------------------------------------
    for target_name, Y_all in targets.items():
        block = {} if target_name == "T_joint" else summary["full_known_types"]
        for kind in FIT_KINDS:
            for wname, wvec in weight_schemes.items():
                if target_name == "T_joint" and wname != "type_equal":
                    continue
                use_null = kind.startswith("NULL")
                base = kind.split("_")[0] if False else kind
                kind_design = "SEM_1" if kind in ("SEM_1", "NULL_1") else ("SEM_2" if kind in ("SEM_2", "NULL_2") else kind)
                desc_keys = null_keys_all if use_null else keys
                X = design(kind_design, desc_keys)
                fit = ridge_fit(X, Y_all[keys], wvec)
                pred = predict(fit, X)
                rec = recon_metrics(Y_all[keys], pred, wvec)
                rec.update({"kind": kind, "weighting": wname, "design": kind_design,
                            "design_rank": fit["rank"], "n_free_coefficients": fit["n_free_coefficients"]})
                block.setdefault(kind, {})[wname] = rec
                if target_name == "T_total" and wname == "type_equal":
                    fits[f"{kind}|T_total|full"] = pred
                if target_name == "T_joint" and wname == "type_equal":
                    fits[f"{kind}|T_joint|full"] = pred
                    summary["T_joint_only"].setdefault(kind, {})["type_equal"] = rec
        if target_name == "T_total":
            summary["full_known_types"] = block

    # ---- 12.2 type hold-out ----------------------------------------------
    fold = {}
    for target_name, Y_all in targets.items():
        block = {}
        for kind in FIT_KINDS:
            use_null = kind.startswith("NULL")
            kind_design = "SEM_1" if kind in ("SEM_1", "NULL_1") else ("SEM_2" if kind in ("SEM_2", "NULL_2") else kind)
            Xf = design(kind_design, null_keys_fit if use_null else fit_keys)
            Xh = design(kind_design, null_keys_ho if use_null else holdout_keys)
            fit = ridge_fit(Xf, Y_all[fit_keys], np.ones(fit_keys.size))
            rec = recon_metrics(Y_all[holdout_keys], predict(fit, Xh), np.ones(holdout_keys.size))
            rec.update({"kind": kind, "design": kind_design, "design_rank": fit["rank"],
                        "n_free_coefficients": fit["n_free_coefficients"]})
            block.setdefault(kind, {})["holdout_all"] = rec
            if target_name == "T_total":
                if subset_all_seen.shape[0]:
                    sel = np.all(comp_seen, axis=1)
                    block[kind]["holdout_components_seen"] = recon_metrics(
                        Y_all[holdout_keys[sel]], predict(fit, Xh[sel]), np.ones(int(sel.sum())))
                if subset_any_unseen.shape[0]:
                    sel = ~np.all(comp_seen, axis=1)
                    block[kind]["holdout_any_component_unseen"] = recon_metrics(
                        Y_all[holdout_keys[sel]], predict(fit, Xh[sel]), np.ones(int(sel.sum())))
        fold[target_name] = block
    summary["type_holdout"] = fold["T_total"]
    summary["type_holdout_T_joint"] = fold["T_joint"]

    np.savez_compressed(FIT_PATH, **{k: v.astype(np.float32) for k, v in fits.items()},
                        fit_keys=fit_keys.astype(np.int32), holdout_keys=holdout_keys.astype(np.int32),
                        keys=keys.astype(np.int32))
    np.savez_compressed(LOCAL / "null_maps.npz", **null_map)
    C.write_json(FIT_SUMMARY, summary)
    return summary


# ---------------------------------------------------------------------------
# replacement tables
# ---------------------------------------------------------------------------


def build_tables(tab: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Full [32000, width] tables for the real forward replacement."""
    st = dict(np.load(STATS_PATH, allow_pickle=False))
    occ, doc = st["occurrence"], st["doc_freq"]
    keys = np.flatnonzero(occ > 0).astype(np.int64)
    Y = np.asarray(tab["T_total"], dtype=np.float64)[keys]
    X_code = design("CODE_ADD", keys)
    X_sem1 = design("SEM_1", keys)
    X_sem2 = design("SEM_2", keys)
    null_keys, _diag = null_mapping(keys, doc, seed=0)
    X_null2 = design("SEM_2", null_keys)
    tables = {}
    w = np.ones(keys.size)
    tables["MEAN"] = np.tile(Y.mean(0), (N_KEYS, 1))
    for name, X in (("CODE_ADD", X_code), ("SEM_1", X_sem1), ("SEM_2", X_sem2), ("NULL_2", X_null2)):
        fit = ridge_fit(X, Y, w)
        pred = predict(fit, design({"NULL_2": "SEM_2"}.get(name, name), np.arange(N_KEYS)))
        tables[name] = pred
    return tables


# ---------------------------------------------------------------------------
# replaced local forward
# ---------------------------------------------------------------------------


def local_forward_factory(local: torch.nn.Module, layout: Mapping[str, tuple[int, int]],
                          table: torch.Tensor, teacher: torch.Tensor, known: torch.Tensor):
    """Replicate ``TULocalContent.forward`` with the three typed-S2 blocks replaced.

    The first linear layer is still applied to the *original* input, then the
    exact teacher projection of those three blocks is subtracted and the
    replacement per-path vector is summed back in.  Nothing else is touched.
    """
    W1 = local.mlp.lins[0].weight
    lo_t, hi_t = int(layout["s2_type"][0]), int(layout["s2_joint"][1])
    Wp = W1[:, lo_t:hi_t].contiguous()
    n_end = N_END

    def fwd(batch, a, b, scal):
        n_atoms = a.shape[0]
        width = a.shape[1]
        idx = batch["bond_index"]
        if idx.numel():
            u, v = idx[:, 0], idx[:, 1]
            degree = a.new_zeros(n_atoms)
            degree = degree.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
            degree = degree.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
            b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b).index_add_(0, v, b)
        else:
            degree = a.new_zeros(n_atoms)
            b_sum = a.new_zeros((n_atoms, width))
        nbr = (a.new_zeros((n_atoms, local.rank)).index_add_(0, batch["nbr_atom"], local.nbr_emb(batch["nbr_code"]))
               if batch["nbr_code"].numel() else a.new_zeros((n_atoms, local.rank)))
        has_paths = bool(batch["path_cat"].numel())
        path_dim = 3 * local.rank + 8
        if has_paths:
            path_joint = local.joint_proj(local.path_u(batch["path_cat"]) * local.path_v(batch["path_end63"]))
            path_emb = torch.cat([local.path_emb(batch["path_cat"]), local.end_emb(batch["path_end63"]),
                                  path_joint, local.close_emb(batch["path_closure"])], dim=1)
            path_agg = a.new_zeros((n_atoms, path_emb.shape[1])).index_add_(0, batch["path_atom"], path_emb)
            gate = local.path_gate(batch["path_cat"])
            content_agg = a.new_zeros((n_atoms, width)).index_add_(
                0, batch["path_atom"], a[batch["path_end"]] * gate)
        else:
            path_agg = a.new_zeros((n_atoms, path_dim))
            content_agg = a.new_zeros((n_atoms, width))
        parts = [a, b_sum, nbr, path_agg, content_agg, scal]
        x = torch.cat(parts, dim=1)
        if x.shape[1] != len(local.local_cols):
            raise RuntimeError(f"local width {x.shape[1]} != {len(local.local_cols)}")
        z = local.mlp.lins[0](x)
        if has_paths:
            z = z - path_agg[:, : (hi_t - lo_t)] @ Wp.T
            key = batch["path_cat"] * n_end + batch["path_end63"]
            vec = torch.where(known[key][:, None], table[key], teacher[key])
            z = z + a.new_zeros((n_atoms, z.shape[1])).index_add_(0, batch["path_atom"], vec)
        h = F.silu(z) if len(local.mlp.lins) > 1 else z
        for i, lin in enumerate(local.mlp.lins):
            if i:
                h = lin(h)
                if i < len(local.mlp.lins) - 1:
                    h = F.silu(h)
        return local.norm(h)

    return fwd


def install(model: torch.nn.Module, layout: Mapping[str, tuple[int, int]], table_np: np.ndarray,
            teacher_np: np.ndarray, known_np: np.ndarray, device: torch.device) -> None:
    table = torch.as_tensor(table_np, dtype=torch.float32, device=device)
    teacher = torch.as_tensor(teacher_np, dtype=torch.float32, device=device)
    known = torch.as_tensor(known_np, dtype=torch.bool, device=device)
    model.local.forward = local_forward_factory(model.local, layout, table, teacher, known)


def restore(model: torch.nn.Module) -> None:
    import tu_model as TU

    model.local.forward = model.local.__class__.forward.__get__(model.local, model.local.__class__)
    _ = TU  # keep the import explicit


def _predict(model: torch.nn.Module, task: Any, rows: np.ndarray, device: torch.device,
             *, capture: bool = False) -> dict[str, Any]:
    model.eval()
    out = []
    h0s, pres = [], []
    hook_holder: dict[str, list[torch.Tensor]] = {"pre": [], "h0": []}
    local = model.local

    def pre_hook(_m, inputs):
        hook_holder["pre"].append(inputs[0].detach())

    handle = local.mlp.lins[0].register_forward_pre_hook(pre_hook)
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), 256):
                chunk = np.asarray(rows[start : start + 256], dtype=np.int64)
                batch = C.DC.PC.to_device(C.DC.PC.collate(task.cache, chunk.tolist()), device)
                a, b, scal, h = model.encode_h(batch)
                logits = model.post_local(h, batch)
                out.append(logits.detach().cpu().numpy().astype(np.float32))
                if capture:
                    h0s.append(h.detach().cpu().numpy().astype(np.float32))
    finally:
        handle.remove()
    res = {"logits": np.concatenate(out, axis=0)}
    if capture:
        res["h0"] = np.concatenate(h0s, axis=0)
        res["preactivation"] = torch.cat(hook_holder["pre"], dim=0).cpu().numpy()
    return res


def run_verify(tab: Mapping[str, np.ndarray], table: dict[str, np.ndarray], *, n_molecules: int = 160) -> dict[str, Any]:
    device = torch.device("cpu")
    model, task, meta = load_teacher("cpu")
    layout = meta["layout"]
    teacher = np.asarray(tab["T_total"], dtype=np.float32)
    known = np.ones(N_KEYS, dtype=bool)
    rows = np.asarray(task.valid_rows, dtype=np.int64)[:n_molecules]
    base = _predict(model, task, rows, device, capture=True)
    install(model, layout, np.asarray(tab["T_total"], np.float32), teacher, known, device)
    rebuilt = _predict(model, task, rows, device, capture=True)
    restore(model)
    out = {
        "n_molecules": int(n_molecules),
        "preactivation_max_abs_error": float(np.abs(rebuilt["preactivation"] - base["preactivation"]).max()),
        "preactivation_rel_rms": float(np.sqrt(((rebuilt["preactivation"] - base["preactivation"]) ** 2).mean())
                                       / np.sqrt((base["preactivation"] ** 2).mean())),
        "h0_max_abs_error": float(np.abs(rebuilt["h0"] - base["h0"]).max()),
        "logits_max_abs_error": float(np.abs(rebuilt["logits"] - base["logits"]).max()),
        "exact_path_count": int(len(rows)),
        "tolerance": {"preactivation_max_abs_error": 1e-4, "logits_max_abs_error": 1e-4},
    }
    out["passed"] = bool(out["preactivation_max_abs_error"] < 1e-4 and out["logits_max_abs_error"] < 1e-4)
    C.write_json(VERIFY_PATH, out)
    if not out["passed"]:
        raise RuntimeError(f"identity reconstruction failed: {out}")
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 200:
        return None
    return float(spearmanr(a[ok], b[ok]).statistic)


def topk_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.01) -> float:
    n = a.size
    k = max(1, int(round(frac * n)))
    ia = np.argsort(-a, kind="mergesort")[:k]
    ib = np.argsort(-b, kind="mergesort")[:k]
    return float(len(set(ia.tolist()) & set(ib.tolist())) / k)


def run_replace(tab: Mapping[str, np.ndarray], tables: dict[str, np.ndarray], *, train_sample: int = 4096) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, task, meta = load_teacher(device)
    layout = meta["layout"]
    teacher = np.asarray(tab["T_total"], dtype=np.float32)
    st = dict(np.load(STATS_PATH, allow_pickle=False))
    known = (st["occurrence"] > 0)
    y_va = np.asarray(task.labels[task.valid_rows], dtype=np.float32)
    y_te = C.y_test()
    train_rows = np.asarray(task.train_rows, dtype=np.int64)
    h = np.asarray([abs(hash(int(r))) % (1 << 61) for r in train_rows])
    sub = train_rows[np.argsort(h)[:train_sample]]
    y_tr = np.asarray(task.labels[sub], dtype=np.float32)
    rows = {"valid": np.asarray(task.valid_rows, dtype=np.int64),
            "test": np.asarray(task.test_rows, dtype=np.int64),
            "train_subset": sub}
    labels = {"valid": y_va, "test": y_te, "train_subset": y_tr}

    # replacement coverage over real paths
    coverage = {}
    for split in ("valid", "test", "train_subset"):
        keys, _mol = path_keys(task.cache, rows[split])
        seen = known[np.clip(keys, 0, N_KEYS - 1)]
        coverage[split] = {"paths": int(keys.size), "known_type_fraction": float(seen.mean()) if keys.size else None,
                           "molecules": int(rows[split].size)}

    record: dict[str, Any] = {"schema": "cc-function-replacement-v1", "created_utc": C.utcnow(),
                              "device": str(device), "teacher": "W2_seed0_best_valid",
                              "coverage": coverage, "variants": {}, "lambda": LAMBDA}
    base: dict[str, dict[str, np.ndarray]] = {}
    for split in rows:
        base[split] = _predict(model, task, rows[split], device, capture=(split != "train_subset"))
        ap = C.macro_ap(labels[split], C.score_logits(base[split]["logits"]))
        record["variants"].setdefault("ORIGINAL", {})[split] = {
            "macro_ap": ap["macro_ap"], "per_task_ap": ap["per_task_ap"],
            "logits_sha256": C.sha256_array(base[split]["logits"]),
            "h0_rms": float(np.sqrt((base[split]["h0"] ** 2).mean())) if "h0" in base[split] else None,
            "preactivation_rms": float(np.sqrt((base[split]["preactivation"] ** 2).mean())) if "preactivation" in base[split] else None,
        }
    for name in REPLACE_ORDER:
        install(model, layout, tables[name], teacher, known, device)
        for split in rows:
            r = _predict(model, task, rows[split], device, capture=(split != "train_subset"))
            ap = C.macro_ap(labels[split], C.score_logits(r["logits"]))
            b = base[split]["logits"]
            per_task_delta = [None if (a is None or bb is None) else float(a - bb) for a, bb in
                              zip(ap["per_task_ap"], record["variants"]["ORIGINAL"][split]["per_task_ap"])]
            entry = {
                "macro_ap": ap["macro_ap"],
                "delta_ap": float(ap["macro_ap"] - record["variants"]["ORIGINAL"][split]["macro_ap"]),
                "per_task_ap": ap["per_task_ap"], "per_task_delta_ap": per_task_delta,
                "logits_rms_delta": float(np.sqrt(((r["logits"] - b) ** 2).mean())),
                "logits_rms": float(np.sqrt((b ** 2).mean())),
            }
            if "h0" in r:
                entry["h0_rel_rms_error"] = float(np.sqrt(((r["h0"] - base[split]["h0"]) ** 2).mean())
                                                  / np.sqrt((base[split]["h0"] ** 2).mean()))
                entry["preactivation_rel_rms_error"] = float(
                    np.sqrt(((r["preactivation"] - base[split]["preactivation"]) ** 2).mean())
                    / np.sqrt((base[split]["preactivation"] ** 2).mean()))
                entry["preactivation_max_abs_error"] = float(np.abs(r["preactivation"] - base[split]["preactivation"]).max())
                entry["spearman_mean"] = float(np.nanmean([s for s in (
                    _spearman(r["logits"][:, t], b[:, t]) for t in range(N_TASKS)) if s is not None]))
                entry["top1pct_overlap_mean"] = float(np.mean([topk_overlap(r["logits"][:, t], b[:, t])
                                                               for t in range(N_TASKS)]))
            record["variants"].setdefault(name, {})[split] = entry
            np.save(LOCAL / f"logits_{name}_{split}.npy", r["logits"].astype(np.float32))
        restore(model)
        C.write_json(LOCAL / "function_replacement_raw.json", record)
        print(f"[replace] {name} done", flush=True)
    if device.type != "cuda":
        record["device_note"] = "cpu forward (no local GPU)"
    C.write_json(LOCAL / "function_replacement_raw.json", record)
    return record
