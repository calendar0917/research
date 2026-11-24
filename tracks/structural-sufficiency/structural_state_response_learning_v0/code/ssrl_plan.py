"""Build the frozen SSRL v0 Train row plan (CPU, no model, no labels in selection).

The plan is a deterministic, label-free sample of real four-molecule rectangles
from the existing ``structural_functional_equivalence_v0`` Train 1M matching
directory, plus an equal-size plain-molecule control set from the original 1M
Train row plan.  Nothing in the selection reads a label, a model prediction, an
error, or a Test row.

Run locally::

    uv run python tracks/structural-sufficiency/structural_state_response_learning_v0/code/ssrl_plan.py

The output is ``data/train_plan.npz`` plus ``data/train_plan_manifest.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ssrl_common as C


def _log(t0: float, msg: str) -> None:
    print(f"[{time.perf_counter() - t0:7.1f}s rss={C.peak_rss_gb():.2f}GB] {msg}", flush=True)


def load_assets() -> dict:
    prefix = C.SFEC0_DATA / "cells_pcqm_train_1m"
    with np.load(str(prefix) + ".npz") as z:
        cells = {k: z[k] for k in (
            "row_id", "mol_id", "C_id", "S_id", "bond_type", "reasm_stereo",
        )}
    with np.load(str(prefix) + ".universe.npz") as z:
        primary = z["primary"].copy()
        rank = z["rank"].copy()
        train_mask = z["train_mask"].copy()
    S = pq.read_table(str(prefix) + ".S.parquet", columns=["S_id", "nS"])
    S = {n: S[n].to_numpy(zero_copy_only=False) for n in S.column_names}
    M = pq.read_table(str(prefix) + ".M.parquet",
                      columns=["mol_id", "mol_key", "row_id"])
    M = {n: M[n].to_numpy(zero_copy_only=False) for n in M.column_names}
    Ctab = pq.read_table(str(prefix) + ".C.parquet", columns=["C_id", "q_id"])
    Ctab = {n: Ctab[n].to_numpy(zero_copy_only=False) for n in Ctab.column_names}
    labels = np.load(C.SFEC1_DATA / "pcqm_labels.npy")
    gids = np.load(C.TRAIN_GIDS).astype(np.int64)
    subset_rows = np.load(C.SUBSET_ROWS).astype(np.int64)
    label_norm = C.load_label_norm()
    return {
        "prefix": prefix, "cells": cells, "primary": primary, "rank": rank,
        "train_mask": train_mask, "S": S, "M": M, "C": Ctab, "labels": labels,
        "gids": gids, "subset_rows": subset_rows, "label_norm": label_norm,
    }


def molecule_clean(assets: dict, t0: float) -> dict:
    cells = assets["cells"]
    M = assets["M"]
    mol_key = M["mol_key"]
    n_mol = len(mol_key)
    stereo = np.asarray(
        [("@" in k) or ("/" in k) or ("\\" in k) for k in mol_key.tolist()], bool
    )
    mol_id = cells["mol_id"].astype(np.int64)
    row_id = cells["row_id"].astype(np.int64)
    pairs = np.unique(np.stack([mol_id, row_id], axis=1), axis=0)
    n_rows = np.bincount(pairs[:, 0].astype(np.int64), minlength=n_mol)
    duplicate_identity = n_rows > 1
    reasm_bad = np.zeros(n_mol, bool)
    np.logical_or.at(reasm_bad, mol_id[cells["reasm_stereo"] == 0], True)
    clean = (~stereo) & (~duplicate_identity) & (~reasm_bad)
    _log(t0, f"molecule clean flags: {int(clean.sum())}/{n_mol} clean; "
             f"stereo {int(stereo.sum())}, duplicate identity {int(duplicate_identity.sum())}, "
             f"directory reasm_stereo==0 {int(reasm_bad.sum())}")
    return {
        "clean": clean, "stereo": stereo,
        "duplicate_identity": duplicate_identity, "reasm_bad": reasm_bad,
        "n_rows": n_rows,
    }


def build_clean_contexts(assets: dict, mol_clean: np.ndarray, t0: float) -> dict:
    cells = assets["cells"]
    n_s = int(assets["S"]["nS"].size)
    mask0 = (assets["primary"][cells["S_id"].astype(np.int64)]
             & (cells["bond_type"] == 1))
    mol_id = cells["mol_id"].astype(np.int64)
    C_id = cells["C_id"].astype(np.int64)
    S_id = cells["S_id"].astype(np.int64)
    b = cells["bond_type"].astype(np.int64)
    key = ((C_id * n_s + S_id) * 8 + b)[mask0]
    mol = mol_id[mask0]
    pair = np.unique(np.stack([key, mol], axis=1), axis=0)
    uk, um = pair[:, 0], pair[:, 1]
    starts = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
    ends = np.r_[starts[1:], len(uk)]
    count = (ends - starts).astype(np.int64)
    cell_key = uk[starts]
    cell_mol = um[starts]
    keep = (count == 1) & mol_clean[cell_mol]
    cell_key = cell_key[keep]
    cell_mol = cell_mol[keep]
    cell_C = (cell_key // (n_s * 8)).astype(np.int64)
    rem = cell_key % (n_s * 8)
    cell_S = (rem // 8).astype(np.int64)
    cell_gid = assets["M"]["row_id"].astype(np.int64)[cell_mol]
    cell_y = assets["labels"][cell_gid].astype(np.float64)
    if not np.isfinite(cell_y).all():
        raise RuntimeError("non-finite Train cell label")
    row_of_gid = np.full(int(assets["gids"].max()) + 1, -1, np.int64)
    row_of_gid[assets["gids"]] = np.arange(assets["gids"].size)
    cell_cache_row = row_of_gid[cell_gid]
    if (cell_cache_row < 0).any():
        raise RuntimeError("a selected Train cell is outside the frozen 1M row plan")
    order = np.argsort(cell_C, kind="stable")
    Cs = cell_C[order]
    ctx_starts = np.flatnonzero(np.r_[True, Cs[1:] != Cs[:-1]])
    ctx_ends = np.r_[ctx_starts[1:], len(Cs)]
    ctx_k = ctx_ends - ctx_starts
    ctx_off = np.r_[0, np.cumsum(ctx_k)].astype(np.int64)
    ctx_C = Cs[ctx_starts].astype(np.int64)
    q_id = assets["C"]["q_id"].astype(np.int64)
    ctx_q = q_id[ctx_C]
    contexts = {
        "C": ctx_C,
        "off": ctx_off,
        "k": ctx_k.astype(np.int64),
        "S": cell_S[order].astype(np.int64),
        "mol": cell_mol[order].astype(np.int64),
        "gid": cell_gid[order].astype(np.int64),
        "y": cell_y[order].astype(np.float64),
        "cache_row": cell_cache_row[order].astype(np.int64),
        "q": ctx_q.astype(np.int64),
        "n_ctx": int(len(ctx_C)),
        "n_cells": int(len(cell_C)),
    }
    _log(t0, f"clean cells {len(cell_C)} in {len(ctx_C)} exact contexts; "
             f"context pair count inside contexts {int((ctx_k * (ctx_k - 1) // 2).sum())}")
    return contexts


def build_pair_observations(contexts: dict, rank: np.ndarray, t0: float) -> dict:
    off = contexts["off"]
    k = contexts["k"]
    S = contexts["S"]
    pair_keys: list[np.ndarray] = []
    ctxs: list[np.ndarray] = []
    pos_a: list[np.ndarray] = []
    pos_b: list[np.ndarray] = []
    max_k = int(k.max()) if k.size else 0
    for kk in range(2, max_k + 1):
        idx = np.flatnonzero(k == kk)
        if idx.size == 0:
            continue
        pos = off[idx][:, None] + np.arange(kk, dtype=np.int64)[None, :]
        r = rank[S[pos]].astype(np.int64)
        ii, jj = np.triu_indices(kk, 1)
        a = r[:, ii]
        b = r[:, jj]
        swap = a > b
        A = np.where(swap, b, a)
        B = np.where(swap, a, b)
        pa = np.where(swap, pos[:, jj], pos[:, ii])
        pb = np.where(swap, pos[:, ii], pos[:, jj])
        pair_keys.append((A * 64 + B).ravel())
        ctxs.append(np.repeat(idx, ii.size))
        pos_a.append(pa.ravel())
        pos_b.append(pb.ravel())
    pk = np.concatenate(pair_keys) if pair_keys else np.zeros(0, np.int64)
    ctx = np.concatenate(ctxs) if ctxs else np.zeros(0, np.int64)
    pa = np.concatenate(pos_a) if pos_a else np.zeros(0, np.int64)
    pb = np.concatenate(pos_b) if pos_b else np.zeros(0, np.int64)
    order = np.argsort(pk, kind="stable")
    pk, ctx, pa, pb = pk[order], ctx[order], pa[order], pb[order]
    starts = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]]) if pk.size else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], len(pk)] if pk.size else np.zeros(0, np.int64)
    pair_types = pk[starts].astype(np.int64) if pk.size else np.zeros(0, np.int64)
    _log(t0, f"pair observations {len(pk)} over {len(pair_types)} canonical pair types")
    return {
        "pair_key": pk.astype(np.int64), "ctx_index": ctx.astype(np.int64),
        "pos_a": pa.astype(np.int64), "pos_b": pb.astype(np.int64),
        "pair_types": pair_types, "starts": starts.astype(np.int64),
        "ends": ends.astype(np.int64),
    }


def select_candidates(assets: dict, contexts: dict, pairs: dict,
                      max_per_type: int, context_pool: int, t0: float) -> dict:
    pair_types = pairs["pair_types"]
    starts = pairs["starts"]
    ends = pairs["ends"]
    pk_all = pairs["pair_key"]
    ctx_all = pairs["ctx_index"]
    pa_all = pairs["pos_a"]
    pb_all = pairs["pos_b"]
    ctx_C = contexts["C"]
    mols = contexts["mol"]
    gid = contexts["gid"]
    y = contexts["y"]
    ctx_q = contexts["q"]

    out = {name: [] for name in (
        "pk", "c1", "c2", "p1a", "p1b", "p2a", "p2b",
        "m1", "m2", "m3", "m4", "g1", "g2", "g3", "g4",
        "y1", "y2", "y3", "y4", "hash", "pt", "q1", "q2",
    )}
    n_possible = []
    n_valid = []
    n_contexts = []
    for pt_idx, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
        pk = int(pair_types[pt_idx])
        cs = ctx_all[a:b]
        pa = pa_all[a:b]
        pb = pb_all[a:b]
        n_ctx = int(len(cs))
        n_contexts.append(n_ctx)
        if n_ctx < 2:
            n_possible.append(0)
            n_valid.append(0)
            continue
        c_ids = ctx_C[cs]
        # fixed-hash context pool; this is the declared subsample that avoids
        # enumerating the (often enormous) full context-pair set.
        h_ctx = C.mix64(np.full(n_ctx, pk, np.int64), c_ids, np.zeros(n_ctx, np.int64))
        take = np.argsort(h_ctx, kind="stable")[: int(min(n_ctx, context_pool))]
        cs = cs[take]
        pa = pa[take]
        pb = pb[take]
        m = int(len(cs))
        ii, jj = np.triu_indices(m, 1)
        c1 = ctx_C[cs[ii]]
        c2 = ctx_C[cs[jj]]
        p1a = pa[ii]
        p1b = pb[ii]
        p2a = pa[jj]
        p2b = pb[jj]
        swap = c1 > c2
        c1, c2 = np.where(swap, c2, c1), np.where(swap, c1, c2)
        p1a, p2a = np.where(swap, p2a, p1a), np.where(swap, p1a, p2a)
        p1b, p2b = np.where(swap, p2b, p1b), np.where(swap, p1b, p2b)
        m1, m2, m3, m4 = mols[p1a], mols[p1b], mols[p2a], mols[p2b]
        valid = ((m1 != m2) & (m1 != m3) & (m1 != m4)
                 & (m2 != m3) & (m2 != m4) & (m3 != m4))
        n_possible.append(int(ii.size))
        n_valid.append(int(valid.sum()))
        if not valid.any():
            continue
        h = C.mix64(np.full(len(c1), pk, np.int64), c1, c2)
        vidx = np.flatnonzero(valid)
        order = np.argsort(h[vidx], kind="stable")
        take2 = vidx[order][: int(max_per_type)]
        out["pk"].append(np.full(len(take2), pk, np.int64))
        out["c1"].append(c1[take2].astype(np.int64))
        out["c2"].append(c2[take2].astype(np.int64))
        out["p1a"].append(p1a[take2].astype(np.int64))
        out["p1b"].append(p1b[take2].astype(np.int64))
        out["p2a"].append(p2a[take2].astype(np.int64))
        out["p2b"].append(p2b[take2].astype(np.int64))
        out["m1"].append(m1[take2].astype(np.int64))
        out["m2"].append(m2[take2].astype(np.int64))
        out["m3"].append(m3[take2].astype(np.int64))
        out["m4"].append(m4[take2].astype(np.int64))
        out["g1"].append(gid[p1a[take2]].astype(np.int64))
        out["g2"].append(gid[p1b[take2]].astype(np.int64))
        out["g3"].append(gid[p2a[take2]].astype(np.int64))
        out["g4"].append(gid[p2b[take2]].astype(np.int64))
        out["y1"].append(y[p1a[take2]].astype(np.float64))
        out["y2"].append(y[p1b[take2]].astype(np.float64))
        out["y3"].append(y[p2a[take2]].astype(np.float64))
        out["y4"].append(y[p2b[take2]].astype(np.float64))
        out["hash"].append(h[take2].astype(np.uint64))
        out["pt"].append(np.full(len(take2), pt_idx, np.int64))
        out["q1"].append(ctx_q[cs[ii]][take2].astype(np.int64))
        out["q2"].append(ctx_q[cs[jj]][take2].astype(np.int64))

    cand = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.uint64 if k == "hash" else np.int64))
            for k, v in out.items()}
    for k in ("y1", "y2", "y3", "y4"):
        cand[k] = cand[k].astype(np.float64)
    _log(t0, f"candidate rectangles: {len(cand['pk'])} selected candidates over "
             f"{len(np.unique(cand['pt']))} pair types; "
             f"possible (pooled) {int(np.sum(n_possible))}, valid {int(np.sum(n_valid))}")
    return {
        "cand": cand,
        "n_possible": np.asarray(n_possible, np.int64),
        "n_valid": np.asarray(n_valid, np.int64),
        "n_contexts": np.asarray(n_contexts, np.int64),
    }


def choose_rectangles(cand: dict, pair_types: np.ndarray, max_rect: int,
                      max_per_type: int, t0: float, topup: bool = False) -> dict:
    pt = cand["pt"]
    n_cand = len(pt)
    type_starts = np.searchsorted(pt, np.arange(len(pair_types)), side="left")
    type_ends = np.searchsorted(pt, np.arange(len(pair_types)), side="right")
    pt_order = np.asarray(
        sorted(range(len(pair_types)), key=lambda i: C.hash64("pair", int(pair_types[i]))),
        np.int64,
    )
    max_rounds = int(max_per_type)
    accepted = np.zeros(n_cand, bool)
    disjoint = np.zeros(n_cand, bool)
    chosen: list[int] = []
    used: set[int] = set()
    m1, m2, m3, m4 = cand["m1"], cand["m2"], cand["m3"], cand["m4"]
    for r in range(max_rounds):
        for p in pt_order.tolist():
            s, e = int(type_starts[p]), int(type_ends[p])
            if s + r >= e:
                continue
            i = s + r
            mols = (int(m1[i]), int(m2[i]), int(m3[i]), int(m4[i]))
            if any(x in used for x in mols):
                continue
            accepted[i] = True
            disjoint[i] = True
            used.update(mols)
            chosen.append(i)
            if len(chosen) >= max_rect:
                break
        if len(chosen) >= max_rect:
            break
    n_disjoint = len(chosen)
    n_topup = 0
    if topup and len(chosen) < max_rect:
        for r in range(max_rounds):
            for p in pt_order.tolist():
                s, e = int(type_starts[p]), int(type_ends[p])
                if s + r >= e:
                    continue
                i = s + r
                if accepted[i]:
                    continue
                accepted[i] = True
                chosen.append(i)
                n_topup += 1
                if len(chosen) >= max_rect:
                    break
            if len(chosen) >= max_rect:
                break
    chosen_arr = np.asarray(chosen, np.int64)
    # stable output order by pair type then rectangle hash
    order = np.lexsort((cand["hash"][chosen_arr], cand["pk"][chosen_arr]))
    chosen_arr = chosen_arr[order]
    selected_pt = cand["pt"][chosen_arr]
    n_types = int(np.unique(selected_pt).size)
    per_type = np.bincount(selected_pt, minlength=len(pair_types))
    pos = per_type[per_type > 0]
    _log(t0, f"chosen rectangles {len(chosen_arr)} (disjoint-first {n_disjoint}, "
             f"top-up {n_topup}) over {n_types} pair types; "
             f"per-type min/median/max {int(pos.min()) if pos.size else 0}/"
             f"{float(np.median(pos)) if pos.size else 0.0}/"
             f"{int(per_type.max()) if per_type.size else 0}")
    return {
        "chosen": chosen_arr,
        "accepted": accepted,
        "disjoint": disjoint,
        "n_disjoint": int(n_disjoint),
        "n_topup": int(n_topup),
        "n_pair_types": n_types,
        "per_type": per_type,
        "type_starts": type_starts,
        "type_ends": type_ends,
        "pt_order": pt_order,
    }


def select_plain(assets: dict, rect_gids: np.ndarray, n_plain: int, t0: float) -> dict:
    gids_1m = assets["gids"]
    labels = assets["labels"]
    M = assets["M"]
    endpoint_gids = np.unique(rect_gids.reshape(-1))
    endpoint_mols = np.unique(M["mol_id"].astype(np.int64)[
        np.isin(M["row_id"].astype(np.int64), endpoint_gids)
    ])
    max_gid = int(gids_1m.max())
    row_to_mol = np.full(max_gid + 1, -1, np.int64)
    row_to_mol[M["row_id"].astype(np.int64)] = M["mol_id"].astype(np.int64)
    cand_mol = row_to_mol[gids_1m]
    exclude = np.isin(gids_1m, endpoint_gids)
    if endpoint_mols.size:
        exclude |= np.isin(cand_mol, endpoint_mols)
    rows_all = np.arange(gids_1m.size, dtype=np.int64)
    salt = C.hash64("plain-salt") & ((1 << 63) - 1)
    h = C.mix64_plain(rows_all, int(salt))
    valid = np.flatnonzero(~exclude)
    order = valid[np.argsort(h[valid], kind="stable")]
    take = order[: int(n_plain)]
    if take.size != int(n_plain):
        raise RuntimeError(f"not enough plain molecules: {take.size} < {n_plain}")
    pg = gids_1m[take].astype(np.int64)
    py = labels[pg].astype(np.float64)
    if not np.isfinite(py).all():
        raise RuntimeError("non-finite plain-molecule label")
    if np.intersect1d(pg, endpoint_gids).size:
        raise RuntimeError("plain molecule overlaps a rectangle endpoint gid")
    if endpoint_mols.size and np.intersect1d(row_to_mol[pg], endpoint_mols).size:
        raise RuntimeError("plain molecule overlaps a rectangle endpoint identity")
    _log(t0, f"plain molecules {take.size} (excluded {int(exclude.sum())} endpoint rows/identities)")
    return {
        "gid": pg, "cache_row": take.astype(np.int64), "target_eV": py,
        "hash": h[take].astype(np.uint64), "n_excluded": int(exclude.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.DATA_DIR / "train_plan.npz"))
    parser.add_argument("--manifest", default=str(C.DATA_DIR / "train_plan_manifest.json"))
    parser.add_argument("--max-rectangles", type=int, default=None)
    parser.add_argument("--max-per-pair-type", type=int, default=None)
    parser.add_argument("--context-pool", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = C.read_json(C.CONFIG_PATH)
    max_rect = int(args.max_rectangles or cfg["train_plan"]["max_rectangles"])
    max_per = int(args.max_per_pair_type or cfg["train_plan"]["max_per_pair_type"])
    pool = int(args.context_pool or cfg["train_plan"]["context_pool_per_pair_type"])
    epochs = int(args.epochs or cfg["horizon_epochs"])
    min_rect = int(cfg["train_plan"]["min_rectangles"])
    min_types = int(cfg["train_plan"]["min_pair_types"])
    label_norm = C.load_label_norm()

    t_start = time.perf_counter()
    assets = load_assets()
    _log(t_start, "assets loaded")
    clean = molecule_clean(assets, t_start)
    contexts = build_clean_contexts(assets, clean["clean"], t_start)
    pairs = build_pair_observations(contexts, assets["rank"], t_start)
    cand_info = select_candidates(assets, contexts, pairs, max_per, pool, t_start)
    chosen_info = choose_rectangles(
        cand_info["cand"], pairs["pair_types"], max_rect, max_per, t_start,
        topup=False,
    )
    chosen = chosen_info["chosen"]
    cand = cand_info["cand"]
    n_rect = int(chosen.size)
    n_types = int(chosen_info["n_pair_types"])

    support = {
        "n_rectangles": n_rect,
        "n_pair_types": n_types,
        "n_disjoint_rectangles": int(chosen_info["n_disjoint"]),
        "n_topup_rectangles": int(chosen_info["n_topup"]),
        "min_rectangles_required": min_rect,
        "min_pair_types_required": min_types,
        "passes_minimum_support": bool(n_rect >= min_rect and n_types >= min_types),
        "n_candidate_rectangles": int(len(cand["pk"])),
        "n_candidate_pair_types": int(len(pairs["pair_types"])),
        "n_contexts_in_clean_pool": int(contexts["n_ctx"]),
        "n_clean_cells": int(contexts["n_cells"]),
        "per_type_selected_min": int(chosen_info["per_type"][chosen_info["per_type"] > 0].min())
        if n_types else 0,
        "per_type_selected_max": int(chosen_info["per_type"].max()) if n_types else 0,
        "per_type_selected_mean": float(n_rect / max(n_types, 1)),
        "context_pool_per_pair_type": int(pool),
        "max_per_pair_type": int(max_per),
    }
    C.write_json(C.DATA_DIR / "plan_support.json", support)
    _log(t_start, f"support: {n_rect} rectangles / {n_types} pair types "
                  f"(minimum {min_rect}/{min_types}); pass={support['passes_minimum_support']}")
    if not support["passes_minimum_support"]:
        raise RuntimeError(
            "INSUFFICIENT_RECTANGLE_SUPPORT: the existing Train matching directory "
            f"gives {n_rect} rectangles over {n_types} pair types; the pre-declared "
            f"minimum is {min_rect}/{min_types}. Stopping without rebuilding data."
        )

    # ---- materialise the selected row plan --------------------------------
    rect_gid = np.stack([cand["g1"][chosen], cand["g2"][chosen],
                         cand["g3"][chosen], cand["g4"][chosen]], axis=1)
    rect_mol = np.stack([cand["m1"][chosen], cand["m2"][chosen],
                         cand["m3"][chosen], cand["m4"][chosen]], axis=1)
    rect_y = np.stack([cand["y1"][chosen], cand["y2"][chosen],
                       cand["y3"][chosen], cand["y4"][chosen]], axis=1)
    rect_z = C.to_z(rect_y, label_norm).astype(np.float64)
    # cache row is exactly the position in the frozen 1M gid array
    row_of_gid = np.full(int(assets["gids"].max()) + 1, -1, np.int64)
    row_of_gid[assets["gids"]] = np.arange(assets["gids"].size)
    rect_cache_row = row_of_gid[rect_gid]
    if (rect_cache_row < 0).any():
        raise RuntimeError("a selected rectangle endpoint is outside the 1M row plan")
    rect_pair_key = cand["pk"][chosen].astype(np.int64)
    rect_A = (rect_pair_key // 64).astype(np.int64)
    rect_B = (rect_pair_key % 64).astype(np.int64)
    rect_ctx1 = cand["c1"][chosen].astype(np.int64)
    rect_ctx2 = cand["c2"][chosen].astype(np.int64)
    rect_hash = cand["hash"][chosen].astype(np.uint64)
    rect_is_disjoint = chosen_info["disjoint"][chosen].astype(bool)
    rect_q1 = cand["q1"][chosen].astype(np.int64)
    rect_q2 = cand["q2"][chosen].astype(np.int64)

    n_plain = 4 * n_rect
    plain = select_plain(assets, rect_gid, n_plain, t_start)
    plain_z = C.to_z(plain["target_eV"], label_norm).astype(np.float64)

    rng = np.random.default_rng(int(cfg["order_seed"]))
    rect_order = np.empty((epochs, n_rect), np.int64)
    plain_order = np.empty((epochs, n_plain), np.int64)
    for e in range(epochs):
        rect_order[e] = rng.permutation(n_rect)
        plain_order[e] = rng.permutation(n_plain)
    # occurrence counts: exactly one appearance per epoch, no accidental drop
    if not np.array_equal(np.sort(rect_order, axis=1), np.broadcast_to(np.arange(n_rect), (epochs, n_rect))):
        raise RuntimeError("rectangle batch plan is not a permutation per epoch")
    if not np.array_equal(np.sort(plain_order, axis=1), np.broadcast_to(np.arange(n_plain), (epochs, n_plain))):
        raise RuntimeError("plain batch plan is not a permutation per epoch")
    steps_per_epoch = int((n_rect + 15) // 16)
    batch_plan_sha = C.sha256_arrays({"rect_order": rect_order, "plain_order": plain_order})

    arrays = {
        "rect_pair_key": rect_pair_key,
        "rect_A_rank": rect_A,
        "rect_B_rank": rect_B,
        "rect_ctx1": rect_ctx1,
        "rect_ctx2": rect_ctx2,
        "rect_q1": rect_q1,
        "rect_q2": rect_q2,
        "rect_gid": rect_gid.astype(np.int64),
        "rect_mol": rect_mol.astype(np.int64),
        "rect_cache_row": rect_cache_row.astype(np.int64),
        "rect_target_eV": rect_y.astype(np.float64),
        "rect_target_z": rect_z.astype(np.float64),
        "rect_hash": rect_hash,
        "rect_is_disjoint": rect_is_disjoint,
        "plain_gid": plain["gid"].astype(np.int64),
        "plain_cache_row": plain["cache_row"].astype(np.int64),
        "plain_target_eV": plain["target_eV"].astype(np.float64),
        "plain_target_z": plain_z.astype(np.float64),
        "plain_hash": plain["hash"],
        "rect_order": rect_order.astype(np.int32),
        "plain_order": plain_order.astype(np.int32),
        "pair_types": pairs["pair_types"].astype(np.int64),
        "pair_type_n_contexts": cand_info["n_contexts"].astype(np.int64),
        "pair_type_n_possible": cand_info["n_possible"].astype(np.int64),
        "pair_type_n_valid": cand_info["n_valid"].astype(np.int64),
        "pair_type_n_selected": chosen_info["per_type"].astype(np.int64),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    plan_sha = C.sha256_file(out)

    all_gids = np.concatenate([rect_gid.ravel(), plain["gid"]])
    train_gid_set = set(int(x) for x in assets["subset_rows"].tolist())
    outside = [int(x) for x in np.unique(all_gids).tolist() if int(x) not in train_gid_set]
    if outside:
        raise RuntimeError(f"{len(outside)} plan rows are outside the frozen 1M Train row plan")
    overlap = np.intersect1d(rect_gid.ravel(), plain["gid"])
    if overlap.size:
        raise RuntimeError("plain molecules overlap rectangle endpoints")
    # every rectangle: four distinct molecules; every endpoint inside the 1M plan
    if not np.all([len(set(row)) == 4 for row in rect_mol.tolist()]):
        raise RuntimeError("a selected rectangle does not have four distinct molecules")
    if rect_is_disjoint.all() and np.unique(rect_mol).size != rect_mol.size:
        raise RuntimeError("a molecule occurs in more than one declared-disjoint rectangle")

    manifest = {
        "schema": "ssrl-v0-train-plan",
        "round": cfg["round"],
        "created_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "config": C.track_rel(C.CONFIG_PATH),
        "plan_npz": C.track_rel(out),
        "plan_sha256": plan_sha,
        "batch_plan_sha256": batch_plan_sha,
        "seed": int(cfg["seed"]),
        "order_seed": int(cfg["order_seed"]),
        "hash_salt": cfg["hash_salt"],
        "epochs": epochs,
        "rectangles": n_rect,
        "pair_types": n_types,
        "plain_molecules": n_plain,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": int(epochs * steps_per_epoch),
        "endpoint_occurrences": int(4 * n_rect),
        "unique_rectangle_endpoint_identities": int(np.unique(rect_mol).size),
        "disjoint_rectangles": int(rect_is_disjoint.sum()),
        "topup_rectangles": int((~rect_is_disjoint).sum()),
        "support": support,
        "clean_molecule_counts": {
            "n_molecules": int(len(assets["M"]["mol_id"])),
            "clean": int(clean["clean"].sum()),
            "stereo": int(clean["stereo"].sum()),
            "duplicate_identity": int(clean["duplicate_identity"].sum()),
            "directory_reasm_stereo_bad": int(clean["reasm_bad"].sum()),
        },
        "selection": {
            "context_pool_per_pair_type": int(pool),
            "max_per_pair_type": int(max_per),
            "max_rectangles": int(max_rect),
            "candidate_contexts": f"first min(n,{pool}) exact contexts per pair type in hash64('ctx', pair_key, C_id) order",
            "candidate_rectangles": "all C(m,2) context pairs among the pooled contexts, filtered to four distinct clean molecules",
            "candidate_order": "SplitMix64(pair_key, C1, C2)",
            "pair_type_order": "hash64('pair', pair_key)",
            "global_order": "round-robin over pair types; molecule-disjoint only, stop at the 8000 cap; no overlapping rectangle is added",
            "label_free": True,
        },
        "data_identity": {
            "train_cells_npz": C.sha256_file(str(C.SFEC0_DATA / "cells_pcqm_train_1m.npz")),
            "train_cells_universe": C.sha256_file(str(C.SFEC0_DATA / "cells_pcqm_train_1m.universe.npz")),
            "train_cells_M": C.sha256_file(str(C.SFEC0_DATA / "cells_pcqm_train_1m.M.parquet")),
            "train_cells_S": C.sha256_file(str(C.SFEC0_DATA / "cells_pcqm_train_1m.S.parquet")),
            "train_cells_C": C.sha256_file(str(C.SFEC0_DATA / "cells_pcqm_train_1m.C.parquet")),
            "labels": C.sha256_file(C.SFEC1_DATA / "pcqm_labels.npy"),
            "train_gids_1m": C.sha256_file(C.TRAIN_GIDS),
            "subset_rows_1m": C.sha256_file(C.SUBSET_ROWS),
            "checkpoint": C.sha256_file(C.LCW_CHECKPOINT),
            "config": C.sha256_file(C.CONFIG_PATH),
        },
        "identity_checks": {
            "all_plan_rows_in_frozen_1m_plan": True,
            "rectangles_four_distinct_molecules": True,
            "plain_overlap_with_endpoints": 0,
            "test_rows_indexed": 0,
            "labels_used_for_selection": False,
            "predictions_used_for_selection": False,
        },
        "notes": [
            "The rectangle set is a fixed-hash subset of the existing Train matching directory; no data was rebuilt.",
            "Rectangles are not required to have equal E_AB or delta_g.",
            "Plain molecules are a fixed-hash sample of the original 1M Train plan, matched 4:1 to rectangles.",
            "The plan contains real Train labels for continued training; no pseudo-label or teacher output exists.",
        ],
    }
    C.write_json(Path(args.manifest), manifest)
    _log(t_start, f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), "
                  f"sha256 {plan_sha}; manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
