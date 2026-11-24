"""Reading the frozen Valid observations, W3 predictions and rectangles.

The heavy lifting is *not* re-implemented: ``pcqm_response._build_valid_observations``
reconstructs the SFEC v1 Valid cells, contexts, canonical pairs, support masks
and the frozen ZERO / PAIR / PAIR_Q / LOCAL estimates, and
``pcqm_response._verify_frozen_metrics`` proves they still reproduce
``valid_transfer.json``.  This module adds only what this round needs:

* one frozen W3 forward on the existing official Valid cache (read-only),
* the LCRC delta of every pair observation in canonical orientation,
* context-level attributes (q, seen-in-1M-prefix, global descriptor, molecule
  identity, clean flags),
* a fixed-hash sample of context pairs ("rectangles") per canonical pair type.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping

import numpy as np

import ssa_common as C

import pcqm_response as PR


def _torch():
    import torch

    return torch


def load_w3() -> dict[str, Any]:
    """Frozen W3 checkpoint, cache and a single Valid inference pass.

    This is exactly the parent path (``pcqm_response._infer_w3``): build the
    scale-3 model with the frozen initialisation, load the frozen
    ``checkpoint_best_valid.pt`` (strict), predict every Valid row with
    ``pq_train.predict_rows`` (batch 1024) and convert z -> eV with the frozen
    label normalisation.  The checkpoint Valid MAE must be reproduced.
    """
    SC, SD, SV, EC, EM, PQD, PQT, FS = PR._imports()
    torch = _torch()
    threads = C.set_torch_threads()

    spec = FS.load_spec(C.LC_OLD_ANALYSIS / "pcqm_spec.json")
    fn = C.read_json(C.LC_OLD_ANALYSIS / "pcqm_feature_norm.json")
    norm = {
        k: np.asarray(fn[k + "_list"], np.float32)
        for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale")
    }
    label_norm = C.read_json(C.LC_OLD_ANALYSIS / "pcqm_label_norm.json")
    model, audit = EM.build_model(spec, norm, scale=3, device="cpu")
    payload = torch.load(C.LCW_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    labels_all = SC.load_pcqm_labels_cached()
    cache = PQD.load_cache(C.LC_OLD_CACHE / "pcqm_valid")
    gids = np.asarray(cache["gids"], np.int64)
    t0 = time.perf_counter()
    pred_z, info = PQT.predict_rows(
        model, cache, np.arange(int(cache["n_molecules"])), torch.device("cpu"),
        batch_size=1024,
    )
    seconds = time.perf_counter() - t0
    pred = PQD.from_z(pred_z, label_norm)
    y = np.asarray(labels_all[gids], np.float64)
    mae = float(np.mean(np.abs(pred - y)))
    if abs(mae - float(payload["valid_mae_eV"])) > 1e-6:
        raise RuntimeError(
            "frozen W3 inference does not reproduce the checkpoint Valid MAE: "
            f"{mae} vs {payload['valid_mae_eV']}"
        )
    return {
        "model": model,
        "spec": spec,
        "norm": norm,
        "label_norm": label_norm,
        "cache": cache,
        "gids": gids,
        "pred_eV": pred,
        "pred_z": np.asarray(pred_z, np.float64),
        "y_eV": y,
        "prediction_fingerprint": {
            "n_rows": int(gids.size),
            "pred_eV_sha256": hashlib.sha256(
                np.ascontiguousarray(pred, "<f8").tobytes()).hexdigest(),
            "pred_z_sha256": hashlib.sha256(
                np.ascontiguousarray(np.asarray(pred_z, np.float64), "<f8").tobytes()).hexdigest(),
            "y_eV_sha256": hashlib.sha256(
                np.ascontiguousarray(y, "<f8").tobytes()).hexdigest(),
            "pred_eV_mean": float(pred.mean()),
            "pred_eV_std": float(pred.std()),
            "y_eV_mean": float(y.mean()),
            "note": "the full Valid arrays stay in memory for this run only; "
                    "these digests replace them in state_replay.json",
        },
        "mae_eV": mae,
        "seconds": float(seconds),
        "batch_seconds": float(info["seconds"]),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_valid_mae_eV": float(payload["valid_mae_eV"]),
        "checkpoint_sha256": C.sha256_file(C.LCW_CHECKPOINT),
        "model_name": str(audit["model_name"]),
        "scale": int(audit["scale"]),
        "parameters": int(audit["total_parameters"]),
        "widths": dict(audit["widths"]),
        "torch_threads": threads,
        "imports": (SC, SD, SV, EC, EM, PQD, PQT, FS),
    }


def build_observations(w3: Mapping[str, Any]) -> dict[str, Any]:
    """SFEC v1 Valid observation table + frozen-metric reproduction check."""
    SC, SD, SV, EC, EM, PQD, PQT, FS = w3["imports"]
    cfg = SC.load_config()
    frozen = C.read_json(C.SFEC_ANALYSIS / "valid_transfer.json")
    obs = PR._build_valid_observations(SC, SD, cfg)
    verify = PR._verify_frozen_metrics(obs, frozen)
    obs["frozen_verify"] = verify
    obs["frozen"] = frozen
    obs["cfg"] = cfg

    ctx = obs["ctx_v"]
    pairs = obs["pairs_v"]
    labels_all = obs["labels"]
    valid_M = obs["valid_M"]
    gids = w3["gids"]
    pred = w3["pred_eV"]

    # ---- molecule identity of every cell position (label aggregation rule) --
    by_row = np.full(int(gids.max()) + 1, np.nan, np.float64)
    by_row[gids] = pred
    cell_mol = ctx.cell_mol_first.astype(np.int64)
    cell_row = np.asarray(valid_M["row_id"])[cell_mol].astype(np.int64)
    if not np.isfinite(by_row[cell_row]).all():
        raise RuntimeError("a Valid cell molecule has no frozen W3 prediction")
    cell_pred = by_row[cell_row]
    obs["cell_mol"] = cell_mol
    obs["cell_row"] = cell_row
    obs["cell_pred_eV"] = cell_pred

    clean = SD.molecule_clean_flags(obs["valid_cells"], valid_M)
    mol_clean = np.asarray(clean["clean"], bool)
    obs["mol_clean"] = mol_clean
    obs["cell_clean"] = (ctx.cell_mol_n.astype(np.int64) == 1) & mol_clean[np.maximum(cell_mol, 0)]

    # ---- LCRC canonical response (identical formula to the parent round) ----
    ri = ctx.S[pairs["pos_i"]]
    rj = ctx.S[pairs["pos_j"]]
    rank_i = obs["rank"][ri].astype(np.int64)
    rank_j = obs["rank"][rj].astype(np.int64)
    sign = np.where(rank_i < rank_j, 1.0, -1.0)
    delta_pred = sign * (cell_pred[pairs["pos_j"]] - cell_pred[pairs["pos_i"]])
    if not np.isfinite(delta_pred).all():
        raise RuntimeError("non-finite LCRC delta prediction")
    obs["delta_pred"] = delta_pred
    obs["rank_i"] = rank_i
    obs["rank_j"] = rank_j

    # ---- context-level attributes -----------------------------------------
    ctxC = ctx.C.astype(np.int64)
    valid_C = SD.load_C(obs["valid_prefix"], columns=["q_id", "hash64", "C_key"])
    train_C = obs["train_C"]
    train_hash = set(int(x) for x in np.asarray(train_C["hash64"]).tolist())
    ctx_hash = np.asarray(valid_C["hash64"])[ctxC]
    ctx_seen = np.asarray([int(h) in train_hash for h in ctx_hash.tolist()], bool)
    obs["ctx_hash"] = ctx_hash
    obs["ctx_seen"] = ctx_seen
    #: q used by the frozen PAIR_Q lookup, mapped into the *Train* q vocabulary
    #: exactly as the parent round does (``obs["q_safe"]`` / ``obs["q_row"]``).
    obs["ctx_q"] = obs["q_safe"][pairs["ctx"]].astype(np.int64)
    if not np.array_equal(obs["q_row"], obs["ctx_q"]):
        raise RuntimeError("context q descriptor differs from the parent q_row")
    #: context-level q (one value per exact context; verified consistent).
    #: Contexts with fewer than two supported cells never appear in a pair, so
    #: they stay -1 and are never used by the rectangle sampler.
    ctx_q = np.full(int(ctx.n_ctx), -1, np.int64)
    ctx_q[pairs["ctx"]] = obs["ctx_q"]
    if not np.array_equal(ctx_q[pairs["ctx"]], obs["ctx_q"]):
        raise RuntimeError("q is not a function of the exact context")
    obs["ctx_q_of_ctx"] = ctx_q

    #: whole-molecule graph descriptor and SMILES of every cell position.
    #: ``valid_M["row_id"]`` is the official dataset row (gid); the cache is
    #: indexed by its own row order, so the two are mapped explicitly.
    row_of_gid = np.full(int(gids.max()) + 1, -1, np.int64)
    row_of_gid[gids] = np.arange(gids.size, dtype=np.int64)
    cell_cache_row = row_of_gid[cell_row]
    if (cell_cache_row < 0).any():
        raise RuntimeError("a Valid cell molecule is not in the Valid cache")
    obs["cell_cache_row"] = cell_cache_row
    obs["mol_row_gid"] = np.asarray(valid_M["row_id"], np.int64)
    obs["mol_cache_row"] = row_of_gid[obs["mol_row_gid"]]
    obs["cell_global"] = np.asarray(w3["cache"]["global"])[cell_cache_row]
    obs["cell_atom_count"] = np.diff(np.asarray(w3["cache"]["atom_ptr"]))[cell_cache_row].astype(np.int64)
    smiles = w3["cache"]["smiles"]
    obs["cell_smiles"] = [str(smiles[int(r)]) for r in cell_cache_row.tolist()]

    obs["w3"] = {
        k: w3[k] for k in (
            "mae_eV", "seconds", "batch_seconds", "checkpoint_epoch",
            "checkpoint_valid_mae_eV", "checkpoint_sha256", "model_name", "scale",
            "parameters", "widths", "torch_threads", "label_norm", "norm",
            "prediction_fingerprint",
        )
    }
    return obs




# ---------------------------------------------------------------------------
# fixed-hash rectangle sampling
# ---------------------------------------------------------------------------

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xC2B2AE3D27D4EB4F)
_C3 = np.uint64(0x165667B19E3779F9)


def mix64(pair_key: np.ndarray, ctx1: np.ndarray, ctx2: np.ndarray) -> np.ndarray:
    """Fixed, label-free 64-bit mixing hash of ``(pair, context, context)``.

    Constants are the SplitMix64 finaliser; the function is a pure function of
    the three identifiers, so no label, prediction or residual can enter the
    sample selection.
    """
    z = (pair_key.astype(np.uint64) * _C1) ^ (ctx1.astype(np.uint64) * _C2) ^ (
        ctx2.astype(np.uint64) * _C3
    )
    z ^= z >> np.uint64(30)
    z *= _M1
    z ^= z >> np.uint64(27)
    z *= _M2
    z ^= z >> np.uint64(31)
    return z


def select_rectangles(obs: Mapping[str, Any], cap: int = C.MAX_RECTANGLES_PER_PAIR) -> dict[str, Any]:
    """At most ``cap`` context pairs per canonical replacement type.

    A rectangle is two *different* exact contexts C1 < C2 (by context index) in
    which both members of the canonical pair (A, B) have a real, evaluable
    Valid cell.  Selection is by the fixed hash of ``(A, B, C1, C2)``; it never
    reads a label, a prediction or an error.
    """
    ctx = obs["ctx_v"]
    pairs = obs["pairs_v"]
    key = obs["key"]
    main = obs["main"]
    sel = np.flatnonzero(main)
    k_sel = key[sel].astype(np.int64)
    c_sel = pairs["ctx"][sel].astype(np.int64)
    order = np.lexsort((c_sel, k_sel))
    sel = sel[order]
    k_sel = k_sel[order]
    c_sel = c_sel[order]
    if len(sel):
        starts = np.flatnonzero(np.r_[True, k_sel[1:] != k_sel[:-1]])
        ends = np.r_[starts[1:], len(sel)]
    else:  # pragma: no cover - Valid always has evaluable pairs
        starts = ends = np.zeros(0, np.int64)

    out = {name: [] for name in
           ("pair_key", "ctx1", "ctx2", "row1", "row2", "pos1_i", "pos1_j",
            "pos2_i", "pos2_j", "hash")}
    pair_types = []
    n_contexts_per_pair = []
    n_possible = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        k = int(k_sel[s])
        cs = c_sel[s:e]
        # contexts are sorted and unique per (pair, context) in the parent table
        if len(cs) < 2:
            continue
        i, j = np.triu_indices(len(cs), 1)
        h = mix64(np.full(len(i), k, np.int64), cs[i], cs[j])
        take = np.argsort(h, kind="stable")[: int(cap)]
        take = take[np.argsort(h[take], kind="stable")]
        rows = sel[s:e]
        p1 = rows[i[take]]
        p2 = rows[j[take]]
        out["pair_key"].append(np.full(len(take), k, np.int64))
        out["ctx1"].append(cs[i[take]])
        out["ctx2"].append(cs[j[take]])
        out["row1"].append(p1)
        out["row2"].append(p2)
        out["pos1_i"].append(pairs["pos_i"][p1])
        out["pos1_j"].append(pairs["pos_j"][p1])
        out["pos2_i"].append(pairs["pos_i"][p2])
        out["pos2_j"].append(pairs["pos_j"][p2])
        out["hash"].append(h[take])
        pair_types.append(k)
        n_contexts_per_pair.append(len(cs))
        n_possible.append(int(len(i)))

    rect = {name: (np.concatenate(v) if v else np.zeros(0, np.int64))
            for name, v in out.items()}
    for name in ("pair_key", "ctx1", "ctx2", "row1", "row2", "pos1_i", "pos1_j",
                 "pos2_i", "pos2_j"):
        rect[name] = rect[name].astype(np.int64)
    rect["hash"] = rect["hash"].astype(np.uint64)
    rect["A"] = (rect["pair_key"] // 64).astype(np.int64)
    rect["B"] = (rect["pair_key"] % 64).astype(np.int64)
    rect["same_q"] = obs["ctx_q_of_ctx"][rect["ctx1"]] == obs["ctx_q_of_ctx"][rect["ctx2"]]
    rect["unseen_both"] = (~obs["ctx_seen"][rect["ctx1"]]) & (~obs["ctx_seen"][rect["ctx2"]])
    rect["clean"] = (
        obs["cell_clean"][rect["pos1_i"]]
        & obs["cell_clean"][rect["pos1_j"]]
        & obs["cell_clean"][rect["pos2_i"]]
        & obs["cell_clean"][rect["pos2_j"]]
    )

    # response columns (cell-level aggregation, canonical orientation) --------
    d = pairs["delta"]
    dp = obs["delta_pred"]
    pq = obs["pair_q"]
    loc = obs["preds"]["LOCAL"]
    for tag, rows in (("1", rect["row1"]), ("2", rect["row2"])):
        rect[f"delta_true_{tag}"] = d[rows]
        rect[f"delta_lcrc_{tag}"] = dp[rows]
        rect[f"delta_pairq_{tag}"] = pq[rows]
        rect[f"delta_local_{tag}"] = loc[rows]
    rect["q1"] = obs["ctx_q_of_ctx"][rect["ctx1"]]
    rect["q2"] = obs["ctx_q_of_ctx"][rect["ctx2"]]

    return {
        "rect": rect,
        "pair_types": np.asarray(pair_types, np.int64),
        "n_contexts_per_pair": np.asarray(n_contexts_per_pair, np.int64),
        "n_possible_per_pair": np.asarray(n_possible, np.int64),
        "cap": int(cap),
        "selection": (
            "per canonical pair type, all unordered context pairs of evaluable "
            "pair observations, ranked by a fixed SplitMix64 hash of "
            "(pair_key, ctx1, ctx2); first <=64 kept. Labels, W3 predictions "
            "and errors are not read by the sampler."
        ),
    }


__all__ = ["build_observations", "load_w3", "mix64", "select_rectangles"]
