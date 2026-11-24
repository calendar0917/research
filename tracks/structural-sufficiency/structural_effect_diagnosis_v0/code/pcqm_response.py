"""PCQM4Mv2: does an existing non-MP whole-graph model predict real replacements?

Frozen W3 model (lcrc_pcqm4mv2_efficiency_width_v1, checkpoint_best_valid.pt,
epoch 39, official Valid MAE 0.127809167 eV) is evaluated once on the existing
official Valid cache.  No model is trained, fine-tuned or selected by any
response score from this round.

The SFEC v1 valid pair observations, cell de-duplication, label aggregation,
support masks and frozen conditional estimates are reconstructed read-only and
verified against ``valid_transfer.json`` before any LCRC delta is computed.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import sed_common as C

MODELS = ("ZERO", "PAIR", "PAIR_Q", "LOCAL", "LCRC")
COMPARISONS = (("LCRC", "PAIR_Q"), ("LCRC", "PAIR"), ("LCRC", "LOCAL"),
               ("LCRC", "ZERO"), ("PAIR_Q", "PAIR"))
THRESHOLDS = (0.10, 0.05)
MIN_VALID_CONTEXTS = 5


def _imports():
    sys.path.insert(0, str(C.SFEC_CODE))
    import sfec1_common as SC
    import sfec1_data as SD
    import sfec1_valid as SV

    sys.path.insert(0, str(C.LCW_CODE))
    import ew_common as EC
    import ew_model as EM
    import pq_data as PQD
    import pq_train as PQT
    import feature_spec as FS

    return SC, SD, SV, EC, EM, PQD, PQT, FS


def _asset_check() -> tuple[bool, list[str]]:
    required = {
        "W3_checkpoint": C.LCW_CHECKPOINT,
        "W3_run_json": C.LCW_RUN / "run.json",
        "valid_cache": C.LC_OLD_CACHE / "pcqm_valid",
        "train1m_cache": C.LC_OLD_CACHE / "pcqm_train1m",
        "subset_rows": C.LC_OLD_CACHE / "pcqm_subset_rows.npy",
        "labels_trainvalid": C.LC_OLD_CACHE / "pcqm_labels_trainvalid.npy",
        "sfec_prepared": C.SFEC_DATA / "prepared_pcqm.npz",
        "sfec_valid_transfer": C.SFEC_ANALYSIS / "valid_transfer.json",
        "sfec_summary": C.SFEC_ANALYSIS / "summary.json",
        "sfec_candidates_csv": C.SFEC_ANALYSIS / "conditional_function_candidates.csv",
        "parent_valid_cells": C.SFEC_DIR.parent / "structural_functional_equivalence_v0"
                              / "data" / "cells_pcqm_valid.npz",
    }
    missing = [str(p) for p in required.values() if not Path(p).exists()]
    return (not missing), missing


def _infer_w3(EC, EM, PQD, PQT, FS, SC, labels_all: np.ndarray) -> dict[str, Any]:
    import torch

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    spec = FS.load_spec(C.LC_OLD_ANALYSIS / "pcqm_spec.json")
    fn = json.loads((C.LC_OLD_ANALYSIS / "pcqm_feature_norm.json").read_text())
    norm = {k: np.asarray(fn[k + "_list"], np.float32) for k in
            ("global_mean", "global_scale", "scalar_mean", "scalar_scale")}
    label_norm = json.loads((C.LC_OLD_ANALYSIS / "pcqm_label_norm.json").read_text())

    model, audit = EM.build_model(spec, norm, scale=3, device="cpu")
    payload = torch.load(C.LCW_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    cache = PQD.load_cache(C.LC_OLD_CACHE / "pcqm_valid")
    gids = np.asarray(cache["gids"], np.int64)
    y = np.asarray(labels_all[gids], np.float64)
    t0 = time.perf_counter()
    pred_z, info = PQT.predict_rows(
        model, cache, np.arange(int(cache["n_molecules"])), torch.device("cpu"),
        batch_size=1024,
    )
    seconds = time.perf_counter() - t0
    pred = PQD.from_z(pred_z, label_norm)
    mae = float(np.mean(np.abs(pred - y)))
    return {
        "gids": gids,
        "pred": pred,
        "y": y,
        "mae_eV": mae,
        "inference_seconds": float(seconds),
        "batch_seconds": float(info["seconds"]),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_valid_mae_eV": float(payload["valid_mae_eV"]),
        "checkpoint_sha256": C.sha256_file(C.LCW_CHECKPOINT),
        "n_rows": int(gids.size),
        "model_parameters": int(audit["total_parameters"]),
        "model_name": str(audit["model_name"]),
        "scale": int(audit["scale"]),
        "torch_threads": int(torch.get_num_threads()),
    }


def _build_valid_observations(SC, SD, cfg) -> dict[str, Any]:
    """Reconstruct SFEC v1 Valid observations from frozen parent tables."""
    prep = {k: v for k, v in np.load(C.SFEC_DATA / "prepared_pcqm.npz").items()}
    # Paths in the SFEC config are relative to the SFEC round directory.
    train_prefix = (C.SFEC_DIR / Path(cfg["datasets"]["pcqm"]["train_prefix"])).resolve()
    valid_prefix = (C.SFEC_DIR / Path(cfg["datasets"]["pcqm"]["valid_prefix"])).resolve()

    train_uni = SD.load_universe(train_prefix)
    train_S = SD.load_S(train_prefix)
    train_C = SD.load_C(train_prefix, columns=["hash64"])
    train_M = SD.load_M(train_prefix, columns=["mol_key", "row_id"])
    valid_cells = SD.load_cells(valid_prefix)
    valid_S = SD.load_S(valid_prefix)
    valid_C = SD.load_C(valid_prefix, columns=["q_id", "hash64", "C_key"])
    valid_M = SD.load_M(valid_prefix, columns=["mol_id", "mol_key", "row_id"])
    labels = SC.load_pcqm_labels_cached()
    primary = train_uni["primary"]
    rank = train_uni["rank"]
    n_s = len(train_S["S_id"])

    s_digest = {SD.digest(str(k)): int(i) for i, k in enumerate(train_S["S_key"].tolist())}
    vmap = np.asarray([s_digest.get(SD.digest(str(k)), -1)
                       for k in valid_S["S_key"].tolist()], np.int64)
    train_mol = set(SD.digest(str(k)) for k in train_M["mol_key"].tolist())
    valid_mol_dup = np.asarray([SD.digest(str(k)) in train_mol
                                for k in valid_M["mol_key"].tolist()], bool)
    cells_v = {k: v.copy() for k, v in valid_cells.items()}
    mapped = vmap[cells_v["S_id"].astype(np.int64)]
    cells_v["S_id"] = mapped
    seen_s = mapped >= 0
    dup_mol = valid_mol_dup[cells_v["mol_id"].astype(np.int64)]
    mask = seen_s & primary[np.maximum(mapped, 0)] & ~dup_mol
    ctx_v = SD.build_contexts(cells_v, mask, labels, n_s, primary, rank)
    pairs_v = SD.build_pairs(ctx_v, rank, canonical=True)

    # Multiplicity of every Valid cell (before the primary/duplicate filter):
    # the model prediction is averaged over exactly the distinct molecule
    # identities used by the label aggregation.
    _mol = valid_cells["mol_id"].astype(np.int64)
    _stride = max(int(n_s), int(valid_cells["S_id"].max()) + 1)
    _key_all = ((valid_cells["C_id"].astype(np.int64) * int(_stride)
                 + valid_cells["S_id"].astype(np.int64)) * 8
                + valid_cells["bond_type"].astype(np.int64))
    _order = np.lexsort((_mol, _key_all))
    _k = _key_all[_order]
    _m = _mol[_order]
    _first = np.r_[True, (_k[1:] != _k[:-1]) | (_m[1:] != _m[:-1])]
    _uk = _k[_first]
    _u, _cnt = np.unique(_uk, return_counts=True)
    valid_cell_multiplicity_max = int(_cnt.max()) if _cnt.size else 0
    valid_cells_multi = int((_cnt > 1).sum())

    train_q = SC.read_json(Path(str(train_prefix) + ".keys.json"))["q_keys"]
    valid_q = SC.read_json(Path(str(valid_prefix) + ".keys.json"))["q_keys"]
    q_map = {SD.digest(str(k)): int(i) for i, k in enumerate(train_q)}
    vq = np.asarray([q_map.get(SD.digest(str(k)), -1) for k in valid_q], np.int64)
    q_ctx = vq[valid_C["q_id"][ctx_v.C].astype(np.int64)]
    n_q = int(prep["n_q"][0])
    q_safe = np.where(q_ctx >= 0, q_ctx, n_q)

    ds = {k: prep[f"disc_{k}"] for k in
          ("pair_key", "A", "B", "n", "median", "mean", "mad", "p80_abs_dev",
           "sd", "sign_pos")}
    tri = {
        "key": prep["triple_true_key"], "n": prep["triple_true_n"],
        "median": prep["triple_true_median"], "mean": prep["triple_true_mean"],
        "p80_abs_dev": prep["triple_true_p80"],
    }
    bvec = prep["shared_b"]
    comp = prep["shared_comp"]
    local_fit = {
        "d": np.vstack([prep["local_d"], np.zeros((1, 64))]),
        "comp": np.vstack([prep["local_comp"], -np.ones((1, 64), dtype=prep["local_comp"].dtype)]),
        "supported": np.r_[prep["local_supported"], False],
    }
    null_fit = {
        "d": np.vstack([prep["local_null_d"], np.zeros((1, 64))]),
        "comp": np.vstack([prep["local_null_comp"], -np.ones((1, 64), dtype=prep["local_null_comp"].dtype)]),
        "supported": np.r_[prep["local_null_supported"], False],
    }

    A = pairs_v["A"].astype(np.int64)
    B = pairs_v["B"].astype(np.int64)
    key = SD.pair_key(A, B)
    mu = SD.lookup_pair(ds, key, "median")
    n_disc = SD.lookup_pair_count(ds, key)
    main = (n_disc >= 10) & (comp[A] >= 0) & (comp[A] == comp[B])
    q_row = q_safe[pairs_v["ctx"]]
    tmed, tcnt = SD.lookup_triple(tri, A, B, q_row, n_q, "median")
    pair_q = np.where(tcnt >= 10, tmed, mu)
    pred_local = SD.predict_local(pairs_v["ctx"], A, B, q_safe, bvec, local_fit)
    preds = {
        "ZERO": np.zeros(len(key), np.float64),
        "PAIR": mu,
        "PAIR_Q": pair_q,
        "LOCAL": pred_local["pred"],
    }

    train_hash = set(int(x) for x in train_C["hash64"].tolist())
    seen_c_ctx = np.asarray([int(h) in train_hash for h in valid_C["hash64"][ctx_v.C].tolist()], bool)
    seen_c_row = seen_c_ctx[pairs_v["ctx"]]
    clean_flags = SD.molecule_clean_flags(valid_cells, valid_M)
    pair_clean = SD.pair_clean_mask(ctx_v, pairs_v, clean_flags["clean"])
    q_supported_row = (q_row < n_q) & local_fit["supported"][np.minimum(q_row, n_q - 1)]
    subsets = {
        "all_evaluable": main,
        "unseen_exact_context": main & ~seen_c_row,
        "unseen_context_q_supported": main & ~seen_c_row & q_supported_row,
        "clean_subset": main & pair_clean,
    }
    return {
        "prep": prep, "n_q": n_q, "train_prefix": train_prefix,
        "rank": rank, "valid_prefix": valid_prefix, "train_M": train_M, "train_C": train_C,
        "valid_cells": valid_cells, "valid_M": valid_M, "labels": labels,
        "ctx_v": ctx_v, "pairs_v": pairs_v, "key": key, "main": main,
        "q_row": q_row, "q_safe": q_safe, "pair_q": pair_q,
        "preds": preds, "subsets": subsets, "n_disc": n_disc,
        "n_cells_unmapped_S": int((~seen_s).sum()),
        "n_cells_duplicate_train_molecule": int(dup_mol.sum()),
        "n_cells_used": int(mask.sum()),
        "valid_cell_multiplicity_max": valid_cell_multiplicity_max,
        "valid_cells_with_multiple_molecules_all": valid_cells_multi,
    }


def _verify_frozen_metrics(obs: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    out = {}
    max_diff = 0.0
    for label, m in obs["subsets"].items():
        ref = frozen["subsets"][label]
        rec = {
            "n_obs": int(m.sum()),
            "n_obs_frozen": int(ref["n_obs"]),
            "models": {},
        }
        if rec["n_obs"] != rec["n_obs_frozen"]:
            raise RuntimeError(f"{label}: observation count does not match SFEC")
        for name in ("ZERO", "PAIR", "PAIR_Q", "LOCAL"):
            mae = float(np.mean(np.abs(obs["preds"][name][m] - obs["pairs_v"]["delta"][m])))
            diff = abs(mae - float(ref["points"][name]["observation"]["mae"]))
            max_diff = max(max_diff, diff)
            rec["models"][name] = {"mae": mae, "frozen_mae": float(ref["points"][name]["observation"]["mae"]),
                                   "abs_diff": diff}
        out[label] = rec
    if max_diff > 1e-12:
        raise RuntimeError(f"frozen SFEC metric reproduction failed: max diff {max_diff}")
    return {"subsets": out, "max_abs_diff": float(max_diff)}


def _metrics(delta: np.ndarray, pred: np.ndarray, row_ctx: np.ndarray,
             n_ctx: int, pair_idx: np.ndarray) -> dict[str, float]:
    base = SD_basic_metric(delta, pred)
    ae = np.abs(pred - delta)
    sd = float(np.std(delta))
    sp = float(np.std(pred))
    corr = float(np.corrcoef(delta, pred)[0, 1]) if sd > 0 and sp > 0 else float("nan")
    slope = float(np.polyfit(delta, pred, 1)[0]) if sd > 0 else float("nan")
    return {
        "mae": base["mae"], "rmse": base["rmse"],
        "sign_accuracy": base["sign_accuracy"],
        "context_equal_mae": float(SD_context_equal_mae(row_ctx, ae, n_ctx)),
        "pair_type_mae": float(SD_pair_macro_mae(pair_idx, ae)),
        "delta_correlation": corr,
        "delta_regression_slope": slope,
        "delta_std_ratio": sp / sd if sd > 0 else float("nan"),
    }


# The two helpers below are rebound to the imported sfec1_data module inside
# run(); keeping module-level names makes the metric formulas explicit.
def SD_basic_metric(delta, pred):  # pragma: no cover - rebound at runtime
    raise RuntimeError("not bound")


def SD_context_equal_mae(row_ctx, ae, n_ctx):  # pragma: no cover
    raise RuntimeError("not bound")


def SD_pair_macro_mae(pair_idx, ae):  # pragma: no cover
    raise RuntimeError("not bound")


def _components(mol_i: np.ndarray, mol_j: np.ndarray) -> np.ndarray:
    n = len(mol_i)
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    first: dict[int, int] = {}
    for idx in range(n):
        for mol in (int(mol_i[idx]), int(mol_j[idx])):
            if mol in first:
                ra, rb = find(idx), find(first[mol])
                if ra != rb:
                    parent[rb] = ra
            else:
                first[mol] = idx
    return np.asarray([find(i) for i in range(n)], np.int64)


def _dependency_block(SD, delta, preds, mol_i, mol_j, *, seed: int) -> dict[str, Any]:
    comp = _components(mol_i, mol_j)
    u, inv = np.unique(comp, return_inverse=True)
    n_comp = len(u)
    sizes = np.bincount(inv)
    boot = SD.bootstrap_obs_context(
        delta, preds, inv.astype(np.int64), n_comp, context_equal=False,
        reps=1000, seed=seed, comparisons=[("LCRC", "PAIR_Q")],
    )
    return {
        "n_observations": int(len(delta)),
        "n_components": int(n_comp),
        "largest_component": int(sizes.max()),
        "largest_component_fraction": float(sizes.max() / max(1, len(delta))),
        "component_bootstrap_observation": boot,
        "note": ("Components are shared-molecule connected components of the "
                 "evaluable observations; this interval is a dependence-robust "
                 "sensitivity, not a replacement for the context bootstrap."),
    }


def _sign_reversal_groups(prep: dict[str, Any], thr: float) -> list[dict[str, Any]]:
    n_q = int(prep["n_q"][0])
    tri_key = prep["triple_true_key"].astype(np.int64)
    tri_n = prep["triple_true_n"].astype(np.int64)
    tri_med = prep["triple_true_median"].astype(np.float64)
    tri_pair = tri_key // n_q
    tri_q = tri_key % n_q
    sup = tri_n >= 10
    pairs = []
    for k in np.unique(tri_pair[sup]).tolist():
        m = (tri_pair == k) & sup
        meds = tri_med[m]
        qs = tri_q[m]
        pos = np.unique(qs[meds > thr])
        neg = np.unique(qs[meds < -thr])
        if len(pos) == 0 or len(neg) == 0:
            continue
        rec = {"pair_key": int(k), "A": int(k) // 64, "B": int(k) % 64,
               "groups": []}
        for qq in np.r_[pos, neg].tolist():
            t = (tri_pair == k) & (tri_q == qq) & sup
            rec["groups"].append({
                "q": int(qq), "discovery_median": float(tri_med[t][0]),
                "n_discovery": int(tri_n[t][0]),
                "sign": 1 if qq in set(pos.tolist()) else -1,
            })
        pairs.append(rec)
    return pairs


def _eval_groups(groups: list[dict[str, Any]], obs: dict[str, Any],
                 threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    key = obs["key"]
    q_row = obs["q_row"]
    main = obs["main"]
    ctx = obs["pairs_v"]["ctx"]
    delta = obs["pairs_v"]["delta"]
    delta_pred = obs["delta_pred"]
    pair_q = obs["pair_q"]
    rows = []
    n_groups = 0
    n_groups_any = 0
    n_groups_supported = 0
    n_obs_supported = 0
    n_pairs = len(groups)
    n_pairs_any = 0
    pair_records = []
    group_true_sign = 0
    group_lcrc_sign = 0
    group_lcrc_true_sign = 0
    abs_true_vs_disc = []
    abs_lcrc_vs_true = []
    abs_pq_vs_true = []
    for rec in groups:
        n_groups += len(rec["groups"])
        any_obs = False
        supported = []
        for g in rec["groups"]:
            m = (key == rec["pair_key"]) & (q_row == g["q"]) & main
            n = int(m.sum())
            n_ctx = int(len(np.unique(ctx[m]))) if n else 0
            if n:
                any_obs = True
                n_groups_any += 1
            vt = float(np.median(delta[m])) if n else float("nan")
            vl = float(np.median(delta_pred[m])) if n else float("nan")
            pq = float(np.median(pair_q[m])) if n else float("nan")
            is_sup = n_ctx >= MIN_VALID_CONTEXTS
            if is_sup:
                n_groups_supported += 1
                n_obs_supported += n
                st = bool(np.sign(vt) == np.sign(g["discovery_median"]))
                sl = bool(np.sign(vl) == np.sign(g["discovery_median"]))
                slt = bool(np.sign(vl) == np.sign(vt))
                group_true_sign += int(st)
                group_lcrc_sign += int(sl)
                group_lcrc_true_sign += int(slt)
                abs_true_vs_disc.append(abs(vt - g["discovery_median"]))
                abs_lcrc_vs_true.append(abs(vl - vt))
                abs_pq_vs_true.append(abs(pq - vt))
                supported.append((g["sign"], vt, vl, is_sup))
            rows.append({
                "threshold": threshold, "A_rank": rec["A"], "B_rank": rec["B"],
                "q_id": g["q"], "discovery_median": g["discovery_median"],
                "n_discovery": g["n_discovery"],
                "valid_n_obs": n, "valid_n_contexts": n_ctx,
                "valid_true_median": vt, "lcrc_median": vl,
                "pair_q_prediction": pq,
                "supported": bool(is_sup),
                "sign_true_matches_discovery": (bool(np.sign(vt) == np.sign(g["discovery_median"]))
                                                 if n else None),
                "sign_lcrc_matches_discovery": (bool(np.sign(vl) == np.sign(g["discovery_median"]))
                                                 if n else None),
                "sign_lcrc_matches_true": (bool(np.sign(vl) == np.sign(vt)) if n else None),
            })
        if any_obs:
            n_pairs_any += 1
        pos_sup = [x for x in supported if x[0] > 0]
        neg_sup = [x for x in supported if x[0] < 0]
        if pos_sup and neg_sup:
            pair_records.append((rec["A"], rec["B"], pos_sup, neg_sup))
    # pair-level reversal
    pair_true_both = sum(1 for a, b, p, n in pair_records
                         if any(x[1] > 0 for x in p) and any(x[1] < 0 for x in n))
    pair_lcrc_both = sum(1 for a, b, p, n in pair_records
                         if any(x[2] > 0 for x in p) and any(x[2] < 0 for x in n))
    pair_true_strict = sum(1 for a, b, p, n in pair_records
                           if all(x[1] > 0 for x in p) and all(x[1] < 0 for x in n))
    pair_lcrc_strict = sum(1 for a, b, p, n in pair_records
                           if all(x[2] > 0 for x in p) and all(x[2] < 0 for x in n))
    summary = {
        "threshold": threshold,
        "n_frozen_pairs": int(n_pairs),
        "n_frozen_groups": int(n_groups),
        "n_groups_with_any_valid_obs": int(n_groups_any),
        "n_groups_supported": int(n_groups_supported),
        "n_obs_in_supported_groups": int(n_obs_supported),
        "n_pairs_with_any_valid_obs": int(n_pairs_any),
        "n_pairs_both_signs_supported": int(len(pair_records)),
        "group_sign_match_true": int(group_true_sign),
        "group_sign_match_lcrc": int(group_lcrc_sign),
        "group_sign_match_lcrc_true": int(group_lcrc_true_sign),
        "group_sign_match_true_rate": float(group_true_sign / max(1, n_groups_supported)),
        "group_sign_match_lcrc_rate": float(group_lcrc_sign / max(1, n_groups_supported)),
        "group_sign_match_lcrc_true_rate": float(group_lcrc_true_sign / max(1, n_groups_supported)),
        "pair_true_both_signs": int(pair_true_both),
        "pair_lcrc_both_signs": int(pair_lcrc_both),
        "pair_true_strict_signs": int(pair_true_strict),
        "pair_lcrc_strict_signs": int(pair_lcrc_strict),
        "median_abs_true_minus_discovery": float(np.median(abs_true_vs_disc)) if abs_true_vs_disc else None,
        "median_abs_lcrc_minus_true": float(np.median(abs_lcrc_vs_true)) if abs_lcrc_vs_true else None,
        "median_abs_pairq_minus_true": float(np.median(abs_pq_vs_true)) if abs_pq_vs_true else None,
        "insufficient_support_note": (
            "Groups with fewer than 5 Valid contexts are listed but excluded "
            "from sign-reproduction rates; this is a support limitation, not "
            "evidence that the model failed to learn the rule."
        ),
    }
    return summary, rows


def _nonzero_candidates(obs: dict[str, Any], tau: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = C.SFEC_ANALYSIS / "conditional_function_candidates.csv"
    rows = []
    key, q_row, main = obs["key"], obs["q_row"], obs["main"]
    ctx, delta, delta_pred, pair_q = (obs["pairs_v"]["ctx"], obs["pairs_v"]["delta"],
                                      obs["delta_pred"], obs["pair_q"])
    with path.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if abs(float(r["tau"]) - float(tau)) > 1e-12:
                continue
            if int(r["near_zero"]) != 0 or int(r["verifiable"]) != 1:
                continue
            A, B, q = int(r["A_rank"]), int(r["B_rank"]), int(r["q_id"])
            k = A * 64 + B
            m = (key == k) & (q_row == q) & main
            n = int(m.sum())
            n_ctx = int(len(np.unique(ctx[m]))) if n else 0
            vt = float(np.median(delta[m])) if n else float("nan")
            vl = float(np.median(delta_pred[m])) if n else float("nan")
            pq = float(np.median(pair_q[m])) if n else float("nan")
            rows.append({
                "tau": tau, "A_rank": A, "B_rank": B, "q_id": q,
                "A_key": r["A_key"], "B_key": r["B_key"], "q": r["q"],
                "discovery_median": float(r["discovery_median"]),
                "discovery_p80_abs_dev": float(r["discovery_p80_abs_dev"]),
                "n_discovery": int(r["n_discovery"]),
                "valid_n_obs": n, "valid_n_contexts": n_ctx,
                "valid_true_median": vt, "lcrc_median": vl,
                "pair_q_prediction": pq,
                "supported": bool(n_ctx >= MIN_VALID_CONTEXTS),
                "sign_true_matches_discovery": (bool(np.sign(vt) == np.sign(float(r["discovery_median"])))
                                                 if n else None),
                "sign_lcrc_matches_discovery": (bool(np.sign(vl) == np.sign(float(r["discovery_median"])))
                                                 if n else None),
            })
    supported = [x for x in rows if x["supported"]]
    summary = {
        "tau": tau,
        "n_candidates": int(len(rows)),
        "n_supported": int(len(supported)),
        "n_obs_supported": int(sum(x["valid_n_obs"] for x in supported)),
        "n_true_sign_match_supported": int(sum(bool(x["sign_true_matches_discovery"]) for x in supported)),
        "n_lcrc_sign_match_supported": int(sum(bool(x["sign_lcrc_matches_discovery"]) for x in supported)),
        "note": ("Only candidates with at least 5 Valid contexts are used for "
                 "sign comparisons; the rest are support-limited."),
    }
    return summary, rows


def run() -> dict[str, Any]:
    t_start = time.perf_counter()
    ok, missing = _asset_check()
    if not ok:
        return {"status": "ASSET_UNAVAILABLE", "missing": missing,
                "seconds": float(time.perf_counter() - t_start)}

    SC, SD, SV, EC, EM, PQD, PQT, FS = _imports()
    global SD_basic_metric, SD_context_equal_mae, SD_pair_macro_mae
    SD_basic_metric = SD.basic_metric
    SD_context_equal_mae = SD.context_equal_mae
    SD_pair_macro_mae = SD.pair_macro_mae

    cfg = SC.load_config()
    frozen = json.loads((C.SFEC_ANALYSIS / "valid_transfer.json").read_text())
    sfec_summary = json.loads((C.SFEC_ANALYSIS / "summary.json").read_text())

    labels_all = SC.load_pcqm_labels_cached()
    inference = _infer_w3(EC, EM, PQD, PQT, FS, SC, labels_all)
    if abs(inference["mae_eV"] - inference["checkpoint_valid_mae_eV"]) > 1e-6:
        raise RuntimeError(
            "frozen W3 inference does not reproduce the checkpoint Valid MAE: "
            f"{inference['mae_eV']} vs {inference['checkpoint_valid_mae_eV']}"
        )

    obs = _build_valid_observations(SC, SD, cfg)
    verify = _verify_frozen_metrics(obs, frozen)
    obs["labels"] = labels_all

    # ------------------------------------------------------------------
    # identity checks
    # ------------------------------------------------------------------
    train_plan = np.load(C.LC_OLD_CACHE / "pcqm_subset_rows.npy").astype(np.int64)
    train_gids = np.load(C.LC_OLD_CACHE / "pcqm_train1m" / "gids.npy").astype(np.int64)
    if not np.array_equal(train_plan[:1_000_000], train_gids):
        raise RuntimeError("W3 train cache is not the frozen 1M row-plan prefix")
    sfec_train_cells = SD.load_cells(obs["train_prefix"])
    sfec_train_row_ids = np.unique(sfec_train_cells["row_id"].astype(np.int64))
    sfec_train_mol_row_ids = np.unique(obs["train_M"]["row_id"].astype(np.int64))
    valid_M = obs["valid_M"]
    valid_row_ids = valid_M["row_id"].astype(np.int64)
    valid_gids = inference["gids"]
    labels_lcrc = np.load(C.LC_OLD_CACHE / "pcqm_labels_trainvalid.npy").astype(np.float64)
    label_diff = float(np.max(np.abs(labels_all[valid_gids] - labels_lcrc[valid_gids])))
    cell_label_diff = float(np.max(np.abs(
        obs["ctx_v"].y - labels_all[valid_row_ids[obs["ctx_v"].cell_mol_first.astype(np.int64)]]
    ))) if len(obs["ctx_v"].y) else 0.0
    identity = {
        "dataset": "pcqm4m-v2",
        "label_unit": "eV (raw HOMO-LUMO gap)",
        "valid_cache_rows": int(valid_gids.size),
        "valid_unique_molecules_in_M": int(len(valid_M["mol_id"])),
        "valid_M_row_ids_all_in_valid_cache": bool(np.isin(valid_row_ids, valid_gids).all()),
        "valid_M_row_ids_missing": int((~np.isin(valid_row_ids, valid_gids)).sum()),
        "sfec_train_cell_row_ids": int(sfec_train_row_ids.size),
        "sfec_train_cell_row_ids_subset_of_w3_plan": bool(np.isin(sfec_train_row_ids, train_gids).all()),
        "sfec_train_cell_row_ids_outside_w3_plan": int((~np.isin(sfec_train_row_ids, train_gids)).sum()),
        "sfec_train_molecule_row_ids": int(sfec_train_mol_row_ids.size),
        "sfec_train_molecule_row_ids_subset_of_w3_plan": bool(np.isin(sfec_train_mol_row_ids, train_gids).all()),
        "sfec_train_molecule_row_ids_outside_w3_plan": int((~np.isin(sfec_train_mol_row_ids, train_gids)).sum()),
        "w3_train_plan_rows": int(train_gids.size),
        "w3_train_plan_first_gid": int(train_gids[0]),
        "w3_train_plan_last_gid": int(train_gids[-1]),
        "w3_train_plan_equals_subset_rows_prefix": bool(np.array_equal(train_plan[:1_000_000], train_gids)),
        "sfec_labels_vs_lcrc_labels_max_abs_diff_eV": label_diff,
        "valid_cells_with_multiple_molecule_identities": int((obs["ctx_v"].cell_mol_n != 1).sum()),
        "valid_cells_with_multiple_molecules_all": int(obs["valid_cells_with_multiple_molecules_all"]),
        "valid_cell_multiplicity_max_all": int(obs["valid_cell_multiplicity_max"]),
        "cell_label_identity_max_abs_diff_eV": cell_label_diff,
        "n_cells_unmapped_S_matches_frozen": bool(
            obs["n_cells_unmapped_S"] == frozen["n_cells_unmapped_S"]),
        "n_cells_duplicate_train_molecule_matches_frozen": bool(
            obs["n_cells_duplicate_train_molecule"] == frozen["n_cells_duplicate_train_molecule"]),
        "n_cells_used_matches_frozen": bool(obs["n_cells_used"] == frozen["n_cells_used"]),
        "n_contexts_used_matches_frozen": bool(obs["ctx_v"].n_ctx == frozen["n_contexts_used"]),
        "n_pair_observations_matches_frozen": bool(len(obs["key"]) == frozen["n_pair_observations"]),
        "n_pair_observations_evaluable_matches_frozen": bool(obs["main"].sum() == frozen["n_pair_observations_evaluable"]),
        "test_labels_read": False,
        "test_policy": "official Test labels are hidden and were never indexed",
    }

    # ------------------------------------------------------------------
    # LCRC delta predictions from independent whole-molecule predictions
    # ------------------------------------------------------------------
    by_row = np.full(int(valid_gids.max()) + 1, np.nan, np.float64)
    by_row[valid_gids] = inference["pred"]
    pred_mol = by_row[valid_row_ids]
    if not np.isfinite(pred_mol).all():
        raise RuntimeError("missing W3 prediction for a Valid molecule")
    ctx = obs["ctx_v"]
    pairs = obs["pairs_v"]
    cell_pred = pred_mol[ctx.cell_mol_first.astype(np.int64)]
    ri = obs["ctx_v"].S[pairs["pos_i"]]
    rj = obs["ctx_v"].S[pairs["pos_j"]]
    rank_i = obs["rank"][ri].astype(np.int64)
    rank_j = obs["rank"][rj].astype(np.int64)
    sign = np.where(rank_i < rank_j, 1.0, -1.0)
    delta_pred = sign * (cell_pred[pairs["pos_j"]] - cell_pred[pairs["pos_i"]])
    if not np.isfinite(delta_pred).all():
        raise RuntimeError("non-finite LCRC delta prediction")
    obs["delta_pred"] = delta_pred
    obs["cell_pred"] = cell_pred
    obs["pred_mol"] = pred_mol

    # ------------------------------------------------------------------
    # frozen-response comparison on the SFEC observation sets
    # ------------------------------------------------------------------
    response_rows = []
    contrast_rows = []
    summary_subsets: dict[str, Any] = {}
    for label, mask in obs["subsets"].items():
        m = np.asarray(mask, bool)
        d = pairs["delta"][m]
        p = {name: np.asarray(obs["preds"][name][m], np.float64) for name in
             ("ZERO", "PAIR", "PAIR_Q", "LOCAL")}
        p["LCRC"] = delta_pred[m]
        row_ctx, n_ctx = SD.compact_ctx(pairs["ctx"][m])
        pair_u, pair_idx = np.unique(obs["key"][m], return_inverse=True)
        n_pairs = int(len(pair_u))
        rec: dict[str, Any] = {
            "n_obs": int(m.sum()), "n_contexts": int(n_ctx),
            "n_pairs": n_pairs, "models": {},
        }
        for name in MODELS:
            rec["models"][name] = _metrics(d, p[name], row_ctx, n_ctx, pair_idx)
            response_rows.append({
                "subset": label, "n_obs": int(m.sum()), "n_contexts": int(n_ctx),
                "n_pairs": n_pairs, "model": name, **rec["models"][name],
            })
        bo = SD.bootstrap_obs_context(
            d, p, row_ctx, n_ctx, context_equal=False, reps=1000, seed=0,
            comparisons=list(COMPARISONS),
        )
        bc = SD.bootstrap_obs_context(
            d, p, row_ctx, n_ctx, context_equal=True, reps=1000, seed=1,
            comparisons=list(COMPARISONS),
        )
        bp = SD.bootstrap_pair_macro(
            d, p, row_ctx, pair_idx, n_ctx, reps=1000, seed=2,
            comparisons=list(COMPARISONS),
        )
        rec["bootstrap"] = {"observation": bo, "context_equal": bc, "pair_type": bp}
        for a, b in COMPARISONS:
            k = f"{a}_minus_{b}"
            for metric_name, boot in (("observation", bo), ("context_equal", bc),
                                      ("pair_type", bp)):
                dd = boot["diffs"][k]
                contrast_rows.append({
                    "subset": label, "metric": metric_name, "contrast": k,
                    "delta": dd["point"], "ci_lo": dd["ci"][0], "ci_hi": dd["ci"][1],
                    "n_boot": 1000,
                })
        # pair-type level robustness
        ae_lcrc = np.abs(p["LCRC"] - d)
        ae_pq = np.abs(p["PAIR_Q"] - d)
        cnt = np.bincount(pair_idx)
        lcrc_pair = np.bincount(pair_idx, weights=ae_lcrc) / np.maximum(cnt, 1)
        pq_pair = np.bincount(pair_idx, weights=ae_pq) / np.maximum(cnt, 1)
        rec["pair_type_lcrc_better_fraction"] = float(np.mean(lcrc_pair < pq_pair))
        rec["pair_type_lcrc_minus_pq_median"] = float(np.median(lcrc_pair - pq_pair))

        # absolute prediction and response-scale diagnostics
        cells = np.unique(np.r_[pairs["pos_i"][m], pairs["pos_j"][m]])
        mols = ctx.cell_mol_first[cells].astype(np.int64)
        mols = np.unique(mols)
        y_mol = labels_all[valid_row_ids[mols]]
        abs_err = pred_mol[mols] - y_mol
        err_i = cell_pred[pairs["pos_i"][m]] - ctx.y[pairs["pos_i"][m]]
        err_j = cell_pred[pairs["pos_j"][m]] - ctx.y[pairs["pos_j"][m]]
        is_a = (rank_i[m] < rank_j[m])
        err_a = np.where(is_a, err_i, err_j)
        err_b = np.where(is_a, err_j, err_i)
        dt = pairs["delta"][m]
        dp = delta_pred[m]
        rec["absolute"] = {
            "n_unique_molecules": int(mols.size),
            "absolute_mae_eV": float(np.mean(np.abs(abs_err))),
            "absolute_rmse_eV": float(np.sqrt(np.mean(abs_err ** 2))),
            "delta_true_abs_mean": float(np.mean(np.abs(dt))),
            "delta_true_abs_median": float(np.median(np.abs(dt))),
            "delta_pred_abs_mean": float(np.mean(np.abs(dp))),
            "delta_pred_abs_median": float(np.median(np.abs(dp))),
            "delta_true_std": float(np.std(dt)),
            "delta_pred_std": float(np.std(dp)),
            "delta_attenuation_std_ratio": float(np.std(dp) / max(np.std(dt), 1e-12)),
            "delta_correlation": float(np.corrcoef(dt, dp)[0, 1]),
            "delta_regression_slope": float(np.polyfit(dt, dp, 1)[0]),
            "absolute_error_correlation_A_B": float(np.corrcoef(err_i, err_j)[0, 1]),
            "absolute_mae_A_eV": float(np.mean(np.abs(err_a))),
            "absolute_mae_B_eV": float(np.mean(np.abs(err_b))),
        }
        if label in ("all_evaluable", "unseen_exact_context", "clean_subset"):
            mol_i = ctx.cell_mol_first[pairs["pos_i"][m]].astype(np.int64)
            mol_j = ctx.cell_mol_first[pairs["pos_j"][m]].astype(np.int64)
            rec["dependency"] = _dependency_block(SD, d, p, mol_i, mol_j, seed=10)
        summary_subsets[label] = rec

    # ------------------------------------------------------------------
    # frozen conditional sign-reversal groups
    # ------------------------------------------------------------------
    frozen_reversal = sfec_summary["phase_e"]["sign_reversal"]
    reversal_summary: dict[str, Any] = {}
    group_rows_all: list[dict[str, Any]] = []
    for thr in THRESHOLDS:
        groups = _sign_reversal_groups(obs["prep"], thr)
        key = str(thr)
        ref = frozen_reversal[key]
        if len(groups) != ref["n_reversal_pairs"] or sum(len(g["groups"]) for g in groups) != ref["n_groups"]:
            raise RuntimeError(f"sign-reversal group reconstruction mismatch at {thr}")
        summ, rows = _eval_groups(groups, obs, thr)
        summ["frozen_counts_match"] = True
        summ["frozen_n_reversal_pairs"] = int(ref["n_reversal_pairs"])
        summ["frozen_n_groups"] = int(ref["n_groups"])
        reversal_summary[f"{thr:.2f}"] = summ
        group_rows_all.extend(rows)

    # non-zero stable conditional candidates (SFEC tau=0.10 primary, 0.20 sensitivity)
    nonzero_summary: dict[str, Any] = {}
    nonzero_rows_all: list[dict[str, Any]] = []
    for tau in (0.10, 0.20):
        summ, rows = _nonzero_candidates(obs, tau)
        nonzero_summary[f"{tau:.2f}"] = summ
        nonzero_rows_all.extend(rows)

    C.write_csv(C.ANALYSIS_DIR / "pcqm_valid_response.csv", response_rows)
    C.write_csv(C.ANALYSIS_DIR / "pcqm_valid_contrasts.csv", contrast_rows)
    C.write_csv(C.ANALYSIS_DIR / "pcqm_sign_reversal_groups.csv", group_rows_all)
    C.write_csv(C.ANALYSIS_DIR / "pcqm_nonzero_candidates.csv", nonzero_rows_all)

    model_info = {
        "name": inference["model_name"],
        "scale": inference["scale"],
        "parameters": inference["model_parameters"],
        "checkpoint": str(C.LCW_CHECKPOINT.relative_to(C.REPO_ROOT)),
        "checkpoint_sha256": inference["checkpoint_sha256"],
        "checkpoint_epoch": inference["checkpoint_epoch"],
        "checkpoint_valid_mae_eV": inference["checkpoint_valid_mae_eV"],
        "inference_valid_mae_eV": inference["mae_eV"],
        "inference_mae_abs_diff_eV": abs(inference["mae_eV"] - inference["checkpoint_valid_mae_eV"]),
        "selection_rule": (
            "pre-existing W3 model on the SFEC-identical 1M Train row plan; "
            "chosen before looking at any replacement-response score. No "
            "checkpoint was compared or selected using this round's response metric."
        ),
        "role": "non-MP whole-graph model; LCRC delta = pred(B) - pred(A)",
        "frozen_inference": True,
    }

    return {
        "status": "OK",
        "model": model_info,
        "identity": identity,
        "frozen_sfec_reproduction": verify,
        "inference": {
            "seconds": inference["inference_seconds"],
            "batch_seconds": inference["batch_seconds"],
            "n_rows": inference["n_rows"],
            "device": "cpu",
            "torch_threads": inference["torch_threads"],
        },
        "subsets": summary_subsets,
        "sign_reversal": reversal_summary,
        "nonzero_candidates": nonzero_summary,
        "artifacts": {
            "response_csv": "analysis/pcqm_valid_response.csv",
            "contrast_csv": "analysis/pcqm_valid_contrasts.csv",
            "sign_reversal_csv": "analysis/pcqm_sign_reversal_groups.csv",
            "nonzero_csv": "analysis/pcqm_nonzero_candidates.csv",
        },
        "resources": {
            "total_seconds": float(time.perf_counter() - t_start),
            "peak_rss_gb": C.peak_rss_gb(),
            "cpu_only": True,
            "target_seconds": 300,
            "hard_ceiling_seconds": 600,
        },
        "notes": {
            "no_training": "no parameter was updated; the W3 checkpoint is read-only",
            "no_pair_labels_in_inference": "each molecule is predicted independently",
            "no_sfec_refit": "ZERO/PAIR/PAIR_Q/LOCAL are frozen SFEC estimates",
            "official_valid_only": "Test labels were never read",
        },
    }
