"""Phase D: frozen 1M models transferred to the official PCQM Valid set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import sfec1_common as C
import sfec1_data as D

MODELS = ["ZERO", "PAIR", "PAIR_Q", "SHARED", "LOCAL", "LOCAL_NULL"]
COMPARISONS = [
    ("PAIR", "ZERO"), ("SHARED", "ZERO"), ("PAIR_Q", "PAIR"),
    ("LOCAL", "SHARED"), ("LOCAL", "LOCAL_NULL"), ("LOCAL", "PAIR_Q"),
    ("LOCAL", "ZERO"),
]


def _prep_stats(prep: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: prep[f"disc_{k}"] for k in
            ("pair_key", "A", "B", "n", "median", "mean", "mad", "p80_abs_dev", "sd", "sign_pos")}


def _triple_from_prep(prep: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {
        "key": prep[f"triple_{prefix}_key"], "n": prep[f"triple_{prefix}_n"],
        "median": prep[f"triple_{prefix}_median"], "mean": prep[f"triple_{prefix}_mean"],
        "p80_abs_dev": prep[f"triple_{prefix}_p80"],
    }


def _extend_fit(fit: dict[str, Any], n_q: int) -> dict[str, Any]:
    return {
        "d": np.vstack([fit["d"], np.zeros((1, 64))]),
        "comp": np.vstack([fit["comp"], -np.ones((1, 64), dtype=fit["comp"].dtype)]),
        "supported": np.r_[fit["supported"], False],
    }


def _subset_rows(
    label: str, mask: np.ndarray, delta: np.ndarray, preds: dict[str, np.ndarray],
    ctx: np.ndarray, pair_idx: np.ndarray, coverage: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_ctx, n_ctx = D.compact_ctx(ctx[mask])
    pid = pair_idx[mask]
    u, pid = np.unique(pid, return_inverse=True)
    d = delta[mask]
    p = {k: v[mask] for k, v in preds.items()}
    points = {}
    for name, pred in p.items():
        base = D.basic_metric(d, pred)
        ae = np.abs(pred - d)
        sq = (pred - d) ** 2
        cnt = np.bincount(row_ctx, minlength=n_ctx).astype(np.float64)
        cae = np.bincount(row_ctx, weights=ae, minlength=n_ctx)
        csq = np.bincount(row_ctx, weights=sq, minlength=n_ctx)
        ok = cnt > 0
        ps = np.bincount(pid, weights=ae)
        pc = np.bincount(pid)
        pok = pc > 0
        points[name] = {
            "observation": {"mae": base["mae"], "rmse": base["rmse"],
                            "sign_accuracy": base["sign_accuracy"]},
            "context": {"mae": float(np.mean(cae[ok] / cnt[ok])),
                        "rmse": float(np.sqrt(np.mean(csq[ok] / cnt[ok]))),
                        "sign_accuracy": base["sign_accuracy"]},
            "pair_type": {"mae": float(np.mean(ps[pok] / pc[pok])),
                          "rmse": float(np.sqrt(np.mean(
                              np.bincount(pid, weights=sq)[pok] / pc[pok]))),
                          "sign_accuracy": base["sign_accuracy"]},
        }
    reps = int(cfg["statistics"]["bootstrap_replicates"])
    seed = int(cfg["statistics"]["bootstrap_seed"])
    bo = D.bootstrap_obs_context(
        d, p, row_ctx, n_ctx, context_equal=False, reps=reps, seed=seed + 301,
        comparisons=COMPARISONS,
    )
    bc = D.bootstrap_obs_context(
        d, p, row_ctx, n_ctx, context_equal=True, reps=reps, seed=seed + 302,
        comparisons=COMPARISONS,
    )
    bp = D.bootstrap_pair_macro(
        d, p, row_ctx, pid, n_ctx, reps=reps, seed=seed + 303,
        comparisons=COMPARISONS,
    )
    result = {
        "subset": label, "n_obs": int(mask.sum()), "n_contexts": int(n_ctx),
        "n_pairs": int(len(u)), "points": points,
        "coverage": {k: float(np.mean(v[mask])) for k, v in coverage.items()},
        "bootstrap": {"observation": bo, "context": bc, "pair_type": bp},
    }
    rows = []
    for name in MODELS:
        for weighting, key, boot in (
            ("observation", "observation", bo), ("context_equal", "context", bc),
            ("pair_type_equal", "pair_type", bp),
        ):
            b = boot["models"].get(name) if boot else None
            rows.append({
                "subset": label, "n_obs": int(mask.sum()), "n_contexts": int(n_ctx),
                "n_pairs": int(len(u)), "row_kind": "model", "weighting": weighting,
                "model": name, "comparison": "",
                "mae": points[name][key]["mae"], "rmse": points[name][key]["rmse"],
                "sign_accuracy": points[name][key]["sign_accuracy"],
                "coverage": (float(np.mean(coverage[name][mask])) if name in coverage else ""),
                "ci_lo": b["ci"][0] if b else "", "ci_hi": b["ci"][1] if b else "",
                "relative_gain_vs_first": "",
            })
    for a, bb in COMPARISONS:
        for weighting, key, boot in (
            ("observation", "observation", bo), ("context_equal", "context", bc),
            ("pair_type_equal", "pair_type", bp),
        ):
            k = f"{a}_minus_{bb}"
            dd = boot["diffs"][k]
            ma = points[a][key]["mae"]
            mb = points[bb][key]["mae"]
            rows.append({
                "subset": label, "n_obs": int(mask.sum()), "n_contexts": int(n_ctx),
                "n_pairs": int(len(u)), "row_kind": "paired_diff", "weighting": weighting,
                "model": "", "comparison": k, "mae": dd["point"], "rmse": "",
                "sign_accuracy": "", "coverage": "",
                "ci_lo": dd["ci"][0], "ci_hi": dd["ci"][1],
                "relative_gain_vs_first": 1.0 - mb / max(ma, 1e-12),
            })
    return result, rows


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    prep = {k: v for k, v in np.load(C.DATA_DIR / "prepared_pcqm.npz").items()}
    train_prefix = (C.PARENT_DIR / Path(cfg["datasets"]["pcqm"]["train_prefix"])).resolve()
    valid_prefix = (C.PARENT_DIR / Path(cfg["datasets"]["pcqm"]["valid_prefix"])).resolve()
    train_uni = D.load_universe(train_prefix)
    train_S = D.load_S(train_prefix)
    train_C = D.load_C(train_prefix, columns=["hash64"])
    train_M = D.load_M(train_prefix, columns=["mol_key"])
    valid_cells = D.load_cells(valid_prefix)
    valid_S = D.load_S(valid_prefix)
    valid_C = D.load_C(valid_prefix, columns=["q_id", "hash64", "C_key"])
    valid_M = D.load_M(valid_prefix, columns=["mol_id", "mol_key"])
    labels = C.load_pcqm_labels_cached()
    primary = train_uni["primary"]
    rank = train_uni["rank"]
    n_s = len(train_S["S_id"])
    # S mapping by exact key digest
    s_digest = {D.digest(str(k)): int(i) for i, k in enumerate(train_S["S_key"].tolist())}
    vmap = np.asarray([s_digest.get(D.digest(str(k)), -1) for k in valid_S["S_key"].tolist()], dtype=np.int64)
    # duplicate complete molecule graph versus the 1M Train prefix
    train_mol = set(D.digest(str(k)) for k in train_M["mol_key"].tolist())
    valid_mol_dup = np.asarray([D.digest(str(k)) in train_mol for k in valid_M["mol_key"].tolist()])
    cells_v = {k: v.copy() for k, v in valid_cells.items()}
    mapped = vmap[cells_v["S_id"].astype(np.int64)]
    cells_v["S_id"] = mapped
    seen_s = mapped >= 0
    dup_mol = valid_mol_dup[cells_v["mol_id"].astype(np.int64)]
    mask = seen_s & primary[np.maximum(mapped, 0)] & ~dup_mol
    ctx_v = D.build_contexts(cells_v, mask, labels, n_s, primary, rank)
    pairs_v = D.build_pairs(ctx_v, rank, canonical=True)
    # q mapping into the frozen train q vocabulary
    train_q = C.read_json(Path(str(train_prefix) + ".keys.json"))["q_keys"]
    valid_q = C.read_json(Path(str(valid_prefix) + ".keys.json"))["q_keys"]
    q_map = {D.digest(str(k)): int(i) for i, k in enumerate(train_q)}
    vq = np.asarray([q_map.get(D.digest(str(k)), -1) for k in valid_q], dtype=np.int64)
    q_ctx = vq[valid_C["q_id"][ctx_v.C].astype(np.int64)]
    n_q = int(prep["n_q"][0])
    q_safe_ctx = np.where(q_ctx >= 0, q_ctx, n_q)
    # frozen discovery models
    ds = _prep_stats(prep)
    tri = _triple_from_prep(prep, "true")
    bvec = prep["shared_b"]
    comp = prep["shared_comp"]
    local_fit = _extend_fit({
        "d": prep["local_d"], "comp": prep["local_comp"],
        "supported": prep["local_supported"],
    }, n_q)
    null_fit = _extend_fit({
        "d": prep["local_null_d"], "comp": prep["local_null_comp"],
        "supported": prep["local_null_supported"],
    }, n_q)
    A = pairs_v["A"].astype(np.int64)
    B = pairs_v["B"].astype(np.int64)
    key = D.pair_key(A, B)
    mu = D.lookup_pair(ds, key, "median")
    n_disc = D.lookup_pair_count(ds, key)
    main = (n_disc >= int(cfg["support"]["p1_min_discovery_contexts"])) & \
           (comp[A] >= 0) & (comp[A] == comp[B])
    q_row = q_safe_ctx[pairs_v["ctx"]]
    tmed, tcnt = D.lookup_triple(tri, A, B, q_row, n_q, "median")
    pair_q = np.where(tcnt >= int(cfg["support"]["conditional_min_discovery_contexts"]), tmed, mu)
    pred_shared = bvec[B] - bvec[A]
    pred_local = D.predict_local(pairs_v["ctx"], A, B, q_safe_ctx, bvec, local_fit)
    pred_null = D.predict_local(pairs_v["ctx"], A, B, q_safe_ctx, bvec, null_fit)
    preds = {
        "ZERO": np.zeros(len(key)), "PAIR": mu, "PAIR_Q": pair_q,
        "SHARED": pred_shared, "LOCAL": pred_local["pred"],
        "LOCAL_NULL": pred_null["pred"],
    }
    coverage = {
        "PAIR_Q": tcnt >= int(cfg["support"]["conditional_min_discovery_contexts"]),
        "LOCAL": pred_local["enabled"],
        "LOCAL_NULL": pred_null["enabled"],
    }
    # context seen / unseen relative to 1M prefix
    train_hash = set(int(x) for x in train_C["hash64"].tolist())
    seen_c_ctx = np.asarray([int(h) in train_hash for h in valid_C["hash64"][ctx_v.C].tolist()])
    seen_c_row = seen_c_ctx[pairs_v["ctx"]]
    # clean subset
    clean_flags = D.molecule_clean_flags(valid_cells, valid_M)
    pair_clean = D.pair_clean_mask(ctx_v, pairs_v, clean_flags["clean"])
    q_supported_row = (q_row < n_q) & local_fit["supported"][np.minimum(q_row, n_q - 1)]
    subset_masks = {
        "all_evaluable": main,
        "unseen_exact_context": main & ~seen_c_row,
        "unseen_context_q_supported": main & ~seen_c_row & q_supported_row,
        "seen_exact_context": main & seen_c_row,
        "clean_subset": main & pair_clean,
    }
    results = {}
    all_rows: list[dict[str, Any]] = []
    pair_idx = np.unique(key, return_inverse=True)[1]
    for label, m in subset_masks.items():
        if not np.any(m):
            results[label] = {"subset": label, "n_obs": 0}
            continue
        res, rows = _subset_rows(label, m, pairs_v["delta"], preds, pairs_v["ctx"], pair_idx,
                                 coverage, cfg)
        results[label] = res
        all_rows.extend(rows)
    payload = {
        "dataset": "pcqm_valid",
        "definition_unseen": "unseen relative to the current 1M Train prefix (C hash64), not relative to the full 3.38M official Train set; the remaining official Train rows were not scanned",
        "n_valid_cells": int(len(valid_cells["C_id"])),
        "n_valid_molecules": int(len(valid_M["mol_id"])),
        "n_valid_C": int(len(valid_C["C_key"])),
        "n_valid_S": int(len(valid_S["S_id"])),
        "n_valid_q": int(len(valid_q)),
        "n_q_unmapped_to_train": int(np.sum(vq < 0)),
        "n_cells_unmapped_S": int((~seen_s).sum()),
        "n_cells_duplicate_train_molecule": int(dup_mol.sum()),
        "n_cells_used": int(mask.sum()),
        "n_contexts_used": int(ctx_v.n_ctx),
        "n_pair_observations": int(len(key)),
        "n_pair_observations_evaluable": int(main.sum()),
        "n_pair_types_evaluable": int(len(np.unique(key[main]))),
        "subsets": results,
        "notes": "Official PCQM Test labels were never read. No Valid label was used for fitting or selection.",
    }
    C.write_csv(C.ANALYSIS_DIR / "valid_transfer.csv", all_rows)
    C.write_json(C.ANALYSIS_DIR / "valid_transfer.json", payload)
    return payload


def main() -> int:
    cfg = C.load_config()
    payload = run(cfg)
    print(json.dumps({k: payload[k] for k in (
        "n_pair_observations", "n_pair_observations_evaluable",
        "n_pair_types_evaluable", "n_q_unmapped_to_train")}, indent=2))
    for label, res in payload.get("subsets", {}).items():
        if res.get("n_obs"):
            pts = res["points"]
            print(label, res["n_obs"], res["n_contexts"], res["n_pairs"],
                  "PAIR", pts["PAIR"]["observation"]["mae"],
                  "PAIR_Q", pts["PAIR_Q"]["observation"]["mae"],
                  "LOCAL", pts["LOCAL"]["observation"]["mae"],
                  "LOCAL_NULL", pts["LOCAL_NULL"]["observation"]["mae"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
