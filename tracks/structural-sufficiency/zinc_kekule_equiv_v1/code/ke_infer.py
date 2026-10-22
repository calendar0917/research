"""Frozen-model inference over the verified Kekule representations.

Run remotely (no RDKit needed; the representation artifact is committed).  For
every valid molecule and every verified representation it rebuilds the P3
object features and the BF S3/S4/F4 cache from the modified raw graph, then
runs the frozen P3B seed-0 and T seed-1 checkpoints.  No checkpoint is
re-selected, no label is used to choose a representation, and the official test
split is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import ke_graph as KG  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"


def _permuted_graph(cats: np.ndarray, edges: Sequence[tuple[int, int]],
                    orders: np.ndarray, n_atoms: int, order: np.ndarray):
    """Return a graph whose atom order is ``order`` (new index k <- old order[k])."""
    inv = np.empty(n_atoms, dtype=np.int64)
    inv[np.asarray(order, dtype=np.int64)] = np.arange(n_atoms, dtype=np.int64)
    new_cats = np.asarray(cats)[np.asarray(order, dtype=np.int64)]
    table: dict[tuple[int, int], int] = {}
    for (u, v), o in zip(edges, orders):
        a, b = int(inv[int(u)]), int(inv[int(v)])
        if a > b:
            a, b = b, a
        table[(a, b)] = int(o)
    new_edges = sorted(table)
    new_orders = np.asarray([table[e] for e in new_edges], dtype=np.int64)
    ei = np.asarray([[u, v] for u, v in new_edges] + [[v, u] for u, v in new_edges],
                    dtype=np.int64).T
    ea = np.asarray([new_orders.tolist() + new_orders.tolist()], dtype=np.int64).reshape(-1)
    return new_cats, new_edges, new_orders, ei, ea


def _objects_for_graphs(graphs: Sequence[dict], *,
                        permute: bool = False, seed: int = 20261022) -> list:
    objs = []
    for k, g in enumerate(graphs):
        cats, edges, orders, ei, ea = g["cats"], g["edges"], g["orders"], g["ei"], g["ea"]
        if permute:
            rng = np.random.default_rng(int(seed) + int(g["row"]))
            order = rng.permutation(len(cats))
            cats, edges, orders, ei, ea = _permuted_graph(
                cats, edges, orders, len(cats), order)
        objs.append(KG.build_objects(cats, ei, ea, k))
    return objs


def _cache_for_rows(graphs: Sequence[dict], split: str, *,
                    permute: bool = False, seed: int = 20261022):
    objs = _objects_for_graphs(graphs, permute=permute, seed=seed)
    return KG.compact_cache(objs, None, split), objs


def _changed_summary(base: dict, new: dict, rows: Sequence[int]) -> dict:
    """Count feature cells / molecules changed for the molecules present in ``new``.

    ``new`` contains only ``rows`` (in that order), while ``base`` is the full
    original-representation cache.  Per-molecule pointer slices are compared so
    subsets of molecules are handled correctly.
    """
    out = {}
    for key, ptr_key in (("atom_feat", "atom_ptr"), ("bond_feat", "bond_ptr"),
                         ("junc_feat", "junc_ptr")):
        a = np.asarray(base[key])
        b = np.asarray(new[key])
        bp = np.asarray(base[ptr_key])
        np_ = np.asarray(new[ptr_key])
        cells = rows_changed = mols = 0
        for k, i in enumerate(rows):
            x0, x1 = int(bp[i]), int(bp[i + 1])
            y0, y1 = int(np_[k]), int(np_[k + 1])
            if (x1 - x0) != (y1 - y0):
                mols += 1
                cells += max(x1 - x0, y1 - y0)
                rows_changed += max(x1 - x0, y1 - y0)
                continue
            diff = np.any(a[x0:x1] != b[y0:y1], axis=1)
            if diff.any():
                mols += 1
                rows_changed += int(diff.sum())
                cells += int(np.count_nonzero(a[x0:x1] != b[y0:y1]))
        out[key] = {"changed_cells": int(cells), "changed_rows": int(rows_changed),
                    "changed_molecules": int(mols)}
    a = np.asarray(base["global_feat"])[np.asarray(rows, dtype=np.int64)]
    b = np.asarray(new["global_feat"])
    out["global_feat"] = {"changed_cells": int(np.count_nonzero(a != b)),
                          "changed_molecules": int(np.any(a != b, axis=1).sum())}
    return out


def _run_all(records: Sequence[dict], models, device, batch_size: int,
             max_reps: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    n = len(records)
    p3 = np.full((n, max_reps), np.nan, dtype=np.float32)
    t1 = np.full((n, max_reps), np.nan, dtype=np.float32)
    changed = [None] * max_reps
    base_cache = None
    for r in range(max_reps):
        rows = [i for i, rec in enumerate(records) if rec["n_reps"] > r]
        if not rows:
            continue
        t0 = time.time()
        graphs = [records[i]["graphs"][r] for i in rows]
        split = f"valid_rep_{r:02d}"
        cache, objs = _cache_for_rows(graphs, split)
        if r == 0:
            base_cache = cache
            changed[r] = {"note": "original representation; reference cache"}
        else:
            changed[r] = _changed_summary(base_cache, cache, rows)
        p = KG.predict_p3(models["p3"], cache, np.arange(len(rows)), device, batch_size)
        p3[np.asarray(rows), r] = p
        bf = KG.build_bf_cache(objs, split)
        q = KG.predict_t1(models["t1"], cache, bf, np.arange(len(rows)), device,
                          split, batch_size)
        t1[np.asarray(rows), r] = q
        changed[r]["seconds"] = float(time.time() - t0)
        print(f"[rep {r}] rows={len(rows)} p3={float(np.mean(p)):.6f} "
              f"t1={float(np.mean(q)):.6f} {time.time() - t0:.1f}s", flush=True)
    return p3, t1, changed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reps", default=str(ANALYSIS / "valid_reps.npz"))
    p.add_argument("--out", default=str(ANALYSIS / "preds_valid.npz"))
    p.add_argument("--checks", default=str(ANALYSIS / "infer_checks.json"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--perm-sample", type=int, default=24)
    args = p.parse_args(argv)

    import torch

    device = torch.device(args.device)
    art = KG.load_reps_artifact(Path(args.reps))
    meta = json.loads(str(art["meta_json"][0]))
    ds = KG.load_pyg("valid")
    max_reps = int(art["rep_orders"].shape[1])
    records = []
    for row in range(len(ds)):
        cats, edges, orig, ei, ea = KG.graph_arrays(ds[row])
        if KG.graph_sha256(cats, ei, ea) != str(art["graph_sha256"][row]):
            raise RuntimeError(f"row {row}: artifact graph hash mismatch")
        ne = int(art["n_edges"][row])
        nr = int(art["n_reps"][row])
        if ne != len(edges):
            raise RuntimeError(f"row {row}: edge count {ne} != {len(edges)}")
        for k, e in enumerate(edges):
            if (int(art["edge_u"][row, k]), int(art["edge_v"][row, k])) != e:
                raise RuntimeError(f"row {row}: edge {k} mismatch")
        if not np.array_equal(np.asarray(art["orig_order"][row, :ne], dtype=np.int64),
                              orig):
            raise RuntimeError(f"row {row}: original order mismatch")
        graphs = []
        for r in range(nr):
            orders = np.asarray(art["rep_orders"][row, r, :ne], dtype=np.int64)
            if (int(art["status"][row]) == 3) and r == 0:
                orders = orig
            ea_rep = KG.apply_orders(ei, ea, edges, orders)
            graphs.append({"row": row, "cats": cats, "edges": edges,
                           "orders": orders, "ei": ei, "ea": ea_rep})
        records.append({"row": row, "cats": cats, "edges": edges,
                        "orig": orig, "ei": ei, "ea": ea,
                        "n_reps": nr, "graphs": graphs,
                        "status": int(art["status"][row])})

    t0 = time.time()
    models = KG.load_models(device)
    p3, t1, feature_changes = _run_all(records, models, device,
                                      args.batch_size, max_reps)
    elapsed = time.time() - t0

    # exact replay check against the frozen stored valid predictions
    p3_stored = np.load(KG.SS / "zinc_recipe_trunk_v2/runs/xBssl_P3N_s0/predictions/valid_best.npz")
    t1_stored = np.load(KG.SS / "zinc_shared_factor_v1/runs/SF_T_s1/predictions/valid_best.npz")
    y = np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(len(ds))])
    check = {}
    for name, pred, stored in (("p3", p3[:, 0], p3_stored["pred"]),
                                ("t1", t1[:, 0], t1_stored["pred"])):
        d = np.asarray(pred, dtype=np.float64) - np.asarray(stored, dtype=np.float64)
        check[name] = {
            "max_abs_diff": float(np.abs(d).max()),
            "mean_abs_diff": float(np.abs(d).mean()),
            "mae_vs_y": float(np.abs(np.asarray(pred, dtype=np.float64) - y).mean()),
            "stored_mae_vs_y": float(np.abs(np.asarray(stored, dtype=np.float64) - y).mean()),
            "tolerance": 1e-4,
            "pass": bool(np.abs(d).max() < 1e-4),
        }

    # permutation check on rows with an alternative representation
    perm_rows = [rec["row"] for rec in records if rec["n_reps"] >= 2][:args.perm_sample]
    perm_check = {"rows": [int(x) for x in perm_rows], "rep_index": 1,
                  "n": len(perm_rows), "tolerance": 1e-3}
    if perm_rows:
        gs = []
        for row in perm_rows:
            rec = records[row]
            gs.append(rec["graphs"][1])
        cu, objs_u = _cache_for_rows(gs, "perm_unperm")
        cp, objs_p = _cache_for_rows(gs, "perm_perm", permute=True)
        pu = KG.predict_p3(models["p3"], cu, np.arange(len(gs)), device, len(gs))
        pp = KG.predict_p3(models["p3"], cp, np.arange(len(gs)), device, len(gs))
        bfu = KG.build_bf_cache(objs_u, "perm_unperm")
        tu = KG.predict_t1(models["t1"], cu, bfu, np.arange(len(gs)), device,
                           "perm_unperm", len(gs))
        bfp = KG.build_bf_cache(objs_p, "perm_perm")
        tp = KG.predict_t1(models["t1"], cp, bfp, np.arange(len(gs)), device,
                           "perm_perm", len(gs))
        perm_check.update({
            "p3_max_abs_diff": float(np.abs(pu - pp).max()),
            "t1_max_abs_diff": float(np.abs(tu - tp).max()),
            "p3_pass": bool(np.abs(pu - pp).max() < 1e-3),
            "t1_pass": bool(np.abs(tu - tp).max() < 1e-3),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        row=np.arange(len(records), dtype=np.int64),
        n_reps=np.asarray([r["n_reps"] for r in records], dtype=np.int8),
        status=np.asarray([r["status"] for r in records], dtype=np.int8),
        n_atoms=np.asarray([len(r["cats"]) for r in records], dtype=np.int16),
        n_aromatic_edges=np.asarray(
            [int(np.count_nonzero(art["aromatic_mask"][r["row"]])) for r in records],
            dtype=np.int16),
        p3b=p3, t1=t1,
        rep_is_original=np.asarray(
            [[(j == 0) if j < rec["n_reps"] else False for j in range(max_reps)]
             for rec in records], dtype=bool),
        rep_changed_edges=np.asarray(
            [[0 if (rec["status"] == 3 and j == 0) else int(np.count_nonzero(
                np.asarray(art["rep_orders"][rec["row"], j, :art["n_edges"][rec["row"]]])
                != np.asarray(art["orig_order"][rec["row"], :art["n_edges"][rec["row"]]])))
              if j < rec["n_reps"] else -1 for j in range(max_reps)]
             for rec in records], dtype=np.int16),
        meta_json=np.asarray([json.dumps({
            "protocol": "zinc-kekule-equiv-v1",
            "official_test_loaded": False,
            "device": str(device),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "batch_size": int(args.batch_size),
            "max_reps": int(max_reps),
            "rdkit_version_artifact": meta.get("rdkit_version"),
            "elapsed_seconds": float(elapsed),
        })], dtype="U"),
    )

    checks = {
        "protocol": "zinc-kekule-equiv-v1",
        "official_test_loaded": False,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "batch_size": int(args.batch_size),
        "elapsed_seconds": float(elapsed),
        "checkpoints": {
            "p3": {"path": str(models["p3_ckpt"].relative_to(KG.REPO)),
                   "epoch": models["p3_epoch"],
                   "parameters": models["p3_parameters"],
                   "recorded_valid_mae": models["p3_valid_recorded"]},
            "t1": {"path": str(models["t1_ckpt"].relative_to(KG.REPO)),
                   "epoch": models["t1_epoch"],
                   "parameters": models["t1_parameters"],
                   "recorded_valid_mae": models["t1_valid_recorded"]},
        },
        "original_prediction_replay": check,
        "feature_rebuild": feature_changes,
        "permutation_check": perm_check,
        "representation_artifact": {
            "path": str(Path(args.reps).relative_to(KG.REPO)) if Path(args.reps).is_relative_to(KG.REPO) else str(args.reps),
            "status_counts": meta.get("status_counts"),
            "n_truncated": meta.get("n_truncated"),
        },
    }
    Path(args.checks).write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
