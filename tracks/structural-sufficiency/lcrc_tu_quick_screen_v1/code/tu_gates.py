"""Minimal correctness gates for the TU adapter, split, model variants and GIN.

Only checks that are *specific to this adaptation* run here:

1. the three split indices are disjoint, complete and reproduce the frozen
   hashes; the standardisation means equal the Train means;
2. hand-computed S1/S2 codes, per-node scalars and global fields on a chain, a
   triangle, an isolated node and a disconnected two-component graph;
3. node renumbering invariance (features, global statistics, model logits);
4. batch split/merge consistency and 2-class / 3-class output shapes;
5. FULL/C1 shared initialisation, the exact deleted block (96 input columns of
   the local MLP) and the equivalence "FULL with the b_mean columns zeroed ==
   C1";
6. every actually used branch receives a finite non-zero gradient.

Nothing here reads a Test label or a Test structural cache.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import tu_common as C  # noqa: E402  (inserts the frozen prep code path)

import lcrc_data as PREP_DATA  # noqa: E402
import tu_adapter as A  # noqa: E402
import tu_data as D  # noqa: E402
import tu_model as M  # noqa: E402

TOL = 1e-6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _record(pairs: Sequence[Sequence[int]], labels: Sequence[int]) -> dict[str, Any]:
    n = len(labels)
    p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    degree = np.zeros(n, dtype=np.int64)
    for a, b in p.tolist():
        degree[a] += 1
        degree[b] += 1
    return {
        "gid": 0,
        "y": 0,
        "n_atoms": n,
        "pairs": p,
        "node_labels": np.asarray(labels, dtype=np.int64),
        "n_physical_edges": int(p.shape[0]),
        "clean": {},
    }


def _toy_cache(
    records: Sequence[Mapping[str, Any]],
    spec,
    node_types: Mapping[str, Any],
    *,
    split: str = "toy",
) -> dict[str, Any]:
    mols = []
    for i, rec in enumerate(records):
        graph, analysis = A.build_tu_graph(rec, spec, node_types, molecule_id=f"toy#{i}")
        mols.append(D.build_tu_molecule(graph, analysis, spec))
    cache = PREP_DATA.concat_molecules(mols, spec, split=split)
    cache["gids"] = np.arange(len(records), dtype=np.int64)
    return cache


def _toy_norm(cache: Mapping[str, Any], spec) -> dict[str, Any]:
    norm = PREP_DATA.fit_norm(cache, spec, source="toy_train_only")
    return {
        k: np.asarray(norm[k], dtype=np.float32)
        for k in ("scalar_mean", "scalar_scale", "global_mean", "global_scale")
    }


def _batch(cache: Mapping[str, Any], rows: Sequence[int], device="cpu") -> dict[str, Any]:
    b = PREP_DATA.collate(cache, rows)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def _all_zero_except_source(logits: np.ndarray, y: np.ndarray) -> float:
    return float(F.cross_entropy(torch.as_tensor(logits), torch.as_tensor(y)).item())


def _grad_report(model: torch.nn.Module, batch: Mapping[str, Any], n_classes: int) -> dict[str, Any]:
    model.train()
    logits = model(batch)
    y = torch.zeros(logits.shape[0], dtype=torch.int64)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    out: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            out[name] = {"grad_norm": 0.0, "present": False}
        else:
            gn = float(param.grad.detach().norm().item())
            out[name] = {"grad_norm": gn, "present": bool(math.isfinite(gn) and gn > 0.0)}
    return out


def _require(cond: bool, msg: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": bool(cond), "message": msg, "details": details or {}}


# ---------------------------------------------------------------------------
# gate 1: split / normalisation
# ---------------------------------------------------------------------------


def gate_split_and_norm(datasets: Sequence[str]) -> dict[str, Any]:
    splits = C.read_json(C.SPLIT_FILE)
    out: dict[str, Any] = {}
    for ds in datasets:
        meta = C.DATASETS[ds]
        rec = splits[ds]
        rows = {s: np.asarray(rec[s]["first_rows"] + rec[s]["last_rows"], dtype=np.int64) for s in ("train", "valid", "test")}
        prep = D.load_prepared(ds)
        n = sum(int(prep["split"][s]["n"]) for s in ("train", "valid", "test"))
        union = np.concatenate([np.asarray(prep["caches"][s]["gids"]) for s in ("train", "valid", "test")])
        disjoint = (
            np.intersect1d(prep["caches"]["train"]["gids"], prep["caches"]["valid"]["gids"]).size == 0
            and np.intersect1d(prep["caches"]["train"]["gids"], prep["caches"]["test"]["gids"]).size == 0
            and np.intersect1d(prep["caches"]["valid"]["gids"], prep["caches"]["test"]["gids"]).size == 0
        )
        complete = bool(np.unique(union).size == meta["n_graphs"] and n == meta["n_graphs"])
        sizes_ok = all(
            int(prep["split"][s]["n"]) == int(meta[f"n_{s}"]) for s in ("train", "valid", "test")
        )
        norm = prep["norm_record"]
        train_mean = np.asarray(prep["caches"]["train"]["global"], dtype=np.float64).mean(0).astype(np.float32)
        norm_ok = bool(
            str(norm["fit_split"]) == "train"
            and not bool(norm["valid_or_test_used"])
            and np.allclose(train_mean, np.asarray(norm["global_mean"], dtype=np.float32), atol=0.0, rtol=0.0)
        )
        out[ds] = _require(
            bool(disjoint and complete and sizes_ok and norm_ok),
            "split is a disjoint complete partition and the norm is fit on Train only",
            {
                "n": int(n),
                "disjoint": bool(disjoint),
                "complete": bool(complete),
                "sizes": {s: int(prep["split"][s]["n"]) for s in ("train", "valid", "test")},
                "class_counts": {s: prep["split"][s]["class_counts"] for s in ("train", "valid", "test")},
                "train_index_sha256": prep["split"]["train"]["index_sha256"],
                "valid_index_sha256": prep["split"]["valid"]["index_sha256"],
                "test_index_sha256": prep["split"]["test"]["index_sha256"],
                "partition_sha256": prep["split"]["partition_sha256"],
                "norm_fit_split": norm["fit_split"],
                "norm_valid_or_test_used": bool(norm["valid_or_test_used"]),
                "norm_global_mean_matches_train": bool(
                    np.allclose(train_mean, np.asarray(norm["global_mean"], dtype=np.float32), atol=0.0, rtol=0.0)
                ),
                "split_rows_probe": {s: rows[s].tolist() for s in rows},
            },
        )
    return out


# ---------------------------------------------------------------------------
# gate 2: hand-computed graph structures
# ---------------------------------------------------------------------------


def gate_hand_computed() -> dict[str, Any]:
    node_types = A.node_type_table("label", [0, 1])
    spec = A.build_tu_spec("NCI1", node_types)
    out: dict[str, Any] = {}

    # ---- chain 0-1-2-3 with node labels [0,1,0,1] ------------------------
    chain = _record([[0, 1], [1, 2], [2, 3]], [0, 1, 0, 1])
    cache = _toy_cache([chain], spec, node_types)
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    nbr_code = np.asarray(cache["nbr_code"])
    path_ptr = np.asarray(cache["path_ptr"])
    expected_nbr_ptr = np.asarray([0, 1, 3, 5, 6])
    expected_nbr_code = np.asarray([2, 0, 0, 2, 2, 0])
    expected_path_ptr = np.asarray([0, 1, 2, 3, 4])
    expected_path_cat = np.asarray([4, 0, 4, 0])
    expected_end63 = np.asarray([0, 2, 0, 2])
    expected_path_end = np.asarray([2, 3, 0, 1])
    expected_closure = np.asarray([0, 0, 0, 0])
    scal = np.asarray(cache["atom_scalar"])
    expected_nd2 = np.log1p(np.asarray([1.0, 1.0, 1.0, 1.0])).astype(np.float32)
    expected_paths = np.log1p(np.asarray([1.0, 1.0, 1.0, 1.0])).astype(np.float32)
    expected_logdeg = np.log1p(np.asarray([1.0, 2.0, 2.0, 1.0])).astype(np.float32)
    g = np.asarray(cache["global"])[0]
    gmap = {name: float(g[i]) for i, (name, _d) in enumerate(C.GLOBAL_FIELDS)}
    expected_chain_global = {
        "log1p_n_atoms": math.log1p(4),
        "log1p_n_bonds": math.log1p(3),
        "log1p_n_junctions": math.log1p(2),
        "cyclomatic_number": 0.0,
        "n_components": 1.0,
        "diameter_max_finite": 3.0,
        "mean_finite_pair_distance": 10.0 / 6.0,
        "frac_unreachable_pairs": 0.0,
        "frac_finite_pairs_ge3": 1.0 / 6.0,
        "bonds_per_atom": 3.0 / 4.0,
        "max_degree": 2.0,
        "terminal_atom_fraction": 0.5,
        "junction_per_bond": 2.0 / 3.0,
        "frac_deg2_junctions": 1.0,
        "frac_deg3plus_junctions": 0.0,
        "distinct_atom_types": 2.0,
    }
    chain_ok = bool(
        np.array_equal(nbr_ptr, expected_nbr_ptr)
        and np.array_equal(nbr_code, expected_nbr_code)
        and np.array_equal(path_ptr, expected_path_ptr)
        and np.array_equal(np.asarray(cache["path_cat"]), expected_path_cat)
        and np.array_equal(np.asarray(cache["path_end63"]), expected_end63)
        and np.array_equal(np.asarray(cache["path_end"]), expected_path_end)
        and np.array_equal(np.asarray(cache["path_closure"]), expected_closure)
        and np.allclose(scal[:, 0], expected_nd2, atol=TOL)
        and np.allclose(scal[:, 1], expected_paths, atol=TOL)
        and np.allclose(scal[:, 2], 0.0, atol=TOL)
        and np.allclose(scal[:, 3], expected_logdeg, atol=TOL)
        and all(abs(gmap[k] - v) <= 1e-6 for k, v in expected_chain_global.items())
    )
    out["chain"] = _require(
        chain_ok,
        "chain S1/S2/scalars/global equal the hand-computed values",
        {
            "nbr_ptr": nbr_ptr.tolist(),
            "nbr_code": nbr_code.tolist(),
            "path_ptr": path_ptr.tolist(),
            "path_cat": np.asarray(cache["path_cat"]).tolist(),
            "path_end63": np.asarray(cache["path_end63"]).tolist(),
            "path_end": np.asarray(cache["path_end"]).tolist(),
            "path_closure": np.asarray(cache["path_closure"]).tolist(),
            "scalars": scal.tolist(),
            "global": g.tolist(),
        },
    )

    # ---- triangle 0-1-2 (all type 0) -------------------------------------
    tri = _record([[0, 1], [1, 2], [2, 0]], [0, 0, 0])
    cache_t = _toy_cache([tri], spec, node_types)
    scal_t = np.asarray(cache_t["atom_scalar"])
    gt = np.asarray(cache_t["global"])[0]
    gmap_t = {name: float(gt[i]) for i, (name, _d) in enumerate(C.GLOBAL_FIELDS)}
    tri_ok = bool(
        np.array_equal(np.asarray(cache_t["nbr_code"]), np.zeros(6, dtype=np.int32))
        and np.array_equal(np.asarray(cache_t["path_ptr"]), np.asarray([0, 2, 4, 6]))
        and np.allclose(np.asarray(cache_t["path_closure"]), 1.0)
        and np.array_equal(np.asarray(cache_t["path_cat"]), np.zeros(6, dtype=np.int32))
        and np.array_equal(np.asarray(cache_t["path_end63"]), np.zeros(6, dtype=np.int32))
        and np.array_equal(np.asarray(cache_t["path_end"]), np.asarray([2, 1, 2, 0, 1, 0]))
        and np.allclose(scal_t[:, 2], 1.0, atol=TOL)
        and np.allclose(scal_t[:, 4], 1.0, atol=TOL)
        and np.allclose(scal_t[:, 5], math.log1p(3.0), atol=TOL)
        and np.allclose(scal_t[:, 6], math.log1p(2.0), atol=TOL)
        and abs(gmap_t["ring_atom_fraction"] - 1.0) <= TOL
        and abs(gmap_t["ring_bond_fraction"] - 1.0) <= TOL
        and abs(gmap_t["n_ring_systems"] - 1.0) <= TOL
        and abs(gmap_t["ring_size3_fraction"] - 1.0) <= TOL
        and abs(gmap_t["cyclomatic_number"] - 1.0) <= TOL
        and abs(gmap_t["distinct_atom_types"] - 1.0) <= TOL
    )
    out["triangle"] = _require(
        tri_ok,
        "triangle closure=1 for every rooted 2-path and the ring statistics match",
        {
            "path_ptr": np.asarray(cache_t["path_ptr"]).tolist(),
            "path_end": np.asarray(cache_t["path_end"]).tolist(),
            "path_closure": np.asarray(cache_t["path_closure"]).tolist(),
            "scalars": scal_t.tolist(),
            "global": gt.tolist(),
        },
    )

    # ---- isolated node ---------------------------------------------------
    iso = _record([], [0])
    cache_i = _toy_cache([iso], spec, node_types)
    gi = np.asarray(cache_i["global"])[0]
    gmap_i = {name: float(gi[i]) for i, (name, _d) in enumerate(C.GLOBAL_FIELDS)}
    iso_ok = bool(
        np.asarray(cache_i["nbr_code"]).size == 0
        and np.asarray(cache_i["path_cat"]).size == 0
        and np.allclose(np.asarray(cache_i["atom_scalar"]), 0.0, atol=TOL)
        and abs(gmap_i["log1p_n_atoms"] - math.log1p(1)) <= TOL
        and abs(gmap_i["log1p_n_bonds"]) <= TOL
        and abs(gmap_i["n_components"] - 1.0) <= TOL
        and abs(gmap_i["cyclomatic_number"]) <= TOL
        and abs(gmap_i["diameter_max_finite"]) <= TOL
        and abs(gmap_i["mean_finite_pair_distance"]) <= TOL
        and abs(gmap_i["distinct_atom_types"] - 1.0) <= TOL
    )
    out["isolated_node"] = _require(
        iso_ok,
        "isolated node keeps finite zero statistics and empty S1/S2",
        {"global": gi.tolist(), "scalars": np.asarray(cache_i["atom_scalar"]).tolist()},
    )

    # ---- disconnected: edge 0-1 plus isolated node 2 ---------------------
    dis = _record([[0, 1]], [0, 0, 0])
    cache_d = _toy_cache([dis], spec, node_types)
    gd = np.asarray(cache_d["global"])[0]
    gmap_d = {name: float(gd[i]) for i, (name, _d) in enumerate(C.GLOBAL_FIELDS)}
    dis_ok = bool(
        np.asarray(cache_d["path_cat"]).size == 0
        and abs(gmap_d["n_components"] - 2.0) <= TOL
        and abs(gmap_d["cyclomatic_number"]) <= TOL
        and abs(gmap_d["frac_unreachable_pairs"] - 2.0 / 3.0) <= TOL
        and abs(gmap_d["diameter_max_finite"] - 1.0) <= TOL
        and abs(gmap_d["mean_finite_pair_distance"] - 1.0) <= TOL
        and abs(gmap_d["frac_finite_pairs_ge3"]) <= TOL
    )
    out["disconnected"] = _require(
        dis_ok,
        "disconnected graph uses the -1 unreachable sentinel and finite distance statistics",
        {"global": gd.tolist()},
    )

    # ---- Train-only label vocabulary / UNK ------------------------------
    unk_node_types = A.node_type_table("label", [0])
    unk_spec = A.build_tu_spec("NCI1", unk_node_types)
    unk = _record([[0, 1]], [0, 5])
    cache_u = _toy_cache([unk], unk_spec, unk_node_types)
    types = np.asarray(cache_u["atom_type"])
    out["unseen_label_maps_to_unk"] = _require(
        bool(types.tolist() == [0, int(unk_node_types["unk_index"])] and unk_spec.n_atom_types == 2),
        "a node label unseen in Train maps to UNK (vocabulary size 1 + UNK)",
        {"atom_type": types.tolist(), "n_atom_types": int(unk_spec.n_atom_types),
         "names": list(unk_spec.atom_type_names)},
    )

    # ---- degree buckets --------------------------------------------------
    deg_node_types = A.node_type_table("degree", None)
    deg_spec = A.build_tu_spec("IMDB-BINARY", deg_node_types)
    deg_cache = _toy_cache([_record([[0, 1], [1, 2], [2, 0]], [0, 0, 0])], deg_spec, deg_node_types)
    feat = np.asarray(deg_cache["atom_feat"])
    k = deg_spec.n_atom_types
    bucket_hits = [
        int(np.argmax(feat[i, k : k + C.N_DEGREE_BUCKETS])) for i in range(3)
    ]
    out["degree_buckets"] = _require(
        bool(bucket_hits == [2, 2, 2] and k == 18),
        "IMDB node types are degree buckets (triangle -> bucket 2, K=18)",
        {"bucket_hits": bucket_hits, "n_atom_types": int(k)},
    )
    return out


# ---------------------------------------------------------------------------
# gate 3: node renumbering invariance
# ---------------------------------------------------------------------------


def gate_permutation() -> dict[str, Any]:
    node_types = A.node_type_table("label", [0, 1])
    spec = A.build_tu_spec("NCI1", node_types)
    # two graphs: a triangle with a pendant chain, and a 5-cycle
    g1 = _record([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5]], [0, 1, 0, 1, 0, 1])
    g2 = _record([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], [1, 1, 0, 0, 1])
    cache = _toy_cache([g1, g2], spec, node_types)
    norm = _toy_norm(cache, spec)

    perm = np.asarray([3, 0, 5, 1, 4, 2], dtype=np.int64)
    perm2 = np.asarray([2, 0, 4, 1, 3], dtype=np.int64)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    inv2 = np.empty_like(perm2)
    inv2[perm2] = np.arange(perm2.size)
    pairs1 = np.asarray(g1["pairs"], dtype=np.int64)
    permuted1 = _record(
        np.sort(np.stack([perm[pairs1[:, 0]], perm[pairs1[:, 1]]], axis=1), axis=1),
        np.asarray(g1["node_labels"])[inv],
    )
    pairs2 = np.asarray(g2["pairs"], dtype=np.int64)
    permuted2 = _record(
        np.sort(np.stack([perm2[pairs2[:, 0]], perm2[pairs2[:, 1]]], axis=1), axis=1),
        np.asarray(g2["node_labels"])[inv2],
    )
    cache_p = _toy_cache([permuted1, permuted2], spec, node_types)

    feat = np.asarray(cache["atom_feat"])
    feat_p = np.asarray(cache_p["atom_feat"])
    same_multiset = np.allclose(
        np.sort(feat[:6], axis=0), np.sort(feat_p[:6], axis=0), atol=TOL
    ) and np.allclose(np.sort(feat[6:], axis=0), np.sort(feat_p[6:], axis=0), atol=TOL)
    same_scalars = np.allclose(
        np.sort(np.asarray(cache["atom_scalar"])[:6], axis=0),
        np.sort(np.asarray(cache_p["atom_scalar"])[:6], axis=0),
        atol=TOL,
    ) and np.allclose(
        np.sort(np.asarray(cache["atom_scalar"])[6:], axis=0),
        np.sort(np.asarray(cache_p["atom_scalar"])[6:], axis=0),
        atol=TOL,
    )
    same_global = np.allclose(np.asarray(cache["global"]), np.asarray(cache_p["global"]), atol=TOL)

    torch.manual_seed(0)
    model = M.TULCRC(spec, norm, 2, "FULL")
    model.eval()
    with torch.no_grad():
        base = model(_batch(cache, [0, 1]))
        ren = model(_batch(cache_p, [0, 1]))
    logit_delta = float((base - ren).abs().max().item())
    ok = bool(same_multiset and same_scalars and same_global and logit_delta < 1e-5)
    return _require(
        ok,
        "node renumbering leaves the feature multiset, global vector and logits unchanged",
        {
            "feature_multiset_equal": bool(same_multiset),
            "scalar_multiset_equal": bool(same_scalars),
            "global_equal": bool(same_global),
            "max_abs_logit_delta": logit_delta,
            "permutation": perm.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# gate 4: batching + output shapes
# ---------------------------------------------------------------------------


def gate_batching_and_shapes() -> dict[str, Any]:
    node_types = A.node_type_table("label", [0, 1])
    spec = A.build_tu_spec("NCI1", node_types)
    records = [
        _record([[0, 1], [1, 2], [2, 0], [2, 3]], [0, 1, 0, 1]),
        _record([[0, 1], [1, 2]], [1, 1, 0]),
        _record([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], [0, 0, 1, 1, 0]),
        _record([[0, 1], [1, 2], [2, 0]], [1, 0, 1]),
    ]
    cache = _toy_cache(records, spec, node_types)
    norm = _toy_norm(cache, spec)
    rows = [0, 1, 2, 3]
    torch.manual_seed(1)
    model = M.TULCRC(spec, norm, 2, "FULL")
    model.eval()
    with torch.no_grad():
        whole = model(_batch(cache, rows))
        parts = torch.cat([model(_batch(cache, [r])) for r in rows], dim=0)
        split = torch.cat([model(_batch(cache, rows[:2])), model(_batch(cache, rows[2:]))], dim=0)
        gin = M.GINPlusGlobal(spec, norm, 2)
        gin.eval()
        gin_whole = gin(_batch(cache, rows))
        gin_split = torch.cat([gin(_batch(cache, rows[:2])), gin(_batch(cache, rows[2:]))], dim=0)
    delta_parts = float((whole - parts).abs().max().item())
    delta_split = float((whole - split).abs().max().item())
    gin_delta = float((gin_whole - gin_split).abs().max().item())

    deg_types = A.node_type_table("degree", None)
    deg_spec = A.build_tu_spec("IMDB-MULTI", deg_types)
    deg_cache = _toy_cache([_record([[0, 1], [1, 2], [2, 0]], [0, 0, 0])], deg_spec, deg_types)
    deg_norm = _toy_norm(deg_cache, deg_spec)
    torch.manual_seed(2)
    three_full = M.TULCRC(deg_spec, deg_norm, 3, "FULL")(_batch(deg_cache, [0]))
    three_c1 = M.TULCRC(deg_spec, deg_norm, 3, "C1")(_batch(deg_cache, [0]))
    three_gin = M.GINPlusGlobal(deg_spec, deg_norm, 3)(_batch(deg_cache, [0]))
    two = M.TULCRC(spec, norm, 2, "FULL")(_batch(cache, [0]))
    ok = bool(
        delta_parts < 1e-5
        and delta_split < 1e-5
        and gin_delta < 1e-5
        and tuple(two.shape) == (1, 2)
        and tuple(three_full.shape) == (1, 3)
        and tuple(three_c1.shape) == (1, 3)
        and tuple(three_gin.shape) == (1, 3)
    )
    return _require(
        ok,
        "batch split/merge consistency for FULL and GIN; 2-class and 3-class logits have C columns",
        {
            "full_batch_vs_single_graph_max_delta": delta_parts,
            "full_batch_vs_two_sub_batches_max_delta": delta_split,
            "gin_batch_vs_two_sub_batches_max_delta": gin_delta,
            "shapes": {
                "full_2class": list(two.shape),
                "full_3class": list(three_full.shape),
                "c1_3class": list(three_c1.shape),
                "gin_3class": list(three_gin.shape),
            },
        },
    )


# ---------------------------------------------------------------------------
# gate 5: FULL/C1 shared init + deleted block
# ---------------------------------------------------------------------------


def gate_shared_init() -> dict[str, Any]:
    node_types = A.node_type_table("label", [0, 1])
    spec = A.build_tu_spec("NCI1", node_types)
    records = [
        _record([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4]], [0, 1, 0, 1, 0]),
        _record([[0, 1], [1, 2], [2, 3], [3, 0]], [1, 1, 0, 0]),
    ]
    cache = _toy_cache(records, spec, node_types)
    norm = _toy_norm(cache, spec)
    n_classes = 2
    full_state = M.build_full_state(spec, norm, n_classes, C.TRAINING_SEED)
    c1_state = M.remap_state("C1", full_state, spec, n_classes)
    shared = M.shared_state_equal("C1", full_state, c1_state, spec)

    full_model = M.TULCRC(spec, norm, n_classes, "FULL")
    full_model.load_state_dict(full_state, strict=True)
    full_model.eval()
    c1_model = M.TULCRC(spec, norm, n_classes, "C1")
    c1_model.load_state_dict(c1_state, strict=True)
    c1_model.eval()

    layout_full = M.local_layout(spec, "FULL")
    layout_c1 = M.local_layout(spec, "C1")
    b_mean_cols = list(range(*layout_full["b_mean"]))
    full_dims = {k: v[1] - v[0] for k, v in layout_full.items()}
    c1_dims = {k: v[1] - v[0] for k, v in layout_c1.items()}
    width_diff = full_dims["b_mean"]
    param_diff = M.parameter_count(full_model) - M.parameter_count(c1_model)
    expected_param_diff = width_diff * int(C.MODEL_CONFIG["local_hidden"])
    mlp_in_full = full_state["local.mlp.lins.0.weight"].shape[1]
    mlp_in_c1 = c1_state["local.mlp.lins.0.weight"].shape[1]

    batch = _batch(cache, [0, 1])
    with torch.no_grad():
        c1_logits = c1_model(batch)
        zeroed_state = {k: v.clone() for k, v in full_state.items()}
        w = zeroed_state["local.mlp.lins.0.weight"]
        w[:, b_mean_cols] = 0.0
        zeroed = M.TULCRC(spec, norm, n_classes, "FULL")
        zeroed.load_state_dict(zeroed_state, strict=True)
        zeroed.eval()
        zeroed_logits = zeroed(batch)
        full_logits = full_model(batch)
    delta = float((zeroed_logits - c1_logits).abs().max().item())
    full_vs_c1 = float((full_logits - c1_logits).abs().max().item())
    checks = {
        "shared_params_bitwise_equal": bool(shared["all_equal"]),
        "n_shared_tensors_checked": int(shared["n_checked"]),
        "local_input_dim_full": int(mlp_in_full),
        "local_input_dim_c1": int(mlp_in_c1),
        "deleted_block": list(layout_full["b_mean"]),
        "deleted_columns": len(b_mean_cols),
        "param_diff": int(param_diff),
        "expected_param_diff": int(expected_param_diff),
        "zeroed_bmean_full_vs_c1_max_abs_logit_delta": delta,
        "full_vs_c1_max_abs_logit_delta": full_vs_c1,
        "c1_has_b_mean": bool(c1_model.local.use_b_mean),
        "full_has_b_mean": bool(full_model.local.use_b_mean),
        "c1_local_layout": {k: list(v) for k, v in layout_c1.items()},
        "full_local_layout": {k: list(v) for k, v in layout_full.items()},
    }
    ok = bool(
        shared["all_equal"]
        and mlp_in_full - mlp_in_c1 == width_diff == 96
        and param_diff == expected_param_diff == 18432
        and delta < 1e-6
        and not c1_model.local.use_b_mean
        and full_model.local.use_b_mean
    )
    return _require(ok, "FULL/C1 share every init tensor; C1 deletes exactly the b_mean block", checks)


# ---------------------------------------------------------------------------
# gate 6: gradients in every used branch
# ---------------------------------------------------------------------------


def gate_gradients() -> dict[str, Any]:
    node_types = A.node_type_table("label", [0, 1])
    spec = A.build_tu_spec("NCI1", node_types)
    records = [
        _record([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4]], [0, 1, 0, 1, 0]),
        _record([[0, 1], [1, 2], [2, 3], [3, 0]], [1, 1, 0, 0]),
    ]
    cache = _toy_cache(records, spec, node_types)
    norm = _toy_norm(cache, spec)
    batch = _batch(cache, [0, 1])
    out: dict[str, Any] = {}
    ok_all = True
    for model_name, cls in (("FULL", None), ("C1", None), ("GIN", None)):
        torch.manual_seed(C.TRAINING_SEED)
        if model_name == "GIN":
            model = M.GINPlusGlobal(spec, norm, 2)
        else:
            model = M.TULCRC(spec, norm, 2, model_name)
        report = _grad_report(model, batch, 2)
        dead = [name for name, row in report.items() if not row["present"]]
        ok = len(dead) == 0
        ok_all = ok_all and ok
        out[model_name] = _require(
            ok,
            f"every trainable tensor of {model_name} receives a finite non-zero gradient",
            {
                "n_tensors": len(report),
                "dead_tensors": dead,
                "grad_norm": {k: v["grad_norm"] for k, v in report.items()},
            },
        )
    # Explicit branch probe: zero only the S2 content gate and confirm a change.
    torch.manual_seed(C.TRAINING_SEED)
    model = M.TULCRC(spec, norm, 2, "FULL")
    model.eval()
    with torch.no_grad():
        base = model(batch)
        gate = model.local.path_gate.weight
        gate.zero_()
        changed = model(batch)
    probe = float((base - changed).abs().max().item())
    out["content_gate_probe"] = _require(
        probe > 1e-8,
        "zeroing the S2 content gate changes the logits (content channel is in the computation)",
        {"max_abs_logit_delta": probe},
    )
    ok_all = ok_all and probe > 1e-8
    return {"ok": bool(ok_all), "checks": out}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_all(datasets: Sequence[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "environment": C.environment(),
        "code_state": C.code_state(),
    }
    payload["split_and_norm"] = gate_split_and_norm(datasets)
    payload["hand_computed"] = gate_hand_computed()
    payload["permutation_invariance"] = gate_permutation()
    payload["batching_and_shapes"] = gate_batching_and_shapes()
    payload["shared_init"] = gate_shared_init()
    payload["gradients"] = gate_gradients()

    def _walk(obj: Any) -> list[bool]:
        if isinstance(obj, dict):
            if "ok" in obj and isinstance(obj["ok"], bool):
                return [obj["ok"]]
            flags: list[bool] = []
            for v in obj.values():
                flags.extend(_walk(v))
            return flags
        return []

    flags = _walk({k: v for k, v in payload.items() if k not in ("environment", "code_state")})
    payload["n_checks"] = len(flags)
    payload["all_passed"] = bool(all(flags))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(sorted(C.DATASETS)))
    parser.add_argument("--out", default=str(C.ANALYSIS / "gates.json"))
    args = parser.parse_args(argv)
    datasets = [d for d in args.datasets.split(",") if d]
    payload = run_all(datasets)
    C.write_json(args.out, payload)
    print(C.stable_json({"all_passed": payload["all_passed"], "n_checks": payload["n_checks"]}))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
