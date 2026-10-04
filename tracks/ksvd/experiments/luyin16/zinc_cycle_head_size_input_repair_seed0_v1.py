"""ZINC independent cycle head — minimal size-input repair (CPU, seed 0).

One change, tested end to end: give the existing independent cycle head the
actual graph size (``N`` = actual ``num_nodes``, ``E`` = number of unique
undirected pairs) in addition to the frozen ``T25`` input, and ask whether

1. the full-fit exact-input conflicts of the old 8000-fit head shrink,
2. a pair of otherwise identical small heads (``Q0``: T25 only, size branch
   gated to 0; ``QNE``: T25 + standardized N/E) fits / transfers better.

Strictly train-only.  The old 8000/2000 internal fold, the frozen chemistry
branch ``O_seed0`` (``h_raw``, g-predictor) and the existing ``T25`` model input
are reused byte-for-byte.  ``c``/``g``/``k`` come from the existing train-only
``target_decomposition.npz``.  N/E come from the actual train graphs
(``zinc_dictionary_real_data_handoff/train.npz``, ``official_test_loaded=False``)
and are computed from ``N``/unique-undirected-pair counts only; no atom/bond
label, ``y``, ``c``, molecule id, SMILES or cycle target enters them.

Official-valid and official-test are never loaded, predicted or scored.

Stage A is always run; stage B is bought only if the pre-registered structure
gate passes.  See ``PROTOCOL.md`` for the frozen rules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

PROTOCOL_VERSION = "zinc-cycle-head-size-input-repair-seed0-v1"

TRACK_ROOT = zjd.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results/zinc_cycle_head_size_input_repair_seed0_v1"
FROZEN_CYCLE = TRACK_ROOT / "results/zinc_frozen_chemistry_learned_cycle_v1"
FROZEN_O = TRACK_ROOT / "results/zinc_full_cycle_target_decomposition_v1"
OVERNIGHT = TRACK_ROOT / "results/zinc_overnight_interface_and_tail_seed0_v1"
WITNESS = TRACK_ROOT / "results/zinc_task_dictionary_and_cycle_witness_seed0_v1"
HANDOFF = zjd.HANDOFF

T25_PATH = FROZEN_CYCLE / "T25_all.npz"
O_PRED_PATH = FROZEN_O / "O_seed0_predictions.npz"
O_SOUP_PATH = FROZEN_O / "O_seed0_raw_soup_state.pt"
DECOMP_PATH = FROZEN_O / "target_decomposition.npz"
PREP_BLOB = zjd.PREP_DIR / "fold_objects.npz"
HANDOFF_TRAIN = HANDOFF / "train.npz"
PU_PRED_PATH = OVERNIGHT / "cpu_P_U_predictions.npz"
PU_INIT_PATH = OVERNIGHT / "cpu_P_U_head_init_state.pt"
PU_SOUP_PATH = OVERNIGHT / "cpu_P_U_head_soup_state.pt"
PU_META_PATH = OVERNIGHT / "cpu_P_U.json"
PU_ANALYSIS_PATH = OVERNIGHT / "cpu_analysis_8k.json"
PS0_PRED_PATH = FROZEN_CYCLE / "P_seed0_predictions.npz"
WITNESS_JSON = WITNESS / "cycle_class_witnesses.json"

EXPECTED = {
    "T25_sha256": "dc2e151610209879941df8b12eaa0eee5d2ff3d11d231bf0f9e3e30294b4591c",
    "fit_idx_sha256": "165e87ef4398ba8ef57411c2f118c4cca4ea74007f2485611b84f0c09bbdd9ea",
    "dev_idx_sha256": "fb8b78063e7c8a5c759bfa7d553738068cee1738a2d67b210e5d9dc492331376",
    "O_pred_file_sha256": "5cee578213d321660c60105fd8c201668e20132600f8412302b54cc43d68db2a",
    "O_soup_file_sha256": "61d4aebbcda061fd118c4faa09e992b437dfa2d3306d8f3cee58adf9ce749ac4",
    "prep_blob_sha256": "968e82dda591546748f44a0b9825ae341774f37cdec05a11580cd58b5ac4d4bb",
    "old_n_classes": 327,
    "old_n_conflict_classes": 4,
    "old_conflicting_rows": 10,
    "old_global_min_l1_per_row": 0.006069723478421038,
    "old_fit_k": {"k=0": 7702, "k=-1": 260, "k=-2": 33, "k<=-3": 5},
    "dev_k": {"k=0": 1926, "k=-1": 65, "k=-2": 7, "k<=-3": 2},
    "P_U_dev_cal": 0.113180,
}

SEED = 0
EPOCHS = 300
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_LAST = 5
TRAIN_GEN_BASE = 20261003
INPUT_GEN_SEED = 20261004
BOOT_SEED = 20261004
N_BOOT = 1000
DELTA = 0.003
G0_TOL = 0.001
TOPOLOGY_IN = 25
HIDDEN = (64, 32)
BASE_PARAMETERS = 3777
SIZE_PARAMETERS = 128
TOTAL_PARAMETERS = BASE_PARAMETERS + SIZE_PARAMETERS
STD_FLOOR = 1.0e-6
REPLAY_TOL = 2.0e-6
SEVERE_MAX = -2
SIZE_GATE_GAP = 1.0e-10
FIXED_DEV_BATCH = 32
FWD_BATCH = 128


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in names})


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_arr(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def write_manifest(out_dir: Path) -> None:
    manifest = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest[str(path.relative_to(out_dir))] = sha_file(path)
    write_json(out_dir / "manifest.json", {"protocol_version": PROTOCOL_VERSION, "files": manifest})


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def stable_id(row: int) -> str:
    return f"train:{int(row):04d}"


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "k=0": k == 0,
        "k=-1": k == -1,
        "k=-2": k == -2,
        "k<=-3": k <= -3,
        "k<=-2": k <= -2,
    }


def group_name(k: int) -> str:
    if int(k) == 0:
        return "k=0"
    if int(k) == -1:
        return "k=-1"
    if int(k) == -2:
        return "k=-2"
    return "k<=-3"


def exact_classes(X: np.ndarray) -> dict[str, Any]:
    """Exact float32 equality classes (no tolerance); NaN forbidden."""
    X = np.ascontiguousarray(X, np.float32)
    if np.isnan(X).any():
        raise RuntimeError("input contains NaN; exact-class semantics undefined")
    negative_zero = int(np.sum(np.signbit(X) & (X == 0)))
    order = np.lexsort(X.T)
    Xs = X[order]
    new_class = np.ones(Xs.shape[0], dtype=bool)
    if Xs.shape[0] > 1:
        new_class[1:] = np.any(Xs[1:] != Xs[:-1], axis=1)
    class_id_sorted = np.cumsum(new_class) - 1
    class_id = np.empty(Xs.shape[0], np.int64)
    class_id[order] = class_id_sorted
    counts = np.bincount(class_id)
    keys = [row.tobytes() for row in Xs]
    byte_classes = len(set(keys))
    return {
        "class_id": class_id,
        "order": order,
        "counts": counts,
        "n_classes": int(counts.size),
        "byte_class_count": int(byte_classes),
        "negative_zero_count": negative_zero,
        "signed_zero_consistent": bool(byte_classes == counts.size),
    }


def class_analysis(X: np.ndarray, c: np.ndarray, k: np.ndarray) -> dict[str, Any]:
    """Full-fit exact-class statistics (medians are diagnostics, not a model)."""
    classes = exact_classes(X)
    class_id = classes["class_id"]
    counts = classes["counts"]
    n = int(X.shape[0])
    med = np.empty(int(counts.size), np.float64)
    spread = np.empty(int(counts.size), np.float64)
    unique_c = np.empty(int(counts.size), np.int64)
    for cid in range(int(counts.size)):
        rows = np.where(class_id == cid)[0]
        vals = c[rows]
        med[cid] = float(np.median(vals))
        spread[cid] = float(np.max(vals) - np.min(vals))
        unique_c[cid] = int(np.unique(vals).size)
    conflict = (counts >= 2) & (unique_c > 1)
    size = counts[class_id]
    residual = np.abs(c - med[class_id])
    masks = group_masks(k)
    group_cost = {}
    group_total = {}
    group_n = {}
    for name, mask in masks.items():
        group_n[name] = int(mask.sum())
        group_cost[name] = float(residual[mask].mean()) if int(mask.sum()) else None
        group_total[name] = float(residual[mask].sum())
    conflict_rows = conflict[class_id]
    return {
        "n_rows": n,
        "n_exact_classes": int(counts.size),
        "n_multi_classes": int((counts >= 2).sum()),
        "n_conflict_classes": int(conflict.sum()),
        "conflict_class_ids": [int(i) for i in np.where(conflict)[0]],
        "repeated_row_fraction": float((size >= 2).mean()),
        "conflicting_row_count": int(conflict_rows.sum()),
        "conflicting_row_fraction": float(conflict_rows.mean()),
        "global_min_l1_per_row": float(residual.mean()),
        "global_min_l1_total": float(residual.sum()),
        "group_cost": group_cost,
        "group_total": group_total,
        "group_n": group_n,
        "class_id": class_id,
        "counts": counts,
        "class_median": med,
        "class_spread": spread,
        "class_unique_c": unique_c,
        "conflict": conflict,
        "residual": residual,
        "byte_class_count": classes["byte_class_count"],
        "negative_zero_count": classes["negative_zero_count"],
        "signed_zero_consistent": classes["signed_zero_consistent"],
    }


def bytes_key(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row, np.float32).tobytes()


# ---------------------------------------------------------------------------
# stage A
# ---------------------------------------------------------------------------


def load_frozen_inputs(log: Any = print) -> dict[str, Any]:
    t25 = np.load(T25_PATH, allow_pickle=False)
    T = np.asarray(t25["T"], np.float32)
    fit_idx = np.asarray(t25["fit_idx"], np.int64)
    dev_idx = np.asarray(t25["dev_idx"], np.int64)
    o = np.load(O_PRED_PATH, allow_pickle=False)
    fit_idx_o = np.asarray(o["fit_idx"], np.int64)
    dev_idx_o = np.asarray(o["dev_idx"], np.int64)
    y_fit = np.asarray(o["fit_y"], np.float64)
    c_fit = np.asarray(o["fit_c"], np.float64)
    h_fit = np.asarray(o["fit_raw"], np.float64)
    y_dev = np.asarray(o["dev_y"], np.float64)
    c_dev = np.asarray(o["dev_c"], np.float64)
    h_dev = np.asarray(o["dev_raw"], np.float64)
    with np.load(DECOMP_PATH, allow_pickle=False) as z:
        y_all = np.asarray(z["y"], np.float64)
        c_all = np.asarray(z["c"], np.float64)
        g_all = np.asarray(z["g"], np.float64)
        k_all = np.asarray(z["k"], np.int64)
        gid_all = np.asarray(z["gid"], np.int64)
    checks = {
        "T25_sha256": sha_arr(T),
        "T25_expected": EXPECTED["T25_sha256"],
        "T25_ok": sha_arr(T) == EXPECTED["T25_sha256"],
        "fit_idx_sha256": hashlib.sha256(fit_idx.tobytes()).hexdigest(),
        "fit_idx_ok": hashlib.sha256(fit_idx.tobytes()).hexdigest() == EXPECTED["fit_idx_sha256"],
        "dev_idx_sha256": hashlib.sha256(dev_idx.tobytes()).hexdigest(),
        "dev_idx_ok": hashlib.sha256(dev_idx.tobytes()).hexdigest() == EXPECTED["dev_idx_sha256"],
        "fit_idx_matches_O": bool(np.array_equal(fit_idx, fit_idx_o)),
        "dev_idx_matches_O": bool(np.array_equal(dev_idx, dev_idx_o)),
        "O_pred_file_sha256": sha_file(O_PRED_PATH),
        "O_pred_file_ok": sha_file(O_PRED_PATH) == EXPECTED["O_pred_file_sha256"],
        "O_soup_file_sha256": sha_file(O_SOUP_PATH),
        "O_soup_file_ok": sha_file(O_SOUP_PATH) == EXPECTED["O_soup_file_sha256"],
        "prep_blob_sha256": sha_file(PREP_BLOB),
        "prep_blob_ok": sha_file(PREP_BLOB) == EXPECTED["prep_blob_sha256"],
        "T25_shape": list(T.shape),
        "n_fit": int(fit_idx.size),
        "n_dev": int(dev_idx.size),
        "overlap": int(len(set(fit_idx.tolist()) & set(dev_idx.tolist()))),
        "y_fit_identity": float(np.max(np.abs(y_fit - y_all[fit_idx]))),
        "y_dev_identity": float(np.max(np.abs(y_dev - y_all[dev_idx]))),
        "c_fit_identity": float(np.max(np.abs(c_fit - c_all[fit_idx]))),
        "k_fit_groups": {name: int(mask.sum()) for name, mask in group_masks(k_all[fit_idx]).items()},
        "k_dev_groups": {name: int(mask.sum()) for name, mask in group_masks(k_all[dev_idx]).items()},
    }
    checks["fit_k_ok"] = {k: checks["k_fit_groups"][k] for k in EXPECTED["old_fit_k"]} == EXPECTED["old_fit_k"]
    checks["dev_k_ok"] = {k: checks["k_dev_groups"][k] for k in EXPECTED["dev_k"]} == EXPECTED["dev_k"]
    checks["gid_matches_index"] = bool(np.array_equal(gid_all, np.arange(10000)))
    log(f"[frozen] T25={checks['T25_ok']} fit={checks['fit_idx_ok']} dev={checks['dev_idx_ok']} "
        f"O_pred={checks['O_pred_file_ok']} O_soup={checks['O_soup_file_ok']} prep={checks['prep_blob_ok']}")
    return {
        "T": T,
        "fit_idx": fit_idx,
        "dev_idx": dev_idx,
        "y_all": y_all,
        "c_all": c_all,
        "g_all": g_all,
        "k_all": k_all,
        "y_fit": y_fit,
        "c_fit": c_fit,
        "h_fit": h_fit,
        "y_dev": y_dev,
        "c_dev": c_dev,
        "h_dev": h_dev,
        "checks": checks,
    }


def compute_size_features(log: Any = print) -> dict[str, Any]:
    """N/E from the actual train graphs, with anomaly + NetworkX audits."""
    h = np.load(HANDOFF_TRAIN, allow_pickle=False)
    ids = np.asarray(h["ids"], np.int64)
    y_handoff = np.asarray(h["y"], np.float64)
    node_ptr = np.asarray(h["node_ptr"], np.int64)
    edge_ptr = np.asarray(h["edge_ptr"], np.int64)
    edge_u_all = np.asarray(h["edge_u"], np.int64)
    edge_v_all = np.asarray(h["edge_v"], np.int64)
    topo_model_input = np.asarray(h["topo_model_input"], np.float32)
    n_graphs = int(node_ptr.size - 1)
    n_nodes = np.diff(node_ptr)
    m_edges = np.diff(edge_ptr)
    if int(edge_ptr[-1]) != int(edge_u_all.size) or int(edge_ptr[-1]) != int(edge_v_all.size):
        raise RuntimeError("edge_ptr does not match edge arrays")
    graph_of_edge = np.repeat(np.arange(n_graphs, dtype=np.int64), m_edges)
    u = edge_u_all
    v = edge_v_all
    self_loop = u == v
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    # unique unordered pair key per edge (self loops included as (a,a))
    key = lo * n_nodes[graph_of_edge] + hi
    # count duplicates: consecutive equal (graph,key) after sorting by (graph,key)
    gkey_order = np.lexsort((key, graph_of_edge))
    gs_all = graph_of_edge[gkey_order]
    ks_all = key[gkey_order]
    same = np.zeros(ks_all.size, dtype=bool)
    if ks_all.size > 1:
        same[1:] = (ks_all[1:] == ks_all[:-1]) & (gs_all[1:] == gs_all[:-1])
    dup_per_graph = np.bincount(gs_all[same], minlength=n_graphs) if ks_all.size else np.zeros(n_graphs, np.int64)
    unique_per_graph = m_edges - dup_per_graph
    E = unique_per_graph
    N = n_nodes
    # reverse-edge semantics (is the stored list a single undirected listing?)
    dir_key = u * n_nodes[graph_of_edge] + v
    rev_key = v * n_nodes[graph_of_edge] + u
    dir_order = np.lexsort((dir_key, graph_of_edge))
    ds = graph_of_edge[dir_order]
    dk = dir_key[dir_order]
    new_d = np.ones(dk.size, dtype=bool)
    if dk.size > 1:
        new_d[1:] = (dk[1:] != dk[:-1]) | (ds[1:] != ds[:-1])
    d_starts = np.where(new_d)[0]
    unique_dir_total = int(d_starts.size)
    rev_present = 0
    if rev_key.size:
        # searchsorted per graph: unique directed sorted keys
        pos = np.searchsorted(dk, rev_key[dir_order])
        pos_ok = pos < dk.size
        match = np.zeros(dk.size, dtype=bool)
        m = pos_ok
        match[m] = dk[pos[m]] == rev_key[dir_order][m]
        # need same graph: since dk sorted overall by graph then key, compare graph too
        gmatch = np.zeros(dk.size, dtype=bool)
        gmatch[m] = ds[pos[m]] == graph_of_edge[dir_order][m]
        rev_present = int(np.sum(match & gmatch))
    # endpoint range audit
    endpoints_ok = True
    bad_lo = int(np.sum((u < 0) | (v < 0)))
    bad_hi = 0
    for gid in range(n_graphs):
        s, e = edge_ptr[gid], edge_ptr[gid + 1]
        if e > s:
            if int(edge_u_all[s:e].max()) >= int(N[gid]) or int(edge_v_all[s:e].max()) >= int(N[gid]):
                bad_hi += 1
    endpoints_ok = bad_lo == 0 and bad_hi == 0
    # independent NetworkX reference, all graphs
    import networkx as nx

    nx_mismatch = []
    for gid in range(n_graphs):
        s, e = int(edge_ptr[gid]), int(edge_ptr[gid + 1])
        G = nx.Graph()
        G.add_nodes_from(range(int(N[gid])))
        for a, b in zip(edge_u_all[s:e].tolist(), edge_v_all[s:e].tolist()):
            G.add_edge(int(a), int(b))
        if G.number_of_nodes() != int(N[gid]) or G.number_of_edges() != int(E[gid]):
            nx_mismatch.append(gid)
    # renumbering invariance: deterministic sample incl. all non-k0 graphs later
    rng = np.random.default_rng(INPUT_GEN_SEED)
    sample = np.sort(rng.choice(n_graphs, size=min(256, n_graphs), replace=False))
    renumber_bad = []
    for gid in sample.tolist():
        s, e = int(edge_ptr[gid]), int(edge_ptr[gid + 1])
        perm = rng.permutation(int(N[gid]))
        Eu = set()
        for a, b in zip(edge_u_all[s:e].tolist(), edge_v_all[s:e].tolist()):
            aa, bb = int(perm[a]), int(perm[b])
            Eu.add((min(aa, bb), max(aa, bb)))
        if len(Eu) != int(E[gid]):
            renumber_bad.append(gid)
    out = {
        "source": str(HANDOFF_TRAIN.relative_to(TRACK_ROOT)),
        "source_file_sha256": sha_file(HANDOFF_TRAIN),
        "official_test_loaded": bool(h["official_test_loaded"]),
        "split_values": [str(h["split"].item())] if np.ndim(h["split"]) == 0 else [str(x) for x in np.asarray(h["split"]).tolist()],
        "n_graphs": n_graphs,
        "ids_are_row_index": bool(np.array_equal(ids, np.arange(n_graphs))),
        "node_ptr_matches_num_nodes": True,  # verified against encoded cache in runner main
        "total_nodes": int(n_nodes.sum()),
        "total_raw_edges": int(m_edges.sum()),
        "self_loop_count": int(self_loop.sum()),
        "total_unique_unordered_pairs": int(m_edges.sum() - dup_per_graph.sum()),
        "duplicate_unordered_pair_count": int(dup_per_graph.sum()),
        "unique_directed_edges": unique_dir_total,
        "reverse_edges_present": rev_present,
        "edge_list_semantics": "single undirected pair list" if rev_present == 0 else "contains reverse duplicates",
        "endpoints_in_range": bool(endpoints_ok),
        "bad_endpoint_counts": {"negative": bad_lo, "out_of_range_graphs": bad_hi},
        "nx_reference_mismatch_graphs": nx_mismatch,
        "renumber_sample": int(sample.size),
        "renumber_mismatch_graphs": renumber_bad,
        "N_summary": {
            "min": int(N.min()), "max": int(N.max()), "mean": float(N.mean()),
            "unique": int(np.unique(N).size),
        },
        "E_summary": {
            "min": int(E.min()), "max": int(E.max()), "mean": float(E.mean()),
            "unique": int(np.unique(E).size),
        },
        "all_ok": bool(
            h["official_test_loaded"] is False or h["official_test_loaded"] == False  # noqa: E712
        ),
        "handoff_y_matches_decomp": None,  # filled by caller
        "handoff_topo_model_input_vs_T25_maxdiff": None,  # filled by caller
    }
    out["all_ok"] = bool(
        out["official_test_loaded"] is False
        and out["split_values"] == ["train"]
        and out["ids_are_row_index"]
        and out["self_loop_count"] == 0
        and out["duplicate_unordered_pair_count"] == 0        and rev_present == 0
        and endpoints_ok
        and not nx_mismatch
        and not renumber_bad
    )
    log(f"[size] N={out['N_summary']} E={out['E_summary']} self_loops={out['self_loop_count']} "
        f"dup={out['duplicate_unordered_pair_count']} rev={rev_present} nx_bad={len(nx_mismatch)} "
        f"renumber_bad={len(renumber_bad)}")
    return {"N": N.astype(np.int64), "E": E.astype(np.int64), "audit": out, "y_handoff": y_handoff,
            "edge_ptr": edge_ptr, "edge_u": edge_u_all, "edge_v": edge_v_all, "node_ptr": node_ptr,
            "topo_model_input": topo_model_input}


def size_scaler(N: np.ndarray, E: np.ndarray, fit_idx: np.ndarray) -> dict[str, float]:
    n_fit = N[fit_idx].astype(np.float64)
    e_fit = E[fit_idx].astype(np.float64)
    return {
        "n_mean": float(n_fit.mean()),
        "n_std": max(float(n_fit.std(ddof=0)), STD_FLOOR),
        "e_mean": float(e_fit.mean()),
        "e_std": max(float(e_fit.std(ddof=0)), STD_FLOOR),
        "std_floor": STD_FLOOR,
        "n_fit_rows": int(fit_idx.size),
        "weighting": "graph-level rows, unweighted, float64",
    }


def standardize_size(N: np.ndarray, E: np.ndarray, scaler: Mapping[str, float]) -> np.ndarray:
    sz = np.empty((N.shape[0], 2), np.float32)
    sz[:, 0] = ((N.astype(np.float64) - scaler["n_mean"]) / scaler["n_std"]).astype(np.float32)
    sz[:, 1] = ((E.astype(np.float64) - scaler["e_mean"]) / scaler["e_std"]).astype(np.float32)
    return sz


def size_vs_t25_relation(T: np.ndarray, N: np.ndarray, E: np.ndarray, fit_idx: np.ndarray,
                         class25: Mapping[str, Any]) -> dict[str, Any]:
    cols = {}
    for j in range(T.shape[1]):
        v = T[fit_idx, j].astype(np.float64)
        if v.std() == 0:
            corr_n = corr_e = 0.0
        else:
            corr_n = float(np.corrcoef(v, N[fit_idx].astype(np.float64))[0, 1])
            corr_e = float(np.corrcoef(v, E[fit_idx].astype(np.float64))[0, 1])
        cols[f"T{j}"] = {"corr_N": corr_n, "corr_E": corr_e}
    max_abs_N = max(abs(v["corr_N"]) for v in cols.values())
    max_abs_E = max(abs(v["corr_E"]) for v in cols.values())
    counts = class25["counts"]
    class_id = class25["class_id"]
    multi = counts[class_id] >= 2
    n_multi_classes = 0
    n_multi_class_with_N_variation = 0
    n_multi_class_with_E_variation = 0
    for cid in np.unique(class_id[multi]):
        rows = class_id == cid
        n_multi_classes += 1
        if np.unique(N[fit_idx][rows]).size > 1:
            n_multi_class_with_N_variation += 1
        if np.unique(E[fit_idx][rows]).size > 1:
            n_multi_class_with_E_variation += 1
    return {
        "per_column_corr": cols,
        "max_abs_corr_with_N": max_abs_N,
        "max_abs_corr_with_E": max_abs_E,
        "multi_T25_classes": int(n_multi_classes),
        "multi_T25_classes_with_N_variation": int(n_multi_class_with_N_variation),
        "multi_T25_classes_with_E_variation": int(n_multi_class_with_E_variation),
        "note": "correlations are fit-row descriptive; T25 does not determine N/E exactly whenever variation > 0",
    }


# ---------------------------------------------------------------------------
# stage A reporting
# ---------------------------------------------------------------------------


def conflict_split_table(c25: Mapping[str, Any], c27: Mapping[str, Any], N: np.ndarray, E: np.ndarray,
                         fit_idx: np.ndarray, c_all: np.ndarray, k_all: np.ndarray,
                         T: np.ndarray, X27: np.ndarray, witnesses: Mapping[str, Any]) -> dict[str, Any]:
    old_ids = c25["conflict_class_ids"]
    out = {"old_conflict_class_ids": old_ids, "classes": []}
    rows_out = []
    for cid in old_ids:
        fit_rows = np.where(c25["class_id"] == cid)[0]
        entry = {
            "old_class_id": int(cid),
            "class_size": int(fit_rows.size),
            "old_c_span": float(c25["class_spread"][cid]),
            "members": [],
            "new_multi_classes": [],
        }
        new_ids = sorted(set(int(c27["class_id"][r]) for r in fit_rows))
        for nid in new_ids:
            nrows = np.where(c27["class_id"] == nid)[0]
            entry["new_multi_classes"].append({
                "new_class_id": int(nid),
                "size": int(nrows.size),
                "c_span": float(c27["class_spread"][nid]),
                "members_in_old_class": [int(r) for r in nrows if r in set(fit_rows.tolist())],
                "k_values": sorted(set(int(k_all[fit_idx[r]]) for r in nrows)),
            })
        for r in fit_rows.tolist():
            row = int(fit_idx[r])
            nid = int(c27["class_id"][r])
            entry["members"].append({
                "fit_position": int(r),
                "stable_id": stable_id(row),
                "k": int(k_all[row]),
                "group": group_name(int(k_all[row])),
                "c": float(c_all[row]),
                "N": int(N[row]),
                "E": int(E[row]),
                "T25_sha256": hashlib.sha256(bytes_key(T[row])).hexdigest(),
                "new_class_id": nid,
                "new_class_size": int(c27["counts"][nid]),
                "new_class_c_span": float(c27["class_spread"][nid]),
                "new_class_k_values": sorted(set(int(k_all[fit_idx[x]]) for x in np.where(c27["class_id"] == nid)[0])),
                "separated_as_singleton": bool(c27["counts"][nid] == 1),
            })
            rows_out.append({
                "old_class_id": int(cid),
                "fit_position": int(r),
                "stable_id": stable_id(row),
                "k": int(k_all[row]),
                "group": group_name(int(k_all[row])),
                "c": float(c_all[row]),
                "N": int(N[row]),
                "E": int(E[row]),
                "new_class_id": nid,
                "new_class_size": int(c27["counts"][nid]),
                "new_class_c_span": float(c27["class_spread"][nid]),
                "new_class_k_values": ";".join(str(v) for v in sorted(set(int(k_all[fit_idx[x]]) for x in np.where(c27["class_id"] == nid)[0]))),
            })
        out["classes"].append(entry)
    # witness cross-check (old positions)
    checks = []
    for cl in witnesses["conflict_classes"]:
        checks.append({
            "witness_class_id": int(cl["class_id"]),
            "witness_positions": cl["member_positions"],
            "matches_reproduced": bool(cl["class_id"] in old_ids and
                                       np.array_equal(np.sort(np.array(cl["member_positions"])),
                                                      np.sort(np.where(c25["class_id"] == cl["class_id"])[0]))),
        })
    out["witness_position_checks"] = checks
    out["witness_all_match"] = bool(all(c["matches_reproduced"] for c in checks))
    return out, rows_out


def tail_rows_structure(c25, c27, N, E, fit_idx, c_all, k_all, T) -> dict[str, Any]:
    out = {"fit_tail_rows": [], "dev_tail_rows": []}
    c_fit = c_all[fit_idx]
    k_fit = k_all[fit_idx]
    tail = np.where(k_fit <= -3)[0]
    for r in tail.tolist():
        row = int(fit_idx[r])
        nid = int(c27["class_id"][r])
        nrows = np.where(c27["class_id"] == nid)[0]
        out["fit_tail_rows"].append({
            "fit_position": int(r),
            "stable_id": stable_id(row),
            "k": int(k_fit[r]),
            "c": float(c_fit[r]),
            "N": int(N[row]),
            "E": int(E[row]),
            "old_class_id": int(c25["class_id"][r]),
            "old_class_size": int(c25["counts"][c25["class_id"][r]]),
            "old_class_in_conflict": bool(c25["conflict"][c25["class_id"][r]]),
            "old_class_c_span": float(c25["class_spread"][c25["class_id"][r]]),
            "new_class_id": nid,
            "new_class_size": int(c27["counts"][nid]),
            "new_class_c_span": float(c27["class_spread"][nid]),
            "new_class_members": [
                {
                    "fit_position": int(x),
                    "stable_id": stable_id(int(fit_idx[x])),
                    "k": int(k_fit[x]),
                    "c": float(c_fit[x]),
                    "N": int(N[int(fit_idx[x])]),
                    "E": int(E[int(fit_idx[x])]),
                }
                for x in nrows.tolist()
            ],
            "new_class_k_values": sorted(set(int(k_fit[x]) for x in nrows.tolist())),
            "still_mixed_k": bool(len(set(int(k_fit[x]) for x in nrows.tolist())) > 1),
        })
    return out


# ---------------------------------------------------------------------------
# stage B — model / training
# ---------------------------------------------------------------------------


def build_base25(seed: int, bias_value: float) -> nn.Module:
    torch.manual_seed(int(seed))
    base = nn.Sequential(
        nn.Linear(TOPOLOGY_IN, HIDDEN[0]),
        nn.SiLU(),
        nn.Linear(HIDDEN[0], HIDDEN[1]),
        nn.SiLU(),
        nn.Linear(HIDDEN[1], 1),
    )
    if int(sum(p.numel() for p in base.parameters())) != BASE_PARAMETERS:
        raise RuntimeError("base head parameter audit failed")
    nn.init.zeros_(base[4].weight)
    with torch.no_grad():
        base[4].bias.copy_(torch.tensor(float(bias_value)))
    return base


class SizeConditionedHead(nn.Module):
    """Original 25->64->32->1 stack plus a no-bias 2->64 size projection."""

    def __init__(self, base: nn.Sequential, gate: float) -> None:
        super().__init__()
        self.fc1 = base[0]
        self.fc2 = base[2]
        self.fc3 = base[4]
        self.w_s = nn.Linear(2, HIDDEN[0], bias=False)
        nn.init.zeros_(self.w_s.weight)
        self.gate = float(gate)

    def forward(self, t25: torch.Tensor, size2: torch.Tensor) -> torch.Tensor:
        z = self.fc1(t25)
        if self.gate != 0.0:
            z = z + self.gate * self.w_s(size2)
        else:
            # keep the zero W_S tensor in the graph for both arms
            z = z + self.gate * F.linear(size2, self.w_s.weight)
        z = F.silu(z)
        z = F.silu(self.fc2(z))
        return self.fc3(z).view(-1)

    def native25_state(self) -> dict[str, torch.Tensor]:
        return {
            "0.weight": self.fc1.weight.detach().clone(),
            "0.bias": self.fc1.bias.detach().clone(),
            "2.weight": self.fc2.weight.detach().clone(),
            "2.bias": self.fc2.bias.detach().clone(),
            "4.weight": self.fc3.weight.detach().clone(),
            "4.bias": self.fc3.bias.detach().clone(),
        }


def make_head(base_init: Mapping[str, torch.Tensor], seed: int, bias_value: float, gate: float) -> SizeConditionedHead:
    base = build_base25(seed, bias_value)
    base.load_state_dict({k: v.clone() for k, v in base_init.items()})
    head = SizeConditionedHead(base, gate)
    if int(sum(p.numel() for p in head.parameters())) != TOTAL_PARAMETERS:
        raise RuntimeError("size-conditioned head parameter audit failed")
    return head


def make_schedule(n: int, seed: int) -> tuple[list[torch.Tensor], str]:
    generator = torch.Generator().manual_seed(int(seed))
    schedule = [torch.randperm(n, generator=generator) for _ in range(EPOCHS)]
    digest = hashlib.sha256()
    for order in schedule:
        digest.update(np.ascontiguousarray(order.numpy(), dtype=np.int64).tobytes())
    return schedule, digest.hexdigest()


def train_arm(
    arm: str,
    gate: float,
    T_fit: np.ndarray,
    size_fit: np.ndarray,
    c_fit: np.ndarray,
    base_init: Mapping[str, torch.Tensor],
    schedule: Sequence[torch.Tensor],
    bias_value: float,
    log: Any = print,
) -> dict[str, Any]:
    n = int(T_fit.shape[0])
    head = make_head(base_init, SEED, bias_value, gate)
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    init_hash = state_hash(init_state)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    T_t = torch.as_tensor(T_fit, dtype=torch.float32)
    S_t = torch.as_tensor(size_fit, dtype=torch.float32)
    c_t = torch.as_tensor(c_fit, dtype=torch.float32)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    steps = 0
    w_s_grad_step1 = None
    w_s_grad_step2 = None
    w_s_grad_last = None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        head.train()
        order = schedule[epoch - 1]
        abs_sum, grad_norm = 0.0, 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            prediction = head(T_t[idx], S_t[idx])
            err = (prediction - c_t[idx]).abs()
            loss = err.mean()
            optimizer.zero_grad()
            loss.backward()
            if steps == 0:
                w_s_grad_step1 = float(head.w_s.weight.grad.norm()) if head.w_s.weight.grad is not None else 0.0
            if steps == 1:
                w_s_grad_step2 = float(head.w_s.weight.grad.norm()) if head.w_s.weight.grad is not None else 0.0
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP))
            optimizer.step()
            abs_sum += float(err.sum())
            steps += 1
        if epoch >= EPOCHS - SOUP_LAST + 1:
            soup[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}
        curve.append({"epoch": int(epoch), "train_task_l1": abs_sum / n, "grad_norm": grad_norm,
                      "w_s_grad_norm": w_s_grad_step1 if epoch == 1 else None,
                      "seconds": float(time.perf_counter() - started)})
        w_s_grad_last = float(head.w_s.weight.grad.norm()) if head.w_s.weight.grad is not None else 0.0
    soup_mean = {key: torch.stack([soup[e][key] for e in sorted(soup)], dim=0).mean(dim=0) for key in soup[EPOCHS]}
    last_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(soup_mean)
    head.eval()
    with torch.no_grad():
        q_fit = head(T_t, S_t).numpy().astype(np.float64)
    w_s_final = float(head.w_s.weight.detach().norm())
    w_s_init = float(init_state["w_s.weight"].norm())
    log(f"[{arm}] steps={steps} seconds={time.perf_counter() - started:.1f} "
        f"w_s_grad_step1={w_s_grad_step1} w_s_final_norm={w_s_final}")
    return {
        "arm": arm,
        "gate": float(gate),
        "head": head,
        "init_state": init_state,
        "init_hash": init_hash,
        "soup_mean": soup_mean,
        "last_state": last_state,
        "soup_hash": state_hash(soup_mean),
        "last_hash": state_hash(last_state),
        "curve": curve,
        "steps": int(steps),
        "soup_members": sorted(soup),
        "q_fit": q_fit,
        "w_s_grad_step1": w_s_grad_step1,
        "w_s_grad_step2": w_s_grad_step2,
        "w_s_grad_last": w_s_grad_last,
        "w_s_final_norm": w_s_final,
        "w_s_init_norm": w_s_init,
        "w_s_delta_norm": float((head.w_s.weight.detach() - init_state["w_s.weight"]).norm()),
        "seconds": float(time.perf_counter() - started),
    }


def smoke_head(base_init, T_fit, size_fit, c_fit, bias_value, log) -> dict[str, Any]:
    head = make_head(base_init, SEED, bias_value, 1.0)
    before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    snapshot_w_s = before["w_s.weight"].clone()
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    T_t = torch.as_tensor(T_fit[:512], dtype=torch.float32)
    S_t = torch.as_tensor(size_fit[:512], dtype=torch.float32)
    c_t = torch.as_tensor(c_fit[:512], dtype=torch.float32)
    grads = {}
    for step in range(1, 5):
        optimizer.zero_grad()
        pred = head(T_t, S_t)
        loss = (pred - c_t).abs().mean()
        loss.backward()
        grads[f"step{step}"] = {
            name: float(p.grad.norm()) if p.grad is not None else None
            for name, p in head.named_parameters()
        }
        optimizer.step()
    changed = {k: bool(not torch.equal(head.state_dict()[k], before[k])) for k in before}
    out = {
        "steps": 4,
        "grad_norms": grads,
        "all_params_changed": bool(all(changed.values())),
        "w_s_changed": bool(not torch.equal(head.w_s.weight.detach(), snapshot_w_s)),
        "w_s_grad_step1": grads["step1"]["w_s.weight"],
        "base_grads_step1_positive": bool(all(
            (grads["step1"][n] or 0.0) > 0 for n in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
        )),
        "ok": bool(all(changed.values())),
    }
    log(f"[smoke] ok={out['ok']} w_s_grad_step1={out['w_s_grad_step1']}")
    return out


def verify_h_raw(blob, train, fold: Mapping[str, Any], log) -> dict[str, Any]:
    fit_idx, dev_idx = fold["fit_idx"], fold["dev_idx"]
    model = zftd.build_full_model(blob, SEED)
    state = torch.load(O_SOUP_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model_hash = state_hash(model.state_dict())
    targets = torch.zeros(10000)
    results = {}
    for name, idx in (("fit", fit_idx), ("dev", dev_idx)):
        data = [train[int(i)] for i in idx.tolist()]
        preds = np.empty(idx.size, np.float64)
        with torch.no_grad():
            for start in range(0, len(data), FWD_BATCH):
                local = list(range(start, min(start + FWD_BATCH, len(data))))
                batch = zftd.make_batch(data, local, targets, torch.device("cpu"))
                out = model(batch, mask=cm.C6_MASK).view(-1).double().numpy()
                preds[start:start + len(local)] = out
        cached = fold["h_fit"] if name == "fit" else fold["h_dev"]
        results[name] = {
            "n": int(idx.size),
            "max_abs_diff": float(np.max(np.abs(preds - cached))),
            "mean_abs_diff": float(np.mean(np.abs(preds - cached))),
            "ok": bool(np.max(np.abs(preds - cached)) <= REPLAY_TOL),
        }
    # fixed dev batch
    dev_data = [train[int(i)] for i in dev_idx.tolist()]
    batch = zftd.make_batch(dev_data, list(range(FIXED_DEV_BATCH)), targets, torch.device("cpu"))
    with torch.no_grad():
        fixed = model(batch, mask=cm.C6_MASK).view(-1).double().numpy()
    fixed_diff = float(np.max(np.abs(fixed - fold["h_dev"][:FIXED_DEV_BATCH])))
    out = {
        "model_state_hash": model_hash,
        "fit": results["fit"],
        "dev": results["dev"],
        "fixed_dev_batch": FIXED_DEV_BATCH,
        "fixed_dev_batch_max_abs_diff": fixed_diff,
        "tolerance": REPLAY_TOL,
        "ok": bool(results["fit"]["ok"] and results["dev"]["ok"] and fixed_diff <= REPLAY_TOL),
    }
    log(f"[h_verify] fit_max={out['fit']['max_abs_diff']:.3e} dev_max={out['dev']['max_abs_diff']:.3e} "
        f"fixed={fixed_diff:.3e} ok={out['ok']}")
    return out


# ---------------------------------------------------------------------------
# stage B — metrics
# ---------------------------------------------------------------------------


def metric_table(residual: np.ndarray, k: np.ndarray, n_total: int) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(k.shape[0]), "mae": float(np.mean(np.abs(residual)))}
    for key, mask in group_masks(k).items():
        if key == "k<=-2":
            continue
        out[key] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(residual[mask]))) if int(mask.sum()) else None,
            "contribution": float(np.abs(residual[mask]).sum() / n_total),
        }
    return out


def bootstrap_gain(res_a: np.ndarray, res_b: np.ndarray, groups: Sequence[np.ndarray] | None,
                   *, seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> dict[str, Any]:
    """gain = mean|a| - mean|b| (positive = b better), shared resample indices."""
    rng = np.random.default_rng(int(seed))
    point = float(np.abs(res_a).mean() - np.abs(res_b).mean())
    gains = np.empty(int(n_boot), np.float64)
    if groups is None:
        n = res_a.size
        for b in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            gains[b] = np.abs(res_a[idx]).mean() - np.abs(res_b[idx]).mean()
    else:
        rows_list = [np.where(g)[0] for g in groups]
        for b in range(int(n_boot)):
            idx = np.concatenate([rng.choice(rows, size=rows.size, replace=True) for rows in rows_list])
            gains[b] = np.abs(res_a[idx]).mean() - np.abs(res_b[idx]).mean()
    return {
        "point": point,
        "ci95": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
        "n_boot": int(n_boot),
        "seed": int(seed),
        "gains": gains,
    }


def paired_bootstrap_self_tests(res_a: np.ndarray, res_b: np.ndarray) -> dict[str, Any]:
    same = bootstrap_gain(res_a, res_a, None)
    swap = bootstrap_gain(res_b, res_a, None)
    fwd = bootstrap_gain(res_a, res_b, None)
    delta = 0.25
    shifted = bootstrap_gain(res_a, res_b + delta, None)
    mirror_max = float(np.max(np.abs(swap["gains"] + fwd["gains"])))
    shift_bound = float(np.max(np.abs(shifted["gains"] - fwd["gains"])))
    return {
        "identical_point": same["point"],
        "identical_ci": same["ci95"],
        "identical_all_zero": bool(np.all(same["gains"] == 0)),
        "swap_mirror_max_abs": mirror_max,
        "constant_shift_delta": delta,
        "constant_shift_observed_max_change": shift_bound,
        "constant_shift_bound_ok": bool(shift_bound <= delta + 1e-12),
        "ok": bool(np.all(same["gains"] == 0) and mirror_max <= 1e-12 and shift_bound <= delta + 1e-12),
    }


def size_gate_diagnostic(head: SizeConditionedHead, T_dev: np.ndarray, size_dev: np.ndarray,
                         y_dev: np.ndarray, h_dev: np.ndarray, b: float, k_dev: np.ndarray) -> dict[str, Any]:
    T_t = torch.as_tensor(T_dev, dtype=torch.float32)
    S_t = torch.as_tensor(size_dev, dtype=torch.float32)
    head.eval()
    with torch.no_grad():
        q_on = head(T_t, S_t).numpy().astype(np.float64)
        old_gate = head.gate
        head.gate = 0.0
        q_off = head(T_t, S_t).numpy().astype(np.float64)
        head.gate = old_gate
    delta = q_on - q_off
    p_on = h_dev + q_on + b
    p_off = h_dev + q_off + b
    out = {
        "delta_pred": {
            "mean_abs": float(np.mean(np.abs(delta))),
            "p95_abs": float(np.percentile(np.abs(delta), 95)),
            "max_abs": float(np.max(np.abs(delta))),
        },
        "overall_mae_gate_on": float(np.mean(np.abs(y_dev - p_on))),
        "overall_mae_gate_off": float(np.mean(np.abs(y_dev - p_off))),
        "group_mae": {},
    }
    for name, mask in group_masks(k_dev).items():
        if name == "k<=-2":
            continue
        out["group_mae"][name] = {
            "n": int(mask.sum()),
            "mae_gate_on": float(np.mean(np.abs(y_dev - p_on)[mask])) if int(mask.sum()) else None,
            "mae_gate_off": float(np.mean(np.abs(y_dev - p_off)[mask])) if int(mask.sum()) else None,
        }
    return out


# ---------------------------------------------------------------------------
# stage B — driver
# ---------------------------------------------------------------------------


def write_tables(out: Mapping[str, Any], fold: Mapping[str, Any], c25: Mapping[str, Any],
                 c27: Mapping[str, Any], size_data: Mapping[str, Any], scaler: Mapping[str, float]) -> None:
    fit_idx, dev_idx = fold["fit_idx"], fold["dev_idx"]
    k_all, y_all, c_all = fold["k_all"], fold["y_all"], fold["c_all"]
    groups = ("k=0", "k=-1", "k=-2", "k<=-3")
    for split in ("fit", "dev"):
        idx = fit_idx if split == "fit" else dev_idx
        n_total = int(idx.size)
        c = c_all[idx]
        k = k_all[idx]
        y = y_all[idx]
        h = fold["h_fit"] if split == "fit" else fold["h_dev"]
        main_rows = []
        group_rows = []
        for arm in ("Q0", "QNE"):
            a = out["arms"][arm]
            q = a["q_fit"] if split == "fit" else a["q_dev"]
            b = a["b_P"]
            targets = {
                "q_vs_c": q - c,
                "P_raw": y - (h + q),
                "P_cal": y - (h + q + b),
            }
            for target, resid in targets.items():
                mt = metric_table(resid, k, n_total)
                main_rows.append({"arm": arm, "split": split, "target": target,
                                  "overall_n": n_total, "overall_mae": mt["mae"], "b_P": b})
                for gname in groups:
                    group_rows.append({"arm": arm, "split": split, "target": target, "group": gname,
                                       "n": mt[gname]["n"], "mae": mt[gname]["mae"],
                                       "contribution": mt[gname]["contribution"]})
        if split == "fit":
            write_csv(RESULTS_DIR / "main_table.csv", main_rows)
            write_csv(RESULTS_DIR / "group_table.csv", group_rows)
        else:
            write_csv(RESULTS_DIR / "main_table_dev.csv", main_rows)
            write_csv(RESULTS_DIR / "group_table_dev.csv", group_rows)
        # per-row complete budget
        nrows = []
        for i in range(n_total):
            row = int(idx[i])
            q0 = out["arms"]["Q0"]["q_fit" if split == "fit" else "q_dev"][i]
            qne = out["arms"]["QNE"]["q_fit" if split == "fit" else "q_dev"][i]
            b0 = out["arms"]["Q0"]["b_P"]
            bne = out["arms"]["QNE"]["b_P"]
            nrows.append({
                "position": i, "stable_id": stable_id(row), "k": int(k[i]), "group": group_name(int(k[i])),
                "c": float(c[i]), "y": float(y[i]), "h_raw": float(h[i]),
                "q_Q0": float(q0), "q_QNE": float(qne),
                "q_err_Q0": float(q0 - c[i]), "q_err_QNE": float(qne - c[i]),
                "P_cal_Q0": float(h[i] + q0 + b0), "P_cal_QNE": float(h[i] + qne + bne),
                "err_P_cal_Q0": float(y[i] - (h[i] + q0 + b0)),
                "err_P_cal_QNE": float(y[i] - (h[i] + qne + bne)),
            })
        write_csv(RESULTS_DIR / ("per_graph_fit.csv" if split == "fit" else "per_graph_dev.csv"), nrows)
    # gain table (Q0 - QNE, positive = QNE better)
    gain_rows = []
    for split in ("fit", "dev"):
        idx = fit_idx if split == "fit" else dev_idx
        n_total = int(idx.size)
        c = c_all[idx]
        k = k_all[idx]
        y = y_all[idx]
        h = fold["h_fit"] if split == "fit" else fold["h_dev"]
        for target in ("q_vs_c", "P_raw", "P_cal"):
            masks = {"overall": np.ones(n_total, dtype=bool), **{g: m for g, m in group_masks(k).items() if g != "k<=-2"}}
            for gname, mask in masks.items():
                maes = {}
                for arm in ("Q0", "QNE"):
                    a = out["arms"][arm]
                    q = a["q_fit"] if split == "fit" else a["q_dev"]
                    b = a["b_P"]
                    resid = {"q_vs_c": q - c, "P_raw": y - (h + q), "P_cal": y - (h + q + b)}[target]
                    maes[arm] = float(np.mean(np.abs(resid[mask])))
                gain_rows.append({"split": split, "target": target, "group": gname, "n": int(mask.sum()),
                                  "Q0_mae": maes["Q0"], "QNE_mae": maes["QNE"],
                                  "gain_Q0_minus_QNE": maes["Q0"] - maes["QNE"]})
    write_csv(RESULTS_DIR / "gain_table.csv", gain_rows)


def online_size_features(raw_row: int, size_data: Mapping[str, Any], scaler: Mapping[str, float]) -> tuple[int, int, np.ndarray]:
    node_ptr, edge_ptr = size_data["node_ptr"], size_data["edge_ptr"]
    u, v = size_data["edge_u"], size_data["edge_v"]
    n = int(node_ptr[int(raw_row) + 1] - node_ptr[int(raw_row)])
    s, e = int(edge_ptr[int(raw_row)]), int(edge_ptr[int(raw_row) + 1])
    pairs = set()
    for a, b in zip(u[s:e].tolist(), v[s:e].tolist()):
        pairs.add((min(a, b), max(a, b)))
    E = len(pairs)
    sz = np.array([
        (n - scaler["n_mean"]) / scaler["n_std"],
        (E - scaler["e_mean"]) / scaler["e_std"],
    ], np.float32)
    return n, E, sz


def wrapper_checks(blob: Mapping[str, Any], train: Sequence[Any], fold: Mapping[str, Any],
                   scaler: Mapping[str, float], heads: Mapping[str, nn.Module],
                   size_data: Mapping[str, Any], arms_out: Mapping[str, Any],
                   log: Any = print) -> dict[str, Any]:
    dev_idx = fold["dev_idx"]
    T = fold["T"]
    model = zftd.build_full_model(blob, SEED)
    model.load_state_dict(torch.load(O_SOUP_PATH, map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head = heads["QNE"]
    head.eval()
    # online size for all dev rows vs stored X27 rows
    online = np.stack([online_size_features(int(r), size_data, scaler)[2] for r in dev_idx.tolist()], axis=0)
    size_all = standardize_size(np.asarray(size_data["N"]), np.asarray(size_data["E"]), scaler)
    stored = size_all[dev_idx]
    size_diff = float(np.max(np.abs(online - stored)))
    # fixed dev batch prediction through the deployable path (graph + T25 + frozen h + fit-only scaler/bias)
    dev_data = [train[int(i)] for i in dev_idx.tolist()]
    targets = torch.zeros(10000)
    batch = zftd.make_batch(dev_data, list(range(FIXED_DEV_BATCH)), targets, torch.device("cpu"))
    T_t = torch.as_tensor(T[dev_idx][:FIXED_DEV_BATCH], dtype=torch.float32)
    S_t = torch.as_tensor(online[:FIXED_DEV_BATCH], dtype=torch.float32)
    b_ne = float(arms_out["QNE"]["b_P"])

    def predict(batch_in, t25, size2, labels=None):  # labels deliberately ignored
        h = model(batch_in, mask=cm.C6_MASK).view(-1).double()
        q = head(t25, size2).double()
        return (h + q + b_ne).numpy()

    with torch.no_grad():
        p1 = predict(batch, T_t, S_t)
        p2 = predict(batch, T_t, S_t)
        rng = np.random.default_rng(INPUT_GEN_SEED)
        labels = (fold["y_all"][rng.permutation(10000)],
                  fold["c_all"][rng.permutation(10000)],
                  fold["k_all"][rng.permutation(10000)])
        p3 = predict(batch, T_t, S_t, labels=labels)
    cached = fold["h_dev"][:FIXED_DEV_BATCH] + np.asarray(arms_out["QNE"]["q_dev"], np.float64)[:FIXED_DEV_BATCH] + b_ne
    out = {
        "online_size_vs_stored_max_abs": size_diff,
        "online_size_ok": bool(size_diff == 0.0),
        "wrapper_vs_cached_max_abs": float(np.max(np.abs(p1 - cached))),
        "wrapper_equals_cached": bool(np.max(np.abs(p1 - cached)) <= REPLAY_TOL),
        "deterministic_repeat_max_abs": float(np.max(np.abs(p1 - p2))),
        "label_permutation_max_abs": float(np.max(np.abs(p1 - p3))),
        "label_permutation_invariant": bool(np.array_equal(p1, p3)),
        "wrapper_receives_labels": False,
        "n_permuted_labels": 3,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    out["ok"] = bool(out["online_size_ok"] and out["wrapper_equals_cached"] and
                     out["deterministic_repeat_max_abs"] == 0.0 and out["label_permutation_invariant"])
    log(f"[wrapper] ok={out['ok']} cached_diff={out['wrapper_vs_cached_max_abs']:.3e} "
        f"size_diff={size_diff:.3e}")
    return out


def run_stage_b(fold: Mapping[str, Any], size_data: Mapping[str, Any], scaler: Mapping[str, float],
                c25: Mapping[str, Any], c27: Mapping[str, Any], X25: np.ndarray, X27: np.ndarray,
                log: Any = print) -> dict[str, Any]:
    fit_idx, dev_idx = fold["fit_idx"], fold["dev_idx"]
    T = fold["T"]
    N, E = size_data["N"], size_data["E"]
    y_all, c_all, k_all = fold["y_all"], fold["c_all"], fold["k_all"]
    h_fit, h_dev = fold["h_fit"], fold["h_dev"]
    y_fit, y_dev = fold["y_fit"], fold["y_dev"]
    c_fit, c_dev = fold["c_fit"], fold["c_dev"]
    k_fit, k_dev = k_all[fit_idx], k_all[dev_idx]
    T_fit, T_dev = T[fit_idx], T[dev_idx]
    size_fit, size_dev = X27[fit_idx, 25:], X27[dev_idx, 25:]
    bias_value = float(np.median(c_fit))
    out: dict[str, Any] = {"bias_value_median_fit_c": bias_value}

    blob = zftd.load_prep_blob()
    train = zftd.load_train_only()
    zftd.apply_prep_train_only(train, blob)
    prep_topology = np.concatenate([d.topology_features.reshape(1, -1).numpy().astype(np.float32) for d in train], axis=0)
    log(f"[stageB] prep topology vs T25 maxdiff={float(np.max(np.abs(prep_topology - fold['T']))):.3e}")
    h_verify = verify_h_raw(blob, train, fold, log)
    out["h_verify"] = h_verify
    write_json(RESULTS_DIR / "h_verify.json", h_verify)

    base = build_base25(SEED, bias_value)
    base_init = {k: v.detach().clone() for k, v in base.state_dict().items()}
    base_init_hash = state_hash(base_init)
    out["base_init_hash"] = base_init_hash
    # published P_U init equality
    pu_init = torch.load(PU_INIT_PATH, map_location="cpu")
    pu_keys = ["0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"]
    init_match = {k: bool(torch.equal(base_init[k], pu_init[k])) for k in pu_keys}
    out["P_U_init_tensor_matches"] = init_match

    schedule, schedule_hash = make_schedule(int(fit_idx.size), TRAIN_GEN_BASE + SEED)
    out["schedule_hash"] = schedule_hash
    out["schedule_epochs"] = len(schedule)
    out["schedule_batches_per_epoch"] = int(np.ceil(fit_idx.size / BATCH_SIZE))
    out["schedule_last_batch_size"] = int(fit_idx.size - (int(np.ceil(fit_idx.size / BATCH_SIZE)) - 1) * BATCH_SIZE)

    smoke = smoke_head(base_init, T_fit, size_fit, c_fit, bias_value, log)
    out["smoke"] = smoke

    log("[stageB] training Q0 ...")
    res_q0 = train_arm("Q0", 0.0, T_fit, size_fit, c_fit, base_init, schedule, bias_value, log)
    log("[stageB] training QNE ...")
    res_qne = train_arm("QNE", 1.0, T_fit, size_fit, c_fit, base_init, schedule, bias_value, log)
    heads = {"Q0": res_q0["head"], "QNE": res_qne["head"]}
    out["arms"] = {}
    with torch.no_grad():
        T_dev_t = torch.as_tensor(T_dev, dtype=torch.float32)
        S_dev_t = torch.as_tensor(size_dev, dtype=torch.float32)
        T_fit_t = torch.as_tensor(T_fit, dtype=torch.float32)
        S_fit_t = torch.as_tensor(size_fit, dtype=torch.float32)
        q0_dev = res_q0["head"](T_dev_t, S_dev_t).numpy().astype(np.float64)
        qne_dev = res_qne["head"](T_dev_t, S_dev_t).numpy().astype(np.float64)
    for name, res, q_dev in (("Q0", res_q0, q0_dev), ("QNE", res_qne, qne_dev)):
        b_P = float(np.median(y_fit - h_fit - res["q_fit"]))
        p_raw_fit = h_fit + res["q_fit"]
        p_raw_dev = h_dev + q_dev
        p_cal_fit = p_raw_fit + b_P
        p_cal_dev = p_raw_dev + b_P
        entry = {
            "b_P": b_P,
            "fit": {
                "q_mae_c": metric_table(res["q_fit"] - c_fit, k_fit, int(fit_idx.size)),
                "P_raw": metric_table(y_fit - p_raw_fit, k_fit, int(fit_idx.size)),
                "P_cal": metric_table(y_fit - p_cal_fit, k_fit, int(fit_idx.size)),
            },
            "dev": {
                "q_mae_c": metric_table(q_dev - c_dev, k_dev, int(dev_idx.size)),
                "P_raw": metric_table(y_dev - p_raw_dev, k_dev, int(dev_idx.size)),
                "P_cal": metric_table(y_dev - p_cal_dev, k_dev, int(dev_idx.size)),
            },
            "q_fit_summary": {
                "mean": float(res["q_fit"].mean()), "std": float(res["q_fit"].std()),
                "min": float(res["q_fit"].min()), "max": float(res["q_fit"].max()),
                "slope_vs_c": float(np.polyfit(c_fit, res["q_fit"], 1)[0]) if np.std(c_fit) > 0 else None,
            },
            "q_dev_summary": {
                "mean": float(q_dev.mean()), "std": float(q_dev.std()),
                "min": float(q_dev.min()), "max": float(q_dev.max()),
                "slope_vs_c": float(np.polyfit(c_dev, q_dev, 1)[0]) if np.std(c_dev) > 0 else None,
            },
            "init_hash": res["init_hash"],
            "soup_hash": res["soup_hash"],
            "steps": res["steps"],
            "soup_members": res["soup_members"],
            "seconds": res["seconds"],
            "w_s_grad_step1": res["w_s_grad_step1"],
            "w_s_grad_step2": res["w_s_grad_step2"],
            "w_s_grad_last": res["w_s_grad_last"],
            "w_s_final_norm": res["w_s_final_norm"],
            "w_s_init_norm": res["w_s_init_norm"],
            "w_s_delta_norm": res["w_s_delta_norm"],
            "q_fit": res["q_fit"],
            "q_dev": q_dev,
            "p_raw_fit": p_raw_fit,
            "p_raw_dev": p_raw_dev,
            "p_cal_fit": p_cal_fit,
            "p_cal_dev": p_cal_dev,
        }
        # native25 replay of Q0 (drop W_S)
        if name == "Q0":
            native = build_base25(SEED, bias_value)
            native.load_state_dict(res["head"].native25_state())
            native.eval()
            with torch.no_grad():
                q_native_dev = native(T_dev_t).view(-1).numpy().astype(np.float64)
                q_native_fit = native(T_fit_t).view(-1).numpy().astype(np.float64)
            entry["native25_replay"] = {
                "fit_max_abs": float(np.max(np.abs(q_native_fit - res["q_fit"]))),
                "dev_max_abs": float(np.max(np.abs(q_native_dev - q_dev))),
                "state_hash": state_hash(res["head"].native25_state()),
            }
        out["arms"][name] = entry

    # Q0 vs published P_U / P_seed0
    pu = np.load(PU_PRED_PATH)
    ps0 = np.load(PS0_PRED_PATH)
    pu_q_dev = np.asarray(pu["q_dev"], np.float64)
    pu_q_fit = np.asarray(pu["q_fit"], np.float64)
    pu_meta = json.loads(PU_META_PATH.read_text())
    pu_b = float(pu_meta["b_P"] if "b_P" in pu_meta else np.nan)
    pu_analysis = json.loads(PU_ANALYSIS_PATH.read_text())
    pu_arm = pu_analysis["arms"]["P_U"]
    pu_b = float(pu_arm["b_P"])
    pu_dev_cal = float(pu_arm["P_cal_dev"]["mae"])
    ps0_b = float(np.asarray(ps0["b_P"]).item())
    repl = {
        "P_U_q_dev_maxdiff_vs_Q0": float(np.max(np.abs(pu_q_dev - out["arms"]["Q0"]["q_dev"]))),
        "P_U_q_fit_maxdiff_vs_Q0": float(np.max(np.abs(pu_q_fit - out["arms"]["Q0"]["q_fit"]))),
        "P_seed0_q_dev_maxdiff_vs_Q0": float(np.max(np.abs(np.asarray(ps0["dev_q"], np.float64) - out["arms"]["Q0"]["q_dev"]))),
        "P_seed0_q_fit_maxdiff_vs_Q0": float(np.max(np.abs(np.asarray(ps0["fit_q"], np.float64) - out["arms"]["Q0"]["q_fit"]))),
        "P_U_b_P": pu_b,
        "P_seed0_b_P": ps0_b,
        "Q0_b_P": out["arms"]["Q0"]["b_P"],
        "P_U_dev_cal_reported": pu_dev_cal,
        "Q0_dev_cal": out["arms"]["Q0"]["dev"]["P_cal"]["mae"],
        "P_U_dev_cal_abs_diff": float(abs(out["arms"]["Q0"]["dev"]["P_cal"]["mae"] - pu_dev_cal)),
        "P_U_dev_cal_matches": bool(abs(out["arms"]["Q0"]["dev"]["P_cal"]["mae"] - pu_dev_cal) <= 2e-6),
        "all_within_1e-6": bool(
            max(
                abs(float(np.max(np.abs(pu_q_dev - out["arms"]["Q0"]["q_dev"])))),
                abs(float(np.max(np.abs(pu_q_fit - out["arms"]["Q0"]["q_fit"])))),
                abs(float(np.max(np.abs(np.asarray(ps0["dev_q"], np.float64) - out["arms"]["Q0"]["q_dev"])))),
                abs(float(np.max(np.abs(np.asarray(ps0["fit_q"], np.float64) - out["arms"]["Q0"]["q_fit"])))),
            ) <= 1e-6
        ),
        "init_tensor_matches": init_match,
    }
    out["q0_replication"] = repl
    write_json(RESULTS_DIR / "q0_replication.json", repl)

    # gains and bootstrap
    q0, qne = out["arms"]["Q0"], out["arms"]["QNE"]
    res_q0_dev_cal = y_dev - q0["p_cal_dev"]
    res_qne_dev_cal = y_dev - qne["p_cal_dev"]
    res_q0_dev_raw = y_dev - q0["p_raw_dev"]
    res_qne_dev_raw = y_dev - qne["p_raw_dev"]
    g0_mask = k_dev == 0
    severity_groups = [k_dev == 0, k_dev == -1, k_dev == -2, k_dev <= -3]
    boot = {
        "main_cal": bootstrap_gain(res_q0_dev_cal, res_qne_dev_cal, None),
        "main_raw": bootstrap_gain(res_q0_dev_raw, res_qne_dev_raw, None),
        "g0_cal": bootstrap_gain(res_q0_dev_cal[g0_mask], res_qne_dev_cal[g0_mask], None),
        "severity_stratified_cal": bootstrap_gain(res_q0_dev_cal, res_qne_dev_cal, severity_groups),
        "self_tests": paired_bootstrap_self_tests(res_q0_dev_cal, res_qne_dev_cal),
    }
    out["bootstrap"] = boot
    write_json(RESULTS_DIR / "bootstrap.json", {
        "main_cal": {k: v for k, v in boot["main_cal"].items() if k != "gains"},
        "main_raw": {k: v for k, v in boot["main_raw"].items() if k != "gains"},
        "g0_cal": {k: v for k, v in boot["g0_cal"].items() if k != "gains"},
        "severity_stratified_cal": {k: v for k, v in boot["severity_stratified_cal"].items() if k != "gains"},
        "self_tests": boot["self_tests"],
    })

    gain_cal = boot["main_cal"]
    gain_raw = boot["main_raw"]
    g0_worsening = float(
        np.mean(np.abs(res_qne_dev_cal[g0_mask])) - np.mean(np.abs(res_q0_dev_cal[g0_mask]))
    )
    gate = {
        "dev_overall_cal_gain": gain_cal["point"],
        "dev_overall_cal_ci": gain_cal["ci95"],
        "dev_overall_raw_gain": gain_raw["point"],
        "dev_G0_cal_worsening": g0_worsening,
        "conditions": {
            "cal_gain_ge_0.003": bool(gain_cal["point"] >= DELTA),
            "cal_ci_lower_gt_0": bool(gain_cal["ci95"][0] > 0.0),
            "raw_gain_gt_0": bool(gain_raw["point"] > 0.0),
            "G0_worsening_le_0.001": bool(g0_worsening <= G0_TOL),
        },
    }
    gate["passed"] = bool(all(gate["conditions"].values()) and h_verify["ok"] and repl["all_within_1e-6"] and smoke["ok"])
    out["performance_gate"] = gate
    write_json(RESULTS_DIR / "performance_gate.json", gate)

    # per-row tables
    per_graph = []
    for i in range(int(dev_idx.size)):
        row = int(dev_idx[i])
        per_graph.append({
            "position": i,
            "stable_id": stable_id(row),
            "k": int(k_dev[i]),
            "group": group_name(int(k_dev[i])),
            "c": float(c_dev[i]),
            "y": float(y_dev[i]),
            "h_raw": float(h_dev[i]),
            "q_Q0": float(q0["q_dev"][i]),
            "q_QNE": float(qne["q_dev"][i]),
            "P_cal_Q0": float(q0["p_cal_dev"][i]),
            "P_cal_QNE": float(qne["p_cal_dev"][i]),
            "err_P_cal_Q0": float(y_dev[i] - q0["p_cal_dev"][i]),
            "err_P_cal_QNE": float(y_dev[i] - qne["p_cal_dev"][i]),
        })
    write_csv(RESULTS_DIR / "per_graph_dev.csv", per_graph)

    # sensitivity: drop Q0's max-error dev row
    worst = int(np.argmax(np.abs(res_q0_dev_cal)))
    keep = np.ones(dev_idx.size, dtype=bool)
    keep[worst] = False
    sens = {
        "drop_Q0_max_error_row": {
            "stable_id": stable_id(int(dev_idx[worst])),
            "k": int(k_dev[worst]),
            "Q0_abs_err": float(abs(res_q0_dev_cal[worst])),
            "gain_cal_drop_row": float(np.mean(np.abs(res_q0_dev_cal[keep])) - np.mean(np.abs(res_qne_dev_cal[keep]))),
            "gain_raw_drop_row": float(np.mean(np.abs(res_q0_dev_raw[keep])) - np.mean(np.abs(res_qne_dev_raw[keep]))),
        }
    }
    out["sensitivity"] = sens

    # size diagnostics
    size_diag = {
        "QNE": {
            "w_s_final_norm": qne["w_s_final_norm"],
            "w_s_delta_norm": qne["w_s_delta_norm"],
            "w_s_grad_step1": qne["w_s_grad_step1"],
            "w_s_grad_step2": qne["w_s_grad_step2"],
            "w_s_grad_last": qne["w_s_grad_last"],
            "gate_on_diagnostic": size_gate_diagnostic(heads["QNE"], T_dev, size_dev, y_dev, h_dev, qne["b_P"], k_dev),
        },
        "Q0": {
            "w_s_final_norm": q0["w_s_final_norm"],
            "w_s_delta_norm": q0["w_s_delta_norm"],
            "w_s_grad_step1": q0["w_s_grad_step1"],
            "w_s_zero_kept": bool(q0["w_s_final_norm"] == 0.0),
        },
    }
    out["size_diagnostics"] = size_diag

    # fit tail rows q errors
    tail_fit = np.where(k_fit <= -3)[0]
    q0_fit_tail = q0["q_fit"][tail_fit]
    qne_fit_tail = qne["q_fit"][tail_fit]
    c_tail = c_fit[tail_fit]
    tail_rows = []
    for j, r in enumerate(tail_fit.tolist()):
        row = int(fit_idx[r])
        e0 = abs(float(q0_fit_tail[j] - c_tail[j]))
        e1 = abs(float(qne_fit_tail[j] - c_tail[j]))
        tail_rows.append({
            "fit_position": int(r),
            "stable_id": stable_id(row),
            "k": int(k_fit[r]),
            "c": float(c_tail[j]),
            "q_Q0": float(q0_fit_tail[j]),
            "q_QNE": float(qne_fit_tail[j]),
            "q_err_Q0": e0,
            "q_err_QNE": e1,
            "relative_drop": (e0 - e1) / e0 if e0 > 0 else None,
            "was_in_conflict_class_X25": bool(c25["conflict"][c25["class_id"][r]]),
            "new_class_id_X27": int(c27["class_id"][r]),
            "new_class_c_span_X27": float(c27["class_spread"][c27["class_id"][r]]),
        })
    agg0 = float(np.mean([r["q_err_Q0"] for r in tail_rows]))
    agg1 = float(np.mean([r["q_err_QNE"] for r in tail_rows]))
    out["fit_tail"] = {
        "rows": tail_rows,
        "q_mae_Q0": agg0,
        "q_mae_QNE": agg1,
        "relative_drop": (agg0 - agg1) / agg0 if agg0 > 0 else None,
        "marker_25pct": bool(agg0 > 0 and (agg0 - agg1) / agg0 >= 0.25),
        "conflict_tail_rows_improved": [
            {"stable_id": r["stable_id"], "k": r["k"], "q_err_Q0": r["q_err_Q0"], "q_err_QNE": r["q_err_QNE"],
             "improved": bool(r["q_err_QNE"] < r["q_err_Q0"])}
            for r in tail_rows if r["was_in_conflict_class_X25"]
        ],
    }
    write_json(RESULTS_DIR / "fit_tail_q_errors.json", out["fit_tail"])
    write_json(RESULTS_DIR / "sensitivity.json", sens)
    write_json(RESULTS_DIR / "size_diagnostics.json", size_diag)
    write_tables(out, fold, c25, c27, size_data, scaler)
    # conflict-row prediction changes on fit
    conflict_rows_fit = np.where(c25["conflict"][c25["class_id"]])[0]
    conflict_changes = []
    for r in conflict_rows_fit.tolist():
        row = int(fit_idx[r])
        conflict_changes.append({
            "fit_position": int(r), "stable_id": stable_id(row), "k": int(k_fit[r]), "c": float(c_fit[r]),
            "q_Q0": float(out["arms"]["Q0"]["q_fit"][r]), "q_QNE": float(out["arms"]["QNE"]["q_fit"][r]),
            "delta_q": float(out["arms"]["QNE"]["q_fit"][r] - out["arms"]["Q0"]["q_fit"][r]),
            "P_cal_Q0": float(out["arms"]["Q0"]["p_cal_fit"][r]),
            "P_cal_QNE": float(out["arms"]["QNE"]["p_cal_fit"][r]),
            "err_P_cal_Q0": float(y_fit[r] - out["arms"]["Q0"]["p_cal_fit"][r]),
            "err_P_cal_QNE": float(y_fit[r] - out["arms"]["QNE"]["p_cal_fit"][r]),
        })
    write_json(RESULTS_DIR / "stage_b_analysis.json", {
        "bias_value_median_fit_c": bias_value,
        "schedule_hash": schedule_hash,
        "base_init_hash": base_init_hash,
        "base_init_matches_P_U": init_match,
        "arms": {name: {
            "b_P": out["arms"][name]["b_P"],
            "q_fit_summary": out["arms"][name]["q_fit_summary"],
            "q_dev_summary": out["arms"][name]["q_dev_summary"],
            "fit": out["arms"][name]["fit"],
            "dev": out["arms"][name]["dev"],
            "steps": out["arms"][name]["steps"],
            "soup_members": out["arms"][name]["soup_members"],
            "seconds": out["arms"][name]["seconds"],
            "w_s_final_norm": out["arms"][name]["w_s_final_norm"],
            "w_s_delta_norm": out["arms"][name]["w_s_delta_norm"],
            "w_s_grad_step1": out["arms"][name]["w_s_grad_step1"],
            "w_s_grad_step2": out["arms"][name]["w_s_grad_step2"],
            "w_s_grad_last": out["arms"][name]["w_s_grad_last"],
            "native25_replay": out["arms"][name].get("native25_replay"),
        } for name in ("Q0", "QNE")},
        "fit_conflict_rows": conflict_changes,
        "sensitivity": sens,
        "size_diagnostics": size_diag,
        "q0_replication": repl,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    })
    dev_tail_rows = []
    for i in np.where(k_dev <= SEVERE_MAX)[0].tolist():
        row = int(dev_idx[i])
        c = float(c_dev[i]); y = float(y_dev[i]); h = float(h_dev[i])
        dev_tail_rows.append({
            "stable_id": stable_id(row), "k": int(k_dev[i]), "c": c, "y": y, "h_raw": h,
            "g_minus_h_abs_before_cycle": abs((y - c) - h),
            "q_Q0": float(out["arms"]["Q0"]["q_dev"][i]),
            "q_QNE": float(out["arms"]["QNE"]["q_dev"][i]),
            "q_err_Q0": abs(float(out["arms"]["Q0"]["q_dev"][i]) - c),
            "q_err_QNE": abs(float(out["arms"]["QNE"]["q_dev"][i]) - c),
            "P_cal_Q0": float(out["arms"]["Q0"]["p_cal_dev"][i]),
            "P_cal_QNE": float(out["arms"]["QNE"]["p_cal_dev"][i]),
            "err_P_cal_Q0": abs(float(y - out["arms"]["Q0"]["p_cal_dev"][i])),
            "err_P_cal_QNE": abs(float(y - out["arms"]["QNE"]["p_cal_dev"][i])),
        })
    write_csv(RESULTS_DIR / "dev_tail_rows.csv", dev_tail_rows)
    wrapper = wrapper_checks(blob, train, fold, scaler, heads, size_data, out["arms"], log)
    out["wrapper_checks"] = wrapper
    write_json(RESULTS_DIR / "wrapper_checks.json", wrapper)

    # save artifacts
    for name, res in (("Q0", res_q0), ("QNE", res_qne)):
        torch.save(res["init_state"], RESULTS_DIR / f"{name}_init_state.pt")
        torch.save(res["last_state"], RESULTS_DIR / f"{name}_last_state.pt")
        torch.save(res["soup_mean"], RESULTS_DIR / f"{name}_soup_state.pt")
        np.savez_compressed(
            RESULTS_DIR / f"{name}_predictions.npz",
            q_fit=res["q_fit"], q_dev=out["arms"][name]["q_dev"],
            b_P=np.array([out["arms"][name]["b_P"]]),
            fit_idx=fit_idx, dev_idx=dev_idx,
        )
        write_json(RESULTS_DIR / f"{name}.json", {
            "protocol_version": PROTOCOL_VERSION,
            "arm": name,
            "gate": res["gate"],
            "seed": SEED,
            "epochs": EPOCHS,
            "steps": res["steps"],
            "soup_members": res["soup_members"],
            "schedule_hash": schedule_hash,
            "init_hash": res["init_hash"],
            "soup_hash": res["soup_hash"],
            "b_P": out["arms"][name]["b_P"],
            "seconds": res["seconds"],
            "curve": res["curve"],
            "w_s_grad_step1": res["w_s_grad_step1"],
            "w_s_grad_step2": res["w_s_grad_step2"],
            "w_s_grad_last": res["w_s_grad_last"],
            "parameter_count": TOTAL_PARAMETERS,
        })
    # native25 replay file for Q0 (projection dropping W_S)
    native_state = res_q0["head"].native25_state()
    torch.save(native_state, RESULTS_DIR / "Q0_native25_state.pt")
    np.savez_compressed(
        RESULTS_DIR / "schedule.npz",
        schedule=np.stack([s.numpy().astype(np.int32) for s in schedule], axis=0),
        schedule_hash=np.array([schedule_hash]),
    )
    # bundle for the deployable wrapper
    bundle = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": "QNE",
        "seed": SEED,
        "head_state": res_qne["soup_mean"],
        "head_init_state": res_qne["init_state"],
        "gate": 1.0,
        "size_scaler": dict(scaler),
        "b_P": float(out["arms"]["QNE"]["b_P"]),
        "o_soup_state_path": str(O_SOUP_PATH.relative_to(TRACK_ROOT)),
        "note": "T25 = existing Full topology25 model input; size2 computed online from N and unique undirected pairs",
    }
    torch.save(bundle, RESULTS_DIR / "deploy_bundle.pt")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description="ZINC cycle head size-input repair (CPU, seed 0)")
    parser.add_argument("--stage", default="all", choices=("A", "B", "all"))
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = print

    fold = load_frozen_inputs(log)
    T, fit_idx, dev_idx = fold["T"], fold["fit_idx"], fold["dev_idx"]
    k_all, c_all, y_all = fold["k_all"], fold["c_all"], fold["y_all"]
    k_fit, k_dev = k_all[fit_idx], k_all[dev_idx]
    c_fit = c_all[fit_idx]

    size_data = compute_size_features(log)
    N, E = size_data["N"], size_data["E"]
    audit = size_data["audit"]
    # cross-checks against the frozen inputs
    audit["handoff_y_matches_decomp"] = bool(np.max(np.abs(size_data["y_handoff"] - y_all)) == 0.0)
    audit["handoff_topo_model_input_vs_T25_maxdiff"] = float(np.max(np.abs(size_data["topo_model_input"] - T)))
    # verify N against the encoded cache num_nodes
    import torch as _torch  # already imported; keep explicit for clarity

    train_graphs = zftd.load_train_only()
    n_nodes_encoded = np.array([int(d.num_nodes) for d in train_graphs], np.int64)
    audit["node_ptr_matches_num_nodes"] = bool(np.array_equal(n_nodes_encoded, np.diff(size_data["node_ptr"])))
    audit["all_ok"] = bool(audit["all_ok"] and audit["handoff_y_matches_decomp"] and audit["node_ptr_matches_num_nodes"])
    write_json(out_dir / "size_features_audit.json", audit)
    np.savez_compressed(out_dir / "size_features.npz", N=N, E=E, fit_idx=fit_idx, dev_idx=dev_idx)

    scaler = size_scaler(N, E, fit_idx)
    write_json(out_dir / "size_scaler.json", scaler)
    size_all = standardize_size(N, E, scaler)
    X25 = T
    X27 = np.concatenate([T, size_all], axis=1).astype(np.float32)
    # standardization merge check
    pairs = np.stack([N, E], axis=1)
    uniq_pairs = np.unique(pairs, axis=0)
    sz32 = np.unique(np.ascontiguousarray(size_all), axis=0)
    merge_check = {
        "distinct_int_pairs": int(uniq_pairs.shape[0]),
        "distinct_float32_standardized_pairs": int(sz32.shape[0]),
        "merged": bool(uniq_pairs.shape[0] != sz32.shape[0]),
        "negative_zero_in_X25": int(np.sum(np.signbit(X25) & (X25 == 0))),
        "negative_zero_in_X27_size": int(np.sum(np.signbit(size_all) & (size_all == 0))),
        "n_mean": scaler["n_mean"], "n_std": scaler["n_std"],
        "e_mean": scaler["e_mean"], "e_std": scaler["e_std"],
    }
    write_json(out_dir / "size_merge_check.json", merge_check)

    # ---- Phase A: full-fit conflict tables ----
    c25 = class_analysis(X25[fit_idx], c_fit, k_fit)
    c27 = class_analysis(X27[fit_idx], c_fit, k_fit)
    x25_json = {k: v for k, v in c25.items() if not isinstance(v, np.ndarray)}
    x27_json = {k: v for k, v in c27.items() if not isinstance(v, np.ndarray)}
    write_json(out_dir / "conflict_x25.json", x25_json)
    write_json(out_dir / "conflict_x27.json", x27_json)

    witnesses = json.loads(WITNESS_JSON.read_text())
    split, split_rows = conflict_split_table(c25, c27, N, E, fit_idx, c_all, k_all, T, X27, witnesses)
    write_json(out_dir / "original_conflict_split.json", split)
    write_csv(out_dir / "original_conflict_split.csv", split_rows)

    tail_struct = tail_rows_structure(c25, c27, N, E, fit_idx, c_all, k_all, T)
    write_json(out_dir / "tail_rows_structure.json", tail_struct)

    relation = size_vs_t25_relation(T, N, E, fit_idx, c25)

    # structure gate
    reduction = float(c25["global_min_l1_per_row"] - c27["global_min_l1_per_row"])
    tail_sep = []
    for r in np.where(k_fit <= -3)[0].tolist():
        was_conflict = bool(c25["conflict"][c25["class_id"][r]])
        span = float(c27["class_spread"][c27["class_id"][r]])
        tail_sep.append({
            "fit_position": int(r),
            "stable_id": stable_id(int(fit_idx[r])),
            "k": int(k_fit[r]),
            "was_in_conflict_class": was_conflict,
            "new_class_span": span,
            "span_le_1e-10": bool(span <= SIZE_GATE_GAP),
        })
    condition1 = bool(fold["checks"]["T25_ok"] and fold["checks"]["fit_idx_ok"] and fold["checks"]["dev_idx_ok"]
                      and fold["checks"]["O_pred_file_ok"] and fold["checks"]["O_soup_file_ok"]
                      and fold["checks"]["prep_blob_ok"] and fold["checks"]["fit_k_ok"] and fold["checks"]["dev_k_ok"]
                      and audit["all_ok"] and merge_check["merged"] is False
                      and not merge_check["negative_zero_in_X25"] and not merge_check["negative_zero_in_X27_size"])
    condition2 = bool(reduction >= 0.001)
    condition3 = bool(any(t["was_in_conflict_class"] and t["span_le_1e-10"] for t in tail_sep))
    structure_gate = {
        "protocol_version": PROTOCOL_VERSION,
        "input_identity_ok": condition1,
        "old_global_min_l1": float(c25["global_min_l1_per_row"]),
        "new_global_min_l1": float(c27["global_min_l1_per_row"]),
        "reduction_abs": reduction,
        "reduction_rel": reduction / float(c25["global_min_l1_per_row"]),
        "reduction_ge_0.001": condition2,
        "tail_row_separation": tail_sep,
        "tail_row_sep_ok": condition3,
        "passed": bool(condition1 and condition2 and condition3),
        "rule": {
            "identity_all_pass": True,
            "global_min_l1_reduction_ge": 0.001,
            "one_conflict_tail_row_new_span_le": SIZE_GATE_GAP,
        },
    }
    write_json(out_dir / "structure_gate.json", structure_gate)
    log(f"[gate] passed={structure_gate['passed']} reduction={reduction:.6f} "
        f"old_classes={c25['n_exact_classes']} new_classes={c27['n_exact_classes']}")

    # dev coverage (computed after the gate is frozen)
    coverage = {}
    for tag, X, cc in (("X25", X25, c25), ("X27", X27, c27)):
        fit_key_cls = {}
        for r in range(int(fit_idx.size)):
            fit_key_cls[bytes_key(X[fit_idx[r]])] = int(cc["class_id"][r])
        covered = np.zeros(int(dev_idx.size), dtype=bool)
        in_conflict = np.zeros(int(dev_idx.size), dtype=bool)
        for i in range(int(dev_idx.size)):
            key = bytes_key(X[dev_idx[i]])
            if key in fit_key_cls:
                covered[i] = True
                in_conflict[i] = bool(cc["conflict"][fit_key_cls[key]])
        coverage[tag] = {
            "n_dev": int(dev_idx.size),
            "covered": int(covered.sum()),
            "coverage_fraction": float(covered.mean()),
            "covered_in_conflict_class": int((covered & in_conflict).sum()),
            "uncovered": int((~covered).sum()),
            "uncovered_tail_rows": [
                {
                    "stable_id": stable_id(int(dev_idx[i])),
                    "k": int(k_dev[i]),
                    "c": float(c_all[dev_idx[i]]),
                }
                for i in np.where(~covered & (k_dev <= SEVERE_MAX))[0].tolist()
            ],
            "tail_rows": [
                {
                    "stable_id": stable_id(int(dev_idx[i])),
                    "k": int(k_dev[i]),
                    "covered": bool(covered[i]),
                    "covered_class_in_conflict": bool(in_conflict[i]) if covered[i] else None,
                }
                for i in np.where(k_dev <= SEVERE_MAX)[0].tolist()
            ],
        }
    write_json(out_dir / "dev_coverage.json", {"protocol_version": PROTOCOL_VERSION, "coverage": coverage})
    cov_rows = []
    for tag in coverage:
        for key in ("n_dev", "covered", "coverage_fraction", "covered_in_conflict_class", "uncovered"):
            cov_rows.append({"input": tag, "metric": key, "value": coverage[tag][key]})
    write_csv(out_dir / "dev_coverage_table.csv", cov_rows)

    write_json(out_dir / "identity_checks.json", {
        "frozen": fold["checks"],
        "size_audit": audit,
        "merge_check": merge_check,
        "relation": relation,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "all_ok": condition1,
    })
    write_json(out_dir / "size_vs_t25_relation.json", relation)

    if args.stage in ("A",):
        write_json(out_dir / "phase_a_summary.json", {
            "protocol_version": PROTOCOL_VERSION,
            "structure_gate": structure_gate,
            "x25": x25_json,
            "x27": x27_json,
            "coverage": coverage,
            "seconds": float(time.perf_counter() - t0),
        })
        log(f"[done] stage A in {time.perf_counter() - t0:.1f}s")
        return 0

    if not structure_gate["passed"]:
        decision = {
            "protocol_version": PROTOCOL_VERSION,
            "final_class": "INPUT_REPAIR_NOT_SUFFICIENT",
            "structure_gate": structure_gate,
            "trained": False,
            "reason": "pre-registered structure gate failed; no head is bought",
        }
        write_json(out_dir / "decision.json", decision)
        write_json(out_dir / "phase_a_summary.json", {
            "protocol_version": PROTOCOL_VERSION,
            "structure_gate": structure_gate,
            "x25": x25_json,
            "x27": x27_json,
            "coverage": coverage,
            "seconds": float(time.perf_counter() - t0),
        })
        write_manifest(out_dir)
        log("[done] structure gate failed; stop before training")
        return 0

    stage_b = run_stage_b(fold, size_data, scaler, c25, c27, X25, X27, log)

    # ---- final decision ----
    perf = stage_b["performance_gate"]
    fit_tail = stage_b["fit_tail"]
    if not stage_b["h_verify"]["ok"] or not stage_b["smoke"]["ok"] or not stage_b["q0_replication"]["all_within_1e-6"] or not stage_b["wrapper_checks"]["ok"]:
        final_class = "INVALID"
    elif perf["passed"]:
        final_class = "TRANSFER_SIGNAL"
    elif fit_tail["marker_25pct"]:
        final_class = "FIT_GAIN_NO_TRANSFER"
    else:
        final_class = "INPUT_REPAIRED_NOT_FIT"
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "final_class": final_class,
        "structure_gate": structure_gate,
        "performance_gate": perf,
        "fit_tail": {
            "q_mae_Q0": fit_tail["q_mae_Q0"],
            "q_mae_QNE": fit_tail["q_mae_QNE"],
            "relative_drop": fit_tail["relative_drop"],
            "marker_25pct": fit_tail["marker_25pct"],
            "conflict_tail_rows_improved": fit_tail["conflict_tail_rows_improved"],
        },
        "q0_replication": stage_b["q0_replication"],
        "h_verify_ok": stage_b["h_verify"]["ok"],
        "wrapper_checks": stage_b["wrapper_checks"],
        "bootstrap": stage_b["bootstrap"],
        "sensitivity": stage_b["sensitivity"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "decision.json", decision)
    write_json(out_dir / "phase_b_summary.json", {
        "protocol_version": PROTOCOL_VERSION,
        "structure_gate": structure_gate,
        "performance_gate": perf,
        "final_class": final_class,
        "seconds": float(time.perf_counter() - t0),
    })
    # manifest of produced files
    write_manifest(out_dir)
    log(f"[done] final_class={final_class} in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
