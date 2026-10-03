"""Part C index fix — correct the per-graph endpoint offset in the edge keys and
recompute the frozen Part C coverage audit (local CPU, read-only).

Bug (old `zinc_g0_transfer_audit_v1/part_c_audit.py::compute_model_keys`): the
edge loop indexed the *concatenated* `node_mask` / `node_desc` arrays with the
per-graph local endpoint ids `u, v`, instead of `offsets[mi] + u`.  The training
path itself is correct: `e2e_dictenv_p1.env_collate` adds `Batch.ptr[mi]` to the
occurrence/endpoint fields.  This is an audit-code bug, not a model bug.

This module exposes ONE production edge-key generator used both by the
invariant tests (Phase 1) and the recompute (Phase 2) so the tested path is the
production path.  All other Part C rules are unchanged; downstream helpers are
imported from the old module (`part_c_audit`) and not modified.

Train-only loading: `load_split("train")` internally deserialises
`encoded_valid.pt`; this module never calls it.  It loads only
`encoded_train.pt` + the train env cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_topology_crossfit_diagnostic_v1 as d
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztf

import sys
SRC = Path("tracks/ksvd/results/zinc_topology_crossfit_diagnostic_v1").resolve()
OLD = Path("tracks/ksvd/results/zinc_g0_transfer_audit_v1").resolve()
OUT = Path(__file__).resolve().parent
FOLDS = ("A", "B")
SHUFFLE_SEED = 20261003

sys.path.insert(0, str(OLD))
import part_c_audit as pc  # noqa: E402  (frozen downstream helpers + constants)


# ---------------------------------------------------------------------------
# train-only data loading
# ---------------------------------------------------------------------------


def load_train_only():
    train = list(torch.load(sdp.CACHE_DIR / "encoded_train.pt", map_location="cpu", weights_only=False))
    p1run.attach_env(train, "train")  # train env cache only
    return train


def unique_edges(d_) -> list[tuple[int, int, int]]:
    u = d_.env_bond_u.numpy()
    v = d_.env_bond_v.numpy()
    t = d_.env_bond_type.numpy()
    seen: dict[tuple[int, int], int] = {}
    for a, b, ty in zip(u.tolist(), v.tolist(), t.tolist()):
        key = (a, b) if a < b else (b, a)
        if key in seen:
            assert seen[key] == ty, "bond type inconsistent across occurrences"
        else:
            seen[key] = ty
    return [(int(a), int(b), int(ty)) for (a, b), ty in seen.items()]


def build_index(train_data, order=None):
    n = len(train_data)
    order = np.arange(n) if order is None else np.asarray(order, np.int64)
    phi_list, atom_list, edges = [], [], []
    for oi in order.tolist():
        d_ = train_data[oi]
        phi_list.append(d_.dict_phi.numpy().astype(np.float32))
        atom_list.append(d_.dict_atom.numpy().astype(np.int64))
        edges.append(unique_edges(d_))
    counts = np.array([p.shape[0] for p in phi_list], np.int64)
    offsets = np.zeros(len(counts) + 1, np.int64)
    offsets[1:] = np.cumsum(counts)
    return {"phi_all": np.concatenate(phi_list, 0), "atom_all": np.concatenate(atom_list, 0),
            "counts": counts, "offsets": offsets, "edges": edges, "order": order}


# ---------------------------------------------------------------------------
# THE production edge-key generator (fixed)
# ---------------------------------------------------------------------------


def edge_keys_for_graph(node_mask, node_desc, edges_mi, node_base):
    """Return (support_keys, desc_keys, sem, gu_list, gv_list) for one graph.

    `node_base` MUST be `offsets[graph]`.  Endpoint ids u,v are graph-local.
    """
    sup, desc, sem, gus, gvs = [], [], [], [], []
    for (u, v, ty) in edges_mi:
        gu, gv = int(node_base) + int(u), int(node_base) + int(v)
        mu, mv = int(node_mask[gu]), int(node_mask[gv])
        du, dv = node_desc[gu], node_desc[gv]
        sup.append((mu, mv) if mu <= mv else (mv, mu))
        desc.append((du, dv) if du <= dv else (dv, du))
        sem.append(int(ty))
        gus.append(gu); gvs.append(gv)
    return sup, desc, sem, gus, gvs


def edge_keys_broken(node_mask, node_desc, edges_mi):
    """Reproduce the old bug: index the global arrays with local u,v (no offset)."""
    sup, desc = [], []
    for (u, v, _ty) in edges_mi:
        mu, mv = int(node_mask[int(u)]), int(node_mask[int(v)])
        du, dv = node_desc[int(u)], node_desc[int(v)]
        sup.append((mu, mv) if mu <= mv else (mv, mu))
        desc.append((du, dv) if du <= dv else (dv, du))
    return sup, desc


def edge_keys_reference(node_mask, node_desc, g):
    """Independent reference: slice each graph's node arrays, then index locally."""
    out = []
    counts = g["counts"]
    offsets = g["offsets"]
    for mi, elist in enumerate(g["edges"]):
        base = int(offsets[mi])
        nm = node_mask[base:base + int(counts[mi])]
        nd = node_desc[base:base + int(counts[mi])]
        for (u, v, ty) in elist:
            mu, mv = int(nm[int(u)]), int(nm[int(v)])
            du, dv = nd[int(u)], nd[int(v)]
            out.append(((mu, mv) if mu <= mv else (mv, mu),
                        (du, dv) if du <= dv else (dv, du), int(ty)))
    return out


# ---------------------------------------------------------------------------
# frozen-model structural codes
# ---------------------------------------------------------------------------


def load_model(fold: str):
    blob = d.load_blob(d.PREP_DIR / f"fold_{fold}.npz")
    model = zjd.make_arm_model("F", blob, seed=zjd.SEED)
    sd = torch.load(SRC / f"F_{fold}_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd["soup_state"])
    model.eval()
    return model


def node_codes(model, phi_all):
    common_dim = int(model.common_dim)
    X = torch.as_tensor(phi_all, dtype=torch.float32)
    codes = []
    with torch.no_grad():
        for s in range(0, X.shape[0], 65536):
            codes.append(model.code(X[s:s + 65536]).cpu().numpy())
    codes = np.concatenate(codes, 0)
    alpha = codes[:, common_dim:]
    mask = np.abs(alpha) > pc.SUPPORT_EPS
    node_mask = (mask.astype(np.int64) * (1 << np.arange(alpha.shape[1], dtype=np.int64))).sum(1)
    node_desc = [pc.digest_vec(phi_all[i]) for i in range(phi_all.shape[0])]
    return node_mask, node_desc, common_dim


def compute_model_keys_fixed(model, g, broken=False):
    """Production path: FIXED endpoint offsets (or `broken=True` for old)."""
    node_mask, node_desc, common_dim = node_codes(model, g["phi_all"])
    edge_support, edge_desc, edge_sem, edge_mol = [], [], [], []
    edge_gu, edge_gv = [], []
    for order_pos, elist in enumerate(g["edges"]):
        mi = int(g["order"][order_pos])  # stable ORIGINAL molecule id
        if broken:
            sup, desc = edge_keys_broken(node_mask, node_desc, elist)
            gus = gvs = []
        else:
            sup, desc, _sem, gus, gvs = edge_keys_for_graph(
                node_mask, node_desc, elist, int(g["offsets"][order_pos]))
        edge_support.extend(sup); edge_desc.extend(desc)
        edge_sem.extend([int(ty) for (_u, _v, ty) in elist])
        edge_mol.extend([mi] * len(elist))
        edge_gu.extend(gus); edge_gv.extend(gvs)
    return {
        "common_dim": common_dim, "node_mask": node_mask, "node_desc": node_desc,
        "node_support_size": np.array([bin(int(m)).count("1") for m in node_mask]),
        "edge_support": edge_support, "edge_desc": edge_desc, "edge_sem": edge_sem,
        "edge_mol": np.asarray(edge_mol, np.int64),
        "edge_gu": np.asarray(edge_gu, np.int64), "edge_gv": np.asarray(edge_gv, np.int64),
        "empty_support_nodes": int((node_mask == 0).sum()),
        "n_nodes": int(node_mask.shape[0]),
    }


def node_mol_map(g):
    """ORIGINAL molecule id for every global node (respects shuffled order)."""
    counts = g["counts"]
    order = g["order"]
    return np.repeat(order.astype(np.int64), counts)


# ---------------------------------------------------------------------------
# Phase 1 invariants
# ---------------------------------------------------------------------------


def invariant_minimal_counterexample():
    # two 2-node graphs, identical local edge (0,1); supports [1,2] and [4,8]
    node_mask = np.array([1, 2, 4, 8], np.int64)
    node_desc = [b"a", b"b", b"c", b"d"]
    g = {"counts": np.array([2, 2]), "offsets": np.array([0, 2, 4]),
         "edges": [[(0, 1, 1)], [(0, 1, 1)]]}
    old = []
    for mi, elist in enumerate(g["edges"]):
        sup, desc = edge_keys_broken(node_mask, node_desc, elist)
        old.append((sup[0], desc[0]))
    new = []
    for mi, elist in enumerate(g["edges"]):
        sup, desc, _sem, _gu, _gv = edge_keys_for_graph(node_mask, node_desc, elist, int(g["offsets"][mi]))
        new.append((sup[0], desc[0]))
    return {
        "old_support_keys": [list(x[0]) for x in old],
        "old_desc_keys": [[b.hex() for b in x[1]] for x in old],
        "old_both_same": bool(old[0] == old[1]),
        "new_support_keys": [list(x[0]) for x in new],
        "new_desc_keys": [[b.hex() for b in x[1]] for x in new],
        "new_both_differ": bool(new[0] != new[1]),
        "expected_new_support": [[1, 2], [4, 8]],
        "pass": bool(old[0] == old[1] and new[0] != new[1]
                     and [list(x[0]) for x in new] == [[1, 2], [4, 8]]),
    }


def invariant_endpoints(train_data, g, keys):
    counts = g["counts"]
    offsets = g["offsets"]
    ok, n_edges = True, 0
    for order_pos, elist in enumerate(g["edges"]):
        base = int(offsets[order_pos]); end = base + int(counts[order_pos])
        for (u, v, _ty) in elist:
            n_edges += 1
            if not (0 <= u < int(counts[order_pos]) and 0 <= v < int(counts[order_pos])):
                ok = False
            gu, gv = base + int(u), base + int(v)
            if not (base <= gu < end and base <= gv < end):
                ok = False
    # true undirected edge count / types unchanged by the offset
    total_unique = int(sum(len(e) for e in g["edges"]))
    return {"all_endpoints_in_graph_interval": bool(ok), "n_real_undirected_edges": total_unique,
            "n_edges_checked": n_edges,
            "edge_count_matches_keys": bool(total_unique == len(keys["edge_support"])),
            "pass": bool(ok and total_unique == len(keys["edge_support"]))}


def invariant_reference(g, keys):
    ref = edge_keys_reference(keys["node_mask"], keys["node_desc"], g)
    prod = [(keys["edge_support"][i], keys["edge_desc"][i], keys["edge_sem"][i])
            for i in range(len(keys["edge_support"]))]
    n_sup = sum(1 for a, b in zip(ref, prod) if a[0] == b[0])
    n_desc = sum(1 for a, b in zip(ref, prod) if a[1] == b[1])
    n_sem = sum(1 for a, b in zip(ref, prod) if a[2] == b[2])
    return {"n_edges": len(ref), "support_match": n_sup, "desc_match": n_desc, "sem_match": n_sem,
            "pass": bool(n_sup == len(ref) and n_desc == len(ref) and n_sem == len(ref))}


def invariant_per_graph_encoding(model, g, keys, n_graphs=8, seed=SHUFFLE_SEED):
    rng = np.random.default_rng(seed)
    cand = np.array([i for i in range(1, len(g["edges"]))])  # non-first graphs
    sizes = g["counts"]
    pick = list(rng.choice(cand, size=min(n_graphs, len(cand)), replace=False))
    # prefer different sizes: add largest/smallest non-first
    order_sizes = np.argsort(sizes)[::-1]
    for i in order_sizes:
        if i != 0 and i not in pick:
            pick.append(int(i))
        if len(pick) >= n_graphs:
            break
    pick = sorted(set(int(i) for i in pick))[:max(n_graphs, len(pick))]
    rows, worst = [], 0.0
    model.eval()
    for order_pos in pick:
        base = int(g["offsets"][order_pos]); end = base + int(g["counts"][order_pos])
        sl = keys["node_mask"][base:end]
        with torch.no_grad():
            z = model.code(torch.as_tensor(g["phi_all"][base:end], dtype=torch.float32)).cpu().numpy()
        alpha = z[:, int(model.common_dim):]
        m_loc = (np.abs(alpha) > pc.SUPPORT_EPS)
        nm_loc = (m_loc.astype(np.int64) * (1 << np.arange(alpha.shape[1], dtype=np.int64))).sum(1)
        sup_match = int(np.sum(nm_loc == sl))
        rows.append({"graph_index": int(order_pos), "n_nodes": int(g["counts"][order_pos]),
                     "support_equal": int(sup_match), "n_nodes_checked": int(len(sl))})
    return {"graphs": rows, "all_support_equal": bool(all(r["support_equal"] == r["n_nodes_checked"] for r in rows)),
            "pass": bool(all(r["support_equal"] == r["n_nodes_checked"] for r in rows))}


def invariant_shuffle(model, train_data, g, keys, train_fold_mols, meta_mols, dev_mols, pen_mol, gid_mol):
    """Shuffle graph order, rebuild keys, map back by stable original id."""
    n = len(train_data)
    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(n)
    g2 = build_index(train_data, order=perm)
    k2 = compute_model_keys_fixed(model, g2, broken=False)
    # per-original-graph edge key multisets
    def group_keys(gx, kx):
        by = {}
        for i, mi in enumerate(kx["edge_mol"].tolist()):
            by.setdefault(int(mi), []).append((kx["edge_support"][i], kx["edge_desc"][i], kx["edge_sem"][i]))
        return by
    b1, b2 = group_keys(g, keys), group_keys(g2, k2)
    ids = sorted(b1.keys())
    sup_ok = sum(1 for i in ids if sorted([x[0] for x in b1[i]]) == sorted([x[0] for x in b2[i]]))
    desc_ok = sum(1 for i in ids if sorted([x[1] for x in b1[i]]) == sorted([x[1] for x in b2[i]]))
    # coverage classification consistency on a sample of original ids
    struct_c1, sem_c1, joint_c1 = pc.count_frequencies(
        keys["edge_support"], keys["edge_sem"], pc.by_mol_map(keys["edge_mol"]), train_fold_mols)
    struct_c2, sem_c2, joint_c2 = pc.count_frequencies(
        k2["edge_support"], k2["edge_sem"], pc.by_mol_map(k2["edge_mol"]), train_fold_mols)
    def cl_of(kx, struct_c, sem_c, joint_c, mol):
        struct, sem = kx["edge_support"], kx["edge_sem"]
        items = [i for i, m in enumerate(kx["edge_mol"].tolist()) if m == mol]
        cs = {}
        for i in items:
            s = struct_c.get(struct[i], 0); mm = sem_c.get(int(sem[i]), 0)
            j = joint_c.get((struct[i], int(sem[i])), 0)
            cs[pc.classify(s, mm, j)] = cs.get(pc.classify(s, mm, j), 0) + 1
        return cs
    sample = ids[:200]
    cls_ok = sum(1 for m in sample if cl_of(keys, struct_c1, sem_c1, joint_c1, m)
                 == cl_of(k2, struct_c2, sem_c2, joint_c2, m))
    return {"n_graphs": n, "support_multiset_match": sup_ok, "desc_multiset_match": desc_ok,
            "n_graphs_compared": len(ids), "class_match": cls_ok, "n_class_compared": len(sample),
            "pass": bool(sup_ok == len(ids) and desc_ok == len(ids) and cls_ok == len(sample))}


# ---------------------------------------------------------------------------
# Phase 2 recompute
# ---------------------------------------------------------------------------


def recompute(train_data, g, keys_by_fold, idx, pen_mol, gid_mol, t25):
    summary = {"meta": {"protocol": "index-fixed recompute of zinc_g0_transfer_audit_v1",
                        "seen_threshold": pc.SEEN, "high": pc.HIGH, "low": pc.LOW,
                        "support_eps": pc.SUPPORT_EPS,
                        "edge_primary": "unordered endpoint residual-support-set pair x bond category",
                        "node_secondary": "single-node residual-support-set x atom category",
                        "descriptor_check": "dict_phi[65] round 1e-6; 16-byte blake2b; edge = unordered endpoint pair",
                        "official_test_loaded": False}, "folds": {}}
    matching = {}
    rows_store = {}
    dev_idx = idx["dev_idx"]
    for fold in FOLDS:
        keys = keys_by_fold[fold]
        keys["atom_all"] = g["atom_all"]
        keys["counts"] = g["counts"]; keys["edges"] = g["edges"]
        train_mols = (idx["a_idx"] if fold == "A" else idx["b_idx"]).tolist()
        meta_mols = (idx["b_idx"] if fold == "A" else idx["a_idx"]).tolist()
        bases = json.loads((SRC / f"F_{fold}.json").read_text())
        meta_pred = np.asarray(bases["meta_predictions_cal"], np.float64)
        meta_y = np.asarray(bases["meta_targets"], np.float64)
        dev_pred = np.asarray(bases["dev_predictions_cal"], np.float64)
        dev_y = np.asarray(bases["dev_targets"], np.float64)
        pred_by_mol = {int(m): (float(meta_pred[k]), float(meta_y[k])) for k, m in enumerate(meta_mols)}
        pred_by_mol.update({int(m): (float(dev_pred[k]), float(dev_y[k])) for k, m in enumerate(dev_idx.tolist())})
        meta_g0 = [int(m) for m in meta_mols if int(pen_mol[m]) == 0]
        dev_g0 = [int(m) for m in dev_idx.tolist() if int(pen_mol[m]) == 0]
        assert len(meta_g0) == 3851 and len(dev_g0) == 1926, (len(meta_g0), len(dev_g0))
        train_g0 = [int(m) for m in train_mols if int(pen_mol[m]) == 0]
        fold_summary = {"n_train_mols": len(train_mols), "n_meta_mols": len(meta_mols),
                        "unique_edge_support_keys": len(set(keys["edge_support"])),
                        "unique_edge_desc_keys": len(set(keys["edge_desc"])),
                        "unique_edge_joint_keys": len(set((keys["edge_support"][i], keys["edge_sem"][i])
                                                          for i in range(len(keys["edge_support"])))),
                        "views": {}}
        matching[fold] = {}
        for view in ("edge", "node"):
            matching[fold][view] = {}
            fold_summary["views"][view] = {}
            for variant in pc.KEY_VARIANTS:
                if view == "node":
                    struct = keys["node_mask"] if variant == "residual_support" else keys["node_desc"]
                    sem = g["atom_all"]; mol = node_mol_map(g)
                    size = keys["node_support_size"]
                else:
                    struct = keys["edge_support"] if variant == "residual_support" else keys["edge_desc"]
                    sem = keys["edge_sem"]; mol = keys["edge_mol"]; size = None
                bm = pc.by_mol_map(mol)
                struct_c, sem_c, joint_c = pc.count_frequencies(struct, sem, bm, train_mols)
                tg = pc.mol_class_rows(struct, sem, bm, size, struct_c, sem_c, joint_c, train_g0[:400])
                train_comp = {c: float(np.mean([r["frac"][c] for r in tg.values() if r["frac"][c] is not None]))
                              for c in pc.CLASSES}
                meta_cl = pc.mol_class_rows(struct, sem, bm, size, struct_c, sem_c, joint_c, meta_g0)
                dev_cl = pc.mol_class_rows(struct, sem, bm, size, struct_c, sem_c, joint_c, dev_g0)
                meta_rows = pc.per_molecule_table(meta_cl, g, t25, pred_by_mol, meta_g0, gid_mol)
                dev_rows = pc.per_molecule_table(dev_cl, g, t25, pred_by_mol, dev_g0, gid_mol)
                fold_summary["views"][view][variant] = {
                    "train_g0_class_fraction": train_comp,
                    "meta_g0": pc.group_stats(meta_rows), "dev_g0": pc.group_stats(dev_rows)}
                rows_store[(fold, view, variant, "meta_g0")] = meta_rows
                rows_store[(fold, view, variant, "dev_g0")] = dev_rows
                m = {}
                for split_name, rows in (("meta_g0", meta_rows), ("dev_g0", dev_rows)):
                    pairs, high, low, thr = pc.match_groups(rows)
                    m[split_name] = {
                        "n_high": len(high), "n_low": len(low), "n_pairs": len(pairs),
                        "unmatched_high_rate": float(1 - len(pairs) / len(high)) if high else None,
                        "unmatched_high_minus_low": pc.group_stats(rows)["mae_high_minus_low_unmatched"],
                        "paired_bootstrap": pc.paired_boot(pairs, rows)}
                matching[fold][view][variant] = m
        summary["folds"][fold] = fold_summary
    return summary, matching, rows_store


def class_composition(rows_store):
    out = {}
    for (f, v, var, s), R in rows_store.items():
        comp = {}
        for c in pc.CLASSES:
            x = np.array([r["frac"][c] if r["frac"][c] is not None else np.nan for r in R], np.float64)
            comp[c] = {"mean": float(np.nanmean(x)), "max": float(np.nanmax(x)),
                       "n_mol_ge_0.05": int(np.sum(x >= 0.05)), "n_mol_ge_0.25": int(np.sum(x >= 0.25))}
        un = np.array([r["unseen_frac"] for r in R], np.float64)
        comp["_unseen_joint"] = {"mean": float(np.nanmean(un)), "max": float(np.nanmax(un))}
        out[f"{f}|{v}|{var}|{s}"] = comp
    return out


def main() -> int:
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only invariants, skip full recompute")
    args = ap.parse_args()

    train_data = load_train_only()
    g = build_index(train_data)
    pen, _ = zjd._load_penalties()
    handoff = np.load(zjd.HANDOFF / "train.npz", allow_pickle=True)
    gid = handoff["canonical_group_id"].astype(np.int64)
    idx = d.load_subfold_index()
    n_mol = len(train_data)
    gid_mol, pen_mol = gid[:n_mol], pen[:n_mol]
    t25, _frame, _meta = ztf.matrices_for_split("train", train_data, "hinge")

    invariants = {"minimal_counterexample": invariant_minimal_counterexample()}
    keys_by_fold = {}
    for fold in FOLDS:
        model = load_model(fold)
        keys = compute_model_keys_fixed(model, g, broken=False)
        keys["atom_all"] = g["atom_all"]
        keys_by_fold[fold] = keys
        train_mols = (idx["a_idx"] if fold == "A" else idx["b_idx"]).tolist()
        meta_mols = (idx["b_idx"] if fold == "A" else idx["a_idx"]).tolist()
        invariants[f"endpoints_{fold}"] = invariant_endpoints(train_data, g, keys)
        invariants[f"reference_{fold}"] = invariant_reference(g, keys)
        invariants[f"per_graph_encoding_{fold}"] = invariant_per_graph_encoding(model, g, keys)
        invariants[f"shuffle_{fold}"] = invariant_shuffle(
            model, train_data, g, keys, train_mols, meta_mols, idx["dev_idx"].tolist(), pen_mol, gid_mol)
        print(f"[invariants {fold}] done {time.perf_counter()-t0:.1f}s", flush=True)

    all_pass = all(v.get("pass", True) for v in invariants.values())
    (OUT / "index_invariants.json").write_text(json.dumps(invariants, indent=2))
    print("invariants pass:", all_pass, flush=True)
    if not all_pass:
        print("INVARIANT FAILURE — stopping before coverage interpretation")
        return 2
    if args.quick:
        return 0

    summary, matching, rows_store = recompute(train_data, g, keys_by_fold, idx, pen_mol, gid_mol, t25)
    (OUT / "coverage_fixed.json").write_text(json.dumps(summary, indent=2))
    (OUT / "matching_fixed.json").write_text(json.dumps(matching, indent=2))
    (OUT / "class_composition_fixed.json").write_text(json.dumps(class_composition(rows_store), indent=2))
    np.savez_compressed(OUT / "per_molecule_fixed.npz", **{
        f"{f}|{v}|{k}|{s}": np.asarray([json.dumps(r) for r in rows])
        for (f, v, k, s), rows in rows_store.items()})
    print("TOTAL", round(time.perf_counter() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())