"""Part C — G0 structure-semantics combination transfer audit (local CPU, read-only).

Two frozen counting views on the real computation path of the two Full soups:

* edge (primary)  : structure = unordered pair of the two endpoint nodes'
                    residual-IHT support sets (|alpha_k| > 1e-8 on the 32-d
                    residual code); semantics = bond category.
* node (secondary): structure = single node's residual-IHT support set;
                    semantics = atom category.

Frequency = number of *distinct canonical training molecules* containing the key
(one molecule counts once however often the key appears).  Joint frequency is
counted the same way on the joint key.

Pre-registered verification key: replace the residual-support structure key by
the full `dict_phi[65]` descriptor key (node = vector rounded to 6 decimals;
edge = unordered sorted pair of the two endpoint descriptor keys).  Descriptor
keys are compared through a 16-byte blake2b digest of the rounded float32 bytes
(equality-preserving).

Held-out = the model's own meta fold G0 (penalty 0) and the outer-dev G0, using
that model's own base calibrated predictions.  No test, no fit, no optimizer.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_topology_crossfit_diagnostic_v1 as d
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1

SRC = Path("tracks/ksvd/results/zinc_topology_crossfit_diagnostic_v1").resolve()
OUT = Path(__file__).resolve().parent
FOLDS = ("A", "B")
SEEN = 5
SUPPORT_EPS = 1e-8
HIGH = 0.25
LOW = 0.05
N_BOOT = 1000
BOOT_SEED = 20261003
CLASSES = ("joint_seen", "joint_rare_marginals_seen", "structure_rare",
           "semantic_rare", "both_rare")
KEY_VARIANTS = ("residual_support", "full_descriptor")


# ---------------------------------------------------------------------------
# global graph data
# ---------------------------------------------------------------------------


def build_graph_index(train_data):
    phi_list, atom_list, edges = [], [], []
    for d_ in train_data:
        phi_list.append(d_.dict_phi.numpy().astype(np.float32))
        atom_list.append(d_.dict_atom.numpy().astype(np.int64))
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
        edges.append([(int(a), int(b), int(ty)) for (a, b), ty in seen.items()])
    counts = np.array([p.shape[0] for p in phi_list], np.int64)
    offsets = np.zeros(len(counts) + 1, np.int64)
    offsets[1:] = np.cumsum(counts)
    return {"phi_all": np.concatenate(phi_list, 0),
            "atom_all": np.concatenate(atom_list, 0),
            "counts": counts, "offsets": offsets, "edges": edges}


def digest_vec(vec: np.ndarray) -> bytes:
    return hashlib.blake2b(
        np.round(np.asarray(vec, np.float64), 6).astype(np.float32).tobytes(),
        digest_size=16).digest()


# ---------------------------------------------------------------------------
# per-model structural codes (FROZEN soup, no grad)
# ---------------------------------------------------------------------------


def compute_model_keys(fold: str, g: dict):
    blob = d.load_blob(d.PREP_DIR / f"fold_{fold}.npz")
    model = zjd.make_arm_model("F", blob, seed=zjd.SEED)
    sd = torch.load(SRC / f"F_{fold}_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd["soup_state"])
    model.eval()
    common_dim = int(model.common_dim)
    phi_all = torch.as_tensor(g["phi_all"], dtype=torch.float32)
    codes = []
    with torch.no_grad():
        for start in range(0, phi_all.shape[0], 65536):
            codes.append(model.code(phi_all[start:start + 65536]).cpu().numpy())
    codes = np.concatenate(codes, 0)
    alpha = codes[:, common_dim:]
    mask = np.abs(alpha) > SUPPORT_EPS
    node_mask = (mask.astype(np.int64) * (1 << np.arange(alpha.shape[1], dtype=np.int64))).sum(1)
    node_support_size = mask.sum(1)
    node_desc = [digest_vec(g["phi_all"][i]) for i in range(phi_all.shape[0])]

    edge_support, edge_desc, edge_sem, edge_mol = [], [], [], []
    for mi, elist in enumerate(g["edges"]):
        for (u, v, ty) in elist:
            mu, mv = int(node_mask[u]), int(node_mask[v])
            edge_support.append((mu, mv) if mu <= mv else (mv, mu))
            du, dv = node_desc[u], node_desc[v]
            edge_desc.append((du, dv) if du <= dv else (dv, du))
            edge_sem.append(int(ty))
            edge_mol.append(mi)
    return {
        "common_dim": common_dim,
        "node_mask": node_mask, "node_support_size": node_support_size, "node_desc": node_desc,
        "edge_support": edge_support, "edge_desc": edge_desc, "edge_sem": edge_sem,
        "edge_mol": np.asarray(edge_mol, np.int64),
        "empty_support_nodes": int((node_support_size == 0).sum()),
        "n_nodes": int(node_mask.shape[0]),
        "support_size_hist": {str(k): int((node_support_size == k).sum())
                              for k in sorted(set(node_support_size.tolist()))},
    }


# ---------------------------------------------------------------------------
# frequency counting + classification
# ---------------------------------------------------------------------------


def item_index(keys, g, view, variant):
    if view == "node":
        struct = keys["node_mask"] if variant == "residual_support" else keys["node_desc"]
        sem = g["atom_all"]
        mol = np.repeat(np.arange(len(g["counts"])), g["counts"])
        size = keys["node_support_size"]
    else:
        struct = keys["edge_support"] if variant == "residual_support" else keys["edge_desc"]
        sem = keys["edge_sem"]
        mol = keys["edge_mol"]
        size = None
    return struct, sem, mol, size


def by_mol_map(mol):
    out: dict[int, list[int]] = {}
    for i, m in enumerate(mol.tolist()):
        out.setdefault(m, []).append(i)
    return out


def count_frequencies(struct, sem, bm, train_mols):
    struct_c, sem_c, joint_c = Counter(), Counter(), Counter()
    for m in train_mols:
        items = bm.get(m, [])
        ss, sm, sj = set(), set(), set()
        for i in items:
            s = struct[i]
            c = int(sem[i])
            ss.add(s); sm.add(c); sj.add((s, c))
        for s in ss:
            struct_c[s] += 1
        for c in sm:
            sem_c[c] += 1
        for k in sj:
            joint_c[k] += 1
    return struct_c, sem_c, joint_c


def classify(s, m, j):
    if j >= SEEN:
        return "joint_seen"
    if s >= SEEN and m >= SEEN:
        return "joint_rare_marginals_seen"
    if s < SEEN and m >= SEEN:
        return "structure_rare"
    if m < SEEN and s >= SEEN:
        return "semantic_rare"
    return "both_rare"


def mol_class_rows(struct, sem, bm, size, struct_c, sem_c, joint_c, mol_ids):
    out = {}
    for m in mol_ids:
        items = bm.get(m, [])
        counts = {c: 0 for c in CLASSES}
        unseen = 0
        sizes = []
        for i in items:
            s = struct_c.get(struct[i], 0)
            mm = sem_c.get(int(sem[i]), 0)
            j = joint_c.get((struct[i], int(sem[i])), 0)
            counts[classify(s, mm, j)] += 1
            if j == 0:
                unseen += 1
            if size is not None:
                sizes.append(int(size[i]))
        n = len(items)
        out[m] = {"n": n, "counts": counts,
                  "frac": {c: (counts[c] / n if n else None) for c in CLASSES},
                  "unseen_joint": unseen, "unseen_frac": (unseen / n if n else None),
                  "support_size_mean": (float(np.mean(sizes)) if sizes else None)}
    return out


def per_molecule_table(cl, g, t25, pred_by_mol, held_mols, gid_mol):
    rows = []
    for m in held_mols:
        r = cl[m]
        cr = np.asarray(t25[m], np.float64)
        pred, y = pred_by_mol[m]
        rows.append({"mol": int(m), "group": int(gid_mol[m]), "n": int(r["n"]),
                     "frac": r["frac"], "counts": r["counts"],
                     "unseen_joint": int(r["unseen_joint"]), "unseen_frac": r["unseen_frac"],
                     "support_size_mean": r["support_size_mean"],
                     "cycle_rank": int(round(cr[14])), "n_nodes": int(g["counts"][m]),
                     "n_edges": int(len(g["edges"][m])),
                     "base_pred": float(pred), "y": float(y),
                     "err": float(pred - y), "abs_err": float(abs(pred - y)),
                     "penalty": int(0)})
    return rows


# ---------------------------------------------------------------------------
# group stats + matching
# ---------------------------------------------------------------------------


def exposure_of(r, key):
    v = r["frac"].get(key)
    return np.nan if v is None else float(v)


def group_stats(rows, key="joint_rare_marginals_seen"):
    exp = np.array([exposure_of(r, key) for r in rows], np.float64)
    ae = np.array([r["abs_err"] for r in rows], np.float64)
    er = np.array([r["err"] for r in rows], np.float64)
    out = {"exposure_key": key, "n_total": len(rows)}
    for name, sel in (("high", exp >= HIGH), ("mid", (exp > LOW) & (exp < HIGH)), ("low", exp <= LOW)):
        if sel.sum() == 0:
            out[name] = {"n": 0}
            continue
        idxs = np.where(sel)[0]
        out[name] = {
            "n": int(sel.sum()), "exposure_mean": float(np.nanmean(exp[sel])),
            "mae": float(ae[sel].mean()), "signed_mean": float(er[sel].mean()),
            "contrib": float(ae[sel].sum() / len(rows)),
            "node_mean": float(np.mean([rows[i]["n_nodes"] for i in idxs])),
            "edge_mean": float(np.mean([rows[i]["n_edges"] for i in idxs])),
            "cycle_rank_mean": float(np.mean([rows[i]["cycle_rank"] for i in idxs])),
        }
    if out.get("high", {}).get("mae") is not None and out.get("low", {}).get("mae") is not None:
        out["mae_high_minus_low_unmatched"] = out["high"]["mae"] - out["low"]["mae"]
    else:
        out["mae_high_minus_low_unmatched"] = None
    return out


def match_groups(rows, key="joint_rare_marginals_seen"):
    thr = (2, 3, 0.25, 0.10)
    exp = np.array([exposure_of(r, key) for r in rows], np.float64)
    order = np.argsort([r["mol"] for r in rows])
    high = [i for i in order if exp[i] >= HIGH]
    low = [i for i in order if exp[i] <= LOW]
    used_groups, used_low, pairs = set(), set(), []
    for hi in high:
        if rows[hi]["group"] in used_groups:
            continue
        best, best_cost = None, None
        for li in low:
            if li in used_low or rows[li]["group"] in used_groups:
                continue
            if rows[li]["cycle_rank"] != rows[hi]["cycle_rank"]:
                continue
            dn = abs(rows[hi]["n_nodes"] - rows[li]["n_nodes"])
            de = abs(rows[hi]["n_edges"] - rows[li]["n_edges"])
            dp = abs(rows[hi]["base_pred"] - rows[li]["base_pred"])
            ds = abs((rows[hi]["frac"].get("structure_rare") or 0.0)
                     - (rows[li]["frac"].get("structure_rare") or 0.0))
            if dn > thr[0] or de > thr[1] or dp > thr[2] or ds > thr[3]:
                continue
            cost = (dn / thr[0]) ** 2 + (de / thr[1]) ** 2 + (dp / thr[2]) ** 2 + (ds / thr[3]) ** 2
            if best_cost is None or cost < best_cost or (cost == best_cost and rows[li]["mol"] < rows[best]["mol"]):
                best, best_cost = li, cost
        if best is not None:
            pairs.append((hi, best))
            used_groups.add(rows[hi]["group"]); used_groups.add(rows[best]["group"])
            used_low.add(best)
    return pairs, high, low, thr


def paired_boot(pairs, rows, n_boot=N_BOOT, seed=BOOT_SEED):
    if not pairs:
        return {"n_pairs": 0}
    diff = np.array([rows[h]["abs_err"] - rows[l]["abs_err"] for h, l in pairs], np.float64)
    signed = np.array([rows[h]["err"] - rows[l]["err"] for h, l in pairs], np.float64)
    n = len(diff)
    rng = np.random.default_rng(int(seed))
    draws = np.array([float(diff[rng.integers(0, n, n)].mean()) for _ in range(int(n_boot))])
    return {"n_pairs": int(n), "mae_high_minus_low": float(diff.mean()),
            "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
            "signed_error_high_minus_low": float(signed.mean()),
            "n_boot": int(n_boot), "seed": int(seed)}


# ---------------------------------------------------------------------------
# channel health
# ---------------------------------------------------------------------------


def channel_health(fold, train_data):
    blob = d.load_blob(d.PREP_DIR / f"fold_{fold}.npz")
    model = zjd.make_arm_model("F", blob, seed=zjd.SEED)
    sd = torch.load(SRC / f"F_{fold}_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd["soup_state"])
    model.eval()
    zjd.apply_prep_blob(train_data, p1run.load_split("valid"), blob)
    batch = p1.env_collate(train_data[:16])
    cap = {}

    def hook(name):
        def fn(_m, _i, out):
            cap[name] = out.detach().clone()
        return fn
    h1 = model.node_encoder.register_forward_hook(hook("node"))
    h2 = model.edge_encoder.register_forward_hook(hook("edge"))
    q = model.reader.net[4].bias.detach().clone()
    with torch.no_grad():
        model.reader.net[4].bias.add_(float(json.loads((SRC / f"F_{fold}.json").read_text())["calibration"]["delta"]))
        model(batch, mask=cm.C6_MASK)
    h1.remove(); h2.remove()
    with torch.no_grad():
        model.reader.net[4].bias.copy_(q)

    def stat(t):
        t2 = t.reshape(-1, t.shape[-1]) if t.dim() > 2 else t
        return {"shape": list(t2.shape), "per_dim_std_max": float(t2.std(0).max()),
                "per_dim_std_mean": float(t2.std(0).mean()),
                "row_std_mean": float(t2.std(1).mean()) if t2.shape[0] > 1 else 0.0,
                "abs_mean": float(t2.abs().mean()),
                "constant_across_rows": bool(float(t2.std(0).max()) < 1e-8)}
    return {"node_encoder_out": stat(cap["node"]), "edge_encoder_out": stat(cap["edge"]),
            "node_channel_constant": stat(cap["node"])["constant_across_rows"],
            "edge_channel_constant": stat(cap["edge"])["constant_across_rows"],
            "n_nodes_in_batch": int(batch.dict_phi.shape[0]),
            "n_bond_occ": int(batch.env_bond_u.shape[0])}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    t0 = time.perf_counter()
    train_data = p1run.load_split("train")
    g = build_graph_index(train_data)
    pen, _ = zjd._load_penalties()
    handoff = np.load(zjd.HANDOFF / "train.npz", allow_pickle=True)
    gid = handoff["canonical_group_id"].astype(np.int64)
    idx = d.load_subfold_index()
    dev_idx = idx["dev_idx"]
    t25 = d.raw_topology25()
    bases = {f: json.loads((SRC / f"F_{f}.json").read_text()) for f in FOLDS}
    n_mol = len(train_data)
    gid_mol, pen_mol = gid[:n_mol], pen[:n_mol]

    summary = {"meta": {
        "n_molecules": n_mol, "n_nodes": int(g["phi_all"].shape[0]), "seen_threshold": SEEN,
        "high": HIGH, "low": LOW, "support_eps": SUPPORT_EPS,
        "primary_view": "edge", "secondary_view": "node",
        "edge_primary": "unordered endpoint residual-support-set pair x bond category",
        "node_secondary": "single-node residual-support-set x atom category",
        "descriptor_check": "dict_phi[65] round 1e-6; edge = unordered endpoint descriptor pair; "
                            "16-byte blake2b digest",
        "official_test_loaded": False}}
    channel = {}
    per_mol_rows = {}
    matching = {}

    for fold in FOLDS:
        keys = compute_model_keys(fold, g)
        summary["meta"][f"common_dim_{fold}"] = keys["common_dim"]
        summary["meta"][f"empty_support_nodes_{fold}"] = keys["empty_support_nodes"]
        summary["meta"][f"support_size_hist_{fold}"] = keys["support_size_hist"]
        train_mols = (idx["a_idx"] if fold == "A" else idx["b_idx"]).tolist()
        meta_mols = (idx["b_idx"] if fold == "A" else idx["a_idx"]).tolist()
        meta_pred = np.asarray(bases[fold]["meta_predictions_cal"], np.float64)
        meta_y = np.asarray(bases[fold]["meta_targets"], np.float64)
        dev_pred = np.asarray(bases[fold]["dev_predictions_cal"], np.float64)
        dev_y = np.asarray(bases[fold]["dev_targets"], np.float64)
        pred_by_mol = {int(m): (float(meta_pred[k]), float(meta_y[k])) for k, m in enumerate(meta_mols)}
        pred_by_mol.update({int(m): (float(dev_pred[k]), float(dev_y[k])) for k, m in enumerate(dev_idx.tolist())})
        meta_g0 = [int(m) for m in meta_mols if int(pen_mol[m]) == 0]
        dev_g0 = [int(m) for m in dev_idx.tolist() if int(pen_mol[m]) == 0]
        assert len(meta_g0) == 3851 and len(dev_g0) == 1926, (len(meta_g0), len(dev_g0))
        train_g0 = [int(m) for m in train_mols if int(pen_mol[m]) == 0]

        fold_summary = {"n_train_mols": len(train_mols), "n_meta_mols": len(meta_mols),
                        "n_meta_g0": len(meta_g0), "n_dev_g0": len(dev_g0), "views": {}}
        matching[fold] = {}
        for view in ("edge", "node"):
            matching[fold][view] = {}
            fold_summary["views"][view] = {}
            for variant in KEY_VARIANTS:
                struct, sem, mol, size = item_index(keys, g, view, variant)
                bm = by_mol_map(mol)
                freq = count_frequencies(struct, sem, bm, train_mols)
                struct_c, sem_c, joint_c = freq
                # descriptive training-fold G0 composition (first 400 for speed)
                tg = mol_class_rows(struct, sem, bm, size, struct_c, sem_c, joint_c, train_g0[:400])
                train_comp = {c: float(np.mean([r["frac"][c] for r in tg.values() if r["frac"][c] is not None]))
                              for c in CLASSES}
                meta_cl = mol_class_rows(struct, sem, bm, size, struct_c, sem_c, joint_c, meta_g0)
                dev_cl = mol_class_rows(struct, sem, bm, size, struct_c, sem_c, joint_c, dev_g0)
                meta_rows = per_molecule_table(meta_cl, g, t25, pred_by_mol, meta_g0, gid_mol)
                dev_rows = per_molecule_table(dev_cl, g, t25, pred_by_mol, dev_g0, gid_mol)
                fold_summary["views"][view][variant] = {
                    "train_g0_class_fraction": train_comp,
                    "meta_g0": group_stats(meta_rows), "dev_g0": group_stats(dev_rows)}
                per_mol_rows[(fold, view, variant, "meta_g0")] = meta_rows
                per_mol_rows[(fold, view, variant, "dev_g0")] = dev_rows
                # matching
                m = {}
                for split_name, rows in (("meta_g0", meta_rows), ("dev_g0", dev_rows)):
                    pairs, high, low, thr = match_groups(rows)
                    gs = group_stats(rows)
                    m[split_name] = {
                        "n_high": len(high), "n_low": len(low),
                        "thresholds": {"cycle_rank_equal": True, "node_diff_le": thr[0],
                                       "edge_diff_le": thr[1], "base_pred_diff_le": thr[2],
                                       "structure_rare_frac_diff_le": thr[3]},
                        "n_pairs": len(pairs),
                        "unmatched_high_rate": float(1 - len(pairs) / len(high)) if high else None,
                        "unmatched_low_rate": float(1 - len(pairs) / max(len(low), 1)),
                        "unmatched_high_minus_low": gs["mae_high_minus_low_unmatched"],
                        "paired_bootstrap": paired_boot(pairs, rows),
                    }
                matching[fold][view][variant] = m
        summary[f"fold_{fold}"] = fold_summary
        print(f"[{fold}] done {time.perf_counter()-t0:.1f}s", flush=True)

    for fold in FOLDS:
        channel[fold] = channel_health(fold, train_data)

    # aggregate evidence grading
    grading = {}
    for view in ("edge", "node"):
        for variant in KEY_VARIANTS:
            folds_ok, diffs, n_pairs = [], {}, {}
            for fold in FOLDS:
                mb = matching[fold][view][variant]["meta_g0"]["paired_bootstrap"]
                diffs[fold] = mb.get("mae_high_minus_low")
                n_pairs[fold] = mb.get("n_pairs", 0)
                folds_ok.append((diffs[fold] is not None and diffs[fold] >= 0.02 and n_pairs[fold] >= 50))
            devb = matching["A"][view][variant]["dev_g0"]["paired_bootstrap"]
            devb2 = matching["B"][view][variant]["dev_g0"]["paired_bootstrap"]
            direction_consistent = all(d is not None and d > 0 for d in diffs.values())
            keys_consistent = True  # filled after loop
            grading[f"{view}:{variant}"] = {
                "meta_mae_high_minus_low": diffs, "meta_n_pairs": n_pairs,
                "both_folds_ge_0.02_and_ge50": bool(all(folds_ok)),
                "direction_consistent": direction_consistent,
                "dev_mae_high_minus_low": {"A": devb.get("mae_high_minus_low"),
                                           "B": devb2.get("mae_high_minus_low")},
                "dev_n_pairs": {"A": devb.get("n_pairs"), "B": devb2.get("n_pairs")},
            }
    # cross-key direction consistency (edge view)
    for view in ("edge", "node"):
        sgn = {variant: grading[f"{view}:{variant}"]["direction_consistent"] for variant in KEY_VARIANTS}
        grading[f"{view}:key_consistency"] = sgn

    (OUT / "part_c_coverage.json").write_text(json.dumps(summary, indent=2))
    (OUT / "part_c_matching.json").write_text(json.dumps({"matching": matching, "grading": grading}, indent=2))
    payload = {f"{f}|{v}|{k}|{s}": np.asarray([json.dumps(r) for r in rows])
               for (f, v, k, s), rows in per_mol_rows.items()}
    np.savez_compressed(OUT / "part_c_per_molecule.npz", **payload)
    (OUT / "part_c_channel_health.json").write_text(json.dumps(channel, indent=2))
    print("TOTAL", round(time.perf_counter() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())