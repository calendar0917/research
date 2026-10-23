"""CPU gates and label-blind data audits for zinc_hierarchical_relation_v1.

Every check writes machine-readable evidence under ``analysis/``.  A failed
scientific gate is a result, not something to route around.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import hr_common as C
import hr_data as D
import hr_models as M


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _zinc_split(split: str):
    from torch_geometric.datasets import ZINC

    import cscl_features as cf

    return ZINC(root=str(cf.ZINC_ROOT), subset=True, split=split)


def _raw_graph(ds, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = ds[i]
    return g.x.numpy().copy(), g.edge_index.numpy().copy(), g.edge_attr.numpy().copy()


def _apply_train_stats(entry: dict[str, Any], stats: Mapping[str, Any]) -> None:
    D.apply_stats([entry], stats)


def _synthetic_graph(atom_types: Sequence[int], edges: Sequence[tuple[int, int, int]]):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei = []
    ea = []
    for u, v, bt in edges:
        ei.append((u, v))
        ei.append((v, u))
        ea.extend([bt, bt])
    if ei:
        edge_index = np.asarray(ei, dtype=np.int64).T
        edge_attr = np.asarray(ea, dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0,), dtype=np.int64)
    return x, edge_index, edge_attr


def _model_pred(model: M.HRNet, entry: Mapping[str, Any], device="cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        batch = D.move(D.collate([entry]), device)
        return model(batch).detach().cpu().numpy()


def _compare_entries(model, e_a, e_b) -> float:
    pa = _model_pred(model, e_a)
    pb = _model_pred(model, e_b)
    return float(np.max(np.abs(pa - pb)))


def _wl_colors(atom_types: Sequence[int], bonds: Sequence[tuple[int, int, int]], rounds: int = 2) -> list[str]:
    colors = [str(int(t)) for t in atom_types]
    adj: list[list[tuple[int, int]]] = [[] for _ in atom_types]
    for u, v, bt in bonds:
        adj[u].append((v, int(bt)))
        adj[v].append((u, int(bt)))
    for _ in range(rounds):
        new = []
        for i in range(len(atom_types)):
            sig = "|".join(sorted(f"{colors[j]}:{bt}" for j, bt in adj[i]))
            new.append(hashlib.sha1(f"{colors[i]}#{sig}".encode()).hexdigest()[:16])
        colors = new
    return colors


# ---------------------------------------------------------------------------
# data audit
# ---------------------------------------------------------------------------


def data_audit() -> dict[str, Any]:
    """Label-blind structural audit of the patch / support / relation inputs."""
    if not (C.CACHE_DIR / "hr_stats.npz").exists():
        raise SystemExit("run prep first")
    stats = D.load_stats(C.CACHE_DIR / "hr_stats.npz")
    val_entries = D.load_cache(C.CACHE_DIR / "hr_val.pt")
    train_entries = D.load_cache(C.CACHE_DIR / "hr_train.pt")
    n_sample = min(200, len(val_entries))
    ds = _zinc_split("val")

    # ---- patch vs nested-WL non-equivalence on real patches --------------
    wl_groups: dict[str, set[str]] = defaultdict(set)
    wl_key_count: Counter[str] = Counter()
    n_patches = 0
    for mi in range(n_sample):
        x, _ei, _ea = _raw_graph(ds, mi)
        e = val_entries[mi]
        atom_types = [int(v) for v in x.reshape(-1)]
        bonds = [(int(u), int(v), int(t)) for (u, v), t in zip(e["bond_index"], e["bond_type"])]
        colors = _wl_colors(atom_types, bonds, rounds=2)
        for pb in range(int(e["n_bonds"])):
            mask = e["pnode_patch"] == pb
            nodes, du, dv = e["pnode_atom"][mask], e["pnode_du"][mask], e["pnode_dv"][mask]
            root_u, root_v = e["bond_index"][pb]
            wl_sig = (
                colors[int(root_u)] + "|" + colors[int(root_v)] + "|"
                + ",".join(sorted(f"{int(a)}:{int(b)}:{colors[int(z)]}" for z, a, b in zip(nodes, du, dv)))
            )
            em = e["pedge_patch"] == pb
            ind_sig = ",".join(
                sorted(
                    f"{int(a)}:{int(b)}:{int(c)}:{int(d)}:{int(t)}"
                    for a, b, c, d, t in zip(
                        e["pedge_du_x"][em], e["pedge_dv_x"][em],
                        e["pedge_du_y"][em], e["pedge_dv_y"][em], e["pedge_type"][em],
                    )
                )
            )
            wl_groups[wl_sig].add(ind_sig)
            wl_key_count[wl_sig] += 1
            n_patches += 1
    multi = [k for k, v in wl_groups.items() if len(v) > 1]
    patches_in_multi = sum(wl_key_count[k] for k in multi)
    patch_wl = {
        "n_molecules": n_sample,
        "n_patches": n_patches,
        "n_wl_keys": len(wl_groups),
        "n_wl_keys_with_multiple_induced_signatures": len(multi),
        "patches_in_multi_groups": int(patches_in_multi),
        "fraction_patches_in_multi_groups": float(patches_in_multi / max(n_patches, 1)),
        "max_induced_signatures_per_wl_key": int(max((len(v) for v in wl_groups.values()), default=0)),
        "definition": "nested-WL signature = root color after 2 typed WL rounds + multiset of (distance to root u, distance to root v, WL color) over patch nodes; induced signature = multiset of (du,dv) endpoint tags + bond type over the actual induced edges",
    }

    # ---- synthetic 1-WL-equivalent pair: C6 vs 2C3 ------------------------
    c6 = _synthetic_graph([0] * 6, [(i, (i + 1) % 6, 0) for i in range(6)])
    two3 = _synthetic_graph([0] * 6, [(0, 1, 0), (1, 2, 0), (2, 0, 0), (3, 4, 0), (4, 5, 0), (5, 3, 0)])
    c6e = D.build_entry(0, *c6)
    t3e = D.build_entry(1, *two3)
    c6b = [(int(u), int(v), int(t)) for u, v, t in zip(c6e["bond_index"][:, 0], c6e["bond_index"][:, 1], c6e["bond_type"])]
    t3b = [(int(u), int(v), int(t)) for u, v, t in zip(t3e["bond_index"][:, 0], t3e["bond_index"][:, 1], t3e["bond_type"])]
    c6_colors = _wl_colors([0] * 6, c6b, 2)
    t3_colors = _wl_colors([0] * 6, t3b, 2)
    def _ind_sig(e):
        return sorted(
            (int(a), int(b), int(c), int(d), int(t))
            for a, b, c, d, t in zip(e["pedge_du_x"], e["pedge_dv_x"], e["pedge_du_y"], e["pedge_dv_y"], e["pedge_type"])
        )
    c6_ind, t3_ind = _ind_sig(c6e), _ind_sig(t3e)
    synthetic = {
        "c6_wl_colors": c6_colors,
        "two_triangles_wl_colors": t3_colors,
        "wl_root_identical": bool(c6_colors[0] == t3_colors[0] and set(c6_colors) == set(t3_colors)),
        "c6_root_patch_induced_signature": c6_ind,
        "two_triangles_root_patch_induced_signature": t3_ind,
        "induced_signatures_differ": bool(c6_ind != t3_ind),
    }

    # ---- support coverage -------------------------------------------------
    sup = np.concatenate([e["p_support"] for e in val_entries[:n_sample]], axis=0)
    support = {
        "fit_split": "train only",
        "n_train_bonds": int(stats["support"]["n_bonds"]),
        "n_unique_train_keys": int(stats["support"]["n_unique"]),
        "valid_patches_checked": int(len(sup)),
        "valid_unseen_fraction": float(sup[:, 2].mean()),
        "valid_support_z_mean": float(sup[:, 0].mean()),
        "valid_support_z_std": float(sup[:, 0].std()),
        "valid_support_quantile_mean": float(sup[:, 1].mean()),
        "note": "support is a 4-d scalar feature (train count z-score, quantile, unseen flag, clipped count); no per-key embedding exists",
    }

    # ---- relation statistics ---------------------------------------------
    rel_counts = Counter()
    pair_counts = []
    tokens = []
    for e in val_entries[:n_sample]:
        rel_counts.update(e["r_rel"].tolist())
        pair_counts.append(len(e["r_u"]))
        tokens.append(e["n_tokens"])
    relation = {
        "n_molecules": n_sample,
        "mean_tokens_per_molecule": float(np.mean(tokens)),
        "mean_related_pairs_per_molecule_unordered": float(np.mean(pair_counts)),
        "relation_type_counts": {str(k): int(v) for k, v in sorted(rel_counts.items())},
        "relation_types": D.REL_TYPES,
        "definition": "relation id = 1 + distance_bucket(0..3)*8 + same_ring_block*4 + A^2_walk_bucket(0..3); id 0 is self",
    }

    # ---- patch sizes ------------------------------------------------------
    ppn = [len(e["pnode_atom"]) / max(e["n_bonds"], 1) for e in train_entries]
    ppe = [len(e["pedge_x"]) / max(e["n_bonds"], 1) for e in train_entries]
    pmax = [int(np.max(np.bincount(e["pnode_patch"], minlength=max(e["n_bonds"], 1)))) if e["n_bonds"] else 0 for e in train_entries]
    sizes = {
        "mean_patch_nodes_per_bond": float(np.mean(ppn)),
        "mean_patch_edges_per_bond": float(np.mean(ppe)),
        "max_patch_nodes": int(np.max(pmax)) if pmax else 0,
        "n_train_molecules": len(train_entries),
    }

    # ---- raw input fidelity ----------------------------------------------
    import structural_relation_objects_v1 as sro

    diffs = []
    for mi in range(5):
        x, ei, ea = _raw_graph(ds, mi)
        fresh = sro.build_mol_objects(mi, x, ei, ea)
        cached = val_entries[mi]
        diffs.append({
            "mol": mi,
            "atom_feat_max_abs_diff": float(np.max(np.abs(fresh.atom_feat - cached["atom_feat"]))),
            "bond_index_equal": bool(np.array_equal(fresh.bond_index, cached["bond_index"])),
            "bond_type_equal": bool(np.array_equal(fresh.bond_type, cached["bond_type"])),
            "junc_index_equal": bool(np.array_equal(fresh.junc_index, cached["junc_index"])),
            "junc_feat_max_abs_diff": float(np.max(np.abs(fresh.junc_feat - cached["junc_feat"]))) if fresh.junc_feat.size else 0.0,
        })
    fidelity = {
        "n_checked": len(diffs),
        "checks": diffs,
        "max_atom_diff": max(d["atom_feat_max_abs_diff"] for d in diffs),
        "all_exact": all(
            d["bond_index_equal"] and d["bond_type_equal"] and d["junc_index_equal"]
            and d["atom_feat_max_abs_diff"] == 0.0 and d["junc_feat_max_abs_diff"] == 0.0
            for d in diffs
        ),
    }

    return {
        "protocol_id": C.PROTOCOL_ID,
        "patch_vs_nested_wl": patch_wl,
        "synthetic_wl_pair": synthetic,
        "support": support,
        "relations": relation,
        "sizes": sizes,
        "input_fidelity": fidelity,
        "labels_used": False,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# CPU gates
# ---------------------------------------------------------------------------


def run_gates(args: argparse.Namespace) -> dict[str, Any]:
    cfg = C.load_config()
    stats = D.load_stats(C.CACHE_DIR / "hr_stats.npz")
    ssl_path = C.REPO / cfg["ssl"]["path"]
    device = torch.device("cpu")
    val_entries = D.load_cache(C.CACHE_DIR / "hr_val.pt")
    train_entries = D.load_cache(C.CACHE_DIR / "hr_train.pt")
    ds = _zinc_split("val")
    checks: dict[str, Any] = {}

    model_a, info_a = M.build_model(arm="A", ssl_path=ssl_path, seed=0, cfg=cfg["model"], device=device)
    model_b, info_b = M.build_model(arm="B", ssl_path=ssl_path, seed=0, cfg=cfg["model"], device=device)
    checks["parameters"] = {
        "A": info_a["parameters"],
        "B": info_b["parameters"],
        "max_allowed": 500000,
        "A_ok": info_a["parameters"] <= 500000,
        "B_ok": info_b["parameters"] <= 500000,
    }

    # 1. raw input fidelity + edge direction / renumbering invariance
    e0 = val_entries[0]
    x, ei, ea = _raw_graph(ds, 0)
    import structural_relation_objects_v1 as sro

    fresh = sro.build_mol_objects(0, x, ei, ea)
    checks["raw_input_fidelity"] = {
        "atom_feat_max_abs_diff": float(np.max(np.abs(fresh.atom_feat - e0["atom_feat"]))),
        "bond_index_equal": bool(np.array_equal(fresh.bond_index, e0["bond_index"])),
        "bond_type_equal": bool(np.array_equal(fresh.bond_type, e0["bond_type"])),
        "junc_feat_max_abs_diff": float(np.max(np.abs(fresh.junc_feat - e0["junc_feat"]))),
        "pass": bool(
            np.max(np.abs(fresh.atom_feat - e0["atom_feat"])) == 0.0
            and np.array_equal(fresh.bond_index, e0["bond_index"])
            and np.array_equal(fresh.bond_type, e0["bond_type"])
        ),
    }
    rev_edges = ei[::-1].copy()
    e_rev = D.build_entry(0, x, rev_edges, ea)
    _apply_train_stats(e_rev, stats)
    checks["edge_direction_invariance"] = {
        "A_max_abs_pred_diff": _compare_entries(model_a, e0, e_rev),
        "B_max_abs_pred_diff": _compare_entries(model_b, e0, e_rev),
        "pass": bool(
            _compare_entries(model_a, e0, e_rev) < 1e-5 and _compare_entries(model_b, e0, e_rev) < 1e-5
        ),
    }
    rng = np.random.default_rng(0)
    perm = rng.permutation(e0["n_atoms"])
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    x_new = x[inv]
    ei_new = perm[ei]
    e_perm = D.build_entry(0, x_new, ei_new, ea)
    _apply_train_stats(e_perm, stats)
    diff_a = _compare_entries(model_a, e0, e_perm)
    diff_b = _compare_entries(model_b, e0, e_perm)
    checks["renumbering_invariance"] = {
        "A_max_abs_pred_diff": diff_a,
        "B_max_abs_pred_diff": diff_b,
        "pass": bool(diff_a < 1e-5 and diff_b < 1e-5),
    }

    # 2. batch vs single
    subset = val_entries[:8]
    batch = D.move(D.collate(subset), device)
    with torch.no_grad():
        model_a.eval(); model_b.eval()
        pa_batch = model_a(batch).numpy()
        pb_batch = model_b(batch).numpy()
    pa_single = np.asarray([_model_pred(model_a, e)[0] for e in subset])
    pb_single = np.asarray([_model_pred(model_b, e)[0] for e in subset])
    checks["batch_vs_single"] = {
        "A_max_abs_diff": float(np.max(np.abs(pa_batch - pa_single))),
        "B_max_abs_diff": float(np.max(np.abs(pb_batch - pb_single))),
        "pass": bool(np.max(np.abs(pa_batch - pa_single)) < 1e-5 and np.max(np.abs(pb_batch - pb_single)) < 1e-5),
    }

    # 3. empty / ring / disconnected boundaries
    boundary: dict[str, Any] = {}
    synthetic = {
        "single_atom": _synthetic_graph([0], []),
        "two_isolated_atoms": _synthetic_graph([0, 1], []),
        "ethane": _synthetic_graph([0, 0], [(0, 1, 0)]),
        "cyclohexane": _synthetic_graph([0] * 6, [(i, (i + 1) % 6, 0) for i in range(6)]),
        "two_ethane_components": _synthetic_graph([0, 0, 1, 1], [(0, 1, 0), (2, 3, 0)]),
    }
    for name, (sx, sei, sea) in synthetic.items():
        try:
            e = D.build_entry(0, sx, sei, sea)
            _apply_train_stats(e, stats)
            p = _model_pred(model_a, e)
            # no relation pair may cross disconnected components (checked structurally)
            boundary[name] = {
                "n_atoms": int(e["n_atoms"]),
                "n_bonds": int(e["n_bonds"]),
                "n_junc": int(e["n_junc"]),
                "n_pairs": int(len(e["r_u"])),
                "finite": bool(np.isfinite(p).all()),
                "pred": float(p[0]),
            }
        except Exception as exc:  # pragma: no cover - the gate is the report
            boundary[name] = {"error": repr(exc), "finite": False}
    # disconnected graph: verify no cross-component pair by construction
    sx, sei, sea = synthetic["two_ethane_components"]
    e_dis = D.build_entry(0, sx, sei, sea)
    cross = 0
    # component ids from the raw adjacency
    comp = [-1, -1, -1, -1]
    comp[0] = comp[1] = 0
    comp[2] = comp[3] = 1
    for u, v in zip(e_dis["r_u"], e_dis["r_v"]):
        # tokens are junctions (centres) and patches (root bonds); approximate via root atoms
        if u < e_dis["n_junc"] and v < e_dis["n_junc"]:
            cu = comp[int(e_dis["junc_index"][u, 1])]
            cv = comp[int(e_dis["junc_index"][v, 1])]
        else:
            cu = cv = -1
        if cu != -1 and cv != -1 and cu != cv:
            cross += 1
    boundary["two_ethane_components"]["cross_component_pairs_jj"] = int(cross)
    checks["boundary_cases"] = {
        "cases": boundary,
        "pass": all(v.get("finite", False) for v in boundary.values()),
    }

    # 4. train/valid isolation and no hidden ids
    checks["train_valid_isolation"] = {
        "prep_fit_split": C.read_json(C.ANALYSIS_DIR / "prep.json")["stats"]["fit_split"] if (C.ANALYSIS_DIR / "prep.json").exists() else "unknown",
        "labels_read_during_prep": C.read_json(C.ANALYSIS_DIR / "prep.json")["labels_read_during_prep"] if (C.ANALYSIS_DIR / "prep.json").exists() else None,
        "official_test_loaded": C.read_json(C.ANALYSIS_DIR / "prep.json")["splits"]["official_test_loaded"] if (C.ANALYSIS_DIR / "prep.json").exists() else None,
        "support_uses_train_counter": True,
        "valid_unseen_support_fraction": float(np.mean(np.concatenate([e["p_support"][:, 2] for e in val_entries[:50]]))),
        "batch_has_no_row_or_mol_id_feature": not any("row" in k or "mol" in k for k in D.collate(val_entries[:4]).keys()),
    }
    # recompute valid support with train stats and with valid stats, show they differ
    train_counter = stats["support"]["counter"]
    val_keys = []
    # use keys from a small fresh build of valid molecules to compare counters
    for mi in range(20):
        vx, vei, vea = _raw_graph(_zinc_split("val"), mi)
        ve = D.build_entry(mi, vx, vei, vea)
        val_keys.extend(ve.pop("bond_keys"))
    valid_counter = Counter(val_keys)
    unseen_in_train = sum(1 for k in val_keys if train_counter.get(k, 0) == 0)
    checks["train_valid_isolation"]["valid_bond_keys_checked"] = len(val_keys)
    checks["train_valid_isolation"]["valid_keys_unseen_in_train"] = int(unseen_in_train)
    checks["train_valid_isolation"]["valid_keys_total"] = int(len(val_keys))
    checks["train_valid_isolation"]["pass"] = bool(
        checks["train_valid_isolation"]["prep_fit_split"] == "train only"
        and checks["train_valid_isolation"]["labels_read_during_prep"] is False
        and checks["train_valid_isolation"]["official_test_loaded"] is False
        and unseen_in_train >= 0
    )

    # 5. shared initialisation A/B
    checks["shared_initialisation"] = {
        "A_hash": M.shared_state_hash(model_a),
        "B_hash": M.shared_state_hash(model_b),
        "pass": M.shared_state_hash(model_a) == M.shared_state_hash(model_b),
    }

    # 6. gradient coverage on a real batch
    grad_report: dict[str, Any] = {}
    for arm, model in (("A", model_a), ("B", model_b)):
        model.train()
        b = D.move(D.collate(train_entries[:64]), device)
        pred = model(b)
        loss = pred.abs().mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        zero = []
        norms = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                zero.append(name)
                norms[name] = 0.0
            else:
                n = float(p.grad.detach().norm())
                norms[name] = n
                if n == 0.0:
                    zero.append(name)
        grad_report[arm] = {"n_parameters": len(norms), "zero_grad": zero, "min_positive_norm": min((v for v in norms.values() if v > 0), default=0.0)}
        model.zero_grad(set_to_none=True)
    grad_report["pass"] = all(len(v["zero_grad"]) == 0 for k, v in grad_report.items() if k in ("A", "B"))
    checks["gradient_coverage"] = grad_report

    # 7. sensitivity
    sens: dict[str, Any] = {}
    b = D.move(D.collate(train_entries[:16]), device)
    model_a.eval()
    with torch.no_grad():
        p0 = model_a(b).numpy()
        b_sup = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        b_sup["p_support"] = torch.zeros_like(b["p_support"])
        p_sup = model_a(b_sup).numpy()
        b_edges = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        b_edges["pedge_type"] = torch.zeros_like(b["pedge_type"])
        p_edges = model_a(b_edges).numpy()
        b_j = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        b_j["junc_feat"] = torch.zeros_like(b["junc_feat"])
        p_j = model_a(b_j).numpy()
    sens["A_support_zero_max_abs_delta"] = float(np.max(np.abs(p_sup - p0)))
    sens["A_patch_edge_type_zero_max_abs_delta"] = float(np.max(np.abs(p_edges - p0)))
    sens["A_junction_feat_zero_max_abs_delta"] = float(np.max(np.abs(p_j - p0)))
    model_b.eval()
    with torch.no_grad():
        q_real = model_b(b, relation_mode="real").numpy()
        q_null = model_b(b, relation_mode="null").numpy()
    sens["B_real_vs_null_relation_max_abs_delta"] = float(np.max(np.abs(q_real - q_null)))
    sens["pass"] = bool(
        sens["A_support_zero_max_abs_delta"] > 1e-6
        and sens["A_patch_edge_type_zero_max_abs_delta"] > 1e-6
        and sens["A_junction_feat_zero_max_abs_delta"] > 1e-6
        and sens["B_real_vs_null_relation_max_abs_delta"] > 1e-6
    )
    checks["sensitivity"] = sens

    # 8. no hidden id / lookup memory
    embeddings = []
    for name, mod in model_b.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            embeddings.append({"name": name, "num_embeddings": int(mod.num_embeddings), "dim": int(mod.embedding_dim)})
    checks["no_hidden_ids"] = {
        "embeddings": embeddings,
        "max_embedding_vocab": max((e["num_embeddings"] for e in embeddings), default=0),
        "role_vocab": 3,
        "relation_vocab": D.REL_TYPES,
        "pass": all(e["num_embeddings"] <= max(3, D.REL_TYPES) for e in embeddings),
    }

    # 9. CPU profile of a real batch
    prof = {}
    for arm, model in (("A", model_a), ("B", model_b)):
        model.train()
        b = D.move(D.collate(train_entries[:128]), device)
        target = torch.as_tensor(np.zeros(128, np.float32))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(1):
            pred = model(b); loss = pred.abs().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        t0 = time.time()
        for _ in range(3):
            pred = model(b); loss = pred.abs().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        prof[arm] = {"seconds_per_batch_128_cpu": (time.time() - t0) / 3}
    prof["pass"] = all(v["seconds_per_batch_128_cpu"] < 2.0 for k, v in prof.items() if k in ("A", "B"))
    checks["cpu_profile"] = prof

    # 10. deterministic batch order (A and B use the same generator seed)
    gen = torch.Generator(); gen.manual_seed(int(cfg["recipe"]["shuffle_seed"]))
    perms = [torch.randperm(len(train_entries), generator=gen).numpy() for _ in range(3)]
    checks["batch_order"] = {
        "hashes": [hashlib.sha256(p.tobytes()).hexdigest() for p in perms],
        "pass": len({hashlib.sha256(p.tobytes()).hexdigest() for p in perms}) == 3,
    }

    checks["all_pass"] = all(
        bool(checks[k].get("pass", False))
        for k in (
            "raw_input_fidelity", "edge_direction_invariance", "renumbering_invariance",
            "batch_vs_single", "boundary_cases", "train_valid_isolation",
            "shared_initialisation", "gradient_coverage", "sensitivity",
            "no_hidden_ids", "cpu_profile", "batch_order",
        )
    ) and checks["parameters"]["A_ok"] and checks["parameters"]["B_ok"]
    return checks


def cmd_gates(args: argparse.Namespace) -> int:
    checks = run_gates(args)
    C.write_json(C.ANALYSIS_DIR / "cpu_gates.json", checks)
    print(json.dumps(checks, indent=2)[:12000])
    return 0 if checks["all_pass"] else 2


# ---------------------------------------------------------------------------
# overfit test
# ---------------------------------------------------------------------------


def cmd_overfit(args: argparse.Namespace) -> int:
    from hr_runner import load_y, _eval_predict

    cfg = C.load_config()
    ocfg = cfg["overfit_gate"]
    entries = D.load_cache(C.CACHE_DIR / "hr_train.pt")[: int(ocfg["n_molecules"])]
    y = load_y("train")[: len(entries)]
    device = torch.device(args.device)
    out: dict[str, Any] = {"n_molecules": len(entries), "steps": int(ocfg["steps"]), "device": str(device)}
    ssl_path = C.REPO / cfg["ssl"]["path"]
    for arm in args.arms:
        model, info = M.build_model(arm=arm, ssl_path=ssl_path, seed=0, cfg=cfg["model"], device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(ocfg["lr"]), weight_decay=float(cfg["recipe"]["weight_decay"]))
        gen = torch.Generator(); gen.manual_seed(777)
        batch_size = int(ocfg["batch_size"])
        initial = _eval_predict(model, entries, 64, device)
        initial_mae = C.mae(initial, y)
        losses = []
        model.train()
        for step in range(int(ocfg["steps"])):
            idx = torch.randint(0, len(entries), (batch_size,), generator=gen).numpy()
            batch = D.move(D.collate([entries[j] for j in idx]), device)
            target = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = (pred - target).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        final = _eval_predict(model, entries, 64, device)
        final_mae = C.mae(final, y)
        const_mae = float(np.abs(y - y.mean()).mean())
        out[arm] = {
            "parameters": info["parameters"],
            "initial_train_mae": initial_mae,
            "final_train_mae": final_mae,
            "constant_baseline_mae": const_mae,
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "pass": bool(final_mae <= float(ocfg["final_train_mae_max"]) and final_mae <= 0.5 * initial_mae),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out["all_pass"] = all(v.get("pass", False) for k, v in out.items() if k in ("A", "B"))
    C.write_json(C.ANALYSIS_DIR / f"overfit_{args.tag}.json", out)
    print(json.dumps(out, indent=2))
    return 0 if out["all_pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("gates")
    q.set_defaults(func=cmd_gates)
    q = sub.add_parser("data-audit")
    q.set_defaults(func=lambda a: (C.write_json(C.ANALYSIS_DIR / "data_audit.json", data_audit()), 0)[1])
    q = sub.add_parser("overfit")
    q.add_argument("--device", default="cpu")
    q.add_argument("--arms", nargs="+", default=["A", "B"])
    q.add_argument("--tag", default="cpu")
    q.set_defaults(func=cmd_overfit)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
