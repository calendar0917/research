"""CPU representation / correctness checks for P3-BF.

Everything here is label-free.  The checks are the declared pre-GPU gate:
exact hand-graph counting against an independent subset enumeration, atom
permutation invariance, real-data object statistics, raw108-root-group
refinement by the new multiplicity-aware content multiset, the declared A/B
example, and train->valid content coverage.  They also cover the narrow model
correctness items (P3B parity, zero new columns, optimizer groups, gradient
opening, empty-set semantics).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
_V2 = _HERE.parents[1] / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T  # noqa: E402
import ad2_models as A2  # noqa: E402
import bf_data as B  # noqa: E402
import bf_models as M  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
RESULTS: dict[str, dict] = {}


def run(name):
    def deco(fn):
        fn.check_name = name

        def wrapped():
            try:
                RESULTS[name] = {"pass": True, "detail": fn()}
            except Exception as exc:  # noqa: BLE001
                RESULTS[name] = {
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
        wrapped.check_name = name
        return wrapped
    return deco


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fake_obj(categories, edges, bond_types=None):
    cats = np.asarray(categories, dtype=np.int64)
    n = len(cats)
    feat = np.zeros((n, 108), dtype=np.float32)
    feat[np.arange(n), cats] = 1.0
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if bond_types is None:
        bond_types = np.ones((len(edges),), dtype=np.int64)
    return SimpleNamespace(
        n_atoms=n,
        atom_feat=feat,
        bond_index=edges,
        bond_type=np.asarray(bond_types, dtype=np.int64),
        bond_feat=np.zeros((len(edges), 7), dtype=np.float32),
        junc_feat=np.zeros((0, 32), dtype=np.float32),
        junc_index=np.zeros((0, 3), dtype=np.int64),
        junc_bond=np.zeros((0, 2), dtype=np.int64),
        global_feat=np.zeros((33,), dtype=np.float32),
    )


def cache_from_obj(obj, split="__temp__"):
    n, e = int(obj.n_atoms), int(len(obj.bond_index))
    j = int(len(obj.junc_index))
    return {
        "split": split,
        "n_molecules": 1,
        "atom_ptr": np.array([0, n], dtype=np.int32),
        "atom_feat": np.asarray(obj.atom_feat, dtype=np.float32),
        "atom_type": np.argmax(np.asarray(obj.atom_feat)[:, : D.N_ATOM_TYPES], axis=1).astype(np.int64),
        "bond_ptr": np.array([0, e], dtype=np.int32),
        "bond_feat": np.asarray(obj.bond_feat, dtype=np.float32).reshape(e, 7),
        "bond_index": np.asarray(obj.bond_index, dtype=np.int16).reshape(e, 2),
        "bond_type": np.asarray(obj.bond_type, dtype=np.int64).reshape(e),
        "junc_ptr": np.array([0, j], dtype=np.int32),
        "junc_feat": np.asarray(obj.junc_feat, dtype=np.float32).reshape(j, 32),
        "junc_index": np.asarray(obj.junc_index, dtype=np.int16).reshape(j, 3),
        "junc_bond": np.asarray(obj.junc_bond, dtype=np.int16).reshape(j, 2),
        "global_feat": np.asarray(obj.global_feat, dtype=np.float32).reshape(1, 33),
        "labels": np.zeros((1,), dtype=np.float32),
    }


def raw_pyg_graph(split: str, index: int):
    ds = D.pyg_zinc(split)
    d = ds[index]
    x = np.asarray(d.x).reshape(-1).astype(np.int64)
    ei = np.asarray(d.edge_index)
    ea = np.asarray(d.edge_attr).reshape(-1).astype(np.int64)
    return x, ei, ea


def build_object(x, edge_index, edge_attr, index: int = 0):
    C.ensure_p3_modules()
    import structural_relation_objects_v1 as sro

    return sro.build_mol_objects(index, x, edge_index, edge_attr)


def permute_raw(x, edge_index, edge_attr, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))  # new_order[new] = old
    old_to_new = np.empty((len(x),), dtype=np.int64)
    old_to_new[perm] = np.arange(len(x), dtype=np.int64)
    return x[perm], old_to_new[edge_index], edge_attr.copy(), old_to_new


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


@run("reference_p3b")
def check_reference():
    norm = D.load_norm()
    valid = D.load_cache("valid")
    rows = np.arange(len(valid["labels"]), dtype=np.int64)
    model, _ = A2.build_model("P3N", norm, 0, "cpu")
    ck = torch.load(M.P3B_CKPT, map_location="cpu", weights_only=False)
    C.load_state(model, ck["state_dict"], "cpu")
    _, mae = T.evaluate(model, valid, rows, candidate="P3N", device="cpu", batch_size=256)
    info = C.read_json(_V2.parent / "runs" / "xBssl_P3N_s0" / "train.json")
    expected = float(info["best_valid"]["valid_mae"])
    assert int(ck["epoch"]) == 377, ck["epoch"]
    assert abs(float(ck["valid_mae"]) - expected) < 1e-12
    assert abs(mae - expected) < 1e-6, (mae, expected)
    assert C.sha256_file(M.P3B_CKPT) == "4c379c93214c003a6c58fd4add4a64ecb4fe23603233738fab18f3885abdc34f"
    return {
        "checkpoint": str(M.P3B_CKPT.relative_to(C.REPO)),
        "sha256": C.sha256_file(M.P3B_CKPT),
        "epoch": int(ck["epoch"]),
        "stored_valid_mae": expected,
        "replay_valid_mae": float(mae),
        "abs_diff": float(abs(mae - expected)),
        "train_label_sha256": info["data"]["train_label_sha256"],
        "valid_label_sha256": info["data"]["valid_label_sha256"],
    }


@run("parameters_and_groups")
def check_parameters():
    norm = D.load_norm()
    model, info = M.build_model(norm, seed=0, device="cpu")
    inherited, new = M.param_groups(model)
    opt = torch.optim.AdamW(
        [
            {"params": [p for _, p in inherited], "lr": 1e-4},
            {"params": [p for _, p in new], "lr": 1e-3},
        ],
        weight_decay=1e-5,
    )
    assert info["parameters"] == 447_505, info["parameters"]
    assert info["parameters_inherited"] == 413_505
    assert info["parameters_new"] == 34_000
    assert C.optimizer_parameter_count(opt) == info["parameters"]
    assert len(opt.param_groups) == 2
    assert all(p.requires_grad for p in model.parameters())
    assert model.w_s.weight.abs().sum().item() == 0.0
    return {
        "parameters": info["parameters"],
        "breakdown": info["parameters_breakdown"],
        "parameters_inherited": info["parameters_inherited"],
        "parameters_new": info["parameters_new"],
        "optimizer_parameters": C.optimizer_parameter_count(opt),
        "param_groups": [
            {"name": "inherited_p3", "tensors": len(inherited), "peak_lr": 1e-4, "min_lr": 1e-6},
            {"name": "new", "tensors": len(new), "peak_lr": 1e-3, "min_lr": 1e-5},
        ],
        "cap": 500_000,
        "headroom": 500_000 - info["parameters"],
        "w_s_zero": True,
        "init": info,
    }


@run("zero_new_columns_parity")
def check_zero_columns():
    norm = D.load_norm()
    valid = D.load_cache("valid")
    rows = np.arange(32, dtype=np.int64)
    p3 = A2.P3NullModel(norm)
    ck = torch.load(M.P3B_CKPT, map_location="cpu", weights_only=False)
    p3.load_state_dict(ck["state_dict"])
    model, _ = M.build_model(norm, seed=0, device="cpu")
    batch = B.collate_bf(valid, rows)
    p3.eval(); model.eval()
    with torch.no_grad():
        y3 = p3(batch)
        yb = model(batch)
        s = model.branch(batch)
    diff = float((y3 - yb).abs().max())
    assert diff < 1e-6, diff
    assert torch.isfinite(s).all()
    return {
        "rows": int(len(rows)),
        "max_abs_diff_vs_p3": diff,
        "s_shape": list(s.shape),
        "s_finite": True,
        "s_abs_max": float(s.abs().max()),
        "s_all_zero_roots": int((s.abs().sum(dim=1) == 0).sum()),
    }


def subset_counts(obj, root: int):
    """Independent subset-based enumeration of the same three families."""
    cats = B.atom_categories(obj)
    bmap = B.bond_map(obj)
    n = int(obj.n_atoms)
    out = {0: Counter(), 1: Counter(), 2: Counter()}
    others = [i for i in range(n) if i != root]
    for k, fam in ((3, 0), (4, 1)):
        for comb in itertools.combinations(others, k):
            if all(bmap.get((root, u)) for u in comb):
                out[fam][B.canonical_key(cats, bmap, root, comb)] += 1
    for comb in itertools.combinations(others, 3):
        ok = False
        for u in comb:
            if bmap.get((root, u)):
                rest = [w for w in comb if w != u]
                if all(bmap.get((u, w)) for w in rest):
                    ok = True
                    break
        if ok:
            out[2][B.canonical_key(cats, bmap, root, comb)] += 1
    return out


@run("hand_graph_counts")
def check_hand():
    cases = {}
    graphs = {
        "star3": fake_obj([0, 1, 1, 2], [(0, 1), (0, 2), (0, 3)]),
        "star4": fake_obj([0, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4)]),
        "fork": fake_obj([0, 1, 2, 3, 3], [(0, 1), (0, 2), (1, 3), (1, 4)]),
        "path": fake_obj([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)]),
        "star3_closure": fake_obj([0, 1, 1, 2], [(0, 1), (0, 2), (0, 3), (1, 2)]),
        "k4": fake_obj([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        "ring6": fake_obj([0] * 6, [(i, (i + 1) % 6) for i in range(6)]),
    }
    # Hand totals are per whole graph.  A low-degree root can still be the
    # root of an F4 object (root=leaf, centre=the former hub), so F4 totals
    # are not concentrated at the hub.
    expected_totals = {
        "star3": {"S3": 1, "S4": 0, "F4": 3},
        "star4": {"S3": 4, "S4": 1, "F4": 12},
        "fork": {"S3": 1, "S4": 0, "F4": 3},
        "path": {"S3": 0, "S4": 0, "F4": 0},
        "star3_closure": {"S3": 1, "S4": 0, "F4": 3},
        "k4": {"S3": 4, "S4": 0, "F4": 4},
        "ring6": {"S3": 0, "S4": 0, "F4": 0},
    }
    key_equal = {}
    for name, obj in graphs.items():
        ours = B.enumerate_rooted_branching(obj)
        totals = {f: int(sum(sum(d.values()) for d in ours[fi].values()))
                  for fi, f in enumerate(B.FAMILIES)}
        assert totals == expected_totals[name], (name, totals)
        # independent subset enumeration must agree pattern-for-pattern
        for root in range(int(obj.n_atoms)):
            ind = subset_counts(obj, root)
            for fi in range(3):
                assert ours[fi][root] == dict(ind[fi]), (name, root, fi, ours[fi][root], ind[fi])
        cases[name] = {
            "n_atoms": int(obj.n_atoms),
            "totals": totals,
            "per_root": {
                str(r): {f: int(sum(ours[fi][r].values())) for fi, f in enumerate(B.FAMILIES)}
                for r in range(int(obj.n_atoms))
            },
        }

    # The extra closure changes S3 content but not the count.
    plain = B.enumerate_rooted_branching(graphs["star3"])[0][0]
    closed = B.enumerate_rooted_branching(graphs["star3_closure"])[0][0]
    assert sum(plain.values()) == sum(closed.values()) == 1
    assert set(plain) != set(closed)
    kp = next(iter(plain))
    kc = next(iter(closed))
    pair_p = np.frombuffer(kp[B.MAX_NODES:], dtype=np.uint8)
    pair_c = np.frombuffer(kc[B.MAX_NODES:], dtype=np.uint8)
    assert int((pair_p == 1).sum()) == 3 and int((pair_c == 1).sum()) == 4
    key_equal["star3_vs_closure"] = {
        "counts": [int(sum(plain.values())), int(sum(closed.values()))],
        "content_keys_differ": True,
        "real_edges": [int((pair_p == 1).sum()), int((pair_c == 1).sum())],
    }
    # SF: star3_closure root0 is also an F4? no: leaf 1 has degree 2 -> only one
    # node left, so F4 stays empty.  fork shows F4.
    return {"graphs": cases, "extra_closure": key_equal, "all_independent_counts_match": True}


@run("atom_permutation_enumeration")
def check_permutation_enum():
    x, ei, ea = raw_pyg_graph("valid", 0)
    obj = build_object(x, ei, ea)
    xp, eip, eap, old_to_new = permute_raw(x, ei, ea, seed=7)
    objp = build_object(xp, eip, eap)
    a = B.enumerate_rooted_branching(obj)
    b = B.enumerate_rooted_branching(objp)
    for root in range(int(obj.n_atoms)):
        new_root = int(old_to_new[root])
        for fi in range(3):
            assert a[fi][root] == b[fi][new_root], (root, new_root, fi)
    return {
        "n_atoms": int(obj.n_atoms),
        "families_checked": list(B.FAMILIES),
        "content_multisets_equal_after_permutation": True,
        "object_totals": {f: int(sum(sum(d.values()) for d in a[fi].values()))
                          for fi, f in enumerate(B.FAMILIES)},
    }


@run("atom_permutation_prediction")
def check_permutation_prediction():
    x, ei, ea = raw_pyg_graph("valid", 0)
    obj = build_object(x, ei, ea)
    xp, eip, eap, _ = permute_raw(x, ei, ea, seed=7)
    objp = build_object(xp, eip, eap)
    for name, o in (("__orig__", obj), ("__perm__", objp)):
        B.register_cache(name, B.build_from_objects([o], name, {"source": "check"}, verbose=False))
    norm = D.load_norm()
    model, _ = M.build_model(norm, seed=0, device="cpu")
    with torch.no_grad():
        model.w_s.weight.normal_(0.0, 0.02)  # open the branch for this check
    model.eval()
    bo = B.collate_bf(cache_from_obj(obj, "__orig__"), np.array([0]))
    bp = B.collate_bf(cache_from_obj(objp, "__perm__"), np.array([0]))
    with torch.no_grad():
        yo = model(bo)
        yp = model(bp)
        so = model.branch(bo)
        sp = model.branch(bp)
    diff = float((yo - yp).abs().max())
    assert diff < 1e-5, diff
    return {
        "prediction_max_abs_diff": diff,
        "s_max_abs_diff": float((so - sp).abs().max()),
        "n_atoms": int(obj.n_atoms),
        "branch_open": True,
    }


@run("batch_regrouping")
def check_batch_regrouping():
    valid = D.load_cache("valid")
    norm = D.load_norm()
    model, _ = M.build_model(norm, seed=0, device="cpu")
    with torch.no_grad():
        model.w_s.weight.normal_(0.0, 0.02)
    model.eval()
    rows2 = np.array([0, 1], dtype=np.int64)
    b2 = B.collate_bf(valid, rows2)
    b1 = B.collate_bf(valid, np.array([0], dtype=np.int64))
    with torch.no_grad():
        y2, y1 = model(b2), model(b1)
        s2, s1 = model.branch(b2), model.branch(b1)
    n0 = int(np.diff(np.asarray(valid["atom_ptr"]))[0])
    d_s = float((s2[:n0] - s1).abs().max())
    d_y = float((y2[0] - y1[0]).abs().max())
    assert d_y < 1e-5 and d_s < 1e-5, (d_y, d_s)
    return {"s_max_abs_diff": d_s, "pred_max_abs_diff": d_y}


@run("multiplicity_and_empty")
def check_multiplicity():
    obj = fake_obj([0, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4)])
    per = B.enumerate_rooted_branching(obj)
    s3 = per[0][0]
    s4 = per[1][0]
    assert len(s3) == 1 and sum(s3.values()) == 4, s3
    assert len(s4) == 1 and sum(s4.values()) == 1, s4
    assert per[2][0] == {}
    # branch empty semantics on synthetic one-molecule batches
    B.register_cache("__mult__", B.build_from_objects([obj], "__mult__", {"source": "check"}, verbose=False))
    cache = cache_from_obj(obj, "__mult__")
    b = B.collate_bf(cache, np.array([0], dtype=np.int64))
    branch = M.BranchingFragmentBranch()
    branch.eval()
    with torch.no_grad():
        s = branch(b)
    root_s = s[0]
    # S3 stats (positions 0..65): mu nonzero; S4 (66..131): mu nonzero; F4 0
    assert float(root_s[:66].abs().sum()) > 0
    assert float(root_s[66:132].abs().sum()) > 0
    assert float(root_s[132:].abs().sum()) == 0.0
    # a pure path has no branching object at any root: exact empty-set zero
    path = fake_obj([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    B.register_cache("__empty__", B.build_from_objects([path], "__empty__", {"source": "check"}, verbose=False))
    b_empty = B.collate_bf(cache_from_obj(path, "__empty__"), np.array([0], dtype=np.int64))
    with torch.no_grad():
        s_empty = branch(b_empty)
    assert float(s_empty.abs().sum()) == 0.0, s_empty
    leaf_s = s_empty
    # expanding counts into multiple entries preserves the statistics
    c = b["occ_count"]
    lo = torch.floor(c / 2).long()
    hi = c.long() - lo
    keep = c > 0
    def dup(t):
        return torch.cat([t[keep], t[keep]], dim=0)
    b2 = dict(b)
    b2["occ_pat"] = dup(b["occ_pat"])
    b2["occ_root"] = dup(b["occ_root"])
    b2["occ_family"] = dup(b["occ_family"])
    b2["occ_count"] = torch.cat([lo[keep], hi[keep]], dim=0).float()
    with torch.no_grad():
        s2 = branch(b2)
    d = float((s - s2).abs().max())
    assert d < 1e-6, d
    return {
        "S3_patterns": len(s3), "S3_multiplicity": int(sum(s3.values())),
        "S4_patterns": len(s4), "S4_multiplicity": int(sum(s4.values())),
        "F4_empty_at_hub": True,
        "path_all_roots_zero": True,
        "compressed_vs_expanded_max_abs_diff": d,
    }


@run("gradient_and_schedule")
def check_gradients():
    valid = D.load_cache("valid")
    norm = D.load_norm()
    rows = np.arange(8, dtype=np.int64)
    batch = B.collate_bf(valid, rows)
    y = torch.as_tensor(np.asarray(valid["labels"])[rows], dtype=torch.float32)
    model, _ = M.build_model(norm, seed=0, device="cpu")
    inherited, new = M.param_groups(model)
    opt = torch.optim.AdamW(
        [
            {"params": [p for _, p in inherited], "lr": 1e-4},
            {"params": [p for _, p in new], "lr": 1e-3},
        ],
        weight_decay=1e-5,
    )
    watch = {
        "w_s": model.w_s.weight,
        "phi": model.branch.phi.lins[0].weight,
        "atom_w_x": model.encoders.atom_encoder.lins[0].weight,
        "reader": model.reader.lins[0].weight,
    }
    records = []
    model.train()
    for step in range(1, 5):
        opt.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = F.l1_loss(pred, y)
        loss.backward()
        g = {k: (float(p.grad.abs().max()) if p.grad is not None else None)
             for k, p in watch.items()}
        opt.step()
        records.append({"step": step, "loss": float(loss), "grad_max": g})
        assert torch.isfinite(loss)
    assert records[0]["grad_max"]["w_s"] and records[0]["grad_max"]["w_s"] > 0
    assert records[0]["grad_max"]["phi"] == 0.0
    assert any((r["grad_max"]["phi"] or 0) > 0 for r in records[1:])
    assert all((r["grad_max"]["atom_w_x"] or 0) > 0 for r in records)
    assert all((r["grad_max"]["reader"] or 0) > 0 for r in records)
    total = 79 * 100
    schedule = {}
    for name, peak, floor in (("inherited_p3", 1e-4, 1e-6), ("new", 1e-3, 1e-5)):
        schedule[name] = {
            str(step): C.cosine_with_warmup_lr(step, total, 200, peak, floor)
            for step in (1, 100, 200, 201, total // 2, total)
        }
    assert schedule["inherited_p3"]["200"] > schedule["inherited_p3"]["201"]
    assert abs(schedule["new"]["1"] - 1e-3 / 200) < 1e-15
    assert abs(schedule["new"][str(total)] - 1e-5) < 1e-15
    return {
        "records": records,
        "phi_first_nonzero_step": next(r["step"] for r in records if (r["grad_max"]["phi"] or 0) > 0),
        "schedule_example_total_updates": total,
        "schedule": schedule,
        "all_watched_parameters_update": True,
    }


@run("cache_integrity")
def check_cache():
    out = {}
    for split in ("train", "valid"):
        audit = B.audit_cache(split)
        assert audit["pattern_bad"] == 0, audit
        assert audit["entries_positive"] and audit["roots_in_range"] and audit["families_ok"]
        meta = audit["meta"]
        assert meta["labels_used"] is False
        out[split] = {
            "patterns": audit["patterns"],
            "entries": audit["entries"],
            "pattern_bad": audit["pattern_bad"],
            "meta": meta,
        }
    frozen = (ANALYSIS / "freeze.json").exists()
    test_exists = B.cache_path("test").exists()
    assert frozen or not test_exists, "test fragment cache exists before the freeze"
    return {"splits": out, "test_cache_exists_before_freeze": bool(test_exists)}


# ---------------------------------------------------------------------------
# raw108-root refinement (representation verification)
# ---------------------------------------------------------------------------


def float32_key(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(vec, dtype=np.float32)).tobytes()


def root_signature_digests(pc, n_mol: int):
    """sha256 of the multiplicity-aware content multiset for every root."""
    pat_atom = np.asarray(pc["pat_atom"])
    pat_pair = np.asarray(pc["pat_pair"])
    pat_family = np.asarray(pc["pat_family"])
    keys = [
        bytes([int(pat_family[p])]) + bytes(pat_atom[p]) + bytes(pat_pair[p])
        for p in range(len(pat_atom))
    ]
    ent_ptr = np.asarray(pc["ent_ptr"])
    ent_pat = np.asarray(pc["ent_pat"])
    ent_root = np.asarray(pc["ent_root"])
    ent_family = np.asarray(pc["ent_family"])
    ent_count = np.asarray(pc["ent_count"])
    atoms_per_mol = np.asarray(pc["atoms_per_mol"])
    out = {}
    for m in range(n_mol):
        s, e = int(ent_ptr[m]), int(ent_ptr[m + 1])
        per = defaultdict(list)
        for i in range(s, e):
            per[int(ent_root[i])].append(
                (int(ent_family[i]), keys[int(ent_pat[i])], int(ent_count[i]))
            )
        for root in range(int(atoms_per_mol[m])):
            items = per.get(root, [])
            items.sort()
            full = b"".join(k + int(c).to_bytes(4, "little") for _, k, c in items)
            content = b"".join(k for _, k, _ in items)
            mult = b"".join(int(c).to_bytes(4, "little") for _, _, c in items)
            out[(m, root)] = (
                hashlib.sha256(full).digest(),
                hashlib.sha256(content).digest(),
                hashlib.sha256(mult).digest(),
            )
    return out


def descriptor_summary(atom_feat_row: np.ndarray) -> dict:
    feat = np.asarray(atom_feat_row, dtype=np.float32)
    return {
        "atom_type": int(np.argmax(feat[0:28])),
        "degree": int(np.argmax(feat[28:33])),
        "in_ring": float(feat[33]),
        "ring_bond_count": int(np.argmax(feat[40:42])),
        "bond_type_counts": feat[42:46].astype(int).tolist(),
        "neighbour_types": np.nonzero(feat[46:74])[0].astype(int).tolist(),
        "radius2_types": np.nonzero(feat[74:102])[0].astype(int).tolist(),
        "scalars": [round(float(v), 6) for v in feat[102:108]],
    }


def content_variants(pc, n_mol: int, wanted: set[tuple[int, int]], limit: int = 3):
    """Human-readable per-root content multiset for selected roots."""
    pat_atom = np.asarray(pc["pat_atom"])
    pat_pair = np.asarray(pc["pat_pair"])
    pat_family = np.asarray(pc["pat_family"])
    ent_ptr = np.asarray(pc["ent_ptr"])
    ent_pat = np.asarray(pc["ent_pat"])
    ent_root = np.asarray(pc["ent_root"])
    ent_family = np.asarray(pc["ent_family"])
    ent_count = np.asarray(pc["ent_count"])
    out = defaultdict(list)
    for m in sorted({m for m, _ in wanted}):
        s, e = int(ent_ptr[m]), int(ent_ptr[m + 1])
        for i in range(s, e):
            key = (m, int(ent_root[i]))
            if key not in wanted:
                continue
            p = int(ent_pat[i])
            out[key].append({
                "family": B.FAMILIES[int(ent_family[i])],
                "atoms": np.asarray(pat_atom[p]).astype(int).tolist(),
                "pairs": np.asarray(pat_pair[p]).astype(int).tolist(),
                "count": int(ent_count[i]),
            })
    for key in out:
        out[key].sort(key=lambda r: (r["family"], r["atoms"], r["pairs"], r["count"]))
    return out


def representation_refinement():
    pc = B.load_cache("train")
    objs, _ = D.load_p3_objects("train")
    n_mol = int(pc["meta"]["n_molecules"])
    assert n_mol == len(objs)
    digests = root_signature_digests(pc, n_mol)

    groups = defaultdict(list)
    for m, obj in enumerate(objs):
        feat = np.asarray(obj.atom_feat)
        for root in range(int(obj.n_atoms)):
            groups[float32_key(feat[root])].append((m, root))

    n_roots = sum(len(v) for v in groups.values())
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    subdiv = {}
    for k, roots in multi.items():
        full = {digests[r][0] for r in roots}
        if len(full) > 1:
            subdiv[k] = (roots, full)
    roots_multi = sum(len(v) for v in multi.values())
    roots_sub = sum(len(v) for v in subdiv.values())
    mols_sub = sorted({m for roots, _ in subdiv.values() for m, _ in roots})
    n_groups_sub_content = 0
    n_groups_sub_mult = 0
    n_groups_sub_both = 0
    for roots, _ in subdiv.values():
        c = len({digests[r][1] for r in roots})
        mm = len({digests[r][2] for r in roots})
        n_groups_sub_content += int(c > 1)
        n_groups_sub_mult += int(mm > 1)
        n_groups_sub_both += int(c > 1 and mm > 1)

    def examples_for(keys, limit=3):
        out = []
        for k in keys[:limit]:
            roots = subdiv[k][0]
            by_sig = defaultdict(list)
            for r in roots:
                by_sig[digests[r][0]].append(r)
            variants = content_variants(pc, n_mol, set(roots))
            out.append({
                "raw108_group_size": len(roots),
                "descriptor": descriptor_summary(np.frombuffer(k, dtype=np.float32)),
                "n_signatures": len(by_sig),
                "variants": [
                    {
                        "roots": [{"molecule": int(m), "root": int(r)} for m, r in sorted(rs)],
                        "content": {f"{m}:{r}": variants.get((m, r), [])
                                    for m, r in sorted(rs)[:3]},
                    }
                    for rs in list(by_sig.values())[:3]
                ],
            })
        return out

    candidates = sorted(
        [k for k in subdiv if 2 <= len(subdiv[k][0]) <= 4],
        key=lambda k: len(subdiv[k][0]),
    )
    examples = examples_for(candidates, limit=4)
    pure_content = [
        k for k in subdiv
        if len({digests[r][1] for r in subdiv[k][0]}) > 1
        and len({digests[r][2] for r in subdiv[k][0]}) == 1
    ]
    pure_mult = [
        k for k in subdiv
        if len({digests[r][1] for r in subdiv[k][0]}) == 1
        and len({digests[r][2] for r in subdiv[k][0]}) > 1
    ]
    pure_content.sort(key=lambda k: len(subdiv[k][0]))
    pure_mult.sort(key=lambda k: len(subdiv[k][0]))
    return {
        "split": "train",
        "n_roots": int(n_roots),
        "n_raw108_groups": int(len(groups)),
        "n_groups_singleton": int(len(groups) - len(multi)),
        "n_groups_multi": int(len(multi)),
        "n_roots_in_multi_groups": int(roots_multi),
        "n_groups_subdivided": int(len(subdiv)),
        "n_roots_in_subdivided_groups": int(roots_sub),
        "n_molecules_in_subdivided_groups": int(len(mols_sub)),
        "fraction_roots_in_multi_groups": float(roots_multi / n_roots),
        "fraction_roots_subdivided": float(roots_sub / n_roots),
        "fraction_molecules_subdivided": float(len(mols_sub) / n_mol),
        "fraction_multi_groups_subdivided": float(len(subdiv) / max(len(multi), 1)),
        "subdivided_by_content": int(n_groups_sub_content),
        "subdivided_by_multiplicity": int(n_groups_sub_mult),
        "subdivided_by_both": int(n_groups_sub_both),
        "pure_content_subdivisions": int(len(pure_content)),
        "pure_multiplicity_subdivisions": int(len(pure_mult)),
        "examples": examples,
        "examples_content_split": examples_for(pure_content, limit=2),
        "examples_multiplicity_split": examples_for(pure_mult, limit=2),
    }


@run("raw108_group_refinement")
def check_refinement():
    return representation_refinement()


@run("ab_collision_example")
def check_ab():
    # root0 type 0; nodes 1,2 type 1; node 3 type 2; node 4 type 3; all single.
    types = [0, 1, 1, 2, 3]
    edges_a = [(0, 1, 1), (0, 2, 1), (1, 3, 1), (1, 4, 1)]
    edges_b = [(0, 1, 1), (0, 2, 1), (1, 3, 1), (2, 4, 1)]

    def build(edges):
        ei, et = [], []
        for a, b, t in edges:
            ei += [(a, b), (b, a)]
            et += [t, t]
        return build_object(np.asarray(types, dtype=np.int64), np.asarray(ei).T, np.asarray(et))

    A, Bb = build(edges_a), build(edges_b)
    raw_equal = bool(np.array_equal(np.asarray(A.atom_feat)[0], np.asarray(Bb.atom_feat)[0]))
    assert raw_equal
    en_a = B.enumerate_rooted_branching(A)
    en_b = B.enumerate_rooted_branching(Bb)
    counts = {
        "A": {f: int(sum(en_a[fi][0].values())) for fi, f in enumerate(B.FAMILIES)},
        "B": {f: int(sum(en_b[fi][0].values())) for fi, f in enumerate(B.FAMILIES)},
    }
    assert counts["A"] == {"S3": 0, "S4": 0, "F4": 1}, counts
    assert counts["B"] == {"S3": 0, "S4": 0, "F4": 0}, counts
    # explicit independent two-hop / degree statistics of root 0
    stats = {}
    for name, edges in (("A", edges_a), ("B", edges_b)):
        adj = defaultdict(set)
        for a, b, _ in edges:
            adj[a].add(b); adj[b].add(a)
        dist2 = set()
        for u in adj[0]:
            for w in adj[u]:
                if w != 0 and w not in adj[0]:
                    dist2.add(w)
        stats[name] = {
            "degree": len(adj[0]),
            "neighbour_types": sorted(int(types[u]) for u in adj[0]),
            "two_hop_types": sorted(int(types[w]) for w in dist2),
            "two_hop_paths": int(sum(len(adj[u]) - 1 for u in adj[0])),
        }
    assert stats["A"] == stats["B"], (stats,)
    # bond/junction channels of the whole molecule are NOT identical
    def obj_multiset(obj):
        return {
            "n_bonds": int(len(obj.bond_index)),
            "n_junctions": int(len(obj.junc_index)),
            "bond_feat_rows": sorted(map(tuple, np.round(np.asarray(obj.bond_feat), 6).tolist())),
            "junc_feat_rows": sorted(map(tuple, np.round(np.asarray(obj.junc_feat), 6).tolist())),
        }
    ma, mb = obj_multiset(A), obj_multiset(Bb)
    bond_same = ma["bond_feat_rows"] == mb["bond_feat_rows"]
    junc_same = ma["junc_feat_rows"] == mb["junc_feat_rows"]
    return {
        "root_raw108_equal": raw_equal,
        "root_stats": stats,
        "family_counts": counts,
        "f4_pattern_A": [
            {"atoms": np.asarray(np.frombuffer(k, dtype=np.uint8)[:5]).astype(int).tolist(),
             "pairs": np.frombuffer(k, dtype=np.uint8)[5:].astype(int).tolist(),
             "count": int(c)}
            for k, c in en_a[2][0].items()
        ],
        "whole_graph_bond_junction": {
            "n_bonds": [ma["n_bonds"], mb["n_bonds"]],
            "n_junctions": [ma["n_junctions"], mb["n_junctions"]],
            "bond_feat_multiset_equal": bond_same,
            "junc_feat_multiset_equal": junc_same,
        },
        "interpretation": (
            "root raw108 collides and F4 separates A/B, but the two whole graphs "
            "have different bond/junction objects; this is a root-descriptor "
            "collision, not a proof that the full P3 is blind to the difference"
        ),
    }


@run("train_valid_new_content")
def check_coverage():
    train = B.load_cache("train")
    valid = B.load_cache("valid")
    def key_set(pc):
        pat_atom = np.asarray(pc["pat_atom"]); pat_pair = np.asarray(pc["pat_pair"])
        fam = np.asarray(pc["pat_family"])
        return {
            f: {bytes([f]) + bytes(pat_atom[p]) + bytes(pat_pair[p])
                for p in range(len(pat_atom)) if int(fam[p]) == f}
            for f in range(len(B.FAMILIES))
        }
    tr = key_set(train)
    va = key_set(valid)
    per_family = {}
    for fi, f in enumerate(B.FAMILIES):
        unseen = va[fi] - tr[fi]
        pat_family = np.asarray(valid["pat_family"])
        ent_pat = np.asarray(valid["ent_pat"])
        ent_count = np.asarray(valid["ent_count"])
        pat_atom = np.asarray(valid["pat_atom"]); pat_pair = np.asarray(valid["pat_pair"])
        occ_total = 0.0
        occ_unseen = 0.0
        for i in range(len(ent_pat)):
            p = int(ent_pat[i])
            if int(pat_family[p]) != fi:
                continue
            c = float(ent_count[i])
            occ_total += c
            if bytes([fi]) + bytes(pat_atom[p]) + bytes(pat_pair[p]) in unseen:
                occ_unseen += c
        per_family[f] = {
            "train_unique_patterns": len(tr[fi]),
            "valid_unique_patterns": len(va[fi]),
            "valid_unseen_patterns": len(unseen),
            "valid_unseen_pattern_fraction": float(len(unseen) / max(len(va[fi]), 1)),
            "valid_occurrence_unseen_fraction": float(occ_unseen / max(occ_total, 1)),
            "valid_occurrences": int(occ_total),
        }
    # per-root full-signature unseen fraction
    tr_dig = root_signature_digests(train, int(train["meta"]["n_molecules"]))
    va_dig = root_signature_digests(valid, int(valid["meta"]["n_molecules"]))
    tr_sig = {v[0] for v in tr_dig.values()}
    n_roots = len(va_dig)
    n_unseen_sig = sum(1 for v in va_dig.values() if v[0] not in tr_sig)
    # roots with >=1 unseen family-pattern
    pat_atom = np.asarray(valid["pat_atom"]); pat_pair = np.asarray(valid["pat_pair"])
    pat_family = np.asarray(valid["pat_family"])
    ent_ptr = np.asarray(valid["ent_ptr"]); ent_pat = np.asarray(valid["ent_pat"])
    ent_root = np.asarray(valid["ent_root"])
    unseen_root = 0
    for m in range(int(valid["meta"]["n_molecules"])):
        s, e = int(ent_ptr[m]), int(ent_ptr[m + 1])
        roots = set()
        for i in range(s, e):
            p = int(ent_pat[i]); f = int(pat_family[p])
            if bytes([f]) + bytes(pat_atom[p]) + bytes(pat_pair[p]) in (va[f] - tr[f]):
                roots.add(int(ent_root[i]))
        unseen_root += len(roots)
    return {
        "per_family": per_family,
        "valid_roots_full_signature_unseen": int(n_unseen_sig),
        "valid_roots_total": int(n_roots),
        "valid_root_full_signature_unseen_fraction": float(n_unseen_sig / max(n_roots, 1)),
        "valid_roots_with_at_least_one_unseen_pattern": int(unseen_root),
        "valid_root_any_unseen_fraction": float(unseen_root / max(n_roots, 1)),
        "note": "descriptive only; low new-content rate is not evidence of sufficiency",
    }


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    B.CACHE.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        if not B.cache_path(split).exists():
            B.save_split(split)
    # run in a fixed order
    for fn in (
        check_reference, check_parameters, check_zero_columns, check_hand,
        check_permutation_enum, check_permutation_prediction, check_batch_regrouping,
        check_multiplicity, check_gradients, check_cache, check_refinement,
        check_ab, check_coverage,
    ):
        fn()
        name = getattr(fn, "check_name", fn.__name__)
        C.log(f"[bf-check] {name}: {'pass' if RESULTS[name]['pass'] else 'FAIL'}")
        if not RESULTS[name]["pass"]:
            C.log(RESULTS[name].get("traceback", RESULTS[name].get("error", "")))
    heavy = {"raw108_group_refinement"}
    out = {"checks": {}}
    for name, value in RESULTS.items():
        if name in heavy and value.get("pass"):
            detail = dict(value["detail"])
            for key in ("examples", "examples_content_split", "examples_multiplicity_split"):
                detail.pop(key, None)
            out["checks"][name] = {"pass": True, "detail": detail}
        else:
            out["checks"][name] = value
    cache_stats = {}
    for split in ("train", "valid"):
        pc = B.load_cache(split)
        cache_stats[split] = {
            "path": str(B.cache_path(split).relative_to(C.REPO)),
            "size_bytes": int(B.cache_path(split).stat().st_size),
            "sha256": C.sha256_file(B.cache_path(split)),
            "meta": pc["meta"],
        }
    C.write_json(ANALYSIS / "cache_stats.json", cache_stats)
    representation = {
        "refinement": RESULTS["raw108_group_refinement"]["detail"]
        if RESULTS["raw108_group_refinement"]["pass"] else RESULTS["raw108_group_refinement"],
        "ab_example": RESULTS["ab_collision_example"]["detail"]
        if RESULTS["ab_collision_example"]["pass"] else RESULTS["ab_collision_example"],
        "train_valid_content": RESULTS["train_valid_new_content"]["detail"]
        if RESULTS["train_valid_new_content"]["pass"] else RESULTS["train_valid_new_content"],
    }
    (ANALYSIS / "representation_check.json").write_text(
        json.dumps(representation, separators=(",", ":"), default=C._json_default) + "\n",
        encoding="utf-8",
    )
    C.write_json(ANALYSIS / "cpu_checks.json", out)
    n_fail = sum(1 for r in RESULTS.values() if not r["pass"])
    C.log(f"[bf-check] {len(RESULTS) - n_fail} passed / {n_fail} failed")
    C.log(json.dumps({k: v["pass"] for k, v in RESULTS.items()}, indent=1))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
