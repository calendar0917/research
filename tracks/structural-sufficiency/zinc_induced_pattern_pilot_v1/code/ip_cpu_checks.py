"""CPU-only checks for the rooted induced-path pilot.

The checks are deliberately small and diagnostic, not a mirrored test suite.
They cover the pre-registered correctness items: enumeration/multiplicity,
six-ring vs chain closure, renumbering, batch isolation, compressed/expanded
multiplicity equivalence, zero-increment P3 equivalence, SSL loading,
optimizer coverage, gradient opening after the zero-initialised projection,
and finite forward behaviour with no paths.
"""

from __future__ import annotations

import json
import sys
import traceback
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
import ip_data  # noqa: E402
import ip_models  # noqa: E402

RESULTS = {}


def run(name):
    def deco(fn):
        try:
            RESULTS[name] = {"pass": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            RESULTS[name] = {
                "pass": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            }
        return fn
    return deco


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


def permutation(obj, seed=0):
    rng = np.random.default_rng(seed)
    new_order = rng.permutation(int(obj.n_atoms))  # new_order[new] = old
    old_to_new = np.empty(int(obj.n_atoms), dtype=np.int64)
    old_to_new[new_order] = np.arange(int(obj.n_atoms), dtype=np.int64)
    return SimpleNamespace(
        n_atoms=int(obj.n_atoms),
        atom_feat=np.asarray(obj.atom_feat)[new_order],
        bond_index=old_to_new[np.asarray(obj.bond_index)],
        bond_type=np.asarray(obj.bond_type).copy(),
        bond_feat=np.asarray(obj.bond_feat).copy(),
        junc_feat=np.asarray(obj.junc_feat).copy(),
        junc_index=old_to_new[np.asarray(obj.junc_index).reshape(-1, 3)].reshape(-1, 3),
        junc_bond=np.asarray(obj.junc_bond).copy(),
        global_feat=np.asarray(obj.global_feat).copy(),
    )


def eval_p3_valid():
    norm = D.load_norm()
    valid = D.load_cache("valid")
    rows = np.arange(len(valid["labels"]), dtype=np.int64)
    model, _ = A2.build_model("P3N", norm, 0, "cpu")
    path = _V2.parent / "runs" / "xBssl_P3N_s0" / "best.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    C.load_state(model, ck["state_dict"], "cpu")
    pred, mae = T.evaluate(model, valid, rows, candidate="P3N", device="cpu", batch_size=256)
    return model, valid, rows, pred, mae, ck


@run("reference_audit")
def check_reference():
    train_json = _V2.parent / "runs" / "xBssl_P3N_s0" / "train.json"
    info = C.read_json(train_json)
    _, _, _, _, mae, ck = eval_p3_valid()
    expected = 0.07622375113883754
    ssl = info["model_info"]["ssl"]
    assert info["parameter_count"] == 413505
    assert info["recipe_variant"] == "B"
    assert info["recipe"]["epochs"] == 600 and info["recipe"]["steps_per_epoch"] == 79
    assert info["best_valid"]["epoch"] == 377
    assert abs(info["best_valid"]["valid_mae"] - expected) < 1e-12
    assert abs(mae - expected) < 1e-6, (mae, expected)
    assert ck["epoch"] == 377
    assert ssl["sha256"] == C.sha256_file(C.P3_SSL_S0)
    return {
        "checkpoint_epoch": int(ck["epoch"]),
        "replay_valid_mae": float(mae),
        "stored_valid_mae": float(info["best_valid"]["valid_mae"]),
        "abs_diff": float(abs(mae - expected)),
        "ssl_source": ssl["source"],
        "ssl_sha256": ssl["sha256"],
        "recipe": {k: info["recipe"][k] for k in ("optimizer", "lr", "min_lr", "weight_decay",
                                                   "warmup_steps", "epochs", "batch_size",
                                                   "steps_per_epoch", "grad_clip")},
        "train_label_sha256": info["data"]["train_label_sha256"],
        "valid_label_sha256": info["data"]["valid_label_sha256"],
    }


@run("p3_null_grouping_semantics")
def check_p3_null_grouping():
    """Verify the corrected reading: P3 NULL uses graph means x local n_b."""
    norm = D.load_norm()
    valid = D.load_cache("valid")
    rows = np.arange(8, dtype=np.int64)
    model, _ = A2.build_model("P3N", norm, 0, "cpu")
    batch = T.collate(valid, rows, "P3N")
    with torch.no_grad():
        a, b, w = model.encoders(batch)
        W, V, n_b = model.group_to_bonds(w, batch)
        n_graphs = int(batch["n_graphs"])
        w_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w)
        cnt = torch.bincount(batch["junc_batch"], minlength=n_graphs).clamp(min=1).to(w.dtype)
        mean_w = w_sum / cnt.unsqueeze(1)
        expected = n_b.unsqueeze(1) * mean_w[batch["bond_batch"]]
        graph_mean_diff = float((W - expected).abs().max())
        local = w.new_zeros((W.shape[0], w.shape[1]))
        local.index_add_(0, batch["junc_bond"][:, 0], w)
        local.index_add_(0, batch["junc_bond"][:, 1], w)
        local_vs_null_diff = float((W - local).abs().max())
    assert graph_mean_diff < 1e-6, graph_mean_diff
    assert local_vs_null_diff > 1e-3, local_vs_null_diff
    return {
        "n_bonds": int(W.shape[0]),
        "max_abs_W_minus_n_b_times_graph_mean": graph_mean_diff,
        "max_abs_W_minus_local_junction_sum": local_vs_null_diff,
        "interpretation": "P3 NULL is n_b * graph-wide mean(w), not the sum of junctions touching the bond",
    }


@run("parameter_and_optimizer_audit")
def check_parameters():
    norm = D.load_norm()
    j, info = ip_models.build_model("J", norm, 0, "cpu")
    q, infoq = ip_models.build_model("Q", norm, 0, "cpu")
    assert info["parameters"] == infoq["parameters"] == 484281
    assert info["parameters"] <= 500_000
    for model in (j, q):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        total = C.parameter_count(model)
        opt_count = C.optimizer_parameter_count(opt)
        assert total == opt_count, (total, opt_count)
        assert all(p.requires_grad for p in model.parameters())
    return {
        "J": info["parameters_breakdown"],
        "Q": infoq["parameters_breakdown"],
        "cap": 500_000,
        "headroom": 500_000 - info["parameters"],
    }


@run("initial_state_and_ssl")
def check_initial_state():
    norm = D.load_norm()
    torch.manual_seed(0)
    p3, _ = A2.build_model("P3N", norm, 0, "cpu")
    j, _ = ip_models.build_model("J", norm, 0, "cpu")
    sp, sj = p3.state_dict(), j.state_dict()
    diffs = {k: float((sp[k].float() - sj[k].float()).abs().max()) for k in sp}
    assert max(diffs.values()) == 0.0, max(diffs.values())
    ssl = torch.load(C.P3_SSL_S0, map_location="cpu", weights_only=False)
    enc = {f"encoders.{name}.{layer}": tensor for name, layers in ssl.items()
           for layer, tensor in layers.items()}
    enc_diffs = {}
    for k, v in enc.items():
        enc_diffs[k] = float((j.state_dict()[k].float() - v.float()).abs().max())
    assert max(enc_diffs.values()) == 0.0, max(enc_diffs.values())
    assert j.training and j.path_branch.training
    j.eval()
    assert not j.training and not j.path_branch.training
    return {
        "p3_tensors_compared": len(sp),
        "p3_max_abs_diff": max(diffs.values()),
        "ssl_tensors_compared": len(enc_diffs),
        "ssl_max_abs_diff": max(enc_diffs.values()),
        "train_eval_switch_ok": True,
    }


@run("zero_increment_equivalence")
def check_zero():
    norm = D.load_norm()
    valid = D.load_cache("valid")
    rows = np.arange(16, dtype=np.int64)
    torch.manual_seed(0)
    p3, _ = A2.build_model("P3N", norm, 0, "cpu")
    j, _ = ip_models.build_model("J", norm, 0, "cpu")
    q, _ = ip_models.build_model("Q", norm, 0, "cpu")
    bj = ip_data.collate_paths(valid, rows, "J")
    bq = ip_data.collate_paths(valid, rows, "Q")
    p3.eval(); j.eval(); q.eval()
    with torch.no_grad():
        y3 = p3(bj)
        yj = j(bj)
        yq = q(bq)
    return {
        "max_abs_J_minus_P3": float((yj - y3).abs().max()),
        "max_abs_Q_minus_P3": float((yq - y3).abs().max()),
        "finite": bool(torch.isfinite(yj).all() and torch.isfinite(yq).all()),
    }


@run("enumeration_multiplicity")
def check_enumeration():
    square = fake_obj([0, 0, 0, 0], [(0, 1), (1, 2), (2, 3), (3, 0)])
    p = ip_data.enumerate_rooted_paths(square)
    totals = {l: sum(sum(d.values()) for d in p[l].values()) for l in ip_data.LENGTHS}
    assert totals == {2: 8, 3: 8, 4: 0, 5: 0}, totals
    # root 0 has two distinct length-2 paths to the two neighbours, both with
    # the same raw typed content (all C, no 0-2 edge): one pattern, count 2.
    assert len(p[2]) == 1
    only = next(iter(p[2].values()))
    assert only == {0: 2, 1: 2, 2: 2, 3: 2}, only
    # all length-2 patterns must have consecutive bond types 1 and no 0-2 edge
    key = next(iter(p[2]))
    pair = np.frombuffer(key[6:], dtype=np.uint8)
    assert pair[0] == 1 and pair[1] == 0 and pair[5] == 1, pair
    return {"totals_by_length": totals, "unique_length2_patterns": len(p[2]),
            "root0_multiplicity": int(only[0]), "pattern_pair_values": pair.tolist()}


@run("six_ring_vs_chain")
def check_ring():
    ring = fake_obj([0, 0, 0, 0, 0, 0], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    chain = fake_obj([0, 0, 0, 0, 0, 0], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    pr = ip_data.enumerate_rooted_paths(ring)[5]
    pc = ip_data.enumerate_rooted_paths(chain)[5]
    assert sum(sum(d.values()) for d in pr.values()) == 12
    assert sum(sum(d.values()) for d in pc.values()) == 2
    # The clockwise length-5 ring trace 0-1-2-3-4-5 and the chain trace have
    # identical atom/sequence labels; only the extra edge (0,5) distinguishes.
    ring_key = next(iter(pr))
    chain_key = next(iter(pc))
    assert ring_key != chain_key
    rp = np.frombuffer(ring_key[6:], dtype=np.uint8)
    cp = np.frombuffer(chain_key[6:], dtype=np.uint8)
    idx = ip_data.PAIR_INDEX.index((0, 5))
    assert rp[idx] == 1, rp  # pair (0,5) is the ring closure
    assert cp[idx] == 0, cp
    assert ip_data.q_key_from_key(ring_key) == ip_data.q_key_from_key(chain_key)
    # Q cache table should have the two extra-edge variants collapsed.
    assert len(pr) == 1 and len(pc) == 1
    return {"ring_length5_occurrences": 12, "chain_length5_occurrences": 2,
            "ring_extra_pair_0_5": int(rp[idx]), "chain_extra_pair_0_5": int(cp[idx]),
            "q_keys_equal": True}


@run("renumbering_equivariance")
def check_renumbering():
    objs, _ = D.load_p3_objects("valid")
    obj = objs[0]
    perm_obj = permutation(obj, seed=7)
    orig_cache = cache_from_obj(obj)
    perm_cache = cache_from_obj(perm_obj)
    po = ip_data.build_from_objects([obj], "__temp__", {"source": "check"}, verbose=False)
    pp = ip_data.build_from_objects([perm_obj], "__temp__", {"source": "check"}, verbose=False)
    ip_data._PATH_CACHE["__temp__"] = po
    bo = ip_data.collate_paths(orig_cache, np.array([0]), "J")
    ip_data._PATH_CACHE["__temp__"] = pp
    bp = ip_data.collate_paths(perm_cache, np.array([0]), "J")
    norm = D.load_norm()
    torch.manual_seed(0)
    model, _ = ip_models.build_model("J", norm, 0, "cpu")
    with torch.no_grad():
        # open the zero projection so the path branch is actually exercised
        model.path_branch.proj.lins[-1].weight.normal_(0.0, 0.02)
    model.eval()
    with torch.no_grad():
        yo = model(bo)
        yp = model(bp)
    return {"abs_pred_diff": float((yo - yp).abs().max()),
            "finite": bool(torch.isfinite(yo).all() and torch.isfinite(yp).all()),
            "n_atoms": int(obj.n_atoms)}


@run("batch_isolation")
def check_batch_isolation():
    valid = D.load_cache("valid")
    norm = D.load_norm()
    torch.manual_seed(0)
    model, _ = ip_models.build_model("J", norm, 0, "cpu")
    with torch.no_grad():
        model.path_branch.proj.lins[-1].weight.normal_(0.0, 0.02)
    model.eval()
    rows2 = np.array([0, 1], dtype=np.int64)
    b2 = ip_data.collate_paths(valid, rows2, "J")
    b1 = ip_data.collate_paths(valid, np.array([0], dtype=np.int64), "J")
    with torch.no_grad():
        d2 = model.path_branch(b2)
        d1 = model.path_branch(b1)
        y2 = model(b2)
        y1 = model(b1)
    n0 = int((np.diff(np.asarray(valid["atom_ptr"]))[0]))
    d_diff = float((d2[:n0] - d1).abs().max())
    y_diff = float((y2[0] - y1[0]).abs().max())
    assert y_diff < 1e-5, y_diff
    return {"d_root_max_abs_diff": d_diff, "pred_max_abs_diff": y_diff}


@run("compression_expansion")
def check_expansion():
    valid = D.load_cache("valid")
    b = ip_data.collate_paths(valid, np.array([0], dtype=np.int64), "J")
    norm = D.load_norm()
    torch.manual_seed(0)
    model, _ = ip_models.build_model("J", norm, 0, "cpu")
    with torch.no_grad():
        model.path_branch.proj.lins[-1].weight.normal_(0.0, 0.02)
    model.eval()
    with torch.no_grad():
        d0 = model.path_branch(b)
    b2 = dict(b)
    # Split every occurrence count c into two entries floor(c/2), ceil(c/2).
    c = b["occ_count"]
    lo = torch.floor(c / 2).long()
    hi = c.long() - lo
    keep = c > 0
    def dup(x):
        return torch.cat([x[keep], x[keep]], dim=0)
    pat = dup(b["occ_pat"]); root = dup(b["occ_root"]); ln = dup(b["occ_len"])
    cnt = torch.cat([lo[keep], hi[keep]], dim=0).float()
    b2["occ_pat"] = pat; b2["occ_root"] = root; b2["occ_len"] = ln; b2["occ_count"] = cnt
    with torch.no_grad():
        d1 = model.path_branch(b2)
    return {"max_abs_diff": float((d0 - d1).abs().max()),
            "original_entries": int(c.numel()), "expanded_entries": int(cnt.numel())}


@run("gradient_flow")
def check_gradients():
    valid = D.load_cache("valid")
    norm = D.load_norm()
    rows = np.arange(8, dtype=np.int64)
    b = ip_data.collate_paths(valid, rows, "J")
    y = torch.as_tensor(np.asarray(valid["labels"])[rows], dtype=torch.float32)
    model, _ = ip_models.build_model("J", norm, 0, "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    phi = model.path_branch.phi.lins[0].weight
    proj = model.path_branch.proj.lins[-1].weight
    records = []
    for step in range(1, 7):
        opt.zero_grad(set_to_none=True)
        pred = model(b)
        loss = F.l1_loss(pred, y)
        loss.backward()
        g_phi = float(phi.grad.abs().max()) if phi.grad is not None else 0.0
        g_proj = float(proj.grad.abs().max()) if proj.grad is not None else 0.0
        records.append({"step": step, "loss": float(loss), "grad_phi": g_phi, "grad_proj": g_proj})
        assert torch.isfinite(loss)
        opt.step()
    assert records[0]["grad_proj"] > 0
    assert records[0]["grad_phi"] == 0.0
    assert any(r["grad_phi"] > 0 for r in records[1:]), records
    return {"records": records, "opened_after_step": next(r["step"] for r in records if r["grad_phi"] > 0)}


@run("no_path_root")
def check_no_path():
    branch = ip_models.RootedPathBranch()
    branch.eval()
    n_atoms = 1
    batch = {
        "pat_atom": torch.zeros((0, 6), dtype=torch.long),
        "pat_pair": torch.zeros((0, 15), dtype=torch.long),
        "pat_len": torch.zeros((0,), dtype=torch.long),
        "pat_node_mask": torch.zeros((0, 6), dtype=torch.bool),
        "pat_pair_mask": torch.zeros((0, 15), dtype=torch.bool),
        "occ_pat": torch.zeros((0,), dtype=torch.long),
        "occ_len": torch.zeros((0,), dtype=torch.long),
        "occ_root": torch.zeros((0,), dtype=torch.long),
        "occ_count": torch.zeros((0,), dtype=torch.float32),
        "n_atoms_total": n_atoms,
    }
    with torch.no_grad():
        d = branch(batch)
    assert d.shape == (n_atoms, 96)
    assert torch.isfinite(d).all()
    assert torch.equal(d, torch.zeros_like(d))
    return {"shape": list(d.shape), "all_zero": True, "finite": True}


@run("cache_integrity")
def check_cache():
    for split in ("train", "valid"):
        pc = ip_data.load_path_cache(split)
        meta = pc["meta"]
        assert meta["labels_used"] is False and meta["test_used"] is False
        assert tuple(meta["lengths"]) == (2, 3, 4, 5)
        ent_ptr = np.asarray(pc["ent_ptr"])
        assert len(ent_ptr) - 1 == meta["n_molecules"]
        ent_count = np.asarray(pc["ent_count"])
        assert int(ent_count.min()) >= 1
        # Every consecutive pair of every pattern must be a real bond type.
        for suffix in ("j", "q"):
            pat_len = np.asarray(pc[f"pat_len_{suffix}"])
            pat_pair = np.asarray(pc[f"pat_pair_{suffix}"])
            for k, (i, j) in enumerate(ip_data.PAIR_INDEX):
                if j == i + 1:
                    vals = pat_pair[pat_len >= j, k]
                    assert len(vals) == 0 or np.isin(vals, (1, 2, 3)).all(), (suffix, i, j)
    assert not (ip_data.CACHE / "path_test.pt").exists()
    return {"splits": ["train", "valid"], "test_cache_exists": False,
            "train_patterns_J": int(len(ip_data.load_path_cache("train")["pat_len_j"])),
            "valid_patterns_J": int(len(ip_data.load_path_cache("valid")["pat_len_j"]))}


def main() -> int:
    # Run in a deterministic order; each check is independent.
    out = {"checks": RESULTS}
    path = _HERE.parent / "analysis" / "cpu_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(path, out)
    C.log(json.dumps(RESULTS, indent=2)[:12000])
    n_fail = sum(1 for r in RESULTS.values() if not r["pass"])
    C.log(f"CPU checks: {len(RESULTS) - n_fail} passed / {n_fail} failed -> {path}")
    C.write_json(path, out)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
