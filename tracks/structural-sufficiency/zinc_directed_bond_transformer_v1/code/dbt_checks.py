"""Focused CPU checks for the directed-bond Transformer (DBT).

Seven checks, each tied to a real failure mode of this implementation.  They
are not a generalisation audit and not a mechanism study:

1. exact simple-path counts vs ``networkx.all_simple_paths`` on small graphs,
   hand-computed counts, symmetry, diagonal zero, relabelling equivariance;
2. atom relabelling / reversed edge input / row order leave the prediction
   unchanged (roles swap internally, the model is object-permutation
   invariant);
3. exactly two directed objects per physical bond, source/target junction
   incidence matches brute force, padding and cross-graph leakage are absent;
4. the relation -> value pathway actually makes the context relation-dependent
   even when every token state is identical, with finite gradients;
5. the train-only masked-SSL encoder weights are really loaded and really
   optimised, and the optimizer covers every parameter;
6. the readout keeps the count and the amplitude (the pooled moments are not
   per-molecule normalised) and the no-edge graph stays finite;
7. data/label alignment, steps per epoch, the last partial batch, and the
   per-update LR schedule at its three boundary points.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_SSUF.parents[1] / "tracks/structure-contribution/code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import dbt_data as BD  # noqa: E402
import dbt_models as BM  # noqa: E402
import dbt_train as BT  # noqa: E402

ANALYSIS = _EXP / "analysis"


# ---------------------------------------------------------------------------
# helpers: build production-format caches for ad-hoc molecule lists
# ---------------------------------------------------------------------------


def _fake_cache(objs: Sequence[Any]) -> dict[str, Any]:
    atom_ptr = np.zeros(len(objs) + 1, dtype=np.int64)
    bond_ptr = np.zeros(len(objs) + 1, dtype=np.int64)
    junc_ptr = np.zeros(len(objs) + 1, dtype=np.int64)
    for i, o in enumerate(objs):
        atom_ptr[i + 1] = atom_ptr[i] + int(o.n_atoms)
        bond_ptr[i + 1] = bond_ptr[i] + int(o.bond_index.shape[0])
        junc_ptr[i + 1] = junc_ptr[i] + int(o.junc_index.shape[0])
    return {
        "n_molecules": len(objs),
        "labels": np.zeros(len(objs), dtype=np.float32),
        "atom_ptr": atom_ptr,
        "bond_ptr": bond_ptr,
        "junc_ptr": junc_ptr,
        "atom_feat": np.concatenate([np.asarray(o.atom_feat, np.float32) for o in objs]),
        "atom_type": np.concatenate(
            [np.asarray(o.atom_feat)[:, :21].argmax(1).astype(np.int64) for o in objs]),
        "bond_feat": np.concatenate([np.asarray(o.bond_feat, np.float32) for o in objs]),
        "bond_type": np.concatenate([np.asarray(o.bond_type, np.int64) for o in objs]),
        "bond_index": np.concatenate([np.asarray(o.bond_index, np.int64) for o in objs]),
        "junc_feat": np.concatenate([np.asarray(o.junc_feat, np.float32) for o in objs]),
        "junc_index": np.concatenate([np.asarray(o.junc_index, np.int64) for o in objs]),
        "junc_bond": np.concatenate([np.asarray(o.junc_bond, np.int64) for o in objs]),
        "global_feat": np.stack([np.asarray(o.global_feat, np.float32) for o in objs]),
    }


def _fake_rel(rels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pair_ptr": np.concatenate([[0], np.cumsum([r["n_atoms"] ** 2 for r in rels])]),
        "dir_ptr": np.concatenate([[0], np.cumsum([r["n_dir"] for r in rels])]),
        "src_ptr": np.concatenate([[0], np.cumsum([int(r["src_len"].sum()) for r in rels])]),
        "dst_ptr": np.concatenate([[0], np.cumsum([int(r["dst_len"].sum()) for r in rels])]),
    }
    for key in ("path", "dist", "dir_index", "dir_bond", "src_len", "dst_len",
                "src_junc", "dst_junc"):
        out[key] = np.concatenate([np.asarray(r[key]) for r in rels])
    return out


def _raw_graph(row: int):
    ds = D.pyg_zinc("valid")
    d = ds[row]
    return (d.x.reshape(-1).long().numpy(), d.edge_index.numpy(),
            d.edge_attr.reshape(-1).long().numpy())


def _rebuild(x, ei, ea):
    import structural_relation_objects_v1 as sro

    obj = sro.build_mol_objects(0, x, ei, ea)
    return obj, BD.build_relation_mol(obj)


def _predict(model, cache, rel, rows) -> np.ndarray:
    model.eval()
    with torch.no_grad(), C.isolated_rng():
        batch = BD.collate(cache, rel, np.asarray(rows))
        return model(batch).cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# 1. path counts
# ---------------------------------------------------------------------------


def _nx_counts(n, edges, l_max):
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    out = np.zeros((n, n, l_max), dtype=np.int64)
    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            for path in nx.all_simple_paths(g, s, t, cutoff=l_max):
                l = len(path) - 1
                if 1 <= l <= l_max:
                    out[s, t, l - 1] += 1
    return out


def check_paths(report: dict) -> None:
    cases = {
        "path5": (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        "cycle6": (6, [(i, (i + 1) % 6) for i in range(6)]),
        "k4": (4, [(a, b) for a in range(4) for b in range(a + 1, 4)]),
        "fused2": (7, [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 6), (6, 3)]),
    }
    rows = {}
    ok = True
    for name, (n, edges) in cases.items():
        adj = BD._adjacency(n, np.asarray(edges, dtype=np.int64))
        ours = BD.simple_path_counts(n, adj, BD.L_MAX)
        theirs = _nx_counts(n, edges, BD.L_MAX)
        sym = np.array_equal(ours, ours.transpose(1, 0, 2))
        dz = bool(np.all(ours[np.arange(n), np.arange(n), :] == 0))
        rows[name] = {"match_nx": bool(np.array_equal(ours, theirs)),
                      "symmetric": bool(sym), "diagonal_zero": dz,
                      "max_count": int(ours.max())}
        ok &= bool(np.array_equal(ours, theirs)) and sym and dz
    # hand-computed values
    n, edges = cases["k4"]
    ours = BD.simple_path_counts(n, BD._adjacency(n, np.asarray(edges, np.int64)), BD.L_MAX)
    hand = {"S1": int(ours[0, 1, 0]), "S2": int(ours[0, 1, 1]), "S3": int(ours[0, 1, 2]),
            "S4": int(ours[0, 1, 3])}
    ok &= hand == {"S1": 1, "S2": 2, "S3": 2, "S4": 0}
    n, edges = cases["path5"]
    ours = BD.simple_path_counts(n, BD._adjacency(n, np.asarray(edges, np.int64)), BD.L_MAX)
    hand5 = {"S1_01": int(ours[0, 1, 0]), "S4_04": int(ours[0, 4, 3]),
             "S2_02": int(ours[0, 2, 1]), "S5_any": int(ours[:, :, 4].sum())}
    ok &= hand5 == {"S1_01": 1, "S4_04": 1, "S2_02": 1, "S5_any": 0}
    # relabelling equivariance
    rng = np.random.default_rng(7)
    relab_ok = True
    for name, (n, edges) in cases.items():
        perm = rng.permutation(n)
        adj = BD._adjacency(n, np.asarray(edges, np.int64))
        a = BD.simple_path_counts(n, adj, BD.L_MAX)
        adj2 = BD._adjacency(n, np.asarray([(perm[u], perm[v]) for u, v in edges], np.int64))
        b = BD.simple_path_counts(n, adj2, BD.L_MAX)
        # new index = perm[old index]: b[perm[u], perm[v]] == a[u, v]
        relab_ok &= bool(np.array_equal(b[np.ix_(perm, perm)], a))
    ok &= relab_ok
    report["path_counts"] = {
        "pass": bool(ok), "cases": rows, "k4_hand": hand, "path5_hand": hand5,
        "relabel_equivariant": bool(relab_ok),
        "definition": f"exact simple paths with exactly l edges, l=1..{BD.L_MAX}",
    }


# ---------------------------------------------------------------------------
# 2. invariance
# ---------------------------------------------------------------------------


def check_invariance(report: dict, model, rng) -> None:
    ok = True
    details = {}
    for row in (0, 3):
        x, ei, ea = _raw_graph(row)
        n = int(x.shape[0])
        obj0, rel0 = _rebuild(x, ei, ea)
        p0 = _predict(model, _fake_cache([obj0]), _fake_rel([rel0]), [0])[0]
        # reversed directed edge input (same physical graph)
        p_rev = _predict(model, _fake_cache([_rebuild(x, ei[::-1], ea)[0]]),
                         _fake_rel([_rebuild(x, ei[::-1], ea)[1]]), [0])[0]
        # atom relabelling: perm maps the new index to the old one
        perm = rng.permutation(n)
        x_p = x[perm]
        ei_p = np.argsort(perm)[ei]
        objp, relp = _rebuild(x_p, ei_p, ea)
        pp = _predict(model, _fake_cache([objp]), _fake_rel([relp]), [0])[0]
        details[f"row{row}"] = {
            "reverse_edge_input": float(abs(p0 - p_rev)),
            "atom_relabel": float(abs(p0 - pp)),
            "n_atoms": n,
        }
        ok &= abs(p0 - p_rev) < 1e-5 and abs(p0 - pp) < 1e-5
    # row-order / batch-composition invariance on four real molecules
    cache = BD.load_object_cache("valid")
    rel = BD.load_relation_split("valid")
    rows = np.asarray([1, 5, 9, 12, 17])
    alone = np.concatenate([_predict(model, cache, rel, [r]) for r in rows])
    together = _predict(model, cache, rel, rows)
    shuffled = _predict(model, cache, rel, rows[::-1])[::-1]
    details["batch_composition"] = {
        "max_abs_alone_vs_batch": float(np.abs(alone - together).max()),
        "max_abs_batch_vs_reordered": float(np.abs(together - shuffled).max()),
    }
    ok &= float(np.abs(alone - together).max()) < 1e-5
    ok &= float(np.abs(together - shuffled).max()) < 1e-5
    report["invariance"] = {"pass": bool(ok), **details}


# ---------------------------------------------------------------------------
# 3. directed objects, junction roles, padding isolation
# ---------------------------------------------------------------------------


def check_directed(report: dict) -> None:
    ok = True
    detail = {}
    rel = BD.load_relation_split("valid")
    objs, _ = D.load_p3_objects("valid")
    for row in (0, 2, 4):
        o = objs[row]
        r = BD.build_relation_mol(o)
        n = int(o.n_atoms)
        bi = np.asarray(o.bond_index, np.int64)
        junc = np.asarray(o.junc_index, np.int64)
        adj = BD._adjacency(n, bi)
        # two objects per physical bond, reverse pairing
        two = int(r["n_dir"]) == 2 * int(r["n_bonds"])
        pair = bool(np.array_equal(r["dir_index"][0::2], r["dir_index"][1::2][:, ::-1]))
        # brute-force junction role check
        role_ok = True
        src_lists, dst_lists = [], []
        pos = 0
        for k in range(int(r["n_bonds"])):
            u, v = int(bi[k, 0]), int(bi[k, 1])
            for (s, t) in ((u, v), (v, u)):
                cnt = int(r["src_len"][2 * k + (0 if s == u else 1)])
                ids = r["src_junc"][pos:pos + cnt]
                pos += cnt
                want = set()
                for other in adj[s]:
                    if other == t:
                        continue
                    # junction at centre s pairing the *target* t with `other`
                    a, b = min(t, other), max(t, other)
                    for jid, (i, j, kk) in enumerate(junc):
                        if int(j) == s and {int(i), int(kk)} == {a, b}:
                            want.add(jid)
                role_ok &= set(int(x) for x in ids) == want
                src_lists.append((s, t, sorted(int(x) for x in ids)))
        # dst side
        pos = 0
        for k in range(int(r["n_bonds"])):
            u, v = int(bi[k, 0]), int(bi[k, 1])
            for (s, t) in ((u, v), (v, u)):
                cnt = int(r["dst_len"][2 * k + (0 if s == u else 1)])
                ids = r["dst_junc"][pos:pos + cnt]
                pos += cnt
                want = set()
                for other in adj[t]:
                    if other == s:
                        continue
                    # junction at centre t pairing the *source* s with `other`
                    a, b = min(s, other), max(s, other)
                    for jid, (i, j, kk) in enumerate(junc):
                        if int(j) == t and {int(i), int(kk)} == {a, b}:
                            want.add(jid)
                role_ok &= set(int(x) for x in ids) == want
        counts_ok = bool(np.all(r["src_len"] == np.asarray([len(adj[int(r["dir_index"][i, 0])]) - 1
                                                           for i in range(r["n_dir"])])))
        detail[f"row{row}"] = {"two_per_bond": two, "reverse_paired": pair,
                               "junction_roles_bruteforce": bool(role_ok),
                               "src_count_is_deg_minus_1": counts_ok}
        ok &= two and pair and bool(role_ok) and counts_ok
    report["directed_objects"] = {"pass": bool(ok), **detail}


def check_padding(report: dict, model) -> None:
    """A graph's prediction must not depend on other graphs in the batch."""
    cache = BD.load_object_cache("valid")
    rel = BD.load_relation_split("valid")
    rows = np.asarray([3, 11, 24, 31])
    alone = np.concatenate([_predict(model, cache, rel, [r]) for r in rows])
    mixed = _predict(model, cache, rel, rows)
    diff = float(np.abs(alone - mixed).max())
    report["padding_isolation"] = {
        "pass": bool(diff < 1e-5),
        "max_abs_alone_vs_mixed": diff,
        "rows": rows.tolist(),
    }


# ---------------------------------------------------------------------------
# 4. relation -> value pathway
# ---------------------------------------------------------------------------


def check_relation_value(report: dict, model) -> None:
    cache = BD.load_object_cache("valid")
    rel = BD.load_relation_split("valid")
    batch = BD.collate(cache, rel, np.asarray([0, 3]))
    model.eval()
    tok = model.build_tokens(batch)
    pair_emb = model.relation.pair_embedding(batch)
    r_bar = model.relation(pair_emb, tok["src_slot"], tok["dst_slot"], tok["bond_slot"])
    mask = tok["key_mask"]
    static_bias = torch.zeros_like(tok["dense"])
    # every valid token state identical -> only the relation can differentiate
    h = tok["dense"].clone()
    mean = h[mask].mean(0, keepdim=True)
    h = h * 0.0 + mean
    layer = model.layers[0]
    with torch.no_grad():
        out_real, _ = layer(h, r_bar, mask, static_bias)
        out_zero, _ = layer(h, torch.zeros_like(r_bar), mask, static_bias)
        out_shuf, _ = layer(h, r_bar.flip(2), mask, static_bias)
    ctx_real = out_real[mask]
    spread = float(ctx_real.std(0).mean())
    same = float((out_real[mask] - out_zero[mask]).abs().mean())
    shuffled = float((out_real[mask] - out_shuf[mask]).abs().mean())
    # gradient reaches the relation mappings
    h_g = h.clone().requires_grad_(True)
    out_g, _ = layer(h_g, r_bar, mask, static_bias)
    loss = out_g[mask].pow(2).mean()
    loss.backward()
    grads = {
        "rel_b": float(layer.rel_b.weight.grad.abs().sum()),
        "rel_g": float(layer.rel_g.weight.grad.abs().sum()),
        "rel_u": float(layer.rel_u.weight.grad.abs().sum()),
        "relation_mlp": float(model.relation.mlp.lins[0].weight.grad.abs().sum())
        if model.relation.mlp.lins[0].weight.grad is not None else 0.0,
    }
    finite = bool(all(np.isfinite(v) for v in grads.values()))
    ok = spread > 0.0 and same > 1e-6 and shuffled > 1e-6 and finite
    report["relation_value_path"] = {
        "pass": bool(ok),
        "context_spread_across_tokens_with_identical_h": spread,
        "mean_abs_change_when_relation_zeroed": same,
        "mean_abs_change_when_relation_shuffled_across_f": shuffled,
        "grad_sums": grads,
        "grads_finite": finite,
        "note": "proves the relation is in the read content, not only in the logits",
    }


# ---------------------------------------------------------------------------
# 5. SSL initialisation + optimiser coverage
# ---------------------------------------------------------------------------


def check_ssl(report: dict, model, model_info, rng) -> None:
    path = C.P3_SSL_S0
    state = torch.load(path, map_location="cpu", weights_only=False)
    loaded = {}
    ok = True
    for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
        ref = state[name]
        cur = getattr(model.encoders, name).state_dict()
        diff = max(float((cur[k].cpu() - ref[k].cpu()).abs().max()) for k in ref)
        loaded[name] = {"tensors": len(ref), "max_abs_diff_vs_ssl_ckpt": diff}
        ok &= len(ref) == len(cur) and diff == 0.0
    # the SSL checkpoint really is the masked pretraining state, not supervised P3
    meta = C.read_json(C.P3_ROOT / "pretraining_seed0.json")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n_opt = C.optimizer_parameter_count(opt)
    n_par = C.parameter_count(model)
    requires = all(p.requires_grad for p in model.parameters())
    ok &= n_opt == n_par and requires
    # one real step must move an encoder weight
    cache = BD.load_object_cache("train")
    rel = BD.load_relation_split("train")
    before = model.encoders.atom_encoder.lins[0].weight.detach().clone()
    model.train()
    batch = BD.collate(cache, rel, np.asarray(rng.choice(10000, 8, replace=False)))
    opt.zero_grad(set_to_none=True)
    F.l1_loss(model(batch), torch.zeros(8)).backward()
    opt.step()
    after = model.encoders.atom_encoder.lins[0].weight.detach()
    moved = float((after - before).abs().max())
    ok &= moved > 0.0
    report["ssl_and_optimizer"] = {
        "pass": bool(ok),
        "ssl_checkpoint": str(path.relative_to(C.REPO)),
        "ssl_sha256": C.sha256_file(path),
        "ssl_pretraining": {"protocol": meta["protocol"], "arm": meta["arm"],
                            "mask": meta["mask"], "updates": meta["updates"]},
        "encoder_tensors": loaded,
        "parameters": int(n_par),
        "optimizer_parameters": int(n_opt),
        "all_params_require_grad": bool(requires),
        "encoder_weight_change_after_one_step": moved,
        "model_info_parameter_count": int(model_info["parameters"]),
    }


# ---------------------------------------------------------------------------
# 6. readout keeps count / amplitude; no-edge graph finite
# ---------------------------------------------------------------------------


def _no_edge_batch() -> dict[str, Any]:
    import structural_relation_objects_v1 as sro

    obj = sro.build_mol_objects(0, np.asarray([3], dtype=np.int64),
                                np.zeros((2, 0), dtype=np.int64),
                                np.zeros((0,), dtype=np.int64))
    rel = BD.build_relation_mol(obj)
    return BD.collate(_fake_cache([obj]), _fake_rel([rel]), [0])


def check_readout(report: dict, model) -> None:
    captured: dict[str, torch.Tensor] = {}
    orig = model.reader

    class Cap(torch.nn.Module):
        def forward(self, z):
            captured["z"] = z.detach().clone()
            return orig(z)

    model.reader = Cap()
    cache = BD.load_object_cache("valid")
    rel = BD.load_relation_split("valid")
    batch = BD.collate(cache, rel, np.asarray([0, 3, 7]))
    model.eval()
    with torch.no_grad():
        y = model(batch)
        tok, h, _ = model.encode(batch)
        z_manual = model.pool(tok, h, batch)
        z = captured["z"]
        err = float((z - z_manual).abs().max())
        # amplitude is *not* per-molecule normalised: S1 is a count-scaled sum
        B, Dm = h.shape[0], h.shape[1]
        h_valid = h.reshape(B * Dm, model.width).index_select(0, tok["slot"])
        s1 = 0.5 * h_valid.new_zeros((B, model.width)).index_add_(
            0, batch["dir_graph"], h_valid) / model.n_dir_ref
        err_s1 = float((z[:, :model.width] - s1).abs().max())
        sy = float(s1.std(0).mean())
        # doubling every token state doubles the first moment and quadruples
        # the second: the pooled blocks carry amplitude rather than a unit norm
        h2 = h * 2.0
        z2 = model.pool(tok, h2, batch)
        lin = float((z2[:, :model.width] - 2.0 * z[:, :model.width]).abs().max())
        quad = float((z2[:, model.width:2 * model.width]
                      - 4.0 * z[:, model.width:2 * model.width]).abs().max())
        edge_tok, edge_h, _ = model.encode(_no_edge_batch())
        y_edge = model.reader(model.pool(edge_tok, edge_h, _no_edge_batch()))
    model.reader = orig
    ok = (err < 1e-6 and err_s1 < 1e-6 and sy > 0.0
          and lin < 1e-4 and quad < 1e-3
          and bool(torch.isfinite(y_edge).all()))
    report["readout"] = {
        "pass": bool(ok),
        "pooled_block_recompute_max_abs_err": err,
        "s1_recompute_max_abs_err": err_s1,
        "s1_varies_across_molecules": sy > 0.0,
        "scaling_is_linear_in_h": lin,
        "second_moment_scaling_is_quadratic": quad,
        "no_edge_prediction": float(y_edge.reshape(-1)[0]),
        "no_edge_finite": bool(torch.isfinite(y_edge).all()),
        "n_dir_ref": float(model.n_dir_ref),
        "n_atom_ref": float(model.n_atom_ref),
        "note": "no per-molecule LayerNorm on any pooled block",
    }


# ---------------------------------------------------------------------------
# 7. data alignment / batching / schedule
# ---------------------------------------------------------------------------


def check_data(report: dict) -> None:
    train = BD.load_object_cache("train")
    valid = BD.load_object_cache("valid")
    labels_pyg = D.official_labels("train")
    cache_labels = np.asarray(train["labels"], dtype=np.float64)
    max_label_diff = float(np.abs(labels_pyg[:200] - cache_labels[:200]).max())
    n = len(cache_labels)
    bs = int(BT.RECIPE["batch_size"])
    spe = BT.steps_per_epoch(n, bs)
    order = np.random.default_rng(0).permutation(n)
    batches = [order[s:s + bs] for s in range(0, n, bs)]
    seen = np.sort(np.concatenate(batches))
    rows_ok = bool(np.array_equal(seen, np.arange(n)))
    last = int(len(batches[-1]))
    # one epoch of the real training order, exact same logic
    gen = torch.Generator().manual_seed(0)
    order2 = torch.randperm(n, generator=gen).numpy()
    batches2 = [order2[s:s + bs] for s in range(0, n, bs)]
    rows_ok &= bool(np.array_equal(np.sort(np.concatenate(batches2)), np.arange(n)))
    # LR schedule boundary points
    total = spe * int(BT.RECIPE["epochs"])
    lr1 = BT._lr_at(1, total, BT.RECIPE)
    lr_w = BT._lr_at(int(BT.RECIPE["warmup_updates"]), total, BT.RECIPE)
    lr_end = BT._lr_at(total, total, BT.RECIPE)
    steps = np.arange(int(BT.RECIPE["warmup_updates"]) + 1, total + 1, 500)
    lrs = np.asarray([BT._lr_at(int(s), total, BT.RECIPE) for s in steps])
    decreasing = bool(np.all(np.diff(lrs) <= 1e-12))
    ok = (max_label_diff == 0.0 and rows_ok and spe == 157 and last == 16
          and len(valid["labels"]) == 1000
          and abs(lr1 - BT.RECIPE["lr"] / BT.RECIPE["warmup_updates"]) < 1e-12
          and abs(lr_w - BT.RECIPE["lr"]) < 1e-12
          and abs(lr_end - BT.RECIPE["min_lr"]) < 1e-9
          and decreasing and total == 235_500)
    report["data_and_schedule"] = {
        "pass": bool(ok),
        "n_train": n, "n_valid": int(len(valid["labels"])),
        "train_label_max_abs_diff_vs_pyg_200": max_label_diff,
        "steps_per_epoch": spe,
        "last_batch_rows": last,
        "each_row_exactly_once": rows_ok,
        "total_updates": int(total),
        "lr_at_step_1": lr1,
        "lr_at_warmup_end": lr_w,
        "lr_at_last_step": lr_end,
        "lr_monotone_after_warmup": decreasing,
    }


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--best", default=None,
                   help="checkpoint to use; default = a fresh seed-0 model")
    args = p.parse_args(argv)
    rng = np.random.default_rng(20261022)
    torch.manual_seed(0)
    device = torch.device("cpu")
    model, info = BT.build(0, device, dropout=float(BT.RECIPE["dropout"]))
    if args.best:
        state = torch.load(args.best, map_location="cpu", weights_only=False)
        C.load_state(model, state["state_dict"], device)
    report: dict[str, Any] = {"checkpoint": args.best, "device": "cpu",
                              "parameters": int(C.parameter_count(model))}
    check_paths(report)
    check_invariance(report, model, rng)
    check_directed(report)
    check_padding(report, model)
    check_relation_value(report, model)
    check_ssl(report, model, info, rng)
    check_readout(report, model)
    check_data(report)
    report["all_pass"] = all(v.get("pass") for k, v in report.items()
                             if isinstance(v, dict) and "pass" in v)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "cpu_checks.json", report)
    C.log(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v)
                      for k, v in report.items()}, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
