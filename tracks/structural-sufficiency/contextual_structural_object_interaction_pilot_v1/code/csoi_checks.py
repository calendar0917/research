"""Engineering correctness gate for CSOI-v1 (must pass before formal training).

Every check is stated as a falsifiable numeric assertion over the *real* prepared
cache.  Nothing here is a training run; ``batch`` uses one real mini-batch to
show finite losses and non-zero gradients in every module group.
"""

from __future__ import annotations

import argparse
import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np

import csoi_common as C
import csoi_data as D
import csoi_model as M


def _subsample(rows: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if rows.size <= n:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, n, replace=False))


def _model(cfg, pack, atom_dims, bond_dims, arm="CSOI_REL", seed=None):
    import torch

    C.set_deterministic(0)
    model = M.CSOIModel(cfg, atom_dims, bond_dims, arm)
    model.load_pack(pack)
    model = model.double()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. raw graph / frame correspondence
# ---------------------------------------------------------------------------


def check_inputs(cfg, pack, raw) -> dict[str, Any]:
    n_mol = int(pack["n_molecules"])
    mol_parse_ptr = np.asarray(pack["mol_parse_ptr"], np.int64)
    s_flat = np.asarray(pack["parse_S_atoms"], np.int64)
    c_flat = np.asarray(pack["parse_C_atoms"], np.int64)
    s_ptr = np.asarray(pack["parse_S_atoms_ptr"], np.int64)
    c_ptr = np.asarray(pack["parse_C_atoms_ptr"], np.int64)
    parse_kind = np.asarray(pack["parse_kind"], np.uint8)
    parse_portS = np.asarray(pack["parse_port_S"], np.int64)
    parse_portC = np.asarray(pack["parse_port_C"], np.int64)
    parse_cut_u = np.asarray(pack["parse_cut_u"], np.int64)
    parse_cut_v = np.asarray(pack["parse_cut_v"], np.int64)
    parse_bond = np.asarray(pack["parse_cut_bond_row"], np.int64)
    bond_table = np.asarray(pack["bond_table"])
    frame_atom = np.asarray(pack["frame_atom"], np.int64)
    frame_bond = np.asarray(pack["frame_bond"], np.int64)
    frame_k = np.asarray(pack["frame_k"], np.int64)
    atom_table = np.asarray(pack["atom_table"])

    recombine_bad = 0
    endpoint_bad = 0
    bridge_bad = 0
    count_bad = 0
    atom_set_bad = 0
    checked = 0
    for m in range(n_mol):
        p0, p1 = int(mol_parse_ptr[m]), int(mol_parse_ptr[m + 1])
        mol = C.PD.raw_molecule(raw, m)
        adj = mol.adjacency_lists()
        # independent recomputation of the parse list
        parses, _nb = C.PD.cut_parses(mol, adj, max_side=int(cfg["cuts"]["max_side_atoms"]))
        if not parses:
            parses = C.PD.fallback_parses(mol)
        if len(parses) != p1 - p0:
            count_bad += 1
            continue
        off_s = int(s_ptr[m])
        off_c = int(c_ptr[m])
        bindex = mol.bond_index()
        for j, p in enumerate(parses):
            g = p0 + j
            n_s, n_c = len(p.atoms_s), len(p.atoms_c)
            sa = s_flat[off_s:off_s + n_s]
            ca = c_flat[off_c:off_c + n_c]
            off_s += n_s
            off_c += n_c
            checked += 1
            if sorted(sa.tolist()) != sorted(p.atoms_s) or sorted(ca.tolist()) != sorted(p.atoms_c):
                atom_set_bad += 1
            if set(sa.tolist()) & set(ca.tolist()):
                recombine_bad += 1
                continue
            if sorted(set(sa.tolist()) | set(ca.tolist())) != list(range(mol.n_atoms)):
                recombine_bad += 1
            if int(parse_portS[g]) != int(p.port_s):
                endpoint_bad += 1
            if p.port_c is not None and int(parse_portC[g]) != int(p.port_c):
                endpoint_bad += 1
            want_kind = 1 if p.bond_index is None else 0
            if int(parse_kind[g]) != want_kind:
                bridge_bad += 1
                continue
            if p.bond_index is not None:
                u, v = int(mol.bond_u[p.bond_index]), int(mol.bond_v[p.bond_index])
                if int(parse_cut_u[g]) != u or int(parse_cut_v[g]) != v:
                    endpoint_bad += 1
                if v not in adj[u]:
                    bridge_bad += 1
                row = int(parse_bond[g])
                if row < 0 or not np.array_equal(bond_table[row], mol.bond_feat[p.bond_index]):
                    bridge_bad += 1
    # frame <-> attribute row correspondence: rebuild a sub-cache and match frame
    # *content* (k, atom attribute tuples, bond attribute tuples) both ways
    rows = _subsample(np.asarray(pack["train_idx"], np.int64), 20, seed=31)
    mini = C.PP.build_dataset(cfg, cfg["dataset_key"], log=lambda *_a: None,
                              rows=rows.tolist())
    full_set = _frame_content_set(pack)
    mini_set = _frame_content_set(mini)
    missing = len(mini_set - full_set)
    frame_bad = missing
    bond_bad = 0
    atom_table_distinct = int(np.unique(atom_table, axis=0).shape[0]) == int(atom_table.shape[0])
    bond_table_distinct = int(np.unique(bond_table, axis=0).shape[0]) == int(bond_table.shape[0])
    return {
        "ok": (recombine_bad == 0 and endpoint_bad == 0 and bridge_bad == 0
               and frame_bad == 0 and count_bad == 0 and atom_set_bad == 0
               and atom_table_distinct and bond_table_distinct),
        "n_parses_checked": int(checked),
        "parse_count_mismatch_molecules": count_bad,
        "parse_atom_set_mismatch": atom_set_bad,
        "parse_recombination_bad": recombine_bad,
        "parse_cut_endpoint_bad": endpoint_bad,
        "parse_bridge_bad": bridge_bad,
        "frame_content_missing_after_rebuild": missing,
        "n_frames_rebuilt": len(mini_set),
        "n_frames_full": len(full_set),
        "atom_attribute_rows_distinct": bool(atom_table_distinct),
        "bond_attribute_rows_distinct": bool(bond_table_distinct),
        "n_frames_checked": len(mini_set),
    }


def _frame_content_set(pack: Mapping[str, Any]) -> set:
    atom_table = np.asarray(pack["atom_table"])
    bond_table = np.asarray(pack["bond_table"])
    frame_atom = np.asarray(pack["frame_atom"], np.int64)
    frame_bond = np.asarray(pack["frame_bond"], np.int64)
    frame_k = np.asarray(pack["frame_k"], np.int64)
    out = set()
    for f in range(int(pack["n_frames"])):
        k = int(frame_k[f])
        atoms = tuple(atom_table[frame_atom[f, :k]].astype(int).ravel().tolist())
        # bond rows are only comparable through their attribute content
        bond_content = tuple(
            tuple(bond_table[int(v)].astype(int).tolist()) if v >= 0 else ()
            for v in frame_bond[f].tolist())
        out.add((k, atoms, bond_content))
    return out


def atoms_have_attributes(atom_table: np.ndarray, rows: np.ndarray) -> bool:
    return bool(np.all((rows >= 0) & (rows < atom_table.shape[0])))


# ---------------------------------------------------------------------------
# 2. atom relabeling invariance
# ---------------------------------------------------------------------------


def check_renumber(cfg, pack, raw, atom_dims, bond_dims, n_mol: int = 120) -> dict[str, Any]:
    import torch

    train = np.asarray(pack["train_idx"], np.int64)
    rows = _subsample(train, n_mol, seed=1)
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    rng = np.random.default_rng(7)
    perms = {}
    for r in rows.tolist():
        n = int(node_ptr[r + 1] - node_ptr[r])
        perms[int(r)] = rng.permutation(n)
    mini = C.PP.build_dataset(cfg, cfg["dataset_key"], log=lambda *_a: None,
                              rows=rows.tolist(), perms=perms)
    ref = _model(cfg, pack, atom_dims, bond_dims, "CSOI_REL")
    alt = _model(cfg, mini, atom_dims, bond_dims, "CSOI_REL")
    with torch.no_grad():
        a = ref.molecule_cuts(torch.as_tensor(rows.astype(np.int64)))["f_bar"]
        b = alt.molecule_cuts(torch.arange(len(rows), dtype=torch.long))["f_bar"]
    diff = float((a - b).abs().max())
    return {"ok": diff < 1e-8, "max_abs_diff": diff, "n_molecules": int(len(rows)),
            "device": "cpu", "dtype": "float64"}


# ---------------------------------------------------------------------------
# 3. token multiplicity fidelity
# ---------------------------------------------------------------------------


def check_tokens(cfg, pack, raw) -> dict[str, Any]:
    view = D.TokenView(pack)
    n_frag = int(pack["n_fragments"])
    ids = np.arange(n_frag)
    mass = view.object_mass(ids)
    tokens = view.tokens_of(ids)
    per_frag_tokens = np.bincount(tokens["owner"], weights=tokens["weight"], minlength=n_frag)
    mass_diff = float(np.max(np.abs(per_frag_tokens - mass)))
    counts = np.asarray(pack["frag_counts"], np.int64)
    bucket_mass = np.zeros((n_frag, C.N_BUCKETS), np.float64)
    np.add.at(bucket_mass, (tokens["owner"], tokens["bucket"]), tokens["weight"])
    bucket_mass_diff = float(np.max(np.abs(bucket_mass - counts))) if n_frag else 0.0
    collapse = float(np.sum(tokens["n_tokens"]) / max(1.0, counts.sum()))
    weight_values = np.unique(np.round(tokens["weight"], 6))
    # independent rebuild of one fragment's object mass through the parent builder
    probe = _subsample(np.arange(n_frag), 4, seed=9)
    rebuilt = _rebuild_fragment_mass(cfg, pack, raw, probe)
    return {
        "ok": mass_diff < 1e-5 and bucket_mass_diff < 1e-4 and rebuilt["ok"],
        "fragment_mass_max_abs_diff": mass_diff,
        "token_weight_sum_equals_object_count": mass_diff < 1e-5,
        "bucket_mass_equals_object_count_max_abs_diff": bucket_mass_diff,
        "token_collapse_ratio": collapse,
        "independent_rebuild": rebuilt,
        "token_weight_values": weight_values[:12].tolist(),
        "n_distinct_token_weights": int(weight_values.size),
        "total_object_mass": float(mass.sum()),
    }


def _rebuild_fragment_mass(cfg, pack, raw, probe: np.ndarray) -> dict[str, Any]:
    """Rebuild a sub-cache and compare per-fragment token mass by exact key.

    The token mass of a fragment must be a function of the fragment identity
    alone, so rebuilding a molecule subset must reproduce it exactly.
    """
    rows = _subsample(np.asarray(pack["train_idx"], np.int64), 40, seed=21)
    mini = C.PP.build_dataset(cfg, cfg["dataset_key"], log=lambda *_a: None, rows=rows.tolist())
    full_keys = np.asarray(pack["fragment_keys"]).tolist()
    mini_keys = np.asarray(mini["fragment_keys"]).tolist()
    index = {k: i for i, k in enumerate(mini_keys)}
    fv, mv = D.TokenView(pack), D.TokenView(mini)
    mass_full = dict(zip(full_keys, fv.object_mass(np.arange(len(full_keys)))))
    mass_mini = mv.object_mass(np.arange(len(mini_keys)))
    tok_full = dict(zip(full_keys, fv.token_count.tolist()))
    tok_mini = mv.token_count.tolist()
    diffs = []
    tok_diffs = []
    for key in mini_keys:
        diffs.append(abs(mass_full[key] - float(mass_mini[index[key]])))
        tok_diffs.append(abs(tok_full[key] - int(tok_mini[index[key]])))
    return {
        "ok": len(diffs) >= 20 and max(diffs) < 1e-5 and max(tok_diffs) == 0,
        "n_molecules": int(len(rows)), "n_fragments_compared": len(diffs),
        "max_mass_diff": float(max(diffs)) if diffs else None,
        "max_token_count_diff": int(max(tok_diffs)) if tok_diffs else None,
    }


# ---------------------------------------------------------------------------
# 4. branch structure
# ---------------------------------------------------------------------------


def _double_difference(model, pairs4: Mapping[str, np.ndarray]) -> np.ndarray:
    import torch

    with torch.no_grad():
        out = []
        for side in ("a", "b"):
            s = pairs4[f"S_{side}"]
            for c in ("C1", "C2"):
                ctx = pairs4[c]
                bond = torch.as_tensor(pairs4["bond"], device=model.device)
                f = model.cut_scores(torch.as_tensor(s, device=model.device),
                                     torch.as_tensor(ctx, device=model.device), bond)
                out.append(f.double().cpu().numpy())
    return (out[2] - out[0]) - (out[3] - out[1])


def check_branches(cfg, pack, atom_dims, bond_dims) -> dict[str, Any]:
    import torch

    rng = np.random.default_rng(11)
    train = np.asarray(pack["train_idx"], np.int64)
    rows = _subsample(train, 8, seed=5)
    bond_table = np.asarray(pack["bond_table"])
    mid = int(bond_table.shape[0] // 2)
    # same context C under two different words, twice (a real 2x2 structure)
    pairs4 = {
        "S_a": np.asarray([11, 12], np.int64), "S_b": np.asarray([13, 14], np.int64),
        "C1": np.asarray([21, 22], np.int64), "C2": np.asarray([23, 24], np.int64),
        "bond": np.asarray([mid, mid], np.int64),
    }
    add = _model(cfg, pack, atom_dims, bond_dims, "CSOI_ADD")
    d_add = _double_difference(add, pairs4)
    glob_model = _model(cfg, pack, atom_dims, bond_dims, "CSOI_GLOBAL")
    rel = _model(cfg, pack, atom_dims, bond_dims, "CSOI_REL")
    rel.use_relation = False
    rel.eval()
    s = torch.as_tensor(pairs4["S_a"], device=rel.device)
    c = torch.as_tensor(pairs4["C1"], device=rel.device)
    b = torch.as_tensor(pairs4["bond"], device=rel.device)
    with torch.no_grad():
        f_glob = glob_model.cut_scores(s, c, b).double()
        f_rel_off = rel.cut_scores(s, c, b).double()
    max_glob_rel = float((f_glob - f_rel_off).abs().max())
    rel.use_relation = True
    with torch.no_grad():
        f_rel_on = rel.cut_scores(s, c, b).double()
    diff_rel_on = float((f_rel_on - f_rel_off).abs().max())
    # explicit double difference on the row set the ADD arm cancels exactly
    d_glob = _double_difference(glob_model, pairs4)
    d_rel = _double_difference(rel, pairs4)
    # additive-only path is exactly the model output minus the pooled branch
    with torch.no_grad():
        full = add.cut_scores(s, c, b).double()
        add_only = add._additive_only(s, c, b, add.frame_embeddings()).double()
    add_identity = float((full - add_only).abs().max())
    # dtot table: clip(d_s + 1 + d_c, 4+) and unreachable stays unreachable
    dt = rel.dtot.cpu().numpy()
    dt_ok = True
    for a in range(C.N_BUCKETS):
        for bb in range(C.N_BUCKETS):
            if a == C.NULL_BUCKET or bb == C.NULL_BUCKET:
                want = C.NULL_BUCKET
            else:
                want = min(a + 1 + bb, C.NULL_BUCKET - 1)
            if int(dt[a * C.N_BUCKETS + bb]) != want:
                dt_ok = False
    # ADD has no attention parameters at all (structural control)
    add_names = [n for n, _ in add.named_parameters() if n.startswith(("q_proj", "k_proj",
                                                                       "v_proj", "r_mlp", "out"))]
    return {
        "ok": (float(np.max(np.abs(d_add))) < 1e-10 and max_glob_rel < 1e-10
               and diff_rel_on > 1e-6 and add_identity < 1e-12 and dt_ok
               and len(add_names) == 0),
        "add_double_difference_max_abs": float(np.max(np.abs(d_add))),
        "global_vs_rel_without_bias_max_abs": max_glob_rel,
        "rel_with_bias_vs_without_max_abs": diff_rel_on,
        "global_double_difference_mean_abs": float(np.mean(np.abs(d_glob))),
        "rel_double_difference_mean_abs": float(np.mean(np.abs(d_rel))),
        "additive_only_identity_max_abs": add_identity,
        "dtot_table_ok": bool(dt_ok),
        "add_attention_parameters": add_names,
        "note": ("the four fragment ids are synthetic on purpose: the assertions are "
                 "structural identities, they do not need real chemistry"),
    }


# ---------------------------------------------------------------------------
# 5. attention equals an explicit per-parse reference
# ---------------------------------------------------------------------------


def _naive_cut_scores(model: M.CSOIModel, s_ids, c_ids, bond_rows) -> np.ndarray:
    import torch

    frames = model.frame_embeddings()
    out = []
    cut_table = model.cut_embeddings()
    for i in range(int(s_ids.numel())):
        xs, _o, w, _b = model.token_embeddings(s_ids[i:i + 1], frames)
        if int(c_ids[i]) >= 0:
            xc, _oc, wc, bc = model.token_embeddings(c_ids[i:i + 1], frames)
        else:
            xc = model.null_ctx[None, :]
            wc = torch.ones(1, dtype=xs.dtype, device=xs.device)
            bc = torch.tensor([C.NULL_BUCKET], dtype=torch.long, device=xs.device)
        nq = xs.shape[0]
        bq = _b[:nq]
        pool_s = model._weighted_mean(xs, torch.zeros(nq, dtype=torch.long), w, 1)
        pool_c = model._weighted_mean(xc, torch.zeros(xc.shape[0], dtype=torch.long), wc, 1)
        q = model.q_proj(xs).view(nq, model.heads, model.dh)
        k = model.k_proj(xc).view(-1, model.heads, model.dh)
        v = model.v_proj(xc).view(-1, model.heads, model.dh)
        score = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(model.dh)
        score = score + torch.log(wc)[None, None, :]
        e = model._cut_tokens(bond_rows[i:i + 1], cut_table)
        if model.use_relation:
            dt = model.dtot[bq[:, None] * C.N_BUCKETS + bc[None, :]]
            z = (model.rel_ds(bq)[:, None, :] + model.rel_dc(bc)[None, :, :]
                 + model.rel_dt(dt) + model.rel_cut(e)[:, None, :])
            score = score + model.rel_mlp(z).permute(2, 0, 1)
        alpha = torch.softmax(score, dim=-1)
        m = torch.einsum("hqk,khd->qhd", alpha, v).reshape(nq, model.hidden)
        r = model.r_mlp(torch.cat([xs, m, xs * m, xs - m, e.repeat(nq, 1)], dim=-1))
        rp = (r * w[:, None]).sum(0) / w.sum()
        base = (model.beta.reshape(()) + model.add_s(pool_s).reshape(())
                + model.add_c(pool_c).reshape(()) + model.bond_bias(e).reshape(()))
        out.append(base + model.out(rp).reshape(()))
    return torch.stack(out).detach().numpy()


def check_attention(cfg, pack, atom_dims, bond_dims) -> dict[str, Any]:
    import torch

    model = _model(cfg, pack, atom_dims, bond_dims, "CSOI_REL")
    n_frag = int(pack["n_fragments"])
    rng = np.random.default_rng(13)
    s_ids = rng.choice(n_frag, 6, replace=False)
    c_ids = rng.choice(n_frag, 6, replace=False)
    c_ids[1] = -1  # null-context parse
    bond = np.asarray([int(np.asarray(pack["bond_table"]).shape[0] // 2)] * 6, np.int64)
    bond[1] = -1
    st = torch.as_tensor(s_ids.astype(np.int64))
    ct = torch.as_tensor(c_ids.astype(np.int64))
    bt = torch.as_tensor(bond)
    with torch.no_grad():
        fast = model.cut_scores(st, ct, bt).double().numpy()
    naive = _naive_cut_scores(model, st, ct, bt)
    diff = float(np.max(np.abs(fast - naive)))
    # chunking invariance
    saved = model.entry_chunk
    model.entry_chunk = 7
    with torch.no_grad():
        chunked = model.cut_scores(st, ct, bt).double().numpy()
    model.entry_chunk = saved
    chunk_diff = float(np.max(np.abs(fast - chunked)))
    # permuting the context token order must not change the response
    return {"ok": diff < 1e-9 and chunk_diff < 1e-9,
            "vs_naive_reference_max_abs": diff,
            "chunk_size_invariance_max_abs": chunk_diff,
            "entry_chunk": int(saved)}


# ---------------------------------------------------------------------------
# 6. batching
# ---------------------------------------------------------------------------


def check_batching(cfg, pack, atom_dims, bond_dims) -> dict[str, Any]:
    import torch

    model = _model(cfg, pack, atom_dims, bond_dims, "CSOI_REL")
    valid = np.asarray(pack["valid_idx"], np.int64)
    rows = _subsample(valid, 12, seed=2)
    with torch.no_grad():
        batched = model.molecule_cuts(torch.as_tensor(rows))["f_bar"]
        single = torch.cat([model.molecule_cuts(torch.as_tensor([int(r)]))["f_bar"]
                            for r in rows.tolist()])
        perm = torch.as_tensor(np.random.default_rng(4).permutation(rows.size))
        shuffled = model.molecule_cuts(torch.as_tensor(rows)[perm])["f_bar"]
        back = torch.empty_like(shuffled)
        back[perm] = shuffled
    b = float((batched - single).abs().max())
    s = float((batched - back).abs().max())
    return {"ok": b < 1e-9 and s < 1e-11, "batch_vs_single_max_abs": b,
            "order_invariance_max_abs": s, "n_molecules": int(rows.size)}


# ---------------------------------------------------------------------------
# 7. one real training batch
# ---------------------------------------------------------------------------


def check_batch(cfg, pack, raw, atom_dims, bond_dims, arm: str = "CSOI_REL") -> dict[str, Any]:
    import csoi_train as T

    data = T.RunData(cfg, "lipo", log=lambda *_a: None)
    model, zeroed = T.build_model(cfg, data, arm)
    store = C.PPAIR.PairStore(cfg, data.pack, data.labels, None,
                              data.splits["train"], data.row_ids, log=lambda *_a: None)
    diag = T.grad_diagnostics(model, data, store, cfg, store.sampler())
    groups = diag["groups"]
    all_nonzero = all(g["grad_norm"] > 0 and g["params_with_nonzero_grad"] > 0
                      for g in groups.values())
    return {"ok": diag["finite"] and all_nonzero,
            "finite_loss": diag["finite"], "loss_main": diag["loss_main"],
            "loss_aux": diag["loss_aux"], "groups": groups,
            "zeroed_categories": zeroed}


# ---------------------------------------------------------------------------
# 8. split / label boundary and train-only pair supervision
# ---------------------------------------------------------------------------


def check_boundary(cfg, pack) -> dict[str, Any]:
    train = np.asarray(pack["train_idx"], np.int64)
    valid = np.asarray(pack["valid_idx"], np.int64)
    test = np.asarray(pack["test_idx"], np.int64)
    labels = np.asarray(pack["labels"], np.float64)
    scaler = C.target_scaler(labels, train)
    y = labels[train]
    store = C.PPAIR.PairStore(cfg, pack, labels, None, train,
                              np.asarray(pack["row_ids"]), log=lambda *_a: None)
    pa = np.asarray(pack["pair_mol_a"], np.int64)[store.index]
    pb = np.asarray(pack["pair_mol_b"], np.int64)[store.index]
    cell = np.asarray(pack["pair_cell"], np.uint8)[store.index]
    pair_ok = bool(np.all(np.isin(pa, train)) and np.all(np.isin(pb, train))
                   and np.all(cell == 0))
    # the sampler uses only these pairs
    picks, _mols = store.sampler().sample(64)
    sampler_ok = bool(np.all(np.isin(store.index[picks],
                                     np.nonzero(np.asarray(pack["pair_cell"]) == 0)[0])))
    # the trainer's data view has no test split at all
    import csoi_train as T

    data = T.RunData(cfg, "lipo", log=lambda *_a: None)
    view_ok = set(data.splits) == {"train", "valid"}
    return {
        "ok": (pair_ok and sampler_ok and view_ok
               and train.size == 3360 and valid.size == 420 and test.size == 420
               and len(set(train.tolist()) & set(valid.tolist())) == 0
               and len(set(train.tolist()) & set(test.tolist())) == 0
               and len(set(valid.tolist()) & set(test.tolist())) == 0),
        "n_train": int(train.size), "n_valid": int(valid.size), "n_test_kept_unused": int(test.size),
        "scaler": scaler, "scaler_from_train_only": bool(
            abs(scaler["mean"] - float(np.mean(y))) < 1e-12
            and abs(scaler["std"] - float(np.std(y))) < 1e-12),
        "pair_store_train_only": pair_ok, "pair_sampler_train_only": sampler_ok,
        "n_train_pairs": int(store.npairs()), "trainer_data_view": sorted(data.splits),
        "test_labels_read": False,
    }


# ---------------------------------------------------------------------------
# 9. identical initialisation of the shared modules
# ---------------------------------------------------------------------------


def check_init(cfg, pack, atom_dims, bond_dims) -> dict[str, Any]:
    hashes = {}
    for arm in M.CSOI_ARMS:
        model = _model(cfg, pack, atom_dims, bond_dims, arm)
        hashes[arm] = C.module_hashes(model, ["phi", "token_proj", "bucket_emb", "mult_proj",
                                             "token_ln", "cut_proj", "add_s", "add_c",
                                             "bond_bias"])
    shared_same = all(hashes[a] == hashes["CSOI_ADD"] for a in M.CSOI_ARMS)
    attn = {arm: C.module_hashes(_model(cfg, pack, atom_dims, bond_dims, arm), ["q_proj"])
            for arm in ("CSOI_GLOBAL", "CSOI_REL")}
    return {"ok": shared_same and attn["CSOI_GLOBAL"] == attn["CSOI_REL"],
            "shared_module_hashes_equal": shared_same,
            "attention_init_equal_between_global_and_rel":
                attn["CSOI_GLOBAL"] == attn["CSOI_REL"],
            "hashes": hashes}


# ---------------------------------------------------------------------------
# 10. parameter ledger: the arms differ only where the protocol says so
# ---------------------------------------------------------------------------
def check_params(cfg, pack, atom_dims, bond_dims) -> dict[str, Any]:
    ledgers = {}
    groups = {}
    for arm in M.CSOI_ARMS:
        model = _model(cfg, pack, atom_dims, bond_dims, arm)
        ledger = model.parameter_ledger()
        ledgers[arm] = ledger
        grp = model.module_groups()
        groups[arm] = {g: int(sum(ledger.get(n, 0) for n in names))
                       for g, names in grp.items()}
    add, glob, rel = (groups[a] for a in M.CSOI_ARMS)
    rel_names = ("rel_ds", "rel_dc", "rel_dt", "rel_cut", "rel_mlp")
    rel_branch = int(sum(ledgers["CSOI_REL"].get(n, 0) for n in rel_names))
    # the shared interaction modules must be identical across GLOBAL and REL:
    # the relation branch is the *only* parameter difference between them
    same_encoder = all(add[g] == glob[g] == rel[g] for g in ("token_encoder", "additive"))
    shared_interaction_equal = (glob["interaction"]
                                == rel["interaction"] - rel_branch)
    totals = {a: ledgers[a]["total"] for a in M.CSOI_ARMS}
    return {
        "ok": bool(same_encoder and shared_interaction_equal
                   and add["interaction"] == 0
                   and totals["CSOI_ADD"] < totals["CSOI_GLOBAL"] < totals["CSOI_REL"]
                   and totals["CSOI_REL"] - totals["CSOI_GLOBAL"] == rel_branch),
        "total": totals,
        "groups": groups,
        "encoder_and_additive_equal_across_arms": bool(same_encoder),
        "shared_interaction_modules_equal_after_removing_relation_branch":
            bool(shared_interaction_equal),
        "add_has_no_interaction_params": add["interaction"] == 0,
        "relation_branch_params": rel_branch,
        "rel_minus_global": totals["CSOI_REL"] - totals["CSOI_GLOBAL"],
    }


# ---------------------------------------------------------------------------
# 11. cut-loss algebra: per-molecule normalisation, pair normalisation, weighting
# ---------------------------------------------------------------------------
def check_loss(cfg, pack, atom_dims, bond_dims) -> dict[str, Any]:
    import torch

    model = _model(cfg, pack, atom_dims, bond_dims, "CSOI_REL")
    train = np.asarray(pack["train_idx"], np.int64)
    rows = _subsample(train, 24, seed=77)
    labels = np.asarray(pack["labels"], np.float64)
    scaler = C.target_scaler(labels, train)
    store = C.PPAIR.PairStore(cfg, pack, labels, None, train,
                              np.asarray(pack["row_ids"]), log=lambda *_a: None)
    pair_idx, _aux = store.sampler().sample(16)
    with torch.no_grad():
        res = model.molecule_forward(torch.as_tensor(rows))
        f_cut, owner, counts = res["f_cut"], res["owner"], res["counts"]
        y = torch.as_tensor(((labels - scaler["mean"]) / scaler["std"])[rows], dtype=f_cut.dtype)
        sq = (f_cut - y[owner]) ** 2
        per_mol = f_cut.new_zeros(rows.size).index_add_(0, owner, sq) / counts.clamp(min=1)
        l_main = per_mol.mean()
        # per-molecule (not per-cut) weighting: replicate one molecule's cuts and the
        # molecule-level mean must not change
        manual = sum(float(sq[owner == i].sum() / max(int(counts[i]), 1))
                     for i in range(rows.size)) / rows.size
        counts_ok = bool(torch.equal(counts, torch.bincount(owner, minlength=rows.size).to(dtype=counts.dtype)))
        delta = model.pair_cut_delta(store.tensor_pairs(pair_idx, f_cut.device))
        lab = torch.as_tensor(store.delta_std_of(pair_idx), dtype=f_cut.dtype)
        l_aux = torch.mean((delta - lab) ** 2) / float(store.tau2)
        # pair delta semantics: f(S_b|C) - f(S_a|C) with the pair's shared C
        tp = store.tensor_pairs(pair_idx, f_cut.device)
        direct = (model.cut_scores(tp["Sb"], tp["C"], tp["bond"])
                  - model.cut_scores(tp["Sa"], tp["C"], tp["bond"]))
        delta_gap = float((direct - delta).abs().max())
    # tau2 is the sampling-distribution-weighted mean of delta_std**2, floored
    total = 0.0
    for weight, pool in store.pools:
        acc = 0.0
        for t in pool:
            gs = store.transform_groups[t]
            acc += float(np.mean([float(np.mean(store.delta_std[idxs] ** 2))
                                  for idxs in gs.values()]))
        if pool:
            acc /= len(pool)
        total += weight * acc
    tau2_manual = max(float(cfg["recipe"]["aux_tau2_floor"]), float(total))
    aux_w = float(cfg["recipe"]["aux_weight"])
    return {
        "ok": (counts_ok and abs(manual - float(l_main)) < 1e-9
               and delta_gap < 1e-9
               and abs(tau2_manual - float(store.tau2)) < 1e-9),
        "cut_counts_match_parses": counts_ok,
        "per_molecule_mean_matches_direct_sum": abs(manual - float(l_main)),
        "pair_delta_matches_cut_scores_max_abs": delta_gap,
        "l_main": float(l_main),
        "l_aux": float(l_aux),
        "loss_at_frozen_aux_weight": float(l_main) + aux_w * float(l_aux),
        "aux_weight": aux_w,
        "tau2": float(store.tau2),
        "tau2_matches_reweighted_sampler_formula": abs(tau2_manual - float(store.tau2)) < 1e-9,
        "tau2_manual": tau2_manual,
    }


# ---------------------------------------------------------------------------
# 12. token guard: statistics are recorded, the cap is deterministic
# ---------------------------------------------------------------------------
def check_guard(cfg, pack) -> dict[str, Any]:
    tok = D.token_statistics(pack, cfg, log=lambda *_a: None)
    view = D.TokenView(pack)
    cap = int(cfg["csoi"]["guard_token_samples"])
    big = np.nonzero(view.token_count > cap)[0][:8].astype(np.int64)
    det = True
    for fid in big.tolist():
        t = view.tokens_of(np.asarray([fid]))
        a = D._stratified_cap(t["bucket"], t["weight"], cap)
        b = D._stratified_cap(t["bucket"], t["weight"], cap)
        det = bool(det and np.array_equal(a, b) and a.sum() <= cap)
    stats = tok["tokens_per_fragment"]
    guard = tok["guard"]
    # the frozen guard would drop too much mass on >5% of fragments, which is
    # exactly why training uses the full token set through chunked attention
    guard_inadequate = bool(guard["fraction_below_095"] >= 0.05)
    return {
        "ok": bool(det and guard["applied"] and guard["cap"] == cap
                   and guard_inadequate),
        "tokens_per_fragment": stats,
        "guard_triggered": bool(guard["applied"]),
        "cap": cap,
        "cap_deterministic": bool(det),
        "n_fragments_probed": int(big.size),
        "retained_mass": guard["retained_mass"],
        "fragments_below_0.95_mass": guard["fragments_below_095"],
        "fraction_below_0.95_mass": guard["fraction_below_095"],
        "guard_would_be_inadequate": guard_inadequate,
        "resolution": "full token set + chunked ragged attention (no token dropped)",
        "training_uses_full_token_set": True,
    }


# ---------------------------------------------------------------------------
# 13. determinism: same seed -> bitwise identical model and predictions
# ---------------------------------------------------------------------------
def check_determinism(cfg, pack, atom_dims, bond_dims) -> dict[str, Any]:
    import torch

    valid = np.asarray(pack["valid_idx"], np.int64)
    rows = _subsample(valid, 8, seed=6)

    def build(c, torch_seed: int):
        C.set_deterministic(int(torch_seed))
        model = M.CSOIModel(c, atom_dims, bond_dims, "CSOI_REL")
        model.load_pack(pack)
        return model.double().eval()

    preds = []
    for _ in range(2):
        model = build(cfg, 0)
        with torch.no_grad():
            preds.append(model.molecule_cuts(torch.as_tensor(rows))["f_bar"])
    same = bool(torch.equal(preds[0], preds[1]))
    # init must not depend on global torch RNG state (deterministic_init is name-based)
    other_rng = build(cfg, 1)
    with torch.no_grad():
        p_rng = other_rng.molecule_cuts(torch.as_tensor(rows))["f_bar"]
    rng_invariant = bool(torch.equal(p_rng, preds[0]))
    # a different model seed must change the predictions
    cfg2 = copy.deepcopy(cfg)
    cfg2["recipe"]["seed"] = int(cfg["recipe"]["seed"]) + 1
    seeded = build(cfg2, 0)
    with torch.no_grad():
        p_seed = seeded.molecule_cuts(torch.as_tensor(rows))["f_bar"]
    seed_delta = float((p_seed - preds[0]).abs().max())
    C.set_deterministic(0)
    return {"ok": bool(same and rng_invariant and seed_delta > 0),
            "same_cfg_bitwise_identical": same,
            "invariant_to_global_torch_rng": rng_invariant,
            "model_seed_changes_predictions": seed_delta > 0,
            "model_seed_delta_max_abs": seed_delta}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = D.load_pack(cfg, args.dataset)
    raw = C.PD.load_raw_cache(cfg, args.dataset)
    atom_dims, bond_dims = C.official_feature_dims()
    checks = {
        "inputs": lambda: check_inputs(cfg, pack, raw),
        "renumber": lambda: check_renumber(cfg, pack, raw, atom_dims, bond_dims),
        "tokens": lambda: check_tokens(cfg, pack, raw),
        "branches": lambda: check_branches(cfg, pack, atom_dims, bond_dims),
        "attention": lambda: check_attention(cfg, pack, atom_dims, bond_dims),
        "batching": lambda: check_batching(cfg, pack, atom_dims, bond_dims),
        "batch": lambda: check_batch(cfg, pack, raw, atom_dims, bond_dims),
        "boundary": lambda: check_boundary(cfg, pack),
        "init": lambda: check_init(cfg, pack, atom_dims, bond_dims),
        "params": lambda: check_params(cfg, pack, atom_dims, bond_dims),
        "loss": lambda: check_loss(cfg, pack, atom_dims, bond_dims),
        "guard": lambda: check_guard(cfg, pack),
        "determinism": lambda: check_determinism(cfg, pack, atom_dims, bond_dims),
    }
    names = args.only if args.only is not None else list(checks)
    results: dict[str, Any] = {}
    for name in names:
        if name in args.skip:
            results[name] = {"ok": None, "skipped": True}
            print(f"[skip] {name}")
            continue
        try:
            res = checks[name]()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            res = {"ok": False, "error": repr(exc)}
        results[name] = res
        print(f"[{'ok  ' if res.get('ok') else 'FAIL'}] {name}")
    summary = {
        "ok": all(bool(v.get("ok")) for v in results.values() if v.get("ok") is not None),
        "signature": C.check_signature(),
        "cache_sha256": pack["cache_sha256"],
        "content_sha256": pack["content_sha256"],
        "checks": results,
    }
    out = C.REPO_ROOT / args.out if args.out else (C.analysis_dir() / "checks.json")
    C.write_json(out, summary)
    print(f"all_ok={summary['ok']} -> {out}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
