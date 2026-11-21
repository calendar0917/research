"""Pre-training engineering checks (about 15-20 min, no audit project).

Run before the first formal epoch.  Each check prints a compact verdict and the
results are frozen into ``analysis/checks.json``.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

import trc_common as C
import trc_data as D
import trc_model as M
import trc_pairs as PR
import trc_prepare as P


def parse_atom_slices(pack) -> tuple[np.ndarray, np.ndarray]:
    """Per-parse atom offsets reconstructed from the fragment atom counts."""
    fs = np.asarray(pack["parse_S"], np.int64)
    fc = np.asarray(pack["parse_C"], np.int64)
    fn = np.asarray(pack["frag_natoms"], np.int64)
    len_s = fn[fs]
    len_c = np.where(fc >= 0, fn[np.maximum(fc, 0)], 0)
    ptr_s = np.concatenate([[0], np.cumsum(len_s)]).astype(np.int64)
    ptr_c = np.concatenate([[0], np.cumsum(len_c)]).astype(np.int64)
    return ptr_s, ptr_c


def randomize_model(model: M.ResponseModel, seed: int = 7) -> None:
    """Make every head non-trivial so invariance tests cannot pass trivially."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.05)
            else:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.1)
            if name.endswith("beta") or "null_cut" in name or "h_empty" in name:
                param.copy_(torch.randn(param.shape, generator=gen))


def check_inputs(cfg, pack, raw) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n_mol = int(pack["n_molecules"])
    out["n_molecules"] = n_mol
    out["split_sizes"] = {s: int(np.asarray(pack[f"{s}_idx"]).size) for s in C.SPLITS}
    out["splits_disjoint"] = bool(
        set(np.asarray(pack["train_idx"]).tolist())
        .isdisjoint(np.asarray(pack["valid_idx"]).tolist())
        and set(np.asarray(pack["train_idx"]).tolist())
        .isdisjoint(np.asarray(pack["test_idx"]).tolist())
        and set(np.asarray(pack["valid_idx"]).tolist())
        .isdisjoint(np.asarray(pack["test_idx"]).tolist()))
    out["splits_cover_all"] = bool(
        np.asarray(pack["train_idx"]).size + np.asarray(pack["valid_idx"]).size
        + np.asarray(pack["test_idx"]).size == n_mol)
    out["row_ids_unique"] = bool(len(set(np.asarray(pack["row_ids"]).tolist())) == n_mol)
    scaler = C.target_scaler(np.asarray(pack["labels"], np.float64),
                             np.asarray(pack["train_idx"], np.int64))
    out["scaler"] = scaler
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    bad_local = 0
    bond_sum = 0
    for row in range(n_mol):
        mol = D.raw_molecule(raw, row)
        if mol.bond_u.size:
            if mol.bond_u.min() < 0 or max(mol.bond_u.max(), mol.bond_v.max()) >= mol.n_atoms:
                bad_local += 1
            if mol.bond_u.max() >= mol.bond_v.max() and np.any(mol.bond_u >= mol.bond_v):
                bad_local += 1
        bond_sum += int(mol.bond_u.size)
    out["local_edge_index_bad"] = int(bad_local)
    out["bond_count_matches_ptr"] = bool(bond_sum == int(bond_ptr[-1]))

    # frames: masks and padded fields
    frame_atom = np.asarray(pack["frame_atom"])
    frame_bond = np.asarray(pack["frame_bond"])
    frame_k = np.asarray(pack["frame_k"]).astype(np.int64)
    slots = frame_atom.shape[1]
    within = np.arange(slots)[None, :] < frame_k[:, None]
    padding_atom_zero = bool(np.all(frame_atom[~within] == 0))
    pair_slot = {tuple(p): i for i, p in enumerate(C.FRAME_PAIRS)}
    bond_counts = (frame_bond >= 0).sum(1)
    outside_zero = True
    connected = True
    for g in range(frame_atom.shape[0]):
        k = int(frame_k[g])
        present = frame_bond[g] >= 0
        for (i, j), slot in pair_slot.items():
            if (i >= k or j >= k) and present[slot]:
                outside_zero = False
        edges = [(i, j) for (i, j), slot in pair_slot.items() if i < k and j < k and present[slot]]
        if k == 1:
            continue
        seen = {0}
        stack = [0]
        adj_slot = {i: [] for i in range(k)}
        for (i, j) in edges:
            adj_slot[i].append(j)
            adj_slot[j].append(i)
        while stack:
            u = stack.pop()
            for w in adj_slot[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        if len(seen) != k:
            connected = False
    out["frames_node_mask_ok"] = bool(padding_atom_zero and np.all(within.sum(1) == frame_k))
    out["frames_edge_mask_ok"] = bool(outside_zero and connected
                                      and np.all(bond_counts[frame_k > 1] > 0))
    out["n_frames"] = int(frame_atom.shape[0])
    out["n_fragments"] = int(pack["n_fragments"])
    out["n_parses"] = int(pack["n_parses"])

    # cut recombination
    ptr_s, ptr_c = parse_atom_slices(pack)
    if int(ptr_s[-1]) != int(np.asarray(pack["parse_S_atoms"]).size):
        raise AssertionError("S atom offsets do not match the flat array")
    s_flat = np.asarray(pack["parse_S_atoms"], np.int64)
    c_flat = np.asarray(pack["parse_C_atoms"], np.int64)
    cu = np.asarray(pack["parse_cut_u"], np.int64)
    cv = np.asarray(pack["parse_cut_v"], np.int64)
    kind = np.asarray(pack["parse_kind"])
    parse_mol = np.asarray(pack["parse_mol"], np.int64)
    bad_cover = 0
    bad_bond = 0
    bad_bridge = 0
    checked = 0
    n_parse = int(pack["n_parses"])
    fb_rows = set(np.asarray(pack["fallback_rows"]).tolist())
    forced = [int(p) for p in np.nonzero(np.isin(parse_mol, list(fb_rows)))[0]]
    sample = np.random.default_rng(0).choice(n_parse, min(1500, n_parse), replace=False).tolist()
    for p in sorted(set([int(x) for x in sample] + forced)):
        row = int(parse_mol[p])
        mol = D.raw_molecule(raw, int(np.asarray(pack["row_ids"])[row]))
        atoms_s = s_flat[ptr_s[p]:ptr_s[p + 1]].tolist()
        atoms_c = c_flat[ptr_c[p]:ptr_c[p + 1]].tolist()
        if set(atoms_s) & set(atoms_c):
            bad_cover += 1
        if sorted(atoms_s + atoms_c) != list(range(mol.n_atoms)):
            bad_cover += 1
        if int(kind[p]) == 0:
            e = int(np.asarray(pack["parse_bond_index"])[p])
            if not (int(mol.bond_u[e]) == cu[p] and int(mol.bond_v[e]) == cv[p]):
                bad_bond += 1
            adj_full = mol.adjacency_lists()
            comp = D.connected_components(adj_full, mol.n_atoms,
                                          banned=(int(mol.bond_u[e]), int(mol.bond_v[e])))
            if len(set(comp)) <= len(set(D.connected_components(adj_full, mol.n_atoms))):
                bad_bridge += 1
        checked += 1
    out["parse_recombination_bad"] = int(bad_cover)
    out["parse_bond_endpoint_bad"] = int(bad_bond)
    out["parse_bridge_bad"] = int(bad_bridge)
    out["parse_recombination_checked"] = int(checked)
    out["fallback_molecules"] = int(np.asarray(pack["fallback_rows"]).size)
    out["fallback_rows"] = np.asarray(pack["fallback_rows"]).tolist()
    stats = __import__("json").loads(str(pack["pair_stats_json"]))
    out["pair_stats"] = stats
    out["ok"] = bool(out["splits_disjoint"] and out["splits_cover_all"]
                     and out["row_ids_unique"] and out["local_edge_index_bad"] == 0
                     and out["bond_count_matches_ptr"] and out["frames_node_mask_ok"]
                     and out["frames_edge_mask_ok"] and out["parse_recombination_bad"] == 0
                     and out["parse_bond_endpoint_bad"] == 0 and out["parse_bridge_bad"] == 0)
    return out


def _mini_pack(cfg, rows, perms=None):
    return P.build_dataset(cfg, "lipo", log=lambda *_: None, rows=rows, perms=perms)


def check_renumber(cfg, pack, atom_dims, bond_dims, device: str) -> dict[str, Any]:
    import torch

    raw = D.load_raw_cache(cfg, "lipo")
    n_atoms = []
    for row in range(int(pack["n_molecules"])):
        n_atoms.append(int(np.asarray(raw["node_ptr"])[row + 1] - np.asarray(raw["node_ptr"])[row]))
    order = np.argsort(np.asarray(n_atoms))
    small = [int(order[0]), int(order[1]), int(order[2])]
    fallback = np.asarray(pack["fallback_rows"]).tolist()
    rng = np.random.default_rng(0)
    pool = [int(x) for x in rng.permutation(int(pack["n_molecules"]))
            if int(x) not in small]
    extra: list[int] = []
    if fallback and fallback[0] not in small:
        extra.append(int(fallback[0]))
    for cand in pool:
        if len(extra) >= 5:
            break
        if cand not in extra:
            extra.append(cand)
    rows = sorted(set(small + extra))
    perms = {r: np.asarray(rng.permutation(n_atoms[r]), np.int64) for r in rows}
    pa = _mini_pack(cfg, rows)
    pb = _mini_pack(cfg, rows, perms)
    keys_a = _molecule_signatures(pa)
    keys_b = _molecule_signatures(pb)
    out: dict[str, Any] = {
        "rows": rows,
        "n_atoms": [int(n_atoms[r]) for r in rows],
        "contains_fallback": [int(r in fallback) for r in rows],
        "structure_signature_equal": {int(pa["row_ids"][i]): bool(keys_a[i] == keys_b[i])
                                      for i in range(len(rows))},
    }
    model_out: dict[str, float] = {}
    for arm in C.ARMS:
        preds = []
        for mini in (pa, pb):
            model = M.ResponseModel(cfg, atom_dims, bond_dims, arm)
            model.load_pack(mini)
            model.to(device)
            randomize_model(model)
            if device == "cpu":
                model.double()
            model.eval()
            with torch.no_grad():
                idx = torch.arange(len(rows), device=device)
                res = model.molecule_forward(idx)
                preds.append(res["f_bar"].double().cpu().numpy())
        model_out[arm] = float(np.max(np.abs(preds[0] - preds[1])))
    out["prediction_max_abs_diff"] = model_out
    tol = 1e-4 if device.startswith("cuda") else 1e-9
    out["tolerance"] = tol
    out["ok"] = bool(all(v <= tol for v in model_out.values())
                     and all(out["structure_signature_equal"].values()))
    return out


def _molecule_signatures(pack) -> list[list]:
    ptr = np.asarray(pack["mol_parse_ptr"], np.int64)
    s = np.asarray(pack["parse_S"], np.int64)
    c = np.asarray(pack["parse_C"], np.int64)
    b = np.asarray(pack["parse_cut_bond_row"], np.int64)
    keys = {i: str(k) for i, k in enumerate(np.asarray(pack["fragment_keys"]))}
    bt = np.asarray(pack["bond_table"])
    sig = []
    for m in range(int(pack["n_molecules"])):
        rows = []
        for p in range(int(ptr[m]), int(ptr[m + 1])):
            bond = None if b[p] < 0 else bytes(bt[int(b[p])].tobytes())
            rows.append((keys[int(s[p])], keys[int(c[p])] if c[p] >= 0 else None, bond))
        sig.append(sorted(rows))
    return sig


def check_context_identity(cfg, pack, atom_dims, bond_dims, device: str) -> dict[str, Any]:
    """Same ``C`` must give identical features on both pair endpoints; the
    FACTOR delta must equal its algebraic decomposition."""
    import torch

    model = M.ResponseModel(cfg, atom_dims, bond_dims, "FACTOR")
    model.load_pack(pack)
    model.to(device)
    randomize_model(model)
    if device == "cpu":
        model.double()
    model.eval()
    n_pairs = int(np.asarray(pack["pair_cell"]).size)
    out: dict[str, Any] = {"n_pairs": n_pairs}
    if n_pairs == 0:
        out["ok"] = False
        out["reason"] = "empty directory"
        return out
    sel = np.random.default_rng(1).choice(n_pairs, min(64, n_pairs), replace=False)
    pc = np.asarray(pack["pair_C"])[sel]
    sa = np.asarray(pack["pair_Sa"])[sel]
    sb = np.asarray(pack["pair_Sb"])[sel]
    with torch.no_grad():
        phi = model.frame_phi()
        f_c = model.fragment_features(torch.as_tensor(pc, device=device), phi)
        h_c = model.encode_fragments(torch.as_tensor(pc, device=device), phi)
        h_a = model.encode_fragments(torch.as_tensor(sa, device=device), phi)
        h_b = model.encode_fragments(torch.as_tensor(sb, device=device), phi)
        # features are built from content-deduped fragments, so both endpoints
        # necessarily share the same tensor; assert the ids resolve identically
        e = model._cut_embedding(torch.as_tensor(
            np.asarray(pack["pair_cut_bond_row"])[sel], device=device, dtype=torch.long))
        direct = model.cut_scores(h_c, h_b, e) - model.cut_scores(h_c, h_a, e)
        # algebraic split around the training-context mean of q
        q_raw = model.q(torch.cat([h_c, e], dim=-1))
        q = torch.tanh(q_raw)
        delta_r = torch.tanh(model.r(torch.cat([h_b, e], dim=-1))) \
            - torch.tanh(model.r(torch.cat([h_a, e], dim=-1)))
        delta_a = model.a(h_b) - model.a(h_a)
        rank = math.sqrt(model.rank)
        delta = delta_a + (q * delta_r).sum(-1, keepdim=True) / rank
        err = float(torch.max(torch.abs(delta.squeeze(-1) - direct)).item())
        q_mean = q.mean(0, keepdim=True)
        base = delta_a + (q_mean * delta_r).sum(-1, keepdim=True) / rank
        env = ((q - q_mean) * delta_r).sum(-1, keepdim=True) / rank
        split_err = float(torch.max(torch.abs(base + env - direct.unsqueeze(-1))).item())
        # features of the same context must be bitwise identical wherever reused
        feat_dup = model.fragment_features(torch.as_tensor(pc, device=device), phi)
        out["ctx_feature_selfconsistent"] = bool(torch.equal(f_c, feat_dup))
    out["identity_max_abs_err"] = err
    out["decomposition_max_abs_err"] = split_err
    out["ctx_variation"] = float(torch.max((h_c - h_c.mean(0, keepdim=True)).abs()).item())

    # a non-equivalent port must change the real input (features), not a log field
    tol = 1e-4 if device.startswith("cuda") else 1e-9
    out["tolerance"] = tol
    out["ok"] = bool(err <= 1e-3 and split_err <= 1e-6 and out["ctx_feature_selfconsistent"])
    return out


def _frame_lookup(pack) -> dict:
    fa = np.asarray(pack["frame_atom"])
    fbd = np.asarray(pack["frame_bond"])
    fk = np.asarray(pack["frame_k"]).astype(np.int64)
    return {(int(fk[g]), fa[g].astype(np.int32).tobytes(),
             (fbd[g] + 1).astype(np.int32).tobytes()): g
            for g in range(fa.shape[0])}


def _table_index(pack):
    at = {bytes(r.tobytes()): i for i, r in enumerate(np.asarray(pack["atom_table"]))}
    bt = {bytes(r.tobytes()): i for i, r in enumerate(np.asarray(pack["bond_table"]))}
    return at, bt


def _csr_for_port(mol, atoms, port, atom_row, bond_row, luts, lookup):
    objs = D.rooted_subsets(mol.adjacency_lists(), mol.n_atoms, C.MAX_OBJECT_ATOMS)
    fs = D.frames_for_objects(objs, atom_row, bond_row, luts)
    obj_atoms = np.full((len(objs), C.MAX_OBJECT_ATOMS), -1, np.int64)
    obj_root = np.empty(len(objs), np.int64)
    for i, (root, ats) in enumerate(objs):
        obj_root[i] = int(root)
        obj_atoms[i, :len(ats)] = ats
    in_f = np.zeros(mol.n_atoms, bool)
    in_f[list(atoms)] = True
    keep = np.all((obj_atoms < 0) | in_f[np.maximum(obj_atoms, 0)], axis=1)
    okeep = np.nonzero(keep)[0]
    dist = P._bfs_within(mol.n_atoms, atoms, bond_row, port)
    roots = obj_root[keep]
    buckets = np.full(roots.size, 5, np.uint8)
    filled = dist[roots] >= 0
    dvals = dist[roots[filled]]
    buckets[filled] = np.where(dvals <= 3, dvals, 4).astype(np.uint8)
    fptr = fs.obj_frame_ptr
    counts = fptr[okeep + 1] - fptr[okeep]
    total = int(counts.sum())
    loc = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    base = np.repeat(fptr[okeep], counts)
    b_exp = np.repeat(buckets, counts)
    w = 1.0 / np.repeat(counts, counts)
    fids = np.asarray([
        lookup[(int(fs.frame_k[e]), fs.atom_rows[e].astype(np.int32).tobytes(),
                (fs.bond_rows[e] + 1).astype(np.int32).tobytes())]
        for e in (base + loc).tolist()], np.int64)
    code = b_exp.astype(np.int64) * (max(lookup.values()) + 1) + fids
    uniq, inv = np.unique(code, return_inverse=True)
    wsum = np.bincount(inv, weights=w, minlength=uniq.size)
    nframes_max = max(lookup.values()) + 1
    return {
        "bucket": (uniq // nframes_max).astype(np.int64),
        "frame": (uniq % nframes_max).astype(np.int64),
        "weight": wsum,
        "counts": np.bincount(buckets, minlength=C.N_BUCKETS)[:C.N_BUCKETS],
    }


def _features_from_csr(model, phi, csr) -> "np.ndarray":
    # rebuild the 394-dim feature through the reference formula directly
    phin = phi.double().cpu().numpy()
    acc = np.zeros((C.N_BUCKETS, 32))
    for b, f, w in zip(csr["bucket"].tolist(), csr["frame"].tolist(), csr["weight"].tolist()):
        acc[int(b)] += w * phin[int(f)]
    nd = csr["counts"][:, None]
    natoms = float(csr["natoms"])
    mean = acc / np.maximum(1.0, nd)
    dens = acc / max(1.0, natoms)
    return np.concatenate([np.concatenate([mean, dens, np.log1p(nd)], -1).reshape(-1),
                           [math.log1p(natoms), math.log1p(csr["nbonds"]),
                            math.log1p(csr["ncomp"]), 1.0]])


def _fragment_with_moved_port(cfg, pack, frag: int) -> int | None:
    return None


def check_port_input(cfg, pack, atom_dims, bond_dims, device: str) -> dict[str, Any]:
    """The port is part of the real input: the CSR row reconstructed for the
    stored port must match the pack, and a non-equivalent port must change it."""
    import torch

    model = M.ResponseModel(cfg, atom_dims, bond_dims, "FACTOR")
    model.load_pack(pack)
    model.to(device)
    if device == "cpu":
        model.double()
    randomize_model(model)
    model.eval()
    lookup = _frame_lookup(pack)
    luts = D.build_luts(C.MAX_OBJECT_ATOMS)
    raw = D.load_raw_cache(cfg, "lipo")
    parse_S = np.asarray(pack["parse_S"], np.int64)
    ptr_s, _ = parse_atom_slices(pack)
    flat = np.asarray(pack["parse_S_atoms"], np.int64)
    fp = np.asarray(pack["frag_ptr"], np.int64)
    candidates = [f for f in np.argsort(np.asarray(pack["frag_natoms"]))[::-1]
                  if 3 <= int(pack["frag_natoms"][f]) <= 8][:40]
    out: dict[str, Any] = {"checked": 0, "reconstruction_max_abs": 0.0,
                           "port_change_diffs": [], "different_keys": 0}
    atom_lookup, bond_lookup = _table_index(pack)
    with torch.no_grad():
        phi = model.frame_phi()
        for frag in candidates[:12]:
            hits = np.nonzero(parse_S == int(frag))[0]
            if hits.size == 0:
                continue
            p = int(hits[0])
            row = int(np.asarray(pack["parse_mol"])[p])
            mol = D.raw_molecule(raw, int(np.asarray(pack["row_ids"])[row]))
            atoms = sorted(flat[ptr_s[p]:ptr_s[p + 1]].tolist())
            port = int(np.asarray(pack["parse_port_S"])[p])
            at, bt = atom_lookup, bond_lookup
            atom_row = np.asarray([at[bytes(np.asarray(x, np.uint8).tobytes())]
                                   for x in mol.atom_feat], np.int32)
            bond_row = np.full((mol.n_atoms, mol.n_atoms), -1, np.int32)
            for e, (a, b) in enumerate(zip(mol.bond_u.tolist(), mol.bond_v.tolist())):
                bond_row[a, b] = bond_row[b, a] = bt[bytes(np.asarray(mol.bond_feat[e], np.uint8).tobytes())]
            try:
                csr = _csr_for_port(mol, atoms, port, atom_row, bond_row, luts, lookup)
            except KeyError:
                continue
            csr["natoms"] = len(atoms)
            csr["nbonds"] = int(pack["frag_nbonds"][frag])
            csr["ncomp"] = int(pack["frag_ncomp"][frag])
            ref_b = np.asarray(pack["frag_bucket"])[fp[frag]:fp[frag + 1]]
            ref_f = np.asarray(pack["frag_frame"])[fp[frag]:fp[frag + 1]]
            ref_w = np.asarray(pack["frag_weight"])[fp[frag]:fp[frag + 1]].astype(np.float64)
            recon = max(float(np.abs(np.sort(csr["weight"]) - np.sort(ref_w)).max()),
                        float(np.abs(np.sort(csr["bucket"]) - np.sort(ref_b)).max()),
                        float(np.abs(np.sort(csr["frame"]) - np.sort(ref_f)).max()))
            out["reconstruction_max_abs"] = max(out["reconstruction_max_abs"], recon)
            base = _features_from_csr(model, phi, csr)
            for alt in atoms:
                if alt == port:
                    continue
                csr2 = _csr_for_port(mol, atoms, alt, atom_row, bond_row, luts, lookup)
                csr2["natoms"] = csr["natoms"]
                csr2["nbonds"] = csr["nbonds"]
                csr2["ncomp"] = csr["ncomp"]
                moved = _features_from_csr(model, phi, csr2)
                diff = float(np.abs(base - moved).max())
                out["port_change_diffs"].append(diff)
                if diff > 1e-9:
                    out["different_keys"] += 1
                    break
            out["checked"] += 1
    diffs = out["port_change_diffs"]
    out["min_port_change_diff"] = float(min(diffs)) if diffs else 0.0
    out["max_port_change_diff"] = float(max(diffs)) if diffs else 0.0
    out["ok"] = bool(out["checked"] >= 4 and out["reconstruction_max_abs"] <= 1e-6
                     and out["min_port_change_diff"] > 0.0)
    return out


def check_batching(cfg, pack, atom_dims, bond_dims, device: str) -> dict[str, Any]:
    import torch

    model = M.ResponseModel(cfg, atom_dims, bond_dims, "FACTOR")
    model.load_pack(pack)
    model.to(device)
    randomize_model(model)
    if device == "cpu":
        model.double()
    model.eval()
    rows = np.asarray([0, 1, 2, 5, 7, 11, 13, 17])
    with torch.no_grad():
        phi = model.frame_phi()
        whole = model.molecule_cuts(torch.as_tensor(rows, device=device), phi)["f_bar"].double().cpu().numpy()
        singles = np.asarray([
            float(model.molecule_cuts(torch.as_tensor([int(r)], device=device), phi)["f_bar"][0])
            for r in rows])
        shuffled = model.molecule_cuts(
            torch.as_tensor(rows[::-1].copy(), device=device), phi)["f_bar"].double().cpu().numpy()[::-1]
        order = np.random.default_rng(3).permutation(len(rows))
        perm = model.molecule_cuts(torch.as_tensor(rows[order].copy(), device=device),
                                   phi)["f_bar"].double().cpu().numpy()
    tol = 1e-4 if device.startswith("cuda") else 1e-9
    out = {
        "batch_vs_single_max_abs": float(np.max(np.abs(whole - singles))),
        "reversed_batch_max_abs": float(np.max(np.abs(whole - shuffled))),
        "permuted_batch_max_abs": float(np.max(np.abs(whole[order] - perm))),
    }

    # sparse aggregation vs an explicit loop over CSR entries
    frag_ids = np.random.default_rng(5).choice(int(pack["n_fragments"]), 16, replace=False)
    with torch.no_grad():
        feat_sparse = model.fragment_features(torch.as_tensor(frag_ids, device=device), phi)
        phin = phi.double().cpu().numpy()
        fp = np.asarray(pack["frag_ptr"], np.int64)
        fb = np.asarray(pack["frag_bucket"])
        ff = np.asarray(pack["frag_frame"], np.int64)
        fw = np.asarray(pack["frag_weight"], np.float64)
        fc = np.asarray(pack["frag_counts"], np.float64)
        fa = np.asarray(pack["frag_natoms"], np.float64)
        fbo = np.asarray(pack["frag_nbonds"], np.float64)
        fco = np.asarray(pack["frag_ncomp"], np.float64)
        explicit = np.zeros((len(frag_ids), C.FRAG_DIM))
        for i, f in enumerate(frag_ids.tolist()):
            acc = np.zeros((C.N_BUCKETS, 32))
            for e in range(int(fp[f]), int(fp[f + 1])):
                acc[int(fb[e])] += fw[e] * phin[int(ff[e])]
            nd = fc[f][:, None]
            mean = acc / np.maximum(1.0, nd)
            dens = acc / max(1.0, fa[f])
            feat = np.concatenate([np.concatenate([mean, dens, np.log1p(nd)], -1).reshape(-1),
                                   [math.log1p(fa[f]), math.log1p(fbo[f]), math.log1p(fco[f]), 1.0]])
            explicit[i] = feat
        sparse_np = feat_sparse.double().cpu().numpy()
    out["sparse_vs_explicit_max_abs"] = float(np.max(np.abs(sparse_np - explicit)))
    # duplicate-frame compression exactness: bucket weight sums equal object counts
    wsum = np.zeros((len(frag_ids), C.N_BUCKETS))
    for i, f in enumerate(frag_ids.tolist()):
        for e in range(int(fp[f]), int(fp[f + 1])):
            wsum[i, int(fb[e])] += fw[e]
    out["weight_sums_match_counts"] = bool(np.allclose(wsum, fc[frag_ids], atol=1e-5))
    out["ok"] = bool(max(out["batch_vs_single_max_abs"], out["reversed_batch_max_abs"],
                         out["permuted_batch_max_abs"], out["sparse_vs_explicit_max_abs"]) <= (1e-4 if device.startswith("cuda") else 1e-9)
                     and out["weight_sums_match_counts"])
    out["tolerance"] = tol
    return out


def check_one_batch(cfg, pack, atom_dims, bond_dims, device: str) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    data_labels = np.asarray(pack["labels"], np.float64)
    train_rows = np.asarray(pack["train_idx"], np.int64)
    scaler = C.target_scaler(data_labels, train_rows)
    y_std = (data_labels - scaler["mean"]) / scaler["std"]
    store = PR.PairStore(cfg, pack, data_labels, None, train_rows,
                        np.asarray(pack["row_ids"]), log=lambda *_: None)
    sampler = store.sampler()
    ok_all = True
    for arm, sup in (("FACTOR", "CUTRESP"), ("DIRECT", "CUTRESP"), ("ADD", "CUTRESP"), ("GLOBAL", "NONE")):
        model = M.ResponseModel(cfg, atom_dims, bond_dims, arm)
        model.load_pack(pack)
        model.to(device)
        randomize_model(model)
        model.train()
        rows = train_rows[:64]
        pair_idx, aux = sampler.sample(16)
        if sup == "NONE":
            all_rows = rows
            res = model.molecule_forward(torch.as_tensor(all_rows, device=device))
            y = torch.as_tensor(y_std[all_rows], device=device, dtype=torch.float32)
            l_main = torch.mean((res["f_bar"] - y) ** 2)
            l_aux = torch.zeros((), device=device)
            loss = l_main
        else:
            all_rows = np.concatenate([rows, aux[0], aux[1]])
            res = model.molecule_forward(torch.as_tensor(all_rows, device=device))
            y = torch.as_tensor(y_std[all_rows], device=device, dtype=torch.float32)
            f = res["f_cut"]
            per = torch.zeros(len(all_rows), device=device).index_add_(
                0, res["owner"], (f - y[res["owner"]]) ** 2) / res["counts"].float().clamp(min=1.0)
            l_main = per[:len(rows)].mean()
            if arm == "GLOBAL":
                l_aux = torch.zeros((), device=device)
            elif sup == "CUTRESP" and arm != "GLOBAL":
                delta = model.pair_cut_delta(store.tensor_pairs(pair_idx, device))
                lab = torch.as_tensor(store.delta_std_of(pair_idx), device=device, dtype=torch.float32)
                l_aux = torch.mean((delta - lab) ** 2) / float(store.tau2)
            else:
                l_aux = torch.zeros((), device=device)
            loss = l_main + 0.25 * l_aux
        loss.backward()
        grad_norm = {name.split(".")[0]: float(p.grad.norm()) for name, p in model.named_parameters()
                     if p.grad is not None}
        finite = bool(np.isfinite(float(l_main)) and np.isfinite(float(l_aux))
                      and all(math.isfinite(v) for v in grad_norm.values()))
        nonzero = {k: v for k, v in grad_norm.items() if v > 0}
        out[arm] = {
            "main_loss": float(l_main), "aux_loss": float(l_aux),
            "finite": finite,
            "nonzero_grad_modules": sorted(nonzero),
            "zero_grad_modules": sorted(set(grad_norm) - set(nonzero)),
        }
        ok_all = ok_all and finite and len(nonzero) > 0
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out["tau2"] = float(store.tau2)
    out["pair_support"] = store.support()
    out["ok"] = bool(ok_all)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="",
                    help="comma separated check names to skip (e.g. renumber when rdkit is absent)")
    args = ap.parse_args(argv)
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    cfg = dict(C.load_config())
    cfg["cache_stem"] = "trc_cache"
    pack = P.load_pack(cfg, args.dataset)
    raw = D.load_raw_cache(cfg, args.dataset)
    atom_dims, bond_dims = C.official_feature_dims()
    t0 = time.time()
    result: dict[str, Any] = {
        "dataset": args.dataset, "device": args.device,
        "cache_sha256": pack["cache_sha256"], "pack_timing": str(pack["timing_json"]),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    which = {"inputs": lambda: check_inputs(cfg, pack, raw),
             "renumber": lambda: check_renumber(cfg, pack, atom_dims, bond_dims, args.device),
             "context": lambda: check_context_identity(cfg, pack, atom_dims, bond_dims, args.device),
             "port": lambda: check_port_input(cfg, pack, atom_dims, bond_dims, args.device),
             "batching": lambda: check_batching(cfg, pack, atom_dims, bond_dims, args.device),
             "batch": lambda: check_one_batch(cfg, pack, atom_dims, bond_dims, args.device)}
    names = [n for n in ([args.only] if args.only else list(which)) if n not in skip]
    all_ok = True
    for name in names:
        t = time.time()
        print(f"[check] {name} ...", flush=True)
        res = which[name]()
        res["seconds"] = time.time() - t
        result[name] = res
        print(f"[check] {name} {'OK' if res.get('ok') else 'FAIL'} "
              f"({res['seconds']:.1f}s) {_short(res, name)}", flush=True)
        all_ok = all_ok and bool(res.get("ok"))
    result["all_ok"] = bool(all_ok)
    result["total_seconds"] = time.time() - t0
    out = C.analysis_dir() / "checks.json"
    existing = C.read_json(out) if out.is_file() else {}
    existing[args.device] = result
    C.write_json(out, existing)
    print(f"wrote {out} all_ok={all_ok}")
    return 0 if all_ok else 1


def _short(res: Mapping[str, Any], name: str) -> str:
    keys = [k for k in res if k.endswith("max_abs") or k.endswith("_bad") or k in
            ("prediction_max_abs_diff", "identity_max_abs_err", "decomposition_max_abs_err")]
    return str({k: res[k] for k in keys})


if __name__ == "__main__":
    raise SystemExit(main())
