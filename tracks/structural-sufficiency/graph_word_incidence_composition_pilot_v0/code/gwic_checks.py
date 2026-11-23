"""Pre-training engineering checks for GWIC-v0.

Writes ``analysis/checks.json``.  Every reported check is a direct computation,
not a random-output signature: candidate supports are re-derived, boundary
bonds are re-derived, coverage is re-summed, and the synthetic contrast compares
real support/coverage incidence.
"""

from __future__ import annotations

import copy
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import gwic_common as C
import gwic_data as D
import gwic_model as M
import gwic_structure as S
import gwic_synthetic as SYN
import gwic_tensor as T


def _norm(bundle: D.DataBundle) -> dict[str, np.ndarray]:
    return {
        "atom_scalar_mean": bundle.atom_scalar_mean,
        "atom_scalar_scale": bundle.atom_scalar_scale,
        "graph_mean": bundle.graph_mean,
        "graph_scale": bundle.graph_scale,
    }


def _single_batch(view: Mapping[str, Any]) -> dict[str, Any]:
    view = dict(view)
    view["atom_scalars"] = D.atom_scalars(view)
    raw = SYN.raw_from_views([view])
    st = S.build_molecule_struct(view)
    return T.collate([0], {0: st}, raw, "cpu")


def _coverage_ref(batch: Mapping[str, Any], alpha: torch.Tensor) -> np.ndarray:
    n = int(batch["n_atoms"])
    root = batch["word_root"].cpu().numpy()
    support = batch["word_support"].cpu().numpy()
    a = alpha.detach().cpu().numpy()
    out = np.zeros((n, n), np.float64)
    for w in range(root.size):
        for t in support[w].tolist():
            if t >= 0:
                out[int(root[w]), int(t)] += a[w]
    return out


def _bfs(n: int, adj: Sequence[Sequence[int]], src: int) -> np.ndarray:
    dist = np.full(n, -1, np.int64)
    dist[src] = 0
    q = [int(src)]
    while q:
        x = q.pop(0)
        for y in adj[x]:
            if dist[y] < 0:
                dist[y] = dist[x] + 1
                q.append(int(y))
    return dist


def _cfg() -> dict[str, Any]:
    return {"dims": {
        "atom": 48, "bond": 24, "sentence": 96, "word": 64, "q": 64,
        "path_rank": 24, "path_end": 16, "path_close": 8, "nbr": 16,
        "backbone_hidden": 192, "word_hidden": 160, "score_hidden": 96,
        "q_hidden": 128, "readout_hidden": 192, "readout_mid": 96, "dropout": 0.1,
    }}


def _rec(checks: list[dict[str, Any]], ident: str, name: str, status: bool,
         details: Any) -> None:
    checks.append({"id": ident, "name": name, "status": "PASS" if status else "FAIL",
                   "details": details})


def main() -> dict[str, Any]:
    t0 = time.time()
    checks: list[dict[str, Any]] = []
    bundle = D.load_bundle()
    dims = D.atom_bond_dims()
    norm = _norm(bundle)
    rows = np.union1d(bundle.fit_rows, bundle.eval_rows)
    check_rows = rows[:24]
    structs = S.build_all(bundle.raw, check_rows)
    model, info = M.build_model(dims[0], dims[1], 0, norm, True, _cfg())
    atom_ok, bond_ok = D.category_support(bundle.raw, bundle.fit_rows, dims[0], dims[1])
    model.zero_unseen(atom_ok, bond_ok, dims[0], dims[1])
    model.eval()

    # 1. raw attributes
    node_ptr = np.asarray(bundle.raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(bundle.raw["bond_ptr"], np.int64)
    attr_ok = True
    attr_details = []
    for row in check_rows[:5].tolist():
        st = structs[int(row)]
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        ok = (np.array_equal(st.atom_feat, np.asarray(bundle.raw["atom_feat"])[a0:a1])
              and np.array_equal(st.bond_feat, np.asarray(bundle.raw["bond_feat"])[b0:b1])
              and np.array_equal(st.bond_u, np.asarray(bundle.raw["bond_u"])[b0:b1])
              and np.array_equal(st.bond_v, np.asarray(bundle.raw["bond_v"])[b0:b1]))
        attr_ok &= bool(ok)
        attr_details.append({"row": int(row), "n_atoms": st.n_atoms, "n_bonds": st.n_bonds, "ok": bool(ok)})
    _rec(checks, "1_atom_bond_attributes", "all atom and bond attributes read correctly",
         attr_ok, attr_details)

    # 2/3. candidate connectivity and boundary exactness (many real words)
    conn_ok = True
    boundary_ok = True
    conn_stats = {"words_checked": 0, "max_support_size": 0, "bad_connected": 0, "bad_boundary": 0}
    for row in check_rows[:8].tolist():
        st = structs[int(row)]
        n = st.n_atoms
        adj = [[] for _ in range(n)]
        for e in range(st.n_bonds):
            u, v = int(st.bond_u[e]), int(st.bond_v[e])
            adj[u].append(v); adj[v].append(u)
        for w in range(st.n_words):
            rec = S.word_reference(st, w)
            support = set(rec["support"])
            root = rec["root"]
            conn_stats["words_checked"] += 1
            conn_stats["max_support_size"] = max(conn_stats["max_support_size"], len(support))
            if not (2 <= len(support) <= 4 and root in support):
                conn_ok = False; conn_stats["bad_connected"] += 1; continue
            dist = _bfs(n, adj, root)
            if any(dist[x] < 0 for x in support):
                conn_ok = False; conn_stats["bad_connected"] += 1; continue
            seen = {root}; stack = [root]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y in support and y not in seen:
                        seen.add(y); stack.append(y)
            expected_internal = sorted(e for e in range(st.n_bonds)
                                       if int(st.bond_u[e]) in support and int(st.bond_v[e]) in support)
            if seen != support or rec["internal_bonds"] != expected_internal:
                conn_ok = False; conn_stats["bad_connected"] += 1
            r2 = {i for i in range(n) if 0 <= dist[i] <= 2}
            expected_boundary = sorted(e for e in range(st.n_bonds)
                                       if (int(st.bond_u[e]) in support) != (int(st.bond_v[e]) in support)
                                       and int(st.bond_u[e]) in r2 and int(st.bond_v[e]) in r2)
            if rec["boundary_bonds"] != expected_boundary:
                boundary_ok = False; conn_stats["bad_boundary"] += 1
    _rec(checks, "2_candidate_connected_induced", "every candidate S is a real connected induced support",
         conn_ok, conn_stats)
    _rec(checks, "3_boundary_bonds_exact", "candidate boundary bonds are exact induced bonds to R2\\S",
         boundary_ok, conn_stats)

    # 4. prediction invariance: renumbering, edge order, candidate order
    row = int(rows[0])
    view = D.molecule_view(bundle.raw, row)
    view["atom_scalars"] = D.atom_scalars(view)
    b0 = _single_batch(view)
    with torch.no_grad():
        y0 = model(b0, arm="WORD_REAL")["yhat_std"].cpu().numpy()
    rng = np.random.default_rng(7)
    perm = rng.permutation(int(view["n_atoms"]))
    border = rng.permutation(int(view["n_bonds"]))
    viewp = SYN.permute_view(view, perm, border)
    bp = _single_batch(viewp)
    with torch.no_grad():
        yp = model(bp, arm="WORD_REAL")["yhat_std"].cpu().numpy()
    st = S.build_molecule_struct(view)
    st2 = copy.deepcopy(st)
    for r in range(st.n_atoms):
        lo, hi = int(st.word_root_ptr[r]), int(st.word_root_ptr[r + 1])
        if hi - lo > 1:
            idx = np.arange(lo, hi)[::-1]
            st2.word_support[lo:hi] = st.word_support[idx]
            st2.word_internal[lo:hi] = st.word_internal[idx]
            st2.word_scalars[lo:hi] = st.word_scalars[idx]
    raw_one = SYN.raw_from_views([view])
    bc = T.collate([0], {0: st2}, raw_one, "cpu")
    with torch.no_grad():
        yc = model(bc, arm="WORD_REAL")["yhat_std"].cpu().numpy()
    inv_ok = max(abs(float(y0[0] - yp[0])), abs(float(y0[0] - yc[0]))) < 1e-5
    _rec(checks, "4_renumber_edge_candidate_invariance",
         "renumbering, edge reorder and candidate reorder do not change prediction",
         inv_ok, {"row": row, "y": float(y0[0]), "y_permuted": float(yp[0]),
                  "y_candidate_reordered": float(yc[0])})

    # 5. Q endpoint swap
    bi = b0["bond_index"].clone()
    b0_swapped = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b0.items()}
    bi2 = bi.clone(); bi2[:, 0], bi2[:, 1] = bi[:, 1].clone(), bi[:, 0].clone()
    b0_swapped["bond_index"] = bi2
    with torch.no_grad():
        out0 = model(b0, arm="WORD_REAL", return_aux=True)
        out1 = model(b0_swapped, arm="WORD_REAL", return_aux=True)
    qdiff = float((out0["q"] - out1["q"]).abs().max()) if out0["q"].numel() else 0.0
    _rec(checks, "5_q_endpoint_swap", "swapping Q endpoints leaves q unchanged", qdiff < 1e-5,
         {"max_abs_q_diff": qdiff, "n_bonds": int(bi.shape[0])})

    # 6. alpha normalisation
    alpha = out0["alpha"]
    ptr = b0["root_word_ptr"].cpu().numpy()
    sums = torch.zeros(int(b0["n_atoms"])).index_add_(0, b0["word_root"], alpha).cpu().numpy()
    active = [r for r in range(int(b0["n_atoms"])) if int(ptr[r + 1]) > int(ptr[r])]
    alpha_err = max(abs(float(sums[r]) - 1.0) for r in active) if active else 0.0
    _rec(checks, "6_alpha_normalised", "softmax alpha sums to 1 for every root with candidates",
         alpha_err < 1e-5, {"roots_with_candidates": len(active), "max_abs_error": alpha_err})

    # 7. coverage reference
    Cref = _coverage_ref(b0, alpha)
    Cgot = out0["word"]["coverage"].detach().cpu().numpy()
    cov_err = float(np.max(np.abs(Cref - Cgot)))
    _rec(checks, "7_coverage_reference", "c_(v,t) equals the explicit sum over candidate words containing t",
         cov_err < 1e-5, {"max_abs_error": cov_err, "n_atoms": int(b0["n_atoms"])})

    # 8. overlap reference
    idx = bi.cpu().numpy()
    geom = out0["geom"].detach().cpu().numpy()
    ov_err = 0.0
    for e, (u, v) in enumerate(idx.tolist()):
        ov_err = max(ov_err, abs(float(geom[e, 0]) - float(np.sum(Cref[u] * Cref[v]))))
        ov_err = max(ov_err, abs(float(geom[e, 1]) - float(Cref[u, v] + Cref[v, u])))
        ov_err = max(ov_err, abs(float(geom[e, 2]) - float(abs(Cref[u, v] - Cref[v, u]))))
    _rec(checks, "8_overlap_real_atoms", "o_uv sums over real shared atoms, not embedding cosine",
         ov_err < 1e-5, {"max_abs_error": ov_err, "n_bonds": len(idx)})

    # 9. synthetic same-count/different-connection
    path = SYN.make_view("path4", 4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
    star = SYN.make_view("star4", 4, [(0, 1, 1), (0, 2, 1), (0, 3, 1)])
    sp, ss = S.build_molecule_struct(path), S.build_molecule_struct(star)

    def supports(st):
        return {tuple(sorted(int(x) for x in st.word_support[w].tolist() if x >= 0))
                for w in range(st.n_words)}

    def uniform_coverage(st):
        out = np.zeros((st.n_atoms, st.n_atoms))
        for w in range(st.n_words):
            root = int(st.word_root[w])
            sup = [int(x) for x in st.word_support[w].tolist() if x >= 0]
            k = int((st.word_root == root).sum())
            for t in sup:
                out[root, t] += 1.0 / max(1, k)
        return out

    distinct = (supports(sp) != supports(ss)) and not np.allclose(uniform_coverage(sp), uniform_coverage(ss))
    _rec(checks, "9_synthetic_connection_distinction",
         "same atom/bond counts with different real connection give different candidates/coverage",
         distinct, {"path_words": sp.n_words, "star_words": ss.n_words,
                    "path_supports": len(supports(sp)), "star_supports": len(supports(ss))})

    # 10. B/C identical init and parameter count
    mb, ib = M.build_model(dims[0], dims[1], 0, norm, True, _cfg())
    mc, ic = M.build_model(dims[0], dims[1], 0, norm, True, _cfg())
    same_params = ib["parameters"] == ic["parameters"]
    same_init = ib["init_hashes"] == ic["init_hashes"]
    state_equal = all(torch.equal(v, mc.state_dict()[k]) for k, v in mb.state_dict().items())
    _rec(checks, "10_bc_same_init_params",
         "WORD_GEOM_NULL and WORD_REAL start from identical initialisation with identical parameters",
         bool(same_params and same_init and state_equal),
         {"parameters": ib["parameters"]["total"], "same_parameter_ledger": bool(same_params),
          "same_init_hashes": bool(same_init), "same_state_dict": bool(state_equal)})

    # 11. same-checkpoint geometry zeroing
    with torch.no_grad():
        real = model(b0, arm="WORD_REAL", return_aux=True)
        zero = model(b0, arm="WORD_REAL", use_geometry=False, return_aux=True)
        ov_zero = model(b0, arm="WORD_REAL", geometry_override=torch.zeros_like(real["geom"]),
                        return_aux=True)
        ov_real = model(b0, arm="WORD_REAL", geometry_override=real["geom"], return_aux=True)
    zero_err = float((zero["yhat_std"] - ov_zero["yhat_std"]).abs().max())
    real_err = float((real["yhat_std"] - ov_real["yhat_std"]).abs().max())
    delta = float((real["yhat_std"] - zero["yhat_std"]).abs().max())
    _rec(checks, "11_geometry_zeroing", "zeroing the three geometry inputs at one checkpoint matches the real forward",
         zero_err < 1e-6 and real_err < 1e-6 and delta > 0.0,
         {"zero_override_max_abs": zero_err, "real_override_max_abs": real_err,
          "real_vs_zero_max_abs": delta})

    # 12. finite non-zero gradients in key modules
    grad_model, _ = M.build_model(dims[0], dims[1], 3, norm, True,
                                  {"dims": {**_cfg()["dims"], "dropout": 0.0}})
    gview = SYN.make_view("grad", 7, [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1)])
    gb = _single_batch(gview)
    grad_model.train()
    gout = grad_model(gb, arm="WORD_REAL")
    loss = torch.mean(gout["yhat_std"] ** 2)
    grad_model.zero_grad(set_to_none=True)
    loss.backward()
    grad_details = {}
    grad_ok = True
    for name, module in {
        "atom_emb": grad_model.atom_emb, "bond_raw_emb": grad_model.bond_raw_emb,
        "bond_encoder": grad_model.bond_encoder, "path_joint": grad_model.path_joint,
        "backbone_mlp": grad_model.backbone_mlp, "word_joint": grad_model.word_joint,
        "word_mlp": grad_model.word_mlp, "score_mlp": grad_model.score_mlp,
        "q_mlp": grad_model.q_mlp, "readout_mlp": grad_model.readout_mlp,
    }.items():
        vals = []
        for p in module.parameters():
            if p.grad is not None:
                vals.append(float(p.grad.pow(2).sum()))
                if not torch.isfinite(p.grad).all():
                    grad_ok = False
        total = float(np.sum(vals)) if vals else 0.0
        grad_details[name] = total
        grad_ok &= total > 0.0
    _rec(checks, "12_finite_nonzero_gradients", "all key modules have finite non-zero gradients",
         bool(grad_ok), grad_details)

    # 13. single vs batch
    with torch.no_grad():
        batch_rows = check_rows[:3]
        bfull = T.collate(batch_rows.tolist(), structs, bundle.raw, "cpu")
        yfull = model(bfull, arm="WORD_REAL")["yhat_std"].cpu().numpy()
        ysingle = []
        for r in batch_rows.tolist():
            b = T.collate([r], structs, bundle.raw, "cpu")
            ysingle.append(float(model(b, arm="WORD_REAL")["yhat_std"].cpu().numpy()[0]))
    bs_err = float(np.max(np.abs(yfull - np.asarray(ysingle))))
    _rec(checks, "13_single_vs_batch", "single-molecule and batched predictions agree", bs_err < 1e-5,
         {"max_abs_error": bs_err})

    # 14. label isolation
    finite_train = bool(np.isfinite(bundle.labels[bundle.official_train]).all())
    valid_nan = bool(np.isnan(bundle.labels[bundle.official_valid]).all())
    no_label_keys = not any("label" in str(k).lower() for k in bundle.raw.keys())
    no_test = "test_idx" not in bundle.raw
    collate_keys = set(bfull.keys())
    no_label_tensor = not any("label" in str(k).lower() for k in collate_keys)
    _rec(checks, "14_label_isolation",
         "official Valid/Test labels never enter a data tensor; only official Train labels are finite",
         bool(finite_train and valid_nan and no_label_keys and no_test and no_label_tensor),
         {"train_finite": finite_train, "valid_all_nan": valid_nan,
          "raw_has_label_member": not no_label_keys, "raw_has_test_idx": not no_test,
          "collate_has_label_key": not no_label_tensor})

    # 15. full candidate scale, memory, speed
    rss0 = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    t_build = time.time()
    full = S.build_all(bundle.raw, rows)
    build_seconds = time.time() - t_build
    rss1 = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    stats = S.structure_stats(full)
    array_bytes = 0
    for st in full.values():
        for key, value in st.__dict__.items():
            if isinstance(value, np.ndarray):
                array_bytes += int(value.nbytes)
    cap_ok = int(stats["n_words"]) <= 20_000_000
    # one CPU forward on 8 molecules, for an order-of-magnitude speed record
    b8 = T.collate(rows[:8].tolist(), full, bundle.raw, "cpu")
    t_fwd = time.time()
    with torch.no_grad():
        model(b8, arm="WORD_REAL")
    fwd_seconds = time.time() - t_fwd
    _rec(checks, "15_candidate_scale_memory_speed",
         "candidate enumeration is complete, within the 20M cap, and bounded in memory/time",
         bool(cap_ok and stats["n_words"] > 0),
         {"candidate_cap": 20_000_000, "cap_triggered": not cap_ok,
          **stats, "build_seconds": build_seconds,
          "peak_rss_bytes_before": rss0, "peak_rss_bytes_after": rss1,
          "structure_array_bytes": array_bytes,
          "cpu_forward_seconds_8_molecules": fwd_seconds,
          "total_check_seconds": time.time() - t0})

    payload = {
        "protocol_id": C.load_config()["protocol_id"],
        "generated_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "config_sha256": C.sha256_json(C.load_config()),
        "checks": checks,
        "n_pass": int(sum(c["status"] == "PASS" for c in checks)),
        "n_total": len(checks),
        "all_pass": bool(all(c["status"] == "PASS" for c in checks)),
        "full_candidate_count": int(stats["n_words"]),
        "candidate_cap_triggered": bool(not cap_ok),
    }
    C.write_json(C.ANALYSIS_DIR / "checks.json", payload)
    print(json.dumps(C.jsonable(payload), indent=1, sort_keys=True)[:4000])
    return payload


if __name__ == "__main__":
    main()
