"""CPU correctness / numerical / semantics gates.  No labels are read except
the frozen valid labels for a save-reload replay identity check (allowed: valid
is the development split).  Official test is never constructed.
"""

from __future__ import annotations

import json
import traceback

import numpy as np
import torch

import nl_bridge
import nl_common as N
import nl_data as ND
import nl_models as NM

torch.set_num_threads(4)
RES: dict = {"checks": {}, "official_test_loaded": False}
FAIL = []


def record(name, ok, **kw):
    RES["checks"][name] = {"pass": bool(ok), **kw}
    if not ok:
        FAIL.append(name)
    print(("PASS " if ok else "FAIL ") + name, kw)


def mini_cache(obj, perm=None, reverse=False):
    mol = N.D.build_molecule_from_object(obj, perm=perm, reverse_bonds=reverse)
    cache = N.D._concat_molecules([mol])
    cache["labels"] = np.zeros(1, dtype=np.float32)
    cache["n_molecules"] = 1
    return cache


def check_bridge_enumeration(n=200):
    import networkx as nx
    bad_bridge = bad_side = 0
    checked = 0
    for split, count in (("train", n), ("valid", n // 2)):
        cache = N.D.load_cache(split)
        atom_ptr = np.asarray(cache["atom_ptr"])
        bond_ptr = np.asarray(cache["bond_ptr"])
        bonds = np.asarray(cache["bond_index"])
        for g in range(count):
            nn = int(atom_ptr[g + 1] - atom_ptr[g])
            bi = bonds[bond_ptr[g]:bond_ptr[g + 1]]
            got = nl_bridge._bridges(nn, bi)
            G = nx.Graph()
            G.add_nodes_from(range(nn))
            G.add_edges_from((int(u), int(v)) for u, v in bi)
            brset = {tuple(sorted(map(int, x))) for x in nx.bridges(G)}
            want = sorted(int(e) for e in range(len(bi))
                          if tuple(sorted((int(bi[e, 0]), int(bi[e, 1])))) in brset)
            if got != want:
                bad_bridge += 1
            for e in got:
                A = nl_bridge._side(nn, bi, e, int(bi[e, 0]))
                B = nl_bridge._side(nn, bi, e, int(bi[e, 1]))
                if sorted(A + B) != list(range(nn)):
                    bad_side += 1
                if len(A) + len(B) != nn:
                    bad_side += 1
            checked += 1
    record("bridge_enumeration", bad_bridge == 0 and bad_side == 0,
           molecules=checked, bridge_mismatch=bad_bridge, side_mismatch=bad_side)


def check_init_identity():
    norm = N.load_norm()
    hist, _ = N.M.build_model("A", norm, 0, "cpu")
    hist_hash = N.L.C.state_hash(hist)
    expected = "11c698ae669dbee1bca93ccc13048163e725b29659d54fe6498fa6440a0cfb1e"
    cache = N.D.load_cache("valid")
    bridge = nl_bridge.load("valid")
    rows = np.arange(3)
    batch = ND.collate(cache, bridge, rows, with_bridge=True)
    out = {}
    for head, mode in (("sce", "real"), ("sce", "marginal"), ("gef", "real"),
                       ("gef", "fixed"), ("gef", "local")):
        model, info = NM.build_model(head, norm, 0, "cpu", mode=mode, lam=N.LAMBDA_GEF)
        trunk_hash = NM._state_hash(model.trunk.state_dict())
        same = all(torch.equal(v, hist.state_dict()[k])
                   for k, v in model.trunk.state_dict().items())
        with torch.no_grad():
            y_new, aux = model(batch, return_aux=True)
            y_hist = hist(batch)
        out[f"{head}-{mode}"] = {
            "params": info["parameters"],
            "head_params": info["parameters_head"],
            "trunk_hash": trunk_hash,
            "trunk_bit_equal": bool(same),
            "max_abs_dy_step0": float((y_new - y_hist).abs().max()),
            "term_std": float(aux["term"].std()),
            "term_nonzero": bool(torch.count_nonzero(aux["term"]).item() > 0),
        }
    ok = (hist_hash == expected) and all(v["trunk_bit_equal"] for v in out.values()) \
        and all(v["term_nonzero"] for v in out.values()) \
        and all(v["params"] <= 500_000 for v in out.values())
    record("init_identity_and_small_nonzero_head", ok, hist_hash=hist_hash,
           expected=expected, models=out)


def check_invariance():
    objs, _ = N.L.D.load_p3_objects("valid")
    rng = np.random.default_rng(20261029)
    norm = N.load_norm()
    models = {}
    for head, mode in (("sce", "real"), ("gef", "real")):
        torch.manual_seed(0)
        m, _ = NM.build_model(head, norm, 0, "cpu", mode=mode, lam=N.LAMBDA_GEF)
        models[(head, mode)] = m
    res = {}
    for head, mode in models:
        m = models[(head, mode)]
        m.eval()
        diffs = {"renumber": 0.0, "bond_flip": 0.0}
        for g in range(4):
            obj = objs[g]
            base = mini_cache(obj)
            bb = nl_bridge.build_split(base)
            rows = np.arange(1)
            batch = ND.collate(base, bb, rows, with_bridge=(head == "sce"))
            with torch.no_grad():
                y0 = m(batch).item()
            n = int(base["atom_ptr"][1])
            perm = rng.permutation(n)
            c2 = mini_cache(obj, perm=perm)
            b2 = nl_bridge.build_split(c2)
            batch2 = ND.collate(c2, b2, rows, with_bridge=(head == "sce"))
            with torch.no_grad():
                y1 = m(batch2).item()
            c3 = mini_cache(obj, reverse=True)
            b3 = nl_bridge.build_split(c3)
            batch3 = ND.collate(c3, b3, rows, with_bridge=(head == "sce"))
            with torch.no_grad():
                y2 = m(batch3).item()
            diffs["renumber"] = max(diffs["renumber"], abs(y0 - y1))
            diffs["bond_flip"] = max(diffs["bond_flip"], abs(y0 - y2))
        res[f"{head}-{mode}"] = diffs
    ok = all(v["renumber"] <= 1e-5 and v["bond_flip"] <= 1e-5 for v in res.values())
    record("renumbering_and_bond_direction_invariance", ok, diffs=res)


def check_batch_split():
    norm = N.load_norm()
    cache = N.D.load_cache("valid")
    bridge = nl_bridge.load("valid")
    rows = np.arange(8)
    res = {}
    for head, mode in (("sce", "real"), ("gef", "real")):
        m, _ = NM.build_model(head, norm, 0, "cpu", mode=mode, lam=N.LAMBDA_GEF)
        m.eval()
        with torch.no_grad():
            y_batch = m(ND.collate(cache, bridge, rows, with_bridge=(head == "sce")))
            singles = torch.cat([
                m(ND.collate(cache, bridge, np.asarray([r]), with_bridge=(head == "sce")))
                for r in rows])
            halves = m(ND.collate(cache, bridge, rows[:4], with_bridge=(head == "sce")))
        res[f"{head}-{mode}"] = {
            "split_max": float((y_batch - singles).abs().max()),
            "half_max": float((y_batch[:4] - halves).abs().max()),
        }
    ok = all(v["split_max"] <= 1e-5 and v["half_max"] <= 1e-5 for v in res.values())
    record("batch_merge_split_invariance", ok, diffs=res)


def check_save_reload():
    norm = N.load_norm()
    cache = N.D.load_cache("valid")
    bridge = nl_bridge.load("valid")
    rows = np.arange(16)
    res = {}
    for head, mode in (("sce", "real"), ("gef", "real")):
        m, _ = NM.build_model(head, norm, 0, "cpu", mode=mode, lam=N.LAMBDA_GEF)
        state = N.L.C.cpu_state(m)
        m2 = NM.load_model(head, norm, state, "cpu", None, mode=mode, lam=N.LAMBDA_GEF)
        m.eval(); m2.eval()
        batch = ND.collate(cache, bridge, rows, with_bridge=(head == "sce"))
        with torch.no_grad():
            y1 = m(batch)
            y2 = m2(batch)
            y3 = m2(batch)  # deterministic re-eval
        res[f"{head}-{mode}"] = {"reload_max": float((y1 - y2).abs().max()),
                                 "repeat_max": float((y2 - y3).abs().max())}
    ok = all(v["reload_max"] == 0.0 and v["repeat_max"] == 0.0 for v in res.values())
    record("save_reload_bit_identical", ok, diffs=res)


def check_gradients_and_steps():
    norm = N.load_norm()
    cache = N.D.load_cache("train")
    bridge = nl_bridge.load("train")
    rows = np.arange(32)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    target = torch.as_tensor(y, dtype=torch.float32)
    res = {}
    for head, mode in (("sce", "real"), ("sce", "marginal"), ("gef", "real"),
                       ("gef", "fixed"), ("gef", "local")):
        m, _ = NM.build_model(head, norm, 0, "cpu", mode=mode, lam=N.LAMBDA_GEF)
        batch = ND.collate(cache, bridge, rows, with_bridge=(head == "sce"))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        head_mod = m.sce if m.sce is not None else m.gef
        steps = []
        prev_term = None
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            pred, aux = m(batch, return_aux=True)
            loss = torch.nn.functional.l1_loss(pred, target)
            loss.backward()
            stats = {}
            for name, p in head_mod.named_parameters():
                g = p.grad
                if g is None:
                    stats[name] = {"used": False}
                else:
                    stats[name] = {"used": True,
                                   "nonzero": bool(torch.count_nonzero(g).item() > 0),
                                   "finite": bool(torch.isfinite(g).all().item()),
                                   "norm": float(g.norm())}
            trunk_grad = float(dict(m.trunk.named_parameters())["f.lins.0.weight"].grad.norm())
            term = aux["term"].detach().clone()
            term_change = None if prev_term is None else float((term - prev_term).abs().max())
            prev_term = term
            steps.append({"loss": float(loss), "params": stats, "trunk_grad": trunk_grad,
                          "term_change": term_change})
            opt.step()
        res[f"{head}-{mode}"] = steps
    ok = True
    for v in res.values():
        for st in v:
            ok = ok and st["trunk_grad"] > 0 and st["loss"] == st["loss"]
            ok = ok and all((not p["used"]) or
                            (p["nonzero"] and p["finite"] and np.isfinite(p["norm"]))
                            for p in st["params"].values())
            ok = ok and all(p["used"] for p in st["params"].values()
                            if mode in ("real", "marginal"))
    record("head_gradients_step1_3_nonzero_finite", ok, steps=res)


def _brute_gef(h, bond_index, w, lam):
    n = h.shape[0]
    W = np.zeros((n, n))
    for e, (u, v) in enumerate(bond_index):
        W[u, v] = W[v, u] = w[e]
    L = np.diag(W.sum(1)) - W
    K = np.linalg.inv(lam * np.eye(n) + L)
    return K, L


def check_gef_math():
    rng = np.random.default_rng(7)
    n, m = 7, 9
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (0, 3), (2, 5)]
    h = torch.tensor(rng.normal(size=(n, 8)), dtype=torch.float32)
    z = torch.tensor(rng.normal(size=(n, 5)), dtype=torch.float32)
    w_np = rng.uniform(0.2, 2.0, size=len(edges))
    w = torch.tensor(w_np, dtype=torch.float32)
    lam = 0.3

    # inject fixed conduction by monkey-patching the module method
    norm = {"global_mean": np.zeros(33, np.float32), "global_scale": np.ones(33, np.float32),
            "scalar_mean": np.zeros(9, np.float32), "scalar_scale": np.ones(9, np.float32)}
    head = NM.GEFHead(8, 1, rank=5, lam=lam, mode="fixed")
    head.eval()
    batch = {
        "bond_index": torch.tensor(edges),
        "bond_batch": torch.zeros(len(edges), dtype=torch.long),
        "atom_counts": torch.tensor([n]),
        "atom_batch": torch.zeros(n, dtype=torch.long),
        "atom_off": torch.tensor([0]),
        "bond_feat": torch.zeros((len(edges), 1)),
    }
    # the fixed mode gives w=1
    K_ref, L_ref = _brute_gef(h.numpy().astype(np.float64), edges, np.ones(len(edges)), lam)
    M = K_ref * (1 - np.eye(n))
    c_ref = (M @ z.numpy().astype(np.float64)) / M.sum(1, keepdims=True) \
        - (z.numpy().astype(np.float64).sum(0) - z.numpy().astype(np.float64)) / (n - 1)
    c, aux = head.field(h, z, torch.zeros((len(edges), 1)), batch, 1)
    Kn = aux["K"].numpy()[0, :n, :n]
    num_err = float(np.abs(Kn - K_ref).max())
    c_err = float(np.abs(c.numpy() - c_ref).max())
    resid = float(np.abs(Kn @ (lam * np.eye(n) + L_ref) - np.eye(n)).max())
    offdiag_min = float(Kn[~np.eye(n, dtype=bool)].min())
    # uniform source -> c = 0
    z0 = torch.ones((n, 5)) * 3.7
    c0, _ = head.field(h, z0, torch.zeros((len(edges), 1)), batch, 1)
    # small graphs
    small = []
    for nn in (1, 2, 3):
        hh = torch.randn(nn, 8)
        zz = torch.randn(nn, 5)
        ee = [(i, i + 1) for i in range(nn - 1)]
        bb = {
            "bond_index": torch.tensor(ee, dtype=torch.long).reshape(-1, 2),
            "bond_batch": torch.zeros(max(len(ee), 0), dtype=torch.long),
            "atom_counts": torch.tensor([nn]),
            "atom_batch": torch.zeros(nn, dtype=torch.long),
            "atom_off": torch.tensor([0]),
            "bond_feat": torch.zeros((len(ee), 1)),
        }
        cc, aa = head.field(hh, zz, torch.zeros((len(ee), 1)), bb, 1)
        small.append({"n": nn, "finite": bool(torch.isfinite(cc).all()),
                      "max_abs": float(cc.abs().max())})
    ok = num_err <= 1e-5 and c_err <= 1e-5 and resid <= 1e-6 and offdiag_min >= -1e-7 \
        and float(c0.abs().max()) <= 1e-6 and all(s["finite"] for s in small)
    record("gef_math_bruteforce", ok, K_max_err=num_err, c_max_err=c_err,
           inverse_residual=resid, K_offdiag_min=offdiag_min,
           uniform_source_c_max=float(c0.abs().max()), small_graphs=small)


def check_structural_witness():
    """Same local-state multiset, different structure correspondence -> different output."""
    norm = N.load_norm()
    n = 4
    rng = np.random.default_rng(11)
    h = torch.tensor(rng.normal(size=(n, 64)), dtype=torch.float32)
    b = torch.zeros((n - 1, 96))
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    # two assignments of the same h multiset to a 4-path; the bridge is edge 1.
    order_a = torch.arange(n)
    order_b = torch.tensor([0, 2, 1, 3])
    res = {}
    for head in ("sce", "gef"):
        torch.manual_seed(0)
        m, _ = NM.build_model(head, norm, 0, "cpu", mode="real", lam=N.LAMBDA_GEF)
        m.eval()
        out = []
        for order in (order_a, order_b):
            hh = h[order]
            batch = {
                "n_graphs": 1,
                "atom_batch": torch.zeros(n, dtype=torch.long),
                "atom_off": torch.tensor([0]),
                "atom_counts": torch.tensor([n]),
                "bond_index": edges,
                "bond_batch": torch.zeros(len(edges), dtype=torch.long),
                "bond_feat": torch.zeros((len(edges), 7)),
                "global": torch.zeros((1, 33)),
                # features used only by the trunk; head is fed hh directly
                "atom_feat": torch.zeros((n, 108)),
                "atom_scalar": torch.zeros((n, 9)),
                "nbr_atom": torch.zeros(0, dtype=torch.long),
                "nbr_code": torch.zeros(0, dtype=torch.long),
                "path_atom": torch.zeros(0, dtype=torch.long),
                "path_cat": torch.zeros(0, dtype=torch.long),
                "path_end63": torch.zeros(0, dtype=torch.long),
                "path_end": torch.zeros(0, dtype=torch.long),
                "path_closure": torch.zeros(0, dtype=torch.long),
                "pair_i": torch.zeros(0, dtype=torch.long),
                "pair_j": torch.zeros(0, dtype=torch.long),
                "pair_cat": torch.zeros(0, dtype=torch.long),
                "pair_graph": torch.zeros(0, dtype=torch.long),
            }
            if head == "sce":
                bb = nl_bridge.build_split({
                    "atom_ptr": np.asarray([0, n]),
                    "bond_ptr": np.asarray([0, len(edges)]),
                    "bond_index": edges.numpy(),
                })
                batch["bridge"] = nl_bridge.collate(
                    {"atom_ptr": np.asarray([0, n]), "bond_ptr": np.asarray([0, len(edges)]),
                     "bond_index": edges.numpy()}, bb, np.arange(1))
                with torch.no_grad():
                    term, aux = m.sce(hh, b, batch, batch["bridge"], 1)
                out.append({"I_e": aux["I_e"].numpy().tolist(), "term": float(term)})
            else:
                with torch.no_grad():
                    term, aux = m.gef(hh, b, batch, 1)
                out.append({"c": aux["c"].numpy().tolist(), "term": float(term)})
        res[head] = out
    ok = abs(res["sce"][0]["I_e"][0] - res["sce"][1]["I_e"][0]) > 1e-4 \
        and abs(res["sce"][0]["term"] - res["sce"][1]["term"]) > 1e-4 \
        and abs(res["gef"][0]["term"] - res["gef"][1]["term"]) > 1e-5
    record("structural_witness_same_multiset", ok, witness=res)


def check_zero_bridge():
    """A 6-cycle has no bridges -> SCE term must be exactly zero."""
    norm = N.load_norm()
    n = 6
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
    cache = {"atom_ptr": np.asarray([0, n]), "bond_ptr": np.asarray([0, len(edges)]),
             "bond_index": edges.numpy()}
    bb = nl_bridge.build_split(cache)
    sub = {"atom_ptr": np.asarray([0, n]), "bond_ptr": np.asarray([0, len(edges)]),
           "bond_index": edges.numpy()}
    bcol = nl_bridge.collate(sub, bb, np.arange(1))
    m, _ = NM.build_model("sce", norm, 0, "cpu", mode="real", lam=N.LAMBDA_GEF)
    m.eval()
    h = torch.randn(n, 64)
    b = torch.zeros(len(edges), 96)
    with torch.no_grad():
        term, aux = m.sce(h, b, {"n_graphs": 1, "atom_batch": torch.zeros(n, dtype=torch.long)},
                          bcol, 1)
    record("zero_bridge_term_exact_zero", bcol["n_bridges"] == 0 and float(term.abs().max()) == 0.0,
           n_bridges=int(bcol["n_bridges"]), term=float(term))


def main():
    np.set_printoptions(suppress=True)
    try:
        check_bridge_enumeration()
        check_init_identity()
        check_invariance()
        check_batch_split()
        check_save_reload()
        check_gradients_and_steps()
        check_gef_math()
        check_structural_witness()
        check_zero_bridge()
    except Exception:
        RES["traceback"] = traceback.format_exc()
        FAIL.append("exception")
        print(RES["traceback"])
    RES["all_pass"] = len(FAIL) == 0
    RES["failed"] = FAIL
    RES["created"] = N.L.now()
    N.ensure_dir(N.ANALYSIS)
    N.L.write_json(N.ANALYSIS / "cpu_checks.json", RES)
    print("ALL_PASS", RES["all_pass"], "failed", FAIL)


if __name__ == "__main__":
    main()
