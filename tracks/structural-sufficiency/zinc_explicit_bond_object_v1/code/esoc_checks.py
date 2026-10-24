"""ESOC-v1 pre-training gates.

Every gate writes into ``analysis/cpu_checks.json`` and must pass before the
formal GPU run.  The gates answer "is the implementation the thing we declared",
not "does it work well".

Suites
------
invariance        atom renumbering, bond-direction flip, batch composition
local_exactness   cached typed induced objects vs an independent networkx enumeration
synthetic         empty / single bond / tree / triangle / K4 / benzene
real_null         parameter, init, data-order and optimiser identity
correspondence    the REAL arm reads pair-edge correspondence (forward + gradient),
                  NULL provably does not; class marginals are identical
witness           synthetic objects with identical marginals/roles, different pairs
numerics          sL2 and AdaRMSNorm vs independent float64 references, RRWP parity
masking           padding, cross-molecule isolation, finiteness
budget            <=500k parameters, optimiser covers every trainable tensor
provenance        no test split, no label read on the static-object path
"""

from __future__ import annotations

import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_SS / "zinc_arch_discovery_v1" / "code"),
           str(_SS / "zinc_relation_composition_probe_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import esoc_data as D  # noqa: E402
import esoc_models as M  # noqa: E402
import esoc_train as T  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _scales() -> dict:
    return T.read_json(ANALYSIS / "scales.json")


def _role_base_consistent() -> bool:
    return D.ROLE_CAP + 1 == M.ROLE_BASE


def _pyg_raw(split: str, index: int) -> dict:
    """Raw atoms/bonds straight from the official PyG dataset (no P3 objects)."""
    from torch_geometric.datasets import ZINC

    ds = ZINC(root=str(C.ZINC_ROOT), subset=True,
              split=("val" if split == "valid" else split))
    d = ds[index]
    atom_type = d.x.reshape(-1).long().numpy()
    ei = d.edge_index.numpy()
    ea = d.edge_attr.reshape(-1).long().numpy()
    seen: dict[tuple[int, int], int] = {}
    for k in range(ei.shape[1]):
        a, b, t = int(ei[0, k]), int(ei[1, k]), int(ea[k])
        key = (a, b) if a < b else (b, a)
        if key in seen and seen[key] != t:
            raise RuntimeError("inconsistent bond type across directions")
        seen[key] = t
    keys = sorted(seen)
    return {
        "atom_type": atom_type,
        "bonds": np.asarray(keys, dtype=np.int64).reshape(-1, 2),
        "bond_type": np.asarray([seen[k] for k in keys], dtype=np.int64),
    }


def _cache_mol(split: str, mol: int) -> dict:
    data = D.load_cache(split)
    ap, bp = data["atom_ptr"], data["bond_ptr"]
    return {
        "atom_type": data["atom_type"][ap[mol]:ap[mol + 1]].astype(np.int64),
        "bonds": data["bond_index"][bp[mol]:bp[mol + 1]].astype(np.int64),
        "bond_type": data["bond_type"][bp[mol]:bp[mol + 1]].astype(np.int64),
        "global33": data["global33"][mol],
    }


def _brute_objects(spec: dict) -> list[dict]:
    """Independent enumeration with networkx (no shared code with esoc_data)."""
    import networkx as nx

    at = np.asarray(spec["atom_type"]).reshape(-1)
    bonds = np.asarray(spec["bonds"]).reshape(-1, 2)
    btype = np.asarray(spec["bond_type"]).reshape(-1)
    G = nx.Graph()
    G.add_nodes_from(range(len(at)))
    for (u, v), t in zip(bonds.tolist(), btype.tolist()):
        G.add_edge(int(u), int(v), bt=int(t))
    out = []
    for k in range(len(bonds)):
        u, v = int(bonds[k, 0]), int(bonds[k, 1])
        P = sorted({u, v} | set(G[u]) | set(G[v]))
        du = nx.single_source_shortest_path_length(G, u)
        dv = nx.single_source_shortest_path_length(G, v)
        et = []
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                i, j = P[a], P[b]
                et.append(int(G[i][j]["bt"]) if G.has_edge(i, j) else 0)
        out.append({
            "nodes": P,
            "types": [int(at[i]) for i in P],
            "roles": [(int(du[i]), int(dv[i])) for i in P],
            "et": et,
            "n_edges": int(sum(1 for x in et if x > 0)),
        })
    return out


def _pred(model, batch) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(batch).double().cpu().numpy()


def _permute_pair_edges(loc_et: torch.Tensor, loc_mask: torch.Tensor, seed: int = 0):
    """Keep the per-object edge-class multiset, permute which pair gets which class."""
    rng = np.random.default_rng(seed)
    et = loc_et.clone()
    B, E, L, _ = et.shape
    tri = torch.triu(torch.ones(L, L, dtype=torch.bool, device=loc_et.device), diagonal=1)
    for b in range(B):
        for e in range(E):
            nm = loc_mask[b, e]
            if not bool(nm.any()):
                continue
            idx = (tri & nm.unsqueeze(1) & nm.unsqueeze(0)).nonzero()
            if len(idx) == 0:
                continue
            vals = [int(et[b, e, i, j]) for i, j in idx.tolist()]
            vals = [vals[k] for k in rng.permutation(len(vals))]
            for (i, j), val in zip(idx.tolist(), vals):
                et[b, e, i, j] = val
                et[b, e, j, i] = val
    return et


# ---------------------------------------------------------------------------
# independent float64 references
# ---------------------------------------------------------------------------
def _ref_sl2(q, k, v, bias, mask):
    B, H, E, dh = q.shape
    out = np.zeros_like(q)
    scale = 1.0 / math.sqrt(dh)
    for b in range(B):
        for h in range(H):
            for i in range(E):
                if not mask[b, i]:
                    continue
                s = np.full(E, -1e9, dtype=np.float64)
                for j in range(E):
                    if not mask[b, j]:
                        continue
                    num = float(np.dot(q[b, h, i], k[b, h, j]))
                    kn = float(np.dot(k[b, h, j], k[b, h, j]))
                    s[j] = (num - 0.5 * kn) * scale + float(bias[b, h, i, j])
                mx = float(s.max())
                w = np.exp(s - mx)
                w = w / w.sum()
                acc = np.zeros(dh, dtype=np.float64)
                for j in range(E):
                    if w[j] != 0.0:
                        acc = acc + w[j] * v[b, h, j]
                out[b, h, i] = acc
    return out


def _ref_adarmsnorm(x, scale, shift, ps, pb, g, eps):
    rms = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)
    mod_s = np.einsum("br,dr->bd", g, ps)
    mod_b = np.einsum("br,dr->bd", g, pb)
    return rms * (1.0 + scale[None, None, :] + mod_s[:, None, :]) \
        + shift[None, None, :] + mod_b[:, None, :]


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------
def suite_invariance(model, device) -> dict:
    out = {}
    spec = _cache_mol("valid", 3)
    batch = D.batch_from_specs([spec], device)
    p0 = _pred(model, batch)

    rng = np.random.default_rng(11)
    perm = rng.permutation(len(spec["atom_type"]))
    inv = np.argsort(perm)
    spec2 = {
        "atom_type": spec["atom_type"][perm],
        "bonds": inv[spec["bonds"]],
        "bond_type": spec["bond_type"],
        "global33": spec["global33"],
    }
    p1 = _pred(model, D.batch_from_specs([spec2], device))
    out["atom_renumbering_max_delta"] = float(np.abs(p0 - p1).max())

    flip = {"atom_type": spec["atom_type"], "bonds": spec["bonds"][:, ::-1],
            "bond_type": spec["bond_type"], "global33": spec["global33"]}
    p2 = _pred(model, D.batch_from_specs([flip], device))
    out["bond_direction_flip_max_delta"] = float(np.abs(p0 - p2).max())

    specs = [_cache_mol("valid", i) for i in (0, 1, 2, 5, 7)]
    pb = D.batch_from_specs(specs, device)
    p_batch = _pred(model, pb)
    p_single = np.concatenate([_pred(model, D.batch_from_specs([s], device)) for s in specs])
    out["batch_vs_single_max_delta"] = float(np.abs(p_batch - p_single).max())
    shuffled = [specs[i] for i in (4, 0, 3, 1, 2)]
    p_shuf = _pred(model, D.batch_from_specs(shuffled, device))
    out["batch_reorder_max_delta"] = float(np.abs(p_shuf - p_batch[[4, 0, 3, 1, 2]]).max())

    out["pass"] = max(out["atom_renumbering_max_delta"],
                      out["bond_direction_flip_max_delta"],
                      out["batch_vs_single_max_delta"],
                      out["batch_reorder_max_delta"]) < 1e-5
    return out


def suite_local_exactness() -> dict:
    data = D.load_cache("valid")
    ap, bp = data["atom_ptr"], data["bond_ptr"]
    mismatches = 0
    compared = 0
    edge_total = 0
    for mol in range(40):
        spec = _cache_mol("valid", mol)
        raw = _pyg_raw("valid", mol)
        if not np.array_equal(spec["atom_type"], raw["atom_type"]):
            raise RuntimeError(f"molecule {mol}: atom types differ from raw PyG")
        if not np.array_equal(spec["bonds"], raw["bonds"]):
            raise RuntimeError(f"molecule {mol}: bond table differs from raw PyG")
        if not np.array_equal(spec["bond_type"], raw["bond_type"]):
            raise RuntimeError(f"molecule {mol}: bond types differ from raw PyG")
        brute = _brute_objects(spec)
        b0, b1 = int(bp[mol]), int(bp[mol + 1])
        cached = {
            "loc_type": data["loc_type"][b0:b1],
            "loc_role": data["loc_role"][b0:b1],
            "loc_et": data["loc_et"][b0:b1],
            "loc_nnodes": data["loc_nnodes"][b0:b1],
            "loc_nedges": data["loc_nedges"][b0:b1],
            "loc_mask": data["loc_mask"][b0:b1],
        }
        for k, ref in enumerate(brute):
            m = len(ref["nodes"])
            compared += 1
            ok = True
            ok &= int(cached["loc_nnodes"][k]) == m
            ok &= int(cached["loc_nedges"][k]) == ref["n_edges"]
            ok &= sorted(cached["loc_type"][k][:m].tolist()) == sorted(ref["types"])
            role_ids = [min(a, 3) * 4 + min(b, 3) for a, b in ref["roles"]]
            ok &= cached["loc_role"][k][:m].tolist() == role_ids
            ok &= bool(np.asarray(cached["loc_mask"][k][:m]).all())
            ok &= not bool(np.asarray(cached["loc_mask"][k][m:]).any())
            pos = 0
            for a in range(m):
                for b in range(a + 1, m):
                    ok &= int(cached["loc_et"][k][a, b]) == ref["et"][pos]
                    ok &= int(cached["loc_et"][k][b, a]) == ref["et"][pos]
                    pos += 1
            edge_total += len(ref["et"])
            mismatches += 0 if ok else 1
    return {"molecules": 40, "bond_objects_compared": compared,
            "pair_entries_compared": edge_total, "mismatches": mismatches,
            "pass": mismatches == 0}


def suite_synthetic(model, device) -> dict:
    cases = {
        "empty_atom": ([6], [], []),
        "single_bond": ([6, 6], [(0, 1)], [1]),
        "propane_tree": ([6, 6, 6], [(0, 1), (1, 2)], [1, 1]),
        "cyclopropane": ([6, 6, 6], [(0, 1), (1, 2), (0, 2)], [1, 1, 1]),
        "K4": ([6] * 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [1] * 6),
        "benzene": ([6] * 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)], [1] * 6),
    }
    out = {"cases": {}}
    ok = True
    for name, (at, bonds, bt) in cases.items():
        spec = {"atom_type": np.asarray(at), "bonds": np.asarray(bonds).reshape(-1, 2),
                "bond_type": np.asarray(bt)}
        batch = D.batch_from_specs([spec], device)
        with torch.no_grad():
            real, info = model(batch, return_info=True)
        entry = {
            "pred_finite": bool(torch.isfinite(real).all()),
            "n_tokens": int(batch["bond_mask"].sum()),
            "loc_nnodes": batch["loc_nnodes"][0, : int(batch["bond_mask"].sum())].tolist(),
            "loc_nedges": batch["loc_nedges"][0, : int(batch["bond_mask"].sum())].tolist(),
            "s1_norm": float(info["s1"].norm()),
            "s2_norm": float(info["s2"].norm()),
        }
        entry["pass"] = entry["pred_finite"]
        if name == "K4":
            entry["pass"] &= all(v == 4 for v in entry["loc_nnodes"])
            entry["pass"] &= all(v == 6 for v in entry["loc_nedges"])
        if name == "benzene":
            # P_b = {u,v} U N(u) U N(v) is a 4-node induced path with 3 bonds
            entry["pass"] &= all(v == 4 for v in entry["loc_nnodes"])
            entry["pass"] &= all(v == 3 for v in entry["loc_nedges"])
        if name == "cyclopropane":
            entry["pass"] &= all(v == 3 for v in entry["loc_nnodes"])
            entry["pass"] &= all(v == 3 for v in entry["loc_nedges"])
        if name == "propane_tree":
            entry["pass"] &= all(v == 3 for v in entry["loc_nnodes"])
            entry["pass"] &= all(v == 2 for v in entry["loc_nedges"])
        if name == "single_bond":
            entry["pass"] &= entry["loc_nnodes"] == [2] and entry["loc_nedges"] == [1]
        if name == "empty_atom":
            entry["pass"] &= entry["n_tokens"] == 0
        out["cases"][name] = entry
        ok &= entry["pass"]
    out["pass"] = ok
    return out


def suite_real_null(scales) -> dict:
    m_real = M.build_model(0, "real", scales)
    m_null = M.build_model(0, "null", scales)
    s_real, s_null = m_real.state_dict(), m_null.state_dict()
    same_keys = sorted(s_real) == sorted(s_null)
    tensor_max = 0.0
    for k in s_real:
        a = s_real[k]
        b = s_null[k]
        if a.shape != b.shape:
            same_keys = False
            continue
        if a.is_floating_point():
            tensor_max = max(tensor_max, float((a - b).abs().max()))
        else:
            tensor_max = max(tensor_max, float((a != b).any()))
    opt_real = T.make_optimizer(m_real, 2e-3, 1e-5)
    opt_null = T.make_optimizer(m_null, 2e-3, 1e-5)
    order_same = all(np.array_equal(T.batch_order(0, e, 10000), T.batch_order(0, e, 10000))
                     for e in (1, 2, 3))
    sched = T.schedule_fn(780000, 15600, 2e-3, 2e-5)
    out = {
        "params_real": M.param_audit(m_real)["param_total"],
        "params_null": M.param_audit(m_null)["param_total"],
        "state_keys_identical": bool(same_keys),
        "state_tensor_max_abs_diff": float(tensor_max),
        "optimizer_defaults_identical": opt_real.defaults == opt_null.defaults,
        "optimizer_param_groups_identical": [
            (len(g["params"]), g["lr"], g["weight_decay"]) for g in opt_real.param_groups
        ] == [(len(g["params"]), g["lr"], g["weight_decay"]) for g in opt_null.param_groups],
        "data_order_identical": bool(order_same),
        "lr_at_0_15600_779999": [sched(0), sched(15600), sched(779999)],
        "lr_monotone_after_warmup": all(sched(s) >= sched(s + 100) for s in range(15600, 779900, 500)),
    }
    out["pass"] = (out["params_real"] == out["params_null"]
                   and out["state_keys_identical"]
                   and out["state_tensor_max_abs_diff"] == 0.0
                   and out["optimizer_defaults_identical"]
                   and out["data_order_identical"]
                   and out["lr_monotone_after_warmup"])
    return out


def suite_correspondence(scales, device) -> dict:
    data = D.load_cache("valid")
    spec = _cache_mol("valid", 0)
    batch = D.batch_from_specs([spec], device)
    out = {}
    results = {}
    for arm in ("real", "null"):
        model = M.build_model(0, arm, scales).to(device)
        model.eval()
        with torch.no_grad():
            h0 = model.local(batch["loc_type"], batch["loc_role"], batch["loc_et"],
                             batch["loc_mask"], batch["loc_nnodes"], batch["loc_nedges"],
                             batch["bond_mask"], arm)
        # class-marginal-preserving permutation of the pair-edge assignment
        et2 = _permute_pair_edges(batch["loc_et"], batch["loc_mask"], seed=5)
        changed = float((et2 != batch["loc_et"]).float().mean())
        batch2 = dict(batch)
        batch2["loc_et"] = et2
        with torch.no_grad():
            h0b = model.local(batch2["loc_type"], batch2["loc_role"], et2,
                              batch2["loc_mask"], batch2["loc_nnodes"], batch2["loc_nedges"],
                              batch2["bond_mask"], arm)
        # single-pair intervention: flip one non-edge pair inside the first object
        et3 = batch["loc_et"].clone()
        tri = torch.triu(torch.ones(et3.shape[-1], et3.shape[-1], dtype=torch.bool, device=device), 1)
        nm = batch["loc_mask"][0, 0]
        cand = (tri & nm[:, None] & nm[None, :] & (et3[0, 0] == 0)).nonzero()
        i, j = cand[0].tolist()
        et3[0, 0, i, j] = 1
        et3[0, 0, j, i] = 1
        batch3 = dict(batch)
        batch3["loc_et"] = et3
        with torch.no_grad():
            h0c = model.local(batch3["loc_type"], batch3["loc_role"], et3,
                              batch3["loc_mask"], batch3["loc_nnodes"], batch3["loc_nedges"],
                              batch3["bond_mask"], arm)
        results[arm] = {
            "token_change_under_pair_permutation": float((h0 - h0b).abs().max()),
            "token_change_under_single_pair_flip": float((h0 - h0c).abs().max()),
            "permuted_pair_fraction": changed,
        }
        # gradient path
        model.train()
        pred = model(batch)
        loss = (pred - torch.zeros_like(pred)).abs().mean()
        loss.backward()
        results[arm]["phi_p_grad_abs_sum"] = float(model.local.phi_p[0].weight.grad.abs().sum())
        results[arm]["phi_n_grad_abs_sum"] = float(model.local.phi_n[0].weight.grad.abs().sum())
        results[arm]["readout_grad_abs_sum"] = float(model.readout[0].weight.grad.abs().sum())
    # class marginals are identical between the original and the permuted input
    tri = torch.triu(torch.ones(batch["loc_et"].shape[-1], batch["loc_et"].shape[-1],
                                dtype=torch.bool, device=device), 1)
    pmask = batch["loc_mask"][:, :, :, None] & batch["loc_mask"][:, :, None, :] & tri
    marg_a = (torch.nn.functional.one_hot(batch["loc_et"], 4).double()
              * pmask[..., None]).sum((2, 3))
    et2 = _permute_pair_edges(batch["loc_et"], batch["loc_mask"], seed=5)
    marg_b = (torch.nn.functional.one_hot(et2, 4).double() * pmask[..., None]).sum((2, 3))
    out["class_marginals_identical"] = bool(torch.equal(marg_a, marg_b))
    out["real"] = results["real"]
    out["null"] = results["null"]
    out["null_is_invariant_to_correspondence"] = results["null"]["token_change_under_pair_permutation"] == 0.0
    out["real_reads_correspondence"] = (results["real"]["token_change_under_pair_permutation"] > 1e-6
                                        and results["real"]["token_change_under_single_pair_flip"] > 1e-6)
    out["gradients_nonzero"] = all(results[a]["phi_p_grad_abs_sum"] > 0 for a in results)
    out["pass"] = (out["null_is_invariant_to_correspondence"]
                   and out["real_reads_correspondence"]
                   and out["class_marginals_identical"]
                   and out["gradients_nonzero"])
    return out


def suite_witness(scales, device) -> dict:
    """Two synthetic local objects: same roles/types/marginals, different pairing."""
    L = 5
    roles = [1, 4, 6, 9, 6]                     # (du,dv) pairs: (0,1),(1,0),(1,2),(2,1),(1,2)
    edges_a = [(0, 1), (0, 2), (3, 4)]          # 3 single bonds
    edges_b = [(0, 1), (0, 3), (2, 4)]          # same count, different correspondence
    # shapes are [B=1, E=1, L] / [B=1, E=1, L, L]: one object, one token
    et = {}
    for name, edges in (("a", edges_a), ("b", edges_b)):
        arr = np.zeros((1, 1, L, L), dtype=np.int64)
        for i, j in edges:
            arr[0, 0, i, j] = 1
            arr[0, 0, j, i] = 1
        et[name] = arr
    types = np.zeros((1, 1, L), dtype=np.int64)
    mask = np.ones((1, 1, L), dtype=bool)
    nnodes = np.array([[L]], dtype=np.float32)
    out = {}
    for arm in ("real", "null"):
        model = M.build_model(0, arm, scales).to(device).eval()
        hs = {}
        for name in ("a", "b"):
            et_t = torch.as_tensor(et[name], device=device)
            t_t = torch.as_tensor(types, device=device)
            r_t = torch.as_tensor(np.asarray([[roles]]), device=device)
            m_t = torch.as_tensor(mask, device=device)
            nb = torch.as_tensor(nnodes, device=device)
            with torch.no_grad():
                h = model.local(t_t, r_t, et_t, m_t, nb,
                                torch.full_like(nb, 3.0),
                                torch.ones(1, 1, dtype=torch.bool, device=device), arm)
            hs[name] = h
        delta = float((hs["a"] - hs["b"]).abs().max())
        marg = {}
        for name in ("a", "b"):
            onehot = torch.nn.functional.one_hot(torch.as_tensor(et[name], device=device), 4).double()
            tri = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)
            marg[name] = float((onehot * tri[None, None, :, :, None]).sum())
        out[arm] = {"token_delta_a_vs_b": delta,
                    "edge_class_count": marg["a"],
                    "same_marginal": marg["a"] == marg["b"]}
    out["real_distinguishes"] = out["real"]["token_delta_a_vs_b"] > 1e-6
    out["null_cannot"] = out["null"]["token_delta_a_vs_b"] == 0.0
    out["pass"] = (out["real_distinguishes"] and out["null_cannot"]
                   and out["real"]["same_marginal"])
    return out


def suite_numerics(scales, device) -> dict:
    torch.manual_seed(3)
    out = {}
    # --- sL2 attention vs float64 triple-loop reference
    blk = M.Block(d=8, heads=2, ffn_mult=2, attn_dropout=0.0).to(device).double()
    B, H, E, dh = 2, 2, 5, 4
    q = torch.randn(B, H, E, dh, dtype=torch.float64, device=device)
    k = torch.randn(B, H, E, dh, dtype=torch.float64, device=device)
    v = torch.randn(B, H, E, dh, dtype=torch.float64, device=device)
    bias = torch.randn(B, H, E, E, dtype=torch.float64, device=device) * 0.1
    mask = torch.ones(B, E, dtype=torch.bool, device=device)
    mask[1, 3:] = False
    ref = _ref_sl2(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                   bias.cpu().numpy(), mask.cpu().numpy())
    ctx, _ = M.sl2_context(q, k, v, bias, mask)
    got = ctx.detach().cpu().numpy()
    keep = np.broadcast_to(mask.cpu().numpy()[:, None, :, None], got.shape)
    out["sl2_max_abs_diff"] = float(np.abs(got - ref)[keep].max())
    out["sl2_rows_compared"] = int(keep.sum()) * B * H

    # --- AdaRMSNorm vs float64 reference
    norm = M.AdaRMSNorm(6, rank=3).to(device).double()
    with torch.no_grad():
        norm.scale.copy_(torch.randn(6, dtype=torch.float64, device=device) * 0.1)
        norm.shift.copy_(torch.randn(6, dtype=torch.float64, device=device) * 0.1)
        norm.proj_s.weight.copy_(torch.randn(6, 3, dtype=torch.float64, device=device) * 0.1)
        norm.proj_b.weight.copy_(torch.randn(6, 3, dtype=torch.float64, device=device) * 0.1)
    x = torch.randn(2, 4, 6, dtype=torch.float64, device=device)
    g = torch.randn(2, 3, dtype=torch.float64, device=device)
    got = norm(x, g).detach().cpu().numpy()
    exp = _ref_adarmsnorm(x.detach().cpu().numpy(), norm.scale.detach().cpu().numpy(),
                          norm.shift.detach().cpu().numpy(),
                          norm.proj_s.weight.detach().cpu().numpy(),
                          norm.proj_b.weight.detach().cpu().numpy(),
                          g.detach().cpu().numpy(), norm.eps)
    out["adaramsnorm_max_abs_diff"] = float(np.abs(got - exp).max())
    # zero-init => pure RMSNorm
    fresh = M.AdaRMSNorm(6, rank=3).to(device).double()
    xr = fresh(x, g).detach().cpu().numpy()
    xn = x.detach().cpu().numpy()
    rms = xn / np.sqrt((xn ** 2).mean(-1, keepdims=True) + fresh.eps)
    out["adaramsnorm_identity_at_init_max_abs_diff"] = float(np.abs(xr - rms).max())

    # --- RRWP: model path vs independent float64 enumeration and vs rc_data
    spec = _cache_mol("valid", 1)
    batch = D.batch_from_specs([spec], device)
    R_model = M.rrwp(batch["bonds"], batch["bond_mask"], batch["n_atoms"]).cpu().numpy()[0]
    R_ref = M.rrwp_reference(spec["bonds"], len(spec["atom_type"]), ksteps=M.KSTEPS)
    out["rrwp_vs_float64_enumeration_max_abs_diff"] = float(np.abs(R_model - R_ref).max())
    try:
        import rc_data as RC
        R_rc = RC.bond_relation(np.asarray(spec["bonds"]), len(spec["atom_type"]),
                                ksteps=M.KSTEPS)
        out["rrwp_vs_relation_probe_max_abs_diff"] = float(np.abs(R_model - R_rc).max())
        out["relation_probe_module"] = str(Path(RC.__file__).relative_to(_REPO))
    except Exception as exc:                      # pragma: no cover - diagnostics only
        out["rrwp_vs_relation_probe_max_abs_diff"] = None
        out["relation_probe_error"] = repr(exc)
    # structural properties: R[0] diagonal 1, shared endpoint 0.5, disjoint 0
    diag = np.diag(R_model[:, :, 0])
    out["r0_diagonal_min"] = float(diag.min())
    out["r0_diagonal_max"] = float(diag.max())
    bonds = spec["bonds"]
    share = [(i, j) for i in range(len(bonds)) for j in range(len(bonds))
             if i != j and set(bonds[i]) & set(bonds[j])]
    disjoint = [(i, j) for i in range(len(bonds)) for j in range(len(bonds))
                if i != j and not (set(bonds[i]) & set(bonds[j]))]
    out["r0_shared_endpoint_values"] = sorted({round(float(R_model[i, j, 0]), 6)
                                               for i, j in share})[:5]
    out["r0_disjoint_max"] = float(max((R_model[i, j, 0] for i, j in disjoint), default=0.0))
    out["rrwp_zero_on_padding"] = bool((R_model.sum(axis=(0, 1)) >= 0).all())

    out["pass"] = (out["sl2_max_abs_diff"] < 1e-12
                   and out["adaramsnorm_max_abs_diff"] < 1e-12
                   and out["adaramsnorm_identity_at_init_max_abs_diff"] < 1e-12
                   and out["rrwp_vs_float64_enumeration_max_abs_diff"] < 1e-6
                   and (out["rrwp_vs_relation_probe_max_abs_diff"] is None
                        or out["rrwp_vs_relation_probe_max_abs_diff"] < 1e-6)
                   and abs(out["r0_diagonal_min"] - 1.0) < 1e-6
                   and abs(out["r0_diagonal_max"] - 1.0) < 1e-6
                   and out["r0_shared_endpoint_values"] == [0.5]
                   and out["r0_disjoint_max"] == 0.0)
    return out


def suite_masking(model, device) -> dict:
    out = {}
    specs = [_cache_mol("valid", i) for i in (0, 4, 9)]
    batch = D.batch_from_specs(specs, device)
    p = _pred(model, batch)
    # extra padding slots must not change anything
    pad_e = 6
    big = dict(batch)
    for key in ("loc_type", "loc_role"):
        pad = torch.full((batch[key].shape[0], pad_e, batch[key].shape[2]), -1,
                         dtype=batch[key].dtype, device=device)
        big[key] = torch.cat([batch[key], pad], dim=1)
    for key in ("loc_et",):
        pad = torch.zeros(batch[key].shape[0], pad_e, batch[key].shape[2],
                          batch[key].shape[3], dtype=batch[key].dtype, device=device)
        big[key] = torch.cat([batch[key], pad], dim=1)
    for key in ("loc_mask", "bond_mask"):
        pad = torch.zeros(batch[key].shape[0], pad_e, *batch[key].shape[2:],
                          dtype=torch.bool, device=device)
        big[key] = torch.cat([batch[key], pad], dim=1)
    for key in ("loc_nnodes", "loc_nedges"):
        pad = torch.zeros(batch[key].shape[0], pad_e, dtype=batch[key].dtype, device=device)
        big[key] = torch.cat([batch[key], pad], dim=1)
    big["bonds"] = torch.cat([batch["bonds"],
                              torch.zeros(batch["bonds"].shape[0], pad_e, 2,
                                          dtype=torch.long, device=device)], dim=1)
    p_big = _pred(model, big)
    out["extra_padding_max_delta"] = float(np.abs(p - p_big).max())

    # cross-molecule isolation: perturb molecule 1 only
    et2 = batch["loc_et"].clone()
    et2[1:2] = _permute_pair_edges(batch["loc_et"][1:2], batch["loc_mask"][1:2], seed=9)
    b2 = dict(batch)
    b2["loc_et"] = et2
    p2 = _pred(model, b2)
    out["molecule1_delta"] = float(abs(p2[1] - p[1]))
    out["other_molecules_max_delta"] = float(max(abs(p2[0] - p[0]), abs(p2[2] - p[2])))

    # finiteness of the whole forward in train mode
    model.train()
    pr = model(batch)
    out["train_forward_finite"] = bool(torch.isfinite(pr).all())
    model.eval()
    with torch.no_grad():
        pr, info = model(batch, return_info=True)
    out["eval_forward_finite"] = bool(torch.isfinite(pr).all()
                                      and torch.isfinite(info["h"]).all()
                                      and torch.isfinite(info["bias"]).all())
    # amplitude preservation: s1 is linear in h, s2 quadratic (no per-graph norm)
    with torch.no_grad():
        h = info["h"] * 3.0
        s1 = (h * batch["bond_mask"][..., None]).sum(1) / model.ref_bonds
        s2 = (h * batch["bond_mask"][..., None]).pow(2).sum(1) / model.ref_bonds
    out["s1_linear_ratio_err"] = float((s1 / (info["s1"] * 3.0) - 1).abs().max())
    out["s2_quadratic_ratio_err"] = float((s2 / (info["s2"] * 9.0) - 1).abs().max())
    out["pass"] = (out["extra_padding_max_delta"] == 0.0
                   and out["other_molecules_max_delta"] == 0.0
                   and out["molecule1_delta"] > 1e-6
                   and out["train_forward_finite"] and out["eval_forward_finite"]
                   and out["s1_linear_ratio_err"] < 1e-5
                   and out["s2_quadratic_ratio_err"] < 1e-5)
    return out


def suite_budget(scales) -> dict:
    model = M.build_model(0, "real", scales)
    audit = M.param_audit(model)
    opt = T.make_optimizer(model, 2e-3, 1e-5)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    no_bn = not any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
                    for m in model.modules())
    out = {
        "param_total": audit["param_total"],
        "param_trainable": audit["param_trainable"],
        "under_cap": audit["param_total"] <= 500_000,
        "per_module": audit["per_module"],
        "n_blocks": audit["n_blocks"],
        "per_block": audit["per_block"],
        "optimizer_covers_all_trainable": opt_ids == model_ids,
        "n_optimizer_tensors": len(opt_ids),
        "n_trainable_tensors": len(model_ids),
        "no_batchnorm": no_bn,
    }
    out["pass"] = (out["under_cap"] and out["optimizer_covers_all_trainable"]
                   and out["no_batchnorm"])
    return out


def suite_provenance() -> dict:
    out = {}
    try:
        D.build_split("test")
        out["test_split_refused"] = False
    except ValueError:
        out["test_split_refused"] = True
    spec = _cache_mol("valid", 0)
    orig = D.AD.official_labels
    D.AD.official_labels = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("label access on the static-object path"))
    try:
        D.build_molecule_objects(spec["atom_type"], spec["bonds"], spec["bond_type"],
                                 len(spec["atom_type"]))
        out["static_objects_are_label_free"] = True
    except RuntimeError:
        out["static_objects_are_label_free"] = False
    finally:
        D.AD.official_labels = orig
    meta_train = D.load_cache("train")["meta"]
    meta_valid = D.load_cache("valid")["meta"]
    out["cache_meta_flags"] = {
        "train": {k: meta_train[k] for k in ("labels_read_by_builder", "official_test_loaded")},
        "valid": {k: meta_valid[k] for k in ("labels_read_by_builder", "official_test_loaded")},
    }
    src = (Path(__file__).parent / "esoc_train.py").read_text(encoding="utf-8")
    out["trainer_only_uses_train_and_valid"] = ("load_cache(\"train\")" in src
                                                and "load_cache(\"valid\")" in src
                                                and "load_cache(\"test\")" not in src)
    out["pass"] = (out["test_split_refused"] and out["static_objects_are_label_free"]
                   and out["trainer_only_uses_train_and_valid"]
                   and all(v["labels_read_by_builder"] is False
                           and v["official_test_loaded"] is False
                           for v in out["cache_meta_flags"].values()))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def gate_overfit(device: str, molecules: int = 64, epochs: int = 200) -> dict:
    """Small-sample fit gate: a model that cannot fit 64 molecules has a bug."""
    res = T.train("real", 0, epochs, device, mode="overfit", max_hours=0.5,
                  subset=molecules, out_dir=(EXP / "runs" / "gate_overfit"))
    curve = np.genfromtxt(EXP / "runs" / "gate_overfit" / "curve.csv",
                          delimiter=",", names=True)
    first = float(curve["train_mae_online"][0])
    last = float(curve["train_mae_online"][-1])
    best_eval = float(res["checkpoints"]["best"]["train_mae_eval_mode"])
    valid = float(res["checkpoints"]["best"]["valid_mae"])
    out = {
        "molecules": molecules,
        "epochs": int(res["curve_note"]["epochs_completed"]),
        "steps_per_epoch": int(res["curve_note"]["steps_per_epoch"]),
        "online_train_first": first,
        "online_train_last": last,
        "eval_mode_train_at_best": best_eval,
        "valid_at_best": valid,
        "wall_seconds": float(res["wall_seconds"]),
        "thresholds": {"eval_mode_train_at_best < 0.30": True,
                       "online_train_last < 0.5 * online_train_first": True},
    }
    out["pass"] = (best_eval < 0.30) and (last < 0.5 * first)
    return out


def run_all(device: str = "cpu") -> dict:
    scales = _scales()
    model = M.build_model(0, "real", scales).to(device).eval()
    torch.manual_seed(0)
    results = {}
    results["role_vocab"] = {"role_cap": D.ROLE_CAP, "model_role_base": M.ROLE_BASE,
                             "consistent": _role_base_consistent(), "pass": _role_base_consistent()}
    results["invariance"] = suite_invariance(model, device)
    results["local_exactness"] = suite_local_exactness()
    results["synthetic"] = suite_synthetic(model, device)
    results["real_null"] = suite_real_null(scales)
    results["correspondence"] = suite_correspondence(scales, device)
    results["witness"] = suite_witness(scales, device)
    results["numerics"] = suite_numerics(scales, device)
    results["masking"] = suite_masking(model, device)
    results["budget"] = suite_budget(scales)
    results["provenance"] = suite_provenance()
    passed = all(v["pass"] for v in results.values())
    payload = {
        "device": device,
        "all_pass": bool(passed),
        "n_suites": len(results),
        "n_pass": sum(1 for v in results.values() if v["pass"]),
        "suites": results,
        "environment": C.environment(),
        "created": C.utcnow(),
    }
    return payload


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gpu-gates", action="store_true",
                   help="also run the small-sample overfit gate (needs a GPU)")
    p.add_argument("--molecules", type=int, default=64)
    p.add_argument("--overfit-epochs", type=int, default=200)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    payload = run_all(args.device)
    if args.gpu_gates:
        payload["suites"]["overfit_small_sample"] = gate_overfit(
            args.device, args.molecules, args.overfit_epochs)
        payload["all_pass"] = all(v["pass"] for v in payload["suites"].values())
        payload["n_pass"] = sum(1 for v in payload["suites"].values() if v["pass"])
        payload["n_suites"] = len(payload["suites"])
    out = Path(args.out) if args.out else (ANALYSIS / "raw" / "cpu_checks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(out, payload)
    print(json.dumps({k: v["pass"] for k, v in payload["suites"].items()}, indent=2))
    print("ALL PASS" if payload["all_pass"] else "FAILED")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
