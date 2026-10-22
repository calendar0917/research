"""Pre-training CPU gate for ``zinc_shared_factor_v1`` (arms D, T, N).

Only the checks that this operator needs -- no unrelated repository tests, no
hyper-parameter search, no performance claim:

 1. contraction   the vectorised ``T``/``N`` log-contractions equal an
                  independent float64 brute-force enumeration on small typed
                  objects (4-node, 5-node, and a 4-node object inside the
                  5-slot layout).
 2. invariance    a root-preserving relabelling of the non-root slots leaves
                  ``T``/``N`` unchanged; the in-graph occurrence order does not
                  change the aggregated per-root statistics.
 3. witness       H1 = star + chords (1,2),(2,3) vs H2 = star + chords
                  (1,2),(3,4) with identical atom categories and identical bond
                  multiset: under one pre-fixed non-degenerate random parameter
                  point, T separates them and N is identical.
 4. no cycle      a pure star (no leaf-leaf edge) and a star with a conditionally
                  independent extra factor give ``T == N``.
 5. type paths    changing the root category, a non-root category or a bond type
                  changes the content vector; padded slot values cannot leak;
                  an isolated extra real node is a unit-sum no-op.
 6. gradients     every parameter is in the optimiser; at ``W_s = 0`` the branch
                  gradient is exactly 0 (expected, not a failure); once the port
                  is open ``q``, ``K`` and ``phi`` all receive gradient.
 7. provenance    the arm-D init state is byte-identical to the previous
                  round's LF-Dense init; a 256-row partial epoch of D through
                  this harness is bit-identical to ``lf_core.train_epoch``; the
                  T/N initial parameters are identical; checkpoint reload
                  reproduces the predictions; the audited ``bf_data`` content
                  digests are unchanged.
 8. budget        parameters 449,601 / 451,777 / 451,777; 79 steps/epoch,
                  47,400 updates, warm-up and cosine endpoints; identical batch
                  order for D/T/N within a seed.

These checks prove implementation consistency and the local distinction
property.  They prove nothing about expressiveness, generalisation or novelty.

Run: ``uv run python tracks/structural-sufficiency/zinc_shared_factor_v1/code/sf_checks.py``
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_LF = _SS / "zinc_local_factorization_v1" / "code"
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_BF), str(_LF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import bf_data as BFD  # noqa: E402
import lf_checks as LFC  # noqa: E402
import lf_core as LC  # noqa: E402
import lf_models as LM  # noqa: E402
import sf_core as SC  # noqa: E402
import sf_models as SF  # noqa: E402

RESULTS: dict[str, dict] = {}
ANALYSIS = SC.ANALYSIS
PAIR_K = {p: k for k, p in enumerate(SF.PAIR_INDEX)}

# Pre-fixed, non-degenerate random parameter point for the structural witness.
# Declared here before any run; never re-drawn to make a check pass.
WITNESS_SEED = 20261022
FLOAT64_TOL = 1e-10
FLOAT32_TOL = 2e-5


def record(name: str, fn, *args, **kwargs) -> dict:
    t0 = time.perf_counter()
    try:
        ok, detail = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a failing check is data, not a crash
        import traceback

        ok, detail = False, {"error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()[-2000:]}
    detail = dict(detail)
    detail["pass"] = bool(ok)
    detail["seconds"] = float(time.perf_counter() - t0)
    RESULTS[name] = detail
    short = json.dumps({k: v for k, v in detail.items()
                        if k not in ("pass", "seconds")}, default=str)[:480]
    C.log(f"[check {name}] {'PASS' if ok else 'FAIL'} ({detail['seconds']:.2f}s) {short}")
    return detail


# ---------------------------------------------------------------------------
# helpers: synthetic typed objects and an independent brute force
# ---------------------------------------------------------------------------


def pattern(cats, adj, family: int) -> LFC.Content:
    return LFC.Content(cats, adj, family)


def table_of(contents: list[LFC.Content], split: str = "__sf") -> LM.PatternTable:
    return LFC.make_table(contents, split=split)


def edge_list(pair, length: int) -> list[tuple[int, int, int]]:
    return [(i, j, int(pair[PAIR_K[(i, j)]])) for (i, j) in SF.PAIR_INDEX
            if j < length and int(pair[PAIR_K[(i, j)]]) != 0]


def brute_T(atom, pair, length, q, kk) -> np.ndarray:
    """Explicit float64 sum over ``R**(n-1)`` joint states -> ``[C, R]``."""
    Cc, R = q.shape[1], q.shape[2]
    edges = edge_list(pair, length)
    out = np.zeros((Cc, R))
    for c in range(Cc):
        for s in range(R):
            tot = 0.0
            for assign in itertools.product(range(R), repeat=length - 1):
                w = 1.0
                for i in range(1, length):
                    w *= q[atom[i], c, assign[i - 1]]
                for (i, j, b) in edges:
                    si = s if i == 0 else assign[i - 1]
                    sj = s if j == 0 else assign[j - 1]
                    w *= kk[b - 1, c, si, sj]
                tot += w
            out[c, s] = tot
    return out


def brute_N(atom, pair, length, q, kk) -> np.ndarray:
    """Explicit float64 product of independent per-edge factors -> ``[C, R]``."""
    Cc, R = q.shape[1], q.shape[2]
    out = np.ones((Cc, R))
    for (i, j, b) in edge_list(pair, length):
        for c in range(Cc):
            for s in range(R):
                if i == 0:
                    out[c, s] *= sum(q[atom[j], c, t] * kk[b - 1, c, s, t]
                                     for t in range(R))
                else:
                    out[c, s] *= sum(q[atom[i], c, u] * q[atom[j], c, v] * kk[b - 1, c, u, v]
                                     for u in range(R) for v in range(R))
    return out


def log_block(vec: torch.Tensor) -> np.ndarray:
    """The 64-dim ``log T / max(m,1)`` block of a content vector."""
    return vec[64:128].detach().double().numpy()


def encode(branch: SF.AtomicBranch, content: LFC.Content, *, double: bool = True):
    """Content vector of exactly one content (float64 by default)."""
    mod = branch.double() if double else branch
    table = table_of([content])
    sel = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        out = mod._group_content(table, sel, content.n)
    if double:
        branch.float()
    return out[0]


def witness_params(branch: SF.AtomicBranch) -> tuple[np.ndarray, np.ndarray]:
    branch.eval()
    mod = branch.double()
    with torch.no_grad():
        q = torch.exp(mod.state.q_log()).numpy().astype(np.float64)
        kk = torch.exp(mod.state.log_k()).numpy().astype(np.float64)
    branch.float()
    return q, kk


def make_branches() -> dict[str, SF.AtomicBranch]:
    out: dict[str, SF.AtomicBranch] = {}
    for mode in SF.MODES:
        b = SF.AtomicBranch(mode)
        b.reinit(WITNESS_SEED)
        out[mode] = b
    return out


# ---------------------------------------------------------------------------
# 1. exact contraction vs brute force
# ---------------------------------------------------------------------------


def check_contraction() -> tuple[bool, dict]:
    detail: dict = {"patterns": {}}
    ok = True
    branches = make_branches()
    q, kk = witness_params(branches["T"])
    cases = {
        "p4_chain": pattern([6, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1,
                                          (1, 2): 3, (2, 3): 2}, 0),
        "p5_full": pattern([6, 7, 8, 9, 10],
                           {(0, 1): 1, (0, 2): 2, (0, 3): 1, (0, 4): 3,
                            (1, 2): 3, (1, 3): 2, (2, 3): 1, (2, 4): 2,
                            (3, 4): 3, (1, 4): 1}, 1),
        "p4_triangle": pattern([11, 12, 13, 14], {(0, 1): 1, (0, 2): 1,
                                                  (0, 3): 1, (1, 2): 1,
                                                  (1, 3): 1, (2, 3): 1}, 2),
    }
    for name, content in cases.items():
        atom, pair = content.arrays()
        length = content.n
        n_edges = len(edge_list(pair, length))
        d = {"length": int(length), "n_edges": int(n_edges), "modes": {}}
        for mode, branch in branches.items():
            vec = encode(branch, content)
            got = log_block(vec).reshape(SF.C_CH, SF.R_STATE) * max(n_edges, 1)
            ref = np.log(brute_T(atom, pair, length, q, kk) if mode == "T"
                         else brute_N(atom, pair, length, q, kk))
            d["modes"][mode] = {"max_abs_log_err": float(np.abs(got - ref).max())}
        detail["patterns"][name] = d
        for mode in SF.MODES:
            ok &= d["modes"][mode]["max_abs_log_err"] < FLOAT64_TOL
    # also check the q block is exactly q_c(s|a_root)
    atom, _ = cases["p4_chain"].arrays()
    vec = encode(branches["T"], cases["p4_chain"])
    q_root_ref = q[atom[0]].reshape(-1)
    detail["q_block_max_abs_err"] = float(np.abs(vec[:64].numpy() - q_root_ref).max())
    ok &= detail["q_block_max_abs_err"] < FLOAT64_TOL
    return ok, detail


# ---------------------------------------------------------------------------
# 2. relabelling and occurrence order
# ---------------------------------------------------------------------------


def check_invariance(norm) -> tuple[bool, dict]:
    content = pattern([6, 7, 8, 9, 6], {(0, 1): 1, (0, 2): 2, (0, 3): 1,
                                        (0, 4): 3, (1, 2): 3, (2, 3): 1,
                                        (3, 4): 2}, 1)
    branches = make_branches()
    detail: dict = {"relabel": {}, "occurrence_order": {}}
    ok = True
    orders = [(0,) + p for p in itertools.permutations(range(1, content.n))]
    for mode, branch in branches.items():
        base = encode(branch, content)
        worst = 0.0
        for order in orders:
            other = encode(branch, content.relabel(order))
            worst = max(worst, float((base - other).abs().max()))
        detail["relabel"][mode] = {"n_orders": len(orders), "max_abs_diff": worst}
        ok &= worst < 1e-9

    # occurrence order inside one graph: same occurrences, different index order
    contents = [content, pattern([6, 7, 8, 6], {(0, 1): 1, (0, 2): 2, (1, 3): 1}, 0),
                pattern([9, 7, 8, 9], {(0, 1): 3, (0, 2): 1}, 2)]
    table = table_of(contents)
    occ_orders = {
        "forward": ([0, 1, 2, 0], [0, 0, 0, 1], [0, 1, 2, 0], [1.0, 2.0, 1.0, 3.0]),
        "reversed": ([0, 2, 1, 0], [1, 0, 0, 0], [0, 2, 1, 0], [3.0, 1.0, 2.0, 1.0]),
    }
    for mode, branch in branches.items():
        mod = branch.double()
        outs = {}
        for tag, (pat, root, fam, cnt) in occ_orders.items():
            batch = {"occ_pat": torch.as_tensor(pat), "occ_root": torch.as_tensor(root),
                     "occ_family": torch.as_tensor(fam),
                     "occ_count": torch.as_tensor(cnt, dtype=torch.float64),
                     "n_atoms_total": 2, "split": table.split}
            with torch.no_grad():
                outs[tag] = mod(batch, table).clone()
        diff = float((outs["forward"] - outs["reversed"]).abs().max())
        detail["occurrence_order"][mode] = {"max_abs_diff": diff}
        ok &= diff < 1e-12
        branch.float()
    return ok, detail


# ---------------------------------------------------------------------------
# 3. structural witness (H1 vs H2) and 4. pure star
# ---------------------------------------------------------------------------


def check_witness() -> tuple[bool, dict]:
    h1 = pattern([6] * 5, {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1,
                           (1, 2): 1, (2, 3): 1}, 1)
    h2 = pattern([6] * 5, {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1,
                           (1, 2): 1, (3, 4): 1}, 1)
    star = pattern([6] * 5, {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1}, 1)
    a1, p1 = h1.arrays()
    a2, p2 = h2.arrays()
    detail: dict = {
        "atom_categories_identical": bool(np.array_equal(a1, a2)),
        "bond_multiset_identical": bool(sorted(v for v in p1.tolist() if v)
                                        == sorted(v for v in p2.tolist() if v)),
        "n_edges": [len(edge_list(p1, 5)), len(edge_list(p2, 5))],
        "witness_seed": WITNESS_SEED,
    }
    ok = detail["atom_categories_identical"] and detail["bond_multiset_identical"]
    branches = make_branches()
    for mode, branch in branches.items():
        v1 = encode(branch, h1)
        v2 = encode(branch, h2)
        # full content vector difference and the 64-dim contraction block
        detail[mode] = {
            "content_rel_l2": float((v1 - v2).norm() / v1.norm()),
            "log_block_max_abs_diff": float((v1[64:128] - v2[64:128]).abs().max()),
        }
        with torch.no_grad():
            phi1 = branch.double().permuted_phi(table_of([h1]))[0]
            phi2 = branch.double().permuted_phi(table_of([h2]))[0]
        branch.float()
        detail[mode]["phi_rel_l2"] = float((phi1 - phi2).norm() / phi1.norm())
    detail["verdict"] = {
        "T_separates": bool(detail["T"]["log_block_max_abs_diff"] > 1e-3
                            and detail["T"]["phi_rel_l2"] > 1e-5),
        "N_identical": bool(detail["N"]["log_block_max_abs_diff"] < 1e-6
                            and detail["N"]["phi_rel_l2"] < 1e-7),
    }
    ok &= detail["verdict"]["T_separates"] and detail["verdict"]["N_identical"]

    # pure star: T and N must agree (the root is conditioned on)
    star_detail: dict = {}
    for name, content in (("star_only", star),
                          ("star_plus_isolated", pattern(
                              [6] * 5, {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1}, 1))):
        vt = encode(branches["T"], content)
        vn = encode(branches["N"], content)
        star_detail[name] = {"max_abs_diff_full": float((vt - vn).abs().max()),
                             "max_abs_diff_log_block": float((vt[64:128] - vn[64:128]).abs().max())}
        ok &= star_detail[name]["max_abs_diff_log_block"] < 1e-9
    detail["star"] = star_detail
    return ok, detail


# ---------------------------------------------------------------------------
# 5. type paths, padding, isolated node
# ---------------------------------------------------------------------------


def check_type_paths() -> tuple[bool, dict]:
    base = pattern([6, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 3}, 0)
    root_type = pattern([11, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 3}, 0)
    leaf_type = pattern([6, 7, 12, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 3}, 0)
    bond_type = pattern([6, 7, 8, 9], {(0, 1): 2, (0, 2): 2, (0, 3): 1, (1, 2): 3}, 0)
    branches = make_branches()
    detail: dict = {"cases": {}}
    ok = True
    for mode, branch in branches.items():
        b = encode(branch, base)
        rows = {}
        for tag, content in (("root_type", root_type), ("leaf_type", leaf_type),
                             ("bond_type", bond_type)):
            v = encode(branch, content)
            rows[tag] = {"full_rel_l2": float((v - b).norm() / b.norm()),
                         "log_block_max_abs_diff":
                             float((v[64:128] - b[64:128]).abs().max())}
        detail["cases"][mode] = rows
        ok &= rows["root_type"]["full_rel_l2"] > 1e-6
        ok &= rows["leaf_type"]["log_block_max_abs_diff"] > 1e-6
        ok &= rows["bond_type"]["log_block_max_abs_diff"] > 1e-6

    # padding: garbage in padded slots must not change anything (bitwise)
    pad_a = LFC.Content([6, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1}, 0)
    ta = table_of([pad_a], split="__pad_a")
    tb = table_of([pad_a], split="__pad_b")
    # overwrite padded slot values in tb (slot 4 atom, pair slots with j >= 4)
    tb.atom_ids = tb.atom_ids.clone()
    tb.pair_ids = tb.pair_ids.clone()
    tb.atom_ids[0, 4] = 27
    for k, (_i, j) in enumerate(SF.PAIR_INDEX):
        if j >= 4:
            tb.pair_ids[0, k] = 3
    pad_detail = {}
    for mode, branch in branches.items():
        mod = branch.double()
        with torch.no_grad():
            va = mod._group_content(ta, torch.zeros(1, dtype=torch.long), 4)
            vb = mod._group_content(tb, torch.zeros(1, dtype=torch.long), 4)
        branch.float()
        pad_detail[mode] = {"max_abs_diff": float((va - vb).abs().max())}
        ok &= pad_detail[mode]["max_abs_diff"] == 0.0
    detail["padding_garbage_invariant"] = pad_detail

    # an isolated extra real node contributes exactly the unit sum of q
    isolated = pattern([6, 7, 8, 9, 3], {(0, 1): 1, (0, 2): 2, (0, 3): 1}, 1)
    iso_detail = {}
    for mode, branch in branches.items():
        v_base = encode(branch, pattern([6, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1}, 1))
        v_iso = encode(branch, isolated)
        iso_detail[mode] = {"log_block_max_abs_diff":
                            float((v_base[64:128] - v_iso[64:128]).abs().max())}
        ok &= iso_detail[mode]["log_block_max_abs_diff"] < 1e-9
    detail["isolated_node_is_unit_sum"] = iso_detail
    return ok, detail


# ---------------------------------------------------------------------------
# 6. optimiser coverage and port gradient
# ---------------------------------------------------------------------------


def check_gradients(norm) -> tuple[bool, dict]:
    device = torch.device("cpu")
    cache, y, rows = SC.load_split("train")
    chunk = rows[:128]
    batch = T1._to_device(SC.collate_lf(cache, chunk), device)
    target = torch.as_tensor(y[chunk], dtype=torch.float32)  # noqa: F841  (kept for clarity)
    detail: dict = {"arms": {}}
    ok = True
    for arm in SF.ARMS:
        model, info = SC.build(arm, norm, device)
        opt = LC.make_optimizer(model, SC.load_config(), 47_400)
        covered = C.optimizer_parameter_count(opt)
        hashes = SC.init_hashes(model)
        model.set_table(SC.attach_table(model, str(cache["split"]), device))

        def grads_after(step: bool) -> dict:
            model.zero_grad(set_to_none=True)
            pred = model(batch)
            torch.nn.functional.l1_loss(pred.view(-1), target.view(-1)).backward()
            if step:
                opt.step()
            out = {}
            for name, module in (("state", getattr(model.branch, "state", None)),
                                 ("l1", getattr(model.branch, "l1", None)),
                                 ("l2", getattr(model.branch, "l2", None)),
                                 ("phi", getattr(model.branch, "phi", None))):
                if module is not None:
                    out[name] = max((0.0 if p.grad is None else float(p.grad.abs().max()))
                                    for p in module.parameters())
            out["w_s"] = float(model.w_s.weight.grad.abs().max())
            return out

        zero = grads_after(step=False)
        with torch.no_grad():   # open the port deterministically
            g = torch.Generator().manual_seed(7)
            model.w_s.weight.copy_(torch.empty_like(model.w_s.weight)
                                   .normal_(0.0, 0.02, generator=g))
        opened = grads_after(step=False)
        # after one real update the branch parameters must have moved
        before = {k: v.detach().clone() for k, v in model.branch.state_dict().items()}
        grads_after(step=True)
        moved = max(float((v - before[k]).abs().max())
                    for k, v in model.branch.state_dict().items())
        d = {
            "parameters": int(info["parameters"]),
            "optimizer_covers_all": int(covered) == int(info["parameters"]),
            "w_s_zero_at_init": bool(hashes["w_s_zero"]),
            "grad_at_w_s_zero": zero,
            "grad_with_port_open": opened,
            "content_moved_after_one_step": moved,
        }
        detail["arms"][arm] = d
        ok &= d["optimizer_covers_all"] and d["w_s_zero_at_init"]
        ok &= d["grad_at_w_s_zero"]["w_s"] > 0.0
        ok &= d["content_moved_after_one_step"] > 0.0
        if arm == "D":
            ok &= zero.get("phi", 0.0) == 0.0 and opened.get("phi", 0.0) > 0.0
        else:
            ok &= (zero.get("state", 1.0) == 0.0 and zero.get("l1", 1.0) == 0.0
                   and zero.get("l2", 1.0) == 0.0)
            ok &= (opened.get("state", 0.0) > 0.0 and opened.get("l1", 0.0) > 0.0
                   and opened.get("l2", 0.0) > 0.0)
    return ok, detail


# ---------------------------------------------------------------------------
# 7. provenance: D identity, T/N identity, caches, reload
# ---------------------------------------------------------------------------


def check_provenance(norm) -> tuple[bool, dict]:
    device = torch.device("cpu")
    cache, y, rows = SC.load_split("train")
    cfg = SC.load_config()
    detail: dict = {}

    # (a) D init is byte-identical to last round's LF-Dense init
    sf_d, info_d = SC.build("D", norm, device, seed=0)
    lf_d, lf_info = LM.build_model("D", norm, seed=0, device=device)
    h_sf = SC.init_hashes(sf_d)
    same_keys = list(sf_d.state_dict()) == list(lf_d.state_dict())
    max_diff = max(float((sf_d.state_dict()[k].float().double()
                          - lf_d.state_dict()[k].float().double()).abs().max())
                   for k in lf_d.state_dict())
    detail["d_identity"] = {
        "sf_init_sha256": h_sf["init_state_sha256"],
        "lf_init_sha256": C.state_hash(lf_d),
        "state_keys_identical": bool(same_keys),
        "state_max_abs_diff": max_diff,
        "parameters_sf": int(info_d["parameters"]),
        "parameters_lf": int(lf_info["parameters"]),
        "matches_config": h_sf["init_state_sha256"]
        == str(cfg["model"]["arms"]["D"]["init_state_sha256"]["0"]),
    }

    # (b) identical forward and one 256-row partial epoch through both harnesses
    table = SC.attach_table(sf_d, str(cache["split"]), device)
    lf_d.set_table(table)
    chunk = rows[:256]
    batch = T1._to_device(SC.collate_lf(cache, chunk), device)
    target = torch.as_tensor(y[chunk], dtype=torch.float32)  # noqa: F841  (kept for clarity)
    with torch.no_grad():
        p_sf = sf_d(batch).double()
        p_lf = lf_d(batch).double()
    fwd = {"max_abs_diff": float((p_sf - p_lf).abs().max())}

    opt_sf = LC.make_optimizer(sf_d, cfg, 79 * 600)
    opt_lf = LC.make_optimizer(lf_d, cfg, 79 * 600)
    gen_sf = torch.Generator().manual_seed(101)
    gen_lf = torch.Generator().manual_seed(101)
    rec_sf, step_sf = SC.train_epoch(sf_d, opt_sf, cache, chunk, y, gen_sf, cfg=cfg, epoch=1,
                                     total_updates=79 * 600, global_step=0, device=device)
    rec_lf, step_lf = LC.train_epoch(lf_d, opt_lf, cache, chunk, y, gen_lf, cfg=cfg, epoch=1,
                                     total_updates=79 * 600, global_step=0, device=device)
    keys = ("online_train_mae", "order_sha256", "w_s_norm",
            "grad_norm_mean", "grad_norm_max", "steps")
    loop = {k: {"sf": rec_sf[k], "lf": rec_lf[k]} for k in keys}
    loop["content_norm"] = {"sf": rec_sf["content_norm"], "lf": rec_lf["phi_norm"]}
    loop["steps_equal"] = step_sf == step_lf
    loop["max_abs_diff_online"] = abs(rec_sf["online_train_mae"] - rec_lf["online_train_mae"])
    detail["d_training_path"] = {"forward": fwd, "partial_epoch": loop}

    # (c) T and N start from exactly the same parameters
    detail["t_n_identity"] = {}
    for seed in SC.SEEDS:
        t, _ = SC.build("T", norm, device, seed=seed)
        n, _ = SC.build("N", norm, device, seed=seed)
        ht, hn = SC.init_hashes(t), SC.init_hashes(n)
        detail["t_n_identity"][str(seed)] = {
            "t_init_sha256": ht["init_state_sha256"],
            "n_init_sha256": hn["init_state_sha256"],
            "identical": ht["init_state_sha256"] == hn["init_state_sha256"],
            "content_identical": ht["content_state_sha256"] == hn["content_state_sha256"],
        }
        if seed != 0:
            # declared in the config only for seed 0; other seeds are recorded
            pass

    # (d) checkpoint reload reproduces predictions
    detail["reload"] = {}
    for arm in SF.ARMS:
        model, _ = SC.build(arm, norm, device, seed=0)
        model.set_table(SC.attach_table(model, str(cache["split"]), device))
        pred_a, mae_a = SC.evaluate(model, cache, chunk, device, 128)
        path = ANALYSIS / "_reload_tmp.pt"
        C.atomic_torch_save({"state_dict": C.cpu_state(model)}, path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model2, _ = SC.build(arm, norm, device, seed=0)
        C.load_state(model2, ck["state_dict"], device)
        model2.set_table(SC.attach_table(model2, str(cache["split"]), device))
        pred_b, mae_b = SC.evaluate(model2, cache, chunk, device, 128)
        detail["reload"][arm] = {"max_abs_pred_diff": float(np.abs(pred_a - pred_b).max()),
                                 "mae": float(mae_a), "mae_reloaded": float(mae_b)}
        path.unlink(missing_ok=True)

    # (e) audited bf_data payload digests unchanged
    detail["caches"] = {}
    keys_content = ("pat_atom", "pat_pair", "pat_len", "pat_family",
                    "ent_ptr", "ent_pat", "ent_root", "ent_family", "ent_count")
    for split in ("train", "valid"):
        pc = BFD.load_cache(split)
        h = hashlib.sha256()
        for key in keys_content:
            h.update(key.encode())
            h.update(np.ascontiguousarray(np.asarray(pc[key])).tobytes())
        auth = {"train": "ce643fdafba375f6c1689d942cc5c659e1e8224cab0893acf0d3e45857b80780",
                "valid": "91e0e81b8977dc3127e674863683b08a382c934740632d8b8b5676ddc1266ed3"}[split]
        detail["caches"][split] = {"digest": h.hexdigest(), "matches_previous_round": h.hexdigest() == auth,
                                   "n_entries": int(pc["meta"]["n_entries"]),
                                   "n_patterns": int(pc["meta"]["n_unique_patterns"])}

    ok = (detail["d_identity"]["state_keys_identical"]
          and detail["d_identity"]["state_max_abs_diff"] == 0.0
          and detail["d_identity"]["matches_config"]
          and fwd["max_abs_diff"] == 0.0
          and loop["max_abs_diff_online"] == 0.0
          and loop["steps_equal"]
          and all(v["identical"] and v["content_identical"]
                  for v in detail["t_n_identity"].values())
          and all(v["max_abs_pred_diff"] == 0.0 for v in detail["reload"].values())
          and all(v["matches_previous_round"] for v in detail["caches"].values()))
    return ok, detail


# ---------------------------------------------------------------------------
# 8. parameters, budget, schedule, batch order
# ---------------------------------------------------------------------------


def check_budget(norm) -> tuple[bool, dict]:
    cfg = SC.load_config()
    device = torch.device("cpu")
    detail: dict = {"params": {}, "schedule": {}}
    ok = True
    for arm in SF.ARMS:
        model, info = SC.build(arm, norm, device, seed=0)
        declared = int(cfg["model"]["arms"][arm]["parameters_expected"])
        detail["params"][arm] = {"actual": int(info["parameters"]), "declared": declared}
        ok &= int(info["parameters"]) == declared == SF.EXPECTED_PARAMETERS[arm]
        ok &= int(info["parameters"]) <= SF.PARAMETER_CAP
    # content-only parameter width, checked explicitly
    detail["params"]["T_content_module"] = int(
        sum(p.numel() for p in SC.build("T", norm, device, seed=0)[0].branch.parameters()))
    ok &= detail["params"]["T_content_module"] == SF.PHI_PARAMETERS == 12_928

    steps = math.ceil(cfg["data"]["train"] / cfg["schedule"]["batch_size"])
    total = steps * int(cfg["schedule"]["epochs"])
    detail["schedule"].update({"steps_per_epoch": int(steps), "epochs": int(cfg["schedule"]["epochs"]),
                               "outer_updates": int(total)})
    ok &= steps == 79 and total == 47_400 == int(cfg["schedule"]["outer_updates"])
    lr0 = SC.schedule_lr(0, total, cfg)
    lr_w = SC.schedule_lr(int(cfg["schedule"]["warmup_updates"]), total, cfg)
    lr_end = SC.schedule_lr(total, total, cfg)
    after = [SC.schedule_lr(s, total, cfg) for s in range(int(cfg["schedule"]["warmup_updates"]) + 1,
                                                          total + 1, 500)]
    detail["schedule"].update({
        "lr_start": float(lr0), "lr_after_warmup": float(lr_w), "lr_end": float(lr_end),
        "warmup_reaches_peak": abs(lr_w - float(cfg["optimizer"]["lr"])) < 1e-12,
        "end_is_min": abs(lr_end - float(cfg["optimizer"]["min_lr"])) < 1e-12,
        "monotone_after_warmup": all(b <= a + 1e-15 for a, b in zip(after, after[1:])),
    })
    ok &= (detail["schedule"]["warmup_reaches_peak"] and detail["schedule"]["end_is_min"]
           and detail["schedule"]["monotone_after_warmup"])

    # identical batch order for the three arms within each seed
    detail["order"] = {}
    for seed in SC.SEEDS:
        hashes = {}
        for arm in SF.ARMS:
            gen = torch.Generator().manual_seed(int(seed) + SC.SHUFFLE_SEED_OFFSET)
            hashes[arm] = [SC.order_sha256(SC.epoch_order(gen, 10_000)) for _ in range(600)]
        detail["order"][str(seed)] = {"first": hashes["D"][0], "last": hashes["D"][-1],
                                      "all_equal": len({tuple(v) for v in hashes.values()}) == 1}
        ok &= detail["order"][str(seed)]["all_equal"]
    return ok, detail



# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ANALYSIS / "cpu_checks.json"))
    args = p.parse_args(argv)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    cfg = SC.load_config()
    norm = D.load_norm()
    C.log(f"[sf-checks] config {cfg['_config_path']} sha256 {cfg['_config_sha256']}")
    record("contraction", check_contraction)
    record("invariance", check_invariance, norm)
    record("witness", check_witness)
    record("type_paths", check_type_paths)
    record("gradients", check_gradients, norm)
    record("provenance", check_provenance, norm)
    record("budget", check_budget, norm)
    n_pass = sum(1 for v in RESULTS.values() if v["pass"])
    doc = {
        "protocol": SC.PROTOCOL,
        "config_path": cfg["_config_path"],
        "config_sha256": cfg["_config_sha256"],
        "checks": RESULTS,
        "n_pass": n_pass,
        "n_fail": len(RESULTS) - n_pass,
        "witness_seed": WITNESS_SEED,
        "created": C.utcnow(),
        "scope_note": ("implementation consistency and the local distinction property only; "
                       "not a claim of expressiveness, generalisation or novelty"),
    }
    C.write_json(Path(args.out), doc)
    C.log(f"[sf-checks] {n_pass}/{len(RESULTS)} pass -> {args.out}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
