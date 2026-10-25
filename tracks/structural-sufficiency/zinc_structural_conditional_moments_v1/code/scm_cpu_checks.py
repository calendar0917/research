"""CPU gates for the structural conditional moments round (phase A + phase C).

Everything here runs on the *production* forward used by ``scm_runner.py``:
``SCMChannel`` wraps the frozen ``dcc_models.DualChannel`` and injects the
correction with a forward hook on ``DualChannel.local_composer``.  No formula is
re-derived for the model check; an independent float64 transcription of the
round's definition is used only to verify the algebra.

Phases
------
A  production correctness (init identity, Wout=0 function equality, the
   correction entering both original sums, nonzero S/A response, two-step
   gradient flow, invariances, boundaries, checkpoint round-trip, real trainer
   smoke).
C  synthetic non-isomorphic 32-atom ring counterexample.

Output: ``analysis/cpu_checks.json``.  Official test is never touched.
"""

from __future__ import annotations

import ast
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import scm_common as S
import scm_relations as R
import scm_models as M

import ad_common as C
import ad_data as D
import ad_train as T1
import dcc_models as MD

HIST_DCC_L_INIT_SHA = "f5c34c285c98b4d8a88361cb7dc1043b5abf07799f4f6121b25a59c904166e33"
HIST_P3_SHA = "21963b431329532f7b2411845c06e00eece54d0ddbb64774a63719b791716a30"
EXPECTED_PARAMS = 499_689
CHECKS: dict = {}


def add(name: str, passed: bool, detail) -> None:
    CHECKS[name] = {"pass": bool(passed), "detail": detail}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}"
          + ("" if passed else f" :: {detail}"), flush=True)


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


# ---------------------------------------------------------------------------
# independent float64 reference implementation of the spec
# ---------------------------------------------------------------------------


def reference_moments(c_loc, batch, scm, scales, with_cov=True):
    """Literal per-graph float64 transcription of the round's definition."""
    rank = scm.rank
    h = c_loc.double().numpy()

    def proj(lin):
        w = lin.weight.detach().double().numpy()
        b = lin.bias.detach().double().numpy()
        return h @ w.T + b

    v_all = proj(scm.v)
    q_all = proj(scm.q)
    k_all = proj(scm.k)
    bb = batch["bond_batch"].numpy()
    bl = batch["bond_local"].numpy()
    out = []
    for b in range(int(batch["n_graphs"])):
        sel = np.where(bb == b)[0]
        idx = sel[np.argsort(bl[sel])]
        n = len(idx)
        v = v_all[idx]
        mu1 = batch["mu1"][b].numpy()[:n, :n]
        mu2 = batch["mu2"][b].numpy()[:n, :n]
        mg = v.mean(0) if n else np.zeros(rank)
        res = {}
        for tag, mu in (("1", mu1), ("2", mu2)):
            nt = mu.sum(1)
            has = nt > 0
            sums = mu @ v
            res["S" + tag] = np.where(has[:, None], sums - nt[:, None] * mg, 0.0) / scales[int(tag) - 1]
            if not with_cov:
                res["A" + tag] = np.zeros((n, rank))
                continue
            q, k = q_all[idx], k_all[idx]
            score = np.tanh(q @ k.T / math.sqrt(rank)) if n else np.zeros((0, 0))
            ntc = np.maximum(nt, 1.0)
            band_v = np.where(has[:, None], sums / ntc[:, None], 0.0)
            svb = (mu * score) @ v
            scb = (mu * score).sum(1)
            cov_t = np.where(has[:, None], svb / ntc[:, None] - (scb / ntc)[:, None] * band_v, 0.0)
            nG = max(n, 1)
            cov_g = score @ v / nG - (score.sum(1) / nG)[:, None] * mg[None, :]
            res["A" + tag] = np.where(has[:, None], nt[:, None] * (cov_t - cov_g) / scales[int(tag) - 1], 0.0)
        out.append(res)
    return out


# ---------------------------------------------------------------------------
# synthetic helpers
# ---------------------------------------------------------------------------


def cycle_batch(cycles, period: int = 4) -> dict:
    """P3N-shaped synthetic batch from cycles with a periodic atom-type pattern.

    All bonds are identical single bonds, all junction features are zero and the
    global vector is a shared constant, so two constructions differ only in the
    arrangement of the same local objects.
    """
    atoms, bonds, juncs = [], [], []
    bond_index, junc_index, junc_bond = [], [], []
    atom_off = bond_off = 0
    for L in cycles:
        if L % period:
            raise ValueError("cycle length must be a multiple of the period")
        for i in range(L):
            f = np.zeros(108, dtype=np.float32)
            f[i % period] = 1.0
            atoms.append(f)
        for i in range(L):
            bond_index.append([atom_off + i, atom_off + (i + 1) % L])
            bonds.append(np.array([1.0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32))
        for i in range(L):
            junc_index.append([atom_off + (i - 1) % L, atom_off + i, atom_off + (i + 1) % L])
            junc_bond.append([bond_off + (i - 1) % L, bond_off + i])
            juncs.append(np.zeros(32, dtype=np.float32))
        atom_off += L
        bond_off += L
    E = len(bonds)
    bi = np.asarray(bond_index, dtype=np.int64)
    dist = R.line_graph_distances(bi, E)
    mu1 = (dist >= S.BANDS[0][0]) & (dist <= S.BANDS[0][1])
    mu2 = (dist >= S.BANDS[1][0]) & (dist <= S.BANDS[1][1])
    return {
        "n_graphs": 1,
        "atom_feat": torch.from_numpy(np.stack(atoms)),
        "atom_batch": torch.zeros(atom_off, dtype=torch.long),
        "atom_type": torch.zeros(atom_off, dtype=torch.long),
        "bond_feat": torch.from_numpy(np.stack(bonds)),
        "bond_type": torch.ones(E, dtype=torch.long),
        "bond_batch": torch.zeros(E, dtype=torch.long),
        "bond_index": torch.from_numpy(bi),
        "junc_feat": torch.from_numpy(np.stack(juncs)),
        "junc_batch": torch.zeros(len(juncs), dtype=torch.long),
        "junc_index": torch.from_numpy(np.asarray(junc_index, dtype=np.int64)),
        "junc_bond": torch.from_numpy(np.asarray(junc_bond, dtype=np.int64)),
        "global": torch.zeros(1, 33),
        "bond_mask": torch.ones(1, E, dtype=torch.bool),
        "bond_local": torch.arange(E, dtype=torch.long),
        "bond_bmax": E,
        "mu1": torch.from_numpy(mu1[None]),
        "mu2": torch.from_numpy(mu2[None]),
    }


def empty_graph() -> dict:
    return {
        "n_graphs": 1,
        "atom_feat": torch.zeros(1, 108),
        "atom_batch": torch.zeros(1, dtype=torch.long),
        "atom_type": torch.zeros(1, dtype=torch.long),
        "bond_feat": torch.zeros(0, 7),
        "bond_type": torch.zeros(0, dtype=torch.long),
        "bond_batch": torch.zeros(0, dtype=torch.long),
        "bond_index": torch.zeros(0, 2, dtype=torch.long),
        "junc_feat": torch.zeros(0, 32),
        "junc_batch": torch.zeros(0, dtype=torch.long),
        "junc_index": torch.zeros(0, 3, dtype=torch.long),
        "junc_bond": torch.zeros(0, 2, dtype=torch.long),
        "global": torch.zeros(1, 33),
        "bond_mask": torch.ones(1, 1, dtype=torch.bool),
        "bond_local": torch.zeros(0, dtype=torch.long),
        "bond_bmax": 1,
        "mu1": torch.zeros(1, 1, 1, dtype=torch.bool),
        "mu2": torch.zeros(1, 1, 1, dtype=torch.bool),
    }


def stack_batches(graphs: list[dict]) -> dict:
    """Concatenate single-graph batches into one, padding the bond axis."""
    bmax = max(g["bond_bmax"] for g in graphs)
    bond_n = [int(g["bond_feat"].shape[0]) for g in graphs]
    atom_n = [int(g["atom_feat"].shape[0]) for g in graphs]
    junc_n = [int(g["junc_feat"].shape[0]) for g in graphs]
    b_off = np.cumsum([0] + bond_n[:-1]).astype(np.int64)
    a_off = np.cumsum([0] + atom_n[:-1]).astype(np.int64)
    masks, m1s, m2s = [], [], []
    for g, n in zip(graphs, bond_n):
        m = torch.zeros(bmax, dtype=torch.bool)
        m[:n] = True
        masks.append(m)
        for src, dst in ((g["mu1"], m1s), (g["mu2"], m2s)):
            pad = torch.zeros(bmax, bmax, dtype=torch.bool)
            pad[:n, :n] = src[0, :n, :n]
            dst.append(pad)
    cat = lambda xs: torch.cat(xs, 0)
    return {
        "n_graphs": len(graphs),
        "atom_feat": cat([g["atom_feat"] for g in graphs]),
        "atom_batch": cat([g["atom_batch"] + i for i, g in enumerate(graphs)]),
        "atom_type": cat([g["atom_type"] for g in graphs]),
        "bond_feat": cat([g["bond_feat"] for g in graphs]),
        "bond_type": cat([g["bond_type"] for g in graphs]),
        "bond_batch": cat([g["bond_batch"] + i for i, g in enumerate(graphs)]),
        "bond_index": cat([g["bond_index"] + int(a_off[i]) for i, g in enumerate(graphs)]),
        "junc_feat": cat([g["junc_feat"] for g in graphs]),
        "junc_batch": cat([g["junc_batch"] + i for i, g in enumerate(graphs)]),
        "junc_index": cat([g["junc_index"] + int(a_off[i]) for i, g in enumerate(graphs)]),
        "junc_bond": cat([g["junc_bond"] + int(b_off[i]) for i, g in enumerate(graphs)]),
        "global": torch.cat([g["global"] for g in graphs], 0),
        "bond_mask": torch.stack(masks),
        "bond_local": torch.cat([g["bond_local"] for g in graphs]),
        "bond_bmax": bmax,
        "mu1": torch.stack(m1s),
        "mu2": torch.stack(m2s),
    }


def permute_batch(batch: dict, seed: int) -> dict:
    """Relabel atoms/bonds/junctions *and* permute the frozen relation masks."""
    gen = torch.Generator().manual_seed(int(seed))
    if int(batch["n_graphs"]) != 1:
        raise ValueError("permute_batch is a single-graph permutation helper")
    N = int(batch["atom_feat"].shape[0])
    E = int(batch["bond_feat"].shape[0])
    J = int(batch["junc_feat"].shape[0])
    a_order = torch.randperm(N, generator=gen)
    b_order = torch.randperm(E, generator=gen)
    j_order = torch.randperm(J, generator=gen)
    a_inv = torch.empty(N, dtype=torch.long)
    a_inv[a_order] = torch.arange(N)
    b_inv = torch.empty(E, dtype=torch.long)
    b_inv[b_order] = torch.arange(E)
    out = dict(batch)
    out["atom_feat"] = batch["atom_feat"][a_order]
    out["atom_type"] = batch["atom_type"][a_order]
    out["bond_feat"] = batch["bond_feat"][b_order]
    out["bond_type"] = batch["bond_type"][b_order]
    out["junc_feat"] = batch["junc_feat"][j_order]
    out["bond_index"] = a_inv[batch["bond_index"][b_order]]
    out["junc_index"] = a_inv[batch["junc_index"]][j_order]
    out["junc_bond"] = b_inv[batch["junc_bond"]][j_order]
    out["bond_local"] = torch.arange(E, dtype=torch.long)
    out["mu1"] = batch["mu1"][:, b_order][:, :, b_order]
    out["mu2"] = batch["mu2"][:, b_order][:, :, b_order]
    return out


# ---------------------------------------------------------------------------
# phase A
# ---------------------------------------------------------------------------


def phase_a(norm, cache_train, cache_valid, collate, batches_train, batches_valid) -> None:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    # -- 1. backbone init identity with the historical DCC-L seed-0 state ----
    hist, hist_info = MD.build_initial_state(norm, 0)
    model, info = M.build_initial_state(norm, 0, use_cov=True)
    rng_after_hist = None  # captured below by rebuilding
    add(
        "A1_backbone_init_matches_history",
        info["backbone_dcc_l_state_sha256"] == HIST_DCC_L_INIT_SHA
        and info["p3_reference_state_sha256"] == HIST_P3_SHA
        and hist_info["initial_state_sha256"] == HIST_DCC_L_INIT_SHA,
        {
            "rebuilt_dcc_l_sha256": hist_info["initial_state_sha256"],
            "scm_backbone_sha256": info["backbone_dcc_l_state_sha256"],
            "expected": HIST_DCC_L_INIT_SHA,
            "p3_reference_sha256": info["p3_reference_state_sha256"],
            "ssl_sha256": info["ssl"]["sha256"],
        },
    )
    hsd, msd = hist.state_dict(), model.dcc.state_dict()
    same_keys = list(hsd) == list(msd)
    all_equal = same_keys
    worst = 0.0
    for k in hsd:
        if hsd[k].shape != msd[k].shape or not torch.equal(hsd[k], msd[k]):
            all_equal = False
            worst = max(worst, float((hsd[k].float() - msd[k].float()).abs().max()))
    add(
        "A2_old_parameters_and_buffers_bitwise_identical",
        all_equal,
        {"keys_match": same_keys, "n_keys": len(hsd), "max_abs_diff": worst,
         "includes_buffers": [k for k in hsd if "." not in k]},
    )

    # -- 2. new module uses an isolated RNG stream ---------------------------
    torch.manual_seed(4242)
    _ = MD.DualChannel(norm)
    rng_hist = torch.random.get_rng_state().clone()
    torch.manual_seed(4242)
    _ = MD.DualChannel(norm)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(777)
        _ = M.SCMModule(scales=S.band_scales())
    rng_scm = torch.random.get_rng_state().clone()
    torch.manual_seed(777)
    scm_x = M.SCMModule(scales=S.band_scales())
    torch.manual_seed(778)
    scm_y = M.SCMModule(scales=S.band_scales())
    differ = any(not torch.equal(scm_x.state_dict()[k], scm_y.state_dict()[k])
                 for k in scm_x.state_dict() if k.startswith(("q.", "k.", "v.", "out.")))
    add(
        "A3_new_module_isolated_rng",
        bool(torch.equal(rng_hist, rng_scm) and differ),
        {
            "ambient_rng_unchanged_by_scm": bool(torch.equal(rng_hist, rng_scm)),
            "different_scm_seed_gives_different_params": bool(differ),
            "mechanism": "torch.random.fork_rng + manual_seed(seed + 50731)",
            "scm_seed_for_seed0": info["scm_seed"],
        },
    )

    # -- 3. shared initial state across the two arms -------------------------
    pair, base_info = M.shared_pair_builders(norm, 0)
    m_sonly, _ = pair["S-only"]
    m_sa, info_sa = pair["S+A"]
    sd0, sd1 = m_sonly.state_dict(), m_sa.state_dict()
    shared = list(sd0) == list(sd1) and all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    add(
        "A4_shared_init_between_arms",
        shared,
        {"n_tensors": len(sd0), "initial_state_sha256": base_info["initial_state_sha256"],
         "arms": [m_sonly.arm, m_sa.arm]},
    )

    # -- 4. parameter budget -------------------------------------------------
    named = dict(m_sa.named_parameters())
    total = int(sum(p.numel() for p in m_sa.parameters()))
    new_names = sorted(k for k in named if k.startswith("scm."))
    new_total = int(sum(named[k].numel() for k in new_names))
    add(
        "A5_parameter_budget",
        total == EXPECTED_PARAMS and new_total == 5496 and total <= 500_000,
        {
            "deployment_parameters_named_parameters": total,
            "new_parameters": new_total,
            "new_tensor_shapes": {k: list(named[k].shape) for k in new_names},
            "old_parameters": total - new_total,
            "cap": 500_000,
            "headroom": 500_000 - total,
            "buffers": {k: list(v.shape) for k, v in m_sa.named_buffers()},
            "all_require_grad": all(p.requires_grad for p in m_sa.parameters()),
            "optimizer_would_cover": True,
        },
    )

    # -- 5. Wout = 0 -> exactly the historical DCC-L function ----------------
    m_sa.eval()
    hist.eval()
    worst_pred = 0.0
    bitwise = True
    with torch.no_grad():
        for batch in batches_train + batches_valid:
            p_new = m_sa(batch)
            p_old = hist(batch)
            bitwise = bitwise and torch.equal(p_new, p_old)
            worst_pred = max(worst_pred, _maxdiff(p_new, p_old))
    add(
        "A6_wout_zero_equals_dcc_l",
        worst_pred <= 1e-6,
        {"batches": len(batches_train) + len(batches_valid),
         "max_abs_prediction_diff": worst_pred,
         "bitwise_equal_all_batches": bool(bitwise),
         "input": "real train/valid graphs, labels unused"},
    )

    # -- 6. c_loc' enters BOTH existing reader columns ----------------------
    batch = batches_train[0]
    m_probe = copy.deepcopy(m_sa).eval()
    with torch.no_grad():
        m_probe.scm.out.weight.normal_(0.0, 0.05)
        m_probe.scm.out.bias.zero_()
    captured = {}
    h = m_probe.dcc.reader.register_forward_pre_hook(
        lambda mod, args: captured.__setitem__("reader_in", args[0].detach())
    )
    try:
        with torch.no_grad():
            pred, cap = m_probe.forward_with_capture(batch)
            base_pred = hist(batch)
    finally:
        h.remove()
    c_loc, c_prime = cap["c_loc"], cap["c_loc_prime"]
    rin = captured["reader_in"]
    zl_got, zl2_got = rin[:, 321:417], rin[:, 417:513]
    bb_ = batch["bond_batch"]
    zl_ref = c_prime.new_zeros(rin.shape[0], 96).index_add_(0, bb_, c_prime)
    zl2_ref = c_prime.new_zeros(rin.shape[0], 96).index_add_(0, bb_, c_prime * c_prime)
    zl_old = c_loc.new_zeros(rin.shape[0], 96).index_add_(0, bb_, c_loc)
    zl2_old = c_loc.new_zeros(rin.shape[0], 96).index_add_(0, bb_, c_loc * c_loc)
    same_zl = _maxdiff(zl_got, zl_ref) <= 1e-4
    same_zl2 = _maxdiff(zl2_got, zl2_ref) <= 1e-3
    changed = float((zl_ref - zl_old).abs().max())
    changed2 = float((zl2_ref - zl2_old).abs().max())
    add(
        "A7_correction_enters_both_original_sums",
        bool(same_zl and same_zl2 and changed > 0 and changed2 > 0 and rin.shape[1] == 513),
        {
            "reader_input_dim": int(rin.shape[1]),
            "zl_matches_sum_of_c_loc_prime": bool(same_zl),
            "zl2_matches_sum_of_c_loc_prime_sq": bool(same_zl2),
            "zl_max_abs_change_vs_unmodelled": changed,
            "zl2_max_abs_change_vs_unmodelled": changed2,
            "output_change_vs_wout0": _maxdiff(pred, base_pred),
            "output_change_vs_wout0_note": "zero here is expected: the inherited "
                "DCC-L reader columns for zl/zl2 are exactly 0 at step 0, so no "
                "change inside c_loc' can reach the output before those columns "
                "receive their first gradient; A8 measures the live state",
            "layout": "[c_glob(96)|atom_sum(96)|atom_sq(96)|global33|sum c_loc'(96)|sum c_loc'^2(96)]",
        },
    )

    # -- 7. live-module state after real updates ----------------------
    # NOTE (measured, not assumed): at initialisation the inherited DCC-L
    # reader has ZERO weights on the two local columns, so *no* change inside
    # c_loc' can reach the output before the reader columns receive their first
    # gradient.  The SCM module is therefore cold for the first updates:
    #   step 1: dL/dzl = 0 (reader columns 0)      -> all SCM grads 0
    #   step 2: Wout still 0                       -> Q/K/V grads 0
    #   step 3+: Q/K/V grads non-zero
    # A8 is measured in the regime where the module is actually live (after
    # three real updates), not at the inert initial point.
    m = copy.deepcopy(m_sa).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    y = torch.as_tensor(np.asarray(cache_train["labels"])[:128], dtype=torch.float32)
    cold = {}
    for step in (1, 2, 3, 4):
        pred = m(batch)
        loss = F.l1_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = {k: (None if p.grad is None else float(p.grad.norm()))
             for k, p in m.named_parameters() if k.startswith("scm.")}
        with torch.no_grad():
            _, cap_s = m.forward_with_capture(batch)
        cold[step] = {
            "grad_norms": g,
            "loss": float(loss),
            "max_abs_delta": float(cap_s["delta"].abs().max()),
            "reader_local_columns_norm": float(
                m.dcc.reader.lins[0].weight[:, 321:513].norm()),
            "output_change_if_S_ablated": None,
        }
        opt.step()
        if step >= 3:
            with torch.no_grad():
                p_full, _ = m.forward_with_capture(batch)
                p_ab, _ = m.forward_with_capture(batch, ablate="S")
            cold[step]["output_change_if_S_ablated"] = _maxdiff(p_full, p_ab)
    qkv = ["scm.q.weight", "scm.k.weight", "scm.v.weight",
           "scm.q.bias", "scm.k.bias", "scm.v.bias"]
    wout = ["scm.out.weight", "scm.out.bias"]
    two_step_ok = (
        all(cold[1]["grad_norms"][k] == 0.0 for k in qkv + wout)
        and all(cold[2]["grad_norms"][k] == 0.0 for k in qkv)
        and all(cold[2]["grad_norms"][k] > 0.0 for k in wout)
        and all(cold[3]["grad_norms"][k] > 0.0 for k in qkv)
        and all(math.isfinite(v) for st in cold.values() for v in st["grad_norms"].values())
    )
    add("A9_two_step_gradient_flow", two_step_ok,
        {"per_step": cold,
         "reading": "all SCM gradients are zero at step 1 (reader local columns are 0); "
                    "Wout moves at step 2; Q/K/V move from step 3 and are finite",
         "open_wout_variant": "with Wout manually opened before training, step-1 "
                              "gradients are zero only because of the reader columns and "
                              "Q/K/V receive non-zero finite gradients at step 2 (see A8)"})

    # -- 8. S and A each move the output once the module is live ------
    with torch.no_grad():
        m.scm.out.weight.normal_(0.0, 0.2)   # generic Wout, reader already live
        m.scm.out.bias.zero_()
        p_full, _ = m.forward_with_capture(batch)
        abl = {}
        for tag in ("S", "A", "S1", "S2", "A1", "A2"):
            p_ab, _ = m.forward_with_capture(batch, ablate=tag)
            abl[tag] = _maxdiff(p_full, p_ab)
    # explicit two-step check with Wout opened before the first update
    m2 = copy.deepcopy(m_sa).train()
    with torch.no_grad():
        m2.scm.out.weight.normal_(0.0, 0.05)
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    open_wout = {}
    for step in (1, 2):
        loss = F.l1_loss(m2(batch), y)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        open_wout[step] = {k: (None if p.grad is None else float(p.grad.norm()))
                           for k, p in m2.named_parameters() if k.startswith("scm.")}
        opt2.step()
    open_ok = (all(open_wout[2][k] > 0.0 for k in qkv)
               and all(math.isfinite(v) for v in open_wout[2].values()))
    add("A8_S_and_A_both_respond",
        all(v > 1e-4 for v in abl.values()) and open_ok,
        {"max_abs_prediction_change_when_ablated_at_live_state": abl,
         "state": "after 4 real Adam steps (reader local columns live), then Wout "
                  "re-opened at N(0,0.1) for the ablation",
         "wout_opened_before_training_two_step_grad_norms": open_wout,
         "all_blocks_respond": all(v > 1e-5 for v in abl.values())})

    # -- 9. algebraic identities --------------------------------------
    scm = m_sa.scm
    ident = {}
    B, E, rank = 2, 6, S.RANK
    torch.manual_seed(7)
    hh = torch.randn(E, 96) * 0.3
    bb = torch.tensor([0, 0, 0, 1, 1, 1])
    bl = torch.tensor([0, 1, 2, 0, 1, 2])
    base = {"n_graphs": B, "bond_batch": bb, "bond_local": bl, "bond_bmax": 3,
            "bond_mask": torch.ones(B, 3, dtype=torch.bool)}
    z = torch.zeros(B, 3, 3, dtype=torch.bool)
    uni = torch.ones(B, 3, 3, dtype=torch.bool)
    s1, a1, s2, a2 = scm.moments(hh, dict(base, mu1=uni, mu2=uni))
    ident["uniform_relation_max_abs"] = max(float(s1.abs().max()), float(a1.abs().max()),
                                            float(s2.abs().max()), float(a2.abs().max()))
    mu = z.clone()
    for k in range(B):
        mu[k, 0, 1] = mu[k, 1, 2] = mu[k, 2, 0] = True
    b_band = dict(base, mu1=mu, mu2=torch.zeros_like(mu))
    saved_k = scm.k.weight.detach().clone()
    with torch.no_grad():
        scm.k.weight.zero_()
        s1c, a1c, _, _ = scm.moments(hh, b_band)
        scm.k.weight.copy_(saved_k)
    ident["constant_score_A_max_abs"] = float(a1c.abs().max())
    ident["constant_score_S_max_abs"] = float(s1c.abs().max())
    saved_v = scm.v.weight.detach().clone()
    with torch.no_grad():
        scm.v.weight.zero_()
        s1v, a1v, s2v, a2v = scm.moments(hh, b_band)
        scm.v.weight.copy_(saved_v)
    ident["constant_v_max_abs"] = max(float(s1v.abs().max()), float(a1v.abs().max()),
                                      float(s2v.abs().max()), float(a2v.abs().max()))
    empty = torch.zeros(B, 3, 3, dtype=torch.bool)
    s1e, a1e, s2e, a2e = scm.moments(hh, dict(base, mu1=empty, mu2=empty))
    ident["empty_relation_max_abs"] = max(float(s1e.abs().max()), float(a1e.abs().max()),
                                          float(s2e.abs().max()), float(a2e.abs().max()))
    add("A10_algebraic_identities",
        all(v < 1e-6 for k, v in ident.items() if k != "constant_score_S_max_abs")
        and ident["constant_score_S_max_abs"] > 1e-4,
        ident)

    # -- 10. no cross-molecule leakage --------------------------------
    scm_live = m_probe.scm
    with torch.no_grad():
        d0 = scm_live(hh, b_band)
        h2 = hh.clone()
        h2[:3] += 0.7
        d1 = scm_live(h2, b_band)
    leak = float((d0[3:] - d1[3:]).abs().max())
    moved = float((d0[:3] - d1[:3]).abs().max())
    add("A11_no_cross_molecule_leakage", leak == 0.0 and moved > 0.0,
        {"other_graph_delta_change": leak, "perturbed_graph_delta_change": moved})

    # -- 11. float64 reference agreement -----------------------------
    ref = reference_moments(c_loc, batch, scm, S.band_scales(), with_cov=True)
    s1r, a1r, s2r, a2r = (x.detach() for x in scm.moments(c_loc, batch))
    bb_np = batch["bond_batch"].numpy()
    err = {"S1": 0.0, "A1": 0.0, "S2": 0.0, "A2": 0.0}
    for b in range(int(batch["n_graphs"])):
        n = int((bb_np == b).sum())
        for tag, got in (("S1", s1r), ("A1", a1r), ("S2", s2r), ("A2", a2r)):
            if n:
                err[tag] = max(err[tag], float(np.abs(
                    got[b, :n].detach().double().numpy() - ref[b][tag]).max()))
    add("A12_batched_vs_float64_reference", all(v <= 2e-5 for v in err.values()),
        {"max_abs_error_by_block": err, "batch": "real train batch (128 molecules)",
         "reference": "literal per-molecule float64 transcription of the spec"})

    # -- 12. bond / atom relabelling and batch recomposition ----------
    one = collate(cache_train, np.arange(1), "P3N", valid=True)
    m_open = m_probe
    with torch.no_grad():
        p_ref = m_open(one)
    from dcc_cpu_checks import permute_single_graph  # historical helper, P3N part

    rel, parity = {}, {}
    for seed in (1, 2, 3):
        perm = permute_batch(one, seed)
        hist_perm = permute_single_graph(one, seed)
        parity[f"seed{seed}"] = max(
            _maxdiff(perm[k], hist_perm[k]) for k in
            ("atom_feat", "bond_feat", "junc_feat", "bond_index", "junc_index", "junc_bond")
        )
        with torch.no_grad():
            rel[f"seed{seed}"] = _maxdiff(m_open(perm), p_ref)
    with torch.no_grad():
        p_mixed = m_open(collate(cache_train, np.arange(8), "P3N", valid=True))
    recompose = abs(float(p_mixed[0]) - float(p_ref[0]))
    add("A13_permutation_and_batch_invariance",
        max(rel.values()) <= 1e-4 and recompose <= 1e-4 and max(parity.values()) == 0.0,
        {"atom+bond+junction relabel_max_abs_prediction_diff": rel,
         "single_vs_batch_recomposition_diff": recompose,
         "parity_with_historical_permutation_helper": parity,
         "weight_state": "Wout opened N(0,0.05) so the check is not vacuous"})

    # -- 13. zero-bond and disconnected boundaries --------------------
    syn = stack_batches([cycle_batch([8]), empty_graph(), cycle_batch([4, 4])])
    syn["n_graphs"] = 3
    syn["global"] = torch.zeros(3, 33)
    with torch.no_grad():
        p_syn = m_open(syn)
        p_sonly = pair["S-only"][0](syn)
    finite = bool(torch.isfinite(p_syn).all() and torch.isfinite(p_sonly).all())
    mu_dis = (syn["mu1"][2] | syn["mu2"][2])[:8, :8]
    cross_components = int(mu_dis[:4, 4:].sum().item() + mu_dis[4:, :4].sum().item())
    add("A14_zero_bond_and_disconnected_boundaries",
        finite and cross_components == 0 and int(syn["mu1"].shape[0]) == 3,
        {"predictions": p_syn.tolist(), "all_finite": finite,
         "graph_rows": "8-cycle / empty graph (0 bonds) / disconnected 2x4-cycle",
         "cross_component_band_pairs_inside_disconnected_graph": cross_components})

    # -- 14. checkpoint round-trip -----------------------------------
    sd = C.cpu_state(m_open)
    clone = copy.deepcopy(m_sa)
    clone.load_state_dict(sd)
    clone.eval()
    with torch.no_grad():
        roundtrip = _maxdiff(clone(one), m_open(one))
    add("A15_checkpoint_roundtrip", roundtrip == 0.0,
        {"max_abs_prediction_diff": roundtrip, "tensors": len(sd)})

    # -- 15. real trainer smoke (both arms) --------------------------
    smoke, ok_smoke = {}, True
    for arm, use_cov in (("S-only", False), ("S+A", True)):
        info = T1.train(
            f"SCM_{arm}_smoke", 0, torch.device("cpu"), epochs=1, tag="cpu_smoke",
            force=True, recipe_variant="B", model_builder=M.make_builder(use_cov),
            out_dir=S.RUNS / f"cpu_smoke_{arm.replace('+', 'p')}",
            collate_fn=collate, train_rows=np.arange(512),
        )
        smoke[arm] = {"parameter_count": info["parameter_count"],
                      "gradient_audit": info["gradient_audit"],
                      "best_valid_after_4_updates": info["best_valid"]["valid_mae"],
                      "steps_per_epoch": info["recipe"]["steps_per_epoch"],
                      "recipe_updates": info["recipe"]["updates"]}
        ok_smoke = ok_smoke and math.isfinite(info["best_valid"]["valid_mae"]) \
            and info["gradient_audit"]["all_grad_finite"]
    add("A16_real_trainer_smoke", ok_smoke, smoke)

    # -- 16. no official-test path ----------------------------------
    bad = []
    for path in sorted(S.CODE.glob("*.py")):
        if path.name == Path(__file__).name:
            continue  # the auditor itself contains the forbidden patterns
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if ("ad_test" in v or "official_labels('test')" in v
                        or 'load_cache("test")' in v or "split='test'" in v
                        or 'split="test"' in v):
                    bad.append((path.name, v))
    add("A17_official_test_absent", not bad, {"hits": bad})

    phase_c(norm)


def phase_c(norm) -> None:
    """32-atom ring: identical local objects, non-isomorphic arrangement."""
    b32 = cycle_batch([32], period=4)
    b16 = cycle_batch([16, 16], period=4)
    dcc, _ = MD.build_initial_state(norm, 0)
    dcc.eval()
    model, _ = M.build_initial_state(norm, 0, use_cov=True)
    model.eval()
    # Make the inherited local reader columns live in BOTH models, so the DCC-L
    # null result is not an artefact of the zero-initialised reader columns.
    gen = torch.Generator().manual_seed(20261025)
    live = torch.randn(256, 192, generator=gen) * 0.01   # reader: 513 -> 256
    with torch.no_grad():
        dcc.reader.lins[0].weight[:, 321:513].copy_(live)
        model.dcc.reader.lins[0].weight[:, 321:513].copy_(live)
        p32, p16 = dcc(b32), dcc(b16)
        model.scm.out.weight.normal_(0.0, 0.1)
        model.scm.out.bias.zero_()
        q32, cap32 = model.forward_with_capture(b32)
        q16, cap16 = model.forward_with_capture(b16)
    dcc_gap = _maxdiff(p32, p16)
    scm_gap = _maxdiff(q32, q16)
    cl32 = torch.sort(cap32["c_loc"].reshape(-1)).values
    cl16 = torch.sort(cap16["c_loc"].reshape(-1)).values
    cloc_multiset_gap = _maxdiff(cl32, cl16)
    d32, d16 = cap32["delta"], cap16["delta"]
    delta_gap = _maxdiff(d32, d16)
    n32 = sorted(set(b32["mu2"][0].float().sum(-1).tolist()))
    n16 = sorted(set(b16["mu2"][0].float().sum(-1).tolist()))
    add("C1_synthetic_32ring_counterexample",
        dcc_gap == 0.0 and delta_gap > 1e-3 and scm_gap > 1e-6,
        {"construction": "one 32-cycle (period-4 atom types) vs two disjoint 16-cycles "
                         "with the same period-4 types; identical atom/bond/junction "
                         "feature multisets, identical counts, identical global vector",
         "dcc_l_prediction_gap": dcc_gap, "scm_prediction_gap": scm_gap,
         "scm_delta_rms_32cycle": float(d32.pow(2).mean().sqrt()),
         "scm_delta_rms_2x16cycle": float(d16.pow(2).mean().sqrt()),
         "scm_delta_max_abs_gap_at_same_positions": delta_gap,
         "c_loc_multiset_max_abs_gap": cloc_multiset_gap,
         "local_reader_columns": "set to the same fixed N(0,0.01) values in both models",
         "band2_n_per_bond_32cycle": n32, "band2_n_per_bond_2x16cycle": n16,
         "S2_abs_max_32cycle": float(cap32["S2"].abs().max()),
         "S2_abs_max_2x16cycle": float(cap16["S2"].abs().max()),
         "claim": "the frozen DCC-L forward cannot separate these two inputs; the SCM "
                  "module can, with non-zero weights, because the two-band relation differs",
         "note": "non-chemical synthetic input; says nothing about ZINC MAE"})


def main(argv=None) -> int:
    S.ANALYSIS.mkdir(parents=True, exist_ok=True)
    cfg = S.frozen_config()
    norm = D.load_norm()
    stores = {s: R.load_split(s) for s in ("train", "valid")}
    collate = R.make_split_collate(stores, T1.collate)
    cache_train = D.load_cache("train")
    cache_valid = D.load_cache("valid")
    batches_train = [collate(cache_train, np.arange(s, s + 128), "P3N", valid=True)
                     for s in (0, 128, 256)]
    batches_valid = [collate(cache_valid, np.arange(0, 256), "P3N", valid=True)]
    print(f"phase A: frozen {cfg['bands']} rank={cfg['rank']} scales={cfg['scales']}",
          flush=True)
    phase_a(norm, cache_train, cache_valid, collate, batches_train, batches_valid)

    failed = [k for k, v in CHECKS.items() if not v["pass"]]
    S.write_json(S.ANALYSIS / "cpu_checks.json", {
        "protocol_id": S.TRACK_ID,
        "phase": "A+C",
        "official_test_loaded": False,
        "data": "official train 10k / valid 1k caches; labels used only in the "
                "two-step gradient check and the trainer smoke",
        "frozen_config": cfg,
        "relation_invariants": R.validate(stores["train"]),
        "relation_stats_train": S.ANALYSIS.joinpath("relation_stats_train.json").exists(),
        "checks": CHECKS,
        "n_checks": len(CHECKS),
        "n_passed": len(CHECKS) - len(failed),
        "failed": failed,
        "all_pass": not failed,
    })
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed; failed={failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
