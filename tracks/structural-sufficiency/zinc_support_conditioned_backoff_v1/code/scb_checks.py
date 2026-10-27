"""SCSB v1 Phase 0 CPU gate.

No GPU, no official valid labels, no official test.  Every check writes a
machine-readable record to ``analysis/cpu_checks.json``.  Any failed check
aborts before GPU work.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import scb_common as S
import scb_model as SM
import scb_train as ST
import ad_common as C
import ad_train as T1
import dcc_models as M
import dcc_runner as DR

OUT = S.ANALYSIS / "cpu_checks.json"
RNG = np.random.default_rng(20261028)
RESULTS: list[dict] = []


def record(name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        detail = fn()
        passed = bool(detail.pop("_passed", True))
        RESULTS.append({"check": name, "passed": passed, "seconds": time.perf_counter() - t0,
                        "detail": detail})
        print(f"[PASS] {name} ({time.perf_counter() - t0:.1f}s)" if passed
              else f"[FAIL] {name} ({time.perf_counter() - t0:.1f}s)", flush=True)
    except Exception as exc:  # noqa: BLE001 - the gate must record, then stop
        RESULTS.append({"check": name, "passed": False, "seconds": time.perf_counter() - t0,
                        "error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)


# --------------------------------------------------------------------- checks
def check_split():
    rows = S.split_rows()
    cfg = S.load_cfg()["data"]
    return {
        "split_sha256": rows["split_sha256"],
        "fit_rows": int(len(rows["fit"])),
        "monitor_rows": int(len(rows["monitor"])),
        "locked_rows": int(len(rows["locked"])),
        "fit_sha256": rows["hashes"]["fit"],
        "monitor_sha256": rows["hashes"]["monitor"],
        "locked_sha256": rows["hashes"]["locked"],
        "cfg_fit_rows_sha256": cfg["fit_rows_sha256"],
        "disjoint_and_cover_10k": True,
        "locked_labels_read": False,
    }


def check_support(cache, support, per_atom, rows):
    src = inspect.getsource(S.build_support) + inspect.getsource(S.alpha_for_rows)
    assert '["labels"]' not in src and "['labels']" not in src, "support/alpha source reads labels"
    assert "monitor" not in src.lower() and "locked" not in src.lower(), "support/alpha source mentions held-out splits"

    # Rebuild with every non-Fit molecule's signature list blanked: the table
    # must be identical (it is Fit-only by construction).
    fit_set = set(rows["fit"].tolist())
    per_fit_only = [per_atom[i] if i in fit_set else ["<heldout>"] * len(per_atom[i])
                    for i in range(len(per_atom))]
    support2 = S.build_support(cache, rows["fit"], per_fit_only)
    assert support["meta"]["df_sha256"] == support2["meta"]["df_sha256"]
    fit_mask = np.zeros_like(support["atom_support_df"], dtype=bool)
    for r in rows["fit"]:
        fit_mask[int(cache["atom_ptr"][r]):int(cache["atom_ptr"][r + 1])] = True
    assert np.array_equal(support["atom_support_df"][fit_mask], support2["atom_support_df"][fit_mask])
    assert np.array_equal(support["atom_support_loo"][fit_mask], support2["atom_support_loo"][fit_mask])

    # Labels are irrelevant: permute them and rebuild.
    cache_perm = dict(cache)
    cache_perm["labels"] = RNG.permutation(np.asarray(cache["labels"]))
    support3 = S.build_support(cache_perm, rows["fit"], per_atom)
    assert support["meta"]["df_sha256"] == support3["meta"]["df_sha256"]

    # Save the support manifest without the vocabulary (keep the cache small).
    manifest = dict(support["meta"])
    manifest["fit_rows_sha256"] = S.sha256_array(rows["fit"])
    manifest["support_vocab_head"] = support["vocab"][:5]
    S.write_json(S.CACHE_DIR / "support_manifest.json", manifest)
    return {
        "vocab_size": int(support["meta"]["vocab_size"]),
        "df_sha256": support["meta"]["df_sha256"],
        "support_table_identical_without_heldout_structures": True,
        "support_table_identical_under_label_permutation": True,
        "source_contains_labels": False,
        "manifest": str(S.CACHE_DIR / "support_manifest.json"),
    }


def check_loo(cache, support, per_atom, rows):
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    fit = rows["fit"]
    fit_sets = [set(per_atom[i]) for i in range(len(per_atom))]
    alpha = S.alpha_for_rows(cache, support, rows["fit"], leave_one_out_for=fit)
    alpha_locked = S.alpha_for_rows(cache, support, rows["locked"])

    # Brute-force LOO on 25 Fit molecules: count Fit molecules other than self.
    fit_pick = RNG.choice(fit, size=25, replace=False)
    max_err = 0.0
    for r in fit_pick.tolist():
        b0, b1 = int(bp[r]), int(bp[r + 1])
        for e in range(b0, b1):
            u, v = int(bi[e, 0]), int(bi[e, 1])
            sig_u, sig_v = per_atom[r][u], per_atom[r][v]
            su = sum(1 for q in fit.tolist() if q != r and sig_u in fit_sets[q])
            sv = sum(1 for q in fit.tolist() if q != r and sig_v in fit_sets[q])
            got = S.alpha_from_support_values(np.asarray([min(su, sv)]))[0]
            max_err = max(max_err, abs(float(got) - float(alpha[e])))

    # Brute-force non-Fit on 25 Locked molecules: count all Fit molecules.
    lock_pick = RNG.choice(rows["locked"], size=25, replace=False)
    max_err_lock = 0.0
    for r in lock_pick.tolist():
        b0, b1 = int(bp[r]), int(bp[r + 1])
        for e in range(b0, b1):
            u, v = int(bi[e, 0]), int(bi[e, 1])
            sig_u, sig_v = per_atom[r][u], per_atom[r][v]
            su = sum(1 for q in fit.tolist() if sig_u in fit_sets[q])
            sv = sum(1 for q in fit.tolist() if sig_v in fit_sets[q])
            got = S.alpha_from_support_values(np.asarray([min(su, sv)]))[0]
            max_err_lock = max(max_err_lock, abs(float(got) - float(alpha_locked[e])))

    # Add-query-then-LOO invariant.
    inv_ok = True
    for split_rows, alpha_base in ((fit_pick, alpha), (lock_pick, alpha_locked)):
        for r in split_rows[:5].tolist():
            df2 = Counter()
            for q in fit.tolist():
                if q != r:
                    df2.update(set(per_atom[q]))
            df2.update(set(per_atom[r]))  # add query to Fit
            b0, b1 = int(bp[r]), int(bp[r + 1])
            for e in range(b0, b1):
                u, v = int(bi[e, 0]), int(bi[e, 1])
                su = df2[per_atom[r][u]] - 1
                sv = df2[per_atom[r][v]] - 1
                got = S.alpha_from_support_values(np.asarray([min(su, sv)]))[0]
                if abs(float(got) - float(alpha_base[e])) > 1e-7:
                    inv_ok = False
    assert max_err < 1e-6, max_err
    assert max_err_lock < 1e-6, max_err_lock
    assert inv_ok
    return {
        "fit_bruteforce_max_alpha_err": float(max_err),
        "locked_bruteforce_max_alpha_err": float(max_err_lock),
        "add_query_then_loo_invariant": bool(inv_ok),
        "n_fit_pick": int(len(fit_pick)),
        "n_locked_pick": int(len(lock_pick)),
    }


def _permuted_graph(atoms, bonds, rng):
    n = len(atoms)
    p = rng.permutation(n)  # new index i holds old atom p[i]
    inv = np.empty(n, dtype=np.int64)
    inv[p] = np.arange(n)
    new_atoms = atoms[p]
    new_bonds = []
    for u, v, t in bonds:
        a, b = int(inv[u]), int(inv[v])
        if rng.random() < 0.5:
            a, b = b, a
        new_bonds.append((a, b, int(t)))
    order = rng.permutation(len(new_bonds))
    new_bonds = [new_bonds[i] for i in order]
    return new_atoms, new_bonds


def check_permutation(cache, per_atom, support):
    graphs = S.DC.molecule_graphs(cache)
    df = {s: int(d) for s, d in zip(support["vocab"], support["df"])}
    picks = RNG.choice(len(graphs), size=50, replace=False)
    sig_ok = alpha_ok = 0
    for r in picks.tolist():
        atoms, bonds = graphs[r]
        new_atoms, new_bonds = _permuted_graph(atoms, bonds, RNG)
        mini = {
            "atom_ptr": np.asarray([0, len(new_atoms)]),
            "bond_ptr": np.asarray([0, len(new_bonds)]),
            "atom_type": np.asarray(new_atoms),
            "bond_type": np.asarray([t for _, _, t in new_bonds]),
            "bond_index": np.asarray([[u, v] for u, v, _ in new_bonds]),
        }
        _, new_per_atom = S.DC.radius2_signature_sets(S.DC.molecule_graphs(mini))
        if sorted(new_per_atom[0]) == sorted(per_atom[r]):
            sig_ok += 1
        # alpha multiset from direct support lookups on both graphs
        def alphas(atom_list, bond_list):
            out = []
            for u, v, _ in bond_list:
                su = df.get(atom_list[u], 0)
                sv = df.get(atom_list[v], 0)
                out.append(float(S.alpha_from_support_values(np.asarray([min(su, sv)]))[0]))
            return sorted(out)
        old_bonds = [(u, v, t) for u, v, t in bonds]
        new_bonds_plain = [(u, v, t) for u, v, t in new_bonds]
        if np.allclose(alphas(per_atom[r], old_bonds), alphas(new_per_atom[0], new_bonds_plain)):
            alpha_ok += 1
    assert sig_ok == len(picks), (sig_ok, len(picks))
    assert alpha_ok == len(picks), (alpha_ok, len(picks))
    return {"molecules": int(len(picks)), "signature_multiset_invariant": int(sig_ok),
            "alpha_multiset_invariant": int(alpha_ok)}


def check_model(cache, rows, support):
    norm = S.DB.fit_only_norm(cache, rows["fit"])
    base, info = M.build_initial_state(norm, 0)
    model, _ = SM.build_scb(0, norm)
    assert SM.param_count(model) == 494193, SM.param_count(model)
    assert SM.param_count(base) == 494193
    assert set(model.state_dict()) == set(base.state_dict())
    for k in model.state_dict():
        assert model.state_dict()[k].shape == base.state_dict()[k].shape
    assert C.state_hash(model) == C.state_hash(base), "SCB initial state != base initial state"

    ck = torch.load(S.DCB_RUNS / "fit8k_s0" / "best_monitor.pt", map_location="cpu", weights_only=False)
    pred_ref = torch.load(S.DCB_RUNS / "fit8k_s0" / "predictions_best_monitor.pt",
                          map_location="cpu", weights_only=False)
    C.load_state(model, ck["state_dict"], torch.device("cpu"))
    pred_none, mae_none = S.eval_with_alpha(model, cache, rows["locked"], None, torch.device("cpu"))
    assert np.max(np.abs(pred_none - np.asarray(pred_ref["locked_pred"]))) < 5e-6
    alpha_one = np.ones(int(np.asarray(cache["bond_ptr"])[-1]), dtype=np.float32)
    pred_one, mae_one = S.eval_with_alpha(model, cache, rows["locked"], alpha_one, torch.device("cpu"))
    max_diff = float(np.max(np.abs(pred_one - pred_none)))
    assert max_diff < 1e-6, max_diff
    assert abs(mae_none - mae_one) < 1e-7

    # alpha=None must also match the parent class forward exactly.
    base_model = M.build_initial_state(norm, 0)[0]
    C.load_state(base_model, ck["state_dict"], torch.device("cpu"))
    p_base, _ = S.eval_with_alpha(base_model, cache, rows["locked"], None, torch.device("cpu"))
    assert np.max(np.abs(p_base - pred_none)) < 1e-7
    return {
        "parameters": SM.param_count(model),
        "initial_state_sha256": C.state_hash(model),
        "ckpt_epoch": int(ck["epoch"]),
        "ckpt_monitor_mae": float(ck["monitor_mae"]),
        "locked_mae_none": float(mae_none),
        "locked_mae_alpha1": float(mae_one),
        "max_abs_diff_vs_saved_L": float(np.max(np.abs(pred_none - np.asarray(pred_ref["locked_pred"])))),
        "max_abs_diff_alpha1_vs_none": max_diff,
        "max_abs_diff_vs_parent_class": float(np.max(np.abs(p_base - pred_none))),
    }


def check_gradient_real(cache, support, rows):
    norm = S.DB.fit_only_norm(cache, rows["fit"])
    model, _ = SM.build_scb(0, norm)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    alpha = S.alpha_for_rows(cache, support, rows["fit"], leave_one_out_for=rows["fit"])
    rng = np.random.default_rng(0)
    batch_rows = rng.choice(rows["fit"], size=128, replace=False)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    batch = DR.collate_p3n(cache, batch_rows, "P3N", valid=False)
    gidx = np.concatenate([np.arange(bp[r], bp[r + 1]) for r in batch_rows])
    a = torch.as_tensor(alpha[gidx], dtype=torch.float32)
    target = torch.as_tensor(np.asarray(cache["labels"])[batch_rows], dtype=torch.float32)
    # Step 1: the reader's new local columns are zero at initialisation, so the
    # local composer is intentionally gradient-dead on the first backward.  Its
    # reader columns receive gradient and wake the path up; step 2 must show a
    # nonzero local-composer gradient.
    pred = model(batch, a)
    assert torch.isfinite(pred).all()
    loss = F.l1_loss(pred.view(-1), target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn1 = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    assert math.isfinite(float(gn1))
    opt.step()
    opt.zero_grad(set_to_none=True)

    pred = model(batch, a)
    assert torch.isfinite(pred).all()
    loss = F.l1_loss(pred.view(-1), target)
    loss.backward()
    finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    lc_grad = model.local_composer.lins[0].weight.grad
    assert finite
    assert lc_grad is not None and float(lc_grad.abs().sum()) > 0
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    assert math.isfinite(float(gn))
    opt.step()
    return {"finite_grads": bool(finite),
            "local_composer_first_layer_grad_abs_sum": float(lc_grad.abs().sum()),
            "grad_norm": float(gn),
            "optimizer_covers_all_parameters": True,
            "parameters": SM.param_count(model)}


def check_alpha(cache, support, rows, per_atom):
    fit = rows["fit"]
    alpha_train = S.alpha_for_rows(cache, support, fit, leave_one_out_for=fit)
    alpha_locked = S.alpha_for_rows(cache, support, rows["locked"])
    s_train = S.bond_support_for_rows(cache, support, fit, leave_one_out_for=fit)
    s_locked = S.bond_support_for_rows(cache, support, rows["locked"])
    locked_groups = S.group_payload(cache, support, rows["locked"], "train")
    primary = locked_groups["masks"]["primary"]

    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    prim_bonds = np.concatenate([np.arange(bp[r], bp[r + 1])
                                 for r in rows["locked"][primary]])
    a_prim = alpha_locked[prim_bonds]
    s_prim = s_locked[prim_bonds]
    frac_le_half = float(np.mean(a_prim <= 0.5))
    frac_zero = float(np.mean(a_prim == 0.0))
    mean_s0 = float(a_prim[s_prim == 0].mean()) if np.any(s_prim == 0) else None
    mean_ge20 = float(a_prim[s_prim >= 20].mean()) if np.any(s_prim >= 20) else None
    assert np.all(a_prim[s_prim == 0] == 0.0)
    assert frac_le_half >= 0.05, frac_le_half
    assert mean_ge20 is not None and mean_ge20 >= 0.85, mean_ge20

    # P arm: exactly the same per-molecule multiset, different positions.
    pa = S.permute_alpha_within_molecules(cache, alpha_locked, rows["locked"], 20261028)
    multiset_ok = True
    for r in rows["locked"].tolist():
        b0, b1 = int(bp[r]), int(bp[r + 1])
        if not np.allclose(np.sort(alpha_locked[b0:b1]), np.sort(pa[b0:b1])):
            multiset_ok = False
    changed = float(np.mean(pa[prim_bonds] != a_prim))
    # Pooled correlation is contaminated by between-molecule alpha means, which
    # the within-molecule permutation deliberately preserves.  The placement
    # question is a within-molecule one, so average the per-molecule correlation.
    per_mol_corr = []
    for r in rows["locked"][primary].tolist():
        b0, b1 = int(bp[r]), int(bp[r + 1])
        x, y = alpha_locked[b0:b1], pa[b0:b1]
        if np.std(x) > 0 and np.std(y) > 0:
            per_mol_corr.append(float(np.corrcoef(x, y)[0, 1]))
    corr = float(np.mean(per_mol_corr)) if per_mol_corr else float("nan")
    pooled_corr = float(np.corrcoef(a_prim, pa[prim_bonds])[0, 1]) if len(a_prim) > 2 else float("nan")
    assert multiset_ok
    assert changed >= 0.5, changed
    assert corr < 0.05, corr

    # alpha on training rows: LOO means truly unseen-only signatures get 0.
    train_bonds = np.concatenate([np.arange(bp[r], bp[r + 1]) for r in fit])
    a_tr = alpha_train[train_bonds]
    s_tr = s_train[train_bonds]
    locked_bonds = np.concatenate([np.arange(bp[r], bp[r + 1]) for r in rows["locked"]])
    a_lock_all = alpha_locked[locked_bonds]
    s_lock_all = s_locked[locked_bonds]

    def by_band(a, s):
        return {
            "s0": {"n": int(np.sum(s == 0)), "mean_alpha": float(a[s == 0].mean()) if np.any(s == 0) else None},
            "s1_4": {"n": int(np.sum((s >= 1) & (s <= 4))), "mean_alpha": float(a[(s >= 1) & (s <= 4)].mean()) if np.any((s >= 1) & (s <= 4)) else None},
            "s5_19": {"n": int(np.sum((s >= 5) & (s <= 19))), "mean_alpha": float(a[(s >= 5) & (s <= 19)].mean()) if np.any((s >= 5) & (s <= 19)) else None},
            "s20p": {"n": int(np.sum(s >= 20)), "mean_alpha": float(a[s >= 20].mean()) if np.any(s >= 20) else None},
        }

    S.write_json(S.CACHE_DIR / "alpha_manifest.json", {
        "fit_alpha_mean": float(a_tr.mean()),
        "fit_alpha_median": float(np.median(a_tr)),
        "fit_alpha_frac_zero": float(np.mean(a_tr == 0.0)),
        "fit_s_zero_bonds": int(np.sum(s_tr == 0)),
        "fit_alpha_by_s": by_band(a_tr, s_tr),
        "locked_all_alpha_mean": float(a_lock_all.mean()),
        "locked_all_alpha_median": float(np.median(a_lock_all)),
        "locked_all_frac_le_0.5": float(np.mean(a_lock_all <= 0.5)),
        "locked_all_alpha_by_s": by_band(a_lock_all, s_lock_all),
        "locked_primary_alpha_by_s": by_band(a_prim, s_prim),
        "locked_primary_alpha_mean": float(a_prim.mean()),
        "locked_primary_frac_le_0.5": frac_le_half,
        "locked_primary_s_zero_frac": frac_zero,
        "locked_primary_s_zero_mean_alpha": mean_s0,
        "locked_primary_s_ge20_mean_alpha": mean_ge20,
        "permutation_changed_frac": changed,
        "permutation_in_mol_corr": corr,
        "permutation_pooled_corr": pooled_corr,
        "denominator": 4.0,
        "labels_used": False,
    })
    return {
        "fit_alpha_mean": float(a_tr.mean()),
        "fit_alpha_median": float(np.median(a_tr)),
        "fit_alpha_frac_zero": float(np.mean(a_tr == 0.0)),
        "fit_s_zero_bonds": int(np.sum(s_tr == 0)),
        "fit_alpha_by_s": by_band(a_tr, s_tr),
        "locked_all_alpha_mean": float(a_lock_all.mean()),
        "locked_all_alpha_median": float(np.median(a_lock_all)),
        "locked_all_frac_le_0.5": float(np.mean(a_lock_all <= 0.5)),
        "locked_all_alpha_by_s": by_band(a_lock_all, s_lock_all),
        "locked_primary_alpha_by_s": by_band(a_prim, s_prim),
        "locked_primary_alpha_mean": float(a_prim.mean()),
        "locked_primary_frac_le_0.5": frac_le_half,
        "locked_primary_frac_zero": frac_zero,
        "locked_primary_s_zero_mean_alpha": mean_s0,
        "locked_primary_s_ge20_mean_alpha": mean_ge20,
        "permutation_multiset_identical": bool(multiset_ok),
        "permutation_changed_frac": changed,
        "permutation_in_mol_corr": corr,
        "permutation_pooled_corr": pooled_corr,
    }


def check_synthetic():
    torch.manual_seed(0)
    norm = {"global_mean": torch.zeros(33), "global_scale": torch.ones(33)}
    model = SM.SCBDualChannel(norm)
    W = 8
    # 6-atom chain 0-1-2-3-4-5.  The P3 junction set is the degree-2 atoms
    # 1,2,3, each incident to two bonds: j0=(0,1), j1=(1,2), j2=(2,3).
    # Bond 0 sees only j0, bond 3 only j2, so Wg != Wl and alpha=0/1 differ.
    w = torch.stack([torch.full((W,), 1.0), torch.full((W,), 2.0),
                     torch.full((W,), 4.0)])
    jb = torch.tensor([[0, 1], [1, 2], [2, 3]])
    batch = {"n_graphs": 1, "junc_bond": jb,
             "junc_batch": torch.tensor([0, 0, 0]), "bond_batch": torch.tensor([0, 0, 0, 0]),
             "bond_feat": torch.zeros(4, 8)}
    Wg, Vg, n_b = model.group_to_bonds(w, batch)
    Wl, Vl, _ = model.local_grouping(w, batch, organized=True)
    a0 = torch.zeros(4, dtype=torch.float32)
    a1 = torch.ones(4, dtype=torch.float32)
    W0 = a0.unsqueeze(1) * Wl + (1 - a0.unsqueeze(1)) * Wg
    V0 = a0.unsqueeze(1) * Vl + (1 - a0.unsqueeze(1)) * Vg
    W1 = a1.unsqueeze(1) * Wl + (1 - a1.unsqueeze(1)) * Wg
    V1 = a1.unsqueeze(1) * Vl + (1 - a1.unsqueeze(1)) * Vg
    assert torch.allclose(W0, Wg) and torch.allclose(V0, Vg)
    assert torch.allclose(W1, Wl) and torch.allclose(V1, Vl)
    assert not torch.allclose(W0, W1), "synthetic graph must distinguish the endpoints"
    return {
        "alpha0_equals_group_statistic": True,
        "alpha1_equals_local_statistic": True,
        "Wg_rows": Wg[:, 0].tolist(),
        "Wl_rows": Wl[:, 0].tolist(),
        "Vg_rows": Vg[:, 0].tolist(),
        "Vl_rows": Vl[:, 0].tolist(),
        "n_b": n_b.tolist(),
    }


def check_groups(cache, support, rows):
    locked = S.group_payload(cache, support, rows["locked"], "train")
    monitor = S.group_payload(cache, support, rows["monitor"], "train")
    counts = S.check_frozen_group_counts(locked, monitor)
    pstats = S.size_cycle_stats(cache, rows["locked"], locked["masks"]["primary"])
    rest = ~locked["masks"]["primary"]
    rstats = S.size_cycle_stats(cache, rows["locked"], rest)
    counts["primary_size_cycle"] = pstats
    counts["locked_rest_size_cycle"] = rstats
    counts["all_locked_size_cycle"] = S.size_cycle_stats(cache, rows["locked"], np.ones(len(rows["locked"]), bool))
    counts["primary_n"] = int(locked["counts"]["primary"])
    assert counts["primary_n"] == S.PHASE1_PRIMARY_N
    return counts


def check_cache():
    out = {}
    for name in ("ad_train.pt", "ad_valid.pt"):
        p = C.CACHE / name
        out[name] = {"exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0,
                     "sha256": S.sha256_file(p) if p.exists() else None}
    for fit in ("fit8k_s0", "fit8k_s1"):
        for ck in ("best_monitor.pt", "final.pt"):
            p = S.DCB_RUNS / fit / ck
            out[f"{fit}/{ck}"] = {"exists": p.exists(), "sha256": S.sha256_file(p) if p.exists() else None}
        for pr in ("predictions_best_monitor.pt", "predictions_final.pt"):
            p = S.DCB_RUNS / fit / pr
            out[f"{fit}/{pr}"] = {"exists": p.exists(), "sha256": S.sha256_file(p) if p.exists() else None}
    assert all(v["exists"] for v in out.values())
    return out


def check_train_loop():
    """One full-horizon epoch of L must match the frozen DCB L curve[0]."""
    dcb = S.read_json(S.DCB_RUNS / "fit8k_s0" / "train.json")["curve"][0]
    info = ST.train("L", 0, "fit8k", torch.device("cpu"), total_updates=47400,
                    eval_every=20, out_root=S.CACHE_DIR / "loopprobe", force=True,
                    max_epochs=1)
    ours = info["curve"][0]
    diffs = {
        "online_train_mae": abs(float(ours["online_train_mae"]) - float(dcb["online_train_mae"])),
        "monitor_mae": abs(float(ours["monitor_mae"]) - float(dcb["monitor_mae"])),
        "lr": abs(float(ours["lr"]) - float(dcb["lr"])),
        "step": int(ours["step"]) - int(dcb["step"]),
    }
    assert diffs["online_train_mae"] < 1e-5, diffs
    assert diffs["monitor_mae"] < 1e-5, diffs
    assert diffs["lr"] < 1e-12, diffs
    assert diffs["step"] == 0, diffs
    return {"dcb_curve0": dcb, "scb_curve0": ours, "diffs": diffs,
            "note": "identical init/state, batch order, LR schedule and 47400 horizon"}


def check_no_test():
    r = S.assert_no_test_access()
    assert not r["forbidden_hits"], r["forbidden_hits"]
    return r


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if args.device != "cpu":
        raise SystemExit("Phase 0 gate is CPU-only; pass --device cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    S.ANALYSIS.mkdir(parents=True, exist_ok=True)

    cache = S.load_train_cache()
    rows = S.split_rows()
    sets, per_atom = S.build_signatures(cache)
    support = S.build_support(cache, rows["fit"], per_atom)
    rows["support"] = support

    record("split_isolation", check_split)
    record("support_fit_only", lambda: check_support(cache, support, per_atom, rows))
    record("leave_one_out", lambda: check_loo(cache, support, per_atom, rows))
    record("permutation_invariance", lambda: check_permutation(cache, per_atom, support))
    record("model_reproduction_L", lambda: check_model(cache, rows, support))
    record("gradients", lambda: check_gradient_real(cache, support, rows))
    record("alpha_distribution", lambda: check_alpha(cache, support, rows, per_atom))
    record("synthetic_endpoints", check_synthetic)
    record("group_counts", lambda: check_groups(cache, support, rows))
    record("train_loop_matches_frozen_L", check_train_loop)
    record("cache_and_checkpoints", check_cache)
    record("no_test_access", check_no_test)
    record("manual_sidecar_provenance", lambda: S.manual_sidecar_info())

    payload = {
        "protocol": S.load_cfg()["protocol_id"],
        "utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "n_checks": len(RESULTS),
        "n_passed": sum(1 for r in RESULTS if r["passed"]),
        "all_passed": all(r["passed"] for r in RESULTS),
        "checks": RESULTS,
    }
    S.write_json(OUT, payload)
    print(json.dumps({k: payload[k] for k in ("n_checks", "n_passed", "all_passed")}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
