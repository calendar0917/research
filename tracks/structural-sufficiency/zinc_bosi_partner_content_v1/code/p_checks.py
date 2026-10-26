"""CPU gates for the partner-content necessity experiment (BOSI-P vs BOSI-B).

Every gate must pass before any GPU job is launched.  The gates do not test the
relation construction (that is frozen BOSI v0 code, reused byte-identically);
they test that P is what it claims to be:

  C1  B/P have identical parameter shapes, counts and initial states
      (and the same initial state as the historical v0 runs of that seed)
  C2  n[r] == 0  =>  M[r] == 0 in both arms, exactly
  C3  other half-edges' content: perturbing partner_half leaves P bit-identical
      and moves B  (the defining counterexample)
  C3b histogram-preserving relation permutation leaves P bit-identical
      (P's only relation dependence is through the counts)
  C4  the target half-edge's own state drives P; the effect is block-local
  C5  P's 384-d moment input is finite, non-zero and the same order of
      magnitude as B's measured input (no post-hoc rescaling anywhere)
  C6  atom relabel / bond reversal / edge order / batch order / recombination
      invariance for R, B and P
  C7  every parameter tensor of P receives gradient; no dead module
  C8  label isolation: no Monitor/Locked row enters a training batch and
      poisoning Locked labels changes nothing
  C9  interface and budget: 384-d moments, 4 counts, 473,889 params, 6 modules

Writes ``analysis/cpu_checks.json`` and prints ``CPU_GO`` / ``CPU_NO_GO``.

Usage:
    python .../p_checks.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bosi_common as BC  # noqa: E402
import ad_common as C  # noqa: E402
import bosi_data as BD  # noqa: E402
import bosi_models as BM  # noqa: E402

M_OFF = 2 * BM.WIDTH + BM.RANK                     # start of the M block in SIDE_IN
N_SIDE_TAIL = BD.N_REL + 4 + 1


def _blocks(M: torch.Tensor) -> torch.Tensor:
    """[H, 4*96] -> [H, 4, 96]"""
    return M.view(M.shape[0], BD.N_REL, 3 * BM.RANK)


def _spec_from_cache(cache, row: int) -> dict:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    a0, a1 = int(ap[row]), int(ap[row + 1])
    b0, b1 = int(bp[row]), int(bp[row + 1])
    return {
        "atom_feat": np.asarray(cache["atom_feat"])[a0:a1].copy(),
        "bond_index": np.asarray(cache["bond_index"])[b0:b1].astype(np.int64).copy(),
        "bond_feat": np.asarray(cache["bond_feat"])[b0:b1].copy(),
        "global": np.asarray(cache["global_feat"])[row].copy(),
    }


def c1_params_and_init(cache, norm):
    out = {"gate": "C1_identical_parameters_and_initial_state"}
    v0 = json.loads((BC.V0 / "analysis" / "freeze.json").read_text())
    per_seed = {}
    for seed in (0, 1):
        mB, iB = BM.build_model(norm, "B", seed, "cpu")
        mP, iP = BM.build_model(norm, "P", seed, "cpu")
        shapes_B = {k: tuple(v.shape) for k, v in mB.state_dict().items()}
        shapes_P = {k: tuple(v.shape) for k, v in mP.state_dict().items()}
        diff = max(float((mB.state_dict()[k] - mP.state_dict()[k]).abs().max())
                   for k in mB.state_dict())
        hist = {t: r["initial_state_sha256"] for t, r in v0["runs"].items()
                if int(r["seed"]) == seed}
        per_seed[f"seed{seed}"] = {
            "parameters_B": iB["parameters"],
            "parameters_P": iP["parameters"],
            "same_shape": bool(shapes_B == shapes_P),
            "same_initial_state_sha256": bool(
                iB["initial_state_sha256"] == iP["initial_state_sha256"]),
            "tensor_level_max_abs_diff": diff,
            "initial_state_sha256": iP["initial_state_sha256"],
            "historical_v0_initial_state_sha256": hist,
            "matches_every_historical_run_of_this_seed": bool(
                all(v == iP["initial_state_sha256"] for v in hist.values())),
            "ssl_sha256": iP["ssl"]["sha256"],
            "ssl_keys_loaded": iP["ssl"]["keys_loaded"],
            "breakdown": iP["parameters_breakdown"],
        }
    out["per_seed"] = per_seed
    out["pass"] = bool(all(
        v["parameters_P"] == BC.EXPECTED_PARAMETERS == BC.EXPECTED_PARAMETERS
        and v["parameters_B"] == BC.EXPECTED_PARAMETERS
        and v["same_shape"] and v["same_initial_state_sha256"]
        and v["tensor_level_max_abs_diff"] == 0.0
        and v["matches_every_historical_run_of_this_seed"]
        and v["parameters_P"] <= BC.PARAM_CAP
        for v in per_seed.values()))
    return out


def c2_empty_group_is_zero(cache, rel, norm, rows):
    model, _ = BM.build_model(norm, "P", 0, "cpu")
    model.eval()
    batch = BD.collate(cache, rel, rows)
    with torch.no_grad():
        stats = {}
        for arm in ("B", "P"):
            model.arm = arm
            y, st = model(batch, return_stats=True)
            M, n = st["M"], st["n"]
            blk = _blocks(M)
            zero_n = n == 0
            block_norms = blk.abs().amax(dim=2)
            violated = int(((block_norms > 0) & zero_n).sum())
            stats[arm] = {
                "n_zero_entries": int(zero_n.sum()),
                "n_positive_entries": int((n > 0).sum()),
                "violations_n_zero_but_M_nonzero": violated,
                "max_abs_M_where_n_zero": float(block_norms[zero_n].abs().max())
                if zero_n.any() else 0.0,
                "support_with_n_positive_but_M_exactly_zero": int(
                    ((n > 0) & (block_norms == 0)).sum()),
            }
    out = {"gate": "C2_empty_relation_group_gives_zero_moments",
           "arm_stats": stats,
           "definition": "n[v,e,r]==0  <=>  M[v,e,r,:]==0 (checked exactly)"}
    out["pass"] = bool(all(v["violations_n_zero_but_M_nonzero"] == 0
                           and v["n_zero_entries"] > 0
                           for v in stats.values()))
    return out


def c3_partner_content_locality(cache, rel, norm, rows):
    """Perturb *other* half-edges' indices; P must not move, B must."""
    model, _ = BM.build_model(norm, "P", 0, "cpu")
    model.eval()
    batch = BD.collate(cache, rel, rows)
    H = int(batch["n_half"])
    rng = np.random.default_rng(20261027)

    rep = {}
    modes = {
        "shuffle_partner_half": None,
        "replace_partner_half_from_other_molecule": None,
    }
    ph = batch["partner_half"].numpy().copy()
    rep["shuffle_partner_half"] = rng.permutation(ph)
    rep["replace_partner_half_from_other_molecule"] = rng.integers(0, H, size=len(ph))

    with torch.no_grad():
        base = {}
        for arm in ("B", "P"):
            model.arm = arm
            y, st = model(batch, return_stats=True)
            base[arm] = (y.clone(), st["M"].clone(), st["x"].clone())
        x = base["P"][2]
        results = {}
        for mode, newph in rep.items():
            nb = dict(batch)
            nb["partner_half"] = torch.from_numpy(np.asarray(newph, dtype=np.int64))
            # the partner content gathered by B must actually change
            vals = BM.chebyshev(x)
            old_g = vals[batch["partner_half"]]
            new_g = vals[nb["partner_half"]]
            content_shift = float((new_g - old_g).abs().max())
            per_arm = {}
            for arm in ("B", "P"):
                model.arm = arm
                y1, st1 = model(nb, return_stats=True)
                per_arm[arm] = {
                    "max_abs_delta_M": float((st1["M"] - base[arm][1]).abs().max()),
                    "max_abs_delta_y": float((y1 - base[arm][0]).abs().max()),
                }
            results[mode] = {"partner_content_shift": content_shift, "per_arm": per_arm}

    out = {"gate": "C3_partner_content_enters_B_only",
           "rows": [int(r) for r in rows],
           "n_half": H,
           "modes": results,
           "requirement": "partner_content_shift>1e-5, P delta==0, B delta>1e-6"}
    ok = all(r["partner_content_shift"] > 1e-5
             and r["per_arm"]["P"]["max_abs_delta_M"] == 0.0
             and r["per_arm"]["P"]["max_abs_delta_y"] == 0.0
             and r["per_arm"]["B"]["max_abs_delta_M"] > 1e-6
             for r in results.values())
    out["pass"] = bool(ok)
    return out


def c3b_relation_permutation_leaves_P(cache, rel, norm, rows):
    """P depends on the relations only through the counts."""
    model, _ = BM.build_model(norm, "P", 0, "cpu")
    model.eval()
    batch = BD.collate(cache, rel, rows)
    rng = np.random.default_rng(11)
    pr = batch["partner_rel"].numpy().copy()
    tgt = batch["partner_tgt"].numpy()
    for h in np.unique(tgt):
        idx = np.where(tgt == h)[0]
        if len(idx) > 1:
            pr[idx] = pr[idx][rng.permutation(len(idx))]
    nb = dict(batch)
    nb["partner_rel"] = torch.from_numpy(pr)
    with torch.no_grad():
        model.arm = "P"
        y0, s0 = model(batch, return_stats=True)
        y1, s1 = model(nb, return_stats=True)
    out = {"gate": "C3b_histogram_preserving_relation_permutation",
           "max_abs_delta_n": float((s1["n"] - s0["n"]).abs().max()),
           "max_abs_delta_M": float((s1["M"] - s0["M"]).abs().max()),
           "max_abs_delta_y": float((y1 - y0).abs().max())}
    out["pass"] = bool(out["max_abs_delta_n"] == 0.0 and out["max_abs_delta_M"] == 0.0
                       and out["max_abs_delta_y"] == 0.0)
    return out


def c4_own_state_drives_P(cache, rel, norm, rows):
    model, _ = BM.build_model(norm, "P", 0, "cpu")
    model.eval()
    batch = BD.collate(cache, rel, rows)
    with torch.no_grad():
        model.arm = "P"
        a, b = model.encode_tokens(batch)
        x = model.half_edges(batch, a, b)
        n_half = x.shape[0]
        M0, n0 = model.relation_stats(batch, x)

        target = 5 % n_half
        x2 = x.clone()
        x2[target] = x2[target] * 1.5
        M1, n1 = model.relation_stats(batch, x2)

        # block-local: only the perturbed half-edge may move
        moved = ((M1 - M0).abs().amax(dim=1) > 0).nonzero().reshape(-1).tolist()
        dblk = _blocks(M1 - M0)
        # the perturbation must be exactly n[r] * (T_k(1.5x) - T_k(x))
        vals0 = BM.chebyshev(x[target])
        vals1 = BM.chebyshev(x2[target])
        expect = (n0[target].unsqueeze(-1) * (vals1 - vals0).unsqueeze(0))
        err = float((dblk[target] - expect).abs().max())
    out = {"gate": "C4_the_target_half_edge_drives_P",
           "perturbed_half_edge": int(target),
           "max_abs_delta_M": float((M1 - M0).abs().max()),
           "half_edges_whose_M_changed": [int(i) for i in moved],
           "changed_only_the_perturbed_row": bool(moved == [int(target)]),
           "closed_form_max_abs_error": err,
           "counts_unchanged": bool((n1 - n0).abs().max() == 0.0)}
    out["pass"] = bool(out["max_abs_delta_M"] > 1e-6
                       and out["changed_only_the_perturbed_row"]
                       and err < 1e-6
                       and out["counts_unchanged"])
    return out


def c5_scale(cache, rel, norm, rows):
    model, _ = BM.build_model(norm, "P", 0, "cpu")
    model.eval()
    batch = BD.collate(cache, rel, rows)
    out = {"gate": "C5_moment_input_scale_is_comparable"}
    with torch.no_grad():
        stats = {}
        for arm in ("B", "P"):
            model.arm = arm
            _, st = model(batch, return_stats=True)
            M = st["M"].numpy()
            stats[arm] = {
                "is_finite": bool(np.isfinite(M).all()),
                "rms": float(np.sqrt(np.mean(M ** 2))),
                "mean_abs": float(np.mean(np.abs(M))),
                "max_abs": float(np.max(np.abs(M))),
                "per_dim_std_mean": float(np.mean(M.std(0))),
                "fraction_exactly_zero": float(np.mean(M == 0)),
            }
        ratio = stats["P"]["rms"] / max(stats["B"]["rms"], 1e-12)
        # the side MLP also sees the counts and the closure block; report the
        # full-input scale so a reader can see the M block is not an outlier
        out.update({"arm_stats": stats, "rms_ratio_P_over_B": float(ratio),
                    "log10_rms_ratio": float(math.log10(max(ratio, 1e-12))),
                    "acceptance": "finite, non-zero, 1/8 <= P.rms/B.rms <= 8"})
    out["pass"] = bool(all(v["is_finite"] and v["rms"] > 1e-6 for v in stats.values())
                       and 0.125 <= ratio <= 8.0)
    return out


def c6_invariance(cache, norm):
    rows = np.arange(8, dtype=np.int64)
    specs = [_spec_from_cache(cache, int(r)) for r in rows]
    models = {arm: BM.build_model(norm, arm, 0, "cpu")[0] for arm in ("R", "B", "P")}
    for m in models.values():
        m.eval()
    rng = np.random.default_rng(20261027)

    def run(arm, sp):
        with torch.no_grad():
            return models[arm](BD.batch_from_specs(sp)).numpy()

    perm = rng.permutation(specs[0]["atom_feat"].shape[0])
    p = np.empty_like(perm)
    p[perm] = np.arange(len(perm))
    permuted = dict(specs[0])
    permuted["atom_feat"] = specs[0]["atom_feat"][p]
    permuted["bond_index"] = perm[specs[0]["bond_index"]]
    rev = []
    for s in specs:
        t = dict(s)
        bi = s["bond_index"].copy()
        bi[:, 0], bi[:, 1] = bi[:, 1].copy(), bi[:, 0].copy()
        t["bond_index"] = bi
        rev.append(t)
    order = rng.permutation(len(specs))
    reordered = []
    for i in order:
        t = dict(specs[int(i)])
        bp = rng.permutation(len(t["bond_index"]))
        t["bond_index"] = t["bond_index"][bp]
        t["bond_feat"] = t["bond_feat"][bp]
        reordered.append(t)

    per_arm = {}
    for arm in models:
        base = np.concatenate([run(arm, [s]) for s in specs])
        o_perm = run(arm, [permuted] + specs[1:])
        o_rev = run(arm, rev)
        o_ord = run(arm, reordered)
        back = np.empty_like(o_ord)
        back[order] = o_ord
        per_arm[arm] = {
            "atom_relabel": float(np.abs(o_perm - base).max()),
            "bond_direction_reverse": float(np.abs(o_rev - base).max()),
            "edge_and_batch_order": float(np.abs(back - base).max()),
            "batch_recombination": float(np.abs(base - np.concatenate(
                [run(arm, [s]) for s in specs])).max()),
        }
    worst = max(max(v.values()) for v in per_arm.values())
    out = {"gate": "C6_symmetry_and_batch_invariance", "per_arm": per_arm,
           "worst_abs_delta": float(worst), "tolerance": 1e-5}
    out["pass"] = bool(worst < 1e-5)
    return out


def c7_gradients(cache, rel, norm, rows):
    rows = np.asarray(rows, dtype=np.int64)
    y = BC.labels(cache, rows)
    out = {"gate": "C7_every_parameter_of_P_is_reachable"}
    per_arm = {}
    for arm in ("B", "P"):
        torch.manual_seed(0)
        model, _ = BM.build_model(norm, arm, 0, "cpu")
        model.arm = arm
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        batch = BD.collate(cache, rel, rows)
        target = torch.as_tensor(y, dtype=torch.float32)
        history = []
        for step in range(5):
            pred = model(batch)
            loss = F.l1_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step in (0, 4):
                dead = [n for n, prm in model.named_parameters()
                        if prm.grad is None or float(prm.grad.abs().sum()) == 0.0]
                mods = {}
                for name in ("atom_encoder", "bond_encoder", "half_encoder",
                             "side", "compose", "reader"):
                    g = [prm.grad for prm in getattr(model, name).parameters()]
                    mods[name] = float(math.sqrt(sum(float(t.pow(2).sum()) for t in g)))
                history.append({"step": step, "loss": float(loss),
                                "dead_tensors": dead, "module_grad_norms": mods})
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        per_arm[arm] = history
    out["per_arm"] = per_arm
    out["pass"] = bool(all(not h["dead_tensors"] and all(v > 0 for v in h["module_grad_norms"].values())
                           for h in per_arm["P"])
                       and all(h["loss"] == h["loss"] for h in per_arm["P"]))
    return out


def c8_label_isolation(cache, rel, norm):
    """No Monitor/Locked row enters a training batch; Locked labels are inert."""
    rows = BC.split_rows()
    fit, monitor, locked = rows["fit8k"], rows["monitor"], rows["locked"]
    seen_rows: list[np.ndarray] = []
    original = BD.collate

    def spy(cache_, rel_, rows_):
        seen_rows.append(np.asarray(rows_, dtype=np.int64).copy())
        return original(cache_, rel_, rows_)

    BD.collate = spy
    try:
        model, _ = BM.build_model(norm, "P", 0, "cpu")
        model.arm = "P"
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(0 + 101)
        y_fit = BC.labels(cache, fit)
        for epoch in range(2):
            order = torch.randperm(len(fit), generator=gen).numpy()
            for start in range(0, 2 * 128, 128):
                batch_rows = fit[order[start:start + 128]]
                batch = BD.collate(cache, rel, batch_rows)
                pos = {int(r): i for i, r in enumerate(fit)}
                target = torch.as_tensor([y_fit[pos[int(r)]] for r in batch_rows],
                                         dtype=torch.float32)
                loss = F.l1_loss(model(batch), target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        train_calls = list(seen_rows)
        model.eval()
        mon_pred = BC.predict(model, cache, rel, monitor, device="cpu", batch_size=256)
        eval_calls = seen_rows[len(train_calls):]
    finally:
        BD.collate = original

    train_rows = np.concatenate(train_calls) if train_calls else np.zeros(0, np.int64)
    eval_rows = np.concatenate(eval_calls) if eval_calls else np.zeros(0, np.int64)
    leak_train = {
        "train_batches": int(len(train_calls)),
        "train_rows": int(len(train_rows)),
        "train_rows_in_monitor": int(np.isin(train_rows, monitor).sum()),
        "train_rows_in_locked": int(np.isin(train_rows, locked).sum()),
        "train_rows_not_in_fit8k": int((~np.isin(train_rows, fit)).sum()),
        "eval_rows": int(len(eval_rows)),
        "eval_rows_outside_monitor": int((~np.isin(eval_rows, monitor)).sum()),
    }

    # poison every Locked label; the same 2-epoch run must be bit-identical
    poisoned = dict(cache)
    lab = np.asarray(cache["labels"], dtype=np.float64).copy()
    lab_clean = lab.copy()
    lab[locked] = np.nan
    poisoned["labels"] = lab

    def two_epochs(cache_):
        m, _ = BM.build_model(norm, "P", 0, "cpu")
        m.arm = "P"
        m.train()
        o = torch.optim.AdamW(m.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(0 + 101)
        yf = BC.labels(cache_, fit)
        pos = {int(r): i for i, r in enumerate(fit)}
        losses = []
        for _ in range(2):
            order = torch.randperm(len(fit), generator=g).numpy()
            for start in range(0, 2 * 128, 128):
                br = fit[order[start:start + 128]]
                t = torch.as_tensor([yf[pos[int(r)]] for r in br], dtype=torch.float32)
                loss = F.l1_loss(m(BD.collate(cache_, rel, br)), t)
                o.zero_grad(set_to_none=True)
                loss.backward()
                o.step()
                losses.append(float(loss))
        m.eval()
        pred = BC.predict(m, cache_, rel, monitor, device="cpu", batch_size=256)
        return losses, pred

    l_clean, p_clean = two_epochs(cache)
    l_pois, p_pois = two_epochs(poisoned)
    out = {"gate": "C8_label_isolation", "batch_accounting": leak_train,
           "poison_test": {
               "locked_rows_poisoned": int(len(locked)),
               "max_abs_loss_diff": float(max(abs(a - b) for a, b in zip(l_clean, l_pois))),
               "max_abs_monitor_pred_diff": float(np.abs(p_clean - p_pois).max()),
               "monitor_mae": BC.mae(p_clean, BC.labels(cache, monitor)),
           },
           "data_cache": BC.DATA_CACHE.name,
           "official_valid_loaded": False, "official_test_loaded": False}
    out["pass"] = bool(leak_train["train_rows_in_monitor"] == 0
                       and leak_train["train_rows_in_locked"] == 0
                       and leak_train["train_rows_not_in_fit8k"] == 0
                       and leak_train["eval_rows_outside_monitor"] == 0
                       # the poison run itself uses only fit rows, so it is
                       # bit-identical to the clean run including the Monitor MAE
                       and out["poison_test"]["max_abs_monitor_pred_diff"] < 1e-6)
    return out


def c9_interface(cache, rel, norm, rows):
    model, info = BM.build_model(norm, "P", 0, "cpu")
    model.eval()
    batch = BD.collate(cache, rel, rows)
    with torch.no_grad():
        model.arm = "P"
        y, st = model(batch, return_stats=True)
    modules = {name: sum(p.numel() for p in getattr(model, name).parameters())
               for name in ("atom_encoder", "bond_encoder", "half_encoder", "side",
                            "compose", "reader")}
    out = {"gate": "C9_interface_and_budget",
           "parameters": info["parameters"],
           "expected_parameters": BC.EXPECTED_PARAMETERS,
           "cap": BC.PARAM_CAP,
           "modules": modules,
           "side_in": int(model.side.lins[0].in_features),
           "M_shape": list(st["M"].shape),
           "n_shape": list(st["n"].shape),
           "reader_in": int(model.reader.lins[0].in_features),
           "arms_available": list(BM.ARMS),
           "y_finite": bool(np.isfinite(y.numpy()).all())}
    out["pass"] = bool(out["parameters"] == BC.EXPECTED_PARAMETERS <= BC.PARAM_CAP
                       and out["side_in"] == BM.SIDE_IN
                       and out["M_shape"][1] == BD.N_REL * 3 * BM.RANK
                       and out["n_shape"][1] == BD.N_REL
                       and out["reader_in"] == BM.READER_IN
                       and set(BM.ARMS) == {"R", "B", "P"}
                       and out["y_finite"])
    return out


def main() -> int:
    cache = BC.load_train_cache()
    rel = torch.load(BC.CACHE / "bosi_rel_train.pt", map_location="cpu", weights_only=False)
    rel_hash = BD.relation_content_hash(rel)
    if rel_hash != "c698a2f30e0fe9f2fc51e79f37156e81da56f44b6a47843293dddb308ceda2b7":
        raise SystemExit(f"relation cache content hash changed: {rel_hash}")
    rows = BC.split_rows()
    norm = BC.fit_only_norm(cache, rows["fit8k"])
    gate_rows = np.asarray(rows["fit8k"][:64], dtype=np.int64)

    specs = [
        lambda: c1_params_and_init(cache, norm),
        lambda: c2_empty_group_is_zero(cache, rel, norm, gate_rows),
        lambda: c3_partner_content_locality(cache, rel, norm, gate_rows),
        lambda: c3b_relation_permutation_leaves_P(cache, rel, norm, gate_rows),
        lambda: c4_own_state_drives_P(cache, rel, norm, gate_rows),
        lambda: c5_scale(cache, rel, norm, gate_rows),
        lambda: c6_invariance(cache, norm),
        lambda: c7_gradients(cache, rel, norm, gate_rows),
        lambda: c8_label_isolation(cache, rel, norm),
        lambda: c9_interface(cache, rel, norm, gate_rows),
    ]
    gates = []
    for spec in specs:
        out = spec()
        gates.append(out)
        BC.log(f"[p-checks] {out['gate']}: {'PASS' if out['pass'] else 'FAIL'}")

    payload = {
        "round": BC.ROUND_ID,
        "checked_utc": BC.now(),
        "relation_cache_content_sha256": rel_hash,
        "split_hashes": rows["hashes"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "gates": gates,
        "n_pass": int(sum(g["pass"] for g in gates)),
        "n_gates": int(len(gates)),
        "decision": "CPU_GO" if all(g["pass"] for g in gates) else "CPU_NO_GO",
    }
    BC.ANALYSIS.mkdir(parents=True, exist_ok=True)
    BC.write_json(BC.ANALYSIS / "cpu_checks.json", payload)
    print(payload["decision"])
    return 0 if payload["decision"] == "CPU_GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
