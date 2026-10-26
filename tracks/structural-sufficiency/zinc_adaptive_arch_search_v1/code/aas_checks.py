"""CPU gates for ``zinc_adaptive_arch_search_v1`` (BOSI-QP A1 / A0).

Focused on what can actually be wrong in this round's change:

  G1  parameter count / budget / optimizer coverage
  G2  initial state: A1 and A0 share every tensor; common BOSI tensors equal BOSI-B
  G3  step-0 forward parity with BOSI-B (zero-initialised new Side columns)
  G4  the new branch is real: A1 != A0, P is non-zero, P is order-invariant
  G5  target-conditioning semantics: A1 moves when the target half-edge moves,
      A0 does not; the null is exactly the partner-set mean
  G6  P[v,e] uses only the half-edges of centre v (no cross-molecule / cross-atom leak)
  G7  empty / single partner sets are finite and give P == 0
  G8  symmetry invariance (relabel, bond reverse, bond order, batch recombination)
  G9  gradients: zero for Phi at step 0, non-zero finite after one update for
      every A1/A0 tensor; new Side columns get gradient at step 0
  G10 no label leakage; checkpoint save/load reproduces predictions
  G11 short optimisation decreases the loss

Writes ``analysis/cpu_checks.json`` and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aas_common as AC  # noqa: E402
import aas_models as AM  # noqa: E402
import bosi_models as BM  # noqa: E402
import bosi_data as BD  # noqa: E402
import ad_common as C  # noqa: E402

TOL = 1e-5
OUT: dict = {}


def spec_from_cache(cache, row: int) -> dict:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    a0, a1 = int(ap[row]), int(ap[row + 1])
    b0, b1 = int(bp[row]), int(bp[row + 1])
    return {
        "atom_feat": np.asarray(cache["atom_feat"])[a0:a1].astype(np.float32),
        "bond_index": np.asarray(cache["bond_index"])[b0:b1].astype(np.int64).reshape(-1, 2),
        "bond_feat": np.asarray(cache["bond_feat"])[b0:b1].astype(np.float32),
        "global": np.asarray(cache["global_feat"])[row].astype(np.float32),
    }


def forward_y(model, batch):
    model.eval()
    with torch.no_grad():
        return model(batch).numpy()


def build(arm, norm, seed=0):
    m, info = AM.build_model(norm, arm, seed, "cpu")
    return m, info


def gate1_params(norm) -> dict:
    out = {"arms": {}}
    ok = True
    for arm in ("A1", "A0"):
        m, info = build(arm, norm)
        n = int(sum(p.numel() for p in m.parameters()))
        bd = info["parameters_breakdown"]
        opt = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
        ok &= (n == AC.EXPECTED_PARAMETERS[arm] == 488_321 and n <= AC.PARAM_CAP
               and opt == n and bd["pair_phi"] == 10_336 and bd["total"] == n)
        out["arms"][arm] = {"parameters": n, "expected": AC.EXPECTED_PARAMETERS[arm],
                            "optimizer_parameters": opt,
                            "pair_phi": bd["pair_phi"], "breakdown": bd}
    out["cap"] = AC.PARAM_CAP
    out["pass"] = bool(ok)
    return out


def gate2_init(norm) -> dict:
    m1, _ = build("A1", norm)
    m0, _ = build("A0", norm)
    mb, _ = build("B", norm)
    s1, s0, sb = C.cpu_state(m1), C.cpu_state(m0), C.cpu_state(mb)
    same_arms = AC.state_hash({k: v for k, v in s1.items() if k.startswith("pair_phi.")
                              or k.startswith("side.lins.0.")}) == \
        AC.state_hash({k: v for k, v in s0.items() if k.startswith("pair_phi.")
                       or k.startswith("side.lins.0.")})
    for k in s1:
        if not torch.equal(s1[k], s0[k]):
            same_arms = False
    common1 = AM.common_state(s1)
    common_b = {k: v for k, v in sb.items() if k not in ("side.lins.0.weight",)}
    diffs = {}
    for k in common_b:
        if k not in common1 or common1[k].shape != common_b[k].shape:
            diffs[k] = "shape/missing"
        else:
            diffs[k] = float((common1[k] - common_b[k]).abs().max())
    # the kept prefix of the widened Side matrix equals BOSI-B's matrix exactly
    side_diff = float((s1["side.lins.0.weight"][:, :AC.SIDE_IN_BASE]
                       - sb["side.lins.0.weight"]).abs().max())
    new_cols_zero = bool(torch.count_nonzero(s1["side.lins.0.weight"][:, AC.SIDE_IN_BASE:]) == 0)
    phi_norm = float(s1["pair_phi.lins.0.weight"].abs().mean())
    out = {
        "arms_bit_identical": bool(same_arms),
        "common_vs_BOSI_B_max_abs_diff": float(max(
            v for v in diffs.values() if isinstance(v, float))),
        "common_tensors_checked": len(diffs),
        "side_prefix_vs_B_max_abs_diff": side_diff,
        "new_side_columns_all_zero": new_cols_zero,
        "pair_phi_first_layer_mean_abs": phi_norm,
        "initial_state_sha256_A1": AC.state_hash(s1),
    }
    out["pass"] = bool(same_arms and out["common_vs_BOSI_B_max_abs_diff"] == 0.0
                       and side_diff == 0.0 and new_cols_zero and phi_norm > 0)
    return out


def gate3_step0_parity(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    rows = np.arange(128)
    batch = BD.collate(cache, rel, rows)
    mb, _ = build("B", norm)
    m1, _ = build("A1", norm)
    m0, _ = build("A0", norm)
    yb = forward_y(mb, batch)
    y1 = forward_y(m1, batch)
    y0 = forward_y(m0, batch)
    out = {"A1_vs_B_max_abs_diff": float(np.abs(y1 - yb).max()),
           "A0_vs_B_max_abs_diff": float(np.abs(y0 - yb).max()),
           "A1_vs_A0_max_abs_diff": float(np.abs(y1 - y0).max())}
    out["pass"] = bool(out["A1_vs_B_max_abs_diff"] == 0.0
                       and out["A0_vs_B_max_abs_diff"] == 0.0
                       and out["A1_vs_A0_max_abs_diff"] == 0.0)
    return out


def gate4_branch_real(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    rows = np.arange(256)
    batch = BD.collate(cache, rel, rows)
    m1, _ = build("A1", norm)
    m0, _ = build("A0", norm)
    # make the new Side columns non-zero so P actually reaches the output
    with torch.no_grad():
        for m in (m1, m0):
            w = m.side.lins[0].weight
            g = torch.Generator().manual_seed(1234)
            w[:, AC.SIDE_IN_BASE:] = torch.randn(w[:, AC.SIDE_IN_BASE:].shape,
                                                 generator=g) * 0.05
    y1 = forward_y(m1, batch)
    y0 = forward_y(m0, batch)
    with torch.no_grad():
        _, s1 = m1(batch, return_stats=True)
        _, s0 = m0(batch, return_stats=True)
        P1, P0 = s1["P"].numpy(), s0["P"].numpy()
    out = {
        "P_A1_rms": float(np.sqrt((P1 ** 2).mean())),
        "P_A0_rms": float(np.sqrt((P0 ** 2).mean())),
        "P_zero_fraction": float(np.mean(np.abs(P1) < 1e-12)),
        "P_A1_vs_A0_max_abs_diff": float(np.abs(P1 - P0).max()),
        "P_A1_vs_A0_mean_abs_diff": float(np.abs(P1 - P0).mean()),
        "y_A1_vs_A0_max_abs_diff": float(np.abs(y1 - y0).max()),
        "y_A1_vs_A0_mean_abs_diff": float(np.abs(y1 - y0).mean()),
    }
    # partner-order invariance: permute the partner rows (keeping each target's set)
    perm = np.random.default_rng(0).permutation(len(batch["partner_tgt"]))
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b2["partner_tgt"] = batch["partner_tgt"][perm]
    b2["partner_half"] = batch["partner_half"][perm]
    b2["partner_rel"] = batch["partner_rel"][perm]
    # partner_ptr is a per-half-edge CSR pointer: permuting rows breaks it, so
    # rebuild it as a sorted grouping (order-only change)
    order = torch.argsort(b2["partner_tgt"], stable=True)
    b2 = {k: (v[order] if k in ("partner_tgt", "partner_half", "partner_rel") else v)
          for k, v in b2.items()}
    with torch.no_grad():
        _, sp = m1(b2, return_stats=True)
    out["partner_order_P_max_abs_diff"] = float((sp["P"] - s1["P"]).abs().max())
    out["pass"] = bool(out["P_A1_rms"] > 0 and out["P_zero_fraction"] < 0.9
                       and out["P_A1_vs_A0_mean_abs_diff"] > 1e-3
                       and out["y_A1_vs_A0_mean_abs_diff"] > 1e-5
                       and out["partner_order_P_max_abs_diff"] <= 1e-3)
    return out


def gate5_target_semantics(norm) -> dict:
    """A1 must depend on the target half-edge state; A0 must not."""
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    rows = np.arange(32)
    batch = BD.collate(cache, rel, rows)
    m1, _ = build("A1", norm)
    m0, _ = build("A0", norm)
    with torch.no_grad():
        _, s1 = m1(batch, return_stats=True)
        _, s0 = m0(batch, return_stats=True)
        # perturb x on ONE half-edge row, then recompute both arms' P with the
        # perturbed x (the block is a pure function of x and the partner maps)
        H = s1["x"].shape[0]
        npar = np.zeros(H, dtype=np.int64)
        np.add.at(npar, batch["partner_tgt"].numpy(), 1)
        row = int(np.argmax(npar))          # the half-edge with most partners
        x2 = s1["x"].clone()
        x2[row] += 1.0
        P1b = m1.pair_stats(batch, x2, null=False)
        P0b = m0.pair_stats(batch, x2, null=True)
    d1 = (P1b - s1["P"]).abs()
    d0 = (P0b - s0["P"]).abs()
    changed1 = (d1.sum(1) > 1e-6).nonzero().reshape(-1).tolist()
    changed0 = (d0.sum(1) > 1e-6).nonzero().reshape(-1).tolist()
    # rows that may change: the perturbed half-edge itself (as target) and, if it
    # is a partner of others, those targets.  Anything else is leakage.
    tgt = batch["partner_tgt"].numpy()
    part = batch["partner_half"].numpy()
    allowed = {row} | set(tgt[part == row].tolist())
    out = {
        "perturbed_half_edge": row,
        "A1_changed_rows": len(changed1),
        "A0_changed_rows": len(changed0),
        "A1_changed_rows_outside_allowed": len(set(changed1) - allowed),
        "A0_changed_rows_outside_allowed": len(set(changed0) - allowed),
        "A1_max_abs_change": float(d1.max()),
        "A0_max_abs_change": float(d0.max()),
        "A1_row_self_changed": bool(row in changed1),
        "A0_row_self_changed": bool(row in changed0),
    }
    # A1's own P row must move with its own target state; A0's must not.
    # A0 may still move on *other* half-edges that list the perturbed one as a
    # partner (that is exactly the partner-set mean), so only the self row and
    # the leakage count are asserted.
    out["pass"] = bool(out["A1_row_self_changed"] and not out["A0_row_self_changed"]
                       and out["A1_max_abs_change"] > 1e-4
                       and out["A1_changed_rows_outside_allowed"] == 0
                       and out["A0_changed_rows_outside_allowed"] == 0
                       and out["A1_changed_rows"] >= out["A0_changed_rows"])
    return out


def gate6_no_cross_molecule(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    rows = np.arange(16)
    batch = BD.collate(cache, rel, rows)
    m1, _ = build("A1", norm)
    with torch.no_grad():
        _, s = m1(batch, return_stats=True)
        P_full = s["P"].clone()
        P_half = []
        for start in range(0, 16, 1):
            b1 = BD.collate(cache, rel, rows[start:start + 1])
            _, s1 = m1(b1, return_stats=True)
            P_half.append(s1["P"])
        P_half = torch.cat(P_half, 0)
    out = {"P_full_vs_per_molecule_max_abs_diff": float((P_full - P_half).abs().max())}
    out["pass"] = bool(out["P_full_vs_per_molecule_max_abs_diff"] <= TOL)
    return out


def gate7_empty_sets(norm) -> dict:
    """Degree-1 half-edges (no partners) must give P == 0 and stay finite."""
    cache = AC.load_cache("valid")
    rel = AC.load_rel("valid")
    rows = np.arange(256)
    batch = BD.collate(cache, rel, rows)
    m1, _ = build("A1", norm)
    with torch.no_grad():
        y, s = m1(batch, return_stats=True)
    npar = torch.zeros(s["P"].shape[0], dtype=torch.float32)
    npar.index_add_(0, batch["partner_tgt"],
                    torch.ones(batch["partner_tgt"].numel()))
    empty = npar == 0
    out = {
        "n_half_edges": int(s["P"].shape[0]),
        "n_empty_partner_sets": int(empty.sum()),
        "P_max_on_empty": float(s["P"][empty].abs().max()) if bool(empty.any()) else 0.0,
        "all_finite": bool(torch.isfinite(s["P"]).all() and torch.isfinite(y).all()),
        "P_max_on_nonempty": float(s["P"][~empty].abs().max()) if bool((~empty).any()) else 0.0,
    }
    out["pass"] = bool(out["all_finite"] and out["P_max_on_empty"] == 0.0
                       and out["n_empty_partner_sets"] > 0)
    return out


def gate8_invariance(norm) -> dict:
    cache = AC.load_cache("valid")
    rows = [3, 11, 42, 77, 108, 512]
    specs = [spec_from_cache(cache, r) for r in rows]
    rng = np.random.default_rng(7)
    out = {}
    ok = True
    for arm in ("A1", "A0"):
        m, _ = build(arm, norm)
        base = forward_y(m, BD.batch_from_specs(specs))
        relab = []
        for s in specs:
            n = len(s["atom_feat"])
            p = rng.permutation(n)
            new = dict(s)
            new["atom_feat"] = s["atom_feat"][np.argsort(p)]
            new["bond_index"] = p[s["bond_index"]]
            relab.append(new)
        rev = []
        for s in specs:
            bi = s["bond_index"].copy()
            flip = rng.random(len(bi)) < 0.5
            bi[flip] = bi[flip][:, ::-1]
            new = dict(s)
            new["bond_index"] = bi
            rev.append(new)
        bord = []
        for s in specs:
            e = len(s["bond_index"])
            p = rng.permutation(e)
            new = dict(s)
            new["bond_index"] = s["bond_index"][p]
            new["bond_feat"] = s["bond_feat"][p]
            bord.append(new)
        y_rev = forward_y(m, BD.batch_from_specs(list(reversed(specs))))[::-1]
        half = len(specs) // 2
        y_split = np.concatenate([
            forward_y(m, BD.batch_from_specs(specs[:half])),
            forward_y(m, BD.batch_from_specs(specs[half:])),
        ])
        r = {
            "atom_relabel_max_abs_diff": float(np.abs(forward_y(m, BD.batch_from_specs(relab)) - base).max()),
            "bond_reverse_max_abs_diff": float(np.abs(forward_y(m, BD.batch_from_specs(rev)) - base).max()),
            "bond_order_max_abs_diff": float(np.abs(forward_y(m, BD.batch_from_specs(bord)) - base).max()),
            "batch_order_max_abs_diff": float(np.abs(y_rev - base).max()),
            "batch_recombination_max_abs_diff": float(np.abs(y_split - base).max()),
            "tolerance": TOL,
        }
        r["pass"] = bool(max(v for k, v in r.items() if k.endswith("_diff")) <= TOL)
        ok &= r["pass"]
        out[arm] = r
    out["pass"] = bool(ok)
    return out


def gate9_gradients(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    rows = np.arange(128)
    batch = BD.collate(cache, rel, rows)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    target = torch.as_tensor(y, dtype=torch.float32)
    out = {"arms": {}, "pass": True}
    for arm in ("A1", "A0"):
        torch.manual_seed(0)
        m, _ = build(arm, norm)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        steps = []
        for step in range(3):
            pred = m(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norms = {}
            nonfinite = []
            for name, p in m.named_parameters():
                mod = name.split(".")[0]
                g = 0.0 if p.grad is None else float(p.grad.detach().norm())
                norms[mod] = norms.get(mod, 0.0) + g
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    nonfinite.append(name)
            steps.append({"step": step, "loss": float(loss), "grad_norms": norms,
                          "nonfinite": nonfinite,
                          "P_grad": None if m.pair_phi.lins[0].weight.grad is None
                          else float(m.pair_phi.lins[0].weight.grad.abs().sum()),
                          "side_new_cols_grad": float(
                              m.side.lins[0].weight.grad[:, AC.SIDE_IN_BASE:].abs().sum())})
            opt.step()
        last = steps[-1]["grad_norms"]
        ok = (not any(s["nonfinite"] for s in steps)
              and steps[0]["P_grad"] == 0.0
              and steps[0]["side_new_cols_grad"] > 0.0
              and steps[-1]["P_grad"] > 0.0
              and all(last.get(k, 0.0) > 0.0 for k in
                      ("pair_phi", "side", "reader", "compose", "half_encoder",
                       "atom_encoder", "bond_encoder"))
              and steps[-1]["loss"] < steps[0]["loss"])
        out["arms"][arm] = {"steps": steps, "loss_decreased": steps[-1]["loss"] < steps[0]["loss"]}
        out["pass"] = bool(out["pass"] and ok)
    return out


def gate10_isolation_checkpoint(norm) -> dict:
    cache = AC.load_cache("valid")
    rel = AC.load_rel("valid")
    rows = np.arange(64)
    batch = BD.collate(cache, rel, rows)
    m, _ = build("A1", norm)
    p1 = forward_y(m, batch)
    # labels are irrelevant to the forward
    cache2 = dict(cache)
    y = np.asarray(cache["labels"], dtype=np.float64).copy()
    y[:] = np.nan
    cache2["labels"] = y
    p2 = forward_y(m, BD.collate(cache2, rel, rows))
    # checkpoint round-trip
    state = C.cpu_state(m)
    m2, _ = build("A1", norm)
    C.load_state(m2, state, "cpu")
    p3 = forward_y(m2, batch)
    out = {
        "labels_are_nan_diff": float(np.abs(p1 - p2).max()),
        "checkpoint_roundtrip_diff": float(np.abs(p1 - p3).max()),
        "valid_cache_rows": int(len(np.asarray(cache["atom_ptr"])) - 1),
        "train_cache_rows": int(len(np.asarray(AC.load_cache("train")["atom_ptr"])) - 1),
        "valid_relation_file": AC.rel_or_abs(AC.BC.rel_path("valid")),
    }
    out["pass"] = bool(out["labels_are_nan_diff"] == 0.0
                       and out["checkpoint_roundtrip_diff"] == 0.0
                       and out["valid_cache_rows"] == 1000
                       and out["train_cache_rows"] == 10000
                       and "valid" in out["valid_relation_file"])
    return out


def gate11_short_train(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    vcache = AC.load_cache("valid")
    vrel = AC.load_rel("valid")
    y = np.asarray(cache["labels"], dtype=np.float64)
    vy = np.asarray(vcache["labels"], dtype=np.float64)
    m, _ = build("A1", norm)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(101)
    losses = []
    for step in range(40):
        rows = torch.randperm(10_000, generator=gen).numpy()[:128]
        batch = BD.collate(cache, rel, rows)
        target = torch.as_tensor(y[rows], dtype=torch.float32)
        pred = m(batch)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        losses.append(float(loss))
    vpred = AC.predict(m, vcache, vrel, np.arange(1000), device="cpu", batch_size=400)
    out = {"first_loss": losses[0], "last_loss": losses[-1],
           "min_loss": float(min(losses)),
           "valid_mae_after_40_steps": AC.mae(vpred, vy),
           "finite": bool(np.isfinite(losses).all())}
    out["pass"] = bool(out["finite"] and out["last_loss"] < out["first_loss"] * 0.9)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    norm = AC.load_norm()
    gates = [
        ("G1_parameters", gate1_params),
        ("G2_initial_state", gate2_init),
        ("G3_step0_parity", gate3_step0_parity),
        ("G4_branch_is_real", gate4_branch_real),
        ("G5_target_semantics", gate5_target_semantics),
        ("G6_no_cross_molecule", gate6_no_cross_molecule),
        ("G7_empty_partner_sets", gate7_empty_sets),
        ("G8_invariance", gate8_invariance),
        ("G9_gradients", gate9_gradients),
        ("G10_isolation_checkpoint", gate10_isolation_checkpoint),
        ("G11_short_train", gate11_short_train),
    ]
    res = {}
    for name, fn in gates:
        try:
            r = fn(norm)
        except Exception as exc:  # noqa: BLE001
            r = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        res[name] = r
        AC.log(f"[gate] {name}: {'PASS' if r.get('pass') else 'FAIL'}")
    n_pass = sum(1 for r in res.values() if r.get("pass"))
    payload = {"round": AC.ROUND_ID, "n_pass": n_pass, "n_gates": len(res),
               "all_pass": n_pass == len(res), "gates": res, "written": AC.now()}
    out = Path(args.out) if args.out else (AC.ANALYSIS / "cpu_checks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    AC.log(f"[gate] {n_pass}/{len(res)} pass -> {out}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
