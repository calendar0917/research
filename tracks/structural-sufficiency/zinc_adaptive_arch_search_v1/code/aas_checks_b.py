"""CPU gates for the SQBC body (direction B): S1 (ordered) vs S0 (symmetric).

  B1  parameter shapes/counts identical and <= cap; S1 and S0 initial state is
      bit-identical (the two arms differ *only* in the forward)
  B2  the two arms produce different predictions
  B3  role swap: S1 changes, S0 is bit-identical (the asymmetry gate)
  B4  empty partner sets give P == 0; everything finite
  B5  symmetry invariance (relabel / bond reverse / bond order / batch split)
  B6  gradients finite and non-zero for every module
  B7  a 40-step fit decreases the loss and produces finite valid predictions
  B8  labels never enter the forward; checkpoint round-trip is exact
  B9  no cross-molecule leak: per-molecule forward == batched forward
  B10 scale report: P, c, z activation RMS (no degenerate zero block)
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
import bosi_data as BD  # noqa: E402
import ad_common as C  # noqa: E402
from aas_checks import forward_y, spec_from_cache  # noqa: E402

TOL = 1e-5


def build(arm, norm, seed=0):
    return AM.build_model(norm, arm, seed, "cpu")


def gate_b1_params(norm) -> dict:
    out = {"arms": {}}
    ok = True
    m1, i1 = build("S1", norm)
    m0, i0 = build("S0", norm)
    s1, s0 = C.cpu_state(m1), C.cpu_state(m0)
    same = set(s1) == set(s0) and all(torch.equal(s1[k], s0[k]) for k in s1)
    for arm, info, m in (("S1", i1, m1), ("S0", i0, m0)):
        n = int(sum(p.numel() for p in m.parameters()))
        opt = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
        out["arms"][arm] = {"parameters": n, "optimizer_parameters": opt,
                            "breakdown": info["parameters_breakdown"]}
        ok &= (n == 381_505 and n <= AC.PARAM_CAP and opt == n)
    out["shapes_identical"] = bool(
        all(tuple(s1[k].shape) == tuple(s0[k].shape) for k in s1))
    out["initial_state_bit_identical"] = bool(same)
    out["initial_state_sha256"] = C.state_hash(m1)
    out["pass"] = bool(ok and out["shapes_identical"] and same)
    return out


def gate_b2_arms_differ(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    batch = BD.collate(cache, rel, np.arange(256))
    m1, _ = build("S1", norm)
    m0, _ = build("S0", norm)
    y1, y0 = forward_y(m1, batch), forward_y(m0, batch)
    out = {"y_S1_vs_S0_max_abs_diff": float(np.abs(y1 - y0).max()),
           "y_S1_vs_S0_mean_abs_diff": float(np.abs(y1 - y0).mean())}
    out["pass"] = bool(out["y_S1_vs_S0_mean_abs_diff"] > 1e-4)
    return out


def gate_b3_role_swap(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    batch = BD.collate(cache, rel, np.arange(128))
    m1, _ = build("S1", norm)
    m0, _ = build("S0", norm)
    with torch.no_grad():
        _, s1 = m1(batch, return_stats=True)
        _, s0 = m0(batch, return_stats=True)
        m1._swap_roles = True
        m0._swap_roles = True
        _, s1s = m1(batch, return_stats=True)
        _, s0s = m0(batch, return_stats=True)
        m1._swap_roles = False
        m0._swap_roles = False
    out = {
        "S1_P_swap_max_abs_diff": float((s1s["P"] - s1["P"]).abs().max()),
        "S0_P_swap_max_abs_diff": float((s0s["P"] - s0["P"]).abs().max()),
        "S0_c_swap_max_abs_diff": float((s0s["c"] - s0["c"]).abs().max()),
    }
    out["pass"] = bool(out["S1_P_swap_max_abs_diff"] > 1e-3
                       and out["S0_P_swap_max_abs_diff"] == 0.0
                       and out["S0_c_swap_max_abs_diff"] == 0.0)
    return out


def gate_b4_empty(norm) -> dict:
    cache = AC.load_cache("valid")
    rel = AC.load_rel("valid")
    batch = BD.collate(cache, rel, np.arange(256))
    m, _ = build("S1", norm)
    with torch.no_grad():
        y, s = m(batch, return_stats=True)
    npar = np.zeros(s["P"].shape[0], dtype=np.int64)
    np.add.at(npar, batch["partner_tgt"].numpy(), 1)
    empty = torch.as_tensor(npar == 0)
    out = {"n_half_edges": int(s["P"].shape[0]),
           "n_empty": int(empty.sum()),
           "P_max_on_empty": float(s["P"][empty].abs().max()) if bool(empty.any()) else 0.0,
           "P_max_on_nonempty": float(s["P"][~empty].abs().max()) if bool((~empty).any()) else 0.0,
           "all_finite": bool(torch.isfinite(s["P"]).all() and torch.isfinite(y).all())}
    out["pass"] = bool(out["all_finite"] and out["P_max_on_empty"] == 0.0
                       and out["n_empty"] > 0)
    return out


def gate_b5_invariance(norm) -> dict:
    cache = AC.load_cache("valid")
    specs = [spec_from_cache(cache, r) for r in [3, 11, 42, 77, 108, 512]]
    rng = np.random.default_rng(11)
    out = {}
    ok = True
    for arm in ("S1", "S0", "S2"):
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
            forward_y(m, BD.batch_from_specs(specs[half:]))])
        r = {
            "atom_relabel_max_abs_diff": float(np.abs(forward_y(m, BD.batch_from_specs(relab)) - base).max()),
            "bond_reverse_max_abs_diff": float(np.abs(forward_y(m, BD.batch_from_specs(rev)) - base).max()),
            "bond_order_max_abs_diff": float(np.abs(forward_y(m, BD.batch_from_specs(bord)) - base).max()),
            "batch_order_max_abs_diff": float(np.abs(y_rev - base).max()),
            "batch_recombination_max_abs_diff": float(np.abs(y_split - base).max()),
        }
        r["pass"] = bool(max(v for k, v in r.items() if k.endswith("_diff")) <= TOL)
        ok &= r["pass"]
        out[arm] = r
    out["pass"] = bool(ok)
    return out


def gate_b6_gradients(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    rows = np.arange(128)
    batch = BD.collate(cache, rel, rows)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    target = torch.as_tensor(y, dtype=torch.float32)
    out = {"arms": {}, "pass": True}
    for arm in ("S1", "S0", "S2"):
        m, _ = build(arm, norm)
        m.train()
        pred = m(batch)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        loss.backward()
        norms, nonfinite, missing = {}, [], []
        for name, p in m.named_parameters():
            if p.grad is None:
                missing.append(name)
                continue
            mod = name.split(".")[0]
            norms[mod] = norms.get(mod, 0.0) + float(p.grad.detach().norm())
            if not bool(torch.isfinite(p.grad).all()):
                nonfinite.append(name)
        ok = (not missing and not nonfinite and norms.get("pair_phi", 0) > 0
              and norms.get("compose", 0) > 0 and norms.get("reader", 0) > 0)
        out["arms"][arm] = {"loss": float(loss), "grad_norms": norms,
                            "missing_grads": missing, "nonfinite": nonfinite}
        out["pass"] = bool(out["pass"] and ok)
    return out


def gate_b7_short_train(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    vcache = AC.load_cache("valid")
    vrel = AC.load_rel("valid")
    y = np.asarray(cache["labels"], dtype=np.float64)
    vy = np.asarray(vcache["labels"], dtype=np.float64)
    m, _ = build("S1", norm)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(101)
    losses = []
    for _ in range(40):
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
           "valid_mae_after_40_steps": AC.mae(vpred, vy),
           "finite": bool(np.isfinite(losses).all())}
    out["pass"] = bool(out["finite"] and out["last_loss"] < out["first_loss"] * 0.9)
    return out


def gate_b8_isolation(norm) -> dict:
    cache = AC.load_cache("valid")
    rel = AC.load_rel("valid")
    rows = np.arange(64)
    batch = BD.collate(cache, rel, rows)
    m, _ = build("S1", norm)
    p1 = forward_y(m, batch)
    cache2 = dict(cache)
    yy = np.asarray(cache["labels"], dtype=np.float64).copy()
    yy[:] = np.nan
    cache2["labels"] = yy
    p2 = forward_y(m, BD.collate(cache2, rel, rows))
    state = C.cpu_state(m)
    m2, _ = build("S1", norm)
    C.load_state(m2, state, "cpu")
    p3 = forward_y(m2, batch)
    out = {"labels_nan_max_abs_diff": float(np.abs(p1 - p2).max()),
           "checkpoint_roundtrip_max_abs_diff": float(np.abs(p1 - p3).max()),
           "params_after_roundtrip_grad": int(sum(
               p.numel() for p in m2.parameters() if p.requires_grad))}
    out["pass"] = bool(out["labels_nan_max_abs_diff"] == 0.0
                       and out["checkpoint_roundtrip_max_abs_diff"] == 0.0)
    return out


def gate_b9_no_cross_molecule(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    rows = np.arange(16)
    m, _ = build("S1", norm)
    p_full = forward_y(m, BD.collate(cache, rel, rows))
    p_one = np.concatenate([forward_y(m, BD.collate(cache, rel, rows[i:i + 1]))
                            for i in range(len(rows))])
    out = {"full_vs_per_molecule_max_abs_diff": float(np.abs(p_full - p_one).max())}
    out["pass"] = bool(out["full_vs_per_molecule_max_abs_diff"] <= TOL)
    return out


def gate_b10_scale(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    m, _ = build("S1", norm)
    with torch.no_grad():
        y, s = m(BD.collate(cache, rel, np.arange(256)), return_stats=True)
    out = {"P_rms": float(s["P"].pow(2).mean().sqrt()),
           "c_rms": float(s["c"].pow(2).mean().sqrt()),
           "z_rms": float(s["z"].pow(2).mean().sqrt()),
           "P_zero_fraction": float((s["P"].abs() < 1e-12).float().mean()),
           "n_junc_mean": float(s["n_junc"].mean())}
    out["pass"] = bool(out["P_rms"] > 1e-3 and out["c_rms"] > 1e-3
                       and out["P_zero_fraction"] < 0.2)
    return out


def gate_b11_s2_matched(norm) -> dict:
    """S2 (global pooling of the identical pair content) is parameter-identical
    to S1 and shares its initial state; only the pooling location differs."""
    m1, i1 = build("S1", norm)
    m2, i2 = build("S2", norm)
    s1, s2 = C.cpu_state(m1), C.cpu_state(m2)
    same = set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    batch = BD.collate(cache, rel, np.arange(256))
    y1, y2 = forward_y(m1, batch), forward_y(m2, batch)
    out = {"parameters_S1": i1["parameters"], "parameters_S2": i2["parameters"],
           "state_bit_identical": bool(same),
           "y_S1_vs_S2_mean_abs_diff": float(np.abs(y1 - y2).mean()),
           "y_S1_vs_S2_max_abs_diff": float(np.abs(y1 - y2).max())}
    out["pass"] = bool(same and i1["parameters"] == i2["parameters"]
                       and out["y_S1_vs_S2_mean_abs_diff"] > 1e-4)
    return out


def gate_b12_s2_short_train(norm) -> dict:
    return gate_b7_short_train(norm, arm="S2") if False else _short_train(norm, "S2")


def _short_train(norm, arm: str) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    y = np.asarray(cache["labels"], dtype=np.float64)
    m, _ = build(arm, norm)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(101)
    losses = []
    for _ in range(40):
        rows = torch.randperm(10_000, generator=gen).numpy()[:128]
        loss = F.l1_loss(m(BD.collate(cache, rel, rows)).view(-1),
                         torch.as_tensor(y[rows], dtype=torch.float32))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        losses.append(float(loss))
    out = {"first_loss": losses[0], "last_loss": losses[-1]}
    out["pass"] = bool(np.isfinite(losses).all() and losses[-1] < losses[0] * 0.9)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    norm = AC.load_norm()
    gates = [("B1_parameters", gate_b1_params), ("B2_arms_differ", gate_b2_arms_differ),
             ("B3_role_swap", gate_b3_role_swap), ("B4_empty_partner_sets", gate_b4_empty),
             ("B5_invariance", gate_b5_invariance), ("B6_gradients", gate_b6_gradients),
             ("B7_short_train", gate_b7_short_train), ("B8_isolation", gate_b8_isolation),
             ("B9_no_cross_molecule", gate_b9_no_cross_molecule),
             ("B10_scale", gate_b10_scale),
             ("B11_S2_matched", gate_b11_s2_matched),
             ("B12_S2_short_train", gate_b12_s2_short_train)]
    res = {}
    for name, fn in gates:
        try:
            r = fn(norm)
        except Exception as exc:  # noqa: BLE001
            r = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        res[name] = r
        AC.log(f"[gate-b] {name}: {'PASS' if r.get('pass') else 'FAIL'}")
    n_pass = sum(1 for r in res.values() if r.get("pass"))
    payload = {"round": AC.ROUND_ID, "body": "SQBC", "n_pass": n_pass,
               "n_gates": len(res), "all_pass": n_pass == len(res),
               "gates": res, "written": AC.now()}
    out = Path(args.out) if args.out else (AC.ANALYSIS / "cpu_checks_b.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    AC.log(f"[gate-b] {n_pass}/{len(res)} pass -> {out}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
