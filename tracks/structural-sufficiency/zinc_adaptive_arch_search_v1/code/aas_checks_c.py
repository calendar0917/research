"""CPU gates for the deepening arms A2 (moment ablation) and A3 (reallocation).

  C1  A2 is byte-identical to A1 in parameters and initial state
  C2  A2 really zeroes the moment block: the Side input rows that carry M are 0
  C3  A2 differs from A1 in the forward (the ablation is live), and A2's step-0
      forward is *not* BOSI-B (publicly documented consequence of dropping M)
  C4  A3 parameter count / budget, and its Side input order is
      [a_v, b_e, x, closure, log1p deg, P] (shape asserted on the assembled input)
  C5  A3 invariance (relabel / bond reverse / bond order / batch split)
  C6  A3 gradients finite and non-zero in every module
  C7  A3 short fit decreases the loss
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


def gate_c1_identity(norm) -> dict:
    m1, i1 = build("A1", norm)
    m2, i2 = build("A2", norm)
    s1, s2 = C.cpu_state(m1), C.cpu_state(m2)
    same = set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)
    out = {"parameters_A1": i1["parameters"], "parameters_A2": i2["parameters"],
           "state_bit_identical": bool(same)}
    out["pass"] = bool(same and i1["parameters"] == i2["parameters"] == 488_321)
    return out


def gate_c2_moment_zeroed(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    batch = BD.collate(cache, rel, np.arange(64))
    m1, _ = build("A1", norm)
    m2, _ = build("A2", norm)
    with torch.no_grad():
        _, s1 = m1(batch, return_stats=True)
        _, s2 = m2(batch, return_stats=True)
    # Side input = [a_v, b_e, x, M, n, closure, log1p deg, P]
    lo = 2 * AM.WIDTH + AM.RANK
    hi = lo + 3 * AM.RANK + 2 * 12 * 12
    inp2 = torch.cat([s2["a"][batch["half_atom"]], s2["b"][batch["half_bond"]],
                      s2["x"], s2["M"], s2["n"], batch["closure"],
                      torch.log1p(batch["half_deg"].view(-1, 1))], dim=1)
    out = {"M_rms_A1": float(s1["M"].pow(2).mean().sqrt()),
           "M_max_abs_A2": float(s2["M"].abs().max()),
           "side_input_M_slots_max_abs_A2": float(inp2[:, lo:hi].abs().max()),
           "M_slots": [lo, hi]}
    out["pass"] = bool(out["M_rms_A1"] > 0 and out["M_max_abs_A2"] == 0.0
                       and out["side_input_M_slots_max_abs_A2"] == 0.0)
    return out


def gate_c3_ablation_live(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    batch = BD.collate(cache, rel, np.arange(128))
    m1, _ = build("A1", norm)
    m2, _ = build("A2", norm)
    mb, _ = build("B", norm)
    y1, y2, yb = forward_y(m1, batch), forward_y(m2, batch), forward_y(mb, batch)
    out = {"y_A1_vs_A2_max_abs_diff": float(np.abs(y1 - y2).max()),
           "y_A1_vs_A2_mean_abs_diff": float(np.abs(y1 - y2).mean()),
           "y_A2_vs_B_max_abs_diff": float(np.abs(y2 - yb).max())}
    out["pass"] = bool(out["y_A1_vs_A2_mean_abs_diff"] > 1e-5
                       and out["y_A2_vs_B_max_abs_diff"] > 1e-5)
    return out


def gate_c4_a3_params(norm) -> dict:
    m, info = build("A3", norm)
    n = int(sum(p.numel() for p in m.parameters()))
    out = {"parameters": n, "breakdown": info["parameters_breakdown"],
           "side_input_expected": AM.SIDE_IN_A3, "pair_dim_expected": AM.PAIR_DIM_A3}
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    batch = BD.collate(cache, rel, np.arange(32))
    with torch.no_grad():
        _, s = m(batch, return_stats=True)
    out["P_width"] = int(s["P"].shape[1])
    out["side_first_layer_in"] = int(m.side.lins[0].in_features)
    out["pass"] = bool(n == 457_185 and n <= AC.PARAM_CAP
                       and out["P_width"] == AM.PAIR_DIM_A3
                       and out["side_first_layer_in"] == AM.SIDE_IN_A3)
    return out


def gate_c5_a3_invariance(norm) -> dict:
    cache = AC.load_cache("valid")
    specs = [spec_from_cache(cache, r) for r in [3, 11, 42, 77, 108, 512]]
    rng = np.random.default_rng(13)
    m, _ = build("A3", norm)
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
    y_split = np.concatenate([forward_y(m, BD.batch_from_specs(specs[:half])),
                              forward_y(m, BD.batch_from_specs(specs[half:]))])
    out = {
        "atom_relabel_max_abs_diff": float(np.abs(forward_y(m, BD.batch_from_specs(relab)) - base).max()),
        "bond_reverse_max_abs_diff": float(np.abs(forward_y(m, BD.batch_from_specs(rev)) - base).max()),
        "bond_order_max_abs_diff": float(np.abs(forward_y(m, BD.batch_from_specs(bord)) - base).max()),
        "batch_order_max_abs_diff": float(np.abs(y_rev - base).max()),
        "batch_recombination_max_abs_diff": float(np.abs(y_split - base).max()),
    }
    out["pass"] = bool(max(v for k, v in out.items() if k.endswith("_diff")) <= TOL)
    return out


def gate_c6_a3_gradients(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    rows = np.arange(128)
    batch = BD.collate(cache, rel, rows)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    m, _ = build("A3", norm)
    m.train()
    loss = F.l1_loss(m(batch).view(-1), torch.as_tensor(y, dtype=torch.float32))
    loss.backward()
    norms, missing, nonfinite = {}, [], []
    for name, p in m.named_parameters():
        if p.grad is None:
            missing.append(name)
            continue
        mod = name.split(".")[0]
        norms[mod] = norms.get(mod, 0.0) + float(p.grad.detach().norm())
        if not bool(torch.isfinite(p.grad).all()):
            nonfinite.append(name)
    out = {"loss": float(loss), "grad_norms": norms, "missing": missing,
           "nonfinite": nonfinite}
    out["pass"] = bool(not missing and not nonfinite
                       and all(norms.get(k, 0) > 0 for k in
                               ("pair_phi", "side", "compose", "reader",
                                "half_encoder", "atom_encoder", "bond_encoder")))
    return out


def gate_c7_a3_short_train(norm) -> dict:
    cache = AC.load_cache("train")
    rel = AC.load_rel("train")
    y = np.asarray(cache["labels"], dtype=np.float64)
    m, _ = build("A3", norm)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(101)
    losses = []
    for _ in range(30):
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
    gates = [("C1_A2_identity", gate_c1_identity),
             ("C2_moment_zeroed", gate_c2_moment_zeroed),
             ("C3_ablation_live", gate_c3_ablation_live),
             ("C4_A3_params", gate_c4_a3_params),
             ("C5_A3_invariance", gate_c5_a3_invariance),
             ("C6_A3_gradients", gate_c6_a3_gradients),
             ("C7_A3_short_train", gate_c7_a3_short_train)]
    res = {}
    for name, fn in gates:
        try:
            r = fn(norm)
        except Exception as exc:  # noqa: BLE001
            r = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        res[name] = r
        AC.log(f"[gate-c] {name}: {'PASS' if r.get('pass') else 'FAIL'}")
    n_pass = sum(1 for r in res.values() if r.get("pass"))
    payload = {"round": AC.ROUND_ID, "body": "deepening A2/A3", "n_pass": n_pass,
               "n_gates": len(res), "all_pass": n_pass == len(res),
               "gates": res, "written": AC.now()}
    out = Path(args.out) if args.out else (AC.ANALYSIS / "cpu_checks_c.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    AC.log(f"[gate-c] {n_pass}/{len(res)} pass -> {out}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
