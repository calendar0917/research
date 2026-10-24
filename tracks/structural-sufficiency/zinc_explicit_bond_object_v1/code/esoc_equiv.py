"""ESOC-v1 equivalence check for the declared performance-only optimisation.

The formal run uses ``torch.compile(mode="reduce-overhead")`` (static shapes,
fused kernels, CUDA graphs).  That is legal only if it does not change the
mathematics, so this script compares, from a **bit-identical initial state** on
**identical batches**:

  1. one full forward/backward from the initial state: loss, every gradient
     tensor, and the post-step parameters (this is the actual maths);
  2. eval-mode predictions on a fixed valid batch;
  3. the *drift* over 10 steps, reported descriptively: compiled kernels
     reassociate fp32 reductions, and the first training steps of this model are
     chaotic (loss moves 1.5 -> 3.6 within 4 steps), so an exact multi-step
     trajectory match is not the right standard.  The formal run uses only one
     code path, so what must be shown is that the code path computes the same
     function, not that two different reduction orders stay in lock-step;
  4. the stochastic path: with dropout/stochastic depth on, the compiled model is
     still stochastic (CUDA graphs own the RNG stream, so the *realisation* is
     not the eager one - only the distribution matters, reported as a mean gap);
  5. structural invariants of the refactor: the frozen bond-slot width and the
     fixed RRWP atom bound do not change any value, fused ``F.rms_norm`` equals
     the explicit formula, construction is seed-deterministic, and REAL/NULL
     still start from bit-identical tensors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
for _p in (str(_HERE), str(_SS / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import esoc_data as D  # noqa: E402
import esoc_models as M  # noqa: E402
import esoc_train as T  # noqa: E402

ANALYSIS = _HERE.parent / "analysis"


def _base(model):
    """Unwrap torch.compile so state dicts keep their plain parameter names."""
    return getattr(model, "_orig_mod", model)


def _freeze_stochastic(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.drop_path = 0.0


def _one_step(net, base, opt, batch, lr):
    for g in opt.param_groups:
        g["lr"] = lr
    p = net(batch)
    loss = (p - batch["y"]).abs().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(base.parameters(), 5.0))
    grads = {n: prm.grad.detach().clone() for n, prm in base.named_parameters()
             if prm.grad is not None}
    opt.step()
    params = {n: prm.detach().clone() for n, prm in base.named_parameters()}
    opt.zero_grad(set_to_none=True)
    return float(loss.detach()), gn, grads, params, p.detach().clone()


def _eval_pred(net, valid_data, rows):
    was = net.training
    net.eval()
    with torch.no_grad():
        out = net(valid_data.batch(rows)).double().cpu().numpy()
    net.train(was)
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arm", default="real")
    ap.add_argument("--drift-steps", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    scales = T.read_json(ANALYSIS / "scales.json")
    train_data = D.SplitTensors(D.load_cache("train"), args.device)
    valid_data = D.SplitTensors(D.load_cache("valid"), args.device)
    order = T.batch_order(0, 1, train_data.n, 32)
    batches = [order[i * 32:(i + 1) * 32] for i in range(args.drift_steps)]
    valid_rows = np.arange(64)
    lr0 = T.schedule_fn(780000, 15600, 2e-3, 2e-5)(0)

    out: dict = {"device": args.device, "arm": args.arm, "torch": torch.__version__,
                 "compile_mode": "reduce-overhead", "drift_steps": args.drift_steps}

    # ---------------- eager reference -------------------------------------
    ref = M.build_model(0, args.arm, scales).to(args.device)
    _freeze_stochastic(ref)
    ref_hash0 = C.state_hash(ref)
    opt_ref = T.make_optimizer(ref, 2e-3, 1e-5)
    loss_r, gn_r, grad_r, par_r, pred_r = _one_step(ref, ref, opt_ref, train_data.batch(batches[0]), lr0)
    vpred_r = _eval_pred(ref, valid_data, valid_rows)

    # ---------------- compiled candidate (same init) -----------------------
    cand = M.build_model(0, args.arm, scales).to(args.device)
    _freeze_stochastic(cand)
    assert C.state_hash(cand) == ref_hash0, "initial state differs between arms"
    net_c = torch.compile(cand, mode="reduce-overhead", dynamic=False)
    opt_c = T.make_optimizer(cand, 2e-3, 1e-5)
    loss_c, gn_c, grad_c, par_c, pred_c = _one_step(net_c, cand, opt_c,
                                                    train_data.batch(batches[0]), lr0)
    vpred_c = _eval_pred(net_c, valid_data, valid_rows)

    out["single_step_loss_eager"] = loss_r
    out["single_step_loss_compiled"] = loss_c
    out["single_step_loss_abs_diff"] = abs(loss_r - loss_c)
    out["single_step_grad_norm_abs_diff"] = abs(gn_r - gn_c)
    out["train_pred_max_abs_diff"] = float((pred_r - pred_c).abs().max())
    out["valid_pred_max_abs_diff"] = float(np.abs(vpred_r - vpred_c).max())
    out["post_step_param_max_abs_diff"] = float(max(
        (par_r[n] - par_c[n]).abs().max() for n in par_r))
    # Per-tensor gradient agreement.  A *relative* criterion is meaningless for
    # tensors whose gradient is ~0 (e.g. the zero-initialised AdaRMSNorm
    # projections before they open), so we report both the absolute difference
    # and the relative difference with a denominator floor, plus the worst
    # tensor by name and scale.
    gscale = max(float(grad_r[n].double().abs().max()) for n in grad_r)
    rows = []
    for n in grad_r:
        a, b = grad_r[n].double(), grad_c[n].double()
        amax = float(a.abs().max())
        rows.append({
            "name": n,
            "grad_abs_max": amax,
            "abs_diff_max": float((a - b).abs().max()),
            "rel_diff": float((a - b).abs().max() / max(amax, 1e-3 * gscale)),
        })
    rows.sort(key=lambda r: r["abs_diff_max"], reverse=True)
    out["grad_tensors_compared"] = len(rows)
    out["grad_scale_max"] = gscale
    out["grad_abs_diff_max"] = rows[0]["abs_diff_max"]
    out["grad_abs_diff_max_tensor"] = rows[0]["name"]
    out["grad_abs_diff_max_tensor_grad"] = rows[0]["grad_abs_max"]
    out["grad_rel_diff_max_floored"] = float(max(r["rel_diff"] for r in rows))
    out["grad_abs_diff_worst5"] = rows[:5]

    # ---------------- drift over several steps (descriptive) ---------------
    drift = []
    for i in range(1, args.drift_steps):
        b = train_data.batch(batches[i])
        lr = T.schedule_fn(780000, 15600, 2e-3, 2e-5)(i)
        l_r, _, _, p_r, _ = _one_step(ref, ref, opt_ref, b, lr)
        l_c, _, _, p_c, _ = _one_step(net_c, cand, opt_c, b, lr)
        drift.append({
            "step": i,
            "loss_eager": l_r,
            "loss_compiled": l_c,
            "loss_abs_diff": abs(l_r - l_c),
            "param_max_abs_diff": float(max((p_r[n] - p_c[n]).abs().max() for n in p_r)),
        })
    out["drift"] = drift
    out["drift_loss_max_abs_diff"] = float(max(d["loss_abs_diff"] for d in drift))
    out["drift_param_max_abs_diff"] = float(max(d["param_max_abs_diff"] for d in drift))
    out["drift_note"] = (
        "compiled kernels reassociate fp32 reductions; the first steps of this model "
        "are chaotic (loss 1.5 -> 3.6 within 4 steps), so a multi-step trajectory is "
        "expected to decorrelate. The formal run uses a single code path."
    )

    # ---------------- stochastic path -------------------------------------
    stoch = {}
    for tag, build in (
        ("eager", lambda: M.build_model(0, args.arm, scales).to(args.device)),
        ("compiled", lambda: torch.compile(M.build_model(0, args.arm, scales).to(args.device),
                                           mode="reduce-overhead", dynamic=False)),
    ):
        m = build()
        base = _base(m)
        opt = T.make_optimizer(base, 2e-3, 1e-5)
        b = train_data.batch(batches[0])
        vals = []
        for r in range(6):
            torch.manual_seed(7000 + r)
            p = m(b)
            loss = (p - b["y"]).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            vals.append(float(loss.detach()))
        stoch[tag] = vals
    out["stochastic_losses"] = stoch
    out["stochastic_eager_is_stochastic"] = len(set(stoch["eager"])) > 1
    out["stochastic_compiled_is_stochastic"] = len(set(stoch["compiled"])) > 1
    out["stochastic_mean_abs_gap"] = float(abs(np.mean(stoch["eager"]) - np.mean(stoch["compiled"])))

    # ---------------- structural invariants -------------------------------
    b8 = train_data.batch(np.arange(8))
    r_var = M.rrwp(b8["bonds"], b8["bond_mask"], b8["n_atoms"]).detach().cpu().numpy()
    r_fix = M.rrwp(b8["bonds"], b8["bond_mask"], b8["n_atoms"],
                   n_max=int(scales["max_atoms"])).detach().cpu().numpy()
    out["rrwp_static_shape_max_abs_diff"] = float(np.abs(r_var - r_fix).max())

    x = torch.randn(5, 7, 64, dtype=torch.float64, device=args.device)
    fused = F.rms_norm(x, (64,), None, 1e-6).detach().cpu().numpy()
    manual = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).detach().cpu().numpy()
    out["rms_norm_fused_max_abs_diff"] = float(np.abs(fused - manual).max())

    fresh = M.build_model(0, args.arm, scales).to("cpu")
    out["construction_seed_deterministic"] = bool(C.state_hash(fresh) == ref_hash0)

    m_null = M.build_model(0, "null", scales).to("cpu")
    s_real = {k: v.detach().cpu() for k, v in fresh.state_dict().items()}
    s_null = {k: v.detach().cpu() for k, v in m_null.state_dict().items()}
    out["real_null_init_state_keys_identical"] = sorted(s_real) == sorted(s_null)
    out["real_null_init_max_abs_diff"] = float(max(
        (s_real[k] - s_null[k]).abs().max() if s_real[k].is_floating_point()
        else float((s_real[k] != s_null[k]).any()) for k in s_real))

    # ---------------- verdict ---------------------------------------------
    out["thresholds"] = {
        "single_step_loss_abs_diff < 1e-4": True,
        "train_pred_max_abs_diff < 1e-4": True,
        "valid_pred_max_abs_diff < 1e-4": True,
        "grad_abs_diff_max / grad_scale < 1e-5": True,
        "post_step_param_max_abs_diff < 1e-5": True,
        "rrwp_static_shape_max_abs_diff == 0": True,
        "rms_norm_fused_max_abs_diff == 0": True,
        "real_null_init_max_abs_diff == 0": True,
        "compiled_still_stochastic": True,
    }
    ok = (out["single_step_loss_abs_diff"] < 1e-4
          and out["train_pred_max_abs_diff"] < 1e-4
          and out["valid_pred_max_abs_diff"] < 1e-4
          and out["grad_abs_diff_max"] / out["grad_scale_max"] < 1e-5
          and out["post_step_param_max_abs_diff"] < 1e-5
          and out["rrwp_static_shape_max_abs_diff"] == 0.0
          and out["rms_norm_fused_max_abs_diff"] == 0.0
          and out["construction_seed_deterministic"]
          and out["real_null_init_state_keys_identical"]
          and out["real_null_init_max_abs_diff"] == 0.0
          and out["stochastic_eager_is_stochastic"]
          and out["stochastic_compiled_is_stochastic"])
    out["pass"] = bool(ok)
    out["environment"] = C.environment()

    dest = Path(args.out) if args.out else (ANALYSIS / "raw" / "compile_equivalence.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(dest, out)
    print(json.dumps({k: v for k, v in out.items() if k != "drift"}, indent=2))
    print(json.dumps(out["drift"], indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
