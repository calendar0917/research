"""ESOC-v1 equivalence check for the declared performance-only optimisation.

The formal run uses ``torch.compile`` (static shapes + fused kernels + CUDA
graphs).  That is allowed only if it does not change the mathematics, so we
compare, from **bit-identical initial state** and on **identical batches and
identical RNG streams**:

  eager reference  vs  compiled candidate

  * every training-step loss,
  * eval-mode predictions on a fixed valid batch,
  * gradient norms and a fixed set of per-tensor gradients after the last step,
  * the state dict before/after the whole trajectory.

Also verifies that the static-shape refactor (frozen bond-slot width 41, fixed
RRWP atom bound) does not change the mathematics relative to per-batch sizing,
and that the fused ``F.rms_norm`` matches the explicit RMS formula.
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
STEPS = 30


def _base(model):
    """Unwrap torch.compile so state dicts keep their plain parameter names."""
    return getattr(model, "_orig_mod", model)


def _trajectory(net, train_data, valid_data, seed: int, arm: str, steps: int,
                batches: list[np.ndarray], valid_rows: np.ndarray, base=None) -> dict:
    base = base if base is not None else _base(net)
    opt = T.make_optimizer(base, 2e-3, 1e-5)
    losses = []
    for i, rows in enumerate(batches[:steps]):
        b = train_data.batch(rows)
        lr = T.schedule_fn(STEPS * 100, 5, 2e-3, 2e-5)(i)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.manual_seed(1000 + i)              # same dropout/drop-path stream
        p = net(b)
        loss = (p - b["y"]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(base.parameters(), 5.0))
        opt.step()
        losses.append((float(loss), gn))
    grads = {}
    for name, prm in base.named_parameters():
        if prm.grad is not None:
            grads[name] = prm.grad.detach().clone()
    opt.zero_grad(set_to_none=True)
    net.eval()
    with torch.no_grad():
        vp = net(valid_data.batch(valid_rows)).double().cpu().numpy()
    net.train()
    return {"losses": losses, "grads": grads, "valid_pred": vp,
            "state": {k: v.detach().clone() for k, v in base.state_dict().items()}}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arm", default="real")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    scales = T.read_json(ANALYSIS / "scales.json")
    train_data = D.SplitTensors(D.load_cache("train"), args.device)
    valid_data = D.SplitTensors(D.load_cache("valid"), args.device)
    order = T.batch_order(0, 1, train_data.n, 32)
    batches = [order[i * 32:(i + 1) * 32] for i in range(args.steps)]
    valid_rows = np.arange(64)

    out: dict = {"device": args.device, "arm": args.arm, "steps": args.steps,
                 "torch": torch.__version__}

    # --- 1) eager reference (per-batch RRWP atom bound, i.e. the pre-optimisation path)
    ref = M.build_model(0, args.arm, scales).to(args.device)
    ref_hash0 = C.state_hash(ref)
    r = _trajectory(ref, train_data, valid_data, 0, args.arm, args.steps,
                    batches, valid_rows)

    # --- 2) compiled candidate from the identical initial state
    cand = M.build_model(0, args.arm, scales).to(args.device)
    assert C.state_hash(cand) == ref_hash0, "initial state differs"
    compiled = torch.compile(cand, mode="reduce-overhead", dynamic=False)
    c = _trajectory(compiled, train_data, valid_data, 0, args.arm, args.steps,
                    batches, valid_rows, base=cand)

    out["loss_max_abs_diff"] = float(max(abs(a[0] - b[0]) for a, b in zip(r["losses"], c["losses"])))
    out["loss_rel_max"] = float(max(abs(a[0] - b[0]) / max(a[0], 1e-12)
                                    for a, b in zip(r["losses"], c["losses"])))
    out["grad_norm_max_abs_diff"] = float(max(abs(a[1] - b[1])
                                              for a, b in zip(r["losses"], c["losses"])))
    out["valid_pred_max_abs_diff"] = float(np.abs(r["valid_pred"] - c["valid_pred"]).max())
    shared = sorted(set(r["state"]) & set(c["state"]))
    out["state_keys_compared"] = len(shared)
    out["final_param_max_abs_diff"] = float(max(
        float((r["state"][k] - c["state"][k]).abs().max())
        for k in shared if r["state"][k].is_floating_point()) or 0.0)
    out["n_params_compared"] = len(r["grads"])
    rel = []
    for name in r["grads"]:
        a, b = r["grads"][name].double(), c["grads"][name].double()
        rel.append(float((a - b).abs().max() / max(float(a.abs().max()), 1e-12)))
    out["grad_tensor_max_rel_diff"] = float(max(rel))

    # --- 3) static shapes must not change the maths: per-batch vs fixed atom bound
    spec_rows = np.arange(8)
    b8 = train_data.batch(spec_rows)
    r_var = M.rrwp(b8["bonds"], b8["bond_mask"], b8["n_atoms"]).detach().cpu().numpy()
    r_fix = M.rrwp(b8["bonds"], b8["bond_mask"], b8["n_atoms"],
                   n_max=int(scales["max_atoms"])).detach().cpu().numpy()
    out["rrwp_static_shape_max_abs_diff"] = float(np.abs(r_var - r_fix).max())

    # --- 4) fused rms_norm vs the explicit formula
    x = torch.randn(5, 7, 64, dtype=torch.float64, device=args.device)
    fused = F.rms_norm(x, (64,), None, 1e-6).cpu().numpy()
    manual = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).cpu().numpy()
    out["rms_norm_fused_max_abs_diff"] = float(np.abs(fused - manual).max())

    # --- 5) determinism: two eager runs of the same seed agree exactly
    ref2 = M.build_model(0, args.arm, scales).to(args.device)
    assert C.state_hash(ref2) == ref_hash0, "construction is not seed-deterministic"
    r2 = _trajectory(ref2, train_data, valid_data, 0, args.arm, args.steps,
                     batches, valid_rows)
    out["eager_replay_loss_max_abs_diff"] = float(max(abs(a[0] - b[0])
                                                     for a, b in zip(r["losses"], r2["losses"])))
    out["eager_replay_valid_pred_max_abs_diff"] = float(
        np.abs(r["valid_pred"] - r2["valid_pred"]).max())

    # --- 6) REAL and NULL must still start from the identical tensor set
    m_null = M.build_model(0, "null", scales).to("cpu")
    keys_same = sorted(m_null.state_dict()) == sorted(ref2.state_dict())
    diff = 0.0
    for k, v in m_null.state_dict().items():
        w = ref2.state_dict()[k]
        diff = max(diff, float((v - w).abs().max()) if v.is_floating_point()
                   else float((v != w).any()))
    out["real_null_init_state_keys_identical"] = bool(keys_same)
    out["real_null_init_max_abs_diff"] = float(diff)

    ok = (out["loss_max_abs_diff"] < 1e-5
          and out["valid_pred_max_abs_diff"] < 1e-4
          and out["final_param_max_abs_diff"] < 1e-4
          and out["grad_tensor_max_rel_diff"] < 1e-3
          and out["rrwp_static_shape_max_abs_diff"] == 0.0
          and out["rms_norm_fused_max_abs_diff"] < 1e-12
          and out["eager_replay_loss_max_abs_diff"] == 0.0
          and out["eager_replay_valid_pred_max_abs_diff"] == 0.0
          and out["real_null_init_state_keys_identical"]
          and out["real_null_init_max_abs_diff"] == 0.0)
    out["pass"] = bool(ok)
    out["environment"] = C.environment()

    dest = Path(args.out) if args.out else (ANALYSIS / "raw" / "compile_equivalence.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(dest, {k: v for k, v in out.items()})
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
