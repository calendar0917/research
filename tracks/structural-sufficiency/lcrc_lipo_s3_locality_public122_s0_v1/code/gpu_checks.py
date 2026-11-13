"""Small GPU-side checks that cannot be done on the CPU-only laptop.

1. the exact NULL3 mean field equals its naive double-loop reference on the
   real GPU kernels, in forward *and* in gradients;
2. the operator is not the ``psi(h, mean_u s3_u)`` shortcut;
3. a freshly built condition is an exact copy of C1 (zero correction) on real
   batches, and the frozen base initialisation hash is reproduced;
4. a single molecule and the same molecule inside a batch give the same value
   (no batch leakage), on the GPU.

Writes ``analysis/gpu_checks.json``. It trains nothing and never touches Test.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np
import torch

import common as C
import data as D
import features as F
import model as MD


def check_null_field(model, batch, device) -> dict[str, Any]:
    n_atoms = int(batch["atom_feat"].shape[0])
    gen = torch.Generator(device="cpu").manual_seed(4242)
    h = torch.randn(n_atoms, 64, generator=gen).to(device)
    s = batch["s3"].clone().to(device)
    # a non-zero output projection, otherwise everything is identically zero
    model.s3.W2.weight.data.normal_(0.0, 0.2, generator=torch.Generator().manual_seed(7))
    group = MD._group_key(batch)

    h1 = h.clone().requires_grad_(True)
    s1 = s.clone().requires_grad_(True)
    h2 = h.clone().requires_grad_(True)
    s2 = s.clone().requires_grad_(True)
    opt = MD.null_field(model.s3, h1, s1, group)
    naive = MD.null_field_naive(model.s3, h2, s2, group)

    params = [model.s3.W_h.weight, model.s3.W_s.weight, model.s3.b1, model.s3.W2.weight, model.s3.b2]
    g_opt = torch.autograd.grad(opt.sum(), params + [h1, s1], retain_graph=True, allow_unused=True)
    g_naive = torch.autograd.grad(naive.sum(), params + [h2, s2], retain_graph=True, allow_unused=True)

    # mean-then-SiLU shortcut, as a wrong reference
    wrong = torch.zeros_like(opt)
    for g in torch.unique(group).tolist():
        idx = torch.nonzero(group == g, as_tuple=False).view(-1)
        mean_s = s[idx].mean(0, keepdim=True).expand(idx.numel(), s.shape[1])
        wrong[idx] = model.s3(h[idx], mean_s)

    scale = float(opt.abs().max()) + 1e-12
    return {
        "n_atoms": n_atoms,
        "n_groups": int(torch.unique(group).numel()),
        "forward_max_abs_diff": float((opt - naive).abs().max()),
        "forward_relative_to_scale": float((opt - naive).abs().max()) / scale,
        "grad_max_abs_diff": float(max(float((a - b).abs().max()) for a, b in zip(g_opt, g_naive))),
        "mean_then_silu_max_abs_diff": float((opt - wrong).abs().max()),
        "operator_scale": scale,
    }


def check_zero_correction(dataset, feats, device) -> dict[str, Any]:
    cache = dataset["train_cache"]
    rows = list(range(8))
    f = F.split_features(feats, "train", cache)
    model, info = MD.build_model(dataset["spec"], dataset["train_norm"], device, "LOCAL3")
    base = C.DM.DissectLCRC(dataset["spec"], dataset["train_norm"], C.VARIANT)
    base.load_state_dict(MD.base_state_dict(model), strict=True)
    base.to(device).eval()
    model.eval()
    batch = D.to_device(D.collate(cache, rows, f), device)
    with torch.no_grad():
        diff = float((base(batch) - model(batch)).abs().max())
        # one molecule alone vs the same molecule inside the batch
        one = D.to_device(D.collate(cache, [rows[3]], f), device)
        single = model(one)
        inside = model(batch)[3]
    return {
        "zero_correction_max_abs_diff": diff,
        "init_state_hash": info["init_state_hash"],
        "init_state_hash_prefix_matches_baseline": info["init_state_hash"].startswith(
            C.BASELINE_EXPECTED["init_state_hash_prefix"]
        ),
        "parameters": info["parameters"],
        "single_vs_batch_abs_diff": float((single - inside).abs()),
    }


def run(device_name: str = "cuda") -> dict[str, Any]:
    device = torch.device(device_name)
    dataset = D.load_dataset(include_test=False)
    feats = F.load_features()
    f = F.split_features(feats, "train", dataset["train_cache"])

    model, info = MD.build_model(dataset["spec"], dataset["train_norm"], device, "NULL3")
    batch = D.to_device(D.collate(dataset["train_cache"], list(range(8)), f), device)
    with torch.no_grad():
        model.s3.W2.weight.data.zero_()
    null_check = check_null_field(model, batch, device)

    zero_check = check_zero_correction(dataset, feats, device)
    checks = {
        "null3_forward": bool(null_check["forward_relative_to_scale"] <= 1e-5),
        "null3_gradient": bool(null_check["grad_max_abs_diff"] <= 1e-3),
        "not_mean_then_silu": bool(null_check["mean_then_silu_max_abs_diff"] > 1e-3),
        "zero_correction_is_c1": bool(zero_check["zero_correction_max_abs_diff"] == 0.0),
        "no_batch_leakage": bool(zero_check["single_vs_batch_abs_diff"] <= 1e-5),
        "init_hash_reproduced": bool(zero_check["init_state_hash_prefix_matches_baseline"]),
    }
    out = {
        "schema": "lcrc-lipo-s3-locality-gpu-checks-v1",
        "round_id": C.ROUND_ID,
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "null3": null_check,
        "zero_correction": zero_check,
        "pass": {**checks, "all": all(checks.values())},
    }
    C.write_json(C.ANALYSIS / "gpu_checks.json", out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GPU-side operator/init checks")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    out = run(args.device)
    C.log(
        "[gpu-checks] "
        f"null3_forward={out['null3']['forward_max_abs_diff']:.2e} "
        f"null3_grad={out['null3']['grad_max_abs_diff']:.2e} "
        f"shortcut={out['null3']['mean_then_silu_max_abs_diff']:.3f} "
        f"zero_corr={out['zero_correction']['zero_correction_max_abs_diff']:.2e} "
        f"single_vs_batch={out['zero_correction']['single_vs_batch_abs_diff']:.2e} "
        f"pass={out['pass']['all']}"
    )
    return 0 if out["pass"]["all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
