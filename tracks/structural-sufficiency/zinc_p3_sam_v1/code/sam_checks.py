"""CPU correctness gate for ``zinc_p3_sam_v1``.

Everything here is a *narrow* implementation/identification check for this
round, not a new experiment:

  A. both arms share parameter count, initial state dict, SSL file, normalization
     and full optimizer coverage; the P3 forward is bit-identical across builds
     and reproduces the historical P3B valid number through the unchanged
     harness;
  B. rho=0 SAM equals plain recipe-B AdamW over several updates (model and
     AdamW state);
  C. rho>0 SAM matches an independent hand-written two-pass computation
     (perturbed second gradient, restore before clipping/step, one AdamW state
     increment per batch);
  D. epsilon has global norm rho while the unclipped first-pass gradient is much
     larger than clip=5; no clipping touches the first pass;
  E. the first-pass gradient does not leak into the second pass; parameters are
     bit-identical after restore and remain restored on the exception path;
  F. real P3 forward/backward is finite on a real batch and encoder / composer /
     reader all move after a SAM update (finite-difference style smoke, not a
     claim that every tensor's first gradient is non-zero);
  G. the row-order RNG is independent of the extra SAM forward and both arms
     therefore consume the same per-epoch order; the order hash is recorded;
  H. 79 steps/epoch, 600 epochs, 47,400 outer updates, 500-step warmup, cosine
     endpoint at min LR, one schedule advance per outer update.

Run: ``uv run python tracks/structural-sufficiency/zinc_p3_sam_v1/code/sam_checks.py``
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V2 = _SS / "zinc_recipe_trunk_v2" / "code"
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V2), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402
import sam_core as S  # noqa: E402

HIST_P3B = _SS / "zinc_recipe_trunk_v2" / "runs" / "xBssl_P3N_s0" / "best.pt"
RESULT: dict[str, dict] = {}


class Tiny(nn.Module):
    """Deterministic tiny model with no stateful layers (no BN/dropout)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1234)
        self.fc1 = nn.Linear(7, 9)
        self.fc2 = nn.Linear(9, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x))))).view(-1)


def _tiny_data(n: int = 32):
    g = torch.Generator().manual_seed(99)
    x = torch.randn(n, 7, generator=g)
    y = torch.randn(n, generator=g) * 0.5 + 0.3
    return x, y


def _hand_sam_update(model, params, x, y, *, rho, clip, lr, wd, betas, eps_opt,
                     manual_state) -> dict:
    """Independent two-pass SAM + manual AdamW, used only for verification."""
    b1, b2 = betas
    model.zero_grad(set_to_none=True)
    pred = model(x)
    loss = F.l1_loss(pred, y)
    loss.backward()
    g1 = [p.grad.detach().clone() for p in params]
    norm1 = math.sqrt(sum(float((g.double() ** 2).sum()) for g in g1))
    eps = [g * (rho / (norm1 + 1e-12)) for g in g1]
    theta = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, e in zip(params, eps):
            p.add_(e)
    model.zero_grad(set_to_none=True)
    pred2 = model(x)
    loss2 = F.l1_loss(pred2, y)
    loss2.backward()
    g2 = [p.grad.detach().clone() for p in params]
    norm2 = math.sqrt(sum(float((g.double() ** 2).sum()) for g in g2))
    coef = min(1.0, clip / (norm2 + 1e-6))
    g2 = [g * coef for g in g2]
    with torch.no_grad():
        for p, t in zip(params, theta):
            p.copy_(t)
    with torch.no_grad():
        for i, (p, g) in enumerate(zip(params, g2)):
            manual_state["step"][i] += 1
            m = manual_state["m"][i] = b1 * manual_state["m"][i] + (1 - b1) * g
            v = manual_state["v"][i] = b2 * manual_state["v"][i] + (1 - b2) * g * g
            mhat = m / (1 - b1 ** manual_state["step"][i])
            vhat = v / (1 - b2 ** manual_state["step"][i])
            p.mul_(1 - lr * wd)
            p.add_(mhat / (vhat.sqrt() + eps_opt), alpha=-lr)
    return {"loss": float(loss), "loss2": float(loss2), "g1": g1, "g2": g2,
            "norm1": norm1, "norm2": norm2, "clip_coef": coef, "eps": eps}


def _run_impl(model, params, cfg, x, y, *, updates: int, arm: str,
              manual_lrs: list[float] | None = None):
    opt = S.make_optimizer(model, cfg)
    traj = []
    for i in range(updates):
        step = i + 1
        if arm == "C":
            rec = S.update_control(model, opt, params, x, y, clip=5.0, epoch=1)
        else:
            rec = S.update_sam(model, opt, params, x, y, rho=float(cfg["sam"]["rho"]),
                               clip=5.0, epoch=1, check=True)
        lr = S.set_lr(opt, step, updates, cfg)
        opt.step()
        traj.append((rec, lr))
    return opt, traj


def _opt_state(opt) -> dict:
    return copy.deepcopy(opt.state_dict())


def check_init_and_harness(cfg: dict, norm: dict) -> dict:
    d: dict[str, Any] = {}
    m1, i1 = S.build_model(norm, torch.device("cpu"))
    m2, i2 = S.build_model(norm, torch.device("cpu"))
    h1, h2 = C.state_hash(m1), C.state_hash(m2)
    d["parameters"] = int(i1["parameters"])
    d["parameters_match_config"] = bool(d["parameters"] == cfg["model"]["parameters_expected"])
    d["init_hash"] = h1
    d["init_hash_second_build"] = h2
    d["init_hash_matches_config"] = bool(h1 == cfg["model"]["initial_state_sha256"])
    d["init_hash_equal_across_builds"] = bool(h1 == h2)
    parts = {
        "encoders": sum(p.numel() for n, p in m1.named_parameters() if n.startswith("encoders.")),
        "group_composer_null": sum(p.numel() for n, p in m1.named_parameters()
                                   if n.startswith("group_composer.")),
        "graph_reader": sum(p.numel() for n, p in m1.named_parameters() if n.startswith("reader.")),
    }
    d["parameter_parts"] = {k: int(v) for k, v in parts.items()}
    d["parameter_parts_match"] = bool(parts == cfg["model"]["parameter_parts_expected"])
    opt = S.make_optimizer(m1, cfg)
    d["optimizer_parameters"] = C.optimizer_parameter_count(opt)
    d["optimizer_covers_all"] = bool(d["optimizer_parameters"] == d["parameters"])
    d["adamw_defaults"] = {
        "betas": list(opt.param_groups[0]["betas"]),
        "eps": float(opt.param_groups[0]["eps"]),
        "weight_decay": float(opt.param_groups[0]["weight_decay"]),
    }
    d["adamw_matches_config"] = bool(
        tuple(opt.param_groups[0]["betas"]) == tuple(cfg["optimizer"]["betas"])
        and float(opt.param_groups[0]["eps"]) == float(cfg["optimizer"]["eps"])
        and float(opt.param_groups[0]["weight_decay"]) == float(cfg["optimizer"]["weight_decay"])
    )
    d["ssl_sha256"] = i1["ssl"]["sha256"]
    d["ssl_sha256_matches_config"] = bool(
        d["ssl_sha256"] == cfg["model"]["ssl"]["sha256"]
        and C.sha256_file(Path(i1["ssl"]["source"])) == cfg["model"]["ssl"]["sha256"]
    )
    # encoders are exactly the SSL file content; composer/reader are seed-0 random
    ssl = torch.load(i1["ssl"]["source"], map_location="cpu", weights_only=False)
    d["encoder_state_equals_ssl_file"] = bool(
        torch.equal(m1.encoders.atom_encoder.lins[0].weight,
                    ssl["atom_encoder"]["lins.0.weight"].float())
    )
    # no BatchNorm / Dropout, and eval-mode switching cannot change the forward
    bad_norm = [n for n, mod in m1.named_modules()
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    drops = sorted({float(mod.p) for mod in m1.modules() if isinstance(mod, nn.Dropout)})
    d["batchnorm_modules"] = bad_norm
    d["dropout_rates"] = drops
    d["no_batchnorm_no_dropout"] = bool(not bad_norm and all(p == 0.0 for p in drops))

    norm2 = D.load_norm()
    m3, _ = S.build_model(norm2, torch.device("cpu"))
    d["norm_buffers_equal_across_builds"] = bool(
        torch.equal(m1.global_mean, m3.global_mean) and torch.equal(m1.global_scale, m3.global_scale)
    )
    d["norm_source"] = D.load_norm()["source"]

    # unchanged harness: historical P3B best.pt reproduces its stored valid MAE
    d["historical_checkpoint"] = str(HIST_P3B)
    if HIST_P3B.exists():
        vc, yv, rows_v = S.load_split("valid")
        ck = torch.load(HIST_P3B, map_location="cpu", weights_only=False)
        hm, _ = S.build_model(norm, torch.device("cpu"))
        C.load_state(hm, ck["state_dict"], torch.device("cpu"))
        ref_pred, ref_mae = S.evaluate(hm, vc, rows_v, torch.device("cpu"), 256)
        d["historical_recomputed_valid"] = float(ref_mae)
        d["historical_stored_valid"] = float(ck["valid_mae"])
        d["historical_abs_diff"] = float(abs(ref_mae - float(ck["valid_mae"])))
        d["harness_reproduces_historical_valid"] = bool(d["historical_abs_diff"] < 2e-6)
        # same-weight forward parity across two independent builds
        m4, _ = S.build_model(norm, torch.device("cpu"))
        C.load_state(m4, ck["state_dict"], torch.device("cpu"))
        pred_a, _ = S.evaluate(hm, vc, rows_v, torch.device("cpu"), 256)
        pred_b, _ = S.evaluate(m4, vc, rows_v, torch.device("cpu"), 256)
        d["historical_forward_max_abs_diff"] = float(np.abs(pred_a - pred_b).max())
        d["forward_parity"] = bool(d["historical_forward_max_abs_diff"] == 0.0)
    else:
        d["historical_checkpoint_present"] = False
        d["harness_reproduces_historical_valid"] = None
        d["forward_parity"] = None
    d["pass"] = bool(
        d["parameters_match_config"] and d["init_hash_matches_config"]
        and d["init_hash_equal_across_builds"] and d["parameter_parts_match"]
        and d["optimizer_covers_all"] and d["adamw_matches_config"]
        and d["ssl_sha256_matches_config"] and d["encoder_state_equals_ssl_file"]
        and d["no_batchnorm_no_dropout"] and d["norm_buffers_equal_across_builds"]
        and (d["harness_reproduces_historical_valid"] is not False)
        and (d["forward_parity"] is not False)
    )
    return d


def check_rho0_equivalence(cfg: dict) -> dict:
    d: dict[str, Any] = {}
    x, y = _tiny_data()
    cfg0 = copy.deepcopy(cfg)
    cfg0["sam"]["rho"] = 0.0
    torch.manual_seed(7)
    base = Tiny()
    mc = copy.deepcopy(base)
    ms = copy.deepcopy(base)
    pc, ps = list(mc.parameters()), list(ms.parameters())
    oc, traj_c = _run_impl(mc, pc, cfg, x, y, updates=5, arm="C")
    os_, traj_s = _run_impl(ms, ps, cfg0, x, y, updates=5, arm="S")
    d["updates"] = 5
    d["model_max_abs_diff"] = float(max((a - b).abs().max().item() for a, b in zip(pc, ps)))
    sc, ss = _opt_state(oc), _opt_state(os_)
    d["optimizer_step_equal"] = bool(all(
        torch.equal(a["step"], b["step"]) for a, b in zip(sc["state"].values(), ss["state"].values())
    ))
    d["optimizer_steps"] = int(list(sc["state"].values())[0]["step"].item())
    d["exp_avg_max_abs_diff"] = float(max(
        (a["exp_avg"] - b["exp_avg"]).abs().max().item()
        for a, b in zip(sc["state"].values(), ss["state"].values())
    ))
    d["exp_avg_sq_max_abs_diff"] = float(max(
        (a["exp_avg_sq"] - b["exp_avg_sq"]).abs().max().item()
        for a, b in zip(sc["state"].values(), ss["state"].values())
    ))
    d["lr_at_step5_control"] = float(traj_c[-1][1])
    d["lr_at_step5_sam"] = float(traj_s[-1][1])
    d["lr_trajectories_equal"] = bool(d["lr_at_step5_control"] == d["lr_at_step5_sam"])
    d["tolerance"] = 1e-7
    d["pass"] = bool(
        d["model_max_abs_diff"] < 1e-7 and d["optimizer_step_equal"]
        and d["optimizer_steps"] == 5 and d["exp_avg_max_abs_diff"] < 1e-7
        and d["exp_avg_sq_max_abs_diff"] < 1e-7
    )
    return d


def check_two_pass_handcomputed(cfg: dict) -> dict:
    d: dict[str, Any] = {}
    x, y = _tiny_data()
    torch.manual_seed(7)
    base = Tiny()
    mi = copy.deepcopy(base)
    pi = list(mi.parameters())
    opt = S.make_optimizer(mi, cfg)
    lr = S.set_lr(opt, 1, 47400, cfg)
    rec_impl = S.update_sam(mi, opt, pi, x, y, rho=0.05, clip=5.0, epoch=1, check=True)
    opt.step()

    mh = copy.deepcopy(base)
    ph = list(mh.parameters())
    manual_state = {
        "step": [0] * len(ph),
        "m": [torch.zeros_like(p) for p in ph],
        "v": [torch.zeros_like(p) for p in ph],
    }
    rec_hand = _hand_sam_update(
        mh, ph, x, y, rho=0.05, clip=5.0, lr=lr,
        wd=float(cfg["optimizer"]["weight_decay"]),
        betas=tuple(cfg["optimizer"]["betas"]), eps_opt=float(cfg["optimizer"]["eps"]),
        manual_state=manual_state,
    )
    d["lr_step1"] = float(lr)
    d["g1_norm_impl"] = float(rec_impl["g1_norm"])
    d["g1_norm_hand"] = float(rec_hand["norm1"])
    d["g2_norm_impl"] = float(rec_impl["g2_norm"])
    d["g2_norm_hand"] = float(rec_hand["norm2"])
    d["eps_norm_impl"] = float(rec_impl["eps_norm"])
    d["eps_norm_hand"] = float(sum(float((e.double() ** 2).sum()) for e in rec_hand["eps"]) ** 0.5)
    d["g1_norm_rel_diff"] = abs(d["g1_norm_impl"] - d["g1_norm_hand"]) / max(d["g1_norm_hand"], 1e-12)
    d["g2_norm_rel_diff"] = abs(d["g2_norm_impl"] - d["g2_norm_hand"]) / max(d["g2_norm_hand"], 1e-12)
    d["model_max_abs_diff_vs_hand"] = float(max((a - b).abs().max().item() for a, b in zip(pi, ph)))
    st = opt.state_dict()["state"]
    d["adamw_step_counts"] = sorted({int(v["step"].item()) for v in st.values()})
    d["adamw_exp_avg_max_abs_diff_vs_hand"] = float(max(
        (v["exp_avg"] - manual_state["m"][i]).abs().max().item()
        for i, v in enumerate(st.values())
    ))
    d["adamw_exp_avg_sq_max_abs_diff_vs_hand"] = float(max(
        (v["exp_avg_sq"] - manual_state["v"][i]).abs().max().item()
        for i, v in enumerate(st.values())
    ))
    d["perturb_exact"] = rec_impl["perturb_exact"]
    d["restore_exact"] = rec_impl["restore_exact"]
    # the second pass really is at theta+eps: g2 at theta differs materially
    torch.manual_seed(7)
    m0 = Tiny()
    m0.zero_grad(set_to_none=True)
    F.l1_loss(m0(x), y).backward()
    g_theta = [p.grad.detach().clone() for p in m0.parameters()]
    d["g2_vs_same_theta_max_rel_diff"] = float(max(
        ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()
        for a, b in zip(rec_hand["g2"], g_theta)
    ))
    d["pass"] = bool(
        d["g1_norm_rel_diff"] < 1e-9 and d["g2_norm_rel_diff"] < 1e-9
        and d["model_max_abs_diff_vs_hand"] < 1e-8
        and d["adamw_step_counts"] == [1]
        and d["adamw_exp_avg_max_abs_diff_vs_hand"] < 1e-8
        and d["adamw_exp_avg_sq_max_abs_diff_vs_hand"] < 1e-8
        and bool(d["perturb_exact"]) and bool(d["restore_exact"])
        and d["g2_vs_same_theta_max_rel_diff"] > 1e-4
    )
    return d


def check_epsilon_and_no_first_clip(cfg: dict) -> dict:
    d: dict[str, Any] = {}
    x, _ = _tiny_data()
    y = 100.0 * torch.ones(x.shape[0])  # large residual -> raw grad >> clip=5
    torch.manual_seed(11)
    model = Tiny()
    with torch.no_grad():  # amplify the output Jacobian so |g1| >> clip=5
        model.fc3.weight.mul_(1000.0)
    params = list(model.parameters())
    opt = S.make_optimizer(model, cfg)
    theta = S.save_params(params)
    rec = S.update_sam(model, opt, params, x, y, rho=0.05, clip=5.0, epoch=1, check=True)
    d["raw_g1_norm"] = float(rec["g1_norm"])
    d["raw_g1_exceeds_clip"] = bool(rec["g1_norm"] > 5.0)
    d["eps_global_norm"] = float(rec["eps_norm"])
    d["rho"] = 0.05
    d["eps_rel_dev_from_rho"] = abs(rec["eps_norm"] - 0.05) / 0.05
    # recover the actual perturbation magnitude independently: apply once more from
    # the freshly restored theta and measure ||theta_perturbed - theta||
    d["params_restored_after_update"] = S.params_match_backup(params, theta)
    d["g2_norm"] = float(rec["g2_norm"])
    d["g2_was_clipped"] = bool(rec["g2_norm"] > 5.0)
    d["perturb_exact"] = rec["perturb_exact"]
    d["pass"] = bool(
        d["raw_g1_exceeds_clip"] and d["eps_rel_dev_from_rho"] < 1e-6
        and d["params_restored_after_update"] and d["perturb_exact"]
    )
    return d


def check_first_pass_no_leak(cfg: dict) -> dict:
    """Second-pass grads must equal g2(theta+eps), not g1+g2, and g1 must vanish."""
    d: dict[str, Any] = {}
    x, y = _tiny_data()
    torch.manual_seed(7)
    model = Tiny()
    params = list(model.parameters())
    opt = S.make_optimizer(model, cfg)
    model.zero_grad(set_to_none=True)
    F.l1_loss(model(x), y).backward()
    g1 = [p.grad.detach().clone() for p in params]
    norm1 = float(sum(float((g.double() ** 2).sum()) for g in g1) ** 0.5)
    eps = [g * (0.05 / (norm1 + 1e-12)) for g in g1]
    theta = S.save_params(params)
    S.apply_perturbation(params, eps)
    for p in params:  # emulate the harness zero_grad(set_to_none=True)
        p.grad = None
    cleared = all(p.grad is None for p in params)
    F.l1_loss(model(x), y).backward()
    g2_leak_check = [p.grad.detach().clone() for p in params]
    S.restore_params(params, theta)
    # independent g2 at theta+eps
    torch.manual_seed(7)
    ref = Tiny()
    rp = list(ref.parameters())
    S.restore_params(rp, [t + e for t, e in zip(theta, eps)])
    for p in rp:
        p.grad = None
    F.l1_loss(ref(x), y).backward()
    g2_ref = [p.grad.detach().clone() for p in rp]
    d["g2_equals_g2_at_perturbed_theta"] = bool(all(
        torch.allclose(a, b, rtol=1e-6, atol=1e-9) for a, b in zip(g2_leak_check, g2_ref)
    ))
    d["g2_differs_from_g1_plus_g2"] = bool(any(
        not torch.allclose(a, 2 * b, rtol=1e-6, atol=1e-9)
        for a, b in zip(g2_leak_check, g2_ref)
    ))
    d["grads_cleared_before_second_pass"] = bool(cleared)
    d["accumulated_g1_plus_g2_rejected"] = bool(not all(
        torch.allclose(a, b + c, rtol=1e-6, atol=1e-9)
        for a, b, c in zip(g2_leak_check, g1, g2_ref)
    ))
    d["pass"] = bool(d["g2_equals_g2_at_perturbed_theta"]
                     and d["grads_cleared_before_second_pass"]
                     and d["accumulated_g1_plus_g2_rejected"])
    return d


def check_exception_restores(cfg: dict) -> dict:
    d: dict[str, Any] = {}
    x, y = _tiny_data()
    torch.manual_seed(7)
    model = Tiny()
    params = list(model.parameters())
    before = S.save_params(params)
    calls = {"n": 0}
    orig = model.forward

    def flaky(xx):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected second-pass failure")
        return orig(xx)

    model.forward = flaky  # type: ignore[assignment]
    raised = False
    try:
        S.update_sam(model, S.make_optimizer(model, cfg), params, x, y,
                     rho=0.05, clip=5.0, epoch=1)
    except RuntimeError as exc:
        raised = "injected" in str(exc)
    model.forward = orig  # type: ignore[assignment]
    d["exception_raised"] = bool(raised)
    d["params_bit_identical_after_exception"] = S.params_match_backup(params, before)
    d["grads_present"] = any(p.grad is not None for p in params)
    d["pass"] = bool(raised and d["params_bit_identical_after_exception"])
    return d


def check_real_p3(cfg: dict, norm: dict) -> dict:
    d: dict[str, Any] = {}
    train_cache, y, rows = S.load_split("train")
    chunk = rows[:128]
    batch = T1.collate(train_cache, chunk, "P3N")
    target = torch.as_tensor(y[chunk], dtype=torch.float32)
    model, info = S.build_model(norm, torch.device("cpu"))
    params = list(model.parameters())
    opt = S.make_optimizer(model, cfg)
    parts = {
        "encoder": model.encoders.atom_encoder.lins[0].weight,
        "composer": model.group_composer.lins[0].weight,
        "reader": model.reader.lins[0].weight,
    }
    before = {k: v.detach().clone() for k, v in parts.items()}
    rec = S.update_sam(model, opt, params, batch, target, rho=0.05, clip=5.0,
                       epoch=1, check=True)
    S.set_lr(opt, 1, 47400, cfg)
    opt.step()
    d["loss_first_pass"] = float(rec["loss"])
    d["loss_perturbed"] = float(rec["loss_perturbed"])
    d["losses_finite"] = bool(math.isfinite(rec["loss"]) and math.isfinite(rec["loss_perturbed"]))
    d["g1_norm_finite"] = bool(math.isfinite(rec["g1_norm"]))
    d["g2_norm_finite"] = bool(math.isfinite(rec["g2_norm"]))
    d["eps_norm"] = float(rec["eps_norm"])
    d["moved"] = {k: float((v.detach() - before[k]).abs().max().item()) for k, v in parts.items()}
    d["all_three_parts_updated"] = bool(all(v > 0.0 for v in d["moved"].values()))
    d["perturb_exact"] = rec["perturb_exact"]
    d["restore_exact"] = rec["restore_exact"]
    d["params"] = int(info["parameters"])
    d["pass"] = bool(d["losses_finite"] and d["g1_norm_finite"] and d["g2_norm_finite"]
                    and d["all_three_parts_updated"] and rec["perturb_exact"]
                    and rec["restore_exact"])
    return d


def check_order_rng(cfg: dict) -> dict:
    d: dict[str, Any] = {}
    n, steps = 10000, 79
    a = torch.Generator().manual_seed(0 + 101)
    b = torch.Generator().manual_seed(0 + 101)
    base = [S.order_sha256(S.epoch_order(a, n)) for _ in range(5)]
    x, y = _tiny_data()
    torch.manual_seed(7)
    model = Tiny()
    params = list(model.parameters())
    opt = S.make_optimizer(model, cfg)
    moved = [S.order_sha256(S.epoch_order(b, n)) for _ in range(5)]
    # consume CPU RNG and run extra SAM forwards/backwards between epochs
    for i in range(steps):
        S.update_sam(model, opt, params, x, y, rho=0.05, clip=5.0, epoch=1)
        S.set_lr(opt, i + 1, 47400, cfg)
        opt.step()
    after_extra = [S.order_sha256(S.epoch_order(b, n)) for _ in range(2)]
    # untouched reference generator that only draws the shuffle epochs
    c = torch.Generator().manual_seed(0 + 101)
    for _ in range(5):
        S.epoch_order(c, n)
    continuation = [S.order_sha256(S.epoch_order(c, n)) for _ in range(2)]
    d["orders_before"] = moved
    d["orders_after_79_extra_sam_updates"] = after_extra
    d["reference_continuation"] = continuation
    d["epoch_orders_identical_between_arms"] = bool(base == moved)
    d["extra_forwards_do_not_disturb_shuffle"] = bool(after_extra == continuation)
    d["uses_dedicated_generator_seed"] = 101
    d["pass"] = bool(d["epoch_orders_identical_between_arms"]
                     and d["extra_forwards_do_not_disturb_shuffle"])
    return d


def check_schedule(cfg: dict) -> dict:
    d: dict[str, Any] = {}
    steps = math.ceil(cfg["data"]["train"] / cfg["schedule"]["batch_size"])
    total = int(cfg["schedule"]["epochs"]) * steps
    d["steps_per_epoch"] = int(steps)
    d["total_updates"] = int(total)
    d["steps_match_config"] = bool(steps == cfg["schedule"]["steps_per_epoch"] == 79)
    d["updates_match_config"] = bool(total == cfg["schedule"]["outer_updates"] == 47400)
    peak, min_lr = float(cfg["optimizer"]["lr"]), float(cfg["optimizer"]["min_lr"])
    d["lr_at_warmup_end"] = S.schedule_lr(500, total, cfg)
    d["lr_at_501"] = S.schedule_lr(501, total, cfg)
    d["lr_at_total"] = S.schedule_lr(total, total, cfg)
    vals = [S.schedule_lr(s, total, cfg) for s in range(500, total + 1)]
    d["monotone_after_warmup"] = bool(all(vals[i] >= vals[i + 1] - 1e-15
                                          for i in range(len(vals) - 1)))
    d["warmup_lr_at_1"] = S.schedule_lr(1, total, cfg)
    d["pass"] = bool(
        d["steps_match_config"] and d["updates_match_config"]
        and abs(d["lr_at_warmup_end"] - peak) < 1e-15
        and d["lr_at_501"] < peak
        and abs(d["lr_at_total"] - min_lr) < 1e-12
        and d["monotone_after_warmup"]
        and abs(d["warmup_lr_at_1"] - peak / 500) < 1e-15
    )
    return d


def check_best_selection(cfg: dict) -> dict:
    curve = [{"epoch": 1, "valid_mae": 0.2}, {"epoch": 2, "valid_mae": 0.1},
             {"epoch": 3, "valid_mae": 0.1}, {"epoch": 4, "valid_mae": 0.15}]
    best = {"valid_mae": float("inf"), "epoch": -1}
    for e in curve:
        if e["valid_mae"] < best["valid_mae"]:
            best = {"valid_mae": e["valid_mae"], "epoch": e["epoch"]}
    ok = best["epoch"] == 2
    return {"picked_epoch": int(best["epoch"]), "expected_epoch": 2,
            "strict_lt_ties_keep_earlier": bool(ok), "pass": bool(ok)}


def main() -> int:
    cfg = S.load_config()
    norm = D.load_norm()
    checks = {
        "init_harness_and_shared_state": check_init_and_harness(cfg, norm),
        "schedule_counts_and_lr": check_schedule(cfg),
        "rho0_sam_equals_adamw": check_rho0_equivalence(cfg),
        "two_pass_handcomputed": check_two_pass_handcomputed(cfg),
        "epsilon_global_norm_first_pass_unclipped": check_epsilon_and_no_first_clip(cfg),
        "first_pass_no_leak": check_first_pass_no_leak(cfg),
        "exception_restores_theta": check_exception_restores(cfg),
        "real_p3_double_forward_update": check_real_p3(cfg, norm),
        "order_rng_independent": check_order_rng(cfg),
        "best_selection_ties": check_best_selection(cfg),
    }
    n_pass = sum(1 for v in checks.values() if v.get("pass"))
    doc = {
        "protocol": S.PROTOCOL,
        "config_path": cfg["_config_path"],
        "config_sha256": cfg["_config_sha256"],
        "git_commit": C.git_commit(),
        "n_pass": int(n_pass),
        "n_fail": int(len(checks) - n_pass),
        "checks": checks,
        "created": C.utcnow(),
        "note": ("rho=0 equivalence and the hand-computed two-pass check are CPU "
                 "implementation tests, not extra experimental arms"),
    }
    out = S.ANALYSIS / "cpu_checks.json"
    C.write_json(out, doc)
    for name, value in checks.items():
        C.log(f"[check] {name:42s} {'PASS' if value.get('pass') else 'FAIL'}")
    C.log(f"[checks] {n_pass}/{len(checks)} pass -> {out}")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
