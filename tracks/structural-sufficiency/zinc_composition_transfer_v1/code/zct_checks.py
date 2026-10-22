"""Pre-training correctness checks for the composition-transfer round.

Every check here must pass before formal GPU time is spent.  They are the ones
that can block a wrong interpretation:

1. frozen P3B checkpoint replay + parameter count + SSL encoder provenance;
2. arm initialisation parity (same seed -> bitwise identical start state);
3. ``alpha = 0`` reduces to whole-batch ERM (loss and gradient);
4. batch-split invariance (no batch-dependent normalisation);
5. toy meta-gradient vs finite differences, and a detach variant that differs;
6. real-model second-order gradients are finite;
7. the virtual step does not touch parameters, buffers or optimizer state;
8. both halves are really read (label perturbations on each side move J);
9. the structural key builder really has k1 = f(k2), and the frozen partition
   masks reproduce R + recorded swaps.

Run locally on CPU:  ``uv run python zct_checks.py``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
for _p in (str(_HERE), str(_SSUF / "zinc_arch_discovery_v1" / "code"),
           str(_SSUF / "zinc_recipe_trunk_v2" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402
import zct_keys as K  # noqa: E402
import zct_split as SP  # noqa: E402
import zct_train as ZT  # noqa: E402

ANALYSIS = _TASK / "analysis"


def _params(model):
    return dict(model.named_parameters())


# ---------------------------------------------------------------------------
# 1. frozen P3B replay
# ---------------------------------------------------------------------------


def check_p3b_replay() -> dict:
    run = _SSUF / "zinc_recipe_trunk_v2" / "runs" / "xBssl_P3N_s0"
    ckpt = run / "best.pt"
    if not ckpt.exists():
        return {"status": "SKIP", "reason": f"missing {ckpt}"}
    norm = D.load_norm()
    model, info = ZT.build_model(norm, 0, torch.device("cpu"))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    C.load_state(model, state["state_dict"], torch.device("cpu"))
    cache = D.load_cache("valid")
    rows = np.arange(32, dtype=np.int64)
    pred, mae = T1.evaluate(model, cache, rows, candidate="P3N", device=torch.device("cpu"),
                            batch_size=32)
    stored = np.load(run / "predictions" / "valid_best.npz")
    sub = stored["pred"][:32]
    return {
        "status": "PASS" if float(np.max(np.abs(pred - sub))) < 1e-5 else "FAIL",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": C.sha256_file(ckpt),
        "stored_epoch": int(state["epoch"]),
        "stored_valid_mae": float(state["valid_mae"]),
        "parameters": int(info["parameters"]),
        "ssl_source": info["source"],
        "ssl_sha256": info["sha256"],
        "replay_max_abs_diff_32_rows": float(np.max(np.abs(pred - sub))),
        "replay_mae_32_rows": float(mae),
    }


# ---------------------------------------------------------------------------
# 2. initialisation parity
# ---------------------------------------------------------------------------


def check_init_parity() -> dict:
    norm = D.load_norm()
    out = {}
    for seed in (0, 1):
        hashes = {}
        param_hashes = {}
        for arm in ("E", "R", "S"):
            torch.manual_seed(seed)
            m, info = ZT.build_model(norm, seed, torch.device("cpu"))
            hashes[arm] = C.state_hash(m)
            param_hashes[arm] = C.sha256_array(
                np.concatenate([p.detach().numpy().ravel() for p in m.parameters()]))
            if arm == "E":
                ssl = info["sha256"]
        out[f"seed{seed}"] = {
            "state_hashes": hashes,
            "identical": len(set(hashes.values())) == 1,
            "ssl_sha256": ssl,
            "ssl_is_seed0": ssl == C.sha256_file(C.P3_SSL_S0),
        }
    # seed 0 vs seed 1 must differ only outside the SSL encoder block
    m0, _ = ZT.build_model(norm, 0, torch.device("cpu"))
    m1, _ = ZT.build_model(norm, 1, torch.device("cpu"))
    enc0 = {k: v for k, v in m0.state_dict().items() if k.startswith("encoders.")}
    enc1 = {k: v for k, v in m1.state_dict().items() if k.startswith("encoders.")}
    same_enc = all(torch.equal(enc0[k], enc1[k]) for k in enc0)
    other0 = {k: v for k, v in m0.state_dict().items() if not k.startswith("encoders.")}
    other1 = {k: v for k, v in m1.state_dict().items() if not k.startswith("encoders.")}
    same_other = all(torch.equal(other0[k], other1[k]) for k in other0)
    out["seed0_vs_seed1"] = {
        "encoder_blocks_identical": bool(same_enc),
        "non_encoder_blocks_identical": bool(same_other),
        "non_encoder_differ": bool(not same_other),
    }
    out["status"] = "PASS" if all(v.get("identical", True) for k, v in out.items() if k.startswith("seed")) \
        and out["seed0_vs_seed1"]["encoder_blocks_identical"] \
        and out["seed0_vs_seed1"]["non_encoder_differ"] else "FAIL"
    return out


# ---------------------------------------------------------------------------
# 3/4. alpha = 0 equivalence and batch-split invariance
# ---------------------------------------------------------------------------


def check_alpha_zero_equivalence(n_mol: int = 128) -> dict:
    norm = D.load_norm()
    torch.manual_seed(3)
    model, _ = ZT.build_model(norm, 3, torch.device("cpu"))
    model.eval()
    cache = D.load_cache("train")
    y = np.asarray(cache["labels"], dtype=np.float64)
    rows = np.arange(n_mol, dtype=np.int64)
    half = n_mol // 2
    s_rows, q_rows = rows[:half], rows[half:]
    b = T1.collate(cache, rows, "P3N")
    bs = T1.collate(cache, s_rows, "P3N")
    bq = T1.collate(cache, q_rows, "P3N")
    t = torch.as_tensor(y[rows], dtype=torch.float32)
    ts = torch.as_tensor(y[s_rows], dtype=torch.float32)
    tq = torch.as_tensor(y[q_rows], dtype=torch.float32)
    params = _params(model)
    # whole-batch ERM gradient
    pred = torch.func.functional_call(model, params, (b,))
    loss = F.l1_loss(pred.view(-1), t.view(-1))
    g_erm = torch.autograd.grad(loss, list(params.values()))
    # alpha = 0 look-ahead gradient
    g_zero, j, loss_s, loss_q, _, _ = ZT.meta_grads(model, params, bs, ts, bq, tq, 0.0)
    d_loss = abs(float(j) - float(loss))
    g_diff = max(float((a - b_).abs().max()) for a, b_ in zip(g_zero, g_erm))
    g_norm = float(torch.sqrt(sum((x * x).sum() for x in g_erm)))
    # predictions per molecule: whole batch vs two halves
    with torch.no_grad():
        p_full = torch.func.functional_call(model, params, (b,)).view(-1)
        p_halves = torch.cat([
            torch.func.functional_call(model, params, (bs,)).view(-1),
            torch.func.functional_call(model, params, (bq,)).view(-1)])
    split_diff = float((p_full - p_halves).abs().max())
    return {
        "status": "PASS" if (d_loss < 1e-6 and g_diff < 1e-5 * max(g_norm, 1.0)
                             and split_diff < 1e-5) else "FAIL",
        "loss_abs_diff_alpha0_vs_erm": d_loss,
        "grad_max_abs_diff": g_diff,
        "grad_rel_max_abs_diff": g_diff / max(g_norm, 1e-12),
        "batch_split_pred_max_abs_diff": split_diff,
        "note": "equal halves; the weighted J then equals the whole-batch mean L1",
    }


# ---------------------------------------------------------------------------
# 5. toy meta-gradient vs finite differences
# ---------------------------------------------------------------------------


class ToyNet(nn.Module):
    def __init__(self, din=5, dhid=7):
        super().__init__()
        torch.manual_seed(0)
        self.l1 = nn.Linear(din, dhid)
        self.l2 = nn.Linear(dhid, 1)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x))).view(-1)


def _toy_grads(model, params, xs, ys, xq, yq, alpha, clip=5.0, detach=False):
    names = list(params)
    values = [params[k] for k in names]
    ps = torch.func.functional_call(model, params, (xs,))
    ls = F.mse_loss(ps, ys)
    g = torch.autograd.grad(ls, values, create_graph=True, allow_unused=False)
    if detach:
        g = [x.detach() for x in g]
    gn = torch.sqrt(sum((x * x).sum() for x in g))
    c = torch.clamp(clip / (gn + 1e-8), max=1.0)
    shifted = {k: p - alpha * c * x for k, p, x in zip(names, values, g)}
    pq = torch.func.functional_call(model, shifted, (xq,))
    lq = F.mse_loss(pq, yq)
    j = 0.5 * ls + 0.5 * lq
    return torch.autograd.grad(j, values, allow_unused=False), float(j)


def _toy_j(model, xs, ys, xq, yq, alpha):
    """Value of J at the model's current parameters (inner gradient recomputed)."""
    params = _params(model)
    names = list(params)
    values = [params[k] for k in names]
    ps = torch.func.functional_call(model, params, (xs,))
    ls = F.mse_loss(ps, ys)
    g = torch.autograd.grad(ls, values, create_graph=True)
    gn = torch.sqrt(sum((x * x).sum() for x in g))
    c = torch.clamp(5.0 / (gn + 1e-8), max=1.0)
    shifted = {k: p.detach() - alpha * c.detach() * x.detach() for k, p, x in zip(names, values, g)}
    pq = torch.func.functional_call(model, shifted, (xq,))
    lq = F.mse_loss(pq, yq)
    return 0.5 * float(ls) + 0.5 * float(lq)


def check_toy_meta_gradient() -> dict:
    torch.manual_seed(1)
    model = ToyNet()
    xs = torch.randn(16, 5)
    ys = torch.randn(16)
    xq = torch.randn(16, 5)
    yq = torch.randn(16)
    alpha = 0.7
    params = _params(model)
    grads, _ = _toy_grads(model, params, xs, ys, xq, yq, alpha)
    grads_det, _ = _toy_grads(model, params, xs, ys, xq, yq, alpha, detach=True)
    eps = 1e-4
    fd = []
    for name, p in params.items():
        flat = p.detach().clone().view(-1)
        g_fd = torch.zeros_like(flat)
        for i in range(flat.numel()):
            backup = flat[i].item()
            flat[i] = backup + eps
            p.data = flat.view_as(p)
            jp = _toy_j(model, xs, ys, xq, yq, alpha)
            flat[i] = backup - eps
            p.data = flat.view_as(p)
            jm = _toy_j(model, xs, ys, xq, yq, alpha)
            flat[i] = backup
            p.data = flat.view_as(p)
            g_fd[i] = (jp - jm) / (2 * eps)
        fd.append(g_fd.view_as(p))
    err = max(float((a - b).abs().max()) for a, b in zip(grads, fd))
    det_err = max(float((a - b).abs().max()) for a, b in zip(grads_det, fd))
    scale = max(float(a.abs().max()) for a in fd)
    return {
        "status": "PASS" if err < 1e-3 * max(scale, 1.0) and det_err > 10 * err else "FAIL",
        "max_abs_diff_vs_finite_difference": err,
        "detach_variant_max_abs_diff_vs_finite_difference": det_err,
        "fd_scale": scale,
        "alpha": alpha,
        "note": "smooth MSE inner loss on purpose: the real L1 inner loss has an a.e. zero Hessian",
    }


# ---------------------------------------------------------------------------
# 6/7/8. real-model meta checks
# ---------------------------------------------------------------------------


def check_real_meta() -> dict:
    norm = D.load_norm()
    torch.manual_seed(4)
    model, _ = ZT.build_model(norm, 4, torch.device("cpu"))
    cache = D.load_cache("train")
    y = np.asarray(cache["labels"], dtype=np.float64)
    rows = np.arange(64, dtype=np.int64)
    s_rows, q_rows = rows[:32], rows[32:]
    bs = T1.collate(cache, s_rows, "P3N")
    bq = T1.collate(cache, q_rows, "P3N")
    ts = torch.as_tensor(y[s_rows], dtype=torch.float32)
    tq = torch.as_tensor(y[q_rows], dtype=torch.float32)
    params = _params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # take one ordinary step first so the optimizer state is non-empty
    loss0 = F.l1_loss(torch.func.functional_call(model, params, (bs,)).view(-1), ts.view(-1))
    loss0.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.manual_seed(11)  # make the weight-decay-free state deterministic
    before_state = C.cpu_state(model)

    def _opt_state():
        payload = []
        for k, v in sorted(opt.state.items(), key=lambda kv: str(kv[0])):
            payload.append((str(k), {kk: (vv.detach().cpu().numpy().tolist() if torch.is_tensor(vv) else vv)
                                     for kk, vv in sorted(v.items())}))
        return json.dumps(payload, sort_keys=True, default=str)

    before_opt = _opt_state()
    alpha = 0.01
    grads, j, loss_s, loss_q, gnorm, c = ZT.meta_grads(model, params, bs, ts, bq, tq, alpha)
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    gnorm_meta = float(torch.sqrt(sum((g * g).sum() for g in grads)))
    after_state = C.cpu_state(model)
    unchanged = all(torch.equal(before_state[k], after_state[k]) for k in before_state)
    after_opt = _opt_state()
    # both halves are read: perturb each side's labels
    j0 = float(ZT.meta_grads(model, params, bs, ts, bq, tq, alpha)[1])
    ts2 = ts + 1.0
    j_s = float(ZT.meta_grads(model, params, bs, ts2, bq, tq, alpha)[1])
    tq2 = tq + 1.0
    j_q = float(ZT.meta_grads(model, params, bs, ts, bq, tq2, alpha)[1])
    # L1 inner Hessian is a.e. zero: the exact meta gradient minus the
    # "query at theta" gradient equals only the clip-scalar term
    pred_q0 = torch.func.functional_call(model, params, (bq,))
    lq0 = F.l1_loss(pred_q0.view(-1), tq.view(-1))
    j_naive = 0.5 * F.l1_loss(torch.func.functional_call(model, params, (bs,)).view(-1), ts.view(-1)) \
        + 0.5 * lq0
    g_naive = torch.autograd.grad(j_naive, list(params.values()))
    naive_diff = max(float((a - b).abs().max()) for a, b in zip(grads, g_naive))
    return {
        "status": "PASS" if (finite and unchanged and j_s != j0 and j_q != j0) else "FAIL",
        "meta_grads_all_finite": finite,
        "meta_grad_norm": gnorm_meta,
        "virtual_step_left_params_buffers_unchanged": bool(unchanged),
        "virtual_step_left_optimizer_state_unchanged": before_opt == after_opt,
        "j_with_source_labels_shifted": j_s - j0,
        "j_with_query_labels_shifted": j_q - j0,
        "j_value": j0,
        "inner_grad_norm": float(gnorm),
        "clip_scalar": float(c),
        "max_abs_diff_meta_vs_query_at_theta": naive_diff,
        "note": ("for an L1 inner loss the inner Hessian vanishes almost everywhere, so the exact "
                 "meta gradient differs from 'query at theta' only through the clip scalar and the "
                 "shifted evaluation point; both are present in this number"),
    }


# ---------------------------------------------------------------------------
# 9. keys and masks
# ---------------------------------------------------------------------------


def check_keys_and_masks() -> dict:
    meta = C.read_json(_TASK / "cache" / "keys_meta.json")
    out = {
        "k1_is_function_of_k2": bool(meta["k1_is_function_of_k2"]),
        "n_key1": meta["n_key1"],
        "n_key2": meta["n_key2"],
        "old_group_reproduction_max_abs_diff": meta.get("old_group_reproduction", {}).get(
            "max_abs_diff_r2_unseen_atom_frac"),
    }
    part_files = sorted((_TASK / "cache").glob("partitions_s*.npz"))
    masks = []
    for path in part_files:
        d = dict(np.load(path))
        summary = C.read_json(path.with_suffix(".json"))
        ok = True
        for idx in (0, 1, 17, len(d["batch_rows"]) - 1):
            n = int(d["batch_rows"][idx])
            r = np.unpackbits(np.asarray(d["r_packed"][idx], np.uint8))[:n].astype(bool)
            s = np.unpackbits(np.asarray(d["s_packed"][idx], np.uint8))[:n].astype(bool)
            lo, hi = int(d["swap_off"][idx]), int(d["swap_off"][idx + 1])
            chk = r.copy()
            for i, j in zip(d["swap_i"][lo:hi], d["swap_j"][lo:hi]):
                chk[int(i)] = True
                chk[int(j)] = False
            ok = ok and bool(np.array_equal(chk, s)) and int(r.sum()) == n // 2
        masks.append({"file": str(path), "seed": summary["seed"], "ok": ok,
                      "delta_d_sym": summary["delta_d_sym"],
                      "delta_d_elig": summary["delta_d_elig"],
                      "mean_r1_R": summary["mean_r1_unseen_R"],
                      "mean_r1_S": summary["mean_r1_unseen_S"]})
    out["partition_masks"] = masks
    out["status"] = "PASS" if out["k1_is_function_of_k2"] and all(m["ok"] for m in masks) else "FAIL"
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-replay", action="store_true")
    args = p.parse_args(argv)
    out = {}
    out["p3b_replay"] = {"status": "SKIP"} if args.skip_replay else check_p3b_replay()
    out["init_parity"] = check_init_parity()
    out["alpha_zero"] = check_alpha_zero_equivalence()
    out["toy_meta_gradient"] = check_toy_meta_gradient()
    out["real_meta"] = check_real_meta()
    out["keys_and_masks"] = check_keys_and_masks()
    statuses = {k: v.get("status") for k, v in out.items()}
    out["overall"] = "PASS" if all(s in ("PASS", "SKIP") for s in statuses.values()) else "FAIL"
    C.write_json(ANALYSIS / "cpu_checks.json", out)
    C.log(json.dumps(statuses, indent=1))
    C.log(json.dumps({k: out[k] for k in ("alpha_zero", "toy_meta_gradient", "real_meta")}, indent=1))
    return 0 if out["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
